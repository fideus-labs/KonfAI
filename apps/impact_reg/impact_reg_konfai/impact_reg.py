# Copyright (c) 2025 Valentin Boussot
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Orchestrator for IMPACT-Reg.

Each IMPACT-Reg *preset* is a self-contained KonfAI app on ``VBoussot/ImpactReg`` (one preset = one app):
its model produces, on the FIXED grid, the moving image resampled onto the fixed image (``MovedImage``)
and the displacement field (``DisplacementField``). This orchestrator adds the registration-specific
logic that does not fit the generic ``konfai-apps`` pipeline, split into three composable operations
(mirroring ``konfai-apps`` infer/eval/uncertainty) so a UI/CLI can run them independently:

- ``register``    : run one or more preset apps on a fixed/moving pair, ensemble their displacement
                    fields (average), and write the moved image, the (averaged) displacement field, the
                    transform, and the per-preset displacement fields (kept for uncertainty);
- ``evaluate``    : given a transform, apply it to the moving image / segmentation / landmarks and run
                    the bundle's evaluation configs (image MAE, segmentation Dice, landmark TRE);
- ``uncertainty`` : from the per-preset displacement fields, compute the voxel-wise spread map.
"""

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from konfai.utils.dataset import read_landmarks, write_landmarks
from konfai.utils.ITK import apply_to_data_transform
from konfai_apps import KonfAIApp
from konfai_apps.app_repository import get_available_apps_on_hf_repo

# Preset apps live on this Hugging Face repo; override with KONFAI_IMPACTREG_REPO to point at a local
# directory of preset folders (each with an app.json) for development / offline use.
IMPACT_REG_KONFAI_REPO = os.environ.get("KONFAI_IMPACTREG_REPO", "VBoussot/ImpactReg")

_ENSEMBLE_DIR = "Ensemble"


def _app_id(preset: str) -> str:
    """Resolve a preset to a KonfAIApp id: a local ``<dir>/<preset>`` path, or ``<repo>:<preset>`` on HF."""
    if Path(IMPACT_REG_KONFAI_REPO).is_dir():
        return str(Path(IMPACT_REG_KONFAI_REPO) / preset)
    return f"{IMPACT_REG_KONFAI_REPO}:{preset}"


def get_available_presets(force_update: bool = False) -> list[str]:
    """List the registration preset apps (local directory or Hugging Face repo).

    A local directory is filtered to app folders whose ``app.json`` declares ``task == "registration"``,
    so non-preset folders (e.g. a legacy evaluation-only app) never surface as a preset.
    """
    if Path(IMPACT_REG_KONFAI_REPO).is_dir():
        presets = []
        for folder in sorted(Path(IMPACT_REG_KONFAI_REPO).iterdir()):
            app_json = folder / "app.json"
            if not app_json.is_file():
                continue
            try:
                if json.loads(app_json.read_text(encoding="utf-8")).get("task") == "registration":
                    presets.append(folder.name)
            except (OSError, json.JSONDecodeError):
                continue
        return presets
    return list(get_available_apps_on_hf_repo(IMPACT_REG_KONFAI_REPO, force_update))


def _find_output(root: Path, name: str) -> Path:
    """Locate a single ``<name>`` produced by a preset inference under ``root``."""
    matches = sorted(root.rglob(name))
    if not matches:
        raise FileNotFoundError(f"Preset inference did not produce '{name}' under {root}.")
    return matches[0]


def _displacement_transform(dvf_path: Path) -> sitk.Transform:
    """Read a displacement-field image (3-component, fixed grid) as a SimpleITK transform."""
    return sitk.DisplacementFieldTransform(sitk.ReadImage(str(dvf_path), sitk.sitkVectorFloat64))


def _neutral_mask(out_path: Path) -> Path:
    """Write a tiny all-ones sentinel — a no-op mask used only to fill the positional gap when the caller
    gives a moving mask but no fixed mask (inputs map positionally, so the fixed-mask slot must be present).

    A whole-image all-ones mask restricts nothing (the model's ``_is_partial_mask`` treats it as absent),
    and in whole-volume mode a mask branch need not share the image grid, so a 2x2x2 sentinel yields a
    byte-identical registration (verified) without reading — or even sizing to — the input. (The common
    no-mask path passes no mask at all; konfai-apps fills both branches with an all-ones default.)
    """
    sitk.WriteImage(sitk.GetImageFromArray(np.ones((2, 2, 2), dtype=np.uint8)), str(out_path))
    return out_path


class ImpactRegKonfAIApp:
    """Run IMPACT-Reg preset apps, ensemble their displacement fields, evaluate, and estimate uncertainty."""

    def __init__(self, download: bool = False, force_update: bool = False) -> None:
        self._download = download
        self._force_update = force_update

    # ------------------------------------------------------------------ register

    def _infer_preset(
        self,
        preset: str,
        fixed_image: Path,
        moving_image: Path,
        fixed_mask: Path | None,
        moving_mask: Path | None,
        work: Path,
        gpu: list[int],
        cpu: int | None,
        quiet: bool,
        tta: int = 0,
        config_overrides: list[str] | None = None,
    ) -> tuple[Path, Path]:
        """Run one preset app on the fixed/moving pair (+ optional masks); return its (moved, displacement) paths.

        Each preset runs through the ``konfai-apps`` CLI in its own subprocess: konfai keeps
        process-global state (its ``Config`` singleton, the ``KONFAI_*`` environment), so several preset
        inferences in one process would clash. The ``-i`` inputs map positionally to the app's input groups.
        Masks are optional: konfai-apps fills any we omit with an all-ones default, so with no mask we pass
        only fixed+moving. Because the mapping is positional, a lone moving mask still needs the fixed-mask
        slot present, so send the pair (defaulting the absent one to an all-ones sentinel) once either is given.
        """
        out = work / preset
        command = ["konfai-apps", "infer", _app_id(preset), "-i", str(fixed_image), "-i", str(moving_image)]
        if fixed_mask is not None or moving_mask is not None:
            command += ["-i", str(fixed_mask or _neutral_mask(work / "FixedMask.mha"))]
            command += ["-i", str(moving_mask or _neutral_mask(work / "MovingMask.mha"))]
        command += ["-o", str(out)]
        if tta:
            command += ["--tta", str(tta)]
        # Preset-parameter tuning: forwarded verbatim to `konfai-apps infer --set` (applies to every preset).
        for override in config_overrides or []:
            command += ["--set", override]
        if gpu:
            command += ["--gpu", *(str(g) for g in gpu)]
        elif cpu is not None:
            command += ["--cpu", str(cpu)]
        if quiet:
            command.append("--quiet")
        if self._download:
            command.append("--download")
        if self._force_update:
            command.append("--force_update")
        subprocess.run(command, check=True)  # nosec B603
        # The model emits both the moved image and the displacement field on the fixed grid; reusing them
        # (rather than re-resampling here) keeps the single-preset path free of any extra image read/write.
        return _find_output(out, "Moved.mha"), _find_output(out, "DVF.mha")

    def register(
        self,
        presets: list[str],
        fixed_images: list[Path],
        moving_images: list[Path],
        fixed_masks: list[Path] = [],
        moving_masks: list[Path] = [],
        output: Path = Path("./Output").resolve(),
        gpu: list[int] = [],
        cpu: int | None = None,
        quiet: bool = False,
        tta: int = 0,
        keep_dvf: bool = False,
        config_overrides: list[str] | None = None,
    ) -> None:
        """Register each fixed/moving pair with the selected presets and ensemble their DVFs.

        Masks are optional and restrict the metric region; when omitted a whole-image mask is auto-filled,
        so every preset app always receives the four inputs (fixed, moving, fixed mask, moving mask) it declares.
        """
        for index, (fixed_image, moving_image) in enumerate(zip(fixed_images, moving_images, strict=True)):
            case_out = output / f"P{index:03d}"
            case_out.mkdir(parents=True, exist_ok=True)
            # The per-preset displacement fields are large; only persist them (under Ensemble/) when the
            # caller asks, so `uncertainty` can measure the ensemble spread afterwards.
            if keep_dvf:
                (case_out / _ENSEMBLE_DIR).mkdir(parents=True, exist_ok=True)
            work = Path(tempfile.mkdtemp(prefix="impact_reg_"))
            try:
                # Masks are optional (they restrict the metric region); pass only those the caller gave and
                # let konfai-apps fill the rest with an all-ones default — no input read on the no-mask path.
                fixed_mask = fixed_masks[index] if index < len(fixed_masks) else None
                moving_mask = moving_masks[index] if index < len(moving_masks) else None

                moved_paths, dvf_paths = [], []
                for preset in presets:
                    moved, dvf = self._infer_preset(
                        preset,
                        fixed_image,
                        moving_image,
                        fixed_mask,
                        moving_mask,
                        work,
                        gpu,
                        cpu,
                        quiet,
                        tta,
                        config_overrides,
                    )
                    if keep_dvf:
                        member = case_out / _ENSEMBLE_DIR / f"{preset}.mha"
                        shutil.copy2(dvf, member)
                        dvf = member
                    moved_paths.append(moved)
                    dvf_paths.append(dvf)

                if len(presets) == 1:
                    # One preset: the model already produced the moved image AND the displacement field on
                    # the fixed grid — reuse them verbatim. No input re-read, no re-resample, and the input
                    # format is whatever the model handled (OME-Zarr included).
                    shutil.copy2(moved_paths[0], case_out / "Moved.mha")
                    shutil.copy2(dvf_paths[0], case_out / "DVF.mha")
                else:
                    # Ensemble: average the presets' displacement fields (all on the fixed grid) and warp the
                    # moving image once with that averaged field — the one output no single preset produced.
                    avg_dvf = self._average_displacement(dvf_paths)
                    sitk.WriteImage(avg_dvf, str(case_out / "DVF.mha"))
                    transform = sitk.DisplacementFieldTransform(sitk.Cast(avg_dvf, sitk.sitkVectorFloat64))
                    moving = sitk.ReadImage(str(moving_image))
                    sitk.WriteImage(
                        sitk.Resample(moving, avg_dvf, transform, sitk.sitkLinear, 0.0, moving.GetPixelID()),
                        str(case_out / "Moved.mha"),
                    )

                # Transform.h5 (consumed by `evaluate` and SlicerImpactReg): the fixed-grid displacement
                # field as a SimpleITK transform.
                sitk.WriteTransform(_displacement_transform(case_out / "DVF.mha"), str(case_out / "Transform.h5"))
            finally:
                shutil.rmtree(work, ignore_errors=True)

    def _average_displacement(self, dvf_paths: list[Path]) -> sitk.Image:
        """Average several presets' displacement fields (all on the same fixed grid) into one field."""
        reference = sitk.ReadImage(str(dvf_paths[0]))
        stack = np.stack([sitk.GetArrayFromImage(sitk.ReadImage(str(p))) for p in dvf_paths], axis=0)
        avg = sitk.GetImageFromArray(stack.mean(axis=0), isVector=True)
        avg.CopyInformation(reference)
        return avg

    # ------------------------------------------------------------------ evaluate

    def evaluate(
        self,
        preset: str,
        fixed_images: list[Path] = [],
        moving_images: list[Path] = [],
        transforms: list[Path] = [],
        gt_fixed_seg: list[Path] = [],
        gt_moving_seg: list[Path] = [],
        gt_fixed_fid: list[Path] = [],
        gt_moving_fid: list[Path] = [],
        mask: list[Path] | None = None,
        output: Path = Path("./Output").resolve(),
        gpu: list[int] = [],
        cpu: int | None = None,
        quiet: bool = False,
    ) -> None:
        """Evaluate a registration on any subset of modalities (image MAE, seg Dice, landmark TRE).

        Every input is optional: whichever modality has its pair present is evaluated. When a transform
        is given it warps the moving data onto the fixed grid first; otherwise the moving data is assumed
        already registered and only resampled onto the fixed grid (identity).
        """
        app = KonfAIApp(_app_id(preset), self._download, self._force_update)
        n_cases = max(len(fixed_images), len(gt_fixed_seg), len(gt_fixed_fid))
        for index in range(n_cases):
            transform_path = transforms[index] if index < len(transforms) else None
            transform = sitk.ReadTransform(str(transform_path)) if transform_path else sitk.Transform()
            eval_out = output / f"P{index:03d}" / "Evaluation"
            work = Path(tempfile.mkdtemp(prefix="impact_reg_eval_"))
            try:
                # Image: moving resampled onto the fixed grid vs fixed (MAE). Mask is optional.
                if index < len(fixed_images) and index < len(moving_images):
                    fixed = sitk.ReadImage(str(fixed_images[index]))
                    moved = work / "moved_image.nii.gz"
                    sitk.WriteImage(
                        sitk.Resample(sitk.ReadImage(str(moving_images[index])), fixed, transform), str(moved)
                    )
                    app.evaluate(
                        inputs=[[fixed_images[index]]],
                        gt=[[moved]],
                        output=eval_out,
                        mask=[[mask[index]]] if mask and index < len(mask) else None,
                        evaluation_file="Evaluation_with_images.yml",
                        gpu=gpu,
                        cpu=cpu,
                        quiet=quiet,
                    )

                # Segmentation: moving seg warped onto fixed vs fixed seg (Dice).
                if index < len(gt_fixed_seg) and index < len(gt_moving_seg):
                    fixed_seg = sitk.ReadImage(str(gt_fixed_seg[index]))
                    moved_seg = work / "moved_seg.nii.gz"
                    sitk.WriteImage(
                        sitk.Resample(
                            sitk.ReadImage(str(gt_moving_seg[index])), fixed_seg, transform, sitk.sitkNearestNeighbor
                        ),
                        str(moved_seg),
                    )
                    app.evaluate(
                        inputs=[[gt_fixed_seg[index]]],
                        gt=[[moved_seg]],
                        output=eval_out,
                        evaluation_file="Evaluation_with_seg.yml",
                        gpu=gpu,
                        cpu=cpu,
                        quiet=quiet,
                    )

                # Landmarks (TRE): the transform is defined on the fixed grid and maps fixed->moving, so the
                # fixed fiducials are displaced by it into moving space and compared against the moving fiducials
                # there (the standard warped-keypoints convention; no field inversion needed). With no transform
                # the raw fiducials are compared, measuring the initial misalignment.
                if index < len(gt_fixed_fid) and index < len(gt_moving_fid):
                    fixed_points = read_landmarks(gt_fixed_fid[index])
                    if transform_path is not None:
                        fixed_points = apply_to_data_transform(fixed_points, {transform: False})
                    moved_fid = work / "moved_fid.fcsv"
                    write_landmarks(fixed_points, moved_fid)
                    app.evaluate(
                        inputs=[[gt_moving_fid[index]]],
                        gt=[[moved_fid]],
                        output=eval_out,
                        evaluation_file="Evaluation_with_fid.yml",
                        gpu=gpu,
                        cpu=cpu,
                        quiet=quiet,
                    )
            finally:
                shutil.rmtree(work, ignore_errors=True)

    # --------------------------------------------------------------- uncertainty

    def uncertainty(
        self,
        preset: str,
        dvfs: list[Path],
        output: Path = Path("./Output").resolve(),
        gpu: list[int] = [],
        cpu: int | None = None,
        quiet: bool = False,
    ) -> None:
        """Estimate registration uncertainty as the voxel-wise spread of an ensemble of displacement fields.

        The per-preset displacement fields are stacked into one multi-component volume (samples as
        components, vector components as the leading image axis) and handed to the preset's generic
        ``Uncertainty.yml`` workflow (``konfai-apps uncertainty``: ``Norm`` magnitude then
        ``StandardDeviation`` over the ensemble).
        """
        if len(dvfs) < 2:
            raise ValueError("Uncertainty needs at least two ensemble displacement fields.")
        work = Path(tempfile.mkdtemp(prefix="impact_reg_unc_"))
        try:
            reference = sitk.ReadImage(str(dvfs[0]))
            rank = reference.GetDimension()
            stack = sitk.GetImageFromArray(
                np.stack([sitk.GetArrayFromImage(sitk.ReadImage(str(p))) for p in dvfs], axis=-1), isVector=True
            )
            # The extra leading image axis holds the vector components (dropped by ``Norm``); the real
            # fixed-grid geometry lives on the remaining axes so the uncertainty map stays aligned.
            stack.SetOrigin((0.0, *reference.GetOrigin()))
            stack.SetSpacing((1.0, *reference.GetSpacing()))
            direction = np.eye(rank + 1)
            direction[1:, 1:] = np.asarray(reference.GetDirection()).reshape(rank, rank)
            stack.SetDirection(direction.flatten())
            sitk.WriteImage(stack, str(work / "DVFs.mha"))

            command = ["konfai-apps", "uncertainty", _app_id(preset), "-i", str(work / "DVFs.mha"), "-o", str(output)]
            if gpu:
                command += ["--gpu", *(str(g) for g in gpu)]
            elif cpu is not None:
                command += ["--cpu", str(cpu)]
            if quiet:
                command.append("--quiet")
            if self._download:
                command.append("--download")
            if self._force_update:
                command.append("--force_update")
            subprocess.run(command, check=True)  # nosec B603
        finally:
            shutil.rmtree(work, ignore_errors=True)
