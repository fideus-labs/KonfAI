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
its model produces the displacement field (``DisplacementField``) on the FIXED grid, and optionally the
moving image resampled onto the fixed one (``MovedImage``). The field is what a registration preset must
declare; the moved image IS that field applied to the moving, so a preset that leaves it out is complete
and this orchestrator derives it. This layer adds the registration-specific
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
from typing import Any

import numpy as np
import SimpleITK as sitk
from konfai.utils.dataset import read_landmarks, write_landmarks
from konfai.utils.ITK import apply_to_data_transform, read_displacement_field
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


def _find_output(root: Path, stem: str, required: bool = True) -> Path | None:
    """Locate the single output named ``stem`` under ``root``, whatever form the preset wrote it in.

    ``required=False`` returns None instead of raising, for an output a preset may legitimately not
    declare: the displacement field is what a registration preset must produce, and the moved image is
    that field applied to the moving -- derivable here (see :func:`_derive_moved`) rather than a second
    thing every preset has to remember to write.

    Matched on the name rather than on a fixed filename: a displacement field may come out as an ITK
    image or as an OME-Zarr store, and a store is a DIRECTORY whose ``Path.stem`` is "DVF.ome" -- so a
    fixed "DVF.mha" finds nothing and the run dies with the output sitting in the directory it listed.
    """
    matches = sorted(root.rglob(f"{stem}.*"))
    if not matches:
        if not required:
            return None
        raise FileNotFoundError(f"Preset inference did not produce '{stem}' under {root}.")
    return matches[0]


def _output_path(dest_dir: Path, stem: str, suffixes: str) -> Path:
    """The path ``<stem><suffixes>`` in ``dest_dir``, with every earlier output of that stem removed.

    A re-run whose presets emit the other form would otherwise leave both ``DVF.mha`` and
    ``DVF.ome.zarr`` behind, and discovery is by stem: ``_find_output`` takes the first match, which
    sorts to the stale one. Only the current run's output is left standing.
    """
    for stale in [p for p in dest_dir.iterdir() if p.name.startswith(f"{stem}.")]:
        shutil.rmtree(stale) if stale.is_dir() else stale.unlink()
    return dest_dir / (stem + suffixes)


def _work_dir(tmp_dir: Path | None, prefix: str) -> Path:
    """A private scratch directory for one command's intermediates.

    Under ``tmp_dir`` when the caller named one, under ``tempfile.gettempdir()`` otherwise — which is
    the same contract every other KonfAI app CLI offers through ``--tmp-dir``.

    THE DEFAULT IS NOT ALWAYS A GOOD PLACE, WHICH IS WHY THE OPTION EXISTS. What is staged here is
    volume-sized: the moved image and the displacement field are written before being collected into
    ``--output``. Where TMPDIR is a tmpfs that traffic is charged to RAM, on top of the volumes the run
    already holds; on a large 3D case that is what fills memory or the temp quota mid-run. A caller who
    knows better — a pipeline node with its results on real disk — names a directory here instead of
    overriding ``TMPDIR`` from outside, which would also move things TMPDIR legitimately owns (torch's
    DataLoader opens its worker sockets there, and AF_UNIX addresses cap at ~108 bytes).

    The directory returned is always freshly created and owned by the caller of this function, never
    ``tmp_dir`` itself: the command removes what it made and leaves the directory it was given.
    """
    if tmp_dir is not None:
        tmp_dir.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=str(tmp_dir) if tmp_dir is not None else None))


def _copy_output(src: Path, dest_dir: Path, stem: str) -> Path:
    """Copy an output beside the results, keeping the form the preset produced (file or store)."""
    dest = _output_path(dest_dir, stem, "".join(src.suffixes))
    (shutil.copytree if src.is_dir() else shutil.copy2)(src, dest)
    if dest.is_dir():
        # A store put at a path already read is invisible to the reader's path-keyed memo, which
        # would otherwise pair the copy's voxels with the replaced store's axes and geometry.
        from konfai.utils.ome_zarr import clear_ome_zarr_cache

        clear_ome_zarr_cache()
    return dest


def _write_displacement_field(field: sitk.Image, dest: Path) -> None:
    """Write a field in the form ``dest`` names, so a derived field matches the ones it came from."""
    if "".join(dest.suffixes).endswith(".ome.zarr"):
        from konfai.utils.dataset import image_to_data
        from konfai.utils.ome_zarr import write_ome_zarr

        # image_to_data yields the channel-first array and the Origin/Spacing/Direction the store must
        # carry -- the same encoding the Dataset backend writes, so the field round-trips through either.
        data, attributes = image_to_data(field)
        write_ome_zarr(
            dest,
            data,
            spacing=field.GetSpacing(),
            origin=field.GetOrigin(),
            attributes=dict(attributes),
            displacement_field=True,
        )
    else:
        sitk.WriteImage(field, str(dest))


def _dataset_entry(path: Path) -> tuple[Any, str, str]:
    """Address one file or store as a konfai ``Dataset`` entry: ``(dataset, group, name)``.

    Dataset is the layer that already knows every format konfai supports -- h5, DICOM, OME-Zarr, and
    every ITK extension through SitkFile, which probes them itself. Going through it is what makes
    this orchestrator format-agnostic without owning a dispatch of its own: a hand-written
    ``if path.is_dir()`` knows exactly the two formats whoever wrote it thought of.

    A dataset addresses ``{root}/{name}/{group}.{ext}``. An orchestrator input is a bare path, so the
    case is empty: the parent directory is the root and the stem is the entry. The extension only
    seeds the format token -- Dataset normalises it (``.ome.zarr`` / ``.zarr`` -> ``omezarr``) and
    re-detects a directory store from disk regardless of what the token said.
    """
    from konfai.utils.dataset import Dataset

    suffixes = "".join(path.suffixes)
    stem = path.name[: len(path.name) - len(suffixes)] if suffixes else path.name
    file_format = "omezarr" if suffixes.lower().endswith((".ome.zarr", ".zarr")) else suffixes.lstrip(".")
    return Dataset(path.parent, file_format or "mha"), stem, ""


def _read_image(path: Path) -> sitk.Image:
    """An image, in whatever format it is stored — read through konfai's Dataset."""
    dataset, group, name = _dataset_entry(path)
    return dataset.read_image(group, name)


def _write_image(image: sitk.Image, dest: Path) -> None:
    """Write an image in the form ``dest`` names — through the same layer that read it.

    So the format out is the format in, for every format konfai writes, and not only for the two an
    ``if`` here would have enumerated.
    """
    dataset, group, name = _dataset_entry(dest)
    dataset.write(group, name, image)


def _derive_moved(moving_image: Path, dvf_path: Path, dest_dir: Path, field: sitk.Image | None = None) -> Path:
    """The moved image, resampled from the moving through the displacement field.

    A preset that emits only a field is complete: the moved image IS that field applied to the moving,
    so deriving it belongs to the orchestrator rather than being a second output every preset has to
    remember to declare.

    FORMAT IN, SAME FORMAT OUT. Both ends go through the dispatch above, so an OME-Zarr moving yields
    an OME-Zarr moved and an ITK one an ITK file; the resample never decides the format. Reading the
    moving with ``sitk.ReadImage`` instead -- what this replaces -- cannot open a store at all, which
    is why the ensemble path could not be used with OME-Zarr inputs.

    Resampled through SimpleITK for the reason konfai's own ``ResampleTransform`` gives: the stored
    displacement is in world (x, y, z) units, and adding it onto a (z, y, x) voxel grid by hand
    transposes the axes and reads millimetres as voxels. The output grid is the field's own -- a
    displacement field is defined ON the fixed grid, so that is where the moved image belongs.
    """
    if field is None:
        field = read_displacement_field(dvf_path)
    # Read the grid off the field BEFORE the transform takes it: DisplacementFieldTransform assumes
    # ownership of the image it is given and leaves it empty behind.
    size, origin = field.GetSize(), field.GetOrigin()
    spacing, direction = field.GetSpacing(), field.GetDirection()
    transform = sitk.DisplacementFieldTransform(field)
    moving = _read_image(moving_image)
    moved = sitk.Resample(
        moving, size, transform, sitk.sitkLinear, origin, spacing, direction, 0.0, moving.GetPixelID()
    )
    dest = _output_path(dest_dir, "Moved", "".join(dvf_path.suffixes))
    _write_image(moved, dest)
    return dest


def _displacement_transform(dvf_path: Path) -> sitk.Transform:
    """Read a displacement field (3-component, fixed grid) as a SimpleITK transform."""
    return sitk.DisplacementFieldTransform(read_displacement_field(dvf_path))


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
        # Hand konfai-apps a workspace we own, which is what every other app CLI does by exposing
        # --tmp-dir. Without it konfai-apps auto-creates one under TMPDIR, writes the prediction to
        # ./Predictions inside it, and copies that into `-o` before deleting it: one extra full-size
        # write of the moved image AND the displacement field, on whatever filesystem TMPDIR names.
        # Given a caller-owned workspace it writes straight into `-o` (see konfai_apps
        # _stage_result_dir / _collect_result), so the copy disappears and `out` is where it always was.
        command += ["--tmp-dir", str(out)]
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
        return _find_output(out, "Moved", required=False), _find_output(out, "DVF")

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
        tmp_dir: Path | None = None,
    ) -> None:
        """Register each fixed/moving pair with the selected presets and ensemble their DVFs.

        Masks are optional and restrict the metric region; when omitted a whole-image mask is auto-filled,
        so every preset app always receives the four inputs (fixed, moving, fixed mask, moving mask) it declares.

        ``tmp_dir`` names where the intermediates are staged; see :func:`_work_dir`.
        """
        for index, (fixed_image, moving_image) in enumerate(zip(fixed_images, moving_images, strict=True)):
            case_out = output / f"P{index:03d}"
            case_out.mkdir(parents=True, exist_ok=True)
            # The per-preset displacement fields are large; only persist them (under Ensemble/) when the
            # caller asks, so `uncertainty` can measure the ensemble spread afterwards.
            if keep_dvf:
                (case_out / _ENSEMBLE_DIR).mkdir(parents=True, exist_ok=True)
            work = _work_dir(tmp_dir, "impact_reg_")
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
                        dvf = _copy_output(dvf, case_out / _ENSEMBLE_DIR, preset)
                    moved_paths.append(moved)
                    dvf_paths.append(dvf)

                if len(presets) == 1:
                    dvf_out = _copy_output(dvf_paths[0], case_out, "DVF")
                    if moved_paths[0] is not None:
                        # The model already produced the moved image on the fixed grid — reuse it
                        # verbatim. No input re-read, no re-resample, and the input format is whatever
                        # the model handled (OME-Zarr included).
                        _copy_output(moved_paths[0], case_out, "Moved")
                    else:
                        # A preset that declares only a field is complete: the moved image is that
                        # field applied to the moving, and producing it belongs here.
                        _derive_moved(moving_image, dvf_out, case_out)
                else:
                    # Ensemble: average the presets' displacement fields (all on the fixed grid) and warp the
                    # moving image once with that averaged field — the one output no single preset produced.
                    avg_dvf = self._average_displacement(dvf_paths)
                    dvf_out = _output_path(case_out, "DVF", "".join(dvf_paths[0].suffixes))
                    _write_displacement_field(avg_dvf, dvf_out)
                    # Through the same derivation as the single-preset path, which reads the moving in
                    # either form: the sitk.ReadImage this replaces cannot open a store at all, so an
                    # ensemble of OME-Zarr inputs failed here and nowhere else.
                    _derive_moved(moving_image, dvf_out, case_out, field=sitk.Cast(avg_dvf, sitk.sitkVectorFloat64))

                # Transform.h5 (consumed by `evaluate` and SlicerImpactReg): the fixed-grid displacement
                # field as a SimpleITK transform.
                sitk.WriteTransform(_displacement_transform(dvf_out), str(case_out / "Transform.h5"))
            finally:
                shutil.rmtree(work, ignore_errors=True)

    def _average_displacement(self, dvf_paths: list[Path]) -> sitk.Image:
        """Average several presets' displacement fields (all on the same fixed grid) into one field.

        A running sum keeps memory flat in the number of members: a few field-sized buffers are live
        at any instant, whatever the size of the ensemble.
        """
        reference = read_displacement_field(dvf_paths[0])
        total = sitk.GetArrayFromImage(reference)
        for path in dvf_paths[1:]:
            total += sitk.GetArrayFromImage(read_displacement_field(path))
        avg = sitk.GetImageFromArray(total / len(dvf_paths), isVector=True)
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
        tmp_dir: Path | None = None,
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
            work = _work_dir(tmp_dir, "impact_reg_eval_")
            try:
                # Image: moving resampled onto the fixed grid vs fixed (MAE). Mask is optional.
                if index < len(fixed_images) and index < len(moving_images):
                    fixed = _read_image(fixed_images[index])
                    moved = work / "moved_image.nii.gz"
                    sitk.WriteImage(
                        sitk.Resample(_read_image(moving_images[index]), fixed, transform), str(moved)
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
                        tmp_dir=work,
                    )

                # Segmentation: moving seg warped onto fixed vs fixed seg (Dice).
                if index < len(gt_fixed_seg) and index < len(gt_moving_seg):
                    fixed_seg = _read_image(gt_fixed_seg[index])
                    moved_seg = work / "moved_seg.nii.gz"
                    sitk.WriteImage(
                        sitk.Resample(
                            _read_image(gt_moving_seg[index]), fixed_seg, transform, sitk.sitkNearestNeighbor
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
                        tmp_dir=work,
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
                        tmp_dir=work,
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
        tmp_dir: Path | None = None,
    ) -> None:
        """Estimate registration uncertainty as the voxel-wise spread of an ensemble of displacement fields.

        The per-preset displacement fields are stacked into one multi-component volume (samples as
        components, vector components as the leading image axis) and handed to the preset's generic
        ``Uncertainty.yml`` workflow (``konfai-apps uncertainty``: ``Norm`` magnitude then
        ``StandardDeviation`` over the ensemble).
        """
        if len(dvfs) < 2:
            raise ValueError("Uncertainty needs at least two ensemble displacement fields.")
        work = _work_dir(tmp_dir, "impact_reg_unc_")
        try:
            reference = read_displacement_field(dvfs[0])
            rank = reference.GetDimension()
            stack = sitk.GetImageFromArray(
                np.stack([sitk.GetArrayFromImage(read_displacement_field(p)) for p in dvfs], axis=-1),
                isVector=True,
            )
            # The extra leading image axis holds the vector components (dropped by ``Norm``); the real
            # fixed-grid geometry lives on the remaining axes so the uncertainty map stays aligned.
            stack.SetOrigin((0.0, *reference.GetOrigin()))
            stack.SetSpacing((1.0, *reference.GetSpacing()))
            direction = np.eye(rank + 1)
            direction[1:, 1:] = np.asarray(reference.GetDirection()).reshape(rank, rank)
            stack.SetDirection(direction.flatten())
            sitk.WriteImage(stack, str(work / "DVFs.mha"))

            # Same workspace hand-off as _infer_preset: without it konfai-apps auto-creates one under
            # TMPDIR and stages Uncertainties there before copying it into -o, which is the staging
            # this option exists to place. `work` is ours and already sits wherever tmp_dir asked for.
            command = ["konfai-apps", "uncertainty", _app_id(preset), "-i", str(work / "DVFs.mha"),
                       "-o", str(output), "--tmp-dir", str(work)]
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
