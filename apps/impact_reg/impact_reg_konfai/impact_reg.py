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
its model produces one thing: the displacement field (``DisplacementField``) on the FIXED grid, in
whatever format it declares. That is the whole contract. The moved image IS that field applied to the
moving, so this orchestrator derives it rather than asking every preset to write it as well. This layer
adds the registration-specific
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
from konfai.utils.ITK import apply_to_data_transform, read_displacement_field
from konfai_apps import KonfAIApp
from konfai_apps.app_repository import get_available_apps_on_hf_repo

# Preset apps live on this Hugging Face repo; override with KONFAI_IMPACTREG_REPO to point at a local
# directory of preset folders (each with an app.json) for development / offline use.
IMPACT_REG_KONFAI_REPO = os.environ.get("KONFAI_IMPACTREG_REPO", "VBoussot/ImpactReg")

_ENSEMBLE_DIR = "Ensemble"

# Writing needs a format named -- nothing is on disk yet to detect one from. The store spellings and
# the ITK transform files need translating; every other suffix is already the token Dataset
# normalises (".mha" -> "mha").
_FORMATS = {".ome.zarr": "omezarr", ".zarr": "omezarr", ".h5": "itktransform", ".tfm": "itktransform"}


def _is_transform_file(path: Path) -> bool:
    """An ITK transform file, as opposed to a displacement field stored as an image or a store."""
    return path.suffix in {".h5", ".tfm"} or path.name.endswith(".itk.txt")


def _as_transform(path: Path) -> "sitk.Transform":
    """The stored registration as one ``sitk.Transform``, whatever form the preset wrote it in."""
    if _is_transform_file(path):
        return sitk.ReadTransform(str(path))
    return sitk.DisplacementFieldTransform(sitk.Cast(read_displacement_field(path), sitk.sitkVectorFloat64))


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


def _find_output_group(root: Path) -> str:
    """The name of the single output group a preset produced under ``root``.

    A preset declares ONE output — its transform, in whatever form and under whatever name it chose.
    konfai writes one dataset per output group (``<run>/<group>/<case>/<group>.<ext>``), so the group
    is the directory holding the cases. Discovering it rather than assuming ``DVF`` is what lets an
    official preset name its output ``Transform`` — where Slicer looks for it — while this pipeline's
    own name theirs ``DVF``, with no branch here.
    """
    runs = [child for child in sorted(root.iterdir()) if child.is_dir()] if root.is_dir() else []
    groups = [group for run in runs for group in sorted(run.iterdir()) if group.is_dir()]
    if len(groups) != 1:
        found = ", ".join(group.name for group in groups) or "none"
        raise FileNotFoundError(
            f"Expected the preset to produce exactly one output group under {root}, found {found}."
            " A registration preset declares one output: its transform."
        )
    return groups[0].name


def _find_outputs(root: Path, stem: str) -> dict[str, Path]:
    """Every output named ``stem`` under ``root``, keyed by the CASE it belongs to.

    Matched on the name rather than on a fixed filename: a displacement field may come out as an ITK
    image or as an OME-Zarr store, and a store is a DIRECTORY whose ``Path.stem`` is "DVF.ome" -- so a
    fixed "DVF.mha" finds nothing and the run dies with the output sitting in the directory it listed.

    A MAPPING, NOT THE FIRST MATCH. konfai-apps writes one dataset per output group --
    ``<run>/<group>/<case>/<group>.<ext>`` -- and one run produces as many cases as the inputs expanded
    to, which is not the number of paths on the command line: a directory is walked so each volume in
    it becomes its own case. Taking ``matches[0]`` therefore kept P000 and dropped every case after it,
    silently, with the results sitting on disk. The case is the entry's parent directory.
    """
    matches = sorted(root.rglob(f"{stem}.*"))
    if not matches:
        raise FileNotFoundError(f"Preset inference did not produce '{stem}' under {root}.")
    return {match.parent.name: match for match in matches}


def _output_path(dest_dir: Path, stem: str, suffixes: str) -> Path:
    """The path ``<stem><suffixes>`` in ``dest_dir``, with every earlier output of that stem removed.

    A re-run whose presets emit the other form would otherwise leave both ``DVF.mha`` and
    ``DVF.ome.zarr`` behind, and discovery is by stem: ``_find_output`` takes the first match, which
    sorts to the stale one. Only the current run's output is left standing.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    for stale in [p for p in dest_dir.iterdir() if p.name.startswith(f"{stem}.")]:
        shutil.rmtree(stale) if stale.is_dir() else stale.unlink()
    return dest_dir / (stem + suffixes)


def _units(paths: list[Path]) -> list[Path]:
    """What an input group expands to, in konfai-apps' own order.

    A file, an OME-Zarr store or a DICOM series is one unit; a plain directory is walked so every
    supported file inside becomes one, sorted so groups pair consistently. Asking konfai-apps rather
    than counting the command line is what keeps this layer's notion of a case identical to the one the
    ``Dataset/P{i:03d}`` staging is built with -- counting arguments instead is how a directory input
    came to register N volumes and report one.
    """
    return [source for source, _ in KonfAIApp._list_input_units(list(paths))] if paths else []


def _neutral_masks(work: Path, side: str, count: int) -> list[Path]:
    """``count`` all-ones sentinels, one per case, standing in for a mask group the caller left out.

    Input groups pair by POSITION, so a group that is present at all must carry as many units as the
    others: one sentinel would pair with the first case and leave the rest unmasked on that side.
    """
    return [_neutral_mask(work / f"{side}Mask_{index:03d}.mha") for index in range(count)]


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


def _stage_group(base: Path, group: str, entries: dict[str, Path]) -> str:
    """One dataset ROOT holding one GROUP — ``base/<group>/<case>/<group><suffixes>`` symlinks —
    returned as the ``path:format`` spec a run consumes. No bytes move.

    One root per group, not one mixed cohort: a directory dataset's backend is detected from its
    first case, and a single store entry flips the whole root to the store backend — the ``.mha``
    beside it stops resolving. A homogeneous root keeps every entry readable whatever mix of forms
    the caller and the presets produced; the run reads all its roots side by side.
    """
    forms = {"".join(source.suffixes).lower() for source in entries.values()}
    if len(forms) > 1:
        raise RuntimeError(
            f"group '{group}' mixes storage forms ({', '.join(sorted(forms))}); a directory dataset"
            " has one backend, so a mixed group cannot be staged. Re-run the producers to one form."
        )
    root = base / group
    suffixes = ""
    for case, source in entries.items():
        case_dir = root / case
        case_dir.mkdir(parents=True, exist_ok=True)
        suffixes = "".join(source.suffixes)
        # Clear every form of the entry, not only the current one: a re-stage that switched forms
        # would otherwise leave two links and discovery by stem finds both.
        for stale in case_dir.glob(f"{group}.*"):
            stale.unlink() if stale.is_symlink() or stale.is_file() else shutil.rmtree(stale)
        link = case_dir / (group + suffixes)
        link.symlink_to(source.resolve())
    return f"{root}:{_FORMATS.get(suffixes.lower(), suffixes.lstrip('.') or 'mha')}"


def _the_output(dest_dir: Path, stem: str) -> Path:
    """The single output named ``stem`` in ``dest_dir`` — one, exactly."""
    matches = sorted(dest_dir.glob(f"{stem}.*"))
    if len(matches) != 1:
        found = ", ".join(path.name for path in matches) or "none"
        raise FileNotFoundError(
            f"Expected exactly one '{stem}' under {dest_dir}, found {found}. More than one is a"
            " stale other-form output left beside the current one: remove it, or re-run register,"
            " which clears the stem before writing."
        )
    return matches[0]


def _run_transform(
    name: str,
    datasets: list[str],
    chains: dict,
    work: Path,
    gpu: list[int],
    cpu: int | None,
    quiet: bool,
) -> None:
    """One streamed TRANSFORM through konfai's Python API, its workspace under ``work``.

    Every derivation runs here: the plan prices each case and routes it (stream, load,
    whole-volume — with the reason), so the orchestrator never holds a volume in RAM and never
    resamples by hand. A designed refusal raises ``KonfAIError``; the CLI prints it.
    """
    from konfai import api

    api.transform(
        name,
        datasets,
        chains,
        gpu=list(gpu),
        cpu=cpu or 1,
        quiet=quiet,
        overwrite=True,
        transforms_dir=work / "Workspaces",
    )


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
        fixed_images: list[Path],
        moving_images: list[Path],
        fixed_masks: list[Path],
        moving_masks: list[Path],
        n_cases: int,
        work: Path,
        gpu: list[int],
        cpu: int | None,
        quiet: bool,
        tta: int = 0,
        config_overrides: list[str] | None = None,
    ) -> tuple[str, dict[str, Path]]:
        """Run one preset app on every case at once; return its output group and its transform per case.

        ONE RUN, NOT ONE PER CASE. Each ``-i`` is an input GROUP, and konfai-apps expands each group's
        paths into units -- a file is one, a store or DICOM series is one, a plain directory is walked
        so each volume in it becomes one -- then pairs the groups by position into ``Dataset/P{i:03d}``.
        So the whole cohort goes in a single invocation and the model is loaded once, rather than once
        per case.

        Each preset still runs in its own subprocess: konfai keeps process-global state (its ``Config``
        singleton, the ``KONFAI_*`` environment), so several preset inferences in one process would clash.

        Masks are optional: konfai-apps fills any we omit with an all-ones default, so with no mask we pass
        only fixed+moving. Because the mapping is positional, a lone moving mask still needs the fixed-mask
        slot present -- filled with one all-ones sentinel per case so the groups keep pairing.
        """
        out = work / preset
        command = ["konfai-apps", "infer", _app_id(preset)]
        command += ["-i", *(str(path) for path in fixed_images)]
        command += ["-i", *(str(path) for path in moving_images)]
        if fixed_masks or moving_masks:
            command += ["-i", *(str(path) for path in fixed_masks or _neutral_masks(work, "Fixed", n_cases))]
            command += ["-i", *(str(path) for path in moving_masks or _neutral_masks(work, "Moving", n_cases))]
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
        # THE PRESET'S CONTRACT IS THE FIELD, AND ONLY THE FIELD. A registration app produces a
        # displacement field on the fixed grid, in whatever format it declares; anything else that can
        # be computed from it -- the moved image above all -- is this layer's job. Looking for a Moved
        # here would make every preset carry an output it does not owe, and a tiled one blend it across
        # every patch seam for a caller that has the field.
        group = _find_output_group(out)
        return group, _find_outputs(out, group)

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
        fields_only: bool = False,
    ) -> None:
        """Register every case with the selected presets and ensemble their DVFs.

        A case is whatever konfai-apps makes of the inputs: one image per group gives one case, and a
        directory per group gives one case per volume inside it, paired by position. So the caller may
        pass single volumes or whole datasets, exactly as ``konfai-apps infer`` accepts them.

        Masks are optional and restrict the metric region; when omitted a whole-image mask is auto-filled,
        so every preset app always receives the four inputs (fixed, moving, fixed mask, moving mask) it declares.

        ``tmp_dir`` names where the intermediates are staged; see :func:`_work_dir`.

        ``fields_only`` writes the transforms and stops there. The moved image is derived FROM the
        transform, at the cost of a full-size resample and a full-size rewrite -- worth it for a
        caller that wants a registration to look at, waste for one that composes the field with
        another and derives its own. A caller that reads only the fields should be able to say so
        rather than pay for an output it deletes.
        """
        # The cases are konfai-apps' to define, not ours to count. It expands each input GROUP into
        # units -- a file, a store, a DICOM series, or every volume inside a plain directory -- and pairs
        # the groups by position. Asking it for the moving group's units is what tells this layer which
        # volume belongs to which case, in the same order it will use, so a dataset in and a single pair
        # in go down one path.
        moving_units = _units(moving_images)
        for side, masks in (("fixed", fixed_masks), ("moving", moving_masks)):
            mask_units = _units(list(masks)) if masks else []
            if masks and len(mask_units) != len(moving_units):
                raise RuntimeError(
                    f"the {side} masks expand to {len(mask_units)} unit(s) for {len(moving_units)}"
                    " case(s); masks pair with cases by position, so the counts must match."
                )

        work = _work_dir(tmp_dir, "impact_reg_")
        try:
            fields_by_preset = {
                preset: self._infer_preset(
                    preset,
                    list(fixed_images),
                    list(moving_images),
                    list(fixed_masks),
                    list(moving_masks),
                    len(moving_units),
                    work,
                    gpu,
                    cpu,
                    quiet,
                    tta,
                    config_overrides,
                )
                for preset in presets
            }
            groups = {group for group, _ in fields_by_preset.values()}
            if len(groups) != 1:
                raise RuntimeError(
                    f"the presets named their output differently ({', '.join(sorted(groups))}); an"
                    " ensemble folds one group, so every member must declare the same one."
                )
            group = groups.pop()
            fields_by_preset = {preset: fields for preset, (_, fields) in fields_by_preset.items()}
            cases = sorted(fields_by_preset[presets[0]])
            for preset, fields in fields_by_preset.items():
                if sorted(fields) != cases:
                    raise RuntimeError(
                        f"preset '{preset}' produced cases {sorted(fields)} where '{presets[0]}' produced "
                        f"{cases}; an ensemble can only be averaged case by case."
                    )
            if len(cases) != len(moving_units):
                raise RuntimeError(
                    f"the presets produced {len(cases)} case(s) for {len(moving_units)} moving unit(s); "
                    "the moved image is derived per case and needs the two to line up."
                )

            for case in cases:
                # konfai-apps already named the cases; reusing its names keeps the two layers' notion of
                # a case identical instead of renumbering from the command line and hoping they agree.
                case_out = output / case
                case_out.mkdir(parents=True, exist_ok=True)
                # The per-preset displacement fields are large; only persist them (under Ensemble/) when the
                # caller asks, so `uncertainty` can measure the ensemble spread afterwards.
                if keep_dvf:
                    (case_out / _ENSEMBLE_DIR).mkdir(parents=True, exist_ok=True)

                dvf_paths = []
                for preset in presets:
                    dvf = fields_by_preset[preset][case]
                    if keep_dvf:
                        dvf = _copy_output(dvf, case_out / _ENSEMBLE_DIR, preset)
                    dvf_paths.append(dvf)

                if len(presets) == 1:
                    _copy_output(dvf_paths[0], case_out, group)
                else:
                    # Ensemble: fold the presets' fields (all on the fixed grid -- and Reduce VERIFIES
                    # the claim) into the averaged DVF, the one output no single preset produced.
                    # Streamed: the fold is incremental, so the peak is one accumulator plus the
                    # member being read, whatever the size of the ensemble.
                    self._ensemble_mean(case, group, presets, dvf_paths, output, work, gpu, cpu, quiet)

            if not fields_only:
                # The moved images in ONE streamed run over the cohort: Resample adopts, per case,
                # the grid of that case's own field (a field is defined ON the fixed grid) and reads
                # the field as the map.
                self._derive_moved(dict(zip(cases, moving_units, strict=True)), group, output, work, gpu, cpu, quiet)
        finally:
            shutil.rmtree(work, ignore_errors=True)

    def _ensemble_mean(
        self,
        case: str,
        group: str,
        presets: list[str],
        dvf_paths: list[Path],
        output: Path,
        work: Path,
        gpu: list[int],
        cpu: int | None,
        quiet: bool,
    ) -> None:
        """Average one case's preset fields into ``<output>/<case>/<group>`` — Reduce(Mean), streamed."""
        from konfai.data.transform import Reduce, Write

        members = _stage_group(
            work / f"ensemble_{case}", "DVF", {preset: dvf for preset, dvf in zip(presets, dvf_paths, strict=True)}
        )
        suffixes = "".join(dvf_paths[0].suffixes)
        _output_path(output / case, group, suffixes)  # drop a stale other-form output before writing
        _run_transform(
            f"impact_reg_ensemble_{case}",
            [members],
            {
                "DVF": {
                    group: [
                        Reduce(operator="Mean", output=case, grid="strict"),
                        Write(dataset=f"{output}:{_FORMATS.get(suffixes.lower(), suffixes.lstrip('.'))}"),
                    ]
                }
            },
            work,
            gpu,
            cpu,
            quiet,
        )

    def _derive_moved(
        self,
        cases: dict[str, Path],
        group: str,
        output: Path,
        work: Path,
        gpu: list[int],
        cpu: int | None,
        quiet: bool,
    ) -> None:
        """The moved images, derived from the transforms — one streamed run over the whole cohort.

        A preset that emits only a field is complete: everything else IS that field. The moved
        image: ``reference: '{case}'`` adopts each case's own DVF grid (a field is defined ON the
        fixed grid) and the field is the map (``field_group``) — one interpolation, streamed, each
        slab's source window sized from the field values it reads.

        NOTHING ELSE IS DERIVED. A preset writes its transform in the form its consumer reads — an ITK
        transform file where Slicer picks it up, an RFC-5 store where this pipeline streams it — so
        there is no second copy of the same field to produce under another name.
        """
        from konfai.data.transform import Resample, Write

        fields = {case: _the_output(output / case, group) for case in cases}
        # The moved image is written in the MOVING's own form: the fields' form may be an ITK
        # transform file, which cannot hold an image. _stage_group refuses a mixed Moving group
        # below, so the first case's form speaks for the cohort.
        suffixes = "".join(next(iter(cases.values())).suffixes)
        for case in fields:
            _output_path(output / case, "Moved", suffixes)  # drop a stale other-form Moved
        moving_root = _stage_group(work / "moved_stage", "Moving", cases)
        field_root = _stage_group(work / "moved_stage", "DVF", fields)
        _run_transform(
            "impact_reg_moved",
            [moving_root, field_root],
            {
                "Moving": {
                    "Moved": [
                        Resample(reference="{case}", reference_group="DVF", field_group="DVF"),
                        Write(dataset=f"{output}:{_FORMATS.get(suffixes.lower(), suffixes.lstrip('.'))}"),
                    ]
                },
            },
            work,
            gpu,
            cpu,
            quiet,
        )

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
        # Every group is expanded the same way ``register`` expands its inputs, so a directory of
        # volumes evaluates case by case instead of collapsing to its first entry. Transforms and
        # landmark files expand too: .h5, .fcsv and .itk.txt are all supported extensions.
        fixed_images, moving_images = _units(fixed_images), _units(moving_images)
        gt_fixed_seg, gt_moving_seg = _units(gt_fixed_seg), _units(gt_moving_seg)
        gt_fixed_fid, gt_moving_fid = _units(gt_fixed_fid), _units(gt_moving_fid)
        transforms = _units(transforms)
        mask = _units(mask) if mask else mask
        n_cases = max(len(fixed_images), len(gt_fixed_seg), len(gt_fixed_fid))
        for index in range(n_cases):
            transform_path = transforms[index] if index < len(transforms) else None
            eval_out = output / f"P{index:03d}" / "Evaluation"
            work = _work_dir(tmp_dir, "impact_reg_eval_")
            try:
                # Image: moving resampled onto the fixed grid vs fixed (MAE). Mask is optional.
                if index < len(fixed_images) and index < len(moving_images):
                    moved = self._warp_onto_fixed(
                        work, "img", fixed_images[index], moving_images[index], transform_path, gpu, cpu, quiet
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

                # Segmentation: moving seg warped onto fixed vs fixed seg (Dice), nearest-neighbour.
                if index < len(gt_fixed_seg) and index < len(gt_moving_seg):
                    moved_seg = self._warp_onto_fixed(
                        work, "seg", gt_fixed_seg[index], gt_moving_seg[index], transform_path, gpu, cpu, quiet
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
                # the raw fiducials are compared, measuring the initial misalignment. Landmarks are a few
                # points, so this is the one place the transform is opened in this process.
                if index < len(gt_fixed_fid) and index < len(gt_moving_fid):
                    fixed_points = read_landmarks(gt_fixed_fid[index])
                    if transform_path is not None:
                        fixed_points = apply_to_data_transform(fixed_points, {_as_transform(transform_path): False})
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

    def _warp_onto_fixed(
        self,
        work: Path,
        kind: str,
        fixed: Path,
        moving: Path,
        transform_path: Path | None,
        gpu: list[int],
        cpu: int | None,
        quiet: bool,
    ) -> Path:
        """The moving warped onto the fixed grid — one streamed Resample; nearest for a ``seg``.

        ``reference: '{case}'`` adopts the fixed grid; the transform, when given, is staged as a
        stored-transform group konfai decodes itself (rigid, affine, spline, field or composite) —
        with its exact affine box or its coefficient-derived bound sizing what each slab reads.
        With no transform the map is the identity and this is a change of grid alone.
        """
        from konfai.data.transform import Resample, Write

        base = work / f"warp_{kind}"
        datasets = [
            _stage_group(base, "Fixed", {"P000": fixed}),
            _stage_group(base, "Moving", {"P000": moving}),
        ]
        resample: dict[str, object] = {"reference": "{case}", "reference_group": "Fixed"}
        if kind == "seg":
            resample["interpolation"] = "nearest"
        if transform_path is not None:
            datasets.append(_stage_group(base, "Reg", {"P000": transform_path}))
            if _is_transform_file(transform_path):
                resample["transforms"] = {"Reg": False}
            else:
                resample["field_group"] = "Reg"
        out_root = work / f"moved_{kind}"
        _run_transform(
            f"impact_reg_eval_{kind}",
            datasets,
            {"Moving": {"Moved": [Resample(**resample), Write(dataset=f"{out_root}:mha")]}},
            work,
            gpu,
            cpu,
            quiet,
        )
        return out_root / "P000" / "Moved.mha"

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

        Each member's magnitude, then the standard deviation across members: ``Magnitude`` is
        pointwise and ``Reduce(Std)`` folds with running moments, so no member is ever whole in RAM
        whatever the size of the ensemble. ``preset`` is accepted for CLI compatibility and unused —
        measuring a spread needs no preset app. Writes ``<output>/uncertainty/Uncertainty.<ext>``.
        """
        del preset
        if len(dvfs) < 2:
            raise ValueError("Uncertainty needs at least two ensemble displacement fields.")
        work = _work_dir(tmp_dir, "impact_reg_unc_")
        try:
            from konfai.data.transform import Magnitude, Reduce, Write

            members = _units(list(dvfs))
            spec = _stage_group(work / "members", "DVF", {f"M{index:03d}": dvf for index, dvf in enumerate(members)})
            suffixes = ".mha" if _is_transform_file(members[0]) else "".join(members[0].suffixes)
            _output_path(output / "uncertainty", "Uncertainty", suffixes)  # drop a stale other-form map
            _run_transform(
                "impact_reg_uncertainty",
                [spec],
                {
                    "DVF": {
                        "Uncertainty": [
                            Magnitude(),
                            Reduce(operator="Std", output="uncertainty", grid="strict"),
                            Write(dataset=f"{output}:{_FORMATS.get(suffixes.lower(), suffixes.lstrip('.'))}"),
                        ]
                    }
                },
                work,
                gpu,
                cpu,
                quiet,
            )
        finally:
            shutil.rmtree(work, ignore_errors=True)
