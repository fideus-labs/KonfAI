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
                    the evaluation configs this package ships (image MAE, segmentation Dice, landmark TRE);
- ``uncertainty`` : from the per-preset displacement fields, compute the voxel-wise spread map.
"""

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from konfai.utils.dataset import Attribute, Dataset, DataStream, is_staging_entry, read_landmarks, write_landmarks
from konfai.utils.dataset.staging import publish
from konfai.utils.errors import EXIT_OUT_OF_MEMORY, EvaluatorError, KonfAIError
from konfai.utils.ITK import displace_points, field_reach, jacobian_statistics
from konfai.utils.utils import format_token, path_format_token, storage_form
from konfai_apps import KonfAIApp
from konfai_apps.app_repository import AppRepositoryError, LocalAppRepository, check_overrides, did_you_mean, pass_cost
from konfai_apps.cli import app_file, app_id, app_names

from impact_reg_konfai import PRESETS_REPO

# Preset apps live on this Hugging Face repo, at the revision this release pins; KONFAI_IMPACTREG_REPO points
# it at a local directory of preset folders (each with an app.json) for development / offline use.
IMPACT_REG_KONFAI_REPO = PRESETS_REPO

_ENSEMBLE_DIR = "Ensemble"

#: The app ``evaluate`` runs: the image, segmentation and landmark evaluation configs, as package data.
_EVALUATION_APP = Path(__file__).resolve().with_name("evaluation")


class _OutOfMemory(KonfAIError):
    """A preset pass that ran out of GPU memory (KonfAI's ``EXIT_OUT_OF_MEMORY``): KonfAI had already resampled or
    tiled it down to its floor."""


# The one place this layer disagrees with konfai's own reading of a suffix: an `.h5` or `.tfm` HERE
# is a registration (what every preset writes), not konfai's monolithic HDF5 dataset. Everything
# else (the store spellings, a DICOM series, plain extensions) is konfai's to resolve.
_TRANSFORM_FORMS = {".h5", ".tfm"}


def _form(path: Path) -> str:
    """The storage form of ``path``: konfai's, which knows a dot in a stem is part of the name."""
    return storage_form(path)


def _format_token(form: str) -> str:
    """The konfai backend token a storage form is read and written through, transforms included."""
    return "itktransform" if form.lower() in _TRANSFORM_FORMS else format_token(form)


def _backend(path: Path) -> str:
    """The backend token ``path`` is read and written through, as it sits on disk.

    A DICOM series is a directory carrying no extension at all, so its token cannot be read off a
    name: ``path_format_token`` looks inside.
    """
    return "itktransform" if _form(path).lower() in _TRANSFORM_FORMS else path_format_token(path)


def _is_transform_file(path: Path) -> bool:
    """An ITK transform file (``.h5``, ``.tfm``, ``.itk.txt``, in any case), as opposed to a displacement
    field stored as an image or a store. One rule with ``_backend``."""
    form = _form(path).lower()
    return form in _TRANSFORM_FORMS or form == ".itk.txt"


def _transform_kinds(path: Path) -> list[str]:
    """The ITK type names a transform file stores, a composite's and its leaves', read off the file alone:
    a displacement field is never loaded to learn that it is one."""
    import h5py

    if h5py.is_hdf5(path):
        # Unlocked, as konfai opens every HDF5 file: HDF5 refuses a second handle whose locking flag differs,
        # and konfai's read pool may already hold this one (the landmarks read the field through it).
        with h5py.File(path, "r", locking=False) as file:
            return [bytes(group["TransformType"][0]).decode() for group in file["TransformGroup"].values()]
    with open(path, errors="replace") as file:
        return [line.split(":", 1)[1].strip() for line in file if line.startswith("Transform:")]


def _is_field(path: Path) -> bool:
    """Whether a stored registration is a displacement field: an image or store, or a field transform file."""
    return not _is_transform_file(path) or _transform_kinds(path)[0].startswith("DisplacementFieldTransform")


def _field(path: Path, work: Path) -> tuple[Dataset, str, str]:
    """A stored displacement field staged as a dataset entry, for KonfAI's field readers."""
    root, _, backend = _stage_group(work / "field", "Reg", {"P000": path}).rpartition(":")
    return Dataset(root, backend), "Reg", "P000"


def _compose_padding(headers: "list[dict[str, _Header]]", budget: int, reach: float) -> list[int]:
    """The edge, in fixed voxels and ``F.pad`` order (x, y, z pairs), the global field is extended by before the tiles'
    displacement reads it: at least 8 voxels of the grid the global pass ran on (``coarse_spacing``, KonfAI's
    ``mode: resample``) and past the longest tile displacement, ``reach``, by one of them. The widest over the cases."""
    from konfai.data.transform.resample import coarse_spacing

    widest = [0, 0, 0]
    for case in headers:
        shape, spacing = case["Fixed"][0][-3:], case["Fixed"][1]
        coarse = coarse_spacing(list(shape), list(spacing), budget) or spacing
        for axis, (step, goal) in enumerate(zip(spacing, coarse, strict=True)):
            widest[axis] = max(widest[axis], math.ceil(max(8 * goal, reach + goal) / step))
    return [extent for extent in widest for _ in range(2)]


def _jacobian_statistics(path: Path, work: Path) -> dict[str, float]:
    """KonfAI's Jacobian statistics of a stored field, under the names the evaluation summary reports."""
    return {f"Transform:Jacobian:{key}": value for key, value in jacobian_statistics(*_field(path, work)).items()}


def _displace(points: np.ndarray, path: Path, work: Path) -> np.ndarray:
    """``points`` (``[N, 3]``, LPS) moved by the stored registration, fixed -> moving, whatever its form: a field
    read around the points alone (KonfAI's ``displace_points``), any other transform read and applied whole."""
    if not _is_field(path):
        transform = sitk.ReadTransform(str(path))
        return np.array([transform.TransformPoint(point) for point in points.astype(np.float64).tolist()])
    return displace_points(points, *_field(path, work))


def _case_key(name: str) -> tuple[int, str]:
    """konfai-apps numbers cases ``P000``..: zero-padded to three digits, longer past ``P999`` --
    so length-then-lexicographic IS its numeric order, without parsing a preset's own naming."""
    return (len(name), name)


def _manifest(preset: str) -> dict:
    """The preset's ``app.json``, from the local preset directory or the Hugging Face repo; empty if unreadable.

    Read cache first, as the preset run resolves its own files: an online read moved the cache to the newest
    preset, which only the runs that read a manifest then picked up.
    """
    try:
        return json.loads(app_file(IMPACT_REG_KONFAI_REPO, preset, "app.json").read_text(encoding="utf-8"))
    except Exception:  # an unreadable manifest declares nothing; resolving the preset reports a bad name itself
        return {}


def _preset_config(preset: str, filename: str) -> dict:
    """One of the preset's prediction configs, as data."""
    from ruamel.yaml import YAML

    return YAML(typ="safe").load(app_file(IMPACT_REG_KONFAI_REPO, preset, filename).read_text(encoding="utf-8")) or {}


#: A key a prediction config leaves out, at its model's default (see ``_agree``).
_UNSET = object()

#: The passes of a preset a ``--set`` scope names: the preset whole or its global pass, and its tile pass.
_PASSES = ("global", "tile")


def _config_value(config: dict, key: str) -> object:
    """What ``--set key=...`` names in a prediction config, resolved as konfai-apps resolves it: a bare name in the
    model's parameter block, a dotted one from the config root. ``_UNSET`` when the config has no such key."""
    node = config if "." in key else LocalAppRepository._model_param_block(config)[1]
    for part in filter(None, key.split(".")):
        if not isinstance(node, dict) or part not in node:
            return _UNSET
        node = node[part]
    return node


def _agree(configs: dict[str, dict], key: str) -> bool:
    """Whether the global and the tile config set ``key`` alike; left out of both, it is at the model's default in
    both alike."""
    return _config_value(configs["global"], key) == _config_value(configs["tile"], key)


def _scopes(override: str, presets: list[str]) -> tuple[str | None, str | None, str]:
    """``[PRESET:][global:|tile:]NAME=VALUE`` -> ``(preset, pass, NAME=VALUE)``, None for a scope left out.

    A scope is what comes before a colon when it holds neither ``=`` nor ``.``, so the colon of a value
    (``ref=repo:file.pt``) never makes one; the preset comes first.
    """
    preset = stage = None
    rest = override
    while stage is None:
        head, colon, tail = rest.partition(":")
        if not colon or "=" in head or "." in head or (head not in _PASSES and preset is not None):
            break
        if head in _PASSES:
            stage = head
        elif head not in presets:
            raise KonfAIError(
                "ImpactReg",
                f"--set {override} names the preset '{head}', which this run does not register with.",
                f"Its presets: {', '.join(presets)}.",
            )
        else:
            preset = head
        rest = tail
    return preset, stage, rest


def _preset_overrides(
    presets: list[str], overrides: list[str] | None, quiet: bool = False
) -> dict[str, dict[str, list[str]]]:
    """Each preset's share of the ``--set`` overrides, per pass, checked against the configs they run before anything
    runs.

    ``NAME=VALUE`` goes to every preset, ``PRESET:NAME=VALUE`` to that one alone: the engines name one knob
    differently (``deformable_iterations`` for FireANTs, ``max_iterations`` for elastix, ``iterations`` for
    ConvexAdam), so an ensemble needs the second form.

    A share has two passes: ``global``, the preset registered whole or its global pass (``tiling["global"]``,
    ``Prediction.yml`` when it declares none), and ``tile``, its tile pass (``tiling["tile"]``, see
    ``_register_tiled``). An override goes to both where their configs set its NAME alike, to the global pass alone
    where they differ: the tile config differs from the global config only where that makes it the deformable stage
    alone (``linear`` switched off, a ``stages`` or ``parameter_maps`` list without the linear stages,
    ``linear_method: none``), and forwarded there, an override of such a parameter would bring a linear stage back
    into every tile. ``global:NAME=VALUE`` and ``tile:NAME=VALUE`` (``PRESET:tile:NAME=VALUE``) give it to that pass
    alone. Where each override goes is said here, before anything runs, and one kept from the tiles even under
    ``quiet``.

    Each pass's share is applied to a copy of its config by konfai-apps' own resolver, so a name, a value or a range
    the pass would refuse stops the run here, that of the tiles included. A config that cannot be read (offline,
    nothing cached) leaves its share to that resolver, in the preset's own process.
    """
    shares: dict[str, list[tuple[str | None, str]]] = {preset: [] for preset in presets}
    for override in overrides or []:
        preset, stage, rest = _scopes(override, presets)
        for member in [preset] if preset else presets:
            shares[member].append((stage, rest))
    passes: dict[str, dict[str, list[str]]] = {preset: {stage: [] for stage in _PASSES} for preset in presets}
    for preset, share in shares.items():
        if not share:
            continue
        tiling = _manifest(preset).get("tiling") or {}
        files = {"global": tiling.get("global") or "Prediction.yml", "tile": tiling.get("tile")}
        configs = {}
        for stage, filename in files.items():
            try:
                if filename:
                    configs[stage] = _preset_config(preset, filename)
            except Exception:  # offline, nothing cached: konfai-apps checks this share in the preset's own process
                continue
        where = {
            ("global", "tile"): "the preset and its tiles",
            ("global",): "the preset, not its tiles" if files["tile"] else files["global"],
            ("tile",): "its tiles, not the preset",
        }
        for stage, override in share:
            name, assigned, _ = override.partition("=")
            name = name.strip()
            shown = f"{stage}:{override}" if stage else override
            if not assigned:
                raise KonfAIError("ImpactReg", f"--set {shown} assigns nothing.", "Write it NAME=VALUE.")
            if stage == "tile" and not files["tile"]:
                raise KonfAIError(
                    "ImpactReg",
                    f"--set {shown}: the preset '{preset}' has no tile pass.",
                    "tile: names the pass of a preset whose app.json declares tiling['tile'], as"
                    " --set PRESET:tile:NAME=VALUE.",
                )
            withheld = False
            if stage:
                targets = [stage]
            elif not files["tile"]:
                targets = ["global"]
            elif tiling.get("global") and len(configs) == 2 and not _agree(configs, name):
                targets, withheld = ["global"], True
            else:
                targets = ["global", "tile"]
            if withheld:
                held = _config_value(configs["tile"], name)
                there = "leaves it out" if held is _UNSET else f"sets it to {held!r}"
                print(
                    f"[ImpactReg] {preset}: --set {shown} -> {where[('global',)]}: {files['tile']} {there}"
                    f" (--set {preset}:tile:{override} sets it in the tiles too).",
                    flush=True,
                )
            elif not quiet:
                print(f"[ImpactReg] {preset}: --set {shown} -> {where[tuple(targets)]}.", flush=True)
            for target in targets:
                passes[preset][target].append(override)
        for stage, config in configs.items():
            try:
                check_overrides(config, passes[preset][stage])
            except AppRepositoryError as error:
                raise KonfAIError(
                    "ImpactReg",
                    f"{preset} ({files[stage]}): {error.args[0]}",
                    f"'impact-reg-konfai show {preset}' lists its parameters. A dotted NAME is a path from its"
                    f" config root, and --set {preset}:NAME=VALUE sets it for this preset alone.",
                ) from None
    return passes


def _plan(manifest: dict, gpu: list[int], max_voxels: int | None) -> tuple[int | None, int | None]:
    """The voxels a preset registers whole and those one of its tiles holds (see ``register``): what the device holds
    at the peak costs its ``app.json`` declares (KonfAI's ``max_voxels``), None for a pass that declares none; or
    ``max_voxels`` for the whole pass, the tiles in proportion to their cost."""
    from konfai.utils.vram import max_voxels as voxels_on

    tiling = manifest.get("tiling") or {}
    whole = pass_cost(manifest, tiling.get("global") or "Prediction.yml")
    tile = pass_cost(manifest, tiling["tile"]) if tiling.get("tile") else None
    if max_voxels:
        device = "vram" if gpu else "ram"
        ratio = whole.get(device, 0) / tile[device] if tile and tile.get(device) else 0
        return max_voxels, int(max_voxels * ratio) if ratio else max_voxels
    at = gpu[0] if gpu else None
    return voxels_on(whole.get("vram"), whole.get("ram"), at), voxels_on(
        tile.get("vram"), tile.get("ram"), at
    ) if tile else None


def _find_output_group(root: Path) -> str:
    """The name of the single output group a preset produced under ``root``.

    A preset declares ONE output: its transform, in whatever form and under whatever name it chose.
    konfai writes one dataset per output group (``<run>/<group>/<case>/<group>.<ext>``), so the group
    is the directory holding the cases. Discovering it rather than assuming ``DVF`` is what lets an
    official preset name its output ``Transform``: where Slicer looks for it, while this pipeline's
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
    image or as an OME-Zarr store, a DIRECTORY whose ``Path.stem`` is "DVF.ome".

    konfai-apps writes one dataset per output group (``<run>/<group>/<case>/<group>.<ext>``), and one run produces
    as many cases as the inputs expanded to, a directory walked into one case per volume. The case is the entry's
    parent directory.
    """
    matches = sorted(root.rglob(f"{stem}.*"))
    if not matches:
        raise FileNotFoundError(f"Preset inference did not produce '{stem}' under {root}.")
    return {match.parent.name: match for match in matches}


def _is_entry(path: Path, stem: str) -> bool:
    """Whether ``path`` is an entry named ``stem``, in any form: extension, store, or bare name; never a writer's
    staging entry. The bare name is a form of its own: a DICOM series is a directory carrying no extension at all.
    """
    return (path.name == stem or path.name.startswith(f"{stem}.")) and not is_staging_entry(path.name)


def _drop_other_forms(dest_dir: Path, stem: str, suffixes: str) -> None:
    """Remove the outputs named ``stem`` in another form than ``<stem><suffixes>``, once that one is published:
    discovery is by stem, so a re-run whose presets emit another form leaves only its own output standing."""
    for stale in [p for p in dest_dir.iterdir() if _is_entry(p, stem) and p.name != stem + suffixes]:
        shutil.rmtree(stale) if stale.is_dir() else stale.unlink()


def _units(paths: list[Path]) -> list[Path]:
    """What an input group expands to, in konfai-apps' own order.

    A file, an OME-Zarr store or a DICOM series is one unit; a plain directory is walked so every
    supported file inside becomes one, sorted so groups pair consistently. Asking konfai-apps keeps this layer's
    notion of a case identical to the one the ``Dataset/P{i:03d}`` staging is built with.
    """
    return [source for source, _ in KonfAIApp._list_input_units(list(paths))] if paths else []


#: An image's ``[C, Z, Y, X]`` shape and its ``(x, y, z)`` spacing, origin and flattened direction, from its header.
_Header = tuple[list[int], list[float], list[float], list[float]]


def _cases(groups: dict[str, list[Path]]) -> list[list[Path]]:
    """``eval``'s groups, keyed by their option, each as long as the case count: the longest group.

    A group holds one entry per case or one used for every case (one atlas against several movings, one
    mask), or is empty; each modality comes as a pair (a fixed side without its moving one scores
    nothing). Anything else is refused.
    """
    for fixed, moving in (("-f", "-m"), ("--gt-fixed-seg", "--gt-moving-seg"), ("--gt-fixed-fid", "--gt-moving-fid")):
        if bool(groups[fixed]) != bool(groups[moving]):
            given, missing = (fixed, moving) if groups[fixed] else (moving, fixed)
            raise EvaluatorError(
                f"{given} is given without {missing}.", "A modality is scored from both sides: give both, or neither."
            )
    count = max(len(entries) for entries in groups.values())
    wrong = [f"{option} has {len(entries)}" for option, entries in groups.items() if len(entries) not in (0, 1, count)]
    if wrong:
        raise EvaluatorError(
            f"{count} cases to evaluate, but {', '.join(wrong)}.",
            "Give each option one entry, used for every case, or one per case in the same order.",
        )
    return [entries * count if len(entries) == 1 else entries for entries in groups.values()]


def _write_evaluation_summary(path: Path, scored: list[tuple[str, Path]]) -> None:
    """``path``: every metric this ``eval`` wrote, per case and over the cohort; ``scored`` lists each case's
    id and the folder of each modality scored for it.

    Each case is evaluated alone, so its own ``Metric_TRAIN.json``, one per modality, names it P000 and
    aggregates one value. Here each case carries its real id (P000.. in input order, as ``register``
    numbers them) and the aggregates are the cohort's, in the evaluator's own schema: ``Statistics.read``
    reads it, as does SlicerImpactReg, which takes the first JSON under ``-o``.
    """
    from konfai.evaluator import Statistics

    summary = Statistics(path)
    for case, folder in scored:
        report = json.loads((folder / "ImpactReg" / "Metric_TRAIN.json").read_text())
        summary.directions.update(report.get("directions", {}))
        values = {key: next(iter(values.values())) for key, values in report["case"].items()}
        summary.add({key: math.nan if value is None else value for key, value in values.items()}, case)
    summary.write([summary.measures])


def _check_presets(presets: list[str]) -> None:
    """Refuse a repeated or unknown preset name before anything runs. A preset whose ``app.json`` reads is known;
    otherwise the listing decides, and an unreachable listing (offline, nothing cached) leaves the name to the
    preset's own resolution."""
    for preset in presets:
        if presets.count(preset) > 1:
            raise KonfAIError("ImpactReg", f"the preset '{preset}' is listed twice.", "Name each preset once.")
        if _manifest(preset):
            continue
        try:
            available = app_names(IMPACT_REG_KONFAI_REPO, force_update=True)
        except Exception:
            return
        if preset not in available:
            raise KonfAIError(
                "ImpactReg",
                f"there is no preset '{preset}' in {IMPACT_REG_KONFAI_REPO}.{did_you_mean(preset, available, 3)}",
                "'impact-reg-konfai list' lists them.",
            )


def _check_inputs(inputs: dict[str, list[Path]], output: Path) -> None:
    """Refuse, before any preset runs, the inputs this run would fail on or destroy.

    The groups pair by position, so each expands to as many volumes as the fixed one. Each group is staged as one
    dataset, read through one backend, so its volumes share one storage form.

    Each case directory register writes, ``<output>/P###``, has its transform, its moved image and its ensemble
    members replaced, so an input inside one would be deleted before it is read.
    """
    count = len(inputs["Fixed"])
    for group, units in inputs.items():
        if len(units) != count:
            raise KonfAIError(
                "ImpactReg",
                f"the {group} input expands to {len(units)} volume(s) for {count} fixed volume(s).",
                "The inputs pair by position: pass as many of each.",
            )
        forms = sorted({_form(unit).lower() or "a DICOM series" for unit in units})
        if len(forms) > 1:
            raise KonfAIError(
                "ImpactReg",
                f"the {group} volumes mix storage forms ({', '.join(forms)}).",
                "Convert them to one: each input group is staged as one dataset, read through one backend.",
            )
    cases = {output.resolve() / f"P{index:03d}" for index in range(count)}
    for group, units in inputs.items():
        for unit in units:
            if cases & set(unit.resolve().parents):
                raise KonfAIError(
                    "ImpactReg",
                    f"the {group} input {unit} lies in {unit.resolve().parent}, a case directory this run writes.",
                    "Pass another -o: register replaces the transform and the moved image of each case it writes.",
                )


def _check_volumes(inputs: dict[str, list[Path]], headers: list[dict[str, _Header]]) -> None:
    """Refuse, before any preset runs, a volume the presets cannot register: the engines are 3-D and single-channel,
    and a 2-D image, a colour one or a time series failed deep inside one of them."""
    for index, case in enumerate(headers):
        for group, (shape, *_) in case.items():
            if len(shape) != 4 or shape[0] != 1:
                raise KonfAIError(
                    "ImpactReg",
                    f"the {group} volume {inputs[group][index]} has the shape {shape}, channels first.",
                    "The presets register 3-D single-channel volumes: extract the channel or the time point first.",
                )


def _read_header(base: Path, group: str, source: Path) -> tuple[list[int], Attribute]:
    """``source``'s shape and attributes, from its header alone, staged as a one-case dataset under ``base``."""
    root, _, backend = _stage_group(base, group, {"P000": source}).rpartition(":")
    shape, attributes = Dataset(root, backend).get_infos(group, "P000")
    return list(shape), attributes


def _same_grid(first: _Header, second: _Header) -> bool:
    """Whether two images lie on one voxel grid: the same extent, spacing, origin and direction."""
    return first[0][-3:] == second[0][-3:] and all(
        np.allclose(one, other, atol=1e-4) for one, other in zip(first[1:], second[1:], strict=True)
    )


def _neutral_fixed_masks(work: Path, fixed: list[Path]) -> list[Path]:
    """One all-ones fixed mask per case, on that case's fixed grid, standing in for the fixed mask the caller left
    out beside a moving mask.

    Input groups pair by POSITION and only a trailing group may be left out, so the fixed-mask slot must be filled
    (a lone fixed mask needs nothing: konfai-apps fills the trailing moving mask itself, on the fixed grid). An
    all-ones mask restricts nothing, and on the fixed grid KonfAI can cut it into the fixed image's patches.

    They take the form of the images they stand beside: konfai-apps reads the staged dataset under the format of
    its first group.
    """
    units = _units(fixed)
    backend, form = _backend(units[0]), _form(units[0])
    masks = []
    for index, unit in enumerate(units):
        case = f"P{index:03d}"
        shape, attributes = _read_header(work / "headers" / case, "Fixed", unit)
        KonfAIApp.write_constant(Dataset(work, backend), "FixedMask", case, [1, *shape[1:]], attributes, 1)
        masks.append(work / case / f"FixedMask{form}")
    return masks


def _work_dir(tmp_dir: Path | None, output: Path, prefix: str) -> Path:
    """A private scratch directory for one command's intermediates.

    Under ``tmp_dir`` when the caller named one, the contract every other KonfAI app CLI offers through
    ``--tmp-dir``; hidden beside ``output`` otherwise, on the results' own filesystem.

    WHAT IS STAGED HERE IS VOLUME-SIZED: every preset's displacement field (24 bytes a fixed voxel, an ITK field
    being float64), and on a tiled pair the pre-warped moving, the tiles' field, the global field on the native grid
    and their composition, about 60 bytes a fixed voxel at the peak. Not the system temporary directory, often a
    tmpfs that charges it to RAM, and beside ``output`` collecting each result is a rename.

    The directory returned is always freshly created and owned by the caller of this function, never
    ``tmp_dir`` itself: the command removes what it made and leaves the directory it was given. Beside ``output``
    means in its parent, or inside ``output`` when that parent is not the user's to write (``-o .`` in a home
    directory under a root-owned ``/home``): the command writes ``output`` in any case.
    """
    if tmp_dir is not None:
        tmp_dir.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix=prefix, dir=tmp_dir))
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix=f".{output.name}.{prefix}", dir=output.parent))
    except PermissionError:
        output.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix=f".{prefix}", dir=output))


def _check_space(work: Path, headers: list[dict[str, _Header]], presets: int) -> None:
    """Warn when the disk under ``work`` holds less than the run may stage there: 24 bytes a fixed voxel for each
    preset's field and for their mean, and 40 more for the moved image and a tiled pass, whose composition holds the
    tiles' field, the global field on the native grid and their sum at once."""
    need = sum(math.prod(case["Fixed"][0][-3:]) for case in headers) * (24 * (presets + (presets > 1)) + 40)
    free = shutil.disk_usage(work).free
    if free < need:
        print(
            f"[ImpactReg] WARNING: {free / 1e9:.1f} GB free under {work.parent}, where this run may stage up to"
            f" {need / 1e9:.1f} GB. --tmp-dir puts the intermediates on another disk.",
            file=sys.stderr,
            flush=True,
        )


def _leave(work: Path, error: BaseException | None) -> None:
    """Remove a command's work dir, unless the command failed: its logs, its configs and the fields that finished
    are then what explains the failure, and the messages of KonfAI and of the presets point into it. An interrupted
    command (Ctrl-C, a stop from Slicer) has nothing to explain, and leaves nothing behind."""
    if isinstance(error, Exception):
        print(f"[ImpactReg] intermediates and logs kept in {work}", file=sys.stderr, flush=True)
    else:
        shutil.rmtree(work, ignore_errors=True)


def _copy_output(src: Path, dest_dir: Path, stem: str, move: bool = False) -> Path:
    """Copy an output beside the results, keeping the form the preset produced (file or store).

    ``move`` moves it instead, for a source in a workspace that is deleted next: a transform on a
    full-resolution grid is tens of gigabytes, and a copy wrote it a second time beside the first.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / (stem + _form(src))
    staging = Path(DataStream.staging_path(str(dest)))
    try:
        if move:
            shutil.move(src, staging)
        else:
            (shutil.copytree if src.is_dir() else shutil.copy2)(src, staging)
        publish(staging, dest)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True) if staging.is_dir() else staging.unlink(missing_ok=True)
        raise
    _drop_other_forms(dest_dir, stem, _form(src))
    if dest.is_dir():
        # A store put at a path already read is invisible to the reader's path-keyed memo, which
        # would otherwise pair the copy's voxels with the replaced store's axes and geometry.
        from konfai.utils.ome_zarr import clear_ome_zarr_cache

        clear_ome_zarr_cache()
    return dest


def _stage_group(base: Path, group: str, entries: dict[str, Path]) -> str:
    """One dataset ROOT holding one GROUP (``base/<group>/<case>/<group><suffixes>`` symlinks)
    returned as the ``path:format`` spec a run consumes. No bytes move.

    One root per group, not one mixed cohort: a directory dataset's backend is detected from its
    first case, and a single store entry flips the whole root to the store backend: the ``.mha``
    beside it stops resolving. A homogeneous root keeps every entry readable whatever mix of forms
    the caller and the presets produced; the run reads all its roots side by side.
    """
    forms = {_form(source).lower() for source in entries.values()}
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
        suffixes = _form(source)
        # Clear every form of the entry, not only the current one: a re-stage that switched forms
        # would otherwise leave two links and discovery by stem finds both. The bare name counts as
        # a form, that is how a DICOM series is stored, as a directory carrying no extension.
        for stale in [entry for entry in case_dir.iterdir() if _is_entry(entry, group)]:
            stale.unlink() if stale.is_symlink() or stale.is_file() else shutil.rmtree(stale)
        # A copy where symlinks are refused (Windows without Developer Mode raises WinError 1314); lower case:
        # konfai probes an entry under its lower-case extension.
        KonfAIApp.symlink(source.resolve(), case_dir / (group + suffixes.lower()))
    return f"{root}:{_backend(next(iter(entries.values())))}"


def _the_output(dest_dir: Path, stem: str) -> Path:
    """The single output named ``stem`` in ``dest_dir``, in any form (a DICOM series carries no extension): one."""
    matches = sorted(path for path in dest_dir.iterdir() if _is_entry(path, stem))
    if len(matches) != 1:
        found = ", ".join(path.name for path in matches) or "none"
        raise FileNotFoundError(
            f"Expected exactly one '{stem}' under {dest_dir}, found {found}. More than one is a"
            " stale other-form output left beside the current one: remove it, or re-run register,"
            " which clears the stem before writing."
        )
    return matches[0]


def _duration(seconds: float) -> str:
    """``1 h 3 min``, ``1 min 12 s`` or ``45 s``."""
    minutes, seconds = divmod(round(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes} min" if hours else f"{minutes} min {seconds} s" if minutes else f"{seconds} s"


def _shown(path: Path) -> Path:
    """``path`` relative to the working directory when it lies under it, as a user would type it."""
    return path.relative_to(Path.cwd()) if Path.cwd() in path.parents else path


def _version(package: str) -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _record(
    output: Path,
    run: dict,
    inputs: dict[str, list[Path]],
    cases: list[str],
    group: str,
    fields_only: bool,
    quiet: bool,
) -> None:
    """Write ``<output>/register.json``, which inputs each case was registered from, into which files, and how, then
    print the run's summary unless ``quiet``. konfai-apps names the cases by position: this says which input
    became ``P000``."""
    keys = {"Fixed": "fixed", "Moving": "moving", "FixedMask": "fixed_mask", "MovingMask": "moving_mask"}
    run["cases"] = {}
    for index, case in enumerate(cases):
        entry = {keys[side]: str(units[index]) for side, units in inputs.items()}
        entry["transform"] = str(_the_output(output / case, group).relative_to(output))
        if not fields_only:
            entry["moved"] = str(_the_output(output / case, "Moved").relative_to(output))
        run["cases"][case] = entry
    (output / "register.json").write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    if quiet:
        return
    presets = run["presets"]
    named = ", ".join(
        f"{preset} ({name})" if (name := _manifest(preset).get("display_name")) else preset for preset in presets
    )
    lines = [
        f"[ImpactReg] Registration completed in {_duration(run['runtime_s'])}"
        + (f", {len(cases)} cases" if len(cases) > 1 else ""),
        f"  Preset:    {named}" if len(presets) == 1 else f"  Presets:   ensemble of {len(presets)}, {named}",
    ]
    for case, entry in run["cases"].items():
        written = [_shown(output / entry[key]) for key in ("transform", "moved") if key in entry]
        if len(cases) > 1:
            pair = f"{Path(entry['fixed']).name} + {Path(entry['moving']).name}"
            lines.append(f"  {case}:      {pair} -> {', '.join(map(str, written))}")
            continue
        lines += [f"  Fixed:     {_shown(Path(entry['fixed']))}", f"  Moving:    {_shown(Path(entry['moving']))}"]
        lines += [f"  {label:<10} {path}" for label, path in zip(("Transform:", "Moved:"), written, strict=False)]
    lines.append(f"  Record:    {_shown(output / 'register.json')}")
    print("\n".join(lines), flush=True)


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
    whole-volume, with the reason), so the orchestrator never holds a volume in RAM and never
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
    if gpu:
        import gc

        import torch

        if torch.cuda.is_initialized():
            # A preset runs next in a child process and sizes its tiles from the whole card: what this
            # process's transform leaves in torch's cache would be missing from it.
            gc.collect()
            torch.cuda.empty_cache()


class ImpactRegKonfAIApp:
    """Run IMPACT-Reg preset apps, ensemble their displacement fields, evaluate, and estimate uncertainty."""

    def __init__(self, download: bool = False, force_update: bool = False) -> None:
        self._download = download
        self._force_update = force_update
        # The --set overrides each preset run took, by its workspace (``label``): ``applied`` in register.json.
        self._applied: dict[str, list[str]] = {}
        # Per preset, the voxels it registers whole and those a tile holds, KonfAI's ``max_voxels``: ``plans`` in
        # register.json. KonfAI's own re-plans after an out-of-memory are in the preset's log.
        self._plans: dict[str, dict[str, int | None]] = {}

    # ------------------------------------------------------------------ register

    def _infer_preset(
        self,
        preset: str,
        fixed_images: list[Path],
        moving_images: list[Path],
        fixed_masks: list[Path],
        moving_masks: list[Path],
        work: Path,
        gpu: list[int],
        cpu: int | None,
        quiet: bool,
        tta: int = 0,
        config_overrides: list[str] | None = None,
        prediction_file: str | None = None,
        label: str | None = None,
        max_voxels: int | None = None,
    ) -> tuple[str, dict[str, Path]]:
        """Run one preset app on every case at once; return its output group and its transform per case.

        ``prediction_file`` runs another of the preset's configs than ``Prediction.yml`` (its global or its tile
        pass, see ``_register_tiled``), and ``label`` names the workspace so the passes of one preset do not
        share it. ``max_voxels`` caps the pass (KonfAI's patch ``max_voxels``): the pair resampled onto a coarser
        grid in ``mode: resample``, cut into tiles in ``mode: tile``; recorded in ``plans``, not in ``applied``.

        ONE RUN, NOT ONE PER CASE. Each ``-i`` is an input GROUP, and konfai-apps expands each group's
        paths into units (a file is one, a store or DICOM series is one, a plain directory is walked
        so each volume in it becomes one), then pairs the groups by position into ``Dataset/P{i:03d}``.
        So the whole cohort goes in a single invocation and the model is loaded once, rather than once
        per case.

        Each preset still runs in its own subprocess: konfai keeps process-global state (its ``Config``
        singleton, the ``KONFAI_*`` environment), so several preset inferences in one process would clash.

        Masks are optional: konfai-apps fills any trailing group we omit with an all-ones default on the fixed
        grid, so with no mask we pass only fixed+moving, and with a lone fixed mask no moving mask. Because the
        mapping is positional, a lone moving mask still needs the fixed-mask slot present: filled with an all-ones
        mask on each case's fixed grid (``_neutral_fixed_masks``).
        """
        out = work / (label or preset)
        # This interpreter's konfai-apps, not the first on PATH: an unactivated environment has none there, and
        # another environment's would run another konfai and install the preset's requirements into it.
        command = [sys.executable, "-m", "konfai_apps", "infer", app_id(IMPACT_REG_KONFAI_REPO, preset)]
        command += ["-i", *(str(path) for path in fixed_images)]
        command += ["-i", *(str(path) for path in moving_images)]
        if moving_masks and not fixed_masks:
            fixed_masks = _neutral_fixed_masks(work / f"{label or preset}_fixed_mask", fixed_images)
        if fixed_masks:
            command += ["-i", *(str(path) for path in fixed_masks)]
        if moving_masks:
            command += ["-i", *(str(path) for path in moving_masks)]
        command += ["-o", str(out / "predictions")]
        # Hand konfai-apps a workspace we own, which is what every other app CLI does by exposing
        # --tmp-dir. Without it konfai-apps auto-creates one under TMPDIR, writes the prediction to
        # ./Predictions inside it, and copies that into `-o` before deleting it: one extra full-size
        # write of the displacement field, on whatever filesystem TMPDIR names. Given a caller-owned
        # workspace it writes straight into `-o` (see konfai_apps _stage_result_dir / _collect_result).
        # Beside `-o`, not `-o` itself: the bundle files it copies into its workspace (app.json, the
        # configs, a preset's own folders) stay out of the directory the output group is discovered in.
        command += ["--tmp-dir", str(out / "workspace")]
        if tta:
            command += ["--tta", str(tta)]
        if prediction_file:
            command += ["--prediction-file", prediction_file]
        # Preset-parameter tuning: this pass's share of --set (see _preset_overrides), for `konfai-apps infer`.
        for override in config_overrides or []:
            command += ["--set", override]
        self._applied[label or preset] = list(config_overrides or [])
        if max_voxels:
            command += ["--max-voxels", str(max_voxels)]
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
        # The engines' temporary files (elastix's images, FireANTs' total field) go to the work dir too: a
        # RAM-backed /tmp would hold several gigabytes a tile of a large pair in memory.
        try:
            subprocess.run(command, check=True, env={**os.environ, "TMPDIR": str(work)})  # nosec B603
        except subprocess.CalledProcessError as error:
            if error.returncode == EXIT_OUT_OF_MEMORY:
                raise _OutOfMemory(
                    "ImpactReg", f"the preset '{preset}' ran out of GPU memory.", f"Its logs are kept in {out}."
                ) from None
            # One message, not the argv of a CalledProcessError after the traceback the preset printed itself.
            raise KonfAIError(
                "ImpactReg",
                f"the preset '{preset}' failed (exit {error.returncode}); its own error is printed above.",
                f"Its configuration and logs are kept in {out}.",
            ) from None
        # THE PRESET'S CONTRACT IS THE FIELD, AND ONLY THE FIELD. A registration app produces a
        # displacement field on the fixed grid, in whatever format it declares; anything else that can
        # be computed from it: the moved image above all: is this layer's job. Looking for a Moved
        # here would make every preset carry an output it does not owe, and a tiled one blend it across
        # every patch seam for a caller that has the field.
        group = _find_output_group(out / "predictions")
        return group, _find_outputs(out / "predictions", group)

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
        max_voxels: int | None = None,
    ) -> None:
        """Register every case with the selected presets and ensemble their DVFs.

        A case is whatever konfai-apps makes of the inputs: one image per group gives one case, and a
        directory per group gives one case per volume inside it, paired by position. So the caller may
        pass single volumes or whole datasets, exactly as ``konfai-apps infer`` accepts them.

        Masks are optional and restrict the metric region; when omitted a whole-image mask is auto-filled,
        so every preset app always receives the four inputs (fixed, moving, fixed mask, moving mask) it declares.

        ``tmp_dir`` names where the intermediates are staged, beside ``output`` when left unset; see
        :func:`_work_dir`.

        A pair too large for the device is never cut in the preset's own config: KonfAI resamples it onto a coarser
        grid and brings the field back onto the native fixed grid (its patch ``mode: resample``). A preset that
        declares a tile pass (``tiling["tile"]`` in its ``app.json``) then refines that at native resolution: its
        deformable stage on native tiles of the pair pre-warped through the first result, cut by KonfAI (``mode:
        tile``); the two fields are composed on the native fixed grid (see ``_register_tiled``). The sizes are what
        the device holds when the run starts at the peak costs the preset declares (``_plan``); ``max_voxels``
        overrides the size registered whole, the tiles following in proportion.

        ``fields_only`` writes the transforms and stops there. The moved image is derived FROM the
        transform, at the cost of a full-size resample and a full-size rewrite: worth it for a
        caller that wants a registration to look at, waste for one that composes the field with
        another and derives its own. A caller that reads only the fields should be able to say so
        rather than pay for an output it deletes.
        """
        # The cases are konfai-apps' to define, not ours to count. It expands each input GROUP into
        # units: a file, a store, a DICOM series, or every volume inside a plain directory, and pairs
        # the groups by position. Asking it for the moving group's units is what tells this layer which
        # volume belongs to which case, in the same order it will use, so a dataset in and a single pair
        # in go down one path. The presets are handed those units rather than the directories: a
        # directory walked again later would also list the work dir, when it sits beside an output
        # inside an input directory.
        started = time.monotonic()
        moving_units = _units(moving_images)
        fixed_units = _units(fixed_images)
        inputs = {"Fixed": fixed_units, "Moving": moving_units}
        if fixed_masks:
            inputs["FixedMask"] = _units(list(fixed_masks))
        if moving_masks:
            inputs["MovingMask"] = _units(list(moving_masks))
        _check_inputs(inputs, output)
        _check_presets(presets)
        overrides = _preset_overrides(presets, config_overrides, quiet)
        self._applied, self._plans = {}, {}

        work = _work_dir(tmp_dir, output, "impact_reg_")
        try:
            if not quiet:
                print(f"[ImpactReg] intermediates in {work}", flush=True)
            headers = self._headers(inputs, work)
            _check_volumes(inputs, headers)
            _check_space(work, headers, len(presets))
            on_grid = None
            fields_by_preset = {}
            for index, preset in enumerate(presets, 1):
                if len(presets) > 1 and not quiet:
                    print(f"[ImpactReg] preset {index}/{len(presets)}: {preset}", flush=True)
                manifest = _manifest(preset)
                tiling = manifest.get("tiling") or {}
                budget, tile_budget = _plan(manifest, gpu, max_voxels)
                self._plans[preset] = {"whole_voxels": budget, "tile_voxels": tile_budget}
                largest_voxels = max(math.prod(case[group][0][-3:]) for case in headers for group in case)
                tiled = bool(tiling.get("tile") and budget and largest_voxels > budget)
                # KonfAI patches each image on its own grid, and a test-time flip turns each about its own centre:
                # the pair goes onto the fixed grid where it would be tiled as it came (tiles with no global pass
                # before them) and wherever it is flipped.
                onto_fixed = bool(tta or (tiled and not tiling.get("global")))
                if onto_fixed:
                    on_grid = on_grid or self._onto_fixed_grid(inputs, headers, work, gpu, cpu, quiet)
                pair = on_grid if onto_fixed else (inputs, headers)
                if tiled:
                    fields_by_preset[preset] = self._register_tiled(
                        preset,
                        tiling,
                        *pair,
                        budget,
                        tile_budget,
                        work,
                        gpu,
                        cpu,
                        quiet,
                        tta,
                        overrides[preset],
                    )
                    continue
                # Registered whole: KonfAI resamples a pair over the budget onto a coarser grid and its field back
                # onto the native fixed grid (the preset's patch ``mode: resample``).
                fields_by_preset[preset] = self._infer_preset(
                    preset,
                    *(pair[0].get(group, []) for group in ("Fixed", "Moving", "FixedMask", "MovingMask")),
                    work,
                    gpu,
                    cpu,
                    quiet,
                    tta,
                    overrides[preset]["global"],
                    max_voxels=budget,
                )
            groups = {group for group, _ in fields_by_preset.values()}
            if len(groups) != 1:
                raise RuntimeError(
                    f"the presets named their output differently ({', '.join(sorted(groups))}); an"
                    " ensemble folds one group, so every member must declare the same one."
                )
            group = groups.pop()
            if group == "Moved" and not fields_only:
                raise RuntimeError(
                    "the preset names its output 'Moved', the name this pipeline writes the derived"
                    " image under; the two would collide in the case directory. Rename the preset's"
                    " output, or pass fields_only."
                )
            fields_by_preset = {preset: fields for preset, (_, fields) in fields_by_preset.items()}
            cases = sorted(fields_by_preset[presets[0]], key=_case_key)
            for preset, fields in fields_by_preset.items():
                if sorted(fields, key=_case_key) != cases:
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
                        dvf = _copy_output(dvf, case_out / _ENSEMBLE_DIR, preset, move=True)
                    dvf_paths.append(dvf)

                if len(presets) == 1:
                    # Kept under Ensemble/ as well when asked; otherwise it is the workspace's, deleted next.
                    _copy_output(dvf_paths[0], case_out, group, move=not keep_dvf)
                else:
                    # Ensemble: fold the presets' fields (all on the fixed grid, and Reduce VERIFIES
                    # the claim) into the averaged DVF, the one output no single preset produced.
                    # Streamed: the fold is incremental, so the peak is one accumulator plus the
                    # member being read, whatever the size of the ensemble.
                    self._ensemble_mean(case, group, presets, dvf_paths, output, work, gpu, cpu, quiet)

            if not fields_only:
                # The moved images in ONE streamed run over the cohort: Resample adopts, per case,
                # the grid of that case's own field (a field is defined ON the fixed grid) and reads
                # the field as the map.
                self._derive_moved(
                    dict(zip(cases, moving_units, strict=True)),
                    group,
                    output,
                    work,
                    gpu,
                    cpu,
                    quiet,
                )
        except BaseException as error:
            _leave(work, error)
            raise
        _leave(work, None)
        run = {
            "presets": presets,
            "repository": IMPACT_REG_KONFAI_REPO,
            "overrides": list(config_overrides or []),
            # What each preset run took of them: a tile pass keeps the stages its config sets (see _preset_overrides).
            "applied": self._applied,
            "plans": self._plans,
            "tta": tta,
            "device": f"cuda:{','.join(map(str, gpu))}" if gpu else f"cpu ({cpu or 1} worker(s))",
            "versions": {package: _version(package) for package in ("impact-reg-konfai", "konfai", "konfai-apps")},
            "runtime_s": round(time.monotonic() - started, 1),
        }
        _record(output, run, inputs, cases, group, fields_only, quiet)

    def _headers(self, inputs: dict[str, list[Path]], work: Path) -> list[dict[str, _Header]]:
        """Per case, every input image's header (``_Header``), by group."""
        headers = []
        for index in range(len(inputs["Fixed"])):
            case = {}
            for group, units in inputs.items():
                shape, attributes = _read_header(work / "headers" / f"{index:03d}", group, units[index])
                case[group] = (
                    shape,
                    *([float(v) for v in attributes.get_np_array(key)] for key in ("Spacing", "Origin", "Direction")),
                )
            headers.append(case)
        return headers

    def _onto_fixed_grid(
        self,
        inputs: dict[str, list[Path]],
        headers: list[dict[str, _Header]],
        work: Path,
        gpu: list[int],
        cpu: int | None,
        quiet: bool,
    ) -> tuple[dict[str, list[Path]], list[dict[str, _Header]]]:
        """``inputs`` with every group that is not on its case's fixed grid resampled onto it, through the identity,
        and the headers that go with them.

        KonfAI cuts every image of a case into patches by voxel index, and a test-time flip turns each image about its
        own centre: both pair the right regions only when the images share one grid, and KonfAI refuses patched
        groups that do not. Onto the fixed grid, since the field is defined there and so still holds for the moving
        as it came. Never unasked: on it the moving loses what lies outside the fixed field of view, and its own
        resolution.
        """
        cases = [f"P{index:03d}" for index in range(len(headers))]
        fixed = dict(zip(cases, inputs["Fixed"], strict=True))
        on_grid, grid_headers = dict(inputs), [dict(case) for case in headers]
        for group in inputs:
            if group == "Fixed" or all(_same_grid(case[group], case["Fixed"]) for case in headers):
                continue
            images = dict(zip(cases, inputs[group], strict=True))
            warped = self._prewarp(group, images, fixed, None, work / "fixed_grid", gpu, cpu, quiet)
            on_grid[group] = [warped[case] for case in cases]
            for case in grid_headers:
                case[group] = case["Fixed"]
        return on_grid, grid_headers

    def _prewarp(
        self,
        group: str,
        images: dict[str, Path],
        fixed: dict[str, Path],
        fields: dict[str, Path] | None,
        work: Path,
        gpu: list[int],
        cpu: int | None,
        quiet: bool,
    ) -> dict[str, Path]:
        """``images`` resampled through the global ``fields`` onto each case's native fixed grid, streamed, in the
        fixed image's own form: the moving side the tiles register, already where the global stages put it. With no
        ``fields``, through the identity: a change of grid alone."""
        from konfai.data.transform import Resample, Write

        base, out = work / f"prewarp_{group}_stage", work / f"prewarp_{group}"
        roots = [_stage_group(base, group, images), _stage_group(base, "Fixed", fixed)]
        if fields is not None:
            roots.append(_stage_group(base, "DVF", fields))
        reference = next(iter(fixed.values()))
        backend, suffix = _backend(reference), _form(reference)
        resample = Resample(
            reference="{case}",
            reference_group="Fixed",
            field_group="DVF" if fields is not None else None,
            interpolation="nearest" if group.endswith("Mask") else "linear",
        )
        _run_transform(
            f"impact_reg_prewarp_{group}",
            roots,
            {group: {group: [resample, Write(dataset=f"{out}:{backend}")]}},
            work,
            gpu,
            cpu,
            quiet,
        )
        return {case: out / case / f"{group}{suffix}" for case in images}

    def _compose(
        self,
        tile_fields: dict[str, Path],
        global_fields: dict[str, Path],
        fixed: dict[str, Path],
        group: str,
        padding: list[int],
        work: Path,
        gpu: list[int],
        cpu: int | None,
        quiet: bool,
    ) -> dict[str, Path]:
        """The one field on the native fixed grid: ``D(x) = D_tiles(x) + D_global(x + D_tiles(x))``, streamed.

        The moving the tiles registered was the global stages' output, so a fixed point goes through the tiles'
        displacement first and the global one after, read where the first lands.

        Where the tiles' displacement points past a face of the fixed grid, the global field is read from its edge
        voxels, repeated outwards, not as the resampling's fill of 0. ``padding`` is that edge (``_compose_padding``).
        """
        from konfai.data.transform import Padding, Reduce, Resample, TensorCast, Write

        base, read = work / "compose_stage", work / "compose_global"
        roots = [_stage_group(base, "Global", global_fields), _stage_group(base, "Fixed", fixed)]
        # An intermediate the size of the native grid: a float32 store, compressed, rather than a second float64
        # transform file beside the tiles' own.
        roots.append(_stage_group(base, "DVF", tile_fields))
        chain = [
            Padding(padding=padding, mode="replicate"),
            Resample(reference="{case}", reference_group="Fixed", field_group="DVF"),
            TensorCast(dtype="float32"),
            Write(dataset=f"{read}:omezarr"),
        ]
        _run_transform("impact_reg_compose_global", roots, {"Global": {"Global": chain}}, work, gpu, cpu, quiet)
        global_on_native = {case: read / case / "Global.ome.zarr" for case in global_fields}
        out, composed = work / "composed", {}
        for case in global_fields:
            # One root per form: the two members are read side by side as the cases of one group.
            members = [
                _stage_group(work / f"compose_{case}_tiles", "DVF", {"tiles": tile_fields[case]}),
                _stage_group(work / f"compose_{case}_global", "DVF", {"global": global_on_native[case]}),
            ]
            _run_transform(
                f"impact_reg_compose_{case}",
                members,
                {
                    "DVF": {
                        group: [
                            Reduce(operator="Sum", output=case, grid="strict"),
                            Write(dataset=f"{out}:{_backend(tile_fields[case])}"),
                        ]
                    }
                },
                work,
                gpu,
                cpu,
                quiet,
            )
            composed[case] = out / case / f"{group}{_form(tile_fields[case])}"
        shutil.rmtree(read, ignore_errors=True)
        return composed

    def _register_tiled(
        self,
        preset: str,
        tiling: dict,
        inputs: dict[str, list[Path]],
        headers: list[dict[str, _Header]],
        budget: int,
        tile_budget: int,
        work: Path,
        gpu: list[int],
        cpu: int | None,
        quiet: bool,
        tta: int,
        config_overrides: dict[str, list[str]],
    ) -> tuple[str, dict[str, Path]]:
        """One preset on a pair too large for the GPU, at native resolution; ``config_overrides`` holds the ``--set``
        share of each pass (``_preset_overrides``).

        1. Its global config (``tiling["global"]``, the preset itself: rigid, affine and deformable) runs once on
           the whole pair, which KonfAI resamples onto a grid of at most ``budget`` voxels and whose field it brings
           back onto the native fixed grid (the config's patch ``mode: resample``).
        2. The moving image and mask go through it onto the native fixed grid, streamed; a fixed mask on another
           grid goes onto it through the identity.
        3. Its deformable stage alone (``tiling["tile"]``) refines that on native tiles, which KonfAI cuts to at most
           ``tile_budget`` voxels (``mode: tile``) and blends over their overlap. A tile
           the fixed mask does not reach gets a zero field without running.
        4. The two are composed into one field on the native fixed grid.

        An out-of-memory in either pass is KonfAI's to answer: it coarsens or cuts further and runs again.
        """
        cases = [f"P{index:03d}" for index in range(len(inputs["Fixed"]))]
        fixed = dict(zip(cases, inputs["Fixed"], strict=True))
        if not quiet:
            shape = max((case["Fixed"][0][-3:] for case in headers), key=math.prod)
            tiles = f"native tiles of at most {tile_budget:,} voxels"
            plan = (
                f"the preset on {tiles}"
                if not tiling.get("global")
                else f"the preset on the whole pair resampled to fit, then its deformable stage on {tiles}"
            )
            print(
                f"[ImpactReg] {preset}: {' x '.join(map(str, shape))} voxels, more than the {budget:,} it registers whole"
                f" {'on this GPU' if gpu else 'on the CPU'}: {plan}.",
                flush=True,
            )
        global_fields = None
        if tiling.get("global"):
            _, global_fields = self._infer_preset(
                preset,
                *(inputs.get(group, []) for group in ("Fixed", "Moving", "FixedMask", "MovingMask")),
                work,
                gpu,
                cpu,
                quiet,
                tta,
                config_overrides["global"],
                prediction_file=tiling["global"],
                label=f"{preset}_global",
                max_voxels=budget,
            )
        tiled = dict(inputs)
        if global_fields is not None:
            for side in ("Moving", "MovingMask"):
                if side in inputs:
                    warped = self._prewarp(
                        side,
                        dict(zip(cases, inputs[side], strict=True)),
                        fixed,
                        global_fields,
                        work / f"{preset}_prewarp",
                        gpu,
                        cpu,
                        quiet,
                    )
                    tiled[side] = [warped[case] for case in cases]
        # KonfAI cuts every group of a case at the same voxels: a fixed mask on another grid than the fixed image
        # goes onto it, through the identity, as the moving side went through the global field.
        if "FixedMask" in inputs and not all(_same_grid(case["FixedMask"], case["Fixed"]) for case in headers):
            warped = self._prewarp(
                "FixedMask",
                dict(zip(cases, inputs["FixedMask"], strict=True)),
                fixed,
                None,
                work / f"{preset}_prewarp",
                gpu,
                cpu,
                quiet,
            )
            tiled["FixedMask"] = [warped[case] for case in cases]
        group, tile_fields = self._infer_preset(
            preset,
            *(tiled.get(group, []) for group in ("Fixed", "Moving", "FixedMask", "MovingMask")),
            work,
            gpu,
            cpu,
            quiet,
            tta,
            config_overrides["tile"],
            prediction_file=tiling["tile"],
            label=f"{preset}_tiles",
            max_voxels=tile_budget,
        )
        if global_fields is None:
            return group, tile_fields
        # Volume-sized intermediates go as soon as nothing reads them: the pre-warped images before the composition,
        # the tiles' field after it.
        shutil.rmtree(work / f"{preset}_prewarp", ignore_errors=True)
        reach = max(field_reach(*_field(field, work / f"{preset}_reach")) for field in tile_fields.values())
        padding = _compose_padding(headers, budget, reach)
        composed = self._compose(
            tile_fields, global_fields, fixed, group, padding, work / f"{preset}_compose", gpu, cpu, quiet
        )
        shutil.rmtree(work / f"{preset}_tiles", ignore_errors=True)
        return group, composed

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
        """Average one case's preset fields into ``<output>/<case>/<group>``: Reduce(Mean), streamed."""
        from konfai.data.transform import Reduce, Write

        members = _stage_group(work / f"ensemble_{case}", "DVF", dict(zip(presets, dvf_paths, strict=True)))
        suffixes, backend = _form(dvf_paths[0]), _backend(dvf_paths[0])
        _run_transform(
            f"impact_reg_ensemble_{case}",
            [members],
            {
                "DVF": {
                    group: [
                        Reduce(operator="Mean", output=case, grid="strict"),
                        Write(dataset=f"{output}:{backend}"),
                    ]
                }
            },
            work,
            gpu,
            cpu,
            quiet,
        )
        _drop_other_forms(output / case, group, suffixes)

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
        """The moved images, derived from the transforms, one streamed run over the whole cohort.

        A preset that emits only a field is complete: everything else IS that field. The moved
        image: ``reference: '{case}'`` adopts each case's own DVF grid (a field is defined ON the
        fixed grid) and the field is the map (``field_group``): one interpolation, streamed, each
        slab's source window sized from the field values it reads.

        NOTHING ELSE IS DERIVED. A preset writes its transform in the form its consumer reads (an ITK
        transform file where Slicer picks it up, an RFC-5 store where this pipeline streams it) so
        there is no second copy of the same field to produce under another name.

        A moved OME-Zarr store is written as a pyramid (KonfAI's ``scale_factors: auto``): its moving came as one,
        and a single level made a viewer load the full resolution to show it at all.
        """
        from konfai.data.transform import Resample, Write

        fields = {case: _the_output(output / case, group) for case in cases}
        # The moved image is written in the MOVING's own form: the fields' form may be an ITK
        # transform file, which cannot hold an image. _stage_group refuses a mixed Moving group
        # below, so the first case's form speaks for the cohort.
        moving = next(iter(cases.values()))
        suffixes, backend = _form(moving), _backend(moving)
        moving_root = _stage_group(work / "moved_stage", "Moving", cases)
        field_root = _stage_group(work / "moved_stage", "DVF", fields)
        _run_transform(
            "impact_reg_moved",
            [moving_root, field_root],
            {
                "Moving": {
                    "Moved": [
                        Resample(reference="{case}", reference_group="DVF", field_group="DVF", interpolation="linear"),
                        Write(
                            dataset=f"{output}:{backend}",
                            scale_factors="auto" if backend == "omezarr" else None,
                        ),
                    ]
                },
            },
            work,
            gpu,
            cpu,
            quiet,
        )
        for case in fields:
            _drop_other_forms(output / case, "Moved", suffixes)

    # ------------------------------------------------------------------ evaluate

    def evaluate(
        self,
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
        already registered and only resampled onto the fixed grid (identity). Each option holds one entry
        per case or one used for every case (``_cases``); a displacement-field transform also gets its
        Jacobian statistics. Case ``P###`` is written to ``<output>/P###/Evaluation/<modality>/`` (Image,
        Segmentation, Landmarks, Jacobian), and the whole call to ``<output>/Evaluation_summary.json``.

        The evaluation configs ship with this package (``_EVALUATION_APP``): every preset carried the same
        three, and resolving one only to read them pip-installed its requirements (itk-impact for the
        default preset) and needed the network.
        """
        app = KonfAIApp(str(_EVALUATION_APP), False, False)
        # Every group is expanded the same way ``register`` expands its inputs, so a directory of
        # volumes evaluates case by case instead of collapsing to its first entry. Transforms and
        # landmark files expand too: .h5, .fcsv and .itk.txt are all supported extensions. A transform
        # FILE is taken as it is: ITK's text .tfm is not an extension konfai's listing knows. register's
        # output folder is not a list of transforms: beside each case's transform it holds the moved image
        # and, with --keep-fields, every preset's field under Ensemble/, each of which would be scored as
        # a case of its own. An ensemble member named itself, or through its Ensemble folder, is a transform.
        expanded = [(path, unit) for path in transforms for unit in ([path] if path.is_file() else _units([path]))]
        strays = [
            unit
            for path, unit in expanded
            if _is_entry(unit, "Moved") or (unit.parent.name == _ENSEMBLE_DIR and path not in (unit, unit.parent))
        ]
        if strays:
            raise EvaluatorError(
                f"--transform takes in '{strays[0]}', register's moved image or ensemble member, not a case's transform.",
                "Name the transforms themselves, e.g. <output>/P*/Transform.h5, not register's output folder.",
            )
        transforms = [unit for _, unit in expanded]
        fixed_images, moving_images, transforms, mask, gt_fixed_seg, gt_moving_seg, gt_fixed_fid, gt_moving_fid = (
            _cases(
                {
                    "-f": _units(fixed_images),
                    "-m": _units(moving_images),
                    "--transform": transforms,
                    "--mask": _units(mask) if mask else [],
                    "--gt-fixed-seg": _units(gt_fixed_seg),
                    "--gt-moving-seg": _units(gt_moving_seg),
                    "--gt-fixed-fid": _units(gt_fixed_fid),
                    "--gt-moving-fid": _units(gt_moving_fid),
                }
            )
        )
        n_cases = max(len(fixed_images), len(gt_fixed_seg), len(gt_fixed_fid))
        scored: list[tuple[str, Path]] = []
        jacobians: dict[Path, dict[str, float]] = {}  # one transform given for every case is measured once
        for index in range(n_cases):
            transform_path = transforms[index] if transforms else None
            # One output folder AND one workspace per modality (Image, Segmentation, Landmarks): the three
            # configs share one train_name and konfai-apps evaluates with overwrite.
            eval_out = output / f"P{index:03d}" / "Evaluation"
            work = _work_dir(tmp_dir, output, "impact_reg_eval_")
            try:
                # Image: moving resampled onto the fixed grid vs fixed (MAE). Mask is optional.
                if fixed_images:
                    moved = self._warp_onto_fixed(
                        work, "img", fixed_images[index], moving_images[index], transform_path, gpu, cpu, quiet
                    )
                    app.evaluate(
                        inputs=[[fixed_images[index]]],
                        gt=[[moved]],
                        output=eval_out / "Image",
                        mask=[[mask[index]]] if mask else None,
                        evaluation_file="Evaluation_with_images.yml",
                        gpu=gpu,
                        cpu=cpu,
                        quiet=quiet,
                        tmp_dir=work / "Image",
                    )
                    scored.append((f"P{index:03d}", eval_out / "Image"))

                # Segmentation: moving seg warped onto fixed vs fixed seg (Dice), nearest-neighbour.
                if gt_fixed_seg:
                    moved_seg = self._warp_onto_fixed(
                        work, "seg", gt_fixed_seg[index], gt_moving_seg[index], transform_path, gpu, cpu, quiet
                    )
                    app.evaluate(
                        inputs=[[gt_fixed_seg[index]]],
                        gt=[[moved_seg]],
                        output=eval_out / "Segmentation",
                        evaluation_file="Evaluation_with_seg.yml",
                        gpu=gpu,
                        cpu=cpu,
                        quiet=quiet,
                        tmp_dir=work / "Segmentation",
                    )
                    scored.append((f"P{index:03d}", eval_out / "Segmentation"))

                # Landmarks (TRE): the transform is defined on the fixed grid and maps fixed->moving, so the
                # fixed fiducials are displaced by it into moving space and compared against the moving fiducials
                # there (the standard warped-keypoints convention; no field inversion needed). With no transform
                # the raw fiducials are compared, measuring the initial misalignment. Landmarks are a few
                # points: a field is read around them alone (``_displace``). Points pair by row,
                # so both files must hold as many; the warped fixed points are the config's FixedFid group.
                if gt_fixed_fid:
                    fixed_points = read_landmarks(gt_fixed_fid[index])
                    moving_count = len(read_landmarks(gt_moving_fid[index]))
                    if len(fixed_points) != moving_count:
                        raise EvaluatorError(
                            f"{len(fixed_points)} fixed vs {moving_count} moving landmarks"
                            f" ('{gt_fixed_fid[index]}', '{gt_moving_fid[index]}').",
                            "Landmarks pair by row order: give both files the same points, in the same order.",
                        )
                    if transform_path is not None:
                        fixed_points = _displace(fixed_points, transform_path, work)
                    moved_fid = work / "moved_fid.fcsv"
                    write_landmarks(fixed_points, moved_fid)
                    app.evaluate(
                        inputs=[[moved_fid]],
                        gt=[[gt_moving_fid[index]]],
                        output=eval_out / "Landmarks",
                        evaluation_file="Evaluation_with_fid.yml",
                        gpu=gpu,
                        cpu=cpu,
                        quiet=quiet,
                        tmp_dir=work / "Landmarks",
                    )
                    scored.append((f"P{index:03d}", eval_out / "Landmarks"))

                # Deformation quality, a property of the transform alone. Only a displacement field is
                # measured: a linear transform's Jacobian is its matrix, the same everywhere, and a B-spline file is
                # not converted to a field.
                if transform_path is not None and _is_field(transform_path):
                    from konfai.evaluator import Statistics

                    if transform_path not in jacobians:
                        jacobians[transform_path] = _jacobian_statistics(transform_path, work)
                    jacobian = Statistics(eval_out / "Jacobian" / "ImpactReg" / "Metric_TRAIN.json")
                    jacobian.filename.parent.mkdir(parents=True, exist_ok=True)
                    jacobian.add(jacobians[transform_path], "P000")
                    jacobian.directions = {
                        key: "max" if key.endswith(":min") else "min" for key in jacobian.measures["P000"]
                    }
                    jacobian.write([jacobian.measures])
                    scored.append((f"P{index:03d}", eval_out / "Jacobian"))
            except BaseException as error:
                _leave(work, error)
                raise
            _leave(work, None)
        _write_evaluation_summary(output / "Evaluation_summary.json", scored)

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
        keep_form: bool = False,
    ) -> Path:
        """The moving warped onto the fixed grid, one streamed Resample; nearest for a ``seg``.

        ``reference: '{case}'`` adopts the fixed grid; the transform, when given, is staged as a
        stored-transform group konfai decodes itself (rigid, affine, spline, field or composite): with its
        exact affine box or its coefficient-derived bound sizing what each slab reads.
        With no transform the map is the identity and this is a change of grid alone.
        Written as MHA, or with ``keep_form`` in the moving's own form (file, OME-Zarr store, DICOM series).
        """
        from konfai.data.transform import Resample, Write

        base = work / f"warp_{kind}"
        datasets = [
            _stage_group(base, "Fixed", {"P000": fixed}),
            _stage_group(base, "Moving", {"P000": moving}),
        ]
        resample: dict[str, object] = {
            "reference": "{case}",
            "reference_group": "Fixed",
            "interpolation": "nearest" if kind == "seg" else "linear",
        }
        if transform_path is not None:
            if _is_transform_file(transform_path):
                resample["transforms"] = {"Reg": False}
            else:
                resample["field_group"] = "Reg"
            datasets.append(_stage_group(base, "Reg", {"P000": transform_path}))
        out_root = work / f"moved_{kind}"
        backend = _backend(moving) if keep_form else "mha"
        _run_transform(
            f"impact_reg_eval_{kind}",
            datasets,
            {"Moving": {"Moved": [Resample(**resample), Write(dataset=f"{out_root}:{backend}")]}},
            work,
            gpu,
            cpu,
            quiet,
        )
        (moved,) = (out_root / "P000").iterdir()
        return moved

    # --------------------------------------------------------------- uncertainty

    def uncertainty(
        self,
        dvfs: list[Path],
        output: Path = Path("./Output").resolve(),
        gpu: list[int] = [],
        cpu: int | None = None,
        quiet: bool = False,
        tmp_dir: Path | None = None,
    ) -> None:
        """Estimate registration uncertainty as the voxel-wise spread of an ensemble of displacement fields.

        The spread of the displacement VECTORS, in mm: the standard deviation of each component across
        the members, then the norm of those three, ``sqrt(s_x^2 + s_y^2 + s_z^2)``. That is the root mean
        square distance of the members to their mean vector (with the N-1 sample correction; two members
        give ``|u1 - u2| / sqrt(2)``), and it does not depend on the axes the field is expressed in.

        The order is the point: the standard deviation of the magnitudes would read two members that agree in
        length but not in direction as certain. ``Reduce(Std)`` folds with running moments and ``Magnitude`` is pointwise on its result, so no member is ever
        whole in RAM whatever the size of the ensemble. Writes ``<output>/uncertainty/Uncertainty.<ext>``.
        """
        if len(dvfs) < 2:
            raise ValueError("Uncertainty needs at least two ensemble displacement fields.")
        work = _work_dir(tmp_dir, output, "impact_reg_unc_")
        try:
            from konfai.data.transform import Magnitude, Reduce, Write

            members = _units(list(dvfs))
            spec = _stage_group(work / "members", "DVF", {f"M{index:03d}": dvf for index, dvf in enumerate(members)})
            suffixes = ".mha" if _is_transform_file(members[0]) else _form(members[0])
            _run_transform(
                "impact_reg_uncertainty",
                [spec],
                {
                    "DVF": {
                        "Uncertainty": [
                            Reduce(operator="Std", output="uncertainty", grid="strict"),
                            Magnitude(),
                            Write(dataset=f"{output}:{_format_token(suffixes)}"),
                        ]
                    }
                },
                work,
                gpu,
                cpu,
                quiet,
            )
            _drop_other_forms(output / "uncertainty", "Uncertainty", suffixes)
        except BaseException as error:
            _leave(work, error)
            raise
        _leave(work, None)

    # --------------------------------------------------------------- apply

    def apply(
        self,
        transform: Path,
        fixed: Path,
        images: list[Path],
        output: Path = Path("./Output").resolve(),
        labels: bool = False,
        gpu: list[int] = [],
        cpu: int | None = None,
        quiet: bool = False,
        tmp_dir: Path | None = None,
    ) -> list[Path]:
        """Warp more images of the moving side onto the fixed grid, through a transform ``register`` wrote.

        ``register`` derives the moved image of the moving image only: this warps a segmentation of it, another
        sequence of the same subject or a mask the same way. Each image here takes
        the resample ``eval`` uses, one streamed pass onto ``fixed``'s grid, nearest-neighbour with ``labels``,
        and is written as ``<output>/<its name>``, in its own form as ``register`` writes ``Moved`` (a store stays
        a store, a DICOM series a series), replacing an earlier output of that name only. An output that would
        replace an input, or two images of one name, is refused before anything runs. The transform maps fixed
        points to moving points: it brings images from the moving side, and no inverse is computed for the other
        direction.
        """
        units = _units(list(images))
        inputs = {path.resolve() for path in [transform, fixed, *units]}
        names = [unit.name for unit in units]
        for name in names:
            if (output / name).resolve() in inputs:
                raise KonfAIError(
                    "Apply", f"'{output / name}' is an input: its output would replace it.", "Name another -o."
                )
            if names.count(name) > 1:
                raise KonfAIError("Apply", f"Two images are named '{name}'.", "Apply them into different -o.")
        work = _work_dir(tmp_dir, output, "impact_reg_apply_")
        written = []
        try:
            for index, image in enumerate(units):
                kind = "seg" if labels else "image"
                moved = self._warp_onto_fixed(
                    work / f"{index:03d}", kind, fixed, image, transform, gpu, cpu, quiet, keep_form=True
                )
                dest = output / image.name
                output.mkdir(parents=True, exist_ok=True)
                if dest.is_dir():  # an earlier store or series of this name; a file is replaced by the move
                    shutil.rmtree(dest)
                shutil.move(moved, dest)
                written.append(dest)
        except BaseException as error:
            _leave(work, error)
            raise
        _leave(work, None)
        return written
