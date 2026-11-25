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

import dataclasses
import itertools
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import SimpleITK as sitk
from konfai import RemoteServer
from konfai.utils.dataset import (
    DENSE_TRANSFORMS_ENV,
    Attribute,
    Dataset,
    entry_files,
    is_staging_entry,
    read_landmarks,
    staged_entry,
    transfer_entry,
)
from konfai.utils.errors import EXIT_OUT_OF_MEMORY, EvaluatorError, KonfAIError
from konfai.utils.ITK import field_reach
from konfai.utils.runtime import apply_cpu_thread_budget
from konfai.utils.utils import (
    SUPPORTED_BACKEND_FORMATS,
    SUPPORTED_FORMATS,
    format_token,
    path_format_token,
    split_format_level,
    storage_form,
)
from konfai_apps import KonfAIApp
from konfai_apps.app_repository import (
    AppRepositoryError,
    LocalAppRepository,
    apply_overrides,
    check_overrides,
    did_you_mean,
    pass_cost,
)
from konfai_apps.cli import app_file, app_id, app_names
from konfai_apps.ensemble import STACK, combine, stack_file
from konfai_apps.ensemble import _token as member_token
from konfai_apps.presets import merge_presets, read_presets, split_preset, user_presets_file

from impact_reg_konfai import PRESETS_REPO, ROUTES
from impact_reg_konfai.plane import is_flat, lift_field, lift_plane, project_field, project_image

# Preset apps live on this Hugging Face repo, at the revision this release pins; KONFAI_IMPACTREG_REPO points
# it at a local directory of preset folders (each with an app.json) for development / offline use.
IMPACT_REG_KONFAI_REPO = PRESETS_REPO

if TYPE_CHECKING:
    from .native import AxisMap


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
    """The storage form of ``path``: konfai's, which knows a dot in a stem is part of the name. An OME-Zarr store whose
    name carries no suffix (an HCS well's image, ``plate.zarr/C/02/1``) is ``.ome.zarr``, the form it is written back
    in: a bare form would name the moved image another output than the one its writer produces."""
    form = storage_form(path)
    if not form and path.is_dir() and path_format_token(path) == "omezarr":
        return ".ome.zarr"
    return form


def _asked_form(kind: str) -> tuple[str, str]:
    """The suffix an output asked in the format ``kind`` (``--output-format``) is published under and the token it
    is written through: a backend token names its file (``itktransform`` an ``.h5``, ``mha.gz`` a compressed
    ``.mha``), an OME-Zarr store is ``.ome.zarr``."""
    if kind in SUPPORTED_BACKEND_FORMATS:
        return (".h5" if kind == "itktransform" else f".{kind.removesuffix('.gz')}"), kind
    token = format_token(f".{kind}")
    return (".ome.zarr" if token == "omezarr" else f".{kind}"), token


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


def _is_flat_field(path: Path) -> bool:
    """Whether a stored registration, a transform file or an OME-Zarr store, is 2-D, read off its header alone."""
    if _backend(path) == "omezarr":
        from konfai.utils.ome_zarr import get_ome_zarr_info, is_affine_store, read_ome_zarr_affine

        if is_affine_store(path):
            return read_ome_zarr_affine(path).GetDimension() == 2
        return "z" not in get_ome_zarr_info(path)["axes"]
    return _is_transform_file(path) and _transform_kinds(path)[0].endswith("_2_2")


def _is_field(path: Path) -> bool:
    """Whether a stored registration is a displacement field: an image, a field store or a field transform file. A
    linear transform file, an RFC-5 affine store or a B-spline (a preset's native form) is not one."""
    if _is_transform_file(path):
        return _transform_kinds(path)[0].startswith("DisplacementFieldTransform")
    from konfai.utils.ome_zarr import is_affine_store

    return not (_backend(path) == "omezarr" and is_affine_store(path))


def _is_bspline(path: Path) -> bool:
    """Whether a stored registration is a transform file holding a B-spline (elastix's native form)."""
    return _is_transform_file(path) and any("BSpline" in kind for kind in _transform_kinds(path))


def _field(path: Path, work: Path) -> tuple[Dataset, str, str]:
    """A stored displacement field staged as a dataset entry, for KonfAI's field readers."""
    if _is_flat_field(path):  # See plane.py.
        path = lift_field(path, work / "plane_field")
    root, _, backend = _stage_group(work / "field", "Reg", {"P000": path}).rpartition(":")
    return Dataset(root, backend), "Reg", "P000"


def _compose_padding(headers: "list[dict[str, _Header]]", budget: int, reach: float) -> list[int]:
    """The edge, in fixed voxels and ``F.pad`` order (x, y, z pairs), the global field is extended by before the tiles'
    displacement reads it: at least 8 voxels of the grid the global pass ran on (``coarse_spacing``, KonfAI's
    ``mode: resample``) and past the longest tile displacement, ``reach``, by one of them. The widest over the cases."""
    from konfai.data.transform.resample import coarse_spacing

    widest = [0, 0, 0]
    for case in headers:
        shape, spacing = case["Fixed"][0][1:], case["Fixed"][1]
        coarse = coarse_spacing(list(shape), list(spacing), budget) or spacing
        for axis, (step, goal) in enumerate(zip(spacing, coarse, strict=True)):
            widest[axis] = max(widest[axis], math.ceil(max(8 * goal, reach + goal) / step))
    return [extent for extent in widest for _ in range(2)]


def _identity(work: Path, fixed: list[Path]) -> Path:
    """The identity map as a transform file, for an evaluation given no transform: 2-D for a 2-D fixed image."""
    flat = bool(fixed) and is_flat(_read_header(work / "identity", "Fixed", fixed[0])[0])
    path = work / "Identity.itk.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteTransform(sitk.AffineTransform(2 if flat else 3), str(path))
    return path


def _case_key(name: str) -> tuple[int, str]:
    """konfai-apps numbers cases ``P000``..: zero-padded to three digits, longer past ``P999`` --
    so length-then-lexicographic IS its numeric order, without parsing a preset's own naming."""
    return (len(name), name)


def _manifest(preset: str) -> dict:
    """The preset's ``app.json`` (its app's, for ``APP#PRESET``), from the local preset directory or the Hugging Face
    repo, with the user's own presets of a Hugging Face app (konfai-apps runs them alike); empty if unreadable.

    Read cache first, as the preset run resolves its own files: an online read moved the cache to the newest
    preset, which only the runs that read a manifest then picked up.
    """
    app, name = split_preset(preset)
    try:
        manifest = json.loads(app_file(IMPACT_REG_KONFAI_REPO, app, "app.json").read_text(encoding="utf-8"))
    except Exception:  # an unreadable manifest declares nothing; resolving the preset reports a bad name itself
        return {}
    presets = read_presets(manifest.get("presets"), f"the app.json of '{app}'")
    if not Path(IMPACT_REG_KONFAI_REPO).expanduser().is_dir():  # a Hugging Face app: the user's own presets too
        own_file = user_presets_file(IMPACT_REG_KONFAI_REPO.partition("@")[0], app)
        presets = merge_presets(presets, own_file, app)  # a clash is refused as konfai-apps refuses it
        manifest["presets"] = {key: entry.to_json() for key, entry in presets.items()}
    chosen = presets.get(name) if name else None
    if chosen is not None and not chosen.tiles:  # a preset that never runs in tiles: its app's tile pass is not its
        manifest["tiling"] = {key: value for key, value in (manifest.get("tiling") or {}).items() if key == "global"}
    return manifest


def _preset_config(preset: str, filename: str) -> dict:
    """One of the preset's prediction configs, as data."""
    from ruamel.yaml import YAML

    app = split_preset(preset)[0]
    return YAML(typ="safe").load(app_file(IMPACT_REG_KONFAI_REPO, app, filename).read_text(encoding="utf-8")) or {}


def _draws_with_seed(config: dict) -> bool:
    """Whether the engine of a preset's ``config`` draws with a seed of its own: a ``seed`` the config sets, or the
    one every engine of this package takes, left to its default."""
    network = ((config.get("Predictor") or {}).get("Model") or {}).get("RegistrationNet")
    return isinstance(network, dict) and ("seed" in network or _config_engine(config) is not None)


def _app_preset_sets(preset: str) -> list[str]:
    """The ``--set`` of the app preset ``APP#PRESET`` names (none for a bare app): they go first in its share, split
    between its passes as any ``--set`` is, so the app is then run bare and nothing applies them twice."""
    app, name = split_preset(preset)
    if name is None:
        return []
    presets = read_presets(_manifest(app).get("presets"), f"the app.json of '{app}'")
    if name not in presets:
        raise KonfAIError(
            "ImpactReg",
            f"the app '{app}' has no preset '{name}'.{did_you_mean(name, presets)}",
            f"Its presets: {', '.join(presets)}." if presets else "It declares none.",
        )
    return list(presets[name].overrides)


#: A key a prediction config leaves out, at its model's default (see ``_agree``).
_UNSET = object()

#: The passes of a preset a ``--set`` scope names: the preset whole or its global pass, and its tile pass.
_PASSES = ("global", "tile")


def _config_value(config: dict, key: str) -> object:
    """What ``--set key=...`` names in a prediction config, resolved as konfai-apps resolves it: a name in the model's
    parameter block (a nested one dotted from there), one that starts at the config root from there. ``_UNSET`` when
    the config has no such key."""
    from_root = "." in key and key.split(".", 1)[0] in config  # Predictor.Dataset...: from the config root
    node = config if from_root else LocalAppRepository._model_param_block(config)[1]
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
    presets: list[str],
    overrides: list[str] | None,
    quiet: bool = False,
    sampled: bool = False,
    seed: int | None = None,
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
    nothing cached) leaves its share to that resolver, in the preset's own process. With ``sampled``, the global share
    is checked against ``Prediction.yml``, which every case of a sampled run reads.
    """
    shares: dict[str, list[tuple[str | None, str]]] = {
        preset: [(None, override) for override in _app_preset_sets(preset)] for preset in presets
    }
    for override in overrides or []:
        preset, stage, rest = _scopes(override, presets)
        for member in [preset] if preset else presets:
            shares[member].append((stage, rest))
    passes: dict[str, dict[str, list[str]]] = {preset: {stage: [] for stage in _PASSES} for preset in presets}
    for preset, share in shares.items():
        if not share and seed is None:
            continue
        tiling = _manifest(preset).get("tiling") or {}
        files = {"global": "Prediction.yml" if sampled else tiling.get("global") or "Prediction.yml"}
        files["tile"] = tiling.get("tile")
        configs = {}
        for stage, filename in files.items():
            try:
                if filename:
                    configs[stage] = _preset_config(preset, filename)
            except Exception:  # offline, nothing cached: konfai-apps checks this share in the preset's own process
                continue
        # --seed: the seed of each pass whose engine draws with one of its own, before any --set of it.
        for stage, config in configs.items():
            if seed is not None and _draws_with_seed(config):
                passes[preset][stage].append(f"seed={seed}")
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
                    f"'impact-reg-konfai show {preset}' lists its parameters. A nested one is dotted from the model,"
                    f" a NAME from Predictor. is a path from the config root, and --set {preset}:NAME=VALUE sets it"
                    " for this preset alone.",
                ) from None
    return passes


def _plan(
    manifest: dict, gpu: list[int], max_voxels: int | None, remote_server: RemoteServer | None = None
) -> tuple[int | None, int | None]:
    """The voxels a preset registers whole and those one of its tiles holds (see ``register``): what the device holds
    at the peak costs its ``app.json`` declares (KonfAI's ``max_voxels``), None for a pass that declares none; or
    ``max_voxels`` for the whole pass, the tiles in proportion to their cost. With ``remote_server``, the device
    is the server's."""
    from konfai.utils.vram import max_voxels as voxels_on

    tiling = manifest.get("tiling") or {}
    whole = pass_cost(manifest, tiling.get("global") or "Prediction.yml")
    tile = pass_cost(manifest, tiling["tile"]) if tiling.get("tile") else None
    if max_voxels:
        device = "vram" if gpu else "ram"
        ratio = whole.get(device, 0) / tile[device] if tile and tile.get(device) else 0
        return max_voxels, int(max_voxels * ratio) if ratio else max_voxels
    at = gpu[0] if gpu else None
    return voxels_on(whole.get("vram"), whole.get("ram"), at, remote_server), voxels_on(
        tile.get("vram"), tile.get("ram"), at, remote_server
    ) if tile else None


@dataclass(frozen=True)
class _Route:
    """Where the cases of one preset register: those of ``sampled`` (by index) sampled, the others whole, or in two
    passes with ``tiled``; ``budget`` and ``tile_budget`` are the preset's plan (``_plan``)."""

    budget: int | None
    tile_budget: int | None
    sampled: list[int]
    tiled: bool


def _route(
    choice: str, preset: str, voxels: list[int], plan: tuple[int | None, int | None], overrides: Sequence[str] = ()
) -> _Route:
    """Each case's route under ``choice`` (``ROUTES``), from ``voxels``, the largest image of each case, and the
    preset's ``plan``. Under auto, a case over the budget goes where the preset can take it: sampled for an engine the
    sampled route drives, else in two passes where the preset declares them, else whole on a grid KonfAI resamples to
    fit. tiled refuses a preset with no tile pass or no declared cost to size it by."""
    budget, tile_budget = plan
    tiles = bool((_manifest(preset).get("tiling") or {}).get("tile")) and bool(budget and tile_budget)
    if choice == "tiled" and not tiles:
        raise KonfAIError(
            "ImpactReg",
            f"{preset} declares no tile pass, or no memory cost to size one by, which --out-of-core tiled needs.",
            "Choose --out-of-core auto, whole or sampled.",
        )
    over = [index for index, size in enumerate(voxels) if budget and size > budget]
    if choice == "sampled":
        return _Route(budget, tile_budget, list(range(len(voxels))), False)
    if choice == "auto" and over and _sampled_engine(preset, overrides):
        return _Route(budget, tile_budget, over, False)
    return _Route(budget, tile_budget, [], choice == "tiled" or (choice == "auto" and bool(over) and tiles))


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


def _stack_beside(output: Path) -> Path | None:
    """The stack a run wrote beside ``output`` (its TTA copies, under -uncertainty), or None."""
    return next((path for path in sorted(output.parent.iterdir()) if _is_entry(path, STACK)), None)


def _is_entry(path: Path, stem: str) -> bool:
    """Whether ``path`` is an entry named ``stem``, in any form: extension, store, or bare name; never a writer's
    staging entry. The bare name is a form of its own: a DICOM series is a directory carrying no extension at all.
    """
    return (path.name == stem or path.name.startswith(f"{stem}.")) and not is_staging_entry(path.name)


def _workspace(name: str) -> str:
    """A folder name for a preset's workspace: ``ConvexAdam#composite`` as ``ConvexAdam_composite``. A ``#`` in a path
    is a URL fragment to zarr, which then wrote an OME-Zarr output into the folder before it; ``/`` and ``:`` of an
    app reference would nest folders and split a dataset spec."""
    return re.sub(r"[^A-Za-z0-9_-]", "_", name)


def _drop_other_forms(dest_dir: Path, stem: str, suffixes: str) -> None:
    """Remove the outputs named ``stem`` in another form than ``<stem><suffixes>``, once that one is published:
    discovery is by stem, so a re-run whose presets emit another form leaves only its own output standing. The
    pixel file of the one kept (a detached header's) stays with it."""
    kept = set(entry_files(dest_dir / (stem + suffixes)))
    for stale in [p for p in dest_dir.iterdir() if _is_entry(p, stem) and p not in kept]:
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


def _some_cases(groups: dict[str, list[Path]], indices: list[int]) -> dict[str, list[Path]]:
    """The units of the cases ``indices``, in every group."""
    return {group: [units[index] for index in indices] for group, units in groups.items()}


def _cases(groups: dict[str, list[Path]]) -> list[list[Path]]:
    """``eval``'s groups, keyed by their option, each as long as the case count (:func:`_broadcast`); each modality
    comes as a pair (a fixed side without its moving one scores nothing)."""
    for fixed, moving in (("-f", "-m"), ("--gt-fixed-seg", "--gt-moving-seg"), ("--gt-fixed-fid", "--gt-moving-fid")):
        if bool(groups[fixed]) != bool(groups[moving]):
            given, missing = (fixed, moving) if groups[fixed] else (moving, fixed)
            raise EvaluatorError(
                f"{given} is given without {missing}.", "A modality is scored from both sides: give both, or neither."
            )
    return _broadcast(groups)


def _broadcast(groups: dict[str, list[Path]]) -> list[list[Path]]:
    """``groups``, keyed by their option, each as long as the case count: the longest group. A group holds one entry
    per case or one used for every case (one atlas against several movings, one mask), or is empty; anything else is
    refused."""
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
        app = split_preset(preset)[0]
        if _manifest(app):
            continue
        try:
            available = app_names(IMPACT_REG_KONFAI_REPO, force_update=True)
        except Exception:
            return
        if app not in available:
            raise KonfAIError(
                "ImpactReg",
                f"there is no preset '{app}' in {IMPACT_REG_KONFAI_REPO}.{did_you_mean(app, available, 3)}",
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
    """Refuse, before any preset runs, an image the presets cannot register: the engines are single-channel, and a
    colour image or a time series failed deep inside one of them."""
    for index, case in enumerate(headers):
        for group, (shape, *_) in case.items():
            if len(shape) not in (3, 4) or shape[0] != 1:
                raise KonfAIError(
                    "ImpactReg",
                    f"the {group} image {inputs[group][index]} has the shape {shape}, channels first.",
                    "The presets register single-channel 2-D or 3-D images: extract the channel or the time point first.",
                )
    if len({is_flat(shape) for case in headers for shape, *_ in case.values()}) > 1:
        raise KonfAIError(
            "ImpactReg",
            "the inputs mix 2-D images and 3-D volumes.",
            "Register them in separate runs: a run's images are all 2-D or all 3-D.",
        )


def _check_time_points(inputs: dict[str, list[Path]]) -> None:
    """Refuse a store of several time points: KonfAI reads the first, so a time series given twice registered its
    first time point onto itself, silently."""
    from konfai.utils.ome_zarr import clear_ome_zarr_cache, get_ome_zarr_info

    for group, units in inputs.items():
        for unit in units:
            if _backend(unit) != "omezarr":
                continue
            clear_ome_zarr_cache(unit)  # a store replaced at this path since an earlier call in this process
            info = get_ome_zarr_info(unit)
            count = info["shape"][info["axes"].index("t")] if "t" in info["axes"] else 1
            if count > 1:
                raise KonfAIError(
                    "ImpactReg",
                    f"the {group} store {unit} has {count} time points, and a preset registers one image.",
                    "Write the time point to a store of its own, e.g. with ngff_zarr: image.data[t] with the dims "
                    "that remain, and register each pair in turn.",
                )


def _read_header(base: Path, group: str, source: Path) -> tuple[list[int], Attribute]:
    """``source``'s shape and attributes, from its header alone, staged as a one-case dataset under ``base``."""
    root, _, backend = _stage_group(base, group, {"P000": source}).rpartition(":")
    shape, attributes = Dataset(root, backend).get_infos(group, "P000")
    return list(shape), attributes


#: The physical sizes of a pair this far apart, on every axis, are more likely a matter of units than of anatomy.
_SIZE_RATIO = 20.0


def _warn_of_other_units(headers: list[dict[str, _Header]], quiet: bool) -> None:
    """Warn of a pair whose physical sizes differ a lot on every axis: a store in micrometres against a file whose
    numbers mean millimetres registers to nothing, silently (NCC 0.14 against 0.93 on the same pair)."""
    if quiet:
        return
    for index, case in enumerate(headers):
        extents = []
        for group in ("Fixed", "Moving"):
            shape, spacing, _, direction = case[group]
            if len(shape) - 1 != len(spacing):
                break  # not an image: _check_volumes says so
            axes = np.reshape(direction, (len(spacing), len(spacing)))
            extents.append(np.abs(axes) @ (np.array(shape[:0:-1]) * spacing))  # along the physical axes
        if len(extents) < 2 or extents[0].size != extents[1].size:
            continue
        fixed, moving = extents
        ratios = np.maximum(fixed / moving, moving / fixed)
        if ratios.min() * (1 + 1e-9) >= _SIZE_RATIO:
            print(
                f"[ImpactReg] P{index:03d}: the fixed and moving images differ x{ratios.min():.3g} or more in physical "
                f"size on every axis ({fixed.tolist()} and {moving.tolist()} mm): check each image's unit and spacing.",
                flush=True,
            )


def _same_grid(first: _Header, second: _Header) -> bool:
    """Whether two images lie on one voxel grid: the same extent, and every voxel of one within a thousandth of a
    voxel of the other's, which leaves the rounding a format stores a header with; read value by value, a spacing
    1e-4 mm off moves the 1000th voxel of a 0.04 mm grid by 2.5 voxels. The two grids are farthest apart at a
    corner."""

    def corners(header: _Header) -> np.ndarray:
        shape, spacing, origin, direction = header
        index = np.array(list(itertools.product(*[(0, extent - 1) for extent in shape[:0:-1]])), dtype=float)
        return np.asarray(origin) + (index * spacing) @ np.reshape(direction, (len(spacing), len(spacing))).T

    return first[0][1:] == second[0][1:] and np.abs(corners(first) - corners(second)).max() <= 1e-3 * min(first[1])


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


def _check_space(work: Path, headers: list[dict[str, _Header]], presets: list[str], sampled: bool) -> None:
    """Warn when the disk under ``work`` holds less than the run may stage there: 24 bytes a fixed voxel for each
    preset's field and for their mean, and 40 more for the moved image and a tiled pass, whose composition holds the
    tiles' field, the global field on the native grid and their sum at once. With ``sampled``, the twins too
    (``_native_twins``): a byte a voxel of every case's masks, on their image's grid, and four a voxel of the
    largest case's images; and for FireANTs the warps of its deformable stage, where they live in files: forward and
    reverse, each with its next step and its two Adam moments (96 bytes a fixed voxel), and an eighth more while a
    scale's generation stands beside the next one's."""
    count = len(presets)
    need = sum(math.prod(case["Fixed"][0][1:]) for case in headers) * (24 * (count + (count > 1)) + 40)
    if sampled:
        sizes = [{group: math.prod(case[group.removesuffix("Mask")][0][1:]) for group in case} for case in headers]
        need += sum(size for case in sizes for group, size in case.items() if group.endswith("Mask"))
        need += max(4 * (case["Fixed"] + case["Moving"]) for case in sizes)
        if any(split_preset(preset)[0] == "FireANTs" for preset in presets):
            need += max(108 * case["Fixed"] for case in sizes)
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


def _publish_output(src: Path, dest: Path, move: bool = False) -> Path:
    """Put the output ``src`` (a file, a store, or a header and the pixel file it names) at ``dest``, staged beside it
    and published once complete (``staged_entry``), so a failed copy or move leaves the previous ``dest`` standing.
    ``move`` moves it instead, for a source in a workspace that is deleted next: a transform on a full-resolution
    grid is tens of gigabytes, and a copy wrote it twice."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.name.lower().endswith(".hdr") and dest.exists():
        # An Analyze pair cannot be replaced whole (staged_entry): the new one is made complete first, the old one goes.
        with tempfile.TemporaryDirectory(dir=dest.parent, prefix=f".{dest.name}.") as scratch:
            ready = Path(scratch) / dest.name
            transfer_entry(src, ready, move)
            for part in entry_files(dest):
                part.unlink()
            return _publish_output(ready, dest, move=True)
    with staged_entry(dest) as staged:
        transfer_entry(src, staged, move)
    if dest.is_dir():
        # A store put at a path already read is invisible to the reader's path-keyed memo, which
        # would otherwise pair the copy's voxels with the replaced store's axes and geometry.
        from konfai.utils.ome_zarr import clear_ome_zarr_cache

        clear_ome_zarr_cache()
    return dest


def _copy_output(src: Path, dest_dir: Path, stem: str, move: bool = False) -> Path:
    """An output beside the results as ``<stem><its form>`` (``_publish_output``), the other forms of ``stem`` retired
    once it stands."""
    dest = _publish_output(src, dest_dir / (stem + _form(src)), move)
    _drop_other_forms(dest_dir, stem, _form(src))
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
        KonfAIApp.link_entry(source.resolve(), case_dir / (group + suffixes.lower()))
    return f"{root}:{_backend(next(iter(entries.values())))}"


def _the_output(dest_dir: Path, stem: str) -> Path:
    """The single output named ``stem`` in ``dest_dir``, in any form (a DICOM series carries no extension): one. The
    pixel file a detached header names is part of that output, not another."""
    matches = sorted(path for path in dest_dir.iterdir() if _is_entry(path, stem))
    companions = {part for match in matches for part in entry_files(match)[1:]}
    matches = [match for match in matches if match not in companions]
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


def _recorded_compacts(output: Path) -> dict[str, list[str]]:
    """The compact transforms the last ``register`` into ``output`` published beside each case's field, by name, as its
    ``register.json`` records them; none without a record."""
    try:
        return json.loads((output / "register.json").read_text(encoding="utf-8")).get("compacts", {})
    except (OSError, ValueError):
        return {}


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


#: The engines whose presets run by the sampled strategy, as the module of ``impact_reg_konfai.models`` a preset's
#: classpath names: the sampled route drives those engines themselves, so a class of another package is refused.
_SAMPLED_ENGINES = ("convexadam", "elastix", "fireants")


def _config_engine(config: dict) -> str | None:
    """The engine of ``_SAMPLED_ENGINES`` a preset's ``config`` names by its classpath; None for any other class."""
    classpath = str(((config.get("Predictor") or {}).get("Model") or {}).get("classpath") or "")
    return next((m for m in _SAMPLED_ENGINES if classpath == f"{__package__}.models.{m}:RegistrationNet"), None)


def _sampled_engine(preset: str, overrides: Sequence[str] = ()) -> str | None:
    """The engine of ``_SAMPLED_ENGINES`` the sampled route drives for ``preset``: the classpath of its config, or the
    one a --set of ``overrides`` gives it; None for any other class."""
    config = _preset_config(preset, "Prediction.yml")
    classpaths = [entry for entry in overrides if entry.partition("=")[0].strip().endswith("classpath")]
    apply_overrides(config, classpaths, lambda: None)
    return _config_engine(config)


@dataclass
class _SampledRun:
    """What one preset's sampled run is given (``_register_sampled``)."""

    preset: str
    config: dict
    inputs: dict[str, list[Path]]
    headers: "list[dict[str, _Header]]"
    work: Path
    gpu: list[int]
    cpu: int | None
    quiet: bool
    region_size: int
    samples: int | None
    epochs: int | None
    output_group: str
    cases: list[str]
    # Each case's view of its fixed and moving image (``native.AxisMap.to_lps``): the twins, and ``headers``, are the
    # pair as a run on the whole pair reorients it.
    maps: "list[dict[str, AxisMap]]"

    @property
    def device_index(self) -> int:
        """The GPU the engine runs on, -1 for the CPU."""
        return self.gpu[0] if self.gpu else -1


def _partial_masks(twins: dict[str, Path], case: str) -> tuple[list[Path | None], int]:
    """The fixed and moving mask twins of ``case`` (``twins``, by group) that restrict their image (some voxels in,
    some out): None for a side that has no mask, or one that restricts nothing; and the voxels the fixed one keeps (0
    when it is None). An empty mask is refused, on either side: it leaves that side no voxel to compare."""
    from .native import nonzero, open_twin

    masks: list[Path | None] = []
    kept = 0
    for group in ("FixedMask", "MovingMask"):
        path = twins.get(group)
        mapped = open_twin(path) if path is not None else None
        inside = nonzero(mapped) if mapped is not None else 0
        if mapped is not None and inside == 0:
            raise KonfAIError(
                "ImpactReg",
                f"{case}: the {group.removesuffix('Mask').lower()} mask holds no voxel, which leaves nothing to"
                " register.",
                "Pass the mask of the region to register, or none.",
            )
        masks.append(path if mapped is not None and inside < math.prod(mapped.array.shape) else None)
        if group == "FixedMask" and masks[0] is not None:
            kept = inside
    return masks, kept


def _viewed(header: _Header, axis_map: "AxisMap") -> _Header:
    """``header`` as ``axis_map`` (``native.AxisMap``) views its grid."""
    shape, spacing, origin, direction = header
    geometry = axis_map.geometry(spacing, origin, direction, shape[1:])
    return ([shape[0], *axis_map.extents(shape[1:])], *geometry)


def _config_output_group(config: dict) -> str:
    """The output group a preset's config names (``Predictor.outputs_dataset.*.OutputDataset.group``)."""
    for sink in config["Predictor"].get("outputs_dataset", {}).values():
        block = sink.get("OutputDataset", sink) if isinstance(sink, dict) else {}
        if "group" in block:
            return str(block["group"])
    raise KonfAIError("ImpactReg", "The preset's config names no output group.", "Give its OutputDataset a group.")


class ImpactRegKonfAIApp:
    """Run IMPACT-Reg preset apps, ensemble their displacement fields, evaluate, and estimate uncertainty."""

    def __init__(
        self, download: bool = False, force_update: bool = False, remote_server: RemoteServer | None = None
    ) -> None:
        self._download = download
        self._force_update = force_update
        # Each preset pass runs on this server when set; the ensemble, the evaluation and the uncertainty
        # stay here, on the fields it sends back.
        self._remote = remote_server
        # The --set overrides each preset run took, by its workspace (``label``): ``applied`` in register.json.
        self._applied: dict[str, list[str]] = {}
        # Per preset, the voxels it registers whole and those a tile holds, KonfAI's ``max_voxels``: ``plans`` in
        # register.json. The final plan per rank of each run, after any out-of-memory re-plan: KonfAI's Plan*.json,
        # ``ran``. It does not describe the cases a rank finished before a restart.
        self._plans: dict[str, dict[str, int | None]] = {}
        self._ran: dict[str, list[dict]] = {}
        # The run's own choices every preset pass takes as they are: --output-format, --seed. The moved image is
        # derived here, so its format (--output-format [Moved=]FORMAT) is this orchestrator's; None: the moving's.
        self._run_args: list[str] = []
        # Whether every preset pass writes its field, never a native form: an ensemble or a TTA averages them.
        self._dense = False
        self._moved_format: str | None = None

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
        uncertainty: bool = False,
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
        out = work / _workspace(label or preset)
        # This interpreter's konfai-apps, not the first on PATH: an unactivated environment has none there, and
        # another environment's would run another konfai and install the preset's requirements into it.
        # The bare app: an APP#PRESET's own --set are already in this pass's share (see _preset_overrides).
        app = split_preset(preset)[0]
        command = [sys.executable, "-m", "konfai_apps", "infer", app_id(IMPACT_REG_KONFAI_REPO, app)]
        command += ["-i", *(str(path) for path in fixed_images)]
        command += ["-i", *(str(path) for path in moving_images)]
        if moving_masks and not fixed_masks:
            fixed_masks = _neutral_fixed_masks(work / _workspace(f"{label or preset}_fixed_mask"), fixed_images)
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
        if uncertainty:  # its copies' stack beside the field: each TTA copy is a member of the ensemble
            command += ["-uncertainty"]
        if prediction_file:
            command += ["--prediction-file", prediction_file]
        # Preset-parameter tuning: this pass's share of --set (see _preset_overrides), for `konfai-apps infer`.
        for override in config_overrides or []:
            command += ["--set", override]
        self._applied[label or preset] = list(config_overrides or [])
        command += self._run_args
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
        env = {**os.environ, "TMPDIR": str(work)}
        if self._dense or label is not None:
            env[DENSE_TRANSFORMS_ENV] = "1"
        if self._remote is not None:
            command += ["--host", self._remote.host, "--port", str(self._remote.port)]
            # Through the environment konfai-apps reads it from, not the argv any process listing shows; a
            # server named without one gets none, not the token this process inherited.
            env.pop("KONFAI_API_TOKEN", None)
            if self._remote.token:
                env["KONFAI_API_TOKEN"] = self._remote.token
        # The engines' temporary files (elastix's images, FireANTs' total field) go to the work dir too: a
        # RAM-backed /tmp would hold several gigabytes a tile of a large pair in memory.
        try:
            subprocess.run(command, check=True, env=env)  # nosec B603
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
        plans = [json.loads(plan.read_text(encoding="utf-8")) for plan in (out / "predictions").glob("*/Plan*.json")]
        if plans:
            self._ran[label or preset] = sorted(plans, key=lambda plan: plan["rank"])
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
        uncertainty: bool = False,
        config_overrides: list[str] | None = None,
        tmp_dir: Path | None = None,
        fields_only: bool = False,
        max_voxels: int | None = None,
        output_format: list[str] | None = None,
        seed: int | None = None,
        out_of_core: str = "auto",
        region_size: int = 64,
        samples: int | None = None,
        epochs: int | None = None,
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

        ``out_of_core`` "sampled" is the second strategy: a pair over the budget registers at native resolution in
        one pass, never resampled and never held whole, its engine scoring a random subset of it at every iteration
        (``_register_sampled``). ``samples`` is how many an iteration draws, anew at each one, which its memory and
        its time follow: elastix that many points, its images mapped from files; ConvexAdam and FireANTs that many
        voxels, in random boxes of ``region_size`` voxels at most, a box of their deformable stages stepping
        ``epochs`` times on average (the stage's iterations when unset). A pair the device holds whole runs as it
        does.

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
        if out_of_core not in ROUTES:
            raise KonfAIError(
                "ImpactReg", f"--out-of-core {out_of_core} is not a route.", f"Use one of {', '.join(ROUTES)}."
            )
        overrides = _preset_overrides(presets, config_overrides, quiet, out_of_core != "tiled", seed)
        self._run_args = ["--seed", str(seed)] if seed is not None else []
        self._moved_format = None
        # The form each output is asked in (None: every output), for a field this layer writes itself (sampled).
        self._forms: dict[str | None, tuple[str, str]] = {}
        for spec in output_format or []:
            name, assigned, file_format = spec.rpartition("=")
            if name != "Moved":
                self._forms[name if assigned else None] = _asked_form(split_format_level(file_format)[0])
            if not assigned or name == "Moved":  # the moved image is derived here: its format is set here
                if split_format_level(file_format)[0] not in SUPPORTED_FORMATS:
                    raise KonfAIError(
                        "ImpactReg",
                        f"--output-format {spec}: '{file_format}' is not a format KonfAI writes."
                        f"{did_you_mean(file_format, SUPPORTED_FORMATS)}",
                        f"The formats: {', '.join(SUPPORTED_FORMATS)}.",
                    )
                self._moved_format = format_token(f".{file_format}")  # one token per form: zarr is omezarr
            if name != "Moved":  # a bare FORMAT is every output's, the engine's transform included
                self._run_args += ["--output-format", spec]
        # A transform keeps its native form (an affine, elastix's rigid then B-spline) where it is the result: one
        # preset, no TTA, registered whole. An ensemble averages fields, and so does a TTA; a tiled run composes them.
        self._dense = len(presets) > 1 or bool(tta)
        self._applied, self._plans, self._ran, self._compact = {}, {}, {}, {}

        work = _work_dir(tmp_dir, output, "impact_reg_")
        try:
            if not quiet:
                print(f"[ImpactReg] intermediates in {work}", flush=True)
            _check_time_points(inputs)
            headers = self._headers(inputs, work)
            _warn_of_other_units(headers, quiet)
            _check_volumes(inputs, headers)
            flat = is_flat(headers[0]["Fixed"][0])
            voxels = [max(math.prod(header[0][1:]) for header in case.values()) for case in headers]
            routes = {
                preset: _route(
                    out_of_core,
                    preset,
                    voxels,
                    _plan(_manifest(preset), gpu, max_voxels, self._remote),
                    overrides[preset]["global"],
                )
                for preset in presets
            }
            if flat:  # a case of a 2-D pair is on its own grid for every preset of an ensemble, or one plane thick
                sampled = sorted({index for route in routes.values() for index in route.sampled})
                for preset, route in routes.items():
                    if (
                        sampled
                        and route.sampled != sampled
                        and _sampled_engine(preset, overrides[preset]["global"]) is None
                    ):
                        raise KonfAIError(
                            "ImpactReg",
                            f"{preset} has no sampled route, which the other presets of this 2-D ensemble take.",
                            "Register the presets apart, or choose --out-of-core whole or tiled.",
                        )
                    routes[preset] = dataclasses.replace(route, sampled=sampled, tiled=route.tiled and not sampled)
            sampled_presets = [preset for preset, route in routes.items() if route.sampled]
            sampled_cases = {f"P{index:03d}" for route in routes.values() for index in route.sampled}
            # The sampled route registers a case on its own grid, of any dimension; the presets a case of a 2-D pair
            # on volumes one plane thick, on one z grid (plane.py), which reads a plane whole.
            native_headers, lifted = headers, inputs
            if flat:
                kept = [index for index in range(len(headers)) if f"P{index:03d}" not in sampled_cases]
                lifted = {
                    group: [
                        unit
                        if index not in kept
                        else lift_plane(unit, work / "planes" / group / f"{index:03d}", min(headers[index]["Fixed"][1]))
                        for index, unit in enumerate(units)
                    ]
                    for group, units in inputs.items()
                }
                headers = self._headers(lifted, work / "planes")
            for preset in sampled_presets if tta else []:
                # --tta as konfai-apps takes it on the other routes: up to the copies the app allows (app.json 'tta').
                allowed = int(_manifest(preset).get("tta") or 0)
                if tta > allowed:
                    print(
                        f"[ImpactReg] {preset}: {tta} test-time augmentation(s) asked, but it allows {allowed}"
                        f" (app.json 'tta'): running {allowed}.",
                        flush=True,
                    )
                if min(tta, allowed):
                    raise KonfAIError(
                        "ImpactReg",
                        f"{preset}: the sampled route registers a pair once, with no test-time augmentation.",
                        "Drop --tta, or choose --out-of-core whole or tiled.",
                    )
            # The twins of a sampled run are written where it runs: on the server, with --host.
            _check_space(work, headers, presets, bool(sampled_presets) and self._remote is None)
            on_grid = None
            fields_by_preset = {}
            for index, preset in enumerate(presets, 1):
                if len(presets) > 1 and not quiet:
                    print(f"[ImpactReg] preset {index}/{len(presets)}: {preset}", flush=True)
                tiling, route = _manifest(preset).get("tiling") or {}, routes[preset]
                self._plans[preset] = {"whole_voxels": route.budget, "tile_voxels": route.tile_budget}
                # The cases registered sampled keep their case ID; the others register as they would alone.
                over = route.sampled
                rest = [index for index in range(len(headers)) if index not in over]
                if over:
                    sampled_run = self._register_sampled if self._remote is None else self._register_sampled_remote
                    fields_by_preset[preset] = sampled = sampled_run(
                        preset,
                        _some_cases(inputs, over),
                        [native_headers[index] for index in over],
                        [f"P{index:03d}" for index in over],
                        work,
                        gpu,
                        cpu,
                        quiet,
                        overrides[preset],
                        region_size,
                        samples,
                        epochs,
                    )
                    rest = [index for index in rest if index not in over]
                    if not rest:
                        continue
                if over and not quiet:
                    print(
                        f"[ImpactReg] {preset}: {', '.join(f'P{index:03d}' for index in rest)} fit the device;"
                        " registered whole.",
                        flush=True,
                    )
                tiled = route.tiled
                # KonfAI patches each image on its own grid, and a test-time flip turns each about its own centre:
                # the pair goes onto the fixed grid where it would be tiled as it came (tiles with no global pass
                # before them) and wherever it is flipped.
                onto_fixed = bool(tta or (tiled and not tiling.get("global")))
                if onto_fixed:
                    on_grid = on_grid or self._onto_fixed_grid(lifted, headers, work, gpu, cpu, quiet)
                pair = on_grid if onto_fixed else (lifted, headers)
                if over:  # the cases that fit, as a cohort of their own: neither tiled nor flipped (no --tta here)
                    pair = (_some_cases(pair[0], rest), [pair[1][index] for index in rest])
                if tiled:
                    fields_by_preset[preset] = self._register_tiled(
                        preset,
                        tiling,
                        *pair,
                        route.budget,
                        route.tile_budget,
                        work,
                        gpu,
                        cpu,
                        quiet,
                        tta,
                        overrides[preset],
                    )
                    continue
                # Registered whole: KonfAI resamples a pair over the budget onto a coarser grid and its field back
                # onto the native fixed grid (the preset's patch ``mode: resample``); --out-of-core whole holds it
                # on its native grid, and refuses a pass KonfAI had to resample once it ran out of memory.
                native = out_of_core == "whole"
                fields_by_preset[preset] = self._infer_preset(
                    preset,
                    *(pair[0].get(group, []) for group in ("Fixed", "Moving", "FixedMask", "MovingMask")),
                    work,
                    gpu,
                    cpu,
                    quiet,
                    tta,
                    overrides[preset]["global"],
                    max_voxels=max(voxels[index] for index in rest) if native else route.budget,
                    uncertainty=uncertainty and bool(tta) and not flat,
                )
                resampled = [
                    plan["resample_voxels"] for plan in self._ran.get(preset, []) if plan.get("resample_voxels")
                ]
                if native and resampled:
                    raise KonfAIError(
                        "ImpactReg",
                        f"{preset}: the pair does not fit the device whole; KonfAI resampled it to"
                        f" {min(resampled):,} voxels once it ran out of memory.",
                        "Choose --out-of-core auto or sampled, which register it at native resolution by boxes.",
                    )
                if over:
                    (sampled_group, sampled_fields), (group, whole_fields) = sampled, fields_by_preset[preset]
                    if group != sampled_group:
                        raise RuntimeError(
                            f"preset '{preset}' named its output '{sampled_group}' sampled and '{group}' whole; the"
                            " cases of one preset must share one group."
                        )
                    ids = {f"P{position:03d}": f"P{index:03d}" for position, index in enumerate(rest)}
                    fields_by_preset[preset] = (
                        group,
                        {**sampled_fields, **{ids[c]: f for c, f in whole_fields.items()}},
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

            # The compact transforms the sampled routes keep beside the dense field (``<preset>_<name>``), by case:
            # the record of a case this run leaves alone stands.
            stale = _recorded_compacts(output)
            compacts = {case: names for case, names in stale.items() if case not in cases}
            for case in cases:
                # konfai-apps already named the cases; reusing its names keeps the two layers' notion of
                # a case identical instead of renumbering from the command line and hoping they agree.
                case_out = output / case
                case_out.mkdir(parents=True, exist_ok=True)
                # A compact transform stands beside the field it was found with: those an earlier run recorded go
                # before the field is replaced, this run's once every output stands (below), so that no failure leaves
                # a field beside another's transform, nor a compact the record does not hold.
                for name in stale.get(case, ()):
                    (case_out / Path(name).name).unlink(missing_ok=True)
                members = [fields_by_preset[preset][case] for preset in presets]
                # A preset run with TTA under --uncertainty hands over its copies' stack: every copy is a member.
                sources = [_stack_beside(member) or member for member in members] if uncertainty else members
                stack = None
                # A sampled field (a MetaImage) is written here in the form asked for it; a preset wrote its own.
                asked = (self._forms.get(group) or self._forms.get(None)) if case in sampled_cases else None
                form, token = asked or (_form(members[0]), member_token(members[0]))
                same = _form(members[0]) == form and member_token(members[0]) == token  # mha.gz is .mha, compressed
                if sources == members and len(presets) == 1 and same:
                    _copy_output(members[0], case_out, group, move=True)
                else:
                    # An ensemble is konfai-apps' (konfai_apps/ensemble.py): every member reduced into the field,
                    # slab by slab, the members on one grid, in ``form``. With --uncertainty their stack is kept
                    # beside it, built here, or for a 2-D pair lifted to volumes once its members are planes again
                    # (below); a case registered on its own grid has planes already.
                    lifted_case = flat and case not in sampled_cases
                    stack = case_out / stack_file(form) if uncertainty and not lifted_case else None
                    combine(sources, "Mean", case_out / f"{group}{form}", token, stack)
                    _drop_other_forms(case_out, group, form)  # an earlier run's, in another form
                # An earlier run's stack, of another transform, goes once this one stands: a failed run keeps both.
                _drop_other_forms(case_out, STACK, stack.name.removeprefix(STACK) if stack else "")

            if not fields_only:
                # The moved images in ONE streamed run over the cohort: Resample adopts, per case,
                # the grid of that case's own field (a field is defined ON the fixed grid) and reads
                # the field as the map.
                self._derive_moved(
                    dict(zip(cases, lifted["Moving"], strict=True)),
                    dict(zip(cases, lifted["Fixed"], strict=True)),
                    sampled_cases if flat else set(),
                    group,
                    output,
                    work,
                    gpu,
                    cpu,
                    quiet,
                )
            if flat:
                for case, moving in zip(cases, moving_units, strict=True):
                    if case in sampled_cases:  # registered on its own grid: its outputs are 2-D already
                        continue
                    project_field(_the_output(output / case, group))
                    if not fields_only:  # in the moving's own spelling (.zarr stays .zarr) unless a form was chosen
                        moved = _the_output(output / case, "Moved")
                        project_image(moved, moved if self._moved_format else output / case / f"Moved{_form(moving)}")
                    if uncertainty and len(presets) > 1:
                        # The members as planes, then their stack: a 2-D ensemble's stack holds 2-D fields.
                        members = [fields_by_preset[preset][case] for preset in presets]
                        for member in members:
                            project_field(member)
                        target = _the_output(output / case, group)
                        stack = output / case / stack_file(_form(members[0]))
                        combine(members, "Mean", target, member_token(members[0]), stack)
            if uncertainty and not quiet:
                unstacked = [case for case in cases if not any(_is_entry(p, STACK) for p in (output / case).iterdir())]
                if unstacked:
                    print(
                        f"[ImpactReg] WARNING: --uncertainty kept no stack for {', '.join(unstacked)}: it needs two or"
                        " more presets, or --tta on a 3-D pair registered whole.",
                        file=sys.stderr,
                        flush=True,
                    )
            for case in cases:
                for preset in presets:
                    for compact in self._compact.get(preset, {}).get(case, ()):
                        dest = output / case / f"{_workspace(preset)}_{compact.name}"
                        compacts.setdefault(case, []).append(_publish_output(compact, dest, move=True).name)
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
            "ran": self._ran,
            "compacts": compacts,
            "tta": tta,
            "out_of_core": out_of_core,
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

    def _register_sampled(
        self,
        preset: str,
        inputs: dict[str, list[Path]],
        headers: "list[dict[str, _Header]]",
        cases: list[str],
        work: Path,
        gpu: list[int],
        cpu: int | None,
        quiet: bool,
        config_overrides: dict[str, list[str]],
        region_size: int,
        samples: int | None,
        epochs: int | None,
    ) -> tuple[str, dict[str, Path]]:
        """One preset on pairs too large for the device by the second out-of-core strategy (``--out-of-core
        sampled``): the preset whole (the config a pair that fits registers with) on the native pair of each of
        ``cases`` (their IDs in the cohort; ``inputs`` and ``headers`` are theirs), never resampled and never held
        whole, each engine evaluating its loss on a random subset of the pair at every iteration, as its own algorithm
        allows. ``config_overrides`` holds the ``--set`` share of each pass; ``applied`` records the one it takes.

        The native pair is first written as twins, from one read of each input by bands (``_native_twins``). Then:

        - ConvexAdam (``_sampled_convexadam``): its coarse stage by regions, its fine stage on random boxes of the
          control grid.
        - elastix (``_sampled_elastix``): elastix as it is, which draws its samples at every iteration, with every
          image of the run mapped from files and paged by the kernel.
        - FireANTs (``_sampled_fireants``): its rigid, affine and deformable stages on random boxes of the pair, the
          warps on the device, in host memory or in files.

        Each leaves the dense field on the native fixed grid, written by regions, and a compact transform beside it.
        """
        import importlib

        from konfai_apps.app_repository import get_app_repository_info

        from .models.intensity import world_transform
        from .native import AxisMap, remap, to_millimetres

        config = _preset_config(preset, "Prediction.yml")
        # A --set of the classpath may name another of ImpactReg's engines, whose class then checks the other --set.
        module = _sampled_engine(preset, config_overrides["global"])
        if module is None:
            raise KonfAIError(
                "ImpactReg",
                f"{preset} cannot run --out-of-core sampled: its engine has no stage that runs on a subset of the"
                " pair.",
                "Choose --out-of-core auto, whole or tiled.",
            )
        # The engine runs here: its requirements are installed as resolving the app installs them.
        get_app_repository_info(app_id(IMPACT_REG_KONFAI_REPO, split_preset(preset)[0]), False).install_requirements()
        try:
            network = importlib.import_module(f".models.{module}", __package__).RegistrationNet
        except ModuleNotFoundError as error:
            raise KonfAIError(
                "ImpactReg",
                f"--out-of-core sampled runs the preset's engine in this process, which needs {error.name}.",
                "Install the preset's requirements (KONFAI_APPS_INSTALL_REQUIREMENTS=0 turns their installation off).",
            ) from error
        apply_overrides(config, config_overrides["global"], lambda: network)
        self._applied[f"{preset}_sampled"] = list(config_overrides["global"])
        # The engine takes the cores a run of one rank takes: a sum split over threads (elastix's metric) then adds
        # in the same order, and the transform is the run in memory's.
        apply_cpu_thread_budget(1)
        # The pair as a run on the whole pair reorients it: each image's voxel axes along LPS, nothing resampled.
        maps = [{group: AxisMap.to_lps(case[group][3]) for group in ("Fixed", "Moving")} for case in headers]
        views = [
            {group: _viewed(header, case_maps[group.removesuffix("Mask")]) for group, header in case.items()}
            for case, case_maps in zip(headers, maps, strict=True)
        ]
        run = _SampledRun(
            preset,
            config,
            inputs,
            views,
            work,
            gpu,
            cpu,
            quiet,
            region_size,
            samples,
            epochs,
            _config_output_group(config),
            cases,
            maps,
        )
        fields = getattr(self, f"_sampled_{module}")(run)
        shutil.rmtree(work / _workspace(f"{preset}_native"), ignore_errors=True)
        # The engines measured the pair in the fixed image's finest voxel: the fields and the compact transforms
        # come back in millimetres, a field on the fixed image's own grid.
        for index, case in enumerate(run.cases):
            voxel = min(headers[index]["Fixed"][1])
            to_millimetres(fields[case], voxel)
            if not maps[index]["Fixed"].identity:
                remap(fields[case], maps[index]["Fixed"].inverse())
            for compact in self._compact.get(preset, {}).get(case, ()):
                if compact.suffix == ".h5":
                    sitk.WriteTransform(world_transform(sitk.ReadTransform(str(compact)), voxel), str(compact))
                else:
                    to_millimetres(compact, voxel)
        return run.output_group, fields

    def _register_sampled_remote(
        self,
        preset: str,
        inputs: dict[str, list[Path]],
        headers: "list[dict[str, _Header]]",
        cases: list[str],
        work: Path,
        gpu: list[int],
        cpu: int | None,
        quiet: bool,
        config_overrides: dict[str, list[str]],
        region_size: int,
        samples: int | None,
        epochs: int | None,
    ) -> tuple[str, dict[str, Path]]:
        """``_register_sampled`` on the konfai-apps server ``--host`` names, by its runner ``impact_reg_sampled``
        (``remote.py``): the cases' images go up in a stream, the server registers them as this machine would, and
        their fields and compact transforms come back; the moved images are derived here, as after a local run."""
        from konfai_apps.app import KonfAIAppClient

        assert self._remote is not None
        app = split_preset(preset)[0]  # its own --set are in this pass's share (see _preset_overrides)
        names = [group for group in ("Fixed", "Moving", "FixedMask", "MovingMask") if inputs.get(group)]
        args = [f"groups={','.join(names)}", f"region_size={region_size}"]
        args += [f"set={override}" for override in config_overrides["global"]]
        args += [f"{name}={value}" for name, value in (("samples", samples), ("epochs", epochs)) if value is not None]
        out = work / _workspace(f"{preset}_remote")
        KonfAIAppClient(app_id(IMPACT_REG_KONFAI_REPO, app), self._remote).run(
            "impact_reg_sampled",
            [[str(path) for path in inputs[group]] for group in names],
            output=out,
            runner_args=args,
            gpu=gpu,
            cpu=cpu,
            quiet=quiet,
        )
        self._applied[f"{preset}_sampled"] = list(config_overrides["global"])
        group = _config_output_group(_preset_config(preset, "Prediction.yml"))
        # The server named the cases it was sent P000, P001, ... in their order; its compact transforms carry its
        # preset's name, which publishing here gives them again.
        recorded, prefix = _recorded_compacts(out), f"{_workspace(app)}_"
        fields, compacts = {}, {}
        for index, case in enumerate(cases):
            sent = out / f"P{index:03d}"
            fields[case] = _the_output(sent, group)
            compacts[case] = []
            for name in recorded.get(sent.name, []):
                compacts[case].append((sent / name).rename(sent / name.removeprefix(prefix)))
        self._compact[preset] = compacts
        return group, fields

    def _native_twins(
        self, run: "_SampledRun", *, chunked: bool, clipped: bool, positive: bool = False
    ) -> Iterator[tuple[int, str, dict[str, Path], list[Path | None], int]]:
        """The native pair of each case as twins under ``<work>/<preset>_native/<case>``, from one read of each input
        by bands (``native.write_twin``): an uncompressed MetaImage per image and mask, which an engine memory-maps,
        or with ``chunked`` an HDF5 image in ITK's layout with chunks, which ITK's readers stream by boxes. Their
        geometry is measured in the fixed image's finest voxel, as ``EngineRegistration`` measures a pair registered
        whole. ``clipped`` winsorises the MetaImage twins (KonfAI's quantiles, by bounded passes). A mask goes onto
        its image's grid first (KonfAI TRANSFORM, streamed), and is written binary: its non-zero voxels, or with
        ``positive`` its positive ones, as the engine reads a mask on the whole pair.

        The masks of every case come first, so that an empty one is refused before any image is written
        (``_partial_masks``); then one case at a time, its images: each case is handed over as its index, its ID, its
        twins by group, its partial masks and the voxels its fixed mask keeps, and its twins go once the next case
        is asked for. The disk holds the masks and the twins of one case."""
        from konfai.data.transform import Resample, TensorCast, Write

        from .native import write_twin

        native_root = run.work / _workspace(f"{run.preset}_native")

        def twin(index: int, case: str, group: str) -> Path:
            root, source = native_root / case, run.inputs[group][index]
            mask = group.endswith("Mask")
            image_group = group.removesuffix("Mask")
            if mask and not _same_grid(run.headers[index][group], run.headers[index][image_group]):
                roots = [
                    _stage_group(root / "stage", group, {case: source}),
                    _stage_group(root / "stage", image_group, {case: run.inputs[image_group][index]}),
                ]
                chain = [
                    Resample(reference="{case}", reference_group=image_group, interpolation="nearest"),
                    TensorCast(dtype="float32"),
                    Write(dataset=f"{root / 'on_grid'}:mha"),
                ]
                _run_transform(
                    f"impact_reg_native_{group}_{case}",
                    roots,
                    {group: {group: chain}},
                    run.work,
                    run.gpu,
                    run.cpu,
                    run.quiet,
                )
                source = root / "on_grid" / case / f"{group}.mha"
            staged, _, backend = _stage_group(root / "stage", group, {case: source}).rpartition(":")
            dataset = Dataset(staged, backend)
            clip = None
            if clipped and not mask:
                # KonfAI's own quantile: the value numpy gives on the whole image, read by bounded passes.
                clip = tuple(float(dataset.read_data_quantile(group, case, q)) for q in (0.0001, 0.9999))
                clip = clip if clip[1] > clip[0] else None  # as intensity.winsorized leaves such an image
            return write_twin(
                dataset,
                group,
                case,
                root / f"{group}{'.h5' if chunked else '.mha'}",
                dtype="uint8" if mask else "float32",
                clip=clip,
                voxel=min(run.headers[index]["Fixed"][1]),
                positive=positive,
                axis_map=run.maps[index][image_group],
            )

        masks = {}
        for index, case in enumerate(run.cases):
            twins = {group: twin(index, case, group) for group in run.inputs if group.endswith("Mask")}
            masks[case] = (twins, *_partial_masks(twins, case))
        for index, case in enumerate(run.cases):
            twins, partial, kept = masks.pop(case)
            twins.update((group, twin(index, case, group)) for group in ("Fixed", "Moving"))
            yield index, case, twins, partial, kept
            shutil.rmtree(native_root / case, ignore_errors=True)

    def _sampled_convexadam(self, run: "_SampledRun") -> dict[str, Path]:
        """The ConvexAdam preset by regions (itk-impact's ``RegionSize``). The coarse stage, when the preset has one:
        one pass over the boxes pools the native features to its coarse grid, its cost volume is built by boxes of
        that grid, never whole, and its field starts the fine stage. The fine stage: one pass over the boxes of
        ``region_size`` voxels, read with their halo, pools the native features to the control grid; where the
        pooled maps fit the device the iterations run on them whole (ConvexAdam's own Static mode), and where
        they do not they live in files of the workspace and every iteration steps every cell, by bricks the
        device holds: the same iterations. The dense field on the fixed grid is written by regions; the field on
        the control grid is kept beside it."""
        from .models.convexadam import LevelSpec, RegistrationNet, refuse_linear_keys, region_geometry, require_regions
        from .models.intensity import engine_of

        refuse_linear_keys(LocalAppRepository._model_param_block(run.config)[1] or {})
        engine = engine_of(run.config, RegistrationNet, LevelSpec)
        engine.check_gradients()
        require_regions()  # before any twin is written
        # The stage's iterations are ConvexAdam's, one step fewer: it returns the field its last iteration evaluates.
        visits = int(run.epochs) if run.epochs else max(engine._iterations - 1, 0)
        # With ``--samples``, an iteration draws that many voxels, in boxes that hold no more.
        samples = run.samples
        fields_root = run.work / _workspace(f"{run.preset}_fields")
        fields: dict[str, Path] = {}
        compact: dict[str, list[Path]] = {}
        # The stages read the images as they are, as ConvexAdam does.
        for index, case, native, masks, _ in self._native_twins(run, chunked=True, clipped=False):
            size = [int(extent) for extent in run.headers[index]["Fixed"][0][1:]][::-1]
            region_size = run.region_size
            if samples is not None:  # boxes that hold no more than an iteration draws, in the pair's dimension
                region_size = min(region_size, max(8, round(samples ** (1 / len(size)))))
            grid_shrink = engine._grid_shrink
            geometry = region_geometry(size, grid_shrink, region_size)
            if not run.quiet and "fine" not in engine._stages:
                print(
                    f"[ImpactReg] {run.preset} {case}: coarse stage by regions (itk-impact ImpactCoarseRegistration)"
                    f" on {' x '.join(map(str, size))} native voxels, its field written by regions.",
                    flush=True,
                )
            elif not run.quiet:
                cells = math.prod(geometry["cells"])
                schedule = (
                    f"one pass over the boxes pools the features to the control grid, then {visits} iterations on"
                    " them, on the device where they fit and from files of the workspace otherwise"
                )
                if samples is not None:
                    schedule = (
                        "one pass over the boxes pools the features to the control grid, on the device where they fit"
                        " and in files of the workspace otherwise;"
                        f" an iteration then steps the boxes it draws, {samples:,} voxels of them, each box"
                        f" {visits} times on average"
                    )
                start = (
                    "from the coarse stage's field, computed by regions too"
                    if "coarse" in engine._stages
                    else "from the identity"
                )
                if engine._moments_init != "none":
                    centres = "of mass" if engine._moments_init == "com" else "of the frames"
                    start += f", the moving image first translated by the centres {centres}"
                print(
                    f"[ImpactReg] {run.preset} {case}: fine stage by random boxes (itk-impact ImpactFineRegistration) on"
                    f" {' x '.join(map(str, size))} native voxels, {start}: control grid every"
                    f" {grid_shrink} voxels ({cells:,} cells, about"
                    f" {cells * len(size) * 4 * 5 / 2**20:,.0f} MiB on the device), boxes of {region_size} voxels"
                    f" ({geometry['regions']} boxes) read with their halo from HDF5 chunks; {schedule}.",
                    flush=True,
                )
            fields[case] = fields_root / case / f"{run.output_group}.mha"
            fields[case].parent.mkdir(parents=True, exist_ok=True)
            result = engine.refine_by_regions(
                native["Fixed"],
                native["Moving"],
                *masks,
                run.device_index,
                region_size,
                visits,
                fields[case],
                samples=samples,
            )
            compact[case] = [result.control_grid]
            self._ran[f"{run.preset}_regions_{case}"] = {
                "resident_features": result.resident,
                "iterations": result.iterations,
                "regions": result.regions,
                "bytes_read": result.bytes_read,
                "streamed": {"fixed": result.streamed[0], "moving": result.streamed[1]},
                "seconds": round(result.seconds, 1),
                "device_peak_bytes": result.device_peak_bytes,
                "loss": [result.metric_values[0], result.metric_values[-1]] if result.metric_values else [],
            }
            if result.coarse is not None:
                self._ran[f"{run.preset}_regions_{case}"]["coarse"] = {
                    "seconds": round(result.coarse[0], 1),
                    "bytes_read": result.coarse[1],
                }
            if not run.quiet:
                streamed = "by regions" if all(result.streamed) else "whole (the readers could not stream)"
                route = "pooled maps resident" if result.resident else "box by box"
                if result.coarse is not None:
                    print(
                        f"[ImpactReg] {run.preset} {case}: coarse stage by regions in {_duration(result.coarse[0])},"
                        f" {result.coarse[1] / 2**30:.2f} GiB read.",
                        flush=True,
                    )
                if "fine" in engine._stages:
                    loss = (
                        f", loss {result.metric_values[0]:.4f} -> {result.metric_values[-1]:.4f}"
                        if result.metric_values
                        else ""
                    )
                    print(
                        f"[ImpactReg] {run.preset} {case}: {route}, {result.iterations:,} iterations in"
                        f" {_duration(result.seconds)},"
                        f" {result.bytes_read / 2**30:.2f} GiB read {streamed}, device peak"
                        f" {(result.device_peak_bytes or 0) / 2**20:,.0f} MiB{loss}.",
                        flush=True,
                    )
        self._compact[run.preset] = compact
        return fields

    def _sampled_elastix(self, run: "_SampledRun") -> dict[str, Path]:
        """The elastix preset on the native pair. elastix draws the samples of its metric at every iteration: what
        kept a large pair out of its reach is that it holds every image whole (the pair, each pyramid level, the
        coefficients of each interpolator). It runs here as it is, with those images mapped from files of the
        workspace and paged by the system (its ``-out-of-core``): the transform is the one of a run in memory. The dense
        field is its transform sampled on the fixed grid band by band; the transform itself is kept beside it."""
        from .models.elastix import ElastixLevelSpec, RegistrationNet
        from .models.elastix_engine import with_shift
        from .models.intensity import engine_of
        from .native import mapped_centre, mmap_image, shift_origin, write_field

        block = LocalAppRepository._model_param_block(run.config)[1]
        # The engine looks its parameter maps up beside the config a KonfAI prediction stages; here they are the
        # preset's own files, or a map of one's own given by an absolute path, as the engine honours it.
        block["parameter_maps"] = [
            name
            if Path(name).is_absolute() and Path(name).is_file()
            else str(app_file(IMPACT_REG_KONFAI_REPO, split_preset(run.preset)[0], Path(name).name))
            for name in block["parameter_maps"]
        ]
        if run.samples is not None:  # the points every map draws, whatever the image holds
            block["spatial_samples"], block["voxel_sampling"] = int(run.samples), 1.0
        engine = engine_of(run.config, RegistrationNet, ElastixLevelSpec)
        if not engine.supports_mapped_images():
            raise KonfAIError(
                "ImpactReg",
                f"{run.preset} cannot run --out-of-core sampled with this elastix ({engine._elastix_bin}): it has no"
                " -out-of-core option, which keeps the images of a registration in files.",
                "Point KONFAI_ELASTIX_DIR at an elastix-IMPACT build that has it.",
            )
        impact = bool(engine._feature_models)
        fields_root = run.work / _workspace(f"{run.preset}_fields")
        fields: dict[str, Path] = {}
        compact: dict[str, list[Path]] = {}
        # Grey values alone are winsorised, as the engine does; IMPACT's models read the intensities they know.
        for index, case, raw, masks, inside in self._native_twins(run, chunked=False, clipped=not impact):
            fixed = mmap_image(raw["Fixed"])
            shift = None
            if engine._moments_init == "com":
                # The moving twin goes there by its origin, as the engine moves a pair registered whole, its centre
                # taken by a pass over the twin; the shift is added back to the transform.
                mapped = [None if path is None else mmap_image(path) for path in masks]
                shift = mapped_centre(mmap_image(raw["Moving"]), mapped[1], "com") - mapped_centre(
                    fixed, mapped[0], "com"
                )
                for path in (raw["Moving"], masks[1]):
                    if path is not None:
                        shift_origin(path, shift)
            # voxel_sampling counts the voxels the sampler draws from: the fixed mask's, when there is one.
            voxels = inside or fixed.array.size
            out = fields_root / case
            out.mkdir(parents=True, exist_ok=True)
            if not run.quiet:
                print(
                    f"[ImpactReg] {run.preset} {case}: elastix on {' x '.join(map(str, fixed.array.shape[::-1]))}"
                    " native voxels, its images kept in files and paged by the system (-out-of-core).",
                    flush=True,
                )
            start = time.perf_counter()
            transform = engine.register_mapped(
                raw["Fixed"],
                raw["Moving"],
                masks[0],
                masks[1],
                run.device_index,
                out / "elastix",
                tuple(v / min(run.headers[index]["Fixed"][1]) for v in run.headers[index]["Fixed"][1]),
                voxels,
            )
            transform = with_shift(transform, None if shift is None else shift.tolist())
            seconds = time.perf_counter() - start
            compact[case] = [out / "Transform.h5"]
            sitk.WriteTransform(transform, str(compact[case][0]))
            fields[case] = write_field(out / f"{run.output_group}.mha", fixed, transform)
            shutil.rmtree(out / "elastix", ignore_errors=True)
            self._ran[f"{run.preset}_mapped_{case}"] = {"seconds": round(seconds, 1), "sampled_voxels": voxels}
            if not run.quiet:
                print(f"[ImpactReg] {run.preset} {case}: registered in {_duration(seconds)}.", flush=True)
        self._compact[run.preset] = compact
        return fields

    def _sampled_fireants(self, run: "_SampledRun") -> dict[str, Path]:
        """The FireANTs preset on boxes of the native pair (``FireANTsEngine.register_sampled``): its rigid
        and affine stages start a scale on random boxes and end it on every box of the level, by packets of
        ``samples`` voxels; its deformable stage (greedy or SyN) takes FireANTs' iteration on the whole grid box by
        box, every box once an epoch, a box holding no more than ``samples``, the pyramid levels in files of the workspace, the warps on the GPU when they fit, in files
        otherwise. The deformable stage is FireANTs' on the whole image, the linear ones with every box in one
        iteration. The displacement field is written by blocks."""
        from .models.fireants import RegistrationNet
        from .models.impact_loss import LevelSpec
        from .models.intensity import engine_of

        engine = engine_of(run.config, RegistrationNet, LevelSpec)
        for index in range(len(run.cases)):
            engine.check_sampled(*(run.headers[index][group][0][1:] for group in ("Fixed", "Moving")))
        fields: dict[str, Path] = {}
        # FireANTs keeps a mask's positive voxels (``fireants.py``), the other engines its non-zero ones.
        for index, case, raw, masks, _ in self._native_twins(run, chunked=False, clipped=False, positive=True):
            out = run.work / _workspace(f"{run.preset}_fields") / case
            out.mkdir(parents=True, exist_ok=True)
            fields[case] = out / f"{run.output_group}.mha"
            if not run.quiet:
                shape = run.headers[index]["Fixed"][0][1:]
                drawn = (
                    f"{run.samples:,} voxels at once" if run.samples is not None else "what the device holds at once"
                )
                print(
                    f"[ImpactReg] {run.preset} {case}: FireANTs on boxes of {' x '.join(map(str, shape[::-1]))} native"
                    f" voxels: {drawn}, the warps on the device when they fit, else in host memory or in files.",
                    flush=True,
                )
            start = time.perf_counter()
            measured = engine.register_sampled(
                raw["Fixed"],
                raw["Moving"],
                masks[0],
                masks[1],
                run.device_index,
                out / "warps",
                fields[case],
                run.region_size,
                run.samples,
                run.epochs,
            )
            shutil.rmtree(out / "warps", ignore_errors=True)
            seconds = time.perf_counter() - start
            self._ran[f"{run.preset}_boxes_{case}"] = {
                "seconds": round(seconds, 1),
                **{key: round(value, 1) if isinstance(value, float) else value for key, value in measured.items()},
            }
            if not run.quiet:
                print(
                    f"[ImpactReg] {run.preset} {case}: registered in {_duration(seconds)},"
                    f" {measured.get('bytes_read', 0) / 2**30:.2f} GiB read"
                    f"{', boxes a scale ' + str(measured['boxes_per_scale']) if 'boxes_per_scale' in measured else ''}.",
                    flush=True,
                )
        return fields

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
            shape = max((case["Fixed"][0][1:] for case in headers), key=math.prod)
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
                        work / _workspace(f"{preset}_prewarp"),
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
                work / _workspace(f"{preset}_prewarp"),
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
        shutil.rmtree(work / _workspace(f"{preset}_prewarp"), ignore_errors=True)
        reach = max(field_reach(*_field(field, work / _workspace(f"{preset}_reach"))) for field in tile_fields.values())
        padding = _compose_padding(headers, budget, reach)
        composed = self._compose(
            tile_fields, global_fields, fixed, group, padding, work / _workspace(f"{preset}_compose"), gpu, cpu, quiet
        )
        shutil.rmtree(work / _workspace(f"{preset}_tiles"), ignore_errors=True)
        return group, composed

    def _derive_moved(
        self,
        cases: dict[str, Path],
        fixed: dict[str, Path],
        planes: set[str],
        group: str,
        output: Path,
        work: Path,
        gpu: list[int],
        cpu: int | None,
        quiet: bool,
    ) -> None:
        """The moved images, derived from the transforms, one streamed run over the cohort for each form its
        transforms and moving images take: a cohort routed case by case holds the dense field of a case registered
        sampled beside the native form of one registered whole, and a staged root has one backend. ``planes`` are the
        cases of a 2-D pair registered on their own grid, apart from the others, lifted to volumes one plane thick.

        A preset that emits only a transform is complete: everything else IS that transform. The moved
        image: ``reference: '{case}'`` adopts each case's own DVF grid (a field is defined ON the
        fixed grid) and the field is the map (``field_group``): one interpolation, streamed, each
        slab's source window sized from the field values it reads. An affine (a linear preset's native form) has no
        grid of its own: the fixed image lends it, and the affine is the map KonfAI decodes (``transforms``).

        NOTHING ELSE IS DERIVED. A preset writes its transform in the form its consumer reads (an ITK
        transform file where Slicer picks it up, an RFC-5 store where this pipeline streams it) so
        there is no second copy of the same field to produce under another name.

        A moved OME-Zarr store is written as a pyramid (KonfAI's ``scale_factors: auto``): its moving came as one,
        and a single level made a viewer load the full resolution to show it at all.
        """
        from konfai.data.transform import Resample, Write

        fields = {case: _the_output(output / case, group) for case in cases}
        forms: dict[tuple[str, bool, str, bool], dict[str, Path]] = {}
        for case, field in fields.items():
            key = (_form(field).lower(), _is_field(field), _form(cases[case]).lower(), case in planes)
            forms.setdefault(key, {})[case] = field
        for number, ((_, dense, _, _), members) in enumerate(forms.items()):
            stage, root = work / "moved_stage" / str(number), work / "moved" / str(number)
            moving = {case: cases[case] for case in members}
            # The moved image is written in the form --output-format chose, else the MOVING's own: the fields' form
            # may be an ITK transform file, which cannot hold an image. The members' moving images share one form.
            backend = self._moved_format or _backend(next(iter(moving.values())))
            roots = [_stage_group(stage, "Moving", moving), _stage_group(stage, "DVF", members)]
            if dense:
                resample = Resample(
                    reference="{case}", reference_group="DVF", field_group="DVF", interpolation="linear"
                )
            else:
                roots.append(_stage_group(stage, "Fixed", {case: fixed[case] for case in members}))
                resample = Resample(
                    reference="{case}", reference_group="Fixed", transforms={"DVF": False}, interpolation="linear"
                )
            # Written apart, then published: a root has one backend, read off what it holds (_stage_group), so beside
            # a store, here or left by an earlier run, a moved file would come out as a store.
            _run_transform(
                f"impact_reg_moved_{number}",
                roots,
                {
                    "Moving": {
                        "Moved": [
                            resample,
                            Write(
                                dataset=f"{root}:{backend}",
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
            for case in members:
                _copy_output(_the_output(root / case, "Moved"), output / case, "Moved", move=True)

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
        measures without reference (Jacobian, bending energy, displacement lengths), and is evaluated alone when no
        modality is given; a B-spline (elastix's native form) gets them on the fixed grid, the fixed image's or
        segmentation's. Case ``P###`` is written to ``<output>/P###/Evaluation/<modality>/`` (Image,
        Segmentation, Landmarks, NoReference), and the whole call to ``<output>/Evaluation_summary.json``.

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
        # and, with --uncertainty, the ensemble's stack, each of which would be scored as a case of its own.
        expanded = [(path, unit) for path in transforms for unit in ([path] if path.is_file() else _units([path]))]
        strays = [unit for _, unit in expanded if _is_entry(unit, "Moved") or _is_entry(unit, STACK)]
        if strays:
            raise EvaluatorError(
                f"--transform takes in '{strays[0]}', register's moved image or ensemble stack, not a case's transform.",
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
        # A transform given alone is a case of its own: its measures without reference, when it is a field, are the
        # evaluation.
        n_cases = max(len(fixed_images), len(gt_fixed_seg), len(gt_fixed_fid)) or len(transforms)
        scored: list[tuple[str, Path]] = []
        # One transform given for every case is measured once: it reads the whole field (on one grid, for a B-spline).
        jacobians: dict[tuple[Path, Path | None], Path] = {}
        for index in range(n_cases):
            # One output folder AND one workspace per modality: the configs share one train_name and konfai-apps
            # evaluates with overwrite.
            eval_out = output / f"P{index:03d}" / "Evaluation"
            work = _work_dir(tmp_dir, output, "impact_reg_eval_")
            try:
                # Each modality is konfai-apps' evaluation of the transform (-i) against the fixed and the moving data
                # (--gt): its config pulls the moving side through the transform onto the fixed grid, or pushes the
                # fixed landmarks into the moving image. With no transform the moving data is taken as registered
                # already, and an identity only resamples it onto the fixed grid.
                transform = (
                    transforms[index]
                    if transforms
                    else _identity(work, (fixed_images or gt_fixed_seg)[index : index + 1])
                )
                runs: list[tuple[str, str, Path, Path, list[list[Path]] | None]] = []
                if fixed_images:
                    case_mask = [[mask[index]]] if mask else None
                    runs.append(
                        ("Image", "Evaluation_with_images.yml", fixed_images[index], moving_images[index], case_mask)
                    )
                if gt_fixed_seg:
                    runs.append(
                        ("Segmentation", "Evaluation_with_seg.yml", gt_fixed_seg[index], gt_moving_seg[index], None)
                    )
                if gt_fixed_fid:
                    fixed_count, moving_count = (
                        len(read_landmarks(path)) for path in (gt_fixed_fid[index], gt_moving_fid[index])
                    )
                    if fixed_count != moving_count:
                        raise EvaluatorError(
                            f"{fixed_count} fixed vs {moving_count} moving landmarks"
                            f" ('{gt_fixed_fid[index]}', '{gt_moving_fid[index]}').",
                            "Landmarks pair by row order: give both files the same points, in the same order.",
                        )
                    runs.append(
                        ("Landmarks", "Evaluation_with_fid.yml", gt_fixed_fid[index], gt_moving_fid[index], None)
                    )
                for modality, evaluation_file, fixed, moving, case_mask in runs:
                    app.evaluate(
                        inputs=[[transform]],
                        gt=[[fixed], [moving]],
                        output=eval_out / modality,
                        mask=case_mask,
                        evaluation_file=evaluation_file,
                        gpu=gpu,
                        cpu=cpu,
                        quiet=quiet,
                        tmp_dir=work / modality,
                    )
                    scored.append((f"P{index:03d}", eval_out / modality))
                # The transform alone, no reference: its Jacobian, bending energy and displacement lengths. A field is
                # measured on its own grid; a B-spline (elastix's native form) has none and is sampled on the fixed
                # one, the fixed image's or segmentation's; a linear transform's Jacobian is its matrix everywhere.
                if transforms:
                    field, grid = _is_field(transform), None
                    if not field and _is_bspline(transform):
                        grid = next((group[index] for group in (fixed_images, gt_fixed_seg) if group), None)
                        if grid is None:
                            print(
                                f"[ImpactReg] P{index:03d}: a B-spline transform is measured without reference on"
                                " the fixed grid; give the fixed image (-f/-m) or segmentation"
                                " (--gt-fixed-seg/--gt-moving-seg).",
                                flush=True,
                            )
                    if field or grid is not None:
                        if (transform, grid) in jacobians:
                            shutil.copytree(jacobians[(transform, grid)], eval_out / "NoReference", dirs_exist_ok=True)
                        else:
                            app.evaluate(
                                inputs=[[transform]],
                                gt=[[grid]] if grid is not None else None,
                                output=eval_out / "NoReference",
                                evaluation_file="Evaluation_no_reference.yml",
                                gpu=gpu,
                                cpu=cpu,
                                quiet=quiet,
                                tmp_dir=work / "NoReference",
                            )
                            jacobians[(transform, grid)] = eval_out / "NoReference"
                        scored.append((f"P{index:03d}", eval_out / "NoReference"))
                    elif not runs:
                        raise EvaluatorError(
                            f"P{index:03d}: nothing to measure on '{transform}'.",
                            "An affine has no measures without reference and a B-spline needs the fixed grid: give a"
                            " modality to score it on (image, segmentation or landmarks).",
                        )
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
        shape, attributes = _read_header(work / f"plane_{kind}", "Fixed", fixed)
        if not is_flat(shape):
            return self._warp_volume(work, kind, fixed, moving, transform_path, gpu, cpu, quiet, keep_form)
        # Through the 3-D stream one plane thick, as register runs a 2-D pair: see plane.py.
        planes = work / f"planes_{kind}"
        z_mm = float(min(attributes.get_np_array("Spacing")))
        moved = self._warp_volume(
            work,
            kind,
            lift_plane(fixed, planes / "Fixed", z_mm),
            lift_plane(moving, planes / "Moving", z_mm),
            lift_field(transform_path, planes) if transform_path and _is_flat_field(transform_path) else transform_path,
            gpu,
            cpu,
            quiet,
            keep_form,
        )
        target = moved.with_name(f"Moved{_form(moving) if keep_form else '.mha'}")
        project_image(moved, target)
        return target

    def _warp_volume(
        self,
        work: Path,
        kind: str,
        fixed: Path,
        moving: Path,
        transform_path: Path | None,
        gpu: list[int],
        cpu: int | None,
        quiet: bool,
        keep_form: bool,
    ) -> Path:
        """``_warp_onto_fixed`` on volumes."""
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
            if not _is_field(transform_path):
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
        return _the_output(out_root / "P000", "Moved")

    # --------------------------------------------------------------- uncertainty

    def uncertainty(
        self,
        stacks: list[Path],
        gt_fixed_fid: list[Path] = [],
        gt_moving_fid: list[Path] = [],
        gt_moving_seg: list[Path] = [],
        output: Path = Path("./Output").resolve(),
        gpu: list[int] = [],
        cpu: int | None = None,
        quiet: bool = False,
        tmp_dir: Path | None = None,
    ) -> None:
        """How much an ensemble's registrations disagree, from the stack ``register --uncertainty`` keeps
        (``<case>/InferenceStack.mha``, or ``.ome.zarr`` beside a store): konfai-apps' uncertainty of the stack through the configs this package ships.

        Always the spread of the displacement vectors, a map with its mean and 95th percentile (``Uncertainty.yml``);
        with the fixed landmarks, their spread through every member (``Uncertainty_with_fid.yml``), and with the
        moving ones too, whether that spread predicts the error (``Uncertainty_calibration.yml``); with the moving
        segmentation, how its propagations disagree (``Uncertainty_with_seg.yml``). Each option holds one entry per case
        or one for every case. Case ``P###`` is written to ``<output>/P###/Uncertainty/<measure>/`` (Map, Landmarks,
        Segmentation), and the whole call to ``<output>/Uncertainty_summary.json``.
        """
        app = KonfAIApp(str(_EVALUATION_APP), False, False)
        stacks, gt_fixed_fid, gt_moving_fid, gt_moving_seg = _broadcast(
            {
                "-i": _units(stacks),
                "--gt-fixed-fid": _units(gt_fixed_fid),
                "--gt-moving-fid": _units(gt_moving_fid),
                "--gt-moving-seg": _units(gt_moving_seg),
            }
        )
        if gt_moving_fid and not gt_fixed_fid:
            raise EvaluatorError(
                "--gt-moving-fid is given without --gt-fixed-fid.", "The spread is the fixed landmarks'."
            )
        scored: list[tuple[str, Path]] = []
        for index, stack in enumerate(stacks):
            out = output / f"P{index:03d}" / "Uncertainty"
            work = _work_dir(tmp_dir, output, "impact_reg_unc_")
            try:
                runs: list[tuple[str, str, list[list[Path]] | None]] = [("Map", "Uncertainty.yml", None)]
                if gt_fixed_fid:
                    landmarks = [[gt_fixed_fid[index]], *([[gt_moving_fid[index]]] if gt_moving_fid else [])]
                    config = "Uncertainty_calibration.yml" if gt_moving_fid else "Uncertainty_with_fid.yml"
                    runs.append(("Landmarks", config, landmarks))
                if gt_moving_seg:
                    runs.append(("Segmentation", "Uncertainty_with_seg.yml", [[gt_moving_seg[index]]]))
                for measure, uncertainty_file, gt in runs:
                    app.uncertainty(
                        inputs=[[stack]],
                        gt=gt,
                        output=out / measure,
                        uncertainty_file=uncertainty_file,
                        gpu=gpu,
                        cpu=cpu,
                        quiet=quiet,
                        tmp_dir=work / measure,
                    )
                    scored.append((f"P{index:03d}", out / measure))
            except BaseException as error:
                _leave(work, error)
                raise
            _leave(work, None)
        _write_evaluation_summary(output / "Uncertainty_summary.json", scored)

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
                written.append(_publish_output(moved, output / image.name, move=True))
        except BaseException as error:
            _leave(work, error)
            raise
        _leave(work, None)
        return written
