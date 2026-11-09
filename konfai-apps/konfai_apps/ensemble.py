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

"""Ensembles of app references, ``konfai-apps infer APP#a APP#b ...`` (docs/design/ensembles.md).

A member of an ensemble is one run of a model, whatever tells it from the others: a preset, a checkpoint, a TTA copy.
A reference's own checkpoints and TTA copies stay in its predictor run, which hands over its stack when it holds
several members and its output when one; each reference is a run of its own. The output is every member of every run
reduced by the combine operator (label maps, integer voxels, are voted whatever it is: a mean of labels is no label);
with ``-uncertainty`` the stack of them all is kept beside it, in the output's form where that form keeps the member
count, the members named in ``konfai-run.json`` in its order.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np
import torch
from konfai.data import reduction
from konfai.data.transform.ensemble import MEMBERS_ATTRIBUTE, holds_stack, member_count
from konfai.utils.dataset import DENSE_TRANSFORMS_ENV, Attribute, Dataset
from konfai.utils.utils import format_token, storage_form

from .errors import AppRepositoryError

STACK = "InferenceStack"
#: The operators a reference ensemble reduces its members with.
OPERATORS: dict[str, Callable[[], reduction.Reduction]] = {
    "Mean": reduction.Mean,
    "Median": reduction.Median,
    "Vote": reduction.Vote,
}
# ponytail: a fixed slab height along the first spatial axis; size it from the memory budget if members get wide.
SLAB_PLANES = 16


def stack_file(form: str) -> str:
    """The stack's file name beside an output of ``form`` (``.ome.zarr``, ``.nii.gz``): in that form where its header
    keeps the member count (:func:`~konfai.data.transform.ensemble.holds_stack`), as MetaImage otherwise. A case's
    ``.h5`` is a transform file."""
    token = "itktransform" if form == ".h5" else format_token(form, directory=not form)
    return f"{STACK}{form if holds_stack(token) else '.mha'}"


def _token(path: Path) -> str:
    """The format token a member file is read and written through: its own form (``nii.gz``, ``mha``), a store's
    or a transform file's backend."""
    from .app import KonfAIApp

    form = storage_form(path)
    backend = KonfAIApp._unit_backend(path, form)
    return backend if backend != "mha" else format_token(form)


def _group(path: Path) -> str:
    """The group a file or store holds: its name without its form (``InferenceStack`` of ``InferenceStack.ome.zarr``)."""
    form = storage_form(path)
    return path.name.removesuffix(form) if form else path.stem


def is_stack(path: Path) -> bool:
    """Whether ``path`` is an ensemble's stack, in whatever form."""
    return _group(path) == STACK


def _entry(path: Path, token: str | None = None) -> tuple[Dataset, str, str]:
    """``path`` (``<root>/<case>/<group>.<ext>``) as the dataset that holds it, its group and its case."""
    return Dataset(str(path.parent.parent), token or _token(path)), _group(path), path.parent.name


def _geometry(header: Attribute) -> np.ndarray | None:
    """A member's grid as one vector (origin, spacing, direction), None for a header that has none."""
    if not all(key in header for key in ("Origin", "Spacing", "Direction")):
        return None
    return np.concatenate(
        [header.get_np_array(key).astype(float).ravel() for key in ("Origin", "Spacing", "Direction")]
    )


def members_of(run: Path) -> dict[tuple[str, str, str], tuple[Path, Path]]:
    """The members one run hands over, by (output folder, case, output name): the output's file, and where its
    members are read, its stack where the run wrote one, the output itself otherwise. ``run`` is an ``infer`` output,
    ``<run>/<train name>/<output folder>/<case>/<name>.<ext>``."""
    from .app import KonfAIApp

    found: dict[tuple[str, str, str], tuple[Path, Path]] = {}
    for case_dir in sorted(path for path in run.glob("*/*/*") if path.is_dir()):
        files = [
            path
            for path in sorted(case_dir.iterdir())
            if KonfAIApp._match_supported(path) or KonfAIApp._directory_volume_suffix(path) is not None
        ]
        stack = next((path for path in files if is_stack(path)), None)
        for path in files:
            if path is stack:
                continue
            found[(case_dir.parent.name, case_dir.name, _group(path))] = (path, stack or path)
    return found


def combine(
    sources: list[Path], operator: str, output: Path, output_token: str, stack: Path | None = None
) -> list[int]:
    """Reduce every member ``sources`` hold (an output, or a stack of several) into ``output``, written through
    ``output_token``, slab by slab along the first spatial axis; and the stack of them all into ``stack`` (beside
    ``output``) when given, in the form its name asks (:func:`stack_file`), whose header keeps the member count. Label
    maps (integer voxels) are voted whatever ``operator`` says: a mean of labels is no label. The members must share a
    grid and a channel count; the output and the stack take the first member's header. Returns each source's member
    count.

    Both are written in a folder of their own, then moved into place with every file their writers left (a ``.mhd``'s
    ``.raw``): written where they go, a folder that holds a store (an earlier run's OME-Zarr) would read as a folder of
    stores and turn the output into one."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".konfai_combine_", dir=output.parent) as staging:
        staged_case = Path(staging) / output.parent.name
        result = _combine_into(
            sources, operator, staged_case / output.name, output_token, staged_case / stack.name if stack else None
        )
        for staged in staged_case.iterdir():
            final = output.parent / staged.name
            if final.is_dir():
                shutil.rmtree(final)
            elif final.exists():
                final.unlink()
            shutil.move(str(staged), str(final))
    return result


def _combine_into(sources: list[Path], operator: str, output: Path, output_token: str, stack: Path | None) -> list[int]:
    """:func:`combine`, written where it is told."""
    entries = [_entry(path) for path in sources]
    infos = [dataset.get_infos(group, case) for dataset, group, case in entries]
    # A stack's count is in its header (every channel a member without one); an output is one member.
    counts = [member_count(*info) if group == STACK else 1 for (_, group, _), info in zip(entries, infos, strict=True)]
    channels = infos[0][0][0] // counts[0]
    spatial = list(infos[0][0][1:])
    geometry = [_geometry(header) for _, header in infos]
    for path, (shape, _), count, grid in zip(sources, infos, counts, geometry, strict=True):
        if list(shape[1:]) != spatial or shape[0] != count * channels:
            raise AppRepositoryError(
                f"'{path}' holds {count} member(s) of {shape[0] // max(count, 1)} channel(s) on {list(shape[1:])}; the"
                f" ensemble's members are {channels} channel(s) on {spatial}.",
                "Ensemble apps whose outputs match: the same outputs, on the same grid.",
            )
        if grid is not None and geometry[0] is not None and not np.allclose(grid, geometry[0], atol=1e-5):
            raise AppRepositoryError(
                f"'{path}' lies on another grid than '{sources[0]}' (origin, spacing or direction): its voxels are not"
                " the same points.",
                "Ensemble apps whose outputs match: the same outputs, on the same grid.",
            )
    header = Attribute(infos[0][1])
    if MEMBERS_ATTRIBUTE in header:
        header.pop(MEMBERS_ATTRIBUTE)
    members = sum(counts)
    reduce: reduction.Reduction | None = None
    stack_header = Attribute(header)
    stack_header[MEMBERS_ATTRIBUTE] = members
    targets = [(_entry(output, output_token), [channels, *spatial], header)]
    if stack is not None:
        targets.append((_entry(stack, format_token(storage_form(stack))), [members * channels, *spatial], stack_header))
    sinks: list[Any] = [None] * len(targets)
    buffers: list[list[np.ndarray]] = [[] for _ in targets]
    with ExitStack() as opened:  # a stream finalised on success, aborted on an error (DataStream.__exit__)
        for first in range(0, spatial[0], SLAB_PLANES):
            region = (
                slice(None),
                slice(first, min(first + SLAB_PLANES, spatial[0])),
                *(slice(None),) * (len(spatial) - 1),
            )
            rows = torch.cat(
                [
                    torch.from_numpy(np.asarray(dataset.read_data_slice(group, case, region)[0]))
                    for dataset, group, case in entries
                ]
            )
            if reduce is None:  # label maps (integer voxels) vote, whatever the operator: a mean of labels is no label
                reduce = OPERATORS[operator if rows.is_floating_point() else "Vote"]()
            reduced = reduce(list(rows.reshape(members, channels, *rows.shape[1:]))).cpu().numpy()
            for index, ((dataset, group, case), shape, attrs) in enumerate(targets):
                data = reduced if index == 0 else rows.cpu().numpy()
                if first == 0:
                    sink = dataset.open_data_stream(group, case, shape, data.dtype, attrs)
                    sinks[index] = opened.enter_context(sink) if sink is not None else None
                if sinks[index] is None:
                    buffers[index].append(data)
                else:
                    sinks[index].write_slice(
                        (slice(None), slice(first, first + data.shape[1]), *(slice(None),) * (data.ndim - 2)), data
                    )
    for ((dataset, group, case), _, attrs), sink, buffer in zip(targets, sinks, buffers, strict=True):
        if sink is None:
            dataset.write(group, case, np.concatenate(buffer, axis=1), attrs)
    return counts


def infer_references(
    references: list[str],
    make_app: Callable[[str], Any],
    output: Path,
    operator: str,
    uncertainty: bool,
    overrides_of: Callable[[str], list[str] | None],
    **infer: Any,
) -> None:
    """``konfai-apps infer`` of several app references as one ensemble: each reference's run (its own checkpoints
    and TTA copies in its stack), then every member reduced by ``operator`` into ``<output>/<output folder>/<case>/``,
    with the stack of them all beside it under ``uncertainty``, and ``konfai-run.json`` naming the members."""
    from .app import RUN_RECORD

    work = Path(tempfile.mkdtemp(prefix="konfai_ensemble_", dir=output.parent if output.parent.exists() else None))
    # Every member is averaged: none keeps a native form (a registration's affine or B-spline), each writes its field.
    dense = os.environ.get(DENSE_TRANSFORMS_ENV)
    os.environ[DENSE_TRANSFORMS_ENV] = "1"
    try:
        runs = []
        for index, reference in enumerate(references):
            run = work / f"member_{index:02d}"
            # Its stack, always: checkpoints its app runs by default are members too (a run of one member writes none).
            make_app(reference).infer(output=run, uncertainty=True, config_overrides=overrides_of(reference), **infer)
            runs.append((reference, run, members_of(run)))
        keys = set(runs[0][2])
        for reference, _, found in runs[1:]:
            if set(found) != keys:
                raise AppRepositoryError(
                    f"'{reference}' writes {sorted(found)}; '{references[0]}' writes {sorted(keys)}.",
                    "Ensemble apps whose outputs match: the same outputs, for the same cases.",
                )
        names: list[str] = []
        for key in sorted(keys):
            folder, case, name = key
            first_output = runs[0][2][key][0]
            target = output / folder / case / f"{name}{storage_form(first_output)}"
            stack = output / folder / case / stack_file(storage_form(first_output)) if uncertainty else None
            counts = combine([found[key][1] for _, _, found in runs], operator, target, _token(first_output), stack)
            if not names:
                names = [
                    reference if count == 1 else f"{reference} [{member + 1}/{count}]"
                    for (reference, _, _), count in zip(runs, counts, strict=True)
                    for member in range(count)
                ]
            print(f"[KonfAI-Apps] {case}: {target}")
        records = [
            json.loads((run / RUN_RECORD).read_text(encoding="utf-8"))
            for _, run, _ in runs
            if (run / RUN_RECORD).is_file()
        ]
        output.mkdir(parents=True, exist_ok=True)
        (output / RUN_RECORD).write_text(
            json.dumps({"ensemble": records, "combine": operator, "members": names}, indent=2) + "\n", encoding="utf-8"
        )
    finally:
        if dense is None:
            os.environ.pop(DENSE_TRANSFORMS_ENV, None)
        else:
            os.environ[DENSE_TRANSFORMS_ENV] = dense
        shutil.rmtree(work, ignore_errors=True)
