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
reduced by the combine operator; with ``-uncertainty`` the stack of them all is kept beside it, in the output's form
where that form keeps the member count, the members named in ``konfai-run.json`` in its order.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from konfai.data import reduction
from konfai.data.transform.ensemble import MEMBERS_ATTRIBUTE
from konfai.utils.dataset import Attribute, Dataset
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
    keeps the member count, as MetaImage otherwise (NIfTI's header does not, nor is a transform file a stack)."""
    return f"{STACK}{form if form in ('.mha', '.nrrd', '.ome.zarr') else '.mha'}"


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


def _member_count(path: Path) -> int:
    """How many members a file holds: a stack's count, read off its header (every channel without one); an output's
    one."""
    dataset, group, case = _entry(path)
    if group != STACK:
        return 1
    shape, header = dataset.get_infos(group, case)
    return int(float(header[MEMBERS_ATTRIBUTE])) if MEMBERS_ATTRIBUTE in header else int(shape[0])


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
) -> tuple[list[int], list[int]]:
    """Reduce every member ``sources`` hold (an output, or a stack of several) into ``output``, written through
    ``output_token``, slab by slab along the first spatial axis; and the stack of them all into ``stack`` when given,
    in the form its name asks (:func:`stack_file`), whose header keeps the member count. The members must share a grid and a channel count; the output
    and the stack take the first member's header. Returns the output's shape and each source's member count.

    Both are written in a folder of their own, then moved into place: written where they go, a folder that holds a
    store (an earlier run's OME-Zarr) would read as a folder of stores and turn the output into one."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".konfai_combine_", dir=output.parent) as staging:
        staged_output = Path(staging) / output.parent.name / output.name
        staged_stack = Path(staging) / output.parent.name / stack.name if stack is not None else None
        result = _combine_into(sources, operator, staged_output, output_token, staged_stack)
        for staged, final in ((staged_output, output), (staged_stack, stack)):
            if staged is None or final is None:
                continue
            final.parent.mkdir(parents=True, exist_ok=True)
            if final.is_dir():
                shutil.rmtree(final)
            elif final.exists():
                final.unlink()
            shutil.move(str(staged), str(final))
    return result


def _combine_into(
    sources: list[Path], operator: str, output: Path, output_token: str, stack: Path | None
) -> tuple[list[int], list[int]]:
    """:func:`combine`, written where it is told."""
    if operator not in OPERATORS:
        raise AppRepositoryError(f"An ensemble has no operator '{operator}'.", f"Choose one of {', '.join(OPERATORS)}.")
    entries = [_entry(path) for path in sources]
    counts = [_member_count(path) for path in sources]
    shapes = [dataset.get_infos(group, case)[0] for dataset, group, case in entries]
    channels = shapes[0][0] // counts[0]
    spatial = list(shapes[0][1:])
    for path, shape, count in zip(sources, shapes, counts, strict=True):
        if list(shape[1:]) != spatial or shape[0] != count * channels:
            raise AppRepositoryError(
                f"'{path}' holds {count} member(s) of {shape[0] // max(count, 1)} channel(s) on {list(shape[1:])}; the"
                f" ensemble's members are {channels} channel(s) on {spatial}.",
                "Ensemble apps whose outputs match: the same outputs, on the same grid.",
            )
    header = Attribute(entries[0][0].get_infos(entries[0][1], entries[0][2])[1])
    if MEMBERS_ATTRIBUTE in header:
        header.pop(MEMBERS_ATTRIBUTE)
    members = sum(counts)
    reduce = OPERATORS[operator]()
    sinks: dict[str, Any] = {}
    buffers: dict[str, list[np.ndarray]] = {"output": [], "stack": []}
    out_entry = _entry(output, output_token)
    stack_entry = _entry(stack, format_token(storage_form(stack))) if stack is not None else None

    def write(
        kind: str, entry: tuple[Dataset, str, str], data: np.ndarray, first: int, shape: list[int], attrs: Attribute
    ) -> None:
        dataset, group, case = entry
        if kind not in sinks:
            sinks[kind] = dataset.open_data_stream(group, case, shape, data.dtype, attrs)
        sink = sinks[kind]
        if sink is None:
            buffers[kind].append(data)
        else:
            sink.write_slice(
                (slice(None), slice(first, first + data.shape[1]), *(slice(None),) * (data.ndim - 2)), data
            )

    stack_header = Attribute(header)
    stack_header[MEMBERS_ATTRIBUTE] = members
    try:
        for first in range(0, spatial[0], SLAB_PLANES):
            region = (
                slice(None),
                slice(first, min(first + SLAB_PLANES, spatial[0])),
                *(slice(None),) * (len(spatial) - 1),
            )
            rows = [
                torch.from_numpy(np.asarray(dataset.read_data_slice(group, case, region)[0]))
                for dataset, group, case in entries
            ]
            per_member = torch.cat(rows).reshape(members, channels, *rows[0].shape[1:])
            reduced = reduce(list(per_member)).cpu().numpy()
            write("output", out_entry, reduced, first, [channels, *spatial], header)
            if stack_entry is not None:
                write(
                    "stack",
                    stack_entry,
                    torch.cat(rows).cpu().numpy(),
                    first,
                    [members * channels, *spatial],
                    stack_header,
                )
    except BaseException:
        for sink in sinks.values():
            if sink is not None:
                sink.abort()
        raise
    for kind, entry, attrs in (("output", out_entry, header), ("stack", stack_entry, stack_header)):
        if entry is None:
            continue
        if sinks.get(kind) is not None:
            sinks[kind].close()
        elif buffers[kind]:
            entry[0].write(entry[1], entry[2], np.concatenate(buffers[kind], axis=1), attrs)
    return [channels, *spatial], counts


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

    own_members = bool(infer.get("ensemble") or infer.get("ensemble_models") or infer.get("tta"))
    work = Path(tempfile.mkdtemp(prefix="konfai_ensemble_", dir=output.parent if output.parent.exists() else None))
    try:
        runs = []
        for index, reference in enumerate(references):
            run = work / f"member_{index:02d}"
            make_app(reference).infer(
                output=run, uncertainty=own_members, config_overrides=overrides_of(reference), **infer
            )
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
            _, counts = combine([found[key][1] for _, _, found in runs], operator, target, _token(first_output), stack)
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
        shutil.rmtree(work, ignore_errors=True)
