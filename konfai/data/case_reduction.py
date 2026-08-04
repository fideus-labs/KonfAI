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

"""Reducing a group's cases into one entry, one region at a time.

A :class:`~konfai.data.patching.DatasetManager` IS one case: its name is the name it reads, the name
it writes, and the name every stage is handed. A reduction has no case name, so it is not a manager
-- it is a consumer of them, which is why it lives here and not in the patching module.

The loop is the per-case loop turned inside out. Instead of walking cases and, within a case, its
regions, it walks the OUTPUT's regions and, within a region, the cases: each reads that region
through its own chain (:meth:`~konfai.data.patching.DatasetManager.read_region`) and the operator
folds them. Peak memory is N regions, never N volumes -- and two regions, whatever N, once the
operator can accumulate.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field

import numpy as np
import torch

from konfai.data.patching import DatasetManager
from konfai.data.reduction import Reduction
from konfai.data.transform import LocalityKind, PatchLocality, Reduce, Transform
from konfai.utils.config import apply_config
from konfai.utils.dataset import (
    Attribute,
    Dataset,
    DataStream,
    _finalize_running_statistics,
    _update_running_statistics,
)
from konfai.utils.errors import ReductionError
from konfai.utils.utils import get_module

#: Geometry keys compared between cases under ``grid: strict``. Direction is in because a flipped
#: axis shows in neither extent nor spacing: averaging two volumes that disagree on it mirrors half
#: the cases into the other half, silently, and the result still looks like a volume.
_GEOMETRY_KEYS = ("Spacing", "Origin", "Direction")

#: What a stage placed AFTER the reduction may declare. Voxel-local is exact on a region; a
#: whole-volume statistic is exact too, because the engine seeds it with a pass of its own.
_POST_KINDS = frozenset({LocalityKind.POINTWISE, LocalityKind.GLOBAL_STAT})

#: Bytes per sample assumed when sizing regions from headers alone, before a dtype is known.
_ASSUMED_ITEMSIZE = 4

#: Keys the ``Reduce`` stage reads from its own mapping. An operator sharing one of these names
#: would silently be handed the stage's value, so the collision is refused instead.
_REDUCE_OWN_KEYS = frozenset({"operator", "output", "grid", "grid_tolerance", "provenance"})


@dataclass(frozen=True)
class ReductionPlan:
    """What the engine will do, before it does any of it."""

    output: str
    cases: list[str]
    spatial: list[int]
    channels: int
    slab_rows: int
    incremental: bool
    stat_pass: bool
    #: Channels a MEMBER's region carries. Separate from ``channels``, the output's, because an
    #: operator may change the count: ``Concat`` writes ``N x C`` where each member holds ``C``, so
    #: charging the members at the output's width over-states the peak by the cohort's size.
    source_channels: int = 0
    working_multiple: float = 0.0
    refusal: str | None = None

    @property
    def streams(self) -> bool:
        return self.refusal is None

    @property
    def buffered_regions(self) -> int:
        """Member regions resident at once: one for a running accumulator, else the whole cohort."""
        return 1 if self.incremental else len(self.cases)

    @property
    def resident_regions(self) -> float:
        """Regions held at the peak, counted in MEMBER regions plus the output's own.

        ``Median`` stacks the buffer into a new tensor and sorts a copy of that, so counting the
        buffer alone under-states its peak threefold -- and it is what a bare ``Reduce`` gets.
        """
        return self.buffered_regions * (1 + self.working_multiple) + 1

    def _region_bytes(self, channels: int) -> int:
        return int(self.slab_rows * np.prod(self.spatial[1:], dtype=np.int64) * channels * _ASSUMED_ITEMSIZE)

    @property
    def region_bytes(self) -> int:
        """One OUTPUT region, the unit the written slab is measured in."""
        return self._region_bytes(self.channels)

    @property
    def peak_bytes(self) -> int:
        # Members at their own width, the output at its, and whatever the operator builds over the
        # buffer it is handed -- which is member-sized, since that is what it was handed.
        #
        # A statistics pass is a second traversal, not a second working set: it holds exactly what
        # one region holds, so the peak is the same whether there are one or two passes.
        members = self.buffered_regions * self._region_bytes(self.source_channels or self.channels)
        return int(members * (1 + self.working_multiple) + self.region_bytes)

    def describe(self) -> str:
        verdict = "STREAM" if self.streams else "REFUSED"
        lines = [f"REDUCE {len(self.cases)} case(s) -> 1 output '{self.output}' -- {verdict}"]
        if not self.streams:
            lines.append(f"    refused: {self.refusal}")
            return "\n".join(lines)
        regime = "incremental accumulator" if self.incremental else "every case resident per region"
        lines.append(
            f"    {self.resident_regions:g} resident region(s) of {self.slab_rows} row(s)"
            f" = {self.peak_bytes / (1 << 30):.2f} GiB  ({regime})"
        )
        if self.stat_pass:
            lines.append("    two passes: the first seeds the whole-volume statistics the chain asks of the RESULT")
        lines.append(f"    cases: {', '.join(self.cases)}")
        return "\n".join(lines)


@dataclass
class _RunningStatistics:
    """Min/Max/Mean/Std accumulated over regions, so the volume is never resident.

    The store-scan recurrence (:func:`konfai.utils.dataset._update_running_statistics`) is the one
    Welford kernel; this feeds it blocks and writes the keys in KonfAI's own spelling.
    """

    _state: dict | None = None

    def update(self, block: torch.Tensor) -> None:
        self._state = _update_running_statistics(self._state, block.detach().cpu().numpy().reshape(1, -1))

    def write_into(self, attribute: Attribute) -> None:
        """Seed the attribute the way the rest of KonfAI already spells these keys: Min/Max bare
        scalars, Mean/Std one-element arrays. A second convention reads back as a string and fails
        inside whichever transform consumed it."""
        if self._state is None or not self._state["count"]:
            raise ReductionError("Statistics were requested over an empty volume.", "Check the output extent.")
        statistics = _finalize_running_statistics(self._state)
        attribute["Min"] = np.float32(statistics["min"])
        attribute["Max"] = np.float32(statistics["max"])
        attribute["Mean"] = np.asarray([statistics["mean"]], dtype=np.float32)
        attribute["Std"] = np.asarray([statistics["std"]], dtype=np.float32)


def split_chain(transforms: list[Transform]) -> tuple[list[Transform], Reduce | None, list[Transform]]:
    """A chain around its ``Reduce``: what runs per case, the stage itself, what runs on the result."""
    for index, transform in enumerate(transforms):
        if isinstance(transform, Reduce):
            return list(transforms[:index]), transform, list(transforms[index + 1 :])
    return list(transforms), None, []


def resolve_operator(reduce: Reduce) -> Reduction:
    """The operator the stage names, as an instance, refusing one that cannot fold a region.

    Configured like every other extension point: its constructor arguments are bound from the same
    mapping the ``Reduce`` itself was read from, so a custom operator takes parameters exactly as a
    custom transform or a custom draw does::

        Reduce:
          operator: mypkg:TrimmedMean
          output: template
          trim: 0.2            # the operator's own parameter

    A chain assembled in Python has no configuration to read, and the operator is then built from
    its own defaults.
    """
    module, name = get_module(reduce.operator_classpath, "konfai.data.reduction")
    factory = getattr(module, name)
    shadowed = sorted(set(inspect.signature(factory.__init__).parameters) & _REDUCE_OWN_KEYS)
    if shadowed:
        raise ReductionError(
            f"'{reduce.operator_classpath}' has parameter(s) {shadowed}, which the Reduce stage"
            " reads for itself from the same mapping.",
            f"Rename them: {sorted(_REDUCE_OWN_KEYS)} belong to Reduce, everything else in the"
            " mapping is the operator's.",
        )
    try:
        operator = apply_config(reduce.konfai_args)(factory)()
    except TypeError as error:
        raise ReductionError(
            f"'{reduce.operator_classpath}' could not be built: {error}.",
            "Give its parameters under the Reduce stage, next to 'operator' and 'output', or give them defaults.",
        ) from error
    if not isinstance(operator, Reduction):
        raise ReductionError(
            f"'{reduce.operator_classpath}' is not a Reduction.",
            "Subclass konfai.data.reduction.Reduction, or use Mean / Median / Concat.",
        )
    if not operator.voxel_local:
        raise ReductionError(
            f"'{reduce.operator_classpath}' does not declare itself voxel-local, so it cannot be"
            " folded one region at a time.",
            "Set voxel_local = True when every output voxel depends only on the same voxel of each"
            " case; an operator reading across space cannot stream.",
        )
    return operator


def check_post_stages(post: list[Transform], output: str) -> None:
    """What may follow a ``Reduce`` in the same chain.

    Each stage after the reduction is handed ONE REGION of the result. A stage reading across space
    -- a halo, a resample, a reorientation -- would take that region for the whole volume and seam
    at every boundary: a plausible result, and a wrong one. Those are deferred, not forbidden:
    end the chain, and read the written volume back in a second chain where the ordinary planner can
    pull regions through it.

    A statistic may follow the reduction, but only over stages that leave the values alone: the stat
    pass measures the FOLD, so an earlier stage that changes the values makes the seed describe a
    volume nobody wrote. This mirrors the same refusal in the per-case planner.
    """
    localities: list[PatchLocality] = []
    for index, stage in enumerate(post):
        locality = stage.patch_locality(Attribute())
        kind = locality.kind
        name = type(stage).__name__
        if kind not in _POST_KINDS:
            raise ReductionError(
                f"stage {index} '{name}' follows the Reduce into '{output}' and declares {kind.name},"
                " which reads across space -- applied one region at a time it would seam at every"
                " region boundary.",
                f"Only voxel-local stages can follow a reduction. End this chain, and put '{name}' in a"
                f" second chain that reads '{output}' back.",
            )
        if kind is LocalityKind.GLOBAL_STAT and not all(previous.statistics_preserving for previous in localities):
            raise ReductionError(
                f"stage {index} '{name}' follows the Reduce into '{output}' and needs whole-volume"
                " statistics, but an earlier stage after the Reduce changes the values -- the"
                " statistic is measured on the fold, so it would not be this stage's input.",
                f"End this chain after the value-changing stage, and put '{name}' in a second chain"
                f" that reads '{output}' back, where its statistic is measured on what it receives.",
            )
        localities.append(locality)


@dataclass
class CaseReduction:
    """Fold every case of a group into one entry, region by region.

    It uses only the public read side of each case's manager -- the streaming machinery already
    planned, accepted or refused per case -- and owns the write side itself, under the output name
    the chain declared.
    """

    managers: list[DatasetManager]
    reduce: Reduce
    post: list[Transform]
    destination: Dataset
    group: str
    slab_rows: int = 64
    operator: Reduction = field(init=False)

    def __post_init__(self) -> None:
        if not self.managers:
            raise ReductionError(
                f"The chain reducing into '{self.reduce.output}' has no case to fold.",
                "Check the dataset and its subset: a reduction over nothing has no result.",
            )
        self.slab_rows = max(1, int(self.slab_rows))
        self.operator = resolve_operator(self.reduce)
        check_post_stages(self.post, self.reduce.output)

    def fit_budget(self, budget_bytes: float | None, cap: int = 64) -> None:
        """Size the regions so the resident ones fit ``budget_bytes``.

        The default region height is safe for one case and not for N: a reduction holds one region
        PER CASE, so the constant that bounds a per-case sweep is off by the number of cases here.
        Half the budget, because the operator's own output and the write buffer live alongside.
        Below one row nothing fits; the plan then reports a peak above the budget and the workflow
        refuses, which is the only honest answer -- there is no whole-volume path to fall back to.

        The budget also goes to the cases' own managers, because a chain crossing a ``Save`` sweeps
        that cache when first read, and ``read_region`` carries no budget of its own.
        """
        for manager in self.managers:
            manager.set_memory_budget(budget_bytes)
        if not budget_bytes or budget_bytes <= 0:
            return
        plan = self.plan()
        row_bytes = plan.resident_regions * plan.region_bytes / max(1, plan.slab_rows)
        if row_bytes <= 0:
            return
        self.slab_rows = max(1, min(cap, int(budget_bytes * 0.5 / row_bytes)))

    # ---------------------------------------------------------------- planning

    @property
    def reference(self) -> DatasetManager:
        """The case whose geometry the output adopts."""
        if not self.reduce.grid.startswith("reference:"):
            return self.managers[0]
        wanted = self.reduce.grid.split(":", 1)[1]
        for manager in self.managers:
            if manager.name == wanted:
                return manager
        raise ReductionError(
            f"grid 'reference:{wanted}' names a case that is not being reduced.",
            f"The cases are: {', '.join(manager.name for manager in self.managers)}.",
        )

    def check_grid(self) -> str | None:
        """Whether the cases agree enough to be folded, or why they do not.

        Compared on the grid each case's chain LANDS on — the folded shape and the plan-evolved
        geometry — not the stored one: the chain exists precisely to bring disagreeing members onto
        one grid (a ``Resample`` before the ``Reduce``), and comparing what is on disk would refuse
        the very cohorts the reduction is for. Read from headers and plans alone, so it costs
        nothing and happens before the first byte. Nothing anywhere can verify that the members
        truly share a space, only that they claim to.

        Compared against :attr:`reference`, the case whose geometry the output adopts, and only
        ``strict`` compares geometry at all: naming a reference is how a cohort says its members
        disagree on their headers and which one to believe, so demanding they agree would refuse
        every cohort the policy exists for.
        """
        reference = self.reference
        others = [manager for manager in self.managers if manager is not reference]
        for manager in others:
            if list(manager.spatial_shape) != list(reference.spatial_shape):
                return (
                    f"case '{manager.name}' lands on extent {list(manager.spatial_shape)}"
                    f" where '{reference.name}' lands on {list(reference.spatial_shape)}"
                )
        if self.reduce.grid != "strict":
            return None
        expected = reference.landed_attributes()
        for manager in others:
            attribute = manager.landed_attributes()
            for key in _GEOMETRY_KEYS:
                # ``strict`` is a promise that the geometries WERE compared, and a key nobody
                # recorded cannot be. Skipping it is quietest exactly where it costs most: a
                # Direction missing from one header is a flip that shows in neither extent nor
                # spacing. Fold on extent alone with 'grid: shape_only' if that is what is meant.
                absent = [
                    name for name, side in ((reference.name, expected), (manager.name, attribute)) if key not in side
                ]
                if absent:
                    return (
                        f"{' and '.join(repr(name) for name in absent)} lands on no {key},"
                        f" which 'grid: strict' compares (use 'grid: shape_only' to fold on extent alone)"
                    )
                left = np.asarray(expected.get_np_array(key), dtype=np.float64).ravel()
                right = np.asarray(attribute.get_np_array(key), dtype=np.float64).ravel()
                if left.shape != right.shape or not np.allclose(left, right, atol=self.reduce.grid_tolerance):
                    return (
                        f"case '{manager.name}' lands on {key} {right.tolist()} where '{reference.name}'"
                        f" lands on {left.tolist()} (grid: strict, tolerance {self.reduce.grid_tolerance})"
                    )
        return None

    def _first_refusal(self) -> str | None:
        """The first reason this reduction cannot stream: a disagreeing grid, or a case that refuses."""
        refusal = self.check_grid()
        if refusal is not None:
            return refusal
        for manager in self.managers:
            case_refusal = manager.stream_refusal(0, apply_augmentations=False)
            if case_refusal is not None:
                return f"case '{manager.name}': {case_refusal}"
        return None

    def _needs_stat_pass(self) -> bool:
        """Whether a stage after the reduction wants whole-volume statistics OF THE RESULT.

        Those cannot be seeded from disk the usual way -- the reduced volume is stored nowhere yet --
        so the engine computes them with a pass of its own. Twice the reads, no intermediate volume,
        and the reduction is deterministic so both passes see the same values.
        """
        return any(stage.patch_locality(Attribute()).kind is LocalityKind.GLOBAL_STAT for stage in self.post)

    def plan(self) -> ReductionPlan:
        reference = self.reference
        return ReductionPlan(
            output=self.reduce.output,
            cases=[manager.name for manager in self.managers],
            spatial=reference.spatial_shape,
            # The operator's own channel map, because only it knows the result's leading axis: a
            # Concat over N cases writes N times the channels, and the plan must probe and size the
            # shape the run will actually open.
            channels=self.operator.output_channels(int(reference.base_shape[0]), len(self.managers)),
            source_channels=int(reference.base_shape[0]),
            slab_rows=self.slab_rows,
            incremental=self.operator.incremental,
            working_multiple=float(self.operator.working_multiple),
            stat_pass=self._needs_stat_pass(),
            refusal=self._first_refusal(),
        )

    # --------------------------------------------------------------- execution

    def _regions(self, spatial: list[int]) -> list[tuple[slice, ...]]:
        """The output's regions: slabs along the first spatial axis, whole in the others."""
        return [
            (slice(start, min(start + self.slab_rows, spatial[0])), *(slice(0, extent) for extent in spatial[1:]))
            for start in range(0, spatial[0], self.slab_rows)
        ]

    def _fold(self, region: tuple[slice, ...]) -> torch.Tensor:
        """One region of the reduced volume: every case reads that region, the operator folds them.

        Each region is presented as ``[1, C, *spatial]`` -- the stack-axis layout every operator is
        written against (see :class:`~konfai.data.reduction.Reduction`) -- and the result comes back
        without it. The axis matters to any operator that places things side by side.
        """
        self.operator.start()
        for manager in self.managers:
            self.operator.accumulate(manager.read_region(region).unsqueeze(0))
        return self.operator.finalize().squeeze(0)

    def _apply_post(self, block: torch.Tensor, attribute: Attribute, rank: int) -> np.ndarray:
        scope = Attribute(attribute)
        for stage in self.post:
            block = stage(self.reduce.output, block, scope)
        array = block.numpy()
        if array.ndim != rank:
            raise ReductionError(
                f"A stage after the Reduce returned a rank-{array.ndim} region where the"
                f" channel-first layout needs rank {rank}.",
                "A transform folding the leading axis must keep it (`keepdim=True`).",
            )
        return array

    def _output_attributes(self, plan: ReductionPlan) -> Attribute:
        # The header is the geometry the reference's chain LANDS on, not the geometry it was stored
        # with: a cohort resampled onto a template grid by its pre-chain must publish that grid --
        # seeding the source's own Spacing here would stamp a wrong header on every round of an
        # atlas build, and nothing about the volume would look wrong.
        attribute = self.reference.landed_attributes()
        if self.reduce.provenance:
            # The deliverable carries its own recipe. A set of cases that changed between two runs
            # would otherwise write a different volume under the same name -- the worst way this can
            # fail, because nothing about the output looks wrong.
            attribute["konfai_reduce_operator"] = self.reduce.operator_classpath
            attribute["konfai_reduce_cases"] = "|".join(plan.cases)
        if plan.stat_pass:
            statistics = _RunningStatistics()
            for region in self._regions(plan.spatial):
                statistics.update(self._fold(region))
            statistics.write_into(attribute)
        return attribute

    def _open_stream(self, spatial: list[int], array: np.ndarray, attribute: Attribute) -> DataStream:
        stream = self.destination.open_data_stream(
            self.group,
            self.reduce.output,
            [int(array.shape[0]), *spatial],
            array.dtype,
            attribute,
            region_shape=[int(array.shape[0]), self.slab_rows, *spatial[1:]],
        )
        if stream is None:
            raise ReductionError(
                f"'{self.destination.filename}' cannot serve region writes, so the reduced volume"
                f" '{self.reduce.output}' could only be written by assembling it in memory.",
                "Write the reduction to an h5 or omezarr destination.",
            )
        return stream

    def materialize(self, rewrite: bool = False) -> bool:
        """Write the reduced entry, or raise saying why it cannot be written this way.

        There is no whole-volume fallback: assembling every case is exactly what this exists to
        avoid, and doing it silently would turn a bounded run into an unannounced OOM. A finished
        output is left alone unless ``rewrite``, which is the resume.
        """
        if not rewrite and self.destination.is_dataset_exist(self.group, self.reduce.output):
            return True
        plan = self.plan()
        if not plan.streams:
            # A grid disagreement gets its own remedy: a Save changes nothing about the grids, so
            # the generic advice would send the reader in a circle.
            remedy = (
                "The members do not land on one grid: resample them onto a common grid before the"
                " Reduce, or declare grid: reference:<case> / shape_only if the cohort is already"
                " aligned."
                if self.check_grid() is not None
                else "A reduction has no whole-volume fallback. Fix the refusing stage, or put a Save"
                " before the Reduce so each case's chain is materialized first."
            )
            raise ReductionError(f"The reduction into '{self.reduce.output}' cannot stream: {plan.refusal}.", remedy)

        spatial = plan.spatial
        attribute = self._output_attributes(plan)
        rank = len(spatial) + 1
        stream: DataStream | None = None
        try:
            for region in self._regions(spatial):
                array = self._apply_post(self._fold(region), attribute, rank)
                if stream is None:
                    stream = self._open_stream(spatial, array, attribute)
                    stream.__enter__()
                stream.write_slice((slice(0, int(array.shape[0])), *region), array)
            if stream is not None:
                stream.close()
        except Exception as exception:
            if stream is not None:
                stream.abort(exception)
            raise
        return True
