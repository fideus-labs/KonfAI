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

The loop walks the OUTPUT's regions and, within a region, the cases: each reads that region through
its own chain (:meth:`~konfai.data.patching.DatasetManager.read_region`) and the operator folds them.
Peak memory is N regions, never N volumes, and two regions, whatever N, once the operator can
accumulate.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field, replace

import numpy as np
import torch

from konfai.data.patching import (
    SWEEP_CLOCK,
    SWEEP_SLAB_ROWS,
    DatasetManager,
    HeldMeter,
    RegionWriter,
    device_capped_budget,
    open_held_meter,
    save_destination,
)
from konfai.data.patching.budget import _START_SHARE, GROWTH_CAP_UNITS, RegionGrowth, device_signals_oom
from konfai.data.reduction import Reduction
from konfai.data.transform import LocalityKind, PatchLocality, Reduce, Save, Transform, stat_seed_valid
from konfai.utils.budget import budget_share
from konfai.utils.dataset import Attribute, Dataset, DataStream
from konfai.utils.dataset.statistics import _finalize_running_statistics, _update_running_statistics
from konfai.utils.errors import ReductionError

#: Geometry keys compared between cases under ``grid: strict``. Direction is in: a flipped axis
#: shows in neither extent nor spacing.
_GEOMETRY_KEYS = ("Spacing", "Origin", "Direction")

#: What a stage placed AFTER the reduction may declare. Voxel-local is exact on a region; a
#: whole-volume statistic is seeded by a pass of its own.
_POST_KINDS = frozenset({LocalityKind.POINTWISE, LocalityKind.GLOBAL_STAT})

#: Bytes per sample assumed when sizing regions from headers alone, before a dtype is known.
_ASSUMED_ITEMSIZE = 4

#: How much of the regions' own share the folds a stat pass KEEPS may take, the regions getting the
#: rest (:data:`~konfai.utils.budget.BUDGET_SHARES`).
_KEPT_FOLDS_SHARE_OF_REGIONS = 0.5


@contextlib.contextmanager
def _awaited(phase: str) -> Iterator[None]:
    """A phase the fold both performs and stands still for: the store's seconds are the loop's."""
    with SWEEP_CLOCK.phase(f"wait({phase})"), SWEEP_CLOCK.phase(phase):
        yield


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
    #: Channels a MEMBER's region carries, separate from ``channels``, the output's: an operator may
    #: change the count (``Concat`` writes ``N x C`` where each member holds ``C``).
    source_channels: int = 0
    working_multiple: float = 0.0
    #: Volumes-worth the MEMBER CHAIN allocates beside the region it is producing
    #: (``DatasetManager.working_multiple()``, the largest ``Transform.working_multiple`` on the
    #: chain). Distinct from ``working_multiple``, the OPERATOR's. Charged ONCE whatever the
    #: cohort's size: ``_fold`` accumulates the members one after another, so one chain replays.
    chain_multiple: float = 0.0
    #: The source window ONE member's region pulls (:attr:`~konfai.data.patching.BlockReads`
    #: ``widest_pull``). Charged ONCE: the members are folded in turn, so one chain pulls at a time.
    #: Zero for a chain whose region is its own source.
    pull_bytes: int = 0
    #: What ONE member's region makes the store decode ABOVE the window it asked for
    #: (:meth:`~konfai.data.patching.DatasetManager.region_reads`). A chunked backend decodes whole
    #: blocks, so below one stored block this is the same figure at every height: charged flat,
    #: never divided by the rows.
    read_bytes: int = 0
    #: Members read from a store that cannot serve a bounded region read (a gzipped NIfTI, a
    #: compressed MetaImage, NRRD), by name, with the store's format: every region asked of such a
    #: member decodes its whole volume, so the fold reads it once per region rather than once.
    unbounded: dict[str, str] = field(default_factory=dict)
    refusal: str | None = None

    @property
    def streams(self) -> bool:
        return self.refusal is None

    @property
    def regions(self) -> int:
        """Output regions the fold walks: slabs of ``slab_rows`` along the first spatial axis."""
        return max(1, -(-int(self.spatial[0]) // max(1, self.slab_rows)))

    @property
    def passes(self) -> int:
        """Traversals of the cohort: one, or two when a statistic of the result is seeded first."""
        return 2 if self.stat_pass else 1

    @property
    def read_factor(self) -> float:
        """How many times a member's source is read in full, priced from the plan alone.

        A store serving bounded region reads is read once per pass; one that cannot, once per region
        and per pass. The figure of the worst member; ``unbounded`` names the members it applies to.
        """
        return float(self.passes * (self.regions if self.unbounded else 1))

    @property
    def buffered_regions(self) -> int:
        """Member regions resident at once: one for a running accumulator, else the whole cohort."""
        return 1 if self.incremental else len(self.cases)

    @property
    def resident_regions(self) -> float:
        """Regions held at the peak, in member regions: the buffer, ``working_multiple`` buffers-worth
        over it, ``chain_multiple`` for the replaying chain, and the output's own. A count for
        ``describe``; ``peak_bytes`` is the figure the plan sizes by."""
        return self.buffered_regions * (1 + self.working_multiple) + self.chain_multiple + 1

    def _region_bytes(self, channels: int) -> int:
        return int(self.slab_rows * np.prod(self.spatial[1:], dtype=np.int64) * channels * _ASSUMED_ITEMSIZE)

    @property
    def region_bytes(self) -> int:
        """One OUTPUT region, the unit the written slab is measured in."""
        return self._region_bytes(self.channels)

    @property
    def peak_bytes(self) -> int:
        # Members at their own width, the output at its, and whatever the operator builds over the
        # buffer. A statistics pass is a second traversal, not a second working set.
        member_bytes = self._region_bytes(self.source_channels or self.channels)
        members = self.buffered_regions * member_bytes
        # Beside the buffered regions, the one replaying chain holds the source window it pulled,
        # its own working buffers (over the larger of the window and the region it lands) and what
        # the store decoded above that window. All three are charged once: one chain runs at a time.
        return int(
            members * (1 + self.working_multiple)
            + self.chain_multiple * max(member_bytes, self.pull_bytes)
            + self.pull_bytes
            + self.region_bytes
            + self.read_bytes
        )

    def describe(self) -> str:
        verdict = "STREAM" if self.streams else "REFUSED"
        header = f"REDUCE {len(self.cases)} case(s) -> 1 output '{self.output}': {verdict}"
        if not self.streams:
            return "\n".join([header, f"    refused: {self.refusal}"])
        return "\n".join(
            [header, *(f"    {line}" for line in self.body_lines()), f"    cases: {', '.join(self.cases)}"]
        )

    def body_lines(self) -> list[str]:
        """What a streaming plan says of itself, between its header and its case list: the regions it
        holds, its passes, and the members it decodes whole."""
        regime = "incremental accumulator" if self.incremental else "every case resident per region"
        lines = [
            f"{self.resident_regions:g} resident region(s) of {self.slab_rows} row(s)"
            f" = {self.peak_bytes / (1 << 30):.2f} GiB  ({regime})"
        ]
        if self.stat_pass:
            lines.append("two passes: the first seeds the whole-volume statistics the chain asks of the RESULT")
        if self.unbounded and self.read_factor > 1:
            formats = ", ".join(sorted(set(self.unbounded.values())))
            per = "one per region and per pass" if self.stat_pass else "one per region"
            lines.append(
                f"reads: {len(self.unbounded)} of {len(self.cases)} member(s) sit on {formats}, which decodes"
                f" the whole volume behind every region read: {self.read_factor:g} decodes per member ({per}),"
                f" {self.read_factor * len(self.unbounded):g} in all"
            )
            lines.append("put a Save ...:h5 before the Reduce so each member is materialized on a bounded store first")
        return lines


@dataclass
class _RunningStatistics:
    """Min/Max/Mean/Std accumulated over regions, so the volume is never resident.

    Feeds blocks to :func:`konfai.utils.dataset.statistics._update_running_statistics` and writes the
    keys in KonfAI's own spelling.
    """

    _state: dict | None = None

    def update(self, block: torch.Tensor) -> None:
        self._state = _update_running_statistics(self._state, block.detach().cpu().numpy().reshape(1, -1))

    def write_into(self, attribute: Attribute) -> None:
        """Seed the attribute the way the rest of KonfAI spells these keys: Min/Max bare scalars,
        Mean/Std one-element arrays."""
        if self._state is None or not self._state["count"]:
            raise ReductionError("Statistics were requested over an empty volume.", "Check the output extent.")
        statistics = _finalize_running_statistics(self._state)
        attribute["StatisticsSeeded"] = np.float32(1.0)
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


def check_post_stages(post: list[Transform], output: str) -> None:
    """What may follow a ``Reduce`` in the same chain.

    Each stage after the reduction is handed ONE REGION of the result, so only voxel-local stages may
    follow: a stage reading across space would seam at every region boundary. End the chain and read
    the written volume back in a second one instead.

    A statistic may follow the reduction, but only over stages that leave the values alone: the stat
    pass measures the FOLD (``stat_seed_valid``, the per-case planner's rule).
    """
    localities: list[PatchLocality] = []
    for index, stage in enumerate(post):
        locality = stage.patch_locality(Attribute())
        kind = locality.kind
        name = type(stage).__name__
        if kind not in _POST_KINDS:
            raise ReductionError(
                f"stage {index} '{name}' follows the Reduce into '{output}' and declares {kind.name},"
                " which reads across space: applied one region at a time it would seam at every"
                " region boundary.",
                f"Only voxel-local stages can follow a reduction. End this chain, and put '{name}' in a"
                f" second chain that reads '{output}' back.",
            )
        if kind is LocalityKind.GLOBAL_STAT and not stat_seed_valid(localities):
            raise ReductionError(
                f"stage {index} '{name}' follows the Reduce into '{output}' and needs whole-volume"
                " statistics, but an earlier stage after the Reduce changes the values: the"
                " statistic is measured on the fold, so it would not be this stage's input.",
                f"End this chain after the value-changing stage, and put '{name}' in a second chain"
                f" that reads '{output}' back, where its statistic is measured on what it receives.",
            )
        localities.append(locality)


@dataclass
class CaseReduction:
    """Fold every case of a group into one entry, region by region.

    It uses only the public read side of each case's manager and owns the write side itself, under
    the output name the chain declared.
    """

    managers: list[DatasetManager]
    reduce: Reduce
    post: list[Transform]
    destination: Dataset
    group: str
    slab_rows: int = 64
    operator: Reduction = field(init=False)
    #: The budget the last fit sized against, in host bytes: what a run-time device re-cap re-fits.
    _budget_bytes: float | None = field(init=False, default=None)
    _kept_folds: list | None = field(init=False, default=None, repr=False, compare=False)
    #: The caller's ceiling on the region height, when it named one.
    _cap: int | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if not self.managers:
            raise ReductionError(
                f"The chain reducing into '{self.reduce.output}' has no case to fold.",
                "Check the dataset and its subset: a reduction over nothing has no result.",
            )
        self.slab_rows = max(1, int(self.slab_rows))
        self.operator = self.reduce.operator
        check_post_stages(self.post, self.reduce.output)

    def fit_budget(self, budget_bytes: float | None, cap: int | None = None) -> None:
        """Size the FIRST region so the resident ones fit ``budget_bytes``; the rest follow what
        the regions hold (:meth:`_folds`).

        The first region starts at ``_START_SHARE`` of the regions' share; ``cap`` bounds the
        growth, ``GROWTH_CAP_UNITS`` slabs by default. Below one row nothing fits, the plan then
        reports a peak above the budget and the workflow refuses: there is no whole-volume path.

        The budget also goes to the cases' own managers: a chain crossing a ``Save`` sweeps that
        cache when first read, and ``read_region`` carries no budget of its own.
        """
        self._budget_bytes = budget_bytes
        self._cap = None if cap is None else max(1, int(cap))
        # THE REMAINDER, not the whole figure: a member's chain holds what it holds WHILE the fold
        # is holding its regions.
        chains = budget_share("chains", budget_bytes)
        for manager in self.managers:
            manager.set_memory_budget(budget_bytes if chains is None else chains)
        if not budget_bytes or budget_bytes <= 0:
            return
        plan = self.plan()
        # What the fold's own share leaves the regions: the folds it keeps live beside them for the
        # whole write pass. Decided here, since the regions are sized against it.
        allowance = budget_share("regions", budget_bytes) or 0.0
        if self.keeps_folds(plan):
            allowance -= self._folded_output_bytes(plan)
        self.slab_rows = self._start_rows(plan, allowance)

    def _cap_rows(self, spatial: Sequence[int]) -> int:
        """The tallest region the growth reaches: the caller's cap, else ``GROWTH_CAP_UNITS`` slabs,
        and the output's own height at most."""
        cap = self._cap if self._cap is not None else GROWTH_CAP_UNITS * SWEEP_SLAB_ROWS
        return max(1, min(int(spatial[0]), cap))

    def _start_rows(self, plan: ReductionPlan, allowance: float) -> int:
        """Where the growth starts: the tallest height up to the cap whose PRICED plan fits
        ``_START_SHARE`` of ``allowance``, failing that ``allowance`` itself; one row when nothing
        fits. Bisected on the price itself: none of what a region costs scales with its rows."""
        ceiling = self._cap_rows(plan.spatial)
        for share in (_START_SHARE, 1.0):
            allowed = allowance * share
            if self._priced_peak(plan, 1) > allowed:
                continue
            low, high = 1, ceiling
            while low < high:
                middle = (low + high + 1) // 2
                if self._priced_peak(plan, middle) <= allowed:
                    low = middle
                else:
                    high = middle - 1
            return low
        return 1

    def _priced_peak(self, plan: ReductionPlan, rows: int) -> int:
        """What ``plan`` prices at ``rows``, leaving the height the sizing is working from alone.
        Only the read fields depend on the height, so only they are recomputed per probe."""
        held = self.slab_rows
        try:
            self.slab_rows = rows
            pull_bytes, read_bytes = self._member_read_bytes(int(plan.source_channels))
        finally:
            self.slab_rows = held
        return replace(plan, slab_rows=rows, pull_bytes=pull_bytes, read_bytes=read_bytes).peak_bytes

    def keeps_folds(self, plan: ReductionPlan) -> bool:
        """Whether the stat pass hands its folds to the write pass instead of re-folding them.

        One rule, two callers: the sizing subtracts what they will hold, the stat pass fills them.
        They may take their share of what the FOLD holds, never of the whole declaration.
        """
        if not plan.stat_pass or not self._budget_bytes or self._budget_bytes <= 0:
            return False
        regions = budget_share("regions", self._budget_bytes)
        return regions is not None and self._folded_output_bytes(plan) <= regions * _KEPT_FOLDS_SHARE_OF_REGIONS

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

        Compared on the grid each case's chain LANDS on, not the stored one, and from headers and
        plans alone, before the first byte. Nothing can verify that the members truly share a space,
        only that they claim to.

        Compared against :attr:`reference`, the case whose geometry the output adopts; only
        ``strict`` compares geometry at all.
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
                # recorded cannot be. Fold on extent alone with 'grid: shape_only' instead.
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

    @staticmethod
    def _member_source(manager: DatasetManager) -> tuple[Dataset, str]:
        """The store and group a member's regions are read from: its own, or its last ``Save``'s cache."""
        saves = [stage for stage in manager.transforms if isinstance(stage, Save)]
        if not saves:
            return manager.dataset, manager.group_src
        return save_destination(saves[-1], manager.dataset, manager.group_dest)

    def _unbounded_members(self) -> dict[str, str]:
        """The members whose region reads decode their whole volume, with their store's format.
        Only an entry on disk is asked: a cache still to write lands on a store serving bounded
        reads."""
        unbounded: dict[str, str] = {}
        for manager in self.managers:
            dataset, group = self._member_source(manager)
            if dataset.is_dataset_exist(group, manager.name) and not dataset.bounded_region_reads(group, manager.name):
                unbounded[manager.name] = dataset.file_format
        return unbounded

    def _needs_stat_pass(self) -> bool:
        """Whether a stage after the reduction wants whole-volume statistics OF THE RESULT. The
        reduced volume is stored nowhere yet, so the engine computes them with a pass of its own:
        twice the reads, no intermediate volume."""
        return any(stage.patch_locality(Attribute()).kind is LocalityKind.GLOBAL_STAT for stage in self.post)

    def plan(self) -> ReductionPlan:
        reference = self.reference
        pull_bytes, read_bytes = self._member_read_bytes(int(reference.base_shape[0]))
        return ReductionPlan(
            output=self.reduce.output,
            cases=[manager.name for manager in self.managers],
            spatial=reference.spatial_shape,
            # The operator's own channel map: a Concat over N cases writes N times the channels,
            # and the plan must size the shape the run will open.
            channels=self.operator.output_channels(int(reference.base_shape[0]), len(self.managers)),
            source_channels=int(reference.base_shape[0]),
            slab_rows=self.slab_rows,
            incremental=self.operator.incremental,
            working_multiple=float(self.operator.working_multiple_for(len(self.managers))),
            # The worst member's: the fold is paced by whichever chain holds the most.
            chain_multiple=max((float(manager.working_multiple()) for manager in self.managers), default=0.0),
            pull_bytes=pull_bytes,
            read_bytes=read_bytes,
            stat_pass=self._needs_stat_pass(),
            unbounded=self._unbounded_members(),
            refusal=self._first_refusal(),
        )

    def _member_read_bytes(self, channels: int) -> tuple[int, int]:
        """What one member's region costs its store at the current height, in bytes: the source
        window it pulls, and what the store decodes above that window. Both are the widest member's
        and both are charged once; ``None`` from a manager contributes nothing."""
        from konfai.data.patching.budget import _SWEEP_ELEMENT_BYTES

        reads = [manager.region_reads(self.slab_rows) for manager in self.managers]
        present = [read for read in reads if read is not None]
        element = max(1, channels) * _SWEEP_ELEMENT_BYTES
        pull = max((read.widest_pull for read in present), default=0)
        excess = max((read.widest_excess for read in present), default=0)
        return int(pull * element), int(excess * element)

    # --------------------------------------------------------------- execution

    def _fold(self, region: tuple[slice, ...]) -> torch.Tensor:
        """One region of the reduced volume: every case reads that region, the operator folds them.
        Each region is presented as ``[1, C, *spatial]``, the stack-axis layout every operator is
        written against (:class:`~konfai.data.reduction.Reduction`); the result comes back without it."""
        with SWEEP_CLOCK.phase("chain"):
            self.operator.start()
        for manager in self.managers:
            # No name holds the region past its accumulate: one that did would keep a second member
            # region resident, which the plan does not price.
            self._accumulate(self._member_region(manager, region))
        with SWEEP_CLOCK.phase("chain"):
            return self.operator.finalize().squeeze(0)

    def _accumulate(self, member: torch.Tensor) -> None:
        """Fold one member region in, on the run's clock: an incremental operator does the
        reduction's own arithmetic here, and the region dies with this frame."""
        with SWEEP_CLOCK.phase("chain"):
            self.operator.accumulate(member)

    def _member_region(self, manager: DatasetManager, region: tuple[slice, ...]) -> torch.Tensor:
        """One member's region, in the stack-axis layout every operator is written against."""
        with _awaited("read"):
            return manager.read_region(region).unsqueeze(0)

    def _folds(self, spatial: list[int]):
        """Every region's fold, in order, the height following what the regions HOLD: the first is
        the one the sizing priced (:meth:`fit_budget`), each one after it is cut by
        :class:`RegionGrowth` from what the last held, judged against the budget less the cache's share."""
        budget = self._budget_bytes if self._budget_bytes and self._budget_bytes > 0 else None
        allowed = None if budget is None else budget - (budget_share("cache", budget) or 0.0)
        growth = RegionGrowth(self.slab_rows, self._cap_rows(spatial), allowed)
        meter = self._open_meter() if allowed else None
        device = self.managers[0]._chain_device if self.managers else None
        start = 0
        while start < int(spatial[0]):
            stop = min(start + growth.rows, int(spatial[0]))
            region = (slice(start, stop), *(slice(0, extent) for extent in spatial[1:]))
            try:
                folded = self._fold(region)
            except torch.cuda.OutOfMemoryError:
                # A region the device could not hold: half the height, this region again. The host
                # has no such signal, and one row that does not fit is the end.
                if not device_signals_oom(device) or growth.rows <= 1:
                    raise
                rows = growth.halve_below_first()
                torch.cuda.empty_cache()
                print(
                    f"[Reduce] '{self.reduce.output}': out of device memory on a {stop - start}-row region;"
                    f" the rest are cut to {rows} row(s).",
                    flush=True,
                )
                continue
            yield region, folded
            held = meter.held() if meter is not None else None
            SWEEP_CLOCK.region(stop - start, held)
            growth.after(held)
            start = stop

    def _open_meter(self) -> HeldMeter | None:
        """What will read the regions, chosen by the route they run on."""
        return open_held_meter(self.managers[0]._chain_device if self.managers else None)

    def _apply_post(self, block: torch.Tensor, attribute: Attribute, rank: int) -> np.ndarray:
        scope = Attribute(attribute)
        with SWEEP_CLOCK.phase("chain"):
            for stage in self.post:
                block = stage(self.reduce.output, block, scope)
        with SWEEP_CLOCK.phase("fetch"):
            array = block.cpu().numpy()
        if array.ndim != rank:
            raise ReductionError(
                f"A stage after the Reduce returned a rank-{array.ndim} region where the"
                f" channel-first layout needs rank {rank}.",
                "A transform folding the leading axis must keep it (`keepdim=True`).",
            )
        return array

    def _output_attributes(self, plan: ReductionPlan) -> Attribute:
        # The header is the geometry the reference's chain LANDS on, not the geometry it was stored
        # with: a cohort resampled onto a template grid must publish that grid.
        attribute = self.reference.landed_attributes()
        if self.reduce.provenance:
            # The deliverable carries its own recipe: a case list that changed between two runs
            # would otherwise write a different volume under the same name.
            attribute["konfai_reduce_operator"] = self.reduce.operator_classpath
            attribute["konfai_reduce_cases"] = "|".join(plan.cases)
        if plan.stat_pass:
            statistics = _RunningStatistics()
            # The folds this pass computes ARE the folds the write pass needs: keep them when they
            # fit their share (:meth:`keeps_folds`). Correctness never depends on the keep.
            self._kept_folds = [] if self.keeps_folds(plan) else None
            # The region is kept beside its fold: the growth changes the height along the pass, so
            # regions re-derived at any one height would misalign with the folds cut at another.
            for region, folded in self._folds(plan.spatial):
                statistics.update(folded)
                if self._kept_folds is not None:
                    self._kept_folds.append((region, folded.cpu()))
            statistics.write_into(attribute)
        return attribute

    def _folded_output_bytes(self, plan: ReductionPlan) -> int:
        return int(np.prod(plan.spatial, dtype=np.int64)) * max(1, int(plan.channels)) * _ASSUMED_ITEMSIZE

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

    def materialize(self, rewrite: bool = False, device: torch.device | None = None) -> bool:
        """Write the reduced entry, or raise saying why it cannot be written this way.

        There is no whole-volume fallback. A finished output is left alone unless ``rewrite``, which
        is the resume. ``device`` is where the fold runs: each member replays its region there, the
        operator folds there, and only the finished block comes back to the host for the write.
        """
        if not rewrite and self.destination.is_dataset_exist(self.group, self.reduce.output):
            return True
        for manager in self.managers:
            manager.set_chain_device(device)
            # --overwrite must reach the MEMBERS, not just this output: each member's read_region
            # resolves satisfied Saves from the previous run's caches unless told to rewrite.
            manager._set_rewrite(rewrite)
        if device is not None and device.type == "cuda":
            # The member regions this fold accumulates live in VRAM: the slabs are sized against the
            # card, not against a budget declared in host bytes.
            declared = self._budget_bytes
            capped = device_capped_budget(declared, device)
            if capped is not None and capped != declared:
                self.fit_budget(capped)
                # The slabs are the card's; the KEEP decision is the host's: kept folds live in host
                # memory (``.cpu()``).
                self._budget_bytes = declared
                # The PLAN printed the host figure, so the run says which budget it worked under.
                print(
                    f"[Reduce] '{self.reduce.output}': regions re-sized for {device} --"
                    f" {self.slab_rows} row(s) under {capped / 2**30:.2f} GiB"
                    f" (min of the declared budget and half of what the card can give this process).",
                    flush=True,
                )
        plan = self.plan()
        if not plan.streams:
            # A grid disagreement gets its own remedy: a Save changes nothing about the grids.
            remedy = (
                "The members do not land on one grid: resample them onto a common grid before the"
                " Reduce, or declare grid: reference:<case> / shape_only if the cohort is already"
                " aligned."
                if self.check_grid() is not None
                else "A reduction has no whole-volume fallback. Fix the refusing stage, or put a Save"
                " before the Reduce so each case's chain is materialized first."
            )
            raise ReductionError(f"The reduction into '{self.reduce.output}' cannot stream: {plan.refusal}.", remedy)

        with SWEEP_CLOCK.phase("sweep"):
            self._write_folds(plan)
        return True

    def _write_folds(self, plan: ReductionPlan) -> None:
        """Every region of the reduced volume, folded and written in order, under one clock."""
        spatial = plan.spatial
        attribute = self._output_attributes(plan)
        rank = len(spatial) + 1
        # The folds a stat pass kept are written as they are; otherwise every region is folded here.
        kept = self._kept_folds
        self._kept_folds = None
        folds = iter(kept) if kept is not None else self._folds(spatial)
        writer = RegionWriter(lambda _key, array, header: self._open_stream(spatial, array, header))
        try:
            for region, folded in folds:
                array = self._apply_post(folded, attribute, rank)
                with _awaited("write"):
                    writer.write(None, (slice(0, int(array.shape[0])), *region), array, attribute)
            with _awaited("write"):
                writer.close()
        except BaseException as exception:
            writer.abort(exception)
            raise
