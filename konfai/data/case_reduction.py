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
folds them. Peak memory is N regions, never N volumes, and two regions, whatever N, once the
operator can accumulate.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from dataclasses import dataclass, field

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
from konfai.data.reduction import Reduction
from konfai.data.transform import LocalityKind, PatchLocality, Reduce, Save, Transform, stat_seed_valid
from konfai.utils.budget import budget_share, format_bytes
from konfai.utils.dataset import (
    Attribute,
    Dataset,
    DataStream,
    _finalize_running_statistics,
    _update_running_statistics,
)
from konfai.utils.errors import ReductionError

#: Geometry keys compared between cases under ``grid: strict``. Direction is in because a flipped
#: axis shows in neither extent nor spacing: averaging two volumes that disagree on it mirrors half
#: the cases into the other half, silently, and the result still looks like a volume.
_GEOMETRY_KEYS = ("Spacing", "Origin", "Direction")

#: What a stage placed AFTER the reduction may declare. Voxel-local is exact on a region; a
#: whole-volume statistic is exact too, because the engine seeds it with a pass of its own.
_POST_KINDS = frozenset({LocalityKind.POINTWISE, LocalityKind.GLOBAL_STAT})

#: Bytes per sample assumed when sizing regions from headers alone, before a dtype is known.
_ASSUMED_ITEMSIZE = 4

#: How much of the regions' own share the folds a stat pass KEEPS may take, the regions getting the
#: rest. Not a share of the declaration: they sit beside the regions for the whole write pass, so
#: they come out of what the regions were given (:data:`~konfai.utils.budget.BUDGET_SHARES`).
_KEPT_FOLDS_SHARE_OF_REGIONS = 0.5


@contextlib.contextmanager
def _awaited(phase: str) -> Iterator[None]:
    """A phase the fold both performs and stands still for.

    The fold has no pipeline: every member's region is read, and every slab written, on its own
    thread. So the store's own seconds ARE the seconds the loop waits, which is what
    :meth:`~konfai.data.patching.SweepClock.report` prints on either side of its bar.
    """
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
    #: Channels a MEMBER's region carries. Separate from ``channels``, the output's, because an
    #: operator may change the count: ``Concat`` writes ``N x C`` where each member holds ``C``, so
    #: charging the members at the output's width over-states the peak by the cohort's size.
    source_channels: int = 0
    working_multiple: float = 0.0
    #: Volumes-worth the MEMBER CHAIN allocates beside the region it is producing
    #: (``DatasetManager.working_multiple()``, the largest ``Transform.working_multiple`` on the
    #: chain: ``Resample`` declares 6.5 for its sampling grid, plus what the case's own field costs
    #: through ``case_working_multiple``). Distinct from ``working_multiple``,
    #: which is the OPERATOR's: a fold is a chain replay per member and then an accumulate, and
    #: pricing only the second half under-states a resampling cohort by the first. Charged ONCE
    #: whatever the cohort's size, because ``_fold`` accumulates the members one after another, so
    #: only one chain is ever replaying.
    chain_multiple: float = 0.0
    #: What ONE member's region makes the store decode ABOVE the window it asked for
    #: (:meth:`~konfai.data.patching.DatasetManager.region_reads`). A chunked backend decodes whole
    #: blocks, so below one stored block this is the SAME figure at every height: it is charged flat
    #: and never divided by the rows. Measured on the prep's cohort at 4.51 GiB for a 17-row region
    #: whose own tensor was 70 MiB, and 4.57 GiB for a 4-row one -- the sizing had been cutting the
    #: height to buy memory that cutting cannot buy.
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

        A store serving bounded region reads is read once per pass. One that cannot decodes the
        whole volume behind every region asked of it: once per region and per pass, so a budget
        that lowers ``slab_rows`` raises the count. The figure of the worst member, which the run is
        paced by; ``unbounded`` names the members it applies to.
        """
        return float(self.passes * (self.regions if self.unbounded else 1))

    @property
    def buffered_regions(self) -> int:
        """Member regions resident at once: one for a running accumulator, else the whole cohort."""
        return 1 if self.incremental else len(self.cases)

    @property
    def resident_regions(self) -> float:
        """Regions held at the peak, in member regions: the buffer, what the operator builds over
        it (``working_multiple`` buffers-worth), what the one replaying chain holds beside the
        region it is producing (``chain_multiple``), and the output's own. A count for
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
        # buffer it is handed, which is member-sized, since that is what it was handed.
        #
        # A statistics pass is a second traversal, not a second working set: it holds exactly what
        # one region holds, so the peak is the same whether there are one or two passes.
        member_bytes = self._region_bytes(self.source_channels or self.channels)
        members = self.buffered_regions * member_bytes
        # The chain replaying a member holds its own buffers beside the region it lands, and it is
        # the members' own width it holds them at. Once: the members are accumulated in turn.
        # Plus what the store decodes to serve ONE member's region: the chain's buffers are built
        # over the window it asked for, the decode materialises the blocks that window falls in, and
        # the two are resident together. One read is in flight at a time (the members accumulate in
        # turn), so it is charged once.
        return int(
            members * (1 + self.working_multiple)
            + self.chain_multiple * member_bytes
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

    Each stage after the reduction is handed ONE REGION of the result. A stage reading across space (a halo, a
    resample, a reorientation) would take that region for the whole volume and seam
    at every boundary: a plausible result, and a wrong one. Those are deferred, not forbidden:
    end the chain, and read the written volume back in a second chain where the ordinary planner can
    pull regions through it.

    A statistic may follow the reduction, but only over stages that leave the values alone: the stat
    pass measures the FOLD, so an earlier stage that changes the values makes the seed describe a
    volume nobody wrote (``stat_seed_valid``, the per-case planner's rule).
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

    It uses only the public read side of each case's manager (the streaming machinery already
    planned, accepted or refused per case), and owns the write side itself, under the output name
    the chain declared.
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
        """Size the regions so the resident ones fit ``budget_bytes``.

        The default region height is safe for one case and not for N: a reduction holds one region
        PER CASE, so the constant that bounds a per-case sweep is off by the number of cases here.
        Half the budget, because the write buffer lives alongside the peak the plan prices.
        Below one row nothing fits; the plan then reports a peak above the budget and the workflow
        refuses, which is the only honest answer: there is no whole-volume path to fall back to.

        THE BUDGET IS A CEILING, NOT A TARGET. ``cap`` defaults to the shortest region that already
        reads what the whole volume reads (``DatasetManager.read_plateau_rows``, closed form from
        the chain's pull maps): past it a taller region pulls no fewer source voxels and only holds
        more. A FIXED cap was wrong in the other direction -- 64 rows however much memory the run
        was given made a cohort measured at 0.11 GiB against a 59.60 GiB budget re-sweep every
        source per 64 rows, which is what a chain resampling through a field pays for, since a
        region's source window is not the region. The plateau is neither: it is where that
        re-sweeping stops, per chain, and the budget may still lower it. What is spent stays
        declared: half a budget the caller named, or ``auto``, which measures the node.

        The budget also goes to the cases' own managers, because a chain crossing a ``Save`` sweeps
        that cache when first read, and ``read_region`` carries no budget of its own.
        """
        self._budget_bytes = budget_bytes
        # THE REMAINDER, not the whole figure. A member's chain holds what it holds WHILE the fold
        # is holding its regions -- its own sweeps fire from inside the fold loop, when the operator
        # already has earlier members in its buffer -- so handing it the full budget six lines above
        # spending half of it on the regions declared the same bytes twice.
        chains = budget_share("chains", budget_bytes)
        for manager in self.managers:
            manager.set_memory_budget(budget_bytes if chains is None else chains)
        if not budget_bytes or budget_bytes <= 0:
            return
        plan = self.plan()
        # What the store decodes does NOT fall with the height: below one stored block the same
        # blocks are decoded whatever the region asks for, and the fold simply asks more often. It
        # is taken off the allowance whole, never divided by the rows -- dividing it is what made a
        # small budget cut the regions, decode the same bytes twice as often, and hold MORE.
        flat = plan.read_bytes
        row_bytes = (plan.peak_bytes - flat) / max(1, plan.slab_rows)
        if row_bytes <= 0:
            return
        # Past the plateau a taller region reads no less and holds more, so that is the ceiling
        # when the caller names none; a chain that cannot price one falls back to the output height.
        ceiling = int(cap) if cap is not None else self._plateau_rows(plan)
        # What the fold's own share leaves the regions: the folds it keeps live beside them, for the
        # whole write pass, so they come out of the same half rather than out of nothing. Decided
        # here and not at the stat pass, because the regions cannot be sized against a decision that
        # has not been taken yet.
        allowance = budget_share("regions", budget_bytes) or 0.0
        if self.keeps_folds(plan):
            allowance -= self._folded_output_bytes(plan)
        allowance -= flat
        self.slab_rows = max(1, min(ceiling, int(max(row_bytes, allowance) / row_bytes)))

    def keeps_folds(self, plan: ReductionPlan) -> bool:
        """Whether the stat pass hands its folds to the write pass instead of re-folding them.

        One rule, two callers: the sizing subtracts what they will hold, the stat pass fills them.
        Two rules would let the regions be cut for folds nobody keeps, or folds be kept the regions
        made no room for. They may take their share of what the FOLD holds, never of the whole
        declaration: a member's chain is holding the rest of it at the same moment.
        """
        if not plan.stat_pass or not self._budget_bytes or self._budget_bytes <= 0:
            return False
        regions = budget_share("regions", self._budget_bytes)
        return regions is not None and self._folded_output_bytes(plan) <= regions * _KEPT_FOLDS_SHARE_OF_REGIONS

    def _plateau_rows(self, plan: ReductionPlan) -> int:
        """The tallest region worth cutting: the WORST member's plateau, because one chain still
        re-reading paces the fold whatever the others do.

        Never below ``SWEEP_SLAB_ROWS``, the floor the sweep applies for the same reason: a chain
        that pulls exactly what it lands (every stage POINTWISE) has a FLAT read curve, so its
        plateau is one row -- and one-row regions pay every fixed per-region cost for one row of
        work. Measured: a ``Clip`` before the fold sized the regions at 1 row without this floor.

        And never the WHOLE VOLUME when no member can price one. A chain with no stage before the
        fold has no segment to read a pull map from, so every member answers None -- and answering
        the output's own height there made the ceiling depend on something that is not a property
        of the data: the same cohort, the same budget and the same voxels read gave 512 rows with
        no stage and 64 with one value-neutral ``Clip`` in front, an eightfold difference from a
        stage that changes nothing. Unpriced reads get the floor, like flat ones: it is the same
        ignorance, and the conservative answer to it is the same.

        The whole-volume answer was also the one place the run-time probe could not help. At full
        height a fold has exactly ONE region, and :meth:`_refit_to_measurement` only ever cuts the
        ones after the first: the net was disarmed exactly where the most was held.
        """
        heights = [manager.read_plateau_rows(plan.spatial) for manager in self.managers]
        measured = [height for height in heights if height is not None]
        floor = min(int(SWEEP_SLAB_ROWS), int(plan.spatial[0]))
        return max(max(measured), floor) if measured else floor

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

        Compared on the grid each case's chain LANDS on (the folded shape and the plan-evolved
        geometry), not the stored one: the chain exists precisely to bring disagreeing members onto
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

    @staticmethod
    def _member_source(manager: DatasetManager) -> tuple[Dataset, str]:
        """The store and group a member's regions are read from: its own, or its last ``Save``'s cache."""
        saves = [stage for stage in manager.transforms if isinstance(stage, Save)]
        if not saves:
            return manager.dataset, manager.group_src
        return save_destination(saves[-1], manager.dataset, manager.group_dest)

    def _unbounded_members(self) -> dict[str, str]:
        """The members whose region reads decode their whole volume, with their store's format.

        Only an entry on disk is asked: a cache the run has still to write is swept onto a store
        that serves region writes, and every such store serves bounded region reads.
        """
        unbounded: dict[str, str] = {}
        for manager in self.managers:
            dataset, group = self._member_source(manager)
            if dataset.is_dataset_exist(group, manager.name) and not dataset.bounded_region_reads(group, manager.name):
                unbounded[manager.name] = dataset.file_format
        return unbounded

    def _needs_stat_pass(self) -> bool:
        """Whether a stage after the reduction wants whole-volume statistics OF THE RESULT.

        Those cannot be seeded from disk the usual way: the reduced volume is stored nowhere yet --
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
            working_multiple=float(self.operator.working_multiple_for(len(self.managers))),
            # The worst member's, because the fold is paced by whichever chain holds the most.
            chain_multiple=max((float(manager.working_multiple()) for manager in self.managers), default=0.0),
            read_bytes=self._member_read_bytes(int(reference.base_shape[0])),
            stat_pass=self._needs_stat_pass(),
            unbounded=self._unbounded_members(),
            refusal=self._first_refusal(),
        )

    def _member_read_bytes(self, channels: int) -> int:
        """What the widest member's region makes its store decode, at the current height.

        The fold is paced by whichever member decodes the most, and one read is in flight at a
        time. ``None`` from a manager (a chain that cannot answer) contributes nothing: the peak
        then says what it did before, which is what the run-time probe is there to correct.
        """
        from konfai.data.patching import _SWEEP_ELEMENT_BYTES

        reads = [manager.region_reads(self.slab_rows) for manager in self.managers]
        widest = max((excess for read in reads if read is not None for excess, _total in [read]), default=0)
        return int(widest * max(1, channels) * _SWEEP_ELEMENT_BYTES)

    # --------------------------------------------------------------- execution

    def _regions(self, spatial: list[int]) -> list[tuple[slice, ...]]:
        """The output's regions: slabs along the first spatial axis, whole in the others."""
        return [
            (slice(start, min(start + self.slab_rows, spatial[0])), *(slice(0, extent) for extent in spatial[1:]))
            for start in range(0, spatial[0], self.slab_rows)
        ]

    def _fold(self, region: tuple[slice, ...]) -> torch.Tensor:
        """One region of the reduced volume: every case reads that region, the operator folds them.

        Each region is presented as ``[1, C, *spatial]`` (the stack-axis layout every operator is
        written against (see :class:`~konfai.data.reduction.Reduction`)), and the result comes back
        without it. The axis matters to any operator that places things side by side.
        """
        with SWEEP_CLOCK.phase("chain"):
            self.operator.start()
        for manager in self.managers:
            # No name holds the region past its accumulate: one that did would keep a second member
            # region resident, which the plan does not price (1162 MiB against 778 on a 5 x 384 MiB
            # float32 cohort folded whole). The read stays outside the phase, and the argument dies
            # with the frame that times the fold of it.
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

    # What the first region's measurement leaves for the rest, as the predictor's accumulation gate
    # keeps a reserve for the same reason (Predictor._ACCUMULATE_MARGIN): the measurement is of the
    # region that just ran, and the next one meets an allocator in a different state.
    _MEASURED_MARGIN = 0.9
    #: The probe's share of the planned height. The probe is the one region that runs BEFORE any
    #: measurement can bound it, so it is the one region that must not be able to kill the run on
    #: its own. At the planned height it could: a fold over registration fields held 1.42x, 1.47x
    #: and 1.50x what its first region was allowed at three budgets, and at an `auto` budget of 77
    #: GiB that first region reached 90 GiB resident on a 122 GiB host, and the host went down
    #: before the probe could read anything. The host gives no OutOfMemoryError to catch: the
    #: kernel kills. A quarter-height probe overshooting by the same 1.5x holds 0.4 of the budget,
    #: which is survivable, and the ratio it measures is the same one -- the halo does not shrink
    #: with the region, so a short region over-holds by MORE than a tall one, and a refit from it
    #: is conservative. Its price is one extra region: seconds, on a fold of minutes.
    _PROBE_SHARE = 0.25

    def _folds(self, spatial: list[int], measure: bool = False):
        """Every region's fold, in order: the loop both passes share.

        With ``measure``, the FIRST region is also the probe. What the plan priced is a model, and a
        model of what a chain holds has to be right about every stage, every store and every bridge
        it crosses; what the region actually held is a fact, and it costs one counter read on work
        the fold had to do anyway. The remaining regions are then cut against the fact.

        Only ever SHORTER. A taller region than the plan allowed would be a promise the plan never
        made, and the first region has already been folded at the planned height: this can correct
        an optimistic price, never spend a budget the sizing declined to spend.
        """
        start, refitted = 0, not measure
        while start < int(spatial[0]):
            # The probe is SHORT. Every later region is cut against what it measured; the probe
            # itself is cut against nothing, so it is sized so that its own overshoot cannot
            # reach the host's limit (_PROBE_SHARE).
            rows = self.slab_rows if refitted else max(1, int(self.slab_rows * self._PROBE_SHARE))
            stop = min(start + rows, int(spatial[0]))
            region = (slice(start, stop), *(slice(0, extent) for extent in spatial[1:]))
            # Only around the region that is actually the probe. The host meter RESETS the
            # process's resident high-water mark to take its reading, and that mark is what the
            # run's closing line reports: resetting it before every region would leave that line
            # describing the last region instead of the run, which is the one thing it exists to
            # say honestly.
            meter = None if refitted else self._open_meter()
            yield region, self._fold(region)
            if not refitted:
                refitted = True
                self._refit_to_measurement(meter, stop - start, spatial)
            start = stop

    def _open_meter(self) -> HeldMeter | None:
        """What will read the probe region, chosen by the route it runs on."""
        return open_held_meter(self.managers[0]._chain_device if self.managers else None)

    def _refit_to_measurement(self, meter: HeldMeter | None, rows: int, spatial: list[int]) -> None:
        """Cut the remaining regions against what the first one HELD, when that is more than the
        declaration allows. The reading is a high-water mark over the region that just ran, so it
        bounds the next one from above, exactly as the predictor's gate reads a forward's transient
        from the batch that just ran (:meth:`Predictor._accumulate_device`).

        Against the whole declaration LESS the chunk cache's share, because that is what the
        reading covers. A share is how a height is chosen and what must not be exceeded is the
        declaration -- but the meter no longer counts the decoded-chunk cache (it outlives the
        region, and charging the region for it cut every region after the probe), so the cache's
        bytes have to come off the other side of the comparison too. Judged against the whole
        budget, a reading that excludes the cache lets the cache be spent twice: once inside the
        allowance, and again by the cache itself. This exists to prevent a kill, not to shave
        bytes. The probe is a short region (_PROBE_SHARE), so what it held is scaled to
        the planned height before it is judged: a probe that held its share of the budget says the
        full region would hold the budget, and a probe that held more says the full region would
        be the kill this exists to prevent. Only ever shorter: a probe that came in under its
        share does not talk the fold into a taller region than the plan allowed.
        """
        del spatial
        held = meter.held() if meter is not None else None
        if held is None or not self._budget_bytes or self._budget_bytes <= 0 or rows <= 0 or held <= 0:
            return
        cache = budget_share("cache", self._budget_bytes) or 0.0
        allowed = (float(self._budget_bytes) - cache) * self._MEASURED_MARGIN
        # What the FULL region would hold, from what the probe held: the halo is a fixed cost the
        # probe paid in full, so scaling by height over-estimates, which is the safe direction.
        projected = held * (self.slab_rows / float(rows))
        if projected <= allowed:
            return
        fitted = max(1, int(self.slab_rows * allowed / projected))
        if fitted >= self.slab_rows:
            return
        print(
            f"[Reduce] '{self.reduce.output}': a {rows}-row probe held {format_bytes(held)}, so the planned"
            f" {self.slab_rows} row(s) would hold {format_bytes(projected)} of the {format_bytes(allowed)}"
            f" allowed -- the rest are cut to {fitted} row(s).",
            flush=True,
        )
        self.slab_rows = fitted

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
        # with: a cohort resampled onto a template grid by its pre-chain must publish that grid --
        # seeding the source's own Spacing here would stamp a wrong header on every round of an
        # atlas build, and nothing about the volume would look wrong.
        attribute = self.reference.landed_attributes()
        if self.reduce.provenance:
            # The deliverable carries its own recipe. A set of cases that changed between two runs
            # would otherwise write a different volume under the same name: the worst way this can
            # fail, because nothing about the output looks wrong.
            attribute["konfai_reduce_operator"] = self.reduce.operator_classpath
            attribute["konfai_reduce_cases"] = "|".join(plan.cases)
        if plan.stat_pass:
            statistics = _RunningStatistics()
            # The folds this pass computes ARE the folds the write pass needs: keep them when they
            # fit their share of what the fold holds (:meth:`keeps_folds`, the same rule the sizing
            # subtracted them by), and the write pass then only applies the post stages. Otherwise
            # the second pass re-folds, as before: correctness never depends on the keep.
            self._kept_folds = [] if self.keeps_folds(plan) else None
            for _region, folded in self._folds(plan.spatial):
                statistics.update(folded)
                if self._kept_folds is not None:
                    self._kept_folds.append(folded.cpu())
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

        There is no whole-volume fallback: assembling every case is exactly what this exists to
        avoid, and doing it silently would turn a bounded run into an unannounced OOM. A finished
        output is left alone unless ``rewrite``, which is the resume.

        ``device`` is where the fold runs: each member replays its region there, the operator folds
        there, and only the finished block comes back to the host for the write.
        """
        if not rewrite and self.destination.is_dataset_exist(self.group, self.reduce.output):
            return True
        for manager in self.managers:
            manager.set_chain_device(device)
            # --overwrite must reach the MEMBERS, not just this output: each member's read_region
            # resolves satisfied Saves from the previous run's caches unless told to rewrite, and a
            # reduction folding stale caches writes a wrong volume whose provenance looks correct.
            manager._set_rewrite(rewrite)
        if device is not None and device.type == "cuda":
            # The member regions this fold accumulates live in VRAM: the slabs are sized against
            # the card, not against a budget declared in host bytes -- a host-sized region set on a
            # 16 GB card is an OOM mid-fold, after the stat pass already paid.
            declared = self._budget_bytes
            capped = device_capped_budget(declared, device)
            if capped is not None and capped != declared:
                self.fit_budget(capped)
                # The slabs are the card's; the KEEP decision is the host's: kept folds live in host
                # memory (``.cpu()``), and judging them against VRAM/2 re-folded outputs the host
                # could have held whole -- a second full pass for nothing.
                self._budget_bytes = declared
                # Said once, because the PLAN printed the host figure: the run must say which
                # budget it actually worked under, or every future OOM is diagnosed off a lie.
                print(
                    f"[Reduce] '{self.reduce.output}': regions re-sized for {device} --"
                    f" {self.slab_rows} row(s) under {capped / 2**30:.2f} GiB"
                    f" (min of the declared budget and half of what the card can give this process).",
                    flush=True,
                )
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

        with SWEEP_CLOCK.phase("sweep"):
            self._write_folds(plan)
        return True

    def _write_folds(self, plan: ReductionPlan) -> None:
        """Every region of the reduced volume, folded and written in order, under one clock."""
        spatial = plan.spatial
        attribute = self._output_attributes(plan)
        rank = len(spatial) + 1
        # The folds a stat pass kept (when the folded output fits half the budget) are written as
        # they are; otherwise every region is folded here, once.
        kept = self._kept_folds
        self._kept_folds = None
        folds = (
            ((region, kept[index]) for index, region in enumerate(self._regions(spatial)))
            if kept is not None
            else self._folds(spatial, measure=True)
        )
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
