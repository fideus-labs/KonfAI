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

"""The TRANSFORM materialization engine: writes one case's chain over the manager's plan/replay API.

:class:`CaseMaterializer` drives one case's ``Save`` outputs to disk by the cheapest route the plan
allows (the streamed slab sweep, the whole-volume load, or one shared read pass across the copies of
an ``Expand``) and prices those routes for the plan. It holds no read state of its own: the rewrite
flag, the swept-entry ledger and the sweep-failure reason live on the manager.
"""

import contextlib
import warnings
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import torch

from konfai.data.patching import (
    CASE_ELEMENT_BYTES,
    FALLBACK_INFLIGHT_FACTOR,
    AugmentedStage,
    DatasetManager,
)
from konfai.data.patching.stage import _ReadStagePlan, _stage_name
from konfai.data.patching.sweep import (
    _PendingSweep,
    _stage_failures_explained,
    _SweepMember,
)
from konfai.data.transform import LocalityKind, Reduce, Resample, Save, Transform
from konfai.utils.budget import format_bytes
from konfai.utils.dataset import Attribute
from konfai.utils.errors import PatchError


class Verdict(StrEnum):
    """What the plan says of an entry, and what the run then does with it: one vocabulary. The engine
    answers STREAM, LOAD (the plan's choice: the case fits and its source serves no bounded region read) or
    WHOLE_VOLUME (the fallback); SKIP, REDUCE and REFUSED are the workflow's."""

    STREAM = "STREAM"
    LOAD = "LOAD"
    WHOLE_VOLUME = "WHOLE-VOLUME"
    SKIP = "SKIP"
    REDUCE = "REDUCE"
    REFUSED = "REFUSED"


class Regime(StrEnum):
    """How a STREAM copy of an Expand case is read: SHARED rides the case's single read pass, SOLO
    sweeps its own (its draw's read geometry is its own)."""

    SHARED = "shared"
    SOLO = "solo"


@dataclass(frozen=True)
class CopyRoute:
    """The engine's answer for one Expand copy: its regime (``None`` when it cannot stream), why it
    sweeps alone, and the per-copy Save a SHARED copy writes."""

    regime: Regime | None
    reason: str | None = None
    sweep: _PendingSweep | None = None


class CaseMaterializer:
    """Write one case's chain to disk over its manager's plan/replay API, and price the routes. Built
    per case and kept for the run: the peak-bytes fold is memoized here, so the plan, the shards and
    the run read the same figure."""

    def __init__(self, manager: DatasetManager) -> None:
        self.manager = manager
        self._peak_case_bytes: int | None = None

    # ---------------------------------------------------------------- the write side

    @contextlib.contextmanager
    def _materialization(
        self, rewrite: bool, fallback_budget_bytes: float | None, device: "torch.device | None"
    ) -> Iterator[None]:
        """What every materialization sets up on the manager: the rewrite mode, the budget its
        sweeps size their slabs against, and the device the chain runs on for this call (opt-in,
        and only here: the transformer's rank is the one caller that knows its device)."""
        self.manager._set_rewrite(rewrite)
        self.manager.set_memory_budget(fallback_budget_bytes)
        with self.manager._chain_device_scope(device):
            yield

    def materialize(
        self,
        a: int = 0,
        rewrite: bool = False,
        fallback_budget_bytes: float | None = None,
        allow_fallback: bool = True,
        prefer_whole: bool = False,
        device: "torch.device | None" = None,
    ) -> Verdict:
        """Write this case's chain to disk by the cheapest path that can, and say which it took.

        The streamed path sweeps every unsatisfied :class:`Save` slab by slab, the whole-volume load
        writes the same caches with more memory. ``allow_fallback=False`` raises instead of
        loading when streaming is unavailable, but a planned ``prefer_whole`` route still loads;
        ``prefer_whole`` is the plan's LOAD choice; ``rewrite=True`` recomputes the case and renames
        over the old entries (``--overwrite``). A chain whose caches all exist writes nothing.
        """
        with self._materialization(rewrite, fallback_budget_bytes, device):
            verdict = self._write_case(a, fallback_budget_bytes, allow_fallback, prefer_whole)
            self.manager.unload()
            return verdict

    def _write_case(
        self, a: int, fallback_budget_bytes: float | None, allow_fallback: bool, prefer_whole: bool
    ) -> Verdict:
        """One case or copy, inside a materialization: streamed if it can, else the whole volume."""
        manager = self.manager
        # With an Expand this materializes a COPY, whose draw must be part of the plan.
        apply_augmentations = manager._expand is not None and a > 0
        if not prefer_whole:
            if manager._stream_ready(a, apply_augmentations=apply_augmentations):
                return Verdict.STREAM
            if not allow_fallback:
                raise PatchError(
                    f"Case '{manager.name}' would take the whole-volume path at run time.",
                    manager.stream_refusal(a, apply_augmentations)
                    or manager._sweep_failure
                    or "the chain cannot stream.",
                    "Nothing was written for this case; the caller forbids the whole-volume fallback.",
                )
        # A case that fell back here must not assemble a volume the budget cannot hold.
        self._enforce_fallback_budget(fallback_budget_bytes)
        self._assemble_and_write(a)
        return Verdict.LOAD if prefer_whole else Verdict.WHOLE_VOLUME

    def _enforce_fallback_budget(self, fallback_budget_bytes: float | None) -> None:
        if fallback_budget_bytes is None:
            return
        case_bytes = self.fallback_working_set_bytes()
        if case_bytes > fallback_budget_bytes:
            raise PatchError(
                f"Case '{self.manager.name}' fell back to the whole-volume path at run time and its"
                f" working set (~{format_bytes(case_bytes)}) exceeds the per-rank budget"
                f" ({format_bytes(fallback_budget_bytes)}).",
                "Nothing was written for this case. Raise 'memory_budget' or make the chain streamable.",
            )

    def _assemble_and_write(self, a: int) -> None:
        """The whole-volume fallback: assemble and write every Save. The caller releases. With an
        :class:`Expand`, the shared part is assembled once and only the copy's draw and the per-copy
        stages run per copy, writing their Saves under the copy's name."""
        with _stage_failures_explained():
            self._assemble_and_write_chain(a)

    def _assemble_and_write_chain(self, a: int) -> None:
        manager = self.manager
        if manager._expand is None:
            manager.load(manager.transforms, [], load_augmentations=False)
            return
        manager.load(manager._expand_pre, [], load_augmentations=False)
        tensor = manager.data[0].clone()
        attribute = Attribute(manager.cache_attributes[0])
        manager._apply_chain(tensor, manager._expand_tail(a), attribute, manager.copy_entry(a))

    def materialize_copies(
        self,
        copies: list[int],
        rewrite: bool = False,
        fallback_budget_bytes: float | None = None,
        allow_fallback: bool = True,
        device: "torch.device | None" = None,
    ) -> dict[int, tuple[Verdict, Regime | None]]:
        """Write the :class:`Expand` copies of this case; returns what each took: the verdict and,
        for a streamed copy, whether it rode the shared pass or its own. Copies whose per-copy stages
        are all pointwise share ONE read pass; a copy whose draw reads regions sweeps its own; a copy
        that cannot stream falls back to the whole volume."""
        manager = self.manager
        if manager._expand is None:
            raise PatchError(
                f"materialize_copies() on case '{manager.name}' whose chain declares no Expand.",
                "Use materialize() for a 1-to-1 chain; copies only exist behind an Expand marker.",
            )
        with self._materialization(rewrite, fallback_budget_bytes, device):
            manager._require_statistics()
            outcomes: dict[int, tuple[Verdict, Regime | None]] = {}
            if not copies:
                return outcomes
            # The caches the copies share (pre-Expand Saves) are swept once, first: every copy's
            # plan reads through them.
            first = manager._resolve_patch_stream_source(copies[0], apply_augmentations=True)
            if first is not None:
                manager._sweep_pending([sweep for sweep in first.pending_sweeps if sweep.entry == manager.name])

            routes = self.classify_copies(copies)
            shared = [(a, route.sweep) for a, route in routes.items() if route.regime is Regime.SHARED and route.sweep]
            solo = [a for a, route in routes.items() if route.regime is Regime.SOLO]
            fallback = [a for a, route in routes.items() if route.regime is None]
            if shared:
                written = self._materialize_shared_pass(shared)
                for a, _sweep in shared:
                    if a in written:
                        outcomes[a] = (Verdict.STREAM, Regime.SHARED)
                    else:
                        solo.append(a)
                if written:
                    manager._invalidate_stream_plans()
            for a in sorted(solo):
                verdict = self._write_case(a, fallback_budget_bytes, allow_fallback, prefer_whole=False)
                outcomes[a] = (verdict, Regime.SOLO if verdict is Verdict.STREAM else None)
            if fallback:
                if not allow_fallback:
                    raise PatchError(
                        f"{len(fallback)} cop(ies) of case '{manager.name}' would take the whole-volume path at run time.",
                        manager.stream_refusal(fallback[0], apply_augmentations=True)
                        or "the copies' chains cannot stream.",
                        "Nothing was written for these copies; the caller forbids the whole-volume fallback.",
                    )
                self._enforce_fallback_budget(fallback_budget_bytes)
                for a in fallback:
                    self._assemble_and_write(a)
                    outcomes[a] = (Verdict.WHOLE_VOLUME, None)
            manager.unload()
            return outcomes

    def classify_copies(self, copies: Iterable[int]) -> dict[int, CopyRoute]:
        """How each copy streams, decided once for the run and the plan alike. A copy joins the
        shared read pass when its per-copy segment is one Save whose stages are all pointwise; it
        sweeps alone when a per-copy stage reads regions, when it crosses several per-copy Saves, or
        when a cache the copies share is still to write. A copy the planner refuses has no regime."""
        manager = self.manager
        routes: dict[int, CopyRoute] = {}
        for a in copies:
            source = manager._resolve_patch_stream_source(a, apply_augmentations=True)
            if source is None:
                routes[a] = CopyRoute(None)
                continue
            per_copy = [sweep for sweep in source.pending_sweeps if sweep.entry != manager.name]
            if len(per_copy) != len(source.pending_sweeps):
                reason = "a cache the copies share is still to write, so this copy sweeps it on its own pass."
                routes[a] = CopyRoute(Regime.SOLO, reason)
            elif not per_copy:
                routes[a] = CopyRoute(Regime.SOLO)  # nothing left to write: its own path writes nothing
            elif len(per_copy) == 1 and self._pointwise_tail(per_copy[0].tail_plans):
                routes[a] = CopyRoute(Regime.SHARED, sweep=per_copy[0])
            else:
                routes[a] = CopyRoute(Regime.SOLO, self._solo_reason(per_copy))
        shared = [a for a, route in routes.items() if route.regime is Regime.SHARED]
        if len(shared) == 1:
            reason = "the only copy of this case still to write; a shared pass with one member is its own sweep."
            routes[shared[0]] = CopyRoute(Regime.SOLO, reason)
        return routes

    @staticmethod
    def _pointwise_tail(plans: Sequence[_ReadStagePlan]) -> bool:
        """Whether a copy's per-copy stages are pure value maps on their slab: a pointwise tail pulls
        exactly the slab the shared prefix landed on, so every such copy can consume one read."""
        return all(plan.kind is LocalityKind.POINTWISE for plan in plans)

    @staticmethod
    def _solo_reason(per_copy: list[_PendingSweep]) -> str:
        """Why a copy with pending per-copy Saves cannot join the shared pass."""
        if len(per_copy) > 1:
            return "the copy's chain crosses more than one per-copy Save, so it sweeps its own passes."
        sweep = per_copy[0]
        for stage, plan in zip(sweep.tail_stages, sweep.tail_plans, strict=False):
            if plan.kind is not LocalityKind.POINTWISE:
                return (
                    f"stage '{_stage_name(stage)}' of this copy declares {plan.kind.name},"
                    " so its read geometry is the draw's own and it sweeps its own pass."
                )
        return "the copy's plan is incomplete; it sweeps its own pass."

    def _materialize_shared_pass(self, shared: list[tuple[int, _PendingSweep]]) -> set[int]:
        """One read pass, N write streams: each slab is computed through the shared prefix once, and
        each copy applies its pointwise tail to a clone and writes into its own stream. On failure
        the copies take their own passes; returns the copies written."""
        manager = self.manager
        reference = shared[0][1]
        # The regime only holds when every copy lands the same grid from the same source and writes
        # into the same store; the mismatch is checked rather than assumed.
        shared = [
            (a, sweep)
            for a, sweep in shared
            if sweep.out_spatial == reference.out_spatial
            and str(sweep.source_dataset.filename) == str(reference.source_dataset.filename)
            and sweep.source_group == reference.source_group
            and sweep.source_entry == reference.source_entry
            and str(sweep.destination.filename) == str(reference.destination.filename)
        ]
        if not shared:
            return set()
        # The shared prefix is planned once and replayed per slab; each copy's header is its own.
        source, prefix_evolved, _refusal = manager._replan_sweep(
            reference, list(reference.stages[: reference.copy_stage_start])
        )
        if source is None:
            return set()
        members: list[_SweepMember] = []
        for a, sweep in shared:
            planned, evolved, _reason = manager._replan_sweep(sweep)
            if planned is None:
                continue
            tail_plans = planned.stage_plans[sweep.copy_stage_start :]
            # Re-planned against the materialized source, so re-checked here: the shared block is a
            # tail stage's region only while every one of them is pointwise.
            if self._pointwise_tail(tail_plans):
                members.append(_SweepMember(a, sweep, evolved, sweep.tail_stages, tail_plans))
        if not members:
            return set()
        written, failure = manager._sweep(source, reference, prefix_evolved, members)
        if failure is not None:
            warnings.warn(
                f"Shared-pass materialization of case '{manager.name}' failed ({failure});"
                " its copies take their own passes.",
                stacklevel=2,
            )
        return written

    # ---------------------------------------------------------------- what the plan asks

    def write_targets(self, a: int = 0) -> list[tuple[Save, list[int], Attribute]]:
        """Every ``Save`` copy ``a`` writes, with the extent and case state it lands at: what a write
        probe must open to be the run's own verdict. Behind an ``Expand`` the stages after the marker
        fold the COPY's grid."""
        manager = self.manager
        spatial = [int(extent) for extent in manager.base_shape[1:]]
        attributes = Attribute(manager.cache_attributes_bak[0])
        targets: list[tuple[Save, list[int], Attribute]] = []
        for stage in manager.chain_stages(a):
            spatial = manager._fold_case_state(stage, spatial, attributes)
            if isinstance(stage, Save):
                targets.append((stage, list(spatial), Attribute(attributes)))
        return targets

    def sub_cap_sweep(self) -> bool:
        """Whether this case's landing is swept in more than one block AND a stage of its chain can
        show it in the values: a resample whose map does not factorise, or a draw sampling through
        an affine. Pointwise and separable chains land the same bytes whatever the decomposition."""
        manager = self.manager
        if not any(
            extent < segment.landing[axis]
            for segment in self.manager.sweep_segments() or []
            for axis, extent in enumerate(manager.sizer_for(segment).sweep_tile())
        ):
            return False
        for stage in manager.chain_stages(0):
            if isinstance(stage, Resample) and stage.slab_height_sensitive(manager.name):
                return True
            if (
                isinstance(stage, AugmentedStage)
                and stage.patch_locality(manager.stored_attributes).kind is LocalityKind.REGRID
            ):
                return True
        return False

    def plan_notes(self, group_dest: str) -> list[str]:
        """The notes the chain's transforms ask the plan to print (``Transform.plan_note``), each
        stage asked about ITS OWN input. Only as far as a ``Reduce``: past it the grid is the
        cohort's."""
        manager = self.manager
        shape = [int(extent) for extent in manager.base_shape[1:]]
        attributes = Attribute(manager.stored_attributes)
        notes: list[str] = []
        for stage in manager.transforms:
            if isinstance(stage, Reduce):
                break
            if not isinstance(stage, Transform):
                continue  # a draw has nothing to add: what its copies cost is the plan's regime column
            note = stage.plan_note(group_dest, manager.name, list(shape), Attribute(attributes))
            if note is not None and note not in notes:
                notes.append(note)
            shape = manager._fold_case_state(stage, shape, attributes)
        return notes

    def peak_case_bytes(self) -> int:
        """The largest single tensor the whole-volume path holds: the chain's shapes folded through
        each stage's own map, at ``CASE_ELEMENT_BYTES`` per element. Headers only."""
        if self._peak_case_bytes is None:
            manager = self.manager
            channels = int(manager.base_shape[0])
            peak = int(np.prod(manager.base_shape, dtype=np.int64))

            # Copy 0 carries no draw, copy 1 carries them all, and a draw widens the grid as readily
            # as a transform does, so the peak is the largest either walk holds.
            copies = [0]
            if manager._expand is not None or any(group.nb for group in manager.data_augmentations_list):
                copies.append(1)
            for a in copies:
                spatial = [int(extent) for extent in manager.base_shape[1:]]
                attributes = Attribute(manager.stored_attributes)
                for stage in manager.chain_stages(a):
                    spatial = manager._fold_case_state(stage, list(spatial), attributes)
                    peak = max(peak, channels * int(np.prod(spatial, dtype=np.int64)))
            self._peak_case_bytes = peak * CASE_ELEMENT_BYTES
        return self._peak_case_bytes

    def fallback_working_set_bytes(self) -> int:
        """The bytes a whole-volume fallback of this case holds at its peak: the case, its in-flight
        copy, and the widest stage's own buffers, all sized on the largest intermediate."""
        return int(self.peak_case_bytes() * (FALLBACK_INFLIGHT_FACTOR + self.manager.working_multiple()))

    def reads_its_source_whole(self, a: int = 0, apply_augmentations: bool = False) -> bool | None:
        """Whether a sweep of this case would decode its stored source whole for every region: the
        store serves no bounded region read (a gzipped NIfTI). ``None`` when the chain cannot stream;
        a Save cache still to write lands on a store serving bounded reads, so it never counts."""
        segments = self.manager.sweep_segments(a, apply_augmentations)
        if segments is None:
            return None
        return any(
            segment.dataset.is_dataset_exist(segment.group, segment.entry)
            and not segment.dataset.bounded_region_reads(segment.group, segment.entry)
            for segment in segments
        )
