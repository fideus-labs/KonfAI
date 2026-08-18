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

A :class:`~konfai.data.patching.DatasetManager` plans and replays a case's chain (the patch and
region reads training, prediction and evaluation share). :class:`CaseMaterializer` is the write
side of that same machinery for the dataset-preparation workflow: it drives one case's ``Save``
outputs to disk by the cheapest route the plan allows (the streamed slab sweep, the whole-volume
load, or one shared read pass across the copies of an ``Expand``), prices those routes for the plan
(peak bytes, working set, predicted source re-reads), and answers why a copy sweeps alone. It
holds no read state of its own: the rewrite flag, the swept-entry ledger and the sweep-failure
reason live on the manager, because a streamed read of a pending ``Save`` sweeps it there too
(``DatasetManager._stream_ready``), training included.
"""

import contextlib
import warnings
from collections.abc import Iterator

import numpy as np
import torch

from konfai.data.patching import (
    CASE_ELEMENT_BYTES,
    FALLBACK_INFLIGHT_FACTOR,
    DatasetManager,
    RegionWriter,
    Stage,
    _open_sweep_stream,
    _PatchStreamSource,
    _PendingSweep,
    _ReadStagePlan,
    _require_channel_first,
    _stage_name,
    _sweep_header,
    _sweep_targets,
)
from konfai.data.transform import LocalityKind, Save, Transform
from konfai.utils.dataset import Attribute, Dataset
from konfai.utils.errors import PatchError


class CaseMaterializer:
    """Write one case's chain to disk over its manager's plan/replay API, and price the routes.

    Built per case (a :class:`~konfai.data.patching.DatasetManager`) and kept for the run: the
    peak-bytes fold is memoized here, so the plan, the shards and the run all read the same figure
    without re-folding the chain.
    """

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
        and only here: this same machinery loads training cases inside DataLoader workers, where a
        CUDA default would be wrong; the transformer's rank is the one caller that knows its
        device)."""
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
        release: bool = True,
    ) -> bool:
        """Write this case's chain to disk by the cheapest path that can; returns whether it streamed.

        The streamed path sweeps every unsatisfied :class:`Save` slab by slab; when the plan refuses
        (a WHOLE_VOLUME stage, a destination without region writes, a failed sweep) the whole-volume
        load writes the same caches, same bytes, more memory. ``allow_fallback=False`` raises instead
        of loading; ``prefer_whole`` is the plan's LOAD choice (no fallback, no refusal);
        ``rewrite=True`` recomputes the case and renames over the old entries (``--overwrite``). A
        chain whose caches all exist writes nothing (the per-case resume). The loaded volume is
        released before returning unless ``release=False`` (the expansion engine's solo copies share
        the assembled prefix).
        """
        manager = self.manager
        with self._materialization(rewrite, fallback_budget_bytes, device):
            # A chain with an Expand materializes a COPY, whose draw must be part of the plan;
            # without one it materializes the case itself, and augmentations have nothing to do
            # with writing.
            apply_augmentations = manager._expand is not None and a > 0
            if prefer_whole:
                # The plan chose to LOAD: the case fits its budget and streaming would re-read the
                # source (:meth:`predicted_stream_read_factor`). A choice, not a fallback: nothing
                # failed, so ``allow_fallback`` is not consulted; the budget check stays as the belt.
                self._enforce_fallback_budget(fallback_budget_bytes)
                self._assemble_and_write(a)
                if release:
                    manager.unload()
                return False
            if manager._stream_ready(a, apply_augmentations=apply_augmentations):
                return True
            if not allow_fallback:
                raise PatchError(
                    f"Case '{manager.name}' would take the whole-volume path at run time.",
                    manager.stream_refusal(a, apply_augmentations)
                    or manager._sweep_failure
                    or "the chain cannot stream.",
                    "Nothing was written for this case; the caller forbids the whole-volume fallback.",
                )
            self._enforce_fallback_budget(fallback_budget_bytes)
            self._assemble_and_write(a)
            if release:
                manager.unload()
            return False

    def _enforce_fallback_budget(self, fallback_budget_bytes: float | None) -> None:
        if fallback_budget_bytes is None:
            return
        # The plan's promise holds at run time too: a case whose sweep failed here (a refusal the
        # probe could not see) must not assemble a volume the budget cannot hold.
        case_bytes = self.fallback_working_set_bytes()
        if case_bytes > fallback_budget_bytes:
            raise PatchError(
                f"Case '{self.manager.name}' fell back to the whole-volume path at run time and its"
                f" working set (~{case_bytes / 2**30:.2f} GiB) exceeds the per-rank budget.",
                "Nothing was written for this case. Raise 'memory_budget' or make the chain streamable.",
            )

    def _assemble_and_write(self, a: int) -> None:
        """The whole-volume fallback: assemble and write every Save. The caller releases.

        Without an :class:`Expand` this is the classic load of the full chain. With one, the shared
        part is assembled once (``load`` keeps it, so the copies of one case reuse the same tensor
        across calls) and only the copy's draw and the per-copy stages run per copy, writing their
        Saves under the copy's name.
        """
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
    ) -> dict[int, str]:
        """Write the :class:`Expand` copies of this case; returns the regime each took,
        ``'stream-shared'`` | ``'stream'`` | ``'whole-volume'``.

        Copies whose per-copy stages are all pointwise share ONE read pass (each slab is read and
        carried through the shared prefix once, then every copy applies its draw into its own
        stream); a copy whose draw reads regions sweeps its own pass; a copy that cannot stream falls
        back to the whole volume, whose shared part is assembled once for all such copies.
        """
        manager = self.manager
        if manager._expand is None:
            raise PatchError(
                f"materialize_copies() on case '{manager.name}' whose chain declares no Expand.",
                "Use materialize() for a 1-to-1 chain; copies only exist behind an Expand marker.",
            )
        with self._materialization(rewrite, fallback_budget_bytes, device):
            manager._require_statistics()
            verdicts: dict[int, str] = {}
            if not copies:
                return verdicts

            # The caches the copies share (pre-Expand Saves) are swept once, first: every copy's
            # plan reads through them, and sweeping them inside each copy's own pass would redo
            # the work.
            first = manager._resolve_patch_stream_source(copies[0], apply_augmentations=True)
            if first is not None:
                shared_pending = [sweep for sweep in first.pending_sweeps if sweep.entry == manager.name]
                if shared_pending:
                    for sweep in shared_pending:
                        # Chained here as on the per-case path: past a failure the next sweep reads
                        # a cache nobody wrote and overwrites the recorded reason with its own
                        # symptom.
                        if not manager._materialize_save(sweep):
                            break
                    manager._invalidate_stream_plans()

            shared: list[tuple[int, _PendingSweep]] = []
            solo: list[int] = []
            fallback: list[int] = []
            for a in copies:
                source = manager._resolve_patch_stream_source(a, apply_augmentations=True)
                if source is None:
                    fallback.append(a)
                    continue
                per_copy = [sweep for sweep in source.pending_sweeps if sweep.entry != manager.name]
                if len(per_copy) != len(source.pending_sweeps):
                    # A shared cache is still pending (its sweep failed): the per-copy path will
                    # retry and fall back on its own terms.
                    solo.append(a)
                elif not per_copy:
                    verdicts[a] = "stream"  # everything this copy writes is already on disk
                elif len(per_copy) == 1 and self._pointwise_tail(per_copy[0]):
                    shared.append((a, per_copy[0]))
                else:
                    solo.append(a)

            if len(shared) == 1:
                # One copy shares with nobody: its own sweep is the same work without the extra
                # clone.
                solo.append(shared.pop()[0])
            if shared:
                written = self._materialize_shared_pass(shared)
                for a, _sweep in shared:
                    if a in written:
                        verdicts[a] = "stream-shared"
                    else:
                        solo.append(a)
                if written:
                    manager._invalidate_stream_plans()

            for a in sorted(solo):
                verdicts[a] = (
                    "stream"
                    if self.materialize(
                        a,
                        rewrite,
                        fallback_budget_bytes=fallback_budget_bytes,
                        allow_fallback=allow_fallback,
                        device=device,
                        release=False,  # the copies share the assembled prefix; released once below
                    )
                    else "whole-volume"
                )
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
                    verdicts[a] = "whole-volume"
            manager.unload()
            return verdicts

    def _pointwise_tail(self, sweep: _PendingSweep) -> bool:
        """Whether everything per-copy in this sweep is a pure value map on its slab.

        That is the shared-pass criterion: a pointwise tail pulls exactly the slab the shared prefix
        landed on, so every such copy can consume one read. A region draw pulls its own geometry and
        a GLOBAL_STAT reads a statistic of ITS input; both get their own pass.
        """
        return all(plan.kind is LocalityKind.POINTWISE for plan in sweep.stage_plans[sweep.copy_stage_start :])

    def _materialize_shared_pass(self, shared: list[tuple[int, _PendingSweep]]) -> set[int]:
        """One read pass, N write streams: each slab is read and computed through the shared prefix
        once, each copy applies its pointwise tail to a clone and writes into its own stream (peak:
        the pulled slab plus one block). On failure every stream is aborted and the copies take
        their own passes; returns the copies written.
        """
        manager = self.manager
        reference = shared[0][1]
        # Defensive: the regime only holds when every copy lands the same grid from the same source
        # and writes into the same store. Built that way, but a draw that lies about its shape map
        # would corrupt N entries at once here, so the mismatch is checked rather than assumed.
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
        prefix = list(reference.stages[: reference.copy_stage_start])
        spatial = list(reference.out_spatial)
        source_spatial = [int(extent) for extent in reference.source_shape[1:]]
        streamable, prefix_plans, prefix_evolved, _refusal = manager._plan_stream_region(
            0,
            prefix,
            reference.source_dataset,
            reference.source_group,
            reference.source_entry,
            Attribute(reference.base_attributes),
            source_spatial,
            landing_shape=spatial,
        )
        if not streamable:
            return set()
        source = _PatchStreamSource(
            reference.source_dataset,
            reference.source_group,
            reference.source_entry,
            list(reference.source_shape),
            prefix,
            prefix_plans,
        )
        # Each copy's header is its OWN full-segment plan state: the tail's geometry and inversion
        # keys belong to the copy, not to the shared prefix.
        active: list[tuple[int, _PendingSweep, list[Stage], Attribute]] = []
        for a, sweep in shared:
            ok, _plans, evolved, _reason = manager._plan_stream_region(
                0,
                sweep.stages,
                sweep.source_dataset,
                sweep.source_group,
                sweep.source_entry,
                Attribute(sweep.base_attributes),
                [int(extent) for extent in sweep.source_shape[1:]],
                landing_shape=list(sweep.out_spatial),
            )
            if ok:
                active.append((a, sweep, list(sweep.stages[sweep.copy_stage_start :]), evolved))
        if not active:
            return set()
        rows = manager._sweep_rows(spatial, int(reference.source_shape[0]))
        sweeps = {a: sweep for a, sweep, _tail, _evolved in active}
        headers: dict[int, Attribute] = {}
        writer = RegionWriter(lambda a, block: _open_sweep_stream(sweeps[a], block, spatial, rows, headers[a]))
        try:
            for slab_index, target in enumerate(_sweep_targets(spatial, rows)):
                tensor, slab_attribute, keys_before = manager._replay_streamed_region(
                    source,
                    target,
                    Attribute(reference.base_attributes),
                    Attribute(prefix_evolved) if slab_index == 0 else None,
                )
                for a, sweep, tail, evolved in active:
                    copy_tensor = tensor.clone() if len(active) > 1 else tensor
                    scope = Attribute(slab_attribute)
                    for stage in tail:
                        copy_tensor = stage(manager.name, copy_tensor, scope)
                    block = copy_tensor.cpu().numpy()
                    _require_channel_first(block, spatial, f"A per-copy stage of '{sweep.group}/{sweep.entry}'")
                    if a not in headers:
                        headers[a] = _sweep_header(evolved, scope, keys_before)
                    try:
                        writer.write(a, (slice(0, int(block.shape[0])), *target), block)
                    except LookupError:
                        raise PatchError(
                            f"destination '{sweep.destination.filename}' refused the region write"
                            f" of '{sweep.group}/{sweep.entry}' after accepting its plan.",
                            "h5 and omezarr always serve region writes; mha only with image geometry.",
                        ) from None
            written = writer.opened
            writer.close()
            for a in written:
                sweep = sweeps[a]
                manager._swept_entries.add((str(sweep.destination.filename), sweep.group, sweep.entry))
            return written
        except BaseException as exception:
            writer.abort(exception)
            if not isinstance(exception, Exception):
                raise  # an interrupt is not a sweep failure: no fallback, and no .tmp left behind
            warnings.warn(
                f"Shared-pass materialization of case '{manager.name}' failed"
                f" ({type(exception).__name__}: {exception}); its copies take their own passes.",
                stacklevel=2,
            )
            return set()

    # ---------------------------------------------------------------- what the plan asks

    def write_targets(self, a: int = 0) -> list[tuple[Save, list[int], Attribute]]:
        """Every ``Save`` copy ``a`` writes, with the extent and case state it lands at.

        What a write probe must open to be the run's own verdict: behind an ``Expand`` the stages
        after the marker fold the COPY's grid, so probing the chain with the marker left in it
        would validate the pre-draw extent and call a destination good for a shape it never sees.
        """
        manager = self.manager
        spatial = [int(extent) for extent in manager.base_shape[1:]]
        attributes = Attribute(manager.cache_attributes_bak[0])
        targets: list[tuple[Save, list[int], Attribute]] = []
        for stage in manager.chain_stages(a):
            spatial = manager._fold_case_state(stage, spatial, attributes)
            if isinstance(stage, Save):
                targets.append((stage, list(spatial), Attribute(attributes)))
        return targets

    def peak_case_bytes(self) -> int:
        """The largest single tensor the whole-volume path holds: the chain's shapes folded through
        ``transform_shape`` (a pad or an upsample holds its largest intermediate), at
        ``CASE_ELEMENT_BYTES`` per element. Headers only, so a floor for a stage that widens the
        dtype beyond what it declares."""
        if self._peak_case_bytes is None:
            manager = self.manager
            spatial = [int(extent) for extent in manager.base_shape[1:]]
            channels = int(manager.base_shape[0])
            peak = int(np.prod(manager.base_shape, dtype=np.int64))
            attributes = Attribute(manager.stored_attributes)
            for stage in manager.transforms:
                if not isinstance(stage, Transform):
                    continue
                spatial = manager._fold_case_state(stage, list(spatial), attributes)
                peak = max(peak, channels * int(np.prod(spatial, dtype=np.int64)))
            self._peak_case_bytes = peak * CASE_ELEMENT_BYTES
        return self._peak_case_bytes

    def working_multiple(self) -> float:
        """What the whole-volume path allocates beyond the case and its in-flight copy, in
        volumes-worth: the largest a stage of the chain declares (``Transform.working_multiple``)."""
        return max(
            (float(stage.working_multiple) for stage in self.manager.transforms if isinstance(stage, Transform)),
            default=0.0,
        )

    def fallback_working_set_bytes(self) -> int:
        """The bytes a whole-volume fallback of this case holds at its peak: the case, its in-flight
        copy, and the widest stage's own buffers, all sized on the largest intermediate."""
        return int(self.peak_case_bytes() * (FALLBACK_INFLIGHT_FACTOR + self.working_multiple()))

    def predicted_stream_read_factor(self, a: int = 0, apply_augmentations: bool = False) -> float | None:
        """About how many times the streamed route reads the source (a halo re-reads its overlap, a
        regrid pulls each slab's window, a store without bounded reads decodes the volume once per
        slab), from the plan's own pull maps: the number the route is chosen with. ``None`` when the
        chain cannot stream."""
        source = self.manager._resolve_patch_stream_source(a, apply_augmentations)
        if source is None:
            return None
        # Each unsatisfied Save sweeps ITS source; past the last boundary the chain reads the
        # materialized cache. The route is priced by the dominant segment: a max, not a sum --
        # the segments read different stores, and one that re-reads is the cost either way.
        factors = [
            self._segment_read_factor(
                sweep.source_dataset,
                sweep.source_group,
                sweep.source_entry,
                [int(extent) for extent in sweep.source_shape],
                list(sweep.out_spatial),
                sweep.stage_plans,
            )
            for sweep in source.pending_sweeps
        ]
        # The head segment past the last boundary is read only if stages follow it: a chain
        # ending on a Write leaves nothing to read from the destination, which does not exist yet.
        if source.stage_plans:
            landed = list(source.stage_plans[-1].out_shape)
            factors.append(
                self._segment_read_factor(
                    source.dataset, source.group, source.entry, list(source.shape), landed, source.stage_plans
                )
            )
        return max(factors) if factors else 1.0

    def _segment_read_factor(
        self,
        dataset: Dataset,
        group: str,
        entry: str,
        source_shape: list[int],
        landed: list[int],
        plans: tuple[_ReadStagePlan, ...],
    ) -> float:
        """One segment's reads over its source's voxels, slab by slab through the plan's own pulls."""
        rows = self.manager._sweep_rows(list(landed), int(source_shape[0]))
        # A source that is not on disk yet is a Save cache this run sweeps first, onto a store that
        # serves region writes, and every such store serves bounded reads: priced as bounded, not
        # as the pessimistic answer a missing entry gets.
        if dataset.is_dataset_exist(group, entry) and not dataset.bounded_region_reads(group, entry):
            return float(max(1, -(-landed[0] // rows)))  # every slab decodes the whole store
        read = 0
        for start in range(0, landed[0], rows):
            span = [slice(start, min(start + rows, landed[0])), *(slice(0, extent) for extent in landed[1:])]
            for plan in reversed(plans):
                span = list(plan.pull(tuple(span))) if plan.pull is not None else span
            read += int(np.prod([max(0, part.stop - part.start) for part in span], dtype=np.int64))
        return float(read) / float(max(1, int(np.prod(source_shape[1:], dtype=np.int64))))

    def expansion_solo_reason(self, a: int) -> str | None:
        """Why copy ``a`` cannot join the shared read pass, for the plan, never a write.

        ``None`` when it can (or when nothing is pending for it). The reason names the first
        per-copy stage whose declared locality is not pointwise: that is the stage whose pull makes
        this copy's read geometry its own.
        """
        source = self.manager._resolve_patch_stream_source(a, apply_augmentations=True)
        if source is None:
            return None
        per_copy = [sweep for sweep in source.pending_sweeps if sweep.entry != self.manager.name]
        if len(per_copy) == 1 and self._pointwise_tail(per_copy[0]):
            return None
        if not per_copy:
            return None
        if len(per_copy) > 1:
            return "the copy's chain crosses more than one per-copy Save, so it sweeps its own passes."
        sweep = per_copy[0]
        for stage, plan in zip(
            sweep.stages[sweep.copy_stage_start :], sweep.stage_plans[sweep.copy_stage_start :], strict=False
        ):
            if plan.kind is not LocalityKind.POINTWISE:
                return (
                    f"stage '{_stage_name(stage)}' of this copy declares {plan.kind.name},"
                    " so its read geometry is the draw's own and it sweeps its own pass."
                )
        return "the copy's plan is incomplete; it sweeps its own pass."
