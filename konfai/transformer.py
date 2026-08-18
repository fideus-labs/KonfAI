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

"""The TRANSFORM workflow: dataset preparation, a declared transform chain applied to every case.

The engine is :class:`~konfai.data.materialize.CaseMaterializer` (the streamed Save sweep with its
whole-volume fallback, over each case's manager), and the product is the plan: before a byte is
written, every (case, chain) is planned on the launcher, each output destination is probed with a
real region-write open (created then removed, so the verdict is the run's own), and the whole thing
opens the run's log, the console getting the one-line summary of it.
Nothing here falls back silently: a chain that cannot stream says which stage refused and why, and
``on_fallback`` decides whether that is information, a warning, or an error.
"""

import contextlib
import json
import os
import shutil
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast

import numpy as np
import torch
import tqdm

from konfai import config_file, transforms_directory
from konfai.data.case_reduction import CaseReduction, split_chain
from konfai.data.data_manager import DataTransform
from konfai.data.materialize import CaseMaterializer, Regime, Verdict
from konfai.data.patching import (
    CASE_ELEMENT_BYTES,
    FALLBACK_INFLIGHT_FACTOR,
    DatasetManager,
    save_destination,
)
from konfai.data.transform import Save, split_expand
from konfai.utils.budget import format_bytes, node_local_ranks
from konfai.utils.config import apply_config, config, strict_config
from konfai.utils.dataset import Attribute, Dataset
from konfai.utils.errors import ConfigError, TransformerError
from konfai.utils.runtime import (
    DistributedObject,
    State,
    _materialized_config,
    configure_workflow_environment,
    get_device,
    record,
    run_distributed_app,
)

_PROBE_ENTRY = "__konfai_plan_probe__"


@dataclass(frozen=True)
class TransformPlanEntry:
    """One (case, chain) line of the plan: the verdict the run itself will act on."""

    case: str  # the case name, the copy's entry name behind an Expand, or a reduction's output name
    group_src: str
    group_dest: str
    verdict: Verdict
    reason: str | None
    #: The largest single tensor the case's chain holds (a reduction: its resident regions).
    case_bytes: int
    #: What this entry holds at its peak, the figure the budget bounds: the engine's whole-volume
    #: working set (the case, one in-flight copy, the widest stage's own buffers), or a
    #: reduction's regions.
    working_set_bytes: int
    #: The cases folded into one, for a reduction. Empty for an ordinary per-case entry. Printed in
    #: full: a reduction whose case list silently changed writes a different volume under the same
    #: name, and nothing about the output would look wrong.
    reduced: tuple[str, ...] = ()
    #: The case this entry is an Expand copy of ("" for anything that is not a copy).
    expanded_from: str = ""
    #: How a STREAM copy is read; None for non-copy entries and non-STREAM verdicts.
    regime: Regime | None = None


@dataclass
class TransformPlan:
    """The plan as an object: what ``--plan`` prints and ``setup`` enforces."""

    entries: list[TransformPlanEntry]
    budget_bytes: float
    budget_desc: str
    world_size: int
    dropped_cases: dict[str, int]
    dtype_hypothesis: str
    #: What the stages themselves asked the plan to say (``Transform.plan_note``): a cost the
    #: columns above have no room for. Part of the plan, not of the run, so ``--plan`` carries it:
    #: a note only worth reading after the bytes are written is not worth printing.
    notes: tuple[str, ...] = ()
    #: Per (group_src, group_dest): the chain spelled out with its destination: the one fact a
    #: reader wants from a plan line ("what runs, and where does it land").
    chain_labels: dict[tuple[str, str], str] = field(default_factory=dict)

    @property
    def fallback_entries(self) -> list[TransformPlanEntry]:
        return [entry for entry in self.entries if entry.verdict is Verdict.WHOLE_VOLUME]

    @property
    def refused_entries(self) -> list[TransformPlanEntry]:
        """Reductions that cannot stream. Unlike a per-case chain they have no whole-volume path to
        fall back to, so these refuse the run whatever ``on_fallback`` says."""
        return [entry for entry in self.entries if entry.verdict is Verdict.REFUSED]

    def budget_violations(self) -> list[TransformPlanEntry]:
        """Entries whose working set exceeds the per-rank budget: a whole-volume fallback (the case,
        its in-flight copy, the widest stage's buffers) or a reduction whose regions are down to one
        row. A headers-only estimate; the report prints the margin."""
        candidates = [
            entry for entry in self.entries if entry.verdict in (Verdict.WHOLE_VOLUME, Verdict.LOAD, Verdict.REDUCE)
        ]
        return [entry for entry in candidates if entry.working_set_bytes > self.budget_bytes]

    def summary(self) -> str:
        """The plan in one line: what the run will do, and where to read the rest.

        The console form, bounded: the full plan's notes and case lists grow with the cohort, so it
        stays in the run's log (and ``--plan`` prints it on demand). What is folded away is counted here,
        so nothing goes missing in silence."""
        counts = Counter(entry.verdict for entry in self.entries)
        verdicts = ", ".join(f"{count} {verdict}" for verdict, count in sorted(counts.items())) or "nothing to do"
        dropped = sum(self.dropped_cases.values())
        return (
            f"[KonfAI] plan over {self.world_size} rank(s) | {len(self.entries)} entr(ies): {verdicts}"
            f" | per-rank budget {format_bytes(self.budget_bytes)} ({self.budget_desc})"
            + (f" | {len(self.notes)} note(s)" if self.notes else "")
            + (f" | {dropped} case(s) dropped" if dropped else "")
        )

    def report(self) -> str:
        """The full plan as text: what opens the run's log, and what ``--plan`` prints on demand."""
        lines = [self._header(), *self._dropped_lines(), *(f"[KonfAI] NOTE: {note}" for note in self.notes)]
        by_chain: dict[tuple[str, str], list[TransformPlanEntry]] = {}
        for entry in self.entries:
            by_chain.setdefault((entry.group_src, entry.group_dest), []).append(entry)
        for (group_src, group_dest), entries in by_chain.items():
            label = self.chain_labels.get((group_src, group_dest), "")
            lines.extend(self._chain_lines(f"{group_src} -> {group_dest}{f' ({label})' if label else ''}", entries))
        return "\n".join(lines)

    def _header(self) -> str:
        return (
            f"[KonfAI] plan over {self.world_size} rank(s) | per-rank budget"
            f" {format_bytes(self.budget_bytes)} ({self.budget_desc})"
            f" | fallback working set = case x {CASE_ELEMENT_BYTES} B x ({FALLBACK_INFLIGHT_FACTOR}"
            f" + the widest stage's own buffers), headers-only estimate | output dtype/channels assumed"
            f" {self.dtype_hypothesis} until the first slab"
        )

    def _dropped_lines(self) -> list[str]:
        return [
            f"[KonfAI] {dropped} case(s) of '{group_src}' are DROPPED: the run keeps"
            " the cases every groups_src shares, minus what 'subset' excludes."
            for group_src, dropped in sorted(self.dropped_cases.items())
            if dropped
        ]

    def _chain_lines(self, chain: str, entries: list[TransformPlanEntry]) -> list[str]:
        """One chain's block: its reduction, or its verdict counts with the reasons behind them."""
        reductions = [entry for entry in entries if entry.reduced]
        if reductions:
            return [line for entry in reductions for line in self._reduction_lines(chain, entry)]
        expanded = any(entry.expanded_from for entry in entries)
        return [
            self._expansion_line(chain, entries) if expanded else self._case_line(chain, entries),
            *self._reason_lines(entries, expanded),
            *self._worst_fallback_lines(entries),
        ]

    def _reduction_lines(self, chain: str, entry: TransformPlanEntry) -> list[str]:
        lines = [f"  {chain}: REDUCE {len(entry.reduced)} case(s) -> 1 output '{entry.case}': {entry.verdict}"]
        if entry.reason:
            lines.extend(f"    {line}" for line in entry.reason.splitlines())
        if entry.verdict is Verdict.REDUCE:
            lines.append(
                f"    peak ~= {format_bytes(entry.working_set_bytes)} vs per-rank budget"
                f" {format_bytes(self.budget_bytes)}"
            )
        lines.append(f"    cases: {', '.join(entry.reduced)}")
        return lines

    @staticmethod
    def _expansion_line(chain: str, entries: list[TransformPlanEntry]) -> str:
        counts = Counter(entry.verdict for entry in entries)
        expanded = [entry for entry in entries if entry.expanded_from]
        cases = len({entry.expanded_from for entry in expanded})
        shared = sum(1 for entry in expanded if entry.regime is Regime.SHARED)
        solo = sum(1 for entry in expanded if entry.regime is Regime.SOLO)
        return (
            f"  {chain}: EXPAND {cases} case(s) ->"
            f" {len(expanded)} cop(ies): {shared} STREAM (shared read pass),"
            f" {solo} STREAM (own pass), {counts[Verdict.WHOLE_VOLUME]} WHOLE-VOLUME,"
            f" {counts[Verdict.SKIP]} SKIP (copy already written)"
        )

    @staticmethod
    def _case_line(chain: str, entries: list[TransformPlanEntry]) -> str:
        counts = Counter(entry.verdict for entry in entries)
        return (
            f"  {chain}: {len(entries)} case(s) --"
            f" {counts[Verdict.STREAM]} STREAM, {counts[Verdict.LOAD]} LOAD,"
            f" {counts[Verdict.WHOLE_VOLUME]} WHOLE-VOLUME,"
            f" {counts[Verdict.SKIP]} SKIP (output already written)"
        )

    @staticmethod
    def _reason_lines(entries: list[TransformPlanEntry], expanded: bool) -> list[str]:
        """Why a case is not a plain STREAM (fallback, LOAD, or a copy's own pass), each reason once
        with how many entries it covers."""
        reasons: Counter[str] = Counter()
        for entry in entries:
            if not entry.reason:
                continue
            if entry.verdict in (Verdict.WHOLE_VOLUME, Verdict.LOAD):
                reasons[f"{entry.verdict}: {entry.reason}"] += 1
            elif entry.regime is Regime.SOLO:
                reasons[f"own pass: {entry.reason}"] += 1
        unit = "cop(ies)" if expanded else "case(s)"
        return [f"    ({count} {unit}) {reason}" for reason, count in reasons.items()]

    def _worst_fallback_lines(self, entries: list[TransformPlanEntry]) -> list[str]:
        fallbacks = [entry.working_set_bytes for entry in entries if entry.verdict is Verdict.WHOLE_VOLUME]
        if not fallbacks:
            return []
        return [
            f"    worst fallback case ~= {format_bytes(max(fallbacks))}"
            f" vs per-rank budget {format_bytes(self.budget_bytes)}"
        ]


@dataclass
class WorkItem:
    """One unit of the run, the same object the plan lines, the shards weigh and a rank executes.

    A plain case, an Expand case with all its copies (the engine shares one read pass across them,
    which a per-copy item could not), or a reduction (every member folded into one output).
    """

    kind: Literal["case", "expansion", "reduction"]
    group_src: str
    group_dest: str
    #: The engine over the case's own manager, or over every member of a reduction. The first one
    #: answers for the chain: dtype, label, terminal Write.
    engines: list[CaseMaterializer]
    #: The terminal Write's store and group: where the item's entries land, and are resumed from.
    destination: Dataset
    group: str
    copies: int = 1  # Expand copies of the case; 1 otherwise
    reduction: CaseReduction | None = None

    @property
    def engine(self) -> CaseMaterializer:
        return self.engines[0]

    @property
    def manager(self) -> DatasetManager:
        return self.engines[0].manager

    @property
    def label(self) -> str:
        """The item as the console names it."""
        return f"Reduce -> {self.group_dest}" if self.kind == "reduction" else self.manager.name

    @property
    def chain_label(self) -> str:
        """The chain spelled out with its destination: what runs, and where it lands."""
        names = [type(stage).__name__ for stage in self.manager.transforms]
        names[-1] += f" {self.destination.filename}:{self.destination.file_format}"
        return " -> ".join(names)

    @property
    def weight_bytes(self) -> int:
        """What the item costs a rank, for balancing: its members' peak bytes, per copy."""
        return sum(engine.peak_case_bytes() for engine in self.engines) * self.copies

    def pending(self, entry_name: str, overwrite: bool) -> bool:
        """Whether this entry is still to write: the one resume test the plan and the run share."""
        return overwrite or not self.destination.is_dataset_exist(self.group, entry_name)


@config()
class Transformer(DistributedObject):
    """The dataset-preparation workflow: it materializes every chain's ``Write`` outputs, plan first."""

    # The ranks share the work list and nothing else: each writes its shard and reports it.
    uses_collectives = False

    def __init__(
        self,
        name: str = "default|TRANSFORM_01",
        on_fallback: Literal["allow", "warn", "error"] = "warn",
        manual_seed: int = 0,
        dataset: DataTransform = DataTransform(),
    ) -> None:
        if os.environ["KONFAI_CONFIG_MODE"] != "Done":
            raise ConfigError("Transformer requires KONFAI_CONFIG_MODE='Done' before initialization.")
        super().__init__(name)
        self.transform_path = transforms_directory() / self.name
        self.on_fallback = on_fallback
        self.manual_seed = int(manual_seed)
        self.dataset = dataset
        # The seed reaches the Expand draws by DERIVATION, not by seeding a global RNG: each chain
        # builds its own stage objects, so what a shared RNG hands them depends on the order the
        # chains happen to be built, and an image and its mask would drift apart. Handed over before
        # prepare(), which is where the chains are bound and the draws happen.
        self.dataset.manual_seed = self.manual_seed
        # prepare() binds the chains and runs every parse-time refusal (terminal Write, output
        # collisions, writes into the source, inference transforms) before any byte is read.
        self.dataset.prepare()
        self._overwrite = False
        self._budget_bytes: float | None = None
        self._items: list[WorkItem] | None = None  # built once by _work_items(), after prepare()
        self._shards: list[list[int]] = []
        self._reductions: dict[str, CaseReduction | None] = {}
        self._planned: dict[tuple[str, str], Verdict] = {}

    def _group_src_of(self, group_dest: str) -> str:
        for group_src in self.dataset.groups_src:
            if group_dest in self.dataset.groups_src[group_src]:
                return group_src
        return "?"

    @staticmethod
    def _terminal_destination(manager: DatasetManager) -> tuple[Dataset, str]:
        # Guarded rather than indexed blindly: an empty chain is refused at parse time, but this
        # runs on every case of every plan and a bare IndexError would say nothing about why.
        terminal = manager.transforms[-1] if manager.transforms else None
        if not isinstance(terminal, Save):
            raise TransformerError(
                f"The chain writing '{manager.group_dest}' does not end with a Write, so it has no destination.",
                "End every chain with Write: {dataset: <path>[:format]}.",
            )
        return save_destination(terminal, manager.dataset, manager.group_dest)

    def _reduction(self, group_dest: str, managers: list[DatasetManager]) -> CaseReduction | None:
        """The reduction this chain declares, or ``None``. Built once per chain and kept: the plan, the
        shards and the run must execute the same object. Its members are re-managed with the
        pre-reduction stages only (a manager holding the ``Reduce`` itself cannot stream)."""
        if group_dest in self._reductions:
            return self._reductions[group_dest]
        reduction = self._build_reduction(group_dest, managers)
        self._reductions[group_dest] = reduction
        return reduction

    def _build_reduction(self, group_dest: str, managers: list[DatasetManager]) -> CaseReduction | None:
        if not managers:
            return None
        pre, reduce, post = split_chain(managers[0].transforms)
        if reduce is None:
            return None
        cases = [
            DatasetManager(
                index=manager.index,
                group_src=manager.group_src,
                group_dest=manager.group_dest,
                name=manager.name,
                dataset=manager.dataset,
                patch=None,
                transforms=list(pre),
                data_augmentations_list=[],
            )
            for manager in managers
        ]
        terminal = post[-1] if post else None
        if not isinstance(terminal, Save):
            raise TransformerError(
                f"The chain reducing into '{reduce.output}' does not end with a Write.",
                "A reduction writes one entry, so the chain carrying it must say where:"
                " Write: {dataset: <path>[:format]}.",
            )
        destination, group = save_destination(terminal, managers[0].dataset, group_dest)
        reduction = CaseReduction(
            managers=cases,
            reduce=reduce,
            post=list(post[:-1]),
            destination=destination,
            group=group,
        )
        reduction.fit_budget(self._budget_bytes)
        return reduction

    def _work_items(self) -> list[WorkItem]:
        """The run's work items, in run order: a chain's cases one by one, or its reduction as one
        item for all of them. Built once and kept: the plan, the shards and the run walk this one
        list, so the run executes exactly what the plan measured."""
        if self._items is not None:
            return self._items
        items: list[WorkItem] = []
        for group_dest, managers in self.dataset.managers.items():
            if not managers:
                continue
            group_src = self._group_src_of(group_dest)
            reduction = self._reduction(group_dest, managers)
            if reduction is not None:
                items.append(
                    WorkItem(
                        "reduction",
                        group_src,
                        group_dest,
                        [CaseMaterializer(manager) for manager in managers],
                        reduction.destination,
                        reduction.group,
                        reduction=reduction,
                    )
                )
                continue
            expansion = split_expand(managers[0].transforms)[1]
            for manager in managers:
                destination, group = self._terminal_destination(manager)
                engine = CaseMaterializer(manager)
                if expansion is None:
                    items.append(WorkItem("case", group_src, group_dest, [engine], destination, group))
                else:
                    items.append(
                        WorkItem("expansion", group_src, group_dest, [engine], destination, group, copies=expansion.nb)
                    )
        self._items = items
        return items

    @staticmethod
    def _dtype_hypothesis(manager: DatasetManager) -> np.dtype:
        """The planned output dtype: the last cast's target if the chain declares one, else float32.

        The engine takes the real dtype from the first computed slab; the plan says so instead of
        pretending to know.
        """
        dtype = np.dtype("float32")
        for transform in manager.transforms:
            declared = getattr(transform, "dtype", None)
            if declared is not None:
                try:
                    dtype = np.dtype(str(declared).replace("torch.", ""))
                except TypeError:
                    pass
        return dtype

    def _probe_write_destinations(
        self, engine: CaseMaterializer, probed: set[tuple[str, str]], a: int = 0
    ) -> str | None:
        """Open a real region-write stream on each Save/Write destination, then remove it.

        This is what makes the plan the run's own verdict: ``can_stream_data`` is a capability
        check, but the refusals that matter (rank, dtype, geometry) live in ``open_data_stream``,
        which the engine only reaches at the first computed slab. The probe pays one entry creation
        per destination and takes it back (with the store, when the probe created it).
        """
        manager = engine.manager

        def key_of(save: Save) -> tuple[str, str]:
            destination, group = save_destination(save, manager.dataset, manager.group_dest)
            return str(destination.filename), group

        # Every case of a chain shares its destinations: past the first case there is nothing to
        # probe, and the fold that sizes the probe (the whole chain, per case) is skipped.
        if all(key_of(stage) in probed for stage in manager.chain_stages(a) if isinstance(stage, Save)):
            return None
        channels = int(manager.base_shape[0])
        dtype = self._dtype_hypothesis(manager)
        for transform, spatial, attributes in engine.write_targets(a):
            key = key_of(transform)
            if key in probed:
                continue
            probed.add(key)
            destination, group = save_destination(transform, manager.dataset, manager.group_dest)
            failure = self._probe_destination(destination, group, [channels, *spatial], dtype, attributes)
            if failure is not None:
                return failure
        return None

    @staticmethod
    def _probe_destination(
        destination: Dataset, group: str, shape: list[int], dtype: np.dtype, attributes: Attribute
    ) -> str | None:
        """One real region-write open, removed immediately: the probe both chains and reductions
        share, so the plan's verdict is the run's own on every kind of output."""
        existed = destination.exists_on_disk()
        try:
            stream = destination.open_data_stream(group, _PROBE_ENTRY, shape, dtype, attributes)
            if stream is None:
                return (
                    f"destination '{destination.filename}' cannot serve region writes"
                    " (h5 and omezarr always can; mha only with image geometry)."
                )
            stream.__enter__()
            stream.abort(RuntimeError("plan probe"))
            return None
        except Exception as error:  # the probe exists to surface exactly these
            return (
                f"destination '{destination.filename}' refuses a"
                f" [{', '.join(str(extent) for extent in shape)}] {dtype} region write:"
                f" {type(error).__name__}: {error}"
            )
        finally:
            Transformer._remove_probe_entry(destination, existed)

    @staticmethod
    def _remove_probe_entry(destination: Dataset, existed_before: bool) -> None:
        """Take back what the probe created beyond its entry: the case directory of a directory store
        (``get_names`` would list it as a case) and, when the store did not exist before, the store
        itself. ``rmdir``, never ``rmtree``: anything in there is not the probe's."""
        with contextlib.suppress(OSError):
            if destination.is_directory:
                (destination.path_on_disk / _PROBE_ENTRY).rmdir()
            if not existed_before:
                path = destination.path_on_disk
                path.rmdir() if path.is_dir() else path.unlink()

    def output_destinations(self) -> list[dict[str, str]]:
        """Every chain's terminal ``Write``, as ``{group_src, group_dest, dataset, path, group, format}``.

        What the run produced, said in the run's own terms. The deliverable never lives under the
        run directory, so this is how a reader: a person, Studio's run panel, the next workflow --
        finds it without parsing the plan or re-reading the config.
        """
        destinations: list[dict[str, str]] = []
        for group_dest, managers in self.dataset.managers.items():
            if not managers:
                continue
            destination, group = self._terminal_destination(managers[0])
            destinations.append(
                {
                    "group_src": self._group_src_of(group_dest),
                    "group_dest": group_dest,
                    # `dataset` is the root as a config names it (`<dataset>:<format>` reads it back);
                    # `path` is what is on disk: the same directory, or the `.h5` file itself.
                    "dataset": str(Path(destination.filename).resolve()),
                    "path": str(destination.path_on_disk.resolve()),
                    "group": group,
                    "format": destination.file_format,
                }
            )
        return destinations

    #: Streaming re-reads at most this much of the source before a case that FITS the budget is
    #: loaded whole instead. At ~1x the two routes read the same bytes and streaming holds one slab
    #: where the load holds the case, so streaming wins the tie; past it the re-reads are the cost.
    _STREAM_WORTH_FACTOR = 1.5

    def _route(self, engine: CaseMaterializer, budget_bytes: float) -> tuple[Verdict, str | None]:
        """``STREAM`` or ``LOAD``. A case whose working set exceeds the budget streams; one that fits
        streams while streaming is no dearer than loading (``predicted_stream_read_factor``), and
        is loaded past that. Expand copies are not routed here: they share one read pass."""
        working_set = engine.fallback_working_set_bytes()
        if working_set <= budget_bytes:
            factor = engine.predicted_stream_read_factor(0, apply_augmentations=False)
            if factor is not None and factor > self._STREAM_WORTH_FACTOR:
                return Verdict.LOAD, (
                    f"fits the per-rank budget (~{format_bytes(working_set)} vs"
                    f" {format_bytes(budget_bytes)}); streaming would read ~{factor:.1f}x the source"
                )
        return Verdict.STREAM, None

    def _entry_refusal(
        self, item: WorkItem, copy_index: int, overwrite: bool, probed: set[tuple[str, str]]
    ) -> tuple[Verdict, str | None] | None:
        """What stops an entry from streaming, or ``None``: an existing output is a resume (SKIP); a
        chain the planner refuses, or whose destination fails the write probe, is the whole-volume
        pass with the refusal."""
        manager = item.manager
        # Copies exist to carry augmentation draws: a real copy index asks the chain with them applied.
        augmented = copy_index > 0
        if not item.pending(manager.copy_entry(copy_index), overwrite):
            return Verdict.SKIP, None
        if not manager.can_stream_patch(copy_index, apply_augmentations=augmented):
            return Verdict.WHOLE_VOLUME, manager.stream_refusal(copy_index, apply_augmentations=augmented)
        probe_failure = self._probe_write_destinations(item.engine, probed, copy_index)
        if probe_failure is not None:
            # The run would fail the sweep at its first slab and fall back: say so now.
            return Verdict.WHOLE_VOLUME, probe_failure
        return None

    @staticmethod
    def _entry(
        item: WorkItem, copy_index: int, verdict: Verdict, reason: str | None, regime: Regime | None = None
    ) -> TransformPlanEntry:
        manager = item.manager
        return TransformPlanEntry(
            manager.copy_entry(copy_index),
            item.group_src,
            item.group_dest,
            verdict,
            reason,
            item.engine.peak_case_bytes(),
            item.engine.fallback_working_set_bytes(),
            expanded_from=manager.name if copy_index else "",
            regime=regime,
        )

    def _plan_item(
        self, item: WorkItem, budget_bytes: float, overwrite: bool, probed: set[tuple[str, str]]
    ) -> list[TransformPlanEntry]:
        """The plan lines of one work item: one for a case or a reduction, one per copy for an expansion."""
        if item.kind == "reduction":
            return [self._plan_reduction(item, overwrite)]
        # The plan prices the very slabs the run will sweep: same budget, same rows.
        item.manager.set_memory_budget(budget_bytes)
        if item.kind == "expansion":
            return self._plan_copies(item, overwrite, probed)
        return [self._plan_case(item, budget_bytes, overwrite, probed)]

    def _plan_reduction(self, item: WorkItem, overwrite: bool) -> TransformPlanEntry:
        """SKIP, REDUCE or REFUSED: a reduction streams or it does not run, so its plan is a refusal
        or the regions it will hold, and its destination is probed like a chain's."""
        reduction = item.reduction
        assert reduction is not None  # nosec B101 - the item was built from the same predicate
        # Read off the chain, not assumed: a pointwise cast is allowed after the Reduce, and
        # the probe below opens the destination with this dtype. A constant here would test
        # a write the run never makes, and mha refuses on dtype.
        reduction_dtype = self._dtype_hypothesis(item.manager)
        reduction_plan = reduction.plan()
        skipped = not item.pending(reduction.reduce.output, overwrite)
        verdict = Verdict.SKIP if skipped else (Verdict.REDUCE if reduction_plan.streams else Verdict.REFUSED)
        reason = reduction_plan.refusal if not reduction_plan.streams else "\n".join(reduction_plan.body_lines())
        if verdict is Verdict.REDUCE:
            # A reduction has no whole-volume fallback, so a destination that would refuse
            # its stream at the first region refuses the plan: probed here, like a chain's.
            probe_failure = self._probe_destination(
                reduction.destination,
                reduction.group,
                [reduction_plan.channels, *reduction_plan.spatial],
                reduction_dtype,
                reduction.reference.landed_attributes(),
            )
            if probe_failure is not None:
                verdict, reason = Verdict.REFUSED, probe_failure
        return TransformPlanEntry(
            case=reduction.reduce.output,
            group_src=item.group_src,
            group_dest=item.group_dest,
            verdict=verdict,
            reason=reason,
            case_bytes=reduction_plan.peak_bytes,
            working_set_bytes=reduction_plan.peak_bytes,
            reduced=tuple(reduction_plan.cases),
        )

    def _plan_copies(self, item: WorkItem, overwrite: bool, probed: set[tuple[str, str]]) -> list[TransformPlanEntry]:
        """One entry per Expand copy, each with its own resume, verdict and, for STREAM, the read
        regime the engine will give it: the copies are the outputs."""
        refusals = {a: self._entry_refusal(item, a, overwrite, probed) for a in range(1, item.copies + 1)}
        routes = item.engine.classify_copies([a for a, refusal in refusals.items() if refusal is None])
        return [
            self._entry(item, a, *refusal)
            if refusal is not None
            else self._entry(item, a, Verdict.STREAM, routes[a].reason, routes[a].regime)
            for a, refusal in refusals.items()
        ]

    def _plan_case(
        self, item: WorkItem, budget_bytes: float, overwrite: bool, probed: set[tuple[str, str]]
    ) -> TransformPlanEntry:
        """A plain case: SKIP, STREAM or LOAD as :meth:`_route` prices it, or WHOLE-VOLUME with the refusal."""
        decision = self._entry_refusal(item, 0, overwrite, probed) or self._route(item.engine, budget_bytes)
        return self._entry(item, 0, *decision)

    def compute_plan(self, world_size: int = 1, overwrite: bool = False) -> TransformPlan:
        """Plan every (case, chain) on the launcher, headers plus one write probe per destination."""
        budget = self.dataset.resolved_budget()
        node_ranks = node_local_ranks(world_size)
        per_rank_budget = budget.per_rank_bytes(node_ranks)
        budget_desc = budget.description
        if not budget.shared_across_ranks and node_ranks > 1:
            # An explicit budget is per rank and never divided: the node holds N of them.
            budget_desc += f", per rank: x{node_ranks} = {format_bytes(per_rank_budget * node_ranks)} on the node"
        # Resolved before anything is planned, because a reduction sizes its regions against it --
        # the plan must measure the same run setup() will enforce.
        self._budget_bytes = per_rank_budget
        entries: list[TransformPlanEntry] = []
        probed: set[tuple[str, str]] = set()
        planned_dtypes: set[str] = set()
        chain_labels: dict[tuple[str, str], str] = {}
        sub_cap_sweeps = False  # a streamed case sweeps below the default height, and its values can show it
        for item in self._work_items():
            chain_labels[(item.group_src, item.group_dest)] = item.chain_label
            # Every manager of a group shares one chain object, so one of them answers for all.
            planned_dtypes.add(str(self._dtype_hypothesis(item.manager)))
            planned = self._plan_item(item, per_rank_budget, overwrite, probed)
            entries.extend(planned)
            if item.kind == "case" and planned[0].verdict is Verdict.STREAM and item.engine.sub_cap_sweep():
                sub_cap_sweeps = True
        dropped: dict[str, int] = {}
        kept = set(self.dataset.case_names)
        for group_src in self.dataset.groups_src:
            available: set[str] = set()
            for dataset in self.dataset.datasets.values():
                if dataset.is_group_exist(group_src):
                    available.update(dataset.get_names(group_src))
            dropped[group_src] = len(available - kept)
        dtype_hypothesis = f"{'/'.join(sorted(planned_dtypes)) or 'float32'} / source channels"
        return TransformPlan(
            entries,
            per_rank_budget,
            budget_desc,
            world_size,
            dropped,
            dtype_hypothesis,
            tuple(self._plan_notes(sub_cap_sweeps)),
            chain_labels,
        )

    #: What a streamed case whose slabs the budget lowered below the default height, and whose chain
    #: interpolates through per-voxel coordinates, is told: its values differ from a taller-slab run.
    _SUB_CAP_NOTE = (
        "the budget lowers the sweep slab height below the default, and a stage of the chain"
        " interpolates through per-voxel coordinates (a resample whose map does not factorise,"
        " a free rotation): the streamed values then differ from a taller-slab run by ~1e-5 of"
        " the data's range"
    )

    def _plan_notes(self, sub_cap_sweeps: bool) -> list[str]:
        """The notes the plan prints beside its lines: the workflow's own, then what the stages ask
        (``Transform.plan_note``), in chain order, each distinct one once."""
        notes: list[str] = [self._SUB_CAP_NOTE] if sub_cap_sweeps else []
        for group_dest, managers in self.dataset.managers.items():
            for manager in managers:
                for note in CaseMaterializer(manager).plan_notes(group_dest):
                    if note not in notes:
                        notes.append(note)
        return notes

    def setup(self, world_size: int):
        """Plan, print, enforce, shard: before any spawn, before any byte."""
        # No overwrite prompt on the run folder: it holds the logs and a config copy, both
        # rewritten in place, and prompting here would break the default per-case resume (a second
        # run would refuse because the first one left its config copy behind).
        os.makedirs(self.transform_path, exist_ok=True)
        config_copy = self.transform_path / config_file().name
        if not (config_copy.exists() and config_copy.samefile(config_file())):  # -c may name the copy itself
            shutil.copyfile(config_file(), config_copy)

        self._overwrite = os.environ.get("KONFAI_OVERWRITE", "False") == "True"
        plan = self.compute_plan(world_size, self._overwrite)
        # The full plan opens the run's log, where a run is read after the fact; the console gets the
        # one-line summary of it. One artifact, not two: the log is already the record of the run,
        # and a second file holding the same text is one more thing to know about.
        kept = record(plan.report())
        print(f"{plan.summary()}" + (f" -> full plan in {kept}" if kept else ""))
        # Where the data went, machine-readable beside the human-readable plan. This run directory
        # holds a log, a plan and a config copy, never the deliverable, which lands wherever each
        # Write pointed. Without this, the one thing a reader wants after the run is the one thing
        # nothing in the run directory names.
        (self.transform_path / "outputs.json").write_text(
            json.dumps(self.output_destinations(), indent=2) + "\n", encoding="utf-8"
        )

        self._guard_sharded_destinations(world_size)
        self._enforce_plan(plan)

        self._budget_bytes = plan.budget_bytes
        # The run executes the route the plan priced (LOAD assembles by choice, not fallback), and
        # the console only speaks when the run DEVIATES from what the plan already printed.
        self._planned = {(entry.group_dest, entry.case): entry.verdict for entry in plan.entries}
        self._shard_work(world_size)

    def _guard_sharded_destinations(self, world_size: int) -> None:
        """Multi-rank runs refuse a single-file Save destination before anything is written."""
        if world_size <= 1:
            return
        for managers in self.dataset.managers.values():
            for transform in managers[0].transforms if managers else []:
                if not isinstance(transform, Save):
                    continue
                destination, _group = save_destination(transform, managers[0].dataset, managers[0].group_dest)
                # The question here is NOT concurrent_write_safe(): that one asks whether two
                # entries of one shared store may be written at once, and answers no for
                # omezarr, which would refuse the very destination this workflow recommends. Ranks
                # shard by CASE, and a directory dataset gives each case its own file or store
                # (<root>/<case>/<group>.<ext>), so their writes are disjoint by construction.
                # Only a single-file store (h5) puts every case in one handle.
                if not destination.is_directory:
                    raise TransformerError(
                        f"--cpu {world_size}: destination '{destination.filename}' is a"
                        " single-file store, and every rank would write into the same file.",
                        "Use one process, or a directory destination (omezarr, mha, nii.gz).",
                    )

    def _enforce_plan(self, plan: TransformPlan) -> None:
        """The refusals the plan's verdicts imply, raised before any byte moves: an output over the
        budget, a whole-volume fallback under ``on_fallback: error``, a reduction that cannot
        stream. Nothing here recomputes a verdict; it only speaks for the plan."""
        violations = plan.budget_violations()
        if violations:
            worst = max(violations, key=lambda entry: entry.working_set_bytes)
            what = (
                f"the reduction into '{worst.case}' folds {len(worst.reduced)} case(s) one region at"
                " a time, and one region already exceeds the budget"
                if worst.reduced
                else f"case '{worst.case}' ({worst.reason or 'the chain cannot stream'})"
            )
            remedy = (
                "Nothing was written. Raise 'memory_budget', reduce fewer cases at once, or use an"
                " incremental operator (Mean folds one case at a time; Median and Concat cannot)."
                if worst.reduced
                else "Nothing was written. Make the chain streamable, raise 'memory_budget', or use"
                " an h5/omezarr destination."
            )
            raise TransformerError(
                f"{len(violations)} output(s) do not fit the per-rank budget"
                f" ({format_bytes(plan.budget_bytes)}): worst is"
                f" ~{format_bytes(worst.working_set_bytes)}: {what}.",
                remedy,
            )
        if self.on_fallback == "error" and plan.fallback_entries:
            first = plan.fallback_entries[0]
            raise TransformerError(
                f"{len(plan.fallback_entries)} case(s) would take the whole-volume path and"
                f" on_fallback is 'error'. First: case '{first.case}' ({first.group_src} ->"
                f" {first.group_dest}): {first.reason or 'the chain cannot stream'}",
                "Nothing was written. Make the chain streamable, or set on_fallback to 'warn' or"
                " 'allow' to accept the whole-volume path.",
            )
        if plan.refused_entries:
            first = plan.refused_entries[0]
            reduction = self._reduction(first.group_dest, self.dataset.managers.get(first.group_dest, []))
            # A grid disagreement gets its own remedy: a Save changes nothing about the grids, so
            # the generic advice would send the reader in a circle.
            remedy = (
                "The members do not land on one grid: resample them onto a common grid before the"
                " Reduce, or declare grid: reference:<case> / shape_only if the cohort is already"
                " aligned."
                if reduction is not None and reduction.check_grid() is not None
                else "A reduction has no whole-volume path to fall back to: folding every case in"
                " memory is what it exists to avoid, so this refuses the run whatever on_fallback"
                " says. Fix the refusing stage, or put a Save before the Reduce."
            )
            raise TransformerError(
                f"The reduction into '{first.case}' ({first.group_src} -> {first.group_dest}) cannot"
                f" stream: {first.reason or 'see the plan'}",
                remedy,
            )
        if self.on_fallback == "warn" and plan.fallback_entries:
            print(
                f"[KonfAI] WARNING: {len(plan.fallback_entries)} case(s) take the whole-volume"
                " path (which stage refused, and why, is in the plan this run wrote to its log). They"
                " fit the budget; set on_fallback: error to refuse them."
            )

    def _shard_work(self, world_size: int) -> None:
        """Split the run into per-rank shards of work items.

        Work items, not cases: an ordinary chain has one per case, a reduction exactly one for all
        of them. Each item still writes its own entry, so the disjoint-writes guard of
        :meth:`_guard_sharded_destinations` holds."""
        items = self._work_items()
        weights = [item.weight_bytes for item in items]
        # Balanced by bytes, not by count: one large case among many small ones would otherwise
        # hold a rank alone while the others finish, and a chain's cases would all land on the
        # first ranks. Heaviest first onto the least-loaded rank (LPT); shards keep the run order.
        ranks = max(1, world_size)
        loads = [0] * ranks
        shards: list[list[int]] = [[] for _ in range(ranks)]
        for item in sorted(range(len(items)), key=lambda index: -weights[index]):
            rank = min(range(ranks), key=loads.__getitem__)
            shards[rank].append(item)
            loads[rank] += weights[item]
        self._shards = [sorted(shard) for shard in shards]

    @property
    def world_size(self) -> int:
        return len(self._shards)

    def rank_dataloaders(self, global_rank: int) -> list:
        return []  # a rank walks its shard of work items; there is no loader

    def run_process(self, world_size: int, global_rank: int, local_rank: int, dataloaders):
        """Materialize this rank's cases. The plan already said what will happen: the console gets
        the deviations from it, the live counter, and one final line that says how it went."""
        del dataloaders
        # The one caller that knows its device: cuda:<rank> when the launch requested GPUs (the
        # runtime narrowed CUDA_VISIBLE_DEVICES to them), CPU otherwise. The chain then runs where
        # the rank runs; regions still land on the host for every write.
        device = get_device(local_rank)
        chain_device = torch.device(f"cuda:{device}") if isinstance(device, int) else device
        started = time.monotonic()
        items = self._work_items()
        shard = self._shards[global_rank]
        counts: Counter[Verdict] = Counter()
        # 'error' holds at run time too: a fallback the plan could not see (a sweep that fails, a
        # field bound exceeded) raises at that case instead of quietly costing a volume.
        allow_fallback = self.on_fallback != "error"

        def description() -> str:
            return (
                f"Transform : {counts[Verdict.STREAM]} streamed | {counts[Verdict.LOAD]} loaded"
                f" | {counts[Verdict.REDUCE]} reduced"
                f" | {counts[Verdict.WHOLE_VOLUME]} whole-volume | {counts[Verdict.SKIP]} skipped"
            )

        failed: list[tuple[str, str, str]] = []
        with tqdm.tqdm(total=len(shard), desc=description(), ncols=0) as progress:
            for item in (items[position] for position in shard):
                try:
                    counts.update(self._run_item(item, chain_device, allow_fallback, progress))
                except Exception as error:  # one case's failure is not the shard's: keep going, list it
                    failed.append((item.group_dest, item.label, f"{type(error).__name__}: {error}"))
                    progress.write(
                        f"[KonfAI] case '{item.label}' ({item.group_dest}) FAILED: {type(error).__name__}: {error}"
                    )
                progress.set_description(description())
                progress.update(1)
        # No collective: each rank reports its own shard (one line for the usual single rank).
        written = sum(counts.values()) - counts[Verdict.SKIP]
        resume = f", {counts[Verdict.SKIP]} already written (--overwrite recomputes)" if counts[Verdict.SKIP] else ""
        who = f"rank {global_rank}/{world_size} " if world_size > 1 else ""
        print(
            f"[KonfAI] {who}done in {time.monotonic() - started:.1f} s: {written} written"
            f" ({counts[Verdict.STREAM]} streamed, {counts[Verdict.LOAD]} loaded,"
            f" {counts[Verdict.WHOLE_VOLUME]} whole-volume, {counts[Verdict.REDUCE]} reduced){resume}"
            + (f", {len(failed)} FAILED" if failed else "")
            + f" -> outputs in {self.transform_path / 'outputs.json'}"
        )
        if failed:
            listed = "\n".join(f"  {group_dest}: '{what}': {reason}" for group_dest, what, reason in failed)
            raise TransformerError(
                f"{len(failed)} of {len(shard)} work item(s) failed on {who or 'this rank '}:\n{listed}",
                "The other items were written; a rerun resumes at the failed ones (their outputs do not exist).",
            )

    def _run_item(
        self, item: WorkItem, chain_device: torch.device, allow_fallback: bool, progress: tqdm.tqdm
    ) -> Counter[Verdict]:
        """One work item: a reduction, an Expand case with all its copies, or a case; returns what
        its entries took."""
        counts: Counter[Verdict] = Counter()
        if item.kind == "reduction":
            reduction = item.reduction
            assert reduction is not None  # nosec B101 - the item was built from the same predicate
            if item.pending(reduction.reduce.output, self._overwrite):
                reduction.materialize(rewrite=self._overwrite, device=chain_device)
                counts[Verdict.REDUCE] += 1
            else:
                counts[Verdict.SKIP] += 1
            return counts
        manager = item.manager
        if item.kind == "expansion":
            # One item per case, all its copies inside: the engine shares one read pass
            # across the copies whose draws allow it, which a per-copy loop could not.
            copies = list(range(1, item.copies + 1))
            todo = [a for a in copies if item.pending(manager.copy_entry(a), self._overwrite)]
            counts[Verdict.SKIP] += len(copies) - len(todo)
            outcomes = item.engine.materialize_copies(
                todo,
                rewrite=self._overwrite,
                fallback_budget_bytes=self._budget_bytes,
                allow_fallback=allow_fallback,
                device=chain_device,
            )
            counts.update(verdict for verdict, _regime in outcomes.values())
            if counts[Verdict.WHOLE_VOLUME]:
                progress.write(
                    f"[KonfAI] case '{manager.name}' ({item.group_dest}):"
                    f" {counts[Verdict.WHOLE_VOLUME]} of {len(copies)} cop(ies) took the whole-volume path"
                )
            return counts
        if not item.pending(manager.name, self._overwrite):
            counts[Verdict.SKIP] += 1
            return counts
        planned = self._planned.get((item.group_dest, manager.name))
        verdict = item.engine.materialize(
            0,
            rewrite=self._overwrite,
            fallback_budget_bytes=self._budget_bytes,
            allow_fallback=allow_fallback,
            prefer_whole=planned is Verdict.LOAD,
            device=chain_device,
        )
        counts[verdict] += 1
        if planned not in (None, verdict):
            # The one thing worth a line: the run did NOT do what the plan said.
            progress.write(
                f"[KonfAI] case '{manager.name}' ({item.group_dest}): {verdict}"
                f" (planned {planned}: {manager.stream_refusal(0) or 'see the log'})"
            )
        return counts


def build_transform(
    transform_file: Path | str | dict = Path("./Transform.yml").resolve(),
    transforms_dir: Path | str = Path("./Transforms").resolve(),
) -> DistributedObject:
    """Build the configured transform workflow without executing it: ``compute_plan()`` is the
    dry run, ``setup()`` prints and enforces the plan. ``transform_file`` may be the config tree as
    a dict; it is materialized to a file here, so the strict read sees what every other reader will.

    The read is strict: a key nothing binds (a typo'd ``memory_budge:``, a ``Clip: {min_val: 0}``)
    is refused with its path instead of being carried along with its default used in its place.
    Everything the workflow reads from the file is bound inside ``__init__`` (the chains, their
    draws, a ``Reduce``'s operator), which is what lets the check close when it returns.
    """
    if isinstance(transform_file, dict):
        transform_file = _materialized_config(transform_file, "Transformer")
    configure_workflow_environment(
        config_path=transform_file,
        root="Transformer",
        state=State.TRANSFORM,
        path_env={"KONFAI_TRANSFORMS_DIRECTORY": transforms_dir},
    )
    os.environ["KONFAI_CONFIG_MODE"] = "Done"
    with strict_config("Transformer"):
        return apply_config()(Transformer)()


def plan_transform(
    overwrite: bool = False,
    gpu: list[int] | None = None,
    cpu: int | None = 1,
    quiet: bool = False,
    transform_file: Path | str | dict = Path("./Transform.yml").resolve(),
    transforms_dir: Path | str = Path("./Transforms").resolve(),
) -> TransformPlan:
    """CLI ``--plan``: build, plan, print, and stop. Same flags and world size as :func:`transform`,
    so the plan shards the way the run will; the plan is the requested output, printed whatever
    ``quiet`` says; nothing is written under ``transforms_dir``. The write probe opens then removes
    one entry per destination and takes back a store it created, so plan mode leaves no output
    behind.
    """
    del quiet
    workflow = build_transform(transform_file=transform_file, transforms_dir=transforms_dir)
    world_size = len(gpu or []) or max(1, int(cpu or 1))
    plan = cast(Transformer, workflow).compute_plan(world_size, bool(overwrite))
    print(plan.report())
    return plan


@run_distributed_app
def transform(
    overwrite: bool = False,
    gpu: list[int] | None = None,
    cpu: int = 1,
    quiet: bool = False,
    transform_file: Path | str | dict = Path("./Transform.yml").resolve(),
    transforms_dir: Path | str = Path("./Transforms").resolve(),
) -> DistributedObject:
    """Build and execute the configured transform workflow.

    ``transform_file`` accepts the config tree as a dict: the pure-Python spelling of the same
    run; the resolved YAML still lands in the workspace as the run's record.
    """
    del overwrite, gpu, cpu, quiet
    return build_transform(transform_file=transform_file, transforms_dir=transforms_dir)
