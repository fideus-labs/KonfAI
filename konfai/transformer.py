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

"""The TRANSFORM workflow: apply a declared transform chain to a dataset. No model.

The engine is :meth:`DatasetManager.materialize` — the streamed Save sweep with its whole-volume
fallback — and the product is the plan: before a byte is written, every (case, chain) is planned on
the launcher, each output destination is probed with a real region-write open (created then
removed, so the printed verdict is the run's own), and the whole thing is printed and kept on disk.
Nothing here falls back silently: a chain that cannot stream says which stage refused and why, and
``on_fallback`` decides whether that is information, a warning, or an error.
"""

import contextlib
import inspect
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np
import tqdm
from ruamel.yaml import YAML

from konfai import config_file, transforms_directory
from konfai.data.case_reduction import CaseReduction, split_chain
from konfai.data.data_manager import DataTransform, _format_gib, _node_local_ranks
from konfai.data.patching import _CASE_ELEMENT_BYTES, _FALLBACK_INFLIGHT_FACTOR, DatasetManager, _save_destination
from konfai.data.transform import Reduce, Save, split_expand
from konfai.utils.config import apply_config, config
from konfai.utils.dataset import Attribute, Dataset
from konfai.utils.errors import ConfigError, TransformerError
from konfai.utils.runtime import (
    DistributedObject,
    State,
    configure_workflow_environment,
    run_distributed_app,
)
from konfai.utils.utils import get_module

_PROBE_ENTRY = "__konfai_plan_probe__"


@dataclass(frozen=True)
class TransformPlanEntry:
    """One (case, chain) line of the plan: the verdict the run itself will act on."""

    case: str  # the case name, the copy's entry name behind an Expand, or a reduction's output name
    group_src: str
    group_dest: str
    verdict: str  # "STREAM" | "WHOLE-VOLUME" | "SKIP" | "REDUCE" | "REFUSED"
    reason: str | None
    case_bytes: int
    #: The cases folded into one, for a reduction. Empty for an ordinary per-case entry. Printed in
    #: full: a reduction whose case list silently changed writes a different volume under the same
    #: name, and nothing about the output would look wrong.
    reduced: tuple[str, ...] = ()
    #: The case this entry is an Expand copy of ("" for anything that is not a copy).
    expanded_from: str = ""
    #: How a STREAM copy is read: "shared" rides the case's single read pass, "solo" sweeps its own
    #: (its draw's read geometry is its own). None for non-copy entries and non-STREAM verdicts.
    regime: str | None = None

    @property
    def working_set_bytes(self) -> int:
        """What this entry holds at its peak — the figure the budget bounds.

        A reduction's ``case_bytes`` is already its resident regions (it holds one region per case,
        not one volume), where a whole-volume fallback holds the case plus one in-flight copy.
        """
        return self.case_bytes if self.reduced else self.case_bytes * _FALLBACK_INFLIGHT_FACTOR


@dataclass
class TransformPlan:
    """The plan as an object — what ``--plan`` prints and ``setup`` enforces."""

    entries: list[TransformPlanEntry]
    budget_bytes: float
    budget_desc: str
    world_size: int
    dropped_cases: dict[str, int]
    dtype_hypothesis: str

    @property
    def fallback_entries(self) -> list[TransformPlanEntry]:
        return [entry for entry in self.entries if entry.verdict == "WHOLE-VOLUME"]

    @property
    def refused_entries(self) -> list[TransformPlanEntry]:
        """Reductions that cannot stream. Unlike a per-case chain they have no whole-volume path to
        fall back to, so these refuse the run whatever ``on_fallback`` says."""
        return [entry for entry in self.entries if entry.verdict == "REFUSED"]

    def budget_violations(self) -> list[TransformPlanEntry]:
        """Entries whose working set exceeds the per-rank budget — the hard constraint.

        Two kinds qualify, and for the same reason: they hold something the budget cannot shrink.
        A whole-volume fallback holds the case (times the in-flight factor). A reduction holds one
        region per case, and once its region is down to a single row there is nothing left to fold —
        so unlike a streamed case, whose slab height simply keeps shrinking, it has to refuse. The
        estimate is from headers alone and the margin is printed, never hidden: an entry within a
        few percent of the budget may still exceed it.
        """
        candidates = [entry for entry in self.entries if entry.verdict in ("WHOLE-VOLUME", "REDUCE")]
        return [entry for entry in candidates if entry.working_set_bytes > self.budget_bytes]

    def report(self) -> str:
        lines = [
            f"[Transformer] plan over {self.world_size} rank(s) | per-rank budget"
            f" {_format_gib(self.budget_bytes)} ({self.budget_desc}) | fallback working set = case"
            f" x {_CASE_ELEMENT_BYTES} B x {_FALLBACK_INFLIGHT_FACTOR} (in-flight copy), headers-only"
            f" estimate | output dtype/channels assumed {self.dtype_hypothesis} until the first slab"
        ]
        for group_src, dropped in sorted(self.dropped_cases.items()):
            if dropped:
                lines.append(
                    f"[Transformer] {dropped} case(s) present in '{group_src}' only are DROPPED:"
                    " several groups_src keep the intersection of their case names."
                )
        by_chain: dict[tuple[str, str], list[TransformPlanEntry]] = {}
        for entry in self.entries:
            by_chain.setdefault((entry.group_src, entry.group_dest), []).append(entry)
        for (group_src, group_dest), entries in by_chain.items():
            reductions = [entry for entry in entries if entry.reduced]
            if reductions:
                for entry in reductions:
                    lines.append(
                        f"  {group_src} -> {group_dest}: REDUCE {len(entry.reduced)} case(s) -> 1"
                        f" output '{entry.case}' -- {entry.verdict}"
                    )
                    if entry.reason:
                        lines.append(f"    {entry.reason}")
                    if entry.verdict == "REDUCE":
                        lines.append(
                            f"    peak ~= {_format_gib(entry.working_set_bytes)} vs per-rank budget"
                            f" {_format_gib(self.budget_bytes)}"
                        )
                    lines.append(f"    cases: {', '.join(entry.reduced)}")
                continue
            counts = dict.fromkeys(("STREAM", "WHOLE-VOLUME", "SKIP"), 0)
            for entry in entries:
                counts[entry.verdict] += 1
            expanded = [entry for entry in entries if entry.expanded_from]
            if expanded:
                cases = len({entry.expanded_from for entry in expanded})
                shared = sum(1 for entry in expanded if entry.regime == "shared")
                solo = sum(1 for entry in expanded if entry.regime == "solo")
                lines.append(
                    f"  {group_src} -> {group_dest}: EXPAND {cases} case(s) ->"
                    f" {len(expanded)} cop(ies) -- {shared} STREAM (shared read pass),"
                    f" {solo} STREAM (own pass), {counts['WHOLE-VOLUME']} WHOLE-VOLUME,"
                    f" {counts['SKIP']} SKIP (copy already written)"
                )
            else:
                lines.append(
                    f"  {group_src} -> {group_dest}: {len(entries)} case(s) --"
                    f" {counts['STREAM']} STREAM, {counts['WHOLE-VOLUME']} WHOLE-VOLUME,"
                    f" {counts['SKIP']} SKIP (output already written)"
                )
            reasons: dict[str, int] = {}
            for entry in entries:
                if entry.reason and (entry.verdict == "WHOLE-VOLUME" or entry.regime == "solo"):
                    label = "WHOLE-VOLUME" if entry.verdict == "WHOLE-VOLUME" else "own pass"
                    reasons[f"{label}: {entry.reason}"] = reasons.get(f"{label}: {entry.reason}", 0) + 1
            for reason, count in reasons.items():
                lines.append(f"    ({count} cop(ies)) {reason}" if expanded else f"    ({count} case(s)) {reason}")
            if counts["WHOLE-VOLUME"]:
                worst = max(entry.working_set_bytes for entry in entries if entry.verdict == "WHOLE-VOLUME")
                lines.append(
                    f"    worst fallback case ~= {_format_gib(worst)}"
                    f" vs per-rank budget {_format_gib(self.budget_bytes)}"
                )
        return "\n".join(lines)


@config()
class Transformer(DistributedObject):
    """Model-less workflow that materializes every chain's ``Write`` outputs, plan first."""

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
        self._shards: list[list[int]] = []
        self._case_names: list[str] = []
        self._reductions: dict[str, CaseReduction | None] = {}

    def _managers(self) -> dict[str, list[DatasetManager]]:
        prepared = self.dataset._prepared_data
        if prepared is None:
            raise TransformerError("The dataset was not prepared before planning.")
        return prepared

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
        return _save_destination(terminal, manager.dataset, manager.group_dest)

    def _reduction(self, group_dest: str, managers: list[DatasetManager]) -> CaseReduction | None:
        """The reduction this chain declares, or ``None`` when it is an ordinary per-case chain.

        The cases are re-managed with the PRE-reduction stages only: the prepared managers carry the
        whole chain, ``Reduce`` included, and a manager holding it would refuse to stream — rightly,
        since as a per-case stage it cannot run at all.

        Built once per chain and kept, because planning, sharding and running all ask for it and
        must get the SAME object: the run has to execute the reduction the plan measured, and
        rebuilding means re-managing every case and re-planning every chain.
        """
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
        destination, group = _save_destination(terminal, managers[0].dataset, group_dest)
        reduction = CaseReduction(
            managers=cases,
            reduce=reduce,
            post=list(post[:-1]),
            destination=destination,
            group=group,
        )
        reduction.fit_budget(self._budget_bytes)
        return reduction

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
        self, manager: DatasetManager, probed: set[tuple[str, str]], a: int = 0
    ) -> str | None:
        """Open a real region-write stream on each Save/Write destination, then remove it.

        This is what makes the plan the run's own verdict: ``can_stream_data`` is a capability
        check, but the refusals that matter (rank, dtype, geometry) live in ``open_data_stream``,
        which the engine only reaches at the first computed slab. The probe pays one entry creation
        per destination — removed immediately — so ``--plan`` touches the output directories.
        """
        channels = int(manager.base_shape[0])
        dtype = self._dtype_hypothesis(manager)
        for transform, spatial, attributes in manager.write_targets(a):
            destination, group = _save_destination(transform, manager.dataset, manager.group_dest)
            key = (str(destination.filename), group)
            if key in probed:
                continue
            probed.add(key)
            failure = self._probe_destination(destination, group, [channels, *spatial], dtype, attributes)
            if failure is not None:
                return failure
        return None

    @staticmethod
    def _probe_destination(
        destination: Dataset, group: str, shape: list[int], dtype: np.dtype, attributes: Attribute
    ) -> str | None:
        """One real region-write open, removed immediately — the probe both chains and reductions
        share, so the plan's verdict is the run's own on every kind of output."""
        try:
            stream = destination.open_data_stream(group, _PROBE_ENTRY, shape, dtype, attributes)
        except Exception as error:  # the probe exists to surface exactly these
            Transformer._remove_probe_entry(destination)
            return (
                f"destination '{destination.filename}' refuses a"
                f" [{', '.join(str(extent) for extent in shape)}] {dtype} region write:"
                f" {type(error).__name__}: {error}"
            )
        if stream is None:
            Transformer._remove_probe_entry(destination)
            return (
                f"destination '{destination.filename}' cannot serve region writes"
                " (h5 and omezarr always can; mha only with image geometry)."
            )
        stream.__enter__()
        stream.abort(RuntimeError("plan probe"))
        Transformer._remove_probe_entry(destination)
        return None

    @staticmethod
    def _remove_probe_entry(destination: Dataset) -> None:
        """Take the probe's case directory back out of a directory store.

        Aborting the stream removes the ENTRY, but a directory dataset gives every case a directory
        of its own (``<root>/<case>/<group>.<ext>``) and that one belongs to the store, not the
        stream. It has to go too: it is shaped exactly like a case, ``get_names`` lists it as one,
        and a dry run must leave the output as it found it. ``rmdir``, never ``rmtree`` -- anything
        actually in there is not the probe's, and then the right move is to leave it alone.
        """
        if not destination.is_directory:
            return
        with contextlib.suppress(OSError):
            (Path(destination.filename) / _PROBE_ENTRY).rmdir()

    def output_destinations(self) -> list[dict[str, str]]:
        """Every chain's terminal ``Write``, as ``{group_src, group_dest, dataset, group, format}``.

        What the run produced, said in the run's own terms. The deliverable never lives under the
        run directory, so this is how a reader -- a person, Studio's run panel, the next workflow --
        finds it without parsing the plan or re-reading the config.
        """
        destinations: list[dict[str, str]] = []
        for group_dest, managers in self._managers().items():
            if not managers:
                continue
            destination, group = self._terminal_destination(managers[0])
            destinations.append(
                {
                    "group_src": self._group_src_of(group_dest),
                    "group_dest": group_dest,
                    "dataset": str(Path(destination.filename).resolve()),
                    "group": group,
                    "format": destination.file_format,
                }
            )
        return destinations

    def compute_plan(self, world_size: int = 1, overwrite: bool = False) -> TransformPlan:
        """Plan every (case, chain) on the launcher, headers plus one write probe per destination."""
        budget = self.dataset.resolved_budget()
        per_rank_budget = budget.per_rank_bytes(_node_local_ranks(world_size))
        # Resolved before anything is planned, because a reduction sizes its regions against it --
        # the plan must measure the same run setup() will enforce.
        self._budget_bytes = per_rank_budget
        entries: list[TransformPlanEntry] = []
        probed: set[tuple[str, str]] = set()
        planned_dtypes: set[str] = set()
        for group_dest, managers in self._managers().items():
            group_src = self._group_src_of(group_dest)
            reduction = self._reduction(group_dest, managers)
            if reduction is not None:
                reduction_dtype = np.dtype("float32")
                planned_dtypes.add(str(reduction_dtype))
                reduction_plan = reduction.plan()
                skipped = not overwrite and reduction.destination.is_dataset_exist(
                    reduction.group, reduction.reduce.output
                )
                verdict = "SKIP" if skipped else ("REDUCE" if reduction_plan.streams else "REFUSED")
                reason = (
                    reduction_plan.refusal
                    if not reduction_plan.streams
                    else reduction_plan.describe().split("\n", 1)[1].strip()
                )
                if verdict == "REDUCE":
                    # A reduction has no whole-volume fallback, so a destination that would refuse
                    # its stream at the first region refuses the plan -- probed here, like a chain's.
                    probe_failure = self._probe_destination(
                        reduction.destination,
                        reduction.group,
                        [reduction_plan.channels, *reduction_plan.spatial],
                        reduction_dtype,
                        reduction.reference.landed_attributes(),
                    )
                    if probe_failure is not None:
                        verdict, reason = "REFUSED", probe_failure
                entries.append(
                    TransformPlanEntry(
                        case=reduction.reduce.output,
                        group_src=group_src,
                        group_dest=group_dest,
                        verdict=verdict,
                        reason=reason,
                        case_bytes=reduction_plan.peak_bytes,
                        reduced=tuple(reduction_plan.cases),
                    )
                )
                continue
            if managers:
                # Every manager of a group shares one chain object, so one of them answers for all.
                planned_dtypes.add(str(self._dtype_hypothesis(managers[0])))
            expansion = split_expand(managers[0].transforms)[1] if managers else None
            for manager in managers:
                case_bytes = int(np.prod(manager.base_shape, dtype=np.int64)) * _CASE_ELEMENT_BYTES
                destination, group = self._terminal_destination(manager)
                if expansion is not None:
                    # One line per copy: the copies are the outputs, each with its own resume, its
                    # own verdict, and -- for STREAM -- the regime that says who pays the reads.
                    for a in range(1, expansion.nb + 1):
                        entry_name = manager.copy_entry(a)
                        regime: str | None = None
                        if not overwrite and destination.is_dataset_exist(group, entry_name):
                            verdict, reason = "SKIP", None
                        elif manager.can_stream_patch(a, apply_augmentations=True):
                            probe_failure = self._probe_write_destinations(manager, probed, a)
                            if probe_failure is None:
                                verdict, reason = "STREAM", manager.expansion_solo_reason(a)
                                regime = "solo" if reason else "shared"
                            else:
                                verdict, reason = "WHOLE-VOLUME", probe_failure
                        else:
                            verdict = "WHOLE-VOLUME"
                            reason = manager.stream_refusal(a, apply_augmentations=True)
                        entries.append(
                            TransformPlanEntry(
                                entry_name,
                                group_src,
                                group_dest,
                                verdict,
                                reason,
                                case_bytes,
                                expanded_from=manager.name,
                                regime=regime,
                            )
                        )
                    continue
                if not overwrite and destination.is_dataset_exist(group, manager.name):
                    verdict, reason = "SKIP", None
                elif manager.can_stream_patch(0, apply_augmentations=False):
                    probe_failure = self._probe_write_destinations(manager, probed)
                    if probe_failure is None:
                        verdict, reason = "STREAM", None
                    else:
                        # The run would fail the sweep at its first slab and fall back: say so now.
                        verdict, reason = "WHOLE-VOLUME", probe_failure
                else:
                    verdict = "WHOLE-VOLUME"
                    reason = manager.stream_refusal(0, apply_augmentations=False)
                entries.append(TransformPlanEntry(manager.name, group_src, group_dest, verdict, reason, case_bytes))
        dropped: dict[str, int] = {}
        kept = set(self.dataset._prepared_train_names)
        for group_src in self.dataset.groups_src:
            available: set[str] = set()
            for dataset in self.dataset.datasets.values():
                if dataset.is_group_exist(group_src):
                    available.update(dataset.get_names(group_src))
            dropped[group_src] = len(available - kept)
        dtype_hypothesis = f"{'/'.join(sorted(planned_dtypes)) or 'float32'} / source channels"
        return TransformPlan(entries, per_rank_budget, budget.description, world_size, dropped, dtype_hypothesis)

    def setup(self, world_size: int):
        """Plan, print, enforce, shard — before any spawn, before any byte."""
        # No overwrite prompt on the run folder: it holds logs, the plan and a config copy, all
        # rewritten in place -- and prompting here would break the default per-case resume (a second
        # run would refuse because the first one left its config copy behind).
        os.makedirs(self.transform_path, exist_ok=True)
        shutil.copyfile(config_file(), self.transform_path / config_file().name)

        self._overwrite = os.environ.get("KONFAI_OVERWRITE", "False") == "True"
        plan = self.compute_plan(world_size, self._overwrite)
        report = plan.report()
        print(report)
        # The plan is an artifact, not a log line: Studio filters routine startup lines from the
        # console stream, and a plan that only ever existed as one is a plan nobody can re-read.
        (self.transform_path / "plan.txt").write_text(report + "\n")
        # Where the data went, machine-readable beside the human-readable plan. This run directory
        # holds a log, a plan and a config copy -- never the deliverable, which lands wherever each
        # Write pointed. Without this, the one thing a reader wants after the run is the one thing
        # nothing in the run directory names.
        (self.transform_path / "outputs.json").write_text(json.dumps(self.output_destinations(), indent=2) + "\n")

        if world_size > 1:
            for managers in self._managers().values():
                for transform in managers[0].transforms if managers else []:
                    if not isinstance(transform, Save):
                        continue
                    destination, _group = _save_destination(transform, managers[0].dataset, managers[0].group_dest)
                    # The question here is NOT concurrent_write_safe(): that one asks whether two
                    # entries of one shared store may be written at once, and answers no for omezarr
                    # -- which would refuse the very destination this workflow recommends. Ranks
                    # shard by CASE, and a directory dataset gives each case its own file or store
                    # (<root>/<case>/<group>.<ext>), so their writes are disjoint by construction.
                    # Only a single-file store (h5) puts every case in one handle.
                    if not destination.is_directory:
                        raise TransformerError(
                            f"--cpu {world_size}: destination '{destination.filename}' is a"
                            " single-file store, and every rank would write into the same file.",
                            "Use one process, or a directory destination (omezarr, mha, nii.gz).",
                        )

        inference_chains = sorted(
            group_dest
            for group_dest, managers in self._managers().items()
            if managers and any(type(t).__name__ == "KonfAIInference" for t in managers[0].transforms)
        )
        if inference_chains:
            print(
                f"[Transformer] NOTE: chain(s) {', '.join(inference_chains)} run a NESTED KonfAI"
                " inference: its GPU and RAM usage live outside the declared memory_budget, and the"
                " plan cannot bound them."
            )

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
                f" ({_format_gib(plan.budget_bytes)}): worst is"
                f" ~{_format_gib(worst.working_set_bytes)} -- {what}.",
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
            reduction = self._reduction(first.group_dest, self._managers().get(first.group_dest, []))
            # A grid disagreement gets its own remedy: a Save changes nothing about the grids, so
            # the generic advice would send the reader in a circle.
            remedy = (
                "The members do not land on one grid: resample them onto a common grid before the"
                " Reduce, or declare grid: reference:<case> / shape_only if the cohort is already"
                " aligned."
                if reduction is not None and reduction.check_grid() is not None
                else "A reduction has no whole-volume path to fall back to -- folding every case in"
                " memory is what it exists to avoid -- so this refuses the run whatever on_fallback"
                " says. Fix the refusing stage, or put a Save before the Reduce."
            )
            raise TransformerError(
                f"The reduction into '{first.case}' ({first.group_src} -> {first.group_dest}) cannot"
                f" stream: {first.reason or 'see the plan'}",
                remedy,
            )
        if self.on_fallback == "warn" and plan.fallback_entries:
            print(
                f"[Transformer] WARNING: {len(plan.fallback_entries)} case(s) take the whole-volume"
                " path (reasons above). They fit the budget; set on_fallback: error to refuse them."
            )

        self._budget_bytes = plan.budget_bytes
        self._case_names = list(self.dataset._prepared_train_names)
        # Work items, not cases: an ordinary chain has one per case, a reduction exactly one for all
        # of them. Each item still writes its own entry, so the disjoint-writes guard above holds.
        items: list[tuple[str, int]] = []
        for group_dest, managers in self._managers().items():
            if self._reduction(group_dest, managers) is not None:
                items.append((group_dest, -1))
            else:
                items.extend((group_dest, index) for index in range(len(managers)))
        shards = np.array_split(np.arange(len(items)), max(1, world_size))
        self._items = items
        self._shards = [[int(index) for index in shard] for shard in shards]
        self.dataloader = [[] for _ in range(max(1, world_size))]

    def run_process(self, world_size: int, global_rank: int, local_rank: int, dataloaders):
        """Materialize this rank's cases, one at a time, and say which path wrote each."""
        del world_size, local_rank, dataloaders
        managers = self._managers()
        shard = self._shards[global_rank]
        counts = {"STREAM": 0, "WHOLE-VOLUME": 0, "SKIP": 0, "REDUCE": 0}
        # 'error' holds at run time too: a fallback the plan could not see (a sweep that fails, a
        # Warp bound exceeded) raises at that case instead of quietly costing a volume.
        allow_fallback = self.on_fallback != "error"

        def description() -> str:
            return (
                f"Transform : {counts['STREAM']} streamed | {counts['REDUCE']} reduced"
                f" | {counts['WHOLE-VOLUME']} whole-volume | {counts['SKIP']} skipped"
            )

        with tqdm.tqdm(total=len(shard), desc=description(), ncols=0) as progress:
            for item in shard:
                group_dest, index = self._items[item]
                group_managers = managers[group_dest]
                if index < 0:
                    reduction = self._reduction(group_dest, group_managers)
                    assert reduction is not None  # nosec B101 - the item was built from the same predicate
                    output = reduction.reduce.output
                    already = reduction.destination.is_dataset_exist(reduction.group, output)
                    if not self._overwrite and already:
                        counts["SKIP"] += 1
                        progress.write(f"[Transformer] '{output}' ({group_dest}): SKIP (already written)")
                    else:
                        reduction.materialize(rewrite=self._overwrite)
                        counts["REDUCE"] += 1
                        progress.write(
                            f"[Transformer] '{output}' ({group_dest}): REDUCE {len(reduction.managers)} case(s)"
                        )
                else:
                    manager = group_managers[index]
                    destination, group = self._terminal_destination(manager)
                    expansion = split_expand(manager.transforms)[1]
                    if expansion is not None:
                        # One item per case, all its copies inside: the engine shares one read pass
                        # across the copies whose draws allow it, which a per-copy loop could not.
                        copies = list(range(1, expansion.nb + 1))
                        todo = [
                            a
                            for a in copies
                            if self._overwrite or not destination.is_dataset_exist(group, manager.copy_entry(a))
                        ]
                        counts["SKIP"] += len(copies) - len(todo)
                        regimes = manager.materialize_copies(
                            todo,
                            rewrite=self._overwrite,
                            fallback_budget_bytes=self._budget_bytes,
                            allow_fallback=allow_fallback,
                        )
                        shared = sum(1 for regime in regimes.values() if regime == "stream-shared")
                        own = sum(1 for regime in regimes.values() if regime == "stream")
                        whole = sum(1 for regime in regimes.values() if regime == "whole-volume")
                        counts["STREAM"] += shared + own
                        counts["WHOLE-VOLUME"] += whole
                        progress.write(
                            f"[Transformer] case '{manager.name}' ({group_dest}): EXPAND"
                            f" {len(copies)} cop(ies) -- {shared} shared pass, {own} own pass,"
                            f" {whole} whole-volume, {len(copies) - len(todo)} skipped"
                        )
                    elif not self._overwrite and destination.is_dataset_exist(group, manager.name):
                        counts["SKIP"] += 1
                        progress.write(f"[Transformer] case '{manager.name}' ({group_dest}): SKIP (already written)")
                    else:
                        streamed = manager.materialize(
                            0,
                            rewrite=self._overwrite,
                            fallback_budget_bytes=self._budget_bytes,
                            allow_fallback=allow_fallback,
                        )
                        verdict = "STREAM" if streamed else "WHOLE-VOLUME"
                        counts[verdict] += 1
                        progress.write(f"[Transformer] case '{manager.name}' ({group_dest}): {verdict}")
                progress.set_description(description())
                progress.update(1)


#: The grammar the strict mode accepts, level by level. ``None`` marks free-form levels (group
#: names); ``"chain"`` marks a transforms mapping, whose keys are classpaths and whose arguments
#: are checked against each stage's own signature.
_STRICT_GRAMMAR: dict[str, object] = {
    "name": None,
    "on_fallback": None,
    "manual_seed": None,
    "Dataset": {
        "dataset_filenames": None,
        "memory_budget": None,
        "subset": None,
        "groups_src": {"*": {"groups_dest": {"*": {"transforms": "chain"}}}},
    },
}


def _stage_class(classpath: str, default_classpath: str = "konfai.data.transform") -> type | None:
    """The class a chain stage's key names, resolved exactly as ``TransformLoader`` will resolve it.

    ``None`` when it resolves nowhere: the loader raises its own error for that case, and refusing
    the arguments of a class that does not exist would report the wrong problem.
    """
    try:
        module, name = get_module(classpath, default_classpath)
        if not hasattr(module, name) and ":" not in classpath and default_classpath == "konfai.data.transform":
            module, name = get_module(classpath, "konfai.data.augmentation")
        candidate = getattr(module, name, None)
    except Exception:  # an unresolvable stage is the loader's error to raise, with its own message
        return None
    return candidate if isinstance(candidate, type) else None


def _stage_arguments(classpath: str, mapping: dict) -> set[str] | None:
    """The argument names one stage accepts, or ``None`` when they cannot be known from here.

    A ``Reduce`` also accepts its operator's own parameters -- ``resolve_operator`` binds them from
    the same mapping -- so its allowed set is the union of the two signatures.
    """
    stage = _stage_class(str(classpath))
    if stage is None:
        return None
    classes = [stage]
    if issubclass(stage, Reduce):
        operator = _stage_class(str(mapping.get("operator", "Median")), "konfai.data.reduction")
        if operator is None:
            return None
        classes.append(operator)
    allowed: set[str] = set()
    for cls in classes:
        if all("__init__" not in vars(base) for base in cls.__mro__ if base is not object):
            # No __init__ anywhere in the MRO: the class takes no arguments at all, and object's
            # own (*args, **kwargs) signature must not read as "accepts anything".
            continue
        try:
            parameters = dict(inspect.signature(cls).parameters)
        except (TypeError, ValueError):
            return None
        if any(p.kind in (p.VAR_KEYWORD, p.VAR_POSITIONAL) for p in parameters.values()):
            return None
        allowed |= set(parameters)
    return allowed


def _reject_unknown_stage_arguments(chain: object, path: str) -> None:
    """A typo'd stage argument is refused like a typo'd structural key.

    The binder reads only the parameters a stage's signature names and materializes the default
    beside anything else -- ``Clip: {min_val: 0}`` clips at -1024 and exits 0. A stage that resolves
    nowhere, takes ``**kwargs`` or hides its signature is left to the loader's own error.
    """
    if not isinstance(chain, dict):
        return
    for classpath, arguments in chain.items():
        if not isinstance(arguments, dict):
            continue
        allowed = _stage_arguments(str(classpath), arguments)
        if allowed is None:
            continue
        for argument in arguments:
            if str(argument) not in allowed:
                raise ConfigError(
                    f"Unknown argument '{argument}' for '{classpath}' at '{path}.{classpath}'.",
                    f"'{classpath}' takes: {sorted(allowed)}. A typo'd argument would otherwise be"
                    " ignored and its default silently used in its place.",
                )


def _reject_unknown_keys(config_path: Path) -> None:
    """Strict mode: any key the TRANSFORM grammar does not know is a parse error.

    The binder reads a key when it exists and materializes the default when it does not — a typo'd
    ``memory_budge:`` silently becomes ``memory_budget: auto``. On this workflow the config IS the
    product, so an unknown key is refused with its exact path instead of being carried along. The
    same bar holds inside a chain: a stage's arguments are checked against its own signature.
    """
    if not config_path.exists():
        return
    with open(config_path) as stream:
        tree = YAML().load(stream)
    if not isinstance(tree, dict):
        return
    if "Transformer" not in tree:
        raise ConfigError(
            f"'{config_path}' declares no 'Transformer' root (found: {sorted(str(key) for key in tree)}).",
            "The TRANSFORM workflow reads the 'Transformer:' block; anything else would be ignored"
            " and a full default block appended to the file in its place.",
        )

    def walk(node: object, grammar: dict[str, object], path: str) -> None:
        if not isinstance(node, dict):
            return
        for key, value in node.items():
            child = grammar.get(str(key), grammar.get("*", "unknown"))
            if child == "unknown":
                raise ConfigError(
                    f"Unknown key '{path}.{key}' in the Transformer configuration.",
                    f"Known keys at this level: {sorted(k for k in grammar if k != '*')}. A typo'd"
                    " key would otherwise be ignored and its default silently used.",
                )
            if child == "chain":
                _reject_unknown_stage_arguments(value, f"{path}.{key}")
            elif isinstance(child, dict):
                walk(value, child, f"{path}.{key}")

    walk(tree["Transformer"], _STRICT_GRAMMAR, "Transformer")


def build_transform(
    transform_file: Path | str = Path("./Transform.yml").resolve(),
    transforms_dir: Path | str = Path("./Transforms").resolve(),
) -> DistributedObject:
    """Build and return the configured transform workflow without executing it.

    The returned object carries the plan: ``compute_plan()`` is the programmatic dry-run, and
    ``setup()`` prints and enforces it. This is the in-process surface — ``transform()`` stays
    ``None``-returning like every workflow entrypoint.
    """
    configure_workflow_environment(
        config_path=transform_file,
        root="Transformer",
        state=State.TRANSFORM,
        path_env={"KONFAI_TRANSFORMS_DIRECTORY": transforms_dir},
    )
    os.environ["KONFAI_CONFIG_MODE"] = "Done"
    _reject_unknown_keys(Path(transform_file))
    return apply_config()(Transformer)()


def plan_transform(
    overwrite: bool = False,
    cpu: int = 1,
    transform_file: Path | str = Path("./Transform.yml").resolve(),
    transforms_dir: Path | str = Path("./Transforms").resolve(),
    **_ignored: object,
) -> TransformPlan:
    """CLI ``--plan``: build, plan, print, and stop — the workflow never runs.

    The probe opens then removes one region-write entry per destination, so even plan mode touches
    the output directories: that is the price of a verdict that is the run's own (the printed plan
    is a measurement, not a prediction).
    """
    workflow = build_transform(transform_file=transform_file, transforms_dir=transforms_dir)
    plan = cast(Transformer, workflow).compute_plan(max(1, int(cpu or 1)), bool(overwrite))
    print(plan.report())
    return plan


@run_distributed_app
def transform(
    overwrite: bool = False,
    gpu: list[int] | None = None,
    cpu: int = 1,
    quiet: bool = False,
    transform_file: Path | str = Path("./Transform.yml").resolve(),
    transforms_dir: Path | str = Path("./Transforms").resolve(),
) -> DistributedObject:
    """Build and execute the configured transform workflow."""
    del overwrite, gpu, cpu, quiet
    return build_transform(transform_file=transform_file, transforms_dir=transforms_dir)
