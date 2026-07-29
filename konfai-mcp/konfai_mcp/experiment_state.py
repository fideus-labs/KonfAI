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

"""Where an experiment stands, read back from its workspace.

Nothing here is accumulated. The stage is **derived** from what the runs actually wrote — the configs,
the imported apps, the checkpoints, the predictions, the metric files, the job records — so it cannot
drift from reality, it survives a restart or a crash, and it is the same answer whether the session was
driven from KonfAI Studio, from Claude Code, or by hand.

Three tables, one per question, keyed by the derived stage:

* ``STAGE_FOCUS`` — what this stage owes, one line, for whoever is driving.
* ``STAGE_ACTIONS`` — the tools that carry it forward, best first. Callers filter by readiness.
* ``DIAGNOSES`` — how a failed run's log reads, so a client gets a cause and a correction rather than
  a traceback.

Consumed by the session summary and the job payload here, and by KonfAI Studio for its own pre-prompt
and action buttons — one definition, no second opinion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, get_args

from ruamel.yaml.error import YAMLError

from .config_io import YAML_SAFE
from .workspace import WORKFLOW_CONFIG_FILES, WORKFLOW_ROOT_KEYS

Stage = Literal[
    "dataset_inspection",
    "action_selection",
    "app_selection",
    "configuration",
    "running",
    "failed",
    "checkpoint_selection",
    "prediction",
    "evaluation",
    "result_review",
    "completed",
]

STAGES: tuple[str, ...] = get_args(Stage)

# What the current stage owes, in one line. This is the guidance an agent should act on *now* — the
# standing rules belong in its system prompt, not here.
STAGE_FOCUS: dict[str, str] = {
    # How a dataset is reported is a standing rule, not a stage instruction: by the time the groups are
    # known the stage has already moved on, so the shape lives in the system prompt where it applies
    # whenever a dataset is described.
    "dataset_inspection": (
        "Inspect the dataset -- inspect_dataset; browse_dataset only if the root is still unknown -- and "
        "report what it holds."
    ),
    "action_selection": (
        "The groups are known; inspect_dataset for what they do not give (shape/spacing, label classes, "
        "incomplete cases). Then ASK which of the four routes they want -- train from scratch, fine-tune "
        "a published app, predict with one as is, or evaluate one -- saying which the data actually "
        "supports (training and evaluation need a target group; plenty of cases with targets favours "
        "training or fine-tuning, few or none favours predicting). Recommend one, in one line, and wait "
        "for their answer: the route is theirs to choose, not yours."
    ),
    "app_selection": (
        "Check the app's expected inputs against the dataset groups, say plainly whether it fits, then "
        "run it as published."
    ),
    "configuration": (
        "Validate the config at level 'train_step' — the default level runs no train step — then launch."
    ),
    "running": "A job is open: wait for it to reach a terminal state, then report the real outcome.",
    "failed": (
        "Diagnose it from the job log: the cause in one line, then the correction, or the single choice "
        "the user has to make."
    ),
    "checkpoint_selection": "Training produced checkpoints: pick the best, say which and why, then predict with it.",
    "prediction": (
        "Predictions exist: evaluate them when the dataset carries a reference group, else open one and "
        "judge it against its input."
    ),
    "evaluation": "Read the metrics: the aggregate per metric, its direction, and the worst cases.",
    "result_review": "Give the result, then propose packaging, comparing or exporting it.",
    "completed": "Say what is genuinely left to do, or that the objective is met.",
}

# The tools that carry each stage forward, best first. A caller keeps the ones its readiness allows —
# these are candidates, not a script.
STAGE_ACTIONS: dict[str, list[str]] = {
    "dataset_inspection": ["inspect_dataset", "browse_dataset", "design_config_strategy", "initialize_session"],
    "action_selection": ["list_apps", "fine_tune_app", "design_config_strategy", "run_train"],
    "app_selection": ["describe_app", "run_app_infer", "fine_tune_app", "list_app_parameters"],
    "configuration": ["validate_config_semantics", "review_config_semantics", "run_train", "write_workflow_config"],
    "running": ["wait_for_job", "read_live_metrics", "get_job_status", "cancel_job"],
    "failed": ["read_job_log", "validate_config_semantics", "get_job_status"],
    "checkpoint_selection": ["run_prediction", "read_training_curves", "package_app_from_session"],
    "prediction": ["run_evaluation", "preview_volume", "package_app_from_session"],
    "evaluation": ["get_run_metrics", "leaderboard", "compare_runs"],
    "result_review": ["package_app_from_session", "compare_runs", "export_run_record", "leaderboard"],
    "completed": ["summarize_session", "package_app_from_session", "leaderboard"],
}


@dataclass(frozen=True)
class Diagnosis:
    """A failed run, read: what happened, what to change, and whether it is the agent's call to fix it."""

    kind: str
    summary: str
    fix: str
    recoverable: bool  # False = a decision only the user can make (which data, which run to overwrite)

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "summary": self.summary, "fix": self.fix, "recoverable": self.recoverable}


# Ordered: the first match wins, so a specific cause is named before the bare traceback it arrived in.
DIAGNOSES: tuple[tuple[Diagnosis, tuple[str, ...]], ...] = (
    (
        Diagnosis(
            "outputs_exist",
            "this run's outputs already exist",
            "re-run overwriting them, or use a new run name",
            False,
        ),
        ("already exists", "--overwrite", "overwrite=true"),
    ),
    (
        Diagnosis(
            "out_of_memory",
            "the run ran out of memory",
            "lower the patch size or the batch size, or run it on a device with more memory",
            True,
        ),
        ("out of memory", "outofmemoryerror", "memoryerror", "cuda oom"),
    ),
    (
        Diagnosis("disk_full", "the disk is full", "free space or move the workspace to a larger volume", False),
        ("no space left", "disk quota exceeded"),
    ),
    (
        Diagnosis(
            "missing_file",
            "an input path does not exist",
            "confirm the dataset path and the group filenames, then correct the config",
            False,
        ),
        ("no such file or directory", "filenotfounderror", "does not exist"),
    ),
    (
        Diagnosis(
            "unsupported_format",
            "an input format is not readable",
            "convert the inputs, or install the matching KonfAI extra (check_external_dependency)",
            False,
        ),
        ("unsupported", "unknown extension", "no backend", "supported_extensions"),
    ),
    (
        Diagnosis(
            "group_mismatch",
            "the dataset groups do not match what the model expects",
            "map the dataset groups onto the expected ones (prepare_dataset_aliases), or pick a fitting app",
            True,
        ),
        ("missing group", "requires group", "input group", "groups_src", "not paired"),
    ),
    (
        Diagnosis(
            "shape_mismatch",
            "a tensor shape does not line up",
            "align the patch size, the channel count and the label classes with the data",
            True,
        ),
        ("size mismatch", "shape mismatch", "cannot be broadcast", "must match the size", "indexerror"),
    ),
    (
        Diagnosis("config_error", "the config is invalid", "fix the offending key, then re-validate", True),
        ("configerror", "validationerror", "keyerror", "is not a valid", "yaml"),
    ),
    (
        Diagnosis(
            "runtime_error",
            "the run raised an exception",
            "read the traceback, correct the cause, then relaunch",
            True,
        ),
        ("traceback (most recent call last)", "error:", "exception"),
    ),
)

CANCELLED = Diagnosis(
    "cancelled",
    "the run was stopped on purpose (the user's Stop button, cancel_job, or a Ctrl-C) — not a failure",
    "nothing to fix; resume or relaunch it when the user is ready",
    False,
)
UNKNOWN = Diagnosis("unknown", "the run failed", "read the job log to find the cause", True)
MISSING_OUTPUTS = Diagnosis(
    "missing_outputs",
    "the job reported success but wrote nothing",
    "check the run name and the output paths in the config, then relaunch",
    True,
)


def diagnose(text: str, *, status: str = "error") -> Diagnosis:
    """Read a failure from a log tail or an error string. Never returns None: an unmatched failure is
    still a failure, and saying "read the log" beats saying nothing."""
    if status in {"killed", "cancelled"}:
        return CANCELLED
    lowered = (text or "").lower()
    for diagnosis, needles in DIAGNOSES:
        if any(needle in lowered for needle in needles):
            return diagnosis
    return UNKNOWN


ACTIVE_STATUS = frozenset({"queued", "running", "waiting"})
FAILED_STATUS = frozenset({"error", "killed", "cancelled"})

# A finished job of this kind leaves the experiment at this stage — the successor step, not "done".
_JOB_KIND_STAGE: dict[str, str] = {
    "train": "checkpoint_selection",
    "finetune": "checkpoint_selection",
    "prediction": "prediction",
    "infer": "prediction",
    "evaluation": "evaluation",
    "evaluate": "evaluation",
    "pipeline": "evaluation",
    "uncertainty": "completed",
}

# What a finished job of each kind must have left in the workspace. A session workflow writes there by
# definition, so "done" with nothing written is a failure however the job record reads. App jobs are
# absent on purpose: they write wherever the user pointed them, so their silence proves nothing.
_EXPECTED_OUTPUT: dict[str, str] = {
    "train": "checkpoints",
    "prediction": "predictions",
    "evaluation": "metrics",
}


@dataclass(frozen=True)
class Facts:
    """What the workspace shows. Every field is read back from disk — none of it is remembered."""

    session: str = ""
    workspace: str = ""
    dataset: str = ""
    groups: list[str] = field(default_factory=list)
    cases: int = 0
    configs: list[str] = field(default_factory=list)
    apps: list[str] = field(default_factory=list)
    checkpoints: list[str] = field(default_factory=list)
    predictions: list[str] = field(default_factory=list)
    evaluations: list[str] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)
    run: str = ""
    job_id: str = ""
    job_kind: str = ""
    job_status: str = ""
    job_error: str = ""
    attempts: int = 0  # consecutive failed jobs, newest first — the retry budget, counted not remembered

    @property
    def has_reference(self) -> bool:
        """Two or more groups per case means the data carries a target — a label map, or a paired
        modality. With a single group there is nothing to train against or score against."""
        return len(self.groups) >= 2


def derive_stage(facts: Facts) -> str:
    """The stage, from the facts alone — in precedence order, most recent evidence first.

    Deliberately not a map from tool names: a client that drove the session with tools this server has
    never heard of, or a user who copied files in by hand, still lands on the right stage."""
    if facts.job_status in ACTIVE_STATUS:
        return "running"
    if facts.job_status in FAILED_STATUS:
        return "failed"
    if facts.job_status == "done" and facts.job_kind in _JOB_KIND_STAGE:
        # The newest evidence is a finished run — but only if it left what its kind must leave.
        expected = _EXPECTED_OUTPUT.get(facts.job_kind)
        if expected is not None and not getattr(facts, expected):
            return "failed"
        return _JOB_KIND_STAGE[facts.job_kind]
    if facts.metrics or facts.evaluations:
        return "result_review"
    if facts.predictions:
        return "prediction"
    if facts.checkpoints:
        return "checkpoint_selection"
    if facts.configs:
        return "configuration"
    if facts.apps:
        return "app_selection"
    # Knowing the path is not knowing the data: a route is only chosen once the groups are in hand.
    if facts.groups:
        return "action_selection"
    return "dataset_inspection"


def stage_actions(stage: str, facts: Facts) -> list[str]:
    """The stage's tools, minus the ones its own facts rule out — nothing to package without a
    checkpoint, nothing to rank without a metric.

    Only *known* facts filter. An action is never dropped for something the workspace cannot tell yet
    (whether the data carries a target, say): a missing fact is not a refusal."""
    blocked: set[str] = set()
    if not facts.checkpoints:
        blocked.add("package_app_from_session")
    if not facts.metrics:
        blocked |= {"leaderboard", "compare_runs", "get_run_metrics"}
    kept = [action for action in STAGE_ACTIONS.get(stage, []) if action not in blocked]
    return kept or ["summarize_session"]


def _names(root: Path, pattern: str, *, dirs: bool = False) -> list[str]:
    """Sorted names matching a glob under ``root`` — empty when the directory is not there yet."""
    if not root.is_dir():
        return []
    found = {path.name for path in root.glob(pattern) if path.is_dir() == dirs}
    return sorted(found)[:50]


# A dataset scan costs ~700ms on a few dozen cases — too much to repeat every turn, and pointless: the
# answer only changes when the data does. Keyed by the directory's own mtime, so adding a case re-derives.
_SCANNED: dict[tuple[str, float], tuple[list[str], int]] = {}


def _scan_dataset(path: Path) -> tuple[list[str], int]:
    """The groups and case count of a dataset directory, read from the data itself.

    The dataset lives outside the workspace, so it is the one fact the workspace cannot supply — and
    without it an agent is told to choose a route for data whose shape nobody has looked at. Structure
    only: no pixel is opened.
    """
    try:
        key = (str(path), path.stat().st_mtime)
    except OSError:
        return [], 0
    if key not in _SCANNED:
        from .dataset_inspection import DatasetInspectionMixin  # local: keeps this module free of numpy

        try:
            scan = DatasetInspectionMixin()._scan_dataset_structure(path)
            _SCANNED[key] = (sorted(scan["groups"]), int(scan["total_cases"]))
        except (OSError, ValueError, KeyError):
            _SCANNED[key] = ([], 0)
    return _SCANNED[key]


def _newest(*roots: Path) -> str:
    """The most recently written run directory across these roots — which run an experiment with no job
    history is about. An imported workspace, or one whose records were cleaned, still names its run."""
    found = [path for root in roots if root.is_dir() for path in root.iterdir() if path.is_dir()]
    return max(found, key=lambda path: path.stat().st_mtime).name if found else ""


def _dataset_from_config(path: Path, workflow: str = "train") -> tuple[str, list[str]]:
    """The dataset root and the group names a workflow config points at.

    Parsed with the safe YAML loader, never with KonfAI's own ``Config``: *reading* a KonfAI config
    rewrites it on disk, and a state probe must leave the experiment exactly as it found it.
    """
    try:
        data = YAML_SAFE.load(path.read_text(encoding="utf-8"))
    except (OSError, YAMLError):
        return "", []
    root = data.get(WORKFLOW_ROOT_KEYS.get(workflow, "Trainer")) if isinstance(data, dict) else None
    dataset = root.get("Dataset") if isinstance(root, dict) else None
    if not isinstance(dataset, dict):
        return "", []
    entries = [e for e in dataset.get("dataset_filenames") or [] if isinstance(e, str) and e not in {"", "None"}]
    # A dataset entry is "<path>:<flags>:<extension>" — the path is the part before those two fields.
    first = entries[0].rsplit(":", 2)[0] if entries else ""
    groups = dataset.get("groups_src")
    return first, sorted(groups)[:12] if isinstance(groups, dict) else []


def _attempts(jobs: list[dict[str, Any]]) -> int:
    """How many runs failed in a row, newest first. The retry budget is counted from the job records
    rather than tracked in a conversation, so it is right after a restart and cannot be talked around."""
    failed = 0
    for job in jobs:
        if str(job.get("status") or "") not in FAILED_STATUS:
            break
        failed += 1
    return failed


def collect_facts(
    workspace: Path,
    *,
    session: str = "",
    jobs: list[dict[str, Any]] | None = None,
    latest_job: dict[str, Any] | None = None,
    dataset: str = "",
    groups: list[str] | None = None,
    cases: int = 0,
) -> Facts:
    """Read an experiment's workspace. ``jobs`` are its job records newest first; ``dataset``/``groups``/
    ``cases`` are what the caller already knows about the input data (it lives outside the workspace).
    Everything else comes off disk."""
    history = jobs if jobs is not None else ([latest_job] if latest_job else [])
    job = latest_job or (history[0] if history else {}) or {}
    if not dataset:
        train_config = workspace / WORKFLOW_CONFIG_FILES.get("train", "Config.yml")
        dataset, config_groups = _dataset_from_config(train_config)
        groups = groups or config_groups
    if dataset and not groups:  # a path is not knowledge of the data: read its structure
        groups, scanned_cases = _scan_dataset(Path(dataset).expanduser())
        cases = cases or scanned_cases
    return Facts(
        session=session,
        workspace=str(workspace),
        dataset=dataset,
        groups=sorted(groups or []),
        cases=cases,
        configs=[name for name in sorted(WORKFLOW_CONFIG_FILES.values()) if (workspace / name).is_file()],
        # An imported app lands in the workspace as its own directory, so it is a fact like any other.
        apps=_names(workspace / "Apps", "*", dirs=True),
        checkpoints=_names(workspace / "Checkpoints", "*", dirs=True),
        predictions=sorted({path.name for path in workspace.glob("**/Predictions/*") if path.is_dir()})[:50],
        evaluations=_names(workspace / "Evaluations", "*", dirs=True),
        metrics=sorted({path.name for path in workspace.glob("**/Metric_*.json")})[:50],
        run=str(job.get("run_name") or "") or _newest(workspace / "Checkpoints", workspace / "Predictions"),
        job_id=str(job.get("job_id") or ""),
        job_kind=str(job.get("kind") or ""),
        job_status=str(job.get("status") or ""),
        job_error=str(job.get("error") or ""),
        attempts=_attempts(history),
    )


MAX_ATTEMPTS = 2  # consecutive failures the agent may correct on its own before the choice becomes the user's


def _failure_focus(facts: Facts, diagnosis: Diagnosis) -> str:
    """What to do about a failure, said outright — the cause and the correction, so the next turn acts on
    it instead of re-deriving it from a traceback, and stops once the budget is spent."""
    run = facts.run or "the run"
    if facts.attempts >= MAX_ATTEMPTS:
        return (
            f"{run} failed {facts.attempts} times in a row: {diagnosis.summary}. Stop correcting it. Lay "
            f"out the options ({diagnosis.fix}) with the trade-off and ask for the one decision you need."
        )
    if not diagnosis.recoverable:
        return (
            f"{run} failed: {diagnosis.summary}. This one is the user's call — offer the choice "
            f"({diagnosis.fix}) and do whichever they pick."
        )
    return (
        f"{run} failed: {diagnosis.summary}. Confirm it in the job log, apply the correction "
        f"({diagnosis.fix}), then relaunch once."
    )


def experiment_state(facts: Facts) -> dict[str, Any]:
    """The compact state payload: where the experiment stands, what that owes, and what carries it on.

    This is the one answer both the MCP session summary and KonfAI Studio's pre-prompt are built from."""
    stage = derive_stage(facts)
    payload: dict[str, Any] = {
        "stage": stage,
        "focus": STAGE_FOCUS[stage],
        "dataset": facts.dataset,
        "groups": facts.groups,
        "cases": facts.cases,
        "has_reference": facts.has_reference,
        "app": facts.apps[0] if facts.apps else "",
        "run": facts.run,
        "job_id": facts.job_id,
        "checkpoints": len(facts.checkpoints),
        "predictions": len(facts.predictions),
        "metrics": len(facts.metrics),
        "attempts": facts.attempts,
        "next_actions": stage_actions(stage, facts),
    }
    if stage == "failed":
        # Landing here from a "done" job can only mean it wrote nothing where its kind must write.
        diagnosis = (
            MISSING_OUTPUTS if facts.job_status == "done" else diagnose(facts.job_error, status=facts.job_status)
        )
        payload["diagnosis"] = diagnosis.as_dict()
        payload["focus"] = _failure_focus(facts, diagnosis)  # the generic line is no use once the cause is known
    elif stage == "running":
        payload["focus"] = (
            f"Job {facts.job_id or facts.run} is still open: wait for it to reach a terminal state, then "
            f"report the real outcome — status, where the outputs landed, the key numbers."
        )
    return payload


def state_line(state: dict[str, Any]) -> str:
    """The state as one line of structured identifiers — what a turn carries instead of a transcript."""
    parts = [f"stage={state.get('stage', '')}"]
    for key in ("dataset", "app", "run", "job_id"):
        if state.get(key):
            parts.append(f"{key}={state[key]}")
    if state.get("groups"):
        parts.append("groups=" + "/".join(str(group) for group in state["groups"][:6]))
    for key in ("cases", "checkpoints", "predictions", "metrics"):
        if state.get(key):
            parts.append(f"{key}={state[key]}")
    diagnosis = state.get("diagnosis")
    if isinstance(diagnosis, dict) and diagnosis.get("kind"):
        parts.append(f"failure={diagnosis['kind']}")
    return " ".join(parts)


def log_tail(path: Path, *, limit: int = 4000) -> str:
    """The tail of a job log — enough to diagnose, never the whole file."""
    try:
        size = path.stat().st_size
        with path.open(encoding="utf-8", errors="replace") as handle:
            if size > limit:
                handle.seek(size - limit)
                handle.readline()
            return handle.read()
    except OSError:
        return ""


def job_diagnosis(job: dict[str, Any], log_path: Path | None) -> dict[str, Any] | None:
    """Diagnose a terminal-bad job from its recorded error and its log tail; None when it did not fail."""
    status = str(job.get("status") or "")
    if status not in FAILED_STATUS:
        return None
    text = str(job.get("error") or "")
    if log_path is not None and not text:
        text = log_tail(log_path)
    return diagnose(text, status=status).as_dict()
