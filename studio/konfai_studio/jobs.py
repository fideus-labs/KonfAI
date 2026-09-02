# SPDX-License-Identifier: Apache-2.0
"""Job discovery and the live feed: reading konfai-mcp job records, tailing a run's runtime/console
logs, turning konfai's runtime lines into structured SSE metric/progress events, and the persistent
``/api/live`` stream."""

from __future__ import annotations

import asyncio
import json
import math
import os
import time
from collections import Counter
from collections.abc import AsyncIterator, Callable
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import Any, NamedTuple

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
from konfai_mcp.live_parse import parse_host_stats, parse_live_metric_line, parse_live_progress, progress_label

from .paths import _sane_session, _session_dir, _session_jobs_dir

router = APIRouter()

_TERMINAL_STATUS = {"done", "error", "killed", "cancelled"}
_LOG_BACKFILL = 32_000  # bytes: on connect, replay only the recent tail of a large log, not its full history
_HOST_KEYS = ("memory_gb", "memory_percent", "memory_gpu_gb", "memory_gpu_percent", "cpu_percent")
_MTIME_LIVE_WINDOW = 8.0  # a run log written this recently reads as live when no job record claims it
#: Lowercase substrings that mark a ``[KonfAI]`` console line as routine startup chatter. A denylist,
#: not an allowlist: everything the framework says is worth a watcher's attention unless it is one of
#: these, and an allowlist silently swallowed each new line nobody thought to add to it.
_KONFAI_ROUTINE = ("memory_budget:", "compute displacement field", "compute in progress")
_PING_EVERY = 10.0  # seconds between keep-alives, so a client can tell a quiet stream from a dead one
_RUN_ROOT_KIND = {
    "Statistics": "train",
    "Predictions": "prediction",
    "Evaluations": "evaluation",
    "Uncertainties": "uncertainty",
    "Transforms": "transform",
}
# App job kinds → the workflow vocabulary the client speaks: an `infer` job runs a prediction, an
# `evaluate` job an evaluation, a `finetune` job a training. Announced under the app vocabulary, a run
# sits outside every panel the client gates on kind (the evaluation table, the prediction browse
# target, the sub-tab order).
_APP_RUN_KIND = {"infer": "prediction", "evaluate": "evaluation", "finetune": "train"}


def _finite(value: Any) -> Any:
    """The same value, with non-finite floats replaced by null.

    A metric can be NaN for ordinary reasons: a Dice against an empty prediction is 0/0, and a diverged
    loss is inf. ``json.dumps`` spells those ``NaN`` / ``Infinity``, which JSON does not define and
    ``JSON.parse`` refuses: the browser throws on that frame and loses every frame behind it, so one NaN
    took the whole live feed down. Null is what the value actually is: absent."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite(item) for item in value]
    return value


def _sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(_finite(event))}\n\n"


def _all_jobs(session: str) -> list[dict[str, Any]]:
    jobs_dir = _session_jobs_dir(session)
    out: list[dict[str, Any]] = []
    if jobs_dir.is_dir():
        for record in jobs_dir.glob("*/job.json"):
            try:
                out.append(json.loads(record.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
    return out


def _epoch(stamp: Any) -> float | None:
    """A job record's timestamp as an epoch float: konfai-mcp writes either a float or an ISO string."""
    if isinstance(stamp, (int, float)):
        return float(stamp)
    if isinstance(stamp, str):
        with suppress(ValueError):
            return datetime.fromisoformat(stamp).timestamp()
    return None


def _job_created(job: dict[str, Any]) -> float:
    """When a job started. Jobs are ordered by this, not by file mtime: a terminal job's json can be
    rewritten later (status monitor), which mtime would misread as 'newest', making the feed follow a
    dead run over a fresh one."""
    return _epoch(job.get("created_at")) or 0.0


def _latest_job(session: str) -> dict[str, Any] | None:
    """A task's most recently created job (its ``job.json`` payload), newest ``created_at`` wins."""
    return max(_all_jobs(session), key=_job_created, default=None)


def _pid_alive(pid: Any) -> bool:
    try:
        value = int(pid)
    except (TypeError, ValueError):
        return False
    if value <= 0:
        return False
    # A crashed job whose parent-reaper died leaves a zombie: it has exited but still owns a pid, so both
    # os.kill(0) and psutil see it as "alive". On Linux read the state directly and treat 'Z' as dead.
    try:
        stat = Path(f"/proc/{value}/stat").read_text(encoding="utf-8")
        return stat.rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, IndexError):
        pass
    try:
        os.kill(value, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class _Claim(NamedTuple):
    """What a job record says about the log it owns: a status, and when it stopped owning it.

    ``ended`` is None when the record does not date its end, and an undated claim is taken at its word."""

    status: str
    ended: float | None

    def status_at(self, mtime: float) -> str:
        """This claim's status, unless a log written since it ended says the run is going again.

        A terminal record describes the run it ended, not the one writing the file now: relaunching
        from the terminal Studio itself offers appends to the same log."""
        outlived = (
            self.status in _TERMINAL_STATUS
            and self.ended is not None
            and mtime > self.ended + 1.0
            and (time.time() - mtime) < _MTIME_LIVE_WINDOW
        )
        return "running" if outlived else self.status


def _claim_of(job: dict[str, Any]) -> _Claim:
    return _Claim(_live_status(job), _epoch(job.get("finished_at")))


def _runtime_log(job: dict[str, Any]) -> Path | None:
    """The runtime log a workflow job names, if it names one."""
    runtime = job.get("runtime_log_path")
    return Path(runtime) if runtime else None


def _job_kind(job: dict[str, Any]) -> str:
    """A job's kind in the workflow vocabulary the client speaks, normalised once for every emitter."""
    kind = str(job.get("kind") or "")
    return _APP_RUN_KIND.get(kind, kind)


def _log_key(job: dict[str, Any]) -> str | None:
    log = _runtime_log(job)
    return str(log) if log else None


def _output_key(job: dict[str, Any]) -> str | None:
    """The output subtree an app job owns. A job that names its own log is claimed by that log instead."""
    output = job.get("output_path")
    return str(Path(output)) if output and not job.get("runtime_log_path") else None


def _newest_claims(jobs: list[dict[str, Any]], key_of: Callable[[dict[str, Any]], str | None]) -> dict[str, _Claim]:
    """The claim of the newest job under each key: one run directory is often written by several jobs
    (re-runs share one ``log_0.txt``), and a fresh run must win over an old one."""
    newest: dict[str, tuple[float, dict[str, Any]]] = {}
    for job in jobs:
        key = key_of(job)
        if key is None:
            continue
        created = _job_created(job)
        if created >= newest.get(key, (float("-inf"), {}))[0]:
            newest[key] = (created, job)
    return {key: _claim_of(job) for key, (_, job) in newest.items()}


def _status_for(job: dict[str, Any], log: Path | None) -> str:
    """A job's status, read against the log it owns: the single reading every block of the feed makes."""
    claim = _claim_of(job)
    if log is not None and log.is_file():
        try:
            return claim.status_at(log.stat().st_mtime)
        except OSError:
            pass  # gone between the check and the read: initialize_session(overwrite=True) rmtrees
    return claim.status


def _live_status(job: dict[str, Any]) -> str:
    """The job's status, but a still-'running' record whose process is gone reads as 'failed': the MCP
    monitor that would flip it may have died (e.g. a server restart), leaving the status stale."""
    status = str(job.get("status") or "")
    if status in {"running", "waiting", "queued"} and not _pid_alive(job.get("pid")):
        return "error"  # a terminal status the UI renders red; the log tail shows the traceback
    return status


def _tail_start(path: Path) -> int:
    """Byte offset to begin following a log at: near the end of an already-large file (so a mid-run
    (re)connect replays only its recent tail), **aligned to the next line start** so the first emitted
    line is never a mid-line fragment."""
    if not path.is_file():
        return 0
    size = path.stat().st_size
    if size <= _LOG_BACKFILL:
        return 0
    with path.open(encoding="utf-8", errors="replace") as handle:
        handle.seek(size - _LOG_BACKFILL)
        handle.readline()  # discard the partial line the byte-seek landed inside
        return handle.tell()


def _tail_lines(path: Path, pos: int, buf: str) -> tuple[list[str], int, str]:
    """Complete new lines appended to ``path`` since byte ``pos``. ``buf`` carries an incomplete trailing
    line between reads so a read landing mid-line never yields a fragment. Returns (lines, new pos, new
    buf). A missing file yields nothing; a truncated/rotated file is clamped."""
    if not path.is_file():
        return [], pos, buf
    with path.open(encoding="utf-8", errors="replace") as handle:
        handle.seek(min(pos, path.stat().st_size))
        buf += handle.read()
        pos = handle.tell()
    parts = buf.split("\n")
    return parts[:-1], pos, parts[-1]  # last element is the (possibly empty) incomplete remainder


def _phase_stage(label: str) -> str:
    """The stage a tqdm phase belongs to, from its label, so the client never guesses. 'Caching Train' →
    caching, 'Training' → training, 'Metric VALIDATION' → evaluation, 'Prediction' → prediction."""
    head = label.split(maxsplit=1)[0].lower() if label else ""
    return {"metric": "evaluation"}.get(head, head or "caching")


def _discover_run_logs(job: dict[str, Any]) -> list[tuple[Path, str, str]]:
    """Every konfai runtime log a job writes, as (log_path, run_name, kind). A workflow job names one
    explicitly (runtime_log_path). An app job writes runtime_log_path=None but produces its logs under
    output_path: infer → one, fine-tune → one per finetuned checkpoint (Statistics/<run>), pipeline →
    predict + evaluate + uncertainty. Each discovered log becomes its own per-run feed (MR2CT_01, 02, …)."""
    runtime = job.get("runtime_log_path")
    if runtime:
        return [(Path(runtime), str(job.get("run_name") or job.get("kind") or "run"), str(job.get("kind") or ""))]
    output = job.get("output_path")
    if not output:
        return []
    base = Path(output)
    job_kind = _job_kind(job)
    found: list[tuple[Path, str, str]] = []
    seen: set[Path] = set()
    for pattern in (
        "*/log_0.txt",
        "Statistics/*/log_0.txt",
        "Predictions/*/log_0.txt",
        "Evaluations/*/log_0.txt",
        "Uncertainties/*/log_0.txt",
        "Transforms/*/log_0.txt",
    ):
        for log in sorted(base.glob(pattern)):
            if log in seen:
                continue
            seen.add(log)
            root = log.parent.parent.name if log.parent.parent != base else ""
            found.append((log, log.parent.name, _RUN_ROOT_KIND.get(root, job_kind)))
    return found


def _runtime_events(line: str, run: str, kind: str, step: int) -> tuple[list[dict[str, Any]], int]:
    """The structured SSE events for one runtime-log line, tagged with its run + kind: a ``metric``
    (per-model values + learning rate + host memory/CPU), or a ``progress`` for a metric-less phase
    (caching, plain inference). Returns (events, new step). The step is a per-run counter."""
    entry = parse_live_metric_line(line)
    if entry is not None:
        step += 1
        values = dict(entry["flat_metrics"])
        if entry["stage"] == "Training":
            # learning rate is a training-only signal, never chart it under validation
            values.update({f"{m['name']}:lr": m["lr"] for m in entry.get("models", []) if m.get("lr")})
        host = {k: entry[k] for k in _HOST_KEYS if k in entry}
        return [
            {
                "type": "metric",
                "run": run,
                "kind": kind,
                "step": step,
                "stage": entry["stage"].lower(),
                # Training/Validation/Prediction, or the eval split ("Metric TRAIN") carried on `label`.
                "label": entry.get("label") or entry["stage"],
                "values": values,
                "progress": entry.get("progress"),
                **host,
            }
        ], step
    progress = parse_live_progress(line)
    if progress is not None:
        label = progress_label(line)
        return [
            {
                "type": "progress",
                "run": run,
                "kind": kind,
                "stage": _phase_stage(label),
                "label": label,
                "progress": progress,
                **parse_host_stats(line),
            }
        ], step
    return [], step


def _transform_outputs(session: str, base: str) -> list[dict[str, str]]:
    """Every chain's destination from a transform run's ``outputs.json``, in the run's own terms.

    A transform's run directory holds a log, a plan and a config copy: the volumes land wherever each
    ``Write`` pointed, one destination per chain, which the workflow records in ``outputs.json``. Each
    entry keeps what the run wrote (``group_src``, ``group_dest``, ``dataset``, ``group``, ``format``)
    plus ``path``: what the run says is on disk (the ``.h5`` file itself for an h5 store, the dataset
    directory otherwise), session-relative when it lives inside the session so Browse resolves it like
    every other path, else as recorded. A manifest from before 1.9 has no ``path``: its ``dataset``
    stands in. Empty for a run that has no manifest.
    """
    manifest = _session_dir(session) / base / "outputs.json"
    if not manifest.is_file():
        return []
    try:
        outputs = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(outputs, list):
        return []
    found: list[dict[str, str]] = []
    for entry in outputs:
        if not isinstance(entry, dict):
            continue
        dataset = entry.get("dataset")
        if not isinstance(dataset, str) or not dataset:
            continue
        path = entry.get("path")
        if not isinstance(path, str) or not path:  # a manifest is data: a non-string path is not one
            path = dataset
        with suppress(ValueError):
            path = str(Path(path).relative_to(_session_dir(session)))
        found.append(
            {
                **{key: str(entry.get(key) or "") for key in ("group_src", "group_dest", "group", "format")},
                "dataset": dataset,
                "path": path,
            }
        )
    return found


def _run_data_dir(session: str, base: str, kind: str, outputs: list[dict[str, str]]) -> str:
    """Where a run's DATA is, when that is not the run directory itself.

    A transform's volumes are wherever its chains wrote (``outputs``: the first chain's destination
    stands for the run, every chain is carried beside it). An app prediction writes its volumes under
    ``<run>/Output`` (the session-root layout uses ``Predictions/<run>/Dataset``, which the client
    already knows how to browse). Pointing Browse at the run directory would open a YAML file for
    someone who just asked to see what was produced. Answers "" for every kind whose data IS under its
    run directory, and the caller keeps its usual path.
    """
    if kind == "prediction" and (_session_dir(session) / base / "Output").is_dir():
        return f"{base}/Output"
    return outputs[0]["path"] if outputs else ""


def _discover_session_runs(
    session: str, jobs: list[dict[str, Any]] | None = None
) -> list[tuple[Path, str, str, str, str]]:
    """Every run of the experiment as (log_path, run_name, kind, status, base): one per runtime log.

    Workflow jobs write under the session roots (Statistics/Predictions/Evaluations); app jobs (infer /
    fine-tune / pipeline / uncertainty) write under an isolated ``output_path`` subtree, so EVERY app job's
    output_path is scanned (not only the newest), else an earlier isolated run vanishes the moment a newer
    job exists. ``base`` is the run dir relative to the session root ("Statistics/<run>" for a session-root
    run, "<app_output>-<hash>/Statistics/<run>" for an isolated app run) the one datum that lets the
    previews / TensorBoard / Browse helpers resolve the real on-disk location. Status comes from the newest
    job that names the log (workflow) or owns the output_path (app), else from how recently the log was
    written. Newest first."""
    base_root = _session_dir(session)
    jobs = _all_jobs(session) if jobs is None else jobs
    by_log = _newest_claims(jobs, _log_key)
    by_output = _newest_claims(jobs, _output_key)
    found: list[tuple[Path, str, str, str, str, float]] = []
    seen: set[Path] = set()

    def add(log: Path, run_name: str, kind: str, app_claim: _Claim | None) -> None:
        if log in seen or not log.is_file():
            return
        seen.add(log)
        try:
            mtime = log.stat().st_mtime
        except OSError:
            return  # gone between the check and the read: initialize_session(overwrite=True) rmtrees
        claim = by_log.get(str(log)) or app_claim
        if claim is not None:
            status = claim.status_at(mtime)
        else:  # nothing claims this log: how recently it was written is all there is to go on
            status = "running" if (time.time() - mtime) < _MTIME_LIVE_WINDOW else "done"
        try:
            base = str(log.parent.relative_to(base_root))
        except ValueError:
            base = log.parent.name
        found.append((log, run_name, kind, status, base, mtime))

    for root, kind in (
        ("Statistics", "train"),
        ("Predictions", "prediction"),
        ("Evaluations", "evaluation"),
        ("Transforms", "transform"),
    ):
        directory = base_root / root
        if directory.is_dir():
            for log in directory.glob("*/log_0.txt"):
                add(log, log.parent.name, kind, None)
    for job in jobs:  # app runs land outside the three roots, under each app job's own output_path
        out_path = job.get("output_path")
        if not out_path:
            continue
        for log, run_name, kind in _discover_run_logs(job):
            add(log, run_name, kind, by_output.get(str(Path(out_path))))
    found.sort(key=lambda row: row[5], reverse=True)
    # Two runs of the same app share a name: its log directory is named after the app, so every re-run
    # produces another `ImpactSynth`. They are different runs (different output directories), and left
    # sharing a name they collapse into one feed whose status is whichever was read last, which is how a
    # finished run comes to stand in for the one that is still going. Disambiguate by what differs.
    duplicated = {key for key, n in Counter((run, kind) for _, run, kind, _, _, _ in found).items() if n > 1}
    return [
        (
            log,
            f"{run} · {base.split('/')[0].rsplit('-', 1)[-1]}" if (run, kind) in duplicated and base else run,
            kind,
            status,
            base,
        )
        for log, run, kind, status, base, _ in found
    ]


@router.get("/api/live")
async def live(session: str = Query("default")) -> StreamingResponse:
    """Tail a task's most recent job in real time (SSE): the console log as raw text, and konfai's runtime
    log as **structured** metrics + progress: parsed by konfai-mcp's own ``live_parse`` (one source of
    truth, no re-implementation here).

    Two logs, two roles. The console wrapper log (header, warm-up prints, crash tracebacks) streams as
    ``log`` lines. konfai writes its per-iteration training tqdm to the runtime file, not stdout, so that
    file streams as ``metric`` events (a stage + flat metric values + the ``progress`` bar) and, for the
    metric-less data-caching phase, ``progress`` events, never as raw log lines, which would bury the
    console tail under thousands of tqdm frames.

    The connection is **persistent**: a job finishing is announced once (terminal status) but the stream
    keeps watching, so the next job the user launches is picked up on the same connection.
    """
    name = _sane_session(session)

    async def gen() -> AsyncIterator[str]:
        last_ping = time.monotonic()
        console_key: str | None = None  # which job's console log is being followed
        job_status: str | None = None  # last status announced for it, so only changes go out
        cpath: Path | None = None
        cpos = 0
        cbuf = ""
        feeds: dict[str, dict[str, Any]] = {}  # log-path -> {run, kind, path, pos, buf, step}: one per run, kept
        announced: dict[str, tuple[str, str]] = {}  # run key -> the (status, base) last emitted for it
        idle_sent = False

        def state_of(run: str, kind: str, status: str, base: str = "") -> list[str]:
            """One run, one place that says where it stands. A run's tab is created by the ``run`` event
            and released by the ``status`` event; emitting them from two blocks let the first mark the
            transition as seen and the second skip it, leaving Stop on screen over a finished run."""
            key = f"{kind}:{run}"
            was = announced.get(key)
            # Keyed on what the frame carries, not on the status alone: the announce block speaks first
            # with no base, and a status-keyed guard then swallowed the discovery frame that knows where
            # the run writes. The client kept an empty base, so Browse opened a config and Delete hid.
            # An empty base is "I do not know where this run writes", not a second answer: keeping the
            # one already announced is what stops the two emitters alternating and re-announcing the
            # same run on every pass of the loop.
            base = base or (was[1] if was else "")
            if was == (status, base):
                return []
            announced[key] = (status, base)
            outputs = _transform_outputs(name, base) if base else []
            out = [
                _sse(
                    {
                        "type": "run",
                        "run": run,
                        "kind": kind,
                        "status": status,
                        "base": base,
                        # Empty unless the run's data lives outside its run directory; the client then
                        # browses the run directory as it does for every other kind.
                        "data": _run_data_dir(name, base, kind, outputs) if base else "",
                        # A transform's chains, one destination each, so the panel offers every one.
                        "outputs": outputs,
                    }
                )
            ]
            if status in _TERMINAL_STATUS and (was is None or was[0] != status):
                out.append(_sse({"type": "status", "run": run, "kind": kind, "status": status}))
            return out

        while True:
            # A quiet experiment sends nothing for minutes, so silence cannot be told from a stream that
            # died, and one that dies is invisible: the page keeps showing the last thing it saw. A ping
            # gives the client something to miss, so it can reconnect on its own.
            if time.monotonic() - last_ping > _PING_EVERY:
                last_ping = time.monotonic()
                yield _sse({"type": "ping"})

            jobs = _all_jobs(name)  # read once: discovery, the console and the announce all ask of it
            latest = max(jobs, key=_job_created, default=None)
            runs = _discover_session_runs(name, jobs)
            if not runs and latest is None:
                if not idle_sent:
                    yield _sse({"type": "idle"})
                    idle_sent = True
                await asyncio.sleep(0.6)
                continue
            idle_sent = False

            # Console (raw text, tracebacks) follows the latest job only; a new job resets the tail. The
            # `job` event names the active run so the client can default its tab to it, it never wipes.
            # Re-sent whenever that job's status moves: the client gates its Stop control and its tab
            # keeping on this value, which a one-shot event would freeze at whatever it was on connect.
            if latest is not None and latest.get("log_path"):
                fresh_console = latest["log_path"] != console_key
                if fresh_console:
                    console_key = latest["log_path"]
                    cpath = Path(console_key)
                    cpos = _tail_start(cpath)
                    cbuf = ""
                    job_status = None
                status = _status_for(latest, _runtime_log(latest))
                if status != job_status:
                    job_status = status
                    yield _sse(
                        {
                            "type": "job",
                            "run": latest.get("run_name") or latest.get("kind") or "job",
                            "kind": _job_kind(latest),
                            "status": status,
                            # Only a new job's console starts empty. Clearing on a status change too
                            # would drop a crashed run's traceback the instant it turned red.
                            "console_reset": fresh_console,
                        }
                    )
            if cpath is not None:
                lines, cpos, cbuf = _tail_lines(cpath, cpos, cbuf)
                for line in lines:
                    stripped = line.lstrip()
                    if not stripped or stripped[0] == "#" or stripped.startswith("[konfai-mcp]"):
                        continue
                    # Startup chatter is hidden; everything else the framework says is shown, so a
                    # new warning is never swallowed by an allowlist nobody updated.
                    if stripped.startswith("[KonfAI]") and any(k in stripped.lower() for k in _KONFAI_ROUTINE):
                        continue
                    yield _sse({"type": "log", "line": line})

            # A workflow job names its own runtime log, so its run is known before a single line is
            # written: announce it and its tab exists from launch, through the warm-up and even if it dies
            # before its first iteration. Discovering that same log later lands on the same (run, kind),
            # so it updates that tab rather than adding one.
            #
            # An app job is NOT announced. It carries two names: its own (`app_MR`) and the one its log
            # directory takes (`ImpactSynth`): so anything announced for it would be a second tab under
            # the wrong name, and every metric would arrive in the other. Its run appears when it writes.
            if latest is not None and latest.get("runtime_log_path") and latest.get("run_name"):
                # Through the same freshness rule as discovery: two blocks announcing the same run from
                # two readings of one record made its tab flip between 'done' and 'running' every poll.
                latest_status = _status_for(latest, Path(latest["runtime_log_path"]))
                for frame in state_of(latest["run_name"], latest.get("kind") or "", latest_status):
                    yield frame

            # Every run of the experiment is followed as its own feed and kept: launching a prediction
            # adds a run, it never clears the training runs. A newly-seen log replays from 0 so its curves
            # rebuild on connect.
            for log, run_name, run_kind, status, base in runs:
                feed_key = str(log)
                if feed_key not in feeds:
                    feeds[feed_key] = {"run": run_name, "kind": run_kind, "path": log, "pos": 0, "buf": "", "step": 0}
                feed = feeds[feed_key]
                # The log path is the run's identity; the name discovered beside it can change under us
                # (a second run of the same app renames the first). Everything this feed emits carries the
                # name it was created with, so its metrics and its outcome land on the same tab.
                for frame in state_of(feed["run"], feed["kind"], status, base):
                    yield frame
                lines, feed["pos"], feed["buf"] = _tail_lines(feed["path"], feed["pos"], feed["buf"])
                for line in lines:
                    events, feed["step"] = _runtime_events(line, feed["run"], feed["kind"], feed["step"])
                    for event in events:
                        yield _sse(event)
            await asyncio.sleep(0.6)

    return StreamingResponse(gen(), media_type="text/event-stream")
