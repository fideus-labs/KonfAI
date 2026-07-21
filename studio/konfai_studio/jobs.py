# SPDX-License-Identifier: Apache-2.0
"""Job discovery and the live feed: reading konfai-mcp job records, tailing a run's runtime/console
logs, turning konfai's runtime lines into structured SSE metric/progress events, and the persistent
``/api/live`` stream."""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import AsyncIterator
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
from konfai_mcp.live_parse import parse_host_stats, parse_live_metric_line, parse_live_progress, progress_label

from .paths import _sane_session, _workspace_root

router = APIRouter()

_TERMINAL_STATUS = {"done", "error", "killed", "cancelled"}
_LOG_BACKFILL = 32_000  # bytes: on connect, replay only the recent tail of a large log, not its full history
_HOST_KEYS = ("memory_gb", "memory_percent", "memory_gpu_gb", "memory_gpu_percent", "cpu_percent")
_MTIME_LIVE_WINDOW = 8.0  # a run log written this recently reads as live when no job record claims it
_RUN_ROOT_KIND = {
    "Statistics": "train",
    "Predictions": "prediction",
    "Evaluations": "evaluation",
    "Uncertainties": "uncertainty",
}


def _sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event)}\n\n"


def _all_jobs(session: str) -> list[dict[str, Any]]:
    jobs_dir = _workspace_root() / "sessions" / session / ".konfai_mcp" / "jobs"
    out: list[dict[str, Any]] = []
    if jobs_dir.is_dir():
        for record in jobs_dir.glob("*/job.json"):
            try:
                out.append(json.loads(record.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
    return out


def _job_created(job: dict[str, Any]) -> float:
    """A job's creation time as an epoch float, from its recorded ``created_at`` (a float epoch or an ISO
    string, depending on which konfai-mcp wrote it). Used to order jobs by when they *started*, not by
    file mtime — a terminal job's json can be rewritten later (status monitor), which mtime would misread
    as 'newest' and make the feed follow a dead run over a fresh one."""
    created = job.get("created_at")
    if isinstance(created, (int, float)):
        return float(created)
    if isinstance(created, str):
        with suppress(ValueError):
            return datetime.fromisoformat(created).timestamp()
    return 0.0


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


def _live_status(job: dict[str, Any]) -> str:
    """The job's status, but a still-'running' record whose process is gone reads as 'failed' — the MCP
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
    """The stage a tqdm phase belongs to, from its label — so the client never guesses. 'Caching Train' →
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
    job_kind = str(job.get("kind") or "")
    found: list[tuple[Path, str, str]] = []
    seen: set[Path] = set()
    for pattern in (
        "*/log_0.txt",
        "Statistics/*/log_0.txt",
        "Predictions/*/log_0.txt",
        "Evaluations/*/log_0.txt",
        "Uncertainties/*/log_0.txt",
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
            # learning rate is a training-only signal — never chart it under validation
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


def _discover_session_runs(session: str) -> list[tuple[Path, str, str, str, str]]:
    """Every run of the experiment as (log_path, run_name, kind, status, base) — one per runtime log.

    Workflow jobs write under the session roots (Statistics/Predictions/Evaluations); app jobs (infer /
    fine-tune / pipeline / uncertainty) write under an isolated ``output_path`` subtree, so EVERY app job's
    output_path is scanned (not only the newest), else an earlier isolated run vanishes the moment a newer
    job exists. ``base`` is the run dir relative to the session root — "Statistics/<run>" for a session-root
    run, "<app_output>-<hash>/Statistics/<run>" for an isolated app run — the one datum that lets the
    previews / TensorBoard / Browse helpers resolve the real on-disk location. Status comes from the newest
    job that names the log (workflow) or owns the output_path (app), else from how recently the log was
    written. Newest first."""
    base_root = _workspace_root() / "sessions" / session
    # A run dir is often written by several jobs (re-runs share one log_0.txt); its status must come from the
    # NEWEST job that names it (workflow) or owns its output subtree (app), so a fresh run wins over an old one.
    jobs = _all_jobs(session)
    status_by_log: dict[str, str] = {}
    newest_by_log: dict[str, float] = {}
    app_status_by_output: dict[str, tuple[float, str]] = {}
    for job in jobs:
        created = _job_created(job)
        runtime = job.get("runtime_log_path")
        if runtime:
            key = str(Path(runtime))
            if created >= newest_by_log.get(key, float("-inf")):
                newest_by_log[key] = created
                status_by_log[key] = _live_status(job)
        elif job.get("output_path"):
            out = str(Path(job["output_path"]))
            if created >= app_status_by_output.get(out, (float("-inf"), ""))[0]:
                app_status_by_output[out] = (created, _live_status(job))
    found: list[tuple[Path, str, str, str, str, float]] = []
    seen: set[Path] = set()

    def add(log: Path, run_name: str, kind: str, app_status: str | None) -> None:
        if log in seen or not log.is_file():
            return
        seen.add(log)
        status = status_by_log.get(str(log)) or app_status
        if status is None:
            status = "running" if (time.time() - log.stat().st_mtime) < _MTIME_LIVE_WINDOW else "done"
        try:
            base = str(log.parent.relative_to(base_root))
        except ValueError:
            base = log.parent.name
        found.append((log, run_name, kind, status, base, log.stat().st_mtime))

    for root, kind in (("Statistics", "train"), ("Predictions", "prediction"), ("Evaluations", "evaluation")):
        directory = base_root / root
        if directory.is_dir():
            for log in directory.glob("*/log_0.txt"):
                add(log, log.parent.name, kind, None)
    for job in jobs:  # app runs land outside the three roots, under each app job's own output_path
        out = job.get("output_path")
        if not out:
            continue
        entry = app_status_by_output.get(str(Path(out)))
        for log, run_name, kind in _discover_run_logs(job):
            add(log, run_name, kind, entry[1] if entry else None)
    found.sort(key=lambda row: row[5], reverse=True)
    return [(log, run_name, kind, status, base) for log, run_name, kind, status, base, _ in found]


@router.get("/api/live")
async def live(session: str = Query("default")) -> StreamingResponse:
    """Tail a task's most recent job in real time (SSE): the console log as raw text, and konfai's runtime
    log as **structured** metrics + progress — parsed by konfai-mcp's own ``live_parse`` (one source of
    truth, no re-implementation here).

    Two logs, two roles. The console wrapper log (header, warm-up prints, crash tracebacks) streams as
    ``log`` lines. konfai writes its per-iteration training tqdm to the runtime file, not stdout, so that
    file streams as ``metric`` events (a stage + flat metric values + the ``progress`` bar) and, for the
    metric-less data-caching phase, ``progress`` events — never as raw log lines, which would bury the
    console tail under thousands of tqdm frames.

    The connection is **persistent**: a job finishing is announced once (terminal status) but the stream
    keeps watching, so the next job the user launches is picked up on the same connection.
    """
    name = _sane_session(session)

    async def gen() -> AsyncIterator[str]:
        console_key: str | None = None  # which job's console log is being followed
        cpath: Path | None = None
        cpos = 0
        cbuf = ""
        feeds: dict[str, dict[str, Any]] = {}  # log-path -> {run, kind, path, pos, buf, step}: one per run, kept
        announced: dict[str, str] = {}  # run key -> terminal status already emitted (so it fires once)
        idle_sent = False

        while True:
            latest = _latest_job(name)
            runs = _discover_session_runs(name)
            if not runs and latest is None:
                if not idle_sent:
                    yield _sse({"type": "idle"})
                    idle_sent = True
                await asyncio.sleep(0.6)
                continue
            idle_sent = False

            # Console (raw text, tracebacks) follows the latest job only; a new job resets the tail. The
            # `job` event names the active run so the client can default its tab to it — it never wipes.
            if latest is not None and latest.get("log_path") and latest["log_path"] != console_key:
                console_key = latest["log_path"]
                cpath = Path(console_key)
                cpos = _tail_start(cpath)
                cbuf = ""
                yield _sse(
                    {
                        "type": "job",
                        "run": latest.get("run_name") or latest.get("kind") or "job",
                        "kind": latest.get("kind") or "",
                        "status": _live_status(latest),
                    }
                )
            if cpath is not None:
                lines, cpos, cbuf = _tail_lines(cpath, cpos, cbuf)
                for line in lines:
                    stripped = line.lstrip()
                    if not stripped or stripped[0] == "#" or stripped.startswith("[konfai-mcp]"):
                        continue
                    # [KonfAI] lines are framework chatter. Surface only genuine OOM/memory-pressure notices
                    # (the auto-patch recovery prints those on retry; hiding them makes a churning run look
                    # silent) -- not the routine memory-budget / cache-plan line, which is a benign startup
                    # decision, not an alert.
                    if stripped.startswith("[KonfAI]") and not any(
                        k in stripped.lower() for k in ("out of memory", "oom", "re-plan", "replan", "retry", "reduc")
                    ):
                        continue
                    yield _sse({"type": "log", "line": line})

            # Every run of the experiment is followed as its own feed and kept — launching a prediction
            # adds a run, it never clears the training runs. A newly-seen log replays from 0 so its curves
            # rebuild on connect.
            for log, run_name, run_kind, status, base in runs:
                feed_key = str(log)
                if feed_key not in feeds:
                    feeds[feed_key] = {"run": run_name, "kind": run_kind, "path": log, "pos": 0, "buf": "", "step": 0}
                    yield _sse({"type": "run", "run": run_name, "kind": run_kind, "status": status, "base": base})
                feed = feeds[feed_key]
                lines, feed["pos"], feed["buf"] = _tail_lines(feed["path"], feed["pos"], feed["buf"])
                for line in lines:
                    events, feed["step"] = _runtime_events(line, feed["run"], feed["kind"], feed["step"])
                    for event in events:
                        yield _sse(event)
                run_key = f"{run_kind}:{run_name}"
                if status in _TERMINAL_STATUS and announced.get(run_key) != status:
                    announced[run_key] = status
                    yield _sse({"type": "status", "run": run_name, "kind": run_kind, "status": status})
            await asyncio.sleep(0.6)

    return StreamingResponse(gen(), media_type="text/event-stream")
