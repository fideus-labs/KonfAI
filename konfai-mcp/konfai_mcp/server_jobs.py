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

from __future__ import annotations

import importlib
import json
import multiprocessing
import os
import re
import signal
import threading
import time
import traceback
import uuid
from collections.abc import Callable
from contextlib import redirect_stderr, redirect_stdout, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

from .server_support import WorkspaceLayout
from .workflows import JOB_RETRY_TOOLS, JobKind


def _run_job(
    *,
    target: str,
    kwargs: dict[str, object],
    cwd: str,
    log_path: str,
) -> None:
    if os.name != "nt":
        # Become a new session / process-group leader so cancel_job can reap the WHOLE tree via killpg,
        # including the mp.spawn DDP grandchildren this job may create -- otherwise they orphan and keep
        # holding GPU memory after the middle process is signalled. Harmless if already a group leader.
        try:
            os.setsid()
        except OSError:
            pass
    os.chdir(cwd)
    # Line-buffered: each completed line reaches the OS immediately, so a SIGTERM (cancel_job's killpg,
    # or an external kill) cannot discard a traceback stuck in a block buffer.
    with (
        Path(log_path).open("a", encoding="utf-8", buffering=1) as handle,
        redirect_stdout(handle),
        redirect_stderr(handle),
    ):
        if os.name != "nt":
            # A SIGTERM before any exception (e.g. cancel during a setup hang) would leave only the header.
            # Record + flush, then re-raise the default handler so the exit code and cancel_job's reaper are unchanged.
            def _on_terminate(signum: int, _frame: object) -> None:
                try:
                    handle.write(f"\n[konfai-mcp] job process terminated by signal {signum}.\n")
                    handle.flush()
                except Exception:
                    pass
                signal.signal(signum, signal.SIG_DFL)
                os.kill(os.getpid(), signum)

            signal.signal(signal.SIGTERM, _on_terminate)
            # SIGUSR1 (KonfAI Studio's on-demand validation) is delivered group-wide but is meant only for
            # the training workers — ignore it in this wrapper so it never terminates the job. A spawned
            # worker starts fresh and the trainer installs its own SIGUSR1 handler; if training runs inline
            # here, that same handler overrides this SIG_IGN.
            signal.signal(signal.SIGUSR1, signal.SIG_IGN)
        try:
            module_name, function_name = target.split(":", 1)
            getattr(importlib.import_module(module_name), function_name)(**kwargs)
        except BaseException:  # pragma: no cover - child process failure reporting
            traceback.print_exc()
            # Flush before re-raising: a teardown signal may kill the process before the with-block closes.
            try:
                handle.flush()
                os.fsync(handle.fileno())
            except OSError:
                pass
            raise


def _proc_returncode(proc: object) -> int | None:
    if hasattr(proc, "is_alive"):
        return None if proc.is_alive() else getattr(proc, "exitcode", None)
    poll = getattr(proc, "poll", None)
    return None if poll is None else poll()


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # EPERM means the pid exists but belongs to another user: it IS alive.
        return True
    except OSError:
        return False
    return True


def _process_create_time(pid: int) -> float | None:
    """The process create-time (Unix ts), used to tell an orphaned job's process from a reused pid."""
    try:
        import psutil

        return float(psutil.Process(pid).create_time())
    except Exception:
        return None


def _pid_is_recovered_job(pid: int | None, create_time: float | None) -> bool:
    """True if `pid` is alive AND still the same process that launched the job (guards against pid reuse).

    After an MCP restart the subprocess handle is gone but the OS process may still be running; comparing
    the recorded create-time to the live one avoids signalling an unrelated process that reused the pid.
    """
    if pid is None or not _pid_alive(pid):
        return False
    try:
        import psutil

        proc = psutil.Process(pid)
        if proc.status() == psutil.STATUS_ZOMBIE:
            return False  # dead but not yet reaped: not a running job
        current: float | None = float(proc.create_time())
    except Exception:
        current = _process_create_time(pid)
    if create_time is None or current is None:
        return True  # no baseline / cannot introspect: fall back to liveness only
    return abs(current - create_time) < 1.0


_ERROR_LINE_RE = re.compile(r"^\s*(?:[\w.]+\.)?\w*(?:Error|Exception|Interrupt(?:edError)?)\b.*")


def _extract_error_excerpt(log_path: Path, max_scan_lines: int = 400) -> str | None:
    """Pull the last exception (header + message body) out of a job log so the payload states WHY it failed.

    KonfAI's error classes and multi-line RuntimeErrors (e.g. SimpleITK) put the actionable message on
    the lines AFTER the ``<Class>:`` header, so the following lines are kept until a blank line or a
    new traceback record -- the header alone would read ``DatasetManagerError:`` with no message.
    """
    from .server_support import read_text_tail

    lines = read_text_tail(log_path, max_lines=max_scan_lines).splitlines()
    match_index: int | None = None
    for index, line in enumerate(lines):
        stripped = line.strip()
        if _ERROR_LINE_RE.match(stripped) or "CUDA out of memory" in stripped:
            match_index = index
    if match_index is None:
        return None
    parts = [lines[match_index].strip()]
    for line in lines[match_index + 1 :]:
        stripped = line.strip()
        if not stripped or stripped.startswith("Traceback") or _ERROR_LINE_RE.match(stripped):
            break
        parts.append(stripped)
        if len(parts) >= 8:
            break
    excerpt = " ".join(parts)[:500]
    # A dead DataLoader worker hides the real cause behind multiprocessing ("Details are lost due to
    # multiprocessing"), so the top-level message is misleading and non-deterministic. Append an
    # actionable KonfAI hint instead of leaving the agent to chase a phantom dtype/conv error.
    lowered = excerpt.lower()
    if "dataloader worker" in lowered and "exited" in lowered:
        excerpt += (
            " [KonfAI hint] A DataLoader worker crashed, so the real error is hidden by multiprocessing "
            "(usually out-of-memory, or an exception inside a dataset transform). Re-run with "
            "'Dataset.num_workers: 0' to "
            "surface the real traceback; if it then trains, the crash was worker memory pressure — reduce "
            "Dataset.num_workers, batch_size, or patch_size."
        )
    if "already exists" in lowered:
        excerpt += (
            " [KonfAI hint] The run refused to overwrite existing outputs. Re-run the same tool with "
            "overwrite=True to replace them (train: deletes the existing Checkpoints/ and Statistics/ for "
            "this run name; evaluation: deletes the existing metric folder; prediction: writes into the "
            "existing Predictions folder WITHOUT clearing it, so delete it first if you need a clean "
            "output) — or launch a new run under a different train_name to keep both."
        )
    return excerpt


def _output_missing_or_empty(path: Path) -> bool:
    """True if a declared output was not produced: a missing path, an empty directory, or a 0-byte file."""
    try:
        if not path.exists():
            return True
        if path.is_dir():
            return not any(path.iterdir())
        return path.stat().st_size == 0
    except OSError:
        return False  # cannot assess -> do not flip a successful run to error on a transient FS issue


@dataclass
class Job:
    job_id: str
    session: str
    kind: JobKind
    command: list[str]
    cwd: Path
    log_path: Path
    config_path: Path
    created_at: float = field(default_factory=time.time)
    status: Literal["queued", "running", "done", "error", "killed"] = "queued"
    pid: int | None = None
    returncode: int | None = None
    started_at: float | None = None
    finished_at: float | None = None
    cancel_requested: bool = False
    error: str | None = None
    run_name: str | None = None
    devices: list[str] | None = None
    runtime_log_path: Path | None = None
    output_path: Path | None = None
    proc_create_time: float | None = None  # process create-time at launch, to detect pid reuse on recovery
    job_dir: Path | None = None
    manifest_path: Path | None = None
    recovered: bool = False
    proc: object | None = field(default=None, repr=False)


class JobRegistry:
    def __init__(self, active_states: set[str], workspace_layout: WorkspaceLayout | None = None) -> None:
        self.active_states = set(active_states)
        self.workspace_layout = workspace_layout
        self.lock = threading.RLock()
        self.jobs: dict[str, Job] = {}
        self._load_persisted_jobs()

    def _job_to_dict(self, job: Job) -> dict[str, Any]:
        return {
            "job_id": job.job_id,
            "session": job.session,
            "kind": job.kind,
            "command": job.command,
            "cwd": str(job.cwd),
            "log_path": str(job.log_path),
            "config_path": str(job.config_path),
            "created_at": job.created_at,
            "status": job.status,
            "pid": job.pid,
            "proc_create_time": job.proc_create_time,
            "returncode": job.returncode,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
            "cancel_requested": job.cancel_requested,
            "error": job.error,
            "run_name": job.run_name,
            "devices": job.devices,
            "runtime_log_path": str(job.runtime_log_path) if job.runtime_log_path is not None else None,
            "output_path": str(job.output_path) if job.output_path is not None else None,
            "job_dir": str(job.job_dir) if job.job_dir is not None else None,
            "manifest_path": str(job.manifest_path) if job.manifest_path is not None else None,
            "recovered": job.recovered,
        }

    def _job_from_dict(self, payload: dict[str, Any]) -> Job:
        return Job(
            job_id=str(payload["job_id"]),
            session=str(payload.get("session", "default")),
            kind=payload["kind"],
            command=[str(value) for value in payload.get("command", [])],
            cwd=Path(payload["cwd"]),
            log_path=Path(payload["log_path"]),
            config_path=Path(payload["config_path"]),
            created_at=float(payload.get("created_at", time.time())),
            status=payload.get("status", "queued"),
            pid=payload.get("pid"),
            proc_create_time=payload.get("proc_create_time"),
            returncode=payload.get("returncode"),
            started_at=payload.get("started_at"),
            finished_at=payload.get("finished_at"),
            cancel_requested=bool(payload.get("cancel_requested", False)),
            error=payload.get("error"),
            run_name=payload.get("run_name"),
            devices=payload.get("devices"),
            runtime_log_path=(
                Path(payload["runtime_log_path"]) if payload.get("runtime_log_path") is not None else None
            ),
            output_path=Path(payload["output_path"]) if payload.get("output_path") is not None else None,
            job_dir=Path(payload["job_dir"]) if payload.get("job_dir") is not None else None,
            manifest_path=(Path(payload["manifest_path"]) if payload.get("manifest_path") is not None else None),
            recovered=bool(payload.get("recovered", False)),
        )

    def _persist_job(self, job: Job) -> None:
        if self.workspace_layout is None:
            return
        job_dir = self.workspace_layout.job_dir(job.job_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        state_path = self.workspace_layout.job_state_path(job.job_id)
        # Atomic write: a crash mid-write must not leave a truncated job.json, because the recovery loop
        # json.loads every record at server start -- one half-written file would otherwise be fatal. Write
        # a sibling temp file (hidden, so it is not matched by the "*/job.json" recovery glob) and rename
        # it into place; a reader only ever sees the whole old or the whole new file.
        tmp_path = state_path.with_name(f".{state_path.name}.{uuid.uuid4().hex}.tmp")
        try:
            tmp_path.write_text(json.dumps(self._job_to_dict(job), indent=2, sort_keys=True), encoding="utf-8")
            os.replace(tmp_path, state_path)
        finally:
            # A failed write/replace must not leak the hidden temp file (it would accumulate every retry);
            # unlink is a no-op after a successful os.replace, which already consumed tmp_path.
            tmp_path.unlink(missing_ok=True)

    def _persist_manifest(
        self, job: Job, *, config_snapshots: dict[str, str], extra_manifest: dict[str, object] | None
    ) -> None:
        if self.workspace_layout is None:
            return
        manifest_path = self.workspace_layout.job_manifest_path(job.job_id)
        manifest: dict[str, object] = {
            "job_id": job.job_id,
            "session": job.session,
            "kind": job.kind,
            "run_name": job.run_name,
            "cwd": str(job.cwd),
            "config_path": str(job.config_path),
            "runtime_log_path": str(job.runtime_log_path) if job.runtime_log_path is not None else None,
            "command": job.command,
            "created_at": job.created_at,
            "config_snapshots": config_snapshots,
        }
        if extra_manifest:
            manifest.update(extra_manifest)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        job.manifest_path = manifest_path

    def _snapshot_configs(self, job: Job) -> dict[str, str]:
        if self.workspace_layout is None:
            return {}
        configs_dir = self.workspace_layout.job_configs_dir(job.job_id)
        configs_dir.mkdir(parents=True, exist_ok=True)
        copied: dict[str, str] = {}
        for filename in ("Config.yml", "Prediction.yml", "Evaluation.yml"):
            source = job.cwd / filename
            if not source.exists():
                continue
            target = configs_dir / filename
            target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
            copied[filename] = str(target)
        return copied

    def _load_persisted_jobs(self) -> None:
        if self.workspace_layout is None or not self.workspace_layout.root.exists():
            return
        jobs_root = self.workspace_layout.jobs_dir()
        if not jobs_root.exists():
            return
        for state_path in sorted(jobs_root.glob("*/job.json")):
            try:
                payload = json.loads(state_path.read_text(encoding="utf-8"))
                job = self._job_from_dict(payload)
            except (OSError, ValueError, KeyError, TypeError):
                # A truncated/corrupt record (e.g. a crash mid-write from an older build) must not be
                # fatal: this loop reads every job.json at server start, so one bad file would otherwise
                # block startup entirely. Skip it and keep recovering the rest.
                continue
            if job.job_dir is None:
                job.job_dir = self.workspace_layout.job_dir(job.job_id)
            if job.manifest_path is None:
                manifest_path = self.workspace_layout.job_manifest_path(job.job_id)
                if manifest_path.exists():
                    job.manifest_path = manifest_path
            if job.status in self.active_states:
                job.recovered = True
                job.proc = None
                if _pid_is_recovered_job(job.pid, job.proc_create_time):
                    # The child outlived the server and is STILL the same process: keep it "running" so it
                    # stays visible AND cancellable (signal()/refresh() fall back to the pid). Marking it
                    # "error" would leave a live GPU-holding orphan that cancel_job cannot reach.
                    job.status = "running"
                else:
                    job.status = "error"
                    job.finished_at = time.time()
                    recovered_error = (
                        "Recovered after MCP server restart; the original process is gone (it died with the "
                        "server, or its pid was reused)."
                    )
                    job.error = f"{job.error} {recovered_error}".strip() if job.error else recovered_error
            self.jobs[job.job_id] = job
            self._persist_job(job)

    def refresh(self, job: Job) -> None:
        with self.lock:
            if job.status not in self.active_states:
                return
            if job.proc is None:
                # Recovered orphan (subprocess handle lost across restart): no returncode is available, so
                # poll the pid. While the same process lives it stays "running"; once it is gone the run
                # finished with an unknown code, so infer the outcome from cancel intent + the log.
                if job.recovered and job.pid is not None:
                    if _pid_is_recovered_job(job.pid, job.proc_create_time):
                        return
                    job.finished_at = time.time()
                    if job.cancel_requested:
                        job.status = "killed"
                    else:
                        job.error = _extract_error_excerpt(job.log_path)
                        if job.error:
                            job.status = "error"
                        elif job.output_path is not None and _output_missing_or_empty(job.output_path):
                            # No returncode survives a restart, so a clean exit can't be told from a kill.
                            # A recovered orphan that produced no declared output was almost certainly
                            # OS-killed (SIGKILL/OOM) -- do not report a misleading "done".
                            job.status = "error"
                            job.error = (
                                f"The recovered job is gone and wrote no output to {job.output_path}. With no "
                                "returncode after a restart, a clean exit cannot be told from a kill (likely "
                                "SIGKILL/OOM) -- treat this run as failed and re-launch it."
                            )
                        else:
                            job.status = "done"
                    self._persist_job(job)
                return
            returncode = _proc_returncode(job.proc)
            if returncode is None:
                return
            job.returncode = returncode
            job.finished_at = time.time()
            if job.cancel_requested:
                job.status = "killed"
            elif returncode == 0:
                job.status = "done"
                # A job that declared an output but exited 0 without producing it did NOT succeed:
                # never report a misleading "done" for an empty result. Generic — keyed only on whether
                # an output was declared (output_path), not on the job kind.
                if job.output_path is not None and _output_missing_or_empty(job.output_path):
                    job.status = "error"
                    job.error = (
                        f"The job exited 0 but wrote no output to {job.output_path}. It reported success "
                        "without producing a result -- the run likely did nothing (e.g. an app that silently "
                        "no-ops on incompatible input, or an output path that was not honoured). Read the job "
                        "log and re-check the inputs before trusting this run."
                    )
            else:
                job.status = "error"
                if job.error is None:
                    # Surface the failure reason in the payload itself; the traceback is in the log,
                    # but a tool-only client should not need a resource read to learn WHY it failed.
                    job.error = _extract_error_excerpt(job.log_path) or (
                        f"Subprocess exited with returncode {returncode}; read the job log for the traceback."
                    )
            self._persist_job(job)

    def payload(self, job: Job, isoformat: Callable[[float | None], str | None]) -> dict[str, Any]:
        self.refresh(job)
        retry_tool = JOB_RETRY_TOOLS.get(job.kind, f"run_{job.kind}")
        # next_actions holds callable tool names only; URIs live in next_resources / resources.
        next_actions = ["get_job_status"]
        next_resources = [f"job://{job.job_id}/log"]
        if job.status in self.active_states:
            next_actions.extend(["wait_for_job", "read_live_metrics", "cancel_job"])
        elif job.status == "done":
            next_actions.extend(["summarize_session", "leaderboard"])
        else:
            next_actions.append("read_job_log")
            next_actions.extend(["validate_config_semantics", retry_tool])
        return {
            "job_id": job.job_id,
            "session": job.session,
            "kind": job.kind,
            "status": job.status,
            "pid": job.pid,
            "returncode": job.returncode,
            "created_at": isoformat(job.created_at),
            "started_at": isoformat(job.started_at),
            "finished_at": isoformat(job.finished_at),
            "cwd": str(job.cwd),
            "config_path": str(job.config_path),
            "log_path": str(job.log_path),
            "runtime_log_path": str(job.runtime_log_path) if job.runtime_log_path is not None else None,
            "run_name": job.run_name,
            "devices": job.devices,
            "command": job.command,
            "error": job.error,
            "recovered": job.recovered,
            # True from the moment cancel_job is called; while the job is still active it signals
            # cancellation-in-progress. A finished job is 'killed' iff this is True — external kills
            # (OOM killer, manual signal) surface as status='error'.
            "cancel_requested": job.cancel_requested,
            "next_actions": next_actions,
            "next_resources": next_resources,
            "resources": {
                "status": f"job://{job.job_id}/status",
                "log": f"job://{job.job_id}/log",
                "manifest": f"job://{job.job_id}/manifest",
            },
            "manifest_path": str(job.manifest_path) if job.manifest_path is not None else None,
        }

    def get(self, job_id: str) -> Job:
        with self.lock:
            if job_id not in self.jobs:
                raise ValueError(f"Unknown job id: {job_id}")
            job = self.jobs[job_id]
            self.refresh(job)
            return job

    def latest(self, kind: str | None = None) -> Job | None:
        with self.lock:
            jobs = list(self.jobs.values())
            if kind is not None:
                jobs = [job for job in jobs if job.kind == kind]
            if not jobs:
                return None
            latest = max(jobs, key=lambda job: job.created_at)
            self.refresh(latest)
            return latest

    def active(self) -> list[Job]:
        with self.lock:
            jobs = list(self.jobs.values())
            for job in jobs:
                self.refresh(job)
            return [job for job in jobs if job.status in self.active_states]

    @staticmethod
    def _devices_conflict(a: list[str] | None, b: list[str] | None) -> bool:
        """Unknown device sets (None) conflict with everything; otherwise conflict = overlap."""
        if a is None or b is None:
            return True
        return bool(set(a) & set(b))

    def find_device_conflicts(self, devices: list[str] | None) -> list[Job]:
        """Return the active jobs whose device set overlaps ``devices``."""
        return [job for job in self.active() if self._devices_conflict(devices, job.devices)]

    def ensure_no_active_job(self) -> None:
        """Raise when the current session workspace already has queued or running jobs."""
        jobs = self.active()
        if jobs:
            running = ", ".join(f"{job.kind}:{job.job_id}" for job in jobs)
            session_name = self.workspace_layout.current_session if self.workspace_layout else "current"
            raise RuntimeError(f"Session '{session_name}' already has active job(s): {running}")

    def launch(
        self,
        *,
        session: str,
        kind: JobKind,
        command: list[str],
        cwd: Path,
        log_path: Path,
        config_path: Path,
        run_name: str | None = None,
        devices: list[str] | None = None,
        runtime_log_path: Path | None = None,
        extra_manifest: dict[str, object] | None = None,
        target: str | None = None,
        kwargs: dict[str, object] | None = None,
    ) -> Job:
        if target is None:
            raise ValueError("Job launch requires a Python target.")
        job = Job(
            job_id=uuid.uuid4().hex[:12],
            session=session,
            kind=kind,
            command=command,
            cwd=cwd,
            log_path=log_path,
            config_path=config_path,
            run_name=run_name,
            devices=devices,
            runtime_log_path=runtime_log_path,
        )
        # Generic: any job that declares an explicit output location (its runner kwargs carry "output")
        # gets it recorded, so refresh() can verify the output actually materialised. No per-kind logic.
        if kwargs is not None and kwargs.get("output"):
            job.output_path = Path(str(kwargs["output"]))
        if self.workspace_layout is not None:
            job.job_dir = self.workspace_layout.job_dir(job.job_id)
            job.job_dir.mkdir(parents=True, exist_ok=True)
        with self.lock:
            # Concurrency is bounded per device: jobs on disjoint devices may run in parallel;
            # overlapping (or unknown) device sets are refused.
            conflicting = self.find_device_conflicts(devices)
            if conflicting:
                active = ", ".join(
                    f"{existing.kind}:{existing.job_id} on {existing.devices or 'unknown devices'}"
                    for existing in conflicting
                )
                raise RuntimeError(
                    f"Session '{session}' already has active job(s) on conflicting device(s): {active}. "
                    "Pick disjoint gpu ids, or wait_for_job / cancel_job first."
                )
            self.jobs[job.job_id] = job
            self._persist_job(job)

        try:
            # Snapshot + manifest are inside the try so a failure here (e.g. disk full) marks the job
            # terminal instead of leaving it stuck "queued" forever -- a queued job counts as active, so
            # it would otherwise block every future launch on its device with no way to clear it.
            config_snapshots = self._snapshot_configs(job)
            self._persist_manifest(job, config_snapshots=config_snapshots, extra_manifest=extra_manifest)
            with job.log_path.open("a", encoding="utf-8") as handle:
                handle.write(f"# KonfAI MCP job {job.job_id}\n")
                handle.write(f"# kind: {job.kind}\n")
                handle.write(f"# session: {job.session}\n")
                handle.write(f"# cwd: {job.cwd}\n")
                if job.run_name is not None:
                    handle.write(f"# run_name: {job.run_name}\n")
                if job.runtime_log_path is not None:
                    handle.write(f"# runtime_log_path: {job.runtime_log_path}\n")
                handle.write(f"# command: {' '.join(job.command)}\n\n")
                handle.flush()
                # Use a "spawn" context so the job runs in a fresh interpreter: training
                # may initialise CUDA, which is unsafe in a forked child when the server
                # process has already touched CUDA (PyTorch requires spawn for this).
                proc = multiprocessing.get_context("spawn").Process(
                    target=_run_job,
                    kwargs={
                        "target": target,
                        "kwargs": dict(kwargs or {}),
                        "cwd": str(cwd),
                        "log_path": str(job.log_path),
                    },
                    daemon=False,
                )
                proc.start()
        except Exception as exc:
            job.error = str(exc)
            job.status = "error"
            job.finished_at = time.time()
            self._persist_job(job)
            raise RuntimeError(f"Failed to launch {kind} job for session '{session}': {exc}") from exc

        with self.lock:
            job.proc = proc
            job.pid = proc.pid
            # Record the process create-time so that, after an MCP restart, an orphaned job's live process
            # can be told apart from an unrelated process that happens to have reused the pid.
            job.proc_create_time = _process_create_time(proc.pid) if proc.pid is not None else None
            job.started_at = time.time()
            job.status = "running"
            self._persist_job(job)
        return job

    def signal(self, job: Job, sig: int) -> None:
        proc = job.proc
        proc_alive = proc is not None and hasattr(proc, "is_alive") and proc.is_alive()
        # A recovered orphan has no proc handle but its OS process may still be alive and killable by pid.
        orphan_alive = proc is None and _pid_is_recovered_job(job.pid, job.proc_create_time)
        if not proc_alive and not orphan_alive:
            self.refresh(job)
            return
        pid = job.pid
        if os.name != "nt" and pid is not None:
            # The child ran os.setsid(), so its pid IS its process-group id: signal the whole group to
            # reap the mp.spawn DDP grandchildren, not just the middle process. killpg(pid) can only hit
            # that group -- before setsid runs (a launch/cancel race) pid is not a pgid and this raises,
            # so it can never signal the server's own group; we then fall back below.
            try:
                os.killpg(pid, sig)
                return
            except (ProcessLookupError, PermissionError, OSError):
                pass
        if proc is not None:
            try:
                if sig == signal.SIGTERM:
                    cast(multiprocessing.Process, proc).terminate()
                else:
                    cast(multiprocessing.Process, proc).kill()
            except ProcessLookupError:
                self.refresh(job)
        elif pid is not None:
            # Recovered orphan with no proc handle and no process group (e.g. a pre-setsid job): signal the
            # pid directly. The identity was already checked by _pid_is_recovered_job above.
            try:
                os.kill(pid, sig)
            except OSError:
                self.refresh(job)

    def notify(self, job: Job, sig: int) -> bool:
        """Best-effort delivery of a **non-terminating** signal (e.g. SIGUSR1) to a job's process group.

        Unlike ``signal()`` it never falls back to killing the process, and it verifies the job is genuinely
        alive (its own proc handle, or a recovered orphan's identity) before signalling — so a reused pid
        can never receive it (SIGUSR1's default action is Terminate). POSIX only; returns whether it landed.
        """
        proc = job.proc
        proc_alive = proc is not None and hasattr(proc, "is_alive") and proc.is_alive()
        orphan_alive = proc is None and _pid_is_recovered_job(job.pid, job.proc_create_time)
        if os.name == "nt" or job.pid is None or not (proc_alive or orphan_alive):
            return False
        with suppress(ProcessLookupError, PermissionError, OSError):
            os.killpg(job.pid, sig)
            return True
        return False

    def cancel(
        self, job_id: str, isoformat: Callable[[float | None], str | None], wait_s: float = 5.0
    ) -> dict[str, Any]:
        job = self.get(job_id)
        if job.status not in self.active_states:
            return self.payload(job, isoformat)

        job.cancel_requested = True
        self.signal(job, signal.SIGTERM)
        deadline = time.time() + max(wait_s, 0.0)
        while time.time() < deadline:
            self.refresh(job)
            if job.status not in self.active_states:
                return self.payload(job, isoformat)
            time.sleep(0.1)

        self.signal(job, signal.SIGKILL)
        time.sleep(0.1)
        self.refresh(job)
        payload = self.payload(job, isoformat)
        self._persist_job(job)
        return payload
