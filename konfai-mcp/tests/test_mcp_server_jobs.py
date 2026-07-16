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

import importlib
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, cast

import pytest

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))
TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))  # so the spawn child can import the mcp_test_helpers job target

from konfai_mcp.server_jobs import Job, JobRegistry, _extract_error_excerpt  # noqa: E402
from konfai_mcp.server_support import WorkspaceLayout  # noqa: E402


def _pid_alive(pid: int) -> bool:
    # A zombie (killed but not yet reaped) is NOT running; treat it as dead.
    try:
        import psutil

        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except Exception:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


def _wait_pid_dead(pid: int, timeout: float) -> bool:
    # Poll instead of a fixed sleep: process teardown timing varies under load (the full suite).
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _pid_alive(pid):
            return True
        time.sleep(0.05)
    return not _pid_alive(pid)


@pytest.mark.skipif(os.name == "nt", reason="orphan pid signalling is POSIX-only")
def test_recovered_orphan_after_restart_stays_running_and_is_cancellable(tmp_path: Path) -> None:
    # A job whose process outlives an MCP restart must NOT be mislabeled "error": it stays "running" (its
    # live pid is verified against the recorded create-time to rule out pid reuse) so the agent can still
    # cancel it. Cancel then reaps the whole group (middle + grandchild) even without the proc handle.
    # Resolve JobRegistry from the live module: another test's load_mcp_server reloads server_jobs, so the
    # module-level import would launch with a stale _run_job that spawn can no longer pickle.
    job_registry_cls = importlib.import_module("konfai_mcp.server_jobs").JobRegistry
    layout = WorkspaceLayout(tmp_path)
    layout.ensure_session_workspace()
    registry = job_registry_cls({"queued", "running"}, workspace_layout=layout)
    pid_file = tmp_path / "pids.txt"

    job = registry.launch(
        session=layout.current_session or "default",
        kind="train",
        command=["x"],
        cwd=tmp_path,
        log_path=tmp_path / "log.txt",
        config_path=tmp_path / "cfg.ref",
        target="mcp_test_helpers:spawn_grandchild_and_idle",
        kwargs={"pid_file": str(pid_file)},
    )

    middle_pid = grandchild_pid = None
    try:
        for _ in range(200):
            if pid_file.exists() and len(pid_file.read_text().split()) >= 2:
                break
            time.sleep(0.05)
        assert pid_file.exists(), "job target never started"
        middle_pid, grandchild_pid = (int(value) for value in pid_file.read_text().split())

        # Simulate an MCP restart: a fresh registry loading the persisted jobs from the same workspace.
        restarted = job_registry_cls({"queued", "running"}, workspace_layout=WorkspaceLayout(tmp_path))
        recovered = restarted.get(job.job_id)
        assert recovered.status == "running"  # not force-marked "error"
        assert recovered.recovered is True
        assert recovered.proc is None

        restarted.cancel(job.job_id, lambda value: None, wait_s=3.0)
        assert _wait_pid_dead(middle_pid, 5.0)
        assert _wait_pid_dead(grandchild_pid, 5.0)
        recovered_after = restarted.get(job.job_id)
        restarted.refresh(recovered_after)
        assert recovered_after.status == "killed"
    finally:
        for pid in (middle_pid, grandchild_pid):
            if pid is not None:
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass


@pytest.mark.skipif(os.name == "nt", reason="process-group reaping (setsid/killpg) is POSIX-only")
def test_cancel_reaps_the_whole_process_group_including_grandchildren(tmp_path: Path) -> None:
    # A job may fan out mp.spawn DDP grandchildren; cancel must reap the WHOLE process group, not just
    # the middle process, or the grandchildren orphan and keep holding GPU memory. The job here spawns a
    # grandchild (as DDP does); after cancel both the middle process and the grandchild must be dead.
    # Resolve JobRegistry from the live module (another test reloads server_jobs), so launch pickles the
    # current _run_job rather than a stale one.
    job_registry_cls = importlib.import_module("konfai_mcp.server_jobs").JobRegistry
    layout = WorkspaceLayout(tmp_path)
    layout.ensure_session_workspace()
    registry = job_registry_cls({"queued", "running"}, workspace_layout=layout)
    pid_file = tmp_path / "pids.txt"

    job = registry.launch(
        session=layout.current_session or "default",
        kind="train",
        command=["x"],
        cwd=tmp_path,
        log_path=tmp_path / "log.txt",
        config_path=tmp_path / "cfg.ref",
        target="mcp_test_helpers:spawn_grandchild_and_idle",
        kwargs={"pid_file": str(pid_file)},
    )

    middle_pid = grandchild_pid = None
    try:
        for _ in range(200):
            if pid_file.exists() and len(pid_file.read_text().split()) >= 2:
                break
            time.sleep(0.05)
        assert pid_file.exists(), "job target never started"
        middle_pid, grandchild_pid = (int(value) for value in pid_file.read_text().split())
        assert _pid_alive(middle_pid) and _pid_alive(grandchild_pid)

        registry.cancel(job.job_id, lambda value: None, wait_s=3.0)
        assert _wait_pid_dead(middle_pid, 5.0), "middle process survived cancel"
        assert _wait_pid_dead(grandchild_pid, 5.0), "mp.spawn grandchild orphaned (process group not reaped)"
    finally:
        for pid in (middle_pid, grandchild_pid):
            if pid is not None:
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass


def test_dataloader_worker_death_gets_actionable_hint(tmp_path: Path) -> None:
    # A dead DataLoader worker masks the real cause behind multiprocessing; the excerpt must add a
    # KonfAI hint pointing at num_workers/use_cache instead of surfacing only the misleading message.
    worker_log = tmp_path / "worker.log"
    worker_log.write_text(
        "RuntimeError: DataLoader worker (pid 51545) exited unexpectedly with exit code 1. "
        "Details are lost due to multiprocessing. Rerunning with num_workers=0 may give better error trace.\n",
        encoding="utf-8",
    )
    excerpt = _extract_error_excerpt(worker_log)
    assert excerpt is not None
    assert "[KonfAI hint]" in excerpt
    assert "num_workers: 0" in excerpt

    # A normal exception must NOT get the worker hint.
    normal_log = tmp_path / "normal.log"
    normal_log.write_text("ValueError: bad config value 'x'\n", encoding="utf-8")
    normal = _extract_error_excerpt(normal_log)
    assert normal is not None
    assert "[KonfAI hint]" not in normal


def test_overwrite_refusal_gets_actionable_hint(tmp_path: Path) -> None:
    # Re-running a train under an existing run name fails on KonfAI's overwrite guard; the excerpt must
    # tell the agent to re-run with overwrite=True (or pick a new name) instead of just echoing the error.
    exists_log = tmp_path / "exists.log"
    exists_log.write_text(
        "konfai.utils.errors.TrainerError: \n"
        "[Trainer] The model '/ws/Checkpoints/run_a' already exists.\n"
        "→\tPass -y/--overwrite to replace it, or remove the existing outputs manually.\n",
        encoding="utf-8",
    )
    excerpt = _extract_error_excerpt(exists_log)
    assert excerpt is not None
    assert "[KonfAI hint]" in excerpt
    assert "overwrite=True" in excerpt


def test_error_excerpt_keeps_the_message_body_after_the_header(tmp_path: Path) -> None:
    # KonfAI error classes print '\n[Type] message' AFTER the '<Class>:' header line, so keeping only
    # the matching line surfaced a bare 'DatasetManagerError:' with no message.
    log = tmp_path / "konfai.log"
    log.write_text(
        "Traceback (most recent call last):\n"
        '  File "x.py", line 1, in <module>\n'
        "konfai.utils.errors.DatasetManagerError: \n"
        "[DatasetManager] Group source 'CT' not found in any dataset.\n"
        "→\tDataset filenames provided: ['/data:a:mha']\n"
        "→\tAvailable groups: ['MR']\n",
        encoding="utf-8",
    )
    excerpt = _extract_error_excerpt(log)
    assert excerpt is not None
    assert "Group source 'CT' not found" in excerpt
    assert "Available groups" in excerpt

    # Multi-line RuntimeErrors (e.g. SimpleITK) keep the lines carrying the actual reason too.
    sitk_log = tmp_path / "sitk.log"
    sitk_log.write_text(
        "RuntimeError: Exception thrown in SimpleITK ImageFileReader_Execute:\n"
        'sitk::ERROR: Unable to determine ImageIO reader for "./Dataset/CASE_000/CT.mha"\n',
        encoding="utf-8",
    )
    sitk_excerpt = _extract_error_excerpt(sitk_log)
    assert sitk_excerpt is not None
    assert "Unable to determine ImageIO reader" in sitk_excerpt


class _ProcDone:
    def __init__(self, returncode: int) -> None:
        self._returncode = returncode
        self.pid = 123

    def poll(self) -> int:
        return self._returncode


def _isoformat(value: float | None) -> str | None:
    return None if value is None else f"ts:{value}"


def test_job_registry_refresh_and_payload() -> None:
    registry = JobRegistry({"queued", "running"})
    job = Job(
        job_id="job1",
        session="default",
        kind="train",
        command=["echo", "ok"],
        cwd=Path("/tmp/workspace"),
        log_path=Path("/tmp/job.log"),
        config_path=Path("/tmp/Config.yml"),
        status="running",
    )
    job.proc = cast(Any, _ProcDone(0))
    registry.jobs[job.job_id] = job

    payload = registry.payload(job, _isoformat)

    assert payload["status"] == "done"
    assert "summarize_session" in payload["next_actions"]
    assert payload["resources"]["status"] == "job://job1/status"


def test_declared_output_but_empty_result_becomes_error(tmp_path: Path) -> None:
    # Generic (no per-kind logic): a job that declared an output and exited 0 without producing it must
    # NOT report a misleading "done". A job with no declared output (output_path=None) is untouched.
    registry = JobRegistry({"queued", "running"})

    def _job(name: str, kind: str, output_path: Path | None) -> Job:
        job = Job(
            job_id=name,
            session="default",
            kind=cast(Any, kind),
            command=["x"],
            cwd=tmp_path,
            log_path=tmp_path / f"{name}.log",
            config_path=tmp_path / "ref",
            status="running",
            output_path=output_path,
        )
        job.proc = cast(Any, _ProcDone(0))
        registry.jobs[name] = job
        return job

    empty_out = tmp_path / "empty"
    empty_out.mkdir()
    j_empty = _job("j_empty", "infer", empty_out)
    registry.refresh(j_empty)
    assert j_empty.status == "error"
    assert "wrote no output" in (j_empty.error or "")

    good_out = tmp_path / "good"
    good_out.mkdir()
    (good_out / "result.mha").write_text("x", encoding="utf-8")
    j_good = _job("j_good", "infer", good_out)
    registry.refresh(j_good)
    assert j_good.status == "done"

    # A workflow with no declared output (train writes to derived dirs) is never flipped.
    j_train = _job("j_train", "train", None)
    registry.refresh(j_train)
    assert j_train.status == "done"


def test_job_registry_rejects_active_session() -> None:
    layout = WorkspaceLayout(Path("/tmp/workspaces"))
    registry = JobRegistry({"queued", "running"}, workspace_layout=layout)
    job = Job(
        job_id="job2",
        session="default",
        kind="train",
        command=["echo"],
        cwd=Path("/tmp/workspace"),
        log_path=Path("/tmp/job.log"),
        config_path=Path("/tmp/Config.yml"),
        status="running",
    )
    registry.jobs[job.job_id] = job

    with pytest.raises(RuntimeError, match="already has active job"):
        registry.ensure_no_active_job()


def test_job_registry_latest_and_get_unknown() -> None:
    registry = JobRegistry({"queued", "running"})
    first = Job(
        job_id="job3",
        session="default",
        kind="train",
        command=["echo"],
        cwd=Path("/tmp/workspace"),
        log_path=Path("/tmp/job.log"),
        config_path=Path("/tmp/Config.yml"),
        created_at=1.0,
    )
    second = Job(
        job_id="job4",
        session="default",
        kind="prediction",
        command=["echo"],
        cwd=Path("/tmp/workspace"),
        log_path=Path("/tmp/job2.log"),
        config_path=Path("/tmp/Prediction.yml"),
        created_at=2.0,
    )
    registry.jobs[first.job_id] = first
    registry.jobs[second.job_id] = second

    latest = registry.latest()
    latest_train = registry.latest(kind="train")

    assert latest is not None
    assert latest_train is not None
    assert latest.job_id == "job4"
    assert latest_train.job_id == "job3"

    with pytest.raises(ValueError, match="Unknown job id"):
        registry.get("missing")


def test_job_registry_recovers_persisted_active_jobs(tmp_path: Path) -> None:
    layout = WorkspaceLayout(tmp_path)
    workspace = layout.ensure_session_workspace()
    layout.jobs_dir().mkdir(parents=True, exist_ok=True)
    job_dir = layout.job_dir("job5")
    job_dir.mkdir(parents=True)
    layout.job_state_path("job5").write_text(
        """
{
  "job_id": "job5",
  "session": "default",
  "kind": "train",
  "command": ["python", "-m", "konfai_mcp.runner", "TRAIN"],
  "cwd": "/tmp/demo",
  "log_path": "/tmp/demo.log",
  "config_path": "/tmp/Config.yml",
  "created_at": 1.0,
  "status": "running",
  "pid": 4321,
  "returncode": null,
  "started_at": 1.5,
  "finished_at": null,
  "cancel_requested": false,
  "error": null,
  "run_name": "RUN_01",
  "runtime_log_path": "/tmp/Statistics/RUN_01/log_0.txt",
  "job_dir": null,
  "manifest_path": null,
  "recovered": false
}
        """.strip(),
        encoding="utf-8",
    )

    assert workspace == tmp_path / "sessions" / "default"
    registry = JobRegistry({"queued", "running"}, workspace_layout=layout)

    payload = registry.payload(registry.get("job5"), _isoformat)

    assert payload["status"] == "error"
    assert payload["recovered"] is True
    assert "restart" in (payload["error"] or "")


def test_corrupt_job_record_does_not_block_server_start(tmp_path: Path) -> None:
    # A crash mid-write can leave a truncated job.json. The recovery loop reads every record at start,
    # so a single corrupt file must be skipped, not make JobRegistry construction fatal (dead server).
    layout = WorkspaceLayout(tmp_path)
    layout.ensure_session_workspace()
    layout.jobs_dir().mkdir(parents=True, exist_ok=True)

    good_dir = layout.job_dir("goodjob")
    good_dir.mkdir(parents=True)
    layout.job_state_path("goodjob").write_text(
        """
{
  "job_id": "goodjob",
  "session": "default",
  "kind": "train",
  "command": ["echo"],
  "cwd": "/tmp/demo",
  "log_path": "/tmp/demo.log",
  "config_path": "/tmp/Config.yml",
  "created_at": 1.0,
  "status": "done"
}
        """.strip(),
        encoding="utf-8",
    )
    bad_dir = layout.job_dir("badjob")
    bad_dir.mkdir(parents=True)
    layout.job_state_path("badjob").write_text('{"job_id": "badjob", "kind": "train"', encoding="utf-8")

    registry = JobRegistry({"queued", "running"}, workspace_layout=WorkspaceLayout(tmp_path))
    assert "goodjob" in registry.jobs  # the intact record still recovers
    assert "badjob" not in registry.jobs  # the truncated record is skipped, not fatal


def test_job_state_write_is_atomic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # _persist_job must write via a temp file + os.replace so a job.json is never observed half-written:
    # a crash at the rename must leave the previous record intact, never a truncated one.
    import json as _json

    layout = WorkspaceLayout(tmp_path)
    layout.ensure_session_workspace()
    registry = JobRegistry({"queued", "running"}, workspace_layout=layout)
    job = Job(
        job_id="atomicjob",
        session="default",
        kind="train",
        command=["echo"],
        cwd=tmp_path,
        log_path=tmp_path / "job.log",
        config_path=tmp_path / "Config.yml",
    )
    registry._persist_job(job)  # first record: status "queued"
    state_path = layout.job_state_path("atomicjob")
    assert state_path.is_file()
    assert _json.loads(state_path.read_text(encoding="utf-8"))["status"] == "queued"
    assert list(state_path.parent.glob("*.tmp")) == []  # temp renamed away, none left behind
    original = state_path.read_text(encoding="utf-8")

    # Crash at the atomic rename while persisting the next state: a bare write_text would truncate the
    # real file here; the temp-file + os.replace design leaves the prior record byte-for-byte intact.
    replace_calls: list[tuple[str, str]] = []

    def failing_replace(src: Any, dst: Any) -> None:
        replace_calls.append((str(src), str(dst)))
        raise OSError("crash at rename")

    monkeypatch.setattr(os, "replace", failing_replace)
    job.status = "done"
    with pytest.raises(OSError, match="crash at rename"):
        registry._persist_job(job)

    assert replace_calls, "persist must go through os.replace (temp file + atomic rename), not a bare write"
    assert state_path.read_text(encoding="utf-8") == original  # reader never sees a half-written "done"


def test_manifest_failure_marks_job_terminal_not_stuck_queued(tmp_path: Path) -> None:
    # A snapshot/manifest failure during launch must mark the job terminal (error), not leave it "queued"
    # forever -- a queued job counts as active and would block every future launch on its device.
    layout = WorkspaceLayout(tmp_path)
    layout.ensure_session_workspace()
    registry = JobRegistry({"queued", "running"}, workspace_layout=layout)

    def _boom(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("disk full while writing manifest")

    registry._persist_manifest = _boom  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="Failed to launch"):
        registry.launch(
            session="default",
            kind="train",
            command=["x"],
            cwd=tmp_path,
            log_path=tmp_path / "log.txt",
            config_path=tmp_path / "cfg.ref",
            target="nonexistent:fn",
            kwargs={},
        )

    assert registry.active() == []  # not stuck active
    registry.ensure_no_active_job()  # does not raise: the device is free again
    statuses = {job.status for job in registry.jobs.values()}
    assert statuses == {"error"}
