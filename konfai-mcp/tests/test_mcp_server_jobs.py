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

import asyncio
import importlib
import json
import multiprocessing
import os
import signal
import sys
import time
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import fastmcp
import pytest

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))
TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))  # so the spawn child can import the mcp_test_helpers job target

import mcp_test_helpers  # noqa: E402
from konfai_mcp.server_jobs import Job, JobRegistry, _extract_error_excerpt  # noqa: E402
from konfai_mcp.server_support import WorkspaceLayout  # noqa: E402
from mcp_test_helpers import install_fake_konfai_runtime, yaml_dump  # noqa: E402


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


class _ProcRunning:
    pid = 124

    def poll(self) -> None:
        return None


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


def test_a_finished_transform_job_carries_where_its_data_went(tmp_path: Path) -> None:
    # A transform's run directory holds no data: each Write landed in the user's tree, recorded in
    # outputs.json beside the runtime log. The payload surfaces it, so "inspect_dataset" has a path.
    registry = JobRegistry({"queued", "running"})
    run_dir = tmp_path / "Transforms" / "PREP"
    run_dir.mkdir(parents=True)
    outputs = [{"group_src": "CT", "group_dest": "CT_out", "dataset": str(tmp_path / "Out"), "format": "h5"}]
    (run_dir / "outputs.json").write_text(json.dumps(outputs), encoding="utf-8")
    job = Job(
        job_id="job2",
        session="default",
        kind="transform",
        command=["echo", "ok"],
        cwd=tmp_path,
        log_path=tmp_path / "job.log",
        config_path=tmp_path / "Transform.yml",
        runtime_log_path=run_dir / "log_0.txt",
        status="running",
    )
    job.proc = cast(Any, _ProcDone(0))
    registry.jobs[job.job_id] = job

    payload = registry.payload(job, _isoformat)

    assert payload["status"] == "done"
    assert payload["outputs"] == outputs
    assert "inspect_dataset" in payload["next_actions"]


@pytest.mark.usefixtures("workspace_root")
def test_a_client_sees_where_a_finished_transform_wrote(
    tmp_path: Path, load_mcp_server: Callable[[], ModuleType]
) -> None:
    """The same manifest, over the wire. payload() can hold `outputs` and the tool still drop them
    on the way out, and what an agent acts on is the response: so this asserts through a client.
    """
    server = load_mcp_server()
    run_dir = tmp_path / "Transforms" / "PREP"
    run_dir.mkdir(parents=True)
    outputs = [{"group_src": "CT", "group_dest": "CT_out", "dataset": str(tmp_path / "Out"), "format": "h5"}]
    (run_dir / "outputs.json").write_text(json.dumps(outputs), encoding="utf-8")
    job = Job(
        job_id="client-transform",
        session="default",
        kind="transform",
        command=["echo", "ok"],
        cwd=tmp_path,
        log_path=tmp_path / "job.log",
        config_path=tmp_path / "Transform.yml",
        runtime_log_path=run_dir / "log_0.txt",
        status="running",
    )
    job.proc = cast(Any, _ProcDone(0))
    server.JOB_REGISTRY.jobs[job.job_id] = job

    async def scenario() -> None:
        async with fastmcp.Client(server.mcp) as client:
            answered = await client.call_tool("get_job_status", {"job_id": job.job_id})
            data = answered.structured_content
            assert data["status"] == "done"
            assert data["outputs"] == outputs
            assert "inspect_dataset" in data["next_actions"]

    asyncio.run(scenario())


def test_a_running_transform_job_does_not_report_a_previous_runs_outputs(tmp_path: Path) -> None:
    # The run directory is reused across runs of one name: while the job is still running, the
    # outputs.json there is a previous run's manifest, not this job's.
    registry = JobRegistry({"queued", "running"})
    run_dir = tmp_path / "Transforms" / "PREP"
    run_dir.mkdir(parents=True)
    stale = [{"group_src": "CT", "group_dest": "CT_out", "dataset": str(tmp_path / "Old"), "format": "h5"}]
    (run_dir / "outputs.json").write_text(json.dumps(stale), encoding="utf-8")
    job = Job(
        job_id="job3",
        session="default",
        kind="transform",
        command=["sleep", "1"],
        cwd=tmp_path,
        log_path=tmp_path / "job.log",
        config_path=tmp_path / "Transform.yml",
        runtime_log_path=run_dir / "log_0.txt",
        status="running",
    )
    job.proc = cast(Any, _ProcRunning())
    registry.jobs[job.job_id] = job

    payload = registry.payload(job, _isoformat)

    assert payload["status"] == "running"
    assert "outputs" not in payload


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
    # forever: a queued job counts as active and would block every future launch on its device.
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


def test_run_resume_and_failed_job_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    load_mcp_server: Callable[[], ModuleType],
) -> None:
    monkeypatch.setenv("KONFAI_MCP_WORKSPACES_ROOT", str(tmp_path / "workspaces"))
    monkeypatch.setenv("KONFAI_MCP_FAKE_SLEEP_S", "0.05")
    mcp_server = load_mcp_server()
    install_fake_konfai_runtime(tmp_path, monkeypatch, mcp_server)

    config = yaml_dump({"Trainer": {"train_name": "FAKE_RUN"}})

    async def scenario() -> None:
        async with fastmcp.Client(mcp_server.mcp) as client:
            await client.call_tool("initialize_session", {"overwrite": True})
            await client.call_tool("write_workflow_config", {"workflow": "train", "content": config})

            with pytest.raises(Exception, match="No checkpoint found to resume from"):
                await client.call_tool("run_resume", {})

            workspace = Path(mcp_server.WORKSPACE_LAYOUT.workspace_dir())
            checkpoint = workspace / "Checkpoints" / "FAKE_RUN" / "epoch_0000.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_text("checkpoint", encoding="utf-8")

            resumed = await client.call_tool("run_resume", {"lr": 0.0005})
            resumed_payload = resumed.structured_content
            assert "RESUME" in resumed_payload["command"]
            done = await client.call_tool(
                "wait_for_job", {"job_id": resumed_payload["job_id"], "timeout_s": 60.0, "poll_interval_s": 0.1}
            )
            done_payload = done.structured_content
            assert done_payload["status"] == "done"
            assert "wait_for_job" not in done_payload["next_actions"]
            assert done_payload["next_resources"] == [f"job://{done_payload['job_id']}/log"]

            manifest = await client.read_resource(f"job://{resumed_payload['job_id']}/manifest")
            manifest_data = json.loads(manifest[0].text)
            assert manifest_data["manifest"]["resume_from"] == str(checkpoint)
            assert manifest_data["manifest"]["lr_override"] == 0.0005

            monkeypatch.setenv("KONFAI_MCP_FAKE_EXIT_CODE", "1")
            failed = await client.call_tool("run_train", {"overwrite": True})
            failed_payload = failed.structured_content
            failed_done = await client.call_tool(
                "wait_for_job", {"job_id": failed_payload["job_id"], "timeout_s": 60.0, "poll_interval_s": 0.1}
            )
            failed_data = failed_done.structured_content
            assert failed_data["status"] == "error"
            assert failed_data["error"], "a crashed job must state WHY in its payload"
            assert "read_job_log" in failed_data["next_actions"]
            assert "run_train" in failed_data["next_actions"]
            assert not any(
                str(action).startswith(("retry:", "read_resource:")) for action in failed_data["next_actions"]
            )

            # 'auto' must pick the console job log for a FAILED job (the traceback lives there).
            log = await client.call_tool(
                "read_job_log",
                {"job_id": failed_payload["job_id"], "grep": "simulated failure"},
            )
            log_data = log.structured_content
            assert "simulated failure" in log_data["content"]
            assert log_data["lines_returned"] >= 1

    asyncio.run(scenario())


def test_run_resume_weights_only_strips_to_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    load_mcp_server: Callable[[], ModuleType],
) -> None:
    """weights_only=True warm-starts a fine-tune: it loads ONLY the Model weights into a jailed
    <stem>_init.pt, so RESUME leaves epoch/iteration at 0 (fresh schedule)."""
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("KONFAI_MCP_WORKSPACES_ROOT", str(tmp_path / "workspaces"))
    monkeypatch.setenv("KONFAI_MCP_FAKE_SLEEP_S", "0.05")
    mcp_server = load_mcp_server()
    install_fake_konfai_runtime(tmp_path, monkeypatch, mcp_server)

    config = yaml_dump({"Trainer": {"train_name": "FAKE_RUN"}})

    async def scenario() -> None:
        async with fastmcp.Client(mcp_server.mcp) as client:
            await client.call_tool("initialize_session", {"overwrite": True})
            await client.call_tool("write_workflow_config", {"workflow": "train", "content": config})

            workspace = Path(mcp_server.WORKSPACE_LAYOUT.workspace_dir())
            checkpoint = workspace / "Checkpoints" / "FAKE_RUN" / "epoch_0009.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            # A full training checkpoint: Model weights beside the counters/optimizer a plain RESUME restores.
            torch.save({"Model": {"w": torch.zeros(2)}, "epoch": 9, "it": 900, "optimizer": {"state": {}}}, checkpoint)

            # A URL cannot be stripped to weights: weights_only demands a local checkpoint.
            with pytest.raises(Exception, match="local checkpoint"):
                await client.call_tool("run_resume", {"weights_only": True, "model": "https://example.com/model.pt"})

            resumed = await client.call_tool("run_resume", {"weights_only": True})
            manifest = await client.read_resource(f"job://{resumed.structured_content['job_id']}/manifest")
            resume_from = Path(json.loads(manifest[0].text)["manifest"]["resume_from"])

            # The resume points at the stripped copy, jailed in the session root, not the raw checkpoint.
            assert resume_from.name == "epoch_0009_init.pt"
            assert resume_from.parent == workspace
            from konfai.utils.runtime import safe_torch_load

            assert set(safe_torch_load(resume_from, "cpu")) == {"Model"}

    asyncio.run(scenario())


def _wait_job_finished(registry: Any, job: Any, timeout: float = 90.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        registry.refresh(job)
        if job.status not in ("queued", "running"):
            return
        time.sleep(0.05)
    raise AssertionError(f"job {job.job_id} never finished (status={job.status})")


def test_job_native_writes_never_reach_the_inherited_stdio(tmp_path: Path) -> None:
    # konfai-mcp speaks JSON-RPC over stdio and a spawned job inherits fds 1/2, but redirect_stdout /
    # redirect_stderr are Python-level only: they rebind sys.stdout/sys.stderr and leave the descriptors
    # alone. Any native write by the job or a descendant (a CUDA/PyTorch banner, a C extension, a DDP
    # worker) therefore landed in the middle of the protocol stream and froze the client on wait_for_job.
    # Stand in for the server's stdio with two files, run a REAL job through _run_job, and require that
    # everything it wrote natively is in the job log and that the inherited stdout stayed byte-clean.
    # Resolve JobRegistry from the live module: another test's load_mcp_server reloads server_jobs, and
    # spawn can only pickle the _run_job currently registered there.
    job_registry_cls = importlib.import_module("konfai_mcp.server_jobs").JobRegistry
    layout = WorkspaceLayout(tmp_path)
    layout.ensure_session_workspace()
    registry = job_registry_cls({"queued", "running"}, workspace_layout=layout)
    marker = "PROTOCOL-CANARY"
    log_path = tmp_path / "log.txt"
    inherited_stdout = tmp_path / "inherited_stdout.bin"
    inherited_stderr = tmp_path / "inherited_stderr.bin"

    saved_stdout_fd = os.dup(1)
    saved_stderr_fd = os.dup(2)
    job = None
    try:
        with inherited_stdout.open("wb") as stdout_sink, inherited_stderr.open("wb") as stderr_sink:
            os.dup2(stdout_sink.fileno(), 1)
            os.dup2(stderr_sink.fileno(), 2)
            try:
                job = registry.launch(
                    session=layout.current_session or "default",
                    kind="train",
                    command=["x"],
                    cwd=tmp_path,
                    log_path=log_path,
                    config_path=tmp_path / "cfg.ref",
                    target="mcp_test_helpers:write_on_native_stdio",
                    kwargs={"marker": marker},
                )
                _wait_job_finished(registry, job)
            finally:
                os.dup2(saved_stdout_fd, 1)
                os.dup2(saved_stderr_fd, 2)
    finally:
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)

    log_text = log_path.read_text(encoding="utf-8")
    assert job is not None and job.status == "done", f"job did not complete:\n{log_text}"
    for expected in (
        f"{marker}-native-stdout",
        f"{marker}-native-stderr",
        f"{marker}-python-print",
        f"{marker}-grandchild",
    ):
        assert expected in log_text, f"'{expected}' missing from the job log:\n{log_text}"
    # The JSON-RPC channel: not "no marker" but not a single byte: anything at all desynchronises it.
    assert inherited_stdout.read_bytes() == b""
    assert marker not in inherited_stderr.read_text(encoding="utf-8", errors="replace")


def test_isolated_api_never_writes_on_the_inherited_stdio(tmp_path: Path) -> None:
    """The validation/smoke-test child inherits the server's stdio, and the server speaks JSON-RPC on it.

    A single stray byte desynchronises the stream: the response to this very call is delivered and then
    destroyed, and the caller waits forever for an answer that already arrived. This covers the path
    `validate_config_semantics`, `run_component_smoke_test` and `import_app` all take: `import_app` runs
    `pip install`, whose `\\r` progress bars are exactly the shape that eats a frame.
    """
    from konfai_mcp.runner import run_api_in_subprocess

    inherited = tmp_path / "inherited_stdout.bin"
    with inherited.open("wb") as sink:
        saved_out, saved_err = os.dup(1), os.dup(2)
        try:
            os.dup2(sink.fileno(), 1)
            os.dup2(sink.fileno(), 2)
            result = run_api_in_subprocess("mcp_test_helpers:noisy_runner_api", {"marker": "PROBE"})
        finally:
            os.dup2(saved_out, 1)
            os.dup2(saved_err, 2)
            os.close(saved_out)
            os.close(saved_err)

    assert result == {"marker": "PROBE"}
    assert inherited.read_bytes() == b"", "not one byte may reach the JSON-RPC channel"


def test_a_failing_isolated_api_keeps_what_the_child_printed(tmp_path: Path) -> None:
    """Its output is the child's only diagnostic: a native crash leaves nothing else behind."""
    from konfai_mcp.runner import run_api_in_subprocess

    with pytest.raises(RuntimeError) as failure:
        run_api_in_subprocess("mcp_test_helpers:failing_runner_api", {"marker": "PROBE"})

    assert "ValueError: expected failure" in str(failure.value)
    assert "PROBE-said-this-before-dying" in str(failure.value)


def test_a_job_stopped_on_purpose_is_killed_not_broken() -> None:
    """Cancel sends SIGTERM. When the process died before the request was recorded, the flag alone was
    False and Stop reported "read the job log for the traceback" over a job that has none."""
    import signal as _signal

    from konfai_mcp.server_jobs import Job, JobRegistry

    class _Exited:
        def __init__(self, code: int) -> None:
            self._code = code

        def poll(self) -> int:
            return self._code

    registry = JobRegistry({"queued", "running"})
    for code, expected in ((-_signal.SIGTERM, "killed"), (-_signal.SIGINT, "killed"), (-_signal.SIGSEGV, "error")):
        job = Job(
            job_id=f"j{code}",
            session="s",
            kind="train",
            command=["x"],
            cwd=Path("."),
            log_path=Path("missing.log"),
            config_path=None,
            status="running",
        )
        job.proc = _Exited(code)
        registry.jobs[job.job_id] = job
        registry.refresh(job)
        assert job.status == expected, f"returncode {code}"


def test_a_weightless_model_is_not_blocked_for_want_of_a_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, load_mcp_server: Callable[[], ModuleType]
) -> None:
    """KonfAI runs a model with zero parameters as constructed (a registration engine has no weights to
    load), and refuses a parameterised one itself. The MCP raised before ever reaching it, so no weightless
    app could be launched through it. The readiness summary still reports the absent checkpoint as advice."""
    monkeypatch.setenv("KONFAI_MCP_WORKSPACES_ROOT", str(tmp_path / "workspaces"))
    server = load_mcp_server()
    session = server.SESSION
    session.workspace_dir().mkdir(parents=True, exist_ok=True)
    (session.workspace_dir() / "Prediction.yml").write_text("Predictor:\n  Model:\n    classpath: x:Y\n")

    assert session.resolve_prediction_models() == []  # nothing to block on
    assert session._workflow_status("prediction")["missing_models"] == ["<checkpoint>"]  # still advised


def test_a_failed_stdio_detach_is_said_in_the_sink(tmp_path: Path) -> None:
    """dup2 can fail on an exotic fd state; the call goes on, but a child running with fd 1 still on the
    protocol must say so where ``run_api_in_subprocess`` already looks: silence here replays the frozen
    stream as an unexplainable mystery."""
    sink = tmp_path / "subprocess.log"
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    child = context.Process(target=mcp_test_helpers.entry_with_broken_dup2, args=(queue, str(sink)))
    child.start()
    result = queue.get(timeout=120)
    child.join(120)

    assert result["ok_transport"] is True, "the call itself must still run and answer"
    assert "could not detach the child stdio" in sink.read_text(encoding="utf-8")
