# SPDX-License-Identifier: Apache-2.0
"""Cancellation must outlive the job's group leader and reach TERM-resistant descendants."""

import os
import signal
import subprocess
import sys
import time
import types
from pathlib import Path

import psutil
import pytest

pytest.importorskip("fastapi")

import konfai_apps.app_server as app_server

_CHILD = """
import os, signal, sys, time
from pathlib import Path
signal.signal(signal.SIGTERM, signal.SIG_IGN)
ready = Path(sys.argv[1])
staging = ready.with_suffix('.tmp')
staging.write_text(str(os.getpid()))
staging.replace(ready)
time.sleep(60)
"""

_LEADER = """
import subprocess, sys, time
subprocess.Popen([sys.executable, '-c', sys.argv[1], sys.argv[2]])
if sys.argv[3] == 'exited':
    sys.exit(0)
time.sleep(60)
"""


def _running(pid: int) -> bool:
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


@pytest.mark.skipif(os.name == "nt", reason="process groups require POSIX")
@pytest.mark.parametrize("leader_state", ["exited", "running"])
def test_cancel_kills_a_resistant_descendant_after_the_leader_exits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, leader_state: str
) -> None:
    ready = tmp_path / "child.pid"
    leader = subprocess.Popen(
        [sys.executable, "-c", _LEADER, _CHILD, str(ready), leader_state],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    killpg = os.killpg
    sent: list[int] = []
    child_pid = None
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ready.exists(), "descendant did not initialize"
        child_pid = int(ready.read_text())
        assert _running(child_pid)
        if leader_state == "exited":
            assert leader.wait(timeout=5) == 0
        else:
            assert leader.poll() is None

        def record(group: int, sig: int) -> None:
            assert group == leader.pid  # only the group created by this test is signalled
            sent.append(sig)
            killpg(group, sig)

        monkeypatch.setattr(app_server.os, "killpg", record)
        job = app_server.Job(
            job_id=f"group-{leader.pid}",
            app_name="test",
            run_dir=tmp_path,
            input_dir=tmp_path,
            output_dir=tmp_path,
            zip_path=tmp_path / "result.zip",
            proc=leader,
            status="running",
        )
        monkeypatch.setitem(app_server.SERVER_STATE.jobs, job.job_id, job)
        response = app_server.kill_job(job.job_id)
        assert response["status"] == "killed" and job.cancelled
        assert signal.SIGTERM in sent and signal.SIGKILL in sent
        leader.wait(timeout=5)
        deadline = time.monotonic() + 5
        while _running(child_pid) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert not _running(child_pid), "descendant survived group cancellation"
    finally:
        try:
            killpg(leader.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        leader.wait(timeout=5)
        # A descendant reparented to init may briefly be a zombie; it holds no GPU or open
        # descriptors. Only the PID recorded by this test may need a final termination request.
        if child_pid is not None and _running(child_pid):
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


class _Without(types.ModuleType):
    """A module view without some attributes: Windows' os has no killpg and its signal no SIGKILL."""

    def __init__(self, real: types.ModuleType, *missing: str) -> None:
        super().__init__(real.__name__)
        self._real, self._missing = real, set(missing)

    def __getattr__(self, name: str):  # type: ignore[no-untyped-def]
        if name in self._missing:
            raise AttributeError(f"module '{self._real.__name__}' has no attribute '{name}'")
        return getattr(self._real, name)


def test_cancel_kills_the_job_and_its_descendants_without_process_groups(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ready = tmp_path / "child.pid"
    leader = subprocess.Popen(
        [sys.executable, "-c", _LEADER, _CHILD, str(ready), "running"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    child_pid = None
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ready.exists(), "descendant did not initialize"
        child_pid = int(ready.read_text())

        monkeypatch.setattr(app_server, "os", _Without(os, "killpg"))
        monkeypatch.setattr(app_server, "signal", _Without(signal, "SIGKILL"))
        job = app_server.Job(
            job_id=f"tree-{leader.pid}",
            app_name="test",
            run_dir=tmp_path,
            input_dir=tmp_path,
            output_dir=tmp_path,
            zip_path=tmp_path / "result.zip",
            proc=leader,
            status="running",
        )
        monkeypatch.setitem(app_server.SERVER_STATE.jobs, job.job_id, job)
        response = app_server.kill_job(job.job_id)
        assert response["status"] == "killed" and job.cancelled
        leader.wait(timeout=5)
        deadline = time.monotonic() + 5
        while _running(child_pid) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert not _running(child_pid), "descendant survived cancellation"
    finally:
        leader.kill()
        leader.wait(timeout=5)
        if child_pid is not None and _running(child_pid):
            psutil.Process(child_pid).kill()
