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

import shutil
from pathlib import Path

import pytest

# Before importing anything that pulls in FastAPI (app_server, the TestClient): a module-level import runs
# at collection, so pytestmark would skip too late and collection would error when FastAPI is absent.
pytest.importorskip("fastapi")

import konfai_apps.app_server as app_server
from fastapi.testclient import TestClient


def _make_job(job_id: str, status: str, zip_path: Path | None = None) -> app_server.Job:
    job = app_server.Job(
        job_id=job_id,
        app_name="demo",
        run_dir=Path(f"/tmp/{job_id}"),
        input_dir=Path(f"/tmp/{job_id}_in"),
        output_dir=Path(f"/tmp/{job_id}_out"),
        zip_path=zip_path or Path(f"/tmp/{job_id}.zip"),
    )
    job.status = status
    return job


@pytest.fixture
def client() -> TestClient:
    with TestClient(app_server.app) as test_client:
        yield test_client


def test_health_endpoint_enforces_bearer_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KONFAI_API_TOKEN", "secret")

    with TestClient(app_server.app) as client:
        missing = client.get("/health")
        wrong = client.get("/health", headers={"Authorization": "Bearer wrong"})
        ok = client.get("/health", headers={"Authorization": "Bearer secret"})

    assert missing.status_code == 401
    assert missing.json()["detail"] == "Missing bearer token"
    assert wrong.status_code == 401
    assert wrong.json()["detail"] == "Invalid token"
    assert ok.status_code == 200
    assert ok.json() == {"status": "ok"}


def test_available_devices_endpoint_returns_visible_gpu_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("KONFAI_API_TOKEN", raising=False)
    monkeypatch.setattr(app_server.konfai, "get_available_devices", lambda: ([0, 2], ["GPU0", "GPU2"]))

    with TestClient(app_server.app) as client:
        response = client.get("/available_devices")

    assert response.status_code == 200
    assert response.json() == {"devices_index": [0, 2], "devices_name": ["GPU0", "GPU2"]}


def test_job_endpoints_report_unknown_job_as_404(client: TestClient) -> None:
    status_response = client.get("/jobs/missing")
    result_response = client.get("/jobs/missing/result")

    assert status_response.status_code == 404
    assert status_response.json()["detail"] == "Unknown job_id"
    assert result_response.status_code == 404
    assert result_response.json()["detail"] == "Unknown job_id"


def test_job_result_endpoint_reports_pending_and_done_jobs(
    client: TestClient,
    tmp_path: Path,
) -> None:
    previous_jobs = dict(app_server.JOBS)
    pending_job = _make_job("pending", "running")
    zip_path = tmp_path / "result.zip"
    zip_path.write_bytes(b"PK\x03\x04demo")
    done_job = _make_job("done", "done", zip_path=zip_path)
    app_server.JOBS.clear()
    app_server.JOBS[pending_job.job_id] = pending_job
    app_server.JOBS[done_job.job_id] = done_job

    try:
        pending = client.get(f"/jobs/{pending_job.job_id}/result")
        done = client.get(f"/jobs/{done_job.job_id}/result")
    finally:
        app_server.JOBS.clear()
        app_server.JOBS.update(previous_jobs)

    assert pending.status_code == 202
    assert pending.json() == {"job_id": "pending", "status": "running"}
    assert done.status_code == 200
    assert done.content == b"PK\x03\x04demo"
    assert done.headers["content-type"] == "application/zip"


@pytest.fixture
def submit(monkeypatch: pytest.MonkeyPatch):
    """POST a job with its execution stubbed; yields the client and the ``(job, cmd)`` pairs it scheduled."""
    monkeypatch.delenv("KONFAI_API_TOKEN", raising=False)
    monkeypatch.setattr(app_server, "_APPS", ["demo"])
    scheduled: list[tuple[app_server.Job, list[str]]] = []

    def record_start_job(job, cmd, requested_gpus):  # type: ignore[no-untyped-def]
        scheduled.append((job, cmd))

        async def _noop() -> None:
            return None

        return _noop()

    monkeypatch.setattr(app_server, "start_job", record_start_job)
    with TestClient(app_server.app) as test_client:
        yield test_client, scheduled
    for job, _ in scheduled:
        shutil.rmtree(job.run_dir, ignore_errors=True)
        app_server.SERVER_STATE.jobs.pop(job.job_id, None)


_EMPTY_ZIP = b"PK\x05\x06" + b"\0" * 18

# Endpoint -> the uploads and group sizes it requires.
_UPLOADS: dict[str, tuple[list, dict[str, str]]] = {
    "infer": ([("inputs", ("a.nii.gz", b"v"))], {"inputs_groups": "1"}),
    "evaluate": (
        [("inputs", ("a.nii.gz", b"v")), ("gt", ("b.nii.gz", b"v"))],
        {"inputs_groups": "1", "gt_groups": "1"},
    ),
    "uncertainty": ([("inputs", ("a.nii.gz", b"v"))], {"inputs_groups": "1"}),
    "pipeline": (
        [("inputs", ("a.nii.gz", b"v")), ("gt", ("b.nii.gz", b"v"))],
        {"inputs_groups": "1", "gt_groups": "1"},
    ),
    "fine_tune": ([("dataset", ("d.zip", _EMPTY_ZIP))], {}),
}


def _post(client: TestClient, endpoint: str, **fields: str):  # type: ignore[no-untyped-def]
    files, groups = _UPLOADS[endpoint]
    return client.post(f"/apps/demo/{endpoint}", files=files, data={**groups, **fields})


@pytest.mark.parametrize(
    ("endpoint", "field"), [("infer", "ensemble_models"), ("pipeline", "ensemble_models"), ("fine_tune", "models")]
)
def test_a_model_list_entry_cannot_add_an_option_to_the_job_command(submit, endpoint: str, field: str) -> None:
    client, scheduled = submit

    response = _post(client, endpoint, **{field: "fold_0,--inputs,/etc/hostname"})

    assert response.status_code == 422, [cmd for _, cmd in scheduled]
    assert field in response.json()["detail"]
    assert scheduled == []


@pytest.mark.parametrize(
    ("endpoint", "field"),
    [
        ("infer", "prediction_file"),
        ("evaluate", "evaluation_file"),
        ("uncertainty", "uncertainty_file"),
        ("pipeline", "prediction_file"),
        ("pipeline", "evaluation_file"),
        ("pipeline", "uncertainty_file"),
        ("fine_tune", "config_file"),
    ],
)
@pytest.mark.parametrize("value", ["/srv/apps/Other/App.yml", "../../App.yml", "sub/../../App.yml"])
def test_a_config_file_field_cannot_leave_the_job_workspace(submit, endpoint: str, field: str, value: str) -> None:
    client, scheduled = submit

    response = _post(client, endpoint, **{field: value})

    assert response.status_code == 422, [cmd for _, cmd in scheduled]
    assert field in response.json()["detail"]
    assert scheduled == []


@pytest.mark.parametrize("value", ["Prediction_tile.yml", "configs/Prediction.yml"])
def test_a_config_file_named_inside_the_app_still_reaches_the_job(submit, value: str) -> None:
    client, scheduled = submit

    response = _post(client, "infer", prediction_file=value, ensemble_models="fold_0, CV_1.pt")

    assert response.status_code == 200
    cmd = scheduled[0][1]
    assert cmd[cmd.index("--prediction_file") + 1] == value
    assert cmd[cmd.index("--ensemble_models") + 1 : cmd.index("--ensemble_models") + 3] == ["fold_0", "CV_1.pt"]


@pytest.mark.parametrize("groups", [None, "", "one", "1", "3", "0,2", "-1,3"])
def test_group_sizes_that_do_not_cover_the_uploads_are_refused(submit, groups: str | None) -> None:
    client, scheduled = submit
    files = [("inputs", ("a.nii.gz", b"v")), ("inputs", ("b.nii.gz", b"v"))]
    data = {} if groups is None else {"inputs_groups": groups}

    response = client.post("/apps/demo/infer", files=files, data=data)

    assert response.status_code == 422, [cmd for _, cmd in scheduled]
    assert scheduled == []
    assert app_server.JOBS == {}


@pytest.mark.parametrize(
    ("script", "status", "error"),
    [
        (
            "from konfai_apps.cli import _exit_on_refusal\n"
            "from konfai.utils.errors import ConfigError\n"
            "with _exit_on_refusal():\n"
            "    raise ConfigError(\"Invalid value 'many' for field 'epochs'.\")",
            422,
            "[Config] Invalid value 'many' for field 'epochs'.",
        ),
        ("import sys; sys.exit(3)", 500, "Subprocess failed (exit code 3)"),
    ],
    ids=["refusal", "crash"],
)
def test_a_refused_job_answers_its_refusal_and_a_crash_its_exit_code(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, script: str, status: int, error: str
) -> None:
    """A designed refusal in the job's process reaches the client as its message, a 422; a crash stays
    a 500 naming the exit code. The refusal was 'Subprocess failed (exit code 1)' and a 500 too."""
    import sys

    monkeypatch.delenv("KONFAI_API_TOKEN", raising=False)
    monkeypatch.delenv("KONFAI_DEBUG", raising=False)
    job = _make_job("refused", "queued")
    job.run_dir = tmp_path
    previous_jobs = dict(app_server.JOBS)
    app_server.JOBS.clear()
    app_server.JOBS[job.job_id] = job
    try:
        app_server._run_job_sync(job, [sys.executable, "-c", script])
        result = client.get(f"/jobs/{job.job_id}/result")
    finally:
        app_server.JOBS.clear()
        app_server.JOBS.update(previous_jobs)

    assert result.status_code == status
    assert result.json()["error"] == error
