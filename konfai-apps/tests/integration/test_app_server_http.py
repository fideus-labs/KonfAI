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

import importlib.util
from pathlib import Path

import konfai_apps.app_server as app_server
import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("fastapi") is None,
    reason="fastapi is not installed",
)


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
