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

"""Contract tests for the remote tunables path (``REMOTE_OPTION_FIELDS``).

Every spec'd tunable must travel client -> endpoint -> job command -> local op, and a tunable that
cannot be honoured must fail loudly (HTTP 422 server-side, ``KonfAIAppClientError`` client-side) —
never be silently dropped.
"""

import inspect
import json
import shutil
import sys
import types
import urllib.parse
from pathlib import Path
from typing import Any

import pytest

# Before importing anything that pulls in FastAPI (app_server, the TestClient): a module-level import runs
# at collection, so pytestmark would skip too late and collection would error when FastAPI is absent.
pytest.importorskip("fastapi")

import konfai_apps.app as app_module
import konfai_apps.app_server as app_server
import konfai_apps.cli as apps_cli_module
from fastapi.testclient import TestClient
from konfai.utils.errors import KonfAIAppClientError
from konfai_apps.remote_options import REMOTE_OPTION_FIELDS

TUNABLE_VALUES: dict[str, Any] = {
    "patch_size": [160, 160, 160],
    "batch_size": 2,
    "config_overrides": ["iterations=300", "Predictor.Dataset.batch_size=2"],
}


def _make_client(app_id: str = "demo") -> app_module.KonfAIAppClient:
    client = app_module.KonfAIAppClient.__new__(app_module.KonfAIAppClient)
    client.app = app_id
    client.remote_server = types.SimpleNamespace(  # type: ignore[attr-defined]
        get_url=lambda: "http://server",
        get_headers=lambda: {},
    )
    return client


class _StubServerResponse:
    """Minimal stand-in for the requests.Response context manager the client consumes."""

    def __init__(self, payload: dict) -> None:
        self._payload = payload
        self.status_code = 200

    def __enter__(self) -> "_StubServerResponse":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _ProxyResponse:
    """Adapt an httpx TestClient response to the requests context-manager protocol."""

    def __init__(self, response) -> None:  # type: ignore[no-untyped-def]
        self._response = response

    def __enter__(self):  # type: ignore[no-untyped-def]
        return self._response

    def __exit__(self, *exc: object) -> bool:
        return False


@pytest.fixture
def remote_stack(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Wire the client's ``requests.post`` into the FastAPI app, with job execution stubbed.

    Yields a record of the submitted posts and the job commands the server would have executed.
    """
    monkeypatch.delenv("KONFAI_API_TOKEN", raising=False)
    monkeypatch.setattr(app_server, "_APPS", ["demo"])

    recorded: dict[str, list] = {"cmds": [], "jobs": [], "posts": []}

    def record_start_job(job, cmd, requested_gpus):  # type: ignore[no-untyped-def]
        # Runs synchronously inside asyncio.create_task(start_job(...)), so the cmd is captured
        # before the submit response returns — no race with the event loop.
        recorded["cmds"].append(cmd)
        recorded["jobs"].append(job)

        async def _noop() -> None:
            return None

        return _noop()

    monkeypatch.setattr(app_server, "start_job", record_start_job)
    monkeypatch.setattr(app_module.KonfAIAppClient, "stream_logs", lambda self, job_id: None)
    monkeypatch.setattr(app_module.KonfAIAppClient, "download_result", lambda self, job_id, output: None)

    with TestClient(app_server.app) as test_client:

        def fake_post(url, files=None, data=None, headers=None, timeout=None):  # type: ignore[no-untyped-def]
            # requests omits None-valued form fields; httpx would send them as empty strings.
            form = {k: v for k, v in (data or {}).items() if v is not None}
            response = test_client.post(
                urllib.parse.urlparse(url).path, files=files or [], data=form, headers=headers or {}
            )
            recorded["posts"].append({"data": form, "response": response})
            return _ProxyResponse(response)

        monkeypatch.setattr(app_module.requests, "post", fake_post)
        yield recorded

    for job in recorded["jobs"]:
        shutil.rmtree(job.run_dir, ignore_errors=True)
        app_server.SERVER_STATE.jobs.pop(job.job_id, None)


def _dispatch_cmd_to_local_op(monkeypatch: pytest.MonkeyPatch, cmd: list[str]) -> tuple[str, dict[str, Any]]:
    """Parse a server-built job command with the real CLI and capture the local op call."""
    calls: list[tuple[str, dict[str, Any]]] = []

    class RecordingLocalApp:
        def __init__(self, app: str, download: bool, force_update: bool) -> None:
            pass

        def __getattr__(self, name: str):  # type: ignore[no-untyped-def]
            def op(**kwargs: Any) -> None:
                calls.append((name, kwargs))

            return op

    monkeypatch.setattr(app_module, "KonfAIApp", RecordingLocalApp)
    monkeypatch.setattr(sys, "argv", list(cmd))
    apps_cli_module.main_apps()
    assert len(calls) == 1
    return calls[0]


def _write_volume(path: Path) -> Path:
    path.write_bytes(b"volume")
    return path


def test_infer_tunables_reach_the_local_op(remote_stack: dict, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    volume = _write_volume(tmp_path / "a.nii.gz")

    _make_client().infer(inputs=[[volume]], output=tmp_path / "out", cpu=1, **TUNABLE_VALUES)

    assert remote_stack["posts"][0]["response"].json()["accepted_options"] == sorted(TUNABLE_VALUES)
    op, kwargs = _dispatch_cmd_to_local_op(monkeypatch, remote_stack["cmds"][0])
    assert op == "infer"
    for name, value in TUNABLE_VALUES.items():
        assert kwargs[name] == value, name


def test_pipeline_tunables_reach_the_local_op(
    remote_stack: dict, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    volume = _write_volume(tmp_path / "a.nii.gz")
    reference = _write_volume(tmp_path / "ref.nii.gz")

    _make_client().pipeline(inputs=[[volume]], gt=[[reference]], output=tmp_path / "out", cpu=1, **TUNABLE_VALUES)

    assert remote_stack["posts"][0]["response"].json()["accepted_options"] == sorted(TUNABLE_VALUES)
    op, kwargs = _dispatch_cmd_to_local_op(monkeypatch, remote_stack["cmds"][0])
    assert op == "pipeline"
    for name, value in TUNABLE_VALUES.items():
        assert kwargs[name] == value, name


def test_fine_tune_config_overrides_reach_the_local_op(
    remote_stack: dict, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset = tmp_path / "cohort"
    (dataset / "P000").mkdir(parents=True)
    _write_volume(dataset / "P000" / "Volume_0.mha")
    overrides = ["lr_decay=0.5", "Trainer.Dataset.batch_size=2"]

    _make_client().fine_tune(dataset=dataset, name="Tune", output=tmp_path / "out", cpu=1, config_overrides=overrides)

    assert remote_stack["posts"][0]["response"].json()["accepted_options"] == ["config_overrides"]
    op, kwargs = _dispatch_cmd_to_local_op(monkeypatch, remote_stack["cmds"][0])
    assert op == "fine_tune"
    assert kwargs["config_overrides"] == overrides


def test_ops_without_tunables_send_no_options_field(remote_stack: dict, tmp_path: Path) -> None:
    volume = _write_volume(tmp_path / "a.nii.gz")
    reference = _write_volume(tmp_path / "ref.nii.gz")

    _make_client().evaluate(inputs=[[volume]], gt=[[reference]], output=tmp_path / "out", cpu=1)

    submit = remote_stack["posts"][0]
    assert "options" not in submit["data"]
    assert submit["response"].json()["accepted_options"] == []


def test_no_tunables_keeps_old_server_compatibility(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    posts: list[dict] = []

    def old_server_post(url, files=None, data=None, headers=None, timeout=None):  # type: ignore[no-untyped-def]
        posts.append(dict(data or {}))
        return _StubServerResponse({"job_id": "job-1"})  # pre-options server: no accepted_options

    monkeypatch.setattr(app_module.requests, "post", old_server_post)
    monkeypatch.setattr(app_module.KonfAIAppClient, "stream_logs", lambda self, job_id: None)
    monkeypatch.setattr(app_module.KonfAIAppClient, "download_result", lambda self, job_id, output: None)
    volume = _write_volume(tmp_path / "a.nii.gz")

    # An explicit empty list means "no overrides": it must not travel as an options payload either.
    _make_client().infer(inputs=[[volume]], output=tmp_path / "out", config_overrides=[])

    assert "options" not in posts[0]


def test_old_server_response_raises_and_names_the_lost_parameters(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def old_server_post(url, files=None, data=None, headers=None, timeout=None):  # type: ignore[no-untyped-def]
        return _StubServerResponse({"job_id": "job-1"})  # accepts the submit but drops the options

    killed: list[str] = []
    monkeypatch.setattr(app_module.requests, "post", old_server_post)
    monkeypatch.setattr(app_module.KonfAIAppClient, "stream_logs", lambda self, job_id: None)
    monkeypatch.setattr(app_module.KonfAIAppClient, "kill_job", lambda self, job_id: killed.append(job_id))
    volume = _write_volume(tmp_path / "a.nii.gz")

    with pytest.raises(KonfAIAppClientError) as excinfo:
        _make_client().infer(inputs=[[volume]], output=tmp_path / "out", **TUNABLE_VALUES)

    message = str(excinfo.value)
    for name in TUNABLE_VALUES:
        assert name in message
    assert killed == ["job-1"]  # the job that would run without the tunables is not left running


def test_unknown_option_is_rejected_with_422_naming_the_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KONFAI_API_TOKEN", raising=False)
    monkeypatch.setattr(app_server, "_APPS", ["demo"])

    with TestClient(app_server.app) as test_client:
        response = test_client.post(
            "/apps/demo/infer",
            files=[("inputs", ("a.nii.gz", b"volume"))],
            data={"options": json.dumps({"mc": 3})},
        )

    assert response.status_code == 422
    assert "mc" in response.json()["detail"]
    assert app_server.JOBS == {}


def test_option_valid_elsewhere_is_rejected_on_an_op_that_does_not_honour_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("KONFAI_API_TOKEN", raising=False)
    monkeypatch.setattr(app_server, "_APPS", ["demo"])

    with TestClient(app_server.app) as test_client:
        response = test_client.post(
            "/apps/demo/evaluate",
            files=[("inputs", ("a.nii.gz", b"volume")), ("gt", ("ref.nii.gz", b"volume"))],
            data={"options": json.dumps({"patch_size": [160, 160, 160]})},
        )

    assert response.status_code == 422
    assert "patch_size" in response.json()["detail"]


def test_malformed_option_value_is_rejected_with_422_naming_the_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KONFAI_API_TOKEN", raising=False)
    monkeypatch.setattr(app_server, "_APPS", ["demo"])

    with TestClient(app_server.app) as test_client:
        response = test_client.post(
            "/apps/demo/infer",
            files=[("inputs", ("a.nii.gz", b"volume"))],
            data={"options": json.dumps({"patch_size": "big"})},
        )

    assert response.status_code == 422
    assert "patch_size" in response.json()["detail"]


def test_spec_fields_are_declared_across_local_client_and_server() -> None:
    # The client filter forwards exactly the spec'd names and the server maps them onto the local
    # op through the CLI: a spec'd field missing from any of the three surfaces is a silent drop.
    for op, fields in REMOTE_OPTION_FIELDS.items():
        local_params = inspect.signature(getattr(app_module.KonfAIApp, op)).parameters
        client_params = inspect.signature(getattr(app_module.KonfAIAppClient, op)).parameters
        endpoint_params = inspect.signature(getattr(app_server, op)).parameters
        assert "options" in endpoint_params, op
        for field in fields:
            assert field in local_params, (op, field)
            assert field in client_params, (op, field)
