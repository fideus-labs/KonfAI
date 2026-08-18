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

import json
import sys
from pathlib import Path
from typing import Any, cast

import konfai_apps.app as app_module
import konfai_apps.cli as apps_cli_module
import pytest


def test_main_apps_dispatches_local_infer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[tuple[str, object]] = []

    class DummyApp:
        def __init__(self, app_name: str, download: bool, force_update: bool) -> None:
            calls.append(("init", (app_name, download, force_update)))

        def infer(self, **kwargs) -> None:
            calls.append(("infer", kwargs))

    monkeypatch.setattr(app_module, "KonfAIApp", DummyApp)
    monkeypatch.setattr(app_module, "KonfAIAppClient", object)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "konfai-apps",
            "infer",
            "demo/app",
            "--inputs",
            str(tmp_path / "input.mha"),
            "--cpu",
            "2",
            "--prediction-file",
            "Prediction.custom.yml",
            "--tta",
            "2",
            "--mc",
            "1",
        ],
    )

    apps_cli_module.main_apps()

    assert calls[0] == ("init", ("demo/app", False, False))
    infer_kwargs = cast(dict[str, Any], calls[1][1])
    assert infer_kwargs["cpu"] == 2
    assert infer_kwargs["gpu"] == []
    assert infer_kwargs["prediction_file"] == "Prediction.custom.yml"
    assert infer_kwargs["tta"] == 2
    assert infer_kwargs["mc"] == 1
    assert infer_kwargs["inputs"] == [[(tmp_path / "input.mha").resolve()]]


def _run_apps_server(monkeypatch: pytest.MonkeyPatch, apps_config: Path) -> str:
    """Drive `main_apps_server` up to its `--apps` validation and return the exit message.

    `main_apps_server` opens with `import uvicorn`, so every path through it -- including the
    validation that never reaches `uvicorn.run` -- needs the server stack importable.
    """
    uvicorn = pytest.importorskip("uvicorn")

    def _refuse_to_serve(*args: object, **kwargs: object) -> None:
        # Both callers must exit during validation. Should that validation ever regress, falling
        # through to the real `uvicorn.run` would bind a port and serve forever, hanging the unit
        # suite instead of failing it -- so make reaching this point a loud failure.
        pytest.fail("main_apps_server reached uvicorn.run: --apps validation did not reject the config")

    monkeypatch.setattr(uvicorn, "run", _refuse_to_serve)
    # --auth off keeps the test on config validation instead of token resolution; delenv first so
    # the pop the CLI performs on a real inherited token is restored at teardown.
    monkeypatch.delenv("KONFAI_API_TOKEN", raising=False)
    monkeypatch.setattr(sys, "argv", ["konfai-apps-server", "--auth", "off", "--apps", str(apps_config)])

    with pytest.raises(SystemExit) as excinfo:
        apps_cli_module.main_apps_server()
    return str(excinfo.value)


def test_main_apps_server_rejects_a_missing_apps_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # A mistyped --apps must name the path it could not find, not start a server with no apps.
    missing = tmp_path / "absent.json"

    assert str(missing) in _run_apps_server(monkeypatch, missing)


def test_main_apps_server_rejects_a_config_without_an_apps_list(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Valid JSON of the wrong shape must be refused up front: the server reads `apps` as a list, so
    # binding the port first would only surface the mistake on the first request.
    apps_config = tmp_path / "apps.json"
    apps_config.write_text(json.dumps({"applications": ["demo/app"]}), encoding="utf-8")

    assert "Invalid config file" in _run_apps_server(monkeypatch, apps_config)
