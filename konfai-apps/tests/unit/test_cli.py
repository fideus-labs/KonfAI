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
