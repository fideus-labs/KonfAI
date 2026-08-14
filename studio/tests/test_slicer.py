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

"""The 3D Slicer hand-over: a running instance gets the volumes through its Web Server module, else
Slicer is launched on them; a refusal names what is wrong."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")
from konfai_studio import server
from starlette.testclient import TestClient


def volume(tmp_path: Path, name: str = "sCT.nii.gz") -> Path:
    path = tmp_path / name
    path.write_bytes(b"\x00")
    return path


def post(paths: list[str]) -> Any:
    with TestClient(server.app) as client:
        return client.post("/api/slicer/open", json={"paths": paths})


def test_a_running_slicer_gets_the_volumes_without_a_second_instance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sct, ct = volume(tmp_path, "sCT.nii.gz"), volume(tmp_path, "CT.mha")
    sent: list[str] = []
    monkeypatch.setattr(server, "_slicer_exec", lambda code: sent.append(code) or True)

    response = post([str(sct), str(ct)])

    assert response.json() == {"ok": True, "via": "webserver"}
    assert str(sct) in sent[0] and str(ct) in sent[0]


def test_without_a_running_slicer_one_is_launched_on_the_volumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sct = volume(tmp_path)
    executable = tmp_path / "Slicer"
    executable.write_text("#!/bin/sh\n")
    launched: list[list[str]] = []
    monkeypatch.setattr(server, "_slicer_exec", lambda code: False)
    monkeypatch.setenv("KONFAI_STUDIO_SLICER", str(executable))
    monkeypatch.setattr(subprocess, "Popen", lambda argv, **kwargs: launched.append(argv))

    response = post([str(sct)])

    assert response.json() == {"ok": True, "via": "launch"}
    assert launched == [[str(executable), str(sct)]]


def test_no_slicer_anywhere_answers_with_the_fix_not_a_500(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sct = volume(tmp_path)
    monkeypatch.setattr(server, "_slicer_exec", lambda code: False)
    monkeypatch.delenv("KONFAI_STUDIO_SLICER", raising=False)
    monkeypatch.setattr(server.shutil, "which", lambda name: None)

    payload = post([str(sct)]).json()

    assert payload["ok"] is False
    assert "KONFAI_STUDIO_SLICER" in payload["detail"]


def test_only_volume_files_are_handed_over(tmp_path: Path) -> None:
    config = tmp_path / "Config.yml"
    config.write_text("Trainer: {}\n")

    assert post([str(config)]).status_code == 415
    assert post([str(tmp_path / "missing.nii.gz")]).status_code == 404
    assert post([]).status_code == 400
