# Copyright (c) 2026 Fideus Labs LLC
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

"""The Scout panel's routes: the session's reports, one report with its observations in millimetres, and its
evidence images."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from konfai_studio import server
from starlette.testclient import TestClient

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


@pytest.fixture
def report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("KONFAI_MCP_WORKSPACES_ROOT", str(tmp_path))
    folder = tmp_path / "sessions" / "exp" / "Scout" / "demo"
    (folder / "evidence").mkdir(parents=True)
    (folder / "evidence" / "v001.png").write_bytes(PNG)
    (folder / "report.md").write_text("# report")
    (folder / "report.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "created_at": "2026-11-07T10:00:00+00:00",
                "case": {
                    "name": "c1",
                    "task": "segmentation",
                    "truth_available": True,
                    "inputs": {
                        "image": {"path": "/data/c1/ct.mha", "sha256": "a"},
                        "prediction": {"path": "/data/c1/seg.mha", "sha256": "b"},
                    },
                },
                "config": {"mode": "active"},
                "provider": {"name": "mock", "model": "heuristic"},
                "conclusion": {
                    "verdict": "defective",
                    "summary": "one error",
                    "confidence": 0.8,
                    "by": "model",
                    "reason": None,
                },
                "observations": [
                    {
                        "id": "o001",
                        "kind": "confirmed_error",
                        "defect": "missing_structure",
                        "severity": "major",
                        "confidence": 0.9,
                        "description": "the cube is missing",
                        "box_mm": {"min": [0, 0, 0], "max": [10, 10, 10]},
                        "plane": "axial",
                        "views": ["v001"],
                        "measurements": ["m001"],
                        "notes": [],
                    }
                ],
                "measurements": [
                    {"id": "m001", "kind": "konfai_eval", "reference": True, "values": {"PRED:TRUTH:Dice:2": 0.0}}
                ],
                "views": [
                    {
                        "id": "v001",
                        "kind": "slice",
                        "plane": "axial",
                        "mode": "both",
                        "position_mm": [5, 5, 5],
                        "sent": True,
                        "px_to_mm": {"origin": [0, 0, 0], "u": [1, 0, 0], "v": [0, 1, 0]},
                        "sha256": "x",
                        "path": "evidence/v001.png",
                    }
                ],
                "usage": {"images_sent": 1},
                "budget": {"exhausted": None},
                "error": None,
            }
        )
    )
    return folder


def test_reports_report_and_evidence(report: Path) -> None:
    with TestClient(server.app) as client:
        listed = client.get("/api/scout/reports", params={"session": "exp"}).json()["reports"]
        assert (
            [r["name"] for r in listed] == ["demo"]
            and listed[0]["verdict"] == "defective"
            and listed[0]["observations"] == 1
        )
        assert client.get("/api/scout/reports", params={"session": "other"}).json() == {"reports": []}
        data = client.get("/api/scout/report", params={"session": "exp", "name": "demo"}).json()
        assert data["image"] == "/data/c1/ct.mha" and data["truth_available"] is True
        assert data["observations"][0]["box_mm"]["max"] == [10, 10, 10] and data["views"] == 1
        assert data["views_meta"] == [
            {
                "id": "v001",
                "kind": "slice",
                "plane": "axial",
                "index": None,
                "position_mm": [5, 5, 5],
                "mode": "both",
                "sent": True,
                "size_px": None,
            }
        ]
        assert data["created_at"] == "2026-11-07T10:00:00+00:00" and data["second_reading"] is None
        assert data["report_md"].endswith("report.md")
        image = client.get("/api/scout/evidence", params={"session": "exp", "name": "demo", "view": "v001"})
        assert image.status_code == 200 and image.headers["content-type"] == "image/png" and image.content == PNG
        assert (
            client.get(
                "/api/scout/evidence", params={"session": "exp", "name": "demo", "view": "../report"}
            ).status_code
            == 400
        )
        assert (
            client.get("/api/scout/evidence", params={"session": "exp", "name": "demo", "view": "v999"}).status_code
            == 404
        )
        assert client.get("/api/scout/report", params={"session": "exp", "name": "nope"}).status_code == 404
        assert client.get("/api/scout/report", params={"session": "exp", "name": "../../exp"}).status_code == 404
        assert client.get("/api/scout/report", params={"session": "exp", "name": ""}).status_code == 404
