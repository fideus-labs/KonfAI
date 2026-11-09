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

"""KonfAI Scout and the FidOpt adapter through the MCP server: a tracked inspection on a synthetic case with the
free heuristic model, the privacy gate, and FidOpt reached only through a (fake) fidopt of its own."""

from __future__ import annotations

import asyncio
import json
import os
import stat
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

fastmcp = pytest.importorskip("fastmcp")
konfai_scout = pytest.importorskip("konfai_scout")
from fastmcp.exceptions import ToolError  # noqa: E402
from mcp_test_helpers import resource_to_text  # noqa: E402


def _call(server: ModuleType, tool: str, **arguments: object) -> object:
    async def scenario() -> object:
        async with fastmcp.Client(server.mcp) as client:
            result = await client.call_tool(tool, arguments)
            return json.loads(resource_to_text(result.content))

    return asyncio.run(scenario())


def _wait(server: ModuleType, job_id: str) -> dict:
    payload = _call(server, "get_job_status", job_id=job_id, wait=True, timeout_s=180.0)
    assert isinstance(payload, dict)
    return payload


@pytest.mark.usefixtures("workspace_root")
def test_an_inspection_runs_as_a_job_and_its_report_reads_back(
    tmp_path: Path, load_mcp_server: Callable[[], ModuleType], monkeypatch: pytest.MonkeyPatch
) -> None:
    from konfai_scout.synthetic import segmentation_case

    case, _ = segmentation_case(tmp_path / "seg")
    server = load_mcp_server()
    monkeypatch.setattr(server.konfai_pkg, "cuda_visible_devices", lambda: [0, 1])
    payload = _call(
        server,
        "run_scout_inspection",
        task="segmentation",
        name="seg_demo",
        inputs={role: str(path) for role, path in case.inputs.items()},
        labels={"1": "sphere", "2": "cube"},
        provider="mock",
        mode="active",
        max_tool_calls=12,
    )
    assert isinstance(payload, dict) and payload["kind"] == "scout" and payload["output"].endswith("Scout/seg_demo")
    assert payload["next_actions"] == ["get_job_status", "read_scout_report"]
    assert payload["devices"] == ["cpu"] and "vram_preflight" not in payload  # the model holds no GPU of this host
    done = _wait(server, payload["job_id"])
    assert done["status"] == "done", done
    listed = _call(server, "read_scout_report")
    assert isinstance(listed, dict) and [r["name"] for r in listed["reports"]] == ["seg_demo"]
    assert listed["reports"][0]["observations"] >= 1 and listed["reports"][0]["model"] == "heuristic"
    report = _call(server, "read_scout_report", name="seg_demo")
    assert isinstance(report, dict) and report["conclusion"]["verdict"] in (
        "defective",
        "acceptable",
        "insufficient_evidence",
    )
    assert report["observations"][0]["views"] and report["views"] and "px_to_mm" not in report["views"][0]
    assert Path(report["report_md"]).exists() and report["case"]["labels"] == {"1": "sphere", "2": "cube"}
    with pytest.raises(ToolError, match="read_scout_report"):
        _call(server, "read_scout_report", name="nope")
    with pytest.raises(ToolError):
        _call(server, "read_scout_report", name="../etc")


@pytest.mark.usefixtures("workspace_root")
def test_images_never_reach_a_cloud_model_unless_allowed(
    tmp_path: Path, load_mcp_server: Callable[[], ModuleType]
) -> None:
    from konfai_scout.synthetic import synthesis_case

    case, _ = synthesis_case(tmp_path / "syn")
    server = load_mcp_server()
    inputs = {role: str(path) for role, path in case.inputs.items()}
    with pytest.raises(ToolError, match="allow_cloud_images"):
        _call(server, "run_scout_inspection", task="synthesis", inputs=inputs, provider="anthropic")
    with pytest.raises(ToolError, match="needs"):
        _call(server, "run_scout_inspection", task="synthesis", inputs={"source": inputs["source"]}, provider="mock")
    with pytest.raises(ToolError, match="does not exist"):
        _call(
            server,
            "run_scout_inspection",
            task="synthesis",
            inputs={**inputs, "target": "/nowhere.mha"},
            provider="mock",
        )
    assert not list((Path(os.environ["KONFAI_MCP_WORKSPACES_ROOT"])).rglob("report.json"))


FAKE_FIDOPT = """#!/usr/bin/env python3
import json, sys
from pathlib import Path
args = sys.argv[1:]
if args[:1] == ["table"]:
    print(json.dumps({args[1]: [{"method": "identity", "machine": "none", "dice": 0.5}]}))
else:
    Path("fidopt_called.json").write_text(json.dumps(args))
"""

FAKE_CORPUS = """
import json
from pathlib import Path
REPOSITORY = Path(__file__).resolve().parent / "repo"
def datasets():
    return ["AMOS_CTCT", "WAWTACE"]
def descriptor(name):
    return {"AMOS_CTCT": {"title": "Abdomen", "modalities": ["ct"], "anatomy": "abdomen", "cases": ["a", "b"],
                          "development": ["c"], "spdx": "CC-BY-4.0", "use": "open"},
            "WAWTACE": {"title": "Liver", "task": "segmentation", "modalities": ["ct"], "anatomy": "liver", "cases": ["x"],
                        "spdx": "CC-BY-4.0"}}[name]
def task(info):
    return info.get("task", "registration")
"""


@pytest.fixture
def fake_fidopt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    package = tmp_path / "fakepkg" / "fidopt"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "corpus.py").write_text(FAKE_CORPUS)
    (package / "repo" / "methods" / "deeds").mkdir(parents=True)
    (package / "repo" / "methods" / "deeds" / "method.json").write_text(
        json.dumps(
            {"title": "deeds", "summary": "discrete", "licence": "BSD-3-Clause", "version": "1", "variants": {"v2": {}}}
        )
    )
    executable = tmp_path / "bin" / "fidopt"
    executable.parent.mkdir()
    executable.write_text(FAKE_FIDOPT)
    executable.chmod(executable.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("KONFAI_MCP_FIDOPT", str(executable))
    monkeypatch.setenv("KONFAI_MCP_FIDOPT_PYTHON", sys.executable)
    corpus = tmp_path / "corpus"  # what fidopt_catalog lists as built
    (corpus / "AMOS_CTCT").mkdir(parents=True)
    monkeypatch.setenv("FIDOPT_DATA", str(corpus))
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "fakepkg"))
    return executable


@pytest.mark.usefixtures("workspace_root", "fake_fidopt")
def test_fidopt_is_read_through_its_own_code_and_run_only_when_allowed(
    tmp_path: Path, load_mcp_server: Callable[[], ModuleType], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KONFAI_MCP_FIDOPT_ALLOWED", "check")
    server = load_mcp_server()
    datasets = _call(server, "fidopt_catalog")
    assert isinstance(datasets, dict) and [d["name"] for d in datasets["datasets"]] == ["AMOS_CTCT", "WAWTACE"]
    assert datasets["datasets"][0]["cases"] == 2 and datasets["datasets"][1]["task"] == "segmentation"
    only = _call(server, "fidopt_catalog", task="segmentation")
    assert isinstance(only, dict) and [d["name"] for d in only["datasets"]] == ["WAWTACE"]
    described = _call(server, "fidopt_catalog", dataset="AMOS_CTCT")
    assert (
        isinstance(described, dict)
        and described["dataset"]["anatomy"] == "abdomen"
        and described["dataset"]["task"] == "registration"
    )
    methods = _call(server, "fidopt_catalog")
    assert (
        isinstance(methods, dict)
        and methods["methods"][0]["name"] == "deeds"
        and methods["methods"][0]["variants"] == ["v2"]
    )
    table = _call(server, "fidopt_table", dataset="AMOS_CTCT")
    assert isinstance(table, dict) and table["table"]["AMOS_CTCT"][0]["method"] == "identity"
    with pytest.raises(ToolError, match="not allowed"):
        _call(server, "fidopt_run", operation="run", method="deeds", dataset="AMOS_CTCT")
    with pytest.raises(ToolError):
        _call(server, "fidopt_catalog", dataset="../x")
    payload = _call(
        server, "fidopt_run", operation="check", method="deeds", dataset="AMOS_CTCT", cases=["a"], scout="mock"
    )
    assert isinstance(payload, dict) and payload["kind"] == "fidopt"
    done = _wait(server, payload["job_id"])
    assert done["status"] == "done", done
    called = json.loads(
        (Path(os.environ["KONFAI_MCP_WORKSPACES_ROOT"]) / "sessions" / "default" / "fidopt_called.json").read_text()
    )
    assert called == ["check", "deeds", "--on", "AMOS_CTCT", "a", "--scout", "mock"]
    for bad in (["--out=/tmp/x"], ["/etc/passwd"], ["../../x"], ["--machine", ".."]):
        with pytest.raises(ValueError, match="no paths"):
            server.FIDOPT.prepare_run("check", "deeds", "", None, None, False, bad)
    spec = server.FIDOPT.prepare_run("check", "deeds", "", None, None, False, ["--try", "a=b,c=1", "--family=f"])
    assert spec["kwargs"]["args"] == ["check", "deeds", "--try", "a=b,c=1", "--family=f"]


def test_the_adapter_says_when_fidopt_is_not_there(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from konfai_mcp.server_scout import FidoptAdapter
    from konfai_mcp.workspace import WorkspaceLayout

    monkeypatch.setenv("KONFAI_MCP_FIDOPT", "/nowhere/fidopt")
    monkeypatch.delenv("KONFAI_MCP_FIDOPT_ALLOWED", raising=False)
    adapter = FidoptAdapter(WorkspaceLayout(tmp_path))
    assert adapter.allowed == ("check",)
    with pytest.raises(ValueError, match="KONFAI_MCP_FIDOPT"):
        adapter.list_datasets()


def test_a_failed_inspection_fails_its_job(monkeypatch: pytest.MonkeyPatch) -> None:
    from konfai_mcp.runner import run_scout_inspection_api

    failed = SimpleNamespace(error="provider error: boom", summary=dict)
    fake = SimpleNamespace(
        InspectionCase=SimpleNamespace(load=str), ScoutConfig=SimpleNamespace(load=str), inspect=lambda *_: failed
    )
    monkeypatch.setitem(sys.modules, "konfai_scout", fake)
    with pytest.raises(RuntimeError, match="provider error: boom"):
        run_scout_inspection_api(case="c", config="k", output="o")
