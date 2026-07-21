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

"""set_live_tunables writes a jailed, revisioned control file for a running training job."""

import asyncio
import importlib
import json
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import fastmcp
import pytest


class _Running:
    """A process handle that reads as still running (poll() -> None)."""

    def poll(self) -> None:
        return None


def test_set_live_tunables_writes_a_revisioned_control_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    load_mcp_server: Callable[[], ModuleType],
) -> None:
    monkeypatch.setenv("KONFAI_MCP_WORKSPACES_ROOT", str(tmp_path / "workspaces"))
    mcp_server = load_mcp_server()
    job_cls = importlib.import_module("konfai_mcp.server_jobs").Job

    async def scenario() -> None:
        async with fastmcp.Client(mcp_server.mcp) as client:
            created = await client.call_tool("initialize_session", {"overwrite": True})
            session_dir = Path(created.structured_content["path"])

            # No running training job yet -> the tool refuses without writing anything.
            none_yet = await client.call_tool("set_live_tunables", {"it_validation": 5})
            assert none_yet.structured_content["ok"] is False

            run_dir = session_dir / "Statistics" / "RUN"
            run_dir.mkdir(parents=True)
            job = job_cls(
                job_id="job1",
                session="default",
                kind="train",
                command=["train"],
                cwd=session_dir,
                log_path=session_dir / "job.log",
                config_path=session_dir / "Config.yml",
                status="running",
                runtime_log_path=run_dir / "log_0.txt",
            )
            job.proc = cast(Any, _Running())
            mcp_server.JOB_REGISTRY.jobs[job.job_id] = job

            result = await client.call_tool("set_live_tunables", {"lr": 1e-4, "it_validation": 5})
            payload = result.structured_content
            assert payload["ok"] is True
            assert payload["revision"] == 1
            assert payload["applied"] == {"lr": 1e-4, "it_validation": 5}
            assert json.loads((run_dir / "control.json").read_text(encoding="utf-8")) == {
                "revision": 1,
                "lr": 1e-4,
                "it_validation": 5,
            }

            # A second call bumps the revision so the trainer applies it once more.
            again = await client.call_tool("set_live_tunables", {"lr": 5e-5})
            assert again.structured_content["revision"] == 2

            # next_actions must be real, registered tool names (anti-drift for this tool's own payload).
            registered = {tool.name for tool in await client.list_tools()}
            assert set(payload["next_actions"]) <= registered

    asyncio.run(scenario())


def test_set_live_tunables_needs_at_least_one_field(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    load_mcp_server: Callable[[], ModuleType],
) -> None:
    monkeypatch.setenv("KONFAI_MCP_WORKSPACES_ROOT", str(tmp_path / "workspaces"))
    mcp_server = load_mcp_server()

    async def scenario() -> None:
        async with fastmcp.Client(mcp_server.mcp) as client:
            await client.call_tool("initialize_session", {"overwrite": True})
            empty = await client.call_tool("set_live_tunables", {})
            assert empty.structured_content["ok"] is False

    asyncio.run(scenario())
