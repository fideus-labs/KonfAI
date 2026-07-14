# SPDX-License-Identifier: Apache-2.0
"""Anti-drift guards: the tool index is generated from the registry, and every
next_actions token emitted by payload builders is a callable registered tool."""

import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

fastmcp = pytest.importorskip("fastmcp")


def _registered_names(mcp_server: ModuleType) -> tuple[set[str], set[str]]:
    async def scenario() -> tuple[set[str], set[str]]:
        async with fastmcp.Client(mcp_server.mcp) as client:
            tools = {tool.name for tool in await client.list_tools()}
            prompts = {prompt.name for prompt in await client.list_prompts()}
            return tools, prompts

    return asyncio.run(scenario())


def test_tool_index_is_generated_from_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    load_mcp_server: Callable[[], ModuleType],
) -> None:
    monkeypatch.setenv("KONFAI_MCP_WORKSPACES_ROOT", str(tmp_path / "workspaces"))
    mcp_server = load_mcp_server()
    tool_names, prompt_names = _registered_names(mcp_server)

    async def scenario() -> None:
        async with fastmcp.Client(mcp_server.mcp) as client:
            index = await client.read_resource("guide://tool-index")
            payload = json.loads("\n".join(getattr(item, "text", str(item)) for item in index))
            assert payload["generated_from_registry"] is True
            assert set(payload["tools"]) == tool_names
            assert set(payload["prompts"]) == prompt_names
            assert all(description for description in payload["tools"].values())

    asyncio.run(scenario())


def test_job_payload_next_actions_are_registered_tools(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    load_mcp_server: Callable[[], ModuleType],
) -> None:
    monkeypatch.setenv("KONFAI_MCP_WORKSPACES_ROOT", str(tmp_path / "workspaces"))
    mcp_server = load_mcp_server()
    tool_names, _ = _registered_names(mcp_server)

    from konfai_mcp.server_jobs import Job, JobRegistry

    registry = JobRegistry({"queued", "running"})
    for kind in ("train", "prediction", "evaluation", "infer", "finetune", "evaluate", "uncertainty", "pipeline"):
        for status in ("running", "done", "error", "killed"):
            job = Job(
                job_id=f"{kind}-{status}",
                session="default",
                kind=kind,  # type: ignore[arg-type]
                command=["fake"],
                cwd=tmp_path,
                log_path=tmp_path / "job.log",
                config_path=tmp_path / "Config.yml",
                status=status,  # type: ignore[arg-type]
            )
            payload = registry.payload(job, lambda value: None)
            unknown = [action for action in payload["next_actions"] if action not in tool_names]
            assert not unknown, f"{kind}/{status} suggests unregistered tools: {unknown}"
            assert all(str(uri).startswith("job://") for uri in payload["next_resources"])


def test_design_strategy_asks_instead_of_ping_ponging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    load_mcp_server: Callable[[], ModuleType],
) -> None:
    dataset_dir = tmp_path / "dataset"
    case_dir = dataset_dir / "CASE_000"
    case_dir.mkdir(parents=True)
    (case_dir / "CT.mha").write_bytes(b"\x00")
    monkeypatch.setenv("KONFAI_MCP_WORKSPACES_ROOT", str(tmp_path / "workspaces"))
    mcp_server = load_mcp_server()

    async def scenario() -> None:
        async with fastmcp.Client(mcp_server.mcp) as client:
            planned = await client.call_tool(
                "design_config_strategy",
                {"task": "segmentation", "dataset_dir": str(dataset_dir)},
            )
            payload = planned.structured_content
            assert payload["unresolved_questions"]
            # An unambiguous single-root dataset must NOT loop back to dataset inspection:
            # the agent should ask the user, then re-call with group_roles/workflows.
            assert payload["next_actions"] == ["design_config_strategy"]
            assert "Ask the user" in payload["how_to_resolve_questions"]

    asyncio.run(scenario())
