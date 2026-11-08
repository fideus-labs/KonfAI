# SPDX-License-Identifier: Apache-2.0
"""What real Claude Code runs against the server asked for (see CHANGELOG): the frictions they met stay fixed."""

import asyncio
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

fastmcp = pytest.importorskip("fastmcp")


def test_the_first_tqdm_line_of_a_stage_carries_no_metric() -> None:
    from konfai_mcp.live_parse import parse_live_metric_line

    stale = "Validation : Loss (UNet(0.001000) : CrossEntropyLoss(1.00) : 0.690995 Dice(1.00) : nan):   0% 0/1 [00:00<?, ?it/s]"
    real = "Validation : Loss (UNet(0.001000) : CrossEntropyLoss(1.00) : 0.690042 Dice(1.00) : 0.042986): 100% 1/1 [00:00<00:00, 171.31it/s]"
    assert parse_live_metric_line(stale) is None
    entry = parse_live_metric_line(real)
    assert entry is not None and entry["stage"] == "Validation" and entry["progress"]["step"] == 1


@pytest.mark.usefixtures("workspace_root")
def test_a_relative_checkpoint_path_is_the_sessions(load_mcp_server: Callable[[], ModuleType]) -> None:
    mcp_server = load_mcp_server()
    workspace = mcp_server.WORKSPACE_LAYOUT.ensure_session_workspace()
    (resolved,) = mcp_server.SESSION.resolve_prediction_models(["Checkpoints/run/last.pt"])
    assert resolved == (workspace / "Checkpoints/run/last.pt").resolve()


@pytest.mark.usefixtures("workspace_root")
def test_a_session_is_created_and_seeded_in_one_call(load_mcp_server: Callable[[], ModuleType]) -> None:
    mcp_server = load_mcp_server()

    async def scenario() -> dict:
        async with fastmcp.Client(mcp_server.mcp) as client:
            created = await client.call_tool(
                "session", {"action": "create", "name": "seeded", "from_example": "Segmentation", "workflows": "train"}
            )
            return created.structured_content

    payload = asyncio.run(scenario())
    assert payload["initialized"]["seeded_from_example"] == "Segmentation"
    assert "Config.yml" in " ".join(payload["initialized"]["copied_files"])
    assert (Path(payload["initialized"]["path"]) / "Config.yml").is_file()


@pytest.mark.usefixtures("workspace_root")
def test_a_bare_builtin_name_is_inspected(load_mcp_server: Callable[[], ModuleType]) -> None:
    mcp_server = load_mcp_server()

    async def scenario() -> dict:
        async with fastmcp.Client(mcp_server.mcp) as client:
            return (await client.call_tool("inspect_object_signature", {"classpath": "Dice"})).structured_content

    payload = asyncio.run(scenario())
    assert payload["ok"] is True, payload.get("limitations")
    assert payload["resolved_classpath"].endswith(":Dice")
