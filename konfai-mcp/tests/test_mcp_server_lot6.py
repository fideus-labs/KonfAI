# SPDX-License-Identifier: Apache-2.0
"""Lot 6: sequential batch runs, cross-validation folds, and volume previews."""

import asyncio
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

fastmcp = pytest.importorskip("fastmcp")

from mcp_test_helpers import install_fake_konfai_runtime, yaml_dump  # noqa: E402


def test_generate_folds_and_run_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    load_mcp_server: Callable[[], ModuleType],
) -> None:
    dataset_dir = tmp_path / "dataset"
    for index in range(4):
        case_dir = dataset_dir / f"CASE_{index:03d}"
        case_dir.mkdir(parents=True)
        (case_dir / "CT.mha").write_bytes(b"\x00")

    monkeypatch.setenv("KONFAI_MCP_WORKSPACES_ROOT", str(tmp_path / "workspaces"))
    monkeypatch.setenv("KONFAI_MCP_FAKE_SLEEP_S", "0.05")
    mcp_server = load_mcp_server()
    install_fake_konfai_runtime(tmp_path, monkeypatch, mcp_server)

    async def scenario() -> None:
        async with fastmcp.Client(mcp_server.mcp) as client:
            await client.call_tool("initialize_session", {"overwrite": True})

            folds = await client.call_tool("generate_folds", {"dataset_dir": str(dataset_dir), "k": 2, "seed": 1})
            folds_data = folds.structured_content
            assert folds_data["k"] == 2 and folds_data["total_cases"] == 4
            members = [case for fold in folds_data["folds"].values() for case in fold["cases"]]
            assert sorted(members) == [f"CASE_{i:03d}" for i in range(4)]
            fold_0 = folds_data["folds"]["fold_0"]
            assert fold_0["train_subset"] == "~folds/fold_0.txt"
            assert Path(fold_0["file"]).read_text(encoding="utf-8").strip().splitlines() == fold_0["cases"]

            with pytest.raises(Exception, match="cannot make k="):
                await client.call_tool("generate_folds", {"dataset_dir": str(dataset_dir), "k": 5})

            # A sequential two-config batch (fake runtime): both runs complete in order.
            await client.call_tool(
                "write_workflow_config",
                {"workflow": "train", "content": yaml_dump({"Trainer": {"train_name": "SWEEP_BASE"}})},
            )
            for name in ("FOLD_A", "FOLD_B"):
                await client.call_tool(
                    "write_session_file",
                    {"relative_path": f"Config_{name}.yml", "content": yaml_dump({"Trainer": {"train_name": name}})},
                )
            batch = await client.call_tool("run_batch", {"config_files": ["Config_FOLD_A.yml", "Config_FOLD_B.yml"]})
            batch_data = batch.structured_content
            assert batch_data["requested"] == 2 and batch_data["completed"] == 2
            assert [result["run_name"] for result in batch_data["results"]] == ["FOLD_A", "FOLD_B"]
            assert all(result["status"] == "done" for result in batch_data["results"])

            missing = await client.call_tool(
                "run_batch", {"config_files": ["Config_MISSING.yml"], "stop_on_error": True}
            )
            missing_data = missing.structured_content
            assert missing_data["completed"] == 0
            assert missing_data["results"][0]["status"] == "launch_error"

    asyncio.run(scenario())


def test_preview_volume_returns_png(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    load_mcp_server: Callable[[], ModuleType],
) -> None:
    pytest.importorskip("SimpleITK")
    from mcp_test_helpers import create_segmentation_dataset

    dataset_dir = tmp_path / "dataset"
    create_segmentation_dataset(dataset_dir)
    monkeypatch.setenv("KONFAI_MCP_WORKSPACES_ROOT", str(tmp_path / "workspaces"))
    mcp_server = load_mcp_server()

    async def scenario() -> None:
        async with fastmcp.Client(mcp_server.mcp) as client:
            volume = dataset_dir / "CASE_000" / "CT.mha"
            preview = await client.call_tool("preview_volume", {"path": str(volume)})
            image_blocks = [block for block in preview.content if getattr(block, "type", "") == "image"]
            assert image_blocks, preview.content
            assert image_blocks[0].mimeType == "image/png"
            assert len(image_blocks[0].data) > 100

            with pytest.raises(Exception, match="Volume file not found"):
                await client.call_tool("preview_volume", {"path": str(dataset_dir / "nope.mha")})

    asyncio.run(scenario())
