# SPDX-License-Identifier: Apache-2.0
"""Tests for the review-driven fixes: directory-aware dataset scan, read-back tools,
metric-direction correction, run_resume, and honest job failure payloads."""

import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

fastmcp = pytest.importorskip("fastmcp")

from mcp_test_helpers import install_fake_konfai_runtime, yaml_dump  # noqa: E402


def _metric_json(metric_name: str, value: float) -> str:
    return json.dumps(
        {
            "case": {metric_name: {"CASE_000": value}},
            "aggregates": {
                metric_name: {
                    "max": value,
                    "min": value,
                    "std": 0.0,
                    "25pc": value,
                    "50pc": value,
                    "75pc": value,
                    "mean": value,
                    "count": 1.0,
                }
            },
        }
    )


def _write_run_metrics(workspace: Path, run_name: str, metric_name: str, value: float, split: str = "TRAIN") -> None:
    metrics = workspace / "Evaluations" / run_name / f"Metric_{split}.json"
    metrics.parent.mkdir(parents=True, exist_ok=True)
    metrics.write_text(_metric_json(metric_name, value), encoding="utf-8")


def test_directory_backed_dataset_entries_are_discovered(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    load_mcp_server: Callable[[], ModuleType],
) -> None:
    dataset_dir = tmp_path / "dataset"
    for idx in range(2):
        case_dir = dataset_dir / f"CASE_{idx:03d}"
        zarr_store = case_dir / "CT.ome.zarr"
        zarr_store.mkdir(parents=True)
        (zarr_store / ".zattrs").write_text("{}", encoding="utf-8")
        dicom_series = case_dir / "MR"
        dicom_series.mkdir()
        (dicom_series / "slice0001.dcm").write_bytes(b"\x00")
        (case_dir / "SEG.mha").write_bytes(b"\x00")
        (case_dir / "notes").mkdir()
        (case_dir / "notes" / "readme.txt").write_text("x", encoding="utf-8")

    monkeypatch.setenv("KONFAI_MCP_WORKSPACES_ROOT", str(tmp_path / "workspaces"))
    mcp_server = load_mcp_server()

    async def scenario() -> None:
        async with fastmcp.Client(mcp_server.mcp) as client:
            inferred = await client.call_tool(
                "inspect_dataset", {"dataset_dir": str(dataset_dir), "include_stats": False}
            )
            data = inferred.structured_content
            assert set(data["groups"]) == {"CT", "MR", "SEG"}
            assert data["groups"]["CT"]["extensions"] == ["zarr"]
            assert data["groups"]["MR"]["extensions"] == ["dicom"]
            assert data["groups"]["CT"]["count"] == 2

    asyncio.run(scenario())


def test_h5_internal_groups_are_discovered(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    load_mcp_server: Callable[[], ModuleType],
) -> None:
    h5py = pytest.importorskip("h5py")
    np = pytest.importorskip("numpy")

    dataset_dir = tmp_path / "dataset"
    for idx in range(2):
        case_dir = dataset_dir / f"CASE_{idx:03d}"
        case_dir.mkdir(parents=True)
        with h5py.File(case_dir / "data.h5", "w") as handle:
            handle.create_dataset("CT", data=np.zeros((2, 4, 4), dtype=np.float32))
            handle.create_dataset("SEG", data=np.zeros((2, 4, 4), dtype=np.uint8))

    monkeypatch.setenv("KONFAI_MCP_WORKSPACES_ROOT", str(tmp_path / "workspaces"))
    mcp_server = load_mcp_server()

    async def scenario() -> None:
        async with fastmcp.Client(mcp_server.mcp) as client:
            inferred = await client.call_tool(
                "inspect_dataset", {"dataset_dir": str(dataset_dir), "include_stats": False}
            )
            data = inferred.structured_content
            assert set(data["groups"]) == {"CT", "SEG"}
            assert data["groups"]["CT"]["extensions"] == ["h5"]

    asyncio.run(scenario())


def test_read_back_tools_and_workspace_jail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    load_mcp_server: Callable[[], ModuleType],
) -> None:
    monkeypatch.setenv("KONFAI_MCP_WORKSPACES_ROOT", str(tmp_path / "workspaces"))
    mcp_server = load_mcp_server()

    async def scenario() -> None:
        async with fastmcp.Client(mcp_server.mcp) as client:
            await client.call_tool("initialize_session", {"overwrite": True})
            await client.call_tool(
                "write_session_file",
                {"relative_path": "Loss.py", "content": "class MyLoss:\n    pass"},
            )

            read_back = await client.call_tool("read_session_file", {"path": "Loss.py"})
            data = read_back.structured_content
            assert "class MyLoss" in data["content"]
            assert data["relative_path"] == "Loss.py"
            assert data["truncated"] is False

            paged = await client.call_tool("read_session_file", {"path": "Loss.py", "max_chars": 5, "offset": 6})
            paged_data = paged.structured_content
            assert paged_data["content"] == "MyLos"
            assert paged_data["truncated"] is True

            absolute_inside = await client.call_tool("read_session_file", {"path": data["path"]})
            assert "class MyLoss" in absolute_inside.structured_content["content"]

            with pytest.raises(Exception, match="escapes the session workspace"):
                await client.call_tool("read_session_file", {"path": "../outside.txt"})

            template = await client.call_tool("read_template_file", {"name": "Segmentation", "filename": "Config.yml"})
            assert "Trainer" in template.structured_content["content"]

            with pytest.raises(Exception, match="Invalid template"):
                await client.call_tool("read_template_file", {"name": "../konfai-mcp", "filename": "README.md"})

            with pytest.raises(Exception, match="Invalid template filename"):
                await client.call_tool("read_template_file", {"name": "Segmentation", "filename": "../Config.yml"})

    asyncio.run(scenario())


def test_metric_direction_and_leaderboard_controls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    load_mcp_server: Callable[[], ModuleType],
) -> None:
    monkeypatch.setenv("KONFAI_MCP_WORKSPACES_ROOT", str(tmp_path / "workspaces"))
    mcp_server = load_mcp_server()

    # A criterion named DiceLoss must rank as minimize despite the 'dice' token.
    direction, source = mcp_server.SESSION._metric_direction("PRED:SEG:DiceLoss")
    assert direction == "min"
    assert source == "heuristic:min"
    assert mcp_server.SESSION._metric_direction("PRED:SEG:Dice")[0] == "max"

    async def scenario() -> None:
        async with fastmcp.Client(mcp_server.mcp) as client:
            await client.call_tool("initialize_session", {"overwrite": True})
            workspace = Path(mcp_server.WORKSPACE_LAYOUT.workspace_dir())
            _write_run_metrics(workspace, "RUN_A", "PRED:SEG:DiceLoss", 0.2)
            _write_run_metrics(workspace, "RUN_B", "PRED:SEG:DiceLoss", 0.5)

            board = await client.call_tool("leaderboard", {"metric": "DiceLoss"})
            board_data = board.structured_content
            assert board_data["best"]["run_name"] == "RUN_A"
            assert board_data["available_splits"] == ["TRAIN"]

            flipped = await client.call_tool("leaderboard", {"metric": "DiceLoss", "direction": "max"})
            assert flipped.structured_content["best"]["run_name"] == "RUN_B"

            run_metrics = await client.call_tool("get_run_metrics", {"run_name": "RUN_B"})
            run_data = run_metrics.structured_content
            assert run_data["metrics"]["case"]["PRED:SEG:DiceLoss"]["CASE_000"] == 0.5
            assert run_data["split"] == "TRAIN"

            with pytest.raises(Exception, match=r"Available runs: .*RUN_A"):
                await client.call_tool("get_run_metrics", {"run_name": "MISSING"})

            with pytest.raises(Exception, match="Available splits"):
                await client.call_tool("leaderboard", {"split": "TEST"})

    asyncio.run(scenario())


def test_run_resume_and_failed_job_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    load_mcp_server: Callable[[], ModuleType],
) -> None:
    monkeypatch.setenv("KONFAI_MCP_WORKSPACES_ROOT", str(tmp_path / "workspaces"))
    monkeypatch.setenv("KONFAI_MCP_FAKE_SLEEP_S", "0.05")
    mcp_server = load_mcp_server()
    install_fake_konfai_runtime(tmp_path, monkeypatch, mcp_server)

    config = yaml_dump({"Trainer": {"train_name": "FAKE_RUN"}})

    async def scenario() -> None:
        async with fastmcp.Client(mcp_server.mcp) as client:
            await client.call_tool("initialize_session", {"overwrite": True})
            await client.call_tool("write_workflow_config", {"workflow": "train", "content": config})

            with pytest.raises(Exception, match="No checkpoint found to resume from"):
                await client.call_tool("run_resume", {})

            workspace = Path(mcp_server.WORKSPACE_LAYOUT.workspace_dir())
            checkpoint = workspace / "Checkpoints" / "FAKE_RUN" / "epoch_0000.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_text("checkpoint", encoding="utf-8")

            resumed = await client.call_tool("run_resume", {"lr": 0.0005})
            resumed_payload = resumed.structured_content
            assert "RESUME" in resumed_payload["command"]
            done = await client.call_tool(
                "wait_for_job", {"job_id": resumed_payload["job_id"], "timeout_s": 60.0, "poll_interval_s": 0.1}
            )
            done_payload = done.structured_content
            assert done_payload["status"] == "done"
            assert "wait_for_job" not in done_payload["next_actions"]
            assert done_payload["next_resources"] == [f"job://{done_payload['job_id']}/log"]

            manifest = await client.read_resource(f"job://{resumed_payload['job_id']}/manifest")
            manifest_data = json.loads(manifest[0].text)
            assert manifest_data["manifest"]["resume_from"] == str(checkpoint)
            assert manifest_data["manifest"]["lr_override"] == 0.0005

            monkeypatch.setenv("KONFAI_MCP_FAKE_EXIT_CODE", "1")
            failed = await client.call_tool("run_train", {"overwrite": True})
            failed_payload = failed.structured_content
            failed_done = await client.call_tool(
                "wait_for_job", {"job_id": failed_payload["job_id"], "timeout_s": 60.0, "poll_interval_s": 0.1}
            )
            failed_data = failed_done.structured_content
            assert failed_data["status"] == "error"
            assert failed_data["error"], "a crashed job must state WHY in its payload"
            assert "read_job_log" in failed_data["next_actions"]
            assert "run_train" in failed_data["next_actions"]
            assert not any(
                str(action).startswith(("retry:", "read_resource:")) for action in failed_data["next_actions"]
            )

            # 'auto' must pick the console job log for a FAILED job (the traceback lives there).
            log = await client.call_tool(
                "read_job_log",
                {"job_id": failed_payload["job_id"], "grep": "simulated failure"},
            )
            log_data = log.structured_content
            assert "simulated failure" in log_data["content"]
            assert log_data["lines_returned"] >= 1

    asyncio.run(scenario())
