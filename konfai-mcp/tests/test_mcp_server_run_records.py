# SPDX-License-Identifier: Apache-2.0
"""Run records: export_run_record, diff_run_configs, compare_runs, and read_training_curves."""

import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

fastmcp = pytest.importorskip("fastmcp")

from mcp_test_helpers import install_fake_konfai_runtime, yaml_dump  # noqa: E402


def _metric_json(metric_name: str, values: dict[str, float]) -> str:
    mean = sum(values.values()) / len(values)
    return json.dumps(
        {
            "case": {metric_name: values},
            "aggregates": {
                metric_name: {
                    "mean": mean,
                    "min": min(values.values()),
                    "max": max(values.values()),
                    "std": 0.0,
                    "count": float(len(values)),
                }
            },
        }
    )


@pytest.mark.usefixtures("workspace_root")
def test_run_records_diffs_comparison_and_curves(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    load_mcp_server: Callable[[], ModuleType],
) -> None:
    monkeypatch.setenv("KONFAI_MCP_FAKE_SLEEP_S", "0.05")
    mcp_server = load_mcp_server()
    install_fake_konfai_runtime(tmp_path, monkeypatch, mcp_server)

    async def scenario() -> None:
        async with fastmcp.Client(mcp_server.mcp) as client:
            await client.call_tool("initialize_session", {"overwrite": True})
            await client.call_tool(
                "write_workflow_config",
                {"workflow": "train", "content": yaml_dump({"Trainer": {"train_name": "FAKE_RUN", "epochs": 1}})},
            )
            first = await client.call_tool("run_train", {})
            first_payload = first.structured_content
            await client.call_tool(
                "wait_for_job", {"job_id": first_payload["job_id"], "timeout_s": 60.0, "poll_interval_s": 0.1}
            )
            await client.call_tool(
                "write_workflow_config",
                {"workflow": "train", "content": yaml_dump({"Trainer": {"train_name": "FAKE_RUN", "epochs": 2}})},
            )
            second = await client.call_tool("run_train", {"overwrite": True})
            second_payload = second.structured_content
            await client.call_tool(
                "wait_for_job", {"job_id": second_payload["job_id"], "timeout_s": 60.0, "poll_interval_s": 0.1}
            )

            record = await client.call_tool("export_run_record", {"run_name": "FAKE_RUN"})
            record_data = record.structured_content
            assert record_data["job"]["job_id"] == second_payload["job_id"]
            assert record_data["manifest"]["environment"]["python"]
            assert "FAKE_RUN" in record_data["config_snapshots"]["Config.yml"]

            diff = await client.call_tool(
                "diff_run_configs",
                {"job_id_a": first_payload["job_id"], "job_id_b": second_payload["job_id"]},
            )
            diff_data = diff.structured_content
            assert diff_data["identical"] is False
            assert "epochs" in diff_data["diff"]

            # Two runs' metrics, aligned comparison (Dice: higher wins).
            workspace = Path(mcp_server.WORKSPACE_LAYOUT.workspace_dir())
            for run, values in (
                ("RUN_A", {"CASE_000": 0.5, "CASE_001": 0.6}),
                ("RUN_B", {"CASE_000": 0.7, "CASE_001": 0.55}),
            ):
                metrics = workspace / "Evaluations" / run / "Metric_TRAIN.json"
                metrics.parent.mkdir(parents=True, exist_ok=True)
                metrics.write_text(_metric_json("PRED:SEG:Dice", values), encoding="utf-8")
            compared = await client.call_tool("compare_runs", {"run_a": "RUN_A", "run_b": "RUN_B"})
            comparison = compared.structured_content["metrics"]["PRED:SEG:Dice"]
            assert comparison["direction"] == "max"
            assert comparison["cases"] == 2
            assert comparison["cases_better_b"] == 1 and comparison["cases_better_a"] == 1
            assert comparison["winner"] == "tie"

    asyncio.run(scenario())


@pytest.mark.usefixtures("workspace_root")
def test_read_training_curves_from_tfevents(
    load_mcp_server: Callable[[], ModuleType],
) -> None:
    pytest.importorskip("tensorboard")
    torch_utils = pytest.importorskip("torch.utils.tensorboard")

    mcp_server = load_mcp_server()

    async def scenario() -> None:
        async with fastmcp.Client(mcp_server.mcp) as client:
            await client.call_tool("initialize_session", {"overwrite": True})
            workspace = Path(mcp_server.WORKSPACE_LAYOUT.workspace_dir())
            writer = torch_utils.SummaryWriter(log_dir=str(workspace / "Statistics" / "CURVE_RUN"))
            for step in range(10):
                writer.add_scalar("Loss/Dice", 1.0 / (step + 1), step)
            writer.close()

            curves = await client.call_tool("read_training_curves", {"run_name": "CURVE_RUN"})
            payload: dict[str, Any] = curves.structured_content
            assert payload["tags"], payload
            tag = payload["tags"][0]
            series = payload["curves"][tag]
            assert len(series) >= 2
            assert series[0]["value"] > series[-1]["value"]

            with pytest.raises(Exception, match="Available runs"):
                await client.call_tool("read_training_curves", {"run_name": "MISSING"})

    asyncio.run(scenario())
