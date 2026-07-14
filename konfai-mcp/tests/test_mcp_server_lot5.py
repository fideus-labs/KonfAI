# SPDX-License-Identifier: Apache-2.0
"""Lot 5: schema drill with literal YAML keys, label statistics, component smoke tests,
model-output enumeration, run records/diffs/curves/comparison."""

import asyncio
import io
import json
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from ruamel.yaml import YAML

fastmcp = pytest.importorskip("fastmcp")

from mcp_test_helpers import install_fake_konfai_runtime, yaml_dump  # noqa: E402

yaml = YAML()


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


def test_config_schema_yaml_keys_and_drill(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    load_mcp_server: Callable[[], ModuleType],
) -> None:
    monkeypatch.setenv("KONFAI_MCP_WORKSPACES_ROOT", str(tmp_path / "workspaces"))
    mcp_server = load_mcp_server()

    async def scenario() -> None:
        async with fastmcp.Client(mcp_server.mcp) as client:
            root = await client.call_tool("describe_config_schema", {"workflow": "train"})
            root_data = root.structured_content
            keys = {field["yaml_key"] for field in root_data["fields"]}
            assert "Model" in keys and "train_name" in keys

            drilled = await client.call_tool("describe_config_schema", {"workflow": "train", "path": "Model"})
            drilled_data = drilled.structured_content
            assert drilled_data["yaml_path"] == ["Trainer", "Model"]
            assert any(field["name"] == "classpath" for field in drilled_data["fields"])

            with pytest.raises(Exception, match="Drillable nested config keys"):
                await client.call_tool("describe_config_schema", {"workflow": "train", "path": "Nope"})

    asyncio.run(scenario())


def test_label_statistics_for_integer_groups(
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
            inspected = await client.call_tool(
                "inspect_dataset",
                {"dataset_dir": str(dataset_dir), "groups": ["SEG", "CT"], "extension": "mha"},
            )
            stats = inspected.structured_content["statistics"]
            assert stats["SEG"]["labels"]["unique"] == [0, 1]
            assert stats["SEG"]["labels"]["presence_cases"]["1"] == 4
            assert 0.0 < stats["SEG"]["labels"]["mean_voxel_fraction"]["1"] < 1.0
            # Float groups carry no label block.
            assert "labels" not in stats["CT"]

    asyncio.run(scenario())


def test_component_smoke_tests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    load_mcp_server: Callable[[], ModuleType],
) -> None:
    monkeypatch.setenv("KONFAI_MCP_WORKSPACES_ROOT", str(tmp_path / "workspaces"))
    mcp_server = load_mcp_server()

    good = (
        "from konfai.data.transform import Transform\n"
        "\n"
        "\n"
        "class Doubler(Transform):\n"
        "    def __init__(self) -> None:\n"
        "        super().__init__()\n"
        "\n"
        "    def transform_shape(self, group_src, name, shape, cache_attribute):\n"
        "        return list(shape)\n"
        "\n"
        "    def __call__(self, name, tensor, cache_attribute):\n"
        "        return tensor * 2\n"
    )
    bad = good.replace("class Doubler", "class Liar").replace("return list(shape)", "return [s + 1 for s in shape]")

    async def scenario() -> None:
        async with fastmcp.Client(mcp_server.mcp) as client:
            await client.call_tool("initialize_session", {"overwrite": True})
            await client.call_tool("write_session_file", {"relative_path": "MyTransform.py", "content": good})
            await client.call_tool("write_session_file", {"relative_path": "BadTransform.py", "content": bad})

            ok = await client.call_tool(
                "run_component_smoke_test", {"classpath": "MyTransform:Doubler", "kind": "transform"}
            )
            ok_data = ok.structured_content
            assert ok_data["ok"] is True
            assert ok_data["predicted_shape"] == ok_data["actual_shape"]

            broken = await client.call_tool(
                "run_component_smoke_test", {"classpath": "BadTransform:Liar", "kind": "transform"}
            )
            broken_data = broken.structured_content
            assert broken_data["ok"] is False
            assert broken_data["predicted_shape"] != broken_data["actual_shape"]
            assert "write_session_file" in broken_data["next_actions"]

            loss = await client.call_tool(
                "run_component_smoke_test", {"classpath": "torch.nn:MSELoss", "kind": "criterion"}
            )
            loss_data = loss.structured_content
            assert loss_data["ok"] is True
            assert loss_data["behaves_as"] == "loss"
            assert loss_data["backward_ok"] is True

    asyncio.run(scenario())


def test_describe_model_outputs_enumerates_module_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    load_mcp_server: Callable[[], ModuleType],
) -> None:
    pytest.importorskip("SimpleITK")
    from mcp_test_helpers import create_segmentation_dataset

    monkeypatch.setenv("KONFAI_MCP_WORKSPACES_ROOT", str(tmp_path / "workspaces"))
    mcp_server = load_mcp_server()

    async def scenario() -> None:
        async with fastmcp.Client(mcp_server.mcp) as client:
            created = await client.call_tool(
                "initialize_session",
                {"from_example": "Segmentation", "overwrite": True, "workflows": ["train"]},
            )
            session_dir = Path(created.structured_content["path"])
            create_segmentation_dataset(session_dir / "Dataset")
            config = yaml.load((session_dir / "Config.yml").read_text(encoding="utf-8"))
            config["Trainer"]["Dataset"]["Patch"]["patch_size"] = [1, 32, 32]
            config["Trainer"]["Model"]["UNet"]["parameters"]["channels"] = [1, 4, 8, 16, 32]
            stream = io.StringIO()
            yaml.dump(config, stream)
            await client.call_tool(
                "write_workflow_config", {"workflow": "train", "content": stream.getvalue(), "overwrite": True}
            )

            outputs = await client.call_tool("describe_model_outputs", {"workflow": "train"})
            payload = outputs.structured_content
            assert payload["ok"] is True
            networks = payload["networks"]
            assert networks, "at least one Network must be discovered"
            paths = [entry["path"] for entries in networks.values() for entry in entries]
            assert paths and any("." in path for path in paths)
            assert any(entry["terminal"] for entries in networks.values() for entry in entries)

    asyncio.run(scenario())


def test_run_records_diffs_comparison_and_curves(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    load_mcp_server: Callable[[], ModuleType],
) -> None:
    monkeypatch.setenv("KONFAI_MCP_WORKSPACES_ROOT", str(tmp_path / "workspaces"))
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


def test_read_training_curves_from_tfevents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    load_mcp_server: Callable[[], ModuleType],
) -> None:
    pytest.importorskip("tensorboard")
    torch_utils = pytest.importorskip("torch.utils.tensorboard")

    monkeypatch.setenv("KONFAI_MCP_WORKSPACES_ROOT", str(tmp_path / "workspaces"))
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
