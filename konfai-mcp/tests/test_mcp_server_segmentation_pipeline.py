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

import asyncio
import io
import json
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from mcp_test_helpers import create_segmentation_dataset, run_job, wait_for_live_metric
from ruamel.yaml import YAML

fastmcp = pytest.importorskip("fastmcp")
pytest.importorskip("SimpleITK")

yaml = YAML()
yaml.default_flow_style = False


def _yaml_dump(data: dict[str, Any]) -> str:
    stream = io.StringIO()
    yaml.dump(data, stream)
    return stream.getvalue()


def _yaml_load(path: Path) -> dict[str, Any]:
    data = yaml.load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise AssertionError(f"Expected a YAML mapping in {path}")
    return data


def _seg_train_config(session_dir: Path, in_channels: int) -> str:
    cfg = _yaml_load(session_dir / "Config.yml")
    t = cfg["Trainer"]
    t["train_name"] = "SEG_DRYRUN"
    t["epochs"] = 1
    t["Dataset"]["Patch"]["patch_size"] = [1, 32, 32]
    t["Dataset"]["batch_size"] = 64
    t["Dataset"]["use_cache"] = False
    t["Dataset"]["inline_augmentations"] = False
    t["Dataset"]["augmentations"] = "None"
    t["Dataset"]["validation"] = 0.25
    t["Model"]["UNet"]["parameters"]["channels"] = [in_channels, 4, 8, 16, 32]
    return _yaml_dump(cfg)


def test_train_step_validation_catches_runtime_only_config_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    load_mcp_server: Callable[[], ModuleType],
) -> None:
    # A config with in_channels=2 but a 1-channel image builds and sets up fine, and only crashes at
    # the first forward. level="setup" must pass it; level="train_step" must catch it BEFORE any job.
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

            await client.call_tool(
                "write_workflow_config",
                {"workflow": "train", "content": _seg_train_config(session_dir, in_channels=2)},
            )
            setup = await client.call_tool("validate_config_semantics", {"workflow": "train", "level": "setup"})
            assert setup.structured_content["ok"] is True  # setup does not run the model

            step = await client.call_tool("validate_config_semantics", {"workflow": "train", "level": "train_step"})
            assert step.structured_content["ok"] is False  # the forward catches the channel mismatch
            assert "channel" in (step.structured_content.get("error") or "").lower()

            # A correct config passes the train-step dry-run (forward runs).
            await client.call_tool(
                "write_workflow_config",
                {"workflow": "train", "content": _seg_train_config(session_dir, in_channels=1)},
            )
            ok_step = await client.call_tool("validate_config_semantics", {"workflow": "train", "level": "train_step"})
            assert ok_step.structured_content["ok"] is True
            assert ok_step.structured_content["train_step"]["forward"] is True

    asyncio.run(scenario())


def _contains_metric_name(payload: Any, metric_name: str) -> bool:
    if isinstance(payload, dict):
        return any(
            metric_name in str(key) or _contains_metric_name(value, metric_name) for key, value in payload.items()
        )
    if isinstance(payload, list):
        return any(_contains_metric_name(item, metric_name) for item in payload)
    return metric_name in str(payload)


def test_mcp_server_segmentation_template_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    load_mcp_server: Callable[[], ModuleType],
) -> None:
    workspace_root = tmp_path / "workspaces"
    monkeypatch.setenv("KONFAI_MCP_WORKSPACES_ROOT", str(workspace_root))

    mcp_server = load_mcp_server()
    client_cls = fastmcp.Client

    async def scenario() -> None:
        async with client_cls(mcp_server.mcp) as client:
            created = await client.call_tool(
                "initialize_session",
                {
                    "from_example": "Segmentation",
                    "overwrite": True,
                    "workflows": ["train", "prediction", "evaluation"],
                },
            )
            session_dir = Path(created.structured_content["path"])
            create_segmentation_dataset(session_dir / "Dataset")

            train_cfg = _yaml_load(session_dir / "Config.yml")
            train_cfg["Trainer"]["train_name"] = "SEG_SMOKE"
            train_cfg["Trainer"]["epochs"] = 2
            train_cfg["Trainer"]["it_validation"] = 1
            train_cfg["Trainer"]["manual_seed"] = 0
            train_cfg["Trainer"]["Dataset"]["Patch"]["patch_size"] = [1, 32, 32]
            train_cfg["Trainer"]["Dataset"]["batch_size"] = 64
            train_cfg["Trainer"]["Dataset"]["use_cache"] = False
            train_cfg["Trainer"]["Dataset"]["inline_augmentations"] = False
            train_cfg["Trainer"]["Dataset"]["augmentations"] = "None"
            train_cfg["Trainer"]["Dataset"]["validation"] = 0.25
            train_cfg["Trainer"]["Model"]["UNet"]["parameters"]["channels"] = [1, 4, 8, 16, 32]
            train_cfg["Trainer"]["Model"]["UNet"]["schedulers"]["StepLR"]["step_size"] = 4

            prediction_cfg = _yaml_load(session_dir / "Prediction.yml")
            prediction_cfg["Predictor"]["train_name"] = "SEG_SMOKE"
            prediction_cfg["Predictor"]["Dataset"]["Patch"]["patch_size"] = [1, 32, 32]
            prediction_cfg["Predictor"]["Dataset"]["batch_size"] = 64
            prediction_cfg["Predictor"]["Model"]["UNet"]["parameters"]["channels"] = [1, 4, 8, 16, 32]
            # The shipped example declares ModelPatch: None (no internal tiling); this smoke test
            # patches at 32x32, so it must REPLACE the value, not mutate into the 'None' marker.
            prediction_cfg["Predictor"]["Model"]["UNet"]["ModelPatch"] = {"patch_size": [32, 32]}

            evaluation_cfg = _yaml_load(session_dir / "Evaluation.yml")
            evaluation_cfg["Evaluator"]["train_name"] = "SEG_SMOKE"
            evaluation_cfg["Evaluator"]["Dataset"]["dataset_filenames"][1] = "./Predictions/SEG_SMOKE/Dataset:i:mha"

            await client.call_tool("write_workflow_config", {"workflow": "train", "content": _yaml_dump(train_cfg)})
            await client.call_tool(
                "write_workflow_config",
                {"workflow": "prediction", "content": _yaml_dump(prediction_cfg)},
            )
            await client.call_tool(
                "write_workflow_config",
                {"workflow": "evaluation", "content": _yaml_dump(evaluation_cfg)},
            )

            config_before_validation = (session_dir / "Config.yml").read_text(encoding="utf-8")
            validation = await client.call_tool(
                "validate_config_semantics",
                {"workflow": "train", "level": "instantiate"},
            )
            assert validation.structured_content["ok"] is True
            # Validation must be side-effect-free on the agent's authored config.
            assert (session_dir / "Config.yml").read_text(encoding="utf-8") == config_before_validation

            train_job = await client.call_tool(
                "run_train",
                {"cpu": 1, "overwrite": True, "quiet": True, "single_process": True},
            )
            await wait_for_live_metric(
                client,
                train_job.structured_content["job_id"],
                lambda stage_metrics: any(
                    key.endswith(":CrossEntropyLoss") for key in stage_metrics.get("flat_metrics", {})
                ),
                max_entries=5,
            )
            done = await client.call_tool(
                "wait_for_job",
                {"job_id": train_job.structured_content["job_id"], "timeout_s": 180.0, "poll_interval_s": 0.2},
            )
            assert done.structured_content["status"] == "done"

            prediction_validation = await client.call_tool(
                "validate_config_semantics",
                {"workflow": "prediction", "level": "instantiate"},
            )
            assert prediction_validation.structured_content["ok"] is True
            await run_job(
                client, "run_prediction", {"cpu": 1, "overwrite": True, "quiet": True, "single_process": True}
            )

            evaluation_validation = await client.call_tool(
                "validate_config_semantics",
                {"workflow": "evaluation", "level": "instantiate"},
            )
            assert evaluation_validation.structured_content["ok"] is True
            await run_job(
                client, "run_evaluation", {"cpu": 1, "overwrite": True, "quiet": True, "single_process": True}
            )

            predicted = sorted((session_dir / "Predictions" / "SEG_SMOKE" / "Dataset").rglob("PRED.mha"))
            assert predicted

            metrics_path = session_dir / "Evaluations" / "SEG_SMOKE" / "Metric_TRAIN.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            assert metrics
            assert _contains_metric_name(metrics, "Dice")

            # The evaluator declares each metric's direction (from the criterion's `maximize`
            # property); Dice is higher-is-better, so it must be tagged "max", not guessed.
            directions = metrics.get("directions", {})
            dice_keys = [name for name in directions if name.split(":")[-1].startswith("Dice") or name.endswith("Dice")]
            assert dice_keys, f"no Dice direction declared in {list(directions)}"
            assert all(directions[name] == "max" for name in dice_keys)

            # The leaderboard consumes that declaration instead of the name heuristic.
            board = await client.call_tool("leaderboard", {"split": "TRAIN"})
            boards = board.structured_content.get("leaderboards", {})
            declared_rows = [
                row for rows in boards.values() for row in rows if "Dice" in row["metric"] and row["direction"] == "max"
            ]
            assert declared_rows and all(row["direction_source"] == "declared" for row in declared_rows)

            summary = await client.read_resource("session://current/summary")
            assert summary

    asyncio.run(scenario())
