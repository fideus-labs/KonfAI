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
import json
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

fastmcp = pytest.importorskip("fastmcp")
TestClient = pytest.importorskip("starlette.testclient").TestClient

from mcp_test_helpers import install_fake_konfai_runtime, yaml_dump  # noqa: E402

MINIMAL_TRAIN = """
Trainer:
  train_name: MCP_TRAIN
""".strip()

# A train config that references a dataset path which does not exist, used to assert
# "not ready" for a genuine reason (missing data) rather than relying on validation
# materialising defaults.
TRAIN_MISSING_DATASET = """
Trainer:
  train_name: MCP_TRAIN
  Dataset:
    dataset_filenames:
      - ./MissingDataset:a:mha
""".strip()

MINIMAL_PREDICTION = """
Predictor:
  train_name: MCP_TRAIN
  Dataset:
    groups_src:
      Labels:
        groups_dest:
          Labels:
            transforms: None
            patch_transforms: None
            is_input: true
  outputs_dataset:
    Model:
      OutputDataset:
        name_class: OutputDataset
        same_as_group: Labels:Labels
        dataset_filename: Dataset:mha
        group: Prediction
        before_reduction_transforms: None
        after_reduction_transforms: None
        final_transforms: None
        patch_combine: None
        reduction: Mean
""".strip()

MINIMAL_EVALUATION = """
Evaluator:
  train_name: MCP_TRAIN
""".strip()


def _metric_json(metric_name: str, value: float) -> str:
    return json.dumps(
        {
            "case": {metric_name: {"CASE_000": value}},
            "aggregates": {metric_name: {"mean": value, "min": value, "max": value, "std": 0.0, "count": 1.0}},
        }
    )


def test_mcp_server_session_loop(
    tmp_path: Path,
    workspace_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    load_mcp_server: Callable[[], ModuleType],
) -> None:
    mcp_server = load_mcp_server()
    monkeypatch.setattr(mcp_server, "KONFAI_VERSION", "test-version")
    monkeypatch.setattr(mcp_server, "konfai_get_available_devices", lambda: ([0], ["Fake GPU"]))
    monkeypatch.setattr(mcp_server, "konfai_get_ram", lambda: (4.0, 16.0))
    monkeypatch.setattr(mcp_server, "konfai_get_vram", lambda devices: (1.5, 8.0))
    client_cls = fastmcp.Client
    dataset_dir = tmp_path / "dataset" / "case_001"
    dataset_dir.mkdir(parents=True)
    for filename in ("CBCT.mha", "CT.mha", "MASK.mha"):
        (dataset_dir / filename).write_text("", encoding="utf-8")

    async def scenario() -> None:
        async with client_cls(mcp_server.mcp) as client:
            tools = await client.list_tools()
            tool_names = {tool.name for tool in tools}
            assert {
                "design_config_strategy",
                "browse_dataset",
                "inspect_dataset",
                "inspect_object_signature",
                "prepare_dataset_aliases",
                "leaderboard",
                "review_config_semantics",
                "summarize_session",
                "validate_config_semantics",
                "initialize_session",
                "write_session_file",
            }.issubset(tool_names)
            descriptions = {tool.name: tool.description for tool in tools}
            assert all(descriptions[name] for name in ("initialize_session", "run_train", "summarize_session"))

            info = await client.read_resource("server://info")
            info_payload = json.loads("\n".join(getattr(item, "text", str(item)) for item in info))
            assert info_payload["konfai_version"] == "test-version"
            assert info_payload["transport"] == "stdio"
            assert info_payload["current_session"] == "default"
            assert info_payload["session_root"] == str(workspace_root / "sessions" / "default")
            assert info_payload["auth"]["configured"] is False
            assert info_payload["auth"]["enforced_on_current_transport"] is False

            capabilities = await client.read_resource("server://capabilities")
            capabilities_payload = json.loads("\n".join(getattr(item, "text", str(item)) for item in capabilities))
            assert capabilities_payload["gpu"]["visible_names"] == ["Fake GPU"]
            assert capabilities_payload["recommended_device"] == {"gpu": [0]}

            tool_index = await client.read_resource("guide://tool-index")
            tool_index_payload = json.loads("\n".join(getattr(item, "text", str(item)) for item in tool_index))
            assert "browse_dataset" in tool_index_payload["tools"]
            assert "design_config_strategy" in tool_index_payload["tools"]
            assert "clarify_task_and_groups" in tool_index_payload["prompts"]

            prompts = await client.list_prompts()
            prompt_names = {prompt.name for prompt in prompts}
            assert {"clarify_task_and_groups", "plan_config_strategy", "debug_config_warning"}.issubset(prompt_names)

            templates = await client.read_resource("templates://list")
            templates_text = "\n".join(getattr(item, "text", str(item)) for item in templates)
            assert "Synthesis" in templates_text
            assert "Segmentation" in templates_text

            strategy = await client.call_tool(
                "design_config_strategy",
                {
                    "dataset_dir": str(dataset_dir.parent),
                    "task": "synthesis",
                    "group_roles": {"CBCT": "input", "CT": "target", "MASK": "support"},
                    "workflows": "train",
                    "modeling_intent": "2d",
                    "example": "Synthesis",
                },
            )
            strategy_data = strategy.structured_content
            assert strategy_data["task"] == "synthesis"
            assert strategy_data["group_roles"]["input"] == ["CBCT"]
            assert strategy_data["selected_example"]["name"] == "Synthesis"
            assert strategy_data["customization_options"]["can_write_local_components"] is True
            assert strategy_data["customization_options"]["signature_tool"] == "inspect_object_signature"

            created = await client.call_tool(
                "initialize_session",
                {"from_example": "Synthesis", "workflows": ["train"], "overwrite": True},
            )
            created_data = created.structured_content
            assert created_data["seeded_from_example"] == "Synthesis"
            assert created_data["session"] == "default"
            assert created_data["workflows"] == ["train"]

            session_dir = Path(created_data["path"])
            assert (session_dir / "Config.yml").exists()
            assert not (session_dir / "Prediction.yml").exists()
            assert not (session_dir / "Model.py").exists()

            inspected_object = await client.call_tool("inspect_object_signature", {"classpath": "Model:UNetpp5"})
            inspected_object_data = inspected_object.structured_content
            assert inspected_object_data["ok"] is False
            assert "write_session_file" in inspected_object_data["next_actions"]

            imported_object = await client.call_tool(
                "inspect_object_signature",
                {"classpath": "json.decoder.JSONDecoder"},
            )
            imported_object_data = imported_object.structured_content
            assert imported_object_data["ok"] is True
            assert imported_object_data["source"] == "imported"
            assert imported_object_data["signature"]
            assert any(parameter["name"] == "strict" for parameter in imported_object_data["parameters"])

            await client.call_tool("initialize_session", {"overwrite": True})
            await client.call_tool("write_workflow_config", {"workflow": "train", "content": TRAIN_MISSING_DATASET})
            await client.call_tool("write_workflow_config", {"workflow": "prediction", "content": MINIMAL_PREDICTION})
            await client.call_tool("write_workflow_config", {"workflow": "evaluation", "content": MINIMAL_EVALUATION})

            validation = await client.call_tool(
                "validate_config_semantics",
                {"workflow": "all", "level": "instantiate"},
            )
            validation_data = validation.structured_content
            assert validation_data["ok"] is False
            assert set(validation_data["results"]) == {"train", "prediction", "evaluation"}

            summary = await client.call_tool(
                "summarize_session", {"include_leaderboard": False, "include_validation": True}
            )
            summary_data = summary.structured_content
            assert summary_data["readiness"]["train"] is False
            assert "review_config_semantics" in summary_data["next_actions"]
            assert summary_data["validation"]["configs"]["Config.yml"]["yaml_valid"] is True

            summary_resource = await client.read_resource("session://current/summary")
            assert summary_resource

    asyncio.run(scenario())


@pytest.mark.usefixtures("workspace_root")
def test_sessions_create_switch_and_cross_session_metrics(
    load_mcp_server: Callable[[], ModuleType],
) -> None:
    mcp_server = load_mcp_server()

    async def scenario() -> None:
        async with fastmcp.Client(mcp_server.mcp) as client:
            await client.call_tool("initialize_session", {"overwrite": True})

            created = await client.call_tool("create_session", {"name": "exp-b"})
            payload = created.structured_content
            assert payload["session"] == "exp-b"
            assert payload["created"] is True
            assert mcp_server.WORKSPACE_LAYOUT.current_session == "exp-b"

            # Metrics written in exp-b are rankable from ANY session via session=...
            workspace = Path(mcp_server.WORKSPACE_LAYOUT.workspace_dir())
            metrics = workspace / "Evaluations" / "RUN_B" / "Metric_TRAIN.json"
            metrics.parent.mkdir(parents=True, exist_ok=True)
            metrics.write_text(_metric_json("PRED:SEG:Dice", 0.9), encoding="utf-8")

            switched = await client.call_tool("switch_session", {"name": "default"})
            assert switched.structured_content["session"] == "default"
            assert mcp_server.WORKSPACE_LAYOUT.current_session == "default"

            board = await client.call_tool("leaderboard", {"metric": "Dice", "session": "exp-b"})
            board_data = board.structured_content
            assert board_data["session"] == "exp-b"
            assert board_data["best"]["run_name"] == "RUN_B"

            run = await client.call_tool("get_run_metrics", {"run_name": "RUN_B", "session": "exp-b"})
            assert run.structured_content["metrics"]["case"]["PRED:SEG:Dice"]["CASE_000"] == 0.9

            with pytest.raises(Exception, match="Unknown session"):
                await client.call_tool("switch_session", {"name": "nope"})
            with pytest.raises(Exception, match="Unknown session"):
                await client.call_tool("leaderboard", {"session": "nope"})

    asyncio.run(scenario())


@pytest.mark.usefixtures("workspace_root")
def test_session_management_refused_when_host_pinned(
    monkeypatch: pytest.MonkeyPatch,
    load_mcp_server: Callable[[], ModuleType],
) -> None:
    """With KONFAI_MCP_SESSION set (a host UI pinned the workspace), session tools are refused.

    A session created or switched behind the host's back runs jobs in a workspace its panels
    never look at (seen live: a Studio chat whose runs landed in an invisible sibling session).
    """
    monkeypatch.setenv("KONFAI_MCP_SESSION", "pinned-exp")
    mcp_server = load_mcp_server()

    async def scenario() -> None:
        async with fastmcp.Client(mcp_server.mcp) as client:
            assert mcp_server.WORKSPACE_LAYOUT.current_session == "pinned-exp"
            for tool, args in (
                ("create_session", {"name": "elsewhere"}),
                ("switch_session", {"name": "default"}),
                ("delete_session", {}),
            ):
                with pytest.raises(Exception, match="pinned"):
                    await client.call_tool(tool, args)
            # Nothing was created behind the host's back.
            assert "elsewhere" not in mcp_server.WORKSPACE_LAYOUT.available_sessions()

    asyncio.run(scenario())


@pytest.mark.usefixtures("workspace_root")
def test_run_train_accepts_single_gpu_int(
    monkeypatch: pytest.MonkeyPatch,
    load_mcp_server: Callable[[], ModuleType],
) -> None:
    mcp_server = load_mcp_server()
    client_cls = fastmcp.Client
    captured: dict[str, Any] = {}

    def fake_launch(
        kind: str,
        command: list[str],
        config_path: Path,
        extra_manifest: dict[str, Any] | None = None,
        target: str | None = None,
        kwargs: dict[str, object] | None = None,
        devices: list[str] | None = None,
    ) -> dict[str, Any]:
        captured["kind"] = kind
        captured["command"] = command
        captured["extra_manifest"] = extra_manifest
        captured["kwargs"] = kwargs
        return {
            "job_id": "job_train_001",
            "status": "queued",
            "session": "default",
            "kind": kind,
            "next_actions": ["wait_for_job"],
        }

    monkeypatch.setattr(mcp_server, "_launch_job", fake_launch)

    async def scenario() -> None:
        async with client_cls(mcp_server.mcp) as client:
            await client.call_tool("initialize_session", {"overwrite": True})
            await client.call_tool("write_workflow_config", {"workflow": "train", "content": MINIMAL_TRAIN})
            launched = await client.call_tool("run_train", {"gpu": 0})
            launched_data = launched.structured_content
            assert launched_data["job_id"] == "job_train_001"
            assert launched_data["session"] == "default"
            assert captured["kind"] == "train"
            assert captured["kwargs"]["gpu"] == [0]
            assert captured["extra_manifest"]["devices"]["gpu"] == [0]
            assert "--gpu" in captured["command"]
            assert "0" in captured["command"]

    asyncio.run(scenario())


@pytest.mark.usefixtures("workspace_root")
def test_run_train_config_file_and_cluster(
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
                {"workflow": "train", "content": yaml_dump({"Trainer": {"train_name": "BASE"}})},
            )
            await client.call_tool(
                "write_session_file",
                {"relative_path": "Config_GAN.yml", "content": yaml_dump({"Trainer": {"train_name": "GAN_RUN"}})},
            )

            job = await client.call_tool("run_train", {"config_file": "Config_GAN.yml"})
            payload = job.structured_content
            assert payload["run_name"] == "GAN_RUN"
            done = await client.call_tool(
                "wait_for_job", {"job_id": payload["job_id"], "timeout_s": 60.0, "poll_interval_s": 0.1}
            )
            assert done.structured_content["status"] == "done"

            # Writing the bad file succeeds: validation happens at launch, not at write.
            await client.call_tool(
                "write_session_file",
                {"relative_path": "Bad.yml", "content": yaml_dump({"Predictor": {}})},
            )
            with pytest.raises(Exception, match="must define the 'Trainer' root key"):
                await client.call_tool("run_train", {"config_file": "Bad.yml"})

            with pytest.raises(Exception, match="cluster expects exactly the keys"):
                await client.call_tool("run_train", {"cluster": {"name": "gpu-queue"}})

            cluster = {"name": "gpu-queue", "memory": 64, "num_nodes": 1, "time_limit": 120}
            slurm_job = await client.call_tool("run_train", {"cluster": cluster, "overwrite": True})
            slurm_payload = slurm_job.structured_content
            assert slurm_payload["devices"] == ["slurm:gpu-queue"]
            await client.call_tool(
                "wait_for_job", {"job_id": slurm_payload["job_id"], "timeout_s": 60.0, "poll_interval_s": 0.1}
            )
            manifest = await client.read_resource(f"job://{slurm_payload['job_id']}/manifest")
            manifest_data = json.loads(manifest[0].text)
            assert manifest_data["manifest"]["cluster"] == cluster

    asyncio.run(scenario())


@pytest.mark.usefixtures("workspace_root")
def test_http_bearer_token_protects_http_transports(
    monkeypatch: pytest.MonkeyPatch,
    load_mcp_server: Callable[[], ModuleType],
) -> None:
    monkeypatch.setenv("KONFAI_MCP_TRANSPORT", "streamable-http")
    monkeypatch.setenv("KONFAI_MCP_BEARER_TOKEN", "dev-token")

    mcp_server = load_mcp_server()
    mcp_server._configure_transport_auth("streamable-http", host="127.0.0.1", port=8123, bearer_token="dev-token")
    app = mcp_server.mcp.http_app(transport="streamable-http", path="/mcp")

    with TestClient(app) as client:
        unauthorized = client.get("/mcp")
        assert unauthorized.status_code == 401
        assert unauthorized.headers["WWW-Authenticate"].startswith("Bearer")

        authorized = client.get("/mcp", headers={"Authorization": "Bearer dev-token"})
        assert authorized.status_code != 401
