# SPDX-License-Identifier: Apache-2.0
"""Sessions, experiment import, cross-session metrics, device-scoped job concurrency,
alternate train configs, and SLURM passthrough."""

import asyncio
import json
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest

fastmcp = pytest.importorskip("fastmcp")

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from mcp_test_helpers import install_fake_konfai_runtime, yaml_dump  # noqa: E402


def _metric_json(metric_name: str, value: float) -> str:
    return json.dumps(
        {
            "case": {metric_name: {"CASE_000": value}},
            "aggregates": {metric_name: {"mean": value, "min": value, "max": value, "std": 0.0, "count": 1.0}},
        }
    )


class _ProcAlive:
    pid = 4242

    def is_alive(self) -> bool:
        return True


def test_sessions_create_switch_and_cross_session_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    load_mcp_server: Callable[[], ModuleType],
) -> None:
    monkeypatch.setenv("KONFAI_MCP_WORKSPACES_ROOT", str(tmp_path / "workspaces"))
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


def test_import_experiment_links_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    load_mcp_server: Callable[[], ModuleType],
) -> None:
    source = tmp_path / "old_experiment"
    (source / "Checkpoints" / "OLD_RUN").mkdir(parents=True)
    (source / "Checkpoints" / "OLD_RUN" / "best.pt").write_text("ckpt", encoding="utf-8")
    (source / "Config.yml").write_text(yaml_dump({"Trainer": {"train_name": "OLD_RUN"}}), encoding="utf-8")
    (source / "Model.py").write_text("class MyNet:\n    pass\n", encoding="utf-8")
    (source / "notes.md").write_text("ignored", encoding="utf-8")

    monkeypatch.setenv("KONFAI_MCP_WORKSPACES_ROOT", str(tmp_path / "workspaces"))
    mcp_server = load_mcp_server()

    async def scenario() -> None:
        async with fastmcp.Client(mcp_server.mcp) as client:
            await client.call_tool("initialize_session", {"overwrite": True})
            imported = await client.call_tool("import_experiment", {"source_dir": str(source)})
            payload = imported.structured_content
            assert sorted(payload["copied"]) == ["Config.yml", "Model.py"]
            assert payload["linked"] == ["Checkpoints"]

            workspace = Path(mcp_server.WORKSPACE_LAYOUT.workspace_dir())
            assert (workspace / "Checkpoints").is_symlink()
            assert (workspace / "Checkpoints" / "OLD_RUN" / "best.pt").read_text(encoding="utf-8") == "ckpt"

            read_back = await client.call_tool("read_session_file", {"path": "Config.yml"})
            assert "OLD_RUN" in read_back.structured_content["content"]

            again = await client.call_tool("import_experiment", {"source_dir": str(source)})
            assert "Config.yml" in again.structured_content["skipped"]

    asyncio.run(scenario())


def test_device_scoped_job_concurrency() -> None:
    from konfai_mcp.server_jobs import Job, JobRegistry

    registry = JobRegistry({"queued", "running"})
    active = Job(
        job_id="gpu0",
        session="default",
        kind="train",
        command=["fake"],
        cwd=Path("/tmp"),
        log_path=Path("/tmp/job.log"),
        config_path=Path("/tmp/Config.yml"),
        status="running",
        devices=["0"],
    )
    active.proc = cast(Any, _ProcAlive())
    registry.jobs[active.job_id] = active

    assert registry.find_device_conflicts(["0", "1"]) == [active]
    assert registry.find_device_conflicts(["1"]) == []
    assert registry.find_device_conflicts(["cpu"]) == []
    # Unknown device sets conflict with everything (safe default).
    assert registry.find_device_conflicts(None) == [active]

    with pytest.raises(RuntimeError, match="conflicting device"):
        registry.launch(
            session="default",
            kind="train",
            command=["fake"],
            cwd=Path("/tmp"),
            log_path=Path("/tmp/job2.log"),
            config_path=Path("/tmp/Config.yml"),
            devices=["0"],
            target="mcp_test_helpers:_fake_job_runtime",
        )


def test_run_train_config_file_and_cluster(
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

            with pytest.raises(Exception, match="must define the 'Trainer' root key"):
                await client.call_tool(
                    "write_session_file",
                    {"relative_path": "Bad.yml", "content": yaml_dump({"Predictor": {}})},
                )
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
