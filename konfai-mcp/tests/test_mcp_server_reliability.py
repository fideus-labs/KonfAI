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
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest
from mcp_test_helpers import install_fake_konfai_runtime, resource_to_text, wait_for_live_metric

fastmcp = pytest.importorskip("fastmcp")

MINIMAL_TRAIN = """
Trainer:
  train_name: FAKE_RUN
""".strip()


async def _create_fake_train_session(client: object) -> None:
    await client.call_tool("initialize_session", {"overwrite": True})  # type: ignore[attr-defined]
    await client.call_tool(  # type: ignore[attr-defined]
        "write_workflow_config",
        {"workflow": "train", "content": MINIMAL_TRAIN},
    )


def test_mcp_server_rejects_invalid_inputs(
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
            with pytest.raises(Exception, match="valid boolean"):
                await client.call_tool("initialize_session", {"overwrite": "not-a-bool"})

            await client.call_tool("initialize_session", {"overwrite": True})

            with pytest.raises(Exception, match="escapes the session workspace"):
                await client.call_tool(
                    "write_session_file",
                    {
                        "relative_path": "../escape.txt",
                        "content": "blocked",
                    },
                )

            with pytest.raises(Exception, match=r"Config\.yml is not valid YAML"):
                await client.call_tool("write_workflow_config", {"workflow": "train", "content": "Trainer: ["})

            with pytest.raises(Exception, match="must define the 'Trainer' root key"):
                await client.call_tool(
                    "write_workflow_config",
                    {
                        "workflow": "train",
                        "content": "Predictor:\n  train_name: wrong\n",
                    },
                )

            with pytest.raises(Exception, match="destructive"):
                await client.call_tool(
                    "prepare_dataset_aliases",
                    {
                        "dataset_dir": str(tmp_path),
                        "rename_map": {"IMG": "CT"},
                        "mode": "move",
                    },
                )

    asyncio.run(scenario())


def test_mcp_server_overwrite_recreates_clean_session_workspace(
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
            await client.call_tool("initialize_session", {"overwrite": True})
            await client.call_tool(
                "write_session_file",
                {
                    "relative_path": "notes/stale.txt",
                    "content": "stale state",
                },
            )
            await client.call_tool(
                "initialize_session",
                {
                    "from_example": "Synthesis",
                    "overwrite": True,
                },
            )

            session_dir = workspace_root / "sessions" / "default"
            assert not (session_dir / "notes" / "stale.txt").exists()
            assert (session_dir / "Config.yml").exists()
            summary = await client.read_resource("session://current/summary")
            assert summary

    asyncio.run(scenario())


def test_mcp_server_serializes_concurrent_train_launches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    load_mcp_server: Callable[[], ModuleType],
) -> None:
    workspace_root = tmp_path / "workspaces"
    monkeypatch.setenv("KONFAI_MCP_WORKSPACES_ROOT", str(workspace_root))

    mcp_server = load_mcp_server()
    install_fake_konfai_runtime(tmp_path, monkeypatch, mcp_server)
    monkeypatch.setenv("KONFAI_MCP_FAKE_SLEEP_S", "0.8")
    client_cls = fastmcp.Client

    async def scenario() -> None:
        async with client_cls(mcp_server.mcp) as client:
            await _create_fake_train_session(client)

        async def launch() -> object:
            async with client_cls(mcp_server.mcp) as client:
                return await client.call_tool("run_train", {"cpu": 1})

        results = await asyncio.gather(launch(), launch(), return_exceptions=True)
        successes = [result for result in results if not isinstance(result, Exception)]
        failures = [result for result in results if isinstance(result, Exception)]

        assert len(successes) == 1
        assert len(failures) == 1
        assert "already has active job" in str(failures[0])

        async with client_cls(mcp_server.mcp) as client:
            job_id = successes[0].structured_content["job_id"]  # type: ignore[attr-defined]
            done = await client.call_tool("wait_for_job", {"job_id": job_id, "timeout_s": 10.0})
            assert done.structured_content["status"] == "done"
            manifest = await client.read_resource(f"job://{job_id}/manifest")
            manifest_text = resource_to_text(manifest)
            assert "config_snapshots" in manifest_text

    asyncio.run(scenario())


def test_mcp_server_timeout_cancel_and_live_metrics_are_stable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    load_mcp_server: Callable[[], ModuleType],
) -> None:
    workspace_root = tmp_path / "workspaces"
    monkeypatch.setenv("KONFAI_MCP_WORKSPACES_ROOT", str(workspace_root))

    mcp_server = load_mcp_server()
    install_fake_konfai_runtime(tmp_path, monkeypatch, mcp_server)
    monkeypatch.setenv("KONFAI_MCP_FAKE_SLEEP_S", "1.2")
    monkeypatch.setenv("KONFAI_MCP_FAKE_STEPS", "4")
    monkeypatch.setenv("KONFAI_MCP_FAKE_METRIC_NAME", "CrossEntropyLoss")
    client_cls = fastmcp.Client

    async def scenario() -> None:
        async with client_cls(mcp_server.mcp) as client:
            await _create_fake_train_session(client)
            started = await client.call_tool("run_train", {"cpu": 1})
            job_id = started.structured_content["job_id"]
            assert "read_live_metrics" in started.structured_content["next_actions"]

            live_metrics = await wait_for_live_metric(
                client,
                job_id,
                lambda stage_metrics: any(
                    key.endswith(":CrossEntropyLoss") for key in stage_metrics.get("flat_metrics", {})
                ),
                timeout_s=5.0,
            )
            assert live_metrics["flat_metrics"]

            with pytest.raises(Exception, match="Timed out while waiting for job"):
                await client.call_tool("wait_for_job", {"job_id": job_id, "timeout_s": 0.1, "poll_interval_s": 0.05})

            canceled = await client.call_tool("cancel_job", {"job_id": job_id, "wait_s": 0.5})
            assert canceled.structured_content["status"] == "killed"
            assert "validate_config_semantics" in canceled.structured_content["next_actions"]

            status = await client.call_tool("get_job_status", {"job_id": job_id})
            assert status.structured_content["status"] == "killed"

            listed = await client.call_tool("list_jobs", {})
            assert job_id in str(listed.structured_content)
            assert "killed" in str(listed.structured_content)

    asyncio.run(scenario())


def test_validate_restores_config_when_subprocess_times_out(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    load_mcp_server: Callable[[], ModuleType],
) -> None:
    # The child restores the authored config in its own finally, but a timeout SIGTERMs it before that
    # runs. Simulate a child that mutated the config (KonfAI rewrites it in place) and was then killed:
    # the parent must restore the authored bytes regardless.
    workspace_root = tmp_path / "workspaces"
    monkeypatch.setenv("KONFAI_MCP_WORKSPACES_ROOT", str(workspace_root))

    mcp_server = load_mcp_server()
    client_cls = fastmcp.Client

    authored = MINIMAL_TRAIN

    def fake_subprocess(target: str, kwargs: dict, *args, **more):  # type: ignore[no-untyped-def]
        config_path = Path(kwargs["config"])
        config_path.write_text("Trainer:\n  train_name: FAKE_RUN\n  materialised_default: 42\n", encoding="utf-8")
        raise TimeoutError("Isolated subprocess exceeded its deadline and was terminated.")

    async def scenario() -> None:
        async with client_cls(mcp_server.mcp) as client:
            await _create_fake_train_session(client)
            config_path = (
                Path((await client.call_tool("summarize_session", {})).structured_content["path"]) / "Config.yml"
            )
            assert config_path.read_text(encoding="utf-8").strip() == authored

            monkeypatch.setattr("konfai_mcp.runner.run_api_in_subprocess", fake_subprocess)
            with pytest.raises(Exception, match=r"deadline|terminated|Timeout"):
                await client.call_tool("validate_config_semantics", {"workflow": "train"})

            # The parent-side finally restored the authored config despite the child being killed.
            assert config_path.read_text(encoding="utf-8").strip() == authored

    asyncio.run(scenario())


def test_mcp_server_handles_burst_polling_without_crashing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    load_mcp_server: Callable[[], ModuleType],
) -> None:
    workspace_root = tmp_path / "workspaces"
    monkeypatch.setenv("KONFAI_MCP_WORKSPACES_ROOT", str(workspace_root))

    mcp_server = load_mcp_server()
    install_fake_konfai_runtime(tmp_path, monkeypatch, mcp_server)
    monkeypatch.setenv("KONFAI_MCP_FAKE_SLEEP_S", "0.8")
    monkeypatch.setenv("KONFAI_MCP_FAKE_STEPS", "6")
    monkeypatch.setenv("KONFAI_MCP_FAKE_METRIC_NAME", "MAE")
    client_cls = fastmcp.Client

    async def scenario() -> None:
        async with client_cls(mcp_server.mcp) as client:
            await _create_fake_train_session(client)
            started = await client.call_tool("run_train", {"cpu": 1})
            job_id = started.structured_content["job_id"]

        async def poll_once() -> tuple[str, str, str]:
            async with client_cls(mcp_server.mcp) as client:
                snapshot = await client.call_tool(
                    "read_live_metrics",
                    {"job_id": job_id, "max_entries": 3},
                )
                status = await client.call_tool("get_job_status", {"job_id": job_id})
                log = await client.read_resource(f"job://{job_id}/log")
                return (
                    snapshot.structured_content["status"],
                    status.structured_content["status"],
                    resource_to_text(log),
                )

        polls = await asyncio.gather(*(poll_once() for _ in range(12)))
        assert all(status in {"running", "done"} for status, _, _ in polls)
        assert all(status in {"running", "done"} for _, status, _ in polls)
        assert all("# KonfAI MCP job" in log_text for _, _, log_text in polls)

        async with client_cls(mcp_server.mcp) as client:
            done = await client.call_tool("wait_for_job", {"job_id": job_id, "timeout_s": 10.0})
            assert done.structured_content["status"] == "done"

    asyncio.run(scenario())
