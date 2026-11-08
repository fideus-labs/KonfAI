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


def test_a_launch_waits_only_when_asked(
    load_mcp_server: Callable[[], ModuleType], monkeypatch: pytest.MonkeyPatch
) -> None:
    mcp_server = load_mcp_server()
    monkeypatch.setattr(
        mcp_server, "_tool_wait_for_job", lambda job_id, timeout_s: {"waited": job_id, "timeout_s": timeout_s}
    )
    assert mcp_server._launched({"job_id": "j1"}, False, None) == {"job_id": "j1"}
    assert mcp_server._launched({"ok": False, "blocked": True}, True, None) == {"ok": False, "blocked": True}
    assert mcp_server._launched({"job_id": "j1"}, True, 30) == {"waited": "j1", "timeout_s": 30}


def test_case_names_follow_the_staging_order(load_mcp_server: Callable[[], ModuleType]) -> None:
    mcp_server = load_mcp_server()
    names = mcp_server._case_names({"inputs": [["a.mha", "b.mha"], ["m1.mha"]]})
    assert names == {"P000": ["a.mha", "m1.mha"], "P001": ["b.mha", None]}
    assert mcp_server._case_names({"extra": {"inputs": [["x.nii.gz"]]}}) == {"P000": ["x.nii.gz"]}
    assert mcp_server._case_names({"ref": "r"}) is None


@pytest.mark.usefixtures("workspace_root")
def test_the_scout_service_follows_the_session(load_mcp_server: Callable[[], ModuleType]) -> None:
    mcp_server = load_mcp_server()

    async def scenario() -> None:
        async with fastmcp.Client(mcp_server.mcp) as client:
            await client.call_tool("session", {"action": "create", "name": "scouted"})

    asyncio.run(scenario())
    assert mcp_server.SCOUT_SERVICE.workspace_layout.current_session == "scouted"
    assert "scouted" in str(mcp_server.SCOUT_SERVICE.root())


@pytest.mark.usefixtures("workspace_root")
def test_a_registration_evaluation_stays_in_the_session_and_off_the_gpu(
    tmp_path: Path, load_mcp_server: Callable[[], ModuleType]
) -> None:
    mcp_server = load_mcp_server()
    fixed, moving = tmp_path / "f.nii.gz", tmp_path / "m.nii.gz"
    fixed.write_bytes(b"x")
    moving.write_bytes(b"x")
    spec = mcp_server.APP_SERVICE.prepare_registration_evaluate(
        fixed_seg=[str(fixed)], moving_seg=[str(moving)], output="AppEvaluations/baseline"
    )
    workspace = mcp_server.WORKSPACE_LAYOUT.workspace_dir()
    assert Path(spec["output"]) == (workspace / "AppEvaluations/baseline").resolve()
    assert spec["kwargs"]["gpu"] == []


@pytest.mark.usefixtures("workspace_root")
def test_a_package_classpath_is_not_mistaken_for_a_session_file(load_mcp_server: Callable[[], ModuleType]) -> None:
    mcp_server = load_mcp_server()

    async def scenario() -> dict:
        async with fastmcp.Client(mcp_server.mcp) as client:
            result = await client.call_tool("inspect_object_signature", {"classpath": "konfai_apps:KonfAIInference"})
            return result.structured_content

    payload = asyncio.run(scenario())
    assert payload["source"] == "imported"  # not "local": no konfai_apps.py sits in the session


@pytest.mark.usefixtures("workspace_root")
def test_every_launcher_can_wait(load_mcp_server: Callable[[], ModuleType]) -> None:
    mcp_server = load_mcp_server()

    async def scenario() -> dict[str, set[str]]:
        async with fastmcp.Client(mcp_server.mcp) as client:
            return {t.name: set(t.input_schema.get("properties", {})) for t in await client.list_tools()}

    params = asyncio.run(scenario())
    launchers = [
        "run_train",
        "run_resume",
        "run_prediction",
        "run_evaluation",
        "run_transform",
        "run_app",
        "fine_tune_app",
        "run_registration_evaluate",
    ]
    assert all({"wait", "timeout_s"} <= params[name] for name in launchers), params


@pytest.mark.usefixtures("workspace_root")
def test_describe_app_checks_a_local_bundle_against_the_schema(
    tmp_path: Path, load_mcp_server: Callable[[], ModuleType]
) -> None:
    import json

    app_dir = tmp_path / "Stale"
    app_dir.mkdir()
    (app_dir / "app.json").write_text(
        json.dumps(
            {
                "display_name": "Stale",
                "description": "d",
                "short_description": "s",
                "task": "synthesis",
                "tta": 0,
                "mc_dropout": 0,
                "models": ["w.safetensors"],
                "inputs": {"Volume_0": {"display_name": "MR", "volume_type": "VOLUME", "required": True}},
                "outputs": {"sCT": {"display_name": "sCT", "volume_type": "VOLUME", "required": True}},
            }
        )
    )
    (app_dir / "w.safetensors").write_bytes(b"")
    (app_dir / "Config.yml").write_text("Trainer:\n  Dataset:\n    use_cache: true\n")
    mcp_server = load_mcp_server()

    async def scenario() -> dict:
        async with fastmcp.Client(mcp_server.mcp) as client:
            return (await client.call_tool("describe_app", {"ref": str(app_dir)})).structured_content

    payload = asyncio.run(scenario())
    assert any("use_cache" in line for line in payload["config_checks"]["Config.yml"])
    assert payload["next_actions"][0] == "fetch_app"
