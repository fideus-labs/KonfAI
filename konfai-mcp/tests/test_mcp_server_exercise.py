# SPDX-License-Identifier: Apache-2.0
"""What real Claude Code runs against the server asked for (see CHANGELOG): the frictions they met stay fixed."""

import asyncio
import json
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


@pytest.mark.usefixtures("workspace_root")
def test_read_only_and_destructive_tools_are_hinted(load_mcp_server: Callable[[], ModuleType]) -> None:
    mcp_server = load_mcp_server()

    async def scenario() -> dict:
        async with fastmcp.Client(mcp_server.mcp) as client:
            return {t.name: t.annotations for t in await client.list_tools()}

    annotations = asyncio.run(scenario())
    assert set(mcp_server.READ_ONLY_TOOLS) <= set(annotations)  # every hinted name is a tool
    assert annotations["read_file"].read_only_hint is True
    assert annotations["delete_run"].destructive_hint is True
    assert not (annotations["run_train"] and annotations["run_train"].read_only_hint)
    # each can import code: session or app code (a runtime check, the model build, the app's parameters) or an
    # installed module a classpath names (a signature, the review summarize_session(include_validation) runs)
    for name in (
        "check_component",
        "check_config",
        "describe_model_outputs",
        "describe_app",
        "inspect_object_signature",
        "summarize_session",
    ):
        assert not (annotations[name] and annotations[name].read_only_hint), name


def test_fidopt_run_leaves_the_corpus_to_fidopt_and_holds_its_default_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, load_mcp_server: Callable[[], ModuleType]
) -> None:
    fake = tmp_path / "fidopt"
    fake.write_text("#!/bin/sh\n")
    monkeypatch.setenv("KONFAI_MCP_FIDOPT", str(fake))
    monkeypatch.setenv("KONFAI_MCP_FIDOPT_ALLOWED", "check,run")
    monkeypatch.delenv("FIDOPT_DATA", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    mcp_server = load_mcp_server()
    # AMOS_CTCT_Seg has no folder of its own (it reads AMOS_CTCT): FidOpt resolves the corpus, not the server
    derived = mcp_server.FIDOPT.prepare_run("run", "lungmask", "AMOS_CTCT_Seg", None, None, False, None)
    assert derived["kwargs"]["args"] == ["run", "lungmask", "AMOS_CTCT_Seg"]
    first = mcp_server.FIDOPT.prepare_run("check", "lungmask", "", None, None, False, None)
    second = mcp_server.FIDOPT.prepare_run("check", "lungmask", "", None, None, False, ["--gpu", "1"])
    assert first["run_name"] != second["run_name"]  # one folder per launch
    assert first["gpu"] == [] and second["gpu"] == [1]
    # run without --gpu holds FidOpt's own default: GPU 0, or none when CUDA_VISIBLE_DEVICES hides every GPU
    assert derived["gpu"] == [0]
    assert (
        mcp_server.FIDOPT.prepare_run("run", "lungmask", "LCTSC_Seg", None, None, False, ["--gpu", "cpu"])["gpu"] == []
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert mcp_server.FIDOPT.prepare_run("run", "lungmask", "LCTSC_Seg", None, None, False, None)["gpu"] == []


@pytest.mark.usefixtures("workspace_root")
def test_compare_runs_pairs_a_metric_under_two_names_and_cases_by_recorded_input(
    load_mcp_server: Callable[[], ModuleType],
) -> None:
    from test_mcp_server_run_records import _metric_json

    mcp_server = load_mcp_server()

    async def scenario() -> dict:
        async with fastmcp.Client(mcp_server.mcp) as client:
            await client.call_tool("initialize_session", {})
            workspace = Path(mcp_server.WORKSPACE_LAYOUT.workspace_dir())
            for run, name, values in (
                ("EXPERIMENT", "PRED:SEG:Dice", {"1PC006": 0.5, "1PC032": 0.6}),
                ("BUNDLE", "Volume_0:Reference_0:Dice", {"P000": 0.7, "P001": 0.55}),
                ("UNRECORDED", "Volume_0:Reference_0:Dice", {"P000": 0.7, "P001": 0.55}),
            ):
                metrics = workspace / "AppEvaluations" / run / "Metric_TRAIN.json"
                metrics.parent.mkdir(parents=True, exist_ok=True)
                payload = json.loads(_metric_json(name, values))
                if run != "EXPERIMENT":  # only the app run declares a direction, under its own name
                    payload["directions"] = {name: "max"}
                metrics.write_text(json.dumps(payload), encoding="utf-8")
            # the app was given 1PC032 then 1PC006: P000 is 1PC032, whatever the sorted names say
            manifest = mcp_server.WORKSPACE_LAYOUT.job_manifest_path("app_job")
            manifest.parent.mkdir(parents=True, exist_ok=True)
            cases = {"P000": ["/data/1PC032/CT.mha"], "P001": ["/data/1PC006/CT.mha"]}
            manifest.write_text(json.dumps({"output": str(workspace / "AppEvaluations" / "BUNDLE"), "cases": cases}))
            result = await client.call_tool(
                "compare_runs", {"run_a": "EXPERIMENT", "run_b": "BUNDLE", "metric": "Dice"}
            )
            with pytest.raises(Exception, match="cannot be paired"):
                await client.call_tool("compare_runs", {"run_a": "EXPERIMENT", "run_b": "UNRECORDED", "metric": "Dice"})
            return result.structured_content

    comparison = asyncio.run(scenario())["metrics"]["PRED:SEG:Dice"]
    assert comparison["metric_b"] == "Volume_0:Reference_0:Dice"
    assert comparison["direction_source"] == "declared"
    assert comparison["case_alignment"] == "by recorded input" and comparison["cases"] == 2
    assert comparison["per_case_delta_b_minus_a"] == {"1PC006": pytest.approx(0.05), "1PC032": pytest.approx(0.1)}


@pytest.mark.usefixtures("workspace_root")
def test_summarize_session_answers_before_the_workspace_exists(load_mcp_server: Callable[[], ModuleType]) -> None:
    mcp_server = load_mcp_server()

    mcp_server._activate_session("ghost")  # named by a host, its workspace never made (Studio's new experiment)

    async def scenario() -> dict:
        async with fastmcp.Client(mcp_server.mcp) as client:
            return (await client.call_tool("summarize_session", {})).structured_content

    payload = asyncio.run(scenario())
    assert payload["exists"] is False and payload["next_actions"][0] == "initialize_session"
