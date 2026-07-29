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
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import fastmcp
import pytest

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from konfai_mcp.server_apps import AppService  # noqa: E402
from konfai_mcp.server_support import WorkspaceLayout  # noqa: E402


def _write_local_app(root: Path, name: str = "TinyLocalApp") -> Path:
    """Create a minimal, offline-resolvable local app bundle and return its folder path."""
    app_dir = root / name
    app_dir.mkdir(parents=True, exist_ok=True)
    (app_dir / "app.json").write_text(
        json.dumps(
            {
                "display_name": "Tiny Local",
                "description": "Local synthesis app for MCP tests",
                "short_description": "tiny local",
                "task": "synthesis",
                "tta": 0,
                "mc_dropout": 0,
                "models": ["tiny.pt"],
                "patch_size": [1, 64, 64],
                "inputs": {"Volume_0": {"display_name": "MR", "volume_type": "VOLUME", "required": True}},
                "outputs": {"sCT": {"display_name": "sCT", "volume_type": "VOLUME", "required": True}},
            }
        ),
        encoding="utf-8",
    )
    (app_dir / "tiny.pt").write_bytes(b"")
    # A train config makes the app finetunable (run_resume warm-starts from Config.yml).
    (app_dir / "Config.yml").write_text("Trainer: {}\n", encoding="utf-8")
    return app_dir


def _service(tmp_path: Path, default_catalog: list[str] | None = None) -> AppService:
    layout = WorkspaceLayout(tmp_path / "workspaces")
    if default_catalog is None:
        return AppService(workspace_layout=layout)
    default_path = tmp_path / "default_catalog.json"
    default_path.write_text(json.dumps({"apps": default_catalog}), encoding="utf-8")
    return AppService(workspace_layout=layout, default_catalog_path=default_path)


def test_describe_app_reads_local_manifest(tmp_path: Path) -> None:
    app_dir = _write_local_app(tmp_path)
    payload = _service(tmp_path).describe_app(str(app_dir))

    assert payload["source"] == "local"
    assert payload["display_name"] == "Tiny Local"
    assert payload["short_description"] == "tiny local"
    assert payload["inputs"] == {"Volume_0": {"display_name": "MR", "volume_type": "VOLUME", "required": True}}
    assert payload["outputs"]["sCT"]["volume_type"] == "VOLUME"
    assert payload["capabilities"] == {"inference": True, "evaluation": False, "uncertainty": False}
    assert payload["checkpoints"] == ["tiny.pt"]
    assert payload["checkpoints_available"] == ["tiny.pt"]
    assert payload["patch_size"] == [1, 64, 64]
    assert payload["task"] == "synthesis"
    # The bundle ships a Config.yml, so it is finetunable and offers fine_tune_app.
    assert payload["finetunable"] is True
    # An inference-capable app routes forward to the run/tune tools instead of dead-ending.
    assert payload["next_actions"][0] == "run_app_infer"
    assert "fine_tune_app" in payload["next_actions"]
    assert "import_app" in payload["next_actions"]  # the modify-then-run path stays offered
    assert "run_app_evaluate" not in payload["next_actions"]


def test_describe_app_no_inference_routes_to_design(tmp_path: Path) -> None:
    """A bundle KonfAI cannot run for inference must route to design_config_strategy, not to a run tool."""
    app_dir = tmp_path / "no_inputs"
    app_dir.mkdir(parents=True, exist_ok=True)
    (app_dir / "app.json").write_text(
        json.dumps(
            {
                "display_name": "No IO",
                "description": "bundle without declared inputs",
                "short_description": "no io",
                "tta": 0,
                "mc_dropout": 0,
                "models": ["tiny.pt"],
                "inputs": {},
                "outputs": {},
            }
        ),
        encoding="utf-8",
    )
    (app_dir / "tiny.pt").write_bytes(b"")

    payload = _service(tmp_path).describe_app(str(app_dir))
    assert payload["capabilities"]["inference"] is False
    assert "run_app_infer" not in payload["next_actions"]
    assert "import_app" not in payload["next_actions"]
    assert "design_config_strategy" in payload["next_actions"]


def test_describe_app_marks_optional_input_not_required(tmp_path: Path) -> None:
    """An optional input reports required=False; the app runtime (not the manifest) supplies its fallback."""
    app_dir = tmp_path / "with_optional"
    app_dir.mkdir(parents=True, exist_ok=True)
    (app_dir / "app.json").write_text(
        json.dumps(
            {
                "display_name": "Masked Reg",
                "description": "Registration app with an optional mask",
                "short_description": "masked reg",
                "task": "registration",
                "tta": 0,
                "mc_dropout": 0,
                "models": ["tiny.pt"],
                "inputs": {
                    "Fixed": {"display_name": "Fixed", "volume_type": "VOLUME", "required": True},
                    "Mask": {"display_name": "Mask", "volume_type": "SEGMENTATION", "required": False},
                },
                "outputs": {"Moved": {"display_name": "Moved", "volume_type": "VOLUME", "required": True}},
            }
        ),
        encoding="utf-8",
    )
    (app_dir / "tiny.pt").write_bytes(b"")

    payload = _service(tmp_path).describe_app(str(app_dir))

    assert payload["inputs"]["Mask"]["required"] is False
    assert payload["inputs"]["Fixed"]["required"] is True
    assert payload["task"] == "registration"


def test_list_apps_override_with_local_path(tmp_path: Path) -> None:
    app_dir = _write_local_app(tmp_path)
    result = _service(tmp_path).list_apps(repos=[str(app_dir)], include_summary=True)

    assert result["count"] == 1
    assert result["errors"] == []
    entry = result["apps"][0]
    assert entry["source"] == "local"
    assert entry["ref"] == str(app_dir)
    assert entry["display_name"] == "Tiny Local"
    assert entry["inputs"] == ["Volume_0"]
    assert entry["outputs"] == ["sCT"]


def test_classify_reference_shapes(tmp_path: Path) -> None:
    app_dir = _write_local_app(tmp_path)
    classify = AppService._classify
    assert classify(str(app_dir)) == "local"
    assert classify("VBoussot/ImpactSynth") == "hf_repo"
    assert classify("VBoussot/ImpactSynth:MyApp") == "hf_app"
    assert classify("localhost:8000:MyApp") == "remote"
    assert classify("localhost:8000") == "remote_server"  # bare host:port = a server to enumerate
    assert classify("192.168.0.5:8080") == "remote_server"
    assert classify("mysteryword") == "unknown"


def test_resolve_catalog_merges_default_workspace_and_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    service = _service(tmp_path, default_catalog=["org/DefaultRepo"])

    workspace_path = service.workspace_layout.apps_catalog_path()
    workspace_path.parent.mkdir(parents=True, exist_ok=True)
    workspace_path.write_text(json.dumps({"apps": ["org/WorkspaceRepo", "org/DefaultRepo"]}), encoding="utf-8")

    env_path = tmp_path / "env_catalog.json"
    env_path.write_text(json.dumps({"apps": ["org/EnvRepo"]}), encoding="utf-8")
    monkeypatch.setenv("KONFAI_MCP_APP_CATALOG", str(env_path))

    refs, provenance = service.resolve_catalog()

    # default first, then workspace (dedup drops the repeated DefaultRepo), then env.
    assert refs == ["org/DefaultRepo", "org/WorkspaceRepo", "org/EnvRepo"]
    assert provenance["default"]["apps"] == ["org/DefaultRepo"]
    assert provenance["workspace"]["apps"] == ["org/WorkspaceRepo", "org/DefaultRepo"]
    assert provenance["env"]["exists"] is True


def test_register_and_unregister_roundtrip(tmp_path: Path) -> None:
    service = _service(tmp_path, default_catalog=[])
    path = service.workspace_layout.apps_catalog_path()

    added = service.register_app_source("org/MyRepo")
    assert added["added"] is True
    assert added["apps"] == ["org/MyRepo"]
    assert json.loads(path.read_text(encoding="utf-8"))["apps"] == ["org/MyRepo"]

    # idempotent
    again = service.register_app_source("org/MyRepo")
    assert again["added"] is False
    assert again["apps"] == ["org/MyRepo"]

    removed = service.unregister_app_source("org/MyRepo")
    assert removed["removed"] is True
    assert removed["apps"] == []
    assert json.loads(path.read_text(encoding="utf-8"))["apps"] == []


def test_read_catalog_file_rejects_malformed(tmp_path: Path) -> None:
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{not valid", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid app catalog JSON"):
        AppService._read_catalog_file(bad_json)

    wrong_shape = tmp_path / "wrong.json"
    wrong_shape.write_text(json.dumps({"models": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="must be a JSON object with an 'apps' list"):
        AppService._read_catalog_file(wrong_shape)


def test_missing_catalog_file_is_empty(tmp_path: Path) -> None:
    assert AppService._read_catalog_file(tmp_path / "does_not_exist.json") == []


def test_import_app_gates_local_and_rejects_remote(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app_dir = _write_local_app(tmp_path)
    service = _service(tmp_path)

    # A local/HF app copies + runs its code and pip-installs -> gated behind allow_untrusted_code.
    with pytest.raises(ValueError, match="allow_untrusted_code=True"):
        service.import_app(str(app_dir))

    # A remote server keeps its code remote and cannot be imported into the session.
    with pytest.raises(ValueError, match="cannot be imported"):
        service.import_app("localhost:8000:MyApp", allow_untrusted_code=True)

    # With the gate confirmed, it dispatches to the spawn-subprocess import API (never in-process).
    calls: list[tuple[str, dict]] = []
    from konfai_mcp import runner as mcp_runner

    def _fake_subprocess(target: str, kwargs: dict) -> dict:
        calls.append((target, kwargs))
        return {
            "files": ["app.json", "tiny.pt"],
            "checkpoints": ["tiny.pt"],
            "configs": {"prediction": "Prediction.yml"},
        }

    monkeypatch.setattr(mcp_runner, "run_api_in_subprocess", _fake_subprocess)
    result = service.import_app(str(app_dir), allow_untrusted_code=True)

    assert calls and calls[0][0] == "konfai_mcp.runner:import_app_api"
    assert calls[0][1]["ref"] == str(app_dir)
    # The target is the session root (in-jail), never a per-app sub-folder.
    assert Path(calls[0][1]["target"]) == service.workspace_layout.ensure_session_workspace()
    assert result["checkpoints"] == ["tiny.pt"]
    assert result["next_actions"] == ["run_resume", "run_prediction", "run_evaluation", "validate_config_semantics"]


def test_import_app_api_copies_weightless_registration_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A weightless registration app (engine-based, no .pt — e.g. ConvexAdam / FireANTs) imports cleanly:
    the runner copies its config into the session and reports checkpoints=[] so run_prediction runs it
    with the config's own engine (no models= needed)."""
    monkeypatch.setenv("KONFAI_APPS_INSTALL_REQUIREMENTS", "0")  # engine deps out of scope for the copy check
    app_dir = tmp_path / "ConvexAdamLike"
    app_dir.mkdir(parents=True, exist_ok=True)
    (app_dir / "app.json").write_text(
        json.dumps(
            {
                "display_name": "ConvexAdam-like",
                "description": "Weightless registration engine app",
                "short_description": "weightless reg",
                "task": "registration",
                "tta": 0,
                "mc_dropout": 0,
                "models": [],  # weightless: the engine comes from the config classpath, not a checkpoint
                "inputs": {
                    "Fixed": {"display_name": "Fixed", "volume_type": "VOLUME", "required": True},
                    "Moving": {"display_name": "Moving", "volume_type": "VOLUME", "required": True},
                },
                "outputs": {"MovedImage": {"display_name": "Moved", "volume_type": "VOLUME", "required": True}},
            }
        ),
        encoding="utf-8",
    )
    (app_dir / "Prediction.yml").write_text("Predictor: {}\n", encoding="utf-8")

    from konfai_mcp.runner import import_app_api

    target = tmp_path / "session"
    payload = import_app_api(ref=str(app_dir), target=str(target))

    assert payload["checkpoints"] == []  # weightless -> run_prediction needs no models=
    assert payload["configs"] == {"prediction": "Prediction.yml"}
    assert (target / "Prediction.yml").is_file()
    assert (target / "app.json").is_file()


def test_list_parameters_gates_and_reads(tmp_path: Path) -> None:
    app_dir = _write_local_app(tmp_path)
    service = _service(tmp_path)

    with pytest.raises(ValueError, match="allow_untrusted_code=True"):
        service.list_parameters(str(app_dir))

    with pytest.raises(ValueError, match="only supported for local or HuggingFace"):
        service.list_parameters("localhost:8000:MyApp", allow_untrusted_code=True)

    # No Prediction.yml in the fixture -> get_parameters returns empty values/constraints (no import).
    result = service.list_parameters(str(app_dir), allow_untrusted_code=True)
    assert result["values"] == {}
    assert result["constraints"] == {}
    assert result["source"] == "local"


def test_export_app_copies_bundle(tmp_path: Path) -> None:
    app_dir = _write_local_app(tmp_path)
    target = tmp_path / "exported"
    result = _service(tmp_path).export_app(str(app_dir), str(target))

    assert Path(result["exported_to"]) == target.resolve()
    assert (target / "app.json").exists()
    assert (target / "tiny.pt").exists()

    with pytest.raises(ValueError, match="Exporting is only supported"):
        _service(tmp_path).export_app("localhost:8000:MyApp", str(tmp_path / "x"))


def test_package_from_session_builds_bundle(tmp_path: Path) -> None:
    layout = WorkspaceLayout(tmp_path / "workspaces")
    workspace = layout.workspace_dir()
    (workspace / "Checkpoints" / "run").mkdir(parents=True)
    (workspace / "Checkpoints" / "run" / "model.pt").write_bytes(b"")
    (workspace / "Prediction.yml").write_text("Predictor: {}\n", encoding="utf-8")

    result = AppService(workspace_layout=layout).package_from_session(
        name="MyBundle", display_name="My App", description="does useful things"
    )

    bundle = Path(result["bundle_path"])
    assert bundle.is_dir()
    assert (bundle / "app.json").exists()
    assert (bundle / "Prediction.yml").exists()
    assert (bundle / "model.pt").exists()
    assert result["checkpoints"] == ["model.pt"]
    assert result["next_actions"] == ["describe_app", "run_app_infer", "import_app"]
    meta = json.loads((bundle / "app.json").read_text(encoding="utf-8"))
    assert meta["display_name"] == "My App"
    assert meta["short_description"] == "My App"


def test_package_from_session_normalizes_prediction_config_and_refreshes_support_files(tmp_path: Path) -> None:
    layout = WorkspaceLayout(tmp_path / "workspaces")
    workspace = layout.workspace_dir()
    (workspace / "Checkpoints" / "run").mkdir(parents=True)
    (workspace / "Checkpoints" / "run" / "model.pt").write_bytes(b"")
    (workspace / "UNet.yml").write_text("modules: []\n", encoding="utf-8")
    (workspace / "Prediction.yml").write_text(
        "Predictor:\n"
        "  Model:\n"
        "    classpath: UNet.yml\n"
        "  Dataset:\n"
        "    dataset_filenames: [/abs/train/Dataset:a:mha]\n"
        "    groups_src:\n"
        "      CT:\n"
        "        groups_dest:\n"
        "          CT: {}\n"  # is_input omitted: KonfAI defaults it to True -> this IS an input
        "  outputs_dataset:\n"
        "    UNet:Head:\n"
        "      OutputDataset:\n"
        "        group: PRED\n"
        "        same_as_group: CT:CT\n",
        encoding="utf-8",
    )
    service = AppService(workspace_layout=layout)

    result = service.package_from_session(name="Norm", display_name="N", description="n")
    bundle = Path(result["bundle_path"])

    from konfai_mcp.server_support import YAML_SAFE

    data = YAML_SAFE.load((bundle / "Prediction.yml").read_text(encoding="utf-8"))
    dataset = data["Predictor"]["Dataset"]
    # The bundle must read the staged app inputs, never the session's training dataset.
    assert dataset["dataset_filenames"] == ["./Dataset/:a:mha"]
    assert list(dataset["groups_src"]) == ["Volume_0"]
    assert data["Predictor"]["outputs_dataset"]["UNet:Head"]["OutputDataset"]["same_as_group"] == "Volume_0:CT"
    assert result["inputs"] == ["CT"]
    assert (bundle / "UNet.yml").read_text(encoding="utf-8") == "modules: []\n"

    # Repackaging under the same name must serve the EDITED support file, not the stale copy.
    (workspace / "UNet.yml").write_text("modules: [edited]\n", encoding="utf-8")
    service.package_from_session(name="Norm", display_name="N", description="n")
    assert (bundle / "UNet.yml").read_text(encoding="utf-8") == "modules: [edited]\n"


def test_normalize_bundled_prediction_config_preserves_windows_drive_path(tmp_path: Path) -> None:
    from konfai_mcp.server_support import YAML_SAFE, yaml_dump_content

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    # A naive split(":", 1) would read the drive letter 'C' as the path and '\\...:a:mha' as the token.
    config = {"Predictor": {"Dataset": {"dataset_filenames": ["C:\\data\\Dataset:a:mha"]}}}
    (bundle / "Prediction.yml").write_text(yaml_dump_content(config), encoding="utf-8")

    _app_service(tmp_path)._normalize_bundled_prediction_configs(bundle, ["Prediction.yml"])

    data = YAML_SAFE.load((bundle / "Prediction.yml").read_text(encoding="utf-8"))
    # The staged root replaces the path; the ':a:mha' accessor/format token survives uncorrupted.
    assert data["Predictor"]["Dataset"]["dataset_filenames"] == ["./Dataset/:a:mha"]


def test_package_from_session_requires_checkpoints_and_config(tmp_path: Path) -> None:
    layout = WorkspaceLayout(tmp_path / "workspaces")
    workspace = layout.workspace_dir()
    workspace.mkdir(parents=True)
    service = AppService(workspace_layout=layout)

    with pytest.raises(ValueError, match="No checkpoints to package"):
        service.package_from_session(name="B", display_name="d", description="d")

    checkpoint = workspace / "model.pt"
    checkpoint.write_bytes(b"")
    with pytest.raises(ValueError, match="No config to package"):
        service.package_from_session(name="B", display_name="d", description="d", checkpoints=[str(checkpoint)])


def test_package_from_session_runs_onnx_export_in_subprocess(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    layout = WorkspaceLayout(tmp_path / "workspaces")
    workspace = layout.workspace_dir()
    (workspace / "Checkpoints" / "run").mkdir(parents=True)
    (workspace / "Checkpoints" / "run" / "model.pt").write_bytes(b"")
    (workspace / "Prediction.yml").write_text("Predictor: {}\n", encoding="utf-8")

    import konfai_apps.bundle as bundle_module

    def _fail_in_process(*args: object, **kwargs: object) -> object:
        raise AssertionError("export_onnx_into_bundle must not run in the server process")

    monkeypatch.setattr(bundle_module, "export_onnx_into_bundle", _fail_in_process)

    calls: list[tuple[str, dict[str, object]]] = []

    def _fake_subprocess(target: str, kwargs: dict[str, object], *rest: object) -> str:
        calls.append((target, kwargs))
        return kwargs["bundle"] + "/model.onnx"  # type: ignore[operator]

    # Patch by dotted path so it lands on the live konfai_mcp.runner module (some suites reload it),
    # which is the exact object package_from_session's `from . import runner` resolves at call time.
    monkeypatch.setattr("konfai_mcp.runner.run_api_in_subprocess", _fake_subprocess)

    result = AppService(workspace_layout=layout).package_from_session(
        name="OnnxBundle", display_name="Onnx App", description="exports onnx", onnx=True
    )

    assert len(calls) == 1
    target, kwargs = calls[0]
    assert target == "konfai_apps.bundle:export_onnx_into_bundle"
    assert kwargs["bundle"] == str(Path(result["bundle_path"]))
    assert kwargs["checkpoint"].endswith("model.pt")  # type: ignore[union-attr]
    assert result["onnx"].endswith("model.onnx")


def test_server_registers_app_tools(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KONFAI_MCP_WORKSPACES_ROOT", str(tmp_path / "ws"))
    import importlib

    server = importlib.reload(importlib.import_module("konfai_mcp.server"))
    try:
        tool_names = (
            "list_apps",
            "describe_app",
            "list_app_parameters",
            "export_app",
            "import_app",
            "register_app_source",
            "unregister_app_source",
            "run_app_infer",
            "run_app_evaluate",
            "run_app_uncertainty",
            "run_app_pipeline",
            "fine_tune_app",
            "package_app_from_session",
        )
        import asyncio

        index = asyncio.run(server.read_tool_index())
        for name in tool_names:
            assert callable(getattr(server, name))
            assert name in index["tools"]

        import konfai_mcp.runner as runner

        assert callable(runner.run_app_api)
        assert callable(runner.run_app_action_api)
        assert callable(runner.run_finetune_api)

        assert "solve_task" in index["prompts"]
        solve = server.prompt_solve_task("segment the liver", "one CT group")
        content = solve[0]["content"]
        for tool in ("run_app_infer", "fine_tune_app", "import_app", "design_config_strategy"):
            assert tool in content

        app_dir = _write_local_app(tmp_path)
        described = server.describe_app(str(app_dir))
        assert described["display_name"] == "Tiny Local"
    finally:
        sys.modules.pop("konfai_mcp.server", None)


def test_set_parameters_preserve_value_types(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    load_mcp_server: Callable[[], ModuleType],
) -> None:
    """export_app must encode --set values so their type survives the YAML re-parse in konfai_apps.

    A string "true"/"1" would otherwise come back as a bool/int; json.dumps keeps it a string while
    leaving genuine ints/bools/floats/lists untouched.
    """
    yaml_safe = pytest.importorskip("ruamel.yaml")

    monkeypatch.setenv("KONFAI_MCP_WORKSPACES_ROOT", str(tmp_path / "workspaces"))
    mcp_server = load_mcp_server()

    captured: dict[str, list[str] | None] = {}

    def fake_export_app(ref, path, *, display_name=None, config_overrides=None, force_update=False):
        captured["config_overrides"] = config_overrides
        return {"ref": ref, "exported_to": str(path)}

    monkeypatch.setattr(mcp_server.APP_SERVICE, "export_app", fake_export_app)

    set_parameters = {"mode": "true", "code": "1", "iterations": 300, "flag": True, "lr": 2.0}

    async def scenario() -> None:
        async with fastmcp.Client(mcp_server.mcp) as client:
            await client.call_tool(
                "export_app",
                {"ref": "org/App", "path": str(tmp_path / "out"), "set_parameters": set_parameters},
            )

    asyncio.run(scenario())

    overrides = captured["config_overrides"]
    assert overrides is not None
    parser = yaml_safe.YAML(typ="safe")
    round_tripped = {name: parser.load(value) for name, _, value in (item.partition("=") for item in overrides)}
    assert round_tripped == set_parameters
    assert isinstance(round_tripped["mode"], str) and isinstance(round_tripped["code"], str)


def _dummy_inputs(tmp_path: Path) -> list[list[str]]:
    volume = tmp_path / "case0" / "Volume_0.mha"
    volume.parent.mkdir(parents=True, exist_ok=True)
    volume.write_bytes(b"")
    return [[str(volume)]]


def test_describe_app_inference_only_is_not_finetunable(tmp_path: Path) -> None:
    """An app with no train Config.yml must not advertise fine_tune_app (it would dead-end)."""
    app_dir = _write_local_app(tmp_path)
    (app_dir / "Config.yml").unlink()

    payload = _service(tmp_path).describe_app(str(app_dir))

    assert payload["finetunable"] is False
    assert "fine_tune_app" not in payload["next_actions"]
    # Inference routing is unaffected.
    assert payload["next_actions"][0] == "run_app_infer"


def test_prepare_infer_gates_local_app(tmp_path: Path) -> None:
    app_dir = _write_local_app(tmp_path)
    service = _service(tmp_path)
    inputs = _dummy_inputs(tmp_path)

    with pytest.raises(ValueError, match="allow_untrusted_code=True"):
        service.prepare_infer(ref=str(app_dir), inputs=inputs)

    spec = service.prepare_infer(ref=str(app_dir), inputs=inputs, allow_untrusted_code=True, tta=2)
    assert spec["kind"] == "infer"
    assert spec["target"] == "konfai_mcp.runner:run_app_api"
    assert spec["kwargs"]["ref"] == str(app_dir)
    assert spec["kwargs"]["tta"] == 2
    assert spec["kwargs"]["inputs"] == [[str(Path(inputs[0][0]).resolve())]]
    assert "app_TinyLocalApp" in spec["output"]


def test_prepare_infer_rejects_a_remote_server_ref(tmp_path: Path) -> None:
    """Remote app servers are not driven from the MCP: the reference is refused, never run as local."""
    service = _service(tmp_path)
    inputs = _dummy_inputs(tmp_path)

    for ref in ("localhost:8000:MyApp", "localhost:8000"):
        with pytest.raises(ValueError, match="remote app server"):
            service.prepare_infer(ref=ref, inputs=inputs, allow_untrusted_code=True)


def test_prepare_infer_rejects_bare_repo_and_bad_inputs(tmp_path: Path) -> None:
    app_dir = _write_local_app(tmp_path)
    service = _service(tmp_path)
    inputs = _dummy_inputs(tmp_path)

    with pytest.raises(ValueError, match="not a single app"):
        service.prepare_infer(ref="org/SomeRepo", inputs=inputs, allow_untrusted_code=True)

    with pytest.raises(ValueError, match="inputs cannot be empty"):
        service.prepare_infer(ref=str(app_dir), inputs=[], allow_untrusted_code=True)

    with pytest.raises(ValueError, match="path not found"):
        service.prepare_infer(ref=str(app_dir), inputs=[[str(tmp_path / "missing.mha")]], allow_untrusted_code=True)


def test_prepare_infer_labels_the_tuned_trial(tmp_path: Path) -> None:
    """A tuned run's default output dir names its parameters, so the leaderboard metrics_path says which
    trial produced the score."""
    app_dir = _write_local_app(tmp_path)
    spec = _service(tmp_path).prepare_infer(
        ref=str(app_dir),
        inputs=_dummy_inputs(tmp_path),
        allow_untrusted_code=True,
        config_overrides=["iterations=300"],
    )
    assert spec["kwargs"]["config_overrides"] == ["iterations=300"]
    assert "app_TinyLocalApp__iterations_300" in spec["output"]


def test_prepare_infer_cpu_without_gpu_forces_cpu(tmp_path: Path) -> None:
    """Selecting CPU must not leave the app on its default (every visible GPU): gpu becomes an empty list."""
    app_dir = _write_local_app(tmp_path)
    spec = _service(tmp_path).prepare_infer(
        ref=str(app_dir), inputs=_dummy_inputs(tmp_path), allow_untrusted_code=True, cpu=4
    )
    assert spec["kwargs"]["gpu"] == []
    assert spec["kwargs"]["cpu"] == 4


def test_prepare_evaluate_requires_gt_and_gates(tmp_path: Path) -> None:
    app_dir = _write_local_app(tmp_path)
    inputs = _dummy_inputs(tmp_path)
    gt_file = tmp_path / "gt" / "Reference_0.mha"
    gt_file.parent.mkdir(parents=True, exist_ok=True)
    gt_file.write_bytes(b"")
    gt = [[str(gt_file)]]
    service = _service(tmp_path)

    with pytest.raises(ValueError, match="allow_untrusted_code=True"):
        service.prepare_evaluate(ref=str(app_dir), inputs=inputs, gt=gt)

    with pytest.raises(ValueError, match="requires ground-truth"):
        service.prepare_evaluate(ref=str(app_dir), inputs=inputs, gt=None, allow_untrusted_code=True)

    spec = service.prepare_evaluate(ref=str(app_dir), inputs=inputs, gt=gt, allow_untrusted_code=True)
    assert spec["kind"] == "evaluate"
    assert spec["target"] == "konfai_mcp.runner:run_app_action_api"
    assert spec["kwargs"]["action"] == "evaluate"
    assert spec["kwargs"]["gt"] == [[str(gt_file.resolve())]]
    assert spec["kwargs"]["extra"]["evaluation_file"] == "Evaluation.yml"


def test_prepare_uncertainty_spec(tmp_path: Path) -> None:
    app_dir = _write_local_app(tmp_path)
    spec = _service(tmp_path).prepare_uncertainty(
        ref=str(app_dir), inputs=_dummy_inputs(tmp_path), allow_untrusted_code=True
    )
    assert spec["kind"] == "uncertainty"
    assert spec["kwargs"]["action"] == "uncertainty"
    assert spec["kwargs"]["gt"] is None
    assert spec["kwargs"]["extra"]["uncertainty_file"] == "Uncertainty.yml"


def test_prepare_pipeline_spec_and_overrides(tmp_path: Path) -> None:
    app_dir = _write_local_app(tmp_path)
    service = _service(tmp_path)
    inputs = _dummy_inputs(tmp_path)

    spec = service.prepare_pipeline(ref=str(app_dir), inputs=inputs, uncertainty=False, allow_untrusted_code=True)
    assert spec["kind"] == "pipeline"
    assert spec["kwargs"]["action"] == "pipeline"
    assert spec["kwargs"]["extra"]["uncertainty"] is False
    assert "config_overrides" not in spec["kwargs"]["extra"]

    tuned = service.prepare_pipeline(
        ref=str(app_dir), inputs=inputs, allow_untrusted_code=True, config_overrides=["iterations=300"]
    )
    # Overrides nested under extra still label the trial's output dir (the leaderboard reads that path).
    assert tuned["kwargs"]["extra"]["config_overrides"] == ["iterations=300"]
    assert "pipeline_TinyLocalApp__iterations_300" in tuned["output"]


def test_prepare_finetune_gates_local_app(tmp_path: Path) -> None:
    app_dir = _write_local_app(tmp_path)
    dataset = tmp_path / "Dataset"
    dataset.mkdir()
    service = _service(tmp_path)

    with pytest.raises(ValueError, match="allow_untrusted_code=True"):
        service.prepare_finetune(ref=str(app_dir), dataset=str(dataset))

    spec = service.prepare_finetune(ref=str(app_dir), dataset=str(dataset), allow_untrusted_code=True, epochs=3)
    assert spec["kind"] == "finetune"
    assert spec["target"] == "konfai_mcp.runner:run_finetune_api"
    assert spec["kwargs"]["dataset"] == str(dataset.resolve())
    assert spec["kwargs"]["epochs"] == 3
    assert "finetune_TinyLocalApp" in spec["output"]


def test_prepare_finetune_rejects_remote_and_bad_dataset(tmp_path: Path) -> None:
    app_dir = _write_local_app(tmp_path)
    service = _service(tmp_path)
    dataset = tmp_path / "Dataset"
    dataset.mkdir()

    with pytest.raises(ValueError, match="remote app server"):
        service.prepare_finetune(ref="localhost:8000:MyApp", dataset=str(dataset), allow_untrusted_code=True)

    with pytest.raises(ValueError, match="dataset must be an existing directory"):
        service.prepare_finetune(ref=str(app_dir), dataset=str(tmp_path / "missing"), allow_untrusted_code=True)

    with pytest.raises(ValueError, match="epochs must be a positive integer"):
        service.prepare_finetune(ref=str(app_dir), dataset=str(dataset), epochs=0, allow_untrusted_code=True)


def test_prepare_finetune_bakes_set_parameters(tmp_path: Path) -> None:
    app_dir = _write_local_app(tmp_path)
    dataset = tmp_path / "Dataset"
    dataset.mkdir()

    spec = _service(tmp_path).prepare_finetune(
        ref=str(app_dir),
        dataset=str(dataset),
        allow_untrusted_code=True,
        config_overrides=["iterations=300"],
    )
    # The overrides reach the runner (so they bake into the training config) ...
    assert spec["kwargs"]["config_overrides"] == ["iterations=300"]
    # ... and the default output dir is param-legible, so the leaderboard metrics_path names the trial.
    assert "finetune_TinyLocalApp__iterations_300" in spec["output"]


def test_app_tools_launch_tracked_app_jobs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The tool -> prepare_* -> job-registry wiring: kind, runner target, devices, manifest and the
    tuned parameters that gate the refine loop (the launch itself is stubbed -- no subprocess)."""
    monkeypatch.setenv("KONFAI_MCP_WORKSPACES_ROOT", str(tmp_path / "ws"))
    import importlib

    server = importlib.reload(importlib.import_module("konfai_mcp.server"))
    try:
        from konfai_mcp.server_jobs import Job

        app_dir = _write_local_app(tmp_path)
        captured: dict[str, object] = {}

        def fake_launch(**kwargs: object) -> Job:
            captured.clear()
            captured.update(kwargs)
            return Job(
                job_id="app-1",
                session="default",
                kind=kwargs["kind"],  # type: ignore[arg-type]
                command=kwargs["command"],  # type: ignore[arg-type]
                cwd=kwargs["cwd"],  # type: ignore[arg-type]
                log_path=kwargs["log_path"],  # type: ignore[arg-type]
                config_path=kwargs["config_path"],  # type: ignore[arg-type]
                run_name=kwargs["run_name"],  # type: ignore[arg-type]
                devices=kwargs["devices"],  # type: ignore[arg-type]
                set_parameters=kwargs.get("set_parameters"),  # type: ignore[arg-type]
            )

        monkeypatch.setattr(server.JOB_REGISTRY, "launch", fake_launch)

        payload = server.run_app_infer(
            ref=str(app_dir),
            inputs=_dummy_inputs(tmp_path),
            allow_untrusted_code=True,
            cpu=2,
            set_parameters={"iterations": 300},
        )
        assert captured["kind"] == "infer"
        assert captured["target"] == "konfai_mcp.runner:run_app_api"
        # json.dumps keeps the value's type through the YAML re-parse; the trial's parameters are recorded
        # on the job so the payload can echo them and gate the refine next_actions.
        assert captured["set_parameters"] == ["iterations=300"]
        assert captured["kwargs"]["config_overrides"] == ["iterations=300"]  # type: ignore[index]
        assert captured["devices"] == ["cpu"]  # cpu without gpu forces a CPU reservation
        assert captured["extra_manifest"]["app_ref"] == str(app_dir)  # type: ignore[index]
        # The child runs chdir'd into the session workspace, which is auto-created for an app job.
        assert captured["kwargs"]["cwd"] == str(server.WORKSPACE_LAYOUT.workspace_dir())  # type: ignore[index]
        assert payload["kind"] == "infer"
        assert payload["output"] == captured["kwargs"]["output"]  # type: ignore[index]
        assert payload["set_parameters"] == ["iterations=300"]

        dataset = tmp_path / "Dataset"
        dataset.mkdir(exist_ok=True)
        tuned = server.fine_tune_app(ref=str(app_dir), dataset=str(dataset), allow_untrusted_code=True, epochs=2, cpu=1)
        assert captured["kind"] == "finetune"
        assert captured["target"] == "konfai_mcp.runner:run_finetune_api"
        assert captured["kwargs"]["epochs"] == 2  # type: ignore[index]
        assert tuned["kind"] == "finetune"
    finally:
        sys.modules.pop("konfai_mcp.server", None)


def test_app_execution_tool_schemas_reach_the_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    load_mcp_server: Callable[[], ModuleType],
) -> None:
    """The five app-execution tools were exercised at function level only; a broken Annotated/Field on
    any of them would surface as a client-side schema hole, invisible to those tests."""
    import fastmcp

    monkeypatch.setenv("KONFAI_MCP_WORKSPACES_ROOT", str(tmp_path / "workspaces"))
    mcp_server = load_mcp_server()

    async def scenario() -> dict[str, dict]:
        async with fastmcp.Client(mcp_server.mcp) as client:
            return {tool.name: (tool.inputSchema or {}) for tool in await client.list_tools()}

    schemas = asyncio.run(scenario())
    expected = {
        "run_app_infer": {"ref", "inputs", "output"},
        "run_app_evaluate": {"ref", "inputs", "gt"},
        "run_app_uncertainty": {"ref", "inputs"},
        "run_app_pipeline": {"ref", "inputs", "gt"},
        "fine_tune_app": {"ref", "dataset", "output"},
    }
    for name, required_params in expected.items():
        assert name in schemas, f"{name} is not exposed to the client"
        properties = schemas[name].get("properties", {})
        missing = required_params - set(properties)
        assert not missing, f"{name} lost parameters on the wire: {missing}"
        # pydantic places a union param's description inside anyOf on some versions (the 3.10 leg):
        # documented means present anywhere in the property's subtree.
        undocumented = [p for p in required_params if '"description"' not in json.dumps(properties[p])]
        assert not undocumented, f"{name} has undocumented parameters: {undocumented}"


def _app_service(tmp_path: Path) -> AppService:
    return AppService(workspace_layout=WorkspaceLayout(tmp_path / "workspaces"))


def _touch(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return str(path)


def _local_app(tmp_path: Path) -> str:
    """A minimal resolvable local app bundle -- enough for the prepare_* path validation."""
    app_dir = tmp_path / "PairedApp"
    app_dir.mkdir(parents=True, exist_ok=True)
    (app_dir / "app.json").write_text(
        json.dumps(
            {
                "display_name": "Paired",
                "description": "app used for pairing checks",
                "short_description": "paired",
                "tta": 0,
                "mc_dropout": 0,
                "models": ["tiny.pt"],
                "inputs": {"Volume_0": {"display_name": "MR", "volume_type": "VOLUME", "required": True}},
                "outputs": {"sCT": {"display_name": "sCT", "volume_type": "VOLUME", "required": True}},
            }
        ),
        encoding="utf-8",
    )
    (app_dir / "tiny.pt").write_bytes(b"")
    return str(app_dir)


def test_check_case_pairing_rejects_mismatched_file_counts(tmp_path: Path) -> None:
    inputs = [[_touch(tmp_path / "in" / f"a{i}.mha") for i in range(3)]]
    gt = [[_touch(tmp_path / "gt" / f"g{i}.mha") for i in range(2)]]
    with pytest.raises(ValueError, match="mismatched case counts"):
        AppService._check_case_pairing([("inputs", inputs), ("gt", gt)])

    equal_gt = [[_touch(tmp_path / "gt2" / f"g{i}.mha") for i in range(3)]]
    AppService._check_case_pairing([("inputs", inputs), ("gt", equal_gt)])  # no raise


def test_check_case_pairing_skips_directory_groups(tmp_path: Path) -> None:
    # A directory expands to an unknown case count downstream, so it is not counted here: a 1-dir
    # group beside a 3-file group must NOT falsely trip the guard.
    (tmp_path / "series").mkdir()
    inputs = [[str(tmp_path / "series")]]
    gt = [[_touch(tmp_path / "gt" / f"g{i}.mha") for i in range(3)]]
    AppService._check_case_pairing([("inputs", inputs), ("gt", gt)])  # no raise


def test_prepare_infer_rejects_unequal_channel_counts(tmp_path: Path) -> None:
    a = _touch(tmp_path / "c" / "a.mha")
    b = _touch(tmp_path / "c" / "b.mha")
    c = _touch(tmp_path / "c" / "c.mha")
    with pytest.raises(ValueError, match="mismatched case counts"):
        # The pairing check runs before the trust gate, so an un-gated call still reports it.
        _app_service(tmp_path).prepare_infer(ref=_local_app(tmp_path), inputs=[[a, b], [c]])
