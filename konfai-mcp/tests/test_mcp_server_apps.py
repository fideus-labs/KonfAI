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

import json
import sys
from pathlib import Path

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
    # The bundle ships a Config.yml, so it is finetunable (run_resume can warm-start from it).
    assert payload["finetunable"] is True
    # A runnable app routes forward to import_app (the single entry) instead of dead-ending.
    assert payload["next_actions"][0] == "import_app"


def test_describe_app_inference_only_still_routes_to_import(tmp_path: Path) -> None:
    """An app with no train Config.yml is not finetunable but is still imported the same way."""
    app_dir = _write_local_app(tmp_path)
    (app_dir / "Config.yml").unlink()

    payload = _service(tmp_path).describe_app(str(app_dir))

    assert payload["finetunable"] is False
    assert payload["next_actions"][0] == "import_app"


def test_describe_app_no_inference_routes_to_design(tmp_path: Path) -> None:
    """A bundle KonfAI cannot run for inference must route to design_config_strategy, not import_app."""
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
        return {"files": ["app.json", "tiny.pt"], "checkpoints": ["tiny.pt"], "configs": {"prediction": "Prediction.yml"}}

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
    assert result["next_actions"] == ["describe_app", "import_app"]
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
            "package_app_from_session",
        )
        import asyncio

        index = asyncio.run(server.read_tool_index())
        for name in tool_names:
            assert callable(getattr(server, name))
            assert name in index["tools"]

        # The app-execution wrappers are retired: an app runs as a normal experiment via import_app.
        for gone in ("run_app_infer", "run_app_evaluate", "run_app_uncertainty", "run_app_pipeline", "fine_tune_app"):
            assert gone not in index["tools"]

        assert "solve_task" in index["prompts"]
        solve = server.prompt_solve_task("segment the liver", "one CT group")
        content = solve[0]["content"]
        for tool in ("import_app", "run_prediction", "design_config_strategy"):
            assert tool in content

        app_dir = _write_local_app(tmp_path)
        described = server.describe_app(str(app_dir))
        assert described["display_name"] == "Tiny Local"
    finally:
        sys.modules.pop("konfai_mcp.server", None)
