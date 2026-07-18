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
from konfai.utils.errors import AppMetadataError, AppRepositoryError
from konfai_apps import app_repository as app_repository_module


def test_get_app_repository_info_rejects_missing_required_metadata_keys(tmp_path: Path) -> None:
    app_dir = tmp_path / "broken_app"
    app_dir.mkdir()
    (app_dir / "app.json").write_text(
        json.dumps(
            {
                "display_name": "Broken App",
                "short_description": "Missing full description",
                "tta": 0,
                "mc_dropout": 0,
            }
        ),
        encoding="utf-8",
    )

    try:
        app_repository_module.get_app_repository_info(str(app_dir), False)
    except AppMetadataError as exc:
        assert "Missing keys in app.json" in str(exc)
        assert "description" in str(exc)
    else:
        raise AssertionError("Expected invalid app metadata to raise AppMetadataError")


def test_get_app_repository_info_supports_local_directory(tmp_path: Path) -> None:
    app_dir = tmp_path / "demo_app"
    app_dir.mkdir()
    (app_dir / "app.json").write_text(
        json.dumps(
            {
                "display_name": "Demo App",
                "description": "Local test app",
                "short_description": "Demo",
                "tta": 0,
                "mc_dropout": 0,
            }
        ),
        encoding="utf-8",
    )

    repo = app_repository_module.get_app_repository_info(str(app_dir), False)

    assert isinstance(repo, app_repository_module.LocalAppRepositoryFromDirectory)
    assert repo.get_display_name() == "Demo App"
    assert repo.get_description() == "Local test app"


def test_get_app_repository_info_prefers_windows_local_path_over_hf_identifier(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    app_dir = tmp_path / "demo_app"
    app_dir.mkdir()
    (app_dir / "app.json").write_text(
        json.dumps(
            {
                "display_name": "Demo App",
                "description": "Local test app",
                "short_description": "Demo",
                "tta": 0,
                "mc_dropout": 0,
            }
        ),
        encoding="utf-8",
    )

    win_path = r"C:\Users\runneradmin\demo_app"

    monkeypatch.setattr(
        app_repository_module, "_resolve_local_app_path", lambda app_id: app_dir if app_id == win_path else None
    )
    monkeypatch.setattr(
        app_repository_module,
        "LocalAppRepositoryFromHF",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Should not resolve Windows local paths as HF repos")
        ),
    )

    repo = app_repository_module.get_app_repository_info(win_path, False)

    assert isinstance(repo, app_repository_module.LocalAppRepositoryFromDirectory)
    assert repo.get_name() == str(app_dir)


def test_local_hf_get_filenames_returns_relative_files_and_ignores_folders(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyFolder:
        def __init__(self, path: str) -> None:
            self.path = path

    class DummyFile:
        def __init__(self, path: str) -> None:
            self.path = path

    monkeypatch.setattr(app_repository_module, "RepoFolder", DummyFolder)
    monkeypatch.setattr(
        app_repository_module.LocalAppRepositoryFromHF,
        "_list_repo_tree",
        staticmethod(
            lambda repo_id, app_name, recursive=False: [
                DummyFolder("demo_app/assets"),
                DummyFile("demo_app/app.json"),
                DummyFile("demo_app/Inference.yml"),
                DummyFile("demo_app/assets/preprocess.py"),
            ]
        ),
    )

    filenames = app_repository_module.LocalAppRepositoryFromHF.get_filenames("org/demo", "demo_app", True)

    assert filenames == ["Inference.yml", "app.json", "assets/preprocess.py"]


def test_is_app_repo_requires_root_app_json() -> None:
    assert app_repository_module.is_app_repo(["Inference.yml", "app.json", "weights/model.pt"])
    assert not app_repository_module.is_app_repo(["docs/app.json", "Inference.yml"])


def test_local_hf_download_syncs_non_model_files_for_current_revision(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class DummyFolder:
        def __init__(self, path: str) -> None:
            self.path = path

    class DummyFile:
        def __init__(self, path: str) -> None:
            self.path = path

    snapshot_dir = tmp_path / "snapshot"
    (snapshot_dir / "demo_app").mkdir(parents=True)

    calls: dict[str, object] = {}

    monkeypatch.setattr(app_repository_module, "RepoFolder", DummyFolder)
    monkeypatch.setattr(app_repository_module.shutil, "rmtree", lambda path: calls.setdefault("removed", str(path)))
    monkeypatch.setattr(
        app_repository_module.LocalAppRepositoryFromHF,
        "_list_repo_tree",
        staticmethod(
            lambda repo_id, app_name, recursive=False: [
                DummyFile("demo_app/app.json"),
                DummyFile("demo_app/Inference.yml"),
                DummyFolder("demo_app/assets"),
                DummyFile("demo_app/assets/preprocess.py"),
                DummyFile("demo_app/model.pt"),
            ]
        ),
    )

    def fake_snapshot_download(**kwargs):
        calls["snapshot"] = kwargs
        return str(snapshot_dir)

    monkeypatch.setattr(app_repository_module, "snapshot_download", fake_snapshot_download)

    result = app_repository_module.LocalAppRepositoryFromHF.download(
        "org/demo@refs/pr/1",
        "demo_app/Inference.yml",
        True,
    )

    assert result == snapshot_dir / "demo_app" / "Inference.yml"
    assert "removed" not in calls
    assert calls["snapshot"] == {
        "repo_id": "org/demo",
        "repo_type": "model",
        "revision": "refs/pr/1",
        "allow_patterns": [
            "demo_app/Inference.yml",
            "demo_app/app.json",
            "demo_app/assets/preprocess.py",
        ],
    }


def test_local_hf_initial_model_availability_comes_from_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    app_json = tmp_path / "app.json"
    app_json.write_text(
        json.dumps(
            {
                "display_name": "Demo App",
                "description": "HF test app",
                "short_description": "Demo",
                "tta": 0,
                "mc_dropout": 0,
                "models": ["model.pt"],
            }
        ),
        encoding="utf-8",
    )
    inference_yml = tmp_path / "Inference.yml"
    inference_yml.write_text("Predictor: {}\n", encoding="utf-8")

    monkeypatch.setattr(
        app_repository_module.LocalAppRepositoryFromHF,
        "get_filenames",
        staticmethod(lambda repo_id, app_name, force_update: ["Inference.yml", "app.json", "model.pt"]),
    )
    monkeypatch.setattr(
        app_repository_module.LocalAppRepositoryFromHF,
        "get_cached_filenames",
        staticmethod(lambda repo_id, app_name: ["Inference.yml", "app.json"]),
    )

    def fake_download(repo_id: str, filename: str, force_update: bool) -> Path:
        if filename.endswith("app.json"):
            return app_json
        if filename.endswith("Inference.yml"):
            return inference_yml
        return tmp_path / Path(filename).name

    monkeypatch.setattr(
        app_repository_module.LocalAppRepositoryFromHF,
        "download",
        staticmethod(fake_download),
    )

    repo = app_repository_module.LocalAppRepositoryFromHF("org/demo", "demo_app", True)

    assert repo.get_checkpoints_name() == ["model.pt"]
    assert repo.get_checkpoints_name_available() == []


def test_local_hf_nested_cached_model_is_reported_available(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    app_json = tmp_path / "app.json"
    app_json.write_text(
        json.dumps(
            {
                "display_name": "Demo App",
                "description": "HF test app",
                "short_description": "Demo",
                "tta": 0,
                "mc_dropout": 0,
                "models": ["model.pt"],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        app_repository_module.LocalAppRepositoryFromHF,
        "get_filenames",
        staticmethod(lambda repo_id, app_name, force_update: ["Inference.yml", "app.json", "weights/model.pt"]),
    )
    monkeypatch.setattr(
        app_repository_module.LocalAppRepositoryFromHF,
        "get_cached_filenames",
        staticmethod(lambda repo_id, app_name: ["Inference.yml", "app.json", "weights/model.pt"]),
    )
    monkeypatch.setattr(
        app_repository_module.LocalAppRepositoryFromHF,
        "download",
        staticmethod(lambda repo_id, filename, force_update: app_json),
    )

    repo = app_repository_module.LocalAppRepositoryFromHF("org/demo", "demo_app", True)

    assert repo.get_checkpoints_name_available() == ["model.pt"]


def test_local_hf_download_inference_refreshes_selected_remote_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    app_json = tmp_path / "app.json"
    app_json.write_text(
        json.dumps(
            {
                "display_name": "Demo App",
                "description": "HF test app",
                "short_description": "Demo",
                "tta": 0,
                "mc_dropout": 0,
                "models": ["CV_0.pt"],
            }
        ),
        encoding="utf-8",
    )
    inference_yml = tmp_path / "Inference.yml"
    inference_yml.write_text("Predictor: {}\n", encoding="utf-8")

    get_filenames_calls: list[bool] = []

    def fake_get_filenames(repo_id: str, app_name: str, force_update: bool) -> list[str]:
        get_filenames_calls.append(force_update)
        if force_update:
            return ["CV_0.pt", "Inference.yml", "app.json"]
        return ["Inference.yml", "app.json"]

    monkeypatch.setattr(
        app_repository_module.LocalAppRepositoryFromHF,
        "get_filenames",
        staticmethod(fake_get_filenames),
    )
    monkeypatch.setattr(
        app_repository_module.LocalAppRepositoryFromHF,
        "get_cached_filenames",
        staticmethod(lambda repo_id, app_name: ["Inference.yml", "app.json"]),
    )

    def fake_download(repo_id: str, filename: str, force_update: bool) -> Path:
        if filename.endswith("app.json"):
            return app_json
        if filename.endswith("Inference.yml"):
            return inference_yml
        return tmp_path / Path(filename).name

    monkeypatch.setattr(
        app_repository_module.LocalAppRepositoryFromHF,
        "download",
        staticmethod(fake_download),
    )

    repo = app_repository_module.LocalAppRepositoryFromHF("org/demo", "demo_app", False)

    models_path, prediction_path, codes_path = repo.download_inference(1, ["CV_0"], "Inference.yml")

    assert models_path == [tmp_path / "CV_0.pt"]
    assert prediction_path == inference_yml
    # download_inference now stages every non-model bundle file (so assets like elastix parameter maps
    # are available in the run workspace, as the apps docs promise).
    assert codes_path == [("Inference.yml", inference_yml), ("app.json", app_json)]
    assert get_filenames_calls == [False, False, True]


def test_local_directory_install_inference_preserves_nested_python_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    app_root = tmp_path / "repo" / "demo_app"
    (app_root / "pkg").mkdir(parents=True)
    (app_root / "app.json").write_text(
        json.dumps(
            {
                "display_name": "Demo App",
                "description": "Local nested app",
                "short_description": "Demo",
                "tta": 0,
                "mc_dropout": 0,
                "models": ["model.pt"],
            }
        ),
        encoding="utf-8",
    )
    (app_root / "Inference.yml").write_text(
        "Predictor:\n  Dataset:\n    augmentations: {}\n    Patch:\n      patch_size: [1, 1, 1]\n    batch_size: 1\n",
        encoding="utf-8",
    )
    (app_root / "model.pt").write_text("weights", encoding="utf-8")
    (app_root / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (app_root / "pkg" / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")

    repo = app_repository_module.LocalAppRepositoryFromDirectory(app_root.parent, app_root.name)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)

    repo.install_inference(
        number_of_augmentation=0,
        number_of_model=1,
        name_of_models=[],
        number_of_mc_dropout=0,
        uncertainty=True,
        prediction_file="Inference.yml",
        available_vram=None,
    )

    assert (workspace / "pkg" / "__init__.py").exists()
    assert (workspace / "pkg" / "helper.py").exists()


def test_local_directory_nested_uncertainty_file_is_detected(tmp_path: Path) -> None:
    app_root = tmp_path / "repo" / "demo_app"
    (app_root / "qa").mkdir(parents=True)
    (app_root / "app.json").write_text(
        json.dumps(
            {
                "display_name": "Demo App",
                "description": "Local nested app",
                "short_description": "Demo",
                "tta": 0,
                "mc_dropout": 0,
            }
        ),
        encoding="utf-8",
    )
    (app_root / "qa" / "Uncertainty.yml").write_text("Predictor: {}\n", encoding="utf-8")

    repo = app_repository_module.LocalAppRepositoryFromDirectory(app_root.parent, app_root.name)

    assert repo.has_capabilities() == (False, False, True)


def test_install_evaluation_stages_non_python_assets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app_root = tmp_path / "repo" / "demo_app"
    app_root.mkdir(parents=True)
    (app_root / "app.json").write_text(
        json.dumps(
            {
                "display_name": "Demo App",
                "description": "Local eval app",
                "short_description": "Demo",
                "tta": 0,
                "mc_dropout": 0,
            }
        ),
        encoding="utf-8",
    )
    (app_root / "Evaluation.yml").write_text("Evaluator: {}\n", encoding="utf-8")
    (app_root / "labels.csv").write_text("id,name\n1,liver\n", encoding="utf-8")
    (app_root / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")
    (app_root / "model.pt").write_text("weights", encoding="utf-8")

    repo = app_repository_module.LocalAppRepositoryFromDirectory(app_root.parent, app_root.name)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)

    repo.install_evaluation("Evaluation.yml")

    assert (workspace / "Evaluation.yml").exists()
    assert (workspace / "labels.csv").exists()
    assert (workspace / "helper.py").exists()
    assert not (workspace / "model.pt").exists()


def test_install_uncertainty_stages_non_python_assets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app_root = tmp_path / "repo" / "demo_app"
    app_root.mkdir(parents=True)
    (app_root / "app.json").write_text(
        json.dumps(
            {
                "display_name": "Demo App",
                "description": "Local uncertainty app",
                "short_description": "Demo",
                "tta": 0,
                "mc_dropout": 0,
            }
        ),
        encoding="utf-8",
    )
    (app_root / "Uncertainty.yml").write_text("Predictor: {}\n", encoding="utf-8")
    (app_root / "lookup.json").write_text("{}\n", encoding="utf-8")
    (app_root / "model.pt").write_text("weights", encoding="utf-8")

    repo = app_repository_module.LocalAppRepositoryFromDirectory(app_root.parent, app_root.name)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)

    repo.install_uncertainty("Uncertainty.yml")

    assert (workspace / "Uncertainty.yml").exists()
    assert (workspace / "lookup.json").exists()
    assert not (workspace / "model.pt").exists()


def _write_two_checkpoint_app(app_root: Path) -> None:
    app_root.mkdir(parents=True, exist_ok=True)
    (app_root / "app.json").write_text(
        json.dumps(
            {
                "display_name": "Demo App",
                "description": "Local app with two checkpoints",
                "short_description": "Demo",
                "tta": 0,
                "mc_dropout": 0,
                "models": ["CV_0.pt", "CV_1.pt"],
            }
        ),
        encoding="utf-8",
    )
    (app_root / "Config.yml").write_text(
        "Trainer:\n  epochs: 100\n  it_validation: 2500\n  train_name: FT_0\n",
        encoding="utf-8",
    )
    (app_root / "CV_0.pt").write_text("weights-0", encoding="utf-8")
    (app_root / "CV_1.pt").write_text("weights-1", encoding="utf-8")


def _install_fine_tune(
    app_root: Path, workspace: Path, name_of_models: list[str], overrides: list[str] | None = None
) -> list[tuple[str, Path]]:
    workspace.mkdir(parents=True, exist_ok=True)
    repo = app_repository_module.LocalAppRepositoryFromDirectory(app_root.parent, app_root.name)
    return repo.install_fine_tune(
        config_file="Config.yml",
        path=workspace,
        display_name="Fine Tuned",
        epochs=3,
        it_validation=5,
        name_of_models=name_of_models,
        overrides=overrides,
    )


def test_install_fine_tune_defaults_to_first_checkpoint(tmp_path: Path) -> None:
    from ruamel.yaml import YAML

    app_root = tmp_path / "repo" / "demo_app"
    _write_two_checkpoint_app(app_root)
    workspace = tmp_path / "workspace"

    models = _install_fine_tune(app_root, workspace, [])

    assert [name for name, _ in models] == ["CV_0.pt"]
    assert models[0][1] == app_root / "CV_0.pt"
    # Shared assets are installed but checkpoints are not copied into the workspace by install.
    assert (workspace / "app.json").exists()
    assert (workspace / "Config.yml").exists()
    assert not (workspace / "CV_0.pt").exists()

    metadata = json.loads((workspace / "app.json").read_text(encoding="utf-8"))
    assert metadata["display_name"] == "Fine Tuned"
    assert metadata["models"] == ["CV_0.pt"]

    with open(workspace / "Config.yml") as file:
        config = YAML().load(file)
    assert config["Trainer"]["epochs"] == 3
    assert config["Trainer"]["it_validation"] == 5


def test_install_fine_tune_applies_model_and_dotted_overrides(tmp_path: Path) -> None:
    """A bare ``--set`` resolves into ``Trainer.Model.<Class>``; a dotted one hits the config root."""
    from ruamel.yaml import YAML

    app_root = tmp_path / "repo" / "demo_app"
    _write_two_checkpoint_app(app_root)
    # A training config with a model block, so a bare-name override has a Trainer.Model.<Class> to land in.
    (app_root / "Config.yml").write_text(
        "Trainer:\n"
        "  epochs: 100\n"
        "  it_validation: 2500\n"
        "  train_name: FT_0\n"
        "  Model:\n"
        "    classpath: net:UNet\n"
        "    UNet:\n"
        "      iterations: 100\n"
        "  Dataset:\n"
        "    batch_size: 1\n",
        encoding="utf-8",
    )
    workspace = tmp_path / "workspace"

    _install_fine_tune(app_root, workspace, [], overrides=["iterations=300", "Trainer.Dataset.batch_size=4"])

    with open(workspace / "Config.yml") as file:
        config = YAML().load(file)
    assert config["Trainer"]["Model"]["UNet"]["iterations"] == 300
    assert config["Trainer"]["Dataset"]["batch_size"] == 4
    # The epochs/it_validation rewrite still applies alongside the overrides.
    assert config["Trainer"]["epochs"] == 3
    assert config["Trainer"]["it_validation"] == 5


def test_install_fine_tune_selects_requested_checkpoint(tmp_path: Path) -> None:
    app_root = tmp_path / "repo" / "demo_app"
    _write_two_checkpoint_app(app_root)
    workspace = tmp_path / "workspace"

    models = _install_fine_tune(app_root, workspace, ["CV_1"])

    assert [name for name, _ in models] == ["CV_1.pt"]
    metadata = json.loads((workspace / "app.json").read_text(encoding="utf-8"))
    assert metadata["models"] == ["CV_1.pt"]


def test_install_fine_tune_selects_multiple_checkpoints(tmp_path: Path) -> None:
    app_root = tmp_path / "repo" / "demo_app"
    _write_two_checkpoint_app(app_root)
    workspace = tmp_path / "workspace"

    models = _install_fine_tune(app_root, workspace, ["CV_0", "CV_1"])

    assert [name for name, _ in models] == ["CV_0.pt", "CV_1.pt"]
    metadata = json.loads((workspace / "app.json").read_text(encoding="utf-8"))
    assert metadata["models"] == ["CV_0.pt", "CV_1.pt"]


def test_install_fine_tune_rejects_unknown_checkpoint(tmp_path: Path) -> None:
    app_root = tmp_path / "repo" / "demo_app"
    _write_two_checkpoint_app(app_root)
    workspace = tmp_path / "workspace"

    with pytest.raises(app_repository_module.AppRepositoryError):
        _install_fine_tune(app_root, workspace, ["CV_9"])


def _write_app_with_requirements(app_root: Path, requirements: str) -> None:
    app_root.mkdir(parents=True, exist_ok=True)
    (app_root / "app.json").write_text(
        json.dumps(
            {
                "display_name": "Demo App",
                "description": "Local app with requirements",
                "short_description": "Demo",
                "tta": 0,
                "mc_dropout": 0,
            }
        ),
        encoding="utf-8",
    )
    (app_root / "requirements.txt").write_text(requirements, encoding="utf-8")


def test_install_requirements_runs_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app_root = tmp_path / "repo" / "demo_app"
    _write_app_with_requirements(app_root, "konfai-nonexistent-xyz==1.2.3\n")
    repo = app_repository_module.LocalAppRepositoryFromDirectory(app_root.parent, app_root.name)

    monkeypatch.delenv("KONFAI_APPS_INSTALL_REQUIREMENTS", raising=False)
    captured: list[list[str]] = []
    monkeypatch.setattr(
        app_repository_module.subprocess, "check_call", lambda cmd, *args, **kwargs: captured.append(cmd)
    )

    repo._install_requirements(repo._get_filenames())

    assert len(captured) == 1
    assert captured[0][:5] == [sys.executable, "-m", "pip", "install", "konfai-nonexistent-xyz==1.2.3"]


def test_install_requirements_opt_out_is_a_noop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app_root = tmp_path / "repo" / "demo_app"
    _write_app_with_requirements(app_root, "konfai-nonexistent-xyz==1.2.3\n")
    repo = app_repository_module.LocalAppRepositoryFromDirectory(app_root.parent, app_root.name)

    monkeypatch.setenv("KONFAI_APPS_INSTALL_REQUIREMENTS", "0")
    calls: list[object] = []
    monkeypatch.setattr(app_repository_module.subprocess, "check_call", lambda *args, **kwargs: calls.append(args))

    repo._install_requirements(repo._get_filenames())

    assert calls == []


def test_install_requirements_skips_protected_and_non_pep508_lines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app_root = tmp_path / "repo" / "demo_app"
    _write_app_with_requirements(
        app_root,
        "\n".join(
            [
                "# comment line",
                "-r other-requirements.txt",
                "--extra-index-url https://example.com/simple",
                "git+https://github.com/foo/bar.git#egg=bar",
                "torch==1.0.0",
                "konfai==0.0.1",
                "konfai-nonexistent-xyz==1.2.3",
            ]
        )
        + "\n",
    )
    repo = app_repository_module.LocalAppRepositoryFromDirectory(app_root.parent, app_root.name)

    monkeypatch.delenv("KONFAI_APPS_INSTALL_REQUIREMENTS", raising=False)
    captured: list[list[str]] = []
    monkeypatch.setattr(
        app_repository_module.subprocess, "check_call", lambda cmd, *args, **kwargs: captured.append(cmd)
    )

    repo._install_requirements(repo._get_filenames())

    assert len(captured) == 1
    cmd = captured[0]
    assert cmd[:5] == [sys.executable, "-m", "pip", "install", "konfai-nonexistent-xyz==1.2.3"]
    assert "torch==1.0.0" not in cmd
    assert "konfai==0.0.1" not in cmd
    assert not any(part.startswith(("-r", "--extra-index-url", "git+")) for part in cmd[4:])


def test_install_requirements_protects_against_non_canonical_spellings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # pip resolves 'konfai_apps' / 'Torch' to the same projects as 'konfai-apps' / 'torch', so the guard
    # must canonicalize (PEP 503) rather than str.lower() -- otherwise these spellings slip past and pip
    # would downgrade a protected core package.
    app_root = tmp_path / "repo" / "demo_app"
    _write_app_with_requirements(
        app_root,
        "\n".join(["konfai_apps==0.0.1", "Torch==1.0.0", "konfai-nonexistent-xyz==1.2.3"]) + "\n",
    )
    repo = app_repository_module.LocalAppRepositoryFromDirectory(app_root.parent, app_root.name)

    monkeypatch.delenv("KONFAI_APPS_INSTALL_REQUIREMENTS", raising=False)
    captured: list[list[str]] = []
    monkeypatch.setattr(
        app_repository_module.subprocess, "check_call", lambda cmd, *args, **kwargs: captured.append(cmd)
    )

    repo._install_requirements(repo._get_filenames())

    assert len(captured) == 1
    cmd = captured[0]
    assert not any("konfai_apps" in part or part.lower().startswith("torch") for part in cmd), cmd
    assert cmd[:5] == [sys.executable, "-m", "pip", "install", "konfai-nonexistent-xyz==1.2.3"]


def _local_repo_with_config(tmp_path: Path, config: str) -> tuple[object, Path]:
    app_root = tmp_path / "repo" / "demo_app"
    app_root.mkdir(parents=True)
    (app_root / "app.json").write_text(
        json.dumps(
            {"display_name": "Demo", "description": "Demo", "short_description": "Demo", "tta": 0, "mc_dropout": 0}
        ),
        encoding="utf-8",
    )
    prediction = app_root / "Prediction.yml"
    prediction.write_text(config, encoding="utf-8")
    repo = app_repository_module.LocalAppRepositoryFromDirectory(app_root.parent, app_root.name)
    return repo, prediction


_CONFIG = (
    "Predictor:\n"
    "  Model:\n"
    "    RegistrationNet:\n"
    "      iterations: 150\n"
    "      learning_rate: 0.2\n"
    "      linear: true\n"
    "      subset_features: []\n"
)


def test_apply_config_overrides_patches_typed_values(tmp_path: Path) -> None:
    from ruamel.yaml import YAML

    repo, prediction = _local_repo_with_config(tmp_path, _CONFIG)
    repo._apply_config_overrides(
        str(prediction),
        [
            "iterations=300",  # bare model-parameter name (the common form)
            "Predictor.Model.RegistrationNet.learning_rate=0.05",  # full dotted path still works
            "linear=false",
            "subset_features=[0, 1, 2]",
        ],
    )
    net = YAML().load(prediction.read_text())["Predictor"]["Model"]["RegistrationNet"]
    assert net["iterations"] == 300 and isinstance(net["iterations"], int)
    assert float(net["learning_rate"]) == 0.05
    assert net["linear"] is False
    assert list(net["subset_features"]) == [0, 1, 2]


def test_apply_config_overrides_noop_when_empty(tmp_path: Path) -> None:
    repo, prediction = _local_repo_with_config(tmp_path, _CONFIG)
    before = prediction.read_text()
    repo._apply_config_overrides(str(prediction), None)
    repo._apply_config_overrides(str(prediction), [])
    assert prediction.read_text() == before


@pytest.mark.parametrize(
    "override",
    [
        "does_not_exist=1",  # unknown bare model parameter
        "Predictor.Model.RegistrationNet.does_not_exist=1",  # unknown leaf key (dotted)
        "Predictor.Missing.iterations=1",  # unknown intermediate key (dotted)
        "no_equals_sign",  # not NAME=VALUE
    ],
)
def test_apply_config_overrides_rejects_bad_override(tmp_path: Path, override: str) -> None:
    repo, prediction = _local_repo_with_config(tmp_path, _CONFIG)
    with pytest.raises(AppRepositoryError):
        repo._apply_config_overrides(str(prediction), [override])


_MODEL_CONFIG = (
    "Predictor:\n"
    "  Model:\n"
    "    classpath: Model:RegistrationNet\n"
    "    RegistrationNet:\n"
    "      iterations: 150\n"  # int -> tunable
    "      learning_rate: 0.2\n"  # float -> tunable
    "      linear: true\n"  # bool -> tunable
    "      voxel_size: [3.0, 3.0, 3.0]\n"  # list[float] -> tunable
    "      pca: [0]\n"  # list[int] -> tunable
    "      subset_features: []\n"  # empty list -> tunable ("list"; compound name, not locked)
    "      distance: [L1]\n"  # list[str] -> tunable
    "      models: [repo:MIND.pt]\n"  # list[str] -> tunable (feature-model choice)
    "      mode: bilinear\n"  # str -> tunable
    "      num_channels: 1\n"  # int -> tunable (config exposes it; hardcode in Model.py to hide it)
    "      channels: [1, 32, 64]\n"  # list[int] -> tunable (idem: hardcode the architecture to hide it)
    "      layers_mask: [true, false]\n"  # list[bool] -> tunable (which model layers to use)
    "      outputs_criterions: None\n"  # structural + KonfAI None string -> excluded
    "      disabled_option: None\n"  # KonfAI None string -> excluded
    "      optimizer:\n"  # structural nested mapping -> excluded
    "        AdamW: {}\n"
)


# A typed model module: the constructor's annotations ARE the constraint declaration.
# No `from __future__ import annotations` — get_parameters reads runtime annotation objects.
_TYPED_MODEL_PY = (
    "from typing import Annotated, Literal\n"
    "from konfai.utils.config import Choices, Range\n"
    "\n\n"
    "class RegistrationNet:\n"
    "    def __init__(\n"
    "        self,\n"
    "        mode: Literal['Static', 'Jacobian'] = 'Static',\n"
    "        spatial_samples: Annotated[int, Range(0, 100000)] = 0,\n"
    "        ref: Annotated[str, Choices(lambda: ['a:x.pt', 'b:y.pt'])] = '',\n"
    "        note: str = '',\n"
    "    ) -> None:\n"
    "        pass\n"
)

_TYPED_MODEL_CONFIG = (
    "Predictor:\n"
    "  Model:\n"
    "    classpath: Model:RegistrationNet\n"
    "    RegistrationNet:\n"
    "      mode: Jacobian\n"
    "      spatial_samples: 2000\n"
    "      ref: a:x.pt\n"
    "      note: hello\n"
)


def test_get_parameters_values_are_the_clean_model_block(tmp_path: Path) -> None:
    # `values` is the model block minus structural wiring — a JSON-clean tree the CLI edits via --set.
    repo, _prediction = _local_repo_with_config(tmp_path, _MODEL_CONFIG)
    result = repo.get_parameters()

    values = result["values"]
    assert values["iterations"] == 150 and isinstance(values["iterations"], int)
    assert values["voxel_size"] == [3.0, 3.0, 3.0]  # nested list -> plain python
    assert values["mode"] == "bilinear"
    assert values["disabled_option"] == "None"  # generic: no value is filtered, only structural KEYS are
    for structural in ("outputs_criterions", "optimizer"):
        assert structural not in values
    # No typed Model.py present -> constraints degrade to empty (an optional UI hint, never fatal).
    assert result["constraints"] == {}


def test_get_parameters_constraints_read_from_model_types(tmp_path: Path) -> None:
    repo, prediction = _local_repo_with_config(tmp_path, _TYPED_MODEL_CONFIG)
    (prediction.parent / "Model.py").write_text(_TYPED_MODEL_PY, encoding="utf-8")
    result = repo.get_parameters()

    assert result["values"] == {"mode": "Jacobian", "spatial_samples": 2000, "ref": "a:x.pt", "note": "hello"}
    # Constraints come from the constructor TYPES: Literal -> choices, Range -> min/max, Choices resolver run
    # by the app (so nothing is fetched here); an untyped field (`note`) simply carries no constraint.
    assert result["constraints"] == {
        "mode": {"choices": ["Static", "Jacobian"]},
        "spatial_samples": {"min": 0, "max": 100000},
        "ref": {"choices": ["a:x.pt", "b:y.pt"]},
    }


def test_save_default_parameters_persists_to_local_config(tmp_path: Path) -> None:
    from ruamel.yaml import YAML

    repo, prediction = _local_repo_with_config(tmp_path, _MODEL_CONFIG)
    repo.save_default_parameters(["iterations=999"])  # bare model-parameter name
    data = YAML().load(prediction.read_text())
    assert data["Predictor"]["Model"]["RegistrationNet"]["iterations"] == 999  # persisted on disk


def test_save_default_parameters_noop_when_empty(tmp_path: Path) -> None:
    repo, prediction = _local_repo_with_config(tmp_path, _MODEL_CONFIG)
    before = prediction.read_text()
    repo.save_default_parameters(None)
    repo.save_default_parameters([])
    assert prediction.read_text() == before


def test_save_default_parameters_missing_config_raises(tmp_path: Path) -> None:
    repo, prediction = _local_repo_with_config(tmp_path, _MODEL_CONFIG)
    prediction.unlink()
    with pytest.raises(AppRepositoryError):
        repo.save_default_parameters(["iterations=1"])


def test_export_app_materialises_local_copy_with_overrides(tmp_path: Path) -> None:
    repo, prediction = _local_repo_with_config(tmp_path, _MODEL_CONFIG)
    (prediction.parent / "model.pt").write_text("weights", encoding="utf-8")

    dest = tmp_path / "exported" / "MyTunedApp"
    repo.export_app(
        dest,
        display_name="My Tuned App",
        config_overrides=["iterations=777"],  # bare model-parameter name
    )

    assert (dest / "Prediction.yml").is_file()
    assert (dest / "model.pt").is_file()  # checkpoints come along
    assert json.loads((dest / "app.json").read_text())["display_name"] == "My Tuned App"

    # Reopen the export as a local app: the tuned default is baked in.
    exported = app_repository_module.LocalAppRepositoryFromDirectory(dest.parent, dest.name)
    assert exported.get_parameters()["values"]["iterations"] == 777


def _remote_info_payload(**overrides) -> dict:
    payload = {
        "app": "demo/app",
        "available": True,
        "display_name": "Demo",
        "description": "demo",
        "short_description": "demo",
        "checkpoints_name": ["m.pt"],
        "checkpoints_name_available": ["m.pt"],
        "maximum_tta": 0,
        "mc_dropout": 0,
        "has_capabilities": [True, False, False],
        "inputs": {"Volume_0": {"display_name": "MR", "volume_type": "VOLUME", "required": True}},
        "outputs": {"sCT": {"display_name": "sCT", "volume_type": "VOLUME", "required": True}},
        "inputs_evaluations": {},
    }
    payload.update(overrides)
    return payload


def _remote_repo_from_payload(monkeypatch: pytest.MonkeyPatch, payload: dict):
    class _Response:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return payload

    monkeypatch.setattr(app_repository_module.requests, "get", lambda *args, **kwargs: _Response())

    class _Server:
        timeout = 5

        def get_url(self) -> str:
            return "http://127.0.0.1:1"

        def get_headers(self) -> dict:
            return {}

    return app_repository_module.AppRepositoryInfoFromRemoteServer(_Server(), "demo/app")


def test_remote_repository_relays_the_server_reported_finetunable(monkeypatch: pytest.MonkeyPatch) -> None:
    # Remote fine-tune runs on the user's server, so the server (which resolves the actual bundle)
    # is the source of truth; the adapter must relay its answer, not hardcode False.
    repo = _remote_repo_from_payload(monkeypatch, _remote_info_payload(finetunable=False))
    assert repo.is_finetunable() is False

    repo = _remote_repo_from_payload(monkeypatch, _remote_info_payload(finetunable=True))
    assert repo.is_finetunable() is True


def test_remote_repository_finetunable_falls_back_to_inference_for_older_servers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A server predating the 'finetunable' field omits it; fall back to the inference capability,
    # which matches the historical behavior (fine-tune offered for any inference-capable app).
    repo = _remote_repo_from_payload(monkeypatch, _remote_info_payload())
    assert repo.is_finetunable() is True

    payload = _remote_info_payload()
    payload["has_capabilities"] = [False, False, False]
    repo = _remote_repo_from_payload(monkeypatch, payload)
    assert repo.is_finetunable() is False


def test_local_finetunable_requires_a_root_level_config_yml(tmp_path: Path) -> None:
    # install_fine_tune resolves 'path / Config.yml' flat, so a nested training/Config.yml must NOT
    # report finetunable (it would advertise a fine-tune that fails at install).
    app_dir = tmp_path / "flat"
    app_dir.mkdir()
    (app_dir / "app.json").write_text(
        json.dumps(
            {
                "display_name": "Flat",
                "description": "d",
                "short_description": "d",
                "tta": 0,
                "mc_dropout": 0,
                "models": ["m.pt"],
                "inputs": {},
                "outputs": {},
            }
        ),
        encoding="utf-8",
    )
    (app_dir / "m.pt").write_bytes(b"")
    repo = app_repository_module.LocalAppRepositoryFromDirectory(app_dir.parent, app_dir.name)
    assert repo.is_finetunable() is False

    (app_dir / "training").mkdir()
    (app_dir / "training" / "Config.yml").write_text("Trainer: {}\n", encoding="utf-8")
    assert repo.is_finetunable() is False

    (app_dir / "Config.yml").write_text("Trainer: {}\n", encoding="utf-8")
    assert repo.is_finetunable() is True


def test_export_copies_a_yaml_model_and_it_resolves(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # An app whose model is a declarative `.yml` (classpath: X.yml) instead of a Model.py must
    # carry that .yml through export, and the copied config must still resolve+build the model
    # (the .yml is looked up next to Prediction.yml). Locks the "apps handle YAML models" path.
    repo, prediction = _local_repo_with_config(tmp_path, "Predictor:\n  Model:\n    classpath: UNetSeg.yml\n")
    (prediction.parent / "UNetSeg.yml").write_text(
        "name: UNetSeg\n"
        "network:\n  in_channels: 1\n  dim: 2\n"
        "modules:\n  - name: Conv\n    type: Conv\n"
        "    args: {dim: 2, in_channels: 1, out_channels: 3, kernel_size: 1}\n",
        encoding="utf-8",
    )

    dest = tmp_path / "exported" / "SegApp"
    repo.export_app(dest, display_name="Seg App")

    # The model .yml travels with the bundle and the classpath is unchanged.
    assert (dest / "UNetSeg.yml").is_file()
    assert "UNetSeg.yml" in (dest / "Prediction.yml").read_text(encoding="utf-8")

    # The exported config resolves and builds the model (relative to the copied Prediction.yml).
    from konfai.network.network import ModelLoader, Network

    monkeypatch.setenv("KONFAI_config_file", str(dest / "Prediction.yml"))
    monkeypatch.setenv("KONFAI_CONFIG_MODE", "Done")
    monkeypatch.setenv("KONFAI_ROOT", "Predictor")
    model = ModelLoader("UNetSeg.yml").get_model(train=False)
    assert isinstance(model, Network)
