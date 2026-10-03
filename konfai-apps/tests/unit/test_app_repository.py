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
from types import SimpleNamespace

import pytest
from konfai_apps import app_repository as app_repository_module
from konfai_apps.errors import AppMetadataError, AppRepositoryError
from ruamel.yaml import YAML


@pytest.fixture(autouse=True)
def _a_process_that_never_read_the_hub(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Each test runs as a process that has not read the Hub yet, over a Hugging Face cache of its own."""
    monkeypatch.setattr(app_repository_module.constants, "HF_HUB_CACHE", str(tmp_path / "hub"))
    app_repository_module._release_tag.cache_clear()


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


def test_a_relative_local_app_path_still_names_the_app_after_a_chdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A run chdirs into its workspace before it reads the app's files: './demo_app' then named nothing
    # and the run died on a misleading 'Prediction.yml not found'.
    app_root = tmp_path / "demo_app"
    _write_app_with_requirements(app_root, "")
    (app_root / "Prediction.yml").write_text("Predictor: {}\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    repo = app_repository_module.get_app_repository_info("./demo_app", False)

    monkeypatch.chdir(tmp_path.parent)

    assert repo._download("Prediction.yml").is_file()


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


@pytest.mark.parametrize(
    ("version", "tags", "expected"),
    [("1.9.0", ["v1.8.6", "v1.9.0"], "v1.9.0"), ("1.9.0", ["v1.8.6"], None), ("1.9.1.dev3+g1234", ["v1.9.1"], None)],
    ids=["tagged", "untagged", "development-build"],
)
def test_an_unpinned_hf_app_takes_the_bundle_tagged_for_this_release(
    monkeypatch: pytest.MonkeyPatch, version: str, tags: list[str], expected: str | None
) -> None:
    """A bundle's configs follow KonfAI, so `main` breaks older releases (batch_size: 0 crashed
    konfai 1.8.5). A release takes the revision tagged with its own version; a development build or a
    repository without that tag keeps `main`, and an explicit @rev always wins."""
    monkeypatch.setattr(app_repository_module.importlib.metadata, "version", lambda name: version)
    refs = SimpleNamespace(tags=[SimpleNamespace(name=tag) for tag in tags])
    monkeypatch.setattr(app_repository_module, "HfApi", lambda: SimpleNamespace(list_repo_refs=lambda repo_id: refs))
    split = app_repository_module.LocalAppRepositoryFromHF._split_repo_reference

    assert split("org/demo") == ("org/demo", expected)
    assert split("org/demo@refs/pr/1") == ("org/demo", "refs/pr/1")


def test_offline_an_unpinned_hf_app_takes_the_release_tag_a_run_cached(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def offline(repo_id: str) -> None:
        raise ConnectionError("offline")

    monkeypatch.setattr(app_repository_module.importlib.metadata, "version", lambda name: "1.9.0")
    monkeypatch.setattr(app_repository_module, "HfApi", lambda: SimpleNamespace(list_repo_refs=offline))
    monkeypatch.setattr(app_repository_module.constants, "HF_HUB_CACHE", str(tmp_path))
    split = app_repository_module.LocalAppRepositoryFromHF._split_repo_reference

    assert split("org/demo") == ("org/demo", None)
    app_repository_module._release_tag.cache_clear()  # a new process
    (tmp_path / "models--org--demo" / "refs").mkdir(parents=True)
    (tmp_path / "models--org--demo" / "refs" / "v1.9.0").write_text("0123abcd", encoding="utf-8")
    assert split("org/demo") == ("org/demo", "v1.9.0")


_DEMO_APP_JSON = json.dumps(
    {"display_name": "Demo", "description": "HF test app", "short_description": "Demo", "tta": 0, "mc_dropout": 0}
)
_HUB_FILES = ["CBCT/app.json", "CBCT/Prediction.yml", "MR/app.json", "MR/model.pt", "SAM2.1_Small.pt", "docs/a.md"]


class _HubFolder(SimpleNamespace):
    pass


def _hub_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cached: list[str]) -> Path:
    """A Hugging Face cache of `org/demo` at `main` holding only `cached`; returns its snapshot folder."""
    repo = tmp_path / "hub" / "models--org--demo"
    snapshot = repo / "snapshots" / "0123abcd"
    snapshot.mkdir(parents=True)
    for name in cached:
        (snapshot / name).parent.mkdir(parents=True, exist_ok=True)
        (snapshot / name).write_text(_DEMO_APP_JSON if name.endswith("app.json") else "x", encoding="utf-8")
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("0123abcd", encoding="utf-8")
    monkeypatch.setattr(app_repository_module, "_release_tag", lambda repo_id, wait=True, refresh=False: None)
    return snapshot


def _fake_hub(monkeypatch: pytest.MonkeyPatch, files: list[str]) -> list[str]:
    """Serve `files` as the Hub tree of any repository, and a download as the files it lands in the cache's
    `main` snapshot; returns the `path_in_repo` of every tree call."""
    calls: list[str] = []
    cached_snapshot = app_repository_module.snapshot_download

    def snapshot_download(repo_id, revision=None, allow_patterns=None, local_files_only=False, **kwargs):
        if local_files_only:
            return cached_snapshot(repo_id, revision=revision, local_files_only=True, **kwargs)
        repo = Path(app_repository_module.constants.HF_HUB_CACHE) / f"models--{repo_id.replace('/', '--')}"
        (repo / "refs").mkdir(parents=True, exist_ok=True)
        (repo / "refs" / "main").write_text("0123abcd", encoding="utf-8")
        for name in allow_patterns:
            (repo / "snapshots" / "0123abcd" / name).parent.mkdir(parents=True, exist_ok=True)
            (repo / "snapshots" / "0123abcd" / name).write_text(_DEMO_APP_JSON, encoding="utf-8")
        return str(repo / "snapshots" / "0123abcd")

    def list_repo_tree(repo_id, path_in_repo=None, recursive=False, revision=None, repo_type=None):
        calls.append(path_in_repo or "")
        prefix = f"{path_in_repo}/" if path_in_repo else ""
        entries: dict[str, SimpleNamespace] = {}
        for path in (path for path in files if path.startswith(prefix)):
            parts = path[len(prefix) :].split("/")
            for depth in range(1, len(parts) if recursive else min(len(parts), 2)):
                folder = prefix + "/".join(parts[:depth])
                entries.setdefault(folder, _HubFolder(path=folder))
            if recursive or len(parts) == 1:
                entries[path] = SimpleNamespace(path=path)
        return iter(entries.values())

    monkeypatch.setattr(app_repository_module, "RepoFolder", _HubFolder)

    def model_info(repo_id, revision=None):
        return SimpleNamespace(siblings=[SimpleNamespace(rfilename=path) for path in files])

    monkeypatch.setattr(
        app_repository_module, "HfApi", lambda: SimpleNamespace(list_repo_tree=list_repo_tree, model_info=model_info)
    )
    monkeypatch.setattr(app_repository_module, "snapshot_download", snapshot_download)
    return calls


def test_an_hf_catalogue_lists_every_app_of_the_hub_and_downloads_their_files_but_the_checkpoints(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A cache filled by one app's run is not the repository's list: online, every app folder of the Hub is
    listed, and its files but the checkpoints land in the cache the catalogue summaries read."""
    snapshot = _hub_cache(monkeypatch, tmp_path, ["CBCT/app.json"])
    _fake_hub(monkeypatch, _HUB_FILES)

    assert app_repository_module.get_available_apps_on_hf_repo("org/demo") == ["CBCT", "MR"]
    assert sorted(app_repository_module.get_downloaded_apps_on_hf_repo("org/demo")) == ["CBCT", "MR"]
    assert (snapshot / "MR" / "app.json").is_file() and not (snapshot / "MR" / "model.pt").exists()


def test_offline_an_hf_catalogue_lists_what_the_cache_holds(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _hub_cache(monkeypatch, tmp_path, ["CBCT/app.json"])
    monkeypatch.setattr(app_repository_module.constants, "HF_HUB_OFFLINE", True)

    assert app_repository_module.get_available_apps_on_hf_repo("org/demo") == ["CBCT"]
    with pytest.raises(AppRepositoryError, match="nothing is cached"):
        app_repository_module.get_available_apps_on_hf_repo("org/other")


@pytest.mark.parametrize(
    ("operation", "whole"),
    [
        (lambda repo, out: repo.download_files(), True),
        (lambda repo, out: repo.export_app(out), True),
        (lambda repo, out: repo.download_app(), False),
    ],
    ids=["download_files", "export_app", "download_app"],
)
def test_a_whole_hf_app_includes_the_files_its_cache_lacks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, operation, whole: bool
) -> None:
    """`konfai-apps download` and an export take the whole app, checkpoints included, even when the cache
    holds only the files an earlier resolution fetched. `--download` keeps to the cached files: Slicer passes
    it on a first run, and the run fetches the checkpoints it selected."""
    snapshot = _hub_cache(monkeypatch, tmp_path, ["MR/app.json"])
    _fake_hub(monkeypatch, _HUB_FILES)
    fetched: list[str] = []

    def download(repo_id: str, filename: str, force_update: bool) -> Path:
        fetched.append(filename)
        path = snapshot / filename
        if not path.exists():
            path.write_text("x", encoding="utf-8")
        return path

    monkeypatch.setattr(app_repository_module.LocalAppRepositoryFromHF, "download", staticmethod(download))
    repo = app_repository_module.LocalAppRepositoryFromHF("org/demo", "MR", False)

    operation(repo, tmp_path / "export")

    assert ("MR/model.pt" in fetched) is whole


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
    # download_inference must stage every non-model bundle file (so assets like elastix parameter maps
    # are available in the run workspace, as the apps docs promise).
    assert codes_path == [("Inference.yml", inference_yml), ("app.json", app_json)]
    assert get_filenames_calls == [False, False, True]


@pytest.mark.parametrize(
    ("config_batch_size", "forced_batch_size", "batch_size"), [(1, None, 1), (0, None, 0), (0, 4, 4)]
)
def test_local_directory_install_inference_preserves_nested_python_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    config_batch_size: int,
    forced_batch_size: int | None,
    batch_size: int,
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
        "Predictor:\n  Dataset:\n    augmentations: {}\n    Patch:\n      patch_size: [1, 1, 1]\n"
        f"    batch_size: {config_batch_size}\n",
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
        forced_batch_size=forced_batch_size,
    )

    assert (workspace / "pkg" / "__init__.py").exists()
    assert (workspace / "pkg" / "helper.py").exists()
    # The app's config decides its batch (0 measures it on the GPU) unless one is forced.
    assert YAML().load(Path("Inference.yml"))["Predictor"]["Dataset"]["batch_size"] == batch_size


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
    app_root: Path,
    workspace: Path,
    name_of_models: list[str],
    overrides: list[str] | None = None,
    batch_size: int | None = None,
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
        batch_size=batch_size,
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


def test_install_fine_tune_writes_the_batch_size_beside_epochs(tmp_path: Path) -> None:
    """The batch size is a training knob like epochs, not a model tunable: it is first-class, and it
    lands under Trainer.Dataset where the trainer reads it."""
    from ruamel.yaml import YAML

    app_root = tmp_path / "repo" / "demo_app"
    _write_two_checkpoint_app(app_root)
    (app_root / "Config.yml").write_text(
        "Trainer:\n  epochs: 100\n  it_validation: 2500\n  train_name: FT_0\n  Dataset:\n    batch_size: 32\n",
        encoding="utf-8",
    )
    workspace = tmp_path / "workspace"

    _install_fine_tune(app_root, workspace, [], batch_size=4)

    with open(workspace / "Config.yml") as file:
        config = YAML().load(file)
    assert config["Trainer"]["Dataset"]["batch_size"] == 4


@pytest.mark.parametrize("batch_size", [0, -2])
def test_install_fine_tune_refuses_a_non_positive_batch_size(tmp_path: Path, batch_size: int) -> None:
    """Zero and negative values reach here from the CLI, the remote transport, and the Python API
    alike: refuse before anything is installed, not once the trainer builds its DataLoader."""
    app_root = tmp_path / "repo" / "demo_app"
    _write_two_checkpoint_app(app_root)
    workspace = tmp_path / "workspace"

    with pytest.raises(AppRepositoryError, match="positive"):
        _install_fine_tune(app_root, workspace, [], batch_size=batch_size)
    assert not workspace.exists() or not any(workspace.iterdir())


def test_install_fine_tune_refuses_a_batch_size_with_no_dataset_block(tmp_path: Path) -> None:
    """A config with no Trainer.Dataset cannot take a batch size: refuse and name the block, instead of
    silently adding a key the trainer never reads."""
    app_root = tmp_path / "repo" / "demo_app"
    _write_two_checkpoint_app(app_root)  # its Config.yml has no Dataset block
    workspace = tmp_path / "workspace"

    with pytest.raises(AppRepositoryError, match=r"Trainer\.Dataset"):
        _install_fine_tune(app_root, workspace, [], batch_size=4)


def test_a_bare_set_name_that_is_not_a_model_parameter_points_at_the_dotted_form(tmp_path: Path) -> None:
    """`--set batch_size=4` failed a real fine-tune with a message that never said which spelling would
    have worked: the error must name the dotted-path form."""
    app_root = tmp_path / "repo" / "demo_app"
    _write_two_checkpoint_app(app_root)
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

    with pytest.raises(AppRepositoryError, match="dotted path"):
        _install_fine_tune(app_root, workspace, [], overrides=["batch_size=4"])


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

    with pytest.raises(AppRepositoryError):
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


def test_a_clipped_tta_request_is_said(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # app.json 'tta' caps the copies: every IMPACT-Reg preset declares 0, and '--tta 2' ran none, silently.
    app_root = tmp_path / "repo" / "demo_app"
    _write_app_with_requirements(app_root, "")
    config = tmp_path / "Prediction.yml"
    config.write_text(
        "Predictor:\n  Dataset:\n    augmentations:\n      DataAugmentation_0:\n        nb: 2\n", encoding="utf-8"
    )
    repo = app_repository_module.LocalAppRepositoryFromDirectory(app_root.parent, app_root.name)

    repo._set_number_of_augmentation(str(config), 2)

    assert "2 test-time augmentation(s) asked, but app 'demo_app' allows 0" in capsys.readouterr().out
    assert YAML().load(config)["Predictor"]["Dataset"]["augmentations"] == {}


def _capture_pip(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Replace the pip subprocess by one that records its command and succeeds."""
    captured: list[list[str]] = []

    class _Pip:
        def __init__(self, command: list[str], **kwargs: object) -> None:
            captured.append(command)
            self.stdout: list[str] = []
            self.returncode = 0

        def __enter__(self) -> "_Pip":
            return self

        def __exit__(self, *exc: object) -> None:
            return None

    monkeypatch.setattr(app_repository_module.subprocess, "Popen", _Pip)
    return captured


def test_install_requirements_runs_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app_root = tmp_path / "repo" / "demo_app"
    _write_app_with_requirements(app_root, "konfai-nonexistent-xyz==1.2.3\n")
    repo = app_repository_module.LocalAppRepositoryFromDirectory(app_root.parent, app_root.name)

    monkeypatch.delenv("KONFAI_APPS_INSTALL_REQUIREMENTS", raising=False)
    captured = _capture_pip(monkeypatch)

    repo._install_requirements(repo._get_filenames())

    assert len(captured) == 1
    assert captured[0][:5] == [sys.executable, "-m", "pip", "install", "-c"]
    assert captured[0][6:] == ["konfai-nonexistent-xyz==1.2.3"]


def test_install_requirements_opt_out_is_a_noop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app_root = tmp_path / "repo" / "demo_app"
    _write_app_with_requirements(app_root, "konfai-nonexistent-xyz==1.2.3\n")
    repo = app_repository_module.LocalAppRepositoryFromDirectory(app_root.parent, app_root.name)

    monkeypatch.setenv("KONFAI_APPS_INSTALL_REQUIREMENTS", "0")
    calls = _capture_pip(monkeypatch)

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
    captured = _capture_pip(monkeypatch)

    repo._install_requirements(repo._get_filenames())

    assert len(captured) == 1
    cmd = captured[0]
    assert cmd[6:] == ["konfai-nonexistent-xyz==1.2.3"]
    assert "torch==1.0.0" not in cmd
    assert "konfai==0.0.1" not in cmd


def test_install_requirements_protects_against_non_canonical_spellings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # pip resolves 'konfai_apps' / 'Torch' to the same projects as 'konfai-apps' / 'torch', so the guard
    # must canonicalize (PEP 503) rather than str.lower(): otherwise these spellings slip past and pip
    # would downgrade a protected core package.
    app_root = tmp_path / "repo" / "demo_app"
    _write_app_with_requirements(
        app_root,
        "\n".join(["konfai_apps==0.0.1", "Torch==1.0.0", "konfai-nonexistent-xyz==1.2.3"]) + "\n",
    )
    repo = app_repository_module.LocalAppRepositoryFromDirectory(app_root.parent, app_root.name)

    monkeypatch.delenv("KONFAI_APPS_INSTALL_REQUIREMENTS", raising=False)
    captured = _capture_pip(monkeypatch)

    repo._install_requirements(repo._get_filenames())

    assert len(captured) == 1
    assert captured[0][6:] == ["konfai-nonexistent-xyz==1.2.3"]


def test_install_requirements_refuses_a_dependency_that_would_replace_torch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # itk-impact pins torch==2.12.* for its C++ ABI: resolving it next to another torch made pip swap
    # the torch the process had already imported. A real pip run, offline, on a local wheel that pins
    # torch the same way: the constraint on the installed torch turns the swap into a refusal naming it.
    import importlib.metadata
    import zipfile

    links = tmp_path / "links"
    links.mkdir()
    with zipfile.ZipFile(links / "konfai_probe_dep-1.0-py3-none-any.whl", "w") as wheel:
        info = "konfai_probe_dep-1.0.dist-info"
        wheel.writestr(
            f"{info}/METADATA",
            "Metadata-Version: 2.1\nName: konfai-probe-dep\nVersion: 1.0\nRequires-Dist: torch==0.0.1\n",
        )
        wheel.writestr(
            f"{info}/WHEEL", "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
        )
        wheel.writestr(f"{info}/RECORD", "")
    app_root = tmp_path / "repo" / "demo_app"
    _write_app_with_requirements(app_root, "konfai-probe-dep\n")
    repo = app_repository_module.LocalAppRepositoryFromDirectory(app_root.parent, app_root.name)
    monkeypatch.delenv("KONFAI_APPS_INSTALL_REQUIREMENTS", raising=False)
    monkeypatch.setenv("PIP_NO_INDEX", "1")
    monkeypatch.setenv("PIP_FIND_LINKS", str(links))
    monkeypatch.setenv("PIP_DISABLE_PIP_VERSION_CHECK", "1")

    with pytest.raises(AppRepositoryError) as refusal:
        repo._install_requirements(repo._get_filenames())

    torch_version = importlib.metadata.version("torch")
    assert "konfai-probe-dep 1.0 depends on torch==0.0.1" in str(refusal.value)
    assert f"torch=={torch_version}" in str(refusal.value)
    assert importlib.metadata.version("torch") == torch_version


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


def test_each_prediction_config_gets_the_cost_app_json_declares_for_its_pass(tmp_path: Path) -> None:
    """The tile pass takes the tile figures, any other config the app's own; an explicit value in the config stays."""
    manifest = {
        "vram_bytes_per_voxel": 1150,
        "ram_bytes_per_voxel": 900,
        "tiling": {"tile": "Prediction_tile.yml", "tile_vram_bytes_per_voxel": 620},
    }
    assert app_repository_module.pass_cost(manifest, "Prediction.yml") == {"vram": 1150.0, "ram": 900.0}
    assert app_repository_module.pass_cost(manifest, "Prediction_tile.yml") == {"vram": 620.0, "ram": 900.0}
    assert app_repository_module.pass_cost({}, "Prediction.yml") == {}

    repo, prediction = _local_repo_with_config(
        tmp_path, "Predictor:\n  Dataset:\n    Patch:\n      patch_size: [0, 0, 0]\n      ram_bytes_per_voxel: 10\n"
    )
    app_json = prediction.with_name("app.json")
    app_json.write_text(json.dumps({**json.loads(app_json.read_text()), **manifest}), encoding="utf-8")
    repo._set_pass_cost(str(prediction))
    patch = YAML().load(prediction.read_text())["Predictor"]["Dataset"]["Patch"]
    assert patch["vram_bytes_per_voxel"] == 1150 and patch["ram_bytes_per_voxel"] == 10


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


def test_apply_config_overrides_keeps_a_bitmask_string_with_leading_zeros(tmp_path: Path) -> None:
    # YAML reads '01' as 1: layers_mask=01 silently selected feature layer 0 instead of layer 1.
    from ruamel.yaml import YAML

    repo, prediction = _local_repo_with_config(tmp_path, _CONFIG + "      layers_mask: '1'\n      name: x\n")
    repo._apply_config_overrides(str(prediction), ["layers_mask=0000001", "name=10", "iterations=0300"])

    net = YAML().load(prediction.read_text())["Predictor"]["Model"]["RegistrationNet"]
    assert net["layers_mask"] == "0000001"
    assert net["name"] == 10  # no text lost: the binder reads it into the parameter's type
    assert net["iterations"] == 300  # an integer key stays an integer


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
# No `from __future__ import annotations`: get_parameters reads runtime annotation objects.
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
    # `values` is the model block minus structural wiring: a JSON-clean tree the CLI edits via --set.
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


def test_apply_config_overrides_refuses_a_value_outside_its_range(tmp_path: Path) -> None:
    # Range bounds were UI hints only: Slicer applied them, the CLI let a negative or huge value through.
    repo, prediction = _local_repo_with_config(tmp_path, _TYPED_MODEL_CONFIG)
    (prediction.parent / "Model.py").write_text(_TYPED_MODEL_PY, encoding="utf-8")

    repo._apply_config_overrides(str(prediction), ["spatial_samples=100000"])
    for override in ("spatial_samples=-3", "spatial_samples=100001"):
        with pytest.raises(AppRepositoryError, match="outside its range"):
            repo._apply_config_overrides(str(prediction), [override])


# A model whose feature models are a dict of dataclass entries, as the IMPACT-Reg engines declare theirs (ModelSpec),
# and whose optimizer is a loader: its block holds the arguments of another callable, not the loader's own.
_NESTED_MODEL_PY = (
    "from dataclasses import dataclass\n"
    "from typing import Annotated, Literal\n"
    "from konfai.utils.config import Range\n"
    "\n\n"
    "@dataclass\n"
    "class Spec:\n"
    "    ref: str\n"
    "    layers_mask: str = '1'\n"
    "    pca: Annotated[int, Range(0, 100)] = 0\n"
    "    norm: Literal['none', 'l2'] = 'none'\n"
    "\n\n"
    "class Loader:\n"
    "    def __init__(self, name: str = 'AdamW') -> None:\n"
    "        self.name = name\n"
    "\n\n"
    "class RegistrationNet:\n"
    "    def __init__(\n"
    "        self,\n"
    "        spatial_samples: Annotated[int, Range(0, 100000)] = 0,\n"
    "        models: dict[str, Spec] = {},\n"
    "        optimizer: Loader = Loader(),\n"
    "    ) -> None:\n"
    "        pass\n"
)

_NESTED_MODEL_CONFIG = (
    "Predictor:\n"
    "  Model:\n"
    "    classpath: Model:RegistrationNet\n"
    "    RegistrationNet:\n"
    "      spatial_samples: 2000\n"
    "      models:\n"
    "        '0':\n"
    "          ref: a:x.pt\n"
    "      optimizer:\n"
    "        lr: 0.001\n"
)


def _nested_repo(tmp_path: Path) -> tuple[object, Path]:
    repo, prediction = _local_repo_with_config(tmp_path, _NESTED_MODEL_CONFIG)
    (prediction.parent / "Model.py").write_text(_NESTED_MODEL_PY, encoding="utf-8")
    return repo, prediction


def _net(prediction: Path) -> dict:
    return YAML().load(prediction.read_text())["Predictor"]["Model"]["RegistrationNet"]


def test_a_dotted_set_is_checked_as_its_bare_name_is(tmp_path: Path) -> None:
    # The dotted spelling of a model parameter skipped the range check of its bare name: iterations=-1 was refused,
    # Predictor.Model.RegistrationNet.iterations=-1 written into the config.
    repo, prediction = _nested_repo(tmp_path)
    repo._apply_config_overrides(str(prediction), ["Predictor.Model.RegistrationNet.spatial_samples=100000"])
    with pytest.raises(AppRepositoryError, match="outside its range"):
        repo._apply_config_overrides(str(prediction), ["Predictor.Model.RegistrationNet.spatial_samples=-3"])


def test_set_adds_a_field_the_model_declares_where_the_config_leaves_it_out(tmp_path: Path) -> None:
    # A preset writes the fields of a feature model it sets, not all of them: a field it leaves at its default
    # (feature_normalization in the ConvexAdam presets) could only be tuned by replacing the whole models block.
    repo, prediction = _nested_repo(tmp_path)
    repo._apply_config_overrides(str(prediction), ["Predictor.Model.RegistrationNet.models.0.norm=l2"])
    assert _net(prediction)["models"]["0"] == {"ref": "a:x.pt", "norm": "l2"}
    with pytest.raises(AppRepositoryError, match="Did you mean 'norm'"):
        repo._apply_config_overrides(str(prediction), ["Predictor.Model.RegistrationNet.models.0.nrom=l2"])


@pytest.mark.parametrize(
    "override",
    [
        "Predictor.Model.RegistrationNet.models.0.pca=-5",  # out of its Range, which the binder reads as a hint
        "Predictor.Model.RegistrationNet.models.0.norm=L2",  # not a choice: the binder refused it once the run started
        "models={0: {ref: 'a:x.pt'}}",  # an integer key: the binder refused it once the run started
        "models={'0': {ref: 'a:x.pt', nrom: l2}}",  # a field Spec lacks: bound as its default, with a warning only
        "models={'0': {ref: 'a:x.pt', layers_mask: 01}}",  # YAML reads the bitmask as 1: the wrong layer, silently
        "Predictor.Model.RegistrationNet.models.1={layers_mask: '1'}",  # an entry without the ref it requires
    ],
)
def test_set_refuses_a_nested_value_the_run_would_refuse_or_misread(tmp_path: Path, override: str) -> None:
    repo, prediction = _nested_repo(tmp_path)
    before = prediction.read_text()
    with pytest.raises(AppRepositoryError):
        repo._apply_config_overrides(str(prediction), [override])
    assert prediction.read_text() == before


def test_set_adds_a_models_entry_and_keeps_a_declared_string_as_given(tmp_path: Path) -> None:
    repo, prediction = _nested_repo(tmp_path)
    repo._apply_config_overrides(
        str(prediction),
        [
            "Predictor.Model.RegistrationNet.models.1={ref: 'b:y.pt', layers_mask: '01'}",  # a second feature model
            "Predictor.Model.RegistrationNet.models.0.layers_mask=01",  # a declared string the config leaves out
        ],
    )
    models = _net(prediction)["models"]
    assert models["1"] == {"ref": "b:y.pt", "layers_mask": "01"} and models["0"]["layers_mask"] == "01"
    # SlicerKonfAI's spelling: the whole block as a flow mapping, every key and string quoted.
    repo._apply_config_overrides(
        str(prediction), ['models={"0": {"ref": "a:x.pt", "layers_mask": "01", "norm": "l2"}}']
    )
    assert _net(prediction)["models"] == {"0": {"ref": "a:x.pt", "layers_mask": "01", "norm": "l2"}}


def test_set_leaves_a_loader_block_to_the_binder(tmp_path: Path) -> None:
    # An optimizer block holds the torch optimizer's arguments, not the fields of its loader: it is not checked
    # against them, and a key inside it must exist, as before.
    repo, prediction = _nested_repo(tmp_path)
    repo._apply_config_overrides(
        str(prediction), ["optimizer={lr: 0.1}", "Predictor.Model.RegistrationNet.optimizer.lr=0.01"]
    )
    assert _net(prediction)["optimizer"] == {"lr": 0.01}
    with pytest.raises(AppRepositoryError, match="does not exist"):
        repo._apply_config_overrides(str(prediction), ["Predictor.Model.RegistrationNet.optimizer.betas=[0.9, 0.99]"])


def test_set_refuses_a_value_that_is_not_yaml(tmp_path: Path) -> None:
    # A malformed value ended the run on a ruamel traceback.
    repo, prediction = _nested_repo(tmp_path)
    with pytest.raises(AppRepositoryError, match="is not YAML"):
        repo._apply_config_overrides(str(prediction), ["spatial_samples=[1, 2"])


def test_get_parameters_constraints_from_an_installed_package_classpath(tmp_path: Path) -> None:
    """A preset that keeps only config + weights points its classpath at an INSTALLED package module (its
    requirements provide it), not a bundled .py. get_parameters must still read the constraints/descriptions
    from that module's typed signature: otherwise moving the model code out of the bundle would silently
    strip an agent's whole parameter surface."""
    import importlib
    import sys

    module_name = "konfai_test_pkg_model_xyz"
    pkg_dir = tmp_path / "site"
    pkg_dir.mkdir()
    (pkg_dir / f"{module_name}.py").write_text(_TYPED_MODEL_PY, encoding="utf-8")
    config = _TYPED_MODEL_CONFIG.replace(
        "classpath: Model:RegistrationNet", f"classpath: {module_name}:RegistrationNet"
    )

    sys.path.insert(0, str(pkg_dir))
    importlib.invalidate_caches()
    try:
        repo, _prediction = _local_repo_with_config(tmp_path, config)  # NOTE: no Model.py written into the bundle
        result = repo.get_parameters()
    finally:
        sys.path.remove(str(pkg_dir))
        sys.modules.pop(module_name, None)

    # Same constraints as the bundle-local case: the classpath resolved to the installed module.
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
    # A server predating the 'finetunable' field omits it; fall back to the inference capability
    # (fine-tune offered for any inference-capable app).
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


def _write_app_with_two_checkpoints(root: Path, declare: bool) -> Path:
    app_root = root / "apps" / "Repacked"
    app_root.mkdir(parents=True)
    metadata = {
        "display_name": "Repacked",
        "description": "d",
        "short_description": "s",
        "tta": 0,
        "mc_dropout": 0,
    }
    if declare:
        metadata["models"] = ["new.pt"]
    (app_root / "app.json").write_text(json.dumps(metadata))
    (app_root / "Prediction.yml").write_text("Predictor: {}\n")
    (app_root / "first.pt").write_bytes(b"first")
    (app_root / "new.pt").write_bytes(b"new")
    return app_root


def test_default_inference_takes_the_checkpoints_app_json_declares(tmp_path: Path, monkeypatch) -> None:
    """A repackaged bundle can hold the previous export's checkpoint beside the declared one; the
    default inference once enumerated the .pt files and picked the old one."""
    monkeypatch.setenv("KONFAI_APPS_INSTALL_REQUIREMENTS", "0")
    app_root = _write_app_with_two_checkpoints(tmp_path, declare=True)
    repo = app_repository_module.LocalAppRepositoryFromDirectory(app_root.parent, app_root.name)
    models_path, _prediction, _codes = repo.download_inference(1, [], "Prediction.yml")
    assert [path.name for path in models_path] == ["new.pt"]
    with pytest.raises(app_repository_module.AppRepositoryError, match="declares 1"):
        repo.download_inference(2, [], "Prediction.yml")


def test_legacy_metadata_without_models_still_enumerates_the_checkpoints(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("KONFAI_APPS_INSTALL_REQUIREMENTS", "0")
    app_root = _write_app_with_two_checkpoints(tmp_path, declare=False)
    repo = app_repository_module.LocalAppRepositoryFromDirectory(app_root.parent, app_root.name)
    models_path, _prediction, _codes = repo.download_inference(2, [], "Prediction.yml")
    assert sorted(path.name for path in models_path) == ["first.pt", "new.pt"]


def test_native_inference_leaves_the_declared_portable_assets_in_the_repository(tmp_path: Path, monkeypatch) -> None:
    """A bundle that also ships its ONNX export copied model.onnx and its tensor data into every
    native run's workspace; the files app.json declares under portable_assets are the portable
    runtime's and stay behind. A bundle that declares none keeps the inclusive contract."""
    monkeypatch.setenv("KONFAI_APPS_INSTALL_REQUIREMENTS", "0")
    app_root = _write_app_with_two_checkpoints(tmp_path, declare=True)
    (app_root / "model.onnx").write_bytes(b"onnx")
    (app_root / "model.onnx.data").write_bytes(b"tensors")
    (app_root / "lookup.json").write_text("{}")
    repo = app_repository_module.LocalAppRepositoryFromDirectory(app_root.parent, app_root.name)
    _models, _prediction, assets = repo.download_inference(1, [], "Prediction.yml")
    assert {name for name, _path in assets} >= {"model.onnx", "model.onnx.data", "lookup.json"}

    metadata = json.loads((app_root / "app.json").read_text())
    metadata["portable_assets"] = ["model.onnx", "model.onnx.data"]
    (app_root / "app.json").write_text(json.dumps(metadata))
    repo = app_repository_module.LocalAppRepositoryFromDirectory(app_root.parent, app_root.name)
    _models, _prediction, assets = repo.download_inference(1, [], "Prediction.yml")
    names = {name for name, _path in assets}
    assert "lookup.json" in names and not names & {"model.onnx", "model.onnx.data"}


def test_a_bundle_with_a_declared_helper_package_imports_it_from_a_fresh_workspace(tmp_path: Path) -> None:
    # The relocatable-bundle contract: the helper package declared at export time reaches the run
    # workspace with the rest of the bundle, and Model.py imports it in a process that knows
    # nothing of the workspace the bundle was packaged from.
    import subprocess

    from konfai_apps.bundle import assemble_bundle

    source = tmp_path / "source"
    (source / "helpers").mkdir(parents=True)
    (source / "helpers" / "__init__.py").write_text("")
    (source / "helpers" / "util.py").write_text("SCALE = 3\n", encoding="utf-8")
    (source / "Model.py").write_text("from helpers.util import SCALE\n\nVALUE = SCALE * 2\n", encoding="utf-8")
    (source / "Prediction.yml").write_text("Predictor:\n  Model:\n    classpath: Model:Net\n", encoding="utf-8")
    (source / "CV_0.pt").write_bytes(b"weights")
    (source / "app.json").write_text(
        json.dumps(
            {"display_name": "Demo", "description": "Demo", "short_description": "Demo", "tta": 0, "mc_dropout": 0}
        ),
        encoding="utf-8",
    )
    bundle = assemble_bundle(
        "Relocatable",
        tmp_path / "bundles",
        source / "app.json",
        [str(source / "Prediction.yml")],
        [str(source / "CV_0.pt")],
        model_py=str(source / "Model.py"),
        support_files={"helpers": "helpers"},
        support_root=source,
    )

    repo = app_repository_module.LocalAppRepositoryFromDirectory(bundle.parent, bundle.name)
    _, _, codes = repo.download_inference(1, [], "Prediction.yml")
    assert {name for name, _ in codes} >= {"Model.py", "helpers/__init__.py", "helpers/util.py"}

    workspace = tmp_path / "run"
    workspace.mkdir()
    for name, path in codes:  # what install_inference does with the same list
        (workspace / name).parent.mkdir(parents=True, exist_ok=True)
        (workspace / name).write_bytes(path.read_bytes())
    completed = subprocess.run(
        [sys.executable, "-c", "import Model; print(Model.VALUE)"],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=True,
        env={"PATH": "/usr/bin:/bin"},
    )
    assert completed.stdout.strip() == "6"


def test_disabling_uncertainty_drops_the_replaced_reduction_block(tmp_path: Path) -> None:
    """Uncertainty off swaps the output reduction to Mean: the old operator's argument block goes with
    it, else the run warns of a key nothing reads (`OutputDataset.Concat`)."""
    from ruamel.yaml import YAML

    prediction = tmp_path / "Prediction.yml"
    prediction.write_text(
        "Predictor:\n"
        "  combine: Concat\n"
        "  outputs_dataset:\n"
        "    Tanh:\n"
        "      OutputDataset:\n"
        "        reduction: Concat\n"
        "        Concat: {}\n"
        "        after_reduction_transforms:\n"
        "          InferenceStack:\n"
        "            mode: mean\n"
    )
    app_repository_module.LocalAppRepository._disable_uncertainty(None, str(prediction))  # type: ignore[arg-type]

    with open(prediction) as file:
        output = YAML().load(file)["Predictor"]["outputs_dataset"]["Tanh"]["OutputDataset"]
    assert output["reduction"] == "Mean"
    assert "Concat" not in output
    assert "InferenceStack" not in output["after_reduction_transforms"]


def test_a_model_argument_the_preset_leaves_out_is_still_tunable(tmp_path: Path) -> None:
    # A preset's YAML lists the knobs it sets; the others keep their defaults and must stay reachable by --set
    # and shown by get_parameters (FireANTs' moments_init, linear_method, mode... are absent from its presets).
    from ruamel.yaml import YAML

    config = _TYPED_MODEL_CONFIG.replace("      note: hello\n", "")
    repo, prediction = _local_repo_with_config(tmp_path, config)
    (prediction.parent / "Model.py").write_text(_TYPED_MODEL_PY, encoding="utf-8")

    assert repo.get_parameters()["values"]["note"] == ""
    repo._apply_config_overrides(str(prediction), ["note=tuned"])
    assert YAML().load(prediction.read_text())["Predictor"]["Model"]["RegistrationNet"]["note"] == "tuned"
    with pytest.raises(AppRepositoryError):  # a typo is still refused
        repo._apply_config_overrides(str(prediction), ["noet=tuned"])


def test_install_requirements_installs_the_no_deps_ones_without_their_dependencies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # FireANTs pins a SimpleITK that has no wheel for recent Pythons: its preset lists it under requirements_no_deps,
    # and what it needs in requirements.txt.
    app_root = tmp_path / "repo" / "demo_app"
    _write_app_with_requirements(app_root, "konfai-nonexistent-dep==1.0\n")
    manifest = json.loads((app_root / "app.json").read_text(encoding="utf-8"))
    manifest["requirements_no_deps"] = ["konfai-nonexistent-pinned>=1.5,<1.6"]
    (app_root / "app.json").write_text(json.dumps(manifest), encoding="utf-8")
    repo = app_repository_module.LocalAppRepositoryFromDirectory(app_root.parent, app_root.name)
    monkeypatch.delenv("KONFAI_APPS_INSTALL_REQUIREMENTS", raising=False)
    captured = _capture_pip(monkeypatch)

    repo._install_requirements(repo._get_filenames())

    assert [command[4] for command in captured] == ["-c", "--no-deps"]
    assert captured[0][-1] == "konfai-nonexistent-dep==1.0" and captured[1][-1] == "konfai-nonexistent-pinned>=1.5,<1.6"


def test_max_voxels_lands_in_the_prediction_patch(tmp_path: Path) -> None:
    """--max-voxels writes Patch.max_voxels, which a --set could not: the key is KonfAI's, not the app config's."""
    from konfai_apps.remote_options import collect_remote_options

    config = tmp_path / "Prediction.yml"
    config.write_text("Predictor:\n  Dataset:\n    Patch:\n      patch_size: [0, 0, 0]\n    batch_size: 1\n")
    app_repository_module.LocalAppRepository._set_patch_size_and_batch_size(None, str(config), max_voxels=5000)

    assert YAML(typ="safe").load(config.read_text())["Predictor"]["Dataset"]["Patch"]["max_voxels"] == 5000
    assert collect_remote_options("infer", {"max_voxels": 5000}) == {"max_voxels": 5000}
