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

"""Presets: ``APP#PRESET``, named lists of ``--set`` applied before the run's own."""

import json
import sys
from pathlib import Path

import pytest
from konfai_apps import app_repository as app_repository_module
from konfai_apps import cli
from konfai_apps.errors import AppRepositoryError
from konfai_apps.presets import Preset, read_presets, split_preset, user_presets_file
from ruamel.yaml import YAML

_CONFIG = (
    "Predictor:\n"
    "  Model:\n"
    "    classpath: Model:RegistrationNet\n"
    "    RegistrationNet:\n"
    "      iterations: 80\n"
    "      grid_spacing: 6\n"
    "      stages: [coarse, fine]\n"
)

_PRESETS = {
    "composite": {"description": "Coarse then fine", "set": []},
    "coarse": {"description": "The coarse search alone", "set": ["stages=[coarse]"]},
    "fine_grid": {"description": "A finer grid, more steps", "set": ["grid_spacing=3", "iterations=150"]},
}


@pytest.fixture(autouse=True)
def _own_caches(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A Hugging Face cache and a presets folder of the test's own."""
    monkeypatch.setattr(app_repository_module.constants, "HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setenv("KONFAI_PRESETS_DIR", str(tmp_path / "presets"))
    app_repository_module._release_tag.cache_clear()


def _app(tmp_path: Path, presets: dict | None = None) -> Path:
    root = tmp_path / "repo" / "ConvexAdam"
    root.mkdir(parents=True)
    manifest = {"display_name": "ConvexAdam", "description": "d", "short_description": "d", "tta": 0, "mc_dropout": 0}
    if presets is not None:
        manifest["presets"] = presets
    (root / "app.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "Prediction.yml").write_text(_CONFIG, encoding="utf-8")
    return root


def _model(path: Path) -> dict:
    return YAML().load(path.read_text())["Predictor"]["Model"]["RegistrationNet"]


@pytest.mark.parametrize(
    ("reference", "expected"),
    [
        ("VBoussot/ImpactReg:ConvexAdam#composite", ("VBoussot/ImpactReg:ConvexAdam", "composite")),
        ("VBoussot/ImpactReg:ConvexAdam", ("VBoussot/ImpactReg:ConvexAdam", None)),
        ("host:8000:ConvexAdam#impact_cbct|secret", ("host:8000:ConvexAdam|secret", "impact_cbct")),
        ("VBoussot/ImpactReg@v1.2:ConvexAdam#v1.0-a", ("VBoussot/ImpactReg@v1.2:ConvexAdam", "v1.0-a")),
    ],
)
def test_a_reference_splits_into_its_app_and_its_preset(reference: str, expected: tuple) -> None:
    assert split_preset(reference) == expected


def test_an_existing_folder_with_a_hash_in_its_name_is_a_folder(tmp_path: Path) -> None:
    folder = tmp_path / "apps#2026"
    folder.mkdir()
    assert split_preset(str(folder)) == (str(folder), None)


@pytest.mark.parametrize("reference", ["repo:App#", "repo:App#two words", "#composite"])
def test_a_malformed_preset_name_is_refused(reference: str) -> None:
    with pytest.raises(AppRepositoryError, match="names no preset"):
        split_preset(reference)


@pytest.mark.parametrize(
    "entries",
    [
        ["composite"],
        {"composite": {"set": ["iterations=3"], "tuned_on": "a dataset"}},  # only description and set
        {"composite": {"set": "iterations=3"}},
        {"composite": {"set": ["iterations"]}},
        {"two words": {"set": []}},
    ],
)
def test_a_preset_is_a_description_and_a_list_of_name_value(entries: object) -> None:
    with pytest.raises(AppRepositoryError):
        read_presets(entries, "app.json")


def test_choosing_a_preset_shows_the_values_the_run_starts_from(tmp_path: Path) -> None:
    root = _app(tmp_path, _PRESETS)
    info = app_repository_module.get_app_repository_info(f"{root}#fine_grid", False)
    assert info.get_preset() == Preset("fine_grid", "A finer grid, more steps", ("grid_spacing=3", "iterations=150"))
    values = info.get_parameters()["values"]
    assert values["grid_spacing"] == 3 and values["iterations"] == 150
    base = app_repository_module.get_app_repository_info(str(root), False)
    assert base.get_preset() is None and base.get_parameters()["values"]["grid_spacing"] == 6
    assert sorted(base.get_presets()) == ["coarse", "composite", "fine_grid"]


def test_an_unknown_preset_is_refused_with_the_app_s_presets(tmp_path: Path) -> None:
    root = _app(tmp_path, _PRESETS)
    with pytest.raises(AppRepositoryError, match="Did you mean 'coarse'") as raised:
        app_repository_module.get_app_repository_info(f"{root}#corse", False)
    assert "coarse, composite, fine_grid" in str(raised.value)


def test_a_preset_that_no_longer_fits_its_app_is_refused_when_chosen(tmp_path: Path) -> None:
    root = _app(tmp_path, {"broken": {"description": "", "set": ["no_such_knob=1"]}})
    app_repository_module.get_app_repository_info(str(root), False)  # the app itself still runs
    with pytest.raises(AppRepositoryError, match="preset 'broken'"):
        app_repository_module.get_app_repository_info(f"{root}#broken", False)


def test_the_preset_comes_first_and_the_run_s_own_set_wins(tmp_path: Path) -> None:
    root = _app(tmp_path, _PRESETS)
    info = app_repository_module.get_app_repository_info(f"{root}#fine_grid", False)
    config = tmp_path / "Prediction.yml"
    config.write_text(_CONFIG, encoding="utf-8")
    info._apply_config_overrides(str(config), ["iterations=200"])
    model = _model(config)
    assert model["grid_spacing"] == 3 and model["iterations"] == 200


def test_saving_a_preset_of_a_local_app_writes_its_app_json(tmp_path: Path) -> None:
    root = _app(tmp_path, _PRESETS)
    info = app_repository_module.get_app_repository_info(str(root), False)
    written = info.save_preset("hippo", "Small crops", ["grid_spacing=3", "iterations=150", "grid_spacing=2"])
    assert written == root / "app.json"
    saved = json.loads(written.read_text())["presets"]["hippo"]
    assert saved == {"description": "Small crops", "set": ["grid_spacing=2", "iterations=150"]}  # the last value, once
    assert _model(root / "Prediction.yml")["grid_spacing"] == 6  # the base config is untouched
    with pytest.raises(AppRepositoryError, match="no_such_knob"):
        info.save_preset("bad", "", ["no_such_knob=1"])


def test_the_preset_save_command_starts_from_another_preset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _app(tmp_path, _PRESETS)
    monkeypatch.setattr(
        sys, "argv", ["konfai-apps", "preset", "save", f"{root}#mine", "--from", "fine_grid", "--set", "iterations=90"]
    )
    cli.main_apps()
    saved = json.loads((root / "app.json").read_text())["presets"]["mine"]
    assert saved["set"] == ["grid_spacing=3", "iterations=90"]


def test_a_preset_of_one_s_own_never_hides_a_published_one(tmp_path: Path) -> None:
    root = _app(tmp_path, _PRESETS)
    own = tmp_path / "own.json"

    class WithOwnPresets(app_repository_module.LocalAppRepositoryFromDirectory):
        def _user_presets_file(self) -> Path:
            return own

    info = WithOwnPresets(root.parent, root.name)
    assert info.save_preset("mine", "", ["iterations=10"]) == own
    assert json.loads(own.read_text()) == {"mine": {"description": "", "set": ["iterations=10"]}}
    assert "mine" in info.get_presets() and "fine_grid" in info.get_presets()
    with pytest.raises(AppRepositoryError, match="publishes a preset 'coarse'"):
        info.save_preset("coarse", "", ["iterations=10"])
    own.write_text(json.dumps({"coarse": {"description": "", "set": []}}), encoding="utf-8")
    with pytest.raises(AppRepositoryError, match="never hides a published one"):
        info.get_presets()


def test_a_user_s_presets_of_a_hub_app_are_kept_by_repository_and_app(tmp_path: Path) -> None:
    assert (
        user_presets_file("VBoussot/ImpactReg", "ConvexAdam") == tmp_path / "presets/VBoussot/ImpactReg/ConvexAdam.json"
    )


def test_saving_defaults_never_bakes_the_chosen_preset_in(tmp_path: Path) -> None:
    root = _app(tmp_path, _PRESETS)
    info = app_repository_module.get_app_repository_info(f"{root}#fine_grid", False)
    info.save_default_parameters(["iterations=99"])
    model = _model(root / "Prediction.yml")
    assert model["iterations"] == 99 and model["grid_spacing"] == 6


def test_an_exported_preset_is_the_copy_s_base_and_its_presets_stay_behind(tmp_path: Path) -> None:
    root = _app(tmp_path, _PRESETS)
    info = app_repository_module.get_app_repository_info(f"{root}#coarse", False)
    info.export_app(tmp_path / "export")
    assert _model(tmp_path / "export" / "Prediction.yml")["stages"] == ["coarse"]
    assert "presets" not in json.loads((tmp_path / "export" / "app.json").read_text())
    app_repository_module.get_app_repository_info(str(root), False).export_app(tmp_path / "plain")
    assert "presets" in json.loads((tmp_path / "plain" / "app.json").read_text())  # nothing baked: they travel


def test_show_lists_the_presets_and_marks_the_chosen_one(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    root = _app(tmp_path, _PRESETS)
    cli._show(f"{root}#coarse")
    out = capsys.readouterr().out
    assert "Presets: ConvexAdam#NAME" in out
    assert " *coarse  The coarse search alone" in out and "--set stages=[coarse]" in out
    assert "Parameters, after preset 'coarse'" in out


def test_a_server_allows_the_presets_of_the_apps_it_serves(monkeypatch: pytest.MonkeyPatch) -> None:
    app_server = pytest.importorskip("konfai_apps.app_server")
    from fastapi import HTTPException

    monkeypatch.setattr(app_server, "_APPS", ["VBoussot/ImpactReg:ConvexAdam"])
    app_server._require_configured_app("VBoussot/ImpactReg:ConvexAdam#coarse")
    with pytest.raises(HTTPException):
        app_server._require_configured_app("VBoussot/ImpactReg:FireANTs#coarse")


def test_a_remote_app_s_preset_travels_encoded_and_the_server_lists_its_presets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    urls: list[str] = []
    listing = {
        "app": "VBoussot/ImpactReg:ConvexAdam",
        "available": True,
        "has_capabilities": [True, False, False],
        "display_name": "ConvexAdam",
        "description": "d",
        "short_description": "d",
        "checkpoints_name": [],
        "checkpoints_name_available": [],
        "maximum_tta": 0,
        "mc_dropout": 0,
        "inputs": {},
        "outputs": {},
        "inputs_evaluations": {},
        "presets": _PRESETS,
    }

    class Response:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return listing

    def get(url: str, **_: object) -> Response:
        urls.append(url)
        return Response()

    monkeypatch.setattr(app_repository_module.requests, "get", get)
    info = app_repository_module.get_app_repository_info("host:8000:VBoussot/ImpactReg:ConvexAdam#coarse", False)
    assert info.get_preset() == Preset("coarse", "The coarse search alone", ("stages=[coarse]",))
    assert urls == ["http://host:8000/repo_apps/VBoussot/ImpactReg:ConvexAdam"]
    assert app_repository_module.quote("ConvexAdam#coarse", safe="/:@") == "ConvexAdam%23coarse"


def test_a_run_records_what_it_ran_and_one_identity_per_configuration(tmp_path: Path) -> None:
    root = _app(tmp_path, _PRESETS)
    chosen = app_repository_module.get_app_repository_info(f"{root}#fine_grid", False)
    record = chosen.record(["iterations=200"])
    assert record["app"] == str(root) and record["revision"] is None and record["preset"] == "fine_grid"
    assert record["set"] == ["grid_spacing=3", "iterations=150", "iterations=200"]  # in the order applied
    base = app_repository_module.get_app_repository_info(str(root), False)
    # the same configuration, reached another way: the same identity
    assert base.record(["iterations=200", "grid_spacing=3"])["identity"] == record["identity"]
    assert base.record(["grid_spacing=3"])["identity"] != record["identity"]


def test_a_hub_app_s_revision_is_the_snapshot_it_runs_from(tmp_path: Path) -> None:
    info = object.__new__(app_repository_module.LocalAppRepositoryFromHF)
    info._repo_id = "VBoussot/ImpactReg"
    snapshot = tmp_path / "hub" / "models--VBoussot--ImpactReg" / "snapshots" / "09d2a191" / "ConvexAdam" / "app.json"
    info._download = lambda filename: snapshot  # type: ignore[method-assign]
    assert info.get_revision() == "09d2a191"
