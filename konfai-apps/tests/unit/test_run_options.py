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

"""``--output-format`` and ``--seed``: the run's own choices, each the ``--set`` it stands for."""

import json
from pathlib import Path

import pytest
from konfai_apps import app_repository as app_repository_module
from konfai_apps import cli
from konfai_apps.app_repository import apply_overrides, run_overrides
from konfai_apps.errors import AppRepositoryError
from ruamel.yaml import YAML

_CONFIG = (
    "Predictor:\n"
    "  Model:\n"
    "    classpath: Model:RegistrationNet\n"
    "    RegistrationNet:\n"
    "      iterations: 80\n"
    "  outputs_dataset:\n"
    "    DisplacementField:\n"
    "      OutputDataset:\n"
    "        dataset_filename: Transform:itktransform\n"
    "        group: Transform\n"
    "    Seg:\n"
    "      OutputDataset:\n"
    "        dataset_filename: ./Dataset:a:mha\n"
    "  manual_seed: 42\n"
)


def _config() -> dict:
    return YAML(typ="safe").load(_CONFIG)


def test_one_format_for_every_output_keeps_each_path_and_flag() -> None:
    assert run_overrides(_config(), ["omezarr"]) == [
        "Predictor.outputs_dataset.DisplacementField.OutputDataset.dataset_filename=Transform:omezarr",
        "Predictor.outputs_dataset.Seg.OutputDataset.dataset_filename=./Dataset:a:omezarr",
    ]


def test_a_format_for_one_output_and_a_pyramid_level() -> None:
    assert run_overrides(_config(), ["Seg=nii.gz", "DisplacementField=omezarr@1"]) == [
        "Predictor.outputs_dataset.Seg.OutputDataset.dataset_filename=./Dataset:a:nii.gz",
        "Predictor.outputs_dataset.DisplacementField.OutputDataset.dataset_filename=Transform:omezarr@1",
    ]


def test_the_seed() -> None:
    assert run_overrides(_config(), seed=7) == ["Predictor.manual_seed=7"]
    assert run_overrides({"Trainer": {"manual_seed": 1}}, seed=3) == ["Trainer.manual_seed=3"]
    assert run_overrides(_config()) == []


@pytest.mark.parametrize("root", ["Predictor", "Trainer"])
def test_the_seed_applies_to_a_config_that_leaves_manual_seed_to_its_default(root: str) -> None:
    config = {root: {"Model": {"classpath": "Model:Net", "Net": {}}}}
    apply_overrides(config, run_overrides(config, seed=7), lambda: None)
    assert config[root]["manual_seed"] == 7
    with pytest.raises(AppRepositoryError, match="does not exist"):  # the one root key a config may omit
        apply_overrides(config, [f"{root}.other_seed=7"], lambda: None)


def test_a_format_for_every_output_of_an_app_that_writes_none_is_refused() -> None:
    with pytest.raises(AppRepositoryError, match="no output file"):
        run_overrides({"Predictor": {"Model": {"classpath": "Model:Net", "Net": {}}}}, ["omezarr"])


@pytest.mark.parametrize(
    ("spec", "message"),
    [("omezar", "Did you mean 'omezarr'"), ("Sge=mha", "Did you mean 'Seg'"), ("Moved=mha", "DisplacementField, Seg")],
)
def test_a_format_or_an_output_the_app_cannot_write_is_refused(spec: str, message: str) -> None:
    with pytest.raises(AppRepositoryError) as raised:
        run_overrides(_config(), [spec])
    assert message in str(raised.value)


def test_they_resolve_as_any_other_set_does() -> None:
    config = _config()
    apply_overrides(config, run_overrides(config, ["omezarr"], seed=5), lambda: None)
    assert config["Predictor"]["outputs_dataset"]["DisplacementField"]["OutputDataset"]["dataset_filename"] == (
        "Transform:omezarr"
    )
    assert config["Predictor"]["manual_seed"] == 5


def test_they_come_after_the_run_s_own_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_repository_module.constants, "HF_HUB_CACHE", str(tmp_path / "hub"))
    root = tmp_path / "repo" / "Engine"
    root.mkdir(parents=True)
    manifest = {"display_name": "E", "description": "d", "short_description": "d", "tta": 0, "mc_dropout": 0}
    (root / "app.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "Prediction.yml").write_text(_CONFIG, encoding="utf-8")
    overrides = cli._with_run_options(str(root), None, "Prediction.yml", ["iterations=3"], ["Seg=mha"], 9)
    assert overrides == [
        "iterations=3",
        "Predictor.manual_seed=9",
        "Predictor.outputs_dataset.Seg.OutputDataset.dataset_filename=./Dataset:a:mha",
    ]
    assert cli._with_run_options(str(root), None, "Prediction.yml", None, None, None) is None
