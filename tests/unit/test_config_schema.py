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

"""The JSON Schemas of the workflow configs accept every config the binder accepts, and flag a mistake."""

from __future__ import annotations

import copy
import sys
from functools import cache
from pathlib import Path

import jsonschema
import konfai.main as main_module
import pytest
from konfai.config_schema import WORKFLOWS, config_schema
from ruamel.yaml import YAML

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
ROOTS = {root: filename for filename, (root, _) in WORKFLOWS.items()}


@cache
def _validator(filename: str) -> jsonschema.Draft7Validator:
    schema = config_schema(filename)
    jsonschema.Draft7Validator.check_schema(schema)
    return jsonschema.Draft7Validator(schema)


def _errors(tree: dict) -> list[str]:
    """Where each error is, as an editor reports it: at the deepest key that explains it."""
    validator = _validator(ROOTS[next(iter(tree))])
    errors = [jsonschema.exceptions.best_match([error]) for error in validator.iter_errors(tree)]
    return ["/".join(map(str, error.absolute_path)) for error in errors]


def _load(path: Path) -> dict:
    return YAML(typ="safe").load(path.read_text())


EXAMPLE_CONFIGS = sorted(
    path
    for path in EXAMPLES.rglob("*.yml")
    if isinstance(tree := _load(path), dict) and next(iter(tree), None) in ROOTS
)


@pytest.mark.parametrize("path", EXAMPLE_CONFIGS, ids=lambda path: str(path.relative_to(EXAMPLES)))
def test_every_example_config_is_valid(path: Path) -> None:
    assert _errors(_load(path)) == []


@pytest.mark.parametrize("command", ["TRAIN", "PREDICTION", "EVALUATION", "TRANSFORM"])
def test_the_defaults_init_writes_are_valid(command: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    filename = next(name for name, (root, _) in WORKFLOWS.items() if main_module._INIT_TARGETS[command][1] == root)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["konfai", command, "--init", "-q", "-c", filename])
    main_module.main()
    assert _errors(_load(tmp_path / filename)) == []


def _segmentation() -> dict:
    return copy.deepcopy(_load(EXAMPLES / "Segmentation" / "Config.yml"))


def _ct_chain(tree: dict) -> dict:
    return tree["Trainer"]["Dataset"]["groups_src"]["CT"]["groups_dest"]["CT"]


def test_a_mistake_is_flagged_where_it_is() -> None:
    misspelled = _segmentation()
    misspelled["Trainer"]["epoch"] = misspelled["Trainer"].pop("epochs")
    wrong_type = _segmentation()
    wrong_type["Trainer"]["epochs"] = "ten"
    unknown_stage = _segmentation()
    _ct_chain(unknown_stage)["transforms"]["Standardise"] = None
    unknown_argument = _segmentation()
    _ct_chain(unknown_argument)["transforms"]["Standardize"]["means"] = 0
    repeated_stage = _segmentation()
    _ct_chain(repeated_stage)["transforms"]["Standardize#2"] = {"means": 0}
    unknown_repeated_stage = _segmentation()
    _ct_chain(unknown_repeated_stage)["transforms"]["Standardise#2"] = None
    missing_argument = _segmentation()
    _ct_chain(missing_argument)["transforms"]["OneHot"] = {}

    assert _errors(misspelled) == ["Trainer"]
    assert _errors(wrong_type) == ["Trainer/epochs"]
    assert _errors(unknown_stage) == ["Trainer/Dataset/groups_src/CT/groups_dest/CT/transforms"]
    assert _errors(unknown_argument) == ["Trainer/Dataset/groups_src/CT/groups_dest/CT/transforms/Standardize"]
    assert _errors(repeated_stage) == ["Trainer/Dataset/groups_src/CT/groups_dest/CT/transforms/Standardize#2"]
    assert _errors(unknown_repeated_stage) == ["Trainer/Dataset/groups_src/CT/groups_dest/CT/transforms"]
    assert _errors(missing_argument) == ["Trainer/Dataset/groups_src/CT/groups_dest/CT/transforms/OneHot"]


def test_the_spellings_the_binder_accepts_are_valid() -> None:
    tree = _segmentation()
    chain = _ct_chain(tree)
    chain["transforms"] = [
        {"Clip": {"min_value": "-1024"}},
        {"Clip": None},
        "Canonical",
        {"monai.transforms:ScaleIntensity": {}},
    ]
    tree["Trainer"]["epochs"] = "5"
    tree["Trainer"]["manual_seed"] = "None"
    assert _errors(tree) == []
