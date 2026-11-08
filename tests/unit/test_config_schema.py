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
from pathlib import Path

import jsonschema
import konfai.main as main_module
import pytest
from konfai.config_schema import WORKFLOWS, config_errors, config_schema
from ruamel.yaml import YAML

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
ROOTS = {root: filename for filename, (root, _) in WORKFLOWS.items()}


def _errors(tree: dict) -> list[tuple[str, str]]:
    return [(mistake.path, mistake.message) for mistake in config_errors(tree, ROOTS[next(iter(tree))])]


@pytest.mark.parametrize("filename", sorted(WORKFLOWS))
def test_each_schema_is_a_valid_draft_7_schema(filename: str) -> None:
    jsonschema.Draft7Validator.check_schema(config_schema(filename))


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


def test_each_mistake_is_reported_at_its_key_with_the_nearest_valid_name() -> None:
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
    list_stage = _segmentation()
    _ct_chain(list_stage)["transforms"] = ["Standardise"]
    missing_argument = _segmentation()
    _ct_chain(missing_argument)["transforms"]["OneHot"] = {}
    integer_key = _segmentation()
    integer_key["Trainer"][42] = 1

    chain = "Trainer.Dataset.groups_src.CT.groups_dest.CT.transforms"
    assert _errors(misspelled) == [("Trainer", "unknown key 'epoch' (did you mean 'epochs'?)")]
    assert _errors(wrong_type) == [("Trainer.epochs", "'ten' is not an integer")]
    assert _errors(unknown_stage) == [(chain, "unknown 'Standardise' (did you mean 'Standardize'?)")]
    assert _errors(unknown_argument) == [(f"{chain}.Standardize", "unknown key 'means' (did you mean 'mean'?)")]
    assert _errors(repeated_stage) == [(f"{chain}.Standardize#2", "unknown key 'means' (did you mean 'mean'?)")]
    assert _errors(unknown_repeated_stage) == [(chain, "unknown 'Standardise#2' (did you mean 'Standardize'?)")]
    assert _errors(list_stage) == [(f"{chain}.0", "unknown 'Standardise' (did you mean 'Standardize'?)")]
    assert _errors(missing_argument) == [(f"{chain}.OneHot", "missing key 'num_classes'")]
    assert _errors(integer_key) == [("Trainer", "unknown key '42'")]


def test_a_misspelled_key_blocks_only_beside_the_key_it_would_default() -> None:
    misspelled = _segmentation()
    misspelled["Trainer"]["epoch"] = misspelled["Trainer"].pop("epochs")
    beside = _segmentation()
    beside["Trainer"]["epoch"] = 3
    filename = ROOTS["Trainer"]
    assert [mistake.unread for mistake in config_errors(misspelled, filename)] == [False]
    assert [mistake.unread for mistake in config_errors(beside, filename)] == [True]


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
