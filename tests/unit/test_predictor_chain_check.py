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

"""PREDICTION warns when a model input is not preprocessed the way its checkpoint trained on it.

The Synthesis example shipped ``Standardize(mask: None)`` in training against
``Standardize(mask: MASK)`` in prediction: the same checkpoint, 409 HU of MAE instead of 98, and
nothing failed. The check reads the resolved config the training run left in ``Statistics/`` and
compares it, stage by stage, with the live one.
"""

import shutil
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from konfai.predictor import Predictor
from ruamel.yaml import YAML

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def _config(root: str, transforms: dict[str, dict[str, Any]] | None) -> dict[str, Any]:
    """The smallest tree the check reads: one source group feeding the model through ``transforms``."""
    chain = {"transforms": transforms if transforms else "None", "patch_transforms": "None", "is_input": True}
    return {root: {"Dataset": {"groups_src": {"CT": {"groups_dest": {"CT": chain}}}}}}


def _place(config: Path | dict[str, Any], target: Path) -> None:
    """Put a config at ``target``: an existing file copied byte for byte, or a tree dumped."""
    if isinstance(config, Path):
        shutil.copyfile(config, target)
    else:
        YAML().dump(config, target)


def _workspace(
    tmp_path: Path,
    trained: Path | dict[str, Any],
    applied: Path | dict[str, Any],
    runs: Sequence[str] = ("RUN",),
) -> list[Path]:
    """A workspace as TRAIN leaves it: a checkpoint per run beside the resolved config of that run.

    Answers the checkpoints, in the order a prediction would name them.
    """
    for run in runs:
        for directory in (tmp_path / "Checkpoints" / run, tmp_path / "Statistics" / run):
            directory.mkdir(parents=True, exist_ok=True)
        (tmp_path / "Checkpoints" / run / "model.pt").write_bytes(b"")
        _place(trained, tmp_path / "Statistics" / run / "Config_0_1.yml")
    _place(applied, tmp_path / "Prediction.yml")
    return [tmp_path / "Checkpoints" / run / "model.pt" for run in runs]


def _report(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    checkpoints: list[Path],
    check_training_transforms: bool = True,
) -> str:
    """What the check prints for a run loading ``checkpoints``; it reads nothing else off the run."""
    monkeypatch.setenv("KONFAI_config_file", str(tmp_path / "Prediction.yml"))
    monkeypatch.setenv("KONFAI_ROOT", "Predictor")
    predictor = SimpleNamespace(
        path_to_models=checkpoints,
        check_training_transforms=check_training_transforms,
    )
    Predictor._report_chain_drift(cast(Predictor, predictor))
    return capsys.readouterr().out


@pytest.mark.parametrize("example", ["Segmentation", "Synthesis", "Registration"])
def test_a_shipped_example_preprocesses_its_inputs_the_way_it_trained(
    example: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    checkpoints = _workspace(tmp_path, EXAMPLES / example / "Config.yml", EXAMPLES / example / "Prediction.yml")
    assert _report(monkeypatch, capsys, tmp_path, checkpoints) == ""


@pytest.mark.parametrize(
    ("trained", "applied", "expected"),
    [
        pytest.param(
            {"Standardize": {"mask": "None"}},
            {"Standardize": {"mask": "MASK"}},
            "'CT:CT' transforms[0] Standardize: mask: 'None' in training, 'MASK' here",
            id="a-changed-argument",
        ),
        pytest.param(
            {"Standardize": {"mask": "None"}},
            {"Standardize": {"mask": "None"}, "Mask": {"mask": "MASK"}},
            "'CT:CT' transforms[1] Mask: applied here, absent from the training chain",
            id="a-stage-only-the-prediction-applies",
        ),
        pytest.param(
            {"Clip": {"min_value": 0}, "Standardize": {"mask": "None"}},
            {"Clip": {"min_value": 0}},
            "'CT:CT' transforms[1] Standardize: in the training chain, not applied here",
            id="a-stage-the-prediction-dropped",
        ),
        pytest.param(
            {"Clip": {"min_value": 0}},
            {"Normalize": {"min_value": 0}},
            "'CT:CT' transforms[0] Normalize: the training chain has 'Clip' at this position",
            id="another-stage-at-the-same-position",
        ),
    ],
)
def test_a_differing_stage_is_named_with_its_group_index_and_arguments(
    trained: dict[str, dict[str, Any]],
    applied: dict[str, dict[str, Any]],
    expected: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    checkpoints = _workspace(tmp_path, _config("Trainer", trained), _config("Predictor", applied))
    report = _report(monkeypatch, capsys, tmp_path, checkpoints)
    assert expected in report
    assert "check_training_transforms" in report


@pytest.mark.parametrize(
    ("trained", "applied"),
    [
        pytest.param(
            {"Statistics": {}, "Standardize": {"mask": "None"}},
            {"Standardize": {"mask": "None"}},
            id="a-stage-that-records-a-fact-and-alters-nothing",
        ),
        pytest.param(
            {"Save": {"dataset": "./Cache:mha"}, "Clip": {"min_value": 0}},
            {"Clip": {"min_value": 0}},
            id="a-save-boundary",
        ),
        pytest.param(
            {"Clip": {"min_value": 0}},
            {"Clip": {"min_value": 0}, "Expand": {"nb": 2}, "Flip": {"prob": 1}},
            id="a-draw-past-the-expand-marker",
        ),
        pytest.param(
            {"Standardize": {"mask": "None", "inverse": False}},
            {"Standardize": {"mask": "None", "inverse": True}},
            id="the-inverse-argument-of-the-output-path",
        ),
        pytest.param(
            {"Clip": {"min_value": 0}},
            {"konfai.data.transform:Clip": {"min_value": 0}},
            id="the-module-qualified-spelling-of-one-stage",
        ),
    ],
)
def test_what_does_not_reach_the_model_input_is_not_a_difference(
    trained: dict[str, dict[str, Any]],
    applied: dict[str, dict[str, Any]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    checkpoints = _workspace(tmp_path, _config("Trainer", trained), _config("Predictor", applied))
    assert _report(monkeypatch, capsys, tmp_path, checkpoints) == ""


def test_a_group_the_model_does_not_read_is_not_compared(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A mask group feeds preprocessing, not the model: what it applies is nobody's contract."""
    applied = _config("Predictor", {"Standardize": {"mask": "MASK"}})
    applied["Predictor"]["Dataset"]["groups_src"]["CT"]["groups_dest"]["CT"]["is_input"] = False
    checkpoints = _workspace(tmp_path, _config("Trainer", {"Standardize": {"mask": "None"}}), applied)
    assert _report(monkeypatch, capsys, tmp_path, checkpoints) == ""


def test_folds_that_spell_the_same_chains_are_one_warning_naming_every_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    checkpoints = _workspace(
        tmp_path,
        _config("Trainer", {"Standardize": {"mask": "None"}}),
        _config("Predictor", {"Standardize": {"mask": "MASK"}}),
        runs=("FOLD_0", "FOLD_1", "FOLD_2"),
    )
    report = _report(monkeypatch, capsys, tmp_path, checkpoints)
    assert report.count("WARNING") == 1
    assert "FOLD_0, FOLD_1, FOLD_2" in report


def test_a_checkpoint_without_a_training_config_says_so_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An app bundle or a hand-copied .pt keeps no resolved config: the check says why, not nothing."""
    checkpoints = _workspace(
        tmp_path,
        _config("Trainer", {"Standardize": {"mask": "None"}}),
        _config("Predictor", {"Standardize": {"mask": "MASK"}}),
        runs=("FOLD_0", "FOLD_1"),
    )
    shutil.rmtree(tmp_path / "Statistics")
    report = _report(monkeypatch, capsys, tmp_path, checkpoints)
    assert report.count("did not run") == 1
    assert "2 checkpoint(s)" in report
    assert "WARNING" not in report


def test_check_training_transforms_false_silences_the_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    checkpoints = _workspace(
        tmp_path,
        _config("Trainer", {"Standardize": {"mask": "None"}}),
        _config("Predictor", {"Standardize": {"mask": "MASK"}}),
    )
    assert _report(monkeypatch, capsys, tmp_path, checkpoints, check_training_transforms=False) == ""


def test_reading_the_training_config_leaves_it_byte_identical(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Binding a config rewrites it; this check only reads, so a run's record stays as it was."""
    checkpoints = _workspace(tmp_path, EXAMPLES / "Synthesis" / "Config.yml", EXAMPLES / "Synthesis" / "Prediction.yml")
    statistics = tmp_path / "Statistics" / "RUN" / "Config_0_1.yml"
    before = statistics.read_bytes()
    _report(monkeypatch, capsys, tmp_path, checkpoints)
    assert statistics.read_bytes() == before
    assert before == (EXAMPLES / "Synthesis" / "Config.yml").read_bytes()
