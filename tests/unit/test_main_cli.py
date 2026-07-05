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

"""Tests for the ``konfai`` CLI (``konfai.main``): subcommand dispatch and the
CLI-facing parameter contract of the backend entry points."""

import inspect
import sys
from pathlib import Path

import pytest

import konfai.evaluator as evaluator_module
import konfai.main as main_module
import konfai.predictor as predictor_module
import konfai.trainer as trainer_module


def test_konfai_help_exits_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["konfai", "--help"])

    with pytest.raises(SystemExit) as exc_info:
        main_module.main()

    assert exc_info.value.code == 0


def test_konfai_train_dispatches_correctly(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_train(**kwargs) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(trainer_module, "train", fake_train)
    monkeypatch.setattr(sys, "argv", ["konfai", "TRAIN", "-c", "Config.yml"])

    main_module.main()

    assert captured["config"] == "Config.yml"


def test_main_prediction_dispatches_config_as_prediction_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def fake_predict(**kwargs) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(predictor_module, "predict", fake_predict)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "konfai",
            "PREDICTION",
            "-c",
            "Prediction.custom.yml",
            "--models",
            str(tmp_path / "checkpoint.pt"),
            "--cpu",
            "1",
        ],
    )

    main_module.main()

    assert captured["prediction_file"] == "Prediction.custom.yml"
    assert captured["cpu"] == 1
    assert captured["gpu"] == []
    assert captured["models"] == [str(tmp_path / "checkpoint.pt")]
    assert "config" not in captured


def test_konfai_eval_dispatches_correctly(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_evaluate(**kwargs) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(evaluator_module, "evaluate", fake_evaluate)
    monkeypatch.setattr(sys, "argv", ["konfai", "EVALUATION", "-c", "Evaluation.yml"])

    main_module.main()

    assert captured["evaluations_file"] == "Evaluation.yml"


def test_predict_evaluate_expose_tensorboard_param():
    """#7 CLI -tb/--tensorboard (dest 'tensorboard') must reach predict()/evaluate()."""
    for fn in (predictor_module.predict, evaluator_module.evaluate):
        params = inspect.signature(fn).parameters
        assert "tensorboard" in params, f"{fn.__name__} must accept 'tensorboard'"
        assert "tb" not in params, f"{fn.__name__} must not use the old 'tb' name"
