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
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
from harness import konfai_cli_command, prepare_experiment_dir, run_workflow, write_image
from konfai.evaluator import build_evaluate
from konfai.predictor import build_predict
from konfai.trainer import build_train

pytestmark = pytest.mark.integration

SimpleITK = pytest.importorskip("SimpleITK")


def _create_prediction_dataset_stub(predictions_dataset_dir: Path) -> None:
    predictions_dataset_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(4):
        case_dir = predictions_dataset_dir / f"CASE_{idx:03d}"
        case_dir.mkdir(parents=True, exist_ok=True)
        sct = np.zeros((3, 16, 16), dtype=np.float32)
        write_image(case_dir / "sCT.mha", sct, SimpleITK.sitkFloat32)


@contextmanager
def _working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _assert_experiment_outputs(
    dataset_dir: Path,
    checkpoints_dir: Path,
    predictions_dir: Path,
    evaluations_dir: Path,
    train_name: str,
) -> None:
    expected_cases = sorted(path.name for path in dataset_dir.iterdir() if path.is_dir())
    checkpoints = sorted((checkpoints_dir / train_name).glob("*.pt"))
    assert checkpoints
    predicted = sorted((predictions_dir / train_name / "Dataset").rglob("sCT.mha"))
    assert len(predicted) == len(expected_cases)
    assert sorted(path.parent.name for path in predicted) == expected_cases
    for path in predicted:
        image = SimpleITK.ReadImage(str(path))
        array = SimpleITK.GetArrayFromImage(image)
        assert array.shape == (3, 16, 16)
        assert np.isfinite(array).all()
    metrics_path = evaluations_dir / train_name / "Metric_TRAIN.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert "case" in metrics
    assert any(key.endswith("MAE") for key in metrics["case"])
    for metric_name, case_values in metrics["case"].items():
        assert sorted(case_values) == expected_cases
        assert all(isinstance(value, (int, float)) for value in case_values.values()), metric_name
        assert all(np.isfinite(value) for value in case_values.values()), metric_name


def test_konfai_cli_user_path(tmp_path: Path) -> None:
    experiment_dir = tmp_path / "experiment_cli"
    train_name = "CLI"
    paths = prepare_experiment_dir(experiment_dir, train_name)
    cli = konfai_cli_command()

    run_workflow(
        [
            *cli,
            "TRAIN",
            "-y",
            "--cpu",
            "1",
            "-q",
            "-c",
            "Config.yml",
            "--checkpoints-dir",
            "Checkpoints",
            "--statistics-dir",
            "Statistics",
        ],
        experiment_dir,
    )
    checkpoints = sorted((paths["checkpoints_dir"] / train_name).glob("*.pt"))
    assert checkpoints

    run_workflow(
        [
            *cli,
            "PREDICTION",
            "-y",
            "--cpu",
            "1",
            "-q",
            "-c",
            "Prediction.yml",
            "--models",
            *[str(path) for path in checkpoints],
            "--predictions-dir",
            "Predictions",
        ],
        experiment_dir,
    )
    run_workflow(
        [
            *cli,
            "EVALUATION",
            "-y",
            "--cpu",
            "1",
            "-q",
            "-c",
            "Evaluation.yml",
            "--evaluations-dir",
            "Evaluations",
        ],
        experiment_dir,
    )
    _assert_experiment_outputs(
        paths["dataset_dir"],
        paths["checkpoints_dir"],
        paths["predictions_dir"],
        paths["evaluations_dir"],
        train_name,
    )


def test_konfai_build_steps_construct_workflows_without_execution(
    tmp_path: Path,
) -> None:
    experiment_dir = tmp_path / "experiment_build"
    train_name = "BUILD"
    paths = prepare_experiment_dir(experiment_dir, train_name)
    _create_prediction_dataset_stub(paths["predictions_dir"] / train_name / "Dataset")

    sys.path.insert(0, str(experiment_dir))
    try:
        with _working_directory(experiment_dir):
            trainer = build_train(
                config=experiment_dir / "Config.yml",
                checkpoints_dir=paths["checkpoints_dir"],
                statistics_dir=experiment_dir / "Statistics",
            )
            predictor = build_predict(
                models=[experiment_dir / "dummy.pt"],
                prediction_file=experiment_dir / "Prediction.yml",
                predictions_dir=paths["predictions_dir"],
            )
            evaluator = build_evaluate(
                evaluations_file=experiment_dir / "Evaluation.yml",
                evaluations_dir=paths["evaluations_dir"],
            )
    finally:
        sys.path.remove(str(experiment_dir))

    assert trainer.name == train_name
    assert predictor.name == train_name
    assert evaluator.name == train_name
