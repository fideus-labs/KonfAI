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

"""A real PREDICTION reads the config its checkpoint's run left in ``Statistics/`` and warns when the
chain it applies to a model input is not that one. The unit suite pins the comparison; this pins the
wiring: the check runs on the checkpoint TRAIN actually wrote, and its lines reach the run's log."""

import subprocess
import sys
from pathlib import Path

import pytest
from harness import prepare_experiment_dir, replace_once, subprocess_env

pytestmark = pytest.mark.integration

pytest.importorskip("SimpleITK")

TRAIN_NAME = "CHAIN"

RUNNER_SOURCE = """
from pathlib import Path

from konfai.predictor import predict
from konfai.trainer import train


def main() -> None:
    root = Path.cwd()
    train(
        overwrite=True,
        gpu=[],
        cpu=1,
        quiet=True,
        tensorboard=False,
        config=root / "Config.yml",
        checkpoints_dir=root / "Checkpoints",
        statistics_dir=root / "Statistics",
    )
    checkpoints = sorted((root / "Checkpoints" / "__TRAIN_NAME__").glob("*.pt"))
    if not checkpoints:
        raise RuntimeError("no checkpoints produced")
    for variant in ["Matching", "Mismatched"]:
        predict(
            models=[checkpoints[-1]],
            overwrite=True,
            gpu=[],
            cpu=1,
            quiet=False,
            tensorboard=False,
            prediction_file=root / f"Prediction{variant}.yml",
            predictions_dir=root / f"Predictions_{variant}",
        )


if __name__ == "__main__":
    main()
""".replace("__TRAIN_NAME__", TRAIN_NAME)

# The training config leaves the MR group untransformed, so standardizing it here is exactly the
# drift the check exists for: same checkpoint, an input it has never seen.
_DRIFT = """            transforms:
              Standardize:
                lazy: false
                mean: None
                std: None
                mask: None
                inverse: false"""


def test_prediction_warns_in_its_log_when_a_model_input_drifts_from_training(tmp_path: Path) -> None:
    experiment_dir = tmp_path / "experiment_chain"
    prepare_experiment_dir(experiment_dir, TRAIN_NAME)
    prediction = (experiment_dir / "Prediction.yml").read_text(encoding="utf-8")
    (experiment_dir / "PredictionMatching.yml").write_text(prediction, encoding="utf-8")
    (experiment_dir / "PredictionMismatched.yml").write_text(
        replace_once(prediction, "            transforms: None", _DRIFT), encoding="utf-8"
    )
    (experiment_dir / "run_chain_check.py").write_text(RUNNER_SOURCE, encoding="utf-8")

    subprocess.run(
        [sys.executable, "run_chain_check.py"],
        cwd=experiment_dir,
        env=subprocess_env(),
        check=True,
    )

    matching = (experiment_dir / "Predictions_Matching" / TRAIN_NAME / "log_0.txt").read_text(encoding="utf-8")
    mismatched = (experiment_dir / "Predictions_Mismatched" / TRAIN_NAME / "log_0.txt").read_text(encoding="utf-8")
    assert "preprocesses a model input differently" not in matching
    assert "did not run" not in matching
    assert f"preprocesses a model input differently from {TRAIN_NAME}" in mismatched
    assert "'MR:MR' transforms[0] Standardize: applied here, absent from the training chain" in mismatched
    assert "check_training_transforms" in mismatched
