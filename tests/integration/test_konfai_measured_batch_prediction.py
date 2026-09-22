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

"""A prediction that measures its batch (``batch_size: 0``): a forward of one patch, then of two, then
the batch those two measurements say the device holds, merged from the loader's single patches. Its
prediction must be voxel-identical to the config-batched run. The model is pointwise (1x1 conv), so the
batch cannot change a voxel and any merge or index mistake shows up as a voxel difference."""

import sys
from pathlib import Path

import numpy as np
import pytest
from harness import prepare_experiment_dir, replace_once, run_workflow

pytestmark = pytest.mark.integration

SimpleITK = pytest.importorskip("SimpleITK")

TRAIN_NAME = "MEASUREDBATCH"

RUNNER_SOURCE = '''
import os
from pathlib import Path

import torch

import konfai.predictor.loop as predictor_loop
import konfai.utils.vram as vram_module
from konfai.predictor import build_predict, predict
from konfai.trainer import train

BATCHES = []


def install_measured_batch_probes() -> None:
    """Measure on a CPU-only run: the device readings are stubbed, and half of 280 usable bytes holds four."""
    original_init = predictor_loop._Predictor.__init__
    original_step = predictor_loop._Predictor._step

    def init_measuring(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.measure_batch_on = 0

    def step_recording(self, batch_sample):
        BATCHES.append(next(iter(batch_sample.values())).tensor.shape[0])
        return original_step(self, batch_sample)

    predictor_loop._Predictor.__init__ = init_measuring
    predictor_loop._Predictor._step = step_recording
    torch.cuda.memory_allocated = lambda device=None: 0
    torch.cuda.reset_peak_memory_stats = lambda device=None: None
    # A forward claims 100 bytes whatever its batch and 10 per patch: 110 for one, 120 for two.
    torch.cuda.max_memory_allocated = lambda device=None: 100 + 10 * BATCHES[-1]
    vram_module.usable_after_oom = lambda device: 280.0


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
    predict(
        models=[checkpoints[-1]],
        overwrite=True,
        gpu=[],
        cpu=1,
        quiet=True,
        tensorboard=False,
        prediction_file=root / "Prediction.yml",
        predictions_dir=root / "Predictions_reference",
    )
    # The workflow normally runs in a spawned child, where a monkeypatch would not survive; run the
    # single rank IN-PROCESS so the stubbed device readings stay visible.
    os.environ["KONFAI_OVERWRITE"] = "True"
    os.environ["KONFAI_VERBOSE"] = "False"
    install_measured_batch_probes()
    predictor = build_predict(
        models=[checkpoints[-1]],
        prediction_file=root / "PredictionMeasured.yml",
        predictions_dir=root / "Predictions_measured",
    )
    with predictor as configured:
        configured.setup(1)
        configured(0)
    # Four cases of three slices: one patch, two, then four at a time, and the one left at the end.
    if BATCHES != [1, 2, 4, 4, 1]:
        raise RuntimeError(f"unexpected batches: {BATCHES}")


if __name__ == "__main__":
    main()
'''


@pytest.fixture(scope="module")
def measured_batch_experiment(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Train once, predict with the config's batch (reference), then with the batch measured."""
    experiment_dir = tmp_path_factory.mktemp("measured_batch") / "experiment"
    paths = prepare_experiment_dir(experiment_dir, TRAIN_NAME)
    base = (experiment_dir / "Prediction.yml").read_text(encoding="utf-8")
    measured = replace_once(base, "batch_size: 16", "batch_size: 0")
    (experiment_dir / "PredictionMeasured.yml").write_text(measured, encoding="utf-8")

    runner_path = experiment_dir / "run_measured_batch_prediction.py"
    runner_path.write_text(RUNNER_SOURCE.replace("__TRAIN_NAME__", TRAIN_NAME), encoding="utf-8")
    run_workflow([sys.executable, str(runner_path)], experiment_dir)
    return {
        "dataset_dir": paths["dataset_dir"],
        "reference": experiment_dir / "Predictions_reference",
        "measured": experiment_dir / "Predictions_measured",
    }


def _prediction_path(predictions_dir: Path, case: str) -> Path:
    path = predictions_dir / TRAIN_NAME / "Dataset" / case / "sCT.mha"
    assert path.exists(), f"missing prediction output: {path}"
    return path


def test_a_measured_batch_is_voxel_identical_to_the_configured_one(
    measured_batch_experiment: dict[str, Path],
) -> None:
    cases = sorted(path.name for path in measured_batch_experiment["dataset_dir"].iterdir() if path.is_dir())
    assert cases, "synthetic dataset is empty"
    for case in cases:
        reference = SimpleITK.ReadImage(str(_prediction_path(measured_batch_experiment["reference"], case)))
        measured = SimpleITK.ReadImage(str(_prediction_path(measured_batch_experiment["measured"], case)))
        assert measured.GetOrigin() == reference.GetOrigin(), case
        assert measured.GetSpacing() == reference.GetSpacing(), case
        assert measured.GetDirection() == reference.GetDirection(), case
        reference_array = SimpleITK.GetArrayFromImage(reference)
        measured_array = SimpleITK.GetArrayFromImage(measured)
        assert measured_array.dtype == reference_array.dtype, case
        np.testing.assert_array_equal(measured_array, reference_array, err_msg=case)
