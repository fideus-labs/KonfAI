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

"""VRAM auto-patching in prediction (``patch_size`` with free ``0`` axes): an OOM mid-run must
shrink the free axes, re-plan the grid, restart, and produce a prediction byte-identical to the
whole-volume run. The model is pointwise (1x1 conv) and the overlap 0, so patched == whole holds
exactly and any grid/mapping/accumulation mistake shows up as a voxel difference."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from test_konfai_core_workflows import _prepare_experiment_dir, _subprocess_env
from test_konfai_ensemble_tta import _replace_once

pytestmark = pytest.mark.integration

SimpleITK = pytest.importorskip("SimpleITK")

TRAIN_NAME = "AUTOPATCH"

RUNNER_SOURCE = '''
import os
from pathlib import Path

import torch

import konfai.predictor as predictor_module
from konfai.predictor import build_predict, predict
from konfai.trainer import train

ATTEMPTS = []


def install_auto_patch_probes() -> None:
    """Force the first attempt to OOM and stub the CUDA readings (this is a CPU-only run)."""
    original_run = predictor_module._Predictor.run

    def run_with_forced_oom(self):
        ATTEMPTS.append(list(self.dataset.get_patch_config()[0]))
        if len(ATTEMPTS) == 1:
            raise torch.cuda.OutOfMemoryError("forced OOM: pretend the full-slice forward does not fit")
        return original_run(self)

    predictor_module._Predictor.run = run_with_forced_oom
    predictor_module.Predictor._transient_at_oom = lambda self, device: None
    predictor_module.Predictor._usable_vram_after_oom = lambda self, device: 1.0


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
    # single rank IN-PROCESS so the forced OOM and the stubbed VRAM readings stay visible.
    os.environ["KONFAI_OVERWRITE"] = "True"
    os.environ["KONFAI_VERBOSE"] = "False"
    install_auto_patch_probes()
    predictor = build_predict(
        models=[checkpoints[-1]],
        prediction_file=root / "PredictionAuto.yml",
        predictions_dir=root / "Predictions_auto",
    )
    with predictor as configured:
        configured.setup(1)
        configured(0)
    # Attempt 1: the free axes at full extent; attempt 2: one fixed 0.8 shrink of the free Y/X axes
    # (the pinned Z=1 never moves). Anything else means the restart loop did not do its job.
    if ATTEMPTS != [[1, 0, 0], [1, 12, 12]]:
        raise RuntimeError(f"unexpected restart sequence: {ATTEMPTS}")


if __name__ == "__main__":
    main()
'''


@pytest.fixture(scope="module")
def auto_patch_experiment(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Train once, predict whole-slice (reference), then with a forced-OOM auto-patch restart."""
    experiment_dir = tmp_path_factory.mktemp("auto_patch") / "experiment"
    paths = _prepare_experiment_dir(experiment_dir, TRAIN_NAME)
    base = (experiment_dir / "Prediction.yml").read_text(encoding="utf-8")
    auto = _replace_once(base, "patch_size: [1, 16, 16]", "patch_size: [1, 0, 0]")
    auto = _replace_once(auto, "overlap: None", "overlap: 0")
    (experiment_dir / "PredictionAuto.yml").write_text(auto, encoding="utf-8")

    runner_path = experiment_dir / "run_auto_patch_prediction.py"
    runner_path.write_text(RUNNER_SOURCE.replace("__TRAIN_NAME__", TRAIN_NAME), encoding="utf-8")
    subprocess.run(
        [sys.executable, str(runner_path)],
        cwd=experiment_dir,
        env=_subprocess_env(),
        check=True,
    )
    return {
        "dataset_dir": paths["dataset_dir"],
        "reference": experiment_dir / "Predictions_reference",
        "auto": experiment_dir / "Predictions_auto",
    }


def _prediction_path(predictions_dir: Path, case: str) -> Path:
    path = predictions_dir / TRAIN_NAME / "Dataset" / case / "sCT.mha"
    assert path.exists(), f"missing prediction output: {path}"
    return path


def test_auto_patch_restart_is_voxel_identical_to_whole_volume(auto_patch_experiment: dict[str, Path]) -> None:
    cases = sorted(path.name for path in auto_patch_experiment["dataset_dir"].iterdir() if path.is_dir())
    assert cases, "synthetic dataset is empty"
    for case in cases:
        reference = SimpleITK.ReadImage(str(_prediction_path(auto_patch_experiment["reference"], case)))
        auto = SimpleITK.ReadImage(str(_prediction_path(auto_patch_experiment["auto"], case)))
        assert auto.GetOrigin() == reference.GetOrigin(), case
        assert auto.GetSpacing() == reference.GetSpacing(), case
        assert auto.GetDirection() == reference.GetDirection(), case
        reference_array = SimpleITK.GetArrayFromImage(reference)
        auto_array = SimpleITK.GetArrayFromImage(auto)
        assert auto_array.dtype == reference_array.dtype, case
        np.testing.assert_array_equal(auto_array, reference_array, err_msg=case)
