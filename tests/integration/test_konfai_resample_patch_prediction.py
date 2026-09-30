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

"""``Patch.mode: resample`` in prediction: a case over ``max_voxels``, or one that runs out of memory, runs whole
on a coarser grid, and its prediction lands back on the case's own grid, never cut into patches. Tile mode on the
same forced out-of-memory still cuts (``test_konfai_auto_patch_prediction``)."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from harness import prepare_experiment_dir, replace_once, run_workflow

pytestmark = pytest.mark.integration

SimpleITK = pytest.importorskip("SimpleITK")

TRAIN_NAME = "RESAMPLEPATCH"

RUNNER_SOURCE = '''
import os
from pathlib import Path

import torch

import konfai.predictor.loop as predictor_loop
import konfai.predictor.workflow as predictor_workflow
import konfai.utils.vram as vram_module
from konfai.predictor import build_predict, predict
from konfai.trainer import train

SHAPES = []


def record_shapes(force_oom: bool) -> None:
    """Record the grid each attempt runs on; with ``force_oom``, the first attempt runs out of memory."""
    original_run = predictor_loop._Predictor.run

    def run(self):
        managers = next(iter(self.dataset.data.values()))
        SHAPES.append([list(manager.shapes[0]) for manager in managers])
        if force_oom and len(SHAPES) == 1:
            raise torch.cuda.OutOfMemoryError("forced OOM")
        return original_run(self)

    predictor_loop._Predictor.run = run
    vram_module.transient_at_oom = lambda device: None
    vram_module.usable_after_oom = lambda device: 1.0


def run(config: str, out: str, checkpoint: Path) -> None:
    predictor = build_predict(models=[checkpoint], prediction_file=Path.cwd() / config, predictions_dir=Path.cwd() / out)
    with predictor as configured:
        configured.setup(1)
        configured(0)


def main() -> None:
    root = Path.cwd()
    train(overwrite=True, gpu=[], cpu=1, quiet=True, tensorboard=False, config=root / "Config.yml",
          checkpoints_dir=root / "Checkpoints", statistics_dir=root / "Statistics")
    checkpoint = sorted((root / "Checkpoints" / "__TRAIN_NAME__").glob("*.pt"))[-1]
    predict(models=[checkpoint], overwrite=True, gpu=[], cpu=1, quiet=True, tensorboard=False,
            prediction_file=root / "Prediction.yml", predictions_dir=root / "Predictions_reference")
    os.environ["KONFAI_OVERWRITE"] = "True"
    os.environ["KONFAI_VERBOSE"] = "False"
    original_run = predictor_loop._Predictor.run
    record_shapes(force_oom=False)
    run("PredictionBudget.yml", "Predictions_budget", checkpoint)
    budget = SHAPES[:]
    SHAPES.clear()
    predictor_loop._Predictor.run = original_run
    predictor_workflow.RESAMPLE_FLOOR_VOXELS = 64  # the toy cases hold 768 voxels
    record_shapes(force_oom=True)
    run("PredictionOom.yml", "Predictions_oom", checkpoint)
    (root / "shapes.txt").write_text(repr({"budget": budget, "oom": SHAPES}))


if __name__ == "__main__":
    main()
'''


@pytest.fixture(scope="module")
def resample_experiment(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    experiment_dir = tmp_path_factory.mktemp("resample_patch") / "experiment"
    paths = prepare_experiment_dir(experiment_dir, TRAIN_NAME)
    base = (experiment_dir / "Prediction.yml").read_text(encoding="utf-8")
    resample = replace_once(base, "patch_size: [1, 16, 16]", "patch_size: [1, 0, 0]\n      mode: resample")
    budget = replace_once(resample, "extend_slice: 0", "extend_slice: 0\n      max_voxels: 200")
    (experiment_dir / "PredictionBudget.yml").write_text(budget, encoding="utf-8")
    (experiment_dir / "PredictionOom.yml").write_text(resample, encoding="utf-8")
    runner_path = experiment_dir / "run_resample_prediction.py"
    runner_path.write_text(RUNNER_SOURCE.replace("__TRAIN_NAME__", TRAIN_NAME), encoding="utf-8")
    run_workflow([sys.executable, str(runner_path)], experiment_dir)
    return {"dir": experiment_dir, "dataset_dir": paths["dataset_dir"]}


def _read(experiment_dir: Path, predictions: str, case: str):
    path = experiment_dir / predictions / TRAIN_NAME / "Dataset" / case / "sCT.mha"
    assert path.exists(), f"missing prediction output: {path}"
    return SimpleITK.ReadImage(str(path))


def test_the_case_runs_coarse_and_whole_and_comes_back_on_its_own_grid(resample_experiment) -> None:
    shapes = eval((resample_experiment["dir"] / "shapes.txt").read_text())
    # Before the run, each case is coarsened to at most 200 voxels (3 x 16 x 16 = 768 on disk).
    assert all(np.prod(shape) <= 200 for shape in shapes["budget"][0])
    # Out of memory at full size, then once more at half the voxels at most.
    assert shapes["oom"][0] == [[3, 16, 16]] * len(shapes["oom"][0])
    assert all(np.prod(shape) <= 768 // 2 for shape in shapes["oom"][1])
    # and the plan the run finished on is the one it records, not the one it started with
    plan = json.loads((resample_experiment["dir"] / "Predictions_oom" / TRAIN_NAME / "Plan.json").read_text())
    assert plan["out_of_memory_restarts"] == 1 and plan["resample_voxels"] <= 768 // 2
    cases = sorted(path.name for path in resample_experiment["dataset_dir"].iterdir() if path.is_dir())
    for predictions in ("Predictions_budget", "Predictions_oom"):
        for case in cases:
            reference = _read(resample_experiment["dir"], "Predictions_reference", case)
            image = _read(resample_experiment["dir"], predictions, case)
            assert image.GetSize() == reference.GetSize(), case
            assert image.GetOrigin() == reference.GetOrigin(), case
            assert image.GetSpacing() == reference.GetSpacing(), case
            assert image.GetDirection() == reference.GetDirection(), case
            # A ramp, coarsened then interpolated back: close, not equal.
            np.testing.assert_allclose(
                SimpleITK.GetArrayFromImage(image), SimpleITK.GetArrayFromImage(reference), atol=0.25, err_msg=case
            )
