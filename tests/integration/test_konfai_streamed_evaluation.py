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

"""Streamed evaluation ends on the same numbers as whole-volume evaluation.

With a ``memory_budget`` too small for a case, evaluation cuts it into the largest disjoint patches
that fit and combines each metric's partial states; this proves end-to-end on disk that the produced
``Metric_TRAIN.json`` matches the whole-volume run within float tolerance -- the invariant the whole
feature stands on -- and that the budget run actually took the patched path.
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

SimpleITK = pytest.importorskip("SimpleITK")

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_NAME = "STREAMED_EVAL_01"


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(REPO_ROOT) if not pythonpath else f"{REPO_ROOT}{os.pathsep}{pythonpath}"
    return env


EVALUATION_TEMPLATE = """Evaluator:
  metrics:
    sCT:
      targets_criterions:
        CT:
          criterions_loader:
            MAE:
              reduction: mean
            MSE:
              reduction: mean
            PSNR:
              dynamic_range: 2.0
  Dataset:
    groups_src:
      CT:
        groups_dest:
          CT:
            transforms:
              TensorCast:
                dtype: float32
                inverse: true
            patch_transforms: None
            is_input: true
      sCT:
        groups_dest:
          sCT:
            transforms:
              TensorCast:
                dtype: float32
                inverse: true
            patch_transforms: None
            is_input: true
    subset: None
    validation: None
    memory_budget: __MEMORY_BUDGET__
    dataset_filenames:
      - __DATASET_DIR__:a:mha
      - __PREDICTIONS_DIR__:i:mha
    use_cache: false
  train_name: __TRAIN_NAME__
"""


def _write_image(path: Path, array: np.ndarray, pixel_id: int) -> None:
    image = SimpleITK.GetImageFromArray(array)
    image.SetSpacing((1.0, 1.0, 1.0))
    image = SimpleITK.Cast(image, pixel_id)
    SimpleITK.WriteImage(image, str(path))


def _create_paired_dataset(dataset_dir: Path, predictions_dir: Path) -> None:
    """Ground-truth CT and a spatially-varying 'prediction' per case, so the metrics are non-trivial."""
    rng = np.random.default_rng(7)
    for idx in range(3):
        (dataset_dir / f"CASE_{idx:03d}").mkdir(parents=True)
        (predictions_dir / f"CASE_{idx:03d}").mkdir(parents=True)
        ct = rng.normal(size=(5, 16, 16)).astype(np.float32)
        sct = (0.85 * ct + 0.1 * rng.normal(size=ct.shape) + 0.02 * idx).astype(np.float32)
        _write_image(dataset_dir / f"CASE_{idx:03d}" / "CT.mha", ct, SimpleITK.sitkFloat32)
        _write_image(predictions_dir / f"CASE_{idx:03d}" / "sCT.mha", sct, SimpleITK.sitkFloat32)


def _run_evaluation(experiment_dir: Path, memory_budget: str) -> str:
    dataset_dir = experiment_dir / "Dataset"
    predictions_dir = experiment_dir / "PredictionsData"
    _create_paired_dataset(dataset_dir, predictions_dir)
    (experiment_dir / "Evaluation.yml").write_text(
        EVALUATION_TEMPLATE.replace("__DATASET_DIR__", str(dataset_dir))
        .replace("__PREDICTIONS_DIR__", str(predictions_dir))
        .replace("__TRAIN_NAME__", TRAIN_NAME)
        .replace("__MEMORY_BUDGET__", memory_budget),
        encoding="utf-8",
    )
    runner = experiment_dir / "run_eval.py"
    runner.write_text(
        textwrap.dedent(
            """
            from pathlib import Path
            from konfai.evaluator import evaluate


            def main() -> None:
                evaluate(
                    overwrite=True,
                    gpu=[],
                    cpu=1,
                    quiet=True,
                    tensorboard=False,
                    evaluations_file=Path.cwd() / "Evaluation.yml",
                    evaluations_dir=Path.cwd() / "Evaluations",
                )


            if __name__ == "__main__":
                main()
            """
        ),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [sys.executable, str(runner)],
        cwd=experiment_dir,
        env=_subprocess_env(),
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, f"evaluation failed:\n{completed.stdout}\n{completed.stderr}"
    return completed.stdout + completed.stderr


def _read_metrics(experiment_dir: Path) -> dict:
    metrics_path = experiment_dir / "Evaluations" / TRAIN_NAME / "Metric_TRAIN.json"
    assert metrics_path.exists(), f"no metrics written under {metrics_path}"
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def test_streamed_evaluation_matches_whole_volume(tmp_path: Path) -> None:
    whole_dir = tmp_path / "whole"
    streamed_dir = tmp_path / "streamed"
    whole_dir.mkdir()
    streamed_dir.mkdir()

    _run_evaluation(whole_dir, "None")
    # A case here is 2 groups x 5x16x16 float32 ~= 10 KiB resident; 8000 bytes cannot hold it, so the
    # run must patch. The log line is the proof the streamed path actually executed.
    streamed_log = _run_evaluation(streamed_dir, "8000b")
    assert "disjoint patches" in streamed_log, f"budget run did not take the patched path:\n{streamed_log}"

    whole = _read_metrics(whole_dir)
    streamed = _read_metrics(streamed_dir)

    whole_cases = whole["case"] if "case" in whole else whole
    streamed_cases = streamed["case"] if "case" in streamed else streamed
    assert set(whole_cases) == set(streamed_cases)
    compared = 0
    for case, values in whole_cases.items():
        if not isinstance(values, dict):
            continue
        assert set(values) == set(streamed_cases[case])
        for key, value in values.items():
            got = streamed_cases[case][key]
            if value is None or (isinstance(value, float) and np.isnan(value)):
                assert got is None or np.isnan(got)
            else:
                assert got == pytest.approx(value, rel=1e-4), f"{case}:{key}: whole={value} streamed={got}"
            compared += 1
    assert compared >= 9  # 3 cases x 3 metrics: the comparison actually exercised the report
