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

"""End-to-end multi-model ensemble + test-time-augmentation (TTA) prediction workflow.

The tiny synthesis model is trained once; a second ensemble member is derived from the
checkpoint by shifting every weight, so the two members differ measurably. PREDICTION then
runs with both checkpoints combined by ``Mean`` (``ModelComposite``) and a deterministic
``Flip`` TTA declared under ``Dataset.augmentations`` in the Prediction config.

``TinySynthNet`` is strictly pointwise (1x1 conv + tanh), so a correctly inverted geometric
TTA is an exact identity on the final volume. This makes the expected ensemble output
computable from two single-model baseline predictions and turns the assertions into sharp
oracles: the ensemble mean must match ``(A + B) / 2`` and a ``Concat`` TTA reduction summed
with the ``Sum`` transform must yield ``A + B`` (one term per TTA branch).
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from test_konfai_core_workflows import _prepare_experiment_dir, _subprocess_env

pytestmark = pytest.mark.integration

SimpleITK = pytest.importorskip("SimpleITK")

TRAIN_NAME = "ENSTTA"

# Deterministic flip TTA: the pipeline probability is 1 and torch.rand() < 1 always holds,
# so the single augmented replica is always flipped along Y and X ([C, Z, Y, X] dims 2, 3).
TTA_AUGMENTATIONS_BLOCK = """\
    augmentations:
      DataAugmentation_0:
        nb: 1
        data_augmentations:
          Flip:
            f_prob:
            - 0
            - 1
            - 1
            prob: 1"""

TTA_SUM_AFTER_REDUCTION_BLOCK = """\
        after_reduction_transforms:
          Sum:
            dim: 0"""

RUNNER_SOURCE = '''
from pathlib import Path

import torch

from konfai.predictor import predict
from konfai.trainer import train


def perturb_checkpoint(source: Path, destination: Path, delta: float = 0.25) -> None:
    """Copy a checkpoint, shifting every floating-point weight by ``delta``."""
    state = torch.load(source, map_location="cpu", weights_only=False)

    def shift(mapping) -> int:
        changed = 0
        for key, value in mapping.items():
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                mapping[key] = value + delta
                changed += 1
            elif isinstance(value, dict):
                changed += shift(value)
        return changed

    if shift(state["Model"]) == 0:
        raise RuntimeError("no floating-point weight was perturbed")
    torch.save(state, destination)


def run_prediction(models, prediction_file: Path, predictions_dir: Path) -> None:
    predict(
        models=models,
        overwrite=True,
        gpu=[],
        cpu=1,
        quiet=True,
        tensorboard=False,
        prediction_file=prediction_file,
        predictions_dir=predictions_dir,
    )


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
    member_a = checkpoints[-1]
    member_b = root / "member_b.pt"
    perturb_checkpoint(member_a, member_b)
    run_prediction([member_a], root / "Prediction.yml", root / "Predictions_single_a")
    run_prediction([member_b], root / "Prediction.yml", root / "Predictions_single_b")
    run_prediction([member_a, member_b], root / "PredictionTTA.yml", root / "Predictions_ensemble")
    run_prediction([member_a, member_b], root / "PredictionTTASum.yml", root / "Predictions_ensemble_sum")


if __name__ == "__main__":
    main()
'''


def _replace_once(content: str, old: str, new: str) -> str:
    assert content.count(old) == 1, f"expected exactly one occurrence of {old!r} in the Prediction template"
    return content.replace(old, new)


def _write_tta_prediction_configs(experiment_dir: Path) -> None:
    base = (experiment_dir / "Prediction.yml").read_text(encoding="utf-8")
    tta = _replace_once(base, "    augmentations: None", TTA_AUGMENTATIONS_BLOCK)
    (experiment_dir / "PredictionTTA.yml").write_text(tta, encoding="utf-8")

    # Same ensemble + TTA, but the TTA branches are concatenated and summed instead of
    # averaged: the output doubles only if both TTA branches were actually produced.
    tta_sum = _replace_once(tta, "        after_reduction_transforms: None", TTA_SUM_AFTER_REDUCTION_BLOCK)
    tta_sum = _replace_once(tta_sum, "        reduction: Mean", "        reduction: Concat")
    tta_sum = _replace_once(tta_sum, "        Mean: {}", "        Concat: {}")
    (experiment_dir / "PredictionTTASum.yml").write_text(tta_sum, encoding="utf-8")


@pytest.fixture(scope="module")
def ensemble_experiment(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Train once, then run the single-model, ensemble+TTA, and ensemble+TTA-sum predictions."""
    experiment_dir = tmp_path_factory.mktemp("ensemble_tta") / "experiment"
    paths = _prepare_experiment_dir(experiment_dir, TRAIN_NAME)
    _write_tta_prediction_configs(experiment_dir)

    runner_path = experiment_dir / "run_ensemble_tta.py"
    runner_path.write_text(RUNNER_SOURCE.replace("__TRAIN_NAME__", TRAIN_NAME), encoding="utf-8")
    subprocess.run(
        [sys.executable, str(runner_path)],
        cwd=experiment_dir,
        env=_subprocess_env(),
        check=True,
    )
    return {
        "experiment_dir": experiment_dir,
        "dataset_dir": paths["dataset_dir"],
        "single_a": experiment_dir / "Predictions_single_a",
        "single_b": experiment_dir / "Predictions_single_b",
        "ensemble": experiment_dir / "Predictions_ensemble",
        "ensemble_sum": experiment_dir / "Predictions_ensemble_sum",
    }


def _case_names(dataset_dir: Path) -> list[str]:
    names = sorted(path.name for path in dataset_dir.iterdir() if path.is_dir())
    assert names, "synthetic dataset is empty"
    return names


def _read_prediction(predictions_dir: Path, case: str) -> np.ndarray:
    path = predictions_dir / TRAIN_NAME / "Dataset" / case / "sCT.mha"
    assert path.exists(), f"missing prediction output: {path}"
    return SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(str(path))).astype(np.float64)


def test_ensemble_tta_prediction_completes_for_every_case(ensemble_experiment: dict[str, Path]) -> None:
    """The M=2/T=2 prediction writes one finite, correctly shaped volume per case.

    With TTA declared, ``OutputDataset.is_done`` only fires once accumulators exist for both
    augmentation indices, so a TTA pipeline that crashes or never emits the augmented branch
    leaves no output file at all.
    """
    for case in _case_names(ensemble_experiment["dataset_dir"]):
        ensemble = _read_prediction(ensemble_experiment["ensemble"], case)
        assert ensemble.shape == (3, 16, 16), case
        assert np.isfinite(ensemble).all(), case

    executed_config = (ensemble_experiment["ensemble"] / TRAIN_NAME / "Prediction.yml").read_text(encoding="utf-8")
    assert "Flip:" in executed_config and "f_prob:" in executed_config
    assert "augmentations: None" not in executed_config


def test_ensemble_mean_combines_both_members_with_identity_tta(ensemble_experiment: dict[str, Path]) -> None:
    """The ensemble output is the mean of both members, with the flip TTA exactly inverted."""
    for case in _case_names(ensemble_experiment["dataset_dir"]):
        single_a = _read_prediction(ensemble_experiment["single_a"], case)
        single_b = _read_prediction(ensemble_experiment["single_b"], case)
        ensemble = _read_prediction(ensemble_experiment["ensemble"], case)

        # Preconditions keeping the oracle below discriminating: the perturbed member must
        # differ from the original, and the prediction must be spatially asymmetric so a
        # missing/incorrect TTA inverse flip cannot cancel out.
        assert np.abs(single_a - single_b).max() > 0.05, case
        assert np.abs(ensemble - ensemble[:, ::-1, ::-1]).max() > 0.02, case

        # The second checkpoint measurably contributed to the combined output.
        assert np.abs(ensemble - single_a).max() > 0.02, case

        # Pointwise model => flip TTA is an exact identity, so Mean-combine + Mean-reduce
        # must reproduce the average of the two single-model baselines (fp16 ensemble math
        # keeps the residual well below the tolerance).
        np.testing.assert_allclose(ensemble, (single_a + single_b) / 2.0, atol=5e-3, err_msg=case)


def test_tta_branches_are_materialized_by_concat_sum_reduction(ensemble_experiment: dict[str, Path]) -> None:
    """Summing the concatenated TTA branches doubles the output: T=2 branches really ran.

    Each TTA branch equals the model-combined mean ``(A + B) / 2``; the ``Concat`` reduction
    followed by the ``Sum`` transform therefore yields ``A + B``. If the augmented branch
    were silently dropped (T=1), the output would be ``(A + B) / 2`` instead — off by a
    factor of two, far outside the tolerance.
    """
    for case in _case_names(ensemble_experiment["dataset_dir"]):
        single_a = _read_prediction(ensemble_experiment["single_a"], case)
        single_b = _read_prediction(ensemble_experiment["single_b"], case)
        summed = _read_prediction(ensemble_experiment["ensemble_sum"], case)

        assert summed.shape == (3, 16, 16), case
        assert np.isfinite(summed).all(), case
        np.testing.assert_allclose(summed, single_a + single_b, atol=1e-2, err_msg=case)
        assert np.abs(summed - (single_a + single_b) / 2.0).max() > 0.05, case
