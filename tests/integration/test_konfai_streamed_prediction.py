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

"""Streamed prediction writes (automatic, no config knob): the slab-by-slab path must be voxel-identical
to the assembled-volume path (obtained via the ``KONFAI_STREAMED_WRITES=0`` kill-switch), must actually
take the streamed writer (its hand-written MetaImage header is observable), and must fall back safely
when the gate refuses (here: TTA).

The geometry variants exercise the write dispatcher end to end, one per region kind and then in
composition: a ``Canonical`` inverse (ORIENTATION — in-slab mirrors), a ``Padding`` inverse (CROP), a
``ResampleToResolution`` inverse on a uint8 chain (RESCALE, streamed through in nearest mode) and on a
float chain (demoted to the buffered tail, since interpolation is only byte-identical whole-volume),
a two-inverse pipe, and the full three-inverse stack (crop + rescale + reorient composed, streamed
end to end). Every variant is compared voxel for voxel against its own kill-switch reference."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from test_konfai_core_workflows import _prepare_experiment_dir, _subprocess_env
from test_konfai_ensemble_tta import TTA_AUGMENTATIONS_BLOCK, _replace_once

pytestmark = pytest.mark.integration

SimpleITK = pytest.importorskip("SimpleITK")

TRAIN_NAME = "STREAMED"

RUNNER_SOURCE = """
import os
from pathlib import Path

from konfai.predictor import predict
from konfai.trainer import train


def run_prediction(prediction_file: Path, predictions_dir: Path, disable_streaming: bool = False) -> None:
    # Streaming has no config knob -- it is automatic. The whole-volume reference is obtained through the
    # global ops kill-switch KONFAI_STREAMED_WRITES=0; the streamed run just leaves it unset.
    os.environ["KONFAI_STREAMED_WRITES"] = "0" if disable_streaming else "1"
    root = Path.cwd()
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
    # Same config both ways -- only the kill-switch differs -- so the outputs must match bit for bit.
    for variant in ["", "Canonical", "Padding", "ResampleLabel", "ResampleFloat", "GeometryPair", "GeometryStack"]:
        run_prediction(
            root / f"Prediction{variant}.yml", root / f"Predictions_{variant or 'base'}_reference",
            disable_streaming=True,
        )
        run_prediction(root / f"Prediction{variant}.yml", root / f"Predictions_{variant or 'base'}_streamed")
    # TTA makes the finalize chain non-local, so streaming refuses and the run completes whole-volume.
    run_prediction(root / "PredictionTTA.yml", root / "Predictions_streamed_tta")


if __name__ == "__main__":
    main()
"""

# One prediction config per write-dispatcher path: the transforms block goes on the INPUT group, so
# the finalize chain carries its inverse. (YAML indentation matches tests/assets/Workflows/Prediction.yml.)
_VARIANT_TRANSFORMS = {
    # ORIENTATION: an identity-direction case reorients onto LPS by mirroring x and y — the inverse
    # mirrors them back inside each slab while the slab axis maps identically.
    "Canonical": """            transforms:
              Canonical:
                inverse: true""",
    # CROP: the inverse drops the padded border and translates what remains.
    "Padding": """            transforms:
              Padding:
                padding: [0, 0, 0, 0, 2, 1]
                mode: constant
                inverse: true""",
    # RESCALE: the inverse resamples back to the stored grid.
    "ResampleLabel": """            transforms:
              ResampleToResolution:
                spacing: [0.5, 0.5, -1.0]
                inverse: true""",
    "ResampleFloat": """            transforms:
              ResampleToResolution:
                spacing: [0.5, 0.5, -1.0]
                inverse: true""",
    # Several region stages compose into one streamed pipe: crop, then flip, straight to the sink.
    "GeometryPair": """            transforms:
              Padding:
                padding: [0, 0, 0, 0, 2, 1]
                mode: constant
                inverse: true
              Flip:
                dims: '0'
                inverse: true""",
    # The full stack the composition exists for — reorient + resample + pad forward, so the finalize
    # chain carries CROP + RESCALE + ORIENTATION in sequence on a uint8 labelmap, streamed end to end.
    "GeometryStack": """            transforms:
              Canonical:
                inverse: true
              ResampleToResolution:
                spacing: [0.5, 0.5, -1.0]
                inverse: true
              Padding:
                padding: [0, 0, 0, 0, 2, 1]
                mode: constant
                inverse: true""",
}

# ResampleLabel/GeometryStack cast to uint8 before the reduction, so the tensor reaching the RESCALE
# stage resamples in nearest mode — the exactness the streamed resample requires. ResampleFloat keeps
# the float chain: the dispatcher must demote it to the buffered tail and stay byte-identical.
_UINT8_BEFORE_REDUCTION = """        before_reduction_transforms:
          TensorCast:
            dtype: uint8
            inverse: false"""

_UINT8_VARIANTS = ("ResampleLabel", "GeometryStack")

# Whether the variant's output is written by the streamed region writer (hand-written MetaImage
# header, no CenterOfRotation) or assembled in the buffer and written classically by SimpleITK.
_VARIANT_USES_STREAMED_WRITER = {
    "base": True,
    "Canonical": True,
    "Padding": True,
    "ResampleLabel": True,
    "ResampleFloat": False,
    "GeometryPair": True,
    "GeometryStack": True,
}


def _write_streamed_prediction_configs(experiment_dir: Path) -> None:
    base = (experiment_dir / "Prediction.yml").read_text(encoding="utf-8")
    tta = _replace_once(base, "    augmentations: None", TTA_AUGMENTATIONS_BLOCK)
    (experiment_dir / "PredictionTTA.yml").write_text(tta, encoding="utf-8")
    for variant, transforms_block in _VARIANT_TRANSFORMS.items():
        config = _replace_once(base, "            transforms: None", transforms_block)
        if variant in _UINT8_VARIANTS:
            config = _replace_once(config, "        before_reduction_transforms: None", _UINT8_BEFORE_REDUCTION)
        (experiment_dir / f"Prediction{variant}.yml").write_text(config, encoding="utf-8")


@pytest.fixture(scope="module")
def streamed_experiment(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Train once, then predict the assembled reference (kill-switch), the automatic streamed run, and a
    TTA run that must fall back to the whole-volume path."""
    experiment_dir = tmp_path_factory.mktemp("streamed") / "experiment"
    paths = _prepare_experiment_dir(experiment_dir, TRAIN_NAME)
    _write_streamed_prediction_configs(experiment_dir)

    runner_path = experiment_dir / "run_streamed_prediction.py"
    runner_path.write_text(RUNNER_SOURCE.replace("__TRAIN_NAME__", TRAIN_NAME), encoding="utf-8")
    subprocess.run(
        [sys.executable, str(runner_path)],
        cwd=experiment_dir,
        env=_subprocess_env(),
        check=True,
    )
    return {
        "dataset_dir": paths["dataset_dir"],
        "experiment_dir": experiment_dir,
        "streamed_tta": experiment_dir / "Predictions_streamed_tta",
    }


def _case_names(dataset_dir: Path) -> list[str]:
    names = sorted(path.name for path in dataset_dir.iterdir() if path.is_dir())
    assert names, "synthetic dataset is empty"
    return names


def _prediction_path(predictions_dir: Path, case: str) -> Path:
    path = predictions_dir / TRAIN_NAME / "Dataset" / case / "sCT.mha"
    assert path.exists(), f"missing prediction output: {path}"
    return path


@pytest.mark.parametrize("variant", ["base", *_VARIANT_TRANSFORMS])
def test_streamed_prediction_is_voxel_identical_to_reference(
    streamed_experiment: dict[str, Path], variant: str
) -> None:
    experiment_dir = streamed_experiment["experiment_dir"]
    for case in _case_names(streamed_experiment["dataset_dir"]):
        reference = SimpleITK.ReadImage(
            str(_prediction_path(experiment_dir / f"Predictions_{variant}_reference", case))
        )
        streamed = SimpleITK.ReadImage(str(_prediction_path(experiment_dir / f"Predictions_{variant}_streamed", case)))
        assert streamed.GetOrigin() == reference.GetOrigin(), (variant, case)
        assert streamed.GetSpacing() == reference.GetSpacing(), (variant, case)
        assert streamed.GetDirection() == reference.GetDirection(), (variant, case)
        reference_array = SimpleITK.GetArrayFromImage(reference)
        streamed_array = SimpleITK.GetArrayFromImage(streamed)
        assert streamed_array.dtype == reference_array.dtype, (variant, case)
        np.testing.assert_array_equal(streamed_array, reference_array, err_msg=f"{variant}/{case}")


@pytest.mark.parametrize("variant", ["base", *_VARIANT_TRANSFORMS])
def test_streamed_prediction_takes_the_expected_writer(streamed_experiment: dict[str, Path], variant: str) -> None:
    """SimpleITK always writes ``CenterOfRotation`` into a MetaImage header; the streamed region writer
    never does. Its absence proves the variant streamed to the sink; its presence proves the buffered
    (chain-split / demoted) variants assembled and wrote classically."""
    experiment_dir = streamed_experiment["experiment_dir"]
    case = _case_names(streamed_experiment["dataset_dir"])[0]
    streamed_header = _prediction_path(experiment_dir / f"Predictions_{variant}_streamed", case).read_bytes()[:2048]
    reference_header = _prediction_path(experiment_dir / f"Predictions_{variant}_reference", case).read_bytes()[:2048]
    assert (b"CenterOfRotation" not in streamed_header) == _VARIANT_USES_STREAMED_WRITER[variant], variant
    assert b"CenterOfRotation" in reference_header, variant


def test_streamed_request_with_tta_falls_back_to_whole_volume_path(streamed_experiment: dict[str, Path]) -> None:
    for case in _case_names(streamed_experiment["dataset_dir"]):
        path = _prediction_path(streamed_experiment["streamed_tta"], case)
        assert b"CenterOfRotation" in path.read_bytes()[:2048], "expected the whole-volume writer"
        array = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(str(path)))
        assert array.shape == (3, 16, 16), case
        assert np.isfinite(array).all(), case
