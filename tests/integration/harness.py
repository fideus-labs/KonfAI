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

"""Shared harness for the KonfAI integration tests: synthetic experiment setup and subprocess plumbing."""

import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

SimpleITK = pytest.importorskip("SimpleITK")

ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets" / "Workflows"
REPO_ROOT = Path(__file__).resolve().parents[2]

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


def write_image(path: Path, array: np.ndarray, pixel_id: int) -> None:
    image = SimpleITK.GetImageFromArray(array)
    image.SetSpacing((1.0, 1.0, 1.0))
    image = SimpleITK.Cast(image, pixel_id)
    SimpleITK.WriteImage(image, str(path))


def _create_synthesis_dataset(dataset_dir: Path) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(4):
        case_dir = dataset_dir / f"CASE_{idx:03d}"
        case_dir.mkdir()
        zz, yy, xx = np.meshgrid(
            np.linspace(-0.2, 0.2, 3, dtype=np.float32),
            np.linspace(-1.0, 1.0, 16, dtype=np.float32),
            np.linspace(-1.0, 1.0, 16, dtype=np.float32),
            indexing="ij",
        )
        mr = np.clip(
            0.45 * yy + 0.35 * xx + zz + (idx - 1.5) * 0.05,
            -0.9,
            0.9,
        ).astype(np.float32)
        ct = np.tanh(1.25 * mr - 0.15).astype(np.float32)
        mask = np.ones_like(mr, dtype=np.uint8)
        mask[:, 0, :] = 0
        mask[:, -1, :] = 0
        write_image(case_dir / "MR.mha", mr, SimpleITK.sitkFloat32)
        write_image(case_dir / "CT.mha", ct, SimpleITK.sitkFloat32)
        write_image(case_dir / "MASK.mha", mask, SimpleITK.sitkUInt8)


def _render_asset_template(template_name: str, replacements: dict[str, str]) -> str:
    content = (ASSETS_DIR / template_name).read_text(encoding="utf-8")
    for placeholder, value in replacements.items():
        content = content.replace(placeholder, value)
    return content


def prepare_experiment_dir(experiment_dir: Path, train_name: str) -> dict[str, Path]:
    dataset_dir = experiment_dir / "Dataset"
    checkpoints_dir = experiment_dir / "Checkpoints"
    predictions_dir = experiment_dir / "Predictions"
    evaluations_dir = experiment_dir / "Evaluations"

    experiment_dir.mkdir(parents=True, exist_ok=True)
    _create_synthesis_dataset(dataset_dir)
    shutil.copy2(ASSETS_DIR / "TinySynth.py", experiment_dir / "TinySynth.py")
    (experiment_dir / "Config.yml").write_text(
        _render_asset_template(
            "Config.yml",
            {
                "__DATASET_DIR__": str(dataset_dir),
                "__TRAIN_NAME__": train_name,
            },
        ),
        encoding="utf-8",
    )
    (experiment_dir / "Prediction.yml").write_text(
        _render_asset_template(
            "Prediction.yml",
            {
                "__DATASET_DIR__": str(dataset_dir),
                "__TRAIN_NAME__": train_name,
            },
        ),
        encoding="utf-8",
    )
    (experiment_dir / "Evaluation.yml").write_text(
        _render_asset_template(
            "Evaluation.yml",
            {
                "__DATASET_DIR__": str(dataset_dir),
                "__PREDICTIONS_DATASET_DIR__": str(predictions_dir / train_name / "Dataset"),
                "__TRAIN_NAME__": train_name,
            },
        ),
        encoding="utf-8",
    )
    return {
        "dataset_dir": dataset_dir,
        "checkpoints_dir": checkpoints_dir,
        "predictions_dir": predictions_dir,
        "evaluations_dir": evaluations_dir,
    }


def subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(REPO_ROOT) if not pythonpath else f"{REPO_ROOT}{os.pathsep}{pythonpath}"
    return env


def konfai_cli_command() -> list[str]:
    cli = shutil.which("konfai")
    if cli is not None:
        return [cli]
    return [sys.executable, "-c", "from konfai.main import main; main()"]


def replace_once(content: str, old: str, new: str) -> str:
    assert content.count(old) == 1, f"expected exactly one occurrence of {old!r}"
    return content.replace(old, new)
