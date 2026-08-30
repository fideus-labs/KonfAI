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

"""One seeded CPU epoch and the prediction it feeds, pinned to the values they produced.

Every other test pins a part: this one pins what a user gets. A four-case cohort, a two-class
network of elementwise parameters (``tests/assets/Workflows/TinySeg.py``), one epoch of AdamW on a
cross-entropy, then the checkpoint predicting label maps through Argmax. What holds it still: no
validation split (drawn from the unseeded global RNG), no shuffle, no augmentation, ``manual_seed``,
and a network whose forward is elementwise, so a value's result does not depend on the patch it
arrives in.

CPU only. A GPU's kernels do not reproduce these bits, and a seed is not portable across devices.

Two numbers move here, and only deliberately: state the move and the reason in the commit, and take
the new values from what the failure prints.
"""

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch
from konfai import api

sitk = pytest.importorskip("SimpleITK")

ASSETS = Path(__file__).resolve().parents[1] / "assets" / "Workflows"

#: The epoch's mean training loss, held to 1e-6 relative: the last bits belong to the order the
#: cross-entropy's reduction sums in, which follows the host's thread count.
GOLDEN_LOSS = 0.307594910264

#: SHA-256 of the predicted label maps, case by case in name order. Exact, because a label map is
#: discrete: the softmax's last bits are 1e-7 from a decision boundary the cohort crosses by 1e-2.
GOLDEN_LABELS = "23ee5ebf95c41d681860199c85b1fe16694666fe1e8d85d1f674d801a095db1d"

CASES = 4
PATCH = {"patch_size": [1, 16, 16], "overlap": "None", "pad_value": 0, "extend_slice": 0}


def _write_cohort(dataset: Path) -> None:
    """A ramp and the label map it thresholds: procedural, so the cohort is part of the pin."""
    slices, rows, columns = np.meshgrid(
        np.linspace(-1.0, 1.0, 2, dtype=np.float32),
        np.linspace(-1.0, 1.0, 16, dtype=np.float32),
        np.linspace(-1.0, 1.0, 16, dtype=np.float32),
        indexing="ij",
    )
    for index in range(CASES):
        case = dataset / f"CASE_{index:03d}"
        case.mkdir(parents=True)
        ramp = np.tanh(1.5 * (rows + 0.5 * columns + 0.25 * slices) + 0.1 * (index - 1.5)).astype(np.float32)
        label = (ramp > 0.0).astype(np.uint8)
        for group, array, pixel in (("MR", ramp, sitk.sitkFloat32), ("LABEL", label, sitk.sitkUInt8)):
            image = sitk.Cast(sitk.GetImageFromArray(np.ascontiguousarray(array)), pixel)
            image.SetSpacing((1.0, 1.0, 1.0))
            sitk.WriteImage(image, str(case / f"{group}.mha"))


def _train_config(dataset: Path) -> dict:
    return {
        "Trainer": {
            "Model": {
                "classpath": "TinySeg:TinySegNet",
                "TinySegNet": {
                    "outputs_criterions": {
                        "Logits": {
                            "targets_criterions": {
                                "LABEL": {
                                    "criterions_loader": {
                                        "CrossEntropyLoss": {
                                            "is_loss": True,
                                            "schedulers": {"Constant": {"nb_step": 0, "value": 1}},
                                            "group": 0,
                                            "start": 0,
                                            "stop": "None",
                                            "accumulation": False,
                                            "reduction": "mean",
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "optimizer": {"name": "AdamW", "lr": 0.01, "weight_decay": 0.0},
                    "schedulers": {"ConstantLR": {"factor": 1.0, "total_iters": 1, "nb_step": 0}},
                },
            },
            "Dataset": {
                "groups_src": {
                    "MR": {"groups_dest": {"MR": {"transforms": "None", "patch_transforms": "None", "is_input": True}}},
                    "LABEL": {
                        "groups_dest": {
                            "LABEL": {
                                # A label map must reach the cross-entropy as an integer.
                                "transforms": {"TensorCast": {"dtype": "int64", "inverse": False}},
                                "patch_transforms": "None",
                                "is_input": False,
                            }
                        }
                    },
                },
                "augmentations": "None",
                "Patch": dict(PATCH),
                "subset": "None",
                "shuffle": False,
                "dataset_filenames": [f"{dataset}:a:mha"],
                "inline_augmentations": False,
                "batch_size": 4,
                "validation": "None",
            },
            "train_name": "GOLDEN",
            "manual_seed": 0,
            "epochs": 1,
            # The cohort is 8 patches of 4: one scored checkpoint at the epoch's end, and the
            # unscored one saved at close.
            "it_validation": 2,
            "autocast": False,
            "gradient_checkpoints": "None",
            "gpu_checkpoints": "None",
            "ema_decay": 0,
            "data_log": "None",
            "save_checkpoint_mode": "ALL",
            "EarlyStopping": "None",
        }
    }


def _predict_config(dataset: Path) -> dict:
    return {
        "Predictor": {
            "Model": {"classpath": "TinySeg:TinySegNet", "TinySegNet": {"outputs_criterions": "None"}},
            "Dataset": {
                "groups_src": {
                    "MR": {"groups_dest": {"MR": {"transforms": "None", "patch_transforms": "None", "is_input": True}}}
                },
                "augmentations": "None",
                "Patch": dict(PATCH),
                "subset": "None",
                "dataset_filenames": [f"{dataset}:a:mha"],
                "batch_size": 4,
            },
            "outputs_dataset": {
                "Head:Softmax": {
                    "OutputDataset": {
                        "name_class": "OutputDataset",
                        "before_reduction_transforms": "None",
                        "after_reduction_transforms": "None",
                        "final_transforms": {
                            "Argmax": {"dim": 0},
                            "TensorCast": {"dtype": "uint8", "inverse": False},
                        },
                        "dataset_filename": "Dataset:mha",
                        "group": "SEG",
                        "same_as_group": "MR:MR",
                        "patch_combine": "None",
                        "reduction": "Mean",
                        "Mean": {},
                    }
                }
            },
            "train_name": "GOLDEN",
            "manual_seed": 0,
            "gpu_checkpoints": "None",
            "autocast": False,
            "combine": "Mean",
            "data_log": "None",
        }
    }


def _scored_checkpoint(workspace: Path) -> Path:
    """The epoch's checkpoint. The one saved at close carries no score, which is ``inf``."""
    scored = [
        path
        for path in sorted(workspace.glob("*.pt"))
        if np.isfinite(torch.load(path, map_location="cpu", weights_only=False)["loss"])
    ]
    assert len(scored) == 1, [path.name for path in sorted(workspace.glob("*.pt"))]
    return scored[0]


def test_a_seeded_cpu_epoch_and_its_prediction_hold_their_values(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(ASSETS))  # where 'TinySeg:TinySegNet' resolves
    dataset = tmp_path / "Dataset"
    _write_cohort(dataset)

    workspace = api.train(
        _train_config(dataset),
        gpu=[],
        cpu=1,
        quiet=True,
        overwrite=True,
        checkpoints_dir=tmp_path / "Checkpoints",
        statistics_dir=tmp_path / "Statistics",
    )
    checkpoint = _scored_checkpoint(workspace)
    loss = float(torch.load(checkpoint, map_location="cpu", weights_only=False)["loss"])
    assert abs(loss - GOLDEN_LOSS) <= 1e-6 * GOLDEN_LOSS, f"the epoch's loss moved to {loss:.12f}"

    predictions = api.predict(
        models=[checkpoint],
        config=_predict_config(dataset),
        gpu=[],
        cpu=1,
        quiet=True,
        overwrite=True,
        predictions_dir=tmp_path / "Predictions",
    )
    label_maps = sorted((predictions / "Dataset").rglob("SEG.mha"))
    assert [path.parent.name for path in label_maps] == [f"CASE_{index:03d}" for index in range(CASES)]

    digest = hashlib.sha256()
    foreground = {}
    for path in label_maps:
        labels = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
        assert labels.dtype == np.uint8 and labels.shape == (2, 16, 16), path.parent.name
        foreground[path.parent.name] = int(labels.sum())
        digest.update(np.ascontiguousarray(labels).tobytes())
    assert digest.hexdigest() == GOLDEN_LABELS, f"the predicted labels moved; foreground voxels: {foreground}"
