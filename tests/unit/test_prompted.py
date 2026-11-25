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

"""A tool that segments from prompts: the points a case holds, and what reaches the tool's engine."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

sitk = pytest.importorskip("SimpleITK")

from konfai import api  # noqa: E402
from konfai.network.prompted import BOX, NEGATIVE, POSITIVE, PromptedNet, PromptedTool, Prompts, kept  # noqa: E402
from konfai.utils.dataset import Attribute, Dataset, read_landmarks, write_landmarks  # noqa: E402
from konfai.utils.errors import KonfAIError  # noqa: E402


def _geometry() -> Attribute:
    attribute = Attribute()
    attribute["Origin"] = np.array([3.0, -1.0, 7.0])
    attribute["Spacing"] = np.array([0.5, 0.5, 2.0])
    attribute["Direction"] = np.eye(3).flatten()
    return attribute


def test_prompts_give_the_clicks_in_their_order_and_the_box_its_corners_span() -> None:
    voxels = np.array([[5, 10, 10], [7, 16, 6], [5, 12, 11], [4, 6, 16], [5, 30, 12]], float)
    prompts = Prompts(voxels, [POSITIVE, BOX, NEGATIVE, BOX, POSITIVE])
    assert [(tuple(v), positive) for v, positive in prompts.clicks] == [
        ((5.0, 10.0, 10.0), True),
        ((5.0, 12.0, 11.0), False),
        ((5.0, 30.0, 12.0), True),
    ]
    assert prompts.box is not None and tuple(prompts.box[0]) == (4, 6, 6) and tuple(prompts.box[1]) == (8, 17, 17)
    assert len(prompts.of("scribble")) == 0 and Prompts(voxels[:1], [POSITIVE]).box is None


def test_what_a_tool_keeps_is_made_once_for_the_process() -> None:
    made = []
    first = kept(("tool", 0), lambda: made.append(1) or object())
    assert kept(("tool", 0), lambda: made.append(1) or object()) is first and len(made) == 1
    assert kept(("tool", 1), lambda: made.append(1) or object()) is not first and len(made) == 2


def test_a_fiducial_file_keeps_the_label_of_each_point(tmp_path: Path) -> None:
    write_landmarks(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), tmp_path / "named.fcsv", [NEGATIVE, "a, b"])
    write_landmarks(np.array([[1.0, 2.0, 3.0]]), tmp_path / "plain.fcsv")
    assert read_landmarks(tmp_path / "named.fcsv")[1] == [NEGATIVE, "a, b"]  # a label may hold a comma
    assert read_landmarks(tmp_path / "plain.fcsv")[1] == ["F-1"]
    with pytest.raises(ValueError, match="one line"):
        write_landmarks(np.array([[1.0, 2.0, 3.0]]), tmp_path / "broken.fcsv", ["first\n#second"])
    dataset = Dataset(tmp_path, "mha")  # read, written as another case, read again: the labels follow the points
    points, attribute = dataset.read_data("named", "")
    dataset.write("PROMPT", "CASE", points, attribute)
    assert read_landmarks(tmp_path / "CASE" / "PROMPT.fcsv")[1] == [NEGATIVE, "a, b"]


def test_the_engine_gets_the_images_then_the_points_as_voxels_of_the_image_it_is_handed() -> None:
    seen = []

    def engine(image, prompts, device):
        seen.append((image, prompts, device))
        mask = np.zeros(sitk.GetArrayViewFromImage(image).shape, np.uint8)
        mask[tuple(np.rint(prompts.clicks[0][0]).astype(int))] = 1
        return mask[None]

    shape = (4, 5, 6)
    points = Attribute()
    points["Labels"] = json.dumps([POSITIVE, NEGATIVE])
    # voxels (z, y, x) = (1, 2, 3) and (3, 0, 5) of a grid of spacing (0.5, 0.5, 2) from (3, -1, 7), in physical (x, y, z)
    physical = torch.tensor([[[4.5, 0.0, 9.0], [5.5, -1.0, 13.0]]], dtype=torch.float64)
    last = torch.zeros(1, 1, *shape)
    last[0, 0, 2, 2, 2] = 1
    net = PromptedNet(engine)
    assert isinstance(net["Segmentation"], PromptedTool) and net.full_precision_outputs
    out = net["Segmentation"](
        torch.rand(1, 1, *shape), physical, last, attributes=[[_geometry()], [points], [_geometry()]]
    )

    image, prompts, device = seen[0]
    assert device == -1 and image.GetSpacing() == (0.5, 0.5, 2.0)
    assert prompts.voxels.tolist() == [[1.0, 2.0, 3.0], [3.0, 0.0, 5.0]] and prompts.kinds == [POSITIVE, NEGATIVE]
    assert prompts.previous is not None and prompts.previous[2, 2, 2] and prompts.previous.sum() == 1
    assert out.shape == (1, 1, *shape) and out[0, 0, 1, 2, 3] == 1 and out.sum() == 1


def test_a_2d_image_takes_its_points_first_two_and_a_last_mask_off_its_grid_is_refused() -> None:
    seen = []
    net = PromptedNet(lambda image, prompts, device: seen.append(prompts) or np.zeros((1, 5, 6), np.uint8), dim=2)
    flat = Attribute()
    flat["Origin"], flat["Spacing"], flat["Direction"] = (
        np.array([3.0, -1.0]),
        np.array([0.5, 0.5]),
        np.eye(2).flatten(),
    )
    points = Attribute()
    points["Labels"] = json.dumps([POSITIVE])
    physical = torch.tensor([[[4.5, 0.0, 0.0]]], dtype=torch.float64)  # the voxel (y, x) = (2, 3); z of a fiducial file
    image = torch.rand(1, 1, 5, 6)
    net["Segmentation"](image, physical, torch.zeros(1, 1, 5, 6), attributes=[[flat], [points], [flat]])
    assert seen[0].voxels.tolist() == [[2.0, 3.0]]
    moved = Attribute(flat)
    moved["Origin"] = np.array([0.0, 0.0])
    with pytest.raises(KonfAIError, match="not on the image's grid"):
        net["Segmentation"](image, physical, torch.zeros(1, 1, 5, 6), attributes=[[flat], [points], [moved]])


MODEL = """
import numpy as np
import SimpleITK as sitk
from konfai.network.prompted import PromptedNet

SEEN = []


def engine(image, prompts, device):
    SEEN.append((image.GetSize(), image.GetSpacing(), prompts))
    mask = np.zeros(sitk.GetArrayViewFromImage(image).shape, np.uint8)
    for voxel, positive in prompts.clicks:
        mask[tuple(np.rint(voxel).astype(int))] = positive
    return mask[None]


class Net(PromptedNet):
    def __init__(self) -> None:
        super().__init__(engine)
"""


def _group(transforms: object) -> dict:
    return {"transforms": transforms, "patch_transforms": None, "is_input": True}


def _predict(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, patch: list[int] | dict | None, chain: object) -> tuple:
    """One case, a positive then a negative click on voxels of its image: the engine's calls and the mask."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "model.py").write_text(MODEL)
    case = tmp_path / "Dataset" / "CASE"
    case.mkdir(parents=True)
    image = sitk.GetImageFromArray(np.random.RandomState(0).rand(20, 48, 40).astype(np.float32))
    image.SetSpacing((0.8, 0.9, 2.5))
    image.SetOrigin((-30.0, 12.0, 5.0))
    image.SetDirection((0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0))  # not the canonical orientation
    last = sitk.Image(image.GetSize(), sitk.sitkUInt8)
    last.CopyInformation(image)
    sitk.WriteImage(image, str(case / "IMAGE.mha"))
    sitk.WriteImage(last, str(case / "PREVIOUS.mha"))
    clicked = [(30, 20, 10), (12, 25, 8)]  # (i, j, k)
    write_landmarks(
        np.array([image.TransformIndexToPhysicalPoint(index) for index in clicked]),
        case / "PROMPT.fcsv",
        [POSITIVE, NEGATIVE],
    )
    workspace = api.predict(
        [],
        {
            "Predictor": {
                "train_name": "PROMPTED",
                "Model": {"classpath": "model:Net"},
                "Dataset": {
                    "dataset_filenames": ["./Dataset:a:mha"],
                    "groups_src": {
                        "IMAGE": {"groups_dest": {"IMAGE": _group(chain)}},
                        "PROMPT": {"groups_dest": {"PROMPT": _group(None)}},
                        "PREVIOUS": {"groups_dest": {"PREVIOUS": _group(chain)}},
                    },
                    "Patch": patch if isinstance(patch, dict) else {"patch_size": patch, "overlap": 0},
                    "batch_size": 1,
                    "num_workers": 0,
                    "augmentations": None,
                },
                "outputs_dataset": {
                    "Segmentation": {
                        "OutputDataset": {
                            "dataset_filename": "Dataset:mha",
                            "group": "PRED",
                            "same_as_group": "IMAGE:IMAGE",
                            "final_transforms": {"TensorCast": {"dtype": "uint8", "inverse": False}},
                        }
                    }
                },
            }
        },
        quiet=True,
        overwrite=True,
    )
    import model  # the module the run imported: what its engine saw

    seen = list(model.SEEN)
    model.SEEN.clear()
    return image, clicked, seen, sitk.ReadImage(str(workspace / "Dataset" / "CASE" / "PRED.mha"))


@pytest.mark.parametrize("patch", [[0, 0, 0], None])
def test_a_prediction_hands_the_points_on_the_grid_its_chain_leaves_and_brings_the_mask_back_on_the_image_s(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, patch: list[int] | None
) -> None:
    image, clicked, seen, predicted = _predict(tmp_path, monkeypatch, patch, {"Canonical": {}})
    ((size, _spacing, prompts),) = seen
    assert size != image.GetSize()  # the image as Canonical leaves it, its axes in another order
    assert prompts.kinds == [POSITIVE, NEGATIVE] and prompts.previous is None
    assert np.allclose(prompts.voxels, np.rint(prompts.voxels), atol=1e-4)  # a voxel of the image stays a voxel
    assert predicted.GetSize() == image.GetSize() and np.allclose(predicted.GetOrigin(), image.GetOrigin())
    mask = sitk.GetArrayFromImage(predicted)
    assert mask.sum() == 1 and mask[clicked[0][::-1]] == 1  # the positive click, where it was given


def test_a_case_cut_in_several_patches_is_refused_a_patch_not_knowing_its_place_in_the_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(KonfAIError, match="patches"):
        _predict(tmp_path, monkeypatch, [10, 0, 0], None)


def test_a_patch_that_may_pad_before_the_image_is_refused_its_points_moving_by_the_padding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(KonfAIError, match="pad before"):
        _predict(tmp_path, monkeypatch, {"patch_size": [24, 52, 44], "overlap": 0, "placement": "end"}, None)


def test_cases_holding_different_numbers_of_points_are_never_batched_together() -> None:
    from types import SimpleNamespace

    from konfai.data.data_manager.sources import DataPrediction

    def points(count: int) -> SimpleNamespace:
        return SimpleNamespace(shapes=[[3]], base_shape=[count, 3])

    def stack(*managers: SimpleNamespace) -> bool:
        return DataPrediction.patches_stack(SimpleNamespace(patch=None, _managers={"PROMPT": list(managers)}))

    assert stack(points(2), points(2)) and not stack(points(1), points(2))


def test_a_session_that_keeps_its_model_builds_and_loads_it_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from konfai.predictor import workflow

    monkeypatch.setattr(workflow, "_KEPT_MODEL", {})
    monkeypatch.delenv("KONFAI_KEEP_MODEL", raising=False)
    for name in ("once", "kept", "again"):
        (tmp_path / name).mkdir()
    _predict(tmp_path / "once", monkeypatch, [0, 0, 0], None)
    assert workflow._KEPT_MODEL == {}  # a model is kept only when the process asks
    monkeypatch.setenv("KONFAI_KEEP_MODEL", "1")
    _predict(tmp_path / "kept", monkeypatch, [0, 0, 0], None)
    kept = workflow._KEPT_MODEL["composite"]
    _, clicked, _, predicted = _predict(tmp_path / "again", monkeypatch, [0, 0, 0], None)
    assert workflow._KEPT_MODEL["composite"] is kept
    assert sitk.GetArrayFromImage(predicted)[clicked[0][::-1]] == 1  # and it still predicts
    kept_recipe, again = (
        (tmp_path / name / "Predictions/PROMPTED/Prediction.yml").read_text() for name in ("kept", "again")
    )
    assert again == kept_recipe  # the kept model's config still bound and resolved
