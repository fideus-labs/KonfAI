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

"""Ensembles of app references: every member of every run reduced into the output, slab by slab, and kept as one
stack under -uncertainty."""

import json
from pathlib import Path

import numpy as np
import pytest

sitk = pytest.importorskip("SimpleITK")

from konfai.data.transform.ensemble import MEMBERS_ATTRIBUTE  # noqa: E402
from konfai.utils.dataset import Dataset  # noqa: E402
from konfai_apps import ensemble  # noqa: E402
from konfai_apps.errors import AppRepositoryError  # noqa: E402


@pytest.fixture(autouse=True)
def _thin_slabs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ensemble, "SLAB_PLANES", 3)  # several slabs on the small volumes below


def _image(array: np.ndarray, path: Path, vector: bool = False) -> Path:
    image = sitk.GetImageFromArray(array, isVector=vector)
    image.SetSpacing((1.5, 1.0, 2.0))
    image.SetOrigin((3.0, -2.0, 1.0))
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(path))
    return path


def test_outputs_and_a_stack_average_their_members(tmp_path: Path) -> None:
    """Two runs of one member and one of two (its stack): the output is the mean of the four, the stack holds them in
    the runs' order, and its header says four."""
    rng = np.random.default_rng(0)
    a, b, c, d = (rng.normal(size=(7, 5, 4)).astype(np.float32) for _ in range(4))
    one = _image(a, tmp_path / "r0" / "P000" / "Output.mha")
    two = _image(b, tmp_path / "r1" / "P000" / "Output.mha")
    stack_path = tmp_path / "r2" / "P000" / "InferenceStack.mha"
    _image(np.stack([c, d], -1), stack_path, vector=True)  # a stack written before the count: a member a channel

    counts = ensemble.combine(
        [one, two, stack_path],
        "Mean",
        tmp_path / "out" / "P000" / "Output.mha",
        "mha",
        tmp_path / "out" / "P000" / "InferenceStack.mha",
    )

    assert counts == [1, 1, 2]
    np.testing.assert_allclose(
        sitk.GetArrayFromImage(sitk.ReadImage(str(tmp_path / "out/P000/Output.mha"))), (a + b + c + d) / 4, atol=1e-6
    )
    stack, header = Dataset(str(tmp_path / "out"), "mha").read_data("InferenceStack", "P000")
    assert list(stack.shape) == [4, 7, 5, 4] and int(float(header[MEMBERS_ATTRIBUTE])) == 4
    np.testing.assert_allclose(stack[3], d, atol=1e-6)


def test_fields_average_into_a_transform_file_and_stack_their_components(tmp_path: Path) -> None:
    """Two displacement fields written as ITK transform files: the output is their mean, a transform SimpleITK reads;
    the stack holds six channels, two members of three."""
    rng = np.random.default_rng(1)
    fields = [rng.normal(size=(6, 5, 4, 3)) for _ in range(2)]
    paths = []
    for index, values in enumerate(fields):
        image = sitk.GetImageFromArray(values, isVector=True)
        image.SetSpacing((1.5, 1.0, 2.0))
        path = tmp_path / f"r{index}" / "P000" / "Transform.h5"
        path.parent.mkdir(parents=True)
        sitk.WriteTransform(sitk.DisplacementFieldTransform(image), str(path))
        paths.append(path)

    ensemble.combine(
        paths,
        "Mean",
        tmp_path / "out" / "P000" / "Transform.h5",
        "itktransform",
        tmp_path / "out" / "P000" / "InferenceStack.mha",
    )

    mean = sitk.DisplacementFieldTransform(sitk.ReadTransform(str(tmp_path / "out/P000/Transform.h5")))
    np.testing.assert_allclose(
        sitk.GetArrayFromImage(mean.GetDisplacementField()), (fields[0] + fields[1]) / 2, atol=1e-5
    )
    stack, header = Dataset(str(tmp_path / "out"), "mha").read_data("InferenceStack", "P000")
    assert list(stack.shape) == [6, 6, 5, 4] and int(float(header[MEMBERS_ATTRIBUTE])) == 2


def test_an_output_of_several_files_moves_into_place_whole(tmp_path: Path) -> None:
    """A MetaImage output is a header and its data file: both reach the case folder, so the output reads back."""
    rng = np.random.default_rng(2)
    a, b = (rng.normal(size=(7, 5, 4)).astype(np.float32) for _ in range(2))
    sources = [_image(a, tmp_path / "r0" / "Output.mha"), _image(b, tmp_path / "r1" / "Output.mha")]

    ensemble.combine(sources, "Mean", tmp_path / "out" / "P000" / "Output.mhd", "mhd")

    assert sorted(path.suffix for path in (tmp_path / "out" / "P000").iterdir()) == [".mhd", ".raw"]
    written = sitk.GetArrayFromImage(sitk.ReadImage(str(tmp_path / "out" / "P000" / "Output.mhd")))
    np.testing.assert_allclose(written, (a + b) / 2, atol=1e-6)


@pytest.mark.parametrize("operator", ["Vote", "Mean"])
def test_label_maps_vote_whatever_the_operator(tmp_path: Path, operator: str) -> None:
    """Label maps (integer voxels) are voted under the operator an app declares for its probabilities too: a mean of
    labels 2 and 5 is no label, and the output keeps the labels' type."""
    labels = [np.full((4, 3, 3), value, dtype=np.uint8) for value in (2, 2, 5)]
    paths = [_image(array, tmp_path / f"r{i}" / "P000" / "Seg.nii.gz") for i, array in enumerate(labels)]
    ensemble.combine(paths, operator, tmp_path / "out" / "P000" / "Seg.nii.gz", "nii.gz")
    image = sitk.ReadImage(str(tmp_path / "out/P000/Seg.nii.gz"))
    assert image.GetPixelID() == sitk.sitkUInt8 and np.all(sitk.GetArrayFromImage(image) == 2)


def test_members_on_two_grids_are_refused(tmp_path: Path) -> None:
    one = _image(np.zeros((4, 3, 3), np.float32), tmp_path / "r0" / "P000" / "Output.mha")
    two = _image(np.zeros((5, 3, 3), np.float32), tmp_path / "r1" / "P000" / "Output.mha")
    with pytest.raises(AppRepositoryError, match="same grid"):
        ensemble.combine([one, two], "Mean", tmp_path / "out" / "P000" / "Output.mha", "mha")


def test_members_with_the_same_shape_on_two_grids_are_refused(tmp_path: Path) -> None:
    """The same shape is not the same grid: a member 100 mm away holds other points, and is not averaged."""
    one = _image(np.zeros((4, 3, 3), np.float32), tmp_path / "r0" / "P000" / "Output.mha")
    two = _image(np.zeros((4, 3, 3), np.float32), tmp_path / "r1" / "P000" / "Output.mha")
    moved = sitk.ReadImage(str(two))
    moved.SetOrigin((103.0, -2.0, 1.0))
    sitk.WriteImage(moved, str(two))
    with pytest.raises(AppRepositoryError, match="another grid"):
        ensemble.combine([one, two], "Mean", tmp_path / "out" / "P000" / "Output.mha", "mha")


def test_a_run_hands_over_its_stack_where_it_wrote_one(tmp_path: Path) -> None:
    """members_of: an output and, beside it, a stack: the members are the stack's; without one, the output."""
    case = tmp_path / "run" / "App" / "Output" / "P000"
    _image(np.zeros((4, 3, 3), np.float32), case / "Output.mha")
    alone = tmp_path / "run" / "App" / "Output" / "P001"
    _image(np.zeros((4, 3, 3), np.float32), alone / "Output.mha")
    _image(np.zeros((4, 3, 3, 2), np.float32), case / "InferenceStack.mha", vector=True)

    found = ensemble.members_of(tmp_path / "run")

    assert found[("Output", "P000", "Output")] == (case / "Output.mha", case / "InferenceStack.mha")
    assert found[("Output", "P001", "Output")] == (alone / "Output.mha", alone / "Output.mha")


def test_the_checkpoints_an_app_runs_by_default_are_members(tmp_path: Path) -> None:
    """An app whose config ensembles two checkpoints, run with no -ensemble: each member run still hands over its
    stack, so a vote or a median sees both checkpoints, and the record names them."""
    rng = np.random.default_rng(1)
    members = {name: rng.normal(size=(2, 4, 3, 3)).astype(np.float32) for name in ("A", "B")}

    class App:
        def __init__(self, name: str) -> None:
            self.name = name

        def infer(self, output: Path, uncertainty: bool, **_: object) -> None:
            case = output / "App" / "Output" / "P000"
            _image(members[self.name].mean(0), case / "Output.mha")
            if uncertainty:
                _image(np.moveaxis(members[self.name], 0, -1), case / "InferenceStack.mha", vector=True)

    ensemble.infer_references(["A", "B"], App, tmp_path / "out", "Median", False, lambda reference: None)

    output = sitk.GetArrayFromImage(sitk.ReadImage(str(tmp_path / "out" / "Output" / "P000" / "Output.mha")))
    np.testing.assert_allclose(output, np.median(np.concatenate(list(members.values())), 0), atol=1e-6)
    record = json.loads((tmp_path / "out" / "konfai-run.json").read_text(encoding="utf-8"))
    assert record["members"] == ["A [1/2]", "A [2/2]", "B [1/2]", "B [2/2]"]
