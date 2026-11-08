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

"""The stack of an ensemble: its members' channels member-major, its member count in its header, whatever a member's
channel count (a displacement field has three)."""

from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("SimpleITK")

from konfai.data.transform import InferenceStack, Magnitude, SegmentationDisagreement, StandardDeviation
from konfai.data.transform.ensemble import MEMBERS_ATTRIBUTE, stack_members
from konfai.utils.dataset import DISPLACEMENT_FIELD_ATTRIBUTE, Attribute, Dataset


def _attributes(field: bool = False) -> Attribute:
    attributes = Attribute()
    if field:
        attributes[DISPLACEMENT_FIELD_ATTRIBUTE] = "true"
    attributes["Origin"] = np.array([1.0, 2.0, 3.0])
    attributes["Spacing"] = np.array([1.5, 1.0, 2.0])
    attributes["Direction"] = np.eye(3).ravel()
    return attributes


@pytest.mark.parametrize("fmt", ["mha", "nrrd", "omezarr"])
def test_a_stack_of_fields_keeps_each_member_s_components(tmp_path: Path, fmt: str) -> None:
    """Three members of a three-component field: nine channels, member-major, and the count in the header, so the
    reader splits members from components; the output (their mean) carries no count. OME-Zarr holds the stack as
    plain channels: no RFC-5 axis types nine components as one field."""
    members = torch.randn(3, 3, 4, 5, 6)
    attributes = _attributes(field=True)
    mean = InferenceStack(dataset=f"{tmp_path}:{fmt}")("P000", members, attributes)

    torch.testing.assert_close(mean, members.mean(0))
    assert MEMBERS_ATTRIBUTE not in attributes
    data, header = Dataset(str(tmp_path), fmt).read_data("InferenceStack", "P000")
    assert list(data.shape) == [9, 4, 5, 6] and DISPLACEMENT_FIELD_ATTRIBUTE in header
    torch.testing.assert_close(stack_members(torch.from_numpy(data), header), members, rtol=0, atol=1e-6)


def test_the_spread_of_a_stack_of_fields_is_the_spread_of_its_vectors() -> None:
    """StandardDeviation then Magnitude: the per-component standard deviation across the members, then its norm, the
    root mean square distance of the members to their mean vector (N - 1). Two members that agree in length but not in
    direction spread; the standard deviation of their lengths read 0."""
    members = torch.zeros(2, 3, 2, 2, 2)
    members[0, 0], members[1, 1] = 5.0, 5.0  # same length, perpendicular
    stack = members.reshape(6, 2, 2, 2)
    header = Attribute()
    header[MEMBERS_ATTRIBUTE] = 2
    spread = Magnitude()("P000", StandardDeviation()("P000", stack, header), header)

    assert list(spread.shape) == [1, 2, 2, 2]
    torch.testing.assert_close(spread, torch.full((1, 2, 2, 2), float(np.hypot(*members[:, :2, 0, 0, 0].std(0)))))
    assert MEMBERS_ATTRIBUTE not in header


def test_a_stack_without_a_count_is_one_member_a_channel() -> None:
    """A stack written before the count (or by hand) keeps its meaning: every channel a member."""
    stack = torch.tensor([[[1.0]], [[3.0]]])[:, :, :, None]  # two one-channel members
    torch.testing.assert_close(StandardDeviation()("P000", stack, Attribute()), stack.std(0)[None])
    labels = torch.tensor([1, 1, 2])[:, None, None, None]
    assert float(SegmentationDisagreement()("P000", labels, Attribute()).flatten()[0]) == pytest.approx(1 / 3)


def test_copies_folded_one_at_a_time_average_their_members(tmp_path: Path) -> None:
    """Two TTA copies of two members of three channels: the stack holds the four members, and the mean divides by
    four, not by the twelve channel rows written."""
    copies = [torch.randn(2, 3, 4, 5, 6), torch.randn(2, 3, 4, 5, 6)]
    stage = InferenceStack(dataset=f"{tmp_path}:mha")
    mean = stage.fold_copies("P000", iter(copies), 2, _attributes())

    torch.testing.assert_close(mean, torch.cat(copies).mean(0))
    data, header = Dataset(str(tmp_path), "mha").read_data("InferenceStack", "P000")
    assert list(data.shape) == [12, 4, 5, 6] and int(float(header[MEMBERS_ATTRIBUTE])) == 4


def test_a_stack_of_fields_is_refused_where_its_count_is_lost(tmp_path: Path) -> None:
    """NIfTI keeps no free header: nine channels would read as nine members, so a stack of fields is refused there."""
    from konfai.utils.errors import TransformError

    with pytest.raises(TransformError, match="NIfTI"):
        InferenceStack(dataset=f"{tmp_path}:nii.gz")("P000", torch.randn(2, 3, 4, 5, 6), _attributes())


@pytest.mark.parametrize("fmt", ["nii.gz", "itktransform"])
def test_a_stack_goes_beside_an_output_that_cannot_hold_it(tmp_path: Path, fmt: str) -> None:
    """Named by no dataset, the stack goes with the output; beside a NIfTI output or a transform file, which cannot
    hold it, as MetaImage, as konfai-apps' ensembles write it."""
    members = torch.randn(2, 3, 4, 5, 6)
    stage = InferenceStack()
    stage.set_datasets([Dataset(str(tmp_path), fmt)])
    stage("P000", members, _attributes(field=True))

    assert [path.name for path in (tmp_path / "P000").iterdir()] == ["InferenceStack.mha"]
    data, header = Dataset(str(tmp_path), "mha").read_data("InferenceStack", "P000")
    torch.testing.assert_close(stack_members(torch.from_numpy(data), header), members, rtol=0, atol=1e-6)


# ---------------------------------------------------------------- a stored transform, one member at a time


def _member_fields(tmp_path: Path, fmt: str = "mha", marked: bool = True) -> tuple[list, Dataset]:
    """Two displacement fields stacked as InferenceStack writes them: [6, z, y, x], two members of three."""
    import SimpleITK as sitk

    z, y, x = np.mgrid[0:8, 0:9, 0:10].astype(np.float64)
    fields = [
        np.stack([a * np.sin(y / 3), b * np.cos(x / 4), 0.5 * np.sin(z / 2)], -1) for a, b in ((1.0, 0.5), (0.3, 1.2))
    ]
    members = torch.from_numpy(np.stack([np.moveaxis(field, -1, 0) for field in fields])).float()
    attributes = _attributes(
        field=marked
    )  # unmarked: fields as NIfTI holds them, told a field when used as a transform
    InferenceStack(dataset=f"{tmp_path / 'stack'}:{fmt}")("P000", members, attributes)
    transforms = []
    for field in fields:
        image = sitk.GetImageFromArray(field, isVector=True)
        image.SetOrigin((1.0, 2.0, 3.0))
        image.SetSpacing((1.5, 1.0, 2.0))
        transforms.append(sitk.DisplacementFieldTransform(image))
    return transforms, Dataset(str(tmp_path / "stack"), fmt)


def test_a_member_view_reads_one_member_by_region(tmp_path: Path) -> None:
    from konfai.data.transform.ensemble import MemberView

    _, dataset = _member_fields(tmp_path)
    stack = dataset.read_data("InferenceStack", "P000")[0]
    view = MemberView(dataset, {"InferenceStack"}, 1)
    shape, header = view.get_infos("InferenceStack", "P000")
    assert shape == [3, 8, 9, 10] and MEMBERS_ATTRIBUTE not in header
    np.testing.assert_array_equal(view.read_data("InferenceStack", "P000")[0], stack[3:6])
    region = (slice(1, 3), slice(2, 5), slice(None), slice(None))
    np.testing.assert_array_equal(view.read_data_slice("InferenceStack", "P000", region)[0], stack[4:6, 2:5])


@pytest.mark.parametrize(("fmt", "marked"), [("mha", True), ("nrrd", True), ("omezarr", True), ("mha", False)])
def test_points_and_images_go_through_every_member_of_a_stack(tmp_path: Path, fmt: str, marked: bool) -> None:
    """TransformPoints through a stack: one set of points per member, [M, N, 3]; Resample: one image per member,
    stacked with their count. Each is SimpleITK's own through that member's field."""
    import SimpleITK as sitk
    from konfai.data.transform import Resample, TransformPoints
    from konfai.utils.dataset import image_to_data

    transforms, dataset = _member_fields(tmp_path, fmt, marked)
    points = np.array([[4.0, 5.0, 6.0], [8.0, 6.0, 10.0], [12.0, 9.0, 14.0]])
    stage = TransformPoints({"InferenceStack": False})
    stage.set_datasets([dataset])
    moved = stage("P000", torch.from_numpy(points), Attribute()).numpy()
    assert moved.shape == (2, 3, 3)
    for member, transform in enumerate(transforms):
        np.testing.assert_allclose(moved[member], [transform.TransformPoint(p) for p in points.tolist()], atol=1e-5)

    image = sitk.GetImageFromArray(np.random.default_rng(0).normal(size=(8, 9, 10)).astype(np.float32))
    image.SetOrigin((1.0, 2.0, 3.0))
    image.SetSpacing((1.5, 1.0, 2.0))
    data, attributes = image_to_data(image)
    resample = Resample(transforms={"InferenceStack": False}, interpolation="linear")
    resample.set_datasets([dataset])
    resample.transform_shape("Volume_0", "P000", list(data.shape[1:]), attributes)  # the plan's probe, member by member
    stacked = resample("P000", torch.from_numpy(data), attributes)
    assert list(stacked.shape) == [2, 8, 9, 10] and int(float(attributes[MEMBERS_ATTRIBUTE])) == 2
    for member, transform in enumerate(transforms):
        want = sitk.GetArrayFromImage(sitk.Resample(image, image, transform, sitk.sitkLinear, 0.0))
        np.testing.assert_allclose(stacked[member].numpy(), want, atol=1e-4)


def test_the_spread_of_points_its_calibration_and_a_quantile() -> None:
    from konfai.metric.measure import PointSpread, Quantile, SpreadCalibration

    rng = np.random.default_rng(3)
    members = torch.from_numpy(rng.normal(size=(1, 4, 6, 3)))
    per_landmark = np.linalg.norm(members[0].numpy().std(0, ddof=1), axis=-1)
    assert float(PointSpread()(members)[0]) == pytest.approx(per_landmark.mean())
    # the error grows with the spread: a calibration of 1
    reference = members[0].mean(0) + torch.from_numpy(per_landmark)[:, None] * torch.tensor([1.0, 0.0, 0.0])
    assert SpreadCalibration()(members, reference[None])[1] == pytest.approx(1.0)
    # ties share their mean rank: spreads 1, 1, 2, 2 against errors 1, 2, 1, 2 say nothing, a calibration of 0
    two = torch.zeros(1, 2, 4, 3)
    two[0, 1, :, 0] = torch.tensor([1.0, 1.0, 2.0, 2.0]) * np.sqrt(2.0)  # the spread of two members: |d| / sqrt(2)
    tied = two[0].mean(0) + torch.tensor([1.0, 2.0, 1.0, 2.0])[:, None] * torch.tensor([0.0, 1.0, 0.0])
    assert SpreadCalibration()(two, tied[None])[1] == pytest.approx(0.0, abs=1e-6)
    values = torch.arange(101.0)[None, None]
    assert Quantile(0.95)(values)[1] == pytest.approx(95.0) and Quantile(0.95).get_name() == "Quantile95"


def test_the_spread_of_fields_is_no_field() -> None:
    """A field's header marks it a displacement field; its spread and its magnitude are images in its unit, and a
    reader would take them for transforms if they kept the mark."""
    from konfai.utils.dataset import DISPLACEMENT_FIELD_ATTRIBUTE

    header = Attribute()
    header[MEMBERS_ATTRIBUTE] = 2
    header[DISPLACEMENT_FIELD_ATTRIBUTE] = "true"
    spread = StandardDeviation()("P000", torch.randn(6, 2, 2, 2), header)
    Magnitude()("P000", spread, header)
    assert DISPLACEMENT_FIELD_ATTRIBUTE not in header and MEMBERS_ATTRIBUTE not in header


def test_a_streamed_save_drops_what_the_chain_removed() -> None:
    """A sweep publishes the plan-time header with what the chain's stages added, and without what they removed: a
    spread written slab by slab is no stack and no field, as the whole-volume pass leaves it."""
    from konfai.data.patching.sweep import _sweep_header

    evolved = _attributes(field=True)
    evolved[MEMBERS_ATTRIBUTE] = 2
    scope = Attribute(evolved)
    before = set(scope.keys())
    StandardDeviation()("P000", torch.randn(6, 2, 2, 2), scope)
    scope["Added"] = 1.0

    header = _sweep_header(evolved, scope, before)
    assert MEMBERS_ATTRIBUTE not in header and DISPLACEMENT_FIELD_ATTRIBUTE not in header
    assert "Added" in header and "Spacing" in header


def test_every_member_pulls_the_case_from_its_own_grid(tmp_path: Path) -> None:
    """A moving image on a grid of its own, pulled onto the fixed one through each member of a stack: the second
    member reads the case where it lies, as the first does, not on the grid the first one left in the header."""
    import SimpleITK as sitk
    from konfai.data.transform import Resample
    from konfai.utils.dataset import image_to_data

    fixed = sitk.Image([10, 9, 8], sitk.sitkFloat32)
    fixed.SetSpacing((1.5, 1.0, 2.0))
    fixed.SetOrigin((1.0, 2.0, 3.0))
    shifts = (0.5, -1.0)
    members = torch.stack([torch.full((3, 8, 9, 10), shift) for shift in shifts])
    InferenceStack(dataset=f"{tmp_path / 'stack'}:mha")("P000", members, _attributes(field=True))
    sitk.WriteImage(fixed, str(tmp_path / "stack" / "P000" / "Fixed.mha"))
    moving = sitk.GetImageFromArray(np.random.default_rng(0).normal(size=(12, 14, 16)).astype(np.float32))
    moving.SetSpacing((1.0, 0.8, 1.5))
    moving.SetOrigin((-2.0, 0.0, 1.0))
    data, header = image_to_data(moving)

    resample = Resample(reference="{case}", reference_group="Fixed", transforms={"InferenceStack": False})
    resample.set_datasets([Dataset(str(tmp_path / "stack"), "mha")])
    resample.transform_shape("Volume_0", "P000", list(data.shape[1:]), header)
    pulled = resample("P000", torch.from_numpy(data), header).numpy()

    for member, shift in enumerate(shifts):
        want = sitk.Resample(moving, fixed, sitk.TranslationTransform(3, [shift] * 3), sitk.sitkLinear, 0.0)
        np.testing.assert_allclose(pulled[member], sitk.GetArrayFromImage(want), atol=1e-4)
    np.testing.assert_allclose(header.get_np_array("Spacing"), fixed.GetSpacing())
