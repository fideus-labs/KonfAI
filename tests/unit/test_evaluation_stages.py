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

"""Stages and criteria that derive what a registration is scored on: a body mask by Otsu's threshold, the labels both
maps hold, landmarks carried by a field, a TRE percentile and the labels a warp lost. Each against the reading it
reproduces: SimpleITK's filter and transform, NumPy's percentile."""

import numpy as np
import pytest
import torch
from konfai.data.transform import Displace, KeepLabels, Otsu
from konfai.utils.dataset import Attribute, Dataset
from konfai.utils.errors import MeasureError, TransformError

sitk = pytest.importorskip("SimpleITK")


def gaussian_filter(array: np.ndarray, sigma: float) -> np.ndarray:
    """A smooth random volume, by SimpleITK: scipy is no dependency."""
    return sitk.GetArrayFromImage(sitk.DiscreteGaussian(sitk.GetImageFromArray(array), sigma**2))


class _Store:
    """One case's groups, as a dataset serves them: an array, channels first, and its geometry."""

    def __init__(self, **groups: tuple[np.ndarray, Attribute]) -> None:
        self.groups = groups
        self.regions: list[tuple[str, tuple[slice, ...]]] = []

    def is_dataset_exist(self, group: str, name: str) -> bool:
        return group in self.groups

    def read_data(self, group: str, name: str) -> tuple[np.ndarray, Attribute]:
        return self.groups[group]

    def read_data_slice(self, group: str, name: str, slices: tuple[slice, ...]) -> tuple[np.ndarray, Attribute]:
        self.regions.append((group, slices))
        return self.groups[group][0][slices], self.groups[group][1]

    def bounded_region_reads(self, group: str, name: str) -> bool:
        return True  # a region is read alone, as from an uncompressed file

    def get_infos(self, group: str, name: str) -> tuple[list[int], Attribute]:
        return list(self.groups[group][0].shape), self.groups[group][1]

    def iter_data_blocks(self, group: str, name: str):
        array = self.groups[group][0]
        return lambda: (array[:, index] for index in range(array.shape[1]))


def _geometry(image: sitk.Image) -> Attribute:
    return Attribute(
        {
            "Origin": np.asarray(image.GetOrigin()),
            "Spacing": np.asarray(image.GetSpacing()),
            "Direction": np.asarray(image.GetDirection()),
        }
    )


@pytest.mark.parametrize("dtype", [np.int16, np.float32])
def test_otsu_gives_itks_mask(dtype):
    rng = np.random.default_rng(0)
    volume = (gaussian_filter(rng.standard_normal((20, 24, 28)), 2) * 4000).astype(dtype)
    reference = sitk.GetArrayFromImage(
        sitk.OtsuThreshold(sitk.Cast(sitk.GetImageFromArray(volume), sitk.sitkFloat32), 0, 1)
    )
    whole = Otsu()("a", torch.from_numpy(volume[None]), Attribute())
    assert whole.dtype == torch.uint8 and whole.shape == (1, 20, 24, 28)
    assert np.array_equal(whole[0].numpy(), reference) and 0.1 < reference.mean() < 0.9


def test_otsu_reads_a_colour_image_as_the_mean_of_its_channels():
    rng = np.random.default_rng(1)
    colour = rng.integers(0, 256, (3, 32, 40)).astype(np.uint8)
    colour[:, 8:24, 10:30] //= 4
    image = sitk.GetImageFromArray(colour.transpose(1, 2, 0), isVector=True)
    grey = sum(sitk.VectorIndexSelectionCast(image, c, sitk.sitkFloat32) for c in range(3)) / 3
    reference = sitk.GetArrayFromImage(sitk.OtsuThreshold(grey, 0, 1))
    assert np.array_equal(Otsu()("a", torch.from_numpy(colour), Attribute())[0].numpy(), reference)


def test_keep_labels_keeps_what_the_other_map_holds():
    fixed = np.zeros((1, 6, 6, 6), np.uint8)
    fixed[0, :2], fixed[0, 2:4], fixed[0, 4:] = 1, 2, 3
    moving = np.zeros((1, 6, 6, 6), np.int32)
    moving[0, 0], moving[0, 1], moving[0, 2] = 3, 1, 300  # holds 0, 1, 3 and a label this map's type cannot
    stage = KeepLabels("MOVING")
    stage.set_datasets([_Store(MOVING=(moving, Attribute()))])
    kept = stage("a", torch.from_numpy(fixed), Attribute())
    assert kept.dtype == torch.uint8 and sorted(torch.unique(kept).tolist()) == [0, 1, 3]
    assert int((kept == 3).sum()) == int((torch.from_numpy(fixed) == 3).sum())  # a kept label keeps every voxel


@pytest.mark.parametrize("dtype", [torch.uint16, torch.bool])
def test_keep_labels_takes_a_map_isin_has_no_kernel_for(dtype):
    fixed = torch.zeros(1, 4, 4, 4, dtype=dtype)
    fixed[0, :2] = 1
    stage = KeepLabels("MOVING")
    stage.set_datasets([_Store(MOVING=(np.zeros((1, 4, 4, 4), np.uint8), Attribute()))])  # holds 0 alone
    kept = stage("a", fixed, Attribute())
    assert kept.dtype == dtype and not kept.any()


@pytest.mark.parametrize("extension", ["mha", "gipl"])
def test_keep_labels_refuses_a_map_it_would_read_whole(tmp_path, extension):
    # A GIPL cannot serve a region alone: its labels would be read off the whole volume, which no plan counts.
    (tmp_path / "a").mkdir()
    sitk.WriteImage(sitk.Image([4, 4, 4], sitk.sitkUInt8) + 1, str(tmp_path / "a" / f"MOVING.{extension}"))
    stage = KeepLabels("MOVING")
    stage.set_datasets([Dataset(str(tmp_path), extension)])
    fixed = torch.tensor([[[[0, 1, 2]]]], dtype=torch.uint8)
    if extension == "gipl":
        with pytest.raises(TransformError, match="MOVING"):
            stage("a", fixed, Attribute())
    else:
        assert stage("a", fixed, Attribute()).flatten().tolist() == [0, 1, 0]


def _field(size, spacing, origin, rotation, components) -> sitk.Image:
    """A smooth random field on an oblique, anisotropic grid, as written: one component per world axis, in mm."""
    rng = np.random.default_rng(2)
    values = np.stack([gaussian_filter(rng.standard_normal(size[::-1]), 2) * 6 for _ in range(components)], -1)
    image = sitk.GetImageFromArray(values.astype(np.float32), isVector=True)
    image.SetSpacing(spacing)
    image.SetOrigin(origin)
    image.SetDirection(rotation)
    return image


def _whole_field(field: sitk.Image, points: np.ndarray) -> np.ndarray:
    """The points moved by SimpleITK's transform of the whole field."""
    transform = sitk.DisplacementFieldTransform(sitk.Cast(field, sitk.sitkVectorFloat64))
    rank = field.GetDimension()
    return np.array([[*transform.TransformPoint(p[:rank].tolist()), *p[rank:]] for p in points])


def _displaced(field: sitk.Image, points: np.ndarray) -> tuple[np.ndarray, "_Store"]:
    store = _Store(DVF=(np.moveaxis(sitk.GetArrayFromImage(field), -1, 0), _geometry(field)))
    stage = Displace("DVF")
    stage.set_datasets([store])
    return stage("a", torch.from_numpy(points), Attribute()).numpy(), store


def test_displace_moves_points_as_the_whole_fields_transform_does_reading_a_block_a_point():
    rotation = sitk.VersorTransform((0.2, -0.3, 0.4), 0.54).GetMatrix()
    field = _field((24, 20, 16), (1.5, 0.8, 2.5), (-30.0, 10.0, 5.0), rotation, 3)
    rng = np.random.default_rng(3)
    inside = [
        field.TransformContinuousIndexToPhysicalPoint(tuple(rng.uniform(0, n - 1) for n in (24, 20, 16)))
        for _ in range(20)
    ]
    beyond = field.TransformContinuousIndexToPhysicalPoint((26.0, 5.0, 5.0))  # past the field: ITK's own rule there
    points = np.array([*inside, beyond])
    moved, store = _displaced(field, points)
    assert np.allclose(moved, _whole_field(field, points), atol=1e-9) and not np.allclose(moved[:-1], points[:-1])
    assert len(store.regions) == len(points)  # one read a point, never the field whole
    assert all(s.stop - s.start <= 2 for _, slices in store.regions for s in slices[1:])


def test_displace_moves_the_first_two_coordinates_of_a_2d_case(tmp_path):
    field = _field((40, 30), (0.5, 0.8), (3.0, -2.0), (1.0, 0.0, 0.0, 1.0), 2)
    points = np.array([[*field.TransformContinuousIndexToPhysicalPoint(p), 0.0] for p in [(5.0, 7.5), (20.2, 12.0)]])
    (tmp_path / "a").mkdir()
    sitk.WriteImage(field, str(tmp_path / "a" / "DVF.mha"))
    stage = Displace("DVF")
    stage.set_datasets([Dataset(str(tmp_path), "mha")])
    moved = stage("a", torch.from_numpy(points), Attribute()).numpy()
    assert np.allclose(moved, _whole_field(field, points), atol=1e-9) and (moved[:, 2] == 0).all()


def test_tre_gives_the_mean_with_each_landmark_or_one_percentile():
    from konfai.metric.measure import TRE

    rng = np.random.default_rng(5)
    moved, moving = torch.tensor(rng.normal(0, 4, (1, 40, 3))), torch.tensor(rng.normal(0, 4, (1, 40, 3)))
    errors = np.linalg.norm(moved[0].numpy() - moving[0].numpy(), axis=1)
    loss, each = TRE()(moved, moving)
    assert float(loss) == pytest.approx(errors.mean()) and each.values.numpy() == pytest.approx(errors)
    loss, value = TRE(percentile=95)(moved, moving)
    assert float(value) == pytest.approx(np.percentile(errors, 95)) == pytest.approx(float(loss))
    with pytest.raises(MeasureError):
        TRE(percentile=120)


def test_missing_labels_counts_what_the_output_lacks_of_the_reference():
    from konfai.metric.measure import MissingLabels

    reference = torch.zeros(2, 1, 8, 8, 8, dtype=torch.int64)
    reference[0, 0, :2], reference[0, 0, 3:5], reference[0, 0, 6:] = 1, 2, 7
    reference[1, 0, :4] = 1
    output = reference.clone()
    output[0][output[0] == 2] = 0  # the first item loses label 2
    output[0, 0, 5] = 9  # a label the reference lacks is not counted
    assert MissingLabels()(output, reference)[1] == 0.5  # one of the first item, none of the second
    probabilities = torch.nn.functional.one_hot(output[:, 0], 10).movedim(-1, 1).float()
    assert MissingLabels()(probabilities, reference)[1] == 0.5  # one channel per label: read as its argmax
    assert MissingLabels(labels=[1, 7])(output, reference)[1] == 0.0
    binary = reference > 0  # a bool map, as Dice takes it
    assert MissingLabels()(torch.zeros_like(binary), binary)[1] == 1.0
    unread = reference[1:].float()
    unread[0, 0, 7] = float("nan")  # voxels the reference does not label: no label, even masked out (NaN * 0)
    assert MissingLabels()(reference[1:], unread)[1] == MissingLabels()(reference[1:], unread, unread == 1)[1] == 0.0
    mask = torch.zeros_like(reference)
    mask[:, :, :3] = 1  # outside the mask nothing is held, so nothing is missed
    assert MissingLabels()(output, reference, mask)[1] == 0.0


def test_missing_labels_compares_a_reference_of_another_shape():
    # The evaluator leaves it out of its same-grid refusal, and the reference is put on the output's grid.
    from types import SimpleNamespace

    from konfai.evaluator import Evaluator
    from konfai.metric.measure import Dice, Jaccard, MissingLabels

    output, reference = torch.ones(1, 1, 4, 4, 4), torch.ones(1, 1, 8, 8, 8)
    evaluator = SimpleNamespace(metrics={"Out": {"Ref": [MissingLabels()]}})
    assert Evaluator._voxelwise_pairs(evaluator, {"Out": output, "Ref": reference}) == {}
    reference[:, :, 4:] = 2  # a label the output lacks, outside the mask
    assert MissingLabels()(output, reference)[1] == 1.0
    mask = (reference == 1).float()  # on the reference's grid: put on the output's with the reference
    assert MissingLabels()(output, reference, mask)[1] == 0.0
    assert Dice()(output, reference, mask)[1] == Jaccard()(output, reference, mask)[1] == {1: pytest.approx(1.0)}


def test_missing_labels_is_weighed_by_its_items():
    # A logged mean weighs each batch by its item count: batches of [0, 0] and [1] missing labels report 1/3.
    from konfai.metric.measure import MissingLabels

    assert MissingLabels.batch_mean


def test_missing_labels_takes_no_loss_role():
    from konfai.metric.measure import MissingLabels
    from konfai.network.network.loaders import CriterionsAttr

    assert not MissingLabels().loss_capable
    with pytest.raises(MeasureError):
        CriterionsAttr(is_loss=True).settle_role(MissingLabels(), "k")
