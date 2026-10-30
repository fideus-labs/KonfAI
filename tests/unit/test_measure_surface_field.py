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

"""The surface criteria (HausdorffDistance, SurfaceDice) and the field criterion (Jacobian)."""

import numpy as np
import pytest
import torch
from konfai.metric.measure import HausdorffDistance, Jacobian, SurfaceDice
from konfai.utils.dataset import Attribute
from konfai.utils.errors import MeasureError


def _attribute(spacing, direction=None) -> Attribute:
    n = len(spacing)
    return Attribute(
        {
            "Origin": np.zeros(n),
            "Spacing": np.asarray(spacing, float),
            "Direction": np.eye(n).ravel() if direction is None else np.asarray(direction, float).ravel(),
        }
    )


def _cubes(shift: int) -> tuple[torch.Tensor, torch.Tensor]:
    """A reference cube of label 1 and the same cube moved by ``shift`` voxels along the last axis, plus a label 2
    the output lacks; ``[1, 1, 24, 24, 24]`` maps."""
    reference = torch.zeros(1, 1, 24, 24, 24, dtype=torch.int64)
    output = torch.zeros(1, 1, 24, 24, 24, dtype=torch.int64)
    reference[..., 4:12, 4:12, 4:12] = 1
    output[..., 4:12, 4:12, 4 + shift : 12 + shift] = 1
    reference[..., 16:20, 16:20, 16:20] = 2
    return output, reference


class TestSurface:
    def test_identical_masks_are_at_distance_zero_and_fully_overlap(self):
        output, reference = _cubes(0)
        _, hd = HausdorffDistance(95.0, labels=[1])(output, reference, attributes=[[_attribute([1, 1, 1])]])
        _, nsd = SurfaceDice(1.0, labels=[1])(output, reference, attributes=[[_attribute([1, 1, 1])]])
        assert hd == {1: 0.0} and nsd == {1: 1.0}

    def test_a_shift_is_measured_in_the_axis_spacing(self):
        output, reference = _cubes(3)
        # spacing (x, y, z) in the attributes; the shift is along the last array axis, x
        _, hd = HausdorffDistance(100.0, labels=[1])(output, reference, attributes=[[_attribute([0.5, 1.0, 2.0])]])
        assert hd[1] == pytest.approx(1.5)
        _, nsd = SurfaceDice(1.5, labels=[1])(output, reference, attributes=[[_attribute([0.5, 1.0, 2.0])]])
        assert nsd[1] == pytest.approx(1.0)
        _, nsd = SurfaceDice(0.1, labels=[1])(output, reference, attributes=[[_attribute([0.5, 1.0, 2.0])]])
        assert 0.0 < nsd[1] < 1.0

    def test_a_label_the_output_lacks_is_infinite_or_zero_and_one_the_reference_lacks_is_not_scored(self):
        output, reference = _cubes(0)
        loss, hd = HausdorffDistance(95.0)(output, reference, attributes=[[_attribute([1, 1, 1])]])
        assert hd[1] == 0.0 and hd[2] == float("inf") and float(loss) == 0.0  # the mean over the finite values
        _, nsd = SurfaceDice(1.0, labels=[1, 2, 3])(output, reference, attributes=[[_attribute([1, 1, 1])]])
        assert nsd[1] == 1.0 and nsd[2] == 0.0 and np.isnan(nsd[3])

    def test_a_mask_sends_the_outside_to_the_background(self):
        output, reference = _cubes(3)
        mask = torch.zeros_like(reference)
        mask[..., 16:20, 16:20, 16:20] = 1  # keeps label 2 only
        _, hd = HausdorffDistance(95.0)(output, reference, mask, attributes=[[_attribute([1, 1, 1])]])
        assert list(hd) == [2]

    def test_a_batch_scores_the_labels_of_every_item(self):
        first, reference_first = _cubes(0)
        second, reference_second = _cubes(0)
        reference_first[reference_first == 2] = 0  # only the second item's reference holds label 2
        output, reference = torch.cat([first, second]), torch.cat([reference_first, reference_second])
        attributes = [[_attribute([1, 1, 1]), _attribute([1, 1, 1])]]
        _, hd = HausdorffDistance(95.0)(output, reference, attributes=attributes)
        assert hd[1] == 0.0 and hd[2] == float("inf")

    def test_the_mean_keeps_its_fraction_on_an_integer_map(self):
        output, reference = _cubes(3)
        loss, hd = HausdorffDistance(100.0, labels=[1])(output, reference, attributes=[[_attribute([0.5, 1.0, 2.0])]])
        assert loss.dtype == torch.float32 and float(loss) == pytest.approx(hd[1]) == pytest.approx(1.5)

    def test_the_spacing_is_the_outputs_when_it_is_given(self):
        output, reference = _cubes(3)
        _, hd = HausdorffDistance(100.0, labels=[1])(
            output,
            reference,
            attributes=[[_attribute([1.0, 1.0, 1.0])]],
            output_attributes=[_attribute([0.5, 1.0, 2.0])],
        )
        assert hd[1] == pytest.approx(1.5)

    def test_two_dimensions(self):
        reference = torch.zeros(1, 1, 16, 16, dtype=torch.uint8)
        output = torch.zeros(1, 1, 16, 16, dtype=torch.uint8)
        reference[..., 2:8, 2:8] = 1
        output[..., 2:8, 4:10] = 1
        _, hd = HausdorffDistance(100.0)(output, reference, attributes=[[_attribute([1, 1])]])
        assert hd[1] == pytest.approx(2.0)

    def test_a_soft_map_is_refused(self):
        with pytest.raises(MeasureError):
            HausdorffDistance()(
                torch.zeros(1, 2, 8, 8, 8), torch.zeros(1, 1, 8, 8, 8), attributes=[[_attribute([1, 1, 1])]]
            )


def _linear_field(matrix: np.ndarray, shape=(12, 14, 16), spacing=(1.0, 1.0, 1.0)) -> torch.Tensor:
    """``u(x) = A x`` in world mm on an axis-aligned grid: ``x + u`` has the constant Jacobian ``I + A``."""
    z, y, x = np.meshgrid(*[np.arange(n) * s for n, s in zip(shape, spacing[::-1], strict=True)], indexing="ij")
    points = np.stack([x, y, z])  # world coordinates (x, y, z) per voxel
    u = np.einsum("ca,a...->c...", matrix, points)
    return torch.from_numpy(u.astype(np.float32)).unsqueeze(0)


def _ones(field: torch.Tensor) -> torch.Tensor:
    return torch.ones(1, 1, *field.shape[2:])


def _jacobian(field: torch.Tensor, *masks: torch.Tensor, attribute: Attribute | None = None) -> dict[str, float]:
    masks = masks or (_ones(field),)
    names = [""] + [f"_{index}" for index in range(1, len(masks))]
    return Jacobian(names)(field, *masks, attributes=[[attribute or _attribute([1] * (field.dim() - 2))]])[1]


class TestJacobian:
    def test_a_linear_field_has_a_constant_determinant(self):
        values = _jacobian(_linear_field(np.diag([0.1, -0.2, 0.3])))
        assert values["sdlogj"] == pytest.approx(0.0, abs=1e-6) and values["folds"] == 0.0  # float32 field

    def test_a_reflection_folds_everywhere(self):
        assert _jacobian(_linear_field(np.diag([-2.0, 0.0, 0.0])))["folds"] == 1.0  # x + u = -x: det -1

    def test_the_spacing_and_direction_are_read(self):
        # u_x = 2 i s along the index axis i, in mm: with x = i s it is 2 x (det 3), with x = -i s it is -2 x (det -1)
        field = _linear_field(np.diag([2.0, 0.0, 0.0]), spacing=(2.0, 2.0, 2.0))
        assert _jacobian(field, attribute=_attribute([2, 2, 2]))["folds"] == 0.0
        assert _jacobian(field, attribute=_attribute([2, 2, 2], direction=np.diag([-1, 1, 1])))["folds"] == 1.0
        # read with a wrong spacing the stretch is 4, not 2: still a constant determinant, still no fold
        values = _jacobian(field, attribute=_attribute([1, 1, 1]))
        assert values["sdlogj"] == pytest.approx(0.0, abs=1e-6)  # the running sums of a constant cancel to rounding

    def test_every_mask_is_reported_from_one_pass(self):
        field = _linear_field(np.diag([-2.0, 0.0, 0.0]))
        field[..., :6, :, :] = 0  # the first six planes do not move (det 1), the other six fold (det -1)
        mask = torch.zeros(1, 1, 12, 14, 16)
        mask[..., :5, :, :] = 1
        values = Jacobian(["", "_still", "_moving"])(
            field, _ones(field), mask, 1 - mask, attributes=[[_attribute([1, 1, 1])]]
        )[1]
        assert values["folds_still"] == 0.0
        assert values["folds_moving"] == pytest.approx(6 / 7)  # planes 5 to 11, of which 6 to 11 fold
        assert values["folds"] == pytest.approx(6 / 12)
        assert set(values) == {"sdlogj", "folds", "sdlogj_still", "folds_still", "sdlogj_moving", "folds_moving"}

    def test_the_log_standard_deviation_crops_two_voxels(self):
        field = torch.zeros(1, 3, 8, 8, 8)
        field[0, 2, 0, :, :] = 5.0  # u_z jumps on the first face only: inside the crop, nothing moves
        assert _jacobian(field)["sdlogj"] == pytest.approx(0.0, abs=1e-12)
        field[0, 2, 4, :, :] = 5.0  # the same jump in the middle is seen
        assert _jacobian(field)["sdlogj"] > 0.1

    def test_two_dimensions(self):
        field = torch.zeros(1, 2, 10, 10)
        field[0, 0] = torch.linspace(0, 9, 10).view(1, 10).expand(10, 10) * 0.5  # u_x = 0.5 x: det 1.5
        values = _jacobian(field)
        assert values["folds"] == 0.0 and values["sdlogj"] == pytest.approx(0.0, abs=1e-6)

    def test_what_is_refused(self):
        field = torch.zeros(1, 3, 8, 8, 8)
        with pytest.raises(MeasureError):
            Jacobian()(field, _ones(field), attributes=[[Attribute()]])  # no geometry
        with pytest.raises(MeasureError):
            Jacobian()(torch.zeros(1, 2, 8, 8, 8), _ones(field), attributes=[[_attribute([1, 1, 1])]])  # 2 of 3
        with pytest.raises(MeasureError):
            Jacobian()(field, attributes=[[_attribute([1, 1, 1])]])  # no mask
        with pytest.raises(MeasureError):
            Jacobian(["a", "b"])(field, _ones(field), attributes=[[_attribute([1, 1, 1])]])  # two names, one mask


class TestJacobianFromHaloPatches:
    """The Jacobian is reducible from patches read with two voxels of halo: the streamed sums equal the whole
    volume's, the crop at the faces included."""

    @staticmethod
    def _streamed(metric: Jacobian, tensors: list[torch.Tensor], patch: list[int], attribute: Attribute) -> dict:
        from konfai.utils.utils import get_patch_slices_from_shape

        shape = tuple(tensors[0].shape[2:])
        states = []
        for slot in get_patch_slices_from_shape(patch, list(shape), 0):
            read = tuple(
                slice(max(0, s.start - metric.halo), min(extent, s.stop + metric.halo))
                for s, extent in zip(slot, shape, strict=True)
            )
            core = tuple(slice(s.start - r.start, s.stop - r.start) for s, r in zip(slot, read, strict=True))
            states.append(
                metric.partial_metric(
                    *[t[(slice(None), slice(None), *read)] for t in tensors], core=core, attributes=[[attribute]]
                )
            )
        return metric.combine_metric(states)[1]

    def test_declares_its_crop_as_halo(self):
        assert Jacobian.reducible and Jacobian.halo == 2

    @pytest.mark.parametrize(
        ("shape", "patch"), [((19, 23, 17), [8, 8, 8]), ((19, 23, 17), [5, 23, 7]), ((37, 29), [16, 13])]
    )
    def test_halo_patches_reproduce_the_whole_volume(self, shape, patch):
        rng = np.random.default_rng(3)
        dimensions = len(shape)
        field = torch.tensor(rng.normal(size=(1, dimensions, *shape)).astype(np.float32) * 2)
        masks = [torch.ones(1, 1, *shape), torch.tensor((rng.random((1, 1, *shape)) > 0.4).astype(np.uint8))]
        attribute = _attribute([1.5, 1.0, 2.0][:dimensions])
        metric = Jacobian(["", "_mask"])
        whole = metric(field, *masks, attributes=[[attribute]])[1]
        streamed = self._streamed(metric, [field, *masks], patch, attribute)
        assert set(streamed) == set(whole)
        for key, value in whole.items():
            assert streamed[key] == pytest.approx(value, rel=1e-9, abs=1e-12), key
