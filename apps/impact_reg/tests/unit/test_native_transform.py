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

"""A registration's native form: elastix's rigid then B-spline written as such where it is the result, in
millimetres, and measured without reference on the fixed grid."""

import json
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk
import torch
from impact_reg_konfai.impact_reg import ImpactRegKonfAIApp
from impact_reg_konfai.models.elastix_engine import with_shift
from impact_reg_konfai.models.intensity import EngineRegistration, world_transform
from konfai.utils.dataset import AFFINE_ATTRIBUTE, TRANSFORM_FILE_ATTRIBUTE, Attribute, image_to_data


def _grid(spacing=(2.0, 2.0, 2.0)) -> sitk.Image:
    grid = sitk.Image([20, 18, 16], sitk.sitkFloat32)
    grid.SetSpacing(spacing)
    grid.SetOrigin((-10.0, 5.0, 3.0))
    return grid


def _composite(grid: sitk.Image, scale: float = 1.0) -> sitk.Transform:
    rigid = sitk.Euler3DTransform([v / scale for v in (5.0, 15.0, 20.0)], 0.05, -0.03, 0.08, (1.0, -0.8, 1.5))
    bspline = sitk.BSplineTransformInitializer(grid, [4, 4, 3], 3)
    bspline.SetParameters(np.random.default_rng(0).normal(0, 0.8, len(bspline.GetParameters())).tolist())
    return sitk.CompositeTransform([rigid, bspline])


def test_a_transform_found_in_voxels_is_written_in_millimetres() -> None:
    """The engines work in the fixed image's finest voxel: the native transform is ``voxel * T(p / voxel)``, its
    rigid part's translation and its B-spline's grid and coefficients scaled."""
    voxel = 2.0
    in_voxels = _grid((1.0, 1.0, 1.0))
    transform = _composite(in_voxels, voxel)
    shifted = with_shift(transform, [1.0, -0.5, 2.0])  # a composite in a composite, as elastix's comes
    for found in (transform, shifted):
        world = world_transform(found, voxel)
        for point in ([0.0, 20.0, 15.0], [12.0, 30.0, 25.0], [-6.0, 14.0, 9.0]):
            want = voxel * np.asarray(found.TransformPoint([v / voxel for v in point]))
            np.testing.assert_allclose(world.TransformPoint(point), want, atol=1e-9)


def test_the_start_is_added_back_to_a_native_transform() -> None:
    """elastix registered a moving image moved by its start: a moving point is T(x) + shift."""
    transform = _composite(_grid())
    shifted = with_shift(transform, [3.0, -2.0, 1.5])
    point = [0.0, 20.0, 15.0]
    np.testing.assert_allclose(shifted.TransformPoint(point), np.add(transform.TransformPoint(point), [3, -2, 1.5]))


class _Engine:
    def __init__(self, result) -> None:
        self.result = result

    def register(self, fixed, moving, device, fixed_mask, moving_mask):
        return self.result


def _forward(engine_result) -> tuple[np.ndarray, Attribute]:
    grid = _grid()
    data, attributes = image_to_data(grid)
    tensor = torch.from_numpy(data).unsqueeze(0)
    branches = [[Attribute(attributes)] for _ in range(4)]
    field = EngineRegistration(_Engine(engine_result))(tensor, tensor, tensor + 1, tensor + 1, branches)
    return field[0].numpy(), branches[0][0]


def test_the_registration_names_its_native_transform_beside_its_field() -> None:
    """A B-spline found in voxels: the header names a transform file holding it in millimetres, which the output
    writes in place of the field the run still carries; a linear one is an affine, as before."""
    voxel = 2.0
    transform = _composite(_grid((1.0, 1.0, 1.0)), voxel)
    field, header = _forward(transform)
    assert TRANSFORM_FILE_ATTRIBUTE in header and AFFINE_ATTRIBUTE not in header
    native = sitk.ReadTransform(str(header[TRANSFORM_FILE_ATTRIBUTE]))
    point = [0.0, 20.0, 15.0]
    np.testing.assert_allclose(native.TransformPoint(point), world_transform(transform, voxel).TransformPoint(point))
    Path(str(header[TRANSFORM_FILE_ATTRIBUTE])).unlink()
    assert field.shape == (3, 16, 18, 20)

    _, header = _forward(sitk.Euler3DTransform())
    assert AFFINE_ATTRIBUTE in header and TRANSFORM_FILE_ATTRIBUTE not in header


def test_a_b_spline_is_measured_without_reference_on_the_fixed_grid(tmp_path: Path) -> None:
    """Its displacement sampled on the fixed image's grid, as SimpleITK samples it: its lengths are that field's."""
    grid = _grid()
    image = sitk.GetImageFromArray(np.random.default_rng(1).normal(size=(16, 18, 20)).astype(np.float32))
    image.CopyInformation(grid)
    sitk.WriteImage(image, str(tmp_path / "fixed.mha"))
    transform = _composite(grid)
    sitk.WriteTransform(transform, str(tmp_path / "Transform.h5"))

    ImpactRegKonfAIApp().evaluate(
        fixed_images=[tmp_path / "fixed.mha"],
        moving_images=[tmp_path / "fixed.mha"],
        transforms=[tmp_path / "Transform.h5"],
        output=tmp_path / "out",
        cpu=1,
        quiet=True,
        tmp_dir=tmp_path / "work",
    )

    summary = json.loads((tmp_path / "out" / "Evaluation_summary.json").read_text())["case"]
    field = sitk.TransformToDisplacementField(
        transform, sitk.sitkVectorFloat32, grid.GetSize(), grid.GetOrigin(), grid.GetSpacing(), grid.GetDirection()
    )
    lengths = np.linalg.norm(sitk.GetArrayFromImage(field), axis=-1)
    got = {key.split(":")[-1]: values["P000"] for key, values in summary.items()}
    assert got["DisplacementMean"] == pytest.approx(lengths.mean(), rel=1e-4)
    assert "JacobianFolding" in got and "BendingEnergy" in got
