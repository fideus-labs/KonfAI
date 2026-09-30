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

import numpy as np
import pytest
import SimpleITK as sitk
from impact_reg_konfai.models.orientation import world_aligned, world_aligned_pair


def _image(direction, size=(20, 16, 12), spacing=(1.0, 1.5, 2.0)) -> sitk.Image:
    zz, yy, xx = np.meshgrid(*[np.linspace(0, 1, n) for n in size[::-1]], indexing="ij")
    image = sitk.GetImageFromArray((np.sin(3 * xx) + np.cos(2 * yy) + zz).astype(np.float32))
    image.SetSpacing(spacing)
    image.SetOrigin((-7.0, 3.0, 11.0))
    image.SetDirection([float(v) for v in np.asarray(direction).ravel()])
    return image


def _back_on_the_source_grid(source: sitk.Image, aligned: sitk.Image, interpolator: int) -> np.ndarray:
    """The world-aligned copy sampled at the source's own voxels, as an array in the source's voxel order."""
    identity = sitk.Transform(3, sitk.sitkIdentity)
    return sitk.GetArrayFromImage(sitk.Resample(aligned, source, identity, interpolator, float("nan")))


def test_an_identity_direction_is_left_as_it_is() -> None:
    image = _image(np.eye(3))
    assert world_aligned(image) is image


def test_a_flipped_and_permuted_image_is_reoriented_without_resampling() -> None:
    image = _image([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
    aligned = world_aligned(image)
    assert aligned.GetDirection() == (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    after = _back_on_the_source_grid(image, aligned, sitk.sitkNearestNeighbor)
    assert np.array_equal(sitk.GetArrayFromImage(image), after)


def test_an_oblique_permuted_image_is_reordered_without_resampling_and_keeps_its_obliquity() -> None:
    rotation = sitk.Euler3DTransform()
    rotation.SetRotation(np.deg2rad(5), 0.0, np.deg2rad(10))
    permuted = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]]) @ np.array(rotation.GetMatrix()).reshape(3, 3)
    image = _image(permuted)
    aligned = world_aligned(image)
    direction = np.array(aligned.GetDirection()).reshape(3, 3)
    assert (np.abs(direction).argmax(axis=0) == [0, 1, 2]).all() and (np.diag(direction) > 0).all()
    reorder = np.linalg.inv(permuted) @ direction  # the new direction is the old one times a signed permutation
    assert np.allclose(reorder, np.round(reorder), atol=1e-6) and np.allclose(np.abs(reorder).sum(0), 1)
    after = _back_on_the_source_grid(image, aligned, sitk.sitkNearestNeighbor)
    assert np.array_equal(sitk.GetArrayFromImage(image), after)  # the same voxels, only reordered


def test_a_mask_gets_the_same_voxel_order_as_its_image_despite_a_rounded_header() -> None:
    image = _image([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
    mask = sitk.Cast(image > 1.0, sitk.sitkUInt8)
    mask.SetDirection([v + 1e-7 for v in mask.GetDirection()])  # the rounding of KonfAI's Attribute round-trip
    fixed, _, fixed_mask, moving_mask = world_aligned_pair(image, image, mask, None)
    assert fixed_mask.GetSize() == fixed.GetSize() and moving_mask is None


@pytest.mark.parametrize("module_name", ["elastix_engine", "fireants"])
def test_the_feature_engines_use_it(module_name: str) -> None:
    import importlib

    module = importlib.import_module(f"impact_reg_konfai.models.{module_name}")
    assert module.world_aligned_pair is world_aligned_pair
