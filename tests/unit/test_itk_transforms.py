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

"""Unit tests for the SimpleITK transform helpers in ``konfai.utils.ITK``."""

import numpy as np
import pytest
from konfai.utils.errors import TransformError

sitk = pytest.importorskip("SimpleITK")

from konfai.utils.ITK import _open_transform, apply_to_data_transform  # noqa: E402


def _identity_displacement_field_transform() -> "sitk.DisplacementFieldTransform":
    field = sitk.Image(4, 4, 4, sitk.sitkVectorFloat64)
    field.SetSpacing((1.0, 1.0, 1.0))
    return sitk.DisplacementFieldTransform(field)


def test_open_transform_invert_displacement_field_without_image_raises() -> None:
    """Inverting a displacement-field transform without a reference image is a typed error, not a crash."""
    transform = _identity_displacement_field_transform()
    with pytest.raises(TransformError, match="reference image"):
        _open_transform({transform: True}, image=None)


def test_open_transform_invert_displacement_field_with_image_succeeds() -> None:
    reference = sitk.Image(4, 4, 4, sitk.sitkFloat32)
    reference.SetSpacing((1.0, 1.0, 1.0))
    transform = _identity_displacement_field_transform()
    result = _open_transform({transform: True}, image=reference)
    assert len(result) == 1


def test_apply_to_data_transform_returns_ndarray() -> None:
    """apply_to_data_transform returns a numpy array (matching its annotation and callers)."""
    points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.double)
    translation = sitk.TranslationTransform(3, (10.0, 20.0, 30.0))
    result = apply_to_data_transform(points, {translation: False})
    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(result, points + np.array([10.0, 20.0, 30.0]))


def test_resample_transform_applies_displacement_in_physical_space() -> None:
    # ResampleTransform used to add the physical (dx, dy, dz) displacement straight onto a (z, y, x)
    # voxel-index grid, transposing x/z and treating millimetres as voxels. A +6 mm translation along X
    # on a 2 mm-X grid must move content 3 voxels along X (not 6 voxels along Z).
    import torch
    from konfai.data.transform import ResampleTransform
    from konfai.utils.dataset import Attribute

    volume = torch.zeros(1, 8, 8, 8, dtype=torch.uint8)
    volume[0, 4, 4, 6] = 1  # (z=4, y=4, x=6)
    attribute = Attribute()
    attribute["Origin"] = np.array([0.0, 0.0, 0.0])
    attribute["Spacing"] = np.array([2.0, 1.0, 1.0])  # (x=2 mm, y=1, z=1)
    attribute["Direction"] = np.eye(3).flatten()

    translation = sitk.TranslationTransform(3, (6.0, 0.0, 0.0))

    class _TransformStore:
        def is_dataset_exist(self, group: str, name: str) -> bool:
            return True

        def read_transform(self, group: str, name: str) -> "sitk.Transform":
            return translation

    transform = ResampleTransform({"reg": False})
    transform.datasets = [_TransformStore()]

    out = transform("case", volume, attribute)
    bright = torch.nonzero(out[0] > 0).tolist()

    assert bright == [[4, 4, 3]]  # moved 6 mm / 2 mm = 3 voxels along X, staying on z=4, y=4
