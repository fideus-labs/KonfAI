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
