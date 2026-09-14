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

"""One ``Resample``: which grid to write on, what map to write it through, and what that fixed.

One stage, one sampler, one idea of where a voxel is: the tests here are for what is only
checkable because there is a single answer to check.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from konfai.data.transform import (
    Resample,
)
from konfai.utils.dataset import Attribute
from oracle_support import geometry

sitk = pytest.importorskip("SimpleITK")

_ORIGIN, _SPACING = [11.5, -3.25, 40.0], [1.3, 0.9, 0.9]
_SHAPE = (24, 30, 32)


def _attributes(origin=None, spacing=None, direction=None) -> Attribute:
    return geometry(_ORIGIN if origin is None else origin, _SPACING if spacing is None else spacing, direction)


def _volume(shape=_SHAPE, seed: int = 0) -> np.ndarray:
    # Noise, not a smooth field: a smooth volume resampled onto a grid that is half a voxel off is
    # still nearly right, so a smooth fixture would pass a map that is wrong by exactly the amount
    # this file exists to catch.
    return np.random.default_rng(seed).normal(size=shape).astype(np.float32)[None]


def _as_image(volume: np.ndarray, attribute: Attribute) -> sitk.Image:
    image = sitk.GetImageFromArray(volume[0])
    image.SetOrigin(attribute.get_np_array("Origin").tolist())
    image.SetSpacing(attribute.get_np_array("Spacing").tolist())
    image.SetDirection(attribute.get_np_array("Direction").tolist())
    return image


def test_an_axis_the_map_leaves_alone_is_left_alone() -> None:
    """A resample of one axis reads the other two, it does not blend them, and says so in the values.

    ``spacing: [-1, -1, 3]`` keeps x and y exactly. Blending them anyway would be two gathers and a
    lerp over the largest tensor in flight, for a result equal to the input; skipping them has to be
    exactly that, not nearly.
    """
    attribute = _attributes(origin=[0.0] * 3, spacing=[1.0, 1.0, 1.0])
    volume = torch.from_numpy(_volume())
    kept = Resample(spacing=[-1.0, -1.0, 1.0])("case", volume.clone(), Attribute(attribute))
    torch.testing.assert_close(kept, volume, rtol=0, atol=0)


def test_a_case_recorded_again_with_the_same_header_is_not_judged_again(monkeypatch) -> None:
    """Every plan records the case again; its coverage is counted once for the grid it was found on."""
    from konfai.data.transform import Resample
    from konfai.utils.dataset import Attribute

    judged = []
    count = Resample._coverage
    monkeypatch.setattr(Resample, "_coverage", classmethod(lambda cls, *args: judged.append(args) or count(*args)))
    stage = Resample(spacing=[2.0, 2.0, 2.0])
    header = {"Origin": np.zeros(3), "Spacing": np.ones(3), "Direction": np.eye(3).ravel()}
    for _ in range(3):
        assert stage.transform_shape("CT", "CASE", [8, 8, 8], Attribute(header)) == [4, 4, 4]
    assert len(judged) == 1

    moved = dict(header, Origin=np.full(3, 1.0))
    stage.transform_shape("CT", "CASE", [8, 8, 8], Attribute(moved))
    assert len(judged) == 2


def test_a_header_that_left_the_geometry_unsaid_is_not_the_identity_header_it_reads_as() -> None:
    """Both read as the identity grid, and only the second cannot place a target grid: the memo of the
    first must not answer for it."""
    from konfai.data.transform import Resample
    from konfai.utils.dataset import Attribute
    from konfai.utils.errors import TransformError

    stage = Resample(spacing=[2.0, 2.0, 2.0])
    explicit = {"Origin": np.zeros(3), "Spacing": np.ones(3), "Direction": np.eye(3).ravel()}
    stage.transform_shape("CT", "CASE", [8, 8, 8], Attribute(explicit))
    stage._grids_of("CASE")

    stage._record("CASE", [8, 8, 8], Attribute())
    with pytest.raises(TransformError):
        stage._grids_of("CASE")
