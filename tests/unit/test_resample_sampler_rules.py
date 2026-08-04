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

"""The rules every sampler in this package obeys, pinned separately from any one sampler.

There are two gather strategies for one arithmetic. ``Resample._resample_offset_region`` maps each
axis independently, so no coordinate volume is built and one ``index_select`` per axis does the work;
``ResampleToReference._sample_at`` cannot, because a displacement is not separable, so it holds a
coordinate per voxel and gathers eight corners flat. Same rules, different loops, and the loops are
different for a measured reason.

Rules kept apart from loops is only true while something checks it. These tests are that: they assert
the RULES -- the inside interval, the tap clamp, round-half-up, the working dtype, the fill -- against
SimpleITK and against each other, so a change to one gather cannot quietly stop matching the other.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from konfai.data.transform import Resample


class _Sampler(Resample):
    """A bare handle on the sampler: these rules belong to `Resample`, not to any stage using it."""

    def __call__(self, name, tensor, cache_attribute):  # pragma: no cover - not the surface tested
        raise NotImplementedError

    def write_stream_cache_attribute(self, cache_attribute, source_spatial_shape) -> None:
        raise NotImplementedError

    def transform_shape(self, shape, cache_attribute):  # pragma: no cover
        raise NotImplementedError

    def inverse(self, name, tensor, cache_attribute):  # pragma: no cover
        raise NotImplementedError

    def patch_locality(self, cache_attribute):  # pragma: no cover
        raise NotImplementedError


_SOURCE = (12, 14, 16)
_SCALES = [1.31, 1.17, 1.23]
_OFFSETS = [0.4, -0.3, 0.2]
_TARGET = (slice(0, 8), slice(0, 9), slice(0, 10))


def _sampler(fill: float = 0.0) -> _Sampler:
    sampler = _Sampler(inverse=False)
    sampler.fill_value = fill
    return sampler


def _volume(offset: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(3)
    return (rng.random((1, *_SOURCE)) * 400 + offset).astype(np.float32)


def _offset_region(tensor: torch.Tensor, fill: float = 0.0, **overrides) -> torch.Tensor:
    arguments = {
        "target_slices": _TARGET,
        "region_starts": [0, 0, 0],
        "scales": _SCALES,
        "n_in": list(_SOURCE),
        "offsets": _OFFSETS,
    }
    arguments.update(overrides)
    return _sampler(fill)._resample_offset_region(tensor, **arguments)  # type: ignore[arg-type]


def test_the_working_dtype_rule_holds_for_a_cpu_half_volume() -> None:
    """A CPU half is accumulated in float32; the eight-corner sum is not done in half.

    torch's CPU half arithmetic is both slow and lossy over a sum of eight terms, and the values a
    microscope or a scanner produces sit exactly where float16 spacing is 2. The threshold separates
    the two: accumulating in half drifts more than twice as far from the float32 answer.
    """
    volume = _volume(offset=2050.0)  # 2050..2450, entirely above 2048, where float16 spacing is 2
    guarded = _offset_region(torch.from_numpy(volume).half()).float()
    reference = _offset_region(torch.from_numpy(volume))

    assert guarded.dtype is torch.float32
    drift = float((guarded - reference).abs().max())
    assert drift < 2.0, f"a half accumulation drifts ~4.3 on this fixture; got {drift}"


@pytest.mark.parametrize("dtype", [torch.uint8, torch.int16, torch.int32, torch.float16, torch.float32, torch.float64])
def test_a_volume_comes_back_as_the_dtype_it_went_in_as(dtype: torch.dtype) -> None:
    """The sampler computes in whatever it must and casts back once. A store's dtype is the store's."""
    volume = torch.from_numpy((_volume() % 120).astype(np.float32)).to(dtype)
    assert _offset_region(volume).dtype is dtype


def test_nearest_is_itk_round_half_up_and_not_a_size_ratio() -> None:
    """``floor(c + 0.5)``, which is a statement about a coordinate.

    ``F.interpolate``'s nearest is ``floor(o * scale)``, a statement about a size RATIO -- it says
    nothing once the target grid carries an origin of its own, which is the whole point of an offset
    map. A label map is still a label map under either rule, so only this catches it.
    """
    labels = torch.arange(int(np.prod(_SOURCE)), dtype=torch.uint8).reshape(1, *_SOURCE) % 7
    got = _offset_region(labels).numpy()[0]

    expected = np.empty_like(got)
    source = labels.numpy()[0]
    for z in range(got.shape[0]):
        for y in range(got.shape[1]):
            for x in range(got.shape[2]):
                index = [
                    int(np.floor(_SCALES[axis] * position + _OFFSETS[axis] + 0.5))
                    for axis, position in enumerate((z, y, x))
                ]
                expected[z, y, x] = source[tuple(np.clip(index, 0, np.array(_SOURCE) - 1))]
    np.testing.assert_array_equal(got, expected)


def test_inside_is_the_half_open_half_voxel_rim() -> None:
    """A sample is inside while its source index is in ``[-0.5, n - 0.5)`` -- SimpleITK's interval.

    The rim beyond the outermost voxel CENTRES is inside and reproduces the border value; a hair past
    it is fill. Getting this wrong shows as a one-voxel frame, which reads as anatomy.
    """
    volume = torch.full((1, 4, 4, 4), 5.0)
    fill = -99.0

    # A target of one voxel per axis, placed by the offset alone.
    def at(offset: float) -> float:
        one = (slice(0, 1), slice(0, 1), slice(0, 1))
        got = _offset_region(
            volume, fill=fill, target_slices=one, scales=[1.0, 1.0, 1.0], n_in=[4, 4, 4], offsets=[offset] * 3
        )
        return float(got.flatten()[0])

    assert at(-0.5) == 5.0, "the open end of the rim is inside"
    assert at(-0.5 - 1e-3) == fill, "a hair before it is not"
    assert at(3.5 - 1e-3) == 5.0, "just short of n - 0.5 is inside"
    assert at(3.5) == fill, "n - 0.5 itself is outside: the interval is half open"


def test_the_separable_sampler_matches_simpleitk() -> None:
    """The independent check. Written against SimpleITK because that is what the arithmetic claims.

    The oracle is skipped here and not at module scope: the rules above -- the working dtype, the
    inside interval, round-half-up -- are checkable without it, and a net that evaporates when an
    optional dependency is missing is the failure mode this file exists to prevent.
    """
    sitk = pytest.importorskip("SimpleITK")
    volume = _volume()
    got = _offset_region(torch.from_numpy(volume)).numpy()[0]

    image = sitk.GetImageFromArray(volume[0])
    image.SetSpacing((1.0, 1.0, 1.0))
    image.SetOrigin((0.0, 0.0, 0.0))
    grid = sitk.Image(*reversed([sl.stop - sl.start for sl in _TARGET]), sitk.sitkFloat32)
    # sitk takes geometry in (x, y, z) where the arrays above are (z, y, x).
    grid.SetSpacing(tuple(reversed(_SCALES)))
    grid.SetOrigin(tuple(reversed(_OFFSETS)))
    want = sitk.GetArrayFromImage(sitk.Resample(image, grid, sitk.Transform(), sitk.sitkLinear, 0.0))

    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-4)
