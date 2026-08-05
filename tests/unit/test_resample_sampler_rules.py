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

"""The rules the one gather obeys, pinned apart from any stage that uses it.

There is a single sampler in KonfAI — ``konfai.data.sampling.gather`` — and every resample, warp and
regrid reaches its voxels through it. That is recent: the rules used to be restated by a separable
sampler and a non-separable one, which is how two of them came to disagree about a half-voxel rim.

One implementation does not make the rules self-evident, it only makes them checkable in one place.
These tests are that place: the inside interval, the tap clamp, round-half-up, the working dtype and
the fill, asserted against SimpleITK and against hand arithmetic.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from konfai.data.sampling import gather

_SOURCE = (12, 14, 16)
_SCALES = [1.31, 1.17, 1.23]
_OFFSETS = [0.4, -0.3, 0.2]
_TARGET = (8, 9, 10)


def _coordinates(
    target_shape: tuple[int, ...] = _TARGET,
    scales: list[float] | None = None,
    offsets: list[float] | None = None,
) -> torch.Tensor:
    """One source index per target voxel for the separable map ``scale * o + offset``, per ARRAY axis.

    Separable is the easy case to write by hand, not a second code path: what the gather receives is
    always a coordinate per voxel, and how it was produced is none of its business.
    """
    scales = _SCALES if scales is None else scales
    offsets = _OFFSETS if offsets is None else offsets
    axes = [
        scales[axis] * torch.arange(extent, dtype=torch.float64) + offsets[axis]
        for axis, extent in enumerate(target_shape)
    ]
    grids = torch.meshgrid(*axes, indexing="ij")
    # The gather wants the physical components last, in (x, y, z) — the mirror of the array axes.
    return torch.stack(list(reversed(grids)), dim=-1)


def _sample(tensor: torch.Tensor, fill: float = 0.0, mode: str | None = None, **overrides) -> torch.Tensor:
    source_shape = list(overrides.pop("source_shape", _SOURCE))
    coordinates = _coordinates(**overrides)
    if mode is None:
        mode = "nearest" if tensor.dtype == torch.uint8 else "linear"
    return gather(tensor, coordinates, [0] * len(source_shape), source_shape, mode, fill)


def _volume(offset: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(3)
    return (rng.random((1, *_SOURCE)) * 400 + offset).astype(np.float32)


def test_the_working_dtype_rule_holds_for_a_cpu_half_volume() -> None:
    """A CPU half is accumulated in float32; the eight-corner sum is not done in half.

    torch's CPU half arithmetic is both slow and lossy over a sum of eight terms, and the values a
    microscope or a scanner produces sit exactly where float16 spacing is 2. The threshold separates
    the two: accumulating in half drifts more than twice as far from the float32 answer.
    """
    volume = _volume(offset=2050.0)  # 2050..2450, entirely above 2048, where float16 spacing is 2
    guarded = _sample(torch.from_numpy(volume).half()).float()
    reference = _sample(torch.from_numpy(volume))

    assert guarded.dtype is torch.float32
    drift = float((guarded - reference).abs().max())
    assert drift < 2.0, f"a half accumulation drifts ~4.3 on this fixture; got {drift}"


@pytest.mark.parametrize("dtype", [torch.uint8, torch.int16, torch.int32, torch.float16, torch.float32, torch.float64])
def test_a_volume_comes_back_as_the_dtype_it_went_in_as(dtype: torch.dtype) -> None:
    """The sampler computes in whatever it must and casts back once. A store's dtype is the store's."""
    volume = torch.from_numpy((_volume() % 120).astype(np.float32)).to(dtype)
    assert _sample(volume, mode="linear").dtype is dtype
    assert _sample(volume, mode="nearest").dtype is dtype


def test_nearest_is_itk_round_half_up_and_not_a_size_ratio() -> None:
    """``floor(c + 0.5)``, which is a statement about a coordinate.

    ``F.interpolate``'s nearest is ``floor(o * scale)``, a statement about a size RATIO -- it says
    nothing once the target grid carries an origin of its own, and it lags the LINEAR map of the same
    stage by ``(scale - 1) / 2`` source voxels, so an image and its label map resampled together come
    out shifted against each other. A label map is still a label map under either rule.
    """
    labels = torch.arange(int(np.prod(_SOURCE)), dtype=torch.uint8).reshape(1, *_SOURCE) % 7
    got = _sample(labels).numpy()[0]

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

    def at(offset: float) -> float:
        got = _sample(
            volume,
            fill=fill,
            target_shape=(1, 1, 1),
            scales=[1.0, 1.0, 1.0],
            offsets=[offset] * 3,
            source_shape=(4, 4, 4),
        )
        return float(got.flatten()[0])

    assert at(-0.5) == 5.0, "the open end of the rim is inside"
    assert at(-0.5 - 1e-3) == fill, "a hair before it is not"
    assert at(3.5 - 1e-3) == 5.0, "just short of n - 0.5 is inside"
    assert at(3.5) == fill, "n - 0.5 itself is outside: the interval is half open"


def test_the_gather_matches_simpleitk() -> None:
    """The independent check. Written against SimpleITK because that is what the arithmetic claims.

    The oracle is skipped here and not at module scope: the rules above -- the working dtype, the
    inside interval, round-half-up -- are checkable without it, and a net that evaporates when an
    optional dependency is missing is the failure mode this file exists to prevent.
    """
    sitk = pytest.importorskip("SimpleITK")
    volume = _volume()
    got = _sample(torch.from_numpy(volume)).numpy()[0]

    image = sitk.GetImageFromArray(volume[0])
    image.SetSpacing((1.0, 1.0, 1.0))
    image.SetOrigin((0.0, 0.0, 0.0))
    grid = sitk.Image(*reversed(list(_TARGET)), sitk.sitkFloat32)
    # sitk takes geometry in (x, y, z) where the arrays above are (z, y, x).
    grid.SetSpacing(tuple(reversed(_SCALES)))
    grid.SetOrigin(tuple(reversed(_OFFSETS)))
    want = sitk.GetArrayFromImage(sitk.Resample(image, grid, sitk.Transform(), sitk.sitkLinear, 0.0))

    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-4)


def test_a_region_reads_the_same_voxels_as_the_whole_volume() -> None:
    """Coordinates are GLOBAL, so handing the gather a window changes almost nothing.

    Almost, and the exception is named: a blend goes to ``grid_sample``, which takes NORMALISED
    coordinates and therefore divides by the extent of the tensor it is handed -- a window, here.
    That is the one region-local number in the path, and it is what a fused kernel costs. Every
    other part of the arithmetic is global, which is why the disagreement stays at rounding rather
    than moving a sample.
    """
    volume = torch.from_numpy(_volume())
    whole = gather(volume, _coordinates(), [0, 0, 0], list(_SOURCE), "linear", 0.0)

    start = [2, 3, 4]
    window = volume[:, start[0] :, start[1] :, start[2] :]
    partial = gather(window, _coordinates(), start, list(_SOURCE), "linear", 0.0)

    # Only where the whole-volume answer read from inside the window can the two agree; elsewhere the
    # window simply does not hold the voxels, which is the caller's contract to respect.
    reach = (slice(None), slice(3, None), slice(4, None), slice(5, None))
    span = float(whole.max() - whole.min())
    torch.testing.assert_close(partial[reach], whole[reach], rtol=0, atol=1e-5 * span)
