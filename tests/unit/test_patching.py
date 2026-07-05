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

"""Unit tests for patch reconstruction and overlap-blending helpers."""

import os

os.environ.setdefault("KONFAI_config_file", "/tmp/konfai-none.yml")
os.environ.setdefault("KONFAI_CONFIG_MODE", "Done")

import pytest  # noqa: E402
import torch  # noqa: E402

from konfai.data.patching import Accumulator, Cosinus, Gaussian, Mean  # noqa: E402
from konfai.utils.errors import PatchError  # noqa: E402
from konfai.utils.utils import get_patch_slices_from_shape  # noqa: E402


def _tile_2d(full: torch.Tensor, patch_size: list[int], overlap: int):
    """Return (patch_slices, patches) tiling the spatial dims of *full* ([B, C, H, W])."""
    patch_slices, _ = get_patch_slices_from_shape(patch_size, list(full.shape[2:]), overlap)
    patches = [full[:, :, sl[0], sl[1]].clone() for sl in patch_slices]
    return patch_slices, patches


def test_accumulator_reconstructs_non_overlapping_tiles():
    """Without blending, non-overlapping patches must reassemble exactly."""
    full = torch.arange(1 * 1 * 4 * 4, dtype=torch.float32).reshape(1, 1, 4, 4)
    patch_slices = [(slice(0, 2), slice(0, 4)), (slice(2, 4), slice(0, 4))]
    acc = Accumulator(patch_slices, [2, 4], patch_combine=None, batch=True)
    acc.add_layer(0, full[:, :, 0:2, :])
    acc.add_layer(1, full[:, :, 2:4, :])
    assert acc.is_full()
    assert torch.equal(acc.assemble(), full)


def test_accumulator_overwrites_overlap_without_combine():
    """With overlap but no blending, patches drawn from one field still reconstruct it."""
    full = torch.arange(1 * 1 * 8 * 8, dtype=torch.float32).reshape(1, 1, 8, 8)
    patch_slices, patches = _tile_2d(full, [4, 4], overlap=2)
    acc = Accumulator(patch_slices, [4, 4], patch_combine=None, batch=True)
    for i, patch in enumerate(patches):
        acc.add_layer(i, patch)
    assert torch.equal(acc.assemble(), full)


def test_accumulator_is_full_tracks_added_patches():
    patch_slices = [(slice(0, 2),), (slice(2, 4),)]
    acc = Accumulator(patch_slices, [2], patch_combine=None, batch=False)
    assert not acc.is_full()
    acc.add_layer(0, torch.zeros(1, 2))
    assert not acc.is_full()
    acc.add_layer(1, torch.zeros(1, 2))
    assert acc.is_full()


def test_assemble_without_any_patch_raises_patch_error():
    """#14: assembling an empty accumulator must raise a typed PatchError, not crash."""
    acc = Accumulator([(slice(0, 2),), (slice(2, 4),)], [2], patch_combine=None, batch=False)
    with pytest.raises(PatchError):
        acc.assemble()


def test_assemble_with_missing_first_patch_does_not_crash():
    """#14 regression: a missing index-0 patch must not raise UnboundLocalError.

    The seed tensor (shape/dtype/device) is taken from the first *present* patch,
    so any single missing patch — including index 0 — assembles cleanly.
    """
    full = torch.arange(1 * 1 * 4 * 4, dtype=torch.float32).reshape(1, 1, 4, 4)
    patch_slices = [(slice(0, 2), slice(0, 4)), (slice(2, 4), slice(0, 4))]
    acc = Accumulator(patch_slices, [2, 4], patch_combine=None, batch=True)
    # Only the second patch is added; index 0 stays None.
    acc.add_layer(1, full[:, :, 2:4, :])
    out = acc.assemble()  # must not raise
    assert out.shape == full.shape
    assert torch.equal(out[:, :, 2:4, :], full[:, :, 2:4, :])


@pytest.mark.parametrize("combine_cls", [Mean, Cosinus])
def test_path_combine_window_is_bounded_and_unit_at_center(combine_cls):
    """Blending windows weight each voxel in [0, 1] and reach 1 at the patch centre."""
    combine = combine_cls()
    combine.set_patch_config([6, 6], 2)
    window = combine.data
    assert window.shape == (6, 6)
    assert float(window.min()) >= 0.0
    assert float(window.max()) <= 1.0 + 1e-6
    assert float(window.max()) == pytest.approx(1.0, abs=1e-4)


def test_cosinus_tapers_more_than_mean_in_overlap():
    """Cosine blending must down-weight the overlap border more than uniform mean."""
    mean = Mean()
    mean.set_patch_config([6, 6], 2)
    cosinus = Cosinus()
    cosinus.set_patch_config([6, 6], 2)
    # The very first row/col sits in the overlap border where cosine tapers to ~0.
    assert float(cosinus.data[0, 0]) < float(mean.data[0, 0])


def test_path_combine_call_applies_window_and_caches_device():
    combine = Mean()
    combine.set_patch_config([6, 6], 2)
    tensor = torch.ones(1, 1, 6, 6)
    weighted = combine(tensor)
    assert torch.allclose(weighted[0, 0], combine.data)
    # The window is cached per (device, dtype) on first use and matches the tensor dtype.
    assert (tensor.device, tensor.dtype) in combine._data_per_device


def test_gaussian_window_favours_centre_and_reassembles_to_a_weighted_average():
    gaussian = Gaussian()
    gaussian.set_patch_config([8, 8], 2)
    # nnU-Net-style importance map: centre weight far exceeds the border, but the edge stays > 0.
    assert float(gaussian.data[4, 4]) > float(gaussian.data[0, 0]) > 0
    # A single patch must still reassemble to its raw values (assemble divides by the accumulated weight).
    accumulator = Accumulator([(slice(0, 8), slice(0, 8))], patch_size=[8, 8], patch_combine=gaussian, batch=True)
    accumulator.add_layer(0, torch.full((1, 1, 8, 8), 3.0))
    out = accumulator.assemble()[0, 0]
    torch.testing.assert_close(out, torch.full((8, 8), 3.0), rtol=0, atol=1e-4)
