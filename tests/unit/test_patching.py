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

"""Unit tests for patch reconstruction and overlap-blending (``konfai.data.patching``)."""

import pytest
import torch
from konfai.data.patching import Accumulator, Cosinus, Gaussian, Mean
from konfai.utils.errors import PatchError
from konfai.utils.utils import get_patch_slices_from_shape


def _tile_2d(full: torch.Tensor, patch_size: list[int], overlap: int):
    """Return (patch_slices, patches) tiling the spatial dims of *full* ([B, C, H, W])."""
    patch_slices, _ = get_patch_slices_from_shape(patch_size, list(full.shape[2:]), overlap)
    patches = [full[:, :, sl[0], sl[1]].clone() for sl in patch_slices]
    return patch_slices, patches


# --------------------------------------------------------------------------------------
# Accumulator reassembly
# --------------------------------------------------------------------------------------


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA for the GPU-accumulation path")
@pytest.mark.parametrize("combine_cls", [None, Mean, Cosinus, Gaussian])
def test_accumulator_gpu_blend_matches_cpu(combine_cls):
    # The v1.6.0 GPU-accumulation feature blends patches on-device; it is only correct if the assembled
    # volume matches the CPU assembly of the same patches. Nothing else in the suite compares the two,
    # so a device-dependent blend regression would pass silently. Reassemble identical overlapping
    # patches on CPU and CUDA and require the outputs to be identical.
    torch.manual_seed(0)
    # 18 = 8 + 5 + 5 tiles exactly at step (patch-overlap)=5, so every patch is full patch_size (the
    # model always emits full-size patches; the Accumulator crops the border tail only after blending).
    full = torch.randn(2, 3, 18, 18)
    patch_size, overlap = [8, 8], 3
    patch_slices, patches = _tile_2d(full, patch_size, overlap)
    assert all(p.shape[2:] == tuple(patch_size) for p in patches), [tuple(p.shape[2:]) for p in patches]

    def assemble_on(device: str) -> torch.Tensor:
        combine = None
        if combine_cls is not None:
            combine = combine_cls()
            combine.set_patch_config(patch_size, overlap)
        acc = Accumulator(patch_slices, patch_size, patch_combine=combine, batch=True)
        for i, patch in enumerate(patches):
            acc.add_layer(i, patch.to(device))
        return acc.assemble()

    cpu = assemble_on("cpu")
    gpu = assemble_on("cuda").cpu()
    assert gpu.shape == cpu.shape
    assert torch.equal(gpu, cpu), (gpu - cpu).abs().max().item()


# --------------------------------------------------------------------------------------
# Blending windows (Mean / Cosinus)
# --------------------------------------------------------------------------------------


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


def test_path_combine_overlap_zero_uses_uniform_weights() -> None:
    """B10: overlap=0 tiles patches without overlap, so the blend window is all ones."""
    for combine_cls in (Mean, Cosinus):
        combine = combine_cls()
        combine.set_patch_config([8, 8, 8], 0)  # must not raise
        assert combine.data.shape == (8, 8, 8)
        assert torch.equal(combine.data, torch.ones(8, 8, 8))


def test_path_combine_overlap_zero_leaves_tensor_unchanged() -> None:
    combine = Mean()
    combine.set_patch_config([4, 4], 0)
    tensor = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    assert torch.equal(combine(tensor), tensor)


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


# --------------------------------------------------------------------------------------
# Overlap-blended reassembly is a partition of unity (no darkened borders)
# --------------------------------------------------------------------------------------


def test_overlap_blend_is_partition_of_unity_at_the_border() -> None:
    # 1-D volume of 20 tiled with patch 8 / overlap 2 -> patches at 0, 6, 12. The border voxels are
    # covered by a single patch, whose edge band weights ~0.5, so the pre-fix sum-without-normalise
    # reassembled them at 0.5 instead of 1.0. Dividing by the accumulated weight restores unity.
    patch_slices = [(slice(0, 8),), (slice(6, 14),), (slice(12, 20),)]
    combine = Mean()  # Cosinus needs >=2D (SimpleITK distance map), covered below.
    combine.set_patch_config([8], 2)
    accumulator = Accumulator(patch_slices, patch_size=[8], patch_combine=combine, batch=True)
    for index in range(len(patch_slices)):
        accumulator.add_layer(index, torch.ones(1, 1, 8))

    out = accumulator.assemble()[0, 0]

    assert out.shape == (20,)
    torch.testing.assert_close(out, torch.ones(20), rtol=0, atol=1e-5)


def test_overlap_blend_corner_not_quartered_in_2d() -> None:
    # A 2-D corner is covered by one patch on both axes, so the pre-fix output was ~0.25 there.
    patch_slices = [
        (slice(0, 8), slice(0, 8)),
        (slice(0, 8), slice(6, 14)),
        (slice(6, 14), slice(0, 8)),
        (slice(6, 14), slice(6, 14)),
    ]
    for combine_cls in (Mean, Cosinus):
        combine = combine_cls()
        combine.set_patch_config([8, 8], 2)
        accumulator = Accumulator(patch_slices, patch_size=[8, 8], patch_combine=combine, batch=True)
        for index in range(len(patch_slices)):
            accumulator.add_layer(index, torch.ones(1, 1, 8, 8))

        out = accumulator.assemble()[0, 0]

        assert out.shape == (14, 14)
        torch.testing.assert_close(out, torch.ones(14, 14), rtol=0, atol=1e-5)


def test_blended_reassembly_preserves_patch_dtype() -> None:
    # The weight-normalised reassembly must not promote a float16 accumulator to float32: a default
    # float32 weight_sum silently doubled the peak memory of large multi-class volumes (the 118-class
    # whole-body segmentation OOM). Many channels make the effect visible in the assembled shape.
    patch_slices = [
        (slice(0, 8), slice(0, 8)),
        (slice(0, 8), slice(6, 14)),
        (slice(6, 14), slice(0, 8)),
        (slice(6, 14), slice(6, 14)),
    ]
    combine = Cosinus()
    combine.set_patch_config([8, 8], 2)
    accumulator = Accumulator(patch_slices, patch_size=[8, 8], patch_combine=combine, batch=True)
    for index in range(len(patch_slices)):
        accumulator.add_layer(index, torch.ones(1, 5, 8, 8, dtype=torch.float16))

    out = accumulator.assemble()

    assert out.dtype == torch.float16
    torch.testing.assert_close(out[0], torch.ones(5, 14, 14, dtype=torch.float16), rtol=0, atol=1e-2)


def test_gaussian_blend_in_fp16_has_no_nan_at_single_coverage_corners() -> None:
    # The 3-D Gaussian corner weight (~7e-10 for a 16^3 patch) underflows fp16 — and the 1e-8 division
    # floor itself rounds to zero in fp16 — so corner voxels covered by a single patch reassembled as
    # 0/0 = NaN. Weights are floored at the dtype's smallest normal instead, keeping the weighted
    # average exact wherever the true weight is representable and recoverable at the corners.
    gaussian = Gaussian()
    gaussian.set_patch_config([16, 16, 16], 8)
    accumulator = Accumulator(
        [(slice(0, 16), slice(0, 16), slice(0, 16))], patch_size=[16, 16, 16], patch_combine=gaussian, batch=True
    )
    accumulator.add_layer(0, torch.full((1, 1, 16, 16, 16), 3.0, dtype=torch.float16))

    out = accumulator.assemble()[0, 0]

    assert not torch.isnan(out).any()
    # Single coverage: dividing by the accumulated weight must recover the raw value, corners included.
    torch.testing.assert_close(out.float(), torch.full((16, 16, 16), 3.0), rtol=0.02, atol=0.02)
