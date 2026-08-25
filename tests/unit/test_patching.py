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

"""Unit tests for ``konfai.data.patching``: patch reconstruction, overlap-blending, and the
DatasetManager patch grids."""

import itertools
import math
import pickle
from typing import cast

import numpy as np
import pytest
import torch
from konfai.data.augmentation import DataAugmentationsList, Rotate
from konfai.data.materialize import CaseMaterializer
from konfai.data.patching import (
    Accumulator,
    Cosinus,
    DatasetManager,
    DatasetPatch,
    Gaussian,
    Mean,
    StreamingAccumulator,
    Trim,
    blend_axes,
    blend_overlap,
)
from konfai.utils.dataset import Dataset
from konfai.utils.errors import PatchError
from konfai.utils.utils import best_sweep_axis, get_patch_slices_from_shape, resolve_overlap


def _tile_2d(full: torch.Tensor, patch_size: list[int], overlap: int):
    """Return (patch_slices, patches) tiling the spatial dims of *full* ([B, C, H, W])."""
    patch_slices = get_patch_slices_from_shape(patch_size, list(full.shape[2:]), overlap)
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
    """#14: a missing index-0 patch must not raise UnboundLocalError.

    The seed tensor (shape/dtype/device) is taken from the first *present* patch,
    so any single missing patch (including index 0) assembles cleanly.
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
    # GPU accumulation blends patches on-device; it is only correct if the assembled
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


@pytest.mark.parametrize("combine_cls", [None, Gaussian, Trim])
@pytest.mark.parametrize("free_axis", [0, 1, 2])
def test_free_axis_reassembles_like_its_concrete_extent(free_axis, combine_cls):
    """A free (``0``) axis spans the full extent and must reassemble identically to the concrete-extent
    patch: the Accumulator must not collapse the axis to a zero-width slice, and the blend window must
    broadcast at any axis position (a *trailing* free axis is what a rank-collapsed window misaligns)."""
    torch.manual_seed(0)
    # 8 = 4 + 2 + 2 tiles at step (patch-overlap)=2, so every tiled patch is full patch_size (the model
    # emits full-size patches; the Accumulator crops the border tail only after blending).
    full = torch.rand(1, 1, 8, 8, 8)
    spatial = list(full.shape[2:])
    free = [4, 4, 4]
    free[free_axis] = 0  # one free axis (single full-extent patch); the other two tile with overlap
    concrete = [p if p > 0 else spatial[i] for i, p in enumerate(free)]
    # A free axis is a single patch spanning the extent -> no taper. The concrete reference mirrors that
    # with overlap 0 on that axis (the free version zeroes it itself via the size-1 kept entry).
    concrete_overlap = [2, 2, 2]
    concrete_overlap[free_axis] = 0

    def assemble(patch_size: list[int], overlap: int | list[int]) -> torch.Tensor:
        slices = get_patch_slices_from_shape(patch_size, spatial, overlap)
        combine = None
        if combine_cls is not None:
            combine = combine_cls()
            kept = blend_axes(patch_size)
            combine.set_patch_config(kept, blend_overlap(overlap, kept))
        acc = Accumulator(slices, patch_size, patch_combine=combine, batch=True)
        for i, sl in enumerate(slices):
            acc.add_layer(i, full[(slice(None), slice(None), *sl)].clone())
        return acc.assemble()

    out = assemble(free, 2)
    assert out.shape == full.shape, out.shape
    assert torch.equal(out, assemble(concrete, concrete_overlap))


def _axis_overlaps(slices, patch_size):
    """Per-axis overlap (patch minus tiling step) inferred from consecutive patch starts."""
    overlaps = []
    for d in range(len(patch_size)):
        starts = sorted({chunk[d].start for chunk in slices})
        step = starts[1] - starts[0] if len(starts) > 1 else patch_size[d]
        overlaps.append(patch_size[d] - step)
    return overlaps


def test_declared_free_axis_keeps_the_fraction_overlap_after_restart_concretization() -> None:
    """An OOM re-plan pins a declared free (``0``) axis to a concrete size in place; the axis must keep the
    fraction overlap default, not fall back to the fixed-patch remainder (near-zero -> seam artifacts)."""
    shape = [512, 256, 256]
    declared = [0, 128, 128]  # free axis 0
    concrete = [64, 128, 128]  # what the restart pins it to

    # Baseline: passing the declared flag explicitly must not change the non-restart result.
    declared_slices = get_patch_slices_from_shape(declared, shape, None, None, True)
    derived_slices = get_patch_slices_from_shape(declared, shape, None)
    assert _axis_overlaps(declared_slices, [512, 128, 128]) == _axis_overlaps(derived_slices, [512, 128, 128])

    # Regression: on the concretized grid the derived flag is False (no ``0`` left) and would take the
    # remainder branch; carrying declared_free_axis=True restores the exact fraction default on the tiled axes.
    remainder = get_patch_slices_from_shape(concrete, shape, None, None, False)
    fixed = get_patch_slices_from_shape(concrete, shape, None, None, True)
    assert _axis_overlaps(remainder, concrete) == [0, 0, 0]
    assert _axis_overlaps(fixed, concrete) == list(resolve_overlap(None, concrete, shape))


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
    # covered by a single patch, whose edge band weights ~0.5, so a sum without normalisation
    # reassembles them at 0.5 instead of 1.0. Dividing by the accumulated weight restores unity.
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
    # A 2-D corner is covered by one patch on both axes, so an unnormalised sum lands at ~0.25 there.
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
    # float32 weight_sum silently doubles the peak memory of large multi-class volumes (a 118-class
    # whole-body segmentation OOMs). Many channels make the effect visible in the assembled shape.
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
    # The 3-D Gaussian corner weight (~7e-10 for a 16^3 patch) underflows fp16, and the 1e-8 division
    # floor itself rounds to zero in fp16, so corner voxels covered by a single patch reassemble as
    # 0/0 = NaN. Weights must be floored at the dtype's smallest normal instead, keeping the weighted
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


# --------------------------------------------------------------------------------------
# StreamingAccumulator: slab-by-slab finalization must equal whole-volume assembly
# --------------------------------------------------------------------------------------


def _padded_patches(full: torch.Tensor, patch_slices, patch_size: list[int]) -> list[torch.Tensor]:
    """Emit model-like patches: always full patch_size, garbage in the padded out-of-volume tail."""
    patches = []
    for sl in patch_slices:
        patch = torch.randn(full.shape[0], *patch_size, dtype=full.dtype)
        extents = [min(s.stop, full.shape[dim + 1]) - s.start for dim, s in enumerate(sl)]
        source = (slice(None), *[slice(s.start, s.start + extent) for s, extent in zip(sl, extents, strict=True)])
        patch[(slice(None), *[slice(0, extent) for extent in extents])] = full[source]
        patches.append(patch)
    return patches


@pytest.mark.parametrize("combine_cls", [None, Mean, Cosinus, Gaussian])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize(
    ("shape", "patch_size", "overlap"),
    [
        ((2, 13, 9, 11), [4, 4, 4], 2),
        ((1, 7, 8), [3, 8], 1),
        ((3, 16, 8, 8), [4, 8, 8], 0),
        ((1, 5, 6, 7), [5, 6, 7], 0),
    ],
)
def test_streaming_accumulator_slabs_match_assemble(combine_cls, dtype, shape, patch_size, overlap):
    torch.manual_seed(0)
    full = torch.randn(shape, dtype=dtype)
    patch_slices = get_patch_slices_from_shape(patch_size, list(shape[1:]), overlap)
    patches = _padded_patches(full, patch_slices, patch_size)

    def _combine():
        if combine_cls is None:
            return None
        combine = combine_cls()
        combine.set_patch_config(patch_size, overlap)
        return combine

    reference = Accumulator(patch_slices, patch_size, patch_combine=_combine(), batch=False)
    streaming = StreamingAccumulator(patch_slices, patch_size, patch_combine=_combine(), batch=False)
    slabs = []
    for index, patch in enumerate(patches):
        reference.add_layer(index, patch.clone())
        slabs += streaming.add_layer(index, patch.clone())
    assert streaming.is_full()
    slabs += streaming.finalize()

    expected = reference.assemble()
    # The slab regions must tile the first spatial axis exactly, in order.
    assert slabs[0][0].start == 0
    assert slabs[-1][0].stop == shape[1]
    assert all(a.stop == b.start for (a, _), (b, _) in itertools.pairwise(slabs))
    assert all(tensor.shape[1] == region.stop - region.start for region, tensor in slabs)
    result = torch.cat([tensor for _, tensor in slabs], dim=1)
    assert torch.equal(result, expected), (result - expected).abs().max().item()


def test_streaming_accumulator_rejects_unordered_patches():
    with pytest.raises(PatchError, match="non-decreasing"):
        StreamingAccumulator([(slice(4, 8),), (slice(0, 4),)], [4], patch_combine=None, batch=False)


def test_streaming_accumulator_refuses_whole_volume_assemble():
    streaming = StreamingAccumulator([(slice(0, 4),), (slice(4, 8),)], [4], patch_combine=None, batch=False)
    streaming.add_layer(0, torch.ones(1, 4))
    with pytest.raises(PatchError, match="assemble"):
        streaming.assemble()


def test_streaming_accumulator_is_reusable_after_finalize():
    full = torch.arange(2 * 8, dtype=torch.float32).reshape(1, 2, 8)
    patch_slices = get_patch_slices_from_shape([1, 8], [2, 8], 0)
    streaming = StreamingAccumulator(patch_slices, [1, 8], patch_combine=None, batch=False)
    for _ in range(2):
        slabs = []
        for index, sl in enumerate(patch_slices):
            slabs += streaming.add_layer(index, full[(slice(None), *sl)])
        slabs += streaming.finalize()
        assert torch.equal(torch.cat([tensor for _, tensor in slabs], dim=1), full)


def test_streaming_accumulator_rejects_out_of_order_arrival():
    # Correctness needs patches to ARRIVE in non-decreasing first-axis-start order, not just the slice
    # list to be sorted. Arriving 0, 2, 3 then the skipped 1 flushes the window past start=2 before
    # patch 1 (start=2) shows up; without the guard its window offset goes negative -> silent misplace.
    patch_slices = get_patch_slices_from_shape([4, 8], [10, 8], 2)  # starts 0, 2, 4, 6; window 4
    starts = [sl[0].start for sl in patch_slices]
    assert starts == [0, 2, 4, 6]
    streaming = StreamingAccumulator(patch_slices, [4, 8], patch_combine=None, batch=False)
    streaming.add_layer(0, torch.ones(1, 4, 8))
    streaming.add_layer(1, torch.ones(1, 4, 8))
    streaming.add_layer(3, torch.ones(1, 4, 8))  # start=6 flushes the window up to 6
    with pytest.raises(PatchError, match="non-decreasing"):
        streaming.add_layer(2, torch.ones(1, 4, 8))  # start=4 < flushed=6


# --------------------------------------------------------------------------------------
# Separable blend weight
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("combine_cls", [Mean, Cosinus, Gaussian])
def test_blend_weight_factorises_into_one_vector_per_axis(combine_cls):
    """``sum_p prod_d w_d == prod_d sum_k w_d``, so the total weight is one vector per axis.

    The patch grid is a full per-axis product and the window is separable, so summing the window over
    the grid factorises exactly. That is what lets the accumulator normalise without ever holding a
    spatial-sized weight buffer.
    """
    shape, patch, overlap = [10, 12, 14], [6, 6, 6], 2
    slices = get_patch_slices_from_shape(patch, shape, overlap)
    combine = combine_cls()
    combine.set_patch_config(patch, overlap)
    accumulator = Accumulator(slices, patch, patch_combine=combine, batch=False)

    naive = torch.zeros(shape)
    for patch_slice in accumulator.patch_slices:
        dest = tuple(slice(s.start, min(s.stop, shape[d])) for d, s in enumerate(patch_slice))
        naive[dest] += combine.data[tuple(slice(0, d.stop - d.start) for d in dest)]

    _, totals = accumulator._weight_geometry()
    assert [total.shape for total in totals] == [torch.Size([extent]) for extent in shape]
    outer = totals[0][:, None, None] * totals[1][None, :, None] * totals[2][None, None, :]
    torch.testing.assert_close(outer, naive, rtol=0, atol=1e-5)


@pytest.mark.parametrize("combine_cls", [Cosinus, Gaussian])
def test_blend_recovers_the_source_in_low_precision(combine_cls):
    """A float16 blend returns the volume it was cut from, borders included.

    Each patch carries its SHARE of the weight, ``w / sum_k w``: a ratio of comparable quantities, so
    it stays in [0, 1]. The raw product underflows float16 at a tapered border (a Gaussian corner is
    ~1e-8 against a 6e-5 smallest normal), and flooring it there, as a weight accumulated in the blend
    dtype forces, leaves those voxels off by ~0.5.
    """
    shape, patch, overlap = [10, 12, 14], [6, 6, 6], 2
    full = torch.rand(2, *shape)
    slices = get_patch_slices_from_shape(patch, shape, overlap)
    combine = combine_cls()
    combine.set_patch_config(patch, overlap)
    accumulator = Accumulator(slices, patch, patch_combine=combine, batch=False)
    for index, patch_slice in enumerate(slices):
        accumulator.add_layer(index, full[(slice(None), *patch_slice)].to(torch.float16))
    torch.testing.assert_close(accumulator.assemble().float(), full, rtol=0, atol=5e-3)


@pytest.mark.parametrize("combine_cls", [Mean, Cosinus, Gaussian])
@pytest.mark.parametrize(
    ("patch", "weighted_axes"),
    [([6, 12, 14], [0]), ([1, 12, 14], []), ([10, 12, 14], [])],
)
def test_an_axis_the_grid_does_not_tile_carries_no_share(combine_cls, patch, weighted_axes):
    """One grid position on an axis makes the patch the whole of the blend weight there, so its share
    is exactly one and the blend skips it: same values, one pass over the patch fewer. A grid that
    tiles a single voxel at a time, or nothing at all, leaves nothing to weight and no staging buffer.
    """
    shape, overlap = [10, 12, 14], 2
    full = torch.rand(2, *shape)
    slices = get_patch_slices_from_shape(patch, shape, overlap)
    combine = combine_cls()
    axes = blend_axes(patch)
    combine.set_patch_config(axes, blend_overlap(overlap, axes))
    accumulator = Accumulator(slices, patch, patch_combine=combine, batch=False)
    for index, patch_slice in enumerate(slices):
        block = full[(slice(None), *patch_slice)]
        # The loader hands the patch over with its singleton axes dropped; the accumulator puts them back.
        tiled = [extent for extent in block.shape[1:] if extent > 1]
        accumulator.add_layer(index, block.reshape(block.shape[0], *tiled))

    # The share is keyed by the patch's own extent, and the blend asks for it at full rank.
    probe = torch.zeros(2, *[min(size, extent) for size, extent in zip(patch, shape, strict=True)])
    weighted = [dim for dim in range(3) if accumulator._share(dim, 0, probe) is not None]
    assert weighted == weighted_axes
    assert (accumulator._weighted is not None) == bool(weighted_axes)
    torch.testing.assert_close(accumulator.assemble(), full, rtol=0, atol=1e-6)


# --------------------------------------------------------------------------------------
# Reassembly contract: per-axis overlap, arrival order, unweighted semantics
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("combine_cls", [Cosinus, Gaussian])
def test_anisotropic_overlap_reassembles_exactly(combine_cls):
    """A different overlap per axis still reconstructs the source.

    The window tapers over each axis's own overlap, so an axis that does not tile (overlap 0) must
    contribute a flat factor rather than the taper of its neighbours.
    """
    # 12=2x6, 14=6+8, 16=4+4+8: every patch is full patch_size (the model emits full-size patches)
    shape, patch, overlap = [12, 14, 16], [6, 8, 8], [0, 2, 4]
    full = torch.rand(2, *shape)
    slices = get_patch_slices_from_shape(patch, shape, overlap)
    combine = combine_cls()
    combine.set_patch_config(patch, overlap)
    accumulator = Accumulator(slices, patch, patch_combine=combine, batch=False)
    for index, patch_slice in enumerate(slices):
        accumulator.add_layer(index, full[(slice(None), *patch_slice)])
    torch.testing.assert_close(accumulator.assemble(), full, rtol=0, atol=1e-5)


@pytest.mark.parametrize("combine_cls", [Mean, Cosinus, Gaussian])
def test_blend_does_not_depend_on_patch_arrival_order(combine_cls):
    """A weighted blend is a sum, so the assembled volume does not depend on the order patches arrive.

    Only the whole-volume accumulator: StreamingAccumulator requires non-decreasing starts by
    construction and rejects anything else.
    """
    shape, patch, overlap = [10, 10, 14], [6, 6, 6], 2  # exact tiling at step 4
    full = torch.rand(2, *shape)
    slices = get_patch_slices_from_shape(patch, shape, overlap)

    def assemble(order: list[int]) -> torch.Tensor:
        combine = combine_cls()
        combine.set_patch_config(patch, overlap)
        accumulator = Accumulator(slices, patch, patch_combine=combine, batch=False)
        for index in order:
            accumulator.add_layer(index, full[(slice(None), *slices[index])])
        return accumulator.assemble()

    forward = list(range(len(slices)))
    torch.testing.assert_close(assemble(forward), assemble(forward[::-1]), rtol=0, atol=1e-6)


def test_unweighted_overlap_is_last_write_wins():
    """Without a combine, an overlapped voxel keeps the LAST patch that covered it.

    Pinned because it is a choice, not a consequence: the winning patch may hold that voxel on its
    own border, with no context behind it, which is what shows up as a seam.
    """
    shape, patch, overlap = [8], [4], 2
    slices = get_patch_slices_from_shape(patch, shape, overlap)
    accumulator = Accumulator(slices, patch, patch_combine=None, batch=False)
    for index, _ in enumerate(slices):
        accumulator.add_layer(index, torch.full((1, 4), float(index)))
    out = accumulator.assemble()[0]
    # starts 0, 2, 4 -> the last patch covering each voxel wins, so the volume reads 0,0,1,1,2,2,2,2
    assert out.tolist() == [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0]


def test_trim_selects_the_most_central_patch():
    """Trim keeps the patch that holds a voxel most centrally, not the last one that wrote it.

    Volume 8, patch 4, overlap 2 -> starts 0, 2, 4. The kept bands are [0,3), [3,5), [5,8): they tile
    the axis, the first and last open to the edge, and every interior voxel sits at least one row
    inside the patch it came from. Compare test_unweighted_overlap_is_last_write_wins.
    """
    slices = get_patch_slices_from_shape([4], [8], 2)
    combine = Trim()
    combine.set_patch_config([4], 2)
    accumulator = Accumulator(slices, [4], patch_combine=combine, batch=False)
    for index, _ in enumerate(slices):
        accumulator.add_layer(index, torch.full((1, 4), float(index)))
    assert accumulator.assemble()[0].tolist() == [0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 2.0]


@pytest.mark.parametrize(
    "shape, patch, overlap",
    [
        ([10, 10, 14], [6, 6, 6], 2),  # every patch reaches full patch_size
        ([9, 11], [6, 6], 4),  # the last patch of each axis is truncated AND needs its tail opened
    ],
)
def test_trim_reassembles_exactly_and_keeps_values_discrete(shape, patch, overlap):
    """The kept bands tile every axis, so the weights sum to one: the volume comes back untouched.

    Values are never averaged, which is what lets a label map survive reassembly. A truncated last
    patch is what Trim is most fragile to: a weighting always has a strictly positive total, so a
    coverage gap only darkens a voxel, where a 0/1 selection leaves it unwritten.
    """
    labels = torch.randint(0, 5, (2, *shape)).float()
    slices = get_patch_slices_from_shape(patch, shape, overlap)
    combine = Trim()
    combine.set_patch_config(patch, overlap)
    accumulator = Accumulator(slices, patch, patch_combine=combine, batch=False)
    for index, patch_slice in enumerate(slices):
        accumulator.add_layer(index, labels[(slice(None), *patch_slice)])
    out = accumulator.assemble()
    assert torch.equal(out, labels)
    assert torch.equal(out, out.round())


@pytest.mark.parametrize(
    "shape, patch, overlap",
    [
        ([8], [4], 2),  # 1-D, exact tiling
        ([10, 12, 14], [6, 6, 6], 2),  # 3-D, no axis divisible by the stride
        ([12, 14, 16], [6, 8, 8], [0, 2, 4]),  # a different overlap per axis, including none
        ([5, 14, 14], [8, 8, 8], 4),  # patch larger than the volume on axis 0
        ([16, 16], [8, 8], 0),  # no overlap at all
        ([10, 10], [10, 10], 2),  # a single patch covering everything
        ([9, 11], [4, 4], 3),  # odd overlap: the two halves must still abut
        ([9, 11], [6, 6], 4),  # truncated last patch whose tail opening is what closes the volume
    ],
)
def test_trim_kept_boxes_partition_the_volume(shape, patch, overlap):
    """Every voxel is written exactly once: no gap, no overlap.

    This is the property the selection path rests on: if the kept boxes tiled imperfectly, assembly
    would leave holes (never written) or race (written twice), and neither shows up as an error.
    """
    slices = get_patch_slices_from_shape(patch, shape, overlap)
    combine = Trim()
    combine.set_patch_config(patch, overlap)
    accumulator = Accumulator(slices, patch, patch_combine=combine, batch=False)

    hits = torch.zeros(shape, dtype=torch.int32)
    for patch_slice in accumulator.patch_slices:
        dest = [slice(s.start, min(s.stop, shape[d])) for d, s in enumerate(patch_slice)]
        box = accumulator._kept_box(patch_slice)
        hits[tuple(slice(d.start + b.start, d.start + b.stop) for d, b in zip(dest, box, strict=True))] += 1

    assert int(hits.min()) == 1, f"{int((hits == 0).sum())} voxel(s) written by no patch"
    assert int(hits.max()) == 1, f"{int((hits > 1).sum())} voxel(s) written by more than one patch"


def test_trim_kept_boxes_are_cached_per_axis_position_not_per_patch():
    """A kept box is the product of one span per axis, so the cache holds one span per (axis,
    start): sum(n_d) entries for prod(n_d) patches, and the grid's starts are read off the slices
    once. Keyed per patch, every patch missed and each miss re-sorted the starts of every patch:
    O(P^2) over a case (15.6 s for 18,000 thin 2.5D patches, 16 ms here, measured).

    The boxes themselves come from the same window nonzero run, checked against a derivation that
    reads the windows position by position.
    """
    shape, patch, overlap = [20, 22, 24], [8, 8, 8], 4
    slices = get_patch_slices_from_shape(patch, shape, overlap)
    combine = Trim()
    combine.set_patch_config(patch, overlap)
    accumulator = Accumulator(slices, patch, patch_combine=combine, batch=False)

    starts = [sorted({s[dim].start for s in accumulator.patch_slices}) for dim in range(len(shape))]
    expected = {}
    for dim, axis_starts in enumerate(starts):
        for position, start in enumerate(axis_starts):
            ones = combine.window(dim, position, len(axis_starts)).nonzero().flatten()
            expected[dim, start] = slice(int(ones[0]), int(ones[-1]) + 1)
    boxes = [accumulator._kept_box(patch_slice) for patch_slice in accumulator.patch_slices]

    assert boxes == [tuple(expected[dim, s.start] for dim, s in enumerate(p)) for p in accumulator.patch_slices]
    assert len(accumulator._kept) == sum(len(axis_starts) for axis_starts in starts) < len(slices)
    assert [list(positions) for positions in accumulator._positions()] == starts


def test_trim_keeps_the_patch_whole_when_there_is_nothing_to_trim():
    """An overlap at least as wide as the patch leaves no central band; keep the patch instead.

    Trimming both sides would give an empty window, and a patch that keeps nothing has no box to
    write: the box derivation would fail on an empty ``nonzero()``.
    """
    for size, overlap in ((4, 4), (3, 4), (1, 2), (2, 3)):
        assert torch.equal(Trim()._window_1d(size, overlap), torch.ones(size)), (size, overlap)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.int16, torch.uint8])
def test_border_patches_pad_with_the_patch_minimum_kept_on_the_device(dtype):
    """Under ``pad_value=None`` a border patch pads with its own minimum (a uint8 map with zero).

    The minimum is filled from the 0-d device tensor, never read back through ``.item()``, which
    drained the CUDA stream once per padded patch: 37 of the 64 patches of a 100^3 case at 32^3,
    measured, and none now. The values are those of the ``.item()`` spelling to the bit, which is
    the oracle here (a padded patch keeps every band, on every axis that pads).
    """
    torch.manual_seed(0)
    shape = [7, 9, 10]
    data = (torch.randn(2, *shape) * 40).to(dtype)
    patch = DatasetPatch(patch_size=[4, 4, 4], overlap=0)
    assert patch.pad_value is None
    patch.load(shape)

    padded = 0
    for index in range(patch.get_size()):
        plan = patch.get_read_plan(list(data.shape), index, 0, True)
        window = data[plan.data_slices]
        pad_with = 0 if dtype is torch.uint8 else float(window.min().item())
        oracle = torch.nn.functional.pad(window, plan.constant_padding, "constant", pad_with)
        padded += any(plan.constant_padding)
        assert torch.equal(patch.get_data(data, index, 0, True), oracle)
    assert padded > 0


@pytest.mark.parametrize("sweep_axis", [0, 1, 2])
def test_streaming_accumulator_sweeps_any_axis(sweep_axis):
    """The window slides along the declared axis, and the volume comes back the same whichever it is.

    The window costs ``patch[axis] x (the other extents)``, so the axis is worth choosing rather than
    fixing: on an anisotropic volume the cheapest sweep can be several times smaller than axis 0. This
    pins that the machinery is axis-agnostic; picking the axis is a separate decision.
    """
    shape, patch, overlap = [10, 10, 14], [6, 6, 6], 2
    full = torch.rand(2, *shape)
    slices = get_patch_slices_from_shape(patch, shape, overlap)
    # The grid is emitted with axis 0 outermost; a sweep along another axis needs that axis outermost.
    rest = [axis for axis in range(3) if axis != sweep_axis]
    ordered = sorted(slices, key=lambda sl: tuple(sl[axis].start for axis in (sweep_axis, *rest)))

    combine = Cosinus()
    combine.set_patch_config(patch, overlap)
    accumulator = StreamingAccumulator(ordered, patch, patch_combine=combine, batch=False, sweep_axis=sweep_axis)
    assert accumulator.footprint_shape[sweep_axis] == 6  # the window, not the extent
    assert accumulator.footprint_shape == [6 if a == sweep_axis else shape[a] for a in range(3)]

    out = torch.zeros(2, *shape)
    finalized = []
    for index, patch_slice in enumerate(ordered):
        finalized += accumulator.add_layer(index, full[(slice(None), *patch_slice)])
    finalized += accumulator.finalize()
    for region, slab in finalized:
        destination = [slice(None)] * 3
        destination[sweep_axis] = region
        out[(slice(None), *destination)] = slab
    torch.testing.assert_close(out, full, rtol=0, atol=1e-5)


@pytest.mark.parametrize("sweep_axis", [0, 1, 2])
def test_patch_grid_orders_by_the_sweep_axis(sweep_axis):
    """The grid holds the same patches whatever the sweep axis; only the order changes.

    The order is what the reassembly window relies on: it slides along that axis, so starts on it must
    never decrease. Emitting a different SET of patches would silently change what is predicted, which
    is why the set is asserted and not just the ordering.
    """
    shape, patch, overlap = [10, 12, 14], [6, 6, 6], 2
    reference = get_patch_slices_from_shape(patch, shape, overlap)
    ordered = get_patch_slices_from_shape(patch, shape, overlap, sweep_axis=sweep_axis)

    assert sorted(map(str, ordered)) == sorted(map(str, reference)), "the grid itself must not change"
    starts = [patch_slice[sweep_axis].start for patch_slice in ordered]
    assert starts == sorted(starts), f"starts on axis {sweep_axis} must not decrease: {starts}"
    if sweep_axis == 0:
        assert ordered == reference, "axis 0 must reproduce the historical order exactly"


@pytest.mark.parametrize(
    "shape, patch, expected",
    [
        ([256, 512, 640], [128, 128, 128], 2),  # the patch divides the last axis most
        ([295, 259, 219], [96, 128, 160], 0),  # axis 0 is already the cheapest here
        ([64, 64, 64], [32, 32, 32], 0),  # a tie falls back to the first axis
        ([100, 20, 20], [10, 20, 20], 0),  # an axis the patch spans whole is the worst sweep
        ([100, 40, 20], [0, 20, 20], 1),  # a free axis spans the extent, so it never wins
    ],
)
def test_best_sweep_axis_is_the_smallest_window(shape, patch, expected):
    """The window costs ``min(patch, extent) x the other extents``; the axis is picked to minimise it.

    Not a preference: on a 256x512x640 volume it is the difference between holding 0.47 GiB and 0.19,
    for the same patches read in the same total time.
    """
    assert best_sweep_axis(patch, shape) == expected
    windows = [
        min(patch[axis] or shape[axis], shape[axis]) * math.prod(shape[d] for d in range(3) if d != axis)
        for axis in range(3)
    ]
    assert windows[best_sweep_axis(patch, shape)] == min(windows)


# DatasetManager construction, one augmentation draw per case, shared by every group
# --------------------------------------------------------------------------------------


def test_two_groups_share_one_construction_draw(streaming_dataset_stub) -> None:
    """Building the label group's manager must reuse the image group's draw, not redraw over it.

    A quarter Rotate transposes per-copy extents, so a per-group redraw leaves the two groups with
    different copy grids: crashing the streamed read of the stale grid's last patch.
    """
    volume = np.zeros((1, 6, 8, 10), dtype=np.float32)
    for seed in range(10):
        torch.manual_seed(seed)
        augmentations = DataAugmentationsList(nb=2, data_augmentations={})
        rotate = Rotate(is_quarter=True)
        rotate.load(1.0)
        augmentations.data_augmentations = [rotate]
        image_manager, label_manager = (
            DatasetManager(
                index=0,
                group_src=group,
                group_dest=group,
                name="CASE_000",
                dataset=cast(Dataset, streaming_dataset_stub(volume)),
                patch=DatasetPatch([4, 4, 4]),
                transforms=[],
                data_augmentations_list=[augmentations],
            )
            for group in ("CT", "SEG")
        )
        assert image_manager.shapes == label_manager.shapes, f"seed {seed}"


def test_chain_device_is_opt_in_and_cpu_is_a_no_op(streaming_dataset_stub) -> None:
    """This same machinery loads training cases inside DataLoader workers: only an explicit
    non-CPU device handed to materialize() may ever move the chain."""
    volume = np.zeros((1, 4, 6, 8), dtype=np.float32)
    manager = DatasetManager(
        index=0,
        group_src="CT",
        group_dest="CT",
        name="CASE_000",
        dataset=cast(Dataset, streaming_dataset_stub(volume)),
        patch=None,
        transforms=[],
        data_augmentations_list=[],
    )
    assert manager._chain_device is None
    CaseMaterializer(manager).materialize(rewrite=True, device=torch.device("cpu"))
    assert manager._chain_device is None


def test_chain_device_lives_only_for_the_materialize_call(streaming_dataset_stub, monkeypatch) -> None:
    """The rank's device routes the chain during the call only: one that outlived it would move a
    later get_data() onto CUDA inside a DataLoader worker."""
    volume = np.zeros((1, 4, 6, 8), dtype=np.float32)
    manager = DatasetManager(
        index=0,
        group_src="CT",
        group_dest="CT",
        name="CASE_000",
        dataset=cast(Dataset, streaming_dataset_stub(volume)),
        patch=None,
        transforms=[],
        data_augmentations_list=[],
    )
    seen: list[torch.device | None] = []
    original = manager._stream_ready

    def spy(*args, **kwargs):
        seen.append(manager._chain_device)
        return original(*args, **kwargs)

    monkeypatch.setattr(manager, "_stream_ready", spy)
    CaseMaterializer(manager).materialize(rewrite=True, device=torch.device("cuda", 0))
    assert seen == [torch.device("cuda", 0)]
    assert manager._chain_device is None


# --------------------------------------------------------------------------------------
# Patch pickling: a rank cuts the grids, it is not shipped them
# --------------------------------------------------------------------------------------


def _multi_copy_patch(patch_size: list[int], shapes: list[list[int]]) -> DatasetPatch:
    """A patch loaded for one copy per shape, carrying the flags its cut and its reads depend on."""
    patch = DatasetPatch(patch_size=list(patch_size))
    patch.pad_to_patch = False
    patch.halo = 3
    patch.free_axis_multiple = [8, 8, 8]
    for a, shape in enumerate(shapes):
        patch.load(shape, a)
    return patch


def test_a_pickled_patch_does_not_carry_its_grids() -> None:
    """mp.spawn pickles the configured object with every manager's patch, and the per-case grids
    dominated those bytes: 100 synthetic cases of 2048 patches shipped 3.69 MB, 47 kB once the cuts
    stay home (32 patches each: 139 kB -> 47 kB)."""
    patch = _multi_copy_patch([16, 16, 16], [[64, 64, 64]])

    assert len(pickle.dumps(patch)) < len(pickle.dumps(patch.get_patch_slices(0)))


@pytest.mark.parametrize(
    "patch_size, pinned",
    [
        ([16, 16, 16], None),
        ([0, 16, 16], None),
        ([1, 8, 8], None),
        ([0, 16, 16], [40, 16, 16]),  # an OOM re-plan pinned the declared free axis in place
    ],
)
def test_an_unpickled_patch_cuts_back_the_same_grids(patch_size, pinned) -> None:
    """The rebuilt grid is the shipped grid: same slices in the same order, same sweep axis, same
    reads. The cut is a pure function of the recorded shape and the configuration, so every input
    it reads must survive the pickle: the halo and ``pad_to_patch`` a reducing consumer sets, the
    model's free-axis multiple, and the DECLARED free axis a restart's concretization erases from
    ``patch_size`` (without it the axis falls back to the fixed-patch remainder overlap)."""
    shapes = [[40, 48, 56], [48, 40, 56], [56, 48, 40]]
    patch = _multi_copy_patch(patch_size, shapes)
    if pinned is not None:
        patch.patch_size[:] = pinned
        for a, shape in enumerate(shapes):
            patch.load(shape, a)
    reference = {a: (patch.get_sweep_axis(a), patch.get_patch_slices(a)) for a in range(len(shapes))}

    restored = pickle.loads(pickle.dumps(patch))

    assert restored._grids == {}, "the cuts must not travel"
    assert (restored.pad_to_patch, restored.halo, restored.free_axis_multiple) == (False, 3, [8, 8, 8])
    assert restored._declared_free_axis == patch._declared_free_axis
    for a, shape in enumerate(shapes):
        assert (restored.get_sweep_axis(a), restored.get_patch_slices(a)) == reference[a]
        assert restored.get_size(a) == patch.get_size(a)
        indices = range(patch.get_size(a))
        assert [restored.read_slices(a, i, shape) for i in indices] == [patch.read_slices(a, i, shape) for i in indices]
        assert [restored.core_in_read(a, i) for i in indices] == [patch.core_in_read(a, i) for i in indices]


def test_reloading_a_copy_recuts_its_grid() -> None:
    """``load`` is the only thing that changes a copy's grid, so a second call must not hand back
    the first one's cut (a model patch re-loads per forward, on the shape it is given)."""
    patch = _multi_copy_patch([16, 16, 16], [[64, 64, 64]])
    first = patch.get_patch_slices(0)

    patch.load([32, 32, 32], 0)

    assert patch.get_patch_slices(0) != first
    assert patch.get_patch_slices(0) == _multi_copy_patch([16, 16, 16], [[32, 32, 32]]).get_patch_slices(0)


def test_a_pickled_manager_hands_every_copy_the_grid_it_was_counted_on(streaming_dataset_stub) -> None:
    """The loader mapping shipped to a rank is ``(case, copy, patch)`` triples counted on the
    parent's grids: a rank that cut different ones would index patches that are not there.

    A quarter ``Rotate`` transposes per-copy extents, so the copies are cut on three different
    grids and a shape confused between them shows up at once.
    """
    torch.manual_seed(0)
    augmentations = DataAugmentationsList(nb=2, data_augmentations={})
    rotate = Rotate(is_quarter=True)
    rotate.load(1.0)
    augmentations.data_augmentations = [rotate]
    manager = DatasetManager(
        index=0,
        group_src="CT",
        group_dest="CT",
        name="CASE_000",
        dataset=cast(Dataset, streaming_dataset_stub(np.zeros((1, 6, 8, 10), dtype=np.float32))),
        patch=DatasetPatch([4, 4, 4]),
        transforms=[],
        data_augmentations_list=[augmentations],
    )
    copies = range(len(manager.shapes))
    reference = {a: (manager.patch.get_sweep_axis(a), manager.patch.get_patch_slices(a)) for a in copies}

    restored = pickle.loads(pickle.dumps(manager))

    assert restored.shapes == manager.shapes
    for a in copies:
        assert (restored.patch.get_sweep_axis(a), restored.patch.get_patch_slices(a)) == reference[a]
