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

"""The read mirror of the write-side patch-streaming dispatcher: ``DatasetManager`` patch reads.

A transform chain streams patches straight from disk when every stage declares its patch locality
(``patch_locality``) and, for region stages, how a target patch pulls back through it
(``stream_region_source``): the planner folds the region stages into one bounded read, runs the
chain on that window, seeds GLOBAL_STAT stages from the stored statistics, and refuses the chains
those declarations cannot honour. Streaming is an optimisation, so every streamed patch must equal
the whole-volume pass sliced on the same grid. These tests prove that per region kind and in
composition, for augmented copies and across epoch re-draws, and pin the planner's classification
and refusals.
"""

from typing import cast

import numpy as np
import pytest
import torch
from konfai.data.augmentation import DataAugmentationsList
from konfai.data.augmentation import Flip as FlipAugmentation
from konfai.data.patching import DatasetManager, DatasetPatch
from konfai.data.transform import (
    Canonical,
    Clip,
    Dilate,
    Flip,
    Gradient,
    LocalityKind,
    Normalize,
    PatchLocality,
    Permute,
    ResampleToShape,
    Softmax,
    TensorCast,
    Transform,
)
from konfai.utils.dataset import Attribute, Dataset
from konfai.utils.errors import PatchError


@pytest.fixture
def build_streaming_manager(streaming_dataset_stub):
    """Factory for a ``DatasetManager`` reading ``volume`` through the in-memory streaming stub."""

    def build(
        volume: np.ndarray,
        transforms: list[Transform],
        patch_size: list[int],
        augmentations: list[DataAugmentationsList] | tuple = (),
    ) -> DatasetManager:
        return DatasetManager(
            index=0,
            group_src="CT",
            group_dest="CT",
            name="CASE_000",
            dataset=cast(Dataset, streaming_dataset_stub(volume)),
            patch=DatasetPatch(list(patch_size)),
            transforms=list(transforms),
            data_augmentations_list=list(augmentations),
        )

    return build


@pytest.fixture
def assert_stream_matches_whole_volume(build_streaming_manager):
    """Every streamed patch must equal the whole-volume pass sliced on the same grid."""

    def check(
        volume: np.ndarray,
        transforms: list[Transform],
        patch_size: list[int],
        *,
        atol: float = 0.0,
    ) -> DatasetManager:
        manager = build_streaming_manager(volume, transforms, patch_size)
        assert manager.can_stream_patch(0)

        size = manager.patch.get_size(0)
        streamed = [manager._get_streamed_data(index, 0, True)[0] for index in range(size)]

        reference_tensor = torch.from_numpy(volume.copy())
        reference_attribute = Attribute()
        for transform in transforms:
            reference_tensor = transform("CASE_000", reference_tensor, reference_attribute)
        reference = [manager.patch.get_data(reference_tensor, index, 0, True) for index in range(size)]

        assert len(streamed) == len(reference) == size
        for got, expected in zip(streamed, reference, strict=False):
            assert got.shape == expected.shape
            if atol == 0.0:
                assert torch.equal(got, expected)
            else:
                np.testing.assert_allclose(got.numpy(), expected.numpy(), atol=atol)
        return manager

    return check


def test_stream_halo_dilate_seam_matches_whole_volume(assert_stream_matches_whole_volume) -> None:
    # Foreground straddling the patch boundary at column 4: a whole-volume dilation spreads across the
    # seam, so a correct HALO read + crop must reproduce it patch-for-patch.
    volume = np.zeros((1, 8, 8), dtype=np.float32)
    volume[0, 3:5, 3:5] = 1.0
    manager = assert_stream_matches_whole_volume(volume, [Dilate(2)], [4, 4])
    # The dispatcher must actually take the region (HALO) path, not fall back.
    assert manager._resolve_patch_stream_source(0, True).region_index == 0


def test_stream_halo_gradient_seam_matches_whole_volume(assert_stream_matches_whole_volume) -> None:
    rng = np.random.default_rng(0)
    volume = rng.standard_normal((1, 8, 8)).astype(np.float32)
    assert_stream_matches_whole_volume(volume, [Gradient()], [4, 4], atol=1e-6)


def test_stream_orientation_flip_remap_matches_whole_volume(assert_stream_matches_whole_volume) -> None:
    volume = np.arange(1 * 8 * 8, dtype=np.float32).reshape(1, 8, 8)
    assert_stream_matches_whole_volume(volume, [Flip("0|1")], [4, 4])


def test_stream_orientation_permute_remap_matches_whole_volume(assert_stream_matches_whole_volume) -> None:
    volume = np.arange(1 * 8 * 8, dtype=np.float32).reshape(1, 8, 8)
    manager = assert_stream_matches_whole_volume(volume, [Permute("1|0")], [4, 4])
    assert manager._resolve_patch_stream_source(0, True).region_index == 0


def test_stream_orientation_permute_border_patch_uses_the_permuted_grid(assert_stream_matches_whole_volume) -> None:
    # Permute swaps the spatial axes, so the patch grid is cut on the PERMUTED extents (7x8, not 8x7).
    # The last patch of the 7-long target axis is one voxel short of patch_size: the streamed patch must
    # be padded against that target grid, not against the source shape, which would leave it short.
    volume = np.arange(1 * 8 * 7, dtype=np.float32).reshape(1, 8, 7)
    assert_stream_matches_whole_volume(volume, [Permute("1|0")], [4, 4])


def test_stream_pointwise_border_patch_pads_after_the_chain(assert_stream_matches_whole_volume) -> None:
    # A 9-long axis leaves the last patch one voxel short of patch_size, so the read plan pads it up.
    # The whole-volume path transforms the volume and only then pads (with the min of the TRANSFORMED
    # patch), so the streamed path must apply the read plan after its chain too -- padding the raw patch
    # first pads in the source domain and then runs the transform over the padding.
    volume = np.arange(3 * 8 * 9, dtype=np.float32).reshape(3, 8, 9)
    assert_stream_matches_whole_volume(volume, [Softmax(0)], [4, 4])


def test_stream_global_stat_before_orientation_region_matches_whole_volume(build_streaming_manager) -> None:
    # GLOBAL_STAT (Normalize, seeded from disk stats) as a pre-pointwise stage in front of an
    # ORIENTATION region transform: both the stat and the remap must compose byte-identically.
    volume = np.arange(1 * 8 * 8, dtype=np.float32).reshape(1, 8, 8)
    transforms: list[Transform] = [Normalize(), Flip("0")]
    manager = build_streaming_manager(volume, transforms, [4, 4])
    assert manager.can_stream_patch(0)
    region_index = manager._resolve_patch_stream_source(0, True).region_index
    assert region_index == 1  # the Flip, not the Normalize

    size = manager.patch.get_size(0)
    streamed = [manager._get_streamed_data(index, 0, True)[0] for index in range(size)]

    minimum = float(volume.min())
    maximum = float(volume.max())
    normalized = torch.from_numpy((2 * (volume - minimum) / (maximum - minimum) - 1).astype(np.float32)).flip(1)
    reference = [manager.patch.get_data(normalized, index, 0, True) for index in range(size)]
    for got, expected in zip(streamed, reference, strict=False):
        np.testing.assert_allclose(got.numpy(), expected.numpy(), atol=1e-6)


def test_stream_pointwise_chain_matches_whole_volume(assert_stream_matches_whole_volume) -> None:
    # A trailing chain of purely POINTWISE transforms streams the exact patch (region_index is None).
    volume = np.arange(1 * 8 * 8, dtype=np.float32).reshape(1, 8, 8)
    manager = assert_stream_matches_whole_volume(volume, [TensorCast("float32"), Flip("0")], [4, 4])
    assert manager._resolve_patch_stream_source(0, True).region_index == 1


def test_stream_composed_orientation_and_halo_matches_whole_volume(assert_stream_matches_whole_volume) -> None:
    # Region stages compose: the dilation's halo pulls through the flip's mirror, so one bounded read
    # serves both and the seam-spreading foreground must still agree bit for bit.
    volume = np.zeros((1, 8, 8), dtype=np.float32)
    volume[0, 3:5, 3:5] = 1.0
    manager = assert_stream_matches_whole_volume(volume, [Flip("0"), Dilate(1)], [4, 4])
    plans = manager._resolve_patch_stream_source(0, True).stage_plans
    assert [plan.kind.value for plan in plans] == ["orientation", "halo"]


def test_stream_composed_rescale_and_orientation_matches_whole_volume(assert_stream_matches_whole_volume) -> None:
    # A resample followed by a flip: the flip's mirror region pulls through the resample's scale
    # window, on the RESAMPLED grid the fold computed between them.
    rng = np.random.default_rng(7)
    volume = (rng.standard_normal((1, 8, 8)).astype(np.float32)) * 100.0
    manager = assert_stream_matches_whole_volume(
        volume, [ResampleToShape(shape=[12, 12]), Flip("0")], [4, 4], atol=1e-3
    )
    plans = manager._resolve_patch_stream_source(0, True).stage_plans
    assert [plan.kind.value for plan in plans] == ["rescale", "orientation"]
    assert tuple(plans[1].in_shape) == (12, 12)


def test_stream_composed_triple_region_chain_matches_whole_volume(assert_stream_matches_whole_volume) -> None:
    # Three region stages in one chain — flip, resample, permute — folded into one bounded read.
    rng = np.random.default_rng(11)
    volume = (rng.standard_normal((1, 8, 6)).astype(np.float32)) * 100.0
    manager = assert_stream_matches_whole_volume(
        volume, [Flip("0"), ResampleToShape(shape=[12, 9]), Permute("1|0")], [4, 4], atol=1e-3
    )
    plans = manager._resolve_patch_stream_source(0, True).stage_plans
    assert [plan.kind.value for plan in plans] == ["orientation", "rescale", "orientation"]
    assert tuple(plans[2].out_shape) == (9, 12)


def test_stream_composed_orientations_with_pointwise_between_match_whole_volume(
    assert_stream_matches_whole_volume,
) -> None:
    # Two orientations with a pointwise stage between them: the fold carries the permuted extents and
    # the value map rides along where the regions put it.
    volume = np.arange(1 * 8 * 6, dtype=np.float32).reshape(1, 8, 6)
    manager = assert_stream_matches_whole_volume(
        volume, [Flip("0"), Clip(min_value=-10.0, max_value=10.0), Permute("1|0")], [4, 4]
    )
    plans = manager._resolve_patch_stream_source(0, True).stage_plans
    assert [plan.kind.value for plan in plans] == ["orientation", "pointwise", "orientation"]
    assert tuple(plans[2].out_shape) == (6, 8)


def test_softmax_channel_axis_is_pointwise_but_spatial_axis_falls_back(
    assert_stream_matches_whole_volume, build_streaming_manager
) -> None:
    # A channel-axis softmax (dim 0) is spatially pointwise and streams the exact patch. A softmax over
    # a SPATIAL axis normalises across the whole extent, so a per-patch softmax would diverge: the
    # contract must declare it WHOLE_VOLUME and the dispatcher must refuse to stream it.
    assert Softmax(0).patch_locality(Attribute()).kind is LocalityKind.POINTWISE
    assert Softmax(1).patch_locality(Attribute()).kind is LocalityKind.WHOLE_VOLUME
    assert Softmax(-1).patch_locality(Attribute()).kind is LocalityKind.WHOLE_VOLUME

    volume = np.arange(3 * 8 * 8, dtype=np.float32).reshape(3, 8, 8)
    assert_stream_matches_whole_volume(volume, [Softmax(0)], [4, 4], atol=1e-6)

    spatial_manager = build_streaming_manager(volume, [Softmax(1)], [4, 4])
    assert spatial_manager.can_stream_patch(0) is False


def test_stream_clip_fixed_bounds_is_pointwise_and_matches_whole_volume(assert_stream_matches_whole_volume) -> None:
    # Fixed float bounds clip each voxel independently: POINTWISE, exact patch, no region transform.
    rng = np.random.default_rng(0)
    volume = (rng.standard_normal((1, 8, 8)).astype(np.float32)) * 100.0
    assert Clip(min_value=-50.0, max_value=50.0).patch_locality(Attribute()).kind is LocalityKind.POINTWISE
    manager = assert_stream_matches_whole_volume(volume, [Clip(min_value=-50.0, max_value=50.0)], [4, 4])
    assert manager._resolve_patch_stream_source(0, True).region_index is None


def test_stream_clip_min_max_is_global_stat_and_matches_whole_volume(streaming_dataset_stub) -> None:
    # 'min'/'max' bounds clip to the volume extremum -- a no-op on that bound -- so the streamed
    # per-patch result is byte-identical to the whole-volume pass, and the dispatcher seeds the
    # global stat from a single read_data_statistics call instead of loading the full volume.
    rng = np.random.default_rng(1)
    volume = (rng.standard_normal((1, 8, 8)).astype(np.float32)) * 100.0
    assert Clip(min_value="min", max_value="max").patch_locality(Attribute()).kind is LocalityKind.GLOBAL_STAT

    stub = streaming_dataset_stub(volume)
    manager = DatasetManager(
        index=0,
        group_src="CT",
        group_dest="CT",
        name="CASE_000",
        dataset=cast(Dataset, stub),
        patch=DatasetPatch([4, 4]),
        transforms=[Clip(min_value="min", max_value="max")],
        data_augmentations_list=[],
    )
    assert manager.can_stream_patch(0)  # planning reads the stat once
    size = manager.patch.get_size(0)
    streamed = [manager._get_streamed_data(index, 0, True)[0] for index in range(size)]

    reference_tensor = Clip(min_value="min", max_value="max")("CASE_000", torch.from_numpy(volume.copy()), Attribute())
    reference = [manager.patch.get_data(reference_tensor, index, 0, True) for index in range(size)]
    for got, expected in zip(streamed, reference, strict=False):
        assert torch.equal(got, expected)

    assert stub.stats_reads == 1  # global stat seeded once from disk, never a full-volume load
    assert stub.full_reads == 0


def test_global_stat_after_float_cast_still_streams_and_matches(build_streaming_manager) -> None:
    """A value-preserving cast ahead of a GLOBAL_STAT stage must not block streaming.

    Casting uint8 to float moves no values, so the stored statistics are still Normalize's own
    input: [TensorCast -> float, Normalize] streams and equals the whole-volume path.
    """
    volume = np.arange(1 * 8 * 8 * 8, dtype=np.uint8).reshape(1, 8, 8, 8)
    transforms: list[Transform] = [TensorCast("float32"), Normalize()]
    streamed = build_streaming_manager(volume, transforms, [4, 4, 4])
    reference = build_streaming_manager(volume, transforms, [4, 4, 4])
    reference.load(transforms, [])

    assert streamed.can_stream_patch(0)
    for index in range(streamed.get_size(0)):
        got = streamed.get_data(index, 0, [], True)
        expected = reference.get_data(index, 0, [], True)
        torch.testing.assert_close(got, expected, rtol=0, atol=0)


def test_clip_percentile_and_mask_bounds_fall_back_to_whole_volume(build_streaming_manager) -> None:
    # A percentile bound needs the whole histogram and a mask reads a second full volume: both
    # genuinely require the whole volume, so the contract declares WHOLE_VOLUME and streaming is off.
    assert (
        Clip(min_value="percentile:1", max_value="percentile:99").patch_locality(Attribute()).kind
        is LocalityKind.WHOLE_VOLUME
    )
    assert Clip(mask="SEG").patch_locality(Attribute()).kind is LocalityKind.WHOLE_VOLUME
    volume = np.arange(1 * 8 * 8, dtype=np.float32).reshape(1, 8, 8)
    manager = build_streaming_manager(volume, [Clip(min_value="percentile:1", max_value="percentile:99")], [4, 4])
    assert not manager.can_stream_patch(0)


def test_stream_orientation_border_patch_is_padded_to_patch_size(assert_stream_matches_whole_volume) -> None:
    # A tiling whose last patch is narrower than patch_size (30 with patch 8 -> border width 6): the
    # whole-volume Patch.get_data pads that border up to patch_size, so the region streamed path must
    # too, otherwise the border patch comes out one-or-more voxels short and cannot batch/reassemble.
    volume = np.arange(1 * 30 * 30, dtype=np.float32).reshape(1, 30, 30)
    manager = assert_stream_matches_whole_volume(volume, [Flip("0|1")], [8, 8])
    assert manager._resolve_patch_stream_source(0, True).region_index == 0
    # Every streamed patch is exactly patch_size, including the borders.
    size = manager.patch.get_size(0)
    for index in range(size):
        assert tuple(manager._get_streamed_data(index, 0, True)[0].shape) == (1, 8, 8)


def test_stream_resample_border_patch_matches_padded_whole_volume(build_streaming_manager) -> None:
    # RESCALE upsample to a grid that tiles unevenly (30 with patch 8 -> border width 6). The whole-
    # volume path resamples the whole volume then pads border patches to patch_size; the streamed
    # resample path must reproduce that padding so border patches are shape- and value-consistent.
    rng = np.random.default_rng(3)
    volume = (rng.standard_normal((1, 20, 20, 20)).astype(np.float32)) * 100.0
    shape = [30, 30, 30]
    patch = [8, 8, 8]

    stream_manager = build_streaming_manager(volume, [ResampleToShape(shape=shape)], patch)
    assert stream_manager.can_stream_patch(0)
    assert stream_manager._resolve_patch_stream_source(0, True).region_index == 0

    reference_manager = build_streaming_manager(volume, [ResampleToShape(shape=shape)], patch)
    reference_manager.load(reference_manager.transforms, [], load_augmentations=False)

    size = stream_manager.patch.get_size(0)
    streamed = [stream_manager._get_streamed_data(index, 0, True)[0] for index in range(size)]
    reference = [reference_manager.patch.get_data(reference_manager.data[0], index, 0, True) for index in range(size)]

    assert len(streamed) == len(reference) == size
    for got, expected in zip(streamed, reference, strict=False):
        assert tuple(got.shape) == tuple(expected.shape) == (1, 8, 8, 8)
        # Interior values match F.interpolate to float32 interpolation-rounding; the border
        # patch is padded to patch_size and byte-consistent in shape.
        np.testing.assert_allclose(got.numpy(), expected.numpy(), atol=1e-3)


def test_stream_resample_nearest_strong_downsampling_matches_whole_volume(build_streaming_manager) -> None:
    # Strong downsampling of a uint8 label map (nearest mode): the nearest voxel of the first output
    # column (floor(o*scale)) falls BELOW the linear tap window's start, so the source read must widen
    # to include it -- otherwise the gather indexes a negative local offset and wraps onto the far
    # edge, silently returning a wrong label. A regular ratio (a plain integer scale) hides the bug;
    # 40 -> 6 (scale 6.67) exposes the sub-pixel offset that pushes the linear start past voxel 0.
    volume = (np.arange(1 * 40 * 40).reshape(1, 40, 40) % 7).astype(np.uint8)
    shape = [6, 6]
    patch = [3, 3]
    stream_manager = build_streaming_manager(volume, [ResampleToShape(shape=shape)], patch)
    assert stream_manager.can_stream_patch(0)

    reference_manager = build_streaming_manager(volume, [ResampleToShape(shape=shape)], patch)
    reference_manager.load(reference_manager.transforms, [], load_augmentations=False)

    size = stream_manager.patch.get_size(0)
    for index in range(size):
        got = stream_manager._get_streamed_data(index, 0, True)[0]
        expected = reference_manager.patch.get_data(reference_manager.data[0], index, 0, True)
        # Nearest is a pure gather: the streamed patch must equal the whole-volume pick bit for bit.
        assert torch.equal(got, expected)


@pytest.mark.parametrize(("n_in", "n_out"), [(5, 3), (7, 3), (10, 7), (3, 7), (4, 6), (8, 8)])
def test_streamed_nearest_resample_matches_whole_volume_at_any_ratio(n_in: int, n_out: int) -> None:
    """The streamed nearest gather must pick the same source voxel as F.interpolate, per axis."""
    volume = (torch.arange(n_in**3, dtype=torch.int32) % 251).to(torch.uint8).reshape(1, n_in, n_in, n_in)
    resample = ResampleToShape(shape=[n_out, n_out, n_out], inverse=False)
    attribute = Attribute()
    attribute["Spacing"] = np.ones(3)
    expected = resample("case", volume.clone(), Attribute(attribute))

    target = tuple(slice(0, n_out) for _ in range(3))
    slices, starts, scales, n_in_list, _ = resample.resample_source_region(target, [n_in] * 3, Attribute(attribute))
    got = resample.resample_region(volume[(slice(None), *slices)], target, starts, scales, n_in_list)
    torch.testing.assert_close(got, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.uint8])
def test_streamed_resample_handles_2d(dtype: torch.dtype) -> None:
    """resample_region must not assume three spatial axes."""
    volume = (torch.arange(1 * 9 * 11, dtype=torch.float32).reshape(1, 9, 11) % 17).to(dtype)
    resample = ResampleToShape(shape=[5, 6], inverse=False)
    attribute = Attribute()
    attribute["Spacing"] = np.ones(2)
    expected = resample("case", volume.clone(), Attribute(attribute))

    target = (slice(0, 5), slice(0, 6))
    slices, starts, scales, n_in_list, _ = resample.resample_source_region(target, [9, 11], Attribute(attribute))
    got = resample.resample_region(volume[(slice(None), *slices)], target, starts, scales, n_in_list)
    torch.testing.assert_close(got, expected, rtol=0, atol=1e-5)


# --------------------------------------------------------------------------------------
# Augmented copies and epoch re-draws stream through the same planner.
# --------------------------------------------------------------------------------------


def _flip_augmentations() -> list[DataAugmentationsList]:
    augmentations = DataAugmentationsList(nb=1, data_augmentations={})
    flip = FlipAugmentation(f_prob=[1.0, 1.0, 1.0])
    flip.load(1.0)
    augmentations.data_augmentations = [flip]
    return [augmentations]


def test_augmented_copy_consumes_the_volume_statistic(build_streaming_manager) -> None:
    """Copy a=1, requested before copy 0 ever streams, still normalizes by the VOLUME's Min/Max."""
    torch.manual_seed(0)
    volume = np.arange(1 * 8 * 8 * 8, dtype=np.float32).reshape(1, 8, 8, 8)
    streamed = build_streaming_manager(volume, [Normalize()], [4, 4, 4], _flip_augmentations())
    reference = build_streaming_manager(volume, [Normalize()], [4, 4, 4], streamed.data_augmentations_list)
    reference.load([Normalize()], reference.data_augmentations_list)

    assert streamed.can_stream_patch(1)
    for index in range(streamed.get_size(1)):
        got = streamed.get_data(index, 1, [], True)
        expected = reference.get_data(index, 1, [], True)
        torch.testing.assert_close(got, expected, rtol=0, atol=1e-6)


@pytest.mark.parametrize("transform_case", ["resample", "canonical"])
def test_replanning_after_epoch_redraw_keeps_the_stored_geometry(build_streaming_manager, transform_case: str) -> None:
    """Epoch 2 must stream the same bytes as epoch 1, with no attribute stack growth.

    Replanning must not read the live case attribute -- it already carries epoch 1's target
    Spacing/Direction -- or a streamed Resample degrades to identity and a streamed Canonical stops
    reorienting from the second epoch on, while the geometry keys stack once more per epoch.
    """
    volume = np.arange(1 * 8 * 8 * 8, dtype=np.float32).reshape(1, 8, 8, 8)
    transform = ResampleToShape(shape=[16, 16, 16]) if transform_case == "resample" else Canonical()
    streamed = build_streaming_manager(volume, [transform], [4, 4, 4], _flip_augmentations())

    assert streamed.can_stream_patch(0)
    first_epoch = [streamed.get_data(index, 0, [], True) for index in range(streamed.get_size(0))]
    keys_after_first = list(streamed.cache_attributes[0].keys())

    streamed.reset_augmentation(reset_state=False)
    assert streamed.can_stream_patch(0)
    second_epoch = [streamed.get_data(index, 0, [], True) for index in range(streamed.get_size(0))]
    keys_after_second = list(streamed.cache_attributes[0].keys())

    for got, expected in zip(second_epoch, first_epoch, strict=True):
        torch.testing.assert_close(got, expected, rtol=0, atol=0)
    assert keys_after_second == keys_after_first


# --------------------------------------------------------------------------------------
# A stage whose declarations the planner cannot honour must be refused, loudly.
# --------------------------------------------------------------------------------------


class _RecordsOnlyInCall(Transform):
    """A region transform recording geometry where a streamed patch throws it away."""

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        return PatchLocality(LocalityKind.ORIENTATION)

    def stream_region_source(
        self, target_slices: tuple[slice, ...], source_spatial_shape: list[int], cache_attribute: Attribute
    ) -> list[slice]:
        return [
            slice(extent - t.stop, extent - t.start)
            for t, extent in zip(target_slices, source_spatial_shape, strict=False)
        ]

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        # Mirroring moves the near corner, and this is the only place it says so.
        cache_attribute["Origin"] = np.asarray([0.0, 0.0, 7.0])
        return tensor.flip(tuple(range(1, tensor.dim())))


def test_a_region_stage_recording_geometry_nowhere_the_case_reads_is_refused(build_streaming_manager) -> None:
    # A declaration this framework cannot honour must fail where it is made, not persist the geometry
    # of the volume as stored and call the run correct.
    volume = np.arange(1 * 8 * 8 * 8, dtype=np.float32).reshape(1, 8, 8, 8)
    manager = build_streaming_manager(volume, [_RecordsOnlyInCall()], [4, 4, 4])
    with pytest.raises(PatchError) as error:
        manager.get_data(0, 0, [], True)
    assert "write_stream_cache_attribute" in str(error.value)
