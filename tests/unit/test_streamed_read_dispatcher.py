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
those declarations cannot honour. The generic per-stage guarantee (every built-in transform and
augmentation, streamed patch == whole-volume pass on the same grid) is enumerated and proven in
``test_transform_locality_contract.py``; this file keeps the dispatcher-specific properties: how
the fold composes region stages into one plan, the planner's classification and state (epoch
re-draws, augmented copies, statistics seeded from disk instead of a full read), its refusals, and
the streamed-resample region primitives the plan is built from.
"""

from typing import cast

import numpy as np
import pytest
import torch
from konfai.data import patching
from konfai.data.augmentation import DataAugmentationsList
from konfai.data.augmentation import Flip as FlipAugmentation
from konfai.data.materialize import CaseMaterializer
from konfai.data.patching import DatasetManager, DatasetPatch
from konfai.data.transform import (
    Canonical,
    Clip,
    Dilate,
    Flip,
    LocalityKind,
    Normalize,
    PatchLocality,
    Permute,
    RegionContext,
    Resample,
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
    manager = assert_stream_matches_whole_volume(volume, [Resample(shape=[12, 12]), Flip("0")], [4, 4], atol=1e-3)
    plans = manager._resolve_patch_stream_source(0, True).stage_plans
    assert [plan.kind.value for plan in plans] == ["regrid", "orientation"]
    assert tuple(plans[1].in_shape) == (12, 12)


def test_stream_composed_triple_region_chain_matches_whole_volume(assert_stream_matches_whole_volume) -> None:
    # Three region stages in one chain (flip, resample, permute) folded into one bounded read.
    rng = np.random.default_rng(11)
    volume = (rng.standard_normal((1, 8, 6)).astype(np.float32)) * 100.0
    manager = assert_stream_matches_whole_volume(
        volume, [Flip("0"), Resample(shape=[12, 9]), Permute("1|0")], [4, 4], atol=1e-3
    )
    plans = manager._resolve_patch_stream_source(0, True).stage_plans
    assert [plan.kind.value for plan in plans] == ["orientation", "regrid", "orientation"]
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


def _geometry_manager(
    stub_class,
    volume: np.ndarray,
    transforms: list[Transform],
    patch: DatasetPatch | None,
    spacing: np.ndarray,
    direction: np.ndarray,
) -> DatasetManager:
    """A manager over the streaming stub with a REAL header: the identity geometry the stub answers
    would make every landing fold below trivially right."""

    class _WithGeometry(stub_class):
        def _attributes(self) -> Attribute:
            attribute = Attribute()
            attribute["Origin"] = np.zeros(volume.ndim - 1)
            attribute["Spacing"] = np.asarray(spacing, dtype=np.float64)
            attribute["Direction"] = np.asarray(direction, dtype=np.float64).flatten()
            return attribute

    return DatasetManager(
        index=0,
        group_src="CT",
        group_dest="CT",
        name="CASE_000",
        dataset=cast(Dataset, _WithGeometry(volume)),
        patch=patch,
        transforms=list(transforms),
        data_augmentations_list=[],
    )


def _fresh_chain_reference(volume: np.ndarray, transforms: list[Transform], attribute: Attribute) -> torch.Tensor:
    """The chain run stage by stage on the live header: the semantics every route must reproduce."""
    reference = torch.from_numpy(volume.copy())
    for stage in transforms:
        reference = stage("CASE_000", reference, attribute)
    return reference


def test_a_resample_behind_a_canonical_lands_on_the_reoriented_grid(streaming_dataset_stub) -> None:
    """The landing fold evolves the case state, so a Resample is judged on what Canonical left.

    The regression this pins: the fold used to hand every stage the STORED header, so the Resample
    recorded the pre-Canonical grid: the whole-volume path then resampled the wrong axis (silently:
    every voxel real, the anatomy at the wrong density), and the patched routes crashed or refused
    with the blame on the stage. The chain is the shipped TotalSegmentator prediction prefix.
    """
    rng = np.random.default_rng(3)
    volume = (rng.standard_normal((1, 9, 10, 11)).astype(np.float32)) * 100.0
    # Direction = canonical @ (x<->z swap): Canonical reorients by a signed permutation, after which
    # the 2.0 mm axis is x. Resampling to 1.5 iso must therefore widen x: (11, 10, 9*2/1.5=12).
    direction = np.diag([-1.0, -1.0, 1.0]) @ np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    spacing = np.asarray([1.5, 1.5, 2.0])

    def chain() -> list[Transform]:
        return [Canonical(), Resample(spacing=[1.5, 1.5, 1.5])]

    whole = _geometry_manager(streaming_dataset_stub, volume, chain(), None, spacing, direction)
    assert whole.shapes[0] == [11, 10, 12]
    whole.load(whole.transforms, [])
    reference = _fresh_chain_reference(volume, chain(), whole.dataset._attributes())
    assert list(reference.shape) == [1, 11, 10, 12]
    assert torch.equal(whole.data[0], reference)

    patched = _geometry_manager(streaming_dataset_stub, volume, chain(), DatasetPatch([4, 4, 4]), spacing, direction)
    assert patched.can_stream_patch(0), patched.stream_refusal(0)
    size = patched.patch.get_size(0)
    for index in range(size):
        streamed = patched._get_streamed_data(index, 0, True)[0]
        # Canonical is an index remap and an axis-aligned spacing change reads one axis at a time on
        # global coordinates: both routes are bit-identical to the whole volume, so no tolerance.
        assert torch.equal(streamed, patched.patch.get_data(reference, index, 0, True))


def test_a_second_resample_reads_the_first_ones_grid(streaming_dataset_stub) -> None:
    """[Resample(3), Resample(1.5)] downsamples then upsamples: the second stage is not a no-op.

    The regression this pins: both stages used to record their grid from the stored header, so the
    second saw the ORIGINAL spacing, concluded nothing changes, and handed its input through: the
    run then wrote a volume at half the asked density with a header claiming otherwise.
    """
    rng = np.random.default_rng(5)
    volume = (rng.standard_normal((1, 8, 10, 10)).astype(np.float32)) * 100.0

    def chain() -> list[Transform]:
        return [Resample(spacing=[3.0, 3.0, 3.0]), Resample(spacing=[1.5, 1.5, 1.5])]

    manager = _geometry_manager(streaming_dataset_stub, volume, chain(), None, np.asarray([1.5] * 3), np.eye(3))
    assert manager.shapes[0] == [8, 10, 10]
    manager.load(manager.transforms, [])
    reference = _fresh_chain_reference(volume, chain(), manager.dataset._attributes())
    assert torch.equal(manager.data[0], reference)

    patched = _geometry_manager(
        streaming_dataset_stub, volume, chain(), DatasetPatch([4, 4, 4]), np.asarray([1.5] * 3), np.eye(3)
    )
    assert patched.can_stream_patch(0), patched.stream_refusal(0)
    for index in range(patched.patch.get_size(0)):
        streamed = patched._get_streamed_data(index, 0, True)[0]
        assert torch.equal(streamed, patched.patch.get_data(reference, index, 0, True))


def test_softmax_channel_axis_is_pointwise_but_spatial_axis_falls_back(build_streaming_manager) -> None:
    # A channel-axis softmax (dim 0) is spatially pointwise (streamed equality: locality contract). A
    # softmax over a SPATIAL axis normalises across the whole extent, so a per-patch softmax would
    # diverge: the declaration must be WHOLE_VOLUME and the dispatcher must refuse to stream it.
    assert Softmax(0).patch_locality(Attribute()).kind is LocalityKind.POINTWISE
    assert Softmax(1).patch_locality(Attribute()).kind is LocalityKind.WHOLE_VOLUME
    assert Softmax(-1).patch_locality(Attribute()).kind is LocalityKind.WHOLE_VOLUME

    volume = np.arange(3 * 8 * 8, dtype=np.float32).reshape(3, 8, 8)
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
    # 'min'/'max' bounds clip to the volume extremum (a no-op on that bound), so the streamed
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


def test_a_saved_clip_bound_is_the_cases_statistic_not_the_regions(streaming_dataset_stub) -> None:
    """``save_clip_min``/``save_clip_max`` record the bound that was applied: the CASE's.

    On a streamed path the dispatcher seeds the case statistic and the tensor in hand is one
    region of it: a bound computed from that region records the region's own extremum on the
    attribute, and whatever reads it downstream (an inverse, a Normalize) then works off a number
    that depends on which patch happened to run.
    """
    rng = np.random.default_rng(4)
    volume = (rng.standard_normal((1, 8, 8)).astype(np.float32)) * 100.0
    stage = Clip(min_value="min", max_value="max", save_clip_min=True, save_clip_max=True)
    manager = DatasetManager(
        index=0,
        group_src="CT",
        group_dest="CT",
        name="CASE_000",
        dataset=cast(Dataset, streaming_dataset_stub(volume)),
        patch=DatasetPatch([4, 4]),
        transforms=[stage],
        data_augmentations_list=[],
    )
    assert manager.can_stream_patch(0)
    # The fixture only pins something if patch 0's extrema differ from the case's.
    patch0 = volume[:, :4, :4]
    assert float(patch0.min()) != float(volume.min()) or float(patch0.max()) != float(volume.max())
    _tensor, attribute = manager._get_streamed_data(0, 0, True)
    assert float(attribute["Min"]) == float(volume.min())
    assert float(attribute["Max"]) == float(volume.max())


def _one_row_budget(manager: DatasetManager) -> float:
    """The smallest budget this chain sweeps under: what one row of its landing holds, priced by the
    production rule rather than restated here."""
    segment = (manager.sweep_segments(0) or [])[-1]
    tile = manager._sweep_shape(segment.landing, segment.plans, 1)
    depth = patching._sweep_pipeline_depth()
    return float(manager.sweep_block_bytes(segment.landing, segment.channels, segment.plans, tile, depth))


def test_the_predicted_read_factor_prices_the_route_not_the_answer(streaming_dataset_stub) -> None:
    """A pointwise chain reads the source once; a store without bounded reads decodes it per slab.

    The factor is what the TRANSFORM verdict routes with: streaming is a memory strategy, and a
    case that fits its budget is loaded whole when streaming would read the source many times over.
    """
    volume = np.zeros((1, 8, 32, 32), dtype=np.float32)

    def manager(stub) -> DatasetManager:
        return DatasetManager(
            index=0,
            group_src="CT",
            group_dest="CT",
            name="CASE_000",
            dataset=cast(Dataset, stub),
            patch=DatasetPatch([4, 8, 8]),
            transforms=[Clip(min_value=-10.0, max_value=10.0)],
            data_augmentations_list=[],
        )

    bounded = manager(streaming_dataset_stub(volume))
    assert CaseMaterializer(bounded).predicted_stream_read_factor(0) == pytest.approx(1.0)

    class _Unbounded(streaming_dataset_stub):
        def bounded_region_reads(self, group_src: str, name: str) -> bool:
            return False

    # A budget that holds one row and no more cuts the sweep into 8 one-row slabs, each decoding
    # the whole store.
    unbounded = manager(_Unbounded(volume))
    unbounded.set_memory_budget(_one_row_budget(unbounded))
    assert CaseMaterializer(unbounded).predicted_stream_read_factor(0) == pytest.approx(8.0)


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


def test_stream_resample_nearest_strong_downsampling_matches_whole_volume(build_streaming_manager) -> None:
    # Strong downsampling of a uint8 label map (nearest mode): the nearest voxel of the first output
    # column (floor(o*scale)) falls BELOW the linear tap window's start, so the source read must widen
    # to include it: otherwise the gather indexes a negative local offset and wraps onto the far
    # edge, silently returning a wrong label. A regular ratio (a plain integer scale) hides the bug;
    # 40 -> 6 (scale 6.67) exposes the sub-pixel offset that pushes the linear start past voxel 0.
    volume = (np.arange(1 * 40 * 40).reshape(1, 40, 40) % 7).astype(np.uint8)
    shape = [6, 6]
    patch = [3, 3]
    stream_manager = build_streaming_manager(volume, [Resample(shape=shape)], patch)
    assert stream_manager.can_stream_patch(0)

    reference_manager = build_streaming_manager(volume, [Resample(shape=shape)], patch)
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
    resample = Resample(shape=[n_out, n_out, n_out], inverse=False)
    attribute = Attribute()
    attribute["Spacing"] = np.ones(3)
    expected = resample("case", volume.clone(), Attribute(attribute))

    target = tuple(slice(0, n_out) for _ in range(3))
    window = resample.stream_region_source("case", target, [n_in] * 3, Attribute(attribute))
    context = RegionContext(tuple(window), target, (n_in,) * 3)
    got = resample.stream_region("case", volume[(slice(None), *window)], context, Attribute(attribute))
    torch.testing.assert_close(got, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.uint8])
def test_streamed_resample_handles_2d(dtype: torch.dtype) -> None:
    """The gather must not assume three spatial axes."""
    volume = (torch.arange(1 * 9 * 11, dtype=torch.float32).reshape(1, 9, 11) % 17).to(dtype)
    resample = Resample(shape=[5, 6], inverse=False)
    attribute = Attribute()
    attribute["Spacing"] = np.ones(2)
    expected = resample("case", volume.clone(), Attribute(attribute))

    target = (slice(0, 5), slice(0, 6))
    window = resample.stream_region_source("case", target, [9, 11], Attribute(attribute))
    context = RegionContext(tuple(window), target, (9, 11))
    got = resample.stream_region("case", volume[(slice(None), *window)], context, Attribute(attribute))
    torch.testing.assert_close(got, expected, rtol=0, atol=0)


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

    Replanning must not read the live case attribute (it already carries epoch 1's target
    Spacing/Direction) or a streamed Resample degrades to identity and a streamed Canonical stops
    reorienting from the second epoch on, while the geometry keys stack once more per epoch.
    """
    volume = np.arange(1 * 8 * 8 * 8, dtype=np.float32).reshape(1, 8, 8, 8)
    transform = Resample(shape=[16, 16, 16]) if transform_case == "resample" else Canonical()
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
        self, name: str, target_slices: tuple[slice, ...], source_spatial_shape: list[int], cache_attribute: Attribute
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


def test_statistics_streams_off_the_seeded_case_numbers(streaming_dataset_stub) -> None:
    """``Statistics`` records the CASE's four numbers: the disk scan already computes them.

    Whole-volume was the default it never needed: seeded, each region restates the case's answer,
    and the recorded ``Image*`` keys equal the volume's own statistics rather than a region's.
    """
    from konfai.data.transform import Statistics

    rng = np.random.default_rng(6)
    volume = (rng.standard_normal((1, 8, 8)).astype(np.float32)) * 100.0
    stub = streaming_dataset_stub(volume)
    manager = DatasetManager(
        index=0,
        group_src="CT",
        group_dest="CT",
        name="CASE_000",
        dataset=cast(Dataset, stub),
        patch=DatasetPatch([4, 4]),
        transforms=[Statistics()],
        data_augmentations_list=[],
    )
    assert manager.can_stream_patch(0), manager.stream_refusal(0)
    _tensor, attribute = manager._get_streamed_data(0, 0, True)
    assert float(attribute["ImageMin"]) == pytest.approx(float(volume.min()))
    assert float(attribute["ImageMax"]) == pytest.approx(float(volume.max()))
    assert float(attribute["ImageMean"]) == pytest.approx(float(volume.mean()), rel=1e-6)
    assert float(attribute["ImageStd"]) == pytest.approx(float(volume.std(ddof=1)), rel=1e-6)
    assert stub.full_reads == 0 and stub.stats_reads == 1


def test_the_read_factor_grows_as_the_budget_cuts_finer_slabs(streaming_dataset_stub) -> None:
    """Streaming finer never reads less: the monotonicity the route's pricing rests on.

    A halo chain re-reads its overlap at every slab boundary, so the factor sits near 1 when one
    slab covers the volume and grows as the budget shrinks the slabs. This restates, on the
    estimator the verdict actually uses, the property the deleted ``read_amplification`` pinned.
    """
    volume = np.zeros((1, 32, 16, 16), dtype=np.float32)
    manager = DatasetManager(
        index=0,
        group_src="CT",
        group_dest="CT",
        name="CASE_000",
        dataset=cast(Dataset, streaming_dataset_stub(volume)),
        patch=DatasetPatch([8, 16, 16]),
        transforms=[Dilate(2)],
        data_augmentations_list=[],
    )
    one_row = _one_row_budget(manager)
    factors = []
    for budget in (None, 8 * one_row, 3 * one_row, one_row):
        manager.set_memory_budget(budget)
        factors.append(CaseMaterializer(manager).predicted_stream_read_factor(0))
    # The ~1.0 first factor holds only while the no-budget sweep covers the volume in ONE slab;
    # a smaller default cap would split it and re-read the Dilate halo at each boundary.
    assert patching.SWEEP_SLAB_ROWS >= volume.shape[1], "the first factor's premise moved"
    assert factors[0] == pytest.approx(1.0, abs=0.2)  # one slab: the whole source, once
    assert factors == sorted(factors), f"the factor must be monotone in fineness, got {factors}"
    assert factors[-1] > 2.0
