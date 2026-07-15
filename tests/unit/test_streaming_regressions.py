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

"""Streamed-path regressions: each test encodes one reviewed failure and must keep failing loudly.

Every scenario here was reproduced on the branch before the fix it pins down:
- a GLOBAL_STAT after a value-preserving cast must still stream (parity with the released behaviour);
- an augmented copy must consume the volume statistic, not each patch's own;
- replanning after an epoch's re-draw must read the stored geometry, not the previous epoch's output;
- two groups of one case must share one augmentation draw at construction;
- the streamed resample must match the whole-volume one in 2D and for nearest at any size ratio.
"""

from typing import cast

import numpy as np
import pytest
import torch
from konfai.data.augmentation import DataAugmentationsList
from konfai.data.augmentation import Flip as FlipAugmentation
from konfai.data.augmentation import Rotate as RotateAugmentation
from konfai.data.patching import DatasetManager, DatasetPatch
from konfai.data.transform import (
    Canonical,
    LocalityKind,
    Normalize,
    PatchLocality,
    ResampleToShape,
    TensorCast,
    Transform,
)
from konfai.utils.dataset import Attribute, Dataset
from konfai.utils.errors import PatchError


class StreamingDatasetStub:
    """In-memory dataset serving whole reads, region reads, and statistics, with full geometry."""

    def __init__(self, volume: np.ndarray) -> None:
        self.volume = volume
        self.full_reads = 0
        self.patch_reads = 0

    def _attributes(self) -> Attribute:
        attribute = Attribute()
        spatial = self.volume.ndim - 1
        attribute["Origin"] = np.zeros(spatial)
        attribute["Spacing"] = np.ones(spatial)
        attribute["Direction"] = np.eye(spatial).flatten()
        return attribute

    def get_infos(self, group_src: str, name: str) -> tuple[list[int], Attribute]:
        return list(self.volume.shape), self._attributes()

    def read_data(self, group_src: str, name: str) -> tuple[np.ndarray, Attribute]:
        self.full_reads += 1
        return self.volume.copy(), self._attributes()

    def read_data_slice(self, group_src: str, name: str, slices: tuple[slice, ...]) -> tuple[np.ndarray, Attribute]:
        self.patch_reads += 1
        return self.volume[slices].copy(), self._attributes()

    def read_data_statistics(self, group_src: str, name: str, channels: list[int] | None = None) -> dict[str, float]:
        data = self.volume if channels is None else self.volume[channels]
        return {
            "min": float(data.min()),
            "max": float(data.max()),
            "mean": float(data.mean()),
            "std": float(data.std(ddof=1)),
        }


_GeometryDatasetStub = StreamingDatasetStub


def _manager(stub: StreamingDatasetStub, transforms, augmentations=(), patch=(4, 4, 4), group="CT") -> DatasetManager:
    return DatasetManager(
        index=0,
        group_src=group,
        group_dest=group,
        name="CASE_000",
        dataset=cast(Dataset, stub),
        patch=DatasetPatch(list(patch)),
        transforms=list(transforms),
        data_augmentations_list=list(augmentations),
    )


def _flip_augmentations() -> list[DataAugmentationsList]:
    augmentations = DataAugmentationsList(nb=1, data_augmentations={})
    flip = FlipAugmentation(f_prob=[1.0, 1.0, 1.0])
    flip.load(1.0)
    augmentations.data_augmentations = [flip]
    return [augmentations]


def test_global_stat_after_float_cast_still_streams_and_matches() -> None:
    """[TensorCast -> float, Normalize] streams (released behaviour) and equals the whole-volume path."""
    volume = np.arange(1 * 8 * 8 * 8, dtype=np.uint8).reshape(1, 8, 8, 8)
    transforms = [TensorCast("float32"), Normalize()]
    streamed = _manager(StreamingDatasetStub(volume), transforms)
    reference = _manager(StreamingDatasetStub(volume), transforms)
    reference.load(transforms, [])

    assert streamed.can_stream_patch(0)
    for index in range(streamed.get_size(0)):
        got = streamed.get_data(index, 0, [], True)
        expected = reference.get_data(index, 0, [], True)
        torch.testing.assert_close(got, expected, rtol=0, atol=0)


def test_augmented_copy_consumes_the_volume_statistic() -> None:
    """Copy a=1, requested before copy 0 ever streams, still normalizes by the VOLUME's Min/Max."""
    torch.manual_seed(0)
    volume = np.arange(1 * 8 * 8 * 8, dtype=np.float32).reshape(1, 8, 8, 8)
    streamed = _manager(StreamingDatasetStub(volume), [Normalize()], _flip_augmentations())
    reference = _manager(StreamingDatasetStub(volume), [Normalize()], streamed.data_augmentations_list)
    reference.load([Normalize()], reference.data_augmentations_list)

    assert streamed.can_stream_patch(1)
    for index in range(streamed.get_size(1)):
        got = streamed.get_data(index, 1, [], True)
        expected = reference.get_data(index, 1, [], True)
        torch.testing.assert_close(got, expected, rtol=0, atol=1e-6)


@pytest.mark.parametrize("transform_case", ["resample", "canonical"])
def test_replanning_after_epoch_redraw_keeps_the_stored_geometry(transform_case: str) -> None:
    """Epoch 2 must stream the same bytes as epoch 1, with no attribute stack growth.

    Replanning used to read the live case attribute -- already carrying epoch 1's target
    Spacing/Direction -- so a streamed Resample degraded to identity and a streamed Canonical stopped
    reorienting from the second epoch on, while the geometry keys stacked once more per epoch.
    """
    volume = np.arange(1 * 8 * 8 * 8, dtype=np.float32).reshape(1, 8, 8, 8)
    transform = ResampleToShape(shape=[16, 16, 16]) if transform_case == "resample" else Canonical()
    streamed = _manager(_GeometryDatasetStub(volume), [transform], _flip_augmentations())

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


def test_two_groups_share_one_construction_draw() -> None:
    """Building the label group's manager must reuse the image group's draw, not redraw over it.

    A quarter Rotate transposes per-copy extents, so a per-group redraw leaves the two groups with
    different copy grids -- crashing the streamed read of the stale grid's last patch.
    """
    volume = np.zeros((1, 6, 8, 10), dtype=np.float32)
    for seed in range(10):
        torch.manual_seed(seed)
        augmentations = DataAugmentationsList(nb=2, data_augmentations={})
        rotate = RotateAugmentation(is_quarter=True)
        rotate.load(1.0)
        augmentations.data_augmentations = [rotate]
        image_manager = _manager(StreamingDatasetStub(volume), [], [augmentations], group="CT")
        label_manager = _manager(StreamingDatasetStub(volume), [], [augmentations], group="SEG")
        assert image_manager.shapes == label_manager.shapes, f"seed {seed}"


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


def test_a_region_stage_recording_geometry_nowhere_the_case_reads_is_refused() -> None:
    # A declaration this framework cannot honour must fail where it is made, not persist the geometry
    # of the volume as stored and call the run correct.
    stub = StreamingDatasetStub(np.arange(1 * 8 * 8 * 8, dtype=np.float32).reshape(1, 8, 8, 8))
    manager = _manager(stub, [_RecordsOnlyInCall()])
    with pytest.raises(PatchError) as error:
        manager.get_data(0, 0, [], True)
    assert "write_stream_cache_attribute" in str(error.value)
