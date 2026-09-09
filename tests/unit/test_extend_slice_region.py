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

"""A 2.5D model reads ``extend_slice`` neighbouring slices around each one-slice patch. The region
route once replayed the chain on the grid slot alone and then applied the plan's reflection and
concatenation to that single slice: the first boundary patch failed in ``F.pad`` and an interior
one came back ``[8, 8]`` where the model expects ``[extend_slice + 1, 8, 8]``. The region target
now comes from the same read plan the whole-volume path cuts with."""

from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("h5py")

from konfai.data.augmentation import DataAugmentationsList
from konfai.data.augmentation import Flip as FlipAugmentation
from konfai.data.patching import DatasetManager, DatasetPatch
from konfai.data.transform import Mask, Standardize
from konfai.utils.dataset import Attribute, Dataset


def _dataset(root: Path) -> Dataset:
    dataset = Dataset(root / "data", "h5")
    values = (np.arange(9 * 6 * 5, dtype=np.float32).reshape(1, 9, 6, 5) % 37) / 11
    mask = np.ones_like(values, dtype=np.uint8)
    mask[:, :, 0, :] = 0
    mask[:, 3:6, 2:4, 1:3] = 0
    attribute = Attribute()
    attribute["Origin"] = [1.0, 2.0, 3.0]
    attribute["Spacing"] = [1.1, 1.2, 1.3]
    attribute["Direction"] = np.eye(3).flatten()
    dataset.write("MR", "CASE_000", values, attribute)
    dataset.write("MASK", "CASE_000", mask, attribute)
    return dataset


def _flips() -> list[DataAugmentationsList]:
    groups = []
    for axis in range(3):
        probability = [0.0, 0.0, 0.0]
        probability[axis] = 1.0
        flip = FlipAugmentation(f_prob=probability)
        flip.load(1.0)
        group = DataAugmentationsList(nb=1, data_augmentations={})
        group.data_augmentations = [flip]
        groups.append(group)
    return groups


def _manager(dataset: Dataset, extend_slice: int) -> DatasetManager:
    stages = [Standardize(mask="MASK", inverse=False), Mask(path="MASK", value_outside=-2)]
    for stage in stages:
        stage.set_datasets([dataset])
    return DatasetManager(
        index=0,
        group_src="MR",
        group_dest="MR",
        name="CASE_000",
        dataset=dataset,
        patch=DatasetPatch([1, 8, 8], pad_value=-1, extend_slice=extend_slice),
        transforms=stages,
        data_augmentations_list=_flips(),
    )


@pytest.mark.parametrize("extend_slice", [2, 4])
def test_region_patches_carry_the_slice_context_of_the_whole_volume_path(tmp_path: Path, extend_slice: int):
    """Every patch of every copy (identity and a flip along each axis), at all nine slice
    positions, equals the whole-volume chain cut by ``Patch.get_data``; the declared reads are
    the plan's windows."""
    dataset = _dataset(tmp_path)
    streamed = _manager(dataset, extend_slice)
    reference = _manager(dataset, extend_slice)
    reference.load(reference.transforms, reference.data_augmentations_list)
    assert reference.loaded

    checked = 0
    for copy in range(4):
        assert streamed.can_stream_patch(copy)
        for index in range(streamed.get_size(copy)):
            got = streamed.get_data(index, copy, [], True)
            expected = reference.get_data(index, copy, [], True)
            assert list(got.shape) == [extend_slice + 1, 8, 8]
            torch.testing.assert_close(got, expected, rtol=2e-6, atol=2e-6)
            source = streamed._resolve_patch_stream_source(copy, True)
            declared = streamed._patch_read_spans(source, index, copy, True)[-1]
            planned = streamed.patch.get_read_plan(streamed.shapes[copy], index, copy, True)
            assert tuple(declared) == planned.data_slices[len(planned.data_slices) - 3 :]
            checked += 1
    assert checked == 4 * 9
