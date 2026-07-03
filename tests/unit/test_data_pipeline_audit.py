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

"""Regression tests for data-pipeline audit fixes (patching / caching / streaming)."""

import os
import threading
from typing import cast

os.environ.setdefault("KONFAI_config_file", "/tmp/konfai-none.yml")
os.environ.setdefault("KONFAI_CONFIG_MODE", "Done")

import numpy as np
import torch
from konfai.data.augmentation import DataAugmentation, DataAugmentationsList
from konfai.data.data_manager import DatasetIter, Group, GroupTransform, _cache_worker_count
from konfai.data.patching import Cosinus, DatasetManager, DatasetPatch, Mean
from konfai.data.transform import TensorCast
from konfai.utils.dataset import Attribute, Dataset, _get_h5_file_lock


def _image_attributes(origin: list[float], spacing: list[float]) -> Attribute:
    attributes = Attribute()
    attributes["Origin"] = np.asarray(origin, dtype=np.float64)
    attributes["Spacing"] = np.asarray(spacing, dtype=np.float64)
    attributes["Direction"] = np.eye(len(origin), dtype=np.float64).reshape(-1)
    return attributes


# --------------------------------------------------------------------------------------
# B10 - PathCombine.set_patch_config with overlap == 0
# --------------------------------------------------------------------------------------


def test_path_combine_overlap_zero_uses_uniform_weights() -> None:
    """overlap=0 tiles patches without overlap, so the blend window is all ones."""
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


# --------------------------------------------------------------------------------------
# B18 - caching worker count must never fall below one
# --------------------------------------------------------------------------------------


def test_cache_worker_count_never_drops_below_one() -> None:
    # 2 CPUs shared across 4 GPUs would be 2 // 4 == 0 without the floor.
    assert _cache_worker_count(2, 4) == 1
    assert _cache_worker_count(1, 4) == 1
    assert _cache_worker_count(8, 2) == 4
    assert _cache_worker_count(7, 2) == 3
    assert _cache_worker_count(4, 0) == 4  # no device -> divisor 1


# --------------------------------------------------------------------------------------
# B3 - patch streaming must persist TensorCast dtype for the inverse
# --------------------------------------------------------------------------------------


class _StreamingDatasetStub:
    def __init__(self, volume: np.ndarray) -> None:
        self.volume = volume

    def get_infos(self, group_src: str, name: str) -> tuple[list[int], Attribute]:
        return list(self.volume.shape), _image_attributes([0.0, 0.0], [1.0, 1.0])

    def read_data(self, group_src: str, name: str) -> tuple[np.ndarray, Attribute]:
        return self.volume.copy(), _image_attributes([0.0, 0.0], [1.0, 1.0])

    def read_data_slice(self, group_src: str, name: str, slices: tuple[slice, ...]) -> tuple[np.ndarray, Attribute]:
        return self.volume[slices].copy(), _image_attributes([0.0, 0.0], [1.0, 1.0])

    def read_data_statistics(self, group_src: str, name: str, channels: list[int] | None = None) -> dict[str, float]:
        data = self.volume if channels is None else self.volume[channels]
        return {
            "min": float(data.min()),
            "max": float(data.max()),
            "mean": float(data.mean()),
            "std": float(data.std(ddof=1)),
        }


def test_streaming_tensorcast_persists_source_dtype_for_inverse() -> None:
    volume = np.arange(1 * 4 * 4, dtype=np.int16).reshape(1, 4, 4)
    dataset_stub = _StreamingDatasetStub(volume)
    manager = DatasetManager(
        index=0,
        group_src="CT",
        group_dest="CT",
        name="CASE_000",
        dataset=cast(Dataset, dataset_stub),
        patch=DatasetPatch([2, 2]),
        transforms=[TensorCast(dtype="float32")],
        data_augmentations_list=[],
    )
    dataset_iter = DatasetIter(
        rank=0,
        data={"CT": [manager]},
        mapping=[(0, 0, 1)],
        groups_src={"CT": Group(groups_dest={"CT": GroupTransform(transforms=None, patch_transforms=None)})},
        inline_augmentations=False,
        data_augmentations_list=[],
        patch_size=[2, 2],
        overlap=None,
        buffer_size=1,
        use_cache=False,
    )

    sample = dataset_iter[0]["CT"].tensor

    assert sample.dtype == torch.float32
    # The forward cast records the source dtype on the persistent case attribute.
    assert "dtype" in manager.cache_attributes[0]
    # ... so the write-time inverse can restore the original dtype without crashing.
    restored = TensorCast(dtype="float32").inverse("CASE_000", sample, Attribute(manager.cache_attributes[0]))
    assert restored.dtype == torch.int16


# --------------------------------------------------------------------------------------
# B11 - reset_augmentation must draw the shared state once per case, not per group
# --------------------------------------------------------------------------------------


class _DummyDataset:
    def __init__(self, array: np.ndarray) -> None:
        self.array = array

    def get_infos(self, group_src: str, name: str) -> tuple[list[int], Attribute]:
        return list(self.array.shape), Attribute({"name": name, "group": group_src})

    def read_data(self, group_src: str, name: str) -> tuple[np.ndarray, Attribute]:
        return self.array.copy(), Attribute({"name": name, "group": group_src})


class _DrawCountingAugmentation(DataAugmentation):
    """Shape-shifting augmentation whose output depends on the draw order."""

    def __init__(self) -> None:
        super().__init__()
        self.draws = 0

    def _state_init(self, index: int, shapes: list[list[int]], caches_attribute: list[Attribute]) -> list[list[int]]:
        self.draws += 1
        new_shape = [2, 4] if self.draws == 1 else [4, 4]
        return [list(new_shape) for _ in shapes]

    def _compute(self, name: str, index: int, tensors: list[torch.Tensor]) -> list[torch.Tensor]:
        return tensors

    def _inverse(self, index: int, a: int, tensor: torch.Tensor) -> torch.Tensor:
        return tensor


def test_reset_augmentation_shares_one_draw_across_destination_groups() -> None:
    array = np.zeros((1, 4, 4), dtype=np.float32)
    dataset = cast(Dataset, _DummyDataset(array))
    augmentation = _DrawCountingAugmentation()
    augmentation.load(1.0)
    augmentations = DataAugmentationsList(nb=1, data_augmentations={})
    augmentations.data_augmentations = [augmentation]

    manager_a = DatasetManager(
        index=0,
        group_src="src",
        group_dest="destA",
        name="case_000",
        dataset=dataset,
        patch=DatasetPatch([2, 2]),
        transforms=[],
        data_augmentations_list=[augmentations],
    )
    manager_b = DatasetManager(
        index=0,
        group_src="src",
        group_dest="destB",
        name="case_000",
        dataset=dataset,
        patch=DatasetPatch([2, 2]),
        transforms=[],
        data_augmentations_list=[augmentations],
    )
    dataset_iter = DatasetIter(
        rank=0,
        data={"destA": [manager_a], "destB": [manager_b]},
        mapping=[(0, 0, 0), (0, 1, 0)],
        groups_src={
            "src": Group(
                groups_dest={
                    "destA": GroupTransform(transforms=None, patch_transforms=None),
                    "destB": GroupTransform(transforms=None, patch_transforms=None),
                }
            )
        },
        inline_augmentations=True,
        data_augmentations_list=[augmentations],
        patch_size=[2, 2],
        overlap=None,
        buffer_size=1,
        use_cache=False,
    )

    augmentation.draws = 0
    dataset_iter.reset_augmentation("Train")

    # A single random draw feeds every destination group of the case.
    assert augmentation.draws == 1
    # Both groups therefore rebuild their augmented patch grid from the same shape.
    assert manager_a.patch.get_size(1) == manager_b.patch.get_size(1)


# --------------------------------------------------------------------------------------
# B6 - concurrent HDF5 access is serialised per file
# --------------------------------------------------------------------------------------


def test_h5_writes_are_serialised_per_file(tmp_path) -> None:
    dataset = Dataset(str(tmp_path / "Volumes"), "h5")
    attrs = _image_attributes([0.0, 0.0], [1.0, 1.0])
    dataset.write("CT", "CASE_000", np.zeros((1, 2, 2), dtype=np.float32), attrs)

    lock = _get_h5_file_lock(str(tmp_path / "Volumes") + ".h5")
    started = threading.Event()
    finished = threading.Event()

    def writer() -> None:
        started.set()
        dataset.write("CT", "CASE_001", np.ones((1, 2, 2), dtype=np.float32), attrs)
        finished.set()

    with lock:  # holding the file lock must block any other writer on the same file
        thread = threading.Thread(target=writer)
        thread.start()
        assert started.wait(1.0)
        assert not finished.wait(0.2), "a second writer proceeded while the file lock was held"

    thread.join(5.0)
    assert finished.is_set()
    data, _ = dataset.read_data("CT", "CASE_001")
    np.testing.assert_array_equal(data, np.ones((1, 2, 2), dtype=np.float32))
