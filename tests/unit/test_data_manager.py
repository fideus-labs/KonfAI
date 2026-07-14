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

"""Tests for ``konfai.data.data_manager``: DDP sharding, train/validation split,
cache workers, and DatasetIter (streaming transforms, inline augmentations)."""

import os
import subprocess
import sys
from typing import cast

import numpy as np
import pytest
import torch
from konfai.data.augmentation import DataAugmentation, DataAugmentationsList
from konfai.data.data_manager import Data, DatasetIter, DataTrain, Group, GroupTransform, _cache_worker_count
from konfai.data.patching import DatasetManager, DatasetPatch
from konfai.data.transform import TensorCast
from konfai.utils.dataset import Attribute, Dataset
from konfai.utils.runtime import State

# --------------------------------------------------------------------------------------
# Data._split — TRAIN/RESUME shards must be equal length to avoid a DDP hang
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("state", [State.TRAIN, State.RESUME])
def test_train_split_equalises_indivisible_shards(monkeypatch: pytest.MonkeyPatch, state: State) -> None:
    # DDP(static_graph=True) needs every rank to run the same number of backward all-reduces per epoch,
    # so shards must be equal length. They are equalised by PADDING (wrapping the shard's own head), not
    # truncating: 7 patches over 3 ranks -> [3, 3, 3]. Every original sample still trains (truncation
    # would permanently drop the tail sample, which _split runs once so no epoch shuffle recovers it).
    monkeypatch.setenv("KONFAI_STATE", str(state))

    mapping = [(index, 0, 0) for index in range(7)]
    shards = Data._split(mapping, 3)

    lengths = [len(shard) for shard in shards]
    assert len(set(lengths)) == 1  # equal length -> no NCCL desync
    flattened = [item for shard in shards for item in shard]
    assert set(flattened) == set(mapping)  # nothing permanently dropped
    assert len(flattened) - len(set(flattened)) <= 3  # only minimal padding duplicates (<= world_size)


def test_train_split_two_ranks_indivisible(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KONFAI_STATE", str(State.TRAIN))

    mapping = [(i, 0, 0) for i in range(5)]
    shards = Data._split(mapping, 2)

    # Equal-length shards via padding; every sample is still present (no tail dropped).
    lengths = [len(shard) for shard in shards]
    assert len(set(lengths)) == 1
    flattened = [item for shard in shards for item in shard]
    assert set(flattened) == set(mapping)


def test_train_split_single_process_keeps_everything(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KONFAI_STATE", str(State.TRAIN))
    mapping = [(i, 0, 0) for i in range(5)]
    assert Data._split(mapping, 1) == [mapping]  # world_size == 1 is a no-op


# --------------------------------------------------------------------------------------
# DataTrain train/validation split — reproducible and seeded from sorted names
# --------------------------------------------------------------------------------------

_SPLIT_PROBE = """
import os
import random

os.environ.setdefault("KONFAI_config_file", "/tmp/konfai-none.yml")
os.environ.setdefault("KONFAI_CONFIG_MODE", "Done")

from konfai.data.data_manager import DataTrain

names = [f"CASE_{i:03d}" for i in range(20)]
data = DataTrain(augmentations=None, validation="0:4")
data._resolve_dataset_sources = lambda: {}
data._resolve_common_names = lambda datasets: ({}, set(names))
data._get_datasets = lambda case_names, dataset_name, augmentations: ({}, [])
random.seed(1234)
data._prepare_datasets()
print(";".join(data._prepared_train_names))
print(";".join(data._prepared_validation_names))
"""


def test_train_validation_split_is_reproducible_across_interpreters():
    """Same seed → same split, whatever the interpreter's string-hash randomization."""
    outputs = []
    for hash_seed in ("0", "424242"):
        env = dict(os.environ, PYTHONHASHSEED=hash_seed)
        result = subprocess.run(
            [sys.executable, "-c", _SPLIT_PROBE],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        outputs.append(result.stdout)
    assert outputs[0] == outputs[1]
    train_names, validation_names = (line.split(";") for line in outputs[0].splitlines())
    assert len(train_names) == 16
    assert len(validation_names) == 4
    assert set(train_names).isdisjoint(validation_names)


def test_train_split_shuffle_draws_from_sorted_names(monkeypatch):
    """The seeded shuffle must receive the case names in sorted order and drive the split."""
    captured: dict[str, list[str]] = {}

    def fake_sample(population, k):
        captured["population"] = list(population)
        assert k == len(population)
        return list(reversed(population))

    monkeypatch.setattr("konfai.data.data_manager.random.sample", fake_sample)

    data = DataTrain(augmentations=None, validation="0:2")
    names = {"CASE_010", "CASE_002", "CASE_001", "CASE_005", "CASE_003"}
    data._resolve_dataset_sources = lambda: {}
    data._resolve_common_names = lambda datasets: ({}, names)
    data._get_datasets = lambda case_names, dataset_name, augmentations: ({}, [])
    data._prepare_datasets()

    assert captured["population"] == sorted(names)
    assert data._prepared_validation_names == ["CASE_010", "CASE_005"]
    assert data._prepared_train_names == ["CASE_003", "CASE_002", "CASE_001"]


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


def _image_attributes(origin: list[float], spacing: list[float]) -> Attribute:
    attributes = Attribute()
    attributes["Origin"] = np.asarray(origin, dtype=np.float64)
    attributes["Spacing"] = np.asarray(spacing, dtype=np.float64)
    attributes["Direction"] = np.eye(len(origin), dtype=np.float64).reshape(-1)
    return attributes


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
# DatasetIter — inline augmentations and per-case state draws
# --------------------------------------------------------------------------------------


class _DummyDataset:
    def __init__(self, array: np.ndarray) -> None:
        self.array = array

    def get_infos(self, group_src: str, name: str) -> tuple[list[int], Attribute]:
        return list(self.array.shape), Attribute({"name": name, "group": group_src})

    def read_data(self, group_src: str, name: str) -> tuple[np.ndarray, Attribute]:
        return self.array.copy(), Attribute({"name": name, "group": group_src})


class _CountingOffsetAugmentation(DataAugmentation):
    def __init__(self) -> None:
        super().__init__()
        self.compute_calls = 0

    def _state_init(
        self,
        index: int,
        shapes: list[list[int]],
        caches_attribute: list[Attribute],
    ) -> list[list[int]]:
        return shapes

    def _compute(
        self,
        name: str,
        index: int,
        tensors: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        self.compute_calls += 1
        return [tensor + (offset + 1) for offset, tensor in enumerate(tensors)]

    def _inverse(self, index: int, a: int, tensor: torch.Tensor) -> torch.Tensor:
        return tensor


def _make_manager(dataset: Dataset, augmentations: DataAugmentationsList, group_dest: str = "dest") -> DatasetManager:
    return DatasetManager(
        index=0,
        group_src="src",
        group_dest=group_dest,
        name="case_000",
        dataset=dataset,
        patch=None,
        transforms=[],
        data_augmentations_list=[augmentations],
    )


def test_inline_augmentations_are_loaded_on_demand() -> None:
    base = np.arange(4, dtype=np.float32).reshape(1, 2, 2)
    dataset = cast(Dataset, _DummyDataset(base))
    augmentation = _CountingOffsetAugmentation()
    augmentation.load(1.0)

    augmentations = DataAugmentationsList(nb=2, data_augmentations={})
    augmentations.data_augmentations = [augmentation]

    manager = _make_manager(dataset, augmentations)
    dataset_iter = DatasetIter(
        rank=0,
        data={"dest": [manager]},
        mapping=[(0, 0, 0), (0, 1, 0), (0, 2, 0)],
        groups_src={"src": Group(groups_dest={"dest": GroupTransform(transforms=None, patch_transforms=None)})},
        inline_augmentations=True,
        data_augmentations_list=[augmentations],
        patch_size=None,
        overlap=None,
        buffer_size=1,
        use_cache=True,
    )

    base_sample = dataset_iter[0]["dest"].tensor
    assert augmentation.compute_calls == 0
    assert manager.loaded is True
    assert manager.augmentationLoaded is False
    assert torch.equal(base_sample, torch.from_numpy(base))

    first_augmented_sample = dataset_iter[1]["dest"].tensor
    assert augmentation.compute_calls == 1
    assert manager.augmentationLoaded is True
    assert torch.equal(first_augmented_sample, torch.from_numpy(base) + 1)

    second_augmented_sample = dataset_iter[2]["dest"].tensor
    assert augmentation.compute_calls == 1
    assert torch.equal(second_augmented_sample, torch.from_numpy(base) + 2)


def test_dataset_iter_can_skip_augmentation_loading_when_validation_disables_them() -> None:
    base = np.arange(4, dtype=np.float32).reshape(1, 2, 2)
    dataset = cast(Dataset, _DummyDataset(base))
    augmentation = _CountingOffsetAugmentation()
    augmentation.load(1.0)

    augmentations = DataAugmentationsList(nb=2, data_augmentations={})
    augmentations.data_augmentations = [augmentation]

    manager = _make_manager(dataset, augmentations)
    dataset_iter = DatasetIter(
        rank=0,
        data={"dest": [manager]},
        mapping=[(0, 0, 0)],
        groups_src={"src": Group(groups_dest={"dest": GroupTransform(transforms=None, patch_transforms=None)})},
        inline_augmentations=False,
        data_augmentations_list=[augmentations],
        patch_size=None,
        overlap=None,
        buffer_size=1,
        apply_augmentations=False,
        use_cache=True,
    )

    dataset_iter.load("Validation")
    base_sample = dataset_iter[0]["dest"].tensor

    assert augmentation.compute_calls == 0
    assert manager.loaded is True
    assert manager.augmentationLoaded is False
    assert torch.equal(base_sample, torch.from_numpy(base))


# --------------------------------------------------------------------------------------
# B11 - reset_augmentation must draw the shared state once per case, not per group
# --------------------------------------------------------------------------------------


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
