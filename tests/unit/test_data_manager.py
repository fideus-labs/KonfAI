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
from konfai.data.data_manager import (
    Data,
    DatasetIter,
    DataTrain,
    Group,
    GroupTransform,
    PredictionSubset,
    TrainSubset,
    WindowedCaseSampler,
    _cache_worker_count,
)
from konfai.data.patching import DatasetManager, DatasetPatch
from konfai.data.transform import TensorCast, Transform
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
# Data._split — PREDICTION/EVALUATION shards must keep every case whole on one rank
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("state", [State.PREDICTION, State.EVALUATION])
@pytest.mark.parametrize("world_size", [2, 3])
def test_prediction_split_keeps_every_case_on_one_rank(
    monkeypatch: pytest.MonkeyPatch, state: State, world_size: int
) -> None:
    # The streamed write (and the TTA aligner) reassemble a case from ALL its patches: a case split
    # across ranks would leave every rank's accumulator forever incomplete.
    monkeypatch.setenv("KONFAI_STATE", str(state))
    mapping = [(case, a, p) for case in range(5) for a in range(2) for p in range(3)]

    shards = Data._split(mapping, world_size)

    assert sorted(entry for shard in shards for entry in shard) == sorted(mapping)
    owner: dict[int, int] = {}
    for rank, shard in enumerate(shards):
        for entry in shard:
            assert owner.setdefault(entry[0], rank) == rank, f"case {entry[0]} split across ranks"


def test_prediction_split_more_ranks_than_cases_leaves_spare_ranks_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KONFAI_STATE", str(State.PREDICTION))
    mapping = [(case, 0, p) for case in range(2) for p in range(4)]

    shards = Data._split(mapping, 4)

    assert sorted(entry for shard in shards for entry in shard) == sorted(mapping)
    non_empty = [shard for shard in shards if shard]
    assert len(non_empty) == 2
    for shard in non_empty:
        assert len({entry[0] for entry in shard}) == 1  # one whole case per busy rank


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
data._get_datasets = lambda case_names, dataset_name, augmentations, index_offset=0: ({}, [])
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
    data._get_datasets = lambda case_names, dataset_name, augmentations, index_offset=0: ({}, [])
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


class _WholeVolumeTransform(Transform):
    """A spatial identity that declares nothing, so its chain can only run on a whole volume.

    Cases here are about what happens once a volume is resident -- the FIFO buffer, the augmentation
    draws -- so they need a chain the streamer refuses. Declaring it is how a chain says so.
    """

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        return tensor


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

    def _compute(self, name: str, index: int, a: int, tensor: torch.Tensor) -> torch.Tensor:
        self.compute_calls += 1
        return tensor + (a + 1)

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
        transforms=[_WholeVolumeTransform()],
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
    # One call per copy: the group's two copies are drawn together, on first demand.
    assert augmentation.compute_calls == 2
    assert manager.augmentationLoaded is True
    assert torch.equal(first_augmented_sample, torch.from_numpy(base) + 1)

    second_augmented_sample = dataset_iter[2]["dest"].tensor
    assert augmentation.compute_calls == 2
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

    def _compute(self, name: str, index: int, a: int, tensor: torch.Tensor) -> torch.Tensor:
        return tensor

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
# WindowedCaseSampler - locality-aware training order, worker sharding, buffer hit rate
# --------------------------------------------------------------------------------------


def _case_major_mapping(n_cases: int, patches_per_case: int) -> list[tuple[int, int, int]]:
    return [(x, 0, p) for x in range(n_cases) for p in range(patches_per_case)]


def _distinct_cases_per_slice(order: list[int], mapping: list[tuple[int, int, int]], slice_len: int) -> int:
    cases = [mapping[i][0] for i in order]
    return max(len(set(cases[k : k + slice_len])) for k in range(0, len(order) - slice_len + 1, slice_len))


def test_windowed_sampler_none_is_exact_global_shuffle() -> None:
    # window=None is the default and MUST be byte-identical to the plain global randperm so it
    # never silently changes training statistics.
    mapping = _case_major_mapping(6, 4)
    sampler = WindowedCaseSampler(mapping, shuffle=True, window=None, batch_size=2, num_workers=1)
    torch.manual_seed(2024)
    got = list(iter(sampler))
    torch.manual_seed(2024)
    expected = torch.randperm(len(mapping)).tolist()
    assert got == expected
    assert len(sampler) == len(mapping)


def test_windowed_sampler_full_window_degenerates_to_global_shuffle() -> None:
    # window == n_cases is the compat escape hatch: it degenerates EXACTLY to the global shuffle.
    mapping = _case_major_mapping(6, 4)
    n_cases = 6
    sampler = WindowedCaseSampler(mapping, shuffle=True, window=n_cases, batch_size=2, num_workers=1)
    torch.manual_seed(11)
    got = list(iter(sampler))
    torch.manual_seed(11)
    expected = torch.randperm(len(mapping)).tolist()
    assert got == expected
    # An oversized window is also the global shuffle (no windowing).
    sampler_big = WindowedCaseSampler(mapping, shuffle=True, window=n_cases + 5, batch_size=2, num_workers=1)
    torch.manual_seed(11)
    assert list(iter(sampler_big)) == expected


def test_windowed_sampler_keeps_a_bounded_set_of_cases_resident() -> None:
    mapping = _case_major_mapping(12, 5)
    for window in (1, 2, 3):
        sampler = WindowedCaseSampler(mapping, shuffle=True, window=window, batch_size=2, num_workers=1)
        order = list(iter(sampler))
        # Every original patch is represented and only bounded padding duplicates are added.
        assert set(order) >= set(range(len(mapping)))
        assert len(order) - len(set(order)) <= sampler.batch_size
        # A window slice (window cases * patches_per_case) touches at most `window` distinct cases.
        assert _distinct_cases_per_slice(order, mapping, window * 5) <= window


def test_windowed_sampler_epoch_length_is_equal_across_ddp_ranks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every rank must run the same number of batches, whatever cases its shard happens to hold.

    ``Data._split`` pads the shards to equal length precisely because DDP(static_graph=True) hangs
    when the ranks disagree on the batch count. A length read from the per-rank case-to-worker
    partition undoes that: the shards carry the same NUMBER of patches but different cases, so the
    partitions -- and the epoch -- come out different sizes.
    """
    monkeypatch.setenv("KONFAI_STATE", str(State.TRAIN))
    # Cases of increasing size, as a real dataset's volumes are: the shards then carry the same
    # patch COUNT but different cases, which is what makes the partitions -- and the length read
    # from them -- differ. A symmetric distribution hides this.
    mapping = [(case, 0, patch) for case, count in enumerate([1, 2, 3, 4, 5, 6, 7, 8]) for patch in range(count)]
    shards = Data._split(mapping, 2)
    assert len({len(shard) for shard in shards}) == 1, "precondition: _split equalises shard length"

    lengths = {len(WindowedCaseSampler(shard, shuffle=True, window=2, batch_size=2, num_workers=2)) for shard in shards}

    assert len(lengths) == 1, f"ranks disagree on the epoch length: {lengths}"


def test_windowed_sampler_shards_cases_across_workers_without_overlap() -> None:
    # A map-style DataLoader sends batch j to worker j % num_workers, and each worker holds its own
    # buffer. The sampler must therefore give each batch only its worker-partition's cases so a case
    # is never loaded by more than one worker (no num_workers-fold RAM/I/O blow-up).
    mapping = _case_major_mapping(16, 4)
    for num_workers in (2, 4):
        sampler = WindowedCaseSampler(mapping, shuffle=True, window=2, batch_size=2, num_workers=num_workers)
        order = list(iter(sampler))
        batches = [order[i : i + 2] for i in range(0, len(order), 2)]
        cases = list(sampler.case_entries.keys())
        partition_of = {case: position % num_workers for position, case in enumerate(cases)}
        worker_cases: dict[int, set[int]] = {w: set() for w in range(num_workers)}
        for batch_index, batch in enumerate(batches):
            for sample_index in batch:
                case = mapping[sample_index][0]
                # every sample in batch j belongs to partition j % num_workers
                assert partition_of[case] == batch_index % num_workers
                worker_cases[batch_index % num_workers].add(case)
        for a in range(num_workers):
            for b in range(a + 1, num_workers):
                assert worker_cases[a].isdisjoint(worker_cases[b])


class _WholeVolumeDataset:
    """In-memory dataset whose patches are non-streamable (forces the FIFO case-load path)."""

    def __init__(self, volume: np.ndarray) -> None:
        self.volume = volume

    def get_infos(self, group_src: str, name: str) -> tuple[list[int], Attribute]:
        return list(self.volume.shape), Attribute({"name": name})

    def read_data(self, group_src: str, name: str) -> tuple[np.ndarray, Attribute]:
        return self.volume.copy(), Attribute({"name": name})


def _reload_count(order: list[int], mapping: list[tuple[int, int, int]], n_cases: int, buffer_size: int) -> int:
    dataset = cast(Dataset, _WholeVolumeDataset(np.zeros((1, 8, 8), dtype=np.float32)))
    augmentations = DataAugmentationsList(nb=0, data_augmentations={})
    augmentation = _CountingOffsetAugmentation()
    augmentation.load(1.0)
    augmentations.data_augmentations = [augmentation]
    managers = [
        DatasetManager(
            index=i,
            group_src="src",
            group_dest="dest",
            name=f"case_{i:03d}",
            dataset=dataset,
            patch=DatasetPatch([4, 4]),
            transforms=[_WholeVolumeTransform()],
            data_augmentations_list=[augmentations],
        )
        for i in range(n_cases)
    ]
    dataset_iter = DatasetIter(
        rank=0,
        data={"dest": managers},
        mapping=mapping,
        groups_src={"src": Group(groups_dest={"dest": GroupTransform(transforms=None, patch_transforms=None)})},
        inline_augmentations=False,
        data_augmentations_list=[augmentations],
        patch_size=[4, 4],
        overlap=None,
        buffer_size=buffer_size,
        use_cache=False,
    )
    reloads = {"n": 0}
    original = dataset_iter._load_data

    def counting_load(index: int, augmentation_index: int | None = None) -> bool:
        loaded = original(index, augmentation_index)
        if loaded:
            reloads["n"] += 1
        return loaded

    dataset_iter._load_data = counting_load  # type: ignore[method-assign]
    for sample_index in order:
        dataset_iter[sample_index]
    return reloads["n"]


def test_windowed_sampler_reaches_one_read_per_case() -> None:
    # The whole point: a windowed epoch loads each volume ~once, versus many times for global shuffle.
    n_cases, patches_per_case = 10, 4
    mapping = _case_major_mapping(n_cases, patches_per_case)

    torch.manual_seed(0)
    global_order = WindowedCaseSampler(mapping, shuffle=True, window=None, batch_size=2, num_workers=1)
    global_reloads = _reload_count(list(iter(global_order)), mapping, n_cases, buffer_size=3)

    windowed = WindowedCaseSampler(mapping, shuffle=True, window=2, batch_size=2, num_workers=1)
    windowed_reloads = _reload_count(list(iter(windowed)), mapping, n_cases, buffer_size=max(3, 2))

    # Global shuffle thrashes (well above one read per case); the window reads each case exactly once.
    assert global_reloads > n_cases
    assert windowed_reloads == n_cases


def test_prediction_subset_order_stays_case_major_and_unwindowed() -> None:
    # The prediction path uses shuffle=False. The sampler must emit the identity (case-major) order and
    # ignore any window, so the prediction buffer keeps hitting ~100% and stays byte-identical.
    mapping = _case_major_mapping(5, 3)
    prediction = PredictionSubset()
    assert prediction.shuffle is False
    assert prediction.shuffle_window is None
    sampler = WindowedCaseSampler(mapping, shuffle=prediction.shuffle, window=None, batch_size=1, num_workers=4)
    assert list(iter(sampler)) == list(range(len(mapping)))
    # A window is inert once shuffle is off: still the case-major identity order.
    windowed = WindowedCaseSampler(mapping, shuffle=False, window=2, batch_size=1, num_workers=4)
    assert list(iter(windowed)) == list(range(len(mapping)))


def test_train_subset_exposes_shuffle_window_knob() -> None:
    # The knob is a plain constructor argument so the reflection config engine can bind it.
    default = TrainSubset()
    assert default.shuffle_window is None
    configured = TrainSubset(shuffle_window=4)
    assert configured.shuffle_window == 4
    assert configured.shuffle is True


def test_the_windowed_order_is_a_permutation_of_the_epoch() -> None:
    # An epoch is one pass over the mapping: the window chooses the order, never the contents. Cases
    # differ in patch count and the partitions are cut by case, so the per-worker streams are uneven
    # by nature -- padding the short ones up to the longest and cutting the result back to length
    # keeps the length right while dropping and repeating almost half of an uneven epoch.
    mapping = [(case, patch, 0) for case in range(12) for patch in range(200 if case < 2 else 2)]
    sampler = WindowedCaseSampler(mapping, shuffle=True, window=4, batch_size=2, num_workers=4)
    order = list(iter(sampler))
    assert len(order) == len(mapping) == len(sampler)
    assert sorted(order) == list(range(len(mapping)))


def test_a_window_keeps_a_worker_reading_each_volume_once() -> None:
    # What the window is for: a case's patches are walked while it is resident, so the FIFO reads it
    # once an epoch rather than once per eviction.
    mapping = [(case, patch, 0) for case in range(24) for patch in range(10)]
    sampler = WindowedCaseSampler(mapping, shuffle=True, window=4, batch_size=2, num_workers=4)
    order = list(iter(sampler))
    for worker in range(4):
        cases = [mapping[index][0] for position, index in enumerate(order) if (position // 2) % 4 == worker]
        resident: list[int] = []
        loads = 0
        for case in cases:
            if case not in resident:
                loads += 1
                resident.append(case)
                if len(resident) > 4:
                    resident.pop(0)
            else:
                resident.append(resident.pop(resident.index(case)))
        assert loads == len(set(cases))


@pytest.mark.parametrize("entries, world_size", [(8, 4), (4, 4), (3, 4), (1, 4)])
def test_every_rank_gets_a_shard_of_the_same_length(entries: int, world_size: int, monkeypatch) -> None:
    # DDP(static_graph=True) needs every rank to run the same number of backward all-reduces. A shard
    # fills itself from its own head, and one holding nothing has no head: fewer entries than ranks
    # left it empty, and an empty rank runs no backward at all -- the hang this equalises against.
    monkeypatch.setenv("KONFAI_STATE", "TRAIN")
    mapping = [(index, 0, 0) for index in range(entries)]
    shards = Data._split(mapping, world_size)
    assert len({len(shard) for shard in shards}) == 1
    assert all(shard for shard in shards)
