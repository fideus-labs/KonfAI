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

"""Tests for ``konfai.data.data_manager``: DDP sharding, train/validation split, prediction
subsets, DataLoader arguments, cache workers, and DatasetIter (streaming transforms, inline
augmentations)."""

import multiprocessing
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch
from konfai.data.augmentation import DataAugmentation, DataAugmentationsList
from konfai.data.data_manager import (
    BatchDataItem,
    Data,
    DataItem,
    DataPrediction,
    DatasetIter,
    DataTrain,
    Group,
    GroupTransform,
    PatchReadOrder,
    PredictionSubset,
    Subset,
    WindowedCaseSampler,
    collate_konfai,
)
from konfai.data.data_manager.samples import _cache_worker_count
from konfai.data.patching import DatasetManager, DatasetPatch
from konfai.data.transform import Gradient, TensorCast, Transform, TransformLoader
from konfai.utils.clock import restart_startup_clock
from konfai.utils.dataset import Attribute, Dataset
from konfai.utils.errors import DatasetManagerError
from konfai.utils.runtime import State
from konfai.utils.utils import split_path_spec
from oracle_support import geometry
from torch.utils.data._utils.pin_memory import pin_memory as torch_pin_memory

# --------------------------------------------------------------------------------------
# Data._split. TRAIN/RESUME shards must be equal length to avoid a DDP hang
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
# Data._split. PREDICTION/EVALUATION shards must keep every case whole on one rank
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


def test_data_split_prediction_keeps_case_patches_together_and_allows_empty_shards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KONFAI_STATE", str(State.PREDICTION))

    shards = Data._split(
        [(0, 0, 0), (0, 0, 1), (1, 0, 0), (2, 0, 0), (2, 0, 1)],
        4,
    )

    # Whole cases dealt largest-first onto the least-loaded rank: the two 2-patch cases land on the
    # first two ranks, the 1-patch case on the third, and the spare rank stays empty.
    assert shards == [
        [(0, 0, 0), (0, 0, 1)],
        [(2, 0, 0), (2, 0, 1)],
        [(1, 0, 0)],
        [],
    ]


def test_one_pass_split_balances_ranks_by_patch_load_and_restarts_identically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An equal-count contiguous split lands a [1000, 10, 10, 10]-patch cohort as 1010 vs 20 on two
    ranks and idles one GPU behind the other; balancing by patch load lands 1000 vs 30."""
    monkeypatch.setenv("KONFAI_STATE", str(State.PREDICTION))
    counts = [1000, 10, 10, 10]
    mapping = [(case, 0, patch) for case, count in enumerate(counts) for patch in range(count)]

    shards = Data._split(mapping, 2)

    assert sorted(len(shard) for shard in shards) == [30, 1000]
    owner: dict[int, int] = {}
    for rank, shard in enumerate(shards):
        for entry in shard:
            assert owner.setdefault(entry[0], rank) == rank, f"case {entry[0]} split across ranks"
        # Within a shard each case keeps its entries in mapping order (read order == write order).
        for case in {entry[0] for entry in shard}:
            assert [entry for entry in shard if entry[0] == case] == [entry for entry in mapping if entry[0] == case]
    # Deterministic: a restart shards identically.
    assert Data._split(mapping, 2) == shards


def test_train_split_stays_contiguous_and_untouched_by_the_one_pass_balancer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # TRAIN shuffles the mapping anyway and its DDP contract is equal-LENGTH shards: the balanced
    # per-case deal is a one-pass-only change.
    monkeypatch.setenv("KONFAI_STATE", str(State.TRAIN))
    counts = [1000, 10, 10, 10]
    mapping = [(case, 0, patch) for case, count in enumerate(counts) for patch in range(count)]

    shards = Data._split(mapping, 2)

    assert shards == [mapping[:515], mapping[515:]]


def test_data_remap_dataset_indices_compacts_sparse_mapping_indices() -> None:
    indices, remapped = Data._remap_dataset_indices([(3, 0, 0), (3, 0, 1), (8, 1, 0), (3, 1, 2)])

    assert indices == [3, 8]
    assert remapped == [(0, 0, 0), (0, 0, 1), (1, 1, 0), (0, 1, 2)]


# --------------------------------------------------------------------------------------
# DataTrain train/validation split: reproducible and seeded from sorted names
# --------------------------------------------------------------------------------------

_SPLIT_PROBE = """
import random

from konfai.data.data_manager import DataTrain

names = [f"CASE_{i:03d}" for i in range(20)]
data = DataTrain(augmentations=None, validation="0:4")
data._resolve_dataset_sources = lambda requested: {}
data._resolve_common_names = lambda datasets, requested: ({}, set(names))
data._get_datasets = lambda case_names, dataset_name, augmentations, index_offset=0, managers=None: ({}, [])
random.seed(1234)
data._prepare_datasets()
print(";".join(data.case_names))
print(";".join(data._validation_names))
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

    monkeypatch.setattr("konfai.data.data_manager.sources.random.sample", fake_sample)

    data = DataTrain(augmentations=None, validation="0:2")
    names = {"CASE_010", "CASE_002", "CASE_001", "CASE_005", "CASE_003"}
    data._resolve_dataset_sources = lambda requested: {}
    data._resolve_common_names = lambda datasets, requested: ({}, names)
    data._get_datasets = lambda case_names, dataset_name, augmentations, index_offset=0, managers=None: ({}, [])
    data._prepare_datasets()

    assert captured["population"] == sorted(names)
    assert data._validation_names == ["CASE_010", "CASE_005"]
    assert data.case_names == ["CASE_003", "CASE_002", "CASE_001"]


def test_data_train_validation_accepts_mixed_case_names_and_case_files(tmp_path: Path) -> None:
    validation_file = tmp_path / "validation.txt"
    validation_file.write_text("CASE_001\nCASE_003\n", encoding="utf-8")
    dataset = DataTrain(
        augmentations=None,
        validation=[str(validation_file), "CASE_002"],
    )

    train_names, validation_names = dataset._split_train_validation_names(
        ["CASE_000", "CASE_001", "CASE_002", "CASE_003"],
    )

    assert train_names == ["CASE_000"]
    assert validation_names == ["CASE_001", "CASE_002", "CASE_003"]


@pytest.mark.parametrize(
    "selector",
    [
        "1:3",
        "0:-2",
        "CASE_001",
        ["CASE_001", "CASE_002"],
        ["~CASE_001"],
        [0, "CASE_002"],
        "file",
        ["file", "CASE_002"],
    ],
    ids=["slice", "negative-slice", "name", "names", "exclusion", "mixed", "file", "file-and-name"],
)
def test_subset_and_validation_accept_the_same_selector_spellings(tmp_path: Path, selector) -> None:
    """'validation:' resolves through the same selector grammar as 'subset:': one set of spellings
    to learn, and every fix or extension lands on both keys at once."""
    names = ["CASE_000", "CASE_001", "CASE_002", "CASE_003"]
    fold = tmp_path / "fold.txt"
    fold.write_text("CASE_001\nCASE_003\n", encoding="utf-8")

    def resolve(spelling):
        return str(fold) if spelling == "file" else spelling

    selector = [resolve(s) for s in selector] if isinstance(selector, list) else resolve(selector)

    kept_by_subset = Subset(selector)(list(names), {})
    dataset = DataTrain(augmentations=None, validation=selector)
    train_names, validation_names = dataset._split_train_validation_names(list(names))

    assert set(validation_names) == kept_by_subset
    assert sorted(train_names + validation_names) == names


def test_a_negative_slice_end_counts_from_the_end_python_style() -> None:
    # '0:-2' once clipped to an empty range and blamed the subset as "too restrictive".
    names = [f"CASE_{index:03d}" for index in range(5)]
    assert Subset("0:-2")(names, {}) == {"CASE_000", "CASE_001", "CASE_002"}
    assert Subset("-2:5")(names, {}) == {"CASE_003", "CASE_004"}


def test_an_unresolvable_validation_selector_is_refused_with_the_accepted_spellings() -> None:
    dataset = DataTrain(augmentations=None, validation="no_such_case_or_file")

    with pytest.raises(DatasetManagerError, match="same selector spellings as 'subset'"):
        dataset._split_train_validation_names(["CASE_000", "CASE_001"])


def test_a_validation_list_with_an_unsupported_element_type_is_refused() -> None:
    dataset = DataTrain(augmentations=None, validation=[0.5, "CASE_000"])

    with pytest.raises(DatasetManagerError, match="Invalid list type"):
        dataset._split_train_validation_names(["CASE_000", "CASE_001"])


def test_data_train_validation_none_keeps_full_dataset_for_training() -> None:
    dataset = DataTrain(
        augmentations=None,
        validation=None,
    )

    train_names, validation_names = dataset._split_train_validation_names(
        ["CASE_000", "CASE_001", "CASE_002"],
    )

    assert train_names == ["CASE_000", "CASE_001", "CASE_002"]
    assert validation_names == []


def test_data_train_validation_augmentations_can_be_disabled() -> None:
    augmentations = DataAugmentationsList(nb=2, data_augmentations={})
    dataset = DataTrain(
        augmentations={"DataAugmentation_0": augmentations},
        validation_augmentations=False,
    )
    dataset._prepared_validation_mapping = [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 2, 0)]

    validation_mapping = dataset._get_validation_mapping()

    assert validation_mapping == [(0, 0, 0), (1, 0, 0)]


def test_data_train_prepare_skips_validation_augmentation_layout_when_disabled(tmp_path: Path) -> None:
    pytest.importorskip("SimpleITK")
    dataset_path = tmp_path / "Dataset"
    dataset_storage = Dataset(dataset_path, "mha")
    volume = np.arange(1 * 4 * 4, dtype=np.float32).reshape(1, 4, 4)
    dataset_storage.write("CT", "CASE_000", volume, _image_attributes([0.0, 0.0], [1.0, 1.0]))
    dataset_storage.write("CT", "CASE_001", volume, _image_attributes([0.0, 0.0], [1.0, 1.0]))

    augmentations = DataAugmentationsList(nb=1, data_augmentations={})
    dataset = DataTrain(
        dataset_filenames=[f"{dataset_path}:mha"],
        groups_src={"CT": Group(groups_dest={"CT": GroupTransform(transforms=None, patch_transforms=None)})},
        augmentations={"DataAugmentation_0": augmentations},
        patch=None,
        validation=["CASE_001"],
        validation_augmentations=False,
    )

    dataset.prepare()

    assert dataset.managers is not None
    assert dataset._validation_managers is not None
    assert dataset.managers["CT"][0].total_augmentations == 1
    assert dataset._validation_managers["CT"][0].total_augmentations == 0


@pytest.mark.parametrize(("validation_augmentations", "built"), [(True, 20), (False, 24)])
def test_a_float_split_builds_each_case_once_and_cuts_the_partitions_from_that_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, validation_augmentations: bool, built: int
) -> None:
    """The float split shares out patch counts, which only a built manager knows. Every case is
    built once, in run order with the training draws, and the partitions are that build's head and
    tail: same names, same indices, one draw per case. Validation without the draws is rebuilt
    (2 cases x 2 groups more), never cut with them."""
    pytest.importorskip("SimpleITK")
    from konfai.data.augmentation import Prob

    names = [f"CASE_{index:03d}" for index in range(10)]
    store = Dataset(tmp_path / "Dataset", "mha")
    for name in names:
        for group in ("CT", "SEG"):
            store.write(group, name, np.zeros((1, 4, 4), np.float32), _image_attributes([0.0, 0.0], [1.0, 1.0]))
    constructed: list[tuple[str, int]] = []

    class CountingManager(DatasetManager):
        def __init__(self, index: int, group_src: str, group_dest: str, name: str, *args, **kwargs) -> None:
            constructed.append((name, index))
            super().__init__(index, group_src, group_dest, name, *args, **kwargs)

    monkeypatch.setattr("konfai.data.data_manager.sources.DatasetManager", CountingManager)
    monkeypatch.delenv("KONFAI_config_file")  # the draw binds its own defaults
    monkeypatch.setenv("KONFAI_ROOT", "Trainer")
    augmentations = DataAugmentationsList(nb=1, data_augmentations={"Flip": Prob(1)})
    dataset = DataTrain(
        dataset_filenames=[f"{tmp_path / 'Dataset'}:mha"],
        groups_src={
            group: Group(groups_dest={group: GroupTransform(transforms=None, patch_transforms=None)})
            for group in ("CT", "SEG")
        },
        augmentations={"DataAugmentation_0": augmentations},
        patch=None,
        subset=Subset(shuffle=False),
        validation=0.2,
        validation_augmentations=validation_augmentations,
    )
    torch.manual_seed(0)
    clock = restart_startup_clock()
    dataset.prepare()

    assert clock.spent("cases") > 0 and clock.spent("grids") > 0  # the startup line's phases
    assert len(constructed) == built
    assert (dataset.case_names, dataset._validation_names) == (names[:8], names[8:])
    assert [manager.index for manager in dataset.managers["SEG"]] == list(range(8))
    assert [(manager.name, manager.index) for manager in dataset._validation_managers["SEG"]] == [
        ("CASE_008", 8),
        ("CASE_009", 9),
    ]
    assert dataset._validation_managers["CT"][0].total_augmentations == int(validation_augmentations)
    flip = augmentations.data_augmentations[0]
    assert sorted(flip.who_index) == list(range(10))  # one draw per case, keyed by its run index


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
    return geometry(origin, spacing)


def test_streaming_tensorcast_persists_source_dtype_for_inverse(streaming_dataset_stub) -> None:
    volume = np.arange(1 * 4 * 4, dtype=np.int16).reshape(1, 4, 4)
    dataset_stub = streaming_dataset_stub(volume)
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


def test_manager_patch_copy_owns_its_patch_size(streaming_dataset_stub) -> None:
    """The copy's grid is cut lazily, so it must not read a list its source can still change."""
    shared = DatasetPatch([4, 4])
    manager = DatasetManager(
        index=0,
        group_src="CT",
        group_dest="CT",
        name="CASE_000",
        dataset=cast(Dataset, streaming_dataset_stub(np.arange(1 * 8 * 8, dtype=np.float32).reshape(1, 8, 8))),
        patch=shared,
        transforms=[],
        data_augmentations_list=[],
    )
    shared.patch_size[:] = [8, 8]  # before any cut: nothing has read the copy's sizes yet

    assert manager.patch.patch_size == [4, 4]
    assert len(manager.patch.get_patch_slices(0)) == 4  # 8x8 in 4x4 patches, not one whole-volume patch


# --------------------------------------------------------------------------------------
# DatasetIter: inline augmentations and per-case state draws
# --------------------------------------------------------------------------------------


class _WholeVolumeTransform(Transform):
    """A spatial identity that declares nothing, so its chain can only run on a whole volume.

    Cases here are about what happens once a volume is resident (the FIFO buffer, the augmentation
    draws), so they need a chain the streamer refuses. Declaring it is how a chain says so.
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
# Destination groups must agree on the patch grid: the mapping is counted on ONE of them
# --------------------------------------------------------------------------------------


def _plain_manager(group_dest: str, array: np.ndarray) -> DatasetManager:
    return DatasetManager(
        index=0,
        group_src="src",
        group_dest=group_dest,
        name="case_000",
        dataset=cast(Dataset, _DummyDataset(array)),
        patch=DatasetPatch([4, 4]),
        transforms=[],
        data_augmentations_list=[],
    )


def test_cross_group_patch_count_check_names_the_case_the_groups_and_their_shapes() -> None:
    agreeing = {
        "A": [_plain_manager("A", np.zeros((1, 8, 8), np.float32))],
        "B": [_plain_manager("B", np.zeros((1, 8, 8), np.float32))],
    }
    Data._check_cross_group_patch_counts(agreeing, 1)  # same grids: no refusal

    disagreeing = {
        "A": [_plain_manager("A", np.zeros((1, 8, 8), np.float32))],
        "B": [_plain_manager("B", np.zeros((1, 8, 4), np.float32))],
    }
    with pytest.raises(DatasetManagerError) as refusal:
        Data._check_cross_group_patch_counts(disagreeing, 1)

    message = str(refusal.value)
    assert "case_000" in message and "'A'" in message and "'B'" in message
    assert "[8, 8]" in message and "[8, 4]" in message


def test_destination_groups_with_disagreeing_grids_are_refused_at_prepare(tmp_path: Path) -> None:
    """Two chains folding a case to different grids used to surface as an IndexError deep in a
    loader worker (last group counted larger) or as silently unenumerated patches (smaller): the
    disagreement is a config error and must be refused before a single patch is read."""
    pytest.importorskip("SimpleITK")
    store = Dataset(tmp_path / "Dataset", "mha")
    store.write("CT", "CASE_000", np.zeros((1, 8, 8), np.float32), _image_attributes([0.0, 0.0], [1.0, 1.0]))
    store.write("SEG", "CASE_000", np.zeros((1, 8, 4), np.float32), _image_attributes([0.0, 0.0], [1.0, 1.0]))
    dataset = DataPrediction(
        augmentations=None,
        dataset_filenames=[f"{tmp_path / 'Dataset'}:mha"],
        groups_src={
            group: Group(groups_dest={group: GroupTransform(transforms=None, patch_transforms=None)})
            for group in ("CT", "SEG")
        },
        patch=DatasetPatch(patch_size=[4, 4], overlap=None),
        subset=PredictionSubset(),
    )

    with pytest.raises(DatasetManagerError, match="disagree on the patch grid"):
        dataset.prepare()


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
    partitions (and the epoch) come out different sizes.
    """
    monkeypatch.setenv("KONFAI_STATE", str(State.TRAIN))
    # Cases of increasing size, as a real dataset's volumes are: the shards then carry the same
    # patch COUNT but different cases, which is what makes the partitions, and the length read
    # from them: differ. A symmetric distribution hides this.
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


# --------------------------------------------------------------------------------------
# PatchReadOrder: the epoch's order, from where it is drawn to where the patches are read
# --------------------------------------------------------------------------------------


def _published(mapping: list[tuple[int, int, int]], order: list[int], batch_size: int = 1) -> PatchReadOrder:
    read_order = PatchReadOrder(mapping, batch_size)
    read_order.publish(torch.as_tensor(order, dtype=torch.int64))
    return read_order


def test_a_case_is_declared_once_per_epoch_in_the_order_its_patches_will_be_read() -> None:
    """A schedule is followed only while the reads match it, so what a case declares must be the
    sequence the loader is about to ask for, and only the part of it still to come."""
    mapping = _case_major_mapping(2, 3)
    order = [4, 0, 3, 2, 5, 1]

    read_order = _published(mapping, order)

    assert read_order.entering(4) == [(0, 1), (0, 0), (0, 2)], "case 1's patches, as they will come"
    assert read_order.entering(0) == [(0, 0), (0, 2), (0, 1)], "case 0's, from where it is entered"
    assert [read_order.entering(index) for index in order[2:]] == [None] * 4, "and never twice"


def test_a_second_epoch_declares_the_order_it_was_published_and_not_the_first_one() -> None:
    """A persistent worker outlives the epoch: an order it kept would declare the previous draw."""
    mapping = _case_major_mapping(1, 2)
    read_order = _published(mapping, [0, 1])
    assert read_order.entering(0) == [(0, 0), (0, 1)]

    read_order.publish(torch.as_tensor([1, 0], dtype=torch.int64))

    assert read_order.entering(1) == [(0, 1), (0, 0)]


def test_a_worker_declares_the_batches_it_is_handed_and_not_the_others(monkeypatch: pytest.MonkeyPatch) -> None:
    """A DataLoader deals one batch in num_workers to each worker, and each holds a cache of its
    own: a worker declaring the whole epoch would deviate from its own first read."""
    monkeypatch.setattr(torch.utils.data, "get_worker_info", lambda: SimpleNamespace(id=0, num_workers=2))
    mapping = _case_major_mapping(4, 2)

    read_order = _published(mapping, list(range(len(mapping))), batch_size=2)

    assert read_order.entering(0) == [(0, 0), (0, 1)], "batch 0 is this worker's"
    assert read_order.entering(4) == [(0, 0), (0, 1)], "and so is batch 2"
    assert read_order.entering(2) is None, "batch 1 goes to the other worker"


def _entering_at_each_draw(read_order: PatchReadOrder, draws, opening, answer) -> None:
    for drawn in draws:
        drawn.wait(30)
        answer.put(read_order.entering(opening.get()))


@pytest.mark.skipif("fork" not in multiprocessing.get_all_start_methods(), reason="no fork on this platform")
def test_every_epoch_s_order_reaches_a_process_forked_before_the_first_draw() -> None:
    """A DataLoader forks its workers before the sampler draws the epoch's order and, when they are
    persistent, never forks them again: an order held in ordinary memory would reach neither the
    first epoch nor the ones after it."""
    mapping = _case_major_mapping(1, 6)
    read_order = PatchReadOrder(mapping, batch_size=1)
    sampler = WindowedCaseSampler(mapping, True, None, 1, 1, read_order)
    context = multiprocessing.get_context("fork")
    draws = [context.Event(), context.Event()]
    opening, answer = context.SimpleQueue(), context.SimpleQueue()
    child = context.Process(target=_entering_at_each_draw, args=(read_order, draws, opening, answer))
    child.start()

    drawn: list[list[int]] = []
    declared: list[list[tuple[int, int]]] = []
    for epoch, event in enumerate(draws):
        torch.manual_seed(epoch)
        drawn.append(list(iter(sampler)))
        opening.put(drawn[-1][0])
        event.set()
        declared.append(answer.get())
    child.join(60)

    assert child.exitcode == 0
    assert drawn[0] != drawn[1], "the two epochs drew the same order: the test would prove nothing"
    assert declared == [[mapping[index][1:] for index in order] for order in drawn]


@pytest.mark.parametrize("transforms", [[], [Gradient()]], ids=["exact-patch", "halo"])
def test_the_windows_declared_are_the_windows_the_patch_route_then_reads(
    streaming_dataset_stub, transforms: list[Transform]
) -> None:
    """The declaration is named from the plans and the read from the run: a window that misses what
    the read asks for deviates the schedule at its first step and buys nothing, in silence."""
    dataset_stub = streaming_dataset_stub(np.arange(1 * 8 * 8, dtype=np.float32).reshape(1, 8, 8))
    manager = DatasetManager(
        index=0,
        group_src="CT",
        group_dest="CT",
        name="CASE_000",
        dataset=cast(Dataset, dataset_stub),
        patch=DatasetPatch([4, 4]),
        transforms=transforms,
        data_augmentations_list=[],
    )
    mapping = [(0, 0, index) for index in range(manager.get_size(0))]
    dataset_iter = DatasetIter(
        rank=0,
        data={"CT": [manager]},
        mapping=mapping,
        groups_src={"CT": Group(groups_dest={"CT": GroupTransform(transforms=None, patch_transforms=None)})},
        inline_augmentations=False,
        data_augmentations_list=[],
        patch_size=[4, 4],
        overlap=None,
        buffer_size=1,
        use_cache=False,
    )
    sampler = WindowedCaseSampler(mapping, False, None, 1, 1, dataset_iter.read_order)

    for index in sampler:
        dataset_iter[index]

    assert dataset_stub.declared, "the case declared nothing"
    assert dataset_stub.regions == dataset_stub.declared


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


def test_subset_exposes_shuffle_window_knob() -> None:
    # The knob is a plain constructor argument so the reflection config engine can bind it.
    default = Subset()
    assert default.shuffle_window is None
    configured = Subset(shuffle_window=4)
    assert configured.shuffle_window == 4
    assert configured.shuffle is True


def test_the_windowed_order_is_a_permutation_of_the_epoch() -> None:
    # An epoch is one pass over the mapping: the window chooses the order, never the contents. Cases
    # differ in patch count and the partitions are cut by case, so the per-worker streams are uneven
    # by nature: padding the short ones up to the longest and cutting the result back to length
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
    # left it empty, and an empty rank runs no backward at all: the hang this equalises against.
    monkeypatch.setenv("KONFAI_STATE", "TRAIN")
    mapping = [(index, 0, 0) for index in range(entries)]
    shards = Data._split(mapping, world_size)
    assert len({len(shard) for shard in shards}) == 1
    assert all(shard for shard in shards)


# --------------------------------------------------------------------------------------
# DataLoader arguments: worker prefetch and persistent workers per workflow
# --------------------------------------------------------------------------------------


def test_data_train_enables_worker_prefetch_when_cache_is_disabled() -> None:
    # The budget resolver flips the regime through this same re-entry point.
    dataset = DataTrain(augmentations=None)
    dataset._configure_data_loading(use_cache=False)

    assert cast(int, dataset.dataLoader_args["num_workers"]) >= 1
    assert dataset.dataLoader_args["prefetch_factor"] == 2
    assert dataset.dataLoader_args["persistent_workers"] is True


def test_inline_augmentations_disable_persistent_workers() -> None:
    # Persistent workers keep a fork-time copy of the dataset and never see the main process's
    # per-epoch reset_augmentation redraw, so inline augmentations would freeze at their first draw.
    # The guard is inline_augmentations AND a non-empty augmentations config, and it overrides an
    # explicit persistent_workers=True.
    augmentations = {"DataAugmentation_0": DataAugmentationsList(nb=1, data_augmentations={})}

    inline = DataTrain(augmentations=augmentations, inline_augmentations=True, persistent_workers=True)
    inline._configure_data_loading(use_cache=False)
    assert cast(int, inline.dataLoader_args["num_workers"]) >= 1
    assert inline.dataLoader_args["persistent_workers"] is False

    # Preloaded (non-inline) augmentations are drawn in the main process: workers may persist.
    preloaded = DataTrain(augmentations=augmentations, inline_augmentations=False)
    preloaded._configure_data_loading(use_cache=False)
    assert preloaded.dataLoader_args["persistent_workers"] is True

    # The inline flag without any configured augmentation redraws nothing: workers may persist.
    without_augmentations = DataTrain(augmentations=None, inline_augmentations=True)
    without_augmentations._configure_data_loading(use_cache=False)
    assert without_augmentations.dataLoader_args["persistent_workers"] is True


def test_data_prediction_disables_persistent_workers() -> None:
    dataset = DataPrediction(augmentations=None, num_workers=2)

    assert dataset.dataLoader_args["num_workers"] == 2
    assert dataset.dataLoader_args["prefetch_factor"] == 2
    assert dataset.dataLoader_args["persistent_workers"] is False


def _prepared_prediction(root: Path, file_format: str, **kwargs) -> DataPrediction:
    """A prediction over two 8x8 cases stored in ``file_format``, its managers built."""
    store = Dataset(root, file_format)
    for name in ("CASE_000", "CASE_001"):
        store.write("CT", name, np.zeros((1, 8, 8), np.float32), _image_attributes([0.0, 0.0], [1.0, 1.0]))
    dataset = DataPrediction(
        augmentations=None,
        dataset_filenames=[f"{root}:{file_format}"],
        groups_src={"CT": Group(groups_dest={"CT": GroupTransform(transforms=None, patch_transforms=None)})},
        patch=DatasetPatch(patch_size=[4, 4], overlap=None),
        subset=PredictionSubset(),
        **kwargs,
    )
    dataset.prepare()
    return dataset


@pytest.mark.parametrize(
    ("file_format", "spins_workers"),
    [("mha", False), ("nii.gz", True)],
    ids=["a region read decodes the region", "a region read decodes the volume"],
)
def test_prediction_spins_workers_only_where_a_patch_read_decodes_the_volume(
    tmp_path: Path, file_format: str, spins_workers: bool
) -> None:
    """One pass in grid order: a batch costs more through shared memory than read in place, unless
    the store has to decode the whole volume per patch, where the decodes then run in parallel."""
    pytest.importorskip("SimpleITK")

    dataset = _prepared_prediction(tmp_path / file_format, file_format)

    assert (dataset.resolved_num_workers > 0) is spins_workers


def test_an_explicit_worker_count_wins_over_the_read_route(tmp_path: Path) -> None:
    pytest.importorskip("SimpleITK")

    dataset = _prepared_prediction(tmp_path / "mha", "mha", num_workers=3)

    assert dataset.resolved_num_workers == 3


def test_data_prediction_forwards_its_declared_augmentations() -> None:
    """Test-time augmentation IS the ``augmentations:`` section of a prediction config.

    The parameter binds whether or not the dataset passes it on, so dropping it costs no error and
    no warning: every TTA copy simply stops existing, and the run reports a plain single-pass
    prediction that looks entirely normal.
    """
    augmentations = DataAugmentationsList(nb=4)
    dataset = DataPrediction(augmentations={"DataAugmentation_0": augmentations})

    assert dataset.data_augmentations_list == {"DataAugmentation_0": augmentations}


def test_data_prediction_disables_workers_for_konfai_inference_transforms() -> None:
    dataset = DataPrediction(
        augmentations=None,
        groups_src={
            "Volume_0": Group(
                groups_dest={
                    "MASK": GroupTransform(
                        transforms={"KonfAIInference": TransformLoader()},
                        patch_transforms=None,
                    )
                }
            )
        },
    )

    assert dataset.requires_single_process_loading is True
    assert dataset.dataLoader_args["num_workers"] == 0
    assert "prefetch_factor" not in dataset.dataLoader_args
    assert "persistent_workers" not in dataset.dataLoader_args


def test_pin_memory_reaches_the_batch_tensor() -> None:
    # ``torch_pin_memory`` is what both DataLoader iterators call on a collated batch; it hands
    # back untouched anything that is neither a tensor, a mapping nor a sequence. A recording
    # subclass stands in for the pinned tensor so the contract holds without a CUDA device.
    pinned: list[torch.Tensor] = []

    class Recording(torch.Tensor):
        def pin_memory(self, *args: object, **kwargs: object) -> "Recording":
            pinned.append(self)
            return self

    item = BatchDataItem(
        name=["case"],
        tensor=torch.zeros(1, 2, 2).as_subclass(Recording),
        attribute=[Attribute()],
        x=[0],
        a=[0],
        p=[0],
        is_input=True,
    )

    batch = torch_pin_memory({"CT": item})

    assert pinned, "pin_memory: true never reached the batch's tensor"
    assert batch["CT"].name == ["case"]
    assert batch["CT"].is_input is True


# --------------------------------------------------------------------------------------
# collate_konfai: a one-pass singleton batches as a view, a training singleton as a copy
# --------------------------------------------------------------------------------------


def _singleton_sample(tensor: torch.Tensor, aliases_cache: bool) -> dict[str, DataItem]:
    return {"CT": DataItem("case", tensor, Attribute(), 0, 0, 0, True, aliases_cache=aliases_cache)}


def test_collate_batches_a_one_pass_singleton_as_a_view_and_a_training_singleton_as_a_copy() -> None:
    """The stack copy is a whole volume per case on the batch_size=1 evaluation path, outside the
    memory budget's sizing; a training item may alias the epoch-spanning cache, so its copy is what
    protects the cache from any downstream in-place op."""
    tensor = torch.arange(4.0).reshape(1, 2, 2)

    view = collate_konfai([_singleton_sample(tensor, aliases_cache=False)])["CT"].tensor
    assert view.data_ptr() == tensor.data_ptr(), "the one-pass singleton must be a view, not a copy"
    assert view.shape == (1, 1, 2, 2)

    copy = collate_konfai([_singleton_sample(tensor, aliases_cache=True)])["CT"].tensor
    assert copy.data_ptr() != tensor.data_ptr(), "a cache-aliasing singleton must be copied"
    assert torch.equal(copy[0], tensor)


def test_collate_still_copies_inside_a_dataloader_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    # A worker's batch travels by STORAGE, and a patch-view's storage is the whole resident case.
    monkeypatch.setattr(torch.utils.data, "get_worker_info", lambda: SimpleNamespace(id=0, num_workers=2))
    tensor = torch.arange(4.0).reshape(1, 2, 2)

    batched = collate_konfai([_singleton_sample(tensor, aliases_cache=False)])["CT"].tensor

    assert batched.data_ptr() != tensor.data_ptr()


def test_one_pass_loaders_mark_their_samples_as_cache_free() -> None:
    # The flag rides the DatasetIter factory: one-pass workflows read each case once, so their
    # items alias no tensor that is read again; training items may alias the cache.
    from konfai.data.data_manager import DataMetric

    assert DataPrediction(augmentations=None).datasetIter.keywords["single_pass"] is True
    assert DataMetric().datasetIter.keywords["single_pass"] is True
    assert DataTrain(augmentations=None).datasetIter.keywords["single_pass"] is False


def test_dataset_iter_marks_items_from_its_single_pass_flag() -> None:
    def dataset_iter(single_pass: bool) -> DatasetIter:
        manager = DatasetManager(
            index=0,
            group_src="src",
            group_dest="dest",
            name="case_000",
            dataset=cast(Dataset, _DummyDataset(np.zeros((1, 2, 2), np.float32))),
            patch=None,
            transforms=[_WholeVolumeTransform()],
            data_augmentations_list=[],
        )
        return DatasetIter(
            rank=0,
            data={"dest": [manager]},
            mapping=[(0, 0, 0)],
            groups_src={"src": Group(groups_dest={"dest": GroupTransform(transforms=None, patch_transforms=None)})},
            inline_augmentations=False,
            data_augmentations_list=[],
            patch_size=None,
            overlap=None,
            buffer_size=1,
            use_cache=False,
            single_pass=single_pass,
        )

    assert dataset_iter(single_pass=True)[0]["dest"].aliases_cache is False
    assert dataset_iter(single_pass=False)[0]["dest"].aliases_cache is True


# --------------------------------------------------------------------------------------
# PredictionSubset: case selection and common-name resolution
# --------------------------------------------------------------------------------------


def test_prediction_subset_none_selects_full_dataset() -> None:
    subset = PredictionSubset(None)

    selected = subset(["CASE_000", "CASE_001", "CASE_002"], {})

    assert selected == {"CASE_000", "CASE_001", "CASE_002"}


def test_prediction_subset_accepts_explicit_index_lists() -> None:
    subset = PredictionSubset([0, 2])

    selected = subset(["CASE_000", "CASE_001", "CASE_002"], {})

    assert selected == {"CASE_000", "CASE_002"}


def test_prediction_subset_accepts_lists_of_case_files(tmp_path: Path) -> None:
    file_a = tmp_path / "subset_a.txt"
    file_b = tmp_path / "subset_b.txt"
    file_a.write_text("CASE_000\nCASE_002\n", encoding="utf-8")
    file_b.write_text("CASE_001\n", encoding="utf-8")
    subset = PredictionSubset([str(file_a), str(file_b)])

    selected = subset(["CASE_000", "CASE_001", "CASE_002", "CASE_003"], {})

    assert selected == {"CASE_000", "CASE_001", "CASE_002"}


def test_prediction_subset_keeps_tilde_file_exclusion_with_file_lists(tmp_path: Path) -> None:
    include_file = tmp_path / "subset_include.txt"
    exclude_file = tmp_path / "subset_exclude.txt"
    include_file.write_text("CASE_000\nCASE_001\nCASE_002\n", encoding="utf-8")
    exclude_file.write_text("CASE_001\n", encoding="utf-8")
    subset = PredictionSubset([str(include_file), f"~{exclude_file}"])

    selected = subset(["CASE_000", "CASE_001", "CASE_002", "CASE_003"], {})

    assert selected == {"CASE_000", "CASE_002"}


def test_prediction_subset_accepts_windows_style_case_list_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    windows_file = r"C:\tmp\subset_a.txt"
    subset = PredictionSubset([windows_file])

    monkeypatch.setattr(
        "konfai.data.data_manager.sources.os.path.exists",
        lambda path: path == windows_file,
    )
    monkeypatch.setattr(
        PredictionSubset,
        "_read_names_from_file",
        staticmethod(lambda filename: ["CASE_000", "CASE_002"] if filename == windows_file else []),
    )

    selected = subset(["CASE_000", "CASE_001", "CASE_002", "CASE_003"], {})

    assert selected == {"CASE_000", "CASE_002"}


class InfoCountingDataset:
    """A root of two cases that counts the headers it was asked to read."""

    def __init__(self) -> None:
        self.info_calls = 0

    @staticmethod
    def get_names(group: str) -> list[str]:
        assert group == "CT"
        return ["CASE_000", "CASE_001"]

    def select_names(self, group: str, requested: set[str] | None) -> list[str]:
        assert requested is None, "neither subset here names its cases"
        return self.get_names(group)

    def get_infos(self, group: str, name: str) -> tuple[list[int], Attribute]:
        assert group == "CT"
        self.info_calls += 1
        return [1, 2, 2], _image_attributes([0.0, 0.0], [1.0, 1.0])


def test_an_evaluation_keeps_its_roots_across_its_two_resolves(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``DataMetric.prepare`` resolves the sources twice (the sizing pass, then the selection) and a
    re-plan resolves them again: one ``Dataset`` per root serves them all, so a header is parsed once
    and answered from its cache after that."""
    pytest.importorskip("SimpleITK")
    from konfai.data.data_manager import DataMetric, GroupMetric, GroupTransformMetric

    store = Dataset(tmp_path / "Dataset", "mha")
    for index in range(4):
        for group in ("CT", "SEG"):
            store.write(
                group, f"CASE_{index:03d}", np.zeros((1, 4, 4), np.float32), _image_attributes([0.0, 0.0], [1.0, 1.0])
            )
    counts = {"datasets": 0, "parsed": 0, "cached": 0}
    dataset_init, dataset_get_infos = Dataset.__init__, Dataset.get_infos

    def counting_init(self, *args, **kwargs):
        counts["datasets"] += 1
        dataset_init(self, *args, **kwargs)

    def counting_get_infos(self, groups, name):
        counts["cached" if (groups, name) in self._infos_cache else "parsed"] += 1
        return dataset_get_infos(self, groups, name)

    monkeypatch.setattr(Dataset, "__init__", counting_init)
    monkeypatch.setattr(Dataset, "get_infos", counting_get_infos)
    data = DataMetric(
        dataset_filenames=[f"{tmp_path / 'Dataset'}:mha"],
        groups_src={
            group: GroupMetric(groups_dest={group: GroupTransformMetric(transforms=None)}) for group in ("CT", "SEG")
        },
    )
    data.prepare()
    assert counts == {"datasets": 1, "parsed": 8, "cached": 8}
    roots = list(data.datasets.values())
    data.patch = DatasetPatch(patch_size=[2, 2])
    data.replan_patch([4, 4])
    assert counts["datasets"] == 1 and counts["parsed"] == 8
    assert list(data.datasets.values()) == roots  # the same objects, listing and headers included


def test_builtin_subset_does_not_read_infos_during_common_name_resolution() -> None:
    dataset = DataPrediction(
        augmentations=None,
        groups_src={"CT": Group(groups_dest={"CT": GroupTransform(transforms=None, patch_transforms=None)})},
    )
    dataset.datasets = {"fake": cast(Dataset, InfoCountingDataset())}

    dataset_name, subset_names = dataset._resolve_common_names({"CT": [("fake", True)]}, None)

    assert dataset_name["CT"]["fake"] == ["CASE_000", "CASE_001"]
    assert subset_names == {"CASE_000", "CASE_001"}
    assert cast(InfoCountingDataset, dataset.datasets["fake"]).info_calls == 0


def test_custom_subset_can_still_request_infos_during_common_name_resolution() -> None:
    class InfoAwareSubset(PredictionSubset):
        def __init__(self) -> None:
            super().__init__(None)
            self.last_infos: dict[str, tuple[list[int], Attribute]] | None = None

        def __call__(self, names: list[str], infos: dict[str, tuple[list[int], Attribute]]) -> set[str]:
            self.last_infos = infos
            return set(names)

    subset = InfoAwareSubset()
    dataset = DataPrediction(
        augmentations=None,
        subset=subset,
        groups_src={"CT": Group(groups_dest={"CT": GroupTransform(transforms=None, patch_transforms=None)})},
    )
    dataset.datasets = {"fake": cast(Dataset, InfoCountingDataset())}

    _dataset_name, subset_names = dataset._resolve_common_names({"CT": [("fake", True)]}, None)

    assert subset_names == {"CASE_000", "CASE_001"}
    assert subset.last_infos is not None
    assert set(subset.last_infos) == {"CASE_000", "CASE_001"}
    assert cast(InfoCountingDataset, dataset.datasets["fake"]).info_calls == 2


# --------------------------------------------------------------------------------------
# split_path_spec: the "path[:flag]:format" dataset specs the groups are configured with
# --------------------------------------------------------------------------------------
def test_split_path_spec_supports_unix_style_dataset_specs() -> None:
    assert split_path_spec("./Dataset") == ("./Dataset", None, "mha")
    assert split_path_spec("./Dataset:mha") == ("./Dataset", None, "mha")
    assert split_path_spec("./Dataset:a:mha", allowed_flags={"a", "i"}) == ("./Dataset", "a", "mha")
    assert split_path_spec("./Predictions/TRAIN_01/Dataset:i:mha", allowed_flags={"a", "i"}) == (
        "./Predictions/TRAIN_01/Dataset",
        "i",
        "mha",
    )


def test_split_path_spec_supports_windows_paths_without_breaking_drive_letters() -> None:
    assert split_path_spec(r"C:\Dataset") == (r"C:\Dataset", None, "mha")
    assert split_path_spec(r"C:\Dataset:mha") == (r"C:\Dataset", None, "mha")
    assert split_path_spec(r"C:\Dataset:a:mha", allowed_flags={"a", "i"}) == (r"C:\Dataset", "a", "mha")


def test_split_path_spec_keeps_a_uri_whole_including_its_level_suffix() -> None:
    """The split runs from the right, and a scheme's own '://' is two colons on its left: parsed
    without knowing it, 's3://bucket:omezarr@3' resolves to the root 's3' with the flag '//bucket',
    which reads the wrong store, or none."""
    assert split_path_spec("s3://bucket") == ("s3://bucket", None, "mha")
    assert split_path_spec("s3://bucket:omezarr") == ("s3://bucket", None, "omezarr")
    assert split_path_spec("s3://bucket/cohort:omezarr@3") == ("s3://bucket/cohort", None, "omezarr@3")
    assert split_path_spec("memory://cohort:i:omezarr", allowed_flags={"a", "i"}) == (
        "memory://cohort",
        "i",
        "omezarr",
    )


def test_a_case_present_in_two_roots_is_read_from_the_first_and_said_so() -> None:
    """dataset_filenames may name several roots; a case of one group in two of them was read from
    the first in silence, so a stale copy left in one root would be read without a word."""
    with pytest.warns(UserWarning, match="Case 'P001' of group 'CT' is in 'A' and in 'B'"):
        chosen = Data._get_source_filename_by_group({"CT": {"A": ["P001", "P002"], "B": ["P001", "P003"]}})
    assert chosen == {"CT": {"P001": "A", "P002": "A", "P003": "B"}}
