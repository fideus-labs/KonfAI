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

from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
from konfai.data.augmentation import DataAugmentationsList
from konfai.data.data_manager import (
    Data,
    DataPrediction,
    DatasetIter,
    DataTrain,
    Group,
    GroupTransform,
    PredictionSubset,
)
from konfai.data.patching import DatasetManager, DatasetPatch
from konfai.data.transform import KonfAIInference, Mask, Normalize, Standardize, TransformLoader
from konfai.utils.dataset import Attribute, Dataset
from konfai.utils.runtime import State

SimpleITK = pytest.importorskip("SimpleITK")


def _image_attributes(origin: list[float], spacing: list[float]) -> Attribute:
    attributes = Attribute()
    attributes["Origin"] = np.asarray(origin, dtype=np.float64)
    attributes["Spacing"] = np.asarray(spacing, dtype=np.float64)
    attributes["Direction"] = np.eye(len(origin), dtype=np.float64).reshape(-1)
    return attributes


def test_dataset_read_data_slice_h5_reads_only_requested_region(tmp_path: Path) -> None:
    dataset = Dataset(tmp_path / "Volumes", "h5")
    volume = np.arange(1 * 4 * 5, dtype=np.float32).reshape(1, 4, 5)
    dataset.write("CT", "CASE_000", volume, _image_attributes([1.0, 2.0], [0.5, 1.5]))

    patch, _ = dataset.read_data_slice("CT", "CASE_000", (slice(None), slice(1, 3), slice(2, 5)))

    np.testing.assert_array_equal(patch, volume[:, 1:3, 2:5])


def test_dataset_read_data_statistics_h5_returns_global_stats_without_loading_full_array(tmp_path: Path) -> None:
    dataset = Dataset(tmp_path / "Volumes", "h5")
    volume = np.arange(1 * 4 * 5, dtype=np.float32).reshape(1, 4, 5)
    dataset.write("CT", "CASE_000", volume, _image_attributes([1.0, 2.0], [0.5, 1.5]))

    stats = dataset.read_data_statistics("CT", "CASE_000")

    assert stats["min"] == pytest.approx(float(volume.min()))
    assert stats["max"] == pytest.approx(float(volume.max()))
    assert stats["mean"] == pytest.approx(float(volume.mean()))
    assert stats["std"] == pytest.approx(float(volume.std(ddof=1)))


def test_dataset_read_data_slice_sitk_reads_requested_patch_and_updates_origin(tmp_path: Path) -> None:
    dataset = Dataset(tmp_path / "Dataset", "mha")
    volume = np.arange(1 * 4 * 5 * 6, dtype=np.float32).reshape(1, 4, 5, 6)
    origin = [10.0, 20.0, 30.0]
    spacing = [0.5, 1.5, 2.0]
    dataset.write("CT", "CASE_000", volume, _image_attributes(origin, spacing))

    patch, attributes = dataset.read_data_slice(
        "CT",
        "CASE_000",
        (slice(None), slice(1, 3), slice(2, 5), slice(3, 6)),
    )

    np.testing.assert_array_equal(patch, volume[:, 1:3, 2:5, 3:6])
    np.testing.assert_allclose(
        attributes.get_np_array("Origin"),
        np.asarray([origin[0] + 3 * spacing[0], origin[1] + 2 * spacing[1], origin[2] + 1 * spacing[2]]),
    )


class StreamingDatasetStub:
    def __init__(self, volume: np.ndarray) -> None:
        self.volume = volume
        self.full_reads = 0
        self.patch_reads = 0
        self.stats_reads = 0

    def get_infos(self, group_src: str, name: str) -> tuple[list[int], Attribute]:
        return list(self.volume.shape), _image_attributes([0.0, 0.0], [1.0, 1.0])

    def read_data(self, group_src: str, name: str) -> tuple[np.ndarray, Attribute]:
        self.full_reads += 1
        return self.volume.copy(), _image_attributes([0.0, 0.0], [1.0, 1.0])

    def read_data_slice(self, group_src: str, name: str, slices: tuple[slice, ...]) -> tuple[np.ndarray, Attribute]:
        self.patch_reads += 1
        return self.volume[slices].copy(), _image_attributes([0.0, 0.0], [1.0, 1.0])

    def read_data_statistics(
        self,
        group_src: str,
        name: str,
        channels: list[int] | None = None,
    ) -> dict[str, float]:
        self.stats_reads += 1
        data = self.volume if channels is None else self.volume[channels]
        return {
            "min": float(data.min()),
            "max": float(data.max()),
            "mean": float(data.mean()),
            "std": float(data.std(ddof=1)),
        }


def test_dataset_iter_streams_patch_reads_when_cache_disabled() -> None:
    volume = np.arange(1 * 4 * 4, dtype=np.float32).reshape(1, 4, 4)
    dataset_stub = StreamingDatasetStub(volume)
    manager = DatasetManager(
        index=0,
        group_src="CT",
        group_dest="CT",
        name="CASE_000",
        dataset=cast(Dataset, dataset_stub),
        patch=DatasetPatch([2, 2]),
        transforms=[],
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

    assert dataset_stub.full_reads == 0
    assert dataset_stub.patch_reads == 1
    assert manager.loaded is False
    np.testing.assert_array_equal(sample.numpy(), volume[:, 0:2, 2:4])


def test_data_train_enables_worker_prefetch_when_cache_is_disabled() -> None:
    dataset = DataTrain(use_cache=False, augmentations=None)

    assert cast(int, dataset.dataLoader_args["num_workers"]) >= 1
    assert dataset.dataLoader_args["prefetch_factor"] == 2
    assert dataset.dataLoader_args["persistent_workers"] is True


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
        "konfai.data.data_manager.os.path.exists",
        lambda path: path == windows_file,
    )
    monkeypatch.setattr(
        PredictionSubset,
        "_read_names_from_file",
        staticmethod(lambda filename: ["CASE_000", "CASE_002"] if filename == windows_file else []),
    )

    selected = subset(["CASE_000", "CASE_001", "CASE_002", "CASE_003"], {})

    assert selected == {"CASE_000", "CASE_002"}


def test_builtin_subset_does_not_read_infos_during_common_name_resolution() -> None:
    class InfoCountingDataset:
        def __init__(self) -> None:
            self.info_calls = 0

        @staticmethod
        def get_names(group: str) -> list[str]:
            assert group == "CT"
            return ["CASE_000", "CASE_001"]

        def get_infos(self, group: str, name: str) -> tuple[list[int], Attribute]:
            assert group == "CT"
            self.info_calls += 1
            return [1, 2, 2], _image_attributes([0.0, 0.0], [1.0, 1.0])

    dataset = DataPrediction(
        augmentations=None,
        groups_src={"CT": Group(groups_dest={"CT": GroupTransform(transforms=None, patch_transforms=None)})},
    )
    dataset.datasets = {"fake": cast(Dataset, InfoCountingDataset())}

    dataset_name, subset_names = dataset._resolve_common_names({"CT": [("fake", True)]})

    assert dataset_name["CT"]["fake"] == ["CASE_000", "CASE_001"]
    assert subset_names == {"CASE_000", "CASE_001"}
    assert cast(InfoCountingDataset, dataset.datasets["fake"]).info_calls == 0


def test_custom_subset_can_still_request_infos_during_common_name_resolution() -> None:
    class InfoCountingDataset:
        def __init__(self) -> None:
            self.info_calls = 0

        @staticmethod
        def get_names(group: str) -> list[str]:
            assert group == "CT"
            return ["CASE_000", "CASE_001"]

        def get_infos(self, group: str, name: str) -> tuple[list[int], Attribute]:
            assert group == "CT"
            self.info_calls += 1
            return [1, 2, 2], _image_attributes([0.0, 0.0], [1.0, 1.0])

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

    _dataset_name, subset_names = dataset._resolve_common_names({"CT": [("fake", True)]})

    assert subset_names == {"CASE_000", "CASE_001"}
    assert subset.last_infos is not None
    assert set(subset.last_infos) == {"CASE_000", "CASE_001"}
    assert cast(InfoCountingDataset, dataset.datasets["fake"]).info_calls == 2


def test_data_train_validation_accepts_mixed_case_names_and_case_files(tmp_path: Path) -> None:
    validation_file = tmp_path / "validation.txt"
    validation_file.write_text("CASE_001\nCASE_003\n", encoding="utf-8")
    dataset = DataTrain(
        augmentations=None,
        validation=[str(validation_file), "CASE_002"],
    )

    train_names, validation_names = dataset._split_train_validation_names(
        ["CASE_000", "CASE_001", "CASE_002", "CASE_003"],
        {},
    )

    assert train_names == ["CASE_000"]
    assert validation_names == ["CASE_001", "CASE_002", "CASE_003"]


def test_data_split_prediction_keeps_case_patches_together_and_allows_empty_shards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KONFAI_STATE", str(State.PREDICTION))

    shards = Data._split(
        [(0, 0, 0), (0, 0, 1), (1, 0, 0), (2, 0, 0), (2, 0, 1)],
        4,
    )

    assert shards == [
        [(0, 0, 0), (0, 0, 1)],
        [(1, 0, 0)],
        [(2, 0, 0), (2, 0, 1)],
        [],
    ]


def test_data_remap_dataset_indices_compacts_sparse_mapping_indices() -> None:
    indices, remapped = Data._remap_dataset_indices([(3, 0, 0), (3, 0, 1), (8, 1, 0), (3, 1, 2)])

    assert indices == [3, 8]
    assert remapped == [(0, 0, 0), (0, 0, 1), (1, 1, 0), (0, 1, 2)]


def test_data_train_validation_none_keeps_full_dataset_for_training() -> None:
    dataset = DataTrain(
        augmentations=None,
        validation=None,
    )

    train_names, validation_names = dataset._split_train_validation_names(
        ["CASE_000", "CASE_001", "CASE_002"],
        {},
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

    assert dataset._prepared_data is not None
    assert dataset._prepared_validation_data is not None
    assert dataset._prepared_data["CT"][0].total_augmentations == 1
    assert dataset._prepared_validation_data["CT"][0].total_augmentations == 0


def test_dataset_iter_streams_base_patch_when_augmentations_are_disabled() -> None:
    volume = np.arange(1 * 4 * 4, dtype=np.float32).reshape(1, 4, 4)
    dataset_stub = StreamingDatasetStub(volume)
    augmentations = DataAugmentationsList(nb=1, data_augmentations={})
    manager = DatasetManager(
        index=0,
        group_src="CT",
        group_dest="CT",
        name="CASE_000",
        dataset=cast(Dataset, dataset_stub),
        patch=DatasetPatch([2, 2]),
        transforms=[],
        data_augmentations_list=[augmentations],
    )
    dataset_iter = DatasetIter(
        rank=0,
        data={"CT": [manager]},
        mapping=[(0, 0, 1)],
        groups_src={"CT": Group(groups_dest={"CT": GroupTransform(transforms=None, patch_transforms=None)})},
        inline_augmentations=False,
        data_augmentations_list=[augmentations],
        patch_size=[2, 2],
        overlap=None,
        buffer_size=1,
        apply_augmentations=False,
        use_cache=False,
    )

    sample = dataset_iter[0]["CT"].tensor

    assert dataset_stub.full_reads == 0
    assert dataset_stub.patch_reads == 1
    assert manager.loaded is False
    assert torch.equal(sample, torch.from_numpy(volume[:, 0:2, 2:4]))


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


def test_data_prediction_disables_persistent_workers_by_default() -> None:
    dataset = DataPrediction(augmentations=None)

    assert cast(int, dataset.dataLoader_args["num_workers"]) >= 1
    assert dataset.dataLoader_args["prefetch_factor"] == 2
    assert dataset.dataLoader_args["persistent_workers"] is False


def test_konfai_inference_raises_clear_error_inside_daemon_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    transform = KonfAIInference()

    class DaemonProcess:
        daemon = True

    monkeypatch.setattr("konfai.data.transform.current_process", lambda: DaemonProcess())

    with pytest.raises(RuntimeError, match=r"Dataset\.num_workers: 0"):
        transform("CASE_000", torch.zeros(1, 4, 4), Attribute())


def test_dataset_iter_streams_patch_reads_with_global_normalize_stats() -> None:
    volume = np.arange(1 * 4 * 4, dtype=np.float32).reshape(1, 4, 4)
    dataset_stub = StreamingDatasetStub(volume)
    normalize = Normalize()
    manager = DatasetManager(
        index=0,
        group_src="CT",
        group_dest="CT",
        name="CASE_000",
        dataset=cast(Dataset, dataset_stub),
        patch=DatasetPatch([2, 2]),
        transforms=[normalize],
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
    expected = (2 * volume[:, 0:2, 2:4] / (volume.max() - volume.min())) - 1

    assert dataset_stub.full_reads == 0
    assert dataset_stub.patch_reads == 1
    assert dataset_stub.stats_reads == 1
    np.testing.assert_allclose(sample.numpy(), expected)


def test_dataset_iter_streams_patch_reads_with_computed_standardize_stats() -> None:
    volume = np.arange(1 * 4 * 4, dtype=np.float32).reshape(1, 4, 4)
    dataset_stub = StreamingDatasetStub(volume)
    standardize = Standardize()
    manager = DatasetManager(
        index=0,
        group_src="CT",
        group_dest="CT",
        name="CASE_000",
        dataset=cast(Dataset, dataset_stub),
        patch=DatasetPatch([2, 2]),
        transforms=[standardize],
        data_augmentations_list=[],
    )
    dataset_iter = DatasetIter(
        rank=0,
        data={"CT": [manager]},
        mapping=[(0, 0, 3)],
        groups_src={"CT": Group(groups_dest={"CT": GroupTransform(transforms=None, patch_transforms=None)})},
        inline_augmentations=False,
        data_augmentations_list=[],
        patch_size=[2, 2],
        overlap=None,
        buffer_size=1,
        use_cache=False,
    )

    sample = dataset_iter[0]["CT"].tensor
    expected = (volume[:, 2:4, 2:4] - volume.mean()) / volume.std(ddof=1)

    assert dataset_stub.full_reads == 0
    assert dataset_stub.patch_reads == 1
    assert dataset_stub.stats_reads == 1
    np.testing.assert_allclose(sample.numpy(), expected)


def test_transform_mask_caches_mha_read_and_reads_file_only_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mask.__call__ must not re-read the .mha file on every invocation."""
    read_count = 0
    original_read_image = SimpleITK.ReadImage

    def counting_read(path: str) -> SimpleITK.Image:
        nonlocal read_count
        read_count += 1
        return original_read_image(path)

    monkeypatch.setattr("konfai.data.transform.sitk.ReadImage", counting_read)

    mask_array = np.ones((4, 4), dtype=np.uint8)
    mask_path = str(tmp_path / "mask.mha")
    SimpleITK.WriteImage(SimpleITK.GetImageFromArray(mask_array), mask_path)

    transform = Mask(path=mask_path, value_outside=0)
    attr = Attribute()

    for case in ("CASE_000", "CASE_001", "CASE_002"):
        transform(case, torch.ones(1, 4, 4), attr)

    assert read_count == 1, f"Expected mask to be read once, got {read_count} reads"


def test_dataset_iter_keeps_cache_lookup_in_sync_with_load_and_unload() -> None:
    dataset_iter = DatasetIter(
        rank=0,
        data={"CT": [cast(DatasetManager, object())]},
        mapping=[],
        groups_src={"CT": Group(groups_dest={"CT": GroupTransform(transforms=None, patch_transforms=None)})},
        inline_augmentations=False,
        data_augmentations_list=[],
        patch_size=None,
        overlap=None,
        buffer_size=1,
        use_cache=True,
    )

    dataset_iter.load_data = lambda *args, **kwargs: True  # type: ignore[method-assign]
    dataset_iter.unload_data = lambda *args, **kwargs: None  # type: ignore[method-assign]

    assert dataset_iter._index_cache == []
    assert dataset_iter._index_cache_lookup == set()

    dataset_iter._load_data(0)

    assert dataset_iter._index_cache == [0]
    assert dataset_iter._index_cache_lookup == {0}

    dataset_iter._unload_data(0)

    assert dataset_iter._index_cache == []
    assert dataset_iter._index_cache_lookup == set()


def test_dataset_get_names_caches_result_and_avoids_repeated_listdir(tmp_path: Path) -> None:
    dataset = Dataset(tmp_path / "Dataset", "mha")
    attrs = _image_attributes([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    volume = np.zeros((1, 4, 4, 4), dtype=np.float32)
    dataset.write("CT", "CASE_000", volume, attrs)
    dataset.write("CT", "CASE_001", volume, attrs)

    first = dataset.get_names("CT")
    cached = dataset.get_names("CT")

    assert first == cached == ["CASE_000", "CASE_001"]
    assert "CT" in dataset._names_cache


def test_dataset_get_names_cache_invalidated_on_write(tmp_path: Path) -> None:
    dataset = Dataset(tmp_path / "Dataset", "mha")
    attrs = _image_attributes([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    volume = np.zeros((1, 4, 4, 4), dtype=np.float32)
    dataset.write("CT", "CASE_000", volume, attrs)

    _ = dataset.get_names("CT")
    assert dataset._names_cache

    dataset.write("CT", "CASE_001", volume, attrs)
    assert not dataset._names_cache
    assert dataset.get_names("CT") == ["CASE_000", "CASE_001"]


def test_dataset_is_dataset_exist_benefits_from_cache(tmp_path: Path) -> None:
    dataset = Dataset(tmp_path / "Dataset", "mha")
    attrs = _image_attributes([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    volume = np.zeros((1, 4, 4, 4), dtype=np.float32)
    dataset.write("CT", "CASE_000", volume, attrs)

    assert dataset.is_dataset_exist("CT", "CASE_000")
    assert "CT" in dataset._names_cache
    assert not dataset.is_dataset_exist("CT", "CASE_999")
