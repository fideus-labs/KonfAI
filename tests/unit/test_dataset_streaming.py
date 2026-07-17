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

import warnings
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
    _check_patch_transform_invertible,
    _check_patch_transform_locality,
    _check_patch_transform_shape,
)
from konfai.data.patching import DatasetManager, DatasetPatch
from konfai.data.transform import (
    Clip,
    Dilate,
    Flip,
    Gradient,
    KonfAIInference,
    LocalityKind,
    Mask,
    Normalize,
    OneHot,
    PatchLocality,
    Permute,
    ResampleToShape,
    Softmax,
    Standardize,
    TensorCast,
    Transform,
    TransformLoader,
)
from konfai.utils import dataset as dataset_module
from konfai.utils.dataset import Attribute, Dataset
from konfai.utils.errors import ConfigError
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


def _write_image(path: Path, compress: bool) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = SimpleITK.ImageFileWriter()
    writer.SetFileName(str(path))
    writer.SetUseCompression(compress)
    writer.Execute(SimpleITK.GetImageFromArray(np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)))
    return path


def _reject_whole_volume_read(*args: object, **kwargs: object) -> None:
    pytest.fail("statistics must be accumulated slab by slab, never by reading the whole volume")


@pytest.mark.parametrize(
    ("filename", "compress", "streams"),
    [
        ("volume.mha", False, True),
        ("volume.mha", True, False),
        ("volume.mhd", False, True),
        ("volume.nii", False, True),
        ("volume.nii.gz", True, False),
        # NrrdImageIO serves no region at all, compressed or not: a slab loop would decode the whole
        # volume once per slab, so it stays on the single whole-volume read.
        ("volume.nrrd", False, False),
        ("volume.nrrd", True, False),
    ],
)
def test_sitk_supports_region_read_matches_itk_streaming_capability(
    tmp_path: Path, filename: str, compress: bool, streams: bool
) -> None:
    path = _write_image(tmp_path / filename, compress)

    assert Dataset.SitkFile._supports_region_read(str(path)) is streams


@pytest.mark.parametrize(
    ("file_format", "compress", "warns"),
    [("nrrd", False, True), ("mha", True, True), ("mha", False, False)],
)
def test_patch_stream_warns_once_per_format_that_cannot_serve_a_disk_region(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, file_format: str, compress: bool, warns: bool
) -> None:
    """A format serving no region re-decodes the whole volume per patch: say so, once for the dataset.

    Two cases x three patches: the warning is about the format, so it must survive neither the patch
    loop nor the second case. Streaming an uncompressed .mha is a win and must stay silent.
    """
    monkeypatch.setattr(dataset_module, "_unstreamed_formats_warned", set())
    dataset = Dataset(tmp_path / "Dataset", file_format)
    volume = np.arange(1 * 4 * 5 * 6, dtype=np.float32).reshape(1, 4, 5, 6)
    for name in ("CASE_000", "CASE_001"):
        dataset.write("CT", name, volume, _image_attributes([10.0, 20.0, 30.0], [0.5, 1.5, 2.0]))
        _write_image(tmp_path / "Dataset" / name / f"CT.{file_format}", compress)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for name in ("CASE_000", "CASE_001"):
            for plane in range(3):
                patch, _ = dataset.read_data_slice(
                    "CT",
                    name,
                    (slice(None), slice(plane, plane + 1), slice(None), slice(None)),
                )
                np.testing.assert_array_equal(patch, volume[:, plane : plane + 1])

    messages = [str(w.message) for w in caught if "cannot serve a disk region" in str(w.message)]
    assert len(messages) == (1 if warns else 0)
    if warns:
        assert f"'.{file_format}' files" in messages[0]
        assert "OME-Zarr or HDF5" in messages[0]


def test_dataset_read_data_statistics_sitk_accumulates_slabs_without_loading_full_volume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = Dataset(tmp_path / "Dataset", "mha")
    volume = np.arange(1 * 4 * 5 * 6, dtype=np.float32).reshape(1, 4, 5, 6)
    dataset.write("CT", "CASE_000", volume, _image_attributes([10.0, 20.0, 30.0], [0.5, 1.5, 2.0]))

    # One slab per plane, so the running merge spans several reads on a volume this small.
    monkeypatch.setattr(dataset_module, "_STATISTICS_CHUNK_ELEMENTS", 1)
    monkeypatch.setattr(SimpleITK, "ReadImage", _reject_whole_volume_read)

    stats = dataset.read_data_statistics("CT", "CASE_000")

    assert stats["min"] == pytest.approx(float(volume.min()))
    assert stats["max"] == pytest.approx(float(volume.max()))
    assert stats["mean"] == pytest.approx(float(volume.mean()))
    assert stats["std"] == pytest.approx(float(volume.std(ddof=1)))


def test_dataset_read_data_statistics_sitk_selects_channels_while_streaming(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = Dataset(tmp_path / "Dataset", "mha")
    volume = np.arange(3 * 4 * 5 * 6, dtype=np.float32).reshape(3, 4, 5, 6)
    dataset.write("CT", "CASE_000", volume, _image_attributes([10.0, 20.0, 30.0], [0.5, 1.5, 2.0]))

    monkeypatch.setattr(dataset_module, "_STATISTICS_CHUNK_ELEMENTS", 1)
    monkeypatch.setattr(SimpleITK, "ReadImage", _reject_whole_volume_read)

    stats = dataset.read_data_statistics("CT", "CASE_000", [0, 2])

    assert stats["mean"] == pytest.approx(float(volume[[0, 2]].mean()))
    assert stats["std"] == pytest.approx(float(volume[[0, 2]].std(ddof=1)))


def test_dataset_read_data_statistics_sitk_keeps_whole_read_for_compressed_volumes(tmp_path: Path) -> None:
    dataset = Dataset(tmp_path / "Dataset", "mha")
    volume = np.arange(1 * 4 * 5 * 6, dtype=np.float32).reshape(1, 4, 5, 6)
    dataset.write("CT", "CASE_000", volume, _image_attributes([10.0, 20.0, 30.0], [0.5, 1.5, 2.0]))
    _write_image(tmp_path / "Dataset" / "CASE_000" / "CT.mha", compress=True)

    stats = dataset.read_data_statistics("CT", "CASE_000")

    compressed = np.arange(4 * 5 * 6, dtype=np.float32)
    assert stats["mean"] == pytest.approx(float(compressed.mean()))
    assert stats["std"] == pytest.approx(float(compressed.std(ddof=1)))


class StreamingDatasetStub:
    def __init__(self, volume: np.ndarray) -> None:
        self.volume = volume
        self.full_reads = 0
        self.patch_reads = 0
        self.stats_reads = 0
        # Identity geometry with one origin/spacing entry per spatial axis (channel-first volume),
        # so geometry-aware transforms (e.g. Resample) get a Spacing matching the volume's rank.
        spatial = volume.ndim - 1
        self._geometry = ([0.0] * spatial, [1.0] * spatial)

    def get_infos(self, group_src: str, name: str) -> tuple[list[int], Attribute]:
        return list(self.volume.shape), _image_attributes(*self._geometry)

    def read_data(self, group_src: str, name: str) -> tuple[np.ndarray, Attribute]:
        self.full_reads += 1
        return self.volume.copy(), _image_attributes(*self._geometry)

    def read_data_slice(self, group_src: str, name: str, slices: tuple[slice, ...]) -> tuple[np.ndarray, Attribute]:
        self.patch_reads += 1
        return self.volume[slices].copy(), _image_attributes(*self._geometry)

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
    # The cache regime is no longer a config knob; the budget resolver flips it through the same
    # re-entry point used here.
    dataset = DataTrain(augmentations=None)
    dataset._configure_data_loading(use_cache=False)

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


def _build_streaming_manager(volume: np.ndarray, transforms: list[Transform], patch_size: list[int]) -> DatasetManager:
    stub = StreamingDatasetStub(volume)
    return DatasetManager(
        index=0,
        group_src="CT",
        group_dest="CT",
        name="CASE_000",
        dataset=cast(Dataset, stub),
        patch=DatasetPatch(patch_size),
        transforms=transforms,
        data_augmentations_list=[],
    )


def _assert_stream_matches_whole_volume(
    volume: np.ndarray,
    transforms: list[Transform],
    patch_size: list[int],
    *,
    atol: float = 0.0,
) -> DatasetManager:
    """Every streamed patch must equal the whole-volume pass sliced on the same grid."""
    manager = _build_streaming_manager(volume, transforms, patch_size)
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


def test_stream_halo_dilate_seam_matches_whole_volume() -> None:
    # Foreground straddling the patch boundary at column 4: a whole-volume dilation spreads across the
    # seam, so a correct HALO read + crop must reproduce it patch-for-patch.
    volume = np.zeros((1, 8, 8), dtype=np.float32)
    volume[0, 3:5, 3:5] = 1.0
    manager = _assert_stream_matches_whole_volume(volume, [Dilate(2)], [4, 4])
    # The dispatcher must actually take the region (HALO) path, not fall back.
    assert manager._resolve_patch_stream_source(0, True).region_index == 0


def test_stream_halo_gradient_seam_matches_whole_volume() -> None:
    rng = np.random.default_rng(0)
    volume = rng.standard_normal((1, 8, 8)).astype(np.float32)
    _assert_stream_matches_whole_volume(volume, [Gradient()], [4, 4], atol=1e-6)


def test_stream_orientation_flip_remap_matches_whole_volume() -> None:
    volume = np.arange(1 * 8 * 8, dtype=np.float32).reshape(1, 8, 8)
    _assert_stream_matches_whole_volume(volume, [Flip("0|1")], [4, 4])


def test_stream_orientation_permute_remap_matches_whole_volume() -> None:
    volume = np.arange(1 * 8 * 8, dtype=np.float32).reshape(1, 8, 8)
    manager = _assert_stream_matches_whole_volume(volume, [Permute("1|0")], [4, 4])
    assert manager._resolve_patch_stream_source(0, True).region_index == 0


def test_stream_orientation_permute_border_patch_uses_the_permuted_grid() -> None:
    # Permute swaps the spatial axes, so the patch grid is cut on the PERMUTED extents (7x8, not 8x7).
    # The last patch of the 7-long target axis is one voxel short of patch_size: the streamed patch must
    # be padded against that target grid, not against the source shape, which would leave it short.
    volume = np.arange(1 * 8 * 7, dtype=np.float32).reshape(1, 8, 7)
    _assert_stream_matches_whole_volume(volume, [Permute("1|0")], [4, 4])


def test_stream_pointwise_border_patch_pads_after_the_chain() -> None:
    # A 9-long axis leaves the last patch one voxel short of patch_size, so the read plan pads it up.
    # The whole-volume path transforms the volume and only then pads (with the min of the TRANSFORMED
    # patch), so the streamed path must apply the read plan after its chain too -- padding the raw patch
    # first pads in the source domain and then runs the transform over the padding.
    volume = np.arange(3 * 8 * 9, dtype=np.float32).reshape(3, 8, 9)
    _assert_stream_matches_whole_volume(volume, [Softmax(0)], [4, 4])


def test_stream_global_stat_before_orientation_region_matches_whole_volume() -> None:
    # GLOBAL_STAT (Normalize, seeded from disk stats) as a pre-pointwise stage in front of an
    # ORIENTATION region transform: both the stat and the remap must compose byte-identically.
    volume = np.arange(1 * 8 * 8, dtype=np.float32).reshape(1, 8, 8)
    transforms: list[Transform] = [Normalize(), Flip("0")]
    manager = _build_streaming_manager(volume, transforms, [4, 4])
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


def test_stream_pointwise_chain_matches_whole_volume() -> None:
    # A trailing chain of purely POINTWISE transforms streams the exact patch (region_index is None).
    volume = np.arange(1 * 8 * 8, dtype=np.float32).reshape(1, 8, 8)
    manager = _assert_stream_matches_whole_volume(volume, [TensorCast("float32"), Flip("0")], [4, 4])
    assert manager._resolve_patch_stream_source(0, True).region_index == 1


def test_stream_composed_orientation_and_halo_matches_whole_volume() -> None:
    # Region stages compose: the dilation's halo pulls through the flip's mirror, so one bounded read
    # serves both and the seam-spreading foreground must still agree bit for bit.
    volume = np.zeros((1, 8, 8), dtype=np.float32)
    volume[0, 3:5, 3:5] = 1.0
    manager = _assert_stream_matches_whole_volume(volume, [Flip("0"), Dilate(1)], [4, 4])
    plans = manager._resolve_patch_stream_source(0, True).stage_plans
    assert [plan.kind.value for plan in plans] == ["orientation", "halo"]


def test_stream_composed_rescale_and_orientation_matches_whole_volume() -> None:
    # A resample followed by a flip: the flip's mirror region pulls through the resample's scale
    # window, on the RESAMPLED grid the fold computed between them.
    rng = np.random.default_rng(7)
    volume = (rng.standard_normal((1, 8, 8)).astype(np.float32)) * 100.0
    manager = _assert_stream_matches_whole_volume(
        volume, [ResampleToShape(shape=[12, 12]), Flip("0")], [4, 4], atol=1e-3
    )
    plans = manager._resolve_patch_stream_source(0, True).stage_plans
    assert [plan.kind.value for plan in plans] == ["rescale", "orientation"]
    assert tuple(plans[1].in_shape) == (12, 12)


def test_stream_composed_triple_region_chain_matches_whole_volume() -> None:
    # Three region stages in one chain — flip, resample, permute — folded into one bounded read.
    rng = np.random.default_rng(11)
    volume = (rng.standard_normal((1, 8, 6)).astype(np.float32)) * 100.0
    manager = _assert_stream_matches_whole_volume(
        volume, [Flip("0"), ResampleToShape(shape=[12, 9]), Permute("1|0")], [4, 4], atol=1e-3
    )
    plans = manager._resolve_patch_stream_source(0, True).stage_plans
    assert [plan.kind.value for plan in plans] == ["orientation", "rescale", "orientation"]
    assert tuple(plans[2].out_shape) == (9, 12)


def test_stream_composed_orientations_with_pointwise_between_match_whole_volume() -> None:
    # Two orientations with a pointwise stage between them: the fold carries the permuted extents and
    # the value map rides along where the regions put it.
    volume = np.arange(1 * 8 * 6, dtype=np.float32).reshape(1, 8, 6)
    manager = _assert_stream_matches_whole_volume(
        volume, [Flip("0"), Clip(min_value=-10.0, max_value=10.0), Permute("1|0")], [4, 4]
    )
    plans = manager._resolve_patch_stream_source(0, True).stage_plans
    assert [plan.kind.value for plan in plans] == ["orientation", "pointwise", "orientation"]
    assert tuple(plans[2].out_shape) == (6, 8)


def test_softmax_channel_axis_is_pointwise_but_spatial_axis_falls_back() -> None:
    # A channel-axis softmax (dim 0) is spatially pointwise and streams the exact patch. A softmax over
    # a SPATIAL axis normalises across the whole extent, so a per-patch softmax would diverge: the
    # contract must declare it WHOLE_VOLUME and the dispatcher must refuse to stream it.
    assert Softmax(0).patch_locality(Attribute()).kind is LocalityKind.POINTWISE
    assert Softmax(1).patch_locality(Attribute()).kind is LocalityKind.WHOLE_VOLUME
    assert Softmax(-1).patch_locality(Attribute()).kind is LocalityKind.WHOLE_VOLUME

    volume = np.arange(3 * 8 * 8, dtype=np.float32).reshape(3, 8, 8)
    _assert_stream_matches_whole_volume(volume, [Softmax(0)], [4, 4], atol=1e-6)

    spatial_manager = _build_streaming_manager(volume, [Softmax(1)], [4, 4])
    assert spatial_manager.can_stream_patch(0) is False


def test_stream_clip_fixed_bounds_is_pointwise_and_matches_whole_volume() -> None:
    # Fixed float bounds clip each voxel independently: POINTWISE, exact patch, no region transform.
    rng = np.random.default_rng(0)
    volume = (rng.standard_normal((1, 8, 8)).astype(np.float32)) * 100.0
    assert Clip(min_value=-50.0, max_value=50.0).patch_locality(Attribute()).kind is LocalityKind.POINTWISE
    manager = _assert_stream_matches_whole_volume(volume, [Clip(min_value=-50.0, max_value=50.0)], [4, 4])
    assert manager._resolve_patch_stream_source(0, True).region_index is None


def test_stream_clip_min_max_is_global_stat_and_matches_whole_volume() -> None:
    # 'min'/'max' bounds clip to the volume extremum -- a no-op on that bound -- so the streamed
    # per-patch result is byte-identical to the whole-volume pass, and the dispatcher seeds the
    # global stat from a single read_data_statistics call instead of loading the full volume.
    rng = np.random.default_rng(1)
    volume = (rng.standard_normal((1, 8, 8)).astype(np.float32)) * 100.0
    assert Clip(min_value="min", max_value="max").patch_locality(Attribute()).kind is LocalityKind.GLOBAL_STAT

    stub = StreamingDatasetStub(volume)
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


def test_clip_percentile_and_mask_bounds_fall_back_to_whole_volume() -> None:
    # A percentile bound needs the whole histogram and a mask reads a second full volume: both
    # genuinely require the whole volume, so the contract declares WHOLE_VOLUME and streaming is off.
    assert (
        Clip(min_value="percentile:1", max_value="percentile:99").patch_locality(Attribute()).kind
        is LocalityKind.WHOLE_VOLUME
    )
    assert Clip(mask="SEG").patch_locality(Attribute()).kind is LocalityKind.WHOLE_VOLUME
    volume = np.arange(1 * 8 * 8, dtype=np.float32).reshape(1, 8, 8)
    manager = _build_streaming_manager(volume, [Clip(min_value="percentile:1", max_value="percentile:99")], [4, 4])
    assert not manager.can_stream_patch(0)


def test_stream_orientation_border_patch_is_padded_to_patch_size() -> None:
    # A tiling whose last patch is narrower than patch_size (30 with patch 8 -> border width 6): the
    # whole-volume Patch.get_data pads that border up to patch_size, so the region streamed path must
    # too, otherwise the border patch comes out one-or-more voxels short and cannot batch/reassemble.
    volume = np.arange(1 * 30 * 30, dtype=np.float32).reshape(1, 30, 30)
    manager = _assert_stream_matches_whole_volume(volume, [Flip("0|1")], [8, 8])
    assert manager._resolve_patch_stream_source(0, True).region_index == 0
    # Every streamed patch is exactly patch_size, including the borders.
    size = manager.patch.get_size(0)
    for index in range(size):
        assert tuple(manager._get_streamed_data(index, 0, True)[0].shape) == (1, 8, 8)


def test_stream_resample_border_patch_matches_padded_whole_volume() -> None:
    # RESCALE upsample to a grid that tiles unevenly (30 with patch 8 -> border width 6). The whole-
    # volume path resamples the whole volume then pads border patches to patch_size; the streamed
    # resample path must reproduce that padding so border patches are shape- and value-consistent.
    rng = np.random.default_rng(3)
    volume = (rng.standard_normal((1, 20, 20, 20)).astype(np.float32)) * 100.0
    shape = [30, 30, 30]
    patch = [8, 8, 8]

    stream_manager = _build_streaming_manager(volume, [ResampleToShape(shape=shape)], patch)
    assert stream_manager.can_stream_patch(0)
    assert stream_manager._resolve_patch_stream_source(0, True).region_index == 0

    reference_manager = _build_streaming_manager(volume, [ResampleToShape(shape=shape)], patch)
    reference_manager.load(reference_manager.transforms, [], load_augmentations=False)

    size = stream_manager.patch.get_size(0)
    streamed = [stream_manager._get_streamed_data(index, 0, True)[0] for index in range(size)]
    reference = [reference_manager.patch.get_data(reference_manager.data[0], index, 0, True) for index in range(size)]

    assert len(streamed) == len(reference) == size
    for got, expected in zip(streamed, reference, strict=False):
        assert tuple(got.shape) == tuple(expected.shape) == (1, 8, 8, 8)
        # Interior values match F.interpolate to float32 interpolation-rounding; the previously
        # short border patch is now padded to patch_size and byte-consistent in shape.
        np.testing.assert_allclose(got.numpy(), expected.numpy(), atol=1e-3)


def test_stream_resample_nearest_strong_downsampling_matches_whole_volume() -> None:
    # Strong downsampling of a uint8 label map (nearest mode): the nearest voxel of the first output
    # column (floor(o*scale)) falls BELOW the linear tap window's start, so the source read must widen
    # to include it -- otherwise the gather indexed a negative local offset and wrapped onto the far
    # edge, silently returning a wrong label. A regular ratio (a plain integer scale) hides the bug;
    # 40 -> 6 (scale 6.67) exposes the sub-pixel offset that pushes the linear start past voxel 0.
    volume = (np.arange(1 * 40 * 40).reshape(1, 40, 40) % 7).astype(np.uint8)
    shape = [6, 6]
    patch = [3, 3]
    stream_manager = _build_streaming_manager(volume, [ResampleToShape(shape=shape)], patch)
    assert stream_manager.can_stream_patch(0)

    reference_manager = _build_streaming_manager(volume, [ResampleToShape(shape=shape)], patch)
    reference_manager.load(reference_manager.transforms, [], load_augmentations=False)

    size = stream_manager.patch.get_size(0)
    for index in range(size):
        got = stream_manager._get_streamed_data(index, 0, True)[0]
        expected = reference_manager.patch.get_data(reference_manager.data[0], index, 0, True)
        # Nearest is a pure gather: the streamed patch must equal the whole-volume pick bit for bit.
        assert torch.equal(got, expected)


# --------------------------------------------------------------------------------------
# patch_transforms — the per-patch opt-in, guarded by the patch-locality contract
#
# A patch transform only ever sees ONE patch, and that is what asking for it there means: a
# GLOBAL_STAT transform handed a patch derives the PATCH's statistic, deliberately. The volume's
# statistic is opted into explicitly, by capturing it case-level with `lazy` (which traverses the
# volume, caches Mean/Std and applies nothing) and letting the patch transform find it. These cover
# both routes, and that neither one leaks a patch's statistic onto the shared case attribute.
# --------------------------------------------------------------------------------------


def _structured_volume() -> np.ndarray:
    """A spatially STRUCTURED signal: each patch has a very different local statistic.

    A uniform-noise volume hides the bug (every patch shares the volume's statistic); a ramp
    makes a patch-local statistic diverge from the volume-global one.
    """
    z, y, x = np.meshgrid(np.arange(16), np.arange(16), np.arange(16), indexing="ij")
    return (100.0 * z + 10.0 * y + 1.0 * x).astype(np.float32)[None]


def _patch_manager(volume: np.ndarray, transforms: list[Transform]) -> DatasetManager:
    return DatasetManager(
        index=0,
        group_src="CT",
        group_dest="CT",
        name="CASE_000",
        dataset=cast(Dataset, StreamingDatasetStub(volume)),
        patch=DatasetPatch([8, 8, 8], overlap=4),
        transforms=transforms,
        data_augmentations_list=[],
    )


def test_patch_transform_standardize_applies_a_lazily_captured_volume_statistic() -> None:
    """`Standardize(lazy=True)` case-level + `Standardize()` per patch == case-level Standardize.

    This is the documented way to standardize per patch by the VOLUME's statistic: the lazy pass
    caches Mean/Std without applying anything, and the patch transform finds them on the attribute.
    """
    volume = _structured_volume()
    case_level = _patch_manager(volume, [Standardize()])
    per_patch = _patch_manager(volume, [Standardize(lazy=True)])

    size = case_level.patch.get_size(0)
    assert size > 1
    for index in range(size):
        expected = case_level.get_data(index, 0, [], True)
        got = per_patch.get_data(index, 0, [Standardize()], True)
        assert torch.equal(got, expected)


def test_patch_transform_standardize_uses_the_patch_own_statistic() -> None:
    """Asked for per-patch, a GLOBAL_STAT transform standardizes the patch by ITS OWN statistic."""
    volume = _structured_volume()
    manager = _patch_manager(volume, [])

    patch = manager.get_data(0, 0, [Standardize()], True)

    source = torch.from_numpy(volume[:, 0:8, 0:8, 0:8])
    expected = (source - source.mean()) / source.std()
    assert torch.equal(patch, expected)
    # The patch's own mean is a long way from the volume's, so this really is the local statistic.
    assert abs(float(source.mean()) - float(torch.from_numpy(volume).mean())) > 100.0


def test_patch_transform_statistic_never_leaks_onto_the_case_attribute() -> None:
    """A patch-local statistic must not reach the attribute the whole case shares.

    Left there, the first patch read would freeze its own Mean/Std for every later patch: neither
    the volume's statistic nor the patch's, and dependent on the order the patches happen to be read.
    """
    volume = _structured_volume()
    manager = _patch_manager(volume, [])

    manager.get_data(0, 0, [Standardize()], True)

    assert "Mean" not in manager.cache_attributes[0]
    assert "Std" not in manager.cache_attributes[0]


def test_patch_transform_standardize_is_independent_of_patch_order() -> None:
    """A patch's own statistic is the patch's alone: reading others first cannot change it."""
    volume = _structured_volume()
    forward = _patch_manager(volume, [])
    backward = _patch_manager(volume, [])

    size = forward.patch.get_size(0)
    first = forward.get_data(0, 0, [Standardize()], True)
    for index in reversed(range(size)):
        backward.get_data(index, 0, [Standardize()], True)
    last = backward.get_data(0, 0, [Standardize()], True)

    assert torch.equal(first, last)


def test_patch_transform_is_identical_across_managers() -> None:
    """A fresh manager per patch -- the per-DataLoader-worker case -- gives the same patch.

    Each worker owns its own cache attribute, so anything a patch records on it makes the result
    depend on which worker drew which patch. Every patch here must be reproducible on its own.
    """
    volume = _structured_volume()
    shared = _patch_manager(volume, [])
    size = shared.patch.get_size(0)

    for index in range(size):
        assert torch.equal(
            _patch_manager(volume, []).get_data(index, 0, [Standardize()], True),
            shared.get_data(index, 0, [Standardize()], True),
        )


def test_patch_transform_overlapping_patches_agree_on_shared_voxel() -> None:
    """With the volume statistic captured lazily, two overlapping patches agree on a shared voxel.

    A fresh manager per patch reproduces the per-DataLoader-worker case: the coefficients come from
    the case-level lazy pass, so they are the same in every worker.
    """
    volume = _structured_volume()
    size = _patch_manager(volume, []).patch.get_size(0)

    values: dict[tuple[int, int, int], list[float]] = {}
    for index in range(size):
        manager = _patch_manager(volume, [Standardize(lazy=True)])
        patch = manager.get_data(index, 0, [Standardize()], True)
        slices = manager.patch.get_read_plan([1, 16, 16, 16], index, 0, True).data_slices
        zs, ys, xs = slices[1], slices[2], slices[3]
        for z in range(zs.start, zs.stop):
            for y in range(ys.start, ys.stop):
                for x in range(xs.start, xs.stop):
                    voxel = float(patch[0, z - zs.start, y - ys.start, x - xs.start])
                    values.setdefault((z, y, x), []).append(voxel)

    shared = [v for v in values.values() if len(v) > 1]
    assert shared, "the patch grid must overlap for this test to mean anything"
    assert max(max(v) - min(v) for v in shared) == 0.0


def test_patch_transform_normalize_applies_a_lazily_captured_volume_range() -> None:
    volume = _structured_volume()
    manager = _patch_manager(volume, [Normalize(lazy=True)])

    patch = manager.get_data(0, 0, [Normalize(min_value=-1, max_value=1)], True)

    # Mapped by the volume's range, so the first patch (low corner of the ramp) stays well
    # below the top of the target interval instead of being stretched onto it.
    assert float(manager.cache_attributes[0]["Min"]) == pytest.approx(float(volume.min()))
    assert float(manager.cache_attributes[0]["Max"]) == pytest.approx(float(volume.max()))
    assert float(patch.max()) < 0.0


def test_lazy_capture_reads_volume_statistics_once_per_case() -> None:
    """The whole-volume statistic is a full disk scan: read it once, not once per patch."""
    volume = _structured_volume()
    stub = StreamingDatasetStub(volume)
    manager = DatasetManager(
        index=0,
        group_src="CT",
        group_dest="CT",
        name="CASE_000",
        dataset=cast(Dataset, stub),
        patch=DatasetPatch([8, 8, 8], overlap=4),
        transforms=[Standardize(lazy=True)],
        data_augmentations_list=[],
    )

    for index in range(manager.patch.get_size(0)):
        manager.get_data(index, 0, [Standardize()], True)

    assert stub.stats_reads == 1
    assert stub.full_reads == 0


def test_patch_transform_reads_no_disk_statistic_when_the_volume_is_loaded() -> None:
    """A loaded volume already holds the answer: the patch path must not go back to disk for it.

    The lazy pass computes Mean/Std from the tensor in hand -- free, and carrying whatever the
    preceding chain did to it -- so a `read_data_statistics` scan here would be both wasted and a
    statistic of the wrong (stored) version of the volume.
    """
    volume = _structured_volume()
    stub = StreamingDatasetStub(volume)
    lazy: list[Transform] = [Standardize(lazy=True)]
    manager = DatasetManager(
        index=0,
        group_src="CT",
        group_dest="CT",
        name="CASE_000",
        dataset=cast(Dataset, stub),
        patch=DatasetPatch([8, 8, 8], overlap=4),
        transforms=lazy,
        data_augmentations_list=[],
    )
    manager.load(lazy, [], load_augmentations=False)
    assert manager.loaded is True

    case_level: list[Transform] = [Standardize()]
    reference = _patch_manager(volume, case_level)
    reference.load(case_level, [], load_augmentations=False)
    for index in range(manager.patch.get_size(0)):
        assert torch.equal(manager.get_data(index, 0, [Standardize()], True), reference.get_data(index, 0, [], True))
    assert stub.stats_reads == 0


# --------------------------------------------------------------------------------------
# Seeding a GLOBAL_STAT from disk reads the statistics of the STORED volume, so it is only that
# transform's own input when nothing before it touched the values.
# --------------------------------------------------------------------------------------


def test_streaming_is_refused_when_a_transform_modifies_values_before_a_global_stat() -> None:
    """[Clip, Standardize] must not stream: on disk lie the PRE-Clip statistics.

    Clipping moves Mean and Std, so seeding Standardize from `read_data_statistics` would standardize
    every patch by a statistic of a volume that no longer exists. Refusing sends the case down the
    whole-volume path, where Standardize computes Mean/Std from the clipped tensor it is handed.
    """
    volume = _structured_volume()
    manager = _patch_manager(volume, [Clip(min_value=200.0, max_value=1000.0), Standardize()])

    assert manager.can_stream_patch(0) is False
    assert "Mean" not in manager.cache_attributes[0]


def test_clip_then_standardize_equals_the_whole_volume_result() -> None:
    """The value every patch must carry: standardized by the CLIPPED volume's statistic."""
    volume = _structured_volume()
    chain: list[Transform] = [Clip(min_value=200.0, max_value=1000.0), Standardize()]
    manager = _patch_manager(volume, chain)
    manager.load(chain, [], load_augmentations=False)

    clipped = torch.from_numpy(volume).clip(200.0, 1000.0)
    expected_volume = (clipped - clipped.mean()) / clipped.std()

    size = manager.patch.get_size(0)
    assert size > 1
    for index in range(size):
        patch = manager.get_data(index, 0, [], True)
        slices = manager.patch.get_read_plan(list(volume.shape), index, 0, True).data_slices
        assert torch.equal(patch, expected_volume[slices])
    # The statistic the rejected seed would have used is a long way from the clipped volume's.
    assert abs(float(torch.from_numpy(volume).mean()) - float(clipped.mean())) > 100.0


def test_streaming_still_seeds_a_global_stat_behind_a_reorientation() -> None:
    """A flip only moves voxels, so the stored statistics are still Standardize's own input."""
    volume = _structured_volume()
    manager = _patch_manager(volume, [Flip(dims="0"), Standardize()])

    assert manager.can_stream_patch(0) is True
    assert "Mean" in manager.cache_attributes[0]


@pytest.mark.parametrize(
    ("transform", "kind"),
    [
        (Standardize(mask="MASK"), LocalityKind.WHOLE_VOLUME),
        (KonfAIInference(), LocalityKind.WHOLE_VOLUME),
        (Gradient(), LocalityKind.HALO),
        (Dilate(dilate=2), LocalityKind.HALO),
        (Flip(), LocalityKind.ORIENTATION),
        (Permute(), LocalityKind.ORIENTATION),
    ],
)
def test_patch_transform_rejects_transforms_that_cannot_run_per_patch(
    monkeypatch: pytest.MonkeyPatch, transform: Transform, kind: LocalityKind
) -> None:
    """A transform that cannot be correct per-patch must fail at config time, never silently."""
    monkeypatch.setenv("KONFAI_ROOT", "Trainer")
    assert transform.patch_locality(Attribute()).kind is kind

    with pytest.raises(ConfigError) as excinfo:
        _check_patch_transform_locality(transform, "CT", "CT")

    message = str(excinfo.value)
    assert type(transform).__name__ in message
    assert "patch_transforms" in message
    assert "transforms" in message


@pytest.mark.parametrize(
    "transform",
    [TensorCast(dtype="float32"), Standardize(mean=[0.0], std=[1.0]), Standardize(), Normalize()],
)
def test_patch_transform_accepts_pointwise_and_global_stat_transforms(
    monkeypatch: pytest.MonkeyPatch, transform: Transform
) -> None:
    monkeypatch.setenv("KONFAI_ROOT", "Trainer")

    _check_patch_transform_locality(transform, "CT", "CT")


class _ShapeChangingPointwise(Transform):
    """What the locality declaration cannot catch: a custom transform that declares POINTWISE and crops."""

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        return PatchLocality(LocalityKind.POINTWISE)

    def transform_shape(self, group_src: str, name: str, shape: list[int], cache_attribute: Attribute) -> list[int]:
        return [size - 1 for size in shape]

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        return tensor[..., :-1, :-1, :-1]


def test_patch_transform_rejects_a_transform_that_changes_the_spatial_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    """A POINTWISE declaration buys a transform past the locality check; the shape it returns does not."""
    monkeypatch.setenv("KONFAI_ROOT", "Trainer")
    transform = _ShapeChangingPointwise()
    _check_patch_transform_locality(transform, "CT", "CT")  # the declaration alone lets it through

    with pytest.raises(ConfigError) as excinfo:
        _check_patch_transform_shape(transform, "CT", "CT")

    message = str(excinfo.value)
    assert "_ShapeChangingPointwise" in message
    assert "Trainer.Dataset.groups_src.CT.groups_dest.CT.patch_transforms" in message


def test_patch_transform_shape_guard_is_spatial_not_channel(monkeypatch: pytest.MonkeyPatch) -> None:
    """OneHot expands the CHANNEL axis and keeps the spatial one: the grid is spatial, so it is allowed."""
    monkeypatch.setenv("KONFAI_ROOT", "Trainer")
    one_hot = OneHot(num_classes=4)
    labels = torch.zeros((1, 4, 5, 6), dtype=torch.int64)
    assert list(one_hot("CASE_000", labels, Attribute()).shape) == [4, 4, 5, 6]  # 1 channel -> 4

    _check_patch_transform_shape(one_hot, "CT", "CT")


def test_group_transform_prepare_guards_the_shape_of_every_patch_transform(monkeypatch: pytest.MonkeyPatch) -> None:
    """The guard runs at config time, from prepare() -- not only when someone calls it directly."""
    monkeypatch.setenv("KONFAI_ROOT", "Trainer")
    monkeypatch.setattr(TransformLoader, "get_transform", lambda *_, **__: _ShapeChangingPointwise())
    group = GroupTransform(transforms=None, patch_transforms={"_ShapeChangingPointwise": TransformLoader()})

    with pytest.raises(ConfigError):
        group.prepare("CT", "CT")


@pytest.mark.parametrize("state", [State.TRAIN, State.RESUME])
def test_per_patch_global_stat_is_allowed_when_training(monkeypatch: pytest.MonkeyPatch, state: State) -> None:
    """Per-patch statistics are a valid, deliberate training use: no forward inverse runs to break."""
    monkeypatch.setenv("KONFAI_ROOT", "Trainer")
    monkeypatch.setenv("KONFAI_STATE", str(state))
    _check_patch_transform_invertible(Standardize(), [], "CT", "CT")


@pytest.mark.parametrize("transform", [Standardize(), Normalize()])
def test_per_patch_global_stat_is_refused_at_prediction(monkeypatch: pytest.MonkeyPatch, transform: Transform) -> None:
    """At prediction the finalize inverse pops a statistic the per-patch scope never left case-level."""
    monkeypatch.setenv("KONFAI_ROOT", "Predictor")
    monkeypatch.setenv("KONFAI_STATE", str(State.PREDICTION))

    with pytest.raises(ConfigError) as excinfo:
        _check_patch_transform_invertible(transform, [], "CT", "CT")

    message = str(excinfo.value)
    assert type(transform).__name__ in message
    assert "Predictor.Dataset.groups_src.CT.groups_dest.CT.patch_transforms" in message
    assert "lazy=True" in message


def test_case_level_lazy_capture_makes_the_patch_statistic_invertible(monkeypatch: pytest.MonkeyPatch) -> None:
    """Standardize(lazy=True) in transforms caches Mean/Std case-level, so the patch consumer inverts."""
    monkeypatch.setenv("KONFAI_ROOT", "Predictor")
    monkeypatch.setenv("KONFAI_STATE", str(State.PREDICTION))
    _check_patch_transform_invertible(Standardize(), [Standardize(lazy=True)], "CT", "CT")


def test_per_patch_global_stat_without_inverse_is_allowed_at_prediction(monkeypatch: pytest.MonkeyPatch) -> None:
    """inverse=False never pops the statistic, so there is nothing to reconstruct and nothing to refuse."""
    monkeypatch.setenv("KONFAI_ROOT", "Predictor")
    monkeypatch.setenv("KONFAI_STATE", str(State.PREDICTION))
    _check_patch_transform_invertible(Standardize(inverse=False), [], "CT", "CT")
