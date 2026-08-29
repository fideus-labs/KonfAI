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

"""Streamed access to stored volumes through ``konfai.utils.dataset`` and the patch pipeline on top.

Region (``read_data_slice``) and statistics reads per storage backend, the read-side caches (pooled
h5 handles, the name cache, the OME-Zarr memo), DatasetIter patch reads that never load a full
volume, and the per-patch transform contract (patch locality, lazily captured volume statistics,
GLOBAL_STAT seeding from the stored statistics)."""

import warnings
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
from konfai.data.augmentation import DataAugmentationsList
from konfai.data.data_manager import (
    DatasetIter,
    Group,
    GroupTransform,
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
    RegionContext,
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
h5py = pytest.importorskip("h5py")


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


def test_h5_read_handle_is_pooled_across_reads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The chunk cache lives on the open handle: repeated patch reads must reuse one handle, not
    rebuild an empty cache per read."""
    import h5py

    dataset = Dataset(tmp_path / "Pooled", "h5")
    volume = np.arange(1 * 6 * 5, dtype=np.float32).reshape(1, 6, 5)
    dataset.write("CT", "CASE_000", volume, _image_attributes([1.0, 2.0], [0.5, 1.5]))

    real_file = h5py.File
    read_opens = 0

    def counting(name, mode="r", *args, **kwargs):
        nonlocal read_opens
        if mode == "r":
            read_opens += 1
        return real_file(name, mode, *args, **kwargs)

    monkeypatch.setattr(h5py, "File", counting)
    for start in range(3):
        patch, _ = dataset.read_data_slice("CT", "CASE_000", (slice(None), slice(start, start + 2), slice(0, 5)))
        np.testing.assert_array_equal(patch, volume[:, start : start + 2])
    assert read_opens == 1


def test_h5_write_invalidates_the_pooled_reader(tmp_path: Path) -> None:
    dataset = Dataset(tmp_path / "Invalidated", "h5")
    attrs = _image_attributes([1.0, 2.0], [0.5, 1.5])
    volume = np.zeros((1, 4, 5), dtype=np.float32)
    dataset.write("CT", "CASE_000", volume, attrs)
    first, _ = dataset.read_data_slice("CT", "CASE_000", (slice(None), slice(0, 4), slice(0, 5)))
    np.testing.assert_array_equal(first, volume)

    replacement = volume + 7
    dataset.write("CT", "CASE_000", replacement, attrs)
    second, _ = dataset.read_data_slice("CT", "CASE_000", (slice(None), slice(0, 4), slice(0, 5)))
    np.testing.assert_array_equal(second, replacement)


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


def test_dataset_iter_streams_patch_reads_when_cache_disabled(streaming_dataset_stub) -> None:
    volume = np.arange(1 * 4 * 4, dtype=np.float32).reshape(1, 4, 4)
    dataset_stub = streaming_dataset_stub(volume)
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


def test_dataset_iter_streams_base_patch_when_augmentations_are_disabled(streaming_dataset_stub) -> None:
    volume = np.arange(1 * 4 * 4, dtype=np.float32).reshape(1, 4, 4)
    dataset_stub = streaming_dataset_stub(volume)
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


def test_dataset_iter_streams_patch_reads_with_global_normalize_stats(streaming_dataset_stub) -> None:
    volume = np.arange(1 * 4 * 4, dtype=np.float32).reshape(1, 4, 4)
    dataset_stub = streaming_dataset_stub(volume)
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


def test_dataset_iter_streams_patch_reads_with_computed_standardize_stats(streaming_dataset_stub) -> None:
    volume = np.arange(1 * 4 * 4, dtype=np.float32).reshape(1, 4, 4)
    dataset_stub = streaming_dataset_stub(volume)
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


def test_dataset_is_dataset_exist_probes_the_entry_without_listing(tmp_path: Path) -> None:
    """Membership is a point question: one probe, not a slice of the directory listing. It used to build
    that listing, which both cost O(N) headers and froze an answer a run producing the group would
    outgrow (a ``Save`` writing into the dataset being read, from a loader worker)."""
    dataset = Dataset(tmp_path / "Dataset", "mha")
    attrs = _image_attributes([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    volume = np.zeros((1, 4, 4, 4), dtype=np.float32)
    dataset.write("CT", "CASE_000", volume, attrs)

    assert dataset.is_dataset_exist("CT", "CASE_000")
    assert not dataset.is_dataset_exist("CT", "CASE_999")
    assert "CT" not in dataset._names_cache, "membership must not enumerate the group"

    assert dataset.get_names("CT") == ["CASE_000"]  # enumeration still memoises, on its own call
    assert "CT" in dataset._names_cache


# --------------------------------------------------------------------------------------
# patch_transforms: the per-patch opt-in, guarded by the patch-locality contract
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


@pytest.fixture
def patch_manager(streaming_dataset_stub):
    """Factory for an overlapping-patch DatasetManager over the in-memory streaming stub."""

    def make(volume: np.ndarray, transforms: list[Transform]) -> DatasetManager:
        return DatasetManager(
            index=0,
            group_src="CT",
            group_dest="CT",
            name="CASE_000",
            dataset=cast(Dataset, streaming_dataset_stub(volume)),
            patch=DatasetPatch([8, 8, 8], overlap=4),
            transforms=transforms,
            data_augmentations_list=[],
        )

    return make


def test_patch_transform_standardize_applies_a_lazily_captured_volume_statistic(patch_manager) -> None:
    """`Standardize(lazy=True)` case-level + `Standardize()` per patch == case-level Standardize.

    This is the documented way to standardize per patch by the VOLUME's statistic: the lazy pass
    caches Mean/Std without applying anything, and the patch transform finds them on the attribute.
    """
    volume = _structured_volume()
    case_level = patch_manager(volume, [Standardize()])
    per_patch = patch_manager(volume, [Standardize(lazy=True)])

    size = case_level.patch.get_size(0)
    assert size > 1
    for index in range(size):
        expected = case_level.get_data(index, 0, [], True)
        got = per_patch.get_data(index, 0, [Standardize()], True)
        assert torch.equal(got, expected)


def test_patch_transform_standardize_uses_the_patch_own_statistic(patch_manager) -> None:
    """Asked for per-patch, a GLOBAL_STAT transform standardizes the patch by ITS OWN statistic."""
    volume = _structured_volume()
    manager = patch_manager(volume, [])

    patch = manager.get_data(0, 0, [Standardize()], True)

    source = torch.from_numpy(volume[:, 0:8, 0:8, 0:8])
    expected = (source - source.mean()) / source.std()
    assert torch.equal(patch, expected)
    # The patch's own mean is a long way from the volume's, so this really is the local statistic.
    assert abs(float(source.mean()) - float(torch.from_numpy(volume).mean())) > 100.0


def test_patch_transform_statistic_never_leaks_onto_the_case_attribute(patch_manager) -> None:
    """A patch-local statistic must not reach the attribute the whole case shares.

    Left there, the first patch read would freeze its own Mean/Std for every later patch: neither
    the volume's statistic nor the patch's, and dependent on the order the patches happen to be read.
    """
    volume = _structured_volume()
    manager = patch_manager(volume, [])

    manager.get_data(0, 0, [Standardize()], True)

    assert "Mean" not in manager.cache_attributes[0]
    assert "Std" not in manager.cache_attributes[0]


def test_patch_transform_standardize_is_independent_of_patch_order(patch_manager) -> None:
    """A patch's own statistic is the patch's alone: reading others first cannot change it."""
    volume = _structured_volume()
    forward = patch_manager(volume, [])
    backward = patch_manager(volume, [])

    size = forward.patch.get_size(0)
    first = forward.get_data(0, 0, [Standardize()], True)
    for index in reversed(range(size)):
        backward.get_data(index, 0, [Standardize()], True)
    last = backward.get_data(0, 0, [Standardize()], True)

    assert torch.equal(first, last)


def test_patch_transform_is_identical_across_managers(patch_manager) -> None:
    """A fresh manager per patch (the per-DataLoader-worker case) gives the same patch.

    Each worker owns its own cache attribute, so anything a patch records on it makes the result
    depend on which worker drew which patch. Every patch here must be reproducible on its own.
    """
    volume = _structured_volume()
    shared = patch_manager(volume, [])
    size = shared.patch.get_size(0)

    for index in range(size):
        assert torch.equal(
            patch_manager(volume, []).get_data(index, 0, [Standardize()], True),
            shared.get_data(index, 0, [Standardize()], True),
        )


def test_patch_transform_overlapping_patches_agree_on_shared_voxel(patch_manager) -> None:
    """With the volume statistic captured lazily, two overlapping patches agree on a shared voxel.

    A fresh manager per patch reproduces the per-DataLoader-worker case: the coefficients come from
    the case-level lazy pass, so they are the same in every worker.
    """
    volume = _structured_volume()
    size = patch_manager(volume, []).patch.get_size(0)

    values: dict[tuple[int, int, int], list[float]] = {}
    for index in range(size):
        manager = patch_manager(volume, [Standardize(lazy=True)])
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


def test_patch_transform_normalize_applies_a_lazily_captured_volume_range(patch_manager) -> None:
    volume = _structured_volume()
    manager = patch_manager(volume, [Normalize(lazy=True)])

    patch = manager.get_data(0, 0, [Normalize(min_value=-1, max_value=1)], True)

    # Mapped by the volume's range, so the first patch (low corner of the ramp) stays well
    # below the top of the target interval instead of being stretched onto it.
    assert float(manager.cache_attributes[0]["Min"]) == pytest.approx(float(volume.min()))
    assert float(manager.cache_attributes[0]["Max"]) == pytest.approx(float(volume.max()))
    assert float(patch.max()) < 0.0


def test_lazy_capture_reads_volume_statistics_once_per_case(streaming_dataset_stub) -> None:
    """The whole-volume statistic is a full disk scan: read it once, not once per patch."""
    volume = _structured_volume()
    stub = streaming_dataset_stub(volume)
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


def test_patch_transform_reads_no_disk_statistic_when_the_volume_is_loaded(
    streaming_dataset_stub, patch_manager
) -> None:
    """A loaded volume already holds the answer: the patch path must not go back to disk for it.

    The lazy pass computes Mean/Std from the tensor in hand (free, and carrying whatever the
    preceding chain did to it), so a `read_data_statistics` scan here would be both wasted and a
    statistic of the wrong (stored) version of the volume.
    """
    volume = _structured_volume()
    stub = streaming_dataset_stub(volume)
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
    reference = patch_manager(volume, case_level)
    reference.load(case_level, [], load_augmentations=False)
    for index in range(manager.patch.get_size(0)):
        assert torch.equal(manager.get_data(index, 0, [Standardize()], True), reference.get_data(index, 0, [], True))
    assert stub.stats_reads == 0


# --------------------------------------------------------------------------------------
# Seeding a GLOBAL_STAT from disk reads the statistics of the STORED volume, so it is only that
# transform's own input when nothing before it touched the values.
# --------------------------------------------------------------------------------------


def test_streaming_is_refused_when_a_transform_modifies_values_before_a_global_stat(patch_manager) -> None:
    """[Clip, Standardize] must not stream: on disk lie the PRE-Clip statistics.

    Clipping moves Mean and Std, so seeding Standardize from `read_data_statistics` would standardize
    every patch by a statistic of a volume that no longer exists. Refusing sends the case down the
    whole-volume path, where Standardize computes Mean/Std from the clipped tensor it is handed.
    """
    volume = _structured_volume()
    manager = patch_manager(volume, [Clip(min_value=200.0, max_value=1000.0), Standardize()])

    assert manager.can_stream_patch(0) is False
    assert "Mean" not in manager.cache_attributes[0]


def test_clip_then_standardize_equals_the_whole_volume_result(patch_manager) -> None:
    """The value every patch must carry: standardized by the CLIPPED volume's statistic."""
    volume = _structured_volume()
    chain: list[Transform] = [Clip(min_value=200.0, max_value=1000.0), Standardize()]
    manager = patch_manager(volume, chain)
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


def test_streaming_still_seeds_a_global_stat_behind_a_reorientation(patch_manager) -> None:
    """A flip only moves voxels, so the stored statistics are still Standardize's own input. The
    plan probe accepts without reading a voxel; the first data read seeds the statistic."""
    volume = _structured_volume()
    manager = patch_manager(volume, [Flip(dims="0"), Standardize()])

    assert manager.can_stream_patch(0) is True
    assert "Mean" not in manager.cache_attributes[0], "a plan probe reads headers only"
    manager.get_data(0, 0, [], True)
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
    """The guard runs at config time, from prepare(): not only when someone calls it directly."""
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


# --------------------------------------------------------------------------------------
# Read-side caching of chunked stores.
#
# Overlapping patch reads revisit the same chunks: the HDF5 read handle carries a chunk cache
# sized for imaging chunks (the library default holds barely one), and the OME-Zarr image
# handle is memoised per (store, level) so a streamed run parses the NGFF metadata once, not
# once per patch: invalidated by every write path, because a store just written must be
# re-read.
# --------------------------------------------------------------------------------------
def test_h5_read_handle_carries_an_imaging_sized_chunk_cache(tmp_path) -> None:
    dataset = Dataset(f"{tmp_path}/store.h5", "h5")
    dataset.write("group", "CASE_000", np.zeros((1, 4, 4, 4), dtype=np.float32), Attribute())
    with Dataset.File(f"{tmp_path}/store.h5", True, "h5") as backend:
        _, nslots, nbytes, _ = backend.h5.id.get_access_plist().get_cache()
    assert nbytes == Dataset.H5File._READ_CHUNK_CACHE_BYTES
    assert nslots == Dataset.H5File._READ_CHUNK_CACHE_SLOTS


def test_ome_zarr_image_is_memoised_per_store_and_invalidated_by_writes(tmp_path) -> None:
    pytest.importorskip("ngff_zarr")
    from konfai.utils.ome_zarr import _load_image, read_ome_zarr_data_slice, write_ome_zarr

    store = tmp_path / "case.ome.zarr"
    write_ome_zarr(store, np.ones((1, 4, 4, 4), dtype=np.float32))
    first = _load_image(str(store), 0)
    assert _load_image(str(store), 0) is first, "the image handle must be memoised per (store, level)"

    write_ome_zarr(store, np.full((1, 4, 4, 4), 7.0, dtype=np.float32))
    data, _ = read_ome_zarr_data_slice(store, tuple(slice(None) for _ in range(4)))
    assert float(data.max()) == 7.0, "a write must invalidate the memo so the new voxels are read"


# --------------------------------------------------------------------------------------
# ``Mask`` streams region by region: POINTWISE, and ``stream_region`` reads only the part of the
# aligned mask the region lines up with, reassembling byte-identically to the whole-volume
# ``__call__`` on both dispatchers (the finalize writer's slabs, the reader's regions).
# --------------------------------------------------------------------------------------
sitk = pytest.importorskip("SimpleITK")


def _write_mask(path, z, y, x, seed=0):
    rng = np.random.default_rng(seed)
    arr = (rng.random((z, y, x)) > 0.4).astype(np.uint8)
    sitk.WriteImage(sitk.GetImageFromArray(arr), str(path))


def _slab_context(z0: int, z1: int, shape: tuple[int, int, int]) -> RegionContext:
    region = (slice(z0, z1), slice(0, shape[1]), slice(0, shape[2]))
    return RegionContext(region, region, shape)


@pytest.mark.parametrize("slab", [1, 4, 5])
def test_mask_stream_region_reassembles_like_whole_volume(tmp_path, slab):
    # Single-channel output (the real case: a masked sCT); a [1,Z,Y,X] mask indexes a 1-channel volume.
    z, y, x = 12, 8, 7
    path = tmp_path / "mask.mha"
    _write_mask(path, z, y, x)
    torch.manual_seed(1)
    volume = torch.randn(1, z, y, x)

    m = Mask(path=str(path), value_outside=-999)
    assert m.patch_locality(Attribute()).kind is LocalityKind.POINTWISE

    whole = m("case", volume.clone(), Attribute())

    streamed = volume.clone()
    for z0 in range(0, z, slab):
        z1 = min(z0 + slab, z)
        rows = m.stream_region("case", streamed[:, z0:z1].clone(), _slab_context(z0, z1, (z, y, x)), Attribute())
        streamed[:, z0:z1] = rows

    assert torch.equal(streamed, whole)


def test_mask_stream_region_masks_the_right_voxels(tmp_path):
    # A concrete check the region is aligned, not just self-consistent: outside the mask -> value_outside,
    # inside -> untouched, and the streamed slabs land on the same voxels as the whole-volume call.
    z, y, x = 6, 5, 4
    path = tmp_path / "m.mha"
    _write_mask(path, z, y, x, seed=3)
    mask = torch.as_tensor(sitk.GetArrayFromImage(sitk.ReadImage(str(path)))).unsqueeze(0)
    m = Mask(path=str(path), value_outside=-1)
    out = torch.ones(1, z, y, x) * 7.0
    for z0 in range(0, z, 2):
        out[:, z0 : z0 + 2] = m.stream_region(
            "c", out[:, z0 : z0 + 2].clone(), _slab_context(z0, z0 + 2, (z, y, x)), Attribute()
        )
    assert torch.equal(out[mask == 0], torch.full_like(out[mask == 0], -1.0))
    assert torch.equal(out[mask != 0], torch.full_like(out[mask != 0], 7.0))
