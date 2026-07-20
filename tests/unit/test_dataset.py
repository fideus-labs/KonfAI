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

"""Tests for ``konfai.utils.dataset``: the ``Attribute`` sidecar, the SITK/HDF5 storage
backends (modes, locking, transforms, path resolution), and ``get_infos`` shape order."""

import os
import stat
import threading
from pathlib import Path

import numpy as np
import pytest
import torch
from konfai.utils.dataset import Attribute, Dataset, _get_h5_file_lock, get_infos, image_to_data
from konfai.utils.errors import DatasetManagerError

sitk = pytest.importorskip("SimpleITK")
h5py = pytest.importorskip("h5py")

# --------------------------------------------------------------------------------------
# B13 - Attribute keys containing '_' are stored raw and must be readable/poppable
# --------------------------------------------------------------------------------------


def test_attribute_underscore_key_is_readable_and_consistent() -> None:
    attribute = Attribute()
    attribute["ITK_InputFilterName"] = "GradientAnisotropicDiffusion"

    # __contains__ reports membership; the getter must agree with it.
    assert "ITK_InputFilterName" in attribute
    assert attribute["ITK_InputFilterName"] == "GradientAnisotropicDiffusion"
    assert attribute.is_info("ITK_InputFilterName", "GradientAnisotropicDiffusion")


def test_attribute_underscore_key_can_be_popped() -> None:
    attribute = Attribute()
    attribute["ITK_InputFilterName"] = "x"

    assert attribute.pop("ITK_InputFilterName") == "x"
    assert "ITK_InputFilterName" not in attribute


def test_attribute_stacked_lookup_still_wins_over_raw_fallback() -> None:
    """The predictor writes ``<key>_0`` explicitly and reads ``<key>`` back (stack scheme)."""
    attribute = Attribute()
    attribute["number_of_channels_per_model_0"] = torch.tensor([2, 3, 4])

    assert "number_of_channels_per_model" in attribute
    channels = attribute.pop_tensor("number_of_channels_per_model")
    assert torch.equal(channels, torch.tensor([2.0, 3.0, 4.0]))


def test_attribute_repeated_set_returns_latest_version() -> None:
    attribute = Attribute()
    attribute["Origin"] = np.asarray([1.0, 1.0, 1.0])
    attribute["Origin"] = np.asarray([5.0, 5.0, 5.0])

    np.testing.assert_array_equal(attribute.get_np_array("Origin"), np.asarray([5.0, 5.0, 5.0]))


# --------------------------------------------------------------------------------------
# HDF5 backend — directories, modes, and per-file locking
# --------------------------------------------------------------------------------------


def test_h5_dataset_creates_nested_parent_directories(tmp_path: Path, image_attributes) -> None:
    # B19 - the parent directory is created with pathlib (nested paths, OS separators).
    dataset = Dataset(tmp_path / "runs" / "exp" / "Volumes", "h5")
    volume = np.arange(1 * 2 * 2, dtype=np.float32).reshape(1, 2, 2)
    dataset.write("CT", "CASE_000", volume, image_attributes([0.0, 0.0], [1.0, 1.0]))

    assert (tmp_path / "runs" / "exp" / "Volumes.h5").exists()
    data, _ = dataset.read_data("CT", "CASE_000")
    np.testing.assert_array_equal(data, volume)


def test_read_data_opens_hdf5_read_only(tmp_path: Path, image_attributes) -> None:
    # read_data must open HDF5 in "r": an r+ open stamps a Date attribute on every read, which
    # mutates the file and breaks concurrent access across DataLoader/DDP processes. On a read-only
    # file an r+ open raises PermissionError, so a successful read here proves the mode is "r".
    volume = np.arange(1 * 3 * 4 * 5, dtype=np.int16).reshape(1, 3, 4, 5)
    dataset = Dataset(tmp_path / "H5DS", "h5")
    dataset.write("CT", "CASE_001", volume, image_attributes([10.0, 20.0, 30.0], [0.5, 1.5, 2.0]))

    h5_files = list(tmp_path.rglob("*.h5"))
    assert h5_files, "the write did not create an .h5 file"
    for h5_file in h5_files:
        os.chmod(h5_file, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)

    try:
        full, _ = dataset.read_data("CT", "CASE_001")
        np.testing.assert_array_equal(full, volume)
    finally:
        for h5_file in h5_files:
            os.chmod(h5_file, stat.S_IRUSR | stat.S_IWUSR)


def test_h5_writes_are_serialised_per_file(tmp_path: Path, image_attributes) -> None:
    # B6 - concurrent HDF5 access is serialised per file.
    dataset = Dataset(str(tmp_path / "Volumes"), "h5")
    attrs = image_attributes([0.0, 0.0], [1.0, 1.0])
    dataset.write("CT", "CASE_000", np.zeros((1, 2, 2), dtype=np.float32), attrs)

    lock = _get_h5_file_lock(dataset.filename + ".h5")  # the store's own key, whatever the OS separator
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


# --------------------------------------------------------------------------------------
# B23 - a missing sitk entry raises a clear error instead of UnboundLocalError
# --------------------------------------------------------------------------------------


def test_sitk_file_to_data_missing_entry_raises_nameerror(tmp_path: Path) -> None:
    root = tmp_path / "Dataset"
    root.mkdir()
    with Dataset.File(f"{root}/", True, "mha", 0) as file:
        with pytest.raises(NameError, match="not found"):
            file.file_to_data("", "missing_case")


# --------------------------------------------------------------------------------------
# B17 - unknown transform types raise a typed error at write/read (no UnboundLocalError,
#        no silent reuse of the previous type)
# --------------------------------------------------------------------------------------


def test_h5_write_unknown_transform_type_raises(tmp_path: Path) -> None:
    dataset = Dataset(tmp_path / "Transforms", "h5")
    composite = sitk.CompositeTransform([sitk.TranslationTransform(3, (1.0, 2.0, 3.0))])
    with pytest.raises(DatasetManagerError, match="Unsupported transform type"):
        dataset.write("T", "CASE_000", composite, Attribute())


def test_sitk_read_unknown_transform_type_raises(tmp_path: Path) -> None:
    dataset = Dataset(tmp_path / "Dataset", "mha")
    dataset.write("Transf", "CASE_000", sitk.TranslationTransform(3, (1.0, 2.0, 3.0)), Attribute())
    with pytest.raises(DatasetManagerError, match="Unsupported transform type"):
        dataset.read_transform("Transf", "CASE_000")


def test_read_transform_unknown_type_attribute_raises(tmp_path: Path) -> None:
    dataset = Dataset(tmp_path / "Transforms", "h5")
    euler = sitk.Euler3DTransform()
    euler.SetParameters((0.1, 0.2, 0.3, 4.0, 5.0, 6.0))
    dataset.write("T", "CASE_000", euler, Attribute())

    with h5py.File(str(tmp_path / "Transforms.h5"), "r+") as handle:
        handle["T/CASE_000"].attrs["0:Transform_0"] = "MysteryTransform_double_3_3"

    with pytest.raises(DatasetManagerError, match="Unsupported transform type"):
        dataset.read_transform("T", "CASE_000")


def test_supported_transform_types_round_trip(tmp_path: Path) -> None:
    dataset = Dataset(tmp_path / "Dataset", "mha")
    euler = sitk.Euler3DTransform()
    euler.SetParameters((0.1, 0.2, 0.3, 4.0, 5.0, 6.0))
    dataset.write("Transf", "CASE_000", euler, Attribute())

    restored = dataset.read_transform("Transf", "CASE_000")

    assert isinstance(restored, sitk.Euler3DTransform)
    np.testing.assert_allclose(restored.GetParameters(), (0.1, 0.2, 0.3, 4.0, 5.0, 6.0))


# --------------------------------------------------------------------------------------
# B15 - the XML branch returns a (data, attributes) tuple, not a bare lxml element
# --------------------------------------------------------------------------------------


def test_xml_file_to_data_returns_tuple_with_parsed_values(tmp_path: Path) -> None:
    dataset = Dataset(tmp_path / "Dataset", "mha")
    attributes = Attribute()
    attributes["path"] = "level1:level2"
    attributes["foo"] = "bar"
    dataset.write("Node", "CASE_000", np.asarray([1.5, 2.5, 3.5]), attributes)

    result = dataset.read_data("Node", "CASE_000")

    assert isinstance(result, tuple) and len(result) == 2
    data, read_attributes = result
    assert isinstance(data, np.ndarray)
    np.testing.assert_allclose(data, [1.5, 2.5, 3.5])
    assert read_attributes["foo"] == "bar"


# --------------------------------------------------------------------------------------
# B24 - streaming path resolution follows the same precedence as full read
# --------------------------------------------------------------------------------------


def test_resolve_data_path_prefers_special_format_like_full_read(tmp_path: Path, image_attributes) -> None:
    root = tmp_path / "Dataset"
    dataset = Dataset(root, "mha")
    dataset.write(
        "Transf",
        "CASE_000",
        np.arange(1 * 2 * 3 * 4, dtype=np.float32).reshape(1, 2, 3, 4),
        image_attributes([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
    )
    euler = sitk.Euler3DTransform()
    euler.SetParameters((0.1, 0.2, 0.3, 4.0, 5.0, 6.0))
    dataset.write("Transf", "CASE_000", euler, Attribute())

    # Both Transf.mha and Transf.itk.txt now exist for the same entry.
    sitk_file = Dataset.SitkFile(f"{root}/CASE_000/", True, "mha")
    resolved = sitk_file._resolve_data_path("Transf")

    # read_data (full path) picks the transform; the streaming resolver must agree.
    assert resolved is not None and resolved.endswith(".itk.txt")
    full, _ = dataset.read_data("Transf", "CASE_000")
    assert full.shape == (1, 6)


def test_resolve_data_path_skips_a_crashed_writer_temporary(tmp_path: Path, image_attributes) -> None:
    # A hard-killed streamed write leaves a ``.tmp`` (header + zero-reserved pixels); the resolver must
    # never hand it back as the volume when the final entry is absent -- glob would otherwise sort it first.
    root = tmp_path / "Dataset"
    (root / "CASE_000").mkdir(parents=True)
    (root / "CASE_000" / "Transf.mha.9999-0.tmp").write_bytes(b"leftover debris")

    sitk_file = Dataset.SitkFile(f"{root}/CASE_000/", True, "mha")
    assert sitk_file._resolve_data_path("Transf") is None


# --------------------------------------------------------------------------------------
# get_infos returns numpy channel-first order for every rank
#
# Patch planning strips the channel from get_infos' shape and feeds the spatial shape to
# transform_shape and the patch reader; the actual pixel reads (image_to_data /
# _file_to_image_slice) are numpy-order [C, (T), (Z), Y, X]. Reversing sitk GetSize() only
# when len == 3 leaves 2-D and 4-D images in sitk (x, y, ...) order, transposed against
# their own pixel data.
# --------------------------------------------------------------------------------------


def test_get_infos_2d_matches_pixel_data(tmp_path: Path) -> None:
    # Non-square 2-D: sitk GetSize() = (x=10, y=4); numpy pixel data is (y=4, x=10).
    path = tmp_path / "img2d.nii.gz"
    sitk.WriteImage(sitk.GetImageFromArray(np.zeros((4, 10), dtype=np.float32)), str(path))

    size, _ = get_infos(path)
    data, _ = image_to_data(sitk.ReadImage(str(path)))

    assert list(size) == list(data.shape)  # [1, 4, 10], not [1, 10, 4]


def test_get_infos_4d_matches_pixel_data(tmp_path: Path) -> None:
    # Genuine 4-D scalar: sitk GetSize() = (5, 4, 3, 2); numpy pixel data is (2, 3, 4, 5).
    path = tmp_path / "img4d.nii.gz"
    sitk.WriteImage(sitk.Image([5, 4, 3, 2], sitk.sitkFloat32), str(path))

    size, _ = get_infos(path)
    data = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))

    assert list(size) == [1, *data.shape]  # [1, 2, 3, 4, 5]


def test_get_infos_3d_unchanged(tmp_path: Path) -> None:
    # The 3-D path must stay reversed.
    path = tmp_path / "img3d.nii.gz"
    sitk.WriteImage(sitk.GetImageFromArray(np.zeros((6, 4, 10), dtype=np.float32)), str(path))

    size, _ = get_infos(path)
    data, _ = image_to_data(sitk.ReadImage(str(path)))

    assert list(size) == list(data.shape) == [1, 6, 4, 10]


def test_sitkfile_get_infos_2d_matches_read_data(tmp_path: Path) -> None:
    # The same contract holds for SitkFile.get_infos, reached through the public Dataset API.
    ds_dir = str(tmp_path / "ds") + "/"
    Path(ds_dir).mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(sitk.GetImageFromArray(np.zeros((4, 10), dtype=np.float32)), ds_dir + "case0.mha")

    file = Dataset.SitkFile(ds_dir, read=True, file_format="mha")
    size, _ = file.get_infos("", "case0")
    data, _ = file.file_to_data("", "case0")

    assert list(size) == list(data.shape)  # [1, 4, 10]


def test_attribute_setitem_accepts_0d_and_autograd_tensors() -> None:
    # Finalize transforms (Normalize, Statistics) store stats computed from the prediction volume,
    # which arrive as 0-d tensors — possibly CUDA-resident and/or still attached to a graph. The
    # host-side string conversion must detach and move them itself.
    attribute = Attribute()
    attribute["ImageMin"] = torch.tensor(3.5)
    attribute["Weight"] = torch.tensor(2.0, requires_grad=True)

    assert float(attribute["ImageMin"]) == 3.5
    assert float(attribute["Weight"]) == 2.0


# --------------------------------------------------------------------------------------
# Directory store-format auto-detection: the read backend is chosen from what is on disk
# (an OME-Zarr/Zarr store or a DICOM series directory), so a ``:mha`` token cannot
# force a store to be mis-read. Plain per-file volumes keep the SitkFile path.
# --------------------------------------------------------------------------------------


def _make_case(root: Path, entry: str, *, is_dir: bool = True, marker: str | None = None, files=()) -> Path:
    case = root / "P000"
    case.mkdir(parents=True, exist_ok=True)
    target = case / entry
    if is_dir:
        target.mkdir()
        if marker:
            (target / marker).write_text("{}", encoding="utf-8")
        for name in files:
            (target / name).write_bytes(b"")
    else:
        target.write_bytes(b"")
    return root


def test_autodetect_ome_zarr_by_suffix(tmp_path: Path) -> None:
    root = _make_case(tmp_path / "ds", "Volume_0.ome.zarr")
    assert Dataset._detect_directory_store_format(str(root)) == "omezarr"


def test_autodetect_zarr_by_group_marker(tmp_path: Path) -> None:
    root = _make_case(tmp_path / "ds", "Volume_0", marker=".zgroup")
    assert Dataset._detect_directory_store_format(str(root)) == "omezarr"


def test_autodetect_dicom_series_directory(tmp_path: Path) -> None:
    root = _make_case(tmp_path / "ds", "Volume_0", files=("000000.dcm",))
    assert Dataset._detect_directory_store_format(str(root)) == "dicom"


def test_autodetect_plain_files_return_none(tmp_path: Path) -> None:
    root = _make_case(tmp_path / "ds", "Volume_0.mha", is_dir=False)
    assert Dataset._detect_directory_store_format(str(root)) is None


def test_init_overrides_mha_token_for_ome_zarr_store(tmp_path: Path) -> None:
    root = _make_case(tmp_path / "ds", "Volume_0.ome.zarr")
    # the token says mha, but the store on disk is OME-Zarr -> the read backend follows the disk
    assert Dataset(str(root), "mha").file_format == "omezarr"


def test_init_keeps_token_for_plain_file_dataset(tmp_path: Path) -> None:
    root = _make_case(tmp_path / "ds", "Volume_0.mha", is_dir=False)
    assert Dataset(str(root), "mha").file_format == "mha"


def test_a_statistics_chunk_is_budgeted_with_its_channels() -> None:
    # A chunk spans every other axis whole, the channels included, and is accumulated in float64. Cut
    # on a plane alone, a 122-channel volume holds 122 times the budget -- 7 GiB where 0.06 was meant.
    from konfai.utils.dataset import _STATISTICS_CHUNK_ELEMENTS, _statistics_chunk_length

    for channels in (1, 4, 122):
        shape = [channels, 400, 512, 512]
        length = _statistics_chunk_length(shape, axis=1)
        held = channels * length * 512 * 512
        # One step is the floor, so a volume whose step alone overflows is read a step at a time.
        assert held <= max(_STATISTICS_CHUNK_ELEMENTS, channels * 512 * 512)


def test_a_statistics_chunk_reaches_further_on_a_thin_volume() -> None:
    from konfai.utils.dataset import _statistics_chunk_length

    assert _statistics_chunk_length([1, 400, 64, 64], axis=1) > _statistics_chunk_length([1, 400, 512, 512], axis=1)


def test_directory_store_detects_extensionless_dicom(tmp_path: Path) -> None:
    # A DICOM series exported with no extension must be detected by content: suffix-only
    # detection leaves it on the SitkFile backend.
    series = tmp_path / "ds" / "case_0" / "ser"
    series.mkdir(parents=True)
    (series / "IM000001").write_bytes(b"\x00" * 128 + b"DICM" + b"\x00" * 32)

    assert Dataset._detect_directory_store_format(f"{tmp_path}/ds/") == "dicom"


def test_dataset_rebase_keeps_h5_a_file_and_directory_formats_a_directory() -> None:
    # Predictor.rebase must not flag an h5 output as a directory: an unconditional trailing "/"
    # makes the single-store writer write the hidden dotfile <dir>/.h5.
    from pathlib import Path

    from konfai.utils.dataset import Dataset

    h5 = Dataset("Dataset", "h5")
    h5.rebase(Path("Predictions/run"))
    assert h5.filename == "Predictions/run/Dataset"  # a file, not "…/Dataset/" -> ".h5"
    assert h5.is_directory is False

    mha = Dataset("Dataset", "mha")
    mha.rebase(Path("Predictions/run"))
    assert mha.filename == "Predictions/run/Dataset/"
    assert mha.is_directory is True


def test_attribute_lookup_is_not_fooled_by_a_prefixing_sibling_key() -> None:
    # Values stack as {key}_{n}; a startswith(key) count treats SpacingOriginal as a second Spacing
    # entry, so a["Spacing"] raises while "Spacing" in a still answers True.
    from konfai.utils.dataset import Attribute

    attribute = Attribute()
    attribute["Spacing"] = "1.0 1.0 2.0"
    attribute["SpacingOriginal"] = "0.5 0.5 1.0"

    assert "Spacing" in attribute
    assert attribute["Spacing"] == "1.0 1.0 2.0"
    assert attribute["SpacingOriginal"] == "0.5 0.5 1.0"


def test_get_infos_reads_only_the_header_for_a_mismatched_extension(tmp_path: Path, monkeypatch) -> None:
    """An entry stored with a different extension than the dataset's file_format must still take the
    header-only path: the file_to_data fallback decodes the whole volume on the patch-planning path."""
    sitk = pytest.importorskip("SimpleITK")
    root = tmp_path / "Dataset"
    root.mkdir()
    image = sitk.GetImageFromArray(np.zeros((4, 5, 6), dtype=np.float32))
    image.SetSpacing((1.5, 1.5, 2.0))
    sitk.WriteImage(image, str(root / "case.nii.gz"))

    with Dataset.File(f"{root}/", True, "mha", 0) as file:
        full_reads: list[str] = []
        original = file.file_to_data
        monkeypatch.setattr(file, "file_to_data", lambda *a, **k: (full_reads.append("hit"), original(*a, **k))[1])
        size, attributes = file.get_infos("", "case")

    assert size == [1, 4, 5, 6]
    assert full_reads == [], "a readable image header must never trigger a full-volume decode"
    assert np.allclose(attributes.get_np_array("Spacing"), [1.5, 1.5, 2.0])
