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

import multiprocessing
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


def test_attribute_built_from_a_store_sidecar_holds_text_a_writer_accepts() -> None:
    """An OME-Zarr sidecar is JSON, so it hands back live lists, not their string form.

    Both doors normalize to text, construction included: a value deep-copied through construction
    untouched reaches ``Image.SetMetaData``, which accepts only ``std::string``: a field that
    can be written but never reopened.
    """
    attribute = Attribute({"WorldReach": [1.19, 2.39, 3.59], "Spacing": (1.5, 1.5, 2.0)})

    assert all(isinstance(value, str) for value in attribute.values())
    # And readable back: a list prints comma-separated, which np.fromstring alone could not read.
    np.testing.assert_allclose(attribute.get_np_array("WorldReach"), [1.19, 2.39, 3.59])
    np.testing.assert_allclose(attribute.get_np_array("Spacing"), [1.5, 1.5, 2.0])


def test_attribute_assigned_a_plain_list_round_trips_as_an_array() -> None:
    """Both doors normalize the same way, so a list assigned in Python reads back like an ndarray."""
    attribute = Attribute()
    attribute["Origin"] = [1.0, 2.0, 3.0]

    np.testing.assert_allclose(attribute.get_np_array("Origin"), [1.0, 2.0, 3.0])


def test_attribute_holding_a_long_array_round_trips_past_numpys_print_threshold() -> None:
    """An attribute is a record, not a display: an elided value is one no reader can parse back."""
    attribute = Attribute()
    attribute["Long"] = np.arange(2000, dtype=float)

    np.testing.assert_allclose(attribute.get_np_array("Long"), np.arange(2000, dtype=float))


# --------------------------------------------------------------------------------------
# HDF5 backend: directories, modes, and per-file locking
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
    # never hand it back as the volume when the final entry is absent: glob would otherwise sort it first.
    root = tmp_path / "Dataset"
    (root / "CASE_000").mkdir(parents=True)
    (root / "CASE_000" / "Transf.mha.9999-0.tmp").write_bytes(b"leftover debris")

    sitk_file = Dataset.SitkFile(f"{root}/CASE_000/", True, "mha")
    assert sitk_file._resolve_data_path("Transf") is None
    # The full read must agree with the slice/statistics paths: a missing entry raises, never returns the
    # temporary as a (partial) volume.
    with pytest.raises(NameError, match="not found"):
        sitk_file.file_to_data("", "Transf")


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
    # which arrive as 0-d tensors: possibly CUDA-resident and/or still attached to a graph. The
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
    # on a plane alone, a 122-channel volume holds 122 times the budget: 7 GiB where 0.06 was meant.
    from konfai.utils.dataset import _STATISTICS_CHUNK_ELEMENTS, _statistics_chunk_length

    for channels in (1, 4, 122):
        shape = [channels, 400, 512, 512]
        length = _statistics_chunk_length(shape, axis=1, budget=_STATISTICS_CHUNK_ELEMENTS)
        held = channels * length * 512 * 512
        # One step is the floor, so a volume whose step alone overflows is read a step at a time.
        assert held <= max(_STATISTICS_CHUNK_ELEMENTS, channels * 512 * 512)


def test_a_statistics_chunk_reaches_further_on_a_thin_volume() -> None:
    from konfai.utils.dataset import _statistics_chunk_length

    thin, wide = [1, 400, 64, 64], [1, 400, 512, 512]
    assert _statistics_chunk_length(thin, 1, budget=1 << 20) > _statistics_chunk_length(wide, 1, budget=1 << 20)


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


def test_get_infos_reads_a_npy_header_off_the_map(tmp_path: Path, monkeypatch) -> None:
    """A ``.npy`` entry answers its shape from the header: the statistics fold and the plan ask for it
    before any block, and a whole load there is the volume in memory the fold exists to avoid."""
    root = tmp_path / "Dataset" / "case"
    root.mkdir(parents=True)
    np.save(root / "params.npy", np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5))
    original = np.load

    def mapped_only(path, *args, **kwargs):
        assert kwargs.get("mmap_mode") == "r", "a .npy is never loaded whole for its header"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", mapped_only)
    dataset = Dataset(str(tmp_path / "Dataset"), "mha")
    assert dataset.get_infos("params", "case")[0] == [2, 3, 4, 5]
    assert dataset.read_data_statistics("params", "case")["max"] == 119.0


def test_a_group_written_through_another_dataset_object_is_seen(tmp_path: Path) -> None:
    """A group can be produced through one Dataset and read through another over the same folder: a
    ``Save`` builds its own (``Save.destination``) while the reader keeps the DataManager's.
    Membership answered from the reader's memoised listing froze at its first lookup, so every case
    written after it read as absent. ImpactSynth masks its own output that way and raised
    ``NameError: Mask : MASK/P002 not found`` from the third case of a batch on."""
    root = tmp_path / "ds"
    root.mkdir()
    reader = Dataset(str(root) + "/", "mha")
    writer = Dataset(str(root) + "/", "mha")

    writer.write("MASK", "P000", np.ones((4, 4, 4), dtype=np.uint8), Attribute())
    assert reader.get_names("MASK") == ["P000"]  # the reader memoises the listing here

    writer.write("MASK", "P001", np.ones((4, 4, 4), dtype=np.uint8), Attribute())
    writer.write("MASK", "P002", np.ones((4, 4, 4), dtype=np.uint8), Attribute())

    assert reader.is_dataset_exist("MASK", "P001")
    assert reader.is_dataset_exist("MASK", "P002")


def test_membership_is_asked_of_disk_not_of_the_listing(tmp_path: Path) -> None:
    """``get_names`` is a planning-time enumeration; asking it whether ONE case exists answers from a
    snapshot. A hit may come from the memo (an entry never disappears mid-run), but a miss must be
    checked, or the listing's age becomes the answer."""
    root = tmp_path / "ds"
    root.mkdir()
    dataset = Dataset(str(root) + "/", "mha")
    (root / "P000").mkdir()
    sitk.WriteImage(sitk.GetImageFromArray(np.ones((4, 4, 4), dtype=np.uint8)), str(root / "P000" / "MASK.mha"))

    assert dataset.get_names("MASK") == ["P000"]  # memoise the listing
    (root / "P001").mkdir()
    sitk.WriteImage(sitk.GetImageFromArray(np.ones((4, 4, 4), dtype=np.uint8)), str(root / "P001" / "MASK.mha"))

    assert dataset.is_dataset_exist("MASK", "P001")
    assert dataset.is_dataset_exist("MASK", "P000")  # the memoised hit still answers
    assert not dataset.is_dataset_exist("MASK", "P999")


def _write_mask_in_child(root: str, case: str) -> None:
    from konfai.utils.dataset import Attribute as ChildAttribute
    from konfai.utils.dataset import Dataset as ChildDataset

    ChildDataset(root, "mha").write("MASK", case, np.ones((4, 4, 4), dtype=np.uint8), ChildAttribute())


def test_membership_sees_an_entry_written_by_another_process(tmp_path: Path) -> None:
    """The loader's ``Save`` runs in a DataLoader worker while the output transform reads in the parent,
    so no in-process memo can be invalidated across that boundary. Membership has to ask the disk."""
    root = str(tmp_path / "ds") + "/"
    Path(root).mkdir()
    reader = Dataset(root, "mha")
    reader.write("MASK", "P000", np.ones((4, 4, 4), dtype=np.uint8), Attribute())
    assert reader.get_names("MASK") == ["P000"]  # the parent memoises its listing here

    child = multiprocessing.get_context("spawn").Process(target=_write_mask_in_child, args=(root, "P001"))
    child.start()
    child.join(120)
    assert child.exitcode == 0, "the writer process failed; the assertion below would prove nothing"

    assert reader.is_dataset_exist("MASK", "P001")


def _write_mask_in_child_h5(root: str, case: str) -> None:
    from konfai.utils.dataset import Attribute as ChildAttribute
    from konfai.utils.dataset import Dataset as ChildDataset

    ChildDataset(root, "h5").write("MASK", case, np.ones((4, 4, 4), dtype=np.uint8), ChildAttribute())


def test_membership_sees_an_h5_entry_written_by_another_process(tmp_path: Path) -> None:
    """A single store answers the same way a directory does. The pooled read handle used to keep serving
    the view it opened on, and reopening alone would not have helped: HDF5 shares a file's metadata state
    across the handles one process holds, so a second handle inherits the first's. The pool now closes a
    handle whose store changed underneath it."""
    pytest.importorskip("h5py")
    root = str(tmp_path / "ds") + "/"
    Path(root).mkdir()
    reader = Dataset(root, "h5")
    reader.write("MASK", "P000", np.ones((4, 4, 4), dtype=np.uint8), Attribute())
    assert reader.get_names("MASK") == ["P000"]

    child = multiprocessing.get_context("spawn").Process(target=_write_mask_in_child_h5, args=(root, "P001"))
    child.start()
    child.join(120)
    assert child.exitcode == 0, "the writer process failed; the assertion below would prove nothing"

    assert reader.is_dataset_exist("MASK", "P001")


def test_an_evicted_h5_handle_goes_back_with_the_view_it_had(tmp_path: Path) -> None:
    """A handle evicted while its file lock is busy returns to the pool instead of being closed. Re-stamping
    it on the way back would hand it the store as it is now and launder a stale view into a fresh-looking
    one, so the write that arrived meanwhile would stay invisible for the rest of the process."""
    pytest.importorskip("h5py")
    from konfai.utils.dataset import _h5_read_pool

    root = str(tmp_path / "ds") + "/"
    Path(root).mkdir()
    dataset = Dataset(root, "h5")
    dataset.write("MASK", "P000", np.ones((4, 4, 4), dtype=np.uint8), Attribute())
    dataset.is_dataset_exist("MASK", "P000")  # pool a handle on the store
    store = dataset.filename + ".h5"
    pooled = _h5_read_pool._handles.pop(store)

    child = multiprocessing.get_context("spawn").Process(target=_write_mask_in_child_h5, args=(root, "P001"))
    child.start()
    child.join(120)
    assert child.exitcode == 0, "the writer process failed; the assertions below would prove nothing"

    # The file lock is reentrant, so only another thread can make it look busy to the evicting one.
    held, release = threading.Event(), threading.Event()

    def hold_the_file_lock() -> None:
        with _get_h5_file_lock(store):
            held.set()
            release.wait(120)

    holder = threading.Thread(target=hold_the_file_lock)
    holder.start()
    try:
        assert held.wait(120)
        _h5_read_pool._close_idle(store, pooled)
        assert _h5_read_pool._handles[store].opened_on == pooled.opened_on, "the view it had, not the store now"
    finally:
        release.set()
        holder.join(120)

    assert dataset.is_dataset_exist("MASK", "P001")


@pytest.mark.parametrize("file_format", ["mha", "h5", "nii.gz"])
def test_read_data_quantile_is_numpys_without_holding_the_volume(
    tmp_path: Path, image_attributes, monkeypatch: pytest.MonkeyPatch, file_format: str
) -> None:
    """The value, dtype and interpolation of numpy.quantile (method 'linear'), from bounded passes:
    a heavy bin (60 % of the voxels at one value), integers, a constant volume, and the top of the
    range all land on the same scalar. A store with bounded region reads is never read whole."""
    if file_format == "nii.gz":
        pytest.importorskip("SimpleITK")
    rng = np.random.default_rng(1)
    volumes = {
        "f32": (rng.random((2, 40, 30, 20)) * 1000).astype(np.float32),
        "ct": np.clip(rng.normal(0, 300, (1, 60, 50, 40)), -1024, 3000).astype(np.int16),
        "air": np.where(rng.random((1, 60, 50, 40)) < 0.6, -1024.0, rng.random((1, 60, 50, 40)) * 100).astype(
            np.float32
        ),
        "const": np.full((1, 8, 8, 8), 3.0, np.float32),
    }
    dataset = Dataset(tmp_path / "store", file_format)
    for name, volume in volumes.items():
        dataset.write("G", name, volume, image_attributes([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]))
    if dataset.bounded_region_reads("G", "f32"):
        monkeypatch.setattr(Dataset, "read_data", lambda *_: pytest.fail("the scan must not read the whole volume"))
    for name, volume in volumes.items():
        # 0 and 1 land on an exact index (weight 0): the branch that returns an order statistic
        # untouched, where numpy still answers in float64 for the int16 volume.
        for q in (0.0, 0.05, 0.5, 0.999, 1.0):
            got = dataset.read_data_quantile("G", name, q)
            expected = np.quantile(volume, q)
            assert type(got) is type(expected), (name, q, got, expected)
            # numpy's own interpolation moved by an ulp between releases: equal, or within one.
            tolerance = (
                2 * np.finfo(expected.dtype).eps * abs(float(expected))
                if np.issubdtype(type(expected), np.floating)
                else 0
            )
            assert abs(float(got) - float(expected)) <= tolerance, (name, q, got, expected)
    # A NaN anywhere makes numpy's quantile NaN; the scan answers the same instead of narrowing a
    # histogram no bin of which can hold the NaN.
    holed = volumes["f32"].copy()
    holed[0, 3, 4, 5] = np.nan
    dataset.write("G", "holed", holed, image_attributes([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]))
    kept = any(np.isnan(block).any() for block in dataset.iter_data_blocks("G", "holed")())
    if kept:  # a NIfTI writer sanitises non-finite values; where the NaN survives, so does numpy's answer
        assert np.isnan(dataset.read_data_quantile("G", "holed", 0.05))


def test_a_transform_file_the_h5_pool_holds_stays_readable_as_a_transform(tmp_path: Path) -> None:
    """HDF5 refuses to open a file this process already holds under the other file-locking flag. The h5
    read pool opens unlocked and keeps its handle for the life of the process, so the itktransform reader
    must open the same way, or a transform file once served by the h5 backend stops being a transform."""
    attributes = Attribute()
    attributes["Origin"] = np.asarray([1.0, 2.0, 3.0])
    attributes["Spacing"] = np.asarray([1.0, 1.0, 2.0])
    attributes["Direction"] = np.eye(3).reshape(-1)
    field = np.arange(3 * 4 * 5 * 6, dtype=np.float32).reshape(3, 4, 5, 6)
    root = tmp_path / "out"
    Dataset(root, "itktransform").write("Transform", "P000", field, attributes)

    # Read through the h5 backend: the pooled handle stays open on the file, unlocked.
    fixed, _ = Dataset(root / "P000" / "Transform", "h5").read_data("TransformGroup/0", "TransformFixedParameters")
    assert fixed.shape == (18,)

    transforms = Dataset(root, "itktransform")
    shape, read_attributes = transforms.get_infos("Transform", "P000")
    assert shape == [3, 4, 5, 6]
    np.testing.assert_array_equal(read_attributes.get_np_array("Origin"), [1.0, 2.0, 3.0])
    region, _ = transforms.read_data_slice("Transform", "P000", (slice(0, 3), slice(1, 3), slice(0, 5), slice(0, 6)))
    np.testing.assert_array_equal(region, field[:, 1:3])


def test_a_streamed_transform_entry_replaces_a_tfm_under_the_h5_name(tmp_path: Path) -> None:
    """ITK picks a transform's IO from its extension, so HDF5 content must land under `.h5`, never be
    renamed onto an existing `.tfm` of the same entry (which is what resolving the final path would do)."""
    attributes = Attribute()
    attributes["Origin"] = np.asarray([0.0, 0.0, 0.0])
    attributes["Spacing"] = np.asarray([1.0, 1.0, 1.0])
    attributes["Direction"] = np.eye(3).reshape(-1)
    root = tmp_path / "out"
    (root / "P000").mkdir(parents=True)
    sitk.WriteTransform(sitk.TranslationTransform(3, (1.0, 2.0, 3.0)), str(root / "P000" / "Transform.tfm"))

    field = np.ones((3, 4, 5, 6), dtype=np.float32)
    dataset = Dataset(root, "itktransform")
    stream = dataset.open_data_stream("Transform", "P000", list(field.shape), field.dtype, attributes)
    assert stream is not None
    with stream:
        stream.write_slice((slice(0, 3), slice(0, 4), slice(0, 5), slice(0, 6)), field)

    assert (root / "P000" / "Transform.h5").exists()
    transform = dataset.read_transform("Transform", "P000")
    assert "DisplacementFieldTransform" in transform.GetName()


def test_the_readers_own_itk_keys_do_not_travel_with_the_volume(tmp_path: Path) -> None:
    """SimpleITK stamps ITK_InputFilterName / ITK_original_direction / ITK_original_spacing on what it
    reads; carried into an output they describe the source, not the volume they land on."""
    sitk = pytest.importorskip("SimpleITK")
    image = sitk.GetImageFromArray(np.zeros((3, 4, 5), dtype=np.float32))
    image.SetMetaData("Study", "phantom")
    sitk.WriteImage(image, str(tmp_path / "x.mha"))
    read = sitk.ReadImage(str(tmp_path / "x.mha"))
    assert any(key.startswith("ITK_") for key in read.GetMetaDataKeys()), "the reader stamps its keys"
    _, attributes = image_to_data(read)
    assert attributes["Study"] == "phantom"
    assert not [key for key in attributes.keys() if key.startswith("ITK_")]


# --------------------------------------------------------------------------------------
# Region reads off the raw pixel block of an uncompressed MetaImage / NIfTI
# --------------------------------------------------------------------------------------

_BLOCK_ORIGIN, _BLOCK_SPACING = [10.0, -20.5, 30.25], [0.7, 1.3, 2.1]
_BLOCK_DIRECTION = np.asarray([0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
# A rotation: ITK's index-to-world arithmetic and numpy's matrix product then differ by an ulp,
# which the record of a region read keeps apart (one rung each).
_BLOCK_ROTATED = np.asarray(
    [[np.cos(0.3), -np.sin(0.3), 0.0], [np.sin(0.3), np.cos(0.3), 0.0], [0.0, 0.0, 1.0]]
).reshape(-1)
# What the block route serves, and what it leaves to ITK: the fixtures of the tests below.
_BLOCK_SERVED = (
    "scalar.mha",
    "vector.mha",
    "rotated.mha",
    "scalar.nii",
    "vector.nii",
    "streamed.mha",
    "streamed.nii",
    "bigendian.mha",
    "plane.mha",
    "identity.nii",
)
_BLOCK_LEFT_TO_ITK = ("compressed.mha", "scalar.nii.gz", "detached.mhd", "scaled.nii", "scalar.nrrd")


def _block_image(data: np.ndarray, direction: np.ndarray) -> "sitk.Image":
    rank = data.ndim - 1
    if data.shape[0] == 1:
        image = sitk.GetImageFromArray(data[0])
    else:
        image = sitk.GetImageFromArray(np.moveaxis(data, 0, -1), isVector=True)
    image.SetOrigin(_BLOCK_ORIGIN[:rank])
    image.SetSpacing(_BLOCK_SPACING[:rank])
    image.SetDirection(direction.reshape(3, 3)[:rank, :rank].reshape(-1).tolist())
    image.SetMetaData("Study", "phantom")
    return image


def _write_block_fixture(root: Path, kind: str) -> tuple[Path, np.ndarray]:
    """One file of ``kind`` under ``root``, and the channel-first array it holds."""
    from konfai.utils.dataset import _MhaDataStream, _NiftiDataStream

    rng = np.random.default_rng(len(kind))
    scalar = (rng.normal(size=(1, 12, 14, 16)) * 100).astype(np.float32)
    vector = (rng.normal(size=(3, 12, 14, 16)) * 100).astype(np.int16)
    path = root / kind
    if kind == "vector.mha":
        sitk.WriteImage(_block_image(vector, _BLOCK_DIRECTION), str(path))
        return path, vector
    if kind == "rotated.mha":
        sitk.WriteImage(_block_image(scalar, _BLOCK_ROTATED), str(path))
        return path, scalar
    if kind == "scalar.nii":
        stored = rng.integers(0, 4000, size=scalar.shape).astype(np.uint16)
        sitk.WriteImage(_block_image(stored, _BLOCK_DIRECTION), str(path))
        return path, stored
    if kind == "vector.nii":
        stored = vector.astype(np.float32)
        sitk.WriteImage(_block_image(stored, _BLOCK_DIRECTION), str(path))
        return path, stored
    if kind in ("streamed.mha", "streamed.nii"):
        stored = vector if kind == "streamed.mha" else vector.astype(np.float32)
        attributes = Attribute()
        attributes["Origin"] = np.asarray(_BLOCK_ORIGIN)
        attributes["Spacing"] = np.asarray(_BLOCK_SPACING)
        attributes["Direction"] = _BLOCK_DIRECTION
        stream_class = _MhaDataStream if kind == "streamed.mha" else _NiftiDataStream
        with stream_class(str(path), list(stored.shape), stored.dtype, attributes) as stream:
            stream.write_slice(tuple(slice(0, extent) for extent in stored.shape), stored)
        return path, stored
    if kind == "bigendian.mha":
        header = (
            "ObjectType = Image\nNDims = 3\nBinaryData = True\nBinaryDataByteOrderMSB = True\n"
            "CompressedData = False\nTransformMatrix = "
            + " ".join(str(v) for v in _BLOCK_DIRECTION.reshape(3, 3).T.reshape(-1))
            + "\nOffset = "
            + " ".join(str(v) for v in _BLOCK_ORIGIN)
            + "\nElementSpacing = "
            + " ".join(str(v) for v in _BLOCK_SPACING)
            + "\nDimSize = 16 14 12\nElementType = MET_FLOAT\nElementDataFile = LOCAL\n"
        )
        path.write_bytes(header.encode() + scalar[0].astype(">f4").tobytes())
        return path, scalar
    if kind == "plane.mha":
        sitk.WriteImage(_block_image(scalar[:, 0], _BLOCK_DIRECTION), str(path))
        return path, scalar[:, 0]
    if kind == "identity.nii":
        # An origin at zero on an identity grid: NIfTI speaks RAS, so the header holds negative
        # zeros, whose sign the record's text keeps or drops exactly as ITK's route does.
        image = _block_image(scalar, np.eye(3).reshape(-1))
        image.SetOrigin([0.0, 0.0, 0.0])
        sitk.WriteImage(image, str(path))
        return path, scalar
    if kind == "scaled.nii":
        import struct

        stored = rng.integers(0, 4000, size=scalar.shape).astype(np.uint16)
        sitk.WriteImage(_block_image(stored, _BLOCK_DIRECTION), str(path))
        header = bytearray(path.read_bytes())
        struct.pack_into("<2f", header, 112, 2.0, 10.0)  # scl_slope, scl_inter: ITK rescales to float
        path.write_bytes(bytes(header))
        return path, stored
    writer = sitk.ImageFileWriter()
    writer.SetFileName(str(path))
    writer.SetUseCompression(kind in ("compressed.mha", "scalar.nii.gz"))
    writer.Execute(_block_image(scalar, _BLOCK_DIRECTION))
    return path, scalar


def _block_backend(path: Path) -> Dataset.SitkFile:
    return Dataset.SitkFile(f"{path.parent}/", True, path.name.split(".", 1)[1])


def _block_region(data: np.ndarray, corner: bool = False) -> tuple[slice, ...]:
    if corner:  # at the volume's own origin, where a zero coordinate keeps or loses its sign
        return (slice(None), *(slice(0, 4) for _ in data.shape[1:]))
    if data.ndim == 4:
        return (slice(None), slice(3, 9), slice(2, 11), slice(5, 13))
    return (slice(None), slice(2, 11), slice(5, 13))


@pytest.mark.parametrize("corner", [False, True], ids=["interior", "corner"])
@pytest.mark.parametrize("kind", _BLOCK_SERVED)
def test_a_region_off_the_raw_block_is_the_one_itk_decodes(
    tmp_path: Path, monkeypatch, kind: str, corner: bool
) -> None:
    """Same bytes, same dtype, same attribute record (keys, order, text) as ITK's streaming reader."""
    from konfai.utils import dataset as dataset_module

    path, data = _write_block_fixture(tmp_path, kind)
    assert dataset_module._pixel_block(str(path)) is not None
    assert Dataset.SitkFile._supports_region_read(str(path))
    region = _block_region(data, corner)
    backend = _block_backend(path)
    got, attributes = backend.file_to_data_slice("", path.name.split(".", 1)[0], region)

    monkeypatch.setattr(dataset_module, "_pixel_block", lambda path: None)
    want, want_attributes = backend.file_to_data_slice("", path.name.split(".", 1)[0], region)

    assert got.dtype == want.dtype and got.dtype.isnative
    np.testing.assert_array_equal(got, want)
    np.testing.assert_array_equal(got, data[region])
    if data.shape[0] > 1 and path.suffix == ".nii":
        # ITK aborts on a region of a vector NIfTI, so its route reads the volume whole and records
        # the volume's origin; the block route records the region's, like every other format.
        index_xyz = np.asarray([item.start for item in reversed(region[1:])], dtype=np.float64)
        direction = want_attributes.get_np_array("Direction").reshape(3, 3)
        expected = want_attributes.get_np_array("Origin") + direction @ (
            index_xyz * want_attributes.get_np_array("Spacing")
        )
        np.testing.assert_array_equal(attributes.get_np_array("Origin"), expected)
        rungs = ("Origin_0", "Origin_1")
        assert {k: v for k, v in attributes.items() if k not in rungs} == {
            k: v for k, v in want_attributes.items() if k not in rungs
        }
    else:
        assert dict(attributes) == dict(want_attributes)


@pytest.mark.parametrize("kind", _BLOCK_LEFT_TO_ITK)
def test_a_file_the_block_route_declines_is_still_read_by_itk(tmp_path: Path, kind: str) -> None:
    """Compressed, detached, rescaled, or another format: the block route steps aside, ITK answers."""
    from konfai.utils import dataset as dataset_module

    path, data = _write_block_fixture(tmp_path, kind)
    assert dataset_module._pixel_block(str(path)) is None
    region = _block_region(data)
    got, attributes = _block_backend(path).file_to_data_slice("", path.name.split(".", 1)[0], region)

    if kind == "scaled.nii":
        np.testing.assert_array_equal(got, data[region].astype(np.float32) * 2.0 + 10.0)
    else:
        np.testing.assert_array_equal(got, data[region])
    assert "Origin" in attributes


def test_a_stepped_region_off_the_raw_block_reads_as_itk_reads_it_whole(tmp_path: Path, monkeypatch) -> None:
    """A step ITK cannot extract is served whole and sliced: the block serves the same values and
    keeps the record ITK's route leaves, the volume's own geometry."""
    from konfai.utils import dataset as dataset_module

    path, data = _write_block_fixture(tmp_path, "vector.mha")
    region = (slice(0, 3, 2), slice(1, 12, 3), slice(0, 14, 2), slice(2, 16, 3))
    backend = _block_backend(path)
    got, attributes = backend.file_to_data_slice("", "vector", region)
    monkeypatch.setattr(dataset_module, "_pixel_block", lambda path: None)
    want, want_attributes = backend.file_to_data_slice("", "vector", region)

    np.testing.assert_array_equal(got, want)
    np.testing.assert_array_equal(got, data[region])
    assert dict(attributes) == dict(want_attributes)


def test_the_raw_block_header_is_read_once_and_follows_a_rewrite(tmp_path: Path, monkeypatch) -> None:
    """ITK reads the header once per file, not once per region; a file rewritten under the same
    name gets a record of its own."""
    from konfai.utils import dataset as dataset_module

    path, data = _write_block_fixture(tmp_path, "scalar.mha")
    reads = {"header": 0}
    real = sitk.ImageFileReader.ReadImageInformation

    def counting(self):
        reads["header"] += 1
        return real(self)

    monkeypatch.setattr(sitk.ImageFileReader, "ReadImageInformation", counting)
    dataset_module._pixel_block_at.cache_clear()
    backend = _block_backend(path)
    for plane in range(10):
        region = (slice(None), slice(plane, plane + 1), slice(None), slice(None))
        got, _ = backend.file_to_data_slice("", "scalar", region)
        np.testing.assert_array_equal(got, data[:, plane : plane + 1])
    assert reads["header"] == 1

    replaced = np.flip(data, axis=1).copy()
    image = _block_image(replaced, _BLOCK_DIRECTION)
    image.SetMetaData("Rewritten", "yes")  # a longer header: the stamp changes even on a coarse clock
    sitk.WriteImage(image, str(path))
    got, attributes = backend.file_to_data_slice("", "scalar", (slice(None), slice(0, 1), slice(None), slice(None)))
    np.testing.assert_array_equal(got, replaced[:, 0:1])
    assert attributes["Rewritten"] == "yes"
    assert reads["header"] == 2


def test_an_h5_sidecar_is_read_once_per_pooled_handle_and_dropped_with_it(tmp_path: Path, monkeypatch) -> None:
    """A patch read costs one hyperslab: the entry's attributes are read off the handle on its first
    read and copied after, and a write of the entry (which drops the handle) brings the new ones."""
    dataset = Dataset(tmp_path / "Sidecar", "h5")
    attributes = Attribute()
    attributes["Origin"] = np.asarray([1.0, 2.0, 3.0])
    attributes["Spacing"] = np.asarray([0.5, 1.5, 2.0])
    attributes["Direction"] = np.eye(3).reshape(-1)
    for index in range(12):
        attributes[f"Key{index}"] = f"value {index}"
    volume = np.arange(4 * 5 * 6, dtype=np.float32).reshape(1, 4, 5, 6)
    dataset.write("CT", "P0", volume, attributes)
    opens = {"attribute": 0}
    real = h5py.AttributeManager.__getitem__

    def counting(self, key):
        opens["attribute"] += 1
        return real(self, key)

    monkeypatch.setattr(h5py.AttributeManager, "__getitem__", counting)
    region = (slice(None), slice(1, 3), slice(0, 5), slice(2, 6))
    records = [dataset.read_data_slice("CT", "P0", region)[1] for _ in range(10)]
    _, whole = dataset.read_data("CT", "P0")

    assert opens["attribute"] == len(attributes)
    assert all(dict(record) == dict(attributes) for record in records)
    assert dict(whole) == dict(attributes)
    records[0]["Origin"] = np.asarray([9.0, 9.0, 9.0])  # a copy: the caller's edits stay the caller's
    assert dataset.read_data_slice("CT", "P0", region)[1]["Origin"] == attributes["Origin"]

    attributes["Study"] = "rewritten"
    dataset.write("CT", "P0", volume + 1, attributes)
    data, record = dataset.read_data_slice("CT", "P0", region)
    np.testing.assert_array_equal(data, (volume + 1)[region])
    assert record["Study"] == "rewritten"


def _attribute_text_through_printing(value) -> str:
    """The normalising door as it stood before the str fast path: every value through the printer."""
    import sys

    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.generic | np.ndarray) and np.issubdtype(value.dtype, np.floating):
        value = np.asarray(value, dtype=np.float64)[()] if isinstance(value, np.generic) else value.astype(np.float64)
    with np.printoptions(threshold=sys.maxsize, floatmode="unique"):
        return str(value).replace("\n", "")


def _former_attribute_copy(attributes: dict) -> Attribute:
    """``Attribute(attributes)`` as it stood: every key deep-copied, every value through the printer."""
    import copy

    copied = Attribute()
    for k, v in attributes.items():
        dict.__setitem__(copied, copy.deepcopy(k), _attribute_text_through_printing(v))
    return copied


def _attribute_fixture_values() -> dict:
    return {
        "Origin": np.asarray([10.0, -20.5, 30.25]),
        "Spacing_0": np.asarray([0.7, 1.3, 2.1], dtype=np.float32),
        "Direction_0": np.eye(3).reshape(-1),
        "Long": np.arange(2000, dtype=np.float64) / 7,
        "Mean": np.float32(0.1),
        "Std": 0.30000000000000004,
        "Count": 12,
        "Flag": True,
        "Nested": [[1.5, 2.0], [3.0, 4.25]],
        "Tensor": torch.tensor([1.0, 2.5, 3.0]),
        "Zero": torch.tensor(0.0),
        "Text": "phantom\nstudy",
        "Empty": "",
        "Ints": np.asarray([1, 2, 3]),
    }


def test_copying_an_attribute_is_the_same_record_as_normalising_it_again() -> None:
    """Every door yields the same text as the printing door did, key for key: from live values,
    from a plain dict of text, and from an Attribute (a dict-level copy)."""
    values = _attribute_fixture_values()
    former = _former_attribute_copy(values)

    from_values = Attribute(values)
    from_text = Attribute(dict(from_values))
    from_attribute = Attribute(from_values)

    assert list(dict(from_values).items()) == list(dict(former).items())
    assert list(dict(from_text).items()) == list(dict(former).items())
    assert list(dict(from_attribute).items()) == list(dict(former).items())
    assert all(type(v) is str for v in dict(from_attribute).values())
    np.testing.assert_array_equal(from_attribute.get_np_array("Long"), values["Long"])
    assert from_attribute["Text"] == "phantomstudy"
    assert Attribute(None) == Attribute({}) == Attribute()


def test_a_copied_attribute_is_independent_of_its_source() -> None:
    source = Attribute(_attribute_fixture_values())
    copied = Attribute(source)
    copied["Origin"] = np.asarray([0.0, 0.0, 0.0])
    copied.pop("Std")

    assert source["Origin"] == Attribute(_attribute_fixture_values())["Origin"]
    assert "Std" in source
    assert source["Std"] == "0.30000000000000004"


def test_assigning_text_keeps_it_as_it_is_but_for_newlines() -> None:
    """The printing door stripped a str's newlines and nothing else; the fast path does the same."""
    attributes = Attribute()
    attributes["Study"] = "phantom\nstudy"
    attributes["Path"] = "a b:c"
    assert attributes["Study"] == "phantomstudy" == _attribute_text_through_printing("phantom\nstudy")
    assert attributes["Path"] == "a b:c"


# --------------------------------------------------------------------------------------
# An array read off a map or an ITK buffer owns its bytes
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["scalar.mha", "scalar.nii", "vector.nii"])
def test_a_full_plane_region_off_the_raw_block_owns_its_bytes(tmp_path: Path, kind: str) -> None:
    """A slab of whole planes is contiguous on the map, so a copy that only guarantees contiguity
    would hand back the map's own pages: read-only, and unmapped once the array holding them is
    gone, under a tensor still pointing at them."""
    path, data = _write_block_fixture(tmp_path, kind)
    region = (slice(None), slice(2, 5), slice(None), slice(None))
    got, _ = _block_backend(path).file_to_data_slice("", path.name.split(".", 1)[0], region)

    assert not isinstance(got, np.memmap) and got.flags.owndata and got.flags.writeable
    tensor = torch.from_numpy(got)  # what the streamed route does with it next
    got += 1
    np.testing.assert_array_equal(tensor.numpy(), data[region] + 1)


def test_image_to_data_owns_the_vector_image_bytes_whatever_its_size() -> None:
    """A vector image of one voxel is contiguous however its axes are moved: the array must still be
    a copy, not a view into a buffer the image takes with it."""
    image = sitk.GetImageFromArray(np.asarray([[[[1.0, 2.0, 3.0]]]], dtype=np.float32), isVector=True)
    data, _ = image_to_data(image)
    del image

    assert data.flags.owndata and data.shape == (3, 1, 1, 1)
    np.testing.assert_array_equal(data.reshape(-1), [1.0, 2.0, 3.0])
