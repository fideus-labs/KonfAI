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
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest
import torch
from konfai.utils.dataset import (
    Attribute,
    Dataset,
    get_infos,
    image_to_data,
    is_staging_entry,
    read_landmarks,
)
from konfai.utils.dataset import raw_block as raw_block_module
from konfai.utils.dataset.h5 import _get_h5_file_lock
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


@pytest.mark.parametrize(
    "values",
    [
        [-100.0, -120.0, 30.0],
        [0.5, 1.25, -3.0],
        [0.8, 0.8, 2.0],
        [-0.0, 0.0],
        [-0.9998476951563913, 0.017452406437283512, 0.0],
        [1e8, 1.0],
        [1e-5, 5.0],
        [0.5, 2000.0],
        [np.nan, 1.0],
        list(np.linspace(-3.0, 3.0, 40)),
    ],
)
def test_attribute_prints_a_float_vector_as_numpy_prints_it(values: list[float]) -> None:
    # The vector fast path answers only where numpy prints positionally on one line; every other
    # vector (exponents, a wrapped line, a NaN) still goes through the printer, and both must agree.
    array = np.asarray(values, dtype=np.float64)
    attribute = Attribute()
    attribute["Origin"] = array

    assert attribute["Origin"] == _attribute_text_through_printing(array)
    np.testing.assert_array_equal(attribute.get_np_array("Origin"), array)


def test_attribute_prints_random_float_vectors_as_numpy_prints_them() -> None:
    rng = np.random.default_rng(7)
    for _ in range(300):
        size = int(rng.integers(1, 13))
        kind = int(rng.integers(0, 4))
        if kind == 0:
            array = rng.integers(-500, 500, size).astype(np.float64)
        elif kind == 1:
            array = np.round(rng.uniform(-300.0, 300.0, size), int(rng.integers(0, 4)))
        elif kind == 2:
            array = rng.uniform(-2.0, 2.0, size)
        else:
            array = rng.choice([0.0, -0.0, 0.5, 1e-4, 999.0, 1000.5, 1e7, 12345678.5, -0.125], size)
        attribute = Attribute()
        attribute["Origin"] = array
        assert attribute["Origin"] == _attribute_text_through_printing(array), array.tolist()


def test_attribute_names_the_key_whose_value_does_not_parse_back_flat() -> None:
    """A >= 2-D value is stored as a nested print (Crop's ``box`` is read back through its own
    parser, so the write door cannot refuse the rank), and reading it back as an array is refused
    with the key and the remedy named, not an anonymous ``ValueError`` deep in numpy."""
    attribute = Attribute()
    attribute["MyMatrix"] = np.eye(3)
    with pytest.raises(DatasetManagerError, match=r"'MyMatrix'.*flat"):
        attribute.get_np_array("MyMatrix")
    with pytest.raises(DatasetManagerError, match="flatten the value"):
        attribute.get_tensor("MyMatrix")
    assert attribute["MyMatrix_0"] == "[[1. 0. 0.] [0. 1. 0.] [0. 0. 1.]]", "the text door still serves it"
    with pytest.raises(DatasetManagerError, match=r"'MyMatrix'"):
        attribute.pop_np_array("MyMatrix")
    attribute["Direction"] = np.eye(3).flatten()  # the flat form parses back exactly
    np.testing.assert_array_equal(attribute.get_np_array("Direction"), np.eye(3).flatten())


# --------------------------------------------------------------------------------------
# HDF5 backend: directories, modes, and per-file locking
# --------------------------------------------------------------------------------------


def test_h5_missing_group_raises_the_designed_refusal_not_attributeerror(tmp_path: Path, image_attributes) -> None:
    """H5File._get_dataset answered None for an absent group and every reader dereferenced it:
    ``AttributeError: 'NoneType' object has no attribute 'shape'`` deep in numpy, where every
    sibling backend names the entry. The backend-level coordinates are the ones the dataset's
    directory branch passes for a directory of case files: ``("", group)``."""
    h5py_module = pytest.importorskip("h5py")
    del h5py_module
    volume = np.arange(1 * 2 * 3 * 4, dtype=np.float32).reshape(1, 2, 3, 4)
    Dataset(tmp_path / "Cases", "h5").write("ct", "CASE_000", volume, image_attributes([0.0] * 3, [1.0] * 3))

    reader = Dataset.H5File(str(tmp_path / "Cases"), True)
    with reader as _:
        for read in (
            lambda: reader.file_to_data("", "missing_group"),
            lambda: reader.get_infos("", "missing_group"),
            lambda: reader.file_to_data_slice("", "missing_group", (slice(None),)),
            lambda: reader.file_to_data("nope", "CASE_000"),
        ):
            with pytest.raises(DatasetManagerError, match="is not in"):
                read()


@pytest.mark.parametrize("file_format", ["h5", "itktransform"])
def test_an_h5_backed_entry_writes_and_reads_back_without_simpleitk(tmp_path: Path, file_format: str) -> None:
    """``konfai[hdf5]`` alone: an array is written by h5py, so no SimpleITK type check may run."""
    script = f"""
import sys
sys.modules["SimpleITK"] = None
import numpy as np
from konfai.utils.dataset import Attribute, Dataset
attributes = Attribute()
attributes["Origin"], attributes["Spacing"], attributes["Direction"] = np.zeros(3), np.ones(3), np.eye(3).ravel()
field = np.arange(3 * 4 * 5 * 6, dtype=np.float32).reshape(3, 4, 5, 6)
dataset = Dataset({str(tmp_path / "out")!r}, {file_format!r})
dataset.write("Field", "CASE_000", field, attributes)
assert np.array_equal(dataset.read_data("Field", "CASE_000")[0], field)
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)


def test_h5_read_chunk_cache_takes_its_slice_of_the_declared_budget() -> None:
    """The HDF5 read pool's rdcc cache was the one decoded-block cache that ignored the declared
    budget: 128 MiB per handle, up to 8 handles, whatever the declaration. Declared, the pool at
    capacity now stays inside the same cache share every other decoded-block cache draws from."""
    pytest.importorskip("h5py")
    from konfai.utils.budget import BUDGET_SHARES, set_per_rank_budget
    from konfai.utils.dataset.h5 import _H5ReadPool

    try:
        set_per_rank_budget(256 << 20)
        expected = int((256 << 20) * BUDGET_SHARES["cache"]) // _H5ReadPool._MAX
        assert Dataset.H5File._read_chunk_cache_bytes() == expected
    finally:
        set_per_rank_budget(None)
    assert Dataset.H5File._read_chunk_cache_bytes() == Dataset.H5File._READ_CHUNK_CACHE_BYTES


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
        with pytest.raises(DatasetManagerError, match="is not in"):
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


@pytest.mark.parametrize("file_format", ["h5", "itk.txt"])
def test_a_composite_of_leaves_with_different_parameter_counts_round_trips(tmp_path: Path, file_format: str) -> None:
    """Euler (6 parameters) then BSpline (375): the leaves' parameter rows differ in length, and
    every backend stores them as one array padded with NaN."""
    euler = sitk.Euler3DTransform()
    euler.SetParameters((0.1, 0.2, 0.3, 1.0, 2.0, 3.0))
    spline = sitk.BSplineTransformInitializer(sitk.Image([8, 8, 8], sitk.sitkFloat32), [2, 2, 2])
    spline.SetParameters(tuple(np.linspace(-1.0, 1.0, spline.GetNumberOfParameters())))
    composite = sitk.CompositeTransform([euler, spline])
    dataset = Dataset(tmp_path / "Transforms", file_format)

    dataset.write("T", "CASE_000", composite, Attribute())
    restored = dataset.read_transform("T", "CASE_000")

    for point in [(1.0, 2.0, 3.0), (4.5, 0.5, 6.0), (7.0, 7.0, 0.0)]:
        np.testing.assert_allclose(restored.TransformPoint(point), composite.TransformPoint(point))


# --------------------------------------------------------------------------------------
# Landmarks: a fiducial file is read in LPS whatever coordinate system its header declares
# --------------------------------------------------------------------------------------


def _fiducial_file(path: Path, coordinate_system: str | None) -> Path:
    """One point (1, 2, 3) in a Slicer fiducial file, with or without the CoordinateSystem line."""
    header = ["# Markups fiducial file version = 4.10"]
    if coordinate_system is not None:
        header.append(f"# CoordinateSystem = {coordinate_system}")
    header.append("# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID")
    path.write_text("\n".join([*header, "vtkMRMLMarkupsFiducialNode_0,1,2,3,0,0,0,1,1,1,0,F-1,,"]) + "\n")
    return path


@pytest.mark.parametrize(
    ("coordinate_system", "expected"),
    [
        ("RAS", [-1.0, -2.0, 3.0]),
        ("0", [1.0, 2.0, 3.0]),
        ("1", [1.0, 2.0, 3.0]),
        ("LPS", [1.0, 2.0, 3.0]),
        (None, [1.0, 2.0, 3.0]),
    ],
)
def test_read_landmarks_returns_lps_points(
    tmp_path: Path, coordinate_system: str | None, expected: list[float]
) -> None:
    """KonfAI's physical space is LPS: an RAS point has its x and y negated. '0' is ambiguous (RAS
    in 3D Slicer before 4.11, LPS in KonfAI up to 1.5.3) and stays LPS."""
    points = read_landmarks(_fiducial_file(tmp_path / "points.fcsv", coordinate_system))

    np.testing.assert_array_equal(points, [expected])


@pytest.mark.parametrize("coordinate_system", ["2", "IJK"])
def test_read_landmarks_refuses_a_coordinate_system_other_than_ras_or_lps(
    tmp_path: Path, coordinate_system: str
) -> None:
    with pytest.raises(DatasetManagerError, match="CoordinateSystem"):
        read_landmarks(_fiducial_file(tmp_path / "points.fcsv", coordinate_system))


_POINTS = np.array([[1.5, -2.25, 3.0], [4.0, 5.0, -6.125]])


def test_landmarks_round_trip_through_a_dataset(tmp_path: Path) -> None:
    """An (N, 3) array is written as a Slicer fiducial file and read back as the same LPS points."""
    dataset = Dataset(tmp_path / "Dataset", "mha")

    dataset.write("Points", "CASE_000", _POINTS, Attribute())

    assert (tmp_path / "Dataset" / "CASE_000" / "Points.fcsv").is_file()
    data, _ = dataset.read_data("Points", "CASE_000")
    np.testing.assert_array_equal(data, _POINTS)


def test_a_polydata_round_trips_through_a_dataset(tmp_path: Path) -> None:
    vtk = pytest.importorskip("vtk")
    points = vtk.vtkPoints()
    for point in _POINTS:
        points.InsertNextPoint(*point)
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    dataset = Dataset(tmp_path / "Dataset", "mha")

    dataset.write("Mesh", "CASE_000", polydata, Attribute())

    assert (tmp_path / "Dataset" / "CASE_000" / "Mesh.vtk").is_file()
    data, _ = dataset.read_data("Mesh", "CASE_000")
    np.testing.assert_array_equal(data, _POINTS)


def test_a_vtk_entry_without_vtk_names_the_extra(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    case = tmp_path / "Dataset" / "CASE_000"
    case.mkdir(parents=True)
    (case / "Mesh.vtk").write_text("# vtk DataFile Version 5.1\n")
    monkeypatch.setitem(sys.modules, "vtk", None)  # an import of vtk now fails as on a bare install

    with pytest.raises(DatasetManagerError, match=r"pip install konfai\[vtk\]"):
        Dataset(tmp_path / "Dataset", "mha").read_data("Mesh", "CASE_000")


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
    with pytest.raises(DatasetManagerError, match="is not in"):
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


class _Listing:
    """What ``os.scandir`` hands back, over entries in a chosen order."""

    def __init__(self, entries: list[os.DirEntry]) -> None:
        self._entries = iter(entries)

    def __iter__(self) -> "_Listing":
        return self

    def __next__(self) -> os.DirEntry:
        return next(self._entries)

    def __enter__(self) -> "_Listing":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def close(self) -> None:
        return None


def _list_by_name(monkeypatch: pytest.MonkeyPatch, reverse: bool) -> None:
    """Every directory listing in name order, or in reverse: a filesystem may list in any order."""
    scandir = os.scandir

    def listed(path: str = ".") -> _Listing:
        with scandir(path) as entries:
            return _Listing(sorted(entries, key=lambda entry: entry.name, reverse=reverse))

    monkeypatch.setattr(os, "scandir", listed)


def _store_cases(root: Path, names: list[str]) -> None:
    for name in names:
        (root / name / "CT.ome.zarr").mkdir(parents=True)


def _file_cases(root: Path, names: list[str]) -> None:
    for name in names:
        (root / name).mkdir(parents=True)
        (root / name / "CT.mha").write_bytes(b"")


def _put(path: Path, data: bytes = b"") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _cases(prefix: str, count: int = 20) -> list[str]:
    return [f"{prefix}_{index:03d}" for index in range(count)]


def _atlas_beside_stores(root: Path) -> None:
    _store_cases(root, _cases("CASE"))
    _put(root / "Atlas" / "atlas.nii.gz")


def _qc_beside_stores(root: Path) -> None:
    _store_cases(root, _cases("patient"))
    _put(root / "QC" / "overview.png")


def _trash_beside_files(root: Path) -> None:
    _file_cases(root, _cases("CASE"))
    _put(root / ".Trash-1000" / "files" / "deleted.dcm")
    (root / ".Trash-1000" / "info").mkdir()


def _trash_beside_stores(root: Path) -> None:
    _store_cases(root, _cases("CASE"))
    _put(root / ".Trash-1000" / "files" / "deleted.dcm")


def _git_beside_stores(root: Path) -> None:
    _store_cases(root, _cases("CASE"))
    _put(root / ".git" / "HEAD", b"ref: refs/heads/main\n")
    _put(root / ".git" / "hooks" / "pre-commit.sample", b"#!/bin/sh\n")
    (root / ".git" / "objects" / "ab").mkdir(parents=True)


def _checkpoints_beside_files(root: Path) -> None:
    _file_cases(root, _cases("CASE"))
    _put(root / ".ipynb_checkpoints" / "labels-checkpoint.xml")


def _one_store(root: Path) -> None:
    _store_cases(root, ["CASE_000"])


def _unreadable_in_every_store_case(root: Path) -> None:
    _store_cases(root, _cases("CASE", 3))
    for case in _cases("CASE", 3):
        (root / case / ".private").mkdir(mode=0)


@pytest.mark.parametrize(
    ("layout", "token", "backend", "cases"),
    [
        (_atlas_beside_stores, "mha", "omezarr", 20),
        (_qc_beside_stores, "mha", "omezarr", 20),
        (_trash_beside_files, "mha", "mha", 20),
        (_trash_beside_stores, "omezarr", "omezarr", 20),
        (_git_beside_stores, "mha", "omezarr", 20),
        (_checkpoints_beside_files, "mha", "mha", 20),
        (_one_store, "mha", "omezarr", 1),
        (_unreadable_in_every_store_case, "mha", "omezarr", 3),
        (_unreadable_in_every_store_case, "omezarr", "omezarr", 3),
    ],
)
def test_the_cases_decide_the_store_form_whatever_sits_beside_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, layout, token: str, backend: str, cases: int
) -> None:
    """A directory beside the cases (an atlas, a QC folder, a trash, a repository, a notebook's
    checkpoints) or one a case holds and nobody may read never decides a root's store form, whatever
    the filesystem's listing order: the form the cases hold does, and every case is listed."""
    root = tmp_path / "ds"
    layout(root)
    found = []
    try:
        for reverse in (False, True):
            _list_by_name(monkeypatch, reverse)
            dataset = Dataset(f"{root}/", token)
            found.append((dataset.file_format, len(dataset.get_names("CT"))))
    finally:
        for private in root.glob("*/.private"):
            private.chmod(0o755)
    assert found == [(backend, cases)] * 2


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root reads every directory")
@pytest.mark.skipif(os.name == "nt", reason="chmod does not take away reading a directory on Windows")
def test_a_root_none_of_whose_cases_can_be_read_is_refused_not_read_as_empty(tmp_path: Path) -> None:
    """With no case readable, nothing was seen to decide the store form: the permission error is raised,
    not a guess that would list no case."""
    root = tmp_path / "ds"
    _store_cases(root, _cases("CASE", 3))
    for case in root.iterdir():
        case.chmod(0)
    try:
        with pytest.raises(PermissionError):
            Dataset(f"{root}/", "mha")
    finally:
        for case in root.iterdir():
            case.chmod(0o755)


def test_a_large_root_is_probed_on_a_bounded_sample(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """At most 16 of a root's directories vote, and at most 16 sub-directories of each that hold something
    are opened: a cohort of thousands, flat or grouped in folders, costs a few listings to probe."""
    flat, nested = tmp_path / "flat", tmp_path / "nested"
    _file_cases(flat, _cases("CASE", 40))
    for fold in ("fold_0", "fold_1"):
        _file_cases(nested / fold, _cases("CASE", 40))
    scandir = os.scandir
    listed: list[str] = []
    monkeypatch.setattr(os, "scandir", lambda path=".": (listed.append(str(path)), scandir(path))[1])

    assert Dataset._detect_directory_store_format(f"{flat}/") is None
    assert len(listed) <= 1 + 16
    listed.clear()
    assert Dataset._detect_directory_store_format(f"{nested}/") is None
    assert len(listed) <= 1 + 2 * (1 + 16)


def test_a_series_beside_its_macos_twins_is_found(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An extensionless series extracted from a macOS archive holds a ``._`` twin per slice: the twins sort
    first, yet the slices are the files read for the DICOM magic."""
    root = tmp_path / "ds"
    for case in _cases("CASE", 4):
        for index in range(20):
            _put(root / case / "CT" / f"IM{index:02d}", b"\0" * 128 + b"DICM")
            _put(root / case / "CT" / f"._IM{index:02d}", b"\0" * 4096)

    detected = []
    for reverse in (False, True):
        _list_by_name(monkeypatch, reverse)
        detected.append(Dataset._detect_directory_store_format(f"{root}/"))
    assert detected == ["dicom", "dicom"]


class _Entry:
    """A listing entry that records every link it is asked to follow."""

    def __init__(self, entry: os.DirEntry, followed: list[str]) -> None:
        self._entry, self._followed = entry, followed
        self.name, self.path = entry.name, entry.path

    def _follows(self, follow_symlinks: bool) -> None:
        if follow_symlinks and self._entry.is_symlink():
            self._followed.append(self.name)

    def is_dir(self, *, follow_symlinks: bool = True) -> bool:
        self._follows(follow_symlinks)
        return self._entry.is_dir(follow_symlinks=follow_symlinks)

    def is_file(self, *, follow_symlinks: bool = True) -> bool:
        self._follows(follow_symlinks)
        return self._entry.is_file(follow_symlinks=follow_symlinks)

    def is_symlink(self) -> bool:
        return self._entry.is_symlink()


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
    from konfai.utils.dataset.statistics import _STATISTICS_CHUNK_ELEMENTS, _statistics_chunk_length

    for channels in (1, 4, 122):
        shape = [channels, 400, 512, 512]
        length = _statistics_chunk_length(shape, axis=1, budget=_STATISTICS_CHUNK_ELEMENTS)
        held = channels * length * 512 * 512
        # One step is the floor, so a volume whose step alone overflows is read a step at a time.
        assert held <= max(_STATISTICS_CHUNK_ELEMENTS, channels * 512 * 512)


def test_a_statistics_chunk_reaches_further_on_a_thin_volume() -> None:
    from konfai.utils.dataset.statistics import _statistics_chunk_length

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


@pytest.mark.parametrize("file_format", ["mha", "nii.gz", "mhd", "hdr", "img"])
def test_a_group_whose_name_holds_a_dot_is_listed_whole(tmp_path: Path, image_attributes, file_format: str) -> None:
    """A dot in the stem belongs to the name (``CT.contrast.nii.gz`` is the group ``CT.contrast``):
    the listing names the groups every lookup answers for (Resample and the Evaluator's maps pick
    their group from it)."""
    dataset = Dataset(tmp_path / "ds", file_format)
    dataset.write("CT.contrast", "case1", np.zeros((1, 2, 2, 2), np.float32), image_attributes([0, 0, 0], [1, 1, 1]))
    dataset.write("MASK", "case1", np.zeros((1, 2, 2, 2), np.uint8), image_attributes([0, 0, 0], [1, 1, 1]))

    assert sorted(dataset.get_group()) == ["CT.contrast", "MASK"]
    assert all(dataset.is_dataset_exist(group, "case1") for group in dataset.get_group())


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
    """A single store answers the same way a directory does: the pool closes a handle whose store
    changed underneath it. Opening a second handle beside it is not enough: HDF5 shares a file's metadata
    state across the handles one process holds, so a second handle inherits the first's view."""
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
    from konfai.utils.dataset.h5 import _h5_read_pool

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


@pytest.mark.parametrize("file_format", ["itk.txt", "fcsv", "xml", "npy", "png", "jpg", "bmp", "dcm", "nrrd.gz"])
def test_a_volume_the_format_cannot_hold_is_refused_by_name(tmp_path: Path, image_attributes, file_format: str) -> None:
    """A format SimpleITK has no writer for, or whose writer refuses this volume, answered with ITK's
    trace and the name of a staging file the user never asked for."""
    pytest.importorskip("SimpleITK")
    dataset = Dataset(tmp_path / "store", file_format)
    volume = np.ones((1, 4, 5, 6), np.float32)
    with pytest.raises(DatasetManagerError, match=f"as '{file_format}'") as refusal:
        dataset.write("CT", "CASE_001", volume, image_attributes([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]))
    assert ".tmp" not in str(refusal.value)


def test_a_dicom_write_that_fails_keeps_itks_error(tmp_path: Path, image_attributes, monkeypatch) -> None:
    """GDCM reports every failed write, a full disk included, as a component type it does not support:
    that phrase says nothing about the format, so it is not read as a refusal."""
    sitk = pytest.importorskip("SimpleITK")

    def full_disk(image, path, *args, **kwargs):
        raise RuntimeError(
            "itkGDCMImageIO.cxx:1400:\nITK ERROR: GDCMImageIO(0x1): DICOM does not support this component type"
        )

    monkeypatch.setattr(sitk, "WriteImage", full_disk)
    dataset = Dataset(tmp_path / "store", "dcm")
    with pytest.raises(RuntimeError, match="component type") as error:
        dataset.write("CT", "CASE_001", np.ones((1, 1, 5, 6), np.uint8), image_attributes([0.0] * 3, [1.0] * 3))
    assert not isinstance(error.value, DatasetManagerError)


@pytest.mark.skipif(os.name == "nt" or os.geteuid() == 0, reason="POSIX directory permissions, not as root")
@pytest.mark.parametrize("file_format", ["mha", "nii.gz"])
def test_a_write_that_fails_on_the_disk_keeps_itks_error(tmp_path: Path, image_attributes, file_format: str) -> None:
    """A format that holds the volume but a directory that refuses it: ITK's error, not a format refusal."""
    pytest.importorskip("SimpleITK")
    case = tmp_path / "store" / "CASE_001"
    case.mkdir(parents=True)
    case.chmod(0o555)
    try:
        with pytest.raises(RuntimeError):
            Dataset(tmp_path / "store", file_format).write(
                "CT", "CASE_001", np.ones((1, 4, 5, 6), np.float32), image_attributes([0.0] * 3, [1.0] * 3)
            )
    finally:
        case.chmod(0o755)


@pytest.mark.parametrize("file_format", ["mha", "nii.gz", "nrrd"])
def test_a_boolean_volume_is_refused_by_name_where_simpleitk_writes_it(
    tmp_path: Path, image_attributes, file_format: str
) -> None:
    """SimpleITK has no boolean pixel type: its TypeError named no entry and no way out."""
    pytest.importorskip("SimpleITK")
    dataset = Dataset(tmp_path / "store", file_format)
    with pytest.raises(DatasetManagerError, match="bool"):
        dataset.write("MASK", "CASE_001", np.ones((1, 4, 5, 6), bool), image_attributes([0.0] * 3, [1.0] * 3))


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
    from konfai.utils.dataset.stream import _MhaDataStream, _NiftiDataStream

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

    path, data = _write_block_fixture(tmp_path, kind)
    assert raw_block_module._pixel_block(str(path)) is not None
    assert Dataset.SitkFile._supports_region_read(str(path))
    region = _block_region(data, corner)
    backend = _block_backend(path)
    got, attributes = backend.file_to_data_slice("", path.name.split(".", 1)[0], region)

    monkeypatch.setattr("konfai.utils.dataset.sitk_file._pixel_block", lambda path: None)
    want, want_attributes = backend.file_to_data_slice("", path.name.split(".", 1)[0], region)

    assert got.dtype == want.dtype and got.dtype.isnative
    np.testing.assert_array_equal(got, want)
    np.testing.assert_array_equal(got, data[region])
    if data.shape[0] > 1 and path.suffix == ".nii":
        # ITK aborts on a region of a vector NIfTI, so its route reads the volume whole; both
        # routes still record the REGION's origin (the shared region-geometry update), and only
        # the first rung of the Origin stack differs: ITK's extract origin on the block route,
        # the volume's on the whole-read one.
        index_xyz = np.asarray([item.start for item in reversed(region[1:])], dtype=np.float64)
        direction = want_attributes.get_np_array("Direction").reshape(3, 3)
        volume_origin = Attribute._parse_array(dict(want_attributes)["Origin_0"])
        expected = volume_origin + direction @ (index_xyz * want_attributes.get_np_array("Spacing"))
        np.testing.assert_array_equal(attributes.get_np_array("Origin"), expected)
        np.testing.assert_array_equal(want_attributes.get_np_array("Origin"), expected)
        rungs = ("Origin_0", "Origin_1")
        assert {k: v for k, v in attributes.items() if k not in rungs} == {
            k: v for k, v in want_attributes.items() if k not in rungs
        }
    else:
        assert dict(attributes) == dict(want_attributes)


def test_every_patch_of_a_grid_records_what_itk_records(tmp_path: Path, monkeypatch) -> None:
    """The geometry a record carries is printed once for the file; each patch keeps its own origin.

    Character for character against ITK's own reader, on every patch of a grid: a record built from
    the file's printed geometry says the same thing as one printed per patch."""

    path, _ = _write_block_fixture(tmp_path, "rotated.mha")
    backend = _block_backend(path)
    windows = [
        (slice(None), slice(z, z + 4), slice(y, y + 4), slice(x, x + 4)) for z in (0, 5) for y in (0, 6) for x in (0, 7)
    ]
    got = [backend.file_to_data_slice("", "rotated", window)[1] for window in windows]
    monkeypatch.setattr("konfai.utils.dataset.sitk_file._pixel_block", lambda path: None)
    want = [backend.file_to_data_slice("", "rotated", window)[1] for window in windows]

    assert [dict(record) for record in got] == [dict(record) for record in want]
    assert len({record["Origin"] for record in got}) == len(windows)


@pytest.mark.parametrize("kind", _BLOCK_LEFT_TO_ITK)
def test_a_file_the_block_route_declines_is_still_read_by_itk(tmp_path: Path, kind: str) -> None:
    """Compressed, detached, rescaled, or another format: the block route steps aside, ITK answers."""

    path, data = _write_block_fixture(tmp_path, kind)
    assert raw_block_module._pixel_block(str(path)) is None
    region = _block_region(data)
    got, attributes = _block_backend(path).file_to_data_slice("", path.name.split(".", 1)[0], region)

    if kind == "scaled.nii":
        np.testing.assert_array_equal(got, data[region].astype(np.float32) * 2.0 + 10.0)
    else:
        np.testing.assert_array_equal(got, data[region])
    assert "Origin" in attributes


def test_a_stepped_region_off_the_raw_block_reads_as_itk_reads_it_whole(tmp_path: Path, monkeypatch) -> None:
    """A step ITK cannot extract is served whole and sliced: the block serves the same values and
    the same record as ITK's route — the volume's own geometry, then the region's shifted origin
    and step-scaled spacing, the record every backend returns for the samples actually kept."""

    path, data = _write_block_fixture(tmp_path, "vector.mha")
    region = (slice(0, 3, 2), slice(1, 12, 3), slice(0, 14, 2), slice(2, 16, 3))
    backend = _block_backend(path)
    got, attributes = backend.file_to_data_slice("", "vector", region)
    monkeypatch.setattr("konfai.utils.dataset.sitk_file._pixel_block", lambda path: None)
    want, want_attributes = backend.file_to_data_slice("", "vector", region)

    np.testing.assert_array_equal(got, want)
    np.testing.assert_array_equal(got, data[region])
    assert dict(attributes) == dict(want_attributes)
    volume_origin = Attribute._parse_array(dict(attributes)["Origin_0"])
    volume_spacing = Attribute._parse_array(dict(attributes)["Spacing_0"])
    direction = attributes.get_np_array("Direction").reshape(3, 3)
    start_xyz = np.asarray([2.0, 0.0, 1.0])
    np.testing.assert_array_equal(
        attributes.get_np_array("Origin"), volume_origin + direction @ (start_xyz * volume_spacing)
    )
    np.testing.assert_array_equal(attributes.get_np_array("Spacing"), volume_spacing * [3.0, 2.0, 3.0])


def test_a_stepped_region_carries_the_same_geometry_record_whatever_the_backend(tmp_path: Path) -> None:
    """The same stepped read of the same logical volume answers one geometry record whatever the
    file format it was stored in: the region's, the first kept sample's world position and the
    step-scaled spacing, not the volume's origin and un-scaled spacing."""
    pytest.importorskip("zarr")
    pytest.importorskip("pydicom")
    volume = np.arange(1 * 6 * 8 * 10, dtype=np.int16).reshape(1, 6, 8, 10)
    origin = np.asarray([10.0, 20.0, 30.0])
    spacing = np.asarray([0.5, 1.5, 2.0])
    direction = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    region = (slice(None), slice(1, 6, 2), slice(0, 8, 3), slice(2, 10, 2))
    start_xyz, step_xyz = np.asarray([2.0, 0.0, 1.0]), np.asarray([2.0, 3.0, 2.0])

    for file_format in ("mha", "omezarr", "dicom"):
        attributes = Attribute()
        attributes["Origin"] = origin
        attributes["Spacing"] = spacing
        attributes["Direction"] = direction.flatten()
        dataset = Dataset(tmp_path / file_format, file_format)
        dataset.write("CT", "CASE_000", volume, attributes)
        data, record = dataset.read_data_slice("CT", "CASE_000", region)

        np.testing.assert_array_equal(data, volume[region], err_msg=file_format)
        np.testing.assert_allclose(
            record.get_np_array("Origin"), origin + direction @ (start_xyz * spacing), err_msg=file_format
        )
        np.testing.assert_allclose(record.get_np_array("Spacing"), spacing * step_xyz, err_msg=file_format)


def test_the_raw_block_header_is_read_once_and_follows_a_rewrite(tmp_path: Path, monkeypatch) -> None:
    """ITK reads the header once per file, not once per region; a file rewritten under the same
    name gets a record of its own."""

    path, data = _write_block_fixture(tmp_path, "scalar.mha")
    reads = {"header": 0}
    real = sitk.ImageFileReader.ReadImageInformation

    def counting(self):
        reads["header"] += 1
        return real(self)

    monkeypatch.setattr(sitk.ImageFileReader, "ReadImageInformation", counting)
    raw_block_module._pixel_block_at.cache_clear()
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


@pytest.mark.parametrize("file_format", ["mhd", "hdr", "img"])
def test_a_detached_format_publishes_both_its_files_under_the_entry_name(tmp_path: Path, file_format: str) -> None:
    """MetaImage .mhd and Analyze .hdr/.img keep the header and the pixels in two files, the header
    naming the pixels. Both land under the entry's own name: no part of the case looks like a
    writer's staging, and a copy of its visible files reads back."""
    volume = np.arange(2 * 3 * 4, dtype=np.float32).reshape(1, 2, 3, 4)
    attributes = Attribute()
    attributes["Origin"] = np.asarray([1.0, 2.0, 3.0])
    attributes["Spacing"] = np.asarray([0.5, 1.5, 2.0])
    attributes["Direction"] = np.eye(3).flatten()
    Dataset(tmp_path / "written", file_format).write("CT", "case1", volume, attributes)

    case = tmp_path / "written" / "case1"
    names = sorted(path.name for path in case.iterdir())
    assert not any(name.startswith(".") or is_staging_entry(name) for name in names), names
    copy = tmp_path / "copied" / "case1"
    copy.mkdir(parents=True)
    for name in names:
        (copy / name).write_bytes((case / name).read_bytes())
    data, _ = Dataset(tmp_path / "copied", file_format).read_data("CT", "case1")
    np.testing.assert_array_equal(data, volume)


def test_a_writer_killed_while_staging_a_detached_format_leaves_no_group(tmp_path: Path) -> None:
    """A .mhd is staged as a hidden directory holding both files under their final names. A writer
    killed before moving them in leaves that directory behind: the listing must not descend into it,
    and the entry it was replacing still reads. Run in a child, since the failure is a hard kill."""
    script = f"""
import os
import shutil
import numpy as np
import SimpleITK as sitk
from konfai.utils.dataset import Attribute, Dataset
attributes = Attribute()
attributes["Origin"] = np.zeros(3)
attributes["Spacing"] = np.ones(3)
attributes["Direction"] = np.eye(3).flatten()
dataset = Dataset({str(tmp_path)!r}, "mhd")
dataset.write("MASK", "c1", np.zeros((1, 2, 3, 4), np.float32), attributes)
dataset.write("CT", "c2", np.zeros((1, 2, 3, 4), np.float32), attributes)
write = sitk.WriteImage
def killed(*args, **kwargs):
    write(*args, **kwargs)
    os._exit(9)
sitk.WriteImage = killed
dataset.write("CT", "c2", np.ones((1, 2, 3, 4), np.float32), attributes)
"""
    run = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert run.returncode == 9, run.stderr[-2000:]
    assert any(path.is_dir() and is_staging_entry(path.name) for path in (tmp_path / "c2").iterdir())

    dataset = Dataset(tmp_path, "mhd")
    assert sorted(dataset.get_group()) == ["CT", "MASK"]
    data, _ = dataset.read_data("CT", "c2")
    np.testing.assert_array_equal(data, np.zeros((1, 2, 3, 4), np.float32))


def test_a_metaimage_write_interrupted_before_its_header_leaves_the_old_pair(tmp_path: Path, monkeypatch) -> None:
    """A .mhd header names its pixels: the new pixels land beside the old ones and the header swaps to them in
    one replace, so a write that fails before that replace leaves the old pair whole, and one that succeeds
    leaves no old pixels behind."""
    attributes = Attribute()
    attributes["Origin"] = np.zeros(3)
    attributes["Direction"] = np.eye(3).flatten()
    dataset = Dataset(tmp_path, "mhd")
    attributes["Spacing"] = np.ones(3)
    dataset.write("CT", "c1", np.ones((1, 4, 4, 4), np.float32), attributes)
    replace = os.replace

    def failing(source: str, target: str) -> None:
        if str(target).endswith(".mhd"):
            raise OSError("interrupted")
        replace(source, target)

    monkeypatch.setattr(os, "replace", failing)
    attributes["Spacing"] = np.full(3, 2.0)
    with pytest.raises(OSError, match="interrupted"):
        dataset.write("CT", "c1", np.full((1, 4, 4, 4), 9, np.float32), attributes)
    data, read = Dataset(tmp_path, "mhd").read_data("CT", "c1")
    assert (data == 1).all() and list(read.get_np_array("Spacing")) == [1.0, 1.0, 1.0]

    monkeypatch.setattr(os, "replace", replace)
    dataset.write("CT", "c1", np.full((1, 4, 4, 4), 9, np.float32), attributes)
    data, _ = Dataset(tmp_path, "mhd").read_data("CT", "c1")
    assert (data == 9).all() and len(list((tmp_path / "c1").glob("CT*.raw"))) == 1

    # A header naming pixels that are not the entry's own (shared, or elsewhere): a rewrite leaves them.
    header = next((tmp_path / "c1").glob("CT.mhd"))
    shared = tmp_path / "c1" / "shared.raw"
    shared.write_bytes(next((tmp_path / "c1").glob("CT.*.raw")).read_bytes())
    header.write_text(header.read_text().replace(next((tmp_path / "c1").glob("CT.*.raw")).name, "shared.raw"))
    dataset.write("CT", "c1", np.ones((1, 4, 4, 4), np.float32), attributes)
    assert shared.exists()


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
    # The sidecar, with the region's origin on top: (1, 2, 3) + (2, 0, 1) * (0.5, 1.5, 2.0), in (x, y, z).
    assert all({key: record[key] for key in attributes} == dict(attributes) for record in records)
    assert all(record.get_np_array("Origin").tolist() == [2.0, 2.0, 5.0] for record in records)
    assert dict(whole) == dict(attributes)
    records[0]["Origin"] = np.asarray([9.0, 9.0, 9.0])  # a copy: the caller's edits stay the caller's
    assert dataset.read_data_slice("CT", "P0", region)[1]["Origin"] == records[1]["Origin"]

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
    again = Attribute(values)  # the small arrays' printed forms now come from the cache

    assert list(dict(from_values).items()) == list(dict(former).items())
    assert list(dict(again).items()) == list(dict(former).items())
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


#: A three-plane slab of the twelve-plane fixture, as a share of the block the read must map. A
#: NIfTI vector volume stores each channel whole, one after the other, so a slab of it reaches from
#: the first channel's first plane to the last channel's last: the span IS the block.
_SLAB_SHARE_OF_THE_BLOCK = {"scalar.mha": 0.25, "scalar.nii": 0.25, "vector.mha": 0.25, "vector.nii": 1.0}


@pytest.mark.parametrize("kind", sorted(_SLAB_SHARE_OF_THE_BLOCK))
def test_a_region_read_maps_the_smallest_span_that_holds_it(tmp_path: Path, kind: str, monkeypatch) -> None:
    """The address space a run needs follows the regions its budget sized, not the source's size."""
    path, data = _write_block_fixture(tmp_path, kind)
    mapped: list[int] = []
    original = np.memmap

    def spy(*args, **kwargs):
        block = original(*args, **kwargs)
        mapped.append(block.nbytes)
        return block

    monkeypatch.setattr(np, "memmap", spy)
    region = (slice(None), slice(2, 5), slice(None), slice(None))
    got, _ = _block_backend(path).file_to_data_slice("", path.name.split(".", 1)[0], region)

    np.testing.assert_array_equal(got, data[region])
    assert mapped == [int(data.nbytes * _SLAB_SHARE_OF_THE_BLOCK[kind])]


def test_image_to_data_owns_the_vector_image_bytes_whatever_its_size() -> None:
    """A vector image of one voxel is contiguous however its axes are moved: the array must still be
    a copy, not a view into a buffer the image takes with it."""
    image = sitk.GetImageFromArray(np.asarray([[[[1.0, 2.0, 3.0]]]], dtype=np.float32), isVector=True)
    data, _ = image_to_data(image)
    del image

    assert data.flags.owndata and data.shape == (3, 1, 1, 1)
    np.testing.assert_array_equal(data.reshape(-1), [1.0, 2.0, 3.0])


@pytest.mark.parametrize("file_format", ["nii", "nii.gz", "mha"])
def test_a_write_the_disk_cut_short_is_refused_and_leaves_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, file_format: str
) -> None:
    """ITK's NIfTI writer does not check its writes: on a full disk it leaves a short file and no
    error. MetaImage raises. Either way nothing is published and no staging file is left behind."""
    sitk = pytest.importorskip("SimpleITK")
    write_image = sitk.WriteImage

    def full_disk(image, path, *args, **kwargs):
        write_image(image, path, *args, **kwargs)
        with open(path, "r+b") as file:
            file.truncate(os.path.getsize(path) // 2)
        if file_format == "mha":
            raise RuntimeError("ITK ERROR: MetaImageIO: File cannot be written")

    monkeypatch.setattr(sitk, "WriteImage", full_disk)
    image = sitk.GetImageFromArray(np.random.default_rng(0).random((8, 16, 16)).astype(np.float32))
    with pytest.raises((DatasetManagerError, RuntimeError), match=r"stopped short|cannot be written"):
        Dataset(tmp_path / "Dataset", file_format).write("CT", "case", image)
    assert [name for _, _, names in os.walk(tmp_path) for name in names] == []


@pytest.mark.parametrize("file_format", ["mha", "nrrd", "omezarr"])
def test_the_geometry_stack_reads_back_as_it_was_written(tmp_path: Path, file_format: str) -> None:
    """A header carries the geometry stack it was written with: read back, the stack is the one
    written, where it grew by one Origin, Spacing and Direction a cycle."""
    if file_format == "omezarr":
        pytest.importorskip("ngff_zarr")
    attributes = Attribute()
    attributes["Origin"] = np.asarray([1.0, 2.0, 3.0])
    attributes["Spacing"] = np.asarray([1.0, 1.0, 2.0])
    attributes["Direction"] = np.eye(3).flatten()
    data = np.zeros((1, 4, 5, 6), np.float32)
    for cycle in range(4):
        Dataset(tmp_path / str(cycle), file_format).write("CT", "case", data, attributes)
        data, attributes = Dataset(tmp_path / str(cycle), file_format).read_data("CT", "case")
    keys = [key for key in dict.keys(attributes) if key.split("_")[0] in ("Origin", "Spacing", "Direction")]
    assert sorted(keys) == ["Direction_0", "Origin_0", "Spacing_0"]
    assert attributes.get_np_array("Origin").tolist() == [1.0, 2.0, 3.0]
