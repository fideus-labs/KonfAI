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

"""End-to-end imaging round-trip tests for the DICOM and OME-Zarr backends.

These tests exercise the real optional backends (pydicom / zarr / ngff-zarr /
SimpleITK) and skip gracefully when they are not installed. They complement the
mostly-mocked ``test_imaging_formats.py`` with full read/write round-trips,
cross-validation against SimpleITK's own DICOM reader, and ngff-zarr interop.
"""

from pathlib import Path

import numpy as np
import pytest
from konfai.utils.errors import DatasetManagerError

# ---------------------------------------------------------------------------
# DICOM round-trips
# ---------------------------------------------------------------------------


def test_dicom_roundtrip_geometry_matches_simpleitk(tmp_path: Path) -> None:
    """KonfAI's DICOM reader must agree with SimpleITK's GDCM reader on geometry."""
    pytest.importorskip("pydicom")
    sitk = pytest.importorskip("SimpleITK")
    from konfai.utils import dicom

    root = tmp_path / "series"
    vol = (np.arange(1 * 5 * 6 * 7).reshape(1, 5, 6, 7) % 100).astype(np.float32)
    origin, spacing = (3.0, 4.0, 5.0), (0.8, 0.9, 2.0)
    dicom.write_dicom_series(root, vol, origin=origin, spacing=spacing, direction=np.eye(3).flatten())

    kvol, kog, ksp, kdir = dicom.read_dicom_series(root)

    reader = sitk.ImageSeriesReader()
    ids = reader.GetGDCMSeriesIDs(str(root))
    reader.SetFileNames(reader.GetGDCMSeriesFileNames(str(root), ids[0]))
    itk = reader.Execute()

    assert np.allclose(kog, itk.GetOrigin(), atol=1e-3)
    assert np.allclose(ksp, itk.GetSpacing(), atol=1e-3)
    assert np.allclose(kdir, itk.GetDirection(), atol=1e-3)
    assert np.allclose(kvol[0], sitk.GetArrayFromImage(itk), atol=1.0)


def test_dicom_left_handed_direction_normalizes_like_simpleitk(tmp_path: Path) -> None:
    """A feet-first (z-down) direction round-trips to a right-handed frame, matching SimpleITK.

    DICOM cannot store an arbitrary left-handed direction: the slice axis is
    derived from positions. KonfAI must reproduce SimpleITK's normalization
    (flipped array + adjusted origin/direction describing the SAME physical volume).
    """
    pytest.importorskip("pydicom")
    sitk = pytest.importorskip("SimpleITK")
    from konfai.utils import dicom

    root = tmp_path / "series"
    # distinct value per slice so ordering is observable
    vol = np.stack([np.full((4, 5), k, np.float32) for k in range(6)])[np.newaxis]
    dicom.write_dicom_series(
        root,
        vol,
        origin=(0.0, 0.0, 30.0),
        spacing=(1.0, 1.0, 2.0),
        direction=np.array([1, 0, 0, 0, 1, 0, 0, 0, -1], float),
    )
    kvol, kog, _ksp, kdir = dicom.read_dicom_series(root)

    reader = sitk.ImageSeriesReader()
    ids = reader.GetGDCMSeriesIDs(str(root))
    reader.SetFileNames(reader.GetGDCMSeriesFileNames(str(root), ids[0]))
    itk = reader.Execute()

    assert np.allclose(kog, itk.GetOrigin(), atol=1e-3)
    assert np.allclose(kdir, itk.GetDirection(), atol=1e-3)
    assert np.allclose(kvol[0], sitk.GetArrayFromImage(itk), atol=1.0)
    # right-handed normalization: z-cosine flips from -1 to +1
    assert np.allclose(kdir.reshape(3, 3)[:, 2], [0, 0, 1], atol=1e-6)


def test_dicom_slice_arity_mismatch_raises_dataset_manager_error(tmp_path: Path) -> None:
    pytest.importorskip("pydicom")
    from konfai.utils import dicom

    root = tmp_path / "series"
    vol = (np.arange(1 * 4 * 5 * 6).reshape(1, 4, 5, 6)).astype(np.float32)
    dicom.write_dicom_series(root, vol, origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0))
    with pytest.raises(DatasetManagerError):
        dicom.read_dicom_series_slice(root, (slice(None), slice(0, 2)))  # 2 slices, expected 4


# ---------------------------------------------------------------------------
# OME-Zarr round-trips & interop
# ---------------------------------------------------------------------------


def test_ome_zarr_output_is_readable_by_ngff_zarr(tmp_path: Path) -> None:
    """KonfAI's hand-written OME-Zarr must stay interoperable with ngff-zarr.

    Guards against the hand-rolled NGFF metadata drifting from the standard the
    ``ngff-zarr`` dependency implements.
    """
    pytest.importorskip("zarr")
    ngff_zarr = pytest.importorskip("ngff_zarr")
    from konfai.utils import ome_zarr

    store = tmp_path / "img.ome.zarr"
    data = (np.arange(1 * 6 * 8 * 10).reshape(1, 6, 8, 10) % 50).astype(np.float32)
    ome_zarr.write_ome_zarr(store, data, spacing=(0.5, 0.6, 2.0), origin=(1.0, 2.0, 3.0))

    mz = ngff_zarr.from_ngff_zarr(str(store))
    img0 = mz.images[0]
    assert tuple(img0.dims) == ("c", "z", "y", "x")
    assert img0.scale == {"c": 1.0, "z": 2.0, "y": 0.6, "x": 0.5}
    assert img0.translation == {"c": 0.0, "z": 3.0, "y": 2.0, "x": 1.0}
    assert np.array_equal(np.asarray(img0.data), data)


def test_ome_zarr_backend_preserves_non_identity_direction(tmp_path: Path) -> None:
    pytest.importorskip("zarr")
    sitk = pytest.importorskip("SimpleITK")
    from konfai.utils.dataset import Dataset

    root = tmp_path / "ds"
    root.mkdir()
    arr = (np.arange(8 * 12 * 10).reshape(8, 12, 10) % 97).astype(np.float32)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((0.7, 0.8, 2.5))
    img.SetOrigin((10.0, -5.0, 3.0))
    direction = (0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)  # 90° in-plane
    img.SetDirection(direction)

    Dataset.OmeZarrFile(str(root), read=False).data_to_file("CASE0", img, None)
    rdata, rattr = Dataset.OmeZarrFile(str(root), read=True).file_to_data("CT", "CASE0")

    assert np.allclose(rdata[0], arr)
    assert np.allclose(rattr.get_np_array("Spacing"), [0.7, 0.8, 2.5])
    assert np.allclose(rattr.get_np_array("Origin"), [10.0, -5.0, 3.0])
    assert np.allclose(rattr.get_np_array("Direction"), direction)


def test_ome_zarr_slice_arity_mismatch_raises_dataset_manager_error(tmp_path: Path) -> None:
    pytest.importorskip("zarr")
    from konfai.utils import ome_zarr

    store = tmp_path / "img.ome.zarr"
    data = (np.arange(1 * 4 * 5 * 6).reshape(1, 4, 5, 6)).astype(np.float32)
    ome_zarr.write_ome_zarr(store, data, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0))
    with pytest.raises(DatasetManagerError):
        ome_zarr.read_ome_zarr_data_slice(store, (slice(None), slice(0, 2)))  # 2 slices, expected 4


# ---------------------------------------------------------------------------
# OME-Zarr resolution (pyramid level) selection — `omezarr@<level>`
# ---------------------------------------------------------------------------


def test_split_format_level_parses_pyramid_suffix() -> None:
    from konfai.utils.utils import split_format_level

    assert split_format_level("omezarr") == ("omezarr", 0)
    assert split_format_level("omezarr@2") == ("omezarr", 2)
    assert split_format_level("ome-zarr@1") == ("ome-zarr", 1)
    assert split_format_level("mha") == ("mha", 0)  # unaffected
    assert split_format_level("C:/data@x") == ("C:/data@x", 0)  # non-numeric ignored


def test_dataset_parses_omezarr_level_field(tmp_path: Path) -> None:
    from konfai.utils.dataset import Dataset

    assert Dataset(tmp_path / "a", "omezarr@2").level == 2
    assert Dataset(tmp_path / "a", "omezarr").level == 0
    coarse = Dataset(tmp_path / "a", "ome-zarr@3")
    assert coarse.file_format == "omezarr" and coarse.level == 3


def test_ome_zarr_level_reads_coarser_resolution(tmp_path: Path) -> None:
    ngff_zarr = pytest.importorskip("ngff_zarr")
    pytest.importorskip("zarr")
    from konfai.utils.dataset import Dataset

    root = tmp_path / "ds"
    root.mkdir()
    data = (np.arange(1 * 16 * 32 * 32).reshape(1, 16, 32, 32) % 50).astype(np.float32)
    image = ngff_zarr.to_ngff_image(
        data,
        dims=["c", "z", "y", "x"],
        scale={"c": 1.0, "z": 2.0, "y": 0.5, "x": 0.5},
        translation={"c": 0.0, "z": 0.0, "y": 0.0, "x": 0.0},
    )
    ngff_zarr.to_ngff_zarr(
        str(root / "CASE0.ome.zarr"), ngff_zarr.to_multiscales(image, scale_factors=[2]), overwrite=True, version="0.4"
    )

    full, attr0 = Dataset.OmeZarrFile(str(root), read=True, level=0).file_to_data("g", "CASE0")
    coarse, attr1 = Dataset.OmeZarrFile(str(root), read=True, level=1).file_to_data("g", "CASE0")

    assert list(full.shape) == [1, 16, 32, 32]
    assert list(coarse.shape) == [1, 8, 16, 16]  # level 1 is downsampled x2
    # spacing doubles at the coarser level
    np.testing.assert_allclose(attr1.get_np_array("Spacing"), 2.0 * attr0.get_np_array("Spacing"))
