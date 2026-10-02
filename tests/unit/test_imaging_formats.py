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

"""Unit tests for konfai/utils/dicom.py and konfai/utils/ome_zarr.py."""

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from konfai.utils.dataset import Attribute, Dataset
from konfai.utils.errors import DatasetManagerError
from konfai.utils.utils import (
    SUPPORTED_FORMATS,
    directory_volume_form,
    path_format_token,
    split_path_spec,
    storage_form,
)
from oracle_support import geometry


def _image_attributes() -> Attribute:
    return geometry((10.0, 20.0, 30.0), (0.5, 1.5, 2.0))


# Writes a 4-slice series to <root>/<case>/CT and dies (no cleanup runs) as it saves the third slice.
_KILLED_AT_THE_THIRD_SLICE = """
import os, sys
import numpy as np
import pydicom.dataset
from konfai.utils.dataset import Attribute, Dataset

save_as = pydicom.dataset.FileDataset.save_as
saved = []

def dying_save_as(self, *args, **kwargs):
    if len(saved) == 2:
        os._exit(9)
    saved.append(self)
    return save_as(self, *args, **kwargs)

pydicom.dataset.FileDataset.save_as = dying_save_as
attributes = Attribute()
attributes["Origin"], attributes["Spacing"], attributes["Direction"] = np.zeros(3), np.ones(3), np.eye(3).flatten()
Dataset(sys.argv[1], "dicom").write("CT", sys.argv[2], np.full((1, 4, 8, 6), 3, dtype=np.int16), attributes)
"""


def test_flatten_transforms_recurses_into_nested_composites() -> None:
    """The transform serializer walks a composite to its leaves. SimpleITK keeps a nested composite
    nested (``GetNthTransform`` returns it as-is), so a single-level walk would hand that composite to
    the per-leaf type switch, which rejects it: the recursion is what keeps a nested chain storable."""
    sitk = pytest.importorskip("SimpleITK")
    from konfai.utils.dataset.attribute import _flatten_transforms

    inner = sitk.CompositeTransform([sitk.Euler3DTransform(), sitk.AffineTransform(3)])
    outer = sitk.CompositeTransform(3)
    outer.AddTransform(sitk.Euler3DTransform())
    outer.AddTransform(inner)  # the nested composite SimpleITK preserves verbatim

    assert isinstance(outer.GetNthTransform(1), sitk.CompositeTransform)  # premise: nesting survives
    leaves = _flatten_transforms(outer)
    assert [type(t).__name__ for t in leaves] == ["Euler3DTransform", "Euler3DTransform", "AffineTransform"]
    assert all(not isinstance(t, sitk.CompositeTransform) for t in leaves)
    assert len(_flatten_transforms(sitk.Euler3DTransform())) == 1  # a lone transform is its own single leaf


# ---------------------------------------------------------------------------
# DICOM tests (no real DICOM files: uses unittest.mock)
# ---------------------------------------------------------------------------


class TestDicomRequirePydicom:
    def test_raises_without_pydicom(self) -> None:
        from konfai.utils import dicom

        with patch.object(dicom, "_PYDICOM_AVAILABLE", False):
            with pytest.raises(DatasetManagerError, match="pydicom is required"):
                dicom._require_pydicom()

    def test_passes_with_pydicom(self) -> None:
        from konfai.utils import dicom

        with patch.object(dicom, "_PYDICOM_AVAILABLE", True):
            dicom._require_pydicom()  # must not raise


class TestDicomDiscoverSeries:
    def test_raises_on_missing_directory(self, tmp_path: Path) -> None:
        from konfai.utils import dicom

        with patch.object(dicom, "_PYDICOM_AVAILABLE", True):
            with pytest.raises(DatasetManagerError, match="does not exist"):
                dicom.discover_series(tmp_path / "nonexistent")

    def test_raises_when_no_dicom_found(self, tmp_path: Path) -> None:
        from konfai.utils import dicom

        (tmp_path / "file.txt").write_text("not a dicom")
        with patch.object(dicom, "_PYDICOM_AVAILABLE", True):
            with patch.object(dicom, "pydicom") as mock_pd:
                mock_pd.dcmread.side_effect = Exception("not dicom")
                with pytest.raises(DatasetManagerError, match="No DICOM files"):
                    dicom.discover_series(tmp_path)


class TestDicomSlicePosition:
    def test_uses_ipp_and_iop(self) -> None:
        from konfai.utils import dicom

        ds = MagicMock()
        # Row = (1, 0, 0), Col = (0, 1, 0) -> normal = (0, 0, 1)
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.ImagePositionPatient = [0, 0, 42.5]
        assert dicom._slice_position(ds) == pytest.approx(42.5)

    def test_falls_back_to_instance_number(self) -> None:
        from konfai.utils import dicom

        ds = MagicMock(spec=[])
        ds.InstanceNumber = 7
        assert dicom._slice_position(ds) == 7.0

    def test_returns_zero_when_no_tags(self) -> None:
        from konfai.utils import dicom

        ds = MagicMock(spec=[])
        assert dicom._slice_position(ds) == 0.0


class TestDicomExtractGeometry:
    def _make_ds(self, ipp: list[float]) -> MagicMock:
        ds = MagicMock()
        ds.ImagePositionPatient = ipp
        ds.PixelSpacing = [0.5, 0.5]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.SliceThickness = 1.0
        return ds

    def test_extracts_correct_spacing(self) -> None:
        from konfai.utils import dicom

        ds0 = self._make_ds([0.0, 0.0, 0.0])
        ds1 = self._make_ds([0.0, 0.0, 3.0])
        _, spacing, _ = dicom.extract_geometry([ds0, ds1])
        assert spacing[0] == pytest.approx(0.5)
        assert spacing[1] == pytest.approx(0.5)
        assert spacing[2] == pytest.approx(3.0)

    def test_fallback_to_slice_thickness_for_single_slice(self) -> None:
        from konfai.utils import dicom

        ds = self._make_ds([0.0, 0.0, 0.0])
        _, spacing, _ = dicom.extract_geometry([ds])
        assert spacing[2] == pytest.approx(1.0)

    def test_converts_pixel_spacing_to_xyz_order(self) -> None:
        from konfai.utils import dicom

        ds = self._make_ds([0.0, 0.0, 0.0])
        ds.PixelSpacing = [1.5, 0.5]
        _, spacing, _ = dicom.extract_geometry([ds])
        np.testing.assert_allclose(spacing, [0.5, 1.5, 1.0])

    def test_raises_on_missing_ipp(self) -> None:
        from konfai.utils import dicom

        ds = MagicMock(spec=[])
        with pytest.raises(DatasetManagerError, match="ImagePositionPatient"):
            dicom.extract_geometry([ds])

    def test_rejects_multi_frame_dicom(self) -> None:
        from konfai.utils import dicom

        ds = self._make_ds([0.0, 0.0, 0.0])
        ds.NumberOfFrames = 3
        with pytest.raises(DatasetManagerError, match="Multi-frame"):
            dicom.extract_geometry([ds])

    def test_rejects_irregular_slice_spacing(self) -> None:
        from konfai.utils import dicom

        # Gaps of 3 mm then 4 mm -> not uniformly spaced.
        slices = [self._make_ds([0.0, 0.0, z]) for z in (0.0, 3.0, 7.0)]
        with pytest.raises(DatasetManagerError, match="not uniformly spaced"):
            dicom.extract_geometry(slices)

    def test_rejects_a_slice_without_orientation(self) -> None:
        """Its position falls back to its InstanceNumber, which is no distance: with two slices there
        is no second gap to disagree with, and that number became the spacing."""
        from konfai.utils import dicom

        ds0 = self._make_ds([0.0, 0.0, 0.0])
        ds1 = MagicMock(spec=["ImagePositionPatient", "PixelSpacing", "SliceThickness", "InstanceNumber"])
        ds1.ImagePositionPatient, ds1.PixelSpacing, ds1.SliceThickness = [0.0, 0.0, 3.0], [0.5, 0.5], 1.0
        ds1.InstanceNumber = 2
        with pytest.raises(DatasetManagerError, match="ImageOrientationPatient"):
            dicom.extract_geometry([ds0, ds1])

    def test_accepts_uniform_multi_slice_spacing(self) -> None:
        from konfai.utils import dicom

        slices = [self._make_ds([0.0, 0.0, z]) for z in (0.0, 3.0, 6.0)]
        _, spacing, _ = dicom.extract_geometry(slices)
        assert spacing[2] == pytest.approx(3.0)


def _write_dicom_slice(
    path: Path,
    pixels: np.ndarray,
    position: tuple[float, float, float],
    orientation: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
) -> None:
    """One slice of series ``1.2.3``, as a scanner exports it; a ``(Y, X, 3)`` array is an RGB slice."""
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = SecondaryCaptureImageStorage
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = "1.2"
    ds.SeriesInstanceUID = "1.2.3"
    ds.Rows, ds.Columns = pixels.shape[:2]
    ds.SamplesPerPixel = 3 if pixels.ndim == 3 else 1
    ds.PhotometricInterpretation = "RGB" if pixels.ndim == 3 else "MONOCHROME2"
    if pixels.ndim == 3:
        ds.PlanarConfiguration = 0
    ds.BitsAllocated = ds.BitsStored = pixels.dtype.itemsize * 8
    ds.HighBit = ds.BitsStored - 1
    ds.PixelRepresentation = 0
    ds.PixelSpacing = [1.0, 1.0]
    ds.SliceThickness = 2.0
    ds.ImageOrientationPatient = list(orientation)
    ds.ImagePositionPatient = list(position)
    ds.PixelData = pixels.tobytes()
    ds.save_as(str(path), enforce_file_format=True)


class TestDicomSeriesTheReaderRefuses:
    """Series whose slices do not make one scalar volume on one grid: every route refuses them by name."""

    @staticmethod
    def _series(root: Path, slices: list[tuple[np.ndarray, tuple[float, float, float], tuple[float, ...]]]) -> Path:
        series = root / "CASE_001" / "CT"
        series.mkdir(parents=True)
        for index, (pixels, position, orientation) in enumerate(slices):
            _write_dicom_slice(series / f"{index:03d}.dcm", pixels, position, orientation)
        return series

    @staticmethod
    def _assert_refused(root: Path, match: str) -> None:
        from konfai.utils import dicom

        dicom.forget_series()
        dataset = Dataset(root, "dicom")
        with pytest.raises(DatasetManagerError, match=match):
            dataset.get_infos("CT", "CASE_001")
        with pytest.raises(DatasetManagerError, match=match):
            dataset.read_data("CT", "CASE_001")

    def test_a_colour_series_is_refused(self, tmp_path: Path) -> None:
        """Three samples per pixel read as a 5-D array while the header route announced ``[1, Z, Y, X]``."""
        pytest.importorskip("pydicom")
        axial = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        rgb = np.full((4, 5, 3), 7, dtype=np.uint8)
        self._series(tmp_path, [(rgb, (0.0, 0.0, 2.0 * z), axial) for z in range(3)])

        self._assert_refused(tmp_path, "SamplesPerPixel")

    def test_slices_sharing_one_position_are_refused(self, tmp_path: Path) -> None:
        """Four frames at one location (a cine) stacked along z with a spacing of 0 mm."""
        pytest.importorskip("pydicom")
        axial = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        pixels = np.arange(20, dtype=np.uint16).reshape(4, 5)
        self._series(tmp_path, [(pixels, (0.0, 0.0, 0.0), axial) for _ in range(4)])

        self._assert_refused(tmp_path, "share one position")

    def test_a_series_mixing_orientations_is_refused(self, tmp_path: Path) -> None:
        """A sagittal slice between two axial ones, at evenly spaced positions along each slice's own
        normal: stacked as one axial volume under the first slice's direction."""
        pytest.importorskip("pydicom")
        axial = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        sagittal = (0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        pixels = np.arange(20, dtype=np.uint16).reshape(4, 5)
        self._series(
            tmp_path,
            [(pixels, (0.0, 0.0, 0.0), axial), (pixels, (2.0, 0.0, 2.0), sagittal), (pixels, (0.0, 0.0, 4.0), axial)],
        )

        self._assert_refused(tmp_path, "orientation")


class TestDicomReadVolume:
    def _make_ds(self, value: float = 0.0) -> MagicMock:
        ds = MagicMock()
        ds.pixel_array = np.full((4, 4), value, dtype=np.int16)
        ds.RescaleSlope = 1.0
        ds.RescaleIntercept = -1000.0
        return ds

    def test_stacks_slices_channel_first(self) -> None:
        from konfai.utils import dicom

        datasets = [self._make_ds(0.0), self._make_ds(1.0)]
        volume = dicom.read_volume(datasets)
        assert volume.shape == (1, 2, 4, 4)

    def test_applies_ct_rescale(self) -> None:
        from konfai.utils import dicom

        datasets = [self._make_ds(0.0)]
        volume = dicom.read_volume(datasets, apply_rescale=True)
        assert volume[0, 0, 0, 0] == pytest.approx(-1000.0)

    def test_skips_rescale_when_disabled(self) -> None:
        from konfai.utils import dicom

        datasets = [self._make_ds(500.0)]
        volume = dicom.read_volume(datasets, apply_rescale=False)
        assert volume[0, 0, 0, 0] == pytest.approx(500.0)

    def test_raises_on_inconsistent_shapes(self) -> None:
        from konfai.utils import dicom

        ds0 = MagicMock()
        ds0.pixel_array = np.zeros((4, 4), dtype=np.int16)
        ds0.RescaleSlope = 1.0
        ds0.RescaleIntercept = 0.0
        ds1 = MagicMock()
        ds1.pixel_array = np.zeros((8, 8), dtype=np.int16)
        ds1.RescaleSlope = 1.0
        ds1.RescaleIntercept = 0.0
        with pytest.raises(DatasetManagerError, match="Inconsistent slice shape"):
            dicom.read_volume([ds0, ds1])


class TestDicomRegionDecode:
    """A region decodes the window of each selected slice, and nothing of the rest of the plane."""

    @staticmethod
    def _former_slice_read(root: Path, region: tuple[slice, ...]) -> np.ndarray:
        """The full-plane route: every selected plane cast and rescaled whole, stacked, then cut."""
        import pydicom
        from konfai.utils import dicom

        info = dicom.get_dicom_info(root)
        z_indices = range(*region[1].indices(info["shape"][1]))
        planes = []
        for index in z_indices:
            ds = pydicom.dcmread(str(info["sorted_files"][index]))
            arr = ds.pixel_array.astype(np.float32)
            planes.append(arr * float(ds.RescaleSlope) + float(ds.RescaleIntercept))
        return np.stack(planes, axis=0)[np.newaxis][region[0], :, region[2], region[3]]

    def test_a_window_cut_before_the_rescale_is_bit_identical_to_the_whole_plane(self, tmp_path: Path) -> None:
        pytest.importorskip("pydicom")
        from konfai.utils import dicom

        root = tmp_path / "CT"
        volume = (np.random.default_rng(3).normal(size=(1, 6, 24, 20)) * 300 - 200).astype(np.float32)
        dicom.write_dicom_series(root, volume, origin=(1.0, 2.0, 3.0), spacing=(0.7, 0.8, 2.5))
        region = (slice(None), slice(1, 5), slice(3, 17), slice(2, 19))

        got, *_ = dicom.read_dicom_series_slice(root, region)
        want = self._former_slice_read(root, region)

        assert got.dtype == want.dtype == np.float32
        assert got.tobytes() == want.tobytes()
        whole, *_ = dicom.read_dicom_series(root)
        assert whole.tobytes() == self._former_slice_read(root, (slice(None),) * 4).tobytes()

    def test_a_region_reads_its_slices_in_order_without_sorting_them_again(self, tmp_path: Path, monkeypatch) -> None:
        """``sorted_files`` is in slice order: the selection is decoded in that order, once each."""
        pytest.importorskip("pydicom")
        from konfai.utils import dicom

        root = tmp_path / "CT"
        volume = np.arange(1 * 5 * 4 * 4, dtype=np.int16).reshape(1, 5, 4, 4)
        dicom.write_dicom_series(root, volume, origin=(0.0,) * 3, spacing=(1.0,) * 3)
        dicom.get_dicom_info(root)
        sorts = {"count": 0}
        real_sort = dicom.sort_series

        def counting(*args, **kwargs):
            sorts["count"] += 1
            return real_sort(*args, **kwargs)

        monkeypatch.setattr(dicom, "sort_series", counting)
        got, *_ = dicom.read_dicom_series_slice(root, (slice(None), slice(1, 4, 2), slice(None), slice(None)))

        np.testing.assert_array_equal(got, volume[:, 1:4:2])
        assert sorts["count"] == 0

    def test_overlapping_region_reads_decode_each_plane_once(self, tmp_path: Path, monkeypatch) -> None:
        """A series stores one file per plane, and a region read touching a z index decodes that
        file whole. The plane cache decodes each file once per pass, so overlapping regions of a sweep
        do not pay one full ``dcmread`` per touched slice per region."""
        pydicom = pytest.importorskip("pydicom")
        from konfai.utils import dicom

        root = tmp_path / "CT"
        volume = np.arange(1 * 6 * 8 * 8, dtype=np.int16).reshape(1, 6, 8, 8)
        dicom.write_dicom_series(root, volume, origin=(0.0,) * 3, spacing=(1.0,) * 3)
        dicom.get_dicom_info(root)
        reads: list[str] = []
        real_dcmread = pydicom.dcmread

        def counting(*args, **kwargs):
            if not kwargs.get("stop_before_pixels", False):
                reads.append(str(args[0]))
            return real_dcmread(*args, **kwargs)

        monkeypatch.setattr(pydicom, "dcmread", counting)
        first, *_ = dicom.read_dicom_series_slice(root, (slice(None), slice(0, 4), slice(1, 6), slice(0, 5)))
        second, *_ = dicom.read_dicom_series_slice(root, (slice(None), slice(2, 6), slice(2, 8), slice(3, 8)))

        np.testing.assert_array_equal(first, volume[:, 0:4, 1:6, 0:5])
        np.testing.assert_array_equal(second, volume[:, 2:6, 2:8, 3:8])
        assert len(reads) == 6, "six distinct planes touched; the two overlapping slices decode once"
        assert len(set(reads)) == 6

    def test_the_plane_cache_takes_the_cache_share_of_a_declared_budget(self) -> None:
        pytest.importorskip("pydicom")
        from konfai.utils import dicom
        from konfai.utils.budget import BUDGET_SHARES, set_per_rank_budget

        try:
            set_per_rank_budget(128 << 20)
            assert dicom._plane_cache_capacity() == int((128 << 20) * BUDGET_SHARES["cache"])
        finally:
            set_per_rank_budget(None)
        assert dicom._plane_cache_capacity() == dicom._PLANE_CACHE_DEFAULT_BYTES

    def test_a_dicom_series_declares_its_plane_as_the_read_grain(self, tmp_path: Path) -> None:
        """One file per z step, decoded as a whole plane: the plan aligns and prices a sweep on the
        plane exactly as it does a memmapped band, instead of trusting a silent None."""
        pytest.importorskip("pydicom")
        volume = np.zeros((1, 3, 6, 5), dtype=np.int16)
        dataset = Dataset(tmp_path / "DICOM", "dicom")
        dataset.write("CT", "CASE_001", volume, _image_attributes())

        assert dataset.read_granularity("CT", "CASE_001") == (1, 1, 6, 5)

    def test_the_series_info_memo_is_unbounded_and_a_write_clears_it(self, tmp_path: Path) -> None:
        """A miss re-reads every slice header; a bound of 64 series missed on every patch of a
        cohort read in any order but case by case. A write of a series is what changes a directory."""
        pytest.importorskip("pydicom")
        from konfai.utils import dicom

        assert dicom._dicom_info.cache_info().maxsize is None
        root = tmp_path / "CT"
        dicom.write_dicom_series(root, np.zeros((1, 3, 4, 4), dtype=np.int16), origin=(0.0,) * 3, spacing=(1.0,) * 3)
        assert dicom.get_dicom_info(root)["shape"] == [1, 3, 4, 4]
        dicom.write_dicom_series(root, np.zeros((1, 5, 4, 4), dtype=np.int16), origin=(0.0,) * 3, spacing=(1.0,) * 3)
        assert dicom.get_dicom_info(root)["shape"] == [1, 5, 4, 4]


# ---------------------------------------------------------------------------
# OME-Zarr tests (no real Zarr store: uses unittest.mock)
# ---------------------------------------------------------------------------


class TestOmeZarrRequireZarr:
    def test_raises_without_zarr(self) -> None:
        from konfai.utils import ome_zarr

        with patch.object(ome_zarr, "_zarr_available", lambda: False):
            with pytest.raises(DatasetManagerError, match="zarr is required"):
                ome_zarr._require_zarr()


class TestDatasetImagingBackends:
    def test_ome_zarr_dataset_round_trip_and_patch_read(self, tmp_path: Path) -> None:
        pytest.importorskip("zarr")
        volume = np.arange(1 * 3 * 4 * 5, dtype=np.int16).reshape(1, 3, 4, 5)
        dataset = Dataset(tmp_path / "OME", "ome-zarr")

        dataset.write("CT", "CASE_001", volume, _image_attributes())

        assert dataset.file_format == "omezarr"
        assert dataset.get_names("CT") == ["CASE_001"]
        assert dataset.get_group() == ["CT"]
        assert dataset.get_infos("CT", "CASE_001")[0] == [1, 3, 4, 5]
        full, attributes = dataset.read_data("CT", "CASE_001")
        patch, patch_attributes = dataset.read_data_slice(
            "CT", "CASE_001", (slice(None), slice(1, 3), slice(1, 4), slice(2, 5))
        )
        np.testing.assert_array_equal(full, volume)
        np.testing.assert_array_equal(patch, volume[:, 1:3, 1:4, 2:5])
        np.testing.assert_allclose(attributes.get_np_array("Spacing"), [0.5, 1.5, 2.0])
        np.testing.assert_allclose(patch_attributes.get_np_array("Origin"), [11.0, 21.5, 32.0])

    def test_ome_zarr_2d_dataset_round_trip(self, tmp_path: Path) -> None:
        pytest.importorskip("zarr")
        volume = np.arange(2 * 4 * 5, dtype=np.uint8).reshape(2, 4, 5)
        attributes = Attribute()
        attributes["Origin"] = np.asarray([10.0, 20.0])
        attributes["Spacing"] = np.asarray([0.5, 1.5])
        attributes["Direction"] = np.eye(2, dtype=np.float64).flatten()
        dataset = Dataset(tmp_path / "OME2D", "omezarr")

        dataset.write("RGB", "CASE_001", volume, attributes)
        result, result_attributes = dataset.read_data("RGB", "CASE_001")

        np.testing.assert_array_equal(result, volume)
        np.testing.assert_allclose(result_attributes.get_np_array("Origin"), [10.0, 20.0])

    def test_dicom_dataset_round_trip_and_patch_read(self, tmp_path: Path) -> None:
        pytest.importorskip("pydicom")
        volume = np.arange(1 * 3 * 4 * 5, dtype=np.int16).reshape(1, 3, 4, 5)
        dataset = Dataset(tmp_path / "DICOM", "dicom")

        dataset.write("CT", "CASE_001", volume, _image_attributes())

        assert dataset.get_names("CT") == ["CASE_001"]
        assert dataset.get_group() == ["CT"]
        assert dataset.get_infos("CT", "CASE_001")[0] == [1, 3, 4, 5]
        full, attributes = dataset.read_data("CT", "CASE_001")
        patch, patch_attributes = dataset.read_data_slice(
            "CT", "CASE_001", (slice(None), slice(1, 3), slice(1, 4), slice(2, 5))
        )
        np.testing.assert_array_equal(full, volume)
        np.testing.assert_array_equal(patch, volume[:, 1:3, 1:4, 2:5])
        np.testing.assert_allclose(attributes.get_np_array("Spacing"), [0.5, 1.5, 2.0])
        np.testing.assert_allclose(patch_attributes.get_np_array("Origin"), [11.0, 21.5, 32.0])

    def test_dicom_slice_read_decodes_only_selected_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pydicom = pytest.importorskip("pydicom")
        volume = np.arange(1 * 4 * 3 * 3, dtype=np.int16).reshape(1, 4, 3, 3)
        dataset = Dataset(tmp_path / "DICOM", "dicom")
        dataset.write("CT", "CASE_001", volume, _image_attributes())
        decoded_files: list[str] = []
        real_dcmread = pydicom.dcmread

        def tracked_dcmread(*args, **kwargs):
            if not kwargs.get("stop_before_pixels", False):
                decoded_files.append(str(args[0]))
            return real_dcmread(*args, **kwargs)

        monkeypatch.setattr(pydicom, "dcmread", tracked_dcmread)
        patch, _ = dataset.read_data_slice("CT", "CASE_001", (slice(None), slice(2, 3), slice(None), slice(None)))

        np.testing.assert_array_equal(patch, volume[:, 2:3])
        assert len(decoded_files) == 1

    def test_dicom_region_reads_are_priced_bounded(self, tmp_path: Path) -> None:
        """The route pricer must see what the decode-count test above proves: a region decodes only
        its slices, so a DICOM source prices as streamable instead of loading whole."""
        pytest.importorskip("pydicom")
        dataset = Dataset(tmp_path / "DICOM", "dicom")
        dataset.write("CT", "CASE_001", np.zeros((1, 2, 3, 3), dtype=np.int16), _image_attributes())

        assert dataset.bounded_region_reads("CT", "CASE_001")

    def test_dicom_round_trip_preserves_rotated_direction(self, tmp_path: Path) -> None:
        pytest.importorskip("pydicom")
        attributes = _image_attributes()
        direction = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        attributes["Direction"] = direction.flatten()
        dataset = Dataset(tmp_path / "DICOM", "dicom")
        dataset.write("CT", "CASE_001", np.zeros((1, 2, 3, 3), dtype=np.int16), attributes)

        _, result_attributes = dataset.read_data("CT", "CASE_001")

        np.testing.assert_allclose(result_attributes.get_np_array("Direction"), direction.flatten())

    def test_dicom_write_preserves_unrelated_files(self, tmp_path: Path) -> None:
        pytest.importorskip("pydicom")
        from konfai.utils import dicom

        root = tmp_path / "DICOM"
        root.mkdir()
        unrelated = root / "keep-me.dcm"
        unrelated.write_bytes(b"not a konfai slice")

        volume = np.zeros((1, 3, 4, 4), dtype=np.int16)
        dicom.write_dicom_series(root, volume, origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0))

        # The series was written, but the unrelated DICOM file was not deleted.
        assert unrelated.exists()
        assert sorted(p.name for p in root.glob("[0-9]*.dcm")) == ["000001.dcm", "000002.dcm", "000003.dcm"]

    def test_a_dicom_writer_killed_mid_series_leaves_the_previous_series_or_no_case(self, tmp_path: Path) -> None:
        """A writer killed at its third slice: a series it was replacing is still the previous one,
        whole, and a first write that died is no case, so a resume writes it again."""
        pytest.importorskip("pydicom")
        root = tmp_path / "DICOM"
        previous = np.full((1, 4, 8, 6), 7, dtype=np.int16)
        Dataset(root, "dicom").write("CT", "CASE_001", previous, _image_attributes())
        for case in ("CASE_001", "CASE_002"):
            child = subprocess.run(
                [sys.executable, "-c", _KILLED_AT_THE_THIRD_SLICE, str(root), case], capture_output=True, text=True
            )
            assert child.returncode == 9, child.stderr

        fresh = Dataset(root, "dicom")
        back, _ = fresh.read_data("CT", "CASE_001")
        np.testing.assert_array_equal(back, previous)
        assert fresh.get_infos("CT", "CASE_001")[0] == [1, 4, 8, 6]
        assert not fresh.is_dataset_exist("CT", "CASE_002")
        assert fresh.get_names("CT") == ["CASE_001"]
        assert fresh.get_group() == ["CT"]

    @pytest.mark.parametrize(
        ("dtype", "outside"), [(np.int64, 2**31 + 5), (np.int64, -(2**31) - 5), (np.uint64, 2**32 + 5)]
    )
    def test_a_dicom_write_refuses_integers_its_pixels_cannot_hold(
        self, tmp_path: Path, dtype: type, outside: int
    ) -> None:
        """A slice stores at most 32 bits: a 64-bit value past that range was cast down and wrapped."""
        pytest.importorskip("pydicom")
        dataset = Dataset(tmp_path / "DICOM", "dicom")
        inside = np.full((1, 2, 3, 3), 1000, dtype=dtype)
        dataset.write("CT", "CASE_001", inside, _image_attributes())
        np.testing.assert_array_equal(dataset.read_data("CT", "CASE_001")[0], inside)

        volume = inside.copy()
        volume[0, 1, 2, 2] = outside
        with pytest.raises(DatasetManagerError, match="do not fit"):
            dataset.write("CT", "CASE_002", volume, _image_attributes())
        assert not dataset.is_dataset_exist("CT", "CASE_002")

    @pytest.mark.parametrize("file_format", ["omezarr", "ome-zarr", "ome_zarr", "zarr"])
    def test_ome_zarr_format_aliases(self, tmp_path: Path, file_format: str) -> None:
        assert Dataset(tmp_path / file_format, file_format).file_format == "omezarr"

    @pytest.mark.parametrize(
        ("name", "form"),
        [
            ("patient.mha", ".mha"),
            ("patient.v2.mha", ".mha"),  # a dot in the STEM is part of the name
            ("CT.contrast.nii.gz", ".nii.gz"),  # the compound extension, not its tail
            ("study.ome.zarr", ".ome.zarr"),  # ".ome.zarr" wins over ".zarr"
            ("scan.UNKNOWN", ".UNKNOWN"),  # nothing matches: the last suffix, as spelled
        ],
    )
    def test_storage_form_is_the_extension_not_every_dotted_segment(self, name: str, form: str) -> None:
        """``"".join(path.suffixes)`` reads ``patient.v2.mha`` as a ``v2.mha`` backend, which lets how a
        caller named a file decide whether a run starts at all."""
        assert storage_form(Path(name)) == form

    def test_a_dicom_series_is_a_directory_volume_with_no_form(self, tmp_path: Path) -> None:
        """A series carries no extension: neither the directory nor, often, the slices. Read off the
        name alone it looks like nothing, and the default file format is what a caller would get."""
        pytest.importorskip("pydicom")
        from konfai.utils import dicom

        root = tmp_path / "SERIES"
        root.mkdir()
        dicom.write_dicom_series(root, np.zeros((1, 3, 4, 4), dtype=np.int16), origin=(0.0,) * 3, spacing=(1.0,) * 3)
        for slice_file in sorted(root.glob("*.dcm")):  # exported without an extension, as scanners do
            slice_file.rename(slice_file.with_suffix(""))

        assert storage_form(root) == ""
        assert directory_volume_form(root) == ""  # a volume, not a directory of volumes
        assert path_format_token(root) == "dicom"

    def test_a_plain_directory_of_volumes_is_not_one(self, tmp_path: Path) -> None:
        root = tmp_path / "CASES"
        root.mkdir()
        Dataset(root / "A", "mha").write("CT", "CASE_001", np.zeros((1, 2, 3, 3), dtype=np.int16), _image_attributes())

        assert directory_volume_form(root) is None

    @pytest.mark.parametrize("file_format", ["dicom", "omezarr", "ome-zarr", "ome_zarr", "zarr", "itktransform"])
    def test_data_manager_path_parser_accepts_imaging_backend(self, file_format: str) -> None:
        assert file_format in SUPPORTED_FORMATS
        assert split_path_spec(
            f"./Dataset:a:{file_format}",
            allowed_flags={"a", "i"},
            supported_formats=SUPPORTED_FORMATS,
        ) == ("./Dataset", "a", file_format)

    @pytest.mark.parametrize("file_format", ["dicom", "omezarr"])
    def test_data_prediction_resolves_imaging_dataset_source(self, tmp_path: Path, file_format: str) -> None:
        from konfai.data.data_manager import DataPrediction, Group, GroupTransform

        volume = np.arange(1 * 2 * 3 * 3, dtype=np.int16).reshape(1, 2, 3, 3)
        root = tmp_path / file_format
        Dataset(root, file_format).write("CT", "CASE_001", volume, _image_attributes())
        prediction_data = DataPrediction(
            augmentations=None,
            dataset_filenames=[f"{root}:a:{file_format}"],
            groups_src={"CT": Group(groups_dest={"CT": GroupTransform(transforms=None, patch_transforms=None)})},
        )

        sources = prediction_data._resolve_dataset_sources()

        assert sources == {"CT": [(str(root), True)]}
