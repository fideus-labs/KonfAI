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

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from konfai.utils.dataset import Attribute, Dataset
from konfai.utils.errors import DatasetManagerError
from konfai.utils.utils import SUPPORTED_EXTENSIONS, split_path_spec


def _image_attributes() -> Attribute:
    attributes = Attribute()
    attributes["Origin"] = np.asarray([10.0, 20.0, 30.0])
    attributes["Spacing"] = np.asarray([0.5, 1.5, 2.0])
    attributes["Direction"] = np.eye(3, dtype=np.float64).flatten()
    return attributes


def test_flatten_transforms_recurses_into_nested_composites() -> None:
    """The transform serializer walks a composite to its leaves. SimpleITK keeps a nested composite
    nested (``GetNthTransform`` returns it as-is), so a single-level walk would hand that composite to
    the per-leaf type switch, which rejects it -- the recursion is what keeps a nested chain storable."""
    sitk = pytest.importorskip("SimpleITK")
    from konfai.utils.dataset import _flatten_transforms

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
# DICOM tests (no real DICOM files — uses unittest.mock)
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

    def test_accepts_uniform_multi_slice_spacing(self) -> None:
        from konfai.utils import dicom

        slices = [self._make_ds([0.0, 0.0, z]) for z in (0.0, 3.0, 6.0)]
        _, spacing, _ = dicom.extract_geometry(slices)
        assert spacing[2] == pytest.approx(3.0)


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


# ---------------------------------------------------------------------------
# OME-Zarr tests (no real Zarr store — uses unittest.mock)
# ---------------------------------------------------------------------------


class TestOmeZarrRequireZarr:
    def test_raises_without_zarr(self) -> None:
        from konfai.utils import ome_zarr

        with patch.object(ome_zarr, "_ZARR_AVAILABLE", False):
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

    @pytest.mark.parametrize("file_format", ["omezarr", "ome-zarr", "ome_zarr", "zarr"])
    def test_ome_zarr_format_aliases(self, tmp_path: Path, file_format: str) -> None:
        assert Dataset(tmp_path / file_format, file_format).file_format == "omezarr"

    @pytest.mark.parametrize("file_format", ["dicom", "omezarr", "ome-zarr", "ome_zarr", "zarr"])
    def test_data_manager_path_parser_accepts_imaging_backend(self, file_format: str) -> None:
        assert file_format in SUPPORTED_EXTENSIONS
        assert split_path_spec(
            f"./Dataset:a:{file_format}",
            allowed_flags={"a", "i"},
            supported_extensions=SUPPORTED_EXTENSIONS,
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
