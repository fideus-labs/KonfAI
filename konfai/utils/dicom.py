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

"""DICOM series reader for KonfAI medical imaging pipelines.

Design rationale
----------------
DICOM is not a folder of independent images.  A CT or MRI acquisition is a
*series* — a collection of .dcm files that together define a 3-D volume.
Reading a DICOM correctly requires:

1. **Series discovery** — group files by SeriesInstanceUID.  A folder may
   contain multiple series (e.g. a T1 and a T2 acquired in the same session).

2. **Slice ordering** — sort slices by ImagePositionPatient (z-component along
   ImageOrientationPatient normal vector), not by filename or InstanceNumber,
   which can be unreliable.

3. **Geometry extraction** — derive spacing_mm (PixelSpacing + SliceThickness /
   derived inter-slice distance), origin (ImagePositionPatient of first slice),
   and direction cosines (ImageOrientationPatient rows and columns + cross
   product for the z-axis).

4. **CT intensity rescale** — apply RescaleSlope and RescaleIntercept to
   convert stored pixel values to Hounsfield Units (HU).  This is mandatory
   for CT and is absent (or identity) for MR.

5. **Error handling** — missing tags, single-slice series, inconsistent spacing,
   non-square pixels, and unsupported transfer syntaxes all need clear messages.

Optional dependency: ``pydicom`` (``pip install konfai[dicom]``).
"""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from konfai.utils.errors import DatasetManagerError

# Zero-padded slice filenames produced by :func:`write_dicom_series` (e.g. ``000001.dcm``).
_SLICE_FILENAME_RE = re.compile(r"^\d{6}\.dcm$")

if TYPE_CHECKING:
    pass

try:
    import pydicom
    from pydicom.dataset import Dataset as DicomDataset
    from pydicom.sequence import Sequence as DicomSequence

    _PYDICOM_AVAILABLE = True
except ImportError:
    _PYDICOM_AVAILABLE = False
    pydicom = None  # type: ignore[assignment]
    DicomDataset = None  # type: ignore[assignment,misc]
    DicomSequence = None  # type: ignore[assignment]


def _require_pydicom() -> None:
    if not _PYDICOM_AVAILABLE:
        raise DatasetManagerError(
            "pydicom is required for DICOM support.",
            "Install it with: pip install konfai[dicom]",
        )


# ---------------------------------------------------------------------------
# Series discovery
# ---------------------------------------------------------------------------


def discover_series(directory: str | Path) -> dict[str, list[Path]]:
    """Return a mapping of SeriesInstanceUID -> sorted list of .dcm paths.

    Parameters
    ----------
    directory:
        Root directory to scan recursively for .dcm files.

    Returns
    -------
    dict[str, list[Path]]
        Keys are SeriesInstanceUID values; values are lists of file paths
        belonging to that series (unsorted at this stage).

    Raises
    ------
    DatasetManagerError
        If ``pydicom`` is not installed or the directory contains no DICOM.
    """
    _require_pydicom()

    root = Path(directory)
    if not root.is_dir():
        raise DatasetManagerError(f"DICOM directory '{root}' does not exist or is not a directory.")

    series: dict[str, list[Path]] = {}
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            fpath = Path(dirpath) / fname
            try:
                ds = pydicom.dcmread(str(fpath), stop_before_pixels=True)
                uid = str(ds.SeriesInstanceUID)
                series.setdefault(uid, []).append(fpath)
            except Exception:  # nosec B112
                # Skip unreadable or non-DICOM files; discovery must not crash on stray content.
                continue

    if not series:
        raise DatasetManagerError(
            f"No DICOM files found under '{root}'.",
            "Ensure the directory contains .dcm files from a valid DICOM series.",
        )
    return series


# ---------------------------------------------------------------------------
# Slice sorting
# ---------------------------------------------------------------------------


def _slice_position(ds: DicomDataset) -> float:
    """Return the signed position of one slice along the acquisition axis.

    Uses ``ImagePositionPatient`` projected onto the slice-normal derived from
    ``ImageOrientationPatient``.  Falls back to ``InstanceNumber`` (unreliable
    but ubiquitous) when geometry tags are absent.
    """
    try:
        iop = [float(x) for x in ds.ImageOrientationPatient]
        ipp = [float(x) for x in ds.ImagePositionPatient]
        row = np.array(iop[:3])
        col = np.array(iop[3:])
        normal = np.cross(row, col)
        return float(np.dot(normal, ipp))
    except AttributeError:
        try:
            return float(ds.InstanceNumber)
        except AttributeError:
            return 0.0


def sort_series(files: list[Path], *, stop_before_pixels: bool = False) -> list[DicomDataset]:
    """Read and sort slices in anatomical order (ascending slice position).

    Parameters
    ----------
    files:
        Unsorted list of paths belonging to one DICOM series.

    Returns
    -------
    list[DicomDataset]
        Datasets sorted by their position along the acquisition normal.
    """
    _require_pydicom()
    datasets = [pydicom.dcmread(str(f), stop_before_pixels=stop_before_pixels) for f in files]
    datasets.sort(key=_slice_position)
    return datasets


def _select_series_files(directory: str | Path, series_uid: str | None = None) -> tuple[str, list[Path]]:
    all_series = discover_series(directory)
    if series_uid is not None:
        if series_uid not in all_series:
            raise DatasetManagerError(
                f"Series '{series_uid}' not found in '{directory}'.",
                f"Available series: {list(all_series.keys())}",
            )
        return series_uid, all_series[series_uid]
    if len(all_series) == 1:
        selected_uid, files = next(iter(all_series.items()))
        return selected_uid, files
    raise DatasetManagerError(
        f"Multiple DICOM series found in '{directory}' ({len(all_series)} series).",
        "Specify 'series_uid' to select one.",
        f"Available UIDs: {list(all_series.keys())}",
    )


# ---------------------------------------------------------------------------
# Geometry extraction
# ---------------------------------------------------------------------------


def extract_geometry(
    datasets: list[DicomDataset],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract origin, spacing, and direction from a sorted DICOM series.

    Parameters
    ----------
    datasets:
        Slice datasets in anatomical order (from :func:`sort_series`).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        - ``origin`` (3,) — physical position of the first voxel (mm).
        - ``spacing`` (3,) — KonfAI/SimpleITK order (x, y, z) in mm.
        - ``direction`` (9,) — row-major 3-by-3 direction cosine matrix, flattened.

    Raises
    ------
    DatasetManagerError
        If required geometry tags are missing or inconsistent.
    """
    if not datasets:
        raise DatasetManagerError("Cannot extract geometry from an empty series.")

    first = datasets[0]

    # Multi-frame / enhanced DICOM (many frames in one file) would be mis-stacked as a
    # single slice, so reject it explicitly rather than produce a wrong volume.
    try:
        number_of_frames = int(getattr(first, "NumberOfFrames", 1) or 1)
    except (TypeError, ValueError):
        number_of_frames = 1
    if number_of_frames > 1:
        raise DatasetManagerError(
            "Multi-frame / enhanced DICOM is not supported.",
            f"The series declares NumberOfFrames={number_of_frames}.",
            "KonfAI expects one frame per file (classic single-frame DICOM).",
        )

    # Origin = ImagePositionPatient of first slice
    try:
        origin = np.array([float(x) for x in first.ImagePositionPatient], dtype=np.float64)
    except AttributeError as exc:
        raise DatasetManagerError(
            "DICOM tag 'ImagePositionPatient' is missing.",
            "This tag is required to determine the volume origin.",
        ) from exc

    # In-plane spacing from PixelSpacing
    try:
        pixel_spacing = [float(x) for x in first.PixelSpacing]
        row_spacing_mm = pixel_spacing[0]
        col_spacing_mm = pixel_spacing[1]
    except AttributeError as exc:
        raise DatasetManagerError(
            "DICOM tag 'PixelSpacing' is missing.",
            "This tag is required to determine voxel dimensions.",
        ) from exc

    # Slice spacing: prefer computed inter-slice distance over SliceThickness.
    # Use the first gap (matches SimpleITK's geometry), but verify the whole series is
    # uniformly spaced so irregular series (localizers, missing/duplicate slices, mixed
    # series) fail loudly instead of silently producing a wrong z-spacing.
    if len(datasets) > 1:
        gaps = np.abs(np.diff([_slice_position(ds) for ds in datasets]))
        slice_spacing_mm = float(gaps[0])
        tolerance = max(1e-2, 1e-2 * slice_spacing_mm)
        if float(np.ptp(gaps)) > tolerance:
            raise DatasetManagerError(
                "DICOM slices are not uniformly spaced.",
                f"Inter-slice gaps range from {float(gaps.min()):.4f} to {float(gaps.max()):.4f} mm.",
                "KonfAI assumes a regular z-spacing; irregular series are not supported.",
            )
    else:
        try:
            slice_spacing_mm = float(first.SliceThickness)
        except AttributeError:
            slice_spacing_mm = 1.0  # fallback; single-slice series

    spacing = np.array([col_spacing_mm, row_spacing_mm, slice_spacing_mm], dtype=np.float64)

    # Direction cosines
    try:
        iop = [float(x) for x in first.ImageOrientationPatient]
    except AttributeError as exc:
        raise DatasetManagerError(
            "DICOM tag 'ImageOrientationPatient' is missing.",
            "This tag is required to determine the volume orientation.",
        ) from exc

    row_cosine = np.array(iop[:3], dtype=np.float64)
    col_cosine = np.array(iop[3:], dtype=np.float64)
    normal_cosine = np.cross(row_cosine, col_cosine)
    direction = np.column_stack([row_cosine, col_cosine, normal_cosine]).flatten()

    return origin, spacing, direction


# ---------------------------------------------------------------------------
# Pixel reading with CT rescale
# ---------------------------------------------------------------------------


def read_volume(
    datasets: list[DicomDataset],
    *,
    apply_rescale: bool = True,
) -> np.ndarray:
    """Stack sorted slices into a channel-first (1, Z, Y, X) float32 array.

    Parameters
    ----------
    datasets:
        Sorted DICOM datasets (from :func:`sort_series`).
    apply_rescale:
        If True, apply RescaleSlope / RescaleIntercept to convert stored
        pixel values to Hounsfield Units (HU) for CT, or to physical signal
        units for modalities that provide these tags.  Set to False to keep
        raw stored pixel integers (e.g., for label maps or QC).

    Returns
    -------
    np.ndarray
        Shape (1, Z, Y, X), dtype float32.  Channel dimension = 1 for scalar
        volumes.

    Raises
    ------
    DatasetManagerError
        If pixel data cannot be read or slices have inconsistent shapes.
    """
    slices = []
    expected_shape: tuple[int, int] | None = None

    for i, ds in enumerate(datasets):
        try:
            arr = ds.pixel_array.astype(np.float32)
        except Exception as exc:
            raise DatasetManagerError(
                f"Cannot read pixel data from DICOM slice {i}.",
                f"Transfer syntax or compression may be unsupported: {exc}",
            ) from exc

        if expected_shape is None:
            expected_shape = arr.shape
        elif arr.shape != expected_shape:
            raise DatasetManagerError(
                f"Inconsistent slice shape at index {i}: expected {expected_shape}, got {arr.shape}.",
                "All slices in a series must have the same rows and columns.",
            )

        if apply_rescale:
            slope = float(getattr(ds, "RescaleSlope", 1.0))
            intercept = float(getattr(ds, "RescaleIntercept", 0.0))
            arr = arr * slope + intercept

        slices.append(arr)

    if not slices:
        raise DatasetManagerError("Series contains no readable slices.")

    volume = np.stack(slices, axis=0)  # (Z, Y, X)
    return volume[np.newaxis]  # (1, Z, Y, X)


@lru_cache(maxsize=64)
def get_dicom_info(
    directory: str | Path,
    *,
    series_uid: str | None = None,
) -> dict[str, Any]:
    """Read DICOM series shape and geometry without decoding pixel data.

    Memoised per directory: input DICOM is read-only for a run, so the series walk and header reads are
    done once instead of on every patch read. Callers that mutate the result must copy it first.
    """
    selected_uid, files = _select_series_files(directory, series_uid)
    datasets = sort_series(files, stop_before_pixels=True)
    origin, spacing, direction = extract_geometry(datasets)
    first = datasets[0]
    try:
        rows = int(first.Rows)
        columns = int(first.Columns)
    except AttributeError as exc:
        raise DatasetManagerError("DICOM Rows/Columns tags are required to determine the volume shape.") from exc
    return {
        "series_uid": selected_uid,
        "files": files,
        "sorted_files": [Path(ds.filename) for ds in datasets],
        "shape": [1, len(datasets), rows, columns],
        "origin": origin,
        "spacing": spacing,
        "direction": direction,
    }


def read_dicom_series_slice(
    directory: str | Path,
    slices: tuple[slice, ...],
    *,
    series_uid: str | None = None,
    apply_rescale: bool = True,
    info: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read only the selected DICOM slices and return updated patch geometry."""
    if info is None:
        info = get_dicom_info(directory, series_uid=series_uid)
    elif series_uid is not None and series_uid != info["series_uid"]:
        raise DatasetManagerError(
            f"series_uid '{series_uid}' does not match the provided info series '{info['series_uid']}'."
        )
    shape = info["shape"]
    if len(slices) != len(shape):
        raise DatasetManagerError(f"Expected {len(shape)} slices, got {len(slices)}.")
    normalized = tuple(slice(*item.indices(size)) for item, size in zip(slices, shape, strict=True))
    channel_indices = range(*normalized[0].indices(shape[0]))
    if list(channel_indices) not in ([0], []):
        raise DatasetManagerError("DICOM stores scalar data and supports only channel 0.")

    z_indices = list(range(*normalized[1].indices(shape[1])))
    selected_files = [info["sorted_files"][index] for index in z_indices]
    datasets = sort_series(selected_files)
    volume = read_volume(datasets, apply_rescale=apply_rescale)
    volume = volume[normalized[0], :, normalized[2], normalized[3]]

    direction_matrix = np.asarray(info["direction"], dtype=np.float64).reshape(3, 3)
    start_xyz = np.asarray([normalized[3].start, normalized[2].start, normalized[1].start], dtype=np.float64)
    spacing = np.asarray(info["spacing"], dtype=np.float64)
    origin = np.asarray(info["origin"], dtype=np.float64) + direction_matrix @ (start_xyz * spacing)
    step_xyz = np.asarray([normalized[3].step, normalized[2].step, normalized[1].step], dtype=np.float64)
    return volume, origin, spacing * step_xyz, np.asarray(info["direction"], dtype=np.float64)


def _encode_pixels(data: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Encode one scalar volume in an uncompressed integer DICOM representation."""
    if np.issubdtype(data.dtype, np.floating):
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            raise DatasetManagerError("Cannot write a DICOM volume containing no finite values.")
        minimum = float(finite.min())
        maximum = float(finite.max())
        slope = (maximum - minimum) / 65535.0 if maximum > minimum else 1.0
        intercept = minimum + 32768.0 * slope
        stored = np.rint((np.nan_to_num(data, nan=minimum) - intercept) / slope).clip(-32768, 32767).astype(np.int16)
        return stored, slope, intercept
    if np.issubdtype(data.dtype, np.signedinteger):
        return data.astype(np.int32 if data.dtype.itemsize > 2 else np.int16), 1.0, 0.0
    if np.issubdtype(data.dtype, np.unsignedinteger):
        return data.astype(np.uint32 if data.dtype.itemsize > 2 else np.uint16), 1.0, 0.0
    raise DatasetManagerError(f"Unsupported DICOM pixel dtype '{data.dtype}'.")


def write_dicom_series(
    directory: str | Path,
    volume: np.ndarray,
    *,
    origin: Sequence[float] | None = None,
    spacing: Sequence[float] | None = None,
    direction: Sequence[float] | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Write a scalar ``C-Z-Y-X`` volume as an uncompressed DICOM series."""
    _require_pydicom()
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

    data = np.asarray(volume)
    if data.ndim == 3:
        data = data[np.newaxis]
    if data.ndim != 4 or data.shape[0] != 1:
        raise DatasetManagerError(f"DICOM writing expects one scalar C-Z-Y-X channel, got shape {data.shape}.")

    stored, slope, intercept = _encode_pixels(data[0])
    origin_array = np.asarray(origin if origin is not None else [0.0, 0.0, 0.0], dtype=np.float64)
    spacing_array = np.asarray(spacing if spacing is not None else [1.0, 1.0, 1.0], dtype=np.float64)
    direction_matrix = np.asarray(
        direction if direction is not None else np.eye(3).flatten(), dtype=np.float64
    ).reshape(3, 3)
    if origin_array.shape != (3,) or spacing_array.shape != (3,):
        raise DatasetManagerError("DICOM origin and spacing must each contain exactly three values.")

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    # Remove only slices previously written by this function (its zero-padded NNNNNN.dcm
    # naming), never unrelated DICOM files that may share the directory.
    for existing in root.glob("*.dcm"):
        if _SLICE_FILENAME_RE.match(existing.name):
            existing.unlink()

    metadata = dict(metadata or {})
    study_uid = str(metadata.get("StudyInstanceUID", generate_uid()))
    series_uid = str(metadata.get("SeriesInstanceUID", generate_uid()))
    frame_uid = str(metadata.get("FrameOfReferenceUID", generate_uid()))
    now = datetime.now()
    bits = stored.dtype.itemsize * 8
    signed = bool(np.issubdtype(stored.dtype, np.signedinteger))
    row_cosine = direction_matrix[:, 0]
    column_cosine = direction_matrix[:, 1]

    for index, pixels in enumerate(stored):
        sop_uid = generate_uid()
        file_meta = FileMetaDataset()
        file_meta.FileMetaInformationVersion = b"\x00\x01"
        file_meta.MediaStorageSOPClassUID = CTImageStorage
        file_meta.MediaStorageSOPInstanceUID = sop_uid
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        file_meta.ImplementationClassUID = generate_uid()
        path = root / f"{index + 1:06d}.dcm"
        dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
        dataset.SOPClassUID = CTImageStorage
        dataset.SOPInstanceUID = sop_uid
        dataset.StudyInstanceUID = study_uid
        dataset.SeriesInstanceUID = series_uid
        dataset.FrameOfReferenceUID = frame_uid
        dataset.PatientName = str(metadata.get("PatientName", "KonfAI^Dataset"))
        dataset.PatientID = str(metadata.get("PatientID", "KonfAI"))
        dataset.Modality = str(metadata.get("Modality", "OT"))
        dataset.StudyDate = str(metadata.get("StudyDate", now.strftime("%Y%m%d")))
        dataset.StudyTime = str(metadata.get("StudyTime", now.strftime("%H%M%S")))
        dataset.SeriesNumber = int(metadata.get("SeriesNumber", 1))
        dataset.InstanceNumber = index + 1
        dataset.ImageType = ["DERIVED", "PRIMARY", "AXIAL"]
        dataset.Rows = int(pixels.shape[0])
        dataset.Columns = int(pixels.shape[1])
        dataset.SamplesPerPixel = 1
        dataset.PhotometricInterpretation = "MONOCHROME2"
        dataset.PixelRepresentation = int(signed)
        dataset.BitsAllocated = bits
        dataset.BitsStored = bits
        dataset.HighBit = bits - 1
        dataset.PixelSpacing = [float(spacing_array[1]), float(spacing_array[0])]
        dataset.SliceThickness = float(spacing_array[2])
        dataset.SpacingBetweenSlices = float(spacing_array[2])
        dataset.ImageOrientationPatient = [*row_cosine.tolist(), *column_cosine.tolist()]
        position = origin_array + direction_matrix[:, 2] * (index * spacing_array[2])
        dataset.ImagePositionPatient = position.tolist()
        dataset.RescaleSlope = float(slope)
        dataset.RescaleIntercept = float(intercept)
        dataset.PixelData = pixels.tobytes()
        dataset.save_as(str(path), enforce_file_format=True)
    return series_uid


# ---------------------------------------------------------------------------
# High-level convenience function
# ---------------------------------------------------------------------------


def read_dicom_series(
    directory: str | Path,
    *,
    series_uid: str | None = None,
    apply_rescale: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read a DICOM series from a directory into a channel-first volume.

    Parameters
    ----------
    directory:
        Path to the folder containing the DICOM series.
    series_uid:
        If the folder contains multiple series, select by SeriesInstanceUID.
        If None and only one series is present, that series is used.
        If None and multiple series are present, raises DatasetManagerError.
    apply_rescale:
        Apply RescaleSlope / RescaleIntercept (True) or keep raw integers.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        - ``volume`` — shape (1, Z, Y, X), dtype float32.
        - ``origin`` — physical origin of the first voxel, mm (shape (3,)).
        - ``spacing`` — voxel size in KonfAI/SimpleITK (x, y, z) order (shape (3,)).
        - ``direction`` — row-major 3-by-3 direction cosine matrix, flat (shape (9,)).

    Raises
    ------
    DatasetManagerError
        On missing deps, missing tags, multi-series ambiguity, or read errors.
    """
    _require_pydicom()

    _selected_uid, files = _select_series_files(directory, series_uid)
    datasets = sort_series(files)
    origin, spacing, direction = extract_geometry(datasets)
    volume = read_volume(datasets, apply_rescale=apply_rescale)
    return volume, origin, spacing, direction
