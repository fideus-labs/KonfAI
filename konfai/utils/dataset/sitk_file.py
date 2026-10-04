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


"""The one-file-per-entry backend over SimpleITK's readers and writers."""

from __future__ import annotations

import contextlib
import functools
import glob
import gzip
import os
import re
import secrets
import shutil
import struct
import warnings
import xml.etree.ElementTree as ET  # nosec B405 - the sidecar is the user's own dataset entry, same trust as lxml before
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None  # type: ignore[assignment]
from konfai.utils.dataset import decompressed
from konfai.utils.dataset.abstract import AbstractFile
from konfai.utils.dataset.attribute import (
    Attribute,
    _encode_transform_leaves,
    data_to_image,
    image_to_data,
    is_an_image,
    region_geometry,
)
from konfai.utils.dataset.landmarks import read_landmarks, write_landmarks
from konfai.utils.dataset.raw_block import (
    _mha_header,
    _nifti_extract_aborts,
    _pixel_block,
    _pixel_block_attributes,
    _pixel_block_region,
)
from konfai.utils.dataset.staging import (
    _REPLACED_MARKER,
    _recover_orphaned_backup,
    _retire_dead_debris,
    is_staging_entry,
)
from konfai.utils.dataset.stream import (
    _MHA_ELEMENT_TYPES,
    _NIFTI_DATATYPES,
    _NRRD_TYPES,
    DataStream,
    _MhaDataStream,
    _NiftiDataStream,
    _NiftiGzipStream,
    _NrrdDataStream,
)
from konfai.utils.errors import DatasetManagerError, KonfAIWarning
from konfai.utils.utils import (
    SUPPORTED_EXTENSIONS,
)

# Formats already reported by _warn_unstreamed_region_read. Keyed by format, not by file: the remedy
# is dataset-wide, so every case of a dataset would otherwise repeat the same warning.
_unstreamed_formats_warned: set[str] = set()


def _require_sitk(path: str, action: str = "read") -> None:
    """The structured refusal for a bare install, called at every SimpleITK touch point: guarding
    the backend whole would refuse the npy/fcsv/xml/vtk entries it serves with no SimpleITK at
    all."""
    if sitk is None:
        raise DatasetManagerError(
            f"SimpleITK is required to {action} '{path}'.",
            "Install it with: pip install konfai[itk] (or konfai[imaging]).",
        )


#: What ITK says when it has no image writer for the extension, or when its writer refuses the
#: volume's pixel type or dimension.
_UNWRITABLE_VOLUME = (
    "Unable to determine ImageIO writer",
    "supports unsigned",  # PNG, JPEG, BMP, TIFF
    "can only write 2-dimensional",  # JPEG
    "cannot write images with a dimension",  # BMP
    "stored pixel type was not specified",  # DICOM, a floating point volume
)
#: The formats whose ITK writer stores any volume it takes, whole: one channel or several, 2-D or 3-D.
_HOLD_ANY_VOLUME = ("mha", "mhd", "nii", "nii.gz", "nrrd", "hdr", "img")


def _mhd_pixels(header: str) -> str | None:
    """The pixel file a detached MetaImage header names, when it is the entry's own: a file beside it, named
    after it, as KonfAI writes it. A pixel file shared with other headers or kept elsewhere is not the
    entry's to remove."""
    fields = _mha_header(header) if os.path.exists(header) else None
    pixels = fields[0].get("ElementDataFile") if fields else None
    stem = os.path.basename(header)[: -len(".mhd")]
    if pixels is None or os.path.basename(pixels) != pixels or not pixels.startswith(f"{stem}."):
        return None
    return pixels


def _unwritable(image: sitk.Image, final: str, file_format: str, reason: str) -> DatasetManagerError:
    return DatasetManagerError(
        f"SimpleITK cannot write the {image.GetDimension()}-D {image.GetPixelIDTypeAsString()} volume"
        f" '{final}' as '{file_format}': {reason}",
        "Write it as mha, nrrd, h5 or omezarr, which hold any.",
    )


def _write_image(image: sitk.Image, path: str, final: str, file_format: str) -> None:
    """``sitk.WriteImage`` of the entry published as ``final``: a format with no writer for this
    volume is refused by name, not by ITK's trace naming the staging file. Any other failure (a
    permission, a full disk) keeps ITK's error.

    A file holding fewer values than the volume is refused too. ITK's GIPL writer stores one channel
    of a vector volume, its PNG writer one plane of a 3-D one, its JPEG writer three channels of
    four, and none of them raises: the header of what was written is what says so. It is read back
    for the formats that may hold less, a header read being a tenth to half of a small write."""
    try:
        sitk.WriteImage(image, path)
    except RuntimeError as error:
        message = str(error)
        if not any(refusal in message.replace(path, "") for refusal in _UNWRITABLE_VOLUME):
            raise
        # ITK's first line is its source location; the reason follows.
        reason = " ".join(message.splitlines()[1:]).replace(path, final) or message
        raise _unwritable(image, final, file_format, reason) from error
    if file_format in _HOLD_ANY_VOLUME:
        return
    written = sitk.ImageFileReader()
    written.SetFileName(path)
    try:
        written.ReadImageInformation()
    except RuntimeError:
        return  # a file ITK cannot read back says so when it is read
    held = written.GetNumberOfComponents() * int(np.prod(written.GetSize(), dtype=np.int64))
    handed = image.GetNumberOfComponentsPerPixel() * image.GetNumberOfPixels()
    if held < handed:
        raise _unwritable(
            image, final, file_format, f"the format holds {held} of its {handed} values, and ITK drops the rest."
        )


def _nifti_declared_bytes(path: str) -> int | None:
    """The uncompressed size a single-file NIfTI-1 header declares (ITK writes no NIfTI-2): where its
    pixels start plus the pixels. ``None`` when the head is short or not such a header."""
    try:
        with (gzip.open if path.endswith(".gz") else open)(path, "rb") as file:
            head = file.read(348)
    except (OSError, EOFError):
        return None
    order = next((order for order in "<>" if len(head) == 348 and struct.unpack(f"{order}i", head[:4])[0] == 348), None)
    if order is None:
        return None
    dims = struct.unpack(f"{order}8h", head[40:56])
    bitpix = struct.unpack(f"{order}h", head[72:74])[0]
    offset = int(struct.unpack(f"{order}f", head[108:112])[0])
    return offset + int(np.prod(dims[1 : dims[0] + 1], dtype=np.int64)) * bitpix // 8


def _check_nifti_written(path: str, final: str) -> None:
    """Refuse a NIfTI that holds fewer bytes than its header declares. ITK's NIfTI writer does not
    check its writes, so a full disk leaves a short file and no error. A gzip file ends on the size of
    what it compressed, modulo 2**32: the check reads its header and its last four bytes."""
    declared = _nifti_declared_bytes(path)
    if path.endswith(".gz"):
        with open(path, "rb") as file:
            file.seek(-4, os.SEEK_END)
            written = struct.unpack("<I", file.read(4))[0]
        complete = declared is not None and written == declared % 2**32
    else:
        complete = declared is not None and os.path.getsize(path) == declared
    if not complete:
        free = shutil.disk_usage(os.path.dirname(os.path.abspath(path))).free
        raise DatasetManagerError(
            f"The write of '{final}' stopped short: the file holds less than its header declares.",
            f"The disk may be full ({free / 2**30:.1f} GiB free there). Free space and run again.",
        )


def _warn_unstreamed_region_read(path: str) -> None:
    """Warn that `path`'s format decodes the whole volume for every patch region read from it.

    `warnings.warn` dedups per call site, which here is one line in a loop over every patch of every
    case: the seen-set is what makes this once per format rather than thousands of times.
    """
    suffix = Path(path).suffix
    if suffix in _unstreamed_formats_warned:
        return
    _unstreamed_formats_warned.add(suffix)
    warnings.warn(
        f"Patch-streaming '{suffix}' files (e.g. '{path}'): this format cannot serve a disk region, "
        "so every patch decodes the whole "
        "volume again: many times the cost of one read. Convert the dataset to a chunked format (OME-Zarr "
        "or HDF5), which KonfAI streams natively, or to an uncompressed .mha/.nii/.nrrd. Warned once per format.",
        KonfAIWarning,
        stacklevel=2,
    )


#: The formats a region write serves: their stream, and the dtypes it holds.
_STREAMS: dict[str, tuple[Callable[..., DataStream], dict[str, Any]]] = {
    "mha": (_MhaDataStream, _MHA_ELEMENT_TYPES),
    "nii": (_NiftiDataStream, _NIFTI_DATATYPES),
    "nii.gz": (_NiftiGzipStream, _NIFTI_DATATYPES),
    "nrrd": (_NrrdDataStream, _NRRD_TYPES),
}
#: The formats whose region writer leaves a file that reads back as the one ITK's writer leaves:
#: voxels, geometry and every attribute. A NIfTI's spells its header otherwise (no qform).
_WHOLE_BY_STREAM = ("mha", "nrrd")


class SitkFile(AbstractFile):
    # SimpleITK raises RuntimeError; the npy, fcsv and xml sidecars ValueError, EOFError or ParseError.
    read_errors = (OSError, RuntimeError, ValueError, EOFError, SyntaxError)

    def __init__(self, filename: str, read: bool, file_format: str) -> None:
        self.filename = filename
        self.read = read
        self.file_format = file_format

    @classmethod
    def open(
        cls,
        filename: str,
        read: bool,
        file_format: str,
        level: int = 0,
        scale_factors: list[int] | str | None = None,
        downsample_method: str | None = None,
    ) -> SitkFile:
        del level, scale_factors, downsample_method
        return cls(f"{filename}/", read, file_format)

    @classmethod
    def can_stream(cls, file_format: str, attributes: Attribute) -> bool:
        # An uncompressed MetaImage, NIfTI or NRRD is a fixed header plus a flat raw block, written through a
        # memmap; a gzipped NIfTI is compressed as its planes come.
        return file_format in _STREAMS and is_an_image(attributes)

    @staticmethod
    def _normalize_slices(slices: tuple[slice, ...], shape: list[int]) -> tuple[slice, ...]:
        if len(slices) != len(shape):
            raise DatasetManagerError(f"Expected {len(shape)} slices, got {len(slices)}.")

        normalized = []
        for item, size in zip(slices, shape, strict=False):
            start, stop, step = item.indices(size)
            normalized.append(slice(start, stop, step))
        return tuple(normalized)

    @staticmethod
    def _supports_direct_slice(slices: tuple[slice, ...]) -> bool:
        return all(item.step in (None, 1) for item in slices)

    @staticmethod
    @functools.cache
    def _supports_region_read(path: str) -> bool:
        """Return whether ITK can serve a region of `path` without decoding the whole volume.

        SimpleITK exposes no equivalent of ImageIOBase::CanStreamRead(), so the streaming IOs are
        mirrored here: MetaImage and NIfTI stream while their pixel data is uncompressed. A compressed
        stream is not seekable, and NrrdImageIO never streams (a NRRD is served off its raw block or
        not at all), so both decode the whole volume for every region asked of them. Getting this
        wrong only ever costs speed, never correctness.

        Cached: the patch path asks this per read, and it opens the file to read a header.
        """
        _require_sitk(path)
        if _pixel_block(path) is not None:
            return True  # a memmap of the raw block reads the region's pages and no other
        image_io = sitk.ImageFileReader.GetImageIOFromFileName(path)
        if image_io == "MetaImageIO":
            # MetaImage announces compression in its ASCII header, ahead of ElementDataFile.
            with open(path, "rb") as file:
                header = file.read(4096)
            return re.search(rb"CompressedData\s*=\s*True", header, re.IGNORECASE) is None
        if image_io == "NiftiImageIO":
            if _nifti_extract_aborts(path):
                return False
            with open(path, "rb") as file:
                return file.read(2) != b"\x1f\x8b"  # gzip magic: a .nii.gz stream
        return False

    def read_granularity(self, name: str) -> tuple[int, ...] | None:
        """A memmapped block is served BAND by band: the read maps the outermost axis the window
        spans and every axis below it whole (:func:`_mapped_band`), then copies its sub-box out
        of that. The pages a window touches are the band's, not its own, and the kernel counts
        them, so a region narrower than a plane costs a plane here exactly as a window narrower
        than a chunk costs a chunk on a chunked store.

        Measured, writing one volume with the tile forced and nothing else in the chain: a
        [58, 116, 116] block held 25 MiB over the floor against a 22.7 MiB band, a
        [200, 64, 64] one held 79 against 78.1, and a full-plane [8, 320, 320] held its own 3.
        One step along the banded axis, everything below it whole: that is what this says.

        A compressed file answers for the twin its regions are read from (:mod:`.decompressed`),
        before that twin exists. ``None`` where ITK decodes instead of mapping (a format only ITK
        reads, a compressed file whose twin cannot be written), where the whole volume is the cost
        and the streaming refusal already says so.
        """
        path = self._resolve_data_path(name)
        if path is not None:
            _require_sitk(path)
        block = (_pixel_block(path) or decompressed.servable(path)) if path is not None else None
        if block is None:
            return None
        shape = [int(extent) for extent in block.shape]
        # The order the map sees, which is the order the region read reorders into: MetaIO's
        # channel axis is the fastest, so an interleaved block is spatial-first with the channel
        # last. The band narrows the first axis of THAT order carrying more than one element,
        # and every axis after it is mapped whole. NIfTI keeps each component's volume whole,
        # one after the other, so a region touches the same band in each component volume it
        # spans: the channel axis sits above the band.
        order = [*range(1, len(shape)), 0] if block.interleaved else list(range(1, len(shape)))
        banded = next((axis for axis in order if shape[axis] > 1), order[-1])
        whole = set(order[order.index(banded) + 1 :])
        return tuple(shape[axis] if axis in whole else 1 for axis in range(len(shape)))

    def _resolve_data_path(self, name: str) -> str | None:
        base = f"{self.filename}{name}"
        memo = self.resolved_paths
        path = memo.get(base) if memo is not None else None
        if path is None:
            path = self._probe_data_path(base)
            if memo is not None and path is not None:  # an absent entry stays a fresh question
                memo[base] = path
        return path

    def _probe_data_path(self, base: str) -> str | None:
        for suffix in (".itk.txt", ".fcsv", ".xml", ".vtk", ".npy"):
            candidate = f"{base}{suffix}"
            if os.path.exists(candidate):
                return candidate

        direct = f"{base}.{self.file_format}"
        if os.path.exists(direct):
            return direct

        # Skip a crashed writer's leftover temporary (``.tmp``): it is a header plus a reserved,
        # zero-filled pixel block that would read back as a plausible partial volume. Deprioritize
        # sidecar halves of paired formats: .raw/.zraw (detached MetaImage/NRRD data, unreadable
        # standalone) and .img (readable via its paired .hdr, but prefer the header half). glob order
        # is unsorted, so a bare matches[0] could hand the .raw half of a .mhd+.raw pair to the reader.
        matches = sorted(
            (candidate for candidate in glob.glob(f"{base}.*") if not is_staging_entry(candidate)),
            key=lambda candidate: candidate.lower().endswith((".raw", ".zraw", ".img")),
        )
        return matches[0] if matches else None

    @staticmethod
    def _spans(normalized: tuple[slice, ...], shape: Sequence[int]) -> bool:
        """Whether a region is the whole volume at unit step, whatever its channels."""
        return normalized[1:] == tuple(slice(0, extent, 1) for extent in shape[1:])

    def _file_to_image_slice(self, name: str, path: str, slices: tuple[slice, ...]) -> tuple[np.ndarray, Attribute]:
        _require_sitk(path)
        found = decompressed.layout(path)
        if found is not None:
            # A region that is the whole volume decodes it once either way: it takes a twin already there
            # and makes none, which would only add a write and a read of the volume.
            whole = self._spans(self._normalize_slices(slices, list(found.shape)), found.shape)
            # Filed under the entry it is read for: a one-pass reader leaving the case removes it,
            # whichever root holds it.
            twin = decompressed.twin(path, self.case, make=not whole)
            if twin is not None:
                # A one-pass reader removes the twin as it leaves the case: a read losing that race to
                # another reader of the entry takes the compressed file, as it would without a twin.
                with contextlib.suppress(OSError, RuntimeError):
                    return self._image_region(name, twin, slices)
        return self._image_region(name, path, slices)

    def _image_region(self, name: str, path: str, slices: tuple[slice, ...]) -> tuple[np.ndarray, Attribute]:
        block = _pixel_block(path)
        if block is not None:
            # The region's bytes off the file, the same bytes ITK's streaming reader decodes
            # through its whole pipeline. The record ITK's route leaves is kept, key for key.
            normalized = self._normalize_slices(slices, list(block.shape))
            if all(item.step > 0 for item in normalized):
                try:
                    data = _pixel_block_region(block, path, normalized)
                except (OSError, ValueError):  # replaced under the stat: ITK answers for it
                    pass
                else:
                    return data, _pixel_block_attributes(block, normalized[1:])
        reader = sitk.ImageFileReader()
        reader.SetFileName(path)
        reader.ReadImageInformation()

        spatial_size_xyz = list(reader.GetSize())
        spatial_shape = list(reversed(spatial_size_xyz))
        data_shape = [reader.GetNumberOfComponents(), *spatial_shape]
        normalized = self._normalize_slices(slices, data_shape)

        if not self._supports_direct_slice(normalized) or _nifti_extract_aborts(path):
            # ITK reads the volume whole here; the record is still the REGION's, like every other
            # backend's: the volume's own geometry, then the shifted origin (and, for a step, the
            # step-scaled spacing) of the samples actually returned.
            data, attributes = self.file_to_data("", name)
            origin, spacing = region_geometry(
                attributes.get_np_array("Origin"),
                attributes.get_np_array("Spacing"),
                attributes.get_np_array("Direction"),
                normalized[1:],
            )
            attributes["Origin"] = origin
            if any(item.step != 1 for item in normalized[1:]):
                attributes["Spacing"] = spacing
            return data[normalized], attributes

        # A compressed file that got no twin was reported with the reason (decompressed._warn_untwinned).
        unstreamed = not self._supports_region_read(path) and decompressed.layout(path) is None
        if unstreamed and not self._spans(normalized, data_shape):
            _warn_unstreamed_region_read(path)

        extract_index_xyz = [item.start for item in reversed(normalized[1:])]
        extract_size_xyz = [item.stop - item.start for item in reversed(normalized[1:])]
        reader.SetExtractIndex(extract_index_xyz)
        reader.SetExtractSize(extract_size_xyz)

        image = reader.Execute()
        data, attributes = image_to_data(image)
        origin, _spacing = region_geometry(
            reader.GetOrigin(), reader.GetSpacing(), reader.GetDirection(), normalized[1:]
        )
        attributes["Origin"] = origin
        return data[normalized[:1] + tuple(slice(None) for _ in normalized[1:])], attributes

    def file_to_data(self, group: str, name: str) -> tuple[np.ndarray, Attribute]:
        path = self._resolve_data_path(name)
        if path is None:
            raise DatasetManagerError(
                f"'{name}' is not in '{self.filename}'.",
                "Check the case name and the group it is looked up under.",
            )
        attributes = Attribute()
        if path.endswith(".itk.txt"):
            _require_sitk(path)
            data = _encode_transform_leaves(sitk.ReadTransform(path), name, attributes)
        elif path.endswith(".fcsv"):
            data = cast(np.ndarray, read_landmarks(Path(path)))
        elif path.endswith(".xml"):
            with open(path, "rb") as xml_file:
                root = ET.parse(xml_file).getroot()  # nosec B314 - user-owned sidecar
            node = root
            while len(node):
                node = node[-1]
            for key, value in node.attrib.items():
                attributes[key] = value
            text = (node.text or "").strip()
            data = np.fromstring(text, sep=",", dtype=np.float64) if text else np.asarray([], dtype=np.float64)
        elif path.endswith(".vtk"):
            try:
                import vtk
            except ImportError as error:
                raise DatasetManagerError(
                    f"vtk is required to read '{path}'.", "Install it with: pip install konfai[vtk]."
                ) from error

            vtk_reader = vtk.vtkPolyDataReader()
            vtk_reader.SetFileName(path)
            vtk_reader.Update()
            points = vtk_reader.GetOutput().GetPoints()
            data = np.asarray([list(points.GetPoint(i)) for i in range(points.GetNumberOfPoints())])
        elif path.endswith(".npy"):
            data = np.load(path)
        else:
            _require_sitk(path)
            block = _pixel_block(path)
            if block is not None:
                # The volume off the file's raw block, as a region read takes its window: the voxels and
                # the record ITK's reader gives, without its buffer and the copy out of it.
                with contextlib.suppress(OSError, ValueError):  # replaced under the stat: ITK answers for it
                    whole = tuple(slice(0, extent, 1) for extent in block.shape)
                    return _pixel_block_region(block, path, whole), _pixel_block_attributes(block, None)
            image = sitk.ReadImage(path)
            data, attributes_tmp = image_to_data(image)
            attributes.update(attributes_tmp)
        return data, attributes

    def file_to_data_slice(self, group: str, name: str, slices: tuple[slice, ...]) -> tuple[np.ndarray, Attribute]:
        path = self._resolve_data_path(name)
        if path is None:
            raise DatasetManagerError(
                f"'{name}' is not in '{self.filename}'.",
                "Check the case name and the group it is looked up under.",
            )

        if path.endswith(".npy"):
            data = np.load(path, mmap_mode="r")[slices]
            return np.asarray(data), Attribute()

        if path.endswith((".itk.txt", ".fcsv", ".xml", ".vtk")):
            data, attributes = self.file_to_data(group, name)
            return data[slices], attributes

        return self._file_to_image_slice(name, path, slices)

    def bounded_region_reads(self, name: str) -> bool:
        path = self._resolve_data_path(name)
        if path is None:
            return False
        if path.endswith(".npy"):
            return True  # np.load(mmap) reads the slice off the map
        if path.endswith((".itk.txt", ".fcsv", ".xml", ".vtk")):
            return False
        # A compressed file is read by region from its uncompressed twin, decompressed once.
        return self._supports_region_read(path) or decompressed.servable(path) is not None

    def is_vtk_polydata(self, obj) -> bool:
        try:
            import vtk

            return isinstance(obj, vtk.vtkPolyData)
        except ImportError:
            return False

    def __enter__(self):
        pass

    def __exit__(self, exc_type, value, traceback):
        pass

    def data_to_file(
        self,
        name: str,
        data: sitk.Image | sitk.Transform | np.ndarray,
        attributes: Attribute | None = None,
    ) -> None:
        if attributes is None:
            attributes = Attribute()
        os.makedirs(self.filename, exist_ok=True)
        if sitk is not None and isinstance(data, sitk.Image):
            for k, v in attributes.items():
                if v and len(v):
                    data.SetMetaData(k, v)
            # Publish by rename, as the streaming writer does: an existence probe answers from disk,
            # so a reader must never meet the entry while it is being written.
            final = f"{self.filename}{name}.{self.file_format}"
            staging = DataStream.staging_path(final)
            if self.file_format in ("mhd", "hdr", "img"):
                # Header and pixels are two files, written in a staging directory and moved in, the header
                # last. A MetaImage header names its pixels, so they land under a name of their own and the
                # header swaps to them in one replace: interrupted, the entry is the old pair or the new one.
                # Analyze names its pixels after its header, so its pair has no such swap.
                while True:  # a writer killed under a reused pid may have left this very name
                    with contextlib.suppress(FileExistsError):
                        os.mkdir(staging)
                        break
                    staging = DataStream.staging_path(final)
                header = os.path.basename(final)
                written = (
                    f"{header[: -len('.mhd')]}.{secrets.token_hex(4)}.mhd" if self.file_format == "mhd" else header
                )
                old_pixels = _mhd_pixels(final) if self.file_format == "mhd" else None
                moved: list[str] = []
                try:
                    _write_image(data, os.path.join(staging, written), final, self.file_format)
                    for part in sorted(os.listdir(staging), key=lambda part: part == written):
                        target = os.path.join(os.path.dirname(final), header if part == written else part)
                        os.replace(os.path.join(staging, part), target)
                        moved.append(target)
                except BaseException:
                    if self.file_format == "mhd":  # the header never swapped: the new pixels belong to no entry
                        for target in moved:
                            with contextlib.suppress(OSError):
                                os.remove(target)
                    raise
                finally:
                    shutil.rmtree(staging, ignore_errors=True)
                if old_pixels is not None:
                    with contextlib.suppress(OSError):
                        os.remove(os.path.join(os.path.dirname(final), old_pixels))
            else:
                try:
                    _write_image(data, staging, final, self.file_format)
                    if self.file_format in ("nii", "nii.gz"):
                        _check_nifti_written(staging, final)
                except BaseException:
                    with contextlib.suppress(OSError):
                        os.remove(staging)
                    raise
                os.replace(staging, final)
            with contextlib.suppress(Exception):
                _retire_dead_debris(Path(final))  # past the publish: housekeeping cannot fail the write
        elif sitk is not None and isinstance(data, sitk.Transform):
            sitk.WriteTransform(data, f"{self.filename}{name}.itk.txt")
        elif self.is_vtk_polydata(data):
            import vtk

            vtk_writer = vtk.vtkPolyDataWriter()
            vtk_writer.SetFileName(f"{self.filename}{name}.vtk")
            vtk_writer.SetInputData(data)
            vtk_writer.Write()
        elif is_an_image(attributes):
            _require_sitk(f"{self.filename}{name}.{self.file_format}", "write")
            # SimpleITK takes a vector volume pixel by pixel, four times the cost of writing it: the
            # region writer is handed the volume whole where its file reads back as SimpleITK's.
            stream = (
                self.open_data_stream(name, list(data.shape), data.dtype, attributes)
                if isinstance(data, np.ndarray) and data.shape[0] > 1 and self.file_format in _WHOLE_BY_STREAM
                else None
            )
            if stream is None:
                self.data_to_file(name, data_to_image(data, attributes), attributes)
            else:
                with stream:
                    stream.write_slice(tuple(slice(0, extent) for extent in data.shape), data)
        elif len(data.shape) == 2 and data.shape[1] == 3 and data.shape[0] > 0:
            data = np.round(data, 4)
            write_landmarks(data, Path(f"{self.filename}{name}.fcsv"))
        elif "path" in attributes:
            if os.path.exists(f"{self.filename}{name}.xml"):
                with open(f"{self.filename}{name}.xml", "rb") as xml_file:
                    root = ET.parse(xml_file).getroot()  # nosec B314 - user-owned sidecar
            else:
                root = ET.Element(name)
            node = root
            path = attributes["path"].split(":")

            for node_name in path:
                node_tmp = node.find(node_name)
                if node_tmp is None:
                    node_tmp = ET.SubElement(node, node_name)
                node = node_tmp
            if attributes is not None:
                for attribute_tmp in attributes.keys():
                    attribute = "_".join(attribute_tmp.split("_")[:-1])
                    if attribute != "path":
                        node.set(attribute, attributes[attribute])
            if data.size > 0:
                node.text = ", ".join(map(str, data.flatten()))
            with open(f"{self.filename}{name}.xml", "wb") as f:
                # ``ET.indent`` replaces whitespace-only text/tails, so a re-read file re-indents
                # cleanly instead of accumulating blank lines.
                ET.indent(root)
                f.write(ET.tostring(root, encoding="utf-8"))
        else:
            np.save(f"{self.filename}{name}.npy", data)

    def open_data_stream(
        self,
        name: str,
        shape: list[int],
        dtype: np.dtype,
        attributes: Attribute,
        region_shape: list[int] | None = None,
    ) -> DataStream | None:
        # A format outside _STREAMS writes whole in one WriteImage call.
        if self.file_format not in _STREAMS or not is_an_image(attributes) or len(shape) < 3:
            return None
        element_dtype = np.dtype(dtype)
        if element_dtype == np.float16:
            # No streamed format has a half-float type; widen float16 to float32 (exact), as
            # data_to_image does, so streamed and whole-volume writes hold identical bytes.
            element_dtype = np.dtype(np.float32)
        dimension = len(shape) - 1
        geometry = (("Origin", dimension), ("Spacing", dimension), ("Direction", dimension * dimension))
        if any(len(attributes.get_np_array(key)) != n for key, n in geometry):
            return None
        stream, types = _STREAMS[self.file_format]
        if element_dtype.name not in types or (stream is not _MhaDataStream and dimension not in (2, 3)):
            return None
        os.makedirs(self.filename, exist_ok=True)
        return stream(f"{self.filename}{name}.{self.file_format}", shape, element_dtype, attributes)

    def is_exist(self, group: str, name: str | None = None) -> bool:
        base = f"{self.filename}{group}"
        if any(os.path.exists(base + "." + ext) for ext in SUPPORTED_EXTENSIONS):
            return True
        # A writer killed mid-replacement left the previous entry under its backup name, which
        # every listing hides: it is the entry, and it goes back under it. One listing of the folder
        # finds the backups of every extension. Then the question is asked of disk again, because
        # the recovery may have declined to a publish that landed meanwhile -- and that publish is
        # an entry too.
        folder, leaf = os.path.split(base)
        try:
            siblings = os.listdir(folder)
        except OSError:
            return False
        backed_up = {sibling.split(_REPLACED_MARKER)[0] for sibling in siblings if _REPLACED_MARKER in sibling}
        finals = [f"{base}.{ext}" for ext in SUPPORTED_EXTENSIONS if f"{leaf}.{ext}" in backed_up]
        for final in finals:
            _recover_orphaned_backup(Path(final))
        return bool(finals) and any(os.path.exists(base + "." + ext) for ext in SUPPORTED_EXTENSIONS)

    def get_infos(self, group: str, name: str) -> tuple[list[int], Attribute]:
        attributes = Attribute()
        # Resolve the actual entry path (any image extension, not only the dataset's file_format):
        # an entry stored with a different extension must still take the header-only read below --
        # the file_to_data fallback decodes the whole volume, a hidden full load on the
        # patch-planning path.
        entry = f"{group if group is not None else ''}{name}"
        path = self._resolve_data_path(entry)
        if path is not None and not path.endswith((".itk.txt", ".fcsv", ".xml", ".vtk", ".npy")):
            _require_sitk(path)
            file_reader = sitk.ImageFileReader()
            file_reader.SetFileName(path)
            file_reader.ReadImageInformation()
            attributes["Origin"] = np.asarray(file_reader.GetOrigin())
            attributes["Spacing"] = np.asarray(file_reader.GetSpacing())
            attributes["Direction"] = np.asarray(file_reader.GetDirection())
            for k in file_reader.GetMetaDataKeys():
                attributes[k] = file_reader.GetMetaData(k)
            # Reverse the spatial size for every rank (see the module-level get_infos).
            size = list(reversed(file_reader.GetSize()))
            size = [file_reader.GetNumberOfComponents(), *size]
        else:
            size = None
            if path is not None and path.endswith(".npy"):
                try:
                    size = list(np.load(path, mmap_mode="r").shape)  # the header alone, no page of the map
                except ValueError:
                    size = None  # an object array cannot be mapped: the full read answers for it
            if size is None:
                data, attributes = self.file_to_data(group if group is not None else "", name)
                size = list(data.shape)
        return size, attributes
