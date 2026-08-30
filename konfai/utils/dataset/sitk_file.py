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
import os
import re
import warnings
from pathlib import Path

import numpy as np
from lxml import etree  # nosec B410

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None  # type: ignore[assignment]
from konfai.utils.dataset.abstract import AbstractFile
from konfai.utils.dataset.attribute import (
    Attribute,
    _encode_transform_leaves,
    data_to_image,
    image_to_data,
    is_an_image,
)
from konfai.utils.dataset.landmarks import read_landmarks, write_landmarks
from konfai.utils.dataset.raw_block import (
    _nifti_extract_aborts,
    _pixel_block,
    _pixel_block_attributes,
    _pixel_block_region,
)
from konfai.utils.dataset.staging import _recover_orphaned_backup, _retire_dead_debris, is_staging_entry
from konfai.utils.dataset.stream import (
    _MHA_ELEMENT_TYPES,
    _NIFTI_DATATYPES,
    DataStream,
    _MhaDataStream,
    _NiftiDataStream,
)
from konfai.utils.errors import DatasetManagerError
from konfai.utils.utils import (
    SUPPORTED_EXTENSIONS,
)

# Formats already reported by _warn_unstreamed_region_read. Keyed by format, not by file: the remedy
# is dataset-wide, so every case of a dataset would otherwise repeat the same warning.
_unstreamed_formats_warned: set[str] = set()


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
        f"Patch-streaming '{suffix}' files (e.g. '{path}'): this format cannot serve a disk region "
        "(NRRD, or any compressed file), so every patch decodes the whole volume again: many times "
        "the cost of one read. Convert the dataset to a chunked format (OME-Zarr or HDF5), which KonfAI "
        "streams natively, or to an uncompressed .mha/.nii. Warned once per format.",
        stacklevel=2,
    )


class SitkFile(AbstractFile):
    def __init__(self, filename: str, read: bool, file_format: str) -> None:
        self.filename = filename
        self.read = read
        self.file_format = file_format

    @staticmethod
    def _normalize_slices(slices: tuple[slice, ...], shape: list[int]) -> tuple[slice, ...]:
        if len(slices) != len(shape):
            raise ValueError(f"Expected {len(shape)} slices, got {len(slices)}.")

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
        stream is not seekable, and NrrdImageIO never streams, so both decode the whole volume for
        every region asked of them. Getting this wrong only ever costs speed, never correctness.

        Cached: the patch path asks this per read, and it opens the file to read a header.
        """
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

        ``None`` where ITK decodes instead of mapping (a compressed stream), where the whole
        volume is the cost and the streaming refusal already says so.
        """
        path = self._resolve_data_path(name)
        block = _pixel_block(path) if path is not None else None
        if block is None:
            return None
        shape = [int(extent) for extent in block.shape]
        # The order the map sees, which is the order the region read reorders into: MetaIO's
        # channel axis is the fastest, so an interleaved block is spatial-first with the channel
        # last. The band narrows the first axis of THAT order carrying more than one element,
        # and every axis after it is mapped whole.
        order = [*range(1, len(shape)), 0] if block.interleaved else list(range(len(shape)))
        banded = next((axis for axis in order if shape[axis] > 1), order[-1])
        whole = set(order[order.index(banded) + 1 :])
        return tuple(shape[axis] if axis in whole else 1 for axis in range(len(shape)))

    def _resolve_data_path(self, name: str) -> str | None:
        base = f"{self.filename}{name}"
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

    def _file_to_image_slice(self, name: str, path: str, slices: tuple[slice, ...]) -> tuple[np.ndarray, Attribute]:
        block = _pixel_block(path)
        if block is not None:
            # The region's bytes off the file, where ITK's streaming reader decodes them through
            # its pipeline: 3.5 ms against 0.09 ms for a 64^3 region of an uncompressed 256^3
            # .mha, the same bytes. The record ITK's route leaves is kept: the region's origin
            # for a direct slice, the volume's for a stepped one, which ITK reads whole.
            normalized = self._normalize_slices(slices, list(block.shape))
            if all(item.step > 0 for item in normalized):
                try:
                    data = _pixel_block_region(block, path, normalized)
                except (OSError, ValueError):  # replaced under the stat: ITK answers for it
                    pass
                else:
                    index_xyz = [item.start for item in reversed(normalized[1:])]
                    direct = self._supports_direct_slice(normalized)
                    return data, _pixel_block_attributes(block, index_xyz if direct else None)
        reader = sitk.ImageFileReader()
        reader.SetFileName(path)
        reader.ReadImageInformation()

        spatial_size_xyz = list(reader.GetSize())
        spatial_shape = list(reversed(spatial_size_xyz))
        data_shape = [reader.GetNumberOfComponents(), *spatial_shape]
        normalized = self._normalize_slices(slices, data_shape)

        if not self._supports_direct_slice(normalized) or _nifti_extract_aborts(path):
            data, attributes = self.file_to_data("", name)
            return data[normalized], attributes

        if not self._supports_region_read(path):
            _warn_unstreamed_region_read(path)

        extract_index_xyz = [item.start for item in reversed(normalized[1:])]
        extract_size_xyz = [item.stop - item.start for item in reversed(normalized[1:])]
        reader.SetExtractIndex(extract_index_xyz)
        reader.SetExtractSize(extract_size_xyz)

        image = reader.Execute()
        data, attributes = image_to_data(image)
        origin = np.asarray(reader.GetOrigin(), dtype=np.float64)
        spacing = np.asarray(reader.GetSpacing(), dtype=np.float64)
        direction = np.asarray(reader.GetDirection(), dtype=np.float64).reshape(len(spacing), len(spacing))
        attributes["Origin"] = origin + direction @ (np.asarray(extract_index_xyz, dtype=np.float64) * spacing)
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
            datas = _encode_transform_leaves(sitk.ReadTransform(path), name, attributes)
            max_len = max(len(v) for v in datas)
            data = np.array([np.pad(v, (0, max_len - len(v)), constant_values=np.nan) for v in datas])
        elif path.endswith(".fcsv"):
            data = read_landmarks(Path(path))
        elif path.endswith(".xml"):
            with open(path, "rb") as xml_file:
                root = etree.parse(xml_file, etree.XMLParser(remove_blank_text=True)).getroot()  # nosec B320
            node = root
            while len(node):
                node = node[-1]
            for key, value in node.attrib.items():
                attributes[key] = value
            text = (node.text or "").strip()
            data = np.fromstring(text, sep=",", dtype=np.float64) if text else np.asarray([], dtype=np.float64)
        elif path.endswith(".vtk"):
            import vtk

            vtk_reader = vtk.vtkPolyDataReader()
            vtk_reader.SetFileName(path)
            vtk_reader.Update()
            data = []
            points = vtk_reader.GetOutput().GetPoints()
            num_points = points.GetNumberOfPoints()
            for i in range(num_points):
                data.append(list(points.GetPoint(i)))
            data = np.asarray(data)
        elif path.endswith(".npy"):
            data = np.load(path)
        else:
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
        return not path.endswith((".itk.txt", ".fcsv", ".xml", ".vtk")) and self._supports_region_read(path)

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
        if isinstance(data, sitk.Image):
            for k, v in attributes.items():
                if v and len(v):
                    data.SetMetaData(k, v)
            # Publish by rename, as the streaming writer does: an existence probe answers from disk,
            # so a reader must never meet the entry while it is being written.
            final = f"{self.filename}{name}.{self.file_format}"
            staging = DataStream.staging_path(final)
            sitk.WriteImage(data, staging)
            os.replace(staging, final)
            with contextlib.suppress(Exception):
                _retire_dead_debris(Path(final))  # past the publish: housekeeping cannot fail the write
        elif isinstance(data, sitk.Transform):
            sitk.WriteTransform(data, f"{self.filename}{name}.itk.txt")
        elif self.is_vtk_polydata(data):
            import vtk

            vtk_writer = vtk.vtkPolyDataWriter()
            vtk_writer.SetFileName(f"{self.filename}{name}.vtk")
            vtk_writer.SetInputData(data)
            vtk_writer.Write()
        elif is_an_image(attributes):
            self.data_to_file(name, data_to_image(data, attributes), attributes)
        elif len(data.shape) == 2 and data.shape[1] == 3 and data.shape[0] > 0:
            data = np.round(data, 4)
            write_landmarks(data, Path(f"{self.filename}{name}.fcsv"))
        elif "path" in attributes:
            if os.path.exists(f"{self.filename}{name}.xml"):
                with open(f"{self.filename}{name}.xml", "rb") as xml_file:
                    root = etree.parse(xml_file, etree.XMLParser(remove_blank_text=True)).getroot()  # nosec B320
                    xml_file.close()
            else:
                root = etree.Element(name)
            node = root
            path = attributes["path"].split(":")

            for node_name in path:
                node_tmp = node.find(node_name)
                if node_tmp is None:
                    node_tmp = etree.SubElement(node, node_name)
                    node.append(node_tmp)
                node = node_tmp
            if attributes is not None:
                for attribute_tmp in attributes.keys():
                    attribute = "_".join(attribute_tmp.split("_")[:-1])
                    if attribute != "path":
                        node.set(attribute, attributes[attribute])
            if data.size > 0:
                node.text = ", ".join(map(str, data.flatten()))
            with open(f"{self.filename}{name}.xml", "wb") as f:
                f.write(etree.tostring(root, pretty_print=True, encoding="utf-8"))
                f.close()
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
        # The region-writable SimpleITK formats are the region-READABLE ones, deliberately:
        # uncompressed MetaImage and NIfTI are a fixed header plus a flat raw block, so the block
        # is reserved and memmapped. Every other format writes whole in one WriteImage call --
        # and streaming into a form the reader must then decode whole would only move the cost.
        if self.file_format not in ("mha", "nii") or not is_an_image(attributes) or len(shape) < 3:
            return None
        element_dtype = np.dtype(dtype)
        if element_dtype == np.float16:
            # Neither format has a half-float type; widen float16 to float32 (exact), as
            # data_to_image does, so streamed and whole-volume writes hold identical bytes.
            element_dtype = np.dtype(np.float32)
        dimension = len(shape) - 1
        geometry = (("Origin", dimension), ("Spacing", dimension), ("Direction", dimension * dimension))
        if any(len(attributes.get_np_array(key)) != n for key, n in geometry):
            return None
        if self.file_format == "nii":
            if dimension not in (2, 3) or element_dtype.name not in _NIFTI_DATATYPES:
                return None
            os.makedirs(self.filename, exist_ok=True)
            return _NiftiDataStream(f"{self.filename}{name}.{self.file_format}", shape, element_dtype, attributes)
        if element_dtype.name not in _MHA_ELEMENT_TYPES:
            return None
        os.makedirs(self.filename, exist_ok=True)
        return _MhaDataStream(f"{self.filename}{name}.{self.file_format}", shape, element_dtype, attributes)

    def is_exist(self, group: str, name: str | None = None) -> bool:
        base = f"{self.filename}{group}"
        if any(os.path.exists(base + "." + ext) for ext in SUPPORTED_EXTENSIONS):
            return True
        # A writer killed mid-replacement left the previous entry under its backup name, which
        # every listing hides: it is the entry, and it goes back under it. Then the question is
        # asked of disk again, because the recovery may have declined to a publish that landed
        # meanwhile -- and that publish is an entry too.
        for ext in SUPPORTED_EXTENSIONS:
            _recover_orphaned_backup(Path(f"{base}.{ext}"))
        return any(os.path.exists(base + "." + ext) for ext in SUPPORTED_EXTENSIONS)

    def get_names(self, group: str) -> list[str]:
        raise NotImplementedError()

    def get_group(self) -> list[str]:
        raise NotImplementedError()

    def get_infos(self, group: str, name: str) -> tuple[list[int], Attribute]:
        attributes = Attribute()
        # Resolve the actual entry path (any image extension, not only the dataset's file_format):
        # an entry stored with a different extension must still take the header-only read below --
        # the file_to_data fallback decodes the whole volume, a hidden full load on the
        # patch-planning path.
        entry = f"{group if group is not None else ''}{name}"
        path = self._resolve_data_path(entry)
        if path is not None and not path.endswith((".itk.txt", ".fcsv", ".xml", ".vtk", ".npy")):
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
