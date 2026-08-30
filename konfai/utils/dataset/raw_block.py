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


"""Bounded reads of an uncompressed MetaImage or NIfTI: the header's pixel block, mapped band by band."""

from __future__ import annotations

import functools
import os
import struct
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NamedTuple, TypeVar

import numpy as np

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None  # type: ignore[assignment]
from konfai.utils.dataset.attribute import Attribute, _attribute_text
from konfai.utils.dataset.stream import _MHA_ELEMENT_TYPES, _NIFTI_DATATYPES

_T = TypeVar("_T")

#: NumPy dtype of each element type the raw-block route reads (the inverses of the writers' tables).
_NIFTI_DTYPES = {code: np.dtype(name) for name, code in _NIFTI_DATATYPES.items()}
_MHA_DTYPES = {token: np.dtype(name) for name, token in _MHA_ELEMENT_TYPES.items()}
#: How much of a file a MetaImage header is looked for in: MetaIO writes a few hundred bytes, user
#: fields a few more; a header that runs past this is read by ITK.
_MHA_HEADER_PROBE_BYTES = 1 << 16


@functools.cache
def _sitk_component_dtypes() -> dict[int, np.dtype]:
    """The NumPy dtype ITK stores each of its scalar and vector pixel types in."""
    kinds = (
        ("UInt8", np.uint8),
        ("Int8", np.int8),
        ("UInt16", np.uint16),
        ("Int16", np.int16),
        ("UInt32", np.uint32),
        ("Int32", np.int32),
        ("UInt64", np.uint64),
        ("Int64", np.int64),
        ("Float32", np.float32),
        ("Float64", np.float64),
    )
    return {getattr(sitk, f"sitk{prefix}{name}"): np.dtype(kind) for prefix in ("", "Vector") for name, kind in kinds}


class _PixelBlock(NamedTuple):
    """An uncompressed MetaImage or NIfTI as a memmap serves it: where its raw pixels start, how
    they are stored, and what ITK reads in its header."""

    offset: int
    dtype: np.dtype  # as stored, byte order included
    interleaved: bool  # MetaIO keeps a pixel's components together; NIfTI keeps each component's volume whole
    shape: tuple[int, ...]  # channel-first
    metadata: Attribute  # the header's own keys, as image_to_data imports them
    probe: Any  # a one-voxel sitk.Image carrying the header's geometry: ITK's own index-to-world arithmetic
    # Origin / Spacing / Direction as an attribute holds them, printed once for the file: every region
    # of a volume records the same spacing and direction, and printing a float array costs 24 us for
    # three elements and 30 us for nine (measured), against 0.04 us to hand text through the same door.
    geometry_text: dict[str, str]

    @property
    def origin(self) -> np.ndarray:
        return np.asarray(self.probe.GetOrigin())

    @property
    def spacing(self) -> np.ndarray:
        return np.asarray(self.probe.GetSpacing())

    @property
    def direction(self) -> np.ndarray:
        return np.asarray(self.probe.GetDirection())


def _mha_raw_block(path: str) -> tuple[int, np.dtype] | None:
    """Where an uncompressed local-data MetaImage keeps its pixels and how; ``None`` for any other."""
    with open(path, "rb") as file:
        head = file.read(_MHA_HEADER_PROBE_BYTES)
    fields: dict[str, str] = {}
    position = 0
    while "ElementDataFile" not in fields:
        end = head.find(b"\n", position)
        if end < 0:
            return None
        key, separator, value = head[position:end].decode("latin-1").partition("=")
        position = end + 1
        if separator:
            fields[key.strip()] = value.strip()
    dtype = _MHA_DTYPES.get(fields.get("ElementType", ""))
    if (
        dtype is None
        or fields["ElementDataFile"] != "LOCAL"
        or "HeaderSize" in fields  # a seek MetaIO applies to LOCAL data too
        or fields.get("BinaryData", "").lower() != "true"
        or fields.get("CompressedData", "false").lower() != "false"
    ):
        return None
    big_endian = fields.get("BinaryDataByteOrderMSB", fields.get("ElementByteOrderMSB", "false")).lower() == "true"
    return position, dtype.newbyteorder(">" if big_endian else "<")


def _nifti_raw_block(path: str) -> tuple[int, np.dtype] | None:
    """Where a single-file uncompressed NIfTI-1 keeps its pixels and how; ``None`` for any other.

    A stored intensity scaling (``scl_slope``/``scl_inter``) is left to ITK, which applies it and
    promotes the pixel type; the block's bytes are then not the volume's values.
    """
    with open(path, "rb") as file:
        header = file.read(348)
    if len(header) < 348 or header[344:348] != b"n+1\x00":  # a .hdr/.img pair keeps its block elsewhere
        return None
    order = next((order for order in ("<", ">") if struct.unpack(f"{order}i", header[:4])[0] == 348), None)
    if order is None:
        return None
    dtype = _NIFTI_DTYPES.get(struct.unpack(f"{order}h", header[70:72])[0])
    (vox_offset,) = struct.unpack(f"{order}f", header[108:112])
    slope, inter = struct.unpack(f"{order}2f", header[112:120])
    if dtype is None or slope not in (0.0, 1.0) or inter != 0.0 or vox_offset < 352 or vox_offset != int(vox_offset):
        return None
    return int(vox_offset), dtype.newbyteorder(order)


@functools.lru_cache(maxsize=4096)
def _pixel_block_at(path: str, stamp: tuple[int, int]) -> _PixelBlock | None:
    """The raw block of ``path`` as it was at ``stamp``, with its header read by ITK once.

    Qualified against ITK's own reading of the header: the element type it reports must be the one
    stored, and the file must hold every element the shape announces. A file that fails either is
    read by ITK, which then answers for it, so a mismatch costs speed and never a wrong value.

    The geometry and the metadata are taken off a one-voxel region ITK extracts, not off its header
    read alone: ITK's NIfTI IO reads the header again before it reads pixels, and the direction it
    then carries differs from the first read's in the sign of its zeros, which the record keeps as
    text. A vector NIfTI, which ITK cannot extract a region of, is the one file whose record comes
    from the header read.
    """
    del stamp  # part of the key: a rewritten file gets a record of its own
    image_io = sitk.ImageFileReader.GetImageIOFromFileName(path)
    if image_io == "MetaImageIO":
        raw, interleaved = _mha_raw_block(path), True
    elif image_io == "NiftiImageIO":
        raw, interleaved = _nifti_raw_block(path), False
    else:
        return None
    if raw is None:
        return None
    offset, dtype = raw
    reader = sitk.ImageFileReader()
    reader.SetFileName(path)
    reader.ReadImageInformation()
    rank = reader.GetDimension()
    shape = (reader.GetNumberOfComponents(), *reversed(reader.GetSize()))
    if (
        rank not in (2, 3)
        or _sitk_component_dtypes().get(reader.GetPixelID()) != dtype.newbyteorder("=")
        or os.path.getsize(path) < offset + int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    ):
        return None
    if not interleaved and shape[0] > 1:
        probe = sitk.Image([1] * rank, sitk.sitkUInt8)
        probe.SetOrigin(reader.GetOrigin())
        probe.SetSpacing(reader.GetSpacing())
        probe.SetDirection(reader.GetDirection())
        for key in reader.GetMetaDataKeys():
            probe.SetMetaData(key, reader.GetMetaData(key))
    else:
        reader.SetExtractIndex([0] * rank)
        reader.SetExtractSize([1] * rank)
        probe = reader.Execute()
    metadata = Attribute()
    for key in probe.GetMetaDataKeys():
        if not key.startswith("ITK_"):  # the reader's own bookkeeping, as image_to_data drops it
            metadata[key] = probe.GetMetaData(key)
    geometry_text = {
        "Origin": _attribute_text(np.asarray(probe.GetOrigin())),
        "Spacing": _attribute_text(np.asarray(probe.GetSpacing())),
        "Direction": _attribute_text(np.asarray(probe.GetDirection())),
    }
    return _PixelBlock(offset, dtype, interleaved, shape, metadata, probe, geometry_text)


def _pixel_block(path: str) -> _PixelBlock | None:
    """The raw block of ``path`` as it is now, or ``None`` when only ITK can read the file."""
    try:
        info = os.stat(path)
    except OSError:
        return None
    return _pixel_block_at(path, (info.st_mtime_ns, info.st_size))


def _mapped_band(path: str, dtype: np.dtype, offset: int, shape: Sequence[int], axis: int, index: slice) -> np.ndarray:
    """The raw block mapped over ``index`` of ``axis`` alone, the axes above it holding one element
    each, indexed as the block is.

    Mapping the block whole makes the address space a run needs follow the file's size rather than
    the budget its regions were sized against: 400 MiB of peak address space for a 128 MiB budget on
    a 156 MiB source against 256 MiB on the same run at 78 MiB. Only the outermost axis holding more
    than one element narrows the map: a region spanning it whole reaches from the block's first
    plane to its last whatever the axes below select.
    """
    plane = int(np.prod(shape[axis + 1 :], dtype=np.int64)) * dtype.itemsize
    start = int(index.start or 0)
    rows = (int(shape[axis]) if index.stop is None else int(index.stop)) - start
    return np.memmap(path, dtype, "r", offset + start * plane, (*shape[:axis], rows, *shape[axis + 1 :]))


def _pixel_block_region(block: _PixelBlock, path: str, normalized: tuple[slice, ...]) -> np.ndarray:
    """One region off the raw block, channel-first and in the native byte order: the bytes ITK
    would decode, read through a memmap that touches the region's pages and no other.

    A copy, always: a slab of whole planes is contiguous on the map, and an array that only
    guaranteed contiguity would be the map's own pages, read-only and unmapped with the map."""
    if block.interleaved:
        # MetaIO's channel axis is the fastest, so the block is spatial-first.
        shape, index = [*block.shape[1:], block.shape[0]], [*normalized[1:], normalized[0]]
    else:
        shape, index = list(block.shape), list(normalized)
    axis = next((k for k, extent in enumerate(shape) if extent > 1), 0)
    mapped = _mapped_band(path, block.dtype, block.offset, shape, axis, index[axis])
    # The band starts where its own slice does, so of that slice only the step still reads on it.
    region = mapped[(*index[:axis], slice(None, None, index[axis].step), *index[axis + 1 :])]
    if block.interleaved:
        region = np.moveaxis(region, -1, 0)
    return np.array(region, dtype=block.dtype.newbyteorder("="), order="C")


def _pixel_block_attributes(block: _PixelBlock, index_xyz: list[int] | None) -> Attribute:
    """The attributes ITK's route records: the header's keys, then the geometry, the origin being
    the region's (at ``index_xyz``) as ITK's extract computes it, then the region's origin again as
    the module computes it. ``None`` is the whole volume's record, as ``file_to_data`` returns it."""
    attributes = Attribute(block.metadata)
    if index_xyz is None:
        attributes["Origin"] = block.geometry_text["Origin"]
    else:
        attributes["Origin"] = np.asarray(block.probe.TransformIndexToPhysicalPoint(index_xyz))
    attributes["Spacing"] = block.geometry_text["Spacing"]
    attributes["Direction"] = block.geometry_text["Direction"]
    if index_xyz is not None:
        direction = block.direction.reshape(len(block.spacing), len(block.spacing))
        attributes["Origin"] = block.origin + direction @ (np.asarray(index_xyz, dtype=np.float64) * block.spacing)
    return attributes


@functools.cache
def _nifti_extract_aborts(path: str) -> bool:
    """Whether an ITK region read of ``path`` would take the process down.

    ITK's NIfTI IO extracts a region of a SCALAR image only. Asked for a region of a vector one (a
    multi-channel .nii or .nii.gz) it frees a buffer twice and aborts the process -- ``double free
    or corruption``, no exception, nothing to catch (measured with the SimpleITK this ships with,
    compressed or not). Such a file is read whole and sliced here.
    """
    if sitk.ImageFileReader.GetImageIOFromFileName(path) != "NiftiImageIO":
        return False
    reader = sitk.ImageFileReader()
    reader.SetFileName(path)
    reader.ReadImageInformation()
    return reader.GetNumberOfComponents() > 1


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


def _store_chunks(shape: list[int], region_shape: list[int] | None, dtype: Any) -> tuple[int, ...] | None:
    """Chunks a store should use, given the region shape its writer declared.

    A region write that straddles a chunk becomes a read-modify-write of it, so the writer's own
    region is the starting point; verbatim it is a gigabyte in one chunk at 2048x2048 float32, paid
    by every later partial read. A region that fits ``CHUNK_TARGET_BYTES`` is taken as it stands; one
    that does not is cut on EVERY axis longer than ``CHUNK_SPATIAL_TILE`` at once, the shape that
    writes fastest (2.4 GB into a (1, 128, 128, 128) uint16 store takes 2.18 s, into
    (1, 128, 640, 128) 3.53 s).

    A covered axis may be cut anywhere; a partial one only into a DIVISOR of the region, since a
    writer advancing in blocks of its declared size starts every block at a multiple of it. One whose
    largest usable divisor would be a sliver is left long. ``None`` when the writer declared nothing.
    """
    from konfai.utils.ome_zarr import CHUNK_SPATIAL_TILE, CHUNK_TARGET_BYTES

    if region_shape is None or len(region_shape) != len(shape):
        return None
    chunk = [max(1, min(int(region), int(extent))) for region, extent in zip(region_shape, shape, strict=True)]
    itemsize = max(1, np.dtype(dtype).itemsize)
    if int(np.prod(chunk, dtype=np.int64)) * itemsize <= CHUNK_TARGET_BYTES:
        return tuple(chunk)
    return tuple(
        min(extent, CHUNK_SPATIAL_TILE) if extent >= int(shape[axis]) else _divisor_tile(extent, CHUNK_SPATIAL_TILE)
        for axis, extent in enumerate(chunk)
    )


def _divisor_tile(extent: int, cap: int) -> int:
    """The largest divisor of ``extent`` that is at most ``cap``, or ``extent`` when that divisor
    would be a sliver (under a quarter of the cap): a chunk axis of one voxel is worse than a long
    one."""
    if extent <= cap:
        return max(1, extent)
    divisor = next((candidate for candidate in range(cap, 0, -1) if extent % candidate == 0), 1)
    return divisor if divisor * 4 >= cap else extent
