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


"""Bounded reads of an uncompressed MetaImage, NIfTI or NRRD: the header's pixel block, mapped band by band."""

from __future__ import annotations

import functools
import os
import struct
import threading
from collections.abc import Sequence
from typing import Any, Literal, NamedTuple

import numpy as np

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None  # type: ignore[assignment]
from konfai.utils.dataset.attribute import Attribute, _attribute_text, push_geometry, region_geometry
from konfai.utils.dataset.stream import _MHA_ELEMENT_TYPES, _NIFTI_DATATYPES

#: NumPy dtype of each element type the raw-block route reads (the inverses of the writers' tables).
_NIFTI_DTYPES = {code: np.dtype(name) for name, code in _NIFTI_DATATYPES.items()}
_MHA_DTYPES = {token: np.dtype(name) for name, token in _MHA_ELEMENT_TYPES.items()}
#: Every spelling the NRRD format gives a type, which the writers' table holds one of.
_NRRD_DTYPES = {
    token: np.dtype(name)
    for name, tokens in {
        "int8": ("int8", "int8_t", "signed char"),
        "uint8": ("uint8", "uint8_t", "uchar", "unsigned char"),
        "int16": ("int16", "int16_t", "short", "short int", "signed short", "signed short int"),
        "uint16": ("uint16", "uint16_t", "ushort", "unsigned short", "unsigned short int"),
        "int32": ("int32", "int32_t", "int", "signed int"),
        "uint32": ("uint32", "uint32_t", "uint", "unsigned int"),
        "int64": ("int64", "int64_t", "longlong", "long long", "long long int", "signed long long"),
        "uint64": ("uint64", "uint64_t", "ulonglong", "unsigned long long", "unsigned long long int"),
        "float32": ("float",),
        "float64": ("double",),
    }.items()
    for token in tokens
}
#: The kinds of a NRRD axis that runs through space or time; any other holds a voxel's components.
_NRRD_DOMAIN_KINDS = ("domain", "space", "time")
#: How much of a file a MetaImage or a NRRD header is looked for in: MetaIO writes a few hundred
#: bytes, user fields a few more; a header that runs past this is read by ITK.
_HEADER_PROBE_BYTES = 1 << 16


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
    """An uncompressed MetaImage, NIfTI or NRRD as a memmap serves it: where its raw pixels start,
    how they are stored, and what ITK reads in its header."""

    offset: int
    dtype: np.dtype  # as stored, byte order included
    interleaved: bool  # MetaIO and NRRD keep a pixel's components together; NIfTI each component's volume whole
    shape: tuple[int, ...]  # channel-first
    metadata: Attribute  # the header's own keys, as image_to_data imports them
    probe: Any  # a one-voxel sitk.Image carrying the header's geometry: ITK's own index-to-world arithmetic
    # Origin / Spacing / Direction as an attribute holds them, printed once for the file: every region
    # of a volume records the same spacing and direction, and numpy's printer costs far more than
    # handing the printed text over.
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


def _mha_header(path: str) -> tuple[dict[str, str], int] | None:
    """A MetaImage header's fields in order, up to ``ElementDataFile`` (MetaIO's last), and the offset
    its local data starts at; ``None`` when no such field is within the probe."""
    with open(path, "rb") as file:
        head = file.read(_HEADER_PROBE_BYTES)
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
    return fields, position


def _mha_raw_block(path: str) -> tuple[int, np.dtype] | None:
    """Where an uncompressed local-data MetaImage keeps its pixels and how; ``None`` for any other."""
    header = _mha_header(path)
    if header is None:
        return None
    fields, position = header
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


class _NrrdHeader(NamedTuple):
    """The header of a NRRD whose data a flat block holds: what it says and where it sits in the file."""

    encoding: str
    encoding_span: tuple[int, int]  # where the encoding is spelled, which a decompressed twin rewrites
    offset: int  # where the data starts, past the blank line
    dtype: np.dtype  # as stored, byte order included
    shape: tuple[int, ...]  # channel-first


def _nrrd_header(path: str) -> _NrrdHeader | None:
    """The header of a NRRD whose pixels sit behind it in one block, a pixel's components together.

    ``None`` for a detached header, skipped lines or bytes, a type or a byte order it does not name,
    a component axis that is not the fastest, and a header that does not end within the probe: ITK
    reads those. The encoding is the caller's to judge.
    """
    with open(path, "rb") as file:
        head = file.read(_HEADER_PROBE_BYTES)
    if not head.startswith(b"NRRD"):
        return None
    fields: dict[str, str] = {}
    encoding_span = (0, 0)
    position = 0
    while True:
        end = head.find(b"\n", position)
        if end < 0:
            return None
        line = head[position:end].rstrip(b"\r").decode("latin-1")
        if not line:
            break  # the blank line: the data follows
        # "field: value"; the magic, a comment and a "key:=value" pair are none of the header's fields.
        key, separator, value = line.partition(": ")
        if separator and not key.startswith("#") and ":=" not in key:
            fields[key.strip().lower()] = value.strip()
            if key.strip().lower() == "encoding":
                encoding_span = (position + len(key) + len(separator), position + len(line))
        position = end + 1
    dtype = _NRRD_DTYPES.get(fields.get("type", ""))
    try:
        sizes = [int(extent) for extent in fields["sizes"].split()]
    except (KeyError, ValueError):
        return None
    kinds = fields.get("kinds", "").split() or ["domain"] * len(sizes)
    directions = fields.get("space directions", "").split() or ["("] * len(sizes)
    if (
        dtype is None
        or len(kinds) != len(sizes)
        or len(directions) != len(sizes)
        or any(key in fields for key in ("data file", "datafile"))
        or any(fields.get(key, "0") != "0" for key in ("line skip", "lineskip", "byte skip", "byteskip"))
    ):
        return None
    orders: dict[str, Literal["<", ">"]] = {"little": "<", "big": ">"}
    order = orders.get(fields.get("endian", "little" if dtype.itemsize == 1 else ""))
    # The axis holding a voxel's components, if there is one, has to be the fastest: the block is
    # then a pixel's components together, as MetaIO stores them.
    components = [
        axis for axis in range(len(sizes)) if kinds[axis] not in _NRRD_DOMAIN_KINDS or directions[axis] == "none"
    ]
    if order is None or components not in ([], [0]):
        return None
    channels, spatial = (sizes[0], sizes[1:]) if components else (1, sizes)
    shape = (channels, *reversed(spatial))
    return _NrrdHeader(fields.get("encoding", ""), encoding_span, end + 1, dtype.newbyteorder(order), shape)


def _nifti_raw_block(path: str) -> tuple[int, np.dtype] | None:
    """Where a single-file uncompressed NIfTI-1 keeps its pixels and how; ``None`` for any other.

    A stored intensity scaling (``scl_slope``/``scl_inter``) is left to ITK, which applies it and
    promotes the pixel type; the block's bytes are then not the volume's values.
    """
    with open(path, "rb") as file:
        header = file.read(348)
    if len(header) < 348 or header[344:348] != b"n+1\x00":  # a .hdr/.img pair keeps its block elsewhere
        return None
    orders: tuple[Literal["<"], Literal[">"]] = ("<", ">")
    order = next((order for order in orders if struct.unpack(f"{order}i", header[:4])[0] == 348), None)
    if order is None:
        return None
    dtype = _NIFTI_DTYPES.get(struct.unpack(f"{order}h", header[70:72])[0])
    (vox_offset,) = struct.unpack(f"{order}f", header[108:112])
    slope, inter = struct.unpack(f"{order}2f", header[112:120])
    if dtype is None or slope not in (0.0, 1.0) or inter != 0.0 or vox_offset < 352 or vox_offset != int(vox_offset):
        return None
    return int(vox_offset), dtype.newbyteorder(order)


#: ITK's header read and its one-voxel extract are not safe from several threads asking their
#: first at once, as a cache fill reading its cases on a thread pool does: one thread at a time
#: qualifies a file. Each file is asked once, so the reads themselves stay concurrent.
_QUALIFYING = threading.Lock()


@functools.lru_cache(maxsize=4096)
def _pixel_block_at(path: str, stamp: tuple[int, int]) -> _PixelBlock | None:
    """The raw block of ``path`` as it was at ``stamp``, with its header read by ITK once."""
    del stamp  # part of the key: a rewritten file gets a record of its own
    with _QUALIFYING:
        return _qualified_block(path)


def _qualified_block(path: str) -> _PixelBlock | None:
    """The raw block of ``path``, or ``None`` when only ITK can read the file.

    Qualified against ITK's own reading of the header: the element type it reports must be the one
    stored, and the file must hold every element the shape announces. A file that fails either is
    read by ITK, which then answers for it, so a mismatch costs speed and never a wrong value.

    The geometry and the metadata are taken off a one-voxel region ITK extracts, not off its header
    read alone: ITK's NIfTI IO reads the header again before it reads pixels, and the direction it
    then carries differs from the first read's in the sign of its zeros, which the record keeps as
    text. Two files take their record from the header read: a vector NIfTI, which ITK cannot extract
    a region of, and a NRRD, which ITK reads whole for one voxel of it.
    """
    image_io = sitk.ImageFileReader.GetImageIOFromFileName(path)
    announced = None  # the shape a NRRD header orders its axes into, which ITK has to read the same
    if image_io == "MetaImageIO":
        raw, interleaved = _mha_raw_block(path), True
    elif image_io == "NiftiImageIO":
        raw, interleaved = _nifti_raw_block(path), False
    elif image_io == "NrrdImageIO":
        nrrd = _nrrd_header(path)
        if nrrd is None or nrrd.encoding != "raw":
            return None
        raw, interleaved, announced = (nrrd.offset, nrrd.dtype), True, nrrd.shape
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
        or announced not in (None, shape)
    ):
        return None
    if image_io == "NrrdImageIO" or (not interleaved and shape[0] > 1):
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


def _pixel_block_attributes(block: _PixelBlock, spatial_slices: tuple[slice, ...] | None) -> Attribute:
    """The attributes ITK's route records for the region ``spatial_slices`` keeps (``None`` is the
    whole volume's record, as ``file_to_data`` returns it).

    A unit-step region carries the record ITK's extract leaves: the region's origin as ITK computes
    it, then the region's origin again as :func:`region_geometry` computes it. A stepped region
    starts from the volume's record (what the ITK route reads, since ITK cannot extract a step) and
    appends the region's shifted origin and step-scaled spacing, so the two routes stay
    key-for-key identical.
    """
    attributes = Attribute(block.metadata)
    stepped = spatial_slices is not None and any(item.step != 1 for item in spatial_slices)
    if spatial_slices is None or stepped:
        push_geometry(attributes, "Origin", block.geometry_text["Origin"])
    else:
        index_xyz = [item.start for item in reversed(spatial_slices)]
        push_geometry(attributes, "Origin", np.asarray(block.probe.TransformIndexToPhysicalPoint(index_xyz)))
    push_geometry(attributes, "Spacing", block.geometry_text["Spacing"])
    push_geometry(attributes, "Direction", block.geometry_text["Direction"])
    if spatial_slices is not None:
        origin, spacing = region_geometry(block.origin, block.spacing, block.direction, spatial_slices)
        attributes["Origin"] = origin
        if stepped:
            attributes["Spacing"] = spacing
    return attributes


def _nifti_extract_aborts(path: str) -> bool:
    """Whether an ITK region read of ``path`` would take the process down.

    ITK's NIfTI IO extracts a region of a SCALAR image only. Asked for a region of a vector one (a
    multi-channel .nii or .nii.gz) it frees a buffer twice and aborts the process -- ``double free
    or corruption``, no exception, nothing to catch (measured with the SimpleITK this ships with,
    compressed or not). Such a file is read whole and sliced here.
    """
    info = os.stat(path)
    return _nifti_extract_aborts_at(path, (info.st_mtime_ns, info.st_size))


@functools.lru_cache(maxsize=4096)
def _nifti_extract_aborts_at(path: str, stamp: tuple[int, int]) -> bool:
    del stamp  # part of the key: a file rewritten with another channel count gets its own answer
    if sitk.ImageFileReader.GetImageIOFromFileName(path) != "NiftiImageIO":
        return False
    reader = sitk.ImageFileReader()
    reader.SetFileName(path)
    reader.ReadImageInformation()
    return reader.GetNumberOfComponents() > 1
