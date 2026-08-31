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


"""Region writes into an entry being published: the contract and the raw-block streams (MetaImage, NIfTI)."""

from __future__ import annotations

import contextlib
import itertools
import mmap
import os
import struct
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from konfai.utils.dataset.staging import _retire_dead_debris

if TYPE_CHECKING:
    from konfai.utils.dataset.attribute import Attribute
    from konfai.utils.dataset.backend import File


class DataStream(ABC):
    """One dataset entry written incrementally, region by region. Obtained from
    ``Dataset.open_data_stream``, which returns ``None`` when the write format cannot serve region writes
    (the caller then assembles the volume and uses ``Dataset.write``). Use as a context manager: a clean
    exit finalizes the entry, an exception removes the partial one so a reader never sees a half-written
    volume.

    The entry lives under a temporary name until the clean exit renames it into place: an existence
    probe (``is_dataset_exist``) or a concurrent reader never sees the entry while it is being written,
    a replaced entry stays readable until its replacement is complete, and a hard-killed writer leaves
    only temporary debris, never a plausible-looking partial volume under the final name. The
    temporary name is unique per stream (PID + sequence): two writers of the same entry (a case
    landing on two workers) each own their temporary, and whichever finalizes last publishes: a
    complete entry either way, never an interleaving of the two."""

    _sequence = itertools.count()

    @staticmethod
    def temporary_suffix() -> str:
        """The per-stream unique suffix a backend appends to its temporary name."""
        return f"{os.getpid()}-{next(DataStream._sequence)}.tmp"

    @staticmethod
    def staging_path(final: str) -> str:
        """The hidden sibling a whole-file writer stages ``final`` under until ``os.replace``: the same
        marker as :meth:`temporary_suffix`, ahead of the extension a format-detecting writer (SimpleITK,
        ITK) picks its IO from, behind a leading dot the readers' ``<name>.*`` glob cannot reach."""
        directory, filename = os.path.split(final)
        stem, _, extension = filename.partition(".")
        return os.path.join(directory, f".{stem}.{DataStream.temporary_suffix()}.{extension}")

    _file: File | None = None
    _finished: bool = False

    def __enter__(self) -> DataStream:
        return self

    @abstractmethod
    def write_slice(self, slices: tuple[slice, ...], data: np.ndarray) -> None:
        """Write ``data`` into the region ``slices`` (channel-first indices, step 1)."""

    @abstractmethod
    def _close(self, success: bool) -> None:
        """Finalize the entry, or remove the partial one when ``success`` is False."""

    def close(self) -> None:
        """Finalize the entry under its final name."""
        self._finish(True, None, None, None)

    def abort(self, error: BaseException | None = None) -> None:
        """Remove the partial entry."""
        if error is None:
            self._finish(False, None, None, None)
        else:
            self._finish(False, type(error), error, error.__traceback__)

    def __exit__(self, exc_type, value, traceback) -> None:
        self._finish(exc_type is None, exc_type, value, traceback)

    #: Where the entry lands on disk, for a backend that publishes a file or a store by rename;
    #: ``None`` for one that stages inside a container (h5).
    published_path: Path | None = None

    def _finish(self, success: bool, exc_type, value, traceback) -> None:
        # Single-shot: a caller may both close() and, on the error path, abort() the same stream (or
        # exit a ``with`` that already closed). Only the first call acts, so the backing file is exited
        # once and a failed close is not overwritten by a second _close on already-released state.
        if self._finished:
            return
        self._finished = True
        try:
            self._close(success)
            if success and (published := self.published_path) is not None:
                with contextlib.suppress(Exception):
                    _retire_dead_debris(published)  # past the publish: housekeeping cannot fail the write
        finally:
            if self._file is not None:
                self._file.__exit__(exc_type, value, traceback)


#: Handing a written page back to the kernel takes ``madvise``, which not every platform has; where
#: it is missing, the pages of a raw-block stream stay resident until the map closes.
_MADV_DONTNEED: int | None = getattr(mmap, "MADV_DONTNEED", None)


class _RawBlockStream(DataStream):
    """A local file whose pixels are one raw block: a header written once, then region writes into
    the block through a map whose pages this process does not keep.

    A shared file mapping holds every page written through it resident until it is unmapped, so a
    stream over a volume ends up holding the whole volume, budget or no budget: 64 MiB of resident
    growth over a 64 MiB volume written in sixteen slabs. Each written region is handed back to the
    kernel instead (``MADV_DONTNEED``), which leaves the bytes in the page cache to be written out
    and takes them out of this process's resident set: 0 MiB over the same sixteen slabs.
    """

    def __init__(self, path: str, header: bytes, dtype: np.dtype, shape: Sequence[int]) -> None:
        self.path = path
        self.published_path = Path(path)
        self._temporary_path = f"{path}.{self.temporary_suffix()}"
        self._dtype = np.dtype(dtype)
        self._offset = len(header)
        elements = int(np.prod(shape, dtype=np.int64))
        with open(self._temporary_path, "wb") as file:
            file.write(header)
            # Reserve the pixel block up front (sparse where the filesystem allows it).
            file.truncate(self._offset + elements * self._dtype.itemsize)
        self._handle = open(self._temporary_path, "r+b")
        self._map = mmap.mmap(self._handle.fileno(), 0)
        self._block = np.frombuffer(self._map, self._dtype, elements, self._offset).reshape(tuple(shape))

    def _write_block(self, index: tuple[slice, ...], values: np.ndarray) -> None:
        """Land ``values`` at ``index`` of the raw block and release the pages they landed on."""
        region = self._block[index]
        region[...] = values
        if _MADV_DONTNEED is None:
            return
        first = self._offset + region.ctypes.data - self._block.ctypes.data
        span = sum((extent - 1) * stride for extent, stride in zip(region.shape, region.strides, strict=True))
        page = first - first % mmap.PAGESIZE
        with contextlib.suppress(OSError):  # a filesystem whose pages cannot be dropped keeps them
            self._map.madvise(_MADV_DONTNEED, page, first + span + region.itemsize - page)

    def _close(self, success: bool) -> None:
        self._map.flush()
        del self._block  # an exported buffer keeps the map open, and the map must close before the file
        self._map.close()
        self._handle.close()
        if success:
            os.replace(self._temporary_path, self.path)
        else:
            os.remove(self._temporary_path)


# NIfTI-1 datatype code for each NumPy dtype a streamed .nii can hold.
_NIFTI_DATATYPES = {
    "uint8": 2,
    "int16": 4,
    "int32": 8,
    "float32": 16,
    "float64": 64,
    "int8": 256,
    "uint16": 512,
    "uint32": 768,
    "int64": 1024,
    "uint64": 1280,
}


class _NiftiDataStream(_RawBlockStream):
    """Uncompressed NIfTI-1 written region by region: a hand-written 348-byte header, then the raw
    block. NIfTI's data order is x fastest with the vector dimension SLOWEST, which is exactly the
    channel-first ``[C, Z, Y, X]`` layout in C order: the block is the region index itself.
    The sform carries the geometry, and NIfTI speaks RAS where the pipeline speaks LPS: the
    affine's first two rows are negated on the way out, the one convention this class owns."""

    def __init__(self, path: str, shape: list[int], dtype: np.dtype, attributes: Attribute) -> None:
        channels, spatial = int(shape[0]), [int(extent) for extent in shape[1:]]
        # The header is written little-endian, so the block must be too.
        block_dtype = np.dtype(dtype).newbyteorder("<")
        rank = len(spatial)  # 2 or 3: a 2-D image is a NIfTI of two dims, its third axis a 1
        size_xyz = [*spatial[::-1], *[1] * (3 - rank)]
        spacing = np.ones(3)
        spacing[:rank] = attributes.get_np_array("Spacing").astype(np.float64)
        origin = np.zeros(3)
        origin[:rank] = attributes.get_np_array("Origin").astype(np.float64)
        direction = np.eye(3)
        direction[:rank, :rank] = attributes.get_np_array("Direction").astype(np.float64).reshape(rank, rank)
        affine = np.concatenate([direction * spacing[np.newaxis, :], origin[:, np.newaxis]], axis=1)
        affine[:2] *= -1.0  # LPS -> RAS
        header = bytearray(348)
        struct.pack_into("<i", header, 0, 348)
        struct.pack_into("<8h", header, 40, rank if channels == 1 else 5, *size_xyz, 1, channels, 1, 1)
        if channels > 1:
            struct.pack_into("<h", header, 68, 1007)  # NIFTI_INTENT_VECTOR
        struct.pack_into("<h", header, 70, _NIFTI_DATATYPES[block_dtype.name])
        struct.pack_into("<h", header, 72, 8 * block_dtype.itemsize)
        struct.pack_into("<8f", header, 76, 1.0, *(float(part) for part in spacing), 1.0, 1.0, 1.0, 1.0)
        struct.pack_into("<f", header, 108, 352.0)  # vox_offset: the header plus the empty-extension flag
        struct.pack_into("<2f", header, 112, 1.0, 0.0)  # scl_slope / scl_inter: identity
        header[123] = 2  # xyzt_units: millimetres
        struct.pack_into("<2h", header, 252, 0, 1)  # qform unused; the sform carries the geometry
        struct.pack_into("<4f", header, 280, *(float(part) for part in affine[0]))
        struct.pack_into("<4f", header, 296, *(float(part) for part in affine[1]))
        struct.pack_into("<4f", header, 312, *(float(part) for part in affine[2]))
        header[344:348] = b"n+1\x00"
        super().__init__(path, bytes(header) + b"\x00\x00\x00\x00", block_dtype, (channels, *spatial))

    def write_slice(self, slices: tuple[slice, ...], data: np.ndarray) -> None:
        self._write_block(slices, data)


# MetaImage ElementType for each NumPy dtype a streamed .mha can hold.
_MHA_ELEMENT_TYPES = {
    "int8": "MET_CHAR",
    "uint8": "MET_UCHAR",
    "int16": "MET_SHORT",
    "uint16": "MET_USHORT",
    "int32": "MET_INT",
    "uint32": "MET_UINT",
    "int64": "MET_LONG_LONG",
    "uint64": "MET_ULONG_LONG",
    "float32": "MET_FLOAT",
    "float64": "MET_DOUBLE",
}


class _MhaDataStream(_RawBlockStream):
    """Uncompressed local-data MetaImage written region by region: a hand-written ASCII header, then
    the flat raw block. MetaIO stores vector pixels interleaved (channel fastest), so the block is
    spatial-first ``[.., Y, X, C]`` and ``write_slice`` moves the channel axis last."""

    def __init__(self, path: str, shape: list[int], dtype: np.dtype, attributes: Attribute) -> None:
        spatial = list(shape[1:])
        # The header declares BinaryDataByteOrderMSB=False, so the block must be explicitly little-endian.
        block_dtype = np.dtype(dtype).newbyteorder("<")
        fields: list[tuple[str, str]] = [
            ("ObjectType", "Image"),
            ("NDims", str(len(spatial))),
            ("BinaryData", "True"),
            ("BinaryDataByteOrderMSB", "False"),
            ("CompressedData", "False"),
            # MetaIO's TransformMatrix is the TRANSPOSE of ITK's Direction (verified against
            # sitk.WriteImage): written in Direction order, every non-symmetric orientation reads
            # back mirrored.
            (
                "TransformMatrix",
                " ".join(str(v) for v in attributes.get_np_array("Direction").reshape(len(spatial), -1).T.ravel()),
            ),
            ("Offset", " ".join(str(v) for v in attributes.get_np_array("Origin"))),
            ("ElementSpacing", " ".join(str(v) for v in attributes.get_np_array("Spacing"))),
            ("DimSize", " ".join(str(v) for v in reversed(spatial))),
        ]
        if shape[0] > 1:
            fields.append(("ElementNumberOfChannels", str(shape[0])))
        # Attribute entries ride along as MetaIO user fields, like WriteImage embeds image metadata.
        fields += [(k, str(v)) for k, v in attributes.items() if str(v) and "\n" not in str(v) and " " not in k]
        fields += [("ElementType", _MHA_ELEMENT_TYPES[block_dtype.name]), ("ElementDataFile", "LOCAL")]
        header = "".join(f"{key} = {value}\n" for key, value in fields).encode("utf-8")
        super().__init__(path, header, block_dtype, (*spatial, shape[0]))

    def write_slice(self, slices: tuple[slice, ...], data: np.ndarray) -> None:
        self._write_block((*slices[1:], slices[0]), np.moveaxis(data, 0, -1))
