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


"""The backend contract every storage format implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None  # type: ignore[assignment]
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from konfai.utils.dataset.attribute import Attribute
    from konfai.utils.dataset.stream import DataStream


class AbstractFile(ABC):
    """One storage backend: how a ``(group, name)`` entry is read, written and enumerated.

    The per-backend FACTS live here as class-level declarations, so a new backend is one module
    plus one :data:`~konfai.utils.dataset.backend.BACKENDS` entry: the dataset consults the class,
    never a format-name branch of its own.
    """

    #: One store holds every case (a single ``.h5`` file); a directory backend keeps one file (or
    #: store directory) per case, and the dataset walks the root itself.
    single_store: bool = False

    #: Whether writes to different entries land in disjoint files, so a background writer may
    #: flush one entry while another thread writes elsewhere in the dataset. A backend whose
    #: entries share handles or metadata (one HDF5 file, a zarr hierarchy, a DICOM series)
    #: declares False and stays serial.
    concurrent_write_safe: bool = True

    #: The suffix a case file carries implicitly on disk (``.h5``), or ``None`` when the case path
    #: is spelled as listed.
    case_file_suffix: str | None = None

    #: Whether this backend reads a remote (URI) root; the rest open a local path.
    reads_remote: bool = False

    #: Whether a written store can hold multiscale levels (``scale_factors``): only a format with
    #: levels may be asked for a pyramid.
    writes_pyramid: bool = False

    #: Whether a case is a directory of entries the backend itself enumerates (``get_group`` on
    #: the case), rather than plain files the dataset walks.
    lists_case_entries: bool = False

    @abstractmethod
    def __init__(self, filename: str, read: bool) -> None:
        pass

    @classmethod
    def open(
        cls,
        filename: str,
        read: bool,
        file_format: str,
        level: int = 0,
        scale_factors: list[int] | None = None,
        downsample_method: str | None = None,
    ) -> AbstractFile:
        """This backend on ``filename``, built from the dispatch's full hand: each backend takes
        the arguments its constructor actually needs and ignores the rest."""
        del file_format, level, scale_factors, downsample_method
        return cls(filename, read)

    @classmethod
    def can_stream(cls, file_format: str, attributes: Attribute) -> bool:
        """Whether this backend can serve incremental region writes for ``file_format``.

        The base answers ``False``: a backend that cannot stream is written whole through
        ``data_to_file``.
        """
        del file_format, attributes
        return False

    @abstractmethod
    def __enter__(self):
        pass

    @abstractmethod
    def __exit__(self, exc_type, value, traceback):
        pass

    @abstractmethod
    def file_to_data(self, group: str, name: str) -> tuple[np.ndarray, Attribute]:
        pass

    @abstractmethod
    def file_to_data_slice(self, group: str, name: str, slices: tuple[slice, ...]) -> tuple[np.ndarray, Attribute]:
        pass

    def bounded_region_reads(self, name: str) -> bool:
        """Whether a region read decodes only the region, or the whole volume behind the scenes.

        The base answers ``False``: getting this wrong only ever costs speed, never correctness,
        and an unknown backend is priced pessimistically. What it prices is the ROUTE: a store
        that decodes the whole volume once per slab makes streaming read the source many times
        over, where loading reads it once.

        A compressed stream answers ``False`` but is not all-or-nothing: ITK decodes forward from
        the start and stops on the region, so a read costs its END offset, not its size. Measured
        on a 256^3 int16 volume, one 32^3 region at z = 0 / 64 / 128 / 224:

        =============  ======  ======  ======  ======
        store            z=0    z=64   z=128   z=224
        =============  ======  ======  ======  ======
        .mha            0.8ms   0.9ms   0.8ms   0.8ms
        .mha (zlib)    14.6ms  41.8ms  70.5ms  112.6ms
        .nii.gz        16.1ms  35.8ms  55.5ms  85.2ms
        =============  ======  ======  ======  ======

        The uncompressed row is flat because it seeks; the other two are linear in depth, and the
        deepest read costs the whole file (102 ms). Sweeping in K regions therefore costs about
        K/2 whole decodes, which is why ``False`` here buys the LOAD verdict -- one ordered read
        -- rather than a streamed route, and why the patch route can only warn and advise a
        chunked store. Blocked compression (Zarr, HDF5) and indexed access into one stream
        (``zran``/``indexed_gzip``, which nibabel uses) both solve it; ITK does neither, and the
        header it writes carries ``CompressedDataSize`` and no compression table.
        """
        del name
        return False

    def read_granularity(self, name: str) -> tuple[int, ...] | None:
        """The stored block a region read is served in, as a ``C[Z]YX`` shape, or ``None``.

        A chunked backend decodes whole blocks, so a window costs the block-aligned hull that
        covers it: a decomposition that straddles the grid pays every block it touches in full.
        ``None`` says a read costs what it asks for, which is what the base answers.
        """
        del name
        return None

    def plan_region_reads(self, name: str, windows: Sequence[tuple[slice, ...]]) -> None:
        """Declare the windows a caller will read from ``name``, in the order it will read them.

        A hint and never a promise: a backend that caches decoded blocks keeps what a later
        window asks for again and drops what none does, which is the fewest decodes any policy
        can reach and none can reach without the future. The base ignores it, as does any caller
        that declares nothing.
        """
        del name, windows

    @abstractmethod
    def data_to_file(
        self,
        name: str,
        data: sitk.Image | sitk.Transform | np.ndarray,
        attributes: Attribute | None = None,
    ) -> None:
        pass

    def open_data_stream(
        self,
        name: str,
        shape: list[int],
        dtype: np.dtype,
        attributes: Attribute,
        region_shape: list[int] | None = None,
    ) -> DataStream | None:
        """Open ``name`` for incremental region writes; ``None`` when this backend cannot."""
        return None

    def get_names(self, group: str) -> list[str]:
        """The cases of ``group`` this store holds. Only a backend that enumerates its own entries
        answers (a single store, a store-per-case directory); a plain-file backend's cases are the
        root's listing, which the dataset walks itself."""
        raise NotImplementedError(f"{type(self).__name__} keeps one file per entry; the dataset lists its root.")

    def get_group(self) -> list[str]:
        """The groups this store holds, under the same contract as :meth:`get_names`."""
        raise NotImplementedError(f"{type(self).__name__} keeps one file per entry; the dataset walks its root.")

    @abstractmethod
    def is_exist(self, group: str, name: str | None = None) -> bool:
        pass

    @abstractmethod
    def get_infos(self, group: str, name: str) -> tuple[list[int], Attribute]:
        pass
