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
    @abstractmethod
    def __init__(self) -> None:
        pass

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

    @abstractmethod
    def get_names(self, group: str) -> list[str]:
        pass

    @abstractmethod
    def get_group(self) -> list[str]:
        pass

    @abstractmethod
    def is_exist(self, group: str, name: str | None = None) -> bool:
        pass

    @abstractmethod
    def get_infos(self, group: str, name: str) -> tuple[list[int], Attribute]:
        pass
