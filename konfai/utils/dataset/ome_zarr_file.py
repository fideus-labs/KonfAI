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


"""The OME-Zarr backend: multiscale stores read and written level by level."""

from __future__ import annotations

import contextlib
import os
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None  # type: ignore[assignment]
from konfai.utils import uri
from konfai.utils.dataset.abstract import AbstractFile
from konfai.utils.dataset.attribute import (
    DISPLACEMENT_FIELD_ATTRIBUTE,
    Attribute,
    displacement_field_to_data,
    image_to_data,
    ome_zarr_attributes,
    region_geometry,
)
from konfai.utils.dataset.staging import _recover_orphaned_backup, _replaced_name, _retire_dead_debris
from konfai.utils.dataset.stream import DataStream
from konfai.utils.errors import DatasetManagerError
from konfai.utils.utils import (
    STORE_FORMS,
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


#: Where each entry's store was resolved on disk, keyed by ``(root, entry)``: the store-suffix
#: probes are one ``fs.info`` round-trip each on a remote root, per patch without this. A write
#: through this backend forgets the memo (it may change the suffix the entry resolves under); a
#: store REPLACED at the same path keeps its resolution, so no other invalidation is owed.
_resolved_store_paths: dict[tuple[str, str], str] = {}


def _forget_resolved_paths() -> None:
    """Drop the entry-path memo: what a write must call, being the one thing that moves a store."""
    _resolved_store_paths.clear()


class _OmeZarrDataStream(DataStream):
    def __init__(
        self,
        array: Any,
        store_path: Path,
        final_path: Path,
        scale_factors: list[int] | None = None,
        downsample_method: str | None = None,
    ) -> None:
        self._array = array
        self._store_path = store_path
        self._final_path = final_path
        self.published_path = Path(final_path)
        self._scale_factors = scale_factors
        self._downsample_method = downsample_method

    def write_slice(self, slices: tuple[slice, ...], data: np.ndarray) -> None:
        self._array[slices] = data

    def _close(self, success: bool) -> None:
        from konfai.utils.ome_zarr import (
            append_ome_zarr_levels,
            clear_ome_zarr_cache,
        )

        if not success:
            shutil.rmtree(self._store_path, ignore_errors=True)
            return
        if self._scale_factors:
            # On the temporary store, so the rename below publishes level 0 and its coarser levels in
            # one step. The levels are grafted beside level 0 (one pass over it, into an array 4^rank
            # times smaller); level 0 itself is not rewritten.
            append_ome_zarr_levels(self._store_path, self._scale_factors, downsample_method=self._downsample_method)
            self._array = None
        replaced = self._final_path.exists()
        backup = self._final_path.with_name(_replaced_name(self._final_path.name))
        if replaced:
            shutil.rmtree(backup, ignore_errors=True)
            os.rename(self._final_path, backup)
        try:
            os.rename(self._store_path, self._final_path)
        except OSError:
            # A concurrent writer of the same entry renamed its complete, identical store into place;
            # keep it and drop ours.
            if not self._final_path.exists():
                if replaced:
                    os.rename(backup, self._final_path)  # a failed publish leaves the old entry in place
                raise
            shutil.rmtree(self._store_path, ignore_errors=True)
        if replaced:
            shutil.rmtree(backup, ignore_errors=True)
        # The reader memoises loaded stores by path, and this rename changes what that path holds.
        # A store replaced by one written through a different code path can differ down to the key
        # its level-0 array lives under, so a stale entry does not merely serve old pixels: it
        # points at a component that is no longer there. This path alone: the sources a cohort is
        # still reading are not what changed.
        clear_ome_zarr_cache(self._final_path)
        _forget_resolved_paths()


class OmeZarrFile(AbstractFile):
    """OME-NGFF backend using chunked Zarr reads for KonfAI patches.

    ``level`` selects the multiscale pyramid resolution to read (0 = full
    resolution, higher = coarser); it comes from the ``omezarr@<level>``
    dataset-spec suffix.

    ``scale_factors`` is the WRITE-side counterpart: it makes the store this backend writes a
    pyramid instead of a single level. Reading indexes a pyramid BY POSITION, so a producer that
    writes one and a consumer that asks for ``@1`` are two halves of the same contract.
    """

    concurrent_write_safe = False  # a store shares metadata across its arrays
    reads_remote = True  # a store is addressed by key, so fsspec serves a URI root
    writes_pyramid = True  # the one format with levels
    lists_case_entries = True  # a case is a directory of stores this backend enumerates

    @classmethod
    def can_stream(cls, file_format: str, attributes: Attribute) -> bool:
        del file_format, attributes
        return True  # zarr chunks materialise as regions land

    def __init__(
        self,
        filename: str,
        read: bool,
        level: int = 0,
        scale_factors: list[int] | None = None,
        downsample_method: str | None = None,
    ) -> None:
        self.filename = filename if filename.endswith("/") else f"{filename}/"
        self.read = read
        self.level = level
        self.scale_factors = list(scale_factors) if scale_factors else None
        self.downsample_method = downsample_method

    def __enter__(self):
        return self

    def __exit__(self, exc_type, value, traceback):
        return None

    def _path(self, name: str, *, writing: bool = False) -> str:
        """Where entry ``name``'s store sits: text, because a remote one is a URI and ``Path``
        eats the second slash of one. Resolved once per ``(root, entry)``: each suffix probe is a
        round-trip on a remote root, and the store's location cannot change mid-run."""
        base = uri.join(self.filename, name)
        if writing:
            uri.refuse_write(self.filename)
            return f"{base}.ome.zarr"
        memo_key = (self.filename, name)
        resolved = _resolved_store_paths.get(memo_key)
        if resolved is not None:
            return resolved
        _resolved_store_paths[memo_key] = resolved = self._resolve_path(name, base)
        return resolved

    def _resolve_path(self, name: str, base: str) -> str:
        # Every spelling is_store_name accepts, or a root whose first case names one of the
        # others is detected as omezarr at setup and then fails to resolve.
        candidates = [f"{base}{form}" for form in STORE_FORMS] + [base]
        for candidate in candidates:
            if uri.is_dir(candidate):
                return candidate
        listed = self._listed_as(name)
        if listed is not None:
            return listed
        if not uri.is_uri(self.filename):
            for candidate in candidates:  # a writer killed mid-replacement left the previous store aside
                _recover_orphaned_backup(Path(candidate))
                if os.path.isdir(candidate):  # recovered here, or published by whoever won the race
                    return candidate
        raise DatasetManagerError(
            f"The OME-Zarr group '{name}' is not in '{self.filename}'.",
            "Check the group name against the store's own arrays.",
        )

    def _listed_as(self, name: str) -> str | None:
        """Where ``name``'s store sits when the directory spells its suffix in another case,
        ``None`` when nothing there is that store.

        ``is_store_name`` and :meth:`get_group` match the suffix case-insensitively, so a
        ``CT.OME.ZARR`` is accepted at setup and listed as ``CT``; on a case-sensitive
        filesystem the probes above, which are the accepted spellings in lower case, all miss
        it. Only the miss pays the listing, and it lists one case's directory.
        """
        prefix, _, stem = name.rpartition("/")
        directory = uri.join(self.filename, prefix) if prefix else self.filename
        wanted = {f"{stem}{form}".lower() for form in STORE_FORMS}
        for entry in uri.list_names(directory):
            if entry.lower() in wanted:
                return uri.join(directory, entry)
        return None

    @staticmethod
    def _attributes(metadata: dict[str, Any]) -> Attribute:
        return ome_zarr_attributes(metadata)

    def file_to_data(self, group: str, name: str) -> tuple[np.ndarray, Attribute]:
        from konfai.utils.ome_zarr import is_displacement_field

        info_shape, _ = self.get_infos(group, name)
        data, attributes = self.file_to_data_slice(group, name, tuple(slice(None) for _ in info_shape))
        # Marked here and not in file_to_data_slice: that one is the streamed path, called once per
        # patch, and re-reading the store's metadata per patch is exactly the overhead _load_image
        # is memoised to avoid. A transform is only ever rebuilt from a whole entry.
        if is_displacement_field(self._path(name)):
            attributes[DISPLACEMENT_FIELD_ATTRIBUTE] = "true"
        return data, attributes

    def file_to_data_slice(self, group: str, name: str, slices: tuple[slice, ...]) -> tuple[np.ndarray, Attribute]:
        from konfai.utils.ome_zarr import read_ome_zarr_data_slice

        path = self._path(name)
        data, metadata = read_ome_zarr_data_slice(path, slices, level=self.level)
        attributes = self._attributes(metadata)
        shape = metadata["shape"]
        normalized = tuple(slice(*item.indices(size)) for item, size in zip(slices, shape, strict=True))
        origin, spacing = region_geometry(
            attributes.get_np_array("Origin"),
            attributes.get_np_array("Spacing"),
            attributes.get_np_array("Direction"),
            normalized[1:],
        )
        attributes["Origin"] = origin
        attributes["Spacing"] = spacing
        return data, attributes

    def bounded_region_reads(self, name: str) -> bool:
        del name
        return True  # zarr is chunked: a slice reads its chunks and nothing else

    def plan_region_reads(self, name: str, windows: Sequence[tuple[slice, ...]]) -> None:
        from konfai.utils.ome_zarr import plan_ome_zarr_reads

        plan_ome_zarr_reads(self._path(name), windows, level=self.level)

    def read_granularity(self, name: str) -> tuple[int, ...] | None:
        from konfai.utils.ome_zarr import ome_zarr_read_granularity

        return ome_zarr_read_granularity(self._path(name), level=self.level)

    def data_to_file(
        self,
        name: str,
        data: sitk.Image | sitk.Transform | np.ndarray,
        attributes: Attribute | None = None,
    ) -> None:
        from konfai.utils.ome_zarr import clear_ome_zarr_cache, write_ome_zarr

        attributes = attributes or Attribute()
        # Two ways to say "this is a field": hand over a DisplacementFieldTransform, or mark the
        # attributes. The second exists because a producer that never builds a transform: the
        # predictor emits arrays: would otherwise have to wrap its output in one purely to be
        # described correctly, and a field too large to hold in memory cannot be wrapped at all.
        displacement_field = DISPLACEMENT_FIELD_ATTRIBUTE in attributes
        if sitk is not None and isinstance(data, sitk.Image):
            data, image_attributes = image_to_data(data)
            attributes.update(image_attributes)
        elif sitk is not None and isinstance(data, sitk.Transform):
            # The parametric transforms the other backends serialise (Euler, affine, B-spline) have
            # no OME-NGFF form; a displacement field does, and it is array-backed, so this backend
            # stores exactly the one kind it can store faithfully.
            data, field_attributes = displacement_field_to_data(data, name)
            attributes.update(field_attributes)
            displacement_field = True
        if not isinstance(data, np.ndarray):
            raise DatasetManagerError("OME-Zarr datasets can only store image arrays.")
        # Staged beside the final store and renamed over it: writing under the final name
        # truncates the destination before a byte lands, so a crash mid-write left a partial
        # store the resume then counted as already written -- and an overwrite lost both
        # versions. The rename is the atomicity every DataStream already holds; the .replaced
        # hop keeps an instant with SOME complete store on disk.
        final = Path(self._path(name, writing=True))
        staging = final.with_name(f"{final.name}.{os.getpid()}.tmp")
        if staging.exists():
            shutil.rmtree(staging)
        write_ome_zarr(
            staging,
            data,
            spacing=attributes.get_np_array("Spacing") if "Spacing" in attributes else None,
            origin=attributes.get_np_array("Origin") if "Origin" in attributes else None,
            attributes=dict(attributes),
            displacement_field=displacement_field,
            scale_factors=self.scale_factors,
            downsample_method=self.downsample_method,
        )
        replaced = final.with_name(f"{final.name}.{os.getpid()}.replaced")
        shutil.rmtree(replaced, ignore_errors=True)
        try:
            if final.exists():
                final.rename(replaced)
            staging.rename(final)
        except BaseException:
            if replaced.exists() and not final.exists():
                replaced.rename(final)
            shutil.rmtree(staging, ignore_errors=True)  # or a full second copy of the entry stays
            raise
        shutil.rmtree(replaced, ignore_errors=True)
        # The reader memoises decoded chunks by path, and this path now holds another store.
        clear_ome_zarr_cache(final)
        _forget_resolved_paths()
        with contextlib.suppress(Exception):
            _retire_dead_debris(final)  # housekeeping past the publish: it cannot fail the write

    def open_data_stream(
        self,
        name: str,
        shape: list[int],
        dtype: np.dtype,
        attributes: Attribute,
        region_shape: list[int] | None = None,
    ) -> DataStream | None:
        from konfai.utils.ome_zarr import create_ome_zarr_store

        if len(shape) not in (3, 4):
            return None
        final_path = Path(self._path(name, writing=True))
        store_path = final_path.with_name(f"{final_path.name}.{DataStream.temporary_suffix()}")
        array = create_ome_zarr_store(
            store_path,
            shape,
            dtype,
            spacing=attributes.get_np_array("Spacing") if "Spacing" in attributes else None,
            origin=attributes.get_np_array("Origin") if "Origin" in attributes else None,
            attributes=dict(attributes),
            displacement_field=DISPLACEMENT_FIELD_ATTRIBUTE in attributes,
            # Chunked against what the writer says it will write, capped to something a reader
            # can open. Guessing the writer's access pattern costs a read-modify-write on every
            # region whose extent straddles a chunk: measured 1.8x on a slab sweep, paid on
            # every byte, and invisible because the bytes are correct either way.
            chunks=_store_chunks(shape, region_shape, dtype),
        )
        # The pyramid cannot be created up front: no level exists until the last region lands --
        # so the stream derives it at finalize, on the TEMPORARY store, before the rename. That
        # order is what keeps publication atomic: a reader never sees a store whose level 0 is
        # complete but whose coarser levels are not.
        return _OmeZarrDataStream(array, store_path, final_path, self.scale_factors, self.downsample_method)

    @classmethod
    def open(
        cls,
        filename: str,
        read: bool,
        file_format: str,
        level: int = 0,
        scale_factors: list[int] | None = None,
        downsample_method: str | None = None,
    ) -> OmeZarrFile:
        del file_format
        return cls(filename, read, level, scale_factors, downsample_method)

    def get_names(self, group: str) -> list[str]:
        return self.get_group()

    def get_group(self) -> list[str]:
        groups = []
        for name in uri.list_names(self.filename):
            form = next((form for form in STORE_FORMS if name.lower().endswith(form)), None)
            if form is not None:
                groups.append(name[: -len(form)])
        return sorted(groups)

    def is_exist(self, group: str, name: str | None = None) -> bool:
        try:
            self._path(f"{group}/{name}" if name else group)
            return True
        except DatasetManagerError:
            return False

    def get_infos(self, group: str, name: str) -> tuple[list[int], Attribute]:
        from konfai.utils.ome_zarr import get_ome_zarr_info, is_displacement_field

        metadata = get_ome_zarr_info(self._path(name), level=self.level)
        axes = [str(axis).lower() for axis in metadata["axes"]]
        axis_sizes = dict(zip(axes, metadata["shape"], strict=True))
        shape = [axis_sizes.get("c", 1), *[axis_sizes[axis] for axis in ("z", "y", "x") if axis in axis_sizes]]
        metadata["shape"] = shape
        attributes = self._attributes(metadata)
        # Marked on the HEADERS path, so a field stays a field on the streamed read too --
        # file_to_data marks it only on the whole-volume read, and a store written from unmarked
        # regions is an ordinary 3-channel image. This is the once-per-case call (Dataset caches
        # it), not the per-patch one, which is why the check belongs here and not in
        # file_to_data_slice.
        if is_displacement_field(self._path(name)):
            attributes[DISPLACEMENT_FIELD_ATTRIBUTE] = "true"
        return shape, attributes
