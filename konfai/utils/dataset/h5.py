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


"""The HDF5 backend: pooled read handles, region streams, the group/entry layout."""

from __future__ import annotations

import contextlib
import os
import threading
import time
import warnings
from pathlib import Path
from typing import Any, NamedTuple, cast

import numpy as np

try:
    import h5py
except ImportError:
    h5py = None  # type: ignore[assignment]
try:
    import SimpleITK as sitk
except ImportError:
    sitk = None  # type: ignore[assignment]
from konfai import current_date
from konfai.utils.budget import budget_share
from konfai.utils.dataset.abstract import AbstractFile
from konfai.utils.dataset.attribute import Attribute, _encode_transform_leaves, image_to_data
from konfai.utils.dataset.staging import _REPLACED_MARKER, _orphaned_backup_names, _replaced_name, is_staging_entry
from konfai.utils.dataset.stream import DataStream
from konfai.utils.errors import DatasetManagerError


def _open_h5(path: str, mode: str, **kwargs: Any) -> Any:
    """Every h5py open in this module. Unlocked, because the flag must agree across a file's handles:
    HDF5 refuses to open a file this process already holds under the other file-locking flag, and the
    read pool keeps a handle open on a store while a stream writes it (the "invisible until finalize"
    read contract reads the store mid-write). A held HDF5 lock would also block every other process's
    open of the file for as long as the handle lives, the pool's whole lifetime. Same-process races
    are held off by the per-file thread lock.
    """
    return h5py.File(path, mode, locking=False, **kwargs)


def _get_h5_file_lock(filename: str) -> threading.RLock:
    """Return the process-wide lock guarding one HDF5 file across worker threads."""
    with _h5_file_locks_guard:
        lock = _h5_file_locks.get(filename)
        if lock is None:
            lock = threading.RLock()
            _h5_file_locks[filename] = lock
        return lock


_h5_file_locks: dict[str, threading.RLock] = {}
_h5_file_locks_guard = threading.Lock()


class _PooledRead(NamedTuple):
    """An open read handle and the store it was opened on, as one thing: the two travel together through
    eviction and re-insertion, so no site can pair a handle with a view it never had.

    The sidecars travel with them: an entry's attributes, read off the handle once and kept for its
    life, so a patch read costs one hyperslab and not one HDF5 attribute open per key on top (measured
    15 opens, 327 us, on a 15-key sidecar beside a 222 us slice). A handle replaced or dropped takes
    its sidecars with it, which is every way the store can have changed underneath them."""

    file: Any
    opened_on: tuple[int, int] | None
    sidecars: dict[str, Attribute]


class _H5ReadPool:
    """Pooled read handles, one per file per process, LRU-bounded.

    The HDF5 chunk cache lives on the open handle, so reusing the handle across patch reads is what
    makes the cache effective: a per-read open rebuilds it empty every time. ``get``/``drop`` must be
    called under the file's lock; a write drops the file's reader so it never serves stale metadata;
    handles inherited across ``fork`` are dropped unused (closing them would flush another process's
    state).

    A handle also stops answering for a store another PROCESS has written (a loader worker producing
    the group its parent reads), so one is kept only while the file it was opened on is unchanged.
    Reopening alone would not do: HDF5 shares a file's metadata state across the handles one process
    holds, so a second handle inherits the first's view. The stale one is closed before the new open."""

    _MAX = 8
    _OPEN_ATTEMPTS = 4
    _OPEN_BACKOFF = 0.05

    def __init__(self) -> None:
        self._handles: dict[str, _PooledRead] = {}
        self._guard = threading.Lock()
        self._pid = os.getpid()

    @staticmethod
    def _stamp(filename: str) -> tuple[int, int] | None:
        """What the store looked like when a handle was opened on it; ``None`` while it does not exist."""
        try:
            info = os.stat(filename)
        except OSError:
            return None
        return info.st_mtime_ns, info.st_size

    def _open(self, filename: str, **open_kwargs: Any) -> _PooledRead:
        """A handle, with the store as it was when it was opened.

        The reopen happens exactly when another process has just written, which is when that process is
        most likely to still be mid-transaction: HDF5 without SWMR then refuses the open. It is transient,
        so it is retried, and each attempt takes its own stamp: a handle is never paired with a view of
        the store taken before the write that made the previous attempt fail."""
        for remaining in reversed(range(self._OPEN_ATTEMPTS)):
            stamp = self._stamp(filename)
            try:
                return _PooledRead(_open_h5(filename, "r", **open_kwargs), stamp, {})
            except OSError:
                if not remaining:
                    raise
                time.sleep(self._OPEN_BACKOFF)
        raise AssertionError("unreachable: the last attempt either returns or raises")

    def get(self, filename: str, **open_kwargs: Any) -> _PooledRead:
        # Read before the open, never after: a write landing in between then leaves a stamp older than
        # the handle, and the next call reopens. The reverse would record a view it never had.
        stamp = self._stamp(filename)
        with self._guard:
            if os.getpid() != self._pid:
                self._handles.clear()
                self._pid = os.getpid()
            pooled = self._handles.pop(filename, None)
        # Outside the pool guard: opening touches the filesystem and may sleep between attempts. The
        # caller holds this file's lock, so no other thread of ours is reading or reopening it here.
        if pooled is not None and (not pooled.file.id.valid or pooled.opened_on != stamp):
            pooled.file.close()
            pooled = None
        if pooled is None:
            pooled = self._open(filename, **open_kwargs)
        with self._guard:
            self._handles[filename] = pooled
            evicted = []
            while len(self._handles) > self._MAX:
                oldest = next(iter(self._handles))
                evicted.append((oldest, self._handles.pop(oldest)))
        for stale_name, stale in evicted:
            self._close_idle(stale_name, stale)
        return pooled

    def drop(self, filename: str) -> None:
        with self._guard:
            pooled = self._handles.pop(filename, None)
        if pooled is not None and pooled.file.id.valid:
            pooled.file.close()

    def close_all(self) -> None:
        """Release every pooled handle: what a workflow leaves behind in the caller's process
        would otherwise keep its outputs open (read-only) for as long as the process lives."""
        with self._guard:
            handles = list(self._handles.items())
            self._handles.clear()
        for filename, pooled in handles:
            # One handle's failing close must not leave the rest open and untracked.
            with _get_h5_file_lock(filename), contextlib.suppress(Exception):
                if pooled.file.id.valid:
                    pooled.file.close()

    def _close_idle(self, filename: str, pooled: _PooledRead) -> None:
        # An evicted handle may be mid-read under its file's lock: close only when that lock is free,
        # otherwise put it back in the pool: an untracked open handle could never be dropped again.
        # It goes back with the stamp it came with: re-stamping would hand it the store as it is now,
        # and a write it never saw would stay invisible for the rest of the process.
        lock = _get_h5_file_lock(filename)
        if lock.acquire(blocking=False):
            try:
                pooled.file.close()
            finally:
                lock.release()
        else:
            with self._guard:
                self._handles.setdefault(filename, pooled)


_h5_read_pool = _H5ReadPool()


def release_read_handles() -> None:
    """Close the process's pooled read handles (h5). A workflow's caller reopening its own output
    for writing needs them gone: HDF5 refuses a write-open of a file this process holds for reading."""
    _h5_read_pool.close_all()


class _H5DataStream(DataStream):
    def __init__(self, dataset: h5py.Dataset, final_name: str) -> None:
        self._dataset = dataset
        self._final_name = final_name

    def write_slice(self, slices: tuple[slice, ...], data: np.ndarray) -> None:
        self._dataset[slices] = data

    def _close(self, success: bool) -> None:
        parent = self._dataset.parent
        temporary_name = self._dataset.name.rsplit("/", 1)[-1]
        if not success:
            del parent[temporary_name]
            return
        backup = _replaced_name(self._final_name)
        replaced = self._final_name in parent
        if replaced:
            if backup in parent:
                del parent[backup]
            parent.move(self._final_name, backup)
        try:
            parent.move(temporary_name, self._final_name)
        except Exception:
            # The old entry comes back where it was: a failed publish leaves the store as it found it.
            if replaced and self._final_name not in parent:
                parent.move(backup, self._final_name)
            raise
        if replaced:
            del parent[backup]


class H5File(AbstractFile):
    single_store = True  # one .h5 file holds every case
    concurrent_write_safe = False  # entries share the file's handles and metadata
    case_file_suffix = ".h5"  # what a case file carries when a directory keeps one per case

    @classmethod
    def can_stream(cls, file_format: str, attributes: Attribute) -> bool:
        del file_format, attributes
        return True  # a dataset written by regions, chunked or contiguous

    # Read-side HDF5 chunk cache, per opened dataset. The library default (1 MB) holds barely one
    # medical-imaging chunk, so overlapping patch reads on a chunked (compressed) store
    # re-decompress the same chunks once per patch. KonfAI writes its own h5 contiguous
    # (unaffected); this serves third-party chunked stores read through the streamed patch path.
    # nslots per the h5py guidance: a prime, well above the chunks the cache can hold.
    _READ_CHUNK_CACHE_BYTES = 128 * 1024 * 1024
    _READ_CHUNK_CACHE_SLOTS = 100003

    @staticmethod
    def _read_chunk_cache_bytes() -> int:
        """What one pooled handle's HDF5 chunk cache may hold: the cache share of the declared
        per-rank budget divided across the pool's handles, so the pool at capacity stays inside
        the one share every decoded-block cache draws from; the fixed default when no budget was
        declared."""
        share = budget_share("cache")
        if share is None:
            return H5File._READ_CHUNK_CACHE_BYTES
        return max(1, int(share) // _H5ReadPool._MAX)

    def __init__(self, filename: str, read: bool) -> None:
        if h5py is None:
            raise DatasetManagerError(
                "An ':h5' dataset needs h5py.",
                "Install it with: pip install konfai[hdf5]",
            )
        self.h5: h5py.File | None = None
        self.filename = filename
        if not self.filename.endswith(".h5"):
            self.filename += ".h5"
        self.read = read
        self._lock: threading.RLock | None = None
        self._sidecars: dict[str, Attribute] | None = None  # the pooled handle's, on a read open

    def __enter__(self):
        # A single HDF5 file cannot be opened concurrently from several threads:
        # the whole open/use/close sequence is serialised per file so that two
        # cache workers never race between the existence check and the "w"/"r+"
        # open (which would truncate each other's data).
        self._lock = _get_h5_file_lock(self.filename)
        self._lock.acquire()
        try:
            if self.read:
                pooled = _h5_read_pool.get(
                    self.filename,
                    rdcc_nbytes=self._read_chunk_cache_bytes(),
                    rdcc_nslots=self._READ_CHUNK_CACHE_SLOTS,
                )
                self.h5, self._sidecars = pooled.file, pooled.sidecars
            else:
                _h5_read_pool.drop(self.filename)
                if not os.path.exists(self.filename):
                    Path(self.filename).parent.mkdir(parents=True, exist_ok=True)
                    self.h5 = _open_h5(self.filename, "w")
                else:
                    self.h5 = _open_h5(self.filename, "r+")
                self.h5.attrs["Date"] = current_date()
        except BaseException:
            self._lock.release()
            self._lock = None
            raise
        return self.h5

    def __exit__(self, exc_type, value, traceback):
        try:
            if self.h5 is not None and not self.read:
                self.h5.close()
        finally:
            if self._lock is not None:
                self._lock.release()
                self._lock = None

    def _sidecar(self, dataset: h5py.Dataset) -> Attribute:
        """The entry's attributes, a copy of the pooled handle's record of them: one attribute open
        per key on the first read of the entry through the handle, none after. A write handle is
        not pooled and reads them off the file."""
        if self._sidecars is None:
            return Attribute(dict(dataset.attrs))
        sidecar = self._sidecars.get(dataset.name)
        if sidecar is None:
            sidecar = self._sidecars[dataset.name] = Attribute(dict(dataset.attrs))
        return Attribute(sidecar)

    def file_to_data(self, groups: str, name: str) -> tuple[np.ndarray, Attribute]:
        dataset = self._require_dataset(groups, name)
        data = np.zeros(dataset.shape, dataset.dtype)
        dataset.read_direct(data)
        return data, self._sidecar(dataset)

    def bounded_region_reads(self, name: str) -> bool:
        del name
        # A hyperslab reads the bytes it covers (contiguous, what KonfAI writes) or the chunks it
        # touches (a third-party chunked store): never the whole volume.
        return True

    def file_to_data_slice(self, groups: str, name: str, slices: tuple[slice, ...]) -> tuple[np.ndarray, Attribute]:
        dataset = self._require_dataset(groups, name)
        data = np.asarray(dataset[slices])
        return data, self._sidecar(dataset)

    def data_to_file(
        self,
        name: str,
        data: sitk.Image | sitk.Transform | np.ndarray,
        attributes: Attribute | None = None,
    ) -> None:
        if self.h5 is None:
            return
        if attributes is None:
            attributes = Attribute()
        if isinstance(data, sitk.Image):
            data, attributes_tmp = image_to_data(data)
            attributes.update(attributes_tmp)
        elif isinstance(data, sitk.Transform):
            data = np.asarray(_encode_transform_leaves(data, name, attributes))

        h5_group, name = self._resolve_group(name)
        # Staged under a temp name and moved, never created under the final one: the invariant
        # every DataStream holds (a hard-killed writer leaves .tmp debris, not a plausible
        # partial entry the resume then SKIPs as done). The old entry is moved aside and put
        # back if the publish fails, so no instant has neither version in the file.
        staging = f"{name}.{DataStream.temporary_suffix()}"
        if staging in h5_group:
            del h5_group[staging]
        self._create_entry(h5_group, staging, attributes, data=data, dtype=data.dtype)
        backup = _replaced_name(name)
        replaced = name in h5_group
        if replaced:
            if backup in h5_group:
                del h5_group[backup]
            h5_group.move(name, backup)
        try:
            h5_group.move(staging, name)
        except Exception:
            if replaced and name not in h5_group:
                h5_group.move(backup, name)
            raise
        if replaced:
            del h5_group[backup]

    @staticmethod
    def _create_entry(h5_group: h5py.Group, key: str, attributes: Attribute, **dataset_kwargs: Any) -> h5py.Dataset:
        """A dataset with its attributes, or nothing: an interrupt between the two must not leave an
        attribute-less entry (or an orphaned temporary) in a file HDF5 never reclaims space from.
        Contiguous: a full-row slab is one byte span, and a patch reads its own bytes, not a chunk."""
        dataset = h5_group.create_dataset(key, chunks=None, **dataset_kwargs)
        try:
            dataset.attrs.update({k: str(v) for k, v in attributes.items()})
        except BaseException:
            del h5_group[key]
            raise
        return dataset

    def _resolve_group(self, name: str) -> tuple[h5py.Group, str]:
        """The (created) parent group a slash-qualified entry name writes into, and its leaf name."""
        h5 = cast(h5py.File, self.h5)
        h5_group: h5py.Group = h5
        if len(name.split("/")) > 1:
            group = "/".join(name.split("/")[:-1])
            if group not in h5:
                h5.create_group(group)
            h5_group = h5[group]
        return h5_group, name.split("/")[-1]

    def open_data_stream(
        self,
        name: str,
        shape: list[int],
        dtype: np.dtype,
        attributes: Attribute,
        region_shape: list[int] | None = None,
    ) -> DataStream | None:
        if self.h5 is None:
            return None
        h5_group, name = self._resolve_group(name)
        temporary_name = f"{name}.{DataStream.temporary_suffix()}"
        dataset = self._create_entry(h5_group, temporary_name, attributes, shape=tuple(shape), dtype=dtype)
        return _H5DataStream(dataset, name)

    def _recovered_key(self, h5_group: h5py.Group, name: str) -> str | None:
        """The key ``name`` answers to when it is missing: its own, or the single backup a DEAD
        writer left of it (see :func:`_recover_orphaned_backup`, the same rule inside a file).

        An h5 file open for READING cannot be renamed in, so the backup is served under its own
        key and put back at the next write open, which is when the move is legal.
        """
        if name in h5_group:
            return name
        backups = _orphaned_backup_names(list(h5_group.keys()), name)
        if len(backups) != 1:
            return None
        warnings.warn(
            f"'{name}' was missing from '{self.filename}' and its previous version was recovered from "
            f"'{backups[0]}': a writer was killed between moving the entry aside and publishing its "
            "replacement. The entry is the one that was there BEFORE that write; run the write again "
            "to replace it.",
            UserWarning,
            stacklevel=3,
        )
        if not self.read:
            h5_group.move(backups[0], name)
            return name
        return backups[0]

    def is_exist(self, group: str, name: str | None = None) -> bool:
        if self.h5 is not None:
            if group in self.h5:
                if isinstance(self.h5[group], h5py.Dataset):
                    return True
                elif name is not None:
                    return self._recovered_key(self.h5[group], name) is not None
                else:
                    return False
        return False

    def get_names(self, groups: str, h5_group: h5py.Group = None) -> list[str]:
        names = []
        if h5_group is None:
            h5_group = self.h5
        group = groups.split("/")[0]
        if group == "":
            names = [
                dataset.name.split("/")[-1]
                for dataset in h5_group.values()
                # ``.tmp`` keys are in-flight (or hard-kill-orphaned) DataStream writes, not entries.
                if isinstance(dataset, h5py.Dataset) and not is_staging_entry(dataset.name)
            ]
            # A backup a dead writer orphaned IS its entry (see _recover_orphaned_backup), and a
            # listing that hid it while the probe and the read recover it would name fewer cases
            # than the store serves: a run would silently skip one.
            names.extend(self._orphaned_entries(h5_group, names))
        elif group == "*":
            for k in h5_group.keys():
                if isinstance(h5_group[k], h5py.Group):
                    names.extend(self.get_names("/".join(groups.split("/")[1:]), h5_group[k]))
        else:
            if group in h5_group:
                names.extend(self.get_names("/".join(groups.split("/")[1:]), h5_group[group]))
        return names

    @staticmethod
    def _orphaned_entries(h5_group: h5py.Group, present: list[str]) -> list[str]:
        """The names whose only version left in this group is one dead writer's backup."""
        keys = list(h5_group.keys())
        missing = {key.split(_REPLACED_MARKER)[0] for key in keys if _REPLACED_MARKER in key} - set(present)
        return sorted(name for name in missing if len(_orphaned_backup_names(keys, name)) == 1)

    def get_group(self) -> list[str]:
        return list(self.h5.keys()) if self.h5 is not None else []

    def _require_dataset(self, groups: str, name: str) -> h5py.Dataset:
        """The entry, or the designed refusal: an absent group resolved ``None`` and every reader
        dereferenced it, an anonymous ``AttributeError`` deep in numpy where the sibling backends
        name the entry."""
        dataset = self._get_dataset(groups, name)
        if dataset is None:
            entry = f"{groups}/{name}" if groups else name
            raise DatasetManagerError(
                f"'{entry}' is not in '{self.filename}'.",
                "Check the case name and the group it is looked up under.",
            )
        return dataset

    def _get_dataset(self, groups: str, name: str, h5_group: h5py.Group = None) -> h5py.Dataset | None:
        if h5_group is None:
            h5_group = self.h5
        if groups != "":
            group = groups.split("/")[0]
        else:
            group = ""
        result = None
        if group == "":
            key = self._recovered_key(h5_group, name)
            if key is not None:
                result = h5_group[key]
        elif group == "*":
            for k in h5_group.keys():
                if isinstance(h5_group[k], h5py.Group):
                    result_tmp = self._get_dataset("/".join(groups.split("/")[1:]), name, h5_group[k])
                    if result_tmp is not None:
                        result = result_tmp
        else:
            if group in h5_group:
                result_tmp = self._get_dataset("/".join(groups.split("/")[1:]), name, h5_group[group])
                if result_tmp is not None:
                    result = result_tmp
        return result

    def get_infos(self, groups: str, name: str) -> tuple[list[int], Attribute]:
        dataset = self._require_dataset(groups, name)
        return dataset.shape, self._sidecar(dataset)
