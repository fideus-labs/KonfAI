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

"""Dataset file abstractions and image conversion utilities for KonfAI."""

from __future__ import annotations

import ast
import contextlib
import copy
import csv
import functools
import glob
import itertools
import math
import os
import re
import shutil
import sys
import threading
import time
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from functools import partial
from pathlib import Path
from typing import Any, NamedTuple, TypeVar, cast

import numpy as np
import torch
from lxml import etree  # nosec B410

try:
    import h5py
except ImportError:
    h5py = None  # type: ignore[assignment]
try:
    import SimpleITK as sitk
except ImportError:
    sitk = None  # type: ignore[assignment]

from konfai import current_date
from konfai.utils.errors import DatasetManagerError
from konfai.utils.utils import SUPPORTED_EXTENSIONS, directory_volume_form, split_format_level

_h5_file_locks: dict[str, threading.RLock] = {}
_h5_file_locks_guard = threading.Lock()


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


class _PooledRead(NamedTuple):
    """An open read handle and the store it was opened on, as one thing: the two travel together through
    eviction and re-insertion, so no site can pair a handle with a view it never had."""

    file: Any
    opened_on: tuple[int, int] | None


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
                return _PooledRead(_open_h5(filename, "r", **open_kwargs), stamp)
            except OSError:
                if not remaining:
                    raise
                time.sleep(self._OPEN_BACKOFF)
        raise AssertionError("unreachable: the last attempt either returns or raises")

    def get(self, filename: str, **open_kwargs: Any) -> Any:
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
        return pooled.file

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


def _attribute_text(value: Any) -> str:
    """One value as an attribute holds it: its printed form, on one line and complete.

    The single place a value stops being a live object, because every consumer takes text --
    ``SetMetaData`` on a SimpleITK image, an h5 attribute, a zarr sidecar. The printed form is left
    as each type prints it: a sequence has two, :meth:`Attribute.get_np_array` reads both, and
    ``ast.literal_eval`` (:meth:`Dataset.read_transform`, on the parameter keys) needs the Python
    one, so normalising to either here breaks the other reader.

    Complete, because an attribute is a record and not a display: NumPy's own printing elides values
    past a threshold, and an elided record is one no reader can parse back. Exact, for the same
    reason: a float is printed as the shortest text that reads back to the very same float64 --
    NumPy's default (8 decimals for an array, the float32-shortest form for a float32 scalar) is a
    display, and a statistic that came back a few ulps off made the whole-volume and the streamed
    path of one chain disagree by that much (measured: a Min/Max rescale, 28% of voxels one ulp
    apart on CUDA).
    """
    if isinstance(value, torch.Tensor):
        # Accept a tensor from any device: attributes are host-side strings, and finalize transforms
        # (Normalize, Statistics, ...) may hand over stats computed on a CUDA-resident volume.
        value = value.detach().cpu().numpy()
    if isinstance(value, np.generic | np.ndarray) and np.issubdtype(value.dtype, np.floating):
        value = np.asarray(value, dtype=np.float64)[()] if isinstance(value, np.generic) else value.astype(np.float64)
    with np.printoptions(threshold=sys.maxsize, floatmode="unique"):
        return str(value).replace("\n", "")


def _store_chunks(shape: list[int], region_shape: list[int] | None, dtype: Any) -> tuple[int, ...] | None:
    """Chunks a store should use, given the region shape its writer declared.

    A region write that straddles a chunk becomes a read-modify-write of the whole chunk, so the
    writer's own region is the honest starting point. Taking it verbatim is not: a slab sweep
    declares the whole trailing plane, which at 2048x2048 float32 is a gigabyte in one chunk (past
    what zarr will hold in a single buffer at 4096x4096), and every later partial read pays it,
    including readers that are not KonfAI.

    So an axis the region covers END TO END is tiled, and an axis where the region is a strict
    sub-range keeps the writer's size. Tiling a fully-covered axis cannot split a region write, since
    every chunk along it falls inside one region either way; splitting a partial axis is exactly the
    read-modify-write this exists to avoid. Innermost axes go first, so the chunk stays long on the
    axis the writer actually advances along.

    ``None`` when the writer declared nothing: the store keeps its own default rather than a guess.
    """
    from konfai.utils.ome_zarr import CHUNK_SPATIAL_TILE, CHUNK_TARGET_BYTES

    if region_shape is None or len(region_shape) != len(shape):
        return None
    chunk = [max(1, min(int(region), int(extent))) for region, extent in zip(region_shape, shape, strict=True)]
    itemsize = max(1, np.dtype(dtype).itemsize)
    covered = [axis for axis, extent in enumerate(shape) if chunk[axis] >= int(extent)]
    for axis in reversed(covered):
        if int(np.prod(chunk, dtype=np.int64)) * itemsize <= CHUNK_TARGET_BYTES:
            break
        chunk[axis] = min(chunk[axis], CHUNK_SPATIAL_TILE)
    return tuple(chunk)


class Attribute(dict[str, Any]):
    """Metadata container storing repeated values with a stack-like naming scheme.

    Values are text, always. Both doors normalize (assignment and construction), so an attribute
    built from a store's own sidecar, which JSON hands back as live lists, is the same thing as one
    assigned in Python. Anything less and a value can be stored and not written back out.
    """

    def __init__(self, attributes: dict[str, Any] | None = None) -> None:
        super().__init__()
        attributes = attributes or {}
        for k, v in attributes.items():
            super().__setitem__(copy.deepcopy(k), _attribute_text(v))

    @staticmethod
    def _is_stack_member(stored_key: str, key: str) -> bool:
        # Values are stacked as ``{key}_{n}``; match that exact pattern (or the bare key) so a sibling that
        # merely shares a prefix (``SpacingOriginal`` vs ``Spacing``) is not miscounted as another entry.
        if stored_key == key:
            return True
        prefix = f"{key}_"
        return stored_key.startswith(prefix) and stored_key[len(prefix) :].isdigit()

    def _count_key(self, key: str) -> int:
        return sum(1 for k in super().keys() if Attribute._is_stack_member(k, key))

    def __getitem__(self, key: str) -> Any:
        i = self._count_key(key)
        if i > 0 and f"{key}_{i - 1}" in super().keys():
            return str(super().__getitem__(f"{key}_{i - 1}"))
        if key in super().keys():
            return str(super().__getitem__(key))
        raise NameError(f"{key} not in cache_attribute")

    def __setitem__(self, key: str, value: Any) -> None:
        result = _attribute_text(value)
        if "_" not in key:
            super().__setitem__(f"{key}_{self._count_key(key)}", result)
        else:
            super().__setitem__(key, result)

    def pop(self, key: str, default: Any = None) -> Any:
        i = self._count_key(key)
        if i > 0 and f"{key}_{i - 1}" in super().keys():
            return super().pop(f"{key}_{i - 1}")
        if key in super().keys():
            return super().pop(key)
        raise NameError(f"{key} not in cache_attribute")

    @staticmethod
    def _parse_array(text: str) -> np.ndarray:
        """Both printed forms of a sequence: NumPy's ``[1.5 1.5 2.]`` and Python's ``[1.5, 1.5, 2.0]``.

        Which one an attribute holds follows from what the writer handed over (an ``ndarray``, or
        the plain list a JSON sidecar gives back), and no reader should have to make that
        distinction. ``np.fromstring`` reads whitespace only, so the commas go first.
        """
        return np.fromstring(text[1:-1].replace(",", " "), sep=" ", dtype=np.double)

    def get_np_array(self, key: str) -> np.ndarray:
        return Attribute._parse_array(self[key])

    def get_tensor(self, key: str) -> torch.Tensor:
        return torch.tensor(self.get_np_array(key)).to(torch.float32)

    def pop_np_array(self, key: str) -> np.ndarray:
        return Attribute._parse_array(self.pop(key))

    def pop_tensor(self, key: str) -> torch.Tensor:
        return torch.tensor(self.pop_np_array(key))

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        return any(Attribute._is_stack_member(k, key) for k in super().keys())

    def is_info(self, key: str, value: str) -> bool:
        return key in self and self[key] == value


# Elements held in memory at once while accumulating statistics chunk by chunk, whatever the backend.
_STATISTICS_CHUNK_ELEMENTS = 8_000_000


def _statistics_chunk_length(shape: list[int] | tuple[int, ...], axis: int) -> int:
    """How far along ``axis`` a chunk may reach to hold about ``_STATISTICS_CHUNK_ELEMENTS``.

    A chunk spans every other axis whole (channels included), so the per-step cost is the volume
    divided by ``axis``; the length is that budget over the per-step cost, floored to one step.
    """
    per_step = int(np.prod([extent for other, extent in enumerate(shape) if other != axis], dtype=np.int64))
    return max(1, _STATISTICS_CHUNK_ELEMENTS // max(1, per_step))


#: Values a quantile scan collects at once when a bin has narrowed this far: the one buffer it holds.
_QUANTILE_COLLECT_CAP = 1 << 22
_QUANTILE_BINS = 1 << 16


def _quantile_positions(count: int, q: float) -> tuple[int, int, float]:
    """The two order statistics ``numpy.quantile(..., method='linear')`` interpolates between, and the
    weight: the same arithmetic as numpy's ``_compute_virtual_index`` (alpha = beta = 1)."""
    virtual = count * q + (1 + q * (1 - 1 - 1)) - 1
    if virtual >= count - 1:
        return count - 1, count - 1, 0.0
    if virtual < 0:
        return 0, 0, 0.0
    previous = int(np.floor(virtual))
    return previous, previous + 1, float(virtual - previous)


def _lerp_like_numpy(a: Any, b: Any, t: float) -> Any:
    """``numpy.lib._function_base_impl._lerp`` on two scalars of the array's dtype and a Python weight."""
    diff = b - a
    result = a + diff * t
    if t >= 0.5:
        result = b - diff * (1 - t)
    return result


def _binned(block: np.ndarray, low: Any, high: Any) -> tuple[np.ndarray, np.ndarray]:
    """The block's values inside ``[low, high]`` and the bin each falls in (monotone in the value)."""
    flat = block.reshape(-1)
    inside = flat[(flat >= low) & (flat <= high)]
    scaled = (inside.astype(np.float64) - float(low)) / (float(high) - float(low)) * _QUANTILE_BINS
    return inside, np.minimum(scaled.astype(np.int64), _QUANTILE_BINS - 1)


def _min_of(current: Any, candidate: Any) -> Any:
    """``min`` of a running value that may not exist yet and a candidate."""
    return candidate if current is None or candidate < current else current


def _max_of(current: Any, candidate: Any) -> Any:
    """``max`` of a running value that may not exist yet and a candidate."""
    return candidate if current is None or candidate > current else current


def _order_statistics(blocks: Callable[[], Iterator[np.ndarray]], q: float) -> tuple[Any, Any, float]:
    """The two order statistics ``numpy.quantile(..., q)`` interpolates between, and the weight, over
    everything the blocks hold, without holding it: one pass counts and bounds the values, then passes
    narrow a value interval by histogram until the rank's bin holds few enough values to collect, or a
    single value.
    """
    count = 0
    low = high = None
    for block in blocks():
        flat = block.reshape(-1)
        if flat.size == 0:
            continue
        if np.issubdtype(flat.dtype, np.floating) and np.isnan(flat).any():
            # numpy.quantile of anything holding a NaN is NaN; a bin can hold no NaN, so the
            # narrowing below would otherwise search a histogram the count does not match.
            return np.nan, np.nan, 0.0
        count += int(flat.size)
        low, high = _min_of(low, flat.min()), _max_of(high, flat.max())
    if count == 0:
        raise ValueError("quantile of an empty volume")
    assert low is not None and high is not None  # nosec B101 - count > 0
    first, second, weight = _quantile_positions(count, q)
    below = 0  # values strictly under the interval, all levels folded
    inside_count = count  # values in the interval
    min_above: Any = None  # the smallest value strictly over the interval
    while True:
        if low == high:
            value = low
            if second == first or first - below + 1 < inside_count:
                return value, value, weight
            return value, min_above if min_above is not None else value, weight
        binned = partial(_binned, low=low, high=high)
        histogram = np.zeros(_QUANTILE_BINS, dtype=np.int64)
        for block in blocks():
            _inside, index = binned(block)
            histogram += np.bincount(index, minlength=_QUANTILE_BINS)
        cumulative = np.cumsum(histogram)
        chosen = int(np.searchsorted(cumulative, first - below, side="right"))
        before_bin = int(cumulative[chosen - 1]) if chosen else 0
        in_bin = int(histogram[chosen])
        collected: list[np.ndarray] = []
        bin_low = bin_high = None
        above_local: Any = None
        for block in blocks():
            inside, index = binned(block)
            if not inside.size:
                continue
            members = inside[index == chosen]
            if members.size:
                bin_low, bin_high = _min_of(bin_low, members.min()), _max_of(bin_high, members.max())
                if in_bin <= _QUANTILE_COLLECT_CAP:
                    collected.append(members)
            higher = inside[index > chosen]
            if higher.size:
                above_local = _min_of(above_local, higher.min())
        if above_local is not None:
            min_above = _min_of(min_above, above_local)
        below += before_bin
        rank_in_bin = first - below
        if collected:
            values = np.sort(np.concatenate(collected))
            value = values[rank_in_bin]
            if second == first:
                return value, value, weight
            second_value = values[rank_in_bin + 1] if rank_in_bin + 1 < values.size else min_above
            return value, second_value if second_value is not None else value, weight
        assert bin_low is not None and bin_high is not None  # nosec B101 - the bin holds target's rank
        low, high, inside_count = bin_low, bin_high, in_bin


def _update_running_statistics(
    state: dict[str, Any] | None,
    array: np.ndarray,
) -> dict[str, Any]:
    """Update running min/max/mean/std from a NumPy chunk, over the volume AND per channel.

    Both grains come from one pass because they come from the same samples: a chunk arrives as
    ``(C, ...)``, so the per-channel figures are the same Welford recurrence applied along axis 0,
    and the whole-volume ones are that recurrence pooled. Computing them separately would mean
    scanning the volume once per channel: three passes over a displacement field to learn three
    numbers.
    """
    values = np.asarray(array, dtype=np.float64)
    per_channel = values.reshape(values.shape[0], -1) if values.ndim > 1 else values.reshape(1, -1)
    flat = per_channel.reshape(-1)
    channels = per_channel.shape[0]
    if flat.size == 0:
        return state or _empty_statistics_state(channels)

    if state is None:
        state = _empty_statistics_state(channels)

    chunk_count = float(flat.size)
    chunk_mean = float(flat.mean())
    chunk_m2 = float(np.square(flat - chunk_mean).sum())

    # Per channel, the same recurrence on vectors, one entry per channel, updated together.
    channel_count = float(per_channel.shape[1])
    channel_mean = per_channel.mean(axis=1)
    channel_m2 = np.square(per_channel - channel_mean[:, None]).sum(axis=1)
    channel_total = state["channel_count"] + channel_count
    if channel_total > 0:
        channel_delta = channel_mean - state["channel_mean"]
        state["channel_mean"] = state["channel_mean"] + channel_delta * channel_count / channel_total
        state["channel_m2"] = (
            state["channel_m2"]
            + channel_m2
            + channel_delta * channel_delta * state["channel_count"] * channel_count / channel_total
        )
        state["channel_count"] = channel_total
        state["channel_min"] = np.minimum(state["channel_min"], per_channel.min(axis=1))
        state["channel_max"] = np.maximum(state["channel_max"], per_channel.max(axis=1))

    total_count = state["count"] + chunk_count
    delta = chunk_mean - state["mean"]
    if total_count > 0:
        state["mean"] += delta * chunk_count / total_count
        state["m2"] += chunk_m2 + delta * delta * state["count"] * chunk_count / total_count
        state["count"] = total_count
        state["min"] = min(state["min"], float(flat.min()))
        state["max"] = max(state["max"], float(flat.max()))
    return state


def _empty_statistics_state(channels: int) -> dict[str, Any]:
    return {
        "count": 0.0,
        "mean": 0.0,
        "m2": 0.0,
        "min": np.inf,
        "max": -np.inf,
        "channel_count": 0.0,
        "channel_mean": np.zeros(channels, dtype=np.float64),
        "channel_m2": np.zeros(channels, dtype=np.float64),
        "channel_min": np.full(channels, np.inf),
        "channel_max": np.full(channels, -np.inf),
    }


def _finalize_running_statistics(state: dict[str, Any] | None) -> dict[str, Any]:
    """Convert a running-statistics state into the public stats dictionary.

    The four scalars are the volume's; the four ``*_per_channel`` lists are the same figures per
    channel, which is what a vector-valued quantity needs: the mean of a displacement field is
    three numbers, and pooling them into one describes nothing.
    """
    if state is None or state["count"] == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0} | {
            f"{key}_per_channel": [0.0] for key in ("min", "max", "mean", "std")
        }
    variance = state["m2"] / (state["count"] - 1) if state["count"] > 1 else 0.0
    channel_variance = state["channel_m2"] / max(1.0, state["channel_count"] - 1)
    return {
        "min": state["min"],
        "max": state["max"],
        "mean": state["mean"],
        "std": math.sqrt(max(variance, 0.0)),
        "min_per_channel": state["channel_min"].tolist(),
        "max_per_channel": state["channel_max"].tolist(),
        "mean_per_channel": state["channel_mean"].tolist(),
        "std_per_channel": np.sqrt(np.maximum(channel_variance, 0.0)).tolist(),
    }


# Formats already reported by _warn_unstreamed_region_read. Keyed by format, not by file: the remedy
# is dataset-wide, so every case of a dataset would otherwise repeat the same warning.
_unstreamed_formats_warned: set[str] = set()


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


def is_an_image(attributes: Attribute) -> bool:
    """Return whether the given attribute set contains image geometry metadata."""
    return "Origin" in attributes and "Spacing" in attributes and "Direction" in attributes


def data_to_image(data: np.ndarray, attributes: Attribute) -> sitk.Image:
    """Convert a NumPy array and KonfAI attributes into a SimpleITK image."""
    if isinstance(data, torch.Tensor):
        # Accept a torch tensor on any device: SimpleITK works on host arrays, so a SITK-backed transform
        # fed a CUDA-resident volume converts here and naturally returns on the CPU (the pipeline then
        # continues on the CPU). This keeps every transform usable regardless of the volume's device.
        data = data.detach().cpu().numpy()
    if not is_an_image(attributes):
        raise NameError("Data is not an image")
    if data.dtype == np.float16:
        # ITK has no half-float pixel type (GetImageFromArray rejects float16), so widen to float32 --
        # exact and lossless. The streamed .mha writer widens the same way, so both write identical bytes.
        data = data.astype(np.float32)
    if data.shape[0] == 1:
        image = sitk.GetImageFromArray(data[0])
    else:
        data = data.transpose(tuple([i + 1 for i in range(len(data.shape) - 1)] + [0]))
        image = sitk.GetImageFromArray(data, isVector=True)
    for k, v in attributes.items():
        if v and len(v):
            image.SetMetaData(k, v)
    image.SetOrigin(attributes.get_np_array("Origin").tolist())
    image.SetSpacing(attributes.get_np_array("Spacing").tolist())
    image.SetDirection(attributes.get_np_array("Direction").tolist())
    return image


# Set on an entry read back from a store that types its component axis as an RFC-5 displacement
# field, so ``Dataset.read_transform`` can rebuild the transform. The underscore matters: a key
# without one is stack-renamed by ``Attribute.__setitem__`` (``Transform`` becomes ``Transform_0``).
DISPLACEMENT_FIELD_ATTRIBUTE = "konfai_displacement_field"


def displacement_field_to_data(transform: sitk.Transform, name: str) -> tuple[np.ndarray, Attribute]:
    """A displacement-field transform as a channel-first array plus its geometry.

    The counterpart of ``_encode_transform_leaves`` for the one transform kind that cannot go through
    it: a displacement field's parameters ARE the field, so serialising it as a parameter vector
    would drop the geometry that makes it meaningful. It travels as an image instead, and the store
    records what it is (see ``write_ome_zarr(displacement_field=True)``).
    """
    if not isinstance(transform, sitk.DisplacementFieldTransform):
        raise DatasetManagerError(
            f"Expected a DisplacementFieldTransform for entry '{name}', got '{type(transform).__name__}'."
        )
    return image_to_data(transform.GetDisplacementField())


def image_to_data(image: sitk.Image) -> tuple[np.ndarray, Attribute]:
    """Convert a SimpleITK image into a channel-first NumPy array and attributes."""
    attributes = Attribute()
    for k in image.GetMetaDataKeys():
        # ``ITK_*`` keys are the reader's own bookkeeping (the input filter's name, the file's original
        # direction and spacing), not the volume's metadata: carried into an output they describe the
        # source of a resampled volume, which nothing should read as the output's.
        if not k.startswith("ITK_"):
            attributes[k] = image.GetMetaData(k)
    # AFTER the metadata import, deliberately. data_to_image stamps every attribute -- the
    # geometry stack included -- back onto the image as metadata text, and the loop above imports
    # it verbatim (versioned keys carry a '_', so they land as-is). Recorded first, the header
    # landed as Origin_0 and the stale text then OVERWROTE that very key: an image read, moved
    # (SetOrigin) and written back kept its old origin, silently. Recorded last, the header
    # appends the next version of the stack, which is the one every reader takes.
    attributes["Origin"] = np.asarray(image.GetOrigin())
    attributes["Spacing"] = np.asarray(image.GetSpacing())
    attributes["Direction"] = np.asarray(image.GetDirection())
    data = sitk.GetArrayFromImage(image)

    if image.GetNumberOfComponentsPerPixel() == 1:
        data = np.expand_dims(data, 0)
    else:
        data = np.transpose(data, (len(data.shape) - 1, *list(range(len(data.shape) - 1))))
    return data, attributes


def ome_zarr_attributes(metadata: dict[str, Any]) -> Attribute:
    """A KonfAI ``Attribute`` (Origin / Spacing / Direction) from an OME-Zarr entry's metadata.

    The store's konfai sidecar wins when present (it carries the full Direction matrix, which NGFF
    scale/translation cannot express) otherwise geometry falls back to the NGFF transforms, Direction
    defaulting to identity. Shared by the Dataset OME-Zarr reader and ``ITK.read_displacement_field``
    so both recover geometry the one same way.
    """
    attributes = Attribute(metadata.get("attributes", {}))
    axes = metadata["axes"]
    scale = dict(zip(axes, metadata.get("scale", []), strict=False))
    translation = dict(zip(axes, metadata.get("translation", []), strict=False))
    spatial_axes = [axis for axis in ("x", "y", "z") if axis in axes]
    if "Spacing" not in attributes:
        attributes["Spacing"] = np.asarray([scale.get(axis, 1.0) for axis in spatial_axes])
    if "Origin" not in attributes:
        attributes["Origin"] = np.asarray([translation.get(axis, 0.0) for axis in spatial_axes])
    if "Direction" not in attributes:
        attributes["Direction"] = np.eye(len(spatial_axes), dtype=np.float64).flatten()
    attributes["OMEAxes"] = np.asarray(axes)
    return attributes


def _flatten_transforms(transform: sitk.Transform) -> list[sitk.Transform]:
    """The leaf transforms of a (possibly nested) composite, in application order.

    ``CompositeTransform.GetNthTransform`` can itself return a composite, so a single-level walk
    leaves a nested composite in the list and the serializer rejects it. Recurse to the leaves.
    """
    if isinstance(transform, sitk.CompositeTransform):
        leaves: list[sitk.Transform] = []
        for i in range(transform.GetNumberOfTransforms()):
            leaves.extend(_flatten_transforms(transform.GetNthTransform(i)))
        return leaves
    return [transform]


def _transform_codec() -> list[tuple[type, str, Any]]:
    """(sitk class, serialized type tag, decode factory) for every supported transform kind.

    Built lazily because ``sitk`` is an optional import.
    """
    return [
        (sitk.Euler3DTransform, "Euler3DTransform_double_3_3", sitk.Euler3DTransform),
        (sitk.AffineTransform, "AffineTransform_double_3_3", lambda: sitk.AffineTransform(3)),
        (sitk.BSplineTransform, "BSplineTransform_double_3_3", lambda: sitk.BSplineTransform(3)),
    ]


def _encode_transform_leaves(transform: sitk.Transform, name: str, attributes: Attribute) -> list[np.ndarray]:
    """Serialize a (possibly composite) transform: record each leaf's type tag and fixed parameters
    into ``attributes`` (``{i}:Transform`` / ``{i}:FixedParameters``) and return the per-leaf
    parameter arrays, in application order."""
    datas: list[np.ndarray] = []
    for i, leaf in enumerate(_flatten_transforms(transform)):
        type_tag = next((tag for sitk_class, tag, _ in _transform_codec() if isinstance(leaf, sitk_class)), None)
        if type_tag is None:
            raise DatasetManagerError(f"Unsupported transform type '{type(leaf).__name__}' for entry '{name}'.")
        attributes[f"{i}:Transform"] = type_tag
        attributes[f"{i}:FixedParameters"] = leaf.GetFixedParameters()

        datas.append(np.asarray(leaf.GetParameters()))
    return datas


def _decode_transform(transform_type: str, name: str) -> sitk.Transform:
    """A fresh transform instance for a serialized type tag."""
    for _, type_tag, factory in _transform_codec():
        if transform_type == type_tag:
            return factory()
    raise DatasetManagerError(f"Unsupported transform type '{transform_type}' for entry '{name}'.")


def get_infos(filename: str | Path) -> tuple[list[int], Attribute]:
    """Read shape and metadata from an image file without loading its full pixel data."""
    attributes = Attribute()
    file_reader = sitk.ImageFileReader()
    file_reader.SetFileName(str(filename))
    file_reader.ReadImageInformation()
    attributes["Origin"] = np.asarray(file_reader.GetOrigin())
    attributes["Spacing"] = np.asarray(file_reader.GetSpacing())
    attributes["Direction"] = np.asarray(file_reader.GetDirection())
    for k in file_reader.GetMetaDataKeys():
        attributes[k] = file_reader.GetMetaData(k)
    # SimpleITK GetSize() is (x, y, [z], ...); KonfAI arrays are numpy-order [C, (Z), Y, X], so the
    # spatial size must be reversed for EVERY rank: a 3-D-only reversal transposes 2-D/4-D data.
    size = list(reversed(file_reader.GetSize()))
    size = [file_reader.GetNumberOfComponents(), *size]
    return size, attributes


def read_landmarks(filename: Path) -> np.ndarray | None:
    """Read Slicer-style fiducial landmarks from disk."""
    data = None
    with open(filename, newline="") as csvfile:
        reader = csv.reader(filter(lambda row: row[0] != "#", csvfile))
        lines = list(reader)
        data = np.zeros((len(list(lines)), 3), dtype=np.double)
        for i, row in enumerate(lines):
            data[i] = np.array(row[1:4], dtype=np.double)
        csvfile.close()
    return data


def write_landmarks(data: np.ndarray, filename: Path) -> None:
    """Write landmarks to the Slicer Markups fiducial CSV-like format."""
    with open(filename, "w") as f:
        f.write(
            "# Markups fiducial file version = 4.6\n# CoordinateSystem = LPS\n#"
            " columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID\n",
        )
        for i in range(data.shape[0]):
            f.write(
                "vtkMRMLMarkupsFiducialNode_"
                + str(i + 1)
                + ","
                + str(data[i, 0])
                + ","
                + str(data[i, 1])
                + ","
                + str(data[i, 2])
                + ",0,0,0,1,1,1,0,F-"
                + str(i + 1)
                + ",,vtkMRMLScalarVolumeNode1\n"
            )
        f.close()


#: The suffix an entry is moved to while its replacement is published. A stream moves the old entry
#: aside rather than deleting it first: neither HDF5 nor a directory swap has an atomic rename-over,
#: and a crash between a delete and the move would lose both. Per pid, so two writers of one entry
#: never share a backup.
_REPLACED_MARKER = ".replaced-"


def _replaced_name(name: str) -> str:
    """Where ``name`` (an h5 key or a directory leaf) is kept while its replacement is published."""
    return f"{name}{_REPLACED_MARKER}{os.getpid()}"


def is_staging_entry(name: str) -> bool:
    """Whether ``name`` (a path or an h5 key) is a writer's staging entry, never a case: an in-flight (or
    hard-kill-orphaned) temporary carrying the ``.tmp`` marker of :meth:`DataStream.temporary_suffix` or
    :meth:`DataStream.staging_path`, or the :func:`_replaced_name` an entry is moved to while its
    replacement is published."""
    leaf = os.path.basename(name)
    return leaf.endswith(".tmp") or ".tmp." in leaf or _REPLACED_MARKER in leaf


# A writer's staging name carries its pid: ``<entry>.<pid>[-n].tmp``, the ``.replaced`` hop it keeps
# the previous version under, or the dotted whole-file form ``.<entry>.<pid>.tmp.<ext>``.
_STAGING_PID = re.compile(r"\.(?:(?P<pid>\d+)(?:-\d+)?\.(?:tmp|replaced)|replaced-(?P<hop>\d+))(?:\.|$)")


def _writer_is_dead(pid: int) -> bool:
    if pid == os.getpid():
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        pass  # alive, someone else's
    return False


def _retire_dead_debris(final: Path) -> None:
    """Remove what earlier, DEAD writers of ``final`` left beside it.

    Every writer here stages under a pid-marked name and publishes by rename, so a hard kill leaves
    a staging file or store the readers already know to skip -- and nothing ever removed: a
    27 GB one-hot store's staging sat beside the published one for good. Publishing an entry is
    the moment its history is settled, so the debris of any writer that no longer runs goes then.
    A LIVE writer's staging is left alone (two writers of one entry are legal, the last rename
    wins), which is what the pid in the name is for.
    """
    entry = final.name.split(".", 1)[0]
    try:
        siblings = list(final.parent.iterdir())
    except OSError:
        return
    for sibling in siblings:
        if sibling == final or not sibling.name.lstrip(".").startswith(f"{entry}."):
            continue
        marker = _STAGING_PID.search(sibling.name)
        if marker is None or not _writer_is_dead(int(marker.group("pid") or marker.group("hop"))):
            continue
        if sibling.is_dir():
            shutil.rmtree(sibling, ignore_errors=True)
        else:
            sibling.unlink(missing_ok=True)


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

    _file: Dataset.File | None = None
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
                _retire_dead_debris(published)
        finally:
            if self._file is not None:
                self._file.__exit__(exc_type, value, traceback)


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


def _create_itk_transform_file(path: str, spatial: list[int], attributes: Attribute) -> tuple[Any, Any]:
    """An ITK displacement-transform HDF5 file with its parameters dataset still to fill.

    Three datasets, as ITK's own writer lays them out: the type (a variable-length ASCII string,
    which is what ITK's reader accepts), the fixed parameters (size, origin, spacing, direction)
    and the parameters, the field buffer with the component fastest, float64. Returns the open file
    and the parameters dataset.
    """
    fixed = np.concatenate(
        [
            np.asarray(spatial[::-1], dtype=np.float64),  # size, in (x, y, z)
            attributes.get_np_array("Origin").astype(np.float64),
            attributes.get_np_array("Spacing").astype(np.float64),
            attributes.get_np_array("Direction").astype(np.float64).reshape(-1),
        ]
    )
    file = _open_h5(path, "w")
    file.create_dataset(
        "TransformGroup/0/TransformType",
        data=[b"DisplacementFieldTransform_double_3_3"],
        dtype=h5py.string_dtype(encoding="ascii"),
    )
    file.create_dataset("TransformGroup/0/TransformFixedParameters", data=fixed)
    parameters = file.create_dataset(
        "TransformGroup/0/TransformParameters", shape=(3 * int(np.prod(spatial)),), dtype=np.float64
    )
    return file, parameters


class _ItkTransformDataStream(DataStream):
    """An ITK displacement-transform file written region by region.

    A slab of the field maps to one contiguous span of the parameters (the buffer is ``[z][y][x]``
    with the component fastest), so full-width leading-axis slabs (what the streamed write
    dispatcher emits) land with plain offset writes. Under a temporary name until the clean exit,
    like every stream.
    """

    def __init__(self, file: Any, parameters: Any, temporary_path: str, final_path: str, spatial: list[int]) -> None:
        self._h5 = file
        self._parameters = parameters
        self._temporary_path = temporary_path
        self._final_path = final_path
        self.published_path = Path(final_path)
        self._spatial = [int(extent) for extent in spatial]

    def write_slice(self, slices: tuple[slice, ...], data: np.ndarray) -> None:
        channels, leading, *rest = slices
        full = (channels.start or 0) == 0 and channels.stop in (None, 3)
        for axis, part in enumerate(rest, start=2):
            full = full and (part.start or 0) == 0 and part.stop in (None, self._spatial[axis])
        if not full:
            raise DatasetManagerError(
                "A transform file writes full-width leading-axis slabs, and this region is not one.",
                "This is a bug if it was reached: the streamed write dispatcher finalizes full rows.",
            )
        block = np.moveaxis(np.asarray(data, dtype=np.float64), 0, -1).ravel()
        offset = 3 * int(leading.start or 0) * int(np.prod(self._spatial[2:], dtype=np.int64))
        self._parameters[offset : offset + block.size] = block

    def _close(self, success: bool) -> None:
        self._h5.close()
        if success:
            os.replace(self._temporary_path, self._final_path)
        else:
            Path(self._temporary_path).unlink(missing_ok=True)


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


class _NiftiDataStream(DataStream):
    """Uncompressed NIfTI-1 written region by region: a hand-written 348-byte header, then a memmap
    over the raw block. NIfTI's data order is x fastest with the vector dimension SLOWEST, which is
    exactly the channel-first ``[C, Z, Y, X]`` layout in C order: the map is the block itself.
    The sform carries the geometry, and NIfTI speaks RAS where the pipeline speaks LPS: the
    affine's first two rows are negated on the way out, the one convention this class owns."""

    def __init__(self, path: str, shape: list[int], dtype: np.dtype, attributes: Attribute) -> None:
        import struct

        self.path = path
        self.published_path = Path(path)
        self._temporary_path = f"{path}.{self.temporary_suffix()}"
        channels, spatial = int(shape[0]), [int(extent) for extent in shape[1:]]
        # The header is written little-endian, so the block must be too.
        self._dtype = np.dtype(dtype).newbyteorder("<")
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
        struct.pack_into("<h", header, 70, _NIFTI_DATATYPES[self._dtype.name])
        struct.pack_into("<h", header, 72, 8 * self._dtype.itemsize)
        struct.pack_into("<8f", header, 76, 1.0, *(float(part) for part in spacing), 1.0, 1.0, 1.0, 1.0)
        struct.pack_into("<f", header, 108, 352.0)  # vox_offset: the header plus the empty-extension flag
        struct.pack_into("<2f", header, 112, 1.0, 0.0)  # scl_slope / scl_inter: identity
        header[123] = 2  # xyzt_units: millimetres
        struct.pack_into("<2h", header, 252, 0, 1)  # qform unused; the sform carries the geometry
        struct.pack_into("<4f", header, 280, *(float(part) for part in affine[0]))
        struct.pack_into("<4f", header, 296, *(float(part) for part in affine[1]))
        struct.pack_into("<4f", header, 312, *(float(part) for part in affine[2]))
        header[344:348] = b"n+1\x00"
        total = int(np.prod([channels, *spatial], dtype=np.int64)) * self._dtype.itemsize
        with open(self._temporary_path, "wb") as file:
            file.write(bytes(header))
            file.write(b"\x00\x00\x00\x00")  # no extensions
            # Reserve the pixel block up front (sparse where the filesystem allows it).
            file.truncate(352 + total)
        self._memmap = np.memmap(
            self._temporary_path, dtype=self._dtype, mode="r+", offset=352, shape=(channels, *spatial)
        )

    def write_slice(self, slices: tuple[slice, ...], data: np.ndarray) -> None:
        self._memmap[slices] = data

    def _close(self, success: bool) -> None:
        self._memmap.flush()
        del self._memmap
        if success:
            os.replace(self._temporary_path, self.path)
        else:
            os.remove(self._temporary_path)


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


class _MhaDataStream(DataStream):
    """Uncompressed local-data MetaImage written region by region: a hand-written ASCII header, then a
    memmap over the flat raw block. MetaIO stores vector pixels interleaved (channel fastest), so the
    map is spatial-first ``[.., Y, X, C]`` and ``write_slice`` moves the channel axis last."""

    def __init__(self, path: str, shape: list[int], dtype: np.dtype, attributes: Attribute) -> None:
        self.path = path
        self.published_path = Path(path)
        self._temporary_path = f"{path}.{self.temporary_suffix()}"
        spatial = list(shape[1:])
        # The header declares BinaryDataByteOrderMSB=False, so the map must be explicitly little-endian.
        self._dtype = np.dtype(dtype).newbyteorder("<")
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
        fields += [("ElementType", _MHA_ELEMENT_TYPES[self._dtype.name]), ("ElementDataFile", "LOCAL")]
        header = "".join(f"{key} = {value}\n" for key, value in fields).encode("utf-8")
        with open(self._temporary_path, "wb") as file:
            file.write(header)
            # Reserve the pixel block up front (sparse where the filesystem allows it).
            file.truncate(len(header) + int(np.prod([*spatial, shape[0]], dtype=np.int64)) * self._dtype.itemsize)
        self._memmap = np.memmap(
            self._temporary_path, dtype=self._dtype, mode="r+", offset=len(header), shape=(*spatial, shape[0])
        )

    def write_slice(self, slices: tuple[slice, ...], data: np.ndarray) -> None:
        self._memmap[(*slices[1:], slices[0])] = np.moveaxis(data, 0, -1)

    def _close(self, success: bool) -> None:
        self._memmap.flush()
        del self._memmap
        if success:
            os.replace(self._temporary_path, self.path)
        else:
            os.remove(self._temporary_path)


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
        # points at a component that is no longer there.
        clear_ome_zarr_cache()


_T = TypeVar("_T")


class Dataset:
    """Filesystem or HDF5-backed dataset abstraction used across KonfAI."""

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
            """
            del name
            return False

        @abstractmethod
        def file_to_data_statistics(
            self,
            group: str,
            name: str,
            channels: list[int] | None = None,
        ) -> dict[str, Any]:
            pass

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

    class H5File(AbstractFile):
        # Read-side HDF5 chunk cache, per opened dataset. The library default (1 MB) holds barely one
        # medical-imaging chunk, so overlapping patch reads on a chunked (compressed) store
        # re-decompress the same chunks once per patch. KonfAI writes its own h5 contiguous
        # (unaffected); this serves third-party chunked stores read through the streamed patch path.
        # nslots per the h5py guidance: a prime, well above the chunks the cache can hold.
        _READ_CHUNK_CACHE_BYTES = 128 * 1024 * 1024
        _READ_CHUNK_CACHE_SLOTS = 100003

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

        def __enter__(self):
            # A single HDF5 file cannot be opened concurrently from several threads:
            # the whole open/use/close sequence is serialised per file so that two
            # cache workers never race between the existence check and the "w"/"r+"
            # open (which would truncate each other's data).
            self._lock = _get_h5_file_lock(self.filename)
            self._lock.acquire()
            try:
                if self.read:
                    self.h5 = _h5_read_pool.get(
                        self.filename,
                        rdcc_nbytes=self._READ_CHUNK_CACHE_BYTES,
                        rdcc_nslots=self._READ_CHUNK_CACHE_SLOTS,
                    )
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

        def file_to_data(self, groups: str, name: str) -> tuple[np.ndarray, Attribute]:
            dataset = self._get_dataset(groups, name)
            data = np.zeros(dataset.shape, dataset.dtype)
            dataset.read_direct(data)
            return data, Attribute(dict(dataset.attrs))

        def bounded_region_reads(self, name: str) -> bool:
            del name
            # A hyperslab reads the bytes it covers (contiguous, what KonfAI writes) or the chunks it
            # touches (a third-party chunked store): never the whole volume.
            return True

        def file_to_data_slice(self, groups: str, name: str, slices: tuple[slice, ...]) -> tuple[np.ndarray, Attribute]:
            dataset = self._get_dataset(groups, name)
            data = np.asarray(dataset[slices])
            return data, Attribute(dict(dataset.attrs))

        def file_to_data_statistics(
            self,
            groups: str,
            name: str,
            channels: list[int] | None = None,
        ) -> dict[str, Any]:
            dataset = self._get_dataset(groups, name)
            if dataset is None:
                raise NameError(f"Dataset '{groups}/{name}' not found in '{self.filename}'.")

            axis = 1 if dataset.ndim > 1 else 0
            chunk_length = _statistics_chunk_length(dataset.shape, axis)
            state: dict[str, Any] | None = None

            for start in range(0, dataset.shape[axis], chunk_length):
                slices = [slice(None)] * dataset.ndim
                slices[axis] = slice(start, min(dataset.shape[axis], start + chunk_length))
                chunk = np.asarray(dataset[tuple(slices)])
                if channels is not None:
                    chunk = chunk[channels]
                state = _update_running_statistics(state, chunk)

            return _finalize_running_statistics(state)

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

        def is_exist(self, group: str, name: str | None = None) -> bool:
            if self.h5 is not None:
                if group in self.h5:
                    if isinstance(self.h5[group], h5py.Dataset):
                        return True
                    elif name is not None:
                        return name in self.h5[group]
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
            elif group == "*":
                for k in h5_group.keys():
                    if isinstance(h5_group[k], h5py.Group):
                        names.extend(self.get_names("/".join(groups.split("/")[1:]), h5_group[k]))
            else:
                if group in h5_group:
                    names.extend(self.get_names("/".join(groups.split("/")[1:]), h5_group[group]))
            return names

        def get_group(self) -> list[str]:
            return list(self.h5.keys()) if self.h5 is not None else []

        def _get_dataset(self, groups: str, name: str, h5_group: h5py.Group = None) -> h5py.Dataset:
            if h5_group is None:
                h5_group = self.h5
            if groups != "":
                group = groups.split("/")[0]
            else:
                group = ""
            result = None
            if group == "":
                if name in h5_group:
                    result = h5_group[name]
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
            dataset = self._get_dataset(groups, name)
            return (
                dataset.shape,
                Attribute(dict(dataset.attrs)),
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

        def _file_to_image_statistics(self, name: str, path: str, channels: list[int] | None) -> dict[str, float]:
            reader = sitk.ImageFileReader()
            reader.SetFileName(path)
            reader.ReadImageInformation()
            data_shape = [reader.GetNumberOfComponents(), *reversed(reader.GetSize())]

            slab_length = _statistics_chunk_length(data_shape, 1)
            state: dict[str, Any] | None = None

            for start in range(0, data_shape[1], slab_length):
                slices: list[slice] = [slice(None)] * len(data_shape)
                slices[1] = slice(start, min(data_shape[1], start + slab_length))
                slab, _ = self._file_to_image_slice(name, path, tuple(slices))
                if channels is not None:
                    slab = slab[channels]
                state = _update_running_statistics(state, slab)

            return _finalize_running_statistics(state)

        def file_to_data(self, group: str, name: str) -> tuple[np.ndarray, Attribute]:
            path = self._resolve_data_path(name)
            if path is None:
                raise NameError(f"Data '{name}' not found in dataset '{self.filename}'.")
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
                raise NameError(f"Data '{name}' not found in dataset '{self.filename}'.")

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

        def file_to_data_statistics(
            self,
            group: str,
            name: str,
            channels: list[int] | None = None,
        ) -> dict[str, Any]:
            path = self._resolve_data_path(name)
            if path is None:
                raise NameError(f"Data '{name}' not found in dataset '{self.filename}'.")

            if path.endswith(".npy"):
                data = np.load(path, mmap_mode="r")
                if channels is not None:
                    data = data[channels]
                return _finalize_running_statistics(_update_running_statistics(None, data))

            if path.endswith((".itk.txt", ".fcsv", ".xml", ".vtk")):
                data, _ = self.file_to_data(group, name)
                if channels is not None:
                    data = data[channels]
                return _finalize_running_statistics(_update_running_statistics(None, data))

            if self._supports_region_read(path):
                return self._file_to_image_statistics(name, path, channels)

            # The whole volume lands in memory here, which the streamed path above exists to avoid. Nothing
            # better is left: a format that cannot serve a region would decode itself once per slab.
            image = sitk.ReadImage(path)
            data = sitk.GetArrayViewFromImage(image)
            if image.GetNumberOfComponentsPerPixel() == 1:
                data = np.expand_dims(data, 0)
            else:
                data = np.transpose(data, (len(data.shape) - 1, *list(range(len(data.shape) - 1))))
            if channels is not None:
                data = data[channels]
            return _finalize_running_statistics(_update_running_statistics(None, data))

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
                _retire_dead_debris(Path(final))
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
                data, attributes = self.file_to_data(group if group is not None else "", name)
                size = list(data.shape)
            return size, attributes

    class OmeZarrFile(AbstractFile):
        """OME-NGFF backend using chunked Zarr reads for KonfAI patches.

        ``level`` selects the multiscale pyramid resolution to read (0 = full
        resolution, higher = coarser); it comes from the ``omezarr@<level>``
        dataset-spec suffix.

        ``scale_factors`` is the WRITE-side counterpart: it makes the store this backend writes a
        pyramid instead of a single level. Reading indexes a pyramid BY POSITION, so a producer that
        writes one and a consumer that asks for ``@1`` are two halves of the same contract.
        """

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

        def _path(self, name: str, *, writing: bool = False) -> Path:
            base = Path(self.filename) / name
            if writing:
                return Path(f"{base}.ome.zarr")
            candidates = [Path(f"{base}.ome.zarr"), Path(f"{base}.zarr"), base]
            for candidate in candidates:
                if candidate.is_dir():
                    return candidate
            raise NameError(f"OME-Zarr group '{name}' not found in '{self.filename}'.")

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
            spacing = attributes.get_np_array("Spacing")
            direction = attributes.get_np_array("Direction").reshape(len(spacing), len(spacing))
            start_xyz = np.asarray([item.start for item in reversed(normalized[1:])], dtype=np.float64)
            step_xyz = np.asarray([item.step for item in reversed(normalized[1:])], dtype=np.float64)
            attributes["Origin"] = attributes.get_np_array("Origin") + direction @ (start_xyz * spacing)
            attributes["Spacing"] = spacing * step_xyz
            return data, attributes

        def bounded_region_reads(self, name: str) -> bool:
            del name
            return True  # zarr is chunked: a slice reads its chunks and nothing else

        def file_to_data_statistics(
            self,
            group: str,
            name: str,
            channels: list[int] | None = None,
        ) -> dict[str, Any]:
            shape, _ = self.get_infos(group, name)
            chunk_length = _statistics_chunk_length(shape, 1)
            state: dict[str, Any] | None = None
            for start in range(0, shape[1], chunk_length):
                slices = [slice(None)] * len(shape)
                slices[1] = slice(start, min(shape[1], start + chunk_length))
                chunk, _ = self.file_to_data_slice(group, name, tuple(slices))
                if channels is not None:
                    chunk = chunk[channels]
                state = _update_running_statistics(state, chunk)
            return _finalize_running_statistics(state)

        def data_to_file(
            self,
            name: str,
            data: sitk.Image | sitk.Transform | np.ndarray,
            attributes: Attribute | None = None,
        ) -> None:
            from konfai.utils.ome_zarr import write_ome_zarr

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
                raise
            shutil.rmtree(replaced, ignore_errors=True)
            _retire_dead_debris(final)

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
            final_path = self._path(name, writing=True)
            store_path = Path(f"{final_path}.{DataStream.temporary_suffix()}")
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

        def get_names(self, group: str) -> list[str]:
            return self.get_group()

        def get_group(self) -> list[str]:
            root = Path(self.filename)
            if not root.is_dir():
                return []
            groups = []
            for path in root.iterdir():
                if path.name.endswith(".ome.zarr"):
                    groups.append(path.name.removesuffix(".ome.zarr"))
                elif path.name.endswith(".zarr"):
                    groups.append(path.name.removesuffix(".zarr"))
            return sorted(groups)

        def is_exist(self, group: str, name: str | None = None) -> bool:
            try:
                self._path(f"{group}/{name}" if name else group)
                return True
            except NameError:
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

    class DicomFile(AbstractFile):
        """DICOM series backend with header-only metadata and slice-level reads."""

        def __init__(self, filename: str, read: bool) -> None:
            self.filename = filename if filename.endswith("/") else f"{filename}/"
            self.read = read

        def __enter__(self):
            return self

        def __exit__(self, exc_type, value, traceback):
            return None

        def _path(self, name: str) -> Path:
            return Path(self.filename) / name

        @staticmethod
        def _attributes(info: dict[str, Any]) -> Attribute:
            attributes = Attribute()
            attributes["Origin"] = np.asarray(info["origin"])
            attributes["Spacing"] = np.asarray(info["spacing"])
            attributes["Direction"] = np.asarray(info["direction"])
            attributes["SeriesInstanceUID"] = info["series_uid"]
            return attributes

        def file_to_data(self, group: str, name: str) -> tuple[np.ndarray, Attribute]:
            from konfai.utils.dicom import read_dicom_series

            data, origin, spacing, direction = read_dicom_series(self._path(name))
            attributes = Attribute()
            attributes["Origin"] = origin
            attributes["Spacing"] = spacing
            attributes["Direction"] = direction
            return data, attributes

        def bounded_region_reads(self, name: str) -> bool:
            del name
            return True  # one file per slice: a region decodes the slices it covers and nothing else

        def file_to_data_slice(self, group: str, name: str, slices: tuple[slice, ...]) -> tuple[np.ndarray, Attribute]:
            from konfai.utils.dicom import get_dicom_info, read_dicom_series_slice

            path = self._path(name)
            info = dict(get_dicom_info(path))  # copy: get_dicom_info is memoised, and we update it below
            data, origin, spacing, direction = read_dicom_series_slice(
                path, slices, series_uid=info["series_uid"], info=info
            )
            info.update(origin=origin, spacing=spacing, direction=direction)
            return data, self._attributes(info)

        def file_to_data_statistics(
            self,
            group: str,
            name: str,
            channels: list[int] | None = None,
        ) -> dict[str, Any]:
            from konfai.utils.dicom import get_dicom_info, read_dicom_series_slice

            path = self._path(name)
            info = get_dicom_info(path)
            shape = info["shape"]
            state: dict[str, Any] | None = None
            for index in range(shape[1]):
                chunk, _, _, _ = read_dicom_series_slice(
                    path,
                    (slice(None), slice(index, index + 1), slice(None), slice(None)),
                    series_uid=info["series_uid"],
                    info=info,
                )
                if channels is not None:
                    chunk = chunk[channels]
                state = _update_running_statistics(state, chunk)
            return _finalize_running_statistics(state)

        def data_to_file(
            self,
            name: str,
            data: sitk.Image | sitk.Transform | np.ndarray,
            attributes: Attribute | None = None,
        ) -> None:
            from konfai.utils.dicom import write_dicom_series

            attributes = attributes or Attribute()
            if sitk is not None and isinstance(data, sitk.Image):
                data, image_attributes = image_to_data(data)
                attributes.update(image_attributes)
            if not isinstance(data, np.ndarray):
                raise DatasetManagerError("DICOM datasets can only store scalar image arrays.")
            spacing = attributes.get_np_array("Spacing") if "Spacing" in attributes else np.ones(3)
            origin = attributes.get_np_array("Origin") if "Origin" in attributes else np.zeros(3)
            direction = attributes.get_np_array("Direction") if "Direction" in attributes else np.eye(3).flatten()
            metadata = {
                key: attributes[key]
                for key in ("PatientName", "PatientID", "Modality", "StudyInstanceUID", "SeriesInstanceUID")
                if key in attributes
            }
            write_dicom_series(
                self._path(name),
                data,
                spacing=spacing,
                origin=origin,
                direction=direction,
                metadata=metadata,
            )

        def get_names(self, group: str) -> list[str]:
            return self.get_group()

        def get_group(self) -> list[str]:
            root = Path(self.filename)
            if not root.is_dir():
                return []
            return sorted(path.name for path in root.iterdir() if path.is_dir() and self.is_exist(path.name))

        def is_exist(self, group: str, name: str | None = None) -> bool:
            from konfai.utils.dicom import get_dicom_info

            try:
                get_dicom_info(self._path(f"{group}/{name}" if name else group))
                return True
            except DatasetManagerError:
                return False

        def get_infos(self, group: str, name: str) -> tuple[list[int], Attribute]:
            from konfai.utils.dicom import get_dicom_info

            info = get_dicom_info(self._path(name))
            return info["shape"], self._attributes(info)

    class ItkTransformFile(AbstractFile):
        """ITK transform files, one ``<case>/<group>.h5`` per entry.

        The write side is the point: ``sitk.WriteTransform`` needs the whole field resident in
        float64, where the FILE is three HDF5 datasets that write by regions, so a displacement
        field streams into a transform any ITK consumer (Slicer first) loads. The read side hands
        back what ``Dataset.read_transform`` decodes: a displacement entry carries its field and
        the displacement marker; any other stored transform, the parameter rows and type keys of
        ``_encode_transform_leaves``.

        Needs ``h5py``, as the ``h5`` backend does: the whole point is to touch the parameters
        region by region, and a run whose peak memory turns on whether an optional import
        succeeded is a run nobody can size.
        """

        def __init__(self, filename: str, read: bool) -> None:
            if h5py is None:
                raise DatasetManagerError(
                    "An ':itktransform' dataset needs h5py.",
                    "Install it with: pip install konfai[hdf5]",
                )
            self.filename = filename
            self.read = read

        def __enter__(self):
            return self

        def __exit__(self, exc_type, value, traceback):
            pass

        def _path(self, name: str) -> str:
            for extension in ("h5", "tfm"):
                candidate = f"{self.filename}{name}.{extension}"
                if os.path.exists(candidate):
                    return candidate
            return f"{self.filename}{name}.h5"

        def file_to_data(self, group: str, name: str) -> tuple[np.ndarray, Attribute]:
            transform = sitk.ReadTransform(self._path(name))
            attributes = Attribute()
            if "DisplacementFieldTransform" in transform.GetName():
                field = sitk.DisplacementFieldTransform(transform).GetDisplacementField()
                data, attributes = image_to_data(field)
                attributes[DISPLACEMENT_FIELD_ATTRIBUTE] = "true"
                return data, attributes
            leaves = _encode_transform_leaves(transform, name, attributes)
            longest = max(len(leaf) for leaf in leaves)
            return (
                np.asarray([np.pad(leaf, (0, longest - len(leaf)), constant_values=np.nan) for leaf in leaves]),
                attributes,
            )

        def bounded_region_reads(self, name: str) -> bool:
            shape, _attributes = self.get_infos("", name)
            return len(shape) == 4 and shape[0] == 3

        def file_to_data_slice(self, group: str, name: str, slices: tuple[slice, ...]) -> tuple[np.ndarray, Attribute]:
            """A region of a displacement entry, decoded from the parameters it maps to alone.

            The buffer is ``[z][y][x]`` with the component fastest, so a span of leading-axis rows
            is one contiguous span of the parameters: read, reshaped, and sliced down to the exact
            region: the peak is the row span, never the field.
            """
            shape, attributes = self.get_infos(group, name)
            leading = slices[1].indices(shape[1]) if len(shape) == 4 else (0, 0, 1)
            if (
                len(shape) != 4
                or shape[0] != 3
                or DISPLACEMENT_FIELD_ATTRIBUTE not in attributes
                or not h5py.is_hdf5(self._path(name))
                or leading[2] < 0  # a reversed leading axis has no contiguous span to read
            ):
                data, attributes = self.file_to_data(group, name)
                return data[slices], attributes
            spatial = shape[1:]
            row = 3 * int(np.prod(spatial[1:], dtype=np.int64))
            with _open_h5(self._path(name), "r") as file:
                span = file["TransformGroup/0/TransformParameters"][leading[0] * row : leading[1] * row]
            block = np.moveaxis(span.reshape(leading[1] - leading[0], *spatial[1:], 3), -1, 0)
            # The span is the axis WITHOUT its step: rows start..stop were read whole, so the step
            # subsamples here, on the reshaped block.
            return (
                np.asarray(block[(slices[0], slice(None, None, leading[2]), *slices[2:])], dtype=np.float32),
                attributes,
            )

        def file_to_data_statistics(
            self,
            group: str,
            name: str,
            channels: list[int] | None = None,
        ) -> dict[str, Any]:
            data, _attributes = self.file_to_data(group, name)
            if channels is not None:
                data = data[channels]
            return _finalize_running_statistics(_update_running_statistics(None, data))

        def data_to_file(
            self,
            name: str,
            data: sitk.Image | sitk.Transform | np.ndarray,
            attributes: Attribute | None = None,
        ) -> None:
            os.makedirs(self.filename, exist_ok=True)
            # Always the `.h5` name: the content is HDF5 and ITK selects its transform IO from the
            # extension, so renaming it onto a resolved existing `.tfm` would corrupt that entry.
            final = os.path.join(self.filename, f"{name}.h5")
            staging = DataStream.staging_path(final)
            if isinstance(data, sitk.Transform):
                sitk.WriteTransform(data, staging)
            else:
                if isinstance(data, sitk.Image):
                    data, attributes = image_to_data(data)
                array = np.asarray(data)
                if attributes is None or array.ndim != 4 or array.shape[0] != 3:
                    raise DatasetManagerError(
                        f"An ':itktransform' entry is a 3-component 3-D displacement field; '{name}' has"
                        f" shape {list(array.shape)}.",
                        "Write the field itself (channel-first, with its geometry), or a sitk.Transform.",
                    )
                spatial = [int(extent) for extent in array.shape[1:]]
                file, parameters = _create_itk_transform_file(staging, spatial, attributes)
                with file:
                    parameters[:] = np.moveaxis(array.astype(np.float64), 0, -1).ravel()
            os.replace(staging, final)
            try:  # one entry per name: a `.tfm` left under the same stem would double it
                os.remove(os.path.join(self.filename, f"{name}.tfm"))
            except FileNotFoundError:
                pass

        def open_data_stream(
            self,
            name: str,
            shape: list[int],
            dtype: np.dtype,
            attributes: Attribute,
            region_shape: list[int] | None = None,
        ) -> DataStream | None:
            del dtype, region_shape  # the parameters are float64 whatever arrives, converted per slab
            if len(shape) != 4 or shape[0] != 3 or not is_an_image(attributes):
                return None
            os.makedirs(self.filename, exist_ok=True)
            spatial = [int(extent) for extent in shape[1:]]
            # The `.h5` name, as data_to_file: HDF5 content renamed onto a resolved `.tfm` is a
            # transform ITK reads with its text IO.
            final = os.path.join(self.filename, f"{name}.h5")
            staging = DataStream.staging_path(final)
            file, parameters = _create_itk_transform_file(staging, spatial, attributes)
            return _ItkTransformDataStream(file, parameters, staging, final, [3, *spatial])

        def _entries(self) -> list[str]:
            # Path.glob matches hidden files, so a writer's staging file is filtered out by name.
            return sorted(
                {
                    path.stem
                    for pattern in ("*.h5", "*.tfm")
                    for path in Path(self.filename).glob(pattern)
                    if not is_staging_entry(path.name)
                }
            )

        def get_names(self, group: str) -> list[str]:
            del group
            return self._entries()

        def get_group(self) -> list[str]:
            return self._entries()

        def is_exist(self, group: str, name: str | None = None) -> bool:
            return os.path.exists(self._path(name if name else group))

        def get_infos(self, group: str, name: str) -> tuple[list[int], Attribute]:
            # A legacy TEXT transform (`#Insight Transform File V1.0`) is served by the read side
            # too; only a real HDF5 file has the parameter datasets this fast path opens.
            if not h5py.is_hdf5(self._path(name)):
                data, attributes = self.file_to_data(group, name)
                return [int(extent) for extent in data.shape], attributes
            with _open_h5(self._path(name), "r") as file:
                kind = bytes(file["TransformGroup/0/TransformType"][0])
                fixed = np.asarray(file["TransformGroup/0/TransformFixedParameters"][()], dtype=np.float64)
            if not kind.startswith(b"DisplacementFieldTransform"):
                data, attributes = self.file_to_data(group, name)
                return [int(extent) for extent in data.shape], attributes
            attributes = Attribute()
            attributes["Origin"] = fixed[3:6]
            attributes["Spacing"] = fixed[6:9]
            attributes["Direction"] = fixed[9:18]
            attributes[DISPLACEMENT_FIELD_ATTRIBUTE] = "true"
            size_xyz = [int(extent) for extent in fixed[0:3]]
            return [3, *size_xyz[::-1]], attributes

    class File:
        def __init__(
            self,
            filename: str,
            read: bool,
            file_format: str,
            level: int = 0,
            scale_factors: list[int] | None = None,
            downsample_method: str | None = None,
        ) -> None:
            self.filename = filename
            self.read = read
            self.file: Dataset.AbstractFile | None = None
            self.file_format = file_format
            self.level = level
            self.scale_factors = scale_factors
            self.downsample_method = downsample_method

        def __enter__(self) -> Dataset.AbstractFile:
            if self.file_format == "h5":
                self.file = Dataset.H5File(self.filename, self.read)
            elif self.file_format == "omezarr":
                self.file = Dataset.OmeZarrFile(
                    self.filename, self.read, self.level, self.scale_factors, self.downsample_method
                )
            elif self.file_format == "dicom":
                self.file = Dataset.DicomFile(self.filename, self.read)
            elif self.file_format == "itktransform":
                self.file = Dataset.ItkTransformFile(self.filename + "/", self.read)
            else:
                self.file = Dataset.SitkFile(self.filename + "/", self.read, self.file_format)
            self.file.__enter__()
            return self.file

        def __exit__(self, exc_type, value, traceback):
            if self.file is not None:
                self.file.__exit__(exc_type, value, traceback)

    def __init__(
        self,
        filename: str | Path,
        file_format: str,
        scale_factors: list[int] | None = None,
        downsample_method: str | None = None,
    ) -> None:
        base_format, self.level = split_format_level(file_format)
        normalized_format = base_format.lower().removeprefix(".").replace("_", "-")
        file_format = {"ome-zarr": "omezarr", "zarr": "omezarr"}.get(normalized_format, normalized_format)
        self.filename, self.is_directory = Dataset._normalize_path(filename, file_format)
        self.file_format = file_format
        # The store backend is auto-detected from what is actually on disk (like SitkFile already probes
        # every supported extension): an OME-Zarr / Zarr / DICOM store is a directory whose type is
        # knowable from its structure, so a ``:mha`` token never forces it to be mis-read. The token then
        # only carries the WRITE format and the OME-Zarr pyramid level (``@N``).
        detected = Dataset._detect_directory_store_format(self.filename) if self.is_directory else None
        if detected is not None:
            self.file_format = detected
        # Write-side pyramid, declared by the Save/Write that owns this destination. Refused here
        # rather than ignored: only OME-NGFF has multiple levels, so a pyramid asked of an mha or an
        # h5 is a request the format cannot serve, and silently writing one level would leave the
        # consumer's ``@1`` resolving to a level that does not exist.
        if scale_factors and self.file_format != "omezarr":
            raise DatasetManagerError(
                f"A pyramid was asked of a '{self.file_format}' destination, which has no levels.",
                "Only ':omezarr' stores levels. Drop scale_factors, or write to ':omezarr'.",
            )
        self.scale_factors = list(scale_factors) if scale_factors else None
        self.downsample_method = downsample_method
        self._names_cache: dict[str, list[str]] = {}
        self._infos_cache: dict[tuple[str, str], tuple[list[int], Attribute]] = {}
        #: Facts a stage derived from an entry's pixels (a Crop's foreground box), keyed by
        #: ``(group, name)``: computed once per volume, whatever the number of chains reading it.
        self.case_facts: dict[tuple[str, str], dict[str, Any]] = {}

    @staticmethod
    def _normalize_path(filename: str | Path, file_format: str) -> tuple[str, bool]:
        # A single-store h5 is one file, every other backend a directory of cases: only the latter gets the
        # trailing slash that marks ``is_directory``. Keep the two in lock-step so a path never ends up a
        # directory-flagged h5 (which would write the hidden dotfile ``<dir>/.h5``). ``as_posix`` keeps the
        # separator forward on every OS, so the stored filename (and the trailing-slash marker) is the same
        # on Windows, where ``prefix / name`` would otherwise render backslashes.
        path = Path(filename).as_posix()
        if file_format != "h5" and not path.endswith("/"):
            path += "/"
        return path, path.endswith("/")

    def rebase(self, prefix: Path) -> None:
        """Prepend ``prefix`` to this dataset's path, re-deriving ``is_directory`` from the format."""
        self.filename, self.is_directory = Dataset._normalize_path(prefix / self.filename, self.file_format)

    @staticmethod
    def _detect_directory_store_format(root: str) -> str | None:
        """Detect a directory dataset's store backend from disk (``omezarr`` / ``dicom``), independent of the
        format token; ``None`` when it is plain per-file volumes (the SitkFile path, which auto-detects the
        extension itself). Probes the first case's entries only: cheap, and cases share one layout."""
        base = Path(root)
        if not base.is_dir():
            return None
        # No sort: any case is representative of the layout, and sorting lists the whole
        # directory for one probe (a resume over N written cases would pay it N times).
        case = next((child for child in base.iterdir() if child.is_dir()), None)
        if case is None:
            return None
        for entry in sorted(case.iterdir()):
            volume = directory_volume_form(entry)
            if volume is not None:
                return "dicom" if volume == "" else "omezarr"
        return None

    @property
    def path_on_disk(self) -> Path:
        """Where the store lives: its root directory, or the ``.h5`` file for a single-file store
        (named with or without the suffix, as the backend opens it)."""
        root = Path(self.filename)
        if self.file_format == "h5" and root.suffix != ".h5":
            return root.with_name(f"{root.name}.h5")
        return root

    def exists_on_disk(self) -> bool:
        return self.path_on_disk.exists()

    def concurrent_write_safe(self) -> bool:
        """Whether writes to different entries land in disjoint files, so a background writer may
        flush one entry while another thread writes elsewhere in the dataset.

        Mirrors the backend dispatch in ``File.__enter__``: everything that is not a single-store
        backend is a :class:`SitkFile` directory, one image file per ``(group, name)``. A single
        store (one HDF5 file, one zarr hierarchy, a DICOM series) shares handles and metadata across
        entries and must stay serial.
        """
        return self.file_format not in ("h5", "omezarr", "dicom")

    def _write_target(self, group: str, name: str) -> tuple[Dataset.File, str]:
        """The file a ``(group, name)`` write lands in and the entry name inside it, caches dropped.

        A directory dataset routes any sub-directory prefix of ``group`` into the file path (one file
        per case); a single store keeps one file and a ``group/name`` entry.
        """
        self._names_cache.clear()
        self._infos_cache.clear()
        self.case_facts.clear()
        if self.is_directory:
            os.makedirs(self.filename, exist_ok=True)
            s_group = group.split("/")
            if len(s_group) > 1:
                name = f"{'/'.join(s_group[:-1])}/{name}"
                group = s_group[-1]
            return (
                Dataset.File(
                    f"{self.filename}{name}",
                    False,
                    self.file_format,
                    self.level,
                    self.scale_factors,
                    self.downsample_method,
                ),
                group,
            )
        return (
            Dataset.File(
                self.filename, False, self.file_format, self.level, self.scale_factors, self.downsample_method
            ),
            f"{group}/{name}",
        )

    def write(
        self,
        group: str,
        name: str,
        data: sitk.Image | sitk.Transform | np.ndarray,
        attributes: Attribute | None = None,
    ) -> None:
        attributes = attributes if attributes is not None else Attribute()
        # A stage may hand back a volume without its channel axis (``Sum(dim=0)`` and
        # ``MergeLabels`` fold the leading axis, which in a TRANSFORM chain is the channel one).
        # An array with as many axes as the geometry has spatial axes IS a single-channel image,
        # and is written as one: read as channel-first it would be a 2-D image with a plane's worth
        # of channels, refused by ITK or, worse, stored that way in silence.
        if (
            isinstance(data, np.ndarray)
            and "Spacing" in attributes
            and data.ndim == len(attributes.get_np_array("Spacing"))
        ):
            data = data[None]
        target, entry = self._write_target(group, name)
        with target as file:
            file.data_to_file(entry, data, attributes)

    def can_stream_data(self, attributes: Attribute) -> bool:
        """Whether ``open_data_stream`` can serve this dataset's write format.

        H5 and OME-Zarr always can; MetaImage ``mha`` needs image geometry to write its header up
        front; every other format only writes whole volumes (use ``write``).
        """
        if self.file_format in ("h5", "omezarr"):
            return True
        if self.file_format == "itktransform":
            return is_an_image(attributes)
        return self.file_format in ("mha", "nii") and is_an_image(attributes)

    def open_data_stream(
        self,
        group: str,
        name: str,
        shape: list[int],
        dtype: np.dtype,
        attributes: Attribute | None = None,
        region_shape: list[int] | None = None,
    ) -> DataStream | None:
        """Open one entry for incremental region writes.

        Returns ``None`` when the write format cannot serve region writes; the caller then assembles
        the volume and uses ``write``. The returned stream is a context manager: a clean exit
        finalizes the entry, an exception removes the partial one.

        ``region_shape`` is the extent the caller will write at a time, channels included. A store
        that chunks on it never pays a read-modify-write; a store left to guess pays one on every
        region that straddles a chunk. Declaring it is the writer's job, it is the only party that
        knows its own access pattern.
        """
        if attributes is None:
            attributes = Attribute()
        file, entry = self._write_target(group, name)
        backend = file.__enter__()
        try:
            stream = backend.open_data_stream(entry, shape, dtype, attributes, region_shape)
        except BaseException:
            file.__exit__(None, None, None)
            raise
        if stream is None:
            file.__exit__(None, None, None)
            return None
        stream._file = file
        return stream

    def _case_path(self, sub_directory: str, name: str) -> str | None:
        """The file a directory dataset stores case ``name`` under, or ``None`` if absent on disk.

        The returned path omits the implicit ``.h5`` suffix h5 case files carry: ``H5File``
        re-appends it on open.
        """
        path = f"{self.filename}{sub_directory}{name}"
        return path if os.path.exists(f"{path}{'.h5' if self.file_format == 'h5' else ''}") else None

    def _resolve_entry(self, groups: str, name: str, action: Callable[[Dataset.AbstractFile, str, str], _T]) -> _T:
        """Run ``action`` on the open file holding ``(groups, name)``: THE place entry resolution lives.

        ``action`` receives the backend and the entry's coordinates INSIDE that file: a directory
        dataset stores one case per file, addressed by ``name``, with the entry keyed by the group
        path's last component, so the coordinates are ``("", group)`` there and ``(groups, name)``
        on a single-file dataset. Raises ``NameError`` when the dataset or the entry is missing.
        """
        if not self.exists_on_disk():
            raise NameError(f"Dataset {self.filename} not found")
        if self.is_directory:
            for sub_directory in self._get_sub_directories(groups):
                path = self._case_path(sub_directory, name)
                if path is not None:
                    with Dataset.File(path, True, self.file_format, self.level) as file:
                        return action(file, "", groups.split("/")[-1])
            raise NameError(f"Dataset entry '{groups}/{name}' not found in {self.filename}.")
        with Dataset.File(self.filename, True, self.file_format, self.level) as file:
            return action(file, groups, name)

    def read_data(self, groups: str, name: str) -> tuple[np.ndarray, Attribute]:
        return self._resolve_entry(groups, name, lambda file, group, entry: file.file_to_data(group, entry))

    def read_data_slice(self, groups: str, name: str, slices: tuple[slice, ...]) -> tuple[np.ndarray, Attribute]:
        return self._resolve_entry(
            groups, name, lambda file, group, entry: file.file_to_data_slice(group, entry, slices)
        )

    def iter_data_blocks(self, groups: str, name: str) -> Callable[[], Iterator[np.ndarray]]:
        """A factory of passes over one entry, block by block along the first spatial axis, each
        block about ``_STATISTICS_CHUNK_ELEMENTS`` elements: what a scan that must never hold the
        volume iterates. A store that cannot serve bounded region reads (gzipped NIfTI, compressed
        MetaImage) is read whole ONCE and kept for every pass the factory serves: decoding it once
        per block would cost more than holding it, and holding it once is what such a store costs
        anyway (the same policy as ``read_data_statistics``)."""
        shape, _ = self.get_infos(groups, name)
        if len(shape) < 2 or not self.bounded_region_reads(groups, name):
            resident: list[np.ndarray] = []

            def whole() -> Iterator[np.ndarray]:
                if not resident:
                    resident.append(self.read_data(groups, name)[0])
                yield resident[0]

            return whole
        rows = _statistics_chunk_length(shape, 1)

        def slabs() -> Iterator[np.ndarray]:
            for start in range(0, int(shape[1]), rows):
                slices = (
                    slice(None),
                    slice(start, min(int(shape[1]), start + rows)),
                    *(slice(None) for _ in shape[2:]),
                )
                yield self.read_data_slice(groups, name, slices)[0]

        return slabs

    def read_data_quantile(self, groups: str, name: str, q: float) -> Any:
        """``numpy.quantile(volume, q)`` (the default ``linear`` method, to the value) without
        holding the volume: bounded passes over :meth:`iter_data_blocks`."""
        low, high, weight = _order_statistics(self.iter_data_blocks(groups, name), float(q))
        if not np.issubdtype(np.asarray(low).dtype, np.inexact):
            # numpy.quantile promotes an integer input to float64 before it interpolates: the
            # difference of two order statistics would wrap on a narrow signed type, and an exact
            # index would answer in the stored dtype where numpy answers in float64.
            low, high = np.float64(low), np.float64(high)
        return _lerp_like_numpy(low, high, weight) if weight else low

    def bounded_region_reads(self, groups: str, name: str) -> bool:
        """Whether a region read of this entry decodes only the region, or the whole volume.

        What it prices is the ROUTE, never the answer: a store that decodes the whole volume once
        per slab (compressed MetaImage, NRRD, gzipped NIfTI) makes streaming read the source many
        times over, where loading reads it once. ``False`` for a missing entry: pessimistic, and
        only ever costing speed.
        """
        try:
            return self._resolve_entry(groups, name, lambda file, _, entry: file.bounded_region_reads(entry))
        except NameError:
            return False

    def read_data_statistics(
        self,
        groups: str,
        name: str,
        channels: list[int] | None = None,
    ) -> dict[str, Any]:
        return self._resolve_entry(
            groups, name, lambda file, group, entry: file.file_to_data_statistics(group, entry, channels)
        )

    def read_transform(self, group: str, name: str) -> sitk.Transform:
        if not self.exists_on_disk():
            raise NameError(f"Dataset {self.filename} not found")
        transform_parameters, attribute = self.read_data(group, name)
        if DISPLACEMENT_FIELD_ATTRIBUTE in attribute:
            # A displacement field carries no parameter vector to decode: the entry IS the transform,
            # and the store said so (NGFF RFC-5). float64 is what DisplacementFieldTransform requires.
            field = sitk.Cast(data_to_image(transform_parameters, attribute), sitk.sitkVectorFloat64)
            return sitk.DisplacementFieldTransform(field)
        transforms_type = [v for k, v in attribute.items() if k.endswith(":Transform_0")]
        transforms = []
        for i, transform_type in enumerate(transforms_type):
            transform = _decode_transform(transform_type, name)
            transform.SetFixedParameters(ast.literal_eval(attribute[f"{i}:FixedParameters"]))
            transform.SetParameters(tuple(transform_parameters[i]))
            transforms.append(transform)
        return sitk.CompositeTransform(transforms) if len(transforms) > 1 else transforms[0]

    def read_image(self, group: str, name: str) -> sitk.Image:
        data, attribute = self.read_data(group, name)
        return data_to_image(data, attribute)

    def get_size(self, group: str) -> int:
        return len(self.get_names(group))

    def is_group_exist(self, group: str) -> bool:
        return self.get_size(group) > 0

    def is_dataset_exist(self, group: str, name: str) -> bool:
        """Whether ``(group, name)`` is on disk, asked of disk at the moment it is asked.

        Deliberately NOT a slice of :meth:`get_names`: that listing is a planning-time snapshot, and a
        group the run itself produces (a ``Save`` writing into the dataset being read) gains cases
        while it is read, through a different ``Dataset`` object and, when the loader has workers, a
        different PROCESS. No memo can be invalidated across that boundary, so membership asks the disk.
        One entry, one probe: O(1) in the number of cases, where the listing is O(N) headers, and cheaper
        than the listing it replaces.
        """
        if not self.exists_on_disk():
            # A store that is not there yet holds nothing: the first probe of every fresh
            # destination, which a single-file backend would otherwise turn into an open error.
            return False
        if self.is_directory:
            # Not _resolve_entry: membership keeps scanning past a case file whose group is absent,
            # and answers False instead of raising.
            entry_group = group.split("/")[-1]
            for sub_directory in self._get_sub_directories(group):
                path = self._case_path(sub_directory, name)
                if path is None:
                    continue
                with Dataset.File(path, True, self.file_format, self.level) as file:
                    if file.is_exist(entry_group):
                        return True
            return False
        with Dataset.File(self.filename, True, self.file_format, self.level) as file:
            # A wildcard group is a path pattern; only the store's own listing expands it.
            return name in file.get_names(group) if "*" in group else file.is_exist(group, name)

    def _get_sub_directories(self, groups: str, sub_directory: str = ""):
        group = groups.split("/")[0]
        sub_directories = []
        if len(groups.split("/")) == 1:
            sub_directories.append(sub_directory)
        elif group == "*":
            for k in os.listdir(f"{self.filename}{sub_directory}"):
                if not os.path.isfile(f"{self.filename}{sub_directory}{k}"):
                    sub_directories.extend(
                        self._get_sub_directories(
                            "/".join(groups.split("/")[1:]),
                            f"{sub_directory}{k}/",
                        )
                    )
        else:
            sub_directory = f"{sub_directory}{group}/"
            if os.path.exists(f"{self.filename}{sub_directory}"):
                sub_directories.extend(self._get_sub_directories("/".join(groups.split("/")[1:]), sub_directory))
        return sub_directories

    def get_names(self, groups: str, index: list[int] | None = None) -> list[str]:
        if index is None and groups in self._names_cache:
            return self._names_cache[groups]

        names = []
        if self.is_directory:
            for sub_directory in self._get_sub_directories(groups):
                group = groups.split("/")[-1]
                if os.path.exists(f"{self.filename}{sub_directory}"):
                    for name in sorted(os.listdir(f"{self.filename}{sub_directory}")):
                        if os.path.isfile(f"{self.filename}{sub_directory}{name}") or self.file_format != "h5":
                            with Dataset.File(
                                f"{self.filename}{sub_directory}{name}",
                                True,
                                self.file_format,
                                self.level,
                            ) as file:
                                if file.is_exist(group):
                                    names.append(name.replace(".h5", "") if self.file_format == "h5" else name)
        else:
            with Dataset.File(self.filename, True, self.file_format, self.level) as file:
                names = file.get_names(groups)

        sorted_names = sorted(names)
        if index is None:
            self._names_cache[groups] = sorted_names
            return sorted_names
        return [name for i, name in enumerate(sorted_names) if i in index]

    def get_group(self) -> list[str]:
        if self.is_directory:
            if self.file_format in {"dicom", "omezarr"}:
                groups_set = set()
                root_path = Path(self.filename)
                for case_path in root_path.iterdir() if root_path.is_dir() else []:
                    if case_path.is_dir():
                        with Dataset.File(str(case_path), True, self.file_format, self.level) as dataset_file:
                            groups_set.update(dataset_file.get_group())
                return sorted(groups_set)
            groups_set = set()
            for root_dir, _, files in os.walk(self.filename):
                for file in files:
                    if file.startswith(".") or is_staging_entry(file):  # a staging write, or its crashed leftover
                        continue
                    path = Path(root_dir, file.split(".")[0]).relative_to(self.filename).as_posix()
                    parts = path.split("/")
                    if len(parts) >= 2:
                        del parts[-2]
                    groups_set.add("/".join(parts))
            groups = list(groups_set)
        else:
            with Dataset.File(self.filename, True, self.file_format, self.level) as dataset_file:
                groups = dataset_file.get_group()
        return list(groups)

    def get_infos(self, groups: str, name: str) -> tuple[list[int], Attribute]:
        # Memoize the header read (SITK reader + ReadImageInformation, or the HDF5/Zarr
        # metadata parse): get_infos is called once per name per group per build-pass at
        # setup, so caching it (like get_names) avoids re-parsing the same header N times.
        # Cache and hand back copies so a caller mutating the geometry cannot poison it.
        cache_key = (groups, name)
        cached = self._infos_cache.get(cache_key)
        if cached is not None:
            shape, attr = cached
            return list(shape), Attribute(attr)
        result = self._resolve_entry(groups, name, lambda file, group, entry: file.get_infos(group, entry))
        self._infos_cache[cache_key] = (list(result[0]), Attribute(result[1]))
        return result

    def get_statistics(self, groups: str) -> dict[str, dict[str, dict[str, float | list[float]]]]:
        names = self.get_names(groups)
        stats = {}
        for name in names:
            data, attr = self.read_data(groups, name)

            min_, max_ = data.min(), data.max()
            mean_ = data.mean()
            std_ = data.std()

            p25, p50, p75 = np.percentile(data, (25, 50, 75))

            stats[name] = {
                "min": float(min_),
                "max": float(max_),
                "mean": float(mean_),
                "std": float(std_),
                "25pc": float(p25),
                "50pc": float(p50),
                "75pc": float(p75),
                "shape": list(data.shape),
                "spacing": attr.get_np_array("Spacing").tolist(),
            }

        result: dict[str, dict[str, dict[str, Any]]] = {}
        result["case"] = {}
        for name, v in stats.items():
            for metric_name, value in v.items():
                if metric_name not in result["case"]:
                    result["case"][metric_name] = {}
                result["case"][metric_name][name] = value

        result["aggregates"] = {}
        tmp: dict[str, list[float]] = {}
        for _, v in stats.items():
            for metric_name, _ in v.items():
                if metric_name not in tmp:
                    tmp[metric_name] = []
                tmp[metric_name].append(v[metric_name])
        for metric_name, values in tmp.items():
            if isinstance(values[0], float):
                result["aggregates"][metric_name] = {
                    "max": float(np.nanmax(values)) if np.any(~np.isnan(values)) else np.nan,
                    "min": float(np.nanmin(values)) if np.any(~np.isnan(values)) else np.nan,
                    "std": float(np.nanstd(values)) if np.any(~np.isnan(values)) else np.nan,
                    "25pc": float(np.nanpercentile(values, 25)) if np.any(~np.isnan(values)) else np.nan,
                    "50pc": float(np.nanpercentile(values, 50)) if np.any(~np.isnan(values)) else np.nan,
                    "75pc": float(np.nanpercentile(values, 75)) if np.any(~np.isnan(values)) else np.nan,
                    "mean": float(np.nanmean(values)) if np.any(~np.isnan(values)) else np.nan,
                    "count": float(np.count_nonzero(~np.isnan(values))) if np.any(~np.isnan(values)) else np.nan,
                }
            else:
                p25, p50, p75 = np.nanpercentile(values, (25, 50, 75))

                result["aggregates"][metric_name] = {
                    "max": np.nanmax(values, axis=0).tolist(),
                    "min": np.nanmin(values, axis=0).tolist(),
                    "std": np.nanstd(values, axis=0).tolist(),
                    "mean": np.nanmean(values, axis=0).tolist(),
                }
        return result
