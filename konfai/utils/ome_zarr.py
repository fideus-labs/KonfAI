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

"""OME-Zarr (OME-NGFF) read/write backend for KonfAI, built on ``ngff-zarr``.

This module is a thin adapter: ``ngff-zarr`` owns all OME-NGFF metadata parsing,
multiscale handling, and (de)serialisation: KonfAI does not re-implement the
spec. We only

1. map between KonfAI's channel-first ``C[Z]YX`` arrays / ``(x, y, z)`` geometry
   and ngff-zarr's ``NgffImage`` (axis-named ``scale``/``translation``), and
2. round-trip KonfAI's full ``Attribute`` sidecar (including the ``Direction``
   matrix, which OME-NGFF cannot express) through a single ``konfai`` group
   attribute, read/written with ``zarr``.

Reads are lazy: ``ngff-zarr`` exposes the array as a chunked store, so slicing
only materialises the requested patch.

Optional dependencies: ``zarr`` + ``ngff-zarr`` (``pip install konfai[omezarr]``).
"""

from __future__ import annotations

import contextlib
import dataclasses
import itertools
import operator
import shutil
import threading
from collections import OrderedDict
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from konfai.utils.errors import DatasetManagerError
from konfai.utils.runtime import map_over_rank_pool

try:
    import zarr

    _ZARR_AVAILABLE = True
except ImportError:
    zarr = None  # type: ignore[assignment]
    _ZARR_AVAILABLE = False

try:
    # dask sits under the same guard because it is ngff-zarr's own hard dependency: the two are
    # present or absent together, and dask.array is used only to describe a store to ngff-zarr.
    import dask.array
    import ngff_zarr  # type: ignore[import-untyped]

    _NGFF_ZARR_AVAILABLE = True
except ImportError:
    dask = None  # type: ignore[assignment]
    ngff_zarr = None  # type: ignore[assignment]
    _NGFF_ZARR_AVAILABLE = False

_KONFAI_ATTR_KEY = "konfai"
_SPATIAL = ("z", "y", "x")


def _native_dtype(dtype: np.dtype) -> np.dtype:
    """``dtype`` in the machine's own byte order."""
    return dtype.newbyteorder("=") if dtype.byteorder == ">" else dtype


def _native_byteorder(array: np.ndarray) -> np.ndarray:
    """The same samples in the machine's own byte order.

    A store may hold big-endian samples (some acquisition software writes them, and zarr keeps the
    dtype it was given), and a non-native array poisons everything downstream in two different ways.
    ``torch.from_numpy`` refuses it outright ("given numpy array has byte order different from the
    native byte order"), which is the loud half. The quiet half is numpy: the flag rides along
    through slicing and arithmetic, so a value read here compares and writes correctly while any
    consumer that reinterprets the buffer (a raw ``.tobytes()``, a memory-mapped write, a C
    extension taking a pointer) sees the bytes swapped. Normalising once at the read boundary is
    what every caller would otherwise have to remember to do by hand.
    """
    return array.astype(_native_dtype(array.dtype), copy=False)


# NGFF RFC-5 types the component axis of a vector field, so a displacement field says what it is on
# disk. Those types only exist from NGFF 0.6 (zarr v3); 0.4 (zarr v2 layout) stays the default
# everywhere else, being the version portable across the whole CI matrix.
_DISPLACEMENT_AXIS_TYPE = "displacement"

_RFC5_VERSION = "0.6"
_DEFAULT_VERSION = "0.4"

#: How a chunk is sized, whether the store is created from a shape alone or from the region shape a
#: streamed writer declares (:func:`konfai.utils.dataset._store_chunks`). One rule, two callers: a
#: chunk is the unit a reader decompresses to reach one voxel, so an oversized one is paid by every
#: partial read forever, and by any consumer that is not KonfAI.
CHUNK_SPATIAL_TILE = 128
CHUNK_TARGET_BYTES = 32 << 20


def _zarr_v3_available() -> bool:
    """Whether the installed zarr can write a v3 store, which NGFF >= 0.5 (RFC-5) requires.

    RFC-5 axis types live only in NGFF 0.6, a zarr v3 layout, and zarr-python 3 needs Python >= 3.11 --
    so on Python 3.10 (zarr 2.x) a displacement field cannot be written. This is the capability the
    RFC-5 write actually depends on: ``coordinateSystems`` (checked in ``_type_component_axis``) tracks
    the ngff-zarr version, not the zarr one, so it alone lets a 2.x store through to an opaque failure.
    """
    if not _ZARR_AVAILABLE:
        return False
    try:
        return int(zarr.__version__.split(".")[0]) >= 3
    except (AttributeError, ValueError):
        return False


def _require_zarr() -> None:
    if not _ZARR_AVAILABLE:
        raise DatasetManagerError(
            "zarr is required for OME-Zarr support.",
            "Install it with: pip install konfai[omezarr]",
        )


def _require_ngff_zarr() -> None:
    _require_zarr()
    if not _NGFF_ZARR_AVAILABLE:
        raise DatasetManagerError(
            "ngff-zarr is required for OME-Zarr support.",
            "Install it with: pip install konfai[omezarr]",
        )


def _require_zarr_v3_for_rfc5() -> None:
    """Both write paths type a component axis, so both need the same capability check.

    Raised rather than silently downgraded to an untyped 0.4 store: a caller asking for a displacement
    field is asking for the one property that makes it readable as a transform, and a store that
    quietly is not one gets found out much later, by a reader that took its three channels for an
    image.
    """
    if not _zarr_v3_available():
        raise DatasetManagerError(
            "Writing an NGFF RFC-5 displacement field needs a zarr v3 store, i.e. "
            "zarr-python >= 3 (Python >= 3.11); this environment has zarr 2.",
            "Install it with: pip install 'zarr>=3' on Python >= 3.11.",
        )


def _read_konfai_attributes(store_path: str | Path) -> dict[str, Any]:
    """KonfAI's proprietary ``Attribute`` sidecar from the store, if present.

    A copy of a memoised read: it is metadata, and a streamed run asks for it once per region.
    """
    return dict(_konfai_attributes(str(store_path)))


@lru_cache(maxsize=8)
def _konfai_attributes(store_path: str) -> dict[str, Any]:
    try:
        group = zarr.open_group(store_path, mode="r")
        return dict(dict(group.attrs).get(_KONFAI_ATTR_KEY, {}).get("attributes", {}))
    except (KeyError, OSError, ValueError, TypeError):
        return {}


@lru_cache(maxsize=8)
def _load_image(store_path: str, level: int) -> Any:
    """Return the ``NgffImage`` for ``level`` of an OME-Zarr store, memoised per (store, level).

    A streamed run reads one patch per call, and re-parsing the NGFF metadata and rebuilding the
    lazy array graph per patch is pure per-read overhead: the image object is lazy (no voxel data),
    so a handful of them is cheap to keep. The key is the path alone, so anything that puts a
    different store at a path already read must call ``clear_ome_zarr_cache()``: see there.

    ``@N`` selects among the levels a store offers, so a single-level store has nothing to select: its
    one level is read whatever ``N`` says (as every other backend does: ``SitkFile`` ignores
    ``self.level`` too). Out of range on a store that IS a pyramid stays an error: asking level 3 of a
    three-level mask beside a four-level image is a real mismatch (it silently pairs 160 µm against
    320 µm), and quietly falling back to level 0 would hide it.
    """
    _require_ngff_zarr()
    try:
        multiscales = ngff_zarr.from_ngff_zarr(str(store_path))
    except (KeyError, IndexError, OSError, TypeError, ValueError) as exc:
        raise DatasetManagerError(
            f"Cannot open OME-Zarr store '{store_path}' (level {level}).",
            "Ensure the directory is a valid OME-NGFF store.",
        ) from exc
    if len(multiscales.images) == 1:
        return multiscales.images[0]
    if not 0 <= level < len(multiscales.images):
        # Its own message: the store is fine, the LEVEL is not. Reporting an out-of-range level as
        # "not a valid OME-NGFF store" sends the reader to inspect a store that has nothing wrong
        # with it: the mismatch is in what was asked of it.
        raise DatasetManagerError(
            f"OME-Zarr store '{store_path}' has {len(multiscales.images)} level(s); level {level} is out of range.",
            f"Ask for a level in 0..{len(multiscales.images) - 1} (0 is the finest).",
        )
    return multiscales.images[level]


def store_identity(store_path: str | Path) -> str:
    """The string a store is keyed by, wherever it is named.

    A reader keys its decoded chunks by the path it was handed, which
    :meth:`~konfai.utils.dataset.OmeZarrFile._path` builds with ``uri.join``, on forward slashes
    whatever the platform. A writer names the same store with a ``Path``, and on Windows
    ``str(Path)`` is backslashed, so the two spellings would not meet and a replaced store would keep
    serving the chunks of the store it replaced. One spelling, taken here.
    """
    return str(store_path).replace("\\", "/")


def clear_ome_zarr_cache(store_path: str | Path | None = None) -> None:
    """Forget the memoised NGFF images, so a store replaced on disk is parsed afresh.

    A named store forgets its own decoded chunks and read schedule and no one else's: an output
    store is created per case, and the inputs' chunks are what the next case reads. The metadata
    memos are cleared whatever the caller names, being one parse each to rebuild.

    The write paths here call it for their own output. Anything that materialises a store by other
    means (copying one over another, say) has to call it too: the memo is keyed on the path, and
    a hit serves the previous store's axes and geometry against the new store's voxels. That reads as
    a shape mismatch when the two differ, and as nothing at all when they do not.
    """
    _load_image.cache_clear()
    _level_path.cache_clear()
    _level_array.cache_clear()
    _konfai_attributes.cache_clear()
    if _CHUNK_CACHE is not None:
        _CHUNK_CACHE.forget(None if store_path is None else store_identity(store_path))


def is_displacement_field(store_path: str | Path) -> bool:
    """Whether the store declares its component axis as an NGFF RFC-5 displacement field.

    This is what lets a DVF be read back as a transform rather than as a 3-channel image, and it is
    read from the store itself: the producer does not have to be trusted, and no sidecar convention
    (a filename, an attribute) has to be agreed on separately.

    A store that predates RFC-5, or an ngff-zarr too old to model it, simply answers False: an
    unreadable or absent store is not a displacement field either, so this never raises.
    """
    if not _NGFF_ZARR_AVAILABLE:
        return False
    try:
        metadata = ngff_zarr.from_ngff_zarr(str(store_path)).metadata
    except Exception:
        # "Not a displacement field" is the only answer this owes: it is asked purely to decide HOW to
        # read an entry, and an absent or unreadable store is not one either.
        return False
    return any(
        axis.name == "c" and axis.type == _DISPLACEMENT_AXIS_TYPE
        for system in getattr(metadata, "coordinateSystems", None) or []
        for axis in system.axes
    )


def _canonical_shape(dims: Sequence[str], shape: Sequence[int]) -> list[int]:
    """Channel-first ``[C, (Z), Y, X]`` shape derived from ngff dims."""
    axis_size = dict(zip(dims, shape, strict=True))
    return [int(axis_size.get("c", 1)), *[int(axis_size[axis]) for axis in _SPATIAL if axis in axis_size]]


def _ordered(values: dict[str, float], dims: Sequence[str]) -> list[float]:
    return [float(values.get(axis, 1.0 if axis == "c" else 0.0)) for axis in dims]


#: A chunk no declared read will ask for again, so the first to go.
_NEVER_AGAIN = 1 << 62


class _ReadSchedule:
    """The chunks each of a caller's declared reads will touch, in the order it will read them.

    LRU is the best a cache can do without the future. With it, the fewest decodes any policy can
    reach is to evict the chunk whose next use is furthest away. Measured on a 513x1331x1776
    resample cut into 40 blocks over 126 chunks of source, chunks decoded on the level array: 148
    under LRU, 138 under this, 126 the floor.

    Followed only while the reads match what was declared, and abandoned at the first that does
    not: a caller that deviates loses the optimisation, never the answer.
    """

    def __init__(self, steps: list[frozenset[tuple]]) -> None:
        self._steps = steps
        self._cursor = -1
        self._uses: dict[tuple, list[int]] = {}
        for step, chunks in enumerate(steps):
            for coords in chunks:
                self._uses.setdefault(coords, []).append(step)
        self._passed: dict[tuple, int] = dict.fromkeys(self._uses, 0)
        self._reading = False

    def advance(self, chunks: frozenset[tuple]) -> bool:
        """Move to the read about to happen; ``False`` when it is not the one declared."""
        self._cursor += 1
        self._reading = True
        return self._cursor < len(self._steps) and self._steps[self._cursor] == chunks

    def finish(self) -> None:
        """The read :meth:`advance` moved to is over: what it touched is behind."""
        self._reading = False

    @property
    def done(self) -> bool:
        """Whether the read just begun is the last one declared."""
        return self._cursor >= len(self._steps) - 1

    def steps_to_next_use(self, coords: tuple) -> int:
        """How many reads away the next one touching ``coords`` is: 0 while the read touching it is
        in progress (a chunk of the window being assembled is not the one to evict for the rest of
        it), ``_NEVER_AGAIN`` when none does."""
        uses = self._uses.get(coords)
        if uses is None:
            return _NEVER_AGAIN
        behind = self._cursor if self._reading else self._cursor + 1
        passed = self._passed[coords]
        while passed < len(uses) and uses[passed] < behind:
            passed += 1
        self._passed[coords] = passed
        return uses[passed] - self._cursor if passed < len(uses) else _NEVER_AGAIN


class _DecodedChunkCache:
    """Decoded chunks of OME-Zarr arrays, kept whole, evicted under a byte cap.

    zarr 3 has no chunk cache of its own: every read decodes every chunk it touches, in full, and a
    streamed run touches the same chunks region after region -- a 31-row slab of a 256-row chunk
    grid decodes 8x the bytes it keeps, and the next slab decodes the same chunks again (measured:
    the same 2.1 GB of useful bytes cost 1.9 s in 256-row slabs and 28 s in 6-row slabs). Kept
    DECODED, so a hit is a memcpy; keyed by (array identity, chunk coordinates), so a store replaced
    on disk is a new identity and never served stale (see :func:`clear_ome_zarr_cache`). Same
    bytes, same order: a read through the cache is the read without it.
    """

    def __init__(self, capacity_bytes: int) -> None:
        self.capacity = int(capacity_bytes)
        self._entries: OrderedDict[tuple, np.ndarray] = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()
        self._schedules: dict[Any, _ReadSchedule] = {}

    def get(self, key: tuple) -> np.ndarray | None:
        with self._lock:
            chunk = self._entries.get(key)
            if chunk is not None:
                self._entries.move_to_end(key)
            return chunk

    def put(self, key: tuple, chunk: np.ndarray) -> None:
        if chunk.nbytes > self.capacity:
            return
        with self._lock:
            if key in self._entries:
                return
            self._entries[key] = chunk
            self._bytes += chunk.nbytes
            self._trim()

    def set_capacity(self, capacity_bytes: int) -> None:
        """Re-cap the cache, evicting down to the new ceiling."""
        with self._lock:
            self.capacity = int(capacity_bytes)
            self._trim()

    def schedule(self, identity: Any, steps: list[frozenset[tuple]]) -> None:
        """Declare the chunks the reads about to happen will touch, in order."""
        with self._lock:
            self._schedules[identity] = _ReadSchedule(steps)

    def begin(self, identity: Any, chunks: frozenset[tuple]) -> None:
        """The chunks the read about to happen touches, which is what advances its schedule."""
        with self._lock:
            schedule = self._schedules.get(identity)
            if schedule is not None and (not schedule.advance(chunks) or schedule.done):
                del self._schedules[identity]  # deviated from, or read to its end: nothing left to plan

    def end(self, identity: Any) -> None:
        """The read begun on ``identity`` is over: what it touched competes on its next use again."""
        with self._lock:
            schedule = self._schedules.get(identity)
            if schedule is not None:
                schedule.finish()

    def forget(self, store_path: str | None = None) -> None:
        """Drop what one store put here, or everything when no store is named."""
        with self._lock:
            if store_path is None:
                self._entries.clear()
                self._schedules.clear()
                self._bytes = 0
                return
            for key in [key for key in self._entries if key[0][0] == store_path]:
                self._bytes -= self._entries.pop(key).nbytes
            for identity in [identity for identity in self._schedules if identity[0] == store_path]:
                del self._schedules[identity]

    def _trim(self) -> None:
        """Down to the ceiling: the chunk whose next use is furthest away, the least recently used
        while nothing is declared. Held under ``self._lock``."""
        while self._bytes > self.capacity and self._entries:
            key = self._furthest() if self._schedules else next(iter(self._entries))
            self._bytes -= self._entries.pop(key).nbytes

    def _furthest(self) -> tuple:
        """The chunk whose next use is furthest: a declared chunk by its schedule, an undeclared one
        by its recency, the ``n``-th most recently used taken as ``n`` reads away (the LRU order it
        competes in: a companion volume read beside a declared source, a mask, has no schedule and
        is read again at the next region all the same), a declared chunk nothing reads again first
        of all. The older wins a tie. Measured on a 48-chunk source swept through a rotated resample
        with a mask on the same grid read beside it, at a cache of 64 chunks: 66 + 66 decodes under
        LRU, 44 + 66 with the companion ranked never-again, 44 + 44 with it ranked by recency."""
        newer = len(self._entries)
        furthest, distance = next(iter(self._entries)), 0
        for key in self._entries:
            newer -= 1
            steps = self._steps_to_next_use(key, undeclared=newer + 1)
            if steps > distance:
                furthest, distance = key, steps
        return furthest

    def _steps_to_next_use(self, key: tuple, undeclared: int) -> int:
        identity, coords = key
        schedule = self._schedules.get(identity)
        return undeclared if schedule is None else schedule.steps_to_next_use(coords)


#: The least the cache is worth: below a chunk or two of a large store, a region touching more
#: chunks than it holds decodes them again. It is what an undeclared budget's share is raised to; a
#: declared budget never gives the cache more than its share, and the plan says when that is under
#: the floor. A cache allowed the floor out of a 128 MiB budget took the budget whole, while the
#: sweep went on sizing its regions against the same 128 MiB: the run was priced at 2x its budget.
CHUNK_CACHE_FLOOR = 256 << 20
#: The share of a DECLARED per-rank budget the cache may take. A third, because a budget is spent on
#: what the chain holds as well, and the cache is the cheapest of the three to give up.
_CHUNK_CACHE_BUDGET_SHARE = 1 / 3


def bound_chunk_cache() -> int:
    """Resize the decoded-chunk cache to the budget this rank published
    (:func:`~konfai.utils.budget.set_per_rank_budget`) and answer its capacity from now on."""
    capacity = chunk_cache_capacity()
    if _CHUNK_CACHE is not None:
        _CHUNK_CACHE.set_capacity(capacity)
    return capacity


def chunk_cache_capacity() -> int:
    """What the cache may hold: its share of the declared budget, never more, so a budget under the
    floor bounds the cache instead of the floor taking the budget whole; a share of what this process
    may allocate when nothing was declared, where the floor is what makes the cache worth having."""
    from konfai.utils.budget import available_memory_bytes, per_rank_budget_bytes

    declared = per_rank_budget_bytes()
    if declared is not None:
        return int(declared * _CHUNK_CACHE_BUDGET_SHARE)
    return max(CHUNK_CACHE_FLOOR, int(available_memory_bytes()[0] * 0.05))


_CHUNK_CACHE: _DecodedChunkCache | None = None


def _chunk_cache() -> _DecodedChunkCache:
    global _CHUNK_CACHE
    if _CHUNK_CACHE is None:
        _CHUNK_CACHE = _DecodedChunkCache(chunk_cache_capacity())
    return _CHUNK_CACHE


@lru_cache(maxsize=8)
def _level_array(store_path: str, level_path: str) -> Any:
    """The zarr array behind one level, opened once: chunk-wise reads go to it directly, not through
    the dask graph ngff-zarr wraps it in (which rebuilds a task per chunk per read)."""
    _require_zarr()
    return zarr.open_group(store_path, mode="r")[level_path]


def _normalized_selection(index: tuple, shape: Sequence[int]) -> tuple[list[slice], list[int]]:
    """``index`` as one unit-step slice per axis, and the axes an integer selection squeezes out."""
    selections: list[slice] = []
    squeeze: list[int] = []
    for axis, item in enumerate(index):
        if isinstance(item, int):
            selections.append(slice(item, item + 1))
            squeeze.append(axis)
        else:
            selections.append(slice(*item.indices(shape[axis])))
    return selections, squeeze


def _touched_chunks(selections: Sequence[slice], chunks: Sequence[int]) -> list[tuple]:
    """The chunk coordinates a normalised selection covers."""
    ranges = [
        range(sel.start // ch, max(sel.start, sel.stop - 1) // ch + 1) if sel.stop > sel.start else range(0)
        for sel, ch in zip(selections, chunks, strict=True)
    ]
    return list(itertools.product(*ranges))


def _read_chunked(store_path: str, level_path: str, array: Any, index: tuple) -> np.ndarray:
    """``array[index]`` assembled chunk by chunk through the decoded-chunk cache.

    Only integer and slice selections with unit step reach here (the reader normalises to those);
    each touched chunk is served from the cache or decoded whole and cached, then the requested
    window is copied out of it. Values are exactly what ``array[index]`` returns.
    """
    cache = _chunk_cache()
    shape, chunks = array.shape, array.chunks
    selections, squeeze = _normalized_selection(index, shape)
    out_shape = tuple(max(0, sel.stop - sel.start) for sel in selections)
    # In the machine's byte order from the start: a big-endian store is converted by the copy that
    # assembles the window, where a pass of its own over the result is a second walk of every byte.
    out = np.empty(out_shape, dtype=_native_dtype(array.dtype))
    identity = (store_identity(store_path), level_path)
    wanted = _touched_chunks(selections, chunks)
    # Begun before an empty selection returns: plan_ome_zarr_reads declared that read too, and a
    # schedule that misses one step ranks every later chunk against the wrong read.
    cache.begin(identity, frozenset(wanted))
    try:
        if wanted:
            _assemble_window(cache, identity, array, selections, wanted, out)
    finally:
        cache.end(identity)
    return out.squeeze(axis=tuple(squeeze)) if squeeze else out


def _assemble_window(
    cache: _DecodedChunkCache,
    identity: tuple,
    array: Any,
    selections: Sequence[slice],
    wanted: list[tuple],
    out: np.ndarray,
) -> None:
    """Fill ``out`` with the ``wanted`` chunks of ``array``, served from ``cache`` or decoded whole
    and cached."""
    shape, chunks = array.shape, array.chunks
    missing = [coords for coords in wanted if cache.get((identity, coords)) is None]
    if missing:
        # ONE zarr read for the chunk-aligned hull of what is missing: zarr decodes the chunks of a
        # single selection in parallel, and one call per chunk would serialise them (measured 1.5x
        # slower than the plain read on a cold pass). The hull may cover chunks already cached
        # when the misses are sparse; those are decoded again -- a bounded waste on a rare shape.
        lo = [min(c[axis] for c in missing) for axis in range(len(chunks))]
        hi = [max(c[axis] for c in missing) for axis in range(len(chunks))]
        hull = tuple(
            slice(lo_ * ch, min((hi_ + 1) * ch, extent))
            for lo_, hi_, ch, extent in zip(lo, hi, chunks, shape, strict=True)
        )
        decoded = np.asarray(array[hull])
        for coords in itertools.product(*(range(lo_, hi_ + 1) for lo_, hi_ in zip(lo, hi, strict=True))):
            key = (identity, coords)
            if cache.get(key) is not None:
                continue
            piece = tuple(
                slice((c - lo_) * ch, min((c - lo_ + 1) * ch, extent - lo_ * ch))
                for c, lo_, ch, extent in zip(coords, lo, chunks, shape, strict=True)
            )
            cache.put(key, np.ascontiguousarray(decoded[piece], dtype=out.dtype))
        del decoded

    def place(coords: tuple) -> None:
        chunk = cache.get((identity, coords))
        if chunk is None:  # evicted between the fill and the read (cache smaller than one hull)
            window = tuple(
                slice(c * ch, min((c + 1) * ch, extent)) for c, ch, extent in zip(coords, chunks, shape, strict=True)
            )
            chunk = np.asarray(array[window])
        # Where this chunk lands in the output, and which part of it.
        src: list[slice] = []
        dst: list[slice] = []
        for c, ch, sel in zip(coords, chunks, selections, strict=True):
            lo = max(sel.start, c * ch)
            hi = min(sel.stop, (c + 1) * ch)
            src.append(slice(lo - c * ch, hi - c * ch))
            dst.append(slice(lo - sel.start, hi - sel.start))
        out[tuple(dst)] = chunk[tuple(src)]

    # One chunk per worker: a chunk owns its window of the output, so the destinations are disjoint,
    # and numpy releases the GIL for the copy.
    map_over_rank_pool(place, wanted)


@lru_cache(maxsize=8)
def _level_path(store_path: str, level: int) -> str | None:
    """The zarr path of one level, from the store's multiscales metadata, memoised beside the image."""
    try:
        datasets = ngff_zarr.from_ngff_zarr(store_path).metadata.datasets
        return str(datasets[level if len(datasets) > 1 else 0].path)
    except Exception:
        return None


def _read_level_window(store_path: str, level: int, image: Any, index: tuple) -> np.ndarray:
    """The window through the decoded-chunk cache when the level's array can be opened directly, else
    the plain lazy read: same values either way."""
    level_path = _level_path(store_path, level)
    if level_path is not None:
        # Any failure to open the level directly means: read through the lazy array, as before.
        with contextlib.suppress(Exception):
            array = _level_array(store_path, level_path)
            if tuple(array.shape) == tuple(image.data.shape):
                return _read_chunked(store_path, level_path, array, index)
    return np.asarray(image.data[index])


def _store_index(
    dims: Sequence[str], canonical_shape: Sequence[int], slices: tuple[slice, ...], timepoint: int
) -> tuple[int | slice, ...]:
    """A KonfAI channel-first ``C[Z]YX`` selection as one per axis of the store, in its own order."""
    if len(slices) != len(canonical_shape):
        raise DatasetManagerError(f"Expected {len(canonical_shape)} slices, got {len(slices)}.")
    normalized = [slice(*item.indices(size)) for item, size in zip(slices, canonical_shape, strict=True)]
    spatial = dict(zip([axis for axis in _SPATIAL if axis in dims], normalized[1:], strict=True))
    return tuple(
        timepoint if axis == "t" else normalized[0] if axis == "c" else spatial.get(axis, slice(None)) for axis in dims
    )


def plan_ome_zarr_reads(
    store_path: str | Path, windows: Sequence[tuple[slice, ...]], *, level: int = 0, timepoint: int = 0
) -> None:
    """Declare the ``C[Z]YX`` windows about to be read from a store, in the order they will be read.

    What the decoded-chunk cache does with it: evict the chunk whose next declared use is furthest
    away instead of the least recently used, which is the fewest decodes any policy can reach. The
    gain is the whole of it where a miss is expensive -- a tight cache, or a store across a network,
    where a miss is a download. Declaring nothing costs nothing: the cache stays LRU.
    """
    if not _NGFF_ZARR_AVAILABLE or not windows:
        return
    with contextlib.suppress(Exception):  # a store this cannot read is simply not scheduled
        image = _load_image(str(store_path), level)
        level_path = _level_path(str(store_path), level)
        if level_path is None:
            return
        array = _level_array(str(store_path), level_path)
        dims = [str(axis).lower() for axis in image.dims]
        canonical_shape = _canonical_shape(dims, image.data.shape)
        steps = [
            frozenset(
                _touched_chunks(
                    _normalized_selection(_store_index(dims, canonical_shape, window, timepoint), array.shape)[0],
                    array.chunks,
                )
            )
            for window in windows
        ]
        _chunk_cache().schedule((store_identity(store_path), level_path), steps)


def read_ome_zarr_data_slice(
    store_path: str | Path,
    slices: tuple[slice, ...],
    *,
    level: int = 0,
    timepoint: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Read a KonfAI channel-first ``C[Z]YX`` patch from an OME-Zarr store (lazy)."""
    image = _load_image(str(store_path), level)
    dims = [str(axis).lower() for axis in image.dims]
    canonical_shape = _canonical_shape(dims, image.data.shape)
    index = _store_index(dims, canonical_shape, slices, timepoint)
    patch = _read_level_window(str(store_path), level, image, index)
    remaining = [axis for axis, selection in zip(dims, index, strict=True) if not isinstance(selection, int)]
    wanted = [axis for axis in ("c", *_SPATIAL) if axis in remaining]
    patch = np.transpose(patch, [remaining.index(axis) for axis in wanted])
    if "c" not in remaining:
        patch = patch[np.newaxis]

    metadata = {
        "axes": dims,
        "shape": canonical_shape,
        "chunks": list(getattr(image.data, "chunks", []) or []),
        "dtype": str(image.data.dtype),
        "scale": _ordered(dict(image.scale), dims),
        "translation": _ordered(dict(image.translation), dims),
        "attributes": _read_konfai_attributes(store_path),
    }
    return _native_byteorder(np.asarray(patch)), metadata


def _spatial_geometry(
    ndim: int,
    shape_label: str,
    spacing: Sequence[float] | None,
    origin: Sequence[float] | None,
) -> tuple[list[str], list[float], list[float]]:
    """The NGFF spatial axes of a channel-first array with ``ndim`` dims, and its per-axis
    scale/translation in axis order: geometry arrives ``(x, y, z)`` (SimpleITK order)."""
    if ndim not in {3, 4}:
        raise DatasetManagerError(f"OME-Zarr writing expects a C-Y-X or C-Z-Y-X array, got {shape_label}.")
    spatial_axes = ["y", "x"] if ndim == 3 else ["z", "y", "x"]
    dimension = len(spatial_axes)
    spacing_xyz = list(spacing if spacing is not None else [1.0] * dimension)
    origin_xyz = list(origin if origin is not None else [0.0] * dimension)
    if len(spacing_xyz) != dimension or len(origin_xyz) != dimension:
        raise DatasetManagerError(
            f"OME-Zarr geometry must contain {dimension} spacing and origin values for {shape_label}."
        )
    coordinate = {"x": (spacing_xyz[0], origin_xyz[0]), "y": (spacing_xyz[1], origin_xyz[1])}
    if dimension == 3:
        coordinate["z"] = (spacing_xyz[2], origin_xyz[2])
    scale_values = [float(coordinate[axis][0]) for axis in spatial_axes]
    translation_values = [float(coordinate[axis][1]) for axis in spatial_axes]
    return spatial_axes, scale_values, translation_values


def _downsample_method(downsample_method: str | None) -> Any:
    """Resolve a downsampling method name to ngff-zarr's enum, defaulting to BIN_SHRINK.

    NOT ngff-zarr's own default, which is ``ITKWASM_GAUSSIAN``: a pyramid is indexed by position and
    read as "the same image, coarser", so a level that has been smoothed is a change of pixels that
    no reader can see. Measured on a real volume, the gaussian keeps a 0.9998 correlation while
    crushing the peak intensity by 20 %: the shape of difference that passes a sanity check and
    resurfaces months later. ``ITKWASM_BIN_SHRINK`` is a plain block mean, so a caller that already
    downsamples by averaging blocks gets the same voxels from this writer.
    """
    _require_ngff_zarr()
    if downsample_method is None:
        return ngff_zarr.Methods.ITKWASM_BIN_SHRINK
    try:
        return ngff_zarr.Methods[downsample_method]
    except KeyError:
        raise DatasetManagerError(
            f"Unknown downsample_method '{downsample_method}'.",
            f"Use one of: {', '.join(sorted(m.name for m in ngff_zarr.Methods))}.",
        ) from None


def _level_zero_scale_factors(scale_factors: Sequence[int]) -> list[int]:
    """KonfAI's factors are per level (``[4, 4]``: each level a quarter of the one above);
    ngff-zarr's are each relative to level 0, so ``[4, 4]`` there writes two identical levels."""
    return list(itertools.accumulate((int(factor) for factor in scale_factors), operator.mul))


def write_ome_zarr(
    store_path: str | Path,
    data: np.ndarray,
    *,
    spacing: Sequence[float] | None = None,
    origin: Sequence[float] | None = None,
    attributes: dict[str, Any] | None = None,
    chunks: Sequence[int] | None = None,
    displacement_field: bool = False,
    scale_factors: Sequence[int] | None = None,
    downsample_method: str | None = None,
) -> None:
    """Write one channel-first KonfAI array as an OME-NGFF store, single-level or a pyramid.

    ``scale_factors`` makes it a pyramid, each factor shrinking the level above it: ``[4]`` writes
    level 0 plus level 0 shrunk 4x per spatial axis, ``[4, 4]`` adds a third at 16x. (ngff-zarr's
    own argument is spelled relative to level 0; :func:`_level_zero_scale_factors` converts, so
    ``[4, 4]`` never writes the same level twice.) Consumers index a pyramid BY POSITION, so the
    order is the contract: 0 finest. Each level carries its OWN scale and translation, and ngff-zarr shifts the
    coarse origin by half the spacing delta, which is the centre-of-voxel convention these stores
    use; getting that wrong biases every voxel by a fraction of a coarse voxel and still looks like a
    plausible image. ``downsample_method`` selects how (see :func:`_downsample_method`).

    ``displacement_field`` writes it as a vector FIELD rather than an image: the component axis is
    typed ``displacement`` (NGFF RFC-5), which is what makes a registration DVF self-describing
    instead of an anonymous 3-channel image. A reader then no longer has to be told out of band that
    the channels are a displacement: the mistake that path invites is silent, not loud: index the
    component axis like any other and you get one third of the field back, and a plausible-looking
    registration with it.

    The NGFF version follows from that flag and is deliberately NOT a parameter. RFC-5 axis types
    exist only from 0.6, so a caller passing both could only ever pass them consistently: an
    invariant worth removing rather than documenting.
    """
    array_data = np.asarray(data)
    # The one write path: the store described and created empty (ngff-zarr's metadata, the
    # caller's chunking), filled by zarr itself, its levels grafted beside level 0. Handing
    # ngff-zarr the resident array instead went through dask -- a full rechunk into a 128 MB block
    # per task -- at a sixth of the throughput (14.4 s vs 2.3 s for a 2.1 GB volume, measured).
    array = create_ome_zarr_store(
        store_path,
        array_data.shape,
        array_data.dtype,
        spacing=spacing,
        origin=origin,
        attributes=attributes,
        chunks=chunks,
        displacement_field=displacement_field,
    )
    array[...] = array_data
    if scale_factors:
        append_ome_zarr_levels(store_path, scale_factors, downsample_method=downsample_method)


def update_konfai_attributes(store_path: str | Path, extra: dict[str, Any]) -> None:
    """Merge ``extra`` into the store's KonfAI attribute sidecar, keeping what is already there.

    For facts that are only knowable once the last region has landed (a streamed field's own bound
    being the case that motivated it), so they can still be recorded without holding the volume.
    """
    if not extra:
        return
    _require_zarr()
    group = zarr.open_group(str(store_path), mode="r+")
    sidecar = dict(dict(group.attrs).get(_KONFAI_ATTR_KEY, {}).get("attributes", {}))
    sidecar.update(extra)
    # Writing through an `r+` group updates the consolidated copy with it, so a sidecar landing here
    # is readable by a consolidated reader without a second consolidation pass. Measured, because the
    # opposite is the plausible assumption: a foreign `zarr.open_group(mode="r")` reads back a bound
    # written this way on both sides of this line.
    group.attrs[_KONFAI_ATTR_KEY] = {"attributes": sidecar}
    clear_ome_zarr_cache(store_path)


def _type_component_axis(multiscales: Any, axis_type: str) -> None:
    """Type the ``c`` axis of every coordinate system, in place.

    The axis type is set on the RFC-5 coordinate systems rather than on the ``NgffImage``, because
    ``to_multiscales`` derives the axes itself and hardcodes ``type="channel"`` for a ``c`` dim --
    tagging the image is a dead assignment on a non-frozen dataclass, and the store comes out an
    ordinary 3-channel image with no error raised anywhere.

    Coordinate systems are also the capability check: an ngff-zarr too old to model RFC-5 has no
    ``coordinateSystems`` on its metadata, and would otherwise write a silently untyped store.
    """
    systems = getattr(multiscales.metadata, "coordinateSystems", None)
    if not systems:
        raise DatasetManagerError(
            f"Writing a '{axis_type}' field needs NGFF RFC-5 coordinate systems, which this ngff-zarr cannot model.",
            "Upgrade it with: pip install 'ngff-zarr>=0.38'",
        )
    for system in systems:
        for axis in system.axes:
            if axis.name == "c":
                axis.type = axis_type


def create_ome_zarr_store(
    store_path: str | Path,
    shape: Sequence[int],
    dtype: Any,
    *,
    spacing: Sequence[float] | None = None,
    origin: Sequence[float] | None = None,
    attributes: dict[str, Any] | None = None,
    chunks: Sequence[int] | None = None,
    displacement_field: bool = False,
) -> Any:
    """Create an empty single-level OME-NGFF store for region-by-region writes.

    Returns the level-0 zarr array: chunks materialise as regions are assigned, and unwritten regions
    read back as zeros. Metadata is complete from the start, so the store is readable at any point
    during the write.

    ngff-zarr writes that metadata, exactly as it does for the whole-array path, so both paths describe
    a store the same way: ``displacement_field`` included, which is the point of routing it through
    ngff-zarr at all. A field too large to assemble in memory is written region by region, so this is
    the ONLY path a real one takes, and until it could type its component axis a DVF came out
    self-describing exactly when it was small enough not to need to be.

    It describes the store from a one-voxel stand-in rather than from an array of the target shape.
    With a single resolution level the metadata does not depend on the extent at all (axes and
    coordinate transformations come from ``dims``, ``scale`` and ``translation``), and the two come
    out byte-identical, verified for 0.4 and 0.6. Handing ngff-zarr the real shape instead costs a
    pass over every chunk of an array that is entirely zeros (~44 ms per chunk, ~33 s for a 13.6 GiB
    field) to write no bytes at all. The real array is then created underneath that metadata, which is
    also what makes its chunking exactly the caller's: the region grid is the one thing ngff-zarr
    cannot infer, and a store whose chunks straddle it turns every region write into a
    read-modify-write.
    """
    clear_ome_zarr_cache(store_path)
    _require_ngff_zarr()
    spatial_axes, scale_values, translation_values = _spatial_geometry(
        len(shape), f"shape {list(shape)}", spacing, origin
    )
    dims = ["c", *spatial_axes]
    scale = {"c": 1.0, **dict(zip(spatial_axes, scale_values, strict=True))}
    translation = {"c": 0.0, **dict(zip(spatial_axes, translation_values, strict=True))}

    if chunks is None:
        spatial_chunks = [min(extent, CHUNK_SPATIAL_TILE) for extent in shape[1:]]
        # Keep one chunk near the target: full spatial tiles, channels split to fit the budget.
        tile_bytes = int(np.prod(spatial_chunks, dtype=np.int64)) * np.dtype(dtype).itemsize
        chunks = [min(shape[0], max(1, CHUNK_TARGET_BYTES // max(1, tile_bytes))), *spatial_chunks]
    chunks = tuple(chunks)

    stand_in = dask.array.zeros((shape[0], *(1,) * len(spatial_axes)), dtype=np.dtype(dtype))
    image = ngff_zarr.to_ngff_image(stand_in, dims=dims, scale=scale, translation=translation)
    multiscales = ngff_zarr.to_multiscales(image, scale_factors=[])
    version = _DEFAULT_VERSION
    if displacement_field:
        _require_zarr_v3_for_rfc5()
        _type_component_axis(multiscales, _DISPLACEMENT_AXIS_TYPE)
        version = _RFC5_VERSION
    # version is explicit because to_ngff_zarr defaults to 0.5, which zarr-python 2 cannot write.
    ngff_zarr.to_ngff_zarr(str(store_path), multiscales, overwrite=True, version=version)

    # The level-0 key comes from the metadata rather than a literal: ngff-zarr builds it from the
    # image name, so "scale0/image" is its convention to change, not ours to hardcode.
    group = zarr.open_group(str(store_path), mode="r+")
    create_array = getattr(group, "create_array", None) or group.create_dataset
    array = create_array(
        multiscales.metadata.datasets[0].path,
        shape=tuple(shape),
        chunks=chunks,
        dtype=np.dtype(dtype),
        fill_value=0,
        overwrite=True,
    )
    # Sidecar last: to_ngff_zarr(overwrite=True) reopens the root with mode="w" and drops every
    # attribute it finds, so writing this first loses Direction, and losing Direction is silent,
    # the reader falls back to identity and returns a plausibly-oriented volume.
    if attributes:
        group.attrs[_KONFAI_ATTR_KEY] = {"attributes": dict(attributes)}
    # ngff-zarr leaves a consolidated index behind, and readers trust it over the arrays themselves --
    # so until it is rebuilt the store still advertises the one-voxel stand-in, whatever is on disk.
    # Last, so that the sidecar written just above is part of what gets indexed.
    zarr.consolidate_metadata(str(store_path))
    return array


def _bin_shrink_multiscales(image: Any, scale_factors: Sequence[int], out_chunks: Any) -> Any:
    """The BIN_SHRINK pyramid computed by ``dask.array.coarsen``, in ngff-zarr's own clothes.

    NOT ngff-zarr's ``ITKWASM_BIN_SHRINK``, for a reason found on real rounds: that method hands
    the 32-bit wasm sandbox one BLOCK at a time, the sandbox traps near 2.5 GiB -- a whole ExaSPIM
    volume as one block -- and it also traps on any multi-block layout whose extent does not divide
    the factor (514 rows @4: every chunking leaves an offending tail, so NO layout is safe). The
    engine's own streamed writes hit both.

    ``coarsen(np.mean, trim_excess)`` over factor-aligned blocks is the same statistic -- the mean
    of each aligned ``factor**rank`` window, the global remainder dropped -- computed lazily with a
    bounded peak, no sandbox, no layout constraint. The cast back to the payload dtype truncates,
    which is the ``static_cast`` ITK's own BinShrink performs.

    The METADATA stays ngff-zarr's: each level's dataset entry is taken from a single-level
    ``to_multiscales`` call on that level's image and repathed, so the axes, transform spelling and
    version handling remain theirs, not a private replica that drifts when they move.
    """
    dims = list(image.dims)
    spatial = [dim for dim in dims if dim in ("z", "y", "x")]
    images = [image]
    previous, previous_absolute = image, 1
    for absolute in scale_factors:
        factor = int(absolute) // previous_absolute
        if factor * previous_absolute != int(absolute) or factor < 1:
            raise DatasetManagerError(
                f"scale_factors {list(scale_factors)} (relative to level 0) do not form a ladder: each"
                " factor must be an integer multiple of the previous one.",
                "Each declared factor shrinks the level above it and must be 2 or more: [2, 2] or [4, 4].",
            )
        data = previous.data
        trim = tuple(
            slice(0, (int(data.shape[axis]) // factor) * factor) if dims[axis] in spatial else slice(None)
            for axis in range(len(dims))
        )
        for axis in range(len(dims)):
            if dims[axis] in spatial and int(data.shape[axis]) < factor:
                raise DatasetManagerError(
                    f"scale factor {int(absolute)} shrinks axis '{dims[axis]}' (extent"
                    f" {int(data.shape[axis])}) to nothing at this level.",
                    "Stop the ladder before the factor outgrows the smallest axis.",
                )
        # coarsen folds within blocks, so every spatial chunk must divide the factor; on the
        # trimmed extent a factor-multiple chunk size guarantees the tail does too.
        chunk = max(factor, (256 // factor) * factor)
        working = data[trim].rechunk({axis: chunk if dims[axis] in spatial else -1 for axis in range(len(dims))})
        coarsened = dask.array.coarsen(
            np.mean, working, {axis: factor for axis in range(len(dims)) if dims[axis] in spatial}
        )
        if not np.issubdtype(np.dtype(data.dtype), np.floating):
            # Round to nearest, half up, BEFORE the cast: astype truncates toward zero, and ITK's
            # BinShrink (the reference for these levels) rounds -- a 0.5 window mean writes 1.
            # Truncation would shift every integer level ~half an LSB down, silently and uniformly.
            coarsened = dask.array.floor(coarsened + 0.5)
        coarsened = coarsened.astype(data.dtype)
        # The centre-of-voxel convention ngff-zarr itself applies: the coarse voxel's centre sits
        # half the spacing delta past the fine one's. Getting this wrong composes every level a
        # fraction of a voxel apart and still looks like an image.
        scale = {dim: previous.scale[dim] * (factor if dim in spatial else 1) for dim in previous.scale}
        translation = {
            dim: previous.translation[dim] + (0.5 * (factor - 1) * previous.scale[dim] if dim in spatial else 0.0)
            for dim in previous.translation
        }
        previous = dataclasses.replace(previous, data=coarsened, scale=scale, translation=translation)
        previous_absolute = int(absolute)
        images.append(previous)

    if out_chunks is not None:
        images = [
            dataclasses.replace(level, data=level.data.rechunk(tuple(out_chunks)))
            if hasattr(level.data, "rechunk")
            else level
            for level in images
        ]
    assembled = ngff_zarr.to_multiscales(image, scale_factors=[], chunks=out_chunks, cache=False)
    datasets = []
    for index, level in enumerate(images):
        level_multiscales = ngff_zarr.to_multiscales(level, scale_factors=[], chunks=out_chunks, cache=False)
        dataset = level_multiscales.metadata.datasets[0]
        path = f"scale{index}/{image.name}"
        for transform_sequence in dataset.coordinateTransformations or []:
            if getattr(transform_sequence, "input", None) is not None and hasattr(transform_sequence.input, "path"):
                transform_sequence.input.path = path
        dataset.path = path
        datasets.append(dataset)
    assembled.images = images
    assembled.metadata = dataclasses.replace(assembled.metadata, datasets=datasets)
    # Both OFF, deliberately: to_ngff_zarr RE-DERIVES every level whose index it can, whenever
    # scale_factors, method and chunks are all set -- to_multiscales records its default method
    # (the gaussian) even when asked for no levels. The levels' data never go through it (see
    # append_ome_zarr_levels), but its metadata write must not try to derive them either.
    assembled.scale_factors = []
    assembled.method = None
    return assembled


def append_ome_zarr_levels(
    store_path: str | Path,
    scale_factors: Sequence[int],
    *,
    downsample_method: str | None = None,
) -> None:
    """Add coarser levels to a store that already holds its level 0.

    The companion of :func:`create_ome_zarr_store`: a store written region by region cannot be given
    ``scale_factors`` up front, because no level exists until the last region lands. This derives the
    pyramid afterwards, from what is on disk, and grafts it BESIDE level 0.

    Level 0 is not rewritten, not moved, not read back whole: each coarser level is computed lazily
    from it and stored straight into a new array of the same group, so the cost is one pass over
    level 0 into a level 16x smaller (measured 55 s -> ~10 s on a 4.9 GB store), with a chunk-sized
    peak. The multiscales metadata that names every level is still ngff-zarr's: it is described from
    one-voxel stand-ins (the metadata does not depend on the extent) into a scratch store and copied
    onto the group's attributes, so the axes, transforms and version spelling remain theirs. The
    KonfAI attribute sidecar is untouched, being a key beside theirs; a displacement field keeps its
    typed component axis through the same call that types it at creation.
    """
    _require_ngff_zarr()
    if not scale_factors:
        return
    store = Path(store_path)
    clear_ome_zarr_cache(store)
    field = is_displacement_field(store)
    base = ngff_zarr.from_ngff_zarr(str(store)).images[0]
    stored_chunks = tuple(int(size) for size in base.data.chunksize)
    if downsample_method in (None, "ITKWASM_BIN_SHRINK"):
        multiscales = _bin_shrink_multiscales(base, _level_zero_scale_factors(scale_factors), stored_chunks)
    else:
        multiscales = ngff_zarr.to_multiscales(
            base,
            scale_factors=_level_zero_scale_factors(scale_factors),
            method=_downsample_method(downsample_method),
            chunks=stored_chunks,
            cache=False,
        )
    version = _DEFAULT_VERSION
    if field:
        _require_zarr_v3_for_rfc5()
        _type_component_axis(multiscales, _DISPLACEMENT_AXIS_TYPE)
        version = _RFC5_VERSION

    group = zarr.open_group(str(store), mode="r+")
    create_array = getattr(group, "create_array", None) or group.create_dataset
    for level, dataset in zip(multiscales.images[1:], multiscales.metadata.datasets[1:], strict=True):
        data = level.data.rechunk(stored_chunks)
        array = create_array(
            dataset.path, shape=data.shape, chunks=stored_chunks, dtype=data.dtype, fill_value=0, overwrite=True
        )
        # Aligned chunks: each zarr chunk is written by exactly one task, so no lock is needed.
        dask.array.store(data, array, lock=False)

    # The metadata LAST, so an interrupted call leaves a store that still reads as its level 0
    # (unreferenced arrays beside it are overwritten by the next call).
    rank = base.data.ndim - 1
    described = dataclasses.replace(
        multiscales,
        images=[
            dataclasses.replace(
                level, data=dask.array.zeros((level.data.shape[0], *(1,) * rank), dtype=level.data.dtype)
            )
            for level in multiscales.images
        ],
        scale_factors=[],
        method=None,
    )
    scratch = store.with_name(f"{store.name}.describing")
    shutil.rmtree(scratch, ignore_errors=True)
    try:
        ngff_zarr.to_ngff_zarr(str(scratch), described, overwrite=True, version=version)
        group.attrs.update(dict(zarr.open_group(str(scratch), mode="r").attrs))
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    zarr.consolidate_metadata(str(store))
    clear_ome_zarr_cache(store)


def get_ome_zarr_info(store_path: str | Path, level: int = 0) -> dict[str, Any]:
    """OME-Zarr metadata, without reading pixel data.

    Three of these keys describe the same level in two different orders, and mixing them is the
    single most productive mistake this module invites. ``shape``, ``scale`` and ``translation``
    follow the STORE's own axes, listed in ``axes``: a scalar volume written without a channel
    axis has three of each. ``canonical_shape`` is the C[Z]YX form the reader indexes. So a caller
    that sizes its slices from ``canonical_shape`` and then reads ``scale[1:]`` for the spatial
    spacing is off by one axis, with plausible numbers and no error.

    ``geometry`` exists so that never has to be reasoned about: it maps each axis NAME to its
    ``(scale, translation)``. Prefer it.
    """
    image = _load_image(str(store_path), level)
    dims = [str(axis).lower() for axis in image.dims]
    try:
        n_levels = len(ngff_zarr.from_ngff_zarr(str(store_path)).images)
    except (OSError, TypeError, ValueError):
        n_levels = 1
    return {
        "axes": dims,
        "shape": list(image.data.shape),
        # The shape the slices of `read_ome_zarr_data_slice` are indexed against. Both are here
        # because they differ whenever the store's axes are not already C[Z]YX, and a caller sizing
        # its slices from "shape" then reads a transposed region, with the right rank, plausible
        # values, and nothing raised. "shape" stays the store's own order; this one is the reader's.
        "canonical_shape": _canonical_shape(dims, image.data.shape),
        "chunks": list(getattr(image.data, "chunks", []) or []),
        "dtype": str(image.data.dtype),
        "scale": _ordered(dict(image.scale), dims),
        "translation": _ordered(dict(image.translation), dims),
        # Keyed by axis name, so no caller has to know which of the two orders it is holding.
        "geometry": {
            axis: {"scale": float(scale), "translation": float(translation)}
            for axis, scale, translation in zip(
                dims,
                _ordered(dict(image.scale), dims),
                _ordered(dict(image.translation), dims),
                strict=True,
            )
        },
        "n_levels": n_levels,
        "attributes": _read_konfai_attributes(store_path),
    }
