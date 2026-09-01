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
   matrix, which OME-NGFF cannot express) through a single ``konfai`` root
   attribute, carried by ngff-zarr beside the OME metadata.

Reads are lazy: ``ngff-zarr`` exposes the array as a chunked store, so slicing
only materialises the requested patch.

Optional dependencies: ``zarr`` + ``ngff-zarr`` (``pip install konfai[omezarr]``).
"""

from __future__ import annotations

import contextlib
import dataclasses
import itertools
import operator
import tempfile
import threading
from collections import OrderedDict
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from konfai.utils import uri
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
# everywhere else, being the version external OME-Zarr readers most widely accept.
_DISPLACEMENT_AXIS_TYPE = "displacement"

# EVERY field store holds its components in the spec's order -- the OUTPUT axes' (dz, dy, dx for a
# zyx store) -- there is no second layout. KonfAI's own convention stays ITK's (dx, dy, dz)
# everywhere in memory; the two orders meet only at this backend's read/write boundary, where the
# components are flipped. An axis-aligned grid is additionally declared a ``displacements``
# transformation mapping the physical coordinate system onto itself through the level-0 array,
# which is what makes it APPLICABLE by a spec reader rather than merely labelled; a grid carrying
# a rotation cannot be declared (RFC-5 maps a field's array to space by scale and translation
# alone) and keeps only the typed axis, its Direction in the sidecar, the marker below saying how
# its components are ordered. A typed store with NEITHER the entry NOR the marker is a pre-1.9
# layout whose components are ITK-ordered: reading it under one convention or the other would be a
# guess with a plausible registration either way, so it is refused by name.
_PHYSICAL_CS = "physical"
_FIELD_COMPONENTS_KEY = "field_components"
_FIELD_COMPONENTS = "output-axes"

_RFC5_VERSION = "0.6"
_DEFAULT_VERSION = "0.4"

#: How a chunk is sized, whether the store is created from a shape alone or from the region shape a
#: streamed writer declares (:func:`konfai.utils.dataset._store_chunks`). One rule, two callers: a
#: chunk is the unit a reader decompresses to reach one voxel, so an oversized one is paid by every
#: partial read forever, and by any consumer that is not KonfAI.
CHUNK_SPATIAL_TILE = 128
CHUNK_TARGET_BYTES = 32 << 20

#: zarr v2 stores keep byte-shuffled blosc-lz4 (what every 1.8.2 store carries) rather than the
#: zarrista writer's zstd-0 default: measured on a CT-like uint16 volume, zstd-0 costs +19 % disk
#: and ~+11 % on the streamed read sweep, because without the shuffle a uint16's high bytes break
#: every run the compressor could fold.
_V2_COMPRESSOR = {"id": "blosc", "cname": "lz4", "clevel": 5, "shuffle": 1, "blocksize": 0}


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


def _read_konfai_attributes(store_path: str | Path) -> dict[str, Any]:
    """KonfAI's proprietary ``Attribute`` sidecar: the ``konfai`` key ngff-zarr carries back beside
    the OME metadata. A copy of a memoised parse: it is metadata, and a streamed run asks for it
    once per region.
    """
    try:
        root = _multiscales(str(store_path)).root_attributes or {}
    except Exception:
        return {}
    return dict(root.get(_KONFAI_ATTR_KEY, {}).get("attributes", {}))


def _from_ngff_zarr(store_path: str | Path) -> Any:
    """ngff-zarr's multiscales for ``store_path``. A remote root goes in as a key-to-bytes mapping
    over its own filesystem (ngff-zarr >= 0.44 reads a remote string as a local path) and as its URL
    when ngff-zarr refuses the mapping, which older releases resolve themselves."""
    if not uri.is_uri(store_path):
        return ngff_zarr.from_ngff_zarr(str(store_path))
    # Through uri.filesystem, so a missing fsspec backend or configuration is the structured
    # DatasetManagerError, never a raw dependency error.
    filesystem = uri.filesystem(store_path)
    _, target = uri.split_scheme(str(store_path))
    try:
        return ngff_zarr.from_ngff_zarr(filesystem.get_mapper(target))
    except ValueError:
        return ngff_zarr.from_ngff_zarr(str(store_path))


@lru_cache(maxsize=8)
def _multiscales(store_path: str) -> Any:
    """ngff-zarr's multiscales for a store, memoised per path: the images, their metadata, and the
    root attributes beside them, all from one parse.

    A streamed run reads one patch per call, and re-parsing the NGFF metadata and rebuilding the
    lazy array graph per patch is pure per-read overhead: the object is lazy (no voxel data), so a
    handful of them is cheap to keep. The key is the path alone, so anything that puts a different
    store at a path already read must call ``clear_ome_zarr_cache()``: see there.
    """
    _require_ngff_zarr()
    return _from_ngff_zarr(store_path)


def _load_image(store_path: str, level: int) -> Any:
    """Return the ``NgffImage`` for ``level`` of an OME-Zarr store, off the memoised parse.

    ``@N`` selects among the levels a store offers, so a single-level store has nothing to select: its
    one level is read whatever ``N`` says (as every other backend does: ``SitkFile`` ignores
    ``self.level`` too). Out of range on a store that IS a pyramid stays an error: asking level 3 of a
    three-level mask beside a four-level image is a real mismatch (it silently pairs 160 µm against
    320 µm), and quietly falling back to level 0 would hide it.
    """
    try:
        multiscales = _multiscales(store_path)
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
    _multiscales.cache_clear()
    _level_array.cache_clear()
    if _CHUNK_CACHE is not None:
        _CHUNK_CACHE.forget(None if store_path is None else store_identity(store_path))


def is_displacement_field(store_path: str | Path) -> bool:
    """Whether the store declares its component axis as an NGFF RFC-5 displacement field.

    This is what lets a DVF be read back as a transform rather than as a 3-channel image, and it is
    read from the store itself: the producer does not have to be trusted, and no sidecar convention
    (a filename, an attribute) has to be agreed on separately.

    A store that predates RFC-5 simply answers False: an unreadable or absent store is not a
    displacement field either, so this never raises.
    """
    if not _NGFF_ZARR_AVAILABLE:
        return False
    try:
        image = _multiscales(str(store_path)).images[0]
    except Exception:
        # "Not a displacement field" is the only answer this owes: it is asked purely to decide HOW to
        # read an entry, and an absent or unreadable store is not one either.
        return False
    return _has_displacement_axis(image)


def _has_displacement_axis(image: Any) -> bool:
    """Whether the image's component axis is typed as an RFC-5 displacement."""
    return (image.axes_types or {}).get("c") == _DISPLACEMENT_AXIS_TYPE


def _component_flip(store_path: str) -> bool:
    """Whether the store holds its components in the spec's order, to flip back to ITK's on read.

    Every field store this backend writes does -- marked by the ``displacements`` entry when the
    grid could be declared, by the sidecar's ``field_components`` when it could not. A store that
    types its axis and carries neither is a pre-1.9 layout whose components are ITK-ordered, and it
    is refused rather than read: under either convention the guess yields a plausible field with
    dx and dz possibly exchanged, which is the silent kind of wrong.
    """
    if not _NGFF_ZARR_AVAILABLE:
        return False
    try:
        multiscales = _multiscales(store_path)
    except Exception:
        return False
    if any(entry.type == "displacements" for entry in multiscales.metadata.coordinateTransformations or []):
        return True
    root = (multiscales.root_attributes or {}).get(_KONFAI_ATTR_KEY) or {}
    if root.get(_FIELD_COMPONENTS_KEY) == _FIELD_COMPONENTS:
        return True
    if _has_displacement_axis(multiscales.images[0]):
        raise DatasetManagerError(
            f"'{store_path}' types its component axis but declares no component order: a"
            " displacement field written by KonfAI < 1.9, whose components are ITK-ordered.",
            "Rewrite it from its source transform with KonfAI >= 1.9, or read it with the release that wrote it.",
        )
    return False


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

    @property
    def held_bytes(self) -> int:
        """What the cache holds right now: decoded chunks, in the bytes they take resident."""
        return self._bytes

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


def chunk_cache_held_bytes() -> int:
    """What the decoded-chunk cache holds resident right now, or 0 with no cache.

    For an instrument reading the process's resident memory over one scope of work: the cache
    outlives that scope by design (it is what a later region asks for again), so what it gained
    during the scope is not the scope's own cost. A fold's probe region read VmHWM over its ten
    members and charged the region 24.4 GiB, 13.2 of which was this cache filling from empty --
    and cut every region after it to 78 % of the height that would have fit.
    """
    return _CHUNK_CACHE.held_bytes if _CHUNK_CACHE is not None else 0


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
    from konfai.utils.budget import available_memory_bytes, budget_share

    share = budget_share("cache")
    if share is not None:
        return int(share)
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
    """``index`` as one unit-step slice per axis, and the axes an integer selection squeezes out.

    A stepped selection is refused rather than normalised: everything downstream counts the voxels
    between ``start`` and ``stop``, so a step would size the output wrongly and fill it from the
    wrong places. Both callers suppress this and fall back to the lazy array, which takes any step.
    """
    selections: list[slice] = []
    squeeze: list[int] = []
    for axis, item in enumerate(index):
        if isinstance(item, int):
            selections.append(slice(item, item + 1))
            squeeze.append(axis)
            continue
        start, stop, step = item.indices(shape[axis])
        if step != 1:
            raise DatasetManagerError(
                f"The chunked read takes unit-step selections; axis {axis} asks for step {step}.",
                "Read this level through the lazy array instead.",
            )
        selections.append(slice(start, stop))
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


def _level_path(store_path: str, level: int) -> str | None:
    """The zarr path of one level, from the store's memoised multiscales metadata."""
    try:
        datasets = _multiscales(store_path).metadata.datasets
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
    return _lazy_window(image.data, index)


def _lazy_window(data: Any, index: tuple) -> np.ndarray:
    """The plain lazy read. A lazy array that refuses a stepped slice (ngff-zarr >= 0.44 wraps the
    level in an adapter that takes unit steps only) is read over the unit-step span and stepped here."""
    try:
        return np.asarray(data[index])
    except NotImplementedError:
        # Each slice normalized against the shape, then its ascending unit-step span; a negative
        # step reads that span backwards from its own end, which lands on the indices the original
        # slice named.
        bounds = tuple(
            slice(*item.indices(size)) if isinstance(item, slice) else item
            for item, size in zip(index, data.shape, strict=True)
        )
        span = tuple(
            slice(item.stop + 1, item.start + 1)
            if isinstance(item, slice) and item.step < 0
            else slice(item.start, item.stop)
            if isinstance(item, slice)
            else item
            for item in bounds
        )
        steps = tuple(slice(None, None, item.step) for item in bounds if isinstance(item, slice))
        return np.asarray(data[span])[steps]


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


def ome_zarr_read_granularity(store_path: str | Path, *, level: int = 0) -> tuple[int, ...] | None:
    """The stored block a read of this level is served in, as a KonfAI ``C[Z]YX`` shape.

    A chunked store decodes whole chunks, so a window costs the chunk-aligned hull that covers it
    (:func:`_assemble_window` issues one read over that hull): a decomposition whose blocks straddle
    the grid materialises every plane it touches, twice over where it lands between two. Read from
    the level's metadata, so it costs nothing and is answerable at plan time. ``None`` when the
    store cannot be opened, where the caller prices a read at what it asks for.
    """
    if not _NGFF_ZARR_AVAILABLE:
        return None
    with contextlib.suppress(Exception):  # a store this cannot read simply has no granularity to state
        image = _load_image(str(store_path), level)
        level_path = _level_path(str(store_path), level)
        if level_path is None:
            return None
        array = _level_array(str(store_path), level_path)
        dims = [str(axis).lower() for axis in image.dims]
        return tuple(_canonical_shape(dims, array.chunks))
    return None


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
    """Read a KonfAI channel-first ``C[Z]YX`` patch from an OME-Zarr store (lazy).

    A conformant displacement store holds its components in RFC-5 order; the channel selection is
    remapped and the patch flipped back, so every caller keeps receiving ITK's (dx, dy, dz)
    whatever layout the store holds.
    """
    image = _load_image(str(store_path), level)
    dims = [str(axis).lower() for axis in image.dims]
    canonical_shape = _canonical_shape(dims, image.data.shape)
    flipped = _component_flip(str(store_path))
    if flipped:
        start, stop, step = slices[0].indices(canonical_shape[0])
        if step != 1:
            raise DatasetManagerError("A displacement store's component axis takes unit-step selections.")
        slices = (slice(canonical_shape[0] - stop, canonical_shape[0] - start), *slices[1:])
    index = _store_index(dims, canonical_shape, slices, timepoint)
    patch = _read_level_window(str(store_path), level, image, index)
    remaining = [axis for axis, selection in zip(dims, index, strict=True) if not isinstance(selection, int)]
    wanted = [axis for axis in ("c", *_SPATIAL) if axis in remaining]
    patch = np.transpose(patch, [remaining.index(axis) for axis in wanted])
    if "c" not in remaining:
        patch = patch[np.newaxis]
    elif flipped:
        # Contiguous, not a reversed view: a negative stride is refused by torch.from_numpy, and
        # every patch this returns is about to become a tensor.
        patch = np.ascontiguousarray(patch[::-1])

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
    """Resolve a downsampling method name to ngff-zarr's enum, defaulting to DASK_BIN_SHRINK.

    NOT ngff-zarr's own default, which is ``ITKWASM_GAUSSIAN``: a pyramid is indexed by position and
    read as "the same image, coarser", so a level that has been smoothed is a change of pixels that
    no reader can see. Measured on a real volume, the gaussian keeps a 0.9998 correlation while
    crushing the peak intensity by 20 %: the shape of difference that passes a sanity check and
    resurfaces months later. ``DASK_BIN_SHRINK`` is a plain block mean with ITK's own BinShrink
    semantics (aligned windows, remainder trimmed, integers rounded half up), computed lazily with
    a bounded peak: it takes any extent and chunk layout the streamed writer leaves, where the wasm
    variant traps on blocks past 2.5 GiB and on tails no chunking can avoid.
    """
    _require_ngff_zarr()
    if downsample_method is None:
        return ngff_zarr.Methods.DASK_BIN_SHRINK
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
    if scale_factors and uri.is_uri(store_path):
        raise DatasetManagerError(
            f"Cannot append pyramid levels to the remote store '{store_path}'.",
            "Levels are derived in place through local paths; write the store locally and upload it.",
        )
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


def _write_skeleton(store_path: str | Path, multiscales: Any, version: str, **kwargs: Any) -> None:
    """The store's metadata and empty arrays, written in place (``to_ngff_zarr(metadata_only=True)``
    describes every level and creates its array without computing a voxel). ngff-zarr writes local
    directories only, so a remote root gets the skeleton written locally and uploaded through the
    root's own filesystem: metadata documents only, before a chunk lands."""
    if not uri.is_uri(store_path):
        ngff_zarr.to_ngff_zarr(
            str(store_path), multiscales, overwrite=True, version=version, metadata_only=True, **kwargs
        )
        return
    filesystem = uri.filesystem(store_path)
    _, target = uri.split_scheme(str(store_path))
    if "/" not in target.strip("/"):
        raise DatasetManagerError(
            f"Refusing to create the store at the filesystem root '{store_path}'.",
            "Name a key under the root (e.g. '.../dataset.ome.zarr'): creating a store replaces what its path holds.",
        )
    if filesystem.exists(target):
        filesystem.rm(target, recursive=True)
    filesystem.makedirs(target, exist_ok=True)
    with tempfile.TemporaryDirectory() as scratch:
        local = Path(scratch) / "skeleton"
        ngff_zarr.to_ngff_zarr(str(local), multiscales, overwrite=True, version=version, metadata_only=True, **kwargs)
        for file in sorted(path for path in local.rglob("*") if path.is_file()):
            filesystem.put_file(str(file), uri.join(target, file.relative_to(local).as_posix()))


def _grid_is_axis_aligned(attributes: dict[str, Any] | None) -> bool:
    """Whether the field's grid carries no rotation.

    RFC-5 maps a field's array to space by scale and translation alone, so only an axis-aligned
    grid can be declared a ``displacements`` transformation; an oriented one keeps the label-only
    layout, its Direction in the sidecar.

    ``Attribute`` versions its keys (``Direction_0``, ``Direction_1``, ...), and the sidecar dict
    arrives here verbatim: the LATEST version is the grid the store describes.
    """
    versions = {
        key: value
        for key, value in (attributes or {}).items()
        if key == "Direction" or (key.startswith("Direction_") and key.removeprefix("Direction_").isdigit())
    }
    if not versions:
        return True
    value = versions[max(versions, key=lambda key: int(key.rsplit("_", 1)[-1]) if "_" in key else -1)]
    flat = np.asarray(
        str(value).replace("[", " ").replace("]", " ").split() if isinstance(value, str) else value, dtype=np.float64
    ).ravel()
    side = round(len(flat) ** 0.5)
    return side * side == len(flat) and bool(np.allclose(flat.reshape(side, side), np.eye(side)))


def _declare_displacements_transform(multiscales: Any) -> None:
    """Mark the store as an RFC-5 ``displacements`` transformation, in place.

    What makes the field APPLICABLE by a spec reader rather than merely labelled: a spatial
    ``physical`` coordinate system, and a ``displacements`` entry mapping it onto itself through
    the level-0 array. The components must then follow the output axes' order, which is
    ``_ComponentFlippedWriter``'s half of the contract.
    """
    from ngff_zarr.v06.zarr_metadata import Axis, CoordinateSystem, CoordinateSystemIdentifier, Displacements

    spatial = [str(dim) for dim in multiscales.images[0].dims if dim in _SPATIAL]
    physical = CoordinateSystem(name=_PHYSICAL_CS, axes=[Axis(name=name, type="space", unit=None) for name in spatial])
    reference = CoordinateSystemIdentifier(name=_PHYSICAL_CS)
    entry = Displacements(
        input=reference, output=reference, path=multiscales.metadata.datasets[0].path, interpolation="linear"
    )
    multiscales.metadata = dataclasses.replace(
        multiscales.metadata,
        coordinateSystems=[*multiscales.metadata.coordinateSystems, physical],
        coordinateTransformations=[entry],
    )


class _ComponentFlippedWriter:
    """The level-0 array of a conformant displacement store, taking ITK-ordered components.

    Every producer in KonfAI hands fields in ITK's component order (dx, dy, dz); the store holds
    the spec's (dz, dy, dx). Flipping at this boundary keeps the two conventions from ever
    meeting: no producer knows about the spec, no store holds a private order. A value of lower
    rank than the array (a scalar fill) has no component identity and broadcasts as it stands.
    """

    def __init__(self, array: Any) -> None:
        self._array = array
        self._channels = int(array.shape[0])

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self._array.shape)

    @property
    def chunks(self) -> tuple[int, ...]:
        return tuple(self._array.chunks)

    @property
    def dtype(self) -> Any:
        return self._array.dtype

    def _remap(self, key: Any) -> tuple[Any, bool]:
        """The store-side selection for a caller's ITK-side one, and whether values need flipping."""
        if key is Ellipsis:
            return key, True
        first, *rest = key if isinstance(key, tuple) else (key,)
        if isinstance(first, int):
            return (self._channels - 1 - first, *rest), False
        start, stop, step = first.indices(self._channels)
        if step != 1:
            raise DatasetManagerError("A displacement store's component axis takes unit-step selections.")
        return (slice(self._channels - stop, self._channels - start), *rest), True

    def __setitem__(self, key: Any, value: Any) -> None:
        target, flip = self._remap(key)
        data = np.asarray(value)
        self._array[target] = np.flip(data, axis=0) if flip and data.ndim == self._array.ndim else data

    def __getitem__(self, key: Any) -> np.ndarray:
        source, flip = self._remap(key)
        data = np.asarray(self._array[source])
        return np.ascontiguousarray(np.flip(data, axis=0)) if flip else data


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
    the ONLY path a real one takes. The KonfAI sidecar rides along as a root attribute beside the OME
    keys, and the array is described from a LAZY zeros of the real shape: ``metadata_only`` creates it
    without computing a voxel, chunked exactly as the caller says: the region grid is the one thing
    ngff-zarr cannot infer, and a store whose chunks straddle it turns every region write into a
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

    data = dask.array.zeros(tuple(shape), dtype=np.dtype(dtype), chunks=chunks)
    image = ngff_zarr.to_ngff_image(data, dims=dims, scale=scale, translation=translation)
    version = _DEFAULT_VERSION
    if displacement_field:
        # Typing the component axis (NGFF RFC-5, so version 0.6) is what lets the store say on disk
        # that its channels are a displacement rather than an ordinary 3-channel image.
        image.axes_types = {"c": _DISPLACEMENT_AXIS_TYPE}
        version = _RFC5_VERSION
    multiscales = ngff_zarr.to_multiscales(image, scale_factors=[], chunks=chunks, cache=False)
    if displacement_field:
        # One layout for every field: components in the spec's order. The entry when the grid can
        # be declared; the sidecar marker either way, so the reader never has to guess.
        if _grid_is_axis_aligned(attributes):
            _declare_displacements_transform(multiscales)
        multiscales.root_attributes = {
            _KONFAI_ATTR_KEY: {"attributes": dict(attributes or {}), _FIELD_COMPONENTS_KEY: _FIELD_COMPONENTS}
        }
    elif attributes:
        multiscales.root_attributes = {_KONFAI_ATTR_KEY: {"attributes": dict(attributes)}}
    # version is explicit because to_ngff_zarr defaults to 0.5; 0.4 stays the portable default. A
    # v3 (RFC-5) store takes the writer's own codec chain, which is what 1.8.2 wrote there too.
    compression = {} if displacement_field else {"compressor": _V2_COMPRESSOR}
    _write_skeleton(store_path, multiscales, version, **compression)

    # The level-0 key comes from the metadata rather than a literal: ngff-zarr builds it from the
    # image name, so "scale0/image" is its convention to change, not ours to hardcode.
    array = zarr.open_group(str(store_path), mode="r+")[multiscales.metadata.datasets[0].path]
    return _ComponentFlippedWriter(array) if displacement_field else array


def append_ome_zarr_levels(
    store_path: str | Path,
    scale_factors: Sequence[int],
    *,
    downsample_method: str | None = None,
) -> None:
    """Add coarser levels to a store that already holds its level 0.

    The companion of :func:`create_ome_zarr_store`: a store written region by region cannot be given
    ``scale_factors`` up front, because no level exists until the last region lands. This derives the
    pyramid afterwards, from what is on disk, and grafts it BESIDE level 0
    (``to_ngff_zarr(start_level=1)``): level 0 is not rewritten, not moved, not read back whole; each
    coarser level is computed lazily from the one before it with a chunk-sized peak, and the
    multiscales metadata that names every level lands last, so an interrupted call leaves a store
    that still reads exactly as its level 0. The KonfAI attribute sidecar rides along as the root
    attributes ngff-zarr read back beside the OME keys; a displacement field keeps its typed
    component axis the same way.
    """
    if uri.is_uri(store_path):
        raise DatasetManagerError(
            f"Cannot append pyramid levels to the remote store '{store_path}'.",
            "Levels are derived in place through local paths; write the store locally and upload it.",
        )
    _require_ngff_zarr()
    if not scale_factors:
        return
    store = Path(store_path)
    clear_ome_zarr_cache(store)
    multiscales = _from_ngff_zarr(store)
    base = multiscales.images[0]
    factors = _level_zero_scale_factors(scale_factors)
    _refuse_factors_outgrowing_an_axis(base, factors)
    derived = ngff_zarr.to_multiscales(
        base,
        scale_factors=factors,
        method=_downsample_method(downsample_method),
        chunks=tuple(int(size) for size in base.data.chunksize),
        cache=False,
    )
    derived.root_attributes = multiscales.root_attributes
    if multiscales.metadata.coordinateTransformations:
        # A conformant field keeps its ``displacements`` entry (and the coordinate system it names)
        # through the append: the entry references level 0, which this never rewrites.
        declared = {system.name for system in derived.metadata.coordinateSystems}
        derived.metadata = dataclasses.replace(
            derived.metadata,
            coordinateSystems=[
                *derived.metadata.coordinateSystems,
                *(s for s in multiscales.metadata.coordinateSystems if s.name not in declared),
            ],
            coordinateTransformations=multiscales.metadata.coordinateTransformations,
        )
    field = _has_displacement_axis(base)
    # The coarse levels take level 0's own compressor, so the store stays uniform whatever wrote
    # it; a v3 store carries a codec chain instead and keeps the writer's default.
    level_zero = zarr.open_group(str(store), mode="r")[multiscales.metadata.datasets[0].path]
    compressor = level_zero.metadata.to_dict().get("compressor")
    ngff_zarr.to_ngff_zarr(
        str(store),
        derived,
        overwrite=False,
        version=_RFC5_VERSION if field else _DEFAULT_VERSION,
        start_level=1,
        **({"compressor": compressor} if compressor else {}),
    )
    clear_ome_zarr_cache(store)


def _refuse_factors_outgrowing_an_axis(base: Any, factors: Sequence[int]) -> None:
    """A factor that shrinks an axis to nothing is refused by name: ngff-zarr would fall back to
    deriving that level from level 0 and write an empty array where a consumer's ``@N`` resolves."""
    for absolute in factors:
        for axis, dim in enumerate(base.dims):
            if dim in _SPATIAL and int(base.data.shape[axis]) // int(absolute) == 0:
                raise DatasetManagerError(
                    f"scale factor {int(absolute)} shrinks axis '{dim}' (extent"
                    f" {int(base.data.shape[axis])}) to nothing at this level.",
                    "Stop the ladder before the factor outgrows the smallest axis.",
                )


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
        n_levels = len(_multiscales(str(store_path)).images)
    except (OSError, TypeError, ValueError):
        n_levels = 1
    scale = _ordered(dict(image.scale), dims)
    translation = _ordered(dict(image.translation), dims)
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
        "scale": scale,
        "translation": translation,
        # Keyed by axis name, so no caller has to know which of the two orders it is holding.
        "geometry": {
            axis: {"scale": float(value), "translation": float(offset)}
            for axis, value, offset in zip(dims, scale, translation, strict=True)
        },
        "n_levels": n_levels,
        "attributes": _read_konfai_attributes(store_path),
    }
