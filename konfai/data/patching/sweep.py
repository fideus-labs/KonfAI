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


"""The streamed sweep of a case: regions cut on the store's blocks, read ahead, landed, written behind."""

import contextlib
import itertools
import queue
import threading
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import numpy as np
import torch

from konfai.data.patching.budget import _PLATEAU_READ_MARGIN
from konfai.data.patching.stage import Stage, _ReadStagePlan
from konfai.data.transform import (
    Save,
)
from konfai.utils.clock import SweepClock
from konfai.utils.dataset import Attribute, Dataset, DataStream, as_channel_first
from konfai.utils.errors import ConfigError, PatchError, TransformError
from konfai.utils.runtime import rank_cpu_share


def save_destination(save: Save, default_dataset: Dataset, default_group: str) -> tuple[Dataset, str]:
    """The dataset and group a :class:`Save` caches into, the manager's own when it names none.

    Public because a planner has to resolve a destination exactly as the engine will: one that probes
    a store the run does not open has verified nothing.
    """
    destination = save.destination
    # No destination of its own: the Save caches into the manager's dataset, whose write format is
    # not this stage's to redecorate: a pyramid asked here would silently not happen.
    if destination is None and save.scale_factors:
        raise ConfigError(
            f"A '{type(save).__name__}' asks for a pyramid but names no dataset of its own.",
            "scale_factors describes a store this stage writes, so give it one:"
            " Write: {dataset: ./Out:omezarr, scale_factors: [4]}.",
        )
    return destination or default_dataset, save.group or default_group


class RegionWriter:
    """Region writes into streams opened at their first block: the one sweep loop of the three
    engines (a Save's sweep, an Expand's shared pass, a Reduce).

    ``write`` opens the stream for a key on its first block (through ``open_stream``, so a refusal
    surfaces where the caller can say why) and writes the block; ``close`` publishes every stream,
    ``abort`` drops every partial entry and is safe after a failure or an interrupt.

    Writes here are synchronous. Whether a caller should overlap them with its compute depends on
    where the compute runs, since the store's encoder competes for the host's cores: measured on a
    300^3 float32 case to OME-Zarr, a host chain writing one slab behind was 50 % slower than
    writing in place, and a host resample of a 513x1331x1776 case stayed at 13.0 s either way,
    where the same resample on a GPU went from 8.2 s to 6.2 s with the write one block behind.
    :class:`_WriteBehind` is what a sweep wraps this in; nothing here assumes it.
    """

    def __init__(self, open_stream: Callable[[Any, np.ndarray, Attribute], DataStream]) -> None:
        self._open_stream = open_stream
        self._streams: dict[Any, DataStream] = {}

    @property
    def opened(self) -> set[Any]:
        return set(self._streams)

    def write(self, key: Any, region: tuple[slice, ...], block: np.ndarray, header: Attribute) -> None:
        """Write ``block`` at ``region`` (channel slice included) into ``key``'s stream, opened on
        its first block with ``header``; a destination that refuses the stream raises there."""
        if key not in self._streams:
            stream = self._open_stream(key, block, header)
            stream.__enter__()
            self._streams[key] = stream
        self._streams[key].write_slice(region, block)

    def close(self) -> None:
        """Publish every stream."""
        for stream in self._streams.values():
            stream.close()

    def abort(self, error: BaseException | None = None) -> None:
        """Drop every partial entry."""
        for stream in self._streams.values():
            stream.abort(error)
        self._streams.clear()


@dataclass(frozen=True)
class _PatchStreamSource:
    """What a copy's patches are read from, and what runs on them once read. ``stage_plans`` mirrors
    ``stages`` one to one (the region plans compose, each pulling through the one before it);
    ``pending_sweeps`` are the unsatisfied :class:`Save` caches to materialize before any patch
    flows; ``entry`` is the case's own name, or the copy's behind an :class:`Expand`.
    """

    dataset: Dataset
    group: str
    entry: str
    shape: list[int]
    stages: list[Stage]
    stage_plans: tuple[_ReadStagePlan, ...]
    pending_sweeps: tuple["_PendingSweep", ...] = ()

    @property
    def region_index(self) -> int | None:
        """The first region stage, or ``None`` for an exact-patch chain."""
        for index, plan in enumerate(self.stage_plans):
            if plan.kind.is_region:
                return index
        return None


@dataclass(frozen=True)
class _PendingSweep:
    """One unsatisfied :class:`Save` and the segment that feeds it: the sweep reads each block of the
    Save's space through the segment (re-planned at sweep time) and region-writes it to
    ``destination``. ``entry``/``source_entry`` are the destination and source entry names (the
    copy's own behind an :class:`Expand`); ``copy_stage_start`` is where the per-copy part of the
    segment begins (``len(stages)`` when it is entirely shared); ``stage_plans`` are the probe-time
    plans, kept for the shared-pass classification only."""

    destination: Dataset
    group: str
    entry: str
    stages: list[Stage]
    source_dataset: Dataset
    source_group: str
    source_entry: str
    source_shape: list[int]
    out_spatial: tuple[int, ...]
    # The case state the segment replays from (copied per block).
    base_attributes: Attribute = field(repr=False)
    copy_stage_start: int = 0
    stage_plans: tuple[_ReadStagePlan, ...] = ()

    @property
    def tail_stages(self) -> tuple[Stage, ...]:
        """The per-copy part of the segment: what a copy applies to a block the copies share."""
        return tuple(self.stages[self.copy_stage_start :])

    @property
    def tail_plans(self) -> tuple[_ReadStagePlan, ...]:
        """The probe-time plans of :attr:`tail_stages`."""
        return self.stage_plans[self.copy_stage_start :]


@dataclass(frozen=True)
class _SweepMember:
    """One stream a sweep writes: the key it is reported under, the :class:`Save` it lands in, the
    case state its plan leaves (its header), and the per-copy tail it applies to each block.

    ``stages``/``stage_plans`` name the tail in the same shape as a :class:`_PatchStreamSource`'s
    chain, so the block dispatch is the one the stages before the marker take. Every tail plan is
    POINTWISE (``CaseMaterializer._pointwise_tail``): the block a member is handed IS its region,
    which is what lets the copies share one read pass."""

    key: Any
    sweep: _PendingSweep
    evolved: Attribute = field(repr=False)
    stages: tuple[Stage, ...] = ()
    stage_plans: tuple[_ReadStagePlan, ...] = ()

    def region_spans(self, target: tuple[slice, ...]) -> list[list[slice]]:
        """The region each tail stage reads for a block of the landing: the block itself, for all of
        them, a pointwise stage pulling exactly what it is handed."""
        return [list(target) for _ in range(len(self.stages) + 1)]


def _sweep_targets(spatial: list[int], tile: Sequence[int]) -> Iterator[tuple[slice, ...]]:
    """The sweep's regions: ``tile``-sized blocks of the landing, innermost axis fastest.

    The order is the source's: consecutive blocks differ on the axis stored contiguously, so the
    chunks one block decodes are the chunks the next one reads.
    """
    steps = [max(1, int(step)) for step in tile]
    for corner in itertools.product(*(range(0, extent, step) for extent, step in zip(spatial, steps, strict=True))):
        yield tuple(
            slice(start, min(start + step, extent)) for start, step, extent in zip(corner, steps, spatial, strict=True)
        )


def _cubic_tile(spatial: list[int], voxels: int, align: int) -> list[int]:
    """The block of at most ``voxels`` closest to a cube inside ``spatial``, aligned to ``align``.

    ``align`` keeps a block a whole number of store chunks wide, so a region write never becomes a
    read-modify-write (:func:`konfai.utils.dataset._store_chunks`); an axis shorter than one step is
    taken whole. Why a cube: :meth:`DatasetManager._sweep_tile`.
    """
    tile = [max(1, int(extent)) for extent in spatial]
    budget = float(max(1, voxels))
    # Shortest axis first: an axis under the ideal side takes its whole extent and hands its slack
    # to the others, which only raises the side, so once one axis is over it every later one is too.
    free = sorted(range(len(spatial)), key=lambda axis: tile[axis])
    for taken, axis in enumerate(free):
        side = budget ** (1.0 / (len(free) - taken))
        if tile[axis] > side:
            for remaining in free[taken:]:
                tile[remaining] = max(1, min(tile[remaining], int(side)))
            break
        budget /= tile[axis]
    return [
        extent if extent >= spatial[axis] or extent < align else extent - extent % align
        for axis, extent in enumerate(tile)
    ]


def _pull_block_spans(
    spatial: list[int], tile: Sequence[int], plans: Sequence["_ReadStagePlan"]
) -> Iterator[list[slice]]:
    """The source window each block of a decomposition into ``tile`` reads, from the plans' own pull
    maps: closed form, no voxel read."""
    for target in _sweep_targets(spatial, tile):
        span = list(target)
        for plan in reversed(plans):
            span = list(plan.pull(tuple(span))) if plan.pull is not None else span
        yield span


def _span_voxels(span: Sequence[slice]) -> int:
    return int(np.prod([max(0, part.stop - part.start) for part in span], dtype=np.int64))


class BlockReads(NamedTuple):
    """What one decomposition's blocks read, in the currencies the sizing spends.

    Every consumer of the enumeration wants an aggregate of it and none wants the blocks, so it is
    walked once and the aggregates are kept: the widest window one block materialises, the widest
    hull the store decodes to serve one, and what the whole decomposition reads. Walking it per
    consumer made the plan spend 96% of its time in ``Resample.stream_region_source``.
    """

    widest_pull: int
    widest_hull: int
    total: int

    @property
    def widest_excess(self) -> int:
        """What the widest read materialises ABOVE the window it asked for: zero on a store that
        serves exactly what it is asked for."""
        return max(0, self.widest_hull - self.widest_pull)


def _pull_block_voxels(spatial: list[int], tile: Sequence[int], plans: Sequence["_ReadStagePlan"]) -> Iterator[int]:
    """The source voxels each block of a decomposition into ``tile`` materialises. Their sum is what
    the decomposition reads, their largest what one block of it holds."""
    return (_span_voxels(span) for span in _pull_block_spans(spatial, tile, plans))


def _plateau_rows(
    spatial: list[int], plans: Sequence["_ReadStagePlan"], tolerance: float = _PLATEAU_READ_MARGIN
) -> int | None:
    """The shortest region height whose decomposition already reads what the tallest one reads,
    within ``tolerance``: the height past which taller regions pull no fewer source voxels.

    Closed form, from the chain's own pull maps (:func:`_pull_block_voxels`): no voxel is read, so
    it is answerable at plan time for any chain, whatever the YAML declares. ``None`` when nothing
    pulls (no plans), where every height reads the same and the question does not arise.

    This is a CAP, never a target: below it a region re-reads what its neighbour already pulled,
    above it the reads are the same and only the working set grows. What a budget then affords is
    the caller's to search for, downward.
    """
    if not plans:
        return None
    total = max(1, int(spatial[0]))
    floor = sum(_pull_block_voxels(spatial, [total, *spatial[1:]], plans))
    if floor <= 0:
        return None
    allowed = floor * (1.0 + max(0.0, float(tolerance)))
    # Geometric, so the scan costs O(log) evaluations however tall the volume is.
    height, ladder = max(1, total // 256), []
    while height < total:
        ladder.append(height)
        height = max(height + 1, int(height * 1.5))
    ladder.append(total)
    for candidate in ladder:
        if sum(_pull_block_voxels(spatial, [candidate, *spatial[1:]], plans)) <= allowed:
            return candidate
    return total


def _sweep_pipeline_depth() -> int:
    """How many blocks a sweep keeps in flight beside the one it is transforming: one ahead and one
    behind, or none when the rank owns a single core (``OMP_NUM_THREADS=1`` means a serial run)."""
    return 1 if rank_cpu_share() > 1 else 0


def _sweep_resident_regions(depth: int) -> tuple[int, int]:
    """How many pulled source regions and how many landed blocks a sweep of pipeline ``depth`` holds
    at once: the region the chain is running on, and, pipelined, the ``depth`` queued ahead of it
    plus the one the reader holds while the queue is full; against the block being landed and,
    pipelined, the one being written behind it."""
    return (1, 1) if depth < 1 else (depth + 2, 2)


class SweepSegment(NamedTuple):
    """One segment the streamed route sweeps: where it reads from, what it lands, how it pulls."""

    dataset: Dataset
    group: str
    entry: str
    source_shape: list[int]
    landing: list[int]
    plans: tuple["_ReadStagePlan", ...]

    @property
    def channels(self) -> int:
        return int(self.source_shape[0])


#: This rank's sweeps, summed over the cases it ran: the unit the report is about.
SWEEP_CLOCK = SweepClock()


class _ReadAhead:
    """One block's read running while the previous one is transformed and written.

    The consumer drains IN ORDER, so a stage object is touched by one thread at a time: the read
    stages by the producer, the tail by the consumer, disjoint by construction. At most ``depth``
    blocks wait, and a consumer that leaves early stops the producer.
    """

    _DONE = object()

    def __init__(self, blocks: Iterator[Any], depth: int) -> None:
        self._blocks = blocks
        self._depth = depth
        self._queue: queue.Queue = queue.Queue(maxsize=max(1, depth))
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._produce, name="konfai-sweep-read", daemon=True)

    def __enter__(self) -> Iterator[Any]:
        if self._depth < 1:
            return self._blocks
        self._thread.start()
        return self._consume()

    def __exit__(self, exc_type, value, traceback) -> None:
        if self._depth < 1:
            return
        # To the end marker, whether the consumer stopped early or an error cut it short: a producer
        # blocked on a full queue cannot reach its own end, and cannot then be joined.
        self._stop.set()
        while self._queue.get()[0] is not _ReadAhead._DONE:
            pass
        self._thread.join()

    def _produce(self) -> None:
        try:
            for block in self._blocks:
                self._queue.put((block, None))
                if self._stop.is_set():
                    break
        except BaseException as error:  # re-raised in the consumer's thread, where the sweep is
            self._queue.put((None, error))
        self._queue.put((_ReadAhead._DONE, None))

    def _consume(self) -> Iterator[Any]:
        while True:
            block, error = self._queue.get()
            if block is _ReadAhead._DONE:
                self._queue.put((block, None))  # left for __exit__, which drains to it
                return
            if error is not None:
                raise error
            yield block


class _HostLanding:
    """The host buffers a device chain's blocks come home into, reused rather than reallocated.

    A fresh allocation per block pays first-touch on pages the transfer overwrites whole, so the
    pageable copy faults page by page: measured on a 216 MiB block, 2.4 GiB/s into a new buffer
    against 10.1 GiB/s into one already resident. Two slots, because the writer holds exactly one
    block while the sweep fills the next, and that is what the sweep already keeps live.

    A block already on the host is handed over as it is: copying one would cost the host chain
    exactly what this saves on the device one.
    """

    _SLOTS = 2

    def __init__(self) -> None:
        self._slots: list[torch.Tensor] = [torch.empty(0, dtype=torch.uint8) for _ in range(_HostLanding._SLOTS)]
        self._next = 0

    def take(self, tensor: torch.Tensor) -> np.ndarray:
        """``tensor``'s values on the host, in a buffer this owns (or its own, if it is host-side)."""
        if tensor.device.type == "cpu":
            return tensor.numpy()
        slot, self._next = self._next, (self._next + 1) % _HostLanding._SLOTS
        needed = tensor.nelement() * tensor.element_size()
        if self._slots[slot].nelement() < needed:
            self._slots[slot] = torch.empty(needed, dtype=torch.uint8)
        landing = self._slots[slot][:needed].view(tensor.dtype).view(tensor.shape)
        landing.copy_(tensor)
        return landing.numpy()


class _WriteBehind:
    """A sweep's :class:`RegionWriter`, driven from one thread that is not the sweep's.

    One worker and one outstanding write, so the order stays the sweep's and the store's own
    encoder keeps its concurrency. EVERY call goes to that thread, not only the writes: the h5
    backend holds a per-file ``RLock`` across a stream's whole life, and a release from a thread
    that did not take it raises.
    """

    def __init__(self, writer: RegionWriter, depth: int) -> None:
        self._writer = writer
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="konfai-sweep-write") if depth else None
        self._pending: Future | None = None

    def write(self, key: Any, region: tuple[slice, ...], block: np.ndarray, header: Attribute) -> None:
        if self._pool is None:
            self._timed_write(key, region, block, header)
            return
        self.flush()  # one outstanding write, so the order is the sweep's and the memory is bounded
        self._pending = self._pool.submit(self._timed_write, key, region, block, header)

    def _timed_write(self, key: Any, region: tuple[slice, ...], block: np.ndarray, header: Attribute) -> None:
        with SWEEP_CLOCK.phase("write"):
            self._writer.write(key, region, block, header)

    def flush(self) -> None:
        """Wait for the outstanding write, raising what it raised."""
        pending, self._pending = self._pending, None
        if pending is not None:
            pending.result()

    def close(self) -> set[Any]:
        """Publish every stream and answer which keys were written."""
        self.flush()
        return self._on_writer(self._publish)

    def abort(self, error: BaseException) -> None:
        """Drop every partial entry, whatever the outstanding write did."""
        with contextlib.suppress(Exception):
            self.flush()
        self._on_writer(self._writer.abort, error)

    def shutdown(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)

    def _publish(self) -> set[Any]:
        written = self._writer.opened
        with SWEEP_CLOCK.phase("write"):
            self._writer.close()
        return written

    def _on_writer(self, work: Callable[..., Any], *args: Any) -> Any:
        return work(*args) if self._pool is None else self._pool.submit(work, *args).result()


def _torch_dtype_hint(error: BaseException) -> str | None:
    """The config change that answers ``error``, when it is a dtype torch has no kernel for.

    torch ships none for several dtypes a store legitimately holds: ``uint16`` is what microscopy
    writes, and torch implements for it neither comparison, nor flip, nor arithmetic, nor scalar
    fill. A chain that touches such a payload raises deep inside a stage with nothing but the
    missing operator's name, where what the reader needs is the dtype and the line that fixes it.
    """
    text = str(error)
    if not isinstance(error, NotImplementedError) or "not implemented for" not in text or text.count("'") < 2:
        return None
    return (
        f"torch has no kernel for a '{text.rsplit(chr(39), 2)[-2]}' payload: put a cast at the head"
        " of the chain (TensorCast, dtype float32, or int32 to keep every value exact) so the stages"
        " work in a dtype torch implements. The store keeps its own."
    )


@contextlib.contextmanager
def _stage_failures_explained() -> Iterator[None]:
    """Re-raise what a chain raised with the config change that answers it, where there is one."""
    try:
        yield
    except NotImplementedError as error:
        hint = _torch_dtype_hint(error)
        if hint is None:
            raise
        raise TransformError(str(error), hint) from error


def _stage_failure(error: BaseException) -> str:
    """What a chain raised, with the hint appended where one applies."""
    hint = _torch_dtype_hint(error)
    return f"{type(error).__name__}: {error}" + (f". {hint}" if hint else "")


def _shares_h5_file(source: Dataset, destination: Dataset) -> bool:
    """Whether a read of ``source`` and a stream into ``destination`` can open the same h5 file.
    By root: a directory root holds one file per case, and the same root is the same file."""
    return (
        source.file_format == "h5" and destination.file_format == "h5" and source.store_root == destination.store_root
    )


def _sweep_header(evolved: Attribute, scope: Attribute, keys_before: set[str]) -> Attribute:
    """The header a sweep publishes: the plan-time case state, completed by what the chain's
    stages added in ``__call__``: the same keys the whole-volume pass would leave at the Save."""
    attributes = Attribute(evolved)
    for key in scope.keys():
        if key not in keys_before and key not in attributes:
            attributes[key] = scope[key]
    return attributes


def _channel_first_block(block: np.ndarray, spatial: list[int], header: Attribute, what: str) -> np.ndarray:
    """The block a region write ships: channel-first, single-channel where the header says so
    (:func:`as_channel_first`, the rule the whole-volume write applies), else refused.

    Writing another rank anyway is the worst outcome available: the header would take the block's
    first spatial extent for a channel count and publish a store of that many "channels", raising
    nothing, while the whole-volume path returns the right rank: the two would silently disagree."""
    block = as_channel_first(block, header)
    if block.ndim != len(spatial) + 1:
        raise PatchError(
            f"{what} returned a rank-{block.ndim} block where the channel-first layout needs rank"
            f" {len(spatial) + 1} (C, {', '.join(str(extent) for extent in spatial)}).",
            "A transform that reduces the leading axis must keep it (`keepdim=True`), so a block"
            " stays C[Z]YX; only then does a region write mean the same thing as the"
            " whole-volume pass.",
        )
    return block


def _open_sweep_stream(
    sweep: _PendingSweep, block: np.ndarray, spatial: list[int], tile: Sequence[int], attributes: Attribute
) -> DataStream:
    """One region-write stream shaped for the sweep: the store chunks divide the block the sweep
    writes (channels included), so no region write ever pays a read-modify-write."""
    stream = sweep.destination.open_data_stream(
        sweep.group,
        sweep.entry,
        [int(block.shape[0]), *spatial],
        block.dtype,
        attributes,
        region_shape=[int(block.shape[0]), *(int(extent) for extent in tile)],
    )
    if stream is None:
        raise PatchError(
            f"destination '{sweep.destination.filename}' refused the region write of"
            f" '{sweep.group}/{sweep.entry}' after accepting its plan.",
            "h5 and omezarr always serve region writes; mha only with image geometry.",
        )
    return stream
