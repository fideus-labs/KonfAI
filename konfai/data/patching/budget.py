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


"""What a sweep may hold: the constants it is priced with, the device cap, the held-memory meter."""

from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from konfai.utils.budget import peak_resident_bytes, reset_resident_peak, resident_bytes, resident_floor

#: The whole-volume statistics every store can serve (``Dataset.read_data_statistics``), by the key a
#: GLOBAL_STAT stage declares: what the plan checks against without reading a voxel, and what the
#: seed reads. The per-channel figures are what a vector-valued quantity needs, where pooling the
#: components into one number describes nothing.
_STREAM_STATS = {
    "Min": "min",
    "Max": "max",
    "Mean": "mean",
    "Std": "std",
    "MinPerChannel": "min_per_channel",
    "MaxPerChannel": "max_per_channel",
    "MeanPerChannel": "mean_per_channel",
    "StdPerChannel": "std_per_channel",
}
_STREAM_STAT_KEYS = frozenset(_STREAM_STATS)

# The unit a region grows by on a store without a block grid, and the height a sweep keeps without
# a budget: with one, the first region is priced (SegmentSizer.growth) and the next ones follow
# what the last held (RegionGrowth), up to GROWTH_CAP_UNITS of it.
SWEEP_SLAB_ROWS = 64


# "not looked up yet", where None is itself an answer (a store with no read granularity to state).
# A class with a by-name ``__reduce__`` rather than a bare ``object()``: the manager is pickled into
# every DataLoader worker (spawn), and a plain ``object()`` unpickles as a NEW instance, so the
# ``is _UNRESOLVED`` test failed in the worker and the sentinel itself got indexed
# (``'object' object is not subscriptable`` in ``_sweep_rows``).
class _Unresolved:
    __slots__ = ()

    def __reduce__(self) -> str:
        return "_UNRESOLVED"

    def __repr__(self) -> str:
        return "_UNRESOLVED"


_UNRESOLVED = _Unresolved()

# The bytes each element travels as through a sweep (float32). What a sweep holds in those elements
# is _sweep_resident_regions, and DatasetManager.sweep_block_bytes prices it.
#
# What a streamed case holds beside its regions, whatever they are: the manager's state, the chain's
# stage objects, the store handles, the allocator's slack. Measured as the resident set above the
# interpreter floor that the priced regions do not account for, on the memory-limit cohort: 25 MiB
# (a bare Write), 8 (Gradient), 11 (Resample), 27 (Resample then Gradient). A floor the sizing cannot
# lower, so the TRANSFORM plan's header states it instead of leaving it to be found in a resident set.
SWEEP_ENGINE_FLOOR_BYTES = 32 << 20
#: How many units a region grows to at most, a unit being the store's block along the sweep axis
#: (``SWEEP_SLAB_ROWS`` on a store without one). Measured on a 2 GiB h5 volume chunked at 64
#: rows (``benchmarks/perf/bench_transform.py``, 2026-09-07): the wall halves from a sub-chunk
#: region to four-to-eight chunk rows (15.6 -> 7.6 s) and is flat past that; on the 513-row
#: ExaSPIM store chunked at 256 rows the knee sits at one chunk row (24 s under 1 GiB, 5.9 s
#: under 8 GiB, flat at 16), and a region under the chunk runs, slower, where a refusal would not.
GROWTH_CAP_UNITS = 8
#: The share of the budget the first region is priced against. The price is a model of what a
#: chain holds, and the worst overshoot on record is a first region at 1.5x its price (a fold over
#: registration fields), on a host that kills without a MemoryError: from a half, that lands at
#: 0.75 of the budget, and the growth finds the rest in one doubling. An eighth was measured to
#: cost a 513-row store two of its three regions (12.2 s against 5.9 under 8 GiB).
_START_SHARE = 0.5
#: A region measured under this share of the budget doubles the next. A third, not a half: what
#: a doubled region holds is not double (the landing buffers, the chain's temporaries and the
#: allocator's slack grow with it), measured at 2.4x on a 2 GiB h5 sweep whose 128-row regions
#: held 0.42 GiB and whose 256-row ones held 1.00; doubled from under a third, the next lands under
#: the budget at that ratio. One measured over the budget halves the next.
_GROW_BELOW = 1.0 / 3.0
#: How much less a cubic block must read for the sweep to take it (``DatasetManager._sweep_tile``).
#: A sheared map measures 0.61 on a 513x1331x1776 rigid+affine; an unsheared one, the margin alone.
_SWEEP_TILE_MARGIN = 0.8
_SWEEP_ELEMENT_BYTES = 4

#: What a whole-volume fallback holds while a case is in flight (the assembled tensor plus one
#: transform output), and the bytes each element travels as.
#:
#: Public because two callers must agree on the figure and neither owns it: the run-time budget check
#: (``CaseMaterializer._enforce_fallback_budget``) refuses a case against it, and the TRANSFORM plan
#: prints and enforces the same number before a byte is written. A plan estimating differently from
#: the run it describes is worse than no plan.
FALLBACK_INFLIGHT_FACTOR = 2
CASE_ELEMENT_BYTES = 4


@dataclass
class RegionGrowth:
    """The height of the regions a route cuts, decided by what the last one HELD.

    The price starts the first region small (``_START_SHARE``); every region measured under a third of
    the budget doubles the next, one measured over the budget halves it, never above ``cap`` and
    never below the first, so every region starts on a multiple of the first and the output's
    chunk grid, cut on the first, is never straddled. Without a budget, or without an instrument
    to read, the height stands.
    """

    rows: int
    cap: int
    budget_bytes: float | None
    #: Regions measured at the current height before it may double again: a pipelined sweep holds
    #: ``depth + 2`` regions in flight, and a reading taken before that many have run at a height
    #: says what the previous height cost. Doubled twice on such readings, a sweep of a 2 GiB
    #: volume reached 340-row regions and 1.52 GiB held under a 1 GiB budget before the meter
    #: caught up, and the high-water mark then held it at the first height for the rest.
    settle: int = 1
    first: int = field(init=False)
    _at_height: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.rows = max(1, int(self.rows))
        self.cap = max(self.rows, int(self.cap))
        self.settle = max(1, int(self.settle))
        self.first = self.rows

    def halve_below_first(self) -> int:
        """Half the height, the first region's included: the answer to a region the device could not
        hold at all (an OutOfMemoryError), where the measured rule only ever halves down to the
        first. The new height is the floor from here on."""
        self.rows = max(1, self.rows // 2)
        self.first, self._at_height = self.rows, 0
        return self.rows

    def after(self, held: int | None) -> int:
        """The height of the regions cut after one that held ``held`` bytes."""
        budget = self.budget_bytes
        self._at_height += 1
        if held is None or not budget or budget <= 0:
            return self.rows
        if held > budget:
            rows = max(self.first, self.rows // 2)
        elif held < budget * _GROW_BELOW and self.rows < self.cap and self._at_height >= self.settle:
            rows = min(self.cap, self.rows * 2)
        else:
            rows = self.rows
        if rows != self.rows:
            self.rows, self._at_height = rows, 0
        return self.rows


def device_signals_oom(device: "torch.device | None") -> bool:
    """Whether a region ``device`` cannot hold is a catchable ``OutOfMemoryError``: a CUDA device
    raises one, the host gets no signal (the kernel kills). What the halve-on-OOM retry asks."""
    return device is not None and device.type == "cuda"


def device_capped_budget(budget_bytes: float | None, device: "torch.device | None") -> float | None:
    """The budget, capped at what ``device`` can actually hold.

    The memory budget is declared in HOST bytes -- ``auto`` measures node RAM -- but on a GPU
    chain the working sets it sizes (swept slabs, whole-volume fallbacks, a reduction's member
    regions) live in VRAM. A 64G budget on a 16 GB card is then not a budget, it is a promise of
    an OOM. Half of what the card can give THIS process: the halving is the slack that covers
    allocations arriving after the reading, since a fold sized once can run for hours.

    What the card can give this process is the free memory plus what this process's own allocator
    is already sitting on, because a cached block is memory the next allocation reuses rather than
    asks the driver for. Reading the free memory alone made the answer depend on WHEN it was read:
    the same fold, on the same idle card, sized its regions at 47 rows under 4.99 GiB in one run
    and 65 rows under 11.58 GiB in another, the difference being how much the process had already
    reserved by the time the fit ran. Region height decides the whole read plan, so a sizing that
    moves with the moment is a run whose cost cannot be reproduced or reasoned about.
    """
    if device is None or device.type != "cuda" or not torch.cuda.is_available():
        return budget_bytes
    free, _total = torch.cuda.mem_get_info(device)
    vram = (free + torch.cuda.memory_reserved(device)) * 0.5
    return vram if budget_bytes is None or budget_bytes <= 0 else min(budget_bytes, vram)


@dataclass(frozen=True)
class HeldMeter:
    """What one scope of work HELD, read by the instrument the route it runs on has.

    A GPU chain has the device allocator, which counts what is in use; a host chain has the kernel's
    resident high-water mark, which counts what the allocator is sitting on as well -- and that is
    the better figure of the two here, since the kernel kills on resident memory. Both answer the
    same question, so a caller asks one thing and never branches on which it got. On the host the
    baseline is the run's resident floor when the workflow recorded one
    (:func:`~konfai.utils.budget.record_resident_floor`), else where the scope started.
    """

    _peak: Callable[[], int | None]
    _baseline: int

    def held(self) -> int | None:
        """Bytes held above where the scope started, or ``None`` if the instrument went quiet."""
        peak = self._peak()
        return None if peak is None else max(0, peak - self._baseline)


def open_held_meter(device: "torch.device | None") -> HeldMeter | None:
    """Start measuring what the next scope holds, or ``None`` where nothing can measure it.

    Reading is a high-water mark either way, so what comes back bounds the NEXT scope of the same
    shape from above -- which is what makes a measured figure usable for sizing the ones that follow
    (:meth:`Predictor._accumulate_device` reads a forward the same way).
    """
    if device is not None and device.type == "cuda" and torch.cuda.is_available():
        try:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            baseline = int(torch.cuda.memory_allocated(device))
        except Exception:  # nosec B110 - an unreadable instrument measures nothing, and says so
            return None

        def device_peak() -> int | None:
            try:
                torch.cuda.synchronize(device)
                return int(torch.cuda.max_memory_allocated(device))
            except Exception:  # nosec B110 - see above
                return None

        return HeldMeter(device_peak, baseline)
    if not reset_resident_peak():
        return None
    # Above the run's resident floor where the workflow recorded one, else above where the scope
    # starts: a region reusing the pages an earlier one freed holds them all the same, and only
    # a baseline taken before any region was read counts them.
    resident = resident_floor() if resident_floor() is not None else resident_bytes()
    if resident is None:
        return None
    # THE CACHE IS NOT THE SCOPE'S. A host peak is the whole process's high-water mark, and the
    # decoded-chunk cache sits inside it: a scope that reads from a store fills the cache on its
    # way, and the cache keeps what it decoded past the scope, for the next one. Charging the scope
    # for that is charging it for a budget line that has its own share (BUDGET_SHARES['cache']).
    # Measured on a fold's probe over ten native members: 24.4 GiB read, 13.2 of it the cache
    # filling from empty, and the fold cut to 78 % of the height its regions actually needed.
    from konfai.utils.ome_zarr import chunk_cache_held_bytes

    cache_at_start = chunk_cache_held_bytes()

    def resident_peak_less_cache() -> int | None:
        peak = peak_resident_bytes()
        if peak is None:
            return None
        return peak - max(0, chunk_cache_held_bytes() - cache_at_start)

    return HeldMeter(resident_peak_less_cache, int(resident))
