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
#: GLOBAL_STAT stage declares.
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
# a budget.
SWEEP_SLAB_ROWS = 64


# "not looked up yet", where None is itself an answer. A by-name ``__reduce__``, not a bare
# ``object()``: the manager is pickled into every DataLoader worker, and a plain ``object()``
# unpickles as a new instance.
class _Unresolved:
    __slots__ = ()

    def __reduce__(self) -> str:
        return "_UNRESOLVED"

    def __repr__(self) -> str:
        return "_UNRESOLVED"


_UNRESOLVED = _Unresolved()

# What a streamed case holds beside its regions: the manager's state, the chain's stage objects, the
# store handles, the allocator's slack. Measured at up to 27 MiB above the interpreter floor. The
# TRANSFORM plan's header states it.
SWEEP_ENGINE_FLOOR_BYTES = 32 << 20
#: How many units a region grows to at most, a unit being the store's block along the sweep axis
#: (``SWEEP_SLAB_ROWS`` on a store without one). The wall is flat past four to eight chunk rows.
GROWTH_CAP_UNITS = 8
#: The share of the budget the first region is priced against. The worst overshoot on record is a
#: first region at 1.5x its price; from a half that lands at 0.75 of the budget, and the growth finds
#: the rest in one doubling.
_START_SHARE = 0.5
#: A region measured under this share of the budget doubles the next. A third, not a half: a doubled
#: region holds up to 2.4x (the landing buffers, the chain's temporaries and the allocator's slack
#: grow with it). One measured over the budget halves the next.
_GROW_BELOW = 1.0 / 3.0
#: How much less a cubic block must read for the sweep to take it (``DatasetManager._sweep_tile``).
_SWEEP_TILE_MARGIN = 0.8
#: The bytes each element travels as through a sweep (float32).
_SWEEP_ELEMENT_BYTES = 4

#: What a whole-volume fallback holds while a case is in flight (the assembled tensor plus one
#: transform output), and the bytes each element travels as. Public: the run-time budget check
#: (``CaseMaterializer._enforce_fallback_budget``) and the TRANSFORM plan must agree on the figure.
FALLBACK_INFLIGHT_FACTOR = 2
CASE_ELEMENT_BYTES = 4


@dataclass
class RegionGrowth:
    """The height of the regions a route cuts, decided by what the last one HELD.

    Every region measured under ``_GROW_BELOW`` of the budget doubles the next, one measured over the
    budget halves it, never above ``cap`` and never below the first, so every region starts on a
    multiple of the first and the output's chunk grid is never straddled. Without a budget, or
    without an instrument to read, the height stands.
    """

    rows: int
    cap: int
    budget_bytes: float | None
    #: Regions measured at the current height before it may double again: a pipelined sweep holds
    #: ``depth + 2`` regions in flight, and a reading taken before that many have run at a height
    #: says what the previous height cost.
    settle: int = 1
    first: int = field(init=False)
    _at_height: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.rows = max(1, int(self.rows))
        self.cap = max(self.rows, int(self.cap))
        self.settle = max(1, int(self.settle))
        self.first = self.rows

    def halve_below_first(self) -> int:
        """Half the height, the first region's included: the answer to an OutOfMemoryError. The new
        height is the floor from here on."""
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
    """Whether a region ``device`` cannot hold raises a catchable ``OutOfMemoryError``: a CUDA device
    does, the host gets no signal (the kernel kills)."""
    return device is not None and device.type == "cuda"


def device_capped_budget(budget_bytes: float | None, device: "torch.device | None") -> float | None:
    """The budget, capped at what ``device`` can hold.

    The memory budget is declared in HOST bytes, but on a GPU chain the working sets it sizes live
    in VRAM. The cap is half of what the card can give THIS process: the free memory plus what the
    process's own allocator already holds, so the figure does not move with when it is read.
    """
    if device is None or device.type != "cuda" or not torch.cuda.is_available():
        return budget_bytes
    free, _total = torch.cuda.mem_get_info(device)
    vram = (free + torch.cuda.memory_reserved(device)) * 0.5
    return vram if budget_bytes is None or budget_bytes <= 0 else min(budget_bytes, vram)


@dataclass(frozen=True)
class HeldMeter:
    """What one scope of work HELD, read by the instrument the route it runs on has.

    A GPU chain reads the device allocator; a host chain reads the kernel's resident high-water mark,
    above the run's resident floor when the workflow recorded one
    (:func:`~konfai.utils.budget.record_resident_floor`), else above where the scope started.
    """

    _peak: Callable[[], int | None]
    _baseline: int

    def held(self) -> int | None:
        """Bytes held above where the scope started, or ``None`` if the instrument went quiet."""
        peak = self._peak()
        return None if peak is None else max(0, peak - self._baseline)


def open_held_meter(device: "torch.device | None") -> HeldMeter | None:
    """Start measuring what the next scope holds, or ``None`` where nothing can measure it. The reading
    is a high-water mark, so it bounds the NEXT scope of the same shape from above."""
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
    # Above the run's resident floor where the workflow recorded one, else above where the scope starts.
    resident = resident_floor() if resident_floor() is not None else resident_bytes()
    if resident is None:
        return None

    # The decoded caches sit inside the host peak and have a budget line of their own
    # (BUDGET_SHARES['cache']): what they took during the scope is not the scope's.
    def cache_held_bytes() -> int:
        from konfai.utils.dicom import plane_cache_held_bytes
        from konfai.utils.ome_zarr import chunk_cache_held_bytes

        return chunk_cache_held_bytes() + plane_cache_held_bytes()

    cache_at_start = cache_held_bytes()

    def resident_peak_less_cache() -> int | None:
        peak = peak_resident_bytes()
        if peak is None:
            return None
        return peak - max(0, cache_held_bytes() - cache_at_start)

    return HeldMeter(resident_peak_less_cache, int(resident))
