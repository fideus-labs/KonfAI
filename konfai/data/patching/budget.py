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
from dataclasses import dataclass

import torch

from konfai.utils.budget import peak_resident_bytes, reset_resident_peak, resident_bytes

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

# Rows per Save-sweep region: what bounds the materialization to a window while the composed region
# reads stay chunk-friendly. A declared memory_budget can only LOWER the height (see
# DatasetManager._sweep_rows); this constant is the cap and the no-budget default on a CPU. The
# VOLUME it allows is what DatasetManager._sweep_tile then shapes into the block actually read.
SWEEP_SLAB_ROWS = 64

# "not looked up yet", where None is itself an answer (a store with no read granularity to state).
_UNRESOLVED = object()

# The bytes each element travels as through a sweep (float32). What a sweep holds in those elements
# is _sweep_resident_regions, and DatasetManager.sweep_block_bytes prices it.
#
# What a streamed case holds beside its regions, whatever they are: the manager's state, the chain's
# stage objects, the store handles, the allocator's slack. Measured as the resident set above the
# interpreter floor that the priced regions do not account for, on the memory-limit cohort: 25 MiB
# (a bare Write), 8 (Gradient), 11 (Resample), 27 (Resample then Gradient). A floor the sizing cannot
# lower, so the TRANSFORM plan's header states it instead of leaving it to be found in a resident set.
SWEEP_ENGINE_FLOOR_BYTES = 32 << 20
#: The most blocks a sweep keeps in flight, whatever the budget leaves room for: past a second
#: one the jitter it absorbs is already absorbed (DatasetManager._sweep_depth).
_SWEEP_MAX_DEPTH = 3
#: The region height cap when the chain runs on a GPU (bounded by free device memory as well).
_SWEEP_SLAB_ROWS_DEVICE = 256
#: How far above the floor a decomposition's reads may sit for its height to count as the plateau
#: (:func:`_plateau_rows`). Measured on the prep's appearance fold, a native volume resampled
#: through a field onto a 514x1331x1775 grid, reads against height as a multiple of the floor:
#: 5 rows 1.79x, 10 1.40, 20 1.19, 41 1.09, 82 1.05, 169 1.02, 514 1.00. The curve is a knee, so
#: anything from a few per cent to a tenth names the same height; five was measured to hold the
#: fold at 17.4 GiB where an uncapped one held 28.7, for 56.9 -> 58.4 s of wall clock.
_PLATEAU_READ_MARGIN = 0.05
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
    same question, so a caller asks one thing and never branches on which it got.
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
    resident = resident_bytes()
    return None if resident is None else HeldMeter(peak_resident_bytes, int(resident))
