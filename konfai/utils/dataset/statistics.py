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


"""Running statistics of a volume read block by block on the store's own grid."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Sequence
from functools import partial
from typing import Any

import numpy as np

from konfai.utils.budget import per_rank_budget_bytes

#: Elements a block of ``Dataset.iter_data_blocks`` holds when no budget was declared: the read grain
#: of a scan (the statistics fold, the quantile scan), whatever the backend.
_STATISTICS_CHUNK_ELEMENTS = 8_000_000
#: Blocks a scan holds at its peak: the map of the one being read, its copy, and the copy the fold has
#: not released yet, a generator reading the next while its caller still names the last. Measured at
#: 95.6 MiB of resident set for a 30.5 MiB block over a 78 MiB case.
#:
#: The scan itself keeps its side of the bargain: 95.4 / 96.0 / 67.1 / 34.4 MiB held over 512 / 128 /
#: 64 / 32 MiB declared, so it is under the budget at 128 and above (where :data:`_STATISTICS_CHUNK_
#: ELEMENTS` caps it) and about 1.1x over at 64 and 32. What a GLOBAL_STAT ROUTE holds is more: 183
#: MiB over the floor at 128 MiB declared, against the 154 the plan announces (regions 104 + engine
#: 32 + cache 18), so 1.19x. The scan frees (RSS falls back between the phases) and the sweep then
#: peaks on top of a residue.
#:
#: Not chased further here, and the reason is the instrument: ``VmHWM`` is a high-water mark of the
#: WHOLE resident set, so it cannot be compared with the ``RssAnon``/``RssFile`` split, which is
#: instantaneous -- and it is anonymous memory alone that an OOM kill weighs. Attributing this needs
#: a sampler for peak anonymous bytes across the phases, not another reading of the mark.
_STATISTICS_BLOCKS_IN_FLIGHT = 3
#: The bytes a scanned element is priced at when the source's own size is not known: what everything
#: else the budget sizes prices an element at. A block is the store's OWN dtype, never a cast copy,
#: so where the store can be asked (:meth:`Dataset._scanned_element_bytes`) it answers instead.
_STATISTICS_ELEMENT_BYTES = 4
#: Elements one running-statistics update takes at once: its float64 temporaries then stay in cache,
#: where a whole block's stream through memory. Below the block on purpose: the block is the READ
#: grain, and a chunked store decodes a chunk once per read that touches it.
_STATISTICS_UPDATE_ELEMENTS = 1 << 18


def chunk_hull_voxels(span: Sequence[slice], granularity: Sequence[int], shape: Sequence[int]) -> int:
    """The voxels a store of ``granularity`` materialises to serve ``span``.

    A chunked read decodes whole blocks and assembles the window out of them, so what it holds is
    the block-aligned hull of the window, never the window: a span that straddles two planes of the
    grid pays both in full, and one aligned to the grid pays exactly itself. The hull is capped at
    the array, so an axis a span covers entirely costs that axis and no more.

    The three sequences describe the same axes, in the same order.
    """
    hull = 1
    for part, block, extent in zip(span, granularity, shape, strict=True):
        low, high = max(0, part.start), min(int(extent), part.stop)
        if high <= low:
            return 0
        step = max(1, int(block))
        hull *= min(int(extent), -(-high // step) * step) - low // step * step
    return hull


def _scan_block_on_the_store_grid(
    rows: int,
    extent: int,
    plane: int,
    granularity: Sequence[int] | None,
    budget: float | None,
    element_bytes: int = _STATISTICS_ELEMENT_BYTES,
) -> tuple[int, int]:
    """The rows one scan block reads, and what reading it holds.

    A chunked store decodes whole blocks, so a scan stepping finer than the store's grain decodes
    the same block again at every step it takes inside it: measured at 1153 MiB decoded to serve a
    13.5 MiB window, 85x, and 170x where the step straddles two -- 212 reads over a volume five
    stored blocks deep. Where the budget can hold a whole stored block the grain is RAISED to it,
    which reads each block once; where it cannot, the grain stays and what the store decodes is
    CHARGED, so an impossible scan is refused by the plan instead of by the kernel.
    """
    block = max(1, int(granularity[0])) if granularity else 0

    def decoded(step: int) -> int:
        """Rows the store materialises to serve one step, at the worst place the walk puts it."""
        if not block:
            return step
        return max(
            chunk_hull_voxels([slice(start, min(extent, start + step))], [block], [extent])
            for start in range(0, extent, step)
        )

    def held_for(step: int) -> int:
        """Blocks in flight, plus what the read in flight decodes above the step it serves."""
        resident = step * _STATISTICS_BLOCKS_IN_FLIGHT + max(0, decoded(step) - step)
        return int(resident * plane * element_bytes)

    if block:
        aligned = max(block, rows // block * block)
        held = held_for(aligned)
        if budget is None or held <= budget:
            return aligned, held
    return rows, held_for(rows)


def _statistics_block_elements(element_bytes: int = _STATISTICS_ELEMENT_BYTES) -> int:
    """Elements one block of a whole-volume scan may hold: its share of the budget this rank
    published, since :data:`_STATISTICS_BLOCKS_IN_FLIGHT` of them are in flight at the peak, each of
    them ``element_bytes`` an element. A fixed read grain made a scan cost the same 95.6 MiB
    whatever the budget said. Without a declared budget, the grain that keeps a chunked store's
    decode whole.
    """
    budget = per_rank_budget_bytes()
    if budget is None:
        return _STATISTICS_CHUNK_ELEMENTS
    return max(1, min(_STATISTICS_CHUNK_ELEMENTS, int(budget / (_STATISTICS_BLOCKS_IN_FLIGHT * element_bytes))))


def _statistics_plane_elements(shape: list[int] | tuple[int, ...], axis: int) -> int:
    """What one step along ``axis`` costs: a chunk spans every other axis whole (channels included)."""
    return int(np.prod([extent for other, extent in enumerate(shape) if other != axis], dtype=np.int64))


def _statistics_chunk_length(shape: list[int] | tuple[int, ...], axis: int, budget: int) -> int:
    """How far along ``axis`` a chunk may reach to hold about ``budget`` elements: that budget over
    the per-step cost, floored to one step."""
    return max(1, budget // max(1, _statistics_plane_elements(shape, axis)))


def _update_pieces(block: np.ndarray) -> Iterator[np.ndarray]:
    """``block`` in pieces of about ``_STATISTICS_UPDATE_ELEMENTS`` along its first spatial axis, one
    running-statistics update each; a vector is one piece."""
    if block.ndim < 2:
        yield block
        return
    rows = _statistics_chunk_length(block.shape, 1, _STATISTICS_UPDATE_ELEMENTS)
    for start in range(0, block.shape[1], rows):
        yield block[:, start : start + rows]


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
