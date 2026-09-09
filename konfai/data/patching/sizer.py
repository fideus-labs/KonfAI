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

"""The sweep's pricing engine, keyed to the segment it prices.

The sizing once lived on the :class:`~konfai.data.patching.manager.DatasetManager` and read manager
state: the WHOLE declared chain's channel folds and the RAW source's read granularity. A segment
past a ``Save`` boundary was therefore priced with another segment's facts -- channel folds applied
twice onto a cache that already holds them, the wrong store's chunk grid, and a copy's draws priced
at zero. A :class:`SegmentSizer` is constructed per segment from explicit inputs, so the price can
only read the segment's own facts -- and it needs no dataset fixture to be tested.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import torch

from konfai.data.patching import budget
from konfai.data.patching import sweep as sweep_module
from konfai.data.patching.budget import (
    _START_SHARE,
    _SWEEP_ELEMENT_BYTES,
    _SWEEP_TILE_MARGIN,
    RegionGrowth,
)
from konfai.data.patching.stage import Stage, _ReadStagePlan
from konfai.data.patching.sweep import (
    BlockReads,
    _cubic_tile,
    _pull_block_spans,
    _span_voxels,
    _sweep_resident_regions,
)
from konfai.utils.budget import format_bytes
from konfai.utils.dataset import chunk_hull_voxels as _chunk_hull_voxels
from konfai.utils.errors import DatasetManagerError


@dataclass
class SegmentSizer:
    """Prices one sweep segment: what a decomposition reads, what a block holds, what fits.

    ``spatial``/``channels``/``plans``/``stages`` are the segment's own landing, source channels,
    region plans and stage list; ``granularity`` is the segment's OWN store's decode grain (spatial
    axes, ``None`` when a read costs what it asks for -- including a cache this run has still to
    write, whose chunks will be the very tile being sized, so its reads align by construction).
    ``block_reads_memo`` is shared across sizers by the owning manager: the geometry walk is the
    expensive part and its key already carries everything a sizer varies.
    """

    spatial: list[int]
    channels: int
    plans: tuple[_ReadStagePlan, ...]
    stages: tuple[Stage, ...]
    granularity: tuple[int, ...] | None
    case: str
    group: str
    budget_bytes: float | None
    device: torch.device | None
    block_reads_memo: dict[tuple, tuple[tuple, BlockReads]] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------ chain facts

    def chain_channels(self) -> tuple[int, int, int, float]:
        """What the channel axis costs along the SEGMENT's stages: the channels the source pulls,
        the channels a block lands with, the widest the segment ever holds, and the volumes-worth
        its widest stage allocates, that one counted on the channels that stage is handed.

        Identity for a segment that keeps the axis. ``OneHot`` is the stage that widens it, and a
        block priced at the source's would be short by its class count. Every stage of the segment
        answers -- a copy's draws included: priced at zero, an Expand copy swept under a budget
        that never heard of its ``grid_sample`` buffers.
        """
        source = held = landed = peak = max(1, int(self.channels))
        working = 0.0
        for stage in self.stages:
            multiple = getattr(stage, "case_working_multiple", None)
            fold = getattr(stage, "output_channels", None)
            if multiple is None or fold is None:
                continue
            working = max(working, float(multiple(self.case)) * held)
            held = landed = max(1, int(fold(held)))
            peak = max(peak, held)
        return source, landed, peak, working

    # ------------------------------------------------------------------ reads

    def _source_extents(self) -> list[int]:
        """The extents the pull spans live in: the first stage's own input, which is the stored
        volume. The landing is a different grid, and a hull capped against it is under-charged
        wherever the source is the larger of the two."""
        if self.plans:
            return [int(extent) for extent in self.plans[0].in_shape]
        return [int(extent) for extent in self.spatial]

    def block_reads(self, tile: Sequence[int]) -> BlockReads:
        """What a decomposition of the landing into ``tile`` reads, walked once and kept.

        The sizing asks the same question of the same decomposition several times over -- the shape
        rule prices the slab and the cube, the ladder is priced height by height -- and every one
        of those goes through the chain's pull maps, which for a ``Resample`` is real geometry per
        block. Keyed by the decomposition AND by the plans
        that map it, whose tuple is held so no identity is reused under the key.
        """
        key = (tuple(self.spatial), tuple(tile), tuple(id(plan) for plan in self.plans), self.granularity)
        held = self.block_reads_memo.get(key)
        if held is not None:
            return held[1]
        extents = self._source_extents() if self.granularity is not None else []
        widest_pull = widest_hull = total = 0
        for span in _pull_block_spans(list(self.spatial), tile, self.plans):
            pull = _span_voxels(span)
            hull = pull if self.granularity is None else _chunk_hull_voxels(span, self.granularity, extents)
            widest_pull, widest_hull, total = max(widest_pull, pull), max(widest_hull, hull), total + hull
        reads = BlockReads(widest_pull, widest_hull, total)
        self.block_reads_memo[key] = (tuple(self.plans), reads)
        return reads

    def decomposition_reads(self, tile: Sequence[int]) -> int:
        """What sweeping the landing in ``tile`` reads from the store, all blocks together.

        The store's own currency: a chunked backend decodes whole blocks, so what a decomposition
        reads is the sum of its blocks' hulls, and a shape is judged on the same figure it is later
        priced with (:meth:`sweep_block_bytes`). Two currencies here and there is how a shape gets
        chosen for pulling little and then costs what its hull costs.
        """
        return self.block_reads(tile).total

    def sweep_block_bytes(self, tile: list[int], depth: int) -> int:
        """What a sweep decomposed into ``tile`` holds at its peak: the source regions it has pulled
        and the blocks it has landed, both counted by :func:`_sweep_resident_regions`, plus what the
        widest stage of the segment allocates on top of the largest of them. Each term is counted on
        the channels it actually holds (:meth:`chain_channels`), at ``_SWEEP_ELEMENT_BYTES`` each.
        Beside this, and outside it, a streamed case holds ``SWEEP_ENGINE_FLOOR_BYTES`` the
        decomposition cannot lower.
        """
        pulled, landed = _sweep_resident_regions(depth)
        block = int(np.prod(tile, dtype=np.int64))
        reads = self.block_reads(tile)
        pull = reads.widest_pull or block
        source, landed_channels, _peak, working = self.chain_channels()
        held = pulled * pull * source + landed * block * landed_channels + working * max(pull, block)
        # A chunked store serves a window by decoding the block-aligned hull that covers it, and
        # assembles the window out of that: one read is in flight at a time, so the hull is resident
        # ONCE, and the window is the part of it the chain keeps. What a straddling region costs is
        # exactly this term, and it does not fall when the region does -- below one stored block a
        # shorter region reads the same bytes and only reads them more often.
        held += reads.widest_excess * source
        return int(held * _SWEEP_ELEMENT_BYTES)

    # ------------------------------------------------------------------ the search

    def sweep_shape(self, rows: int) -> list[int]:
        """The block ``rows`` rows of the landing become: the slab itself, or the cube of the same
        volume where that pulls less.

        A region pulls the BOUNDING BOX of its own image under the chain's maps, so a slab spanning
        the trailing plane pays that plane's extent for every degree of shear where a cube pays its
        side: 1.79x the image against 1.09x on a 513x1331x1776 rigid+affine. Both are priced against
        the plans' own pull maps, and the cube wins only by ``_SWEEP_TILE_MARGIN``: the decomposition
        is also the shape a store gets chunked in. Without plans, the slab.
        """
        slab, cube = self._slab(rows), self._cube(rows)
        if cube == slab or not self.plans:
            return slab
        cheaper = self.decomposition_reads(cube) <= (self.decomposition_reads(slab) * _SWEEP_TILE_MARGIN)
        return cube if cheaper else slab

    def unit_rows(self) -> int:
        """What a region grows by: the store's block along the sweep axis, else ``SWEEP_SLAB_ROWS``.
        A region a whole number of blocks tall reads each stored block once; one that straddles the
        grid decodes both blocks it touches, for every region."""
        block = int(self.granularity[0]) if self.granularity is not None else 1
        return block if block > 1 else int(budget.SWEEP_SLAB_ROWS)

    def cap_rows(self) -> int:
        """The tallest region the growth reaches: ``GROWTH_CAP_UNITS`` units, the landing at most."""
        units = budget.GROWTH_CAP_UNITS * max(self.unit_rows(), int(budget.SWEEP_SLAB_ROWS))
        return max(1, min(int(self.spatial[0]), units))

    def _priced(self, rows: int, depth: int) -> int:
        return self.sweep_block_bytes(self.sweep_shape(rows), depth)

    def _slab(self, rows: int) -> list[int]:
        return [min(int(rows), int(self.spatial[0])), *(int(extent) for extent in self.spatial[1:])]

    def _cube(self, rows: int) -> list[int]:
        from konfai.utils.ome_zarr import CHUNK_SPATIAL_TILE

        voxels = int(rows) * int(np.prod(self.spatial[1:], dtype=np.int64))
        return _cubic_tile(self.spatial, voxels, CHUNK_SPATIAL_TILE)

    def _tallest(self, allowance: float, depth: int, shape: Callable[[int], list[int]]) -> int | None:
        """The tallest height up to the cap whose priced block, in ``shape``, holds inside
        ``allowance``; ``None`` when one row does not. Bisected on the price itself: none of what a
        region costs scales with its rows (a halo is a constant, a rotated map's box grows with
        the diagonal, a chunked store decodes whole blocks whatever the height)."""
        if self.sweep_block_bytes(shape(1), depth) > allowance:
            return None
        low, high = 1, self.cap_rows()
        while low < high:
            middle = (low + high + 1) // 2
            if self.sweep_block_bytes(shape(middle), depth) <= allowance:
                low = middle
            else:
                high = middle - 1
        return low

    def start(self, depth: int | None = None) -> tuple[list[int], RegionGrowth]:
        """The block the first region covers and how the regions grow from it (:class:`RegionGrowth`).

        The shape is decided at the height the whole budget buys (:meth:`sweep_shape`: the slab, or
        the cube where a sheared map makes it pull less), because the shape is a fact of the pull
        geometry and not of the memory. A slab then starts at the tallest height whose price holds
        inside ``_START_SHARE`` of the budget, snapped down to a whole number of the store's blocks
        when one fits, and grows from there; a cube starts where the budget's own price puts it,
        since it cannot grow across the plane and would pay a small start for the whole case
        (measured: 84 cubes of 256^3 against 37 slabs, +25 % of wall on a 513-row store under 8
        GiB). Without a budget the unit stands, as it always did. The budget is what a sweep may
        HOLD, so it is the priced block (:meth:`sweep_block_bytes`) that is held to it, never the
        landed rows alone.

        A budget one row does not fit is a refusal naming both figures, with the read-ahead given
        up first: the queue is the one part of the price the sizing chose, and a sweep about to
        refuse has no clock to buy with it.
        """
        depth = sweep_module._sweep_pipeline_depth() if depth is None else depth
        budget_bytes = self.budget_bytes
        cap = self.cap_rows()
        if not budget_bytes or budget_bytes <= 0:
            rows = min(self.unit_rows(), cap)
            return self.sweep_shape(rows), RegionGrowth(rows, cap, None)
        full = self._tallest(budget_bytes, depth, self.sweep_shape)
        if full is None:
            held = self.sweep_block_bytes(self.sweep_shape(1), depth)
            serial = self.sweep_block_bytes(self.sweep_shape(1), 0) if depth > 0 else None
            if serial is not None and serial <= budget_bytes:
                return self.sweep_shape(1), RegionGrowth(1, cap, budget_bytes)
            raise DatasetManagerError(
                f"'{self.case}': no region of '{self.group}' fits the per-rank memory budget"
                f" ({format_bytes(budget_bytes)}): the smallest one this chain can sweep holds"
                f" {format_bytes(held)}"
                + (f", and {format_bytes(serial)} with the read-ahead given up" if serial is not None else "")
                + ".",
                "Raise 'memory_budget'.",
            )
        if self.sweep_shape(full) != self._slab(full):
            return self._cube(full), RegionGrowth(full, cap, budget_bytes)
        rows = self._tallest(budget_bytes * _START_SHARE, depth, self._slab) or full
        block = int(self.granularity[0]) if self.granularity is not None else 1
        if 1 < block and rows < block <= full:
            # A region under one stored block holds the block's decoded hull all the same and only
            # lands less of it: where a whole block fits the budget, the block is the start.
            rows = block
        if 1 < block <= rows:
            rows -= rows % block  # a whole number of blocks reads each stored block once
        return self._slab(rows), RegionGrowth(rows, cap, budget_bytes)

    def growth(self, depth: int | None = None) -> RegionGrowth:
        return self.start(depth)[1]

    def sweep_depth(self, tile: list[int]) -> int:
        """How many blocks the sweep keeps in flight beside the one it transforms: the rank's
        pipeline depth while the priced block still holds inside the budget with it, none
        otherwise. The queue is bought, and a sweep that cannot afford it stops buying it: three
        source regions resident become one, a quarter to a third of the block on a chain whose
        stage buffers do not dominate."""
        depth = sweep_module._sweep_pipeline_depth()
        budget_bytes = self.budget_bytes
        if not depth or not budget_bytes or budget_bytes <= 0:
            return depth
        return depth if self.sweep_block_bytes(tile, depth) <= budget_bytes else 0

    def sweep_tile(self, depth: int | None = None) -> list[int]:
        """The block the first region covers (:meth:`start`): the chunk the output is cut in."""
        return self.start(depth)[0]
