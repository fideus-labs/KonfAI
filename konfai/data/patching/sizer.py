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

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import torch

from konfai.data.patching import budget
from konfai.data.patching import sweep as sweep_module
from konfai.data.patching.budget import _SWEEP_ELEMENT_BYTES, _SWEEP_SLAB_ROWS_DEVICE, _SWEEP_TILE_MARGIN
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

        The sizing asks the same question of the same decomposition many times over -- the shape
        search prices each candidate and then judges its reads, the height search bisects, and the
        plateau walks a ladder -- and every one of those goes through the chain's pull maps, which
        for a ``Resample`` is real geometry per block. Keyed by the decomposition AND by the plans
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
        from konfai.utils.ome_zarr import CHUNK_SPATIAL_TILE

        spatial = self.spatial
        slab = [min(int(rows), int(spatial[0])), *(int(extent) for extent in spatial[1:])]
        voxels = int(rows) * int(np.prod(spatial[1:], dtype=np.int64))
        cube = _cubic_tile(spatial, voxels, CHUNK_SPATIAL_TILE)
        if cube == slab or not self.plans:
            return slab
        cheaper = self.decomposition_reads(cube) <= (self.decomposition_reads(slab) * _SWEEP_TILE_MARGIN)
        return cube if cheaper else slab

    def grid_rows(self, cap: int) -> list[int]:
        """The heights that land on the store's block grid, up to ``cap``.

        A decomposition aligned to the grid reads each stored block exactly once; one that straddles
        reads both blocks it touches, for every region, and holds the larger hull. There are only a
        handful of such heights under any cap, so they are worth trying outright rather than hoping
        a search over every height finds them.
        """
        if self.granularity is None:
            return []
        block = max(1, int(self.granularity[0]))
        # A grain of one row is met by every height, so there is no shortlist to try: a store banded
        # along its leading axis (a memmap) says its grain on the axes BELOW, and enumerating every
        # height here would hand the search the whole range one at a time.
        if block <= 1:
            return []
        return list(range(block, int(cap) + 1, block))

    def rows_within(self, depth: int | None, budget_bytes: float, cap: int) -> int:
        """The tallest region up to ``cap`` rows whose priced block holds inside ``budget_bytes``,
        ``1`` when none does: the one search both ceilings (the rank's budget, the device's free
        memory) are answered by."""
        depth = sweep_module._sweep_pipeline_depth() if depth is None else depth
        low, high = 1, max(1, int(cap))
        while low < high:
            middle = (low + high + 1) // 2
            if self.sweep_block_bytes(self.sweep_shape(middle), depth) <= budget_bytes:
                low = middle
            else:
                high = middle - 1
        return low

    def best_tile(self, depth: int, budget_bytes: float, candidates: Sequence[int]) -> list[int]:
        """The affordable candidate whose decomposition reads the least, the first one otherwise.

        The search bisects on the height, which asks the price to rise with it. It does not: a
        stored block is decoded whole, so the price steps rather than climbs, and the shape rule
        may answer a cube at one height and a slab at the next. Bisection lands somewhere
        affordable, not on the best region the budget buys.

        Judged on reads and not on landed voxels, because that is what the sweep spends: a region
        that lands a few more rows by straddling the store's grid reads both blocks it touches, for
        every region of the case. Ties go to the taller block, which pays the per-region costs
        fewer times.
        """
        best: list[int] | None = None
        best_reads = 0
        for rows in candidates:
            tile = self.sweep_shape(rows)
            if self.sweep_block_bytes(tile, depth) > budget_bytes:
                continue
            reads = self.decomposition_reads(tile)
            taller = best is not None and np.prod(tile, dtype=np.int64) > np.prod(best, dtype=np.int64)
            if best is None or reads < best_reads or (reads == best_reads and taller):
                best, best_reads = tile, reads
        return best if best is not None else self.sweep_shape(candidates[0])

    def sweep_rows(self, depth: int | None = None) -> int:
        """The tallest region the sweep will cut whatever the budget: ``budget.SWEEP_SLAB_ROWS`` on
        a CPU, taller on a GPU as its free memory allows. What the budget then affords is
        :meth:`sweep_tile`'s.

        The device's share is held to the SAME price as everything else (:meth:`sweep_block_bytes`),
        which counts the source a region pulls and what the widest stage allocates on top of it.
        """
        cap = max(1, int(budget.SWEEP_SLAB_ROWS))
        # Never below the store's own block: a region shorter than one reads it whole regardless
        # (the hull is what a chunked read decodes), so cutting under it buys no memory back and
        # only reads the same bytes again for the next region.
        if self.granularity is not None:
            cap = max(cap, int(self.granularity[0]))
        if self.device is not None and self.device.type == "cuda":
            # On a GPU the transfers and launches per region are the cost: taller regions, as far as
            # a quarter of the free device memory allows (measured +10-20 % at 500^3 over 64 rows).
            free_bytes, _total = torch.cuda.mem_get_info(self.device)
            cap = max(cap, self.rows_within(depth, free_bytes * 0.25, _SWEEP_SLAB_ROWS_DEVICE))
        return cap

    def tile_within(self, depth: int, budget_bytes: float | None) -> tuple[list[int], int]:
        """The best block a sweep of ``depth`` can afford, and what it holds: the search alone.

        No refusal and no fallback, because two callers ask it two different questions -- whether a
        deeper queue still buys the same block (:meth:`keeps_the_block`) and what to do when none
        of them fits (:meth:`sweep_tile`) -- and a search that answered either for them would
        answer the other one wrong.
        """
        cap = self.sweep_rows(depth)
        if not budget_bytes or budget_bytes <= 0:
            return self.sweep_shape(cap), 0
        # The bisection never takes one row as affordable: the caller answers for it. What it finds
        # is then judged against the store's own heights, because the price steps rather than climbs
        # and bisection lands somewhere affordable, not on the best region the budget buys.
        low = self.rows_within(depth, budget_bytes, cap)
        tile = self.best_tile(depth, budget_bytes, [low, *self.grid_rows(cap)])
        return tile, self.sweep_block_bytes(tile, depth)

    def keeps_the_block(self, tile: list[int], depth: int) -> bool:
        """Whether a queue of ``depth`` both affords ``tile`` and still picks it.

        Asked of the search and not of :meth:`sweep_tile`, which falls back to no queue at all: a
        depth that cannot hold the block would come back holding it, and every depth would look
        affordable.
        """
        budget_bytes = self.budget_bytes
        found, held = self.tile_within(depth, budget_bytes)
        return found == tile and (not budget_bytes or budget_bytes <= 0 or held <= budget_bytes)

    def sweep_depth(self, tile: list[int]) -> int:
        """How many blocks to keep in flight, raised only while that changes nothing but the clock.

        A deeper queue absorbs the jitter between stages of uneven cost, and it is paid in resident
        blocks, which the sizing takes out of the block. Raised only while the block it allows is
        still ``tile``: a smaller block is a different decomposition, which re-chunks the output
        (the tile IS the store's chunk shape) and, on a map that does not factorise, moves the
        written values. Where the block is bounded by something other than the budget, the extra
        blocks are free, and the cap is what bounds them: on a 513x1331x1776 sweep in 40 blocks, a
        second block in flight recovers 0.5 s of a 6.7 s run and a third recovers none.
        """
        depth = sweep_module._sweep_pipeline_depth()
        # DOWN BEFORE UP. `tile` may be the one the sizing found only after giving the queue up
        # (:meth:`sweep_tile`), and a run that kept the queue anyway would hold what the sizing was
        # never told about -- the budget's whole promise, lost to a default nobody revisited.
        while depth and not self.keeps_the_block(tile, depth):
            depth -= 1
        while depth and depth < budget._SWEEP_MAX_DEPTH and self.keeps_the_block(tile, depth + 1):
            depth += 1
        return depth

    def sweep_tile(self, depth: int | None = None) -> list[int]:
        """The block one sweep region covers: the tallest the cap allows that still holds inside the
        budget, in the shape that pulls the least (:meth:`sweep_shape`).

        The budget is what a sweep may HOLD, so it is the priced block (:meth:`sweep_block_bytes`)
        that is held to it, never the landed rows alone: a REGRID pulling eight source voxels per
        landed one, or a stage declaring eight volumes-worth of buffers, costs what it costs. The
        search is over the height, because that is the one free parameter of the decomposition.
        """
        depth = sweep_module._sweep_pipeline_depth() if depth is None else depth
        budget_bytes = self.budget_bytes
        tile, held = self.tile_within(depth, budget_bytes)
        if not budget_bytes or budget_bytes <= 0 or held <= budget_bytes:
            return tile
        # THE READ-AHEAD IS THE ONE PART OF THE PRICE THE SIZING CHOSE. Everything else in the block
        # is what the chain must hold to run at all; the queue is bought, and what it buys is wall
        # clock (sweep_depth: half a second of a 6.7 s run). A sweep about to refuse has no clock to
        # buy, so it gives the queue up and asks once more. Three source regions resident become one,
        # which is a quarter to a third of the block on a chain whose stage buffers dominate -- a
        # narrow band, and inside it the difference is running against not running.
        serial = None
        if depth > 0:
            candidate, serial = self.tile_within(0, budget_bytes)
            if serial <= budget_bytes:
                return candidate
        raise DatasetManagerError(
            f"'{self.case}': no region of '{self.group}' fits the per-rank memory budget"
            f" ({format_bytes(budget_bytes)}): the smallest one this chain can sweep holds"
            f" {format_bytes(held)}"
            + (f", and {format_bytes(serial)} with the read-ahead given up" if serial is not None else "")
            + ".",
            "Raise 'memory_budget'.",
        )
