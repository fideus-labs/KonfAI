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


"""Reassembly of a case from its patches: whole, or slab by slab as they complete."""

from collections.abc import Callable
from itertools import pairwise
from typing import cast

import torch

from konfai.data.patching.blend import PathCombine
from konfai.utils.errors import PatchError


class Accumulator:
    """Accumulate patch predictions and reassemble them into a full tensor."""

    def __init__(
        self,
        patch_slices: list[tuple[slice, ...]],
        patch_size: list[int],
        patch_combine: PathCombine | None = None,
        batch: bool = True,
        sweep_axis: int = 0,
    ) -> None:
        # Which spatial axis the window slides along. The patch grid is emitted with this axis
        # outermost, so a patch's arrival finalizes everything behind it on that axis and nothing
        # else. 0 is what get_patch_slices_from_shape produces today.
        self.sweep_axis = sweep_axis
        self.patch_slices: list[tuple[slice, ...]] = []
        self.shape = max([[v.stop for v in patch] for patch in patch_slices])

        if patch_size is not None and not all(p == 0 for p in patch_size):
            # The last patch of an axis is padded up to the patch size for the model, then cropped; a
            # free axis (0) spans the full extent, so concretise it here or ``s.start + 0`` would
            # collapse the axis to a zero-width slice.
            concrete = [size if size > 0 else self.shape[dim] for dim, size in enumerate(patch_size)]
            for patch in patch_slices:
                slices = [slice(s.start, s.start + concrete[dim]) for dim, s in enumerate(patch)]
                self.patch_slices.append(tuple(slices))
        else:
            self.patch_slices = patch_slices
        self.patch_size = patch_size
        self.patch_combine = patch_combine
        self.batch = batch
        self._n = 2 if batch else 1
        self._count = len(patch_slices)
        self._filled = 0
        self._done = [False] * self._count
        # Patches are blended into this running buffer as they arrive (see add_layer), instead of
        # being kept until assembly: holding every patch of a large multi-class case (e.g. ~70 patches
        # of a 122-channel whole-body segmentation ≈ tens of GB) was the dominant reassembly RAM cost.
        self._result: torch.Tensor | None = None
        self._weighted: torch.Tensor | None = None
        # Blend-weight geometry: fixed for the accumulator's life, so it survives _reset. The grid,
        # the shares and the kept spans are all per (axis, start): sum(n_d) entries for a grid of
        # prod(n_d) patches, and one dict lookup per patch and axis to find them.
        self._geometry: tuple[list[list[torch.Tensor]], list[torch.Tensor]] | None = None
        self._grid: list[dict[int, int]] | None = None
        self._shares: dict[tuple[int, int, int, torch.dtype, torch.device], torch.Tensor | None] = {}
        self._kept: dict[tuple[int, int], slice] = {}

    def add_layer(self, index: int, layer: torch.Tensor) -> list[tuple[slice, torch.Tensor]]:
        """Blend one patch in; returns the slabs this completes (none for the whole-volume base)."""
        # Blend each patch straight into the running accumulator and drop the patch, rather than
        # storing all patches for a single assemble() at the end. The overlap blend is a weighted sum,
        # so accumulating incrementally is equivalent; re-adding an index is a no-op (last-write wins is
        # not possible once blended, and the prediction pipeline adds each patch exactly once).
        if self._done[index]:
            return []
        if self._result is None:
            # Allocate to the ACTUAL volume extent (self.shape), not the patch-size-extended grid. The
            # last patch of each axis is padded up to patch_size for the model, but that padded tail lies
            # OUTSIDE the volume; blending it would over-allocate the accumulator by up to
            # (patch_size - overlap) per axis (then get cropped away). We crop each patch to its in-volume
            # part at blend time instead, so nothing outside the volume is ever allocated.
            n = self._n
            self._result = torch.zeros(list(layer.shape[:n]) + list(self.shape), dtype=layer.dtype, device=layer.device)
        self._blend(layer, self.patch_slices[index])
        self._done[index] = True
        self._filled += 1
        return []

    def _blend(self, layer: torch.Tensor, patch_slice: tuple[slice, ...], row_offset: int = 0) -> None:
        """Blend one patch at its in-volume destination, ``row_offset`` rows down on the first spatial
        axis (the streaming window origin: 0 for the whole-volume base)."""
        n = self._n
        data = layer
        for dim, s in enumerate(patch_slice):
            if s.stop - s.start == 1:
                data = data.unsqueeze(dim=dim + n)
        # Clamp each spatial destination to the volume and crop the patch to it, BEFORE weighting: the
        # padded tail of a border patch lies outside the volume, so it has no share of the blend weight
        # to compute and every index in _weighted_patch stays in range.
        dest = [slice(s.start, min(s.stop, self.shape[dim])) for dim, s in enumerate(patch_slice)]
        data = data[tuple([slice(None)] * n + [slice(0, d.stop - d.start) for d in dest])]
        sweep = self.sweep_axis
        dest[sweep] = slice(dest[sweep].start - row_offset, dest[sweep].stop - row_offset)
        slices_dest = tuple([slice(cast(torch.Tensor, self._result).shape[i]) for i in range(n)] + dest)
        result = cast(torch.Tensor, self._result)
        lead = slices_dest[:n]
        if self.patch_combine is None:
            result[slices_dest] = data
        elif self.patch_combine.selects:
            # The kept regions partition the volume: nothing to weight, nothing to sum. Writing the box
            # this patch owns IS the operation, and it is what carries a discrete output through,
            # where a weighting would invent values between its classes.
            box = self._kept_box(patch_slice)
            kept = [slice(d.start + b.start, d.start + b.stop) for d, b in zip(dest, box, strict=True)]
            result[(*lead, *kept)] = data[(*lead, *box)]
        else:
            result[slices_dest] += self._weighted_patch(data, patch_slice)

    def _kept_box(self, patch_slice: tuple[slice, ...]) -> tuple[slice, ...]:
        """The sub-box a selection keeps of this patch: the run of ones in its window, per axis.

        Pure geometry, so it is read once from the host-side windows and cached per grid position
        along each axis. Deriving it from the share instead would read a device tensor: a host sync
        on every patch, which costs more than the weighting it replaces.
        """
        return tuple(self._kept_span(dim, s.start) for dim, s in enumerate(patch_slice))

    def _kept_span(self, dim: int, start: int) -> slice:
        key = (dim, start)
        if key not in self._kept:
            windows, _ = self._weight_geometry()
            kept = windows[dim][self._position(dim, start)].nonzero().flatten()
            self._kept[key] = slice(int(kept[0]), int(kept[-1]) + 1)
        return self._kept[key]

    def _weight_geometry(self) -> tuple[list[list[torch.Tensor]], list[torch.Tensor]]:
        """Per axis: the blend window, and its sum over the patch grid.

        The outer product of those sums is the total weight covering each voxel. It factorises because
        the patch grid is a full per-axis product and the window is itself separable
        (``sum_p prod_d w_d == prod_d sum_k w_d``), so the total is one vector per axis and never exists
        as a volume, on a 320-row window of a 1331x1775 volume, 13 KB instead of 2.8 GB.

        The patch extent comes from the slices, not from the window: a free axis carries a single
        broadcast entry (the ModelPatch blend-window contract) that must cover the whole extent.

        The window is asked for per grid position, because a selection opens its border patches to the
        volume edge (see Trim); a weighting returns the same window everywhere.
        """
        if self._geometry is None:
            combine = cast(PathCombine, self.patch_combine)
            windows, totals = [], []
            for dim, extent in enumerate(self.shape):
                starts = list(self._positions()[dim])
                length = self.patch_slices[0][dim].stop - self.patch_slices[0][dim].start
                per_position, total = [], torch.zeros(extent)
                for position, start in enumerate(starts):
                    window = combine.window(dim, position, len(starts))
                    window = window.expand(length) if window.numel() == 1 else window
                    stop = min(start + length, extent)
                    total[start:stop] += window[: stop - start]
                    per_position.append(window)
                windows.append(per_position)
                totals.append(total)
            self._geometry = (windows, totals)
        return self._geometry

    def _positions(self) -> list[dict[int, int]]:
        """Per axis, the patch grid's starts in order, each mapped to its position along the axis.

        One pass over the slices for the accumulator's life. Rebuilding the sorted starts per patch
        made every lookup O(P): 0.07 ms per patch at P = 1331 and 15.6 s over a case of 18,000
        thin 2.5D patches, against 16 ms for that case here.
        """
        if self._grid is None:
            self._grid = [
                {
                    start: position
                    for position, start in enumerate(sorted({patch[dim].start for patch in self.patch_slices}))
                }
                for dim in range(len(self.shape))
            ]
        return self._grid

    def _position(self, dim: int, start: int) -> int:
        return self._positions()[dim][start]

    def _share(self, dim: int, start: int, data: torch.Tensor) -> torch.Tensor | None:
        """This patch's fraction of the blend weight along one axis, ``w / sum_k w``, cached per axis.

        ``None`` where that fraction is exactly one on every voxel of the axis, which is what an axis
        holding a single grid position gives (the patch is the whole of the weight there, so the total
        IS its own window): the blend then skips a full pass over the patch, bit for bit the same
        values. A patch grid that tiles one axis and spans the other two is that case twice over.
        """
        extent = data.shape[self._n + dim]
        key = (dim, start, extent, data.dtype, data.device)
        if key not in self._shares:
            windows, totals = self._weight_geometry()
            window = windows[dim][self._position(dim, start)]
            share = window[:extent] / totals[dim][start : start + extent]
            share = share.to(device=data.device, dtype=data.dtype)
            self._shares[key] = None if torch.equal(share, torch.ones_like(share)) else share
        return self._shares[key]

    def _weighted_patch(self, data: torch.Tensor, patch_slice: tuple[slice, ...]) -> torch.Tensor:
        """``data`` scaled by its SHARE of the blend weight at each voxel, one axis at a time.

        Normalising per patch rather than dividing the assembled volume by an accumulated weight drops
        both the spatial-sized weight buffer and the final division pass over every channel. The shares
        sum to one per voxel by construction (the total is the sum over the same grid), so the blend
        stays exact, and each factor is a ratio of comparable quantities, so it lives in [0, 1] where
        the raw product underflows fp16 and needed a floor.

        Only the axes whose share is not identically one are applied (see ``_share``), each one pass
        over the patch: a grid tiled along one axis and spanning the other two hands the patch back
        untouched. Into a staging buffer the patches share otherwise, one patch-sized allocation per
        accumulator instead of per blend, and out of place: the caller's tensor is never touched, so
        the OOM retry (which re-blends the same patch on the CPU) never re-weights it.
        """
        shares = []
        for dim, s in enumerate(patch_slice):
            share = self._share(dim, s.start, data)
            if share is not None:
                view = [1] * data.ndim
                view[self._n + dim] = -1
                shares.append(share.view(view))
        if not shares:
            return data
        if (
            self._weighted is None
            or self._weighted.shape != data.shape
            or self._weighted.dtype != data.dtype
            or self._weighted.device != data.device
        ):
            self._weighted = torch.empty_like(data)
        torch.mul(data, shares[0], out=self._weighted)
        for share in shares[1:]:
            self._weighted.mul_(share)
        return self._weighted

    def _along_sweep(self, span: slice, lead: int) -> tuple[slice, ...]:
        """Index a result tensor over ``span`` on the sweep axis and whole on every other."""
        spatial = [span if dim == self.sweep_axis else slice(None) for dim in range(len(self.shape))]
        return tuple([slice(None)] * lead + spatial)

    def _sweep_shape(self, extent: int) -> list[int]:
        """The volume's spatial shape with the sweep axis cut down to ``extent``."""
        return [extent if dim == self.sweep_axis else size for dim, size in enumerate(self.shape)]

    def is_empty(self) -> bool:
        """True until the first patch has been blended in (no volume-sized buffer allocated yet)."""
        return self._result is None

    @property
    def footprint_shape(self) -> list[int]:
        """Spatial shape held in memory at once: the whole volume for the base accumulator (overridden
        by StreamingAccumulator, which keeps only a window). Used to size the blend device."""
        return self.shape

    def is_full(self) -> bool:
        # O(1): a running counter avoids re-scanning every slot after each added patch
        # (re-scanning per patch would be O(P^2) per case).
        return self._filled == self._count

    def assemble(self) -> torch.Tensor:
        if self._result is None:
            raise PatchError(
                "Accumulator.assemble() was called before any patch was added.",
                f"Expected up to {self._count} patch(es) via add_layer() before assembling.",
                "Add at least one patch (and check is_full()) before calling assemble().",
            )
        result = self._result
        # Nothing to normalise: each patch was blended in with its share of the weight, so the shares
        # already sum to one per voxel. No final crop either: patches are cropped at blend time.
        self._reset()
        return result

    def _reset(self) -> None:
        self._result = None
        self._weighted = None
        self._filled = 0
        self._done = [False] * self._count


class StreamingAccumulator(Accumulator):
    """Accumulator holding only the active window along the first spatial axis; it yields each finalized
    slab as its patches complete, so peak memory is two patch extents (the window and its slide room).

    The patch-grid order (``get_patch_slices_from_shape`` iterates ``itertools.product`` with the first
    spatial axis outermost) has patch starts along that axis never decreasing, so when a patch starting
    at ``z`` arrives, every voxel before ``z`` has already received all of its patches and the region up
    to ``z`` is final. ``add_layer`` returns those finalized slabs (``assemble()``'s values, from the
    same blend and weight arithmetic applied slab by slab), and ``finalize()`` flushes the tail.
    """

    def __init__(
        self,
        patch_slices: list[tuple[slice, ...]],
        patch_size: list[int],
        patch_combine: PathCombine | None = None,
        batch: bool = True,
        sweep_axis: int = 0,
    ) -> None:
        super().__init__(patch_slices, patch_size, patch_combine, batch, sweep_axis)
        sweep = self.sweep_axis
        self._window = min(
            max(patch[sweep].stop - patch[sweep].start for patch in self.patch_slices), self.shape[sweep]
        )
        starts = [patch[sweep].start for patch in self.patch_slices]
        if (starts and starts[0] != 0) or any(
            0 > current - previous or current - previous > self._window for previous, current in pairwise(starts)
        ):
            raise PatchError(
                "StreamingAccumulator requires patches starting at row 0 and ordered by non-decreasing start on"
                " the first spatial axis, advancing by at most one patch extent per step.",
                "get_patch_slices_from_shape generates this order; a custom patch source must preserve it.",
            )
        self._flushed = 0

    def add_layer(self, index: int, layer: torch.Tensor) -> list[tuple[slice, torch.Tensor]]:
        if self._done[index]:
            return []
        n = self._n
        patch_slice = self.patch_slices[index]
        # Correctness rests on patches ARRIVING in non-decreasing axis-0-start order, not just on the
        # slice list being sorted: a patch whose start is already flushed would write at a negative
        # window offset (dest[0] below), which torch silently reads from the end -> misplaced data, no
        # error. Fail loud instead. The prediction loop preserves per-case order (shuffle=False), so
        # this only guards a future misuse (e.g. a shuffling sampler).
        if patch_slice[self.sweep_axis].start < self._flushed:
            raise PatchError(
                f"StreamingAccumulator received patch start {patch_slice[self.sweep_axis].start} after flushing to "
                f"{self._flushed}: patches must arrive in non-decreasing first-spatial-axis order.",
                "Add patches in the order get_patch_slices_from_shape generates them (no shuffling).",
            )
        slabs = self._advance_to(patch_slice[self.sweep_axis].start)
        if self._result is None:
            self._result = torch.zeros(
                [*layer.shape[:n], *self._sweep_shape(self._window)],
                dtype=layer.dtype,
                device=layer.device,
            )
        self._blend(layer, patch_slice, row_offset=self._flushed)
        self._done[index] = True
        self._filled += 1
        return slabs

    def finalize(self) -> list[tuple[slice, torch.Tensor]]:
        """Flush the remaining window and reset for reuse; call once ``is_full()``."""
        slabs = self._advance_to(self.shape[self.sweep_axis])
        self._reset()
        self._flushed = 0
        return slabs

    @property
    def footprint_shape(self) -> list[int]:
        # Only the window is resident, so the blend-device budget is the window's: a huge volume streams
        # on the GPU within bounded VRAM. Blend and IEEE-correctly-rounded finalize ops (+, *, /, argmax,
        # cast) are bit-identical CPU/CUDA; only a transcendental-terminated float output (Softmax/Sigmoid)
        # can differ by ~1 ULP between a window on the GPU and a whole-volume reference on the CPU.
        return self._sweep_shape(self._window)

    def assemble(self) -> torch.Tensor:
        raise PatchError(
            "StreamingAccumulator does not assemble a whole volume.",
            "Consume the slabs returned by add_layer() and finalize() instead.",
        )

    def _advance_to(self, z: int) -> list[tuple[slice, torch.Tensor]]:
        """Finalize the window up to ``z`` (absolute) and shift the window origin there."""
        z = min(z, self.shape[self.sweep_axis])
        if self._result is None or z <= self._flushed:
            return []
        n = self._n
        length = z - self._flushed
        if length > self._window:
            raise PatchError(
                f"StreamingAccumulator asked to finalize {length} rows at once with a {self._window}-row window.",
                "Patch starts may advance by at most one patch extent per step (checked at construction).",
            )
        # Cloned: the window slides over these rows right after, so the slab handed out must not be a
        # view of it. Nothing else to do: the blend weights already sum to one over these voxels.
        slab = self._result[self._along_sweep(slice(0, length), n)].clone()
        keep = self._window - length
        # .clone(): source and destination views overlap when length < window.
        self._result[self._along_sweep(slice(0, keep), n)] = self._result[
            self._along_sweep(slice(length, self._window), n)
        ].clone()
        self._result[self._along_sweep(slice(keep, self._window), n)] = 0
        region = slice(self._flushed, z)
        self._flushed = z
        return [(region, slab)]


class SlabRegionStream:
    """Slab in → slab out through one region stage, with bounded lookahead.

    The write mirror of the read dispatcher's single-region rule: finalized slabs arrive in order along
    the first spatial axis (the :class:`StreamingAccumulator`'s order), and each output region is
    emitted as soon as the input region it pulls has arrived, so only a sliding window of the input is
    ever resident. The stage itself is two injected callables, both pure region arithmetic + tensor
    work, so no stage kind has streaming code of its own:

    - ``pull(target_slices) -> source_slices``: the clamped input region an output region is computed
      from (a transform's ``stream_region_target``/``stream_region_source``, or a halo enlargement).
    - ``produce(window, target_slices, source_slices) -> tensor``: the output block for
      ``target_slices``, given exactly the pulled window.

    The schedule is derived from ``pull`` alone: a probe finds which output axis the input slab axis
    feeds and in which direction (a mirrored or permuted axis streams too, through the sink's
    random-access region writes), and emission advances along that axis as far as the arrived input
    allows. Any per-axis monotone map works; nothing here names a stage.
    """

    def __init__(
        self,
        pull: Callable[[tuple[slice, ...]], list[slice]],
        produce: Callable[[torch.Tensor, tuple[slice, ...], list[slice]], torch.Tensor],
        in_spatial_shape: list[int],
        out_spatial_shape: list[int],
    ) -> None:
        self._pull = pull
        self._produce = produce
        self._in_shape = [int(s) for s in in_spatial_shape]
        self._out_shape = [int(s) for s in out_spatial_shape]
        self._axis, self._ascending = self._probe_axis()
        self._slabs: list[tuple[int, torch.Tensor]] = []
        self._complete_to = 0
        self._emitted = 0

    def _probe_axis(self) -> tuple[int, bool]:
        """Which output axis the input slab axis feeds, and whether in ascending order.

        Probing one output row per axis against the full-region pull identifies the axis whose region
        controls input axis 0; comparing the first and last rows' pulls gives the direction. A wrong
        pick can never corrupt the output (emission is gated on the pull of the real regions); it only
        buffers more, so a map no probe can tell apart falls back to axis 0 ascending.
        """
        full = tuple(slice(0, n) for n in self._out_shape)
        baseline = self._pull(full)[0]
        for axis, extent in enumerate(self._out_shape):
            probe = list(full)
            probe[axis] = slice(0, 1)
            first = self._pull(tuple(probe))[0]
            if (first.start, first.stop) == (baseline.start, baseline.stop):
                continue
            probe[axis] = slice(extent - 1, extent)
            last = self._pull(tuple(probe))[0]
            return axis, first.start <= last.start
        return 0, True

    def _prefix_slices(self, m: int) -> tuple[slice, ...]:
        """The first ``m`` output rows in iteration order, as absolute region slices."""
        extent = self._out_shape[self._axis]
        rows = slice(0, m) if self._ascending else slice(extent - m, extent)
        slices = [slice(0, n) for n in self._out_shape]
        slices[self._axis] = rows
        return tuple(slices)

    def push(self, region: slice, slab: torch.Tensor) -> list[tuple[tuple[slice, ...], torch.Tensor]]:
        """Take the next finalized slab (rows ``region`` of the input) and emit what it completes."""
        if region.start != self._complete_to:
            raise PatchError(
                f"SlabRegionStream received rows [{region.start}, {region.stop}) after "
                f"[0, {self._complete_to}): slabs must arrive contiguously from the first row.",
                "Push the slabs exactly as the StreamingAccumulator yields them.",
            )
        self._slabs.append((region.start, slab))
        self._complete_to = region.stop
        return self._emit()

    def finalize(self) -> list[tuple[tuple[slice, ...], torch.Tensor]]:
        """Emit whatever the completed input still allows and verify nothing is left behind."""
        emitted = self._emit()
        if self._emitted != self._out_shape[self._axis]:
            raise PatchError(
                f"SlabRegionStream finalized with {self._emitted} of {self._out_shape[self._axis]} "
                f"output rows emitted (input complete to {self._complete_to} of {self._in_shape[0]}).",
                "Push every slab of the input before finalizing.",
            )
        self._slabs.clear()
        return emitted

    def _emit(self) -> list[tuple[tuple[slice, ...], torch.Tensor]]:
        extent = self._out_shape[self._axis]
        # The pull of an iteration prefix grows monotonically with it, so the furthest emittable row is
        # a binary search. O(log rows) pull calls per push, and a pull may be more than slice
        # arithmetic (a declaration may copy the attribute it reads).
        low, high = self._emitted, extent
        while low < high:
            middle = (low + high + 1) // 2
            if self._pull(self._prefix_slices(middle))[0].stop <= self._complete_to:
                low = middle
            else:
                high = middle - 1
        m = low
        if m == self._emitted:
            return []
        rows = slice(self._emitted, m) if self._ascending else slice(extent - m, extent - self._emitted)
        target = list(self._prefix_slices(m))
        target[self._axis] = rows
        source = self._pull(tuple(target))
        window = self._window(source)
        result = self._produce(window, tuple(target), source)
        self._emitted = m
        if m < extent:
            remaining = [slice(0, n) for n in self._out_shape]
            remaining[self._axis] = slice(m, extent) if self._ascending else slice(0, extent - m)
            keep_from = self._pull(tuple(remaining))[0].start
            self._slabs = [
                (start, slab) for start, slab in self._slabs if start + slab.shape[-len(self._in_shape)] > keep_from
            ]
        else:
            self._slabs.clear()
        return [(tuple(target), result)]

    def _window(self, source: list[slice]) -> torch.Tensor:
        """The buffered input restricted to ``source``: rows gathered from the arrived slabs, the
        other axes sliced in place."""
        axis0 = source[0]
        n_lead = self._slabs[0][1].dim() - len(self._in_shape)
        lead = (slice(None),) * n_lead
        pieces = []
        for start, slab in self._slabs:
            stop = start + slab.shape[n_lead]
            lo, hi = max(start, axis0.start), min(stop, axis0.stop)
            if lo < hi:
                pieces.append(slab[(*lead, slice(lo - start, hi - start))])
        window = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=n_lead)
        if window.shape[n_lead] != axis0.stop - axis0.start:
            raise PatchError(
                f"SlabRegionStream window rows [{axis0.start}, {axis0.stop}) are not fully buffered.",
                "The pull map must never reach below rows already pruned or above rows arrived.",
            )
        return window[(*lead, slice(None), *source[1:])]


class SlabAligner:
    """Merge several slab streams over the same axis into jointly finalized intervals.

    The cross-stream mirror of :class:`StreamingAccumulator`: each stream (a TTA copy's accumulator)
    emits finalized slabs in non-decreasing order along the shared first spatial axis, and a consumer
    that needs every stream's rows together (a cross-copy reduction) can only advance to the
    slowest frontier. ``push`` takes one stream's new slabs and returns the intervals that just
    became complete, each carrying every stream's rows; only the inter-stream skew is ever buffered,
    and a single stream passes through untouched. Nothing here knows what a stream is: it is pure
    interval arithmetic over ``nb_streams`` ordered emitters.
    """

    def __init__(self, nb_streams: int, lead_dims: int = 1) -> None:
        self._nb_streams = nb_streams
        self._lead = lead_dims
        self._pending: dict[int, list[tuple[int, torch.Tensor]]] = {}
        self._frontiers: dict[int, int] = {}
        self._consumed = 0
        self._completed: set[int] = set()

    @property
    def complete(self) -> bool:
        """Whether every stream has pushed its last slab."""
        return len(self._completed) == self._nb_streams

    def push(
        self, stream: int, slabs: list[tuple[slice, torch.Tensor]], finished: bool = False
    ) -> list[tuple[slice, dict[int, torch.Tensor]]]:
        """Take ``stream``'s freshly finalized slabs and return the newly joint-final intervals."""
        pending = self._pending.setdefault(stream, [])
        for region, slab in slabs:
            pending.append((region.start, slab))
            self._frontiers[stream] = region.stop
        if finished:
            self._completed.add(stream)
        if len(self._frontiers) < self._nb_streams:
            return []
        joint = min(self._frontiers.values())
        if joint <= self._consumed:
            return []
        lead = (slice(None),) * self._lead
        rows: dict[int, torch.Tensor] = {}
        for key in sorted(self._pending):
            pieces = [
                slab[
                    (
                        *lead,
                        slice(max(start, self._consumed) - start, min(start + slab.shape[self._lead], joint) - start),
                    )
                ]
                for start, slab in self._pending[key]
                if start < joint and start + slab.shape[self._lead] > self._consumed
            ]
            if not pieces:
                raise PatchError(
                    f"SlabAligner stream {key} holds no rows for [{self._consumed}, {joint}).",
                    "Push the slabs exactly as the accumulators finalize them, without skipping a stream.",
                )
            rows[key] = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=self._lead)
            self._pending[key] = [
                (start, slab) for start, slab in self._pending[key] if start + slab.shape[self._lead] > joint
            ]
        interval = slice(self._consumed, joint)
        self._consumed = joint
        return [(interval, rows)]
