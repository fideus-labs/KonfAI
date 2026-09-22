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

"""VRAM-driven patch sizing: measure on the real run, shrink one step on OOM, restart.

A model's VRAM footprint cannot be computed from headers (it is its activations), so it is
MEASURED, and measured for free: the real workflow run is the probe. The contract, shared by
prediction and training, is ``transient(step) + resident(patch) <= free_VRAM x margin``, where each
workflow declares its step (a forward; a forward+backward+optimizer step) and its resident set
(accumulators and the streamed assembly window; parameters, gradients and optimizer state). The
provisional grid starts at the worst case's full extent; when a step runs out of memory, the caller
catches it, asks :func:`next_patch_candidate` for one shrink step (more patches along the free axes, as
many as the last measured transient says, one more when the OOM left no number) re-plans the grid and
restarts. When everything fits (the common case) nothing here runs at all.

A prediction's batch is measured the same way (``batch_size: 0``): its first forward runs one patch,
its second two, and :func:`measured_batch` extrapolates the batch the rest run at.
"""

import math
from bisect import bisect_left
from typing import Any

import torch

from konfai.utils.errors import ConfigError
from konfai.utils.utils import (
    OverlapSpec,
    concretize_patch_size,
    free_axis_rounding,
    resolve_overlap,
    size_free_axes,
)

#: Fraction of the free VRAM a step may claim; the reserve absorbs allocator fragmentation and
#: transients the measured run did not exercise (mirrors the accumulation gate's margin).
VRAM_BUDGET_SAFETY_FRACTION = 0.8


def usable_vram(free_bytes: float, resident_bytes: float = 0.0, margin: float = VRAM_BUDGET_SAFETY_FRACTION) -> float:
    """The VRAM a step's transient may claim: free memory under the safety margin, minus what must
    stay resident alongside the step (accumulators and the streamed assembly window for prediction;
    nothing extra for training, whose resident set is already allocated when ``free_bytes`` is read).
    """
    return free_bytes * margin - resident_bytes


#: The share of the usable VRAM a measured batch's forward may claim. The rest stays free for the convolution
#: workspace and the allocator: at the edge of memory a batch runs slower per patch (ImpactSynth, 64 slices
#: against 32: 5 % slower), and past the half the throughput has nothing left to gain.
BATCH_SHARE = 0.5


def power_of_two_floor(value: int) -> int:
    """The largest power of two not above ``value`` (1 for anything under 2)."""
    return 1 << (max(1, value).bit_length() - 1)


def measured_batch(spent_one: int, spent_two: int, usable: float) -> int:
    """The largest power of two whose forward fits ``BATCH_SHARE`` of ``usable``, from what the forwards of
    one patch and of two claimed: what the second patch added is what each patch costs, the rest of the
    first what any batch costs."""
    per_patch = max(spent_two - spent_one, 1)
    fits = int((usable * BATCH_SHARE - max(spent_one - per_patch, 0)) // per_patch)
    return power_of_two_floor(fits)


def transient_at_oom(device: int | None) -> int | None:
    """The failed step's transient (CUDA peak over resident), ``None`` off CUDA or when unreadable."""
    if device is None:
        return None
    try:
        transient = int(torch.cuda.max_memory_allocated(device) - torch.cuda.memory_allocated(device))
    except Exception:  # nosec B110 - an unreadable measurement falls back to one split per restart
        return None
    return transient if transient > 0 else None


def reset_peak(device: int | None) -> None:
    """Drop the failed attempt's high-water mark, so the rerun measures its own steps: the mark
    only rises, and the full-extent attempt's would overstate every later transient."""
    if device is None:
        return
    try:
        torch.cuda.reset_peak_memory_stats(device)
    except Exception:  # nosec B110 - stale stats only cost precision, never correctness
        pass


def usable_after_oom(device: int | None) -> float:
    """The VRAM the next attempt's step may claim, read once the failed state is freed; ``0.0``
    (which refuses the restart) off CUDA or when unreadable."""
    if device is None:
        return 0.0
    try:
        torch.cuda.empty_cache()
        free, _ = torch.cuda.mem_get_info(device)
    except Exception:  # nosec B110
        return 0.0
    return usable_vram(free)


def balanced_patch(extent: int, count: int, overlap: OverlapSpec) -> int:
    """The shortest patch that covers ``extent`` in ``count`` patches overlapping as ``overlap`` says (20 % when
    ``None``): the axis cut into equal parts, the fewest voxels that count computes."""

    def covers(size: int) -> bool:
        try:
            voxels = resolve_overlap(overlap, [size], [extent])[0]
        except ConfigError:  # an overlap as long as the patch
            return False
        return size + (count - 1) * (size - voxels) >= extent

    return bisect_left(range(extent + 1), True, key=covers)


def _split(extent: int, size: int, overlap: OverlapSpec, multiple: int) -> int | None:
    """The patch an axis takes with one more patch along it: the balanced patch of the next count that is shorter
    than ``size``, rounded up to the model's ``multiple``; ``None`` at the floor."""
    if size <= multiple:
        return None
    for count in range(max(2, extent // size + 1), extent + 1):
        shorter = -(-balanced_patch(extent, count, overlap) // multiple) * multiple
        if shorter < size:
            return shorter
    return None


def next_patch_candidate(
    candidate: list[int],
    patch_size: list[int] | None,
    shape: list[int] | tuple[int, ...],
    measured_bytes: int | None,
    usable_bytes: float,
    snap: list[int] | None = None,
    overlap: OverlapSpec = None,
) -> list[int] | None:
    """One shrink step toward a patch whose step fits ``usable_bytes``; ``None`` = nothing smaller.

    ``candidate`` is the size that just failed; ``patch_size`` is the user's per-axis convention
    (``0`` = free, ``N`` = pinned, ``None`` = all free): only free axes move, and they move by patch
    COUNT. The free axis whose patch is the longest takes one more patch, and each patch of an axis is the
    shortest that covers it in that count with its ``overlap``: the fewest patches, the axis cut into
    equal parts (two patches of 295 voxels cover 531 overlapping by 59, where two of 392 overlap by 253).
    With a measured transient the axes split until the patch's voxels fit ``usable / measured`` of the
    failed one (activations are ~linear in voxels: one restart); without one, or when the measurement
    claims the candidate already fits, one split per restart. Sizes round up to the model's valid
    multiples (``snap``). ``None`` means no smaller candidate exists (every free axis at its floor, or
    ``usable_bytes`` leaves the step no memory at all): the caller owns the error message.
    """
    free = [d for d, p in enumerate(patch_size) if p == 0] if patch_size is not None else list(range(len(candidate)))
    if not free or usable_bytes <= 0:
        return None
    fits = None
    if measured_bytes is not None and measured_bytes > usable_bytes:
        fits = math.prod(candidate) * usable_bytes / measured_bytes
    sizes = list(candidate)
    while True:
        splits = {}
        for axis in free:
            axis_overlap = overlap[axis] if isinstance(overlap, list) else overlap
            multiple = free_axis_rounding(snap, axis, len(candidate))
            shorter = _split(int(shape[axis]), sizes[axis], axis_overlap, multiple)
            if shorter is not None:
                splits[axis] = shorter
        if not splits:
            break
        longest = max(splits, key=lambda axis: sizes[axis])
        sizes[longest] = splits[longest]
        if fits is None or math.prod(sizes) <= fits:
            break
    return sizes if sizes != list(candidate) else None


class VramAutoPatchMixin:
    """The auto-patch state and shrink policy the training and prediction workflows share.

    The state lives on the workflow object itself: the free-axis template captured from the user's
    patch (a per-axis ``0`` marks a FREE axis and opts into the OOM restart loop), the current
    candidate, and the model's per-axis input multiple. Each workflow keeps only its own injection
    points around this: the trainer its multi-rank shrink rendezvous, the predictor its
    accumulation reserve and output reset.
    """

    #: The workflow's dataset (set by the subclass __init__): the grids re-cut on a re-plan.
    dataset: Any

    def _capture_vram_patch_template(self, patch: Any) -> None:
        """Capture the user's free-axis convention before any re-plan materialises sizes over it."""
        self._vram_patch_template: list[int] | None = (
            [int(size) for size in patch.patch_size]
            if patch is not None and patch.patch_size is not None and any(size == 0 for size in patch.patch_size)
            else None
        )
        self._vram_patch_overlap: OverlapSpec = patch.overlap if patch is not None else None
        self._vram_patch_candidate: list[int] | None = None
        #: Per-axis input multiple the model needs (its downsampling factor); a free axis snaps to
        #: it. The subclass sets it once the model graph is final.
        self._downsampling_factor: list[int] | None = None

    def _presize_free_axes(self) -> bool:
        """Round the free patch axes up to the model's valid input multiple before the first step,
        so the network's encoder/decoder skips align instead of crashing on a non-divisible extent.
        Every rank rounds the same worst case to the same size, so no rendezvous is needed here
        (unlike the OOM shrink). True when the grids were re-cut: the caller re-fetches its loaders.
        """
        if self._vram_patch_template is None:
            # No free axis to size; also keeps non-image groups (fiducials, transforms) out of worst_case_shape.
            return False
        sized = size_free_axes(self._vram_patch_template, self.dataset.worst_case_shape(), self._downsampling_factor)
        if sized is None:
            return False
        self._adopt_patch_candidate(sized)
        return True

    def _adopt_patch_candidate(self, candidate: list[int]) -> None:
        """Record ``candidate`` and re-cut every prepared grid to it."""
        self._vram_patch_candidate = candidate
        self.dataset.replan_patch(candidate)

    def _shrunken_patch(self, measured: int | None, usable: float) -> list[int] | None:
        """One shrink step for the free patch axes after a CUDA OOM (``None`` = not auto, or floor).

        The first OOM starts from the worst prepared case at full extent (the size the failed grid
        effectively ran); later ones shrink the current candidate further.
        """
        if self._vram_patch_template is None:
            return None
        worst = self.dataset.worst_case_shape()
        if worst is None:
            return None
        candidate = self._vram_patch_candidate or concretize_patch_size(
            self._vram_patch_template, worst, self._downsampling_factor
        )
        return next_patch_candidate(
            candidate,
            self._vram_patch_template,
            worst,
            measured,
            usable,
            self._downsampling_factor,
            self._vram_patch_overlap,
        )
