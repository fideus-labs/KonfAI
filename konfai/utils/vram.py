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
catches it, asks :func:`next_patch_candidate` for one shrink step (scaled by the last measured
transient when there is one, a fixed factor when the OOM left no number) re-plans the grid and
restarts. When everything fits (the common case) nothing here runs at all.
"""

from typing import Any

import torch

from konfai.utils.utils import concretize_patch_size, size_free_axes

#: Fraction of the free VRAM a step may claim; the reserve absorbs allocator fragmentation and
#: transients the measured run did not exercise (mirrors the accumulation gate's margin).
VRAM_BUDGET_SAFETY_FRACTION = 0.8

#: Per-axis shrink applied when an OOM left no usable measurement to scale from.
_OOM_SHRINK_STEP = 0.8


def usable_vram(free_bytes: float, resident_bytes: float = 0.0, margin: float = VRAM_BUDGET_SAFETY_FRACTION) -> float:
    """The VRAM a step's transient may claim: free memory under the safety margin, minus what must
    stay resident alongside the step (accumulators and the streamed assembly window for prediction;
    nothing extra for training, whose resident set is already allocated when ``free_bytes`` is read).
    """
    return free_bytes * margin - resident_bytes


def transient_at_oom(device: int | None) -> int | None:
    """The failed step's transient (CUDA peak over resident), ``None`` off CUDA or when unreadable."""
    if device is None:
        return None
    try:
        transient = int(torch.cuda.max_memory_allocated(device) - torch.cuda.memory_allocated(device))
    except Exception:  # nosec B110 - an unreadable measurement falls back to the fixed shrink step
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


def next_patch_candidate(
    candidate: list[int],
    patch_size: list[int] | None,
    shape: list[int] | tuple[int, ...],
    measured_bytes: int | None,
    usable_bytes: float,
    snap: list[int] | None = None,
) -> list[int] | None:
    """One shrink step toward a patch whose step fits ``usable_bytes``; ``None`` = nothing smaller.

    ``candidate`` is the size that just failed; ``patch_size`` is the user's per-axis convention
    (``0`` = free, ``N`` = pinned, ``None`` = all free): only free axes move. With a measured
    transient the free axes scale ISOTROPICALLY by ``(usable / measured) ** (1 / n_free)`` (the
    volume ratio activations follow, ~linear in voxels: one step lands near the target); without one: or when
    the measurement claims the candidate already fits, so scaling would not shrink: each
    free axis takes the fixed OOM step. Sizes snap DOWN to the model's valid multiples, floored at
    ``min(snap, extent)``. ``None`` means no smaller candidate exists (every free axis at its floor,
    or ``usable_bytes`` leaves the step no memory at all): the caller owns the error message.
    """
    free = [d for d, p in enumerate(patch_size) if p == 0] if patch_size is not None else list(range(len(candidate)))
    if not free or usable_bytes <= 0:
        return None
    if measured_bytes is not None and measured_bytes > usable_bytes:
        ratio = (usable_bytes / measured_bytes) ** (1.0 / len(free))
    else:
        ratio = _OOM_SHRINK_STEP

    def snapped(axis: int, value: int) -> int:
        if snap is None or snap[axis] <= 1:
            return max(1, value)
        return max(min(snap[axis], int(shape[axis])), (value // snap[axis]) * snap[axis])

    shrunk = list(candidate)
    for axis in free:
        shrunk[axis] = min(snapped(axis, int(candidate[axis] * ratio)), candidate[axis])
    return shrunk if shrunk != list(candidate) else None


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
            candidate, self._vram_patch_template, worst, measured, usable, self._downsampling_factor
        )
