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

A model's VRAM footprint cannot be computed from headers -- it is its activations -- so it is
MEASURED, and measured for free: the real workflow run is the probe. The contract, shared by
prediction and training, is ``transient(step) + resident(patch) <= free_VRAM x margin``, where each
workflow declares its step (a forward; a forward+backward+optimizer step) and its resident set
(accumulators and the streamed assembly window; parameters, gradients and optimizer state). The
provisional grid starts at the worst case's full extent; when a step runs out of memory, the caller
catches it, asks :func:`next_patch_candidate` for one shrink step -- scaled by the last measured
transient when there is one, a fixed factor when the OOM left no number -- re-plans the grid and
restarts. When everything fits (the common case) nothing here runs at all.
"""

from collections.abc import Callable

import torch

#: Fraction of the free VRAM a step may claim; the reserve absorbs allocator fragmentation and
#: transients the measured run did not exercise (mirrors the accumulation gate's margin).
VRAM_BUDGET_SAFETY_FRACTION = 0.8

#: Per-axis shrink applied when an OOM left no usable measurement to scale from.
_OOM_SHRINK_STEP = 0.8


def measure_transient_bytes(run: Callable[[], None], device: torch.device | int) -> int | None:
    """Measure the transient VRAM one ``run()`` peaks above the resident set, or ``None`` on OOM.

    The caller provides the run (a forward for prediction, forward+backward for training) so this
    stays model-agnostic; an out-of-memory run is reported as ``None`` -- a valid "does not fit"
    measurement -- with the partial allocations released. ``max_memory_allocated`` is a running
    high-water mark, so a stale peak can only over-estimate the transient, never under.
    """
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    resident = torch.cuda.memory_allocated(device)
    try:
        run()
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None
    torch.cuda.synchronize(device)
    return int(torch.cuda.max_memory_allocated(device) - resident)


def usable_vram(free_bytes: float, resident_bytes: float = 0.0, margin: float = VRAM_BUDGET_SAFETY_FRACTION) -> float:
    """The VRAM a step's transient may claim: free memory under the safety margin, minus what must
    stay resident alongside the step (accumulators and the streamed assembly window for prediction;
    nothing extra for training, whose resident set is already allocated when ``free_bytes`` is read).
    """
    return free_bytes * margin - resident_bytes


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
    (``0`` = free, ``N`` = pinned, ``None`` = all free) -- only free axes move. With a measured
    transient the free axes scale ISOTROPICALLY by ``(usable / measured) ** (1 / n_free)`` (the
    volume ratio activations follow, ~linear in voxels: one step lands near the target); without one
    -- or when the measurement claims the candidate already fits, so scaling would not shrink -- each
    free axis takes the fixed OOM step. Sizes snap DOWN to the model's valid multiples, floored at
    ``min(snap, extent)``. ``None`` means no smaller candidate exists (every free axis at its floor,
    or ``usable_bytes`` leaves the step no memory at all) -- the caller owns the error message.
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
