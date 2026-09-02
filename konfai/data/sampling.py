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

"""Where each target voxel reads from, and the gather that reads it, in torch, on the volume's device.

Two halves, and only the first one varies. The COORDINATE PRODUCER turns a decoded transform into
one source index per target voxel; the GATHER is ITK's sampler and is written once. A stage that
resamples onto another grid, one that warps through a field and one that does both differ only in
what they hand the producer.

Nothing here calls SimpleITK. A BSpline is evaluated from its coefficient grid and a dense field
from its samples, with the same tensor-product kernel arithmetic ITK uses (verified to 8e-14
against ``TransformPoint``), so a chain that resamples through a stored transform stays on the GPU
from the read to the write.
"""

from __future__ import annotations

import contextlib
import contextvars
import itertools
import time
from collections.abc import Iterator

import numpy as np
import torch

from konfai.data.geometry import (
    SUPPORTED_SPLINE_ORDERS,
    AffineMap,
    AffineStage,
    DisplacementStage,
    Grid,
    SpatialStages,
    TransformBound,
)

#: Coordinates are accumulated in float64 and only the gather runs in the payload's dtype. A world
#: coordinate is an origin of hundreds of millimetres plus an index of hundreds of voxels, and in
#: float32 that sum keeps about four digits past the voxel: enough to move a sample across a
#: voxel boundary on a large grid, which is a visibly different value on anything that is not
#: smooth. Measured on a 64x512x512 slab: float32 coordinates disagree with SimpleITK by 0.35 on
#: data in [0, 1], float64 by 3e-05 (which is the gather's own float32 rounding).
_COORDINATE_DTYPE = torch.float64

#: The walk's dtype for the CURRENT context, entered by :func:`coordinate_precision`. A contextvar
#: and not a parameter threaded through eight signatures: the choice belongs to the stage that
#: knows its data (an intensity resample can trade the last bits for bandwidth, a label resample
#: must not), and every helper below reads the same answer without the plumbing.
_COORDINATE_DTYPE_VAR: contextvars.ContextVar[torch.dtype | None] = contextvars.ContextVar(
    "konfai_coordinate_dtype", default=None
)


def _walk_dtype() -> torch.dtype:
    return _COORDINATE_DTYPE_VAR.get() or _COORDINATE_DTYPE


@contextlib.contextmanager
def coordinate_precision(dtype: torch.dtype) -> Iterator[None]:
    """Run the coordinate walk under ``dtype`` for the duration of the block.

    The default (float64) is the BIT-EXACT contract with SimpleITK and stays untouched unless a
    caller opts out. float32 halves the walk's bytes -- measured ~1.6x on a bandwidth-bound GPU
    walk, and twice the rows per slab -- at a coordinate error of about |world| / 2^24: a few 1e-4
    of a voxel on world coordinates of order 1e4-1e5 units, growing with the magnitude. A NEAREST
    pick within that band of a .5 boundary lands on the other voxel. Intensity chains may take the
    trade; label chains and anything that must reproduce sitk.Resample bit for bit must not.
    """
    token = _COORDINATE_DTYPE_VAR.set(dtype)
    try:
        yield
    finally:
        _COORDINATE_DTYPE_VAR.reset(token)


# --------------------------------------------------------------------------------------------------
# The rules every sampler below obeys. There are two gather strategies for one arithmetic: per-axis
# maps where the coordinate is separable, eight flat corners where a displacement makes it not: and
# the strategies differ for a measured reason. The RULES must not: written out at each site they
# drift, and the drift is silent because a resampled volume looks right either way.


def sampling_dtype(tensor: torch.Tensor) -> torch.dtype:
    """The dtype to accumulate a weighted sum of ``tensor``'s voxels in.

    An integer input has no arithmetic of its own to interpolate with. A CPU half does, and it should
    not be used: torch's CPU Half kernels are missing from older releases and lossy over a sum of
    eight terms, at values a scanner actually produces. A CUDA half keeps its own; every mode has a
    Half kernel there, and upcasting a whole multi-class volume would double its memory for nothing.
    """
    if not tensor.is_floating_point():
        return torch.float32
    if tensor.device.type == "cpu" and tensor.dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return tensor.dtype


def nearest_index(coordinate: torch.Tensor) -> torch.Tensor:
    """ITK's nearest: round half UP on the continuous source index.

    ``torch.round`` breaks a tie to the even index, and ``F.interpolate``'s nearest is
    ``floor(o * scale)``: a statement about a size RATIO, which says nothing once the target grid
    carries an origin of its own. On a label map either wrong rule still yields a label map.
    """
    return torch.floor(coordinate + 0.5).to(torch.long)


def window_index(index: torch.Tensor, n_in: int, region_start: int, window: int) -> torch.Tensor:
    """A global source index as an offset into the sub-region that was actually read.

    Clamped twice, and both matter: to the SOURCE first, so a tap past the volume reproduces the
    border value rather than wrapping, and to the WINDOW second, so it stays inside the buffer on
    hand. The second clamp is only ever load-bearing where the first already put the sample outside,
    which the caller masks to fill: or, for a halo'd read, where the declared bound was checked.
    """
    return torch.clamp(torch.clamp(index, 0, n_in - 1) - region_start, 0, window - 1)


def _kernel_weights(offset: torch.Tensor, order: int) -> torch.Tensor:
    """The 1-D B-spline weight at distance ``offset``: ITK's own kernels.

    Order 1 is the linear hat; order 3 is ``itkBSplineKernelFunction``'s cubic. Both are
    non-negative and sum to one over their support, which is what makes ``sup |values|`` a bound
    on the displacement at every point rather than at the samples.
    """
    distance = offset.abs()
    if order == 1:
        return torch.clamp(1.0 - distance, min=0.0)
    if order == 3:
        near = (4.0 - 6.0 * distance**2 + 3.0 * distance**3) / 6.0
        far = (2.0 - distance) ** 3 / 6.0
        return torch.where(distance < 1.0, near, torch.where(distance < 2.0, far, torch.zeros_like(distance)))
    # Unreachable: DisplacementStage refuses any other order where it is built.
    raise ValueError(f"No B-spline kernel of order {order}: KonfAI evaluates {SUPPORTED_SPLINE_ORDERS}.")


def _cubic_weights(offset: torch.Tensor) -> torch.Tensor:
    """Keys' cubic convolution at a = -1/2 (Catmull-Rom): the interpolating cubic.

    Four taps per axis, exact through the samples (weight 1 at distance 0, 0 at 1 and 2) and exact
    on polynomials up to degree two, which is what the a = -1/2 choice buys. NOT the order-3
    B-spline above: that one smooths unless its input is prefiltered coefficients, which a
    displacement stage stores and an image does not. The weights go negative between 1 and 2:
    that is what preserves an edge, and why a blend can overshoot the values it read.
    """
    distance = offset.abs()
    near = 1.0 + distance * distance * (1.5 * distance - 2.5)
    far = 2.0 - distance * (4.0 - distance * (2.5 - 0.5 * distance))
    return torch.where(distance < 1.0, near, torch.where(distance < 2.0, far, torch.zeros_like(distance)))


def _saturate_overshoot(blended: torch.Tensor, payload_dtype: torch.dtype) -> torch.Tensor:
    """A cubic blend can overshoot the payload's range, and an integer cast WRAPS instead of
    saturating: the bright rim a negative lobe builds beside an edge would come back as the
    opposite extreme. Clamping to the dtype's range is the saturating cast ITK performs."""
    if payload_dtype.is_floating_point:
        return blended
    info = torch.iinfo(payload_dtype)
    return blended.clamp(float(info.min), float(info.max))


def _apply(points_xyz: torch.Tensor, affine: AffineMap, device: torch.device) -> torch.Tensor:
    """``translation + Σ_j column_j · p_j``, accumulated in ``j`` order. ITK's own association.

    Not ``points @ matrix.T + offset``. That is the same sum in a different order, which BLAS is
    free to reassociate, and the two answers differ in the last bit; a continuous index landing on
    an exact half then rounds to the other voxel. The one matmul saved is not worth a label map
    that disagrees with ``sitk.Resample`` in whole columns.
    """
    matrix = torch.tensor(affine.matrix, dtype=_walk_dtype(), device=device)
    out = torch.tensor(affine.translation, dtype=_walk_dtype(), device=device).expand(points_xyz.shape).clone()
    for j in range(affine.rank):
        out = out + points_xyz[..., j, None] * matrix[:, j]
    return out


def _to_index(world_xyz: torch.Tensor, grid: Grid, device: torch.device) -> torch.Tensor:
    """World to continuous index, in ITK's arithmetic: subtract the origin FIRST, then accumulate.

    ``TransformPhysicalPointToContinuousIndex`` sums ``M[i][j] * (p_j - O_j)`` from zero. Folding the
    origin into a translation instead (which is what a generic inverted affine map is) reassociates
    a difference of large world coordinates against a small one and moves the last bit.
    """
    matrix = torch.tensor(grid.world_to_index.matrix, dtype=_walk_dtype(), device=device)
    origin = torch.tensor(grid.origin_xyz, dtype=_walk_dtype(), device=device)
    shifted = world_xyz - origin
    out = torch.zeros_like(world_xyz)
    for j in range(grid.rank):
        out = out + shifted[..., j, None] * matrix[:, j]
    return out


def _displacement_at(stage: DisplacementStage, world_xyz: torch.Tensor, device: torch.device) -> torch.Tensor:
    """The stage's displacement at each world point: zero where its grid does not reach.

    ITK returns the identity outside a BSpline's valid region and outside a field's domain, so a
    point past the values simply moves by nothing. That is also what makes a region of a field a
    legal stand-in for the whole of it: the part it does not cover was going to be identity here
    anyway, which is exactly why the region handed in must COVER the target region, and why a
    short one degrades in silence rather than raising.

    The corner walk is arranged so nothing is computed twice, and ONLY so: each axis's tap weights
    and clamped positions are built once and the corners read them, and the per-corner gather pulls
    all components through one ``index_select``. Every float op the original per-corner form ran is
    still run, on the same values, in the same association, so the result is bit-identical: the
    weight product multiplies axis 0 first, the corner sum accumulates in ``itertools.product``
    order, and a gather copies without rounding.
    """
    rank = stage.grid.rank
    index = _to_index(world_xyz, stage.grid, device)  # continuous index on the value grid, (x, y, z)
    extent_xyz = [int(stage.grid.size_zyx[rank - 1 - axis]) for axis in range(rank)]

    if stage.order != 1:
        # ITK admits a continuous index within ~4 ulps of the spline's valid-region END by nudging
        # it JUST inside (``InsideValidRegion``): the support then fits and the value is the
        # continuous limit, the outermost tap's weight vanishing at the boundary. Without the nudge
        # the whole last plane of a grid commensurate with the coefficient mesh is silently the
        # identity, and a commensurate grid is exactly what a fitted transform domain produces.
        for axis in range(rank):
            end = float(extent_xyz[axis] - 2)
            ulp = float(np.spacing(np.float64(max(1.0, end))))
            at_end = (index[..., axis] - end).abs() <= 4.0 * ulp
            index[..., axis] = index[..., axis].masked_fill(at_end, end - ulp)

    taps = stage.order + 1
    shift = (stage.order - 1) // 2
    # Per axis, per tap, once, on a CONTIGUOUS copy of that axis: ``index[..., axis]`` is a stride-
    # rank view, and every elementwise op over it ran ~7x slower than over a packed tensor
    # (measured). The copy holds the very same floats. ``base`` is derived per axis and dropped
    # with it, and ``index`` is released before the corner loop: the walk's transient peak is made
    # of tensors that outlive their use.
    #
    # The two domains ITK actually implements, and they differ. A BSpline is the identity unless
    # its whole support lies in the coefficient grid (``InsideValidRegion``), so its edge falls off
    # a control point early. A dense field is an ordinary image: it interpolates anywhere in
    # ``[-0.5, n - 0.5)`` with the taps clamped, which reaches half a voxel past the outermost
    # samples. Using the spline's rule for a field blanks that rim.
    inside = torch.ones(index.shape[:-1], dtype=torch.bool, device=device)
    weight_at: list[list[torch.Tensor]] = []
    position_at: list[list[torch.Tensor]] = []
    for axis in range(rank):
        coordinate = index[..., axis]
        base = torch.floor(coordinate) - shift
        if stage.order == 1:
            inside &= (coordinate >= -0.5) & (coordinate < extent_xyz[axis] - 0.5)
        else:
            inside &= (base >= 0) & (base + taps - 1 < extent_xyz[axis])
        weights, positions = [], []
        for tap in range(taps):
            position = base + tap
            weights.append(_kernel_weights(coordinate - position, stage.order))
            # int32: every clamped position is below the axis extent, and the flat index it feeds
            # stays below the window's voxel count, which no field window approaches 2**31 of.
            positions.append(position.to(torch.int32).clamp(0, extent_xyz[axis] - 1))
        weight_at.append(weights)
        position_at.append(positions)
        del coordinate, base
    del index

    # (voxel, component), contiguous: a gather then lands each voxel's components side by side,
    # which is the layout the accumulator wants -- the (component, voxel) form needed a strided
    # transpose per corner, measured at 2.6x the cost for the same numbers.
    flat_values = stage.tensor_by_voxel(device, _walk_dtype())
    out = torch.zeros((*inside.shape, rank), dtype=_walk_dtype(), device=device)
    for corner in itertools.product(range(taps), repeat=rank):
        weight = weight_at[0][corner[0]]
        for axis in range(1, rank):
            weight = weight * weight_at[axis][corner[axis]]
        # ``values`` is (component, Z, Y, X) and row-major over the spatial axes, so the flat index
        # runs the ARRAY axes outermost-first with x fastest: the mirror of the physical order the
        # coordinates arrive in. Running it the other way samples the field transposed, which warps
        # the anatomy somewhere plausible and wrong.
        # int64 for the flat index: a full-resolution field window exceeds 2**31 voxels, and the
        # per-axis int32 positions widen exactly on the first multiply.
        flat_index = position_at[rank - 1][corner[rank - 1]].to(torch.long)
        for array_axis in range(1, rank):
            axis = rank - 1 - array_axis
            flat_index = flat_index * int(stage.grid.size_zyx[array_axis]) + position_at[axis][corner[axis]]
        # One gather for all components; a copy, so the numbers are the stored ones. NOT addcmul_:
        # a fused multiply-add rounds once where ``out + g * w`` rounds twice, and the last bit is
        # the contract.
        gathered = flat_values[flat_index.reshape(-1)]
        # In place: the same two roundings as ``out = out + g * w`` (verified bitwise), one fewer
        # full-size allocation per corner -- and per-corner allocations are what the walk's
        # transient peak is made of.
        out += gathered.reshape(out.shape) * weight.unsqueeze(-1)
        del gathered, flat_index, weight
    out *= inside.unsqueeze(-1)
    return out


def source_index(
    target_grid: Grid,
    source_grid: Grid,
    stages: SpatialStages,
    device: torch.device,
    budget_bytes: float | None = None,
) -> torch.Tensor:
    """One source continuous index per voxel of ``target_grid``, shaped ``(*size_zyx, rank)`` in ``(x, y, z)``.

    ``target_grid`` is the REGION's grid: its own origin, not the volume's.

    THE WORLD POINT IS MATERIALISED, never folded away. Target index to world and world to source
    index compose into one affine that is algebraically identical and one matmul cheaper, and it is
    the wrong arithmetic: ITK takes the two steps separately, so the two associations disagree in the
    last bit, and a continuous index landing EXACTLY on ``k + 0.5`` (which is what a target grid
    commensurate with its source produces, in whole columns at a time) then rounds to a different
    voxel. Measured against ``sitk.Resample``: 150 of 1050 voxels of a label map, every one of them
    a legal-looking label. Consecutive affine STAGES are still folded, that is between two world
    points, where nothing rounds to an index.
    """
    rank = target_grid.rank
    rows_total = int(target_grid.size_zyx[0])
    # The walk holds several float64 tensors per voxel at once (world, index, per-tap weights and
    # positions, the corner gather), so a large region's TRANSIENTS dwarf its result: ~30 GB beside
    # a 3.6 GB answer, measured on an ExaSPIM slab. Slabbing the leading array axis bounds them
    # without touching a value: every op here is per-voxel, and a sub-range arange holds the very
    # integers the full one holds, so a slab computes bit for bit what the whole region computes.
    rows = walk_rows(target_grid, stages, device, budget_bytes)
    if rows >= rows_total:
        return source_index_rows(target_grid, source_grid, stages, device, 0, rows_total)
    out = torch.empty(
        (rows_total, *[int(e) for e in target_grid.size_zyx[1:]], rank), dtype=_walk_dtype(), device=device
    )
    for start in range(0, rows_total, rows):
        stop = min(rows_total, start + rows)
        out[start:stop] = source_index_rows(target_grid, source_grid, stages, device, start, stop)
    return out


#: How long a probed default budget stands before the machine is asked again. The probe reads the
#: cgroup, /proc and the device's free memory (197-211 us per call, measured) and is asked once per
#: patch; a budget only sizes a slab, and a slab changes no value, so a second of staleness costs
#: nothing.
_BUDGET_TTL_SECONDS = 1.0
_default_budgets: dict[tuple, tuple[float, float]] = {}
_now = time.monotonic


def _default_walk_budget(device: torch.device) -> float:
    """The budget for a walk on ``device``, probed once per :data:`_BUDGET_TTL_SECONDS`.

    THE RUN'S DECLARED BUDGET FIRST, and the machine's only when nothing was declared. A walk that
    sizes itself from free memory is a stage that holds what the machine happens to have rather
    than what the run promised: measured on an oblique resample, whose map does not factorise so
    the general walk runs, the stage held 21 volumes-worth of its input against a declaration of 3
    -- and would have held more on an emptier machine. That is also why the declaration could not
    be right: what the stage holds was not a property of the stage.

    Undeclared, the machine's own share stands (:func:`~konfai.data.patching.device_capped_budget`):
    on CUDA half the free VRAM, on CPU a quarter of what this process may still allocate.

    IT IS THE CHAIN SHARE AGAIN, DELIBERATELY, and that reads like double-spending: the sweep takes
    regions + chains (85%) and the block price already folds in what the chain holds, so a walk
    asking for chains again could reach 120% of the declaration on paper. It does not, because the
    two are not independent -- the walk's target grid IS the sweep's region, so it slabs inside a
    block that ``sweep_block_bytes`` already sized against ``Resample``'s declaration, and the block
    price binds first. Measured on an oblique resample (verified non-separable, on the card, where
    the host hands a non-factorising map to ITK and this walk never runs): peak against declaration
    0.10x at 4 GiB, 0.13 at 2, 0.19 at 1, 0.29 at 512 MiB, 0.42 at 256, monotone and never near 100.

    Bounding it by the stage's own declaration instead (``case_working_multiple`` times the block it
    was handed) was built and MEASURED WORSE: that product is the LARGER of the two at these region
    sizes, so it loosens rather than tightens, and the peak went 141 -> 188 MiB at 512 and 103 -> 123
    at 256 with the seconds unchanged. This share is the tighter bound available. Do not re-derive it.
    """
    from konfai.data.patching import device_capped_budget
    from konfai.utils.budget import available_memory_bytes, budget_share, per_rank_budget_bytes

    declared = per_rank_budget_bytes()
    key = (str(device), declared)
    now = _now()
    cached = _default_budgets.get(key)
    if cached is not None and now < cached[0]:
        return cached[1]
    wanted = budget_share("chains", declared) if declared else available_memory_bytes()[0] * 0.25
    budget = device_capped_budget(wanted, device) or 0.0
    _default_budgets[key] = (now + _BUDGET_TTL_SECONDS, budget)
    return budget


def walk_rows(target_grid: Grid, stages: SpatialStages, device: torch.device, budget_bytes: float | None = None) -> int:
    """How many leading-axis rows of ``target_grid`` one walk slab may cover under ``budget_bytes``.

    Priced from the walk itself: per voxel it holds the world point, the continuous index and the
    output (rank each), a weight and a position per axis per tap, and the gather's flat index and
    result -- doubled, because torch materializes each binary op's result beside its operands. The
    default budget is the machine's (:func:`_default_walk_budget`).
    """
    if budget_bytes is None:
        budget_bytes = _default_walk_budget(torch.device(device))
    plane = 1
    for extent in target_grid.size_zyx[1:]:
        plane *= int(extent)
    rank = target_grid.rank
    taps = max((stage.order + 1 for stage in stages if isinstance(stage, DisplacementStage)), default=2)
    itemsize = torch.tensor([], dtype=_walk_dtype()).element_size()
    per_voxel = 2 * itemsize * (3 * rank + 2 * rank * taps + rank + 1)
    total = int(target_grid.size_zyx[0])
    rows = max(1, min(total, int(budget_bytes) // max(1, plane * per_voxel)))
    # Balanced slabs: 31 rows under a 15-row ceiling is 16 + 15, never 15 + 15 + 1 -- a one-row
    # tail pays every fixed cost of a walk (meshgrid, affines, the field's taps) for one row's
    # worth of work. Same ceiling, same values, one launch fewer.
    return -(-total // -(-total // rows))


def source_index_rows(
    target_grid: Grid,
    source_grid: Grid,
    stages: SpatialStages,
    device: torch.device,
    start: int,
    stop: int,
) -> torch.Tensor:
    """:func:`source_index` over rows ``[start, stop)`` of the leading array axis.

    The row indices stay GLOBAL to ``target_grid`` and go through ITS map: re-deriving a chunk as a
    ``sub_grid`` would fold the start into a new origin, a different association whose world points
    disagree with the whole region's in the last bit: which is the whole disagreement
    :func:`source_index`'s docstring forbids.
    """
    rank = target_grid.rank
    axes = [torch.arange(int(extent), dtype=_walk_dtype(), device=device) for extent in reversed(target_grid.size_zyx)]
    axes[-1] = torch.arange(start, stop, dtype=_walk_dtype(), device=device)
    # meshgrid in array order (Z, Y, X) with the physical components last, so the tensor indexes
    # like the volume it will sample.
    grids = torch.meshgrid(*reversed(axes), indexing="ij")
    world = _apply(torch.stack(list(reversed(grids)), dim=-1), target_grid.index_to_world, device)

    pending = AffineMap.identity(rank)
    for stage in stages:
        if isinstance(stage, AffineStage):
            pending = pending.then(stage.map)
            continue
        if not pending.is_identity:
            world = _apply(world, pending, device)
            pending = AffineMap.identity(rank)
        world = world + _displacement_at(stage, world, device)  # not +=: the stage READS world
    if not pending.is_identity:
        world = _apply(world, pending, device)
    return _to_index(world, source_grid, device)


def _is_diagonal(matrix: np.ndarray) -> bool:
    """Whether every off-diagonal entry is EXACTLY zero: no tolerance, deliberately.

    A tolerance would admit maps whose separable form is only nearly the general one, and the two
    would then disagree in the last bit at exactly the coordinates that round to a different voxel.
    Exact is also not restrictive: an axis-aligned or axis-flipped grid gives exact zeros, and the
    inverse of such a matrix keeps them.
    """
    return not np.any(matrix - np.diag(np.diag(matrix)))


def separable_source_index(
    target_grid: Grid,
    source_grid: Grid,
    stages: SpatialStages,
    device: torch.device,
) -> list[torch.Tensor] | None:
    """One source index per target ROW of each array axis, or ``None`` when the map does not factorise.

    THE SAME ARITHMETIC, with the terms that are exactly zero left out. Each component of
    :func:`source_index` is ``translation_k + Σ_j p_j · M[k, j]``; with ``M`` diagonal every ``j ≠ k``
    contributes ``p_j · 0.0``, and adding an exact zero changes no float. So this is bit-identical
    where it applies rather than merely close, which is what lets one case take it and another the
    general path without the two ever having to be reconciled.

    A stored map factorises on the same terms. Affine stages fold into the one map the general
    walk applies between its two world points (:func:`source_index_rows`'s ``pending``: the same
    ``then`` chain from the identity, so the fold here is that map bit for bit), and a fold that
    is exactly diagonal (a translation, an axis-aligned scale, their composition) is applied per
    component as the walk applies it: ``translation_k + w_k · P[k, k]``, the other columns' terms
    being exact zeros. A displacement stage never factorises.

    What it saves is not arithmetic but MEMORY TRAFFIC: the general path materialises a coordinate
    per voxel and gathers eight corners through a flat index over the whole volume.
    """
    rank = target_grid.rank
    pending = AffineMap.identity(rank)
    for stage in stages:
        if not isinstance(stage, AffineStage):
            return None
        pending = pending.then(stage.map)
    forward, backward = target_grid.index_to_world, source_grid.world_to_index
    if not (_is_diagonal(forward.matrix) and _is_diagonal(pending.matrix) and _is_diagonal(backward.matrix)):
        return None
    axes: list[torch.Tensor] = []
    for array_axis in range(rank):
        axis = rank - 1 - array_axis  # the physical component this array axis runs along
        index = torch.arange(int(target_grid.size_zyx[array_axis]), dtype=_walk_dtype(), device=device)
        world = float(forward.translation[axis]) + index * float(forward.matrix[axis, axis])
        if not pending.is_identity:
            world = float(pending.translation[axis]) + world * float(pending.matrix[axis, axis])
        axes.append((world - float(source_grid.origin_xyz[axis])) * float(backward.matrix[axis, axis]))
    return axes


def blend_order(target_grid: Grid, source_grid: Grid) -> list[int]:
    """The array axes in the order to blend them: most source voxels per target voxel first.

    Each axis is reduced to its output extent before the next one reads it, so blending the axis
    that shrinks most FIRST makes every later pass smaller. On an isotropic change of spacing that
    is worth nothing; on a thick-slice CT brought to isotropic (64x512x512 at 3x0.7x0.7 mm, where
    z triples while y and x shrink): it is 44 M intermediate elements against 110 M.

    Keyed on the two grids' SPACINGS, never on the extents in hand: a streamed region and the whole
    volume have to blend in the same order or they stop being bit-identical, and their extents
    differ by construction while their spacings do not.
    """
    rank = target_grid.rank
    scale = [
        abs(float(target_grid.spacing_xyz[rank - 1 - axis] / source_grid.spacing_xyz[rank - 1 - axis]))
        for axis in range(rank)
    ]
    return sorted(range(rank), key=lambda array_axis: -scale[array_axis])


def gather_separable(
    source: torch.Tensor,
    axes: list[torch.Tensor],
    source_starts_zyx: list[int],
    source_shape_zyx: list[int],
    mode: str,
    fill: float,
    blend: list[int] | None = None,
) -> torch.Tensor:
    """:func:`gather`'s rules over a map that factorises, one index per axis, no volume.

    Same interval, same tap clamp, same round-half-up, same fill. It sums the tensor product in a
    different ORDER than the general gather (axis by axis rather than corner by corner), so the
    two agree to float rounding rather than bit for bit. That costs nothing: a map either factorises
    or it does not, so no case is ever served by both. The equality that has to be exact is a
    streamed region against the whole volume, and it is: the per-axis coordinates are global, so a
    region takes a sub-range of the very numbers the whole volume takes.
    """
    rank = len(axes)
    window_zyx = [int(extent) for extent in source.shape[1:]]
    extent_zyx = [int(axis.numel()) for axis in axes]
    device = source.device

    inside_axes = [(axis >= -0.5) & (axis < source_shape_zyx[array_axis] - 0.5) for array_axis, axis in enumerate(axes)]
    out_shape = [int(source.shape[0]), *extent_zyx]
    if not all(bool(mask.any()) for mask in inside_axes):
        # Working dtype then cast, as the masked tail fills: a float32 detour quantizes a float64
        # fill and a fully-outside slab then differs from the unslabbed pass in its fill bits.
        return torch.full(out_shape, fill, device=device, dtype=sampling_dtype(source)).type(source.dtype)

    def local(index: torch.Tensor, array_axis: int) -> torch.Tensor:
        return window_index(index, source_shape_zyx[array_axis], source_starts_zyx[array_axis], window_zyx[array_axis])

    def broadcast(values: torch.Tensor, array_axis: int) -> torch.Tensor:
        shape = [1] * (rank + 1)
        shape[array_axis + 1] = -1
        return values.reshape(shape)

    def is_identity(array_axis: int) -> bool:
        """Whether this axis reads itself: the same voxels, in order, unblended."""
        axis = axes[array_axis]
        if int(axis.numel()) != window_zyx[array_axis] or source_starts_zyx[array_axis] != 0:
            return False
        return bool(torch.equal(axis, torch.arange(int(axis.numel()), dtype=axis.dtype, device=device)))

    def taps(tensor: torch.Tensor, array_axis: int, index: torch.Tensor) -> torch.Tensor:
        """``tensor.index_select(array_axis + 1, index)``, spelled as an index: the same copy of the
        same voxels. index_select on an inner axis walks a scalar loop on the host (measured 0.5 to
        1.5 s against 9 ms for the index on a [1, 256, 256, 256] float32, 0.3 ms either way on CUDA)."""
        return tensor[(*[slice(None)] * (array_axis + 1), index)]

    out = source if mode == "nearest" else source.type(sampling_dtype(source))
    for array_axis in range(rank) if blend is None else blend:
        # An axis the map leaves alone is read by leaving it alone: two gathers and a blend that
        # would reproduce the input exactly, over the largest tensor in flight, for nothing.
        if is_identity(array_axis):
            continue
        axis = axes[array_axis]
        if mode == "nearest":
            out = taps(out, array_axis, local(nearest_index(axis), array_axis))
            continue
        if mode == "cubic":
            # The same axis-by-axis reduction as the blend below, four taps instead of two. The
            # weights come from the GLOBAL coordinate, so a region sums the very numbers the whole
            # volume sums.
            base = torch.floor(axis) - 1.0
            index = base.to(torch.long)
            blended: torch.Tensor | None = None
            for tap in range(4):
                weight = broadcast(_cubic_weights(axis - (base + tap)).to(out.dtype), array_axis)
                term = taps(out, array_axis, local(index + tap, array_axis)) * weight
                blended = term if blended is None else blended + term
            out = blended if blended is not None else out
            continue
        # One axis at a time, not eight corners at once. The tensor product is the same sum, but
        # blending axis by axis reduces each extent before the next axis reads it: six gathers over
        # shrinking tensors instead of twenty-four over the largest one.
        base = torch.floor(axis)
        share = broadcast((axis - base).to(out.dtype), array_axis)
        index = base.to(torch.long)
        low = taps(out, array_axis, local(index, array_axis))
        high = taps(out, array_axis, local(index + 1, array_axis))
        # lerp fuses the three passes `low * (1 - w) + high * w` into one. Exact at w = 0, which is
        # the only endpoint reachable: w is `x - floor(x)`, so it never reaches 1.
        out = torch.lerp(low, high, share)
    # No float trip for a nearest pick: copies stay in the payload's own dtype the whole way, the
    # rule `gather` already holds -- a float32 detour rounds every integer label above 2**24, and
    # the two paths then disagree on the very volumes they are supposed to serve identically.
    if mode == "cubic":
        out = _saturate_overshoot(out, source.dtype)

    if all(bool(mask.all()) for mask in inside_axes):
        # Nothing of this region falls outside the source, which is the ordinary case for a resample
        # within a volume, so neither the mask nor the pass that applies it is built at all.
        return out.type(source.dtype)
    mask = inside_axes[0]
    for array_axis in range(1, rank):
        mask = mask.unsqueeze(-1) & inside_axes[array_axis]
    return out.masked_fill(~mask.unsqueeze(0), fill).type(source.dtype)


def gather(
    source: torch.Tensor,
    coordinates_xyz: torch.Tensor,
    source_starts_zyx: list[int],
    source_shape_zyx: list[int],
    mode: str,
    fill: float,
) -> torch.Tensor:
    """ITK's sampler at an arbitrary coordinate per voxel, over the region actually read.

    The rule is ``sitk.Resample``'s, and it is the same one the separable samplers in
    ``konfai.data.transform`` obey: a sample is inside while its continuous source index lies in
    ``[-0.5, n - 0.5)``; inside, the interpolation taps clamp to the buffer, so the half-voxel rim
    past the outermost voxel centres reproduces the border value instead of falling off; outside,
    the sample is ``fill``.

    ``source`` covers ``source_starts_zyx`` onward of a volume of ``source_shape_zyx``; the
    coordinates are GLOBAL indices of that volume.

    THE TWO MODES TAKE DIFFERENT ROUTES, and the difference is what each can afford. Nearest is one
    gather on the exact index, because the pick is discontinuous and the last bit decides it. Linear
    is eight gathers over a flat index, which ``grid_sample`` does as one fused kernel, at the cost
    of normalising by the extent it is handed, so a streamed region and the whole volume agree to
    ~1e-5 rather than exactly.
    """
    rank = coordinates_xyz.shape[-1]
    window_zyx = [int(extent) for extent in source.shape[1:]]
    extent_zyx = list(coordinates_xyz.shape[:-1])
    device = source.device

    inside = torch.ones(extent_zyx, dtype=torch.bool, device=device)
    for axis in range(rank):
        array_axis = rank - 1 - axis
        coordinate = coordinates_xyz[..., axis]
        inside &= (coordinate >= -0.5) & (coordinate < source_shape_zyx[array_axis] - 0.5)

    out_shape = [int(source.shape[0]), *extent_zyx]
    if not bool(inside.any()):
        # In the WORKING dtype then cast, exactly as the masked path fills: filling in float32
        # first quantizes a float64 fill, and a walk slab that clears the source then disagrees
        # with the unslabbed pass in the last bits of its fill value.
        return torch.full(out_shape, fill, device=device, dtype=sampling_dtype(source)).type(source.dtype)

    # A nearest pick copies voxels: no blend, no working dtype, and no float trip for a label,
    # whose values above 2**24 a float32 cannot carry back (gather_separable holds the same rule).
    work = source if mode == "nearest" else source.type(sampling_dtype(source))
    if mode == "nearest":
        # One gather, on exact index arithmetic: a nearest pick is discontinuous, so the last bit of
        # a coordinate decides which voxel it lands on. Measured over 300 random grid pairs, a
        # single-precision coordinate moves 2.6% of the picks, in a label map, 2.6% of the voxels
        # wearing a different label. There are no eight corners to fuse here, so nothing is bought
        # by handing it to a kernel that normalises coordinates.
        flat = torch.zeros(extent_zyx, dtype=torch.long, device=device)
        for array_axis in range(rank):
            index = nearest_index(coordinates_xyz[..., rank - 1 - array_axis])
            local = window_index(
                index, source_shape_zyx[array_axis], source_starts_zyx[array_axis], window_zyx[array_axis]
            )
            flat = flat * window_zyx[array_axis] + local
        # An index, not index_select: the same copy, 3.7x faster on the host over 16.7M voxels
        # (27 vs 125 ms at one channel, 171 vs 477 at four, measured; 2 ms either way on CUDA).
        picked = work.reshape(int(work.shape[0]), -1)[:, flat.reshape(-1)].reshape(out_shape)
        return picked.masked_fill(~inside.unsqueeze(0), fill).type(source.dtype)

    if mode == "cubic":
        # Keys' four taps per axis, corner by corner over a flat index: grid_sample has no cubic in
        # 3-D, and the corner walk keeps the coordinates GLOBAL, so a streamed region and the whole
        # volume agree exactly where the fused blend below pays ~1e-5 to normalisation.
        base_xyz = torch.floor(coordinates_xyz) - 1.0
        out = torch.zeros(out_shape, dtype=work.dtype, device=device)
        for corner in itertools.product(range(4), repeat=rank):
            weight = torch.ones(extent_zyx, dtype=_walk_dtype(), device=device)
            flat = torch.zeros(extent_zyx, dtype=torch.long, device=device)
            for array_axis in range(rank):
                axis = rank - 1 - array_axis
                position = base_xyz[..., axis] + corner[axis]
                weight = weight * _cubic_weights(coordinates_xyz[..., axis] - position)
                local = window_index(
                    position.to(torch.long),
                    source_shape_zyx[array_axis],
                    source_starts_zyx[array_axis],
                    window_zyx[array_axis],
                )
                flat = flat * window_zyx[array_axis] + local
            gathered = work.reshape(int(work.shape[0]), -1)[:, flat.reshape(-1)].reshape(out_shape)
            out = out + gathered * weight.to(work.dtype).unsqueeze(0)
        out = _saturate_overshoot(out, source.dtype)
        return out.masked_fill(~inside.unsqueeze(0), fill).type(source.dtype)

    # A BLEND goes through grid_sample, which is one fused kernel where the eight corners are eight
    # gathers over a flat index. `align_corners=False` IS ITK's domain: a sample is inside while
    # its continuous index lies in [-0.5, n - 0.5): and `padding_mode="border"` IS ITK's tap clamp;
    # what grid_sample has no notion of is the FILL, so the mask computed above still applies it.
    #
    # It costs the bit-identity between a streamed region and the whole volume, and that is the whole
    # of what it costs: grid_sample takes NORMALISED coordinates, so it divides by the extent of the
    # tensor handed to it, and a region is handed a window. The two agree to ~1e-5 instead of
    # exactly. Deliberate, and asked for: 4x on a warp.
    # grid_sample orders the last dimension (x, y, z): the mirror of the array axes, which is the
    # order the coordinates already arrive in. The grid counts voxels in float32 WHATEVER the
    # payload: a half grid quantizes a coordinate at ~2^-11 of the window extent: 0.06 voxel on a
    # 512 axis, far past the ~1e-5 band above, and the window is upcast with it because
    # grid_sample takes one dtype. sampling_dtype's keep-half trade was measured for the values.
    #
    # Each axis is normalised straight into the grid it belongs in. Keeping them and stacking them
    # first holds the whole grid twice, once in the coordinates' float64 and once in the cast of
    # it, where the cast is what the assignment does anyway.
    blend_dtype = torch.float32 if work.dtype in (torch.float16, torch.bfloat16) else work.dtype
    sampling = torch.empty((*coordinates_xyz.shape[:-1], rank), dtype=blend_dtype, device=coordinates_xyz.device)
    for array_axis in range(rank):
        axis = rank - 1 - array_axis
        extent = window_zyx[array_axis]
        shifted = coordinates_xyz[..., axis] - float(source_starts_zyx[array_axis])
        sampling[..., axis] = (2.0 * shifted + 1.0) / extent - 1.0
    out = torch.nn.functional.grid_sample(
        work.to(blend_dtype).unsqueeze(0),
        sampling.unsqueeze(0),
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    ).squeeze(0)
    return out.masked_fill(~inside.unsqueeze(0), fill).type(source.dtype)


def source_window(
    target_grid: Grid,
    source_grid: Grid,
    bound: TransformBound,
    margin: int = 1,
) -> tuple[slice, ...]:
    """The source window a target region pulls, from the bound alone: no voxel sampled.

    Closed form, which is what lets a cost model price a decomposition without doing the bounding
    work for it: the target region's world box, mapped through the bound (an exact affine hull
    grown by the residual), read back as a clamped index window on the source.
    """
    return source_grid.index_window(bound.map_box(target_grid.world_box()), margin)
