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

Two halves. The COORDINATE PRODUCER turns a decoded transform into one source index per target
voxel; the GATHER is ITK's sampler and is written once.

Nothing here resamples through SimpleITK. A BSpline is evaluated from its coefficient grid and a
dense field from its samples, with the same tensor-product kernel arithmetic ITK uses. The one thing
read off SimpleITK is the world-to-index matrix ITK holds for a grid (:func:`scanline_index`).
"""

from __future__ import annotations

import contextlib
import contextvars
import itertools
import math
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
    WorldBox,
)

#: Coordinates are accumulated in float64 and only the gather runs in the payload's dtype. In
#: float32 a world coordinate keeps about four digits past the voxel, enough to move a sample across
#: a voxel boundary on a large grid.
_COORDINATE_DTYPE = torch.float64

#: The walk's dtype for the CURRENT context, entered by :func:`coordinate_precision`. The choice
#: belongs to the stage that knows its data: an intensity resample may trade the last bits for
#: bandwidth, a label resample may not.
_COORDINATE_DTYPE_VAR: contextvars.ContextVar[torch.dtype | None] = contextvars.ContextVar(
    "konfai_coordinate_dtype", default=None
)


def _walk_dtype() -> torch.dtype:
    return _COORDINATE_DTYPE_VAR.get() or _COORDINATE_DTYPE


@contextlib.contextmanager
def coordinate_precision(dtype: torch.dtype) -> Iterator[None]:
    """Run the coordinate walk under ``dtype`` for the duration of the block.

    The default (float64) is the BIT-EXACT contract with SimpleITK. float32 halves the walk's bytes
    at a coordinate error of about |world| / 2^24, and a NEAREST pick within that band of a .5
    boundary lands on the other voxel. Intensity chains may take the trade; label chains and
    anything that must reproduce sitk.Resample bit for bit must not.
    """
    token = _COORDINATE_DTYPE_VAR.set(dtype)
    try:
        yield
    finally:
        _COORDINATE_DTYPE_VAR.reset(token)


# --------------------------------------------------------------------------------------------------
# The rules every sampler below obeys. Two gather strategies for one arithmetic: per-axis maps where
# the coordinate is separable, eight flat corners where a displacement makes it not.


def sampling_dtype(tensor: torch.Tensor) -> torch.dtype:
    """The dtype to accumulate a weighted sum of ``tensor``'s voxels in.

    An integer input has no arithmetic of its own to interpolate with. A CPU half is lossy over a sum
    of eight terms and is upcast; a CUDA half keeps its own.
    """
    if not tensor.is_floating_point():
        return torch.float32
    if tensor.device.type == "cpu" and tensor.dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return tensor.dtype


#: The dtypes labels arrive in: a byte label map, an ``Argmax``'s int64, a boolean mask. Scanners store
#: intensities as int16, uint16 or int32, so those stay images.
LABEL_DTYPES = (torch.uint8, torch.int64, torch.bool)


def default_interpolation(tensor: torch.Tensor) -> str:
    """``nearest`` for a label dtype, ``linear`` otherwise: the interpolation a config that states none gets.

    A dtype cannot settle this for every volume (an int16 label map exists), so a stated ``interpolation``
    always wins over it.
    """
    return "nearest" if tensor.dtype in LABEL_DTYPES else "linear"


def nearest_index(coordinate: torch.Tensor) -> torch.Tensor:
    """ITK's nearest: round half UP on the continuous source index. Not ``torch.round``, which ties
    to the even index, and not ``F.interpolate``'s ``floor(o * scale)``, which is a statement about a
    size ratio and says nothing once the target grid carries an origin of its own."""
    return torch.floor(coordinate + 0.5).to(torch.long)


def window_index(index: torch.Tensor, n_in: int, region_start: int, window: int) -> torch.Tensor:
    """A global source index as an offset into the sub-region that was actually read.

    Clamped twice: to the SOURCE first, so a tap past the volume reproduces the border value rather
    than wrapping, and to the WINDOW second, so it stays inside the buffer on hand.
    """
    return torch.clamp(torch.clamp(index, 0, n_in - 1) - region_start, 0, window - 1)


def _kernel_weights(offset: torch.Tensor, order: int) -> torch.Tensor:
    """The 1-D B-spline weight at distance ``offset``: ITK's own kernels. Order 1 is the linear hat,
    order 3 ``itkBSplineKernelFunction``'s cubic. Both are non-negative and sum to one over their
    support, which makes ``sup |values|`` a bound on the displacement at every point."""
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
    """Keys' cubic convolution at a = -1/2 (Catmull-Rom), four taps per axis, exact through the
    samples. NOT the order-3 B-spline above, which smooths unless its input is prefiltered
    coefficients. The weights go negative between 1 and 2, so a blend can overshoot what it read."""
    distance = offset.abs()
    near = 1.0 + distance * distance * (1.5 * distance - 2.5)
    far = 2.0 - distance * (4.0 - distance * (2.5 - 0.5 * distance))
    return torch.where(distance < 1.0, near, torch.where(distance < 2.0, far, torch.zeros_like(distance)))


def _saturate_overshoot(blended: torch.Tensor, payload_dtype: torch.dtype) -> torch.Tensor:
    """A cubic blend can overshoot the payload's range, and an integer cast WRAPS instead of
    saturating. Clamping to the dtype's range is the saturating cast ITK performs."""
    if payload_dtype.is_floating_point:
        return blended
    info = torch.iinfo(payload_dtype)
    return blended.clamp(float(info.min), float(info.max))


#: The signed dtype of each unsigned one torch has no ``masked_fill`` for: the same bits, one view away.
_SIGNED_VIEW = {torch.uint16: torch.int16, torch.uint32: torch.int32, torch.uint64: torch.int64}


def _masked_fill(values: torch.Tensor, mask: torch.Tensor, fill: float) -> torch.Tensor:
    """``values.masked_fill(mask, fill)`` in ``values``' own dtype, whatever it is. An integer payload
    takes the fill as ITK casts its default pixel value (-1 is 255 in a uint8 volume), and uint16,
    uint32 and uint64 are filled through the signed view of the same bits."""
    if values.is_floating_point() or values.dtype == torch.bool:
        return values.masked_fill(mask, fill)
    value = torch.tensor(fill, dtype=torch.float64).to(values.dtype)
    signed = _SIGNED_VIEW.get(values.dtype)
    if signed is None:
        return values.masked_fill(mask, value.item())
    return values.view(signed).masked_fill(mask, value.view(signed).item()).view(values.dtype)


def _picked(values: torch.Tensor, index: tuple[slice | torch.Tensor, ...]) -> torch.Tensor:
    """``values[index]`` in ``values``' own dtype, whatever it is: CUDA has no index kernel for uint16, uint32 or
    uint64, which are read through the signed view of the same bits."""
    signed = _SIGNED_VIEW.get(values.dtype)
    return values[index] if signed is None else values.view(signed)[index].view(values.dtype)


def _apply(points_xyz: torch.Tensor, affine: AffineMap, device: torch.device) -> torch.Tensor:
    """``translation + Σ_j column_j · p_j``, accumulated in ``j`` order: ITK's own association. Not
    ``points @ matrix.T + offset``, which BLAS is free to reassociate; the two answers differ in the
    last bit and a continuous index landing on an exact half rounds to the other voxel."""
    matrix = torch.tensor(affine.matrix, dtype=_walk_dtype(), device=device)
    out = torch.tensor(affine.translation, dtype=_walk_dtype(), device=device).expand(points_xyz.shape).clone()
    for j in range(affine.rank):
        out = out + points_xyz[..., j, None] * matrix[:, j]
    return out


def _to_index(
    world_xyz: torch.Tensor, grid: Grid, device: torch.device, world_to_index: np.ndarray | None = None
) -> torch.Tensor:
    """World to continuous index, in ITK's arithmetic: subtract the origin FIRST, then accumulate
    ``M[i][j] * (p_j - O_j)`` from zero, as ``TransformPhysicalPointToContinuousIndex`` does. Folding
    the origin into a translation instead moves the last bit. ``M`` is the grid's exact inverse unless
    ``world_to_index`` is given."""
    matrix = torch.tensor(
        grid.world_to_index.matrix if world_to_index is None else world_to_index, dtype=_walk_dtype(), device=device
    )
    origin = torch.tensor(grid.origin_xyz, dtype=_walk_dtype(), device=device)
    shifted = world_xyz - origin
    out = torch.zeros_like(world_xyz)
    for j in range(grid.rank):
        out = out + shifted[..., j, None] * matrix[:, j]
    return out


def _displacement_at(stage: DisplacementStage, world_xyz: torch.Tensor, device: torch.device) -> torch.Tensor:
    """The stage's displacement at each world point: zero where its grid does not reach, ITK
    returning the identity outside a BSpline's valid region and outside a field's domain. The region
    handed in must COVER the target region; a short one degrades in silence rather than raising."""
    rank = stage.grid.rank
    index = _to_index(world_xyz, stage.grid, device)  # continuous index on the value grid, (x, y, z)
    extent_xyz = [int(stage.grid.size_zyx[rank - 1 - axis]) for axis in range(rank)]

    if stage.order != 1:
        # ITK admits a continuous index within ~4 ulps of the spline's valid-region END by nudging
        # it JUST inside (``InsideValidRegion``). Without the nudge the whole last plane of a grid
        # commensurate with the coefficient mesh is silently the identity.
        for axis in range(rank):
            end = float(extent_xyz[axis] - 2)
            ulp = float(np.spacing(np.float64(max(1.0, end))))
            at_end = (index[..., axis] - end).abs() <= 4.0 * ulp
            index[..., axis] = index[..., axis].masked_fill(at_end, end - ulp)

    taps = stage.order + 1
    shift = (stage.order - 1) // 2
    # Per axis, per tap, once, on a CONTIGUOUS copy of that axis: ``index[..., axis]`` is a stride-
    # rank view and every elementwise op over it is far slower than over a packed tensor. The copy
    # holds the very same floats, and ``index`` is released before the corner loop.
    #
    # The two domains ITK implements differ. A BSpline is the identity unless its whole support lies
    # in the coefficient grid (``InsideValidRegion``). A dense field interpolates anywhere in
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
    # which is the layout the accumulator wants.
    flat_values = stage.tensor_by_voxel(device, _walk_dtype())
    out = torch.zeros((*inside.shape, rank), dtype=_walk_dtype(), device=device)
    for corner in itertools.product(range(taps), repeat=rank):
        weight = weight_at[0][corner[0]]
        for axis in range(1, rank):
            weight = weight * weight_at[axis][corner[axis]]
        # ``values`` is (component, Z, Y, X) and row-major over the spatial axes, so the flat index
        # runs the ARRAY axes outermost-first with x fastest, the mirror of the physical order the
        # coordinates arrive in. Running it the other way samples the field transposed. int64 for the
        # flat index: a full-resolution field window exceeds 2**31 voxels.
        flat_index = position_at[rank - 1][corner[rank - 1]].to(torch.long)
        for array_axis in range(1, rank):
            axis = rank - 1 - array_axis
            flat_index = flat_index * int(stage.grid.size_zyx[array_axis]) + position_at[axis][corner[axis]]
        # One gather for all components; a copy, so the numbers are the stored ones. NOT addcmul_:
        # a fused multiply-add rounds once where ``out + g * w`` rounds twice.
        gathered = flat_values[flat_index.reshape(-1)]
        # In place: the same two roundings as ``out = out + g * w``, one full-size allocation fewer.
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
    last bit, and a continuous index landing EXACTLY on ``k + 0.5`` then rounds to a different voxel.
    Consecutive affine STAGES are still folded, which is between two world points, where nothing
    rounds to an index.
    """
    rank = target_grid.rank
    rows_total = int(target_grid.size_zyx[0])
    # The walk holds several float64 tensors per voxel at once (world, index, per-tap weights and
    # positions, the corner gather), so a large region's TRANSIENTS dwarf its result. Slabbing the
    # leading array axis bounds them without touching a value: every op here is per-voxel, and a
    # sub-range arange holds the very integers the full one holds.
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


#: How long a probed default budget stands before the machine is asked again. A budget only sizes a
#: slab, and a slab changes no value, so a second of staleness costs nothing.
_BUDGET_TTL_SECONDS = 1.0
_default_budgets: dict[tuple, tuple[float, float]] = {}
_now = time.monotonic


def _default_walk_budget(device: torch.device) -> float:
    """The budget for a walk on ``device``, probed once per :data:`_BUDGET_TTL_SECONDS`: the run's
    declared budget's chain share, and only when nothing was declared the machine's own share
    (:func:`~konfai.data.patching.device_capped_budget`), half the free VRAM or a quarter of the host."""
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

    Priced from the walk itself: per voxel the world point, the continuous index and the output
    (rank each), a weight and a position per axis per tap, and the gather's flat index and result,
    doubled for the temporaries torch materializes. The default is :func:`_default_walk_budget`.
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
    # Balanced slabs: 31 rows under a 15-row ceiling is 16 + 15, never 15 + 15 + 1, a one-row tail
    # paying every fixed cost of a walk for one row's worth of work.
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

    The row indices stay GLOBAL to ``target_grid`` and go through ITS map: a ``sub_grid`` would fold
    the start into a new origin, whose world points disagree with the whole region's in the last bit.
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


def scanline_map(target_grid: Grid, source_grid: Grid, stages: SpatialStages) -> bool:
    """Whether a resample reads this map as ``sitk.Resample`` does (:func:`scanline_index`): a change
    of grid and nothing else, up to three axes, where ITK's linear blend is the one :func:`_itk_linear`
    reproduces. Whatever the two directions: a target that keeps the source's direction, permuted or
    oblique, puts target voxels EXACTLY half-way between two source voxels, where the last bit of the
    index decides the pick. A map whose matrices are exactly diagonal is read by the separable path
    before this is asked."""
    return not stages and target_grid.rank <= 3


def scanline_index(
    target_grid: Grid, source_grid: Grid, region_zyx: tuple[slice, ...], device: torch.device
) -> torch.Tensor:
    """The source continuous index ``sitk.Resample`` reads each voxel of ``region_zyx`` at, through no map.

    ITK resamples through a linear map scanline by scanline (``ResampleImageFilter::
    LinearThreadedGenerateData``): the index at x = 0 and at x = N of the WHOLE output grid, the voxel
    at x taking ``start + (x / N) * (end - start)``, each end through the world-to-index matrix ITK
    holds (:func:`~konfai.utils.ITK.physical_point_to_index`). Every term is global to
    ``target_grid``, so a region reads exactly what the whole grid reads.
    """
    from konfai.utils.ITK import physical_point_to_index

    dtype = _walk_dtype()
    extent = int(target_grid.size_zyx[-1])
    rows = [torch.arange(int(part.start), int(part.stop), dtype=dtype, device=device) for part in region_zyx[:-1]]
    grids = list(torch.meshgrid(*rows, indexing="ij")) if rows else []
    shape = tuple(int(part.stop - part.start) for part in region_zyx[:-1])
    world_to_index = physical_point_to_index(source_grid)

    def at(x: float) -> torch.Tensor:
        index = torch.stack([torch.full(shape, x, dtype=dtype, device=device), *reversed(grids)], dim=-1)
        return _to_index(_apply(index, target_grid.index_to_world, device), source_grid, device, world_to_index)

    start = at(0.0)
    along = at(float(extent)) - start
    columns = region_zyx[-1]
    # Divided by a tensor: CUDA divides by a Python scalar through its reciprocal, one bit away from the quotient
    # ITK computes, which moves an exact half-voxel tie to the other voxel.
    alpha = torch.arange(int(columns.start), int(columns.stop), dtype=dtype, device=device) / torch.tensor(
        float(extent), dtype=dtype, device=device
    )
    return start.unsqueeze(-2) + alpha.unsqueeze(-1) * along.unsqueeze(-2)


def _is_diagonal(matrix: np.ndarray) -> bool:
    """Whether every off-diagonal entry is EXACTLY zero: no tolerance. A tolerance would admit maps
    whose separable form is only nearly the general one, and an axis-aligned or axis-flipped grid
    gives exact zeros anyway, as does the inverse of such a matrix."""
    return not np.any(matrix - np.diag(np.diag(matrix)))


def separable_source_index(
    target_grid: Grid,
    source_grid: Grid,
    stages: SpatialStages,
    device: torch.device,
    region_zyx: tuple[slice, ...] | None = None,
) -> list[torch.Tensor] | None:
    """One source index per target ROW of each array axis, or ``None`` when the map does not factorise.

    ``region_zyx`` takes the rows of one region, still indexed on ``target_grid``: a region's own
    grid starts from an origin rounded once more, which moves an exact half-voxel tie.

    THE SAME ARITHMETIC, with the terms that are exactly zero left out: each component of
    :func:`source_index` is ``translation_k + Σ_j p_j · M[k, j]``, and with ``M`` diagonal every
    ``j ≠ k`` adds an exact zero. Bit-identical where it applies, not merely close.

    A stored map factorises on the same terms. Affine stages fold into the one map the general walk
    applies between its two world points (:func:`source_index_rows`'s ``pending``), and a fold that
    is exactly diagonal is applied per component as the walk applies it. A displacement stage never
    factorises. What it saves is not arithmetic but MEMORY TRAFFIC.
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
        rows = slice(0, int(target_grid.size_zyx[array_axis])) if region_zyx is None else region_zyx[array_axis]
        index = torch.arange(int(rows.start), int(rows.stop), dtype=_walk_dtype(), device=device)
        world = float(forward.translation[axis]) + index * float(forward.matrix[axis, axis])
        if not pending.is_identity:
            world = float(pending.translation[axis]) + world * float(pending.matrix[axis, axis])
        axes.append((world - float(source_grid.origin_xyz[axis])) * float(backward.matrix[axis, axis]))
    return axes


def blend_order(target_grid: Grid, source_grid: Grid) -> list[int]:
    """The array axes in the order to blend them: most source voxels per target voxel first.

    Each axis is reduced to its output extent before the next one reads it, so blending the axis
    that shrinks most FIRST makes every later pass smaller.

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

    Same interval, same tap clamp, same round-half-up, same fill. It sums the tensor product axis by
    axis rather than corner by corner, so the two agree to float rounding rather than bit for bit,
    and no case is ever served by both. A streamed region and the whole volume do agree exactly: the
    per-axis coordinates are global.
    """
    rank = len(axes)
    window_zyx = [int(extent) for extent in source.shape[1:]]
    extent_zyx = [int(axis.numel()) for axis in axes]
    device = source.device

    inside_axes = [(axis >= -0.5) & (axis < source_shape_zyx[array_axis] - 0.5) for array_axis, axis in enumerate(axes)]
    out_shape = [int(source.shape[0]), *extent_zyx]
    if not all(bool(mask.any()) for mask in inside_axes):
        # Working dtype then cast, as the masked tail fills: a float32 detour quantizes a float64 fill.
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
        same voxels. index_select on an inner axis walks a scalar loop on the host."""
        taps_index: tuple[slice | torch.Tensor, ...] = (*[slice(None)] * (array_axis + 1), index)
        return _picked(tensor, taps_index)

    out = source if mode == "nearest" else source.type(sampling_dtype(source))
    for array_axis in range(rank) if blend is None else blend:
        # An axis the map leaves alone is read by leaving it alone.
        if is_identity(array_axis):
            continue
        axis = axes[array_axis]
        if mode == "nearest":
            out = taps(out, array_axis, local(nearest_index(axis), array_axis))
            continue
        if mode == "cubic":
            # The same axis-by-axis reduction as the blend below, four taps instead of two. The
            # weights come from the GLOBAL coordinate, so a region sums what the whole volume sums.
            base = torch.floor(axis) - 1.0
            index = base.to(torch.long)
            blended: torch.Tensor | None = None
            for tap in range(4):
                weight = broadcast(_cubic_weights(axis - (base + tap)).to(out.dtype), array_axis)
                term = taps(out, array_axis, local(index + tap, array_axis)) * weight
                blended = term if blended is None else blended + term
            out = blended if blended is not None else out
            continue
        # One axis at a time, not eight corners at once. The tensor product is the same sum, and
        # blending axis by axis reduces each extent before the next axis reads it.
        base = torch.floor(axis)
        share = broadcast((axis - base).to(out.dtype), array_axis)
        index = base.to(torch.long)
        low = taps(out, array_axis, local(index, array_axis))
        high = taps(out, array_axis, local(index + 1, array_axis))
        # lerp fuses the three passes `low * (1 - w) + high * w` into one. Exact at w = 0, which is
        # the only endpoint reachable: w is `x - floor(x)`, so it never reaches 1.
        out = torch.lerp(low, high, share)
    # No float trip for a nearest pick: copies stay in the payload's own dtype the whole way, the
    # rule `gather` already holds. A float32 detour rounds every integer label above 2**24.
    if mode == "cubic":
        out = _saturate_overshoot(out, source.dtype)

    if all(bool(mask.all()) for mask in inside_axes):
        # Nothing of this region falls outside the source, so neither the mask nor the pass that
        # applies it is built at all.
        return out.type(source.dtype)
    mask = inside_axes[0]
    for array_axis in range(1, rank):
        mask = mask.unsqueeze(-1) & inside_axes[array_axis]
    return _masked_fill(out, ~mask.unsqueeze(0), fill).type(source.dtype)


#: Voxels the host takes ITK's route for at a time (:func:`gather`): a chunk's corners and values stay
#: in cache, up to four times faster per voxel than a slab-sized pass, and a few MiB are held beside
#: the output whatever the slab.
_HOST_ITK_VOXELS = 1 << 16


def gather(
    source: torch.Tensor,
    coordinates_xyz: torch.Tensor,
    source_starts_zyx: list[int],
    source_shape_zyx: list[int],
    mode: str,
    fill: float,
    itk_blend: bool = False,
) -> torch.Tensor:
    """ITK's sampler at an arbitrary coordinate per voxel, over the region actually read.

    The rule is ``sitk.Resample``'s, and it is the same one the separable samplers in
    ``konfai.data.transform`` obey: a sample is inside while its continuous source index lies in
    ``[-0.5, n - 0.5)``; inside, the interpolation taps clamp to the buffer, so the half-voxel rim
    past the outermost voxel centres reproduces the border value; outside, the sample is ``fill``.

    ``source`` covers ``source_starts_zyx`` onward of a volume of ``source_shape_zyx``; the
    coordinates are GLOBAL indices of that volume.

    Nearest is one gather on the exact index, the pick being discontinuous. Linear goes through
    ``grid_sample``, which normalises by the extent it is handed, so a streamed region and the whole
    volume agree to ~1e-5 rather than exactly; ``itk_blend`` blends as ITK does instead
    (:func:`_itk_linear`), exactly, for a map whose index is ITK's (:func:`scanline_map`).
    """
    rank = coordinates_xyz.shape[-1]
    window_zyx = [int(extent) for extent in source.shape[1:]]
    extent_zyx = list(coordinates_xyz.shape[:-1])
    device = source.device

    voxels = math.prod(extent_zyx)
    if itk_blend and device.type == "cpu" and voxels > _HOST_ITK_VOXELS:
        # Each voxel's value is its own, so the host takes ITK's route a cache-sized chunk at a time.
        points, source = coordinates_xyz.reshape(-1, rank), source.contiguous()
        out = torch.empty((int(source.shape[0]), voxels), dtype=source.dtype)
        for first in range(0, voxels, _HOST_ITK_VOXELS):
            chunk = points[first : first + _HOST_ITK_VOXELS]
            out[:, first : first + len(chunk)] = gather(
                source, chunk, source_starts_zyx, source_shape_zyx, mode, fill, itk_blend
            )
        return out.reshape(int(source.shape[0]), *extent_zyx)

    inside = torch.ones(extent_zyx, dtype=torch.bool, device=device)
    for axis in range(rank):
        array_axis = rank - 1 - axis
        coordinate = coordinates_xyz[..., axis]
        inside &= (coordinate >= -0.5) & (coordinate < source_shape_zyx[array_axis] - 0.5)

    out_shape = [int(source.shape[0]), *extent_zyx]
    if not bool(inside.any()):
        # In the WORKING dtype then cast, exactly as the masked path fills: filling in float32 first
        # quantizes a float64 fill.
        return torch.full(out_shape, fill, device=device, dtype=sampling_dtype(source)).type(source.dtype)

    if itk_blend and mode == "linear":
        blended = _itk_cast(_itk_linear(source, coordinates_xyz, source_starts_zyx, source_shape_zyx))
        # The fill in the payload's own dtype, as ITK casts it: a float32 trip would round a large one.
        return _masked_fill(blended.type(source.dtype), ~inside.unsqueeze(0), fill)

    # A nearest pick copies voxels: no blend, no working dtype, and no float trip for a label, whose
    # values above 2**24 a float32 cannot carry back.
    work = source if mode == "nearest" else source.type(sampling_dtype(source))
    if mode == "nearest":
        # One gather, on exact index arithmetic: a nearest pick is discontinuous, so the last bit of
        # a coordinate decides which voxel it lands on. There are no eight corners to fuse here.
        flat = torch.zeros(extent_zyx, dtype=torch.long, device=device)
        for array_axis in range(rank):
            index = nearest_index(coordinates_xyz[..., rank - 1 - array_axis])
            local = window_index(
                index, source_shape_zyx[array_axis], source_starts_zyx[array_axis], window_zyx[array_axis]
            )
            flat = flat * window_zyx[array_axis] + local
        # An index, not index_select: the same copy, several times faster on the host.
        picked = _picked(work.reshape(int(work.shape[0]), -1), (slice(None), flat.reshape(-1))).reshape(out_shape)
        if itk_blend:
            picked = _itk_cast(picked)
        return _masked_fill(picked, ~inside.unsqueeze(0), fill).type(source.dtype)

    if mode == "cubic":
        # Keys' four taps per axis, corner by corner over a flat index: grid_sample has no cubic in
        # 3-D, and the corner walk keeps the coordinates GLOBAL, so a region agrees exactly.
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

    # A BLEND goes through grid_sample, one fused kernel where the eight corners are eight gathers
    # over a flat index. `align_corners=False` IS ITK's domain and `padding_mode="border"` IS ITK's
    # tap clamp; grid_sample has no notion of the FILL, so the mask computed above still applies it.
    # It costs the bit-identity between a streamed region and the whole volume: grid_sample takes
    # NORMALISED coordinates, so it divides by the extent of the tensor handed to it, and a region is
    # handed a window. The two agree to ~1e-5 instead of exactly.
    #
    # grid_sample orders the last dimension (x, y, z), the order the coordinates already arrive in.
    # The grid counts voxels in float32 WHATEVER the payload: a half grid quantizes a coordinate at
    # ~2^-11 of the window extent, and the window is upcast with it because grid_sample takes one
    # dtype. Each axis is normalised straight into the grid it belongs in: keeping them and stacking
    # them first would hold the whole grid twice.
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


def _itk_cast(values: torch.Tensor) -> torch.Tensor:
    """ITK's cast into a float32 output (``CastPixelWithBoundsChecking``): an infinity lands on the
    largest finite value of its sign. A float64 output keeps it."""
    if values.dtype != torch.float32:
        return values
    bound = float(torch.finfo(torch.float32).max)
    return values.clamp_(-bound, bound)


def _itk_linear(
    source: torch.Tensor, coordinates_xyz: torch.Tensor, source_starts_zyx: list[int], source_shape_zyx: list[int]
) -> torch.Tensor:
    """``LinearInterpolateImageFunction``'s value at each coordinate, blended in the walk's dtype and
    returned in the sampling dtype: the base tap ``floor(c)`` held at the first voxel, then
    ``v0 + (v1 - v0) * d`` folded along x, then y, then z. An axis whose distance is not positive, or
    whose upper tap lies past the source, is not blended, as ITK's branches skip it: a NaN or an inf
    on a tap the blend gives no weight stays out of the value.

    Channel by channel over one set of corner offsets: what the blend holds beside its output does
    not grow with the channel count, which :func:`walk_rows` does not price."""
    rank = coordinates_xyz.shape[-1]
    dtype = _walk_dtype()
    window_zyx = [int(extent) for extent in source.shape[1:]]
    # Flat window offsets of the 2**rank corners, one leading dimension per axis, z first, x last.
    corners = torch.zeros((), dtype=torch.long, device=source.device)
    blends: list[tuple[torch.Tensor, torch.Tensor]] = []
    stride = 1
    for axis in range(rank):
        array_axis = rank - 1 - axis
        coordinate = coordinates_xyz[..., axis].to(dtype)
        base = torch.clamp(torch.floor(coordinate), min=0.0)
        index = base.to(torch.long)
        extent = source_shape_zyx[array_axis]
        distance = coordinate - base
        blends.append(((distance > 0) & (index + 1 < extent), distance))
        place = (extent, source_starts_zyx[array_axis], window_zyx[array_axis])
        taps = torch.stack([window_index(index, *place), window_index(index + 1, *place)]) * stride
        corners = taps.reshape(2, *([1] * axis), *taps.shape[1:]) + corners
        stride *= window_zyx[array_axis]
    flat_source = source.reshape(int(source.shape[0]), -1)
    out = torch.empty(
        (int(source.shape[0]), *coordinates_xyz.shape[:-1]), dtype=sampling_dtype(source), device=source.device
    )
    for channel in range(int(source.shape[0])):
        values = flat_source[channel][corners].to(dtype)
        for folded, (blend, distance) in enumerate(blends):
            low, high = values.unbind(rank - 1 - folded)
            values = torch.where(blend, (high - low).mul_(distance).add_(low), low)
        out[channel] = values
    return out


def source_window(
    target_grid: Grid,
    source_grid: Grid,
    bound: TransformBound,
    margin: int = 1,
) -> tuple[slice, ...]:
    """The source window a target region pulls, from the bound alone: no voxel sampled.

    Closed form, so a cost model can price a decomposition without doing the bounding work for it:
    the target region's world box, mapped through the bound (an exact affine hull grown by the
    residual), read back as a clamped index window on the source.
    """
    return source_grid.index_window(bound.map_box(target_grid.world_box()), margin)


def _face_points(target_grid: Grid, onto: Grid, stages: SpatialStages) -> np.ndarray:
    """The region's face voxels sent through the map, as continuous indices on ``onto``, ``(N, rank)``."""
    rank = target_grid.rank
    faces = []
    for axis, extent in enumerate(target_grid.size_zyx):
        for face in {0, int(extent) - 1}:
            face_grid = target_grid.sub_grid(
                tuple(
                    slice(face, face + 1) if a == axis else slice(0, int(n)) for a, n in enumerate(target_grid.size_zyx)
                )
            )
            # Each displacement is walked on the lattice window its face reaches, never the whole field.
            box = face_grid.world_box()
            local: list[AffineStage | DisplacementStage] = []
            for stage in stages:
                if isinstance(stage, DisplacementStage):
                    stage = stage.over(box)
                box = stage.bound().map_box(box)
                local.append(stage)
            faces.append(source_index(face_grid, onto, tuple(local), torch.device("cpu")).reshape(-1, rank))
    return torch.cat(faces).numpy()


def walked_window(target_grid: Grid, source_grid: Grid, stages: SpatialStages, margin: int = 1) -> tuple[slice, ...]:
    """The source window a target region pulls, from the map walked along the region's faces.

    A map that does not fold sends the region's boundary to the boundary of its image, so the
    extremes the walk meets on the faces are the extremes of the whole region. That holds for the
    fields a registration writes; a field that folds reaches past its faces, and the region then
    reads outside its window. Faces rather than :func:`source_window`'s bound: a displacement's range
    widens every region by the whole of its variation where the region itself moves by far less.
    """
    points = _face_points(target_grid, source_grid, stages)
    # A sample at continuous index c reads floor(c) - (margin - 1) to floor(c) + margin: two taps
    # linear or nearest, four cubic. A millionth of a voxel covers the sampler's own rounding of c.
    low = np.floor(points.min(axis=0) - 1e-6) - (margin - 1)
    high = np.floor(points.max(axis=0) + 1e-6) + margin
    return source_grid.continuous_window(low, high, 0)


def walked_box(target_grid: Grid, stages: SpatialStages) -> WorldBox:
    """Where the map sends the region's voxels, as a world box: :func:`walked_window`'s walk."""
    rank = target_grid.rank
    world = Grid((1,) * rank, np.zeros(rank), np.ones(rank), np.eye(rank))
    points = _face_points(target_grid, world, stages)
    return WorldBox(points.min(axis=0), points.max(axis=0))
