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

Nothing here calls SimpleITK. A BSpline is evaluated from its coefficient grid and a dense field
from its samples, with the same tensor-product kernel arithmetic ITK uses.
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


def _apply(points_xyz: torch.Tensor, affine: AffineMap, device: torch.device) -> torch.Tensor:
    """``translation + Σ_j column_j · p_j``, accumulated in ``j`` order: ITK's own association. Not
    ``points @ matrix.T + offset``, which BLAS is free to reassociate; the two answers differ in the
    last bit and a continuous index landing on an exact half rounds to the other voxel."""
    matrix = torch.tensor(affine.matrix, dtype=_walk_dtype(), device=device)
    out = torch.tensor(affine.translation, dtype=_walk_dtype(), device=device).expand(points_xyz.shape).clone()
    for j in range(affine.rank):
        out = out + points_xyz[..., j, None] * matrix[:, j]
    return out


def _to_index(world_xyz: torch.Tensor, grid: Grid, device: torch.device) -> torch.Tensor:
    """World to continuous index, in ITK's arithmetic: subtract the origin FIRST, then accumulate
    ``M[i][j] * (p_j - O_j)`` from zero, as ``TransformPhysicalPointToContinuousIndex`` does. Folding
    the origin into a translation instead moves the last bit."""
    matrix = torch.tensor(grid.world_to_index.matrix, dtype=_walk_dtype(), device=device)
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
) -> list[torch.Tensor] | None:
    """One source index per target ROW of each array axis, or ``None`` when the map does not factorise.

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
        index = torch.arange(int(target_grid.size_zyx[array_axis]), dtype=_walk_dtype(), device=device)
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
        return tensor[taps_index]

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
    past the outermost voxel centres reproduces the border value; outside, the sample is ``fill``.

    ``source`` covers ``source_starts_zyx`` onward of a volume of ``source_shape_zyx``; the
    coordinates are GLOBAL indices of that volume.

    Nearest is one gather on the exact index, the pick being discontinuous. Linear goes through
    ``grid_sample``, which normalises by the extent it is handed, so a streamed region and the whole
    volume agree to ~1e-5 rather than exactly.
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
        # In the WORKING dtype then cast, exactly as the masked path fills: filling in float32 first
        # quantizes a float64 fill.
        return torch.full(out_shape, fill, device=device, dtype=sampling_dtype(source)).type(source.dtype)

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
        picked = work.reshape(int(work.shape[0]), -1)[:, flat.reshape(-1)].reshape(out_shape)
        return picked.masked_fill(~inside.unsqueeze(0), fill).type(source.dtype)

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
