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

"""Where each target voxel reads from, and the gather that reads it — in torch, on the volume's device.

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

import itertools

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
#: float32 that sum keeps about four digits past the voxel -- enough to move a sample across a
#: voxel boundary on a large grid, which is a visibly different value on anything that is not
#: smooth. Measured on a 64x512x512 slab: float32 coordinates disagree with SimpleITK by 0.35 on
#: data in [0, 1], float64 by 3e-05 (which is the gather's own float32 rounding).
_COORDINATE_DTYPE = torch.float64


def _kernel_weights(offset: torch.Tensor, order: int) -> torch.Tensor:
    """The 1-D B-spline weight at distance ``offset`` — ITK's own kernels.

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


def _apply(points_xyz: torch.Tensor, affine: AffineMap, device: torch.device) -> torch.Tensor:
    """``translation + Σ_j column_j · p_j``, accumulated in ``j`` order — ITK's own association.

    Not ``points @ matrix.T + offset``. That is the same sum in a different order, which BLAS is
    free to reassociate, and the two answers differ in the last bit; a continuous index landing on
    an exact half then rounds to the other voxel. The one matmul saved is not worth a label map
    that disagrees with ``sitk.Resample`` in whole columns.
    """
    matrix = torch.tensor(affine.matrix, dtype=_COORDINATE_DTYPE, device=device)
    out = torch.tensor(affine.translation, dtype=_COORDINATE_DTYPE, device=device).expand(points_xyz.shape).clone()
    for j in range(affine.rank):
        out = out + points_xyz[..., j, None] * matrix[:, j]
    return out


def _to_index(world_xyz: torch.Tensor, grid: Grid, device: torch.device) -> torch.Tensor:
    """World to continuous index, in ITK's arithmetic: subtract the origin FIRST, then accumulate.

    ``TransformPhysicalPointToContinuousIndex`` sums ``M[i][j] * (p_j - O_j)`` from zero. Folding the
    origin into a translation instead — which is what a generic inverted affine map is — reassociates
    a difference of large world coordinates against a small one and moves the last bit.
    """
    matrix = torch.tensor(grid.world_to_index.matrix, dtype=_COORDINATE_DTYPE, device=device)
    origin = torch.tensor(grid.origin_xyz, dtype=_COORDINATE_DTYPE, device=device)
    shifted = world_xyz - origin
    out = torch.zeros_like(world_xyz)
    for j in range(grid.rank):
        out = out + shifted[..., j, None] * matrix[:, j]
    return out


def _displacement_at(stage: DisplacementStage, world_xyz: torch.Tensor, device: torch.device) -> torch.Tensor:
    """The stage's displacement at each world point — zero where its grid does not reach.

    ITK returns the identity outside a BSpline's valid region and outside a field's domain, so a
    point past the values simply moves by nothing. That is also what makes a region of a field a
    legal stand-in for the whole of it: the part it does not cover was going to be identity here
    anyway — which is exactly why the region handed in must COVER the target region, and why a
    short one degrades in silence rather than raising.
    """
    rank = stage.grid.rank
    index = _to_index(world_xyz, stage.grid, device)  # continuous index on the value grid, (x, y, z)
    values = torch.tensor(stage.values, dtype=_COORDINATE_DTYPE, device=device)
    extent_xyz = [int(stage.grid.size_zyx[rank - 1 - axis]) for axis in range(rank)]

    base = torch.floor(index) - (stage.order - 1) // 2
    taps = stage.order + 1
    # The two domains ITK actually implements, and they differ. A BSpline is the identity unless its
    # whole support lies in the coefficient grid (``InsideValidRegion``), so its edge falls off a
    # control point early. A dense field is an ordinary image: it interpolates anywhere in
    # ``[-0.5, n - 0.5)`` with the taps clamped, which reaches half a voxel past the outermost
    # samples. Using the spline's rule for a field blanks that rim -- 10% of the voxels of a small
    # volume, all of them at the border, all of them silently un-warped.
    inside = torch.ones(index.shape[:-1], dtype=torch.bool, device=device)
    for axis in range(rank):
        if stage.order == 1:
            inside &= (index[..., axis] >= -0.5) & (index[..., axis] < extent_xyz[axis] - 0.5)
        else:
            inside &= (base[..., axis] >= 0) & (base[..., axis] + taps - 1 < extent_xyz[axis])

    flat_values = values.reshape(rank, -1)
    out = torch.zeros(index.shape, dtype=_COORDINATE_DTYPE, device=device)
    for corner in itertools.product(range(taps), repeat=rank):
        weight = torch.ones(index.shape[:-1], dtype=_COORDINATE_DTYPE, device=device)
        positions_xyz = []
        for axis in range(rank):
            position = base[..., axis] + corner[axis]
            weight = weight * _kernel_weights(index[..., axis] - position, stage.order)
            positions_xyz.append(position.to(torch.long).clamp(0, extent_xyz[axis] - 1))
        # ``values`` is (component, Z, Y, X) and row-major over the spatial axes, so the flat index
        # runs the ARRAY axes outermost-first with x fastest -- the mirror of the physical order the
        # coordinates arrive in. Running it the other way samples the field transposed, which warps
        # the anatomy somewhere plausible and wrong.
        flat_index = torch.zeros(index.shape[:-1], dtype=torch.long, device=device)
        for array_axis in range(rank):
            flat_index = flat_index * int(stage.grid.size_zyx[array_axis]) + positions_xyz[rank - 1 - array_axis]
        gathered = torch.stack([flat_values[c].index_select(0, flat_index.reshape(-1)) for c in range(rank)], dim=-1)
        out = out + gathered.reshape(out.shape) * weight.unsqueeze(-1)
    return out * inside.unsqueeze(-1)


def source_index(
    target_grid: Grid,
    source_grid: Grid,
    stages: SpatialStages,
    device: torch.device,
) -> torch.Tensor:
    """One source continuous index per voxel of ``target_grid``, shaped ``(*size_zyx, rank)`` in ``(x, y, z)``.

    ``target_grid`` is the REGION's grid — its own origin, not the volume's.

    THE WORLD POINT IS MATERIALISED, never folded away. Target index to world and world to source
    index compose into one affine that is algebraically identical and one matmul cheaper, and it is
    the wrong arithmetic: ITK takes the two steps separately, so the two associations disagree in the
    last bit — and a continuous index landing EXACTLY on ``k + 0.5`` (which is what a target grid
    commensurate with its source produces, in whole columns at a time) then rounds to a different
    voxel. Measured against ``sitk.Resample``: 150 of 1050 voxels of a label map, every one of them
    a legal-looking label. Consecutive affine STAGES are still folded — that is between two world
    points, where nothing rounds to an index.
    """
    rank = target_grid.rank
    axes = [
        torch.arange(int(extent), dtype=_COORDINATE_DTYPE, device=device) for extent in reversed(target_grid.size_zyx)
    ]
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
        world = world + _displacement_at(stage, world, device)
    if not pending.is_identity:
        world = _apply(world, pending, device)
    return _to_index(world, source_grid, device)


def _is_diagonal(matrix: np.ndarray) -> bool:
    """Whether every off-diagonal entry is EXACTLY zero — no tolerance, deliberately.

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
    where it applies rather than merely close -- which is what lets one case take it and another the
    general path without the two ever having to be reconciled.

    What it saves is not arithmetic but MEMORY TRAFFIC: the general path materialises a coordinate
    per voxel and gathers eight corners through a flat index over the whole volume.
    """
    if stages:
        return None
    forward, backward = target_grid.index_to_world, source_grid.world_to_index
    if not (_is_diagonal(forward.matrix) and _is_diagonal(backward.matrix)):
        return None
    rank = target_grid.rank
    axes: list[torch.Tensor] = []
    for array_axis in range(rank):
        axis = rank - 1 - array_axis  # the physical component this array axis runs along
        index = torch.arange(int(target_grid.size_zyx[array_axis]), dtype=_COORDINATE_DTYPE, device=device)
        world = float(forward.translation[axis]) + index * float(forward.matrix[axis, axis])
        axes.append((world - float(source_grid.origin_xyz[axis])) * float(backward.matrix[axis, axis]))
    return axes


def blend_order(target_grid: Grid, source_grid: Grid) -> list[int]:
    """The array axes in the order to blend them: most source voxels per target voxel first.

    Each axis is reduced to its output extent before the next one reads it, so blending the axis
    that shrinks most FIRST makes every later pass smaller. On an isotropic change of spacing that
    is worth nothing; on a thick-slice CT brought to isotropic -- 64x512x512 at 3x0.7x0.7 mm, where
    z triples while y and x shrink -- it is 44 M intermediate elements against 110 M.

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
    """:func:`gather`'s rules over a map that factorises — one ``index_select`` per axis, no volume.

    Same interval, same tap clamp, same round-half-up, same fill. It sums the tensor product in a
    different ORDER than the general gather -- axis by axis rather than corner by corner -- so the
    two agree to float rounding rather than bit for bit. That costs nothing: a map either factorises
    or it does not, so no case is ever served by both. The equality that has to be exact is a
    streamed region against the whole volume, and it is: the per-axis coordinates are global, so a
    region takes a sub-range of the very numbers the whole volume takes.
    """
    from konfai.data.transform import nearest_index, sampling_dtype, window_index

    rank = len(axes)
    window_zyx = [int(extent) for extent in source.shape[1:]]
    extent_zyx = [int(axis.numel()) for axis in axes]
    device = source.device

    inside_axes = [(axis >= -0.5) & (axis < source_shape_zyx[array_axis] - 0.5) for array_axis, axis in enumerate(axes)]
    out_shape = [int(source.shape[0]), *extent_zyx]
    if not all(bool(mask.any()) for mask in inside_axes):
        return torch.full(out_shape, fill, device=device, dtype=torch.float32).type(source.dtype)

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

    out = source if mode == "nearest" else source.type(sampling_dtype(source))
    for array_axis in range(rank) if blend is None else blend:
        # An axis the map leaves alone is read by leaving it alone: two gathers and a blend that
        # would reproduce the input exactly, over the largest tensor in flight, for nothing.
        if is_identity(array_axis):
            continue
        axis = axes[array_axis]
        if mode == "nearest":
            out = out.index_select(array_axis + 1, local(nearest_index(axis), array_axis))
            continue
        # One axis at a time, not eight corners at once. The tensor product is the same sum, but
        # blending axis by axis reduces each extent before the next axis reads it -- six gathers over
        # shrinking tensors instead of twenty-four over the largest one.
        base = torch.floor(axis)
        share = broadcast((axis - base).to(out.dtype), array_axis)
        index = base.to(torch.long)
        low = out.index_select(array_axis + 1, local(index, array_axis))
        high = out.index_select(array_axis + 1, local(index + 1, array_axis))
        # lerp fuses the three passes `low * (1 - w) + high * w` into one. Exact at w = 0, which is
        # the only endpoint reachable: w is `x - floor(x)`, so it never reaches 1.
        out = torch.lerp(low, high, share)
    if mode == "nearest" and not out.is_floating_point():
        out = out.type(torch.float32)

    if all(bool(mask.all()) for mask in inside_axes):
        # Nothing of this region falls outside the source, which is the ordinary case for a resample
        # within a volume -- so neither the mask nor the pass that applies it is built at all.
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
    is eight gathers over a flat index, which ``grid_sample`` does as one fused kernel -- at the cost
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
        return torch.full(out_shape, fill, device=device, dtype=torch.float32).type(source.dtype)

    from konfai.data.transform import nearest_index, sampling_dtype, window_index

    work = source.type(sampling_dtype(source))
    if mode == "nearest":
        # One gather, on exact index arithmetic: a nearest pick is discontinuous, so the last bit of
        # a coordinate decides which voxel it lands on. Measured over 300 random grid pairs, a
        # single-precision coordinate moves 2.6% of the picks -- in a label map, 2.6% of the voxels
        # wearing a different label. There are no eight corners to fuse here, so nothing is bought
        # by handing it to a kernel that normalises coordinates.
        flat = torch.zeros(extent_zyx, dtype=torch.long, device=device)
        for array_axis in range(rank):
            index = nearest_index(coordinates_xyz[..., rank - 1 - array_axis])
            local = window_index(
                index, source_shape_zyx[array_axis], source_starts_zyx[array_axis], window_zyx[array_axis]
            )
            flat = flat * window_zyx[array_axis] + local
        picked = work.reshape(int(work.shape[0]), -1).index_select(1, flat.reshape(-1)).reshape(out_shape)
        return picked.masked_fill(~inside.unsqueeze(0), fill).type(source.dtype)

    # A BLEND goes through grid_sample, which is one fused kernel where the eight corners are eight
    # gathers over a flat index. `align_corners=False` IS ITK's domain -- a sample is inside while
    # its continuous index lies in [-0.5, n - 0.5) -- and `padding_mode="border"` IS ITK's tap clamp;
    # what grid_sample has no notion of is the FILL, so the mask computed above still applies it.
    #
    # It costs the bit-identity between a streamed region and the whole volume, and that is the whole
    # of what it costs: grid_sample takes NORMALISED coordinates, so it divides by the extent of the
    # tensor handed to it, and a region is handed a window. The two agree to ~1e-5 instead of
    # exactly. Deliberate, and asked for -- 4x on a warp.
    local_axes = []
    for array_axis in range(rank):
        axis = rank - 1 - array_axis
        extent = window_zyx[array_axis]
        shifted = coordinates_xyz[..., axis] - float(source_starts_zyx[array_axis])
        local_axes.append((2.0 * shifted + 1.0) / extent - 1.0)
    # grid_sample orders the last dimension (x, y, z) -- the mirror of the array axes, which is the
    # order the coordinates already arrive in.
    sampling = torch.stack([local_axes[rank - 1 - axis] for axis in range(rank)], dim=-1).to(work.dtype)
    out = torch.nn.functional.grid_sample(
        work.unsqueeze(0), sampling.unsqueeze(0), mode="bilinear", padding_mode="border", align_corners=False
    ).squeeze(0)
    return out.masked_fill(~inside.unsqueeze(0), fill).type(source.dtype)


def source_window(
    target_grid: Grid,
    source_grid: Grid,
    bound: TransformBound,
    margin: int = 1,
) -> tuple[slice, ...]:
    """The source window a target region pulls, from the bound alone — no voxel sampled.

    Closed form, which is what lets a cost model price a decomposition without doing the bounding
    work for it: the target region's world box, mapped through the bound (an exact affine hull
    grown by the residual), read back as a clamped index window on the source.
    """
    return source_grid.index_window(bound.map_box(target_grid.world_box()), margin)


def read_amplification(
    target_grid: Grid,
    source_grid: Grid,
    bound: TransformBound,
    regions: list[tuple[slice, ...]],
    margin: int = 1,
) -> float:
    """How many times the source's voxels this decomposition reads, in total.

    The honest name for what streaming costs: every region's source window is read whole, the
    windows overlap, and the finer the decomposition the more they overlap. Monotone in fineness,
    so it is a property of the decomposition and not of the transform alone.
    """
    total = 0
    for region in regions:
        window = source_window(target_grid.sub_grid(region), source_grid, bound, margin)
        total += int(np.prod([part.stop - part.start for part in window], dtype=np.int64))
    return float(total) / float(np.prod(source_grid.size_zyx, dtype=np.int64))
