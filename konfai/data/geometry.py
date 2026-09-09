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

"""Grids, boxes and affine maps in world coordinates: the value vocabulary a resample shares.

Two axis orders coexist in every KonfAI header: array data is ``(Z, Y, X)``, physical geometry
(``Origin``, ``Spacing``, ``Direction``) is ``(x, y, z)``. The types here carry the order in the
field name (``size_zyx``, ``origin_xyz``), so a mixed expression reads as wrong at the call site.

Everything is plain float64 numpy: no torch, no SimpleITK, so a value built here crosses the
``mp.spawn`` pickle boundary as data. The SimpleITK plumbing that produces it lives in
``konfai.utils.ITK`` behind its import guard.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np

from konfai.utils.errors import TransformError

if TYPE_CHECKING:
    import torch

    from konfai.utils.dataset import Attribute

#: The geometry keys a grid is read from, in the order refusals name them.
_GEOMETRY_KEYS = ("Origin", "Spacing", "Direction")


def _as_float(values: object, rank: int, what: str, count: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).ravel()
    if array.size != count:
        raise TransformError(
            f"{what} does not describe a {rank}-dimensional grid: got {array.size} value(s), expected {count}."
        )
    return array


@dataclass(frozen=True)
class AffineMap:
    """``q_xyz = matrix @ p_xyz + translation_xyz``, in world coordinates."""

    matrix: np.ndarray
    translation: np.ndarray

    @staticmethod
    def identity(rank: int) -> AffineMap:
        return AffineMap(np.eye(rank), np.zeros(rank))

    @property
    def rank(self) -> int:
        return int(self.translation.size)

    @property
    def is_identity(self) -> bool:
        return bool(np.array_equal(self.matrix, np.eye(self.rank)) and not self.translation.any())

    def apply(self, points_xyz: np.ndarray) -> np.ndarray:
        """Map points of shape ``(..., rank)``, accumulating exactly as ITK does:
        ``translation + Σ_j column_j · p_j`` with ``j`` ascending, the association of
        ``TransformIndexToPhysicalPoint``. A matmul sums in whatever order BLAS picks, one ULP away
        on an oblique grid, and one ULP of origin costs a streamed slab its bit-identity."""
        points = np.asarray(points_xyz, dtype=np.float64)
        out = np.broadcast_to(self.translation, points.shape).copy()
        for j in range(self.rank):
            out += points[..., j, np.newaxis] * self.matrix[:, j]
        return out

    def then(self, outer: AffineMap) -> AffineMap:
        """The composition ``outer(self(p))``."""
        return AffineMap(outer.matrix @ self.matrix, outer.matrix @ self.translation + outer.translation)

    def inverted(self) -> AffineMap:
        """The inverse map, or a refusal when the matrix is singular: that means a degenerate grid
        or transform, and ``pinv`` would hand back a map resampling plausibly from the wrong place."""
        try:
            inverse = np.linalg.inv(self.matrix)
        except np.linalg.LinAlgError:
            raise TransformError(
                "This affine map is singular and cannot be inverted.",
                "The grid or stored transform behind it collapses at least one axis; check its"
                " Spacing/Direction (or the transform's matrix) for a zero row.",
            ) from None
        return AffineMap(inverse, -inverse @ self.translation)


@dataclass(frozen=True)
class WorldBox:
    """An axis-aligned box in world coordinates, ``low_xyz`` to ``high_xyz`` inclusive."""

    low_xyz: np.ndarray
    high_xyz: np.ndarray

    def grown(self, radius_xyz: np.ndarray | float) -> WorldBox:
        radius = np.broadcast_to(np.asarray(radius_xyz, dtype=np.float64), self.low_xyz.shape)
        return WorldBox(self.low_xyz - radius, self.high_xyz + radius)

    def extended(self, low_xyz: np.ndarray, high_xyz: np.ndarray) -> WorldBox:
        """This box plus a per-component interval: each end moved by its own end of it. The
        asymmetric form of :meth:`grown`, and the one a signed displacement bound needs. An interval
        that does not straddle zero MOVES the box instead of widening it."""
        return WorldBox(
            self.low_xyz + np.asarray(low_xyz, dtype=np.float64),
            self.high_xyz + np.asarray(high_xyz, dtype=np.float64),
        )

    def image_under(self, affine: AffineMap) -> WorldBox:
        """The axis-aligned hull of this box's image under ``affine``, from centre and half-extents:
        the image of centre ``c`` is ``A c + b``, and the largest reach of ``A h`` over the corners is
        ``|A| h``. Equal to the hull of the ``2^rank`` mapped corners, in O(rank²)."""
        centre = (self.low_xyz + self.high_xyz) / 2.0
        half = (self.high_xyz - self.low_xyz) / 2.0
        mapped = affine.apply(centre)
        reach = np.abs(affine.matrix) @ half
        return WorldBox(mapped - reach, mapped + reach)


#: How close to a whole number a voxel count must be before it is taken to BE that number. A spacing
#: of 0.7 mm is not representable in binary, so 90 voxels of it re-cut at 1.5 mm come to
#: 41.999999999999997 and would truncate to 41. The band is narrower than any density a header states.
_COUNT_TOLERANCE = 1e-6


def _voxel_count(extent: int, spacing: float, wanted: float) -> int:
    """How many voxels of ``wanted`` size cover ``extent`` voxels of ``spacing``: truncated."""
    return int(np.floor(extent * spacing / wanted + _COUNT_TOLERANCE))


@dataclass(frozen=True)
class Grid:
    """A stored volume's sampling grid: extent in array order, geometry in physical order."""

    size_zyx: tuple[int, ...]
    origin_xyz: np.ndarray
    spacing_xyz: np.ndarray
    direction_xyz: np.ndarray

    @classmethod
    def identity(cls, spatial_shape: list[int]) -> Grid:
        """The grid of a volume with no geometry: unit spacing, origin zero, axes as stored, which is
        what a header carrying no ``Origin``/``Spacing``/``Direction`` means. Under it a world
        coordinate IS an index, so a resample onto another grid degenerates to the size ratio."""
        rank = len(spatial_shape)
        return cls(tuple(int(extent) for extent in spatial_shape), np.zeros(rank), np.ones(rank), np.eye(rank))

    @classmethod
    def from_header(cls, spatial_shape: list[int], attribute: Attribute, what: str) -> tuple[Grid, frozenset[str]]:
        """The grid a header describes AND which of its keys it did not say, never a refusal. The
        identity stands in for what is absent and the caller is told what it stood in for: an extent
        change needs no key, a density change the ``Spacing`` alone, a reference grid all three."""
        rank = len(spatial_shape)
        identity = cls.identity(spatial_shape)
        missing = frozenset(key for key in _GEOMETRY_KEYS if key not in attribute)
        origin = (
            identity.origin_xyz
            if "Origin" in missing
            else _as_float(attribute.get_np_array("Origin"), rank, f"the Origin of {what}", rank)
        )
        spacing = (
            identity.spacing_xyz
            if "Spacing" in missing
            else _as_float(attribute.get_np_array("Spacing"), rank, f"the Spacing of {what}", rank)
        )
        direction = (
            identity.direction_xyz
            if "Direction" in missing
            else _as_float(attribute.get_np_array("Direction"), rank, f"the Direction of {what}", rank * rank).reshape(
                rank, rank
            )
        )
        if not np.all(spacing > 0.0):
            raise TransformError(
                f"The Spacing of {what} is {spacing.tolist()}.",
                "A spacing is a physical extent per voxel and must be positive on every axis.",
            )
        return cls(identity.size_zyx, origin, spacing, direction), missing

    @classmethod
    def of(cls, spatial_shape: list[int], attribute: Attribute, what: str) -> Grid:
        """The grid a header describes, or a refusal naming ``what`` and the missing key."""
        grid, missing = cls.from_header(spatial_shape, attribute, what)
        if missing:
            raise TransformError(
                f"The geometry of {what} is needed and its header carries no {', '.join(sorted(missing))}.",
                "Resampling onto another grid happens in physical space: without an origin, a"
                " spacing and a direction there is no space to do it in. Use a source whose"
                " geometry is readable (mha, nii, h5, or an OME-Zarr written by KonfAI).",
            )
        return grid

    @staticmethod
    def readable(attribute: Attribute) -> bool:
        """Whether a header carries a full geometry: total, read-only, no I/O."""
        return all(key in attribute for key in _GEOMETRY_KEYS)

    @property
    def rank(self) -> int:
        return len(self.size_zyx)

    @cached_property
    def index_to_world(self) -> AffineMap:
        """Continuous index ``(x, y, z)`` to world: ``p = O + D S i``, ITK's own association. Cached
        (the grid is frozen): a streamed walk asks per SLAB, and rebuilding the product, and for
        :attr:`world_to_index` re-INVERTING it, thousands of times per region was measurable."""
        return AffineMap(self.direction_xyz @ np.diag(self.spacing_xyz), np.asarray(self.origin_xyz, dtype=np.float64))

    @cached_property
    def world_to_index(self) -> AffineMap:
        return self.index_to_world.inverted()

    def _index_box(self, region_zyx: tuple[slice, ...] | None) -> tuple[np.ndarray, np.ndarray]:
        """The region's outer faces as continuous indices ``(x, y, z)``: ``start - 0.5 .. stop - 0.5``.
        Outer faces and not voxel centres: a sample is inside a grid while its continuous index lies
        in ``[-0.5, n - 0.5)``, so a bound built on centres is short by that half voxel."""
        if region_zyx is None:
            region_zyx = tuple(slice(0, extent) for extent in self.size_zyx)
        low = np.array([float(part.start) - 0.5 for part in reversed(region_zyx)])
        high = np.array([float(part.stop) - 0.5 for part in reversed(region_zyx)])
        return low, high

    def world_box(self, region_zyx: tuple[slice, ...] | None = None) -> WorldBox:
        """The axis-aligned world hull of a region's outer faces (the whole grid when ``None``)."""
        low, high = self._index_box(region_zyx)
        return WorldBox(low, high).image_under(self.index_to_world)

    def continuous_box(self, box: WorldBox) -> tuple[np.ndarray, np.ndarray]:
        """A world box as a continuous-index box ``(low_xyz, high_xyz)`` on this grid."""
        image = box.image_under(self.world_to_index)
        return image.low_xyz, image.high_xyz

    def index_window(self, box: WorldBox, margin: int) -> tuple[slice, ...]:
        """The clamped array-order window a world box needs, grown by ``margin`` whole voxels:
        ``floor``/``ceil`` on the continuous-index box plus the margin the interpolation taps reach,
        clamped to a non-empty window exactly as ``Resample._offset_window`` clamps."""
        low, high = self.continuous_box(box)
        window: list[slice] = []
        for axis in range(self.rank - 1, -1, -1):
            start = int(np.floor(low[axis])) - margin
            stop = int(np.ceil(high[axis])) + 1 + margin
            extent = int(self.size_zyx[self.rank - 1 - axis])
            start = min(max(start, 0), extent - 1)
            window.append(slice(start, min(max(stop, start + 1), extent)))
        return tuple(window)

    def resampled(
        self,
        spacing_xyz: np.ndarray | None = None,
        size_zyx: tuple[int, ...] | None = None,
        align: str = "extent",
    ) -> Grid:
        """The same anatomy at another sampling density: give a spacing, or give a count.

        A component left at zero keeps that axis as it is. Whichever is given, the other follows,
        and ``align`` decides where the new grid SITS:

        - ``extent``: the outer faces coincide, so both grids cover exactly the same box and a
          target index reads ``scale * (i + 0.5) - 0.5`` of the source. What ``F.interpolate`` does.
        - ``origin``: voxel zero's CENTRE stays put, so a target index reads ``scale * i`` and the
          far edge moves by whatever the count rounded away.

        Under ``extent`` the spacing is derived from the counts and not from the request: a count is
        a whole number, so the density that actually covers the box is ``n_src / n_dst`` times the
        source's, and recording the requested one describes a grid nobody sampled.
        """
        if (spacing_xyz is None) == (size_zyx is None):
            raise TransformError("A resampled grid is defined by a spacing or by a count, and by exactly one of them.")
        if align not in ("extent", "origin"):
            raise TransformError(
                f"'{align}' is not a way to place a resampled grid.",
                "Use align: extent to keep the field of view (the outer faces coincide) or align:"
                " origin to keep voxel zero's centre where it is.",
            )
        requested_xyz: np.ndarray | None = None
        if size_zyx is not None:
            counts = [
                int(want) if int(want) > 0 else int(have) for want, have in zip(size_zyx, self.size_zyx, strict=True)
            ]
        else:
            wanted_xyz = np.asarray(spacing_xyz, dtype=np.float64)
            requested_xyz = np.where(wanted_xyz > 0.0, wanted_xyz, self.spacing_xyz)
            counts = [
                _voxel_count(have, source, wanted)
                for have, source, wanted in zip(self.size_zyx, self.spacing_xyz[::-1], requested_xyz[::-1], strict=True)
            ]
        size = tuple(max(1, count) for count in counts)
        # The density that actually covers the source's box with `size` voxels.
        covering_xyz = self.spacing_xyz * np.array(self.size_zyx[::-1], dtype=np.float64) / np.array(size[::-1])
        if align == "origin":
            return Grid(
                size, self.origin_xyz, covering_xyz if requested_xyz is None else requested_xyz, self.direction_xyz
            )
        return Grid(
            size,
            self.origin_xyz + 0.5 * (self.direction_xyz @ (covering_xyz - self.spacing_xyz)),
            covering_xyz,
            self.direction_xyz,
        )

    def sub_grid(self, region_zyx: tuple[slice, ...]) -> Grid:
        """The grid of a region: same spacing and direction, the origin of its first voxel. A region
        left at the volume's origin replays the volume's first slab wherever it lands. The origin is
        ``index_to_world`` of the region's start, one application of the parent's own map."""
        start_xyz = np.array([float(part.start) for part in reversed(region_zyx)])
        return Grid(
            tuple(int(part.stop - part.start) for part in region_zyx),
            self.index_to_world.apply(start_xyz),
            self.spacing_xyz,
            self.direction_xyz,
        )


# ---------------------------------------------------------------------------------- index remaps
# A signed permutation of the axes is the one map that is an exact index remap: values only change
# place, so an ORIENTATION stage's bijection promise (and everything preserves_statistics lets a
# later stage trust) rests on the predicate below. It is written once, here.

#: Whether a matrix entry stands for exactly 0 or +/-1, by the matrix's provenance. Two tolerances:
#: a draw's quarter-turn affine is composed from float32 cosines and lands within ~1e-7 of the value
#: it stands for; a header reorientation is a float64 product and lands within a few double ulps.
#: One shared 1e-6 would remap a float64 direction whose obliqueness is real, not rounding.
SIGNED_PERMUTATION_ATOL_FLOAT32 = 1e-6
SIGNED_PERMUTATION_ATOL_FLOAT64 = 1e-9

#: Per output SPATIAL axis in array order: ``(source_axis, mirrored)``, which source axis it reads
#: and whether it reads it backwards.
AxisRemap = list[tuple[int, bool]]


def signed_permutation(matrix: object, atol: float) -> AxisRemap | None:
    """The exact index remap ``matrix`` is, or ``None`` where it must be sampled.

    ``matrix`` maps an output coordinate onto the input it comes from, in physical ``(x, y, z)``
    order: it is a signed permutation exactly when every column holds a single +/-1 and every row
    carries unit weight, which is what the three tests together admit. The answer is in array order,
    where physical axis ``k`` is array axis ``n - 1 - k``.
    """
    linear = np.asarray(matrix, dtype=np.float64)
    n = int(linear.shape[0])
    magnitude = np.abs(linear)
    unit = np.ones(n)
    # rtol=0: the default 1e-5 relative slack against unit targets would swamp atol and let a
    # near-axis rotation with off-axis terms around 5e-6 pass as a permutation.
    if not np.allclose(magnitude.sum(axis=0), unit, rtol=0.0, atol=atol):
        return None
    if not np.allclose(magnitude.max(axis=0), unit, rtol=0.0, atol=atol):
        return None
    if not np.allclose(magnitude.sum(axis=1), unit, rtol=0.0, atol=atol):
        return None
    remap: AxisRemap = []
    for column in reversed(range(n)):
        row = int(magnitude[:, column].argmax())
        remap.append((n - 1 - row, bool(linear[row, column] < 0)))
    return remap


def remap_shape(shape: list[int], remap: AxisRemap) -> list[int]:
    """The spatial extents a remap lands on: output axis ``k`` carries the extent of the axis it
    reads. What a permutation preserves is the volume, not which axis holds an extent."""
    return [int(shape[source]) for source, _ in remap]


def remap_region(target_slices: tuple[slice, ...], source_shape: list[int], remap: AxisRemap) -> list[slice]:
    """The source region a target region reads under a remap. Output axis ``k``'s slice lands on the
    source axis it reads; a MIRRORED axis reads the mirror region ``[n - stop, n - start)`` of it.
    The remap covers every axis exactly once, so every source axis is assigned."""
    source_slices: list[slice] = [slice(0, int(extent)) for extent in source_shape]
    for target, (source, mirrored) in zip(target_slices, remap, strict=True):
        extent = int(source_shape[source])
        source_slices[source] = (
            slice(extent - target.stop, extent - target.start) if mirrored else slice(target.start, target.stop)
        )
    return source_slices


def invert_remap(remap: AxisRemap) -> AxisRemap:
    """The remap undoing ``remap``: source axis ``s`` reads back the output axis that carried it,
    mirrored exactly where the forward read was."""
    inverted: AxisRemap = [(0, False)] * len(remap)
    for axis, (source, mirrored) in enumerate(remap):
        inverted[source] = (axis, mirrored)
    return inverted


def apply_remap(tensor: torch.Tensor, remap: AxisRemap) -> torch.Tensor:
    """The remap, materialised on a tensor whose trailing axes are the spatial ones. Leading axes are
    left in place. ``flip`` materialises the permuted view even for an empty mirror list, so the
    result never aliases the tensor it was read from."""
    offset = tensor.dim() - len(remap)
    dims = list(range(offset)) + [offset + source for source, _ in remap]
    flips = [offset + axis for axis, (_, mirrored) in enumerate(remap) if mirrored]
    return tensor.permute(dims).flip(flips)


@dataclass(frozen=True)
class TransformBound:
    """What a stored transform is guaranteed to do: an exact affine part and a bounded interval.

    ``T(p)`` lies in ``affine(p) + [low_xyz, high_xyz]`` for every ``p``, per world component. For a
    linear transform the interval is empty and the statement is exact; for a BSpline it is the range
    of the coefficients; for a dense field the range of its values. The affine part is read
    structurally off the transform, never probed: a probe under-bounds.

    SIGNED, NOT A RADIUS. A displacement field solved between two frames carries the offset between
    them in its values, and an interval that does not straddle zero MOVES a region's window rather
    than widening it. The same two reductions produce either
    (:attr:`DisplacementStage.range_xyz`), so the tighter one is free.
    """

    affine: AffineMap
    low_xyz: np.ndarray
    high_xyz: np.ndarray

    @staticmethod
    def exact(affine: AffineMap) -> TransformBound:
        return TransformBound(affine, np.zeros(affine.rank), np.zeros(affine.rank))

    @staticmethod
    def interval(low_xyz: np.ndarray, high_xyz: np.ndarray) -> TransformBound:
        """A pure displacement whose value lies in ``[low_xyz, high_xyz]`` per component."""
        low = np.asarray(low_xyz, dtype=np.float64)
        return TransformBound(AffineMap.identity(int(low.size)), low, np.asarray(high_xyz, dtype=np.float64))

    @staticmethod
    def shift(residual_xyz: np.ndarray) -> TransformBound:
        """A pure displacement bounded in magnitude only, ``± residual_xyz``, for a caller that knows
        a radius and not a range. Anything that can state both ends should say so with
        :meth:`interval`: this one is twice as wide wherever the range is one-sided."""
        radius = np.asarray(residual_xyz, dtype=np.float64)
        return TransformBound.interval(-radius, radius)

    @property
    def residual_xyz(self) -> np.ndarray:
        """The symmetric envelope of the interval, for a caller that wants one number per axis."""
        return np.maximum(np.abs(self.low_xyz), np.abs(self.high_xyz))

    def after(self, inner: TransformBound) -> TransformBound:
        """The bound of ``self(inner(p))``: interval arithmetic through the outer affine. NOT
        ``|A| @ residual``, which is right only for an interval centred on zero: a negative entry of
        ``A`` sends the inner interval's low end to the outer's high. Splitting the matrix into its
        non-negative and non-positive parts holds either way and reduces to ``|A| @ r`` when
        ``low = -high``."""
        matrix = self.affine.matrix
        rise, fall = np.maximum(matrix, 0.0), np.minimum(matrix, 0.0)
        return TransformBound(
            inner.affine.then(self.affine),
            rise @ inner.low_xyz + fall @ inner.high_xyz + self.low_xyz,
            rise @ inner.high_xyz + fall @ inner.low_xyz + self.high_xyz,
        )

    def map_box(self, box: WorldBox) -> WorldBox:
        """Where the image of ``box`` is guaranteed to lie."""
        return box.image_under(self.affine).extended(self.low_xyz, self.high_xyz)


@dataclass(frozen=True)
class AffineStage:
    """One affine step of a decoded transform: exact, in world coordinates."""

    map: AffineMap

    def bound(self) -> TransformBound:
        return TransformBound.exact(self.map)


#: The B-spline orders KonfAI evaluates: the linear hat a dense field is read through, and the cubic
#: ITK writes a BSplineTransform with. ITK also writes orders 0 and 2, which have no kernel here, so
#: the refusal belongs where the value is built, not mid-run and per region where it is sampled.
SUPPORTED_SPLINE_ORDERS = (1, 3)


@dataclass(frozen=True)
class DisplacementStage:
    """One displacement step: ``p + d(p)``, with ``d`` interpolated off a value grid.

    One shape for the two non-linear things a stored transform can be: a BSpline is order-3
    coefficients on a coarse control grid, a dense field order-1 samples on its own grid. Both
    kernels are non-negative and sum to one, so ``sup |values|`` per component bounds the
    displacement at every point, which is what replaces walking a region's boundary.

    ``values`` is ``(rank, Z, Y, X)`` float64, components in physical ``(x, y, z)`` order, world
    units. ITK applies no direction matrix to them. Outside the grid's reach the displacement is zero.
    """

    grid: Grid
    values: np.ndarray
    order: int
    #: The values as a tensor, per (device, dtype), built on first use and living exactly as long as
    #: the stage does. Excluded from equality, hashing and pickling.
    _tensors: dict = field(default_factory=dict, init=False, repr=False, compare=False, hash=False)

    def __post_init__(self) -> None:
        if self.order not in SUPPORTED_SPLINE_ORDERS:
            raise TransformError(
                f"A displacement of B-spline order {self.order} is not one KonfAI evaluates"
                f" (orders {', '.join(str(order) for order in SUPPORTED_SPLINE_ORDERS)}).",
                "Write the transform as a displacement field, or as a cubic BSpline, which is what"
                " every registration that produces one writes by default.",
            )

    def tensor(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """``values`` on ``device`` in ``dtype``, uploaded once per (device, dtype) for the stage's life."""
        import torch

        key = (str(device), str(dtype), "czyx")
        cached = self._tensors.get(key)
        if cached is None:
            cached = self._tensors[key] = torch.tensor(self.values, dtype=dtype, device=device)
        return cached

    def tensor_by_voxel(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """``values`` as ``(voxel, component)``, contiguous: the layout a per-voxel gather wants."""
        key = (str(device), str(dtype), "vc")
        cached = self._tensors.get(key)
        if cached is None:
            rank = int(self.values.shape[0])
            cached = self._tensors[key] = self.tensor(device, dtype).reshape(rank, -1).transpose(0, 1).contiguous()
        return cached

    def __getstate__(self) -> dict:
        state = dict(self.__dict__)
        state["_tensors"] = {}
        state.pop("range_xyz", None)
        state.pop("bound_xyz", None)
        return state

    @cached_property
    def range_xyz(self) -> tuple[np.ndarray, np.ndarray]:
        """``(min, max)`` per component: one pass over the field, kept for the stage's life.

        Every pull map asks for it (per patch, per plan block, per pushed slab), so it is kept.
        ``cached_property`` writes through ``__dict__``, which a frozen dataclass allows; the entry
        is dropped from the pickle beside ``_tensors``. Two reductions and no temporary:
        ``np.abs(...).max()`` would first write a values-sized copy and then walk it again.
        """
        flat = self.values.reshape(self.values.shape[0], -1)
        return flat.min(axis=1), flat.max(axis=1)

    @cached_property
    def bound_xyz(self) -> np.ndarray:
        """``sup |values|`` per component, the range's symmetric envelope."""
        low, high = self.range_xyz
        return np.maximum(np.abs(low), np.abs(high))

    def bound(self) -> TransformBound:
        # Clamped to include zero: the stage applies NO displacement outside its grid, so a target
        # region past the field's edge still needs its identity-mapped samples.
        low, high = self.range_xyz
        return TransformBound.interval(np.minimum(low, 0.0), np.maximum(high, 0.0))


#: A decoded stored transform: stages in APPLICATION order (first applied first). SimpleITK's
#: ``CompositeTransform`` applies its list in reverse: the decoder normalizes that here, once.
SpatialStages = tuple["AffineStage | DisplacementStage", ...]


def bound_of(stages: SpatialStages, rank: int) -> TransformBound:
    """The bound of the whole decoded map, folded in application order."""
    folded = TransformBound.exact(AffineMap.identity(rank))
    for stage in stages:
        folded = stage.bound().after(folded)
    return folded
