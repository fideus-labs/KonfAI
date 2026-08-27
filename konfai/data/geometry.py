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

Two axis orders coexist in every KonfAI header, and the classic failure is a confusion between
them: array data is ``(Z, Y, X)``, physical geometry (``Origin``,
``Spacing``, ``Direction``) is ``(x, y, z)``. The types here carry the order in the field name (``size_zyx``,
``origin_xyz``), so a mixed expression reads as wrong at the call site instead of
resampling perfectly onto the wrong place.

Everything is plain float64 numpy: no torch, no SimpleITK. A value built here crosses the
``mp.spawn`` pickle boundary as data, and the SimpleITK plumbing that produces it lives in
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
        """Map points of shape ``(..., rank)``, accumulating exactly as ITK does.

        ``translation + Σ_j column_j · p_j`` with ``j`` ascending: the association of
        ``TransformIndexToPhysicalPoint``. A matmul sums in whatever order BLAS picks, which is
        one ULP away on an oblique grid, and one ULP of origin is the difference between a
        streamed slab that is bit-identical to the whole volume and one that is merely close.
        """
        points = np.asarray(points_xyz, dtype=np.float64)
        out = np.broadcast_to(self.translation, points.shape).copy()
        for j in range(self.rank):
            out += points[..., j, np.newaxis] * self.matrix[:, j]
        return out

    def then(self, outer: AffineMap) -> AffineMap:
        """The composition ``outer(self(p))``."""
        return AffineMap(outer.matrix @ self.matrix, outer.matrix @ self.translation + outer.translation)

    def inverted(self) -> AffineMap:
        """The inverse map, or a refusal when the matrix is singular.

        Refused rather than returned as garbage: a singular matrix here means a degenerate grid or
        transform, and ``pinv`` would hand back a map that resamples plausibly from the wrong place.
        """
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
        """This box plus a per-component interval: each end moved by its own end of it.

        The asymmetric form of :meth:`grown`, and the one a signed displacement bound needs. An
        interval that does not straddle zero MOVES the box instead of widening it, which is the
        difference between a field's reach and twice its largest value.
        """
        return WorldBox(
            self.low_xyz + np.asarray(low_xyz, dtype=np.float64),
            self.high_xyz + np.asarray(high_xyz, dtype=np.float64),
        )

    def image_under(self, affine: AffineMap) -> WorldBox:
        """The axis-aligned hull of this box's image under ``affine``.

        Centre and half-extents: the image of centre ``c`` is ``A c + b``, and the largest reach of
        ``A h`` over the corners is ``|A| h`` because each corner coordinate is ``±h_k``. Equal to
        the hull of the ``2^rank`` mapped corners (pinned by a test against that enumeration) in
        O(rank²) and with no corner loop to get wrong.
        """
        centre = (self.low_xyz + self.high_xyz) / 2.0
        half = (self.high_xyz - self.low_xyz) / 2.0
        mapped = affine.apply(centre)
        reach = np.abs(affine.matrix) @ half
        return WorldBox(mapped - reach, mapped + reach)


#: How close to a whole number a voxel count must be before it is taken to BE that number. A
#: spacing of 0.7 mm is not representable in binary, so 90 voxels of it re-cut at 1.5 mm come to
#: 41.999999999999997 and truncate to 41, one slice of anatomy dropped, and a spacing recorded
#: that no longer covers what was read. The band is far narrower than any density a header states.
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
        """The grid of a volume with no geometry: unit spacing, origin zero, axes as stored.

        What a header carrying no ``Origin``/``Spacing``/``Direction`` means when the question is
        only a change of extent. Under it a world coordinate IS an index, so a resample onto another
        grid degenerates to the size ratio it always was, which is how one engine serves a case
        whose geometry is known and one whose is not, instead of a physical path and a ratio path
        that have to be kept agreeing.
        """
        rank = len(spatial_shape)
        return cls(tuple(int(extent) for extent in spatial_shape), np.zeros(rank), np.ones(rank), np.eye(rank))

    @classmethod
    def from_header(cls, spatial_shape: list[int], attribute: Attribute, what: str) -> tuple[Grid, frozenset[str]]:
        """The grid a header describes AND which of its keys it did not say, never a refusal.

        The identity stands in for what is absent, and the caller is told what it stood in for: what
        a missing key costs depends on the question. An extent change needs none of them; a density
        change needs the ``Spacing`` and nothing else; a reference grid or a stored map needs a real
        physical space and so needs all three. Deciding that here, once, for every caller would
        either refuse a resample that was answerable or answer one that was not.
        """
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
        """Continuous index ``(x, y, z)`` to world: ``p = O + D S i``: ITK's own association.

        Cached (the grid is frozen): a streamed walk asks per SLAB, and rebuilding the product --
        and, for :attr:`world_to_index`, re-INVERTING it -- thousands of times per region was
        measurable wall time. The cache returns the same floats it always returned, only once.
        """
        return AffineMap(self.direction_xyz @ np.diag(self.spacing_xyz), np.asarray(self.origin_xyz, dtype=np.float64))

    @cached_property
    def world_to_index(self) -> AffineMap:
        return self.index_to_world.inverted()

    def _index_box(self, region_zyx: tuple[slice, ...] | None) -> tuple[np.ndarray, np.ndarray]:
        """The region's outer faces as continuous indices ``(x, y, z)``: ``start - 0.5 .. stop - 0.5``.

        Outer faces and not voxel centres, deliberately: a sample is inside a grid while its
        continuous index lies in ``[-0.5, n - 0.5)`` (the sampler rule of
        ``Resample._resample_offset_region``), so a bound built on centres is short by the half
        voxel that rule reaches.
        """
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
        """The clamped array-order window a world box needs, grown by ``margin`` whole voxels.

        ``floor``/``ceil`` on the continuous-index box, plus the margin the interpolation taps
        reach; clamped to a non-empty window, exactly as ``Resample._offset_window`` clamps: a region entirely
        off the grid is a real place for a regrid (every sample takes the
        fill) and a zero-width read is not something every backend serves.
        """
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
        and ``align`` decides where the new grid SITS: the one real choice here, worth a quarter
        of a voxel of anatomy, and made silently by every library that offers only one of them:

        - ``extent``: the outer faces coincide, so both grids cover exactly the same box and a
          target index reads ``scale * (i + 0.5) - 0.5`` of the source. What ``F.interpolate`` does,
          and what KonfAI has always done.
        - ``origin``: voxel zero's CENTRE stays put, so a target index reads ``scale * i`` and the
          far edge moves by whatever the count rounded away. What resampling onto a grid that
          shares an origin does.

        Under ``extent`` the spacing is derived from the counts and not from the request: a count is
        a whole number, so the density that actually covers the box is ``n_src / n_dst`` times the
        source's, and recording the requested one instead is a header that describes a grid nobody
        sampled (up to a millimetre of drift across a volume, measured).
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
        """The grid of a region: same spacing and direction, the origin of its first voxel.

        The load-bearing line of every streamed regrid: a region left at the volume's origin
        replays the volume's first slab wherever it lands, and the output still looks like an
        image. The origin is ``index_to_world`` of the region's start, one application of the
        parent's own map, never a second association that could land a slab origin apart from it.
        """
        start_xyz = np.array([float(part.start) for part in reversed(region_zyx)])
        return Grid(
            tuple(int(part.stop - part.start) for part in region_zyx),
            self.index_to_world.apply(start_xyz),
            self.spacing_xyz,
            self.direction_xyz,
        )


@dataclass(frozen=True)
class TransformBound:
    """What a stored transform is guaranteed to do: an exact affine part and a bounded interval.

    ``T(p)`` lies in ``affine(p) + [low_xyz, high_xyz]`` for every ``p``, per world component. For a
    linear transform the interval is empty and the statement is exact; for a BSpline it is the range
    of the coefficients (non-negative basis functions summing to one make every displacement a
    convex combination of them, so it lies between their smallest and largest); for a dense field it
    is the range of its values. The affine part is read structurally off the transform, never
    probed: a probe measures a local gradient and extrapolates it, which under-bounds (measured).

    SIGNED, NOT A RADIUS. A displacement field solved between two frames carries the offset between
    them in its values, and an interval that does not straddle zero MOVES a region's window rather
    than widening it. Measured on an ExaSPIM field whose z component runs [-28.1, -22.2] mm on a
    volume 20.6 mm thick: as a radius it reaches 28.1 mm either way, so every region pulls the whole
    volume and the fold refuses (23.57 GiB held against a 19.01 GiB budget); as an interval it
    reaches 5.9 mm, and a 24-row region pulls 175 source rows of 514. The same two reductions
    produce either (:attr:`DisplacementStage.range_xyz`), so the tighter one is free.
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
        """A pure displacement bounded in magnitude only, ``± residual_xyz``.

        For a caller that knows a radius and not a range. Anything that can state both ends should
        say so with :meth:`interval`: this one is twice as wide wherever the range is one-sided.
        """
        radius = np.asarray(residual_xyz, dtype=np.float64)
        return TransformBound.interval(-radius, radius)

    @property
    def residual_xyz(self) -> np.ndarray:
        """The symmetric envelope of the interval, for a caller that wants one number per axis."""
        return np.maximum(np.abs(self.low_xyz), np.abs(self.high_xyz))

    def after(self, inner: TransformBound) -> TransformBound:
        """The bound of ``self(inner(p))``: interval arithmetic through the outer affine.

        NOT ``|A| @ residual``. That is right for an interval centred on zero and wrong for one that
        is not: a negative entry of ``A`` sends the inner interval's low end to the outer's high,
        and taking absolute values first loses which end went where -- so a rotation folded onto a
        one-sided field would be bounded by a box that does not contain it. Splitting the matrix
        into its non-negative and non-positive parts is the same arithmetic written to hold either
        way, and it reduces to ``|A| @ r`` when ``low = -high``.
        """
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
#: ITK writes a BSplineTransform with. ITK will happily write orders 0 and 2, which decode as
#: readily as any other and have no kernel here, so the refusal belongs where the value is built,
#: not where it is finally sampled, which is mid-run and per region.
SUPPORTED_SPLINE_ORDERS = (1, 3)


@dataclass(frozen=True)
class DisplacementStage:
    """One displacement step: ``p + d(p)``, with ``d`` interpolated off a value grid.

    One shape for the two non-linear things a stored transform can be. A BSpline is order-3
    coefficients on a coarse control grid; a dense field is order-1 samples on its own grid. Both
    kernels are non-negative and sum to one, so the displacement anywhere is a convex combination
    of ``values`` and ``sup |values|`` per component bounds it at every point: the bound that
    replaces walking a region's boundary, which under-bounds a wiggle narrower than the region
    and costs more than the resample it serves (both measured).

    ``values`` is ``(rank, Z, Y, X)`` float64, components in physical ``(x, y, z)`` order, world
    units. ITK applies no direction matrix to them (verified in ``itkBSplineTransform.hxx``).
    Outside the grid's reach the displacement is zero: ITK returns the identity there.
    """

    grid: Grid
    values: np.ndarray
    order: int
    #: The values as a tensor, per (device, dtype), built on first use and living exactly as long
    #: as the stage does: a stage is evaluated once per region per case, and re-uploading a whole
    #: field for each was the copy the arithmetic never saw. Frozen dataclass, so a mutable field
    #: hides behind object.__setattr__ once; it is excluded from equality and pickling by name.
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

        Every pull map asks for it (per patch, per plan block, per pushed slab), so it is kept:
        a thousand patches recomputing one constant 3-vector cost 73 s on a 3x160x256x256 field.
        ``cached_property`` writes through ``__dict__``, which a frozen dataclass allows; the entry
        is dropped from the pickle beside ``_tensors``.

        Two reductions and no temporary, which is what the pair costs and what its magnitude cost
        before it: ``np.abs(...).max()`` would first write a values-sized copy and then walk it
        again. On the 31 M-voxel field that copy is 72 ms and 250 MiB; on a native ExaSPIM field
        window (3 x 141 x 1331 x 1775) it is 4 GiB, and the fold spent 26.4 s of its 179 s here.
        Keeping both ends instead of the larger magnitude measured 79 ms against 80 on a 184
        M-value window: the range is free, and it is the one a region can be sized from.
        """
        flat = self.values.reshape(self.values.shape[0], -1)
        return flat.min(axis=1), flat.max(axis=1)

    @cached_property
    def bound_xyz(self) -> np.ndarray:
        """``sup |values|`` per component, the range's symmetric envelope."""
        low, high = self.range_xyz
        return np.maximum(np.abs(low), np.abs(high))

    def bound(self) -> TransformBound:
        # The interval is clamped to include zero: the stage applies NO displacement outside its
        # grid, so a target region past the field's edge still needs its identity-mapped samples.
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
