# SPDX-License-Identifier: Apache-2.0
"""World coordinates for an inspection: a grid, a box in millimetres, and the planes a volume is looked at on.

Array data is ``[(C,) Z, Y, X]``; physical geometry (origin, spacing, direction) is ``(x, y, z)`` in LPS, as
SimpleITK and KonfAI keep it. A plane is an array axis (0 = Z, 1 = Y, 2 = X) named after the world axis its normal
mostly runs along, so an oblique volume is viewed on its own slices and every coordinate handed out stays physical.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np

#: The planes, each named after the world axis its normal mostly follows (z, y, x).
PLANES = ("axial", "coronal", "sagittal")
PLANE_WORLD_AXIS = {"axial": 2, "coronal": 1, "sagittal": 0}
#: The negative and positive ends of each world axis (x, y, z) in LPS.
LPS_ENDS = (("R", "L"), ("A", "P"), ("I", "S"))


def _triple(values: object, pad: float, count: int = 3) -> tuple[float, ...]:
    out = [float(v) for v in np.asarray(values, dtype=float).ravel()]
    return tuple(out + [pad] * (count - len(out)))


@dataclass(frozen=True)
class Grid:
    """A regular grid in world coordinates: ``world = origin + direction @ diag(spacing) @ index_xyz``."""

    size_xyz: tuple[int, int, int]
    spacing_xyz: tuple[float, float, float]
    origin_xyz: tuple[float, float, float]
    #: Row-major 3x3 (ITK): column ``j`` is the world direction of index axis ``j`` (x, y, z).
    direction: tuple[float, ...]

    @staticmethod
    def of(size: object, spacing: object, origin: object, direction: object) -> Grid:
        """A grid from what an image reader hands out; a 2-D image is one plane thick along z."""
        size_list = [int(v) for v in np.asarray(size).ravel()]
        matrix = np.asarray(direction, dtype=float).ravel()
        if len(size_list) == 2:
            full = np.eye(3)
            full[:2, :2] = matrix.reshape(2, 2)
            matrix = full.ravel()
        return Grid(
            tuple((size_list + [1, 1])[:3]),  # type: ignore[arg-type]
            _triple(spacing, 1.0),  # type: ignore[arg-type]
            _triple(origin, 0.0),  # type: ignore[arg-type]
            tuple(float(v) for v in matrix),
        )

    def __post_init__(self) -> None:
        object.__setattr__(self, "size_xyz", tuple(int(v) for v in self.size_xyz))
        if len(self.direction) != 9:
            raise ValueError(f"a grid direction holds 9 values, got {len(self.direction)}")

    @property
    def matrix(self) -> np.ndarray:
        return np.asarray(self.direction, dtype=float).reshape(3, 3)

    @property
    def shape_zyx(self) -> tuple[int, int, int]:
        return (self.size_xyz[2], self.size_xyz[1], self.size_xyz[0])

    @cached_property
    def _to_world(self) -> np.ndarray:
        return self.matrix @ np.diag(self.spacing_xyz)

    @cached_property
    def _to_index(self) -> np.ndarray:
        return np.linalg.inv(self._to_world)

    def index_to_world(self, ijk: object) -> np.ndarray:
        """Continuous index ``(i, j, k)`` (x, y, z order) to world millimetres; any leading shape."""
        return np.asarray(ijk, dtype=float) @ self._to_world.T + np.asarray(self.origin_xyz)

    def world_to_index(self, xyz: object) -> np.ndarray:
        return (np.asarray(xyz, dtype=float) - np.asarray(self.origin_xyz)) @ self._to_index.T

    def axis_world_vector(self, array_axis: int) -> np.ndarray:
        """The world step of one voxel along an array axis (0 = Z, 1 = Y, 2 = X), in mm."""
        return self._to_world[:, 2 - array_axis].copy()

    @cached_property
    def bounds(self) -> WorldBox:
        """The world box of the voxels (their outer faces, half a voxel beyond the centres)."""
        corners = np.array(
            [
                [i, j, k]
                for i in (-0.5, self.size_xyz[0] - 0.5)
                for j in (-0.5, self.size_xyz[1] - 0.5)
                for k in (-0.5, self.size_xyz[2] - 0.5)
            ]
        )
        return WorldBox.of_points(self.index_to_world(corners))

    @cached_property
    def plane_axes(self) -> dict[str, int]:
        """Each plane's array axis: the one whose normal is closest to the plane's world axis, assigned uniquely so
        an oblique grid still gets three distinct planes."""
        cosines = np.abs(self.matrix)  # rows: world axis, columns: index axis
        assignment: dict[str, int] = {}
        taken: set[int] = set()
        order = sorted(PLANES, key=lambda p: -cosines[PLANE_WORLD_AXIS[p]].max())
        for plane in order:
            row = cosines[PLANE_WORLD_AXIS[plane]]
            free = [j for j in range(3) if j not in taken]
            best = max(free, key=lambda j: row[j])
            taken.add(best)
            assignment[plane] = 2 - best
        return assignment

    def plane_axis(self, plane: str) -> int:
        if plane not in PLANE_WORLD_AXIS:
            raise ValueError(f"unknown plane {plane!r}; one of {', '.join(PLANES)}")
        return self.plane_axes[plane]

    def plane_of(self, array_axis: int) -> str:
        return next(name for name, axis in self.plane_axes.items() if axis == array_axis)

    def index_along(self, array_axis: int, xyz_mm: object) -> int:
        """The slice index along an array axis nearest to a world point, clamped to the grid."""
        index = self.world_to_index(xyz_mm)[2 - array_axis]
        return int(min(max(round(float(index)), 0), self.shape_zyx[array_axis] - 1))

    def center_mm(self) -> np.ndarray:
        return self.index_to_world([(n - 1) / 2 for n in self.size_xyz])

    def dominant_world_axis(self, array_axis: int) -> tuple[int, int]:
        """The world axis an array axis mostly runs along, and the sign of that run."""
        vector = self.axis_world_vector(array_axis)
        world = int(np.argmax(np.abs(vector)))
        return world, (1 if vector[world] >= 0 else -1)

    def describe(self) -> dict[str, object]:
        extent = [n * s for n, s in zip(self.size_xyz, self.spacing_xyz, strict=True)]
        return {
            "size_xyz": list(self.size_xyz),
            "spacing_mm_xyz": [round(s, 4) for s in self.spacing_xyz],
            "origin_mm_xyz": [round(o, 3) for o in self.origin_xyz],
            "direction": [round(v, 6) for v in self.direction],
            "extent_mm_xyz": [round(e, 2) for e in extent],
            "bounds_mm": self.bounds.to_dict(),
            "planes": {
                plane: {"array_axis": axis, "slices": self.shape_zyx[axis]} for plane, axis in self.plane_axes.items()
            },
        }


@dataclass(frozen=True)
class WorldBox:
    """An axis-aligned box in world millimetres (LPS)."""

    min_xyz: tuple[float, float, float]
    max_xyz: tuple[float, float, float]

    def __post_init__(self) -> None:
        lo = np.minimum(self.min_xyz, self.max_xyz)
        hi = np.maximum(self.min_xyz, self.max_xyz)
        object.__setattr__(self, "min_xyz", tuple(float(v) for v in lo))
        object.__setattr__(self, "max_xyz", tuple(float(v) for v in hi))

    @staticmethod
    def of_points(points: object) -> WorldBox:
        array = np.asarray(points, dtype=float).reshape(-1, 3)
        return WorldBox(tuple(array.min(axis=0)), tuple(array.max(axis=0)))  # type: ignore[arg-type]

    @staticmethod
    def around(center_xyz: object, size_mm: object) -> WorldBox:
        center = np.asarray(center_xyz, dtype=float)
        half = np.broadcast_to(np.asarray(size_mm, dtype=float), (3,)) / 2
        return WorldBox(tuple(center - half), tuple(center + half))  # type: ignore[arg-type]

    @staticmethod
    def from_dict(value: object) -> WorldBox:
        if isinstance(value, dict):
            value = {key.removesuffix("_mm"): v for key, v in value.items() if v is not None}
        if isinstance(value, dict) and "min" in value and "max" in value:
            return WorldBox(_triple(value["min"], 0.0), _triple(value["max"], 0.0))  # type: ignore[arg-type]
        if isinstance(value, dict) and "center" in value and "size" in value:
            return WorldBox.around(value["center"], value["size"])
        raise ValueError("a box is {min: [x, y, z], max: [x, y, z]} or {center: [x, y, z], size: mm} in millimetres")

    def to_dict(self) -> dict[str, list[float]]:
        return {"min": [round(v, 3) for v in self.min_xyz], "max": [round(v, 3) for v in self.max_xyz]}

    @property
    def center(self) -> np.ndarray:
        return (np.asarray(self.min_xyz) + np.asarray(self.max_xyz)) / 2

    @property
    def size(self) -> np.ndarray:
        return np.asarray(self.max_xyz) - np.asarray(self.min_xyz)

    @property
    def volume_mm3(self) -> float:
        return float(np.prod(self.size))

    def intersection(self, other: WorldBox) -> WorldBox | None:
        lo = np.maximum(self.min_xyz, other.min_xyz)
        hi = np.minimum(self.max_xyz, other.max_xyz)
        if np.any(hi <= lo):
            return None
        return WorldBox(tuple(lo), tuple(hi))  # type: ignore[arg-type]

    def index_box(self, grid: Grid) -> tuple[tuple[int, int, int], tuple[int, int, int]] | None:
        """The voxels whose centres may fall inside, as ``(lo_zyx, hi_zyx)`` with ``hi`` exclusive, clamped to the
        grid: the index-space bounding box of the eight corners, so on an oblique grid it is a superset. ``None``
        when no voxel is inside."""
        corners = np.array(
            [
                [x, y, z]
                for x in (self.min_xyz[0], self.max_xyz[0])
                for y in (self.min_xyz[1], self.max_xyz[1])
                for z in (self.min_xyz[2], self.max_xyz[2])
            ]
        )
        indices = grid.world_to_index(corners)
        lo_xyz = np.ceil(indices.min(axis=0) - 1e-6).astype(int)
        hi_xyz = np.floor(indices.max(axis=0) + 1e-6).astype(int) + 1
        lo_xyz = np.clip(lo_xyz, 0, grid.size_xyz)
        hi_xyz = np.clip(hi_xyz, 0, grid.size_xyz)
        if np.any(hi_xyz <= lo_xyz):
            return None
        return (int(lo_xyz[2]), int(lo_xyz[1]), int(lo_xyz[0])), (int(hi_xyz[2]), int(hi_xyz[1]), int(hi_xyz[0]))

    @staticmethod
    def of_index_box(grid: Grid, lo_zyx: tuple[int, int, int], hi_zyx: tuple[int, int, int]) -> WorldBox:
        """The world box the voxels ``lo:hi`` (array order, ``hi`` exclusive) cover, outer faces included."""
        lo = np.asarray(lo_zyx[::-1], dtype=float) - 0.5
        hi = np.asarray(hi_zyx[::-1], dtype=float) - 0.5
        corners = np.array([[i, j, k] for i in (lo[0], hi[0]) for j in (lo[1], hi[1]) for k in (lo[2], hi[2])])
        return WorldBox.of_points(grid.index_to_world(corners))
