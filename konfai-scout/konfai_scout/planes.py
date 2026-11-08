# SPDX-License-Identifier: Apache-2.0
"""How a volume is looked at: its planes, named after the world axis each normal mostly follows, the radiological
frame of each, and the millimetre boxes the model speaks in. The geometry itself is KonfAI's
(:class:`konfai.data.geometry.Grid`, :class:`~konfai.data.geometry.WorldBox`): here only what a viewer adds to it."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from konfai.data.geometry import Grid, WorldBox

#: The planes, each named after the world axis its normal mostly follows (z, y, x).
PLANES = ("axial", "coronal", "sagittal")
PLANE_WORLD_AXIS = {"axial": 2, "coronal": 1, "sagittal": 0}
#: The negative and positive ends of each world axis (x, y, z) in LPS.
LPS_ENDS = (("R", "L"), ("A", "P"), ("I", "S"))


def promoted(grid: Grid) -> Grid:
    """A rank-3 grid for any grid: a 2-D image is one plane thick along z, at unit spacing."""
    if grid.rank == 3:
        return grid
    direction = np.eye(3)
    direction[:2, :2] = grid.direction_xyz
    return Grid(
        (1, *grid.size_zyx),
        np.append(np.asarray(grid.origin_xyz, dtype=np.float64), 0.0),
        np.append(np.asarray(grid.spacing_xyz, dtype=np.float64), 1.0),
        direction,
    )


def axis_world_vector(grid: Grid, array_axis: int) -> np.ndarray:
    """The world step of one voxel along an array axis (0 = Z, 1 = Y, 2 = X), in mm."""
    return np.asarray(grid.index_to_world.matrix[:, 2 - array_axis], dtype=np.float64)


def plane_axes(grid: Grid) -> dict[str, int]:
    """Each plane's array axis: the one whose normal is closest to the plane's world axis, assigned uniquely so an
    oblique grid still gets three distinct planes."""
    cosines = np.abs(np.asarray(grid.direction_xyz))  # rows: world axis, columns: index axis
    assignment: dict[str, int] = {}
    taken: set[int] = set()
    for plane in sorted(PLANES, key=lambda p: -cosines[PLANE_WORLD_AXIS[p]].max()):
        row = cosines[PLANE_WORLD_AXIS[plane]]
        best = max((j for j in range(3) if j not in taken), key=lambda j: row[j])
        taken.add(best)
        assignment[plane] = 2 - best
    return assignment


def plane_axis(grid: Grid, plane: str) -> int:
    if plane not in PLANE_WORLD_AXIS:
        raise ValueError(f"unknown plane {plane!r}; one of {', '.join(PLANES)}")
    return plane_axes(grid)[plane]


def dominant_world_axis(grid: Grid, array_axis: int) -> tuple[int, int]:
    """The world axis an array axis mostly runs along, and the sign of that run."""
    vector = axis_world_vector(grid, array_axis)
    world = int(np.argmax(np.abs(vector)))
    return world, (1 if vector[world] >= 0 else -1)


def index_along(grid: Grid, array_axis: int, xyz_mm: object) -> int:
    """The slice index along an array axis nearest to a world point, clamped to the grid."""
    index = grid.world_to_index.apply(np.asarray(xyz_mm, dtype=np.float64))[2 - array_axis]
    return int(min(max(round(float(index)), 0), grid.size_zyx[array_axis] - 1))


def same_grid(a: Grid, b: Grid) -> bool:
    """Whether two grids sample the same voxels, within what a header round trip loses (``Grid.same_as`` is
    exact): the same extent, and the same geometry to a tenth of a micron."""
    return tuple(a.size_zyx) == tuple(b.size_zyx) and all(
        np.allclose(getattr(a, key), getattr(b, key), rtol=1e-5, atol=1e-4)
        for key in ("origin_xyz", "spacing_xyz", "direction_xyz")
    )


# ------------------------------------------------------------------------------------------ boxes


def box(low_xyz: object, high_xyz: object) -> WorldBox:
    low, high = np.asarray(low_xyz, dtype=np.float64), np.asarray(high_xyz, dtype=np.float64)
    return WorldBox(np.minimum(low, high), np.maximum(low, high))


def box_around(center_xyz: object, size_mm: object) -> WorldBox:
    center = np.asarray(center_xyz, dtype=np.float64)
    half = np.broadcast_to(np.asarray(size_mm, dtype=np.float64), (3,)) / 2
    return WorldBox(center - half, center + half)


def box_from_dict(value: object) -> WorldBox:
    """``{min, max}`` or ``{center, size}`` in millimetres (``_mm`` suffixes tolerated)."""
    if isinstance(value, dict):
        value = {str(key).removesuffix("_mm"): v for key, v in value.items() if v is not None}
        if "min" in value and "max" in value:
            return box(_triple(value["min"]), _triple(value["max"]))
        if "center" in value and "size" in value:
            return box_around(_triple(value["center"]), value["size"])
    raise ValueError("a box is {min: [x, y, z], max: [x, y, z]} or {center: [x, y, z], size: mm} in millimetres")


def _triple(values: object) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).ravel()
    return np.concatenate([array, np.zeros(3)])[:3]


def box_to_dict(value: WorldBox) -> dict[str, list[float]]:
    return {"min": [round(float(v), 3) for v in value.low_xyz], "max": [round(float(v), 3) for v in value.high_xyz]}


def box_center(value: WorldBox) -> np.ndarray:
    return (value.low_xyz + value.high_xyz) / 2


def box_size(value: WorldBox) -> np.ndarray:
    return value.high_xyz - value.low_xyz


def box_intersection(a: WorldBox, b: WorldBox) -> WorldBox | None:
    low, high = np.maximum(a.low_xyz, b.low_xyz), np.minimum(a.high_xyz, b.high_xyz)
    return None if np.any(high <= low) else WorldBox(low, high)


def window_of(grid: Grid, value: WorldBox) -> tuple[slice, slice, slice] | None:
    """The voxels of a box as a clamped array-order window (``Grid.index_window``), or None when the box misses the
    grid: KonfAI clamps to a border voxel, a viewer must say nothing is there."""
    low, high = grid.continuous_box(value)
    extents = np.asarray(grid.size_zyx[::-1], dtype=np.float64)
    if np.any(high < -0.5) or np.any(low > extents - 0.5):
        return None
    window = grid.index_window(value, 0)
    return window[0], window[1], window[2]


def describe_grid(grid: Grid) -> dict[str, object]:
    size_xyz = [int(n) for n in reversed(grid.size_zyx)]
    spacing = [float(s) for s in grid.spacing_xyz]
    return {
        "size_xyz": size_xyz,
        "spacing_mm_xyz": [round(s, 4) for s in spacing],
        "origin_mm_xyz": [round(float(o), 3) for o in grid.origin_xyz],
        "direction": [round(float(v), 6) for v in np.asarray(grid.direction_xyz).ravel()],
        "extent_mm_xyz": [round(n * s, 2) for n, s in zip(size_xyz, spacing, strict=True)],
        "bounds_mm": box_to_dict(grid.world_box()),
        "planes": {
            plane: {"array_axis": axis, "slices": grid.size_zyx[axis]} for plane, axis in plane_axes(grid).items()
        },
    }


# ------------------------------------------------------------------------------------------ frames


@dataclass(frozen=True)
class PlaneFrame:
    """How a plane is laid out on screen: which array axes run along rows and columns, flipped or not, and the
    direction letters at the left, right, top and bottom edges."""

    plane: str
    array_axis: int
    rows_axis: int
    cols_axis: int
    flip_rows: bool
    flip_cols: bool
    edges: tuple[str, str, str, str]


def plane_frame(grid: Grid, plane: str) -> PlaneFrame:
    """Radiological convention: the patient's left on the right of an axial or coronal plane, the anterior side on
    the left of a sagittal one, superior at the top."""
    axis = plane_axis(grid, plane)
    rows_axis, cols_axis = [a for a in range(3) if a != axis]
    wanted = {0: 1, 1: 1, 2: -1}  # along x: towards L; along y: towards P; along z: S at the top
    world_c, sign_c = dominant_world_axis(grid, cols_axis)
    world_r, sign_r = dominant_world_axis(grid, rows_axis)
    flip_cols, flip_rows = sign_c != wanted[world_c], sign_r != wanted[world_r]
    sign_c, sign_r = wanted[world_c], wanted[world_r]
    right, left = LPS_ENDS[world_c][1 if sign_c > 0 else 0], LPS_ENDS[world_c][0 if sign_c > 0 else 1]
    bottom, top = LPS_ENDS[world_r][1 if sign_r > 0 else 0], LPS_ENDS[world_r][0 if sign_r > 0 else 1]
    return PlaneFrame(plane, axis, rows_axis, cols_axis, flip_rows, flip_cols, (left, right, top, bottom))
