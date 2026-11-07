# SPDX-License-Identifier: Apache-2.0
"""What a viewer adds to KonfAI's geometry: plane naming, the radiological frame, boxes and windows in millimetres,
on oblique and anisotropic grids, and a 2-D grid promoted to one plane."""

import numpy as np
import pytest
from konfai.data.geometry import Grid
from konfai_scout.planes import (
    PLANES,
    box_around,
    box_from_dict,
    box_to_dict,
    center_mm,
    dominant_world_axis,
    index_along,
    index_to_world,
    plane_axes,
    plane_axis,
    plane_frame,
    plane_of,
    promoted,
    window_of,
    world_box_of,
)


def oblique_grid() -> Grid:
    angle = np.deg2rad(25.0)
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]])
    return Grid((20, 40, 30), np.array([-40.0, 25.0, 10.0]), np.array([0.7, 1.1, 2.5]), rotation)


def identity_grid() -> Grid:
    return Grid((10, 10, 10), np.zeros(3), np.ones(3), np.eye(3))


def test_planes_name_the_dominant_axis_and_stay_distinct():
    assert plane_axes(identity_grid()) == {"axial": 0, "coronal": 1, "sagittal": 2}
    assert plane_axes(oblique_grid()) == {"axial": 0, "coronal": 1, "sagittal": 2}
    swapped = Grid((10, 10, 10), np.zeros(3), np.ones(3), np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], float))
    assert plane_axes(swapped) == {"axial": 2, "coronal": 1, "sagittal": 0}
    for plane in PLANES:
        assert plane_of(identity_grid(), plane_axis(identity_grid(), plane)) == plane
    with pytest.raises(ValueError):
        plane_axis(identity_grid(), "oblique")


def test_the_radiological_frame():
    grid = identity_grid()
    assert plane_frame(grid, "axial").edges == ("R", "L", "A", "P")
    assert plane_frame(grid, "coronal").edges == ("R", "L", "S", "I")
    assert plane_frame(grid, "sagittal").edges == ("A", "P", "S", "I")
    coronal = plane_frame(grid, "coronal")
    assert coronal.flip_rows and not coronal.flip_cols  # z runs upward on screen
    flipped = Grid((10, 10, 10), np.zeros(3), np.ones(3), np.diag([-1.0, 1.0, 1.0]))  # x stored towards the right
    assert plane_frame(flipped, "axial").flip_cols and plane_frame(flipped, "axial").edges == ("R", "L", "A", "P")
    assert dominant_world_axis(oblique_grid(), 2) == (0, 1)


def test_windows_cover_every_voxel_centre_inside_a_box():
    grid = oblique_grid()
    box = box_around(center_mm(grid), (12.0, 9.0, 11.0))
    window = window_of(grid, box)
    assert window is not None
    zz, yy, xx = np.indices(grid.size_zyx)
    centres = grid.index_to_world.apply(np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1).astype(float))
    inside = np.all((centres >= box.low_xyz) & (centres <= box.high_xyz), axis=1)
    assert inside.any()
    for z, y, x in zip(zz.ravel()[inside], yy.ravel()[inside], xx.ravel()[inside], strict=True):
        assert window[0].start <= z < window[0].stop and window[1].start <= y < window[1].stop
        assert window[2].start <= x < window[2].stop
    measured = world_box_of(grid, window)
    assert np.all(centres[inside] >= measured.low_xyz - 1e-6) and np.all(centres[inside] <= measured.high_xyz + 1e-6)
    assert window_of(grid, box_around((500.0, 500.0, 500.0), 10.0)) is None


def test_index_along_and_box_forms():
    grid = oblique_grid()
    assert index_along(grid, 0, index_to_world(grid, [0, 0, 7])) == 7
    assert index_along(grid, 2, index_to_world(grid, [-10, 0, 0])) == 0
    a = box_from_dict({"min": [0, 0, 0], "max": [2, 4, 6]})
    b = box_from_dict({"center": [1, 2, 3], "size": 2})
    c = box_from_dict({"center_mm": [1, 2, 3], "size_mm": 2})
    assert box_to_dict(a) == {"min": [0.0, 0.0, 0.0], "max": [2.0, 4.0, 6.0]}
    assert box_to_dict(b) == box_to_dict(c) == {"min": [0.0, 1.0, 2.0], "max": [2.0, 3.0, 4.0]}
    assert box_to_dict(box_from_dict({"min": [2, 4, 6], "max": [0, 0, 0]})) == box_to_dict(a)  # corners in any order
    with pytest.raises(ValueError):
        box_from_dict({"min": [0, 0, 0]})


def test_a_2d_grid_is_promoted_to_one_plane():
    flat = Grid((32, 64), np.array([1.0, 2.0]), np.array([0.5, 0.5]), np.eye(2))
    grid = promoted(flat)
    assert grid.size_zyx == (1, 32, 64) and grid.rank == 3
    assert grid.spacing_xyz.tolist() == [0.5, 0.5, 1.0] and grid.origin_xyz.tolist() == [1.0, 2.0, 0.0]
    assert plane_axes(grid)["axial"] == 0
    assert promoted(grid) is grid
