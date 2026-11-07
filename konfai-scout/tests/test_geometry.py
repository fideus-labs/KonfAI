# SPDX-License-Identifier: Apache-2.0
"""World coordinates: round trips, plane naming, boxes, on oblique and anisotropic grids."""

import numpy as np
import pytest
from konfai_scout.geometry import PLANES, Grid, WorldBox


def oblique_grid() -> Grid:
    angle = np.deg2rad(25.0)
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]])
    return Grid((30, 40, 20), (0.7, 1.1, 2.5), (-40.0, 25.0, 10.0), tuple(rotation.ravel()))


def test_world_index_round_trip_on_an_oblique_anisotropic_grid():
    grid = oblique_grid()
    rng = np.random.default_rng(0)
    points = rng.uniform(-100, 100, (50, 3))
    assert np.allclose(grid.index_to_world(grid.world_to_index(points)), points)
    assert np.allclose(grid.index_to_world([0, 0, 0]), grid.origin_xyz)
    assert np.allclose(grid.index_to_world([1, 0, 0]) - grid.index_to_world([0, 0, 0]), grid.matrix[:, 0] * 0.7)


def test_planes_name_the_dominant_axis_and_stay_distinct():
    identity = Grid((10, 10, 10), (1, 1, 1), (0, 0, 0), tuple(np.eye(3).ravel()))
    assert identity.plane_axes == {"axial": 0, "coronal": 1, "sagittal": 2}
    assert oblique_grid().plane_axes == {"axial": 0, "coronal": 1, "sagittal": 2}
    swapped = Grid(
        (10, 10, 10), (1, 1, 1), (0, 0, 0), tuple(np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], float).ravel())
    )
    assert swapped.plane_axes == {"axial": 2, "coronal": 1, "sagittal": 0}
    assert set(oblique_grid().plane_axes) == set(PLANES)
    for plane in PLANES:
        assert identity.plane_of(identity.plane_axis(plane)) == plane


def test_index_box_covers_every_voxel_centre_inside_the_box():
    grid = oblique_grid()
    box = WorldBox.around(grid.center_mm(), (12.0, 9.0, 11.0))
    lo, hi = box.index_box(grid)
    zz, yy, xx = np.indices(grid.shape_zyx)
    centres = grid.index_to_world(np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1))
    inside = np.all((centres >= box.min_xyz) & (centres <= box.max_xyz), axis=1)
    assert inside.any()
    for z, y, x in zip(zz.ravel()[inside], yy.ravel()[inside], xx.ravel()[inside], strict=True):
        assert lo[0] <= z < hi[0] and lo[1] <= y < hi[1] and lo[2] <= x < hi[2]
    measured = WorldBox.of_index_box(grid, lo, hi)
    assert measured.intersection(box) is not None
    # every centre inside the box is inside what was measured; the margin between centres is not promised
    assert np.all(centres[inside] >= np.asarray(measured.min_xyz) - 1e-6)
    assert np.all(centres[inside] <= np.asarray(measured.max_xyz) + 1e-6)


def test_an_empty_or_outside_box_holds_no_voxel():
    grid = oblique_grid()
    assert WorldBox((500.0, 500.0, 500.0), (510.0, 510.0, 510.0)).index_box(grid) is None
    assert WorldBox((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)).volume_mm3 == 0.0


def test_a_2d_image_is_one_plane_thick():
    grid = Grid.of((64, 32), (0.5, 0.5), (1.0, 2.0), (1.0, 0.0, 0.0, 1.0))
    assert grid.size_xyz == (64, 32, 1)
    assert grid.spacing_xyz == (0.5, 0.5, 1.0)
    assert grid.origin_xyz == (1.0, 2.0, 0.0)
    assert grid.plane_axes["axial"] == 0 and grid.shape_zyx == (1, 32, 64)


def test_box_forms():
    a = WorldBox.from_dict({"min": [0, 0, 0], "max": [2, 4, 6]})
    b = WorldBox.from_dict({"center": [1, 2, 3], "size": 2})
    c = WorldBox.from_dict({"center_mm": [1, 2, 3], "size_mm": 2})
    assert a.center.tolist() == [1, 2, 3] and b.size.tolist() == [2, 2, 2] and c == b
    assert WorldBox((2, 4, 6), (0, 0, 0)) == a  # corners in any order
    with pytest.raises(ValueError):
        WorldBox.from_dict({"min": [0, 0, 0]})


def test_index_along_and_direction_signs():
    grid = oblique_grid()
    assert grid.index_along(0, grid.index_to_world([0, 0, 7])) == 7
    assert grid.index_along(2, grid.index_to_world([-10, 0, 0])) == 0
    world, sign = grid.dominant_world_axis(2)  # the X array axis mostly runs along world x, positively
    assert (world, sign) == (0, 1)
