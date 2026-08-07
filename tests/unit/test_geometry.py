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

"""The geometry vocabulary against SimpleITK, not against itself.

Every rule here is pinned to the external oracle: two KonfAI paths agree by construction,
including on a grid placed in the wrong place.
"""

import itertools

import numpy as np
import pytest
from konfai.data.geometry import AffineMap, Grid, TransformBound, WorldBox
from konfai.utils.dataset import Attribute
from konfai.utils.errors import TransformError

sitk = pytest.importorskip("SimpleITK")


def _oblique_direction() -> np.ndarray:
    a, b, c = np.deg2rad(17.0), np.deg2rad(-23.0), np.deg2rad(11.0)
    rz = np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[np.cos(b), 0.0, np.sin(b)], [0.0, 1.0, 0.0], [-np.sin(b), 0.0, np.cos(b)]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(c), -np.sin(c)], [0.0, np.sin(c), np.cos(c)]])
    return rz @ ry @ rx


def _grid(direction: np.ndarray | None = None) -> Grid:
    return Grid(
        size_zyx=(29, 33, 48),
        origin_xyz=np.array([-31.0, 12.5, 4.25]),
        spacing_xyz=np.array([0.7, 1.3, 0.9]),
        direction_xyz=np.eye(3) if direction is None else direction,
    )


def _image(grid: Grid) -> "sitk.Image":
    image = sitk.GetImageFromArray(np.zeros(grid.size_zyx, np.float32))
    image.SetOrigin(tuple(grid.origin_xyz))
    image.SetSpacing(tuple(grid.spacing_xyz))
    image.SetDirection(tuple(grid.direction_xyz.ravel()))
    return image


class TestGridAgainstSimpleITK:
    def test_index_to_world_is_transform_index_to_physical_point(self):
        grid = _grid(_oblique_direction())
        image = _image(grid)
        for index_xyz in ([0.0, 0.0, 0.0], [47.0, 32.0, 28.0], [3.5, -0.5, 17.25]):
            want = np.array(image.TransformContinuousIndexToPhysicalPoint(index_xyz))
            got = grid.index_to_world.apply(np.asarray(index_xyz))
            np.testing.assert_allclose(got, want, rtol=0.0, atol=1e-12)

    def test_sub_grid_origin_is_the_slab_origin(self):
        # The load-bearing identity is internal, and bit-for-bit: the slab origin IS the parent
        # map applied to the slab's first index: a recomputation that associated differently
        # would shear a streamed regrid at its seams. Agreement with ITK itself holds only to
        # tolerance: numpy does not reproduce ITK's product bit-for-bit on every ISA (arm64
        # contracts with FMA where x86-64 does not), so that half shares the neighbouring
        # test's 1e-12.
        grid = _grid(_oblique_direction())
        image = _image(grid)
        region = (slice(11, 21), slice(0, 33), slice(5, 48))
        sub = grid.sub_grid(region)
        assert sub.size_zyx == (10, 33, 43)
        np.testing.assert_array_equal(sub.origin_xyz, grid.index_to_world.apply(np.array([5.0, 0.0, 11.0])))
        want = np.array(image.TransformIndexToPhysicalPoint([5, 0, 11]))
        np.testing.assert_allclose(sub.origin_xyz, want, rtol=0.0, atol=1e-12)

    def test_world_to_index_round_trips(self):
        grid = _grid(_oblique_direction())
        points = np.random.RandomState(0).uniform(-200, 200, size=(50, 3))
        back = grid.world_to_index.apply(grid.index_to_world.apply(points))
        np.testing.assert_allclose(back, points, rtol=0.0, atol=1e-9)

    def test_world_box_covers_the_outer_faces(self):
        grid = _grid(_oblique_direction())
        image = _image(grid)
        box = grid.world_box((slice(4, 9), slice(0, 33), slice(0, 48)))
        corners = itertools.product((3.5, 8.5), (-0.5, 32.5), (-0.5, 47.5))
        for z, y, x in corners:
            point = np.array(image.TransformContinuousIndexToPhysicalPoint([x, y, z]))
            assert np.all(point >= box.low_xyz - 1e-9) and np.all(point <= box.high_xyz + 1e-9)


class TestGridOf:
    def _attribute(self, **overrides: object) -> Attribute:
        attribute = Attribute()
        values: dict[str, np.ndarray] = {
            "Origin": np.array([-31.0, 12.5, 4.25]),
            "Spacing": np.array([0.7, 1.3, 0.9]),
            "Direction": np.eye(3).ravel(),
        }
        values.update({key: np.asarray(value) for key, value in overrides.items()})
        for key, value in values.items():
            attribute[key] = value
        return attribute

    def test_reads_a_full_header(self):
        grid = Grid.of([29, 33, 48], self._attribute(), "case 'A'")
        assert grid.size_zyx == (29, 33, 48)
        np.testing.assert_array_equal(grid.spacing_xyz, [0.7, 1.3, 0.9])

    def test_refuses_a_missing_key_naming_it(self):
        attribute = self._attribute()
        attribute.pop("Spacing")
        with pytest.raises(TransformError, match="Spacing"):
            Grid.of([29, 33, 48], attribute, "case 'A'")

    def test_refuses_a_non_positive_spacing(self):
        with pytest.raises(TransformError, match="positive"):
            Grid.of([29, 33, 48], self._attribute(Spacing=np.array([0.7, 0.0, 0.9])), "case 'A'")

    def test_readable_is_total_and_quiet(self):
        assert Grid.readable(self._attribute())
        assert not Grid.readable(Attribute())


class TestWorldBoxImage:
    def test_image_under_equals_the_corner_hull(self):
        # The identity that licenses never enumerating corners: |A| on the half-extents equals the
        # hull of the 2^rank mapped corners, for any affine.
        rng = np.random.RandomState(7)
        for _ in range(200):
            affine = AffineMap(rng.randn(3, 3), rng.randn(3) * 100.0)
            low = rng.uniform(-100, 0, 3)
            high = low + rng.uniform(0.1, 200, 3)
            box = WorldBox(low, high)
            corners = np.array([affine.apply(np.array(c)) for c in itertools.product(*zip(low, high, strict=True))])
            image = box.image_under(affine)
            np.testing.assert_allclose(image.low_xyz, corners.min(axis=0), rtol=0.0, atol=1e-9)
            np.testing.assert_allclose(image.high_xyz, corners.max(axis=0), rtol=0.0, atol=1e-9)


class TestAffineMap:
    def test_then_composes_in_stated_order(self):
        inner = AffineMap(np.diag([2.0, 1.0, 1.0]), np.array([1.0, 0.0, 0.0]))
        outer = AffineMap(np.eye(3), np.array([0.0, 10.0, 0.0]))
        point = np.array([1.0, 1.0, 1.0])
        np.testing.assert_array_equal(inner.then(outer).apply(point), outer.apply(inner.apply(point)))

    def test_inverted_refuses_a_singular_matrix(self):
        singular = AffineMap(np.diag([1.0, 0.0, 1.0]), np.zeros(3))
        with pytest.raises(TransformError, match="singular"):
            singular.inverted()


class TestTransformBound:
    def test_after_transports_the_residual_through_the_outer_matrix(self):
        # T = A ∘ (id + d): the residual scales by |A|, not by 1. An affine of scale 3 around a
        # 5 mm spline reaches 15 mm: the naive bound is short by 10 voxels of every slab edge.
        scale = TransformBound.exact(AffineMap(np.diag([3.0, 1.0, 1.0]), np.zeros(3)))
        wiggle = TransformBound.shift(np.array([5.0, 5.0, 5.0]))
        composed = scale.after(wiggle)
        np.testing.assert_array_equal(composed.residual_xyz, [15.0, 5.0, 5.0])
        # And the other order leaves it alone.
        np.testing.assert_array_equal(wiggle.after(scale).residual_xyz, [5.0, 5.0, 5.0])

    def test_map_box_contains_the_true_image_of_a_nonlinear_map(self):
        rng = np.random.RandomState(3)
        affine = AffineMap(np.eye(3) + rng.randn(3, 3) * 0.1, rng.randn(3) * 10.0)
        residual = np.array([4.0, 2.0, 1.0])
        bound = TransformBound(affine, residual)
        box = WorldBox(np.array([-20.0, -10.0, 0.0]), np.array([15.0, 25.0, 30.0]))
        mapped = bound.map_box(box)
        points = rng.uniform(box.low_xyz, box.high_xyz, size=(500, 3))
        # Any map of the form affine + bounded wiggle stays inside.
        images = affine.apply(points) + rng.uniform(-1.0, 1.0, size=(500, 3)) * residual
        assert np.all(images >= mapped.low_xyz - 1e-9) and np.all(images <= mapped.high_xyz + 1e-9)
