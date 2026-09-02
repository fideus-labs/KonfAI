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
        bound = TransformBound(affine, -residual, residual)
        box = WorldBox(np.array([-20.0, -10.0, 0.0]), np.array([15.0, 25.0, 30.0]))
        mapped = bound.map_box(box)
        points = rng.uniform(box.low_xyz, box.high_xyz, size=(500, 3))
        # Any map of the form affine + bounded wiggle stays inside.
        images = affine.apply(points) + rng.uniform(-1.0, 1.0, size=(500, 3)) * residual
        assert np.all(images >= mapped.low_xyz - 1e-9) and np.all(images <= mapped.high_xyz + 1e-9)

    def test_a_one_sided_interval_moves_the_box_instead_of_widening_it(self):
        # The whole reason the bound is a range and not a radius. A field that displaces every point
        # by the same 30 mm is an offset, not a spread: the region it reads sits 30 mm away and is
        # no larger. Priced as a radius it would be 60 mm wider AND still centred where it started,
        # which on a volume thinner than 60 mm is every voxel there is.
        box = WorldBox(np.zeros(3), np.array([10.0, 10.0, 10.0]))
        offset = TransformBound.interval(np.full(3, -30.0), np.full(3, -30.0))
        moved = offset.map_box(box)
        np.testing.assert_array_equal(moved.low_xyz, [-30.0, -30.0, -30.0])
        np.testing.assert_array_equal(moved.high_xyz, [-20.0, -20.0, -20.0])
        # The radius spelling of the same field is the one that cannot say that.
        radius = TransformBound.shift(np.full(3, 30.0)).map_box(box)
        assert np.all(radius.high_xyz - radius.low_xyz > moved.high_xyz - moved.low_xyz)

    def test_after_carries_a_one_sided_interval_through_a_sign_flip(self):
        # |A| @ residual is right for an interval centred on zero and wrong for one that is not: a
        # negative entry sends the inner low end to the outer high end. Here the outer affine
        # mirrors x, so a field that only ever pushes -x must come out only ever pushing +x. The
        # symmetric arithmetic would answer [-4, +4] on every axis: containing, but four times the
        # width, and it hides which side the map actually reaches.
        mirror = TransformBound.exact(AffineMap(np.diag([-1.0, 1.0, 1.0]), np.zeros(3)))
        pushes = TransformBound.interval(np.array([-4.0, 0.0, 0.0]), np.array([-2.0, 0.0, 0.0]))
        folded = mirror.after(pushes)
        np.testing.assert_array_equal(folded.low_xyz, [2.0, 0.0, 0.0])
        np.testing.assert_array_equal(folded.high_xyz, [4.0, 0.0, 0.0])
        # And it still contains the map it bounds, which is the property the sign trick must keep.
        rng = np.random.RandomState(11)
        points = rng.uniform(-50.0, 50.0, size=(400, 3))
        displaced = points + np.stack([rng.uniform(-4.0, -2.0, 400), np.zeros(400), np.zeros(400)], axis=-1)
        images = mirror.affine.apply(displaced)
        box = folded.map_box(WorldBox(points.min(axis=0), points.max(axis=0)))
        assert np.all(images >= box.low_xyz - 1e-9) and np.all(images <= box.high_xyz + 1e-9)


class TestDisplacementStageBound:
    def test_the_bound_is_one_pass_per_stage_and_stays_out_of_the_pickle(self):
        import pickle

        from konfai.data.geometry import DisplacementStage

        values = np.random.RandomState(5).normal(0.0, 3.0, (3, 6, 7, 8))
        stage = DisplacementStage(_grid(), values, 1)
        # Computed once and kept: asked of the property itself rather than of whichever numpy call
        # it happens to make, so the claim survives a change of arithmetic. It did not: the body
        # used to call np.abs and the count was the probe.
        assert "bound_xyz" not in stage.__dict__
        first = stage.bound_xyz
        assert "bound_xyz" in stage.__dict__
        for _ in range(5):
            stage.bound()
            assert stage.bound_xyz is first
        # sup |v| per component, however it is computed: no values-sized temporary is needed for it.
        np.testing.assert_array_equal(first, np.abs(values).reshape(3, -1).max(axis=1))
        # Signs both ways, since max(max v, -min v) is only sup |v| if both ends are looked at.
        for skewed in (np.abs(values), -np.abs(values), values * 0.0):
            skewed_stage = DisplacementStage(_grid(), skewed, 1)
            np.testing.assert_array_equal(skewed_stage.bound_xyz, np.abs(skewed).reshape(3, -1).max(axis=1))
        # The cache is per instance, not per pickle: a rank rebuilds it from the values it receives.
        assert "bound_xyz" not in stage.__getstate__() and "range_xyz" not in stage.__getstate__()
        again = pickle.loads(pickle.dumps(stage))
        np.testing.assert_array_equal(again.bound_xyz, first)

    def test_the_stage_bounds_a_one_sided_field_by_where_it_reaches(self):
        from konfai.data.geometry import DisplacementStage

        # A field that only ever pushes one way -- which is what a registration between two frames
        # writes, the offset between them baked into every voxel. The stage must say where it
        # reaches, not how far: as a magnitude this field is worth 12 either side, and it never
        # sends a point anywhere but 8 to 12 units below where it started.
        values = np.random.RandomState(7).uniform(-12.0, -8.0, (3, 5, 6, 7))
        stage = DisplacementStage(_grid(), values, 1)
        low, high = stage.range_xyz
        np.testing.assert_allclose(low, values.reshape(3, -1).min(axis=1))
        np.testing.assert_allclose(high, values.reshape(3, -1).max(axis=1))
        bound = stage.bound()
        # The BOUND still includes zero: outside its grid the stage displaces nothing, so a region
        # past the field's edge keeps its identity-mapped samples in the source window.
        np.testing.assert_array_equal(bound.low_xyz, low)
        np.testing.assert_array_equal(bound.high_xyz, np.zeros(3))
        # The envelope is still there for whoever wants one number, and it is the old sup |v|.
        np.testing.assert_array_equal(stage.bound_xyz, np.abs(values).reshape(3, -1).max(axis=1))


class TestSignedPermutation:
    """The one predicate and the one region rule every orientation stage shares."""

    def test_the_predicate_admits_exactly_the_signed_permutations(self):
        from konfai.data.geometry import SIGNED_PERMUTATION_ATOL_FLOAT64, signed_permutation

        # x<->z swap with x mirrored: output phys x reads input z, output z reads -input x.
        matrix = np.asarray([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
        # Array order (z, y, x): output z reads x mirrored... derived below against apply_remap.
        remap = signed_permutation(matrix, SIGNED_PERMUTATION_ATOL_FLOAT64)
        assert remap is not None and sorted(source for source, _ in remap) == [0, 1, 2]
        # Unit column sums alone pass an averaging matrix; unit peaks alone a superposing one; a
        # rank-deficient matrix reading one axis twice passes both column tests. All three refuse.
        averaging = np.linalg.inv(np.asarray([[0.5, 0.25, 0.25], [0.25, 0.5, 0.25], [0.25, 0.25, 0.5]]))
        superposing = np.linalg.inv(np.asarray([[1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 1.0]]))
        degenerate = np.asarray([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
        for refused in (averaging, superposing, degenerate):
            assert signed_permutation(refused, SIGNED_PERMUTATION_ATOL_FLOAT64) is None

    def test_a_float32_quarter_turn_is_admitted_at_its_own_tolerance(self):
        import torch
        from konfai.data.geometry import SIGNED_PERMUTATION_ATOL_FLOAT32, signed_permutation

        # The float32 provenance the looser constant exists for: a quarter-turn matrix composed
        # from float32 cosines lands within ~1e-7 of the 0/+-1 entries it stands for, and must
        # still be read as the exact remap it is.
        angles = torch.deg2rad(torch.tensor([90.0, 270.0, 180.0]))
        cos, sin = torch.cos(angles), torch.sin(angles)
        matrix = torch.tensor([[cos[0], -sin[0], 0.0], [sin[0], cos[0], 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32)
        assert not torch.equal(matrix, torch.round(matrix))  # the entries really are off the lattice
        assert signed_permutation(matrix, SIGNED_PERMUTATION_ATOL_FLOAT32) is not None

    def test_remap_region_is_the_index_image_of_apply_remap(self):
        import itertools

        import torch
        from konfai.data.geometry import (
            SIGNED_PERMUTATION_ATOL_FLOAT64,
            apply_remap,
            invert_remap,
            remap_region,
            remap_shape,
            signed_permutation,
        )

        rng = np.random.default_rng(3)
        volume = torch.from_numpy(rng.standard_normal((2, 5, 6, 7)).astype(np.float32))
        axes = np.eye(3)
        for order in itertools.permutations(range(3)):
            for signs in itertools.product((1.0, -1.0), repeat=3):
                matrix = np.stack([axes[axis] * sign for axis, sign in zip(order, signs, strict=True)], axis=1)
                remap = signed_permutation(matrix, SIGNED_PERMUTATION_ATOL_FLOAT64)
                assert remap is not None
                out = apply_remap(volume, remap)
                assert list(out.shape[1:]) == remap_shape([5, 6, 7], remap)
                # The region contract: reading the remapped source region and remapping IT
                # reproduces the target patch exactly, mirrors included ([n - stop, n - start)).
                target = tuple(slice(1, extent - 1) for extent in out.shape[1:])
                source = remap_region(target, [5, 6, 7], remap)
                torch.testing.assert_close(
                    apply_remap(volume[(slice(None), *source)], remap),
                    out[(slice(None), *target)],
                    rtol=0,
                    atol=0,
                )
                # The inverse remap undoes the forward, extents and values alike.
                back = apply_remap(out, invert_remap(remap))
                torch.testing.assert_close(back, volume, rtol=0, atol=0)

    def test_apply_remap_materialises_the_copy(self):
        import torch
        from konfai.data.geometry import apply_remap

        volume = torch.arange(8.0).reshape(1, 2, 4)
        # Even the identity remap with no mirror is a copy: a remapped copy may be handed on while
        # the source tensor lives its own life.
        out = apply_remap(volume, [(0, False), (1, False)])
        assert out.data_ptr() != volume.data_ptr()
        torch.testing.assert_close(out, volume, rtol=0, atol=0)
