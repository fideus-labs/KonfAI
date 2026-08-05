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

"""What a decoded transform's bound must satisfy, against SimpleITK.

Two properties, and both matter. CONTAINMENT: the true map never leaves the bound — a bound that
is short reads a source window the resample then samples outside of, which returns background
rather than failing. NON-VACUITY: the bound is not the whole world — a bound that contains
everything contains the truth and buys nothing.
"""

import numpy as np
import pytest
from konfai.data.geometry import AffineStage, DisplacementStage, bound_of
from konfai.utils.errors import TransformError

sitk = pytest.importorskip("SimpleITK")

from konfai.utils.ITK import decode_transform_stages, invert_stages  # noqa: E402

SIZE = (32, 40, 48)


def _image(oblique: bool = True) -> "sitk.Image":
    image = sitk.GetImageFromArray(np.zeros(SIZE, np.float32))
    image.SetSpacing((0.8, 1.2, 1.5))
    image.SetOrigin((10.0, -5.0, 2.0))
    if oblique:
        a, b = np.deg2rad(20.0), np.deg2rad(15.0)
        rz = np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])
        ry = np.array([[np.cos(b), 0.0, np.sin(b)], [0.0, 1.0, 0.0], [-np.sin(b), 0.0, np.cos(b)]])
        image.SetDirection(tuple((rz @ ry).ravel()))
    return image


def _euler(image: "sitk.Image") -> "sitk.Transform":
    transform = sitk.Euler3DTransform()
    transform.SetCenter(image.TransformContinuousIndexToPhysicalPoint([(s - 1) / 2 for s in image.GetSize()]))
    transform.SetRotation(0.11, -0.2, 0.31)
    transform.SetTranslation((3.0, -2.0, 1.0))
    return transform


def _affine(image: "sitk.Image") -> "sitk.Transform":
    transform = sitk.AffineTransform(3)
    transform.SetCenter(image.TransformContinuousIndexToPhysicalPoint([(s - 1) / 2 for s in image.GetSize()]))
    transform.SetMatrix(np.array([[1.1, 0.05, 0.0], [0.0, 0.9, 0.07], [0.02, 0.0, 1.2]]).ravel())
    transform.SetTranslation((4.0, -3.0, 2.0))
    return transform


def _bspline(image: "sitk.Image", mesh: int = 6, amplitude: float = 9.0) -> "sitk.Transform":
    transform = sitk.BSplineTransformInitializer(image, [mesh] * 3, 3)
    size = np.asarray(transform.GetParameters()).size
    transform.SetParameters(list(np.random.RandomState(1).uniform(-amplitude, amplitude, size)))
    return transform


def _field(image: "sitk.Image") -> "sitk.Transform":
    filt = sitk.TransformToDisplacementFieldFilter()
    filt.SetReferenceImage(image)
    return sitk.DisplacementFieldTransform(sitk.Cast(filt.Execute(_bspline(image)), sitk.sitkVectorFloat64))


def _world_points(image: "sitk.Image", count: int = 4000, overshoot: float = 6.0) -> np.ndarray:
    """Points across the grid AND past its edges — the bound must hold everywhere, not just inside."""
    rng = np.random.RandomState(0)
    index = np.stack([rng.uniform(-overshoot, extent + overshoot, count) for extent in image.GetSize()], axis=1)
    return np.array([image.TransformContinuousIndexToPhysicalPoint(list(point)) for point in index])


def _cases(image: "sitk.Image") -> list[tuple[str, "sitk.Transform"]]:
    return [
        ("euler", _euler(image)),
        ("affine", _affine(image)),
        ("bspline coarse", _bspline(image, mesh=4)),
        ("bspline fine", _bspline(image, mesh=12)),
        ("field", _field(image)),
        ("composite affine+bspline", sitk.CompositeTransform([_affine(image), _bspline(image)])),
        ("composite affine+euler", sitk.CompositeTransform([_affine(image), _euler(image)])),
    ]


@pytest.mark.parametrize("oblique", [False, True], ids=["axis-aligned", "oblique"])
def test_the_bound_contains_the_map_it_bounds(oblique: bool):
    image = _image(oblique)
    world = _world_points(image)
    for label, transform in _cases(image):
        bound = bound_of(decode_transform_stages(transform), 3)
        truth = np.array([transform.TransformPoint(point) for point in world])
        excess = np.abs(truth - bound.affine.apply(world)) - bound.residual_xyz
        assert excess.max() <= 1e-9, f"{label}: the bound is short by {excess.max():.4g} mm"


@pytest.mark.parametrize("oblique", [False, True], ids=["axis-aligned", "oblique"])
def test_the_bound_is_not_vacuous(oblique: bool):
    # Containment alone is satisfied by an infinite box. The residual must stay within a small
    # multiple of the displacement actually reached, or the halo it sizes is the whole volume.
    image = _image(oblique)
    world = _world_points(image)
    for label, transform in _cases(image):
        bound = bound_of(decode_transform_stages(transform), 3)
        if not bound.residual_xyz.any():
            continue
        truth = np.array([transform.TransformPoint(point) for point in world])
        reached = np.abs(truth - bound.affine.apply(world)).max(axis=0)
        assert (bound.residual_xyz <= 4.0 * np.maximum(reached, 1e-6)).all(), f"{label}: bound far too loose"


def test_a_linear_transform_is_bounded_exactly():
    image = _image()
    world = _world_points(image)
    for transform in (_euler(image), _affine(image), sitk.TranslationTransform(3, (5.0, -1.0, 2.0))):
        bound = bound_of(decode_transform_stages(transform), 3)
        np.testing.assert_array_equal(bound.residual_xyz, np.zeros(3))
        truth = np.array([transform.TransformPoint(point) for point in world])
        np.testing.assert_allclose(bound.affine.apply(world), truth, rtol=0.0, atol=1e-9)


def test_probing_the_affine_part_of_a_bspline_is_worse_than_the_structural_one():
    # Why the affine part is read structurally and never by finite differences. Probing a NON-linear
    # map measures a local gradient and extrapolates it across the whole grid: around an interior
    # point of this spline that implies a residual of ~19 mm where the sup-norm of the coefficients
    # says 9. A design that probes pays for a halo twice as wide and still has no theorem behind it.
    image = _image()
    transform = _bspline(image)
    world = _world_points(image)
    structural = bound_of(decode_transform_stages(transform), 3)

    centre = np.array(image.TransformContinuousIndexToPhysicalPoint([(s - 1) / 2 for s in image.GetSize()]))
    base = np.array(transform.TransformPoint(tuple(centre)))
    columns = [np.array(transform.TransformPoint(tuple(centre + np.eye(3)[k]))) - base for k in range(3)]
    probed = np.stack(columns, axis=1)
    truth = np.array([transform.TransformPoint(point) for point in world])
    probed_residual = np.abs(truth - (world @ probed.T + (base - probed @ centre))).max(axis=0)
    assert (probed_residual > structural.residual_xyz).any()


def test_the_bound_is_a_theorem_and_the_sampled_maximum_is_not():
    # The sup-norm bound must dominate what any sample can reach — that is the whole claim. A design
    # that sized its halo from a dense sample would be under a bound it never proved.
    image = _image()
    transform = _bspline(image)
    world = _world_points(image, count=20000)
    bound = bound_of(decode_transform_stages(transform), 3)
    truth = np.array([transform.TransformPoint(point) for point in world])
    assert np.all(np.abs(truth - world).max(axis=0) <= bound.residual_xyz + 1e-9)


class TestRefusals:
    def test_a_transform_that_decomposes_into_nothing_is_refused_by_name(self):
        class Opaque:
            def GetDimension(self):
                return 3

            def GetName(self):
                return "MadeUpTransform"

            def IsLinear(self):
                return False

        with pytest.raises(TransformError, match="MadeUpTransform"):
            decode_transform_stages(Opaque())

    def test_a_non_finite_field_is_refused_rather_than_bounded(self):
        image = _image()
        filt = sitk.TransformToDisplacementFieldFilter()
        filt.SetReferenceImage(image)
        array = sitk.GetArrayFromImage(filt.Execute(_bspline(image)))
        array[0, 0, 0, 0] = np.nan
        field = sitk.GetImageFromArray(array, isVector=True)
        field.CopyInformation(image)
        transform = sitk.DisplacementFieldTransform(sitk.Cast(field, sitk.sitkVectorFloat64))
        with pytest.raises(TransformError, match="non-finite"):
            decode_transform_stages(transform)


class TestCompositeOrder:
    def test_the_stages_are_decoded_in_application_order(self):
        # SimpleITK applies a CompositeTransform's list in REVERSE (last added runs first), while
        # GetNthTransform(0) is the first added. Folding a bound in list order composes the wrong
        # way round; the decoder normalizes it, and this is what pins that.
        translate = sitk.TranslationTransform(3, (100.0, 0.0, 0.0))
        scale = sitk.ScaleTransform(3, (2.0, 2.0, 2.0))
        composite = sitk.CompositeTransform([translate, scale])
        point = np.array([1.0, 1.0, 1.0])
        np.testing.assert_allclose(composite.TransformPoint(tuple(point)), [102.0, 2.0, 2.0])

        stages = decode_transform_stages(composite)
        folded = bound_of(stages, 3)
        np.testing.assert_allclose(folded.affine.apply(point), [102.0, 2.0, 2.0], rtol=0.0, atol=1e-9)

    def test_the_residual_is_transported_through_an_outer_scale(self):
        image = _image(oblique=False)
        spline = _bspline(image, mesh=5, amplitude=5.0)
        alone = bound_of(decode_transform_stages(spline), 3)
        scaled = bound_of(
            decode_transform_stages(sitk.CompositeTransform([sitk.ScaleTransform(3, (3.0,) * 3), spline])), 3
        )
        # The spline runs first, then the scale: its reach is multiplied by 3, not left alone.
        np.testing.assert_allclose(scaled.residual_xyz, 3.0 * alone.residual_xyz, rtol=1e-9)


class TestInverse:
    def test_an_all_affine_map_inverts_algebraically(self):
        image = _image()
        transform = sitk.CompositeTransform([_affine(image), _euler(image)])
        inverse = invert_stages(decode_transform_stages(transform), 3)
        assert inverse is not None
        forward = bound_of(decode_transform_stages(transform), 3).affine
        back = bound_of(inverse, 3).affine
        points = _world_points(image, count=200)
        np.testing.assert_allclose(back.apply(forward.apply(points)), points, rtol=0.0, atol=1e-6)

    @pytest.mark.parametrize("factory", [_bspline, _field], ids=["bspline", "field"])
    def test_a_non_affine_map_has_no_algebraic_inverse(self, factory):
        assert invert_stages(decode_transform_stages(factory(_image())), 3) is None


class TestDisplacementStage:
    def test_a_bspline_decodes_to_its_coefficient_grid(self):
        image = _image()
        stages = decode_transform_stages(_bspline(image, mesh=7))
        assert len(stages) == 1
        stage = stages[0]
        assert isinstance(stage, DisplacementStage)
        assert stage.order == 3
        # The bound IS the sup-norm of the coefficients, per component.
        np.testing.assert_allclose(stage.bound_xyz, np.abs(stage.values.reshape(3, -1)).max(axis=1))

    def test_a_field_decodes_to_order_one_on_its_own_grid(self):
        image = _image()
        stages = decode_transform_stages(_field(image))
        stage = stages[0]
        assert isinstance(stage, DisplacementStage)
        assert stage.order == 1
        assert stage.grid.size_zyx == SIZE

    def test_an_affine_stage_carries_no_residual(self):
        stages = decode_transform_stages(_euler(_image()))
        assert isinstance(stages[0], AffineStage)
        np.testing.assert_array_equal(stages[0].bound().residual_xyz, np.zeros(3))

    def test_the_end_plane_of_a_bsplines_valid_region_is_warped_as_itk_warps_it(self):
        """ITK admits a continuous index ON the valid-region end (``InsideValidRegion`` nudges it
        back inside), so identity there is wrong bytes. The support slides one control point down,
        which changes no value — the outermost tap's weight is exactly zero at an integer offset.
        The regression this pins: that plane used to fail the inside test, and a grid commensurate
        with its coefficient mesh hits it in whole planes at a time, every voxel silently unmoved.
        """
        import torch
        from konfai.data.sampling import _displacement_at

        transform = _bspline(_image(oblique=False), mesh=5)
        stage = decode_transform_stages(transform)[0]
        assert isinstance(stage, DisplacementStage)
        extent = np.asarray(list(reversed(stage.grid.size_zyx)), dtype=np.float64)  # (x, y, z)
        rng = np.random.RandomState(2)
        index = rng.uniform(1.0, extent - 2.0 - 1e-6, size=(60, 3))
        for axis in range(3):  # each axis in turn pinned exactly on its end plane
            index[axis::3, axis] = extent[axis] - 2.0
        world = stage.grid.index_to_world.apply(index)
        got = _displacement_at(stage, torch.from_numpy(world), torch.device("cpu")).numpy()
        truth = np.array([np.asarray(transform.TransformPoint(tuple(point))) - point for point in world])
        np.testing.assert_allclose(got, truth, rtol=0, atol=1e-12)
        assert np.abs(truth).max() > 0.1, "the fixture must displace the end planes, or this pins nothing"
