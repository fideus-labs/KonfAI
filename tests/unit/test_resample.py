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

"""One ``Resample``: which grid to write on, what map to write it through, and what that fixed.

``ResampleToResolution``, ``ResampleToShape``, ``ResampleToReference``, ``ResampleTransform`` and
``Warp`` were five stages answering two questions between them, each with its own sampler and its
own idea of where a voxel is. They are now five spellings of this one, and the tests here are for
what only became checkable once there was a single answer to check.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from konfai.data.transform import (
    Resample,
    ResampleToReference,
    ResampleToResolution,
    ResampleToShape,
    ResampleTransform,
    Warp,
)
from konfai.utils.dataset import Attribute
from konfai.utils.errors import TransformError

sitk = pytest.importorskip("SimpleITK")

_ORIGIN, _SPACING = [11.5, -3.25, 40.0], [1.3, 0.9, 0.9]
_SHAPE = (24, 30, 32)


def _attributes(origin=None, spacing=None, direction=None) -> Attribute:
    attribute = Attribute()
    attribute["Origin"] = np.asarray(_ORIGIN if origin is None else origin, dtype=np.float64)
    attribute["Spacing"] = np.asarray(_SPACING if spacing is None else spacing, dtype=np.float64)
    attribute["Direction"] = (np.eye(3) if direction is None else direction).reshape(-1)
    return attribute


def _volume(shape=_SHAPE, seed: int = 0) -> np.ndarray:
    # Noise, not a smooth field: a smooth volume resampled onto a grid that is half a voxel off is
    # still nearly right, so a smooth fixture would pass a map that is wrong by exactly the amount
    # this file exists to catch.
    return np.random.default_rng(seed).normal(size=shape).astype(np.float32)[None]


def _as_image(volume: np.ndarray, attribute: Attribute) -> sitk.Image:
    image = sitk.GetImageFromArray(volume[0])
    image.SetOrigin(attribute.get_np_array("Origin").tolist())
    image.SetSpacing(attribute.get_np_array("Spacing").tolist())
    image.SetDirection(attribute.get_np_array("Direction").tolist())
    return image


# ------------------------------------------------------------------ one class, five spellings


@pytest.mark.parametrize(
    ("alias", "unified"),
    [
        (lambda: ResampleToResolution(spacing=[2.0, 1.5, 1.5]), lambda: Resample(spacing=[2.0, 1.5, 1.5])),
        (lambda: ResampleToShape(shape=[12, 20, 22]), lambda: Resample(shape=[12, 20, 22])),
        (lambda: ResampleToShape(shape=[0, 20, 0]), lambda: Resample(shape=[0, 20, 0])),
    ],
)
def test_a_spelling_and_the_unified_stage_are_the_same_stage(alias, unified) -> None:
    """The published names are argument translations, not behaviour of their own.

    Kept because they appear in shipped configs and in every bundle on the hub -- and kept THIN,
    because a spelling that carries logic is a second implementation waiting to drift.
    """
    volume = torch.from_numpy(_volume())
    left, right = alias(), unified()
    assert isinstance(left, Resample)
    assert left.apply_inverse == right.apply_inverse

    got = left("case", volume.clone(), _attributes())
    want = right("case", volume.clone(), _attributes())
    torch.testing.assert_close(got, want, rtol=0, atol=0)


def test_every_spelling_is_the_one_class() -> None:
    for stage in (
        ResampleToResolution(),
        ResampleToShape(),
        ResampleToReference(entry="x"),
        ResampleTransform(transforms={"reg": False}),
        Warp(field="./x:h5", group="DVF"),
    ):
        assert isinstance(stage, Resample)


def test_the_three_ways_to_name_a_target_grid_are_exclusive() -> None:
    with pytest.raises(TransformError, match="three ways to say the same thing"):
        Resample(spacing=[1.0, 1.0, 1.0], shape=[4, 4, 4])


# ------------------------------------------------------------------ where the new grid sits


def test_extent_alignment_keeps_the_field_of_view_and_origin_alignment_keeps_voxel_zero() -> None:
    """The one silent choice in the family, made explicit.

    ``extent`` makes the outer faces coincide, which is ``F.interpolate``'s map and what KonfAI has
    always done; ``origin`` keeps voxel zero's centre where it is, which is what resampling onto a
    grid that shares an origin does. A quarter of a voxel of anatomy separates them.
    """
    attribute = _attributes()
    volume = torch.from_numpy(_volume())

    Resample(spacing=[2.0, 1.5, 1.5], align="extent")("case", volume.clone(), attribute)
    extent_origin = attribute.get_np_array("Origin").copy()
    extent_spacing = attribute.get_np_array("Spacing").copy()

    attribute = _attributes()
    Resample(spacing=[2.0, 1.5, 1.5], align="origin")("case", volume.clone(), attribute)

    np.testing.assert_allclose(attribute.get_np_array("Origin"), _ORIGIN)
    np.testing.assert_allclose(attribute.get_np_array("Spacing"), [2.0, 1.5, 1.5])
    # Extent alignment puts voxel zero half the spacing change away, on every axis.
    np.testing.assert_allclose(extent_origin, np.asarray(_ORIGIN) + 0.5 * (extent_spacing - np.asarray(_SPACING)))


def test_an_unknown_alignment_is_refused_at_construction() -> None:
    with pytest.raises(TransformError, match="unknown align"):
        Resample(spacing=[1.0, 1.0, 1.0], align="corners")


# ------------------------------------------------------------------ the header describes the data


@pytest.mark.parametrize(
    "kwargs",
    [
        {"spacing": [2.0, 1.5, 1.5]},
        {"spacing": [2.0, 1.5, 1.5], "align": "origin"},
        {"shape": [12, 20, 22]},
        {"spacing": [0.9, 1.7, 1.1]},
    ],
)
def test_the_recorded_header_describes_the_grid_that_was_actually_sampled(kwargs) -> None:
    """The check neither predecessor could pass, because neither wrote a placement at all.

    ``ResampleToResolution`` recorded the spacing that was ASKED FOR while sampling at ``n_in/n_out``
    times the source's -- up to a millimetre of drift across a volume -- and left the Origin alone
    while sampling half a spacing-change away from it. Nothing downstream could see either: the
    voxels are all real, and the header is the only witness.

    So the oracle is built FROM THE RECORDED HEADER. If the two disagree, resampling the source onto
    the grid the header claims cannot reproduce the voxels the stage returned.
    """
    volume = _volume()
    attribute = _attributes()
    got = Resample(**kwargs)("case", torch.from_numpy(volume.copy()), attribute).numpy()[0]

    grid = sitk.Image(*reversed(got.shape), sitk.sitkFloat32)
    grid.SetOrigin(attribute.get_np_array("Origin").tolist())
    grid.SetSpacing(attribute.get_np_array("Spacing").tolist())
    grid.SetDirection(attribute.get_np_array("Direction").tolist())
    want = sitk.GetArrayFromImage(
        sitk.Resample(_as_image(volume, _attributes()), grid, sitk.Transform(), sitk.sitkLinear, 0.0)
    )
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-4)


def test_an_oblique_case_records_an_oblique_header() -> None:
    """A direction is carried through, not quietly dropped -- and the data follows it."""
    angle = np.deg2rad(23.0)
    cos, sin = float(np.cos(angle)), float(np.sin(angle))
    direction = np.asarray([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])
    volume = _volume()
    attribute = _attributes(direction=direction)
    got = Resample(spacing=[2.0, 1.5, 1.5])("case", torch.from_numpy(volume.copy()), attribute).numpy()[0]

    np.testing.assert_allclose(attribute.get_np_array("Direction").reshape(3, 3), direction)
    grid = sitk.Image(*reversed(got.shape), sitk.sitkFloat32)
    grid.SetOrigin(attribute.get_np_array("Origin").tolist())
    grid.SetSpacing(attribute.get_np_array("Spacing").tolist())
    grid.SetDirection(direction.reshape(-1).tolist())
    want = sitk.GetArrayFromImage(
        sitk.Resample(_as_image(volume, _attributes(direction=direction)), grid, sitk.Transform(), sitk.sitkLinear, 0.0)
    )
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-4)


# ------------------------------------------------------------------ an image and its label map


def test_a_label_map_lands_on_the_same_voxels_as_the_image_beside_it() -> None:
    """The bug that had no symptom: a mask resampled with its CT came out shifted against it.

    ``F.interpolate``'s nearest reads ``floor(o * scale)`` where its linear reads
    ``scale * (o + 0.5) - 0.5`` -- so the label map lagged the image of the SAME stage by
    ``(scale - 1) / 2`` source voxels. At 0.5 mm resampled to 3 mm that is 2.5 source voxels, 1.25 mm
    of anatomy, and both volumes are entirely plausible on their own.

    A ramp makes it visible: linear interpolation of a linear function is exact, so the resampled
    image IS the continuous source coordinate, and the resampled label map must be its rounding.
    """
    extent = 48
    ramp = np.broadcast_to(np.arange(extent, dtype=np.float32).reshape(1, 1, -1), (1, 4, 4, extent))
    attribute = _attributes(origin=[0.0, 0.0, 0.0], spacing=[0.5, 1.0, 1.0])

    image = Resample(spacing=[3.0, 1.0, 1.0])("case", torch.from_numpy(np.ascontiguousarray(ramp)), attribute)
    labels = Resample(spacing=[3.0, 1.0, 1.0])(
        "case", torch.from_numpy(np.ascontiguousarray(ramp).astype(np.uint8)), _attributes([0.0] * 3, [0.5, 1.0, 1.0])
    )

    coordinate = image.numpy()[0, 0, 0]
    picked = labels.numpy()[0, 0, 0].astype(np.int64)
    # The edges clamp, so the interior is where the ramp still reads its own coordinate.
    interior = slice(1, -1)
    np.testing.assert_array_equal(picked[interior], np.floor(coordinate[interior] + 0.5).astype(np.int64))
    assert float(np.abs(picked[interior] - coordinate[interior]).max()) <= 0.5


# ------------------------------------------------------------------ the count


def test_a_spacing_that_binary_cannot_hold_does_not_lose_a_slice() -> None:
    """90 voxels of 0.7 mm re-cut at 1.5 mm is 42.0 -- and in float64 it is 41.999999999999997.

    Truncating that gives 41: one slice of anatomy dropped, and a recorded spacing that no longer
    covers what was read. The old float32 round-trip happened to land above; nothing said so.
    """
    attribute = _attributes(origin=[0.0] * 3, spacing=[0.7, 0.7, 0.7])
    shape = Resample(spacing=[1.5, 1.5, 1.5]).transform_shape("", "case", [90, 90, 90], attribute)
    assert shape == [42, 42, 42]


# ------------------------------------------------------------------ one grid change, one map


def test_a_change_of_grid_and_a_warp_are_one_interpolation(tmp_path) -> None:
    """Asked for together they compose into one coordinate per voxel, checked against sitk's own.

    Two stages would interpolate the same voxels twice, and the second pass invents none of what the
    first smoothed away -- which is the whole reason an atlas's appearance is rebuilt from native
    volumes rather than from warped ones.
    """
    from konfai.utils.dataset import Dataset

    field_shape, field_origin, field_spacing = (6, 8, 9), [10.0, -5.0, 38.0], [4.0, 3.5, 3.0]
    field = np.zeros((3, *field_shape), dtype=np.float32)
    for component, value in enumerate((1.5, -2.0, 0.75)):
        field[component] = value

    store = Dataset(tmp_path / "DVF", "h5")
    store.write("DVF", "case", field, _attributes(field_origin, field_spacing))

    volume = _volume()
    attribute = _attributes()
    stage = Resample(spacing=[2.0, 1.5, 1.5], field=str(tmp_path / "DVF") + ":h5", field_group="DVF")
    stage.set_datasets([store])
    got = stage("case", torch.from_numpy(volume.copy()), attribute).numpy()[0]

    grid = sitk.Image(*reversed(got.shape), sitk.sitkFloat32)
    grid.SetOrigin(attribute.get_np_array("Origin").tolist())
    grid.SetSpacing(attribute.get_np_array("Spacing").tolist())
    vector = sitk.GetImageFromArray(np.moveaxis(field, 0, -1).astype(np.float64), isVector=True)
    vector.SetOrigin(field_origin)
    vector.SetSpacing(field_spacing)
    want = sitk.GetArrayFromImage(
        sitk.Resample(
            _as_image(volume, _attributes()),
            grid,
            sitk.DisplacementFieldTransform(sitk.Cast(vector, sitk.sitkVectorFloat64)),
            sitk.sitkLinear,
            0.0,
        )
    )
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-3)


# ------------------------------------------------------------------ which loop runs


def test_a_map_that_factorises_takes_the_separable_loop() -> None:
    """The optimisation needs a test, or it can be lost to a refactor with everything still green.

    A grid change between axis-aligned volumes reads one axis at a time; a rotation between them, or
    a displacement, cannot and falls to the coordinate volume. Both are correct — the difference is
    43x versus 660x of ``F.interpolate`` on a CT-sized case, which no assertion about values shows.
    """
    from konfai.data.geometry import Grid
    from konfai.data.sampling import separable_source_index

    device = torch.device("cpu")
    source = Grid(_SHAPE, np.asarray(_ORIGIN), np.asarray(_SPACING), np.eye(3))
    aligned = source.resampled(spacing_xyz=np.asarray([2.0, 1.5, 1.5]))
    assert separable_source_index(aligned, source, (), device) is not None

    angle = np.deg2rad(23.0)
    cos, sin = float(np.cos(angle)), float(np.sin(angle))
    turned = Grid(
        aligned.size_zyx,
        aligned.origin_xyz,
        aligned.spacing_xyz,
        np.asarray([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]]),
    )
    assert separable_source_index(turned, source, (), device) is None, "a rotation does not factorise"

    # A flip is still axis-aligned, so it factorises -- the test that the check is not merely
    # "is the direction the identity".
    flipped = Grid(source.size_zyx, source.origin_xyz, source.spacing_xyz, np.diag([-1.0, -1.0, 1.0]))
    assert separable_source_index(aligned, flipped, (), device) is not None


def test_the_two_loops_agree_where_both_can_serve_the_same_map() -> None:
    """Different summation orders, same answer to float rounding — and the same fill, exactly."""
    from konfai.data.geometry import Grid
    from konfai.data.sampling import gather, gather_separable, separable_source_index, source_index

    device = torch.device("cpu")
    volume = torch.from_numpy(_volume())
    source = Grid(_SHAPE, np.asarray(_ORIGIN), np.asarray(_SPACING), np.eye(3))
    # Placed so part of the target reaches past the case, which is where the fill rule shows.
    target = Grid((14, 18, 20), np.asarray(_ORIGIN) - 4.0, np.asarray([2.0, 1.6, 1.6]), np.eye(3))

    axes = separable_source_index(target, source, (), device)
    assert axes is not None
    fast = gather_separable(volume, axes, [0, 0, 0], list(_SHAPE), "linear", -999.0)
    general = gather(volume, source_index(target, source, (), device), [0, 0, 0], list(_SHAPE), "linear", -999.0)

    np.testing.assert_array_equal((fast == -999.0).numpy(), (general == -999.0).numpy())
    torch.testing.assert_close(fast, general, rtol=1e-6, atol=1e-5)


def test_the_blend_order_is_the_same_for_a_region_as_for_the_whole_volume() -> None:
    """Axes are blended most-reduced-first, and the key must not be the extents in hand.

    Blending an axis reduces it before the next one reads it, so the order decides how much data
    every later pass moves -- 9x on a thick-slice CT brought to isotropic. But the order also decides
    the SUMMATION order, so a region that chose differently from the whole volume would stop being
    bit-identical to it, which is the one equality the streaming design rests on. Keyed on the two
    grids' spacings, which a region shares with its volume, and not on their extents, which it does
    not.
    """
    from konfai.data.geometry import Grid
    from konfai.data.sampling import blend_order

    source = Grid((64, 512, 512), np.zeros(3), np.asarray([0.7, 0.7, 3.0]), np.eye(3))
    target = source.resampled(spacing_xyz=np.asarray([1.0, 1.0, 1.0]))
    whole = blend_order(target, source)

    # z triples while y and x shrink, so y and x are blended first.
    assert whole == [1, 2, 0]
    for start, stop in ((0, 8), (17, 41), (target.size_zyx[0] - 3, target.size_zyx[0])):
        region = target.sub_grid((slice(start, stop), slice(0, target.size_zyx[1]), slice(0, target.size_zyx[2])))
        assert blend_order(region, source) == whole


def test_an_axis_the_map_leaves_alone_is_left_alone() -> None:
    """A resample of one axis reads the other two, it does not blend them — and says so in the values.

    ``spacing: [-1, -1, 3]`` keeps x and y exactly. Blending them anyway would be two gathers and a
    lerp over the largest tensor in flight, for a result equal to the input; skipping them has to be
    exactly that, not nearly.
    """
    attribute = _attributes(origin=[0.0] * 3, spacing=[1.0, 1.0, 1.0])
    volume = torch.from_numpy(_volume())
    kept = Resample(spacing=[-1.0, -1.0, 1.0])("case", volume.clone(), Attribute(attribute))
    torch.testing.assert_close(kept, volume, rtol=0, atol=0)
