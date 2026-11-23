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

"""Other libraries' resampling, to the bit: scipy's zoom and scikit-image's resize (``konfai.data.spline``), the grid
rules of ``Resample(grid=...)``, ``interpolation: nearest_even`` (torch's ``grid_sample``) and
``Canonical(resample=False)`` (nibabel's ``as_closest_canonical``). Each reference library is optional: its test
skips where it is not installed."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from konfai.data import spline
from konfai.data.transform import Canonical, LocalityKind, Resample
from konfai.utils.dataset import Attribute
from konfai.utils.errors import TransformError
from oracle_support import assert_same_as_scipy, geometry

_RNG = np.random.default_rng(7)
_VOLUME = (_RNG.normal(size=(13, 17, 11)) * 100).astype(np.float64)


def _ndi():
    return pytest.importorskip("scipy.ndimage")


@pytest.mark.parametrize("order", [0, 1, 3])
@pytest.mark.parametrize("corners", [False, True], ids=["extent", "corners"])
@pytest.mark.parametrize("shape", [(20, 9, 11), (7, 30, 5), (1, 4, 30)])
def test_zoom_is_scipys_to_the_bit(order: int, corners: bool, shape: tuple[int, ...]) -> None:
    ndi = _ndi()
    want = ndi.zoom(_VOLUME, np.divide(shape, _VOLUME.shape), order=order, mode="nearest", grid_mode=not corners)
    assert_same_as_scipy(spline.zoom(_VOLUME, shape, order, corners), want)


@pytest.mark.parametrize("shape", [(20, 9, 11), (7, 30, 5)])
def test_resize_is_skimages_to_the_bit(shape: tuple[int, ...]) -> None:
    resize = pytest.importorskip("skimage.transform").resize
    want = resize(_VOLUME, shape, 3, mode="edge", anti_aliasing=False)
    assert_same_as_scipy(spline.resize(_VOLUME, shape, 3), want)


@pytest.mark.parametrize(("shape", "order"), [((30, 50, 20), 3), ((30, 50, 20), 1)])
def test_resample_faces_resizes_each_channel_as_scikit_image(shape: tuple[int, ...], order: int) -> None:
    data = (_RNG.normal(size=(2, 12, 40, 33)) * 300).astype(np.float32)
    want = np.stack([spline.resize(channel.astype(np.float64), shape, order) for channel in data])
    np.testing.assert_array_equal(spline.resample_faces(data, shape, order, None), want)


@pytest.mark.parametrize(("shape", "axis"), [((40, 60, 66), 0), ((12, 70, 20), 2)])
def test_resample_faces_resizes_slices_alone_then_picks_the_axis_apart_by_nearest_neighbour(
    shape: tuple[int, ...], axis: int
) -> None:
    data = (_RNG.normal(size=(1, 12, 40, 33)) * 300).astype(np.float32)
    out = spline.resample_faces(data, shape, 3, axis)
    source = np.moveaxis(data[0].astype(np.float64), axis, 0)
    plane = tuple(extent for a, extent in enumerate(shape) if a != axis)
    for k, picked in enumerate(np.moveaxis(out[0], axis, 0)):
        # the output slice k reads the input slice nearest its centre, n / m (k + 1/2) - 1/2, rounded half up
        j = min(int(np.floor(source.shape[0] / shape[axis] * (k + 0.5) - 0.5 + 0.5)), source.shape[0] - 1)
        np.testing.assert_array_equal(picked, spline.resize(source[max(j, 0)], plane, 3))


@pytest.mark.parametrize(
    ("current", "new", "apart"),
    [
        ((3, 1, 1), (1, 1, 1), None),  # a ratio of 3 is not over the threshold
        ((1, 1, 1), (5, 1, 1), 0),  # judged on the second spacing when the first is isotropic
        ((0.5, 0.5, 0.5), (1, 1, 1), None),
        ((4, 1, 4), (1, 1, 1), None),  # two coarsest axes: none apart
        ((1, 1, 6), (1, 1, 1), 2),
    ],
)
def test_the_axis_resampled_apart_is_the_only_one_over_three_times_the_finest(
    current: tuple, new: tuple, apart
) -> None:
    assert spline.separate_axis(list(current), list(new)) == apart


# A case as SimpleITK reads it: (z, y, x), spacing (x, y, z).
_ANISOTROPIC = ([0.7, 0.7, 4.0], (12, 40, 33))
_PELVIS = ([1.0, 1.0, 2.5], (16, 40, 33))


@pytest.mark.parametrize(
    ("case", "interpolation", "back"),
    [
        (_PELVIS, "bspline", None),
        (_ANISOTROPIC, "bspline", None),
        (_ANISOTROPIC, "linear", None),
        # the image by B-spline, the logits carried back by order 1
        (_PELVIS, "bspline", "linear"),
        (_ANISOTROPIC, "bspline", "linear"),
    ],
)
def test_a_resample_on_the_faces_grid_counts_its_voxels_and_comes_back(
    case, interpolation: str, back: str | None
) -> None:
    """``grid: faces`` counts ``round(s / s' * n)`` voxels per axis, resamples them through
    ``spline.resample_faces`` (the coarse axis apart where it is over the threshold), and its inverse brings the
    case back onto its own grid."""
    spacing, shape = case
    plans = [1.0, 1.0, 3.0]
    order = 3 if interpolation == "bspline" else 1
    data = (_RNG.normal(size=(1, *shape)) * 300).astype(np.float32)
    stage = Resample(spacing=plans, grid="faces", interpolation=interpolation, inverse_interpolation=back)
    attribute = Attribute(geometry([3.0, -2.0, 7.0], spacing, None))
    out = stage("case", torch.from_numpy(data.copy()), attribute)
    counted = tuple(round(s / p * n) for n, s, p in zip(shape, spacing[::-1], plans[::-1], strict=True))
    assert tuple(out.shape[1:]) == counted
    axis = spline.separate_axis(list(spacing[::-1]), list(plans[::-1]))
    if interpolation == "bspline" or axis is not None:
        np.testing.assert_array_equal(out.numpy(), spline.resample_faces(data, counted, order, axis).astype(np.float32))
    restored = stage.inverse("case", out.clone(), attribute)
    assert tuple(restored.shape[1:]) == shape


def test_a_spline_resample_declares_the_whole_volume_and_a_resample_onto_its_own_spacing_streams() -> None:
    def kind(stage: Resample, spacing: list[float]) -> LocalityKind:
        return stage.patch_locality(Attribute(geometry([0.0] * 3, spacing, None))).kind

    assert kind(Resample(spacing=[1.0, 1.0, 3.0], grid="faces", interpolation="bspline"), [1.0, 1.0, 2.5]) == (
        LocalityKind.WHOLE_VOLUME
    )
    assert kind(Resample(spacing=[1.0, 1.0, 3.0], grid="faces", interpolation="bspline"), [1.0, 1.0, 3.0]) == (
        LocalityKind.REGRID
    )
    # Linear on the faces grid is ITK's linear, until an axis is resampled apart.
    assert kind(Resample(spacing=[1.0, 1.0, 3.0], grid="faces"), [1.0, 1.0, 2.5]) == LocalityKind.REGRID
    assert kind(Resample(spacing=[1.0, 1.0, 1.0], grid="faces"), [0.7, 0.7, 4.0]) == LocalityKind.WHOLE_VOLUME


def test_zoom_counts_and_samples_as_scipy_and_records_the_zoomed_spacing() -> None:
    """``ndimage.zoom`` (corners aligned) by ``s / s'``, the header's
    columns divided by that zoom, voxel 0 where it was."""
    ndi = _ndi()
    spacing, wanted = np.array([0.79, 0.79, 2.5]), np.array([1.5, 1.5, 1.5])
    data = (_RNG.normal(size=(1, 16, 40, 33)) * 300).astype(np.float32)
    factor = (spacing / wanted)[::-1]

    stage = Resample(spacing=wanted.tolist(), grid="zoom", interpolation="bspline")
    attribute = Attribute(geometry([3.0, -2.0, 7.0], spacing.tolist(), None))
    out = stage("case", torch.from_numpy(data.copy()), attribute)
    want = ndi.zoom(data[0].astype(np.float64), factor, order=3, mode="nearest").astype(np.float32)
    np.testing.assert_array_equal(out[0].numpy(), want)
    np.testing.assert_array_equal(attribute.get_np_array("Spacing"), spacing / (spacing / wanted))
    np.testing.assert_array_equal(attribute.get_np_array("Origin"), [3.0, -2.0, 7.0])

    # Linear streams through ITK's sampler on the grid zoom sampled, not on the header it records.
    linear = Resample(spacing=wanted.tolist(), grid="zoom")
    case = Attribute(geometry([3.0, -2.0, 7.0], spacing.tolist(), None))
    out = linear("case", torch.from_numpy(data.copy()), case)
    want = ndi.zoom(data[0].astype(np.float64), factor, order=1, mode="nearest")
    np.testing.assert_allclose(out[0].numpy(), want, rtol=0, atol=1e-3)
    back = linear.inverse("case", out.clone(), case)
    want_back = ndi.zoom(want, np.divide(data.shape[1:], want.shape), order=1, mode="nearest")
    np.testing.assert_allclose(back[0].numpy(), want_back, rtol=0, atol=1e-3)


def test_the_centers_grid_counts_round_n_minus_one_s_over_s_prime_plus_one() -> None:
    spacing = [0.79, 0.85, 2.5]
    stage = Resample(spacing=[1.5, 1.5, 1.5], grid="centers")
    attribute = Attribute(geometry([3.0, -2.0, 7.0], spacing, None))
    shape = stage.transform_shape("Volume", "case", [16, 40, 33], attribute)
    assert shape == [int(np.round((n - 1) * s / 1.5 + 1.0)) for n, s in zip([16, 40, 33], spacing[::-1], strict=True)]


def test_nearest_even_picks_as_grid_sample() -> None:
    """``grid_sample(mode='nearest')`` sends ties to the even voxel: at half the spacing every other voxel of the new
    grid lies EXACTLY between two of the case's."""
    labels = torch.from_numpy(_RNG.integers(0, 200, size=(1, 5, 9, 17)).astype(np.uint8))
    stage = Resample(spacing=[1.0, 1.0, 1.0], grid="centers", interpolation="nearest_even", inverse=False)
    out = stage("case", labels.clone(), Attribute(geometry([0.0] * 3, [2.0, 2.0, 2.0], None)))

    # The source index of target voxel j is j / 2: normalised for align_corners=True, exactly, as the extents
    # minus one are powers of two.
    extents = zip(out.shape[1:], labels.shape[1:], strict=True)
    axes = [torch.arange(m, dtype=torch.float64) / 2 * 2 / (n - 1) - 1 for m, n in extents]
    z, y, x = torch.meshgrid(*axes, indexing="ij")
    grid = torch.stack([x, y, z], dim=-1)[None]
    want = torch.nn.functional.grid_sample(labels[None].double(), grid, mode="nearest", align_corners=True)[0]
    torch.testing.assert_close(out, want.to(torch.uint8), rtol=0, atol=0)
    # ITK's nearest takes the other voxel of every tie.
    up = Resample(spacing=[1.0, 1.0, 1.0], grid="centers", interpolation="nearest", inverse=False)
    assert not torch.equal(up("case", labels.clone(), Attribute(geometry([0.0] * 3, [2.0, 2.0, 2.0], None))), out)


def test_nearest_even_streams_what_the_whole_volume_holds() -> None:
    from test_resample import _inverse_streamed

    labels = torch.from_numpy(_RNG.integers(0, 200, size=(1, 9, 17, 13)).astype(np.uint8))
    stage = Resample(spacing=[2.0, 2.0, 2.0], grid="centers", interpolation="nearest_even")
    case = Attribute(geometry([1.0, 2.0, 3.0], [1.0, 1.0, 1.0], None))
    held = stage("case", labels.clone(), case)
    whole = stage.inverse("CASE", held.clone(), Attribute(case))
    torch.testing.assert_close(_inverse_streamed(stage, held, case, rows=2), whole, rtol=0, atol=0)


def test_canonical_without_resampling_is_nibabels_closest_canonical() -> None:
    """A CT tilted 12.5 degrees about x: permuted and flipped onto the nearest RAS axes, never resampled, its
    direction as oblique as it was. The same array and the same affine as ``nibabel.as_closest_canonical``."""
    nib = pytest.importorskip("nibabel")
    tilt = np.deg2rad(12.5)
    rotation = np.array([[1.0, 0, 0], [0, np.cos(tilt), -np.sin(tilt)], [0, np.sin(tilt), np.cos(tilt)]])
    permuted = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    direction = rotation @ permuted  # LPS, as SimpleITK writes it
    origin, spacing = np.array([-11.3, 5.9, 3.3]), np.array([0.77, 1.31, 2.05])
    volume = _RNG.normal(size=(1, 7, 9, 11)).astype(np.float32)
    attribute = Attribute(geometry(origin.tolist(), spacing.tolist(), direction))

    out = Canonical(resample=False)("case", torch.from_numpy(volume.copy()), attribute)

    lps_to_ras = np.diag([-1.0, -1.0, 1.0])
    affine = np.eye(4)
    affine[:3, :3] = lps_to_ras @ direction @ np.diag(spacing)
    affine[:3, 3] = lps_to_ras @ origin
    want = nib.as_closest_canonical(nib.Nifti1Image(volume[0].transpose(2, 1, 0), affine))
    np.testing.assert_array_equal(out[0].numpy().transpose(2, 1, 0), np.asarray(want.dataobj))
    got = np.eye(4)
    written = attribute.get_np_array("Direction").reshape(3, 3) @ np.diag(attribute.get_np_array("Spacing"))
    got[:3, :3] = lps_to_ras @ written
    got[:3, 3] = lps_to_ras @ attribute.get_np_array("Origin")
    # KonfAI keeps a header in float32, as NIfTI does.
    np.testing.assert_allclose(got, want.affine, rtol=0, atol=1e-5)

    stage = Canonical(resample=False)
    assert stage.patch_locality(Attribute(geometry(origin.tolist(), spacing.tolist(), direction))).kind == (
        LocalityKind.ORIENTATION
    )
    case = Attribute(geometry(origin.tolist(), spacing.tolist(), direction))
    back = stage.inverse("case", stage("case", torch.from_numpy(volume.copy()), case), case)
    np.testing.assert_array_equal(back.numpy(), volume)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"spacing": [1.0, 1.0, 1.0], "grid": "elastix"},
        {"shape": [4, 4, 4], "grid": "faces"},
        {"spacing": [1.0, 1.0, 1.0], "grid": "centers", "interpolation": "bspline"},
        {"spacing": [1.0, 1.0, 1.0], "align": "origin", "interpolation": "bspline"},
        {"spacing": [1.0, 1.0, 1.0], "transforms": {"reg": False}, "interpolation": "bspline"},
    ],
)
def test_a_rule_or_a_spline_it_cannot_honour_is_refused(kwargs: dict) -> None:
    with pytest.raises(TransformError):
        Resample(**kwargs)


def test_a_map_on_the_faces_grid_is_refused() -> None:
    # The faces grid resamples an anisotropic axis apart, through konfai.data.spline on the source's own axes: a
    # map there would be dropped without a word.
    with pytest.raises(TransformError, match="faces grid"):
        Resample(spacing=[1, 1, 1], grid="faces", interpolation="linear", transforms={"T": False})


def test_growing_a_one_voxel_axis_on_the_zoom_grid_is_refused_by_name() -> None:
    # scipy's zoom copies the voxel along such an axis, on a grid with no extent there, which no inverse undoes.
    attribute = Attribute()
    attribute["Spacing"], attribute["Origin"], attribute["Direction"] = (
        np.array([1.0, 1.0, 4.0]),
        np.zeros(3),
        np.eye(3).flatten(),
    )
    with pytest.raises(TransformError, match="one-voxel axis of case 'c'"):
        Resample(spacing=[1, 1, 1], grid="zoom", interpolation="linear").transform_shape("G", "c", [1, 5, 6], attribute)


@pytest.mark.parametrize(
    "settings",
    [
        {"grid": "faces", "interpolation": "linear"},
        {"grid": "faces", "interpolation": "bspline"},
        {"grid": "zoom", "interpolation": "linear"},
        {"grid": "zoom", "interpolation": "bspline"},
    ],
)
def test_the_numpy_routes_take_bfloat16_as_float32_and_give_it_back(settings) -> None:
    # NumPy has no bfloat16: those routes read it as float32 and cast back, logits on the way back included.
    image = torch.from_numpy(_RNG.normal(size=(2, 7, 11, 9)).astype(np.float32))
    stage = Resample(spacing=[1.5, 1.5, 2.0], inverse_interpolation="linear", **settings)
    want = stage("case", image.clone(), Attribute(geometry([0.0] * 3, [1.0, 1.0, 1.0], None)))
    case = Attribute(geometry([0.0] * 3, [1.0, 1.0, 1.0], None))
    got = stage("case", image.to(torch.bfloat16), case)
    assert got.dtype == torch.bfloat16
    torch.testing.assert_close(got, want.to(torch.bfloat16), rtol=0, atol=0.05)
    back = stage.inverse("case", got.clone(), Attribute(case))
    assert back.dtype == torch.bfloat16 and tuple(back.shape) == tuple(image.shape)
