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

"""A prompted model's prompts on a grid: ``PointsOnGrid``, ``DrawPrompts``, ``Crop`` around the prompts and pasted
back onto a group, and the ops the portable compiler writes for them."""

from pathlib import Path

import numpy as np
import pytest
import torch

sitk = pytest.importorskip("SimpleITK")

from konfai.data.transform import Crop, DrawPrompts, LocalityKind, PointsOnGrid  # noqa: E402
from konfai.data.transform.prompts import (  # noqa: E402
    BOX,
    NEGATIVE,
    POSITIVE,
    centre,
    distance_ball,
    interactions,
    offset_of,
    on_grid,
    read_prompts,
)
from konfai.portable import OP_MAP, _channels_after, _result_dtype, compile_pipeline, transform_manifest  # noqa: E402
from konfai.utils.dataset import Attribute, Dataset, write_landmarks  # noqa: E402
from konfai.utils.errors import ConfigError, TransformError  # noqa: E402

SHAPE = (10, 12, 14)  # z, y, x
CHANNELS = [BOX, "", POSITIVE, NEGATIVE, "", ""]


def _case(root: Path, voxels: list[list[int]], kinds: list[str], last: np.ndarray | None = None) -> Dataset:
    """A case P000 of a CT on an oblique grid, its prompts at the voxels (x, y, z) given, and its last mask."""
    image = sitk.GetImageFromArray(np.arange(np.prod(SHAPE), dtype=np.float32).reshape(SHAPE))
    image.SetSpacing((0.8, 1.1, 2.5))
    image.SetOrigin((-10.0, 20.0, 5.0))
    turn = np.radians(20)
    image.SetDirection((np.cos(turn), -np.sin(turn), 0, np.sin(turn), np.cos(turn), 0, 0, 0, 1))
    (root / "P000").mkdir(parents=True)
    sitk.WriteImage(image, str(root / "P000" / "CT.mha"))
    points = np.array([image.TransformIndexToPhysicalPoint(voxel) for voxel in voxels])
    write_landmarks(points, root / "P000" / "Points.fcsv", kinds)
    mask = sitk.GetImageFromArray(np.zeros(SHAPE, np.uint8) if last is None else last)
    mask.CopyInformation(image)
    sitk.WriteImage(mask, str(root / "P000" / "Last.mha"))
    return Dataset(str(root), "mha")


def _bound(stage, dataset: Dataset):
    stage.set_datasets([dataset])
    return stage


def _half(value: float) -> float:
    return float(np.float16(np.float32(value)))


def test_the_ball_is_nninteractives_distance_transform_of_scikit_images_ball() -> None:
    morphology = pytest.importorskip("skimage.morphology")
    ndimage = pytest.importorskip("scipy.ndimage")
    for radius in range(1, 9):
        distance = ndimage.distance_transform_edt(morphology.ball(radius) >= 0.5)
        assert np.array_equal(distance_ball(radius), distance / distance.max()), radius
    # The centre alone: nothing outside it to measure from.
    assert np.array_equal(distance_ball(0), np.ones((1, 1, 1)))


def test_points_land_on_simpleitks_continuous_index_in_array_order(tmp_path: Path) -> None:
    dataset = _case(tmp_path, [[3, 4, 5], [13, 0, 9]], [POSITIVE, BOX])
    points, _ = dataset.read_data("Points", "P000")
    _, header = dataset.read_data("CT", "P000")
    voxels = _bound(PointsOnGrid("CT", "none"), dataset)("P000", torch.from_numpy(points), Attribute()).numpy()
    assert np.allclose(voxels, [[5, 4, 3], [9, 0, 13]], atol=1e-9)
    # Each point takes one index per axis of the reference: three on the CT, two on a slice.
    sitk.WriteImage(sitk.Image(14, 12, sitk.sitkUInt8), str(tmp_path / "P000" / "Slice.mha"))
    for reference, rank in (("CT", 3), ("Slice", 2)):
        stage = _bound(PointsOnGrid(reference), dataset)
        assert stage.transform_shape("Points", "P000", [3], Attribute()) == [rank]
        assert stage("P000", torch.from_numpy(points), Attribute()).shape == (2, rank)
    assert np.array_equal(on_grid(points, header, 3), [[5, 4, 3], [9, 0, 13]])
    shifted = points + np.array([0.4, 0, 0])  # 0.4 mm along x: half a voxel of 0.8 mm on a turned axis
    assert np.array_equal(on_grid(shifted, header, 3, "trunc"), np.trunc(on_grid(shifted, header, 3, "none")))
    # A header without a geometry is the identity grid of the rank handed.
    assert np.array_equal(on_grid(np.array([[1.0, 2.0, 7.0]]), Attribute(), 2), [[2, 1]])
    with pytest.raises(TransformError, match="indices"):
        interactions(np.array([[np.nan, 0, 0]]), [POSITIVE], list(SHAPE))
    with pytest.raises(TransformError, match="2 or 3 axes"):
        on_grid(points, Attribute(), 4)
    with pytest.raises(TransformError, match="rounding"):
        PointsOnGrid("CT", "floor")


def test_prompts_draw_the_last_mask_the_box_and_the_balls_each_weighing_more_than_the_last(tmp_path: Path) -> None:
    last = np.zeros(SHAPE, np.uint8)
    last[2:4, 3:5, 4:6] = 1
    dataset = _case(tmp_path, [[2, 3, 6], [6, 5, 6], [5, 6, 4], [12, 11, 9]], [BOX, BOX, POSITIVE, NEGATIVE], last)
    data, attribute = dataset.read_data("CT", "P000")
    stage = _bound(DrawPrompts("Points", CHANNELS, previous="Last", growth=True), dataset)
    out = stage("P000", torch.from_numpy(data), attribute).numpy()
    assert out.shape == (1 + 1 + 6, *SHAPE) and stage.output_channels(1) == 8
    assert np.array_equal(out[0], data[0]) and np.array_equal(out[1], last)
    # Three interactions at (1/0.98)^k in float16, then over the last's: the box, then the clicks.
    intensities = [1 / 0.98, 1 / 0.98 / 0.98, 1 / 0.98 / 0.98 / 0.98]
    reciprocal = np.float32(1) / np.float32(intensities[-1])
    weigh = lambda value: _half(np.float32(_half(value)) * reciprocal)  # noqa: E731
    assert set(np.unique(out[2])) == {0.0, weigh(intensities[0])}
    assert np.count_nonzero(out[2]) == (6 - 2 + 1) * (5 - 3 + 1) * 1  # the box on its slice z = 6, x 2..6, y 3..5
    assert out[4][4, 6, 5] == weigh(intensities[1]) and out[5][9, 11, 12] == weigh(intensities[2]) == 1.0
    assert not out[3].any() and not out[6].any() and not out[7].any()
    # The ball of the last click is cut by the volume's far corner: what is left of it is the ball's own values.
    stored = (distance_ball(4)[:5, :5, :6] * intensities[2]).astype(np.float32).astype(np.float16)
    assert np.array_equal(out[5][5:, 7:, 8:], (stored.astype(np.float32) * reciprocal).astype(np.float16))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_prompts_are_drawn_beside_a_volume_held_on_the_device(tmp_path: Path) -> None:
    last = np.zeros(SHAPE, np.uint8)
    last[2:4, 3:5, 4:6] = 1
    dataset = _case(tmp_path, [[2, 3, 6], [6, 5, 6], [5, 6, 4]], [BOX, BOX, POSITIVE], last)
    data, attribute = dataset.read_data("CT", "P000")
    stage = _bound(DrawPrompts("Points", CHANNELS, previous="Last", growth=True), dataset)
    host = stage("P000", torch.from_numpy(data), Attribute(attribute))
    held = stage("P000", torch.from_numpy(data).cuda(), Attribute(attribute))
    assert held.device.type == "cuda" and torch.equal(held.cpu(), host)


def test_prompts_without_growth_weigh_one_and_refuse_a_kind_no_channel_draws(tmp_path: Path) -> None:
    dataset = _case(tmp_path, [[5, 6, 4]], [NEGATIVE])
    data, attribute = dataset.read_data("CT", "P000")
    out = _bound(DrawPrompts("Points", [NEGATIVE], radius=1), dataset)("P000", torch.from_numpy(data), attribute)
    assert (
        out.shape[0] == 2
        and out[1][4, 6, 5] == 1.0
        and out[1].sum() == pytest.approx(1 + 6 * distance_ball(1)[0, 1, 1])
    )
    with pytest.raises(TransformError, match="no channel draws"):
        _bound(DrawPrompts("Points", [POSITIVE]), dataset)("P000", torch.from_numpy(data), attribute)
    with pytest.raises(TransformError, match="draws the kinds"):
        DrawPrompts("Points", ["scribble"])
    with pytest.raises(TransformError, match="0 to 16"):
        DrawPrompts("Points", [NEGATIVE], radius=17)
    with pytest.raises(TransformError, match="two channels"):
        DrawPrompts("Points", [POSITIVE, "", "", POSITIVE])
    # A last mask is placed on the volume handed by their two geometries, or by neither.
    placed = Attribute({"Origin": np.zeros(3), "Spacing": np.ones(3), "Direction": np.eye(3).ravel()})
    with pytest.raises(TransformError, match="not cut from"):
        offset_of(placed, list(SHAPE), Attribute(), list(SHAPE), "The last mask")
    with pytest.raises(TransformError, match="3-D"):
        _bound(DrawPrompts("Points", [NEGATIVE]), dataset)("P000", torch.zeros(1, 4, 5), attribute)


def test_a_box_past_the_window_draws_nothing_on_it_and_still_weighs(tmp_path: Path) -> None:
    # The Crop around the last click checks the box on the whole image; drawn on the window without the last mask, a
    # box the window leaves out spans none of it, as nnInteractive's window cut from its drawing on the image.
    dataset = _case(tmp_path, [[0, 0, 0], [1, 1, 1], [12, 10, 8]], [BOX, BOX, POSITIVE])
    data, attribute = dataset.read_data("CT", "P000")
    crop = _bound(Crop(shape=[4, 4, 4], points="Points", fill=0.0), dataset)
    crop.transform_shape("CT", "P000", list(SHAPE), attribute)
    window = crop("P000", torch.from_numpy(data), attribute)
    drawn = _bound(DrawPrompts("Points", [BOX, POSITIVE], growth=True), dataset)("P000", window, attribute)
    assert drawn.shape == (3, 4, 4, 4) and not drawn[1].any() and drawn[2].max() == 1.0
    # On the image, a box off it takes the voxel nnInteractive's session gives it (`_add_validated_bbox_interaction`,
    # 2.7.0): the image's first before it, its last for a box starting at its end, none for one starting further.
    z = SHAPE[0]
    for corners, low, high in (
        ([[-5, -5, -5], [-2, -2, -2]], [0, 0, 0], [1, 1, 1]),
        ([[z, 2, 2], [z + 3, 4, 4]], [z - 1, 2, 2], [z, 5, 5]),
        ([[z + 1, 2, 2], [z + 3, 4, 4]], [z, 2, 2], [z, 5, 5]),
    ):
        ((kind, found_low, found_high),) = interactions(np.array(corners), [BOX, BOX], list(SHAPE))
        assert (kind, found_low.tolist(), found_high.tolist()) == (BOX, low, high)
    # On a window, the clip stands.
    ((_, low, high),) = interactions(np.array([[-5, -5, -5], [-2, -2, -2]]), [BOX, BOX], list(SHAPE), image=False)
    assert (low.tolist(), high.tolist()) == ([0, 0, 0], [-1, -1, -1])


def test_prompts_drawn_on_a_window_are_those_drawn_on_the_image_then_cut(tmp_path: Path) -> None:
    # The window around the last click runs past the image; a click's ball is cut by the image's edge inside it; the box
    # is clipped to the image: drawn on the window, at its place on the last mask's grid, they are the image's drawing
    # cut to the window, the last mask 0 past the image.
    last = (np.random.default_rng(3).uniform(size=SHAPE) > 0.5).astype(np.uint8)
    voxels, kinds = [[0, 3, 6], [5, 15, 9], [12, 2, 1], [13, 11, 8]], [BOX, BOX, POSITIVE, NEGATIVE]
    dataset = _case(tmp_path, voxels, kinds, last)
    data, attribute = dataset.read_data("CT", "P000")
    draw = DrawPrompts("Points", CHANNELS, previous="Last", growth=True)
    whole = _bound(draw, dataset)("P000", torch.from_numpy(data), attribute)
    crop = _bound(Crop(shape=[8, 9, 10], points="Points", fill=0.0), dataset)
    crop.transform_shape("CT", "P000", list(SHAPE), attribute)
    cut = crop("P000", whole, Attribute(attribute))
    window_attribute = Attribute(attribute)
    window = _bound(Crop(shape=[8, 9, 10], points="Points", fill=0.0), dataset)
    window.transform_shape("CT", "P000", list(SHAPE), window_attribute)
    whole_reads: list[str] = []
    read_data = dataset.read_data
    dataset.read_data = lambda group, name: (whole_reads.append(group), read_data(group, name))[1]  # type: ignore[method-assign]
    drawn = _bound(draw, dataset)("P000", window("P000", torch.from_numpy(data), window_attribute), window_attribute)
    assert drawn.shape == cut.shape == (8, 8, 9, 10) and torch.equal(drawn, cut)
    # The last mask is read where the window lies on it, never whole.
    assert "Last" not in whole_reads


def test_a_window_around_the_last_interaction_is_pasted_back_onto_the_last_mask(tmp_path: Path) -> None:
    last = np.full(SHAPE, 7, np.uint8)
    dataset = _case(tmp_path, [[5, 6, 4], [1, 2, 8]], [POSITIVE, NEGATIVE], last)
    data, attribute = dataset.read_data("CT", "P000")
    stage = _bound(Crop(shape=[6, 6, 6], points="Points", fill=-1.0, onto="Last"), dataset)
    assert stage.transform_shape("CT", "P000", list(SHAPE), attribute) == [6, 6, 6]
    window = stage("P000", torch.from_numpy(data), attribute)
    # Centred on the last click, (z, y, x) = (8, 2, 1): [5, 11) x [-1, 5) x [-2, 4), padded with the fill.
    assert np.array_equal(window[0, :5, 1:, 2:], data[0, 5:10, 0:5, 0:4]) and (window[0, 5] == -1).all()
    back = stage.inverse("P000", torch.zeros_like(window) + 3, attribute)
    assert back.shape == (1, *SHAPE) and back.dtype == window.dtype
    assert (back[0, 5:10, 0:5, 0:4] == 3).all() and back.sum() == 3 * 5 * 5 * 4 + 7 * (np.prod(SHAPE) - 5 * 5 * 4)
    with pytest.raises(TransformError, match="no shape"):
        Crop(points="Points")
    # A click past the image by more than half the window: the window holds none of it.
    far = _case(tmp_path / "far", [[-10, 6, 4]], [POSITIVE])
    _, header = far.read_data("CT", "P000")
    with pytest.raises(TransformError, match="holds no voxel"):
        _bound(Crop(shape=[6, 6, 6], points="Points", fill=0.0), far).transform_shape("CT", "P000", list(SHAPE), header)


def test_a_window_is_pasted_back_onto_a_group_on_the_volumes_grid_only(tmp_path: Path) -> None:
    dataset = _case(tmp_path, [[5, 6, 4]], [POSITIVE])
    moved = sitk.ReadImage(str(tmp_path / "P000" / "Last.mha"))
    moved.SetOrigin((-9.0, 20.0, 5.0))  # one voxel and a bit along x: the same extent on another grid
    sitk.WriteImage(moved, str(tmp_path / "P000" / "Last.mha"))
    data, attribute = dataset.read_data("CT", "P000")
    stage = _bound(Crop(shape=[6, 6, 6], points="Points", onto="Last"), dataset)
    stage.transform_shape("CT", "P000", list(SHAPE), attribute)
    window = stage("P000", torch.from_numpy(data), attribute)
    with pytest.raises(TransformError, match="not cut from"):
        stage.inverse("P000", window, attribute)
    # A thousandth of a millimetre at a spacing of a thousandth is a voxel: the origin is compared in voxels.
    fine = Attribute({"Origin": np.array([1000.0, 0, 0]), "Spacing": np.full(3, 0.001), "Direction": np.eye(3).ravel()})
    near = Attribute({**{key: fine[key] for key in ("Spacing", "Direction")}, "Origin": np.array([1000.001, 0, 0])})
    assert offset_of(near, [4, 4, 4], fine, [4, 4, 4], "x").tolist() == [0, 0, 1]
    # Without a geometry, a window cannot be placed: a crop around the prompts refuses it, as two extents do.
    with pytest.raises(TransformError, match="no geometry"):
        _bound(Crop(shape=[6, 6, 6], points="Points"), dataset).transform_shape("CT", "P000", list(SHAPE), Attribute())
    with pytest.raises(TransformError, match="neither carries"):
        offset_of(Attribute(), [4, 4, 4], Attribute(), [2, 2, 2], "x")


def _window(root: Path, last: np.ndarray | None = None):
    """A window of P000's CT around its one click, the stage that cut it onto ``Last`` and the header it left."""
    dataset = _case(root, [[5, 6, 4]], [POSITIVE], last)
    data, attribute = dataset.read_data("CT", "P000")
    stage = _bound(Crop(shape=[6, 6, 6], points="Points", onto="Last"), dataset)
    stage.transform_shape("CT", "P000", list(SHAPE), attribute)
    return stage, stage("P000", torch.from_numpy(data), attribute), attribute


@pytest.mark.parametrize("stored", ["a voxel further along x", "two channels"])
def test_a_window_is_not_pasted_onto_a_group_a_voxel_off_or_of_other_channels(tmp_path: Path, stored: str) -> None:
    # Pasted index for index: a group of the volume's extent a whole voxel away is cut from another place, and one of
    # other channels would take the window's on each of its own.
    stage, window, attribute = _window(tmp_path)
    last = sitk.ReadImage(str(tmp_path / "P000" / "Last.mha"))
    if stored == "two channels":
        both = sitk.GetImageFromArray(np.zeros((*SHAPE, 2), np.uint8), isVector=True)
        both.CopyInformation(last)
        last = both
    else:
        last.SetOrigin(last.TransformIndexToPhysicalPoint((1, 0, 0)))
    sitk.WriteImage(last, str(tmp_path / "P000" / "Last.mha"))
    with pytest.raises(TransformError, match="on another grid than the volume's"):
        stage.inverse("P000", window, attribute)


def test_a_window_is_pasted_where_it_is_held_and_never_region_by_region(tmp_path: Path) -> None:
    stage, window, attribute = _window(tmp_path)
    # Streamed, the inverse fills what the window leaves with its ``fill``: the paste reads the group, whole.
    pasted = stage.inverse_patch_locality(attribute)
    assert pasted.kind is LocalityKind.WHOLE_VOLUME and "'Last'" in str(pasted.reason)
    assert Crop(shape=[6, 6, 6], points="Points").inverse_patch_locality(attribute).kind is LocalityKind.REGRID
    # The group is read on the host and joins the window where it is held: a device with no storage stands for a GPU.
    back = stage.inverse("P000", window.to("meta"), attribute)
    assert back.device.type == "meta" and back.shape == (1, *SHAPE)


def test_a_box_is_centred_on_the_middle_of_its_span_rounded_half_to_even() -> None:
    # nnInteractive's round((low + high) / 2): the box [z, z + 1) of one slice on z or z + 1, whichever is even.
    assert centre(BOX, np.array([3, 2, 2]), np.array([4, 6, 7])).tolist() == [4, 4, 4]  # 3.5, 4, 4.5
    assert centre(BOX, np.array([2, 2, 2]), np.array([3, 6, 7])).tolist() == [2, 4, 4]  # 2.5, 4, 4.5
    assert centre(POSITIVE, np.array([3, 2, 2]), np.array([4, 3, 3])).tolist() == [3, 2, 2]


def test_prompts_of_an_unknown_kind_or_without_a_kind_each_are_refused(tmp_path: Path) -> None:
    dataset = _case(tmp_path, [[5, 6, 4], [1, 2, 8]], [POSITIVE, "scribble"])
    np.save(tmp_path / "P000" / "Bare.npy", np.zeros((2, 3)))  # points no file labels
    stage = _bound(DrawPrompts("Points", [POSITIVE]), dataset)
    with pytest.raises(TransformError, match=r"hold the kinds \['scribble'\] for 2 points"):
        read_prompts(stage, "Points", "P000")
    with pytest.raises(TransformError, match=r"hold the kinds \[\] for 2 points"):
        read_prompts(stage, "Bare", "P000")


def test_a_grid_of_another_spacing_or_direction_is_not_placed_on_the_volume_handed() -> None:
    grid = {"Origin": np.zeros(3), "Spacing": np.ones(3), "Direction": np.eye(3).ravel()}
    assert offset_of(Attribute(grid), [4, 4, 4], Attribute(grid), [4, 4, 4], "x").tolist() == [0, 0, 0]
    # The same origin, so the same first voxel, and every other voxel elsewhere.
    turned = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    for key, other in (("Spacing", np.array([1.0, 1.0, 2.0])), ("Direction", turned)):
        with pytest.raises(TransformError, match="not cut from"):
            offset_of(Attribute({**grid, key: other}), [4, 4, 4], Attribute(grid), [4, 4, 4], "x")


@pytest.mark.parametrize(("count", "held"), [(548, True), (549, False)])
def test_interactions_weighing_more_than_float16_holds_are_refused(tmp_path: Path, count: int, held: bool) -> None:
    # The k-th interaction is stored at (1 / 0.98)^k in float16, whose largest value the 549th passes.
    dataset = _case(tmp_path, [[5, 6, 4]] * count, [POSITIVE] * count)
    data, attribute = dataset.read_data("CT", "P000")
    stage = _bound(DrawPrompts("Points", [POSITIVE], radius=0, growth=True), dataset)
    if held:
        assert stage("P000", torch.from_numpy(data), attribute)[1].max() == 1.0
    else:
        with pytest.raises(TransformError, match="weigh more than float16 holds"):
            stage("P000", torch.from_numpy(data), attribute)


def test_the_last_mask_is_what_lies_above_a_half_and_a_draw_is_float32(tmp_path: Path) -> None:
    last = np.zeros(SHAPE, np.float32)
    last[1, 1, 1], last[2, 2, 2], last[3, 3, 3] = 0.25, 0.75, 0.5
    dataset = _case(tmp_path, [[5, 6, 4]], [POSITIVE], last)
    data, attribute = dataset.read_data("CT", "P000")
    for growth in (False, True):
        stage = _bound(DrawPrompts("Points", [POSITIVE], previous="Last", growth=growth), dataset)
        out = stage("P000", torch.from_numpy(data).to(torch.int16), attribute)
        assert out[1].sum() == 1 and out[1, 2, 2, 2] == 1
        # Whatever the volume holds and whatever the prompts are stored in, which the program is told.
        assert out.dtype == torch.float32
    assert _result_dtype({"op": "draw_prompts"}, "int16") == _result_dtype({"op": "draw_prompts"}, None) == "float32"


def test_the_schema_states_what_the_runtime_reads_of_the_prompt_ops() -> None:
    # The runtime reads a draw with no radius at 0 and points with no rounding at rint, and refuses a crop around the
    # prompts that names no shape and a draw of a kind twice or of one konfai does not know: the schema says as much.
    import jsonschema
    from konfai.portable_schema import program_schema

    validator = jsonschema.Draft202012Validator({"$defs": program_schema()["$defs"], "$ref": "#/$defs/chain_step"})
    assert validator.is_valid({"op": "draw_prompts", "points_group": "PROMPTS", "kinds": [POSITIVE]})
    # Each kind on one channel, of the kinds konfai draws; "" reserves as many empty channels as wanted.
    assert validator.is_valid({"op": "draw_prompts", "points_group": "PROMPTS", "kinds": CHANNELS})
    assert not validator.is_valid({"op": "draw_prompts", "points_group": "PROMPTS", "kinds": [POSITIVE, POSITIVE]})
    assert not validator.is_valid({"op": "draw_prompts", "points_group": "PROMPTS", "kinds": ["click"]})
    assert validator.is_valid({"op": "points_on_grid", "reference_group": "CT"})
    assert not validator.is_valid({"op": "crop", "points_group": "PROMPTS", "fill": 0.0})
    assert validator.is_valid({"op": "crop", "points_group": "PROMPTS", "shape": [4, 4, 4], "fill": 0.0})
    assert validator.is_valid({"op": "crop", "onto_group": "LAST"})


def test_the_prompt_stages_compile_to_ops_reading_their_groups_beside_the_case() -> None:
    assert OP_MAP["PointsOnGrid"]({"reference": "CT"}) == {
        "op": "points_on_grid",
        "reference_group": "CT",
        "rounding": "rint",
    }
    assert _result_dtype({"op": "points_on_grid"}, "float32") == "float64"
    # A stage konfai refuses to build is refused as it compiles.
    with pytest.raises(TransformError, match="no shape"):
        compile_pipeline({"Crop": {"points": "PROMPTS"}})
    chain = {
        "Standardize": {"inverse": False},
        "DrawPrompts": {"points": "PROMPTS", "previous": "LAST", "channels": CHANNELS, "growth": True},
        "Crop": {"shape": [192, 192, 192], "points": "PROMPTS", "fill": 0, "onto": "LAST"},
    }
    config = {
        "Predictor": {
            "Model": {"classpath": "Probe"},
            "Dataset": {"groups_src": {"g": {"groups_dest": {"g": {"is_input": True, "transforms": chain}}}}},
            "outputs_dataset": {"H": {"OutputDataset": {"after_reduction_transforms": {"Argmax": {"dim": 0}}}}},
        }
    }
    manifest, _folds, _per_copy = transform_manifest(config, "Predictor")
    draw = {"op": "draw_prompts", "points_group": "PROMPTS", "previous_group": "LAST", "kinds": CHANNELS, "radius": 4}
    crop = {
        "op": "crop",
        "shape": [192] * 3,
        "fill": 0.0,
        "points_group": "PROMPTS",
        "onto_group": "LAST",
        "id": "Crop",
    }
    assert manifest["preprocessing"] == [{"op": "standardize"}, draw | {"growth": True}, crop]
    assert manifest["postprocessing"] == [{"op": "argmax", "dim": 0}, {"op": "inverse", "of": "Crop"}]
    # A graph folded after the drawing is exported for the channels it leaves: the volume's, the last mask, the kinds.
    assert _channels_after(draw, 1, 3) == 8
    assert manifest["reads"] == [
        {"group": "PROMPTS", "channels": 1, "preprocessing": []},
        {"group": "LAST", "channels": 1, "preprocessing": []},
    ]
    # A group the chain of another input reads beside the case is read too, as stored.
    masked = {"min_value": "min", "max_value": 1.0}
    other = {"g2": {"is_input": True, "transforms": {"Clip": {**masked, "mask": "MORE"}}}}
    config["Predictor"]["Dataset"]["groups_src"]["g2"] = {"groups_dest": other}
    manifest, _folds, _per_copy = transform_manifest(config, "Predictor")
    assert [read["group"] for read in manifest["reads"]] == ["g2", "PROMPTS", "LAST", "MORE"]
    # The first group is read as stored where another group's chain reads it beside the case.
    other["g2"]["transforms"] = {"Clip": {**masked, "mask": "g"}}
    manifest, _folds, _per_copy = transform_manifest(config, "Predictor")
    assert [read["group"] for read in manifest["reads"]] == ["g2", "PROMPTS", "LAST", "g"]
    # The prompts are drawn in the first group's chain: the graph takes every other group as it is stored, so a draw
    # that adds channels to one is refused.
    other["g2"]["transforms"] = {"DrawPrompts": {"points": "MORE", "channels": [POSITIVE]}}
    with pytest.raises(ConfigError, match="the chain of group 'g2' hands the model 2 channels of the 1"):
        transform_manifest(config, "Predictor")
    other["g2"]["transforms"] = {"Clip": {**masked, "mask": "g"}}
    # A group the model reads as stored and a stage reads too would be listed as the stage's alone: refused.
    config["Predictor"]["Dataset"]["groups_src"]["LAST"] = {"groups_dest": {"LAST": {"is_input": True}}}
    with pytest.raises(ConfigError, match="reads the group 'LAST' as stored"):
        transform_manifest(config, "Predictor")
    del config["Predictor"]["Dataset"]["groups_src"]["LAST"]
    # The prompts are handed the input group's chain alone: a stage drawing them after the forward is refused.
    final = {"DrawPrompts": {"points": "PROMPTS", "channels": [POSITIVE]}}
    config["Predictor"]["outputs_dataset"]["H"]["OutputDataset"]["final_transforms"] = final
    with pytest.raises(ConfigError, match="after the forward"):
        transform_manifest(config, "Predictor")
    # So is every stage reading a group beside the case there, but a mask and a class's groups.
    for final in (
        {"PointsOnGrid": {"reference": "CT"}},
        {"HistogramMatching": {"reference_group": "CT"}},
        {"Crop": {"shape": [4, 4, 4], "onto": "CT"}},
        {"Crop": {"source": "CT"}},
        {"Resample": {"field_group": "CT"}},
    ):
        config["Predictor"]["outputs_dataset"]["H"]["OutputDataset"]["final_transforms"] = final
        with pytest.raises(ConfigError, match="reads the case's group 'CT'"):
            transform_manifest(config, "Predictor")
