# SPDX-License-Identifier: Apache-2.0
"""Rendering: every mode of every task, the radiological convention, the pixel-to-world affine, zoom, montage."""

import io

import numpy as np
import pytest
from konfai_scout.config import ScoutConfig
from konfai_scout.planes import box_from_dict, box_size, plane_frame
from konfai_scout.views import DEFAULT_MODE, MODES, Renderer, ViewError
from PIL import Image


def renderer(fixtures, key, tmp_path, max_px=256) -> Renderer:
    case, _ = fixtures[key]
    case.stage(tmp_path / "work" / "stage")
    return Renderer(case, tmp_path / "work", ScoutConfig(max_px=max_px), tmp_path / "evidence")


@pytest.mark.parametrize(
    "key",
    [
        "segmentation",
        "registration",
        "synthesis",
        "segmentation_no_truth",
        "registration_no_ref",
        "synthesis_no_target",
    ],
)
def test_every_available_mode_renders(fixtures, tmp_path, key):
    r = renderer(fixtures, key, tmp_path)
    modes = r.modes()
    assert DEFAULT_MODE[r.case.task] in modes
    for mode in modes:
        png, view = r.render("axial", mode)
        image = Image.open(io.BytesIO(png))
        assert image.size[0] == 256 or image.size[1] in (256 + 18, 256)
        assert view.path.endswith(".png") and (tmp_path / "evidence" / f"{view.id}.png").exists()
        assert view.px_to_mm is not None and view.window is not None
    missing = set(MODES[r.case.task]) - set(modes)
    if key.endswith(("no_truth", "no_ref", "no_target")):
        assert missing, "a case without references offers fewer modes"
        for mode in missing:
            with pytest.raises(ViewError):
                r.render("axial", mode)


@pytest.mark.parametrize("key", ["segmentation", "segmentation_oblique"])
def test_the_affine_points_at_the_drawn_voxels(fixtures, tmp_path, key):
    case, _ = fixtures[key]
    r = renderer(fixtures, key, tmp_path, max_px=512)
    prediction = case.volume("prediction")
    index = prediction.grid.size_zyx[0] // 2
    png, view = r.render("axial", "prediction", index=index)
    image = np.asarray(Image.open(io.BytesIO(png)).convert("RGB")).astype(int)
    labels = prediction.plane(0, index)
    where = np.argwhere(labels == 1)
    assert len(where) > 20
    origin, u, v = (np.asarray(view.px_to_mm[k]) for k in ("origin", "u", "v"))
    basis = np.stack([u, v], axis=1)
    hits = 0
    for row, col in where:
        world = case.grid.index_to_world.apply(np.array([col, row, index], dtype=float))
        px, py = np.linalg.lstsq(basis, world - origin, rcond=None)[0]
        px, py = round(px), round(py)
        assert 0 <= px < image.shape[1] and 0 <= py < image.shape[0]
        red, green, blue = image[py, px]
        hits += red > green + 25 and red > blue + 25  # label 1 is drawn red over a grey image
    assert hits / len(where) > 0.9


def test_zoom_crops_to_the_box_and_keeps_the_size(fixtures, tmp_path):
    case, defects = fixtures["segmentation"]
    r = renderer(fixtures, "segmentation", tmp_path)
    box = box_from_dict(defects["spurious_blob"]["box_mm"])
    _png, view = r.render("coronal", "prediction", zoom=box)
    assert view.kind == "slice" and max(view.size_px) in (256, 256 + 18)
    shown = box_from_dict(view.box_mm)
    assert np.all(shown.low_xyz <= box.high_xyz) and np.all(shown.high_xyz >= box.low_xyz)
    assert np.prod(box_size(shown)) < np.prod(box_size(case.grid.world_box())) / 10
    with pytest.raises(ViewError):
        r.render("axial", "prediction", zoom=box_from_dict({"min": [900, 900, 900], "max": [910, 910, 910]}))


def test_a_zoom_on_an_oblique_grid_reports_where_it_looks(fixtures, tmp_path):
    """On a tilted plane the slice's centre is far from the zoom: the position given is the zoom's, the tilt named."""
    case, _ = fixtures["segmentation_oblique"]
    r = renderer(fixtures, "segmentation_oblique", tmp_path)
    world = case.grid.world_box()
    centre = world.low_xyz + (world.high_xyz - world.low_xyz) * np.array([0.3, 0.3, 0.5])
    _png, view = r.render("axial", "image", zoom=box_from_dict({"center": centre.tolist(), "size": 10}))
    assert np.linalg.norm(np.asarray(view.position_mm) - centre) <= 2 * max(case.grid.spacing_xyz)
    assert "tilted" in r._plane_name(plane_frame(r.grid, "axial"))
    straight = renderer(fixtures, "segmentation", tmp_path / "straight")
    assert straight._plane_name(plane_frame(straight.grid, "axial")) == "axial"


def test_montage_tiles_cover_the_plane(fixtures, tmp_path):
    r = renderer(fixtures, "synthesis", tmp_path, max_px=600)
    png, view = r.montage("axial", "synthetic", count=9)
    assert view.kind == "montage" and len(view.extra["tiles"]) == 9
    indices = view.extra["indices"]
    assert indices == sorted(indices) and indices[0] > 0 and indices[-1] < r.grid.size_zyx[0] - 1
    image = Image.open(io.BytesIO(png))
    assert image.size[0] <= 620 and image.size[1] > 100


def test_windows_and_unknown_modes_are_checked(fixtures, tmp_path):
    r = renderer(fixtures, "segmentation", tmp_path)
    _png, view = r.render("axial", "image", window="ct_soft")
    assert view.window == [-160.0, 240.0]
    _png, view = r.render("axial", "image", window=[-50, 50])
    assert view.window == [-50.0, 50.0]
    with pytest.raises(ViewError):
        r.render("axial", "image", window="ct_whatever")
    with pytest.raises(ViewError):
        r.render("axial", "nothing")
    with pytest.raises(ValueError):
        r.render("oblique", "image")


def test_a_role_on_another_grid_is_refused(fixtures, tmp_path):
    import SimpleITK as sitk
    from konfai_scout.case import InspectionCase
    from konfai_scout.evaluation import EvaluationError, RegionEvaluator

    case, _ = fixtures["segmentation"]
    shifted = sitk.ReadImage(str(case.inputs["prediction"]))
    shifted.SetOrigin(tuple(o + s for o, s in zip(shifted.GetOrigin(), shifted.GetSpacing(), strict=True)))
    sitk.WriteImage(shifted, str(tmp_path / "prediction.mha"))
    moved = InspectionCase(
        "segmentation", "moved", {**case.inputs, "prediction": tmp_path / "prediction.mha"}, labels=case.labels
    )
    moved.stage(tmp_path / "work" / "stage")
    assert moved.volume("prediction").grid.size_zyx == moved.grid.size_zyx  # the same voxels, one voxel away
    r = Renderer(moved, tmp_path / "work", ScoutConfig(max_px=200), tmp_path / "evidence")
    with pytest.raises(ViewError, match="another grid"):
        r.render("axial", "prediction")
    with pytest.raises(EvaluationError, match="another grid"):
        RegionEvaluator(moved, ScoutConfig(), tmp_path / "work", None).measure(moved.grid.world_box())
    _png, view = r.render("axial", "image")  # the reference itself is fine
    assert view.mode == "image"


def test_a_2d_case_renders_one_plane(tmp_path):
    import SimpleITK as sitk
    from konfai_scout.case import InspectionCase

    rng = np.random.default_rng(0)
    image = rng.normal(0, 10, (40, 60)).astype(np.float32)
    labels = np.zeros((40, 60), dtype=np.uint8)
    labels[10:20, 20:40] = 1
    for name, array in (("image", image), ("prediction", labels)):
        im = sitk.GetImageFromArray(array)
        im.SetSpacing((0.5, 0.5))
        sitk.WriteImage(im, str(tmp_path / f"{name}.mha"))
    case = InspectionCase(
        "segmentation", "flat", {"image": tmp_path / "image.mha", "prediction": tmp_path / "prediction.mha"}
    )
    case.stage(tmp_path / "work" / "stage")
    assert case.grid.size_zyx == (1, 40, 60) and case.volume("image").rank == 2
    r = Renderer(case, tmp_path / "work", ScoutConfig(max_px=240), tmp_path / "evidence")
    _png, view = r.render("axial", "prediction")
    assert view.index == 0 and view.size_px[0] == 240
    assert case.volume("prediction").plane(0, 0).shape == (40, 60)
    assert np.array_equal(case.volume("image").region((0, 5, 10), (1, 7, 12))[0, 0], image[5:7, 10:12])
