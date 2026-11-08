# SPDX-License-Identifier: Apache-2.0
"""Synthetic cases with known defects, for tests, examples and the integrations' own tests: no patient data, a few
seconds to build, and a record of where each planted defect sits in millimetres."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from konfai.data.geometry import Grid

from .case import InspectionCase
from .planes import box_to_dict, index_to_world, world_box_of


def _sitk():
    import SimpleITK as sitk

    return sitk


def _direction(oblique: bool) -> tuple[float, ...]:
    if not oblique:
        return tuple(np.eye(3).ravel())
    angle = np.deg2rad(12.0)
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]])
    tilt = np.deg2rad(7.0)
    rotation = rotation @ np.array(
        [[1.0, 0.0, 0.0], [0.0, np.cos(tilt), -np.sin(tilt)], [0.0, np.sin(tilt), np.cos(tilt)]]
    )
    return tuple(rotation.ravel())


def _image(array: np.ndarray, grid: Grid, vector: bool = False):
    sitk = _sitk()
    if vector:
        image = sitk.GetImageFromArray(
            np.ascontiguousarray(np.moveaxis(array.astype(np.float32), 0, -1)), isVector=True
        )
    else:
        image = sitk.GetImageFromArray(np.ascontiguousarray(array))
    image.SetSpacing([float(v) for v in grid.spacing_xyz])
    image.SetOrigin([float(v) for v in grid.origin_xyz])
    image.SetDirection([float(v) for v in np.asarray(grid.direction_xyz).ravel()])
    return image


def _write(image, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    _sitk().WriteImage(image, str(path), False)
    return path


def _sphere(shape: tuple[int, int, int], center: tuple[float, float, float], radius: float) -> np.ndarray:
    zz, yy, xx = np.ogrid[: shape[0], : shape[1], : shape[2]]
    return (zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2 <= radius**2


def _grid(shape: tuple[int, int, int], spacing: tuple[float, float, float], oblique: bool) -> Grid:
    return Grid(
        tuple(shape),
        np.array([-20.0, 10.0, 5.0]),
        np.asarray(spacing, dtype=np.float64),
        np.asarray(_direction(oblique)).reshape(3, 3),
    )


def _box_of(mask: np.ndarray, grid: Grid) -> dict[str, list[float]]:
    where = np.argwhere(mask)
    window = tuple(slice(int(a), int(b) + 1) for a, b in zip(where.min(axis=0), where.max(axis=0), strict=True))
    return box_to_dict(world_box_of(grid, window))


def segmentation_case(
    folder: Path | str,
    shape: tuple[int, int, int] = (24, 48, 48),
    spacing: tuple[float, float, float] = (1.0, 1.0, 2.0),
    oblique: bool = False,
    with_truth: bool = True,
    clean: bool = False,
) -> tuple[InspectionCase, dict[str, object]]:
    """An image with a sphere (label 1) and a cube (label 2). The prediction misses the cube, shifts the sphere by
    two voxels, punches a hole in it, and adds a spurious blob of label 1 (so label 1 is in two pieces). ``clean``
    makes the prediction the truth itself: a negative control, where any error reported is invented."""
    folder = Path(folder)
    grid = _grid(shape, spacing, oblique)
    rng = np.random.default_rng(0)
    sphere = _sphere(shape, (shape[0] / 2, shape[1] / 2, shape[2] / 2 - 6), 7)
    cube = np.zeros(shape, dtype=bool)
    cube[4:10, 6:16, 30:42] = True
    truth = np.zeros(shape, dtype=np.int16)
    truth[sphere] = 1
    truth[cube] = 2
    image = (-100.0 + (truth == 1) * 400.0 + (truth == 2) * 300.0 + rng.normal(0, 15, shape)).astype(np.float32)
    prediction = np.zeros(shape, dtype=np.int16)
    prediction[np.roll(sphere, 2, axis=1)] = 1
    hole = _sphere(shape, (shape[0] / 2, shape[1] / 2 + 2, shape[2] / 2 - 6), 2.5)
    prediction[hole] = 0
    blob = _sphere(shape, (shape[0] / 2 + 4, shape[1] / 2 + 14, shape[2] / 2 + 14), 3)
    prediction[blob] = 1
    if clean:
        prediction = truth.copy()
    inputs = {
        "image": _write(_image(image, grid), folder / "image.mha"),
        "prediction": _write(_image(prediction, grid), folder / "prediction.mha"),
    }
    if with_truth:
        inputs["truth"] = _write(_image(truth, grid), folder / "truth.mha")
    case = InspectionCase(
        "segmentation",
        folder.name or "synthetic",
        inputs,
        labels={1: "sphere", 2: "cube"},
        notes="synthetic phantom: a sphere and a cube",
    )
    defects: dict[str, object] = {}
    if not clean:
        defects = {
            "missing_cube": {"defect": "missing_structure", "label": 2, "box_mm": _box_of(cube, grid)},
            "spurious_blob": {"defect": "spurious_structure", "label": 1, "box_mm": _box_of(blob, grid)},
            "hole": {"defect": "holes", "label": 1, "box_mm": _box_of(hole, grid)},
            "shifted_sphere": {"defect": "boundary_error", "label": 1, "shift_voxels_yxz": [2, 0, 0]},
        }
    (folder / "defects.json").write_text(json.dumps(defects, indent=2))
    return case, defects


def registration_case(
    folder: Path | str,
    shape: tuple[int, int, int] = (24, 48, 48),
    spacing: tuple[float, float, float] = (1.0, 1.0, 2.0),
    oblique: bool = False,
    with_references: bool = True,
    fold: bool = True,
    clean: bool = False,
) -> tuple[InspectionCase, dict[str, object]]:
    """A moving image with a sphere, a cube and a gradient; the fixed image is the moving one pulled through a smooth
    known field, so that field is the exact registration. The field under inspection is the exact one with a region
    where it is zeroed (a residual misalignment) and, with ``fold``, a sharp bump that folds the map. ``clean``
    inspects the exact field itself: a negative control."""
    sitk = _sitk()
    folder = Path(folder)
    grid = _grid(shape, spacing, oblique)
    rng = np.random.default_rng(1)
    moving_labels = np.zeros(shape, dtype=np.int16)
    sphere = _sphere(shape, (shape[0] / 2, shape[1] / 2, shape[2] / 2 - 6), 8)
    moving_labels[sphere] = 1
    moving_labels[4:10, 6:16, 30:42] = 2
    zz, yy, xx = np.indices(shape).astype(np.float32)
    moving = (
        -100.0
        + (moving_labels == 1) * 400.0
        + (moving_labels == 2) * 300.0
        + 2.0 * xx
        + 1.5 * yy
        + rng.normal(0, 10, shape)
    ).astype(np.float32)
    # the exact field, in mm along world axes: a smooth sinusoid, up to ~3 mm
    index = np.stack([xx, yy, zz], axis=0)
    world = (
        np.einsum("ij,j...->i...", grid.index_to_world.matrix, index) + np.asarray(grid.origin_xyz)[:, None, None, None]
    )
    exact = np.stack(
        [
            3.0 * np.sin(world[1] / 15.0),
            2.0 * np.cos(world[0] / 12.0),
            1.0 * np.sin(world[2] / 10.0),
        ],
        axis=0,
    ).astype(np.float32)
    exact_image = _image(exact, grid, vector=True)
    transform = sitk.DisplacementFieldTransform(sitk.Cast(exact_image, sitk.sitkVectorFloat64))
    moving_image = _image(moving, grid)
    fixed_image = sitk.Resample(moving_image, moving_image, transform, sitk.sitkLinear, -100.0, sitk.sitkFloat32)
    fixed_labels = sitk.Resample(
        _image(moving_labels, grid), moving_image, transform, sitk.sitkNearestNeighbor, 0, sitk.sitkInt16
    )
    inspected = exact.copy()
    zeroed = np.zeros(shape, dtype=bool)
    zeroed[8:18, 26:40, 8:22] = True
    defects: dict[str, object] = {}
    if not clean:
        inspected[:, zeroed] = 0.0
        defects["zeroed_field"] = {
            "defect": "residual_misalignment",
            "box_mm": _box_of(zeroed, grid),
            "displacement_lost_mm": float(np.linalg.norm(exact[:, zeroed], axis=0).mean()),
        }
    if fold and not clean:
        bump = _sphere(shape, (shape[0] / 2, 10, shape[2] / 2 + 12), 3)
        inspected[0, bump] += 12.0  # a 12 mm jump over a few voxels: det J < 0 at its edges
        defects["fold"] = {"defect": "folding", "box_mm": _box_of(bump, grid)}
    inputs = {
        "fixed": _write(fixed_image, folder / "fixed.mha"),
        "moving": _write(moving_image, folder / "moving.mha"),
        "field": _write(_image(inspected, grid, vector=True), folder / "field.mha"),
    }
    _write(exact_image, folder / "exact_field.mha")
    if with_references:
        inputs["fixed_labels"] = _write(fixed_labels, folder / "fixed_labels.mha")
        inputs["moving_labels"] = _write(_image(moving_labels, grid), folder / "moving_labels.mha")
        fixed_points = np.array(
            [
                index_to_world(grid, [24.0, 24.0, 12.0]),
                index_to_world(grid, [35.0, 10.0, 6.0]),
                index_to_world(grid, [12.0, 30.0, 14.0]),
            ]
        )
        moving_points = np.array([transform.TransformPoint([float(v) for v in p]) for p in fixed_points])
        for name, points in (("fixed_landmarks", fixed_points), ("moving_landmarks", moving_points)):
            rows = "\n".join(f"F-{i + 1},{x:.6f},{y:.6f},{z:.6f}" for i, (x, y, z) in enumerate(points))
            path = folder / f"{name}.fcsv"
            path.write_text(
                f"# Markups fiducial file version = 4.6\n# CoordinateSystem = LPS\n# columns = id,x,y,z\n{rows}\n"
            )
            inputs[name] = path
    case = InspectionCase(
        "registration",
        folder.name or "synthetic",
        inputs,
        labels={1: "sphere", 2: "cube"},
        notes="synthetic phantom pulled through a known smooth field",
    )
    (folder / "defects.json").write_text(json.dumps(defects, indent=2))
    return case, defects


def synthesis_case(
    folder: Path | str,
    shape: tuple[int, int, int] = (24, 48, 48),
    spacing: tuple[float, float, float] = (1.0, 1.0, 2.0),
    oblique: bool = False,
    with_target: bool = True,
    clean: bool = False,
) -> tuple[InspectionCase, dict[str, object]]:
    """An MR-like source and a CT-like target with a sphere; the synthetic image is the target blurred, with a
    +150 HU bias in a box and a small invented bright structure. ``clean`` makes it the target under fresh noise: a
    negative control."""
    from scipy import ndimage

    folder = Path(folder)
    grid = _grid(shape, spacing, oblique)
    rng = np.random.default_rng(2)
    sphere = _sphere(shape, (shape[0] / 2, shape[1] / 2, shape[2] / 2), 9)
    bone = _sphere(shape, (shape[0] / 2, shape[1] / 2 - 14, shape[2] / 2 + 10), 4)
    target = (-100.0 + sphere * 150.0 + bone * 900.0 + rng.normal(0, 12, shape)).astype(np.float32)
    source = (50.0 + sphere * 700.0 + bone * (-30.0) + rng.normal(0, 25, shape)).astype(np.float32)
    synthetic = ndimage.gaussian_filter(target, 1.2).astype(np.float32)
    biased = np.zeros(shape, dtype=bool)
    biased[6:18, 8:22, 26:44] = True
    synthetic[biased] += 150.0
    invented = _sphere(shape, (shape[0] / 2 + 5, shape[1] / 2 + 12, shape[2] / 2 - 14), 3)
    synthetic[invented] = 700.0
    if clean:
        synthetic = (target + rng.normal(0, 8, shape)).astype(np.float32)
    inputs = {
        "source": _write(_image(source, grid), folder / "source.mha"),
        "synthetic": _write(_image(synthetic, grid), folder / "synthetic.mha"),
    }
    if with_target:
        inputs["target"] = _write(_image(target, grid), folder / "target.mha")
    case = InspectionCase(
        "synthesis", folder.name or "synthetic", inputs, notes="synthetic MR-like source to CT-like target"
    )
    defects: dict[str, object] = {}
    if not clean:
        defects = {
            "bias": {"defect": "intensity_bias", "box_mm": _box_of(biased, grid), "bias_hu": 150.0},
            "invented": {"defect": "hallucinated_structure", "box_mm": _box_of(invented, grid)},
            "blur": {"defect": "blur", "sigma_voxels": 1.2},
        }
    (folder / "defects.json").write_text(json.dumps(defects, indent=2))
    return case, defects


MAKERS = {"segmentation": segmentation_case, "registration": registration_case, "synthesis": synthesis_case}


def make_all(root: Path | str, oblique: bool = False, clean: bool = False) -> dict[str, InspectionCase]:
    root = Path(root)
    return {task: maker(root / task, oblique=oblique, clean=clean)[0] for task, maker in MAKERS.items()}


if __name__ == "__main__":  # pragma: no cover
    import sys

    cases = make_all(sys.argv[1] if len(sys.argv) > 1 else "fixtures")
    for task, case in cases.items():
        print(task, {role: str(path) for role, path in case.inputs.items()})
