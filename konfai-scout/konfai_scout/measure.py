# SPDX-License-Identifier: Apache-2.0
"""Descriptive statistics of a region, measured on the case alone: no reference, no metric.

What needs a reference (Dice, MAE, TRE...) and what KonfAI already measures (a field's Jacobian, bending energy,
displacement lengths) go through KonfAI Eval (:mod:`konfai_scout.evaluation`). Here: how much of each label a region
holds and in how many pieces, the intensities, the length of the displacements; plus the Jacobian determinant of one
plane, for display only.
"""

from __future__ import annotations

import numpy as np

from .geometry import Grid


def intensity_stats(values: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if mask is not None:
        array = array[np.asarray(mask, dtype=bool)]
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"voxels": 0}
    p1, p50, p99 = np.percentile(array, (1, 50, 99))
    return {
        "voxels": int(array.size),
        "mean": round(float(array.mean()), 4),
        "std": round(float(array.std()), 4),
        "min": round(float(array.min()), 4),
        "p1": round(float(p1), 4),
        "median": round(float(p50), 4),
        "p99": round(float(p99), 4),
        "max": round(float(array.max()), 4),
    }


def label_stats(
    labels: np.ndarray, grid: Grid, names: dict[int, str] | None = None, offset_zyx: tuple[int, int, int] = (0, 0, 0)
) -> dict[str, dict[str, object]]:
    """Per label: voxels, volume in mm³, connected components (26-connectivity), bounding box in mm."""
    from scipy import ndimage

    array = np.asarray(labels)
    voxel_mm3 = float(np.prod(grid.spacing_xyz))
    out: dict[str, dict[str, object]] = {}
    for label in (int(v) for v in np.unique(array) if v != 0):
        mask = array == label
        components, count = ndimage.label(mask, structure=np.ones((3, 3, 3)))
        del components
        where = np.argwhere(mask)
        lo = where.min(axis=0) + np.asarray(offset_zyx)
        hi = where.max(axis=0) + np.asarray(offset_zyx) + 1
        from .geometry import WorldBox

        box = WorldBox.of_index_box(grid, tuple(int(v) for v in lo), tuple(int(v) for v in hi))  # type: ignore[arg-type]
        entry: dict[str, object] = {
            "voxels": int(mask.sum()),
            "volume_mm3": round(float(mask.sum() * voxel_mm3), 2),
            "components": int(count),
            "bbox_mm": box.to_dict(),
        }
        if names and label in names:
            entry["name"] = names[label]
        out[str(label)] = entry
    return out


def label_overlap(a: np.ndarray, b: np.ndarray) -> dict[str, dict[str, float]]:
    """Per label present in either map: the voxels only in ``a``, only in ``b``, in both. A description, not Dice:
    Dice and the rest come from KonfAI Eval."""
    out: dict[str, dict[str, float]] = {}
    for label in (int(v) for v in np.union1d(np.unique(a), np.unique(b)) if v != 0):
        in_a, in_b = a == label, b == label
        out[str(label)] = {
            "only_a": int((in_a & ~in_b).sum()),
            "only_b": int((~in_a & in_b).sum()),
            "both": int((in_a & in_b).sum()),
        }
    return out


def displacement_stats(field: np.ndarray) -> dict[str, float]:
    """Lengths of the vectors of a field ``[C, ...]`` in its unit (mm)."""
    lengths = np.linalg.norm(np.asarray(field, dtype=np.float64), axis=0).ravel()
    lengths = lengths[np.isfinite(lengths)]
    if lengths.size == 0:
        return {"voxels": 0}
    p95 = np.percentile(lengths, 95)
    return {
        "voxels": int(lengths.size),
        "mean_mm": round(float(lengths.mean()), 4),
        "p95_mm": round(float(p95), 4),
        "max_mm": round(float(lengths.max()), 4),
    }


def jacobian_determinant(field: np.ndarray, grid: Grid) -> np.ndarray:
    """``det(I + du/dx)`` of a field ``[3, Z, Y, X]`` (components x, y, z in world mm) over its voxels, derivatives in
    world axes through the grid's direction and spacing, as KonfAI's ``jacobian_statistics`` takes them. A block one
    voxel thick along an axis has no derivative there (the identity's row)."""
    values = np.asarray(field, dtype=np.float64)
    to_index = np.linalg.inv(grid.matrix @ np.diag(grid.spacing_xyz))  # world offset -> index offset
    # du_c / d(index_a) for a along x, y, z: block axes 3, 2, 1
    by_index = np.stack(
        [np.gradient(values, axis=axis) if values.shape[axis] > 1 else np.zeros_like(values) for axis in (3, 2, 1)],
        axis=-1,
    )  # [c, z, y, x, a]
    gradient = by_index @ to_index  # du_c / dx_w: [c, z, y, x, w]
    jacobian = np.moveaxis(gradient, 0, -2) + np.eye(3)  # [z, y, x, c, w]
    return np.linalg.det(jacobian)
