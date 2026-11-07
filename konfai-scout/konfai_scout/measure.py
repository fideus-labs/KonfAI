# SPDX-License-Identifier: Apache-2.0
"""Descriptive statistics of a region, measured on the case alone: no reference, no metric.

What needs a reference (Dice, MAE, TRE...) and what KonfAI already measures (a field's Jacobian, bending energy,
displacement lengths) go through KonfAI Eval (:mod:`konfai_scout.evaluation`). Here: how much of each label a region
holds and in how many pieces, the intensities, the length of the displacements; plus the Jacobian determinant of a
block for display, by KonfAI's own derivative (:func:`konfai.utils.ITK.field_jacobian_determinant`).
"""

from __future__ import annotations

import numpy as np
from konfai.data.geometry import Grid

from .planes import box_to_dict, world_box_of


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
    voxel_mm3 = float(np.prod(np.asarray(grid.spacing_xyz, dtype=np.float64)))
    out: dict[str, dict[str, object]] = {}
    for label in (int(v) for v in np.unique(array) if v != 0):
        mask = array == label
        _, count = ndimage.label(mask, structure=np.ones((3, 3, 3)))
        where = np.argwhere(mask)
        lo = where.min(axis=0) + np.asarray(offset_zyx)
        hi = where.max(axis=0) + np.asarray(offset_zyx) + 1
        window = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi, strict=True))
        entry: dict[str, object] = {
            "voxels": int(mask.sum()),
            "volume_mm3": round(float(mask.sum() * voxel_mm3), 2),
            "components": int(count),
            "bbox_mm": box_to_dict(world_box_of(grid, window)),
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
    """``det(I + du/dx)`` of a field block ``[3, Z, Y, X]`` (components x, y, z in world mm) on ``grid``, by
    KonfAI's own derivative: the one its Jacobian statistics are computed with."""
    from konfai.utils.ITK import field_jacobian_determinant

    to_index = np.linalg.inv(np.asarray(grid.direction_xyz, dtype=np.float64) @ np.diag(np.asarray(grid.spacing_xyz)))
    return field_jacobian_determinant(np.asarray(field, dtype=np.float64), to_index)
