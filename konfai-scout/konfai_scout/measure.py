# SPDX-License-Identifier: Apache-2.0
"""Descriptive statistics of a region, measured on the case alone: no reference, no metric.

What needs a reference (Dice, MAE, TRE...) and what KonfAI already measures (a field's Jacobian, bending energy,
displacement lengths) go through KonfAI Eval (:mod:`konfai_scout.evaluation`). Here: how much of each label a region
holds and in how many pieces, and the intensities.
"""

from __future__ import annotations

import numpy as np
from konfai.data.geometry import Grid

from .planes import box_of


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
        entry: dict[str, object] = {
            "voxels": int(mask.sum()),
            "volume_mm3": round(float(mask.sum() * voxel_mm3), 2),
            "components": int(count),
            "bbox_mm": box_of(mask, grid, offset_zyx),
        }
        if names and label in names:
            entry["name"] = names[label]
        out[str(label)] = entry
    return out
