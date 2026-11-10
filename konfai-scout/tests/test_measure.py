# SPDX-License-Identifier: Apache-2.0
import numpy as np
from konfai.data.geometry import Grid
from konfai_scout.measure import intensity_stats, label_stats


def test_label_stats_count_volume_pieces_and_box():
    grid = Grid((6, 10, 10), np.zeros(3), np.array([1.0, 2.0, 3.0]), np.eye(3))
    labels = np.zeros((6, 10, 10), dtype=np.int16)
    labels[1:3, 1:3, 1:3] = 1
    labels[4:6, 7:9, 7:9] = 1
    labels[0, 0, 0] = 2
    stats = label_stats(labels, grid, {1: "a"})
    assert stats["1"]["voxels"] == 16 and stats["1"]["components"] == 2 and stats["1"]["volume_mm3"] == 96.0
    assert stats["1"]["name"] == "a" and stats["2"]["components"] == 1
    assert stats["1"]["bbox_mm"]["min"] == [0.5, 1.0, 1.5]


def test_intensities_inside_a_mask():
    values = np.arange(100, dtype=float).reshape(10, 10)
    stats = intensity_stats(values, values > 50)
    assert stats["voxels"] == 49 and stats["min"] == 51 and stats["max"] == 99
    assert intensity_stats(np.array([np.nan]))["voxels"] == 0
