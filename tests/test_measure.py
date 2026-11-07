# SPDX-License-Identifier: Apache-2.0
import numpy as np
from konfai_scout.geometry import Grid
from konfai_scout.measure import displacement_stats, intensity_stats, jacobian_determinant, label_overlap, label_stats


def test_label_stats_count_volume_pieces_and_box():
    grid = Grid((10, 10, 6), (1.0, 2.0, 3.0), (0, 0, 0), tuple(np.eye(3).ravel()))
    labels = np.zeros((6, 10, 10), dtype=np.int16)
    labels[1:3, 1:3, 1:3] = 1
    labels[4:6, 7:9, 7:9] = 1
    labels[0, 0, 0] = 2
    stats = label_stats(labels, grid, {1: "a"})
    assert stats["1"]["voxels"] == 16 and stats["1"]["components"] == 2 and stats["1"]["volume_mm3"] == 96.0
    assert stats["1"]["name"] == "a" and stats["2"]["components"] == 1
    assert stats["1"]["bbox_mm"]["min"] == [0.5, 1.0, 1.5]


def test_jacobian_of_identity_and_of_a_scaling():
    grid = Grid((8, 8, 8), (1.0, 1.0, 1.0), (0, 0, 0), tuple(np.eye(3).ravel()))
    field = np.zeros((3, 8, 8, 8))
    assert np.allclose(jacobian_determinant(field, grid), 1.0)
    xx = np.indices((8, 8, 8))[2].astype(float)
    field[0] = 0.5 * xx  # x' = 1.5 x
    assert np.allclose(jacobian_determinant(field, grid)[:, :, 1:-1], 1.5)
    oblique = Grid(
        (8, 8, 8), (0.5, 1.0, 2.0), (0, 0, 0), tuple(np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], float).ravel())
    )
    assert np.allclose(jacobian_determinant(np.zeros((3, 8, 8, 8)), oblique), 1.0)


def test_intensity_displacement_and_overlap():
    values = np.arange(100, dtype=float).reshape(10, 10)
    stats = intensity_stats(values, values > 50)
    assert stats["voxels"] == 49 and stats["min"] == 51 and stats["max"] == 99
    assert intensity_stats(np.array([np.nan]))["voxels"] == 0
    field = np.zeros((3, 4, 4, 4))
    field[1] = 3.0
    assert displacement_stats(field) == {"voxels": 64, "mean_mm": 3.0, "p95_mm": 3.0, "max_mm": 3.0}
    a, b = np.array([[1, 1, 0]]), np.array([[1, 0, 2]])
    assert label_overlap(a, b) == {
        "1": {"only_a": 1, "only_b": 0, "both": 1},
        "2": {"only_a": 0, "only_b": 1, "both": 0},
    }
