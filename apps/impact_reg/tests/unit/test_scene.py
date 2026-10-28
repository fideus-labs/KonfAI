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

"""The scene ``impact-reg-konfai scene`` writes, and what it must leave alone."""

from pathlib import Path

import numpy as np
import pytest
from konfai.utils.errors import KonfAIError


def test_a_results_directory_is_never_replaced_by_a_scene(tmp_path: Path) -> None:
    # '-o Output/P000' named the case directory: the scene would have deleted Transform.h5 with it.
    pytest.importorskip("ngff_zarr")
    pytest.importorskip("itk")
    from impact_reg_konfai.scene import write_scene

    case = tmp_path / "P000"
    case.mkdir()
    (case / "Transform.h5").write_bytes(b"kept")
    with pytest.raises(KonfAIError, match="other than an OME-Zarr store"):
        write_scene(case, case, case / "Transform.h5", case)
    assert (case / "Transform.h5").read_bytes() == b"kept"


def test_a_coarser_field_covers_every_fixed_voxel() -> None:
    # Its nodes at the blocks' centres alone stopped 1.5 voxels short of each edge (factor 4), where a reader that does
    # not extrapolate the field moves nothing.
    from impact_reg_konfai.scene import _coarser

    shape, origin, spacing = (9, 10, 11), np.array([1.0, 2.0, 3.0]), np.array([0.5, 0.25, 2.0])
    vectors = np.random.default_rng(0).normal(size=(*shape, 3))
    blocks, start, step = _coarser(vectors, origin, spacing, np.eye(3), 4)
    end = start + step * (np.array(blocks.shape[:3][::-1]) - 1)  # x, y, z, as origin
    assert (start <= origin).all() and (end >= origin + spacing * (np.array(shape[::-1]) - 1)).all()
    np.testing.assert_array_equal(blocks[0], blocks[1])  # the added node repeats the edge block


def test_an_oblique_image_is_refused_and_a_flipped_one_keeps_its_orientation(tmp_path: Path) -> None:
    # Converted as the nearest axis-aligned grid, an oblique pair's rigid result landed 19 voxels off, silently; a
    # flipped or permuted grid converts exactly once its RFC-4 orientation reaches the conversion.
    itk = pytest.importorskip("itk")
    ngff_zarr = pytest.importorskip("ngff_zarr")
    from impact_reg_konfai.scene import _image

    angle = np.radians(20)
    for name, direction in (
        ("oblique", [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]),
        ("flipped", [[0, 0, -1], [1, 0, 0], [0, -1, 0]]),
    ):
        image = itk.image_from_array(np.zeros((4, 5, 6), np.float32))
        image.SetDirection(itk.matrix_from_array(np.array(direction, dtype=np.float64)))
        itk.imwrite(image, str(tmp_path / f"{name}.mha"))
    with pytest.raises(KonfAIError, match="oblique"):
        _image(ngff_zarr, itk, tmp_path / "oblique.mha", "fixed")
    flipped = _image(ngff_zarr, itk, tmp_path / "flipped.mha", "fixed")
    assert flipped.axes_orientations["z"].value == "left-to-right"
