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

import os
from pathlib import Path

import numpy as np
import pytest
from konfai.utils.errors import KonfAIError


def test_a_scene_replaces_neither_an_input_nor_a_directory_that_is_not_a_scene(tmp_path: Path) -> None:
    # '-o Output/P000' named the case directory: the scene would have deleted Transform.h5 with it. '-o' naming an
    # input store deleted it before its data was read, and any other OME-Zarr store passed for a scene.
    pytest.importorskip("ngff_zarr")
    pytest.importorskip("itk")
    from impact_reg_konfai.scene import write_scene

    case = tmp_path / "P000"
    case.mkdir()
    (case / "Transform.h5").write_bytes(b"kept")
    store, image = tmp_path / "fixed.ome.zarr", tmp_path / "image.ome.zarr"
    for path in (store, image):
        path.mkdir()
        (path / "zarr.json").write_text('{"attributes": {"ome": {"multiscales": []}}}')
    for output in (case, store):
        with pytest.raises(KonfAIError, match="overlaps the input"):
            write_scene(store, store, case / "Transform.h5", output)
    with pytest.raises(KonfAIError, match="other than an OME-Zarr scene"):
        write_scene(store, store, case / "Transform.h5", image)
    assert (case / "Transform.h5").read_bytes() == b"kept" and (store / "zarr.json").is_file()


def test_a_store_keeps_its_orientation_each_axis_its_unit_and_a_failed_write_the_scene_before(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The scene dropped a store's RFC-4 orientation and converted the whole field with the x axis's unit: a 2 um
    # shift along z, on a store in nanometres along x, came out as 2000. A failed archive took the previous one along.
    itk = pytest.importorskip("itk")
    ngff_zarr = pytest.importorskip("ngff_zarr")
    from impact_reg_konfai.scene import write_scene

    data = np.zeros((4, 5, 6), np.float32)
    data[1:3, 1:4, 2:5] = 1
    image = ngff_zarr.to_ngff_image(
        data,
        dims=["z", "y", "x"],
        scale={"z": 2.0, "y": 650.0, "x": 650.0},
        translation={"z": 0.0, "y": 0.0, "x": 0.0},
        axes_units={"z": "micrometer", "y": "nanometer", "x": "nanometer"},
    )
    values = ngff_zarr.AnatomicalOrientationValues
    image.axes_orientations = {
        "z": ngff_zarr.AnatomicalOrientation(values.inferior_to_superior),
        "y": ngff_zarr.AnatomicalOrientation(values.posterior_to_anterior),
        "x": ngff_zarr.AnatomicalOrientation(values.left_to_right),
    }
    for name in ("fixed", "moving"):
        ngff_zarr.to_ome_zarr(str(tmp_path / f"{name}.ome.zarr"), ngff_zarr.to_multiscales(image, scale_factors=[]))
    voxel = np.array([650e-6, 650e-6, 2e-3])  # mm along x, y, z: one voxel each way
    field = itk.image_from_array(np.broadcast_to(voxel, (4, 5, 6, 3)).copy(), is_vector=True)
    field.SetSpacing(voxel.tolist())
    transform = itk.DisplacementFieldTransform[itk.D, 3].New()
    transform.SetDisplacementField(field)
    itk.transformwrite([transform], str(tmp_path / "Transform.h5"))
    paths = (tmp_path / "fixed.ome.zarr", tmp_path / "moving.ome.zarr", tmp_path / "Transform.h5")

    write_scene(*paths, tmp_path / "Scene.ozx")
    scene = ngff_zarr.NgffScene.from_ome_zarr(str(tmp_path / "Scene.ozx"))
    np.testing.assert_allclose(np.asarray(scene.coordinateTransformations[0].affine)[:, 3], [2.0, 650.0, 650.0])
    assert scene.images["fixed"].images[0].axes_orientations == image.axes_orientations

    written = (tmp_path / "Scene.ozx").read_bytes()

    def interrupted(*args: object, **kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(ngff_zarr, "write_store_to_zip", interrupted)
    with pytest.raises(OSError, match="disk full"):
        write_scene(*paths, tmp_path / "Scene.ozx")
    assert (tmp_path / "Scene.ozx").read_bytes() == written

    write_scene(*paths, tmp_path / "Scene.zarr")  # a directory store: the scene before is moved aside, not deleted
    kept = (tmp_path / "Scene.zarr" / "zarr.json").read_text()
    replace = os.replace

    def unpublished(source: object, target: object) -> None:
        if Path(str(source)).name == "scene.zarr":
            raise OSError("rename failed")
        replace(source, target)

    monkeypatch.setattr(os, "replace", unpublished)
    with pytest.raises(OSError, match="rename failed"):
        write_scene(*paths, tmp_path / "Scene.zarr")
    assert (tmp_path / "Scene.zarr" / "zarr.json").read_text() == kept
    assert not list(tmp_path.glob(".scene_*"))


def test_a_permuted_field_grid_takes_each_axis_unit_and_an_oblique_one_is_refused(tmp_path: Path) -> None:
    # The spacing was scaled by the physical axes in index order: a grid whose first index runs along z took x's unit.
    itk = pytest.importorskip("itk")
    from impact_reg_konfai.scene import _field

    field = itk.image_from_array(np.zeros((2, 3, 4, 3)), is_vector=True)
    field.SetSpacing([1.0, 2.0, 3.0])
    field.SetDirection(itk.matrix_from_array(np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])))
    transform = itk.DisplacementFieldTransform[itk.D, 3].New()
    transform.SetDisplacementField(field)
    itk.transformwrite([transform], str(tmp_path / "Transform.h5"))
    spacing = _field(itk, tmp_path / "Transform.h5", np.array([1.0, 10.0, 100.0]))[2]
    np.testing.assert_allclose(spacing, [100.0, 20.0, 3.0])
    angle = np.radians(20)  # an oblique grid: no axis of its own to take a unit from
    field.SetDirection(
        itk.matrix_from_array(
            np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        )
    )
    transform.SetDisplacementField(field)  # its fixed parameters, the grid written, are taken here
    itk.transformwrite([transform], str(tmp_path / "Transform.h5"))
    with pytest.raises(KonfAIError, match="oblique"):
        _field(itk, tmp_path / "Transform.h5", np.ones(3))


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
    flipped, _ = _image(ngff_zarr, itk, tmp_path / "flipped.mha", "fixed")
    assert flipped.axes_orientations["z"].value == "left-to-right"
