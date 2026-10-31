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

"""A 2-D image through the 3-D presets: lifted to one plane, registered as a plane, brought back to 2-D."""

import json
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk
import torch
from impact_reg_konfai.impact_reg import _is_flat_field
from impact_reg_konfai.models.impact_loss import ModelSpec, plane_model
from impact_reg_konfai.models.intensity import EngineRegistration
from impact_reg_konfai.plane import is_flat, lift_field, lift_plane, project_field, project_image
from konfai.utils.dataset import Attribute
from konfai.utils.errors import KonfAIError, TransformError
from konfai.utils.ome_zarr import is_displacement_field, write_ome_zarr


def _flat(values: np.ndarray, spacing=(0.5, 0.25), origin=(3.0, -2.0)) -> sitk.Image:
    image = sitk.GetImageFromArray(values.astype(np.float32))
    image.SetSpacing(spacing)
    image.SetOrigin(origin)
    return image


def test_the_engine_registers_the_plane_of_a_volume_one_plane_thick() -> None:
    # Whole, the volume keeps its z axis on the way in; a tile comes without it, which KonfAI drops from a patch one
    # plane thick. Either way the engine sees the 2-D image, and KonfAI adds the axis back to the field.
    seen = []

    class Engine:
        def register(self, fixed, moving, device_index, fixed_mask, moving_mask):
            seen.append((fixed.GetDimension(), fixed.GetSpacing(), fixed_mask.GetDimension()))
            return np.stack([np.full((4, 5), 0.5), np.full((4, 5), -1.0)]).astype(np.float32)  # in voxels

    geometry = Attribute()
    geometry["Origin"] = np.zeros(3)
    geometry["Spacing"] = np.array([0.1, 0.2, 7.0])  # z, the plane's thickness, is no voxel of the image
    geometry["Direction"] = np.eye(3).flatten()
    for tensor in (torch.zeros(1, 1, 1, 4, 5), torch.zeros(1, 1, 4, 5)):
        field = EngineRegistration(Engine())(tensor, tensor, tensor + 1, tensor + 1, [[geometry]] * 4).numpy()
        assert field.shape == (1, 3, 4, 5)
        np.testing.assert_allclose(field[0, 0], 0.05)
        np.testing.assert_allclose(field[0, 1], -0.1)
        np.testing.assert_allclose(field[0, 2], 0.0)
    assert seen == [(2, pytest.approx((1.0, 2.0)), 2)] * 2
    # The moving plane elsewhere along z is reached by the field, which the plane's registration cannot see.
    elsewhere = Attribute()
    elsewhere["Origin"] = np.array([0.0, 0.0, 1.5])
    elsewhere["Spacing"], elsewhere["Direction"] = geometry["Spacing"], geometry["Direction"]
    tensor = torch.zeros(1, 1, 1, 4, 5)
    field = EngineRegistration(Engine())(
        tensor, tensor, tensor + 1, tensor + 1, [[geometry], [elsewhere], [geometry], [elsewhere]]
    )
    np.testing.assert_allclose(field.numpy()[0, 2], 1.5)


def test_a_file_goes_through_one_plane_and_comes_back_in_its_form(tmp_path: Path) -> None:
    values = np.arange(20.0).reshape(4, 5)
    sitk.WriteImage(_flat(values), str(tmp_path / "image.nii.gz"))

    lifted = lift_plane(tmp_path / "image.nii.gz", tmp_path / "planes", 0.25)
    volume = sitk.ReadImage(str(lifted))
    assert lifted.suffix == ".mha" and volume.GetSize() == (5, 4, 1) and volume.GetSpacing() == (0.5, 0.25, 0.25)
    # A file of its name in another directory lifts beside it, not over it.
    assert lift_plane(tmp_path / "image.nii.gz", tmp_path / "planes" / "001", 0.25) != lifted
    assert volume.GetOrigin() == (3.0, -2.0, 0.0)

    project_image(lifted, tmp_path / "Moved.nii.gz")
    moved = sitk.ReadImage(str(tmp_path / "Moved.nii.gz"))
    assert moved.GetDimension() == 2 and moved.GetSpacing() == (0.5, 0.25) and not lifted.exists()
    np.testing.assert_array_equal(sitk.GetArrayFromImage(moved), values)


def test_a_store_goes_through_one_plane_and_comes_back_a_2d_pyramid_in_its_unit(tmp_path: Path) -> None:
    ngff_zarr = pytest.importorskip("ngff_zarr")
    values = np.arange(512.0 * 512).reshape(512, 512).astype(np.float32)
    image = ngff_zarr.to_ngff_image(
        values,
        dims=["y", "x"],
        scale={"y": 0.25, "x": 0.5},
        translation={"y": -2.0, "x": 3.0},
        axes_units={"y": "micrometer", "x": "micrometer"},
    )
    ngff_zarr.to_ngff_zarr(str(tmp_path / "image.ome.zarr"), ngff_zarr.to_multiscales(image, scale_factors=[]))

    lifted = lift_plane(tmp_path / "image.ome.zarr", tmp_path / "planes", 0.00025)  # 0.25 um, in millimetres
    volume = ngff_zarr.from_ngff_zarr(str(lifted)).images[0]
    assert list(volume.dims) == ["z", "y", "x"] and volume.data.shape == (1, 512, 512)
    assert volume.scale["z"] == pytest.approx(0.25) and volume.axes_units["z"] == "micrometer"

    project_image(lifted, tmp_path / "Moved.ome.zarr")
    moved = ngff_zarr.from_ngff_zarr(str(tmp_path / "Moved.ome.zarr"))
    finest = moved.images[0]
    assert len(moved.images) > 1 and "z" not in finest.dims and finest.axes_units["x"] == "micrometer"
    assert finest.scale["x"] == pytest.approx(0.5) and finest.translation["y"] == pytest.approx(-2.0)
    np.testing.assert_array_equal(np.squeeze(np.asarray(finest.data)), values)


def test_a_one_plane_field_comes_back_as_the_2d_field_it_is(tmp_path: Path) -> None:
    array = np.stack([np.full((1, 4, 5), 1.5), np.full((1, 4, 5), -2.0), np.zeros((1, 4, 5))], axis=-1)
    field = sitk.GetImageFromArray(array, isVector=True)
    field.SetSpacing((0.5, 0.25, 0.25))
    path = tmp_path / "Transform.h5"
    sitk.WriteTransform(sitk.DisplacementFieldTransform(field), str(path))

    project_field(path)
    transform = sitk.DisplacementFieldTransform(sitk.ReadTransform(str(path)))
    assert transform.GetDimension() == 2
    assert transform.GetDisplacementField().GetSpacing() == (0.5, 0.25)
    np.testing.assert_allclose(transform.TransformPoint((1.0, 0.5)), (2.5, -1.5))


def test_a_2d_field_goes_one_plane_thick_for_the_later_steps(tmp_path: Path) -> None:
    array = np.stack([np.full((4, 5), 1.5), np.full((4, 5), -2.0)], axis=-1)
    field = sitk.GetImageFromArray(array, isVector=True)
    field.SetSpacing((0.5, 0.25))
    field.SetOrigin((3.0, -2.0))
    path = tmp_path / "Transform.h5"
    sitk.WriteTransform(sitk.DisplacementFieldTransform(field), str(path))

    lifted = lift_field(path, tmp_path / "planes")
    transform = sitk.DisplacementFieldTransform(sitk.ReadTransform(str(lifted)))
    grid = transform.GetDisplacementField()
    assert (
        grid.GetSize() == (5, 4, 1) and grid.GetSpacing() == (0.5, 0.25, 0.25) and grid.GetOrigin() == (3.0, -2.0, 0.0)
    )
    np.testing.assert_allclose(transform.TransformPoint((3.5, -1.5, 0.0)), (5.0, -3.5, 0.0))

    project_field(lifted)
    np.testing.assert_allclose(sitk.ReadTransform(str(lifted)).TransformPoint((3.5, -1.5)), (5.0, -3.5))

    with pytest.raises(KonfAIError, match="ITK transform file or an OME-Zarr store"):
        project_field(tmp_path / "Field.mha")

    rigid = tmp_path / "Rigid.tfm"
    sitk.WriteTransform(sitk.Euler2DTransform(), str(rigid))
    with pytest.raises(ValueError, match="reads displacement fields"):
        lift_field(rigid, tmp_path / "planes")


def test_a_one_plane_store_comes_back_as_a_2d_rfc5_field_and_goes_one_plane_thick_again(tmp_path: Path) -> None:
    ngff_zarr = pytest.importorskip("ngff_zarr")
    # KonfAI hands a field in millimetres, (dx, dy, dz): 1.5 and -2 mm, on a grid in micrometres.
    values = np.stack([np.full((1, 4, 5), 1.5), np.full((1, 4, 5), -2.0), np.full((1, 4, 5), 7.0)]).astype(np.float32)
    path = tmp_path / "Transform.ome.zarr"
    write_ome_zarr(
        path,
        values,
        spacing=(0.5, 0.25, 0.25),
        origin=(3.0, -2.0, 0.0),
        attributes={"OMEUnits": "micrometer micrometer micrometer"},
        displacement_field=True,
    )
    assert not _is_flat_field(path)

    project_field(path)

    assert _is_flat_field(path) and is_displacement_field(path)
    assert [entry.name for entry in tmp_path.iterdir()] == [path.name]  # nothing left beside it
    ome = json.loads((path / "zarr.json").read_text(encoding="utf-8"))["attributes"]["ome"]
    assert ome["multiscales"][0]["coordinateTransformations"][0]["type"] == "displacements"
    field = ngff_zarr.from_ome_zarr(str(path)).images[0]
    assert list(field.dims) == ["c", "y", "x"] and field.data.dtype == np.float32
    assert field.scale["x"] == pytest.approx(500.0) and field.axes_units["y"] == "micrometer"
    assert field.translation["x"] == pytest.approx(3000.0)
    # RFC-5: the components in the axes' order and unit, the one along z gone.
    np.testing.assert_allclose(np.asarray(field.data), np.stack([np.full((4, 5), -2000.0), np.full((4, 5), 1500.0)]))

    lifted = lift_field(path, tmp_path / "planes")
    assert lifted.name == "Transform.h5"
    transform = sitk.DisplacementFieldTransform(sitk.ReadTransform(str(lifted)))
    assert transform.GetDisplacementField().GetSize() == (5, 4, 1)
    np.testing.assert_allclose(transform.TransformPoint((3.5, -1.5, 0.0)), (5.0, -3.5, 0.0))


def test_an_image_store_is_no_flat_field_to_lift(tmp_path: Path) -> None:
    path = tmp_path / "Image.ome.zarr"
    write_ome_zarr(path, np.zeros((1, 4, 5), dtype=np.float32), spacing=(0.5, 0.25), origin=(0.0, 0.0))
    assert _is_flat_field(path)
    with pytest.raises(TransformError, match="not a displacement field"):
        lift_field(path, tmp_path / "planes")


def test_a_3d_model_gives_way_to_its_2d_version_or_is_refused() -> None:
    mind = ModelSpec(ref="VBoussot/impact-torchscript-models:MIND/R1D2_3D.pt", voxel_size=[1.5, 1.5, 1.5])
    assert plane_model(mind).ref.endswith("MIND/R1D2_2D.pt") and plane_model(mind).voxel_size == [1.5, 1.5, 1.5]
    with pytest.raises(ValueError, match="no 2-D version"):
        plane_model(ModelSpec(ref="VBoussot/impact-torchscript-models:TS/M730.pt"))


def test_a_header_tells_a_2d_image_from_a_volume_one_plane_thick() -> None:
    assert is_flat([1, 680, 535]) and not is_flat([1, 1, 680, 535])
