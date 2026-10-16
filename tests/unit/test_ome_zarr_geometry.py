# SPDX-License-Identifier: Apache-2.0
"""Physical positions survive standard OME-Zarr metadata, including patches and pyramid levels."""

from pathlib import Path

import numpy as np
import pytest
from konfai.utils.dataset import Attribute, Dataset, ome_zarr_attributes
from konfai.utils.errors import DatasetManagerError
from konfai.utils.ome_zarr import get_ome_zarr_info, write_ome_zarr

nz = pytest.importorskip("ngff_zarr")
sitk = pytest.importorskip("SimpleITK")
zarr = pytest.importorskip("zarr")


@pytest.mark.parametrize("angle,flip", [(0.37, False), (0.37, True), (0.0, False)])
@pytest.mark.parametrize("dimension", [2, 3])
def test_standard_geometry_maps_voxels_patches_and_levels(
    tmp_path: Path, angle: float, flip: bool, dimension: int
) -> None:
    direction = np.eye(dimension)
    direction[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    if flip:
        direction[:, 0] *= -1
    spacing = np.array([0.7, 1.3, 2.1][:dimension])
    origin = np.array([17.0, -21.0, 33.0][:dimension])
    shape = (1,) + (8,) * dimension
    pixels = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
    attributes = Attribute()
    attributes["Direction"] = direction.ravel()
    attributes["Origin"] = origin
    attributes["Spacing"] = spacing
    attributes["PatientID"] = "procedural"
    path = tmp_path / "case" / "image.ome.zarr"
    path.parent.mkdir()
    write_ome_zarr(path, pixels, spacing=spacing, origin=origin, attributes=attributes, scale_factors=[2])
    from ngff_zarr.validate import validate

    validate(dict(zarr.open_group(str(path), mode="r").attrs), version="0.6")
    multiscales = nz.from_ngff_zarr(str(path))
    private = multiscales.root_attributes["konfai"]["attributes"]
    assert list(private) == ["PatientID_0"]
    world = next(s for s in multiscales.metadata.coordinateSystems if s.name == "world")
    assert all(a.orientation is not None for a in world.axes if a.type == "space")
    transformation = multiscales.metadata.coordinateTransformations[0]
    assert transformation.transformations[0].type == ("affine" if flip else "rotation")
    # A reader independent of KonfAI can apply the stored mapping to physical points.
    wasm = nz.ngff_transform_to_itk_transform(transformation, list(multiscales.images[0].dims))[0]
    matrix = np.asarray(wasm.parameters[: dimension**2]).reshape(dimension, dimension)
    offset = np.asarray(wasm.parameters[dimension**2 :])
    original = sitk.Image([8] * dimension, sitk.sitkUInt16)
    original.SetSpacing(spacing.tolist())
    original.SetOrigin(origin.tolist())
    original.SetDirection(direction.ravel().tolist())
    index = [2] * dimension
    np.testing.assert_allclose(matrix @ (spacing * index) + offset, original.TransformIndexToPhysicalPoint(index))
    data, patch = Dataset(str(tmp_path), "omezarr").read_data_slice(
        "image", "case", (slice(None),) + (slice(2, 6),) * dimension
    )
    np.testing.assert_array_equal(data, pixels[(slice(None),) + (slice(2, 6),) * dimension])
    np.testing.assert_allclose(patch.get_np_array("Origin"), original.TransformIndexToPhysicalPoint(index))
    coarse = ome_zarr_attributes(get_ome_zarr_info(path, 1))
    np.testing.assert_allclose(
        coarse.get_np_array("Origin"), original.TransformContinuousIndexToPhysicalPoint([0.5] * dimension)
    )
    np.testing.assert_allclose(coarse.get_np_array("Direction"), direction.ravel(), atol=1e-14)
    np.testing.assert_allclose(coarse.get_np_array("Spacing"), 2 * spacing)


def test_private_geometry_is_refused_instead_of_silently_lost(tmp_path: Path) -> None:
    image = nz.to_ngff_image(np.zeros((4, 4), np.uint8), dims=["y", "x"])
    multiscales = nz.to_multiscales(image, scale_factors=[])
    multiscales.root_attributes = {"konfai": {"attributes": {"Direction_0": "0 -1 1 0"}}}
    nz.to_ngff_zarr(str(tmp_path), multiscales, version="0.5")
    with pytest.raises(DatasetManagerError, match="private KonfAI"):
        get_ome_zarr_info(tmp_path)


def test_mixed_units_with_a_rotation_keep_physical_positions(tmp_path: Path) -> None:
    attributes = Attribute()
    attributes["Direction"] = [0, -1, 0, 1, 0, 0, 0, 0, 1]
    attributes["OMEUnits"] = "micrometer millimeter micrometer"
    write_ome_zarr(
        tmp_path, np.zeros((1, 4, 4, 4), np.uint8), spacing=[0.02, 0.03, 0.04], origin=[3, 5, 7], attributes=attributes
    )
    geometry = ome_zarr_attributes(get_ome_zarr_info(tmp_path))
    np.testing.assert_allclose(geometry.get_np_array("Origin"), [3, 5, 7])
    np.testing.assert_allclose(geometry.get_np_array("Spacing"), [0.02, 0.03, 0.04])
    np.testing.assert_allclose(geometry.get_np_array("Direction"), attributes.get_np_array("Direction"))


def test_rfc4_ras_frame_is_converted_to_itk_lps(tmp_path: Path) -> None:
    from ngff_zarr.rfc4 import RAS

    image = nz.to_ngff_image(
        np.zeros((4, 4, 4), np.uint8),
        dims=["z", "y", "x"],
        translation={"x": 3.0, "y": 5.0, "z": 7.0},
    )
    image.axes_orientations = RAS
    nz.to_ngff_zarr(str(tmp_path), nz.to_multiscales(image, scale_factors=[]), version="0.6")
    attributes = ome_zarr_attributes(get_ome_zarr_info(tmp_path))
    np.testing.assert_array_equal(attributes.get_np_array("Origin"), [-3, -5, 7])
    np.testing.assert_array_equal(attributes.get_np_array("Direction"), np.diag([-1, -1, 1]).ravel())


def test_multiple_physical_mappings_are_not_silently_selected(tmp_path: Path) -> None:
    import dataclasses

    from ngff_zarr.v06.zarr_metadata import CoordinateSystemIdentifier, Translation

    multiscales = nz.to_multiscales(nz.to_ngff_image(np.zeros((4, 4), np.uint8), dims=["y", "x"]), scale_factors=[])
    intrinsic = multiscales.metadata.intrinsic_coordinate_system
    world = dataclasses.replace(intrinsic, name="world")
    multiscales.metadata.coordinateSystems.append(world)
    multiscales.metadata.coordinateTransformations = [
        Translation(
            input=CoordinateSystemIdentifier(name=intrinsic.name),
            output=CoordinateSystemIdentifier(name=world.name),
            translation=[0, x],
            name=f"alternative_{x}",
        )
        for x in (0, 1)
    ]
    nz.to_ngff_zarr(str(tmp_path), multiscales, version="0.6")
    with pytest.raises(DatasetManagerError, match="unambiguous"):
        get_ome_zarr_info(tmp_path)


def test_a_small_field_rotation_is_not_discarded_as_identity(tmp_path: Path) -> None:
    angle = 1e-8
    direction = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    attributes = Attribute()
    attributes["Direction"] = direction.ravel()
    write_ome_zarr(tmp_path, np.zeros((2, 4, 4), np.float32), attributes=attributes, displacement_field=True)
    geometry = get_ome_zarr_info(tmp_path)["standard_geometry"]
    np.testing.assert_allclose(geometry["Direction"], direction.ravel(), atol=1e-20, rtol=1e-14)


def test_a_nonlinear_image_mapping_is_not_silently_ignored(tmp_path: Path) -> None:
    import dataclasses

    from ngff_zarr.v06.zarr_metadata import CoordinateSystemIdentifier, Displacements

    multiscales = nz.to_multiscales(nz.to_ngff_image(np.zeros((4, 4), np.uint8), dims=["y", "x"]), scale_factors=[])
    intrinsic = multiscales.metadata.intrinsic_coordinate_system
    multiscales.metadata.coordinateSystems.append(dataclasses.replace(intrinsic, name="world"))
    multiscales.metadata.coordinateTransformations = [
        Displacements(
            input=CoordinateSystemIdentifier(name=intrinsic.name),
            output=CoordinateSystemIdentifier(name="world"),
            path="field",
            interpolation="linear",
        )
    ]
    nz.to_ngff_zarr(str(tmp_path), multiscales, version="0.6")
    with pytest.raises(DatasetManagerError, match="nonlinear"):
        get_ome_zarr_info(tmp_path)


def test_standard_geometry_ignores_private_axis_and_unit_hints(tmp_path: Path) -> None:
    image = nz.to_ngff_image(
        np.zeros((4, 4), np.uint8), dims=["y", "x"], scale={"y": 2, "x": 3}, translation={"y": 5, "x": 7}
    )
    multiscales = nz.to_multiscales(image, scale_factors=[])
    multiscales.root_attributes = {"konfai": {"attributes": {"OMEAxes_0": "y x", "OMEUnits_0": "meter meter"}}}
    nz.to_ngff_zarr(str(tmp_path), multiscales, version="0.5")
    geometry = get_ome_zarr_info(tmp_path)["standard_geometry"]
    np.testing.assert_array_equal(geometry["Origin"], [7, 5])
    np.testing.assert_array_equal(geometry["Spacing"], [3, 2])
