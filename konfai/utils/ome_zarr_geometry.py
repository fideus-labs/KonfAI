# SPDX-License-Identifier: Apache-2.0
"""Map RFC-5 coordinate systems to KonfAI's ITK geometry, without reading voxels."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from konfai.utils.errors import DatasetManagerError


def without_geometry(attributes: dict[str, Any] | None) -> dict[str, Any]:
    from konfai.utils.dataset.attribute import Attribute

    return {
        k: v
        for k, v in (attributes or {}).items()
        if not any(
            Attribute._is_stack_member(k, key) for key in ("Origin", "Spacing", "Direction", "OMEAxes", "OMEUnits")
        )
    }


def declare_geometry(multiscales: Any, attributes: dict[str, Any] | None, units: dict[str, str]) -> None:
    """Declare the intrinsic-to-world mapping; the intrinsic grid starts at zero."""
    from ngff_zarr.rfc4 import LPS
    from ngff_zarr.v06.zarr_metadata import (
        Affine,
        Axis,
        CoordinateSystem,
        CoordinateSystemIdentifier,
        Rotation,
        TransformSequence,
        Translation,
    )

    from konfai.utils.dataset.attribute import Attribute
    from konfai.utils.ome_zarr import millimetres_per_unit

    image = multiscales.images[0]
    spatial = [d for d in image.dims if d in "zyx"]
    n = len(spatial)
    record = Attribute(attributes or {})
    flat = record.get_np_array("Direction") if "Direction" in record else np.eye(n).ravel()
    if flat.size != n * n:
        raise DatasetManagerError(f"OME-Zarr direction must contain {n * n} values.")
    direction = flat.reshape(n, n)
    if not np.isfinite(direction).all() or np.linalg.det(direction) == 0:
        raise DatasetManagerError("OME-Zarr direction must be finite and invertible.")
    factors = np.array([millimetres_per_unit(units.get(d)) or 1.0 for d in spatial])
    # Units belong to coordinates, so a rotation between differently scaled axes is conjugated.
    linear = direction[::-1, ::-1] * factors[None, :] / factors[:, None]
    matrix = np.eye(len(image.dims))
    indices = [image.dims.index(d) for d in spatial]
    matrix[np.ix_(indices, indices)] = linear
    translation = [float(image.translation.get(d, 0.0)) for d in image.dims]
    for d in image.dims:
        image.translation[d] = 0.0
    # to_multiscales has already materialised the per-level transform metadata.
    for dataset in multiscales.metadata.datasets:
        for entry in dataset.coordinateTransformations:
            for part in getattr(entry, "transformations", [entry]):
                if part.type == "translation":
                    part.translation = [0.0] * len(image.dims)
    linear_transform: Rotation | Affine
    if np.allclose(matrix.T @ matrix, np.eye(len(matrix)), atol=1e-10, rtol=0) and np.isclose(np.linalg.det(matrix), 1):
        linear_transform = Rotation(rotation=matrix.tolist())
    else:
        linear_transform = Affine(affine=np.column_stack((matrix, np.zeros(len(matrix)))).tolist())
    intrinsic = multiscales.metadata.intrinsic_coordinate_system
    world = CoordinateSystem(
        name="world",
        axes=[
            Axis(
                name=a.name,
                type=a.type,
                unit=a.unit,
                discrete=a.discrete,
                orientation=LPS.get(a.name) if a.type == "space" else None,
            )
            for a in intrinsic.axes
        ],
    )
    transform = TransformSequence(
        input=CoordinateSystemIdentifier(name=intrinsic.name),
        output=CoordinateSystemIdentifier(name=world.name),
        transformations=[linear_transform, Translation(translation=translation)],
        name="intrinsic_to_world",
    )
    multiscales.metadata = dataclasses.replace(
        multiscales.metadata,
        coordinateSystems=[intrinsic, world],
        coordinateTransformations=[transform],
    )


def read_geometry(multiscales: Any, image: Any) -> dict[str, Any]:
    """Resolve one linear intrinsic-to-world mapping, in millimetres and ITK xyz order."""
    import ngff_zarr
    from ngff_zarr.rfc4 import anatomical_orientation_to_itk_direction

    from konfai.utils.ome_zarr import millimetres_per_unit

    dims = list(image.dims)
    xyz = [d for d in "xyz" if d in dims]
    n = len(xyz)
    metadata = multiscales.metadata
    intrinsic = getattr(metadata, "intrinsic_coordinate_system", None)
    transforms = [t for t in (metadata.coordinateTransformations or []) if t.type != "displacements"]
    matrix, offset = np.eye(n), np.zeros(n)
    units = {k: v for k, v in (image.axes_units or {}).items() if v}
    orientations = {k: v for k, v in (image.axes_orientations or {}).items() if v}
    private = (multiscales.root_attributes or {}).get("konfai", {}).get("attributes", {})
    if not transforms and without_geometry(private) != private:
        raise DatasetManagerError(
            "This OME-Zarr store keeps geometry in private KonfAI attributes.",
            "Rewrite it with RFC-5 coordinate transformations before reading it.",
        )
    if transforms:
        if (
            len(transforms) != 1
            or intrinsic is None
            or transforms[0].input is None
            or transforms[0].output is None
            or transforms[0].input.name != intrinsic.name
        ):
            raise DatasetManagerError("OME-Zarr requires one unambiguous intrinsic-to-world transformation.")
        transform = transforms[0]
        systems = [s for s in metadata.coordinateSystems if s.name == transform.output.name]
        if len(systems) != 1 or [a.name for a in systems[0].axes] != dims:
            raise DatasetManagerError("OME-Zarr world axes must match the intrinsic axes in order.")
        try:
            entry = ngff_zarr.ngff_transform_to_itk_transform(transform, dims)[0]
        except (ValueError, TypeError, NotImplementedError) as exc:
            raise DatasetManagerError(f"Unsupported OME-Zarr grid transformation: {exc}") from exc
        if entry.transformType.transformParameterization.value != "Affine":
            raise DatasetManagerError("A nonlinear OME-Zarr transformation cannot describe an image grid.")
        parameters = np.asarray(entry.parameters)
        matrix, offset = parameters[: n * n].reshape(n, n), parameters[n * n :]
        units = {a.name: a.unit for a in systems[0].axes if a.unit}
        orientations = {a.name: a.orientation for a in systems[0].axes if a.orientation}
    factors = np.array([millimetres_per_unit(units.get(d)) or 1.0 for d in xyz])
    basis = np.eye(n)
    for j, d in enumerate(xyz):
        if d in orientations:
            orientation = orientations[d]
            value = orientation["value"] if isinstance(orientation, dict) else orientation.value
            vector = np.asarray(anatomical_orientation_to_itk_direction(value))
            if np.any(vector[n:] != 0):
                raise DatasetManagerError("OME-Zarr anatomical orientation leaves the image's spatial dimensions.")
            basis[:, j] = vector[:n]
    scale = np.array([image.scale[d] for d in xyz])
    translation = np.array([image.translation[d] for d in xyz])
    linear = basis @ (factors[:, None] * matrix) @ np.diag(scale)
    spacing = np.linalg.norm(linear, axis=0)
    origin = basis @ (factors * (matrix @ translation + offset))
    if (
        not np.isfinite(linear).all()
        or not np.isfinite(origin).all()
        or np.any(spacing <= 0)
        or np.linalg.det(linear) == 0
    ):
        raise DatasetManagerError("OME-Zarr geometry must be finite and invertible, with positive spacing.")
    return {
        "Origin": origin.tolist(),
        "Spacing": spacing.tolist(),
        "Direction": (linear / spacing).ravel().tolist(),
        "units": units,
    }
