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

"""SimpleITK-based helpers for geometric transforms, resampling, and masking."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None  # type: ignore[assignment]

from konfai.utils.errors import TransformError

if TYPE_CHECKING:
    from konfai.data.geometry import AffineMap, AffineStage, DisplacementStage, Grid, SpatialStages


def _require_simpleitk() -> None:
    """Raise a clear project error when an ITK-only path is used without SimpleITK."""
    if sitk is None:
        raise TransformError("SimpleITK is required for this operation. Install it with `pip install konfai[itk]`.")


def read_displacement_field(path: str | Path) -> sitk.Image:
    """A displacement field, from an ITK image file OR an NGFF RFC-5 OME-Zarr store.

    A field written as a store cannot go through ``sitk.ReadImage``, and reading it as an ordinary
    image is worse than failing: the component axis looks like any other, so indexing it yields one
    third of the field and a registration that is wrong without being obviously wrong. This is the one
    place that knows how to open either form, so no caller has to decide.

    The result is ``sitkVectorFloat64`` -- what ``DisplacementFieldTransform`` requires.
    """
    _require_simpleitk()
    path = Path(path)
    if not path.is_dir():
        return sitk.ReadImage(str(path), sitk.sitkVectorFloat64)

    from konfai.utils.dataset import data_to_image, ome_zarr_attributes
    from konfai.utils.ome_zarr import get_ome_zarr_info, is_displacement_field, read_ome_zarr_data_slice

    if not is_displacement_field(path):
        raise TransformError(
            f"'{path}' is an OME-Zarr image, not a displacement field: its component axis is not "
            "typed as an NGFF RFC-5 displacement.",
            "A three-component volume is a perfectly ordinary image, so the store's own declaration "
            "is the only thing that tells them apart -- write the field with "
            "write_ome_zarr(displacement_field=True).",
        )
    axes = get_ome_zarr_info(path)["axes"]
    n_axes = 1 + sum(axis in axes for axis in ("z", "y", "x"))  # channel-first C[Z]YX
    data, metadata = read_ome_zarr_data_slice(path, tuple(slice(None) for _ in range(n_axes)))
    # Origin, Spacing and Direction together: NGFF scale/translation alone cannot express the
    # direction matrix, so the geometry comes from the konfai sidecar through data_to_image.
    field = data_to_image(data, ome_zarr_attributes(metadata))
    return sitk.Cast(field, sitk.sitkVectorFloat64)


def _invert_via_displacement_field(transform: sitk.Transform, image: sitk.Image) -> sitk.DisplacementFieldTransform:
    if image is None:
        raise TransformError(
            "Inverting a non-linear transform requires a reference image to sample the displacement field, "
            "but none was provided."
        )
    displacement_field_filter = sitk.TransformToDisplacementFieldFilter()
    displacement_field_filter.SetReferenceImage(image)
    displacement_field = displacement_field_filter.Execute(transform)
    iterative_inverse = sitk.IterativeInverseDisplacementFieldImageFilter()
    iterative_inverse.SetNumberOfIterations(20)
    return sitk.DisplacementFieldTransform(iterative_inverse.Execute(displacement_field))


def _copy_transform(transform_cls: type[sitk.Transform], transform: sitk.Transform, invert: bool) -> sitk.Transform:
    transform = transform_cls(transform)
    if invert:
        transform = transform_cls(transform.GetInverse())
    return transform


def _image_like(array: np.ndarray, reference: sitk.Image) -> sitk.Image:
    result = sitk.GetImageFromArray(array)
    result.CopyInformation(reference)
    return result


def _open_transform(
    transform_files: dict[str | sitk.Transform, bool], image: sitk.Image = None
) -> list[sitk.Transform]:
    _require_simpleitk()
    transforms: list[sitk.Transform] = []

    for transform_file, invert in transform_files.items():
        if isinstance(transform_file, str):
            transform = sitk.ReadTransform(transform_file + ".itk.txt")
        else:
            transform = transform_file
        if transform.GetName() == "TranslationTransform":
            transform = _copy_transform(sitk.TranslationTransform, transform, invert)
        elif transform.GetName() == "Euler3DTransform":
            transform = _copy_transform(sitk.Euler3DTransform, transform, invert)
        elif transform.GetName() == "VersorRigid3DTransform":
            transform = _copy_transform(sitk.VersorRigid3DTransform, transform, invert)
        elif transform.GetName() == "AffineTransform":
            transform = _copy_transform(sitk.AffineTransform, transform, invert)
        elif transform.GetName() == "DisplacementFieldTransform":
            if invert:
                transform = _invert_via_displacement_field(transform, image)
        else:
            transform = sitk.BSplineTransform(transform)
            if invert:
                transform = _invert_via_displacement_field(transform, image)
        transforms.append(transform)
    if len(transforms) == 0:
        transforms.append(sitk.Euler3DTransform())
    return transforms


def compose_transform(
    transform_files: dict[str | sitk.Transform, bool], image: sitk.Image = None
) -> sitk.CompositeTransform:
    transforms = _open_transform(transform_files, image)
    result = sitk.CompositeTransform(transforms)
    return result


def apply_to_data_transform(data: np.ndarray, transform_files: dict[str | sitk.Transform, bool]) -> np.ndarray:
    transforms = compose_transform(transform_files)
    result = np.copy(data)
    for i in range(data.shape[0]):
        result[i, :] = transforms.TransformPoint(np.asarray(data[i, :], dtype=np.double))
    return result


def box_with_mask(mask: sitk.Image, label: list[int], dilatations: list[int]) -> np.ndarray:
    _require_simpleitk()

    dilatations = [int(np.ceil(d / s)) for d, s in zip(dilatations, reversed(mask.GetSpacing()), strict=False)]

    data = sitk.GetArrayFromImage(mask)
    border = np.where(np.isin(sitk.GetArrayFromImage(mask), label))
    box = []
    for w, dilatation, s in zip(border, dilatations, data.shape, strict=False):
        box.append([max(np.min(w) - dilatation, 0), min(np.max(w) + dilatation, s)])
    box = np.asarray(box)
    return box


def crop_with_mask(image: sitk.Image, box: np.ndarray) -> sitk.Image:
    _require_simpleitk()
    data = sitk.GetArrayFromImage(image)

    for i, w in enumerate(box):
        data = np.delete(data, slice(w[1], data.shape[i]), i)
        data = np.delete(data, slice(0, w[0]), i)

    origin = np.asarray(image.GetOrigin())
    matrix = np.asarray(image.GetDirection()).reshape((len(origin), len(origin)))
    origin = origin.dot(matrix)
    for i, w in enumerate(box):
        origin[-i - 1] += w[0] * np.asarray(image.GetSpacing())[-i - 1]
    origin = origin.dot(np.linalg.inv(matrix))

    result = sitk.GetImageFromArray(data)
    result.SetOrigin(origin)
    result.SetSpacing(image.GetSpacing())
    result.SetDirection(image.GetDirection())
    return result


def _linear_map(transform: sitk.Transform) -> AffineMap:
    """The exact world map of a linear transform: ``T(p) = M p + T(0)``.

    ``M`` comes from ``GetMatrix`` where the type has one -- the number ITK itself resamples with,
    read past the centre/translation parameterisation that differs between Euler, Similarity,
    Scale and Affine -- and from ``T(e_k) - T(0)`` otherwise. The offset is ``T(0)`` directly
    rather than assembled from centre and translation: one call, no cancellation, and true for
    every parameterisation at once.

    Probing is sound HERE and nowhere else in this file: for an affine map the columns are the map,
    exactly, by linearity. For a non-linear one the same arithmetic measures a local gradient and
    extrapolates it, which under-bounds -- which is why a BSpline's affine part is the identity and
    all of its reach lives in the residual.
    """
    from konfai.data.geometry import AffineMap

    rank = int(transform.GetDimension())
    offset = np.asarray(transform.TransformPoint((0.0,) * rank), dtype=np.float64)
    if hasattr(transform, "GetMatrix"):
        matrix = np.asarray(transform.GetMatrix(), dtype=np.float64).reshape(rank, rank)
    else:
        basis = np.eye(rank)
        columns = [
            np.asarray(transform.TransformPoint(tuple(basis[k])), dtype=np.float64) - offset for k in range(rank)
        ]
        matrix = np.stack(columns, axis=1)
    return AffineMap(matrix, offset)


def _grid_of_image(image: sitk.Image) -> Grid:
    from konfai.data.geometry import Grid

    rank = int(image.GetDimension())
    return Grid(
        tuple(int(extent) for extent in reversed(image.GetSize())),
        np.asarray(image.GetOrigin(), dtype=np.float64),
        np.asarray(image.GetSpacing(), dtype=np.float64),
        np.asarray(image.GetDirection(), dtype=np.float64).reshape(rank, rank),
    )


def _displacement_stage(grid: Grid, per_component: list[np.ndarray], order: int, what: str) -> DisplacementStage:
    from konfai.data.geometry import DisplacementStage

    values = np.stack([component.astype(np.float64, copy=False) for component in per_component])
    if not np.isfinite(values).all():
        raise TransformError(
            f"{what} carries a non-finite displacement value, so no bound on its reach exists.",
            "A NaN or infinite coefficient means the transform was written from a failed solve;"
            " re-export it, or drop it from 'transforms:'.",
        )
    return DisplacementStage(grid, values, order)


def decode_transform_stages(transform: sitk.Transform) -> SpatialStages:
    """A stored transform as geometry stages in APPLICATION order, or a refusal naming the type.

    ``CompositeTransform`` applies its member list in REVERSE (the last added runs first — verified
    against SimpleITK, where ``GetNthTransform(0)`` is nonetheless the first added); the reversal is
    normalized here, once, so every consumer reads stages first-applied-first.
    """
    _require_simpleitk()
    if isinstance(transform, sitk.CompositeTransform):
        stages: list[AffineStage | DisplacementStage] = []
        for index in reversed(range(transform.GetNumberOfTransforms())):
            # Downcast restores the member's concrete type where GetNthTransform hands back the
            # generic wrapper, which the isinstance dispatch below cannot read.
            stages.extend(decode_transform_stages(transform.GetNthTransform(index).Downcast()))
        return tuple(stages)
    from konfai.data.geometry import AffineStage

    if isinstance(transform, sitk.BSplineTransform):
        coefficients = transform.GetCoefficientImages()
        arrays = [sitk.GetArrayFromImage(component) for component in coefficients]
        return (
            _displacement_stage(
                _grid_of_image(coefficients[0]), arrays, int(transform.GetOrder()), "this BSpline transform"
            ),
        )
    if isinstance(transform, sitk.DisplacementFieldTransform):
        field = transform.GetDisplacementField()
        array = sitk.GetArrayFromImage(field)  # (Z, Y, X, rank), components (x, y, z)
        components = [np.ascontiguousarray(array[..., k]) for k in range(array.shape[-1])]
        return (_displacement_stage(_grid_of_image(field), components, 1, "this displacement field"),)
    if transform.IsLinear():
        return (AffineStage(_linear_map(transform)),)
    raise TransformError(
        f"A stored '{transform.GetName()}' decomposes into no bounded map: how far a target region"
        " reaches into its source is unknown, so the region it must read is unbounded.",
        "Convert it to a displacement field when it is written, or use a rigid/affine/BSpline"
        " transform, which all decompose.",
    )


def invert_stages(stages: SpatialStages, rank: int) -> SpatialStages | None:
    """The exact inverse of an all-affine decoded map, or ``None`` when one is not algebraic.

    A BSpline or a field inverts by an iterative dense solve, not an algebraic step, and a field
    solved per region is not the restriction of the field solved once — so a non-affine inverse is
    ``None`` here and the caller refuses with the remedy, rather than resampling through a guess.
    """
    from konfai.data.geometry import AffineMap, AffineStage

    if not all(isinstance(stage, AffineStage) for stage in stages):
        return None
    folded = AffineMap.identity(rank)
    for stage in stages:
        folded = folded.then(cast("AffineStage", stage).map)
    return (AffineStage(folded.inverted()),)
