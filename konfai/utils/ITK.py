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

import numpy as np
import torch

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None  # type: ignore[assignment]
import torch.nn.functional as F

from konfai.utils.errors import TransformError


def _require_simpleitk() -> None:
    """Raise a clear project error when an ITK-only path is used without SimpleITK."""
    if sitk is None:
        raise TransformError("SimpleITK is required for this operation. Install it with `pip install konfai[itk]`.")


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


def _open_rigid_transform(transform_files: dict[str | sitk.Transform, bool]) -> tuple[np.ndarray, np.ndarray]:
    transforms = _open_transform(transform_files)
    matrix_result = np.identity(3)
    translation_result = np.array([0, 0, 0])

    for transform in transforms:
        if hasattr(transform, "GetMatrix"):
            matrix = np.linalg.inv(np.array(transform.GetMatrix(), dtype=np.double).reshape((3, 3)))
            translation = -np.asarray(transform.GetTranslation(), dtype=np.double)
            center = np.asarray(transform.GetCenter(), dtype=np.double)
        else:
            matrix = np.eye(len(transform.GetOffset()))
            translation = -np.asarray(transform.GetOffset(), dtype=np.double)
            center = np.asarray([0] * len(transform.GetOffset()), dtype=np.double)

        translation_center = np.linalg.inv(matrix).dot(matrix.dot(translation - center) + center)
        translation_result = np.linalg.inv(matrix_result).dot(translation_center) + translation_result
        matrix_result = matrix.dot(matrix_result)
    return np.linalg.inv(matrix_result), -translation_result


def compose_transform(
    transform_files: dict[str | sitk.Transform, bool], image: sitk.Image = None
) -> sitk.CompositeTransform:
    transforms = _open_transform(transform_files, image)
    result = sitk.CompositeTransform(transforms)
    return result


def flatten_transform(transform_files: dict[str | sitk.Transform, bool]) -> sitk.AffineTransform:
    [matrix, translation] = _open_rigid_transform(transform_files)
    transform = sitk.AffineTransform(3)
    transform.SetMatrix(matrix.flatten())
    transform.SetTranslation(translation)
    return transform


def apply_to_image_rigid_transform(image: sitk.Image, transform_files: dict[str | sitk.Transform, bool]) -> sitk.Image:
    [matrix, translation] = _open_rigid_transform(transform_files)
    matrix = np.linalg.inv(matrix)
    translation = -translation
    data = sitk.GetArrayFromImage(image)
    result = sitk.GetImageFromArray(data)
    result.SetDirection(matrix.dot(np.array(image.GetDirection()).reshape((3, 3))).flatten())
    result.SetOrigin(matrix.dot(np.array(image.GetOrigin()) + translation))
    result.SetSpacing(image.GetSpacing())
    return result


def apply_to_data_transform(data: np.ndarray, transform_files: dict[str | sitk.Transform, bool]) -> np.ndarray:
    transforms = compose_transform(transform_files)
    result = np.copy(data)
    # _LPS = lambda matrix: np.array([-matrix[0], -matrix[1], matrix[2]], dtype=np.double)
    for i in range(data.shape[0]):
        result[i, :] = transforms.TransformPoint(np.asarray(data[i, :], dtype=np.double))
    return result


def resample_itk(
    image_reference: sitk.Image,
    image: sitk.Image,
    transform_files: dict[str | sitk.Transform, bool],
    mask=False,
    default_pixel_value: float | None = None,
    torch_resample: bool = False,
) -> sitk.Image:
    _require_simpleitk()
    if torch_resample:
        input_tensor = torch.tensor(sitk.GetArrayFromImage(image)).unsqueeze(0)
        vectors = [torch.arange(0, s) for s in input_tensor.shape[1:]]
        grids = torch.meshgrid(vectors, indexing="ij")
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        transform_to_displacement_field_filter = sitk.TransformToDisplacementFieldFilter()
        transform_to_displacement_field_filter.SetReferenceImage(image)
        transform_to_displacement_field_filter.SetNumberOfThreads(16)
        new_locs = grid + torch.tensor(
            sitk.GetArrayFromImage(
                transform_to_displacement_field_filter.Execute(compose_transform(transform_files, image))
            )
        ).unsqueeze(0).permute(0, 4, 1, 2, 3)
        shape = new_locs.shape[2:]
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)
        new_locs = new_locs.permute(0, 2, 3, 4, 1)
        new_locs = new_locs[..., [2, 1, 0]]
        result_data = F.grid_sample(
            input_tensor.unsqueeze(0).float(),
            new_locs.float(),
            align_corners=True,
            padding_mode="border",
            mode="nearest" if input_tensor.dtype == torch.uint8 else "bilinear",
        ).squeeze(0)
        result_data = result_data.type(torch.uint8) if input_tensor.dtype == torch.uint8 else result_data
        result = sitk.GetImageFromArray(result_data.squeeze(0).numpy())
        result.CopyInformation(image_reference)
        return result
    else:
        return sitk.Resample(
            image,
            image_reference,
            compose_transform(transform_files, image),
            sitk.sitkNearestNeighbor if mask else sitk.sitkBSpline,
            (
                default_pixel_value
                if default_pixel_value is not None
                else (0 if mask else int(np.min(sitk.GetArrayFromImage(image))))
            ),
        )


def parametermap_to_transform(
    path_src: str,
) -> sitk.Transform | list[sitk.Transform]:
    _require_simpleitk()
    transform = sitk.ReadParameterFile(path_src)

    def array_format(x):
        return [float(i) for i in x]

    dimension = int(transform["FixedImageDimension"][0])

    if transform["Transform"][0] == "EulerTransform":
        if dimension == 2:
            result = sitk.Euler2DTransform()
        else:
            result = sitk.Euler3DTransform()
        parameters = array_format(transform["TransformParameters"])
        fixed_parameters = [*array_format(transform["CenterOfRotationPoint"]), 0]
    elif transform["Transform"][0] == "TranslationTransform":
        result = sitk.TranslationTransform(dimension)
        parameters = array_format(transform["TransformParameters"])
        fixed_parameters = []
    elif transform["Transform"][0] == "AffineTransform":
        result = sitk.AffineTransform(dimension)
        parameters = array_format(transform["TransformParameters"])
        fixed_parameters = [*array_format(transform["CenterOfRotationPoint"]), 0]
    elif transform["Transform"][0] == "BSplineStackTransform":
        parameters = array_format(transform["TransformParameters"])
        grid_size = array_format(transform["GridSize"])
        grid_origin = array_format(transform["GridOrigin"])
        grid_spacing = array_format(transform["GridSpacing"])
        grid_direction = (
            np.asarray(array_format(transform["GridDirection"])).reshape((dimension, dimension)).T.flatten()
        )
        fixed_parameters = np.concatenate([grid_size, grid_origin, grid_spacing, grid_direction])

        nb = int(array_format(transform["Size"])[-1])
        sub = int(np.prod(grid_size)) * dimension
        results = []
        for i in range(nb):
            result = sitk.BSplineTransform(dimension)
            sub_parameters = np.asarray(parameters[i * sub : (i + 1) * sub])
            result.SetFixedParameters(fixed_parameters)
            result.SetParameters(sub_parameters)
            results.append(result)
        return results
    elif transform["Transform"][0] == "AffineLogStackTransform":
        parameters = array_format(transform["TransformParameters"])
        fixed_parameters = [*array_format(transform["CenterOfRotationPoint"]), 0]

        nb = int(transform["NumberOfSubTransforms"][0])
        sub = dimension * 4
        results = []
        for i in range(nb):
            result = sitk.AffineTransform(dimension)
            sub_parameters = np.asarray(parameters[i * sub : (i + 1) * sub])
            result.SetFixedParameters(fixed_parameters)

            matrix = torch.from_numpy(sub_parameters[: dimension * dimension].reshape(dimension, dimension)).to(
                torch.float64
            )
            matrix_exp = torch.matrix_exp(matrix).cpu().numpy().reshape(-1)

            params = np.concatenate([matrix_exp, sub_parameters[-dimension:]])
            result.SetParameters(params)
            results.append(result)
        return results
    elif transform["Transform"][0] == "BSplineTransform":
        result = sitk.BSplineTransform(dimension)

        parameters = array_format(transform["TransformParameters"])
        grid_size = array_format(transform["GridSize"])
        grid_origin = array_format(transform["GridOrigin"])
        grid_spacing = array_format(transform["GridSpacing"])
        grid_direction = np.array(array_format(transform["GridDirection"])).reshape((dimension, dimension)).T.flatten()
        fixed_parameters = np.concatenate([grid_size, grid_origin, grid_spacing, grid_direction])
    else:
        raise NameError(f"Transform {transform['Transform'][0]} doesn't exist")
    result.SetFixedParameters(fixed_parameters)
    result.SetParameters(parameters)
    return result


def _resample(data: torch.Tensor, size: list[int]) -> torch.Tensor:
    if data.dtype == torch.uint8:
        mode = "nearest"
    elif len(data.shape) < 4:
        mode = "bilinear"
    else:
        mode = "trilinear"
    return (
        torch.nn.functional.interpolate(
            data.type(torch.float32).unsqueeze(0),
            size=tuple(reversed(size)),
            mode=mode,
        )
        .squeeze(0)
        .type(data.dtype)
    )


def resample_isotropic(image: sitk.Image, spacing: list[float] | None = None) -> sitk.Image:
    _require_simpleitk()
    spacing = spacing or [1.0, 1.0, 1.0]
    resize_factor = [y / x for x, y in zip(spacing, image.GetSpacing(), strict=False)]
    result = sitk.GetImageFromArray(
        _resample(
            torch.tensor(sitk.GetArrayFromImage(image)).unsqueeze(0),
            [int(size * factor) for size, factor in zip(image.GetSize(), resize_factor, strict=False)],
        )
        .squeeze(0)
        .numpy()
    )
    result.SetDirection(image.GetDirection())
    result.SetOrigin(image.GetOrigin())
    result.SetSpacing(spacing)
    return result


def resample_resize(image: sitk.Image, size: list[int] | None = None):
    _require_simpleitk()
    size = size or [100, 512, 512]
    result = sitk.GetImageFromArray(
        _resample(torch.tensor(sitk.GetArrayFromImage(image)).unsqueeze(0), size).squeeze(0).numpy()
    )
    result.SetDirection(image.GetDirection())
    result.SetOrigin(image.GetOrigin())
    result.SetSpacing([x / y * z for x, y, z in zip(image.GetSize(), size, image.GetSpacing(), strict=False)])
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


def format_mask_label(mask: sitk.Image, labels: list[tuple[int, int]]) -> sitk.Image:
    _require_simpleitk()
    data = sitk.GetArrayFromImage(mask)
    result_data = np.zeros_like(data, np.uint8)

    for label_old, label_new in labels:
        result_data[np.where(data == label_old)] = label_new

    result = sitk.GetImageFromArray(result_data)
    result.CopyInformation(mask)
    return result


def get_flat_label(mask: sitk.Image, labels: None | list[int] = None) -> sitk.Image:
    _require_simpleitk()
    data = sitk.GetArrayFromImage(mask)
    result_data = np.zeros_like(data, np.uint8)
    if labels is not None:
        for label in labels:
            result_data[np.where(data == label)] = 1
    else:
        result_data[np.where(data > 0)] = 1
    result = sitk.GetImageFromArray(result_data)
    result.CopyInformation(mask)
    return result


def clip_and_cast(image: sitk.Image, min_value: float, max_value: float, dtype: np.dtype) -> sitk.Image:
    _require_simpleitk()
    data = sitk.GetArrayFromImage(image)
    data[np.where(data > max_value)] = max_value
    data[np.where(data < min_value)] = min_value
    result = sitk.GetImageFromArray(data.astype(dtype))
    result.CopyInformation(image)
    return result
