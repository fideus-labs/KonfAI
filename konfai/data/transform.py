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

"""Tensor and image transforms used in KonfAI preprocessing and postprocessing."""

import os
import tempfile
from abc import ABC, abstractmethod
from multiprocessing import current_process, get_context
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F

from konfai import cuda_visible_devices
from konfai.utils.config import apply_config
from konfai.utils.dataset import Attribute, Dataset, data_to_image, image_to_data
from konfai.utils.errors import TransformError
from konfai.utils.ITK import box_with_mask, crop_with_mask
from konfai.utils.runtime import NeedDevice
from konfai.utils.utils import get_module, split_path_spec


class Transform(NeedDevice, ABC):
    """Base class for transforms operating on tensors and cached attributes."""

    supports_dataloader_workers = True

    def __init__(self) -> None:
        NeedDevice.__init__(self)
        self.datasets: list[Dataset] = []

    def set_datasets(self, datasets: list[Dataset]):
        self.datasets = datasets

    def transform_shape(self, group_src: str, name: str, shape: list[int], cache_attribute: Attribute) -> list[int]:
        return shape

    @abstractmethod
    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        pass


class TransformInverse(Transform, ABC):
    """Base class for transforms that can also invert their effect."""

    def __init__(self, inverse: bool) -> None:
        super().__init__()
        self.apply_inverse = inverse

    @abstractmethod
    def inverse(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        pass


class TransformLoader:
    """Resolve and instantiate transform classes from KonfAI configuration."""

    def __init__(self) -> None:
        pass

    def get_transform(self, classpath: str, konfai_args: str) -> Transform:
        module, name = get_module(classpath, "konfai.data.transform")
        return apply_config(f"{konfai_args}.{classpath}")(getattr(module, name))()


class Clip(Transform):
    """Clip tensor intensities to a fixed or data-dependent value range."""

    def __init__(
        self,
        min_value: float | str = -1024,
        max_value: float | str = 1024,
        save_clip_min: bool = False,
        save_clip_max: bool = False,
        mask: str | None = None,
    ) -> None:
        super().__init__()
        if isinstance(min_value, float) and isinstance(max_value, float) and max_value <= min_value:
            raise ValueError(
                f"[Clip] Invalid clipping range: max_value ({max_value}) must be greater than min_value ({min_value})"
            )
        self.min_value = min_value
        self.max_value = max_value
        self.save_clip_min = save_clip_min
        self.save_clip_max = save_clip_max
        self.mask = mask

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        mask = None
        if self.mask is not None:
            for dataset in self.datasets:
                if dataset.is_dataset_exist(self.mask, name):
                    mask, _ = dataset.read_data(self.mask, name)
                    break
        if mask is None and self.mask is not None:
            raise ValueError(
                f"Requested mask '{self.mask}' is not present in any dataset. "
                "Check your dataset group names or configuration."
            )
        if mask is None:
            tensor_masked = tensor
        else:
            tensor_masked = tensor[mask == 1]

        if isinstance(self.min_value, str):
            if self.min_value == "min":
                min_value = torch.min(tensor_masked)
            elif self.min_value.startswith("percentile:"):
                try:
                    percentile = float(self.min_value.split(":")[1])
                    min_value = np.percentile(tensor_masked, percentile)
                except (IndexError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid format for min_value: '{self.min_value}'. Expected 'percentile:<float>'"
                    ) from exc
            else:
                raise TypeError(
                    f"Unsupported string for min_value: '{self.min_value}'."
                    "Must be a float, 'min', or 'percentile:<float>'."
                )
        else:
            min_value = self.min_value

        if isinstance(self.max_value, str):
            if self.max_value == "max":
                max_value = torch.max(tensor_masked)
            elif self.max_value.startswith("percentile:"):
                try:
                    percentile = float(self.max_value.split(":")[1])
                    max_value = np.percentile(tensor_masked, percentile)
                except (IndexError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid format for max_value: '{self.max_value}'. Expected 'percentile:<float>'"
                    ) from exc
            else:
                raise TypeError(
                    f"Unsupported string for max_value: '{self.max_value}'."
                    " Must be a float, 'max', or 'percentile:<float>'."
                )
        else:
            max_value = self.max_value

        # Resolved bounds may be a torch 0-d tensor ("min"/"max") or a numpy scalar
        # ("percentile:<p>"); coerce to a Python float so the in-place assignments below are valid
        # for a torch tensor across numpy/torch versions.
        min_value = float(min_value)
        max_value = float(max_value)

        # Fast path: one fused in-place clamp instead of two float()-copy + where-scatter passes.
        # Restricted to float32 (integer tensors reject float bounds; float16/float64 would compare
        # at a different precision than the legacy float()-cast scatter) and to non-NaN bounds: a
        # NaN bound — from a dynamic min/max/percentile over data containing NaN — makes clamp_
        # propagate NaN to the whole tensor, whereas the legacy scatter no-ops on it (NaN
        # comparisons are False). All other cases keep the exact original behaviour byte-for-byte.
        if tensor.dtype == torch.float32 and min_value == min_value and max_value == max_value:
            tensor.clamp_(min=min_value, max=max_value)
        else:
            tensor[torch.where(tensor.float() < min_value)] = min_value
            tensor[torch.where(tensor.float() > max_value)] = max_value
        if self.save_clip_min:
            cache_attribute["Min"] = min_value
        if self.save_clip_max:
            cache_attribute["Max"] = max_value
        return tensor


class Normalize(TransformInverse):
    """Map intensities to a target min/max interval and optionally invert it."""

    def __init__(
        self,
        lazy: bool = False,
        channels: list[int] | None = None,
        min_value: float = -1,
        max_value: float = 1,
        inverse: bool = True,
    ) -> None:
        super().__init__(inverse)
        if max_value <= min_value:
            raise ValueError(
                f"[Normalize] Invalid range: max_value ({max_value}) must be greater than min_value ({min_value})"
            )
        self.lazy = lazy
        self.min_value = min_value
        self.max_value = max_value
        self.channels = channels

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        if "Min" not in cache_attribute:
            if self.channels:
                cache_attribute["Min"] = torch.min(tensor[self.channels])
            else:
                cache_attribute["Min"] = torch.min(tensor)
        if "Max" not in cache_attribute:
            if self.channels:
                cache_attribute["Max"] = torch.max(tensor[self.channels])
            else:
                cache_attribute["Max"] = torch.max(tensor)
        if not self.lazy:
            input_min = float(cache_attribute["Min"])
            input_max = float(cache_attribute["Max"])
            norm = input_max - input_min

            if norm == 0:
                print(f"[WARNING] Norm is zero for case '{name}': input is constant with value = {self.min_value}.")
                if self.channels:
                    for channel in self.channels:
                        tensor[channel].fill_(self.min_value)
                else:
                    tensor.fill_(self.min_value)
            else:
                if self.channels:
                    for channel in self.channels:
                        tensor[channel] = (self.max_value - self.min_value) * (
                            tensor[channel] - input_min
                        ) / norm + self.min_value
                else:
                    tensor = (self.max_value - self.min_value) * (tensor - input_min) / norm + self.min_value

        return tensor

    def inverse(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        if self.lazy:
            return tensor
        else:
            input_min = float(cache_attribute.pop("Min"))
            input_max = float(cache_attribute.pop("Max"))
            return (tensor - self.min_value) * (input_max - input_min) / (self.max_value - self.min_value) + input_min


class UnNormalize(Transform):
    def __init__(self, min_value: int = -1024, max_value: int = 3071) -> None:
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        return (tensor + 1) / 2 * (self.max_value - self.min_value) + self.min_value


class Standardize(TransformInverse):
    """Standardize tensors using cached or computed mean and standard deviation."""

    def __init__(
        self,
        lazy: bool = False,
        mean: list[float] | None = None,
        std: list[float] | None = None,
        mask: str | None = None,
        inverse: bool = True,
    ) -> None:
        super().__init__(inverse)
        self.lazy = lazy
        self.mean = mean
        self.std = std
        self.mask = mask

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        mask = None
        if self.mask is not None:
            for dataset in self.datasets:
                if dataset.is_dataset_exist(self.mask, name):
                    mask, _ = dataset.read_data(self.mask, name)
                    break
        if mask is None and self.mask is not None:
            raise ValueError(
                f"Requested mask '{self.mask}' is not present in any dataset."
                " Check your dataset group names or configuration."
            )
        if mask is None:
            tensor_masked = tensor
        else:
            tensor_masked = tensor[mask == 1]

        if "Mean" not in cache_attribute:
            cache_attribute["Mean"] = (
                torch.tensor([torch.mean(tensor_masked.type(torch.float32))])
                if self.mean is None
                else torch.tensor(self.mean)
            )

        if "Std" not in cache_attribute:
            cache_attribute["Std"] = (
                torch.tensor([torch.std(tensor_masked.type(torch.float32))])
                if self.std is None
                else torch.tensor(self.std)
            )
        if self.lazy:
            return tensor
        else:
            mean = self._broadcast(cache_attribute.get_tensor("Mean"), tensor)
            std = self._broadcast(cache_attribute.get_tensor("Std"), tensor)
            return (tensor - mean) / std

    @staticmethod
    def _broadcast(stat: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
        """Shape a scalar or per-channel statistic to broadcast over a channel-first tensor."""
        if stat.numel() > 1:
            return stat.reshape(-1, *([1] * (tensor.dim() - 1)))
        return stat

    def inverse(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        if self.lazy:
            return tensor
        else:
            mean = self._broadcast(cache_attribute.pop_tensor("Mean"), tensor)
            std = self._broadcast(cache_attribute.pop_tensor("Std"), tensor)
            return tensor * std + mean


class TensorCast(TransformInverse):
    def __init__(self, dtype: str = "float32", inverse: bool = True) -> None:
        super().__init__(inverse)
        self.dtype: torch.dtype = getattr(torch, dtype)

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        cache_attribute["dtype"] = str(tensor.dtype).replace("torch.", "")
        return tensor.type(self.dtype)

    @staticmethod
    def safe_dtype_cast(dtype_str: str) -> torch.dtype:
        try:
            return getattr(torch, dtype_str)
        except AttributeError as exc:
            raise ValueError(f"Unsupported dtype: {dtype_str}") from exc

    def inverse(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        return tensor.to(TensorCast.safe_dtype_cast(cache_attribute.pop("dtype")))


class Padding(TransformInverse):
    def __init__(self, padding: list[int] = [0, 0, 0, 0, 0, 0], mode: str = "constant", inverse: bool = True) -> None:
        super().__init__(inverse)
        self.padding = padding
        self.mode = mode

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        if "Origin" in cache_attribute and "Spacing" in cache_attribute and "Direction" in cache_attribute:
            origin = torch.tensor(cache_attribute.get_np_array("Origin"))
            matrix = torch.tensor(cache_attribute.get_np_array("Direction").reshape((len(origin), len(origin))))
            origin = torch.matmul(origin, matrix)
            for dim in range(len(self.padding) // 2):
                origin[dim] -= self.padding[dim * 2] * cache_attribute.get_np_array("Spacing")[dim]
            cache_attribute["Origin"] = torch.matmul(origin, torch.inverse(matrix))
        result = F.pad(
            tensor.unsqueeze(0),
            tuple(self.padding),
            self.mode.split(":")[0],
            float(self.mode.split(":")[1]) if len(self.mode.split(":")) == 2 else 0,
        ).squeeze(0)
        return result

    def transform_shape(self, group_src: str, name: str, shape: list[int], cache_attribute: Attribute) -> list[int]:
        for dim in range(len(self.padding) // 2):
            shape[-dim - 1] += sum(self.padding[dim * 2 : dim * 2 + 2])
        return shape

    def inverse(self, name: str, tensor: torch.Tensor, cache_attribute: dict[str, torch.Tensor]) -> torch.Tensor:
        if "Origin" in cache_attribute and "Spacing" in cache_attribute and "Direction" in cache_attribute:
            cache_attribute.pop("Origin")
        slices = [slice(0, shape) for shape in tensor.shape]
        for dim in range(len(self.padding) // 2):
            slices[-dim - 1] = slice(self.padding[dim * 2], tensor.shape[-dim - 1] - self.padding[dim * 2 + 1])
        result = tensor[tuple(slices)]
        return result


class Squeeze(TransformInverse):
    def __init__(self, dim: int, inverse: bool = True) -> None:
        super().__init__(inverse)
        self.dim = dim

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        return tensor.squeeze(self.dim)

    def inverse(self, name: str, tensor: torch.Tensor, cache_attribute: dict[str, Any]) -> torch.Tensor:
        return tensor.unsqueeze(self.dim)


class Resample(TransformInverse, ABC):
    def __init__(self, inverse: bool) -> None:
        super().__init__(inverse)

    def _resample(self, tensor: torch.Tensor, size: list[int]) -> torch.Tensor:
        if tensor.dtype == torch.uint8:
            mode = "nearest"
        elif len(tensor.shape) < 4:
            mode = "bilinear"
        else:
            mode = "trilinear"

        return (
            F.interpolate(tensor.type(torch.float32).unsqueeze(0), size=tuple(size), mode=mode)
            .squeeze(0)
            .type(tensor.dtype)
            .cpu()
        )

    @abstractmethod
    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        pass

    @abstractmethod
    def transform_shape(self, group_src: str, name: str, shape: list[int], cache_attribute: Attribute) -> list[int]:
        pass

    def inverse(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        cache_attribute.pop_np_array("Size")
        size_1 = cache_attribute.pop_np_array("Size")
        if "Spacing" in cache_attribute:
            cache_attribute.pop_np_array("Spacing")
        return self._resample(tensor, [int(size) for size in size_1])


class ResampleToResolution(Resample):
    def __init__(self, spacing: list[float] = [1.0, 1.0, 1.0], inverse: bool = True) -> None:
        super().__init__(inverse)
        self.spacing = torch.tensor([0 if s < 0 else s for s in spacing])

    def transform_shape(self, group_src: str, name: str, shape: list[int], cache_attribute: Attribute) -> list[int]:
        if "Spacing" not in cache_attribute:
            raise TransformError(
                "Missing 'Spacing' in cache attributes, the data is likely not a valid image.",
                "Make sure your input is a image (e.g., .nii, .mha) with proper metadata.",
            )
        if len(shape) != len(self.spacing):
            raise TransformError(f"Shape and spacing dimensions do not match: shape={shape}, spacing={self.spacing}")
        image_spacing = cache_attribute.get_tensor("Spacing")
        resize_factor = torch.tensor(
            [s / i_s if s > 0 else 1.0 for s, i_s in zip(self.spacing, image_spacing, strict=False)]
        )
        return [int(x) for x in (torch.tensor(shape) * 1 / resize_factor.flip(0))]

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        image_spacing = cache_attribute.get_tensor("Spacing")
        spacing = self.spacing
        resize_factor = torch.tensor(
            [
                s / i_s if s > 0 else 1.0
                for s, i_s in zip(self.spacing, cache_attribute.get_tensor("Spacing"), strict=False)
            ]
        )
        cache_attribute["Spacing"] = torch.tensor(
            [float(s) if s > 0 else float(i_s) for s, i_s in zip(spacing, image_spacing, strict=False)]
        )
        cache_attribute["Size"] = np.asarray([int(x) for x in torch.tensor(tensor.shape[1:])])
        size = [int(x) for x in (torch.tensor(tensor.shape[1:]) * 1 / resize_factor.flip(0))]
        cache_attribute["Size"] = np.asarray(size)
        return self._resample(tensor, size)


class ResampleToShape(Resample):
    def __init__(self, shape: list[float] = [100, 256, 256], inverse: bool = True) -> None:
        super().__init__(inverse)
        self.shape = torch.tensor([0 if s < 0 else s for s in shape])

    def transform_shape(self, group_src: str, name: str, shape: list[int], cache_attribute: Attribute) -> list[int]:
        if "Spacing" not in cache_attribute:
            raise TransformError(
                "Missing 'Spacing' in cache attributes, the data is likely not a valid image.",
                "Make sure your input is a image (e.g., .nii, .mha) with proper metadata.",
            )
        if len(shape) != len(self.shape):
            raise TransformError(f"Shape and target dimensions do not match: shape={shape}, target_shape={self.shape}")
        new_shape = self.shape.clone()
        for i, s in enumerate(self.shape):
            if s == 0:
                new_shape[i] = shape[i]
        return new_shape

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        shape = self.shape.clone()
        image_shape = torch.tensor([int(x) for x in torch.tensor(tensor.shape[1:])])
        for i, s in enumerate(self.shape):
            if s == 0:
                shape[i] = image_shape[i]
        if "Spacing" in cache_attribute:
            cache_attribute["Spacing"] = torch.flip(
                image_shape / shape * torch.flip(cache_attribute.get_tensor("Spacing"), dims=[0]),
                dims=[0],
            )
        cache_attribute["Size"] = image_shape
        cache_attribute["Size"] = shape
        return self._resample(tensor, shape)


class ResampleTransform(TransformInverse):
    def __init__(self, transforms: dict[str, bool], inverse: bool = True) -> None:
        super().__init__(inverse)
        self.transforms = transforms

    def transform_shape(self, group_src: str, name: str, shape: list[int], cache_attribute: Attribute) -> list[int]:
        return shape

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        if len(tensor.shape) != 4:
            raise NameError("Input size should be 5 dim")
        image = data_to_image(tensor, cache_attribute)

        vectors = [torch.arange(0, s) for s in tensor.shape[1:]]
        grids = torch.meshgrid(vectors, indexing="ij")
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)

        transforms = []
        for transform_group, invert in self.transforms.items():
            transform = None
            for dataset in self.datasets:
                if dataset.is_dataset_exist(transform_group, name):
                    transform = dataset.read_transform(transform_group, name)
                    break
            if transform is None:
                raise NameError(f"Tranform : {transform_group}/{name} not found")
            if isinstance(transform, sitk.BSplineTransform):
                if invert:
                    transform_to_displacement_field_filter = sitk.TransformToDisplacementFieldFilter()
                    transform_to_displacement_field_filter.SetReferenceImage(image)
                    displacement_field = transform_to_displacement_field_filter.Execute(transform)
                    iterative_inverse_displacement_field_image_filter = (
                        sitk.IterativeInverseDisplacementFieldImageFilter()
                    )
                    iterative_inverse_displacement_field_image_filter.SetNumberOfIterations(20)
                    inverse_displacement_field = iterative_inverse_displacement_field_image_filter.Execute(
                        displacement_field
                    )
                    transform = sitk.DisplacementFieldTransform(inverse_displacement_field)
            else:
                if invert:
                    transform = transform.GetInverse()
            transforms.append(transform)
        result_transform = sitk.CompositeTransform(transforms)

        transform_to_displacement_field_filter = sitk.TransformToDisplacementFieldFilter()
        transform_to_displacement_field_filter.SetReferenceImage(image)
        transform_to_displacement_field_filter.SetNumberOfThreads(16)
        new_locs = grid + torch.tensor(
            sitk.GetArrayFromImage(transform_to_displacement_field_filter.Execute(result_transform))
        ).unsqueeze(0).permute(0, 4, 1, 2, 3)
        shape = new_locs.shape[2:]
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)
        new_locs = new_locs.permute(0, 2, 3, 4, 1)
        new_locs = new_locs[..., [2, 1, 0]]
        result = (
            F.grid_sample(
                tensor.to(self.device).unsqueeze(0).float(),
                new_locs.to(self.device).float(),
                align_corners=True,
                padding_mode="border",
                mode="nearest" if tensor.dtype == torch.uint8 else "bilinear",
            )
            .squeeze(0)
            .cpu()
        )
        return result.type(torch.uint8) if tensor.dtype == torch.uint8 else result

    def inverse(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        # TODO
        return tensor


class Mask(Transform):
    def __init__(self, path: str = "./default.mha", value_outside: int = 0) -> None:
        super().__init__()
        self.path = path
        self.value_outside = value_outside
        self._cached_mask: torch.Tensor | None = None

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        if self.path.endswith(".mha"):
            if self._cached_mask is None:
                self._cached_mask = torch.tensor(sitk.GetArrayFromImage(sitk.ReadImage(self.path))).unsqueeze(0)
            mask = self._cached_mask
        else:
            mask = None
            for dataset in self.datasets:
                if dataset.is_dataset_exist(self.path, name):
                    mask, _ = dataset.read_data(self.path, name)
                    break
            if mask is None:
                raise NameError(f"Mask : {self.path}/{name} not found")
        tensor[torch.tensor(mask) == 0] = self.value_outside
        return tensor


class Dilate(Transform):
    def __init__(self, dilate: int = 1) -> None:
        super().__init__()
        if dilate < 0:
            raise ValueError(f"[Dilate] 'dilate' must be >= 0, got {dilate}")
        self.dilate = dilate

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        if self.dilate == 0:
            return tensor

        data = (tensor > 0).to(torch.float32)
        spatial_dims = data.dim() - 1
        d = self.dilate
        k = 2 * d + 1

        # A cubic (box) structuring element is separable: dilating by a k**n box equals n successive
        # 1-D max-pools, one per spatial axis. This is bit-identical to a single k**n max-pool (max is
        # associative and the box is the Minkowski sum of 1-D segments) for ~k**(n-1)x fewer comparisons
        # — the k**3 dense pool is the dominant cost of the whole-volume mask load.
        if spatial_dims == 2:
            data = F.max_pool2d(data, kernel_size=(k, 1), stride=1, padding=(d, 0))
            data = F.max_pool2d(data, kernel_size=(1, k), stride=1, padding=(0, d))
        elif spatial_dims == 3:
            data = F.max_pool3d(data, kernel_size=(k, 1, 1), stride=1, padding=(d, 0, 0))
            data = F.max_pool3d(data, kernel_size=(1, k, 1), stride=1, padding=(0, d, 0))
            data = F.max_pool3d(data, kernel_size=(1, 1, k), stride=1, padding=(0, 0, d))
        else:
            raise ValueError(
                "[Dilate] Unsupported tensor shape for "
                f"'{name}': expected [C,H,W] or [C,D,H,W], got {list(tensor.shape)}"
            )

        return data.to(tensor.dtype)


class Sum(Transform):
    def __init__(self, dim: int = 0) -> None:
        super().__init__()
        self.dim = dim

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        if "number_of_channels_per_model" in cache_attribute:
            number_of_channels = cache_attribute.pop_tensor("number_of_channels_per_model")
            result = tensor[0]
            for i, t in enumerate(tensor[1:]):
                t[t != 0] += int(number_of_channels[i]) - 1
                result += t
            return result
        else:
            return torch.sum(tensor, dim=self.dim).to(tensor.dtype)


class MergeLabels(Transform):
    """Merge the per-model argmax label maps of a ``combine: Concat`` ensemble into one global map.

    Each model's ``Argmax`` produces a LOCAL class index (``0`` = background). A model's
    non-background labels are shifted past every earlier model's foreground classes -- by the
    CUMULATIVE sum of the earlier models' foreground counts (``nb_class - 1``) -- so the models'
    disjoint label ranges tile a single global label space.

    This is the label-space counterpart of ``InferenceStack`` (which averages *same-class*
    probability ensembles): use ``MergeLabels`` when the models segment DIFFERENT structures, e.g.
    the 5-task TotalSegmentator ensemble (organs / vertebrae / cardiac / muscles / ribs). Requires
    ``number_of_channels_per_model`` in the attribute (written by the ``Concat`` reduction).
    """

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        if "number_of_channels_per_model" not in cache_attribute:
            raise TransformError(
                "MergeLabels expects a multi-model 'combine: Concat' output: "
                "'number_of_channels_per_model' is missing from the attribute.",
            )
        number_of_channels = cache_attribute.pop_tensor("number_of_channels_per_model")
        result = tensor[0]
        offset = int(number_of_channels[0]) - 1
        for i, t in enumerate(tensor[1:]):
            t[t != 0] += offset
            result += t
            offset += int(number_of_channels[i + 1]) - 1
        return result


class Gradient(Transform):
    def __init__(self, per_dim: bool = False):
        super().__init__()
        self.per_dim = per_dim

    @staticmethod
    def _image_gradient_2d(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dx = image[:, 1:, :] - image[:, :-1, :]
        dy = image[:, :, 1:] - image[:, :, :-1]
        return torch.nn.ConstantPad2d((0, 0, 0, 1), 0)(dx), torch.nn.ConstantPad2d((0, 1, 0, 0), 0)(dy)

    @staticmethod
    def _image_gradient_3d(
        image: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dx = image[:, 1:, :, :] - image[:, :-1, :, :]
        dy = image[:, :, 1:, :] - image[:, :, :-1, :]
        dz = image[:, :, :, 1:] - image[:, :, :, :-1]
        return (
            torch.nn.ConstantPad3d((0, 0, 0, 0, 0, 1), 0)(dx),
            torch.nn.ConstantPad3d((0, 0, 0, 1, 0, 0), 0)(dy),
            torch.nn.ConstantPad3d((0, 1, 0, 0, 0, 0), 0)(dz),
        )

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        result = torch.stack(
            (Gradient._image_gradient_3d(tensor) if len(tensor.shape) == 4 else Gradient._image_gradient_2d(tensor)),
            dim=1,
        ).squeeze(0)
        if not self.per_dim:
            result = torch.sigmoid(result * 3)
            result = result.norm(dim=0)
            result = torch.unsqueeze(result, 0)

        return result


class Argmax(Transform):
    def __init__(self, dim: int = 0) -> None:
        super().__init__()
        self.dim = dim

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        return torch.argmax(tensor, dim=self.dim).unsqueeze(self.dim)


class Softmax(Transform):
    def __init__(self, dim: int = 0) -> None:
        super().__init__()
        self.dim = dim

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        return torch.softmax(tensor, dim=self.dim)


class FlatLabel(Transform):
    def __init__(self, labels: list[int] | None = None) -> None:
        super().__init__()
        self.labels = labels

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        data = torch.zeros_like(tensor)
        if self.labels:
            for label in self.labels:
                data[torch.where(tensor == label)] = 1
        else:
            data[torch.where(tensor > 0)] = 1
        return data


class Save(Transform):
    def __init__(self, dataset: str, group: str | None = None) -> None:
        super().__init__()
        self.dataset = dataset
        self.group = group

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        return tensor


class Flatten(Transform):
    def __init__(self) -> None:
        super().__init__()

    def transform_shape(self, group_src: str, name: str, shape: list[int], cache_attribute: Attribute) -> list[int]:
        return [np.prod(np.asarray(shape))]

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        return tensor.flatten()


class Permute(TransformInverse):
    def __init__(self, dims: str = "1|0|2", inverse: bool = True) -> None:
        super().__init__(inverse)
        self.dims = [0] + [int(d) + 1 for d in dims.split("|")]

    def transform_shape(self, group_src: str, name: str, shape: list[int], cache_attribute: Attribute) -> list[int]:
        return [shape[it - 1] for it in self.dims[1:]]

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        return tensor.permute(tuple(self.dims))

    def inverse(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        return tensor.permute(tuple(np.argsort(self.dims)))


class Flip(TransformInverse):
    def __init__(self, dims: str = "1|0|2", inverse: bool = True) -> None:
        super().__init__(inverse)

        self.dims = [int(d) + 1 for d in str(dims).split("|")]

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        return tensor.flip(tuple(self.dims))

    def inverse(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        return tensor.flip(tuple(self.dims))


class Canonical(TransformInverse):
    def __init__(self, inverse: bool = True) -> None:
        super().__init__(inverse)
        self.canonical_direction = torch.diag(torch.tensor([-1, -1, 1])).to(torch.double)

    @staticmethod
    def _affine_matrix(matrix: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            (
                torch.cat((matrix, translation.unsqueeze(0).T), dim=1),
                torch.tensor([[0, 0, 0, 1]]),
            ),
            dim=0,
        )

    @staticmethod
    def _resample_affine(data: torch.Tensor, matrix: torch.Tensor):
        if data.dtype == torch.uint8:
            mode = "nearest"
        else:
            mode = "bilinear"
        return (
            torch.nn.functional.grid_sample(
                data.unsqueeze(0).type(torch.float32),
                torch.nn.functional.affine_grid(
                    matrix[:, :-1, ...].type(torch.float32),
                    [1, *list(data.shape)],
                    align_corners=True,
                ),
                align_corners=True,
                mode=mode,
                padding_mode="reflection",
            )
            .squeeze(0)
            .type(data.dtype)
        )

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        spacing = cache_attribute.get_tensor("Spacing")
        initial_matrix = cache_attribute.get_tensor("Direction").reshape(3, 3).to(torch.double)
        initial_origin = cache_attribute.get_tensor("Origin")
        cache_attribute["Direction"] = (self.canonical_direction).flatten()
        matrix = Canonical._affine_matrix(self.canonical_direction @ initial_matrix.inverse(), torch.tensor([0, 0, 0]))
        center_voxel = torch.tensor(
            [(tensor.shape[-i - 1] - 1) * spacing[i] / 2 for i in range(3)],
            dtype=torch.double,
        )
        center_physical = initial_matrix @ center_voxel + initial_origin
        cache_attribute["Origin"] = center_physical - (self.canonical_direction @ center_voxel)
        return Canonical._resample_affine(tensor, matrix.unsqueeze(0))

    def inverse(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        cache_attribute.pop("Direction")
        cache_attribute.pop("Origin")
        matrix = Canonical._affine_matrix(
            (
                self.canonical_direction
                @ cache_attribute.get_tensor("Direction").to(torch.double).reshape(3, 3).inverse()
            ).inverse(),
            torch.tensor([0, 0, 0]),
        )
        return Canonical._resample_affine(tensor, matrix.unsqueeze(0))


class HistogramMatching(Transform):
    def __init__(self, reference_group: str) -> None:
        super().__init__()
        self.reference_group = reference_group

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        image = data_to_image(tensor, cache_attribute)
        image_ref = None
        for dataset in self.datasets:
            if dataset.is_dataset_exist(self.reference_group, name):
                image_ref = dataset.read_image(self.reference_group, name)
        if image_ref is None:
            raise NameError(f"Image : {self.reference_group}/{name} not found")
        matcher = sitk.HistogramMatchingImageFilter()
        matcher.SetNumberOfHistogramLevels(256)
        matcher.SetNumberOfMatchPoints(1)
        matcher.SetThresholdAtMeanIntensity(True)
        result, _ = image_to_data(matcher.Execute(image, image_ref))
        return torch.tensor(result)


class SelectLabel(Transform):
    def __init__(self, labels: list[str]) -> None:
        super().__init__()
        self.labels = [label[1:-1].split(",") for label in labels]

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        data = torch.zeros_like(tensor)
        for old_label, new_label in self.labels:
            data[tensor == int(old_label)] = int(new_label)
        return data


class OneHot(TransformInverse):
    def __init__(self, num_classes: int, inverse: bool = True) -> None:
        super().__init__(inverse)
        self.num_classes = num_classes

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        result = (
            F.one_hot(tensor.type(torch.int64), num_classes=self.num_classes)
            .permute(0, len(tensor.shape), *[i + 1 for i in range(len(tensor.shape) - 1)])
            .float()
            .squeeze(0)
        )
        return result

    def inverse(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        return torch.argmax(tensor, dim=1).unsqueeze(1)


# Published app used by KonfAIInference when the configuration leaves repo/model unset.
DEFAULT_INFERENCE_REPO_ID = "VBoussot/MRSegmentator-KonfAI"
DEFAULT_INFERENCE_MODEL_NAME = "MRSegmentator"


class KonfAIInference(Transform):
    supports_dataloader_workers = False

    def __init__(
        self,
        repo_id: str = DEFAULT_INFERENCE_REPO_ID,
        model_name: str = DEFAULT_INFERENCE_MODEL_NAME,
        checkpoints_name: list[str] = ["fold_0"],
        number_of_tta: int = 0,
        number_of_mc: int = 0,
        per_channel: bool = False,
    ):
        super().__init__()
        self.repo_id = repo_id
        self.model_name = model_name
        self.checkpoints_name = checkpoints_name
        self.number_of_tta = number_of_tta
        self.number_of_mc = number_of_mc
        self.per_channel = per_channel

    def infer_entry(self, dataset_path: Path, output_path: Path, gpu: list[int]):
        try:
            from konfai_apps import KonfAIApp
        except ImportError as exc:  # pragma: no cover - depends on optional install
            raise RuntimeError(
                "KonfAIInference requires the standalone 'konfai-apps' package. "
                "Install it from the repository with 'pip install -e ./konfai-apps'."
            ) from exc

        # Nested KonfAI runs must choose their own rendezvous ports instead of
        # inheriting the parent's already-bound distributed settings.
        os.environ.pop("KONFAI_MASTER_PORT", None)
        os.environ.pop("KONFAI_TENSORBOARD_PORT", None)

        konfai_app = KonfAIApp(f"{self.repo_id}:{self.model_name}", False, False)
        konfai_app.infer(
            [[dataset_path]],
            output_path,
            0,
            self.checkpoints_name,
            self.number_of_tta,
            mc=0,
            uncertainty=False,
            gpu=gpu,
        )

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        if current_process().daemon:
            raise RuntimeError(
                "KonfAIInference cannot run inside daemon DataLoader workers. "
                "Use 'Dataset.num_workers: 0' for pipelines that include this transform."
            )
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "Dataset"
            if self.per_channel:
                for i, channel in enumerate(tensor):
                    image = data_to_image(channel.unsqueeze(0).numpy(), cache_attribute)
                    (dataset_path / f"P{i:03d}").mkdir(parents=True, exist_ok=True)
                    sitk.WriteImage(image, str(dataset_path / f"P{i:03d}" / "Volume.mha"))
            else:
                image = data_to_image(tensor.numpy(), cache_attribute)

                (dataset_path / "P000").mkdir(parents=True, exist_ok=True)
                sitk.WriteImage(image, str(dataset_path / "P000" / "Volume.mha"))

            ctx = get_context("spawn")

            p = ctx.Process(
                target=self.infer_entry, args=(dataset_path, Path(tmpdir) / "Output", cuda_visible_devices())
            )
            p.start()
            p.join()

            if p.exitcode != 0:
                raise RuntimeError("Inference process failed")

            return self._reassemble_output(Path(tmpdir) / "Output")

    @staticmethod
    def _reassemble_output(output_dir: Path) -> torch.Tensor:
        result = []
        for file in sorted(output_dir.rglob("*.mha")):
            if file.name != "InferenceStack.mha":
                result.append(torch.from_numpy(image_to_data(sitk.ReadImage(str(file)))[0]))
        return torch.stack(result, dim=1).squeeze(0)


class InferenceStack(Transform):
    def __init__(self, dataset: str, name: str, mode: str = "mean"):
        super().__init__()
        self.dataset = None
        if dataset:
            filename, _, file_format = split_path_spec(dataset)
            self.dataset = Dataset(filename, file_format)
        self.name = name
        self.mode = mode

    def __call__(self, name: str, tensors: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        if tensors.shape[0] == 1:
            return tensors.squeeze(0)
        if self.mode == "Seg":
            _tensors = torch.argmax(torch.softmax(tensors, dim=1), dim=1).to(torch.uint8)
        else:
            _tensors = tensors.squeeze(1)
        dataset = self.dataset if self.dataset else self.datasets[-1]
        dataset.write("InferenceStack", name, _tensors.float().cpu().numpy(), cache_attribute)
        return (
            torch.median(tensors.float(), dim=0).values.to(tensors.dtype)
            if self.mode == "median"
            else tensors.float().mean(0).to(tensors.dtype)
        )


class Norm(Transform):
    """Vector magnitude over the trailing component axis.

    Reduces a stacked vector field (e.g. a displacement-field ensemble ``[N, (D), H, W, C]``) to
    per-sample magnitudes ``[N, (D), H, W]``, typically before ``Variance``/``StandardDeviation``.
    The trailing tensor axis is the first geometry axis (numpy order is reversed), so that axis is
    dropped from ``Origin``/``Spacing``/``Direction``.
    """

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, name: str, tensors: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        if "Origin" in cache_attribute:
            origin = cache_attribute.pop_np_array("Origin")
            spacing = cache_attribute.pop_np_array("Spacing")
            direction = cache_attribute.pop_np_array("Direction")
            rank = len(origin)
            cache_attribute["Origin"] = origin[1:]
            cache_attribute["Spacing"] = spacing[1:]
            cache_attribute["Direction"] = direction.reshape(rank, rank)[1:, 1:].flatten()
        return torch.linalg.norm(tensors.float(), dim=-1)

    def transform_shape(self, group_src: str, name: str, shape: list[int], cache_attribute: Attribute) -> list[int]:
        return shape[:-1]


class Variance(Transform):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, name: str, tensors: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        return tensors.float().var(0).unsqueeze(0) if tensors.shape[0] > 1 else torch.zeros_like(tensors[0])


class SegmentationDisagreement(Transform):
    def __init__(self, ignore_background: bool = False) -> None:
        super().__init__()
        self.ignore_background = ignore_background

    def __call__(self, name: str, tensors: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        # tensors shape: [N, ...] with N segmentations and integer labels per voxel
        if tensors.shape[0] <= 1:
            return torch.zeros_like(tensors[0], dtype=torch.float32).unsqueeze(0)

        tensors = tensors.long()

        if self.ignore_background:
            valid = tensors != 0
        else:
            valid = torch.ones_like(tensors, dtype=torch.bool)

        disagreement = torch.zeros_like(tensors[0], dtype=torch.float32)

        # per-voxel disagreement = 1 - (frequency of majority label / number of valid segmentations)
        unique_labels = torch.unique(tensors)
        counts = []
        for label in unique_labels:
            counts.append(((tensors == label) & valid).sum(dim=0))

        counts = torch.stack(counts, dim=0)  # [L, ...]
        max_count = counts.max(dim=0).values
        valid_count = valid.sum(dim=0)

        non_empty = valid_count > 0
        disagreement[non_empty] = 1.0 - (max_count[non_empty].float() / valid_count[non_empty].float())

        return disagreement.unsqueeze(0)


class Percentage(Transform):
    def __init__(self, baseline: float) -> None:
        super().__init__()
        self.baseline = baseline

    def __call__(self, name: str, tensors: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        return tensors / self.baseline * 100.0


class StandardDeviation(Transform):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, name: str, tensors: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        return tensors.float().std(0).unsqueeze(0) if tensors.shape[0] > 1 else torch.zeros_like(tensors[0])


class Statistics(Transform):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, name: str, tensors: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        cache_attribute["ImageMin"] = tensors.float().min()
        cache_attribute["ImageMax"] = tensors.float().max()
        cache_attribute["ImageMean"] = tensors.float().mean()
        cache_attribute["ImageStd"] = tensors.float().std()
        return tensors


class Crop(TransformInverse):
    def __init__(self, inverse: bool = True) -> None:
        super().__init__(inverse)

    def transform_shape(self, group_src: str, name: str, shape: list[int], cache_attribute: Attribute) -> list[int]:
        # The crop box is content-dependent (foreground bounding box), so the output shape
        # cannot be known without the pixel data. If the box was already computed and persisted
        # as a sidecar attribute, reuse it and skip the read; otherwise compute it once from the
        # volume. (A fully-lazy variant would require deferring patch planning past _load().)
        # ``shape`` is already the channel-stripped spatial shape (patching strips [C, *spatial]
        # before calling transform_shape), so the crop box — one row per spatial axis — aligns with
        # ``shape`` directly, exactly like ``__call__`` aligns it with ``tensor.shape[1:]``.
        if "box" in cache_attribute:
            box = self._parse_box(cache_attribute["box"])
            return [int(s - a - b) for (a, b), s in zip(box, shape, strict=False)]
        data = None
        for dataset in self.datasets:
            if dataset.is_dataset_exist(group_src, name):
                data, _ = dataset.read_data(group_src, name)
                break
        if data is None:
            return shape
        treshold = np.percentile(data, 5)
        image = data_to_image((data > treshold).astype(np.uint8), cache_attribute)
        box = box_with_mask(image, [1], [0] * (len(data.shape) - 1))
        for i, ((_, b), s) in enumerate(zip(box, shape, strict=False)):
            box[i][1] = s - b
        cache_attribute["box"] = box
        return [int(s - a - b) for (a, b), s in zip(box, shape, strict=False)]

    @staticmethod
    def _parse_box(box_str: str) -> np.ndarray:
        flat = np.fromstring(box_str.replace("[", " ").replace("]", " "), sep=" ", dtype=np.int64)
        return flat.reshape(-1, 2)

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        if "box" not in cache_attribute:
            return tensor
        box = self._parse_box(cache_attribute["box"])
        for i, ((_, b), s) in enumerate(zip(box, tensor.shape[1:], strict=False)):
            box[i][1] = s - b
        if "Origin" in cache_attribute and "Spacing" in cache_attribute and "Direction" in cache_attribute:
            origin = torch.tensor(cache_attribute.get_np_array("Origin"))
            matrix = torch.tensor(cache_attribute.get_np_array("Direction").reshape((len(origin), len(origin))))
            origin = torch.matmul(origin, matrix)
            for dim in range(box.shape[0]):
                origin[-dim - 1] += box[dim][0] * cache_attribute.get_np_array("Spacing")[-dim - 1]
            cache_attribute["Origin"] = torch.matmul(origin, torch.inverse(matrix))

        image = data_to_image(tensor.numpy(), cache_attribute)
        result = crop_with_mask(image, box)
        data, _ = image_to_data(result)
        return torch.from_numpy(data)

    def inverse(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        if "box" not in cache_attribute:
            return tensor
        box = self._parse_box(cache_attribute.pop("box"))
        cache_attribute.pop_np_array("Origin")
        padding = []
        for b in reversed(box):
            padding.extend([b[0], b[1]])
        result = F.pad(tensor.unsqueeze(0), tuple(padding), "replicate").squeeze(0)
        return result
