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

"""Patch extraction, accumulation, and patch-combination helpers for KonfAI."""

import copy
import itertools
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from functools import partial

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F

from konfai.data.augmentation import DataAugmentationsList
from konfai.data.transform import Clip, Normalize, Save, Standardize, TensorCast, Transform
from konfai.utils.config import apply_config, config
from konfai.utils.dataset import Attribute, Dataset
from konfai.utils.errors import PatchError
from konfai.utils.utils import SUPPORTED_EXTENSIONS, get_module, get_patch_slices_from_shape, split_path_spec


@dataclass(frozen=True)
class PatchReadPlan:
    """Precomputed slicing and padding instructions for one patch request."""

    data_slices: tuple[slice, ...]
    reflect_padding: tuple[int, ...]
    constant_padding: tuple[int, ...]
    concatenate_extend_slice: bool


class PathCombine(ABC):
    """Base class for overlap-aware weighting schemes applied during patch assembly."""

    def __init__(self) -> None:
        self.data: torch.Tensor
        self.overlap: int
        self._data_per_device: dict[torch.device, torch.Tensor] = {}

    """
    A = slice(0, overlap)
    B = slice(-overlap, None)
    C = slice(overlap, -overlap)

    1D
        A+B
    2D :
        AA+AB+BA+BB

        AC+BC
        CA+CB
    3D :
        AAA+AAB+ABA+ABB+BAA+BAB+BBA+BBB

        CAA+CAB+CBA+CBB
        ACA+ACB+BCA+BCB
        AAC+ABC+BAC+BBC

        CCA+CCB
        CAC+CBC
        ACC+BCC

    """

    def set_patch_config(self, patch_size: list[int], overlap: int):
        self._data_per_device.clear()
        if overlap <= 0:
            self.data = torch.ones(patch_size)
            return
        self.data = F.pad(
            torch.ones([size - overlap * 2 for size in patch_size]),
            [overlap] * 2 * len(patch_size),
            mode="constant",
            value=0,
        )
        self.data = self._set_function(self.data, overlap)
        dim = len(patch_size)

        a = slice(0, overlap)
        b = slice(-overlap, None)
        c = slice(overlap, -overlap)

        for i in range(dim):
            slices_badge = list(itertools.product(*[[a, b] for _ in range(dim - i)]))
            for indexs in itertools.combinations([0, 1, 2], i):
                result = []
                for slices_tuple in slices_badge:
                    slices_list = list(slices_tuple)
                    for index in indexs:
                        slices_list.insert(index, c)
                    result.append(tuple(slices_list))
                for patch, s in zip(PathCombine._normalise([self.data[s] for s in result]), result, strict=False):
                    self.data[s] = patch

    @staticmethod
    def _normalise(patchs: list[torch.Tensor]) -> list[torch.Tensor]:
        data_sum = torch.sum(torch.concat([patch.unsqueeze(0) for patch in patchs], dim=0), dim=0)
        return [d / data_sum for d in patchs]

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.device not in self._data_per_device:
            self._data_per_device[tensor.device] = self.data.to(tensor.device)
        return self._data_per_device[tensor.device] * tensor

    @abstractmethod
    def _set_function(self, data: torch.Tensor, overlap: int) -> torch.Tensor:
        pass


class Mean(PathCombine):
    """Uniform patch-combination strategy for overlapping predictions."""

    def __init__(self) -> None:
        super().__init__()

    def _set_function(self, data: torch.Tensor, overlap: int) -> torch.Tensor:
        return torch.ones_like(self.data)


class Cosinus(PathCombine):
    """Cosine-based weighting strategy for smoother overlap blending."""

    def __init__(self) -> None:
        super().__init__()

    def _function_sides(self, overlap: int, x: float):
        return np.clip(np.cos(np.pi / (2 * (overlap + 1)) * x), 0, 1)

    def _set_function(self, data: torch.Tensor, overlap: int) -> torch.Tensor:
        image = sitk.GetImageFromArray(np.asarray(data, dtype=np.uint8))
        danielsson_distance_map_image_filter = sitk.DanielssonDistanceMapImageFilter()
        distance = torch.tensor(sitk.GetArrayFromImage(danielsson_distance_map_image_filter.Execute(image)))
        return distance.apply_(partial(self._function_sides, overlap))


class Accumulator:
    """Accumulate patch predictions and reassemble them into a full tensor."""

    def __init__(
        self,
        patch_slices: list[tuple[slice, ...]],
        patch_size: list[int],
        patch_combine: PathCombine | None = None,
        batch: bool = True,
    ) -> None:
        self._layer_accumulator: list[torch.Tensor | None] = [None] * len(patch_slices)
        self.patch_slices: list[tuple[slice, ...]] = []

        if patch_size is not None and not all(p == 0 for p in patch_size):
            for patch in patch_slices:
                slices: list[slice] = []
                for s, shape in zip(patch, patch_size, strict=False):
                    slices.append(slice(s.start, s.start + shape))
                self.patch_slices.append(tuple(slices))
        else:
            self.patch_slices = patch_slices

        self.shape = max([[v.stop for v in patch] for patch in patch_slices])
        self.patch_size = patch_size
        self.patch_combine = patch_combine
        self.batch = batch
        self._filled = 0

    def add_layer(self, index: int, layer: torch.Tensor) -> None:
        if self._layer_accumulator[index] is None:
            self._filled += 1
        self._layer_accumulator[index] = layer

    def is_full(self) -> bool:
        # O(1): a running counter avoids re-scanning every slot after each added patch
        # (the completion check ran once per patch, i.e. O(P^2) per case).
        return self._filled == len(self.patch_slices)

    def assemble(self) -> torch.Tensor:
        n = 2 if self.batch else 1
        reference = next((layer for layer in self._layer_accumulator if layer is not None), None)
        if reference is None:
            raise PatchError(
                "Accumulator.assemble() was called before any patch was added.",
                f"Expected up to {len(self.patch_slices)} patch(es) via add_layer() before assembling.",
                "Add at least one patch (and check is_full()) before calling assemble().",
            )
        result = torch.zeros(
            (list(reference.shape[:n]) + list(max([[v.stop for v in patch] for patch in self.patch_slices]))),
            dtype=reference.dtype,
            device=reference.device,
        )
        # Overlap blending weights each patch (edge bands < 1 so interior overlaps sum to unity).
        # A voxel covered by fewer patches (a volume border without whole-image padding) would sum
        # to < 1 and come out darkened (x0.5 edges, x0.25 corners), so divide by the accumulated
        # weight. The weight map is identical for every patch and constant across batch/channels, so
        # it is computed once and accumulated spatially only; with padding weight_sum is ~1 and the
        # division is a near no-op.
        combine = self.patch_combine
        weight_sum = torch.zeros(result.shape[n:], device=result.device) if combine is not None else torch.empty(0)
        weight_patch: torch.Tensor | None = None
        for patch_slice, data in zip(self.patch_slices, self._layer_accumulator, strict=False):
            if data is not None:
                slices_dest = tuple([slice(result.shape[i]) for i in range(n)] + list(patch_slice))

                for dim, s in enumerate(patch_slice):
                    if s.stop - s.start == 1:
                        data = data.unsqueeze(dim=dim + n)
                if combine is not None:
                    result[slices_dest] += combine(data)
                    if weight_patch is None:
                        weight_patch = combine(torch.ones_like(data))[(0,) * n]
                    weight_sum[patch_slice] += weight_patch
                else:
                    result[slices_dest] = data
        if combine is not None:
            result = result / weight_sum.clamp(min=1e-8)
        result = result[tuple([slice(None, None)] + [slice(0, s) for s in self.shape])]

        self._layer_accumulator.clear()
        self._filled = 0
        return result


class Patch(ABC):
    """Abstract base class for dataset-level and model-level patch definitions."""

    @abstractmethod
    def __init__(
        self,
        patch_size: list[int],
        overlap: int | None,
        pad_value: float | None = 0,
        extend_slice: int = 0,
    ) -> None:
        if extend_slice != 0 and patch_size is not None and patch_size[0] != 1:
            raise ValueError(
                "`extend_slice` can only be used when patch_size[0] == 1 "
                f"(got patch_size[0]={patch_size[0]}, extend_slice={extend_slice})"
            )
        self.patch_size = patch_size
        self.overlap = overlap
        if isinstance(self.overlap, int):
            if self.overlap < 0:
                self.overlap = None
        self._patch_slices: dict[int, list[tuple[slice, ...]]] = {}
        self._nb_patch_per_dim: dict[int, list[tuple[int, bool]]] = {}
        self.pad_value = pad_value
        self.extend_slice = extend_slice

    def load(self, shape: list[int], a: int = 0) -> None:
        self._patch_slices[a], self._nb_patch_per_dim[a] = get_patch_slices_from_shape(
            self.patch_size, shape, self.overlap
        )

    @abstractmethod
    def init(self, key: str):
        pass

    def get_patch_slices(self, a: int = 0):
        return self._patch_slices[a]

    def get_read_plan(
        self, data_shape: list[int] | tuple[int, ...], index: int, a: int, is_input: bool
    ) -> PatchReadPlan:
        slices_pre = [slice(None) for _ in data_shape[: -len(self._patch_slices[a][0])]]
        extend_slice = self.extend_slice if is_input else 0

        bottom = extend_slice // 2
        top = int(np.ceil(extend_slice / 2))
        s = slice(
            (
                self._patch_slices[a][index][0].start - bottom
                if self._patch_slices[a][index][0].start - bottom >= 0
                else 0
            ),
            (
                self._patch_slices[a][index][0].stop + top
                if self._patch_slices[a][index][0].stop + top <= data_shape[len(slices_pre)]
                else data_shape[len(slices_pre)]
            ),
        )
        slices = [s, *list(self._patch_slices[a][index][1:])]
        reflect_padding = [0 for _ in range((len(slices) - 1) * 2)] + [0, 0]
        if extend_slice > 0 and (s.stop - s.start) < bottom + top + 1:
            if self._patch_slices[a][index][0].start - bottom < 0:
                reflect_padding[-2] = bottom - self._patch_slices[a][index][0].start
            if self._patch_slices[a][index][0].stop + top > data_shape[len(slices_pre)]:
                reflect_padding[-1] = self._patch_slices[a][index][0].stop + top - data_shape[len(slices_pre)]

        constant_padding = []
        if self.patch_size is not None and not all(p == 0 for p in self.patch_size):
            for dim_it, _slice in enumerate(reversed(slices)):
                p = (
                    0
                    if _slice.start + self.patch_size[-dim_it - 1] <= data_shape[-dim_it - 1]
                    else self.patch_size[-dim_it - 1] - (data_shape[-dim_it - 1] - _slice.start)
                )
                constant_padding.append(0)
                constant_padding.append(p)

        return PatchReadPlan(
            data_slices=tuple(slices_pre + slices),
            reflect_padding=tuple(reflect_padding),
            constant_padding=tuple(constant_padding),
            concatenate_extend_slice=extend_slice > 0,
        )

    def apply_read_plan(self, data: torch.Tensor, plan: PatchReadPlan) -> torch.Tensor:
        data_sliced = data
        if any(plan.reflect_padding):
            data_sliced = F.pad(data_sliced, plan.reflect_padding, "reflect")
        if any(plan.constant_padding):
            data_sliced = F.pad(
                data_sliced,
                plan.constant_padding,
                "constant",
                (
                    0
                    if data_sliced.dtype == torch.uint8
                    else (self.pad_value if self.pad_value is not None else float(data.min().item()))
                ),
            )
        if self.patch_size is not None and not all(p == 0 for p in self.patch_size):
            for d in [i for i, v in enumerate(reversed(self.patch_size)) if v == 1]:
                data_sliced = torch.squeeze(data_sliced, dim=len(data_sliced.shape) - d - 1)
        return (
            torch.cat([data_sliced[:, i, ...] for i in range(data_sliced.shape[1])], dim=0)
            if plan.concatenate_extend_slice
            else data_sliced
        )

    def get_data(self, data: torch.Tensor, index: int, a: int, is_input: bool) -> list[torch.Tensor]:
        plan = self.get_read_plan(list(data.shape), index, a, is_input)
        data_sliced = data[plan.data_slices]
        return self.apply_read_plan(data_sliced, plan)

    def get_size(self, a: int = 0) -> int:
        return len(self._patch_slices[a])


@config("Patch")
class DatasetPatch(Patch):
    """Patch definition applied when sampling data from datasets."""

    def __init__(
        self,
        patch_size: list[int] = [128, 128, 128],
        overlap: int | None = None,
        pad_value: float | None = None,
        extend_slice: int = 0,
    ) -> None:
        super().__init__(patch_size, overlap, pad_value, extend_slice)

    def init(self, key: str = ""):
        pass


@config()
class ModelPatch(Patch):
    """Patch definition applied inside model graphs during prediction or training."""

    def __init__(
        self,
        patch_size: list[int] = [128, 128, 128],
        overlap: int | None = None,
        patch_combine: str | None = None,
        pad_value: float | None = None,
        extend_slice: int = 0,
    ) -> None:
        super().__init__(patch_size, overlap, pad_value, extend_slice)
        self._patch_combine = patch_combine
        self.patch_combine: PathCombine | None = None

    def init(self, key: str):
        if self._patch_combine is not None:
            module, name = get_module(self._patch_combine, "konfai.data.patching")
            self.patch_combine = apply_config(key)(getattr(module, name))()
        if self.patch_size is not None and self.overlap is not None:
            if self.patch_combine is not None:
                self.patch_combine.set_patch_config([i for i in self.patch_size if i > 1], self.overlap)
        else:
            self.patch_combine = None

    def disassemble(self, *data_list: torch.Tensor) -> Iterator[list[torch.Tensor]]:
        for i in range(self.get_size()):
            yield [self.get_data(data, i, 0, True) for data in data_list]


class DatasetManager:
    """Cache-backed manager for one dataset case and one source/destination group."""

    def __init__(
        self,
        index: int,
        group_src: str,
        group_dest: str,
        name: str,
        dataset: Dataset,
        patch: DatasetPatch | None,
        transforms: list[Transform],
        data_augmentations_list: list[DataAugmentationsList],
    ) -> None:
        self.group_src = group_src
        self.group_dest = group_dest
        self.name = name
        self.index = index
        self.dataset = dataset
        self.transforms = transforms
        self.loaded = False
        self.augmentationLoaded = False
        self.cache_attributes: list[Attribute] = []
        _shape, cache_attribute = self.dataset.get_infos(self.group_src, name)
        self.base_shape = list(_shape)
        self.cache_attributes.append(cache_attribute)
        _shape = list(_shape[1:])

        self.data: list[torch.Tensor] = []
        self.augmented_data: dict[int, torch.Tensor] = {}
        self.total_augmentations = 0

        for transform_function in transforms:
            _shape = transform_function.transform_shape(self.group_src, self.name, _shape, cache_attribute)

        self.patch = (
            DatasetPatch(
                patch_size=patch.patch_size,
                overlap=patch.overlap,
                pad_value=patch.pad_value,
                extend_slice=patch.extend_slice,
            )
            if patch
            else DatasetPatch(_shape)
        )
        self.patch.load(_shape, 0)
        self.shape = _shape
        self.data_augmentations_list = data_augmentations_list
        self._patch_stream_sources: dict[bool, tuple[Dataset, str, list[int], list[Transform]] | None] = {}
        self._stream_attributes_persisted: set[int] = set()
        self.reset_augmentation()
        self.cache_attributes_bak = copy.deepcopy(self.cache_attributes)

    def reset_augmentation(self, reset_state: bool = True):
        self.cache_attributes[:] = self.cache_attributes[:1]
        self.augmented_data.clear()
        self.total_augmentations = 0
        i = 1
        for data_augmentations in self.data_augmentations_list:
            shape = []
            caches_attribute = []
            for _ in range(data_augmentations.nb):
                shape.append(self.shape)
                caches_attribute.append(copy.deepcopy(self.cache_attributes[0]))

            for data_augmentation in data_augmentations.data_augmentations:
                if reset_state:
                    data_augmentation.reset_state(self.index)
                shape = data_augmentation.state_init(self.index, shape, caches_attribute)
            for it, s in enumerate(shape):
                self.cache_attributes.append(caches_attribute[it])
                self.patch.load(s, i)
                i += 1
            self.total_augmentations += data_augmentations.nb
        self.augmentationLoaded = self.total_augmentations == 0

    def load(
        self,
        pre_transform: list[Transform],
        data_augmentations_list: list[DataAugmentationsList],
        load_augmentations: bool = True,
    ) -> None:
        if not self.loaded:
            self._load(pre_transform)
        if load_augmentations and not self.augmentationLoaded:
            self._load_augmentation(data_augmentations_list)

    def _load(self, pre_transform: list[Transform]):
        self.cache_attributes = copy.deepcopy(self.cache_attributes_bak)
        i = len(pre_transform)
        data = None
        for transform_function in reversed(pre_transform):
            if isinstance(transform_function, Save):
                if transform_function.dataset:
                    if len(transform_function.dataset.split(":")) > 1:
                        filename, file_format = transform_function.dataset.split(":")
                    else:
                        filename = transform_function.dataset
                        file_format = "mha"
                    dataset = Dataset(filename, file_format)
                else:
                    dataset = self.dataset
                group_dest = transform_function.group if transform_function.group else self.group_dest
                if dataset.is_dataset_exist(group_dest, self.name):
                    data, attrib = dataset.read_data(group_dest, self.name)
                    self.cache_attributes[0].update(attrib)
                    break
            i -= 1

        if i == 0:
            data, _ = self.dataset.read_data(self.group_src, self.name)

        data = torch.from_numpy(data)

        if len(pre_transform):
            for transform_function in pre_transform[i:]:
                data = transform_function(self.name, data, self.cache_attributes[0])
                if isinstance(transform_function, Save):
                    if transform_function.dataset:
                        if len(transform_function.dataset.split(":")) > 1:
                            filename, file_format = transform_function.dataset.split(":")
                        else:
                            filename = transform_function.dataset
                            file_format = "mha"
                        dataset = Dataset(filename, file_format)
                    else:
                        dataset = self.dataset
                    group_dest = transform_function.group if transform_function.group else self.group_dest
                    dataset.write(
                        group_dest,
                        self.name,
                        data.numpy(),
                        self.cache_attributes[0],
                    )
        self.data.append(data)

        for i in range(len(self.cache_attributes) - 1):
            self.cache_attributes[i + 1].update(self.cache_attributes[0])
        self.loaded = True

    def _load_augmentation(self, data_augmentations_list: list[DataAugmentationsList]) -> None:
        start_index = 1
        for data_augmentations in data_augmentations_list:
            self._load_augmentation_group(start_index, data_augmentations)
            start_index += data_augmentations.nb
        self.augmentationLoaded = len(self.augmented_data) == self.total_augmentations

    def _load_augmentation_group(self, start_index: int, data_augmentations: DataAugmentationsList) -> None:
        if data_augmentations.nb == 0:
            return

        indices = range(start_index, start_index + data_augmentations.nb)
        if all(index in self.augmented_data for index in indices):
            return

        a_data = [self.data[0].clone() for _ in range(data_augmentations.nb)]
        for data_augmentation in data_augmentations.data_augmentations:
            if data_augmentation.groups is None or self.group_dest in data_augmentation.groups:
                a_data = data_augmentation(self.name, self.index, a_data)

        for index, data in zip(indices, a_data, strict=False):
            self.augmented_data[index] = data
        self.augmentationLoaded = len(self.augmented_data) == self.total_augmentations

    def _get_tensor(self, a: int) -> torch.Tensor:
        if a == 0:
            return self.data[0]

        if a not in self.augmented_data:
            start_index = 1
            for data_augmentations in self.data_augmentations_list:
                stop_index = start_index + data_augmentations.nb
                if start_index <= a < stop_index:
                    self._load_augmentation_group(start_index, data_augmentations)
                    break
                start_index = stop_index
            else:
                raise IndexError(f"Augmentation index {a} out of range for dataset '{self.name}'.")

        return self.augmented_data[a]

    @staticmethod
    def _required_stream_stats(transform: Transform) -> tuple[set[str], list[int] | None] | None:
        if isinstance(transform, Normalize):
            return {"Min", "Max"}, transform.channels
        if isinstance(transform, Standardize):
            required_stats = set()
            if transform.mean is None:
                required_stats.add("Mean")
            if transform.std is None:
                required_stats.add("Std")
            return required_stats, None
        if isinstance(transform, Clip):
            required_stats = set()
            if isinstance(transform.min_value, str):
                if transform.min_value != "min":
                    return None
                required_stats.add("Min")
            if isinstance(transform.max_value, str):
                if transform.max_value != "max":
                    return None
                required_stats.add("Max")
            return required_stats, None
        return None

    def _ensure_stream_stats(
        self,
        source_dataset: Dataset,
        source_group: str,
        cache_attribute: Attribute,
        required_stats: set[str],
        channels: list[int] | None = None,
    ) -> bool:
        missing_stats = [key for key in required_stats if key not in cache_attribute]
        if not missing_stats:
            return True

        stats = source_dataset.read_data_statistics(source_group, self.name, channels)
        stats_mapping = {
            "Min": stats["min"],
            "Max": stats["max"],
            "Mean": stats["mean"],
            "Std": stats["std"],
        }
        for key in missing_stats:
            if key in {"Mean", "Std"}:
                cache_attribute[key] = np.asarray([stats_mapping[key]], dtype=np.float32)
            else:
                cache_attribute[key] = stats_mapping[key]
        return all(key in cache_attribute for key in required_stats)

    def _supports_patch_stream_transform(
        self,
        transform: Transform,
        source_dataset: Dataset,
        source_group: str,
        cache_attribute: Attribute,
    ) -> bool:
        if isinstance(transform, TensorCast):
            return True
        if isinstance(transform, Clip) and transform.mask is not None:
            return False
        if isinstance(transform, Standardize) and transform.mask is not None:
            return False
        required_stream_stats = self._required_stream_stats(transform)
        if required_stream_stats is None:
            return False
        required_stats, channels = required_stream_stats
        return self._ensure_stream_stats(source_dataset, source_group, cache_attribute, required_stats, channels)

    @staticmethod
    def _dataset_from_spec(dataset_spec: str) -> Dataset:
        filename, _, file_format = split_path_spec(
            dataset_spec,
            default_format="mha",
            supported_extensions=SUPPORTED_EXTENSIONS,
        )
        return Dataset(filename, file_format)

    def _resolve_patch_stream_source(
        self,
        apply_augmentations: bool = True,
    ) -> tuple[Dataset, str, list[int], list[Transform]] | None:
        if apply_augmentations in self._patch_stream_sources:
            return self._patch_stream_sources[apply_augmentations]

        source_dataset = self.dataset
        source_group = self.group_src
        source_shape = self.base_shape
        trailing_transforms = self.transforms

        for index in range(len(self.transforms) - 1, -1, -1):
            transform = self.transforms[index]
            if isinstance(transform, Save):
                dataset = self._dataset_from_spec(transform.dataset) if transform.dataset else self.dataset
                group = transform.group if transform.group else self.group_dest
                if dataset.is_dataset_exist(group, self.name):
                    source_dataset = dataset
                    source_group = group
                    source_shape, _ = dataset.get_infos(group, self.name)
                    trailing_transforms = self.transforms[index + 1 :]
                    break

        stream_cache_attribute = Attribute(self.cache_attributes[0])
        if (not apply_augmentations or len(self.data_augmentations_list) == 0) and all(
            self._supports_patch_stream_transform(
                transform,
                source_dataset,
                source_group,
                stream_cache_attribute,
            )
            for transform in trailing_transforms
        ):
            self.cache_attributes[0] = Attribute(stream_cache_attribute)
            self.cache_attributes_bak[0] = Attribute(stream_cache_attribute)
            self._patch_stream_sources[apply_augmentations] = (
                source_dataset,
                source_group,
                list(source_shape),
                trailing_transforms,
            )
        else:
            self._patch_stream_sources[apply_augmentations] = None
        return self._patch_stream_sources[apply_augmentations]

    def can_stream_patch(self, a: int, apply_augmentations: bool = True) -> bool:
        return a == 0 and self._resolve_patch_stream_source(apply_augmentations) is not None

    def _get_streamed_data(
        self,
        index: int,
        a: int,
        is_input: bool,
        apply_augmentations: bool = True,
    ) -> tuple[torch.Tensor, Attribute]:
        stream_source = self._resolve_patch_stream_source(apply_augmentations)
        if stream_source is None:
            raise RuntimeError("Patch streaming requested on a dataset manager without a streaming source.")

        source_dataset, source_group, source_shape, transforms = stream_source
        plan = self.patch.get_read_plan(source_shape, index, a, is_input)
        data, attributes = source_dataset.read_data_slice(source_group, self.name, plan.data_slices)
        tensor = self.patch.apply_read_plan(torch.from_numpy(data), plan)
        cache_attribute = Attribute(self.cache_attributes[a])
        cache_attribute.update(attributes)
        persist = a not in self._stream_attributes_persisted
        keys_before = set(cache_attribute.keys()) if persist else set()
        for transform in transforms:
            tensor = transform(self.name, tensor, cache_attribute)
        if persist:
            # Streaming applies the trailing transforms per patch on this throwaway
            # copy, but state they record for their own inversion (e.g. TensorCast's
            # source dtype) must survive on the persistent case attribute, exactly as
            # the non-streamed path stores it while transforming the full volume.
            persistent = self.cache_attributes[a]
            persistent_keys = set(persistent.keys())
            for key, value in cache_attribute.items():
                if key not in keys_before and key not in persistent_keys:
                    dict.__setitem__(persistent, key, value)
            self._stream_attributes_persisted.add(a)
        return tensor, cache_attribute

    def unload(self) -> None:
        self.data.clear()
        self.augmented_data.clear()
        self.loaded = False
        self.augmentationLoaded = self.total_augmentations == 0

    def unload_augmentation(self) -> None:
        self.augmented_data.clear()
        self.augmentationLoaded = self.total_augmentations == 0

    def get_data(
        self,
        index: int,
        a: int,
        patch_transforms: list[Transform],
        is_input: bool,
        apply_augmentations: bool = True,
    ) -> torch.Tensor:
        if not self.loaded and self.can_stream_patch(a, apply_augmentations):
            data, _ = self._get_streamed_data(index, a, is_input, apply_augmentations)
        else:
            data = self.patch.get_data(self._get_tensor(a), index, a, is_input)
        for transform_function in patch_transforms:
            data = transform_function(self.name, data, self.cache_attributes[a])
        return data

    def get_size(self, a: int) -> int:
        return self.patch.get_size(a)
