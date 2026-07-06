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
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
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
        self._data_per_device: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}

    def set_patch_config(self, patch_size: list[int], overlap: int) -> None:
        self._data_per_device.clear()
        self.overlap = overlap
        if overlap <= 0:
            self.data = torch.ones(patch_size)
            return
        # The per-patch weight is the outer product of one 1-D window per axis. It is separable by
        # construction, so a per-axis partition of unity stays a partition of unity in N-D and overlapping
        # patches blend without darkening — no distance map or explicit renormalisation loop is needed, and
        # the trailing ``result / weight_sum`` in Accumulator.assemble stays exact at the volume borders.
        data = self._window_1d(patch_size[0], overlap)
        for size in patch_size[1:]:
            data = data.unsqueeze(-1) * self._window_1d(size, overlap)
        self.data = data

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        key = (tensor.device, tensor.dtype)
        if key not in self._data_per_device:
            # Match the tensor dtype: the weight has no reason to carry more precision than the data it
            # scales, and a float64 weight would upcast the whole (channels x volume) blend.
            self._data_per_device[key] = self.data.to(device=tensor.device, dtype=tensor.dtype)
        return self._data_per_device[key] * tensor

    @abstractmethod
    def _window_1d(self, size: int, overlap: int) -> torch.Tensor:
        """Return the 1-D blend weight along one axis (length ``size``, ``overlap`` voxels tapered per side)."""


class Mean(PathCombine):
    """Uniform weighting: overlapping patches are plain-averaged by the assembly normalisation."""

    def _window_1d(self, size: int, overlap: int) -> torch.Tensor:
        return torch.ones(size)


class Cosinus(PathCombine):
    """Raised-cosine (sin**2) taper: a smooth partition of unity across the patch overlap."""

    def _window_1d(self, size: int, overlap: int) -> torch.Tensor:
        window = torch.ones(size)
        # sin**2 ramp over the overlap; the neighbouring patch's cos**2 ramp is its complement, so the two
        # sum to exactly one across the overlap. The +0.5 phase keeps the very edge > 0 so a single-patch
        # border (where result/weight_sum must recover the raw value) never divides by zero.
        ramp = torch.sin((torch.arange(overlap, dtype=torch.float32) + 0.5) / overlap * (torch.pi / 2)) ** 2
        window[:overlap] = ramp
        window[size - overlap :] = ramp.flip(0)
        return window


class Gaussian(PathCombine):
    """nnU-Net-style Gaussian importance weighting.

    Favours patch centres — where the model sees the most surrounding context — and down-weights the
    borders, which suppresses seam artefacts. Not a partition of unity, but ``Accumulator.assemble``
    normalises by the accumulated weight, so overlapping patches still form a correct weighted average.
    """

    def __init__(self, sigma_scale: float = 0.125) -> None:
        super().__init__()
        self.sigma_scale = sigma_scale

    def _window_1d(self, size: int, overlap: int) -> torch.Tensor:
        sigma = max(size * self.sigma_scale, 1e-6)
        center = (size - 1) / 2
        coords = torch.arange(size, dtype=torch.float32)
        return torch.exp(-((coords - center) ** 2) / (2.0 * sigma**2))


class Accumulator:
    """Accumulate patch predictions and reassemble them into a full tensor."""

    def __init__(
        self,
        patch_slices: list[tuple[slice, ...]],
        patch_size: list[int],
        patch_combine: PathCombine | None = None,
        batch: bool = True,
    ) -> None:
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
        self._n = 2 if batch else 1
        self._count = len(patch_slices)
        self._filled = 0
        self._done = [False] * self._count
        # Patches are blended into these running buffers as they arrive (see add_layer), instead of
        # being kept until assembly: holding every patch of a large multi-class case (e.g. ~70 patches
        # of a 122-channel whole-body segmentation ≈ tens of GB) was the dominant reassembly RAM cost.
        self._result: torch.Tensor | None = None
        self._weight_sum: torch.Tensor | None = None
        self._weight_patch: torch.Tensor | None = None

    def add_layer(self, index: int, layer: torch.Tensor) -> None:
        # Blend each patch straight into the running accumulator and drop the patch, rather than
        # storing all patches for a single assemble() at the end. The overlap blend is a weighted sum,
        # so accumulating incrementally is equivalent; re-adding an index is a no-op (last-write wins is
        # not possible once blended, and the prediction pipeline adds each patch exactly once).
        if self._done[index]:
            return
        n = self._n
        if self._result is None:
            # Allocate to the ACTUAL volume extent (self.shape), not the patch-size-extended grid. The
            # last patch of each axis is padded up to patch_size for the model, but that padded tail lies
            # OUTSIDE the volume; blending it would over-allocate the accumulator by up to
            # (patch_size - overlap) per axis (then get cropped away). We crop each patch to its in-volume
            # part at blend time instead, so nothing outside the volume is ever allocated.
            self._result = torch.zeros(list(layer.shape[:n]) + list(self.shape), dtype=layer.dtype, device=layer.device)
            if self.patch_combine is not None:
                # Match the result dtype so the final ``result / weight_sum`` does not promote the whole
                # (channels x volume) accumulator to float32.
                self._weight_sum = torch.zeros(list(self.shape), dtype=layer.dtype, device=layer.device)
        patch_slice = self.patch_slices[index]
        data = layer
        for dim, s in enumerate(patch_slice):
            if s.stop - s.start == 1:
                data = data.unsqueeze(dim=dim + n)
        # Clamp each spatial destination to the volume, and crop the patch (and its weight window) to the
        # matching in-volume extent so the padded tail of border patches is discarded, not stored.
        dest = [slice(s.start, min(s.stop, self.shape[dim])) for dim, s in enumerate(patch_slice)]
        crop = tuple([slice(None)] * n + [slice(0, d.stop - d.start) for d in dest])
        slices_dest = tuple([slice(self._result.shape[i]) for i in range(n)] + dest)
        # Overlap blending weights each patch (edge bands < 1 so interior overlaps sum to unity).
        # A voxel covered by fewer patches (a volume border without whole-image padding) would sum
        # to < 1 and come out darkened (x0.5 edges, x0.25 corners), so divide by the accumulated weight.
        if self.patch_combine is not None and self._weight_sum is not None:
            self._result[slices_dest] += self.patch_combine(data)[crop]
            if self._weight_patch is None:
                self._weight_patch = self.patch_combine(torch.ones_like(data))[(0,) * n]
            self._weight_sum[tuple(dest)] += self._weight_patch[crop[n:]]
        else:
            self._result[slices_dest] = data[crop]
        self._done[index] = True
        self._filled += 1

    def is_full(self) -> bool:
        # O(1): a running counter avoids re-scanning every slot after each added patch
        # (the completion check ran once per patch, i.e. O(P^2) per case).
        return self._filled == self._count

    def assemble(self) -> torch.Tensor:
        if self._result is None:
            raise PatchError(
                "Accumulator.assemble() was called before any patch was added.",
                f"Expected up to {self._count} patch(es) via add_layer() before assembling.",
                "Add at least one patch (and check is_full()) before calling assemble().",
            )
        result = self._result
        # With padding weight_sum is ~1 and the division is a near no-op; the clamp guards borders
        # that no patch (fully) covered. In-place so we do not materialise a second
        # (channels x volume) tensor; weight_sum already matches result.dtype (fp16), so the division
        # stays in fp16 with no float32 promotion (same values as the out-of-place form).
        if self.patch_combine is not None and self._weight_sum is not None:
            result.div_(self._weight_sum.clamp(min=1e-8))
        # No final crop: patches are cropped to the volume at blend time, so result is already self.shape.

        self._result = None
        self._weight_sum = None
        self._weight_patch = None
        self._filled = 0
        self._done = [False] * self._count
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
