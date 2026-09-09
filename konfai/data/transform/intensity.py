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


"""Value transforms: clipping, normalization, casting, histogram matching, statistics."""

import numpy as np
import torch

from konfai.data.transform.base import LocalityKind, PatchLocality, Transform, TransformInverse, sitk
from konfai.utils.dataset import Attribute, Dataset, data_to_image, image_to_data
from konfai.utils.dataset.statistics import read_masked_data_statistics
from konfai.utils.errors import DatasetManagerError, TransformError
from konfai.utils.ITK import _require_simpleitk


def _seeded_scalar(cache_attribute: Attribute, key: str) -> float:
    """A seeded statistic, as whoever seeded it wrote it: a bare scalar or a one-element array."""
    try:
        return float(cache_attribute[key])
    except (TypeError, ValueError):
        return float(cache_attribute.get_tensor(key).reshape(-1)[0])


def _dataset_holding(datasets: list[Dataset], group: str, name: str) -> Dataset:
    """The dataset holding the case's ``group``, or a refusal naming it."""
    for dataset in datasets:
        if dataset.is_dataset_exist(group, name):
            return dataset
    raise DatasetManagerError(
        f"No dataset holds '{group}' for case '{name}'.",
        "Check the group name against the datasets the run reads.",
    )


class _MaskedStatisticsSeed:
    """The masked whole-volume statistics of a stage's own group, per case, from the stores.

    A masked ``Clip``/``Standardize`` needs the case's statistic under the mask before its first
    region: the two volumes are scanned once per case, streamed, and memoised here.
    ``transform_shape`` records the group the chain reads, which ``__call__`` is never told.
    """

    def __init__(self, mask: str) -> None:
        self.mask = mask
        self.group: str | None = None
        self._by_case: dict[str, dict[str, float]] = {}

    def record_group(self, group_src: str) -> None:
        if group_src:
            self.group = group_src

    def statistics(self, datasets: list[Dataset], name: str) -> dict[str, float]:
        cached = self._by_case.get(name)
        if cached is not None:
            return cached
        if self.group is None:
            raise TransformError(
                "The masked statistic has no group to scan: the chain was never planned.",
                "Report this: transform_shape() records the group before any region flows.",
            )
        stats = read_masked_data_statistics(
            _dataset_holding(datasets, self.group, name),
            self.group,
            _dataset_holding(datasets, self.mask, name),
            self.mask,
            name,
        )
        self._by_case[name] = stats
        return stats


class Clip(Transform):
    """Clip tensor intensities to a fixed or data-dependent value range."""

    working_multiple = 2.5

    def __init__(
        self,
        min_value: float | str = -1024,
        max_value: float | str = 1024,
        save_clip_min: bool = False,
        save_clip_max: bool = False,
        mask: str | None = None,
    ) -> None:
        super().__init__()
        if isinstance(min_value, int | float) and isinstance(max_value, int | float) and max_value <= min_value:
            raise ValueError(
                f"[Clip] Invalid clipping range: max_value ({max_value}) must be greater than min_value ({min_value})"
            )
        self.min_value = min_value
        self.max_value = max_value
        self.save_clip_min = save_clip_min
        self.save_clip_max = save_clip_max
        self.mask = mask
        self._masked_seed = _MaskedStatisticsSeed(mask) if mask is not None else None

    def transform_shape(self, group_src: str, name: str, shape: list[int], cache_attribute: Attribute) -> list[int]:
        # Identity on the shape; a masked bound records the group the masked disk scan needs.
        if self._masked_seed is not None:
            self._masked_seed.record_group(group_src)
        return shape

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        # A percentile bound needs the whole histogram. A 'min'/'max' bound needs a global statistic:
        # a seeded disk one, or under a mask a masked disk scan the stage seeds itself (GLOBAL_STAT
        # with no key). Fixed float bounds are POINTWISE.
        stat_keys: set[str] = set()
        for bound, key in ((self.min_value, "Min"), (self.max_value, "Max")):
            if isinstance(bound, str):
                if bound == key.lower():  # exactly as __call__ matches it; "MIN" is refused there
                    stat_keys.add(key)
                else:
                    return PatchLocality(
                        LocalityKind.WHOLE_VOLUME,
                        reason=f"a '{bound}' bound needs the whole histogram; fixed values or"
                        " 'min'/'max' (a seeded statistic) stream",
                    )
        if not stat_keys:
            return PatchLocality(LocalityKind.POINTWISE)
        if self.mask is not None:
            return PatchLocality(LocalityKind.GLOBAL_STAT)
        return PatchLocality(LocalityKind.GLOBAL_STAT, stat_keys=frozenset(stat_keys))

    def _masked_values(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        """The tensor's values under the mask, on the whole-volume path."""
        mask = self.read_companion(self.mask, name)  # type: ignore[arg-type]
        if tuple(mask.shape) != tuple(tensor.shape):
            raise TransformError(
                f"The mask '{self.mask}' has shape {list(mask.shape)} where the tensor in hand has"
                f" {list(tensor.shape)}: it cannot be indexed against a region.",
                "A masked bound needs the whole volume here; report this if the chain was planned.",
            )
        return tensor[mask == 1]

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        seeded_masked = self.mask is not None and "StatisticsSeeded" in cache_attribute
        selected: torch.Tensor | None = None

        def values() -> torch.Tensor:
            nonlocal selected
            if selected is None:
                selected = tensor if self.mask is None else self._masked_values(name, tensor)
            return selected

        if isinstance(self.min_value, str):
            if self.min_value == "min":
                # Seeded first: on a streamed path a bound computed here would be one region's. A
                # masked bound seeds from the masked disk scan, a bare seed being an unmasked stage's.
                if seeded_masked:
                    min_value = self._masked_seed.statistics(self.datasets, name)["min"]  # type: ignore[union-attr]
                elif self.mask is None and "StatisticsSeeded" in cache_attribute and "Min" in cache_attribute:
                    min_value = _seeded_scalar(cache_attribute, "Min")
                else:
                    min_value = torch.min(values())
            elif self.min_value.startswith("percentile:"):
                try:
                    percentile = float(self.min_value.split(":")[1])
                    # ``np.percentile`` cannot coerce a CUDA tensor; ``.cpu()`` is a no-op on a host one.
                    min_value = np.percentile(values().detach().cpu(), percentile)
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
                if seeded_masked:
                    max_value = self._masked_seed.statistics(self.datasets, name)["max"]  # type: ignore[union-attr]
                elif self.mask is None and "StatisticsSeeded" in cache_attribute and "Max" in cache_attribute:
                    max_value = _seeded_scalar(cache_attribute, "Max")
                else:
                    max_value = torch.max(values())
            elif self.max_value.startswith("percentile:"):
                try:
                    percentile = float(self.max_value.split(":")[1])
                    max_value = np.percentile(values().detach().cpu(), percentile)
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

        # A resolved bound may be a torch 0-d tensor or a numpy scalar; the assignments below want a
        # Python float.
        min_value = float(min_value)
        max_value = float(max_value)

        # Fast path: one fused in-place clamp, for float32 and non-NaN bounds only. An integer tensor
        # rejects float bounds, float16/float64 compare at another precision than the fallback, and
        # clamp_ propagates a NaN bound where the fallback fill no-ops on it.
        if tensor.dtype == torch.float32 and min_value == min_value and max_value == max_value:
            tensor.clamp_(min=min_value, max=max_value)
        else:
            tensor.masked_fill_(tensor.float() < min_value, min_value)
            tensor.masked_fill_(tensor.float() > max_value, max_value)
        if self.save_clip_min:
            cache_attribute["Min"] = min_value
        if self.save_clip_max:
            cache_attribute["Max"] = max_value
        return tensor


class Normalize(TransformInverse):
    """Map intensities to a target min/max interval and optionally invert it."""

    # The rescale is a chain of out-of-place ops, so a volume-worth stands next to the result.
    working_multiple = 1.0

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

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        # Rescaling uses the volume-global Min/Max, seeded once so every patch sees the same range.
        return PatchLocality(LocalityKind.GLOBAL_STAT, stat_keys=frozenset({"Min", "Max"}), stat_channels=self.channels)

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

    def inverse_patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        # The inverse only pops what the forward stacked: a per-voxel affine map.
        return PatchLocality(LocalityKind.POINTWISE)

    def inverse(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        if self.lazy:
            return tensor
        else:
            input_min = float(cache_attribute.pop("Min"))
            input_max = float(cache_attribute.pop("Max"))
            return (tensor - self.min_value) * (input_max - input_min) / (self.max_value - self.min_value) + input_min


class UnNormalize(Transform):
    # The rescale is a chain of out-of-place ops, so a volume-worth stands next to the result.
    working_multiple = 1.0

    locality = LocalityKind.POINTWISE

    def __init__(self, min_value: int = -1024, max_value: int = 3071) -> None:
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        return (tensor + 1) / 2 * (self.max_value - self.min_value) + self.min_value


class Standardize(TransformInverse):
    """Standardize tensors using cached or computed mean and standard deviation."""

    working_multiple = 1.0  # the float copy the statistics are taken on

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
        self._masked_seed = _MaskedStatisticsSeed(mask) if mask is not None else None

    def transform_shape(self, group_src: str, name: str, shape: list[int], cache_attribute: Attribute) -> list[int]:
        # Identity on the shape; a masked statistic records the group the masked disk scan needs.
        if self._masked_seed is not None:
            self._masked_seed.record_group(group_src)
        return shape

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        # Any of mean/std left unset is a global statistic: a seeded disk one, or under a mask a
        # masked disk scan the stage seeds itself once per case. With both given, POINTWISE.
        stat_keys: set[str] = set()
        if self.mean is None:
            stat_keys.add("Mean")
        if self.std is None:
            stat_keys.add("Std")
        if not stat_keys:
            return PatchLocality(LocalityKind.POINTWISE)
        if self.mask is not None:
            return PatchLocality(LocalityKind.GLOBAL_STAT)
        return PatchLocality(LocalityKind.GLOBAL_STAT, stat_keys=frozenset(stat_keys))

    def _masked_values(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        """The tensor's values under the mask, on the whole-volume path."""
        mask = self.read_companion(self.mask, name)  # type: ignore[arg-type]
        if tuple(mask.shape) != tuple(tensor.shape):
            raise TransformError(
                f"The mask '{self.mask}' has shape {list(mask.shape)} where the tensor in hand has"
                f" {list(tensor.shape)}: it cannot be indexed against a region.",
                "A masked statistic needs the whole volume here; report this if the chain was planned.",
            )
        return tensor[mask == 1]

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        if self.mask is not None and (self.mean is None or self.std is None) and "StatisticsSeeded" in cache_attribute:
            # A streamed region: the mask cannot be indexed against it, and a bare 'Mean' seed may be
            # an unmasked stage's, so the masked statistic is scanned from the stores once.
            stats = self._masked_seed.statistics(self.datasets, name)  # type: ignore[union-attr]
            mean_value = torch.tensor(self.mean) if self.mean is not None else torch.tensor([float(stats["mean"])])
            std_value = torch.tensor(self.std) if self.std is not None else torch.tensor([float(stats["std"])])
            if "Mean" not in cache_attribute:
                cache_attribute["Mean"] = mean_value
            if "Std" not in cache_attribute:
                cache_attribute["Std"] = std_value
            if self.lazy:
                return tensor
            mean = self._broadcast(mean_value.to(tensor.device), tensor)
            std = self._broadcast(std_value.to(tensor.device), tensor)
            return (tensor - mean) / std

        selected: torch.Tensor | None = None

        def values() -> torch.Tensor:
            nonlocal selected
            if selected is None:
                selected = tensor if self.mask is None else self._masked_values(name, tensor)
            return selected

        if "Mean" not in cache_attribute:
            cache_attribute["Mean"] = (
                torch.tensor([torch.mean(values().type(torch.float32))])
                if self.mean is None
                else torch.tensor(self.mean)
            )

        if "Std" not in cache_attribute:
            cache_attribute["Std"] = (
                torch.tensor([torch.std(values().type(torch.float32))]) if self.std is None else torch.tensor(self.std)
            )
        if self.lazy:
            return tensor
        else:
            mean = self._broadcast(cache_attribute.get_tensor("Mean").to(tensor.device), tensor)
            std = self._broadcast(cache_attribute.get_tensor("Std").to(tensor.device), tensor)
            return (tensor - mean) / std

    @staticmethod
    def _broadcast(stat: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
        """Shape a scalar or per-channel statistic to broadcast over a channel-first tensor."""
        if stat.numel() > 1:
            return stat.reshape(-1, *([1] * (tensor.dim() - 1)))
        return stat

    def inverse_patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        # Mask or not, the inverse only pops the Mean/Std the forward stacked: a per-voxel affine map.
        return PatchLocality(LocalityKind.POINTWISE)

    def inverse(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        if self.lazy:
            return tensor
        else:
            # The stats parse back as float64 on the CPU; float32 on the volume's device keeps an
            # fp16 output from being promoted.
            mean = self._broadcast(cache_attribute.pop_tensor("Mean").to(tensor.device, torch.float32), tensor)
            std = self._broadcast(cache_attribute.pop_tensor("Std").to(tensor.device, torch.float32), tensor)
            return tensor * std + mean


class TensorCast(TransformInverse):
    working_multiple = 1.0

    # Wide enough to hold every dtype a volume is read as (int8/int16/uint8/float32) with no value moved.
    _VALUE_PRESERVING_DTYPES = frozenset({torch.float32, torch.float64})

    def __init__(self, dtype: str = "float32", inverse: bool = True) -> None:
        super().__init__(inverse)
        self.dtype: torch.dtype = getattr(torch, dtype)

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        # The stored volume's statistics stay a later GLOBAL_STAT's input only where the cast keeps
        # every value: float32 holds an int16 exactly, float16 runs out of mantissa at 2048.
        return PatchLocality(
            LocalityKind.POINTWISE, preserves_statistics=self.dtype in TensorCast._VALUE_PRESERVING_DTYPES
        )

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


class HistogramMatching(Transform):
    """Match a volume's intensity distribution onto a reference group's.

    Whole-volume: the LUT is built from the volume's 256-bin histogram, which is not a statistic
    ``GLOBAL_STAT`` names.
    """

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
            raise DatasetManagerError(
                f"The reference '{self.reference_group}/{name}' is not in any dataset.",
                "Add the group to a dataset the run reads, or name one that is there.",
            )
        _require_simpleitk()
        matcher = sitk.HistogramMatchingImageFilter()
        matcher.SetNumberOfHistogramLevels(256)
        matcher.SetNumberOfMatchPoints(1)
        matcher.SetThresholdAtMeanIntensity(True)
        result, _ = image_to_data(matcher.Execute(image, image_ref))
        return torch.tensor(result)


class Statistics(Transform):
    """Record the volume's Min/Max/Mean/Std on the case, under ``Image*`` keys.

    The four numbers are what the disk-statistics scan computes, so a streamed chain seeds them
    (``GLOBAL_STAT``) and each region restates the case's answer.
    """

    working_multiple = 2.0

    _KEYS = (("Min", "ImageMin"), ("Max", "ImageMax"), ("Mean", "ImageMean"), ("Std", "ImageStd"))

    alters_values = False

    def __init__(self) -> None:
        super().__init__()

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        return PatchLocality(LocalityKind.GLOBAL_STAT, stat_keys=frozenset({"Min", "Max", "Mean", "Std"}))

    def __call__(self, name: str, tensors: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        trusted = "StatisticsSeeded" in cache_attribute
        for seeded, recorded in self._KEYS:
            if not trusted or seeded not in cache_attribute:
                cache_attribute[recorded] = getattr(tensors.float(), seeded.lower())()
                continue
            cache_attribute[recorded] = _seeded_scalar(cache_attribute, seeded)
        return tensors
