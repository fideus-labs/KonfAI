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

import warnings

import numpy as np
import torch

from konfai.data.transform.base import LocalityKind, PatchLocality, Transform, TransformInverse, sitk
from konfai.utils.dataset import Attribute, Dataset, data_to_image, image_to_data
from konfai.utils.dataset.statistics import read_masked_data_statistics
from konfai.utils.errors import DatasetManagerError, KonfAIWarning, TransformError
from konfai.utils.ITK import _require_simpleitk


def _seeded_scalar(cache_attribute: Attribute, key: str) -> float:
    """A seeded statistic, as whoever seeded it wrote it: a bare scalar or a one-element array."""
    try:
        return float(cache_attribute[key])
    except (TypeError, ValueError):
        return float(cache_attribute.get_tensor(key).reshape(-1)[0])


def _writes_into(volume: torch.Tensor, stat: torch.Tensor) -> bool:
    """Whether an op of ``volume`` with ``stat`` keeps its shape and dtype: it then writes into the
    volume, where the out-of-place spelling allocates a second one."""
    return (
        volume.is_floating_point()
        and torch.result_type(volume, stat) == volume.dtype
        and (stat.numel() == 1 or stat.shape[0] == volume.shape[0])
    )


def _shifted(tensor: torch.Tensor, by: float | torch.Tensor) -> torch.Tensor:
    """``tensor - by`` in a tensor of its own. A stored integer is cast once and shifted in place:
    the subtraction would hold the cast beside its result. A bool tensor refuses the subtraction."""
    dtype = torch.result_type(tensor, by)
    if tensor.dtype in (dtype, torch.bool):
        return tensor - by
    cast = tensor.to(dtype)
    return cast.sub_(by) if not isinstance(by, torch.Tensor) or _writes_into(cast, by) else cast - by


def _clamps_as_integers(dtype: torch.dtype, low: float, high: float) -> bool:
    """Whether the fills of a ``Clip`` to ``[low, high]`` are the integer clamp of a tensor of ``dtype``:
    whole bounds the dtype holds, over values a float32 comparison reads exactly. The fills compare a
    float copy of the volume, which a stored CT is then clipped without."""
    if dtype not in (torch.int8, torch.uint8, torch.int16):
        return False
    held = torch.iinfo(dtype)
    return all(bound.is_integer() and held.min <= bound <= held.max for bound in (low, high))


def _dataset_holding(datasets: list[Dataset], group: str, name: str) -> Dataset:
    """The dataset holding the case's ``group``, or a refusal naming it."""
    for dataset in datasets:
        if dataset.is_dataset_exist(group, name):
            return dataset
    raise DatasetManagerError(
        f"No dataset holds '{group}' for case '{name}'.",
        "Check the group name against the datasets the run reads.",
    )


def _percentile_of(bound: str, statistic: str, argument: str) -> float | None:
    """The percentile a string bound of ``Clip`` names, ``None`` for the statistic itself."""
    if bound == statistic:
        return None
    try:
        percentile = float(bound.split(":")[1]) if bound.startswith("percentile:") else -1.0
    except (IndexError, ValueError):
        percentile = -1.0
    if not 0.0 <= percentile <= 100.0:
        raise TransformError(
            f"'Clip' was given {argument}={bound!r}.",
            f"A bound is a number, '{statistic}', or 'percentile:<p>' with p in [0, 100] (percentile:99.5).",
        )
    return percentile


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


def _masked_values(stage: "Clip | Standardize", name: str, tensor: torch.Tensor) -> torch.Tensor:
    """The tensor's values under ``stage``'s mask, on the whole-volume path."""
    mask = stage.read_companion(stage.mask, name)  # type: ignore[arg-type]
    if tuple(mask.shape) != tuple(tensor.shape):
        raise TransformError(
            f"The mask '{stage.mask}' has shape {list(mask.shape)} where the tensor in hand has {list(tensor.shape)}.",
            "The mask is read on the stored grid: apply this stage before the stages that change the grid.",
        )
    return tensor[mask != 0]


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
            raise TransformError(
                f"'Clip' was given max_value={max_value}, which is not above min_value={min_value}.",
                "Give a max_value greater than min_value.",
            )
        for bound, statistic, argument in ((min_value, "min", "min_value"), (max_value, "max", "max_value")):
            if isinstance(bound, str):
                _percentile_of(bound, statistic, argument)
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
        saved = {key for key, save in (("Min", self.save_clip_min), ("Max", self.save_clip_max)) if save}
        return PatchLocality(LocalityKind.GLOBAL_STAT, stat_keys=frozenset(stat_keys), records=frozenset(saved))

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        seeded_masked = self.mask is not None and "StatisticsSeeded" in cache_attribute
        selected: torch.Tensor | None = None

        def values() -> torch.Tensor:
            nonlocal selected
            if selected is None:
                selected = tensor if self.mask is None else _masked_values(self, name, tensor)
            return selected

        if isinstance(self.min_value, str):
            percentile = _percentile_of(self.min_value, "min", "min_value")
            if percentile is None:
                # Seeded first: on a streamed path a bound computed here would be one region's. A
                # masked bound seeds from the masked disk scan, a bare seed being an unmasked stage's.
                if seeded_masked:
                    min_value = self._masked_seed.statistics(self.datasets, name)["min"]  # type: ignore[union-attr]
                elif self.mask is None and "StatisticsSeeded" in cache_attribute and "Min" in cache_attribute:
                    min_value = _seeded_scalar(cache_attribute, "Min")
                else:
                    min_value = torch.min(values())
            else:
                # ``np.percentile`` cannot coerce a CUDA tensor; ``.cpu()`` is a no-op on a host one.
                min_value = np.percentile(values().detach().cpu(), percentile)
        else:
            min_value = self.min_value

        if isinstance(self.max_value, str):
            percentile = _percentile_of(self.max_value, "max", "max_value")
            if percentile is None:
                if seeded_masked:
                    max_value = self._masked_seed.statistics(self.datasets, name)["max"]  # type: ignore[union-attr]
                elif self.mask is None and "StatisticsSeeded" in cache_attribute and "Max" in cache_attribute:
                    max_value = _seeded_scalar(cache_attribute, "Max")
                else:
                    max_value = torch.max(values())
            else:
                max_value = np.percentile(values().detach().cpu(), percentile)
        else:
            max_value = self.max_value

        # A resolved bound may be a torch 0-d tensor or a numpy scalar; the assignments below want a
        # Python float.
        min_value = float(min_value)
        max_value = float(max_value)

        # Fast paths: one fused in-place clamp, for float32 with non-NaN bounds and for a stored integer
        # with whole bounds. float16/float64 compare at another precision than the fallback, and
        # clamp_ propagates a NaN bound where the fallback fill no-ops on it.
        if tensor.dtype == torch.float32 and min_value == min_value and max_value == max_value:
            tensor.clamp_(min=min_value, max=max_value)
        elif _clamps_as_integers(tensor.dtype, min_value, max_value):
            tensor.clamp_(min=int(min_value), max=int(max_value))
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

    # The inverse holds one intermediate volume next to its result; the forward writes into its result.
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
            raise TransformError(
                f"'Normalize' was given max_value={max_value}, which is not above min_value={min_value}.",
                "Give a max_value greater than min_value.",
            )
        self.lazy = lazy
        self.min_value = min_value
        self.max_value = max_value
        self.channels = channels

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        # Rescaling uses the volume-global Min/Max, seeded once so every patch sees the same range.
        return PatchLocality(
            LocalityKind.GLOBAL_STAT,
            stat_keys=frozenset({"Min", "Max"}),
            stat_channels=self.channels,
            takes_present=True,
        )

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

            # Never into the tensor handed over: a patch transform is handed a view of the loaded case,
            # which later patches read again. Every branch yields the floating dtype the affine map below
            # does: an integer case filled or written in place would stay integer.
            dtype = torch.result_type(tensor, 1.0)
            if norm == 0:
                warnings.warn(
                    f"Norm is zero for case '{name}': input is constant with value = {input_min}.",
                    KonfAIWarning,
                    stacklevel=2,
                )
                if self.channels:
                    tensor = tensor.to(dtype, copy=True)
                    for channel in self.channels:
                        tensor[channel].fill_(self.min_value)
                else:
                    tensor = torch.full_like(tensor, self.min_value, dtype=dtype)
            else:
                # One new volume, then in place: each out-of-place step would allocate a volume of its own.
                span = self.max_value - self.min_value
                if self.channels:
                    tensor = tensor.to(dtype, copy=True)
                    for channel in self.channels:
                        tensor[channel].sub_(input_min).mul_(span).div_(norm).add_(self.min_value)
                else:
                    tensor = _shifted(tensor, input_min).mul_(span).div_(norm).add_(self.min_value)

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
            # The product by a float is floating whatever the tensor, so the rest writes into it.
            restored = (tensor - self.min_value) * (input_max - input_min)
            return restored.div_(self.max_value - self.min_value).add_(input_min)


class UnNormalize(Transform):
    # One intermediate volume stands next to the result.
    working_multiple = 1.0

    locality = LocalityKind.POINTWISE

    def __init__(self, min_value: int = -1024, max_value: int = 3071) -> None:
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        # The division is floating whatever the tensor, so the rest writes into it.
        return ((tensor + 1) / 2).mul_(self.max_value - self.min_value).add_(self.min_value)


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
        return PatchLocality(LocalityKind.GLOBAL_STAT, stat_keys=frozenset(stat_keys), takes_present=True)

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
            return self._standardized(tensor, mean_value, std_value)

        selected: torch.Tensor | None = None

        def values() -> torch.Tensor:
            # Cast once: the mean and the std of an integer case would each convert the volume.
            nonlocal selected
            if selected is None:
                selected = (tensor if self.mask is None else _masked_values(self, name, tensor)).type(torch.float32)
            return selected

        if "Mean" not in cache_attribute:
            cache_attribute["Mean"] = (
                torch.tensor([torch.mean(values())]) if self.mean is None else torch.tensor(self.mean)
            )

        if "Std" not in cache_attribute:
            cache_attribute["Std"] = torch.tensor([torch.std(values())]) if self.std is None else torch.tensor(self.std)
        if self.lazy:
            return tensor
        return self._standardized(tensor, cache_attribute.get_tensor("Mean"), cache_attribute.get_tensor("Std"))

    def _standardized(self, tensor: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        centered = _shifted(tensor, self._broadcast(mean.to(tensor.device), tensor))
        std = self._broadcast(std.to(tensor.device), tensor)
        return centered.div_(std) if _writes_into(centered, std) else centered / std

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
            scaled = tensor * std
            return scaled.add_(mean) if _writes_into(scaled, mean) else scaled + mean


class TensorCast(TransformInverse):
    working_multiple = 1.0

    # Wide enough to hold every dtype a volume is read as (int8/int16/uint8/float32) with no value moved.
    _VALUE_PRESERVING_DTYPES = frozenset({torch.float32, torch.float64})

    def __init__(self, dtype: str = "float32", inverse: bool = True) -> None:
        super().__init__(inverse)
        self.dtype: torch.dtype = TensorCast.safe_dtype_cast(dtype)

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
        dtype = getattr(torch, dtype_str, None)
        if not isinstance(dtype, torch.dtype):
            raise TransformError(f"'{dtype_str}' is not a torch dtype.", "Name one: float32, float16, int16, uint8...")
        return dtype

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

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        return PatchLocality(
            LocalityKind.WHOLE_VOLUME, reason="its lookup table is built from the whole volume's histogram"
        )

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        image = data_to_image(tensor, cache_attribute)
        reference = self.dataset_holding(self.reference_group, name)
        if reference is None:
            raise DatasetManagerError(
                f"The reference '{self.reference_group}/{name}' is not in any dataset.",
                "Add the group to a dataset the run reads, or name one that is there.",
            )
        image_ref = reference.read_image(self.reference_group, name)
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

    def __init__(self) -> None:
        super().__init__()

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        return PatchLocality(
            LocalityKind.GLOBAL_STAT, stat_keys=frozenset({"Min", "Max", "Mean", "Std"}), records=frozenset()
        )

    def __call__(self, name: str, tensors: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        trusted = "StatisticsSeeded" in cache_attribute
        values: torch.Tensor | None = None
        for seeded, recorded in self._KEYS:
            if not trusted or seeded not in cache_attribute:
                if values is None:
                    values = tensors.float()
                cache_attribute[recorded] = getattr(values, seeded.lower())()
                continue
            cache_attribute[recorded] = _seeded_scalar(cache_attribute, seeded)
        return tensors
