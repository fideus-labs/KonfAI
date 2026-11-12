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

import inspect
import warnings
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
import torch

from konfai.data.transform.base import (
    MEASURING_STATISTICS,
    LocalityKind,
    PatchLocality,
    Transform,
    TransformInverse,
    sitk,
)
from konfai.utils.config import apply_config
from konfai.utils.dataset import Attribute, Dataset, data_to_image, image_to_data
from konfai.utils.dataset.statistics import read_masked_data_statistics, read_nonzero_data_statistics
from konfai.utils.errors import DatasetManagerError, KonfAIWarning, TransformError
from konfai.utils.ITK import _require_simpleitk
from konfai.utils.utils import get_module, module_attribute


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


def percentile_key(percentile: float) -> str:
    """The statistic a ``percentile:<p>`` bound of ``Clip`` is, as a scope and a seed name it: ``p`` to the last
    digit, so two percentiles never share a name."""
    return f"Percentile:{percentile!r}"


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
    region: the two volumes are scanned once per case, streamed, and memoised here. With no mask
    (``Standardize(nonzero=True)``) the selection is the volume's own non-zero voxels, per channel.
    ``transform_shape`` records the group the chain reads, which ``__call__`` is never told.
    """

    def __init__(self, mask: str | None) -> None:
        self.mask = mask
        self.group: str | None = None
        self._by_case: dict[str, dict[str, Any]] = {}

    def record_group(self, group_src: str) -> None:
        if group_src:
            self.group = group_src

    def statistics(self, datasets: list[Dataset], name: str) -> dict[str, Any]:
        cached = self._by_case.get(name)
        if cached is not None:
            return cached
        if self.group is None:
            raise TransformError(
                "The masked statistic has no group to scan: the chain was never planned.",
                "Report this: transform_shape() records the group before any region flows.",
            )
        holder = _dataset_holding(datasets, self.group, name)
        if self.mask is None:
            stats = read_nonzero_data_statistics(holder, self.group, name)
        else:
            stats = read_masked_data_statistics(
                holder, self.group, _dataset_holding(datasets, self.mask, name), self.mask, name
            )
        self._by_case[name] = stats
        return stats


def _stated(stated: list[float] | None, channel: int, measured: float) -> float:
    """The number a stage states for ``channel`` (one for every channel, or one per channel), else ``measured``."""
    if stated is None:
        return measured
    return float(stated[0] if len(stated) == 1 else stated[channel])


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
        # A 'min'/'max' bound needs a global statistic: a seeded disk one, or under a mask a masked disk scan the
        # stage seeds itself (GLOBAL_STAT with no key). A percentile is one too, the stored volume's read exactly
        # by bounded passes; under a mask it needs the whole histogram. Fixed float bounds are POINTWISE. A bound
        # is recorded, under its statistic's name, only when it is saved.
        stat_keys: set[str] = set()
        percentiles: set[str] = set()
        saved = {key for key, save in (("Min", self.save_clip_min), ("Max", self.save_clip_max)) if save}
        bounds = ((self.min_value, "Min", "min_value"), (self.max_value, "Max", "max_value"))
        for bound, key, argument in bounds:
            if isinstance(bound, str):
                percentile = _percentile_of(bound, key.lower(), argument)
                if percentile is None:
                    stat_keys.add(key)
                elif self.mask is not None:
                    return PatchLocality(
                        LocalityKind.WHOLE_VOLUME,
                        reason=f"a '{bound}' bound under a mask needs the whole histogram; fixed values,"
                        " 'min'/'max' or a percentile without a mask stream",
                    )
                else:
                    percentiles.add(percentile_key(percentile))
                    if key in saved:
                        saved.add(percentile_key(percentile))
        if not stat_keys and not percentiles:
            return PatchLocality(LocalityKind.POINTWISE)
        if self.mask is not None:
            return PatchLocality(LocalityKind.GLOBAL_STAT)
        return PatchLocality(
            LocalityKind.GLOBAL_STAT, stat_keys=frozenset(stat_keys | percentiles), records=frozenset(saved)
        )

    def _percentile(
        self, values: Callable[[], torch.Tensor], percentile: float, cache_attribute: Attribute, save: bool
    ) -> float:
        """The ``percentile`` of the volume: seeded on a streamed region, else measured (``np.percentile``), and
        recorded when the bound is saved or a pass over the whole volume measures it for the regions."""
        key = percentile_key(percentile)
        if self.mask is None and "StatisticsSeeded" in cache_attribute and key in cache_attribute:
            return _seeded_scalar(cache_attribute, key)
        # ``np.percentile`` cannot coerce a CUDA tensor; ``.cpu()`` is a no-op on a host one.
        value = float(np.percentile(values().detach().cpu(), percentile))
        if save or MEASURING_STATISTICS in cache_attribute:
            cache_attribute[key] = value
        return value

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
                min_value = self._percentile(values, percentile, cache_attribute, self.save_clip_min)
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
                max_value = self._percentile(values, percentile, cache_attribute, self.save_clip_max)
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
    """Standardize tensors using cached or computed mean and standard deviation.

    ``nonzero`` standardizes each channel over its voxels that are not exactly 0: its mean and population
    standard deviation (N, not N - 1; 0 taken as 1) over those voxels, applied to them alone, the zeros kept,
    in float32. Per channel always: KonfAI states a ``Standardize``'s numbers per channel. Not invertible (a voxel at 0 and one at the mean both land on 0), so it takes ``inverse: false``.
    """

    working_multiple = 1.0  # the float copy the statistics are taken on

    def __init__(
        self,
        lazy: bool = False,
        mean: list[float] | None = None,
        std: list[float] | None = None,
        mask: str | None = None,
        inverse: bool = True,
        nonzero: bool = False,
    ) -> None:
        super().__init__(inverse)
        if nonzero and mask is not None:
            raise TransformError(
                f"'Standardize' was given nonzero=True and mask={mask!r}.",
                "The non-zero voxels are the mask: drop one of the two.",
            )
        if nonzero and inverse:
            raise TransformError(
                "'Standardize' with nonzero=True cannot be inverted: a voxel at 0 and one at the mean both land on 0.",
                "Set inverse: false.",
            )
        self.lazy = lazy
        self.mean = mean
        self.std = std
        self.mask = mask
        self.nonzero = nonzero
        self._masked_seed = _MaskedStatisticsSeed(mask) if mask is not None or nonzero else None

    def case_working_multiple(self, name: str) -> float:
        # The non-zero path holds, per channel, the mask, the gathered values and their centred copy.
        return 2.5 if self.nonzero else self.working_multiple

    def transform_shape(self, group_src: str, name: str, shape: list[int], cache_attribute: Attribute) -> list[int]:
        # Identity on the shape; a masked statistic records the group the masked disk scan needs.
        if self._masked_seed is not None:
            self._masked_seed.record_group(group_src)
        return shape

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        # Any of mean/std left unset is a global statistic: a seeded disk one, or under a mask (or over
        # the non-zero voxels) a disk scan the stage seeds itself once per case. With both given, POINTWISE.
        stat_keys: set[str] = set()
        if self.mean is None:
            stat_keys.add("Mean")
        if self.std is None:
            stat_keys.add("Std")
        if not stat_keys:
            return PatchLocality(LocalityKind.POINTWISE)
        if self._masked_seed is not None:
            return PatchLocality(LocalityKind.GLOBAL_STAT)
        return PatchLocality(LocalityKind.GLOBAL_STAT, stat_keys=frozenset(stat_keys), takes_present=True)

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        if self.nonzero:
            return self._nonzero(name, tensor, cache_attribute)
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

    def _nonzero(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        """Each channel over its non-zero voxels, the statistics in float32 (``torch.mean`` and the population
        ``torch.std`` of the gathered values) or, on a streamed region,
        seeded from the disk scan of the stored volume; what was applied is recorded per channel."""
        scanned = (
            self._masked_seed.statistics(self.datasets, name)  # type: ignore[union-attr]
            if (self.mean is None or self.std is None) and "StatisticsSeeded" in cache_attribute
            else None
        )
        out = tensor.to(torch.float32, copy=True)
        means: list[float] = []
        stds: list[float] = []
        for index, channel in enumerate(out):
            selected = channel != 0
            values = channel[selected]
            if scanned is not None:
                mean, std = float(scanned["mean_per_channel"][index]), float(scanned["std_per_channel"][index])
            elif values.numel():
                mean, std = float(torch.mean(values)), float(torch.std(values, unbiased=False)) or 1.0
            else:
                mean, std = 0.0, 1.0  # applied to no voxel: such a channel is left as it is
            mean, std = _stated(self.mean, index, mean), _stated(self.std, index, std)
            means.append(mean)
            stds.append(std)
            if not self.lazy and values.numel():
                channel[selected] = (values - mean).div_(std)
        cache_attribute["Mean"] = torch.tensor(means)
        cache_attribute["Std"] = torch.tensor(stds)
        return tensor if self.lazy else out

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


class Fill(Transform):
    """Set every voxel to ``value``, the case's grid and geometry kept, in float32: an image of ones (or zeros) of
    the case, for instance to resample into the share of another grid it covers (``Resample``'s ``fill`` marks the
    rest)."""

    working_multiple = 0.0
    locality = LocalityKind.POINTWISE

    def __init__(self, value: float = 1.0) -> None:
        super().__init__()
        self.value = float(value)

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        del name, cache_attribute
        return torch.full_like(tensor, self.value, dtype=torch.float32)


class TensorCast(TransformInverse):
    working_multiple = 1.0

    # Wide enough to hold every dtype a volume is read as (int8/int16/uint8/float32) with no value moved.
    _VALUE_PRESERVING_DTYPES = frozenset({torch.float32, torch.float64})

    def __init__(self, dtype: str = "float32", inverse: bool = True, round: bool = False) -> None:
        super().__init__(inverse)
        self.dtype: torch.dtype = TensorCast.safe_dtype_cast(dtype)
        #: Round to the nearest integer first, half to even (numpy's ``rint``), where a cast truncates.
        self.round = round

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        # The stored volume's statistics stay a later GLOBAL_STAT's input only where the cast keeps
        # every value: float32 holds an int16 exactly, float16 runs out of mantissa at 2048.
        return PatchLocality(
            LocalityKind.POINTWISE,
            preserves_statistics=self.dtype in TensorCast._VALUE_PRESERVING_DTYPES and not self.round,
        )

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        cache_attribute["dtype"] = str(tensor.dtype).replace("torch.", "")
        if self.round and tensor.is_floating_point():
            tensor = torch.round(tensor)
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

    alters_values = False

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


class CaseStatistic(Transform):
    """One float32 channel on the input's grid, every voxel the case's statistic the class ``statistic`` names (a
    classpath, ``model:NoiseSigma``). That class reads ``halo`` voxels past a region, edge-replicated past the volume's
    border, maps the float64 block to a 1-D ``partial`` of the region's own voxels, and turns the float64 sum of the
    partials into the 0-d statistic with ``finish``. Streamed, the stored volume is folded by bounded slabs first.
    Every other key of the stage's mapping is an argument of the class (``statistic: model:NoiseSigma, coils: 0``)."""

    # A float64 copy of the volume padded by the halo, and what the class's partial makes of it in float64 (a
    # Laplacian residual and its square hold three volumes more).
    working_multiple = 6.0

    def __init__(self, statistic: str) -> None:
        super().__init__()
        module, name = get_module(statistic, "konfai.data.transform")
        self.factory = module_attribute(module, name)
        if "statistic" in inspect.signature(self.factory).parameters:
            raise TransformError(
                f"'{statistic}' has a parameter 'statistic', which the CaseStatistic stage reads for itself.",
                "Rename it: every other key of the CaseStatistic mapping is the class's.",
            )
        self._statistic: Any = None
        self.key = f"Statistic:{statistic}"

    def prepare(self, konfai_args: str) -> None:
        # Bound here, not lazily: a strict read of the config counts a key only if something read it.
        self._statistic = apply_config(konfai_args)(self.factory)()

    @property
    def statistic(self) -> Any:
        """The bound class; a stage built in Python, never prepared, builds it from its defaults."""
        if self._statistic is None:
            self._statistic = self.factory()
        return self._statistic

    def output_channels(self, channels: int) -> int:
        return 1

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        return PatchLocality(
            LocalityKind.GLOBAL_STAT, stat_keys=frozenset({self.key}), records=frozenset(), measure=self._measure
        )

    def _fold(self, blocks: Iterable[Any]) -> float:
        partials = (self.statistic.partial(torch.as_tensor(block, dtype=torch.float64)) for block in blocks)
        return float(self.statistic.finish(sum(partials)))

    def _measure(self, dataset: Dataset, group: str, entry: str) -> float:
        return self._fold(dataset.iter_data_blocks(group, entry, int(self.statistic.halo))())

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        if "StatisticsSeeded" in cache_attribute and self.key in cache_attribute:
            value = float(cache_attribute[self.key])
        else:
            halo = int(self.statistic.halo)
            padded = torch.nn.functional.pad(tensor[None].double(), (halo, halo) * (tensor.dim() - 1), "replicate")
            value = self._fold([padded[0]])
            if MEASURING_STATISTICS in cache_attribute:
                cache_attribute[self.key] = value
        return torch.full((1, *tensor.shape[1:]), value, dtype=torch.float32, device=tensor.device)
