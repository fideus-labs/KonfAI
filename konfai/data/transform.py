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

import itertools
import os
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from multiprocessing import current_process, get_context
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None  # type: ignore[assignment]
import torch.nn.functional as F

from konfai import cuda_visible_devices
from konfai.utils.config import apply_config
from konfai.utils.dataset import Attribute, Dataset, data_to_image, image_to_data
from konfai.utils.errors import TransformError
from konfai.utils.ITK import _require_simpleitk, box_with_mask, crop_with_mask
from konfai.utils.runtime import NeedDevice
from konfai.utils.utils import get_module, split_path_spec


class LocalityKind(Enum):
    """How a transform's output at one voxel depends on its input (its patch-locality contract).

    A transform DECLARES its contract via :meth:`Transform.patch_locality`; the patch-streaming
    dispatcher (``konfai.data.patching``) reads the declaration and reads only the source region a
    target patch actually needs, instead of materialising the whole volume.

    - ``POINTWISE``   -- output voxel depends only on the same voxel (and its channels): read the
      exact patch.
    - ``HALO``        -- bounded neighbourhood: read the patch enlarged by ``halo`` per axis, crop after.
    - ``ORIENTATION`` -- flip/permute: read the index-remapped source region.
    - ``CROP``        -- the source region is the target region TRANSLATED: reading it IS the answer,
      so the stage is not re-applied to it. Unlike a reorientation this drops the voxels outside the
      box, so it is no bijection and the stored volume's statistics are not its output's.
    - ``GLOBAL_STAT`` -- needs whole-volume stats (``stat_keys`` subset of Min/Max/Mean/Std), obtained
      once from disk and cached: read the exact patch + the cached stat.
    - ``RESCALE``     -- resample: source region via the scale mapping + interpolation halo.
    - ``WHOLE_VOLUME``-- genuinely needs the whole volume: the dispatcher falls back to a full load.
    """

    POINTWISE = "pointwise"
    HALO = "halo"
    ORIENTATION = "orientation"
    CROP = "crop"
    GLOBAL_STAT = "global_stat"
    RESCALE = "rescale"
    WHOLE_VOLUME = "whole_volume"

    @property
    def preserves_statistics(self) -> bool:
        """
        Indicates whether the locality kind preserves whole-volume statistics.
        
        Returns:
            bool: `True` for orientation transforms, which reorder voxels without changing their values; `False` for other locality kinds.
        """
        return self is LocalityKind.ORIENTATION


@dataclass(frozen=True)
class PatchLocality:
    """A transform's declared patch-locality contract (see :class:`LocalityKind`).

    ``halo`` is the per-spatial-axis neighbourhood radius in array order (Z, Y, X); a length-1
    tuple broadcasts to every axis. ``stat_keys`` are the ``Attribute`` keys a ``GLOBAL_STAT``
    transform reads before running (a subset of ``Min``/``Max``/``Mean``/``Std``). ``stat_channels``
    restricts the statistic to those channels (``Normalize.channels``).
    """

    kind: LocalityKind
    halo: tuple[int, ...] = ()
    stat_keys: frozenset[str] = field(default_factory=frozenset)
    stat_channels: list[int] | None = None
    # Overrides the kind-level default (see LocalityKind.preserves_statistics): a POINTWISE transform
    # that maps no value (TensorCast to a float dtype) may declare True so a later GLOBAL_STAT can
    # still seed from the stored volume.
    preserves_statistics: bool | None = None

    @property
    def statistics_preserving(self) -> bool:
        """Indicates whether the locality contract preserves input statistics.
        
        Returns:
        	bool: `True` if statistics are preserved, `False` otherwise.
        """
        if self.preserves_statistics is not None:
            return self.preserves_statistics
        return self.kind.preserves_statistics


class Transform(NeedDevice, ABC):
    """Base class for transforms operating on tensors and cached attributes."""

    supports_dataloader_workers = True

    def __init__(self) -> None:
        NeedDevice.__init__(self)
        self.datasets: list[Dataset] = []

    def set_datasets(self, datasets: list[Dataset]):
        self.datasets = datasets

    def transform_shape(self, group_src: str, name: str, shape: list[int], cache_attribute: Attribute) -> list[int]:
        """
        Preserve the input spatial shape.
        
        Parameters:
            shape (list[int]): The input tensor shape.
        
        Returns:
            list[int]: The unchanged input shape.
        """
        return shape

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        """Declare how this transform's output depends on its input, for patch streaming.

        Answered from the transform's own ``__init__`` config and, where the honest answer depends on
        the image, from ``cache_attribute`` -- the case's SOURCE metadata, as the volume is stored.
        The dispatcher reads the header before any voxel, so a transform whose contract the image
        decides (a reorientation that is only a flip when the direction cosines are axis-aligned, a
        resample whose halo is the case's own scale) can still declare it up front.

        The default ``WHOLE_VOLUME`` is the safety net: any transform (including third-party custom
        ones) that does not override this falls to the whole-volume path, so nothing silently breaks.

        An override is bound by three rules:

        - **READ-ONLY.** Never write to ``cache_attribute``. A declaration is made once, for the whole
          case, and what it wrote would be one patch's answer imposed on every other -- the
          first-patch-wins bug the streamed paths are built to avoid. The dispatcher hands over a
          private copy, so a write cannot reach the case; it is simply lost.
        - **NO I/O.** Read the attribute already in hand, nothing else. Whether the outside world can
          honour the declaration (are the disk statistics readable, does a mask group exist) is the
          dispatcher's call, and it already makes it.
        - **TOTAL.** Answer for ANY case. The metadata may be absent -- the config-time checks probe
          with an empty ``Attribute``, and a group carries only what its writer stored -- so a missing
          key must return ``WHOLE_VOLUME``, never raise.
        """
        return PatchLocality(LocalityKind.WHOLE_VOLUME)

    def stream_region_source(
        self,
        target_slices: tuple[slice, ...],
        source_spatial_shape: list[int],
        cache_attribute: Attribute,
    ) -> list[slice]:
        """
        Map target-patch spatial slices to the source region required by the transform.
        
        Parameters:
        	target_slices (tuple[slice, ...]): Spatial slices of the target patch.
        	source_spatial_shape (list[int]): Spatial shape of the source tensor.
        	cache_attribute (Attribute): Source metadata used to determine the mapping.
        
        Returns:
        	list[slice]: Spatial slices identifying the source region to read.
        
        Raises:
        	TransformError: If the transform declares a region-based locality but does not provide a mapping.
        """
        raise TransformError(
            f"{type(self).__name__} declared a region patch-locality but does not implement stream_region_source().",
            "Implement stream_region_source() or declare a non-region patch_locality().",
        )

    def write_stream_cache_attribute(self, cache_attribute: Attribute, source_spatial_shape: list[int]) -> None:
        """
        Records any geometry metadata required for inverse processing of a streamed transform.
        
        Parameters:
            cache_attribute (Attribute): Persistent metadata for the transformed volume.
            source_spatial_shape (list[int]): Full spatial shape of the source volume.
        
        Returns:
            None
        """

    @abstractmethod
    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        """
        Apply the transform to a tensor.
        
        Parameters:
            name (str): Identifier of the tensor within its dataset.
            tensor (torch.Tensor): Input tensor to transform.
            cache_attribute (Attribute): Metadata and intermediate values shared with the transform chain.
        
        Returns:
            torch.Tensor: Transformed tensor.
        """
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
        """
        Configure intensity clipping bounds and optional bound caching.
        
        Parameters:
            min_value (float | str): Lower clipping bound or a data-dependent bound specification.
            max_value (float | str): Upper clipping bound or a data-dependent bound specification.
            save_clip_min (bool): Whether to store the resolved lower bound for later use.
            save_clip_max (bool): Whether to store the resolved upper bound for later use.
            mask (str | None): Optional dataset path used to determine data-dependent bounds.
        
        Raises:
            ValueError: If both bounds are numeric and the upper bound is less than or equal to the lower bound.
        """
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

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        # A mask reads a separate full volume, and a percentile bound needs the whole histogram:
        # both force a whole-volume load. A 'min'/'max' bound needs a global disk statistic
        # (GLOBAL_STAT); fixed float bounds clip each voxel independently (POINTWISE).
        """
        Classify the input region required to apply clipping.
        
        Returns:
            PatchLocality: Whole-volume locality when a mask or unsupported dynamic
            bound is used, global-statistic locality for minimum or maximum bounds,
            or pointwise locality for fixed bounds.
        """
        if self.mask is not None:
            return PatchLocality(LocalityKind.WHOLE_VOLUME)
        stat_keys: set[str] = set()
        for bound, key in ((self.min_value, "Min"), (self.max_value, "Max")):
            if isinstance(bound, str):
                if bound.lower() == key.lower():
                    stat_keys.add(key)
                else:
                    return PatchLocality(LocalityKind.WHOLE_VOLUME)
        if not stat_keys:
            return PatchLocality(LocalityKind.POINTWISE)
        return PatchLocality(LocalityKind.GLOBAL_STAT, stat_keys=frozenset(stat_keys))

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        """
        Clip tensor values to configured or data-derived lower and upper bounds.
        
        Parameters:
            name (str): Dataset item name used to locate the optional mask.
            tensor (torch.Tensor): Tensor whose values are clipped.
            cache_attribute (Attribute): Attributes where resolved bounds may be stored.
        
        Returns:
            torch.Tensor: The clipped input tensor.
        """
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
                    # ``np.percentile`` cannot coerce a CUDA tensor (finalize slots may hand Clip a
                    # GPU-resident volume); ``.cpu()`` is a no-op view on a host tensor.
                    min_value = np.percentile(tensor_masked.detach().cpu(), percentile)
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
                    max_value = np.percentile(tensor_masked.detach().cpu(), percentile)
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
        """
        Configure intensity normalization to a target range.
        
        Parameters:
            lazy (bool): Whether to store normalization statistics without applying the transform.
            channels (list[int] | None): Channel indices used for per-channel normalization.
            min_value (float): Lower bound of the target range.
            max_value (float): Upper bound of the target range.
            inverse (bool): Whether inverse transformation is enabled.
        
        Raises:
            ValueError: If `max_value` is less than or equal to `min_value`.
        """
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
        # Rescaling uses the volume-global Min/Max (restricted to self.channels); the dispatcher reads
        # those once from disk and seeds them so every patch (and inverse()) sees the same range.
        """
        Declare the global statistics required to normalize tensor patches.
        
        Parameters:
        	cache_attribute (Attribute): Cached transform metadata used to determine the applicable statistics.
        
        Returns:
        	PatchLocality: A global-statistics locality contract for the `Min` and `Max` values, optionally restricted to configured channels.
        """
        return PatchLocality(LocalityKind.GLOBAL_STAT, stat_keys=frozenset({"Min", "Max"}), stat_channels=self.channels)

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        """
        Normalize tensor values to a configured intensity range.
        
        Parameters:
        	name (str): Identifier used when reporting constant-input cases.
        	tensor (torch.Tensor): Tensor whose values are normalized.
        	cache_attribute (Attribute): Stores or provides the input minimum and maximum values.
        
        Returns:
        	torch.Tensor: The normalized tensor, or the original tensor when lazy normalization is enabled.
        """
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
        """
        Initialize intensity clipping bounds.
        
        Parameters:
            min_value (int): Lower clipping bound.
            max_value (int): Upper clipping bound.
        """
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        """Describe how the transform maps output patches to input regions.
        
        Parameters:
        	cache_attribute (Attribute): Cached metadata used to determine the locality contract.
        
        Returns:
        	PatchLocality: A pointwise locality contract.
        """
        return PatchLocality(LocalityKind.POINTWISE)

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        """Maps values from the interval [-1, 1] to the configured output interval.
        
        Parameters:
        	name (str): Dataset item name.
        	tensor (torch.Tensor): Input tensor.
        
        Returns:
        	torch.Tensor: Tensor with values linearly mapped to [min_value, max_value].
        """
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
        """
        Configure standardization using optional statistics, masking, and lazy evaluation.
        
        Parameters:
        	lazy (bool): Whether to defer applying standardization until inverse processing.
        	mean (list[float] | None): Optional mean values to use instead of computing them.
        	std (list[float] | None): Optional standard deviation values to use instead of computing them.
        	mask (str | None): Optional dataset path identifying the mask used to compute statistics.
        	inverse (bool): Whether inverse processing is enabled.
        """
        super().__init__(inverse)
        self.lazy = lazy
        self.mean = mean
        self.std = std
        self.mask = mask

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        # A mask reads a separate full volume (whole-volume). Any of mean/std left unset is taken from
        # a volume-global disk statistic (GLOBAL_STAT); when both are given, the standardization is a
        # per-voxel affine map with constant coefficients (POINTWISE).
        """
        Declare the patch-locality requirements for standardization.
        
        Parameters:
            cache_attribute (Attribute): Cached transform metadata used to determine available statistics.
        
        Returns:
            PatchLocality: Whole-volume locality when a mask is configured, global-statistics locality for unset mean or standard deviation values, or pointwise locality when both are configured.
        """
        if self.mask is not None:
            return PatchLocality(LocalityKind.WHOLE_VOLUME)
        stat_keys: set[str] = set()
        if self.mean is None:
            stat_keys.add("Mean")
        if self.std is None:
            stat_keys.add("Std")
        if not stat_keys:
            return PatchLocality(LocalityKind.POINTWISE)
        return PatchLocality(LocalityKind.GLOBAL_STAT, stat_keys=frozenset(stat_keys))

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        """
        Standardize tensor values using configured or computed mean and standard deviation.
        
        Parameters:
            name (str): Dataset item name used to locate the optional mask.
            tensor (torch.Tensor): Tensor to standardize.
            cache_attribute (Attribute): Attributes used to read or store the mean and standard deviation.
        
        Returns:
            torch.Tensor: Standardized tensor, or the original tensor when lazy mode is enabled.
        
        Raises:
            ValueError: If a configured mask cannot be found in any dataset.
        """
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
            mean = self._broadcast(cache_attribute.get_tensor("Mean").to(tensor.device), tensor)
            std = self._broadcast(cache_attribute.get_tensor("Std").to(tensor.device), tensor)
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
            # The stats parse back as float64 on the CPU; move them to the volume's device (the finalize
            # chain runs where the volume was blended, possibly CUDA) and compute in float32 so a
            # whole-volume fp16 output is not promoted to a float64 copy.
            mean = self._broadcast(cache_attribute.pop_tensor("Mean").to(tensor.device, torch.float32), tensor)
            std = self._broadcast(cache_attribute.pop_tensor("Std").to(tensor.device, torch.float32), tensor)
            return tensor * std + mean


class TensorCast(TransformInverse):
    def __init__(self, dtype: str = "float32", inverse: bool = True) -> None:
        """
        Initialize a transform that casts tensors to the specified PyTorch data type and optionally supports inverse casting.
        
        Parameters:
        	dtype (str): Name of the target PyTorch data type.
        	inverse (bool): Whether inverse casting is enabled.
        """
        super().__init__(inverse)
        self.dtype: torch.dtype = getattr(torch, dtype)

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        # A cast to a floating dtype keeps every value (up to fp rounding on a downcast), so the stored
        # volume's Min/Max/Mean/Std are still a later GLOBAL_STAT's input statistics; an integer cast
        # truncates and may not preserve them.
        """Describe the patch-locality and statistics-preservation behavior of the cast."""
        return PatchLocality(LocalityKind.POINTWISE, preserves_statistics=self.dtype.is_floating_point)

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        """
        Cast the tensor to the configured data type while recording its original data type.
        
        Parameters:
            cache_attribute (Attribute): Metadata storage used to preserve the original data type for inversion.
        
        Returns:
            torch.Tensor: The tensor converted to the configured data type.
        """
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
        """
        Configure the dimension to squeeze and whether inverse processing is enabled.
        
        Parameters:
        	dim (int): Dimension to squeeze.
        	inverse (bool): Whether inverse processing is enabled.
        """
        super().__init__(inverse)
        self.dim = dim

    def transform_shape(self, group_src: str, name: str, shape: list[int], cache_attribute: Attribute) -> list[int]:
        # ``shape`` is the channel-stripped spatial shape (patching strips [C, *spatial] before folding),
        # so the runtime tensor is [C, *shape] and ``self.dim`` indexes into that. Squeezing the channel
        # (axis 0) leaves the spatial grid untouched; squeezing a spatial axis drops it from the grid --
        # but only when it is size 1, exactly as ``torch.squeeze`` does (a non-singleton axis is a no-op).
        """
        Determine the spatial shape after squeezing a singleton tensor dimension.
        
        Parameters:
        	group_src (str): Source group associated with the tensor.
        	name (str): Tensor name.
        	shape (list[int]): Channel-stripped spatial shape.
        	cache_attribute (Attribute): Cached transform metadata.
        
        Returns:
        	list[int]: The spatial shape with the selected singleton axis removed, or the original shape when the axis is not singleton.
        """
        axis = self.dim if self.dim >= 0 else self.dim + len(shape) + 1
        if 1 <= axis <= len(shape) and shape[axis - 1] == 1:
            return shape[: axis - 1] + shape[axis:]
        return shape

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        return tensor.squeeze(self.dim)

    def inverse(self, name: str, tensor: torch.Tensor, cache_attribute: dict[str, Any]) -> torch.Tensor:
        return tensor.unsqueeze(self.dim)


class Resample(TransformInverse, ABC):
    def __init__(self, inverse: bool) -> None:
        super().__init__(inverse)

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        # The source region is derived from the scale mapping (read from cache_attribute['Spacing']
        # by the dispatcher); a small interpolation halo is added by resample_source_region.
        """
        Describe the input-region contract for resampling transforms.
        
        Parameters:
        	cache_attribute (Attribute): Cached transform metadata used to determine the scale mapping.
        
        Returns:
        	PatchLocality: A rescaling locality contract.
        """
        return PatchLocality(LocalityKind.RESCALE)

    def _resample(self, tensor: torch.Tensor, size: list[int]) -> torch.Tensor:
        """Resample a tensor to the specified spatial dimensions.
        
        Parameters:
            tensor (torch.Tensor): Tensor to resample.
            size (list[int]): Target spatial dimensions.
        
        Returns:
            torch.Tensor: Resampled tensor with the input dtype and device.
        """
        if tensor.dtype == torch.uint8:
            mode = "nearest"
        elif len(tensor.shape) < 4:
            mode = "bilinear"
        else:
            mode = "trilinear"

        # Interpolate in the tensor's own float dtype on CUDA. The model output is float16 and CUDA has
        # had Half kernels for every mode for years — upcasting the whole (channels x volume) tensor to
        # float32 doubled the memory of a multi-class output resample for no argmax benefit. On the CPU,
        # keep the historical float32 compute: Half CPU kernels are missing from older torch releases and
        # 1.5.8 always computed this path in float32. Integer inputs (uint8 labels) still need a float
        # grid for interpolation.
        if not tensor.is_floating_point() or (
            tensor.device.type == "cpu" and tensor.dtype in (torch.float16, torch.bfloat16)
        ):
            work = tensor.type(torch.float32)
        else:
            work = tensor
        # Return on the input's device (interpolate preserves it): a CPU input stays on the CPU, a
        # GPU-resident output volume stays on the GPU so the whole finalize runs where the volume is.
        return F.interpolate(work.unsqueeze(0), size=tuple(size), mode=mode).squeeze(0).type(tensor.dtype)

    @abstractmethod
    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        pass

    @abstractmethod
    def transform_shape(self, group_src: str, name: str, shape: list[int], cache_attribute: Attribute) -> list[int]:
        pass

    def inverse(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        """
        Restore the tensor to its previously recorded spatial shape.
        
        Parameters:
        	tensor (torch.Tensor): Tensor to resample.
        	cache_attribute (Attribute): Metadata containing the original size and optional spacing.
        
        Returns:
        	torch.Tensor: Tensor resampled to the recorded original spatial dimensions.
        """
        cache_attribute.pop_np_array("Size")
        size_1 = cache_attribute.pop_np_array("Size")
        if "Spacing" in cache_attribute:
            cache_attribute.pop_np_array("Spacing")
        return self._resample(tensor, [int(size) for size in size_1])

    # Every patch derives its source coordinates from the same global scale (n_in / n_out, from the
    # truncated integer sizes F.interpolate itself uses), which is what makes the streamed patches
    # agree with the whole-volume call and with each other across a seam.
    def _stream_mode(self, tensor: torch.Tensor) -> str:
        """
        Select the interpolation mode for resampling a tensor.
        
        Parameters:
        	tensor (torch.Tensor): Tensor whose dtype and dimensionality determine the interpolation mode.
        
        Returns:
        	str: `"nearest"` for unsigned 8-bit tensors, `"bilinear"` for tensors with fewer than four dimensions, or `"trilinear"` otherwise.
        """
        if tensor.dtype == torch.uint8:
            return "nearest"
        return "bilinear" if len(tensor.shape) < 4 else "trilinear"

    def resample_source_region(
        self,
        target_slices: tuple[slice, ...],
        source_spatial_shape: list[int],
        cache_attribute: Attribute,
        halo: int = 1,
    ) -> tuple[list[slice], list[int], list[float], list[int], list[int]]:
        """
        Map a target-grid patch to the source region required for resampling.
        
        Parameters:
            target_slices (tuple[slice, ...]): Target-grid patch slices in array axis order.
            source_spatial_shape (list[int]): Source spatial dimensions in array axis order.
            cache_attribute (Attribute): Cached transform metadata used to determine the output shape.
            halo (int): Additional source-region margin around the resampling footprint.
        
        Returns:
            tuple: Source slices, source-region starting indices, per-axis scale factors,
            source dimensions, and output dimensions, all in array axis order.
        """
        n_in = [int(s) for s in source_spatial_shape]
        n_out = [int(s) for s in self.transform_shape("", "", list(n_in), cache_attribute)]
        scales = [n_in[k] / n_out[k] for k in range(len(n_in))]
        source_slices: list[slice] = []
        region_starts: list[int] = []
        for k, sl in enumerate(target_slices):
            o0, o1 = sl.start, sl.stop
            smin = int(np.floor(scales[k] * (o0 + 0.5) - 0.5))
            smax = int(np.floor(scales[k] * ((o1 - 1) + 0.5) - 0.5))
            a = max(0, smin - halo)
            b = min(n_in[k], smax + 2 + halo)
            source_slices.append(slice(a, b))
            region_starts.append(a)
        return source_slices, region_starts, scales, n_in, n_out

    def resample_region(
        self,
        sub_tensor: torch.Tensor,
        target_slices: tuple[slice, ...],
        region_starts: list[int],
        scales: list[float],
        n_in: list[int],
    ) -> torch.Tensor:
        """
        Interpolate a source sub-region to the requested target patch.
        
        Parameters:
            sub_tensor (torch.Tensor): Source sub-region with channel-first layout.
            target_slices (tuple[slice, ...]): Target spatial slices to generate.
            region_starts (list[int]): Global source index of the first voxel in each axis.
            scales (list[float]): Source-to-target scale for each spatial axis.
            n_in (list[int]): Full source size along each spatial axis.
        
        Returns:
            torch.Tensor: Interpolated target patch.
        """
        mode = self._stream_mode(sub_tensor)
        dev = sub_tensor.device
        ndim = len(target_slices)
        if mode == "nearest":
            out = sub_tensor
            for k in range(ndim):
                # Take the axis's index map from F.interpolate itself, so streamed nearest picks the
                # same source voxel as the whole-volume call for every size ratio.
                src = torch.arange(n_in[k], device=dev, dtype=torch.float32).reshape(1, 1, -1)
                n_out_k = round(n_in[k] / scales[k])
                index = F.interpolate(src, size=n_out_k, mode="nearest").long().flatten()
                index = index[target_slices[k].start : target_slices[k].stop] - region_starts[k]
                out = out.index_select(k + 1, index)
            return out

        if not sub_tensor.is_floating_point() or (
            sub_tensor.device.type == "cpu" and sub_tensor.dtype in (torch.float16, torch.bfloat16)
        ):
            work = sub_tensor.type(torch.float32)
        else:
            work = sub_tensor
        taps: list[tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]] = []
        for k in range(ndim):
            o = torch.arange(target_slices[k].start, target_slices[k].stop, device=dev, dtype=work.dtype)
            src = torch.clamp(scales[k] * (o + 0.5) - 0.5, min=0.0)
            i0 = torch.floor(src).long()
            i1 = torch.clamp(i0 + 1, max=n_in[k] - 1)
            lam = src - i0.to(work.dtype)
            taps.append(((i0 - region_starts[k], 1 - lam), (i1 - region_starts[k], lam)))
        out_shape = [work.shape[0]] + [sl.stop - sl.start for sl in target_slices]
        out = torch.zeros(out_shape, device=dev, dtype=work.dtype)
        for combo in itertools.product(*taps):
            gathered = work
            weight = torch.ones([1] * (ndim + 1), device=dev, dtype=work.dtype)
            for k, (idx, lam) in enumerate(combo):
                gathered = gathered.index_select(k + 1, idx)
                shape = [1] * (ndim + 1)
                shape[k + 1] = -1
                weight = weight * lam.reshape(shape)
            out += gathered * weight
        return out.type(sub_tensor.dtype)

    @abstractmethod
    def write_stream_cache_attribute(self, cache_attribute: Attribute, source_spatial_shape: list[int]) -> None:
        """Store spacing and size metadata needed to invert streamed resampling.
        
        Parameters:
            cache_attribute (Attribute): Metadata updated with the resampling state.
            source_spatial_shape (list[int]): Full spatial shape before resampling.
        """


class ResampleToResolution(Resample):
    def __init__(self, spacing: list[float] = [1.0, 1.0, 1.0], inverse: bool = True) -> None:
        """Initialize a resampling transform with the target voxel spacing.
        
        Parameters:
        	spacing (list[float]): Target voxel spacing for each spatial axis. Negative values are treated as zero.
        	inverse (bool): Whether inverse transformation support is enabled.
        """
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
        """
        Resample a tensor to the configured voxel spacing.
        
        Parameters:
        	name (str): Dataset item name associated with the tensor.
        	tensor (torch.Tensor): Tensor whose spatial dimensions are resampled.
        	cache_attribute (Attribute): Metadata updated with the resulting spacing and size.
        
        Returns:
        	torch.Tensor: The resampled tensor.
        """
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

    def write_stream_cache_attribute(self, cache_attribute: Attribute, source_spatial_shape: list[int]) -> None:
        """
        Update cached spacing and original size metadata for streamed resampling.
        
        Parameters:
        	cache_attribute (Attribute): Metadata cache to update.
        	source_spatial_shape (list[int]): Spatial shape of the untransformed source volume.
        """
        image_spacing = cache_attribute.get_tensor("Spacing")
        spacing = self.spacing
        resize_factor = torch.tensor(
            [s / i_s if s > 0 else 1.0 for s, i_s in zip(self.spacing, image_spacing, strict=False)]
        )
        cache_attribute["Spacing"] = torch.tensor(
            [float(s) if s > 0 else float(i_s) for s, i_s in zip(spacing, image_spacing, strict=False)]
        )
        cache_attribute["Size"] = np.asarray([int(x) for x in source_spatial_shape])
        size = [int(x) for x in (torch.tensor([int(s) for s in source_spatial_shape]) * 1 / resize_factor.flip(0))]
        cache_attribute["Size"] = np.asarray(size)


class ResampleToShape(Resample):
    def __init__(self, shape: list[float] = [100, 256, 256], inverse: bool = True) -> None:
        """
        Configure resampling to a target spatial shape.
        
        Parameters:
        	shape (list[float]): Target spatial dimensions; values less than zero are treated as zero, which preserves the corresponding input dimension.
        	inverse (bool): Whether inverse transformation support is enabled.
        """
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
        """
        Resample the tensor to the configured spatial shape.
        
        Entries configured as zero retain their corresponding input spatial dimension.
        Updates cached spacing and size metadata when applicable.
        
        Parameters:
            tensor (torch.Tensor): Tensor to resample.
            cache_attribute (Attribute): Metadata updated with the resulting size and spacing.
        
        Returns:
            torch.Tensor: Tensor resampled to the target spatial shape.
        """
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

    def write_stream_cache_attribute(self, cache_attribute: Attribute, source_spatial_shape: list[int]) -> None:
        """Update cached spatial metadata for streaming resampling to the configured target shape.
        
        Parameters:
        	cache_attribute (Attribute): Metadata updated with the resampled spacing and size.
        	source_spatial_shape (list[int]): Spatial dimensions of the original image.
        """
        shape = self.shape.clone()
        image_shape = torch.tensor([int(s) for s in source_spatial_shape])
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


class ResampleTransform(TransformInverse):
    """Resample a volume through stored transforms (a displacement field, an affine).

    Whole-volume: nothing in the format bounds the stored displacement, so no halo can be declared
    from the header alone.
    """

    def __init__(self, transforms: dict[str, bool], inverse: bool = True) -> None:
        """Initialize the transform inverse with its configured transform settings.
        
        Parameters:
        	transforms (dict[str, bool]): Transform names mapped to their enabled states.
        	inverse (bool): Whether inverse processing is enabled.
        """
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

        _require_simpleitk()
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
        raise NotImplementedError(
            "ResampleTransform.inverse is not implemented; set `inverse: false` on this transform "
            "(it defaults to true)."
        )


class Mask(Transform):
    """Set everything outside a mask to a constant.

    Whole-volume: ``__call__`` does not know where its tensor sits, so it cannot read the matching
    region of the mask (``Clip(mask=)`` and ``Standardize(mask=)`` load whole volumes for the same
    reason).
    """

    def __init__(self, path: str = "./default.mha", value_outside: int = 0) -> None:
        """Initialize a transform that sets values outside a mask to a specified constant.
        
        Parameters:
        	path (str): Path to the mask image or dataset attribute.
        	value_outside (int): Value assigned to tensor elements outside the mask.
        """
        super().__init__()
        self.path = path
        self.value_outside = value_outside
        self._cached_mask: torch.Tensor | None = None

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        if self.path.endswith(".mha"):
            _require_simpleitk()
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
        # Index on the tensor's own device so the mask works whether the volume is on CPU or GPU
        # (``torch.as_tensor`` keeps a torch mask as-is and wraps a numpy one, moving it to the device).
        tensor[torch.as_tensor(mask, device=tensor.device) == 0] = self.value_outside
        return tensor


class Dilate(Transform):
    def __init__(self, dilate: int = 1) -> None:
        """Initialize the dilation transform.
        
        Parameters:
        	dilate (int): Number of voxels by which to expand the mask in each spatial direction. Must be greater than or equal to zero.
        
        Raises:
        	ValueError: If `dilate` is negative.
        """
        super().__init__()
        if dilate < 0:
            raise ValueError(f"[Dilate] 'dilate' must be >= 0, got {dilate}")
        self.dilate = dilate

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        # A box dilation of radius ``dilate`` spreads foreground by at most ``dilate`` voxels per axis:
        # a bounded HALO. At the true border the separable max-pool padding matches the whole-volume
        # result once the halo clamps, so seams are byte-identical. Radius 0 is a spatial identity.
        if self.dilate == 0:
            return PatchLocality(LocalityKind.POINTWISE)
        return PatchLocality(LocalityKind.HALO, halo=(self.dilate,))

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        """
        Expand positive regions in a 2D or 3D tensor by the configured dilation radius.
        
        Parameters:
        	name (str): Tensor identifier used in unsupported-shape error messages.
        	tensor (torch.Tensor): Channel-first 2D or 3D tensor to dilate.
        	cache_attribute (Attribute): Transform metadata cache.
        
        Returns:
        	torch.Tensor: Dilated tensor with the original data type.
        
        Raises:
        	ValueError: If the tensor is neither channel-first 2D nor channel-first 3D.
        """
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
        """
        Initialize the dimension used by the transform.
        
        Parameters:
        	dim (int): Dimension along which the transform operates.
        """
        super().__init__()
        self.dim = dim

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        # Pointwise only when reducing the leading channel/model axis (dim 0); a spatial sum spans
        # the whole extent, so it falls back to the whole volume.
        """
        Classifies the locality of the sum operation based on its reduction dimension.
        
        Parameters:
        	cache_attribute (Attribute): Cached transform metadata.
        
        Returns:
        	PatchLocality: Pointwise locality when reducing the leading channel or model axis; whole-volume locality otherwise.
        """
        if self.dim == 0:
            return PatchLocality(LocalityKind.POINTWISE)
        return PatchLocality(LocalityKind.WHOLE_VOLUME)

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        """
        Sum tensor values along the configured dimension, accounting for per-model label offsets when provided.
        
        Parameters:
            tensor (torch.Tensor): Tensor to aggregate.
            cache_attribute (Attribute): Cached transform metadata containing optional per-model channel counts.
        
        Returns:
            torch.Tensor: Aggregated tensor with the input dtype.
        """
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

    Models are assumed to segment disjoint structures, but boundaries disagree in practice: a voxel
    claimed by several models takes the label of the LAST model in ensemble order (adding the global
    ids instead would fabricate a label belonging to neither model).
    """

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        # Merges the leading model axis per voxel; spatial support is a single voxel.
        return PatchLocality(LocalityKind.POINTWISE)

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        """
        Merge per-model segmentation labels into a single global label map.
        
        Parameters:
        	name (str): Identifier for the transformed data.
        	tensor (torch.Tensor): Per-model label maps to merge.
        	cache_attribute (Attribute): Metadata containing the channel count for each model.
        
        Returns:
        	torch.Tensor: A label map with foreground labels shifted into a shared label space.
        """
        if "number_of_channels_per_model" not in cache_attribute:
            raise TransformError(
                "MergeLabels expects a multi-model 'combine: Concat' output: "
                "'number_of_channels_per_model' is missing from the attribute.",
            )
        number_of_channels = cache_attribute.pop_tensor("number_of_channels_per_model")
        result = tensor[0].clone()
        offset = int(number_of_channels[0]) - 1
        for i, t in enumerate(tensor[1:]):
            foreground = t != 0
            result[foreground] = (t[foreground] + offset).to(result.dtype)
            offset += int(number_of_channels[i + 1]) - 1
        return result


class Gradient(Transform):
    def __init__(self, per_dim: bool = False):
        """Initialize gradient computation with optional per-dimension output.
        
        Parameters:
        	per_dim (bool): Whether to retain separate gradient components for each spatial dimension.
        """
        super().__init__()
        self.per_dim = per_dim

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        # First-difference gradient: each output voxel reads its immediate neighbour, a HALO of radius
        # 1. The far-edge ConstantPad reproduces the whole-volume border once the halo clamps there.
        """
        Declare that each output voxel requires an immediate neighboring input voxel.
        
        Returns:
            PatchLocality: A halo locality contract with radius one.
        """
        return PatchLocality(LocalityKind.HALO, halo=(1,))

    @staticmethod
    def _image_gradient_2d(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute first differences along the two spatial dimensions of a 2D image.
        
        Parameters:
        	image (torch.Tensor): A channel-first 2D image tensor.
        
        Returns:
        	tuple[torch.Tensor, torch.Tensor]: The differences along the height and width dimensions, padded to the input spatial shape.
        """
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
        """
        Initialize the dimension used by the transform.
        
        Parameters:
        	dim (int): Dimension along which the transform operates.
        """
        super().__init__()
        self.dim = dim

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        # Pointwise ONLY when reducing the channel axis (dim 0). Over a spatial axis the argmax spans
        # the whole extent, so a per-patch argmax would diverge -- fall back to the whole volume.
        """Declare the patch-locality contract for the argmax operation.
        
        Parameters:
        	cache_attribute (Attribute): Cached transform metadata.
        
        Returns:
        	PatchLocality: Pointwise locality when reducing the channel axis; whole-volume locality otherwise.
        """
        if self.dim == 0:
            return PatchLocality(LocalityKind.POINTWISE)
        return PatchLocality(LocalityKind.WHOLE_VOLUME)

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        """
        Selects the index of the maximum value along the configured dimension while preserving that dimension.
        
        Returns:
        	torch.Tensor: Tensor containing the maximum-value indices with the selected dimension retained.
        """
        return torch.argmax(tensor, dim=self.dim).unsqueeze(self.dim)


class Softmax(Transform):
    def __init__(self, dim: int = 0) -> None:
        """
        Initialize the dimension used by the transform.
        
        Parameters:
        	dim (int): Dimension along which the transform operates.
        """
        super().__init__()
        self.dim = dim

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        # Pointwise ONLY when reducing the channel axis (dim 0). Over a spatial axis softmax normalises
        # across the whole extent, so a per-patch softmax would diverge -- fall back to the whole volume.
        """
        Classifies the softmax operation's patch-locality requirements.
        
        Parameters:
        	cache_attribute (Attribute): Cached transform metadata.
        
        Returns:
        	PatchLocality: Pointwise locality when softmax reduces the channel axis; whole-volume locality otherwise.
        """
        if self.dim == 0:
            return PatchLocality(LocalityKind.POINTWISE)
        return PatchLocality(LocalityKind.WHOLE_VOLUME)

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        return torch.softmax(tensor, dim=self.dim)


class FlatLabel(Transform):
    def __init__(self, labels: list[int] | None = None) -> None:
        super().__init__()
        self.labels = labels

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        """Describe how the transform maps output patches to input regions.
        
        Parameters:
        	cache_attribute (Attribute): Cached metadata used to determine the locality contract.
        
        Returns:
        	PatchLocality: A pointwise locality contract.
        """
        return PatchLocality(LocalityKind.POINTWISE)

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        """
        Convert selected label values into a binary mask.
        
        Parameters:
        	name (str): Sample name associated with the tensor.
        	tensor (torch.Tensor): Tensor containing scalar labels.
        	cache_attribute (Attribute): Metadata associated with the transformation.
        
        Returns:
        	torch.Tensor: A tensor containing `1` for selected labels and `0` elsewhere.
        """
        data = torch.zeros_like(tensor)
        if self.labels:
            for label in self.labels:
                data[torch.where(tensor == label)] = 1
        else:
            data[torch.where(tensor > 0)] = 1
        return data


class Save(Transform):
    def __init__(self, dataset: str, group: str | None = None) -> None:
        """Initialize the transform with a dataset path and optional data group.
        
        Parameters:
        	dataset (str): Dataset path or identifier.
        	group (str | None): Optional group containing the relevant data.
        """
        super().__init__()
        self.dataset = dataset
        self.group = group

    # WHOLE_VOLUME on purpose: a Save still in the chain must WRITE the preprocessed volume, and the
    # streamed path never has one to write. (A Save whose cache exists is a source boundary instead.)

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
        """
        Determine the spatial shape after applying the configured permutation.
        
        Parameters:
            shape (list[int]): Input shape including the channel dimension.
        
        Returns:
            list[int]: Spatial shape reordered according to the configured dimensions.
        """
        return [shape[it - 1] for it in self.dims[1:]]

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        """
        Declare that the transform performs an orientation-only spatial mapping.
        
        Parameters:
        	cache_attribute (Attribute): Cached transform metadata.
        
        Returns:
        	PatchLocality: An orientation locality contract.
        """
        return PatchLocality(LocalityKind.ORIENTATION)

    def stream_region_source(
        self,
        target_slices: tuple[slice, ...],
        source_spatial_shape: list[int],
        cache_attribute: Attribute,
    ) -> list[slice]:
        # Output spatial axis k comes from input axis ``self.dims[k + 1] - 1`` (self.dims is
        # channel-inclusive). Placing each target slice at its source axis yields the source region
        # whose permutation reproduces the target patch exactly.
        """
        Map a target patch to the source region required by the permutation.
        
        Parameters:
        	target_slices (tuple[slice, ...]): Spatial slices defining the target patch.
        	source_spatial_shape (list[int]): Spatial dimensions of the unpermuted source tensor.
        	cache_attribute (Attribute): Transform metadata.
        
        Returns:
        	list[slice]: Source spatial slices whose permutation produces the target patch.
        """
        source_slices = [slice(0, n) for n in source_spatial_shape]
        for k, sl in enumerate(target_slices):
            source_slices[self.dims[k + 1] - 1] = slice(sl.start, sl.stop)
        return source_slices

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        """
        Reorders the tensor dimensions according to the configured permutation.
        
        Parameters:
        	name (str): The data item name.
        	tensor (torch.Tensor): The tensor to reorder.
        	cache_attribute (Attribute): Metadata associated with the tensor.
        
        Returns:
        	torch.Tensor: The tensor with permuted dimensions.
        """
        return tensor.permute(tuple(self.dims))

    def inverse(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        return tensor.permute(tuple(np.argsort(self.dims)))


class Flip(TransformInverse):
    def __init__(self, dims: str = "1|0|2", inverse: bool = True) -> None:
        """
        Initialize a spatial-axis flip transform.
        
        Parameters:
        	dims (str): Pipe-separated spatial axis indices to flip, such as ``"1|0|2"``.
        	inverse (bool): Whether inverse transformation support is enabled.
        """
        super().__init__(inverse)

        self.dims = [int(d) + 1 for d in str(dims).split("|")]

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        """
        Declare that the transform performs an orientation-only spatial mapping.
        
        Parameters:
        	cache_attribute (Attribute): Cached transform metadata.
        
        Returns:
        	PatchLocality: An orientation locality contract.
        """
        return PatchLocality(LocalityKind.ORIENTATION)

    def stream_region_source(
        self,
        target_slices: tuple[slice, ...],
        source_spatial_shape: list[int],
        cache_attribute: Attribute,
    ) -> list[slice]:
        # A flipped spatial axis reads the mirror region ``[n - stop, n - start)``; applying the flip
        # to that sub-region reproduces the target patch. Non-flipped axes read the identity region.
        """
        Map a target patch to the source region required by the configured spatial flips.
        
        Parameters:
        	target_slices (tuple[slice, ...]): Spatial slices defining the target patch.
        	source_spatial_shape (list[int]): Spatial dimensions of the unflipped source tensor.
        	cache_attribute (Attribute): Transform metadata.
        
        Returns:
        	list[slice]: Source spatial slices corresponding to the target patch.
        """
        source_slices: list[slice] = []
        for k, sl in enumerate(target_slices):
            n = source_spatial_shape[k]
            if (k + 1) in self.dims:
                source_slices.append(slice(n - sl.stop, n - sl.start))
            else:
                source_slices.append(slice(sl.start, sl.stop))
        return source_slices

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        """Flip the tensor along the configured spatial dimensions.
        
        Parameters:
        	name (str): Dataset item name.
        	tensor (torch.Tensor): Tensor to flip.
        	cache_attribute (Attribute): Cached transform metadata.
        
        Returns:
        	torch.Tensor: Tensor flipped along the configured dimensions.
        """
        return tensor.flip(tuple(self.dims))

    def inverse(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        return tensor.flip(tuple(self.dims))


class Canonical(TransformInverse):
    """Reorient a volume onto the canonical (LPS) direction cosines.

    An orthogonal reorientation is a signed permutation of the axes: an exact index remap (values only
    change place, so whole-volume statistics survive); only an oblique direction is resampled. A remap
    that permutes axes transposes the extents it swaps, so ``transform_shape`` folds the patch grid
    onto the reoriented shape.
    """

    # An orthonormal direction's entries are exactly 0 or +/-1 when it is axis-aligned, but the
    # reorientation is a product with an inverse, so it lands within a few double ulps of them.
    _AXIS_ALIGNED_ATOL = 1e-9

    def __init__(self, inverse: bool = True) -> None:
        """
        Initialize the canonical reorientation transform.
        
        Parameters:
            inverse (bool): Whether to apply the inverse operation.
        """
        super().__init__(inverse)
        self.canonical_direction = torch.diag(torch.tensor([-1, -1, 1])).to(torch.double)

    def _reorientation(self, cache_attribute: Attribute) -> torch.Tensor:
        """The map taking an output coordinate onto the input it comes from, in (x, y, z).

        A voxel sits at ``D @ (spacing * index) + origin``, so the map is ``D^-1 @ C`` (with the
        target spacing carried along the permutation, see ``_carried``) -- NOT the rotation
        ``C @ D^-1``, which only agrees where the two commute.
        """
        initial_matrix = cache_attribute.get_tensor("Direction").reshape(3, 3).to(torch.double)
        return initial_matrix.inverse() @ self.canonical_direction

    @classmethod
    def _index_remap(cls, reorientation: torch.Tensor) -> list[tuple[int, bool]] | None:
        """
        Determine the source axis and orientation for an axis-aligned reorientation.
        
        Parameters:
        	reorientation (torch.Tensor): Coordinate mapping from output axes to input axes.
        
        Returns:
        	list[tuple[int, bool]] | None: A mapping in array-axis order, where each tuple contains
        	the source axis index and whether it is reversed; `None` if the mapping mixes axes.
        """
        n = reorientation.shape[0]
        unit = torch.ones(n, dtype=reorientation.dtype)
        columns = reorientation.abs()
        if not torch.allclose(columns.sum(0), unit, atol=cls._AXIS_ALIGNED_ATOL):
            return None
        if not torch.allclose(columns.amax(0), unit, atol=cls._AXIS_ALIGNED_ATOL):
            return None
        remap = []
        for c in reversed(range(n)):
            r = int(columns[:, c].argmax())
            remap.append((n - 1 - r, bool(reorientation[r, c] < 0)))
        return remap

    def _orthogonal_remap(self, cache_attribute: Attribute) -> list[tuple[int, bool]] | None:
        """
        Determine whether the cached direction represents an exact axis permutation and reflection.
        
        Parameters:
            cache_attribute (Attribute): Cached image metadata containing the direction cosines.
        
        Returns:
            list[tuple[int, bool]] | None: Axis remapping entries as source-axis and reflection pairs, or
            `None` when direction metadata is unavailable, has an invalid size, or represents an oblique
            orientation.
        """
        if "Direction" not in cache_attribute or cache_attribute.get_np_array("Direction").size != 9:
            return None
        return Canonical._index_remap(self._reorientation(cache_attribute))

    @staticmethod
    def _carried(per_physical_axis: torch.Tensor, remap: list[tuple[int, bool]] | None) -> torch.Tensor:
        """
        Reorders a per-physical-axis quantity according to an array-axis remapping.
        
        Parameters:
            per_physical_axis (torch.Tensor): Values ordered by physical axis.
            remap (list[tuple[int, bool]] | None): Array-axis mapping to source axes, or None when no remapping applies.
        
        Returns:
            torch.Tensor: The quantity reordered for the remapped axes.
        """
        if remap is None:
            return per_physical_axis
        # The remap is in array order and these are (x, y, z): read in array order, gather, restore.
        return per_physical_axis.flip(0)[[source for source, _ in remap]].flip(0)

    @staticmethod
    def _half_extent(spatial_shape: list[int], spacing: torch.Tensor) -> torch.Tensor:
        """Half a grid's physical extent along each axis, in (x, y, z). A shape is in array order."""
        return torch.tensor(
            [(spatial_shape[-axis - 1] - 1) * spacing[axis] / 2 for axis in range(len(spatial_shape))],
            dtype=torch.double,
        )

    @staticmethod
    def _affine_matrix(matrix: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
        """
        Construct a homogeneous affine transformation matrix.
        
        Parameters:
        	matrix (torch.Tensor): The linear transformation matrix.
        	translation (torch.Tensor): The translation vector.
        
        Returns:
        	torch.Tensor: A four-by-four affine matrix containing the linear transformation and translation.
        """
        return torch.cat(
            (
                torch.cat((matrix, translation.unsqueeze(0).T), dim=1),
                torch.tensor([[0, 0, 0, 1]]),
            ),
            dim=0,
        )

    @staticmethod
    def _resample_affine(data: torch.Tensor, matrix: torch.Tensor):
        """
        Resamples tensor data using an affine transformation matrix.
        
        Parameters:
            data (torch.Tensor): Tensor to resample.
            matrix (torch.Tensor): Homogeneous affine transformation matrix.
        
        Returns:
            torch.Tensor: Affinely resampled tensor with the same dtype as `data`.
        """
        if data.dtype == torch.uint8:
            mode = "nearest"
        else:
            mode = "bilinear"
        # Sample in the data's own device and float dtype: the model output is float16 on the GPU, and
        # affine_grid/grid_sample support float16 on CPU and CUDA. Building the grid on the data's device
        # (instead of a CPU float32 grid) keeps the whole reorientation on-device — no host round-trip and
        # no float32 upcast of the (channels x volume) tensor. Integer inputs still need a float grid.
        # Accepted trade-off: an fp16 grid quantizes the sampling coordinates (up to ~0.1 voxel at 512^3),
        # chosen over the ~2x transient memory of a float32 grid + volume upcast.
        work = data if data.is_floating_point() else data.type(torch.float32)
        grid = torch.nn.functional.affine_grid(
            matrix[:, :-1, ...].to(device=work.device, dtype=work.dtype),
            [1, *list(data.shape)],
            align_corners=True,
        )
        return (
            torch.nn.functional.grid_sample(
                work.unsqueeze(0),
                grid,
                align_corners=True,
                mode=mode,
                padding_mode="reflection",
            )
            .squeeze(0)
            .type(data.dtype)
        )

    def transform_shape(self, group_src: str, name: str, shape: list[int], cache_attribute: Attribute) -> list[int]:
        # ``shape`` is the channel-stripped SPATIAL shape, and the patch grid is folded from what this
        # returns: a remap that transposes extents moves the grid onto the reoriented volume.
        """
        Determine the spatial shape after canonical reorientation.
        
        Parameters:
            shape (list[int]): Channel-stripped spatial dimensions.
            cache_attribute (Attribute): Metadata used to determine the input orientation.
        
        Returns:
            list[int]: The spatial dimensions reordered for an orthogonal reorientation, or the original shape when no such remap applies.
        """
        remap = self._orthogonal_remap(cache_attribute)
        if remap is None:
            return shape
        return [shape[source] for source, _ in remap]

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        # Only the case can say which reorientation this is, so only the header can answer. An orthogonal
        # one -- mirroring or permuting -- remaps indices, which is what ORIENTATION streams; an oblique
        # one is resampled from the whole volume.
        """
        Determine the patch-locality contract for the current reorientation.
        
        Parameters:
        	cache_attribute (Attribute): Cached image geometry used to determine whether the reorientation is orthogonal.
        
        Returns:
        	PatchLocality: An orientation contract for axis permutations or flips, otherwise a whole-volume contract.
        """
        if self._orthogonal_remap(cache_attribute) is None:
            return PatchLocality(LocalityKind.WHOLE_VOLUME)
        return PatchLocality(LocalityKind.ORIENTATION)

    def stream_region_source(
        self,
        target_slices: tuple[slice, ...],
        source_spatial_shape: list[int],
        cache_attribute: Attribute,
    ) -> list[slice]:
        # Target axis k reads source axis ``source``, so the target slice IS the source's -- taken at the
        # far end ``[n - stop, n - start)`` where the remap reads that axis backwards. Flipping the region
        # read reproduces the patch: a flip restricted to a contiguous region is that region reversed.
        # Both the slices and the remap are in array order, and the remap covers every axis exactly once.
        """
        Map target patch slices to the source regions required for an exact canonical reorientation.
        
        Parameters:
            target_slices (tuple[slice, ...]): Spatial slices of the requested target patch.
            source_spatial_shape (list[int]): Spatial dimensions of the source tensor.
            cache_attribute (Attribute): Cached geometry metadata used to determine the axis remapping.
        
        Returns:
            list[slice]: Source-axis slices corresponding to the target patch, including reversed slices for mirrored axes.
        
        Raises:
            TransformError: If the direction cannot be represented as an exact orthogonal axis remapping.
        """
        remap = self._orthogonal_remap(cache_attribute)
        if remap is None:
            raise TransformError(
                "Canonical declared a region patch-locality for a direction it cannot remap exactly.",
                "Report this: patch_locality() and stream_region_source() disagree about the case.",
            )
        source_slices = [slice(None)] * len(remap)
        for target, (source, mirrored) in zip(target_slices, remap, strict=False):
            extent = source_spatial_shape[source]
            source_slices[source] = (
                slice(extent - target.stop, extent - target.start) if mirrored else slice(target.start, target.stop)
            )
        return source_slices

    def write_stream_cache_attribute(self, cache_attribute: Attribute, source_spatial_shape: list[int]) -> None:
        """
        Update cached geometry metadata for a canonical reorientation using the full source volume shape.
        
        Parameters:
        	cache_attribute (Attribute): Geometry metadata to update.
        	source_spatial_shape (list[int]): Full source volume spatial dimensions used to preserve the volume center.
        """
        initial_matrix = cache_attribute.get_tensor("Direction").reshape(3, 3).to(torch.double)
        initial_origin = cache_attribute.get_tensor("Origin")
        spacing = cache_attribute.get_tensor("Spacing").to(torch.double)
        remap = self._orthogonal_remap(cache_attribute)
        half_extent = Canonical._half_extent(source_spatial_shape, spacing)
        cache_attribute["Direction"] = self.canonical_direction.flatten()
        cache_attribute["Spacing"] = Canonical._carried(spacing, remap)
        # The reorientation fixes the volume's centre, so the new origin is that centre stepped back by
        # the canonical half-extent -- the TARGET grid's, which a permutation has carried onto other
        # axes. The extent is the VOLUME's, never a patch's: it is an argument rather than the handed
        # tensor's shape.
        center = initial_matrix @ half_extent + initial_origin
        cache_attribute["Origin"] = center - self.canonical_direction @ Canonical._carried(half_extent, remap)

    def _reorient(self, tensor: torch.Tensor, reorientation: torch.Tensor) -> torch.Tensor:
        """
        Reorients a tensor using exact axis remapping when possible and affine resampling otherwise.
        
        Parameters:
            tensor (torch.Tensor): Channel-first tensor to reorient.
            reorientation (torch.Tensor): Spatial reorientation matrix.
        
        Returns:
            torch.Tensor: The reoriented tensor.
        """
        remap = Canonical._index_remap(reorientation)
        if remap is None:
            matrix = Canonical._affine_matrix(reorientation, torch.tensor([0, 0, 0]))
            return Canonical._resample_affine(tensor, matrix.unsqueeze(0))
        # The remap is spatial and the tensor is channel-first, so the channel axes lead it unpermuted.
        offset = tensor.dim() - len(remap)
        dims = list(range(offset)) + [offset + source for source, _ in remap]
        flips = [offset + axis for axis, (_, mirrored) in enumerate(remap) if mirrored]
        # flip materialises the permuted view, so the result never aliases the tensor it was read from.
        return tensor.permute(dims).flip(flips)

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        # Read the source geometry before recording the canonical one over it: the attribute stacks.
        """
        Reorients a tensor to the canonical coordinate system and records the geometry needed for inverse transformation.
        
        Parameters:
        	name (str): Dataset item name.
        	tensor (torch.Tensor): Tensor to reorient.
        	cache_attribute (Attribute): Geometry metadata updated with the canonical orientation.
        
        Returns:
        	torch.Tensor: The reoriented tensor.
        """
        reorientation = self._reorientation(cache_attribute)
        self.write_stream_cache_attribute(cache_attribute, list(tensor.shape[1:]))
        return self._reorient(tensor, reorientation)

    def inverse(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        # Popping restores the source geometry, which is what the inverse remap is then read from.
        """
        Restore the tensor's original orientation and geometry.
        
        Parameters:
        	cache_attribute (Attribute): Cached source geometry used to determine the inverse reorientation.
        
        Returns:
        	torch.Tensor: The tensor reoriented to its original coordinate system.
        """
        cache_attribute.pop("Direction")
        cache_attribute.pop("Spacing")
        cache_attribute.pop("Origin")
        return self._reorient(tensor, self._reorientation(cache_attribute).inverse())


class HistogramMatching(Transform):
    """Match a volume's intensity distribution onto a reference group's.

    Whole-volume: the LUT is built from the volume's 256-bin histogram, which is not a statistic
    ``GLOBAL_STAT`` names and cannot be read back out of the sitk filter.
    """

    def __init__(self, reference_group: str) -> None:
        """Initialize histogram matching with the dataset group containing reference images.
        
        Parameters:
            reference_group (str): Dataset group used to locate reference images.
        """
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
        _require_simpleitk()
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

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        """Describe how the transform maps output patches to input regions.
        
        Parameters:
        	cache_attribute (Attribute): Cached metadata used to determine the locality contract.
        
        Returns:
        	PatchLocality: A pointwise locality contract.
        """
        return PatchLocality(LocalityKind.POINTWISE)

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        """
        Remap configured label identifiers to their corresponding output values.
        
        Parameters:
            name (str): Dataset item name.
            tensor (torch.Tensor): Tensor containing scalar label identifiers.
            cache_attribute (Attribute): Transformation metadata cache.
        
        Returns:
            torch.Tensor: Tensor containing the remapped labels, with unmatched positions set to zero.
        """
        data = torch.zeros_like(tensor)
        for old_label, new_label in self.labels:
            data[tensor == int(old_label)] = int(new_label)
        return data


class OneHot(TransformInverse):
    def __init__(self, num_classes: int, inverse: bool = True) -> None:
        super().__init__(inverse)
        self.num_classes = num_classes

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        # Expands each voxel's scalar label into a one-hot channel vector (spatially pointwise).
        """Declares that the transform operates independently at each spatial position.
        
        Returns:
            PatchLocality: A pointwise locality contract.
        """
        return PatchLocality(LocalityKind.POINTWISE)

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        """
        Convert scalar class labels into a one-hot encoded tensor.
        
        Parameters:
        	tensor (torch.Tensor): Tensor containing integer class labels.
        
        Returns:
        	torch.Tensor: One-hot encoded labels with the class dimension first.
        """
        result = (
            F.one_hot(tensor.type(torch.int64), num_classes=self.num_classes)
            .permute(0, len(tensor.shape), *[i + 1 for i in range(len(tensor.shape) - 1)])
            .float()
            .squeeze(0)
        )
        return result

    def inverse(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        # Argmax the CLASS axis (the one sized num_classes) and re-insert it, restoring a [.., 1, *spatial]
        # label map. The predictor feeds this per-sample output[i] = [num_classes, *spatial] (class axis 0),
        # but a batched [B, num_classes, *spatial] (class axis 1) is also handled, so it never argmaxes a
        # batch or spatial axis.
        class_dim = 0 if tensor.shape[0] == self.num_classes else 1
        return torch.argmax(tensor, dim=class_dim).unsqueeze(class_dim)


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
        _require_simpleitk()
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "Dataset"
            if self.per_channel:
                for i, channel in enumerate(tensor):
                    image = data_to_image(channel.unsqueeze(0), cache_attribute)
                    (dataset_path / f"P{i:03d}").mkdir(parents=True, exist_ok=True)
                    sitk.WriteImage(image, str(dataset_path / f"P{i:03d}" / "Volume.mha"))
            else:
                image = data_to_image(tensor, cache_attribute)

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
        """
        Initialize an inference ensemble stack writer and reducer.
        
        Parameters:
        	dataset (str): Optional dataset path specification for storing the ensemble stack.
        	name (str): Name used when writing the ensemble stack.
        	mode (str): Reduction mode: ``"mean"``, ``"median"``, or ``"Seg"``.
        """
        super().__init__()
        self.dataset = None
        if dataset:
            filename, _, file_format = split_path_spec(dataset)
            self.dataset = Dataset(filename, file_format)
        self.name = name
        self.mode = mode

    # patch_locality stays the WHOLE_VOLUME default: the member reduction is pointwise, but __call__
    # also WRITES the whole per-member stack to disk (like Save), which a per-patch pass cannot do. It
    # is an ensemble/finalize transform, never in a streaming input chain, so this costs no streaming.

    def __call__(self, name: str, tensors: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        """
        Reduce an inference ensemble and optionally persist its member predictions.
        
        Parameters:
        	name (str): Identifier used when writing the ensemble stack.
        	tensors (torch.Tensor): Ensemble predictions with members along the first dimension.
        	cache_attribute (Attribute): Metadata associated with the predictions.
        
        Returns:
        	torch.Tensor: The single prediction tensor, or the ensemble median or mean cast to the input dtype.
        """
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

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        # Variance across the leading member axis at each voxel -- no spatial neighbour.
        """Declare that variance is computed independently at each spatial position."""
        return PatchLocality(LocalityKind.POINTWISE)

    def __call__(self, name: str, tensors: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        # Keep the leading member axis in BOTH branches: the N>1 var(0) drops it and re-adds it via
        # unsqueeze(0), so the single-member zeros must unsqueeze too or the output rank is off by one.
        """
        Compute voxel-wise variance across ensemble members.
        
        Returns:
        	torch.Tensor: Variance values with a leading singleton member axis.
        """
        return (
            tensors.float().var(0).unsqueeze(0) if tensors.shape[0] > 1 else torch.zeros_like(tensors[0]).unsqueeze(0)
        )


class SegmentationDisagreement(Transform):
    def __init__(self, ignore_background: bool = False) -> None:
        super().__init__()
        self.ignore_background = ignore_background

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        # Per-voxel majority disagreement across the members. The global torch.unique only widens the
        # label set with labels absent at a given voxel, which contribute zero counts there and never
        # change that voxel's majority -- so the result is decided voxel by voxel.
        return PatchLocality(LocalityKind.POINTWISE)

    def __call__(self, name: str, tensors: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        # tensors shape: [N, ...] with N segmentations and integer labels per voxel
        """Calculate per-voxel disagreement among multiple segmentation label maps.
        
        Parameters:
        	name (str): Sample or volume identifier.
        	tensors (torch.Tensor): Segmentation maps with ensemble members along the first axis.
        	cache_attribute (Attribute): Transform metadata.
        
        Returns:
        	torch.Tensor: Disagreement values in the range from 0 to 1 with a leading singleton axis.
        """
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
        """Initialize the percentage transform with a baseline value.
        
        Parameters:
        	baseline (float): Value used as the reference for percentage calculations.
        """
        super().__init__()
        self.baseline = baseline

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        """Describe how the transform maps output patches to input regions.
        
        Parameters:
        	cache_attribute (Attribute): Cached metadata used to determine the locality contract.
        
        Returns:
        	PatchLocality: A pointwise locality contract.
        """
        return PatchLocality(LocalityKind.POINTWISE)

    def __call__(self, name: str, tensors: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        """Calculate tensor values as percentages of a baseline.
        
        Parameters:
        	name (str): The associated data item name.
        	tensors (torch.Tensor): Values to convert to percentages.
        	cache_attribute (Attribute): Transform metadata.
        
        Returns:
        	torch.Tensor: Values divided by the baseline and multiplied by 100.
        """
        return tensors / self.baseline * 100.0


class StandardDeviation(Transform):
    def __init__(self) -> None:
        super().__init__()

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        # Standard deviation across the leading member axis at each voxel -- no spatial neighbour.
        """
        Declare voxel-wise locality for standard deviation computation.
        
        Returns:
            PatchLocality: A pointwise locality contract.
        """
        return PatchLocality(LocalityKind.POINTWISE)

    def __call__(self, name: str, tensors: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        """
        Compute the standard deviation across ensemble members at each voxel.
        
        Parameters:
            tensors (torch.Tensor): Ensemble predictions with members along the first dimension.
        
        Returns:
            torch.Tensor: Per-voxel standard deviations with a leading singleton dimension, or zeros when only one member is present.
        """
        return (
            tensors.float().std(0).unsqueeze(0) if tensors.shape[0] > 1 else torch.zeros_like(tensors[0]).unsqueeze(0)
        )


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
    """Crop a volume to the bounding box of its foreground.

    The content-dependent box is computed once (``transform_shape``) and kept on the case as ``box``
    margins; cropping is then the translation ``out[o] = volume[o + start]``, so a target patch reads
    its shifted source region. Dropped voxels mean the stored volume's statistics are not the output's
    (hence ``LocalityKind.CROP``).
    """

    def __init__(self, inverse: bool = True) -> None:
        super().__init__(inverse)

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        # Total: the box is a fact ``transform_shape`` puts on the case before the dispatcher reads any
        # declaration, but a group carries only what its writer stored, and without it there is no
        # translation to make -- only the read that would find one.
        """
        Describe the source-region requirements for cropping based on cached crop geometry.
        
        Parameters:
        	cache_attribute (Attribute): Cached transform metadata used to determine whether crop geometry is available.
        
        Returns:
        	PatchLocality: Whole-volume locality when crop geometry is unavailable; crop locality when cached geometry is present.
        """
        if "box" not in cache_attribute:
            return PatchLocality(LocalityKind.WHOLE_VOLUME)
        return PatchLocality(LocalityKind.CROP)

    def stream_region_source(
        self,
        target_slices: tuple[slice, ...],
        source_spatial_shape: list[int],
        cache_attribute: Attribute,
    ) -> list[slice]:
        # Output index o holds source index o + start, so the region behind a target patch is that
        # patch's own slices stepped forward by the box's near margin.
        """
        Map a cropped target region to the corresponding source region.
        
        Parameters:
        	target_slices (tuple[slice, ...]): Slices defining the target region.
        	source_spatial_shape (list[int]): Spatial dimensions of the source volume.
        	cache_attribute (Attribute): Cached crop metadata containing the crop margins.
        
        Returns:
        	list[slice]: Source slices corresponding to the target region.
        """
        box = Crop._parse_box(cache_attribute["box"])
        return [
            slice(target.start + int(start), target.stop + int(start))
            for target, (start, _) in zip(target_slices, box, strict=False)
        ]

    def write_stream_cache_attribute(self, cache_attribute: Attribute, source_spatial_shape: list[int]) -> None:
        """
        Updates cached origin metadata to reflect the near-corner offset of a streamed crop.
        
        Parameters:
        	cache_attribute (Attribute): Cached geometry and crop metadata.
        	source_spatial_shape (list[int]): Original spatial dimensions; unused.
        """
        if "box" not in cache_attribute:
            return
        if not {"Origin", "Spacing", "Direction"} <= set(cache_attribute.keys()):
            return
        # The crop keeps the box's near corner, so the new origin is the physical point that corner
        # already sat on: the old origin stepped along each axis by its own margin. A margin is in
        # array order and the geometry is in (x, y, z), hence the reversed indexing.
        box = Crop._parse_box(cache_attribute["box"])
        origin = torch.tensor(cache_attribute.get_np_array("Origin"))
        matrix = torch.tensor(cache_attribute.get_np_array("Direction").reshape((len(origin), len(origin))))
        origin = torch.matmul(origin, matrix)
        for dim in range(box.shape[0]):
            origin[-dim - 1] += box[dim][0] * cache_attribute.get_np_array("Spacing")[-dim - 1]
        cache_attribute["Origin"] = torch.matmul(origin, torch.inverse(matrix))

    def transform_shape(self, group_src: str, name: str, shape: list[int], cache_attribute: Attribute) -> list[int]:
        # The crop box is content-dependent (foreground bounding box), so the output shape
        # cannot be known without the pixel data. If the box was already computed and persisted
        # as a sidecar attribute, reuse it and skip the read; otherwise compute it once from the
        # volume. (A fully-lazy variant would require deferring patch planning past _load().)
        # ``shape`` is already the channel-stripped spatial shape (patching strips [C, *spatial]
        # before calling transform_shape), so the crop box — one row per spatial axis — aligns with
        # ``shape`` directly, exactly like ``__call__`` aligns it with ``tensor.shape[1:]``.
        """
        Determine the spatial shape after cropping to the detected foreground region.
        
        Parameters:
            group_src (str): Dataset group containing the source volume.
            name (str): Name of the source volume.
            shape (list[int]): Channel-stripped spatial dimensions.
            cache_attribute (Attribute): Metadata used to reuse or store the crop box.
        
        Returns:
            list[int]: Spatial dimensions after applying the crop box, or the input shape when source data is unavailable.
        """
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
        """
        Crop the tensor according to the cached bounding box and update its spatial metadata.
        
        Parameters:
            name (str): Dataset item name.
            tensor (torch.Tensor): Tensor to crop.
            cache_attribute (Attribute): Cached transform and spatial metadata.
        
        Returns:
            torch.Tensor: Cropped tensor, or the original tensor when no bounding box is cached.
        """
        if "box" not in cache_attribute:
            return tensor
        box = self._parse_box(cache_attribute["box"])
        self.write_stream_cache_attribute(cache_attribute, list(tensor.shape[1:]))
        # The box carries the FAR margin, so the stop it crops at is the one the extent in hand decides.
        for i, ((_, b), s) in enumerate(zip(box, tensor.shape[1:], strict=False)):
            box[i][1] = s - b
        image = data_to_image(tensor, cache_attribute)
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
