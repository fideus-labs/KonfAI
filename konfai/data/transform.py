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
from konfai.utils.config import _escape_key_component, apply_config
from konfai.utils.dataset import Attribute, Dataset, DataStream, data_to_image, image_to_data
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
    - ``SLAB``        -- per-voxel value map, plus a side effect that needs the slab's place in the
      volume (a per-region side write): the streamed-WRITE dispatcher runs it through
      :meth:`Transform.stream_slab` with region context; the read dispatcher has no such context and
      treats it as ``WHOLE_VOLUME``.
    - ``WHOLE_VOLUME``-- genuinely needs the whole volume: the dispatcher falls back to a full load.
    """

    POINTWISE = "pointwise"
    HALO = "halo"
    ORIENTATION = "orientation"
    CROP = "crop"
    GLOBAL_STAT = "global_stat"
    RESCALE = "rescale"
    SLAB = "slab"
    WHOLE_VOLUME = "whole_volume"

    @property
    def preserves_statistics(self) -> bool:
        """Whether this kind leaves every whole-volume statistic of its input untouched.

        Only a reorientation does: a flip or a permute is a bijection on the voxels, so the multiset of
        values -- and therefore Min/Max/Mean/Std over it -- is exactly the input's. Every other kind may
        map values (``POINTWISE``, ``GLOBAL_STAT``), mix neighbours (``HALO``) or interpolate
        (``RESCALE``). This is what decides whether the statistics of the STORED volume are still those
        of a later transform's own input (see ``DatasetManager._plan_stream_region``).
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
        """Map a target-patch's spatial slices to the source spatial region to read (region kinds).

        Overridden by the kinds whose source region is an index remap of the target's -- ``ORIENTATION``
        maps it and reorients what it reads, ``CROP`` maps it and is done. ``HALO`` and ``RESCALE`` are
        handled generically by the dispatcher, so the base raises for any transform that declares a
        region kind without providing the remap.

        ``cache_attribute`` is the case's SOURCE metadata, under the same rules as
        :meth:`patch_locality`: a remap the image decides (a reorientation whose mirrored axes are the
        case's own direction cosines) reads it here, and reads nothing else.
        """
        raise TransformError(
            f"{type(self).__name__} declared a region patch-locality but does not implement stream_region_source().",
            "Implement stream_region_source() or declare a non-region patch_locality().",
        )

    def stream_slab(
        self,
        name: str,
        tensor: torch.Tensor,
        region: slice,
        spatial_shape: list[int],
        cache_attribute: Attribute,
    ) -> torch.Tensor:
        """Run this transform on one finalized slab — rows ``region`` of a ``spatial_shape`` volume.

        The streamed-write dispatcher calls this instead of ``__call__`` for a ``SLAB`` declaration:
        the value map is per-voxel, so the default whole-volume call is exact on the slab, but the
        stage's side effect needs to know where the slab sits — which is what a ``SLAB`` transform
        overrides this to read. Slabs arrive in order and tile the volume exactly once per case.
        """
        del region, spatial_shape
        return self(name, tensor, cache_attribute)

    def stream_abort(self, name: str) -> None:
        """Drop whatever ``stream_slab`` holds open for ``name`` after a mid-case failure.

        Called by the streamed-write dispatcher when a case dies between slabs, so a ``SLAB`` stage's
        region sink or buffer does not outlive the case. The base holds nothing.
        """

    def write_stream_cache_attribute(self, cache_attribute: Attribute, source_spatial_shape: list[int]) -> None:
        """Record the geometry a whole-volume ``__call__`` would, given the FULL source shape.

        Called once per case, on the persistent attribute, for the stage that owns a streamed region.
        A transform whose geometry rewrite depends on the volume's EXTENT (a reorientation's new
        origin is the corner it mirrors onto) cannot compute it from a patch, which is all its
        ``__call__`` is handed while streaming: it writes the case-level answer here instead, and the
        patch-local one it wrote on the way is dropped rather than persisted. The base is a no-op --
        a transform that leaves geometry alone has nothing to record.
        """

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

    def inverse_patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        """Declare how ``inverse``'s output depends on its input, for the streamed-write dispatcher.

        The write mirror of :meth:`patch_locality`: a prediction's finalize chain applies transforms
        INVERTED, so the streamed-write gate asks each one about its inverse. ``cache_attribute`` is the
        finalize-time state — the case's attribute as ``inverse`` will receive it, with everything the
        forward pass pushed still stacked on it — under the same three rules (read-only, no I/O, total).

        The default derives from the forward contract where the derivation is safe for any subclass: a
        per-voxel value map inverts to a per-voxel value map, and an index remap inverts to an index
        remap. Every other kind falls to ``WHOLE_VOLUME`` — an inverse that is streamable anyway
        (``Padding``'s crop, ``Resample``'s rescale) declares itself.
        """
        forward = self.patch_locality(cache_attribute)
        if forward.kind in (LocalityKind.POINTWISE, LocalityKind.ORIENTATION):
            return forward
        return PatchLocality(LocalityKind.WHOLE_VOLUME)

    def inverse_transform_shape(self, shape: list[int], cache_attribute: Attribute) -> list[int]:
        """The spatial shape ``inverse`` produces from ``shape`` (write mirror of ``transform_shape``).

        ``cache_attribute`` is the finalize-time state, as in :meth:`inverse_patch_locality`. The
        default is the identity, exactly as (in)exact as ``transform_shape``'s: a shape-changing
        inverse must override it, and the streamed-write dispatcher only trusts it for the kinds
        :meth:`inverse_patch_locality` declared streamable.
        """
        return shape

    def stream_region_target(
        self,
        target_slices: tuple[slice, ...],
        source_spatial_shape: list[int],
        cache_attribute: Attribute,
    ) -> list[slice]:
        """Map a region of ``inverse``'s OUTPUT to the region of its INPUT it is computed from.

        The write mirror of :meth:`stream_region_source`, with the same direction of travel: the slices
        are in the space being produced (here the written image), the shape is the space being consumed
        (here the finalized accumulator), and the answer is the consumed region. The streamed-write
        dispatcher holds a sliding window of finalized slabs and emits each output region once the
        region this returns has arrived. ``cache_attribute`` is the finalize-time state; a transform
        whose remap is read from what its own ``inverse`` pops accounts for those pops on a copy.
        """
        raise TransformError(
            f"{type(self).__name__} declared a region inverse patch-locality but does not implement"
            " stream_region_target().",
            "Implement stream_region_target() or declare a non-region inverse_patch_locality().",
        )


class TransformLoader:
    """Resolve and instantiate transform classes from KonfAI configuration."""

    def __init__(self) -> None:
        pass

    def get_transform(self, classpath: str, konfai_args: str) -> Transform:
        module, name = get_module(classpath, "konfai.data.transform")
        # A key is read as a dotted path, and a classpath naming its module carries dots of its own.
        transform = apply_config(f"{konfai_args}.{_escape_key_component(classpath)}")(getattr(module, name))()
        if isinstance(transform, Transform):
            return transform
        return Foreign(transform, classpath)


class Foreign(Transform):
    """A transform from another framework, as the loader hands it over.

    Name the class where a transform goes and its arguments under it::

        transforms:
          monai.transforms:ScaleIntensity:
            minv: 0.0
            maxv: 1.0

    The class must be callable on one tensor and return the transformed tensor, which is what
    torchvision's transforms, TorchIO's and MONAI's array transforms all are. MONAI's dictionary
    transforms (``ScaleIntensityd``) take a dictionary of keys instead: a KonfAI group is the key,
    so name the array class.

    The class must be DETERMINISTIC: a transform runs on each group of a case in turn, so a random
    one would draw again for the label and misalign it from the image. Name it under the
    augmentations instead, where a draw is made once for the case and every group is handed it.

    It reads the whole volume, which is what a class saying nothing about where its output comes
    from is owed. The shape is checked rather than assumed: the patch grid is planned on the shape a
    transform announces, and this one announces the shape it was given. Geometry is left as it
    stands, which a transform of the intensities alone leaves. A class that resamples, crops or
    reorients owns both, and a ``Transform`` subclass is what states them.
    """

    def __init__(self, transform, classpath: str) -> None:
        super().__init__()
        self.classpath = classpath
        self.transform = transform

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        result = self.transform(tensor)
        if not isinstance(result, torch.Tensor):
            result = torch.as_tensor(np.asarray(result))
        if list(result.shape) != list(tensor.shape):
            raise TransformError(
                f"'{self.classpath}' returned the shape {list(result.shape)} for an input of {list(tensor.shape)}.",
                "Subclass Transform and implement transform_shape() to declare the shape it returns.",
            )
        return result


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

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        # A mask reads a separate full volume, and a percentile bound needs the whole histogram:
        # both force a whole-volume load. A 'min'/'max' bound needs a global disk statistic
        # (GLOBAL_STAT); fixed float bounds clip each voxel independently (POINTWISE).
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
        # at a different precision than the float()-cast scatter in the else branch below) and to
        # non-NaN bounds: a NaN bound — from a dynamic min/max/percentile over data containing NaN —
        # makes clamp_ propagate NaN to the whole tensor, whereas the fallback scatter no-ops on it
        # (NaN comparisons are False). Every other case takes that fallback, unchanged.
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

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        # Rescaling uses the volume-global Min/Max (restricted to self.channels); the dispatcher reads
        # those once from disk and seeds them so every patch (and inverse()) sees the same range.
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
        # The forward needs the volume's Min/Max (GLOBAL_STAT); the inverse only pops what the forward
        # stacked, so on the finalize-time attribute it is a per-voxel affine map.
        return PatchLocality(LocalityKind.POINTWISE)

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

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        return PatchLocality(LocalityKind.POINTWISE)

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

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        # A mask reads a separate full volume (whole-volume). Any of mean/std left unset is taken from
        # a volume-global disk statistic (GLOBAL_STAT); when both are given, the standardization is a
        # per-voxel affine map with constant coefficients (POINTWISE).
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

    def inverse_patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        # Mask or not, the inverse only pops the Mean/Std the forward stacked: a per-voxel affine map.
        return PatchLocality(LocalityKind.POINTWISE)

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
    # Wide enough to hold every dtype a volume is read as (int8/int16/uint8/float32) with no value moved.
    _VALUE_PRESERVING_DTYPES = frozenset({torch.float32, torch.float64})

    def __init__(self, dtype: str = "float32", inverse: bool = True) -> None:
        super().__init__(inverse)
        self.dtype: torch.dtype = getattr(torch, dtype)

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        # The promise is that the stored volume's Min/Max/Mean/Std are still a later GLOBAL_STAT's
        # input statistics, and a cast keeps them only where it keeps every value. The dtype a volume
        # is stored as is not on its header, so the target is what has to hold whatever that is:
        # float32 holds an int16 or a float32 exactly, and float16 holds neither -- it runs out of
        # mantissa at 2048, where a CT reaches 3000. An integer cast truncates.
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

    def inverse_patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        # The inverse drops the padded border and keeps a translated copy of what remains: a CROP.
        return PatchLocality(LocalityKind.CROP)

    def inverse_transform_shape(self, shape: list[int], cache_attribute: Attribute) -> list[int]:
        shape = list(shape)
        for dim in range(len(self.padding) // 2):
            shape[-dim - 1] -= sum(self.padding[dim * 2 : dim * 2 + 2])
        return shape

    def stream_region_target(
        self,
        target_slices: tuple[slice, ...],
        source_spatial_shape: list[int],
        cache_attribute: Attribute,
    ) -> list[slice]:
        # Output index o holds input index o + pad_before: a written region pulls its own slices stepped
        # forward by the leading pad (padding pairs are in reversed axis order, like F.pad).
        before = [0] * len(target_slices)
        for dim in range(min(len(self.padding) // 2, len(before))):
            before[-dim - 1] = self.padding[dim * 2]
        return [slice(t.start + b, t.stop + b) for t, b in zip(target_slices, before, strict=False)]

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

    # WHOLE_VOLUME on purpose: squeeze/unsqueeze changes the tensor rank, and the streamed write sizes
    # each slab from the pre-finalize accumulator grid -- a rank change past it cannot region-stream.

    def transform_shape(self, group_src: str, name: str, shape: list[int], cache_attribute: Attribute) -> list[int]:
        # ``shape`` is the channel-stripped spatial shape (patching strips [C, *spatial] before folding),
        # so the runtime tensor is [C, *shape] and ``self.dim`` indexes into that. Squeezing the channel
        # (axis 0) leaves the spatial grid untouched; squeezing a spatial axis drops it from the grid --
        # but only when it is size 1, exactly as ``torch.squeeze`` does (a non-singleton axis is a no-op).
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
        return PatchLocality(LocalityKind.RESCALE)

    def _resample(self, tensor: torch.Tensor, size: list[int]) -> torch.Tensor:
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

    def _inverse_geometry(self, cache_attribute: Attribute) -> list[int]:
        """Pop the Size/Spacing stack the forward pushed and return the size the inverse restores."""
        cache_attribute.pop_np_array("Size")
        size_1 = cache_attribute.pop_np_array("Size")
        if "Spacing" in cache_attribute:
            cache_attribute.pop_np_array("Spacing")
        return [int(size) for size in size_1]

    def inverse(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        return self._resample(tensor, self._inverse_geometry(cache_attribute))

    def inverse_patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        # The inverse rescales back to the size the forward stacked: patch-native (RESCALE) whenever
        # that stack is on the finalize-time attribute, judged on a copy (a declaration never pops).
        try:
            self._inverse_geometry(Attribute(cache_attribute))
        except NameError:
            return PatchLocality(LocalityKind.WHOLE_VOLUME)
        return PatchLocality(LocalityKind.RESCALE)

    def inverse_transform_shape(self, shape: list[int], cache_attribute: Attribute) -> list[int]:
        try:
            return self._inverse_geometry(Attribute(cache_attribute))
        except NameError:
            return shape

    def stream_region_target(
        self,
        target_slices: tuple[slice, ...],
        source_spatial_shape: list[int],
        cache_attribute: Attribute,
    ) -> list[slice]:
        # The inverse rescales the accumulator (n_in) back to the stored size: a written region pulls
        # through the same coordinate formula as the forward read, with the roles swapped.
        n_in = [int(s) for s in source_spatial_shape]
        n_out = self.inverse_transform_shape(list(n_in), cache_attribute)
        scales = [n_in[k] / n_out[k] for k in range(len(n_in))]
        return Resample.source_window(target_slices, scales, n_in)

    # Every patch derives its source coordinates from the same global scale (n_in / n_out, from the
    # truncated integer sizes F.interpolate itself uses), which is what makes the streamed patches
    # agree with the whole-volume call and with each other across a seam.
    def _stream_mode(self, tensor: torch.Tensor) -> str:
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
        """Map a TARGET-grid patch to the minimal SOURCE region to read.

        Returns ``(source_slices, region_starts, scales, n_in, n_out)`` — all in
        array axis order (Z, Y, X). The ``halo`` is a pure safety margin (the
        formula's ``+2`` already captures the i1 neighbour); nearest needs none.
        """
        n_in = [int(s) for s in source_spatial_shape]
        n_out = [int(s) for s in self.transform_shape("", "", list(n_in), cache_attribute)]
        scales = [n_in[k] / n_out[k] for k in range(len(n_in))]
        source_slices = Resample.source_window(target_slices, scales, n_in, halo)
        return source_slices, [s.start for s in source_slices], scales, n_in, n_out

    @staticmethod
    def source_window(
        target_slices: tuple[slice, ...] | list[slice],
        scales: list[float],
        n_in: list[int],
        halo: int = 1,
    ) -> list[slice]:
        """The clamped source region a target region reads from, per axis, given the scales.

        Covers BOTH samplers, because the same window serves either mode: the linear taps around the
        half-pixel source (``scale * (o + 0.5) - 0.5``, plus the ``+2``/``halo`` margin for the i1
        neighbour) AND the voxel nearest picks (``floor(o * scale)`` -- F.interpolate's own nearest
        index). Under strong downsampling the nearest voxel of the first output column falls BELOW the
        linear window's start, so omitting it read one voxel short and the gather wrapped a negative
        local index onto the far edge.
        """
        source_slices: list[slice] = []
        for k, sl in enumerate(target_slices):
            smin = int(np.floor(scales[k] * (sl.start + 0.5) - 0.5))
            smax = int(np.floor(scales[k] * ((sl.stop - 1) + 0.5) - 0.5))
            near_lo = int(np.floor(sl.start * scales[k]))
            near_hi = int(np.floor((sl.stop - 1) * scales[k]))
            start = min(smin - halo, near_lo)
            stop = max(smax + 2 + halo, near_hi + 1)
            source_slices.append(slice(max(0, start), min(n_in[k], stop)))
        return source_slices

    def resample_region(
        self,
        sub_tensor: torch.Tensor,
        target_slices: tuple[slice, ...],
        region_starts: list[int],
        scales: list[float],
        n_in: list[int],
    ) -> torch.Tensor:
        """Interpolate a source sub-region to the target patch extent.

        ``sub_tensor`` is ``[C, (z, y, x)]`` covering ``source_slices``;
        ``region_starts`` are the global source indices of its first voxel per
        axis. Uses the same global coordinate formula as the whole-volume path,
        indexing the sub-region as ``sub[i - region_start]``.
        """
        mode = self._stream_mode(sub_tensor)
        dev = sub_tensor.device
        ndim = len(target_slices)
        if mode == "nearest":
            indices = []
            for k in range(ndim):
                # Take the axis's index map from F.interpolate itself, so streamed nearest picks the
                # same source voxel as the whole-volume call for every size ratio.
                src = torch.arange(n_in[k], device=dev, dtype=torch.float32).reshape(1, 1, -1)
                n_out_k = round(n_in[k] / scales[k])
                index = F.interpolate(src, size=n_out_k, mode="nearest").long().flatten()
                indices.append(index[target_slices[k].start : target_slices[k].stop] - region_starts[k])
            # One gather over broadcast index views instead of one volume copy per axis (nearest is a
            # pure coordinate gather, so composing the axes changes no value).
            return sub_tensor[(slice(None), *torch.meshgrid(*indices, indexing="ij"))]

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
        """Record the same 'Spacing'/'Size' stack a whole-volume ``__call__`` would.

        Called once per case on the persistent attribute so ``inverse()`` at
        prediction time pops exactly what the non-streamed path pushed. Uses the
        FULL source shape, never the halo'd sub-region.
        """


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

    def write_stream_cache_attribute(self, cache_attribute: Attribute, source_spatial_shape: list[int]) -> None:
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

    def write_stream_cache_attribute(self, cache_attribute: Attribute, source_spatial_shape: list[int]) -> None:
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

    Per-voxel, so it declares ``SLAB``: the value map is exact on a slab, and the only thing that
    needs the slab's place in the volume is *which rows of the mask to read*. The mask is assumed
    aligned to the volume at this point, so a slab reads the matching rows of the mask (a dataset mask
    region-read, a ``.mha`` mask sliced from the one cached copy) instead of loading the whole volume.
    ``__call__`` (the whole-volume path, and the read side, which has no region to place) stays exact.
    """

    def __init__(self, path: str = "./default.mha", value_outside: int = 0) -> None:
        super().__init__()
        self.path = path
        self.value_outside = value_outside
        self._cached_mask: torch.Tensor | None = None

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        return PatchLocality(LocalityKind.SLAB)

    def _apply(self, tensor: torch.Tensor, mask: torch.Tensor | np.ndarray) -> torch.Tensor:
        # Index on the tensor's own device so the mask works whether the volume is on CPU or GPU
        # (``torch.as_tensor`` keeps a torch mask as-is and wraps a numpy one, moving it to the device).
        tensor[torch.as_tensor(mask, device=tensor.device) == 0] = self.value_outside
        return tensor

    def _cached_mha(self) -> torch.Tensor:
        _require_simpleitk()
        if self._cached_mask is None:
            self._cached_mask = torch.tensor(sitk.GetArrayFromImage(sitk.ReadImage(self.path))).unsqueeze(0)
        return self._cached_mask

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        if self.path.endswith(".mha"):
            return self._apply(tensor, self._cached_mha())
        for dataset in self.datasets:
            if dataset.is_dataset_exist(self.path, name):
                mask, _ = dataset.read_data(self.path, name)
                return self._apply(tensor, mask)
        raise NameError(f"Mask : {self.path}/{name} not found")

    def stream_slab(
        self,
        name: str,
        tensor: torch.Tensor,
        region: slice,
        spatial_shape: list[int],
        cache_attribute: Attribute,
    ) -> torch.Tensor:
        # Read only the slab's rows of the (aligned) mask, so the output streams within a window: a
        # dataset mask is region-read; a ``.mha`` mask is sliced from the single cached copy (the mask
        # is 1-channel, far smaller than the C-channel output it would otherwise hold whole).
        if self.path.endswith(".mha"):
            return self._apply(tensor, self._cached_mha()[:, region])
        slices = (slice(None), region, *(slice(0, extent) for extent in spatial_shape[1:]))
        for dataset in self.datasets:
            if dataset.is_dataset_exist(self.path, name):
                mask, _ = dataset.read_data_slice(self.path, name, slices)
                return self._apply(tensor, mask)
        raise NameError(f"Mask : {self.path}/{name} not found")


class Dilate(Transform):
    def __init__(self, dilate: int = 1) -> None:
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

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        # Pointwise only when reducing the leading channel/model axis (dim 0); a spatial sum spans
        # the whole extent, so it falls back to the whole volume.
        if self.dim == 0:
            return PatchLocality(LocalityKind.POINTWISE)
        return PatchLocality(LocalityKind.WHOLE_VOLUME)

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

    Models are assumed to segment disjoint structures, but boundaries disagree in practice: a voxel
    claimed by several models takes the label of the LAST model in ensemble order (adding the global
    ids instead would fabricate a label belonging to neither model).
    """

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        # Merges the leading model axis per voxel; spatial support is a single voxel.
        return PatchLocality(LocalityKind.POINTWISE)

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
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
        super().__init__()
        self.per_dim = per_dim

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        # First-difference gradient: each output voxel reads its immediate neighbour, a HALO of radius
        # 1. The far-edge ConstantPad reproduces the whole-volume border once the halo clamps there.
        return PatchLocality(LocalityKind.HALO, halo=(1,))

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

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        # Pointwise ONLY when reducing the channel axis (dim 0). Over a spatial axis the argmax spans
        # the whole extent, so a per-patch argmax would diverge -- fall back to the whole volume.
        if self.dim == 0:
            return PatchLocality(LocalityKind.POINTWISE)
        return PatchLocality(LocalityKind.WHOLE_VOLUME)

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        return torch.argmax(tensor, dim=self.dim).unsqueeze(self.dim)


class Softmax(Transform):
    def __init__(self, dim: int = 0) -> None:
        super().__init__()
        self.dim = dim

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        # Pointwise ONLY when reducing the channel axis (dim 0). Over a spatial axis softmax normalises
        # across the whole extent, so a per-patch softmax would diverge -- fall back to the whole volume.
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
        return PatchLocality(LocalityKind.POINTWISE)

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

    # WHOLE_VOLUME by declaration, yet the case may still stream: a Save whose cache exists is a
    # source boundary, and an unsatisfied Save with a streamable prefix is materialized slab by slab
    # first (DatasetManager._materialize_save). Only an unsweepable prefix loads the whole volume.

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

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
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
        source_slices = [slice(0, n) for n in source_spatial_shape]
        for k, sl in enumerate(target_slices):
            source_slices[self.dims[k + 1] - 1] = slice(sl.start, sl.stop)
        return source_slices

    def inverse_transform_shape(self, shape: list[int], cache_attribute: Attribute) -> list[int]:
        result = list(shape)
        for k, d in enumerate(self.dims[1:]):
            result[d - 1] = shape[k]
        return result

    def stream_region_target(
        self,
        target_slices: tuple[slice, ...],
        source_spatial_shape: list[int],
        cache_attribute: Attribute,
    ) -> list[slice]:
        # Input axis k carries output axis ``dims[k + 1] - 1``: a written region pulls, per input axis,
        # the slice of the output axis it came from.
        return [slice(target_slices[d - 1].start, target_slices[d - 1].stop) for d in self.dims[1:]]

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        return tensor.permute(tuple(self.dims))

    def inverse(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        return tensor.permute(tuple(np.argsort(self.dims)))


class Flip(TransformInverse):
    def __init__(self, dims: str = "1|0|2", inverse: bool = True) -> None:
        super().__init__(inverse)

        self.dims = [int(d) + 1 for d in str(dims).split("|")]

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        return PatchLocality(LocalityKind.ORIENTATION)

    def stream_region_source(
        self,
        target_slices: tuple[slice, ...],
        source_spatial_shape: list[int],
        cache_attribute: Attribute,
    ) -> list[slice]:
        # A flipped spatial axis reads the mirror region ``[n - stop, n - start)``; applying the flip
        # to that sub-region reproduces the target patch. Non-flipped axes read the identity region.
        source_slices: list[slice] = []
        for k, sl in enumerate(target_slices):
            n = source_spatial_shape[k]
            if (k + 1) in self.dims:
                source_slices.append(slice(n - sl.stop, n - sl.start))
            else:
                source_slices.append(slice(sl.start, sl.stop))
        return source_slices

    def stream_region_target(
        self,
        target_slices: tuple[slice, ...],
        source_spatial_shape: list[int],
        cache_attribute: Attribute,
    ) -> list[slice]:
        # A flip is its own inverse: a written region pulls exactly the region the forward would read.
        return self.stream_region_source(target_slices, source_spatial_shape, cache_attribute)

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
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
        """Per output SPATIAL axis, the source axis it reads and whether it reads it mirrored.

        ``reorientation`` maps an output coordinate onto the input it comes from, so it is an exact
        remap exactly when it is a signed permutation: output physical axis ``c`` then reads input
        physical axis ``r``, backwards where the sign is negative. Anything else mixes axes. Axes are
        returned in array order, where physical axis k is array axis ``n - 1 - k``. The test (every
        column of L1 norm 1 with peak 1) admits exactly the signed permutations: unit column sums
        alone would also pass an axis-averaging matrix.
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
        """The exact index remap this case's reorientation is, or ``None`` where it is not one.

        Total: a case whose header carries no usable direction cosines has no remap to make, and an
        oblique one has none to make either -- both answer ``None`` rather than raise, and the resample
        is what answers for them.
        """
        if "Direction" not in cache_attribute or cache_attribute.get_np_array("Direction").size != 9:
            return None
        return Canonical._index_remap(self._reorientation(cache_attribute))

    @staticmethod
    def _carried(per_physical_axis: torch.Tensor, remap: list[tuple[int, bool]] | None) -> torch.Tensor:
        """Carry a per-physical-axis quantity along a remap: output axis c takes the axis it reads.

        A spacing and a half-extent travel with the axis they belong to -- what a reorientation
        preserves is the volume's physical extent, not which axis carries it. An oblique direction is
        resampled onto the input's own grid, so without a remap nothing moves.
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
        remap = self._orthogonal_remap(cache_attribute)
        if remap is None:
            return shape
        return [shape[source] for source, _ in remap]

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        # Only the case can say which reorientation this is, so only the header can answer. An orthogonal
        # one -- mirroring or permuting -- remaps indices, which is what ORIENTATION streams; an oblique
        # one is resampled from the whole volume.
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

    def _inverse_remap(self, cache_attribute: Attribute) -> list[tuple[int, bool]] | None:
        """The forward remap judged on the state ``inverse`` runs from: the popped-to source direction.

        The inverse pops the canonical geometry and reorients back through the SOURCE direction under
        it, so its streamability is the popped state's — evaluated on a copy, since a declaration
        never mutates the case. A matrix and its inverse are signed permutations together, so the
        forward remap answers for both; ``None`` where the case is oblique or carries no direction.
        """
        scoped = Attribute(cache_attribute)
        if "Direction" not in scoped:
            return None
        scoped.pop("Direction")
        return self._orthogonal_remap(scoped)

    def inverse_patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        if self._inverse_remap(cache_attribute) is None:
            return PatchLocality(LocalityKind.WHOLE_VOLUME)
        return PatchLocality(LocalityKind.ORIENTATION)

    def inverse_transform_shape(self, shape: list[int], cache_attribute: Attribute) -> list[int]:
        # transform_shape reads target axis k's extent from source axis ``source``; the inverse puts
        # each extent back on the axis it came from.
        remap = self._inverse_remap(cache_attribute)
        if remap is None:
            return shape
        result = list(shape)
        for k, (source, _) in enumerate(remap):
            result[source] = shape[k]
        return result

    def stream_region_target(
        self,
        target_slices: tuple[slice, ...],
        source_spatial_shape: list[int],
        cache_attribute: Attribute,
    ) -> list[slice]:
        # Canonical axis k holds source axis ``source``'s content: a written region pulls, per input
        # axis, the slice of the output axis it carries — taken mirrored within the input extent where
        # the remap reads that axis backwards (a flip restricted to a region is that region reversed).
        remap = self._inverse_remap(cache_attribute)
        if remap is None:
            raise TransformError(
                "Canonical declared a region inverse patch-locality for a direction it cannot remap exactly.",
                "Report this: inverse_patch_locality() and stream_region_target() disagree about the case.",
            )
        source_slices: list[slice] = []
        for k, (source, mirrored) in enumerate(remap):
            target = target_slices[source]
            extent = source_spatial_shape[k]
            source_slices.append(
                slice(extent - target.stop, extent - target.start) if mirrored else slice(target.start, target.stop)
            )
        return source_slices

    def _reorient(self, tensor: torch.Tensor, reorientation: torch.Tensor) -> torch.Tensor:
        """Apply a reorientation: an exact index remap where it is one, a resample where it is not.

        An orthogonal reorientation is a bijection on the voxels, so it must reproduce the input's
        multiset bit for bit -- which only a permute and a flip do.
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
        reorientation = self._reorientation(cache_attribute)
        self.write_stream_cache_attribute(cache_attribute, list(tensor.shape[1:]))
        return self._reorient(tensor, reorientation)

    def inverse(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        # Popping restores the source geometry, which is what the inverse remap is then read from.
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
        return PatchLocality(LocalityKind.POINTWISE)

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
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
        return PatchLocality(LocalityKind.POINTWISE)

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
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
        super().__init__()
        self.dataset = None
        if dataset:
            filename, _, file_format = split_path_spec(dataset)
            self.dataset = Dataset(filename, file_format)
        self.name = name
        self.mode = mode
        self._stack_sinks: dict[str, DataStream] = {}
        self._stack_buffers: dict[str, list[np.ndarray]] = {}

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        # The member reduction is per-voxel; the per-member stack write is the side effect that needs
        # the slab's place in the volume, which is exactly what SLAB declares (whole-volume on the
        # read side, streamed region by region on the write side via ``stream_slab``).
        return PatchLocality(LocalityKind.SLAB)

    def _stack(self, tensors: torch.Tensor) -> np.ndarray:
        if self.mode == "Seg":
            _tensors = torch.argmax(torch.softmax(tensors, dim=1), dim=1).to(torch.uint8)
        else:
            _tensors = tensors.squeeze(1)
        return _tensors.float().cpu().numpy()

    def _reduce(self, tensors: torch.Tensor) -> torch.Tensor:
        return (
            torch.median(tensors.float(), dim=0).values.to(tensors.dtype)
            if self.mode == "median"
            else tensors.float().mean(0).to(tensors.dtype)
        )

    def __call__(self, name: str, tensors: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        if tensors.shape[0] == 1:
            return tensors.squeeze(0)
        dataset = self.dataset if self.dataset else self.datasets[-1]
        dataset.write("InferenceStack", name, self._stack(tensors), cache_attribute)
        return self._reduce(tensors)

    def stream_slab(
        self,
        name: str,
        tensor: torch.Tensor,
        region: slice,
        spatial_shape: list[int],
        cache_attribute: Attribute,
    ) -> torch.Tensor:
        """The whole-volume call, region by region: reduce the members per voxel and write the slab's
        rows of the per-member stack into a region sink opened at the first slab. A destination that
        cannot serve region writes falls back to buffering the stack and writing it classically at
        the last slab — the memory cost of the whole-volume path, never a lost stack."""
        if tensor.shape[0] == 1:
            return tensor.squeeze(0)
        stack = self._stack(tensor)
        dataset = self.dataset if self.dataset else self.datasets[-1]
        if name not in self._stack_sinks and name not in self._stack_buffers:
            sink = dataset.open_data_stream(
                "InferenceStack", name, [stack.shape[0], *spatial_shape], stack.dtype, cache_attribute
            )
            if sink is None:
                self._stack_buffers[name] = []
            else:
                self._stack_sinks[name] = sink
        if name in self._stack_buffers:
            self._stack_buffers[name].append(stack)
            if region.stop == spatial_shape[0]:
                whole = np.concatenate(self._stack_buffers.pop(name), axis=1)
                dataset.write("InferenceStack", name, whole, cache_attribute)
        else:
            target = (slice(0, stack.shape[0]), region, *(slice(0, extent) for extent in spatial_shape[1:]))
            self._stack_sinks[name].write_slice(target, stack)
            if region.stop == spatial_shape[0]:
                self._stack_sinks.pop(name).__exit__(None, None, None)
        return self._reduce(tensor)

    def stream_abort(self, name: str) -> None:
        self._stack_buffers.pop(name, None)
        sink = self._stack_sinks.pop(name, None)
        if sink is not None:
            sink.__exit__(None, None, None)


class Norm(Transform):
    """Vector magnitude over the trailing component axis.

    Reduces a stacked vector field (e.g. a displacement-field ensemble ``[N, (D), H, W, C]``) to
    per-sample magnitudes ``[N, (D), H, W]``, typically before ``Variance``/``StandardDeviation``.
    The trailing tensor axis is the first geometry axis (numpy order is reversed), so that axis is
    dropped from ``Origin``/``Spacing``/``Direction``.
    """

    def __init__(self) -> None:
        super().__init__()

    # WHOLE_VOLUME on purpose: the magnitude drops the trailing spatial axis, and the streamed write
    # sizes each slab from the pre-finalize accumulator grid -- a rank change past it cannot region-stream.

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
        return PatchLocality(LocalityKind.POINTWISE)

    def __call__(self, name: str, tensors: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        # Keep the leading member axis in BOTH branches: the N>1 var(0) drops it and re-adds it via
        # unsqueeze(0), so the single-member zeros must unsqueeze too or the output rank is off by one.
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

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        return PatchLocality(LocalityKind.POINTWISE)

    def __call__(self, name: str, tensors: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        return tensors / self.baseline * 100.0


class StandardDeviation(Transform):
    def __init__(self) -> None:
        super().__init__()

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        # Standard deviation across the leading member axis at each voxel -- no spatial neighbour.
        return PatchLocality(LocalityKind.POINTWISE)

    def __call__(self, name: str, tensors: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
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
        box = Crop._parse_box(cache_attribute["box"])
        return [
            slice(target.start + int(start), target.stop + int(start))
            for target, (start, _) in zip(target_slices, box, strict=False)
        ]

    def write_stream_cache_attribute(self, cache_attribute: Attribute, source_spatial_shape: list[int]) -> None:
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
