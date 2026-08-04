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
    - ``REGRID``      -- resample onto ANOTHER grid. ``RESCALE``'s source and target cover the same
      physical box and differ only in sampling density, so a size ratio is the whole of its map;
      this one's target is a grid in its own right, placed by its own origin, so the map carries an
      offset as well as a scale and part of the target may read from outside the source altogether.
      The stage owns both halves: it declares the source region a target region pulls
      (:meth:`Transform.stream_region_source`) and interpolates it (:meth:`Transform.stream_region`).
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
    REGRID = "regrid"
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
class RegionContext:
    """Where a streamed region sits, for a stage that needs to know.

    ``source`` is the part of the stage's INPUT the tensor covers, ``target`` the part of its OUTPUT
    it must produce; they differ whenever the stage moves or resizes data (a halo read, a resample,
    a warp onto another grid). ``source_shape`` and ``target_shape`` are the two whole extents those
    regions are cut from -- a region alone cannot say how far it is from an edge.
    """

    source: tuple[slice, ...]
    target: tuple[slice, ...]
    source_shape: tuple[int, ...]
    target_shape: tuple[int, ...]


@dataclass(frozen=True)
class PatchLocality:
    """A transform's declared patch-locality contract (see :class:`LocalityKind`).

    ``halo`` is the per-spatial-axis neighbourhood radius in array order (Z, Y, X); a length-1
    tuple broadcasts to every axis. ``stat_keys`` are the ``Attribute`` keys a ``GLOBAL_STAT``
    transform reads before running (a subset of ``Min``/``Max``/``Mean``/``Std``). ``stat_channels``
    restricts the statistic to those channels (``Normalize.channels``).

    ``reason`` is how a ``WHOLE_VOLUME`` declaration explains itself.
    """

    kind: LocalityKind
    halo: tuple[int, ...] = ()
    stat_keys: frozenset[str] = field(default_factory=frozenset)
    stat_channels: list[int] | None = None
    # Overrides the kind-level default (see LocalityKind.preserves_statistics): a POINTWISE transform
    # that maps no value (TensorCast to a float dtype) may declare True so a later GLOBAL_STAT can
    # still seed from the stored volume.
    preserves_statistics: bool | None = None
    #: Why this stage needs the whole volume, in the words the plan prints. A stage that is
    #: INHERENTLY whole-volume (it changes the tensor's rank) leaves this None and the planner says
    #: so generically. A stage that is whole-volume only because something was left undeclared owes
    #: the reader that sentence: "it needs the whole volume" reads as a property of the transform
    #: when it is in fact a property of the configuration, and the reader then has nothing to change.
    reason: str | None = None

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

    def prepare(self, konfai_args: str) -> None:
        """Told where this stage's own configuration lives, once, right after it was built.

        The loader knows the subtree a stage read its arguments from; a stage that instantiates
        something ELSE from configuration — an operator named by classpath — cannot know it, and
        without this would have to build that object with no arguments at all. The base holds
        nothing: only a stage with a sub-object of its own overrides it.
        """

    def plan_note(self, group_dest: str, name: str, shape: list[int], cache_attribute: Attribute) -> str | None:
        """Something about this case the plan should say, beyond its regime and its cost.

        A stage can be correct, stream, fit the budget, and still surprise the reader — a cost the
        plan has no column for. The plan is where a run is read before it is trusted, so that is
        where the sentence belongs, rather than in a viewer afterwards.

        Answered from headers on the launcher, per (chain, case), under :meth:`patch_locality`'s
        rules: read-only, no volume read, and an answer for any case. Identical notes are printed
        once, so a note about the STAGE may repeat per case without repeating on the page, while a
        note about the CASE stays one line each.

        The base holds nothing: most stages have nothing to add to their regime and their bytes.
        """
        del group_dest, name, shape, cache_attribute
        return None

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

    def stream_region(
        self,
        name: str,
        tensor: torch.Tensor,
        context: RegionContext,
        cache_attribute: Attribute,
    ) -> torch.Tensor:
        """Apply this stage to a region, told WHERE that region sits in the volume.

        The dispatcher already computes this position -- it has to, to know what to read -- and by
        default throws it away, because almost nothing needs it: a value map gives the same answer
        wherever its input came from. Override this when the answer does depend on the place, which
        in practice means a stage reading a SECOND volume aligned with the first (a displacement
        field, a mask, a bias field): ``region`` says which part of that companion to read.

        ``context`` says which part of the input the tensor covers and which part of the output is
        expected back. The default delegates to :meth:`__call__`, so every existing transform keeps
        its behaviour and the whole-volume path stays the reference: an override must give the same
        answer as ``__call__`` would on the full volume, restricted to ``context.target``.
        """
        del context
        return self(name, tensor, cache_attribute)

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
        if not hasattr(module, name) and ":" not in classpath:
            # A bare name the transform package does not have may be an AUGMENTATION: those are
            # stages too, and a chain declares them exactly where they apply (see Expand). Looked up
            # second, so a transform never loses its name to a same-named draw.
            module, name = get_module(classpath, "konfai.data.augmentation")
            if not hasattr(module, name):
                raise TransformError(
                    f"No transform or augmentation is named '{name}'.",
                    self._closest_stage_name(name)
                    + "A bare name resolves in konfai.data.transform, then konfai.data.augmentation;"
                    " use 'module:Class' for a class anywhere else.",
                )
        if not hasattr(module, name):
            # The qualified form reaches here: the module imported, the class in it did not exist.
            raise TransformError(
                f"'{classpath}' names no '{name}' in module '{module.__name__}'.",
                "Check the class name, or drop the module to resolve a KonfAI stage by name alone.",
            )
        factory = getattr(module, name)
        if not isinstance(factory, type):
            raise TransformError(
                f"'{classpath}' names a {type(factory).__name__}, not a stage class.",
                "A chain stage is a class -- a Transform, a DataAugmentation, or a foreign class to"
                " wrap. Name one, e.g. 'Clip' or 'monai.transforms:ScaleIntensity'.",
            )
        # A key is read as a dotted path, and a classpath naming its module carries dots of its own.
        subtree = f"{konfai_args}.{_escape_key_component(classpath)}"
        transform = apply_config(subtree)(factory)()
        if isinstance(transform, Transform):
            transform.prepare(subtree)
            return transform
        if _is_augmentation(transform):
            # A draw is handed over as itself: the manager binds it to a copy (AugmentedStage) once
            # it knows which copy it is planning, which is the one thing the loader cannot know.
            transform.load(1.0)
            return transform
        return Foreign(transform, classpath)

    @staticmethod
    def _closest_stage_name(name: str) -> str:
        """A 'did you mean' over BOTH stage namespaces, so the suggestion is never Python's own
        guess from whichever module happened to fail last."""
        import difflib

        from konfai.data import augmentation

        candidates = {
            candidate
            for namespace in (globals(), vars(augmentation))
            for candidate, obj in namespace.items()
            if isinstance(obj, type)
            and not candidate.startswith("_")
            and any(base.__name__ in ("Transform", "DataAugmentation") for base in obj.__mro__)
        }
        closest = difflib.get_close_matches(name, sorted(candidates), n=1)
        return f"Closest name: '{closest[0]}'. " if closest else ""


def _is_augmentation(candidate: object) -> bool:
    """Whether this object is a KonfAI draw, asked without importing the augmentation module here.

    ``konfai.data.augmentation`` imports this module, so the dependency only runs one way; the check
    walks the class's own ancestry instead of using ``isinstance``.
    """
    return any(base.__name__ == "DataAugmentation" for base in type(candidate).__mro__)


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
        # Half kernels for every mode, so upcasting the whole (channels x volume) tensor to float32 would
        # double the memory of a multi-class output resample for no argmax benefit. On the CPU, compute in
        # float32: Half CPU kernels are missing from older torch releases. Integer inputs (uint8 labels)
        # still need a float grid for interpolation.
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
        offsets: list[float] | None = None,
    ) -> list[slice]:
        """The clamped source region a target region reads from, per axis, given the scales.

        Covers BOTH samplers, because the same window serves either mode: the linear taps around the
        half-pixel source (``scale * (o + 0.5) - 0.5``, plus the ``+2``/``halo`` margin for the i1
        neighbour) AND the voxel nearest picks (``floor(o * scale)`` -- F.interpolate's own nearest
        index). Under strong downsampling the nearest voxel of the first output column falls BELOW the
        linear window's start: the window must include it, or the gather wraps a negative local index
        onto the far edge.

        ``offsets`` generalises the map to ``source = scale * target + offset``, for a resample whose
        target grid is placed by its own origin rather than sharing the source's box (``REGRID``).
        Left ``None``, every coordinate below is the one this always computed -- the half-pixel map
        is not re-derived through a more general formula that would round differently, because the
        paths that must stay bit-identical to ``F.interpolate`` run exactly this code.
        """
        if offsets is not None:
            return Resample._offset_window(target_slices, scales, offsets, n_in, halo)
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

    @staticmethod
    def _offset_window(
        target_slices: tuple[slice, ...] | list[slice],
        scales: list[float],
        offsets: list[float],
        n_in: list[int],
        halo: int,
    ) -> list[slice]:
        """``source_window`` for an offset map, clamped to a non-empty region of the source.

        Non-empty even when the target region lies entirely off the source, which is a real place
        for a ``REGRID`` and not an error: every sample there is out of bounds and takes the fill,
        so what the read returns is never looked at -- but a zero-width read is not something every
        backend serves, and one voxel costs nothing.
        """
        source_slices: list[slice] = []
        for k, sl in enumerate(target_slices):
            first = scales[k] * sl.start + offsets[k]
            last = scales[k] * (sl.stop - 1) + offsets[k]
            low, high = (first, last) if first <= last else (last, first)
            # floor(low) is the linear map's i0 and bounds its nearest pick; +2 reaches i1 past
            # floor(high), matching the `smax + 2` the half-pixel window uses for the same reason.
            start = int(np.floor(low)) - halo
            stop = int(np.floor(high)) + 2 + halo
            start = min(max(start, 0), n_in[k] - 1)
            source_slices.append(slice(start, min(max(stop, start + 1), n_in[k])))
        return source_slices

    def resample_region(
        self,
        sub_tensor: torch.Tensor,
        target_slices: tuple[slice, ...],
        region_starts: list[int],
        scales: list[float],
        n_in: list[int],
        offsets: list[float] | None = None,
    ) -> torch.Tensor:
        """Interpolate a source sub-region to the target patch extent.

        ``sub_tensor`` is ``[C, (z, y, x)]`` covering ``source_slices``;
        ``region_starts`` are the global source indices of its first voxel per
        axis. Uses the same global coordinate formula as the whole-volume path,
        indexing the sub-region as ``sub[i - region_start]``.

        ``offsets`` generalises the map to ``source = scale * target + offset``, as in
        :meth:`source_window`, and a sample landing outside the source then takes
        :attr:`fill_value`. Left ``None``, this runs the half-pixel code it always did.
        """
        if offsets is not None:
            return self._resample_offset_region(sub_tensor, target_slices, region_starts, scales, n_in, offsets)
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

    #: What a sample landing outside the source is worth. Only an offset map can land outside at
    #: all, so only a ``REGRID`` stage ever reads this, and it sets it from its own configuration.
    fill_value: float = 0.0

    def _resample_offset_region(
        self,
        sub_tensor: torch.Tensor,
        target_slices: tuple[slice, ...],
        region_starts: list[int],
        scales: list[float],
        n_in: list[int],
        offsets: list[float],
    ) -> torch.Tensor:
        """``resample_region`` for an offset map: ITK's sampler, and a fill where the source stops.

        THE SAMPLING RULE IS ``sitk.Resample``'S, deliberately: a sample is inside while its
        continuous source index lies in ``[-0.5, n - 0.5)``; inside, the interpolation taps are
        clamped to the buffer, so the half-voxel rim beyond the outermost voxel CENTRES reproduces
        the border value rather than falling off; outside, the sample is :attr:`fill_value`. Written
        against SimpleITK because that is what an independent check of this arithmetic will be, and
        a sampler that is only nearly the same as the reference makes every such check a negotiation.

        Coordinates are global (the target index, not its offset within the region), so a region and
        the whole volume put the same sample in the same place -- which is what makes the streamed
        and whole-volume paths equal by construction here rather than by agreement.
        """
        device = sub_tensor.device
        ndim = len(target_slices)
        window = [int(extent) for extent in sub_tensor.shape[1:]]
        # Coordinates in float32 whatever the payload: a float16 volume would otherwise index itself
        # with float16 coordinates, which cannot even count the voxels of a large axis.
        coordinates = [
            scales[k] * torch.arange(sl.start, sl.stop, device=device, dtype=torch.float32) + offsets[k]
            for k, sl in enumerate(target_slices)
        ]
        inside = [(axis >= -0.5) & (axis < n_in[k] - 0.5) for k, axis in enumerate(coordinates)]
        out_shape = [int(sub_tensor.shape[0])] + [sl.stop - sl.start for sl in target_slices]
        if not all(bool(axis.any()) for axis in inside):
            # Nothing of this region is on the source. Real for a REGRID -- a target grid may reach
            # past its case -- and worth its own exit: the gather below would read a window that was
            # only ever clamped to something legal, and then be overwritten by the fill anyway.
            return torch.full(out_shape, self.fill_value, device=device, dtype=torch.float32).type(sub_tensor.dtype)

        def local(index: torch.Tensor, k: int) -> torch.Tensor:
            """A global source index as an offset into the window that was read, kept in range."""
            return torch.clamp(torch.clamp(index, 0, n_in[k] - 1) - region_starts[k], 0, window[k] - 1)

        if self._stream_mode(sub_tensor) == "nearest":
            # floor(c + 0.5) is ITK's nearest -- round half up -- not F.interpolate's floor(o * scale),
            # which is a statement about a size ratio and says nothing once a grid has its own origin.
            picks = [local(torch.floor(axis + 0.5).long(), k) for k, axis in enumerate(coordinates)]
            gathered = sub_tensor[(slice(None), *torch.meshgrid(*picks, indexing="ij"))]
            out = gathered if gathered.is_floating_point() else gathered.type(torch.float32)
        else:
            if not sub_tensor.is_floating_point() or (
                sub_tensor.device.type == "cpu" and sub_tensor.dtype in (torch.float16, torch.bfloat16)
            ):
                work = sub_tensor.type(torch.float32)
            else:
                work = sub_tensor
            taps = []
            for k, axis in enumerate(coordinates):
                base = torch.floor(axis)
                weight = (axis - base).to(work.dtype)
                index = base.long()
                taps.append(((local(index, k), 1 - weight), (local(index + 1, k), weight)))
            out = torch.zeros([work.shape[0], *out_shape[1:]], device=device, dtype=work.dtype)
            for combo in itertools.product(*taps):
                gathered = work
                weight = torch.ones([1] * (ndim + 1), device=device, dtype=work.dtype)
                for k, (index, lam) in enumerate(combo):
                    gathered = gathered.index_select(k + 1, index)
                    shape = [1] * (ndim + 1)
                    shape[k + 1] = -1
                    weight = weight * lam.reshape(shape)
                out += gathered * weight
        # The axes' masks compose by outer product: a sample is inside where every axis of it is.
        mask = inside[0]
        for axis in inside[1:]:
            mask = mask.unsqueeze(-1) & axis
        # Filled while still floating, then cast ONCE: torch implements masked_fill for float dtypes
        # and not for every integer one -- uint16 is a microscope's native dtype and has no fill at
        # all -- so filling after the cast fails on exactly the volumes this stage is built for.
        return out.masked_fill(~mask.unsqueeze(0), self.fill_value).type(sub_tensor.dtype)

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


class ResampleToReference(Resample):
    """Resample a case onto the grid of a declared reference — extent, spacing, origin, direction.

    The stage that makes a cohort foldable. ``ResampleToResolution`` lines up SPACINGS and
    ``ResampleToShape`` lines up EXTENTS, but both leave each case where it was: the cases still
    sit at different origins, so folding them (``Reduce``) can only proceed by looking away
    (``grid: shape_only``). This adopts the reference's grid whole, which is what gives
    ``grid: strict`` — it compares ``Spacing``, ``Origin`` and ``Direction`` — something true to
    check. That is the atlas-template build: bring every case onto one grid, then take the median.

    THE REFERENCE IS A STORED IMAGE, NOT A LIST OF NUMBERS. A grid is fifteen numbers in two axis
    orders at once (``shape`` counts (Z, Y, X); ``Origin``, ``Spacing`` and ``Direction`` are
    physical (x, y, z)), and transcribing them by hand is the mistake this file's history says is
    always made — silently, because a transposed grid resamples perfectly well onto the wrong
    place. Naming an image cannot make that mistake: the header IS the declaration. It is also what
    an atlas loop needs, where round N+1's reference is round N's own output.

    WHAT IT REFUSES, rather than resample onto a grid it cannot honestly reach:

    - a case, or a reference, whose header carries no ``Origin``/``Spacing``/``Direction`` — with no
      geometry there is no physical space to resample IN, and a size ratio would silently stand in
      for one;
    - a reference whose ``Direction`` differs from the case's — the two grids' axes then do not line
      up, and the map is a rotation, not a scale and a shift per axis. ``Canonical`` first;
    - a case that does not meet the reference grid at all — the output would be pure ``fill``, and
      an all-background member is a plausible, wrong contribution to a median.

    Everything else it declares. A case that reaches only part of the reference is legal and
    common — the rest takes ``fill`` — and the plan prints how much of the grid each case covers,
    because "most of this template is fill" is not something to discover in a viewer.
    """

    def __init__(
        self,
        entry: str,
        group: str | None = None,
        dataset: str | None = None,
        fill: float = 0.0,
        inverse: bool = True,
    ) -> None:
        super().__init__(inverse)
        if not entry or not str(entry).strip():
            raise TransformError(
                "'ResampleToReference' needs an 'entry': the stored image whose grid to adopt.",
                "Name it, e.g. ResampleToReference: {entry: 822174, group: Volume}.",
            )
        self.entry = str(entry).strip()
        self.group = group
        self.fill_value = float(fill)
        # A root of its own, exactly as Warp takes one for its field. Left out, the reference is
        # looked up in the run's own dataset_filenames -- the common case, where the grid to adopt
        # is one member of the very cohort being brought together.
        self.reference_dataset: Dataset | None = None
        if dataset is not None and str(dataset).strip():
            filename, _flag, file_format = split_path_spec(str(dataset), default_format="mha")
            self.reference_dataset = Dataset(Path(filename), file_format)
        self._grid: tuple[list[int], np.ndarray, np.ndarray, np.ndarray] | None = None
        # Each case's map, kept from where its own header was in hand. See _recorded().
        self._maps: dict[str, tuple[list[int], list[float], list[float]]] = {}

    # ------------------------------------------------------------------ the reference

    def _roots(self) -> list[Dataset]:
        return [self.reference_dataset] if self.reference_dataset is not None else list(self.datasets)

    def _group_in(self, dataset: Dataset) -> str:
        """Which group of ``dataset`` holds the reference — the declared one, or its only one."""
        if self.group is not None:
            return self.group
        groups = [str(group) for group in dataset.get_group()]
        if len(groups) == 1:
            return groups[0]
        raise TransformError(
            f"'ResampleToReference' cannot tell which group of '{dataset.filename}' holds entry"
            f" '{self.entry}': it has {len(groups)} ({', '.join(sorted(groups)) or 'none'}).",
            "Name it: ResampleToReference: {entry: " + self.entry + ", group: <group>}.",
        )

    def reference_grid(self) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray]:
        """The reference's grid, read from its header once: extent (Z, Y, X) and Origin/Spacing/Direction.

        Headers only, and memoized: a grid is declared once for the stage while a case is one of
        many, so re-reading it per case would be the same answer bought again.
        """
        if self._grid is not None:
            return self._grid
        roots = self._roots()
        if not roots:
            raise TransformError(
                f"'ResampleToReference' has no dataset to look entry '{self.entry}' up in.",
                "Give the stage a root of its own -- ResampleToReference: {entry: "
                + self.entry
                + ", dataset: ./Reference:omezarr} -- or run it in a workflow, which hands its"
                " dataset_filenames to every stage.",
            )
        for dataset in roots:
            group = self._group_in(dataset)
            if dataset.is_dataset_exist(group, self.entry):
                shape, attribute = dataset.get_infos(group, self.entry)
                spatial = [int(extent) for extent in shape[1:]]
                origin, spacing, direction = self._geometry(attribute, len(spatial), f"reference '{self.entry}'")
                self._grid = (spatial, origin, spacing, direction)
                return self._grid
        raise TransformError(
            f"'ResampleToReference' cannot find entry '{self.entry}'"
            + (f" in group '{self.group}'" if self.group is not None else "")
            + f" in {', '.join(str(dataset.filename) for dataset in roots)}.",
            "Check the entry name and its group; the reference is looked up by entry, not by the"
            " case being processed, because one grid serves the whole cohort.",
        )

    @staticmethod
    def _geometry(attribute: Attribute, rank: int, what: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(Origin, Spacing, Direction)`` in physical (x, y, z), or a refusal naming what is missing."""
        missing = [key for key in ("Origin", "Spacing", "Direction") if key not in attribute]
        if missing:
            raise TransformError(
                f"'ResampleToReference' needs the geometry of {what} and its header carries no {', '.join(missing)}.",
                "Resampling onto another grid happens in physical space: without an origin, a"
                " spacing and a direction there is no space to do it in. Use a source whose"
                " geometry is readable (mha, nii, h5, or an OME-Zarr written by KonfAI).",
            )
        origin = np.asarray(attribute.get_np_array("Origin"), dtype=np.float64).ravel()
        spacing = np.asarray(attribute.get_np_array("Spacing"), dtype=np.float64).ravel()
        direction = np.asarray(attribute.get_np_array("Direction"), dtype=np.float64).ravel()
        if origin.size != rank or spacing.size != rank or direction.size != rank * rank:
            raise TransformError(
                f"'ResampleToReference' read a geometry of {what} that does not describe a"
                f" {rank}-dimensional grid: Origin {origin.size}, Spacing {spacing.size},"
                f" Direction {direction.size} (expected {rank}, {rank}, {rank * rank}).",
            )
        if not np.all(spacing > 0.0):
            raise TransformError(
                f"'ResampleToReference' read a Spacing of {spacing.tolist()} on {what}.",
                "A spacing is a physical extent per voxel and must be positive on every axis.",
            )
        return origin, spacing, direction.reshape(rank, rank)

    # ------------------------------------------------------------------ the map

    def grid_map(
        self, name: str, shape: list[int], cache_attribute: Attribute
    ) -> tuple[list[int], list[float], list[float]]:
        """``(target extent, scales, offsets)`` in array order — where each target voxel reads from.

        A target voxel ``o`` is the physical point ``O_ref + D (S_ref * o)``; the source index of
        that point is ``(D^-1 (p - O_src)) / S_src``. With one shared ``D`` the two compose to
        ``scale * o + offset`` per axis, which is the whole map -- and the reason a differing
        ``D`` is refused rather than approximated: it would make the map a rotation, whose source
        region for a target box is a rotated box no per-axis window can describe.
        """
        target, ref_origin, ref_spacing, ref_direction = self.reference_grid()
        where = f"case '{name}'" if name else "the case"
        if len(target) != len(shape):
            raise TransformError(
                f"'ResampleToReference' cannot resample {where}, which has {len(shape)} spatial"
                f" axis/axes, onto reference '{self.entry}', which has {len(target)}.",
            )
        origin, spacing, direction = self._geometry(cache_attribute, len(shape), where)
        if not np.allclose(direction, ref_direction, rtol=0.0, atol=1e-6):
            raise TransformError(
                f"'ResampleToReference' will not resample {where} onto reference '{self.entry}':"
                f" their Direction cosines differ ({direction.ravel().tolist()} against"
                f" {ref_direction.ravel().tolist()}).",
                "The two grids' axes do not line up, so the map between them is a rotation rather"
                " than a scale and a shift per axis. Bring them to a common orientation first"
                " (Canonical), or make the reference one of the cases as they are stored.",
            )
        # (x, y, z) throughout, then reversed once at the end: Origin/Spacing are physical order and
        # the scales/offsets a region window is cut with are array order.
        scale_xyz = ref_spacing / spacing
        offset_xyz = (direction.T @ (ref_origin - origin)) / spacing
        scales = [float(value) for value in scale_xyz[::-1]]
        offsets = [float(value) for value in offset_xyz[::-1]]
        self._refuse_if_disjoint(name, shape, target, scales, offsets)
        if name:
            self._maps[name] = (target, scales, offsets)
        return target, scales, offsets

    def _recorded(self, name: str) -> tuple[list[int], list[float], list[float]]:
        """The map computed for this case back when its own header was in hand.

        A REGION read hands back the REGION's ``Origin`` — honest about what it read, and not the
        case's. A map recomputed from the attribute a streamed region arrives with would therefore
        place the case by the corner of whichever slab is being written, and slide it further with
        every slab; every voxel would still be an interpolation of real data, and nothing about the
        result would look wrong.

        Where a case's placement is known is where its own header is: :meth:`transform_shape`, which
        the manager calls for every case as it is built, before any part of it is read.
        """
        recorded = self._maps.get(name)
        if recorded is None:
            raise TransformError(
                f"'ResampleToReference' was asked for a region of case '{name}' before its grid was established.",
                "This is a bug if it was reached: transform_shape records the map of every case as"
                " its manager is built, and a region is only ever streamed afterwards.",
            )
        return recorded

    def _refuse_if_disjoint(
        self, name: str, shape: list[int], target: list[int], scales: list[float], offsets: list[float]
    ) -> None:
        """Refuse a case that does not meet the reference grid anywhere.

        Its output would be ``fill`` from edge to edge. That is not an error the arithmetic can
        find -- every voxel of it is exactly what was asked for -- so it is one nothing downstream
        would report: a median over the cohort would simply be pulled toward the background by a
        member that contributed no anatomy. Counted from the headers, before a byte is read.
        """
        covered = self.coverage(shape, target, scales, offsets)
        if covered > 0.0:
            return
        where = f"case '{name}'" if name else "the case"
        raise TransformError(
            f"'ResampleToReference' would write {where} as nothing but 'fill': it does not overlap"
            f" reference '{self.entry}' anywhere, so no voxel of the reference grid reads from it.",
            "The two are in different places in physical space. Check that they share a frame"
            " (an acquisition's stage coordinates are not an anatomical one), pick a reference the"
            " cohort actually surrounds, or drop this case with 'subset'.",
        )

    @staticmethod
    def coverage(shape: list[int], target: list[int], scales: list[float], offsets: list[float]) -> float:
        """The fraction of the reference grid that reads from inside the case, from headers alone.

        The product of the per-axis fractions, which is exact: the sampled set is a box, so a target
        voxel is inside exactly where every one of its axes is.
        """
        fraction = 1.0
        for axis, extent in enumerate(target):
            index = np.arange(extent) * scales[axis] + offsets[axis]
            inside = int(np.count_nonzero((index >= -0.5) & (index < shape[axis] - 0.5)))
            fraction *= inside / extent if extent else 0.0
        return float(fraction)

    #: Below this, a case is worth a line in the plan: it reaches only part of the reference grid
    #: and the rest of what it writes is fill. Above it, the note would round to "100.0%" and say
    #: nothing, and a plan that says nothing on every line is one nobody reads.
    _WORTH_SAYING = 0.999

    def plan_note(self, group_dest: str, name: str, shape: list[int], cache_attribute: Attribute) -> str | None:
        del group_dest
        target, scales, offsets = self.grid_map(name, shape, cache_attribute)
        covered = self.coverage(shape, target, scales, offsets)
        if covered >= self._WORTH_SAYING:
            return None
        return (
            f"case '{name}' covers {covered * 100:.1f}% of reference '{self.entry}';"
            f" the rest of what it writes is fill ({self.fill_value:g})"
        )

    # ------------------------------------------------------------------ the contract

    def transform_shape(self, group_src: str, name: str, shape: list[int], cache_attribute: Attribute) -> list[int]:
        return self.grid_map(name, shape, cache_attribute)[0]

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        # Not RESCALE: that kind's map is a size ratio, computed by the dispatcher from the two
        # extents, and this stage's target grid has an origin of its own. REGRID hands the map back
        # to the stage -- stream_region_source below -- which is the only place it is known.
        return PatchLocality(LocalityKind.REGRID)

    def stream_region_source(
        self, target_slices: tuple[slice, ...], source_spatial_shape: list[int], cache_attribute: Attribute
    ) -> list[slice]:
        shape = [int(extent) for extent in source_spatial_shape]
        _target, scales, offsets = self.grid_map("", shape, cache_attribute)
        return Resample.source_window(target_slices, scales, shape, offsets=offsets)

    def stream_region(
        self, name: str, tensor: torch.Tensor, context: RegionContext, cache_attribute: Attribute
    ) -> torch.Tensor:
        # The recorded map, not one read off `cache_attribute`: what arrives here describes the
        # REGION, down to an Origin of its own. See _recorded().
        del cache_attribute
        _target, scales, offsets = self._recorded(name)
        return self.resample_region(
            tensor,
            tuple(context.target),
            [sl.start for sl in context.source],
            scales,
            [int(extent) for extent in context.source_shape],
            offsets,
        )

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        shape = [int(extent) for extent in tensor.shape[1:]]
        target, scales, offsets = self.grid_map(name, shape, cache_attribute)
        # The same sampler the streamed path runs, over one region that happens to be the whole
        # grid: equality between the two paths is then a property of the code, not a claim about it.
        result = self.resample_region(
            tensor, tuple(slice(0, extent) for extent in target), [0] * len(shape), scales, shape, offsets
        )
        self.write_stream_cache_attribute(cache_attribute, shape)
        return result

    def write_stream_cache_attribute(self, cache_attribute: Attribute, source_spatial_shape: list[int]) -> None:
        target, origin, spacing, direction = self.reference_grid()
        # The case now IS the reference grid, header and all -- which is the point, and what
        # Reduce's `grid: strict` reads back. Pushed, not replaced: the source geometry stays
        # underneath for inverse() to pop back to.
        cache_attribute["Spacing"] = spacing
        cache_attribute["Origin"] = origin
        cache_attribute["Direction"] = direction.ravel()
        cache_attribute["Size"] = np.asarray([int(extent) for extent in source_spatial_shape])
        cache_attribute["Size"] = np.asarray(target)

    # ------------------------------------------------------------------ the inverse

    def _inverse_geometry(self, cache_attribute: Attribute) -> list[int]:
        size = super()._inverse_geometry(cache_attribute)
        # The forward pushed Origin and Direction as well as Spacing and Size; the parent pops the
        # two it knows about, and these are the two it does not.
        for key in ("Origin", "Direction"):
            cache_attribute.pop_np_array(key)
        return size

    def inverse_patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        return PatchLocality(
            LocalityKind.WHOLE_VOLUME,
            reason=(
                "a resample onto a reference grid inverts to a resample back off it, and the region"
                " remap for that inverse is not declared -- so a prediction finalize through this"
                " stage assembles the volume. The forward direction streams"
            ),
        )

    def inverse(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        shape = [int(extent) for extent in tensor.shape[1:]]
        target = self._inverse_geometry(cache_attribute)
        # _inverse_geometry restored the case's own geometry, so this is the forward map again --
        # read off the same headers -- and the inverse is that map solved for the other index.
        _spatial, scales, offsets = self.grid_map(name, target, cache_attribute)
        back_scales = [1.0 / scale for scale in scales]
        back_offsets = [-offset / scale for offset, scale in zip(offsets, scales, strict=True)]
        return self.resample_region(
            tensor, tuple(slice(0, extent) for extent in target), [0] * len(shape), back_scales, shape, back_offsets
        )

    def stream_region_target(
        self, target_slices: tuple[slice, ...], source_spatial_shape: list[int], cache_attribute: Attribute
    ) -> list[slice]:
        raise TransformError(
            "'ResampleToReference' declares a whole-volume inverse and has no target region remap.",
            "This is a bug if it was reached: the streamed-write dispatcher should not ask a"
            " WHOLE_VOLUME inverse for its regions.",
        )


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
        _require_simpleitk()
        image = data_to_image(tensor, cache_attribute)

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

        # Resample through SimpleITK so the stored transform is applied in physical space: spacing,
        # direction and the (x, y, z) mm units of the displacement are all honoured. A hand-rolled
        # grid_sample would add the physical (dx, dy, dz) displacement straight onto a (z, y, x)
        # voxel-index grid, transposing the x/z axes and treating millimetres as voxels.
        interpolator = sitk.sitkNearestNeighbor if tensor.dtype == torch.uint8 else sitk.sitkLinear
        resampled = sitk.Resample(image, image, result_transform, interpolator, 0.0)
        data, _ = image_to_data(resampled)
        result = torch.from_numpy(np.ascontiguousarray(data))
        return result.to(torch.uint8) if tensor.dtype == torch.uint8 else result.float()

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
    """Write the chain's state here, and become a source boundary.

    ``scale_factors`` writes an OME-NGFF PYRAMID instead of a single level: ``[4]`` adds a level 1 at
    a quarter of the extent per axis, ``[4, 4]`` a level 2 at a sixteenth. Every reader indexes a
    pyramid BY POSITION -- ``:omezarr@1`` is the second entry, not one named "1" -- so the order is
    the contract, 0 finest. It applies on both write paths: assembled in memory, or region by region,
    where the levels are derived once the last region has landed.

    ``downsample_method`` names how the coarse levels are derived, and its default is
    ``ITKWASM_BIN_SHRINK`` (block averaging), NOT ngff-zarr's own ``ITKWASM_GAUSSIAN``. Measured on a
    real volume, the Gaussian holds a 0.9998 correlation while crushing peak intensity by 20 % --
    exactly the shape of a difference that passes a sanity check and resurfaces months later.
    """

    def __init__(
        self,
        dataset: str,
        group: str | None = None,
        scale_factors: list[int] | None = None,
        downsample_method: str | None = None,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.group = group
        if scale_factors and any(int(factor) < 2 for factor in scale_factors):
            raise TransformError(
                f"'{type(self).__name__}' was given a scale factor below 2 in {list(scale_factors)}.",
                "A pyramid level shrinks its parent, so each factor is 2 or more: scale_factors: [4]"
                " writes one extra level at a quarter of the extent per axis.",
            )
        self.scale_factors = [int(factor) for factor in scale_factors] if scale_factors else None
        self.downsample_method = downsample_method

    # WHOLE_VOLUME by declaration, yet the case may still stream: a Save whose cache exists is a
    # source boundary, and an unsatisfied Save with a streamable prefix is materialized slab by slab
    # first (DatasetManager._materialize_save). Only an unsweepable prefix loads the whole volume.

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        return tensor


class Write(Save):
    """A :class:`Save` that is a deliverable, not a cache.

    Same object, same boundary semantics, one difference that is the point: ``dataset`` has no
    default, so a bare ``Write:`` fails at config time instead of silently writing into the source
    tree (a bare ``Save:`` binds ``dataset`` to nothing and falls back to the manager's own
    dataset). The TRANSFORM workflow plans, resumes and reports on its ``Write`` stages; a ``Save``
    between them stays an opportunistic milestone — never written when a satisfied ``Write``
    downstream lets the boundary skip the whole prefix.
    """

    def __init__(
        self,
        dataset: str,
        group: str | None = None,
        scale_factors: list[int] | None = None,
        downsample_method: str | None = None,
    ) -> None:
        if not dataset or not str(dataset).strip():
            raise TransformError(
                "'Write' needs a destination: its 'dataset' is empty.",
                "Declare where the deliverable lands, e.g. Write: {dataset: ./Out:omezarr}. For an"
                " opportunistic cache next to the source, use 'Save' instead.",
            )
        super().__init__(dataset, group, scale_factors, downsample_method)


class Warp(Transform):
    """Resample a case through a displacement field defined on the same grid.

    ``output(p) = input(p + d(p))``, with ``d`` read from ``field`` in world units. Field and case
    must share a grid: this is the shape update of an atlas build, where the field was solved on the
    very grid it is applied to. Warping ONTO A DIFFERENT grid is a resample as well as a warp, and
    is not this stage.

    THE HALO IS DECLARED, AND VERIFIED. How far a target voxel reaches into the source is the
    displacement itself, so the region this needs is the target enlarged by the largest
    displacement. That bound is then CHECKED against every region actually read: a field that
    exceeds it raises, rather than quietly sampling zeros — which would look like a dark rim around
    the moved anatomy and nothing else.

    ``max_displacement`` is in the same world units as ``Spacing`` (micrometres for these stores),
    and takes ``auto``: a field records its own per-component bound when KonfAI writes it, so
    ``auto`` reads it back from the headers instead of asking you for a number you would have to
    measure. It is the cohort's bound, not this case's — a locality is declared once for the stage,
    while the field is per case — so it over-reads for a gentle case and is never short for a wild
    one. Give a number when you know one and want the tightest halo.

    With no bound at all — ``0.0``, an ``auto`` the headers cannot answer, or a case with no
    ``Spacing`` — the stage declares ``WHOLE_VOLUME`` and says which, because a `Warp` that silently
    costs the whole volume is this stage's expensive failure: the result is right, so nothing looks
    wrong except the peak memory the reader was streaming to control.
    """

    def __init__(
        self,
        field: str,
        group: str | None = None,
        max_displacement: float | str = 0.0,
        interpolation: str = "linear",
    ) -> None:
        super().__init__()
        if not field or not str(field).strip():
            raise TransformError(
                "'Warp' needs a 'field': the displacement field to resample through.",
                "Declare it, e.g. Warp: {field: ./DVF:omezarr, max_displacement: 250.0}.",
            )
        if interpolation not in ("linear", "nearest"):
            raise TransformError(
                f"'Warp' has an unknown interpolation '{interpolation}'.",
                "Use 'linear' for an image or 'nearest' for a label map.",
            )
        filename, _flag, file_format = split_path_spec(str(field), default_format="mha")
        self.field_dataset = Dataset(Path(filename), file_format)
        self.field_group = group
        self.auto_displacement = isinstance(max_displacement, str) and max_displacement.strip().lower() == "auto"
        if isinstance(max_displacement, str) and not self.auto_displacement:
            try:
                max_displacement = float(max_displacement)
            except ValueError:
                raise TransformError(
                    f"'Warp' has a max_displacement of '{max_displacement}', which is neither a number nor 'auto'.",
                    "Give a distance in the case's world units (max_displacement: 250.0), or 'auto'"
                    " to read the bound the fields recorded when they were written.",
                ) from None
        # Per component, in the field's own (x, y, z) order. A scalar bound broadcasts to all three;
        # `auto` fills this from the headers on first use. Per component and not one number, because
        # these grids are anisotropic: one collapsed maximum over-reads the fine axes.
        self.max_displacement = 0.0 if self.auto_displacement else float(max_displacement)
        self._auto_bound: list[float] | None = None
        self._auto_resolved = False
        self.interpolation = interpolation

    def _component_bound(self) -> list[float] | None:
        """The per-component bound this stage warps within, or ``None`` when it has none.

        For ``auto``, the largest bound any field in the group recorded, read from headers alone and
        memoized. If a single entry carries no bound the answer is ``None``: a maximum over the
        others would be a bound for them and a guess for that one, and this number is what sizes the
        region every read depends on.
        """
        if not self.auto_displacement:
            return [self.max_displacement] * 3 if self.max_displacement > 0.0 else None
        if self._auto_resolved:
            return self._auto_bound
        self._auto_resolved = True
        from konfai.utils.ome_zarr import DISPLACEMENT_BOUND_ATTRIBUTE

        bound: list[float] = []
        try:
            group = self._group_for(None)
            # The header reads belong inside: a directory store lists its entries from the filesystem
            # alone, so an unreadable field can only surface here, one entry at a time.
            for entry in self.field_dataset.get_names(group):
                _shape, attribute = self.field_dataset.get_infos(group, entry)
                if DISPLACEMENT_BOUND_ATTRIBUTE not in attribute:
                    return self._auto_bound
                recorded = [float(value) for value in attribute.get_np_array(DISPLACEMENT_BOUND_ATTRIBUTE).ravel()]
                bound = recorded if not bound else [max(a, b) for a, b in zip(bound, recorded, strict=False)]
        except Exception:  # an unreadable field dataset is a whole-volume answer, not a crash
            return self._auto_bound
        if bound and max(bound) > 0.0:
            self._auto_bound = bound
        return self._auto_bound

    def _spacing(self, cache_attribute: Attribute) -> list[float] | None:
        """The case's spacing in array order (z, y, x); ``Spacing`` is stored (x, y, z)."""
        if "Spacing" not in cache_attribute:
            return None
        spacing = [float(value) for value in np.asarray(cache_attribute.get_np_array("Spacing")).ravel()]
        return list(reversed(spacing))

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        bound = self._component_bound()
        if bound is None:
            return PatchLocality(
                LocalityKind.WHOLE_VOLUME,
                reason=(
                    "max_displacement is 'auto' and the fields carry no recorded bound to read"
                    " (KonfAI records one on an OME-Zarr field it writes; other formats and other"
                    " producers do not)"
                    if self.auto_displacement
                    else "no 'max_displacement' is declared"
                )
                + " -- how far this warp reaches into its source is unknown and the region it must"
                " read is unbounded. Declare it in the case's world units (e.g."
                " max_displacement: 250.0) to stream with a halo",
            )
        spacing = self._spacing(cache_attribute)
        if spacing is None:
            return PatchLocality(
                LocalityKind.WHOLE_VOLUME,
                reason=(
                    "the case carries no 'Spacing', so a displacement in world units cannot be"
                    " turned into a halo in voxels"
                ),
            )
        # The bound is per component in (x, y, z); a halo is per array axis in (z, y, x).
        per_axis = list(reversed(bound))[-len(spacing) :] if len(bound) >= len(spacing) else [max(bound)] * len(spacing)
        halo = tuple(
            int(np.ceil(value / extent)) if extent > 0 else 0 for value, extent in zip(per_axis, spacing, strict=False)
        )
        return PatchLocality(LocalityKind.HALO, halo=halo)

    def _group_for(self, name: str | None) -> str:
        if self.field_group is not None:
            return self.field_group
        groups = [str(group) for group in self.field_dataset.get_group()]
        if len(groups) == 1:
            return groups[0]
        where = f"the field for case '{name}'" if name is not None else "the fields"
        raise TransformError(
            f"'Warp' cannot tell which group of '{self.field_dataset.filename}' holds {where}: it has {len(groups)}.",
            "Name it: Warp: {field: ./DVF:omezarr, group: DVF}.",
        )

    def _read_field(self, name: str, region: tuple[slice, ...] | None, channels: int) -> torch.Tensor:
        group = self._group_for(name)
        if region is None:
            data, _attributes = self.field_dataset.read_data(group, name)
        else:
            data, _attributes = self.field_dataset.read_data_slice(group, name, (slice(None), *region))
        field = torch.from_numpy(np.ascontiguousarray(data)).float()
        if field.shape[0] != channels:
            raise TransformError(
                f"The field for case '{name}' has {field.shape[0]} component(s) where the case has"
                f" {channels} spatial axis/axes.",
                "A displacement field carries one component per spatial axis, component-first.",
            )
        return field

    def _sample(self, tensor: torch.Tensor, field: torch.Tensor, spacing: list[float]) -> torch.Tensor:
        """``output(p) = input(p + d(p))`` over the block handed in, sampled with grid_sample.

        The field's components are (x, y, z) where the array axes are (z, y, x) -- the two orders
        meet here, and reversing one of them is a warp that looks plausible and moves the anatomy
        the wrong way.
        """
        extent = list(tensor.shape[1:])
        axes = torch.meshgrid(*[torch.arange(size, dtype=torch.float32) for size in extent], indexing="ij")
        sample = []
        for axis in range(len(extent)):
            # field component for array axis `axis` (z,y,x) is the reversed one (x,y,z)
            displacement = field[len(extent) - 1 - axis] / spacing[axis]
            sample.append(axes[axis] + displacement)
        # grid_sample wants the LAST dim ordered (x, y, z) and coordinates normalised to [-1, 1].
        grid = torch.stack(
            [2.0 * sample[axis] / max(1, extent[axis] - 1) - 1.0 for axis in reversed(range(len(extent)))], dim=-1
        )
        moved = F.grid_sample(
            tensor.unsqueeze(0).float(),
            grid.unsqueeze(0),
            mode="bilinear" if self.interpolation == "linear" else "nearest",
            padding_mode="zeros",
            align_corners=True,
        )
        return moved.squeeze(0).to(tensor.dtype)

    def _check_declared_bound(self, field: torch.Tensor, name: str) -> None:
        """The declaration is a promise about the region that was read; check it against the samples.

        Per component, matching how the halo was derived: a field that stays under the collapsed
        maximum can still exceed the bound on one axis, which is the axis whose halo was too small.
        """
        bound = self._component_bound()
        if bound is None or not field.numel():
            return
        for component in range(field.shape[0]):
            declared = bound[component] if component < len(bound) else max(bound)
            largest = float(field[component].abs().max())
            if largest > declared:
                raise TransformError(
                    f"The field for case '{name}' displaces up to {largest:.3f} on component"
                    f" {component}, beyond the {declared:.3f} this stage sized its region from.",
                    "Raise max_displacement to at least the field's true maximum, or use"
                    " max_displacement: auto: the region read is sized from that number, so a larger"
                    " displacement samples outside what was read.",
                )

    def stream_region(
        self, name: str, tensor: torch.Tensor, context: RegionContext, cache_attribute: Attribute
    ) -> torch.Tensor:
        spacing = self._spacing(cache_attribute)
        if spacing is None:
            return self(name, tensor, cache_attribute)
        field = self._read_field(name, context.source, len(tensor.shape) - 1)
        self._check_declared_bound(field, name)
        return self._sample(tensor, field, spacing)

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        spacing = self._spacing(cache_attribute)
        if spacing is None:
            raise TransformError(
                f"'Warp' needs the case's Spacing to turn a world displacement into voxels, and"
                f" case '{name}' declares none.",
                "Use a source whose geometry is readable (mha, nii, h5 or omezarr written by KonfAI).",
            )
        field = self._read_field(name, None, len(tensor.shape) - 1)
        return self._sample(tensor, field, spacing)


class Reduce(Transform):
    """Fold every case of a group into one volume, at fixed voxel.

    The stage that changes a chain's CARDINALITY: everything before it runs once per case, this
    folds the cases together, everything after it runs once on the result. A chain carrying one is
    driven by the reduction engine rather than the per-case loop, so it is never applied as an
    ordinary transform -- ``__call__`` says so rather than quietly reducing one case to itself.

    ``operator`` is a classpath resolved against :mod:`konfai.data.reduction` (``Mean``, ``Median``,
    ``Concat``, or your own :class:`~konfai.data.reduction.Reduction`). ``output`` is the entry name
    the result is written under, and it is required: a reduction has no case name to inherit, and
    letting it borrow one member's would tie the deliverable to iteration order.

    ``grid`` decides how much agreement between members is demanded before a byte is read:
    ``strict`` compares extents AND geometry (Spacing/Origin/Direction) within ``grid_tolerance``;
    ``shape_only`` compares extents alone, the honest escape hatch for volumes already resampled
    together but carrying approximate headers; ``reference:<case>`` adopts that member's geometry
    for the output and still demands equal extents. Nothing can verify that the members truly live
    in a common space -- only that they claim to, which is why the claim is checked and printed.
    """

    def __init__(
        self,
        operator: str = "Median",
        output: str = "",
        grid: str = "strict",
        grid_tolerance: float = 1e-6,
        provenance: bool = True,
    ) -> None:
        super().__init__()
        if not output or not str(output).strip():
            raise TransformError(
                "'Reduce' needs an 'output': the name its single result is written under.",
                "Declare it, e.g. Reduce: {operator: Median, output: template}. A reduction has no"
                " case name to inherit -- borrowing a member's would tie the deliverable to"
                " iteration order.",
            )
        policy = str(grid)
        reference = policy.split(":", 1)[1].strip() if policy.startswith("reference:") else ""
        if policy not in ("strict", "shape_only") and not reference:
            raise TransformError(
                f"'Reduce' has an unknown grid policy '{grid}'.",
                "Use 'strict' (extents + geometry), 'shape_only' (extents only) or"
                " 'reference:<case>' (adopt that member's geometry) -- 'reference:' alone names no"
                " case.",
            )
        self.operator_classpath = str(operator)
        # Where this stage was configured from, so its operator binds its own parameters from the
        # same mapping -- None when the chain was built in Python, where there is no config to read.
        self.konfai_args: str | None = None
        self.output = str(output).strip()
        self.grid = f"reference:{reference}" if reference else policy
        self.grid_tolerance = float(grid_tolerance)
        self.provenance = bool(provenance)

    def prepare(self, konfai_args: str) -> None:
        self.konfai_args = konfai_args

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        # A cardinality marker, not a per-case stage: the reduction engine SPLITS it out of the chain
        # before any manager is built, so this declaration is only the safety net for a chain that
        # reached the ordinary planner by mistake -- where refusing to stream is the right answer.
        return PatchLocality(LocalityKind.WHOLE_VOLUME)

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        raise TransformError(
            f"'Reduce' (output '{self.output}') was applied to one case, which reduces nothing.",
            "A chain containing Reduce is run by the reduction engine of the TRANSFORM workflow; it"
            " has no meaning as an ordinary per-case transform.",
        )


class ShapeUpdate(Transform):
    """Pull a template toward the mean SHAPE a displacement field carries, leaving its pose alone.

    ``output = -step * (field - t)``, where ``t`` is the field's per-component spatial mean in world
    units. Resampling a template through the result moves it along the cohort's shape residual at
    ANTs' gradient step — the shape update of an atlas build.

    WHY ``t`` IS STRIPPED RATHER THAN APPLIED. A total field maps template coordinates into each
    specimen's OWN world frame, so its spatial mean is dominated by the frame-to-frame offset, not by
    any pose error of the template. Applying it translates the template out of its own grid and clips
    the anatomy; stripping it is what keeps the template anchored.

    Per-component on purpose: a translation has as many parts as the field has components, and the
    pooled mean of all of them describes nothing. That is why this declares ``MeanPerChannel`` — the
    statistic is read once from the stored volume, and the stage is then a value map, so a field of
    any size runs region by region.
    """

    def __init__(self, step: float = 0.25) -> None:
        super().__init__()
        self.step = float(step)

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        return PatchLocality(LocalityKind.GLOBAL_STAT, stat_keys=frozenset({"MeanPerChannel"}))

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        if "MeanPerChannel" not in cache_attribute:
            # Whole volume in hand: the statistic IS this tensor's, so take it rather than demand a
            # seed. Recorded on the case, as the streamed path records its own, so both leave the
            # same state behind and an inverse finds it where it expects.
            cache_attribute["MeanPerChannel"] = tensor.reshape(int(tensor.shape[0]), -1).to(torch.float32).mean(dim=1)
        mean = torch.as_tensor(cache_attribute.get_np_array("MeanPerChannel"), dtype=torch.float32)
        if mean.numel() != tensor.shape[0]:
            raise TransformError(
                f"'ShapeUpdate' was handed {mean.numel()} component mean(s) for a"
                f" {tensor.shape[0]}-component field on '{name}'.",
                "The statistic is read per channel from the stored entry: check that the entry is the"
                " displacement field itself and not a derived volume.",
            )
        centred = tensor.to(torch.float32) - mean.reshape(-1, *([1] * (tensor.dim() - 1)))
        return -self.step * centred


class Expand(Transform):
    """Turn one case into ``nb`` copies, at a declared point of the chain — ``Reduce``'s mirror.

    The stage that changes a chain's cardinality the other way. Everything BEFORE it runs once per
    case (a ``Save`` there is a cache every copy shares); everything AFTER it runs once per copy,
    and a ``Save``/``Write`` there writes one entry per copy.

    It multiplies, and nothing else: the draws are ordinary stages of the chain, declared where they
    apply, so transforms and augmentations interleave freely after the marker::

        transforms:
          Clip:   {min_value: 0.0, max_value: 400.0}   # once per case
          Expand: {nb: 8, pattern: "{name}_r{a:02d}"}
          Rotate: {a_min: -15, a_max: 15}              # a draw, per copy
          ResampleToResolution: {spacing: [2, 2, 2]}   # a transform, per copy
          Brightness: {b_std: 0.2}                     # another draw, per copy
          Write:  {dataset: ./Augmented:omezarr}

    Each draw is parameterised on the grid the stages before it leave, so a shape-changing draw hands
    the next stage its own extent — a chain, exactly like the transforms it sits among.

    ``pattern`` names each copy's entry: ``str.format`` over ``{name}`` (the case) and ``{a}`` (the
    copy ordinal, 1-based). Both tokens are required — without ``{a}`` every copy of a case writes
    over the previous one, without ``{name}`` every case does.

    Every draw after this marker is parameterised from ``(seed, case, which draw this is)`` rather
    than from a shared RNG, whose consumption order two chains cannot agree on. Left unset, ``seed``
    is the run's ``manual_seed``, so an image chain and its mask chain produce matching copies —
    copy ``k`` of the mask carries copy ``k`` of the image's rotation. Set it to decouple one chain
    deliberately: that is the only way to ask two chains for DIFFERENT copies of the same cases.
    """

    def __init__(self, nb: int = 2, pattern: str = "{name}_{a:02d}", seed: int | None = None) -> None:
        super().__init__()
        self.seed = None if seed is None else int(seed)
        if int(nb) < 1:
            raise TransformError(
                f"'Expand' asks for {nb} copies.",
                "A cardinality is at least one: nb: 8 writes eight entries per case.",
            )
        self.nb = int(nb)
        pattern = str(pattern)
        try:
            first, second = pattern.format(name="case", a=1), pattern.format(name="case", a=2)
        except (KeyError, IndexError, ValueError) as error:
            raise TransformError(
                f"'Expand' cannot format its pattern '{pattern}': {error}.",
                "The pattern is a str.format template over {name} and {a}, e.g. pattern: '{name}_r{a:02d}'.",
            ) from error
        if "{name" not in pattern or first == second:
            raise TransformError(
                f"'Expand' has a pattern ('{pattern}') that does not vary over "
                + ("{name}" if "{name" not in pattern else "{a}")
                + ", so its entries would collide.",
                "Use both tokens, e.g. pattern: '{name}_r{a:02d}': {name} keeps cases apart,"
                " {a} keeps a case's copies apart.",
            )
        self.pattern = pattern

    @property
    def draw_seed(self) -> int:
        """The seed the copies are actually drawn from: this marker's own, or the run's."""
        return 0 if self.seed is None else self.seed

    def entry(self, name: str, a: int) -> str:
        """The entry name copy ``a`` of case ``name`` writes under."""
        return self.pattern.format(name=name, a=a)

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        # A cardinality marker, not a per-case stage: the dispatcher splices the copy's own draw at
        # this position and never runs the marker itself. This declaration is only the safety net for
        # a chain that reached a workflow without expansion semantics, where refusing is right.
        return PatchLocality(LocalityKind.WHOLE_VOLUME)

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        raise TransformError(
            "'Expand' was applied to one tensor, which expands nothing.",
            "A chain containing Expand is run by the TRANSFORM workflow, which replaces the marker"
            " with each copy's draw; it has no meaning as an ordinary per-case transform.",
        )


def split_expand(transforms: list[Transform]) -> tuple[list[Transform], "Expand | None", list[Transform]]:
    """A chain around its ``Expand``: what runs once per case, the marker, what runs per copy."""
    for index, transform in enumerate(transforms):
        if isinstance(transform, Expand):
            return list(transforms[:index]), transform, list(transforms[index + 1 :])
    return list(transforms), None, []


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
        config_overrides: list[str] | None = None,
    ):
        super().__init__()
        self.repo_id = repo_id
        self.model_name = model_name
        self.checkpoints_name = checkpoints_name
        self.number_of_tta = number_of_tta
        self.number_of_mc = number_of_mc
        self.per_channel = per_channel
        # Generic 'NAME=VALUE' overrides for the nested run's own config (the --set mechanism), so a caller
        # can tune it without editing the bundle -- not a memory workaround (never shrink a trained
        # segmentation's patch_size: it degrades the result; the allocator hint below keeps memory in check).
        self.config_overrides = config_overrides

    def plan_note(self, group_dest: str, name: str, shape: list[int], cache_attribute: Attribute) -> str | None:
        del name, shape, cache_attribute
        return (
            f"chain '{group_dest}' runs a NESTED KonfAI inference: its GPU and RAM usage live"
            " outside the declared memory_budget, and the plan cannot bound them"
        )

    def infer_entry(self, dataset_path: Path, output_path: Path, gpu: list[int]):
        # Defragment the nested run's CUDA allocator: a heavy model (e.g. a 3D segmentation a metric relies
        # on) can OOM on a large volume purely from reserved-but-unallocated blocks even though the live
        # footprint fits -- so it runs at its trained patch_size, not a shrunk one. setdefault so an
        # explicit caller setting still wins.
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
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
            config_overrides=self.config_overrides,
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

            # Release the caller's cached GPU blocks so the nested run (its own process, same physical
            # device) is not squeezed by memory this process is only holding in reserve.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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
                self._stack_sinks.pop(name).close()
        return self._reduce(tensor)

    def stream_abort(self, name: str) -> None:
        self._stack_buffers.pop(name, None)
        sink = self._stack_sinks.pop(name, None)
        if sink is not None:
            # Abort, not close: finalizing would publish the partial stack under its final name.
            sink.abort()


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
        # already sat on. A margin is in array order and the geometry is in (x, y, z), hence the
        # reversed indexing.
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
