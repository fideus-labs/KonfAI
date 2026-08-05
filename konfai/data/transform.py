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
from dataclasses import dataclass, field
from enum import Enum
from multiprocessing import current_process, get_context
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None  # type: ignore[assignment]
import torch.nn.functional as F

from konfai import cuda_visible_devices
from konfai.data.geometry import (
    _GEOMETRY_KEYS,
    AffineMap,
    AffineStage,
    DisplacementStage,
    Grid,
    SpatialStages,
    TransformBound,
    bound_of,
)
from konfai.data.sampling import (
    blend_order,
    gather,
    gather_separable,
    separable_source_index,
    source_index,
    source_window,
)
from konfai.utils.config import _escape_key_component, apply_config, record_given_arguments
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
    - ``REGRID``      -- resample onto another grid: a change of sampling density, of placement, or
      both, possibly through a map. The target is a grid in its own right, so part of it may read
      from outside the source altogether and the source region is no mere scaling of the target's.
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
    REGRID = "regrid"
    SLAB = "slab"
    WHOLE_VOLUME = "whole_volume"

    @property
    def preserves_statistics(self) -> bool:
        """Whether this kind leaves every whole-volume statistic of its input untouched.

        Only a reorientation does: a flip or a permute is a bijection on the voxels, so the multiset of
        values -- and therefore Min/Max/Mean/Std over it -- is exactly the input's. Every other kind may
        map values (``POINTWISE``, ``GLOBAL_STAT``), mix neighbours (``HALO``) or interpolate
        (``REGRID``). This is what decides whether the statistics of the STORED volume are still those
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

    def __init_subclass__(cls, **kwargs: object) -> None:
        # Every stage records its constructor arguments as given, so konfai.api can write the
        # config tree back from live objects -- the binder's mirror, declared once, on the base.
        super().__init_subclass__(**kwargs)
        record_given_arguments(cls)

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
        name: str,
        target_slices: tuple[slice, ...],
        source_spatial_shape: list[int],
        cache_attribute: Attribute,
    ) -> list[slice]:
        """Map a target-patch's spatial slices to the source spatial region to read (region kinds).

        Overridden by the kinds whose source region is an index remap of the target's -- ``ORIENTATION``
        maps it and reorients what it reads, ``CROP`` maps it and is done, ``REGRID`` maps it through
        its own geometry. ``HALO`` is handled generically by the dispatcher, so the base raises for
        any other transform that declares a region kind without providing the remap.

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

    def write_stream_cache_attribute(
        self, cache_attribute: Attribute, source_spatial_shape: list[int], name: str = ""
    ) -> None:
        """Record the geometry a whole-volume ``__call__`` would, given the FULL source shape.

        Called once per case, on the persistent attribute, for the stage that owns a streamed region.
        A transform whose geometry rewrite depends on the volume's EXTENT (a reorientation's new
        origin is the corner it mirrors onto) cannot compute it from a patch, which is all its
        ``__call__`` is handed while streaming: it writes the case-level answer here instead, and the
        patch-local one it wrote on the way is dropped rather than persisted. The base is a no-op --
        a transform that leaves geometry alone has nothing to record.

        ``name`` is the case the fold walks — what a per-case answer (a ``Resample`` whose
        reference follows the case) resolves against; a stage whose answer is case-blind ignores it.
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
        (``Padding``'s crop, ``Resample``'s change of grid) declares itself.
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

    def inverse_stream_cache_attribute(self, cache_attribute: Attribute, source_spatial_shape: list[int]) -> None:
        """State the attribute transition ``inverse`` makes, instead of performing it.

        The write mirror of :meth:`write_stream_cache_attribute`, and it exists because the streamed-
        write dispatcher plans a pipe by walking a ONE-VOXEL probe through it: a stage whose inverse
        restores a whole volume cannot be run on that probe just to learn what it pops. The base is a
        no-op -- an inverse that pops nothing has nothing to state, and one whose transition is cheap
        to perform is simply run.
        """

    def stream_region_inverse(
        self,
        name: str,
        tensor: torch.Tensor,
        context: RegionContext,
        cache_attribute: Attribute,
    ) -> torch.Tensor:
        """Apply ``inverse`` to a region, told WHERE that region sits — the mirror of
        :meth:`Transform.stream_region`.

        ``context.target`` is the region of the inverse's OUTPUT being produced and ``context.source``
        the region of its input on hand. The default delegates to :meth:`inverse`, so an involutive
        index remap (whose pulled block already IS the answer's input) keeps working untouched.
        """
        del context
        return self.inverse(name, tensor, cache_attribute)

    def stream_region_target(
        self,
        name: str,
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


def _seeded_scalar(cache_attribute: Attribute, key: str) -> float:
    """A seeded statistic, as whoever seeded it wrote it: a bare scalar or a one-element array.

    ``float()`` reads the first form and ``get_tensor`` the second.
    """
    try:
        return float(cache_attribute[key])
    except (TypeError, ValueError):
        return float(cache_attribute.get_tensor(key).reshape(-1)[0])


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
            return PatchLocality(
                LocalityKind.WHOLE_VOLUME,
                reason=f"the bounds are read under mask '{self.mask}', a second whole volume; drop the mask to stream",
            )
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
                # Seeded-first, as Normalize reads it: on a streamed path the dispatcher has read
                # the CASE's statistic from disk and the tensor in hand is one region of it --
                # computed here, the bound (and what save_clip_min records) would be the region's.
                if self.mask is None and "StatisticsSeeded" in cache_attribute and "Min" in cache_attribute:
                    min_value = _seeded_scalar(cache_attribute, "Min")
                else:
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
                if self.mask is None and "StatisticsSeeded" in cache_attribute and "Max" in cache_attribute:
                    max_value = _seeded_scalar(cache_attribute, "Max")
                else:
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
            return PatchLocality(
                LocalityKind.WHOLE_VOLUME,
                reason=f"the statistics are taken under mask '{self.mask}', a second whole volume;"
                " drop the mask to stream",
            )
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
        name: str,
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


# ---------------------------------------------------------------------------------------------
# One resample. Two questions: which grid to write on, and what map to write it through.
# ---------------------------------------------------------------------------------------------


class _TargetGrid(ABC):
    """Which grid a resample writes on — the ``to`` half of the question."""

    #: The geometry keys this target cannot be built without. An extent change needs none; a
    #: density change needs the Spacing; adopting another grid needs a real physical space.
    needs: frozenset[str] = frozenset()

    @abstractmethod
    def of(self, source: Grid, name: str) -> Grid:
        """The grid a case stored on ``source`` is written on."""

    def set_datasets(self, datasets: list[Dataset]) -> None:  # noqa: B027 - only a reference has one
        """The run's roots, for a target that has an image of its own to look up."""

    @abstractmethod
    def describe(self) -> str:
        """The target named as a refusal or a plan line names it."""


class _OwnGrid(_TargetGrid):
    """No change of grid: the map moves what the voxels hold, not where they are."""

    def of(self, source: Grid, name: str) -> Grid:
        del name
        return source

    def describe(self) -> str:
        return "the case's own grid"


class _DerivedGrid(_TargetGrid):
    """The case's own grid at another density — a spacing, or a count, and where it sits."""

    def __init__(self, spacing: list[float] | None, shape: list[int] | None, align: str) -> None:
        # A value <= 0 is the KEEP-THIS-AXIS sentinel, normalised to 0 here: that axis takes the
        # source's own density/extent in `of`, so a request rescales only the axes it names.
        self.spacing = None if spacing is None else np.asarray([max(0.0, float(value)) for value in spacing])
        self.shape = None if shape is None else tuple(max(0, int(value)) for value in shape)
        self.align = align
        # A density is meaningless without the density it starts from; a count is not.
        self.needs = frozenset({"Spacing"}) if spacing is not None else frozenset()

    def of(self, source: Grid, name: str) -> Grid:
        where = f"case '{name}'" if name else "the case"
        if self.spacing is not None:
            if self.spacing.size != source.rank:
                raise TransformError(
                    f"'Resample' was given a spacing of {self.spacing.size} value(s) and {where} has"
                    f" {source.rank} spatial axis/axes."
                )
            return source.resampled(spacing_xyz=self.spacing, align=self.align)
        shape = cast("tuple[int, ...]", self.shape)
        if len(shape) != source.rank:
            raise TransformError(
                f"'Resample' was given a shape of {len(shape)} value(s) and {where} has"
                f" {source.rank} spatial axis/axes."
            )
        return source.resampled(size_zyx=shape, align=self.align)

    def describe(self) -> str:
        if self.spacing is not None:
            return f"a spacing of {[float(value) for value in self.spacing]}"
        return f"a shape of {list(cast('tuple[int, ...]', self.shape))}"


class _ReferenceGrid(_TargetGrid):
    """The grid of a STORED image: extent, spacing, origin and direction, read from its header.

    The target that makes a cohort foldable. A spacing lines up densities and a shape lines up
    extents, but both leave each case where it was; this adopts one grid whole, which is what gives
    ``Reduce``'s ``grid: strict`` something true to compare. That is the atlas-template build.

    THE REFERENCE IS AN IMAGE, NOT A LIST OF NUMBERS. A grid is fifteen numbers in two axis orders
    at once, and transcribing them by hand is the mistake this file's history says is always made --
    silently, because a transposed grid resamples perfectly well onto the wrong place. Naming an
    image cannot make it: the header IS the declaration. It is also what an atlas loop needs, where
    round N+1's reference is round N's own output.

    An entry containing ``{case}`` FOLLOWS THE CASE: each case adopts the grid of its own entry --
    ``reference: '{case}', reference_group: DVF`` puts every moved image on its own field's grid,
    which is the registration idiom (a displacement field is defined ON the fixed grid). The
    literal spelling stays one lookup for the whole cohort; a per-case one is one per case,
    headers only either way.
    """

    needs = frozenset(_GEOMETRY_KEYS)

    def __init__(self, entry: str, group: str | None, dataset: str | None) -> None:
        self.entry = str(entry).strip()
        self.group = group
        # A root of its own, or the run's: left out, the grid to adopt is one member of the very
        # cohort being brought together.
        self.dataset: Dataset | None = None
        if dataset is not None and str(dataset).strip():
            filename, _flag, file_format = split_path_spec(str(dataset), default_format="mha")
            self.dataset = Dataset(Path(filename), file_format)
        self.roots: list[Dataset] = []
        self._grids: dict[str, Grid] = {}

    def set_datasets(self, datasets: list[Dataset]) -> None:
        self.roots = list(datasets)

    def _roots(self) -> list[Dataset]:
        return [self.dataset] if self.dataset is not None else list(self.roots)

    def _group_in(self, dataset: Dataset) -> str:
        """Which group of ``dataset`` holds the reference — the declared one, or its only one."""
        if self.group is not None:
            return self.group
        groups = [str(group) for group in dataset.get_group()]
        if len(groups) == 1:
            return groups[0]
        raise TransformError(
            f"'Resample' cannot tell which group of '{dataset.filename}' holds reference"
            f" '{self.entry}': it has {len(groups)} ({', '.join(sorted(groups)) or 'none'}).",
            "Name it: Resample: {reference: " + self.entry + ", reference_group: <group>}.",
        )

    def _entry_for(self, name: str) -> str:
        """The entry to adopt for ``name`` — literal, or the case's own when it says ``{case}``."""
        if "{case}" not in self.entry:
            return self.entry
        if not name:
            raise TransformError(
                f"'Resample' has a per-case reference ('{self.entry}') and no case to resolve it for.",
                "A per-case reference adopts, for each case, the grid of that case's own entry in"
                " reference_group; it has no single grid to answer a caseless probe with.",
            )
        return self.entry.replace("{case}", name)

    def grid(self, name: str = "") -> Grid:
        """The reference's grid, read from its header once per distinct entry.

        Headers only, and memoized by ENTRY: a literal reference is one lookup for the whole
        cohort, a per-case one is one per case -- never the same answer bought again.
        """
        entry = self._entry_for(name)
        cached = self._grids.get(entry)
        if cached is not None:
            return cached
        roots = self._roots()
        if not roots:
            raise TransformError(
                f"'Resample' has no dataset to look reference '{entry}' up in.",
                "Give the stage a root of its own -- Resample: {reference: "
                + entry
                + ", reference_dataset: ./Reference:omezarr} -- or run it in a workflow, which hands"
                " its dataset_filenames to every stage.",
            )
        for dataset in roots:
            group = self._group_in(dataset)
            if dataset.is_dataset_exist(group, entry):
                shape, attribute = dataset.get_infos(group, entry)
                grid = Grid.of([int(extent) for extent in shape[1:]], attribute, f"reference '{entry}'")
                self._grids[entry] = grid
                return grid
        raise TransformError(
            f"'Resample' cannot find reference '{entry}'"
            + (f" in group '{self.group}'" if self.group is not None else "")
            + f" in {', '.join(str(dataset.filename) for dataset in roots)}.",
            "Check the entry name and its group. A literal reference is looked up by entry -- one"
            " grid serves the whole cohort; a '{case}' reference expects every case to have its own"
            " entry in that group.",
        )

    def of(self, source: Grid, name: str) -> Grid:
        grid = self.grid(name)
        if grid.rank != source.rank:
            where = f"case '{name}'" if name else "the case"
            raise TransformError(
                f"'Resample' cannot resample {where}, which has {source.rank} spatial axis/axes,"
                f" onto reference '{self._entry_for(name)}', which has {grid.rank}."
            )
        return grid

    def describe(self) -> str:
        return f"reference '{self.entry}'" + (" (per case)" if "{case}" in self.entry else "")


class Resample(TransformInverse):
    """Resample a case: onto another grid, through a stored map, or both — in one interpolation.

    Every resample in KonfAI is these two questions, and this is the only stage that answers them.

    **Which grid to write on** — at most one of:

    - nothing (the default): the case's own grid. The map moves the anatomy; the voxels stay put.
    - ``spacing``: the same field of view at another density. A component left at ``0`` keeps its axis.
    - ``shape``: the same field of view at a given count. A component left at ``0`` keeps its axis.
    - ``reference``: the grid of a stored image, adopted whole — extent, spacing, origin, direction.
      ``'{case}'`` in the entry follows the case: each case adopts the grid of its OWN entry in
      ``reference_group`` — ``reference: '{case}', reference_group: DVF`` lands every moved image
      on its own field's grid, which is where a displacement field is defined.

    **What map to write it through** — any of, composed in this order:

    - ``field``: a displacement field, read in world units at each TARGET voxel. Its own grid, its
      own spacing: a field solved at 120 um moves a volume stored at 30 um without being upsampled.
    - ``transforms``: transforms stored beside the cases — rigid, affine, BSpline, dense field, or a
      composite of them — mapping GROUP to whether to invert it. The LAST declared is applied first,
      which is SimpleITK's own composite order.

    Left out, the map is the identity and this is a change of grid and nothing else.

    ONE INTERPOLATION, ALWAYS. A grid change and a warp asked for together are composed into a single
    coordinate per target voxel and the source is read once, at the displaced point. Doing it as two
    stages resamples twice, and a volume interpolated twice has lost detail the second pass invented
    no more of -- which is the whole reason an atlas's appearance is rebuilt from native volumes.

    IT STREAMS, and what a region reads is known before a voxel is touched. A rigid or affine map is
    an exact affine, so the source box of a target region is that region's box mapped through it. A
    BSpline and a dense field are values on a grid read through a non-negative kernel that sums to
    one, so the sup-norm of those values bounds the displacement at EVERY point -- a theorem, not a
    sample of the boundary. A field on disk is bounded by ``max_displacement`` instead, which is then
    CHECKED against every region actually read.

    ``align`` decides where a ``spacing`` or a ``shape`` grid SITS, and it is the one silent choice
    in the family -- a quarter of a voxel of anatomy, made differently by every library that offers
    only one of them. ``extent`` keeps the field of view (the outer faces coincide); ``origin`` keeps
    voxel zero's centre where it is. A ``reference`` states its own placement and ignores this.

    WHAT IT REFUSES, rather than resample from a window it cannot size or in a space it does not have:

    - a case whose header carries no ``Origin``/``Spacing``/``Direction`` when the answer needs
      physical space (a reference, a stored transform, a field). A plain ``spacing``/``shape``
      resample does not: with no geometry a world coordinate IS an index, and the ratio is the map;
    - a transform type that decomposes into no bounded map, naming the type;
    - ``invert: true`` on anything but a rigid or affine map: inverting a spline or a field is a
      dense solve over the whole grid, and a field solved per region is not the restriction of the
      field solved once. Store the inverse, or invert it where it is written;
    - a field with no ``max_displacement`` to size its region from;
    - a case that does not meet the target grid anywhere -- judged THROUGH the declared map, so a
      stored rigid bridging two scanner frames is not mistaken for disjointness. The output would
      be ``fill`` from edge to edge, and an all-background member is a plausible, wrong
      contribution to a median.

    A refusal the whole-volume path can serve -- an undeclared field bound, a case with no
    geometry -- declares ``WHOLE_VOLUME`` with its reason and the run proceeds assembled: the chain
    only stops being bounded, and says so in the plan. One that no route can serve -- a map that
    cannot be decoded, read or inverted, or a disjoint case -- refuses as the plan is built,
    before a byte is written. A case reaching only PART of the target grid is legal and common --
    the rest takes ``fill`` -- and the plan prints how much of the grid it covers.
    """

    def __init__(
        self,
        spacing: list[float] | None = None,
        shape: list[int] | None = None,
        reference: str | None = None,
        reference_group: str | None = None,
        reference_dataset: str | None = None,
        transforms: dict[str, bool] | None = None,
        field: str | None = None,
        field_group: str | None = None,
        max_displacement: float | str = 0.0,
        align: str = "extent",
        interpolation: str | None = None,
        fill: float = 0.0,
        inverse: bool = True,
    ) -> None:
        super().__init__(inverse)
        if interpolation is not None and interpolation not in ("linear", "nearest"):
            raise TransformError(
                f"'Resample' has an unknown interpolation '{interpolation}'.",
                "Use 'linear' for an image or 'nearest' for a label map. Left unset, uint8 is taken"
                " for a label map and everything else is interpolated.",
            )
        self.interpolation = interpolation
        self.fill_value = float(fill)
        self._target = self._target_from(spacing, shape, reference, reference_group, reference_dataset, align)
        if transforms is not None and not transforms:
            raise TransformError(
                "'Resample' was given an empty 'transforms'.",
                "Name a group and say whether to invert it -- transforms: {reg: false} -- or drop the"
                " argument: without it the map is the identity and this is a change of grid alone.",
            )
        self.transforms = transforms
        declared = (field is not None and str(field).strip()) or field_group is not None
        if not declared and _is_declared_displacement(max_displacement):
            raise TransformError(
                f"'Resample' was given a max_displacement of {max_displacement!r} and no field to apply.",
                "Name the field the displacement belongs to -- field: ./DVF:omezarr, or field_group:"
                " DVF for fields stored beside the cases -- or drop max_displacement: it sizes the"
                " region a field is read from and means nothing without one.",
            )
        self.displacement: _DisplacementSource | None = (
            _DisplacementSource(field, field_group, max_displacement) if declared else None
        )
        #: Per case: the grid its own header describes. Recorded where that header is in hand --
        #: transform_shape, called for every case as the manager is built. A region read hands back
        #: the REGION's Origin, so a grid rebuilt from what a streamed region arrives with would
        #: place the case by the corner of whichever slab is being written, and slide it further
        #: with every slab; every voxel would still be an interpolation of real data.
        self._grids: dict[str, Grid] = {}
        #: Per case: the geometry keys its header did not carry (see :meth:`Grid.from_header`).
        self._assumed: dict[str, frozenset[str]] = {}
        self._stored: dict[str, SpatialStages] = {}
        self._refusal: str | None = None
        self._probed = False

    @staticmethod
    def _target_from(
        spacing: list[float] | None,
        shape: list[int] | None,
        reference: str | None,
        reference_group: str | None,
        reference_dataset: str | None,
        align: str,
    ) -> _TargetGrid:
        named = [name for name, value in (("spacing", spacing), ("shape", shape), ("reference", reference)) if value]
        if len(named) > 1:
            raise TransformError(
                f"'Resample' was given {' and '.join(named)}, which are three ways to say the same thing.",
                "A resample writes on one grid: give its density (spacing), its extent (shape) or the"
                " image whose grid to adopt (reference) -- and only one of them.",
            )
        if align not in ("extent", "origin"):
            raise TransformError(
                f"'Resample' has an unknown align '{align}'.",
                "Use align: extent to keep the field of view (the outer faces coincide, which is what"
                " KonfAI has always done) or align: origin to keep voxel zero's centre where it is.",
            )
        if reference:
            return _ReferenceGrid(reference, reference_group, reference_dataset)
        if spacing is not None or shape is not None:
            return _DerivedGrid(spacing, shape, align)
        if reference_group is not None or reference_dataset is not None:
            raise TransformError(
                "'Resample' was told where to find a reference but not which one.",
                "Name the entry whose grid to adopt: Resample: {reference: 822174, reference_group: Volume}.",
            )
        return _OwnGrid()

    def set_datasets(self, datasets: list[Dataset]) -> None:
        super().set_datasets(datasets)
        self._target.set_datasets(datasets)
        # A field declared by group alone lives beside the cases, so it looks in the same roots.
        if self.displacement is not None:
            self.displacement.roots = list(datasets)

    # ------------------------------------------------------------------ the two grids

    @property
    def _needs(self) -> frozenset[str]:
        """The geometry keys this configuration cannot be answered without.

        A stored map or a reference grid is applied in physical space and needs all three; a change
        of density needs the density it starts from; a change of extent needs nothing at all. Being
        exact about this is what lets one class serve a headerless array and a real volume.
        """
        if self.transforms is not None or self.displacement is not None:
            return frozenset(_GEOMETRY_KEYS)
        return self._target.needs

    def _record(self, name: str, shape: list[int], cache_attribute: Attribute) -> Grid:
        """The case's own grid, remembered under its name, with what its header left unsaid."""
        where = f"case '{name}'" if name else "the case"
        grid, missing = Grid.from_header(list(shape), cache_attribute, where)
        self._assumed[name] = missing
        if name:
            self._grids[name] = grid
        return grid

    def _source_grid(self, name: str) -> Grid:
        grid = self._grids.get(name)
        if grid is None:
            raise TransformError(
                f"'Resample' was asked for a region of case '{name}' before its grid was established.",
                "This is a bug if it was reached: transform_shape records the grid of every case as"
                " its manager is built, and a region is only ever streamed afterwards.",
            )
        return grid

    def _target_of(self, name: str) -> tuple[Grid, Grid]:
        """``(source, target)`` — needs only what BUILDING the target grid needs.

        Split from :meth:`_grids_of` because the two questions have different answers: the output
        SHAPE of a warp on the case's own grid is the case's own shape, knowable with no geometry at
        all, while SAMPLING it is not. Refusing the shape too would take down the plan of a chain
        whose honest answer is to fall back to the whole volume and say so.
        """
        source = self._source_grid(name)
        absent = self._assumed.get(name, frozenset())
        lacking = [key for key in _GEOMETRY_KEYS if key in absent and key in self._target.needs]
        if lacking:
            raise TransformError(
                f"'Resample' cannot place {self._target.describe()} for case '{name}': its header"
                f" carries no {', '.join(lacking)}.",
                "A density is meaningless without the density it starts from, and another grid"
                " cannot be adopted without a physical space to adopt it in. Use a source whose"
                " geometry is readable (mha, nii, h5, or an OME-Zarr written by KonfAI).",
            )
        return source, self._target.of(source, name)

    def _grids_of(self, name: str) -> tuple[Grid, Grid]:
        source = self._source_grid(name)
        absent = self._assumed.get(name, frozenset())
        lacking = [key for key in _GEOMETRY_KEYS if key in absent and key in self._needs]
        if lacking:
            raise TransformError(
                f"'Resample' needs the geometry of case '{name}' to resample it onto"
                f" {self._target.describe()}, and its header carries no {', '.join(lacking)}.",
                "Resampling onto another grid, or through a stored map, happens in physical space:"
                " without an origin, a spacing and a direction there is no space to do it in. Use a"
                " source whose geometry is readable (mha, nii, h5, or an OME-Zarr written by KonfAI).",
            )
        return source, self._target.of(source, name)

    # ------------------------------------------------------------------ the map

    def _stored_stages(self, name: str) -> SpatialStages:
        """This case's stored transforms, decoded and composed, in application order."""
        if name in self._stored:
            return self._stored[name]
        from konfai.utils.ITK import decode_transform_stages, invert_stages

        _require_simpleitk()
        rank = self._source_grid(name).rank
        stages: list[AffineStage | DisplacementStage] = []
        # Reversed: a CompositeTransform applies its members last-first, and this stage has always
        # built one from `transforms` in declaration order. Decoding normalizes each member to
        # application order, so the declared list is reversed here to mean the same thing it did.
        for group in reversed(list(cast("dict[str, bool]", self.transforms))):
            invert = self.transforms[group] if self.transforms else False
            stored = None
            for dataset in self.datasets:
                if dataset.is_dataset_exist(group, name):
                    stored = dataset.read_transform(group, name)
                    break
            if stored is None:
                raise TransformError(
                    f"'Resample' found no transform for case '{name}' in group '{group}'.",
                    "Every case needs an entry in every group named under 'transforms:'. Check the"
                    " group name, or drop the cases that have no transform with 'subset'.",
                )
            decoded = decode_transform_stages(stored)
            if invert:
                inverted = invert_stages(decoded, rank)
                if inverted is None:
                    raise TransformError(
                        f"'Resample' cannot invert group '{group}' for case '{name}': it is not a rigid or affine map.",
                        "Inverting a spline or a displacement field is a dense solve over the whole"
                        " grid, and a field solved per region is not the restriction of the field"
                        f" solved once. Store the inverse field instead, or set '{group}: false' and"
                        " invert it where it is written.",
                    )
                decoded = inverted
            stages.extend(decoded)
        self._stored[name] = tuple(stages)
        return self._stored[name]

    def _field_stage(self, name: str, region: Grid) -> DisplacementStage:
        """The declared field over ``region``, read on its own grid and no wider.

        The field is evaluated at the TARGET's world points, so the window it needs is that region's
        own world box -- no halo, whatever the displacement is. What the halo sizes is the SOURCE
        read, which is a different question answered by the bound.
        """
        source = cast("_DisplacementSource", self.displacement)
        shape, attribute = source.infos(name)
        spatial = [int(extent) for extent in shape[1:]]
        grid = Grid.of(spatial, attribute, f"the field for case '{name}'")
        window = grid.index_window(region.world_box(), margin=1)
        values = source.read(name, window, len(spatial))
        source.check_bound(values, name)
        return DisplacementStage(grid.sub_grid(window), values.numpy(), order=1)

    def _stages(self, name: str, region: Grid) -> SpatialStages:
        """The whole map over one target region, in application order."""
        stages: list[AffineStage | DisplacementStage] = []
        if self.displacement is not None:
            stages.append(self._field_stage(name, region))
        if self.transforms is not None:
            stages.extend(self._stored_stages(name))
        return tuple(stages)

    def _bound(self, name: str) -> TransformBound:
        """What the map is guaranteed to do — from declarations and coefficients, no voxel read."""
        rank = self._source_grid(name).rank
        folded = TransformBound.exact(AffineMap.identity(rank))
        if self.displacement is not None:
            declared = self.displacement.component_bound()
            if declared is None:
                raise TransformError(self.displacement.undeclared_reason())
            folded = TransformBound.shift(np.asarray(declared[:rank], dtype=np.float64)).after(folded)
        if self.transforms is not None:
            folded = bound_of(self._stored_stages(name), rank).after(folded)
        return folded

    # ------------------------------------------------------------------ the contract

    def transform_shape(self, group_src: str, name: str, shape: list[int], cache_attribute: Attribute) -> list[int]:
        del group_src
        self._record(name, [int(extent) for extent in shape], cache_attribute)
        _source, target = self._target_of(name)
        if name:
            self._require_runnable(name)
            self._refuse_if_disjoint(name)
        return [int(extent) for extent in target.size_zyx]

    def _require_runnable(self, name: str) -> None:
        """Refuse AT PLAN TIME a map neither route can apply.

        A refusal the whole-volume path can serve — an undeclared field bound — stays a locality
        answer, and the run proceeds assembled. A stored transform that cannot be decoded, read or
        inverted fails the streamed path and the whole-volume one at the same line, so declaring
        WHOLE_VOLUME for it would print a plan the run then contradicts by dying per case, after
        bytes are written. ``transform_shape`` runs for every case as the plan is built, which is
        the earliest the failure is knowable and the only place it costs nothing.
        """
        if self.transforms is None:
            return
        try:
            self._stored_stages(name)
        except TransformError:
            raise
        except Exception as error:  # a corrupt store fails both routes; name the case and the cure
            raise TransformError(
                f"'Resample' cannot read the map for case '{name}', so no route can apply it:"
                f" {type(error).__name__}: {error}.",
                "Check the group names under 'transforms:' and that every case has an entry in each.",
            ) from error

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        # The geometry is judged on the attribute in hand -- the case's own header, as the base
        # contract has it -- and not on what the cohort has been seen to carry: one case of a group
        # may lack an Origin while the rest have one, and a declaration made per case is the honest
        # one. A config-time probe hands over an empty header, which reads as a case with none.
        lacking = [key for key in _GEOMETRY_KEYS if key in self._needs and key not in cache_attribute]
        if lacking:
            return PatchLocality(
                LocalityKind.WHOLE_VOLUME,
                reason=(
                    f"resampling onto {self._target.describe()} happens in physical space and this"
                    f" case carries no {', '.join(lacking)}. Use a source whose geometry is readable"
                    " (mha, nii, h5, or an OME-Zarr written by KonfAI)"
                ),
            )
        if not self._probed:
            self._probed = True
            self._refusal = self._probe_cohort()
        if self._refusal is not None:
            return PatchLocality(LocalityKind.WHOLE_VOLUME, reason=self._refusal)
        return PatchLocality(LocalityKind.REGRID)

    def _probe_cohort(self) -> str | None:
        """Whether every case this stage will see is boundable, or the sentence saying which is not.

        The COHORT's answer, not one case's: a locality is declared once for the stage while the
        cases are many, so a group whose entries are not uniformly decodable must fall back for all
        of them rather than for the ones that happen to be planned first. Exceptions are swallowed
        into a reason -- this runs inside the plan, where a raise would take the run down instead of
        costing it the whole-volume path. The GEOMETRY is not judged here; that is per case, and
        :meth:`patch_locality` reads it off the header it is handed.
        """
        if self.transforms is not None and sitk is None:
            return (
                "SimpleITK is not installed, and a stored transform is applied in physical space by"
                " it. Install it (pip install konfai[itk]) to stream this stage"
            )
        # The field's bound is the COHORT's, read from declarations and headers, so it is answered
        # before any case has been seen -- and it is what a config-time probe is really asking.
        if self.displacement is not None and self.displacement.component_bound() is None:
            return self.displacement.undeclared_reason()
        for name in self._grids:
            try:
                self._bound(name)
            except TransformError as error:
                # Both halves of the refusal: the first says what is wrong, the second what to
                # change. A plan line carrying only the first tells the reader nothing to do.
                return " ".join(str(part).strip() for part in error.args if part)
            except Exception:  # an unreadable transform is a whole-volume answer, not a crash
                return (
                    f"the map for case '{name}' could not be read, so what it does to a region is"
                    " unknown. Check the group names under 'transforms:'/'field:' and that every"
                    " case has an entry in each"
                )
        return None

    def stream_region_source(
        self, name: str, target_slices: tuple[slice, ...], source_spatial_shape: list[int], cache_attribute: Attribute
    ) -> list[slice]:
        del source_spatial_shape, cache_attribute
        source, target = self._grids_of(name)
        return list(source_window(target.sub_grid(tuple(target_slices)), source, self._bound(name)))

    def stream_region(
        self, name: str, tensor: torch.Tensor, context: RegionContext, cache_attribute: Attribute
    ) -> torch.Tensor:
        # The recorded grid, not one read off `cache_attribute`: what arrives here describes the
        # REGION, down to an Origin of its own. See _record().
        del cache_attribute
        return self._sample(name, tensor, tuple(context.target), [part.start for part in context.source])

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        shape = [int(extent) for extent in tensor.shape[1:]]
        # Re-recorded on every call: the whole-volume path is handed the case's own evolved header,
        # never a region's, and a grid recorded by an earlier walk may describe the stored volume
        # rather than this stage's true input (a Canonical upstream, a Resample before this one).
        self._record(name, shape, cache_attribute)
        source, target = self._grids_of(name)
        # The same call the streamed path makes, over one region that happens to be the whole grid:
        # equality between the two paths is then a property of the code, not a claim about it.
        whole = tuple(slice(0, extent) for extent in target.size_zyx)
        result = self._sample(name, tensor, whole, [0] * source.rank)
        self.write_stream_cache_attribute(cache_attribute, shape, name)
        return result

    def _sample(
        self, name: str, sub_tensor: torch.Tensor, target_slices: tuple[slice, ...], region_starts: list[int]
    ) -> torch.Tensor:
        source, target = self._grids_of(name)
        region = target.sub_grid(target_slices)
        stages = self._stages(name, region)
        shape, mode = list(source.size_zyx), self._mode(sub_tensor)
        # A map that factorises is read one axis at a time, which is the same arithmetic without the
        # terms that are zero and without a coordinate per voxel -- and it is most maps, because most
        # volumes are stored axis-aligned. The general form is what a rotation or a displacement
        # needs, and the two are bit-identical wherever both apply.
        axes = separable_source_index(region, source, stages, sub_tensor.device)
        if axes is not None:
            order = blend_order(target, source)
            return gather_separable(sub_tensor, axes, region_starts, shape, mode, self.fill_value, order)
        coordinates = source_index(region, source, stages, sub_tensor.device)
        return gather(sub_tensor, coordinates, region_starts, shape, mode, self.fill_value)

    def _mode(self, tensor: torch.Tensor) -> str:
        """``nearest`` or ``linear`` — what a sampler asks before it blends anything.

        A dtype cannot settle this on its own: a CT is int16 and so is nothing else about it. The
        heuristic therefore claims ``uint8`` and nothing more, and ``interpolation`` answers for
        everything it cannot know. Getting it wrong is silent -- two blended labels give a third
        that was in no input, in a volume that is still a label map.
        """
        declared = self.interpolation or ("nearest" if tensor.dtype == torch.uint8 else "linear")
        return "nearest" if declared == "nearest" else "linear"

    def write_stream_cache_attribute(
        self, cache_attribute: Attribute, source_spatial_shape: list[int], name: str = ""
    ) -> None:
        """Push the target grid over the source's, so the case now IS the grid it was written on.

        Pushed and not replaced: the source geometry stays underneath for :meth:`inverse` to pop back
        to, which is the whole of the stack this and ``_inverse_geometry`` share.
        """
        shape = [int(extent) for extent in source_spatial_shape]
        source, missing = Grid.from_header(shape, cache_attribute, f"case '{name}'" if name else "the case")
        target = self._target.of(source, name)
        written = {
            "Spacing": target.spacing_xyz,
            "Origin": target.origin_xyz,
            "Direction": target.direction_xyz.ravel(),
        }
        for key in _GEOMETRY_KEYS:
            # Only over a geometry that was there: a case stored without an Origin is resampled by
            # ratio, and inventing one for it would be a header nobody measured. Key by key, and not
            # all-or-nothing, because ``inverse`` pops exactly what is present -- so what is pushed
            # and what is popped are the same condition, read at the two ends.
            if key not in missing:
                cache_attribute[key] = written[key]
        cache_attribute["Size"] = np.asarray(shape)
        cache_attribute["Size"] = np.asarray([int(extent) for extent in target.size_zyx])

    # ------------------------------------------------------------------ the plan

    #: Below this, a case is worth a line in the plan: it reaches only part of the target grid and
    #: the rest of what it writes is fill. Above it, the note would round to "100.0%" and say
    #: nothing, and a plan that says nothing on every line is one nobody reads.
    _WORTH_SAYING = 0.999

    #: How many probes per axis the coverage estimate uses. Coverage is a volume ratio between two
    #: boxes that a rotation makes a polytope, so it is counted rather than solved; capped because
    #: it is a plan line, not a result.
    _COVERAGE_PROBES = 24

    def coverage(self, name: str) -> float:
        """The fraction of the target grid that reads from inside the recorded case."""
        source, target = self._target_of(name)
        return self._coverage(source, target, self._map_bound(name))

    def _map_bound(self, name: str) -> TransformBound | None:
        """The declared map's bound, for a coverage judged where the samples actually land.

        ``None`` when there is no map — or when nothing bounds it: a coverage that cannot be judged
        must not refuse, and the unboundable configurations carry a fallback reason of their own.
        """
        if self.transforms is None and self.displacement is None:
            return None
        try:
            return self._bound(name)
        except Exception:  # an unreadable or unbounded map answers None, never a crash
            return None

    @classmethod
    def _coverage(cls, source: Grid, target: Grid, bound: TransformBound | None = None) -> float:
        """The fraction of ``target`` that reads from inside ``source``, from geometry alone.

        Judged THROUGH the declared map's affine part: a stored transform is what makes a
        cross-frame pair meet — an MR and a CT in different scanner frames with a rigid bridging
        them — and a coverage judged before applying it would call every such registration
        disjoint. The residual (a spline's or a field's sup-norm) only ever moves a sample by a
        bounded amount, so it widens the inside band rather than moving the lattice. Counted on a
        capped lattice rather than solved, because the sampled set is a box only while the grids
        are axis-aligned and a rotation makes it a polytope.
        """
        axes = [
            np.linspace(0.0, float(extent) - 1.0, min(cls._COVERAGE_PROBES, int(extent)))
            for extent in reversed(target.size_zyx)
        ]
        lattice = np.stack([axis.ravel() for axis in np.meshgrid(*axes, indexing="ij")], axis=-1)
        to_world = target.index_to_world if bound is None else target.index_to_world.then(bound.affine)
        index = to_world.then(source.world_to_index).apply(lattice)
        margin_xyz = (
            np.zeros(source.rank)
            if bound is None
            # A world-space residual box reaches |W2I| @ r in index space, component-wise.
            else np.abs(source.world_to_index.matrix) @ np.asarray(bound.residual_xyz, dtype=np.float64)
        )
        inside = np.ones(index.shape[0], dtype=bool)
        for axis in range(source.rank):
            extent = float(source.size_zyx[source.rank - 1 - axis])
            inside &= (index[:, axis] >= -0.5 - margin_xyz[axis]) & (index[:, axis] < extent - 0.5 + margin_xyz[axis])
        return float(np.count_nonzero(inside)) / float(inside.size)

    def _refuse_if_disjoint(self, name: str) -> None:
        """Refuse a case that does not meet the target grid anywhere.

        Its output would be ``fill`` from edge to edge. That is not an error the arithmetic can
        find -- every voxel of it is exactly what was asked for -- so it is one nothing downstream
        would report: a median over the cohort would simply be pulled toward the background by a
        member that contributed no anatomy. Counted from the headers, before a byte is read.
        """
        if self._target_is_own or self.coverage(name) > 0.0:
            return
        where = f"case '{name}'" if name else "the case"
        raise TransformError(
            f"'Resample' would write {where} as nothing but 'fill': it does not overlap"
            f" {self._target.describe()} anywhere, so no voxel of the target grid reads from it.",
            "The two are in different places in physical space. Check that they share a frame (an"
            " acquisition's stage coordinates are not an anatomical one), pick a target the cohort"
            " actually surrounds, or drop this case with 'subset'.",
        )

    def plan_note(self, group_dest: str, name: str, shape: list[int], cache_attribute: Attribute) -> str | None:
        """What this case covers of the target grid — measured on the header HANDED OVER.

        Not on the grid recorded for the case: the plan asks a stage about its own input, which the
        stages before it decide, and a note answered from the stored header would describe a volume
        that no longer exists by the time this stage sees it. Nothing is recorded here either, for
        the mirror reason -- a question must not move the state a region read depends on.
        """
        del group_dest
        try:
            source, missing = Grid.from_header([int(extent) for extent in shape], cache_attribute, f"case '{name}'")
            if missing & self._target.needs:
                return None
            covered = self._coverage(source, self._target.of(source, name), self._map_bound(name))
        except TransformError:
            return None
        if covered >= self._WORTH_SAYING:
            return None
        return (
            f"case '{name}' covers {covered * 100:.1f}% of {self._target.describe()};"
            f" the rest of what it writes is fill ({self.fill_value:g})"
        )

    # ------------------------------------------------------------------ the inverse

    def _inverse_geometry(self, cache_attribute: Attribute) -> list[int]:
        """Pop the geometry stack the forward pushed and return the size the inverse restores."""
        cache_attribute.pop_np_array("Size")
        size = cache_attribute.pop_np_array("Size")
        for key in _GEOMETRY_KEYS:
            # Present iff the forward pushed it (see write_stream_cache_attribute): popping restores
            # the case's own, and a key the case never had is one this never wrote.
            if key in cache_attribute:
                cache_attribute.pop_np_array(key)
        return [int(extent) for extent in size]

    @staticmethod
    def _grid_from(cache_attribute: Attribute, shape: list[int]) -> Grid:
        if Grid.readable(cache_attribute):
            return Grid.of(shape, cache_attribute, "the case")
        return Grid.identity(shape)

    def _inverse_grids(self, cache_attribute: Attribute, shape: list[int]) -> tuple[Grid, Grid]:
        """``(what the accumulator is on, what to write back onto)`` — both off the pushed stack.

        The forward stacked the source geometry under the target's, so the inverse needs no memory
        of the case: it reads the grid it is holding, pops, and reads the grid it is restoring. A
        copy is popped when the caller is only asking, because a declaration never mutates.
        """
        held = self._grid_from(cache_attribute, [int(extent) for extent in shape])
        restored_shape = self._inverse_geometry(cache_attribute)
        return held, self._grid_from(cache_attribute, restored_shape)

    def inverse_patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        if self.transforms is not None or self.displacement is not None:
            return PatchLocality(
                LocalityKind.WHOLE_VOLUME,
                reason=(
                    "resampling through a map inverts to resampling through its inverse, and that"
                    " inverse is not declared here -- so a prediction finalize through this stage"
                    " assembles the volume. The forward direction streams"
                ),
            )
        try:
            self._inverse_geometry(Attribute(cache_attribute))
        except NameError:
            return PatchLocality(
                LocalityKind.WHOLE_VOLUME,
                reason=(
                    "the grid this stage resampled off is not on the attribute it is being asked to"
                    " invert, so the shape it restores is unknown here. The forward direction streams"
                ),
            )
        return PatchLocality(LocalityKind.REGRID)

    def inverse_transform_shape(self, shape: list[int], cache_attribute: Attribute) -> list[int]:
        try:
            return self._inverse_geometry(Attribute(cache_attribute))
        except NameError:
            return shape

    def inverse_stream_cache_attribute(self, cache_attribute: Attribute, source_spatial_shape: list[int]) -> None:
        del source_spatial_shape
        self._inverse_geometry(cache_attribute)

    def inverse(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        if self._target_is_own and (self.transforms is not None or self.displacement is not None):
            raise TransformError(
                "'Resample' has no inverse here: it changes no grid, so undoing it is undoing its"
                " map -- which is applying a different map, not this one backwards.",
                "Set 'inverse: false' on this stage, or declare a second Resample with the inverse"
                " transforms in the chain that needs it.",
            )
        held, restored = self._inverse_grids(cache_attribute, [int(extent) for extent in tensor.shape[1:]])
        whole = tuple(slice(0, extent) for extent in restored.size_zyx)
        return self._resample_between(restored, held, tensor, whole, [0] * restored.rank)

    def stream_region_inverse(
        self, name: str, tensor: torch.Tensor, context: RegionContext, cache_attribute: Attribute
    ) -> torch.Tensor:
        del name
        held, restored = self._inverse_grids(cache_attribute, [int(extent) for extent in context.source_shape])
        return self._resample_between(
            restored, held, tensor, tuple(context.target), [part.start for part in context.source]
        )

    def stream_region_target(
        self, name: str, target_slices: tuple[slice, ...], source_spatial_shape: list[int], cache_attribute: Attribute
    ) -> list[slice]:
        del name
        held, restored = self._inverse_grids(Attribute(cache_attribute), [int(e) for e in source_spatial_shape])
        identity = TransformBound.exact(AffineMap.identity(restored.rank))
        return list(source_window(restored.sub_grid(tuple(target_slices)), held, identity))

    def _resample_between(
        self,
        target: Grid,
        source: Grid,
        tensor: torch.Tensor,
        target_slices: tuple[slice, ...],
        region_starts: list[int],
    ) -> torch.Tensor:
        """One region of ``target``, read off ``source`` with no map between them."""
        region = target.sub_grid(target_slices)
        shape, mode = list(source.size_zyx), self._mode(tensor)
        axes = separable_source_index(region, source, (), tensor.device)
        if axes is not None:
            order = blend_order(target, source)
            return gather_separable(tensor, axes, region_starts, shape, mode, self.fill_value, order)
        coordinates = source_index(region, source, (), tensor.device)
        return gather(tensor, coordinates, region_starts, shape, mode, self.fill_value)

    @property
    def _target_is_own(self) -> bool:
        return isinstance(self._target, _OwnGrid)


class ResampleToResolution(Resample):
    """Deprecated spelling of ``Resample: {spacing: ...}``."""

    def __init__(self, spacing: list[float] = [1.0, 1.0, 1.0], inverse: bool = True) -> None:
        super().__init__(spacing=spacing, inverse=inverse)


class ResampleToShape(Resample):
    """Deprecated spelling of ``Resample: {shape: ...}``."""

    def __init__(self, shape: list[int] = [100, 256, 256], inverse: bool = True) -> None:
        super().__init__(shape=shape, inverse=inverse)


class ResampleToReference(Resample):
    """Deprecated spelling of ``Resample: {reference: ...}``."""

    def __init__(
        self,
        entry: str,
        group: str | None = None,
        dataset: str | None = None,
        field: str | None = None,
        field_group: str | None = None,
        max_displacement: float | str = 0.0,
        fill: float = 0.0,
        interpolation: str | None = None,
        inverse: bool = True,
    ) -> None:
        if not entry or not str(entry).strip():
            raise TransformError(
                "'ResampleToReference' needs an 'entry': the stored image whose grid to adopt.",
                "Name it, e.g. Resample: {reference: 822174, reference_group: Volume}.",
            )
        super().__init__(
            reference=entry,
            reference_group=group,
            reference_dataset=dataset,
            field=field,
            field_group=field_group,
            max_displacement=max_displacement,
            fill=fill,
            interpolation=interpolation,
            inverse=inverse,
        )


class ResampleTransform(Resample):
    """Deprecated spelling of ``Resample: {transforms: ...}``."""

    def __init__(
        self,
        transforms: dict[str, bool],
        interpolation: str | None = None,
        fill: float = 0.0,
        inverse: bool = False,
    ) -> None:
        if not transforms:
            raise TransformError(
                "'ResampleTransform' needs at least one group of stored transforms to apply.",
                "Name it and say whether to invert it, e.g. Resample: {transforms: {reg: false}}.",
            )
        super().__init__(transforms=transforms, interpolation=interpolation, fill=fill, inverse=inverse)


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
        return PatchLocality(
            LocalityKind.WHOLE_VOLUME,
            reason=f"dim {self.dim} reduces a spatial axis, which spans the whole extent; dim: 0"
            " reduces the channels and streams",
        )

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
        return PatchLocality(
            LocalityKind.WHOLE_VOLUME,
            reason=f"dim {self.dim} reduces a spatial axis, which spans the whole extent; dim: 0"
            " reduces the channels and streams",
        )

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
        return PatchLocality(
            LocalityKind.WHOLE_VOLUME,
            reason=f"dim {self.dim} reduces a spatial axis, which spans the whole extent; dim: 0"
            " reduces the channels and streams",
        )

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


class _DisplacementSource:
    """A displacement field on disk: where it is, how far it reaches, and how to read a region of it.

    :class:`Resample` is its one owner — the family's spellings all resolve to it — so every refusal
    speaks as ``Resample`` and names ``field_group``, the argument the user declared.
    """

    def __init__(
        self,
        field: str | None,
        group: str | None,
        max_displacement: float | str,
    ) -> None:
        # A root of its own, or none: with no ``field`` path the fields are a GROUP of the run's own
        # dataset_filenames, one entry per case — which is how a cohort registered in place stores
        # them, beside the volumes they were solved on.
        self.dataset: Dataset | None = None
        if field is not None and str(field).strip():
            filename, _flag, file_format = split_path_spec(str(field), default_format="mha")
            self.dataset = Dataset(Path(filename), file_format)
        elif group is None:
            raise TransformError(
                "'Resample' has neither a 'field' path nor a group to find the fields in.",
                "Name the store — Resample: {field: ./DVF:omezarr} — or, for fields stored beside"
                " the cases, the group they are in: Resample: {field_group: DVF}.",
            )
        self.group = group
        #: The run's own roots, handed over by the owner; only consulted when there is no path.
        self.roots: list[Dataset] = []
        self.auto = isinstance(max_displacement, str) and max_displacement.strip().lower() == "auto"
        if isinstance(max_displacement, str) and not self.auto:
            try:
                max_displacement = float(max_displacement)
            except ValueError:
                raise TransformError(
                    f"'Resample' has a max_displacement of '{max_displacement}', which is neither a number nor 'auto'.",
                    "Give a distance in the case's world units (max_displacement: 250.0), or 'auto'"
                    " to read the bound the fields recorded when they were written.",
                ) from None
        # Per component, in the field's own (x, y, z) order. A scalar bound broadcasts to all three;
        # `auto` fills this from the headers on first use. Per component and not one number, because
        # these grids are anisotropic: one collapsed maximum over-reads the fine axes.
        self.max_displacement = 0.0 if self.auto else float(max_displacement)
        self._auto_bound: list[float] | None = None
        self._auto_resolved = False

    def component_bound(self) -> list[float] | None:
        """The per-component bound this stage warps within, or ``None`` when it has none.

        For ``auto``, the largest bound any field in the group recorded, read from headers alone and
        memoized. If a single entry carries no bound the answer is ``None``: a maximum over the
        others would be a bound for them and a guess for that one, and this number is what sizes the
        region every read depends on.
        """
        if not self.auto:
            return [self.max_displacement] * 3 if self.max_displacement > 0.0 else None
        if self._auto_resolved:
            return self._auto_bound
        self._auto_resolved = True
        from konfai.utils.ome_zarr import DISPLACEMENT_BOUND_ATTRIBUTE

        bound: list[float] = []
        try:
            group = self.group_for(None)
            # Every root, not the first that answers: this is the COHORT's bound, and a field declared
            # by group alone is looked up beside the cases, which a run may spread over several stores.
            roots = [self.dataset] if self.dataset is not None else list(self.roots)
            # The header reads belong inside: a directory store lists its entries from the filesystem
            # alone, so an unreadable field can only surface here, one entry at a time.
            for root in roots:
                for entry in root.get_names(group):
                    _shape, attribute = root.get_infos(group, entry)
                    if DISPLACEMENT_BOUND_ATTRIBUTE not in attribute:
                        return self._auto_bound
                    recorded = [float(value) for value in attribute.get_np_array(DISPLACEMENT_BOUND_ATTRIBUTE).ravel()]
                    bound = recorded if not bound else [max(a, b) for a, b in zip(bound, recorded, strict=False)]
        except Exception:  # an unreadable field dataset is a whole-volume answer, not a crash
            return self._auto_bound
        if bound and max(bound) > 0.0:
            self._auto_bound = bound
        return self._auto_bound

    def undeclared_reason(self) -> str:
        """Why there is no bound, in the words the plan prints."""
        return (
            "max_displacement is 'auto' and the fields carry no recorded bound to read"
            " (KonfAI records one on an OME-Zarr field it writes; other formats and other"
            " producers do not)"
            if self.auto
            else "no 'max_displacement' is declared"
        ) + (
            " -- how far this reaches into its source is unknown and the region it must read is"
            " unbounded. Declare it in the case's world units (e.g. max_displacement: 250.0) to"
            " stream with a halo"
        )

    def group_for(self, name: str | None) -> str:
        if self.group is not None:
            return self.group
        if self.dataset is None:  # unreachable: a source with no path was given a group to use
            raise TransformError(
                "'Resample' has no field store of its own and no group to look for one in.",
                "Name the group the fields are in: Resample: {field_group: DVF}.",
            )
        groups = [str(group) for group in self.dataset.get_group()]
        if len(groups) == 1:
            return groups[0]
        where = f"the field for case '{name}'" if name is not None else "the fields"
        raise TransformError(
            f"'Resample' cannot tell which group of '{self.dataset.filename}' holds {where}: it has {len(groups)}.",
            "Name it: Resample: {field: ./DVF:omezarr, field_group: DVF}.",
        )

    def _root_for(self, name: str | None) -> Dataset:
        """The store this case's field is in: the declared one, or whichever run root holds it."""
        if self.dataset is not None:
            return self.dataset
        group = self.group_for(name)
        for root in self.roots:
            if name is None or root.is_dataset_exist(group, name):
                return root
        raise TransformError(
            f"'Resample' cannot find a field for case '{name}' in group '{group}' of"
            f" {', '.join(str(root.filename) for root in self.roots) or 'any dataset'}.",
            "A field declared by group alone is looked up beside the cases, one entry per case."
            " Give the store a path of its own instead: Resample: {field: ./DVF:omezarr}.",
        )

    def infos(self, name: str) -> tuple[list[int], Attribute]:
        """The field entry's shape and header, without reading a voxel of it."""
        return self._root_for(name).get_infos(self.group_for(name), name)

    def read(self, name: str, region: tuple[slice, ...] | None, channels: int) -> torch.Tensor:
        group = self.group_for(name)
        root = self._root_for(name)
        if region is None:
            data, _attributes = root.read_data(group, name)
        else:
            data, _attributes = root.read_data_slice(group, name, (slice(None), *region))
        field = torch.from_numpy(np.ascontiguousarray(data)).float()
        if field.shape[0] != channels:
            raise TransformError(
                f"The field for case '{name}' has {field.shape[0]} component(s) where the case has"
                f" {channels} spatial axis/axes.",
                "A displacement field carries one component per spatial axis, component-first.",
            )
        return field

    def check_bound(self, field: torch.Tensor, name: str) -> None:
        """The declaration is a promise about the region that was read; check it against the samples.

        Per component, matching how the halo was derived: a field that stays under the collapsed
        maximum can still exceed the bound on one axis, which is the axis whose halo was too small.
        """
        bound = self.component_bound()
        if bound is None or not field.numel():
            return
        for component in range(field.shape[0]):
            declared = bound[component] if component < len(bound) else max(bound)
            largest = float(field[component].abs().max())
            if largest > declared:
                raise TransformError(
                    f"The field for case '{name}' displaces up to {largest:.3f} on component"
                    f" {component}, beyond the {declared:.3f} 'Resample' sized its region from.",
                    "Raise max_displacement to at least the field's true maximum, or use"
                    " max_displacement: auto: the region read is sized from that number, so a larger"
                    " displacement samples outside what was read.",
                )


def _is_declared_displacement(max_displacement: float | str) -> bool:
    """Whether a ``max_displacement`` was actually asked for, rather than left at its default."""
    if isinstance(max_displacement, str):
        return bool(max_displacement.strip())
    return float(max_displacement) != 0.0


class Warp(Resample):
    """Deprecated spelling of ``Resample: {field: ...}`` — a warp on the case's own grid."""

    def __init__(
        self,
        field: str,
        group: str | None = None,
        max_displacement: float | str = 0.0,
        interpolation: str = "linear",
    ) -> None:
        if not field or not str(field).strip():
            raise TransformError(
                "'Warp' needs a 'field': the displacement field to resample through.",
                "Declare it, e.g. Resample: {field: ./DVF:omezarr, max_displacement: 250.0}.",
            )
        super().__init__(
            field=field,
            field_group=group,
            max_displacement=max_displacement,
            interpolation=interpolation,
            inverse=False,
        )


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
        name: str,
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
        name: str,
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
        name: str,
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
        name: str,
        target_slices: tuple[slice, ...],
        source_spatial_shape: list[int],
        cache_attribute: Attribute,
    ) -> list[slice]:
        # A flip is its own inverse: a written region pulls exactly the region the forward would read.
        return self.stream_region_source(name, target_slices, source_spatial_shape, cache_attribute)

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
            return PatchLocality(
                LocalityKind.WHOLE_VOLUME,
                reason="the case's direction cosines are oblique (or unreadable), so the"
                " reorientation is a resample of the whole volume rather than an index remap",
            )
        return PatchLocality(LocalityKind.ORIENTATION)

    def stream_region_source(
        self,
        name: str,
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

    def write_stream_cache_attribute(
        self, cache_attribute: Attribute, source_spatial_shape: list[int], name: str = ""
    ) -> None:
        # Nothing to state for a case this cannot reorient: no geometry, or a direction that is not
        # 3-D. Its __call__ fails loudly before reaching here; a landing fold must not fail for it.
        del name
        if not Grid.readable(cache_attribute) or cache_attribute.get_np_array("Direction").size != 9:
            return
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
            return PatchLocality(
                LocalityKind.WHOLE_VOLUME,
                reason="the direction this inverse restores is oblique (or not on the attribute),"
                " so the reorientation back is a resample of the whole volume",
            )
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
        name: str,
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
        self.write_stream_cache_attribute(cache_attribute, list(tensor.shape[1:]), name)
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


class Magnitude(Transform):
    """Vector magnitude over the CHANNEL axis: ``[C, ...]`` becomes ``[1, ...]``.

    :class:`Norm`'s channel-first sibling. ``Norm`` folds the trailing axis of a stacked ensemble
    and is whole-volume by construction (a rank change past the streamed write); a stored vector
    volume — a displacement field read as a case — is channel-first, and its magnitude at a voxel
    reads that voxel alone: POINTWISE, so it streams.
    """

    def __init__(self) -> None:
        super().__init__()

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        return PatchLocality(LocalityKind.POINTWISE)

    def __call__(self, name: str, tensors: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        return torch.linalg.norm(tensors.float(), dim=0, keepdim=True)


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
    """Record the volume's Min/Max/Mean/Std on the case, under ``Image*`` keys.

    Streams: the four numbers are exactly what the disk-statistics scan already computes, so a
    streamed chain seeds them (``GLOBAL_STAT``) and each region restates the case's answer instead
    of a region's own.
    """

    _KEYS = (("Min", "ImageMin"), ("Max", "ImageMax"), ("Mean", "ImageMean"), ("Std", "ImageStd"))

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
            return PatchLocality(
                LocalityKind.WHOLE_VOLUME,
                reason="the case carries no 'box' yet; the foreground box is computed and recorded"
                " as the chain is planned, and only a read can find it",
            )
        return PatchLocality(LocalityKind.CROP)

    def stream_region_source(
        self,
        name: str,
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

    def write_stream_cache_attribute(
        self, cache_attribute: Attribute, source_spatial_shape: list[int], name: str = ""
    ) -> None:
        del name
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
        self.write_stream_cache_attribute(cache_attribute, list(tensor.shape[1:]), name)
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
