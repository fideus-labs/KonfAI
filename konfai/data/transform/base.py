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


"""The transform contract: locality, regions, the base classes and the loader."""

import importlib
import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import torch

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None  # type: ignore[assignment]
from konfai.utils.config import _escape_key_component, apply_config, record_given_arguments
from konfai.utils.dataset import Attribute, Dataset
from konfai.utils.errors import TransformError
from konfai.utils.runtime import NeedDevice
from konfai.utils.utils import get_module


class LocalityKind(Enum):
    """How a transform's output at one voxel depends on its input: its patch-locality contract.

    A transform declares it via :meth:`Transform.patch_locality`; the patch-streaming dispatcher
    (``konfai.data.patching``) then reads only the source region a target patch needs.

    - ``POINTWISE``: the output voxel depends only on the same voxel and its channels: the exact patch.
    - ``HALO``: a bounded neighbourhood: the patch enlarged by ``halo`` per axis, cropped after.
    - ``ORIENTATION``: flip/permute: the index-remapped source region.
    - ``CROP``: the source region is the target region translated, so the stage is not re-applied
      to it. It drops voxels, so the stored volume's statistics are not its output's.
    - ``GLOBAL_STAT``: needs whole-volume statistics (``stat_keys``, a subset of Min/Max/Mean/Std),
      read once from disk and cached: the exact patch plus the cached statistic.
    - ``REGRID``: resample onto another grid, possibly through a map. The stage owns both halves:
      the source region a target region pulls (:meth:`Transform.stream_region_source`) and the
      interpolation (:meth:`Transform.stream_region`).
    - ``SLAB``: a per-voxel value map plus a side effect needing the written OUTPUT's slabs in
      order: the streamed-write dispatcher runs :meth:`Transform.stream_slab`; the read dispatcher
      treats it as ``WHOLE_VOLUME``. A stage that only needs to know where its region sits is
      ``POINTWISE`` and reads the place from :meth:`Transform.stream_region`.
    - ``WHOLE_VOLUME``: needs the whole volume: the dispatcher falls back to a full load.
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
    def is_region(self) -> bool:
        """Whether this kind is a region stage: its read is a remapped region of its source.

        Region stages compose; the streamed read and write dispatchers must agree on this set.
        """
        return self in (LocalityKind.HALO, LocalityKind.ORIENTATION, LocalityKind.CROP, LocalityKind.REGRID)

    @property
    def preserves_statistics(self) -> bool:
        """Whether this kind leaves every whole-volume statistic of its input untouched.

        Only a reorientation does: a flip or a permute is a bijection on the voxels. This decides
        whether the stored volume's statistics are still a later transform's own input's.
        """
        return self is LocalityKind.ORIENTATION


@dataclass(frozen=True)
class RegionContext:
    """Where a streamed region sits, for a stage that needs to know.

    ``source`` is the part of the stage's INPUT the tensor covers, ``target`` the part of its OUTPUT
    it must produce; they differ whenever the stage moves or resizes data. ``source_shape`` is the
    whole extent the source region is cut from.
    """

    source: tuple[slice, ...]
    target: tuple[slice, ...]
    source_shape: tuple[int, ...]


@dataclass(frozen=True)
class PatchLocality:
    """A transform's declared patch-locality contract (see :class:`LocalityKind`).

    ``halo`` is the per-spatial-axis neighbourhood radius in array order (Z, Y, X); a length-1
    tuple broadcasts to every axis. ``stat_keys`` are the ``Attribute`` keys a ``GLOBAL_STAT``
    transform reads before running (a subset of ``Min``/``Max``/``Mean``/``Std``). ``stat_channels``
    restricts the statistic to those channels (``Normalize.channels``). ``reason`` is why a
    ``WHOLE_VOLUME`` declaration needs the whole volume; the plan prints it.
    """

    kind: LocalityKind
    halo: tuple[int, ...] = ()
    stat_keys: frozenset[str] = field(default_factory=frozenset)
    stat_channels: list[int] | None = None
    # Overrides the kind-level default: a POINTWISE transform that maps no value (TensorCast to a
    # float dtype) may declare True so a later GLOBAL_STAT can still seed from the stored volume.
    preserves_statistics: bool | None = None
    #: Why this stage needs the whole volume, in the words the plan prints. A stage that is
    #: inherently whole-volume (it changes the tensor's rank) leaves this None; one that is
    #: whole-volume because of its configuration must say so.
    reason: str | None = None

    @property
    def statistics_preserving(self) -> bool:
        if self.preserves_statistics is not None:
            return self.preserves_statistics
        return self.kind.preserves_statistics


def stat_seed_valid(upstream: Iterable[PatchLocality]) -> bool:
    """Whether a ``GLOBAL_STAT`` stage's seed still describes its own input.

    The seed is measured before the chain runs, so it holds only while every stage between the
    measurement and the statistic leaves the values untouched.
    """
    return all(locality.statistics_preserving for locality in upstream)


class Transform(NeedDevice, ABC):
    """Base class for transforms operating on tensors and cached attributes.

    The contract is tiered and every default is fail-safe:

    - **Tier 0**: implement ``__call__`` alone. The stage runs on the whole volume (the default
      declaration is ``WHOLE_VOLUME``) and keeps its shape and channels.
    - **Tier 1**: set the :attr:`locality` class attribute (plus :attr:`halo` for a bounded
      neighbourhood); override :meth:`transform_shape` / :meth:`output_channels` only if the stage
      changes the spatial shape or the channel count.
    - **Tier 2**: the method overrides, where the answer depends on the case (:meth:`patch_locality`)
      or the stage owns a region's geometry or reads beside it (:meth:`stream_region_source`,
      :meth:`stream_region`, :meth:`plan_region_reads`, :meth:`stream_slab`,
      :meth:`write_stream_cache_attribute`).
    """

    #: Tier-1 declaration: the one :class:`LocalityKind` this stage's contract is, when unconditional.
    #: ``None`` (the default) keeps the fail-safe ``WHOLE_VOLUME``. A declaration that depends on the
    #: configuration or the case, or carries ``stat_keys`` or a ``reason``, overrides the method.
    locality: LocalityKind | None = None

    #: Tier-1 companion to a ``HALO`` :attr:`locality`: the per-spatial-axis radius in array order
    #: (a length-1 tuple broadcasts to every axis), exactly as :class:`PatchLocality` carries it.
    halo: tuple[int, ...] = ()

    #: The loader's resolution sentence for a bare name both stage namespaces define, surfaced as
    #: the default :meth:`plan_note`; ``None`` for the unambiguous rest.
    _ambiguous_name_note: str | None = None

    #: What ``__call__`` allocates ON TOP of its input and its output, in volumes-worth of the case.
    #: Every sizing route reads it: the sweep prices a region with it, a reduction charges the member
    #: chain by it, the whole-volume fallback is sized against it.
    #:
    #: Two for a stage that declares nothing: a store serves int16 or uint8, so a stage materialises
    #: a float copy first and holds its own working copy on top. A stage that holds nothing declares
    #: 0.0; over-declaring costs a shorter region, under-declaring costs the run.
    working_multiple: float = 2.0

    def case_working_multiple(self, name: str) -> float:
        """:attr:`working_multiple` for ONE case, when what the stage holds depends on the
        configuration rather than the class (a ``Resample`` through a field at the case's own
        resolution holds more than through one solved coarser). Answered from headers, never values.
        """
        return float(self.working_multiple)

    #: Whether the stage changes the values it is handed. A stage that returns its input untouched
    #: (Statistics, Save) declares False: the PREDICTION chain check ignores it.
    alters_values: bool = True

    def __init_subclass__(cls, **kwargs: object) -> None:
        # Constructor arguments are recorded as given, so konfai.api can write the config tree back.
        super().__init_subclass__(**kwargs)
        record_given_arguments(cls)

    def __init__(self) -> None:
        NeedDevice.__init__(self)
        self.datasets: list[Dataset] = []

    def set_datasets(self, datasets: list[Dataset]):
        self.datasets = datasets

    def read_companion(self, group: str, name: str) -> np.ndarray:
        """The case's ``group`` volume, whole, from whichever dataset holds it."""
        for dataset in self.datasets:
            if dataset.is_dataset_exist(group, name):
                return dataset.read_data(group, name)[0]
        raise ValueError(
            f"Requested group '{group}' is not present in any dataset. Check your dataset group names or configuration."
        )

    def transform_shape(self, group_src: str, name: str, shape: list[int], cache_attribute: Attribute) -> list[int]:
        return shape

    def output_channels(self, channels: int) -> int:
        """How many channels this transform returns for ``channels`` in: the channel-axis twin of
        :meth:`transform_shape`, for the plan's memory arithmetic.

        Identity by default. A stage that widens the axis must say so: the plan sizes a case and
        every streamed slab from the chain's widest channel count. A stage that narrows may stay
        silent.
        """
        return channels

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        """Declare how this transform's output depends on its input, for patch streaming.

        Answered from the transform's own ``__init__`` config and, where the answer depends on the
        image, from ``cache_attribute``: the case's SOURCE metadata, as the volume is stored. The
        base answers from :attr:`locality` where one is set; otherwise ``WHOLE_VOLUME``.

        An override is bound by three rules:

        - **READ-ONLY.** Never write to ``cache_attribute``: the dispatcher hands over a private
          copy, so a write is lost.
        - **NO I/O.** Read the attribute in hand, nothing else. Whether the outside world can honour
          the declaration is the dispatcher's call.
        - **TOTAL.** Answer for ANY case: the config-time checks probe with an empty ``Attribute``,
          so a missing key must return ``WHOLE_VOLUME``, never raise.
        """
        if self.locality is not None:
            return PatchLocality(self.locality, halo=self.halo)
        return PatchLocality(LocalityKind.WHOLE_VOLUME)

    def stream_region_source(
        self,
        name: str,
        target_slices: tuple[slice, ...],
        source_spatial_shape: list[int],
        cache_attribute: Attribute,
    ) -> list[slice]:
        """Map a target patch's spatial slices to the source spatial region to read (region kinds).

        Overridden by the kinds whose source region is an index remap of the target's
        (``ORIENTATION``, ``CROP``, ``REGRID``); ``HALO`` is handled by the dispatcher. The base
        raises for any other transform declaring a region kind. ``cache_attribute`` is the case's
        SOURCE metadata, under the same rules as :meth:`patch_locality`.
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
        """Run this transform on one finalized slab: rows ``region`` of a ``spatial_shape`` volume.

        The streamed-write dispatcher calls this instead of ``__call__`` for a ``SLAB`` declaration:
        the slabs arrive in order and tile the output exactly once per case.
        """
        del region, spatial_shape
        return self(name, tensor, cache_attribute)

    def prepare(self, konfai_args: str) -> None:
        """Told where this stage's own configuration lives, once, right after it was built.

        Only a stage that instantiates something else from configuration (an operator named by
        classpath) overrides it. The base holds nothing.
        """

    def plan_note(self, group_dest: str, name: str, shape: list[int], cache_attribute: Attribute) -> str | None:
        """Something about this case the plan should say, beyond its regime and its cost.

        Answered from headers on the launcher, per (chain, case), under :meth:`patch_locality`'s
        rules. Identical notes are printed once. The base carries the loader's resolution sentence
        for a bare name both stage namespaces define.
        """
        del group_dest, name, shape, cache_attribute
        return self._ambiguous_name_note

    def stream_abort(self, name: str) -> None:
        """Drop whatever ``stream_slab`` holds open for ``name`` after a mid-case failure.

        Called by the streamed-write dispatcher when a case dies between slabs. The base holds nothing.
        """

    def write_stream_cache_attribute(
        self, cache_attribute: Attribute, source_spatial_shape: list[int], name: str = ""
    ) -> None:
        """Record the geometry a whole-volume ``__call__`` would, given the FULL source shape.

        Called once per case, on the persistent attribute, for the stage that owns a streamed
        region: a geometry rewrite that depends on the volume's extent cannot be computed from a
        patch. The patch-local answer ``__call__`` wrote is dropped. The base is a no-op.

        ``name`` is the case the fold walks, for a per-case answer (a ``Resample`` whose reference
        follows the case).
        """

    def stream_region(
        self,
        name: str,
        tensor: torch.Tensor,
        context: RegionContext,
        cache_attribute: Attribute,
    ) -> torch.Tensor:
        """Apply this stage to a region, told WHERE that region sits in the volume.

        Override it when the answer depends on the place: a stage reading a second volume aligned
        with the first (a displacement field, a mask). ``context`` says which part of the input the
        tensor covers and which part of the output is expected back. The default delegates to
        :meth:`__call__`; an override must give the same answer as ``__call__`` on the full volume,
        restricted to ``context.target``.
        """
        del context
        return self(name, tensor, cache_attribute)

    def plan_region_reads(self, name: str, contexts: Sequence[RegionContext]) -> None:
        """Declare, before a sweep reads its first region, the companion windows :meth:`stream_region`
        will read: ``contexts`` are the ones it will be handed, in that order.

        A stage reading a companion volume per region maps each context to its window and declares
        the sequence to the dataset holding it (:meth:`~konfai.utils.dataset.Dataset.plan_region_reads`).
        A hint: neither what is read nor its values depend on it. The base declares nothing.
        """
        del name, contexts

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

        The write mirror of :meth:`patch_locality`, under the same three rules. ``cache_attribute``
        is the finalize-time state: the case's attribute as ``inverse`` will receive it. The default
        keeps a ``POINTWISE`` or ``ORIENTATION`` forward contract; every other kind falls to
        ``WHOLE_VOLUME``, and a streamable inverse declares itself.
        """
        forward = self.patch_locality(cache_attribute)
        if forward.kind in (LocalityKind.POINTWISE, LocalityKind.ORIENTATION):
            return forward
        return PatchLocality(LocalityKind.WHOLE_VOLUME)

    def inverse_transform_shape(self, shape: list[int], cache_attribute: Attribute) -> list[int]:
        """The spatial shape ``inverse`` produces from ``shape`` (write mirror of ``transform_shape``).

        Identity by default; a shape-changing inverse must override it. The streamed-write
        dispatcher trusts it only for the kinds :meth:`inverse_patch_locality` declared streamable.
        """
        return shape

    def inverse_stream_cache_attribute(self, cache_attribute: Attribute, source_spatial_shape: list[int]) -> None:
        """State the attribute transition ``inverse`` makes, instead of performing it.

        The write mirror of :meth:`write_stream_cache_attribute`: the streamed-write dispatcher plans
        a pipe on a one-voxel probe, on which an inverse restoring a whole volume cannot run. The
        base is a no-op.
        """

    def stream_region_inverse(
        self,
        name: str,
        tensor: torch.Tensor,
        context: RegionContext,
        cache_attribute: Attribute,
    ) -> torch.Tensor:
        """Apply ``inverse`` to a region, told WHERE it sits: the mirror of :meth:`Transform.stream_region`.

        ``context.target`` is the region of the inverse's OUTPUT being produced and ``context.source``
        the region of its input on hand. The default delegates to :meth:`inverse`.
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

        The write mirror of :meth:`stream_region_source`: the slices are in the written image, the
        shape is the finalized accumulator's, the answer is the consumed region. ``cache_attribute``
        is the finalize-time state; a transform whose remap depends on what ``inverse`` pops
        accounts for those pops on a copy.
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

    def get_transform(self, classpath: str, konfai_args: str, prefer_augmentation: bool = False) -> Transform:
        """Build the stage ``classpath`` names. A bare name resolves in ``konfai.data.transform``,
        then in ``konfai.data.augmentation``; ``prefer_augmentation`` reverses the order (past an
        Expand marker, a name both packages define is the draw)."""
        first, second = ("konfai.data.augmentation", "konfai.data.transform")
        if not prefer_augmentation:
            first, second = second, first
        module, name = get_module(classpath, first)
        ambiguity: str | None = None
        if ":" not in classpath and hasattr(module, name):
            ambiguity = self._ambiguity_sentence(name, first, second, prefer_augmentation)
            if ambiguity is not None:
                warnings.warn(ambiguity, stacklevel=2)
        if not hasattr(module, name) and ":" not in classpath:
            module, name = get_module(classpath, second)
            if not hasattr(module, name):
                raise TransformError(
                    f"No transform or augmentation is named '{name}'.",
                    self._closest_stage_name(name) + f"A bare name resolves in {first}, then {second};"
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
                "A chain stage is a class: a Transform, a DataAugmentation, or a foreign class to"
                " wrap. Name one, e.g. 'Clip' or 'monai.transforms:ScaleIntensity'.",
            )
        # A key is read as a dotted path, and a classpath naming its module carries dots of its own.
        subtree = f"{konfai_args}.{_escape_key_component(classpath)}"
        transform = apply_config(subtree)(factory)()
        if isinstance(transform, Transform):
            if ambiguity is not None:
                # Surfaced again as the stage's plan_note, so the TRANSFORM plan records which class ran.
                transform._ambiguous_name_note = ambiguity
            transform.prepare(subtree)
            return transform
        if _is_augmentation(transform):
            # A draw is handed over as itself: the manager binds it to a copy once it knows which.
            transform.load(1.0)
            return transform
        return Foreign(transform, classpath)

    @staticmethod
    def _ambiguity_sentence(name: str, winner: str, loser: str, prefer_augmentation: bool) -> str | None:
        """One sentence naming what a bare name resolved to and the qualified spelling of the loser,
        when both stage namespaces define it (Flip, Mask, Permute, Foreign)."""
        if not hasattr(importlib.import_module(loser), name):
            return None
        if prefer_augmentation:
            marker = "past an Expand marker, a bare name is the copies' draw"
        else:
            marker = "before any Expand marker, a bare name is the transform"
        return (
            f"'{name}' resolved to {winner}.{name} ({marker});"
            f" spell '{loser}:{name}' for the {loser.rsplit('.', 1)[-1]}."
        )

    @staticmethod
    def _closest_stage_name(name: str) -> str:
        """A 'did you mean' over BOTH stage namespaces."""
        import difflib

        from konfai.data import augmentation

        candidates = {
            candidate
            for namespace in (vars(importlib.import_module("konfai.data.transform")), vars(augmentation))
            for candidate, obj in namespace.items()
            if isinstance(obj, type)
            and not candidate.startswith("_")
            and any(base.__name__ in ("Transform", "DataAugmentation") for base in obj.__mro__)
        }
        # Every resample-ish spelling and Warp are the one Resample stage.
        if "Resample" in name or name == "Warp":
            return "Closest name: 'Resample' (the 1.8 spelling of every resample and Warp). "
        closest = difflib.get_close_matches(name, sorted(candidates), n=1)
        return f"Closest name: '{closest[0]}'. " if closest else ""


def _is_augmentation(candidate: object) -> bool:
    """Whether this object is a KonfAI draw, without importing the augmentation module here
    (``konfai.data.augmentation`` imports this module)."""
    return any(base.__name__ == "DataAugmentation" for base in type(candidate).__mro__)


class Foreign(Transform):
    """A transform from another framework, as the loader hands it over.

    Name the class where a transform goes and its arguments under it::

        transforms:
          monai.transforms:ScaleIntensity:
            minv: 0.0
            maxv: 1.0

    The class must be callable on one tensor and return the transformed tensor (torchvision's
    transforms, TorchIO's and MONAI's array transforms). MONAI's dictionary transforms
    (``ScaleIntensityd``) take a dictionary of keys: name the array class.

    The class must be DETERMINISTIC: a transform runs on each group of a case in turn, so a random
    one would misalign the label from the image. Name it under the augmentations instead.

    It reads the whole volume, must return the shape it was given (checked), and leaves geometry
    as it stands. A class that resamples, crops or reorients needs a ``Transform`` subclass.
    """

    # working_multiple is not declared: a foreign callable's allocations cannot be measured here.

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
