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
    """How a transform's output at one voxel depends on its input (its patch-locality contract).

    A transform DECLARES its contract via :meth:`Transform.patch_locality`; the patch-streaming
    dispatcher (``konfai.data.patching``) reads the declaration and reads only the source region a
    target patch actually needs, instead of materialising the whole volume.

    - ``POINTWISE``: output voxel depends only on the same voxel (and its channels): read the
      exact patch.
    - ``HALO``: bounded neighbourhood: read the patch enlarged by ``halo`` per axis, crop after.
    - ``ORIENTATION``: flip/permute: read the index-remapped source region.
    - ``CROP``: the source region is the target region TRANSLATED: reading it IS the answer,
      so the stage is not re-applied to it. Unlike a reorientation this drops the voxels outside the
      box, so it is no bijection and the stored volume's statistics are not its output's.
    - ``GLOBAL_STAT``: needs whole-volume stats (``stat_keys`` subset of Min/Max/Mean/Std), obtained
      once from disk and cached: read the exact patch + the cached stat.
    - ``REGRID``: resample onto another grid: a change of sampling density, of placement, or
      both, possibly through a map. The target is a grid in its own right, so part of it may read
      from outside the source altogether and the source region is no mere scaling of the target's.
      The stage owns both halves: it declares the source region a target region pulls
      (:meth:`Transform.stream_region_source`) and interpolates it (:meth:`Transform.stream_region`).
    - ``SLAB``: per-voxel value map, plus a side effect that needs the slabs of the written OUTPUT to
      arrive in order and tile it once (a per-member stack written beside the result): the
      streamed-WRITE dispatcher runs it through :meth:`Transform.stream_slab`; the read dispatcher
      has no such tiling and treats it as ``WHOLE_VOLUME``. A stage that merely needs to know WHERE
      its region sits (a mask read beside the volume) is ``POINTWISE`` and reads the place from
      :meth:`Transform.stream_region`, which both dispatchers hand it.
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
    def is_region(self) -> bool:
        """Whether this kind is a region stage: its read is a remapped region of its source.

        Region stages compose, so the streamed read and write dispatchers both carry any run of them
        between their pointwise stages; the set must stay the same on both sides, or a kind added to
        one would silently fall to the whole-volume path (write) or to a refusal (read) on the other.
        """
        return self in (LocalityKind.HALO, LocalityKind.ORIENTATION, LocalityKind.CROP, LocalityKind.REGRID)

    @property
    def preserves_statistics(self) -> bool:
        """Whether this kind leaves every whole-volume statistic of its input untouched.

        Only a reorientation does: a flip or a permute is a bijection on the voxels, so the multiset of
        values (and therefore Min/Max/Mean/Std over it) is exactly the input's. Every other kind may
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
    a warp onto another grid). ``source_shape`` is the whole extent the source region is cut from: a
    region alone cannot say how far it is from an edge.
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


def stat_seed_valid(upstream: Iterable[PatchLocality]) -> bool:
    """Whether a ``GLOBAL_STAT`` stage's seed still describes its own input.

    The seed is measured before the chain runs (on the stored volume, or on the fold a ``Reduce``
    wrote), so it holds only while every stage between the measurement and the statistic leaves the
    values untouched. Every planner that seeds a statistic must apply this one rule.
    """
    return all(locality.statistics_preserving for locality in upstream)


class Transform(NeedDevice, ABC):
    """Base class for transforms operating on tensors and cached attributes.

    The contract is tiered, and every default is fail-safe, so a stage owes only what its behaviour
    actually needs:

    - **Tier 0 — correct**: implement ``__call__`` alone. The stage runs on the whole volume
      (the default declaration is ``WHOLE_VOLUME``), keeps its shape and channels, and nothing
      silently breaks.
    - **Tier 1 — streaming**: set the :attr:`locality` class attribute (plus :attr:`halo` for a
      bounded neighbourhood), and override :meth:`transform_shape` / :meth:`output_channels` only
      if the stage changes the spatial shape or the channel count. A per-voxel value map is one
      attribute away from streaming.
    - **Tier 2 — streaming-aware**: the method overrides, needed only where the answer depends on
      the case (:meth:`patch_locality` read off the header) or where the stage owns a region's
      geometry or reads beside it (:meth:`stream_region_source`, :meth:`stream_region`,
      :meth:`plan_region_reads`, :meth:`stream_slab`, :meth:`write_stream_cache_attribute`).
    """

    #: Tier-1 declaration: the one :class:`LocalityKind` this stage's contract is, when it is
    #: unconditional. The base :meth:`patch_locality` answers from it; ``None`` (the default) keeps
    #: the fail-safe ``WHOLE_VOLUME``. A declaration that depends on the configuration or the case
    #: overrides the method instead, as does one carrying ``stat_keys`` or a ``reason``.
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
    #: TWO, not zero, for a stage that says nothing. A default of zero meant silence read as "this
    #: stage holds nothing", which is the most optimistic reading available and the one that kills a
    #: run: 33 of the 39 stages KonfAI ships declared nothing, and nine of them held something --
    #: up to fifteen volumes-worth.
    #:
    #: Two because of the dtype a store serves. A CT and an MR are int16, a label map uint8, and a
    #: stage cannot work in those: it materialises a float copy first and then holds its own working
    #: copy on top. Measured on stages of four lines each, the shape someone writes on a first try:
    #: ``(x - x.mean()) / x.std()`` holds 1.00 on float32 and 2.00 on int16, a threshold-and-cast
    #: 1.25 and 2.25, and ``tensor * 2`` nothing at all either way. One covered the float32 reading
    #: of a chain whose source is float32, which is the rarer half of this domain.
    #:
    #: Declaring is therefore for the CHEAP case, and it is the safe direction to be wrong in: a
    #: stage that truly holds nothing says 0.0 and gets taller regions, and a mistake there costs a
    #: shorter region rather than the run. tests/unit/test_transform_working_multiple.py measures
    #: every built-in against the CUDA allocator, on BOTH dtypes, and fails on any that holds more
    #: than it declares.
    working_multiple: float = 2.0

    def case_working_multiple(self, name: str) -> float:
        """:attr:`working_multiple` for ONE case, when what the stage holds is a property of the
        configuration rather than of the class.

        A class attribute cannot answer for a stage whose buffers follow a companion volume: a
        ``Resample`` through a field at the case's own resolution holds three channels of it beside
        its sampling grid, and through a field solved four times coarser it holds a sixteenth of
        that. Both are the same class with the same declaration. Answered from headers, never from
        values: the plan may not read a voxel.
        """
        return float(self.working_multiple)

    #: Whether the stage changes the values it is handed. A stage that records a fact on the case
    #: (Statistics) or writes what passes through (Save) returns its input untouched, so a chain
    #: that drops it reads the same to a model: the PREDICTION chain check ignores it.
    alters_values: bool = True

    def __init_subclass__(cls, **kwargs: object) -> None:
        # Every stage records its constructor arguments as given, so konfai.api can write the
        # config tree back from live objects: the binder's mirror, declared once, on the base.
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

        Identity by default. A stage that WIDENS the axis must say so: the plan sizes a case, and
        every streamed slab, from the channels a chain holds at its widest, and a one-hot priced at
        its source's single channel loaded a 50-class volume onto a 2 GB budget and ran out of
        memory 50 channels later. A stage that narrows may stay silent, that only makes the plan
        conservative.
        """
        return channels

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        """Declare how this transform's output depends on its input, for patch streaming.

        Answered from the transform's own ``__init__`` config and, where the honest answer depends on
        the image, from ``cache_attribute``: the case's SOURCE metadata, as the volume is stored.
        The dispatcher reads the header before any voxel, so a transform whose contract the image
        decides (a reorientation that is only a flip when the direction cosines are axis-aligned, a
        resample whose halo is the case's own scale) can still declare it up front.

        The base answers from the :attr:`locality` attribute where one is set; otherwise the
        default ``WHOLE_VOLUME`` is the safety net: any transform (including third-party custom
        ones) that declares nothing falls to the whole-volume path, so nothing silently breaks.

        An override is bound by three rules:

        - **READ-ONLY.** Never write to ``cache_attribute``. A declaration is made once, for the whole
          case, and what it wrote would be one patch's answer imposed on every other: the
          first-patch-wins bug the streamed paths are built to avoid. The dispatcher hands over a
          private copy, so a write cannot reach the case; it is simply lost.
        - **NO I/O.** Read the attribute already in hand, nothing else. Whether the outside world can
          honour the declaration (are the disk statistics readable, does a mask group exist) is the
          dispatcher's call, and it already makes it.
        - **TOTAL.** Answer for ANY case. The metadata may be absent: the config-time checks probe
          with an empty ``Attribute``, and a group carries only what its writer stored, so a missing
          key must return ``WHOLE_VOLUME``, never raise.
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
        """Map a target-patch's spatial slices to the source spatial region to read (region kinds).

        Overridden by the kinds whose source region is an index remap of the target's: ``ORIENTATION``
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
        """Run this transform on one finalized slab: rows ``region`` of a ``spatial_shape`` volume.

        The streamed-write dispatcher calls this instead of ``__call__`` for a ``SLAB`` declaration:
        the value map is per-voxel, so the default whole-volume call is exact on the slab, but the
        stage's side effect needs the slabs in order, tiling the output exactly once per case, which
        is the one thing this write-side hook promises and :meth:`stream_region` does not.
        """
        del region, spatial_shape
        return self(name, tensor, cache_attribute)

    def prepare(self, konfai_args: str) -> None:
        """Told where this stage's own configuration lives, once, right after it was built.

        The loader knows the subtree a stage read its arguments from; a stage that instantiates
        something ELSE from configuration (an operator named by classpath) cannot know it, and
        without this would have to build that object with no arguments at all. The base holds
        nothing: only a stage with a sub-object of its own overrides it.
        """

    def plan_note(self, group_dest: str, name: str, shape: list[int], cache_attribute: Attribute) -> str | None:
        """Something about this case the plan should say, beyond its regime and its cost.

        A stage can be correct, stream, fit the budget, and still surprise the reader: a cost the
        plan has no column for. The plan is where a run is read before it is trusted, so that is
        where the sentence belongs, rather than in a viewer afterwards.

        Answered from headers on the launcher, per (chain, case), under :meth:`patch_locality`'s
        rules: read-only, no volume read, and an answer for any case. Identical notes are printed
        once, so a note about the STAGE may repeat per case without repeating on the page, while a
        note about the CASE stays one line each.

        The base carries only what the loader recorded: the resolution sentence for a bare name
        both stage namespaces define, so the plan says which class actually ran.
        """
        del group_dest, name, shape, cache_attribute
        return self._ambiguous_name_note

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

        ``name`` is the case the fold walks: what a per-case answer (a ``Resample`` whose
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

        The dispatcher already computes this position (it has to, to know what to read), and by
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

    def plan_region_reads(self, name: str, contexts: Sequence[RegionContext]) -> None:
        """Declare, before a sweep reads its first region, what :meth:`stream_region` will read
        beside the tensor it is handed: ``contexts`` are the ones it will be handed, in that order.

        A stage reading a companion volume per region (a mask) maps each context to the window it
        will read and declares the sequence to the dataset holding it
        (:meth:`~konfai.utils.dataset.Dataset.plan_region_reads`): a store that caches decoded
        blocks then keeps what a later region asks for again and drops what none does. A hint:
        neither what is read nor its values depend on it. The base declares nothing.
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

        The write mirror of :meth:`patch_locality`: a prediction's finalize chain applies transforms
        INVERTED, so the streamed-write gate asks each one about its inverse. ``cache_attribute`` is the
        finalize-time state (the case's attribute as ``inverse`` will receive it, with everything the
        forward pass pushed still stacked on it) under the same three rules (read-only, no I/O, total).

        The default derives from the forward contract where the derivation is safe for any subclass: a
        per-voxel value map inverts to a per-voxel value map, and an index remap inverts to an index
        remap. Every other kind falls to ``WHOLE_VOLUME``: an inverse that is streamable anyway
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
        no-op: an inverse that pops nothing has nothing to state, and one whose transition is cheap
        to perform is simply run.
        """

    def stream_region_inverse(
        self,
        name: str,
        tensor: torch.Tensor,
        context: RegionContext,
        cache_attribute: Attribute,
    ) -> torch.Tensor:
        """Apply ``inverse`` to a region, told WHERE that region sits: the mirror of
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

    def get_transform(self, classpath: str, konfai_args: str, prefer_augmentation: bool = False) -> Transform:
        """Build the stage ``classpath`` names. A bare name resolves in ``konfai.data.transform``,
        then in ``konfai.data.augmentation``: those are stages too, declared exactly where they apply
        (see Expand). ``prefer_augmentation`` reverses the order: past an Expand marker the chain is
        the copies' draws, so a name both packages have (Flip, Mask, Permute) is the draw there."""
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
            # A draw is handed over as itself: the manager binds it to a copy (AugmentedStage) once
            # it knows which copy it is planning, which is the one thing the loader cannot know.
            transform.load(1.0)
            return transform
        return Foreign(transform, classpath)

    @staticmethod
    def _ambiguity_sentence(name: str, winner: str, loser: str, prefer_augmentation: bool) -> str | None:
        """One sentence naming what a bare name resolved to and the qualified spelling of the loser,
        when both stage namespaces define it (Flip, Mask, Permute, Foreign): adding or removing an
        Expand above such a name silently swaps a deterministic transform for a per-copy draw, and
        with default arguments neither the binder nor strict_config would say so."""
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
        """A 'did you mean' over BOTH stage namespaces, so the suggestion is never Python's own
        guess from whichever module happened to fail last."""
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
        # Every resample-ish spelling and Warp are the one Resample stage; difflib alone offers
        # 'EulerTransform' for 'ResampleTransform' and nothing for 'Warp'.
        if "Resample" in name or name == "Warp":
            return "Closest name: 'Resample' (the 1.8 spelling of every resample and Warp). "
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

    # Not declared: what a foreign callable allocates between its input and its output is its own,
    # and nothing here can measure it. The base default is what an unknown stage is priced at.

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
