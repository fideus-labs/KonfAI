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


"""A chain stage as the patch engine sees it: its locality, its halo, its pull maps, its draw."""

import contextlib
import hashlib
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, TypeGuard, cast

import torch

from konfai.data.augmentation import DataAugmentation
from konfai.data.transform import (
    LocalityKind,
    PatchLocality,
    RegionContext,
)
from konfai.utils.dataset import Attribute
from konfai.utils.runtime import preserved_rng, seed_all

# How far a halo may reach, as a fraction of the patch it surrounds. See DatasetManager._affords_halo.
_MAX_HALO_FRACTION = 0.5


def _halo_radii(halo: tuple[int, ...], n_axes: int) -> list[int]:
    """The per-axis radius a declared halo means, in array order (one radius covers every axis)."""
    if not halo:
        return [0] * n_axes
    return [halo[k] if k < len(halo) else halo[-1] for k in range(n_axes)]


class Stage(Protocol):
    """One step of what a case's copy is made of, as the patch-streaming dispatcher sees it.

    A copy is its group's transforms followed by the augmentations drawn for it, and streaming asks the
    same three things of every step: what its output depends on, which source region a target patch
    needs, and to run on one tensor. A ``Transform`` answers them as itself. An augmentation is
    parameterised per case and per copy, so it answers them bound to one (see :class:`AugmentedStage`).
    """

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality: ...

    def stream_region_source(
        self,
        name: str,
        target_slices: tuple[slice, ...],
        source_spatial_shape: list[int],
        cache_attribute: Attribute,
    ) -> list[slice]: ...

    def write_stream_cache_attribute(
        self, cache_attribute: Attribute, source_spatial_shape: list[int], name: str = ""
    ) -> None: ...

    def stream_region(
        self, name: str, tensor: torch.Tensor, context: RegionContext, cache_attribute: Attribute
    ) -> torch.Tensor: ...

    def plan_region_reads(self, name: str, contexts: Sequence[RegionContext]) -> None: ...

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor: ...


def _spatial(shape: object) -> list[int]:
    """A folded shape as plain Python ints, whatever the stage returned it as.

    ``transform_shape`` is user-facing: a stage that computes its target grid with torch or numpy
    hands back that library's scalars, and they travel unnoticed until something outside Python asks
    for a real int: a zarr store refusing "Expected an iterable of integers", a header holding
    ``tensor(128)``. Normalising at the fold makes ``shapes`` hold what it is typed as.
    """
    return [int(extent) for extent in cast("Sequence[Any]", shape)]


def _is_draw(stage: object) -> TypeGuard[DataAugmentation]:
    """Whether this chain entry is an augmentation: a stage the manager binds to a copy."""
    return isinstance(stage, DataAugmentation)


@contextlib.contextmanager
def _drawn_from(*key: object) -> Iterator[None]:
    """Seed the global RNGs (random, numpy, torch on every device) from ``key`` for the duration,
    then restore them. Two chains of one case (an image and its mask) hold different draw objects
    and derive the same copies from the same key (the Expand's seed, the case's name, which draw
    this is). ``blake2b``, not ``hash()``: string hashing is salted per process.
    """
    digest = hashlib.blake2b("|".join(str(part) for part in key).encode(), digest_size=4).digest()
    with preserved_rng():
        seed_all(int.from_bytes(digest, "big"))
        yield


def _stage_name(stage: Stage) -> str:
    """What to CALL a stage in a message: a draw's own class, never the adapter that binds it.

    Every refusal and every regime note names the stage that caused it, and 'AugmentedStage' names
    nothing a user wrote: the point of saying which stage refused is lost if the answer is the
    wrapper's name for all of them.
    """
    if isinstance(stage, AugmentedStage):
        return type(stage.augmentation).__name__
    return type(stage).__name__


@dataclass(frozen=True)
class AugmentedStage:
    """One augmentation, bound to the case and the copy whose draw it carries.

    An augmentation is parameterised per (case, copy); binding both makes it answer the Stage
    protocol like a plain transform.
    """

    augmentation: DataAugmentation
    index: int
    a: int

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        return self.augmentation.patch_locality(self.index, self.a, cache_attribute)

    def output_channels(self, channels: int) -> int:
        """Only Mask/Permute reshape, and neither folds the channel axis: a draw keeps it."""
        return channels

    def case_working_multiple(self, name: str) -> float:
        """What this copy's draw allocates beyond its block, in volumes-worth of it.

        From the draw's locality: a REGRID draw resamples through ``grid_sample``, which builds the
        pull box's coordinate grid (one volume per spatial axis) beside the landed block; any other
        draw returns a fresh tensor or a view over a field of its own (one volume). Priced at zero,
        an Expand copy's draws swept under a budget that never heard of their buffers.
        """
        del name
        kind = self.patch_locality(Attribute()).kind
        return 4.0 if kind is LocalityKind.REGRID else 1.0

    def stream_region_source(
        self,
        name: str,
        target_slices: tuple[slice, ...],
        source_spatial_shape: list[int],
        cache_attribute: Attribute,
    ) -> list[slice]:
        # A draw is bound to (case index, copy), not to the case's NAME: the name a region stage
        # needs to find its own per-case map means nothing to an augmentation.
        del name
        return self.augmentation.stream_region_source(self.index, self.a, target_slices, source_spatial_shape)

    def stream_region(
        self, name: str, tensor: torch.Tensor, context: RegionContext, cache_attribute: Attribute
    ) -> torch.Tensor:
        del cache_attribute  # a draw reads no case metadata; the place is what it may need
        return self.augmentation.stream_region(name, self.index, self.a, tensor, context)

    def plan_region_reads(self, name: str, contexts: Sequence[RegionContext]) -> None:
        """A draw reads no companion volume beside its region: nothing to declare."""

    def write_stream_cache_attribute(
        self, cache_attribute: Attribute, source_spatial_shape: list[int], name: str = ""
    ) -> None:
        """An augmentation draws a copy of the case rather than restating its geometry: nothing to record."""

    def stream_shape(self, shape: list[int]) -> list[int]:
        """The spatial shape this copy's draw produces from ``shape`` (its slot in the shape fold)."""
        return self.augmentation.stream_shape(self.index, self.a, shape)

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        return self.augmentation.compute(name, self.index, self.a, tensor)


# The pull maps are callable dataclasses, not closures, because a plan crosses a process boundary:
# the launcher plans every case, `mp.spawn` then pickles the workflow object whole, and a local
# function cannot be pickled. Everything a plan memoizes must survive that trip.


@dataclass(frozen=True)
class _HaloPull:
    """A halo stage's pull map: the region enlarged by the radius, clamped to the volume."""

    radii: list[int]
    shape: list[int]

    def __call__(self, target: tuple[slice, ...]) -> list[slice]:
        return [
            slice(max(0, t.start - radius), min(extent, t.stop + radius))
            for t, radius, extent in zip(target, self.radii, self.shape, strict=False)
        ]


@dataclass(frozen=True)
class _RemapPull:
    """An index-remap stage's pull map, bound to the case and the state the stages before it left.

    The case NAME is bound here because a stage instance is shared by every case of a manager
    (``DatasetManager`` hands the same transforms list to each), while a map read from a stored
    transform or a reference header is per case. A pull that could not say which case it was for
    would build one case's window from another case's map, and a window that is short does not
    raise, it returns the fill.
    """

    remap: Callable[[str, tuple[slice, ...], list[int], Attribute], list[slice]]
    shape: list[int]
    attribute: Attribute
    name: str = ""

    def __call__(self, target: tuple[slice, ...]) -> list[slice]:
        return self.remap(self.name, target, list(self.shape), Attribute(self.attribute))


@dataclass(frozen=True)
class _ReadStagePlan:
    """One chain stage as the composed streamed read runs it: its declared kind, the spatial shapes
    on either side, and (for a region stage) the pull map from a region of its output to the region
    of its input it is computed from, bound to the case state the stages before it left.

    ``run_pull``, when set, is the pull the RUN walks instead: a stage that sizes its windows from
    the data it reads (a declared field) measures there, while ``pull`` stays headers-only for the
    plan's pricing: the estimator must never read a voxel."""

    kind: LocalityKind
    in_shape: tuple[int, ...]
    out_shape: tuple[int, ...]
    pull: Callable[[tuple[slice, ...]], list[slice]] | None
    run_pull: Callable[[tuple[slice, ...]], list[slice]] | None = None

    def region_context(self, source: Sequence[slice], target: Sequence[slice]) -> RegionContext:
        """Where one region of this stage sits: the part of its input read, the part of its output due."""
        return RegionContext(tuple(source), tuple(target), tuple(self.in_shape))


@dataclass(frozen=True)
class PatchReadPlan:
    """Precomputed slicing and padding instructions for one patch request."""

    data_slices: tuple[slice, ...]
    reflect_padding: tuple[int, ...]
    constant_padding: tuple[int, ...]
    concatenate_extend_slice: bool
