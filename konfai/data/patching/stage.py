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

    Streaming asks three things of every step: what its output depends on, which source region a
    target patch needs, and to run on one tensor. A ``Transform`` answers as itself; an augmentation
    answers bound to one case and copy (:class:`AugmentedStage`).
    """

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality: ...

    def output_channels(self, channels: int) -> int: ...

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
    """A folded shape as plain Python ints: a stage's ``transform_shape`` may hand back torch or numpy
    scalars."""
    return [int(extent) for extent in cast("Sequence[Any]", shape)]


def _is_draw(stage: object) -> TypeGuard[DataAugmentation]:
    """Whether this chain entry is an augmentation: a stage the manager binds to a copy."""
    return isinstance(stage, DataAugmentation)


@contextlib.contextmanager
def _drawn_from(*key: object) -> Iterator[None]:
    """Seed the global RNGs (random, numpy, torch on every device) from ``key`` for the duration, then
    restore them. Two chains of one case derive the same copies from the same key. ``blake2b``, not
    ``hash()``: string hashing is salted per process."""
    digest = hashlib.blake2b("|".join(str(part) for part in key).encode(), digest_size=4).digest()
    with preserved_rng():
        seed_all(int.from_bytes(digest, "big"))
        yield


def _stage_name(stage: Stage) -> str:
    """What to CALL a stage in a message: a draw's own class, never the adapter that binds it."""
    if isinstance(stage, AugmentedStage):
        return type(stage.augmentation).__name__
    return type(stage).__name__


@dataclass(frozen=True)
class AugmentedStage:
    """One augmentation, bound to the case and the copy whose draw it carries, so it answers the Stage
    protocol like a plain transform."""

    augmentation: DataAugmentation
    index: int
    a: int

    @property
    def selected(self) -> bool:
        """Whether the draw selected this copy: for one it did not, the draw is the identity."""
        return self.a in self.augmentation.who_index.get(self.index, ())

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        return self.augmentation.patch_locality(self.index, self.a, cache_attribute)

    def output_channels(self, channels: int) -> int:
        """Only Mask/Permute reshape, and neither folds the channel axis: a draw keeps it."""
        return channels

    def case_working_multiple(self, name: str) -> float:
        """What this copy's draw allocates beyond its block, in volumes-worth of it: a REGRID draw builds
        the pull box's coordinate grid, whose build peaks at ten volumes (measured on a 100x512x512
        case, in-plane and oblique alike, the walk's slabs under it), any other draw one volume."""
        del name
        kind = self.patch_locality(Attribute()).kind
        return 10.0 if kind is LocalityKind.REGRID else 1.0

    def stream_region_source(
        self,
        name: str,
        target_slices: tuple[slice, ...],
        source_spatial_shape: list[int],
        cache_attribute: Attribute,
    ) -> list[slice]:
        # A draw is bound to (case index, copy), not to the case's NAME.
        del name
        return self.augmentation.stream_region_source(self.index, self.a, target_slices, source_spatial_shape)

    def stream_region(
        self, name: str, tensor: torch.Tensor, context: RegionContext, cache_attribute: Attribute
    ) -> torch.Tensor:
        return self.augmentation.stream_region(name, self.index, self.a, tensor, context, cache_attribute)

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
        return self.augmentation.compute(name, self.index, self.a, tensor, cache_attribute)


# The pull maps are callable dataclasses, not closures: a plan is pickled whole by `mp.spawn`.


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

    The case NAME is bound: a stage instance is shared by every case of a manager, while a map read
    from a stored transform or a reference header is per case.
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

    ``run_pull``, when set, is the pull the RUN walks instead; ``pull`` stays headers-only for the
    plan's pricing: the estimator must never read a voxel."""

    kind: LocalityKind
    in_shape: tuple[int, ...]
    out_shape: tuple[int, ...]
    pull: Callable[[tuple[slice, ...]], list[slice]] | None
    run_pull: Callable[[tuple[slice, ...]], list[slice]] | None = None
    #: The statistics this stage is seeded with: what a whole-volume pass measured on its own input, or
    #: the store's when they still describe it.
    measured: tuple[tuple[str, str], ...] = ()
    #: The seed's keys the whole-volume route leaves in the scope once the stage ran.
    kept: frozenset[str] = frozenset()

    def seed(self, scope: Attribute) -> dict[str, int]:
        """Push the seed right before the stage runs, and say how deep each key then stacks: the
        whole-volume route leaves each stage's statistic on top of the scope when that stage runs, and a
        scope seeded once for the chain would hand one stage's number to every stage reading the same
        key."""
        pushed: dict[str, int] = {}
        for key, value in self.measured:
            scope[key] = value
            pushed[key] = scope._count_key(key)
        return pushed

    def unseed(self, scope: Attribute, pushed: dict[str, int]) -> None:
        """Leave the scope, once the stage ran, as the whole-volume route leaves it: a stage that
        recorded the key itself (a saving Clip) stacked the seed's value twice, and one that leaves it
        under another name or not at all (Statistics, a bound not saved) leaves the seed to take back."""
        for key, depth in pushed.items():
            if scope._count_key(key) > depth or key not in self.kept:
                scope.pop(key)

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
