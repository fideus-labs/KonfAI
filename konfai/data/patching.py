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

"""Patch extraction, accumulation, and patch-combination helpers for KonfAI."""

import contextlib
import copy
import hashlib
import random
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any, Protocol, TypeGuard, cast

import numpy as np
import torch
import torch.nn.functional as F

from konfai.data.augmentation import DataAugmentation, DataAugmentationsList
from konfai.data.transform import (
    Expand,
    LocalityKind,
    PatchLocality,
    RegionContext,
    Save,
    Transform,
    split_expand,
    stat_seed_valid,
)
from konfai.utils.config import apply_config, config
from konfai.utils.dataset import Attribute, Dataset, DataStream
from konfai.utils.errors import ConfigError, PatchError
from konfai.utils.utils import (
    SUPPORTED_FORMATS,
    OverlapSpec,
    best_sweep_axis,
    concretize_patch_size,
    env_flag,
    free_axis_rounding,
    get_module,
    get_patch_slices_from_shape,
    resolve_overlap,
    split_path_spec,
)

# How far a halo may reach, as a fraction of the patch it surrounds. See DatasetManager._affords_halo.
_MAX_HALO_FRACTION = 0.5
#: The whole-volume statistics every store can serve (``Dataset.read_data_statistics``): what a
#: GLOBAL_STAT stage may declare, and what the plan checks against without reading a voxel.
_STREAM_STAT_KEYS = frozenset(
    {"Min", "Max", "Mean", "Std", "MinPerChannel", "MaxPerChannel", "MeanPerChannel", "StdPerChannel"}
)

# Rows per Save-sweep slab: full-plane slabs keep the materialization bounded to a window while the
# composed region reads stay chunk-friendly. A declared memory_budget can only LOWER the height
# (see DatasetManager._sweep_rows); this constant is the cap and the no-budget default.
_SWEEP_SLAB_ROWS = 64

# What a sweep holds per row while in flight (the pulled source slab and the landed block) and
# the bytes each element travels as (float32 through the chain). See DatasetManager._sweep_rows.
_SWEEP_RESIDENT_SLABS = 2
_SWEEP_ELEMENT_BYTES = 4

#: What a whole-volume fallback holds while a case is in flight (the assembled tensor plus one
#: transform output), and the bytes each element travels as.
#:
#: Public because two callers must agree on the figure and neither owns it: the run-time budget check
#: (``_enforce_fallback_budget``) refuses a case against it, and the TRANSFORM plan prints and
#: enforces the same number before a byte is written. A plan estimating differently from the run it
#: describes is worse than no plan.
FALLBACK_INFLIGHT_FACTOR = 2
CASE_ELEMENT_BYTES = 4


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
    """Seed the global RNGs from ``key`` for the duration, then put them back exactly as they were.

    What lets two chains augment a case coherently. Each ``groups_dest`` builds its own transform
    objects, so an image chain and its mask chain hold different ``Rotate`` instances and cannot
    coordinate through shared state, and a mask rotated by a different angle than its image is a
    silently ruined dataset, not an error. Deriving the seed from what the two chains agree on (the
    Expand's seed, the case, which draw this is) makes them draw the same thing without knowing
    about each other.

    ``blake2b`` and not ``hash()``: string hashing is salted per process, and the same config must
    draw the same copies on every run and on every rank of one run.

    The state is restored because it is global: reseeding everything downstream would make every
    other random decision a function of how many copies were declared. CUDA's generators are part of
    that state: ``torch.manual_seed`` seeds every device, so restoring the CPU generator alone
    would leave every later GPU draw reseeded from here.
    """
    digest = hashlib.blake2b("|".join(str(part) for part in key).encode(), digest_size=4).digest()
    seed = int.from_bytes(digest, "big")
    cpu_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    states = (random.getstate(), np.random.get_state())
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        yield
    finally:
        random.setstate(states[0])
        np.random.set_state(states[1])
        torch.random.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


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
        # A draw is parameterised by the case and the copy, never by where in the volume it lands:
        # nothing an augmentation does depends on the region's position.
        del context
        return self(name, tensor, cache_attribute)

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


def save_destination(save: Save, default_dataset: Dataset, default_group: str) -> tuple[Dataset, str]:
    """The dataset and group a :class:`Save` caches into, the manager's own when it names none.

    Public because a planner has to resolve a destination exactly as the engine will: one that probes
    a store the run does not open has verified nothing.
    """
    if save.dataset:
        # Resolved once per Save: the chain is shared by every case's manager, and constructing a
        # Dataset probes the destination directory on disk.
        if save.destination is None:
            filename, _, file_format = split_path_spec(
                save.dataset,
                default_format="mha",
                supported_formats=SUPPORTED_FORMATS,
            )
            save.destination = Dataset(filename, file_format, save.scale_factors, save.downsample_method)
        dataset = save.destination
    else:
        # No destination of its own: the Save caches into the manager's dataset, whose write format
        # is not this stage's to redecorate: a pyramid asked here would silently not happen.
        if save.scale_factors:
            raise ConfigError(
                f"A '{type(save).__name__}' asks for a pyramid but names no dataset of its own.",
                "scale_factors describes a store this stage writes, so give it one:"
                " Write: {dataset: ./Out:omezarr, scale_factors: [4]}.",
            )
        dataset = default_dataset
    return dataset, save.group if save.group else default_group


@dataclass(frozen=True)
class _PatchStreamSource:
    """What a copy's patches are read from, and what runs on them once read.

    ``stage_plans`` mirrors ``stages`` one to one: the region plans locate every stage that reads
    somewhere other than the target patch (they compose, each pulling through the one before it); the
    rest are pointwise, so they run wherever the regions put them. ``pending_sweeps`` are the
    unsatisfied :class:`Save` caches the source reads through: each must be materialized (and the
    source re-resolved from disk) before any patch flows.

    ``entry`` is the entry name read from the source: the case's own, unless the source is a
    satisfied per-copy cache behind an :class:`Expand`, whose entries carry the copy's name.
    """

    dataset: Dataset
    group: str
    entry: str
    shape: list[int]
    stages: list[Stage]
    stage_plans: tuple[_ReadStagePlan, ...]
    pending_sweeps: tuple["_PendingSweep", ...] = ()

    @property
    def region_index(self) -> int | None:
        """The first region stage, or ``None`` for an exact-patch chain."""
        for index, plan in enumerate(self.stage_plans):
            if plan.kind.is_region:
                return index
        return None


@dataclass(frozen=True)
class _PendingSweep:
    """One unsatisfied :class:`Save` and the segment that feeds it. Materializing reads each slab of
    the Save's own space through the segment (re-planned at sweep time, see ``_materialize_save``)
    and region-writes it to ``destination``, after which the Save is a satisfied source boundary.

    ``entry`` names the destination entry (the copy's own behind an :class:`Expand`),
    ``source_entry`` the entry the segment reads. ``copy_stage_start`` is where the per-copy part of
    the segment begins: the :class:`Expand` splice point, or ``len(stages)`` for a segment that is
    entirely shared between copies; the expansion engine reads it to share one read pass across the
    copies whose per-copy stages are pointwise. ``stage_plans`` are the probe-time plans of the
    segment, kept for that classification only: the sweep itself re-plans (see
    ``_materialize_save``)."""

    destination: Dataset
    group: str
    entry: str
    stages: list[Stage]
    source_dataset: Dataset
    source_group: str
    source_entry: str
    source_shape: list[int]
    out_spatial: tuple[int, ...]
    # The case state the segment replays from (copied per slab).
    base_attributes: Attribute = field(repr=False)
    copy_stage_start: int = 0
    stage_plans: tuple[_ReadStagePlan, ...] = ()


@dataclass(frozen=True)
class PatchReadPlan:
    """Precomputed slicing and padding instructions for one patch request."""

    data_slices: tuple[slice, ...]
    reflect_padding: tuple[int, ...]
    constant_padding: tuple[int, ...]
    concatenate_extend_slice: bool


class PathCombine(ABC):
    """Base class for overlap-aware weighting schemes applied during patch assembly."""

    def __init__(self) -> None:
        self.data: torch.Tensor
        self.windows_1d: list[torch.Tensor]
        self.overlap: int
        self.overlaps: list[int]
        self._data_per_device: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}

    def set_patch_config(self, patch_size: list[int], overlap: int | list[int]) -> None:
        self._data_per_device.clear()
        overlaps = [overlap] * len(patch_size) if isinstance(overlap, int) else list(overlap)
        self.overlap = max(overlaps)
        self.overlaps = overlaps
        # The per-patch weight is the outer product of one 1-D window per axis. It is separable by
        # construction, so a per-axis partition of unity stays a partition of unity in N-D and overlapping
        # patches blend without darkening: no distance map or explicit renormalisation loop is needed, and
        # the trailing normalisation in Accumulator.assemble stays exact at the volume borders.
        # Each axis tapers over its own overlap, so anisotropic overlaps blend correctly per axis.
        #
        # The factors are kept, not just their product: Accumulator normalises by a per-axis sum rather
        # than a volume-sized buffer (see _weight_factors). Stored rather than re-derived, because the
        # all-flat case below never calls _window_1d: re-deriving would turn its uniform count into a
        # tapered weighting.
        if all(o <= 0 for o in overlaps):
            self.windows_1d = [torch.ones(size) for size in patch_size]
            self.data = torch.ones(patch_size)
            return
        self.windows_1d = [
            self._window_1d(size, axis_overlap) if axis_overlap > 0 else torch.ones(size)
            for size, axis_overlap in zip(patch_size, overlaps, strict=True)
        ]
        data = self.windows_1d[0]
        for window in self.windows_1d[1:]:
            data = data.unsqueeze(-1) * window
        self.data = data

    @property
    def selects(self) -> bool:
        """Whether the window picks ONE patch per voxel (values 0 or 1) rather than weighting several.

        A selection makes the kept regions a partition of the volume, so assembly writes each one
        instead of accumulating a weighted sum: see Accumulator._blend.
        """
        return False

    def window(self, dim: int, position: int, count: int) -> torch.Tensor:
        """The 1-D window along ``dim`` for the patch at grid ``position`` of ``count`` on that axis.

        A weighting is the same wherever the patch sits; a selection opens its border patches so the
        kept bands still reach the volume edge (see :class:`Trim`).
        """
        del position, count
        return self.windows_1d[dim]

    def weight(self, tensor: torch.Tensor) -> torch.Tensor:
        """The raw per-voxel window on ``tensor``'s device and dtype (cached per pair).

        The window UNNORMALISED, as declared. Assembly does not go through it: Accumulator divides each
        window by the total over the patches covering a voxel and applies that share instead, so the
        quantity it multiplies by is always in [0, 1]. This stays for a caller that wants the window
        itself.
        """
        key = (tensor.device, tensor.dtype)
        if key not in self._data_per_device:
            # Match the tensor dtype: the weight has no reason to carry more precision than the data it
            # scales, and a float64 weight would upcast the whole (channels x volume) blend.
            weight = self.data.to(device=tensor.device, dtype=tensor.dtype)
            if weight.is_floating_point():
                # A Gaussian tail underflows the target dtype (a 96^3 patch corner is ~6e-11: zero in
                # fp16). Floor it at the smallest normal so a caller scaling by this window on its own
                # does not silently drop those voxels.
                weight = weight.clamp(min=torch.finfo(weight.dtype).tiny)
            self._data_per_device[key] = weight
        return self._data_per_device[key]

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.weight(tensor) * tensor

    @abstractmethod
    def _window_1d(self, size: int, overlap: int) -> torch.Tensor:
        """Return the 1-D blend weight along one axis (length ``size``, ``overlap`` voxels tapered per side)."""


def blend_overlap(overlap: "int | float | str | list[int | float | str]", patch_size: list[int]) -> list[int]:
    """Per-axis blend overlap for a concrete ``patch_size`` (int broadcast, ``%``/fraction resolved).

    The blend has no volume extent: every axis whose patch is longer than one voxel is treated as tiled
    (the untiled-axis-0 rule is already applied by the slicing plan), so an int is the same voxel
    overlap on every such axis.
    """
    if isinstance(overlap, int):
        return [overlap if size > 1 else 0 for size in patch_size]
    return resolve_overlap(overlap, patch_size, [size + 1 for size in patch_size])


class Mean(PathCombine):
    """Uniform weighting: overlapping patches are plain-averaged by the assembly normalisation."""

    def _window_1d(self, size: int, overlap: int) -> torch.Tensor:
        return torch.ones(size)


class Cosinus(PathCombine):
    """Raised-cosine (sin**2) taper: a smooth partition of unity across the patch overlap."""

    def _window_1d(self, size: int, overlap: int) -> torch.Tensor:
        window = torch.ones(size)
        # sin**2 ramp over the overlap; the neighbouring patch's cos**2 ramp is its complement, so the two
        # sum to exactly one across the overlap. The +0.5 phase keeps the very edge > 0 so a single-patch
        # border, whose share is then the whole of the weight there, recovers the raw value.
        ramp = torch.sin((torch.arange(overlap, dtype=torch.float32) + 0.5) / overlap * (torch.pi / 2)) ** 2
        window[:overlap] = ramp
        window[size - overlap :] = ramp.flip(0)
        return window


class Trim(PathCombine):
    """Selection instead of weighting: each voxel comes from the patch that holds it most centrally.

    An interior patch keeps its central ``patch - overlap`` band, so the kept bands tile the axis
    exactly and the weights already sum to one; the first and last patch of an axis open to the volume
    edge. Every kept voxel therefore carries at least half the overlap of context on each side, where
    the unweighted default keeps whichever patch wrote last: possibly one holding that voxel on its
    own border with no context behind it, which is what a seam is.

    The values are 0 or 1, so the patch is selected rather than averaged: a discrete output (a label
    map, an argmax) survives reassembly, where any weighting would invent values between its classes.
    """

    @property
    def selects(self) -> bool:
        return True

    def _window_1d(self, size: int, overlap: int) -> torch.Tensor:
        if overlap >= size:
            # Nothing central left to keep: trimming both sides would leave an empty band, and a patch
            # that keeps nothing has no box to write. Keep the patch whole instead: the axis stops
            # being a partition and falls back to last-write-wins on it, which is what it was before.
            return torch.ones(size)
        window = torch.zeros(size)
        # Split an odd overlap so consecutive kept bands abut: k*stride + hi == (k+1)*stride + lo.
        window[overlap // 2 : size - (overlap - overlap // 2)] = 1.0
        return window

    def window(self, dim: int, position: int, count: int) -> torch.Tensor:
        overlap = self.overlaps[dim]
        window = self.windows_1d[dim]
        if overlap <= 0 or (position > 0 and position < count - 1):
            return window
        window = window.clone()
        if position == 0:
            window[: overlap // 2] = 1.0
        if position == count - 1:
            window[window.shape[0] - (overlap - overlap // 2) :] = 1.0
        return window


class Gaussian(PathCombine):
    """nnU-Net-style Gaussian importance weighting.

    Favours patch centres (where the model sees the most surrounding context), and down-weights the
    borders, which suppresses seam artefacts. Not a partition of unity, but ``Accumulator.assemble``
    normalises by the accumulated weight, so overlapping patches still form a correct weighted average.
    """

    def __init__(self, sigma_scale: float = 0.125) -> None:
        super().__init__()
        self.sigma_scale = sigma_scale

    def _window_1d(self, size: int, overlap: int) -> torch.Tensor:
        sigma = max(size * self.sigma_scale, 1e-6)
        center = (size - 1) / 2
        coords = torch.arange(size, dtype=torch.float32)
        return torch.exp(-((coords - center) ** 2) / (2.0 * sigma**2))


class Accumulator:
    """Accumulate patch predictions and reassemble them into a full tensor."""

    def __init__(
        self,
        patch_slices: list[tuple[slice, ...]],
        patch_size: list[int],
        patch_combine: PathCombine | None = None,
        batch: bool = True,
        sweep_axis: int = 0,
    ) -> None:
        # Which spatial axis the window slides along. The patch grid is emitted with this axis
        # outermost, so a patch's arrival finalizes everything behind it on that axis and nothing
        # else. 0 is what get_patch_slices_from_shape produces today.
        self.sweep_axis = sweep_axis
        self.patch_slices: list[tuple[slice, ...]] = []
        self.shape = max([[v.stop for v in patch] for patch in patch_slices])

        if patch_size is not None and not all(p == 0 for p in patch_size):
            # The last patch of an axis is padded up to the patch size for the model, then cropped; a
            # free axis (0) spans the full extent, so concretise it here or ``s.start + 0`` would
            # collapse the axis to a zero-width slice.
            concrete = [size if size > 0 else self.shape[dim] for dim, size in enumerate(patch_size)]
            for patch in patch_slices:
                slices = [slice(s.start, s.start + concrete[dim]) for dim, s in enumerate(patch)]
                self.patch_slices.append(tuple(slices))
        else:
            self.patch_slices = patch_slices
        self.patch_size = patch_size
        self.patch_combine = patch_combine
        self.batch = batch
        self._n = 2 if batch else 1
        self._count = len(patch_slices)
        self._filled = 0
        self._done = [False] * self._count
        # Patches are blended into this running buffer as they arrive (see add_layer), instead of
        # being kept until assembly: holding every patch of a large multi-class case (e.g. ~70 patches
        # of a 122-channel whole-body segmentation ≈ tens of GB) was the dominant reassembly RAM cost.
        self._result: torch.Tensor | None = None
        self._weighted: torch.Tensor | None = None
        # Blend-weight geometry: fixed for the accumulator's life, so it survives _reset.
        self._geometry: tuple[list[list[torch.Tensor]], list[torch.Tensor]] | None = None
        self._shares: dict[tuple[int, int, int, torch.dtype, torch.device], torch.Tensor] = {}
        self._boxes: dict[tuple[int, ...], tuple[slice, ...]] = {}

    def add_layer(self, index: int, layer: torch.Tensor) -> list[tuple[slice, torch.Tensor]]:
        """Blend one patch in; returns the slabs this completes (none for the whole-volume base)."""
        # Blend each patch straight into the running accumulator and drop the patch, rather than
        # storing all patches for a single assemble() at the end. The overlap blend is a weighted sum,
        # so accumulating incrementally is equivalent; re-adding an index is a no-op (last-write wins is
        # not possible once blended, and the prediction pipeline adds each patch exactly once).
        if self._done[index]:
            return []
        if self._result is None:
            # Allocate to the ACTUAL volume extent (self.shape), not the patch-size-extended grid. The
            # last patch of each axis is padded up to patch_size for the model, but that padded tail lies
            # OUTSIDE the volume; blending it would over-allocate the accumulator by up to
            # (patch_size - overlap) per axis (then get cropped away). We crop each patch to its in-volume
            # part at blend time instead, so nothing outside the volume is ever allocated.
            n = self._n
            self._result = torch.zeros(list(layer.shape[:n]) + list(self.shape), dtype=layer.dtype, device=layer.device)
        self._blend(layer, self.patch_slices[index])
        self._done[index] = True
        self._filled += 1
        return []

    def _blend(self, layer: torch.Tensor, patch_slice: tuple[slice, ...], row_offset: int = 0) -> None:
        """Blend one patch at its in-volume destination, ``row_offset`` rows down on the first spatial
        axis (the streaming window origin: 0 for the whole-volume base)."""
        n = self._n
        data = layer
        for dim, s in enumerate(patch_slice):
            if s.stop - s.start == 1:
                data = data.unsqueeze(dim=dim + n)
        # Clamp each spatial destination to the volume and crop the patch to it, BEFORE weighting: the
        # padded tail of a border patch lies outside the volume, so it has no share of the blend weight
        # to compute and every index in _weighted_patch stays in range.
        dest = [slice(s.start, min(s.stop, self.shape[dim])) for dim, s in enumerate(patch_slice)]
        data = data[tuple([slice(None)] * n + [slice(0, d.stop - d.start) for d in dest])]
        sweep = self.sweep_axis
        dest[sweep] = slice(dest[sweep].start - row_offset, dest[sweep].stop - row_offset)
        slices_dest = tuple([slice(cast(torch.Tensor, self._result).shape[i]) for i in range(n)] + dest)
        result = cast(torch.Tensor, self._result)
        lead = slices_dest[:n]
        if self.patch_combine is None:
            result[slices_dest] = data
        elif self.patch_combine.selects:
            # The kept regions partition the volume: nothing to weight, nothing to sum. Writing the box
            # this patch owns IS the operation, and it is what carries a discrete output through,
            # where a weighting would invent values between its classes.
            box = self._kept_box(patch_slice)
            kept = [slice(d.start + b.start, d.start + b.stop) for d, b in zip(dest, box, strict=True)]
            result[(*lead, *kept)] = data[(*lead, *box)]
        else:
            result[slices_dest] += self._weighted_patch(data, patch_slice)

    def _kept_box(self, patch_slice: tuple[slice, ...]) -> tuple[slice, ...]:
        """The sub-box a selection keeps of this patch: the run of ones in its window, per axis.

        Pure geometry, so it is read once from the host-side windows and cached per grid position.
        Deriving it from the share instead would read a device tensor: a host sync on every patch,
        which costs more than the weighting it replaces.
        """
        key = tuple(s.start for s in patch_slice)
        if key not in self._boxes:
            windows, _ = self._weight_geometry()
            box = []
            for dim, s in enumerate(patch_slice):
                kept = windows[dim][self._starts(dim).index(s.start)].nonzero().flatten()
                box.append(slice(int(kept[0]), int(kept[-1]) + 1))
            self._boxes[key] = tuple(box)
        return self._boxes[key]

    def _weight_geometry(self) -> tuple[list[list[torch.Tensor]], list[torch.Tensor]]:
        """Per axis: the blend window, and its sum over the patch grid.

        The outer product of those sums is the total weight covering each voxel. It factorises because
        the patch grid is a full per-axis product and the window is itself separable
        (``sum_p prod_d w_d == prod_d sum_k w_d``), so the total is one vector per axis and never exists
        as a volume, on a 320-row window of a 1331x1775 volume, 13 KB instead of 2.8 GB.

        The patch extent comes from the slices, not from the window: a free axis carries a single
        broadcast entry (the ModelPatch blend-window contract) that must cover the whole extent.

        The window is asked for per grid position, because a selection opens its border patches to the
        volume edge (see Trim); a weighting returns the same window everywhere.
        """
        if self._geometry is None:
            combine = cast(PathCombine, self.patch_combine)
            windows, totals = [], []
            for dim, extent in enumerate(self.shape):
                starts = self._starts(dim)
                length = self.patch_slices[0][dim].stop - self.patch_slices[0][dim].start
                per_position, total = [], torch.zeros(extent)
                for position, start in enumerate(starts):
                    window = combine.window(dim, position, len(starts))
                    window = window.expand(length) if window.numel() == 1 else window
                    stop = min(start + length, extent)
                    total[start:stop] += window[: stop - start]
                    per_position.append(window)
                windows.append(per_position)
                totals.append(total)
            self._geometry = (windows, totals)
        return self._geometry

    def _starts(self, dim: int) -> list[int]:
        """The patch grid's positions along one axis, in order."""
        return sorted({patch[dim].start for patch in self.patch_slices})

    def _share(self, dim: int, start: int, data: torch.Tensor) -> torch.Tensor:
        """This patch's fraction of the blend weight along one axis, ``w / sum_k w``, cached per axis."""
        extent = data.shape[self._n + dim]
        key = (dim, start, extent, data.dtype, data.device)
        if key not in self._shares:
            windows, totals = self._weight_geometry()
            window = windows[dim][self._starts(dim).index(start)]
            share = window[:extent] / totals[dim][start : start + extent]
            self._shares[key] = share.to(device=data.device, dtype=data.dtype)
        return self._shares[key]

    def _weighted_patch(self, data: torch.Tensor, patch_slice: tuple[slice, ...]) -> torch.Tensor:
        """``data`` scaled by its SHARE of the blend weight at each voxel, one axis at a time.

        Normalising per patch rather than dividing the assembled volume by an accumulated weight drops
        both the spatial-sized weight buffer and the final division pass over every channel. The shares
        sum to one per voxel by construction (the total is the sum over the same grid), so the blend
        stays exact, and each factor is a ratio of comparable quantities, so it lives in [0, 1] where
        the raw product underflows fp16 and needed a floor.

        Into a staging buffer the patches share, one patch-sized allocation per accumulator instead of
        per blend, and out of place: the caller's tensor is never touched, so the OOM retry (which
        re-blends the same patch on the CPU) never re-weights it.
        """
        if (
            self._weighted is None
            or self._weighted.shape != data.shape
            or self._weighted.dtype != data.dtype
            or self._weighted.device != data.device
        ):
            self._weighted = torch.empty_like(data)
        for dim, s in enumerate(patch_slice):
            view = [1] * data.ndim
            view[self._n + dim] = -1
            share = self._share(dim, s.start, data).view(view)
            if dim == 0:
                torch.mul(data, share, out=self._weighted)
            else:
                self._weighted.mul_(share)
        return self._weighted

    def _along_sweep(self, span: slice, lead: int) -> tuple[slice, ...]:
        """Index a result tensor over ``span`` on the sweep axis and whole on every other."""
        spatial = [span if dim == self.sweep_axis else slice(None) for dim in range(len(self.shape))]
        return tuple([slice(None)] * lead + spatial)

    def _sweep_shape(self, extent: int) -> list[int]:
        """The volume's spatial shape with the sweep axis cut down to ``extent``."""
        return [extent if dim == self.sweep_axis else size for dim, size in enumerate(self.shape)]

    def is_empty(self) -> bool:
        """True until the first patch has been blended in (no volume-sized buffer allocated yet)."""
        return self._result is None

    @property
    def footprint_shape(self) -> list[int]:
        """Spatial shape held in memory at once: the whole volume for the base accumulator (overridden
        by StreamingAccumulator, which keeps only a window). Used to size the blend device."""
        return self.shape

    def is_full(self) -> bool:
        # O(1): a running counter avoids re-scanning every slot after each added patch
        # (re-scanning per patch would be O(P^2) per case).
        return self._filled == self._count

    def assemble(self) -> torch.Tensor:
        if self._result is None:
            raise PatchError(
                "Accumulator.assemble() was called before any patch was added.",
                f"Expected up to {self._count} patch(es) via add_layer() before assembling.",
                "Add at least one patch (and check is_full()) before calling assemble().",
            )
        result = self._result
        # Nothing to normalise: each patch was blended in with its share of the weight, so the shares
        # already sum to one per voxel. No final crop either: patches are cropped at blend time.
        self._reset()
        return result

    def _reset(self) -> None:
        self._result = None
        self._weighted = None
        self._filled = 0
        self._done = [False] * self._count


class StreamingAccumulator(Accumulator):
    """Accumulator holding only the active window along the first spatial axis; it yields each finalized
    slab as its patches complete, so peak memory is two patch extents (the window and its slide room).

    The patch-grid order (``get_patch_slices_from_shape`` iterates ``itertools.product`` with the first
    spatial axis outermost) has patch starts along that axis never decreasing, so when a patch starting
    at ``z`` arrives, every voxel before ``z`` has already received all of its patches and the region up
    to ``z`` is final. ``add_layer`` returns those finalized slabs (``assemble()``'s values, from the
    same blend and weight arithmetic applied slab by slab), and ``finalize()`` flushes the tail.
    """

    def __init__(
        self,
        patch_slices: list[tuple[slice, ...]],
        patch_size: list[int],
        patch_combine: PathCombine | None = None,
        batch: bool = True,
        sweep_axis: int = 0,
    ) -> None:
        super().__init__(patch_slices, patch_size, patch_combine, batch, sweep_axis)
        sweep = self.sweep_axis
        self._window = min(
            max(patch[sweep].stop - patch[sweep].start for patch in self.patch_slices), self.shape[sweep]
        )
        starts = [patch[sweep].start for patch in self.patch_slices]
        if (starts and starts[0] != 0) or any(
            0 > current - previous or current - previous > self._window for previous, current in pairwise(starts)
        ):
            raise PatchError(
                "StreamingAccumulator requires patches starting at row 0 and ordered by non-decreasing start on"
                " the first spatial axis, advancing by at most one patch extent per step.",
                "get_patch_slices_from_shape generates this order; a custom patch source must preserve it.",
            )
        self._flushed = 0

    def add_layer(self, index: int, layer: torch.Tensor) -> list[tuple[slice, torch.Tensor]]:
        if self._done[index]:
            return []
        n = self._n
        patch_slice = self.patch_slices[index]
        # Correctness rests on patches ARRIVING in non-decreasing axis-0-start order, not just on the
        # slice list being sorted: a patch whose start is already flushed would write at a negative
        # window offset (dest[0] below), which torch silently reads from the end -> misplaced data, no
        # error. Fail loud instead. The prediction loop preserves per-case order (shuffle=False), so
        # this only guards a future misuse (e.g. a shuffling sampler).
        if patch_slice[self.sweep_axis].start < self._flushed:
            raise PatchError(
                f"StreamingAccumulator received patch start {patch_slice[self.sweep_axis].start} after flushing to "
                f"{self._flushed}: patches must arrive in non-decreasing first-spatial-axis order.",
                "Add patches in the order get_patch_slices_from_shape generates them (no shuffling).",
            )
        slabs = self._advance_to(patch_slice[self.sweep_axis].start)
        if self._result is None:
            self._result = torch.zeros(
                [*layer.shape[:n], *self._sweep_shape(self._window)],
                dtype=layer.dtype,
                device=layer.device,
            )
        self._blend(layer, patch_slice, row_offset=self._flushed)
        self._done[index] = True
        self._filled += 1
        return slabs

    def finalize(self) -> list[tuple[slice, torch.Tensor]]:
        """Flush the remaining window and reset for reuse; call once ``is_full()``."""
        slabs = self._advance_to(self.shape[self.sweep_axis])
        self._reset()
        self._flushed = 0
        return slabs

    @property
    def footprint_shape(self) -> list[int]:
        # Only the window is resident, so the blend-device budget is the window's: a huge volume streams
        # on the GPU within bounded VRAM. Blend and IEEE-correctly-rounded finalize ops (+, *, /, argmax,
        # cast) are bit-identical CPU/CUDA; only a transcendental-terminated float output (Softmax/Sigmoid)
        # can differ by ~1 ULP between a window on the GPU and a whole-volume reference on the CPU.
        return self._sweep_shape(self._window)

    def assemble(self) -> torch.Tensor:
        raise PatchError(
            "StreamingAccumulator does not assemble a whole volume.",
            "Consume the slabs returned by add_layer() and finalize() instead.",
        )

    def _advance_to(self, z: int) -> list[tuple[slice, torch.Tensor]]:
        """Finalize the window up to ``z`` (absolute) and shift the window origin there."""
        z = min(z, self.shape[self.sweep_axis])
        if self._result is None or z <= self._flushed:
            return []
        n = self._n
        length = z - self._flushed
        if length > self._window:
            raise PatchError(
                f"StreamingAccumulator asked to finalize {length} rows at once with a {self._window}-row window.",
                "Patch starts may advance by at most one patch extent per step (checked at construction).",
            )
        # Cloned: the window slides over these rows right after, so the slab handed out must not be a
        # view of it. Nothing else to do: the blend weights already sum to one over these voxels.
        slab = self._result[self._along_sweep(slice(0, length), n)].clone()
        keep = self._window - length
        # .clone(): source and destination views overlap when length < window.
        self._result[self._along_sweep(slice(0, keep), n)] = self._result[
            self._along_sweep(slice(length, self._window), n)
        ].clone()
        self._result[self._along_sweep(slice(keep, self._window), n)] = 0
        region = slice(self._flushed, z)
        self._flushed = z
        return [(region, slab)]


class SlabRegionStream:
    """Slab in → slab out through one region stage, with bounded lookahead.

    The write mirror of the read dispatcher's single-region rule: finalized slabs arrive in order along
    the first spatial axis (the :class:`StreamingAccumulator`'s order), and each output region is
    emitted as soon as the input region it pulls has arrived, so only a sliding window of the input is
    ever resident. The stage itself is two injected callables, both pure region arithmetic + tensor
    work, so no stage kind has streaming code of its own:

    - ``pull(target_slices) -> source_slices``: the clamped input region an output region is computed
      from (a transform's ``stream_region_target``/``stream_region_source``, or a halo enlargement).
    - ``produce(window, target_slices, source_slices) -> tensor``: the output block for
      ``target_slices``, given exactly the pulled window.

    The schedule is derived from ``pull`` alone: a probe finds which output axis the input slab axis
    feeds and in which direction (a mirrored or permuted axis streams too, through the sink's
    random-access region writes), and emission advances along that axis as far as the arrived input
    allows. Any per-axis monotone map works; nothing here names a stage.
    """

    def __init__(
        self,
        pull: Callable[[tuple[slice, ...]], list[slice]],
        produce: Callable[[torch.Tensor, tuple[slice, ...], list[slice]], torch.Tensor],
        in_spatial_shape: list[int],
        out_spatial_shape: list[int],
    ) -> None:
        self._pull = pull
        self._produce = produce
        self._in_shape = [int(s) for s in in_spatial_shape]
        self._out_shape = [int(s) for s in out_spatial_shape]
        self._axis, self._ascending = self._probe_axis()
        self._slabs: list[tuple[int, torch.Tensor]] = []
        self._complete_to = 0
        self._emitted = 0

    def _probe_axis(self) -> tuple[int, bool]:
        """Which output axis the input slab axis feeds, and whether in ascending order.

        Probing one output row per axis against the full-region pull identifies the axis whose region
        controls input axis 0; comparing the first and last rows' pulls gives the direction. A wrong
        pick can never corrupt the output (emission is gated on the pull of the real regions); it only
        buffers more, so a map no probe can tell apart falls back to axis 0 ascending.
        """
        full = tuple(slice(0, n) for n in self._out_shape)
        baseline = self._pull(full)[0]
        for axis, extent in enumerate(self._out_shape):
            probe = list(full)
            probe[axis] = slice(0, 1)
            first = self._pull(tuple(probe))[0]
            if (first.start, first.stop) == (baseline.start, baseline.stop):
                continue
            probe[axis] = slice(extent - 1, extent)
            last = self._pull(tuple(probe))[0]
            return axis, first.start <= last.start
        return 0, True

    def _prefix_slices(self, m: int) -> tuple[slice, ...]:
        """The first ``m`` output rows in iteration order, as absolute region slices."""
        extent = self._out_shape[self._axis]
        rows = slice(0, m) if self._ascending else slice(extent - m, extent)
        slices = [slice(0, n) for n in self._out_shape]
        slices[self._axis] = rows
        return tuple(slices)

    def push(self, region: slice, slab: torch.Tensor) -> list[tuple[tuple[slice, ...], torch.Tensor]]:
        """Take the next finalized slab (rows ``region`` of the input) and emit what it completes."""
        if region.start != self._complete_to:
            raise PatchError(
                f"SlabRegionStream received rows [{region.start}, {region.stop}) after "
                f"[0, {self._complete_to}): slabs must arrive contiguously from the first row.",
                "Push the slabs exactly as the StreamingAccumulator yields them.",
            )
        self._slabs.append((region.start, slab))
        self._complete_to = region.stop
        return self._emit()

    def finalize(self) -> list[tuple[tuple[slice, ...], torch.Tensor]]:
        """Emit whatever the completed input still allows and verify nothing is left behind."""
        emitted = self._emit()
        if self._emitted != self._out_shape[self._axis]:
            raise PatchError(
                f"SlabRegionStream finalized with {self._emitted} of {self._out_shape[self._axis]} "
                f"output rows emitted (input complete to {self._complete_to} of {self._in_shape[0]}).",
                "Push every slab of the input before finalizing.",
            )
        self._slabs.clear()
        return emitted

    def _emit(self) -> list[tuple[tuple[slice, ...], torch.Tensor]]:
        extent = self._out_shape[self._axis]
        # The pull of an iteration prefix grows monotonically with it, so the furthest emittable row is
        # a binary search. O(log rows) pull calls per push, and a pull may be more than slice
        # arithmetic (a declaration may copy the attribute it reads).
        low, high = self._emitted, extent
        while low < high:
            middle = (low + high + 1) // 2
            if self._pull(self._prefix_slices(middle))[0].stop <= self._complete_to:
                low = middle
            else:
                high = middle - 1
        m = low
        if m == self._emitted:
            return []
        rows = slice(self._emitted, m) if self._ascending else slice(extent - m, extent - self._emitted)
        target = list(self._prefix_slices(m))
        target[self._axis] = rows
        source = self._pull(tuple(target))
        window = self._window(source)
        result = self._produce(window, tuple(target), source)
        self._emitted = m
        if m < extent:
            remaining = [slice(0, n) for n in self._out_shape]
            remaining[self._axis] = slice(m, extent) if self._ascending else slice(0, extent - m)
            keep_from = self._pull(tuple(remaining))[0].start
            self._slabs = [
                (start, slab) for start, slab in self._slabs if start + slab.shape[-len(self._in_shape)] > keep_from
            ]
        else:
            self._slabs.clear()
        return [(tuple(target), result)]

    def _window(self, source: list[slice]) -> torch.Tensor:
        """The buffered input restricted to ``source``: rows gathered from the arrived slabs, the
        other axes sliced in place."""
        axis0 = source[0]
        n_lead = self._slabs[0][1].dim() - len(self._in_shape)
        lead = (slice(None),) * n_lead
        pieces = []
        for start, slab in self._slabs:
            stop = start + slab.shape[n_lead]
            lo, hi = max(start, axis0.start), min(stop, axis0.stop)
            if lo < hi:
                pieces.append(slab[(*lead, slice(lo - start, hi - start))])
        window = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=n_lead)
        if window.shape[n_lead] != axis0.stop - axis0.start:
            raise PatchError(
                f"SlabRegionStream window rows [{axis0.start}, {axis0.stop}) are not fully buffered.",
                "The pull map must never reach below rows already pruned or above rows arrived.",
            )
        return window[(*lead, slice(None), *source[1:])]


class SlabAligner:
    """Merge several slab streams over the same axis into jointly finalized intervals.

    The cross-stream mirror of :class:`StreamingAccumulator`: each stream (a TTA copy's accumulator)
    emits finalized slabs in non-decreasing order along the shared first spatial axis, and a consumer
    that needs every stream's rows together (a cross-copy reduction) can only advance to the
    slowest frontier. ``push`` takes one stream's new slabs and returns the intervals that just
    became complete, each carrying every stream's rows; only the inter-stream skew is ever buffered,
    and a single stream passes through untouched. Nothing here knows what a stream is: it is pure
    interval arithmetic over ``nb_streams`` ordered emitters.
    """

    def __init__(self, nb_streams: int, lead_dims: int = 1) -> None:
        self._nb_streams = nb_streams
        self._lead = lead_dims
        self._pending: dict[int, list[tuple[int, torch.Tensor]]] = {}
        self._frontiers: dict[int, int] = {}
        self._consumed = 0
        self._completed: set[int] = set()

    @property
    def complete(self) -> bool:
        """Whether every stream has pushed its last slab."""
        return len(self._completed) == self._nb_streams

    def push(
        self, stream: int, slabs: list[tuple[slice, torch.Tensor]], finished: bool = False
    ) -> list[tuple[slice, dict[int, torch.Tensor]]]:
        """Take ``stream``'s freshly finalized slabs and return the newly joint-final intervals."""
        pending = self._pending.setdefault(stream, [])
        for region, slab in slabs:
            pending.append((region.start, slab))
            self._frontiers[stream] = region.stop
        if finished:
            self._completed.add(stream)
        if len(self._frontiers) < self._nb_streams:
            return []
        joint = min(self._frontiers.values())
        if joint <= self._consumed:
            return []
        lead = (slice(None),) * self._lead
        rows: dict[int, torch.Tensor] = {}
        for key in sorted(self._pending):
            pieces = [
                slab[
                    (
                        *lead,
                        slice(max(start, self._consumed) - start, min(start + slab.shape[self._lead], joint) - start),
                    )
                ]
                for start, slab in self._pending[key]
                if start < joint and start + slab.shape[self._lead] > self._consumed
            ]
            if not pieces:
                raise PatchError(
                    f"SlabAligner stream {key} holds no rows for [{self._consumed}, {joint}).",
                    "Push the slabs exactly as the accumulators finalize them, without skipping a stream.",
                )
            rows[key] = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=self._lead)
            self._pending[key] = [
                (start, slab) for start, slab in self._pending[key] if start + slab.shape[self._lead] > joint
            ]
        interval = slice(self._consumed, joint)
        self._consumed = joint
        return [(interval, rows)]


class Patch(ABC):
    """Abstract base class for dataset-level and model-level patch definitions."""

    @abstractmethod
    def __init__(
        self,
        patch_size: list[int],
        overlap: OverlapSpec = None,
        pad_value: float | None = 0,
        extend_slice: int = 0,
    ) -> None:
        if extend_slice != 0 and patch_size is not None and patch_size[0] != 1:
            raise ValueError(
                "`extend_slice` can only be used when patch_size[0] == 1 "
                f"(got patch_size[0]={patch_size[0]}, extend_slice={extend_slice})"
            )
        self.patch_size = patch_size
        self.overlap = overlap
        if isinstance(self.overlap, int):
            if self.overlap < 0:
                self.overlap = None
        self._patch_slices: dict[int, list[tuple[slice, ...]]] = {}
        self._sweep_axis: dict[int, int] = {}
        self._nb_patch_per_dim: dict[int, list[tuple[int, bool]]] = {}
        self.pad_value = pad_value
        self.extend_slice = extend_slice
        # Models need every patch at the declared size, so the last patch of an axis is padded up to it.
        # A consumer that REDUCES patches instead (streamed evaluation) must see only in-volume voxels:
        # padded ones would pollute its running sums, so it turns this off and takes the cropped patch.
        self.pad_to_patch = True
        # The model's per-axis downsampling factor a FREE (``0``) axis rounds up to, set before the grids
        # are cut so each case's whole-axis extent lands on a valid model input. ``None`` outside a model
        # (evaluation) or for a network that never downsamples.
        self.free_axis_multiple: list[int] | None = None
        # Whether a free (``0``) axis was DECLARED. Captured now because the OOM re-plan later pins
        # ``patch_size`` to a concrete size in place, erasing the ``0`` the overlap default keys on.
        self._declared_free_axis: bool = (
            patch_size is not None and any(p == 0 for p in patch_size) and not all(p == 0 for p in patch_size)
        )

    def load(self, shape: list[int], a: int = 0) -> None:
        # The grid decides its own sweep axis and the reassembly reads it back (get_sweep_axis): one
        # source of truth, because a grid emitted for one axis and reassembled along another hands out
        # regions that are not final, with nothing to report it.
        self._sweep_axis[a] = best_sweep_axis(
            concretize_patch_size(self.patch_size, shape, self.free_axis_multiple), shape
        )
        self._patch_slices[a], self._nb_patch_per_dim[a] = get_patch_slices_from_shape(
            self.patch_size,
            shape,
            self.overlap,
            self.free_axis_multiple,
            self._declared_free_axis,
            self._sweep_axis[a],
        )

    def get_sweep_axis(self, a: int = 0) -> int:
        """The axis this grid is ordered by, and so the one reassembly must slide along."""
        return self._sweep_axis[a]

    @abstractmethod
    def init(self, key: str):
        pass

    def get_patch_slices(self, a: int = 0):
        return self._patch_slices[a]

    def get_read_plan(
        self, data_shape: list[int] | tuple[int, ...], index: int, a: int, is_input: bool
    ) -> PatchReadPlan:
        slices_pre = [slice(None) for _ in data_shape[: -len(self._patch_slices[a][0])]]
        extend_slice = self.extend_slice if is_input else 0

        bottom = extend_slice // 2
        top = int(np.ceil(extend_slice / 2))
        s = slice(
            (
                self._patch_slices[a][index][0].start - bottom
                if self._patch_slices[a][index][0].start - bottom >= 0
                else 0
            ),
            (
                self._patch_slices[a][index][0].stop + top
                if self._patch_slices[a][index][0].stop + top <= data_shape[len(slices_pre)]
                else data_shape[len(slices_pre)]
            ),
        )
        slices = [s, *list(self._patch_slices[a][index][1:])]
        reflect_padding = [0 for _ in range((len(slices) - 1) * 2)] + [0, 0]
        if extend_slice > 0 and (s.stop - s.start) < bottom + top + 1:
            if self._patch_slices[a][index][0].start - bottom < 0:
                reflect_padding[-2] = bottom - self._patch_slices[a][index][0].start
            if self._patch_slices[a][index][0].stop + top > data_shape[len(slices_pre)]:
                reflect_padding[-1] = self._patch_slices[a][index][0].stop + top - data_shape[len(slices_pre)]

        constant_padding = []
        if self.pad_to_patch and self.patch_size is not None:
            nspatial = len(slices)
            for dim_it, _slice in enumerate(reversed(slices)):
                axis = nspatial - 1 - dim_it
                extent = data_shape[-dim_it - 1]
                declared = self.patch_size[axis]
                if declared != 0:
                    target = declared
                else:
                    # A FREE axis pads up to THIS case's extent rounded to the model's downsampling
                    # multiple, so a small heterogeneous case still reaches the network at a valid input
                    # size (the up-front worst-case sizing only guarantees the largest case).
                    m = free_axis_rounding(self.free_axis_multiple, axis, nspatial)
                    target = ((extent + m - 1) // m) * m if m > 1 else extent
                p = 0 if _slice.start + target <= extent else target - (extent - _slice.start)
                constant_padding.append(0)
                constant_padding.append(p)

        return PatchReadPlan(
            data_slices=tuple(slices_pre + slices),
            reflect_padding=tuple(reflect_padding),
            constant_padding=tuple(constant_padding),
            concatenate_extend_slice=extend_slice > 0,
        )

    def apply_read_plan(self, data: torch.Tensor, plan: PatchReadPlan) -> torch.Tensor:
        data_sliced = data
        if any(plan.reflect_padding):
            data_sliced = F.pad(data_sliced, plan.reflect_padding, "reflect")
        if any(plan.constant_padding):
            data_sliced = F.pad(
                data_sliced,
                plan.constant_padding,
                "constant",
                (
                    0
                    if data_sliced.dtype == torch.uint8
                    else (self.pad_value if self.pad_value is not None else float(data.min().item()))
                ),
            )
        if self.patch_size is not None and not all(p == 0 for p in self.patch_size):
            for d in [i for i, v in enumerate(reversed(self.patch_size)) if v == 1]:
                data_sliced = torch.squeeze(data_sliced, dim=len(data_sliced.shape) - d - 1)
        return (
            torch.cat([data_sliced[:, i, ...] for i in range(data_sliced.shape[1])], dim=0)
            if plan.concatenate_extend_slice
            else data_sliced
        )

    def get_data(self, data: torch.Tensor, index: int, a: int, is_input: bool) -> list[torch.Tensor]:
        plan = self.get_read_plan(list(data.shape), index, a, is_input)
        data_sliced = data[plan.data_slices]
        return self.apply_read_plan(data_sliced, plan)

    def get_size(self, a: int = 0) -> int:
        return len(self._patch_slices[a])


@config("Patch")
class DatasetPatch(Patch):
    """Patch definition applied when sampling data from datasets."""

    def __init__(
        self,
        patch_size: list[int] = [128, 128, 128],
        overlap: OverlapSpec = None,
        pad_value: float | None = None,
        extend_slice: int = 0,
    ) -> None:
        super().__init__(patch_size, overlap, pad_value, extend_slice)

    def init(self, key: str = ""):
        pass


@config()
class ModelPatch(Patch):
    """Patch definition applied inside model graphs during prediction or training."""

    def __init__(
        self,
        patch_size: list[int] = [128, 128, 128],
        overlap: OverlapSpec = None,
        patch_combine: str | None = None,
        pad_value: float | None = None,
        extend_slice: int = 0,
    ) -> None:
        super().__init__(patch_size, overlap, pad_value, extend_slice)
        self._patch_combine = patch_combine
        self.patch_combine: PathCombine | None = None

    def init(self, key: str):
        if self._patch_combine is not None:
            module, name = get_module(self._patch_combine, "konfai.data.patching")
            self.patch_combine = apply_config(key)(getattr(module, name))()
        if self.patch_size is not None and self.overlap is not None:
            if self.patch_combine is not None:
                # Keep every axis so the weight broadcasts against the patch whatever the axis position
                # (dropping trailing axes misaligns the broadcast); a singleton (1) or free (0) axis
                # carries a uniform weight, a >1 axis its tapered window.
                kept = [i if i > 1 else 1 for i in self.patch_size]
                self.patch_combine.set_patch_config(kept, blend_overlap(self.overlap, kept))
        else:
            self.patch_combine = None

    def disassemble(self, *data_list: torch.Tensor) -> Iterator[list[torch.Tensor]]:
        for i in range(self.get_size()):
            yield [self.get_data(data, i, 0, True) for data in data_list]


class DatasetManager:
    """Cache-backed manager for one dataset case and one source/destination group."""

    def __init__(
        self,
        index: int,
        group_src: str,
        group_dest: str,
        name: str,
        dataset: Dataset,
        patch: DatasetPatch | None,
        transforms: list[Transform],
        data_augmentations_list: list[DataAugmentationsList],
    ) -> None:
        self.group_src = group_src
        self.group_dest = group_dest
        self.name = name
        self.index = index
        self.dataset = dataset
        self.transforms = transforms
        self.loaded = False
        self.augmentationLoaded = False
        self.cache_attributes: list[Attribute] = []
        _shape, cache_attribute = self.dataset.get_infos(self.group_src, name)
        self.base_shape = list(_shape)
        self.cache_attributes.append(cache_attribute)
        _shape = list(_shape[1:])

        self.data: list[torch.Tensor] = []
        self.augmented_data: dict[int, torch.Tensor] = {}
        self.total_augmentations = 0

        # The chain around its Expand: pre runs once per case, post once per copy. Without a marker
        # the split is (everything, None, []) and every fold below reduces to the plain per-case chain.
        self._expand_pre, self._expand, self._expand_post = split_expand(transforms)
        # The landing fold, on a working state: the run re-makes every transition itself, so the
        # case baseline the walks and replays start from must not carry them twice.
        folding = Attribute(cache_attribute)
        for transform_function in self._expand_pre:
            _shape = self._fold_case_state(transform_function, _shape, folding)
        self._adopt_case_facts(folding, cache_attribute)
        # The grid and case state at the Expand point: what the first per-copy stage is handed.
        self._shape_at_expand = list(_shape)
        self._attributes_at_expand = Attribute(cache_attribute)
        # The un-augmented landing of the per-copy tail. A draw is the identity here, because copy 0
        # carries none: the real per-copy grids are folded in reset_augmentation, stage by stage.
        for transform_function in self._expand_post:
            if _is_draw(transform_function):
                continue
            _shape = self._fold_case_state(transform_function, _shape, folding)
        self._adopt_case_facts(folding, cache_attribute)

        self.patch = (
            DatasetPatch(
                patch_size=patch.patch_size,
                overlap=patch.overlap,
                pad_value=patch.pad_value,
                extend_slice=patch.extend_slice,
            )
            if patch
            else DatasetPatch(_shape)
        )
        if patch is not None:
            # The manager works on its own copy (per-case grids); carry the reduction-vs-model contract
            # with it, or a streamed evaluation would silently get padded border patches back.
            self.patch.pad_to_patch = patch.pad_to_patch
            # Carry the model's downsampling multiple too, so each per-case free axis rounds up to a valid
            # input size on this copy's grid, not just on the up-front worst-case sizing.
            self.patch.free_axis_multiple = patch.free_axis_multiple
            # Carry the DECLARED free-axis flag: after an OOM re-plan the source patch_size is already
            # concrete, so this fresh copy could not re-derive it, and the free axis must keep the
            # fraction overlap default, not fall back to the fixed-patch remainder.
            self.patch._declared_free_axis = patch._declared_free_axis
        self.patch.load(_shape, 0)
        # The spatial grid each copy's patches are cut on: the un-augmented copy's is the source shape
        # folded by the transforms, and a copy whose draw changes shape (Permute, Mask) has its own.
        self.shapes: list[list[int]] = [_shape]
        self.data_augmentations_list = data_augmentations_list
        self._patch_stream_sources: dict[tuple[int, bool], _PatchStreamSource | None] = {}
        self._stream_refusals: dict[tuple[int, bool], str] = {}
        self._stream_evolved: dict[tuple[int, bool], Attribute] = {}
        self._stream_attributes_persisted: set[int] = set()
        self._peak_case_bytes: int | None = None
        # A GLOBAL_STAT stage's whole-volume statistic is read on the process that runs the chain,
        # at first data access, never by a plan probe: the plan checks the source can provide it
        # (headers only) and the run reads it once. ``_statistics_deferred`` remembers that a plan
        # was resolved without its seed, so the run replans before reading.
        self._statistics_seeded = False
        self._statistics_deferred = False
        # Why a Save sweep gave up for this case, or None. One field rather than a flag beside a
        # warning: the flag is what reroutes the case, the sentence is what a caller with no
        # fallback has to raise with, and they must never disagree.
        self._sweep_failure: str | None = None
        # Rewrite mode: every satisfied-Save probe answers "not written yet", so the case recomputes
        # from the source and each stream's finalize renames over the old entry. Never the default --
        # the boundary IS the per-case resume.
        self._rewrite_saves = False
        # The per-rank budget the sweeps size their slabs against (None = the fixed _SWEEP_SLAB_ROWS).
        self._sweep_budget_bytes: float | None = None
        self._chain_device: torch.device | None = None
        self._disk_statistics: dict[tuple[Dataset, str, str, tuple[int, ...] | None], dict[str, float]] = {}
        # Save caches already swept by THIS run, keyed by (store, group, entry): under --overwrite the
        # existence probe answers "not written", and without this ledger every copy of an Expand chain
        # would re-sweep the same shared pre-Expand cache once per copy.
        self._swept_entries: set[tuple[str, str, str]] = set()
        # reset_state=False: the first manager built for a case draws (state_init draws a missing
        # index), and every later group's manager reuses that draw: redrawing here would give each
        # group its own geometry and desynchronise the per-copy patch grids across groups.
        self.reset_augmentation(reset_state=False)
        self.cache_attributes_bak = copy.deepcopy(self.cache_attributes)
        # The case as STORED, untouched forever: a boundary-based plan legitimately rewrites the
        # backup with the cache's header, but a rewrite replan must start from the original source.
        self._cache_attributes_pristine = copy.deepcopy(self.cache_attributes_bak)

    def reset_augmentation(self, reset_state: bool = True):
        self.cache_attributes[:] = self.cache_attributes[:1]
        self.shapes[:] = self.shapes[:1]
        self.augmented_data.clear()
        # An augmented copy's stream source is only as good as the draw it was planned for: a halo is
        # the draw's own, and a re-draw is a new one. Drop every plan, so the next request replans
        # against the draw the copies actually carry.
        self._patch_stream_sources.clear()
        self._stream_refusals.clear()
        self._stream_evolved.clear()
        self._stream_attributes_persisted.clear()
        self.total_augmentations = 0
        if self._expand is not None:
            self._draw_expand_copies(reset_state)
        else:
            self._draw_augmentation_lists(reset_state)
        self.augmentationLoaded = self.total_augmentations == 0

    def _draw_expand_copies(self, reset_state: bool) -> None:
        """Draw the :class:`Expand` copies by walking the per-copy tail STAGE BY STAGE.

        This is what makes a draw a stage like any other: each one is parameterised on the grid and
        the case state the stages before it leave, so a transform between two draws is seen by the
        second, and a shape-changing draw is seen by the transform after it. Walking the tail once
        per copy (rather than drawing everything up front and folding the transforms afterwards)
        is the only order that makes ``T, draw, T, draw`` mean what it reads like.

        Each draw is seeded from ``(Expand.seed, case, draw class, which one of that class it is)``
        (everything two chains of the same case agree on, and nothing they do not), so an image
        chain and its mask chain draw the same copies without ever meeting. Keyed on the draw's
        CLASS and its rank among its own kind, not on its position in the tail: an intensity draw
        one chain declares and the other has no use for must not shift the geometric draws they
        share.
        """
        expand = self._expand
        assert expand is not None  # nosec B101 - the caller checked
        shapes = [list(self._shape_at_expand) for _ in range(expand.nb)]
        attributes = [copy.deepcopy(self._attributes_at_expand) for _ in range(expand.nb)]
        # The copies' walk states, apart from the baselines above: the landing fold evolves the
        # geometry, and a streamed replay must start from the case as stored.
        foldings = [Attribute(attribute) for attribute in attributes]
        drawn: dict[str, int] = {}
        for stage in self._expand_post:
            if _is_draw(stage):
                if reset_state:
                    stage.reset_state(self.index)
                kind = type(stage).__name__
                occurrence = drawn.get(kind, 0)
                drawn[kind] = occurrence + 1
                # One draw, every copy at once: state_init IS the per-copy sampler, and it wants the
                # copies' current grids, which the stages before it just folded.
                # Keyed by the case's NAME, not its index: the index is a position in the run's
                # case list, and a different `subset` (or a second run over image and mask with
                # different subsets) must not hand a case other copies.
                with _drawn_from(expand.draw_seed, self.name, kind, occurrence):
                    shapes = stage.state_init(self.index, shapes, foldings)
                continue
            for index in range(expand.nb):
                shapes[index] = self._fold_case_state(stage, shapes[index], foldings[index])
        for index in range(expand.nb):
            # As at the case-level folds: the box a per-copy Crop computed is a case fact, and
            # losing it re-reads the volume once per later fold of that copy.
            self._adopt_case_facts(foldings[index], attributes[index])
            self.cache_attributes.append(attributes[index])
            self.shapes.append(list(shapes[index]))
            self.patch.load(list(shapes[index]), index + 1)
        self.total_augmentations = expand.nb

    def _draw_augmentation_lists(self, reset_state: bool) -> None:
        """The training form: copies declared as ``Dataset.augmentations`` lists, applied after the
        whole chain."""
        i = 1
        for data_augmentations in self.data_augmentations_list:
            shape = []
            caches_attribute = []
            for _ in range(data_augmentations.nb):
                shape.append(list(self.shapes[0]))
                caches_attribute.append(copy.deepcopy(self.cache_attributes[0]))

            for data_augmentation in data_augmentations.data_augmentations:
                if reset_state:
                    data_augmentation.reset_state(self.index)
                shape = data_augmentation.state_init(self.index, shape, caches_attribute)
            for it, s in enumerate(shape):
                self.cache_attributes.append(caches_attribute[it])
                self.shapes.append(s)
                self.patch.load(s, i)
                i += 1
            self.total_augmentations += data_augmentations.nb

    def load(
        self,
        pre_transform: list[Transform],
        data_augmentations_list: list[DataAugmentationsList],
        load_augmentations: bool = True,
    ) -> None:
        if not self.loaded:
            self._load(pre_transform)
        if load_augmentations and not self.augmentationLoaded:
            self._load_augmentation(data_augmentations_list)

    def _load(self, pre_transform: list[Transform]):
        self.cache_attributes = copy.deepcopy(self.cache_attributes_bak)
        i = len(pre_transform)
        data = None
        for transform_function in reversed(pre_transform):
            if isinstance(transform_function, Save):
                dataset, group_dest = save_destination(transform_function, self.dataset, self.group_dest)
                if not self._rewrite_saves and dataset.is_dataset_exist(group_dest, self.name):
                    data, attrib = dataset.read_data(group_dest, self.name)
                    self.cache_attributes[0].update(attrib)
                    break
            i -= 1

        if i == 0:
            data, _ = self.dataset.read_data(self.group_src, self.name)

        data = torch.from_numpy(data)
        if self._chain_device is not None:
            data = data.to(self._chain_device)

        if len(pre_transform):
            data = self._apply_chain(data, pre_transform[i:], self.cache_attributes[0], self.name)
        self.data.append(data)

        for i in range(len(self.cache_attributes) - 1):
            self.cache_attributes[i + 1].update(self.cache_attributes[0])
        self.loaded = True

    def _apply_chain(
        self, tensor: torch.Tensor, transforms: Sequence[Stage], attribute: Attribute, entry: str
    ) -> torch.Tensor:
        """Apply stages in order on an assembled tensor, writing each Save's cache under ``entry``.

        The one whole-volume applicator: ``_load`` drives it with the case's own name, and the
        expansion fallback with a copy's name: the entry is the only thing that differs between
        assembling a case and assembling one of its copies.
        """
        for transform_function in transforms:
            tensor = transform_function(self.name, tensor, attribute)
            if isinstance(transform_function, Save):
                dataset, group_dest = save_destination(transform_function, self.dataset, self.group_dest)
                dataset.write(group_dest, entry, tensor.cpu().numpy(), attribute)
        return tensor

    def _load_augmentation(self, data_augmentations_list: list[DataAugmentationsList]) -> None:
        start_index = 1
        for data_augmentations in data_augmentations_list:
            self._load_augmentation_group(start_index, data_augmentations)
            start_index += data_augmentations.nb
        self.augmentationLoaded = len(self.augmented_data) == self.total_augmentations

    def _load_augmentation_group(self, start_index: int, data_augmentations: DataAugmentationsList) -> None:
        if data_augmentations.nb == 0:
            return

        indices = range(start_index, start_index + data_augmentations.nb)
        if all(index in self.augmented_data for index in indices):
            return

        a_data = [self.data[0].clone() for _ in range(data_augmentations.nb)]
        for data_augmentation in data_augmentations.data_augmentations:
            if data_augmentation.groups is None or self.group_dest in data_augmentation.groups:
                a_data = data_augmentation(self.name, self.index, a_data)

        for index, data in zip(indices, a_data, strict=False):
            self.augmented_data[index] = data
        self.augmentationLoaded = len(self.augmented_data) == self.total_augmentations

    def _augmentation_group(self, a: int) -> tuple[int, DataAugmentationsList]:
        """The augmentation list copy *a* belongs to, and the copy index that list starts at."""
        start_index = 1
        for data_augmentations in self.data_augmentations_list:
            if start_index <= a < start_index + data_augmentations.nb:
                return start_index, data_augmentations
            start_index += data_augmentations.nb
        raise IndexError(f"Augmentation index {a} out of range for dataset '{self.name}'.")

    def _augmentation_stages(self, a: int) -> list[Stage]:
        """The augmentations copy *a* is made of, each bound to it.

        Copy 0 is made of none: it is the tensor the transforms produced, which is why it is the one
        copy that has a counterpart on disk to stream from at all. The rest carry their list's draw,
        minus whatever that draw does not address to this group.
        """
        if a == 0:
            return []
        start_index, data_augmentations = self._augmentation_group(a)
        return [
            AugmentedStage(data_augmentation, self.index, a - start_index)
            for data_augmentation in data_augmentations.data_augmentations
            if data_augmentation.groups is None or self.group_dest in data_augmentation.groups
        ]

    def _expand_tail(self, a: int) -> list[Stage]:
        """The per-copy tail of an :class:`Expand` chain, as copy ``a`` runs it.

        The tail IS the declared order: a transform stays itself, a draw is bound to this copy. The
        two kinds are the same species to everything downstream (the planner reads one contract,
        the replay calls one signature), which is why they can be written in any order.
        """
        if a == 0:
            # Copy 0 is the case itself: it carries no draw, so the tail is its transforms alone --
            # the same landing __init__ folds, and what a probe or a header asks for by default.
            return [stage for stage in self._expand_post if not _is_draw(stage)]
        return [AugmentedStage(stage, self.index, a - 1) if _is_draw(stage) else stage for stage in self._expand_post]

    def _get_tensor(self, a: int) -> torch.Tensor:
        if a == 0:
            return self.data[0]
        if a not in self.augmented_data:
            self._load_augmentation_group(*self._augmentation_group(a))
        return self.augmented_data[a]

    def copy_entry(self, a: int) -> str:
        """The entry name copy ``a`` writes (and resumes) under, behind this chain's ``Expand``.

        Copy 0 is the case itself (the un-augmented tensor has no draw of its own to name), and so
        is every copy of a chain with no ``Expand``, where nothing per-copy is ever written.
        """
        if self._expand is None or a == 0:
            return self.name
        return self._expand.entry(self.name, a)

    def _read_disk_statistics(
        self,
        source_dataset: Dataset,
        source_group: str,
        source_entry: str,
        channels: list[int] | None,
    ) -> dict[str, float]:
        """Read (and memoise) the whole-volume statistics of one on-disk group for this case.

        ``read_data_statistics`` scans the stored volume without materialising it, but it is still a
        full pass: memoise it per (dataset, group, entry, channels) so a per-patch consumer (whose
        ``inverse()`` pops the seeded keys back out of the cache attribute at prediction time) does
        not re-scan the volume once per patch.
        """
        key = (source_dataset, source_group, source_entry, tuple(channels) if channels is not None else None)
        if key not in self._disk_statistics:
            self._disk_statistics[key] = source_dataset.read_data_statistics(source_group, source_entry, channels)
        return self._disk_statistics[key]

    def _require_statistics(self) -> None:
        """Run-time entry: from here on the plan seeds the statistics it deferred, and a plan resolved
        without them is resolved again. Called by every path that reads data through the plan."""
        if self._statistics_seeded:
            return
        self._statistics_seeded = True
        if self._statistics_deferred:
            self._statistics_deferred = False
            self._invalidate_stream_plans()

    def _ensure_stream_stats(
        self,
        source_dataset: Dataset,
        source_group: str,
        source_entry: str,
        cache_attribute: Attribute,
        required_stats: set[str],
        channels: list[int] | None = None,
    ) -> bool:
        missing_stats = [key for key in required_stats if key not in cache_attribute]
        if not missing_stats:
            return True

        stats = self._read_disk_statistics(source_dataset, source_group, source_entry, channels)
        stats_mapping = {
            "Min": stats["min"],
            "Max": stats["max"],
            "Mean": stats["mean"],
            "Std": stats["std"],
            # The same figures per channel: what a vector-valued quantity needs, where pooling the
            # components into one number describes nothing (the mean of a displacement field is a
            # translation, and it has three parts).
            "MinPerChannel": stats.get("min_per_channel"),
            "MaxPerChannel": stats.get("max_per_channel"),
            "MeanPerChannel": stats.get("mean_per_channel"),
            "StdPerChannel": stats.get("std_per_channel"),
        }
        for key in missing_stats:
            value = stats_mapping.get(key)
            if value is None:
                continue
            if key.endswith("PerChannel"):
                cache_attribute[key] = np.asarray(value, dtype=np.float32)
            elif key in {"Mean", "Std"}:
                cache_attribute[key] = np.asarray([value], dtype=np.float32)
            else:
                cache_attribute[key] = value
        return all(key in cache_attribute for key in required_stats)

    def _affords_halo(self, a: int, halo: tuple[int, ...]) -> bool:
        """Whether a halo of this radius still buys copy *a* anything over loading the volume.

        Every patch pays the halo on every side and the patches tile the volume, so streaming a case
        reads ``prod(1 + 2 * halo_k / patch_k)`` times its bytes: the multiple streaming pays to keep
        one volume off the heap. Half a patch doubles every axis: 8x the reads in 3D. Past that the
        multiple runs away: a halo of one whole patch is 27x, while the saving is still just the
        one volume.
        """
        patch_size = self.patch.patch_size
        extent = (
            self.shapes[a]
            if patch_size is None or all(p == 0 for p in patch_size)
            else [min(p, s) for p, s in zip(patch_size, self.shapes[a], strict=False)]
        )
        return all(
            radius <= _MAX_HALO_FRACTION * size
            for radius, size in zip(_halo_radii(halo, len(extent)), extent, strict=False)
        )

    def _plan_stream_region(
        self,
        a: int,
        stages: list[Stage],
        source_dataset: Dataset,
        source_group: str,
        source_entry: str,
        cache_attribute: Attribute,
        source_spatial_shape: list[int],
        landing_shape: list[int] | None = None,
        seed_statistics: bool = True,
    ) -> tuple[bool, tuple[_ReadStagePlan, ...], Attribute, str | None]:
        """Validate a chain's locality declarations and plan its region stages, which compose.

        Returns ``(streamable, stage_plans, evolved, refusal)``: ``refusal`` naming the stage and
        the reason when the chain cannot stream (``None`` when it can): the stage's identity is only
        in hand at the moment each check fails, so the reason is built here: dropping it would
        leave every fallback silent. ``evolved`` is the case state the plan
        leaves, which a :class:`Save` sweep writes as its cache header. The chain streams when every
        stage is pointwise, a region kind (``HALO``/``ORIENTATION``/``CROP``/``REGRID``: any
        number, each pulling through the one before it), or a ``GLOBAL_STAT`` with a pre-populated
        statistic. The plan walks the chain once with one evolving case state, so each stage declares
        against, and remaps from: the geometry the stages before it left, and folds the spatial
        shapes stage by stage; a fold that does not land on ``landing_shape`` (the copy's own grid by
        default) refuses (the safety net for a stage whose shape map is not declared). Any
        ``WHOLE_VOLUME`` declaration, an unreadable ``GLOBAL_STAT``, a ``REGRID`` whose own
        declaration refuses (a stage answers for its own inputs), or a halo too wide to be worth
        reading rejects streaming. ``seed_statistics=False`` accepts a missing statistic instead of reading
        it, for a chain fed by a cache that is not materialized yet, whose re-resolution seeds it
        from the real entry. Nothing here names a stage: each declares its own contract, and this is
        where the declarations are read, which is why a transform and an augmentation are planned
        side by side.
        """
        evolved = Attribute(cache_attribute)
        shape = [int(extent) for extent in source_spatial_shape]
        localities: list[PatchLocality] = []
        plans: list[_ReadStagePlan] = []

        def refuse(reason: str) -> tuple[bool, tuple[_ReadStagePlan, ...], Attribute, str]:
            """A refusal carries the state folded so far, so the caller can still read the geometry
            the chain reached before the stage that stopped it."""
            return False, (), evolved, reason

        for stage_index, stage in enumerate(stages):
            loc = stage.patch_locality(Attribute(evolved))
            localities.append(loc)
            label = f"stage {stage_index} '{_stage_name(stage)}'"
            if loc.kind in (LocalityKind.WHOLE_VOLUME, LocalityKind.SLAB):
                # SLAB is a write-side contract: its side effect needs the slab's place in the
                # OUTPUT, which a patch read has no notion of. A stage that is whole-volume only
                # because something was left undeclared says so itself (PatchLocality.reason), so
                # the reader is told what to change instead of what happened.
                return refuse(f"{label} declares {loc.kind.name}: {loc.reason or 'it needs the whole volume'}.")
            if loc.kind is LocalityKind.GLOBAL_STAT:
                # The seed is the STORED volume's statistic; otherwise ([Clip(-200, 400), Standardize()])
                # every patch would be standardized by the pre-Clip statistic: fall back to the whole volume.
                if not stat_seed_valid(localities[:-1]):
                    return refuse(
                        f"{label} needs whole-volume statistics, but an earlier stage changes the values"
                        ": the stored volume's statistic is not this stage's input."
                    )
                unknown = sorted(set(loc.stat_keys) - _STREAM_STAT_KEYS)
                if unknown:
                    return refuse(f"{label} needs statistics {unknown} that no source can provide.")
                if seed_statistics and self._statistics_seeded:
                    self._ensure_stream_stats(
                        source_dataset,
                        source_group,
                        source_entry,
                        cache_attribute,
                        set(loc.stat_keys),
                        loc.stat_channels,
                    )
                elif seed_statistics:
                    self._statistics_deferred = True
                # The evolving case state carries the seed too: a Save sweep writes it as the cache
                # header, exactly as the whole-volume pass leaves the statistic in the attribute.
                for stat_key in loc.stat_keys:
                    if stat_key in cache_attribute and stat_key not in evolved:
                        evolved[stat_key] = cache_attribute[stat_key]
            if loc.kind is LocalityKind.HALO and not self._affords_halo(a, loc.halo):
                return refuse(
                    f"{label} declares a halo of {loc.halo} that is too wide for this grid to be worth"
                    " reading (over half the patch extent per axis)."
                )
            plan = self._plan_read_stage(stage, loc, shape, evolved)
            plans.append(plan)
            shape = list(plan.out_shape)
        expected = landing_shape if landing_shape is not None else self.shapes[a]
        if shape != [int(extent) for extent in expected]:
            return refuse(
                f"the chain's shapes fold to {shape} but the target grid is"
                f" {[int(extent) for extent in expected]}: a stage's shape map is missing or wrong."
            )
        return True, tuple(plans), evolved, None

    def _plan_read_stage(
        self, stage: Stage, loc: PatchLocality, shape: list[int], evolved: Attribute
    ) -> "_ReadStagePlan":
        """One stage's slot in the composed plan: its shapes, its pull map, and (for a region stage)
        the geometry it leaves for the stages after it (``write_stream_cache_attribute``)."""
        if not loc.kind.is_region:
            return _ReadStagePlan(loc.kind, tuple(shape), tuple(shape), None)
        if loc.kind is LocalityKind.HALO:
            return _ReadStagePlan(
                loc.kind, tuple(shape), tuple(shape), _HaloPull(_halo_radii(loc.halo, len(shape)), list(shape))
            )
        # ORIENTATION / CROP / REGRID: the stage's own remap, on the state the stages before it left.
        pull = _RemapPull(stage.stream_region_source, list(shape), Attribute(evolved), self.name)
        measured = getattr(stage, "measured_region_source", None)
        run_pull = (
            _RemapPull(measured, list(shape), Attribute(evolved), self.name)
            if measured is not None and getattr(stage, "measures_at_run", False)
            else None
        )
        out = self._stage_out_shape(stage, shape, Attribute(evolved))
        stage.write_stream_cache_attribute(evolved, list(shape), self.name)
        return _ReadStagePlan(loc.kind, tuple(shape), tuple(out), pull, run_pull)

    def _stage_out_shape(self, stage: Stage, shape: list[int], attribute: Attribute) -> list[int]:
        """The spatial shape one stage folds ``shape`` to: a transform's map or a draw's own.

        The one dispatch between the two Stage species' shape vocabularies: a ``Transform`` restates
        its fold as ``transform_shape``, an :class:`AugmentedStage` as its draw's ``stream_shape``.
        Shape only: the geometry transition is :meth:`_fold_case_state`'s half.
        """
        if isinstance(stage, Transform):
            return [int(e) for e in stage.transform_shape(self.group_src, self.name, list(shape), attribute)]
        return [int(e) for e in cast(AugmentedStage, stage).stream_shape(list(shape))]

    def _fold_case_state(self, stage: Stage, shape: list[int], attribute: Attribute) -> list[int]:
        """Fold one stage over the evolving case state: the shape through its map, the geometry
        through its stated transition: the idiom :meth:`_plan_read_stage` runs per region stage.

        Every landing fold goes through here, so a stage is judged on the state the stages before it
        left rather than on the stored header: a ``Resample`` behind a ``Canonical`` records the
        reoriented grid, and a second ``Resample`` sees the first one's spacing.
        """
        out = self._stage_out_shape(stage, shape, attribute)
        if isinstance(stage, Transform):
            stage.write_stream_cache_attribute(attribute, list(shape), self.name)
        return out

    @staticmethod
    def _adopt_case_facts(folding: Attribute, case: Attribute) -> None:
        """Keep what a landing fold computed about the CASE (Crop's content-derived box) off its
        walk state. The geometry the fold evolved is the walk's own (the run re-makes those
        transitions); the box is expensive, immutable per case, and read by every later fold, the
        streamed replays and the run itself."""
        if "box" in folding and "box" not in case:
            case["box"] = folding["box"]

    def chain_stages(self, a: int = 0) -> list[Stage]:
        """The ordered stages copy ``a`` is made of: the one definition of what a copy IS.

        Behind an :class:`Expand`, the shared prefix, then the copy's own draw at the marker's
        position, then the per-copy tail. Without one, the chain itself, with any draw appended
        last: the training order, where an augmentation is a copy of the chain's whole result.
        """
        if self._expand is None:
            return [*self.transforms, *self._augmentation_stages(a)]
        return [*self._expand_pre, *self._expand_tail(a)]

    def write_targets(self, a: int = 0) -> list[tuple[Save, list[int], Attribute]]:
        """Every ``Save`` copy ``a`` writes, with the extent and case state it lands at.

        What a write probe must open to be the run's own verdict: behind an ``Expand`` the stages
        after the marker fold the COPY's grid, so probing the chain with the marker left in it
        would validate the pre-draw extent and call a destination good for a shape it never sees.
        """
        spatial = [int(extent) for extent in self.base_shape[1:]]
        attributes = Attribute(self.cache_attributes_bak[0])
        targets: list[tuple[Save, list[int], Attribute]] = []
        for stage in self.chain_stages(a):
            spatial = self._fold_case_state(stage, spatial, attributes)
            if isinstance(stage, Save):
                targets.append((stage, list(spatial), Attribute(attributes)))
        return targets

    def _resolve_patch_stream_source(self, a: int, apply_augmentations: bool = True) -> _PatchStreamSource | None:
        key = (a, apply_augmentations)
        if key in self._patch_stream_sources:
            return self._patch_stream_sources[key]

        source_dataset = self.dataset
        source_group = self.group_src
        source_entry = self.name
        source_shape = list(self.base_shape)
        # Plan from the case as STORED (the pristine backup), never from the live attribute: the live
        # one carries what earlier patches or epochs wrote (a Resample's target Spacing, a Canonical's
        # canonical Direction), and planning from it would hand a stage its own output as the
        # description of its input on every epoch after the first.
        stream_cache_attribute = Attribute(self.cache_attributes_bak[0])
        pending: list[_PendingSweep] = []
        trailing_transforms: list[Stage] = []
        sweep_refusal: str | None = None
        # The entry name Saves write under: the case's own before the Expand marker, the copy's own
        # after it. `splice_at` is where the per-copy stages begin WITHIN the current segment (None =
        # the segment is entirely shared), which the expansion engine reads to share one read pass.
        entry = self.name
        splice_at: int | None = None
        past_expand = False

        walked: list[Stage] = (
            list(self.transforms) if self._expand is None else [*self._expand_pre, self._expand, *self._expand_tail(a)]
        )
        for transform in walked:
            if isinstance(transform, Expand):
                # The marker is replaced by the copy's own draw: everything before it is the case's,
                # everything after it (including every Save destination) is the copy's.
                splice_at = len(trailing_transforms)
                past_expand = True
                entry = self.copy_entry(a)
                continue
            if isinstance(transform, Save):
                dataset, group = save_destination(transform, self.dataset, self.group_dest)
                if not self._rewrite_saves and dataset.is_dataset_exist(group, entry):
                    source_dataset, source_group, source_entry = dataset, group, entry
                    source_shape, boundary_attributes = dataset.get_infos(group, entry)
                    source_shape = list(source_shape)
                    # Streaming from a Save cache: the stored volume is the cache, so the stages after
                    # the boundary read its geometry: stacked over the source keys exactly as the
                    # whole-volume cache-hit merges the cached header.
                    stream_cache_attribute = Attribute(self.cache_attributes_bak[0])
                    for attribute_key, attribute_value in boundary_attributes.items():
                        stream_cache_attribute[attribute_key] = attribute_value
                    pending.clear()
                    trailing_transforms = []
                    # A satisfied per-copy cache already holds the draw: the new segment is entirely
                    # per-copy, which `splice_at = 0` says.
                    splice_at = 0 if past_expand else None
                    sweep_refusal = None
                    continue
                planned, planned_attribute, planned_refusal = self._plan_save_sweep(
                    dataset,
                    group,
                    entry,
                    trailing_transforms,
                    splice_at if splice_at is not None else len(trailing_transforms),
                    source_dataset,
                    source_group,
                    source_entry,
                    source_shape,
                    stream_cache_attribute,
                    seed_statistics=not pending,
                )
                if planned is not None and planned_attribute is not None:
                    stream_cache_attribute = planned_attribute
                    source_dataset, source_group, source_entry = dataset, group, entry
                    source_shape = [source_shape[0], *planned.out_spatial]
                    pending.append(planned)
                    trailing_transforms = []
                    splice_at = 0 if past_expand else None
                    continue
                # An unplannable Save stays in the chain, where its WHOLE_VOLUME declaration refuses
                # the whole plan: keep the sweep's own reason, or the chain-level one would only
                # ever say "Save needs the whole volume" and mask the actual cause.
                if sweep_refusal is None:
                    sweep_refusal = planned_refusal
            trailing_transforms.append(transform)

        # What copy `a` is. Without an Expand, the training order: the trailing transforms, then
        # its own draw appended last. With one, the draw was spliced at the marker above, and the
        # whole thing is planned as one chain either way: a region transform and a region
        # augmentation are then two regions, which is exactly what they are.
        if self._expand is None:
            stages = trailing_transforms + (self._augmentation_stages(a) if apply_augmentations else [])
        else:
            stages = trailing_transforms

        streamable, stage_plans, evolved, chain_refusal = self._plan_stream_region(
            a,
            stages,
            source_dataset,
            source_group,
            source_entry,
            stream_cache_attribute,
            list(source_shape[1:]),
            seed_statistics=not pending,
        )
        if not streamable:
            self._stream_refusals[key] = sweep_refusal or chain_refusal or "the chain cannot stream."
            self._patch_stream_sources[key] = None
        elif pending:
            # The pending source only answers the regime probes: no attribute is persisted and no
            # patch flows from it: the sweeps run at first data access, and the source is then
            # re-resolved from the materialized caches (the satisfied-Save path above).
            self._patch_stream_sources[key] = _PatchStreamSource(
                source_dataset, source_group, source_entry, source_shape, stages, stage_plans, tuple(pending)
            )
        else:
            self.cache_attributes[a] = Attribute(stream_cache_attribute)
            self.cache_attributes_bak[a] = Attribute(stream_cache_attribute)
            self._patch_stream_sources[key] = _PatchStreamSource(
                source_dataset, source_group, source_entry, source_shape, stages, stage_plans
            )
        # The state the whole plan lands on, kept for consumers that need the LANDED geometry (a
        # reduction seeding its output header) without re-walking the chain. Recorded only when the
        # plan HOLDS: a refused plan folded as far as the stage that refused and no further, and half
        # a fold is not a geometry: it is a Spacing from before the resample meant to change it.
        # An unset key is what lets ``landed_attributes`` answer with the stored state instead.
        if streamable:
            self._stream_evolved[key] = Attribute(evolved)
        return self._patch_stream_sources[key]

    def _plan_save_sweep(
        self,
        destination: Dataset,
        group: str,
        entry: str,
        segment: list[Stage],
        copy_stage_start: int,
        source_dataset: Dataset,
        source_group: str,
        source_entry: str,
        source_shape: list[int],
        base_attributes: Attribute,
        seed_statistics: bool,
    ) -> tuple[_PendingSweep | None, Attribute | None, str | None]:
        """Plan the materialization of one unsatisfied :class:`Save`, or refuse with the reason and
        leave it on the whole-volume path: the segment feeding it must itself stream, and the
        destination must serve region writes (probed by capability, so a refusal costs nothing).
        Returns ``(sweep, evolved, None)`` on success (the pending sweep and the case state its
        cache will carry, which the stages after the Save plan against), and ``(None, None,
        reason)`` on refusal."""
        if self._sweep_failure is not None:
            return (
                None,
                None,
                f"an earlier sweep failed for this case, so every Save takes the whole-volume"
                f" path. {self._sweep_failure}",
            )
        if not env_flag("KONFAI_STREAMED_WRITES", True):
            return None, None, "KONFAI_STREAMED_WRITES=0 disables streamed writes."
        landing = [int(extent) for extent in source_shape[1:]]
        probe = Attribute(base_attributes)
        for stage in segment:
            landing = self._fold_case_state(stage, landing, probe)
        planning = Attribute(base_attributes)
        streamable, stage_plans, evolved, refusal = self._plan_stream_region(
            0,
            segment,
            source_dataset,
            source_group,
            source_entry,
            planning,
            [int(extent) for extent in source_shape[1:]],
            landing_shape=landing,
            seed_statistics=seed_statistics,
        )
        if not streamable:
            return None, None, refusal
        if not destination.can_stream_data(evolved):
            return (
                None,
                None,
                f"destination '{destination.filename}' cannot serve region writes for this entry"
                " (h5 and omezarr always can; mha only with image geometry).",
            )
        sweep = _PendingSweep(
            destination,
            group,
            entry,
            list(segment),
            source_dataset,
            source_group,
            source_entry,
            list(source_shape),
            tuple(landing),
            planning,
            min(copy_stage_start, len(segment)),
            stage_plans,
        )
        return sweep, evolved, None

    def can_stream_patch(self, a: int, apply_augmentations: bool = True) -> bool:
        return self._resolve_patch_stream_source(a, apply_augmentations) is not None

    def stream_refusal(self, a: int = 0, apply_augmentations: bool = True) -> str | None:
        """Why this copy cannot stream (the reified refusal) or ``None`` when it can.

        Resolves the plan (a probe, never a write) and hands back the first stage-level reason the
        planner met: the whole-volume fallback stays available, but it stops being silent."""
        if self._resolve_patch_stream_source(a, apply_augmentations) is not None:
            return None
        return self._stream_refusals.get((a, apply_augmentations), "the chain cannot stream.")

    @property
    def spatial_shape(self) -> list[int]:
        """The spatial extent this case's chain lands on: the source folded by every stage."""
        return list(self.shapes[0])

    @property
    def stored_attributes(self) -> Attribute:
        """The case as STORED: the geometry of the entry on disk, before any stage ran.

        A copy, and pristine on purpose: the live attribute carries what earlier regions or epochs
        wrote into it, so anything planning against it would be handed a stage's own output as the
        description of its input.
        """
        return Attribute(self.cache_attributes_bak[0])

    def landed_attributes(self, a: int = 0) -> Attribute:
        """The case state the chain LANDS on: stored geometry folded by every stage's rewrite.

        This is what an output built FROM the chain's result must carry as its header (a Resample's
        target ``Spacing``, a Canonical's direction), where :attr:`stored_attributes` is the source's
        own. Resolved from the plan (a probe, never a write); a chain that cannot stream answers with
        the stored state, the only honest one available without assembling the volume.
        """
        self._resolve_patch_stream_source(a, apply_augmentations=False)
        evolved = self._stream_evolved.get((a, False))
        return Attribute(evolved) if evolved is not None else self.stored_attributes

    def read_region(self, target: tuple[slice, ...], a: int = 0, apply_augmentations: bool = False) -> torch.Tensor:
        """Run this case's chain over one region of its output, reading only what that region pulls.

        The public form of the streamed read: the region stages' pull maps fold backward to the
        stored volume, so one bounded read serves the whole chain and the volume is never assembled.
        ``target`` indexes the SPATIAL axes of :attr:`spatial_shape`; every channel is read.

        Raises rather than falling back, because a caller asking for one region has already decided
        the volume does not fit: quietly loading it would answer a question they did not ask.
        :meth:`stream_refusal` gives the reason before committing to this path.

        Goes through :meth:`_stream_ready`, so a chain reading through an unwritten ``Save`` gets it
        swept first, exactly as the DataLoader path does: resolving the source directly would plan
        green against caches that do not exist yet and fail at the first region.
        """
        if not self._stream_ready(a, apply_augmentations):
            raise PatchError(
                f"Case '{self.name}' cannot stream its chain, so it cannot serve a region.",
                self.stream_refusal(a, apply_augmentations)
                or self._sweep_failure
                or "See stream_refusal() for the refusing stage.",
            )
        source = self._resolve_patch_stream_source(a, apply_augmentations)
        if source is None:  # pragma: no cover - _stream_ready just resolved it
            raise PatchError(
                f"Case '{self.name}' cannot stream its chain, so it cannot serve a region.",
                self.stream_refusal(a, apply_augmentations) or "See stream_refusal() for the refusing stage.",
            )
        tensor, _attribute, _keys = self._replay_streamed_region(source, target, self.stored_attributes, None)
        return tensor

    @contextlib.contextmanager
    def _chain_device_scope(self, device: "torch.device | None") -> Iterator[None]:
        """Route the chain onto ``device`` for one materialization, restored on every exit path:
        a device that outlived the call would move a later ``get_data()`` onto CUDA inside a
        DataLoader worker."""
        previous = self._chain_device
        self._chain_device = device if device is not None and device.type != "cpu" else None
        try:
            yield
        finally:
            self._chain_device = previous

    def materialize(
        self,
        a: int = 0,
        rewrite: bool = False,
        fallback_budget_bytes: float | None = None,
        allow_fallback: bool = True,
        prefer_whole: bool = False,
        device: "torch.device | None" = None,
        release: bool = True,
    ) -> bool:
        """Write this case's chain to disk by the cheapest path that can, and say which one it took.

        Two branches, and they are the whole write side: the streamed one sweeps every unsatisfied
        :class:`Save` slab by slab and never holds the volume; when it refuses (a WHOLE_VOLUME
        stage, a destination that cannot serve region writes, a sweep that failed) the classic load
        writes the same caches from the assembled volume. The bytes are the same either way; only
        the peak memory differs, which is why the caller must be told the answer rather than left to
        assume the cheap one. Returns whether the streamed path wrote the case.
        ``allow_fallback=False`` raises instead of taking the second branch, for a caller whose
        contract is that a fallback refuses rather than costs a volume.

        Never routed through :meth:`get_data`: a case with no declared patch is a single
        whole-volume patch, so asking for data would read back the volume this just wrote. The
        volume is released before returning, so a caller sweeping a dataset holds one case at a
        time. A chain whose caches all exist already writes nothing and streams: that is the
        per-case resume; a chain with no :class:`Save` at all writes nothing either, which is why
        declaring an output is checked before a case is ever handed here. ``rewrite=True`` recomputes
        the case from the source and every stream's finalize renames over the old entry: the forced
        path behind ``--overwrite``, never the default. ``release=False`` keeps the loaded volume
        for the caller that owns the release: the expansion engine, whose solo copies of one case
        share the assembled prefix.
        """
        self._set_rewrite(rewrite)
        self.set_memory_budget(fallback_budget_bytes)
        # Opt-in only, and only here: this same machinery loads training cases inside DataLoader
        # workers, where a CUDA default would be wrong. The transformer's rank is the one caller
        # that knows its device.
        with self._chain_device_scope(device):
            # A chain with an Expand materializes a COPY, whose draw must be part of the plan;
            # without one it materializes the case itself, and augmentations have nothing to do
            # with writing.
            apply_augmentations = self._expand is not None and a > 0
            if prefer_whole:
                # The plan chose to LOAD: the case fits its budget and streaming would re-read the
                # source (:meth:`predicted_stream_read_factor`). A choice, not a fallback: nothing
                # failed, so ``allow_fallback`` is not consulted; the budget check stays as the belt.
                self._enforce_fallback_budget(fallback_budget_bytes)
                self._assemble_and_write(a)
                if release:
                    self.unload()
                return False
            if self._stream_ready(a, apply_augmentations=apply_augmentations):
                return True
            if not allow_fallback:
                raise PatchError(
                    f"Case '{self.name}' would take the whole-volume path at run time.",
                    self.stream_refusal(a, apply_augmentations) or self._sweep_failure or "the chain cannot stream.",
                    "Nothing was written for this case; the caller forbids the whole-volume fallback.",
                )
            self._enforce_fallback_budget(fallback_budget_bytes)
            self._assemble_and_write(a)
            if release:
                self.unload()
            return False

    def set_memory_budget(self, budget_bytes: float | None) -> None:
        """The per-rank budget this case's streamed sweeps size their slabs against.

        Public because the budget reaches a manager from more than one door: :meth:`materialize`
        takes it as an argument, while a reduction only ever calls :meth:`read_region`: which
        sweeps pending Saves as a side effect and must sweep them under the same bound.
        """
        self._sweep_budget_bytes = budget_bytes

    def _set_rewrite(self, rewrite: bool) -> None:
        if rewrite == self._rewrite_saves:
            return
        # The memoized plans were probed under the other boundary answer: replan. And replan from
        # the case as STORED: an earlier boundary-based plan wrote the OUTPUT's header into the
        # backup (its Spacing, its Size), and a rewrite planned from that geometry re-writes
        # untransformed data over the deliverable without an error.
        self._rewrite_saves = rewrite
        self._patch_stream_sources.clear()
        self._stream_refusals.clear()
        self._stream_evolved.clear()
        self._swept_entries.clear()
        self._sweep_failure = None
        self.cache_attributes_bak = copy.deepcopy(self._cache_attributes_pristine)
        self.cache_attributes = copy.deepcopy(self._cache_attributes_pristine)

    def peak_case_bytes(self) -> int:
        """The largest single tensor the whole-volume path holds, from the chain's declared shapes.

        The stored extent is a floor, not the answer: a chain that pads or resamples upward holds
        its LARGEST intermediate, and that is the one a budget has to cover. Every stage declares
        that extent through ``transform_shape``, so the fold is exact for the stages that know their
        own geometry and identity for the rest: it can only bring the figure closer, never lower.

        Headers-only still, and still a floor for the one thing it cannot see: a stage that widens
        the dtype beyond what it declares.
        """
        if self._peak_case_bytes is None:
            spatial = [int(extent) for extent in self.base_shape[1:]]
            channels = int(self.base_shape[0])
            peak = int(np.prod(self.base_shape, dtype=np.int64))
            attributes = Attribute(self.stored_attributes)
            for stage in self.transforms:
                if not isinstance(stage, Transform):
                    continue
                spatial = self._fold_case_state(stage, list(spatial), attributes)
                peak = max(peak, channels * int(np.prod(spatial, dtype=np.int64)))
            self._peak_case_bytes = peak * CASE_ELEMENT_BYTES
        return self._peak_case_bytes

    def predicted_stream_read_factor(self, a: int = 0, apply_augmentations: bool = False) -> float | None:
        """~How many times the streamed route reads the source, priced from the plan alone.

        Streaming is a memory strategy, not a speed strategy: splitting re-reads: a halo re-reads
        its overlap, a regrid pulls each slab's window through its map, and a store that cannot
        serve bounded region reads decodes the whole volume once per slab, where loading reads the
        source once. This is the number the route is CHOSEN with, never the answer: headers only,
        one representative slab priced through the plan's own pull maps. ``None`` when the chain
        cannot stream (there is no route to price). A chain reading through unmaterialized Save
        caches is priced on its top-level plans: a proxy, close enough to route by.
        """
        source = self._resolve_patch_stream_source(a, apply_augmentations)
        if source is None:
            return None
        # Each unsatisfied Save sweeps ITS source; past the last boundary the chain reads the
        # materialized cache. The route is priced by the dominant segment: a max, not a sum --
        # the segments read different stores, and one that re-reads is the cost either way.
        factors = [
            self._segment_read_factor(
                sweep.source_dataset,
                sweep.source_group,
                sweep.source_entry,
                [int(extent) for extent in sweep.source_shape],
                list(sweep.out_spatial),
                sweep.stage_plans,
            )
            for sweep in source.pending_sweeps
        ]
        # The head segment past the last boundary is read only if stages follow it: a chain
        # ending on a Write leaves nothing to read from the destination, which does not exist yet.
        if source.stage_plans:
            landed = list(source.stage_plans[-1].out_shape)
            factors.append(
                self._segment_read_factor(
                    source.dataset, source.group, source.entry, list(source.shape), landed, source.stage_plans
                )
            )
        return max(factors) if factors else 1.0

    def _segment_read_factor(
        self,
        dataset: Dataset,
        group: str,
        entry: str,
        source_shape: list[int],
        landed: list[int],
        plans: tuple[_ReadStagePlan, ...],
    ) -> float:
        """One segment's reads over its source's voxels, slab by slab through the plan's own pulls."""
        rows = self._sweep_rows(list(landed), int(source_shape[0]))
        if not dataset.bounded_region_reads(group, entry):
            return float(max(1, -(-landed[0] // rows)))  # every slab decodes the whole store
        read = 0
        for start in range(0, landed[0], rows):
            span = [slice(start, min(start + rows, landed[0])), *(slice(0, extent) for extent in landed[1:])]
            for plan in reversed(plans):
                span = list(plan.pull(tuple(span))) if plan.pull is not None else span
            read += int(np.prod([max(0, part.stop - part.start) for part in span], dtype=np.int64))
        return float(read) / float(max(1, int(np.prod(source_shape[1:], dtype=np.int64))))

    def _enforce_fallback_budget(self, fallback_budget_bytes: float | None) -> None:
        if fallback_budget_bytes is None:
            return
        # The plan's promise holds at run time too: a case whose sweep failed here (a refusal the
        # probe could not see) must not assemble a volume the budget cannot hold.
        case_bytes = self.peak_case_bytes() * FALLBACK_INFLIGHT_FACTOR
        if case_bytes > fallback_budget_bytes:
            raise PatchError(
                f"Case '{self.name}' fell back to the whole-volume path at run time and its"
                f" working set (~{case_bytes / 2**30:.2f} GiB) exceeds the per-rank budget.",
                "Nothing was written for this case. Raise 'memory_budget' or make the chain streamable.",
            )

    def _assemble_and_write(self, a: int) -> None:
        """The whole-volume fallback: assemble and write every Save. The caller releases.

        Without an :class:`Expand` this is the classic load of the full chain. With one, the shared
        part is assembled once (``load`` keeps it, so the copies of one case reuse the same tensor
        across calls) and only the copy's draw and the per-copy stages run per copy, writing their
        Saves under the copy's name.
        """
        if self._expand is None:
            self.load(self.transforms, [], load_augmentations=False)
            return
        self.load(self._expand_pre, [], load_augmentations=False)
        tensor = self.data[0].clone()
        attribute = Attribute(self.cache_attributes[0])
        self._apply_chain(tensor, self._expand_tail(a), attribute, self.copy_entry(a))

    def materialize_copies(
        self,
        copies: list[int],
        rewrite: bool = False,
        fallback_budget_bytes: float | None = None,
        allow_fallback: bool = True,
        device: "torch.device | None" = None,
    ) -> dict[int, str]:
        """Write the :class:`Expand` copies of this case by the cheapest regime each can take.

        The expansion engine, and the reason expansion is not just a loop over
        :meth:`materialize`: reads are decompression-bound, so N independent sweeps decompress the
        same source N times. Copies whose per-copy stages are all pointwise share ONE read pass:
        each slab is read and carried through the shared prefix once, then every copy applies its
        own draw and writes into its own stream. A copy whose draw reads regions (a rotation's remap, a
        halo) sweeps its own pass; a copy that cannot stream falls back to the whole volume, whose
        shared part is assembled once for all such copies. Returns the regime each copy took:
        ``'stream-shared'`` | ``'stream'`` | ``'whole-volume'``.
        """
        if self._expand is None:
            raise PatchError(
                f"materialize_copies() on case '{self.name}' whose chain declares no Expand.",
                "Use materialize() for a 1-to-1 chain; copies only exist behind an Expand marker.",
            )
        self._set_rewrite(rewrite)
        self.set_memory_budget(fallback_budget_bytes)
        self._require_statistics()
        with self._chain_device_scope(device):
            verdicts: dict[int, str] = {}
            if not copies:
                return verdicts

            # The caches the copies share (pre-Expand Saves) are swept once, first: every copy's
            # plan reads through them, and sweeping them inside each copy's own pass would redo
            # the work.
            first = self._resolve_patch_stream_source(copies[0], apply_augmentations=True)
            if first is not None:
                shared_pending = [sweep for sweep in first.pending_sweeps if sweep.entry == self.name]
                if shared_pending:
                    for sweep in shared_pending:
                        # Chained here as on the per-case path: past a failure the next sweep reads
                        # a cache nobody wrote and overwrites the recorded reason with its own
                        # symptom.
                        if not self._materialize_save(sweep):
                            break
                    self._invalidate_stream_plans()

            shared: list[tuple[int, _PendingSweep]] = []
            solo: list[int] = []
            fallback: list[int] = []
            for a in copies:
                source = self._resolve_patch_stream_source(a, apply_augmentations=True)
                if source is None:
                    fallback.append(a)
                    continue
                per_copy = [sweep for sweep in source.pending_sweeps if sweep.entry != self.name]
                if len(per_copy) != len(source.pending_sweeps):
                    # A shared cache is still pending (its sweep failed): the per-copy path will
                    # retry and fall back on its own terms.
                    solo.append(a)
                elif not per_copy:
                    verdicts[a] = "stream"  # everything this copy writes is already on disk
                elif len(per_copy) == 1 and self._pointwise_tail(per_copy[0]):
                    shared.append((a, per_copy[0]))
                else:
                    solo.append(a)

            if len(shared) == 1:
                # One copy shares with nobody: its own sweep is the same work without the extra
                # clone.
                solo.append(shared.pop()[0])
            if shared:
                written = self._materialize_shared_pass(shared)
                for a, _sweep in shared:
                    if a in written:
                        verdicts[a] = "stream-shared"
                    else:
                        solo.append(a)
                if written:
                    self._invalidate_stream_plans()

            for a in sorted(solo):
                verdicts[a] = (
                    "stream"
                    if self.materialize(
                        a,
                        rewrite,
                        fallback_budget_bytes=fallback_budget_bytes,
                        allow_fallback=allow_fallback,
                        device=device,
                        release=False,  # the copies share the assembled prefix; released once below
                    )
                    else "whole-volume"
                )
            if fallback:
                if not allow_fallback:
                    raise PatchError(
                        f"{len(fallback)} cop(ies) of case '{self.name}' would take the whole-volume path at run time.",
                        self.stream_refusal(fallback[0], apply_augmentations=True)
                        or "the copies' chains cannot stream.",
                        "Nothing was written for these copies; the caller forbids the whole-volume fallback.",
                    )
                self._enforce_fallback_budget(fallback_budget_bytes)
                for a in fallback:
                    self._assemble_and_write(a)
                    verdicts[a] = "whole-volume"
            self.unload()
            return verdicts

    def _pointwise_tail(self, sweep: _PendingSweep) -> bool:
        """Whether everything per-copy in this sweep is a pure value map on its slab.

        That is the shared-pass criterion: a pointwise tail pulls exactly the slab the shared prefix
        landed on, so every such copy can consume one read. A region draw pulls its own geometry and
        a GLOBAL_STAT reads a statistic of ITS input; both get their own pass.
        """
        return all(plan.kind is LocalityKind.POINTWISE for plan in sweep.stage_plans[sweep.copy_stage_start :])

    def expansion_solo_reason(self, a: int) -> str | None:
        """Why copy ``a`` cannot join the shared read pass, for the plan, never a write.

        ``None`` when it can (or when nothing is pending for it). The reason names the first
        per-copy stage whose declared locality is not pointwise: that is the stage whose pull makes
        this copy's read geometry its own.
        """
        source = self._resolve_patch_stream_source(a, apply_augmentations=True)
        if source is None:
            return None
        per_copy = [sweep for sweep in source.pending_sweeps if sweep.entry != self.name]
        if len(per_copy) == 1 and self._pointwise_tail(per_copy[0]):
            return None
        if not per_copy:
            return None
        if len(per_copy) > 1:
            return "the copy's chain crosses more than one per-copy Save, so it sweeps its own passes."
        sweep = per_copy[0]
        for stage, plan in zip(
            sweep.stages[sweep.copy_stage_start :], sweep.stage_plans[sweep.copy_stage_start :], strict=False
        ):
            if plan.kind is not LocalityKind.POINTWISE:
                return (
                    f"stage '{_stage_name(stage)}' of this copy declares {plan.kind.name},"
                    " so its read geometry is the draw's own and it sweeps its own pass."
                )
        return "the copy's plan is incomplete; it sweeps its own pass."

    @staticmethod
    def _sweep_targets(spatial: list[int], rows: int) -> Iterator[tuple[slice, ...]]:
        """The sweep's slabs: whole rows of the landing's first axis, full extent on the rest."""
        for start in range(0, spatial[0], rows):
            yield (slice(start, min(start + rows, spatial[0])), *(slice(0, extent) for extent in spatial[1:]))

    @staticmethod
    def _sweep_header(evolved: Attribute, scope: Attribute, keys_before: set[str]) -> Attribute:
        """The header a sweep publishes: the plan-time case state, completed by what the chain's
        stages added in ``__call__``: the same keys the whole-volume pass would leave at the Save."""
        attributes = Attribute(evolved)
        for key in scope.keys():
            if key not in keys_before and key not in attributes:
                attributes[key] = scope[key]
        return attributes

    @staticmethod
    def _require_channel_first(block: np.ndarray, spatial: list[int], what: str) -> None:
        """Refuse a slab that lost the channel-first layout. Writing it anyway is the worst outcome
        available: the header would take the slab's first spatial extent for a channel count and
        publish a store of that many "channels", raising nothing, while the whole-volume path
        returns the right rank: the two would silently disagree."""
        if block.ndim != len(spatial) + 1:
            raise PatchError(
                f"{what} returned a rank-{block.ndim} slab where the channel-first layout needs rank"
                f" {len(spatial) + 1} (C, {', '.join(str(extent) for extent in spatial)}).",
                "A transform that reduces the leading axis must keep it (`keepdim=True`), so a slab"
                " stays C[Z]YX; only then does a region write mean the same thing as the"
                " whole-volume pass.",
            )

    @staticmethod
    def _open_sweep_stream(
        sweep: _PendingSweep, block: np.ndarray, spatial: list[int], rows: int, attributes: Attribute
    ) -> DataStream | None:
        """One region-write stream shaped for the sweep: the store chunks on whole slabs (channels
        included), so no region write ever pays a read-modify-write."""
        return sweep.destination.open_data_stream(
            sweep.group,
            sweep.entry,
            [int(block.shape[0]), *spatial],
            block.dtype,
            attributes,
            region_shape=[int(block.shape[0]), rows, *spatial[1:]],
        )

    def _materialize_shared_pass(self, shared: list[tuple[int, _PendingSweep]]) -> set[int]:
        """One read pass, N write streams: the optimal regime of the expansion engine.

        Each slab of the shared landing is read and computed through the shared prefix ONCE; each
        copy then applies its pointwise tail to a clone and writes into its own stream. Peak memory
        is the pulled slab plus one copy's block, whatever N is. On failure every open stream is
        aborted (entries publish by rename, so no partial copy survives), and the copies are
        handed back to their own passes.
        """
        reference = shared[0][1]
        # Defensive: the regime only holds when every copy lands the same grid from the same source
        # and writes into the same store. Built that way, but a draw that lies about its shape map
        # would corrupt N entries at once here, so the mismatch is checked rather than assumed.
        shared = [
            (a, sweep)
            for a, sweep in shared
            if sweep.out_spatial == reference.out_spatial
            and str(sweep.source_dataset.filename) == str(reference.source_dataset.filename)
            and sweep.source_group == reference.source_group
            and sweep.source_entry == reference.source_entry
            and str(sweep.destination.filename) == str(reference.destination.filename)
        ]
        if not shared:
            return set()
        prefix = list(reference.stages[: reference.copy_stage_start])
        spatial = list(reference.out_spatial)
        source_spatial = [int(extent) for extent in reference.source_shape[1:]]
        streamable, prefix_plans, prefix_evolved, _refusal = self._plan_stream_region(
            0,
            prefix,
            reference.source_dataset,
            reference.source_group,
            reference.source_entry,
            Attribute(reference.base_attributes),
            source_spatial,
            landing_shape=spatial,
        )
        if not streamable:
            return set()
        source = _PatchStreamSource(
            reference.source_dataset,
            reference.source_group,
            reference.source_entry,
            list(reference.source_shape),
            prefix,
            prefix_plans,
        )
        # Each copy's header is its OWN full-segment plan state: the tail's geometry and inversion
        # keys belong to the copy, not to the shared prefix.
        active: list[tuple[int, _PendingSweep, list[Stage], Attribute]] = []
        for a, sweep in shared:
            ok, _plans, evolved, _reason = self._plan_stream_region(
                0,
                sweep.stages,
                sweep.source_dataset,
                sweep.source_group,
                sweep.source_entry,
                Attribute(sweep.base_attributes),
                [int(extent) for extent in sweep.source_shape[1:]],
                landing_shape=list(sweep.out_spatial),
            )
            if ok:
                active.append((a, sweep, list(sweep.stages[sweep.copy_stage_start :]), evolved))
        if not active:
            return set()
        rows = self._sweep_rows(spatial, int(reference.source_shape[0]))
        streams: dict[int, DataStream] = {}
        try:
            for slab_index, target in enumerate(self._sweep_targets(spatial, rows)):
                tensor, slab_attribute, keys_before = self._replay_streamed_region(
                    source,
                    target,
                    Attribute(reference.base_attributes),
                    Attribute(prefix_evolved) if slab_index == 0 else None,
                )
                for a, sweep, tail, evolved in active:
                    copy_tensor = tensor.clone() if len(active) > 1 else tensor
                    scope = Attribute(slab_attribute)
                    for stage in tail:
                        copy_tensor = stage(self.name, copy_tensor, scope)
                    block = copy_tensor.cpu().numpy()
                    self._require_channel_first(block, spatial, f"A per-copy stage of '{sweep.group}/{sweep.entry}'")
                    if a not in streams:
                        attributes = self._sweep_header(evolved, scope, keys_before)
                        stream = self._open_sweep_stream(sweep, block, spatial, rows, attributes)
                        if stream is None:
                            raise PatchError(
                                f"destination '{sweep.destination.filename}' refused the region write"
                                f" of '{sweep.group}/{sweep.entry}' after accepting its plan.",
                                "h5 and omezarr always serve region writes; mha only with image geometry.",
                            )
                        stream.__enter__()
                        streams[a] = stream
                    streams[a].write_slice((slice(0, int(block.shape[0])), *target), block)
            written: set[int] = set()
            for a, sweep, _tail, _evolved in active:
                stream = streams.get(a)
                if stream is not None:
                    stream.close()
                    self._swept_entries.add((str(sweep.destination.filename), sweep.group, sweep.entry))
                    written.add(a)
            return written
        except BaseException as exception:
            for stream in streams.values():
                stream.abort(exception)
            if not isinstance(exception, Exception):
                raise  # an interrupt is not a sweep failure: no fallback, and no .tmp left behind
            warnings.warn(
                f"Shared-pass materialization of case '{self.name}' failed"
                f" ({type(exception).__name__}: {exception}); its copies take their own passes.",
                stacklevel=2,
            )
            return set()

    def _stream_ready(self, a: int, apply_augmentations: bool = True) -> bool:
        """Whether this copy can stream its patches, materializing what that requires.

        Resolves the source and, the first time a case whose chain reads through unmaterialized Save
        caches is actually asked for data, sweeps them and re-resolves from disk: all data then
        flows through the satisfied-boundary path, exactly as if the caches had always existed. The
        regime probes (``can_stream_patch``) answer from the plan alone and never write."""
        self._require_statistics()
        source = self._resolve_patch_stream_source(a, apply_augmentations)
        if source is None:
            return False
        if not source.pending_sweeps:
            return True
        for sweep in source.pending_sweeps:
            # Stop at the first failure. The sweeps are CHAINED: each one's source is the previous
            # one's destination, so past a failure the next reads a cache nobody wrote, fails too,
            # and overwrites the recorded reason with its own. _sweep_failure has to keep the cause,
            # not the last symptom.
            if not self._materialize_save(sweep):
                break
        # Every copy replans: the pending plans pointed at caches that did not exist yet (or, after a
        # failure, never will: _sweep_failed reroutes them to the whole-volume path).
        self._invalidate_stream_plans()
        return not self._sweep_failed and self._resolve_patch_stream_source(a, apply_augmentations) is not None

    @property
    def _sweep_failed(self) -> bool:
        return self._sweep_failure is not None

    def _invalidate_stream_plans(self) -> None:
        self._patch_stream_sources.clear()
        self._stream_refusals.clear()
        self._stream_evolved.clear()

    def _materialize_save(self, sweep: _PendingSweep) -> bool:
        """Write one Save cache slab by slab, without ever holding the volume.

        Each slab of the Save's own space is read through the segment (the same composed replay a
        patch uses), and region-written. The segment is re-planned here: its source is materialized
        by now, so a statistic the pending plan deferred seeds from the real entry (a disk-scan seed,
        so a GLOBAL_STAT segment matches the classic cache to float rounding (the streamed read's
        own contract), where every other kind matches it exactly). The cache appears only when
        complete (a DataStream renames on finalize), so a concurrent resolve never takes a partial
        entry as a source boundary; on failure the partial entry is removed, ``_sweep_failure``
        records why, and the case falls back to the whole-volume path, which writes the cache
        classically. Every ``False`` below leaves that reason behind: a caller with no fallback (a
        reduction reading through this cache) raises with it, so the answer is not "see the warning
        it emitted" for a warning a filter may have swallowed."""
        ledger_key = (str(sweep.destination.filename), sweep.group, sweep.entry)
        if ledger_key in self._swept_entries:
            # Already swept by THIS run: under --overwrite the existence probe answers "not written",
            # and re-sweeping a cache the copies share would redo the same work once per copy.
            return True
        if not self._rewrite_saves and sweep.destination.is_dataset_exist(sweep.group, sweep.entry):
            return True
        streamable, stage_plans, evolved, refusal = self._plan_stream_region(
            0,
            sweep.stages,
            sweep.source_dataset,
            sweep.source_group,
            sweep.source_entry,
            sweep.base_attributes,
            [int(extent) for extent in sweep.source_shape[1:]],
            landing_shape=list(sweep.out_spatial),
        )
        if not streamable:
            # The plan probe said yes and the re-plan against the materialized source says no: that
            # is new information, and it is the whole reason this case is about to cost a volume.
            return self._sweep_failed_because(
                sweep, refusal or "the segment feeding it no longer plans against its materialized source."
            )
        source = _PatchStreamSource(
            sweep.source_dataset,
            sweep.source_group,
            sweep.source_entry,
            list(sweep.source_shape),
            sweep.stages,
            stage_plans,
        )
        spatial = list(sweep.out_spatial)
        rows = self._sweep_rows(spatial, int(sweep.source_shape[0]))
        stream: DataStream | None = None
        try:
            for slab_index, target in enumerate(self._sweep_targets(spatial, rows)):
                # The first slab hands a throwaway case scope to the replay so the region-geometry
                # check runs: a region stage recording geometry nowhere the case can read refuses the
                # sweep (PatchError -> whole-volume fallback), exactly as the patch path refuses it.
                tensor, slab_attribute, keys_before = self._replay_streamed_region(
                    source, target, Attribute(sweep.base_attributes), Attribute(evolved) if slab_index == 0 else None
                )
                block = tensor.cpu().numpy()
                self._require_channel_first(
                    block, spatial, f"A stage of the chain writing '{sweep.group}/{sweep.entry}'"
                )
                if stream is None:
                    attributes = self._sweep_header(evolved, slab_attribute, keys_before)
                    stream = self._open_sweep_stream(sweep, block, spatial, rows, attributes)
                    if stream is None:
                        return self._sweep_failed_because(
                            sweep,
                            f"destination '{sweep.destination.filename}' refused the region write"
                            " after accepting its plan.",
                        )
                    stream.__enter__()
                stream.write_slice((slice(0, int(block.shape[0])), *target), block)
            if stream is not None:
                stream.close()
            self._swept_entries.add(ledger_key)
            return True
        except BaseException as exception:
            if stream is not None:
                stream.abort(exception)
            if not isinstance(exception, Exception):
                raise  # an interrupt is not a sweep failure: no fallback, and no .tmp left behind
            return self._sweep_failed_because(sweep, f"{type(exception).__name__}: {exception}")

    def _sweep_failed_because(self, sweep: _PendingSweep, reason: str) -> bool:
        """Record why a sweep gave up, warn, and answer ``False``: the one exit for all of them.

        The reason is kept, not only warned: a caller with a whole-volume fallback treats this as
        information, but one without (a reduction reading through this cache) has to raise, and it
        can only be as specific as what was kept here.
        """
        self._sweep_failure = f"'{sweep.group}/{sweep.entry}' could not be written region by region: {reason}"
        warnings.warn(f"{self._sweep_failure} Falling back to the whole-volume path.", stacklevel=3)
        return False

    def _sweep_rows(self, spatial: list[int], channels: int) -> int:
        """How many rows one sweep slab holds: the declared budget folded down, never up.

        The fixed default height is the CAP: chunk-friendly, needing no budget --
        and a declared ``memory_budget`` can only lower it: a sweep holds the pulled slab and the
        landed block (plus the write buffer), so half the budget over that working set is the honest
        height. Below one row nothing fits; the row is the floor, and the fallback budget check is
        what refuses a case whose single row cannot fit.

        ``channels`` is the SOURCE's, and ``_SWEEP_ELEMENT_BYTES`` assumes four: the landed block's
        channel count and dtype are not known until the first slab has been read, and the height has
        to be fixed before that so every slab writes whole chunks. A chain that multiplies channels
        (a one-hot, a field synthesis) is therefore under-counted here; the per-case fallback budget
        check runs on real shapes and is the authority that refuses.
        """
        cap = max(1, int(_SWEEP_SLAB_ROWS))
        budget = self._sweep_budget_bytes
        if not budget or budget <= 0:
            return cap
        plane = int(np.prod(spatial[1:], dtype=np.int64)) * max(1, int(channels)) * _SWEEP_ELEMENT_BYTES
        if plane <= 0:
            return cap
        return max(1, min(cap, int(budget * 0.5 / (plane * _SWEEP_RESIDENT_SLABS))))

    def _get_streamed_data(
        self,
        index: int,
        a: int,
        is_input: bool,
        apply_augmentations: bool = True,
    ) -> tuple[torch.Tensor, Attribute]:
        self._require_statistics()
        stream_source = self._resolve_patch_stream_source(a, apply_augmentations)
        if stream_source is None:
            raise RuntimeError("Patch streaming requested on a dataset manager without a streaming source.")
        if stream_source.pending_sweeps:
            raise PatchError(
                "Streamed read on a source with unmaterialized Save caches.",
                "Report this: _stream_ready() must run the sweeps before any patch flows.",
            )

        if stream_source.region_index is None:
            # POINTWISE / GLOBAL_STAT only: read the exact patch and run the whole chain on it.
            plan = self.patch.get_read_plan(stream_source.shape, index, a, is_input)
            data, attributes = stream_source.dataset.read_data_slice(
                stream_source.group, stream_source.entry, plan.data_slices
            )
            tensor = torch.from_numpy(data)
            cache_attribute = Attribute(self.cache_attributes_bak[a])
            cache_attribute.update(attributes)
            # Says the Min/Max/Mean/Std here are the planner's DISK seeds, not a mid-chain stage's
            # own bookkeeping (a Normalize pushes 'Min' for its inverse). Set before keys_before,
            # so it never persists past this read.
            cache_attribute["StatisticsSeeded"] = 1.0
            persist = a not in self._stream_attributes_persisted
            keys_before = set(cache_attribute.keys()) if persist else set()
            # Told where the patch sits, like every region: a per-voxel stage reading a companion
            # volume (a mask) reads the part that lines up with it.
            spatial = tuple(int(extent) for extent in stream_source.shape[1:])
            region = tuple(plan.data_slices[len(plan.data_slices) - len(spatial) :])
            context = RegionContext(region, region, spatial, spatial)
            for stage in stream_source.stages:
                tensor = stage.stream_region(self.name, tensor, context, cache_attribute)
            # The read plan is applied AFTER the chain, as the whole-volume path transforms before
            # Patch.get_data cuts: padding first would feed f(pad) to the model on every border patch.
            tensor = self.patch.apply_read_plan(tensor, plan)
            if persist:
                self._persist_stream_attributes(a, cache_attribute, keys_before)
            return tensor, cache_attribute

        return self._get_streamed_region_data(index, a, stream_source, is_input)

    def _finalize_stream_patch(self, tensor: torch.Tensor, index: int, a: int, is_input: bool) -> torch.Tensor:
        """Pad/format a target-extent streamed patch to ``patch_size`` like the whole-volume path.

        The region streamed paths produce a patch at the raw target-slice extent, but the
        overlap tiling can leave the last patch narrower than ``patch_size`` (integer-floor stride).
        The whole-volume ``Patch.get_data`` pads that border patch up to ``patch_size`` via
        ``apply_read_plan``; running the streamed patch through the SAME read plan makes border patches
        byte-identical between the two paths instead of one voxel short. The plan is built on
        ``self.shapes[a]``: the spatial grid this copy's patch slices were themselves cut on, which is
        the source shape reordered by a Permute and resized by a Resample.
        """
        plan = self.patch.get_read_plan(self.shapes[a], index, a, is_input)
        return self.patch.apply_read_plan(tensor, plan)

    def _persist_stream_attributes(self, a: int, cache_attribute: Attribute, keys_before: set[str]) -> None:
        # State a transform records for its own inversion (TensorCast's source dtype) must reach the
        # persistent attribute, as it would on the whole-volume path. Only NEWLY-added keys are copied:
        # a seeded GLOBAL_STAT or a case-level geometry key must not take a patch-local value.
        persistent = self.cache_attributes[a]
        persistent_keys = set(persistent.keys())
        for key, value in cache_attribute.items():
            if key not in keys_before and key not in persistent_keys:
                dict.__setitem__(persistent, key, value)
        self._stream_attributes_persisted.add(a)

    def _get_streamed_region_data(
        self,
        index: int,
        a: int,
        stream_source: _PatchStreamSource,
        is_input: bool,
    ) -> tuple[torch.Tensor, Attribute]:
        """Patch-native region chain: one target patch replayed through the composed region plans,
        padded back to ``patch_size`` like the whole-volume path (see ``_replay_streamed_region``,
        which a Save sweep drives with slab targets instead of patch targets)."""
        if self._expand is not None:
            self._refold_copy_records(a, stream_source)
        target_slices = tuple(self.patch.get_patch_slices(a)[index])
        # Each patch re-runs the chain from the state the whole-volume pass started from: the case as
        # stored (plus planned stats), never the live attribute: that one carries the chain's own
        # output.
        persist = a not in self._stream_attributes_persisted
        tensor, cache_attribute, keys_before = self._replay_streamed_region(
            stream_source,
            target_slices,
            Attribute(self.cache_attributes_bak[a]),
            self.cache_attributes[a] if persist else None,
        )
        tensor = self._finalize_stream_patch(tensor, index, a, is_input)
        if persist:
            self._persist_stream_attributes(a, cache_attribute, keys_before)
        return tensor, cache_attribute

    def _refold_copy_records(self, a: int, stream_source: _PatchStreamSource) -> None:
        """Re-fold copy ``a``'s chain state before replaying a region of it.

        A stage keys its per-case records by the CASE name (a stored transform is looked up by
        it), so the copies of an Expand share one key and the last WALK's records win. The write
        sweeps re-plan before sweeping and the whole-volume path re-records at call time; the
        patch replay is the consumer left over, and reading two copies interleaved would otherwise
        hand one copy the other's grids. Headers only: no voxel is read.
        """
        if not stream_source.stages:
            return
        shape = list(stream_source.stage_plans[0].in_shape)
        state = Attribute(self.cache_attributes_bak[a])
        for stage in stream_source.stages:
            shape = self._fold_case_state(stage, shape, state)

    def _replay_streamed_region(
        self,
        stream_source: _PatchStreamSource,
        target_slices: tuple[slice, ...],
        cache_attribute: Attribute,
        case_attribute: Attribute | None,
    ) -> tuple[torch.Tensor, Attribute, set[str]]:
        """Read only the source region a target region pulls, composed, and run the chain on it.

        The region stages' pull maps fold backward from the target region down to the stored volume (each
        stage's region pulls through the one before it), so one bounded read serves any number of
        them; the chain then runs forward over the sub-region, each stage on the region pair the same
        fold computed for it. HALO reads the region enlarged by the declared radius (clamped to the
        volume) and crops it back after the stage: the stage's own edge padding reproduces the
        whole-volume border once the clamp reaches the true border, so seams agree. ORIENTATION
        applies the index remap to what its region read; a CROP's remap IS its action, so the stage is
        not re-applied; REGRID interpolates the sub-region to its target extent. Only the composed
        region is requested; whether that avoids decoding the whole volume depends on the storage
        format: compressed MetaImage and NRRD decode the full volume per read
        (see ``_supports_region_read``).

        ``cache_attribute`` is this region's own scope, evolved by the chain; ``case_attribute``,
        when given, receives each region stage's case-level geometry (computed once, from the FULL
        shape of the stage's input). Returns the region tensor, the evolved scope, and the keys the
        scope held before the chain ran (whatever the chain added is what the caller may persist).
        """
        stages, plans = stream_source.stages, stream_source.stage_plans
        n_prefix = len(stream_source.shape) - len(target_slices)

        spans: list[list[slice]] = [list(target_slices)]
        for plan in reversed(plans):
            pull = plan.run_pull or plan.pull
            spans.append(pull(tuple(spans[-1])) if pull is not None else list(spans[-1]))
        spans.reverse()

        data_slices = tuple([slice(None)] * n_prefix + spans[0])
        data, attributes = stream_source.dataset.read_data_slice(stream_source.group, stream_source.entry, data_slices)
        tensor = torch.from_numpy(data)
        if self._chain_device is not None:
            tensor = tensor.to(self._chain_device)

        cache_attribute.update(attributes)
        cache_attribute["StatisticsSeeded"] = 1.0  # same contract as the pointwise route above
        keys_before = set(cache_attribute.keys())

        for stage, plan, source, target in zip(stages, plans, spans[:-1], spans[1:], strict=True):
            if not plan.kind.is_region:
                # The span is handed over rather than dropped: a stage reading a second aligned
                # volume needs to know WHICH part of it lines up with this region, and the
                # dispatcher is the only thing that knows. The default hook ignores it.
                tensor = stage.stream_region(
                    self.name,
                    tensor,
                    RegionContext(tuple(source), tuple(target), tuple(plan.in_shape), tuple(plan.out_shape)),
                    cache_attribute,
                )
                continue
            # A region stage's geometry writes describe the region's extent, not the volume's: give it
            # a throwaway scope, and write the case-level answer once from the FULL shape below
            # (write_stream_cache_attribute).
            scoped = Attribute(cache_attribute)
            if plan.kind is not LocalityKind.CROP:
                # A HALO stage is handed the ENLARGED region it asked for, and told so: what it
                # returns is cropped back to the target just below.
                tensor = stage.stream_region(
                    self.name,
                    tensor,
                    RegionContext(tuple(source), tuple(target), tuple(plan.in_shape), tuple(plan.out_shape)),
                    scoped,
                )
                if plan.kind is LocalityKind.HALO:
                    lead = tensor.dim() - len(target)
                    crop = [slice(t.start - s.start, t.stop - s.start) for t, s in zip(target, source, strict=False)]
                    tensor = tensor[(*[slice(None)] * lead, *crop)]
            if case_attribute is not None:
                stage.write_stream_cache_attribute(case_attribute, list(plan.in_shape), self.name)
                self._check_region_geometry_reaches_the_case(stage, scoped, cache_attribute)

        return tensor, cache_attribute, keys_before

    def _check_region_geometry_reaches_the_case(
        self, region_stage: Stage, scoped: Attribute, cache_attribute: Attribute
    ) -> None:
        """Refuse a region stage that records geometry nowhere the case can read it.

        A region stage is handed a patch, so what it records about the extent is one patch's answer:
        the scope it records into is thrown away, and ``write_stream_cache_attribute`` is what reaches
        the case. A stage that records in ``__call__`` alone streams a whole run and leaves the case
        the geometry it was stored with. Recording in both is what a reorientation does: the check
        is on recording in neither.
        """
        recorded = {key for key in scoped.keys() if key not in cache_attribute or scoped[key] != cache_attribute[key]}
        if not recorded:
            return
        if type(region_stage).write_stream_cache_attribute is not Transform.write_stream_cache_attribute:
            return
        raise PatchError(
            f"'{type(region_stage).__name__}' recorded {sorted(recorded)} on the scope a streamed region"
            " is handed, which is dropped, and implements no write_stream_cache_attribute().",
            "Record the case's answer in write_stream_cache_attribute(): it is given the whole volume's"
            " shape, where a patch's extent cannot say it.",
        )

    def unload(self) -> None:
        self.data.clear()
        self.augmented_data.clear()
        self.loaded = False
        self.augmentationLoaded = self.total_augmentations == 0

    def unload_augmentation(self) -> None:
        self.augmented_data.clear()
        self.augmentationLoaded = self.total_augmentations == 0

    def get_data(
        self,
        index: int,
        a: int,
        patch_transforms: list[Transform],
        is_input: bool,
        apply_augmentations: bool = True,
    ) -> torch.Tensor:
        if not self.loaded and self._stream_ready(a, apply_augmentations):
            data, _ = self._get_streamed_data(index, a, is_input, apply_augmentations)
        else:
            if not self.loaded:
                # A failed Save sweep lands here past the buffered-path guard (which saw a pending
                # plan and skipped the full load): load classically, which writes the caches too.
                self.load(self.transforms, self.data_augmentations_list, load_augmentations=False)
            data = self.patch.get_data(self._get_tensor(a), index, a, is_input)
        if patch_transforms:
            # Per-patch scope: writing to the shared case attribute would freeze the first patch's
            # derived statistic for every other patch. A case-level statistic (`Standardize(lazy=True)`
            # in `transforms`) is inherited through the copy.
            cache_attribute = Attribute(self.cache_attributes[a])
            for transform_function in patch_transforms:
                data = transform_function(self.name, data, cache_attribute)
        return data

    def get_size(self, a: int) -> int:
        return self.patch.get_size(a)
