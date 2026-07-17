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

import copy
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from itertools import pairwise
from typing import Protocol, cast

import numpy as np
import torch
import torch.nn.functional as F

from konfai.data.augmentation import DataAugmentation, DataAugmentationsList
from konfai.data.transform import LocalityKind, PatchLocality, Resample, Save, Transform
from konfai.utils.config import apply_config, config
from konfai.utils.dataset import Attribute, Dataset
from konfai.utils.errors import PatchError
from konfai.utils.utils import (
    SUPPORTED_EXTENSIONS,
    OverlapSpec,
    get_module,
    get_patch_slices_from_shape,
    resolve_overlap,
    split_path_spec,
)

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
        target_slices: tuple[slice, ...],
        source_spatial_shape: list[int],
        cache_attribute: Attribute,
    ) -> list[slice]: ...

    def write_stream_cache_attribute(self, cache_attribute: Attribute, source_spatial_shape: list[int]) -> None: ...

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor: ...


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
        target_slices: tuple[slice, ...],
        source_spatial_shape: list[int],
        cache_attribute: Attribute,
    ) -> list[slice]:
        return self.augmentation.stream_region_source(self.index, self.a, target_slices, source_spatial_shape)

    def write_stream_cache_attribute(self, cache_attribute: Attribute, source_spatial_shape: list[int]) -> None:
        """An augmentation draws a copy of the case rather than restating its geometry: nothing to record."""

    def stream_shape(self, shape: list[int]) -> list[int]:
        """The spatial shape this copy's draw produces from ``shape`` (its slot in the shape fold)."""
        return self.augmentation.stream_shape(self.index, self.a, shape)

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        return self.augmentation.compute(name, self.index, self.a, tensor)


# The region kinds a composed streamed read (or write) carries between its pointwise stages.
_REGION_KINDS = (LocalityKind.HALO, LocalityKind.ORIENTATION, LocalityKind.CROP, LocalityKind.RESCALE)


def _halo_pull(radii: list[int], shape: list[int]) -> Callable[[tuple[slice, ...]], list[slice]]:
    """A halo stage's pull map: the region enlarged by the radius, clamped to the volume."""

    def pull(target: tuple[slice, ...]) -> list[slice]:
        return [
            slice(max(0, t.start - radius), min(extent, t.stop + radius))
            for t, radius, extent in zip(target, radii, shape, strict=False)
        ]

    return pull


def _remap_pull(
    remap: Callable[[tuple[slice, ...], list[int], Attribute], list[slice]],
    shape: list[int],
    attribute: Attribute,
) -> Callable[[tuple[slice, ...]], list[slice]]:
    """An index-remap stage's pull map, bound to the case state the stages before it left."""

    def pull(target: tuple[slice, ...]) -> list[slice]:
        return remap(target, list(shape), Attribute(attribute))

    return pull


def _scale_pull(scales: list[float], shape: list[int]) -> Callable[[tuple[slice, ...]], list[slice]]:
    """A rescale stage's pull map: the source window of the scale mapping plus its interpolation halo."""

    def pull(target: tuple[slice, ...]) -> list[slice]:
        return Resample.source_window(target, scales, shape)

    return pull


@dataclass(frozen=True)
class _ReadStagePlan:
    """One chain stage as the composed streamed read runs it: its declared kind, the spatial shapes
    on either side, and — for a region stage — the pull map from a region of its output to the region
    of its input it is computed from, bound to the case state the stages before it left."""

    kind: LocalityKind
    in_shape: tuple[int, ...]
    out_shape: tuple[int, ...]
    pull: Callable[[tuple[slice, ...]], list[slice]] | None


@dataclass(frozen=True)
class _PatchStreamSource:
    """What a copy's patches are read from, and what runs on them once read.

    ``stage_plans`` mirrors ``stages`` one to one: the region plans locate every stage that reads
    somewhere other than the target patch (they compose, each pulling through the one before it); the
    rest are pointwise, so they run wherever the regions put them.
    """

    dataset: Dataset
    group: str
    shape: list[int]
    stages: list[Stage]
    stage_plans: tuple[_ReadStagePlan, ...]

    @property
    def region_index(self) -> int | None:
        """The first region stage, or ``None`` for an exact-patch chain."""
        for index, plan in enumerate(self.stage_plans):
            if plan.kind in _REGION_KINDS:
                return index
        return None


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
        self.overlap: int
        self._data_per_device: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}

    def set_patch_config(self, patch_size: list[int], overlap: int | list[int]) -> None:
        self._data_per_device.clear()
        overlaps = [overlap] * len(patch_size) if isinstance(overlap, int) else list(overlap)
        self.overlap = max(overlaps)
        if all(o <= 0 for o in overlaps):
            self.data = torch.ones(patch_size)
            return
        # The per-patch weight is the outer product of one 1-D window per axis. It is separable by
        # construction, so a per-axis partition of unity stays a partition of unity in N-D and overlapping
        # patches blend without darkening — no distance map or explicit renormalisation loop is needed, and
        # the trailing ``result / weight_sum`` in Accumulator.assemble stays exact at the volume borders.
        # Each axis tapers over its own overlap, so anisotropic overlaps blend correctly per axis.
        data = self._window_1d(patch_size[0], overlaps[0]) if overlaps[0] > 0 else torch.ones(patch_size[0])
        for size, axis_overlap in zip(patch_size[1:], overlaps[1:], strict=True):
            window = self._window_1d(size, axis_overlap) if axis_overlap > 0 else torch.ones(size)
            data = data.unsqueeze(-1) * window
        self.data = data

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        key = (tensor.device, tensor.dtype)
        if key not in self._data_per_device:
            # Match the tensor dtype: the weight has no reason to carry more precision than the data it
            # scales, and a float64 weight would upcast the whole (channels x volume) blend.
            weight = self.data.to(device=tensor.device, dtype=tensor.dtype)
            if weight.is_floating_point():
                # A Gaussian tail can underflow the target dtype (a 96^3 patch corner is ~6e-11 — zero in
                # fp16), and a voxel covered only by such a corner then divides 0 by 0 at assembly. Floor
                # the weight at the dtype's smallest normal so single-coverage voxels stay recoverable.
                weight = weight.clamp(min=torch.finfo(weight.dtype).tiny)
            self._data_per_device[key] = weight
        return self._data_per_device[key] * tensor

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
        # border (where result/weight_sum must recover the raw value) never divides by zero.
        ramp = torch.sin((torch.arange(overlap, dtype=torch.float32) + 0.5) / overlap * (torch.pi / 2)) ** 2
        window[:overlap] = ramp
        window[size - overlap :] = ramp.flip(0)
        return window


class Gaussian(PathCombine):
    """nnU-Net-style Gaussian importance weighting.

    Favours patch centres — where the model sees the most surrounding context — and down-weights the
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
    ) -> None:
        self.patch_slices: list[tuple[slice, ...]] = []

        if patch_size is not None and not all(p == 0 for p in patch_size):
            for patch in patch_slices:
                slices: list[slice] = []
                for s, shape in zip(patch, patch_size, strict=False):
                    slices.append(slice(s.start, s.start + shape))
                self.patch_slices.append(tuple(slices))
        else:
            self.patch_slices = patch_slices

        self.shape = max([[v.stop for v in patch] for patch in patch_slices])
        self.patch_size = patch_size
        self.patch_combine = patch_combine
        self.batch = batch
        self._n = 2 if batch else 1
        self._count = len(patch_slices)
        self._filled = 0
        self._done = [False] * self._count
        # Patches are blended into these running buffers as they arrive (see add_layer), instead of
        # being kept until assembly: holding every patch of a large multi-class case (e.g. ~70 patches
        # of a 122-channel whole-body segmentation ≈ tens of GB) was the dominant reassembly RAM cost.
        self._result: torch.Tensor | None = None
        self._weight_sum: torch.Tensor | None = None
        self._weight_patch: torch.Tensor | None = None

    def add_layer(self, index: int, layer: torch.Tensor) -> list[tuple[slice, torch.Tensor]] | None:
        # Blend each patch straight into the running accumulator and drop the patch, rather than
        # storing all patches for a single assemble() at the end. The overlap blend is a weighted sum,
        # so accumulating incrementally is equivalent; re-adding an index is a no-op (last-write wins is
        # not possible once blended, and the prediction pipeline adds each patch exactly once).
        if self._done[index]:
            return None
        n = self._n
        if self._result is None:
            # Allocate to the ACTUAL volume extent (self.shape), not the patch-size-extended grid. The
            # last patch of each axis is padded up to patch_size for the model, but that padded tail lies
            # OUTSIDE the volume; blending it would over-allocate the accumulator by up to
            # (patch_size - overlap) per axis (then get cropped away). We crop each patch to its in-volume
            # part at blend time instead, so nothing outside the volume is ever allocated.
            self._result = torch.zeros(list(layer.shape[:n]) + list(self.shape), dtype=layer.dtype, device=layer.device)
            if self.patch_combine is not None:
                # Match the result dtype so the final ``result / weight_sum`` does not promote the whole
                # (channels x volume) accumulator to float32.
                self._weight_sum = torch.zeros(list(self.shape), dtype=layer.dtype, device=layer.device)
        patch_slice = self.patch_slices[index]
        data = layer
        for dim, s in enumerate(patch_slice):
            if s.stop - s.start == 1:
                data = data.unsqueeze(dim=dim + n)
        # Clamp each spatial destination to the volume, and crop the patch (and its weight window) to the
        # matching in-volume extent so the padded tail of border patches is discarded, not stored.
        dest = [slice(s.start, min(s.stop, self.shape[dim])) for dim, s in enumerate(patch_slice)]
        crop = tuple([slice(None)] * n + [slice(0, d.stop - d.start) for d in dest])
        slices_dest = tuple([slice(self._result.shape[i]) for i in range(n)] + dest)
        # Overlap blending weights each patch (edge bands < 1 so interior overlaps sum to unity).
        # A voxel covered by fewer patches (a volume border without whole-image padding) would sum
        # to < 1 and come out darkened (x0.5 edges, x0.25 corners), so divide by the accumulated weight.
        if self.patch_combine is not None and self._weight_sum is not None:
            self._result[slices_dest] += self.patch_combine(data)[crop]
            if self._weight_patch is None:
                # Spatial-only ones: the weight is per-voxel, and deriving it from a channel-sized
                # ones_like would allocate (and weight) C copies just to index one back out.
                self._weight_patch = self.patch_combine(
                    torch.ones(data.shape[n:], dtype=data.dtype, device=data.device)
                )
            self._weight_sum[tuple(dest)] += self._weight_patch[crop[n:]]
        else:
            self._result[slices_dest] = data[crop]
        self._done[index] = True
        self._filled += 1
        return None

    def is_empty(self) -> bool:
        """True until the first patch has been blended in (no volume-sized buffer allocated yet)."""
        return self._result is None

    @property
    def footprint_shape(self) -> list[int]:
        """Spatial shape held in memory at once — the whole volume for the base accumulator (overridden
        by StreamingAccumulator, which keeps only a window). Used to size the blend device."""
        return self.shape

    def is_full(self) -> bool:
        # O(1): a running counter avoids re-scanning every slot after each added patch
        # (the completion check ran once per patch, i.e. O(P^2) per case).
        return self._filled == self._count

    def assemble(self) -> torch.Tensor:
        if self._result is None:
            raise PatchError(
                "Accumulator.assemble() was called before any patch was added.",
                f"Expected up to {self._count} patch(es) via add_layer() before assembling.",
                "Add at least one patch (and check is_full()) before calling assemble().",
            )
        result = self._result
        # With padding weight_sum is ~1 and the division is a near no-op; the clamp guards borders
        # that no patch (fully) covered. In-place so we do not materialise a second
        # (channels x volume) tensor; weight_sum already matches result.dtype (fp16), so the division
        # stays in fp16 with no float32 promotion (same values as the out-of-place form).
        # The floor must be representable in the accumulation dtype: 1e-8 rounds to zero in fp16
        # (divide-by-zero -> NaN), so clamp at the dtype's smallest normal instead.
        if self.patch_combine is not None and self._weight_sum is not None:
            result.div_(self._weight_sum.clamp(min=torch.finfo(self._weight_sum.dtype).tiny))
        # No final crop: patches are cropped to the volume at blend time, so result is already self.shape.

        self._result = None
        self._weight_sum = None
        self._weight_patch = None
        self._filled = 0
        self._done = [False] * self._count
        return result


class StreamingAccumulator(Accumulator):
    """Accumulator holding only the active window along the first spatial axis; it yields each finalized
    slab as its patches complete, so peak memory is one patch extent.

    The patch-grid order (``get_patch_slices_from_shape`` iterates ``itertools.product`` with the first
    spatial axis outermost) has patch starts along that axis never decreasing, so when a patch starting
    at ``z`` arrives, every voxel before ``z`` has already received all of its patches and the region up
    to ``z`` is final. ``add_layer`` returns those finalized slabs — ``assemble()``'s values, from the
    same blend and weight arithmetic applied slab by slab — and ``finalize()`` flushes the tail.
    """

    def __init__(
        self,
        patch_slices: list[tuple[slice, ...]],
        patch_size: list[int],
        patch_combine: PathCombine | None = None,
        batch: bool = True,
    ) -> None:
        super().__init__(patch_slices, patch_size, patch_combine, batch)
        self._window = min(max(patch[0].stop - patch[0].start for patch in self.patch_slices), self.shape[0])
        starts = [patch[0].start for patch in self.patch_slices]
        if any(0 > current - previous or current - previous > self._window for previous, current in pairwise(starts)):
            raise PatchError(
                "StreamingAccumulator requires patches ordered by non-decreasing start on the first spatial axis,"
                " advancing by at most one patch extent per step.",
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
        if patch_slice[0].start < self._flushed:
            raise PatchError(
                f"StreamingAccumulator received patch start {patch_slice[0].start} after flushing to "
                f"{self._flushed}: patches must arrive in non-decreasing first-spatial-axis order.",
                "Add patches in the order get_patch_slices_from_shape generates them (no shuffling).",
            )
        slabs = self._advance_to(patch_slice[0].start)
        if self._result is None:
            self._result = torch.zeros(
                [*layer.shape[:n], self._window, *self.shape[1:]],
                dtype=layer.dtype,
                device=layer.device,
            )
            if self.patch_combine is not None:
                self._weight_sum = torch.zeros(self._result.shape[n:], dtype=layer.dtype, device=layer.device)
        data = layer
        for dim, s in enumerate(patch_slice):
            if s.stop - s.start == 1:
                data = data.unsqueeze(dim=dim + n)
        # Same blend as the parent (clamped destination, cropped patch, weighted overlap); the only
        # difference is that the destination's first spatial index is relative to the window origin.
        dest = [slice(s.start, min(s.stop, self.shape[dim])) for dim, s in enumerate(patch_slice)]
        crop = tuple([slice(None)] * n + [slice(0, d.stop - d.start) for d in dest])
        dest[0] = slice(dest[0].start - self._flushed, dest[0].stop - self._flushed)
        slices_dest = tuple([slice(self._result.shape[i]) for i in range(n)] + dest)
        if self.patch_combine is not None and self._weight_sum is not None:
            self._result[slices_dest] += self.patch_combine(data)[crop]
            if self._weight_patch is None:
                # Spatial-only ones: the weight is per-voxel, and deriving it from a channel-sized
                # ones_like would allocate (and weight) C copies just to index one back out.
                self._weight_patch = self.patch_combine(
                    torch.ones(data.shape[n:], dtype=data.dtype, device=data.device)
                )
            self._weight_sum[tuple(dest)] += self._weight_patch[crop[n:]]
        else:
            self._result[slices_dest] = data[crop]
        self._done[index] = True
        self._filled += 1
        return slabs

    def finalize(self) -> list[tuple[slice, torch.Tensor]]:
        """Flush the remaining window and reset for reuse; call once ``is_full()``."""
        slabs = self._advance_to(self.shape[0])
        self._result = None
        self._weight_sum = None
        self._weight_patch = None
        self._filled = 0
        self._done = [False] * self._count
        self._flushed = 0
        return slabs

    @property
    def footprint_shape(self) -> list[int]:
        # Only the window is resident, so the blend-device budget is the window's: a huge volume streams
        # on the GPU within bounded VRAM. Blend and IEEE-correctly-rounded finalize ops (+, *, /, argmax,
        # cast) are bit-identical CPU/CUDA; only a transcendental-terminated float output (Softmax/Sigmoid)
        # can differ by ~1 ULP between a window on the GPU and a whole-volume reference on the CPU.
        return [self._window, *self.shape[1:]]

    def assemble(self) -> torch.Tensor:
        raise PatchError(
            "StreamingAccumulator does not assemble a whole volume.",
            "Consume the slabs returned by add_layer() and finalize() instead.",
        )

    def _advance_to(self, z: int) -> list[tuple[slice, torch.Tensor]]:
        """Finalize the window up to ``z`` (absolute) and shift the window origin there."""
        z = min(z, self.shape[0])
        if self._result is None or z <= self._flushed:
            return []
        n = self._n
        length = z - self._flushed
        lead = (slice(None),) * n
        slab = self._result[(*lead, slice(0, length))]
        if self.patch_combine is not None and self._weight_sum is not None:
            # Same division and fp16 floor as Accumulator.assemble, applied to the finalized slab.
            slab = slab / self._weight_sum[:length].clamp(min=torch.finfo(self._weight_sum.dtype).tiny)
        else:
            slab = slab.clone()
        keep = self._window - length
        # .clone(): source and destination views overlap when length < window.
        self._result[(*lead, slice(0, keep))] = self._result[(*lead, slice(length, self._window))].clone()
        self._result[(*lead, slice(keep, self._window))] = 0
        if self._weight_sum is not None:
            self._weight_sum[:keep] = self._weight_sum[length : self._window].clone()
            self._weight_sum[keep:] = 0
        region = slice(self._flushed, z)
        self._flushed = z
        return [(region, slab)]


class SlabRegionStream:
    """Slab in → slab out through one region stage, with bounded lookahead.

    The write mirror of the read dispatcher's single-region rule: finalized slabs arrive in order along
    the first spatial axis (the :class:`StreamingAccumulator`'s order), and each output region is
    emitted as soon as the input region it pulls has arrived — so only a sliding window of the input is
    ever resident. The stage itself is two injected callables, both pure region arithmetic + tensor
    work, so no stage kind has streaming code of its own:

    - ``pull(target_slices) -> source_slices`` — the clamped input region an output region is computed
      from (a transform's ``stream_region_target``/``stream_region_source``, or a halo enlargement).
    - ``produce(window, target_slices, source_slices) -> tensor`` — the output block for
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
        pick can never corrupt the output — emission is gated on the pull of the real regions — it only
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
        # a binary search — O(log rows) pull calls per push, and a pull may be more than slice
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
        """The buffered input restricted to ``source`` — rows gathered from the arrived slabs, the
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
    that needs every stream's rows together — a cross-copy reduction — can only advance to the
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
        self._nb_patch_per_dim: dict[int, list[tuple[int, bool]]] = {}
        self.pad_value = pad_value
        self.extend_slice = extend_slice
        # Models need every patch at the declared size, so the last patch of an axis is padded up to it.
        # A consumer that REDUCES patches instead (streamed evaluation) must see only in-volume voxels:
        # padded ones would pollute its running sums, so it turns this off and takes the cropped patch.
        self.pad_to_patch = True

    def load(self, shape: list[int], a: int = 0) -> None:
        self._patch_slices[a], self._nb_patch_per_dim[a] = get_patch_slices_from_shape(
            self.patch_size, shape, self.overlap
        )

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
        if self.pad_to_patch and self.patch_size is not None and not all(p == 0 for p in self.patch_size):
            for dim_it, _slice in enumerate(reversed(slices)):
                p = (
                    0
                    if _slice.start + self.patch_size[-dim_it - 1] <= data_shape[-dim_it - 1]
                    else self.patch_size[-dim_it - 1] - (data_shape[-dim_it - 1] - _slice.start)
                )
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
                kept = [i for i in self.patch_size if i > 1]
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

        for transform_function in transforms:
            _shape = transform_function.transform_shape(self.group_src, self.name, _shape, cache_attribute)

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
        self.patch.load(_shape, 0)
        # The spatial grid each copy's patches are cut on: the un-augmented copy's is the source shape
        # folded by the transforms, and a copy whose draw changes shape (Permute, Mask) has its own.
        self.shapes: list[list[int]] = [_shape]
        self.data_augmentations_list = data_augmentations_list
        self._patch_stream_sources: dict[tuple[int, bool], _PatchStreamSource | None] = {}
        self._stream_attributes_persisted: set[int] = set()
        self._disk_statistics: dict[tuple[Dataset, str, tuple[int, ...] | None], dict[str, float]] = {}
        # reset_state=False: the first manager built for a case draws (state_init draws a missing
        # index), and every later group's manager reuses that draw -- redrawing here would give each
        # group its own geometry and desynchronise the per-copy patch grids across groups.
        self.reset_augmentation(reset_state=False)
        self.cache_attributes_bak = copy.deepcopy(self.cache_attributes)

    def reset_augmentation(self, reset_state: bool = True):
        self.cache_attributes[:] = self.cache_attributes[:1]
        self.shapes[:] = self.shapes[:1]
        self.augmented_data.clear()
        # An augmented copy's stream source is only as good as the draw it was planned for: a halo is
        # the draw's own, and a re-draw is a new one. Drop every plan, so the next request replans
        # against the draw the copies actually carry.
        self._patch_stream_sources.clear()
        self._stream_attributes_persisted.clear()
        self.total_augmentations = 0
        i = 1
        for data_augmentations in self.data_augmentations_list:
            shape = []
            caches_attribute = []
            for _ in range(data_augmentations.nb):
                shape.append(self.shapes[0])
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
        self.augmentationLoaded = self.total_augmentations == 0

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
                if transform_function.dataset:
                    if len(transform_function.dataset.split(":")) > 1:
                        filename, file_format = transform_function.dataset.split(":")
                    else:
                        filename = transform_function.dataset
                        file_format = "mha"
                    dataset = Dataset(filename, file_format)
                else:
                    dataset = self.dataset
                group_dest = transform_function.group if transform_function.group else self.group_dest
                if dataset.is_dataset_exist(group_dest, self.name):
                    data, attrib = dataset.read_data(group_dest, self.name)
                    self.cache_attributes[0].update(attrib)
                    break
            i -= 1

        if i == 0:
            data, _ = self.dataset.read_data(self.group_src, self.name)

        data = torch.from_numpy(data)

        if len(pre_transform):
            for transform_function in pre_transform[i:]:
                data = transform_function(self.name, data, self.cache_attributes[0])
                if isinstance(transform_function, Save):
                    if transform_function.dataset:
                        if len(transform_function.dataset.split(":")) > 1:
                            filename, file_format = transform_function.dataset.split(":")
                        else:
                            filename = transform_function.dataset
                            file_format = "mha"
                        dataset = Dataset(filename, file_format)
                    else:
                        dataset = self.dataset
                    group_dest = transform_function.group if transform_function.group else self.group_dest
                    dataset.write(
                        group_dest,
                        self.name,
                        data.numpy(),
                        self.cache_attributes[0],
                    )
        self.data.append(data)

        for i in range(len(self.cache_attributes) - 1):
            self.cache_attributes[i + 1].update(self.cache_attributes[0])
        self.loaded = True

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

    def _get_tensor(self, a: int) -> torch.Tensor:
        if a == 0:
            return self.data[0]
        if a not in self.augmented_data:
            self._load_augmentation_group(*self._augmentation_group(a))
        return self.augmented_data[a]

    def _read_disk_statistics(
        self,
        source_dataset: Dataset,
        source_group: str,
        channels: list[int] | None,
    ) -> dict[str, float]:
        """Read (and memoise) the whole-volume statistics of one on-disk group for this case.

        ``read_data_statistics`` scans the stored volume without materialising it, but it is still a
        full pass: memoise it per (dataset, group, channels) so a per-patch consumer -- whose
        ``inverse()`` pops the seeded keys back out of the cache attribute at prediction time -- does
        not re-scan the volume once per patch.
        """
        key = (source_dataset, source_group, tuple(channels) if channels is not None else None)
        if key not in self._disk_statistics:
            self._disk_statistics[key] = source_dataset.read_data_statistics(source_group, self.name, channels)
        return self._disk_statistics[key]

    def _ensure_stream_stats(
        self,
        source_dataset: Dataset,
        source_group: str,
        cache_attribute: Attribute,
        required_stats: set[str],
        channels: list[int] | None = None,
    ) -> bool:
        missing_stats = [key for key in required_stats if key not in cache_attribute]
        if not missing_stats:
            return True

        stats = self._read_disk_statistics(source_dataset, source_group, channels)
        stats_mapping = {
            "Min": stats["min"],
            "Max": stats["max"],
            "Mean": stats["mean"],
            "Std": stats["std"],
        }
        for key in missing_stats:
            if key in {"Mean", "Std"}:
                cache_attribute[key] = np.asarray([stats_mapping[key]], dtype=np.float32)
            else:
                cache_attribute[key] = stats_mapping[key]
        return all(key in cache_attribute for key in required_stats)

    def _affords_halo(self, a: int, halo: tuple[int, ...]) -> bool:
        """Whether a halo of this radius still buys copy *a* anything over loading the volume.

        Every patch pays the halo on every side and the patches tile the volume, so streaming a case
        reads ``prod(1 + 2 * halo_k / patch_k)`` times its bytes -- the multiple streaming pays to keep
        one volume off the heap. Half a patch doubles every axis: 8x the reads in 3D. Past that the
        multiple runs away -- a halo of one whole patch is 27x -- while the saving is still just the
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
        cache_attribute: Attribute,
        source_spatial_shape: list[int],
    ) -> tuple[bool, tuple[_ReadStagePlan, ...]]:
        """Validate a copy's locality declarations and plan its region stages, which compose.

        Returns ``(streamable, stage_plans)``. The chain streams when every stage is pointwise, a
        region kind (``HALO``/``ORIENTATION``/``CROP``/``RESCALE`` — any number, each pulling through
        the one before it), or a ``GLOBAL_STAT`` with a pre-populated statistic. The plan walks the
        chain once with one evolving case state, so each stage declares against — and remaps from —
        the geometry the stages before it left, and folds the spatial shapes stage by stage; a fold
        that does not land on the copy's own grid refuses (the safety net for a stage whose shape map
        is not declared). Any ``WHOLE_VOLUME`` declaration, an unreadable ``GLOBAL_STAT``, a
        ``RESCALE`` without a known ``Spacing`` (or that is not a :class:`Resample`), or a halo too
        wide to be worth reading rejects streaming. Nothing here names a stage: each declares its own
        contract, and this is where the declarations are read -- which is why a transform and an
        augmentation are planned side by side.
        """
        evolved = Attribute(cache_attribute)
        shape = [int(extent) for extent in source_spatial_shape]
        localities: list[PatchLocality] = []
        plans: list[_ReadStagePlan] = []
        for stage in stages:
            loc = stage.patch_locality(Attribute(evolved))
            localities.append(loc)
            if loc.kind is LocalityKind.WHOLE_VOLUME:
                return False, ()
            if loc.kind is LocalityKind.GLOBAL_STAT:
                # The seed is the STORED volume's statistic, which is this transform's input only when
                # every earlier stage preserves it; otherwise ([Clip(-200, 400), Standardize()]) every
                # patch would be standardized by the pre-Clip statistic -- fall back to the whole volume.
                if not all(previous.statistics_preserving for previous in localities[:-1]):
                    return False, ()
                if not self._ensure_stream_stats(
                    source_dataset, source_group, cache_attribute, set(loc.stat_keys), loc.stat_channels
                ):
                    return False, ()
            if loc.kind is LocalityKind.HALO and not self._affords_halo(a, loc.halo):
                return False, ()
            if loc.kind is LocalityKind.RESCALE and (not isinstance(stage, Resample) or "Spacing" not in evolved):
                # A resample is patch-native only when the source geometry is known: the scale is read
                # from the evolving 'Spacing' (a free geometry stat, no read_data_statistics).
                return False, ()
            plan = self._plan_read_stage(stage, loc, shape, evolved)
            if plan is None:
                return False, ()
            plans.append(plan)
            shape = list(plan.out_shape)
        if shape != [int(extent) for extent in self.shapes[a]]:
            return False, ()
        return True, tuple(plans)

    def _plan_read_stage(
        self, stage: Stage, loc: PatchLocality, shape: list[int], evolved: Attribute
    ) -> "_ReadStagePlan | None":
        """One stage's slot in the composed plan: its shapes, its pull map, and — for a region stage —
        the geometry it leaves for the stages after it (``write_stream_cache_attribute``)."""
        if loc.kind not in _REGION_KINDS:
            return _ReadStagePlan(loc.kind, tuple(shape), tuple(shape), None)
        if loc.kind is LocalityKind.HALO:
            return _ReadStagePlan(
                loc.kind, tuple(shape), tuple(shape), _halo_pull(_halo_radii(loc.halo, len(shape)), list(shape))
            )
        if loc.kind is LocalityKind.RESCALE:
            resample = cast(Resample, stage)
            out = [int(e) for e in resample.transform_shape(self.group_src, self.name, list(shape), Attribute(evolved))]
            scales = [shape[k] / out[k] for k in range(len(shape))]
            resample.write_stream_cache_attribute(evolved, list(shape))
            return _ReadStagePlan(loc.kind, tuple(shape), tuple(out), _scale_pull(scales, list(shape)))
        # ORIENTATION / CROP: the stage's own remap, evaluated on the state the stages before it left.
        pull = _remap_pull(stage.stream_region_source, list(shape), Attribute(evolved))
        if isinstance(stage, Transform):
            out = [int(e) for e in stage.transform_shape(self.group_src, self.name, list(shape), Attribute(evolved))]
        else:
            out = [int(e) for e in cast(AugmentedStage, stage).stream_shape(list(shape))]
        stage.write_stream_cache_attribute(evolved, list(shape))
        return _ReadStagePlan(loc.kind, tuple(shape), tuple(out), pull)

    @staticmethod
    def _dataset_from_spec(dataset_spec: str) -> Dataset:
        filename, _, file_format = split_path_spec(
            dataset_spec,
            default_format="mha",
            supported_extensions=SUPPORTED_EXTENSIONS,
        )
        return Dataset(filename, file_format)

    def _resolve_patch_stream_source(self, a: int, apply_augmentations: bool = True) -> _PatchStreamSource | None:
        key = (a, apply_augmentations)
        if key in self._patch_stream_sources:
            return self._patch_stream_sources[key]

        source_dataset = self.dataset
        source_group = self.group_src
        source_shape = self.base_shape
        boundary_attributes: Attribute | None = None
        trailing_transforms: list[Stage] = list(self.transforms)

        for index in range(len(self.transforms) - 1, -1, -1):
            transform = self.transforms[index]
            if isinstance(transform, Save):
                dataset = self._dataset_from_spec(transform.dataset) if transform.dataset else self.dataset
                group = transform.group if transform.group else self.group_dest
                if dataset.is_dataset_exist(group, self.name):
                    source_dataset = dataset
                    source_group = group
                    source_shape, boundary_attributes = dataset.get_infos(group, self.name)
                    trailing_transforms = list(self.transforms[index + 1 :])
                    break

        # What copy `a` is: the trailing transforms, then its own draw. The draw is applied to the
        # transformed volume, so it goes last, and the whole thing is planned as one chain -- a region
        # transform and a region augmentation are then two regions, which is exactly what they are.
        stages = trailing_transforms + (self._augmentation_stages(a) if apply_augmentations else [])

        # Plan from the case as STORED (the pristine backup), never from the live attribute: the live
        # one carries what earlier patches or epochs wrote (a Resample's target Spacing, a Canonical's
        # canonical Direction), and planning from it would hand a stage its own output as the
        # description of its input on every epoch after the first.
        stream_cache_attribute = Attribute(self.cache_attributes_bak[0])
        if boundary_attributes is not None:
            # Streaming from a Save cache: the stored volume is the cache, so the stages after the
            # boundary read its geometry -- stacked over the source keys exactly as the whole-volume
            # cache-hit merges the cached header.
            for attribute_key, attribute_value in boundary_attributes.items():
                stream_cache_attribute[attribute_key] = attribute_value
        streamable, stage_plans = self._plan_stream_region(
            a, stages, source_dataset, source_group, stream_cache_attribute, list(source_shape[1:])
        )
        if streamable:
            self.cache_attributes[a] = Attribute(stream_cache_attribute)
            self.cache_attributes_bak[a] = Attribute(stream_cache_attribute)
            self._patch_stream_sources[key] = _PatchStreamSource(
                source_dataset, source_group, list(source_shape), stages, stage_plans
            )
        else:
            self._patch_stream_sources[key] = None
        return self._patch_stream_sources[key]

    def can_stream_patch(self, a: int, apply_augmentations: bool = True) -> bool:
        return self._resolve_patch_stream_source(a, apply_augmentations) is not None

    def _get_streamed_data(
        self,
        index: int,
        a: int,
        is_input: bool,
        apply_augmentations: bool = True,
    ) -> tuple[torch.Tensor, Attribute]:
        stream_source = self._resolve_patch_stream_source(a, apply_augmentations)
        if stream_source is None:
            raise RuntimeError("Patch streaming requested on a dataset manager without a streaming source.")

        if stream_source.region_index is None:
            # POINTWISE / GLOBAL_STAT only: read the exact patch and run the whole chain on it.
            plan = self.patch.get_read_plan(stream_source.shape, index, a, is_input)
            data, attributes = stream_source.dataset.read_data_slice(stream_source.group, self.name, plan.data_slices)
            tensor = torch.from_numpy(data)
            cache_attribute = Attribute(self.cache_attributes_bak[a])
            cache_attribute.update(attributes)
            persist = a not in self._stream_attributes_persisted
            keys_before = set(cache_attribute.keys()) if persist else set()
            for stage in stream_source.stages:
                tensor = stage(self.name, tensor, cache_attribute)
            # The read plan is applied AFTER the chain, as the whole-volume path transforms before
            # Patch.get_data cuts: padding first would feed f(pad) to the model on every border patch.
            tensor = self.patch.apply_read_plan(tensor, plan)
            if persist:
                self._persist_stream_attributes(a, cache_attribute, keys_before)
            return tensor, cache_attribute

        return self._get_streamed_region_data(index, a, stream_source, is_input)

    def _finalize_stream_patch(self, tensor: torch.Tensor, index: int, a: int, is_input: bool) -> torch.Tensor:
        """Pad/format a target-extent streamed patch to ``patch_size`` like the whole-volume path.

        The region and RESCALE streamed paths produce a patch at the raw target-slice extent, but the
        overlap tiling can leave the last patch narrower than ``patch_size`` (integer-floor stride).
        The whole-volume ``Patch.get_data`` pads that border patch up to ``patch_size`` via
        ``apply_read_plan``; running the streamed patch through the SAME read plan makes border patches
        byte-identical between the two paths instead of one voxel short. The plan is built on
        ``self.shapes[a]`` -- the spatial grid this copy's patch slices were themselves cut on, which is
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
        """Patch-native region chain: read only the source region a target patch pulls, composed.

        The region stages' pull maps fold backward from the target patch down to the stored volume —
        each stage's region pulls through the one before it — so one bounded read serves any number of
        them; the chain then runs forward over the sub-region, each stage on the region pair the same
        fold computed for it. HALO reads the region enlarged by the declared radius (clamped to the
        volume) and crops it back after the stage — the stage's own edge padding reproduces the
        whole-volume border once the clamp reaches the true border, so seams agree. ORIENTATION
        applies the index remap to what its region read; a CROP's remap IS its action, so the stage is
        not re-applied; RESCALE interpolates the sub-region to its target extent. Only the composed
        region is requested; whether that avoids decoding the whole volume depends on the storage
        format -- compressed MetaImage and NRRD decode the full volume per read
        (see ``_supports_region_read``).
        """
        stages, plans = stream_source.stages, stream_source.stage_plans
        target_slices = tuple(self.patch.get_patch_slices(a)[index])
        n_prefix = len(stream_source.shape) - len(target_slices)

        spans: list[list[slice]] = [list(target_slices)]
        for plan in reversed(plans):
            spans.append(plan.pull(tuple(spans[-1])) if plan.pull is not None else list(spans[-1]))
        spans.reverse()

        data_slices = tuple([slice(None)] * n_prefix + spans[0])
        data, attributes = stream_source.dataset.read_data_slice(stream_source.group, self.name, data_slices)
        tensor = torch.from_numpy(data)

        # Each patch re-runs the chain from the state the whole-volume pass started from: the case as
        # stored (plus planned stats), never the live attribute -- that one carries the chain's own
        # output.
        cache_attribute = Attribute(self.cache_attributes_bak[a])
        cache_attribute.update(attributes)
        persist = a not in self._stream_attributes_persisted
        keys_before = set(cache_attribute.keys()) if persist else set()

        for stage, plan, source, target in zip(stages, plans, spans[:-1], spans[1:], strict=True):
            if plan.kind not in _REGION_KINDS:
                tensor = stage(self.name, tensor, cache_attribute)
                continue
            # A region stage's geometry writes describe the patch's extent, not the volume's: give it
            # a throwaway scope, and write the case-level answer once from the FULL shape below
            # (write_stream_cache_attribute).
            scoped = Attribute(cache_attribute)
            if plan.kind is LocalityKind.RESCALE:
                tensor = cast(Resample, stage).resample_region(
                    tensor,
                    tuple(target),
                    [s.start for s in source],
                    [plan.in_shape[k] / plan.out_shape[k] for k in range(len(plan.in_shape))],
                    list(plan.in_shape),
                )
            elif plan.kind is not LocalityKind.CROP:
                tensor = stage(self.name, tensor, scoped)
                if plan.kind is LocalityKind.HALO:
                    lead = tensor.dim() - len(target)
                    crop = [slice(t.start - s.start, t.stop - s.start) for t, s in zip(target, source, strict=False)]
                    tensor = tensor[(*[slice(None)] * lead, *crop)]
            if persist:
                # The stage's own geometry writes went to the patch-scoped copy above, so this is
                # where the case gets them: computed once, from the FULL shape of the stage's input,
                # against the attribute that still describes the whole volume.
                stage.write_stream_cache_attribute(self.cache_attributes[a], list(plan.in_shape))
                self._check_region_geometry_reaches_the_case(stage, scoped, cache_attribute)

        tensor = self._finalize_stream_patch(tensor, index, a, is_input)

        if persist:
            self._persist_stream_attributes(a, cache_attribute, keys_before)
        return tensor, cache_attribute

    def _check_region_geometry_reaches_the_case(
        self, region_stage: Stage, scoped: Attribute, cache_attribute: Attribute
    ) -> None:
        """Refuse a region stage that records geometry nowhere the case can read it.

        A region stage is handed a patch, so what it records about the extent is one patch's answer:
        the scope it records into is thrown away, and ``write_stream_cache_attribute`` is what reaches
        the case. A stage that records in ``__call__`` alone streams a whole run and leaves the case
        the geometry it was stored with. Recording in both is what a reorientation does -- the check
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
        if not self.loaded and self.can_stream_patch(a, apply_augmentations):
            data, _ = self._get_streamed_data(index, a, is_input, apply_augmentations)
        else:
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
