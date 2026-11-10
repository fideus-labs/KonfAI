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


"""How overlapping patches are weighted back into one volume."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from functools import reduce

import numpy as np
import torch

from konfai.utils.errors import ConfigError
from konfai.utils.utils import (
    resolve_overlap,
)


def outer_product(vectors: Sequence[torch.Tensor]) -> torch.Tensor:
    """The outer product of 1-D ``vectors``, one axis each in their order."""
    return reduce(lambda acc, vector: acc.unsqueeze(-1) * vector, vectors)


class PathCombine(ABC):
    """Base class for overlap-aware weighting schemes applied during patch assembly."""

    #: A floor under the patch's whole weight, the product of its per-axis windows (``Gaussian``: 1e-3), applied on a
    #: grid placed ``monai`` only. The weight is then no longer a product of one vector per axis, and ``Accumulator``
    #: normalises it voxel by voxel.
    min_weight: float | None = None

    def __init__(self) -> None:
        self.windows_1d: list[torch.Tensor]
        self.overlap: int
        self.overlaps: list[int]
        self._data_per_device: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}

    def set_patch_config(self, patch_size: list[int], overlap: int | list[int]) -> None:
        self._data_per_device.clear()
        overlaps = [overlap] * len(patch_size) if isinstance(overlap, int) else list(overlap)
        self.overlap = max(overlaps)
        self.overlaps = overlaps
        # The per-patch weight is the outer product of one 1-D window per axis, each axis tapering over its
        # own overlap. The factors are kept: Accumulator normalises by a per-axis sum (_weight_geometry). Built
        # on a zero overlap too: a grid placed monai or spread overlaps where the stride does not, and MONAI's and
        # nnU-Net's Gaussian weighs that band as any other.
        self.windows_1d = [self._window_1d(size, o) for size, o in zip(patch_size, overlaps, strict=True)]

    @property
    def data(self) -> torch.Tensor:
        """The per-voxel window: the outer product of the per-axis factors.

        Derived, not stored: the object is pickled to every rank, and assembly goes through the factors
        themselves (``Accumulator._weight_geometry``). :meth:`weight` caches what it builds per device and dtype.
        """
        return outer_product(self.windows_1d)

    @property
    def selects(self) -> bool:
        """Whether the window picks ONE patch per voxel (values 0 or 1) rather than weighting several.

        A selection is a partition of the volume: assembly writes each region instead of accumulating
        a weighted sum (``Accumulator._blend``).
        """
        return False

    def weight(self, tensor: torch.Tensor) -> torch.Tensor:
        """The raw per-voxel window on ``tensor``'s device and dtype (cached per pair), unnormalised.

        Assembly does not go through it: Accumulator divides each window by the total over the patches
        covering a voxel, so what it multiplies by is always in [0, 1].
        """
        key = (tensor.device, tensor.dtype)
        if key not in self._data_per_device:
            # Match the tensor dtype: a float64 weight would upcast the whole blend.
            weight = self.data.to(device=tensor.device, dtype=tensor.dtype)
            if weight.is_floating_point():
                # A Gaussian tail underflows fp16 (a 96^3 patch corner is ~6e-11): floor at the smallest normal.
                weight = weight.clamp(min=torch.finfo(weight.dtype).tiny)
            self._data_per_device[key] = weight
        return self._data_per_device[key]

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.weight(tensor) * tensor

    @abstractmethod
    def _window_1d(self, size: int, overlap: int) -> torch.Tensor:
        """Return the 1-D blend weight along one axis (length ``size``, ``overlap`` voxels tapered per side)."""


def blend_axes(patch_size: list[int]) -> list[int]:
    """The blend window's axes for a ``patch_size``: every axis kept, a free (0) or singleton (1) axis
    as a single broadcast entry.

    ``Accumulator`` reads one window per spatial axis of the volume, so the list must not drop an axis.
    """
    return [size if size > 1 else 1 for size in patch_size]


def blend_overlap(overlap: "int | float | str | list[int | float | str]", patch_size: list[int]) -> list[int]:
    """Per-axis blend overlap for a concrete ``patch_size`` (int broadcast, ``%``/fraction resolved).

    Every axis whose patch is longer than one voxel is treated as tiled, so an int is the same voxel
    overlap on every such axis.
    """
    if isinstance(overlap, int):
        if overlap < 0:
            raise ConfigError(f"overlap: {overlap} must be >= 0 voxels.")
        for size in patch_size:
            if size > 1 and overlap >= size:
                raise ConfigError(f"overlap: {overlap} voxels must be smaller than the patch size {size}.")
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
        # sum to one. The +0.5 phase keeps the very edge > 0.
        ramp = torch.sin((torch.arange(overlap, dtype=torch.float32) + 0.5) / overlap * (torch.pi / 2)) ** 2
        window[:overlap] = ramp
        window[size - overlap :] = ramp.flip(0)
        return window


class Trim(PathCombine):
    """Selection instead of weighting: each voxel comes from the patch that holds it most centrally.

    The accumulator keeps each patch up to halfway between its centre and each neighbour's
    (``Accumulator._kept_span``), so the kept bands tile the axis exactly however the patches sit: an
    interior patch of an even grid keeps its central ``patch - overlap`` band (the window below), the
    first and last patch of an axis open to the volume edge. The values are 0 or 1, so the patch is
    selected rather than averaged: a discrete output (a label map, an argmax) survives reassembly.
    """

    @property
    def selects(self) -> bool:
        return True

    def _window_1d(self, size: int, overlap: int) -> torch.Tensor:
        if overlap >= size:
            # Nothing central left to keep: the patch stays whole and the axis falls back to last-write-wins.
            return torch.ones(size)
        window = torch.zeros(size)
        # Split an odd overlap so consecutive kept bands abut: k*stride + hi == (k+1)*stride + lo.
        window[overlap // 2 : size - (overlap - overlap // 2)] = 1.0
        return window


class Gaussian(PathCombine):
    """nnU-Net-style Gaussian importance weighting.

    Not a partition of unity: the accumulator blends each patch in with its share of the total weight. On a grid
    placed ``monai`` (``Patch.placement``), a patch's weight, the product of the per-axis Gaussians, is floored at
    ``1e-3``, as MONAI's ``compute_importance_map`` clamps its map: the corners of overlapping patches then weigh as
    much as one another.
    """

    min_weight = 1e-3

    def __init__(self, sigma_scale: float = 0.125) -> None:
        super().__init__()
        self.sigma_scale = sigma_scale

    def _window_1d(self, size: int, overlap: int) -> torch.Tensor:
        sigma = max(size * self.sigma_scale, 1e-6)
        center = (size - 1) / 2
        coords = torch.arange(size, dtype=torch.float32)
        return torch.exp(-((coords - center) ** 2) / (2.0 * sigma**2))


class GaussianFloat16(Gaussian):
    """A Gaussian window whose sums are kept in ``float16``: each patch times a window that is ``float16``, its sum
    and the sum of the windows kept in ``float16``, one divided by the other at the end. Near ties between classes
    are decided by that rounding: a logit near 64 moves by 0.06.

    The window is scipy's ``gaussian_filter`` of the impulse at ``size // 2`` (``sigma_scale`` of each extent, mode
    constant), scaled to ten at its centre, its zeros raised to its smallest other value. It is not a product of one
    vector per axis once rounded, so the accumulator takes it whole.
    """

    #: The accumulator keeps a sum and a sum of weights in float16 (``Accumulator._blend``) instead of shares.
    float16_sums = True
    value_scaling = 10.0

    def __init__(self, sigma_scale: float = 0.125) -> None:
        super().__init__(sigma_scale)
        self._weights: dict[tuple[int, ...], torch.Tensor] = {}

    def weights(self, shape: list[int]) -> torch.Tensor:
        """The window of a patch of ``shape``, ``float16``."""
        key = tuple(shape)
        if key not in self._weights:
            self._weights[key] = self._compute_weights(shape)
        return self._weights[key]

    def _compute_weights(self, shape: list[int]) -> torch.Tensor:
        # scipy's ``gaussian_filter`` of the centred impulse (mode constant): one product per axis and no sum, so
        # the outer product of scipy's truncated kernels, cut at the window's edges, is it to the bit.
        from konfai.data.transform.antialias import _kernel

        window = np.ones(())
        for size in shape:
            half = _kernel(size * self.sigma_scale).numpy()
            kernel, radius, centre = np.concatenate([half[::-1], half[1:]]), half.size - 1, size // 2
            axis = np.zeros(size)
            first, last = max(0, centre - radius), min(size, centre + radius + 1)
            axis[first:last] = kernel[first - centre + radius : last - centre + radius]
            window = np.multiply.outer(window, axis)
        rounded = torch.from_numpy(window / np.max(window) * self.value_scaling).to(torch.float16)
        rounded[rounded == 0] = rounded[rounded != 0].min()
        return rounded
