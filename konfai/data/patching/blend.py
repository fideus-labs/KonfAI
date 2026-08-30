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

import torch

from konfai.utils.errors import ConfigError
from konfai.utils.utils import (
    resolve_overlap,
)


class PathCombine(ABC):
    """Base class for overlap-aware weighting schemes applied during patch assembly."""

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
            return
        self.windows_1d = [
            self._window_1d(size, axis_overlap) if axis_overlap > 0 else torch.ones(size)
            for size, axis_overlap in zip(patch_size, overlaps, strict=True)
        ]

    @property
    def data(self) -> torch.Tensor:
        """The per-voxel window: the outer product of the per-axis factors.

        Derived, not stored: it is a patch of floats that ``Network.init`` builds before the spawn and
        every rank then unpickles (a 128^3 ModelPatch at overlap 16 pickles to 8 391 775 bytes with it,
        1355 without), while assembly goes through the factors themselves (see ``_weight_factors``).
        :meth:`weight` caches what it builds per device and dtype.
        """
        data = self.windows_1d[0]
        for window in self.windows_1d[1:]:
            data = data.unsqueeze(-1) * window
        return data

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


def blend_axes(patch_size: list[int]) -> list[int]:
    """The blend window's axes for a ``patch_size``: every axis kept, a free (0) or singleton (1) axis
    as a single broadcast entry.

    ``Accumulator`` reads one window per spatial axis of the volume, and the window has to broadcast
    against the patch at any axis position. Dropping the untiled axes shortens the list, so the
    window is looked up on the wrong axis (or not at all).
    """
    return [size if size > 1 else 1 for size in patch_size]


def blend_overlap(overlap: "int | float | str | list[int | float | str]", patch_size: list[int]) -> list[int]:
    """Per-axis blend overlap for a concrete ``patch_size`` (int broadcast, ``%``/fraction resolved).

    The blend has no volume extent: every axis whose patch is longer than one voxel is treated as tiled
    (the untiled-axis-0 rule is already applied by the slicing plan), so an int is the same voxel
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
    borders, which suppresses seam artefacts. Not a partition of unity: the accumulator blends each
    patch in with its share of the total weight, so overlapping patches still form a weighted average.
    """

    def __init__(self, sigma_scale: float = 0.125) -> None:
        super().__init__()
        self.sigma_scale = sigma_scale

    def _window_1d(self, size: int, overlap: int) -> torch.Tensor:
        sigma = max(size * self.sigma_scale, 1e-6)
        center = (size - 1) / 2
        coords = torch.arange(size, dtype=torch.float32)
        return torch.exp(-((coords - center) ** 2) / (2.0 * sigma**2))
