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

"""Intensity draws for medical images: noise, blur, resolution, gamma, contrast.

The colour draws next door describe a photograph through a 3x4 colour matrix and take 1 or 3
channels. These describe an acquisition: the noise a scanner adds, the point spread it has, the
resolution it was reconstructed at, and the transfer curves a protocol changes. They take any
channel count, and each declares the locality that lets its patches stream.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as functional

from konfai.data.augmentation.base import DataAugmentation, _hashed_normal_field
from konfai.data.augmentation.placed import PlacedDraw
from konfai.data.transform import LocalityKind, PatchLocality, RegionContext
from konfai.utils.dataset import Attribute
from konfai.utils.errors import AugmentationError


def _blur_radius(sigma: float) -> int:
    """Where a Gaussian is cut: three sigmas, and never less than one voxel."""
    return max(1, math.ceil(3.0 * sigma))


def _draw_range(low: float, high: float, count: int) -> list[float]:
    return (low + torch.rand(count) * (high - low)).tolist()


class GaussianNoise(PlacedDraw):
    """Additive Gaussian noise, one standard deviation per copy.

    The field is a function of the voxel's place in the volume, so a region holds its own part of it
    and a patch reads what the whole volume would have given it.
    """

    def __init__(self, std_min: float = 0.0, std_max: float = 0.1, groups: list[str] | None = None) -> None:
        super().__init__(groups)
        if std_max < std_min:
            raise AugmentationError("GaussianNoise: std_max is below std_min.", "Swap the two bounds.")
        self.std_min, self.std_max = float(std_min), float(std_max)
        self.stds: dict[int, list[float]] = {}
        self.seeds: dict[int, list[int]] = {}

    def _state_init(self, index: int, shapes: list[list[int]], caches_attribute: list[Attribute]) -> list[list[int]]:
        del caches_attribute
        self.stds[index] = _draw_range(self.std_min, self.std_max, len(shapes))
        self.seeds[index] = torch.randint(0, 2**31 - 1, (len(shapes),)).tolist()
        return shapes

    def _apply(
        self, index: int, a: int, tensor: torch.Tensor, offsets: tuple[int, ...], full: tuple[int, ...]
    ) -> torch.Tensor:
        std = self.stds[index][a]
        if std == 0:
            return tensor
        field = _hashed_normal_field(self.seeds[index][a], tuple(tensor.shape), offsets, full, tensor.device)
        return (tensor.float() + field * std).to(tensor.dtype)


class GaussianBlur(DataAugmentation):
    """Separable Gaussian blur, one sigma per copy, with the kernel's own radius as its halo.

    ``in_plane`` blurs the two innermost axes only, which is what a 2.5D stack wants: the channels a
    patch carries are neighbouring slices, and blurring along the slice axis mixes them before the
    network ever sees them as separate inputs.
    """

    def __init__(
        self,
        sigma_min: float = 0.5,
        sigma_max: float = 1.0,
        in_plane: bool = False,
        groups: list[str] | None = None,
    ) -> None:
        super().__init__(groups)
        if sigma_max < sigma_min:
            raise AugmentationError("GaussianBlur: sigma_max is below sigma_min.", "Swap the two bounds.")
        self.sigma_min, self.sigma_max = float(sigma_min), float(sigma_max)
        self.in_plane = bool(in_plane)
        self.sigmas: dict[int, list[float]] = {}
        self.ranks: dict[int, int] = {}

    def _state_init(self, index: int, shapes: list[list[int]], caches_attribute: list[Attribute]) -> list[list[int]]:
        del caches_attribute
        self.sigmas[index] = _draw_range(self.sigma_min, self.sigma_max, len(shapes))
        self.ranks[index] = len(shapes[0])
        return shapes

    @staticmethod
    def _kernel(sigma: float, device: torch.device) -> torch.Tensor:
        """The 1-D kernel, cut at three sigmas: the radius the halo declares is this one."""
        radius = _blur_radius(sigma)
        positions = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
        line = torch.exp(-0.5 * (positions / sigma) ** 2)
        return line / line.sum()

    def _patch_locality(self, index: int, a: int, cache_attribute: Attribute) -> PatchLocality:
        del cache_attribute
        radius, rank = _blur_radius(self.sigmas[index][a]), self.ranks.get(index, 1)
        # Per axis, not broadcast: a 2.5D patch is one slice thick, and a radius on that axis asks
        # for a neighbourhood wider than the patch, which the plan refuses.
        halo = (0, *([radius] * (rank - 1))) if self.in_plane and rank > 1 else (radius,)
        return PatchLocality(LocalityKind.HALO, halo=halo)

    def _compute(self, name: str, index: int, a: int, tensor: torch.Tensor) -> torch.Tensor:
        del name
        sigma = self.sigmas[index][a]
        if sigma == 0:
            return tensor
        rank = tensor.ndim - 1
        if rank not in (2, 3):
            raise AugmentationError(
                f"GaussianBlur takes a 2D or 3D tensor, got {tuple(tensor.shape)}.",
                "Give the group a [C, Y, X] or [C, Z, Y, X] tensor.",
            )
        blurred = tensor.unsqueeze(0).float()
        axes = range(1, rank) if self.in_plane else range(rank)
        for axis in axes:
            blurred = self._blur_axis(blurred, axis, rank, sigma)
        return blurred.squeeze(0).to(tensor.dtype)

    @classmethod
    def _blur_axis(cls, tensor: torch.Tensor, axis: int, rank: int, sigma: float) -> torch.Tensor:
        """One separable pass: the 1-D kernel laid on ``axis`` and one padded convolution."""
        line = cls._kernel(sigma, tensor.device)
        shape = [1] * rank
        shape[axis] = line.numel()
        channels = int(tensor.shape[1])
        kernel = line.reshape(1, 1, *shape).repeat(channels, 1, *([1] * rank))
        padding = [0] * rank
        padding[axis] = _blur_radius(sigma)
        convolution = functional.conv2d if rank == 2 else functional.conv3d
        return convolution(tensor, kernel, padding=tuple(padding), groups=channels)

    def _inverse(self, index: int, a: int, tensor: torch.Tensor) -> torch.Tensor:
        del index, a
        return tensor


class SimulateLowResolution(DataAugmentation):
    """Resample each plane down and back up, simulating a coarser in-plane acquisition.

    The grid the resize samples on is laid over the plane's WHOLE extent, so a region cannot be
    degraded on its own terms. It is degraded on the plane's terms instead: told where it sits and
    how wide the plane is, a region reproduces the values the whole plane would have given it, and
    the halo is how far the two resamplings reach.
    """

    def __init__(
        self,
        factor_min: float = 1.0,
        factor_max: float = 2.0,
        groups: list[str] | None = None,
    ) -> None:
        super().__init__(groups)
        if factor_min < 1.0:
            raise AugmentationError(
                "SimulateLowResolution: factor_min is below 1.",
                "A factor of 1 keeps the resolution; below 1 raises it.",
            )
        if factor_max < factor_min:
            raise AugmentationError("SimulateLowResolution: factor_max is below factor_min.", "Swap the two bounds.")
        self.factor_min, self.factor_max = float(factor_min), float(factor_max)
        self.factors: dict[int, list[float]] = {}
        self.ranks: dict[int, int] = {}

    def _state_init(self, index: int, shapes: list[list[int]], caches_attribute: list[Attribute]) -> list[list[int]]:
        del caches_attribute
        self.factors[index] = _draw_range(self.factor_min, self.factor_max, len(shapes))
        self.ranks[index] = len(shapes[0])
        return shapes

    def _patch_locality(self, index: int, a: int, cache_attribute: Attribute) -> PatchLocality:
        del cache_attribute
        factor = self.factors[index][a]
        # Down then up: an output reaches one coarse sample away, and a coarse sample one factor of
        # input away, each with a tap on either side. The slice axis is untouched.
        radius = math.ceil(factor) + 2
        rank = self.ranks.get(index, 2)
        return PatchLocality(LocalityKind.HALO, halo=(0, radius, radius) if rank > 2 else (radius, radius))

    @staticmethod
    def _degrade_axis(values: torch.Tensor, axis: int, offset: int, full: int, factor: float) -> torch.Tensor:
        """One axis, resampled down to ``full / factor`` samples and back, on the plane's own grid.

        Every coordinate is absolute, so the window's place in the plane is what decides its values.
        ``align_corners=False``: sample ``k`` of a length-``n`` grid sits at ``(k + 0.5) * span - 0.5``.
        """
        device, local = values.device, int(values.shape[axis])
        coarse = max(1, round(full / factor))
        lay = [1] * values.ndim
        lay[axis] = -1

        def tap(coordinate: torch.Tensor, extent: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            # The coordinate is bounded first: past the edge the grid takes the edge sample itself,
            # it does not extrapolate from the two nearest.
            coordinate = coordinate.clamp(0.0, float(extent - 1))
            low = coordinate.floor()
            return low, (low + 1).clamp(0, extent - 1), coordinate - low

        def gather(index: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            # A coarse sample reads the input; an index outside the window is clamped to it, and the
            # values it feeds land in the halo the dispatcher crops away.
            source = (index + 0.5) * full / coarse - 0.5
            low, high, blend = tap(source, full)
            left = values.index_select(axis, (low - offset).clamp(0, local - 1).long())
            right = values.index_select(axis, (high - offset).clamp(0, local - 1).long())
            sampled = left * (1.0 - blend.reshape(lay)) + right * blend.reshape(lay)
            return sampled * weight

        positions = torch.arange(offset, offset + local, device=device, dtype=torch.float32)
        low, high, blend = tap((positions + 0.5) * coarse / full - 0.5, coarse)
        return gather(low, (1.0 - blend).reshape(lay)) + gather(high, blend.reshape(lay))

    def _degrade(
        self, index: int, a: int, tensor: torch.Tensor, offsets: tuple[int, ...], full: tuple[int, ...]
    ) -> torch.Tensor:
        factor = self.factors[index][a]
        if factor == 1.0:
            return tensor
        if tensor.ndim not in (3, 4):
            raise AugmentationError(
                f"SimulateLowResolution takes a 2D or 3D tensor, got {tuple(tensor.shape)}.",
                "Give the group a [C, Y, X] or [C, Z, Y, X] tensor.",
            )
        degraded = tensor.float()
        for axis in (tensor.ndim - 2, tensor.ndim - 1):
            spatial = axis - 1
            degraded = self._degrade_axis(degraded, axis, int(offsets[spatial]), int(full[spatial]), factor)
        return degraded.to(tensor.dtype)

    def _compute(self, name: str, index: int, a: int, tensor: torch.Tensor) -> torch.Tensor:
        del name
        full = tuple(int(extent) for extent in tensor.shape[1:])
        return self._degrade(index, a, tensor, tuple(0 for _ in full), full)

    def _stream_region(
        self, name: str, index: int, a: int, tensor: torch.Tensor, context: RegionContext
    ) -> torch.Tensor:
        del name
        offsets = tuple(int(part.start) for part in context.source)
        return self._degrade(index, a, tensor, offsets, tuple(int(extent) for extent in context.source_shape))

    def _inverse(self, index: int, a: int, tensor: torch.Tensor) -> torch.Tensor:
        del index, a
        return tensor


class _CaseStatisticDraw(DataAugmentation):
    """A draw whose parameters are a function of the whole case's values.

    The statistic is the case's, on both routes: the whole-volume pass measures it from the tensor it
    is handed, which is the case, and records it in the scope; a streamed region reads what the plan
    seeded there. A region never describes itself, or the same voxel would take one value through a
    patch and another through a volume.
    """

    stat_keys: frozenset[str] = frozenset()

    def __init__(self, groups: list[str] | None = None) -> None:
        super().__init__(groups)
        self.statistics: dict[tuple[int, int], dict[str, torch.Tensor]] = {}

    def _patch_locality(self, index: int, a: int, cache_attribute: Attribute) -> PatchLocality:
        del index, a, cache_attribute
        return PatchLocality(LocalityKind.GLOBAL_STAT, stat_keys=self.stat_keys)

    def _measure(self, tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def _seed_statistics(
        self, index: int, a: int, tensor: torch.Tensor, cache_attribute: Attribute, whole: bool
    ) -> None:
        if whole:
            measured = self._measure(tensor)
            for key, value in measured.items():
                cache_attribute[key] = value.detach().cpu().numpy()
            self.statistics[(index, a)] = measured
            return
        missing = sorted(key for key in self.stat_keys if key not in cache_attribute)
        if missing:
            raise AugmentationError(
                f"'{type(self).__name__}' was handed a region without {missing}, which describe the whole case.",
                "Report this: a GLOBAL_STAT draw reaching a region unseeded is a planning fault, and"
                " describing the region instead would make the same voxel take two values.",
            )
        # The scope stringifies what it holds: the array comes back through the parser, not the item.
        self.statistics[(index, a)] = {key: cache_attribute.get_tensor(key).to(tensor.device) for key in self.stat_keys}

    def _shaped(self, index: int, a: int, key: str, tensor: torch.Tensor) -> torch.Tensor:
        """The statistic laid out to broadcast against ``tensor``: one value per channel."""
        value = self.statistics[(index, a)][key].to(tensor.device, torch.float32)
        return value.reshape(-1, *([1] * (tensor.ndim - 1)))

    def _inverse(self, index: int, a: int, tensor: torch.Tensor) -> torch.Tensor:
        del index, a
        return tensor


class Gamma(_CaseStatisticDraw):
    """Gamma on the case's own range, one exponent per copy.

    The values are mapped to ``[0, 1]`` by the case's per-channel extrema, raised to the exponent,
    and mapped back, so the draw changes the transfer curve and not the range.
    """

    stat_keys = frozenset({"MinPerChannel", "MaxPerChannel"})

    def __init__(
        self,
        gamma_min: float = 0.7,
        gamma_max: float = 1.5,
        eps: float = 1e-6,
        groups: list[str] | None = None,
    ) -> None:
        super().__init__(groups)
        if gamma_max < gamma_min:
            raise AugmentationError("Gamma: gamma_max is below gamma_min.", "Swap the two bounds.")
        self.gamma_min, self.gamma_max, self.eps = float(gamma_min), float(gamma_max), float(eps)
        self.gammas: dict[int, list[float]] = {}

    def _state_init(self, index: int, shapes: list[list[int]], caches_attribute: list[Attribute]) -> list[list[int]]:
        del caches_attribute
        self.gammas[index] = _draw_range(self.gamma_min, self.gamma_max, len(shapes))
        return shapes

    def _measure(self, tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        flat = tensor.float().reshape(int(tensor.shape[0]), -1)
        return {"MinPerChannel": flat.min(1).values, "MaxPerChannel": flat.max(1).values}

    def _compute(self, name: str, index: int, a: int, tensor: torch.Tensor) -> torch.Tensor:
        del name
        low = self._shaped(index, a, "MinPerChannel", tensor)
        scale = (self._shaped(index, a, "MaxPerChannel", tensor) - low).clamp_min(self.eps)
        unit = ((tensor.float() - low) / scale).clamp(0, 1)
        return (unit.pow(self.gammas[index][a]) * scale + low).to(tensor.dtype)


class ContrastAroundMean(_CaseStatisticDraw):
    """Scale each value's distance to the case's per-channel mean, one factor per copy.

    The colour ``Contrast`` next door scales towards zero through a colour matrix; this one scales
    towards the mean the case actually has, which is what an intensity protocol changes.
    """

    stat_keys = frozenset({"MeanPerChannel"})

    def __init__(
        self,
        factor_min: float = 0.75,
        factor_max: float = 1.25,
        groups: list[str] | None = None,
    ) -> None:
        super().__init__(groups)
        if factor_max < factor_min:
            raise AugmentationError("ContrastAroundMean: factor_max is below factor_min.", "Swap the two bounds.")
        self.factor_min, self.factor_max = float(factor_min), float(factor_max)
        self.factors: dict[int, list[float]] = {}

    def _state_init(self, index: int, shapes: list[list[int]], caches_attribute: list[Attribute]) -> list[list[int]]:
        del caches_attribute
        self.factors[index] = _draw_range(self.factor_min, self.factor_max, len(shapes))
        return shapes

    def _measure(self, tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"MeanPerChannel": tensor.float().reshape(int(tensor.shape[0]), -1).mean(1)}

    def _compute(self, name: str, index: int, a: int, tensor: torch.Tensor) -> torch.Tensor:
        del name
        mean = self._shaped(index, a, "MeanPerChannel", tensor)
        return ((tensor.float() - mean) * self.factors[index][a] + mean).to(tensor.dtype)
