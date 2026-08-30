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


"""Draws that depend on where a voxel sits: noise, cut-outs, a companion mask."""

from abc import abstractmethod
from pathlib import Path

import torch

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None  # type: ignore[assignment]
from konfai.data.augmentation.base import DataAugmentation, _hashed_normal_field, _require_simpleitk
from konfai.data.transform import LocalityKind, PatchLocality, RegionContext
from konfai.utils.dataset import Attribute
from konfai.utils.errors import AugmentationError


class PlacedDraw(DataAugmentation):
    """A per-voxel draw whose value at a voxel is a function of the voxel's place in the volume.

    POINTWISE: a region computes exactly its part, given where it sits (``offsets``) and the volume's
    spatial extent (``full``), which the whole volume passes as zeros and its own shape.
    """

    def _patch_locality(self, index: int, a: int, cache_attribute: Attribute) -> PatchLocality:
        return PatchLocality(LocalityKind.POINTWISE)

    @abstractmethod
    def _apply(
        self, index: int, a: int, tensor: torch.Tensor, offsets: tuple[int, ...], full: tuple[int, ...]
    ) -> torch.Tensor:
        pass

    def _compute(self, name: str, index: int, a: int, tensor: torch.Tensor) -> torch.Tensor:
        full = tuple(int(extent) for extent in tensor.shape[1:])
        return self._apply(index, a, tensor, tuple(0 for _ in full), full)

    def _stream_region(
        self, name: str, index: int, a: int, tensor: torch.Tensor, context: RegionContext
    ) -> torch.Tensor:
        offsets = tuple(int(part.start) for part in context.source)
        return self._apply(index, a, tensor, offsets, tuple(int(extent) for extent in context.source_shape))

    def _inverse(self, index: int, a: int, tensor: torch.Tensor) -> torch.Tensor:
        # A value draw moves no voxel: the inverse a TTA applies to a prediction is the tensor itself.
        return tensor


class Noise(PlacedDraw):
    def __init__(
        self,
        n_std: float,
        noise_step: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        groups: list[str] | None = None,
    ) -> None:
        super().__init__(groups)
        self.n_std = n_std
        self.noise_step = noise_step

        self.ts: dict[int, list[torch.Tensor]] = {}
        self.field_seeds: dict[int, list[int]] = {}  #: one field seed per copy, drawn with the step
        self.betas = torch.linspace(beta_start, beta_end, noise_step)
        self.betas = Noise.enforce_zero_terminal_snr(self.betas)
        self.alphas = 1 - self.betas
        self.alpha_hat = torch.concat((torch.ones(1), torch.cumprod(self.alphas, dim=0)))
        self.max_T = 0.0

    @staticmethod
    def enforce_zero_terminal_snr(betas: torch.Tensor):
        alphas = 1 - betas
        alphas_bar = alphas.cumprod(0)
        alphas_bar_sqrt = alphas_bar.sqrt()
        alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
        alphas_bar_sqrt_t = alphas_bar_sqrt[-1].clone()
        alphas_bar_sqrt -= alphas_bar_sqrt_t
        alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_t)
        alphas_bar = alphas_bar_sqrt**2
        alphas = alphas_bar[1:] / alphas_bar[:-1]
        alphas = torch.cat([alphas_bar[0:1], alphas])
        betas = 1 - alphas
        return betas

    def load(self, prob: float):
        # Every copy is drawn: the probability scales the noise step, not whether a copy gets one.
        self._prob = 1.0
        self.max_T = prob * self.noise_step

    def _state_init(self, index: int, shapes: list[list[int]], caches_attribute: list[Attribute]) -> list[list[int]]:
        if int(self.max_T) == 0:
            self.ts[index] = [0 for _ in shapes]
        else:
            self.ts[index] = [torch.randint(0, int(self.max_T), (1,)) for _ in shapes]
        self.field_seeds[index] = [int(torch.randint(0, 2**31 - 1, (1,))) for _ in shapes]
        return shapes

    def _apply(
        self, index: int, a: int, tensor: torch.Tensor, offsets: tuple[int, ...], full: tuple[int, ...]
    ) -> torch.Tensor:
        alpha_hat_t = self.alpha_hat[self.ts[index][a]].to(tensor.device).reshape(*[1 for _ in tensor.shape])
        field = _hashed_normal_field(self.field_seeds[index][a], tuple(tensor.shape), offsets, full, tensor.device)
        return alpha_hat_t.sqrt() * tensor + (1 - alpha_hat_t).sqrt() * field * self.n_std


class CutOUT(PlacedDraw):
    def __init__(
        self,
        c_prob: float,
        cutout_size: int,
        value: float,
        groups: list[str] | None = None,
    ) -> None:
        super().__init__(groups)
        self.c_prob = c_prob
        self.cutout_size = cutout_size
        self.centers: dict[int, list[torch.Tensor]] = {}
        self.value = value

    def _state_init(self, index: int, shapes: list[list[int]], caches_attribute: list[Attribute]) -> list[list[int]]:
        self.centers[index] = [torch.rand((3) if len(shape) == 3 else (2)) for shape in shapes]
        return shapes

    def _apply(
        self, index: int, a: int, tensor: torch.Tensor, offsets: tuple[int, ...], full: tuple[int, ...]
    ) -> torch.Tensor:
        center = self.centers[index][a]
        masks = []
        for i, w in enumerate(tensor.shape[1:]):
            re = [1] * i + [-1] + [1] * (len(tensor.shape[1:]) - i - 1)
            positions = torch.arange(offsets[i], offsets[i] + w).reshape(re)
            masks.append(
                ((positions + 0.5) / full[i] - center[i].reshape([1, 1])).abs()
                >= torch.tensor(self.cutout_size).reshape([1, 1]) / 2
            )
        result = masks[0]
        for mask in masks[1:]:
            result = torch.logical_or(result, mask)
        # The bool mask broadcasts over the channels: repeated C times and re-tested against 1 it
        # was a copy and a pass over C times the volume (measured 54-60 ms at 8x128^3, 34-40 without).
        return torch.where(result.unsqueeze(0).to(tensor.device), tensor, torch.tensor(self.value).to(tensor.device))


class Mask(DataAugmentation):
    def __init__(self, mask: str, value: float, groups: list[str] | None = None) -> None:
        _require_simpleitk()
        super().__init__(groups)
        self.mask_path = Path(mask)
        if not self.mask_path.is_file():
            raise AugmentationError(f"Mask file '{self.mask_path}' does not exist.")
        reader = sitk.ImageFileReader()
        reader.SetFileName(str(self.mask_path))
        reader.ReadImageInformation()
        self.mask_shape = tuple(reversed(reader.GetSize()))
        self._mask: torch.Tensor | None = None
        self.positions: dict[int, list[torch.Tensor]] = {}
        self.value = value

    def _load_mask(self) -> torch.Tensor:
        if self._mask is None:
            self._mask = torch.from_numpy(sitk.GetArrayFromImage(sitk.ReadImage(str(self.mask_path))))
        return self._mask

    def _state_init(self, index: int, shapes: list[list[int]], caches_attribute: list[Attribute]) -> list[list[int]]:
        self.positions[index] = [
            torch.rand((3) if len(shape) == 3 else (2))
            * (torch.tensor([max(s1 - s2, 0) for s1, s2 in zip(torch.tensor(shape), self.mask_shape, strict=False)]))
            for shape in shapes
        ]
        return [list(self.mask_shape) for _ in shapes]

    def _stream_shape(self, index: int, a: int, shape: list[int]) -> list[int]:
        # The mask's own grid, the extent state_init gave the copy: the draw crops or pads to it.
        return list(self.mask_shape)

    # WHOLE_VOLUME on purpose: the output grid is the mask's, and the mask volume is already resident
    # at that extent: there is no whole-volume read left for a declaration to save.
    def _compute(self, name: str, index: int, a: int, tensor: torch.Tensor) -> torch.Tensor:
        mask = self._load_mask()
        position = self.positions[index][a]
        slices = [slice(None, None)] + [
            slice(int(s1), int(s1) + s2) for s1, s2 in zip(position, mask.shape, strict=False)
        ]
        padding = []
        for s1, s2 in zip(reversed(tensor.shape), reversed(mask.shape), strict=False):
            padding.append(0)
            padding.append(s2 - s1 if s1 < s2 else 0)
        value = (
            torch.tensor(0, dtype=torch.uint8)
            if tensor.dtype == torch.uint8
            else torch.tensor(self.value).to(tensor.device)
        )
        return torch.where(
            mask.to(tensor.device) == 1,
            torch.nn.functional.pad(tensor, tuple(padding), mode="constant", value=value)[tuple(slices)],
            value,
        )

    def _inverse(self, index: int, a: int, tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Mask augmentation has no inverse; do not use it for invertible TTA.")
