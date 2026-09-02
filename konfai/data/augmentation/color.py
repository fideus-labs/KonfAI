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


"""Draws on intensities: brightness, contrast, luma, hue, saturation."""

import numpy as np
import torch

from konfai.data.augmentation.base import DataAugmentation, _axis_rotation_matrix, _scale_matrix, _translate_matrix
from konfai.data.transform import LocalityKind
from konfai.utils.dataset import Attribute
from konfai.utils.errors import AugmentationError


class ColorTransform(DataAugmentation):
    # The draw is a colour matrix applied to each voxel on its own: no neighbour, no coordinate,
    # no extent. Whatever region a voxel is read in, it comes out the same.
    locality = LocalityKind.POINTWISE

    def __init__(self, groups: list[str] | None = None) -> None:
        super().__init__(groups)
        self.matrix: dict[int, list[torch.Tensor]] = {}

    def _compute(self, name: str, index: int, a: int, tensor: torch.Tensor) -> torch.Tensor:
        matrix = self.matrix[index][a]
        result = tensor.reshape([*tensor.shape[:1], int(np.prod(tensor.shape[1:]))])
        if tensor.shape[0] == 3:
            matrix = matrix.to(tensor.device)
            result = matrix[:, :3, :3] @ result.float() + matrix[:, :3, 3:]
        elif tensor.shape[0] == 1:
            matrix = matrix[:, :3, :].mean(dim=1, keepdims=True).to(tensor.device)
            result = result.float() * matrix[:, :, :3].sum(dim=2, keepdims=True) + matrix[:, :, 3:]
        else:
            raise AugmentationError("Image must be RGB (3 channels) or L (1 channel)")
        return result.reshape(tensor.shape)

    def _inverse(self, index: int, a: int, tensor: torch.Tensor) -> torch.Tensor:
        return tensor


class Brightness(ColorTransform):
    def __init__(self, b_std: float, groups: list[str] | None = None) -> None:
        super().__init__(groups)
        self.b_std = b_std

    def _state_init(self, index: int, shapes: list[list[int]], caches_attribute: list[Attribute]) -> list[list[int]]:
        brightness = torch.Tensor.repeat((torch.randn(len(shapes)) * self.b_std).unsqueeze(1), [1, 3])
        self.matrix[index] = [torch.unsqueeze(_translate_matrix(value), dim=0) for value in brightness]
        return shapes


class Contrast(ColorTransform):
    def __init__(self, c_std: float, groups: list[str] | None = None) -> None:
        super().__init__(groups)
        self.c_std = c_std

    def _state_init(self, index: int, shapes: list[list[int]], caches_attribute: list[Attribute]) -> list[list[int]]:
        contrast = torch.exp2(torch.randn(len(shapes)) * self.c_std)
        self.matrix[index] = [torch.unsqueeze(_scale_matrix(value.expand(3)), dim=0) for value in contrast]
        return shapes


class LumaFlip(ColorTransform):
    def __init__(self, groups: list[str] | None = None) -> None:
        super().__init__(groups)
        self.v = torch.tensor([1, 1, 1, 0]) / torch.sqrt(torch.tensor(3))

    def _state_init(self, index: int, shapes: list[list[int]], caches_attribute: list[Attribute]) -> list[list[int]]:
        luma = torch.floor(torch.rand([len(shapes), 1, 1]) * 2)
        self.matrix[index] = [torch.unsqueeze((torch.eye(4) - 2 * self.v.ger(self.v) * value), dim=0) for value in luma]
        return shapes


class HUE(ColorTransform):
    def __init__(self, hue_max: float, groups: list[str] | None = None) -> None:
        super().__init__(groups)
        self.hue_max = hue_max
        self.v = torch.tensor([1, 1, 1]) / torch.sqrt(torch.tensor(3))

    def _state_init(self, index: int, shapes: list[list[int]], caches_attribute: list[Attribute]) -> list[list[int]]:
        theta = (torch.rand([len(shapes)]) * 2 - 1) * np.pi * self.hue_max
        self.matrix[index] = [torch.unsqueeze(_axis_rotation_matrix(value, self.v), dim=0) for value in theta]
        return shapes


class Saturation(ColorTransform):
    def __init__(self, s_std: float, groups: list[str] | None = None) -> None:
        super().__init__(groups)
        self.s_std = s_std
        self.v = torch.tensor([1, 1, 1, 0]) / torch.sqrt(torch.tensor(3))

    def _state_init(self, index: int, shapes: list[list[int]], caches_attribute: list[Attribute]) -> list[list[int]]:
        saturation = torch.exp2(torch.randn(len(shapes)) * self.s_std)
        # Keep the luma component (v vT) at unit gain and scale only the orthogonal chroma component
        # (I - v vT) by the saturation factor. Scaling the whole matrix instead, (v vT + (I - v vT)) * s
        # = I * s, is a uniform per-channel gain (contrast) that never mixes toward luma. With this form
        # s=1 is identity, s=0 collapses to greyscale, s>1 boosts saturation.
        self.matrix[index] = [
            (self.v.ger(self.v) + (torch.eye(4) - self.v.ger(self.v)) * value).unsqueeze(0) for value in saturation
        ]
        return shapes
