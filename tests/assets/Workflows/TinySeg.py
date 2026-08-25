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

import torch
from konfai.network import network


class Bands(torch.nn.Module):
    """Two logits per voxel from one intensity, as elementwise ``y_c = w_c * x + b_c``.

    Elementwise for the reason ``TinySynth.Affine`` is: the same input value gives the same output
    bits whatever the tensor shape, where a 1x1 convolution routes through shape-dependent GEMM
    kernels that can differ by one ULP between patch sizes.
    """

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([-1.0, 1.0]))
        self.bias = torch.nn.Parameter(torch.tensor([0.25, -0.25]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = (1, -1, *([1] * (x.dim() - 2)))
        return x * self.weight.view(shape) + self.bias.view(shape)


class Head(network.ModuleArgsDict):
    def __init__(self) -> None:
        super().__init__()
        self.add_module("Softmax", torch.nn.Softmax(dim=1))


class TinySegNet(network.Network):
    """Two-class segmentation: ``Logits`` carries the cross-entropy, ``Head:Softmax`` the output."""

    def __init__(
        self,
        optimizer: network.OptimizerLoader = network.OptimizerLoader(),
        schedulers: dict[str, network.LRSchedulersLoader] = {"default|ConstantLR": network.LRSchedulersLoader(0)},
        outputs_criterions: dict[str, network.TargetCriterionsLoader] = {"Logits": network.TargetCriterionsLoader()},
    ) -> None:
        super().__init__(
            in_channels=1,
            optimizer=optimizer,
            schedulers=schedulers,
            outputs_criterions=outputs_criterions,
            dim=2,
        )
        self.add_module("Logits", Bands())
        self.add_module("Head", Head())
