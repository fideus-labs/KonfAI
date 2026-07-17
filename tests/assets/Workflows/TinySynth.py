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


class Head(network.ModuleArgsDict):
    def __init__(self):
        super().__init__()
        self.add_module("Tanh", torch.nn.Tanh())


class Affine(torch.nn.Module):
    """Per-voxel ``y = w * x + b`` as elementwise ops.

    The same input value gives the same output bits whatever the tensor shape, which is what the
    patched-equals-whole byte-identity tests compare across. A 1x1 convolution is not that: it
    routes through shape-dependent GEMM kernels whose FMA tail handling can differ by one ULP
    between patch sizes (observed on macOS arm64 Accelerate).
    """

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.bias = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight + self.bias


class TinySynthNet(network.Network):
    def __init__(
        self,
        optimizer: network.OptimizerLoader = network.OptimizerLoader(),
        schedulers: dict[str, network.LRSchedulersLoader] = {"default|ConstantLR": network.LRSchedulersLoader(0)},
        outputs_criterions: dict[str, network.TargetCriterionsLoader] = {"Head:Tanh": network.TargetCriterionsLoader()},
    ) -> None:
        super().__init__(
            in_channels=1,
            optimizer=optimizer,
            schedulers=schedulers,
            outputs_criterions=outputs_criterions,
            dim=2,
        )
        self.add_module("Projection", Affine())
        self.add_module("Head", Head())
