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


class Concat(torch.nn.Module):
    def forward(self, image: torch.Tensor, statistic: torch.Tensor) -> torch.Tensor:
        return torch.cat([image, statistic], dim=1)


class Head(network.ModuleArgsDict):
    def __init__(self) -> None:
        super().__init__()
        self.add_module("Softmax", torch.nn.Softmax(dim=1))


class TwoInputNet(network.Network):
    """Reads the image and a constant statistic channel: two logits, the image and the statistic."""

    def __init__(
        self,
        optimizer: network.OptimizerLoader = network.OptimizerLoader(),
        schedulers: dict[str, network.LRSchedulersLoader] = {"default|ConstantLR": network.LRSchedulersLoader(0)},
        outputs_criterions: dict[str, network.TargetCriterionsLoader] = {"Logits": network.TargetCriterionsLoader()},
    ) -> None:
        super().__init__(
            in_channels=1, optimizer=optimizer, schedulers=schedulers, outputs_criterions=outputs_criterions, dim=2
        )
        self.add_module("Logits", Concat(), in_branch=[0, 1])
        self.add_module("Head", Head())
