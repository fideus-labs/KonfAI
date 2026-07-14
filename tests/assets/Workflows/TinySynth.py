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
        self.add_module(
            "Projection",
            torch.nn.Conv2d(1, 1, kernel_size=1, bias=True),
        )
        self.add_module("Head", Head())
