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

from functools import partial

import segmentation_models_pytorch as smp
import torch

from konfai.data.patching import ModelPatch
from konfai.network import blocks, network
from konfai.utils.config import config


class Head(network.ModuleArgsDict):

    def __init__(self):
        super().__init__()
        self.add_module("Tanh", torch.nn.Tanh())


@config()
class UNetpp5(network.Network):

    def __init__(
        self,
        optimizer: network.OptimizerLoader = network.OptimizerLoader(),
        schedulers: dict[str, network.LRSchedulersLoader] = {
            "default:ReduceLROnPlateau": network.LRSchedulersLoader(0)
        },
        patch: ModelPatch | None = None,
        outputs_criterions: dict[str, network.TargetCriterionsLoader] = {"default": network.TargetCriterionsLoader()},
        dim: int = 2,
    ):
        super().__init__(
            in_channels=5,
            optimizer=optimizer,
            schedulers=schedulers,
            patch=patch,
            outputs_criterions=outputs_criterions,
            dim=dim,
        )
        self.add_module(
            "model",
            smp.UnetPlusPlus(encoder_name="resnet34", encoder_weights=None, in_channels=5, classes=1, activation=None),
        )
        self.add_module("Head", Head())


@config()
class Discriminator(network.Network):

    class DiscriminatorNLayers(network.ModuleArgsDict):

        def __init__(self, channels: list[int], strides: list[int], dim: int) -> None:
            super().__init__()
            block_config = partial(
                blocks.BlockConfig,
                kernel_size=4,
                padding=1,
                bias=False,
                activation=partial(torch.nn.LeakyReLU, negative_slope=0.2, inplace=False),
                norm_mode=blocks.NormMode.SYNCBATCH,
            )
            for i, (in_channels, out_channels, stride) in enumerate(zip(channels, channels[1:], strides)):
                self.add_module(
                    f"Layer_{i}",
                    blocks.ConvBlock(in_channels, out_channels, [block_config(stride=stride)], dim),
                )

    class DiscriminatorHead(network.ModuleArgsDict):

        def __init__(self, channels: int, dim: int) -> None:
            super().__init__()
            self.add_module(
                "Conv",
                blocks.get_torch_module("Conv", dim)(
                    in_channels=channels,
                    out_channels=1,
                    kernel_size=4,
                    stride=1,
                    padding=1,
                ),
            )

    def __init__(
        self,
        optimizer: network.OptimizerLoader = network.OptimizerLoader(),
        schedulers: dict[str, network.LRSchedulersLoader] = {
            "default:ReduceLROnPlateau": network.LRSchedulersLoader(0)
        },
        outputs_criterions: dict[str, network.TargetCriterionsLoader] = {"default": network.TargetCriterionsLoader()},
        nb_batch_per_step: int = 1,
        dim: int = 3,
    ) -> None:
        super().__init__(
            in_channels=1,
            optimizer=optimizer,
            schedulers=schedulers,
            outputs_criterions=outputs_criterions,
            dim=dim,
            nb_batch_per_step=nb_batch_per_step,
        )
        channels = [1, 32, 64, 64, 64]
        strides = [2, 2, 2, 2, 1]
        self.add_module("Sample", torch.nn.Identity())
        self.add_module("Layers", Discriminator.DiscriminatorNLayers(channels, strides, dim))
        self.add_module("Head", Discriminator.DiscriminatorHead(channels[-1], dim))


@config()
class Gan(network.Network):

    def __init__(
        self,
        generator: UNetpp5 = UNetpp5(),
        discriminator: Discriminator = Discriminator(),
    ) -> None:
        super().__init__()
        self.add_module("Generator_A_to_B", generator, in_branch=[0], out_branch=["pB"])
        self.add_module("Discriminator_B", discriminator, in_branch=[1], out_branch=[-1], requires_grad=True)
        self.add_module("detach", blocks.Detach(), in_branch=["pB"], out_branch=["pB_detach"])
        self.add_module("Discriminator_pB_detach", discriminator, in_branch=["pB_detach"], out_branch=[-1])
        self.add_module("Discriminator_pB", discriminator, in_branch=["pB"], out_branch=[-1], requires_grad=False)
