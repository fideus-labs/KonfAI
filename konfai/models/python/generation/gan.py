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

import itertools
from functools import partial

import torch
from konfai.data.patching import ModelPatch
from konfai.network import blocks, network


class Discriminator(network.Network):
    class DiscriminatorNLayers(network.ModuleArgsDict):
        def __init__(self, channels: list[int], strides: list[int], dim: int) -> None:
            super().__init__()
            block_config = partial(
                blocks.BlockConfig,
                kernel_size=4,
                padding=1,
                bias=False,
                activation=partial(torch.nn.LeakyReLU, negative_slope=0.2, inplace=True),
                norm_mode=blocks.NormMode.SYNCBATCH,
            )
            for i, (in_channels, out_channels, stride) in enumerate(zip(channels, channels[1:], strides, strict=False)):
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
            self.add_module(
                "AdaptiveAvgPool",
                blocks.get_torch_module("AdaptiveAvgPool", dim)(tuple([1] * dim)),
            )
            self.add_module("Flatten", torch.nn.Flatten(1))

    def __init__(
        self,
        optimizer: network.OptimizerLoader = network.OptimizerLoader(),
        schedulers: dict[str, network.LRSchedulersLoader] = {
            "default|ReduceLROnPlateau": network.LRSchedulersLoader(0)
        },
        outputs_criterions: dict[str, network.TargetCriterionsLoader] = {"default": network.TargetCriterionsLoader()},
        nb_batch_per_step: int = 64,
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
        channels = [1, 16, 32, 64, 64]
        strides = [2, 2, 2, 1]
        self.add_module("Layers", Discriminator.DiscriminatorNLayers(channels, strides, dim))
        self.add_module("Head", Discriminator.DiscriminatorHead(channels[-1], dim))


class Generator(network.Network):
    class GeneratorStem(network.ModuleArgsDict):
        def __init__(self, in_channels: int, out_channels: int, dim: int) -> None:
            super().__init__()
            self.add_module(
                "ConvBlock",
                blocks.ConvBlock(
                    in_channels,
                    out_channels,
                    block_configs=[blocks.BlockConfig(bias=False, activation="ReLU", norm_mode="SYNCBATCH")],
                    dim=dim,
                ),
            )

    class GeneratorHead(network.ModuleArgsDict):
        def __init__(self, in_channels: int, out_channels: int, dim: int) -> None:
            super().__init__()
            self.add_module(
                "ConvBlock",
                blocks.ConvBlock(
                    in_channels,
                    in_channels,
                    block_configs=[blocks.BlockConfig(bias=False, activation="ReLU", norm_mode="SYNCBATCH")],
                    dim=dim,
                ),
            )
            self.add_module(
                "Conv",
                blocks.get_torch_module("Conv", dim)(in_channels, out_channels, kernel_size=1, bias=False),
            )
            self.add_module("Tanh", torch.nn.Tanh())

    class GeneratorDownSample(network.ModuleArgsDict):
        def __init__(self, in_channels: int, out_channels: int, dim: int) -> None:
            super().__init__()
            self.add_module(
                "ConvBlock",
                blocks.ConvBlock(
                    in_channels,
                    out_channels,
                    block_configs=[blocks.BlockConfig(stride=2, bias=False, activation="ReLU", norm_mode="SYNCBATCH")],
                    dim=dim,
                ),
            )

    class GeneratorUpSample(network.ModuleArgsDict):
        def __init__(self, in_channels: int, out_channels: int, dim: int) -> None:
            super().__init__()
            self.add_module(
                "ConvBlock",
                blocks.ConvBlock(
                    in_channels,
                    out_channels,
                    block_configs=[blocks.BlockConfig(bias=False, activation="ReLU", norm_mode="SYNCBATCH")],
                    dim=dim,
                ),
            )
            self.add_module(
                "Upsample",
                torch.nn.Upsample(scale_factor=2, mode="bilinear" if dim < 3 else "trilinear"),
            )

    class GeneratorEncoder(network.ModuleArgsDict):
        def __init__(self, channels: list[int], dim: int) -> None:
            super().__init__()
            for i, (in_channels, out_channels) in enumerate(itertools.pairwise(channels)):
                self.add_module(
                    f"DownSample_{i}",
                    Generator.GeneratorDownSample(in_channels=in_channels, out_channels=out_channels, dim=dim),
                )

    class GeneratorResnetBlock(network.ModuleArgsDict):
        def __init__(self, channels: int, dim: int):
            super().__init__()
            self.add_module(
                "Conv_0",
                blocks.get_torch_module("Conv", dim)(channels, channels, kernel_size=3, padding=1, bias=False),
            )
            self.add_module("Norm", torch.nn.LeakyReLU(0.2, inplace=True))
            self.add_module(
                "Conv_1",
                blocks.get_torch_module("Conv", dim)(channels, channels, kernel_size=3, padding=1, bias=False),
            )
            self.add_module("Residual", blocks.Add(), in_branch=[0, 1])

    class GeneratorNResnetBlock(network.ModuleArgsDict):
        def __init__(self, channels: int, nb_conv: int, dim: int) -> None:
            super().__init__()
            for i in range(nb_conv):
                self.add_module(
                    f"ResnetBlock_{i}",
                    Generator.GeneratorResnetBlock(channels=channels, dim=dim),
                )

    class GeneratorDecoder(network.ModuleArgsDict):
        def __init__(self, channels: list[int], dim: int) -> None:
            super().__init__()
            for i, (in_channels, out_channels) in enumerate(
                zip(reversed(channels), reversed(channels[:-1]), strict=False)
            ):
                self.add_module(
                    f"UpSample_{i}",
                    Generator.GeneratorUpSample(in_channels=in_channels, out_channels=out_channels, dim=dim),
                )

    class GeneratorAutoEncoder(network.ModuleArgsDict):
        def __init__(self, ngf: int, dim: int) -> None:
            super().__init__()
            channels = [ngf, ngf * 2]
            self.add_module("Encoder", Generator.GeneratorEncoder(channels, dim))
            self.add_module(
                "NResBlock",
                Generator.GeneratorNResnetBlock(channels=channels[-1], nb_conv=6, dim=dim),
            )
            self.add_module("Decoder", Generator.GeneratorDecoder(channels, dim))

    def __init__(
        self,
        optimizer: network.OptimizerLoader = network.OptimizerLoader(),
        schedulers: dict[str, network.LRSchedulersLoader] = {
            "default|ReduceLROnPlateau": network.LRSchedulersLoader(0)
        },
        patch: ModelPatch = ModelPatch(),
        outputs_criterions: dict[str, network.TargetCriterionsLoader] = {"default": network.TargetCriterionsLoader()},
        nb_batch_per_step: int = 64,
        dim: int = 3,
    ) -> None:
        super().__init__(
            optimizer=optimizer,
            in_channels=1,
            schedulers=schedulers,
            patch=patch,
            outputs_criterions=outputs_criterions,
            dim=dim,
            nb_batch_per_step=nb_batch_per_step,
        )
        ngf = 32
        self.add_module("Stem", Generator.GeneratorStem(1, ngf, dim))
        self.add_module("AutoEncoder", Generator.GeneratorAutoEncoder(ngf, dim))
        self.add_module("Head", Generator.GeneratorHead(in_channels=ngf, out_channels=1, dim=dim))

    def get_name(self):
        return "Generator"


class Gan(network.Network):
    def __init__(
        self,
        generator: Generator = Generator(),
        discriminator: Discriminator = Discriminator(),
    ) -> None:
        super().__init__()
        self.add_module(
            "Discriminator_B",
            discriminator,
            in_branch=[1],
            out_branch=[-1],
            requires_grad=True,
        )
        self.add_module("Generator_A_to_B", generator, in_branch=[0], out_branch=["pB"])

        self.add_module("detach", blocks.Detach(), in_branch=["pB"], out_branch=["pB_detach"])
        self.add_module(
            "Discriminator_pB_detach",
            discriminator,
            in_branch=["pB_detach"],
            out_branch=[-1],
        )

        self.add_module(
            "Discriminator_pB",
            discriminator,
            in_branch=["pB"],
            out_branch=[-1],
            requires_grad=False,
        )
