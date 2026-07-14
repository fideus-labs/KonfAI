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
from konfai.data.patching import ModelPatch
from konfai.network import blocks, network


class NestedUNet(network.Network):
    class NestedUNetBlock(network.ModuleArgsDict):
        def __init__(
            self,
            channels: list[int],
            nb_conv_per_stage: int,
            block_config: blocks.BlockConfig,
            downsample_mode: blocks.DownsampleMode,
            upsample_mode: blocks.UpsampleMode,
            attention: bool,
            block: type,
            dim: int,
            i: int = 0,
        ) -> None:
            super().__init__()
            if i > 0:
                self.add_module(
                    downsample_mode.name,
                    blocks.downsample(
                        in_channels=channels[0],
                        out_channels=channels[1],
                        downsample_mode=downsample_mode,
                        dim=dim,
                    ),
                )

            self.add_module(
                f"X_{i}_{0}",
                block(
                    in_channels=channels[(1 if downsample_mode == blocks.DownsampleMode.CONV_STRIDE and i > 0 else 0)],
                    out_channels=channels[1],
                    block_configs=[block_config] * nb_conv_per_stage,
                    dim=dim,
                ),
                out_branch=[f"X_{i}_{0}"],
            )
            if len(channels) > 2:
                self.add_module(
                    f"UNetBlock_{i + 1}",
                    NestedUNet.NestedUNetBlock(
                        channels[1:],
                        nb_conv_per_stage,
                        block_config,
                        downsample_mode,
                        upsample_mode,
                        attention,
                        block,
                        dim,
                        i + 1,
                    ),
                    in_branch=[f"X_{i}_{0}"],
                    out_branch=[f"X_{i + 1}_{j}" for j in range(len(channels) - 2)],
                )
                for j in range(len(channels) - 2):
                    self.add_module(
                        f"X_{i}_{j + 1}_{upsample_mode.name}",
                        blocks.upsample(
                            in_channels=channels[2],
                            out_channels=channels[1],
                            upsample_mode=upsample_mode,
                            dim=dim,
                        ),
                        in_branch=[f"X_{i + 1}_{j}"],
                        out_branch=[f"X_{i + 1}_{j}"],
                    )
                    self.add_module(
                        f"SkipConnection_{i}_{j + 1}",
                        blocks.Concat(),
                        in_branch=[f"X_{i + 1}_{j}"] + [f"X_{i}_{r}" for r in range(j + 1)],
                        out_branch=[f"X_{i}_{j + 1}"],
                    )
                    self.add_module(
                        f"X_{i}_{j + 1}",
                        block(
                            in_channels=(
                                (channels[1] * (j + 1) + channels[2])
                                if upsample_mode != blocks.UpsampleMode.CONV_TRANSPOSE
                                else channels[1] * (j + 2)
                            ),
                            out_channels=channels[1],
                            block_configs=[block_config] * nb_conv_per_stage,
                            dim=dim,
                        ),
                        in_branch=[f"X_{i}_{j + 1}"],
                        out_branch=[f"X_{i}_{j + 1}"],
                    )

    class NestedUNetHead(network.ModuleArgsDict):
        def __init__(self, in_channels: int, nb_class: int, activation: str, dim: int) -> None:
            super().__init__()
            self.add_module(
                "Conv",
                blocks.get_torch_module("Conv", dim)(
                    in_channels=in_channels,
                    out_channels=nb_class,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ),
            )
            if activation == "Softmax":
                self.add_module("Softmax", torch.nn.Softmax(dim=1))
                self.add_module("Argmax", blocks.ArgMax(dim=1))
            elif activation == "Tanh":
                self.add_module("Tanh", torch.nn.Tanh())

    def __init__(
        self,
        optimizer: network.OptimizerLoader = network.OptimizerLoader(),
        schedulers: dict[str, network.LRSchedulersLoader] = {
            "default|ReduceLROnPlateau": network.LRSchedulersLoader(0)
        },
        outputs_criterions: dict[str, network.TargetCriterionsLoader] = {"default": network.TargetCriterionsLoader()},
        patch: ModelPatch | None = None,
        dim: int = 3,
        channels: list[int] = [1, 64, 128, 256, 512, 1024],
        nb_class: int = 2,
        block_config: blocks.BlockConfig = blocks.BlockConfig(),
        nb_conv_per_stage: int = 2,
        downsample_mode: str = "MAXPOOL",
        upsample_mode: str = "CONV_TRANSPOSE",
        attention: bool = False,
        block_type: str = "Conv",
        activation: str = "Softmax",
    ) -> None:
        super().__init__(
            in_channels=channels[0],
            optimizer=optimizer,
            schedulers=schedulers,
            outputs_criterions=outputs_criterions,
            patch=patch,
            dim=dim,
        )

        self.add_module(
            "UNetBlock_0",
            NestedUNet.NestedUNetBlock(
                channels,
                nb_conv_per_stage,
                block_config,
                downsample_mode=blocks.DownsampleMode[downsample_mode],
                upsample_mode=blocks.UpsampleMode[upsample_mode],
                attention=attention,
                block=blocks.ConvBlock if block_type == "Conv" else blocks.ResBlock,
                dim=dim,
            ),
            out_branch=[f"X_0_{j + 1}" for j in range(len(channels) - 2)],
        )
        for j in range(len(channels) - 2):
            self.add_module(
                f"Head_{j}",
                NestedUNet.NestedUNetHead(
                    in_channels=channels[1],
                    nb_class=nb_class,
                    activation=activation,
                    dim=dim,
                ),
                in_branch=[f"X_0_{j + 1}"],
                out_branch=[-1],
            )


class UNetpp(network.Network):
    class ResNetEncoderLayer(network.ModuleArgsDict):
        def __init__(
            self,
            in_channel: int,
            out_channel: int,
            nb_block: int,
            dim: int,
            downsample_mode: blocks.DownsampleMode,
        ):
            super().__init__()
            for i in range(nb_block):
                if downsample_mode == blocks.DownsampleMode.MAXPOOL and i == 0:
                    self.add_module(
                        "DownSample",
                        blocks.get_torch_module("MaxPool", dim)(
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            dilation=1,
                            ceil_mode=False,
                        ),
                    )
                self.add_module(
                    f"ResBlock_{i}",
                    blocks.ResBlock(
                        in_channel,
                        out_channel,
                        [
                            blocks.BlockConfig(
                                3,
                                (2 if downsample_mode == blocks.DownsampleMode.CONV_STRIDE and i == 0 else 1),
                                1,
                                False,
                                "ReLU;True",
                                blocks.NormMode.BATCH,
                            ),
                            blocks.BlockConfig(3, 1, 1, False, None, blocks.NormMode.BATCH),
                        ],
                        dim=dim,
                    ),
                )
                in_channel = out_channel

    @staticmethod
    def resnet_encoder(channels: list[int], layers: list[int], dim: int) -> list[torch.nn.Module]:
        modules = []
        modules.append(
            blocks.ConvBlock(
                channels[0],
                channels[1],
                [blocks.BlockConfig(7, 2, 3, False, "ReLU", blocks.NormMode.BATCH)],
                dim=dim,
            )
        )
        for i, (in_channel, out_channel, layer) in enumerate(zip(channels[1:], channels[2:], layers, strict=False)):
            modules.append(
                UNetpp.ResNetEncoderLayer(
                    in_channel,
                    out_channel,
                    layer,
                    dim,
                    (blocks.DownsampleMode.MAXPOOL if i == 0 else blocks.DownsampleMode.CONV_STRIDE),
                )
            )
        return modules

    class UNetPPBlock(network.ModuleArgsDict):
        def __init__(
            self,
            encoder_channels: list[int],
            decoder_channels: list[int],
            encoders: list[torch.nn.Module],
            upsample_mode: blocks.UpsampleMode,
            dim: int,
            i: int = 0,
        ) -> None:
            super().__init__()
            self.add_module(f"X_{i}_{0}", encoders[0], out_branch=[f"X_{i}_{0}"])
            if len(encoder_channels) > 2:
                self.add_module(
                    f"UNetBlock_{i + 1}",
                    UNetpp.UNetPPBlock(
                        encoder_channels[1:],
                        decoder_channels[1:],
                        encoders[1:],
                        upsample_mode,
                        dim,
                        i + 1,
                    ),
                    in_branch=[f"X_{i}_{0}"],
                    out_branch=[f"X_{i + 1}_{j}" for j in range(len(encoder_channels) - 2)],
                )
                for j in range(len(encoder_channels) - 2):
                    in_channels = (
                        decoder_channels[3]
                        if j == len(encoder_channels) - 3 and len(encoder_channels) > 3
                        else encoder_channels[2]
                    )
                    out_channel = decoder_channels[2] if j == len(encoder_channels) - 3 else encoder_channels[1]
                    self.add_module(
                        f"X_{i}_{j + 1}_{upsample_mode.name}",
                        blocks.upsample(
                            in_channels=in_channels,
                            out_channels=out_channel,
                            upsample_mode=upsample_mode,
                            dim=dim,
                        ),
                        in_branch=[f"X_{i + 1}_{j}"],
                        out_branch=[f"X_{i + 1}_{j}"],
                    )
                    self.add_module(
                        f"SkipConnection_{i}_{j + 1}",
                        blocks.Concat(),
                        in_branch=[f"X_{i + 1}_{j}"] + [f"X_{i}_{r}" for r in range(j + 1)],
                        out_branch=[f"X_{i}_{j + 1}"],
                    )
                    self.add_module(
                        f"X_{i}_{j + 1}",
                        blocks.ConvBlock(
                            in_channels=encoder_channels[1] * (j + 1)
                            + (in_channels if upsample_mode != blocks.UpsampleMode.CONV_TRANSPOSE else out_channel),
                            out_channels=out_channel,
                            block_configs=[blocks.BlockConfig(3, 1, 1, False, "ReLU;True", blocks.NormMode.BATCH)] * 2,
                            dim=dim,
                        ),
                        in_branch=[f"X_{i}_{j + 1}"],
                        out_branch=[f"X_{i}_{j + 1}"],
                    )

    class UNetPPHead(network.ModuleArgsDict):
        def __init__(self, in_channels: int, out_channels: int, nb_class: int, dim: int) -> None:
            super().__init__()
            self.add_module(
                "Upsample",
                blocks.upsample(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    upsample_mode=blocks.UpsampleMode.UPSAMPLE,
                    dim=dim,
                ),
            )
            self.add_module(
                "ConvBlock",
                blocks.ConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    block_configs=[blocks.BlockConfig(3, 1, 1, False, "ReLU;True", blocks.NormMode.BATCH)] * 2,
                    dim=dim,
                ),
            )
            self.add_module(
                "Conv",
                blocks.get_torch_module("Conv", dim)(
                    in_channels=out_channels,
                    out_channels=nb_class,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
            )
            if nb_class > 1:
                self.add_module("Softmax", torch.nn.Softmax(dim=1))
                self.add_module("Argmax", blocks.ArgMax(dim=1))
            else:
                self.add_module("Tanh", torch.nn.Tanh())

    def __init__(
        self,
        optimizer: network.OptimizerLoader = network.OptimizerLoader(),
        schedulers: dict[str, network.LRSchedulersLoader] = {
            "default|ReduceLROnPlateau": network.LRSchedulersLoader(0)
        },
        outputs_criterions: dict[str, network.TargetCriterionsLoader] = {"default": network.TargetCriterionsLoader()},
        patch: ModelPatch | None = None,
        encoder_channels: list[int] = [1, 64, 64, 128, 256, 512],
        decoder_channels: list[int] = [256, 128, 64, 32, 16, 1],
        layers: list[int] = [3, 4, 6, 3],
        dim: int = 2,
    ) -> None:
        super().__init__(
            in_channels=encoder_channels[0],
            optimizer=optimizer,
            schedulers=schedulers,
            outputs_criterions=outputs_criterions,
            patch=patch,
            dim=dim,
        )
        self.add_module(
            "Block_0",
            UNetpp.UNetPPBlock(
                encoder_channels,
                decoder_channels[::-1],
                UNetpp.resnet_encoder(encoder_channels, layers, dim),
                blocks.UpsampleMode.UPSAMPLE,
                dim=dim,
            ),
            out_branch=[f"X_0_{j + 1}" for j in range(len(encoder_channels) - 2)],
        )
        self.add_module(
            "Head",
            UNetpp.UNetPPHead(
                in_channels=decoder_channels[-3],
                out_channels=decoder_channels[-2],
                nb_class=decoder_channels[-1],
                dim=dim,
            ),
            in_branch=[f"X_0_{len(encoder_channels) - 2}"],
            out_branch=[-1],
        )
