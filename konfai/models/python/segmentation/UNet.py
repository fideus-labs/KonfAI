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


class UNetHead(network.ModuleArgsDict):
    def __init__(self, in_channels: int, nb_class: int, dim: int, level: int) -> None:
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
        self.add_module("Softmax", torch.nn.Softmax(dim=1))
        self.add_module("Argmax", blocks.ArgMax(dim=1))


class UNetBlock(network.ModuleArgsDict):
    def __init__(
        self,
        channels: list[int],
        nb_conv_per_stage: int,
        block_config: blocks.BlockConfig,
        downsample_mode: blocks.DownsampleMode,
        upsample_mode: blocks.UpsampleMode,
        attention: bool,
        block: type,
        nb_class: int,
        dim: int,
        i: int = 0,
    ) -> None:
        super().__init__()
        block_config_stride = block_config
        if i > 0:
            if downsample_mode != blocks.DownsampleMode.CONV_STRIDE:
                self.add_module(
                    downsample_mode.name,
                    blocks.downsample(
                        in_channels=channels[0],
                        out_channels=channels[1],
                        downsample_mode=downsample_mode,
                        dim=dim,
                    ),
                )
            else:
                block_config_stride = blocks.BlockConfig(
                    block_config.kernel_size,
                    2,
                    block_config.padding,
                    block_config.bias,
                    block_config.activation,
                    block_config.norm_mode,
                )
        self.add_module(
            "DownConvBlock",
            block(
                in_channels=channels[0],
                out_channels=channels[1],
                block_configs=[block_config_stride] + [block_config] * (nb_conv_per_stage - 1),
                dim=dim,
            ),
        )
        if len(channels) > 2:
            self.add_module(
                f"UNetBlock_{i + 1}",
                UNetBlock(
                    channels[1:],
                    nb_conv_per_stage,
                    block_config,
                    downsample_mode,
                    upsample_mode,
                    attention,
                    block,
                    nb_class,
                    dim,
                    i + 1,
                ),
            )
            self.add_module(
                "UpConvBlock",
                block(
                    in_channels=(
                        (channels[1] + channels[2])
                        if upsample_mode != blocks.UpsampleMode.CONV_TRANSPOSE
                        else channels[1] * 2
                    ),
                    out_channels=channels[1],
                    block_configs=[block_config] * nb_conv_per_stage,
                    dim=dim,
                ),
            )
            if nb_class > 0:
                self.add_module("Head", UNetHead(channels[1], nb_class, dim, i), out_branch=[-1])
        if i > 0:
            skip_branch = 1
            if attention:
                # Write the gated skip to a branch NOT used inside Attention: an out_branch that
                # matches an internal name (e.g. [1] == W_g's branch) would capture that half-
                # resolution projection instead of the final Multiply output (like DDPM uses [2]).
                self.add_module(
                    "Attention",
                    blocks.Attention(f_g=channels[1], f_l=channels[0], f_int=channels[0], dim=dim),
                    in_branch=[1, 0],
                    out_branch=[2],
                )
                skip_branch = 2
            self.add_module(
                upsample_mode.name,
                blocks.upsample(
                    in_channels=channels[1],
                    out_channels=channels[0],
                    upsample_mode=upsample_mode,
                    dim=dim,
                    kernel_size=2,
                    stride=2,
                ),
            )
            self.add_module("SkipConnection", blocks.Concat(), in_branch=[0, skip_branch])


class UNet(network.Network):
    def __init__(
        self,
        optimizer: network.OptimizerLoader = network.OptimizerLoader(),
        schedulers: dict[str, network.LRSchedulersLoader] = {
            "default|ReduceLROnPlateau": network.LRSchedulersLoader(0)
        },
        outputs_criterions: dict[str, network.TargetCriterionsLoader] = {
            "UNetBlock_0:Head:Argmax": network.TargetCriterionsLoader()
        },
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
            UNetBlock(
                channels,
                nb_conv_per_stage,
                block_config,
                downsample_mode=blocks.DownsampleMode[downsample_mode],
                upsample_mode=blocks.UpsampleMode[upsample_mode],
                attention=attention,
                block=blocks.ConvBlock if block_type == "Conv" else blocks.ResBlock,
                nb_class=nb_class,
                dim=dim,
            ),
        )
