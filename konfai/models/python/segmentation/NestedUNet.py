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

from typing import Literal

import torch
from konfai.data.patching import ModelPatch
from konfai.network import blocks, network
from konfai.utils.errors import ConfigError


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
        block_type: Literal["Conv", "Res"] = "Conv",
        activation: str = "Softmax",
    ) -> None:
        if attention:
            # The flag reaches every nested block and no block reads it: asking for attention gates
            # here builds the plain model and says nothing, so ask for a model that exists instead.
            raise ConfigError("NestedUNet has no attention gates; set 'attention: false'.")
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
