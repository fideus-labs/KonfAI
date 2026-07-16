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
from konfai.network import blocks, network


class VAE(network.Network):
    """Convolutional encoder-decoder (deterministic autoencoder).

    Note: this network has no latent sampling — it is a plain autoencoder, not a
    variational one. For a reparameterised latent (mu/log_std/z + KL-ready named
    outputs) use ``blocks.LatentDistribution`` as a bottleneck, as ``LinearVAE``
    demonstrates.
    """

    class AutoEncoderBlock(network.ModuleArgsDict):
        def __init__(
            self,
            channels: list[int],
            nb_conv_per_stage: int,
            block_config: blocks.BlockConfig,
            downsample_mode: blocks.DownsampleMode,
            upsample_mode: blocks.UpsampleMode,
            dim: int,
            block: type,
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
                "DownBlock",
                block(
                    in_channels=channels[(1 if downsample_mode == blocks.DownsampleMode.CONV_STRIDE and i > 0 else 0)],
                    out_channels=channels[1],
                    block_configs=[block_config] * nb_conv_per_stage,
                    dim=dim,
                ),
            )
            if len(channels) > 2:
                self.add_module(
                    f"AutoEncoder_{i + 1}",
                    VAE.AutoEncoderBlock(
                        channels[1:],
                        nb_conv_per_stage,
                        block_config,
                        downsample_mode,
                        upsample_mode,
                        dim,
                        block,
                        i + 1,
                    ),
                )
                self.add_module(
                    "UpBlock",
                    block(
                        in_channels=(
                            channels[2] if upsample_mode != blocks.UpsampleMode.CONV_TRANSPOSE else channels[1]
                        ),
                        out_channels=channels[1],
                        block_configs=[block_config] * nb_conv_per_stage,
                        dim=dim,
                    ),
                )
            if i > 0:
                self.add_module(
                    upsample_mode.name,
                    blocks.upsample(
                        in_channels=channels[1],
                        out_channels=channels[0],
                        upsample_mode=upsample_mode,
                        dim=dim,
                    ),
                )

    class VAEHead(network.ModuleArgsDict):
        def __init__(self, in_channels: int, out_channels: int, dim: int) -> None:
            super().__init__()
            self.add_module(
                "Conv",
                blocks.get_torch_module("Conv", dim)(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
            )
            self.add_module("Tanh", torch.nn.Tanh())

    def __init__(
        self,
        optimizer: network.OptimizerLoader = network.OptimizerLoader(),
        schedulers: dict[str, network.LRSchedulersLoader] = {
            "default|ReduceLROnPlateau": network.LRSchedulersLoader(0)
        },
        outputs_criterions: dict[str, network.TargetCriterionsLoader] = {"default": network.TargetCriterionsLoader()},
        dim: int = 3,
        channels: list[int] = [1, 64, 128, 256, 512, 1024],
        block_config: blocks.BlockConfig = blocks.BlockConfig(),
        nb_conv_per_stage: int = 2,
        downsample_mode: str = "MAXPOOL",
        upsample_mode: str = "CONV_TRANSPOSE",
        block_type: Literal["Conv", "Res"] = "Conv",
    ) -> None:

        super().__init__(
            in_channels=channels[0],
            init_type="normal",
            optimizer=optimizer,
            schedulers=schedulers,
            outputs_criterions=outputs_criterions,
            dim=dim,
            nb_batch_per_step=1,
        )
        self.add_module(
            "AutoEncoder_0",
            VAE.AutoEncoderBlock(
                channels,
                nb_conv_per_stage,
                block_config,
                downsample_mode=blocks.DownsampleMode[downsample_mode],
                upsample_mode=blocks.UpsampleMode[upsample_mode],
                dim=dim,
                block=blocks.ConvBlock if block_type == "Conv" else blocks.ResBlock,
            ),
        )
        self.add_module("Head", VAE.VAEHead(channels[1], channels[0], dim))


class LinearVAE(network.Network):
    """Fully-connected variational autoencoder for flat feature vectors.

    Encoder → variational bottleneck (``LatentDistribution``: reparameterised
    sampling with addressable ``Latent.mu`` / ``Latent.log_std`` outputs for the
    KL term) → decoder. Dimensions are parameterised; the variational sampling
    is provided by ``LatentDistribution``.
    """

    class LinearVAEDenseLayer(network.ModuleArgsDict):
        def __init__(self, in_features: int, out_features: int) -> None:
            super().__init__()
            self.add_module("Linear", torch.nn.Linear(in_features, out_features))
            self.add_module("Activation", torch.nn.LeakyReLU())

    class LinearVAEHead(network.ModuleArgsDict):
        def __init__(self, in_features: int, out_features: int) -> None:
            super().__init__()
            self.add_module("Linear", torch.nn.Linear(in_features, out_features))
            self.add_module("Tanh", torch.nn.Tanh())

    def __init__(
        self,
        optimizer: network.OptimizerLoader = network.OptimizerLoader(),
        schedulers: dict[str, network.LRSchedulersLoader] = {
            "default|ReduceLROnPlateau": network.LRSchedulersLoader(0)
        },
        outputs_criterions: dict[str, network.TargetCriterionsLoader] = {"default": network.TargetCriterionsLoader()},
        in_features: int = 784,
        hidden_features: int = 256,
        latent_dim: int = 32,
    ) -> None:
        super().__init__(
            in_channels=1,
            init_type="normal",
            optimizer=optimizer,
            schedulers=schedulers,
            outputs_criterions=outputs_criterions,
            dim=1,
            nb_batch_per_step=1,
        )
        self.add_module("Encoder", LinearVAE.LinearVAEDenseLayer(in_features, hidden_features))
        self.add_module("Latent", blocks.LatentDistribution(shape=[hidden_features], latent_dim=latent_dim))
        self.add_module("Head", LinearVAE.LinearVAEHead(hidden_features, in_features))
