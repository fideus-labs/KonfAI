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

import importlib
import itertools

import torch
from konfai.data.patching import ModelPatch
from konfai.network import blocks, network
from konfai.utils.errors import ConfigError


class MappingNetwork(network.ModuleArgsDict):
    def __init__(
        self,
        z_dim: int,
        c_dim: int,
        w_dim: int,
        num_layers: int,
        embed_features: int,
        layer_features: int,
    ):
        super().__init__()

        self.add_module("Concat_1", blocks.Concat(), in_branch=[0, 1])

        features = [z_dim + embed_features if c_dim > 0 else 0] + [layer_features] * (num_layers - 1) + [w_dim]
        if c_dim > 0:
            self.add_module("Linear", torch.nn.Linear(c_dim, embed_features), out_branch=["Embed"])

        self.add_module("Noise", blocks.NormalNoise(z_dim), in_branch=["Embed"])
        if c_dim > 0:
            self.add_module("Concat", blocks.Concat(), in_branch=[0, "Embed"])

        for i, (in_features, out_features) in enumerate(itertools.pairwise(features)):
            self.add_module(f"Linear_{i}", torch.nn.Linear(in_features, out_features))


class ModulatedConv(torch.nn.Module):
    class _ModulatedConv(torch.nn.Module):
        def __init__(self, w_dim: int, conv: torch.nn.modules.conv._ConvNd, dim: int) -> None:
            super().__init__()
            self.affine = torch.nn.Linear(w_dim, conv.in_channels)
            self.isConv = True
            self.in_channels = conv.in_channels
            self.out_channels = conv.out_channels
            self.padding = conv.padding
            self.stride = conv.stride
            if isinstance(conv, torch.nn.modules.conv._ConvTransposeNd):
                self.weight = torch.nn.parameter.Parameter(
                    torch.randn((conv.in_channels, conv.out_channels, *conv.kernel_size))
                )
                self.isConv = False
            else:
                self.weight = torch.nn.parameter.Parameter(
                    torch.randn((conv.out_channels, conv.in_channels, *conv.kernel_size))
                )
            conv.forward = self.forward
            self.styles = None
            self.dim = dim

        def set_style(self, styles: torch.Tensor) -> None:
            self.styles = styles

        def forward(self, tensor: torch.Tensor) -> torch.Tensor:
            b = tensor.shape[0]
            self.affine.to(tensor.device)
            styles = self.affine(self.styles)
            w1 = (
                styles.reshape(b, -1, 1, *[1 for _ in range(self.dim)])
                if not self.isConv
                else styles.reshape(b, 1, -1, *[1 for _ in range(self.dim)])
            )
            w2 = self.weight.unsqueeze(0).to(tensor.device)
            weights = w2 * (w1 + 1)

            d = torch.rsqrt(
                (weights**2).sum(
                    dim=tuple([i + 2 for i in range(len(weights.shape) - 2)]),
                    keepdim=True,
                )
                + 1e-8
            )
            weights = weights * d

            tensor = tensor.reshape(1, -1, *tensor.shape[2:])

            _, _, *ws = weights.shape
            if not self.isConv:
                out = getattr(
                    importlib.import_module("torch.nn.functional"),
                    f"conv_transpose{self.dim}d",
                )(
                    tensor,
                    weights.reshape(b * self.in_channels, *ws),
                    stride=self.stride,
                    padding=self.padding,
                    groups=b,
                )
            else:
                out = getattr(
                    importlib.import_module("torch.nn.functional"),
                    f"conv{self.dim}d",
                )(
                    tensor,
                    weights.reshape(b * self.out_channels, *ws),
                    padding=self.padding,
                    groups=b,
                    stride=self.stride,
                )

            out = out.reshape(-1, self.out_channels, *out.shape[2:])
            return out

    def __init__(self, w_dim: int, module: torch.nn.Module) -> None:
        super().__init__()
        self.w_dim = w_dim
        self.module = module
        self.convs = torch.nn.ModuleList()
        self.module.apply(self.apply)

    def forward(self, tensor: torch.Tensor, styles: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            conv.set_style(styles.clone())
        return self.module(tensor)

    def apply(self, module: torch.nn.Module):
        if isinstance(module, torch.nn.modules.conv._ConvNd):
            del module.weight
            module.bias = None

            str_dim = module.__class__.__name__[-2:]
            dim = 1
            if str_dim == "2d":
                dim = 2
            elif str_dim == "3d":
                dim = 3
            self.convs.append(ModulatedConv._ModulatedConv(self.w_dim, module, dim=dim))


class UNetBlock(network.ModuleArgsDict):
    def __init__(
        self,
        w_dim: int,
        channels: list[int],
        nb_conv_per_stage: int,
        block_config: blocks.BlockConfig,
        downsample_mode: blocks.DownsampleMode,
        upsample_mode: blocks.UpsampleMode,
        attention: bool,
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
            "DownConvBlock",
            blocks.ConvBlock(
                in_channels=channels[(1 if downsample_mode == blocks.DownsampleMode.CONV_STRIDE and i > 0 else 0)],
                out_channels=channels[1],
                block_configs=[block_config] * nb_conv_per_stage,
                dim=dim,
            ),
        )
        if len(channels) > 2:
            self.add_module(
                f"UNetBlock_{i + 1}",
                UNetBlock(
                    w_dim,
                    channels[1:],
                    nb_conv_per_stage,
                    block_config,
                    downsample_mode,
                    upsample_mode,
                    attention,
                    dim,
                    i + 1,
                ),
                in_branch=[0, 1],
            )
            self.add_module(
                "UpConvBlock",
                ModulatedConv(
                    w_dim,
                    blocks.ConvBlock(
                        (
                            (channels[1] + channels[2])
                            if upsample_mode != blocks.UpsampleMode.CONV_TRANSPOSE
                            else channels[1] * 2
                        ),
                        out_channels=channels[1],
                        block_configs=[block_config] * nb_conv_per_stage,
                        dim=dim,
                    ),
                ),
                in_branch=[0, 1],
            )
        if i > 0:
            if attention:
                self.add_module(
                    "Attention",
                    blocks.Attention(f_g=channels[1], f_l=channels[0], f_int=channels[0], dim=dim),
                    in_branch=["Skip", 0],
                    out_branch=["Skip"],
                )
            self.add_module(
                upsample_mode.name,
                ModulatedConv(
                    w_dim,
                    blocks.upsample(
                        in_channels=channels[1],
                        out_channels=channels[0],
                        upsample_mode=upsample_mode,
                        dim=dim,
                    ),
                ),
                in_branch=[0, 1],
            )
            self.add_module("SkipConnection", blocks.Concat(), in_branch=[0, "Skip"])


class Generator(network.Network):
    class GeneratorHead(network.ModuleArgsDict):
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
        patch: ModelPatch = ModelPatch(),
        outputs_criterions: dict[str, network.TargetCriterionsLoader] = {"default": network.TargetCriterionsLoader()},
        channels: list[int] = [1, 64, 128, 256, 512, 1024],
        nb_batch_per_step: int = 64,
        z_dim: int = 512,
        c_dim: int = 1,
        w_dim: int = 512,
        dim: int = 3,
    ) -> None:
        if c_dim < 1:
            # The mapping network is conditional-only: with c_dim=0 the 'Embed' branch is never
            # produced, the first projection gets in_features=0 and the forward still expects a
            # conditioning input. Reject instead of building a graph guaranteed to crash.
            raise ConfigError(f"cStyleGan.Generator is conditional-only and requires c_dim >= 1, got c_dim={c_dim}.")
        super().__init__(
            optimizer=optimizer,
            in_channels=channels[0],
            schedulers=schedulers,
            patch=patch,
            outputs_criterions=outputs_criterions,
            dim=dim,
            nb_batch_per_step=nb_batch_per_step,
        )

        self.add_module(
            "MappingNetwork",
            MappingNetwork(
                z_dim=z_dim,
                c_dim=c_dim,
                w_dim=w_dim,
                num_layers=8,
                embed_features=w_dim,
                layer_features=w_dim,
            ),
            in_branch=[1, 2],
            out_branch=["Style"],
        )
        nb_conv_per_stage = 2
        block_config = blocks.BlockConfig(
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            activation="ReLU",
            norm_mode="INSTANCE",
        )
        self.add_module(
            "UNetBlock_0",
            UNetBlock(
                w_dim,
                channels,
                nb_conv_per_stage,
                block_config,
                downsample_mode=blocks.DownsampleMode.MAXPOOL,
                upsample_mode=blocks.UpsampleMode.CONV_TRANSPOSE,
                attention=False,
                dim=dim,
            ),
            in_branch=[0, "Style"],
        )
        self.add_module(
            "Head",
            Generator.GeneratorHead(in_channels=channels[1], out_channels=1, dim=dim),
        )
