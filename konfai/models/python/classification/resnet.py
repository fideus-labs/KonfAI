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

from abc import ABC

import torch
from konfai.data.patching import ModelPatch
from konfai.network import blocks, network
from konfai.utils.errors import ConfigError

# torchvision checkpoint URLs and the (dim, in_channels, depths, widths, num_classes, use_bottleneck) that reproduce them:
# resnet18:  https://download.pytorch.org/models/resnet18-5c106cde.pth
#   dim=2, in_channels=3, depths=[2, 2, 2, 2], widths=[64, 64, 128, 256, 512], num_classes=1000, use_bottleneck=False
# resnet34:  https://download.pytorch.org/models/resnet34-333f7ec4.pth
#   dim=2, in_channels=3, depths=[3, 4, 6, 3], widths=[64, 64, 128, 256, 512], num_classes=1000, use_bottleneck=False
# resnet50:  https://download.pytorch.org/models/resnet50-19c8e357.pth
#   dim=2, in_channels=3, depths=[3, 4, 6, 3], widths=[64, 256, 512, 1024, 2048], num_classes=1000, use_bottleneck=True
# resnet101: https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
#   dim=2, in_channels=3, depths=[3, 4, 23, 3], widths=[64, 256, 512, 1024, 2048], num_classes=1000, use_bottleneck=True
# resnet152: https://download.pytorch.org/models/resnet152-b121ed2d.pth
#   dim=2, in_channels=3, depths=[3, 8, 36, 3], widths=[64, 256, 512, 1024, 2048], num_classes=1000, use_bottleneck=True
# resnext50_32x4d:  https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
# resnext101_32x8d: https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth
# wide_resnet50_2:  https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth
# wide_resnet101_2: https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth


class AbstractResBlock(network.ModuleArgsDict, ABC):
    def __init__(self, in_channels: int, out_channels: int, downsample: bool, dim: int):
        super().__init__()


class ResBlock(AbstractResBlock):
    def __init__(self, in_channels: int, out_channels: int, downsample: bool, dim: int):
        super().__init__(in_channels, out_channels, downsample, dim)
        # As in torchvision BasicBlock: project the shortcut whenever the block is strided OR
        # changes channel count, otherwise the residual Add receives mismatched tensors.
        if downsample or in_channels != out_channels:
            self.add_module(
                "Shortcut",
                blocks.ConvBlock(
                    in_channels,
                    out_channels,
                    [
                        blocks.BlockConfig(
                            kernel_size=1,
                            stride=2 if downsample else 1,
                            padding=0,
                            bias=False,
                            activation="None",
                            norm_mode=blocks.NormMode.BATCH.name,
                        )
                    ],
                    dim=dim,
                    alias=[["0"], ["1"], []],
                ),
                in_branch=[1],
                out_branch=[1],
                alias=["downsample"],
            )

        self.add_module(
            "ConvBlock_0",
            blocks.ConvBlock(
                in_channels,
                out_channels,
                [
                    blocks.BlockConfig(
                        kernel_size=3,
                        stride=2 if downsample else 1,
                        padding=1,
                        bias=False,
                        activation="ReLU",
                        norm_mode=blocks.NormMode.BATCH.name,
                    )
                ],
                dim=dim,
                alias=[["conv1"], ["bn1"], []],
            ),
        )
        self.add_module(
            "ConvBlock_1",
            blocks.ConvBlock(
                out_channels,
                out_channels,
                [
                    blocks.BlockConfig(
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False,
                        activation="None",
                        norm_mode=blocks.NormMode.BATCH.name,
                    )
                ],
                dim=dim,
                alias=[["conv2"], ["bn2"], []],
            ),
        )
        self.add_module("Residual", blocks.Add(), in_branch=[0, 1])
        self.add_module("ReLU", torch.nn.ReLU())


class ResBottleneckBlock(AbstractResBlock):
    def __init__(self, in_channels: int, out_channels: int, downsample: bool, dim: int):
        super().__init__(in_channels, out_channels, downsample, dim)
        self.add_module(
            "ConvBlock_0",
            blocks.ConvBlock(
                in_channels,
                out_channels // 4,
                [
                    blocks.BlockConfig(
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=False,
                        activation="ReLU",
                        norm_mode=blocks.NormMode.BATCH.name,
                    )
                ],
                dim=dim,
                alias=[["conv1"], ["bn1"], []],
            ),
        )
        self.add_module(
            "ConvBlock_1",
            blocks.ConvBlock(
                out_channels // 4,
                out_channels // 4,
                [
                    blocks.BlockConfig(
                        kernel_size=3,
                        stride=2 if downsample else 1,
                        padding=1,
                        bias=False,
                        activation="ReLU",
                        norm_mode=blocks.NormMode.BATCH.name,
                    )
                ],
                dim=dim,
                alias=[["conv2"], ["bn2"], []],
            ),
        )
        self.add_module(
            "ConvBlock_2",
            blocks.ConvBlock(
                out_channels // 4,
                out_channels,
                [
                    blocks.BlockConfig(
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=False,
                        # No activation on conv3/bn3 before the residual add (torchvision
                        # Bottleneck); a ReLU here diverges from the pretrained architecture.
                        activation="None",
                        norm_mode=blocks.NormMode.BATCH.name,
                    )
                ],
                dim=dim,
                alias=[["conv3"], ["bn3"], []],
            ),
        )

        if downsample or in_channels != out_channels:
            self.add_module(
                "Shortcut",
                blocks.ConvBlock(
                    in_channels,
                    out_channels,
                    [
                        blocks.BlockConfig(
                            kernel_size=1,
                            stride=2 if downsample else 1,
                            padding=0,
                            bias=False,
                            activation="None",
                            norm_mode=blocks.NormMode.BATCH.name,
                        )
                    ],
                    dim=dim,
                    alias=[["0"], ["1"], []],
                ),
                in_branch=[1],
                out_branch=[1],
                alias=["downsample"],
            )

        self.add_module("Residual", blocks.Add(), in_branch=[0, 1])
        self.add_module("ReLU", torch.nn.ReLU())


class ResNetStage(network.ModuleArgsDict):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
        block: type[AbstractResBlock],
        downsample: bool,
        dim: int,
    ):
        super().__init__()
        self.add_module(
            "BottleNeckBlock_0",
            block(
                in_channels=in_channels,
                out_channels=out_channels,
                downsample=downsample,
                dim=dim,
            ),
            alias=["0"],
        )
        for i in range(1, depth):
            self.add_module(
                f"BottleNeckBlock_{i}",
                block(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    downsample=False,
                    dim=dim,
                ),
                alias=[f"{i}"],
            )


class ResNetStem(network.ModuleArgsDict):
    def __init__(self, in_channels: int, out_features: int, dim: int):
        super().__init__()
        self.add_module(
            "ConvBlock",
            blocks.ConvBlock(
                in_channels,
                out_features,
                [
                    blocks.BlockConfig(
                        kernel_size=7,
                        stride=2,
                        padding=3,
                        bias=False,
                        activation="ReLU",
                        norm_mode=blocks.NormMode.BATCH.name,
                    )
                ],
                dim=dim,
                alias=[["conv1"], ["bn1"], []],
            ),
        )
        self.add_module(
            "MaxPool",
            blocks.get_torch_module("MaxPool", dim)(kernel_size=3, stride=2, padding=1),
        )


class ResNetEncoder(network.ModuleArgsDict):
    def __init__(
        self,
        in_channels: int,
        depths: list[int],
        widths: list[int],
        use_bottleneck: bool,
        dim: int,
    ):
        super().__init__()
        if len(widths) != len(depths) + 1:
            raise ConfigError(
                f"ResNet expects len(widths) == len(depths) + 1 (stem width + one width per stage), "
                f"got {len(widths)} widths for {len(depths)} depths.",
            )
        if any(depth < 1 for depth in depths):
            raise ConfigError(f"ResNet stage depths must all be >= 1, got {depths}.")
        self.add_module("ResNetStem", ResNetStem(in_channels, widths[0], dim=dim))

        for i, (in_channels, out_channels, depth) in enumerate(list(zip(widths[:], widths[1:], depths, strict=False))):
            self.add_module(
                f"ResNetStage_{i}",
                ResNetStage(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    depth=depth,
                    block=ResBottleneckBlock if use_bottleneck else ResBlock,
                    downsample=i != 0,
                    dim=dim,
                ),
                alias=[f"layer{i + 1}"],
            )


class Head(network.ModuleArgsDict):
    def __init__(self, in_features: int, num_classes: int, dim: int):
        super().__init__()
        self.add_module("AdaptiveAvgPool", blocks.get_torch_module("AdaptiveAvgPool", dim)(1))
        self.add_module("Flatten", torch.nn.Flatten(1))
        self.add_module(
            "Linear",
            torch.nn.Linear(in_features, num_classes),
            pretrained=False,
            alias=["fc"],
        )
        self.add_module("Unsqueeze", blocks.Unsqueeze(2))


class ResNet(network.Network):
    def __init__(
        self,
        optimizer: network.OptimizerLoader = network.OptimizerLoader(),
        schedulers: dict[str, network.LRSchedulersLoader] = {
            "default|ReduceLROnPlateau": network.LRSchedulersLoader(0)
        },
        outputs_criterions: dict[str, network.TargetCriterionsLoader] = {"default": network.TargetCriterionsLoader()},
        patch: ModelPatch = ModelPatch(),
        dim: int = 3,
        in_channels: int = 1,
        depths: list[int] = [2, 2, 2, 2],
        widths: list[int] = [64, 64, 128, 256, 512],
        num_classes: int = 10,
        use_bottleneck=False,
    ):
        super().__init__(
            in_channels=in_channels,
            optimizer=optimizer,
            schedulers=schedulers,
            outputs_criterions=outputs_criterions,
            dim=dim,
            patch=patch,
            init_type="trunc_normal",
            init_gain=0.02,
        )
        self.add_module(
            "ResNetEncoder",
            ResNetEncoder(
                in_channels=in_channels,
                depths=depths,
                widths=widths,
                use_bottleneck=use_bottleneck,
                dim=dim,
            ),
        )
        self.add_module("Head", Head(in_features=widths[-1], num_classes=num_classes, dim=dim))
