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
from typing import cast

import numpy as np
import torch
from konfai.data import augmentation
from konfai.data.patching import Attribute, ModelPatch
from konfai.models.python.generation.ddpm import DDPM
from konfai.models.python.segmentation import NestedUNet, UNet
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

    class DiscriminatorBlock(network.ModuleArgsDict):
        def __init__(
            self,
            channels: list[int] = [1, 16, 32, 64, 64],
            strides: list[int] = [2, 2, 2, 1],
            dim: int = 3,
        ) -> None:
            super().__init__()
            self.add_module("Layers", Discriminator.DiscriminatorNLayers(channels, strides, dim))
            self.add_module("Head", Discriminator.DiscriminatorHead(channels[-1], dim))

    def __init__(
        self,
        optimizer: network.OptimizerLoader = network.OptimizerLoader(),
        schedulers: dict[str, network.LRSchedulersLoader] = {
            "default|ReduceLROnPlateau": network.LRSchedulersLoader(0)
        },
        outputs_criterions: dict[str, network.TargetCriterionsLoader] = {"default": network.TargetCriterionsLoader()},
        channels: list[int] = [1, 16, 32, 64, 64],
        strides: list[int] = [2, 2, 2, 1],
        nb_batch_per_step: int = 1,
        dim: int = 3,
    ) -> None:
        super().__init__(
            in_channels=1,
            optimizer=optimizer,
            schedulers=schedulers,
            outputs_criterions=outputs_criterions,
            nb_batch_per_step=nb_batch_per_step,
            dim=dim,
            init_type="kaiming",
        )
        self.add_module(
            "DiscriminatorModel",
            Discriminator.DiscriminatorBlock(channels, strides, dim),
        )


class DiscriminatorADA(network.Network):
    class DDPMTE(torch.nn.Module):
        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            self.linear_0 = torch.nn.Linear(in_channels, out_channels)
            self.siLU = torch.nn.SiLU()
            self.linear_1 = torch.nn.Linear(out_channels, out_channels)

        def forward(self, tensor: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return tensor + self.linear_1(self.siLU(self.linear_0(t))).reshape(
                tensor.shape[0], -1, *[1 for _ in range(len(tensor.shape) - 2)]
            )

    class DiscriminatorNLayers(network.ModuleArgsDict):
        def __init__(
            self,
            channels: list[int],
            strides: list[int],
            time_embedding_dim: int,
            dim: int,
        ) -> None:
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
                    f"Te_{i}",
                    DiscriminatorADA.DDPMTE(time_embedding_dim, in_channels),
                    in_branch=[0, 1],
                )
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

    class UpdateP(torch.nn.Module):
        """Adaptive-augmentation (ADA) probability controller.

        `p` and `_it` are intentional running training state: every `n` calls the
        augmentation probability `p` is nudged toward `ada_target` based on the
        discriminator measure. They are deliberately kept as plain Python scalars
        (not registered buffers) so the checkpoint format stays compatible with
        existing pretrained weights. `_it` only matters modulo `n`, so it is never
        reset; `p` accumulates by design.
        """

        def __init__(self):
            super().__init__()
            self._it = 0
            self.n = 4
            self.ada_target = 0.25
            self.ada_interval = 0.001
            self.ada_kimg = 500

            self.measure = None
            self.names = []
            self.p = 0

        def set_measure(self, measure: network.Measure, names: list[str]):
            self.measure = measure
            self.names = names

        def forward(self, tensor: torch.Tensor) -> torch.Tensor:
            if self.measure is not None and self._it % self.n == 0:
                value = sum([v for k, v in self.measure.get_last_values(self.n).items() if k in self.names])
                adjust = np.sign(self.ada_target - value) * (self.ada_interval)
                self.p += adjust
                self.p = np.clip(self.p, 0, 1)
            self._it += 1
            return torch.tensor(self.p).to(tensor.device)

    class DiscriminatorAugmentation(torch.nn.Module):
        def __init__(self, dim: int):
            super().__init__()

            self.data_augmentations: dict[augmentation.DataAugmentation, float] = {}
            pixel_blitting = {
                augmentation.Flip([1 / 3] * 3 if dim == 3 else [1 / 2] * 2): 0,
                augmentation.Rotate(a_min=0, a_max=360, is_quarter=True): 0,
                augmentation.Translate(-5, 5, is_int=True): 0,
            }

            self.data_augmentations.update(
                {cast(augmentation.DataAugmentation, k): v for k, v in pixel_blitting.items()}
            )
            geometric = {
                augmentation.Scale(0.2): 0,
                augmentation.Rotate(a_min=0, a_max=360): 0,
                augmentation.Scale(0.2): 0,
                augmentation.Rotate(a_min=0, a_max=360): 0,
                augmentation.Translate(-5, 5): 0,
                augmentation.Elastix(16, 16): 0.5,
            }
            self.data_augmentations.update({cast(augmentation.DataAugmentation, k): v for k, v in geometric.items()})
            color = {
                augmentation.Brightness(0.2): 0.0,
                augmentation.Contrast(0.5): 0.0,
                augmentation.Saturation(1): 0.0,
                augmentation.HUE(1): 0.0,
                augmentation.LumaFlip(): 0.0,
            }
            self.data_augmentations.update({cast(augmentation.DataAugmentation, k): v for k, v in color.items()})

            corruptions = {
                augmentation.Noise(1): 1,
                augmentation.CutOUT(0.5, 1, -1): 0.3,
            }
            self.data_augmentations.update({cast(augmentation.DataAugmentation, k): v for k, v in corruptions.items()})

        def _set_p(self, prob: float):
            for aug, p in self.data_augmentations.items():
                aug.load(prob * p)

        def forward(self, tensor: torch.Tensor, prob: torch.Tensor) -> torch.Tensor:
            self._set_p(prob.item())
            out = tensor
            for aug in self.data_augmentations.keys():
                aug.state_init(
                    None,
                    [tensor.shape[2:]] * tensor.shape[0],
                    [Attribute()] * tensor.shape[0],
                )
                out = aug("", 0, list(out))
            return torch.cat([data.unsqueeze(0) for data in out], 0)

    class DiscriminatorBlock(network.ModuleArgsDict):
        def __init__(
            self,
            channels: list[int] = [1, 16, 32, 64, 64],
            strides: list[int] = [2, 2, 2, 1],
            dim: int = 3,
        ) -> None:
            super().__init__()
            self.add_module("Prob", DiscriminatorADA.UpdateP(), out_branch=["p"])
            self.add_module(
                "Sample",
                DiscriminatorADA.DiscriminatorAugmentation(dim),
                in_branch=[0, "p"],
            )
            self.add_module(
                "t",
                DDPM.DDPMTimeEmbedding(1000, 100),
                in_branch=[0, "p"],
                out_branch=["te"],
            )
            self.add_module(
                "Layers",
                DiscriminatorADA.DiscriminatorNLayers(channels, strides, 100, dim),
                in_branch=[0, "te"],
            )
            self.add_module("Head", DiscriminatorADA.DiscriminatorHead(channels[-1], dim))

    def __init__(
        self,
        optimizer: network.OptimizerLoader = network.OptimizerLoader(),
        schedulers: dict[str, network.LRSchedulersLoader] = {
            "default|ReduceLROnPlateau": network.LRSchedulersLoader(0)
        },
        outputs_criterions: dict[str, network.TargetCriterionsLoader] = {"default": network.TargetCriterionsLoader()},
        channels: list[int] = [1, 16, 32, 64, 64],
        strides: list[int] = [2, 2, 2, 1],
        nb_batch_per_step: int = 1,
        dim: int = 3,
    ) -> None:
        super().__init__(
            in_channels=1,
            optimizer=optimizer,
            schedulers=schedulers,
            outputs_criterions=outputs_criterions,
            nb_batch_per_step=nb_batch_per_step,
            dim=dim,
            init_type="kaiming",
        )
        self.add_module(
            "DiscriminatorModel",
            DiscriminatorADA.DiscriminatorBlock(channels, strides, dim),
        )

    def initialized(self):
        self["DiscriminatorModel"]["Prob"].set_measure(
            self.measure,
            ["Discriminator_B.DiscriminatorModel.Head.Conv:None:PatchGanLoss"],
        )


class GeneratorV1(network.Network):
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
                    block_configs=[
                        blocks.BlockConfig(
                            stride=2,
                            bias=False,
                            activation="ReLU",
                            norm_mode="SYNCBATCH",
                        )
                    ],
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
                    GeneratorV1.GeneratorDownSample(in_channels=in_channels, out_channels=out_channels, dim=dim),
                )

    class GeneratorResnetBlock(network.ModuleArgsDict):
        def __init__(self, channels: int, dim: int):
            super().__init__()
            self.add_module(
                "Conv_0",
                blocks.get_torch_module("Conv", dim)(channels, channels, kernel_size=3, padding=1, bias=False),
            )
            self.add_module("Norm_0", torch.nn.SyncBatchNorm(channels))
            self.add_module("Activation_0", torch.nn.LeakyReLU(0.2, inplace=True))

            self.add_module(
                "Conv_1",
                blocks.get_torch_module("Conv", dim)(channels, channels, kernel_size=3, padding=1, bias=False),
            )
            self.add_module("Norm_1", torch.nn.SyncBatchNorm(channels))
            self.add_module("Residual", blocks.Add(), in_branch=[0, 1])

    class GeneratorNResnetBlock(network.ModuleArgsDict):
        def __init__(self, channels: int, nb_conv: int, dim: int) -> None:
            super().__init__()
            for i in range(nb_conv):
                self.add_module(
                    f"ResnetBlock_{i}",
                    GeneratorV1.GeneratorResnetBlock(channels=channels, dim=dim),
                )

    class GeneratorDecoder(network.ModuleArgsDict):
        def __init__(self, channels: list[int], dim: int) -> None:
            super().__init__()
            for i, (in_channels, out_channels) in enumerate(
                zip(reversed(channels), reversed(channels[:-1]), strict=False)
            ):
                self.add_module(
                    f"UpSample_{i}",
                    GeneratorV1.GeneratorUpSample(in_channels=in_channels, out_channels=out_channels, dim=dim),
                )

    class GeneratorAutoEncoder(network.ModuleArgsDict):
        def __init__(self, ngf: int, dim: int) -> None:
            super().__init__()
            channels = [ngf, ngf * 2]
            self.add_module("Encoder", GeneratorV1.GeneratorEncoder(channels, dim))
            self.add_module(
                "NResBlock",
                GeneratorV1.GeneratorNResnetBlock(channels=channels[-1], nb_conv=6, dim=dim),
            )
            self.add_module("Decoder", GeneratorV1.GeneratorDecoder(channels, dim))

    class GeneratorBlock(network.ModuleArgsDict):
        def __init__(self, ngf: int, dim: int) -> None:
            super().__init__()
            self.add_module("Stem", GeneratorV1.GeneratorStem(3, ngf, dim))
            self.add_module("AutoEncoder", GeneratorV1.GeneratorAutoEncoder(ngf, dim))
            self.add_module(
                "Head",
                GeneratorV1.GeneratorHead(in_channels=ngf, out_channels=1, dim=dim),
            )

    def __init__(
        self,
        optimizer: network.OptimizerLoader = network.OptimizerLoader(),
        schedulers: dict[str, network.LRSchedulersLoader] = {
            "default|ReduceLROnPlateau": network.LRSchedulersLoader(0)
        },
        patch: ModelPatch = ModelPatch(),
        outputs_criterions: dict[str, network.TargetCriterionsLoader] = {"default": network.TargetCriterionsLoader()},
        dim: int = 3,
    ) -> None:
        super().__init__(
            optimizer=optimizer,
            in_channels=3,
            schedulers=schedulers,
            patch=patch,
            outputs_criterions=outputs_criterions,
            dim=dim,
        )
        self.add_module("GeneratorModel", GeneratorV1.GeneratorBlock(32, dim))


class GeneratorV2(network.Network):
    class NestedUNetHead(network.ModuleArgsDict):
        def __init__(self, in_channels: list[int], dim: int) -> None:
            super().__init__()
            self.add_module(
                "Conv",
                blocks.get_torch_module("Conv", dim)(
                    in_channels=in_channels[1],
                    out_channels=1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ),
            )
            self.add_module("Tanh", torch.nn.Tanh())

    class GeneratorBlock(network.ModuleArgsDict):
        def __init__(
            self,
            channels: list[int],
            block_config: blocks.BlockConfig,
            nb_conv_per_stage: int,
            downsample_mode: str,
            upsample_mode: str,
            attention: bool,
            block_type: str,
            dim: int,
        ) -> None:
            super().__init__()
            self.add_module(
                "UNetBlock_0",
                NestedUNet.NestedUNet.NestedUNetBlock(
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
            self.add_module(
                "Head",
                GeneratorV2.NestedUNetHead(channels[:2], dim=dim),
                in_branch=[f"X_0_{len(channels) - 2}"],
            )

    def __init__(
        self,
        optimizer: network.OptimizerLoader = network.OptimizerLoader(),
        schedulers: dict[str, network.LRSchedulersLoader] = {
            "default|ReduceLROnPlateau": network.LRSchedulersLoader(0)
        },
        outputs_criterions: dict[str, network.TargetCriterionsLoader] = {"default": network.TargetCriterionsLoader()},
        patch: ModelPatch | None = None,
        channels: list[int] = [1, 64, 128, 256, 512, 1024],
        block_config: blocks.BlockConfig = blocks.BlockConfig(),
        nb_conv_per_stage: int = 2,
        downsample_mode: str = "MAXPOOL",
        upsample_mode: str = "CONV_TRANSPOSE",
        attention: bool = False,
        block_type: str = "Conv",
        dim: int = 3,
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
            "GeneratorModel",
            GeneratorV2.GeneratorBlock(
                channels,
                block_config,
                nb_conv_per_stage,
                downsample_mode,
                upsample_mode,
                attention,
                block_type,
                dim,
            ),
        )


class GeneratorV3(network.Network):
    class NestedUNetHead(network.ModuleArgsDict):
        def __init__(self, in_channels: list[int], dim: int) -> None:
            super().__init__()
            self.add_module(
                "Conv",
                blocks.get_torch_module("Conv", dim)(
                    in_channels=in_channels[1],
                    out_channels=1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ),
            )
            self.add_module("Tanh", torch.nn.Tanh())

    class GeneratorBlock(network.ModuleArgsDict):
        def __init__(
            self,
            channels: list[int],
            block_config: blocks.BlockConfig,
            nb_conv_per_stage: int,
            downsample_mode: str,
            upsample_mode: str,
            attention: bool,
            block_type: str,
            dim: int,
        ) -> None:
            super().__init__()
            self.add_module(
                "UNetBlock_0",
                UNet.UNetBlock(
                    channels,
                    nb_conv_per_stage,
                    block_config,
                    downsample_mode=blocks.DownsampleMode[downsample_mode],
                    upsample_mode=blocks.UpsampleMode[upsample_mode],
                    attention=attention,
                    block=blocks.ConvBlock if block_type == "Conv" else blocks.ResBlock,
                    nb_class=1,
                    dim=dim,
                ),
                out_branch=[f"X_0_{j + 1}" for j in range(len(channels) - 2)],
            )
            self.add_module(
                "Head",
                GeneratorV3.NestedUNetHead(channels[:2], dim=dim),
                in_branch=[f"X_0_{len(channels) - 2}"],
            )

    def __init__(
        self,
        optimizer: network.OptimizerLoader = network.OptimizerLoader(),
        schedulers: dict[str, network.LRSchedulersLoader] = {
            "default|ReduceLROnPlateau": network.LRSchedulersLoader(0)
        },
        outputs_criterions: dict[str, network.TargetCriterionsLoader] = {"default": network.TargetCriterionsLoader()},
        patch: ModelPatch | None = None,
        channels: list[int] = [1, 64, 128, 256, 512, 1024],
        block_config: blocks.BlockConfig = blocks.BlockConfig(),
        nb_conv_per_stage: int = 2,
        downsample_mode: str = "MAXPOOL",
        upsample_mode: str = "CONV_TRANSPOSE",
        attention: bool = False,
        block_type: str = "Conv",
        dim: int = 3,
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
            "GeneratorModel",
            GeneratorV3.GeneratorBlock(
                channels,
                block_config,
                nb_conv_per_stage,
                downsample_mode,
                upsample_mode,
                attention,
                block_type,
                dim,
            ),
            out_branch=["pB"],
        )


class DiffusionGan(network.Network):
    def __init__(
        self,
        generator: GeneratorV1 = GeneratorV1(),
        discriminator: DiscriminatorADA = DiscriminatorADA(),
    ) -> None:
        super().__init__()
        self.add_module("Generator_A_to_B", generator, in_branch=[0], out_branch=["pB"])
        self.add_module(
            "Discriminator_B",
            discriminator,
            in_branch=[1],
            out_branch=[-1],
            requires_grad=True,
        )
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


class DiffusionGanV2(network.Network):
    def __init__(
        self,
        generator: GeneratorV2 = GeneratorV2(),
        discriminator: Discriminator = Discriminator(),
    ) -> None:
        super().__init__()
        self.add_module("Generator_A_to_B", generator, in_branch=[0], out_branch=["pB"])
        self.add_module(
            "Discriminator_B",
            discriminator,
            in_branch=[1],
            out_branch=[-1],
            requires_grad=True,
        )
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


class CycleGanDiscriminator(network.Network):
    def __init__(
        self,
        optimizer: network.OptimizerLoader = network.OptimizerLoader(),
        schedulers: dict[str, network.LRSchedulersLoader] = {
            "default|ReduceLROnPlateau": network.LRSchedulersLoader(0)
        },
        outputs_criterions: dict[str, network.TargetCriterionsLoader] = {"default": network.TargetCriterionsLoader()},
        patch: ModelPatch | None = None,
        channels: list[int] = [1, 16, 32, 64, 64],
        strides: list[int] = [2, 2, 2, 1],
        dim: int = 3,
    ) -> None:
        super().__init__(
            in_channels=1,
            optimizer=optimizer,
            schedulers=schedulers,
            outputs_criterions=outputs_criterions,
            patch=patch,
            dim=dim,
        )
        self.add_module(
            "Discriminator_A",
            Discriminator.DiscriminatorBlock(channels, strides, dim),
            in_branch=[0],
            out_branch=[0],
        )
        self.add_module(
            "Discriminator_B",
            Discriminator.DiscriminatorBlock(channels, strides, dim),
            in_branch=[1],
            out_branch=[1],
        )


class CycleGanGeneratorV1(network.Network):
    def __init__(
        self,
        optimizer: network.OptimizerLoader = network.OptimizerLoader(),
        schedulers: dict[str, network.LRSchedulersLoader] = {
            "default|ReduceLROnPlateau": network.LRSchedulersLoader(0)
        },
        outputs_criterions: dict[str, network.TargetCriterionsLoader] = {"default": network.TargetCriterionsLoader()},
        patch: ModelPatch | None = None,
        dim: int = 3,
    ) -> None:
        super().__init__(
            in_channels=1,
            optimizer=optimizer,
            schedulers=schedulers,
            outputs_criterions=outputs_criterions,
            patch=patch,
            dim=dim,
        )
        self.add_module(
            "Generator_A_to_B",
            GeneratorV1.GeneratorBlock(32, dim),
            in_branch=[0],
            out_branch=["pB"],
        )
        self.add_module(
            "Generator_B_to_A",
            GeneratorV1.GeneratorBlock(32, dim),
            in_branch=[1],
            out_branch=["pA"],
        )


class CycleGanGeneratorV2(network.Network):
    def __init__(
        self,
        optimizer: network.OptimizerLoader = network.OptimizerLoader(),
        schedulers: dict[str, network.LRSchedulersLoader] = {
            "default|ReduceLROnPlateau": network.LRSchedulersLoader(0)
        },
        outputs_criterions: dict[str, network.TargetCriterionsLoader] = {"default": network.TargetCriterionsLoader()},
        patch: ModelPatch | None = None,
        channels: list[int] = [1, 64, 128, 256, 512, 1024],
        block_config: blocks.BlockConfig = blocks.BlockConfig(),
        nb_conv_per_stage: int = 2,
        downsample_mode: str = "MAXPOOL",
        upsample_mode: str = "CONV_TRANSPOSE",
        attention: bool = False,
        block_type: str = "Conv",
        dim: int = 3,
    ) -> None:
        super().__init__(
            in_channels=1,
            optimizer=optimizer,
            schedulers=schedulers,
            outputs_criterions=outputs_criterions,
            patch=patch,
            dim=dim,
        )
        self.add_module(
            "Generator_A_to_B",
            GeneratorV2.GeneratorBlock(
                channels,
                block_config,
                nb_conv_per_stage,
                downsample_mode,
                upsample_mode,
                attention,
                block_type,
                dim,
            ),
            in_branch=[0],
            out_branch=["pB"],
        )
        self.add_module(
            "Generator_B_to_A",
            GeneratorV2.GeneratorBlock(
                channels,
                block_config,
                nb_conv_per_stage,
                downsample_mode,
                upsample_mode,
                attention,
                block_type,
                dim,
            ),
            in_branch=[1],
            out_branch=["pA"],
        )


class CycleGanGeneratorV3(network.Network):
    def __init__(
        self,
        optimizer: network.OptimizerLoader = network.OptimizerLoader(),
        schedulers: dict[str, network.LRSchedulersLoader] = {
            "default|ReduceLROnPlateau": network.LRSchedulersLoader(0)
        },
        outputs_criterions: dict[str, network.TargetCriterionsLoader] = {"default": network.TargetCriterionsLoader()},
        patch: ModelPatch | None = None,
        channels: list[int] = [1, 64, 128, 256, 512, 1024],
        block_config: blocks.BlockConfig = blocks.BlockConfig(),
        nb_conv_per_stage: int = 2,
        downsample_mode: str = "MAXPOOL",
        upsample_mode: str = "CONV_TRANSPOSE",
        attention: bool = False,
        block_type: str = "Conv",
        dim: int = 3,
    ) -> None:
        super().__init__(
            in_channels=1,
            optimizer=optimizer,
            schedulers=schedulers,
            outputs_criterions=outputs_criterions,
            patch=patch,
            dim=dim,
        )
        self.add_module(
            "Generator_A_to_B",
            GeneratorV3.GeneratorBlock(
                channels,
                block_config,
                nb_conv_per_stage,
                downsample_mode,
                upsample_mode,
                attention,
                block_type,
                dim,
            ),
            in_branch=[0],
            out_branch=["pB"],
        )
        self.add_module(
            "Generator_B_to_A",
            GeneratorV3.GeneratorBlock(
                channels,
                block_config,
                nb_conv_per_stage,
                downsample_mode,
                upsample_mode,
                attention,
                block_type,
                dim,
            ),
            in_branch=[1],
            out_branch=["pA"],
        )


class DiffusionCycleGan(network.Network):
    def __init__(
        self,
        generators: CycleGanGeneratorV3 = CycleGanGeneratorV3(),
        discriminators: CycleGanDiscriminator = CycleGanDiscriminator(),
    ) -> None:
        super().__init__()
        self.add_module("Generator", generators, in_branch=[0, 1], out_branch=["pB", "pA"])
        self.add_module(
            "Discriminator",
            discriminators,
            in_branch=[0, 1],
            out_branch=[-1],
            requires_grad=True,
        )

        self.add_module("Generator_identity", generators, in_branch=[1, 0], out_branch=[-1])

        self.add_module("Generator_p", generators, in_branch=["pA", "pB"], out_branch=[-1])

        self.add_module("detach_pA", blocks.Detach(), in_branch=["pA"], out_branch=["pA_detach"])
        self.add_module("detach_pB", blocks.Detach(), in_branch=["pB"], out_branch=["pB_detach"])

        self.add_module(
            "Discriminator_p_detach",
            discriminators,
            in_branch=["pA_detach", "pB_detach"],
            out_branch=[-1],
        )
        self.add_module(
            "Discriminator_p",
            discriminators,
            in_branch=["pA", "pB"],
            out_branch=[-1],
            requires_grad=False,
        )
