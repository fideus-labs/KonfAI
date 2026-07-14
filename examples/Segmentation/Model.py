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

"""The Python form of this example's model — the same U-Net as ``UNet.yml``.

KonfAI accepts a model as a declarative ``.yml`` graph (``classpath: UNet.yml``) or as a
Python ``network.Network`` built with ``add_module`` (``classpath: Model:UNet``). This file
is the second form: the same architecture (channels ``[1,32,64,128,256]``, max-pool
downsampling, ``Conv->ReLU`` blocks, deep-supervision heads), so swapping the ``classpath``
in ``Config.yml`` trains the same network. Use the YAML form for a no-code, shareable model;
reach for the Python form when a model needs a custom ``forward`` or logic a declarative graph
cannot express (see the Synthesis example).
"""

import torch
from konfai.data.patching import ModelPatch
from konfai.network import blocks, network


def _conv_block(in_channels: int, out_channels: int, dim: int) -> blocks.ConvBlock:
    """Two ``Conv(3x3) -> ReLU`` layers, no normalization — matching ``UNet.yml``'s ConvBlock."""
    cfg = blocks.BlockConfig(kernel_size=3, stride=1, padding=1, activation="ReLU", norm_mode="NONE")
    return blocks.ConvBlock(in_channels, out_channels, [cfg, cfg], dim)


class Head(network.ModuleArgsDict):
    """1x1 conv to the class logits, then Softmax + ArgMax."""

    def __init__(self, in_channels: int, nb_class: int, dim: int) -> None:
        super().__init__()
        self.add_module("Conv", blocks.get_torch_module("Conv", dim)(in_channels, nb_class, kernel_size=1))
        self.add_module("Softmax", torch.nn.Softmax(dim=1))
        self.add_module("Argmax", blocks.ArgMax(dim=1))


class UNetBlock(network.ModuleArgsDict):
    """One resolution level, recursively nesting the next (coarser) one.

    Non-top levels max-pool down on entry and transpose-conv + skip-concat back on exit; the
    coarsest level (two channel entries) has no nested block. Levels above the bottleneck emit
    a deep-supervision ``Head`` (a terminal ``out_branch=[-1]`` output).
    """

    def __init__(self, channels: list[int], nb_class: int, dim: int, is_top: bool = True) -> None:
        super().__init__()
        if not is_top:
            self.add_module("MAXPOOL", blocks.get_torch_module("MaxPool", dim)(kernel_size=2, stride=2))
        self.add_module("DownConvBlock", _conv_block(channels[0], channels[1], dim))
        if len(channels) > 2:
            self.add_module("UNetBlock", UNetBlock(channels[1:], nb_class, dim, is_top=False))
            self.add_module("UpConvBlock", _conv_block(channels[1] * 2, channels[1], dim))
            self.add_module("Head", Head(channels[1], nb_class, dim), out_branch=[-1])
        if not is_top:
            self.add_module(
                "CONV_TRANSPOSE",
                blocks.get_torch_module("ConvTranspose", dim)(channels[1], channels[0], kernel_size=2, stride=2),
            )
            self.add_module("SkipConnection", blocks.Concat(), in_branch=[0, 1])


class UNet(network.Network):
    """The example U-Net, built in Python — equivalent to ``UNet.yml``."""

    def __init__(
        self,
        optimizer: network.OptimizerLoader = network.OptimizerLoader(),
        schedulers: dict[str, network.LRSchedulersLoader] = {
            "default|ReduceLROnPlateau": network.LRSchedulersLoader(0)
        },
        outputs_criterions: dict[str, network.TargetCriterionsLoader] = {"default": network.TargetCriterionsLoader()},
        patch: ModelPatch | None = None,
        dim: int = 2,
        channels: list[int] = [1, 32, 64, 128, 256],
        nb_class: int = 41,
    ) -> None:
        super().__init__(
            in_channels=channels[0],
            optimizer=optimizer,
            schedulers=schedulers,
            outputs_criterions=outputs_criterions,
            patch=patch,
            dim=dim,
        )
        self.add_module("UNetBlock_0", UNetBlock(channels, nb_class, dim))
