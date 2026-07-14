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

"""Parametric UNet++ -- weight-exact and forward-exact with ``smp.UnetPlusPlus`` (ResNet encoder).

This is the Python counterpart of the fixed declarative ``UNetpp.yml`` used by the ImpactSynth app.
It builds, module-for-module and in forward-execution order, the network produced by
``segmentation_models_pytorch.UnetPlusPlus(encoder_name="resnet34", encoder_weights=None,
in_channels=IN, classes=CLS, activation=None)`` -- a **ResNet-18/34 encoder** feeding a **UNet++
nested (dense) decoder** -- so a real smp checkpoint loads straight in through the execution-order
bridge ``konfai.utils.pretrained.transfer_weights_by_execution_order`` and the KonfAI logits are
``torch.allclose`` with the reference output. Reference it as
``classpath: segmentation.unetplusplus.UNetPlusPlus``.

It contains **no** ``segmentation_models_pytorch`` import: the encoder and decoder blocks are pure
torch/KonfAI graph nodes (``ResNetBasicBlock``, ``ConvBlock``, ``Concat``, ``Upsample``), so the model
is safe by construction and depends only on the core package.

**Encoder** (``smp``'s ``ResNetEncoder``, itself ``torchvision``'s ``ResNet``): a stem
``Conv(7x7, stride 2) -> BatchNorm -> ReLU`` at 1/2 resolution, a ``MaxPool(3, stride 2)``, then four
residual stages of ``torchvision`` ``BasicBlock``\\ s (``ResNetBasicBlock``) with block counts
``[3, 4, 6, 3]`` for resnet34 / ``[2, 2, 2, 2]`` for resnet18 and channel widths ``[64, 128, 256, 512]``
(first block of stages 2-4 strided). The five feature maps consumed by the decoder are the stem output
(64ch, 1/2) and the four stage outputs (64/128/256/512ch at 1/4..1/32).

**Decoder** (``smp``'s ``UnetPlusPlusDecoder``): the UNet++ dense grid. Decoder node ``x_{d}_{l}`` fuses
the nearest-neighbour upsample of its shallower-column predecessor with every same-resolution dense node
already built and the matching encoder skip, then runs a ``DecoderBlock`` =
``Conv(3x3) -> BatchNorm -> ReLU`` twice. The channel arithmetic replicates smp exactly
(``in_channels``/``skip_channels``/``out_channels`` from ``decoder_channels`` and the reversed encoder
widths). The final ``x_{0}_{depth}`` node has no skip. A ``Conv(3x3)`` segmentation head emits raw logits
(``activation=None``); the terminal node is ``SegmentationHead``.
"""

import torch
from konfai.data.patching import ModelPatch
from konfai.network import blocks, network
from konfai.utils.config import config
from konfai.utils.errors import ConfigError

# BasicBlock ResNet encoders (expansion 1). Bottleneck backbones (resnet50+) are not reproduced here.
_RESNET_LAYERS: dict[str, list[int]] = {
    "resnet18": [2, 2, 2, 2],
    "resnet34": [3, 4, 6, 3],
}
_STAGE_CHANNELS = [64, 128, 256, 512]  # torchvision ResNet stage widths (planes) for a BasicBlock net
_STEM_CHANNELS = 64


@config()
class UNetPlusPlus(network.Network):
    """``smp.UnetPlusPlus`` (ResNet-18/34 encoder + UNet++ decoder) as a pure KonfAI model.

    ``load`` never re-initialises: an smp checkpoint is transferred in through the execution-order
    bridge, and the trainer's ``load(init=True)`` would silently destroy those weights with
    ``init_type`` noise. Checkpoint loading is unaffected.
    """

    def __init__(
        self,
        optimizer: network.OptimizerLoader = network.OptimizerLoader(),
        schedulers: dict[str, network.LRSchedulersLoader] = {
            "default|ReduceLROnPlateau": network.LRSchedulersLoader(0)
        },
        outputs_criterions: dict[str, network.TargetCriterionsLoader] = {"default": network.TargetCriterionsLoader()},
        patch: ModelPatch | None = None,
        nb_batch_per_step: int = 1,
        dim: int = 2,
        in_channels: int = 3,
        classes: int = 1,
        encoder_name: str = "resnet34",
        decoder_channels: list[int] = [256, 128, 64, 32, 16],
        activation: str | None = None,
    ) -> None:
        if dim not in (2, 3):
            raise ConfigError(
                f"UNetPlusPlus supports dim 2 or 3, got dim={dim}.",
                "Use dim: 2 for slice-wise / 2.5D inputs (the smp-compatible setting) or dim: 3 for volumes.",
            )
        if encoder_name not in _RESNET_LAYERS:
            raise ConfigError(
                f"UNetPlusPlus supports the BasicBlock ResNet encoders {sorted(_RESNET_LAYERS)}, got '{encoder_name}'.",
                "resnet18/resnet34 use torchvision's BasicBlock; deeper Bottleneck backbones are not reproduced.",
            )
        if len(decoder_channels) != 5:
            raise ConfigError(
                f"'decoder_channels' must have 5 entries (got {len(decoder_channels)}: {decoder_channels}).",
                "smp's UnetPlusPlus decoder has one channel width per encoder stage (encoder_depth=5).",
            )

        super().__init__(
            in_channels=in_channels,
            optimizer=optimizer,
            schedulers=schedulers,
            outputs_criterions=outputs_criterions,
            patch=patch,
            nb_batch_per_step=nb_batch_per_step,
            dim=dim,
        )

        layers = _RESNET_LAYERS[encoder_name]

        # ----- Encoder: stem + four residual (BasicBlock) stages ---------------------------- #
        # smp's ResNetEncoder yields six features; the decoder uses the last five: the stem output
        # (``enc_c1``, 1/2 res) and the four stage outputs (``enc_l1..enc_l4``, 1/4..1/32 res).
        self.add_module(
            "StemConv",
            blocks.get_torch_module("Conv", dim)(
                in_channels, _STEM_CHANNELS, kernel_size=7, stride=2, padding=3, bias=False
            ),
            in_branch=[0],
            out_branch=["enc_c1"],
        )
        self.add_module(
            "StemNorm",
            blocks.get_norm(blocks.NormMode.BATCH, _STEM_CHANNELS, dim),
            in_branch=["enc_c1"],
            out_branch=["enc_c1"],
        )
        self.add_module("StemReLU", torch.nn.ReLU(inplace=False), in_branch=["enc_c1"], out_branch=["enc_c1"])
        self.add_module(
            "StemMaxPool",
            blocks.get_torch_module("MaxPool", dim)(kernel_size=3, stride=2, padding=1),
            in_branch=["enc_c1"],
            out_branch=["enc_mp"],
        )

        stage_in = _STEM_CHANNELS
        for stage, n_blocks in enumerate(layers):
            planes = _STAGE_CHANNELS[stage]
            stride = 1 if stage == 0 else 2  # stage 0 (layer1) follows the maxpool at 1/4 and never strides
            source = "enc_mp" if stage == 0 else f"enc_l{stage}"
            out_branch = f"enc_l{stage + 1}"
            for block_idx in range(n_blocks):
                self.add_module(
                    f"Encoder_{stage + 1}_Block_{block_idx}",
                    blocks.ResNetBasicBlock(
                        in_channels=stage_in if block_idx == 0 else planes,
                        out_channels=planes,
                        dim=dim,
                        stride=stride if block_idx == 0 else 1,
                        conv_bias=False,
                        norm_mode="BATCH",
                    ),
                    in_branch=[source if block_idx == 0 else out_branch],
                    out_branch=[out_branch],
                )
            stage_in = planes

        # ----- UNet++ nested decoder -------------------------------------------------------- #
        # Reproduce smp's UnetPlusPlusDecoder channel arithmetic exactly. ``encoder_channels`` is the
        # decoder's view of the encoder widths: drop the input feature, reverse so the head is first.
        encoder_channels = [in_channels, _STEM_CHANNELS, _STEM_CHANNELS, *_STAGE_CHANNELS[1:]]
        reversed_channels = encoder_channels[1:][::-1]  # e.g. [512, 256, 128, 64, 64]
        head_channels = reversed_channels[0]
        in_ch_list = [head_channels, *decoder_channels[:-1]]  # [512, 256, 128, 64, 32]
        skip_ch_list = [*reversed_channels[1:], 0]  # [256, 128, 64, 64, 0]
        out_ch_list = list(decoder_channels)  # [256, 128, 64, 32, 16]
        depth = len(in_ch_list) - 1  # 4

        # ``feat[i]`` is smp's reversed encoder feature: feat[0] = deepest stage (enc_l4), feat[depth]
        # = stem output (enc_c1). ``dx(d, level)`` is the branch holding dense node ``x_{d}_{level}``.
        feat = ["enc_l4", "enc_l3", "enc_l2", "enc_l1", "enc_c1"]

        def dx(depth_idx: int, level: int) -> str:
            return f"dx_{depth_idx}_{level}"

        def channels(depth_idx: int, level: int) -> tuple[int, int]:
            """smp's per-block (conv1_in, out) channels, keyed by (depth_idx, level=layer_idx)."""
            if depth_idx == 0:
                in_ch = in_ch_list[level]
                skip_ch = skip_ch_list[level] * (level + 1)
                out_ch = out_ch_list[level]
            else:
                out_ch = skip_ch_list[level]
                skip_ch = skip_ch_list[level] * (level + 1 - depth_idx)
                in_ch = skip_ch_list[level - 1]
            return in_ch + skip_ch, out_ch

        def add_decoder_node(name: str, up_source: str, skip_sources: list[str], conv_in: int, conv_out: int) -> str:
            """One DecoderBlock: nearest upsample, optional dense concat, then two Conv-BN-ReLU."""
            out_branch = dx(*[int(part) for part in name.split("_")[1:]])
            self.add_module(
                f"{name}_up",
                torch.nn.Upsample(scale_factor=2.0, mode="nearest"),
                in_branch=[up_source],
                out_branch=[f"{name}_up"],
            )
            conv_input = f"{name}_up"
            if skip_sources:
                self.add_module(
                    f"{name}_cat",
                    blocks.Concat(),
                    in_branch=[f"{name}_up", *skip_sources],
                    out_branch=[f"{name}_c"],
                )
                conv_input = f"{name}_c"
            self.add_module(
                f"{name}_conv",
                blocks.ConvBlock(
                    in_channels=conv_in,
                    out_channels=conv_out,
                    block_configs=[_decoder_block_config(dim) for _ in range(2)],
                    dim=dim,
                ),
                in_branch=[conv_input],
                out_branch=[out_branch],
            )
            return out_branch

        # Dense connections, built in smp's forward-execution order so weighted leaves pair one-to-one.
        for layer_idx in range(depth):
            for depth_idx in range(depth - layer_idx):
                if layer_idx == 0:
                    name = f"x_{depth_idx}_{depth_idx}"
                    up_source = feat[depth_idx]
                    skip_sources = [feat[depth_idx + 1]]
                    conv_in, conv_out = channels(depth_idx, depth_idx)
                else:
                    dense_l_i = depth_idx + layer_idx
                    name = f"x_{depth_idx}_{dense_l_i}"
                    up_source = dx(depth_idx, dense_l_i - 1)
                    skip_sources = [dx(idx, dense_l_i) for idx in range(depth_idx + 1, dense_l_i + 1)]
                    skip_sources.append(feat[dense_l_i + 1])
                    conv_in, conv_out = channels(depth_idx, dense_l_i)
                add_decoder_node(name, up_source, skip_sources, conv_in, conv_out)

        # Final full-resolution node ``x_0_depth`` has no skip (smp: DecoderBlock(in, 0, out)).
        final_in, final_out = channels(0, depth)
        final_branch = add_decoder_node(f"x_0_{depth}", dx(0, depth - 1), [], final_in, final_out)

        # ----- Segmentation head (smp SegmentationHead, activation=None -> raw logits) ------ #
        self.add_module(
            "SegmentationHead",
            blocks.get_torch_module("Conv", dim)(
                out_ch_list[-1], classes, kernel_size=3, stride=1, padding=1, bias=True
            ),
            in_branch=[final_branch],
            out_branch=[-1] if activation is None else ["seg"],
        )
        if activation is not None:
            self.add_module(
                "Activation",
                _activation_module(activation),
                in_branch=["seg"],
                out_branch=[-1],
            )

    def load(
        self,
        state_dict: dict,
        init: bool = True,
        ema: bool = False,
        override_lr: float | None = None,
    ) -> None:
        del init  # an smp checkpoint is transferred in; never re-initialise loaded weights
        super().load(state_dict, init=False, ema=ema, override_lr=override_lr)


def _decoder_block_config(dim: int) -> blocks.BlockConfig:
    """One smp ``Conv2dReLU``: Conv(3x3, bias=False) -> BatchNorm -> ReLU.

    ``dim`` is unused here (kernel/stride/padding are dimension-agnostic) but kept for symmetry with the
    other model helpers; the block config is materialised into ``dim``-d layers by ``ConvBlock``.
    """
    del dim
    return blocks.BlockConfig(
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        activation="ReLU",
        norm_mode="BATCH",
    )


def _activation_module(activation: str) -> torch.nn.Module:
    """Map an smp activation name to the matching curated torch module (raw logits stay activation=None)."""
    mapping = {
        "sigmoid": torch.nn.Sigmoid,
        "tanh": torch.nn.Tanh,
        "softmax": lambda: torch.nn.Softmax(dim=1),
        "identity": torch.nn.Identity,
    }
    if activation not in mapping:
        raise ConfigError(
            f"Unsupported activation '{activation}' for UNetPlusPlus.",
            f"Use one of {sorted(mapping)} or activation: null for raw logits.",
        )
    return mapping[activation]()
