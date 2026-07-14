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

"""Parametric ResidualEncoderUNet -- weight-exact with nnU-Net's residual-encoder U-Net.

This is the residual-encoder counterpart of ``konfai/models/python/segmentation/plainconvunet.py``.
It builds, module-for-module and in forward-execution order, the network produced by
``dynamic_network_architectures.architectures.unet.ResidualEncoderUNet`` (the "ResEnc" nnU-Net
backbone -- e.g. the ImpactSeg body model), so a **real ResEnc nnU-Net checkpoint of any depth**
loads straight in through the execution-order bridge
``konfai.utils.pretrained.transfer_weights_by_execution_order`` and the KonfAI logits are
``torch.allclose`` with the reference output. Reference it as
``classpath: segmentation.residualencoderunet.ResidualEncoderUNet``.

The **decoder is identical** to ``PlainConvUNet``'s (nnU-Net shares ``UNetDecoder`` between the two
backbones): a transpose conv (kernel = stride = the matching encoder stride), a concat with the
encoder skip, ``n_conv_per_stage_decoder`` conv blocks, and a 1x1 seg head at **every** decoder
resolution (native deep supervision -- what a real ResEnc checkpoint carries). Building every head
gives full parameter-count equality with the reference.

Only the **encoder** differs. Following nnU-Net's ``ResidualEncoder``:

* a **stem** = one ``Conv(stride 1) -> Norm -> LeakyReLU`` block at full resolution (never strided);
* then ``n_stages`` residual stages, stage ``k`` stacking ``n_blocks_per_stage[k]`` ResNet-D
  ``ResidualBlockD`` blocks. The **first** block of a stage carries that stage's stride (nnU-Net
  strided-conv downsampling; stage 0 keeps stride 1 = full resolution), the rest are stride 1.
  Each stage's output is a decoder skip.

``ResidualBlockD`` (see ``konfai.network.blocks``) is the ResNet-D basic block: a strided residual
is downsampled with an ``AvgPool`` and, when channels change, projected with a ``1x1`` conv
(``bias=False`` always) + norm; the main path is ``Conv -> Norm -> LeakyReLU -> Conv -> Norm``; the
two are summed and a final ``LeakyReLU`` applied.

``strides`` / ``kernel_sizes`` accept both isotropic ints and per-axis lists per stage, so real
anisotropic plans load unchanged. The segmentation heads emit **raw logits** (no softmax/argmax),
like nnU-Net; the head of decoder stage ``j`` is the trace node ``SegHead_j`` and is the
weight-exact comparison point.
"""

from typing import Any

from konfai.data.patching import ModelPatch
from konfai.network import blocks, network
from konfai.utils.config import config
from konfai.utils.errors import ConfigError


def _as_stage_list(value: int | list[int], length: int, name: str) -> list[int]:
    """Broadcast an int to a per-stage list, or validate a list of the expected length."""
    if isinstance(value, int):
        return [value] * length
    if len(value) != length:
        raise ConfigError(
            f"'{name}' must have {length} entries (got {len(value)}: {value}).",
            "It is broadcast per resolution stage, so its length must match the topology.",
        )
    return list(value)


def _as_kernel_list(kernel_sizes: int | list[Any], n_stages: int) -> list[Any]:
    """Broadcast an int kernel to all stages, or validate a per-stage list (each int or per-axis list)."""
    if isinstance(kernel_sizes, int):
        return [kernel_sizes] * n_stages
    if len(kernel_sizes) != n_stages:
        raise ConfigError(
            f"'kernel_sizes' must have n_stages={n_stages} entries (got {len(kernel_sizes)}: {kernel_sizes}).",
            "One kernel per resolution stage; each entry is an int or a per-axis list.",
        )
    return list(kernel_sizes)


def _kernel_padding(kernel_size: int | list[int]) -> int | list[int]:
    """Same-padding for the decoder convs: ``(k - 1) // 2`` per axis (nnU-Net's ConvDropoutNormReLU)."""
    if isinstance(kernel_size, int):
        return (kernel_size - 1) // 2
    return [(k - 1) // 2 for k in kernel_size]


def _conv_block_config(
    stride: int | list[int], kernel_size: int | list[int], bias: bool, negative_slope: float
) -> blocks.BlockConfig:
    """One nnU-Net decoder/stem conv block: Conv(bias) -> InstanceNorm(affine) -> LeakyReLU.

    ``stride``/``kernel_size`` are each an int (isotropic) or a per-axis list (anisotropic); both
    pass straight through the convolution with same-padding derived from the kernel.
    """
    stride_value: Any = stride
    kernel_value: Any = kernel_size
    padding_value: Any = _kernel_padding(kernel_size)
    return blocks.BlockConfig(
        kernel_size=kernel_value,
        stride=stride_value,
        padding=padding_value,
        bias=bias,
        activation=f"LeakyReLU;{negative_slope}",
        norm_mode="INSTANCE_AFFINE",
    )


@config()
class ResidualEncoderUNet(network.Network):
    """nnU-Net ResidualEncoderUNet of arbitrary depth as a KonfAI model (2D or 3D).

    ``load`` never re-initialises: a ResEnc nnU-Net checkpoint is meant to be transferred in through
    the execution-order bridge, and the trainer's ``load(init=True)`` would silently destroy those
    weights with ``init_type`` noise. Checkpoint loading is unaffected.
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
        dim: int = 3,
        in_channels: int = 1,
        n_stages: int = 6,
        features_per_stage: list[int] = [32, 64, 128, 256, 320, 320],
        strides: list[int | list[int]] = [1, 2, 2, 2, 2, 2],
        kernel_sizes: int | list[Any] = 3,
        n_blocks_per_stage: int | list[int] = [1, 3, 4, 6, 6, 6],
        n_conv_per_stage_decoder: int | list[int] = 1,
        num_classes: int = 2,
        conv_bias: bool = True,
        negative_slope: float = 1e-2,
        deep_supervision: bool = True,
    ) -> None:
        if dim not in (2, 3):
            raise ConfigError(
                f"ResidualEncoderUNet supports dim 2 or 3, got dim={dim}.",
                "Use dim: 2 for slice-wise / 2.5D inputs or dim: 3 for volumetric inputs.",
            )
        if n_stages < 2:
            raise ConfigError(
                f"ResidualEncoderUNet needs at least 2 stages (got n_stages={n_stages}).",
                "A U-Net requires one encoder stage above the bottleneck to form a decoder stage.",
            )
        if len(features_per_stage) != n_stages:
            raise ConfigError(
                f"'features_per_stage' must have n_stages={n_stages} entries "
                f"(got {len(features_per_stage)}: {features_per_stage}).",
                "One feature width per resolution stage; the last entry is the bottleneck.",
            )
        if len(strides) != n_stages:
            raise ConfigError(
                f"'strides' must have n_stages={n_stages} entries (got {len(strides)}: {strides}).",
                "One stride per resolution stage; entry 0 is the full-resolution stage (usually 1).",
            )
        kernel_list = _as_kernel_list(kernel_sizes, n_stages)
        n_blocks_encoder = _as_stage_list(n_blocks_per_stage, n_stages, "n_blocks_per_stage")
        n_conv_decoder = _as_stage_list(n_conv_per_stage_decoder, n_stages - 1, "n_conv_per_stage_decoder")

        super().__init__(
            in_channels=in_channels,
            optimizer=optimizer,
            schedulers=schedulers,
            outputs_criterions=outputs_criterions,
            patch=patch,
            nb_batch_per_step=nb_batch_per_step,
            dim=dim,
        )

        # ----- Encoder: stem + residual stages ---------------------------------------------- #
        # The stem is one non-strided conv block (nnU-Net's ResidualEncoder stem never downsamples)
        # and produces ``features_per_stage[0]`` channels feeding stage 0. Its output lives on branch
        # ``stem`` and is consumed by stage 0 only (it is NOT a decoder skip).
        self.add_module(
            "Stem",
            blocks.ConvBlock(
                in_channels=in_channels,
                out_channels=features_per_stage[0],
                block_configs=[_conv_block_config(1, kernel_list[0], conv_bias, negative_slope)],
                dim=dim,
            ),
            in_branch=[0],
            out_branch=["stem"],
        )

        # Stage k stacks ``n_blocks_encoder[k]`` residual blocks; the FIRST block carries stride
        # ``strides[k]`` (stage 0 = full resolution) and the channel change, the rest are stride 1 at
        # ``features_per_stage[k]``. The stage output is written to branch ``enc{k}`` = decoder skip k.
        for k in range(n_stages):
            stage_in = features_per_stage[0] if k == 0 else features_per_stage[k - 1]
            for i in range(n_blocks_encoder[k]):
                block_in = stage_in if i == 0 else features_per_stage[k]
                block_stride: Any = strides[k] if i == 0 else 1
                if i == 0:
                    source = "stem" if k == 0 else f"enc{k - 1}"
                else:
                    source = f"enc{k}"
                self.add_module(
                    f"Encoder_{k}_Block_{i}",
                    blocks.ResidualBlockD(
                        in_channels=block_in,
                        out_channels=features_per_stage[k],
                        dim=dim,
                        kernel_size=kernel_list[k],
                        stride=block_stride,
                        conv_bias=conv_bias,
                        negative_slope=negative_slope,
                    ),
                    in_branch=[source],
                    out_branch=[f"enc{k}"],
                )

        # ----- Decoder (identical to PlainConvUNet -- nnU-Net shares UNetDecoder) ------------ #
        # Decoder stage j runs coarsest-first (j=0 upsamples the bottleneck), matching nnU-Net's
        # ``UNetDecoder`` execution order: transpose conv, concat(skip), conv blocks, then the 1x1
        # seg head -- before moving to the next (finer) stage.
        for j in range(n_stages - 1):
            below_index = n_stages - 1 - j  # encoder feature feeding the transpose conv
            skip_index = n_stages - 2 - j  # encoder stage providing the skip connection
            below_channels = features_per_stage[below_index]
            skip_channels = features_per_stage[skip_index]
            transpose_stride: Any = strides[below_index]

            self.add_module(
                f"Up_{j}",
                blocks.get_torch_module("ConvTranspose", dim)(
                    in_channels=below_channels,
                    out_channels=skip_channels,
                    kernel_size=transpose_stride,
                    stride=transpose_stride,
                    padding=0,
                    bias=conv_bias,
                ),
                in_branch=[f"enc{n_stages - 1}" if j == 0 else f"dec{j - 1}"],
                out_branch=[f"up{j}"],
            )
            self.add_module(
                f"Skip_{j}",
                blocks.Concat(),
                in_branch=[f"up{j}", f"enc{skip_index}"],
                out_branch=[f"up{j}"],
            )
            self.add_module(
                f"Decoder_{j}",
                blocks.ConvBlock(
                    in_channels=2 * skip_channels,
                    out_channels=skip_channels,
                    block_configs=[
                        _conv_block_config(1, kernel_list[skip_index], conv_bias, negative_slope)
                        for _ in range(n_conv_decoder[j])
                    ],
                    dim=dim,
                ),
                in_branch=[f"up{j}"],
                out_branch=[f"dec{j}"],
            )
            # 1x1 seg head at this resolution -- raw logits, a named terminal output. With
            # ``deep_supervision`` (default) every decoder stage gets a head, matching nnU-Net's
            # full parameter count. With ``deep_supervision=False`` only the finest (full-resolution,
            # j == n_stages - 2) head is built and traversed -- the single-output configuration used
            # e.g. by the ImpactSeg body model, so a checkpoint trained that way pairs leaf-for-leaf.
            if deep_supervision or j == n_stages - 2:
                self.add_module(
                    f"SegHead_{j}",
                    blocks.get_torch_module("Conv", dim)(
                        in_channels=skip_channels,
                        out_channels=num_classes,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=True,
                    ),
                    in_branch=[f"dec{j}"],
                    out_branch=[-1],
                )

    def load(
        self,
        state_dict: dict,
        init: bool = True,
        ema: bool = False,
        override_lr: float | None = None,
    ) -> None:
        del init  # a ResEnc nnU-Net checkpoint is transferred in; never re-initialise loaded weights
        super().load(state_dict, init=False, ema=ema, override_lr=override_lr)
