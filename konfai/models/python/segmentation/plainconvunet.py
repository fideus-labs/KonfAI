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

"""Parametric PlainConvUNet -- weight-exact with nnU-Net's plain U-Net, at any topology.

This is the Python counterpart of the declarative ``konfai/models/yaml/PlainConvUNet.yml``.
The YAML template is fixed to a 4-stage graph because a declarative config cannot loop; this
class builds an encoder/decoder graph of **arbitrary depth** directly from the nnU-Net
hyper-parameters (``n_stages``, ``features_per_stage``, ``strides``, ``kernel_sizes``, ``n_conv_per_stage``,
``n_conv_per_stage_decoder``, ``num_classes``). It reproduces, module-for-module and in
forward-execution order, ``dynamic_network_architectures.architectures.unet.PlainConvUNet``
(the "plain conv" nnU-Net backbone used by nnU-Net, TotalSegmentator and MRSeg), so a **real
nnU-Net checkpoint of any depth** loads straight in through the execution-order bridge
``konfai.utils.pretrained.transfer_weights_by_execution_order`` and the KonfAI logits are
``torch.allclose`` with the reference output. Reference it as
``classpath: segmentation.plainconvunet.PlainConvUNet``.

nnU-Net signature reproduced here (Isensee et al., Nature Methods 18, 2021):

* every conv block is ``Conv(bias=True) -> InstanceNorm(affine=True, track_running_stats=False)
  -> LeakyReLU(0.01)`` (``norm_mode=INSTANCE_AFFINE`` maps byte-for-byte to nnU-Net's
  ``norm_op=InstanceNorm*d, norm_op_kwargs={"affine": True}``);
* each encoder stage has ``n_conv_per_stage`` conv blocks; the **first** conv of each stage
  carries that stage's stride (nnU-Net strided-conv downsampling, never a pooling layer),
  stage 0 keeps stride 1 = full resolution;
* each decoder stage upsamples with a transpose conv whose kernel size **and** stride both
  equal the matching encoder stride, then concatenates ``(transpose_output, encoder_skip)`` in
  that order, then applies ``n_conv_per_stage_decoder`` conv blocks;
* a 1x1 segmentation head is built at **every** decoder resolution (native deep supervision --
  exactly what nnU-Net's checkpoint carries), each exposed as a named terminal output.

The segmentation heads emit **raw logits** (no softmax/argmax inside), like nnU-Net: the head
of decoder stage ``j`` is the trace node ``SegHead_j`` and is the weight-exact comparison
point. Building every head (rather than only the executed one, as the fixed YAML does) gives
full parameter-count equality with the reference -- there is no built-but-unused head gap.

``strides`` accepts both isotropic ints (e.g. ``2``) and per-axis lists (e.g. ``[1, 2, 2]``)
per stage, so real anisotropic TotalSegmentator / MRSeg configs load unchanged, e.g.
``strides=[1, [1, 2, 2], [2, 2, 2], [2, 2, 2]]``.
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
    """Broadcast an int kernel to all stages, or validate a per-stage list.

    nnU-Net picks ``kernel_sizes`` independently of ``strides`` (a stage may use kernel 1 or an
    anisotropic ``[1, 3, 3]`` while its stride is ``[1, 2, 2]``), so each entry is an int or a
    per-axis list.
    """
    if isinstance(kernel_sizes, int):
        return [kernel_sizes] * n_stages
    if len(kernel_sizes) != n_stages:
        raise ConfigError(
            f"'kernel_sizes' must have n_stages={n_stages} entries (got {len(kernel_sizes)}: {kernel_sizes}).",
            "One kernel per resolution stage; each entry is an int or a per-axis list.",
        )
    return list(kernel_sizes)


def _kernel_padding(kernel_size: int | list[int]) -> int | list[int]:
    """Same-padding for an odd kernel: ``k // 2`` per axis (nnU-Net's ``(k - 1) // 2`` for odd k)."""
    if isinstance(kernel_size, int):
        return kernel_size // 2
    return [k // 2 for k in kernel_size]


def _conv_block_config(stride: int | list[int], kernel_size: int | list[int]) -> blocks.BlockConfig:
    """One nnU-Net conv block: Conv(bias) -> InstanceNorm(affine) -> LeakyReLU(0.01).

    ``stride`` and ``kernel_size`` are each an int (isotropic) or a per-axis list (anisotropic);
    nnU-Net picks them INDEPENDENTLY, so both pass straight through to the convolution. Padding is
    same-padding derived from the kernel, so a checkpoint with kernel != 3 (or an anisotropic
    ``[1, 3, 3]``) loads without a shape mismatch.
    """
    stride_value: Any = stride
    kernel_value: Any = kernel_size
    padding_value: Any = _kernel_padding(kernel_size)
    return blocks.BlockConfig(
        kernel_size=kernel_value,
        stride=stride_value,
        padding=padding_value,
        bias=True,
        activation="LeakyReLU",
        norm_mode="INSTANCE_AFFINE",
    )


@config()
class PlainConvUNet(network.Network):
    """nnU-Net PlainConvUNet of arbitrary depth as a KonfAI model (2D or 3D).

    ``load`` never re-initialises: an nnU-Net checkpoint is meant to be transferred in through
    the execution-order bridge, and the trainer's ``load(init=True)`` would silently destroy
    those weights with ``init_type`` noise. Checkpoint loading is unaffected.
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
        n_stages: int = 4,
        features_per_stage: list[int] = [32, 64, 128, 256],
        strides: list[int | list[int]] = [1, 2, 2, 2],
        kernel_sizes: int | list[Any] = 3,
        n_conv_per_stage: int | list[int] = 2,
        n_conv_per_stage_decoder: int | list[int] = 2,
        num_classes: int = 2,
    ) -> None:
        if dim not in (2, 3):
            raise ConfigError(
                f"PlainConvUNet supports dim 2 or 3, got dim={dim}.",
                "Use dim: 2 for slice-wise / 2.5D inputs or dim: 3 for volumetric inputs.",
            )
        if n_stages < 2:
            raise ConfigError(
                f"PlainConvUNet needs at least 2 stages (got n_stages={n_stages}).",
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
        n_conv_encoder = _as_stage_list(n_conv_per_stage, n_stages, "n_conv_per_stage")
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

        # ----- Encoder (contracting path) --------------------------------------------------- #
        # Stage k has ``n_conv_encoder[k]`` blocks; the first block carries stride ``strides[k]``
        # (stage 0 = full resolution), the rest are stride 1. Output written to branch ``enc{k}``.
        stage_in = in_channels
        for k in range(n_stages):
            block_configs = [_conv_block_config(strides[k], kernel_list[k])] + [
                _conv_block_config(1, kernel_list[k]) for _ in range(n_conv_encoder[k] - 1)
            ]
            self.add_module(
                f"Encoder_{k}",
                blocks.ConvBlock(
                    in_channels=stage_in,
                    out_channels=features_per_stage[k],
                    block_configs=block_configs,
                    dim=dim,
                ),
                in_branch=[0 if k == 0 else f"enc{k - 1}"],
                out_branch=[f"enc{k}"],
            )
            stage_in = features_per_stage[k]

        # ----- Decoder (expanding path) ----------------------------------------------------- #
        # Decoder stage j runs coarsest-first (j=0 upsamples the bottleneck), exactly matching
        # nnU-Net's ``UNetDecoder`` execution order: transpose conv, concat(skip), conv blocks,
        # then the stage's 1x1 seg head -- before moving to the next (finer) stage.
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
                    bias=True,
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
                    block_configs=[_conv_block_config(1, kernel_list[skip_index]) for _ in range(n_conv_decoder[j])],
                    dim=dim,
                ),
                in_branch=[f"up{j}"],
                out_branch=[f"dec{j}"],
            )
            # 1x1 seg head at this resolution -- raw logits, a named terminal output. Building
            # every head (native deep supervision) is what makes the parameter count match the
            # reference exactly, with no built-but-unused head gap.
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
        del init  # an nnU-Net checkpoint is transferred in; never re-initialise loaded weights
        super().load(state_dict, init=False, ema=ema, override_lr=override_lr)
