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

"""Shared construction helpers for the nnU-Net backbones (PlainConvUNet / ResidualEncoderUNet).

nnU-Net shares ``UNetDecoder`` between its two backbones; this module mirrors that sharing so the
two KonfAI counterparts cannot drift apart. Encoder construction and the per-model padding /
activation conventions stay in each model file.
"""

from collections.abc import Callable
from typing import Any

from konfai.network import blocks, network
from konfai.utils.errors import ConfigError


def validate_topology(
    name: str, dim: int, n_stages: int, features_per_stage: list[int], strides: list[int | list[int]]
) -> None:
    """The nnU-Net topology checks shared by both backbones."""
    if dim not in (2, 3):
        raise ConfigError(
            f"{name} supports dim 2 or 3, got dim={dim}.",
            "Use dim: 2 for slice-wise / 2.5D inputs or dim: 3 for volumetric inputs.",
        )
    if n_stages < 2:
        raise ConfigError(
            f"{name} needs at least 2 stages (got n_stages={n_stages}).",
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


def as_stage_list(value: int | list[int], length: int, name: str, unit: str) -> list[int]:
    """Broadcast an int to a per-stage list, or validate a list of the expected length."""
    values = [value] * length if isinstance(value, int) else list(value)
    if len(values) != length:
        raise ConfigError(
            f"'{name}' must have {length} entries (got {len(values)}: {values}).",
            "It is broadcast per resolution stage, so its length must match the topology.",
        )
    if any(count < 1 for count in values):
        raise ConfigError(
            f"'{name}' entries must all be >= 1 (got {values}).",
            f"Each stage must contain at least one {unit}; a zero count builds an invalid graph.",
        )
    return values


def as_kernel_list(kernel_sizes: int | list[Any], n_stages: int) -> list[Any]:
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


def build_unet_decoder(
    net: network.Network,
    *,
    n_stages: int,
    features_per_stage: list[int],
    strides: list[int | list[int]],
    kernel_list: list[Any],
    n_conv_decoder: list[int],
    num_classes: int,
    dim: int,
    conv_bias: bool,
    block_config: Callable[[int | list[int], int | list[int]], blocks.BlockConfig],
    build_head: Callable[[int], bool],
) -> None:
    """Append nnU-Net's shared ``UNetDecoder`` to ``net``.

    Decoder stage j runs coarsest-first (j=0 upsamples the bottleneck), exactly matching nnU-Net's
    ``UNetDecoder`` execution order: transpose conv (kernel = stride = the matching encoder stride),
    concat ``(transpose_output, encoder_skip)`` in that order, ``n_conv_decoder[j]`` conv blocks,
    then the stage's 1x1 seg head -- before moving to the next (finer) stage. Heads emit raw logits;
    ``build_head(j)`` decides which stages get one (deep supervision), and a head's bias is always
    True regardless of ``conv_bias``, like nnU-Net's seg layers.
    """
    for j in range(n_stages - 1):
        below_index = n_stages - 1 - j  # encoder feature feeding the transpose conv
        skip_index = n_stages - 2 - j  # encoder stage providing the skip connection
        below_channels = features_per_stage[below_index]
        skip_channels = features_per_stage[skip_index]
        transpose_stride: Any = strides[below_index]

        net.add_module(
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
        net.add_module(
            f"Skip_{j}",
            blocks.Concat(),
            in_branch=[f"up{j}", f"enc{skip_index}"],
            out_branch=[f"up{j}"],
        )
        net.add_module(
            f"Decoder_{j}",
            blocks.ConvBlock(
                in_channels=2 * skip_channels,
                out_channels=skip_channels,
                block_configs=[block_config(1, kernel_list[skip_index]) for _ in range(n_conv_decoder[j])],
                dim=dim,
            ),
            in_branch=[f"up{j}"],
            out_branch=[f"dec{j}"],
        )
        if build_head(j):
            net.add_module(
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
