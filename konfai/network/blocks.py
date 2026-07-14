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

"""Reusable network blocks and tensor graph helpers for KonfAI models."""

import ast
import importlib
import warnings
from collections.abc import Callable
from enum import Enum
from typing import Any

import torch

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None  # type: ignore[assignment]

from konfai.network import network
from konfai.utils.config import config
from konfai.utils.ITK import _require_simpleitk


class NormMode(Enum):
    """Enumeration of normalization layers supported by KonfAI blocks."""

    NONE = (0,)
    BATCH = 1
    INSTANCE = 2
    GROUP = 3
    LAYER = 4
    SYNCBATCH = 5
    INSTANCE_AFFINE = 6


def get_norm(norm_mode: Enum, channels: int, dim: int) -> torch.nn.Module | None:
    """Instantiate the normalization layer matching the requested mode."""
    if norm_mode == NormMode.BATCH:
        return get_torch_module("BatchNorm", dim=dim)(channels, affine=True, track_running_stats=True)
    if norm_mode == NormMode.INSTANCE:
        return get_torch_module("InstanceNorm", dim=dim)(channels, affine=False, track_running_stats=False)
    if norm_mode == NormMode.INSTANCE_AFFINE:
        return get_torch_module("InstanceNorm", dim=dim)(channels, affine=True, track_running_stats=False)
    if norm_mode == NormMode.SYNCBATCH:
        return torch.nn.SyncBatchNorm(channels, affine=True, track_running_stats=True)
    if norm_mode == NormMode.GROUP:
        return torch.nn.GroupNorm(num_groups=32, num_channels=channels)
    if norm_mode == NormMode.LAYER:
        return torch.nn.GroupNorm(num_groups=1, num_channels=channels)
    return None


class UpsampleMode(Enum):
    CONV_TRANSPOSE = (0,)
    UPSAMPLE = (1,)


class DownsampleMode(Enum):
    MAXPOOL = (0,)
    AVGPOOL = (1,)
    CONV_STRIDE = 2


def get_torch_module(name_fonction: str, dim: int | None = None) -> torch.nn.Module:
    """Return a dimensional PyTorch module class such as ``Conv2d`` or ``Conv3d``."""
    return getattr(
        importlib.import_module("torch.nn"),
        f"{name_fonction}" + (f"{dim}d" if dim is not None else ""),
    )


@config()
class BlockConfig:
    """Configuration object describing one convolutional block stage."""

    def __init__(
        self,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias=True,
        activation: str | Callable[[], torch.nn.Module] | None = "ReLU",
        norm_mode: str | NormMode | Callable[[int], torch.nn.Module] = "NONE",
    ) -> None:
        self.kernel_size = kernel_size
        self.bias = bias
        self.stride = stride
        self.padding = padding
        self.activation = activation
        self.norm_mode = norm_mode
        self.norm: NormMode | Callable[[int], torch.nn.Module] | None = None
        if isinstance(norm_mode, str):
            self.norm = NormMode[norm_mode]
        else:
            self.norm = norm_mode

    def get_conv(self, in_channels: int, out_channels: int, dim: int) -> torch.nn.Conv3d:
        return get_torch_module("Conv", dim=dim)(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=self.bias,
        )

    def get_norm(self, channels: int, dim: int) -> torch.nn.Module:
        if self.norm is None:
            return None
        return get_norm(self.norm, channels, dim) if isinstance(self.norm, NormMode) else self.norm(channels)

    def get_activation(self) -> torch.nn.Module:
        if self.activation is None:
            return None
        if isinstance(self.activation, str):
            return (
                get_torch_module(self.activation.split(";")[0])(
                    *[ast.literal_eval(value) for value in self.activation.split(";")[1:]]
                )
                if self.activation != "None"
                else torch.nn.Identity()
            )
        return self.activation()


class ConvBlock(network.ModuleArgsDict):
    """Sequential convolution, normalization, and activation block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        block_configs: list[BlockConfig],
        dim: int,
        alias: list[list[str]] = [[], [], []],
    ) -> None:
        super().__init__()
        for i, block_config in enumerate(block_configs):
            self.add_module(
                f"Conv_{i}",
                block_config.get_conv(in_channels, out_channels, dim),
                alias=alias[0],
            )
            norm = block_config.get_norm(out_channels, dim)
            if norm is not None:
                self.add_module(f"Norm_{i}", norm, alias=alias[1])
            activation = block_config.get_activation()
            if activation is not None:
                self.add_module(f"Activation_{i}", activation, alias=alias[2])
            in_channels = out_channels


class ResBlock(network.ModuleArgsDict):
    """Residual block with optional projection on the skip path."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        block_configs: list[BlockConfig],
        dim: int,
        alias: list[list[str]] = [[], [], [], [], []],
    ) -> None:
        super().__init__()
        for i, block_config in enumerate(block_configs):
            self.add_module(
                f"Conv_{i}",
                block_config.get_conv(in_channels, out_channels, dim),
                alias=alias[0],
            )
            norm = block_config.get_norm(out_channels, dim)
            if norm is not None:
                self.add_module(f"Norm_{i}", norm, alias=alias[1])
            activation = block_config.get_activation()
            if activation is not None:
                self.add_module(f"Activation_{i}", activation, alias=alias[2])

            if in_channels != out_channels:
                self.add_module(
                    "Conv_skip",
                    get_torch_module("Conv", dim)(
                        in_channels,
                        out_channels,
                        1,
                        block_config.stride,
                        bias=block_config.bias,
                    ),
                    alias=alias[3],
                    in_branch=[1],
                    out_branch=[1],
                )
                self.add_module(
                    "Norm_skip",
                    block_config.get_norm(out_channels, dim),
                    alias=alias[4],
                    in_branch=[1],
                    out_branch=[1],
                )
            in_channels = out_channels

        self.add_module("Add", Add(), in_branch=[0, 1])
        self.add_module(f"Norm_{i + 1}", torch.nn.ReLU(inplace=True))


class ResidualBlockD(network.ModuleArgsDict):
    """nnU-Net ResNet-D basic residual block, reproduced as a routed KonfAI graph.

    Module-for-module and in forward-execution order equal to
    ``dynamic_network_architectures.building_blocks.residual.BasicBlockD`` (the block the
    nnU-Net ``ResidualEncoderUNet`` stacks). It reproduces the ResNet-D structure of He et al.,
    *Bag of Tricks for Image Classification with CNNs* (CVPR 2019):

    * **Skip path first** (matching ``BasicBlockD.forward``, which evaluates ``self.skip(x)``
      before the main path): when the block is strided, the residual is downsampled with an
      ``AvgPool`` of kernel = stride (never a strided conv); when the channel count changes, it is
      then projected with a ``1x1`` conv whose ``bias`` is **always** ``False`` -- independent of
      ``conv_bias`` -- followed by a norm. When neither applies the skip is the identity (no extra
      module, the residual falls back to the block input branch).
    * **Main path** on branch ``0``: ``Conv(stride) -> Norm -> LeakyReLU -> Conv(stride 1) -> Norm``
      (the second conv carries no activation, exactly as ``BasicBlockD``).
    * The two paths are summed and a final ``LeakyReLU`` is applied.

    Executing the skip modules before the main-path convs is what makes the block transparent to
    the execution-order weight bridge (``konfai.utils.pretrained``): its weighted leaves fire in
    the same order as ``BasicBlockD`` (skip conv/norm, then conv1, then conv2).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dim: int,
        kernel_size: int | list[int] = 3,
        stride: int | list[int] = 1,
        conv_bias: bool = True,
        negative_slope: float = 1e-2,
        norm_mode: str | NormMode = "INSTANCE_AFFINE",
    ) -> None:
        super().__init__()
        stride_list = list(stride) if isinstance(stride, (list, tuple)) else [stride] * dim
        has_stride = any(value != 1 for value in stride_list)
        requires_projection = in_channels != out_channels
        norm = NormMode[norm_mode] if isinstance(norm_mode, str) else norm_mode
        padding = (
            [(value - 1) // 2 for value in kernel_size]
            if isinstance(kernel_size, (list, tuple))
            else (kernel_size - 1) // 2
        )

        # ----- Skip path (evaluated first, ResNet-D style) --------------------------------- #
        if has_stride:
            self.add_module(
                "SkipPool",
                get_torch_module("AvgPool", dim)(kernel_size=stride, stride=stride),
                in_branch=[0],
                out_branch=[1],
            )
        if requires_projection:
            self.add_module(
                "SkipConv",
                get_torch_module("Conv", dim)(
                    in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False
                ),
                in_branch=[1 if has_stride else 0],
                out_branch=[1],
            )
            self.add_module("SkipNorm", get_norm(norm, out_channels, dim), in_branch=[1], out_branch=[1])

        # ----- Main path ------------------------------------------------------------------- #
        self.add_module(
            "Conv1",
            get_torch_module("Conv", dim)(
                in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=conv_bias
            ),
        )
        self.add_module("Norm1", get_norm(norm, out_channels, dim))
        self.add_module("Nonlin1", torch.nn.LeakyReLU(negative_slope=negative_slope, inplace=False))
        self.add_module(
            "Conv2",
            get_torch_module("Conv", dim)(
                out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=conv_bias
            ),
        )
        self.add_module("Norm2", get_norm(norm, out_channels, dim))

        # ----- Residual sum + final activation --------------------------------------------- #
        self.add_module("Add", Add(), in_branch=[0, 1])
        self.add_module("Nonlin2", torch.nn.LeakyReLU(negative_slope=negative_slope, inplace=False))


class ResNetBasicBlock(network.ModuleArgsDict):
    """torchvision ResNet ``BasicBlock`` reproduced as a routed KonfAI graph (the ResNet-18/34 block).

    Module-for-module and in forward-execution order equal to
    ``torchvision.models.resnet.BasicBlock`` (the block the ``resnet18``/``resnet34`` encoders of
    ``segmentation_models_pytorch`` stack). It reproduces the post-activation residual block of He et al.,
    *Deep Residual Learning* (CVPR 2016):

    * **Main path first** (matching ``BasicBlock.forward``, which computes the residual branch before it
      evaluates ``self.downsample(x)``): on branch ``1`` ``Conv(stride) -> Norm -> ReLU -> Conv(stride 1)
      -> Norm`` (the second conv carries no activation), keeping the block input untouched on branch ``0``.
    * **Projection skip** on branch ``2``, evaluated *after* the main path (exactly as ``BasicBlock`` calls
      ``self.downsample(x)`` only after ``bn2``): a ``1x1`` ``Conv(stride) -> Norm``. It is built when the
      block is strided or changes channel count; otherwise the residual is the identity (block input).
    * The two paths are summed onto branch ``0`` and a final ``ReLU`` is applied.

    Evaluating the main-path convs before the downsample projection is what makes the block transparent to
    the execution-order weight bridge (``konfai.utils.pretrained``): its weighted leaves fire in the same
    order as ``BasicBlock`` (conv1, bn1, conv2, bn2, downsample.0, downsample.1). ``conv_bias`` defaults to
    ``False`` (torchvision/timm ResNets never bias their convs) and the norm is ``BatchNorm`` (running stats,
    so a checkpoint's BN buffers travel through the bridge and the ``eval()`` forward matches).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dim: int,
        stride: int | list[int] = 1,
        conv_bias: bool = False,
        downsample: bool | None = None,
        norm_mode: str | NormMode = "BATCH",
    ) -> None:
        super().__init__()
        stride_list = list(stride) if isinstance(stride, (list, tuple)) else [stride] * dim
        has_stride = any(value != 1 for value in stride_list)
        if downsample is None:
            downsample = has_stride or in_channels != out_channels
        norm = NormMode[norm_mode] if isinstance(norm_mode, str) else norm_mode

        # ----- Main path on branch 1 (block input preserved on branch 0) ------------------- #
        self.add_module(
            "Conv1",
            get_torch_module("Conv", dim)(
                in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=conv_bias
            ),
            in_branch=[0],
            out_branch=[1],
        )
        self.add_module("Norm1", get_norm(norm, out_channels, dim), in_branch=[1], out_branch=[1])
        self.add_module("Nonlin1", torch.nn.ReLU(inplace=False), in_branch=[1], out_branch=[1])
        self.add_module(
            "Conv2",
            get_torch_module("Conv", dim)(
                out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=conv_bias
            ),
            in_branch=[1],
            out_branch=[1],
        )
        self.add_module("Norm2", get_norm(norm, out_channels, dim), in_branch=[1], out_branch=[1])

        # ----- Projection skip on branch 2 (evaluated after the main path, ResNet style) --- #
        if downsample:
            self.add_module(
                "SkipConv",
                get_torch_module("Conv", dim)(
                    in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=conv_bias
                ),
                in_branch=[0],
                out_branch=[2],
            )
            self.add_module("SkipNorm", get_norm(norm, out_channels, dim), in_branch=[2], out_branch=[2])
            self.add_module("Add", Add(), in_branch=[1, 2], out_branch=[0])
        else:
            self.add_module("Add", Add(), in_branch=[1, 0], out_branch=[0])

        # ----- Final activation on branch 0 ------------------------------------------------ #
        self.add_module("Nonlin2", torch.nn.ReLU(inplace=False), in_branch=[0], out_branch=[0])


def _same_padding(kernel_size: int | list[int]) -> int | list[int]:
    """Same-padding for an odd kernel: ``(k - 1) // 2`` per axis (isotropic int or per-axis list)."""
    if isinstance(kernel_size, (list, tuple)):
        return [(value - 1) // 2 for value in kernel_size]
    return (kernel_size - 1) // 2


class ResidualStage(network.ModuleArgsDict):
    """One encoder resolution stage: a stack of ``n_blocks`` :class:`ResidualBlockD` as a single node.

    A generic, reusable building block for residual encoders (e.g. nnU-Net's ``ResidualEncoder``): the
    **first** block carries ``stride`` and the ``in_channels -> out_channels`` change (nnU-Net strided-conv
    downsampling), and the remaining ``n_blocks - 1`` blocks are stride 1 at ``out_channels``. The blocks
    are wired sequentially (each on branch ``0``), so the whole stage is a drop-in single node with one
    input and one output whose weighted leaves fire in the same order as the equivalent flat stack -- it
    stays transparent to the execution-order weight bridge (``konfai.utils.pretrained``).
    """

    def __init__(
        self,
        dim: int,
        in_channels: int,
        out_channels: int,
        n_blocks: int,
        stride: int | list[int] = 1,
        kernel_size: int | list[int] = 3,
        conv_bias: bool = True,
        negative_slope: float = 1e-2,
        norm_mode: str | NormMode = "INSTANCE_AFFINE",
    ) -> None:
        super().__init__()
        if n_blocks < 1:
            raise ValueError(f"ResidualStage needs at least one block, got n_blocks={n_blocks}.")
        for i in range(n_blocks):
            self.add_module(
                f"Block_{i}",
                ResidualBlockD(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    dim=dim,
                    kernel_size=kernel_size,
                    stride=stride if i == 0 else 1,
                    conv_bias=conv_bias,
                    negative_slope=negative_slope,
                    norm_mode=norm_mode,
                ),
            )


class DecoderStage(network.ModuleArgsDict):
    """One U-Net decoder resolution stage as a single two-input node: upsample, concat skip, conv block.

    A generic, reusable building block for U-Net decoders (nnU-Net's ``UNetDecoder`` shares it between the
    plain and residual encoders). It is a **two-input** node -- ``in_branch: [coarser, skip]`` -- that runs
    ``ConvTranspose(in_channels -> skip_channels, kernel = stride = upsample_stride)`` on the coarser input,
    concatenates the encoder ``skip`` (transpose output first, then skip), then a :class:`ConvBlock` of
    ``n_conv`` convs mapping ``2 * skip_channels -> skip_channels`` -- each Conv -> InstanceNorm(affine) ->
    LeakyReLU with same-padding. It yields one output. ``upsample_stride`` sets both the transpose kernel and
    stride so anisotropic plans (per-axis lists) upsample exactly the axes their encoder downsampled.
    """

    def __init__(
        self,
        dim: int,
        in_channels: int,
        skip_channels: int,
        n_conv: int,
        kernel_size: int | list[int] = 3,
        conv_bias: bool = True,
        negative_slope: float = 1e-2,
        upsample_stride: int | list[int] = 2,
        norm_mode: str | NormMode = "INSTANCE_AFFINE",
    ) -> None:
        super().__init__()
        if n_conv < 1:
            raise ValueError(f"DecoderStage needs at least one conv, got n_conv={n_conv}.")
        # Upsample the coarser input (branch 0) to the skip resolution/width; kernel = stride = upsample.
        self.add_module(
            "Up",
            get_torch_module("ConvTranspose", dim)(
                in_channels=in_channels,
                out_channels=skip_channels,
                kernel_size=upsample_stride,
                stride=upsample_stride,
                padding=0,
                bias=conv_bias,
            ),
            in_branch=[0],
            out_branch=[0],
        )
        # Concatenate the encoder skip (branch 1); transpose output FIRST, then skip -- nnU-Net order.
        self.add_module("Skip", Concat(), in_branch=[0, 1], out_branch=[0])
        # Conv block: 2 * skip_channels -> skip_channels, then skip_channels -> skip_channels for the rest.
        # ``kernel_size``/``padding`` pass through as Any so a per-axis (anisotropic) list is accepted.
        kernel_value: Any = kernel_size
        padding_value: Any = _same_padding(kernel_size)
        conv_config = BlockConfig(
            kernel_size=kernel_value,
            stride=1,
            padding=padding_value,
            bias=conv_bias,
            activation=f"LeakyReLU;{negative_slope}",
            norm_mode=norm_mode,
        )
        self.add_module(
            "Conv",
            ConvBlock(
                in_channels=2 * skip_channels,
                out_channels=skip_channels,
                block_configs=[conv_config for _ in range(n_conv)],
                dim=dim,
            ),
            in_branch=[0],
            out_branch=[0],
        )


class ResNetStage(network.ModuleArgsDict):
    """One torchvision ResNet encoder stage: a stack of ``n_blocks`` :class:`ResNetBasicBlock` as a single node.

    A generic, reusable building block for the ResNet-18/34 encoders that ``segmentation_models_pytorch``
    stacks under its U-Net / UNet++ decoders (each is torchvision's ``ResNet`` ``layer1..layer4``). The
    **first** block carries the stage ``stride`` and the ``in_channels -> out_channels`` change -- its ``1x1``
    projection ``downsample`` skip is built automatically when the stride or the channel count changes -- and
    the remaining ``n_blocks - 1`` blocks are stride 1 at ``out_channels`` with identity skips. The blocks run
    sequentially on branch ``0``, so the whole stage is a drop-in single node with one input and one output
    whose weighted leaves fire in the same order as the equivalent flat ``BasicBlock`` stack: it stays
    transparent to the execution-order weight bridge (``konfai.utils.pretrained``). ``conv_bias`` defaults to
    ``False`` and the norm to ``BatchNorm`` -- the torchvision/timm ResNet convention.
    """

    def __init__(
        self,
        dim: int,
        in_channels: int,
        out_channels: int,
        n_blocks: int,
        stride: int | list[int] = 1,
        conv_bias: bool = False,
        norm_mode: str | NormMode = "BATCH",
    ) -> None:
        super().__init__()
        if n_blocks < 1:
            raise ValueError(f"ResNetStage needs at least one block, got n_blocks={n_blocks}.")
        for i in range(n_blocks):
            self.add_module(
                f"Block_{i}",
                ResNetBasicBlock(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    dim=dim,
                    stride=stride if i == 0 else 1,
                    conv_bias=conv_bias,
                    norm_mode=norm_mode,
                ),
            )


class UNetPlusPlusNode(network.ModuleArgsDict):
    """One UNet++ dense-decoder node as a single multi-input node: upsample, dense concat, then Conv-Norm-ReLU.

    A generic, reusable building block for the nested (dense) UNet++ decoder of
    ``segmentation_models_pytorch`` (``UnetPlusPlusDecoder``). It is a **multi-input** node --
    ``in_branch: [coarser, skip_0, skip_1, ...]`` -- that reproduces one grid node ``x_{d}_{l}``:

    * ``Upsample(scale_factor=2, mode='nearest')`` on the shallower-column predecessor (branch ``0``);
    * a :class:`Concat` of the upsampled feature FIRST, then every same-resolution dense skip and the matching
      encoder skip (smp order), when any skip is provided;
    * a :class:`ConvBlock` of ``n_conv`` ``Conv -> BatchNorm -> ReLU`` blocks (smp's ``Conv2dReLU``) mapping the
      concatenated width ``up_channels + sum(skip_channels)`` to ``out_channels``.

    ``skip_channels`` is the list of per-skip channel widths in ``in_branch`` order: its length wires the
    concat (``n_skip + 1`` inputs) and its sum fixes the conv's input width, so the node self-describes the
    dense fusion. An empty ``skip_channels`` (the final full-resolution ``x_0_depth`` node) drops the concat and
    convolves the upsampled feature alone. The weighted leaves fire in ``ConvBlock`` order, so the node stays
    transparent to the execution-order weight bridge (``konfai.utils.pretrained``).
    """

    def __init__(
        self,
        dim: int,
        up_channels: int,
        skip_channels: list[int],
        out_channels: int,
        n_conv: int = 2,
        kernel_size: int | list[int] = 3,
        conv_bias: bool = False,
        norm_mode: str | NormMode = "BATCH",
    ) -> None:
        super().__init__()
        if n_conv < 1:
            raise ValueError(f"UNetPlusPlusNode needs at least one conv, got n_conv={n_conv}.")
        # Branch 0 = the shallower-column predecessor to upsample; branches 1.. = the dense / encoder skips.
        self.add_module(
            "Up",
            torch.nn.Upsample(scale_factor=2.0, mode="nearest"),
            in_branch=[0],
            out_branch=[0],
        )
        conv_in = up_channels
        if skip_channels:
            # Nearest-upsampled feature FIRST, then every same-resolution dense skip and the encoder skip.
            self.add_module(
                "Cat",
                Concat(),
                in_branch=list(range(len(skip_channels) + 1)),
                out_branch=[0],
            )
            conv_in = up_channels + sum(skip_channels)
        kernel_value: Any = kernel_size
        padding_value: Any = _same_padding(kernel_size)
        conv_config = BlockConfig(
            kernel_size=kernel_value,
            stride=1,
            padding=padding_value,
            bias=conv_bias,
            activation="ReLU",
            norm_mode=norm_mode,
        )
        self.add_module(
            "Conv",
            ConvBlock(
                in_channels=conv_in,
                out_channels=out_channels,
                block_configs=[conv_config for _ in range(n_conv)],
                dim=dim,
            ),
            in_branch=[0],
            out_branch=[0],
        )


def downsample(in_channels: int, out_channels: int, downsample_mode: DownsampleMode, dim: int) -> torch.nn.Module:
    """Return the downsampling module matching the requested strategy."""
    if downsample_mode == DownsampleMode.MAXPOOL:
        return get_torch_module("MaxPool", dim=dim)(2)
    if downsample_mode == DownsampleMode.AVGPOOL:
        return get_torch_module("AvgPool", dim=dim)(2)
    if downsample_mode == DownsampleMode.CONV_STRIDE:
        return get_torch_module("Conv", dim)(in_channels, out_channels, kernel_size=2, stride=2, padding=0)


def upsample(
    in_channels: int,
    out_channels: int,
    upsample_mode: UpsampleMode,
    dim: int,
    kernel_size: int | list[int] = 2,
    stride: int | list[int] = 2,
):
    """Return the upsampling module matching the requested strategy."""
    if upsample_mode == UpsampleMode.CONV_TRANSPOSE:
        return get_torch_module("ConvTranspose", dim=dim)(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
        )
    else:
        if dim == 3:
            upsample_method = "trilinear"
        if dim == 2:
            upsample_method = "bilinear"
        if dim == 1:
            upsample_method = "linear"
        return torch.nn.Upsample(scale_factor=2, mode=upsample_method.lower(), align_corners=False)


class Unsqueeze(torch.nn.Module):
    def __init__(self, dim: int = 0):
        super().__init__()
        self.dim = dim

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.unsqueeze(tensor, self.dim)

    def extra_repr(self):
        return f"dim={self.dim}"


class Permute(torch.nn.Module):
    def __init__(self, dims: list[int]):
        super().__init__()
        self.dims = dims

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.permute(tensor, self.dims)

    def extra_repr(self):
        return f"dims={self.dims}"


class ToChannels(Permute):
    def __init__(self, dim: int):
        super().__init__([0, dim + 1, *[i + 1 for i in range(dim)]])


class ToFeatures(Permute):
    def __init__(self, dim: int):
        super().__init__([0, *[i + 2 for i in range(dim)], 1])


class Add(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, *tensor: torch.Tensor) -> torch.Tensor:
        return torch.sum(torch.stack(tensor), dim=0)


class Multiply(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, *tensor: torch.Tensor) -> torch.Tensor:
        return torch.mul(*tensor)


class ClipNormalize(torch.nn.Module):
    """Clip to a stored intensity range, then standardize: ``(clamp(x, min, max) - mean) / std``.

    The four scalars are buffers restored from the checkpoint, not config values, so a
    per-checkpoint input normalization (e.g. a CT window baked into a trained model) travels
    with its weights. It has no learnable parameters. Used as a declarative model's first node
    so that normalization stays part of the model rather than a separate preprocessing step.
    """

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("clip_min", torch.empty(1))
        self.register_buffer("clip_max", torch.empty(1))
        self.register_buffer("mean", torch.empty(1))
        self.register_buffer("std", torch.empty(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (torch.clamp(x, self.clip_min, self.clip_max) - self.mean) / self.std


class Concat(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, *tensor: torch.Tensor) -> torch.Tensor:
        return torch.cat(tensor, dim=1)


# ---------------------------------------------------------------------------
# Debug-only blocks
#
# These pass the tensor through unchanged (or stop the graph) and exist only to
# inspect intermediate activations while developing a model. They are inert
# unless explicitly wired into a graph via `add_module`, have side effects
# (stdout / disk / raising), and must NOT be left in a production model.
# ---------------------------------------------------------------------------
class Print(torch.nn.Module):
    """Debug block: print the tensor shape and pass it through unchanged."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        print(tensor.shape)
        return tensor


class Write(torch.nn.Module):
    """Debug block: write the first channel of the first sample to disk as a volume.

    The destination path is explicit (no silent hardcoded location) and each
    call warns, since writing to disk inside a forward pass is a development-only
    side effect.
    """

    def __init__(self, path: str = "./Data.mha") -> None:
        super().__init__()
        self.path = path

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        warnings.warn(
            f"Debug `Write` block is writing a volume to '{self.path}' during forward; "
            "remove it from production models.",
            stacklevel=2,
        )
        _require_simpleitk()
        sitk.WriteImage(sitk.GetImageFromArray(tensor.clone()[0][0].cpu().numpy()), self.path)
        return tensor


class Exit(torch.nn.Module):
    """Debug block: stop the forward pass by raising, to halt at a chosen point."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("The debug Exit block was executed.")


class Detach(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.detach()


class Negative(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return -tensor


class GetShape(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.tensor(tensor.shape)


class ArgMax(torch.nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.argmax(tensor, dim=self.dim).unsqueeze(self.dim)


class Select(torch.nn.Module):
    def __init__(self, slices: list[slice]) -> None:
        super().__init__()
        self.slices = tuple(slices)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        result = tensor[self.slices]
        for i in reversed(range(len(result.shape))):
            if result.shape[i] == 1:
                result = result.squeeze(dim=i)
        return result


class NormalNoise(torch.nn.Module):
    def __init__(self, dim: int | None = None) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.dim is not None:
            return torch.randn(self.dim).to(tensor.device)
        else:
            return torch.randn_like(tensor).to(tensor.device)


class Const(torch.nn.Module):
    def __init__(self, shape: list[int], std: float) -> None:
        super().__init__()
        self.noise = torch.nn.parameter.Parameter(torch.randn(shape) * std)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.noise.to(tensor.device)


class Subset(torch.nn.Module):
    def __init__(self, slices: list[slice]):
        super().__init__()
        self.slices = [slice(None, None), slice(None, None), *slices]

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor[self.slices]


class View(torch.nn.Module):
    def __init__(self, size: list[int]):
        super().__init__()
        self.size = size

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.view(self.size)


class LatentDistribution(network.ModuleArgsDict):
    class LatentDistributionLinear(torch.nn.Module):
        def __init__(self, shape: list[int], latent_dim: int) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(torch.prod(torch.tensor(shape)), latent_dim)

        def forward(self, tensor: torch.Tensor) -> torch.Tensor:
            return torch.unsqueeze(self.linear(tensor), 1)

    class LatentDistributionDecoder(torch.nn.Module):
        def __init__(self, shape: list[int], latent_dim: int) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(latent_dim, torch.prod(torch.tensor(shape)))
            self.shape = shape

        def forward(self, tensor: torch.Tensor) -> torch.Tensor:
            return self.linear(tensor).view(-1, *[int(i) for i in self.shape])

    class LatentDistributionZ(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, mu: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
            return torch.exp(log_std / 2) * torch.randn_like(mu) + mu

    def __init__(self, shape: list[int], latent_dim: int) -> None:
        super().__init__()
        self.add_module("Flatten", torch.nn.Flatten(1))
        self.add_module(
            "mu",
            LatentDistribution.LatentDistributionLinear(shape, latent_dim),
            out_branch=[1],
        )
        self.add_module(
            "log_std",
            LatentDistribution.LatentDistributionLinear(shape, latent_dim),
            out_branch=[2],
        )

        self.add_module(
            "z",
            LatentDistribution.LatentDistributionZ(),
            in_branch=[1, 2],
            out_branch=[3],
        )
        self.add_module("Concat", Concat(), in_branch=[1, 2, 3])
        self.add_module(
            "DecoderInput",
            LatentDistribution.LatentDistributionDecoder(shape, latent_dim),
            in_branch=[3],
        )


class Attention(network.ModuleArgsDict):
    def __init__(self, f_g: int, f_l: int, f_int: int, dim: int):
        super().__init__()
        self.add_module(
            "W_x",
            get_torch_module("Conv", dim=dim)(in_channels=f_l, out_channels=f_int, kernel_size=1, stride=2, padding=0),
            in_branch=[0],
            out_branch=[0],
        )
        self.add_module(
            "W_g",
            get_torch_module("Conv", dim=dim)(in_channels=f_g, out_channels=f_int, kernel_size=1, stride=1, padding=0),
            in_branch=[1],
            out_branch=[1],
        )
        self.add_module("Add", Add(), in_branch=[0, 1])
        self.add_module("ReLU", torch.nn.ReLU(inplace=True))
        self.add_module(
            "Conv",
            get_torch_module("Conv", dim=dim)(in_channels=f_int, out_channels=1, kernel_size=1, stride=1, padding=0),
        )
        self.add_module("Sigmoid", torch.nn.Sigmoid())
        self.add_module("Upsample", torch.nn.Upsample(scale_factor=2))
        self.add_module("Multiply", Multiply(), in_branch=[2, 0])


class PositionalEmbedding(torch.nn.Module):
    """Add a learnable positional embedding to a token sequence.

    The input is a token sequence shaped ``[B, num_tokens, embedding_dim]`` (batch, sequence, feature),
    the layout produced by a patch embedding. A single learnable parameter of shape
    ``[1, num_tokens, embedding_dim]`` is broadcast over the batch and added, giving every token position a
    trainable offset. This is the ``learnable`` positional encoding of the Vision Transformer; the parameter
    is registered directly on the module so it is a self-contained, single-purpose building block.
    """

    def __init__(self, num_tokens: int, embedding_dim: int) -> None:
        super().__init__()
        self.positional_embedding = torch.nn.Parameter(torch.zeros(1, num_tokens, embedding_dim))

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor + self.positional_embedding

    def extra_repr(self) -> str:
        _, num_tokens, embedding_dim = self.positional_embedding.shape
        return f"num_tokens={num_tokens}, embedding_dim={embedding_dim}"


class MultiHeadSelfAttention(torch.nn.Module):
    """Multi-head self-attention over a token sequence (the Transformer encoder attention).

    Mirrors the self-attention of MONAI's ``SABlock``/ViT in its default configuration: a single packed
    ``qkv`` linear projects the ``[B, num_tokens, hidden_size]`` sequence into queries, keys and values,
    scaled dot-product attention is computed per head with scale ``head_dim ** -0.5``, and an ``out_proj``
    linear mixes the concatenated heads back to ``hidden_size``. ``qkv_bias`` toggles the bias of the packed
    projection (the ViT default is ``False``); ``out_proj`` always carries a bias. Both projections are child
    ``Linear`` leaves, so the module is transparent to execution-order weight transfer.
    """

    def __init__(self, hidden_size: int, num_heads: int, qkv_bias: bool = False) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads}).")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = torch.nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)
        self.out_proj = torch.nn.Linear(hidden_size, hidden_size)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, tokens, hidden = tensor.shape
        qkv = self.qkv(tensor).reshape(batch, tokens, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        attention = ((query @ key.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        output = (attention @ value).permute(0, 2, 1, 3).reshape(batch, tokens, hidden)
        return self.out_proj(output)

    def extra_repr(self) -> str:
        return f"num_heads={self.num_heads}, head_dim={self.head_dim}"
