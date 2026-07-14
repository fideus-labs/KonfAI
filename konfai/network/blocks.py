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
