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

import torch
import torch.nn.functional as F
from konfai.data.patching import ModelPatch
from konfai.network import blocks, network

# Reference ConvNeXt checkpoint URLs and the (depths, dims) that reproduce them:
# convnext_tiny_1k:  https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth
#   depths=[3, 3, 9, 3], dims=[96, 192, 384, 768]
# convnext_small_1k: https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth
#   depths=[3, 3, 27, 3], dims=[96, 192, 384, 768]
# convnext_base_1k:  https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth
#   depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024]
# convnext_large_1k: https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth
#   depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536]
# convnext_tiny_22k:   https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth
# convnext_small_22k:  https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth
# convnext_base_22k:   https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth
# convnext_large_22k:  https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth
# convnext_xlarge_22k: https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth
#   depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048]


class LayerNorm(torch.nn.Module):
    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        data_format: str = "channels_last",
    ):
        super().__init__()
        self.weight = torch.nn.parameter.Parameter(torch.ones(normalized_shape))
        self.bias = torch.nn.parameter.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            if len(x.shape) == 3:
                x = self.weight[:, None] * x + self.bias[:, None]
            elif len(x.shape) == 4:
                x = self.weight[:, None, None] * x + self.bias[:, None, None]
            else:
                x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            return x

    def extra_repr(self):
        return f"normalized_shape={self.normalized_shape}, eps={self.eps}, data_format={self.data_format})"


class DropPath(torch.nn.Module):
    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob, 3)}"


class LayerScaler(torch.nn.Module):
    def __init__(self, init_value: float, dimensions: int, dim: int = 2):
        super().__init__()
        self.init_value = init_value
        # Per-channel scale broadcast over `dim` spatial axes: (C, 1, ..., 1). Sizing it for `dim`
        # keeps 2-D behaviour (C, 1, 1) identical while making 3-D (C, 1, 1, 1) broadcast against
        # [B, C, D, H, W] instead of pairing C against the depth axis and crashing when D != C.
        self.gamma = torch.nn.Parameter(torch.ones((dimensions,) + (1,) * dim) * init_value)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.gamma * tensor

    def extra_repr(self):
        return f"init_value={self.init_value}"


class BottleNeckBlock(network.ModuleArgsDict):
    def __init__(self, features: int, drop_p: float, layer_scaler_init_value: float, dim: int):
        super().__init__()
        self.add_module(
            "Conv_0",
            blocks.get_torch_module("Conv", dim)(features, features, kernel_size=7, padding=3, groups=features),
            alias=["dwconv"],
        )
        self.add_module("ToFeatures", blocks.ToFeatures(dim))
        self.add_module("LayerNorm", LayerNorm(features, eps=1e-6), alias=["norm"])
        self.add_module("Linear_1", torch.nn.Linear(features, features * 4), alias=["pwconv1"])
        self.add_module("GELU", torch.nn.GELU())
        self.add_module("Linear_2", torch.nn.Linear(features * 4, features), alias=["pwconv2"])
        self.add_module("ToChannels", blocks.ToChannels(dim))
        self.add_module(
            "LayerScaler",
            LayerScaler(init_value=layer_scaler_init_value, dimensions=features, dim=dim),
            alias=[""],
        )
        self.add_module("StochasticDepth", DropPath(drop_p))
        self.add_module("Residual", blocks.Add(), in_branch=[0, 1])


class DownSample(network.ModuleArgsDict):
    def __init__(self, in_features: int, out_features: int, dim: int):
        super().__init__()
        self.add_module(
            "LayerNorm",
            LayerNorm(in_features, eps=1e-6, data_format="channels_first"),
            alias=["0"],
        )
        self.add_module(
            "Conv",
            blocks.get_torch_module("Conv", dim)(in_features, out_features, kernel_size=2, stride=2),
            alias=["1"],
        )


class ConvNexStage(network.ModuleArgsDict):
    def __init__(self, features: int, depth: int, drop_p: list[float], dim: int):
        super().__init__()
        for i in range(depth):
            self.add_module(
                f"BottleNeckBlock_{i}",
                BottleNeckBlock(
                    features=features,
                    drop_p=drop_p[i],
                    layer_scaler_init_value=1e-6,
                    dim=dim,
                ),
                alias=[f"{i}"],
            )


class ConvNextStem(network.ModuleArgsDict):
    def __init__(self, in_features: int, out_features: int, dim: int):
        super().__init__()
        self.add_module(
            "Conv",
            blocks.get_torch_module("Conv", dim)(in_features, out_features, kernel_size=4, stride=4),
            alias=["0"],
        )
        self.add_module(
            "LayerNorm",
            LayerNorm(out_features, eps=1e-6, data_format="channels_first"),
            alias=["1"],
        )


class ConvNextEncoder(network.ModuleArgsDict):
    def __init__(
        self,
        in_channels: int,
        depths: list[int],
        widths: list[int],
        drop_p: float,
        dim: int,
    ):
        super().__init__()
        self.add_module(
            "ConvNextStem",
            ConvNextStem(in_channels, widths[0], dim=dim),
            alias=["downsample_layers.0"],
        )

        drop_probs = [x.item() for x in torch.linspace(0, drop_p, sum(depths))]
        self.add_module(
            "ConvNexStage_0",
            ConvNexStage(
                features=widths[0],
                depth=depths[0],
                drop_p=drop_probs[: depths[0]],
                dim=dim,
            ),
            alias=["stages.0"],
        )

        for i, (in_features, out_features) in enumerate(list(itertools.pairwise(widths))):
            self.add_module(
                f"DownSample_{i + 1}",
                DownSample(in_features=in_features, out_features=out_features, dim=dim),
                alias=[f"downsample_layers.{i + 1}"],
            )
            self.add_module(
                f"ConvNexStage_{i + 1}",
                ConvNexStage(
                    features=out_features,
                    depth=depths[i + 1],
                    drop_p=drop_probs[sum(depths[: i + 1]) : sum(depths[: i + 2])],
                    dim=dim,
                ),
                alias=[f"stages.{i + 1}"],
            )


class Head(network.ModuleArgsDict):
    def __init__(self, in_features: int, num_classes: list[int], dim: int) -> None:
        super().__init__()
        self.add_module(
            "AdaptiveAvgPool",
            blocks.get_torch_module("AdaptiveAvgPool", dim)(tuple([1] * dim)),
        )
        self.add_module("Flatten", torch.nn.Flatten(1))
        self.add_module("LayerNorm", torch.nn.LayerNorm(in_features, eps=1e-6), alias=["norm"])

        for i, nb_classe in enumerate(num_classes):
            self.add_module(
                f"Linear_{i}",
                torch.nn.Linear(in_features, nb_classe),
                pretrained=False,
                alias=["head"],
                out_branch=[i + 1],
            )
            self.add_module(
                f"Unsqueeze_{i}",
                blocks.Unsqueeze(2),
                in_branch=[i + 1],
                out_branch=[-1],
            )


class ConvNeXt(network.Network):
    def __init__(
        self,
        optimizer: network.OptimizerLoader = network.OptimizerLoader(),
        schedulers: dict[str, network.LRSchedulersLoader] = {
            "default|ReduceLROnPlateau": network.LRSchedulersLoader(0)
        },
        outputs_criterions: dict[str, network.TargetCriterionsLoader] = {"Head": network.TargetCriterionsLoader()},
        patch: ModelPatch = ModelPatch(),
        dim: int = 3,
        in_channels: int = 1,
        depths: list[int] = [3, 3, 27, 3],
        widths: list[int] = [128, 256, 512, 1024],
        drop_p: float = 0.1,
        num_classes: list[int] = [4, 7],
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
            "ConvNextEncoder",
            ConvNextEncoder(
                in_channels=in_channels,
                depths=depths,
                widths=widths,
                drop_p=drop_p,
                dim=dim,
            ),
        )
        self.add_module("Head", Head(in_features=widths[-1], num_classes=num_classes, dim=dim))
