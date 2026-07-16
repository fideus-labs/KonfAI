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

"""MIND -- Modality-Independent Neighbourhood Descriptor (Heinrich et al. 2012).

A hand-crafted, parameter-free feature extractor: fixed neighbourhood-shift convolution
kernels turn an image into a self-similarity descriptor that is robust across modalities.
It carries no learnable weights, so it is used frozen as a feature backbone for
feature/perceptual/registration losses (the descriptor the IMPACT loss consumes). Reference
it as ``classpath: features.mind.MIND``; attach a loss to its ``Descriptor`` output.

The descriptor maths are ported verbatim from the reference MIND implementation
(github.com/vboussot/ImpactLoss ``Data/Models/builds/Mind``); this module returns the
descriptor tensor directly instead of the reference's one-element list.
"""

import torch
import torch.nn.functional as F
from konfai.data.patching import ModelPatch
from konfai.network import network
from konfai.utils.config import config
from konfai.utils.errors import ConfigError


def _min_max_normalize(x: torch.Tensor) -> torch.Tensor:
    vmin = x.min()
    return (x - vmin) / (x.max() - vmin + 1e-6)


def _pdist_squared(x: torch.Tensor) -> torch.Tensor:
    xx = (x**2).sum(dim=1).unsqueeze(2)
    yy = xx.permute(0, 2, 1)
    dist = xx + yy - 2.0 * torch.bmm(x.permute(0, 2, 1), x)
    dist[dist != dist] = 0
    return torch.clamp(dist, min=0.0)


class _MindDescriptor(torch.nn.Module):
    """Compute the MIND self-similarity descriptor in 2D or 3D with fixed shift kernels."""

    def __init__(self, dim: int, radius: int, dilation: int) -> None:
        super().__init__()
        self.dim = dim
        self.radius = radius
        self.dilation = dilation

        if dim == 3:
            neighbourhood = torch.tensor([[0, 1, 1], [1, 1, 0], [1, 0, 1], [1, 1, 2], [2, 1, 1], [1, 2, 1]]).long()
            size = 6
        else:
            neighbourhood = torch.tensor([[1, 0], [0, 1], [1, 2], [2, 1]]).long()
            size = 4

        dist = _pdist_squared(neighbourhood.t().unsqueeze(0).float()).squeeze(0)
        grid_x, grid_y = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
        mask = (grid_x > grid_y).view(-1) & (dist == 2).view(-1)
        idx1 = neighbourhood.unsqueeze(1).repeat(1, size, 1).view(-1, dim)[mask, :]
        idx2 = neighbourhood.unsqueeze(0).repeat(size, 1, 1).view(-1, dim)[mask, :]
        num_pairs = idx1.shape[0]

        conv_module = torch.nn.Conv3d if dim == 3 else torch.nn.Conv2d
        kernel_shape = (num_pairs, 1) + (3,) * dim
        stride = 3**dim
        mshift1 = torch.zeros(*kernel_shape)
        mshift2 = torch.zeros(*kernel_shape)
        flat1 = torch.arange(num_pairs) * stride + sum(idx1[:, d] * 3 ** (dim - 1 - d) for d in range(dim))
        flat2 = torch.arange(num_pairs) * stride + sum(idx2[:, d] * 3 ** (dim - 1 - d) for d in range(dim))
        mshift1.view(-1)[flat1] = 1
        mshift2.view(-1)[flat2] = 1

        self.conv1 = conv_module(1, num_pairs, kernel_size=3, stride=1, padding=0, bias=False, dilation=dilation)
        self.conv2 = conv_module(1, num_pairs, kernel_size=3, stride=1, padding=0, bias=False, dilation=dilation)
        self.conv1.weight = torch.nn.Parameter(mshift1, requires_grad=False)
        self.conv2.weight = torch.nn.Parameter(mshift2, requires_grad=False)

        pad_module = torch.nn.ReplicationPad3d if dim == 3 else torch.nn.ReplicationPad2d
        self.rpad1 = pad_module(dilation)
        self.rpad2 = pad_module(radius)
        self.avg_pool = F.avg_pool3d if dim == 3 else F.avg_pool2d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _min_max_normalize(x)
        ssd = self.avg_pool(
            self.rpad2((self.conv1(self.rpad1(x)) - self.conv2(self.rpad1(x))) ** 2),
            self.radius * 2 + 1,
            stride=1,
        )
        mind = ssd - torch.min(ssd, dim=1, keepdim=True)[0]
        mind_var = torch.mean(mind, dim=1, keepdim=True)
        mind_var_mean = mind_var.mean()
        mind_var = torch.clamp(mind_var, mind_var_mean * 0.001, mind_var_mean * 1000)
        # A constant patch gives mind_var == 0 everywhere (the clamp bounds are then both 0),
        # and 0/0 would send NaN into the loss; floor the denominator instead.
        mind_var = mind_var.clamp_min(torch.finfo(mind_var.dtype).eps)
        mind = mind / mind_var
        return torch.exp(-mind)


@config()
class MIND(network.Network):
    """MIND descriptor as a frozen KonfAI feature model (2D or 3D).

    ``load`` never re-initialises: the shift kernels are fixed by construction, and the
    trainer's ``load(init=True)`` would overwrite them with ``init_type`` noise.
    """

    def __init__(
        self,
        optimizer: network.OptimizerLoader = network.OptimizerLoader(),
        schedulers: dict[str, network.LRSchedulersLoader] = {
            "default|ReduceLROnPlateau": network.LRSchedulersLoader(0)
        },
        patch: ModelPatch | None = None,
        outputs_criterions: dict[str, network.TargetCriterionsLoader] = {"default": network.TargetCriterionsLoader()},
        dim: int = 3,
        radius: int = 2,
        dilation: int = 2,
    ) -> None:
        if dim not in (2, 3):
            raise ConfigError(f"MIND supports dim 2 or 3, got dim={dim}.")
        super().__init__(
            in_channels=1,
            optimizer=optimizer,
            schedulers=schedulers,
            patch=patch,
            outputs_criterions=outputs_criterions,
            dim=dim,
        )
        self.add_module("Descriptor", _MindDescriptor(dim, radius, dilation))

    def load(
        self,
        state_dict: dict,
        init: bool = True,
        ema: bool = False,
        override_lr: float | None = None,
    ):
        del init  # the descriptor's shift kernels are fixed by construction
        super().load(state_dict, init=False, ema=ema, override_lr=override_lr)
