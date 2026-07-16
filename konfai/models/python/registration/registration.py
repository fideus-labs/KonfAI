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

import math

import torch
import torch.nn.functional as F
from konfai.models.python.segmentation import UNet
from konfai.network import blocks, network
from konfai.utils.errors import ConfigError
from torch.nn.parameter import Parameter


class VoxelMorph(network.Network):
    def __init__(
        self,
        optimizer: network.OptimizerLoader = network.OptimizerLoader(),
        schedulers: dict[str, network.LRSchedulersLoader] = {
            "default|ReduceLROnPlateau": network.LRSchedulersLoader(0)
        },
        outputs_criterions: dict[str, network.TargetCriterionsLoader] = {"default": network.TargetCriterionsLoader()},
        dim: int = 3,
        channels: list[int] = [4, 16, 32, 32, 32],
        block_config: blocks.BlockConfig = blocks.BlockConfig(),
        nb_conv_per_stage: int = 2,
        downsample_mode: str = "MAXPOOL",
        upsample_mode: str = "CONV_TRANSPOSE",
        attention: bool = False,
        shape: list[int] = [192, 192, 192],
        int_steps: int = 7,
        int_downsize: int = 2,
        nb_batch_per_step: int = 1,
        rigid: bool = False,
    ):
        if dim not in (2, 3):
            raise ConfigError(f"VoxelMorph supports dim 2 or 3, got dim={dim}.")
        if len(shape) != dim:
            raise ConfigError(f"VoxelMorph 'shape' must have {dim} spatial dimensions, got {shape}.")
        super().__init__(
            in_channels=channels[0],
            optimizer=optimizer,
            schedulers=schedulers,
            outputs_criterions=outputs_criterions,
            dim=dim,
            nb_batch_per_step=nb_batch_per_step,
        )
        self.add_module("Concat", blocks.Concat(), in_branch=[0, 1], out_branch=["input_concat"])
        self.add_module(
            "UNetBlock_0",
            UNet.UNetBlock(
                channels,
                nb_conv_per_stage,
                block_config,
                downsample_mode=blocks.DownsampleMode[downsample_mode],
                upsample_mode=blocks.UpsampleMode[upsample_mode],
                attention=attention,
                block=blocks.ConvBlock,
                nb_class=0,
                dim=dim,
            ),
            in_branch=["input_concat"],
            out_branch=["unet"],
        )

        if rigid:
            self.add_module(
                "Flow",
                Rigid(channels[1], shape, dim),
                in_branch=["unet"],
                out_branch=["pos_flow"],
            )
        else:
            self.add_module(
                "Flow",
                Flow(channels[1], int_steps, int_downsize, shape, dim),
                in_branch=["unet"],
                out_branch=["pos_flow"],
            )
        self.add_module(
            "MovingImageResample",
            SpatialTransformer(shape, rigid=rigid),
            in_branch=[1, "pos_flow"],
            out_branch=["moving_image_resample"],
        )


class Flow(network.ModuleArgsDict):
    def __init__(
        self,
        in_channels: int,
        int_steps: int,
        int_downsize: int,
        shape: list[int],
        dim: int,
    ) -> None:
        super().__init__()
        self.add_module(
            "Head",
            blocks.get_torch_module("Conv", dim)(
                in_channels=in_channels,
                out_channels=dim,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )
        self["Head"].weight = Parameter(torch.distributions.Normal(0, 1e-5).sample(self["Head"].weight.shape))
        self["Head"].bias = Parameter(torch.zeros(self["Head"].bias.shape))

        if int_steps > 0 and int_downsize > 1:
            self.add_module("DownSample", ResizeTransform(int_downsize))

        if int_steps > 0:
            self.add_module(
                "Integrate_pos_flow",
                VecInt([int(extent / int_downsize) for extent in shape], int_steps),
            )

        if int_steps > 0 and int_downsize > 1:
            self.add_module("Upsample_pos_flow", ResizeTransform(1 / int_downsize))


def rigid_affine(parameters: torch.Tensor, dim: int) -> torch.Tensor:
    """Build a ``[B, dim, dim + 1]`` affine from ``[B, dim * (dim + 1) / 2]`` rigid parameters.

    The leading ``dim * (dim - 1) / 2`` parameters are rotation generators, the rest the translation.
    The rotation is the matrix exponential of the skew-symmetric matrix they fill: ``exp`` maps
    ``so(n)`` onto ``SO(n)``, so every parameter value is a proper rotation whatever the optimizer
    does to it -- no re-orthogonalisation, no gimbal lock -- and zero is the identity, which is what
    ``Rigid.init`` starts from.
    """
    n_angles = dim * (dim - 1) // 2
    angles, translation = parameters[:, :n_angles], parameters[:, n_angles:]
    skew = torch.zeros((parameters.shape[0], dim, dim), device=parameters.device, dtype=parameters.dtype)
    rows, cols = torch.triu_indices(dim, dim, offset=1, device=parameters.device)
    skew[:, rows, cols] = angles
    rotation = torch.matrix_exp(skew - skew.transpose(1, 2))
    return torch.cat([rotation, translation.unsqueeze(-1)], dim=-1)


class Rigid(network.ModuleArgsDict):
    """Regress a rigid transform: ``dim * (dim - 1) / 2`` rotation generators and ``dim`` translations.

    Three parameters in 2-D, six in 3-D -- a rotation and a translation, which is what rigid means.
    """

    def __init__(self, in_channels: int, shape: list[int], dim: int) -> None:
        super().__init__()
        self.add_module("ToFeatures", torch.nn.Flatten(1))
        self.add_module("Head", torch.nn.Linear(in_channels * math.prod(shape), dim * (dim + 1) // 2))

    def init(self, init_type: str, init_gain: float) -> None:
        # Zero parameters are the identity transform: the skew generator is zero, so its exponential
        # is the identity rotation, and the translation is zero.
        self["Head"].weight.data.fill_(0)
        self["Head"].bias.data.zero_()


class MaskFlow(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mask: torch.Tensor, *flows: torch.Tensor):
        result = torch.zeros_like(flows[0])
        for i, flow in enumerate(flows):
            result = result + torch.where(mask == i + 1, flow, torch.tensor(0))
        return result


class SpatialTransformer(torch.nn.Module):
    """Warp ``src`` by ``flow``, in 2-D or 3-D.

    Rigid takes a flat per-sample translation ``[B, dim]``; otherwise ``flow`` is a dense
    displacement field ``[B, dim, *size]`` added to the identity grid.
    """

    def __init__(self, size: list[int], rigid: bool = False) -> None:
        super().__init__()
        self.rigid = rigid
        self.dim = len(size)
        if not rigid:
            vectors = [torch.arange(0, s) for s in size]
            grids = torch.meshgrid(vectors, indexing="ij")
            grid = torch.stack(grids)
            grid = torch.unsqueeze(grid, 0)
            grid = grid.type(torch.float)
            self.register_buffer("grid", grid)

    def forward(self, src: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        if self.rigid:
            return F.grid_sample(
                src,
                # align_corners must match between grid generation and sampling (and the
                # non-rigid path below, which also samples with align_corners=True).
                F.affine_grid(rigid_affine(flow, self.dim), src.size(), align_corners=True),
                align_corners=True,
                mode="bilinear",
            )
        new_locs = self.grid + flow
        shape = flow.shape[2:]
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)
        # grid_sample wants the axis component trailing, and ordered x..z -- the reverse of the
        # tensor's (z,)y,x indexing.
        new_locs = new_locs.permute(0, *range(2, len(shape) + 2), 1)
        return F.grid_sample(src, new_locs[..., list(reversed(range(len(shape))))], align_corners=True, mode="bilinear")


class VecInt(torch.nn.Module):
    def __init__(self, inshape: list[int], nsteps: int):
        super().__init__()
        if nsteps < 0:
            raise ConfigError(f"nsteps should be >= 0, found: {nsteps}")
        self.nsteps = nsteps
        self.scale = 1.0 / (2**self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec: torch.Tensor):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec


class ResizeTransform(torch.nn.Module):
    def __init__(self, size: float):
        super().__init__()
        self.factor = 1.0 / size

    def forward(self, x: torch.Tensor):
        if self.factor < 1:
            x = F.interpolate(
                x,
                align_corners=True,
                scale_factor=self.factor,
                mode="trilinear" if x.dim() == 5 else "bilinear",
                recompute_scale_factor=True,
            )
            x = self.factor * x
        elif self.factor > 1:
            x = self.factor * x
            x = F.interpolate(
                x,
                align_corners=True,
                scale_factor=self.factor,
                mode="trilinear" if x.dim() == 5 else "bilinear",
                recompute_scale_factor=True,
            )
        return x
