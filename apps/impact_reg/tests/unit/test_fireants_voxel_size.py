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

"""voxel_size in the FireANTs engine: the loss resamples onto the model's grid, in mm, as itk-impact does for elastix
and ConvexAdam."""

from pathlib import Path
from typing import Optional

import pytest
import torch
from impact_reg_konfai.models.fireants import ImpactFeatureLoss, ModelSpec, _ImpactCore
from konfai.metric.measure.impact import _statistics, resampled


class _Echo(torch.nn.Module):
    """A 3D feature model whose only layer is its input: its features are the image it was given."""

    def forward(
        self,
        x: torch.Tensor,
        nb_layers: torch.Tensor,
        stats: Optional[torch.Tensor] = None,  # noqa: UP045 (TorchScript)
        direction: Optional[torch.Tensor] = None,  # noqa: UP045 (TorchScript)
    ) -> list[torch.Tensor]:
        return [x[:, :1] * 1.0]


def _echo(tmp_path: Path) -> str:
    path = tmp_path / "echo.pt"
    torch.jit.script(_Echo()).save(str(path))
    return str(path)


def test_dense_jacobian_compares_the_images_on_the_model_grid(tmp_path: Path) -> None:
    core = _ImpactCore(ModelSpec(ref=_echo(tmp_path), distance="L2"), False)
    core.model.voxel_size, core.extent = [2.0, 2.0, 2.0], [16.0, 16.0, 8.0]  # 1 mm voxels, [S, P, L] = [8, 16, 16]
    moved = torch.rand(1, 1, 8, 16, 16, requires_grad=True)
    fixed = torch.rand(1, 1, 8, 16, 16)
    value = core.distances(moved, fixed, None, [("L2", 0)], 0, 5)
    on_grid = [resampled(image, (4, 8, 8)) for image in (moved, fixed)]
    core.model.voxel_size = None
    assert torch.allclose(value, core.distances(*on_grid, None, [("L2", 0)], 0, 5))
    value.sum().backward()
    assert moved.grad is not None and moved.grad.abs().sum() > 0  # through the resampling


def test_a_sampled_patch_at_the_model_resolution_keeps_the_feature_at_the_point(tmp_path: Path) -> None:
    # The echo model's centre feature is the image at the point, whatever the patch's resolution.
    core = _ImpactCore(ModelSpec(ref=_echo(tmp_path), distance="L2"), False)
    core.model.voxel_size, core.extent = [2.0, 2.0, 2.0], [20.0, 20.0, 20.0]
    moved, fixed = torch.rand(1, 1, 20, 20, 20), torch.rand(1, 1, 20, 20, 20)
    centres = torch.tensor([[6, 7, 8], [10, 10, 10], [13, 9, 11]])
    value = core.sampled_distances(moved, fixed, centres, 3, [("L2", 0)], 0, 5)
    network = core.model.network(torch.device("cpu"))
    whole = [network(*core.model.inputs(image, _statistics(image)[0]))[0][0] for image in (moved, fixed)]
    at = (slice(None), centres[:, 0], centres[:, 1], centres[:, 2])
    assert torch.allclose(value[0], (whole[0][at] - whole[1][at]).pow(2).mean(), atol=1e-6)


def test_static_extracts_on_the_model_grid_and_brings_the_features_back(tmp_path: Path) -> None:
    loss = ImpactFeatureLoss(
        [[ModelSpec(ref=_echo(tmp_path), voxel_size=[2.0, 2.0, 2.0])]], "Static", False, 5, 0, 0, False
    )
    fixed, moving = torch.rand(1, 1, 8, 16, 16), torch.rand(1, 1, 10, 16, 16)
    extents = ([16.0, 16.0, 8.0], [16.0, 16.0, 10.0])
    volumes = loss.extract(fixed, moving, 0, 0.0, extents)
    assert volumes[0].shape[2:] == fixed.shape[2:] and volumes[1].shape[2:] == moving.shape[2:]
    loss.cores[0].model.voxel_size = None
    on_model = loss.extract(resampled(fixed, (4, 8, 8)), resampled(moving, (5, 8, 8)), 0, 0.0)
    for volume, reference, image in zip(volumes, on_model, (fixed, moving), strict=True):
        assert torch.allclose(volume, resampled(reference, tuple(image.shape[2:]), padding="border"), atol=1e-5)


def test_a_voxel_size_takes_one_value_per_image_axis(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="expected 3"):
        ImpactFeatureLoss([[ModelSpec(ref=_echo(tmp_path), voxel_size=[2.0, 2.0])]], "Static", False, 5, 0, 0, False)
