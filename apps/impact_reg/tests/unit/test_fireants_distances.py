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

"""The FireANTs feature distances mirror the itk-impact C++ metric (ITKIMPACT ImpactLoss.h) so a FireANTs
preset offers the same set as the ConvexAdam / elastix presets. Features are ``[B, C, *spatial]`` with the
channel axis at dim 1. Because FireANTs optimises by autograd (not the C++'s analytic gradient), each loss is
the plain differentiable value and Dice is the SOFT (unrounded) overlap. This locks the formulas + direction.
"""

import numpy as np
import pytest
import torch

from impact_reg_konfai.models.fireants import (
    _EPS,
    _CosineDistance,
    _DISTANCES,
    _NCCDistance,
    _SoftDiceDistance,
)


def _features() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    return torch.rand(2, 4, 3, 3, 3, requires_grad=True), torch.rand(2, 4, 3, 3, 3)


def test_the_five_distances_are_registered() -> None:
    assert set(_DISTANCES) == {"L1", "L2", "Dice", "Cosine", "NCC"}


def test_cosine_matches_the_cpp_formula_and_is_differentiable() -> None:
    x, y = _features()
    xn, yn = x.detach().numpy(), y.numpy()
    dot = (xn * yn).sum(1)
    ref = -(dot / (np.sqrt((xn**2).sum(1)) * np.sqrt((yn**2).sum(1)) + _EPS)).mean()
    loss = _CosineDistance()(x, y)
    assert loss.item() == pytest.approx(ref, abs=1e-5)
    assert torch.autograd.grad(loss, x)[0].abs().sum() > 0
    # aligned features -> minimal cosine distance (-1)
    assert _CosineDistance()(y, y).item() == pytest.approx(-1.0, abs=1e-4)


def test_soft_dice_matches_the_reference_and_beats_random_when_aligned() -> None:
    x, y = _features()
    xn, yn = np.clip(x.detach().numpy(), 0, None), np.clip(y.numpy(), 0, None)
    inter = (xn * yn).sum(1)
    union = (xn + yn).sum(1)
    ref = 1 - ((2 * inter + _EPS) / (union + _EPS)).mean()
    loss = _SoftDiceDistance()(x, y)
    assert loss.item() == pytest.approx(ref, abs=1e-5)
    assert torch.autograd.grad(loss, x)[0].abs().sum() > 0  # soft -> non-zero gradient (the C++ round would zero it)
    assert _SoftDiceDistance()(y, y).item() < loss.item()  # identical overlaps better than the random pair


def test_ncc_matches_per_channel_pearson_and_is_maximal_when_aligned() -> None:
    x, y = _features()
    channels = x.shape[1]
    xf = x.detach().numpy().transpose(1, 0, 2, 3, 4).reshape(channels, -1)
    yf = y.numpy().transpose(1, 0, 2, 3, 4).reshape(channels, -1)
    xf = xf - xf.mean(1, keepdims=True)
    yf = yf - yf.mean(1, keepdims=True)
    ref = -((xf * yf).sum(1) / (np.sqrt((xf**2).sum(1) * (yf**2).sum(1)) + _EPS)).mean()
    loss = _NCCDistance()(x, y)
    assert loss.item() == pytest.approx(ref, abs=1e-5)
    assert torch.autograd.grad(loss, x)[0].abs().sum() > 0
    assert _NCCDistance()(y, y).item() == pytest.approx(-1.0, abs=1e-4)  # perfect correlation
