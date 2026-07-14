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

"""KonfAI MIND vs the reference implementation, on synthetic input.

The reference ``Mind3D``/``Mind2D`` below are the verbatim descriptor from the source the
IMPACT loss ships (github.com/vboussot/ImpactLoss ``Data/Models/builds/Mind``), used as the
oracle: our KonfAI model must reproduce its descriptor bit-for-bit.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from konfai.models.python.features.mind import MIND
from konfai.network.network import Network


def _pdist_squared3d(x):
    xx = (x**2).sum(dim=1).unsqueeze(2)
    yy = xx.permute(0, 2, 1)
    dist = xx + yy - 2.0 * torch.bmm(x.permute(0, 2, 1), x)
    dist[dist != dist] = 0
    return torch.clamp(dist, 0.0, np.inf)


class _RefMind3D(nn.Module):
    """Verbatim reference MIND-3D (fixed shift kernels)."""

    def __init__(self, radius=2, dilation=2):
        super().__init__()
        self.radius = radius
        self.dilation = dilation
        six = torch.tensor([[0, 1, 1], [1, 1, 0], [1, 0, 1], [1, 1, 2], [2, 1, 1], [1, 2, 1]]).long()
        dist = _pdist_squared3d(six.t().unsqueeze(0).float()).squeeze(0)
        gx, gy = torch.meshgrid(torch.arange(6), torch.arange(6), indexing="ij")
        mask = (gx > gy).view(-1) & (dist == 2).view(-1)
        idx1 = six.unsqueeze(1).repeat(1, 6, 1).view(-1, 3)[mask, :]
        idx2 = six.unsqueeze(0).repeat(6, 1, 1).view(-1, 3)[mask, :]
        mshift1 = torch.zeros(12, 1, 3, 3, 3)
        mshift1.view(-1)[torch.arange(12) * 27 + idx1[:, 0] * 9 + idx1[:, 1] * 3 + idx1[:, 2]] = 1
        mshift2 = torch.zeros(12, 1, 3, 3, 3)
        mshift2.view(-1)[torch.arange(12) * 27 + idx2[:, 0] * 9 + idx2[:, 1] * 3 + idx2[:, 2]] = 1
        self.conv1 = nn.Conv3d(1, 12, 3, 1, 0, bias=False, dilation=dilation)
        self.conv2 = nn.Conv3d(1, 12, 3, 1, 0, bias=False, dilation=dilation)
        self.conv1.weight = nn.Parameter(mshift1, requires_grad=False)
        self.conv2.weight = nn.Parameter(mshift2, requires_grad=False)
        self.rpad1 = nn.ReplicationPad3d(dilation)
        self.rpad2 = nn.ReplicationPad3d(radius)

    def forward(self, x):
        vmin = x.min()
        x = (x - vmin) / (x.max() - vmin + 1e-6)
        ssd = F.avg_pool3d(
            self.rpad2((self.conv1(self.rpad1(x)) - self.conv2(self.rpad1(x))) ** 2), self.radius * 2 + 1, 1
        )
        mind = ssd - torch.min(ssd, dim=1, keepdim=True)[0]
        mind_var = torch.mean(mind, dim=1, keepdim=True)
        m = mind_var.mean()
        mind_var = torch.clamp(mind_var, m * 0.001, m * 1000)
        mind /= mind_var
        return torch.exp(-mind)


def test_konfai_mind3d_matches_the_reference_descriptor() -> None:
    torch.manual_seed(0)
    x = torch.randn(1, 1, 16, 16, 16)

    ref = _RefMind3D(radius=2, dilation=2).eval()
    model = MIND(dim=3, radius=2, dilation=2)
    model.set_name("MIND")
    model.eval()

    with torch.no_grad():
        expected = ref(x)
        ours = None
        for _, out in model.named_forward(x):
            ours = out

    assert isinstance(ours, torch.Tensor), "the MIND graph must emit a tensor, not a list"
    assert ours.shape == expected.shape
    assert torch.allclose(ours, expected, atol=1e-6), (ours - expected).abs().max().item()


def test_konfai_mind_is_a_frozen_network_and_survives_load_init() -> None:
    model = MIND(dim=3)
    model.set_name("MIND")
    assert isinstance(model, Network)
    # No learnable parameters (all shift kernels are frozen).
    assert not any(p.requires_grad for p in model.parameters(pretrained=False))

    kernel_before = model["Descriptor"].conv1.weight.detach().clone()
    model.load({}, init=True)  # the trainer's fresh-start call
    kernel_after = model["Descriptor"].conv1.weight.detach().clone()
    assert torch.equal(kernel_before, kernel_after), "the fixed MIND kernels must survive load(init=True)"


def test_konfai_mind2d_builds_and_forwards() -> None:
    model = MIND(dim=2, radius=1, dilation=1)
    model.set_name("MIND2D")
    model.eval()
    x = torch.randn(1, 1, 24, 24)
    with torch.no_grad():
        out = None
        for _, o in model.named_forward(x):
            out = o
    assert out is not None and out.dim() == 4 and out.shape[2:] == (24, 24)
