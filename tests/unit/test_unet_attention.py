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

"""Regression test: UNet(attention=True) must build and forward."""

import torch
from konfai.models.segmentation.UNet import UNet


def _forward_last(attention: bool) -> torch.Tensor:
    net = UNet(dim=2, channels=[1, 8, 16], nb_class=2, attention=attention)
    outputs = list(net.named_forward(torch.randn(1, 1, 32, 32)))
    return outputs[-1][1]


def test_unet_attention_forwards_without_branch_collision() -> None:
    # out_branch=[1] collided with the Attention block's internal W_g branch, so the parent captured
    # a half-resolution projection and Concat crashed on a size mismatch. The gated skip must reach
    # the skip connection, so the network forwards to a full-resolution output.
    attended = _forward_last(attention=True)
    plain = _forward_last(attention=False)

    assert attended.shape[-2:] == (32, 32)
    assert attended.shape == plain.shape
