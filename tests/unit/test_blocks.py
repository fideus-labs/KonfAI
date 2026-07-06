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

"""Tests for individual building blocks in ``konfai.network.blocks``."""

import pytest
import torch
from konfai.network.blocks import Exit, LatentDistribution, Select, Unsqueeze


def test_vae_latent_uses_gaussian_noise():
    """#3 LatentDistributionZ must sample N(0,1), not U[0,1]."""
    layer = LatentDistribution.LatentDistributionZ()
    mu = torch.zeros(200_000)
    log_std = torch.zeros(200_000)
    z = layer(mu, log_std)  # == epsilon
    assert abs(float(z.mean())) < 0.05, "mean should be ~0"
    assert abs(float(z.std()) - 1.0) < 0.05, "std should be ~1 (Gaussian), not ~0.29 (uniform)"


def test_unsqueeze_forward_accepts_tensor():
    """#8 Unsqueeze.forward(tensor) must work on a single tensor."""
    assert Unsqueeze(dim=1)(torch.randn(3, 4)).shape == (3, 1, 4)


def test_select_squeezes_size_one_dims_by_size():
    """#12 Select must squeeze dimensions whose size is 1, not the dim at index 1."""
    out = Select([slice(0, 1), slice(None), slice(None)])(torch.randn(1, 5, 6))
    assert out.shape == (5, 6)
    # a tensor with no size-1 dims is unchanged
    out2 = Select([slice(None), slice(None)])(torch.randn(4, 5))
    assert out2.shape == (4, 5)


def test_debug_exit_block_raises_runtime_error() -> None:
    with pytest.raises(RuntimeError, match="debug Exit block"):
        Exit()(torch.ones(1))
