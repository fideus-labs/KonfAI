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

"""Tests for the built-in model definitions in ``konfai.models``."""

import pytest
import torch
from konfai.models.python.classification.convNeXt import LayerScaler
from konfai.models.python.generation.ddpm import DDPM
from konfai.models.python.generation.diffusionGan import CycleGanDiscriminator
from konfai.models.python.generation.vae import LinearVAE
from konfai.models.python.registration.registration import VoxelMorph
from konfai.models.python.representation.representation import Adaptation
from konfai.models.python.segmentation.UNet import UNet

# --------------------------------------------------------------------------------------
# UNet
# --------------------------------------------------------------------------------------


def _unet_forward_last(attention: bool) -> torch.Tensor:
    net = UNet(dim=2, channels=[1, 8, 16], nb_class=2, attention=attention)
    outputs = list(net.named_forward(torch.randn(1, 1, 32, 32)))
    return outputs[-1][1]


def test_layer_scaler_broadcasts_over_2d_and_3d() -> None:
    # gamma must scale per channel over ANY spatial rank. For 3-D [B, C, D, H, W] with D != C the old
    # (C, 1, 1) gamma paired C against the depth axis and crashed; sizing it for `dim` fixes it while
    # keeping the 2-D (C, 1, 1) shape (checkpoint-compatible).
    scaler_3d = LayerScaler(init_value=1e-6, dimensions=4, dim=3)
    assert tuple(scaler_3d.gamma.shape) == (4, 1, 1, 1)
    x3 = torch.randn(2, 4, 3, 5, 5)  # C=4 != D=3
    assert torch.allclose(scaler_3d(x3), x3 * 1e-6)

    scaler_2d = LayerScaler(init_value=1e-6, dimensions=4, dim=2)
    assert tuple(scaler_2d.gamma.shape) == (4, 1, 1)  # unchanged 2-D layout
    x2 = torch.randn(2, 4, 5, 5)
    assert torch.allclose(scaler_2d(x2), x2 * 1e-6)


def test_unet_attention_forwards_without_branch_collision() -> None:
    # out_branch=[1] collided with the Attention block's internal W_g branch, so the parent captured
    # a half-resolution projection and Concat crashed on a size mismatch. The gated skip must reach
    # the skip connection, so the network forwards to a full-resolution output.
    attended = _unet_forward_last(attention=True)
    plain = _unet_forward_last(attention=False)

    assert attended.shape[-2:] == (32, 32)
    assert attended.shape == plain.shape


# --------------------------------------------------------------------------------------
# Generation models
# --------------------------------------------------------------------------------------


def test_linear_vae_is_parameterized_and_variational():
    """#17 LinearVAE must be parameterized (no hardcoded dims) and sample a latent."""
    model = LinearVAE(in_features=32, hidden_features=16, latent_dim=4)
    x = torch.randn(2, 32)
    outputs = dict(model.named_forward(x))
    assert outputs["Head.Tanh"].shape == (2, 32)  # reconstruction matches input size
    assert "Latent.mu" in outputs and "Latent.log_std" in outputs  # KL-ready outputs
    # The latent is sampled: the reconstruction differs across RNG draws.
    torch.manual_seed(0)
    first = dict(model.named_forward(x))["Head.Tanh"]
    torch.manual_seed(1)
    second = dict(model.named_forward(x))["Head.Tanh"]
    assert not torch.allclose(first, second)


def test_cyclegan_discriminator_initialized_no_keyerror():
    """#CycleGan: initialized() must not index a missing 'Sample' submodule on load."""
    model = CycleGanDiscriminator()
    # Must not raise KeyError('Sample').
    model.initialized()


# --------------------------------------------------------------------------------------
# Representation models
# --------------------------------------------------------------------------------------


def test_adaptation_sets_requires_grad_at_construction():
    """#18 Adaptation must configure requires_grad in __init__, not on every forward."""
    adaptation = Adaptation()
    # State is correct immediately after construction, before any forward pass.
    assert all(not p.requires_grad for p in adaptation.Encoder_1.parameters())
    assert all(p.requires_grad for p in adaptation.FCT_1.parameters())


# --------------------------------------------------------------------------------------
# Experimental models fail fast with a clear message
# --------------------------------------------------------------------------------------


def test_ddpm_is_marked_experimental() -> None:
    # DDPM cannot execute a forward pass (broken time-embedding wiring); constructing it must raise
    # an actionable error instead of crashing opaquely deep in the graph later.
    with pytest.raises(NotImplementedError, match="experimental"):
        DDPM()


def test_voxelmorph_rejects_3d_configuration() -> None:
    # VoxelMorph's warping components are 2-D-hardcoded, so its own dim=3 default used to crash.
    with pytest.raises(NotImplementedError, match="dim=2"):
        VoxelMorph(dim=3)
