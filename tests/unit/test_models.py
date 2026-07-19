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
from konfai.models.python.registration.registration import VoxelMorph, rigid_affine
from konfai.models.python.representation.representation import Adaptation
from konfai.models.python.segmentation.NestedUNet import NestedUNet
from konfai.models.python.segmentation.UNet import UNet
from konfai.utils.errors import ConfigError

# --------------------------------------------------------------------------------------
# UNet
# --------------------------------------------------------------------------------------


def _unet_forward_last(attention: bool) -> torch.Tensor:
    net = UNet(dim=2, channels=[1, 8, 16], nb_class=2, attention=attention)
    outputs = list(net.named_forward(torch.randn(1, 1, 32, 32)))
    return outputs[-1][1]


def test_layer_scaler_broadcasts_over_2d_and_3d() -> None:
    # gamma must scale per channel over ANY spatial rank: for 3-D [B, C, D, H, W] with D != C a
    # (C, 1, 1) gamma pairs C against the depth axis and crashes. Sized for `dim`, it keeps the
    # 2-D (C, 1, 1) shape (checkpoint-compatible).
    scaler_3d = LayerScaler(init_value=1e-6, dimensions=4, dim=3)
    assert tuple(scaler_3d.gamma.shape) == (4, 1, 1, 1)
    x3 = torch.randn(2, 4, 3, 5, 5)  # C=4 != D=3
    assert torch.allclose(scaler_3d(x3), x3 * 1e-6)

    scaler_2d = LayerScaler(init_value=1e-6, dimensions=4, dim=2)
    assert tuple(scaler_2d.gamma.shape) == (4, 1, 1)  # unchanged 2-D layout
    x2 = torch.randn(2, 4, 5, 5)
    assert torch.allclose(scaler_2d(x2), x2 * 1e-6)


def test_unet_attention_forwards_without_branch_collision() -> None:
    # out_branch=[1] collides with the Attention block's internal W_g branch: the parent captures
    # a half-resolution projection and Concat crashes on a size mismatch. The gated skip must reach
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


# --------------------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------------------


def _run_graph(model, *inputs: torch.Tensor) -> torch.Tensor:
    out = None
    with torch.no_grad():
        for _, tensor in model.named_forward(*inputs):
            out = tensor
    return out


@pytest.mark.parametrize("rigid", [False, True])
@pytest.mark.parametrize("dim", [2, 3])
def test_voxelmorph_warps_in_two_and_three_dimensions(dim: int, rigid: bool) -> None:
    # Every warping component (SpatialTransformer, ResizeTransform, VecInt) must support dim=3 (the
    # default): a 2-D-hardcoded component raises on construction and forbids 3-D registration.
    shape = [32] * dim
    moved = _run_graph(
        VoxelMorph(dim=dim, shape=shape, rigid=rigid),
        torch.randn(1, 3, *shape),
        torch.randn(1, 1, *shape),
    )
    assert moved.shape == (1, 1, *shape)


def test_voxelmorph_rejects_a_shape_that_contradicts_dim() -> None:
    with pytest.raises(ConfigError, match="spatial dimensions"):
        VoxelMorph(dim=2, shape=[192, 192, 192])


@pytest.mark.parametrize("dim", [2, 3])
def test_rigid_affine_is_a_proper_rotation_and_starts_at_identity(dim: int) -> None:
    # The rotation is exp of a skew-symmetric generator, so it must land in SO(n) for ANY parameter
    # value the optimizer produces -- orthogonal with det +1, never a reflection or a shear -- and
    # zero parameters must be the identity, which is what Rigid.init relies on.
    n_parameters = dim * (dim + 1) // 2
    affine = rigid_affine(torch.randn(4, n_parameters), dim)
    rotation = affine[:, :, :dim]

    assert torch.allclose(rotation @ rotation.transpose(1, 2), torch.eye(dim).expand(4, dim, dim), atol=1e-5)
    assert torch.allclose(torch.linalg.det(rotation), torch.ones(4), atol=1e-5)
    assert torch.allclose(rigid_affine(torch.zeros(1, n_parameters), dim), torch.eye(dim, dim + 1)[None], atol=1e-6)


def test_rigid_head_regresses_a_rotation_not_only_a_translation() -> None:
    # The head must regress rotation parameters too: 2 numbers wired straight into the translation
    # column make a "rigid" transform that cannot rotate at all.
    model = VoxelMorph(dim=2, shape=[32, 32], rigid=True)
    model.eval()
    model["Flow"]["Head"].bias.data = torch.tensor([1.2, 0.0, 0.0])  # rotation only, no translation

    moving = torch.zeros(1, 1, 32, 32)
    moving[0, 0, 8:12, :] = 1.0
    moved = _run_graph(model, torch.zeros(1, 3, 32, 32), moving)

    assert not torch.allclose(moved, moving, atol=1e-3)


def test_nested_unet_refuses_attention_it_does_not_have() -> None:
    # attention reaches every nested block and no block reads it, so asking for gates must raise
    # instead of silently building the plain model.
    with pytest.raises(ConfigError, match="attention"):
        NestedUNet(dim=2, channels=[1, 8, 16, 32], attention=True)
