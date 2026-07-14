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

"""The shipped AttentionUNet.yml is a faithful, structurally-strict Attention U-Net.

Validation level: structural-strict; it is NOT weight-exact with MONAI 1.4.0
``monai.networks.nets.AttentionUnet``. The two graphs differ by construction:

* gate resolution: KonfAI's shipped ``blocks.Attention`` (Oktay et al., 2018)
  gates the skip against the COARSER decoder feature (``g`` at 1/2 the skip
  resolution; ``W_x`` stride 2 + Upsample x2), whereas MONAI's ``AttentionBlock``
  gates at the SAME resolution (stride-1 ``W_g`` and ``W_x``);
* normalization: KonfAI's gate and this file's ``ConvBlock`` stages use no norm,
  whereas MONAI inserts BatchNorm after every convolution;
* down/up sampling: this file uses MaxPool + ConvTranspose (as ``UNet.yml``),
  whereas MONAI uses strided-convolution down and up sampling;
* output head: both emit a single head (no deep supervision).

The structural asserts (build + forward on 2D and 3D + correct segmentation
output shape + the three attention gates present as named modules) run WITHOUT
MONAI so CI still validates the catalog entry. The oracle test
``importorskip``s MONAI, checks the segmentation output shapes agree and the
total parameter counts land in a tight band, and documents that the state_dict
structures differ (so weight-transfer is impossible).
"""

from pathlib import Path

import pytest
import torch
from konfai.network.network import Network
from konfai.utils.model_builder import build_model_from_yaml

CATALOG = Path(__file__).resolve().parents[2] / "konfai" / "models" / "yaml"
ATTENTION_UNET_YML = CATALOG / "AttentionUNet.yml"

# Small channel widths keep the test fast while preserving the exact 4-level
# topology of the shipped defaults (5-entry channels -> three attention gates).
CHANNELS_2D = [1, 8, 16, 32, 64]
CHANNELS_3D = [1, 4, 8, 16, 32]
NB_CLASS = 3

# Deterministic parameter counts of the two small configurations above; they
# lock the graph topology (any accidental change to the wiring moves them).
EXPECTED_PARAMS_2D = 124902
EXPECTED_PARAMS_3D = 88878

# Dotted paths of the three additive attention gates, one per skip connection.
EXPECTED_GATE_PATHS = {
    "UNetBlock_0.UNetBlock_1.AttentionGate",
    "UNetBlock_0.UNetBlock_1.UNetBlock_2.AttentionGate",
    "UNetBlock_0.UNetBlock_1.UNetBlock_2.UNetBlock_3.AttentionGate",
}
EXPECTED_TERMINAL_PATHS = ["UNetBlock_0.Head"]


def _build(dim: int, channels: list[int], nb_class: int = NB_CLASS) -> Network:
    return build_model_from_yaml(
        yaml_path=str(ATTENTION_UNET_YML),
        parameters={"dim": dim, "channels": channels, "nb_class": nb_class},
    )


def _attention_gate_paths(net: Network) -> list[str]:
    return [name for name, _, _ in net.named_module_args_dict() if name.split(".")[-1] == "AttentionGate"]


def _terminal_paths(net: Network) -> list[str]:
    return [name for name, _, args in net.named_module_args_dict() if "-1" in args.out_branch]


def _forward_trace(net: Network, x: torch.Tensor) -> dict[str, torch.Tensor]:
    net.eval()
    with torch.no_grad():
        return dict(net.named_forward(x))


def _head_softmax(trace: dict[str, torch.Tensor]) -> torch.Tensor:
    softmaxes = [value for key, value in trace.items() if key.endswith("Head.Softmax")]
    assert softmaxes, "the segmentation head must expose a Softmax output"
    return softmaxes[-1]


def test_builds_as_network_with_three_attention_gates() -> None:
    net = _build(2, CHANNELS_2D)
    assert isinstance(net, Network)
    gates = _attention_gate_paths(net)
    assert len(gates) == 3
    assert set(gates) == EXPECTED_GATE_PATHS
    assert _terminal_paths(net) == EXPECTED_TERMINAL_PATHS


def test_default_catalog_entry_builds() -> None:
    net = build_model_from_yaml(yaml_path=str(ATTENTION_UNET_YML))
    assert isinstance(net, Network)
    assert set(_attention_gate_paths(net)) == EXPECTED_GATE_PATHS


def test_forward_2d_preserves_spatial_shape_and_class_channels() -> None:
    net = _build(2, CHANNELS_2D)
    torch.manual_seed(0)
    x = torch.randn(1, CHANNELS_2D[0], 64, 64)
    trace = _forward_trace(net, x)
    softmax = _head_softmax(trace)
    final = list(trace.values())[-1]
    assert softmax.shape == (1, NB_CLASS, 64, 64)
    assert final.shape == (1, 1, 64, 64)
    assert torch.isfinite(softmax).all()
    assert sum(p.numel() for p in net.parameters()) == EXPECTED_PARAMS_2D


def test_forward_3d_preserves_spatial_shape_and_class_channels() -> None:
    net = _build(3, CHANNELS_3D)
    torch.manual_seed(0)
    x = torch.randn(1, CHANNELS_3D[0], 32, 32, 32)
    trace = _forward_trace(net, x)
    softmax = _head_softmax(trace)
    final = list(trace.values())[-1]
    assert softmax.shape == (1, NB_CLASS, 32, 32, 32)
    assert final.shape == (1, 1, 32, 32, 32)
    assert torch.isfinite(softmax).all()
    assert sum(p.numel() for p in net.parameters()) == EXPECTED_PARAMS_3D


def test_attention_gate_multiplies_the_skip_at_full_resolution() -> None:
    # The gate's terminal Multiply re-weights the ORIGINAL full-resolution skip
    # by the upsampled attention coefficients, so its output keeps the skip's
    # f_l channels (channels[1]) and full spatial size for the finest gate.
    net = _build(2, CHANNELS_2D)
    torch.manual_seed(0)
    x = torch.randn(1, CHANNELS_2D[0], 64, 64)
    trace = _forward_trace(net, x)
    gated = trace["UNetBlock_0.UNetBlock_1.AttentionGate.Multiply"]
    assert gated.shape == (1, CHANNELS_2D[1], 64, 64)
    assert torch.isfinite(gated).all()


def test_matches_monai_attention_unet_structurally() -> None:
    # Oracle: attempt weight-exact equivalence with MONAI's AttentionUnet. It is
    # impossible here (documented divergence: half- vs same-resolution gating,
    # BatchNorm, strided-conv sampling), so we validate structurally instead --
    # identical segmentation output shape and a tight parameter-count band.
    monai = pytest.importorskip("monai")
    from monai.networks.nets import AttentionUnet

    channels = [1, 16, 32, 64, 128]
    nb_class = 2
    konfai_net = _build(2, channels, nb_class)
    monai_net = AttentionUnet(
        spatial_dims=2,
        in_channels=channels[0],
        out_channels=nb_class,
        channels=tuple(channels[1:]),
        strides=(2, 2, 2),
    )

    torch.manual_seed(0)
    x = torch.randn(1, channels[0], 64, 64)
    konfai_net.eval()
    monai_net.eval()
    with torch.no_grad():
        monai_out = monai_net(x)
        konfai_softmax = _head_softmax(_forward_trace(konfai_net, x))

    # Both are Attention U-Nets: identical segmentation output shape.
    assert konfai_softmax.shape == monai_out.shape == (1, nb_class, 64, 64)

    konfai_sd = konfai_net.state_dict()[konfai_net.get_name()]
    konfai_shapes = [tuple(v.shape) for v in konfai_sd.values()]
    monai_shapes = [tuple(v.shape) for v in monai_net.state_dict().values()]
    # Graphs differ by construction -> weight transfer is impossible (not weight-exact).
    assert konfai_shapes != monai_shapes
    assert len(monai_shapes) != len(konfai_shapes)

    konfai_params = sum(p.numel() for p in konfai_net.parameters())
    monai_params = sum(p.numel() for p in monai_net.parameters())
    assert monai.__version__  # the MONAI oracle was actually imported
    assert abs(konfai_params - monai_params) / monai_params < 0.05
