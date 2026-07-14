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

"""The parametric UNetPlusPlus is weight-exact AND forward-exact with ``smp.UnetPlusPlus``.

``konfai.models.python.segmentation.unetplusplus.UNetPlusPlus`` builds a ResNet encoder + UNet++ nested
decoder graph directly from the smp hyper-parameters, without importing ``segmentation_models_pytorch``.
These tests build a **real** ``smp.UnetPlusPlus`` (resnet18/resnet34 encoder, no pretrained weights),
transfer its weights into the KonfAI graph through the execution-order bridge, then assert that the KonfAI
segmentation logits are ``torch.allclose`` (maxdiff < 1e-4) with the reference output on a random input --
plus full parameter-count equality. One config is the exact ImpactSynth "MR" model (5-channel 2D input,
1 class), reproduced in smp forward-execution order so the 117 weighted leaves pair one-to-one.
"""

import pytest
import torch
from konfai.models.python.segmentation.unetplusplus import UNetPlusPlus
from konfai.network.network import Network
from konfai.utils.pretrained import transfer_weights_by_execution_order

# Each config: (id, encoder_name, in_channels, classes, expected_leaves).
CONFIGS = [
    # The EXACT ImpactSynth "MR" backbone: smp.UnetPlusPlus(resnet34, in_channels=5, classes=1). The
    # smp forward runs 117 weighted leaves (72 encoder + 44 decoder + 1 seg head).
    ("impactsynth_mr_resnet34", "resnet34", 5, 1, 117),
    # A second resnet34 config (RGB-like input, 2 classes) proves the channel plumbing is parametric.
    ("resnet34_in3_cls2", "resnet34", 3, 2, 117),
    # resnet18 (BasicBlock layers [2, 2, 2, 2]) proves the encoder block-count loop scales down: fewer
    # residual blocks -> 85 weighted leaves (40 encoder + 44 decoder + 1 seg head).
    ("resnet18_in1_cls4", "resnet18", 1, 4, 85),
]


def _build_oracle(encoder_name: str, in_channels: int, classes: int) -> torch.nn.Module:
    smp = pytest.importorskip("segmentation_models_pytorch")
    return smp.UnetPlusPlus(
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=in_channels,
        classes=classes,
        activation=None,
    )


def _random_input(in_channels: int) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(1, in_channels, 64, 64)


@pytest.mark.parametrize(
    ("encoder_name", "in_channels", "classes", "expected_leaves"),
    [config[1:] for config in CONFIGS],
    ids=[config[0] for config in CONFIGS],
)
def test_unetplusplus_is_weight_and_forward_exact(
    encoder_name: str, in_channels: int, classes: int, expected_leaves: int
) -> None:
    oracle = _build_oracle(encoder_name, in_channels, classes)
    net = UNetPlusPlus(dim=2, in_channels=in_channels, classes=classes, encoder_name=encoder_name)

    x = _random_input(in_channels)

    # The bridge pairs weighted leaves in forward-execution order; it raises if the graphs are not
    # weight-exact (different leaf count or a shape mismatch), so a green transfer already proves
    # structural equivalence.
    transferred = transfer_weights_by_execution_order(
        net,
        oracle,
        target_forward=lambda: list(net.named_forward(x)),
        source_forward=lambda: oracle(x),
    )
    assert transferred == expected_leaves

    net.eval()
    oracle.eval()
    with torch.no_grad():
        trace = dict(net.named_forward(x))
        reference = oracle(x)

    konfai_logits = trace["SegmentationHead"]
    assert konfai_logits.shape == reference.shape == (1, classes, 64, 64)
    max_diff = (konfai_logits - reference).abs().max().item()
    assert max_diff < 1e-4, f"UNetPlusPlus diverges from smp: maxdiff={max_diff:.2e}"

    # Full parameter-count equality (and identical number of parameter tensors): the reproduction has
    # no built-but-unused module gap with smp.
    assert sum(p.numel() for p in net.parameters()) == sum(p.numel() for p in oracle.parameters())
    assert sum(1 for _ in net.parameters()) == sum(1 for _ in oracle.parameters())


def test_impactsynth_mr_is_forward_exact_and_has_expected_size() -> None:
    """The exact ImpactSynth config: forward-exact, 117 weighted leaves, 26,084,881 parameters."""
    net = UNetPlusPlus(dim=2, in_channels=5, classes=1, encoder_name="resnet34")
    oracle = _build_oracle("resnet34", 5, 1)

    x = _random_input(5)
    transferred = transfer_weights_by_execution_order(
        net,
        oracle,
        target_forward=lambda: list(net.named_forward(x)),
        source_forward=lambda: oracle(x),
    )
    assert transferred == 117
    assert sum(p.numel() for p in net.parameters()) == 26_084_881

    net.eval()
    oracle.eval()
    with torch.no_grad():
        finest = dict(net.named_forward(x))["SegmentationHead"]
        reference = oracle(x)
    assert finest.shape == reference.shape == (1, 1, 64, 64)
    assert (finest - reference).abs().max().item() < 1e-4


# --------------------------------------------------------------------------- #
# Structural tests: build and forward without the oracle (run on any CI).
# --------------------------------------------------------------------------- #
def test_unetplusplus_builds_and_forwards_without_oracle() -> None:
    classes = 3
    net = UNetPlusPlus(dim=2, in_channels=2, classes=classes, encoder_name="resnet18")
    assert isinstance(net, Network)
    assert net.get_name() == "UNetPlusPlus"

    net.eval()
    torch.manual_seed(0)
    x = torch.randn(1, 2, 64, 64)
    with torch.no_grad():
        trace = dict(net.named_forward(x))

    # One terminal segmentation head at full input resolution with the requested class count.
    seg = trace["SegmentationHead"]
    assert seg.shape == (1, classes, 64, 64)

    # The dense decoder grid built all 11 nodes (10 in the triangular grid + the final x_0_4 head).
    dense_nodes = {key.split("_up")[0] for key in trace if key.endswith("_up")}
    assert dense_nodes == {f"x_{d}_{ll}" for ll in range(4) for d in range(ll + 1)} | {"x_0_4"}


def test_unetplusplus_activation_appends_a_terminal_module() -> None:
    # activation=None keeps the seg conv terminal (raw logits); a named activation adds a bounded head.
    net = UNetPlusPlus(dim=2, in_channels=1, classes=1, encoder_name="resnet18", activation="sigmoid")
    net.eval()
    torch.manual_seed(0)
    x = torch.randn(1, 1, 64, 64)
    with torch.no_grad():
        trace = dict(net.named_forward(x))
    assert "Activation" in trace
    out = trace["Activation"]
    assert out.shape == (1, 1, 64, 64)
    assert out.min().item() >= 0.0 and out.max().item() <= 1.0  # sigmoid range


def test_unetplusplus_rejects_bottleneck_encoders() -> None:
    from konfai.utils.errors import ConfigError

    with pytest.raises(ConfigError):
        UNetPlusPlus(dim=2, in_channels=3, classes=1, encoder_name="resnet50")


def test_unetplusplus_load_does_not_reinitialise_weights() -> None:
    # The trainer calls load(init=True) at start-up; UNetPlusPlus must force init=False so a transferred
    # smp checkpoint (or any loaded weights) is never overwritten with init noise.
    net = UNetPlusPlus(dim=2, in_channels=1, classes=1, encoder_name="resnet18")
    snapshot = {name: param.detach().clone() for name, param in net.named_parameters()}

    net.load({}, init=True)

    for name, param in net.named_parameters():
        assert torch.equal(param, snapshot[name]), f"parameter {name} was re-initialised by load(init=True)"
