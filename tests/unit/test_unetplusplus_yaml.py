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

"""The declarative UNetPlusPlus.yml is forward-exact with the parametric UNetPlusPlus.

``UNetPlusPlus.yml`` reproduces, block-for-block and in forward-execution order, the parametric
``konfai.models.python.segmentation.unetplusplus.UNetPlusPlus`` (the ImpactSynth "MR" backbone: a
torchvision ResNet-34 encoder + a UNet++ nested decoder). It is authored at the BLOCK level from two
generic composite blocks, ``ResNetStage`` (a stack of ``ResNetBasicBlock``) and ``UNetPlusPlusNode`` (a
multi-input upsample / dense-concat / two-conv grid node), so the graph reads as the architecture --
encoder stages + the UNet++ decoder grid + head -- not a conv-by-conv unroll.

Because the weighted leaves execute in the same order as the parametric model, the parametric weights
transfer straight into the YAML graph through ``transfer_weights_by_execution_order`` and the YAML logits
are ``torch.allclose`` with the parametric output (maxdiff < 1e-4). The parametric model is itself
weight-exact with a real ``smp.UnetPlusPlus`` checkpoint (see test_unetplusplus_parametric.py), so the
YAML inherits that equivalence. A structural build+forward test runs on any CI without ``smp``.
"""

from pathlib import Path

import torch
from konfai.models.python.segmentation.unetplusplus import UNetPlusPlus
from konfai.network.network import Network
from konfai.utils.model_builder import build_model_from_yaml
from konfai.utils.pretrained import transfer_weights_by_execution_order

CATALOG = Path(__file__).resolve().parents[2] / "konfai" / "models" / "yaml"
UNETPLUSPLUS_YML = CATALOG / "UNetPlusPlus.yml"

# The exact ImpactSynth "MR" backbone: 2D, 5-channel input, resnet34 encoder, 1 class, raw logits.
DIM = 2
IN_CHANNELS = 5
NUM_CLASSES = 1
# smp's UnetPlusPlus(resnet34) runs 117 weighted leaves: 72 encoder (stem 2 + 16 BasicBlocks) + 44
# decoder (11 grid nodes x 2 Conv2dReLU x [conv + norm]) + 1 seg head.
EXPECTED_WEIGHTED_LEAVES = 117


def _build_yaml(
    *,
    dim: int = DIM,
    in_channels: int = IN_CHANNELS,
    num_classes: int = NUM_CLASSES,
    n_blocks_per_stage: list | None = None,
) -> Network:
    parameters: dict = {"dim": dim, "in_channels": in_channels, "num_classes": num_classes}
    if n_blocks_per_stage is not None:
        parameters["n_blocks_per_stage"] = n_blocks_per_stage
    return build_model_from_yaml(yaml_path=str(UNETPLUSPLUS_YML), parameters=parameters)


def _build_reference(encoder_name: str = "resnet34") -> UNetPlusPlus:
    return UNetPlusPlus(
        dim=DIM,
        in_channels=IN_CHANNELS,
        classes=NUM_CLASSES,
        encoder_name=encoder_name,
        activation=None,
    )


def test_unetplusplus_yaml_is_forward_exact() -> None:
    """Transfer the parametric UNetPlusPlus into the YAML graph; the logits must match."""
    yaml_net = _build_yaml()
    reference = _build_reference()

    torch.manual_seed(0)
    x = torch.randn(1, IN_CHANNELS, 64, 64)

    # The bridge pairs weighted leaves in forward-execution order and raises if the two graphs are not
    # weight-exact (different leaf count or a shape mismatch), so a green transfer already proves the
    # YAML blocks line up with the parametric model leaf-for-leaf.
    transferred = transfer_weights_by_execution_order(
        yaml_net,
        reference,
        target_forward=lambda: list(yaml_net.named_forward(x)),
        source_forward=lambda: list(reference.named_forward(x)),
    )
    assert transferred == EXPECTED_WEIGHTED_LEAVES

    # Equal weighted-leaf count AND equal total parameter count (no built-but-unused module gap).
    assert sum(1 for _ in yaml_net.parameters()) == sum(1 for _ in reference.parameters())
    assert sum(p.numel() for p in yaml_net.parameters()) == sum(p.numel() for p in reference.parameters())

    yaml_net.eval()
    reference.eval()
    with torch.no_grad():
        yaml_trace = dict(yaml_net.named_forward(x))
        reference_trace = dict(reference.named_forward(x))

    yaml_logits = yaml_trace["SegmentationHead"]
    reference_logits = reference_trace["SegmentationHead"]
    assert yaml_logits.shape == reference_logits.shape == (1, NUM_CLASSES, 64, 64)
    max_diff = (yaml_logits - reference_logits).abs().max().item()
    assert max_diff < 1e-4, f"YAML diverges from the parametric model: maxdiff={max_diff:.2e}"


def test_unetplusplus_yaml_is_forward_exact_resnet18() -> None:
    """The block-count parameter scales the encoder down to resnet18 (BasicBlock layers [2, 2, 2, 2])."""
    yaml_net = _build_yaml(n_blocks_per_stage=[2, 2, 2, 2])
    reference = _build_reference(encoder_name="resnet18")

    torch.manual_seed(1)
    x = torch.randn(1, IN_CHANNELS, 64, 64)

    # resnet18 = 40 encoder + 44 decoder + 1 head = 85 weighted leaves.
    transferred = transfer_weights_by_execution_order(
        yaml_net,
        reference,
        target_forward=lambda: list(yaml_net.named_forward(x)),
        source_forward=lambda: list(reference.named_forward(x)),
    )
    assert transferred == 85

    yaml_net.eval()
    reference.eval()
    with torch.no_grad():
        yaml_logits = dict(yaml_net.named_forward(x))["SegmentationHead"]
        reference_logits = dict(reference.named_forward(x))["SegmentationHead"]
    assert (yaml_logits - reference_logits).abs().max().item() < 1e-4


def test_unetplusplus_yaml_builds_and_forwards() -> None:
    """Structural build + forward without the reference (runs on any CI)."""
    num_classes = 3
    net = _build_yaml(in_channels=2, num_classes=num_classes)
    assert isinstance(net, Network)
    assert net.get_name() == "UNetPlusPlus"

    net.eval()
    torch.manual_seed(0)
    x = torch.randn(1, 2, 64, 64)
    with torch.no_grad():
        trace = dict(net.named_forward(x))

    # Full-resolution raw-logits head at the requested class count.
    assert trace["SegmentationHead"].shape == (1, num_classes, 64, 64)
    # The dense decoder grid built all 11 nodes (10 in the triangular grid + the final x_0_4 head): the
    # last leaf yielded under each node's nested name is its output.
    dense_nodes = {key.split(".")[0] for key in trace if key.startswith("x_")}
    assert dense_nodes == {f"x_{d}_{ll}" for ll in range(4) for d in range(ll + 1)} | {"x_0_4"}


def test_unetplusplus_yaml_node_is_a_multi_input_upsample_concat_conv() -> None:
    # A mid-grid node (x_0_1) upsamples its predecessor, concatenates 2 dense skips + the encoder skip,
    # then two Conv-BatchNorm-ReLU blocks mapping 256 + 128 + 128 = 512 -> 128.
    net = _build_yaml()
    node = net["x_0_1"]
    assert [type(module).__name__ for module in node.values()] == ["Upsample", "Concat", "ConvBlock"]
    conv_block = node["Conv"]
    assert conv_block["Conv_0"].in_channels == 512
    assert conv_block["Conv_0"].out_channels == 128
    assert conv_block["Conv_1"].in_channels == 128

    # The final full-resolution node has no skip: it drops the concat and convolves the upsample alone.
    final = net["x_0_4"]
    assert [type(module).__name__ for module in final.values()] == ["Upsample", "ConvBlock"]
    assert final["Conv"]["Conv_0"].in_channels == 32
