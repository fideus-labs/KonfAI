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

"""The torchvision-exact ResNet-18 catalog entry: pretrained weights drive the KonfAI graph.

``ResNet18.yml`` is built only from the curated model-builder registry, yet its weighted leaves
execute in exactly torchvision's forward order (the 1x1 downsample runs after the two 3x3 convs on
the skip branch). That makes it weight-exact: ``transfer_weights_by_execution_order`` pairs all 41
leaves 1:1 and the KonfAI classifier logits match torchvision's output. A structural test also
exercises a small variant without torchvision installed.
"""

from pathlib import Path

import pytest
import torch
from konfai.network.network import Network
from konfai.utils.model_builder import build_model_from_yaml
from konfai.utils.pretrained import transfer_weights_by_execution_order

CATALOG = Path(__file__).resolve().parents[2] / "konfai" / "models" / "yaml"
RESNET18 = CATALOG / "ResNet18.yml"
TORCHVISION_RESNET18_PARAMS = 11_689_512


def _resnet18_yaml(parameters: dict | None = None) -> Network:
    return build_model_from_yaml(yaml_path=str(RESNET18), parameters=parameters)


def test_defaults_reproduce_the_torchvision_resnet18_parameter_count() -> None:
    net = _resnet18_yaml()
    total = sum(param.numel() for param in net.parameters())
    assert total == TORCHVISION_RESNET18_PARAMS, total


def test_torchvision_pretrained_weights_drive_the_konfai_graph() -> None:
    torchvision = pytest.importorskip("torchvision")

    # weights=None keeps the ImageNet download out of CI; the point is graph-exactness, not the
    # actual ImageNet values, so a fixed-seed random reference is a faithful stand-in for a checkpoint.
    torch.manual_seed(0)
    reference = torchvision.models.resnet18(weights=None)
    reference.eval()

    net = _resnet18_yaml()

    torch.manual_seed(1)
    inputs = torch.randn(1, 3, 64, 64)

    # No torchvision->KonfAI key map is supplied: the bridge pairs leaves by forward-execution order.
    transferred = transfer_weights_by_execution_order(
        target=net,
        source=reference,
        target_forward=lambda: list(net.named_forward(inputs)),
        source_forward=lambda: reference(inputs),
    )
    assert transferred == 41

    net.eval()
    with torch.no_grad():
        expected = reference(inputs)
        logits = net.forward_tensor(inputs)

    assert logits.shape == expected.shape == (1, 1000)
    max_diff = (logits - expected).abs().max().item()
    assert torch.allclose(logits, expected, atol=1e-4), max_diff


def test_small_variant_forward_shape_without_torchvision() -> None:
    num_classes = 4
    net = _resnet18_yaml(
        {
            "dim": 2,
            "in_channels": 3,
            "num_classes": num_classes,
            "widths": [8, 8, 16, 32, 64],
        }
    )
    net.eval()
    torch.manual_seed(0)
    inputs = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        logits = net.forward_tensor(inputs)
    assert logits.shape == (1, num_classes)
