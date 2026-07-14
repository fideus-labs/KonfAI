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

"""The torchvision-exact VGG-16 feature extractor: pretrained weights drive five named outputs.

``VGG16.yml`` is built only from the curated model-builder registry (Conv/ReLU/MaxPool), yet its 13
convolutions execute in exactly torchvision's ``features`` forward order. That makes it weight-exact:
``transfer_weights_by_execution_order`` pairs all 13 conv leaves 1:1, and each of the five KonfAI
block-boundary outputs (``Block_0.Out`` .. ``Block_4.Out``) reproduces the corresponding torchvision
intermediate activation. The five outputs are the multi-layer feature maps a user attaches a
perceptual / feature / IMPACT loss to via ``outputs_criterions``. A structural test also exercises a
small variant without torchvision installed.
"""

from pathlib import Path

import pytest
import torch
from konfai.network.network import Network
from konfai.utils.model_builder import build_model_from_yaml
from konfai.utils.pretrained import transfer_weights_by_execution_order

CATALOG = Path(__file__).resolve().parents[2] / "konfai" / "models" / "yaml"
VGG16 = CATALOG / "VGG16.yml"

# The 13 convolutions of ``torchvision.models.vgg16().features[:31]`` (no classifier, no BatchNorm).
TORCHVISION_VGG16_FEATURES_PARAMS = 14_714_688
# The five block-boundary outputs a user references in ``outputs_criterions`` (dotted -> ":"): the
# torchvision ``features`` slices [0:4] / [4:9] / [9:16] / [16:23] / [23:30].
BLOCK_OUTPUTS = ["Block_0.Out", "Block_1.Out", "Block_2.Out", "Block_3.Out", "Block_4.Out"]
BLOCK_CHANNELS = [64, 128, 256, 512, 512]
# torchvision ``features`` indices at which each block's feature map is produced (1-based count of
# modules consumed): after features[0:4], [0:9], [0:16], [0:23], [0:30].
FEATURE_BOUNDARIES = [4, 9, 16, 23, 30]


def _vgg16_yaml(parameters: dict | None = None) -> Network:
    return build_model_from_yaml(yaml_path=str(VGG16), parameters=parameters)


def _named_outputs(net: Network, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
    outputs: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for name, out in net.named_forward(inputs):
            outputs[name] = out
    return outputs


def test_defaults_reproduce_the_torchvision_vgg16_feature_parameter_count() -> None:
    net = _vgg16_yaml()
    total = sum(param.numel() for param in net.parameters())
    assert total == TORCHVISION_VGG16_FEATURES_PARAMS, total


def test_torchvision_pretrained_weights_drive_the_five_konfai_feature_outputs() -> None:
    torchvision = pytest.importorskip("torchvision")

    # weights=None keeps the ImageNet download out of CI; the point is graph-exactness, not the actual
    # ImageNet values, so a fixed-seed random reference is a faithful stand-in for a real checkpoint.
    torch.manual_seed(0)
    reference = torchvision.models.vgg16(weights=None).features[:31]
    reference.eval()

    net = _vgg16_yaml()

    torch.manual_seed(1)
    inputs = torch.randn(1, 3, 64, 64)

    # No torchvision->KonfAI key map is supplied: the bridge pairs leaves by forward-execution order.
    transferred = transfer_weights_by_execution_order(
        target=net,
        source=reference,
        target_forward=lambda: list(net.named_forward(inputs)),
        source_forward=lambda: reference(inputs),
    )
    assert transferred == 13  # the 13 convolutions of the VGG-16 feature trunk

    # torchvision references: run the sliced feature stack sequentially and capture the five
    # block-boundary activations.
    expected: list[torch.Tensor] = []
    with torch.no_grad():
        activation = inputs
        for index, layer in enumerate(reference):
            activation = layer(activation)
            if (index + 1) in FEATURE_BOUNDARIES:
                expected.append(activation.clone())
    assert len(expected) == len(BLOCK_OUTPUTS)

    net.eval()
    outputs = _named_outputs(net, inputs)

    for path, reference_activation in zip(BLOCK_OUTPUTS, expected, strict=True):
        assert path in outputs, f"missing terminal output '{path}'"
        produced = outputs[path]
        assert produced.shape == reference_activation.shape
        max_diff = (produced - reference_activation).abs().max().item()
        assert torch.allclose(produced, reference_activation, atol=1e-4), (path, max_diff)


def test_five_named_terminal_outputs_expose_the_feature_channel_schedule() -> None:
    net = _vgg16_yaml()
    net.eval()
    torch.manual_seed(0)
    inputs = torch.randn(1, 3, 32, 32)
    outputs = _named_outputs(net, inputs)

    for path, channels in zip(BLOCK_OUTPUTS, BLOCK_CHANNELS, strict=True):
        assert path in outputs, f"missing terminal output '{path}'"
        assert outputs[path].shape[1] == channels, (path, outputs[path].shape)


def test_small_variant_forward_shapes_without_torchvision() -> None:
    widths = [4, 8, 16, 16, 16]
    net = _vgg16_yaml({"dim": 2, "in_channels": 3, "widths": widths})
    net.eval()
    torch.manual_seed(0)
    inputs = torch.randn(1, 3, 32, 32)
    outputs = _named_outputs(net, inputs)

    # Block 0 keeps the input resolution; each later block halves it (a leading max-pool).
    expected_sizes = [32, 16, 8, 4, 2]
    for path, channels, size in zip(BLOCK_OUTPUTS, widths, expected_sizes, strict=True):
        assert path in outputs, f"missing terminal output '{path}'"
        assert tuple(outputs[path].shape) == (1, channels, size, size), (path, outputs[path].shape)
