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

"""The declarative NestedUNet.yml and ResNet.yml must equal the Python models.

This extends the "declarative models can replace Python models" property to the
remaining feed-forward built-ins: the shipped catalog's ``NestedUNet.yml`` and
``ResNet.yml`` (konfai/models/yaml) must produce graphs whose ``add_module``
naming (the ``outputs_criterions`` dotted paths), branch routing, state_dict
keys/shapes, and forward behaviour are identical to the hand-written
``konfai.models`` classes configured with the same hyperparameters.
"""

from pathlib import Path

import pytest
import torch
from konfai.models.python.classification.resnet import ResNet
from konfai.models.python.segmentation.NestedUNet import NestedUNet
from konfai.network.network import Network
from konfai.utils.model_builder import build_model_from_yaml, list_registered_modules

CATALOG = Path(__file__).resolve().parents[2] / "konfai" / "models" / "yaml"
NESTED_UNET_YML = CATALOG / "NestedUNet.yml"
RESNET_YML = CATALOG / "ResNet.yml"

# Small hyperparameters keep the test fast while preserving the exact graph
# topology of the Python defaults (6-entry channels / 5-entry widths).
NESTED_UNET_CHANNELS = [1, 2, 4, 8, 16, 32]
RESNET_WIDTHS = [4, 4, 8, 16, 32]

INPUT_SHAPE = (1, 1, 32, 32)


def _build_nested_unet_pair() -> tuple[Network, Network]:
    yaml_net = build_model_from_yaml(
        yaml_path=str(NESTED_UNET_YML),
        parameters={"dim": 2, "channels": NESTED_UNET_CHANNELS, "nb_class": 3},
    )
    python_net = NestedUNet(dim=2, channels=NESTED_UNET_CHANNELS, nb_class=3)
    return yaml_net, python_net


def _build_resnet_pair() -> tuple[Network, Network]:
    yaml_net = build_model_from_yaml(
        yaml_path=str(RESNET_YML),
        parameters={"dim": 2, "in_channels": 1, "widths": RESNET_WIDTHS, "num_classes": 5},
    )
    python_net = ResNet(patch=None, dim=2, in_channels=1, widths=RESNET_WIDTHS, num_classes=5)
    return yaml_net, python_net


MODEL_PAIRS = {
    "NestedUNet": _build_nested_unet_pair,
    "ResNet": _build_resnet_pair,
}

# Dotted module paths flagged as network outputs (``out_branch: [-1]``): the
# names a YAML user must be able to reuse in documented ``outputs_criterions``.
EXPECTED_TERMINAL_PATHS = {
    "NestedUNet": ["Head_0", "Head_1", "Head_2", "Head_3"],
    "ResNet": [],
}

pair_case = pytest.mark.parametrize("model_name", sorted(MODEL_PAIRS), ids=sorted(MODEL_PAIRS))


def _flat_state_dict(net: Network) -> dict[str, torch.Tensor]:
    return net.state_dict()[net.get_name()]


def _routing_table(net: Network) -> dict[str, tuple[list[str], list[str], list[str], bool]]:
    return {
        name: (list(args.in_branch), list(args.out_branch), list(args.alias), args.pretrained)
        for name, _, args in net.named_module_args_dict()
    }


def _terminal_output_paths(net: Network) -> list[str]:
    return [name for name, _, args in net.named_module_args_dict() if "-1" in args.out_branch]


def _assert_tensors_equal(name: str, yaml_out: torch.Tensor, python_out: torch.Tensor) -> None:
    assert yaml_out.dtype == python_out.dtype, name
    assert yaml_out.shape == python_out.shape, name
    if yaml_out.is_floating_point():
        assert torch.allclose(yaml_out, python_out), name
    else:
        assert torch.equal(yaml_out, python_out), name


def test_resnet_registry_primitives_are_registered() -> None:
    registered = set(list_registered_modules())
    assert {"AdaptiveAvgPool", "Add", "Flatten", "Linear", "ReLU", "Unsqueeze"} <= registered


@pair_case
def test_state_dict_keys_and_shapes_match_one_to_one(model_name: str) -> None:
    yaml_net, python_net = MODEL_PAIRS[model_name]()

    yaml_sd = _flat_state_dict(yaml_net)
    python_sd = _flat_state_dict(python_net)

    assert list(yaml_sd) == list(python_sd)
    for key in python_sd:
        assert yaml_sd[key].shape == python_sd[key].shape, key


@pair_case
def test_terminal_output_paths_match(model_name: str) -> None:
    yaml_net, python_net = MODEL_PAIRS[model_name]()

    assert _terminal_output_paths(python_net) == EXPECTED_TERMINAL_PATHS[model_name]
    assert _terminal_output_paths(yaml_net) == EXPECTED_TERMINAL_PATHS[model_name]


@pair_case
def test_module_paths_and_branch_routing_match(model_name: str) -> None:
    yaml_net, python_net = MODEL_PAIRS[model_name]()

    yaml_routing = _routing_table(yaml_net)
    python_routing = _routing_table(python_net)

    assert list(yaml_routing) == list(python_routing)
    assert yaml_routing == python_routing


@pair_case
def test_forward_traces_are_identical_with_shared_weights(model_name: str) -> None:
    yaml_net, python_net = MODEL_PAIRS[model_name]()
    yaml_net.load_state_dict(_flat_state_dict(python_net))
    yaml_net.eval()
    python_net.eval()

    torch.manual_seed(0)
    inputs = torch.randn(*INPUT_SHAPE)

    with torch.no_grad():
        yaml_trace = list(yaml_net.named_forward(inputs))
        python_trace = list(python_net.named_forward(inputs))

    assert [name for name, _ in yaml_trace] == [name for name, _ in python_trace]
    for (name, yaml_out), (_, python_out) in zip(yaml_trace, python_trace, strict=True):
        _assert_tensors_equal(name, yaml_out, python_out)
