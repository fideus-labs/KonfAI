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

"""The shipped SegResNet.yml is weight-exact with MONAI 1.4.0 SegResNet.

Validation level achieved: **weight-exact vs MONAI SegResNet**. The declarative
``SegResNet.yml`` reproduces MONAI's plain SegResNet (no VAE branch) at its
default hyperparameters (init_filters 8, widths [8, 16, 32, 64], blocks_down
(1, 2, 2, 4), blocks_up (1, 1, 1)) exactly: the pre-activation ResBlock
(norm-act-conv, norm-act-conv, +identity), bias-free convolutions,
GroupNorm(num_groups=8) and align_corners=False linear upsampling all match, so
transferring MONAI's ``state_dict`` by shape and running a fixed seeded input in
``eval`` mode gives ``torch.allclose`` logits in both 2D and 3D.

Divergence from MONAI is limited to KonfAI's segmentation convention: the YAML
appends a ``Softmax`` + ``ArgMax`` inference head after the logits convolution
(``Head:Conv``). Equivalence is therefore asserted at the logits conv, before
that head; parameter shapes/count are unaffected (Softmax and ArgMax are
parameter-free).

The structural asserts (build, forward, shapes, terminal head, parameter count)
run without MONAI so CI still validates the catalog entry; the oracle test
``importorskip``s MONAI and is skipped cleanly where it is absent.
"""

import re
from pathlib import Path

import pytest
import torch
from konfai.network.network import Network
from konfai.utils.model_builder import build_model_from_yaml

CATALOG = Path(__file__).resolve().parents[2] / "konfai" / "models" / "yaml"
SEGRESNET_YML = CATALOG / "SegResNet.yml"

# MONAI SegResNet(init_filters=8, in=1, out=2, blocks_down=(1,2,2,4), blocks_up=(1,1,1)).
NB_CLASS = 2
PARAM_COUNT_2D = 394986
PARAM_COUNT_3D = 1176186
NUM_PARAM_TENSORS = 83

INPUT_2D = (1, 1, 32, 32)
INPUT_3D = (1, 1, 16, 16, 16)

# MONAI groups a ResBlock's two norms then its two convs; the KonfAI graph
# registers them in forward order (norm1, conv1, norm2, conv2), and MONAI lists
# every ``up_layers`` block before every ``up_samples`` block. Names therefore
# differ, so MONAI weights are transferred through this explicit shape-checked map.
_STAGE = {
    "down_layers.0.1": "Enc0_ResBlock0",
    "down_layers.1.0": "Enc1_Down",
    "down_layers.1.1": "Enc1_ResBlock0",
    "down_layers.1.2": "Enc1_ResBlock1",
    "down_layers.2.0": "Enc2_Down",
    "down_layers.2.1": "Enc2_ResBlock0",
    "down_layers.2.2": "Enc2_ResBlock1",
    "down_layers.3.0": "Enc3_Down",
    "down_layers.3.1": "Enc3_ResBlock0",
    "down_layers.3.2": "Enc3_ResBlock1",
    "down_layers.3.3": "Enc3_ResBlock2",
    "down_layers.3.4": "Enc3_ResBlock3",
    "up_layers.0.0": "Dec0_ResBlock0",
    "up_layers.1.0": "Dec1_ResBlock0",
    "up_layers.2.0": "Dec2_ResBlock0",
}
_RESBLOCK_LEAF = {
    "norm1.weight": "Norm1.weight",
    "norm1.bias": "Norm1.bias",
    "norm2.weight": "Norm2.weight",
    "norm2.bias": "Norm2.bias",
    "conv1.conv.weight": "Conv1.weight",
    "conv2.conv.weight": "Conv2.weight",
}


def _monai_to_konfai(key: str) -> str:
    """Translate a MONAI SegResNet ``state_dict`` key to the KonfAI graph key."""
    if key == "convInit.conv.weight":
        return "ConvInit.weight"
    up = re.fullmatch(r"up_samples\.(\d)\.0\.conv\.weight", key)
    if up:
        return f"Dec{up.group(1)}_UpConv.weight"
    conv_final = {
        "conv_final.0.weight": "Head.Norm.weight",
        "conv_final.0.bias": "Head.Norm.bias",
        "conv_final.2.conv.weight": "Head.Conv.weight",
        "conv_final.2.conv.bias": "Head.Conv.bias",
    }
    if key in conv_final:
        return conv_final[key]
    for stage, name in _STAGE.items():
        if key.startswith(stage + "."):
            suffix = key[len(stage) + 1 :]
            if suffix == "conv.weight":  # strided downsampling conv
                return f"{name}.weight"
            return f"{name}.{_RESBLOCK_LEAF[suffix]}"
    raise KeyError(key)


def _build(dim: int, upsample_mode: str) -> Network:
    return build_model_from_yaml(
        yaml_path=str(SEGRESNET_YML),
        parameters={"dim": dim, "upsample_mode": upsample_mode, "nb_class": NB_CLASS},
    )


def _flat_state_dict(net: Network) -> dict[str, torch.Tensor]:
    return net.state_dict()[net.get_name()]


def _logits(net: Network, inputs: torch.Tensor) -> torch.Tensor:
    """Run the graph and return the pre-Softmax logits (the ``Head.Conv`` output)."""
    net.eval()
    logits = None
    with torch.no_grad():
        for name, out in net.named_forward(inputs):
            if name == "Head.Conv":
                logits = out
    assert logits is not None, "the graph never produced a 'Head.Conv' output"
    return logits


# --------------------------------------------------------------------------- #
# Structural-strict asserts: build + forward + shapes. These run WITHOUT MONAI.
# --------------------------------------------------------------------------- #
DIM_CASES = pytest.mark.parametrize(
    ("dim", "upsample_mode", "input_shape", "param_count"),
    [
        (2, "bilinear", INPUT_2D, PARAM_COUNT_2D),
        (3, "trilinear", INPUT_3D, PARAM_COUNT_3D),
    ],
    ids=["2d", "3d"],
)


def test_default_catalog_entry_builds() -> None:
    net = build_model_from_yaml(yaml_path=str(SEGRESNET_YML))
    assert isinstance(net, Network)
    assert len(list(net.parameters())) == NUM_PARAM_TENSORS


@DIM_CASES
def test_builds_with_expected_parameter_count(dim, upsample_mode, input_shape, param_count) -> None:
    net = _build(dim, upsample_mode)
    assert isinstance(net, Network)
    assert sum(p.numel() for p in net.parameters()) == param_count
    assert len(list(net.parameters())) == NUM_PARAM_TENSORS


def test_terminal_segmentation_head_is_the_only_output() -> None:
    net = _build(2, "bilinear")
    terminal = [name for name, _, args in net.named_module_args_dict() if "-1" in args.out_branch]
    assert terminal == ["Head"]
    # The logits module a user references in ``outputs_criterions`` must exist.
    assert any(name == "Head.Conv" for name, _, _ in net.named_module_args_dict())


@DIM_CASES
def test_forward_preserves_spatial_and_channels(dim, upsample_mode, input_shape, param_count) -> None:
    net = _build(dim, upsample_mode)
    net.eval()
    torch.manual_seed(0)
    inputs = torch.randn(*input_shape)

    logits = _logits(net, inputs)
    # Logits: channels == nb_class, spatial preserved.
    assert logits.shape == (input_shape[0], NB_CLASS, *input_shape[2:])

    # Terminal ArgMax label map: single channel, spatial preserved.
    label_map = net.forward_tensor(inputs)
    assert label_map.shape == (input_shape[0], 1, *input_shape[2:])
    assert label_map.dtype == torch.int64


# --------------------------------------------------------------------------- #
# Oracle asserts: weight-exact equivalence with MONAI (skipped when absent).
# --------------------------------------------------------------------------- #
@DIM_CASES
def test_weight_exact_vs_monai_segresnet(dim, upsample_mode, input_shape, param_count) -> None:
    monai = pytest.importorskip("monai")
    from monai.networks.nets import SegResNet

    reference = SegResNet(
        spatial_dims=dim,
        init_filters=8,
        in_channels=1,
        out_channels=NB_CLASS,
        blocks_down=(1, 2, 2, 4),
        blocks_up=(1, 1, 1),
    )
    assert sum(p.numel() for p in reference.parameters()) == param_count, monai.__version__

    net = _build(dim, upsample_mode)
    target = _flat_state_dict(net)

    remapped: dict[str, torch.Tensor] = {}
    for monai_key, tensor in reference.state_dict().items():
        konfai_key = _monai_to_konfai(monai_key)
        assert konfai_key in target, konfai_key
        assert target[konfai_key].shape == tensor.shape, (monai_key, konfai_key)
        remapped[konfai_key] = tensor
    assert set(remapped) == set(target), "the MONAI -> KonfAI key map is not a bijection"
    net.load_state_dict(remapped)

    reference.eval()
    torch.manual_seed(0)
    inputs = torch.randn(*input_shape)
    with torch.no_grad():
        expected = reference(inputs)
    logits = _logits(net, inputs)

    assert logits.shape == expected.shape
    assert torch.allclose(logits, expected, atol=1e-5)
