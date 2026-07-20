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

"""Declarative catalog models are equivalent to their Python / oracle references.

One section per model. Each section keeps its original validation level (documented in the
section banner): weight-exact vs an external oracle (MONAI / torchvision / the parametric
Python model), or structural-strict where the graphs differ by construction. Oracle paths are
guarded by ``pytest.importorskip`` so CI without the optional dependency still validates the
catalog entries structurally. PlainConvUNet and SegResNet keep their own files
(test_yaml_model_plainconvunet.py, test_yaml_model_segresnet.py).
"""

from pathlib import Path

import pytest
import torch
from konfai.models.python.classification.resnet import ResNet
from konfai.models.python.segmentation.NestedUNet import NestedUNet
from konfai.models.python.segmentation.residualencoderunet import ResidualEncoderUNet
from konfai.network.blocks import BlockConfig, MultiHeadSelfAttention, PositionalEmbedding
from konfai.network.network import Network
from konfai.utils.errors import ConfigError
from konfai.utils.model_builder import build_model_from_yaml, list_registered_modules
from konfai.utils.pretrained import (
    _parametric_leaves_in_execution_order,
    transfer_weights_by_execution_order,
)

CATALOG = Path(__file__).resolve().parents[2] / "konfai" / "models" / "yaml"

UNET_YML = Path(__file__).resolve().parents[2] / "examples" / "Segmentation" / "UNet.yml"


# =========================================================================================== #
# UNet: the declarative UNet.yml must build a model equivalent to the Python UNet.
#
# This locks the key "declarative models can replace Python models" property for the
# feed-forward subset: the shipped example ``examples/Segmentation/UNet.yml`` must produce a
# graph with the same parameter count and forward behaviour as the hand-written
# ``konfai.models.python.segmentation.UNet`` configured identically.
# =========================================================================================== #
def _build_yaml_unet():
    params = {"dim": 2, "channels": [1, 32, 64, 128, 256], "nb_class": 41}
    return build_model_from_yaml(yaml_path=str(UNET_YML), parameters=params)


def test_example_unet_yaml_builds_and_forwards():
    net = _build_yaml_unet()
    x = torch.randn(1, 1, 64, 64)
    with torch.no_grad():
        y = net.forward_tensor(x)
    assert y.shape == (1, 1, 64, 64)  # ArgMax head -> single index channel


def test_example_unet_yaml_matches_python_unet_param_count():
    from konfai.models.python.segmentation.UNet import UNet

    yaml_net = _build_yaml_unet()
    block_config = BlockConfig(kernel_size=3, stride=1, padding=1, bias=True, activation="ReLU", norm_mode="NONE")
    python_net = UNet(
        dim=2,
        channels=[1, 32, 64, 128, 256],
        nb_class=41,
        block_config=block_config,
        nb_conv_per_stage=2,
        downsample_mode="MAXPOOL",
        upsample_mode="CONV_TRANSPOSE",
        attention=False,
        block_type="Conv",
    )
    n_yaml = sum(p.numel() for p in yaml_net.parameters())
    n_python = sum(p.numel() for p in python_net.parameters())
    assert n_yaml == n_python, f"yaml={n_yaml} python={n_python}"


# =========================================================================================== #
# Built-ins: the declarative NestedUNet.yml and ResNet.yml must equal the Python models.
#
# This extends the "declarative models can replace Python models" property to the remaining
# feed-forward built-ins: the shipped catalog's ``NestedUNet.yml`` and ``ResNet.yml``
# (konfai/models/yaml) must produce graphs whose ``add_module`` naming (the
# ``outputs_criterions`` dotted paths), branch routing, state_dict keys/shapes, and forward
# behaviour are identical to the hand-written ``konfai.models`` classes configured with the
# same hyperparameters.
# =========================================================================================== #
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


# =========================================================================================== #
# AttentionUNet: the shipped AttentionUNet.yml is a faithful, structurally-strict
# Attention U-Net.
#
# Validation level: structural-strict; it is NOT weight-exact with MONAI 1.4.0
# ``monai.networks.nets.AttentionUnet``. The two graphs differ by construction:
#
# * gate resolution: KonfAI's shipped ``blocks.Attention`` (Oktay et al., 2018)
#   gates the skip against the COARSER decoder feature (``g`` at 1/2 the skip
#   resolution; ``W_x`` stride 2 + Upsample x2), whereas MONAI's ``AttentionBlock``
#   gates at the SAME resolution (stride-1 ``W_g`` and ``W_x``);
# * normalization: KonfAI's gate and this section's ``ConvBlock`` stages use no norm,
#   whereas MONAI inserts BatchNorm after every convolution;
# * down/up sampling: this section uses MaxPool + ConvTranspose (as ``UNet.yml``),
#   whereas MONAI uses strided-convolution down and up sampling;
# * output head: both emit a single head (no deep supervision).
#
# The structural asserts (build + forward on 2D and 3D + correct segmentation
# output shape + the three attention gates present as named modules) run WITHOUT
# MONAI so CI still validates the catalog entry. The oracle test
# ``importorskip``s MONAI, checks the segmentation output shapes agree and the
# total parameter counts land in a tight band, and documents that the state_dict
# structures differ (so weight-transfer is impossible).
# =========================================================================================== #
ATTENTION_UNET_YML = CATALOG / "AttentionUNet.yml"

# Small channel widths keep the test fast while preserving the exact 4-level
# topology of the shipped defaults (5-entry channels -> three attention gates).
CHANNELS_2D = [1, 8, 16, 32, 64]
CHANNELS_3D = [1, 4, 8, 16, 32]
ATTENTION_NB_CLASS = 3

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
ATTENTION_TERMINAL_PATHS = ["UNetBlock_0.Head"]


def _build_attention_unet(dim: int, channels: list[int], nb_class: int = ATTENTION_NB_CLASS) -> Network:
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
    net = _build_attention_unet(2, CHANNELS_2D)
    assert isinstance(net, Network)
    gates = _attention_gate_paths(net)
    assert len(gates) == 3
    assert set(gates) == EXPECTED_GATE_PATHS
    assert _terminal_paths(net) == ATTENTION_TERMINAL_PATHS


def test_default_catalog_entry_builds() -> None:
    net = build_model_from_yaml(yaml_path=str(ATTENTION_UNET_YML))
    assert isinstance(net, Network)
    assert set(_attention_gate_paths(net)) == EXPECTED_GATE_PATHS


def test_forward_2d_preserves_spatial_shape_and_class_channels() -> None:
    net = _build_attention_unet(2, CHANNELS_2D)
    torch.manual_seed(0)
    x = torch.randn(1, CHANNELS_2D[0], 64, 64)
    trace = _forward_trace(net, x)
    softmax = _head_softmax(trace)
    final = list(trace.values())[-1]
    assert softmax.shape == (1, ATTENTION_NB_CLASS, 64, 64)
    assert final.shape == (1, 1, 64, 64)
    assert torch.isfinite(softmax).all()
    assert sum(p.numel() for p in net.parameters()) == EXPECTED_PARAMS_2D


def test_forward_3d_preserves_spatial_shape_and_class_channels() -> None:
    net = _build_attention_unet(3, CHANNELS_3D)
    torch.manual_seed(0)
    x = torch.randn(1, CHANNELS_3D[0], 32, 32, 32)
    trace = _forward_trace(net, x)
    softmax = _head_softmax(trace)
    final = list(trace.values())[-1]
    assert softmax.shape == (1, ATTENTION_NB_CLASS, 32, 32, 32)
    assert final.shape == (1, 1, 32, 32, 32)
    assert torch.isfinite(softmax).all()
    assert sum(p.numel() for p in net.parameters()) == EXPECTED_PARAMS_3D


def test_attention_gate_multiplies_the_skip_at_full_resolution() -> None:
    # The gate's terminal Multiply re-weights the ORIGINAL full-resolution skip
    # by the upsampled attention coefficients, so its output keeps the skip's
    # f_l channels (channels[1]) and full spatial size for the finest gate.
    net = _build_attention_unet(2, CHANNELS_2D)
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
    konfai_net = _build_attention_unet(2, channels, nb_class)
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


# =========================================================================================== #
# DynUNet: the declarative DynUNet.yml is weight-exact with MONAI's DynUNet.
#
# DynUNet.yml is KonfAI's nnU-Net-style dynamic U-Net: every stage is
# Conv -> InstanceNorm -> LeakyReLU, downsampling uses strided convolutions,
# upsampling uses transpose convolutions, and deep-supervision heads are attached
# at the coarser decoder resolutions and marked terminal with ``out_branch: [-1]``.
#
# The structural tests (build + 2D/3D forward + head shapes) run WITHOUT MONAI so
# CI still validates the catalog entry. The oracle test uses
# ``pytest.importorskip("monai")`` and asserts a *weight-exact* equivalence: the
# graph's parameter list matches MONAI DynUNet position-by-position, so MONAI's
# weights transfer in and every segmentation head's logits are ``torch.allclose``
# with MONAI's.
#
# Two deliberate, documented convention differences do NOT affect the logits and
# are handled by the oracle test:
#
# * KonfAI heads append Softmax + ArgMax after the 1x1 logit conv (the catalog
#   convention shared with UNet.yml / NestedUNet.yml); the weight-exact comparison
#   is made on each head's ``Conv`` output, which is exactly MONAI's UnetOutBlock.
# * KonfAI emits each deep-supervision head at its native decoder resolution;
#   MONAI upsamples and stacks them in train mode (and drops them in eval). The
#   comparison uses MONAI's pre-upsampling ``heads[i]`` tensors, which are
#   byte-identical to the native-resolution heads.
# =========================================================================================== #
DYNUNET_YML = CATALOG / "DynUNet.yml"

# Small hyperparameters keep the test fast while preserving the exact DynUNet
# topology: 4 encoder stages -> 3 decoder stages -> 2 deep-supervision heads
# plus the full-resolution head. ``channels`` is [in, f0, f1, f2, f3].
CHANNELS = [1, 8, 16, 32, 64]
NB_CLASS = 3

# The full-resolution head plus the two deep-supervision heads, in the order in
# which they are declared (matching MONAI's output_block, then heads[0..1]).
EXPECTED_TERMINAL_HEADS = ["Head", "DeepSupervisionHead_1", "DeepSupervisionHead_2"]


def _build_yaml(dim: int) -> Network:
    return build_model_from_yaml(
        yaml_path=str(DYNUNET_YML),
        parameters={"dim": dim, "channels": CHANNELS, "nb_class": NB_CLASS},
    )


def _terminal_heads(net: Network) -> list[str]:
    return [name for name, _, args in net.named_module_args_dict() if "-1" in args.out_branch]


def _spatial(dim: int) -> tuple[int, ...]:
    return (32, 32) if dim == 2 else (16, 16, 16)


# --------------------------------------------------------------------------- #
# Structural-strict tests (run without MONAI).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dim", [2, 3])
def test_dynunet_builds_as_a_network(dim: int) -> None:
    net = _build_yaml(dim)
    assert isinstance(net, Network)
    assert net.get_name() == "DynUNet"


@pytest.mark.parametrize("dim", [2, 3])
def test_dynunet_deep_supervision_heads_are_terminal(dim: int) -> None:
    net = _build_yaml(dim)
    assert _terminal_heads(net) == EXPECTED_TERMINAL_HEADS


@pytest.mark.parametrize("dim", [2, 3])
def test_dynunet_forward_head_shapes(dim: int) -> None:
    net = _build_yaml(dim)
    net.eval()
    spatial = _spatial(dim)
    torch.manual_seed(0)
    x = torch.randn(1, CHANNELS[0], *spatial)

    with torch.no_grad():
        trace = dict(net.named_forward(x))

    half = tuple(s // 2 for s in spatial)
    quarter = tuple(s // 4 for s in spatial)

    # Softmax outputs carry the nb_class channel; spatial is preserved per level.
    assert trace["Head.Softmax"].shape == (1, NB_CLASS, *spatial)
    assert trace["DeepSupervisionHead_1.Softmax"].shape == (1, NB_CLASS, *half)
    assert trace["DeepSupervisionHead_2.Softmax"].shape == (1, NB_CLASS, *quarter)

    # The terminal ArgMax collapses the class axis to a single discrete channel.
    assert trace["Head.Argmax"].shape == (1, 1, *spatial)
    assert trace["DeepSupervisionHead_1.Argmax"].shape == (1, 1, *half)
    assert trace["DeepSupervisionHead_2.Argmax"].shape == (1, 1, *quarter)


@pytest.mark.parametrize("dim", [2, 3])
def test_dynunet_uses_instance_norm_and_leaky_relu(dim: int) -> None:
    # The nnU-Net signature: Conv -> InstanceNorm(affine=False) -> LeakyReLU.
    net = _build_yaml(dim)
    encoder0 = net["Encoder0"]
    module_types = [type(module).__name__ for module in encoder0.values()]
    assert module_types == [
        f"Conv{dim}d",
        f"InstanceNorm{dim}d",
        "LeakyReLU",
        f"Conv{dim}d",
        f"InstanceNorm{dim}d",
        "LeakyReLU",
    ]
    norm = encoder0["Norm_0"]
    assert norm.affine is False
    assert norm.track_running_stats is False
    # nnU-Net downsamples with a strided convolution, not a pooling layer.
    down_conv = net["Encoder1"]["Conv_0"]
    assert down_conv.stride == (2,) * dim


# --------------------------------------------------------------------------- #
# Oracle test: weight-exact vs MONAI DynUNet (skips cleanly without MONAI).
# --------------------------------------------------------------------------- #
def _build_monai(dim: int):
    pytest.importorskip("monai")
    from monai.networks.nets import DynUNet

    return DynUNet(
        spatial_dims=dim,
        in_channels=CHANNELS[0],
        out_channels=NB_CLASS,
        kernel_size=[3, 3, 3, 3],
        strides=[1, 2, 2, 2],
        upsample_kernel_size=[2, 2, 2],
        filters=CHANNELS[1:],
        norm_name="instance",
        deep_supervision=True,
        deep_supr_num=2,
        res_block=False,
    )


@pytest.mark.parametrize("dim", [2, 3])
def test_dynunet_parameter_shapes_match_monai_position_by_position(dim: int) -> None:
    yaml_net = _build_yaml(dim)
    monai_net = _build_monai(dim)

    yaml_params = list(yaml_net.parameters())
    monai_params = list(monai_net.parameters())

    assert len(yaml_params) == len(monai_params)
    assert [tuple(p.shape) for p in yaml_params] == [tuple(p.shape) for p in monai_params]
    assert sum(p.numel() for p in yaml_params) == sum(p.numel() for p in monai_params)


@pytest.mark.parametrize("dim", [2, 3])
def test_dynunet_is_weight_exact_with_monai(dim: int) -> None:
    yaml_net = _build_yaml(dim)
    monai_net = _build_monai(dim)

    # Transfer MONAI's weights into the YAML graph by shape-ordered position.
    with torch.no_grad():
        for yaml_param, monai_param in zip(list(yaml_net.parameters()), list(monai_net.parameters()), strict=True):
            assert yaml_param.shape == monai_param.shape
            yaml_param.copy_(monai_param)

    yaml_net.eval()
    monai_net.eval()
    spatial = _spatial(dim)
    torch.manual_seed(0)
    x = torch.randn(1, CHANNELS[0], *spatial)

    # MONAI in eval mode returns only the full-resolution head logits.
    with torch.no_grad():
        monai_main = monai_net(x)
        trace = dict(yaml_net.named_forward(x))

    assert trace["Head.Conv"].shape == monai_main.shape
    assert torch.allclose(trace["Head.Conv"], monai_main, atol=1e-5)

    # MONAI's deep-supervision heads are populated by a train-mode forward and
    # stored (pre-upsampling) on ``monai_net.heads``.
    monai_net.train()
    with torch.no_grad():
        monai_net(x)
    monai_heads_by_spatial = {tuple(h.shape[2:]): h for h in monai_net.heads}

    for head_name in ("DeepSupervisionHead_1", "DeepSupervisionHead_2"):
        logits = trace[f"{head_name}.Conv"]
        monai_head = monai_heads_by_spatial[tuple(logits.shape[2:])]
        assert logits.shape == monai_head.shape, head_name
        assert torch.allclose(logits, monai_head, atol=1e-5), head_name


# =========================================================================================== #
# ResNet-18: the torchvision-exact ResNet-18 catalog entry: pretrained weights drive the
# KonfAI graph.
#
# ``ResNet18.yml`` is built only from the curated model-builder registry, yet its weighted
# leaves execute in exactly torchvision's forward order (the 1x1 downsample runs after the two
# 3x3 convs on the skip branch). That makes it weight-exact:
# ``transfer_weights_by_execution_order`` pairs all 41 leaves 1:1 and the KonfAI classifier
# logits match torchvision's output. A structural test also exercises a small variant without
# torchvision installed.
# =========================================================================================== #
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


# =========================================================================================== #
# UNETR: the shipped UNETR.yml (UNEt TRansformer) catalog entry.
#
# Validation level: structural-strict. The graph builds as a KonfAI ``Network`` and a forward
# on a fixed 3D (and 2D) input returns a segmentation map at the input resolution with
# ``out_channels`` channels; the encoder/skip topology is UNETR's (a 12-layer ViT with skips
# reshaped back to volumes after layers 3/6/9 and the final norm, four transpose-convolution
# upsampling decoder stages). The MONAI oracle path (guarded by ``pytest.importorskip``)
# asserts the OUTPUT SHAPE equals MONAI 1.4.0 ``UNETR`` at matching hyperparameters.
#
# This entry is NOT weight-exact to MONAI's UNETR: the residual convolution blocks are KonfAI
# ``ResBlock`` (which differ from MONAI's ``UnetResBlock`` in second-activation placement and
# ReLU vs LeakyReLU), and, as in ViT.yml, the positional embedding is a standalone leaf and
# the attention is ``MultiHeadSelfAttention``. The structural asserts run WITHOUT MONAI so CI
# validates the entry.
# =========================================================================================== #
UNETR_YML = CATALOG / "UNETR.yml"

OUT_CHANNELS = 2
UNETR_NUM_LAYERS = 12
# Transformer hyperparameters shared with the ViT section below (identical values there).
HIDDEN = 64
NUM_HEADS = 8
MLP_DIM = 128
FEATURE_SIZE = 16
IMG = 32

# Feature-volume shapes the skip/decoder path must produce for the 3D default (grid 2^3, feature_size 16).
EXPECTED_3D_FEATURES = {
    "Skip3.ToVolume": (1, HIDDEN, 2, 2, 2),
    "Skip6.ToVolume": (1, HIDDEN, 2, 2, 2),
    "Skip9.ToVolume": (1, HIDDEN, 2, 2, 2),
    "Bottleneck.ToVolume": (1, HIDDEN, 2, 2, 2),
    "Encoder1.Add": (1, FEATURE_SIZE, 32, 32, 32),
    "Encoder2.Up1_Res.Add": (1, FEATURE_SIZE * 2, 16, 16, 16),
    "Encoder3.Up0_Res.Add": (1, FEATURE_SIZE * 4, 8, 8, 8),
    "Encoder4.TranspInit": (1, FEATURE_SIZE * 8, 4, 4, 4),
}

TWO_D_OVERRIDES = {"dim": 2, "num_tokens": 4, "proj_shape": [-1, 2, 2, HIDDEN], "proj_axes": [0, 3, 1, 2]}


def _build_unetr(parameters: dict | None = None) -> Network:
    return build_model_from_yaml(yaml_path=str(UNETR_YML), parameters=parameters)


def test_unetr_builds_as_network() -> None:
    net = _build_unetr()
    assert isinstance(net, Network)
    assert net.name == "UNETR"


def test_unetr_has_twelve_transformer_layers_and_one_positional_embedding() -> None:
    net = _build_unetr()
    attentions = [m for m in net.modules() if isinstance(m, MultiHeadSelfAttention)]
    positional = [m for m in net.modules() if isinstance(m, PositionalEmbedding)]
    assert len(attentions) == UNETR_NUM_LAYERS
    assert len(positional) == 1


def test_unetr_terminal_output_conv_is_marked() -> None:
    net = _build_unetr()
    terminal = [name for name, _, args in net.named_module_args_dict() if "-1" in args.out_branch]
    assert terminal == ["Out"]


def test_unetr_forward_3d_returns_segmentation_map_at_input_resolution() -> None:
    net = _build_unetr()
    net.eval()
    inputs = torch.randn(2, 1, IMG, IMG, IMG)
    logits = net.forward_tensor(inputs)
    assert logits.shape == (2, OUT_CHANNELS, IMG, IMG, IMG)


def test_unetr_forward_2d_returns_segmentation_map_at_input_resolution() -> None:
    net = _build_unetr(TWO_D_OVERRIDES)
    net.eval()
    inputs = torch.randn(2, 1, IMG, IMG)
    logits = net.forward_tensor(inputs)
    assert logits.shape == (2, OUT_CHANNELS, IMG, IMG)


def test_unetr_skip_and_decoder_feature_volumes_match_the_unetr_topology() -> None:
    net = _build_unetr()
    net.eval()
    inputs = torch.randn(1, 1, IMG, IMG, IMG)
    seen: dict[str, tuple[int, ...]] = {}
    with torch.no_grad():
        for name, out in net.named_forward(inputs):
            if name in EXPECTED_3D_FEATURES:
                seen[name] = tuple(out.shape)
    assert seen == EXPECTED_3D_FEATURES


def test_unetr_parameter_count_in_sane_band() -> None:
    net = _build_unetr()
    n_params = sum(p.numel() for p in net.parameters())
    # 12 transformer layers + a CNN decoder at feature_size 16 land in the low millions.
    assert 1_000_000 < n_params < 20_000_000


@pytest.mark.parametrize(
    "spatial_dims,input_shape,overrides",
    [
        (3, (2, 1, IMG, IMG, IMG), None),
        (2, (2, 1, IMG, IMG), TWO_D_OVERRIDES),
    ],
    ids=["3d", "2d"],
)
def test_unetr_output_shape_matches_monai(
    spatial_dims: int, input_shape: tuple[int, ...], overrides: dict | None
) -> None:
    pytest.importorskip("monai")
    from monai.networks.nets import UNETR

    net = _build_unetr(overrides)
    net.eval()
    reference = UNETR(
        in_channels=1,
        out_channels=OUT_CHANNELS,
        img_size=tuple([IMG] * spatial_dims),
        feature_size=FEATURE_SIZE,
        hidden_size=HIDDEN,
        mlp_dim=MLP_DIM,
        num_heads=NUM_HEADS,
        spatial_dims=spatial_dims,
    ).eval()

    inputs = torch.randn(*input_shape)
    with torch.no_grad():
        mine = net.forward_tensor(inputs)
        theirs = reference(inputs)
    assert mine.shape == theirs.shape == (input_shape[0], OUT_CHANNELS, *([IMG] * spatial_dims))


def test_unetr_hidden_size_is_tunable() -> None:
    # proj_shape's channel dim references hidden_size, so a non-default width reshapes the token
    # sequence correctly and forwards (it is not pinned to the hardcoded 64).
    net = _build_unetr({"hidden_size": 32})
    net.eval()
    with torch.no_grad():
        logits = net.forward_tensor(torch.randn(2, 1, IMG, IMG, IMG))
    assert logits.shape == (2, OUT_CHANNELS, IMG, IMG, IMG)


# =========================================================================================== #
# VGG-16: the torchvision-exact VGG-16 feature extractor: pretrained weights drive five named
# outputs.
#
# ``VGG16.yml`` is built only from the curated model-builder registry (Conv/ReLU/MaxPool), yet
# its 13 convolutions execute in exactly torchvision's ``features`` forward order. That makes
# it weight-exact: ``transfer_weights_by_execution_order`` pairs all 13 conv leaves 1:1, and
# each of the five KonfAI block-boundary outputs (``Block_0.Out`` .. ``Block_4.Out``)
# reproduces the corresponding torchvision intermediate activation. The five outputs are the
# multi-layer feature maps a user attaches a perceptual / feature / IMPACT loss to via
# ``outputs_criterions``. A structural test also exercises a small variant without torchvision
# installed.
# =========================================================================================== #
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


# =========================================================================================== #
# ViT: the shipped ViT.yml Vision Transformer catalog entry.
#
# Validation level: structural-strict PLUS verified numerical equivalence of the encoder to
# MONAI 1.4.0 ``ViT(classification=False)``. The structural asserts (build, forward on 2D and
# 3D, output shapes, terminal head, parameter band) run WITHOUT MONAI so CI validates the
# entry; the MONAI oracle path is guarded by ``pytest.importorskip`` and proves the
# transformer maths match token-for-token.
#
# Numerical equivalence is demonstrated by transferring MONAI's weights into the KonfAI graph
# in forward-execution order (the mechanism of ``konfai.utils.pretrained``) and copying the
# single positional embedding, after which the normalised token features match
# ``torch.allclose``.
#
# Documented divergences from MONAI's ViT (why the shipped
# ``transfer_weights_by_execution_order`` cannot be called on the raw pair, and why the
# classifier is not token-identical):
#   * the learnable positional embedding is a standalone ``PositionalEmbedding`` leaf here,
#     whereas MONAI stores it as a parameter on the non-leaf ``PatchEmbeddingBlock``; the
#     leaf-pairing bridge therefore counts one extra leaf on the KonfAI side and refuses the
#     pair, so the equivalence test transfers the shared leaves and copies the positional
#     embedding explicitly;
#   * classification uses global-average-pooling over the token sequence instead of a
#     prepended ``cls`` token (MONAI's ``classification=True`` default) -- the ENCODER is
#     identical, only the head differs;
#   * MONAI 1.4.0's transformer block allocates unused cross-attention parameters that this
#     graph omits.
# =========================================================================================== #
VIT_YML = CATALOG / "ViT.yml"

NUM_LAYERS = 4
VIT_NUM_CLASSES = 2
PATCH = 16


def _build_vit(dim: int, num_tokens: int) -> Network:
    return build_model_from_yaml(yaml_path=str(VIT_YML), parameters={"dim": dim, "num_tokens": num_tokens})


def _encoder_features(net: Network, inputs: torch.Tensor) -> torch.Tensor:
    features = None
    with torch.no_grad():
        for name, out in net.named_forward(inputs):
            if name == "Encoder.Norm":
                features = out
    assert features is not None, "the ViT graph never produced an 'Encoder.Norm' token-feature output"
    return features


def test_vit_builds_as_network() -> None:
    net = _build_vit(dim=3, num_tokens=8)
    assert isinstance(net, Network)
    assert net.name == "ViT"


def test_vit_registry_primitives_are_registered() -> None:
    registered = set(list_registered_modules())
    assert {"PositionalEmbedding", "MultiHeadSelfAttention"} <= registered


@pytest.mark.parametrize(
    "dim,input_shape,num_tokens",
    [(2, (2, 1, IMG, IMG), 4), (3, (2, 1, IMG, IMG, IMG), 8)],
    ids=["2d", "3d"],
)
def test_vit_forward_shapes(dim: int, input_shape: tuple[int, ...], num_tokens: int) -> None:
    net = _build_vit(dim=dim, num_tokens=num_tokens)
    net.eval()
    inputs = torch.randn(*input_shape)

    logits = net.forward_tensor(inputs)
    # KonfAI classifier convention: [B, num_classes, 1].
    assert logits.shape == (input_shape[0], VIT_NUM_CLASSES, 1)

    features = _encoder_features(net, inputs)
    # Normalised token sequence: [B, num_tokens, hidden_size].
    assert features.shape == (input_shape[0], num_tokens, HIDDEN)


def test_vit_terminal_head_is_marked() -> None:
    net = _build_vit(dim=3, num_tokens=8)
    terminal = [name for name, _, args in net.named_module_args_dict() if "-1" in args.out_branch]
    assert terminal == ["Head"]


def test_vit_has_four_encoder_layers_each_with_self_attention() -> None:
    net = _build_vit(dim=3, num_tokens=8)
    attentions = [m for m in net.modules() if isinstance(m, MultiHeadSelfAttention)]
    positional = [m for m in net.modules() if isinstance(m, PositionalEmbedding)]
    assert len(attentions) == NUM_LAYERS
    assert len(positional) == 1


def test_vit_parameter_count_in_sane_band() -> None:
    net = _build_vit(dim=3, num_tokens=8)
    n_params = sum(p.numel() for p in net.parameters())
    # 4 pre-norm layers at hidden=64, mlp=128 land around ~4e5 parameters.
    assert 100_000 < n_params < 5_000_000


def test_vit_encoder_is_numerically_equivalent_to_monai() -> None:
    pytest.importorskip("monai")
    from monai.networks.nets import ViT as MonaiViT

    net = _build_vit(dim=3, num_tokens=8)
    net.eval()
    reference = MonaiViT(
        in_channels=1,
        img_size=(IMG, IMG, IMG),
        patch_size=(PATCH, PATCH, PATCH),
        hidden_size=HIDDEN,
        mlp_dim=MLP_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        classification=False,
        spatial_dims=3,
    ).eval()

    torch.manual_seed(0)
    inputs = torch.randn(1, 1, IMG, IMG, IMG)

    def encoder_forward() -> None:
        for name, _ in net.named_forward(inputs):
            if name == "Encoder.Norm":
                break

    # Pair the weighted leaves by forward-execution order, exactly as the pretrained bridge does. The
    # positional embedding is skipped on the KonfAI side because MONAI keeps it as a non-leaf parameter.
    target_leaves = [
        module
        for module in _parametric_leaves_in_execution_order(net, encoder_forward)
        if not isinstance(module, PositionalEmbedding)
    ]
    source_leaves = _parametric_leaves_in_execution_order(reference, lambda: reference(inputs))
    assert len(target_leaves) == len(source_leaves) > 0

    for target_leaf, source_leaf in zip(target_leaves, source_leaves, strict=True):
        target_shapes = {key: tuple(value.shape) for key, value in target_leaf.state_dict().items()}
        source_shapes = {key: tuple(value.shape) for key, value in source_leaf.state_dict().items()}
        assert target_shapes == source_shapes
        target_leaf.load_state_dict(source_leaf.state_dict())

    positional = next(module for module in net.modules() if isinstance(module, PositionalEmbedding))
    with torch.no_grad():
        positional.positional_embedding.copy_(reference.patch_embedding.position_embeddings)

    features = _encoder_features(net, inputs)
    reference_features, _ = reference(inputs)
    assert features.shape == reference_features.shape == (1, 8, HIDDEN)
    assert torch.allclose(features, reference_features, atol=1e-5), (features - reference_features).abs().max().item()


def test_shipped_bridge_refuses_the_raw_pair_because_of_the_positional_embedding() -> None:
    pytest.importorskip("monai")
    from monai.networks.nets import ViT as MonaiViT

    net = _build_vit(dim=3, num_tokens=8)
    net.eval()
    reference = MonaiViT(
        in_channels=1,
        img_size=(IMG, IMG, IMG),
        patch_size=(PATCH, PATCH, PATCH),
        hidden_size=HIDDEN,
        mlp_dim=MLP_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        classification=False,
        spatial_dims=3,
    ).eval()
    inputs = torch.randn(1, 1, IMG, IMG, IMG)

    # The KonfAI graph carries the positional-embedding (and classifier-head) weights as extra leaves the
    # MONAI feature encoder does not expose, so the execution-order bridge honestly refuses the pair.
    with pytest.raises(ConfigError, match="different number of weighted leaves"):
        transfer_weights_by_execution_order(
            target=net,
            source=reference,
            target_forward=lambda: list(net.named_forward(inputs)),
            source_forward=lambda: reference(inputs),
        )


# =========================================================================================== #
# VNet: the declarative VNet.yml is weight-exact with MONAI's ``monai.networks.nets.VNet``.
#
# V-Net (Milletari et al., 2016, arXiv:1606.04797) is a 3D residual encoder-decoder.
# The shipped catalog entry ``konfai/models/yaml/VNet.yml`` is assembled from the
# curated model-builder registry only. This section validates it at the strongest
# honest level:
#
# * Structural (always runs, no MONAI): the file builds into a ``Network`` and runs
#   a correct-shaped forward on both a 3D and a 2D input, its nine residual ``Add``
#   nodes and five ``Concat`` skips are present, and it exposes a single terminal
#   ``Head`` (``out_branch: [-1]``).
# * Weight-exact oracle (skipped without MONAI): built with the canonical channel
#   schedule the graph carries the identical 81 parametric leaves, in the same
#   forward-execution order and with the same shapes, as
#   ``VNet(spatial_dims=3, act='prelu', bias=False)``. Transferring MONAI's weights
#   leaf-by-leaf reproduces its (pre-softmax) forward output to ``torch.allclose``
#   in eval mode.
#
# MONAI's ``VNet`` registers each block's activation *before* its convolution and
# defaults to ELU (not in the registry); we instantiate the oracle with
# ``act='prelu'`` — matching V-Net's canonical channel-wise PReLU — and pair the
# leaves in forward-execution order rather than by state_dict key order.
# =========================================================================================== #
VNET_YML = CATALOG / "VNet.yml"

# Default (canonical) V-Net at in=1, stem=16, out=2 has exactly this many params;
# it matches MONAI's VNet(act='prelu', bias=False) parameter count one-to-one.
CANONICAL_PARAM_COUNT = 45_601_516
CANONICAL_LEAF_COUNT = 81


def _build(**parameters: object) -> Network:
    return build_model_from_yaml(yaml_path=str(VNET_YML), parameters=parameters or None)


def _parametric_leaves_in_exec_order(model: torch.nn.Module, run) -> list[torch.nn.Module]:
    """Collect parametric leaf modules in forward-execution order via hooks."""
    order: list[torch.nn.Module] = []

    def hook(module: torch.nn.Module, _inputs: object, _output: object) -> None:
        order.append(module)

    handles = []
    for _, module in model.named_modules():
        is_leaf = len(list(module.children())) == 0
        if is_leaf and len(list(module.parameters(recurse=False))) > 0:
            handles.append(module.register_forward_hook(hook))
    model.eval()
    with torch.no_grad():
        run()
    for handle in handles:
        handle.remove()
    return order


def _yaml_logits(net: Network, x: torch.Tensor) -> torch.Tensor:
    """Pre-softmax output produced by the ``out_conv2`` node."""
    logits: torch.Tensor | None = None
    with torch.no_grad():
        for name, out in net.named_forward(x):
            if name == "out_conv2":
                logits = out
    assert logits is not None, "the VNet graph must expose an 'out_conv2' logits node"
    return logits


def test_vnet_builds_into_a_network() -> None:
    assert isinstance(_build(), Network)


def test_vnet_3d_forward_has_correct_shape() -> None:
    net = _build()
    net.eval()
    x = torch.randn(1, 1, 16, 16, 16)
    logits = _yaml_logits(net, x)
    # segmentation head: channels == out_channels (default 2), spatial preserved.
    assert tuple(logits.shape) == (1, 2, 16, 16, 16)
    with torch.no_grad():
        argmax = net.forward_tensor(x)
    # terminal ArgMax collapses the class axis to a single label channel.
    assert tuple(argmax.shape) == (1, 1, 16, 16, 16)


def test_vnet_2d_build_also_runs() -> None:
    # V-Net is inherently 3D, but the graph is dim-parametrized so a 2D build is a
    # fast smoke that exercises the same routing.
    net = _build(dim=2, out_channels=3)
    net.eval()
    x = torch.randn(1, 1, 16, 16)
    logits = _yaml_logits(net, x)
    assert tuple(logits.shape) == (1, 3, 16, 16)
    with torch.no_grad():
        argmax = net.forward_tensor(x)
    assert tuple(argmax.shape) == (1, 1, 16, 16)


def test_vnet_has_residual_adds_and_skip_concats() -> None:
    net = _build()
    adds = [name for name, module, _ in net.named_module_args_dict() if type(module).__name__ == "Add"]
    concats = [name for name, module, _ in net.named_module_args_dict() if type(module).__name__ == "Concat"]
    # One residual Add per stage: input + four encoder + four decoder.
    assert len(adds) == 9
    # Input-repeat concat + one skip concat per decoder stage.
    assert len(concats) == 5


def test_vnet_exposes_a_single_terminal_head() -> None:
    net = _build()
    terminal = [name for name, _, args in net.named_module_args_dict() if "-1" in args.out_branch]
    assert terminal == ["Head"]


def test_vnet_default_param_count_is_canonical() -> None:
    net = _build()
    total = sum(p.numel() for p in net.parameters())
    assert total == CANONICAL_PARAM_COUNT


def test_vnet_weight_exact_vs_monai() -> None:
    pytest.importorskip("monai")
    from monai.networks.nets import VNet

    x = torch.randn(1, 1, 16, 16, 16)
    yaml_net = _build()
    monai_net = VNet(spatial_dims=3, in_channels=1, out_channels=2, act="prelu", bias=False)

    yaml_leaves = _parametric_leaves_in_exec_order(yaml_net, lambda: list(yaml_net.named_forward(x)))
    monai_leaves = _parametric_leaves_in_exec_order(monai_net, lambda: monai_net(x))

    # Identical parametric spine: same number of leaves, same types, same shapes,
    # in the same forward-execution order.
    assert len(yaml_leaves) == len(monai_leaves) == CANONICAL_LEAF_COUNT
    assert sum(p.numel() for p in yaml_net.parameters()) == sum(p.numel() for p in monai_net.parameters())
    for index, (yaml_leaf, monai_leaf) in enumerate(zip(yaml_leaves, monai_leaves, strict=True)):
        assert type(yaml_leaf).__name__ == type(monai_leaf).__name__, index
        yaml_shapes = [tuple(p.shape) for p in yaml_leaf.parameters(recurse=False)]
        monai_shapes = [tuple(p.shape) for p in monai_leaf.parameters(recurse=False)]
        assert yaml_shapes == monai_shapes, index

    # Transfer MONAI's weights (params + BatchNorm buffers) leaf-by-leaf.
    with torch.no_grad():
        for yaml_leaf, monai_leaf in zip(yaml_leaves, monai_leaves, strict=True):
            yaml_leaf.load_state_dict(monai_leaf.state_dict())

    yaml_net.eval()
    monai_net.eval()
    with torch.no_grad():
        monai_out = monai_net(x)
    yaml_out = _yaml_logits(yaml_net, x)

    assert yaml_out.shape == monai_out.shape
    assert torch.allclose(yaml_out, monai_out, atol=1e-5), (yaml_out - monai_out).abs().max().item()


# =========================================================================================== #
# ResidualEncoderUNet: the declarative ResidualEncoderUNet.yml is forward-exact with the
# parametric ResidualEncoderUNet.
#
# ``ResidualEncoderUNet.yml`` reproduces, stage-for-stage and in forward-execution order, the
# parametric ``konfai.models.python.segmentation.residualencoderunet.ResidualEncoderUNet``
# (the nnU-Net ResEnc backbone -- the ImpactSeg "body" topology). It is authored at the STAGE
# level from two generic composite blocks, ``ResidualStage`` (a stack of ``ResidualBlockD``)
# and ``DecoderStage`` (a two-input upsample/concat/conv block), so the ~20-node graph reads
# as the architecture, not a conv-by-conv unroll.
#
# Because the weighted leaves execute in the same order as the parametric model, the
# parametric weights transfer straight into the YAML graph through
# ``transfer_weights_by_execution_order`` and the YAML logits are ``torch.allclose`` with the
# parametric output (maxdiff < 1e-4). The parametric model is itself weight-exact with a real
# ResEnc nnU-Net checkpoint (see test_residualencoderunet_parametric.py), so the YAML inherits
# that equivalence. A structural build+forward test runs on any CI without the reference.
# =========================================================================================== #
RESIDUALENCODERUNET_YML = CATALOG / "ResidualEncoderUNet.yml"

# The exact ImpactSeg "body" topology: 2D, 5-channel input, 6 stages, 12 classes.
DIM = 2
IN_CHANNELS = 5
N_STAGES = 6
FEATURES_PER_STAGE = [24, 48, 96, 192, 256, 256]
STRIDES = [1, 2, 2, 2, 2, 2]
N_BLOCKS_PER_STAGE = [1, 2, 2, 3, 3, 3]
N_CONV_PER_STAGE_DECODER = [1, 1, 1, 1, 1]
NUM_CLASSES = 12
# Weighted leaves executed by both graphs (deep_supervision off -> a single seg head):
# 66 encoder (stem 2 + residual stages 64) + 15 decoder (5 * (transpose + conv + norm)) + 1 head.
EXPECTED_WEIGHTED_LEAVES = 82


def _build_resenc(
    *,
    dim: int = DIM,
    in_channels: int = IN_CHANNELS,
    features_per_stage: list = FEATURES_PER_STAGE,
    strides: list = STRIDES,
    n_blocks_per_stage: list = N_BLOCKS_PER_STAGE,
    n_conv_per_stage_decoder: list = N_CONV_PER_STAGE_DECODER,
    num_classes: int = NUM_CLASSES,
) -> Network:
    return build_model_from_yaml(
        yaml_path=str(RESIDUALENCODERUNET_YML),
        parameters={
            "dim": dim,
            "in_channels": in_channels,
            "features_per_stage": features_per_stage,
            "strides": strides,
            "n_blocks_per_stage": n_blocks_per_stage,
            "n_conv_per_stage_decoder": n_conv_per_stage_decoder,
            "num_classes": num_classes,
        },
    )


def _build_reference() -> ResidualEncoderUNet:
    # deep_supervision=False: the parametric model builds and runs only the finest seg head, exactly
    # the single-head ImpactSeg checkpoint configuration the declarative YAML reproduces.
    return ResidualEncoderUNet(
        dim=DIM,
        in_channels=IN_CHANNELS,
        n_stages=N_STAGES,
        features_per_stage=FEATURES_PER_STAGE,
        strides=STRIDES,
        n_blocks_per_stage=N_BLOCKS_PER_STAGE,
        n_conv_per_stage_decoder=N_CONV_PER_STAGE_DECODER,
        num_classes=NUM_CLASSES,
        deep_supervision=False,
    )


def test_residualencoderunet_yaml_is_forward_exact() -> None:
    """Transfer the parametric ResidualEncoderUNet into the YAML graph; the logits must match."""
    yaml_net = _build_resenc()
    reference = _build_reference()

    torch.manual_seed(0)
    x = torch.randn(1, IN_CHANNELS, 128, 128)

    # The bridge pairs weighted leaves in forward-execution order and raises if the two graphs are not
    # weight-exact (different leaf count or a shape mismatch), so a green transfer already proves the
    # YAML stages line up with the parametric model leaf-for-leaf.
    transferred = transfer_weights_by_execution_order(
        yaml_net,
        reference,
        target_forward=lambda: list(yaml_net.named_forward(x)),
        source_forward=lambda: list(reference.named_forward(x)),
    )
    assert transferred == EXPECTED_WEIGHTED_LEAVES

    # Equal weighted-leaf count AND equal total parameter count (the parametric deep_supervision=False
    # model builds exactly the single head the YAML carries).
    assert sum(1 for _ in yaml_net.parameters()) == sum(1 for _ in reference.parameters())
    assert sum(p.numel() for p in yaml_net.parameters()) == sum(p.numel() for p in reference.parameters())

    yaml_net.eval()
    reference.eval()
    with torch.no_grad():
        yaml_trace = dict(yaml_net.named_forward(x))
        reference_trace = dict(reference.named_forward(x))

    yaml_logits = yaml_trace["SegHead"]
    reference_logits = reference_trace[f"SegHead_{N_STAGES - 2}"]
    assert yaml_logits.shape == reference_logits.shape == (1, NUM_CLASSES, 128, 128)
    max_diff = (yaml_logits - reference_logits).abs().max().item()
    assert max_diff < 1e-4, f"YAML diverges from the parametric model: maxdiff={max_diff:.2e}"


def test_residualencoderunet_yaml_builds_and_forwards() -> None:
    """Structural build + forward without the reference (runs on any CI)."""
    num_classes = 3
    # The declarative graph has a fixed 6-stage depth, so overrides stay 6 encoder / 5 decoder wide;
    # only feature widths, block counts, dim, in_channels and class count are freely parametric here.
    net = _build_resenc(
        dim=3,
        in_channels=1,
        features_per_stage=[4, 8, 16, 32, 32, 32],
        strides=[1, 2, 2, 2, 2, 2],
        n_blocks_per_stage=[1, 2, 2, 2, 2, 2],
        n_conv_per_stage_decoder=[1, 1, 1, 1, 1],
        num_classes=num_classes,
    )
    assert isinstance(net, Network)
    assert net.get_name() == "ResidualEncoderUNet"

    net.eval()
    torch.manual_seed(0)
    # A 6-stage net downsamples 2^5 = 32x, so the input must be divisible by 32 and leave the
    # bottleneck with > 1 spatial element (InstanceNorm requires it); 64 is the smallest such size.
    x = torch.randn(1, 1, 64, 64, 64)
    with torch.no_grad():
        trace = dict(net.named_forward(x))

    # Full-resolution raw-logits head at the requested class count.
    assert trace["SegHead"].shape == (1, num_classes, 64, 64, 64)
    # The stem never downsamples; encoder resolutions follow the strides (64/32/16/8/4/2). The stage
    # output is the last leaf yielded under that stage's nested name.
    encoder0_output = trace[[key for key in trace if key.startswith("Encoder_0.")][-1]]
    encoder4_output = trace[[key for key in trace if key.startswith("Encoder_4.")][-1]]
    assert encoder0_output.shape[-3:] == (64, 64, 64)
    assert encoder4_output.shape[-3:] == (4, 4, 4)


def test_residualencoderunet_yaml_uses_avgpool_skip_on_the_bottleneck() -> None:
    # Stage 5 keeps 256->256 at stride 2: no channel change, so its first residual block downsamples the
    # skip with an AvgPool (no projection conv/norm) -- the generic ResidualStage must reproduce that.
    net = _build_resenc()
    bottleneck_block = net["Encoder_5"]["Block_0"]
    child_types = [type(module).__name__ for module in bottleneck_block.values()]
    assert "SkipPool" in bottleneck_block._modules  # avgpool residual downsample
    assert "SkipConv" not in bottleneck_block._modules  # no projection: channels unchanged
    assert child_types.count("AvgPool2d") == 1


def test_residualencoderunet_yaml_decoder_stage_is_a_two_input_node() -> None:
    # Each DecoderStage takes [coarser, skip]; the transpose upsamples the coarser feature and the
    # concat puts the transpose output first, then the encoder skip (nnU-Net order).
    net = _build_resenc()
    decoder0 = net["Decoder_0"]
    assert [type(module).__name__ for module in decoder0.values()] == ["ConvTranspose2d", "Concat", "ConvBlock"]
    transpose = decoder0["Up"]
    assert transpose.stride == (2, 2)
    assert transpose.in_channels == FEATURES_PER_STAGE[5]
    assert transpose.out_channels == FEATURES_PER_STAGE[4]
    conv_block = decoder0["Conv"]
    # First decoder conv maps 2*skip -> skip (the concatenated transpose + skip).
    assert conv_block["Conv_0"].in_channels == 2 * FEATURES_PER_STAGE[4]
    assert conv_block["Conv_0"].out_channels == FEATURES_PER_STAGE[4]


def test_residualencoderunet_yaml_stem_tracks_kernel_size_and_negative_slope() -> None:
    # The stem is spelled out (Conv same-padding + InstanceNorm + LeakyReLU) so it tracks the exposed
    # knobs like the stages: a non-3 kernel keeps the spatial size and the stem LeakyReLU follows
    # negative_slope (a stem hardcoding padding 1 and slope 0.01 breaks both).
    net = build_model_from_yaml(
        yaml_path=str(RESIDUALENCODERUNET_YML),
        parameters={"kernel_size": 5, "negative_slope": 0.2},
    )
    net.eval()
    with torch.no_grad():
        out = net.forward_tensor(torch.randn(1, IN_CHANNELS, 128, 128))
    assert out.shape == (1, NUM_CLASSES, 128, 128)
    stem_slopes = {
        module.negative_slope
        for name, module in net.named_modules()
        if isinstance(module, torch.nn.LeakyReLU) and "Stem" in name
    }
    assert stem_slopes == {0.2}
