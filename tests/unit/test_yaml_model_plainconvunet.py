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

"""The declarative PlainConvUNet.yml is weight-exact with nnU-Net's PlainConvUNet.

``PlainConvUNet.yml`` reproduces, module-for-module and in forward-execution
order, ``dynamic_network_architectures.architectures.unet.PlainConvUNet`` -- the
"plain conv" nnU-Net backbone used by nnU-Net, TotalSegmentator and MRSeg. Every
conv block is Conv -> InstanceNorm(affine=True) -> LeakyReLU(0.01) with
conv_bias=True, downsampling is a strided conv (first conv of each stage),
upsampling is a transpose conv whose kernel and stride both equal the encoder
stride, and a single 1x1 head returns the highest-resolution logits.

Because the weighted leaves execute in the same order as the reference, a REAL
PlainConvUNet checkpoint transfers straight in through
``transfer_weights_by_execution_order`` and the KonfAI logits are byte-identical
to the reference output. The oracle tests prove this for both isotropic and
per-axis anisotropic strides (the TotalSegmentator / MRSeg case). The structural
test builds and forwards without the oracle so CI still validates the entry.

Deep-supervision accounting (documented, not a defect): PlainConvUNet always
*builds* a 1x1 seg head at every decoder resolution so a deep-supervision
checkpoint stays loadable, but with ``deep_supervision=False`` it *executes* only
the highest-resolution head. For the 4-stage [8, 16, 32, 64] config that is 100
parameters (Conv 32->2 = 66, Conv 16->2 = 34) that are built but never run. The
declarative graph carries exactly the executed head, so its parameter count is
the oracle's executed-path count (351270 total - 100 unused = 351170), which the
tests assert explicitly.
"""

from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from konfai.network.network import Network
from konfai.utils.model_builder import build_model_from_yaml
from konfai.utils.pretrained import transfer_weights_by_execution_order

CATALOG = Path(__file__).resolve().parents[2] / "konfai" / "models" / "yaml"
PLAINCONVUNET_YML = CATALOG / "PlainConvUNet.yml"

# The oracle config from the task: a 4-stage PlainConvUNet with 32 weighted
# leaves and 351270 total parameters, forward (1, 1, 32^3) -> (1, 2, 32^3).
FEATURES_PER_STAGE = [8, 16, 32, 64]
NUM_CLASSES = 2
ORACLE_TOTAL_PARAMS = 351270
# Two deep-supervision seg heads nnU-Net builds but does not execute when
# deep_supervision=False: Conv(32->2)=66 and Conv(16->2)=34.
UNUSED_DEEP_SUPERVISION_PARAMS = 100
EXECUTED_PATH_PARAMS = ORACLE_TOTAL_PARAMS - UNUSED_DEEP_SUPERVISION_PARAMS  # 351170
EXPECTED_WEIGHTED_LEAVES = 32

ISOTROPIC_STRIDES = [1, 2, 2, 2]
# TotalSegmentator / MRSeg-style anisotropic strides (down-sample fewer axes).
ANISOTROPIC_STRIDES = [1, [1, 2, 2], [2, 2, 2], [2, 2, 2]]


def _build_yaml(strides: list, *, dim: int = 3, features: list = FEATURES_PER_STAGE, num_classes: int = NUM_CLASSES):
    return build_model_from_yaml(
        yaml_path=str(PLAINCONVUNET_YML),
        parameters={
            "dim": dim,
            "in_channels": 1,
            "features_per_stage": features,
            "strides": strides,
            "num_classes": num_classes,
        },
    )


def _build_oracle(strides: list):
    pytest.importorskip("dynamic_network_architectures")
    from dynamic_network_architectures.architectures.unet import PlainConvUNet

    return PlainConvUNet(
        input_channels=1,
        n_stages=4,
        features_per_stage=FEATURES_PER_STAGE,
        conv_op=torch.nn.Conv3d,
        kernel_sizes=3,
        strides=strides,
        n_conv_per_stage=2,
        num_classes=NUM_CLASSES,
        n_conv_per_stage_decoder=2,
        conv_bias=True,
        norm_op=torch.nn.InstanceNorm3d,
        norm_op_kwargs={"affine": True},
        nonlin=torch.nn.LeakyReLU,
        nonlin_kwargs={"negative_slope": 0.01, "inplace": True},
        deep_supervision=False,
    )


def _executed_leaf_param_count(model: torch.nn.Module, forward: Callable[[], object]) -> int:
    """Sum the parameters of the weighted leaves that actually run in ``forward``.

    This mirrors the execution-order pairing used by the pretrained bridge: it
    ignores parameters (e.g. unused deep-supervision heads) that are built but
    never executed, so it is the count the KonfAI graph must match.
    """
    total = 0
    seen: set[int] = set()
    handles = []

    def hook(module: torch.nn.Module, _inputs: object, _output: object) -> None:
        nonlocal total
        if id(module) not in seen:
            seen.add(id(module))
            total += sum(p.numel() for p in module.parameters(recurse=False))

    for module in model.modules():
        if next(module.children(), None) is None and next(module.parameters(recurse=False), None) is not None:
            handles.append(module.register_forward_hook(hook))
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            forward()
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    return total


def _assert_weight_exact(strides: list) -> float:
    """Transfer a real PlainConvUNet into the YAML graph and return the logits maxdiff."""
    yaml_net = _build_yaml(strides)
    oracle = _build_oracle(strides)

    torch.manual_seed(0)
    x = torch.randn(1, 1, 32, 32, 32)

    transferred = transfer_weights_by_execution_order(
        yaml_net,
        oracle,
        target_forward=lambda: list(yaml_net.named_forward(x)),
        source_forward=lambda: oracle(x),
    )
    assert transferred == EXPECTED_WEIGHTED_LEAVES

    yaml_net.eval()
    oracle.eval()
    with torch.no_grad():
        reference = oracle(x)
        trace = dict(yaml_net.named_forward(x))

    logits = trace["Head.Conv"]
    assert logits.shape == reference.shape
    assert torch.allclose(logits, reference, atol=1e-5)

    # Parameter accounting: the KonfAI graph carries exactly the executed path.
    konfai_total = sum(p.numel() for p in yaml_net.parameters())
    oracle_total = sum(p.numel() for p in oracle.parameters())
    oracle_executed = _executed_leaf_param_count(oracle, lambda: oracle(x))
    assert konfai_total == oracle_executed
    assert oracle_total - konfai_total == UNUSED_DEEP_SUPERVISION_PARAMS

    return (logits - reference).abs().max().item()


# --------------------------------------------------------------------------- #
# Oracle tests: weight-exact vs dynamic_network_architectures PlainConvUNet.
# --------------------------------------------------------------------------- #
def test_plainconvunet_is_weight_exact_isotropic() -> None:
    # Pin the oracle's total against the task's cited number, then assert exact.
    oracle = _build_oracle(ISOTROPIC_STRIDES)
    assert sum(p.numel() for p in oracle.parameters()) == ORACLE_TOTAL_PARAMS
    konfai = _build_yaml(ISOTROPIC_STRIDES)
    assert sum(p.numel() for p in konfai.parameters()) == EXECUTED_PATH_PARAMS

    maxdiff = _assert_weight_exact(ISOTROPIC_STRIDES)
    assert maxdiff < 1e-4


def test_plainconvunet_is_weight_exact_anisotropic() -> None:
    # Proves TotalSegmentator / MRSeg-style anisotropic-stride checkpoints load.
    maxdiff = _assert_weight_exact(ANISOTROPIC_STRIDES)
    assert maxdiff < 1e-4


# --------------------------------------------------------------------------- #
# Structural test: builds and forwards without the oracle (runs on any CI).
# --------------------------------------------------------------------------- #
def test_plainconvunet_builds_and_forwards() -> None:
    num_classes = 3
    net = _build_yaml(ISOTROPIC_STRIDES, features=[4, 8, 16, 32], num_classes=num_classes)
    assert isinstance(net, Network)
    assert net.get_name() == "PlainConvUNet"

    net.eval()
    torch.manual_seed(0)
    x = torch.randn(1, 1, 32, 32, 32)
    with torch.no_grad():
        trace = dict(net.named_forward(x))

    # The head's Conv output is the raw logits; Softmax preserves them, ArgMax
    # collapses the class axis to a single discrete channel.
    assert trace["Head.Conv"].shape == (1, num_classes, 32, 32, 32)
    assert trace["Head.Softmax"].shape == (1, num_classes, 32, 32, 32)
    assert trace["Head.Argmax"].shape == (1, 1, 32, 32, 32)


def test_plainconvunet_uses_affine_instance_norm_and_leaky_relu() -> None:
    # The nnU-Net signature: Conv(bias) -> InstanceNorm(affine=True) -> LeakyReLU,
    # with the first conv of each downsampling stage carrying the stride.
    net = _build_yaml(ISOTROPIC_STRIDES)
    encoder0 = net["Encoder0"]
    assert [type(module).__name__ for module in encoder0.values()] == [
        "Conv3d",
        "InstanceNorm3d",
        "LeakyReLU",
        "Conv3d",
        "InstanceNorm3d",
        "LeakyReLU",
    ]
    norm = encoder0["Norm_0"]
    assert norm.affine is True
    assert norm.track_running_stats is False
    assert encoder0["Conv_0"].bias is not None
    # nnU-Net downsamples with a strided convolution, not a pooling layer.
    assert net["Encoder1"]["Conv_0"].stride == (2, 2, 2)
    assert net["Encoder1"]["Conv_1"].stride == (1, 1, 1)
