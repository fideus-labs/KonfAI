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

"""Shared harness for the model-oracle tests: catalog paths, oracle builders, and capture helpers."""

from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from konfai.network.network import Network

REPO = Path(__file__).resolve().parents[2]
CATALOG = REPO / "konfai" / "models" / "yaml"


def seeded_input(*shape: int, seed: int = 0) -> torch.Tensor:
    """A deterministic random input tensor."""
    torch.manual_seed(seed)
    return torch.randn(*shape)


def flat_state_dict(net: Network) -> dict[str, torch.Tensor]:
    """The network's own tensors (``Network.state_dict`` nests them under the network name)."""
    return net.state_dict()[net.get_name()]


def terminal_output_paths(net: Network) -> list[str]:
    """Dotted module paths flagged as network outputs (``out_branch: [-1]``)."""
    return [name for name, _, args in net.named_module_args_dict() if "-1" in args.out_branch]


def forward_trace(net: Network, *inputs: torch.Tensor) -> dict[str, torch.Tensor]:
    """Every named module output of an eval-mode, no-grad forward."""
    net.eval()
    with torch.no_grad():
        return dict(net.named_forward(*inputs))


def capture_output(net: Network, inputs: torch.Tensor, name: str) -> torch.Tensor:
    """The (last) output the named module produces during an eval-mode forward."""
    captured: torch.Tensor | None = None
    net.eval()
    with torch.no_grad():
        for module_name, out in net.named_forward(inputs):
            if module_name == name:
                captured = out
    assert captured is not None, f"the graph never produced a '{name}' output"
    return captured


def capture_oracle_seg_outputs(
    oracle: torch.nn.Module, n_stages: int, run: Callable[[], object]
) -> dict[int, torch.Tensor]:
    """The oracle's per-stage seg-layer outputs indexed by decoder stage (coarsest-first).

    Hooking ``seg_layers`` directly captures the outputs by build/execution index, which is the
    order the KonfAI ``SegHead_j`` heads use (nnU-Net itself returns them finest-first).
    """
    captured: dict[int, torch.Tensor] = {}
    handles = [
        oracle.decoder.seg_layers[j].register_forward_hook(  # type: ignore[union-attr]
            lambda _module, _inputs, output, index=j: captured.__setitem__(index, output)
        )
        for j in range(n_stages - 1)
    ]
    try:
        with torch.no_grad():
            run()
    finally:
        for handle in handles:
            handle.remove()
    return captured


def parametric_leaves_in_execution_order(model: torch.nn.Module, run: Callable[[], object]) -> list[torch.nn.Module]:
    """Parametric leaf modules in forward-execution order, collected via hooks."""
    order: list[torch.nn.Module] = []

    def hook(module: torch.nn.Module, _inputs: object, _output: object) -> None:
        order.append(module)

    handles = []
    for module in model.modules():
        is_leaf = next(module.children(), None) is None
        if is_leaf and next(module.parameters(recurse=False), None) is not None:
            handles.append(module.register_forward_hook(hook))
    model.eval()
    try:
        with torch.no_grad():
            run()
    finally:
        for handle in handles:
            handle.remove()
    return order


def executed_leaf_param_count(model: torch.nn.Module, run: Callable[[], object]) -> int:
    """Sum the parameters of the weighted leaves that actually execute in ``run``.

    This mirrors the execution-order pairing used by the pretrained bridge: it ignores
    parameters (e.g. unused deep-supervision heads) that are built but never executed, so it is
    the count a KonfAI graph carrying exactly the executed path must match.
    """
    leaves = {id(module): module for module in parametric_leaves_in_execution_order(model, run)}
    return sum(p.numel() for module in leaves.values() for p in module.parameters(recurse=False))


def build_plainconv_oracle(
    n_stages: int,
    features: list,
    strides: list,
    kernel_sizes: object = 3,
    n_conv: int = 2,
    n_conv_decoder: int = 2,
    num_classes: int = 2,
    deep_supervision: bool = True,
) -> torch.nn.Module:
    """A real nnU-Net PlainConvUNet (3D) with the nnU-Net conv/norm/nonlin signature."""
    pytest.importorskip("dynamic_network_architectures")
    from dynamic_network_architectures.architectures.unet import PlainConvUNet

    return PlainConvUNet(
        input_channels=1,
        n_stages=n_stages,
        features_per_stage=features,
        conv_op=torch.nn.Conv3d,
        kernel_sizes=kernel_sizes,
        strides=strides,
        n_conv_per_stage=n_conv,
        num_classes=num_classes,
        n_conv_per_stage_decoder=n_conv_decoder,
        conv_bias=True,
        norm_op=torch.nn.InstanceNorm3d,
        norm_op_kwargs={"affine": True},
        nonlin=torch.nn.LeakyReLU,
        nonlin_kwargs={"negative_slope": 0.01, "inplace": True},
        deep_supervision=deep_supervision,
    )


def build_resenc_oracle(
    dim: int,
    in_channels: int,
    n_stages: int,
    features: list,
    strides: list,
    kernel_sizes: object,
    n_blocks: object,
    n_conv_decoder: object,
    num_classes: int,
    deep_supervision: bool = True,
) -> torch.nn.Module:
    """A real nnU-Net ResidualEncoderUNet with the nnU-Net ResEnc conv/norm/nonlin signature."""
    pytest.importorskip("dynamic_network_architectures")
    from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet

    conv_op = torch.nn.Conv2d if dim == 2 else torch.nn.Conv3d
    norm_op = torch.nn.InstanceNorm2d if dim == 2 else torch.nn.InstanceNorm3d
    return ResidualEncoderUNet(
        input_channels=in_channels,
        n_stages=n_stages,
        features_per_stage=features,
        conv_op=conv_op,
        kernel_sizes=kernel_sizes,
        strides=strides,
        n_blocks_per_stage=n_blocks,
        num_classes=num_classes,
        n_conv_per_stage_decoder=n_conv_decoder,
        conv_bias=True,
        norm_op=norm_op,
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        dropout_op=None,
        nonlin=torch.nn.LeakyReLU,
        nonlin_kwargs={"negative_slope": 1e-2, "inplace": True},
        deep_supervision=deep_supervision,
    )


def build_smp_unetplusplus(encoder_name: str, in_channels: int, classes: int) -> torch.nn.Module:
    """A real ``smp.UnetPlusPlus`` (no pretrained weights, raw logits)."""
    smp = pytest.importorskip("segmentation_models_pytorch")
    return smp.UnetPlusPlus(
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=in_channels,
        classes=classes,
        activation=None,
    )
