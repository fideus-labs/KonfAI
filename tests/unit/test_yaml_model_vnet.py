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

"""The declarative VNet.yml is weight-exact with MONAI's ``monai.networks.nets.VNet``.

V-Net (Milletari et al., 2016, arXiv:1606.04797) is a 3D residual encoder-decoder.
The shipped catalog entry ``konfai/models/yaml/VNet.yml`` is assembled from the
curated model-builder registry only. This test validates it at the strongest
honest level:

* Structural (always runs, no MONAI): the file builds into a ``Network`` and runs
  a correct-shaped forward on both a 3D and a 2D input, its nine residual ``Add``
  nodes and five ``Concat`` skips are present, and it exposes a single terminal
  ``Head`` (``out_branch: [-1]``).
* Weight-exact oracle (skipped without MONAI): built with the canonical channel
  schedule the graph carries the identical 81 parametric leaves, in the same
  forward-execution order and with the same shapes, as
  ``VNet(spatial_dims=3, act='prelu', bias=False)``. Transferring MONAI's weights
  leaf-by-leaf reproduces its (pre-softmax) forward output to ``torch.allclose``
  in eval mode.

MONAI's ``VNet`` registers each block's activation *before* its convolution and
defaults to ELU (not in the registry); we instantiate the oracle with
``act='prelu'`` — matching V-Net's canonical channel-wise PReLU — and pair the
leaves in forward-execution order rather than by state_dict key order.
"""

from pathlib import Path

import pytest
import torch
from konfai.network.network import Network
from konfai.utils.model_builder import build_model_from_yaml

CATALOG = Path(__file__).resolve().parents[2] / "konfai" / "models" / "yaml"
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
