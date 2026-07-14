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

"""The declarative DynUNet.yml is weight-exact with MONAI's DynUNet.

DynUNet.yml is KonfAI's nnU-Net-style dynamic U-Net: every stage is
Conv -> InstanceNorm -> LeakyReLU, downsampling uses strided convolutions,
upsampling uses transpose convolutions, and deep-supervision heads are attached
at the coarser decoder resolutions and marked terminal with ``out_branch: [-1]``.

The structural tests (build + 2D/3D forward + head shapes) run WITHOUT MONAI so
CI still validates the catalog entry. The oracle test uses
``pytest.importorskip("monai")`` and asserts a *weight-exact* equivalence: the
graph's parameter list matches MONAI DynUNet position-by-position, so MONAI's
weights transfer in and every segmentation head's logits are ``torch.allclose``
with MONAI's.

Two deliberate, documented convention differences do NOT affect the logits and
are handled by the oracle test:

* KonfAI heads append Softmax + ArgMax after the 1x1 logit conv (the catalog
  convention shared with UNet.yml / NestedUNet.yml); the weight-exact comparison
  is made on each head's ``Conv`` output, which is exactly MONAI's UnetOutBlock.
* KonfAI emits each deep-supervision head at its native decoder resolution;
  MONAI upsamples and stacks them in train mode (and drops them in eval). The
  comparison uses MONAI's pre-upsampling ``heads[i]`` tensors, which are
  byte-identical to the native-resolution heads.
"""

from pathlib import Path

import pytest
import torch
from konfai.network.network import Network
from konfai.utils.model_builder import build_model_from_yaml

CATALOG = Path(__file__).resolve().parents[2] / "konfai" / "models" / "yaml"
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
