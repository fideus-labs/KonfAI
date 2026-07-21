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

"""The pretrained bridge: load an external checkpoint into a KonfAI catalog graph by execution order.

This proves the end-to-end story behind the YAML catalog — a MONAI-trained SegResNet checkpoint drives
the KonfAI ``SegResNet.yml`` graph WITHOUT a hand-written key map — and that the bridge fails loudly on
a non-equivalent pair.
"""

from pathlib import Path

import pytest
import torch
from konfai.network.network import Network
from konfai.utils.errors import ConfigError
from konfai.utils.model_builder import build_model_from_yaml
from konfai.utils.pretrained import transfer_weights_by_execution_order

CATALOG = Path(__file__).resolve().parents[2] / "konfai" / "models" / "yaml"
NB_CLASS = 2


def _segresnet_yaml(dim: int) -> Network:
    upsample_mode = "bilinear" if dim == 2 else "trilinear"
    return build_model_from_yaml(
        yaml_path=str(CATALOG / "SegResNet.yml"),
        parameters={"dim": dim, "upsample_mode": upsample_mode, "nb_class": NB_CLASS},
    )


def _konfai_logits(net: Network, inputs: torch.Tensor) -> torch.Tensor:
    logits = None
    with torch.no_grad():
        for name, out in net.named_forward(inputs):
            if name == "Head.Conv":
                logits = out
    assert logits is not None, "the SegResNet graph never produced a 'Head.Conv' logits output"
    return logits


@pytest.mark.parametrize("dim,input_shape", [(2, (1, 1, 32, 32)), (3, (1, 1, 16, 16, 16))])
def test_monai_pretrained_weights_drive_the_konfai_graph(dim: int, input_shape: tuple[int, ...]) -> None:
    pytest.importorskip("monai")
    from monai.networks.nets import SegResNet

    # Stand-in for a pretrained checkpoint: a MONAI SegResNet with its (random) trained weights.
    reference = SegResNet(
        spatial_dims=dim,
        init_filters=8,
        in_channels=1,
        out_channels=NB_CLASS,
        blocks_down=(1, 2, 2, 4),
        blocks_up=(1, 1, 1),
    )
    reference.eval()

    net = _segresnet_yaml(dim)
    torch.manual_seed(0)
    inputs = torch.randn(*input_shape)

    # The bridge pairs leaves by forward-execution order — no MONAI->KonfAI key map is supplied.
    transferred = transfer_weights_by_execution_order(
        target=net,
        source=reference,
        target_forward=lambda: list(net.named_forward(inputs)),
        source_forward=lambda: reference(inputs),
    )
    assert transferred > 0

    with torch.no_grad():
        expected = reference(inputs)
    logits = _konfai_logits(net, inputs)
    assert logits.shape == expected.shape
    assert torch.allclose(logits, expected, atol=1e-5), (logits - expected).abs().max().item()


def test_bridge_refuses_a_non_equivalent_pair() -> None:
    pytest.importorskip("monai")
    from monai.networks.nets import SegResNet

    net = _segresnet_yaml(dim=2)
    # A SegResNet with a different width schedule is NOT weight-exact to the catalog build.
    wrong = SegResNet(spatial_dims=2, init_filters=16, in_channels=1, out_channels=NB_CLASS)
    wrong.eval()
    inputs = torch.randn(1, 1, 32, 32)
    with pytest.raises(ConfigError, match=r"not weight-exact|different number of weighted leaves"):
        transfer_weights_by_execution_order(
            target=net,
            source=wrong,
            target_forward=lambda: list(net.named_forward(inputs)),
            source_forward=lambda: wrong(inputs),
        )


def test_bridge_refuses_a_target_tensor_no_leaf_owns() -> None:
    # MultiheadAttention holds in_proj_weight itself while having an out_proj child, so it is not a
    # childless leaf and the trace never hooks it. Copying must refuse rather than leave in_proj_weight
    # at its random init and report success.
    source = torch.nn.MultiheadAttention(embed_dim=8, num_heads=2, batch_first=True)
    target = torch.nn.MultiheadAttention(embed_dim=8, num_heads=2, batch_first=True)
    inputs = torch.randn(1, 4, 8)
    untouched = target.in_proj_weight.clone()

    with pytest.raises(ConfigError, match=r"owned by no traced leaf"):
        transfer_weights_by_execution_order(
            target=target,
            source=source,
            target_forward=lambda: target(inputs, inputs, inputs),
            source_forward=lambda: source(inputs, inputs, inputs),
        )
    assert torch.equal(target.in_proj_weight, untouched)


def test_bridge_ignores_source_branches_the_forward_skips() -> None:
    # The mirror of the check above: a source head this configuration never runs (nnU-Net's
    # deep-supervision seg_layers, inactive when deep supervision is off) has no target counterpart to
    # feed, so it must NOT block an otherwise-complete transfer.
    class Reference(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.used = torch.nn.Linear(4, 4)
            self.deep_supervision_head = torch.nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.used(x)

    class Konfai(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.used = torch.nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.used(x)

    source, target = Reference(), Konfai()
    with torch.no_grad():
        source.used.weight.add_(1.0)
    inputs = torch.randn(1, 4)

    transferred = transfer_weights_by_execution_order(
        target=target,
        source=source,
        target_forward=lambda: target(inputs),
        source_forward=lambda: source(inputs),
    )
    assert transferred == 1
    assert torch.equal(target.used.weight, source.used.weight)


def test_bridge_refuses_a_weight_tied_target() -> None:
    # Two target leaves sharing one Parameter (weight tying) would each be loaded in turn, so the earlier
    # leaf would silently keep the later leaf's source weights. The bridge must refuse, not mis-load.
    class TwoLinear(torch.nn.Module):
        def __init__(self, tie: bool) -> None:
            super().__init__()
            self.a = torch.nn.Linear(4, 4)
            self.b = torch.nn.Linear(4, 4)
            if tie:
                self.b.weight = self.a.weight

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.b(self.a(x))

    target = TwoLinear(tie=True)
    source = TwoLinear(tie=False)
    inputs = torch.randn(1, 4)

    with pytest.raises(ConfigError, match=r"ties a tensor"):
        transfer_weights_by_execution_order(
            target=target,
            source=source,
            target_forward=lambda: target(inputs),
            source_forward=lambda: source(inputs),
        )


def test_bridge_refuses_a_buffer_tied_target() -> None:
    # load_state_dict writes persistent buffers too, so a buffer shared across two weighted leaves would be
    # overwritten by the later leaf's source just like a tied parameter. The bridge must refuse that as well.
    class TwoLinearSharedBuffer(torch.nn.Module):
        def __init__(self, tie: bool) -> None:
            super().__init__()
            self.a = torch.nn.Linear(4, 4)
            self.b = torch.nn.Linear(4, 4)
            self.a.register_buffer("scale", torch.ones(4))
            self.b.register_buffer("scale", self.a.scale if tie else torch.ones(4))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.b(self.a(x) * self.a.scale) * self.b.scale

    target = TwoLinearSharedBuffer(tie=True)
    source = TwoLinearSharedBuffer(tie=False)
    inputs = torch.randn(1, 4)

    with pytest.raises(ConfigError, match=r"ties a tensor"):
        transfer_weights_by_execution_order(
            target=target,
            source=source,
            target_forward=lambda: target(inputs),
            source_forward=lambda: source(inputs),
        )
