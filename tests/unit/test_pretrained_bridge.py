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
