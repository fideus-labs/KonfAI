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

"""The shipped UNETR.yml (UNEt TRansformer) catalog entry.

Validation level: structural-strict. The graph builds as a KonfAI ``Network`` and a forward on a fixed 3D
(and 2D) input returns a segmentation map at the input resolution with ``out_channels`` channels; the
encoder/skip topology is UNETR's (a 12-layer ViT with skips reshaped back to volumes after layers 3/6/9 and
the final norm, four transpose-convolution upsampling decoder stages). The MONAI oracle path (guarded by
``pytest.importorskip``) asserts the OUTPUT SHAPE equals MONAI 1.4.0 ``UNETR`` at matching hyperparameters.

This entry is NOT weight-exact to MONAI's UNETR: the residual convolution blocks are KonfAI ``ResBlock``
(which differ from MONAI's ``UnetResBlock`` in second-activation placement and ReLU vs LeakyReLU), and, as in
ViT.yml, the positional embedding is a standalone leaf and the attention is ``MultiHeadSelfAttention``. The
structural asserts run WITHOUT MONAI so CI validates the entry.
"""

from pathlib import Path

import pytest
import torch
from konfai.network.blocks import MultiHeadSelfAttention, PositionalEmbedding
from konfai.network.network import Network
from konfai.utils.model_builder import build_model_from_yaml

CATALOG = Path(__file__).resolve().parents[2] / "konfai" / "models" / "yaml"
UNETR_YML = CATALOG / "UNETR.yml"

OUT_CHANNELS = 2
NUM_LAYERS = 12
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
    assert len(attentions) == NUM_LAYERS
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
