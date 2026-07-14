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

"""The declarative ResidualEncoderUNet.yml is forward-exact with the parametric ResidualEncoderUNet.

``ResidualEncoderUNet.yml`` reproduces, stage-for-stage and in forward-execution order, the parametric
``konfai.models.python.segmentation.residualencoderunet.ResidualEncoderUNet`` (the nnU-Net ResEnc
backbone -- the ImpactSeg "body" topology). It is authored at the STAGE level from two generic composite
blocks, ``ResidualStage`` (a stack of ``ResidualBlockD``) and ``DecoderStage`` (a two-input
upsample/concat/conv block), so the ~20-node graph reads as the architecture, not a conv-by-conv unroll.

Because the weighted leaves execute in the same order as the parametric model, the parametric weights
transfer straight into the YAML graph through ``transfer_weights_by_execution_order`` and the YAML logits
are ``torch.allclose`` with the parametric output (maxdiff < 1e-4). The parametric model is itself
weight-exact with a real ResEnc nnU-Net checkpoint (see test_residualencoderunet_parametric.py), so the
YAML inherits that equivalence. A structural build+forward test runs on any CI without the reference.
"""

from pathlib import Path

import torch
from konfai.models.python.segmentation.residualencoderunet import ResidualEncoderUNet
from konfai.network.network import Network
from konfai.utils.model_builder import build_model_from_yaml
from konfai.utils.pretrained import transfer_weights_by_execution_order

CATALOG = Path(__file__).resolve().parents[2] / "konfai" / "models" / "yaml"
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


def _build_yaml(
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
    yaml_net = _build_yaml()
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
    net = _build_yaml(
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
    net = _build_yaml()
    bottleneck_block = net["Encoder_5"]["Block_0"]
    child_types = [type(module).__name__ for module in bottleneck_block.values()]
    assert "SkipPool" in bottleneck_block._modules  # avgpool residual downsample
    assert "SkipConv" not in bottleneck_block._modules  # no projection: channels unchanged
    assert child_types.count("AvgPool2d") == 1


def test_residualencoderunet_yaml_decoder_stage_is_a_two_input_node() -> None:
    # Each DecoderStage takes [coarser, skip]; the transpose upsamples the coarser feature and the
    # concat puts the transpose output first, then the encoder skip (nnU-Net order).
    net = _build_yaml()
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
