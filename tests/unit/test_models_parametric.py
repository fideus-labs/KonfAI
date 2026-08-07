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

"""The parametric Python models are weight-exact with their external oracles at any topology.

Three families, one CONFIGS table each: ``PlainConvUNet`` and ``ResidualEncoderUNet`` vs
``dynamic_network_architectures`` (nnU-Net), and ``UNetPlusPlus`` vs
``segmentation_models_pytorch``. Each test transfers a **real** oracle into the KonfAI graph
through the execution-order bridge and asserts the KonfAI logits are ``torch.allclose`` with the
reference output: plus full parameter-count equality (no built-but-unused module gap). The
topologies cover isotropic and anisotropic strides, anisotropic kernels, extra depth, and the
exact ImpactSeg / ImpactSynth checkpoints, which is the whole point: any oracle checkpoint of
any depth must load. Structural tests (build + forward + ``load`` weight preservation) run
without the oracles so any CI validates the models.
"""

import pytest
import torch
from konfai.models.python.segmentation.plainconvunet import PlainConvUNet
from konfai.models.python.segmentation.residualencoderunet import ResidualEncoderUNet
from konfai.models.python.segmentation.unetplusplus import UNetPlusPlus
from konfai.network.network import Network
from konfai.utils.pretrained import transfer_weights_by_execution_order
from model_oracles import (
    build_plainconv_oracle,
    build_resenc_oracle,
    build_smp_unetplusplus,
    capture_oracle_seg_outputs,
    seeded_input,
)

pytestmark = pytest.mark.slow


# =========================================================================================== #
# PlainConvUNet vs dynamic_network_architectures PlainConvUNet (the "plain conv" nnU-Net
# backbone). Deep supervision on, so every segmentation head is built AND executed; every
# decoder resolution's logits must match the reference seg layer's output.
# =========================================================================================== #
# Each config: (id, n_stages, features_per_stage, strides, kernel_sizes, n_conv_per_stage,
#               n_conv_per_stage_decoder, num_classes).
PLAINCONV_CONFIGS = [
    # 1. 4 stages, isotropic strides, kernel 3, 3D (nnU-Net 3D full-res default depth).
    ("4stage_isotropic", 4, [8, 16, 32, 64], [1, 2, 2, 2], 3, 2, 2, 2),
    # 2. 5 stages, isotropic strides, 3D (deeper: proves the stage loop scales).
    ("5stage_isotropic", 5, [8, 16, 32, 64, 128], [1, 2, 2, 2, 2], 3, 2, 2, 2),
    # 3. 4 stages with per-axis anisotropic strides (TotalSegmentator / MRSeg style).
    ("4stage_anisotropic", 4, [8, 16, 32, 64], [1, [1, 2, 2], [2, 2, 2], [2, 2, 2]], 3, 2, 2, 2),
    # 4. Non-default per-stage conv count, with encoder != decoder depth, on a small config
    #    (proves the encoder and decoder per-stage loops are independent and correct).
    ("3stage_nconv3", 3, [8, 16, 32], [1, 2, 2], 3, 3, 2, 2),
    # 5. ANISOTROPIC KERNELS decoupled from strides ([1, 3, 3] at stage 0 while stride is [1, 2, 2])
    #: kernel_sizes and strides are independent in nnU-Net; a real anisotropic-spacing plan
    #    uses kernel != 3 and kernel != stride.
    (
        "4stage_anisotropic_kernels",
        4,
        [8, 16, 32, 64],
        [1, [1, 2, 2], [2, 2, 2], [2, 2, 2]],
        [[1, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
        2,
        2,
        2,
    ),
    # 6. Per-stage isotropic kernels != 3 (kernel 1 at the stem, kernel 5 deeper): proves padding
    #    is derived from the kernel per stage, not hardcoded.
    ("4stage_mixed_kernels", 4, [8, 16, 32, 64], [1, 2, 2, 2], [1, 3, 5, 3], 2, 2, 2),
]


@pytest.mark.parametrize(
    ("n_stages", "features", "strides", "kernel_sizes", "n_conv", "n_conv_decoder", "num_classes"),
    [config[1:] for config in PLAINCONV_CONFIGS],
    ids=[config[0] for config in PLAINCONV_CONFIGS],
)
def test_plainconvunet_is_weight_exact(
    n_stages: int,
    features: list,
    strides: list,
    kernel_sizes: object,
    n_conv: int,
    n_conv_decoder: int,
    num_classes: int,
) -> None:
    oracle = build_plainconv_oracle(n_stages, features, strides, kernel_sizes, n_conv, n_conv_decoder, num_classes)
    net = PlainConvUNet(
        dim=3,
        in_channels=1,
        n_stages=n_stages,
        features_per_stage=features,
        strides=strides,
        kernel_sizes=kernel_sizes,
        n_conv_per_stage=n_conv,
        n_conv_per_stage_decoder=n_conv_decoder,
        num_classes=num_classes,
    )

    x = seeded_input(1, 1, 32, 32, 32)

    # The bridge pairs weighted leaves in forward-execution order; it raises if the graphs are
    # not weight-exact (different leaf count or a shape mismatch), so a green transfer already
    # proves structural equivalence.
    transferred = transfer_weights_by_execution_order(
        net,
        oracle,
        target_forward=lambda: list(net.named_forward(x)),
        source_forward=lambda: oracle(x),
    )

    net.eval()
    oracle.eval()
    oracle_seg = capture_oracle_seg_outputs(oracle, n_stages, lambda: oracle(x))
    with torch.no_grad():
        trace = dict(net.named_forward(x))

    # Every decoder resolution's head must match the reference seg layer's output.
    for j in range(n_stages - 1):
        konfai_logits = trace[f"SegHead_{j}"]
        reference = oracle_seg[j]
        assert konfai_logits.shape == reference.shape
        assert torch.allclose(konfai_logits, reference, atol=1e-4), (
            f"SegHead_{j} diverges: maxdiff={(konfai_logits - reference).abs().max().item():.2e}"
        )

    # The finest head is full resolution with the requested class count.
    assert trace[f"SegHead_{n_stages - 2}"].shape == (1, num_classes, 32, 32, 32)

    # Full parameter-count equality: we build every seg head, so there is no gap with nnU-Net
    # (which always builds all of them). This is the whole point of the parametric model.
    konfai_total = sum(p.numel() for p in net.parameters())
    oracle_total = sum(p.numel() for p in oracle.parameters())
    assert konfai_total == oracle_total

    # Sanity on the leaf accounting: encoder + decoder weighted leaves.
    expected_leaves = 2 * sum([n_conv] * n_stages) + (n_stages - 1) * (1 + 2 * n_conv_decoder + 1)
    assert transferred == expected_leaves


# =========================================================================================== #
# ResidualEncoderUNet vs dynamic_network_architectures ResidualEncoderUNet (the nnU-Net ResEnc
# backbone). One config is the exact ImpactSeg "body" model (5-channel 2D input, 6 stages, 12
# classes); the others cover isotropic/anisotropic strides, anisotropic kernels, extra depth,
# and the avgpool-only residual skip (stage with a stride but no channel change).
# =========================================================================================== #
# Each config: (id, dim, in_channels, n_stages, features_per_stage, strides, kernel_sizes,
#               n_blocks_per_stage, n_conv_per_stage_decoder, num_classes).
RESENC_CONFIGS = [
    # 1. The EXACT ImpactSeg "body" model: 2D, 5 input channels, 6 stages, 12 classes. This is the
    #    real ResEnc nnU-Net whose checkpoint has 572 state-dict tensors (encoder duplicated under
    #    decoder.encoder) and 11,845,036 parameters: see the dedicated test below.
    (
        "impactseg_2d",
        2,
        5,
        6,
        [24, 48, 96, 192, 256, 256],
        [1, 2, 2, 2, 2, 2],
        3,
        [1, 2, 2, 3, 3, 3],
        [1, 1, 1, 1, 1],
        12,
    ),
    # 2. 3D, isotropic strides, uniform blocks (deeper residual encoder proves the stage loop scales).
    ("4stage_3d_isotropic", 3, 1, 4, [8, 16, 32, 64], [1, 2, 2, 2], 3, 2, 1, 2),
    # 3. 3D, per-axis anisotropic strides (TotalSegmentator / MRSeg style), varying blocks per stage,
    #    decoder depth 2.
    (
        "4stage_3d_anisotropic",
        3,
        2,
        4,
        [8, 16, 32, 64],
        [1, [1, 2, 2], [2, 2, 2], [2, 2, 2]],
        3,
        [1, 2, 2, 2],
        2,
        3,
    ),
    # 4. 2D with equal channels across a strided stage (16->16 at stride 2): exercises the ResNet-D
    #    avgpool-only skip (has_stride but no channel projection, so the skip carries NO weights).
    ("4stage_2d_avgpool_skip", 2, 3, 4, [16, 16, 32, 32], [1, 2, 2, 2], 3, [1, 2, 2, 2], 1, 2),
    # 5. ANISOTROPIC KERNELS decoupled from strides ([1, 3, 3] at stage 0 while stride is [1, 2, 2]).
    (
        "4stage_3d_anisotropic_kernels",
        3,
        1,
        4,
        [8, 16, 32, 64],
        [1, [1, 2, 2], [2, 2, 2], [2, 2, 2]],
        [[1, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
        2,
        1,
        2,
    ),
]


def _konfai_resenc(
    dim: int,
    in_channels: int,
    n_stages: int,
    features: list,
    strides: list,
    kernel_sizes: object,
    n_blocks: object,
    n_conv_decoder: object,
    num_classes: int,
) -> ResidualEncoderUNet:
    return ResidualEncoderUNet(
        dim=dim,
        in_channels=in_channels,
        n_stages=n_stages,
        features_per_stage=features,
        strides=strides,
        kernel_sizes=kernel_sizes,
        n_blocks_per_stage=n_blocks,
        n_conv_per_stage_decoder=n_conv_decoder,
        num_classes=num_classes,
        conv_bias=True,
    )


def _resenc_input(dim: int, in_channels: int) -> torch.Tensor:
    spatial = (64, 64) if dim == 2 else (32, 32, 32)
    return seeded_input(1, in_channels, *spatial)


def _expected_resenc_leaf_count(
    in_channels: int,
    n_stages: int,
    features: list,
    strides: list,
    n_blocks: object,
    n_conv_decoder: object,
) -> int:
    """Weighted leaves executed by both graphs (deep supervision on): stem + residual stages + decoder.

    Each residual block contributes 4 leaves on the main path (conv1+norm1, conv2+norm2); the first
    block of a stage adds 2 more when the channel count changes (a 1x1 projection conv + norm): a
    stride-only skip is an avgpool, which carries NO weights. The decoder mirrors PlainConvUNet.
    """
    n_blocks_list = [n_blocks] * n_stages if isinstance(n_blocks, int) else list(n_blocks)
    n_conv_list = [n_conv_decoder] * (n_stages - 1) if isinstance(n_conv_decoder, int) else list(n_conv_decoder)
    leaves = 2  # stem conv + norm
    for k in range(n_stages):
        stage_in = features[0] if k == 0 else features[k - 1]
        projection = 2 if stage_in != features[k] else 0
        leaves += (4 + projection) + 4 * (n_blocks_list[k] - 1)
    for j in range(n_stages - 1):
        leaves += 1 + 2 * n_conv_list[j] + 1  # transpose conv + conv blocks + seg head
    return leaves


@pytest.mark.parametrize(
    (
        "dim",
        "in_channels",
        "n_stages",
        "features",
        "strides",
        "kernel_sizes",
        "n_blocks",
        "n_conv_decoder",
        "num_classes",
    ),
    [config[1:] for config in RESENC_CONFIGS],
    ids=[config[0] for config in RESENC_CONFIGS],
)
def test_residualencoderunet_is_weight_exact(
    dim: int,
    in_channels: int,
    n_stages: int,
    features: list,
    strides: list,
    kernel_sizes: object,
    n_blocks: object,
    n_conv_decoder: object,
    num_classes: int,
) -> None:
    oracle = build_resenc_oracle(
        dim, in_channels, n_stages, features, strides, kernel_sizes, n_blocks, n_conv_decoder, num_classes
    )
    net = _konfai_resenc(
        dim, in_channels, n_stages, features, strides, kernel_sizes, n_blocks, n_conv_decoder, num_classes
    )

    x = _resenc_input(dim, in_channels)

    # The bridge pairs weighted leaves in forward-execution order; it raises if the graphs are not
    # weight-exact (different leaf count or a shape mismatch), so a green transfer already proves
    # structural equivalence.
    transferred = transfer_weights_by_execution_order(
        net,
        oracle,
        target_forward=lambda: list(net.named_forward(x)),
        source_forward=lambda: oracle(x),
    )
    assert transferred == _expected_resenc_leaf_count(
        in_channels, n_stages, features, strides, n_blocks, n_conv_decoder
    )

    net.eval()
    oracle.eval()
    oracle_seg = capture_oracle_seg_outputs(oracle, n_stages, lambda: oracle(x))
    with torch.no_grad():
        trace = dict(net.named_forward(x))

    # Every decoder resolution's head must match the reference seg layer's output (forward-exact,
    # not just weight-transferred).
    for j in range(n_stages - 1):
        konfai_logits = trace[f"SegHead_{j}"]
        reference = oracle_seg[j]
        assert konfai_logits.shape == reference.shape
        max_diff = (konfai_logits - reference).abs().max().item()
        assert max_diff < 1e-4, f"SegHead_{j} diverges: maxdiff={max_diff:.2e}"

    # The finest head carries the requested class count on the channel axis at full input resolution.
    assert trace[f"SegHead_{n_stages - 2}"].shape[:2] == (1, num_classes)

    # Full parameter-count equality: we build every seg head, so there is no gap with nnU-Net (which
    # always builds all of them). This is the whole point of the parametric model.
    assert sum(p.numel() for p in net.parameters()) == sum(p.numel() for p in oracle.parameters())
    assert sum(1 for _ in net.parameters()) == sum(1 for _ in oracle.parameters())


def test_impactseg_body_model_is_forward_exact() -> None:
    """The exact ImpactSeg config: forward-exact, 572-tensor checkpoint, and the deep-supervision-off
    single output equals the KonfAI finest head."""
    dim, in_channels, n_stages, features = 2, 5, 6, [24, 48, 96, 192, 256, 256]
    strides, kernel_sizes = [1, 2, 2, 2, 2, 2], 3
    n_blocks, n_conv_decoder, num_classes = [1, 2, 2, 3, 3, 3], [1, 1, 1, 1, 1], 12

    net = _konfai_resenc(
        dim, in_channels, n_stages, features, strides, kernel_sizes, n_blocks, n_conv_decoder, num_classes
    )
    # deep_supervision=True so all five seg heads execute and pair with the KonfAI all-heads graph.
    oracle_ds = build_resenc_oracle(
        dim, in_channels, n_stages, features, strides, kernel_sizes, n_blocks, n_conv_decoder, num_classes, True
    )

    x = _resenc_input(dim, in_channels)
    transferred = transfer_weights_by_execution_order(
        net,
        oracle_ds,
        target_forward=lambda: list(net.named_forward(x)),
        source_forward=lambda: oracle_ds(x),
    )
    assert transferred == 86  # 66 encoder (stem + residual stages) + 20 decoder weighted leaves

    # The real checkpoint carries 572 state-dict tensors (encoder duplicated under decoder.encoder,
    # which the KonfAI graph does not reproduce) and 11,845,036 parameters.
    assert len(oracle_ds.state_dict()) == 572
    assert sum(p.numel() for p in net.parameters()) == 11_845_036
    assert sum(p.numel() for p in net.parameters()) == sum(p.numel() for p in oracle_ds.parameters())

    net.eval()
    with torch.no_grad():
        trace = dict(net.named_forward(x))

    # The checkpoint is deep_supervision=False: it returns exactly one output. Load the transferred
    # weights into a deep_supervision=False reference and confirm its single logits map equals the
    # KonfAI finest head bit-for-bit (deep supervision only changes which heads are RETURNED).
    oracle_off = build_resenc_oracle(
        dim, in_channels, n_stages, features, strides, kernel_sizes, n_blocks, n_conv_decoder, num_classes, False
    )
    oracle_off.load_state_dict(oracle_ds.state_dict())
    oracle_off.eval()
    with torch.no_grad():
        single_output = oracle_off(x)

    finest = trace[f"SegHead_{n_stages - 2}"]
    assert finest.shape == single_output.shape == (1, num_classes, 64, 64)
    assert (finest - single_output).abs().max().item() < 1e-4


# =========================================================================================== #
# UNetPlusPlus vs smp.UnetPlusPlus (ResNet encoder + UNet++ nested decoder, built without
# importing segmentation_models_pytorch). One config is the exact ImpactSynth "MR" model
# (5-channel 2D input, 1 class), reproduced in smp forward-execution order so the 117 weighted
# leaves pair one-to-one.
# =========================================================================================== #
# Each config: (id, encoder_name, in_channels, classes, expected_leaves).
UNETPP_CONFIGS = [
    # The EXACT ImpactSynth "MR" backbone: smp.UnetPlusPlus(resnet34, in_channels=5, classes=1). The
    # smp forward runs 117 weighted leaves (72 encoder + 44 decoder + 1 seg head).
    ("impactsynth_mr_resnet34", "resnet34", 5, 1, 117),
    # A second resnet34 config (RGB-like input, 2 classes) proves the channel plumbing is parametric.
    ("resnet34_in3_cls2", "resnet34", 3, 2, 117),
    # resnet18 (BasicBlock layers [2, 2, 2, 2]) proves the encoder block-count loop scales down: fewer
    # residual blocks -> 85 weighted leaves (40 encoder + 44 decoder + 1 seg head).
    ("resnet18_in1_cls4", "resnet18", 1, 4, 85),
]


@pytest.mark.parametrize(
    ("encoder_name", "in_channels", "classes", "expected_leaves"),
    [config[1:] for config in UNETPP_CONFIGS],
    ids=[config[0] for config in UNETPP_CONFIGS],
)
def test_unetplusplus_is_weight_and_forward_exact(
    encoder_name: str, in_channels: int, classes: int, expected_leaves: int
) -> None:
    oracle = build_smp_unetplusplus(encoder_name, in_channels, classes)
    net = UNetPlusPlus(dim=2, in_channels=in_channels, classes=classes, encoder_name=encoder_name)

    x = seeded_input(1, in_channels, 64, 64)

    # The bridge pairs weighted leaves in forward-execution order; it raises if the graphs are not
    # weight-exact (different leaf count or a shape mismatch), so a green transfer already proves
    # structural equivalence.
    transferred = transfer_weights_by_execution_order(
        net,
        oracle,
        target_forward=lambda: list(net.named_forward(x)),
        source_forward=lambda: oracle(x),
    )
    assert transferred == expected_leaves

    net.eval()
    oracle.eval()
    with torch.no_grad():
        trace = dict(net.named_forward(x))
        reference = oracle(x)

    konfai_logits = trace["SegmentationHead"]
    assert konfai_logits.shape == reference.shape == (1, classes, 64, 64)
    max_diff = (konfai_logits - reference).abs().max().item()
    assert max_diff < 1e-4, f"UNetPlusPlus diverges from smp: maxdiff={max_diff:.2e}"

    # Full parameter-count equality (and identical number of parameter tensors): the reproduction has
    # no built-but-unused module gap with smp.
    assert sum(p.numel() for p in net.parameters()) == sum(p.numel() for p in oracle.parameters())
    assert sum(1 for _ in net.parameters()) == sum(1 for _ in oracle.parameters())


def test_impactsynth_mr_is_forward_exact_and_has_expected_size() -> None:
    """The exact ImpactSynth config: forward-exact, 117 weighted leaves, 26,084,881 parameters."""
    net = UNetPlusPlus(dim=2, in_channels=5, classes=1, encoder_name="resnet34")
    oracle = build_smp_unetplusplus("resnet34", 5, 1)

    x = seeded_input(1, 5, 64, 64)
    transferred = transfer_weights_by_execution_order(
        net,
        oracle,
        target_forward=lambda: list(net.named_forward(x)),
        source_forward=lambda: oracle(x),
    )
    assert transferred == 117
    assert sum(p.numel() for p in net.parameters()) == 26_084_881

    net.eval()
    oracle.eval()
    with torch.no_grad():
        finest = dict(net.named_forward(x))["SegmentationHead"]
        reference = oracle(x)
    assert finest.shape == reference.shape == (1, 1, 64, 64)
    assert (finest - reference).abs().max().item() < 1e-4


def test_unetplusplus_builds_and_forwards_without_oracle() -> None:
    classes = 3
    net = UNetPlusPlus(dim=2, in_channels=2, classes=classes, encoder_name="resnet18")
    assert isinstance(net, Network)
    assert net.get_name() == "UNetPlusPlus"

    net.eval()
    x = seeded_input(1, 2, 64, 64)
    with torch.no_grad():
        trace = dict(net.named_forward(x))

    # One terminal segmentation head at full input resolution with the requested class count.
    seg = trace["SegmentationHead"]
    assert seg.shape == (1, classes, 64, 64)

    # The dense decoder grid built all 11 nodes (10 in the triangular grid + the final x_0_4 head).
    dense_nodes = {key.split("_up")[0] for key in trace if key.endswith("_up")}
    assert dense_nodes == {f"x_{d}_{ll}" for ll in range(4) for d in range(ll + 1)} | {"x_0_4"}


def test_unetplusplus_activation_appends_a_terminal_module() -> None:
    # activation=None keeps the seg conv terminal (raw logits); a named activation adds a bounded head.
    net = UNetPlusPlus(dim=2, in_channels=1, classes=1, encoder_name="resnet18", activation="sigmoid")
    net.eval()
    x = seeded_input(1, 1, 64, 64)
    with torch.no_grad():
        trace = dict(net.named_forward(x))
    assert "Activation" in trace
    out = trace["Activation"]
    assert out.shape == (1, 1, 64, 64)
    assert out.min().item() >= 0.0 and out.max().item() <= 1.0  # sigmoid range


def test_unetplusplus_rejects_bottleneck_encoders() -> None:
    from konfai.utils.errors import ConfigError

    with pytest.raises(ConfigError):
        UNetPlusPlus(dim=2, in_channels=3, classes=1, encoder_name="resnet50")


# =========================================================================================== #
# Structural tests shared by the nnU-Net family: build and forward with deep supervision, and
# ``load`` weight preservation. They run without any oracle installed.
# =========================================================================================== #
DEEP_SUPERVISION_BUILDS = {
    "PlainConvUNet": lambda: PlainConvUNet(
        dim=3,
        in_channels=1,
        n_stages=4,
        features_per_stage=[4, 8, 16, 32],
        strides=[1, 2, 2, 2],
        num_classes=3,
    ),
    "ResidualEncoderUNet": lambda: ResidualEncoderUNet(
        dim=3,
        in_channels=1,
        n_stages=4,
        features_per_stage=[4, 8, 16, 32],
        strides=[1, 2, 2, 2],
        n_blocks_per_stage=[1, 2, 2, 2],
        n_conv_per_stage_decoder=1,
        num_classes=3,
    ),
}


@pytest.mark.parametrize("model_name", sorted(DEEP_SUPERVISION_BUILDS), ids=sorted(DEEP_SUPERVISION_BUILDS))
def test_builds_and_forwards_deep_supervision(model_name: str) -> None:
    n_stages = 4
    num_classes = 3
    net = DEEP_SUPERVISION_BUILDS[model_name]()
    assert isinstance(net, Network)
    assert net.get_name() == model_name

    net.eval()
    x = seeded_input(1, 1, 32, 32, 32)
    with torch.no_grad():
        trace = dict(net.named_forward(x))

    # One deep-supervision head per decoder resolution.
    seg_heads = [key for key in trace if key.startswith("SegHead_")]
    assert len(seg_heads) == n_stages - 1

    # Every head carries the requested class count on the channel axis.
    for key in seg_heads:
        assert trace[key].shape[:2] == (1, num_classes)

    # The finest head is full resolution; the coarsest is progressively downsampled. With
    # input 32 and strides [1, 2, 2, 2] the encoder resolutions are 32/16/8/4, so decoder
    # stage 0 (skip = encoder stage 2) runs at 8 and the finest stage at 32.
    assert trace[f"SegHead_{n_stages - 2}"].shape == (1, num_classes, 32, 32, 32)
    assert trace["SegHead_0"].shape == (1, num_classes, 8, 8, 8)


LOAD_PRESERVATION_BUILDS = {
    "PlainConvUNet": lambda: PlainConvUNet(
        dim=3,
        in_channels=1,
        n_stages=3,
        features_per_stage=[4, 8, 16],
        strides=[1, 2, 2],
        num_classes=2,
    ),
    "ResidualEncoderUNet": lambda: ResidualEncoderUNet(
        dim=3,
        in_channels=1,
        n_stages=3,
        features_per_stage=[4, 8, 16],
        strides=[1, 2, 2],
        n_blocks_per_stage=[1, 2, 2],
        n_conv_per_stage_decoder=1,
        num_classes=2,
    ),
    "UNetPlusPlus": lambda: UNetPlusPlus(dim=2, in_channels=1, classes=1, encoder_name="resnet18"),
}


@pytest.mark.parametrize("model_name", sorted(LOAD_PRESERVATION_BUILDS), ids=sorted(LOAD_PRESERVATION_BUILDS))
def test_load_does_not_reinitialise_weights(model_name: str) -> None:
    # The trainer calls load(init=True) at start-up; every oracle-backed model must force init=False
    # so a transferred checkpoint (or any loaded weights) is never overwritten with init noise.
    net = LOAD_PRESERVATION_BUILDS[model_name]()
    snapshot = {name: param.detach().clone() for name, param in net.named_parameters()}

    # Empty state dict: with the override, load(init=True) applies no init and loads nothing, so
    # every parameter is preserved. Without the override, init=True would re-initialise them.
    net.load({}, init=True)

    for name, param in net.named_parameters():
        assert torch.equal(param, snapshot[name]), f"parameter {name} was re-initialised by load(init=True)"
