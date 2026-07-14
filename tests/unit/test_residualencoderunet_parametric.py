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

"""The parametric ResidualEncoderUNet is weight-exact with nnU-Net's ResEnc U-Net at any topology.

``konfai.models.python.segmentation.residualencoderunet.ResidualEncoderUNet`` builds an
encoder/decoder graph of arbitrary depth from the nnU-Net ResEnc hyper-parameters. These tests
transfer a **real** ``dynamic_network_architectures.architectures.unet.ResidualEncoderUNet`` (deep
supervision on, so every segmentation head is built AND executed) into the KonfAI graph through the
execution-order bridge and assert, for **every** decoder resolution, that the KonfAI seg head's
logits are ``torch.allclose`` with the reference seg layer's output -- plus full parameter-count
equality. One config is the exact ImpactSeg "body" model (5-channel 2D input, 6 stages, 12 classes,
deep supervision off in the checkpoint); the others cover isotropic/anisotropic strides, anisotropic
kernels, extra depth, and the avgpool-only residual skip (stage with a stride but no channel change).
"""

from collections.abc import Callable

import pytest
import torch
from konfai.models.python.segmentation.residualencoderunet import ResidualEncoderUNet
from konfai.network.network import Network
from konfai.utils.pretrained import transfer_weights_by_execution_order

# Each config: (id, dim, in_channels, n_stages, features_per_stage, strides, kernel_sizes,
#               n_blocks_per_stage, n_conv_per_stage_decoder, num_classes).
CONFIGS = [
    # 1. The EXACT ImpactSeg "body" model: 2D, 5 input channels, 6 stages, 12 classes. This is the
    #    real ResEnc nnU-Net whose checkpoint has 572 state-dict tensors (encoder duplicated under
    #    decoder.encoder) and 11,845,036 parameters -- see the dedicated test below.
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


def _build_oracle(
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
    pytest.importorskip("dynamic_network_architectures")
    from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet as OracleResEncUNet

    conv_op = torch.nn.Conv2d if dim == 2 else torch.nn.Conv3d
    norm_op = torch.nn.InstanceNorm2d if dim == 2 else torch.nn.InstanceNorm3d
    return OracleResEncUNet(
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
        deep_supervision=deep_supervision,  # build AND run ALL seg heads, matching the KonfAI graph
    )


def _build_konfai(
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


def _random_input(dim: int, in_channels: int) -> torch.Tensor:
    torch.manual_seed(0)
    spatial = (64, 64) if dim == 2 else (32, 32, 32)
    return torch.randn(1, in_channels, *spatial)


def _capture_oracle_seg_outputs(
    oracle: torch.nn.Module, n_stages: int, run: Callable[[], object]
) -> dict[int, torch.Tensor]:
    """Return the oracle's per-stage seg-layer outputs indexed by decoder stage (coarsest-first).

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


def _expected_leaf_count(
    in_channels: int,
    n_stages: int,
    features: list,
    strides: list,
    n_blocks: object,
    n_conv_decoder: object,
) -> int:
    """Weighted leaves executed by both graphs (deep supervision on): stem + residual stages + decoder.

    Each residual block contributes 4 leaves on the main path (conv1+norm1, conv2+norm2); the first
    block of a stage adds 2 more when the channel count changes (a 1x1 projection conv + norm) -- a
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
    [config[1:] for config in CONFIGS],
    ids=[config[0] for config in CONFIGS],
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
    oracle = _build_oracle(
        dim, in_channels, n_stages, features, strides, kernel_sizes, n_blocks, n_conv_decoder, num_classes
    )
    net = _build_konfai(
        dim, in_channels, n_stages, features, strides, kernel_sizes, n_blocks, n_conv_decoder, num_classes
    )

    x = _random_input(dim, in_channels)

    # The bridge pairs weighted leaves in forward-execution order; it raises if the graphs are not
    # weight-exact (different leaf count or a shape mismatch), so a green transfer already proves
    # structural equivalence.
    transferred = transfer_weights_by_execution_order(
        net,
        oracle,
        target_forward=lambda: list(net.named_forward(x)),
        source_forward=lambda: oracle(x),
    )
    assert transferred == _expected_leaf_count(in_channels, n_stages, features, strides, n_blocks, n_conv_decoder)

    net.eval()
    oracle.eval()
    oracle_seg = _capture_oracle_seg_outputs(oracle, n_stages, lambda: oracle(x))
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

    net = _build_konfai(
        dim, in_channels, n_stages, features, strides, kernel_sizes, n_blocks, n_conv_decoder, num_classes
    )
    # deep_supervision=True so all five seg heads execute and pair with the KonfAI all-heads graph.
    oracle_ds = _build_oracle(
        dim, in_channels, n_stages, features, strides, kernel_sizes, n_blocks, n_conv_decoder, num_classes, True
    )

    x = _random_input(dim, in_channels)
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
    oracle_off = _build_oracle(
        dim, in_channels, n_stages, features, strides, kernel_sizes, n_blocks, n_conv_decoder, num_classes, False
    )
    oracle_off.load_state_dict(oracle_ds.state_dict())
    oracle_off.eval()
    with torch.no_grad():
        single_output = oracle_off(x)

    finest = trace[f"SegHead_{n_stages - 2}"]
    assert finest.shape == single_output.shape == (1, num_classes, 64, 64)
    assert (finest - single_output).abs().max().item() < 1e-4


# --------------------------------------------------------------------------- #
# Structural tests: build and forward without the oracle (run on any CI).
# --------------------------------------------------------------------------- #
def test_residualencoderunet_builds_and_forwards_deep_supervision() -> None:
    n_stages = 4
    num_classes = 3
    net = ResidualEncoderUNet(
        dim=3,
        in_channels=1,
        n_stages=n_stages,
        features_per_stage=[4, 8, 16, 32],
        strides=[1, 2, 2, 2],
        n_blocks_per_stage=[1, 2, 2, 2],
        n_conv_per_stage_decoder=1,
        num_classes=num_classes,
    )
    assert isinstance(net, Network)
    assert net.get_name() == "ResidualEncoderUNet"

    net.eval()
    torch.manual_seed(0)
    x = torch.randn(1, 1, 32, 32, 32)
    with torch.no_grad():
        trace = dict(net.named_forward(x))

    # One deep-supervision head per decoder resolution.
    seg_heads = [key for key in trace if key.startswith("SegHead_")]
    assert len(seg_heads) == n_stages - 1
    for key in seg_heads:
        assert trace[key].shape[:2] == (1, num_classes)

    # Finest head is full resolution; coarsest is progressively downsampled. Input 32 with strides
    # [1, 2, 2, 2] gives encoder resolutions 32/16/8/4, so decoder stage 0 (skip = encoder stage 2)
    # runs at 8 and the finest stage at 32.
    assert trace[f"SegHead_{n_stages - 2}"].shape == (1, num_classes, 32, 32, 32)
    assert trace["SegHead_0"].shape == (1, num_classes, 8, 8, 8)


def test_residualencoderunet_load_does_not_reinitialise_weights() -> None:
    # The trainer calls load(init=True) at start-up; ResidualEncoderUNet must force init=False so a
    # transferred nnU-Net checkpoint (or any loaded weights) is never overwritten with init noise.
    net = ResidualEncoderUNet(
        dim=3,
        in_channels=1,
        n_stages=3,
        features_per_stage=[4, 8, 16],
        strides=[1, 2, 2],
        n_blocks_per_stage=[1, 2, 2],
        n_conv_per_stage_decoder=1,
        num_classes=2,
    )
    snapshot = {name: param.detach().clone() for name, param in net.named_parameters()}

    net.load({}, init=True)

    for name, param in net.named_parameters():
        assert torch.equal(param, snapshot[name]), f"parameter {name} was re-initialised by load(init=True)"
