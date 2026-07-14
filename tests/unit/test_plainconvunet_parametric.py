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

"""The parametric PlainConvUNet is weight-exact with nnU-Net's PlainConvUNet at any topology.

``konfai.models.python.segmentation.plainconvunet.PlainConvUNet`` builds an encoder/decoder
graph of arbitrary depth from the nnU-Net hyper-parameters. These tests transfer a **real**
``dynamic_network_architectures.architectures.unet.PlainConvUNet`` (deep supervision on, so all
segmentation heads are built) into the KonfAI graph through the execution-order bridge and
assert, for **every** decoder resolution, that the KonfAI seg head's logits are ``torch.allclose``
with the reference seg layer's output -- plus full parameter-count equality (no built-but-unused
head gap). The topologies cover isotropic and anisotropic strides, extra depth, and a non-default
per-stage conv count, which is the whole point: any nnU-Net checkpoint of any depth must load.
"""

from collections.abc import Callable

import pytest
import torch
from konfai.models.python.segmentation.plainconvunet import PlainConvUNet
from konfai.network.network import Network
from konfai.utils.pretrained import transfer_weights_by_execution_order

# Each config: (id, n_stages, features_per_stage, strides, kernel_sizes, n_conv_per_stage,
#               n_conv_per_stage_decoder, num_classes).
CONFIGS = [
    # 1. 4 stages, isotropic strides, kernel 3, 3D (nnU-Net 3D full-res default depth).
    ("4stage_isotropic", 4, [8, 16, 32, 64], [1, 2, 2, 2], 3, 2, 2, 2),
    # 2. 5 stages, isotropic strides, 3D (deeper -- proves the stage loop scales).
    ("5stage_isotropic", 5, [8, 16, 32, 64, 128], [1, 2, 2, 2, 2], 3, 2, 2, 2),
    # 3. 4 stages with per-axis anisotropic strides (TotalSegmentator / MRSeg style).
    ("4stage_anisotropic", 4, [8, 16, 32, 64], [1, [1, 2, 2], [2, 2, 2], [2, 2, 2]], 3, 2, 2, 2),
    # 4. Non-default per-stage conv count, with encoder != decoder depth, on a small config
    #    (proves the encoder and decoder per-stage loops are independent and correct).
    ("3stage_nconv3", 3, [8, 16, 32], [1, 2, 2], 3, 3, 2, 2),
    # 5. ANISOTROPIC KERNELS decoupled from strides ([1, 3, 3] at stage 0 while stride is [1, 2, 2])
    #    -- kernel_sizes and strides are independent in nnU-Net; a real anisotropic-spacing plan
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
    # 6. Per-stage isotropic kernels != 3 (kernel 1 at the stem, kernel 5 deeper) -- proves padding
    #    is derived from the kernel per stage, not hardcoded.
    ("4stage_mixed_kernels", 4, [8, 16, 32, 64], [1, 2, 2, 2], [1, 3, 5, 3], 2, 2, 2),
]


def _build_oracle(
    n_stages: int,
    features: list,
    strides: list,
    kernel_sizes: object,
    n_conv: int,
    n_conv_decoder: int,
    num_classes: int,
) -> torch.nn.Module:
    pytest.importorskip("dynamic_network_architectures")
    from dynamic_network_architectures.architectures.unet import PlainConvUNet as OraclePlainConvUNet

    return OraclePlainConvUNet(
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
        deep_supervision=True,  # build ALL decoder seg heads, like a real nnU-Net checkpoint
    )


def _build_konfai(
    n_stages: int,
    features: list,
    strides: list,
    kernel_sizes: object,
    n_conv: int,
    n_conv_decoder: int,
    num_classes: int,
) -> PlainConvUNet:
    return PlainConvUNet(
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


def _capture_oracle_seg_outputs(
    oracle: torch.nn.Module, n_stages: int, run: Callable[[], object]
) -> dict[int, torch.Tensor]:
    """Return the oracle's per-stage seg-layer outputs indexed by decoder stage (coarsest-first).

    nnU-Net returns the seg outputs finest-first; hooking the ``seg_layers`` directly captures
    them by build/execution index instead, which is the order the KonfAI ``SegHead_j`` heads use.
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


@pytest.mark.parametrize(
    ("n_stages", "features", "strides", "kernel_sizes", "n_conv", "n_conv_decoder", "num_classes"),
    [config[1:] for config in CONFIGS],
    ids=[config[0] for config in CONFIGS],
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
    oracle = _build_oracle(n_stages, features, strides, kernel_sizes, n_conv, n_conv_decoder, num_classes)
    net = _build_konfai(n_stages, features, strides, kernel_sizes, n_conv, n_conv_decoder, num_classes)

    torch.manual_seed(0)
    x = torch.randn(1, 1, 32, 32, 32)

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
    oracle_seg = _capture_oracle_seg_outputs(oracle, n_stages, lambda: oracle(x))
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


# --------------------------------------------------------------------------- #
# Structural test: builds and forwards without the oracle (runs on any CI).
# --------------------------------------------------------------------------- #
def test_plainconvunet_builds_and_forwards_deep_supervision() -> None:
    n_stages = 4
    num_classes = 3
    net = PlainConvUNet(
        dim=3,
        in_channels=1,
        n_stages=n_stages,
        features_per_stage=[4, 8, 16, 32],
        strides=[1, 2, 2, 2],
        num_classes=num_classes,
    )
    assert isinstance(net, Network)
    assert net.get_name() == "PlainConvUNet"

    net.eval()
    torch.manual_seed(0)
    x = torch.randn(1, 1, 32, 32, 32)
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


def test_plainconvunet_load_does_not_reinitialise_weights() -> None:
    # The trainer calls load(init=True) at start-up; PlainConvUNet must force init=False so a
    # transferred nnU-Net checkpoint (or any loaded weights) is never overwritten with init noise.
    net = PlainConvUNet(
        dim=3,
        in_channels=1,
        n_stages=3,
        features_per_stage=[4, 8, 16],
        strides=[1, 2, 2],
        num_classes=2,
    )
    snapshot = {name: param.detach().clone() for name, param in net.named_parameters()}

    # Empty state dict: with the override, load(init=True) applies no init and loads nothing, so
    # every parameter is preserved. Without the override, init=True would re-initialise them.
    net.load({}, init=True)

    for name, param in net.named_parameters():
        assert torch.equal(param, snapshot[name]), f"parameter {name} was re-initialised by load(init=True)"
