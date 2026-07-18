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
"""``Network.downsampling_factor`` reads the per-axis input divisor off the graph -- the coarsest
downsampling the branch register reaches -- used to size a free patch axis to a valid extent for the
model. Parallel branches (a residual shortcut beside the main path) reduce the same level once."""

import pytest
import torch
from konfai.models.python.classification.resnet import ResBlock
from konfai.models.python.segmentation.plainconvunet import PlainConvUNet
from konfai.network import blocks
from konfai.network.network import Network

# (id, strides, expected per-axis divisor). The divisor is the product of the encoder strides per axis.
CONFIGS = [
    ("isotropic_5stage", [1, 2, 2, 2, 2], [16, 16, 16]),
    ("anisotropic_totalseg", [1, [1, 2, 2], [2, 2, 2], [2, 2, 2]], [4, 8, 8]),
    ("shallow", [1, 2], [2, 2, 2]),
]


@pytest.mark.parametrize("name, strides, expected", CONFIGS, ids=[c[0] for c in CONFIGS])
def test_downsampling_factor_reads_encoder_strides(name, strides, expected):
    model = PlainConvUNet(
        dim=3,
        in_channels=1,
        n_stages=len(strides),
        features_per_stage=[8 * 2**i for i in range(len(strides))],
        strides=strides,
        num_classes=2,
    )
    assert model.downsampling_factor() == expected


def test_downsampling_factor_none_without_downsampling():
    """A model that never downsamples imposes no divisibility constraint."""

    class _Flat(Network):
        def __init__(self) -> None:
            super().__init__()
            self.add_module("conv", torch.nn.Conv3d(1, 4, 3, padding=1))

    assert _Flat().downsampling_factor() is None


def test_downsampling_factor_counts_a_main_path_avgpool_but_not_the_decoder_upsample():
    """AvgPool IS a downsampler (a model may pool on its main path instead of striding); it must count,
    or the factor undercounts and the model receives a non-divisible extent. A decoder ConvTranspose
    grows the grid and must not count."""

    class _OneLevel(Network):
        def __init__(self) -> None:
            super().__init__()
            self.add_module("down", torch.nn.AvgPool3d(2, stride=2))  # main-path pool: counts
            self.add_module("conv", torch.nn.Conv3d(1, 4, 3, padding=1))
            self.add_module("up", torch.nn.ConvTranspose3d(4, 4, 2, stride=2))  # must NOT count

    assert _OneLevel().downsampling_factor() == [2, 2, 2]


def test_downsampling_factor_does_not_inflate_a_parallel_residual_avgpool():
    """A residual shortcut's AvgPool runs PARALLEL to the strided main conv (KonfAI's ResNet-D block):
    both reduce the same level, merged by the ``Add``. Counting AvgPool must not double-count here — the
    branch-aware max sees the two parallel /2 branches as one /2, not /4."""

    class _ResDBlock(Network):
        def __init__(self) -> None:
            super().__init__()
            self.add_module("Main", torch.nn.Conv3d(1, 4, 3, stride=2, padding=1))  # branch 0: /2
            self.add_module("SkipPool", torch.nn.AvgPool3d(2, stride=2), in_branch=[1], out_branch=[1])  # /2
            self.add_module("Add", blocks.Add(), in_branch=[0, 1])

    assert _ResDBlock().downsampling_factor() == [2, 2, 2]


def test_downsampling_factor_counts_a_parallel_strided_shortcut_once():
    """A torchvision-style residual block (KonfAI ``ResBlock``) strides its main conv AND its
    projection shortcut in parallel -- both reduce the SAME level, merged by the residual ``Add``. A
    flat ``modules()`` walk multiplies the two strides and double-counts the level; the branch trace
    follows the two parallel branches to their merge and counts it once. Two strided blocks reduce by
    2 each -> [4, 4, 4], not the [16, 16, 16] the double-count would report."""

    class _TwoStridedResiduals(Network):
        def __init__(self) -> None:
            super().__init__()
            self.add_module("Block_0", ResBlock(1, 4, downsample=True, dim=3))
            self.add_module("Block_1", ResBlock(4, 8, downsample=True, dim=3))

    assert _TwoStridedResiduals().downsampling_factor() == [4, 4, 4]


def test_downsampling_factor_sees_inside_an_opaque_wrapped_module():
    """A third-party net added as ONE plain leaf (the smp/torchvision wrapping pattern) hides its graph
    from the branch trace; its internal strides must still count — a lost factor disables the free-axis
    rounding entirely and the model crashes on a non-divisible extent."""

    class _Wrapped(Network):
        def __init__(self) -> None:
            super().__init__()
            self.add_module(
                "Model",
                torch.nn.Sequential(
                    torch.nn.Conv2d(1, 8, 3, stride=2, padding=1),
                    torch.nn.Conv2d(8, 16, 3, stride=2, padding=1),
                    torch.nn.ConvTranspose2d(16, 8, 2, stride=2),
                ),
            )

    assert _Wrapped().downsampling_factor() == [4, 4]


def test_downsampling_factor_aligns_mixed_dimensionalities_to_the_trailing_axes():
    """A leaf of lower dimensionality (a 2D side head in a 3D graph) acts on the LAST axes: it must
    neither crash the computation nor lock the factor's rank to 2."""

    class _Mixed(Network):
        def __init__(self) -> None:
            super().__init__()
            self.add_module("head2d", torch.nn.Conv2d(1, 4, 3, stride=1, padding=1))  # stride 1: inert
            self.add_module("down_a", torch.nn.Conv3d(1, 4, 3, stride=2, padding=1))
            self.add_module("down_b", torch.nn.Conv3d(4, 8, 3, stride=2, padding=1))
            self.add_module("slice_pool", torch.nn.MaxPool2d(2))  # 2D stride 2 -> trailing [1, 2, 2]

    assert _Mixed().downsampling_factor() == [4, 8, 8]


def test_downsampling_factor_seeds_a_nested_block_from_all_its_inputs():
    """A nested routed block reading several branches at DIFFERENT factors must see each at its own
    resolution: if it downsamples a non-first input, seeding every internal branch from the first input
    would undercount that deeper level. Here a block takes a /2 branch and a /8 branch and pools the /8
    one further to /16 -- that /16 is the graph's coarsest level and must surface."""

    class _Deeper(Network):
        def __init__(self) -> None:
            super().__init__()
            # Build two branches at different depths: branch 0 at /2, branch 'deep' at /8.
            self.add_module("Shallow", torch.nn.Conv3d(1, 4, 3, stride=2, padding=1), out_branch=["shallow"])
            self.add_module("Deep", torch.nn.Conv3d(1, 4, 3, stride=8, padding=1), out_branch=["deep"])

            class _Fuse(Network):
                def __init__(self) -> None:
                    super().__init__()
                    # Reads [shallow (/2), deep (/8)]; pools the SECOND input (branch '1') to /16.
                    self.add_module("PoolDeep", torch.nn.MaxPool3d(2), in_branch=[1], out_branch=[1])
                    self.add_module("Merge", blocks.Add(), in_branch=[1, 0])

            self.add_module("Fuse", _Fuse(), in_branch=["shallow", "deep"], out_branch=["fused"])

    assert _Deeper().downsampling_factor() == [16, 16, 16]
