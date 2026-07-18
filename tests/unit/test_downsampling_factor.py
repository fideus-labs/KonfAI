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
"""``Network.downsampling_factor`` reads the per-axis input divisor off the graph (the product of the
encoder's downsampling strides), used to size a free patch axis to a valid extent for the model."""

import pytest
import torch
from konfai.models.python.segmentation.plainconvunet import PlainConvUNet
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


def test_downsampling_factor_ignores_the_residual_avgpool_and_transpose_upsample():
    """Only MaxPool and stride>1 Conv shrink the skip path; a residual AvgPool and a decoder
    ConvTranspose must not inflate the factor (they would double-count one level)."""

    class _OneLevel(Network):
        def __init__(self) -> None:
            super().__init__()
            self.add_module("down", torch.nn.Conv3d(1, 4, 3, stride=2, padding=1))  # counts
            self.add_module("residual", torch.nn.AvgPool3d(2, stride=2))  # must NOT count
            self.add_module("up", torch.nn.ConvTranspose3d(4, 4, 2, stride=2))  # must NOT count

    assert _OneLevel().downsampling_factor() == [2, 2, 2]
