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

"""The declarative UNet.yml must build a model equivalent to the Python UNet.

This locks the key "declarative models can replace Python models" property for
the feed-forward subset: the shipped example ``examples/Segmentation/UNet.yml``
must produce a graph with the same parameter count and forward behaviour as the
hand-written ``konfai.models.segmentation.UNet`` configured identically.
"""

from pathlib import Path

import torch
from konfai.network.blocks import BlockConfig
from konfai.utils.model_builder import build_model_from_yaml

UNET_YML = Path(__file__).resolve().parents[2] / "examples" / "Segmentation" / "UNet.yml"


def _build_yaml_unet():
    params = {"dim": 2, "channels": [1, 32, 64, 128, 256], "nb_class": 41}
    return build_model_from_yaml(yaml_path=str(UNET_YML), parameters=params)


def test_example_unet_yaml_builds_and_forwards():
    net = _build_yaml_unet()
    x = torch.randn(1, 1, 64, 64)
    with torch.no_grad():
        y = net.forward_tensor(x)
    assert y.shape == (1, 1, 64, 64)  # ArgMax head -> single index channel


def test_example_unet_yaml_matches_python_unet_param_count():
    from konfai.models.segmentation.UNet import UNet

    yaml_net = _build_yaml_unet()
    block_config = BlockConfig(kernel_size=3, stride=1, padding=1, bias=True, activation="ReLU", norm_mode="NONE")
    python_net = UNet(
        dim=2,
        channels=[1, 32, 64, 128, 256],
        nb_class=41,
        block_config=block_config,
        nb_conv_per_stage=2,
        downsample_mode="MAXPOOL",
        upsample_mode="CONV_TRANSPOSE",
        attention=False,
        block_type="Conv",
    )
    n_yaml = sum(p.numel() for p in yaml_net.parameters())
    n_python = sum(p.numel() for p in python_net.parameters())
    assert n_yaml == n_python, f"yaml={n_yaml} python={n_python}"
