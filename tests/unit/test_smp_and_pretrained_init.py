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

"""The SMP encoder-backed wrapper and the pretrained-weights-survive-init contract.

``trainer`` calls ``model.load(state_dict, init=True)`` at training start; for a model that
arrives fully constructed (MinimalModel-wrapped external class, SMP with a pretrained
encoder), re-initialisation would silently destroy the pretrained weights with
``init_type`` noise. ``MinimalModel.init`` and ``SMP.init`` are therefore no-ops — locked here.
"""

import pytest
import torch
from konfai.network.network import MinimalModel, Network
from konfai.utils.errors import ConfigError


def _first_conv_weight(module: torch.nn.Module) -> torch.Tensor:
    for child in module.modules():
        if isinstance(child, torch.nn.Conv2d):
            return child.weight.detach().clone()
    raise AssertionError("no Conv2d found")


def test_minimal_model_preserves_external_weights_through_load_init() -> None:
    external = torch.nn.Sequential(torch.nn.Conv2d(1, 4, 3, padding=1), torch.nn.ReLU())
    before = _first_conv_weight(external)

    wrapped = MinimalModel(external, dim=2)
    wrapped.set_name("External")
    wrapped.load({}, init=True)  # what the trainer does at a fresh training start

    after = _first_conv_weight(wrapped)
    assert torch.equal(before, after), "load(init=True) must not clobber an external model's weights"


def test_plain_network_still_gets_initialised() -> None:
    # Models built from scratch keep KonfAI's init behaviour (only fully-constructed
    # externals are exempt): a plain Network child IS re-initialised by load(init=True).
    net = Network(in_channels=1, dim=2, init_type="normal", init_gain=5.0)
    net.add_module("Conv", torch.nn.Conv2d(1, 4, 3, padding=1))
    net.set_name("Scratch")
    before = _first_conv_weight(net)
    net.load({}, init=True)
    after = _first_conv_weight(net)
    assert not torch.equal(before, after), "a from-scratch Network must still be initialised"


def test_smp_wrapper_builds_forwards_and_keeps_pretrained_encoder() -> None:
    smp = pytest.importorskip("segmentation_models_pytorch")

    from konfai.models.python.segmentation.smp import SMP

    model = SMP(arch="Unet", encoder_name="resnet18", encoder_weights="imagenet", in_channels=3, classes=2)
    model.set_name("SMP")
    before = _first_conv_weight(model)

    # Same weights as the raw SMP construction (wrapping adds nothing, removes nothing).
    raw = smp.create_model(arch="Unet", encoder_name="resnet18", encoder_weights="imagenet", in_channels=3, classes=2)
    raw.eval()
    model.eval()
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        expected = raw(x)
        ours = None
        for _, out in model.named_forward(x):
            ours = out
    assert ours is not None and ours.shape == expected.shape

    # The ImageNet encoder survives the trainer's load(init=True).
    model.load({}, init=True)
    after = _first_conv_weight(model)
    assert torch.equal(before, after), "the pretrained SMP encoder must survive load(init=True)"


def test_smp_wrapper_is_2d_only() -> None:
    pytest.importorskip("segmentation_models_pytorch")
    from konfai.models.python.segmentation.smp import SMP

    with pytest.raises(ConfigError, match="2D-only"):
        SMP(dim=3)
