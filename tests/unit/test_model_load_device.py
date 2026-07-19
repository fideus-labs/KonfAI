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

import pytest
import torch
from konfai.network.network import Network
from konfai.predictor import Mean, ModelComposite, _colocate_loaded_modules


class _LateHeadNetwork(Network):
    """Model whose load() appends a checkpoint-sized module, mimicking TotalSegmentator's Head.Conv."""

    def __init__(self) -> None:
        super().__init__(in_channels=1)
        self.add_module("Stem", torch.nn.Conv3d(1, 2, kernel_size=1))

    def load(self, state_dict, init: bool = True, ema: bool = False):  # type: ignore[override]
        # A head sized from the checkpoint, created at load time -> defaults to CPU.
        self.add_module("Head", torch.nn.Conv3d(2, int(state_dict["nb_class"]), kernel_size=1))

    def forward(self, batch_sample, output_layers=[]):  # type: ignore[override]
        return []


def test_colocate_is_a_safe_noop_when_model_is_all_cpu() -> None:
    # With no device-placed parameter there is nothing to co-locate; the helper must be a no-op.
    model = torch.nn.Sequential(torch.nn.Conv3d(1, 2, 1), torch.nn.Conv3d(2, 3, 1))
    _colocate_loaded_modules(model)
    assert all(not p.is_cuda for p in model.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="device co-location only manifests on GPU")
def test_ensemble_load_colocates_late_added_head_on_gpu() -> None:
    # The TotalSegmentator pattern: the model is placed on the GPU, then a per-model load() appends
    # a Head on CPU. The forward must not hit "Input cuda, weight CPU".
    composite = ModelComposite(_LateHeadNetwork(), Mean())
    Network.to(composite, 0)  # place on cuda:0, exactly as the predictor does before inference

    composite.load([{"nb_class": 5}])  # single source -> triggers _ensure_model_loaded(0)
    model = composite["Model_0"]

    head = dict(model.named_modules())["Head"]
    assert list(head.parameters()), "test setup: Head should have parameters"
    assert all(p.is_cuda for p in head.parameters()), "load-added Head must be co-located onto the GPU"
    # the whole model must live on a single device
    assert len({p.device for p in model.parameters()}) == 1
