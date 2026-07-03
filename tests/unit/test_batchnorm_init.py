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

"""Regression test: BatchNorm gamma must initialise around 1, not 0."""

import torch
from konfai.network.network import ModuleArgsDict


def test_init_func_centres_batchnorm_gamma_on_one() -> None:
    # gamma initialised around 0 scaled the normalised activations to ~0, stalling early training.
    batch_norm = torch.nn.BatchNorm2d(128)

    ModuleArgsDict.init_func(batch_norm, "normal", 0.02)

    assert abs(batch_norm.weight.mean().item() - 1.0) < 0.02
    assert batch_norm.bias.abs().max().item() < 1e-6
