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

"""Regression test: PerceptualLoss.forward must unpack its targets into _compute."""

import torch
from konfai.metric.measure import PerceptualLoss


def test_perceptual_loss_forward_unpacks_targets() -> None:
    # forward(output, *targets) must hand each target to _compute(output, *targets) as its own
    # positional tensor; the pre-fix code passed the whole tuple as a single argument, so the
    # preprocessing/feature-extraction path received a tuple and crashed.
    loss = object.__new__(PerceptualLoss)
    loss.shape = [128, 128, 128]  # len != 2 -> the non-slice branch is taken
    loss.models = {None: object()}  # short-circuit the lazy model placement on device index None

    recorded: dict[str, tuple] = {}

    def fake_compute(output, *targets):
        recorded["targets"] = targets
        return torch.zeros(1)

    loss._compute = fake_compute  # type: ignore[method-assign]

    PerceptualLoss.forward(loss, torch.randn(1, 1, 8, 8), torch.randn(1, 1, 8, 8))

    assert len(recorded["targets"]) == 1
    assert torch.is_tensor(recorded["targets"][0])
