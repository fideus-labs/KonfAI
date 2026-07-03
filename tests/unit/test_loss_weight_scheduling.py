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

"""Regression test: the gradient loss uses the current weight, not a stale first one."""

import torch
from konfai.network.network import Measure


def _loss_record() -> Measure.Loss:
    return Measure.Loss("l", "out", "tgt", 0, is_loss=True, accumulation=False)


def test_get_loss_uses_current_iteration_weight() -> None:
    # reset_loss clears _loss every iteration but _weight keeps growing for the logging windows.
    # get_loss must pair the current loss with the current weight; the pre-fix code zipped from the
    # front, so a loss-weight scheduler that changes the weight had no effect on the gradient.
    record = _loss_record()

    record.reset_loss()
    record.add(2.0, torch.tensor([3.0]))
    assert record.get_loss().item() == 6.0  # 2 * 3

    record.reset_loss()  # next iteration; _weight is now [2.0, 5.0]
    record.add(5.0, torch.tensor([1.0]))
    assert record.get_loss().item() == 5.0  # 5 * 1, not the stale 2 * 1


def test_get_loss_handles_multiple_accumulated_patches() -> None:
    # Accumulation mode adds several (weight, loss) pairs per iteration; the trailing weights must
    # still line up one-to-one with the current losses.
    record = _loss_record()

    record.reset_loss()
    record.add(1.0, torch.tensor([10.0]))  # a previous iteration leaves a weight behind
    record.reset_loss()
    record.add(0.5, torch.tensor([2.0]))
    record.add(0.5, torch.tensor([4.0]))

    assert record.get_loss().item() == 1.5  # mean(0.5 * 2, 0.5 * 4)
