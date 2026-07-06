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

"""Tests for the learning-rate schedulers in ``konfai.metric.schedulers``."""

import torch
from konfai.metric.schedulers import PolyLRScheduler


def test_polylr_resync_resumes_from_last_epoch():
    """#scheduler: PolyLR must honour a resync that sets last_epoch (RESUME fast-forward)."""
    param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([param], lr=0.1)
    scheduler = PolyLRScheduler(opt, initial_lr=0.1, max_steps=100)

    # A freshly built PolyLR keeps last_epoch == -1 so the network resync guard fires.
    assert scheduler.last_epoch == -1

    # Network.load() resync: fast-forward to iteration 50.
    scheduler.last_epoch = 50
    scheduler.step()

    expected = 0.1 * (1 - 50 / 100) ** 0.9
    assert opt.param_groups[0]["lr"] == expected
    assert scheduler.last_epoch == 51


def test_polylr_fresh_run_unchanged():
    """Fresh training (no resync) still steps from the internal counter 0, 1, 2 ..."""
    param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([param], lr=0.1)
    scheduler = PolyLRScheduler(opt, initial_lr=0.1, max_steps=100)

    lrs = []
    for _ in range(3):
        scheduler.step()
        lrs.append(opt.param_groups[0]["lr"])

    assert lrs[0] == 0.1 * (1 - 1 / 100) ** 0.9
    assert lrs[1] == 0.1 * (1 - 2 / 100) ** 0.9
    assert lrs[2] == 0.1 * (1 - 3 / 100) ** 0.9
    assert scheduler.last_epoch == -1
