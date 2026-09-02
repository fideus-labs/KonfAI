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

"""Every scheduler documented in docs/source/reference/components/schedulers.md instantiates and
steps once: a torch signature change (LambdaLR dropped ``verbose``) otherwise ships as a
config-reachable crash that no test sees."""

import pytest
import torch
from konfai.metric.schedulers import Constant, CosineAnnealing, PolyLRScheduler, Warmup


def _optimizer() -> torch.optim.Optimizer:
    return torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)


# --------------------------------------------------------------------------------------
# B. Learning-rate schedulers (built against a real optimizer)
# --------------------------------------------------------------------------------------


def test_warmup_instantiates_and_steps() -> None:
    optimizer = _optimizer()
    scheduler = Warmup(optimizer, warmup_steps=4)

    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1 * 1 / 5)
    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1 * 2 / 5)


def test_polylr_instantiates_and_steps() -> None:
    optimizer = _optimizer()
    scheduler = PolyLRScheduler(optimizer, initial_lr=0.1, max_steps=100)

    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1 * (1 - 1 / 100) ** 0.9)


@pytest.mark.parametrize(
    ("name", "kwargs"),
    [
        ("StepLR", {"step_size": 20, "gamma": 0.5}),
        ("CosineAnnealingLR", {"T_max": 10}),
        ("ReduceLROnPlateau", {}),
    ],
)
def test_documented_torch_lr_schedulers_instantiate_and_step(name: str, kwargs: dict) -> None:
    # The LR loader resolves against torch.optim.lr_scheduler first: the doc's torch examples
    # must construct and step against a real optimizer on the installed torch.
    optimizer = _optimizer()
    scheduler = getattr(torch.optim.lr_scheduler, name)(optimizer, **kwargs)
    optimizer.step()
    if name == "ReduceLROnPlateau":
        scheduler.step(0.5)
    else:
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] > 0.0


# --------------------------------------------------------------------------------------
# A. Criterion-weight schedulers (scalar, no optimizer)
# --------------------------------------------------------------------------------------


def test_constant_steps_and_reports_its_value() -> None:
    scheduler = Constant(value=2.0)
    scheduler.step(5)
    assert scheduler.get_value() == 2.0


def test_cosine_annealing_steps_and_reports_the_annealed_value() -> None:
    scheduler = CosineAnnealing(start_value=1.0, eta_min=0.0, t_max=100)
    assert scheduler.get_value() == pytest.approx(1.0)
    scheduler.step(50)
    assert scheduler.get_value() == pytest.approx(0.5)
