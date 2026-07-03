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

"""Regression test: dict-payload metrics (Dice, TRE) must not crash the logging windows."""

import numpy as np
import pytest
import torch
from konfai.network.network import Measure


def test_loss_add_summarises_dict_metric_payload() -> None:
    # Dice/TRE return (tensor, {label: value}); the pre-fix code stored the dict in _values, so the
    # np.nanmean over _values in get_last_values/format_loss raised TypeError on every batch.
    record = Measure.Loss("Dice", "out", "tgt", 0, is_loss=False, accumulation=False)

    record.add(1.0, (torch.tensor([0.7]), {"1": 0.6, "2": 0.8, "3": float("nan")}))

    # The dict is summarised to a scalar (nan-mean of 0.6 and 0.8), and the logging mean is safe.
    assert isinstance(record._values[-1], float)
    assert record._values[-1] == pytest.approx(0.7)
    assert np.nanmean(record._values) == pytest.approx(0.7)


def test_loss_add_keeps_plain_scalar_metric() -> None:
    # A regular (tensor, float) metric is unchanged.
    record = Measure.Loss("MSE", "out", "tgt", 0, is_loss=False, accumulation=False)
    record.add(1.0, (torch.tensor([0.5]), 0.5))
    assert record._values[-1] == pytest.approx(0.5)
