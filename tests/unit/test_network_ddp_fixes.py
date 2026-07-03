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

"""Regression tests for the network / DDP / scheduler batch (see AUDIT.md §Training)."""

import os

os.environ.setdefault("KONFAI_config_file", "/tmp/konfai-none.yml")
os.environ.setdefault("KONFAI_CONFIG_MODE", "Done")

from unittest.mock import MagicMock  # noqa: E402

import torch  # noqa: E402

from konfai.metric.schedulers import Constant, PolyLRScheduler  # noqa: E402
from konfai.network.network import Measure  # noqa: E402
from konfai.utils.dataset import Attribute  # noqa: E402


class _Attr:
    def __init__(self) -> None:
        self.start = 0
        self.stop = None
        self.schedulers = {Constant(1.0): 1}
        self.group = 0
        self.is_loss = True
        self.accumulation = True


def _make_accumulating_measure(scaler) -> tuple[Measure, torch.Tensor]:
    """Build a minimal Measure that triggers the accumulation-backward branch."""
    measure = Measure.__new__(Measure)
    criterion = torch.nn.MSELoss()
    key = f"out:tgt:{criterion.__class__.__name__}"
    measure.outputs_criterions = {"out": {"tgt": {criterion: _Attr()}}}
    measure._loss = {0: {key: Measure.Loss(criterion.__class__.__name__, "out", "tgt", 0, True, True)}}
    measure.scaler = scaler
    output = torch.zeros(1, 1, 2, 2, requires_grad=True)
    return measure, output


def test_accumulation_backward_uses_scaler_scale():
    """#AMP: accumulation losses must be scaled before backward when a GradScaler is set."""
    scaler = MagicMock()
    scaled = MagicMock()
    scaler.scale.return_value = scaled

    measure, output = _make_accumulating_measure(scaler)
    target = torch.ones(1, 1, 2, 2)
    measure.update("out", output, {"tgt": (target, [Attribute()])}, it=0, nb_patch=1, training=True)

    # The loss must go through scaler.scale(...).backward(), never a bare loss.backward().
    scaler.scale.assert_called_once()
    scaled.backward.assert_called_once()
    # Bare backward would have populated grads directly; the scaler intercepts it.
    assert output.grad is None


def test_accumulation_backward_without_scaler_is_plain_backward():
    """Without a scaler the accumulation path must still back-propagate normally."""
    measure, output = _make_accumulating_measure(None)
    target = torch.ones(1, 1, 2, 2)
    measure.update("out", output, {"tgt": (target, [Attribute()])}, it=0, nb_patch=1, training=True)

    assert output.grad is not None
    assert torch.count_nonzero(output.grad) > 0


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


def test_cyclegan_discriminator_initialized_no_keyerror():
    """#CycleGan: initialized() must not index a missing 'Sample' submodule on load."""
    from konfai.models.generation.diffusionGan import CycleGanDiscriminator

    model = CycleGanDiscriminator()
    # Must not raise KeyError('Sample').
    model.initialized()
