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

"""Tests for the criteria in ``konfai.metric.measure`` (Dice, SSIM, Variance,
PerceptualLoss plumbing, and optional-dependency errors)."""

import numpy as np
import pytest
import torch
from konfai.metric.measure import SSIM, Dice, PerceptualLoss, Variance, _require_optional
from konfai.utils.errors import MeasureError


def _one_hot(target: torch.Tensor, nb_channels: int) -> torch.Tensor:
    output = torch.zeros(1, nb_channels, *target.shape[2:])
    for label in range(nb_channels):
        output[0, label] = (target[0, 0] == label).float()
    return output


class TestDice:
    def test_loss_averages_over_present_labels_only(self):
        """A perfect prediction must give a loss of 0 even when some requested labels are absent."""
        target = torch.zeros(1, 1, 4, 4)
        target[0, 0, 0, :] = 1
        target[0, 0, 1, :] = 2
        output = _one_hot(target, 4)

        loss, per_label = Dice(labels=[1, 2, 3])(output, target)

        # Labels 1 and 2 are perfectly predicted (Dice = 1), label 3 is absent:
        # mean Dice = (1 + 1) / 2 = 1, hence loss = 1 - 1 = 0.
        assert loss.item() == pytest.approx(0.0, abs=1e-6)
        assert per_label[1] == pytest.approx(1.0, abs=1e-6)
        assert per_label[2] == pytest.approx(1.0, abs=1e-6)
        assert np.isnan(per_label[3])

    def test_loss_is_zero_when_no_requested_label_is_present(self):
        target = torch.zeros(1, 1, 4, 4)
        target[0, 0, 0, :] = 1
        output = _one_hot(target, 6)

        loss, per_label = Dice(labels=[5])(output, target)

        assert loss.item() == 0.0
        assert np.isnan(per_label[5])

    def test_default_labels_exclude_background(self):
        """With ``labels=None`` the per-case mean must not include the background (label 0)."""
        target = torch.zeros(1, 1, 4, 4)
        target[0, 0, 0, :] = 1  # 4 foreground voxels
        output = torch.zeros(1, 1, 4, 4, dtype=torch.uint8)
        output[0, 0, 0, :2] = 1  # 2 of them predicted

        loss, per_label = Dice(labels=None)(output, target)

        # Dice(label 1) = 2 * 2 / (2 + 4) = 2/3; the background Dice (24/26)
        # must not enter the average.
        assert set(per_label) == {1}
        assert per_label[1] == pytest.approx(2 / 3, abs=1e-5)
        assert loss.item() == pytest.approx(1 / 3, abs=1e-5)

    def test_default_labels_support_multichannel_output(self):
        target = torch.zeros(1, 1, 4, 4)
        target[0, 0, 0, :] = 1
        target[0, 0, 1, :] = 2
        output = _one_hot(target, 3)

        loss, per_label = Dice(labels=None)(output, target)

        assert set(per_label) == {1, 2}
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_mask_preserves_float_probabilities(self):
        """Masking must not quantize a probability map."""
        target = torch.zeros(1, 1, 2, 2)
        target[0, 0, 0, :] = 1
        output = torch.empty(1, 2, 2, 2)
        output[0, 1] = torch.tensor([[0.9, 0.9], [0.1, 0.1]])
        output[0, 0] = 1 - output[0, 1]
        mask = torch.ones(1, 1, 2, 2)

        loss, per_label = Dice(labels=[1])(output, target, mask)

        # Soft Dice(label 1) = 2 * (0.9 + 0.9) / ((0.9 + 0.9 + 0.1 + 0.1) + 2) = 0.9.
        assert per_label[1] == pytest.approx(0.9, abs=1e-5)
        assert loss.item() == pytest.approx(0.1, abs=1e-5)

    def test_mask_restricts_the_computation(self):
        target = torch.zeros(1, 1, 2, 2)
        target[0, 0, 0, :] = 1
        output = torch.zeros(1, 1, 2, 2, dtype=torch.uint8)
        output[0, 0, 0, 0] = 1  # correct
        output[0, 0, 1, 0] = 1  # wrong, but masked out
        mask = torch.zeros(1, 1, 2, 2)
        mask[0, 0, :, 0] = 1  # first column only

        _, per_label = Dice(labels=[1])(output, target, mask)

        # Inside the mask: prediction {(0,0),(1,0)}, target {(0,0)} ->
        # Dice = 2 * 1 / (2 + 1) = 2/3.
        assert per_label[1] == pytest.approx(2 / 3, abs=1e-5)


class TestSSIM:
    dynamic_range = 4.0

    @staticmethod
    def _volumes() -> tuple[torch.Tensor, torch.Tensor]:
        rng = np.random.default_rng(0)
        x = torch.tensor(rng.normal(size=(1, 1, 16, 16, 16)), dtype=torch.float32)
        y = x.clone()
        # Keep the first slice identical so a slice-0-only computation would return 1.0.
        y[0, 0, 1:] += 0.5 * torch.tensor(rng.normal(size=(15, 16, 16)), dtype=torch.float32)
        return x, y

    @staticmethod
    def _expected(x: torch.Tensor, y: torch.Tensor, dynamic_range: float) -> float:
        structural_similarity = pytest.importorskip("skimage.metrics").structural_similarity
        return float(structural_similarity(x[0, 0].numpy(), y[0, 0].numpy(), data_range=dynamic_range))

    def test_without_mask_returns_volume_ssim(self):
        pytest.importorskip("skimage.metrics")
        x, y = self._volumes()
        expected = self._expected(x, y, self.dynamic_range)

        loss, value = SSIM(dynamic_range=self.dynamic_range)(x, y)

        assert isinstance(loss, torch.Tensor)
        assert value == pytest.approx(expected, abs=1e-5)

    def test_with_mask_covers_the_whole_volume(self):
        pytest.importorskip("skimage.metrics")
        x, y = self._volumes()
        expected = self._expected(x, y, self.dynamic_range)
        assert expected < 0.99
        mask = torch.ones(1, 1, 16, 16, 16)

        _, value = SSIM(dynamic_range=self.dynamic_range)(x, y, mask)

        assert value == pytest.approx(expected, abs=1e-5)

    def test_identical_volumes_give_one(self):
        pytest.importorskip("skimage.metrics")
        x, _ = self._volumes()

        _, value = SSIM(dynamic_range=self.dynamic_range)(x, x.clone())

        assert value == pytest.approx(1.0, abs=1e-6)


class TestVariance:
    def test_single_channel_reports_zero(self):
        """A single sample along the reduced axis must give 0, not NaN."""
        output = torch.arange(16.0).reshape(1, 1, 4, 4)

        variance, value = Variance()(output)

        assert not torch.isnan(variance)
        assert variance.item() == pytest.approx(0.0)
        assert value == pytest.approx(0.0)

    def test_multi_channel_uses_unbiased_variance(self):
        """With several samples the unbiased (N-1) variance is averaged."""
        output = torch.tensor([1.0, 3.0]).reshape(1, 2, 1, 1)

        variance, value = Variance()(output)

        # Unbiased var of [1, 3] = ((1-2)^2 + (3-2)^2) / (2 - 1) = 2.0.
        assert variance.item() == pytest.approx(2.0)
        assert value == pytest.approx(2.0)


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


def test_missing_metric_dependency_raises_actionable_error():
    """Optional criterion deps must surface an actionable MeasureError, not ImportError."""
    with pytest.raises(MeasureError) as excinfo:
        _require_optional("konfai_definitely_missing_pkg_zzz", criterion="SSIM", extra="ssim")
    message = str(excinfo.value)
    assert "SSIM" in message
    assert "konfai[ssim]" in message
