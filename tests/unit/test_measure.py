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
from konfai.metric.measure import SSIM, Dice, FocalLoss, PerceptualLoss, Variance, _require_optional
from konfai.utils.errors import MeasureError


def _one_hot(target: torch.Tensor, nb_channels: int) -> torch.Tensor:
    output = torch.zeros(1, nb_channels, *target.shape[2:])
    for label in range(nb_channels):
        output[0, label] = (target[0, 0] == label).float()
    return output


class TestFocalLoss:
    def test_does_not_cross_pair_samples_for_batch_greater_than_one(self):
        # The alpha weighting must stay per-voxel: the per-element loss shape must match the gathered
        # log-prob shape [B, 1, *spatial], NOT broadcast to a [B, B, *spatial] cross-product between
        # samples. Regression guard for the spurious unsqueeze that corrupted any batch > 1.
        import torch.nn.functional as F

        torch.manual_seed(0)
        batch, num_classes, height, width = 2, 3, 4, 4
        output = torch.randn(batch, num_classes, height, width)
        target = torch.randint(0, num_classes, (batch, 1, height, width)).float()

        focal = FocalLoss(alpha=[0.5, 2.0, 0.5], reduction="none")
        loss = focal(output, target)
        assert tuple(loss.shape[:2]) == (batch, 1)  # not (batch, batch)

        # Value equals the correct per-voxel reference.
        tgt = target.long()
        log_pt = F.log_softmax(output, dim=1).gather(1, tgt)
        pt = torch.exp(F.log_softmax(output, dim=1)).gather(1, tgt)
        at = focal.alpha[tgt]
        reference = -at * ((1 - pt) ** focal.gamma) * log_pt
        assert torch.allclose(loss, reference)


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


class TestImpactRegPCA:
    """The IMPACT registration loss can reduce its deep features to their top-``pca`` principal
    components (itk-impact parity). The basis is fitted on the TARGET features and reused for the
    output, so both feature maps live in the same reduced space before the per-layer distance."""

    @staticmethod
    def _core(pca: int):
        from konfai.metric.measure import IMPACTReg

        core = IMPACTReg.__new__(IMPACTReg)
        torch.nn.Module.__init__(core)
        core.pca = pca
        return core

    def test_transform_reduces_channels(self):
        core = self._core(3)
        basis = torch.linalg.qr(torch.randn(8, 3))[0]  # orthonormal [8, 3]
        out = core._pca_transform(torch.randn(2, 8, 4, 5, 6), basis)
        assert out.shape == (2, 3, 4, 5, 6)

    def test_transform_centres_by_own_channel_mean(self):
        core = self._core(2)
        torch.manual_seed(0)
        basis = torch.linalg.qr(torch.randn(6, 2))[0]
        # Distinct per-channel constants: each channel is spatially flat, so per-CHANNEL mean-centring zeros
        # it -> projects to 0. A global/cross-channel mean would leave the per-channel offsets and project to
        # a non-zero value (~0.79 here), so this input discriminates the correct centring from that bug.
        const = torch.arange(1.0, 7.0).reshape(1, 6, 1, 1, 1).expand(1, 6, 3, 3, 3).contiguous()
        out = core._pca_transform(const, basis)
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-5)

    def test_project_reduces_both_maps_to_top_k(self):
        core = self._core(2)
        moved, fixed = torch.randn(1, 6, 3, 3, 3), torch.randn(1, 6, 3, 3, 3)
        mp, fp = core._pca_project(moved, fixed)
        assert mp.shape == (1, 2, 3, 3, 3) and fp.shape == (1, 2, 3, 3, 3)
        assert torch.isfinite(mp).all() and torch.isfinite(fp).all()

    def test_project_clamps_k_to_channel_count(self):
        core = self._core(99)  # more components than channels
        moved, fixed = torch.randn(1, 4, 2, 2, 2), torch.randn(1, 4, 2, 2, 2)
        mp, fp = core._pca_project(moved, fixed)
        assert mp.shape[1] == 4 and fp.shape[1] == 4  # k clamped to C

    def test_project_first_component_recovers_dominant_channel(self):
        """Correctness: the basis is the top eigenvector of the TARGET channel-covariance, so with a
        single dominant channel the reduced component is (up to sign) that channel's centred signal."""
        core = self._core(1)
        fixed = torch.zeros(1, 4, 4, 4, 4)
        fixed[0, 0] = torch.randn(4, 4, 4) * 10.0  # channel 0 carries almost all the variance
        fixed[0, 1:] = torch.randn(3, 4, 4, 4) * 0.01
        _mp, fp = core._pca_project(fixed.clone(), fixed)
        proj = fp[0, 0].flatten()
        centred_ch0 = (fixed[0, 0] - fixed[0, 0].mean()).flatten()
        corr = torch.corrcoef(torch.stack([proj, centred_ch0]))[0, 1].abs()
        assert corr > 0.99


def test_accuracy_reports_per_batch_not_a_lifetime_running_fraction() -> None:
    # Accuracy used to accumulate n/corrects on the instance forever, so it reported one fraction that
    # blended every epoch and both splits. It must now report the current batch (the logging window means
    # and resets it), so an all-correct batch is 1.0 and a following all-wrong batch is 0.0 -- not 0.5.
    from konfai.metric.measure import Accuracy

    accuracy = Accuracy()
    logits = torch.tensor([[9.0, 0.0, 0.0], [0.0, 9.0, 0.0]])  # argmax -> [0, 1]

    all_correct = accuracy(logits, torch.tensor([0, 1]))
    all_wrong = accuracy(logits, torch.tensor([1, 0]))

    assert all_correct.item() == pytest.approx(1.0)
    assert all_wrong.item() == pytest.approx(0.0)  # not blended with the previous batch


def test_fid_preprocess_images_runs() -> None:
    # FID.preprocess_images called torch.nn.functional.resize / .normalize(mean, std), neither of which
    # exists there -- the metric could not execute. It now uses torchvision.transforms.functional.
    pytest.importorskip("torchvision")
    from konfai.metric.measure import FID

    out = FID.preprocess_images(torch.zeros(2, 1, 64, 64))

    assert out.shape == (2, 3, 299, 299)


def test_lpips_preprocessing_follows_input_device() -> None:
    # LPIPS.preprocessing hardcoded .to(0), crashing on a CPU-only host and pinning every DDP rank to
    # GPU 0. It now keeps the input's device (the model is moved to it lazily in _loss).
    from konfai.metric.measure import LPIPS

    out = LPIPS.preprocessing(torch.zeros(1, 1, 8, 8))

    assert out.device == torch.device("cpu")
    assert out.shape == (1, 3, 8, 8)


def test_perceptual_loss_applies_every_loss_to_the_target() -> None:
    # The inner loop zipped the losses against the targets, so the default {Gram, L1Loss} on a single
    # reference silently used only Gram. Every configured loss must reach the (single) target layer.
    from unittest.mock import MagicMock

    loss = object.__new__(PerceptualLoss)
    loss.preprocessing = lambda tensor: tensor  # type: ignore[method-assign]

    model = MagicMock()
    model.get_layers.return_value = [("L", torch.zeros(1, 1, 2, 2))]
    loss.models = {None: model}

    applied: list[str] = []

    def make_loss(tag: str):
        def loss_fn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            applied.append(tag)
            return torch.zeros(1)

        return loss_fn

    loss.modules_loss = {"L": {make_loss("gram"): 1.0, make_loss("l1"): 1.0}}

    loss._compute(torch.zeros(1, 1, 2, 2), torch.zeros(1, 1, 2, 2))

    assert set(applied) == {"gram", "l1"}  # both, not just the first
