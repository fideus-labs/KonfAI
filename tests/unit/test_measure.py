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
import torch.nn.functional as F
from konfai.metric.measure import SSIM, Dice, FocalLoss, KLDivergence, PerceptualLoss, Variance, _require_optional
from konfai.network.network import CriterionsAttr
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
        # samples (a spurious unsqueeze corrupts any batch > 1).
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


@pytest.mark.parametrize("criterion", ["Dice", "FocalLoss"])
def test_a_criterion_accepts_the_integer_label_map_a_segmentation_target_is(criterion: str) -> None:
    """A segmentation target comes off disk as integer labels, not as floats.

    Both criteria resample the target onto the output grid, and nearest-neighbour interpolation has
    no integer kernel, so the dtype the dataset actually produces reached torch as an unsupported
    one. Every other test here hands a float, which is why the training path was the first to meet it.
    """
    output = torch.rand(1, 3, 8, 8)
    target = torch.randint(0, 3, (1, 1, 8, 8))
    assert target.dtype is torch.int64

    loss = Dice()(output, target)[0] if criterion == "Dice" else FocalLoss()(output, target)

    assert torch.isfinite(loss).all()


class TestOnGrid:
    def test_a_target_on_the_output_grid_is_handed_back_as_is(self):
        # No resample on the output's own grid: the integer label map keeps its dtype and its storage,
        # where the unconditional float resample cost 3.7 s of copy plus float `unique` at 512^3.
        output = torch.zeros(1, 3, 6, 5)
        target = torch.randint(0, 3, (1, 1, 6, 5)).to(torch.uint8)

        assert Dice.on_grid(output, target) is target

    def test_a_target_on_another_grid_is_resampled_nearest(self):
        output = torch.zeros(1, 3, 6, 4)
        target = torch.tensor([[[[1, 2], [3, 4]]]], dtype=torch.uint8)

        resampled = Dice.on_grid(output, target)

        assert tuple(resampled.shape) == (1, 1, 6, 4)
        assert set(resampled.unique().tolist()) == {1, 2, 3, 4}  # picked, never blended
        assert torch.equal(resampled, F.interpolate(target.float(), (6, 4), mode="nearest"))

    def test_focal_loss_takes_the_integer_target_on_its_own_grid(self):
        output = torch.randn(1, 3, 4, 4)
        target = torch.randint(0, 3, (1, 1, 4, 4)).to(torch.uint8)

        loss = FocalLoss(alpha=[0.5, 2.0, 0.5])(output, target)

        assert torch.isfinite(loss)
        assert loss.item() == pytest.approx(FocalLoss(alpha=[0.5, 2.0, 0.5])(output, target.float()).item())


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


def _dice_per_label_oracle(labels, output, target, mask=None):
    """The per-label spelling Dice was scored with before the batched one: one float expansion and
    two sums per label, kept here as the oracle both routes are pinned against."""
    if mask is not None:
        mask = torch.where(mask == 1, 1, 0)
        output, target = output * mask.to(output.dtype), target * mask.to(target.dtype)
    result = {}
    loss = torch.tensor(0, dtype=torch.float32)
    if labels is None:
        labels = [int(label) for label in torch.unique(target) if int(label) != 0]
    count = 0
    for label in labels:
        tp = (target == label).float()
        if tp.any().item():
            pp = (output[:, label].unsqueeze(1) if output.shape[1] > 1 else (output == label)).float()
            dice = (2.0 * (pp * tp).sum() + 1e-6) / (pp.sum() + tp.sum() + 1e-6)
            loss += dice
            count += 1
            result[label] = dice.item()
        else:
            result[label] = np.nan
    return (1 - loss / count if count else loss), result


def _assert_same_scores(got, expected, abs_tol):
    assert set(got[1]) == set(expected[1])
    for label, value in expected[1].items():
        if np.isnan(value):
            assert np.isnan(got[1][label])
        else:
            assert got[1][label] == pytest.approx(value, abs=abs_tol)
    assert got[0].item() == pytest.approx(expected[0].item(), abs=abs_tol)


class TestDiceConfusionMatrix:
    """Hard labels are scored from one confusion matrix; the per-label oracle above pins the values."""

    @staticmethod
    def _pair(structured: bool, dtype: torch.dtype):
        rng = np.random.default_rng(7)
        shape = (1, 1, 12, 13, 11)
        if structured:
            grid = np.stack(np.meshgrid(*[np.linspace(-1, 1, s) for s in shape[2:]], indexing="ij"))
            target = np.clip((np.sqrt((grid**2).sum(0)) * 6).astype(np.int64), 0, 5)[None, None]
            flips = rng.random(shape) < 0.15
            output = np.where(flips, rng.integers(0, 6, size=shape), target)
        else:
            target = rng.integers(0, 6, size=shape)
            output = rng.integers(0, 6, size=shape)
        # Label 4 never predicted, label 5 never in the reference, label 9 in neither.
        output[output == 4] = 0
        target[target == 5] = 0
        return torch.tensor(output).to(dtype), torch.tensor(target).to(dtype)

    @pytest.mark.parametrize("labels", [None, [1, 2, 3], [-1, 2]])
    def test_a_negative_label_and_a_nan_voxel_score_as_the_oracle_scores_them(self, labels):
        """``bincount`` takes no negative index and no NaN: a label of -1 (an ignore convention) is
        a label like any other, a NaN voxel is no label, as the per-label ``==`` always read them."""
        output, target = self._pair(True, torch.int16)
        output[output == 2] = -1
        target[target == 3] = -1
        _assert_same_scores(Dice(labels=labels)(output, target), _dice_per_label_oracle(labels, output, target), 1e-6)

        output, target = self._pair(True, torch.float32)
        output[output == 1] = float("nan")
        target[target == 2] = float("nan")
        # The oracle's own label discovery is torch.unique, which hands it the NaN: told the labels.
        held = labels or sorted(int(v) for v in torch.unique(target[~target.isnan()]).tolist() if v != 0)
        _assert_same_scores(Dice(labels=labels)(output, target), _dice_per_label_oracle(held, output, target), 1e-6)

    @pytest.mark.parametrize("structured", [False, True])
    @pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
    @pytest.mark.parametrize("labels", [None, [1, 2, 3, 4, 5, 9]])
    @pytest.mark.parametrize("masked", [False, True])
    def test_matches_the_per_label_oracle(self, structured, dtype, labels, masked):
        output, target = self._pair(structured, dtype)
        mask = (torch.rand(output.shape) > 0.3).to(torch.uint8) if masked else None
        args = (output, target, mask) if masked else (output, target)

        got = Dice(labels=labels)(*args)
        expected = _dice_per_label_oracle(labels, output, target, mask)

        # Exact integer counts against float32 sums: the 1e-6 smooth term is the only rounding left.
        _assert_same_scores(got, expected, abs_tol=1e-6)
        if labels is not None:
            assert np.isnan(got[1][9]) and np.isnan(got[1][5])  # absent from the reference
            reference = int(((target == 4) & (mask == 1)).sum()) if masked else int((target == 4).sum())
            assert got[1][4] == pytest.approx(1e-6 / (reference + 1e-6), abs=1e-9)  # never predicted

    @pytest.mark.parametrize("seed", range(4))
    def test_soft_path_matches_the_oracle_to_the_last_float32_bits(self, seed):
        """One reduction per label over the whole batch instead of a sum accumulated label by
        label: the reduction order moves, the values do not beyond float32 rounding (measured
        max |diff| 6e-8 on the loss over these seeds, 7.5e-9 per label)."""
        torch.manual_seed(seed)
        output = torch.softmax(torch.randn(2, 5, 6, 7, 5), dim=1)
        target = torch.randint(0, 5, (2, 1, 6, 7, 5))
        target[target == 3] = 0

        for labels in (None, [1, 2, 3, 4]):
            got = Dice(labels=labels)(output, target)
            expected = _dice_per_label_oracle(labels, output, target)
            _assert_same_scores(got, expected, abs_tol=1e-7)

    def test_soft_path_keeps_the_gradient_the_per_label_oracle_gives(self):
        torch.manual_seed(11)
        probabilities = torch.softmax(torch.randn(2, 6, 5, 5), dim=1)
        target = torch.randint(0, 6, (2, 1, 5, 5))

        gradients = []
        for score in (lambda o, t: Dice()(o, t), lambda o, t: _dice_per_label_oracle(None, o, t)):
            output = probabilities.clone().requires_grad_(True)
            gradients.append(torch.autograd.grad(score(output, target)[0], output)[0])

        assert torch.isfinite(gradients[0]).all()
        assert (gradients[0] - gradients[1]).abs().max() < 1e-7 * gradients[1].abs().max()

    def test_soft_loss_refuses_a_reference_label_the_probability_map_has_no_channel(self):
        """The channels are gathered in one slice: an out-of-range label would be a device-side
        assert on CUDA, so it is refused on the spot."""
        output = torch.softmax(torch.randn(1, 3, 4, 4), dim=1)
        target = torch.full((1, 1, 4, 4), 7)

        with pytest.raises(MeasureError, match="no channel for"):
            Dice()(output, target)

    def test_soft_loss_of_a_reference_holding_no_label_scores_nothing(self):
        output = torch.softmax(torch.randn(1, 3, 4, 4), dim=1)
        target = torch.zeros(1, 1, 4, 4, dtype=torch.int64)

        loss, per_label = Dice()(output, target)
        assert loss.item() == 0.0 and per_label == {}
        assert np.isnan(Dice(labels=[1, 2])(output, target)[1][2])

    def test_streamed_sums_carry_the_predicted_mass_of_a_label_the_patch_reference_lacks(self):
        # labels=None: a patch predicting label 2 where its reference has none must still count
        # that mass in the whole-case ratio; a state built from the patch's own label set dropped it.
        target = torch.zeros(1, 1, 4, 4, dtype=torch.uint8)
        target[..., :2, :] = 2  # label 2 in the first half only
        output = torch.full((1, 1, 4, 4), 2, dtype=torch.uint8)  # predicted everywhere
        metric = Dice(labels=None)

        whole = metric(output, target)
        states = [metric.partial_metric(output[..., :2, :], target[..., :2, :])]
        states.append(metric.partial_metric(output[..., 2:, :], target[..., 2:, :]))
        combined = metric.combine_metric(states)

        assert whole[1][2] == pytest.approx(2 * 8 / (16 + 8), abs=1e-6)
        assert combined[1][2] == pytest.approx(whole[1][2], abs=1e-9)
        assert set(combined[1]) == {2}  # a label no reference holds is not reported

    @pytest.mark.parametrize("soft", [False, True])
    def test_combine_reproduces_the_oracle_from_the_same_patches(self, soft):
        torch.manual_seed(5)
        if soft:
            output = torch.softmax(torch.randn(1, 6, 8, 9, 7), dim=1)
        else:
            output = torch.randint(0, 6, (1, 1, 8, 9, 7)).to(torch.uint8)
        target = torch.randint(0, 6, (1, 1, 8, 9, 7)).to(torch.uint8)
        target[target == 5] = 0
        for labels in (None, [1, 2, 5, 9]):
            metric = Dice(labels=labels)
            states = [
                metric.partial_metric(output[..., z : z + 3, :, :], target[..., z : z + 3, :, :]) for z in (0, 3, 6)
            ]
            _assert_same_scores(metric.combine_metric(states), _dice_per_label_oracle(labels, output, target), 1e-6)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="counts CUDA synchronisations")
    def test_syncs_do_not_grow_with_the_label_count(self):
        import warnings

        output = torch.softmax(torch.randn(1, 41, 8, 8, 8, device="cuda"), dim=1)
        target = torch.randint(0, 41, (1, 1, 8, 8, 8), device="cuda").to(torch.uint8)
        hard = torch.randint(0, 41, (1, 1, 8, 8, 8), device="cuda").to(torch.uint8)
        counts = {}
        torch.cuda.set_sync_debug_mode("warn")
        try:
            for name, args in (("soft", (output, target)), ("hard", (hard, target))):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    Dice()(*args)
                    Dice().partial_metric(*args)
                counts[name] = len(caught)
        finally:
            torch.cuda.set_sync_debug_mode("default")
        # forward + partial_metric: bincount (one sync each) and one .tolist() each, never one per label.
        assert counts["soft"] <= 8 and counts["hard"] <= 12, counts


class TestSaveMaps:
    """A SaveMap metric reads its scalar and its map off one difference buffer; the two-pass
    spellings they replace are the oracles."""

    @staticmethod
    def _mae_oracle(reduction, output, target, mask=None):
        from konfai.metric.measure import MAE

        args = (output, target) if mask is None else (output, target, mask)
        loss, value = MAE(reduction)(*args)
        if mask is None:
            map_ = torch.nn.L1Loss(reduction="none")(output.float(), target.float())
        else:
            mask64 = torch.where(mask == 1, 1, 0)
            map_ = torch.nn.L1Loss(reduction="none")(output.float() * mask64, target.float() * mask64)
        return loss, value, map_.to(output.dtype).cpu()

    @pytest.mark.parametrize("reduction", ["mean", "sum"])
    @pytest.mark.parametrize("masked", [False, True])
    @pytest.mark.parametrize("batch", [1, 2])
    def test_mae_scalar_and_map_are_bit_identical_to_the_two_pass_spelling(self, reduction, masked, batch):
        from konfai.metric.measure import MAESaveMap

        torch.manual_seed(11)
        output = torch.rand(batch, 2, 7, 6, 5) * 4000 - 1000
        target = output + torch.randn(batch, 2, 7, 6, 5) * 100
        mask = (torch.rand(batch, 1, 7, 6, 5) > 0.4).to(torch.uint8) if masked else None
        args = (output, target, mask) if masked else (output, target)

        loss, value, map_ = MAESaveMap(reduction)(*args)
        expected_loss, expected_value, expected_map = self._mae_oracle(reduction, output, target, mask)

        assert value == expected_value
        assert loss.item() == expected_loss.item()
        assert torch.equal(map_, expected_map)
        assert torch.equal(MAESaveMap(reduction).partial_map(*args), expected_map)

    def test_mae_with_an_empty_mask_is_nan_and_its_map_zero(self):
        from konfai.metric.measure import MAESaveMap

        output, target = torch.rand(1, 1, 4, 4), torch.rand(1, 1, 4, 4)
        mask = torch.zeros(1, 1, 4, 4, dtype=torch.uint8)

        _, value, map_ = MAESaveMap()(output, target, mask)

        assert np.isnan(value)
        assert torch.equal(map_, torch.zeros(1, 1, 4, 4))

    def test_dice_map_of_uint8_labels_does_not_wrap(self):
        # |0 - 3| on uint8 labels is 3; the unmasked `output - target` wrapped to 253, and only the
        # masked path's accidental int64 promotion got it right.
        from konfai.metric.measure import DiceSaveMap

        output = torch.tensor([[[[0, 3, 2]]]], dtype=torch.uint8)
        target = torch.tensor([[[[3, 0, 2]]]], dtype=torch.uint8)

        _, _, map_ = DiceSaveMap()(output, target)

        assert map_.dtype == torch.uint8
        assert map_.tolist() == [[[[3, 3, 0]]]]

    @pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
    def test_dice_masked_map_is_bit_identical_to_the_promoted_spelling(self, dtype):
        from konfai.metric.measure import DiceSaveMap

        torch.manual_seed(2)
        output = torch.randint(0, 6, (1, 1, 6, 7, 5)).to(dtype)
        target = torch.randint(0, 6, (1, 1, 6, 7, 5)).to(dtype)
        mask = (torch.rand(1, 1, 6, 7, 5) > 0.3).to(torch.uint8)
        mask64 = torch.where(mask == 1, 1, 0)
        expected = torch.nn.L1Loss(reduction="none")(output * mask64, target * mask64).to(torch.uint8)

        _, _, map_ = DiceSaveMap()(output, target, mask)

        assert torch.equal(map_, expected)
        assert torch.equal(DiceSaveMap().partial_map(output, target, mask), expected)


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
        x, _ = self._volumes()

        _, value = SSIM(dynamic_range=self.dynamic_range)(x, x.clone())

        assert value == pytest.approx(1.0, abs=1e-6)

    # . The torch port against skimage, its oracle ----------------------------------------------
    # skimage filters in float32 and means in float64; the port does the same, so the two agree to
    # a few 1e-8 relative (measured max 4e-8 on 64x72x80 and 256^3 inputs): pinned at 1e-6 absolute.

    @staticmethod
    def _pair(kind: str, shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(0)
        grid = np.stack(np.meshgrid(*[np.linspace(-1, 1, s) for s in shape], indexing="ij"))
        if kind == "random":
            x = rng.normal(size=shape).astype(np.float32) * 500
            y = x + rng.normal(size=shape).astype(np.float32) * 100
        elif kind == "smooth":
            x = (np.sin(4 * grid[0]) * np.cos(3 * grid[1]) * 1000).astype(np.float32)
            y = (x * 0.9 + 50 * np.cos(5 * grid[-1])).astype(np.float32)
        elif kind == "masked":  # zeros outside a ball, what a body mask leaves
            inside = (grid**2).sum(0) < 0.7
            x = (rng.normal(size=shape).astype(np.float32) * 500) * inside
            y = (x + rng.normal(size=shape).astype(np.float32) * 100) * inside
        elif (
            kind == "offset"
        ):  # a constant 1000 against the same with a 5 HU step: the variance is nothing beside the mean
            x = np.full(shape, 1000.0, dtype=np.float32)
            y = x.copy()
            y[tuple(slice(s // 2, None) for s in shape)] += 5.0
        else:  # a CT-like field: a large mean, a little noise
            x = (2500.0 + np.sin(4 * grid[0]) * 30 + rng.normal(size=shape) * 50).astype(np.float32)
            y = (x + rng.normal(size=shape) * 20).astype(np.float32)
        return x, y

    @pytest.mark.parametrize("kind", ["random", "smooth", "masked", "offset", "hu"])
    @pytest.mark.parametrize("shape", [(24, 27, 21), (40, 33)])
    def test_matches_skimage(self, kind, shape):
        structural_similarity = pytest.importorskip("skimage.metrics").structural_similarity
        x, y = self._pair(kind, shape)
        expected = float(structural_similarity(x.astype(np.float64), y.astype(np.float64), data_range=4095.0))

        _, value = SSIM(dynamic_range=4095.0)(torch.tensor(x)[None, None], torch.tensor(y)[None, None])

        assert value == pytest.approx(expected, abs=1e-6)

    def test_channels_are_averaged_as_channel_axis_zero(self):
        structural_similarity = pytest.importorskip("skimage.metrics").structural_similarity
        x, y = self._pair("random", (3, 20, 22, 19))
        expected = float(
            structural_similarity(x.astype(np.float64), y.astype(np.float64), data_range=4095.0, channel_axis=0)
        )

        _, value = SSIM(dynamic_range=4095.0)(torch.tensor(x)[None], torch.tensor(y)[None])

        assert value == pytest.approx(expected, abs=1e-6)

    def test_a_mask_multiplies_the_pair_and_scores_the_whole_extent(self):
        # The masked mode: both volumes are zeroed outside the mask and the mean still runs over the
        # whole cropped extent, exactly what skimage sees when handed the masked volumes.
        structural_similarity = pytest.importorskip("skimage.metrics").structural_similarity
        x, y = self._pair("random", (18, 21, 17))
        mask = (np.random.default_rng(1).random(x.shape) > 0.4).astype(np.uint8)
        expected = float(structural_similarity(x * mask, y * mask, data_range=4095.0))

        _, value = SSIM(dynamic_range=4095.0)(
            torch.tensor(x)[None, None], torch.tensor(y)[None, None], torch.tensor(mask)[None, None]
        )

        assert value == pytest.approx(expected, abs=1e-6)

    def test_batch_items_are_averaged_and_an_empty_mask_item_skipped(self):
        x1, y1 = self._pair("random", (12, 14, 13))
        x2, y2 = self._pair("smooth", (12, 14, 13))
        x = torch.tensor(np.stack([x1, x2]))[:, None]
        y = torch.tensor(np.stack([y1, y2]))[:, None]
        metric = SSIM(dynamic_range=4095.0)
        first = metric(x[:1], y[:1])[1]
        second = metric(x[1:], y[1:])[1]

        _, both = metric(x, y)
        mask = torch.ones(2, 1, 12, 14, 13, dtype=torch.uint8)
        mask[1] = 0
        _, first_only = metric(x, y, mask)
        _, none = metric(x, y, torch.zeros_like(mask))

        assert both == pytest.approx((first + second) / 2, abs=1e-12)
        assert first_only == pytest.approx(first, abs=1e-12)
        assert np.isnan(none)

    def test_slabs_reproduce_the_one_shot_map(self):
        # The slab cut is a memory bound, not a value: every slab size gives the same map sum.
        x, y = self._pair("random", (30, 16, 15))
        xt, yt = torch.tensor(x)[None], torch.tensor(y)[None]
        whole = SSIM._ssim(xt, yt, None, 4095.0)
        original = SSIM.slab_bytes
        try:
            for slab_bytes in (1, 16 * 15 * 4 * 9, 16 * 15 * 4 * 13):
                SSIM.slab_bytes = slab_bytes
                assert SSIM._ssim(xt, yt, None, 4095.0) == pytest.approx(whole, rel=1e-7)
        finally:
            SSIM.slab_bytes = original

    def test_an_extent_below_the_window_is_refused(self):
        with pytest.raises(MeasureError, match="7-voxel window"):
            SSIM(dynamic_range=1.0)(torch.rand(1, 1, 6, 9, 9), torch.rand(1, 1, 6, 9, 9))


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
    # positional tensor; passing the whole tuple as a single argument hands the
    # preprocessing/feature-extraction path a tuple and crashes.
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
    # Accuracy must report the current batch (the logging window means and resets it): accumulating
    # n/corrects on the instance forever blends every epoch and both splits into one fraction. An
    # all-correct batch is 1.0 and a following all-wrong batch is 0.0, not 0.5.
    from konfai.metric.measure import Accuracy

    accuracy = Accuracy()
    logits = torch.tensor([[9.0, 0.0, 0.0], [0.0, 9.0, 0.0]])  # argmax -> [0, 1]

    all_correct = accuracy(logits, torch.tensor([0, 1]))
    all_wrong = accuracy(logits, torch.tensor([1, 0]))

    assert all_correct.item() == pytest.approx(1.0)
    assert all_wrong.item() == pytest.approx(0.0)  # not blended with the previous batch


def test_fid_preprocess_images_runs() -> None:
    # FID.preprocess_images must use torchvision.transforms.functional: torch.nn.functional has no
    # resize / normalize(mean, std), so calling them there means the metric cannot execute.
    pytest.importorskip("torchvision")
    from konfai.metric.measure import FID

    out = FID.preprocess_images(torch.zeros(2, 1, 64, 64))

    assert out.shape == (2, 3, 299, 299)


def test_lpips_preprocessing_follows_input_device() -> None:
    # LPIPS.preprocessing must keep the input's device (the model is moved to it lazily in _loss):
    # a hardcoded .to(0) crashes a CPU-only host and pins every DDP rank to GPU 0.
    from konfai.metric.measure import LPIPS

    out = LPIPS.preprocessing(torch.zeros(1, 1, 8, 8))

    assert out.device == torch.device("cpu")
    assert out.shape == (1, 3, 8, 8)


def test_perceptual_loss_applies_every_loss_to_the_target() -> None:
    # Every configured loss must reach the (single) target layer: zipping the losses against the
    # targets lets the default {Gram, L1Loss} on a single reference silently use only Gram.
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


class _FakeFeatureModel(torch.nn.Module):
    """model(x, nb_layer, stats) -> [x, pooled(x), ...] like an IMPACT TorchScript extractor."""

    def forward(self, x: torch.Tensor, nb_layer: torch.Tensor, stats: torch.Tensor = None) -> list[torch.Tensor]:
        feats = [x.contiguous()]
        while len(feats) < int(nb_layer):
            feats.append(torch.nn.functional.avg_pool2d(feats[-1], 2))
        return feats


class TestMaskedFeatureLoss:
    """The single implementation behind IMPACTReg / IMPACTSynth / SAM_Perceptual."""

    @staticmethod
    def _run(weights, mask=None, patch_shape=None, project=None, x=None, y=None):
        from konfai.metric.measure import _masked_feature_loss

        torch.manual_seed(0)
        x = torch.rand(1, 1, 32, 32) if x is None else x
        y = torch.rand(1, 1, 32, 32) if y is None else y
        triple = lambda t: [t, torch.tensor([len(weights)]), torch.tensor([[0.0, 0.5, 1.0, 0.2]])]  # noqa: E731
        return _masked_feature_loss(
            _FakeFeatureModel(), triple(x), triple(y), weights, torch.nn.L1Loss(), mask, patch_shape, project=project
        )

    def test_zero_weight_skips_the_layer(self):
        loss_one, _ = self._run([0.0, 1.0])
        loss_both, _ = self._run([1.0, 1.0])
        assert loss_one.item() < loss_both.item()

    def test_unmasked_equals_weighted_layer_sum(self):
        torch.manual_seed(0)
        x, y = torch.rand(1, 1, 32, 32), torch.rand(1, 1, 32, 32)
        loss, true_nb = self._run([0.5, 2.0], x=x, y=y)
        expected = 0.5 * torch.nn.functional.l1_loss(x, y) + 2.0 * torch.nn.functional.l1_loss(
            torch.nn.functional.avg_pool2d(x, 2), torch.nn.functional.avg_pool2d(y, 2)
        )
        assert true_nb == 1
        assert torch.allclose(loss, expected.reshape(1), atol=1e-6)

    def test_mask_restricts_the_loss_support(self):
        torch.manual_seed(0)
        x = torch.rand(1, 1, 32, 32)
        y = x.clone()
        y[:, :, 16:, 16:] += 10.0  # large error outside the mask only
        mask = torch.zeros(1, 1, 32, 32, dtype=torch.uint8)
        mask[:, :, :8, :8] = 1
        loss, _ = self._run([1.0], mask=mask, x=x, y=y)
        assert loss.item() < 1e-6

    def test_patches_without_mask_are_not_counted(self):
        mask = torch.zeros(1, 1, 32, 32, dtype=torch.uint8)
        mask[:, :, :16, :16] = 1  # only the first 16x16 tile
        _, true_nb = self._run([1.0], mask=mask, patch_shape=[16, 16])
        assert true_nb == 1  # 4 tiles, 3 skipped

    def test_project_hook_is_applied(self):
        halve = lambda a, b: (a / 2, b / 2)  # noqa: E731
        loss_raw, _ = self._run([1.0])
        loss_projected, _ = self._run([1.0], project=halve)
        assert torch.allclose(loss_projected, loss_raw / 2, atol=1e-6)

    def test_nan_layer_contributes_nothing(self):
        class NaNModel(torch.nn.Module):
            def forward(self, x, nb_layer, stats=None):
                return [x, torch.full_like(x, torch.nan)]

        from konfai.metric.measure import _masked_feature_loss

        x, y = torch.rand(1, 1, 8, 8), torch.rand(1, 1, 8, 8)
        triple = lambda t: [t, torch.tensor([2]), torch.tensor([[0.0, 0.5, 1.0, 0.2]])]  # noqa: E731
        loss, _ = _masked_feature_loss(NaNModel(), triple(x), triple(y), [1.0, 1.0], torch.nn.L1Loss(), None, None)
        assert torch.isfinite(loss).all()


# --------------------------------------------------------------------------- #
# accepts_init is read from the criterion, not its CriterionsAttr record.
# --------------------------------------------------------------------------- #
def test_accepts_init_flag_lives_on_the_criterion_not_the_attr() -> None:
    # Measure.init must read the capability flag from the criterion (the dict key), not from the
    # CriterionsAttr value (which never carries it): otherwise CriterionWithInit.init() is skipped
    # and graph-rewiring criteria such as KLDivergence train against the wrong channels silently.
    criterion = KLDivergence(shape=[16, 16])
    assert getattr(criterion, "accepts_init", False) is True
    assert getattr(CriterionsAttr(), "accepts_init", False) is False


def test_fid_builds_on_cpu_and_follows_the_input_device():
    # A hardcoded .cuda() at construction crashes CPU-only hosts; the model must be built on the
    # CPU and moved to the evaluated tensor's device in forward.
    pytest.importorskip("torchvision")
    pytest.importorskip("scipy")
    from konfai.metric.measure import FID

    metric = FID()
    assert next(metric.inception_model.parameters()).device.type == "cpu"


class TestSSIMFromHaloPatches:
    """SSIM is reducible from patches read with the window's radius of halo, each scoring the map
    voxels centred in its own grid slot: the streamed sum equals the whole-volume sum to float64
    summation order, including the slots at the volume's faces, where the halo is cut short and the
    whole-volume map is cropped the same."""

    @staticmethod
    def _grid(shape: tuple[int, ...], patch: list[int]) -> list[tuple[slice, ...]]:
        from konfai.utils.utils import get_patch_slices_from_shape

        return get_patch_slices_from_shape(patch, list(shape), 0)

    @staticmethod
    def _streamed(metric: SSIM, tensors: list[torch.Tensor], patch: list[int]) -> float:
        shape = tuple(tensors[0].shape[2:])
        states = []
        for slot in TestSSIMFromHaloPatches._grid(shape, patch):
            read = tuple(
                slice(max(0, s.start - metric.halo), min(extent, s.stop + metric.halo))
                for s, extent in zip(slot, shape, strict=True)
            )
            core = tuple(slice(s.start - r.start, s.stop - r.start) for s, r in zip(slot, read, strict=True))
            states.append(metric.partial_metric(*[t[(slice(None), slice(None), *read)] for t in tensors], core=core))
        return metric.combine_metric(states)[1]

    def test_declares_its_window_radius_as_halo(self):
        assert SSIM.reducible and SSIM.halo == 3

    @pytest.mark.parametrize("masked", [False, True])
    @pytest.mark.parametrize(
        ("shape", "patch"), [((37, 29), [16, 13]), ((19, 23, 17), [8, 8, 8]), ((19, 23, 17), [5, 23, 7])]
    )
    def test_halo_patches_reproduce_the_whole_volume(self, shape, patch, masked):
        # Every grid here leaves a partial slot on some axis; [5, 23, 7] also spans an axis whole.
        rng = np.random.default_rng(2)
        x = torch.tensor(rng.normal(size=(1, 2, *shape)).astype(np.float32) * 500)
        y = x + torch.tensor(rng.normal(size=x.shape).astype(np.float32) * 100)
        tensors = [x, y]
        if masked:
            tensors.append(torch.tensor((rng.random((1, 1, *shape)) > 0.4).astype(np.uint8)))
        metric = SSIM(dynamic_range=4095.0)
        whole = metric(*tensors)[1]

        assert self._streamed(metric, tensors, patch) == pytest.approx(whole, rel=1e-9)

    def test_a_slot_without_a_valid_centre_contributes_nothing(self):
        # A read too thin for the window (a 2-wide slot at a face, read 5 wide) has no map voxel of
        # its own: an empty state, never an error, the whole volume's map lies in the other slots.
        x, y = torch.rand(1, 1, 12, 12), torch.rand(1, 1, 12, 12)
        metric = SSIM(dynamic_range=1.0)
        thin = metric.partial_metric(x[..., :5, :], y[..., :5, :], core=(slice(0, 2), slice(0, 12)))
        rest = metric.partial_metric(x[..., 0:, :], y[..., 0:, :], core=(slice(2, 12), slice(0, 12)))

        assert thin == ("items", [(0.0, 0, True)])
        assert metric.combine_metric([thin, rest])[1] == pytest.approx(metric(x, y)[1], rel=1e-12)

    def test_an_empty_mask_item_is_skipped_as_the_whole_volume_skips_it(self):
        x, y = torch.rand(2, 1, 9, 9), torch.rand(2, 1, 9, 9)
        mask = torch.ones(2, 1, 9, 9, dtype=torch.uint8)
        mask[1] = 0
        metric = SSIM(dynamic_range=1.0)
        states = [metric.partial_metric(x[..., :7], y[..., :7], mask[..., :7], core=(slice(0, 9), slice(0, 4)))]
        states.append(metric.partial_metric(x[..., 1:], y[..., 1:], mask[..., 1:], core=(slice(0, 9), slice(3, 8))))

        assert metric.combine_metric(states)[1] == pytest.approx(metric(x, y, mask)[1], rel=1e-12)
        assert np.isnan(metric.combine_metric([metric.partial_metric(x, y, torch.zeros_like(mask))])[1])
