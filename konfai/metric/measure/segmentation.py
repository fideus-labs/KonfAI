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


"""Overlap criteria over label maps."""

from functools import partial, reduce
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from konfai.metric.measure.base import Criterion, MaskedLoss
from konfai.utils.errors import MeasureError

#: Per-label sums of one (output, reference) pair, what every Dice is computed from and the state a
#: streamed patch hands back: (intersection, predicted, reference, present in the reference).
LabelSums = dict[int, tuple[float, float, float, bool]]


class Dice(Criterion):
    maximize = True  # reported value is the Dice coefficient (higher-is-better); DiceSaveMap inherits it

    @staticmethod
    def on_grid(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """A label map on the output's spatial grid.

        Nearest picks values rather than blending them, so every label comes back exactly; on the
        output's own grid it is the identity, and skipping it keeps the map's integer dtype (the
        resample runs in float: nearest-neighbour interpolation has no integer kernel).
        """
        if tuple(target.shape[2:]) == tuple(output.shape[2:]):
            return target
        return F.interpolate(target.float(), output.shape[2:], mode="nearest")

    @staticmethod
    def _may_hold_a_negative(tensor: torch.Tensor) -> bool:
        """Whether a map's dtype can hold a label below zero: an unsigned map is read as it is."""
        return tensor.dtype not in (torch.bool, torch.uint8, torch.uint16, torch.uint32, torch.uint64)

    @staticmethod
    def _bins(maps: list[torch.Tensor], labels: list[int] | None) -> tuple[list[torch.Tensor], int, int, int]:
        """The maps as bin indices, the offset a bin carries (label = bin + offset), the NaN bin (1
        when there is one, at index 0, else 0) and the bins to count.

        ``bincount`` takes no negative index and sizes its result by the largest one, so a signed
        map is shifted by its smallest label when that is below zero (one sync for the lot), and
        a float map's NaN voxels, no label at all, take bin 0 with every label moved up one. An
        unsigned integer map is read as it is.
        """
        nan_masks = [torch.isnan(tensor).flatten() if tensor.is_floating_point() else None for tensor in maps]
        indices = []
        for tensor, nan_mask in zip(maps, nan_masks, strict=True):
            dtype = torch.uint8 if tensor.dtype is torch.bool else torch.int64 if nan_mask is not None else tensor.dtype
            index = tensor.to(dtype).flatten()
            if nan_mask is not None:
                index = index.masked_fill_(nan_mask, 0)  # a whole number, so the minimum below is a label's
            indices.append(index)
        signed = [index for index, tensor in zip(indices, maps, strict=True) if Dice._may_hold_a_negative(tensor)]
        # One reduction per map and nothing more: stacking the minima to reduce them again adds a
        # kernel and a device copy, 0.0967 -> 0.1040 ms for the counts of a [8, 1, 256, 256] int64
        # map (CUDA events).
        lowest = int(reduce(torch.minimum, (index.min() for index in signed))) if signed else 0
        nan_bin = int(any(nan_mask is not None for nan_mask in nan_masks))
        offset = min(0, lowest, *(labels or [0])) - nan_bin
        if offset:
            indices = [index - offset for index in indices]
        for index, nan_mask in zip(indices, nan_masks, strict=True):
            if nan_mask is not None:
                index.masked_fill_(nan_mask, 0)
        return indices, offset, nan_bin, (max(labels) - offset + 1 if labels else 0)

    @staticmethod
    def _labels_held(offset: int, nan_bin: int, counts: list[list[int]]) -> list[int]:
        """Every label some map holds, ascending, but the background and the NaN bin."""
        return [
            bin_ + offset
            for bin_ in range(nan_bin, max(len(count) for count in counts))
            if bin_ + offset != 0 and any(bin_ < len(count) and count[bin_] for count in counts)
        ]

    @staticmethod
    def _reference_counts(target: torch.Tensor, labels: list[int] | None) -> tuple[list[int], list[int]]:
        """The labels to score and each one's voxel count in the reference: one ``bincount``, one sync.

        ``labels=None`` scores every label the reference holds except the background, ascending as
        ``torch.unique`` listed them.
        """
        (reference,), offset, nan_bin, minlength = Dice._bins([target], labels)
        counts = torch.bincount(reference, minlength=minlength).tolist()
        if labels is None:
            labels = Dice._labels_held(offset, nan_bin, [counts])
        return labels, [counts[label - offset] if 0 <= label - offset < len(counts) else 0 for label in labels]

    @staticmethod
    def _hard_sums(output: torch.Tensor, target: torch.Tensor, labels: list[int] | None) -> LabelSums:
        """The sums of a hard-label pair, exact integers from three ``bincount`` passes over the flat
        maps (the reference, the prediction, the reference where the two agree): every label at
        once and one ``.tolist()`` for the lot, where a per-label ``==`` built two float volumes and
        synchronised twice.

        With ``labels=None`` every label either map holds gets its sums, so a patch's predicted mass
        for a label its reference lacks still reaches the whole-case ratio in ``combine_metric``;
        only the labels the reference holds are scored.
        """
        (predicted_labels, reference_labels), offset, nan_bin, minlength = Dice._bins([output, target], labels)
        counts = [
            torch.bincount(reference_labels[reference_labels == predicted_labels], minlength=minlength),
            torch.bincount(predicted_labels, minlength=minlength),
            torch.bincount(reference_labels, minlength=minlength),
        ]
        length = max(count.numel() for count in counts)
        intersection, predicted, reference = torch.stack(
            [F.pad(count, (0, length - count.numel())) for count in counts]
        ).tolist()
        if labels is None:
            labels = Dice._labels_held(offset, nan_bin, [predicted, reference])
        return {
            label: (
                (
                    intersection[label - offset],
                    predicted[label - offset],
                    reference[label - offset],
                    reference[label - offset] > 0,
                )
                if 0 <= label - offset < length
                else (0, 0, 0, False)
            )
            for label in labels
        }

    @staticmethod
    def _soft_sums(output: torch.Tensor, target: torch.Tensor, labels: list[int] | None) -> LabelSums:
        """The sums of a probability map against a label map, per channel: the intersection is
        taken only where the reference holds the label (it is zero elsewhere), and one
        ``.tolist()`` brings every sum back.

        A label at a time, where the loss batches them: this route carries no gradient, so its peak
        is one channel and never the label count. Batched it read 1050 MiB instead of 346 and took
        4.44 ms instead of 1.95 on a ``[1, 41, 128, 128, 128]`` patch of 40 labels.

        With ``labels=None`` every label either map holds gets its sums, so a patch's predicted mass
        for a label its reference lacks still reaches the whole-case ratio in ``combine_metric``.
        """
        scored = labels if labels is not None else list(range(1, output.shape[1]))
        _, reference = Dice._reference_counts(target, scored)
        sums: list[torch.Tensor] = []
        for label, count in zip(scored, reference, strict=True):
            if label >= output.shape[1]:  # no channel: nothing predicted
                sums.extend([output.new_zeros(()), output.new_zeros(())])
                continue
            predicted = output[:, label].unsqueeze(1).float()
            sums.append(predicted.sum())
            # A bool mask multiplies in the probability's own dtype: no float copy of the reference.
            sums.append((predicted * (target == label)).sum() if count else predicted.new_zeros(()))
        values = torch.stack(sums).tolist() if sums else []
        return {
            label: (values[2 * index + 1], values[2 * index], float(count), count > 0)
            for index, (label, count) in enumerate(zip(scored, reference, strict=True))
        }

    @staticmethod
    def _score(sums: LabelSums, labels: list[int]) -> tuple[float, dict[int, float]]:
        """Every label's Dice from its sums, the smooth term applied once, NaN for a label the
        reference lacks; and the loss ``1 - mean`` over the scored ones (0 when none is)."""
        result: dict[int, float] = {}
        total, count = 0.0, 0
        for label in labels:
            intersection, predicted, reference, present = sums.get(label, (0, 0, 0, False))
            if present:
                dice = (2.0 * intersection + 1e-6) / (predicted + reference + 1e-6)
                result[label] = dice
                total += dice
                count += 1
            else:
                result[label] = np.nan
        return (1 - total / count if count else 0.0), result

    @staticmethod
    def _soft_loss(
        labels: list[int] | None, output: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, dict[int, float]]:
        """The differentiable soft Dice over a probability map's channels, one per label the
        reference holds, every label in one expression: their channels gathered in one slice, the
        reference one comparison against the label vector on an axis of its own, and each sum one
        reduction. That axis sits before the target's channels, so a reference of any channel count
        broadcasts against a gathered channel exactly as a per-label ``target == label`` did.

        A slice per label cost a gradient write into the whole logits tensor each: 22.7 ``add_`` and
        28.7 ``fill_`` per step, 10.2 ms of the 28.2 ms of GPU time of a training step of
        ``examples/Segmentation`` (batch 8, 41 channels, 256x256, autocast + channels_last).
        Backward held a float copy of both operands per label already, so training peaks lower
        (242 -> 180 MiB on that batch); with no gradient to build it is the label count that peaks
        where the loop peaked at one channel (6 -> 180 MiB), and ``_soft_sums`` is the frugal route.

        The reference's own mass is the voxel count ``_reference_counts`` already holds, the exact
        integer the metric route's ``_score`` divides by. The readouts come back in one
        ``.tolist()``: a ``.item()`` per label drained the CUDA queue mid-loss, twice per label.
        """
        labels, reference = Dice._reference_counts(target, labels)
        held = [label for label, count in zip(labels, reference, strict=True) if count]
        held_counts = [count for count in reference if count]
        if not held:
            return torch.tensor(0, dtype=torch.float32).to(output.device), dict.fromkeys(labels, np.nan)
        channels = output.shape[1]
        if min(held) < -channels or max(held) >= channels:
            raise MeasureError(
                f"Dice was asked to score labels {held} of a probability map of {channels} channels",
                "the reference holds a label the prediction has no channel for",
                "Give the criterion the `labels:` it should score, or predict one channel per label.",
            )
        label_vector = torch.tensor(held, device=output.device)
        probabilities = output[:, label_vector].float().unsqueeze(2)
        on_reference = target.unsqueeze(1) == label_vector.view(1, -1, *(1,) * (target.dim() - 1))
        summed = (0, 2, *range(3, probabilities.dim()))
        intersection = (probabilities * on_reference).sum(summed)
        reference_mass = torch.tensor(held_counts, dtype=torch.float32, device=output.device)
        dices = (2.0 * intersection + 1e-6) / (probabilities.sum(summed) + reference_mass + 1e-6)
        values = iter(dices.tolist())
        result = {label: next(values) if count else np.nan for label, count in zip(labels, reference, strict=True)}
        return 1 - dices.mean(), result

    @staticmethod
    def _loss(
        labels: list[int] | None, output: torch.Tensor, *targets: torch.Tensor
    ) -> tuple[torch.Tensor, dict[int, float]]:
        target = Dice.on_grid(output, targets[0])
        if output.shape[1] > 1:
            return Dice._soft_loss(labels, output, target)
        sums = Dice._hard_sums(output, target, labels)
        value, result = Dice._score(sums, labels if labels is not None else [label for label in sums if sums[label][3]])
        return torch.tensor(value, dtype=torch.float32, device=output.device), result

    reducible = True

    def __init__(self, labels: list[int] | None = None) -> None:
        super().__init__()
        self._labels = labels
        self.loss = partial(Dice._loss, labels)

    @staticmethod
    def _masked(output: torch.Tensor, targets: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, torch.Tensor]:
        """The pair with every voxel outside the mask sent to the background, when a mask is given.
        A bool mask multiplies in each tensor's own dtype: ``torch.where(mask == 1, 1, 0)`` was
        8 B/voxel of int64 for the same product."""
        mask = MaskedLoss.get_mask(list(targets[1:]))
        if mask is None:
            return output, targets[0]
        mask = mask == 1
        return output * mask, targets[0] * mask

    def forward(self, output: torch.Tensor, *targets: torch.Tensor) -> tuple[torch.Tensor, dict[int, float]]:
        return self.loss(*self._masked(output, targets))

    def partial_metric(self, output: torch.Tensor, *targets: torch.Tensor) -> Any:
        output, target = self._masked(output, targets)
        if tuple(target.shape[2:]) != tuple(output.shape[2:]):
            raise MeasureError(
                "Dice can only stream patches when output and target share the spatial grid",
                f"output {tuple(output.shape[2:])} vs target {tuple(target.shape[2:])}",
                "Evaluate this pair whole (unset the memory budget) or resample the prediction first.",
            )
        if output.shape[1] > 1:
            return Dice._soft_sums(output, target, self._labels)
        return Dice._hard_sums(output, target, self._labels)

    def combine_metric(self, states: list[Any]) -> Any:
        # The whole volume's sums are its patches' sums added, and every dice is the ratio of the
        # GLOBAL sums, the smooth term applied once here, never a mean of per-patch dices. With
        # ``labels: None`` the scored labels are those some patch's reference holds, ascending as
        # torch.unique lists them.
        sums: LabelSums = {}
        for state in states:
            for label, (intersection, predicted, reference, present) in state.items():
                total = sums.get(label, (0, 0, 0, False))
                sums[label] = (
                    total[0] + intersection,
                    total[1] + predicted,
                    total[2] + reference,
                    total[3] or present,
                )
        labels = self._labels if self._labels is not None else sorted(label for label in sums if sums[label][3])
        value, result = Dice._score(sums, labels)
        return torch.tensor(value), result


class DiceSaveMap(Dice):
    def __init__(self, labels: list[int] | None = None, dataset: str | None = None, group: str | None = None) -> None:
        super().__init__(labels)
        self.dataset = dataset
        self.group = group

    @staticmethod
    def _map(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Per-voxel label disagreement ``|output - target|`` as uint8, from ``max - min``: it needs
        no signed intermediate, where a uint8 ``output - target`` wraps (|0 - 3| came out 253)."""
        return (torch.maximum(output, target) - torch.minimum(output, target)).to(torch.uint8).cpu()

    def partial_map(self, output: torch.Tensor, *targets: torch.Tensor) -> torch.Tensor:
        """Per-voxel label disagreement (masked where a mask is given). VOXEL-LOCAL by construction:
        a patch's map equals the same region of the whole-case map, which is what lets the streamed
        evaluation write it region by region instead of needing the whole case."""
        return self._map(*self._masked(output, targets))

    def forward(self, output: torch.Tensor, *targets: torch.Tensor):  # type: ignore[override]
        output, target = self._masked(output, targets)
        loss, true_loss = self.loss(output, target)
        return loss, true_loss, self._map(output, target)

    def get_name(self) -> str:
        return "Dice"
