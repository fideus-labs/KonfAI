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

import weakref
from functools import partial, reduce
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from konfai.metric.measure.base import (
    Criterion,
    CriterionOutput,
    CriterionWithAttribute,
    LabelledValues,
    MaskedLoss,
    _require_optional,
)
from konfai.utils.errors import MeasureError

#: Per-label sums of one (output, reference) pair, what every Dice is computed from and the state a
#: streamed patch hands back: (intersection, predicted, reference, present in the reference).
LabelSums = dict[int, tuple[float, float, float, bool]]


class Dice(Criterion):
    maximize = True  # reported value is the Dice coefficient (higher-is-better); DiceSaveMap inherits it
    resamples_target = True  # a target of another shape is put on the output's grid (on_grid)

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
    def _bin_index(tensor: torch.Tensor) -> torch.Tensor:
        """``tensor`` in a dtype ``bincount`` takes: a bool map as uint8, a float map (NaN voxels
        to be masked by the caller) and the wide unsigned maps as int64. ``bincount`` has no kernel
        for uint16/uint32/uint64 on the CPU, and uint16 is what a label map is often stored as. A
        uint64 label past the int64 range has no bin: refused, not wrapped."""
        if tensor.dtype is torch.bool:
            return tensor.to(torch.uint8)
        if tensor.is_floating_point() or tensor.dtype in (torch.uint16, torch.uint32):
            return tensor.to(torch.int64)
        if tensor.dtype is torch.uint64:
            index = tensor.view(torch.int64)
            if bool((index < 0).any()):
                raise MeasureError(
                    "A uint64 label map holds a label above 2^63-1, which no bin can count.",
                    "Relabel the map (labels are small integers) or store it as uint32/int64.",
                )
            return index
        return tensor

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
            index = Dice._bin_index(tensor).flatten()
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
        maps (the reference, the prediction, the reference where the two agree): every label at once
        and one ``.tolist()`` for the lot.

        With ``labels=None`` every label either map holds gets its sums, so a patch's predicted mass
        for a label its reference lacks still reaches the whole-case ratio in ``combine_metric``;
        only the labels the reference holds are scored.
        """
        # A float map is binned by truncation: a probability would read as background.
        if output.is_floating_point() and bool(torch.frac(output.detach()).nan_to_num_().any()):
            raise MeasureError(
                "Dice reads a one-channel output as a label map, and this one holds non-integer values.",
                "Give a probability map one channel per label (background and foreground), "
                "or threshold it to labels before Dice scores it.",
            )
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
        """The sums of a probability map against a label map, per channel: the intersection is taken
        only where the reference holds the label (it is zero elsewhere), and one ``.tolist()`` brings
        every sum back.

        A label at a time, where the loss batches them: this route carries no gradient, so its peak is
        one channel and never the label count (346 MiB and 1.95 ms against 1050 MiB and 4.44 ms
        batched, on a ``[1, 41, 128, 128, 128]`` patch of 40 labels).

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
    ) -> tuple[torch.Tensor, dict[int, float] | LabelledValues]:
        """The differentiable soft Dice over a probability map's channels, one per label the reference
        holds, every label in one expression: their channels gathered in one slice, the reference one
        comparison against the label vector on an axis of its own, and each sum one reduction. That
        axis sits before the target's channels, so a reference of any channel count broadcasts against
        a gathered channel exactly as a per-label ``target == label`` did.

        One gathered slice, because the gradient of a slice writes into the whole logits tensor, once
        per slice. Its peak grows with the label count, so without a gradient to build ``_soft_sums``
        is the route.

        The reference's own mass is the voxel count ``_reference_counts`` already holds. The per-label
        dices leave as a ``LabelledValues`` read off the device lazily (``Measure._materialize``),
        never a ``.tolist()`` in the forward.
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
        values = dices.new_full((len(labels),), float("nan"))
        values[[index for index, count in enumerate(reference) if count]] = dices.detach()
        return 1 - dices.mean(), LabelledValues(values, list(labels))

    @staticmethod
    def _loss(
        labels: list[int] | None, output: torch.Tensor, *targets: torch.Tensor
    ) -> tuple[torch.Tensor, dict[int, float] | LabelledValues]:
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
        self._probabilities_checked = False

    def _check_probabilities(self, output: torch.Tensor) -> None:
        """Refuse, on the first call, a multi-channel output outside [0, 1]: Dice reads it as
        probabilities, and logits (a Conv head with no Softmax) would score outside [0, 1] without a
        word. One synchronisation, then none."""
        if self._probabilities_checked or output.shape[1] < 2 or not output.is_floating_point():
            return
        low, high = torch.stack(torch.aminmax(output.detach())).tolist()
        if not -1e-3 <= low <= high <= 1 + 1e-3:  # NaN fails every comparison
            raise MeasureError(
                f"Dice reads a {output.shape[1]}-channel output as probabilities, and this one spans"
                f" [{low:.3g}, {high:.3g}].",
                "Attach Dice to the output of a Softmax (or Sigmoid), not to the logits before it.",
            )
        self._probabilities_checked = True

    @staticmethod
    def _masked(output: torch.Tensor, targets: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, torch.Tensor]:
        """The pair with every voxel outside the mask sent to the background, when a mask is given.
        The reference and the masks are put on the output's grid first (``on_grid``), or a mask on
        either grid of a pair on two could not multiply both. The bool mask multiplies in each
        tensor's own dtype: an int64 ``torch.where`` was 8 B/voxel for the same product."""
        targets = tuple(Dice.on_grid(output, target) for target in targets)
        mask = MaskedLoss.get_mask(list(targets[1:]))
        if mask is None:
            return output, targets[0]
        return Dice._times(output, mask), Dice._times(targets[0], mask)

    @staticmethod
    def _times(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """``tensor * mask``. torch promotes a bool with every dtype but uint16, uint32 and uint64, which take the
        mask in their own dtype (a label map written as uint16)."""
        unpromoted = tensor.dtype in (torch.uint16, torch.uint32, torch.uint64)
        return tensor * (mask.to(tensor.dtype) if unpromoted else mask)

    def forward(self, output: torch.Tensor, *targets: torch.Tensor) -> CriterionOutput:
        self._check_probabilities(output)
        return self.loss(*self._masked(output, targets))

    def partial_metric(self, output: torch.Tensor, *targets: torch.Tensor) -> Any:
        self._check_probabilities(output)
        if any(tuple(target.shape[2:]) != tuple(output.shape[2:]) for target in targets):
            raise MeasureError(
                "Dice can only stream patches when output and target share the spatial grid",
                f"output {tuple(output.shape[2:])} vs targets {[tuple(target.shape[2:]) for target in targets]}",
                "Evaluate this pair whole (unset the memory budget) or resample the prediction first.",
            )
        output, target = self._masked(output, targets)
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
        value, result = self._score(sums, labels)
        return torch.tensor(value), result


class MissingLabels(Criterion):
    """How many labels of the reference the output lacks: the structures a segmentation did not find, or those a
    registration's warp lost. ``labels`` restricts the count to those labels, where the reference holds them;
    without it every label of the reference but 0 counts. Extra targets are masks, as for ``Dice``. An output of one
    channel per label, as ``Dice`` reads it, counts its argmax. Metric only."""

    loss_capable = False
    batch_mean = True  # the mean of one count per item
    resamples_target = True  # a reference of another shape is put on the output's grid (Dice.on_grid)

    def __init__(self, labels: list[int] | None = None) -> None:
        super().__init__()
        self._labels = labels

    def forward(self, output: torch.Tensor, *targets: torch.Tensor) -> CriterionOutput:
        if output.shape[1] > 1:
            output = output.argmax(1, keepdim=True)
        # only a float map is rounded: torch has no round for a bool one
        output, target = (map_.round() if map_.is_floating_point() else map_ for map_ in Dice._masked(output, targets))
        missing = []
        for item in range(output.shape[0]):
            reference = torch.unique(target[item])
            held = set(reference[reference.isfinite()].long().tolist()) - {0}  # NaN is no label, as for Dice
            written = set(torch.unique(output[item]).long().tolist())
            missing.append(len((held if self._labels is None else held & set(self._labels)) - written))
        value = float(np.mean(missing))
        return torch.tensor(value, dtype=torch.float32, device=output.device), value


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

    def forward(self, output: torch.Tensor, *targets: torch.Tensor) -> CriterionOutput:
        self._check_probabilities(output)
        output, target = self._masked(output, targets)
        loss, true_loss = self.loss(output, target)
        return loss, true_loss, self._map(output, target)

    def get_name(self) -> str:
        return "Dice"


class _LabelRatio(Dice):
    """A per-label score of a label map against the reference's from the sums Dice counts (intersection, predicted
    and reference voxels), every label the reference holds but the background, NaN for one it lacks; the reported
    value is their mean. The sums of disjoint patches add up, so it streams as Dice does."""

    loss_capable = False  # a ratio of counts, no gradient

    @staticmethod
    def _ratio(intersection: float, predicted: float, reference: float) -> float:
        raise NotImplementedError

    @classmethod
    def _score(cls, sums: LabelSums, labels: list[int]) -> tuple[float, dict[int, float]]:
        result = {
            label: (cls._ratio(*sums[label][:3]) if label in sums and sums[label][3] else np.nan) for label in labels
        }
        scored = [value for value in result.values() if not np.isnan(value)]
        mean = float(np.mean(scored)) if scored else 0.0
        return (1 - mean if cls.maximize else mean), result

    def forward(self, output: torch.Tensor, *targets: torch.Tensor) -> CriterionOutput:
        return self.combine_metric([self.partial_metric(output, *(Dice.on_grid(output, target) for target in targets))])


class Jaccard(_LabelRatio):
    """The Jaccard index (intersection over union) of each label: the overlap Dice measures, without its weight on
    the intersection."""

    @staticmethod
    def _ratio(intersection: float, predicted: float, reference: float) -> float:
        return (intersection + 1e-6) / (predicted + reference - intersection + 1e-6)


class VolumeDifference(_LabelRatio):
    """The relative volume difference of each label, ``|V_output - V_reference| / V_reference``: 0 when the two hold
    as many voxels, whatever their place."""

    maximize = False

    @staticmethod
    def _ratio(intersection: float, predicted: float, reference: float) -> float:
        del intersection
        return abs(predicted - reference) / reference


#: The last pair's surface distances, so HD95, ASSD and SurfaceDice of one case share one computation.
_LAST_SURFACES: tuple[Any, Any, tuple, Any] | None = None


def _surface_distances(
    output: torch.Tensor, target: torch.Tensor, spacing: list[float], labels: list[int] | None
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray] | None], float]:
    """Per label the reference holds: the distances (in the spacing's unit) from each surface voxel of the output's
    label to the reference's surface, and back; ``None`` for a label the output lacks. And the diagonal of the
    grid, the distance a lacking label is scored at.

    A surface is the label's contour voxels (face-connected), and a distance the one from a voxel centre to the
    nearest contour voxel's (SimpleITK's Maurer map, in physical spacing). Each label is cropped to the bounding
    box of the two maps' label and padded with one voxel of background: the grid's border is a surface too (as
    MedPy and DeepMind's surface-distance count it), and every contour lies inside the crop, so the distances are
    exact.
    """
    global _LAST_SURFACES
    key = (tuple(labels) if labels is not None else None, tuple(spacing))
    last = _LAST_SURFACES
    if last is not None and last[0]() is output and last[1]() is target and last[2] == key:
        return last[3]
    sitk = _require_optional("SimpleITK", criterion="HD95, ASSD and SurfaceDice", extra="itk")
    predicted = output[0, 0].detach().cpu().numpy()
    reference = target[0, 0].detach().cpu().numpy()
    if labels is None:
        labels = [int(label) for label in np.unique(reference) if label != 0]
    xyz_spacing = [float(value) for value in spacing][: predicted.ndim]

    def contour(mask: np.ndarray) -> tuple[Any, np.ndarray]:
        image = sitk.GetImageFromArray(mask.astype(np.uint8))
        image.SetSpacing(xyz_spacing)
        edge = sitk.BinaryContour(image, fullyConnected=False, backgroundValue=0, foregroundValue=1)
        return edge, sitk.GetArrayFromImage(edge).astype(bool)

    def distance_to(edge: Any) -> np.ndarray:
        distances = sitk.SignedMaurerDistanceMap(
            edge, insideIsPositive=False, squaredDistance=False, useImageSpacing=True
        )
        return np.abs(sitk.GetArrayFromImage(distances))

    distances: dict[int, tuple[np.ndarray, np.ndarray] | None] = {}
    for label in labels:
        a, b = predicted == label, reference == label
        if not b.any():
            continue
        if not a.any():
            distances[label] = None
            continue
        union = np.argwhere(a | b)
        box = tuple(slice(int(low), int(high) + 1) for low, high in zip(union.min(0), union.max(0), strict=True))
        (edge_a, voxels_a), (edge_b, voxels_b) = contour(np.pad(a[box], 1)), contour(np.pad(b[box], 1))
        distances[label] = (distance_to(edge_b)[voxels_a], distance_to(edge_a)[voxels_b])
    diagonal = float(np.linalg.norm(np.array(predicted.shape[::-1]) * np.array(xyz_spacing)))
    _LAST_SURFACES = (weakref.ref(output), weakref.ref(target), key, (distances, diagonal))
    return distances, diagonal


class _SurfaceDistance(CriterionWithAttribute):
    """A per-label score of a label map against the reference's from their surfaces, in the reference's physical
    spacing (both maps on one grid): every label the reference holds but the background, NaN for one it lacks;
    the reported value is their mean. Read whole: a surface distance has no partial state."""

    loss_capable = False  # distances read off SimpleITK's maps, no gradient

    def __init__(self, labels: list[int] | None = None) -> None:
        super().__init__()
        self.labels = labels

    def _of(self, distances: tuple[np.ndarray, np.ndarray] | None, diagonal: float) -> float:
        raise NotImplementedError

    def forward(  # type: ignore[override]
        self, output: torch.Tensor, *targets: torch.Tensor, attributes: list[list[Any]]
    ) -> CriterionOutput:
        if output.shape[0] != 1 or output.shape[1] != 1:
            raise MeasureError(
                f"{self.get_name()} reads one label map per case; got {list(output.shape[:2])} (batch, channels).",
                "Evaluate one case at a time, on a label map (Argmax a probability map first).",
            )
        spacing = attributes[0][0].get_np_array("Spacing").tolist()
        per_label, diagonal = _surface_distances(output, Dice.on_grid(output, targets[0]), spacing, self.labels)
        result = {label: self._of(distances, diagonal) for label, distances in per_label.items()}
        scored = list(result.values())
        mean = float(np.mean(scored)) if scored else 0.0
        return torch.tensor((1 - mean) if self.maximize else mean), result


class HD95(_SurfaceDistance):
    """The 95th percentile Hausdorff distance of each label, in mm: the larger of the two directed 95th percentiles
    (output surface to reference surface, and back). A label the output lacks scores the grid's diagonal."""

    def _of(self, distances: tuple[np.ndarray, np.ndarray] | None, diagonal: float) -> float:
        if distances is None:
            return diagonal
        return float(max(np.percentile(distances[0], 95), np.percentile(distances[1], 95)))


class ASSD(_SurfaceDistance):
    """The average symmetric surface distance of each label, in mm: the mean of every distance from one surface to
    the other, both ways. A label the output lacks scores the grid's diagonal."""

    def _of(self, distances: tuple[np.ndarray, np.ndarray] | None, diagonal: float) -> float:
        if distances is None:
            return diagonal
        return float((distances[0].sum() + distances[1].sum()) / (distances[0].size + distances[1].size))


class SurfaceDice(_SurfaceDistance):
    """The surface Dice (normalised surface distance) of each label at ``tolerance`` mm: the share of both surfaces
    within ``tolerance`` of the other, which leaves an annotation's own uncertainty unscored. A label the output
    lacks scores 0."""

    maximize = True

    def __init__(self, tolerance: float = 1.0, labels: list[int] | None = None) -> None:
        super().__init__(labels)
        self.tolerance = float(tolerance)

    def _of(self, distances: tuple[np.ndarray, np.ndarray] | None, diagonal: float) -> float:
        del diagonal
        if distances is None:
            return 0.0
        within = (distances[0] <= self.tolerance).sum() + (distances[1] <= self.tolerance).sum()
        return float(within / (distances[0].size + distances[1].size))
