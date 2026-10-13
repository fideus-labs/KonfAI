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


"""Surface criteria over label maps: the robust Hausdorff distance and the surface Dice (the normalised surface
distance), computed as google-deepmind/surface-distance computes them, the code Metrics Reloaded scores with. A
boundary voxel is a surface element whose area its 2x2x2 neighbourhood gives (``surface_tables``), the
distance from one surface to the other comes from a distance transform, and the percentile and the overlap weigh
every element by its area. Each label is scored on the box that holds it in both maps, the labels in parallel: the
distance transforms release the interpreter.
"""

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from scipy import ndimage

from konfai.metric.measure import surface_tables as tables
from konfai.metric.measure.base import CriterionOutput, CriterionWithAttribute, output_grid
from konfai.metric.measure.segmentation import Dice
from konfai.utils.dataset import Attribute
from konfai.utils.errors import DatasetManagerError, MeasureError

#: (distances from the reference surface to the output's, their areas, distances the other way, their areas), each
#: pair sorted by distance.
SurfaceDistances = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def surface_distances(reference: np.ndarray, output: np.ndarray, spacing: tuple[float, ...]) -> SurfaceDistances:
    """The distances between the surfaces of two boolean masks, over the box that holds both. Empty when both masks
    are empty; infinite distances from the surface that exists to the one that does not."""
    if len(spacing) == 3:
        areas = tables.create_table_neighbour_code_to_surface_area(spacing)
        kernel, full = tables.ENCODE_NEIGHBOURHOOD_3D_KERNEL, 0b11111111
    elif len(spacing) == 2:
        areas = tables.create_table_neighbour_code_to_contour_length(spacing)
        kernel, full = tables.ENCODE_NEIGHBOURHOOD_2D_KERNEL, 0b1111
    else:
        raise MeasureError(f"A surface distance is defined on a 2D or 3D mask, not on {len(spacing)} dimensions.")
    held = np.nonzero(reference | output)
    if len(held[0]) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    box = tuple(slice(int(axis.min()), int(axis.max()) + 1) for axis in held)

    def cropped(mask: np.ndarray) -> np.ndarray:
        # one zero voxel past the box on every axis, as the reference code pads: the 2x2x2 kernel then sees every face
        out = np.zeros([b.stop - b.start + 1 for b in box], np.uint8)
        out[(slice(0, -1),) * len(box)] = mask[box]
        return out

    codes = [ndimage.correlate(cropped(mask), kernel, mode="constant", cval=0) for mask in (reference, output)]
    borders = [(code != 0) & (code != full) for code in codes]
    to_surface = [
        ndimage.distance_transform_edt(~border, sampling=spacing) if border.any() else np.full(border.shape, np.inf)
        for border in borders
    ]

    def sorted_pair(side: int) -> tuple[np.ndarray, np.ndarray]:
        distances, area = to_surface[1 - side][borders[side]], areas[codes[side][borders[side]]]
        order = np.lexsort((area, distances))  # by distance, then by area, as the reference code sorts
        return distances[order], area[order]

    return (*sorted_pair(0), *sorted_pair(1))


def robust_hausdorff(distances: SurfaceDistances, percent: float) -> float:
    """The ``percent`` percentile of the distances, each surface element counting by its area, the larger of the two
    directions; infinite when one surface is missing."""

    def one_way(values: np.ndarray, area: np.ndarray) -> float:
        if len(values) == 0:
            return float("inf")
        index = int(np.searchsorted(np.cumsum(area) / np.sum(area), percent / 100.0))
        return float(values[min(index, len(values) - 1)])

    return max(one_way(distances[0], distances[1]), one_way(distances[2], distances[3]))


def surface_dice(distances: SurfaceDistances, tolerance: float) -> float:
    """The area of the surface elements within ``tolerance`` of the other surface, over the area of both surfaces."""
    near = np.sum(distances[1][distances[0] <= tolerance]) + np.sum(distances[3][distances[2] <= tolerance])
    total = np.sum(distances[1]) + np.sum(distances[3])
    return float(near / total) if total > 0 else float("nan")


class _SurfaceCriterion(CriterionWithAttribute):
    """One value per label from the surfaces of a label map and its reference, on each label's box; the reported
    value is the mean over the labels scored, NaN for a label the reference lacks. Extra targets are masks, as for
    ``Dice``. The spacing comes from the output's attributes."""

    accepts_output_attributes = True

    def __init__(self, labels: list[int] | None = None) -> None:
        super().__init__()
        self._labels = labels

    def _value(self, distances: SurfaceDistances) -> float:
        raise NotImplementedError()

    @staticmethod
    def _spacing(grid: list[Attribute], item: int, dimensions: int) -> tuple[float, ...]:
        """The voxel spacing in the array's axis order (``(x, y, z)`` in the attributes, ``[z, y, x]`` in memory)."""
        try:
            spacing = grid[item].get_np_array("Spacing")
        except (IndexError, KeyError, DatasetManagerError) as error:
            raise MeasureError(
                "A surface distance needs the voxel spacing, which the output's attributes do not carry.",
                "Read the label map from a format that keeps its geometry (mha, nii, omezarr).",
            ) from error
        if len(spacing) != dimensions:
            raise MeasureError(f"A spacing of {len(spacing)} values for a {dimensions}D label map.")
        return tuple(float(s) for s in spacing[::-1])

    def _per_label(self, output: np.ndarray, reference: np.ndarray, spacing: tuple[float, ...]) -> dict[int, float]:
        boxes = [
            {int(k) + 1: box for k, box in enumerate(ndimage.find_objects(m)) if box is not None}
            for m in (reference, output)
        ]
        labels = self._labels if self._labels is not None else sorted(boxes[0])

        def one(label: int) -> float:
            if label not in boxes[0]:
                return float("nan")  # the reference lacks it: not scored
            if label not in boxes[1]:
                return self._missing()
            box = tuple(
                slice(min(p.start, q.start), max(p.stop, q.stop))
                for p, q in zip(boxes[0][label], boxes[1][label], strict=True)
            )
            return self._value(surface_distances(reference[box] == label, output[box] == label, spacing))

        with ThreadPoolExecutor(min(8, os.cpu_count() or 1)) as pool:
            return dict(zip(labels, pool.map(one, labels), strict=True))

    def _missing(self) -> float:
        """The value of a label the reference holds and the output lacks."""
        raise NotImplementedError()

    def forward(  # type: ignore[override]  # the added keywords are this subclass's contract
        self,
        output: torch.Tensor,
        *targets: torch.Tensor,
        attributes: list[list[Attribute]],
        output_attributes: list[Attribute] | None = None,
    ) -> CriterionOutput:
        if output.shape[1] != 1:
            raise MeasureError(
                f"{self.get_name()} scores a hard label map, one channel; got {output.shape[1]} channels.",
                "Put an Argmax before it.",
            )
        output, target = Dice._masked(output, targets)
        target = Dice.on_grid(output, target)
        per_item = [
            # a map resampled onto the output's grid comes back float: find_objects takes integer labels
            self._per_label(
                output[item, 0].round().to(torch.int64).cpu().numpy(),
                target[item, 0].round().to(torch.int64).cpu().numpy(),
                self._spacing(output_grid(attributes, output_attributes), item, output.dim() - 2),
            )
            for item in range(output.shape[0])
        ]
        labels = sorted({label for scores in per_item for label in scores})
        values = {}
        for label in labels:
            scored = [scores[label] for scores in per_item if not np.isnan(scores.get(label, np.nan))]
            values[label] = float(np.mean(scored)) if scored else float("nan")
        finite = [v for v in values.values() if np.isfinite(v)]
        mean = float(np.mean(finite)) if finite else float("nan")
        return torch.tensor(mean if np.isfinite(mean) else 0.0, dtype=torch.float32, device=output.device), values


class HausdorffDistance(_SurfaceCriterion):
    """The robust Hausdorff distance in mm, the ``percentile`` of the surface distances (95 by default: HD95). A
    label the output lost is infinite and left out of the mean."""

    def __init__(self, percentile: float = 95.0, labels: list[int] | None = None) -> None:
        super().__init__(labels)
        self._percentile = float(percentile)

    def _value(self, distances: SurfaceDistances) -> float:
        return robust_hausdorff(distances, self._percentile)

    def _missing(self) -> float:
        return float("inf")


class SurfaceDice(_SurfaceCriterion):
    """The surface Dice at ``tolerance`` mm (the normalised surface distance of Metrics Reloaded): the share of both
    surfaces that lies within the tolerance of the other. A label the output lost scores 0."""

    maximize = True

    def __init__(self, tolerance: float = 1.0, labels: list[int] | None = None) -> None:
        super().__init__(labels)
        self._tolerance = float(tolerance)

    def _value(self, distances: SurfaceDistances) -> float:
        return surface_dice(distances, self._tolerance)

    def _missing(self) -> float:
        return 0.0
