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


"""The regularity of a displacement field: the Jacobian determinant of ``x + u(x)``, the standard deviation of its
log (SDlogJ) and the fraction of folds (a determinant at or below zero), within any number of masks.

The output is the field, one component per channel, in mm along the world axes of its geometry (``Origin``,
``Spacing``, ``Direction`` in its attributes, as ITK writes them); every target is a mask, and the statistics are
reported within each. The determinant is ITK's ``DisplacementFieldJacobianDeterminant``: the components rotated
into the index axes, central differences over the spacing, the edge voxel repeated past the faces. It is computed
once, slab by slab in float64, so the largest field never holds a derivative map whole.
"""

from collections.abc import Iterator
from typing import Any

import numpy as np
import torch

from konfai.metric.measure.base import CriterionOutput, CriterionWithAttribute
from konfai.utils.dataset import Attribute
from konfai.utils.errors import DatasetManagerError, MeasureError

#: Bytes of one float64 map per slab: a slab holds the components and their derivatives, a dozen maps.
SLAB_BYTES = 8 << 20
LOG_FOUR = float(np.log(4.0))


def _geometry(field: list[Attribute], item: int, dimensions: int) -> tuple[np.ndarray, np.ndarray]:
    """The spacing along the array's axes (``[z, y, x]``) and the direction (index to world, ITK's row-major)."""
    try:
        spacing = field[item].get_np_array("Spacing")
        direction = field[item].get_np_array("Direction").reshape(dimensions, dimensions)
    except (IndexError, KeyError, ValueError, DatasetManagerError) as error:
        raise MeasureError(
            "A field's Jacobian needs its spacing and direction, which the output's attributes do not carry.",
            "Read the field from a format that keeps its geometry (mha, nii, omezarr).",
        ) from error
    if len(spacing) != dimensions:
        raise MeasureError(f"A spacing of {len(spacing)} values for a field of {dimensions} components.")
    return spacing[::-1].astype(np.float64), direction.astype(np.float64)


def determinants(
    field: torch.Tensor, spacing: np.ndarray, direction: np.ndarray
) -> Iterator[tuple[slice, torch.Tensor]]:
    """``det(I + grad u)`` of a ``[D, *spatial]`` field, slab by slab along the first spatial axis: each slab's rows
    and its determinant map, float64. ``u`` is given along the world axes; ``direction^T`` brings it onto the index
    axes, whose order in memory is the reverse of the attributes' ``(x, y, z)``."""
    dimensions, spatial = field.shape[0], field.shape[1:]
    if dimensions != len(spatial) or dimensions not in (2, 3):
        raise MeasureError(
            f"A field has one component per spatial axis; got {dimensions} components on a {len(spatial)}D grid."
        )
    # world component c to index axis a: u_index[a] = sum_c u[c] direction[c, a]; array axis k is index axis D-1-k
    to_array_axes = torch.tensor(direction[:, ::-1].copy(), dtype=torch.float64, device=field.device)
    plane = int(np.prod(spatial[1:])) * 8
    planes = max(3, SLAB_BYTES // plane)
    for start in range(0, spatial[0], planes):
        stop = min(spatial[0], start + planes)
        rows = slice(max(0, start - 1), min(spatial[0], stop + 1))  # one neighbour each side for the differences
        u = torch.einsum("ca,c...->a...", to_array_axes, field[(slice(None), rows)].to(torch.float64))
        own = slice(start - rows.start, stop - rows.start)
        gradient = []  # gradient[b][a]: d u_a / d x_b over the slab's own rows
        for axis in range(dimensions):
            # central differences, the edge voxel repeated past the faces: (u[1] - u[0]) / 2h there
            length = u.shape[axis + 1]
            derivative = torch.empty_like(u)
            derivative.narrow(axis + 1, 1, length - 2).copy_(u.narrow(axis + 1, 2, length - 2)).sub_(
                u.narrow(axis + 1, 0, length - 2)
            )
            derivative.narrow(axis + 1, 0, 1).copy_(u.narrow(axis + 1, 1, 1)).sub_(u.narrow(axis + 1, 0, 1))
            derivative.narrow(axis + 1, length - 1, 1).copy_(u.narrow(axis + 1, length - 1, 1)).sub_(
                u.narrow(axis + 1, length - 2, 1)
            )
            gradient.append(derivative[(slice(None), own)].div_(2 * spacing[axis]))
        j = [[gradient[b][a] for b in range(dimensions)] for a in range(dimensions)]
        for a in range(dimensions):
            j[a][a] += 1
        if dimensions == 2:
            determinant = j[0][0] * j[1][1] - j[0][1] * j[1][0]
        else:  # the explicit 3x3 determinant: elementwise, where a batched LU is not
            determinant = (
                j[0][0] * (j[1][1] * j[2][2] - j[1][2] * j[2][1])
                - j[0][1] * (j[1][0] * j[2][2] - j[1][2] * j[2][0])
                + j[0][2] * (j[1][0] * j[2][1] - j[1][1] * j[2][0])
            )
        yield slice(start, stop), determinant


class Jacobian(CriterionWithAttribute):
    """SDlogJ and the fraction of folds of a displacement field, within each mask given as a target, from one pass
    over the field. The value reported is SDlogJ within the first mask; the dict holds ``sdlogj<name>`` and
    ``folds<name>`` per mask, ``names`` giving each mask's suffix (``["_body", "_mask"]``; by position when left
    out). ``whole`` adds the whole field, under that suffix (``""``), reported first.
    SDlogJ is the standard deviation of ``log(clip(det J + 3, 1e-9, 1e9))`` with ``crop`` voxels left
    out on every face (when every extent allows it); the folds count every voxel of the mask.

    Reducible from disjoint patches read with ``crop`` voxels of halo: a slot scores its own voxels, their
    derivatives taken from the halo, and leaves out those within ``crop`` of the patch's faces, which is the crop at
    a volume face and nothing inside, as the whole volume crops. The running sums (count, sum, sum of squares of
    the log; voxels, folds) add up exactly.
    """

    reducible = True
    halo = 2

    accepts_output_attributes = True

    def __init__(self, names: list[str] | None = None, crop: int = 2, whole: str | None = None) -> None:
        super().__init__()
        self._names = names
        self._whole = whole
        self._crop = int(crop)
        if self._crop > self.halo:
            raise MeasureError(
                f"Jacobian streams with a halo of {self.halo} voxels: a crop of {crop} cannot be served."
            )

    def _sums(
        self,
        field: torch.Tensor,
        masks: list[torch.Tensor | None],
        spacing: np.ndarray,
        direction: np.ndarray,
        core: tuple[slice, ...] | None = None,
    ) -> np.ndarray:
        """Per mask: the count, sum and sum of squares of the log over the slot's voxels at least ``crop`` from the
        patch's faces, then the slot's voxel count and folds. ``core`` is the slot within the patch (the whole patch
        when ``None``); a mask of ``None`` is the whole field."""
        spatial = field.shape[1:]
        if core is None:
            core = tuple(slice(0, extent) for extent in spatial)
        # Nothing is cropped on a volume too thin for it. Seen from a patch, an axis is the volume's own when the
        # slot spans the whole read (no halo on either side); a thin slot at a far face of a thick volume still crops.
        thin = any(
            extent <= 2 * self._crop and c.start == 0 and c.stop == extent
            for c, extent in zip(core, spatial, strict=True)
        )
        crop = 0 if thin else self._crop
        kept = tuple(
            slice(max(c.start, crop), min(c.stop, extent - crop)) for c, extent in zip(core, spatial, strict=True)
        )
        totals = np.zeros((len(masks), 5))

        def within(rows: slice, bounds: slice) -> slice:
            return slice(max(rows.start, bounds.start) - rows.start, min(rows.stop, bounds.stop) - rows.start)

        for rows, determinant in determinants(field, spacing, direction):
            folded = determinant <= 0
            # centred on log 4, the value of a field that does not move: the running sums of a constant cancel exactly
            log = torch.log(torch.clamp(determinant + 3, 1e-9, 1e9)) - LOG_FOUR
            slot, inner = within(rows, core[0]), within(rows, kept[0])
            for index, mask in enumerate(masks):
                patch_mask = None if mask is None else mask[rows]
                if slot.stop > slot.start:
                    region = (slot, *core[1:])
                    if patch_mask is None:
                        totals[index, 3] += folded[region].numel()
                        totals[index, 4] += float(folded[region].sum())
                    else:
                        inside = patch_mask[region]
                        totals[index, 3] += float(inside.sum())
                        totals[index, 4] += float(folded[region][inside].sum())
                if inner.stop > inner.start and all(k.stop > k.start for k in kept[1:]):
                    region = (inner, *kept[1:])
                    cropped = log[region].reshape(-1) if patch_mask is None else log[region][patch_mask[region]]
                    totals[index, 0] += cropped.numel()
                    totals[index, 1] += float(cropped.sum())
                    totals[index, 2] += float((cropped * cropped).sum())
        return totals

    @staticmethod
    def _values(totals: np.ndarray) -> tuple[float, float]:
        count, total, squares, voxels, folded = totals
        sdlogj = float(np.sqrt(max(squares / count - (total / count) ** 2, 0.0))) if count else float("nan")
        return sdlogj, float(folded / voxels) if voxels else float("nan")

    def _names_for(self, targets: tuple[torch.Tensor, ...]) -> list[str]:
        if not targets and self._whole is None:
            raise MeasureError(
                "Jacobian needs at least one mask as target, or the whole field.",
                "Give it mask groups, the whole field or both: DVF: {BODY: {Jacobian: {whole: '', names: ['_body']}}}.",
            )
        names = self._names if self._names is not None else [f"_{index}" for index in range(len(targets))]
        if len(names) != len(targets):
            raise MeasureError(f"Jacobian was given {len(names)} names for {len(targets)} masks.")
        return ([self._whole] if self._whole is not None else []) + list(names)

    def _totals(
        self,
        output: torch.Tensor,
        targets: tuple[torch.Tensor, ...],
        field: list[Attribute],
        core: tuple[slice, ...] | None,
    ) -> np.ndarray:
        whole: list[torch.Tensor | None] = [None] if self._whole is not None else []
        totals = np.zeros((len(whole) + len(targets), 5))
        for item in range(output.shape[0]):
            spacing, direction = _geometry(field, item, output.dim() - 2)
            masks = whole + [target[item, 0].to(device=output.device) > 0 for target in targets]
            totals += self._sums(output[item], masks, spacing, direction, core)
        return totals

    def _report(self, output: torch.Tensor, names: list[str], totals: np.ndarray) -> CriterionOutput:
        values: dict[str, float] = {}
        for name, total in zip(names, totals, strict=True):
            values[f"sdlogj{name}"], values[f"folds{name}"] = self._values(total)
        first = values[f"sdlogj{names[0]}"]
        return output.new_tensor(first if np.isfinite(first) else 0.0), values

    def forward(  # type: ignore[override]  # the added keywords are this subclass's contract
        self,
        output: torch.Tensor,
        *targets: torch.Tensor,
        attributes: list[list[Attribute]],
        output_attributes: list[Attribute] | None = None,
    ) -> CriterionOutput:
        names = self._names_for(targets)
        return self._report(
            output, names, self._totals(output, targets, self.output_grid(attributes, output_attributes), None)
        )

    def partial_metric(  # type: ignore[override]  # core= and the attributes are this metric's streamed contract
        self,
        output: torch.Tensor,
        *targets: torch.Tensor,
        core: tuple[slice, ...] | None = None,
        attributes: list[list[Attribute]],
        output_attributes: list[Attribute] | None = None,
    ) -> Any:
        """The running sums of one patch's slot: (names, totals, a tensor to report on)."""
        names = self._names_for(targets)
        field = self.output_grid(attributes, output_attributes)
        return names, self._totals(output, targets, field, core), output[:1, :1, :1].detach()

    def combine_metric(self, states: list[Any]) -> CriterionOutput:
        names, _, on = states[0]
        return self._report(on, names, sum(totals for _, totals, _ in states))
