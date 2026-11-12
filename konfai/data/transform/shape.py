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


"""Extent and orientation transforms: padding, cropping, axis permutation, flips, canonical orientation, gradients."""

import itertools
import math
from typing import Annotated, Any, ClassVar, cast

import numpy as np
import torch
import torch.nn.functional as F

from konfai.data.geometry import (
    SIGNED_PERMUTATION_ATOL_FLOAT64,
    AxisRemap,
    Grid,
    apply_remap,
    invert_remap,
    remap_region,
    remap_shape,
    signed_permutation,
)
from konfai.data.sampling import default_interpolation, gather, source_index_rows, walk_rows
from konfai.data.transform.base import (
    _RANK_CHANGE,
    LocalityKind,
    PatchLocality,
    RegionContext,
    Transform,
    TransformInverse,
    sitk,
)
from konfai.data.transform.resample import _resample_with_sitk
from konfai.utils.config import Range
from konfai.utils.dataset import Attribute, Dataset
from konfai.utils.errors import TransformError


class Padding(TransformInverse):
    """Pad the volume's borders (``padding`` pairs in ``F.pad`` order: last axis first).

    A pad is a translation of the volume into a larger grid plus a border it fills, so it streams as
    a REGRID: a target region pulls the source rows it covers and the stage fills what the clamp cut.
    """

    working_multiple = 0.0

    locality = LocalityKind.REGRID

    def __init__(self, padding: list[int] = [0, 0, 0, 0, 0, 0], mode: str = "constant", inverse: bool = True) -> None:
        super().__init__(inverse)
        self.padding = padding
        self.mode = mode

    def _mode(self) -> tuple[str, float]:
        parts = self.mode.split(":")
        return parts[0], float(parts[1]) if len(parts) == 2 else 0.0

    def _pairs(self, dims: int) -> list[tuple[int, int]]:
        """``(before, after)`` per spatial axis in array order, zero where the config names none."""
        pairs = [(0, 0)] * dims
        for dim in range(min(len(self.padding) // 2, dims)):
            pairs[-dim - 1] = (int(self.padding[dim * 2]), int(self.padding[dim * 2 + 1]))
        return pairs

    def _shift_origin(self, cache_attribute: Attribute) -> None:
        if not Grid.readable(cache_attribute):
            return
        # ITK's own association: a point is ``O + D (index * spacing)``, so an index shift moves the
        # origin through the direction itself, which is not its inverse transpose once it shears.
        origin = cache_attribute.get_np_array("Origin")
        spacing = cache_attribute.get_np_array("Spacing")
        matrix = cache_attribute.get_np_array("Direction").reshape((len(origin), len(origin)))
        shift = np.zeros(len(origin))
        for dim in range(min(len(self.padding) // 2, len(origin))):  # F.pad order: (x, y, z)
            shift[dim] = -self.padding[dim * 2] * spacing[dim]
        cache_attribute["Origin"] = origin + matrix @ shift

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        self._shift_origin(cache_attribute)
        mode, value = self._mode()
        return F.pad(tensor.unsqueeze(0), tuple(self.padding), mode, value).squeeze(0)

    def transform_shape(self, group_src: str, name: str, shape: list[int], cache_attribute: Attribute) -> list[int]:
        return [extent + before + after for extent, (before, after) in zip(shape, self._pairs(len(shape)), strict=True)]

    def write_stream_cache_attribute(
        self, cache_attribute: Attribute, source_spatial_shape: list[int], name: str = ""
    ) -> None:
        del source_spatial_shape, name
        self._shift_origin(cache_attribute)

    def stream_region_source(
        self, name: str, target_slices: tuple[slice, ...], source_spatial_shape: list[int], cache_attribute: Attribute
    ) -> list[slice]:
        del name, cache_attribute
        circular = self._mode()[0] == "circular"
        pull: list[slice] = []
        for target, (before, _after), extent in zip(
            target_slices, self._pairs(len(target_slices)), source_spatial_shape, strict=False
        ):
            low, high = max(0, target.start - before), min(extent, target.stop - before)
            if target.start < before:  # reaches the low border: keep the rows a reflection reads back from
                low, high = 0, max(high, min(extent, before - target.start + 1))
            if target.stop > extent + before:  # reaches the high border
                high, low = extent, min(low, max(0, extent - (target.stop - extent - before) - 1))
            if circular and (target.start < before or target.stop > extent + before):
                low, high = 0, extent  # a circular border is filled from the opposite end of the axis
            pull.append(slice(low, max(high, low + 1)))
        return pull

    def stream_region(
        self, name: str, tensor: torch.Tensor, context: RegionContext, cache_attribute: Attribute
    ) -> torch.Tensor:
        del name, cache_attribute
        mode, value = self._mode()
        return _pad_region(tensor, context, [before for before, _ in self._pairs(len(context.target))], mode, value)

    def inverse_patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        # The inverse drops the padded border and keeps what remains: a CROP.
        return PatchLocality(LocalityKind.CROP)

    def inverse_transform_shape(self, shape: list[int], cache_attribute: Attribute) -> list[int]:
        return [extent - before - after for extent, (before, after) in zip(shape, self._pairs(len(shape)), strict=True)]

    def stream_region_target(
        self,
        name: str,
        target_slices: tuple[slice, ...],
        source_spatial_shape: list[int],
        cache_attribute: Attribute,
    ) -> list[slice]:
        # Output index o holds input index o + pad_before.
        return [
            slice(target.start + before, target.stop + before)
            for target, (before, _after) in zip(target_slices, self._pairs(len(target_slices)), strict=True)
        ]

    def inverse(self, name: str, tensor: torch.Tensor, cache_attribute: dict[str, torch.Tensor]) -> torch.Tensor:
        if "Origin" in cache_attribute and "Spacing" in cache_attribute and "Direction" in cache_attribute:
            cache_attribute.pop("Origin")
        spatial = tensor.shape[1:]
        crops = [
            slice(before, extent - after)
            for extent, (before, after) in zip(spatial, self._pairs(len(spatial)), strict=True)
        ]
        return tensor[(slice(None), *crops)]


def _pad_region(
    tensor: torch.Tensor, context: RegionContext, before: list[int], mode: str, value: float
) -> torch.Tensor:
    """The region ``context.target`` of a padded volume, from the block ``tensor`` its input holds at
    ``context.source``: input index ``i`` lands at ``i + before`` on each axis, and ``mode`` fills what the
    block does not reach."""
    pads: list[tuple[int, int]] = []
    crops: list[slice] = []
    for target, source, shift in zip(context.target, context.source, before, strict=True):
        start, stop = source.start + shift, source.stop + shift  # where the block sits in the output
        low, high = max(0, start - target.start), max(0, target.stop - stop)
        pads.append((low, high))
        crops.append(slice(target.start - (start - low), target.stop - (start - low)))
    padded = F.pad(tensor.unsqueeze(0), tuple(x for pair in reversed(pads) for x in pair), mode, value).squeeze(0)
    return padded[(slice(None), *crops)]


def _pulled(target_slices: tuple[slice, ...], shifts: list[int], extents: list[int]) -> list[slice]:
    """The rows of a volume of ``extents`` that a region of a shifted copy pulls, input index ``i`` landing at
    ``i + shift``: past the volume, its nearest row, which an edge repeats."""
    pull = []
    for target, shift, extent in zip(target_slices, shifts, extents, strict=False):
        low = min(max(target.start - int(shift), 0), int(extent) - 1)
        pull.append(slice(low, max(min(target.stop - int(shift), int(extent)), low + 1)))
    return pull


def _margined(tensor: torch.Tensor, box: np.ndarray, border: tuple[str, float]) -> torch.Tensor:
    """``tensor`` with ``box``'s margins (before, after per spatial axis) taken off, a margin below zero added with
    ``border`` (``F.pad``'s mode and value; ``F.pad`` takes a negative pad as a crop). A crop alone is a view."""
    if (box >= 0).all():
        crops = [
            slice(int(near), extent - int(far)) for (near, far), extent in zip(box, tensor.shape[1:], strict=False)
        ]
        return tensor[(slice(None), *crops)]
    mode, value = border
    pads = tuple(-int(margin) for margins in reversed(box) for margin in margins)
    return F.pad(tensor.unsqueeze(0), pads, mode, value).squeeze(0)


class Squeeze(TransformInverse):
    working_multiple = 0.0

    def __init__(self, dim: int, inverse: bool = True) -> None:
        super().__init__(inverse)
        self.dim = dim

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        return PatchLocality(LocalityKind.WHOLE_VOLUME, reason=_RANK_CHANGE)

    def transform_shape(self, group_src: str, name: str, shape: list[int], cache_attribute: Attribute) -> list[int]:
        # ``shape`` is the channel-stripped spatial shape, so the runtime tensor is [C, *shape] and
        # ``self.dim`` indexes into that. A spatial axis is dropped only when it is size 1.
        axis = self.dim if self.dim >= 0 else self.dim + len(shape) + 1
        if 1 <= axis <= len(shape) and shape[axis - 1] == 1:
            return shape[: axis - 1] + shape[axis:]
        return shape

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        return tensor.squeeze(self.dim)

    def inverse(self, name: str, tensor: torch.Tensor, cache_attribute: dict[str, Any]) -> torch.Tensor:
        return tensor.unsqueeze(self.dim)


class Crop(TransformInverse):
    """Crop a volume to the bounding box of a foreground.

    The content-dependent box is computed once (``transform_shape``) and kept on the case as ``box``
    margins; cropping is then the translation ``out[o] = volume[o + start]``. Dropped voxels make it
    a ``LocalityKind.CROP``, the stored volume's statistics not being the output's.

    The foreground is, by default, the stored volume's voxels above its 5th percentile; with ``nonzero``
    its voxels other than 0; with ``labels`` the voxels holding one of
    them. With ``source`` the box is measured on another group of the same case (a segmentation, say),
    its non-zero voxels or its ``labels``, and carried onto the grid this stage is handed through
    physical space: the voxels whose centres lie in the box's outer faces. On the same grid that is the
    box itself, index for index. ``margin_mm`` grows the box by ``int(margin_mm / spacing)`` voxels per
    axis, the spacing read as float32 as a NIfTI header stores it, clipped to the volume. Without ``source`` the box is measured on the stored volume, so an
    input a stage before moved onto another grid is refused, unless ``handed``: the box is then that of
    the volume as the stages before this one in its chain leave it, found by replaying them once on the
    stored volume as the chain is planned (the volume is held whole, on the host, for that replay), as
    the portable runtime crops the volume it is handed.

    Spacing and direction are kept and the origin moves to the box's near corner,
    ``O + D (start * spacing)`` (ITK's ``RegionOfInterest``); the inverse restores it and pads the
    dropped voxels back, by replicating the edge or, with ``fill``, with that value.

    A case with no foreground is refused by name, unless ``empty: fill``: the box is then the near corner
    voxel grown by the margin, on which the stages after it run, and the inverse writes ``fill`` over the
    whole input grid, dropping what they computed: with ``fill: 0``, an empty segmentation.

    With ``shape`` there is no foreground to find: the volume becomes exactly ``shape`` voxels per axis in array order
    (an axis at 0 or below kept whole), read off the header: its centre where it is larger, the first ``(n - c) // 2``
    dropped, and padded with ``fill`` (its edge without) where it is smaller, ``(c - n) // 2`` before. A margin of
    the box below zero is such a pad: the crop then streams as a REGRID, as ``Padding`` does, and the inverse takes
    the pad back as it puts the dropped voxels back.
    """

    working_multiple = 0.0

    #: Set by the portable compiler while it builds the workflow: a Crop whose ``source`` no dataset of the build
    #: holds plans no box. The group is another app's output (app.json ``needs``), staged beside the case at run
    #: time, where the runtime measures the box; a konfai run keeps refusing the absent group.
    plan_without_source: ClassVar[bool] = False

    def __init__(
        self,
        source: str | None = None,
        labels: list[int] | None = None,
        margin_mm: float = 0.0,
        fill: float | None = None,
        nonzero: bool = False,
        handed: bool = False,
        empty: str = "refuse",
        inverse: bool = True,
        shape: Annotated[
            list[int] | None,
            Range(0, 4096),
            "The volume's centre, cropped and padded to exactly this many voxels per axis in array order (an axis at 0"
            " kept whole); None crops to a foreground instead.",
        ] = None,
    ) -> None:
        super().__init__(inverse)
        if shape is not None and (source is not None or labels is not None or nonzero or handed or margin_mm):
            raise TransformError(
                f"'Crop' was given shape: {shape} and a foreground to find its box in.",
                "A shape crops the volume's centre: give it without source, labels, nonzero, handed or margin_mm.",
            )
        if empty not in ("refuse", "fill") or (empty == "fill" and fill is None):
            raise TransformError(
                f"'Crop' was given empty: {empty!r} and fill: {fill}.",
                "empty is refuse (a case with no foreground is refused) or fill (the inverse writes fill"
                " everywhere), which needs fill to be a value.",
            )
        if margin_mm < 0:
            raise TransformError(
                f"'Crop' was given margin_mm: {margin_mm}.",
                "The margin grows the box; give a length in mm of 0 or more.",
            )
        if handed and source is not None:
            raise TransformError(
                f"'Crop' was given both handed and source: '{source}'.",
                "handed measures the box on the volume this stage is handed, source on another group: give one.",
            )
        self.source = source
        self.labels = None if labels is None else [int(label) for label in labels]
        self.margin_mm = float(margin_mm)
        self.fill = fill
        self.nonzero = bool(nonzero)
        self.handed = bool(handed)
        self.empty = empty
        self.shape = None if shape is None else [int(extent) for extent in shape]

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        # ``transform_shape`` puts the box on the case, but a group carries only what its writer stored.
        if "box" not in cache_attribute:
            return PatchLocality(
                LocalityKind.WHOLE_VOLUME,
                reason="the case carries no 'box' yet; the foreground box is computed and recorded"
                " as the chain is planned, and only a read can find it",
            )
        # A margin below zero pads: the region pulls what of the volume it covers and fills the rest.
        return PatchLocality(LocalityKind.REGRID if self._pads(cache_attribute) else LocalityKind.CROP)

    def _pads(self, cache_attribute: Attribute) -> bool:
        return bool((self._parse_box(cache_attribute["box"]) < 0).any())

    def _border(self) -> tuple[str, float]:
        """How a pad fills: with ``fill``, else with the edge."""
        return ("replicate", 0.0) if self.fill is None else ("constant", float(self.fill))

    def stream_region_source(
        self,
        name: str,
        target_slices: tuple[slice, ...],
        source_spatial_shape: list[int],
        cache_attribute: Attribute,
    ) -> list[slice]:
        # Output index o holds source index o + start; a pad pulls only what of the volume the region covers.
        box = Crop._parse_box(cache_attribute["box"])
        if self._pads(cache_attribute):
            return _pulled(target_slices, [-int(start) for start, _ in box], source_spatial_shape)
        return [
            slice(target.start + int(start), target.stop + int(start))
            for target, (start, _) in zip(target_slices, box, strict=False)
        ]

    def stream_region(
        self, name: str, tensor: torch.Tensor, context: RegionContext, cache_attribute: Attribute
    ) -> torch.Tensor:
        if "box" not in cache_attribute or not self._pads(cache_attribute):
            return self(name, tensor, cache_attribute)
        mode, value = self._border()
        box = Crop._parse_box(cache_attribute["box"])
        return _pad_region(tensor, context, [-int(start) for start, _ in box], mode, value)

    def write_stream_cache_attribute(
        self, cache_attribute: Attribute, source_spatial_shape: list[int], name: str = ""
    ) -> None:
        del name
        # A read stores the geometry stacked (``Origin_0``): only the stack-aware ``in`` sees it.
        if "box" not in cache_attribute or not Grid.readable(cache_attribute):
            return
        # The new origin is the physical point the box's near corner sat on, ``O + D (start * spacing)``.
        box = Crop._parse_box(cache_attribute["box"])
        origin = cache_attribute.get_np_array("Origin")
        spacing = cache_attribute.get_np_array("Spacing")
        matrix = cache_attribute.get_np_array("Direction").reshape((len(origin), len(origin)))
        shift = np.zeros(len(origin))
        for dim in range(box.shape[0]):  # box rows are in array order, the geometry in (x, y, z)
            shift[-dim - 1] = box[dim][0] * spacing[-dim - 1]
        cache_attribute["Origin"] = origin + matrix @ shift

    def transform_shape(self, group_src: str, name: str, shape: list[int], cache_attribute: Attribute) -> list[int]:
        # The crop box is the foreground bounding box, so the output shape needs the pixel data: a
        # box already on the case is reused, otherwise it is computed once. It has one row per
        # spatial axis, aligned with the channel-stripped ``shape``.
        if any(isinstance(stage, Crop) for stage in self.preceding or ()):
            raise TransformError(
                "'Crop' keeps its box under 'box', where an earlier 'Crop' of the chain keeps its own.",
                "Crop once.",
            )
        if "box" in cache_attribute:
            known = self._parse_box(cache_attribute["box"])
            return [int(s - a - b) for (a, b), s in zip(known, shape, strict=False)]
        box: np.ndarray | None
        if self.shape is not None:
            # Margins, not a box: a margin below zero is a pad.
            margins = self._centre(shape)
            cache_attribute["box"] = margins
            return [int(s - a - b) for (a, b), s in zip(margins, shape, strict=True)]
        if self.handed:
            holder = self.dataset_holding(group_src, name)
            if holder is None:
                return shape
            box = self._handed_box(holder, group_src, name, shape)
            where = f"'{group_src}/{name}' as the stages before 'Crop' leave it"
        elif self.source is None:
            holder = self.dataset_holding(group_src, name)
            if holder is None:
                return shape
            Crop._require_the_stored_grid(holder, group_src, name, shape, cache_attribute)
            box = self._foreground_box(holder, group_src, name, self.labels, nonzero=self.nonzero)
            where = f"'{group_src}/{name}'"
        else:
            holder = self.dataset_holding(self.source, name)
            if holder is None:
                if Crop.plan_without_source:
                    return shape
                raise TransformError(
                    f"'Crop' measures its box on the group '{self.source}', and no dataset of the run holds it"
                    f" for case '{name}'.",
                    f"Put '{self.source}' next to the case's other groups (the output of the app that makes it).",
                )
            box = self._foreground_box(holder, self.source, name, self.labels, nonzero=True)
            if box is not None:
                box = Crop._onto_the_handed_grid(box, holder, self.source, name, shape, cache_attribute)
            where = f"'{self.source}/{name}'"
        if box is None:
            self._nothing_to_crop(where)
            # empty: fill. The near corner voxel, grown as any box: a zoom Resample cannot undo a 1-voxel axis.
            box = np.zeros((len(shape), 2), dtype=np.int64)
            cache_attribute["boxempty"] = 1
        box = self._grown(box, shape, cache_attribute)
        for i, ((_, b), s) in enumerate(zip(box, shape, strict=False)):
            # The scan reports the last foreground index; the box carries the margin after it.
            box[i][1] = max(int(s - b - 1), 0)
        cache_attribute["box"] = box
        return [int(s - a - b) for (a, b), s in zip(box, shape, strict=False)]

    def _centre(self, shape: list[int]) -> np.ndarray:
        """The margins per axis that make a volume of ``shape`` exactly ``self.shape``, mri_synthseg's ``crop_volume``
        then ``pad_volume``: ``(n - c) // 2`` dropped before where it is larger, ``(c - n) // 2`` added before (a
        margin below zero) where it is smaller, the rest after."""
        rows = []
        for extent, wanted in zip(shape, cast("list[int]", self.shape), strict=True):
            wanted = extent if wanted <= 0 else wanted
            before = (extent - wanted) // 2 if extent >= wanted else -((wanted - extent) // 2)
            rows.append([before, extent - wanted - before])
        return np.asarray(rows, dtype=np.int64)

    def _handed_box(self, holder: Dataset, group_src: str, name: str, shape: list[int]) -> np.ndarray | None:
        """The foreground box of the volume as the stages before this one leave it: those stages replayed
        once, whole, on the stored volume and a scratch copy of its header (``transform_shape`` then the
        call, as the plan and the whole-volume route run them). Memoised on the dataset by the rule, the
        stages before it (the chain's own, built once) and the shape they hand, so a chain replays once a case."""
        if self.preceding is None:
            raise TransformError(
                "'Crop' with handed measures its box on the volume it is handed, and this stage was not bound in"
                " a group's chain, so the stages before it are unknown.",
                "Put the Crop in a group's transforms, or drop handed.",
            )
        memo = holder.case_facts.setdefault((group_src, name), {})
        rule = f"labels {self.labels}" if self.labels is not None else "nonzero" if self.nonzero else "percentile"
        # The stages themselves, not their names: two chains of one group (CT_1mm, CT_2mm) name the same classes. By
        # identity (a stage may compare equal to another, or not hash), and kept beside their box, so a freed chain's
        # ids are never another chain's while the box is remembered.
        key = ("box handed", rule, tuple(id(stage) for stage in self.preceding), tuple(shape))
        if key not in memo:
            self._price_replay(holder, group_src, name, shape)
            array, stored = holder.read_data(group_src, name)
            tensor, attribute = torch.from_numpy(array), Attribute(stored)
            for stage in self.preceding:
                stage.transform_shape(group_src, name, list(tensor.shape[1:]), attribute)
                tensor = stage(name, tensor, attribute)
            if [int(extent) for extent in tensor.shape[1:]] != [int(extent) for extent in shape]:
                raise TransformError(
                    f"'Crop' replayed the stages before it on '{group_src}/{name}' and got {list(tensor.shape[1:])}"
                    f" where the plan hands it {list(shape)}.",
                    "Report this: a stage's transform_shape disagrees with its call.",
                )
            memo[key] = (tuple(self.preceding), self._box_of(tensor))
        box = memo[key][1]
        return None if box is None else np.array(box, copy=True)

    def _price_replay(self, holder: Dataset, group_src: str, name: str, handed: list[int]) -> None:
        """Refuse a replay whose whole volume, in float32 at the larger of its stored and handed extents (a
        Resample before it may enlarge it), with the most any stage before it holds, is over this rank's memory
        budget: it is read before any plan prices it."""
        from konfai.utils.budget import node_local_ranks, per_rank_budget_bytes, resolve_memory_budget

        stored, _ = holder.get_infos(group_src, name)
        channels = stored[0] if len(stored) > len(handed) else 1
        voxels = channels * max(math.prod(stored[-len(handed) :]), math.prod(handed))
        stages = cast("list[Transform]", self.preceding)
        needed = voxels * 4 * (1 + max((stage.case_working_multiple(name) for stage in stages), default=1.0))
        budget = per_rank_budget_bytes() or resolve_memory_budget(None).per_rank_bytes(node_local_ranks())
        if needed > budget:
            raise TransformError(
                f"'Crop' with handed replays the stages before it on '{group_src}/{name}' whole: {needed / 2**30:.1f}"
                f" GiB, over this rank's memory budget of {budget / 2**30:.1f} GiB.",
                "Raise the memory budget, measure the box on another group with 'source', or drop handed.",
            )

    def _box_of(self, tensor: torch.Tensor) -> np.ndarray | None:
        """First and last foreground index per spatial axis of ``tensor``, every channel voting, by this
        stage's rule: ``labels``, ``nonzero``, else above its 5th percentile (numpy's quantile, as
        ``Dataset.read_data_quantile`` reads the stored volume's); ``None`` for none."""
        values = tensor.detach().cpu().numpy()
        if self.labels is not None:
            voxels = np.isin(values, np.asarray(self.labels))
        elif self.nonzero:
            voxels = values != 0
        else:
            voxels = values > np.quantile(values, 0.05)
        foreground = np.any(voxels, axis=0)
        return np.asarray(self._spans(foreground), dtype=np.int64) if foreground.any() else None

    @staticmethod
    def _spans(foreground: np.ndarray) -> list[list[int]]:
        """The first and last index of ``foreground`` along each axis (it holds one voxel at least)."""
        others = [tuple(o for o in range(foreground.ndim) if o != axis) for axis in range(foreground.ndim)]
        return [[int(hits[0]), int(hits[-1])] for hits in (np.flatnonzero(np.any(foreground, axis=o)) for o in others)]

    def _nothing_to_crop(self, where: str) -> None:
        """Refuse a case with no foreground in ``where``, by this stage's rule, unless ``empty: fill``."""
        if self.empty == "fill":
            return
        if self.labels is not None:
            what = f"no voxel holding one of the labels {self.labels}"
        elif self.nonzero or self.source is not None:
            what = "no voxel other than 0"
        else:
            what = "every voxel is at or under its 5th percentile"
        raise TransformError(
            f"'Crop' found no foreground in {where}: {what}.",
            "There is nothing to crop to: give the Crop empty: fill to write its fill everywhere, drop it for this"
            " case or crop to another mask.",
        )

    def _grown(self, box: np.ndarray, shape: list[int], cache_attribute: Attribute) -> np.ndarray:
        """``box`` (first and last index per array axis) grown by ``margin_mm`` and clipped to ``shape``,
        by ``int(margin_mm / spacing)`` voxels per axis, the spacing read as float32."""
        if self.margin_mm > 0:
            spacing_zyx = (
                cache_attribute.get_np_array("Spacing")[::-1] if "Spacing" in cache_attribute else np.ones(len(shape))
            )
            grow = (self.margin_mm / spacing_zyx.astype(np.float32).astype(np.float64)).astype(np.int64)
            box = box + np.stack([-grow, grow], axis=1)
        box[:, 0] = np.maximum(box[:, 0], 0)
        box[:, 1] = np.minimum(box[:, 1], np.asarray(shape, dtype=np.int64) - 1)
        if np.any(box[:, 0] > box[:, 1]):
            raise TransformError(
                f"'Crop' found the box of '{self.source}' outside the volume it crops.",
                "The two groups do not overlap in physical space: check that they belong to the same case.",
            )
        return box

    @staticmethod
    def _onto_the_handed_grid(
        box: np.ndarray, holder: Dataset, source: str, name: str, shape: list[int], cache_attribute: Attribute
    ) -> np.ndarray:
        """The source's box (first and last index per array axis, on its own grid) as the first and last
        voxels of the handed grid whose centres lie within its outer faces. The same grid, the same box."""
        source_shape, source_attribute = holder.get_infos(source, name)
        spatial = [int(extent) for extent in source_shape[1:]]
        if not Grid.readable(source_attribute) and not Grid.readable(cache_attribute):
            if spatial != [int(extent) for extent in shape]:
                raise TransformError(
                    f"'Crop' measures its box on '{source}/{name}', of extent {spatial}, for a volume of"
                    f" extent {list(shape)}, and neither carries a geometry to relate them.",
                    "Give both groups their header (mha, nii, h5), or the same extent.",
                )
            return box
        source_grid = Grid.of(spatial, source_attribute, f"'{source}/{name}'")
        handed = Grid.of([int(extent) for extent in shape], cache_attribute, "the input of 'Crop'")
        low, high = source_grid.box_on(handed, tuple(slice(int(a), int(b) + 1) for a, b in box))
        # Centres on a face count in: the tolerance keeps a box on the handed lattice exactly itself.
        first = np.ceil(low - 1e-6).astype(np.int64)[::-1]
        last = np.floor(high + 1e-6).astype(np.int64)[::-1]
        return np.stack([first, last], axis=1)

    @staticmethod
    def _require_the_stored_grid(
        source: Dataset, group_src: str, name: str, shape: list[int], cache_attribute: Attribute
    ) -> None:
        """Refuse an input on another grid than the stored volume's: the box is measured on the stored
        volume, so it would land elsewhere. The tolerance absorbs a geometry carried through float32."""
        stored_shape, stored = source.get_infos(group_src, name)
        spatial = [int(extent) for extent in stored_shape[1:]]
        if [int(extent) for extent in shape] == spatial:
            handed, _ = Grid.from_header(spatial, cache_attribute, "the input of 'Crop'")
            kept, _ = Grid.from_header(spatial, stored, f"'{group_src}/{name}'")
            if all(
                np.allclose(a, b, rtol=1e-6, atol=1e-6)
                for a, b in (
                    (handed.origin_xyz, kept.origin_xyz),
                    (handed.spacing_xyz, kept.spacing_xyz),
                    (handed.direction_xyz, kept.direction_xyz),
                )
            ):
                return
        raise TransformError(
            f"'Crop' measures its box on the stored volume '{group_src}/{name}', but a stage before it moved"
            " the volume onto another grid: the box would cut the wrong region.",
            "Place Crop before the stages that change the grid (Resample, Padding, Canonical),"
            " measure the box on another group with 'source', or crop the written volume in a second"
            " TRANSFORM run.",
        )

    @staticmethod
    def _foreground_box(
        source: Dataset, group: str, name: str, labels: list[int] | None = None, nonzero: bool = False
    ) -> np.ndarray | None:
        """The bounding box of the foreground of ``group``, first and last index per spatial axis, from
        bounded passes over the stored volume, every channel voting, ``None`` for none. The foreground is
        the voxels holding one of ``labels``; without them, the non-zero voxels (``nonzero``) or those
        above the 5th percentile. Memoised on the dataset."""
        memo = source.case_facts.setdefault((group, name), {})
        key = "box" if labels is None and not nonzero else f"box {labels if labels is not None else 'nonzero'}"
        if key not in memo:
            if labels is not None:
                wanted = np.asarray(labels)
            elif not nonzero:
                threshold = source.read_data_quantile(group, name, 0.05)
            first: np.ndarray | None = None
            last: np.ndarray | None = None
            offset = 0
            for block in source.iter_data_blocks(group, name)():
                if labels is not None:
                    voxels = np.isin(block, wanted)
                else:
                    voxels = block != 0 if nonzero else block > threshold
                foreground = np.any(voxels, axis=0)  # every channel votes
                if foreground.any():
                    spans = np.asarray(Crop._spans(foreground))
                    spans[0] += offset  # the block's first row in the volume
                    first = spans[:, 0] if first is None else np.minimum(first, spans[:, 0])
                    last = spans[:, 1] if last is None else np.maximum(last, spans[:, 1])
                offset += int(block.shape[1])
            memo[key] = None if first is None or last is None else np.stack([first, last], axis=1).astype(np.int64)
        return None if memo[key] is None else np.array(memo[key], copy=True)

    @staticmethod
    def _parse_box(box_str: str) -> np.ndarray:
        flat = np.fromstring(box_str.replace("[", " ").replace("]", " "), sep=" ", dtype=np.int64)
        return flat.reshape(-1, 2)

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        if "box" not in cache_attribute:
            return tensor
        box = self._parse_box(cache_attribute["box"])
        self.write_stream_cache_attribute(cache_attribute, list(tensor.shape[1:]), name)
        return _margined(tensor, box, self._border())

    def _pop_box(self, cache_attribute: Attribute) -> tuple[np.ndarray, bool]:
        """Take back what the forward recorded: the box, and whether the case had no foreground."""
        box = self._parse_box(cache_attribute.pop("box"))
        # Only an ``empty: fill`` stage records it; another Crop of the chain leaves it to that one.
        empty = self.empty == "fill" and "boxempty" in cache_attribute
        if empty:
            cache_attribute.pop("boxempty")
        if Grid.readable(cache_attribute):  # the forward pushed an origin only on a case with a geometry
            cache_attribute.pop("Origin")
        return box, empty

    def inverse(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        if "box" not in cache_attribute:
            return tensor
        box, empty = self._pop_box(cache_attribute)
        mode, value = self._border()
        restored = _margined(tensor, -box, (mode, value))
        return restored.fill_(value) if empty else restored

    def inverse_patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        # Padding the box back is Padding's forward: a REGRID pulling the rows of the box a region covers and
        # filling the rest. With no box the inverse hands its input back.
        return PatchLocality(LocalityKind.REGRID if "box" in cache_attribute else LocalityKind.POINTWISE)

    def inverse_transform_shape(self, shape: list[int], cache_attribute: Attribute) -> list[int]:
        if "box" not in cache_attribute:
            return shape
        box = self._parse_box(cache_attribute["box"])
        return [int(extent + near + far) for extent, (near, far) in zip(shape, box, strict=False)]

    def inverse_stream_cache_attribute(self, cache_attribute: Attribute, source_spatial_shape: list[int]) -> None:
        del source_spatial_shape
        if "box" in cache_attribute:
            self._pop_box(cache_attribute)

    def stream_region_target(
        self, name: str, target_slices: tuple[slice, ...], source_spatial_shape: list[int], cache_attribute: Attribute
    ) -> list[slice]:
        # Output index o holds input index o - start; past the box, the box's nearest row, which the edge repeats.
        del name
        box = self._parse_box(cache_attribute["box"])
        return _pulled(target_slices, [int(start) for start, _ in box], source_spatial_shape)

    def stream_region_inverse(
        self, name: str, tensor: torch.Tensor, context: RegionContext, cache_attribute: Attribute
    ) -> torch.Tensor:
        del name
        box, empty = self._pop_box(Attribute(cache_attribute))
        mode, value = self._border()
        region = _pad_region(tensor, context, [int(start) for start, _ in box], mode, value)
        return region.fill_(value) if empty else region


def _has_geometry(cache_attribute: Attribute, rank: int) -> bool:
    """Whether the case carries a header of ``rank`` spatial axes, the one a remap can restate."""
    return Grid.readable(cache_attribute) and cache_attribute.get_np_array("Direction").size == rank * rank


def _record_remap_geometry(cache_attribute: Attribute, remap: AxisRemap, source_spatial_shape: list[int]) -> None:
    """Record the header of the volume ``remap`` reindexes, every voxel kept on its physical point
    (SimpleITK's ``PermuteAxes`` and ``Flip``): output axis ``k`` takes the spacing and direction
    column of the axis it reads, negated where it reads it backwards, and the origin moves to the far
    end of each mirrored axis. A case with no header of that rank is left alone."""
    rank = len(remap)
    if not _has_geometry(cache_attribute, rank):
        return
    origin = cache_attribute.get_np_array("Origin")
    spacing = cache_attribute.get_np_array("Spacing")
    matrix = cache_attribute.get_np_array("Direction").reshape((rank, rank))
    new_origin, new_spacing, new_matrix = origin.copy(), spacing.copy(), matrix.copy()
    for axis, (source, mirrored) in enumerate(remap):  # array order; the geometry is in (x, y, z)
        target, read = rank - 1 - axis, rank - 1 - source
        new_spacing[target] = spacing[read]
        new_matrix[:, target] = -matrix[:, read] if mirrored else matrix[:, read]
        if mirrored:
            new_origin += matrix[:, read] * spacing[read] * (int(source_spatial_shape[source]) - 1)
    cache_attribute["Origin"] = new_origin
    cache_attribute["Spacing"] = new_spacing
    cache_attribute["Direction"] = new_matrix.reshape(-1)


def _pop_remap_geometry(cache_attribute: Attribute, rank: int) -> None:
    """Hand back the header :func:`_record_remap_geometry` recorded over."""
    if _has_geometry(cache_attribute, rank):
        for key in ("Origin", "Spacing", "Direction"):
            cache_attribute.pop(key)


def _refuse_axes_off_the_case(stage: str, axes: list[int], rank: int, example: str) -> None:
    """Refuse spatial ``axes`` (0-based) a case of ``rank`` spatial axes does not have, as ``Flip()`` and
    ``Permute()`` name three by default."""
    if axes and max(axes) >= rank:
        raise TransformError(
            f"'{stage}' names the spatial axes {'|'.join(map(str, axes))}, and the case has {rank}.",
            f'Name the axes of a {rank}-D case: dims: "{example}".',
        )


class Permute(TransformInverse):
    """Reorder the spatial axes: ``dims`` names the new order, ``|``-separated (``"1|0|2"``).

    The header follows the axes, so every voxel keeps its physical point (SimpleITK's ``PermuteAxes``).
    """

    working_multiple = 0.0

    locality = LocalityKind.ORIENTATION

    def __init__(self, dims: str = "1|0|2", inverse: bool = True) -> None:
        super().__init__(inverse)
        try:
            self.dims = [0] + [int(d) + 1 for d in str(dims).split("|")]
        except ValueError:
            raise TransformError(
                f"'Permute' cannot read dims={dims!r}.",
                "dims is the new spatial axis order as a '|'-separated string: dims: \"1|0|2\" (not a list).",
            ) from None

    def _remap(self) -> AxisRemap:
        # Output spatial axis k reads input axis ``self.dims[k + 1] - 1``, never mirrored.
        return [(d - 1, False) for d in self.dims[1:]]

    def _check(self, rank: int) -> None:
        axes = [d - 1 for d in self.dims[1:]]
        _refuse_axes_off_the_case("Permute", axes, rank, "|".join(map(str, reversed(range(rank)))))
        if len(axes) != rank:
            raise TransformError(
                f"'Permute' orders {len(axes)} spatial axes, and the case has {rank}.",
                f'Give every axis once: dims: "{"|".join(map(str, reversed(range(rank))))}".',
            )

    def transform_shape(self, group_src: str, name: str, shape: list[int], cache_attribute: Attribute) -> list[int]:
        self._check(len(shape))
        return remap_shape(shape, self._remap())

    def stream_region_source(
        self,
        name: str,
        target_slices: tuple[slice, ...],
        source_spatial_shape: list[int],
        cache_attribute: Attribute,
    ) -> list[slice]:
        return remap_region(target_slices, source_spatial_shape, self._remap())

    def inverse_transform_shape(self, shape: list[int], cache_attribute: Attribute) -> list[int]:
        return remap_shape(shape, invert_remap(self._remap()))

    def stream_region_target(
        self,
        name: str,
        target_slices: tuple[slice, ...],
        source_spatial_shape: list[int],
        cache_attribute: Attribute,
    ) -> list[slice]:
        # The write mirror pulls through the inverse remap.
        return remap_region(target_slices, source_spatial_shape, invert_remap(self._remap()))

    def write_stream_cache_attribute(
        self, cache_attribute: Attribute, source_spatial_shape: list[int], name: str = ""
    ) -> None:
        del name
        _record_remap_geometry(cache_attribute, self._remap(), source_spatial_shape)

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        self._check(tensor.dim() - 1)
        self.write_stream_cache_attribute(cache_attribute, list(tensor.shape[1:]), name)
        return tensor.permute(tuple(self.dims))

    def inverse(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        _pop_remap_geometry(cache_attribute, len(self.dims) - 1)
        return tensor.permute(tuple(np.argsort(self.dims)))


class Flip(TransformInverse):
    """Reverse the spatial axes ``dims`` names, ``|``-separated (``"0|2"``).

    The header follows the axes, so every voxel keeps its physical point (SimpleITK's ``Flip``).
    """

    working_multiple = 0.0

    locality = LocalityKind.ORIENTATION

    def __init__(self, dims: str = "1|0|2", inverse: bool = True) -> None:
        super().__init__(inverse)

        self.dims = [int(d) + 1 for d in str(dims).split("|")]

    def _remap(self, rank: int) -> AxisRemap:
        # The identity permutation, mirrored on the flipped axes.
        return [(k, (k + 1) in self.dims) for k in range(rank)]

    def stream_region_source(
        self,
        name: str,
        target_slices: tuple[slice, ...],
        source_spatial_shape: list[int],
        cache_attribute: Attribute,
    ) -> list[slice]:
        return remap_region(target_slices, source_spatial_shape, self._remap(len(target_slices)))

    def stream_region_target(
        self,
        name: str,
        target_slices: tuple[slice, ...],
        source_spatial_shape: list[int],
        cache_attribute: Attribute,
    ) -> list[slice]:
        # A flip is its own inverse: a written region pulls exactly the region the forward would read.
        return self.stream_region_source(name, target_slices, source_spatial_shape, cache_attribute)

    def write_stream_cache_attribute(
        self, cache_attribute: Attribute, source_spatial_shape: list[int], name: str = ""
    ) -> None:
        del name
        _record_remap_geometry(cache_attribute, self._remap(len(source_spatial_shape)), source_spatial_shape)

    def transform_shape(self, group_src: str, name: str, shape: list[int], cache_attribute: Attribute) -> list[int]:
        _refuse_axes_off_the_case("Flip", [d - 1 for d in self.dims], len(shape), "|".join(map(str, range(len(shape)))))
        return shape

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        self.transform_shape("", name, list(tensor.shape[1:]), cache_attribute)
        self.write_stream_cache_attribute(cache_attribute, list(tensor.shape[1:]), name)
        return tensor.flip(tuple(self.dims))

    def inverse(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        _pop_remap_geometry(cache_attribute, tensor.dim() - 1)
        return tensor.flip(tuple(self.dims))


class Canonical(TransformInverse):
    """Reorient a volume onto RAS, the orientation NIfTI is canonical in.

    The pipeline writes direction cosines in the SimpleITK convention, where the identity is LPS, so
    RAS is ``diag(-1, -1, 1)`` there and a volume already in LPS is reoriented by this stage.

    An orthogonal reorientation is a signed permutation of the axes, an exact index remap; only an
    oblique direction is resampled, in world space onto the grid it records, ``fill`` where the source
    does not reach (the corners the turn uncovers: ``-1024`` is air for a CT). The inverse reads a
    prediction back, whose values are not the source's, so it writes zero there. That grid takes its
    extents and spacing from the signed permutation nearest the oblique direction, so a slightly
    oblique sagittal case keeps its field of view. A remap that permutes axes transposes the extents
    it swaps, so ``transform_shape`` folds the patch grid onto the reoriented shape.

    ``resample: false`` never resamples: an oblique case too is permuted and flipped onto the axes
    nearest RAS, its direction left as oblique as it was (``D @ M``), its centre fixed: nibabel's
    ``as_closest_canonical``. An index remap
    always, so it always streams. On an orthogonal case the two are the same remap.
    """

    working_multiple = 3.0  # an oblique case: ITK's filter on the host; the device walk slabs itself against its budget

    def __init__(self, inverse: bool = True, fill: float = 0.0, resample: bool = True) -> None:
        super().__init__(inverse)
        self.fill_value = float(fill)
        self.resample = resample
        self.canonical_direction = torch.diag(torch.tensor([-1, -1, 1])).to(torch.double)

    def _reorientation(self, cache_attribute: Attribute) -> torch.Tensor:
        """The map taking an output coordinate onto the input it comes from, in (x, y, z): a voxel
        sits at ``D @ (spacing * index) + origin``, so the map is ``D^-1 @ C``."""
        initial_matrix = cache_attribute.get_tensor("Direction").reshape(3, 3).to(torch.double)
        return initial_matrix.inverse() @ self.canonical_direction

    def _orthogonal_remap(self, cache_attribute: Attribute) -> AxisRemap | None:
        """The exact index remap this case's reorientation is, ``None`` where it is not one: a case
        with no usable direction cosines and an oblique one both answer ``None``."""
        if "Direction" not in cache_attribute or cache_attribute.get_np_array("Direction").size != 9:
            return None
        if not self.resample:
            return self._grid_remap(cache_attribute)
        return signed_permutation(self._reorientation(cache_attribute), SIGNED_PERMUTATION_ATOL_FLOAT64)

    def _grid_remap(self, cache_attribute: Attribute) -> AxisRemap | None:
        """The remap the canonical grid takes its extents and spacing from: the exact one, or for an
        oblique case the nearest, so the grid it is resampled onto holds its field of view. ``None``
        where the case has no usable direction cosines."""
        if "Direction" not in cache_attribute or cache_attribute.get_np_array("Direction").size != 9:
            return None
        return Canonical._nearest_remap(self._reorientation(cache_attribute))

    @staticmethod
    def _nearest_remap(reorientation: torch.Tensor) -> AxisRemap:
        """The signed permutation carrying the most of the reorientation's weight, in the array order
        ``signed_permutation`` answers in, and equal to its answer where there is one."""
        linear = reorientation.numpy()
        rows = max(
            itertools.permutations(range(3)),
            key=lambda candidate: sum(abs(linear[row, column]) for column, row in enumerate(candidate)),
        )
        return [(2 - rows[column], bool(linear[rows[column], column] < 0)) for column in reversed(range(3))]

    @staticmethod
    def _nearest_matrix(reorientation: torch.Tensor) -> torch.Tensor:
        """The signed permutation :meth:`_nearest_remap` picks, as a matrix in ``(x, y, z)``."""
        remap = Canonical._nearest_remap(reorientation)
        matrix = torch.zeros((3, 3), dtype=torch.double)
        for array_column, (array_row, mirrored) in enumerate(remap):
            matrix[2 - array_row, 2 - array_column] = -1.0 if mirrored else 1.0
        return matrix

    @staticmethod
    def _carried(per_physical_axis: torch.Tensor, remap: list[tuple[int, bool]] | None) -> torch.Tensor:
        """Carry a per-physical-axis quantity along a remap: output axis c takes the axis it reads. A
        reorientation preserves the physical extent, not which axis carries it."""
        if remap is None:
            return per_physical_axis
        # The remap is in array order and these are (x, y, z): read in array order, gather, restore.
        return per_physical_axis.flip(0)[[source for source, _ in remap]].flip(0)

    @staticmethod
    def _half_extent(spatial_shape: list[int], spacing: torch.Tensor) -> torch.Tensor:
        """Half a grid's physical extent along each axis, in (x, y, z). A shape is in array order."""
        return torch.tensor(
            [(spatial_shape[-axis - 1] - 1) * spacing[axis] / 2 for axis in range(len(spatial_shape))],
            dtype=torch.double,
        )

    @staticmethod
    def _resample(tensor: torch.Tensor, target: Grid, source: Grid, fill: float) -> torch.Tensor:
        """``tensor``, on ``source``, read onto ``target`` in world space as ``sitk.Resample`` reads it:
        the dtype's interpolation, ``fill`` outside ``source``. Resample's samplers: ITK's filter on the
        host, the torch walk slab by slab elsewhere."""
        mode = default_interpolation(tensor)
        starts, extent = [0] * source.rank, list(source.size_zyx)
        if tensor.device.type == "cpu" and sitk is not None:
            resampled = _resample_with_sitk(tensor, target, source, (), starts, mode, fill)
            if resampled is not None:
                return resampled
        out = torch.empty((int(tensor.shape[0]), *target.size_zyx), dtype=tensor.dtype, device=tensor.device)
        rows_total = int(target.size_zyx[0])
        rows = walk_rows(target, (), tensor.device)
        for start in range(0, rows_total, rows):
            stop = min(rows_total, start + rows)
            coordinates = source_index_rows(target, source, (), tensor.device, start, stop)
            out[:, start:stop] = gather(tensor, coordinates, starts, extent, mode, fill)
        return out

    def transform_shape(self, group_src: str, name: str, shape: list[int], cache_attribute: Attribute) -> list[int]:
        # The patch grid is folded from what this returns.
        remap = self._grid_remap(cache_attribute)
        if remap is None:
            return shape
        return remap_shape(shape, remap)

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        # An orthogonal reorientation remaps indices, which ORIENTATION streams; an oblique one is
        # resampled from the whole volume.
        if self._orthogonal_remap(cache_attribute) is None:
            return PatchLocality(
                LocalityKind.WHOLE_VOLUME,
                reason="the case's direction cosines are oblique (or unreadable), so the"
                " reorientation is a resample of the whole volume rather than an index remap",
            )
        return PatchLocality(LocalityKind.ORIENTATION)

    def stream_region_source(
        self,
        name: str,
        target_slices: tuple[slice, ...],
        source_spatial_shape: list[int],
        cache_attribute: Attribute,
    ) -> list[slice]:
        remap = self._orthogonal_remap(cache_attribute)
        if remap is None:
            raise TransformError(
                "Canonical declared a region patch-locality for a direction it cannot remap exactly.",
                "Report this: patch_locality() and stream_region_source() disagree about the case.",
            )
        return remap_region(target_slices, source_spatial_shape, remap)

    def write_stream_cache_attribute(
        self, cache_attribute: Attribute, source_spatial_shape: list[int], name: str = ""
    ) -> None:
        # Nothing to state for a case this cannot reorient: no geometry, or a non-3-D direction.
        del name
        if not Grid.readable(cache_attribute) or cache_attribute.get_np_array("Direction").size != 9:
            return
        initial_matrix = cache_attribute.get_tensor("Direction").reshape(3, 3).to(torch.double)
        initial_origin = cache_attribute.get_tensor("Origin")
        spacing = cache_attribute.get_tensor("Spacing").to(torch.double)
        remap = self._grid_remap(cache_attribute)
        half_extent = Canonical._half_extent(source_spatial_shape, spacing)
        # Unresampled, the axes are the case's own, permuted and flipped: D @ M, oblique as it was.
        direction = (
            self.canonical_direction
            if self.resample
            else initial_matrix @ self._nearest_matrix(self._reorientation(cache_attribute))
        )
        cache_attribute["Direction"] = direction.flatten()
        cache_attribute["Spacing"] = Canonical._carried(spacing, remap)
        # The reorientation fixes the volume's centre, so the new origin is that centre stepped back
        # by the target grid's canonical half-extent. The extent is the volume's, never a patch's.
        center = initial_matrix @ half_extent + initial_origin
        cache_attribute["Origin"] = center - direction @ Canonical._carried(half_extent, remap)

    def _inverse_remap(self, cache_attribute: Attribute, grid: bool = False) -> AxisRemap | None:
        """The forward remap judged on the state ``inverse`` runs from, evaluated on a copy since a
        declaration never mutates the case: the exact one, or with ``grid`` the one the canonical grid
        took its extents from. ``None`` where there is none or the case has no direction."""
        scoped = Attribute(cache_attribute)
        if "Direction" not in scoped:
            return None
        scoped.pop("Direction")
        return self._grid_remap(scoped) if grid else self._orthogonal_remap(scoped)

    def inverse_patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        if self._inverse_remap(cache_attribute) is None:
            return PatchLocality(
                LocalityKind.WHOLE_VOLUME,
                reason="the direction this inverse restores is oblique (or not on the attribute),"
                " so the reorientation back is a resample of the whole volume",
            )
        return PatchLocality(LocalityKind.ORIENTATION)

    def inverse_transform_shape(self, shape: list[int], cache_attribute: Attribute) -> list[int]:
        # The inverse puts each extent back on the axis it came from.
        remap = self._inverse_remap(cache_attribute, grid=True)
        if remap is None:
            return shape
        return remap_shape(shape, invert_remap(remap))

    def stream_region_target(
        self,
        name: str,
        target_slices: tuple[slice, ...],
        source_spatial_shape: list[int],
        cache_attribute: Attribute,
    ) -> list[slice]:
        # A written region pulls through the inverse remap.
        remap = self._inverse_remap(cache_attribute)
        if remap is None:
            raise TransformError(
                "Canonical declared a region inverse patch-locality for a direction it cannot remap exactly.",
                "Report this: inverse_patch_locality() and stream_region_target() disagree about the case.",
            )
        return remap_region(target_slices, source_spatial_shape, invert_remap(remap))

    def _reorient(
        self,
        tensor: torch.Tensor,
        reorientation: torch.Tensor,
        onto: Attribute,
        off: Attribute,
        grid: AxisRemap,
        fill: float,
    ) -> torch.Tensor:
        """Apply a reorientation: an exact index remap where it is one, where it is not a resample from
        the grid ``off`` describes onto the one ``onto`` does, whose extents are the tensor's carried
        along ``grid``, ``fill`` where ``off`` does not reach. An orthogonal one reproduces the input's
        values bit for bit."""
        remap = signed_permutation(reorientation, SIGNED_PERMUTATION_ATOL_FLOAT64) if self.resample else grid
        if remap is None:
            shape = [int(extent) for extent in tensor.shape[1:]]
            return Canonical._resample(
                tensor,
                Grid.of(remap_shape(shape, grid), onto, "the reoriented case"),
                Grid.of(shape, off, "the case"),
                fill,
            )
        return apply_remap(tensor, remap)

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        # Read the source geometry before recording the canonical one over it: the attribute stacks.
        reorientation = self._reorientation(cache_attribute)
        source = Attribute(cache_attribute)
        self.write_stream_cache_attribute(cache_attribute, list(tensor.shape[1:]), name)
        return self._reorient(
            tensor, reorientation, cache_attribute, source, Canonical._nearest_remap(reorientation), self.fill_value
        )

    def inverse(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        # Popping restores the source geometry, which is what the inverse remap is then read from.
        held = Attribute(cache_attribute)
        cache_attribute.pop("Direction")
        cache_attribute.pop("Spacing")
        cache_attribute.pop("Origin")
        reorientation = self._reorientation(cache_attribute)
        back = invert_remap(Canonical._nearest_remap(reorientation))
        return self._reorient(tensor, reorientation.inverse(), cache_attribute, held, back, 0.0)


class Gradient(Transform):
    #: The one destination the differences are written into, one channel per spatial axis.
    working_multiple = 5.0

    # Each output voxel reads its immediate neighbour: a halo of radius 1.
    locality = LocalityKind.HALO
    halo = (1,)

    def __init__(self, per_dim: bool = False):
        super().__init__()
        self.per_dim = per_dim

    @staticmethod
    def _differences(image: torch.Tensor) -> torch.Tensor:
        """The first difference along each spatial axis, subtracted straight into its own slice of
        one destination, whose far edge is the zero it is created with."""
        rank = len(image.shape) - 1
        out = torch.zeros((image.shape[0], rank, *image.shape[1:]), dtype=image.dtype, device=image.device)
        for axis in range(rank):
            ahead, behind = [slice(None)] * (rank + 1), [slice(None)] * (rank + 1)
            ahead[1 + axis], behind[1 + axis] = slice(1, None), slice(None, -1)
            landing: list[Any] = [slice(None), axis, *[slice(None)] * rank]
            landing[2 + axis] = slice(None, -1)
            torch.sub(image[tuple(ahead)], image[tuple(behind)], out=out[tuple(landing)])
        return out

    def output_channels(self, channels: int) -> int:
        """One channel per spatial axis, per input channel, when the axes are kept separate. Three
        and not the case's own rank, which is not among the arguments."""
        return channels * 3 if self.per_dim else channels

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        # (C, rank, *spatial): the axis dimension is its own, so a multi-channel case takes its norm
        # across the axes.
        result = Gradient._differences(tensor)
        if self.per_dim:
            return result.flatten(0, 1)
        # In place where the dtype has the arithmetic: an integer difference takes the widening one.
        result = result.mul_(3).sigmoid_() if result.is_floating_point() else torch.sigmoid(result * 3)
        return result.norm(dim=1)


class Flatten(Transform):
    working_multiple = 0.0

    def __init__(self) -> None:
        super().__init__()

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        return PatchLocality(LocalityKind.WHOLE_VOLUME, reason=_RANK_CHANGE)

    def transform_shape(self, group_src: str, name: str, shape: list[int], cache_attribute: Attribute) -> list[int]:
        return [int(np.prod(np.asarray(shape)))]

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        return tensor.flatten()
