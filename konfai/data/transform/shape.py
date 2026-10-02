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
from typing import Any

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
        pads: list[tuple[int, int]] = []
        crops: list[slice] = []
        for target, source, (before, _after) in zip(
            context.target, context.source, self._pairs(len(context.target)), strict=True
        ):
            start, stop = source.start + before, source.stop + before  # where the block sits in the output
            low, high = max(0, start - target.start), max(0, target.stop - stop)
            pads.append((low, high))
            crops.append(slice(target.start - (start - low), target.stop - (start - low)))
        mode, value = self._mode()
        padded = F.pad(tensor.unsqueeze(0), tuple(x for pair in reversed(pads) for x in pair), mode, value).squeeze(0)
        return padded[(slice(None), *crops)]

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
    """Crop a volume to the bounding box of its foreground.

    The content-dependent box is computed once (``transform_shape``) and kept on the case as ``box``
    margins; cropping is then the translation ``out[o] = volume[o + start]``. Dropped voxels make it
    a ``LocalityKind.CROP``, the stored volume's statistics not being the output's. The box is
    measured on the stored volume, so an input a stage before moved onto another grid is refused.

    Spacing and direction are kept and the origin moves to the box's near corner,
    ``O + D (start * spacing)`` (ITK's ``RegionOfInterest``); the inverse restores it.
    """

    working_multiple = 0.0

    def __init__(self, inverse: bool = True) -> None:
        super().__init__(inverse)

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        # ``transform_shape`` puts the box on the case, but a group carries only what its writer stored.
        if "box" not in cache_attribute:
            return PatchLocality(
                LocalityKind.WHOLE_VOLUME,
                reason="the case carries no 'box' yet; the foreground box is computed and recorded"
                " as the chain is planned, and only a read can find it",
            )
        return PatchLocality(LocalityKind.CROP)

    def stream_region_source(
        self,
        name: str,
        target_slices: tuple[slice, ...],
        source_spatial_shape: list[int],
        cache_attribute: Attribute,
    ) -> list[slice]:
        # Output index o holds source index o + start.
        box = Crop._parse_box(cache_attribute["box"])
        return [
            slice(target.start + int(start), target.stop + int(start))
            for target, (start, _) in zip(target_slices, box, strict=False)
        ]

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
        if "box" in cache_attribute:
            box = self._parse_box(cache_attribute["box"])
            return [int(s - a - b) for (a, b), s in zip(box, shape, strict=False)]
        source = self.dataset_holding(group_src, name)
        if source is None:
            return shape
        Crop._require_the_stored_grid(source, group_src, name, shape, cache_attribute)
        box = self._foreground_box(source, group_src, name)
        for i, ((_, b), s) in enumerate(zip(box, shape, strict=False)):
            # The scan reports the last foreground index; the box carries the margin after it.
            box[i][1] = max(int(s - b - 1), 0)
        cache_attribute["box"] = box
        return [int(s - a - b) for (a, b), s in zip(box, shape, strict=False)]

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
            " or crop the written volume in a second TRANSFORM run.",
        )

    @staticmethod
    def _foreground_box(source: Dataset, group_src: str, name: str) -> np.ndarray:
        """The bounding box of the voxels above the 5th percentile, first and last index per spatial
        axis, from bounded passes over the stored volume. Memoised on the dataset."""
        memo = source.case_facts.setdefault((group_src, name), {})
        if "box" not in memo:
            threshold = source.read_data_quantile(group_src, name, 0.05)
            first: np.ndarray | None = None
            last: np.ndarray | None = None
            offset = 0
            for block in source.iter_data_blocks(group_src, name)():
                foreground = np.any(block > threshold, axis=0)  # every channel votes
                if foreground.any():
                    axes_first, axes_last = [], []
                    for axis in range(foreground.ndim):
                        hits = np.flatnonzero(
                            np.any(foreground, axis=tuple(o for o in range(foreground.ndim) if o != axis))
                        )
                        axes_first.append(int(hits[0]) + (offset if axis == 0 else 0))
                        axes_last.append(int(hits[-1]) + (offset if axis == 0 else 0))
                    first = np.asarray(axes_first) if first is None else np.minimum(first, axes_first)
                    last = np.asarray(axes_last) if last is None else np.maximum(last, axes_last)
                offset += int(block.shape[1])
            if first is None or last is None:
                raise TransformError(
                    f"'Crop' found no foreground in '{group_src}/{name}': every voxel is at or under its 5th percentile.",
                    "The volume is constant; drop the Crop for this case or crop it to a mask instead.",
                )
            memo["box"] = np.stack([first, last], axis=1).astype(np.int64)
        return np.array(memo["box"], copy=True)

    @staticmethod
    def _parse_box(box_str: str) -> np.ndarray:
        flat = np.fromstring(box_str.replace("[", " ").replace("]", " "), sep=" ", dtype=np.int64)
        return flat.reshape(-1, 2)

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        if "box" not in cache_attribute:
            return tensor
        box = self._parse_box(cache_attribute["box"])
        self.write_stream_cache_attribute(cache_attribute, list(tensor.shape[1:]), name)
        # The box carries the far margin, so the extent in hand decides the stop it crops at.
        crops = [
            slice(int(near), extent - int(far)) for (near, far), extent in zip(box, tensor.shape[1:], strict=False)
        ]
        return tensor[(slice(None), *crops)]

    def inverse(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        if "box" not in cache_attribute:
            return tensor
        box = self._parse_box(cache_attribute.pop("box"))
        if Grid.readable(cache_attribute):  # the forward pushed an origin only on a case with a geometry
            cache_attribute.pop("Origin")
        padding = []
        for b in reversed(box):
            padding.extend([b[0], b[1]])
        result = F.pad(tensor.unsqueeze(0), tuple(padding), "replicate").squeeze(0)
        return result


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
    """

    working_multiple = 3.0  # an oblique case: ITK's filter on the host; the device walk slabs itself against its budget

    def __init__(self, inverse: bool = True, fill: float = 0.0) -> None:
        super().__init__(inverse)
        self.fill_value = float(fill)
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
        cache_attribute["Direction"] = self.canonical_direction.flatten()
        cache_attribute["Spacing"] = Canonical._carried(spacing, remap)
        # The reorientation fixes the volume's centre, so the new origin is that centre stepped back
        # by the target grid's canonical half-extent. The extent is the volume's, never a patch's.
        center = initial_matrix @ half_extent + initial_origin
        cache_attribute["Origin"] = center - self.canonical_direction @ Canonical._carried(half_extent, remap)

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
        remap = signed_permutation(reorientation, SIGNED_PERMUTATION_ATOL_FLOAT64)
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
