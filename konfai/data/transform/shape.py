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
from konfai.data.transform.base import LocalityKind, PatchLocality, RegionContext, Transform, TransformInverse
from konfai.utils.dataset import Attribute, Dataset
from konfai.utils.errors import TransformError


class Padding(TransformInverse):
    """Pad the volume's borders (``padding`` pairs in ``F.pad`` order: last axis first).

    A pad is a translation of the volume into a larger grid plus a border it fills, so it streams
    as a REGRID: a target region pulls the source rows it covers (clamped to the volume, kept wide
    enough for a reflection to read from) and the stage fills what the clamp cut. The origin moves
    with the volume, written once from the full extent (``write_stream_cache_attribute``).
    """

    # Measured at 0.00 on the CUDA allocator: it holds nothing beyond what it is handed.
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
        if "Origin" in cache_attribute and "Spacing" in cache_attribute and "Direction" in cache_attribute:
            origin = torch.tensor(cache_attribute.get_np_array("Origin"))
            matrix = torch.tensor(cache_attribute.get_np_array("Direction").reshape((len(origin), len(origin))))
            origin = torch.matmul(origin, matrix)
            for dim in range(len(self.padding) // 2):
                origin[dim] -= self.padding[dim * 2] * cache_attribute.get_np_array("Spacing")[dim]
            cache_attribute["Origin"] = torch.matmul(origin, torch.inverse(matrix))

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
        pull: list[slice] = []
        for target, (before, _after), extent in zip(
            target_slices, self._pairs(len(target_slices)), source_spatial_shape, strict=False
        ):
            low, high = max(0, target.start - before), min(extent, target.stop - before)
            if target.start < before:  # reaches the low border: keep the rows a reflection reads back from
                low, high = 0, max(high, min(extent, before - target.start + 1))
            if target.stop > extent + before:  # reaches the high border
                high, low = extent, min(low, max(0, extent - (target.stop - extent - before) - 1))
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
        # The inverse drops the padded border and keeps a translated copy of what remains: a CROP.
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
        # Output index o holds input index o + pad_before: a written region pulls its own slices stepped
        # forward by the leading pad.
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
    # Measured at 0.00 on the CUDA allocator: it holds nothing beyond what it is handed.
    working_multiple = 0.0

    def __init__(self, dim: int, inverse: bool = True) -> None:
        super().__init__(inverse)
        self.dim = dim

    # WHOLE_VOLUME on purpose: squeeze/unsqueeze changes the tensor rank, and the streamed write sizes
    # each slab from the pre-finalize accumulator grid: a rank change past it cannot region-stream.

    def transform_shape(self, group_src: str, name: str, shape: list[int], cache_attribute: Attribute) -> list[int]:
        # ``shape`` is the channel-stripped spatial shape (patching strips [C, *spatial] before folding),
        # so the runtime tensor is [C, *shape] and ``self.dim`` indexes into that. Squeezing the channel
        # (axis 0) leaves the spatial grid untouched; squeezing a spatial axis drops it from the grid --
        # but only when it is size 1, exactly as ``torch.squeeze`` does (a non-singleton axis is a no-op).
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
    margins; cropping is then the translation ``out[o] = volume[o + start]``, so a target patch reads
    its shifted source region. Dropped voxels mean the stored volume's statistics are not the output's
    (hence ``LocalityKind.CROP``).
    """

    # Measured at 0.00 on the CUDA allocator: it holds nothing beyond what it is handed.
    working_multiple = 0.0

    def __init__(self, inverse: bool = True) -> None:
        super().__init__(inverse)

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        # Total: the box is a fact ``transform_shape`` puts on the case before the dispatcher reads any
        # declaration, but a group carries only what its writer stored, and without it there is no
        # translation to make: only the read that would find one.
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
        # Output index o holds source index o + start, so the region behind a target patch is that
        # patch's own slices stepped forward by the box's near margin.
        box = Crop._parse_box(cache_attribute["box"])
        return [
            slice(target.start + int(start), target.stop + int(start))
            for target, (start, _) in zip(target_slices, box, strict=False)
        ]

    def write_stream_cache_attribute(
        self, cache_attribute: Attribute, source_spatial_shape: list[int], name: str = ""
    ) -> None:
        del name
        if "box" not in cache_attribute:
            return
        if not {"Origin", "Spacing", "Direction"} <= set(cache_attribute.keys()):
            return
        # The crop keeps the box's near corner, so the new origin is the physical point that corner
        # already sat on. A margin is in array order and the geometry is in (x, y, z), hence the
        # reversed indexing.
        box = Crop._parse_box(cache_attribute["box"])
        origin = torch.tensor(cache_attribute.get_np_array("Origin"))
        matrix = torch.tensor(cache_attribute.get_np_array("Direction").reshape((len(origin), len(origin))))
        origin = torch.matmul(origin, matrix)
        for dim in range(box.shape[0]):
            origin[-dim - 1] += box[dim][0] * cache_attribute.get_np_array("Spacing")[-dim - 1]
        cache_attribute["Origin"] = torch.matmul(origin, torch.inverse(matrix))

    def transform_shape(self, group_src: str, name: str, shape: list[int], cache_attribute: Attribute) -> list[int]:
        # The crop box is content-dependent (foreground bounding box), so the output shape
        # cannot be known without the pixel data. If the box was already computed and persisted
        # as a sidecar attribute, reuse it and skip the read; otherwise compute it once from the
        # volume. (A fully-lazy variant would require deferring patch planning past _load().)
        # ``shape`` is already the channel-stripped spatial shape (patching strips [C, *spatial]
        # before calling transform_shape), so the crop box (one row per spatial axis) aligns with
        # ``shape`` directly, exactly like ``__call__`` aligns it with ``tensor.shape[1:]``.
        if "box" in cache_attribute:
            box = self._parse_box(cache_attribute["box"])
            return [int(s - a - b) for (a, b), s in zip(box, shape, strict=False)]
        source = next((dataset for dataset in self.datasets if dataset.is_dataset_exist(group_src, name)), None)
        if source is None:
            return shape
        box = self._foreground_box(source, group_src, name)
        for i, ((_, b), s) in enumerate(zip(box, shape, strict=False)):
            # The scan reports the LAST foreground index; the box carries the margin AFTER it, so
            # the far margin is what lies past that row. Off by one, the crop cut it off.
            box[i][1] = max(int(s - b - 1), 0)
        cache_attribute["box"] = box
        return [int(s - a - b) for (a, b), s in zip(box, shape, strict=False)]

    @staticmethod
    def _foreground_box(source: Dataset, group_src: str, name: str) -> np.ndarray:
        """The bounding box (first and last index per spatial axis) of the voxels above the 5th
        percentile, from bounded passes over the stored volume: the threshold by a quantile scan,
        the box by one more pass. Memoised on the dataset: every chain of the case asks for the
        same box, and it is the same volume."""
        memo = source.case_facts.setdefault((group_src, name), {})
        if "box" not in memo:
            threshold = source.read_data_quantile(group_src, name, 0.05)
            first: np.ndarray | None = None
            last: np.ndarray | None = None
            offset = 0
            for block in source.iter_data_blocks(group_src, name)():
                foreground = np.any(block > threshold, axis=0)  # every channel votes, as the vector image did
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
        # The box carries the FAR margin, so the stop it crops at is the one the extent in hand decides.
        crops = [
            slice(int(near), extent - int(far)) for (near, far), extent in zip(box, tensor.shape[1:], strict=False)
        ]
        return tensor[(slice(None), *crops)]

    def inverse(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        if "box" not in cache_attribute:
            return tensor
        box = self._parse_box(cache_attribute.pop("box"))
        cache_attribute.pop_np_array("Origin")
        padding = []
        for b in reversed(box):
            padding.extend([b[0], b[1]])
        result = F.pad(tensor.unsqueeze(0), tuple(padding), "replicate").squeeze(0)
        return result


class Permute(TransformInverse):
    """Reorder the spatial axes: ``dims`` names the new order, ``|``-separated (``"1|0|2"``)."""

    # Measured at 0.00 on the CUDA allocator: it holds nothing beyond what it is handed.
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
        # Output spatial axis k reads input axis ``self.dims[k + 1] - 1`` (self.dims is
        # channel-inclusive), never mirrored.
        return [(d - 1, False) for d in self.dims[1:]]

    def transform_shape(self, group_src: str, name: str, shape: list[int], cache_attribute: Attribute) -> list[int]:
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
        # The write mirror pulls through the inverse remap: input axis k carries the output axis
        # that read it.
        return remap_region(target_slices, source_spatial_shape, invert_remap(self._remap()))

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        return tensor.permute(tuple(self.dims))

    def inverse(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        return tensor.permute(tuple(np.argsort(self.dims)))


class Flip(TransformInverse):
    # Measured at 0.00 on the CUDA allocator: it holds nothing beyond what it is handed.
    working_multiple = 0.0

    locality = LocalityKind.ORIENTATION

    def __init__(self, dims: str = "1|0|2", inverse: bool = True) -> None:
        super().__init__(inverse)

        self.dims = [int(d) + 1 for d in str(dims).split("|")]

    def stream_region_source(
        self,
        name: str,
        target_slices: tuple[slice, ...],
        source_spatial_shape: list[int],
        cache_attribute: Attribute,
    ) -> list[slice]:
        # A mirror moves no axis: the remap is the identity permutation, mirrored on the flipped axes.
        remap: AxisRemap = [(k, (k + 1) in self.dims) for k in range(len(target_slices))]
        return remap_region(target_slices, source_spatial_shape, remap)

    def stream_region_target(
        self,
        name: str,
        target_slices: tuple[slice, ...],
        source_spatial_shape: list[int],
        cache_attribute: Attribute,
    ) -> list[slice]:
        # A flip is its own inverse: a written region pulls exactly the region the forward would read.
        return self.stream_region_source(name, target_slices, source_spatial_shape, cache_attribute)

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        return tensor.flip(tuple(self.dims))

    def inverse(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        return tensor.flip(tuple(self.dims))


class Canonical(TransformInverse):
    """Reorient a volume onto RAS, the orientation NIfTI is canonical in.

    The pipeline writes direction cosines in the SimpleITK convention, where the identity is LPS, so
    RAS is ``diag(-1, -1, 1)`` there: a volume already in LPS IS reoriented by this stage. That is
    the same target as nibabel's ``as_closest_canonical``, which is what the NIfTI tooling around a
    dataset expects.

    An orthogonal reorientation is a signed permutation of the axes: an exact index remap (values only
    change place, so whole-volume statistics survive); only an oblique direction is resampled. A remap
    that permutes axes transposes the extents it swaps, so ``transform_shape`` folds the patch grid
    onto the reoriented shape.
    """

    working_multiple = 3.0  # an oblique case is resampled: the resample's own figure

    def __init__(self, inverse: bool = True) -> None:
        super().__init__(inverse)
        self.canonical_direction = torch.diag(torch.tensor([-1, -1, 1])).to(torch.double)

    def _reorientation(self, cache_attribute: Attribute) -> torch.Tensor:
        """The map taking an output coordinate onto the input it comes from, in (x, y, z).

        A voxel sits at ``D @ (spacing * index) + origin``, so the map is ``D^-1 @ C`` (with the
        target spacing carried along the permutation, see ``_carried``): NOT the rotation
        ``C @ D^-1``, which only agrees where the two commute.
        """
        initial_matrix = cache_attribute.get_tensor("Direction").reshape(3, 3).to(torch.double)
        return initial_matrix.inverse() @ self.canonical_direction

    def _orthogonal_remap(self, cache_attribute: Attribute) -> AxisRemap | None:
        """The exact index remap this case's reorientation is, or ``None`` where it is not one.

        Total: a case whose header carries no usable direction cosines has no remap to make, and an
        oblique one has none to make either, both answer ``None`` rather than raise, and the resample
        is what answers for them.
        """
        if "Direction" not in cache_attribute or cache_attribute.get_np_array("Direction").size != 9:
            return None
        return signed_permutation(self._reorientation(cache_attribute), SIGNED_PERMUTATION_ATOL_FLOAT64)

    @staticmethod
    def _carried(per_physical_axis: torch.Tensor, remap: list[tuple[int, bool]] | None) -> torch.Tensor:
        """Carry a per-physical-axis quantity along a remap: output axis c takes the axis it reads.

        A spacing and a half-extent travel with the axis they belong to: what a reorientation
        preserves is the volume's physical extent, not which axis carries it. An oblique direction is
        resampled onto the input's own grid, so without a remap nothing moves.
        """
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
    def _affine_matrix(matrix: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            (
                torch.cat((matrix, translation.unsqueeze(0).T), dim=1),
                torch.tensor([[0, 0, 0, 1]]),
            ),
            dim=0,
        )

    @staticmethod
    def _resample_affine(data: torch.Tensor, matrix: torch.Tensor):
        if data.dtype == torch.uint8:
            mode = "nearest"
        else:
            mode = "bilinear"
        # Sample in the data's own device and float dtype: the model output is float16 on the GPU, and
        # affine_grid/grid_sample support float16 on CPU and CUDA. Building the grid on the data's device
        # (instead of a CPU float32 grid) keeps the whole reorientation on-device: no host round-trip and
        # no float32 upcast of the (channels x volume) tensor. Integer inputs still need a float grid.
        # Accepted trade-off: an fp16 grid quantizes the sampling coordinates (up to ~0.1 voxel at 512^3),
        # chosen over the ~2x transient memory of a float32 grid + volume upcast.
        work = data if data.is_floating_point() else data.type(torch.float32)
        grid = torch.nn.functional.affine_grid(
            matrix[:, :-1, ...].to(device=work.device, dtype=work.dtype),
            [1, *list(data.shape)],
            align_corners=True,
        )
        return (
            torch.nn.functional.grid_sample(
                work.unsqueeze(0),
                grid,
                align_corners=True,
                mode=mode,
                padding_mode="reflection",
            )
            .squeeze(0)
            .type(data.dtype)
        )

    def transform_shape(self, group_src: str, name: str, shape: list[int], cache_attribute: Attribute) -> list[int]:
        # ``shape`` is the channel-stripped SPATIAL shape, and the patch grid is folded from what this
        # returns: a remap that transposes extents moves the grid onto the reoriented volume.
        remap = self._orthogonal_remap(cache_attribute)
        if remap is None:
            return shape
        return remap_shape(shape, remap)

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        # Only the case can say which reorientation this is, so only the header can answer. An orthogonal
        # one (mirroring or permuting) remaps indices, which is what ORIENTATION streams; an oblique
        # one is resampled from the whole volume.
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
        # Nothing to state for a case this cannot reorient: no geometry, or a direction that is not
        # 3-D. Its __call__ fails loudly before reaching here; a landing fold must not fail for it.
        del name
        if not Grid.readable(cache_attribute) or cache_attribute.get_np_array("Direction").size != 9:
            return
        initial_matrix = cache_attribute.get_tensor("Direction").reshape(3, 3).to(torch.double)
        initial_origin = cache_attribute.get_tensor("Origin")
        spacing = cache_attribute.get_tensor("Spacing").to(torch.double)
        remap = self._orthogonal_remap(cache_attribute)
        half_extent = Canonical._half_extent(source_spatial_shape, spacing)
        cache_attribute["Direction"] = self.canonical_direction.flatten()
        cache_attribute["Spacing"] = Canonical._carried(spacing, remap)
        # The reorientation fixes the volume's centre, so the new origin is that centre stepped back by
        # the canonical half-extent: the TARGET grid's, which a permutation has carried onto other
        # axes. The extent is the VOLUME's, never a patch's: it is an argument rather than the handed
        # tensor's shape.
        center = initial_matrix @ half_extent + initial_origin
        cache_attribute["Origin"] = center - self.canonical_direction @ Canonical._carried(half_extent, remap)

    def _inverse_remap(self, cache_attribute: Attribute) -> AxisRemap | None:
        """The forward remap judged on the state ``inverse`` runs from: the popped-to source direction.

        The inverse pops the canonical geometry and reorients back through the SOURCE direction under
        it, so its streamability is the popped state's: evaluated on a copy, since a declaration
        never mutates the case. A matrix and its inverse are signed permutations together, so the
        forward remap answers for both; ``None`` where the case is oblique or carries no direction.
        """
        scoped = Attribute(cache_attribute)
        if "Direction" not in scoped:
            return None
        scoped.pop("Direction")
        return self._orthogonal_remap(scoped)

    def inverse_patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        if self._inverse_remap(cache_attribute) is None:
            return PatchLocality(
                LocalityKind.WHOLE_VOLUME,
                reason="the direction this inverse restores is oblique (or not on the attribute),"
                " so the reorientation back is a resample of the whole volume",
            )
        return PatchLocality(LocalityKind.ORIENTATION)

    def inverse_transform_shape(self, shape: list[int], cache_attribute: Attribute) -> list[int]:
        # transform_shape reads target axis k's extent from source axis ``source``; the inverse puts
        # each extent back on the axis it came from.
        remap = self._inverse_remap(cache_attribute)
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
        # Canonical axis k holds source axis ``source``'s content: a written region pulls through
        # the inverse remap, per input axis, the slice of the output axis it carries.
        remap = self._inverse_remap(cache_attribute)
        if remap is None:
            raise TransformError(
                "Canonical declared a region inverse patch-locality for a direction it cannot remap exactly.",
                "Report this: inverse_patch_locality() and stream_region_target() disagree about the case.",
            )
        return remap_region(target_slices, source_spatial_shape, invert_remap(remap))

    def _reorient(self, tensor: torch.Tensor, reorientation: torch.Tensor) -> torch.Tensor:
        """Apply a reorientation: an exact index remap where it is one, a resample where it is not.

        An orthogonal reorientation is a bijection on the voxels, so it must reproduce the input's
        multiset bit for bit, which only a permute and a flip do.
        """
        remap = signed_permutation(reorientation, SIGNED_PERMUTATION_ATOL_FLOAT64)
        if remap is None:
            matrix = Canonical._affine_matrix(reorientation, torch.tensor([0, 0, 0]))
            return Canonical._resample_affine(tensor, matrix.unsqueeze(0))
        return apply_remap(tensor, remap)

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        # Read the source geometry before recording the canonical one over it: the attribute stacks.
        reorientation = self._reorientation(cache_attribute)
        self.write_stream_cache_attribute(cache_attribute, list(tensor.shape[1:]), name)
        return self._reorient(tensor, reorientation)

    def inverse(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        # Popping restores the source geometry, which is what the inverse remap is then read from.
        cache_attribute.pop("Direction")
        cache_attribute.pop("Spacing")
        cache_attribute.pop("Origin")
        return self._reorient(tensor, self._reorientation(cache_attribute).inverse())


class Gradient(Transform):
    #: The one destination the differences are written into, one channel per spatial axis. Measured
    #: as VmHWM around the call on a 256^3 float32 block, beyond what it is handed and what it
    #: returns: 3.0 flattened (where the destination is working memory) and 0.0 per_dim (where the
    #: destination IS the output).
    working_multiple = 5.0

    # First-difference gradient: each output voxel reads its immediate neighbour, a HALO of radius
    # 1. The far-edge ConstantPad reproduces the whole-volume border once the halo clamps there.
    locality = LocalityKind.HALO
    halo = (1,)

    def __init__(self, per_dim: bool = False):
        super().__init__()
        self.per_dim = per_dim

    @staticmethod
    def _differences(image: torch.Tensor) -> torch.Tensor:
        """The first difference along each spatial axis, written where it belongs.

        One destination, not three differences plus three copies of them padded plus a stack of
        those: the far edge is the zero the destination is created with, and each difference is
        subtracted straight into its own slice of it.
        """
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
        """One channel per spatial axis, per input channel, when the axes are kept separate.

        Three and not the case's own rank, because the rank is not among this method's arguments and
        over-stating it costs a shorter region where under-stating it costs the run: a 2-D case is
        charged half a channel more than it takes.
        """
        return channels * 3 if self.per_dim else channels

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        # (C, rank, *spatial): the axis dimension is its OWN, never folded into the channels by a
        # squeeze. Squeezing dropped it only when the case had one channel, so a multi-channel case
        # took its norm across the channels instead of across the axes and handed the writer a
        # rank-5 array (measured on a 3-channel case: [1, 3, 128, 256, 256], refused by OME-Zarr).
        result = Gradient._differences(tensor)
        if self.per_dim:
            return result.flatten(0, 1)
        # In place where the dtype has the arithmetic: an integer difference has no sigmoid of
        # its own and takes the widening one, exactly as it did.
        result = result.mul_(3).sigmoid_() if result.is_floating_point() else torch.sigmoid(result * 3)
        return result.norm(dim=1)


class Flatten(Transform):
    # Measured at 0.00 on the CUDA allocator: it holds nothing beyond what it is handed.
    working_multiple = 0.0

    def __init__(self) -> None:
        super().__init__()

    def transform_shape(self, group_src: str, name: str, shape: list[int], cache_attribute: Attribute) -> list[int]:
        return [np.prod(np.asarray(shape))]

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        return tensor.flatten()
