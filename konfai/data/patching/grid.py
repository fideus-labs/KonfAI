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


"""The patch grid of a case and of a model input."""

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from typing import Any, NamedTuple

import numpy as np
import torch
import torch.nn.functional as F

from konfai.data.patching.blend import PathCombine, blend_axes, blend_overlap
from konfai.data.patching.stage import PatchReadPlan, _HaloPull
from konfai.utils.config import apply_config, config
from konfai.utils.utils import (
    OverlapSpec,
    best_sweep_axis,
    concretize_patch_size,
    free_axis_rounding,
    get_module,
    get_patch_slices_from_shape,
)


class _PatchGrid(NamedTuple):
    """One copy's cut: the axis the patches are ordered by, and the patches themselves."""

    sweep_axis: int
    slices: list[tuple[slice, ...]]


class Patch(ABC):
    """Abstract base class for dataset-level and model-level patch definitions."""

    @abstractmethod
    def __init__(
        self,
        patch_size: list[int],
        overlap: OverlapSpec = None,
        pad_value: float | None = 0,
        extend_slice: int = 0,
    ) -> None:
        if extend_slice != 0 and patch_size is not None and patch_size[0] != 1:
            raise ValueError(
                "`extend_slice` can only be used when patch_size[0] == 1 "
                f"(got patch_size[0]={patch_size[0]}, extend_slice={extend_slice})"
            )
        self.patch_size = patch_size
        self.overlap = overlap
        if isinstance(self.overlap, int):
            if self.overlap < 0:
                self.overlap = None
        #: The spatial shape each copy's grid is cut on, recorded by ``load``.
        self._shapes: dict[int, list[int]] = {}
        #: The cut itself, made on first use and left out of the pickle (``__getstate__``).
        self._grids: dict[int, _PatchGrid] = {}
        self.pad_value = pad_value
        self.extend_slice = extend_slice
        # Models need every patch at the declared size, so the last patch of an axis is padded up to it.
        # A consumer that REDUCES patches instead (streamed evaluation) must see only in-volume voxels:
        # padded ones would pollute its running sums, so it turns this off and takes the cropped patch.
        self.pad_to_patch = True
        #: Voxels of context read past each face of a grid patch, clamped to the volume, for a consumer
        #: that reduces patches but scores through a window (a metric's halo). The grid keeps its
        #: disjoint slots (``get_patch_slices``); ``core_in_read`` says where a slot sits in its read.
        #: Unpadded only (``pad_to_patch`` False): a model input padded to the patch has no core.
        self.halo = 0
        # The model's per-axis downsampling factor a FREE (``0``) axis rounds up to, set before the grids
        # are cut so each case's whole-axis extent lands on a valid model input. ``None`` outside a model
        # (evaluation) or for a network that never downsamples.
        self.free_axis_multiple: list[int] | None = None
        # Whether a free (``0``) axis was DECLARED. Captured now because the OOM re-plan later pins
        # ``patch_size`` to a concrete size in place, erasing the ``0`` the overlap default keys on.
        self._declared_free_axis: bool = (
            patch_size is not None and any(p == 0 for p in patch_size) and not all(p == 0 for p in patch_size)
        )

    def load(self, shape: list[int], a: int = 0) -> None:
        """Record the spatial shape copy ``a``'s grid is cut on; the cut itself waits for a consumer."""
        self._shapes[a] = list(shape)
        self._grids.pop(a, None)

    def _grid(self, a: int) -> _PatchGrid:
        """Copy ``a``'s cut, made once. A pure function of the recorded shape and this patch's own
        configuration, so a rank rebuilds it instead of unpickling it (``__getstate__``)."""
        grid = self._grids.get(a)
        if grid is None:
            shape = self._shapes[a]
            # The grid decides its own sweep axis and the reassembly reads it back (get_sweep_axis): one
            # source of truth, because a grid emitted for one axis and reassembled along another hands out
            # regions that are not final, with nothing to report it.
            sweep_axis = best_sweep_axis(concretize_patch_size(self.patch_size, shape, self.free_axis_multiple), shape)
            slices = get_patch_slices_from_shape(
                self.patch_size,
                shape,
                self.overlap,
                self.free_axis_multiple,
                self._declared_free_axis,
                sweep_axis,
            )
            grid = self._grids[a] = _PatchGrid(sweep_axis, slices)
        return grid

    def __getstate__(self) -> dict[str, Any]:
        """The cuts stay out of the pickle: ``mp.spawn`` pickles the configured object once per
        rank, and every manager's per-case grids dominated its bytes. Only the shapes travel, and
        each rank cuts the same grids back from them on first use.

        Cutting costs more than unpickling would have (100 cases of 2048 patches: 31 ms against 81
        ms in process); it is the transfer, paid once per rank, that dominates. On a 2-rank CPU
        spawn of 1000 such cases the payload goes from 35.2 to 0.44 MiB and the second rank holds
        every grid 3.0 s after the launcher's stamp instead of 4.8 s.
        """
        return {**self.__dict__, "_grids": {}}

    def get_sweep_axis(self, a: int = 0) -> int:
        """The axis this grid is ordered by, and so the one reassembly must slide along."""
        return self._grid(a).sweep_axis

    @abstractmethod
    def init(self, key: str):
        pass

    def get_patch_slices(self, a: int = 0) -> list[tuple[slice, ...]]:
        return self._grid(a).slices

    def read_slices(self, a: int, index: int, shape: Sequence[int]) -> list[slice]:
        """The region patch ``index`` of copy ``a`` reads: its grid slot, widened by the halo and
        clamped to the volume (``shape`` may carry leading non-spatial axes)."""
        slot = self._grid(a).slices[index]
        if not self.halo:
            return list(slot)
        spatial = [int(extent) for extent in shape[len(shape) - len(slot) :]]
        return _HaloPull([self.halo] * len(slot), spatial)(slot)

    def core_in_read(self, a: int, index: int) -> tuple[slice, ...]:
        """Where patch ``index``'s grid slot sits within its read: the halo in from each face, less
        where the volume's own face cut the halo short."""
        return tuple(
            slice(min(self.halo, s.start), min(self.halo, s.start) + s.stop - s.start)
            for s in self._grid(a).slices[index]
        )

    def get_read_plan(
        self, data_shape: list[int] | tuple[int, ...], index: int, a: int, is_input: bool
    ) -> PatchReadPlan:
        slot = self.read_slices(a, index, data_shape)
        slices_pre = [slice(None) for _ in data_shape[: -len(slot)]]
        extend_slice = self.extend_slice if is_input else 0

        bottom = extend_slice // 2
        top = int(np.ceil(extend_slice / 2))
        s = slice(
            (slot[0].start - bottom if slot[0].start - bottom >= 0 else 0),
            (slot[0].stop + top if slot[0].stop + top <= data_shape[len(slices_pre)] else data_shape[len(slices_pre)]),
        )
        slices = [s, *slot[1:]]
        reflect_padding = [0 for _ in range((len(slices) - 1) * 2)] + [0, 0]
        if extend_slice > 0 and (s.stop - s.start) < bottom + top + 1:
            if slot[0].start - bottom < 0:
                reflect_padding[-2] = bottom - slot[0].start
            if slot[0].stop + top > data_shape[len(slices_pre)]:
                reflect_padding[-1] = slot[0].stop + top - data_shape[len(slices_pre)]

        constant_padding = []
        if self.pad_to_patch and self.patch_size is not None:
            nspatial = len(slices)
            for dim_it, _slice in enumerate(reversed(slices)):
                axis = nspatial - 1 - dim_it
                extent = data_shape[-dim_it - 1]
                declared = self.patch_size[axis]
                if declared != 0:
                    target = declared
                else:
                    # A FREE axis pads up to THIS case's extent rounded to the model's downsampling
                    # multiple, so a small heterogeneous case still reaches the network at a valid input
                    # size (the up-front worst-case sizing only guarantees the largest case).
                    m = free_axis_rounding(self.free_axis_multiple, axis, nspatial)
                    target = ((extent + m - 1) // m) * m if m > 1 else extent
                p = 0 if _slice.start + target <= extent else target - (extent - _slice.start)
                constant_padding.append(0)
                constant_padding.append(p)

        return PatchReadPlan(
            data_slices=tuple(slices_pre + slices),
            reflect_padding=tuple(reflect_padding),
            constant_padding=tuple(constant_padding),
            concatenate_extend_slice=extend_slice > 0,
        )

    def apply_read_plan(self, data: torch.Tensor, plan: PatchReadPlan) -> torch.Tensor:
        data_sliced = data
        if any(plan.reflect_padding):
            data_sliced = F.pad(data_sliced, plan.reflect_padding, "reflect")
        if any(plan.constant_padding):
            data_sliced = self._pad_constant(data_sliced, plan.constant_padding)
        if self.patch_size is not None and not all(p == 0 for p in self.patch_size):
            for d in [i for i, v in enumerate(reversed(self.patch_size)) if v == 1]:
                data_sliced = torch.squeeze(data_sliced, dim=len(data_sliced.shape) - d - 1)
        return (
            torch.cat([data_sliced[:, i, ...] for i in range(data_sliced.shape[1])], dim=0)
            if plan.concatenate_extend_slice
            else data_sliced
        )

    def _pad_constant(self, data: torch.Tensor, padding: tuple[int, ...]) -> torch.Tensor:
        """``data`` padded (``F.pad`` pair order) with the configured value, a uint8 map with zero,
        and anything else with its own minimum when no value is configured.

        That minimum never leaves the device: the bands are filled from the 0-d tensor after a zero
        pad, so a padded patch costs no host round trip (37 of the 64 patches of a 100^3 case at
        32^3 are padded, measured).
        """
        if data.dtype == torch.uint8:
            return F.pad(data, padding, "constant", 0)
        if self.pad_value is not None:
            return F.pad(data, padding, "constant", self.pad_value)
        padded = F.pad(data, padding, "constant", 0)
        lowest = data.min()
        for pair, (before, after) in enumerate(zip(padding[::2], padding[1::2], strict=True)):
            dim = padded.dim() - 1 - pair
            if before:
                padded.narrow(dim, 0, before).copy_(lowest)
            if after:
                padded.narrow(dim, padded.shape[dim] - after, after).copy_(lowest)
        return padded

    def get_data(self, data: torch.Tensor, index: int, a: int, is_input: bool) -> torch.Tensor:
        plan = self.get_read_plan(list(data.shape), index, a, is_input)
        data_sliced = data[plan.data_slices]
        return self.apply_read_plan(data_sliced, plan)

    def get_size(self, a: int = 0) -> int:
        return len(self._grid(a).slices)


@config("Patch")
class DatasetPatch(Patch):
    """Patch definition applied when sampling data from datasets."""

    def __init__(
        self,
        patch_size: list[int] = [128, 128, 128],
        overlap: OverlapSpec = None,
        pad_value: float | None = None,
        extend_slice: int = 0,
    ) -> None:
        super().__init__(patch_size, overlap, pad_value, extend_slice)

    def init(self, key: str = ""):
        pass


@config()
class ModelPatch(Patch):
    """Patch definition applied inside model graphs during prediction or training."""

    def __init__(
        self,
        patch_size: list[int] = [128, 128, 128],
        overlap: OverlapSpec = None,
        patch_combine: str | None = None,
        pad_value: float | None = None,
        extend_slice: int = 0,
    ) -> None:
        super().__init__(patch_size, overlap, pad_value, extend_slice)
        self._patch_combine = patch_combine
        self.patch_combine: PathCombine | None = None

    def init(self, key: str):
        if self._patch_combine is not None:
            module, name = get_module(self._patch_combine, "konfai.data.patching")
            self.patch_combine = apply_config(key)(getattr(module, name))()
        if self.patch_size is not None and self.overlap is not None:
            if self.patch_combine is not None:
                kept = blend_axes(self.patch_size)
                self.patch_combine.set_patch_config(kept, blend_overlap(self.overlap, kept))
        else:
            self.patch_combine = None

    def disassemble(self, *data_list: torch.Tensor) -> Iterator[list[torch.Tensor]]:
        for i in range(self.get_size()):
            yield [self.get_data(data, i, 0, True) for data in data_list]
