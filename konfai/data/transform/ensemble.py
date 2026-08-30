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


"""Folds over a stacked member axis: spread, disagreement, magnitudes, inference stacks."""

from collections.abc import Callable

import numpy as np
import torch

from konfai.data.transform.base import LocalityKind, PatchLocality, Transform
from konfai.utils.dataset import Attribute, Dataset, DataStream
from konfai.utils.utils import split_path_spec


class _MemberSpread(Transform):
    """A per-voxel spread across the leading member axis (no spatial neighbour); one member spreads 0."""

    # Measured at 2.00 on the CUDA allocator, in volumes-worth of what it is handed.
    working_multiple = 2.0
    _spread: Callable[[torch.Tensor, int], torch.Tensor]

    def __init__(self) -> None:
        super().__init__()

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        return PatchLocality(LocalityKind.POINTWISE)

    def __call__(self, name: str, tensors: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        # The member axis stays in both branches: var/std drop it and unsqueeze re-adds it.
        if tensors.shape[0] > 1:
            return self._spread(tensors.float(), 0).unsqueeze(0)
        return torch.zeros_like(tensors[0]).unsqueeze(0)


class Variance(_MemberSpread):
    _spread = staticmethod(torch.var)


class StandardDeviation(_MemberSpread):
    _spread = staticmethod(torch.std)


class SegmentationDisagreement(Transform):
    # What it holds beyond its input and its output: the pairwise comparison over the model axis: measured 9.33 on the CUDA allocator.
    working_multiple = 24.5

    def __init__(self, ignore_background: bool = False) -> None:
        super().__init__()
        self.ignore_background = ignore_background

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        # Per-voxel majority disagreement across the members. The global torch.unique only widens the
        # label set with labels absent at a given voxel, which contribute zero counts there and never
        # change that voxel's majority, so the result is decided voxel by voxel.
        return PatchLocality(LocalityKind.POINTWISE)

    def __call__(self, name: str, tensors: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        # tensors shape: [N, ...] with N segmentations and integer labels per voxel
        if tensors.shape[0] <= 1:
            return torch.zeros_like(tensors[0], dtype=torch.float32).unsqueeze(0)

        tensors = tensors.long()

        if self.ignore_background:
            valid = tensors != 0
        else:
            valid = torch.ones_like(tensors, dtype=torch.bool)

        disagreement = torch.zeros_like(tensors[0], dtype=torch.float32)

        # per-voxel disagreement = 1 - (frequency of majority label / number of valid segmentations)
        unique_labels = torch.unique(tensors)
        counts = []
        for label in unique_labels:
            counts.append(((tensors == label) & valid).sum(dim=0))

        counts = torch.stack(counts, dim=0)  # [L, ...]
        max_count = counts.max(dim=0).values
        valid_count = valid.sum(dim=0)

        non_empty = valid_count > 0
        disagreement[non_empty] = 1.0 - (max_count[non_empty].float() / valid_count[non_empty].float())

        return disagreement.unsqueeze(0)


class Percentage(Transform):
    # What it holds beyond its input and its output: the quantile's own copy: measured 1.00 on the CUDA allocator.
    working_multiple = 1.0

    def __init__(self, baseline: float) -> None:
        super().__init__()
        self.baseline = baseline

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        return PatchLocality(LocalityKind.POINTWISE)

    def __call__(self, name: str, tensors: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        return tensors / self.baseline * 100.0


class Magnitude(Transform):
    """Vector magnitude over the CHANNEL axis: ``[C, ...]`` becomes ``[1, ...]``.

    :class:`Norm`'s channel-first sibling. ``Norm`` folds the trailing axis of a stacked ensemble
    and is whole-volume by construction (a rank change past the streamed write); a stored vector
    volume (a displacement field read as a case) is channel-first, and its magnitude at a voxel
    reads that voxel alone: POINTWISE, so it streams.
    """

    # Measured at 1.00 on the CUDA allocator, in volumes-worth of what it is handed.
    working_multiple = 1.0

    def __init__(self) -> None:
        super().__init__()

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        return PatchLocality(LocalityKind.POINTWISE)

    def __call__(self, name: str, tensors: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        return torch.linalg.norm(tensors.float(), dim=0, keepdim=True)


class Norm(Transform):
    """Vector magnitude over the trailing component axis.

    Reduces a stacked vector field (e.g. a displacement-field ensemble ``[N, (D), H, W, C]``) to
    per-sample magnitudes ``[N, (D), H, W]``, typically before ``Variance``/``StandardDeviation``.
    The trailing tensor axis is the first geometry axis (numpy order is reversed), so that axis is
    dropped from ``Origin``/``Spacing``/``Direction``.
    """

    # Measured at 2.00 on the CUDA allocator, in volumes-worth of what it is handed.
    working_multiple = 2.0

    def __init__(self) -> None:
        super().__init__()

    # WHOLE_VOLUME on purpose: the magnitude drops the trailing spatial axis, and the streamed write
    # sizes each slab from the pre-finalize accumulator grid: a rank change past it cannot region-stream.

    def __call__(self, name: str, tensors: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        if "Origin" in cache_attribute:
            origin = cache_attribute.pop_np_array("Origin")
            spacing = cache_attribute.pop_np_array("Spacing")
            direction = cache_attribute.pop_np_array("Direction")
            rank = len(origin)
            cache_attribute["Origin"] = origin[1:]
            cache_attribute["Spacing"] = spacing[1:]
            cache_attribute["Direction"] = direction.reshape(rank, rank)[1:, 1:].flatten()
        return torch.linalg.norm(tensors.float(), dim=-1)

    def transform_shape(self, group_src: str, name: str, shape: list[int], cache_attribute: Attribute) -> list[int]:
        return shape[:-1]


class InferenceStack(Transform):
    # Measured at 0.00 on the CUDA allocator: it holds nothing beyond what it is handed.
    working_multiple = 0.0

    def __init__(self, dataset: str, name: str, mode: str = "mean"):
        super().__init__()
        self.dataset = None
        if dataset:
            filename, _, file_format = split_path_spec(dataset)
            self.dataset = Dataset(filename, file_format)
        self.name = name
        self.mode = mode
        self._stack_sinks: dict[str, DataStream] = {}
        self._stack_buffers: dict[str, list[np.ndarray]] = {}

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        # The member reduction is per-voxel; the per-member stack write is the side effect that needs
        # the slab's place in the volume, which is exactly what SLAB declares (whole-volume on the
        # read side, streamed region by region on the write side via ``stream_slab``).
        return PatchLocality(LocalityKind.SLAB)

    def _stack(self, tensors: torch.Tensor) -> np.ndarray:
        if self.mode == "Seg":
            _tensors = torch.argmax(torch.softmax(tensors, dim=1), dim=1).to(torch.uint8)
        else:
            _tensors = tensors.squeeze(1)
        return _tensors.float().cpu().numpy()

    def _reduce(self, tensors: torch.Tensor) -> torch.Tensor:
        if self.mode != "median":
            # The mean has to accumulate wider, and torch materialises that copy however it is spelled.
            return tensors.float().mean(0).to(tensors.dtype)
        # A median SELECTS: the element picked is the same under any monotone cast and comes back in
        # its own dtype, so the stack is widened only where torch has no median kernel for it
        # (uint16, uint32, uint64, bool).
        try:
            return torch.median(tensors, dim=0).values
        except NotImplementedError:
            return torch.median(tensors.float(), dim=0).values.to(tensors.dtype)

    def __call__(self, name: str, tensors: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        if tensors.shape[0] == 1:
            return tensors.squeeze(0)
        dataset = self.dataset if self.dataset else self.datasets[-1]
        dataset.write("InferenceStack", name, self._stack(tensors), cache_attribute)
        return self._reduce(tensors)

    def stream_slab(
        self,
        name: str,
        tensor: torch.Tensor,
        region: slice,
        spatial_shape: list[int],
        cache_attribute: Attribute,
    ) -> torch.Tensor:
        """The whole-volume call, region by region: reduce the members per voxel and write the slab's
        rows of the per-member stack into a region sink opened at the first slab. A destination that
        cannot serve region writes falls back to buffering the stack and writing it classically at
        the last slab: the memory cost of the whole-volume path, never a lost stack."""
        if tensor.shape[0] == 1:
            return tensor.squeeze(0)
        stack = self._stack(tensor)
        dataset = self.dataset if self.dataset else self.datasets[-1]
        if name not in self._stack_sinks and name not in self._stack_buffers:
            sink = dataset.open_data_stream(
                "InferenceStack", name, [stack.shape[0], *spatial_shape], stack.dtype, cache_attribute
            )
            if sink is None:
                self._stack_buffers[name] = []
            else:
                self._stack_sinks[name] = sink
        if name in self._stack_buffers:
            self._stack_buffers[name].append(stack)
            if region.stop == spatial_shape[0]:
                whole = np.concatenate(self._stack_buffers.pop(name), axis=1)
                dataset.write("InferenceStack", name, whole, cache_attribute)
        else:
            target = (slice(0, stack.shape[0]), region, *(slice(0, extent) for extent in spatial_shape[1:]))
            self._stack_sinks[name].write_slice(target, stack)
            if region.stop == spatial_shape[0]:
                self._stack_sinks.pop(name).close()
        return self._reduce(tensor)

    def stream_abort(self, name: str) -> None:
        self._stack_buffers.pop(name, None)
        sink = self._stack_sinks.pop(name, None)
        if sink is not None:
            # Abort, not close: finalizing would publish the partial stack under its final name.
            sink.abort()
