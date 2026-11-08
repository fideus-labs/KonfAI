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

from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from typing import Any, cast

import numpy as np
import torch

from konfai.data.transform.base import _RANK_CHANGE, LocalityKind, PatchLocality, Transform
from konfai.utils.dataset import DISPLACEMENT_FIELD_ATTRIBUTE, Attribute, Dataset, DataStream
from konfai.utils.errors import TransformError
from konfai.utils.utils import split_path_spec

#: The header key of a stack: how many members it holds. A stack is one volume whose channels are its members'
#: channels, member-major (``[M x C, *spatial]``), so one member of ``C`` channels is ``C`` consecutive channels.
#: The underscore keeps the key out of ``Attribute``'s stack renaming.
MEMBERS_ATTRIBUTE = "konfai_members"


def holds_stack(file_format: str) -> bool:
    """Whether a dataset of ``file_format`` keeps a stack's member count in its header: MetaImage, NRRD, OME-Zarr,
    KonfAI's HDF5. NIfTI's header keeps no free key, and a transform file holds one transform."""
    return file_format.split("@")[0] in ("mha", "nrrd", "omezarr", "h5")


def member_count(shape: Iterable[int], header: Attribute) -> int:
    """How many members a stack of ``shape`` holds: its header's count; one a channel without it, as a stack of
    one-channel members always was."""
    return int(float(header[MEMBERS_ATTRIBUTE])) if MEMBERS_ATTRIBUTE in header else int(next(iter(shape)))


def stack_members(tensors: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
    """A stack ``[M x C, *spatial]`` as ``[M, C, *spatial]``, its member count read off its header (:func:`member_count`),
    and the count dropped from ``cache_attribute``: what a stage makes of the members is no stack any more."""
    members = member_count(tensors.shape, cache_attribute)
    if MEMBERS_ATTRIBUTE in cache_attribute:
        cache_attribute.pop(MEMBERS_ATTRIBUTE)
    if members < 1 or tensors.shape[0] % members:
        raise TransformError(
            f"A stack of {tensors.shape[0]} channels does not hold {members} members of one channel count.",
            "Read the stack InferenceStack wrote, its header unchanged.",
        )
    return tensors.reshape(members, tensors.shape[0] // members, *tensors.shape[1:])


class MemberView:
    """A dataset whose stacks under ``groups`` read as member ``member`` alone: its channels, by region as the stack
    is, its header without the member count. Every other group, and every other call, is the dataset's own: a stage
    that applies a stored transform reads one member of a stack as the entry it was."""

    def __init__(self, dataset: Any, groups: set[str], member: int) -> None:
        self._dataset, self._groups, self._member = dataset, groups, member

    def __getattr__(self, name: str) -> Any:
        return getattr(self._dataset, name)

    def _narrow(self, group: str, name: str) -> tuple[list[int], int, int, Attribute] | None:
        """For a stack of ``groups``: its shape, the member's first channel, its channel count and its header (the
        stack's, without the count); ``None`` for anything else."""
        shape, header = self._dataset.get_infos(group, name)
        if group not in self._groups or MEMBERS_ATTRIBUTE not in header:
            return None
        channels = int(shape[0]) // member_count(shape, header)
        header = Attribute(header)
        header.pop(MEMBERS_ATTRIBUTE)
        return shape, self._member * channels, channels, header

    def get_infos(self, group: str, name: str) -> tuple[list[int], Attribute]:
        narrowed = self._narrow(group, name)
        if narrowed is None:
            return self._dataset.get_infos(group, name)
        shape, _, channels, header = narrowed
        return [channels, *shape[1:]], header

    def read_data(self, group: str, name: str) -> tuple[np.ndarray, Attribute]:
        narrowed = self._narrow(group, name)
        if narrowed is None:
            return self._dataset.read_data(group, name)
        shape, _, channels, _ = narrowed
        return self.read_data_slice(group, name, (slice(0, channels), *(slice(0, extent) for extent in shape[1:])))

    def read_data_slice(self, group: str, name: str, region: tuple[slice, ...]) -> tuple[np.ndarray, Attribute]:
        narrowed = self._narrow(group, name)
        if narrowed is None:
            return self._dataset.read_data_slice(group, name, region)
        _, first, channels, member_header = narrowed
        start, stop, _ = region[0].indices(channels)
        data, _ = self._dataset.read_data_slice(group, name, (slice(first + start, first + stop), *region[1:]))
        return data, member_header


@contextmanager
def as_member(stage: Any, stacked: set[str], member: int) -> Iterator[None]:
    """``stage`` reading member ``member`` of the stacks under ``stacked`` alone (:class:`MemberView`), its datasets
    given back after: a stage that applies a stored transform applies one member at a time."""
    datasets = stage.datasets
    stage.datasets = [cast(Dataset, MemberView(dataset, stacked, member)) for dataset in datasets]
    try:
        yield
    finally:
        stage.datasets = datasets


def stacked_groups(stage: Any, groups: Iterable[str], name: str) -> tuple[set[str], int]:
    """Which of ``groups`` hold a stack for case ``name`` (``stage.dataset_holding`` finds them), and its member
    count: 0 when none does. Stacks of one stage share a count: their members pair by position."""
    stacked, counts = set(), set()
    for group in groups:
        dataset = stage.dataset_holding(group, name)
        if dataset is None or getattr(dataset, "get_infos", None) is None:
            continue  # absent, or a store that serves transforms alone: no stack there
        header = dataset.get_infos(group, name)[1]
        if MEMBERS_ATTRIBUTE in header:
            stacked.add(group)
            counts.add(member_count([], header))
    if len(counts) > 1:
        raise TransformError(
            f"The stacks of case '{name}' hold {sorted(counts)} members: one member of each is applied together.",
            "Stack the same ensemble's members in every group a stage reads.",
        )
    return stacked, next(iter(counts), 0)


def _no_field(cache_attribute: Attribute) -> None:
    """Drop the displacement-field mark: what a stage made of a field (its spread, its magnitude) is no field."""
    if DISPLACEMENT_FIELD_ATTRIBUTE in cache_attribute:
        cache_attribute.pop(DISPLACEMENT_FIELD_ATTRIBUTE)


def _stack_attributes(cache_attribute: Attribute, members: int, dataset: Dataset, channels: int) -> Attribute:
    """The stack's own header: the output's, and its member count. The output itself is no stack. Members of several
    channels are told apart by the count alone, so their stack goes where a header keeps it."""
    if channels > 1 and dataset.file_format.split("@")[0] in ("nii", "nii.gz"):
        raise TransformError(
            f"A stack of members of {channels} channels needs its member count in its header, which NIfTI does not keep.",
            "Write the stack where a header holds it: InferenceStack(dataset='<path>:mha'), or nrrd, or omezarr.",
        )
    attributes = Attribute(cache_attribute)
    attributes[MEMBERS_ATTRIBUTE] = members
    return attributes


class _MemberSpread(Transform):
    """A per-voxel spread across the leading member axis (no spatial neighbour); one member spreads 0."""

    working_multiple = 2.0
    _spread: Callable[[torch.Tensor, int], torch.Tensor]

    locality = LocalityKind.POINTWISE

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, name: str, tensors: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        members = stack_members(tensors, cache_attribute)
        _no_field(cache_attribute)  # a spread of fields is no field
        if members.shape[0] > 1:
            return self._spread(members.float(), 0)
        return torch.zeros_like(members[0])


class Variance(_MemberSpread):
    _spread = staticmethod(torch.var)


class StandardDeviation(_MemberSpread):
    _spread = staticmethod(torch.std)


class SegmentationDisagreement(Transform):
    working_multiple = 24.5

    # Per-voxel majority disagreement across the members.
    locality = LocalityKind.POINTWISE

    def __init__(self, ignore_background: bool = False) -> None:
        super().__init__()
        self.ignore_background = ignore_background

    def __call__(self, name: str, tensors: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        # tensors shape: [N, ...] with N segmentations and integer labels per voxel
        tensors = stack_members(tensors, cache_attribute)[:, 0]
        if tensors.shape[0] <= 1:
            return torch.zeros_like(tensors[0], dtype=torch.float32).unsqueeze(0)

        tensors = tensors.long()

        if self.ignore_background:
            valid = tensors != 0
        else:
            valid = torch.ones_like(tensors, dtype=torch.bool)

        disagreement = torch.zeros_like(tensors[0], dtype=torch.float32)

        # per-voxel disagreement = 1 - (majority label count / number of valid segmentations)
        # A running maximum over the labels: one count is held at a time, whatever the label count.
        labels = torch.unique(tensors)
        max_count = torch.zeros_like(tensors[0])
        for label in labels:
            torch.maximum(max_count, ((tensors == label) & valid).sum(dim=0), out=max_count)
        valid_count = valid.sum(dim=0)

        non_empty = valid_count > 0
        disagreement[non_empty] = 1.0 - (max_count[non_empty].float() / valid_count[non_empty].float())

        return disagreement.unsqueeze(0)


class Percentage(Transform):
    working_multiple = 1.0

    locality = LocalityKind.POINTWISE

    def __init__(self, baseline: float) -> None:
        super().__init__()
        self.baseline = baseline

    def __call__(self, name: str, tensors: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        return (tensors / self.baseline).mul_(100.0)


class Magnitude(Transform):
    """Vector magnitude over the channel axis: ``[C, ...]`` becomes ``[1, ...]``.

    :class:`Norm`'s channel-first sibling, for a stored vector volume such as a displacement field.
    """

    working_multiple = 1.0

    locality = LocalityKind.POINTWISE

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, name: str, tensors: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        _no_field(cache_attribute)  # a field's magnitude is an image, in its unit: a reader must not take it for one
        return torch.linalg.norm(tensors.float(), dim=0, keepdim=True)


class Norm(Transform):
    """Vector magnitude over the trailing component axis.

    Reduces a stacked vector field (e.g. ``[N, (D), H, W, C]``) to per-sample magnitudes
    ``[N, (D), H, W]``. The trailing tensor axis is the first geometry axis, so it is dropped from
    ``Origin``/``Spacing``/``Direction``.
    """

    working_multiple = 2.0

    def __init__(self) -> None:
        super().__init__()

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        return PatchLocality(LocalityKind.WHOLE_VOLUME, reason=_RANK_CHANGE)

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
    working_multiple = 0.0

    def __init__(self, dataset: str | None = None, name: str | None = None, mode: str = "mean"):
        super().__init__()
        if mode not in ("mean", "median", "Seg"):
            raise TransformError(
                f"InferenceStack mode '{mode}' is unknown.", "Set mode to 'mean', 'median' or 'Seg' (spelled so)."
            )
        self.dataset = None
        if dataset:
            filename, _, file_format = split_path_spec(dataset)
            self.dataset = Dataset(filename, file_format)
        self.name = name
        self.mode = mode
        self._stack_sinks: dict[str, DataStream] = {}
        self._stack_buffers: dict[str, list[np.ndarray]] = {}

    # The member reduction is per-voxel; the per-member stack write needs the slab's place in the
    # volume, which is what SLAB declares.
    locality = LocalityKind.SLAB

    def _destination(self) -> Dataset:
        """Where the stack goes: the dataset the config names; else the output's, or MetaImage beside the output where
        its form cannot hold the stack (:func:`holds_stack`), as konfai-apps' ensembles do."""
        if self.dataset:
            return self.dataset
        output = self.datasets[-1]
        return output if holds_stack(output.file_format) else Dataset(output.filename, "mha")

    def _stack(self, tensors: torch.Tensor) -> np.ndarray:
        """The members ``[M, C, *spatial]`` as a stack: member-major channels ``[M x C, *spatial]``; a label map per
        member in ``Seg`` mode, kept integer: a reader tells a stack of labels from one of probabilities by its type."""
        if self.mode == "Seg":  # a softmax keeps the order it is taken of
            return torch.argmax(tensors, dim=1).to(torch.uint8).cpu().numpy()
        return tensors.reshape(tensors.shape[0] * tensors.shape[1], *tensors.shape[2:]).float().cpu().numpy()

    def _reduce(self, tensors: torch.Tensor) -> torch.Tensor:
        if self.mode != "median":
            return tensors.mean(0, dtype=torch.float32).to(tensors.dtype)
        # A median selects, so the stack is widened only where torch has no median kernel.
        try:
            return torch.median(tensors, dim=0).values
        except NotImplementedError:
            return torch.median(tensors.float(), dim=0).values.to(tensors.dtype)

    def __call__(self, name: str, tensors: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        if tensors.shape[0] == 1:
            return tensors.squeeze(0)
        dataset = self._destination()
        dataset.write(
            "InferenceStack",
            name,
            self._stack(tensors),
            _stack_attributes(cache_attribute, tensors.shape[0], dataset, tensors.shape[1]),
        )
        return self._reduce(tensors)

    @property
    def folds_copies(self) -> bool:
        """Whether :meth:`fold_copies` can stand for the call: a median needs every member at once."""
        return self.mode != "median"

    def fold_copies(
        self, name: str, copies: Iterable[torch.Tensor], count: int, cache_attribute: Attribute
    ) -> torch.Tensor:
        """The call over the concatenation of ``count`` copies (``[members, C, ...]`` each), one copy at a time:
        each copy's stack rows are written as they come, where the destination serves region writes, and its
        members are added to a running total."""
        dataset = self._destination()
        total, dtype, sink = torch.empty(0), torch.float32, None
        rows: list[np.ndarray] = []
        written = total_members = channels = 0  # the stack's rows (member channels), its members, a member's channels
        try:
            for members in copies:
                stack = self._stack(members)
                if written == 0:
                    total, dtype, channels = members.sum(0, dtype=torch.float32), members.dtype, members.shape[1]
                    if count * members.shape[0] > 1:
                        shape = [count * len(stack), *stack.shape[1:]]
                        header = _stack_attributes(cache_attribute, count * members.shape[0], dataset, channels)
                        sink = dataset.open_data_stream("InferenceStack", name, shape, stack.dtype, header)
                else:
                    total.add_(members.sum(0, dtype=torch.float32))
                if sink is None:
                    rows.append(stack)
                else:
                    region = (slice(written, written + len(stack)), *(slice(0, extent) for extent in stack.shape[1:]))
                    sink.write_slice(region, stack)
                written += len(stack)
                total_members += members.shape[0]
        except BaseException:
            if sink is not None:
                sink.abort()  # abort, not close: finalizing would publish a partial stack
            raise
        if sink is not None:
            sink.close()
        elif total_members > 1:
            dataset.write(
                "InferenceStack",
                name,
                np.concatenate(rows),
                _stack_attributes(cache_attribute, total_members, dataset, channels),
            )
        return total.div_(total_members).to(dtype)

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
        cannot serve region writes buffers until the last slab."""
        if tensor.shape[0] == 1:
            return tensor.squeeze(0)
        stack = self._stack(tensor)
        dataset = self._destination()
        header = _stack_attributes(cache_attribute, tensor.shape[0], dataset, tensor.shape[1])
        if name not in self._stack_sinks and name not in self._stack_buffers:
            sink = dataset.open_data_stream(
                "InferenceStack", name, [stack.shape[0], *spatial_shape], stack.dtype, header
            )
            if sink is None:
                self._stack_buffers[name] = []
            else:
                self._stack_sinks[name] = sink
        if name in self._stack_buffers:
            self._stack_buffers[name].append(stack)
            if region.stop == spatial_shape[0]:
                whole = np.concatenate(self._stack_buffers.pop(name), axis=1)
                dataset.write("InferenceStack", name, whole, header)
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
