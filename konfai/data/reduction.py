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

"""Operators that aggregate several tensors into one, and the contract that lets them stream.

Two callers, one vocabulary: the predictor reduces the copies of one case (an ensemble's models, a
TTA draw's augmentations), the transform workflow reduces one region across N cases. Both fold a
leading axis at fixed voxel, region by region.

A reduction is an extension point: subclass :class:`Reduction`, reference it by classpath.
"""

from abc import ABC, abstractmethod
from typing import ClassVar

import torch

from konfai.utils.errors import ReductionError


class Reduction(ABC):
    """Aggregate a list of tensors, one per case, into one.

    Implement :meth:`__call__`; override the incremental protocol (:meth:`start`,
    :meth:`accumulate`, :meth:`finalize`) when the operator can fold cases one at a time.

    Both engines hand over ``[1, K, C, *spatial]``: a singleton stack axis, the axis that
    distinguishes the folded things (the models of an ensemble; the case's channels for a cohort),
    then the volume.
    """

    #: ``True`` declares a per-voxel operation over the case axis: every output voxel depends only on
    #: the SAME voxel of each input (a fold along dim 0 or 1). Anything reading across spatial
    #: positions must stay ``False``. The streamed gates trust this flag and check nothing else: a
    #: wrong ``True`` corrupts a streamed output (each region reduced with its own cases only); a
    #: wrong ``False`` costs the whole-volume path.
    voxel_local: bool = False

    #: Whether :meth:`accumulate` can fold cases one at a time (working set: two regions, whatever
    #: N is). A wrong ``True`` is a wrong plan; a wrong ``False`` only costs memory.
    incremental: bool = False

    #: Buffers-worth this operator allocates on top of the regions it is handed (a stack, a sorted
    #: copy). The plan multiplies it into the peak it sizes regions against.
    working_multiple: float = 0.0

    def working_multiple_for(self, cases: int) -> float:
        """:attr:`working_multiple` for a fold of ``cases`` members, which is what the plan asks.

        The attribute is the contract and the worst case; an operator whose route depends on the
        count (``Median`` selects the middle by network up to five members, and sorts past that)
        answers what THAT route holds, so the plan can cut taller slabs where the cheaper route runs.
        """
        return float(self.working_multiple)

    @abstractmethod
    def __call__(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError()

    def output_channels(self, channels: int, cases: int) -> int:
        """How many channels folding ``cases`` volumes of ``channels`` leaves; the same by default.
        The plan sizes and probes the output with it (``Concat`` widens)."""
        del cases
        return channels

    def start(self) -> None:
        """Begin one region. The default buffers, so a non-incremental operator needs nothing."""
        self._buffer: list[torch.Tensor] = []

    def accumulate(self, tensor: torch.Tensor) -> None:
        """Fold one case's region in. The default keeps it, which is what ``__call__`` expects."""
        self._buffer.append(tensor)

    def finalize(self) -> torch.Tensor:
        """The reduced region, and the end of this region's state."""
        result = self(self._buffer)
        self._buffer = []
        return result


def _averaged_dtype(reference: torch.dtype) -> torch.dtype:
    """The dtype an average of ``reference`` values belongs in: floating inputs keep their own,
    integer inputs widen to float32 (the mean of 1 and 2 is not an integer)."""
    return reference if reference.is_floating_point else torch.float32


class Mean(Reduction):
    """Average the cases element-wise.

    Accumulated in float32 whatever the inputs are, then returned as :func:`_averaged_dtype` says.
    """

    voxel_local = True
    incremental = True

    def __call__(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        if len(tensors) == 1:
            return tensors[0].to(_averaged_dtype(tensors[0].dtype))
        self.start()
        for tensor in tensors:
            self.accumulate(tensor)
        return self.finalize()

    def start(self) -> None:
        self._total: torch.Tensor | None = None
        self._dtype: torch.dtype = torch.float32
        self._count = 0

    def accumulate(self, tensor: torch.Tensor) -> None:
        if self._total is None:
            self._total, self._dtype = tensor.float().clone(), tensor.dtype
        else:
            self._total.add_(tensor.float())
        self._count += 1

    def finalize(self) -> torch.Tensor:
        if self._total is None:
            raise ReductionError("Mean.finalize() with no case accumulated.", "Accumulate at least one case.")
        result = self._total.div_(self._count).to(_averaged_dtype(self._dtype))
        self._total, self._count = None, 0
        return result


class Std(Reduction):
    """The element-wise standard deviation across cases: the ensemble-spread map.

    Welford's running moments, so the peak is two float32 accumulators plus the case being read,
    whatever N is. Unbiased (N-1), matching ``torch.std``; a single case has no spread and
    finalizes to zeros.
    """

    voxel_local = True
    incremental = True
    # Two persistent accumulators (mean, m2) plus, per accumulate: the float copy of the case,
    # ``delta``, and ``value - mean`` again after the mean moved: five buffers beside the region.
    working_multiple = 5.0

    def __call__(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        self.start()
        for tensor in tensors:
            self.accumulate(tensor)
        return self.finalize()

    def start(self) -> None:
        self._count = 0
        self._mean: torch.Tensor | None = None
        self._m2: torch.Tensor | None = None

    def accumulate(self, tensor: torch.Tensor) -> None:
        value = tensor.float()
        self._count += 1
        if self._mean is None or self._m2 is None:
            self._mean, self._m2 = value.clone(), torch.zeros_like(value)
            return
        delta = value - self._mean
        self._mean.add_(delta / self._count)
        self._m2.addcmul_(delta, value - self._mean)

    def finalize(self) -> torch.Tensor:
        if self._mean is None or self._m2 is None:
            raise ReductionError("Std.finalize() with no case accumulated.", "Accumulate at least one case.")
        result = torch.zeros_like(self._mean) if self._count < 2 else (self._m2 / (self._count - 1)).sqrt_()
        self._mean, self._m2, self._count = None, None, 0
        return result


class Median(Reduction):
    """The element-wise median across cases, averaging the middle pair on an even count as
    ``numpy.median`` does (``torch.median`` returns the lower one). Not incremental: every case
    is resident at one region. Not for label maps: the average of two labels is a third one; fold
    segmentations with :class:`Vote`.
    """

    voxel_local = True
    # ``torch.stack`` copies the buffer and the sort along the case axis returns values and
    # int64 indices over that: measured at 4x the stack it is handed (6 x 16 MiB float32 cases).
    # A fold of three, four or five members takes the network instead and costs far less; the
    # attribute is the worst case, :meth:`working_multiple_for` is what the plan asks.
    working_multiple = 4.0

    #: What the selection networks below hold beside the members they are handed, measured on a
    #: 24 MiB member (float32): the sort's own 4.0 is what anything wider still costs.
    _NETWORK_MULTIPLE: ClassVar[dict[int, float]] = {1: 1.0, 2: 1.5, 3: 1.0, 4: 2.5, 5: 1.5}

    def working_multiple_for(self, cases: int) -> float:
        return self._NETWORK_MULTIPLE.get(cases, float(self.working_multiple))

    @staticmethod
    def _median_of_three(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return torch.maximum(torch.minimum(a, b), torch.minimum(torch.maximum(a, b), c))

    def __call__(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        dtype = _averaged_dtype(tensors[0].dtype)
        members = [tensor.to(dtype) for tensor in tensors]
        if len(members) == 1:
            return members[0]
        # Three to five members is what a fold has, and there the middle is SELECTED by a network of
        # element-wise min/max rather than found by sorting the whole stack: same values to the bit,
        # a fraction of the time (CUDA 7.20 -> 0.45 ms at three, 8.42 -> 1.10 at five; CPU 55 -> 26
        # at three), and no stack to hold, which is what lets the planner cut taller slabs. Beyond
        # five the sort is simpler and no slower: what torch.quantile computes without its
        # interpolation machinery (1.5-2x on CPU, 3.5x on CUDA, measured).
        low, high = self._middle_pair(members)
        return low if low is high else torch.lerp(low, high, 0.5)

    def _middle_pair(self, members: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """The one middle member of an odd fold (the same tensor twice), or the two an even fold
        averages: by network up to five members, off a sorted stack past it."""
        minimum, maximum = torch.minimum, torch.maximum
        count = len(members)
        if count == 2:
            first, second = members
            return minimum(first, second), maximum(first, second)
        if count == 3:
            middle = self._median_of_three(*members)
            return middle, middle
        if count == 4:
            a, b, c, d = members
            a, b = minimum(a, b), maximum(a, b)
            c, d = minimum(c, d), maximum(c, d)
            second, third = maximum(a, c), minimum(b, d)
            return minimum(second, third), maximum(second, third)
        if count == 5:
            a, b, c, d, e = members
            a, b = minimum(a, b), maximum(a, b)
            c, d = minimum(c, d), maximum(c, d)
            a, c = minimum(a, c), maximum(a, c)  # a is the fold's smallest: out of the running
            b, d = minimum(b, d), maximum(b, d)  # d is its largest: out too
            middle = self._median_of_three(b, c, e)
            return middle, middle
        ranked = torch.stack(members, dim=0).sort(dim=0).values
        if count % 2:
            middle = ranked[count // 2]
            return middle, middle
        return ranked[count // 2 - 1], ranked[count // 2]


class Vote(Reduction):
    """The label the most cases agree on, per voxel: the operator for folding segmentations. It
    picks and never blends, keeps the input's dtype, and breaks a tie on the smallest label (the
    same volume on every run and rank). Not incremental.
    """

    voxel_local = True
    # The sorted copy of the stack, and one stack-sized mask beside it.
    working_multiple = 2.0

    def __call__(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        if len(tensors) == 1:
            return tensors[0]
        # NOT torch.mode: its tie-break is the smallest label on CPU and the LARGEST on CUDA
        # (measured: 19% of a six-member fold differed by device), so a --gpu reduce and a CPU
        # reduce wrote different label maps under one provenance. Sorted along the case axis and
        # counted member by member, the first count that is not beaten IS the smallest label with
        # the most votes, on every device. Per voxel this is cases^2 comparisons over the stack and
        # nothing else: no unique over the volume (a sort of every voxel, 31 s per 512^3 on a
        # CPU where this takes 2 s), no per-label count planes.
        ranked = torch.stack(tensors, dim=0).sort(dim=0).values
        best, best_count = ranked[0], (ranked == ranked[0]).sum(dim=0, dtype=torch.int16)
        for member in ranked[1:]:
            count = (ranked == member).sum(dim=0, dtype=torch.int16)
            better = count > best_count
            best = torch.where(better, member, best)
            best_count = torch.where(better, count, best_count)
        return best


class Concat(Reduction):
    """Concatenate the cases along the channel dimension."""

    voxel_local = True  # along the channel axis: per voxel
    working_multiple = 0.0  # the concatenation is the output region, charged at its own width

    def __call__(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat(tensors, dim=1)

    def output_channels(self, channels: int, cases: int) -> int:
        return channels * cases
