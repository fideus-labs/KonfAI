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
            self._total, self._dtype = tensor.to(torch.float32, copy=True), tensor.dtype
        else:
            # Promoted inside the add kernel, whose cast to float32 is the one float() made: the
            # same bits, without the float32 copy of the member that float() put beside the total
            # (64 MiB per 32 MiB fp16 member: one allocation per member on CUDA, none here,
            # measured; the CPU iterator casts into a temporary either way). A member wider than
            # float32 would be added at its own width and rounded after, so it is narrowed first.
            promoted = torch.promote_types(torch.float32, tensor.dtype) is torch.float32
            self._total.add_(tensor if promoted else tensor.float())
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
    # THE MIDDLE IS SELECTED, NEVER SORTED. A sort along the case axis copies the stack and returns
    # int64 indices over it -- eight bytes an element whatever the members weigh -- so ten uint16
    # regions of 33 x 1331 x 1775 (1.45 GiB) sorted at 6.0x their own size, and ten float32 ones
    # at 4.0x (peak resident above the members, measured). A selection network of element-wise
    # min/max holds a WINDOW of the k+1 smallest members seen so far, in the averaging dtype, and
    # inserts each member into it: no stack, no indices, and the members stay in the dtype they
    # arrived in. Ten uint16 members: 1.8x, against 6.0x. Twice the arithmetic of the sort (3.4 s
    # against 1.6 on that region) on a fold whose clock is the disk by 40 to 1, and whose regions
    # the planner may now cut two to three times taller.
    #
    # The attribute is the worst case the plan may see; :meth:`working_multiple_for` prices the
    # network for the count it is handed.
    working_multiple = 2.5
    #: What the hand-written networks (three to five) hold beside the members they are handed,
    #: measured on a 24 MiB float32 member.
    _NETWORK_MULTIPLE: ClassVar[dict[int, float]] = {1: 1.0, 2: 1.5, 3: 1.0, 4: 2.5, 5: 1.5}
    #: Past five, the window: k+1 float32 buffers for k = count // 2, and the two it blends,
    #: measured on a 293 MiB uint16 member at ten.
    _WINDOW_MULTIPLE = 1.8

    def working_multiple_for(self, cases: int) -> float:
        if cases > 5:
            return self._WINDOW_MULTIPLE
        return self._NETWORK_MULTIPLE.get(cases, float(self.working_multiple))

    @staticmethod
    def _median_of_three(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return torch.maximum(torch.minimum(a, b), torch.minimum(torch.maximum(a, b), c))

    def __call__(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        dtype = _averaged_dtype(tensors[0].dtype)
        if len(tensors) == 1:
            return tensors[0].to(dtype)
        # The members are handed over in the dtype they arrived in and widened one at a time as the
        # network takes them: torch has no integer min/max kernel on the CPU, and widening ten
        # members up front is what put ten float32 copies beside ten uint16 regions.
        low, high = self._middle_pair(tensors, dtype)
        return low if low is high else torch.lerp(low, high, 0.5)

    def _middle_pair(self, members: list[torch.Tensor], dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """The one middle member of an odd fold (the same tensor twice), or the two an even fold
        averages: by a hand-written network up to five members, by the insertion window past it.
        ``dtype`` is what the network computes in; the members are widened to it as they enter."""
        minimum, maximum = torch.minimum, torch.maximum
        count = len(members)
        if count > 5:
            return self._middle_pair_by_window(members, dtype)
        members = [member.to(dtype) for member in members]
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
        a, b, c, d, e = members
        a, b = minimum(a, b), maximum(a, b)
        c, d = minimum(c, d), maximum(c, d)
        a, c = minimum(a, c), maximum(a, c)  # a is the fold's smallest: out of the running
        b, d = minimum(b, d), maximum(b, d)  # d is its largest: out too
        middle = self._median_of_three(b, c, e)
        return middle, middle

    @staticmethod
    def _middle_pair_by_window(members: list[torch.Tensor], dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """The middle pair of any count, by an insertion window of the ``k + 1`` smallest seen.

        The middle of ``count`` members is rank ``count // 2`` (0-based; the pair ``count // 2 - 1``
        and ``count // 2`` on an even count). A window that keeps the ``k + 1`` smallest members
        seen so far, ``k = count // 2``, holds those ranks exactly once every member has passed
        through it: a member larger than the whole window can be no smaller than rank ``k + 1`` of
        the members seen, so dropping it off the end loses nothing the answer needs. Each insertion
        is a chain of element-wise min/max, which is what makes the selection exact -- the same
        values a full sort returns, to the bit (pinned against ``torch.sort`` in the tests).
        """
        count = len(members)
        keep = count // 2 + 1
        window: list[torch.Tensor] = []
        for member in members:
            window.append(member.to(dtype))
            for index in range(len(window) - 1, 0, -1):
                lower, upper = window[index - 1], window[index]
                window[index - 1], window[index] = torch.minimum(lower, upper), torch.maximum(lower, upper)
            if len(window) > keep:
                window.pop()
        middle = window[count // 2]
        if count % 2:
            return middle, middle
        return window[count // 2 - 1], middle


class Vote(Reduction):
    """The label the most cases agree on, per voxel: the operator for folding segmentations. It
    picks and never blends, keeps the input's dtype, and breaks a tie on the smallest label (the
    same volume on every run and rank). Not incremental.
    """

    voxel_local = True
    # What the running best holds beside the members: two counts, the best label, the mask
    # deciding it and the comparisons behind it, 2 x itemsize + 6 bytes per voxel (measured 6 to
    # 9 on uint8 and int16 members, at two and at six members). A fixed set of planes, so as a
    # share of the cohort it shrinks with the count: 16 bytes over the plan's 4 per member, and
    # the attribute is the two-member fold, the worst case.
    working_multiple = 2.0

    def working_multiple_for(self, cases: int) -> float:
        return 4.0 / cases if cases > 1 else 0.0

    def __call__(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        if len(tensors) == 1:
            return tensors[0]
        # NOT torch.mode: its tie-break is the smallest label on CPU and the LARGEST on CUDA
        # (measured: 16% of a six-member fold differed by device), so a --gpu reduce and a CPU
        # reduce wrote different label maps under one provenance. Every member is a candidate
        # counted against the others, and the running best keeps a strictly larger count or the
        # same count on a smaller label: the smallest label with the most votes, on every device
        # and whatever the members' order. Per voxel this is cases^2 comparisons and nothing else:
        # no stack sorted along the case axis (whose int64 indices alone were 8 bytes per voxel and
        # member: a 128-row slab of six 512x512 uint8 members peaked 1920 MiB sorted, 256 MiB
        # here, 64 -> 11 ms on CUDA, measured), no unique over the volume, no per-label planes.
        best, best_votes = tensors[0], self._votes_for(0, tensors)
        for index in range(1, len(tensors)):
            member, votes = tensors[index], self._votes_for(index, tensors)
            better = (votes > best_votes) | ((votes == best_votes) & (member < best))
            best = torch.where(better, member, best)
            best_votes = torch.where(better, votes, best_votes)
        return best

    @staticmethod
    def _votes_for(index: int, tensors: list[torch.Tensor]) -> torch.Tensor:
        """How many members hold member ``index``'s label, per voxel: its own vote and every
        equal one.

        Counted in uint8 from the mask viewed as uint8 (a bool is one byte holding 0 or 1): the
        same dtype on both sides of the add, so the CPU iterator adds in place where a bool into
        a wider count went through a cast temporary (1156 -> 784 ms per six-member int16 slab on
        the host, 29 -> 21 ms on CUDA, measured). int16 past 255 members, where uint8 would wrap.
        """
        member = tensors[index]
        votes = torch.ones_like(member, dtype=torch.uint8 if len(tensors) < 256 else torch.int16)
        for other_index, other in enumerate(tensors):
            if other_index != index:
                votes.add_((other == member).view(torch.uint8))
        return votes


class Concat(Reduction):
    """Concatenate the cases along the channel dimension."""

    voxel_local = True  # along the channel axis: per voxel
    working_multiple = 0.0  # the concatenation is the output region, charged at its own width

    def __call__(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat(tensors, dim=1)

    def output_channels(self, channels: int, cases: int) -> int:
        return channels * cases
