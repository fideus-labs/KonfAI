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

Two callers, one vocabulary. The predictor reduces the copies of ONE case: an ensemble's models,
a TTA draw's augmentations. The transform workflow reduces one region across N CASES. Both fold a
leading axis at fixed voxel, which is why both can run region by region and neither ever has to
hold a whole volume.

A reduction is a KonfAI extension point: subclass :class:`Reduction`, reference it by classpath.
"""

from abc import ABC, abstractmethod

import torch

from konfai.utils.errors import ReductionError


class Reduction(ABC):
    """Aggregate a list of tensors, one per case, into one.

    Implement :meth:`__call__`. Override the incremental protocol as well when the operator can
    accumulate, which is what keeps a large number of them from being resident all at once.

    **The layout both engines hand over is** ``[1, K, C, *spatial]``: a singleton stack axis, then
    whatever axis distinguishes the things being folded (the models of an ensemble; for a cohort
    there is one per case, so ``K`` is the case's channels), then the volume. One convention,
    because ``Concat`` puts the folded things SIDE BY SIDE, which means naming an axis, and an
    operator cannot name one that its callers disagree about.
    """

    #: Streamed contract. ``True`` declares this a **pure per-voxel** operation over the case axis:
    #: every output voxel depends only on the SAME voxel of each input, never on a spatial neighbour.
    #: The streamed gates read this to decide whether a chain may run region by region.
    #:
    #: Rules for a custom reduction:
    #:
    #: - **Default False.** Leave it False unless you are sure: an unknown reduction then takes the
    #:   whole-volume path, costing the streaming optimisation but never correctness.
    #: - **Set True only if voxel-local.** Reducing/stacking along the case axis (dim 0) or the
    #:   channel axis (dim 1) is fine: both are orthogonal to the spatial axes. Anything reading
    #:   across spatial positions (a blur, a resample, a global argmax over Z) must stay False.
    #: - **A wrong True corrupts the streamed output** (each region would be reduced with only its
    #:   own cases); the gate trusts this flag and checks nothing else.
    voxel_local: bool = False

    #: Whether :meth:`accumulate` can fold cases one at a time. ``True`` bounds the working set at
    #: two regions however many cases there are; ``False`` keeps all N resident until :meth:`finalize`.
    #: A planner reads this to size its regions, so a wrong ``True`` is a wrong plan, and a
    #: wrong ``False`` only costs memory.
    incremental: bool = False

    #: Regions this operator allocates ON TOP of the ones it is handed, counted in buffers-worth.
    #: ``0`` folds what it already holds; an operator that stacks its inputs into a new tensor, or
    #: sorts a copy of them, is holding that many buffers again while it runs. The plan multiplies
    #: this into the peak it sizes regions against, so leaving it at ``0`` for an operator that
    #: copies is a plan promising a working set the run then exceeds.
    working_multiple: float = 0.0

    @abstractmethod
    def __call__(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError()

    def output_channels(self, channels: int, cases: int) -> int:
        """How many channels folding ``cases`` volumes of ``channels`` leaves. Same by default.

        The one thing about the result's SHAPE that only the operator knows. Nothing else can:
        ``transform_shape`` maps spatial extents and says nothing about the leading axis, so a
        planner asked to size the output has no way to tell an averaging operator from a stacking
        one. Overriding it wrongly costs a plan that probes a shape the run never writes, never a
        wrong volume: the write opens on the block it actually holds.
        """
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
    """The dtype an average of ``reference`` values belongs in.

    Floating inputs keep their own: an ensemble of float16 predictions must not silently double the
    bytes its output takes on disk. Integer inputs widen and stay widened, because the mean of 1 and
    2 is not 1: rounding an average back into an integer grid is a wrong number, not a narrower one.
    """
    return reference if reference.is_floating_point else torch.float32


class Mean(Reduction):
    """Average the cases element-wise.

    Accumulated in float32 whatever the inputs are, then returned as :func:`_averaged_dtype` says.
    """

    voxel_local = True
    incremental = True

    def __call__(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        if len(tensors) == 1:
            # Its own mean: skip the clone-and-accumulate, which for a whole-volume multi-class
            # output is a large no-op allocation.
            return tensors[0].to(_averaged_dtype(tensors[0].dtype))
        accumulator = tensors[0].float().clone()
        for tensor in tensors[1:]:
            accumulator.add_(tensor.float())
        return accumulator.div_(len(tensors)).to(_averaged_dtype(tensors[0].dtype))

    def start(self) -> None:
        self._total: torch.Tensor | None = None
        self._dtype: torch.dtype = torch.float32
        self._count = 0

    def accumulate(self, tensor: torch.Tensor) -> None:
        # A running total, so the inputs are never all resident: peak is one accumulator plus the
        # case being read, whatever N is. This is the whole point of `incremental`.
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
    """The element-wise median across cases.

    ``torch.median`` returns the LOWER of the two middle values for an even count, where a median is
    their average: over two cases it hands back the element-wise minimum, which is not a median
    of anything. An even number of cases is ordinary, so this averages the middle pair as
    ``numpy.median`` does; on an odd count the two agree and nothing changes.

    Not incremental, and cannot be: a median needs every case before it can name the middle one.
    Its working set is therefore every case at one region, which is what bounds the region size
    rather than the volume.

    **Not for label maps.** Averaging the middle pair means the result can be a value that was in no
    input (over labels 1 and 5 it is 3, a different structure), and over exactly two cases it is
    the mean, so the robustness the name promises is gone. Fold segmentations with :class:`Vote`.
    """

    voxel_local = True
    # ``torch.stack`` copies the buffer into a new tensor and ``torch.quantile`` sorts a copy of
    # that: two buffers-worth live alongside the one already held, for the duration of the call.
    working_multiple = 2.0

    def __call__(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        if len(tensors) == 1:
            return tensors[0].to(_averaged_dtype(tensors[0].dtype))
        middle = torch.quantile(torch.stack(tensors, dim=0).float(), 0.5, dim=0)
        return middle.to(_averaged_dtype(tensors[0].dtype))


class Vote(Reduction):
    """The label the most cases agree on, per voxel. The operator for folding SEGMENTATIONS.

    ``Mean`` and ``Median`` both answer with a value that was in no input: the median of labels 1 and
    5 is 3, which is a different structure. A label map made of invented labels is still a label map,
    so nothing downstream reports it, and the dtype widens to float32 on top. This one picks and never
    blends, and the result keeps the input's dtype.

    A tie goes to the SMALLEST label, so a cohort folds to the same volume on every run and on every
    rank. That is arbitrary but it has to be *something*, and an arbitrary rule stated here beats a
    stable-sort detail nobody can see.

    Not incremental: a majority needs every case before it can be counted.
    """

    voxel_local = True
    # ``torch.mode`` sorts a copy of the stack it is handed, alongside the stack itself.
    working_multiple = 2.0

    def __call__(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        if len(tensors) == 1:
            return tensors[0]
        return torch.mode(torch.stack(tensors, dim=0), dim=0).values.to(tensors[0].dtype)


class Concat(Reduction):
    """Concatenate the cases along the channel dimension."""

    # Cats along the channel axis, orthogonal to the spatial axes: per-voxel, so region-local.
    voxel_local = True
    # Nothing on top of the buffer: the concatenation IS the output region, and the plan charges that
    # separately at this operator's own (wider) output width.
    working_multiple = 0.0

    def __call__(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat(tensors, dim=1)

    def output_channels(self, channels: int, cases: int) -> int:
        # The one operator that does not keep the channel count: the cases end up side by side.
        return channels * cases
