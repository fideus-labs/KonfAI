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


"""The order patches are read in, and the sampler that walks it per rank."""

from collections.abc import Iterator
from typing import TYPE_CHECKING, cast

import torch
from torch.utils import data
from torch.utils.data import Sampler

if TYPE_CHECKING:
    from konfai.data.patching import DatasetPatch


def _interleaved_case_entries(patches: list["DatasetPatch"], entries: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """One case's ``(copy, patch)`` entries ordered so the copies advance together along the slab axis.

    A streamed TTA write reduces the copies slab by slab, so it can only advance to the slowest
    copy's frontier: walked copy-major, the first copy would be complete (and fully retained)
    before the second began. Ordering by each patch's declared first-spatial-axis start bounds that
    skew at one patch extent, whatever grid each copy was cut on. The sort is total on
    ``(start, copy, patch)``, so within a copy the order is untouched: per-copy accumulation is
    byte-identical either way, and the whole-volume path reduces at the end whatever the order.

    ``patches`` holds every destination group's grid for the case: one shared order must serve them
    all, so if the groups disagree on the slab starts (or a group cannot even index an entry) the
    plain order is kept. The interleave is a memory bound, never a correctness requirement.
    """

    def starts(patch: "DatasetPatch") -> list[int] | None:
        try:
            return [patch.get_patch_slices(copy)[index][0].start for copy, index in entries]
        except (IndexError, KeyError):
            return None

    reference = starts(patches[0])
    if reference is None or any(starts(patch) != reference for patch in patches[1:]):
        return entries
    order = dict(zip(entries, reference, strict=True))
    return sorted(entries, key=lambda entry: (order[entry], *entry))


class PatchReadOrder:
    """The epoch's patch order, published from where it is drawn to where the patches are read.

    A store that caches decoded chunks decides what to keep from the sequence it is told is coming
    (:meth:`~konfai.utils.dataset.Dataset.plan_region_reads`). On the patch route that sequence is
    the sampler's, drawn at the epoch's start in the parent, while the reads happen in the
    DataLoader's workers, each holding a cache of its own: a worker is forked before the draw, and a
    persistent one is never forked again, so the order reaches it through shared memory or not at
    all. Nothing but the order travels: neither what is read nor its values depend on this.

    A worker is handed one batch in ``num_workers``, so what it declares for a case is the
    subsequence of its own batches. Their stride is learnt from the first batch it is given, which
    assumes only that the batches are dealt round-robin, not which worker opens the epoch.
    """

    def __init__(self, mapping: list[tuple[int, int, int]], batch_size: int) -> None:
        self._mapping = mapping
        self._batch_size = max(1, batch_size)
        self._order = torch.zeros(len(mapping), dtype=torch.int64).share_memory_()
        self._epoch = torch.zeros(1, dtype=torch.int64).share_memory_()
        self._epoch_read = 0
        self._entries: dict[int, list[tuple[int, int]]] = {}

    def publish(self, order: torch.Tensor) -> None:
        """The order the epoch is about to be walked in, from the sampler that drew it."""
        self._order.copy_(order)
        self._epoch += 1

    def entering(self, index: int) -> list[tuple[int, int]] | None:
        """The ``(copy, patch)`` entries of ``index``'s case this process still has to read, in the
        order it will read them; ``None`` once the case has been entered this epoch, and while no
        sampler has published an order at all."""
        self._regroup(index)
        return self._entries.pop(self._mapping[index][0], None)

    def _regroup(self, index: int) -> None:
        """Group this process's remaining reads by case, once per epoch, from the batch ``index``
        opens: a batch is ``batch_size`` consecutive positions, and the next batch this process is
        handed is ``num_workers`` batches on."""
        epoch = int(self._epoch[0])
        if epoch == self._epoch_read:
            return
        self._epoch_read = epoch
        order = self._order.tolist()
        worker = data.get_worker_info()
        stride = self._batch_size * (worker.num_workers if worker is not None else 1)
        entries: dict[int, list[tuple[int, int]]] = {}
        for start in range(order.index(index), len(order), stride):
            for position in range(start, min(start + self._batch_size, len(order))):
                case, copy, patch = self._mapping[order[position]]
                entries.setdefault(case, []).append((copy, patch))
        self._entries = entries


class WindowedCaseSampler(Sampler[int]):
    """Locality-aware training order: shuffle cases, window them, shuffle patches within each window.

    ``DatasetIter`` loads each non-streamable case into a FIFO buffer, so a global patch shuffle
    reloads a volume repeatedly (once per patch that lands after an eviction. Keeping only
    ``window`` cases in play at a time) their patches shuffled together, emitted before advancing: reads each
    volume ~once. ``window`` is the decorrelation knob: ``1`` is perfect locality, and
    ``None`` (default) or ``>= n_cases`` is a single all-cases window, i.e. a plain global shuffle,
    byte for byte.

    A map-style ``DataLoader`` sends batch ``j`` to worker ``j % num_workers`` and gives each worker its
    own buffer, so the cases are partitioned across workers (greedy least-loaded by patch count, see
    ``_partitions``) and the per-worker windowed batches are round-robin interleaved: batch ``j`` then
    carries only worker ``j % num_workers``'s cases, and every volume is read by exactly one worker.
    """

    def __init__(
        self,
        mapping: list[tuple[int, int, int]],
        shuffle: bool,
        window: int | None,
        batch_size: int,
        num_workers: int,
        read_order: PatchReadOrder | None = None,
    ) -> None:
        self.mapping = mapping
        self.shuffle = shuffle
        # Where the epoch's order is published for the processes that read the patches; a sampler
        # asked for nothing but its order publishes nowhere.
        self.read_order = read_order
        self.batch_size = max(1, batch_size)
        self.num_workers = max(1, num_workers)
        self.case_entries: dict[int, list[int]] = {}
        for index, entry in enumerate(mapping):
            self.case_entries.setdefault(entry[0], []).append(index)
        # Windowing bites only when a window is both smaller than the case count and shardable across
        # the workers; anything else is a single all-cases window, i.e. a plain global shuffle.
        n_cases = len(self.case_entries)
        self.window = window if window is not None and 0 < window < n_cases and self.num_workers <= n_cases else None

    def _partitions(self) -> list[list[int]]:
        """The cases each worker walks, balanced by the patches they hold.

        A worker is handed whole cases, because a case is what its buffer keeps resident. Handing out
        an equal COUNT of them leaves the patch counts as uneven as the cases are, and it is patches
        that are walked: the workers then run out at different times, and the batches of whoever is
        left shift onto the workers that finished: a case landing on two of them, each reading the
        volume. Give the next case to whoever holds the fewest patches so far, largest first.
        """
        loads = [0] * self.num_workers
        partitions: list[list[int]] = [[] for _ in range(self.num_workers)]
        for case in sorted(self.case_entries, key=lambda case: -len(self.case_entries[case])):
            worker = min(range(self.num_workers), key=lambda worker: loads[worker])
            partitions[worker].append(case)
            loads[worker] += len(self.case_entries[case])
        return partitions

    def _windowed_order(self) -> list[int]:
        generator = torch.Generator().manual_seed(int(torch.randint(0, 2**31 - 1, (1,)).item()))
        window = cast(int, self.window)

        def shuffled(items: list[int]) -> list[int]:
            return [items[i] for i in torch.randperm(len(items), generator=generator).tolist()]

        streams: list[list[int]] = []
        for partition in self._partitions():
            cases = shuffled(partition)
            stream: list[int] = []
            for start in range(0, len(cases), window):
                stream += shuffled(
                    [index for case in cases[start : start + window] for index in self.case_entries[case]]
                )
            streams.append(stream)

        # Round-robin the workers a batch at a time, so batch j lands on worker j % num_workers and a
        # window stays resident while its patches are walked.
        #
        # Three things are wanted here and only two of them fit. An epoch must be one pass over the
        # mapping; a worker must keep whole cases, since a case is what its buffer holds; and a case
        # should stay on one worker, which needs every stream the same length. A case does not split,
        # so streams of equal length are not something `_partitions` can always hand over, one case
        # of 200 patches beside ten of 2 is longer on its own than a quarter of the epoch.
        #
        # So a short stream runs out and the ones still going shift onto the workers that finished:
        # a case lands on two of them and each reads its volume. The epoch stays exact and the reads
        # stay close to 1x: far below the redundant reads of no window at all. Padding the streams
        # instead buys the affinity back by walking part of the epoch twice and the rest not at all,
        # which is not a trade to make.
        batch = self.batch_size
        order: list[int] = []
        for start in range(0, max((len(stream) for stream in streams), default=0), batch):
            for stream in streams:
                order += stream[start : start + batch]
        return order

    def __iter__(self) -> Iterator[int]:
        if not self.shuffle:
            order = torch.arange(len(self.mapping))
        elif self.window is None:
            order = torch.randperm(len(self.mapping))
        else:
            order = torch.as_tensor(self._windowed_order(), dtype=torch.int64)
        if self.read_order is not None:
            self.read_order.publish(order)
        return iter(order.tolist())

    def __len__(self) -> int:
        # One epoch is one pass over the mapping: windowing chooses the ORDER, not the size. This is
        # what keeps the ranks in step: `Data._split` gives them equal-length shards, but not equal
        # cases, so any length read from the per-rank cases (their partitions, or even whether the
        # window engages at all) would differ and hang DDP's collectives.
        return len(self.mapping)
