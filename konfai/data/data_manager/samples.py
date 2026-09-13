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


"""What a loader yields: items, batches, and the torch dataset over the cases' managers."""

import ctypes
import os
import threading
import traceback
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from typing import TypeAlias

import torch
import tqdm
from torch.cuda import device_count
from torch.utils import data

from konfai.data.augmentation import DataAugmentationsList
from konfai.data.data_manager.groups import Group, GroupMetric, GroupOut, _chains
from konfai.data.data_manager.order import PatchReadOrder
from konfai.data.patching import DatasetManager
from konfai.utils.dataset import Attribute
from konfai.utils.runtime import get_cpu_info, get_memory, get_memory_info, memory_forecast
from konfai.utils.utils import OverlapSpec

# A cached case is a float32 tensor (torch's default dtype, and the default TensorCast's target), so
# bytes are counted at 4/element from the header shape alone, not the on-disk dtype, and without
# modelling transforms that shrink or grow the cached tensor.
_CACHE_ELEMENT_BYTES = 4


def _cache_worker_count(cpu_count: int, device_count: int) -> int:
    """Number of caching threads: CPUs shared across devices, but never below one."""
    divisor = device_count if device_count > 0 else 1
    return max(1, cpu_count // divisor)


#: Said once per process: a chain that cannot serve a region costs a whole case per patch, and the
#: reader needs the refusing stage named once, not once per item.
_said_why_a_case_is_materialized = False

#: Said once per process, beside it: what a streamed case costs in reads of its own voxels.
_said_what_streaming_reads = False


@dataclass(frozen=True)
class DataItem:
    """Single tensor sample together with dataset metadata and patch indices."""

    name: str
    tensor: torch.Tensor
    attribute: Attribute
    x: int
    a: int
    p: int
    is_input: bool
    #: Whether ``tensor`` may alias a tensor the loader keeps and reads again (the training cache,
    #: re-read every epoch): the collate must then batch a COPY, never a view a downstream in-place
    #: op could write through. One-pass loaders clear it, and their singletons batch as views.
    aliases_cache: bool = True


@dataclass(frozen=True)
class BatchDataItem:
    """Batch-level representation of multiple :class:`DataItem` objects."""

    name: list[str]
    tensor: torch.Tensor  # [B, ...]
    attribute: list[Attribute]
    x: list[int]
    a: list[int]
    p: list[int]
    is_input: bool

    def pin_memory(self) -> "BatchDataItem":
        """The batch with its tensor in page-locked memory, so the upload is a real DMA.

        ``torch``'s pinner walks tensors, mappings and sequences and hands anything else back
        untouched: without this method ``pin_memory: true`` reaches nothing and every upload
        stays a pageable copy the host has to wait for.
        """
        return replace(self, tensor=self.tensor.pin_memory())


Sample: TypeAlias = dict[str, DataItem]
BatchSample: TypeAlias = dict[str, BatchDataItem]


def _batch_tensor(items: list[DataItem]) -> torch.Tensor:
    """The batch tensor: a view for a singleton that aliases no re-read cache, a stacked copy else.

    ``torch.stack`` copies, and on the ``batch_size=1`` evaluation path that copy is a whole volume
    per case, outside the memory budget's sizing. The view stays in the main process: a worker's
    batch travels by STORAGE, and a patch-view's storage is the whole resident case.
    """
    if len(items) == 1 and not items[0].aliases_cache and data.get_worker_info() is None:
        return items[0].tensor.unsqueeze(0)
    return torch.stack([it.tensor for it in items], dim=0)


def collate_konfai(batch: list[Sample]) -> BatchSample:
    """Collate KonfAI samples into the batch structure expected by the workflows."""
    batch_sample: BatchSample = {}
    for k in batch[0].keys():
        items = [b[k] for b in batch]
        batch_sample[k] = BatchDataItem(
            tensor=_batch_tensor(items),
            x=[it.x for it in items],
            a=[it.a for it in items],
            p=[it.p for it in items],
            attribute=[it.attribute for it in items],
            name=[it.name for it in items],
            is_input=items[0].is_input,
        )
    return batch_sample


def _return_freed_heap() -> None:
    """Hand the heap glibc is holding back to the kernel.

    Reading a case allocates a volume-sized buffer and frees it once the chain has landed its much
    smaller output. glibc raises its mmap threshold the first time such a block is freed, so every
    later one is served from the heap, which grows to its high-water mark and never shrinks: the
    process stays resident for what it READ rather than for what it KEPT. One trim once the fill is
    done gives it back. Measured on a 3 mm resample of a 1 mm case, held per case for the life of
    the run: 3.6 MiB becomes 1.2, which is what the cache holds. Absent outside glibc, where the
    question does not arise.
    """
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


class DatasetIter(data.Dataset):
    """Torch dataset view over KonfAI dataset managers and patch mappings."""

    def __init__(
        self,
        rank: int,
        data: dict[str, list[DatasetManager]],
        mapping: list[tuple[int, int, int]],
        groups_src: Mapping[str, Group | GroupMetric | GroupOut],
        inline_augmentations: bool,
        data_augmentations_list: list[DataAugmentationsList],
        patch_size: list[int] | None,
        overlap: OverlapSpec,
        buffer_size: int,
        apply_augmentations: bool = True,
        use_cache=True,
        batch_size: int = 1,
        single_pass: bool = False,
    ) -> None:
        self.rank = rank
        self.data = data
        self.mapping = mapping
        # A one-pass workflow never re-reads what a sample's tensor could alias, so its items may
        # batch as views; a training loader's items may alias the epoch-spanning cache and may not.
        self.single_pass = single_pass
        self.patch_size = patch_size
        self.overlap = overlap
        self.groups_src = groups_src
        self.apply_augmentations = apply_augmentations
        self.data_augmentations_list = data_augmentations_list if apply_augmentations else []
        self.use_cache = use_cache
        self.nb_dataset = len(data[next(iter(data.keys()))])
        self.buffer_size = buffer_size
        self._index_cache: list[int] = []
        self._statistics_warmed = False
        self._index_cache_lookup: set[int] = set()
        self.inline_augmentations = inline_augmentations
        self.has_augmented_samples = self.apply_augmentations and any(a > 0 for _, a, _ in mapping)
        self.read_order = PatchReadOrder(mapping, batch_size)

    def get_patch_config(self) -> tuple[list[int] | None, OverlapSpec]:
        return self.patch_size, self.overlap

    def to(self, device: int):
        for _group_src, _group_dest, chain in _chains(self.groups_src):
            chain.to(device)
        for data_augmentations in self.data_augmentations_list:
            for data_augmentation in data_augmentations.data_augmentations:
                data_augmentation.to(device)

    def get_dataset_from_index(self, group_dest: str, index: int) -> DatasetManager:
        return self.data[group_dest][index]

    def reset_augmentation(self, label):
        if self.inline_augmentations and self.has_augmented_samples and len(self.data_augmentations_list) > 0:
            for index in range(self.nb_dataset):
                # Augmentation objects are shared across destination groups AND across the train and
                # validation loaders, so the per-case draw is cached by the manager's own augmentation
                # index (globally unique, offset for validation), not the loader-local position: else a
                # validation case would reset (and reuse) a train case's draw and folded shape.
                case_index = next(iter(self.data.values()))[index].index
                for data_augmentations in self.data_augmentations_list:
                    for data_augmentation in data_augmentations.data_augmentations:
                        data_augmentation.reset_state(case_index)
                for _group_src, group_dest, _chain in _chains(self.groups_src):
                    self.data[group_dest][index].unload_augmentation()
                    self.data[group_dest][index].reset_augmentation(reset_state=False)
            self.load(label + " Augmentation")

    def load(self, label: str):
        if self.use_cache:
            memory_init = get_memory()

            indexs = list(range(self.nb_dataset))
            if len(indexs) > 0:
                memory_lock = threading.Lock()

                def desc(i: int = 0):
                    return (
                        f"Caching {label}: "
                        f"{get_memory_info()} | "
                        f"{memory_forecast(memory_init, i, self.nb_dataset)} | "
                        f"{get_cpu_info()}"
                    )

                pbar = tqdm.tqdm(total=len(indexs), desc=desc(), leave=False)
                stop_event = threading.Event()

                def process(index):
                    if stop_event.is_set():
                        return
                    self._load_data(index)
                    with memory_lock:
                        pbar.set_description(desc(pbar.n + 1))
                        pbar.update(1)

                cpu_count = os.cpu_count() or 1
                try:
                    with ThreadPoolExecutor(max_workers=_cache_worker_count(cpu_count, device_count())) as executor:
                        future_to_index = {executor.submit(process, index): index for index in indexs}
                        for fut in as_completed(future_to_index):
                            index = future_to_index[fut]
                            try:
                                fut.result()
                            except Exception as e:
                                stop_event.set()
                                for f in future_to_index:
                                    f.cancel()
                                tb = traceback.format_exc()
                                raise RuntimeError(
                                    f"Error while caching {label} (index={index})\n"
                                    f"{type(e).__name__}: {e}\n\n"
                                    f"Traceback (worker):\n{tb}"
                                ) from e

                except KeyboardInterrupt:
                    stop_event.set()
                    try:
                        for f in future_to_index:
                            f.cancel()
                    except Exception:  # nosec B110
                        pass
                    raise
                finally:
                    pbar.close()
                    _return_freed_heap()
        else:
            self._warm_stream_statistics(label)

    def _warm_stream_statistics(self, label: str) -> None:
        """Read every streamed case's disk statistics here, where one process holds them for the run.

        The memo lives on the manager, so a DataLoader worker forked for an epoch starts without it
        and scans the case again: with W workers and E epochs a chain wanting a whole-volume mean
        reads the cohort W x E times over. The scan is the same read whoever makes it, so making it
        once, before the fork, is the whole fix. A chain wanting no statistic costs a plan here.
        """
        if self._statistics_warmed:
            return  # the statistic describes the case as stored: one pass answers every epoch
        self._statistics_warmed = True
        # Every (case, copy) the epoch will ask for. A copy's statistic is its own: measured here it
        # reaches every worker through the fork, where measured in a worker it dies with it.
        copies: dict[int, set[int]] = {}
        for case, copy, _patch in self.mapping:
            copies.setdefault(case, set()).add(copy)
        work = [
            (self.data[group_dest][case], copy)
            for _group_src, group_dest, _chain in _chains(self.groups_src)
            for case, drawn in sorted(copies.items())
            for copy in sorted(drawn)
        ]
        if not work:
            return
        pbar = tqdm.tqdm(total=len(work), desc=f"Scanning {label}: {get_cpu_info()}", leave=False)
        try:
            with ThreadPoolExecutor(max_workers=_cache_worker_count(os.cpu_count() or 1, device_count())) as executor:
                futures = [
                    executor.submit(manager.warm_stream_statistics, copy, self.apply_augmentations)
                    for manager, copy in work
                ]
                for fut in as_completed(futures):
                    fut.result()
                    pbar.update(1)
        finally:
            pbar.close()

    def _load_data(self, index: int, augmentation_index: int | None = None) -> bool:
        loaded = False
        for group_src, group_dest, _chain in _chains(self.groups_src):
            loaded |= self.load_data(group_src, group_dest, index, augmentation_index)
        if loaded and index not in self._index_cache_lookup:
            self._index_cache.append(index)
            self._index_cache_lookup.add(index)
        return loaded

    def load_data(self, group_src: str, group_dest: str, index: int, augmentation_index: int | None = None) -> bool:
        item = self.data[group_dest][index]
        if augmentation_index is not None and item.can_stream_patch(augmentation_index, self.apply_augmentations):
            return False
        try:
            item.load(
                self.groups_src[group_src][group_dest].transforms,
                self.data_augmentations_list,
                load_augmentations=self.apply_augmentations and not self.inline_augmentations,
            )
        except Exception as e:
            raise RuntimeError(
                f"Error while loading data "
                f"(group_src={group_src}, group_dest={group_dest}, "
                f"index={index}, name={item.name}) : "
                f"{type(e).__name__}: {e}"
            ) from e
        return True

    def _unload_data(self, index: int) -> None:
        if index in self._index_cache_lookup:
            self._index_cache_lookup.remove(index)
            self._index_cache.remove(index)
        for _group_src, group_dest, _chain in _chains(self.groups_src):
            self.unload_data(group_dest, index)

    def unload_data(self, group_dest: str, index: int) -> None:
        return self.data[group_dest][index].unload()

    def _say_why_a_case_is_materialized(self, case: int, a: int) -> None:
        """Name the stage that costs a case its whole volume, once per process.

        The check above has already resolved the plan, so the refusal is read from it rather than
        asked for: naming it costs a dictionary lookup, and the flag makes it one boolean test per
        item after the first. What a reader can act on is the chain, so the chain is what it names.
        """
        global _said_why_a_case_is_materialized
        _said_why_a_case_is_materialized = True
        for _group_src, group_dest, _chain in _chains(self.groups_src):
            refusal = self.data[group_dest][case].stream_refusal(a, self.apply_augmentations)
            if refusal is None:
                continue
            print(
                f"[KonfAI] {group_dest}: a patch cannot be read as a region, so a case is materialized"
                f" whole to serve its patches. {refusal}"
            )
            print(
                f"[KonfAI] {group_dest}: declare the statistic on the stage that wants it, or move that"
                " stage ahead of the one that changes the values, or cut the chain with a Save: each of"
                " the three leaves the statistic describing what the stage is handed, which is what a"
                " region needs to be read on its own."
            )

    def _say_what_streaming_reads(self, case: int, a: int) -> None:
        """Say what a streamed case costs in reads, once per process.

        A patch pulls the window its chain needs, not the patch: a resample widens it on every axis
        and a 2.5D stack widens it again, so a case can be read many times over in one pass. Nothing
        refuses on the figure, because materializing the case instead is what a shuffled order over a
        cohort makes expensive. It is said so a reader can act on the budget, the patch or the chain.
        """
        global _said_what_streaming_reads
        _said_what_streaming_reads = True
        for _group_src, group_dest, chain in _chains(self.groups_src):
            manager = self.data[group_dest][case]
            factor = manager.streamed_read_amplification(a, chain.is_input, self.apply_augmentations)
            if factor is None or factor < 2.0:
                continue
            print(
                f"[KonfAI] {group_dest}: streaming reads this case {factor:.1f} times over per pass."
                " A resample and a 2.5D stack each widen what a patch pulls. A 'shuffle_window' keeps"
                " a few cases under the reader, so neighbouring patches fall on windows already in"
                " hand, and it draws a batch from those cases rather than from the cohort; a larger"
                " patch, a coarser target spacing or a budget that fits the case cost nothing in how"
                " the batches are drawn."
            )

    def _declare_case_reads(self, index: int) -> None:
        """Tell each group's store the patches this process will read of the case ``index`` enters,
        in the order it will read them: once per case, at the first patch of it that arrives."""
        entries = self.read_order.entering(index)
        if entries is None:
            return
        case = self.mapping[index][0]
        for _group_src, group_dest, chain in _chains(self.groups_src):
            self.data[group_dest][case].plan_patch_reads(entries, chain.is_input, self.apply_augmentations)

    def __len__(self) -> int:
        return len(self.mapping)

    def __getitem__(self, index: int) -> Sample:
        sample: Sample = {}
        x, a, p = self.mapping[index]
        needs_full_load = any(
            not self.data[group_dest][x].can_stream_patch(a, self.apply_augmentations)
            for _group_src, group_dest, _chain in _chains(self.groups_src)
        )
        if needs_full_load and not _said_why_a_case_is_materialized:
            self._say_why_a_case_is_materialized(x, a)
        elif not needs_full_load and not _said_what_streaming_reads:
            self._say_what_streaming_reads(x, a)
        if x not in self._index_cache_lookup and needs_full_load:
            if len(self._index_cache) >= self.buffer_size and not self.use_cache:
                self._unload_data(self._index_cache[0])
            self._load_data(x, a)

        self._declare_case_reads(index)

        for _group_src, group_dest, chain in _chains(self.groups_src):
            dataset = self.data[group_dest][x]
            sample[f"{group_dest}"] = DataItem(
                dataset.name,
                dataset.get_data(p, a, chain.patch_transforms, chain.is_input, self.apply_augmentations),
                dataset.cache_attributes[a],
                x,
                a,
                p,
                chain.is_input,
                aliases_cache=not self.single_pass,
            )
        return sample
