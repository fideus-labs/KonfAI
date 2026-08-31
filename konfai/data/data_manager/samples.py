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


def collate_konfai(batch: list[Sample]) -> BatchSample:
    """Collate KonfAI samples into the batch structure expected by the workflows."""
    batch_sample: BatchSample = {}
    for k in batch[0].keys():
        items = [b[k] for b in batch]
        batch_sample[k] = BatchDataItem(
            tensor=torch.stack([it.tensor for it in items], dim=0),
            x=[it.x for it in items],
            a=[it.a for it in items],
            p=[it.p for it in items],
            attribute=[it.attribute for it in items],
            name=[it.name for it in items],
            is_input=items[0].is_input,
        )
    return batch_sample


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
    ) -> None:
        self.rank = rank
        self.data = data
        self.mapping = mapping
        self.patch_size = patch_size
        self.overlap = overlap
        self.groups_src = groups_src
        self.apply_augmentations = apply_augmentations
        self.data_augmentations_list = data_augmentations_list if apply_augmentations else []
        self.use_cache = use_cache
        self.nb_dataset = len(data[next(iter(data.keys()))])
        self.buffer_size = buffer_size
        self._index_cache: list[int] = []
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
            )
        return sample
