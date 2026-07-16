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

"""Dataset assembly, subset selection, and dataloader orchestration for KonfAI."""

import math
import os
import random
import re
import threading
import traceback
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import partial
from typing import TypeAlias, cast

import numpy as np
import torch
import tqdm
from torch.cuda import device_count
from torch.utils import data
from torch.utils.data import DataLoader, Sampler

from konfai import konfai_root, konfai_state
from konfai.data.augmentation import DataAugmentationsList
from konfai.data.patching import DatasetManager, DatasetPatch
from konfai.data.transform import LocalityKind, Transform, TransformInverse, TransformLoader
from konfai.utils.config import config
from konfai.utils.dataset import Attribute, Dataset
from konfai.utils.errors import ConfigError, DatasetManagerError
from konfai.utils.runtime import (
    State,
    available_memory_bytes,
    get_cpu_info,
    get_memory,
    get_memory_info,
    memory_forecast,
)
from konfai.utils.utils import SUPPORTED_EXTENSIONS, split_path_spec

# A cached case is a float32 tensor (torch's default dtype, and the default TensorCast's target), so
# bytes are counted at 4/element from the header shape alone -- not the on-disk dtype, and without
# modelling transforms that shrink or grow the cached tensor.
_CACHE_ELEMENT_BYTES = 4

# Fraction of the detected node memory an ``"auto"`` budget offers the cache; the rest is reserved for
# the model's optimizer/gradient state, DataLoader worker copies, CUDA pinned staging buffers, and
# allocator slack. Caching runs with zero DataLoader workers, so a fifth of the node held back is ample.
_AUTO_MEMORY_SAFETY_FRACTION = 0.8

# Decimal (10^n) and binary (2^n) suffixes; "" / "b" are bytes. Case is folded before lookup.
_MEMORY_UNIT_BYTES: dict[str, int] = {
    "": 1, "b": 1,
    "k": 10**3, "kb": 10**3, "kib": 2**10,
    "m": 10**6, "mb": 10**6, "mib": 2**20,
    "g": 10**9, "gb": 10**9, "gib": 2**30,
    "t": 10**12, "tb": 10**12, "tib": 2**40,
}  # fmt: skip


def _format_gib(num_bytes: float) -> str:
    return f"{num_bytes / 2**30:.2f} GiB"


def _parse_memory_budget_bytes(value: str | float) -> int:
    """Parse an explicit memory budget to bytes: a bare number is GiB, a string carries its own unit.

    KonfAI reports RAM in GiB throughout, so an unadorned ``24`` reads as ``24 GiB`` -- whether it
    arrives as a number or, through the YAML binding, as the string ``"24"``. A string may name its
    unit -- decimal ``GB``/``MB`` (10^n) or binary ``GiB``/``MiB`` (2^n), case-insensitive, optional
    space (``"24GB"``, ``"32 GiB"``, ``"512mb"``); ``"b"`` means bytes. ``"auto"`` is resolved by the
    caller, not here.
    """
    if not isinstance(value, str):
        if float(value) <= 0:
            raise ConfigError(
                f"memory_budget: {value!r} must be a positive size.",
                "Use a positive number in GiB (e.g. 24), a unit string ('24GB'), 'auto', or None.",
            )
        return int(float(value) * 2**30)
    match = re.fullmatch(r"\s*(?P<number>[0-9]*\.?[0-9]+)\s*(?P<unit>[a-z]*)\s*", value.lower())
    if match is not None and match.group("unit") in _MEMORY_UNIT_BYTES:
        if float(match.group("number")) <= 0:
            raise ConfigError(
                f"memory_budget: '{value}' must be a positive size.",
                "Use a positive number in GiB (e.g. 24), a unit string ('24GB'), 'auto', or None.",
            )
        unit = match.group("unit")
        # A bare numeric string is the YAML face of a bare number: GiB, not bytes.
        factor = 2**30 if unit == "" else _MEMORY_UNIT_BYTES[unit]
        return int(float(match.group("number")) * factor)
    raise ConfigError(
        f"memory_budget: '{value}' is not a valid memory size.",
        "Use a number in GiB (e.g. 24), a unit string ('24GB', '32GiB', '512MB'), 'auto', or None to "
        "keep 'use_cache' as given.",
    )


def _cache_worker_count(cpu_count: int, device_count: int) -> int:
    """Number of caching threads: CPUs shared across devices, but never below one."""
    divisor = device_count if device_count > 0 else 1
    return max(1, cpu_count // divisor)


def _check_patch_transform_locality(transform: Transform, group_src: str, group_dest: str) -> None:
    """Reject a transform whose per-patch result cannot equal its case-level result.

    Only POINTWISE and GLOBAL_STAT are correct on one patch (per-patch GLOBAL_STAT means: derive the
    statistic from this patch; use ``lazy=True`` case-level to feed it the volume's). The messages in
    ``reasons`` below say why each other kind is rejected. Probed with an empty ``Attribute`` (config
    time has no case), so an image-decided kind answers WHOLE_VOLUME -- the right answer here.
    """
    kind = transform.patch_locality(Attribute()).kind
    if kind in (LocalityKind.POINTWISE, LocalityKind.GLOBAL_STAT):
        return
    name = type(transform).__name__
    location = f"{konfai_root()}.Dataset.groups_src.{group_src}.groups_dest.{group_dest}"
    reasons = {
        LocalityKind.HALO: (
            f"'{name}' reads a neighbourhood around each voxel, so run per-patch it would see no data"
            " beyond the patch border and corrupt every patch edge. Running it per-patch is not"
            " supported yet."
        ),
        LocalityKind.ORIENTATION: (
            f"'{name}' reorients its input: applied to one patch it reorients that patch about its own"
            " extent, which is not the whole volume reoriented and then cut into patches."
        ),
        LocalityKind.CROP: (
            f"'{name}' crops its input to a box measured on the whole volume: applied to one patch it"
            " crops that patch about its own extent, and cuts the patch grid predictions are"
            " reassembled onto down to what is left."
        ),
        LocalityKind.RESCALE: (
            f"'{name}' resamples its input: applied to one patch it rescales that patch about its own"
            " extent and changes the patch grid predictions are reassembled onto."
        ),
        LocalityKind.WHOLE_VOLUME: f"'{name}' needs the whole volume.",
    }
    raise ConfigError(
        f"{location}.patch_transforms: {reasons[kind]}",
        f"Move '{name}' to {location}.transforms, where it runs once on the whole volume.",
    )


def _check_patch_transform_shape(transform: Transform, group_src: str, group_dest: str) -> None:
    """Reject a patch_transform that resizes the patch it is handed.

    The patch grid is folded from the CASE-level ``transforms`` only (``DatasetManager``), so a
    patch_transform that changes the spatial shape hands back a patch the batch cannot collate and the
    ``Accumulator`` cannot write onto the grid. ``_check_patch_transform_locality`` above takes the
    transform at its word; this is the structural check, and it is asked of ``transform_shape`` -- the
    contract every transform already owes the patch planner -- with distinct extents, so a swap or a
    resize of any single axis shows up. Only the SPATIAL shape is at stake: patching hands
    ``transform_shape`` the channel-stripped shape, so a transform that changes only the channel count
    (``OneHot``) is not caught here, and must not be -- the grid it feeds is spatial.

    Runs after the locality check, which is what makes the bare probe attribute safe: by here the
    transform is POINTWISE or GLOBAL_STAT, and the kinds whose ``transform_shape`` needs real geometry
    (``Resample`` reads ``Spacing``) have already been rejected.
    """
    spatial_shape = [7, 11, 13]
    shape = list(transform.transform_shape(group_src, "", list(spatial_shape), Attribute()))
    if shape == spatial_shape:
        return
    name = type(transform).__name__
    location = f"{konfai_root()}.Dataset.groups_src.{group_src}.groups_dest.{group_dest}"
    raise ConfigError(
        f"{location}.patch_transforms: '{name}' changes the spatial shape of its input"
        f" ({spatial_shape} -> {shape}), but a patch must keep the shape the patch grid cut it to.",
        f"Move '{name}' to {location}.transforms, where the patch grid is folded from its"
        " transform_shape(); a patch_transform must be spatially shape-preserving.",
    )


def _check_patch_transform_invertible(
    transform: Transform, case_transforms: list[Transform], group_src: str, group_dest: str
) -> None:
    """Reject a per-patch global statistic at prediction, whose inverse cannot be reconstructed.

    A ``GLOBAL_STAT`` transform is allowed in ``patch_transforms`` (see
    ``_check_patch_transform_locality``): run per patch it standardizes each patch by that patch's OWN
    statistic, which is what asking for it per-patch means -- correct, and the deliberate training use.
    But the per-patch statistic lives in the per-patch attribute scope and never reaches the case
    attribute, so at prediction the finalize inverse -- which seeds every patch from the CASE attribute
    and pops the statistic -- has nothing to pop. Nor could it: the reassembled volume was normalised
    patch by patch with different coefficients, so a single case-level inverse cannot un-apply it. Refuse
    here, at config time, rather than fail deep in the inverse with a bare ``NameError``.

    A case-level ``transforms`` entry that derives the SAME statistic rescues it: run once on the whole
    volume it caches that statistic on the case attribute (``Standardize(lazy=True)`` caches Mean/Std and
    applies nothing), which the per-patch inverse then inherits and pops. So the patch transform is only
    un-invertible when nothing case-level captures its statistic.

    Training-only use stays valid: the check is gated on the prediction state, where the inverse actually
    runs (``RESUME``/``TRAIN`` never invert patch_transforms, and evaluation drops them entirely).
    """
    if konfai_state() != str(State.PREDICTION):
        return
    if not (isinstance(transform, TransformInverse) and transform.apply_inverse):
        return
    locality = transform.patch_locality(Attribute())
    if locality.kind is not LocalityKind.GLOBAL_STAT:
        return
    if any(
        (case_locality := case.patch_locality(Attribute())).kind is LocalityKind.GLOBAL_STAT
        and locality.stat_keys <= case_locality.stat_keys
        for case in case_transforms
    ):
        return
    name = type(transform).__name__
    location = f"{konfai_root()}.Dataset.groups_src.{group_src}.groups_dest.{group_dest}"
    raise ConfigError(
        f"{location}.patch_transforms: '{name}' derives its statistic from each patch, but prediction"
        " must invert it and a per-patch statistic cannot be un-applied to the reassembled volume.",
        f"Capture the volume-global statistic case-level instead: put '{name}(lazy=True)' in"
        f" {location}.transforms (it traverses the whole volume, caches the statistic and applies"
        f" nothing), and keep '{name}()' in patch_transforms to consume it.",
    )


class GroupTransform:
    """Collection of transforms attached to one source-to-destination group path."""

    def __init__(
        self,
        transforms: dict[str, TransformLoader] | None = {
            "default|Normalize|Standardize|Unsqueeze|TensorCast|ResampleIsotropic|ResampleResize": TransformLoader()
        },
        patch_transforms: dict[str, TransformLoader] | None = {
            "default|Normalize|Standardize|Unsqueeze|TensorCast|ResampleIsotropic|ResampleResize": TransformLoader()
        },
        is_input: bool = True,
    ) -> None:
        self._transforms = transforms
        self._patch_transforms = patch_transforms
        self.transforms: list[Transform] = []
        self.patch_transforms: list[Transform] = []
        self.is_input = is_input

    def prepare(self, group_src: str, group_dest: str) -> None:
        self.transforms = []
        self.patch_transforms = []
        if self._transforms is not None:
            for classpath, transform_loader in self._transforms.items():
                transform = transform_loader.get_transform(
                    classpath,
                    konfai_args=f"{konfai_root()}.Dataset.groups_src.{group_src}.groups_dest.{group_dest}.transforms",
                )
                self.transforms.append(transform)
        if self._patch_transforms is not None:
            for classpath, transform_loader in self._patch_transforms.items():
                transform = transform_loader.get_transform(
                    classpath,
                    konfai_args=f"{konfai_root()}.Dataset.groups_src.{group_src}"
                    f".groups_dest.{group_dest}.patch_transforms",
                )
                _check_patch_transform_locality(transform, group_src, group_dest)
                _check_patch_transform_shape(transform, group_src, group_dest)
                _check_patch_transform_invertible(transform, self.transforms, group_src, group_dest)
                self.patch_transforms.append(transform)

    def set_datasets(self, datasets: list[Dataset]) -> None:
        for transform in self.transforms:
            transform.set_datasets(datasets)
        for transform in self.patch_transforms:
            transform.set_datasets(datasets)

    def to(self, device: int):
        for transform in self.transforms:
            transform.to(device)
        for transform in self.patch_transforms:
            transform.to(device)

    def __str__(self) -> str:
        params = {"transforms": self.transforms, "patch_transforms": self.patch_transforms}
        return str(params)

    def __repr__(self) -> str:
        return str(self)


class GroupTransformMetric(GroupTransform):
    """Metric-specific group transform that omits patch-time transforms."""

    def __init__(
        self,
        transforms: dict[str, TransformLoader] = {
            "default|Normalize|Standardize|Unsqueeze|TensorCast|ResampleIsotropic|ResampleResize": TransformLoader()
        },
    ):
        super().__init__(transforms, {})


class Group(dict[str, GroupTransform]):
    """Mapping of destination group names to transform pipelines."""

    def __init__(
        self,
        groups_dest: dict[str, GroupTransform] = {"default|Labels": GroupTransform()},
    ):
        super().__init__(groups_dest)


class GroupMetric(dict[str, GroupTransformMetric]):
    """Metric-oriented variant of :class:`Group` used during evaluation."""

    def __init__(
        self,
        groups_dest: dict[str, GroupTransformMetric] = {"default|group_dest": GroupTransformMetric()},
    ):
        super().__init__(groups_dest)


class WindowedCaseSampler(Sampler[int]):
    """Locality-aware training order: shuffle cases, window them, shuffle patches within each window.

    ``DatasetIter`` loads each non-streamable case into a FIFO buffer, so a global patch shuffle
    reloads a volume repeatedly -- once per patch that lands after an eviction. Keeping only
    ``window`` cases in play at a time — their patches shuffled together, emitted before advancing —
    reads each volume ~once. ``window`` is the decorrelation knob: ``1`` is perfect locality, and
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
    ) -> None:
        self.mapping = mapping
        self.shuffle = shuffle
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
        left shift onto the workers that finished -- a case landing on two of them, each reading the
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
        # so streams of equal length are not something `_partitions` can always hand over -- one case
        # of 200 patches beside ten of 2 is longer on its own than a quarter of the epoch.
        #
        # So a short stream runs out and the ones still going shift onto the workers that finished:
        # a case lands on two of them and each reads its volume. The epoch stays exact and the reads
        # stay close to 1x -- far below the redundant reads of no window at all. Padding the streams
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
            return iter(range(len(self.mapping)))
        if self.window is None:
            return iter(torch.randperm(len(self.mapping)).tolist())
        return iter(self._windowed_order())

    def __len__(self) -> int:
        # One epoch is one pass over the mapping -- windowing chooses the ORDER, not the size. This is
        # what keeps the ranks in step: `Data._split` gives them equal-length shards, but not equal
        # cases, so any length read from the per-rank cases (their partitions, or even whether the
        # window engages at all) would differ and hang DDP's collectives.
        return len(self.mapping)


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
        groups_src: Mapping[str, Group | GroupMetric],
        inline_augmentations: bool,
        data_augmentations_list: list[DataAugmentationsList],
        patch_size: list[int] | None,
        overlap: int | None,
        buffer_size: int,
        apply_augmentations: bool = True,
        use_cache=True,
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

    def get_patch_config(self) -> tuple[list[int] | None, int | None]:
        return self.patch_size, self.overlap

    def to(self, device: int):
        for group_src in self.groups_src:
            for group_dest in self.groups_src[group_src]:
                self.groups_src[group_src][group_dest].to(device)
        for data_augmentations in self.data_augmentations_list:
            for data_augmentation in data_augmentations.data_augmentations:
                data_augmentation.to(device)

    def get_dataset_from_index(self, group_dest: str, index: int) -> DatasetManager:
        return self.data[group_dest][index]

    def reset_augmentation(self, label):
        if self.inline_augmentations and self.has_augmented_samples and len(self.data_augmentations_list) > 0:
            for index in range(self.nb_dataset):
                # Augmentation objects are shared across destination groups, so the
                # random parameters for a case are drawn once here; each group then
                # rebuilds its patch grid from that single draw, keeping every group
                # of the case aligned on the same transform.
                for data_augmentations in self.data_augmentations_list:
                    for data_augmentation in data_augmentations.data_augmentations:
                        data_augmentation.reset_state(index)
                for group_src in self.groups_src:
                    for group_dest in self.groups_src[group_src]:
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
        for group_src in self.groups_src:
            for group_dest in self.groups_src[group_src]:
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
        for group_src in self.groups_src:
            for group_dest in self.groups_src[group_src]:
                self.unload_data(group_dest, index)

    def unload_data(self, group_dest: str, index: int) -> None:
        return self.data[group_dest][index].unload()

    def __len__(self) -> int:
        return len(self.mapping)

    def __getitem__(self, index: int) -> Sample:
        sample: Sample = {}
        x, a, p = self.mapping[index]
        needs_full_load = any(
            not self.data[group_dest][x].can_stream_patch(a, self.apply_augmentations)
            for group_src in self.groups_src
            for group_dest in self.groups_src[group_src]
        )
        if x not in self._index_cache_lookup and needs_full_load:
            if len(self._index_cache) >= self.buffer_size and not self.use_cache:
                self._unload_data(self._index_cache[0])
            self._load_data(x, a)

        for group_src in self.groups_src:
            for group_dest in self.groups_src[group_src]:
                dataset = self.data[group_dest][x]
                sample[f"{group_dest}"] = DataItem(
                    dataset.name,
                    dataset.get_data(
                        p,
                        a,
                        self.groups_src[group_src][group_dest].patch_transforms,
                        self.groups_src[group_src][group_dest].is_input,
                        self.apply_augmentations,
                    ),
                    dataset.cache_attributes[a],
                    x,
                    a,
                    p,
                    self.groups_src[group_src][group_dest].is_input,
                )
        return sample


class Subset:
    def __init__(
        self,
        subset: str | list[int] | list[str] | None = None,
        shuffle: bool = True,
        shuffle_window: int | None = None,
    ) -> None:
        self.subset = subset
        self.shuffle = shuffle
        self.shuffle_window = shuffle_window

    @staticmethod
    def _read_names_from_file(filename: str) -> list[str]:
        with open(filename) as f:
            return [name.strip() for name in f if name.strip()]

    def requires_infos(self) -> bool:
        """Return whether this subset implementation needs per-sample metadata."""
        return self.__class__.__call__ is not Subset.__call__

    @staticmethod
    def _is_slice_selector(subset: str) -> bool:
        start, sep, end = subset.partition(":")
        if sep == "":
            return False
        return start.lstrip("-").isdigit() and end.lstrip("-").isdigit()

    def _resolve_selector(self, subset: str | int, names: list[str]) -> tuple[set[int], bool]:
        size = len(names)
        name_to_index = {name: i for i, name in enumerate(names)}

        if isinstance(subset, int):
            return {subset}, False
        if subset.startswith("~"):
            excluded = subset[1:]
            if os.path.exists(excluded):
                exclude_names = set(self._read_names_from_file(excluded))
                return {i for i, name in enumerate(names) if name in exclude_names}, True
            if excluded in name_to_index:
                return {name_to_index[excluded]}, True
            return set(), True
        if os.path.exists(subset):
            selected_names = set(self._read_names_from_file(subset))
            return {i for i, name in enumerate(names) if name in selected_names}, False
        if self._is_slice_selector(subset):
            start, _, end = subset.partition(":")
            r = np.clip(
                np.asarray([int(start), int(end)]),
                0,
                size,
            )
            return set(range(int(r[0]), int(r[1]))), False
        if subset in name_to_index:
            return {name_to_index[subset]}, False
        return set(), False

    def _get_index(self, subset: str | int, names: list[str]) -> list[int]:
        index, is_exclusion = self._resolve_selector(subset, names)
        if is_exclusion:
            return [i for i in range(len(names)) if i not in index]
        return sorted(index)

    def __call__(self, names: list[str], infos: dict[str, tuple[list[int], Attribute]]) -> set[str]:
        names = sorted(names)
        size = len(names)

        if self.subset is None:
            index = list(range(0, size))
        elif isinstance(self.subset, list):
            if len(self.subset) == 0:
                index = []
            else:
                include_index: set[int] = set()
                exclude_index: set[int] = set()
                has_include = False
                for s in self.subset:
                    resolved_index, is_exclusion = self._resolve_selector(s, names)
                    if is_exclusion:
                        exclude_index.update(resolved_index)
                    else:
                        include_index.update(resolved_index)
                        has_include = True
                index_set = include_index if has_include else set(range(size))
                index = sorted(index_set.difference(exclude_index))
        else:
            index = self._get_index(self.subset, names)
        return {names[i] for i in index}

    def __str__(self):
        return f"Subset : {self.subset} shuffle : {self.shuffle} shuffle_window : {self.shuffle_window}"


class TrainSubset(Subset):
    def __init__(
        self,
        subset: str | list[int] | list[str] | None = None,
        shuffle: bool = True,
        shuffle_window: int | None = None,
    ) -> None:
        super().__init__(subset, shuffle, shuffle_window)


class PredictionSubset(Subset):
    def __init__(self, subset: str | list[int] | list[str] | None = None) -> None:
        super().__init__(subset, False, None)


class Data(ABC):
    """Abstract base class shared by training, prediction, and evaluation datasets."""

    @staticmethod
    def _configured_transform_requires_single_process(classpath: str) -> bool:
        for transform_name in classpath.split("|"):
            candidate = transform_name.split(":")[-1].split(".")[-1].split("/")[0]
            if candidate == "KonfAIInference":
                return True
        return False

    @classmethod
    def _groups_require_single_process_loading(cls, groups_src: Mapping[str, Group | GroupMetric]) -> bool:
        for group in groups_src.values():
            for group_transform in group.values():
                for configured_transforms in (group_transform._transforms, group_transform._patch_transforms):
                    if configured_transforms is None:
                        continue
                    if any(
                        cls._configured_transform_requires_single_process(classpath)
                        for classpath in configured_transforms
                    ):
                        return True
        return False

    @staticmethod
    def _read_names_from_file(filename: str) -> list[str]:
        with open(filename) as f:
            return [name.strip() for name in f if name.strip()]

    @classmethod
    def _resolve_name_selectors(cls, selectors: list[str]) -> set[str]:
        resolved_names: set[str] = set()
        for selector in selectors:
            if os.path.exists(selector):
                resolved_names.update(cls._read_names_from_file(selector))
            else:
                resolved_names.add(selector)
        return resolved_names

    @abstractmethod
    def __init__(
        self,
        dataset_filenames: list[str],
        groups_src: Mapping[str, Group | GroupMetric],
        patch: DatasetPatch | None,
        use_cache: bool,
        subset: Subset,
        batch_size: int,
        validation: float | str | list[int] | list[str] | None,
        validation_augmentations: bool,
        inline_augmentations: bool,
        data_augmentations_list: dict[str, DataAugmentationsList],
        num_workers: int | None,
        pin_memory: bool,
        prefetch_factor: int | None,
        persistent_workers: bool | None,
        memory_budget: str | float | None,
    ) -> None:
        self.dataset_filenames = dataset_filenames
        self.subset = subset
        self.groups_src = groups_src
        self.patch = patch
        self.validation = validation
        self.validation_augmentations = validation_augmentations
        self.data_augmentations_list = data_augmentations_list
        self.batch_size = batch_size
        self.inline_augmentations = inline_augmentations
        self.memory_budget = memory_budget
        self.requires_single_process_loading = self._groups_require_single_process_loading(groups_src)

        # A window keeps ``shuffle_window`` cases resident, so the FIFO buffer must be at least that
        # large or a window would evict its own cases before their patches are consumed. Unwindowed,
        # one batch plus the case being read is all a loader ever holds at once.
        window = subset.shuffle_window
        self._buffer_size = batch_size + 1 if window is None else max(batch_size + 1, window)
        self._num_workers = num_workers
        self._pin_memory = pin_memory
        self._prefetch_factor = prefetch_factor
        self._persistent_workers = persistent_workers
        # ``memory_budget`` may later override ``use_cache`` (once the dataset size is known, in
        # ``get_data``), which reshapes the loader; both paths funnel through the same builder.
        self._configure_data_loading(use_cache)
        self.data: list[list[dict[str, list[DatasetManager]]]] = []
        self.mapping: list[list[list[tuple[int, int, int]]]] = []
        self.datasets: dict[str, Dataset] = {}
        self._prepared_data: dict[str, list[DatasetManager]] | None = None
        self._prepared_validation_data: dict[str, list[DatasetManager]] | None = None
        self._prepared_mapping: list[tuple[int, int, int]] = []
        self._prepared_validation_mapping: list[tuple[int, int, int]] = []
        self._prepared_train_names: list[str] = []
        self._prepared_validation_names: list[str] = []

    def _configure_data_loading(self, use_cache: bool) -> None:
        """Build the loader from the cache regime: the DatasetIter factory and the worker settings.

        Called once from ``__init__`` with the declared ``use_cache`` and, when a ``memory_budget``
        overrides it, again from ``get_data`` with the derived value. Caching preloads every case up
        front, so it defaults to zero DataLoader workers; the streaming/buffer path spins workers up.
        """
        self.use_cache = use_cache
        self.datasetIter = partial(
            DatasetIter,
            groups_src=self.groups_src,
            inline_augmentations=self.inline_augmentations,
            patch_size=self.patch.patch_size if self.patch is not None else None,
            overlap=self.patch.overlap if self.patch is not None else None,
            buffer_size=self._buffer_size,
            use_cache=use_cache,
        )
        resolved_num_workers = self._num_workers
        if self.requires_single_process_loading:
            resolved_num_workers = 0
        elif resolved_num_workers is None:
            resolved_num_workers = max(1, min(os.cpu_count() or 1, 4)) if not use_cache else 0
        self.resolved_num_workers: int = resolved_num_workers
        self.dataLoader_args: dict[str, object] = {
            "num_workers": resolved_num_workers,
            "pin_memory": self._pin_memory,
            "collate_fn": collate_konfai,
        }
        if resolved_num_workers > 0:
            self.dataLoader_args["prefetch_factor"] = 2 if self._prefetch_factor is None else self._prefetch_factor
            self.dataLoader_args["persistent_workers"] = (
                True if self._persistent_workers is None else self._persistent_workers
            )

    def _estimate_cached_bytes(self) -> int:
        """Raw in-RAM size of the whole prepared dataset, from headers alone (no voxel read).

        Sums ``prod(shape) x 4`` over every case of every source group, once per COPY the cache holds:
        a cached case is its base tensor PLUS one per augmentation draw, which validation only makes
        when ``validation_augmentations``. See ``_CACHE_ELEMENT_BYTES``: this is an honest header-only
        estimate that ignores size-changing transforms (an augmentation's ``Mask`` included). It also
        counts the tensors themselves, not the allocator's arenas around them: those settle about a
        third higher (measured), which is over the "auto" safety fraction, so a dataset landing within
        a few percent of an "auto" budget can still be caching more than the budget names.
        """
        total = 0
        for prepared, copies in (
            (self._prepared_data, Data._get_nb_augmentation(self._get_data_augmentations(True))),
            (
                self._prepared_validation_data,
                Data._get_nb_augmentation(self._get_data_augmentations(self.validation_augmentations)),
            ),
        ):
            for managers in (prepared or {}).values():
                for manager in managers:
                    total += int(np.prod(manager.base_shape, dtype=np.int64)) * _CACHE_ELEMENT_BYTES * copies
        return total

    def _resolve_cache_regime(self, world_size: int) -> None:
        """Derive ``use_cache`` from ``memory_budget``, or leave the declared value untouched.

        ``None`` keeps today's behaviour exactly. Otherwise the cache is chosen iff the per-rank
        dataset (``dataset / world_size`` -- ``Data._split`` shards cases across ranks) fits the
        per-rank budget: an explicit budget is taken as declared per rank; ``"auto"`` divides the
        detected node memory (cgroup-capped) by the ranks sharing it, so its ``world_size`` cancels
        and it reduces to "does the whole dataset fit the node". The decision is logged once here --
        ``get_data`` runs on the launcher alone, before any worker is spawned.
        """
        if self.memory_budget is None:
            return
        world_size = max(1, world_size)
        n_cases = len(self._prepared_train_names) + len(self._prepared_validation_names)
        dataset_bytes = self._estimate_cached_bytes()
        per_rank_bytes = dataset_bytes / world_size

        if isinstance(self.memory_budget, str) and self.memory_budget.strip().lower() == "auto":
            node_bytes, source = available_memory_bytes()
            per_rank_budget = node_bytes * _AUTO_MEMORY_SAFETY_FRACTION / world_size
            budget_desc = f"auto: {_format_gib(node_bytes)} {source} x {_AUTO_MEMORY_SAFETY_FRACTION:.0%}, per-rank"
        else:
            per_rank_budget = float(_parse_memory_budget_bytes(self.memory_budget))
            budget_desc = f"{self.memory_budget!r}, per-rank"

        use_cache = per_rank_bytes <= per_rank_budget
        self._configure_data_loading(use_cache)

        decision = f"CACHE the whole dataset in RAM ({self.resolved_num_workers} loader workers)"
        if not use_cache:
            case_bytes = dataset_bytes / max(1, n_cases)
            decision = (
                f"STREAM/BUFFER, no cache; FIFO working set ~= {self._buffer_size} cases x "
                f"{_format_gib(case_bytes)} = {_format_gib(self._buffer_size * case_bytes)} per worker"
            )
        print(
            f"[KonfAI] memory_budget: dataset ~= {_format_gib(dataset_bytes)} over {n_cases} cases | "
            f"per-rank ~= {_format_gib(per_rank_bytes)} across {world_size} rank(s) | "
            f"budget {_format_gib(per_rank_budget)} ({budget_desc}) -> {decision}"
        )

    def _get_data_augmentations(self, apply_augmentations: bool = True) -> list[DataAugmentationsList]:
        return list(self.data_augmentations_list.values()) if apply_augmentations else []

    @staticmethod
    def _get_nb_augmentation(data_augmentations_list: list[DataAugmentationsList]) -> int:
        return max(int(np.sum([data_augmentation.nb for data_augmentation in data_augmentations_list]) + 1), 1)

    def _get_validation_mapping(self) -> list[tuple[int, int, int]]:
        if self.validation_augmentations:
            return self._prepared_validation_mapping
        return [entry for entry in self._prepared_validation_mapping if entry[1] == 0]

    def prepare(self) -> None:
        """Instantiate config-driven transforms and augmentations before runtime."""
        if self._prepared_data is not None and self._prepared_validation_data is not None:
            return

        model_have_input = False
        last_group_src: str | None = None
        for group_src in self.groups_src:
            last_group_src = group_src
            for group_dest in self.groups_src[group_src]:
                self.groups_src[group_src][group_dest].prepare(group_src, group_dest)
                model_have_input |= self.groups_src[group_src][group_dest].is_input

        if self.patch is not None:
            self.patch.init()

        if not model_have_input:
            raise DatasetManagerError(
                "At least one group must be defined with 'is_input: true' to provide input to the network."
            )

        if last_group_src is not None:
            for key, data_augmentations in self.data_augmentations_list.items():
                data_augmentations.prepare(key)
        self._prepare_datasets()

    def _resolve_dataset_sources(self) -> dict[str, list[tuple[str, bool]]]:
        datasets: dict[str, list[tuple[str, bool]]] = {}
        if self.dataset_filenames is None or len(self.dataset_filenames) == 0:
            raise DatasetManagerError("No dataset filenames were provided")
        self.datasets = {}
        for dataset_filename in self.dataset_filenames:
            if dataset_filename is None:
                raise DatasetManagerError(
                    "Invalid dataset entry: 'None' received.",
                    "Each dataset must be a valid path string (e.g., './Dataset/', './Dataset/:mha, "
                    "'./Dataset/:a:mha', './Dataset/:i:mha').",
                    "Please check your 'dataset_filenames' list for missing or null entries.",
                )
            filename, flag, file_format = split_path_spec(
                dataset_filename,
                default_format="mha",
                allowed_flags={"a", "i"},
                supported_extensions=SUPPORTED_EXTENSIONS,
            )
            append = flag != "i"

            if file_format.split("@", 1)[0] not in SUPPORTED_EXTENSIONS:
                raise DatasetManagerError(
                    f"Unsupported file format '{file_format}'.",
                    f"Supported extensions are: {', '.join(SUPPORTED_EXTENSIONS)}",
                )

            dataset = Dataset(filename, file_format)
            self.datasets[filename] = dataset
            for group in self.groups_src:
                if dataset.is_group_exist(group):
                    datasets.setdefault(group, []).append((filename, append))

        for group_src in self.groups_src:
            if group_src not in datasets:
                raise DatasetManagerError(
                    f"Group source '{group_src}' not found in any dataset.",
                    f"Dataset filenames provided: {self.dataset_filenames}",
                    f"Available groups across all datasets: "
                    f"{[f'{f} {d.get_group()}' for f, d in self.datasets.items()]}\n"
                    f"Please check that an entry in the dataset with the name '{group_src}' exists.",
                )

            for group_dest in self.groups_src[group_src]:
                self.groups_src[group_src][group_dest].set_datasets(list(self.datasets.values()))

        for _group_src, entries in datasets.items():
            for _key, data_augmentations in self.data_augmentations_list.items():
                data_augmentations.set_datasets([self.datasets[filename] for filename, _ in entries])
            break
        return datasets

    def _resolve_common_names(
        self,
        datasets: dict[str, list[tuple[str, bool]]],
    ) -> tuple[dict[str, dict[str, list[str]]], set[str]]:
        dataset_name: dict[str, dict[str, list[str]]] = {}
        subset_requires_infos = self.subset.requires_infos()
        dataset_info: dict[str, dict[str, dict[str, tuple[list[int], Attribute]]]] | None = (
            {} if subset_requires_infos else None
        )
        empty_infos: dict[str, tuple[list[int], Attribute]] = {}
        names: set[str] = set()
        for group in self.groups_src:
            names_by_group = set()
            dataset_name[group] = {}
            if dataset_info is not None:
                dataset_info[group] = {}
            for filename, _ in datasets[group]:
                group_names = self.datasets[filename].get_names(group)
                names_by_group.update(group_names)
                dataset_name[group][filename] = group_names
                if dataset_info is not None:
                    dataset_info[group][filename] = {
                        name: self.datasets[filename].get_infos(group, name) for name in group_names
                    }
            if len(names) == 0:
                names.update(names_by_group)
            else:
                names = names.intersection(names_by_group)
        if len(names) == 0:
            raise DatasetManagerError(
                f"No data was found for groups {list(self.groups_src.keys())}: although each group contains data "
                "from a dataset, there are no common dataset names shared across all groups, the intersection is empty."
            )

        subset_names: set[str] = set()
        for group in dataset_name:
            subset_names_bygroup: set[str] = set()
            for filename, append in datasets[group]:
                resolved_subset = self.subset(
                    dataset_name[group][filename],
                    dataset_info[group][filename] if dataset_info is not None else empty_infos,
                )
                if append:
                    subset_names_bygroup.update(resolved_subset)
                elif len(subset_names_bygroup) == 0:
                    subset_names_bygroup.update(resolved_subset)
                else:
                    subset_names_bygroup = subset_names_bygroup.intersection(resolved_subset)
            if len(subset_names) == 0:
                subset_names.update(subset_names_bygroup)
            else:
                subset_names = subset_names.intersection(subset_names_bygroup)

        if len(subset_names) == 0:
            raise DatasetManagerError(
                "All data entries were excluded by the subset filter.",
                f"Dataset entries found: {', '.join(names)}",
                f"Subset object applied: {self.subset}",
                f"Subset requested : {', '.join(subset_names)}",
                "None of the dataset entries matched the given subset.",
                "Please check your 'subset' configuration — it may be too restrictive or incorrectly formatted.",
                "Examples of valid subset formats:",
                "\tsubset: [0, 1]            # explicit indices",
                "\tsubset: [./A.txt, ./B.txt]# union of multiple files",
                "\tsubset: 0:10              # slice notation",
                "\tsubset: ./Validation.txt  # external file",
                "\tsubset: None              # to disable filtering",
            )
        return dataset_name, subset_names

    @staticmethod
    def _get_source_filename_by_group(
        dataset_name: dict[str, dict[str, list[str]]],
    ) -> dict[str, dict[str, str]]:
        source_filename_by_group: dict[str, dict[str, str]] = {}
        for group_src, filenames_by_group in dataset_name.items():
            source_filename_by_group[group_src] = {}
            for filename, group_names in filenames_by_group.items():
                for name in group_names:
                    source_filename_by_group[group_src].setdefault(name, filename)
        return source_filename_by_group

    def _get_case_entry_counts(
        self,
        names: list[str],
        dataset_name: dict[str, dict[str, list[str]]],
        data_augmentations_list: list[DataAugmentationsList],
    ) -> list[int]:
        if len(names) == 0:
            return []

        source_filename_by_group = self._get_source_filename_by_group(dataset_name)
        nb_augmentation = self._get_nb_augmentation(data_augmentations_list)
        nb_patch = [[0] * nb_augmentation for _ in names]

        for group_src in self.groups_src:
            for group_dest in self.groups_src[group_src]:
                datasets = [
                    DatasetManager(
                        i,
                        group_src,
                        group_dest,
                        name,
                        self.datasets[source_filename_by_group[group_src][name]],
                        patch=self.patch,
                        transforms=self.groups_src[group_src][group_dest].transforms,
                        data_augmentations_list=data_augmentations_list,
                    )
                    for i, name in enumerate(names)
                ]
                nb_patch = [[dataset.get_size(a) for a in range(nb_augmentation)] for dataset in datasets]

        return [int(np.sum(case_patch_counts)) for case_patch_counts in nb_patch]

    def _resolve_validation_indices(
        self,
        subset_names: list[str],
        case_entry_counts: list[int] | None = None,
    ) -> list[int]:
        index: list[int] = []
        if isinstance(self.validation, float):
            if self.validation <= 0 or self.validation >= 1:
                raise DatasetManagerError(
                    "Validation must be a float between 0 and 1.",
                    f"Received: {self.validation}",
                    "Example: validation = 0.2  # for a 20% validation split",
                )
            if case_entry_counts is None:
                raise DatasetManagerError("Internal error: missing case entry counts for float validation split.")
            threshold = math.floor(sum(case_entry_counts) * (1 - self.validation))
            cumulative = 0
            for dataset_index, count in enumerate(case_entry_counts):
                cumulative += count
                if cumulative > threshold:
                    index = list(range(dataset_index, len(subset_names)))
                    break
        elif isinstance(self.validation, str):
            if ":" in self.validation:
                index = list(range(int(self.validation.split(":")[0]), int(self.validation.split(":")[1])))
            elif os.path.exists(self.validation):
                validation_names = []
                with open(self.validation) as f:
                    for name in f:
                        validation_names.append(name.strip())
                index = [i for i, n in enumerate(subset_names) if n in validation_names]
            else:
                raise DatasetManagerError(
                    f"Invalid string value for 'validation': '{self.validation}'",
                    "Expected one of the following formats:",
                    "\t• A slice string like '0:10'",
                    "\t• A path to a text file listing validation sample names (e.g., './val.txt')",
                    "\t• A list of text files listing validation sample names",
                    "\t• A float between 0 and 1 (e.g., 0.2)",
                    "\t• A list of sample names or indices",
                    "The provided value is neither a valid slice nor a readable file.",
                    "Please fix your 'validation' setting in the configuration.",
                )
        elif isinstance(self.validation, list):
            if len(self.validation) == 0:
                index = []
            elif all(isinstance(item, int) for item in self.validation):
                index = cast(list[int], self.validation)
            elif all(isinstance(item, str) for item in self.validation):
                validation_name_set = self._resolve_name_selectors(cast(list[str], self.validation))
                index = [i for i, n in enumerate(subset_names) if n in validation_name_set]
            else:
                element_types = sorted({type(item).__name__ for item in self.validation})
                raise DatasetManagerError(
                    f"Invalid list type for 'validation': elements of type {element_types} are not supported.",
                    "Supported list element types are:",
                    "\t• int  → list of indices (e.g., [0, 1, 2])",
                    "\t• str  → list of sample names or file paths",
                    f"Received list: {self.validation}",
                )
        return index

    def _split_train_validation_names(
        self,
        subset_names: list[str],
        dataset_name: dict[str, dict[str, list[str]]],
    ) -> tuple[list[str], list[str]]:
        case_entry_counts = None
        dataset_size = len(subset_names)
        if isinstance(self.validation, float):
            case_entry_counts = self._get_case_entry_counts(
                subset_names,
                dataset_name,
                self._get_data_augmentations(True),
            )
            dataset_size = int(sum(case_entry_counts))

        index = self._resolve_validation_indices(subset_names, case_entry_counts)
        index_set = set(index)
        validation_names = [name for i, name in enumerate(subset_names) if i in index_set]
        validation_names_set = set(validation_names)
        train_names = [name for name in subset_names if name not in validation_names_set]

        if len(train_names) == 0:
            raise DatasetManagerError(
                "No data left for training after applying the validation split.",
                f"Dataset size: {dataset_size}",
                f"Validation setting: {self.validation}",
                "Please reduce the validation size, increase the dataset, or disable validation.",
            )

        if self.validation is not None and len(validation_names) == 0:
            raise DatasetManagerError(
                "No data left for validation after applying the validation split.",
                f"Dataset size: {dataset_size}",
                f"Validation setting: {self.validation}",
                "Please increase the validation size, increase the dataset, or disable validation.",
            )

        return train_names, validation_names

    def _prepare_datasets(self) -> None:
        """Resolve dataset files, validate subsets, and precompute train/validation mappings."""
        datasets = self._resolve_dataset_sources()
        dataset_name, subset_names = self._resolve_common_names(datasets)
        subset_names_list = sorted(subset_names)
        if self.subset.shuffle:
            subset_names_list = random.sample(subset_names_list, len(subset_names_list))  # nosec B311
        train_names, validation_names = self._split_train_validation_names(
            subset_names_list,
            dataset_name,
        )
        train_data, train_mapping = self._get_datasets(
            train_names,
            dataset_name,
            self._get_data_augmentations(True),
        )
        validation_data, validate_mapping = self._get_datasets(
            validation_names,
            dataset_name,
            self._get_data_augmentations(self.validation_augmentations),
        )

        self._prepared_data = train_data
        self._prepared_validation_data = validation_data
        self._prepared_mapping = train_mapping
        self._prepared_validation_mapping = validate_mapping
        self._prepared_train_names = train_names
        self._prepared_validation_names = validation_names

    def _get_datasets(
        self,
        names: list[str],
        dataset_name: dict[str, dict[str, list[str]]],
        data_augmentations_list: list[DataAugmentationsList],
    ) -> tuple[dict[str, list[DatasetManager]], list[tuple[int, int, int]]]:
        nb_dataset = len(names)
        nb_patch: list[list[int]]
        data = {}
        mapping = []
        source_filename_by_group = self._get_source_filename_by_group(dataset_name)
        nb_augmentation = self._get_nb_augmentation(data_augmentations_list)

        for group_src in self.groups_src:
            for group_dest in self.groups_src[group_src]:
                data[group_dest] = [
                    DatasetManager(
                        i,
                        group_src,
                        group_dest,
                        name,
                        self.datasets[source_filename_by_group[group_src][name]],
                        patch=self.patch,
                        transforms=self.groups_src[group_src][group_dest].transforms,
                        data_augmentations_list=data_augmentations_list,
                    )
                    for i, name in enumerate(names)
                ]
                nb_patch = [[dataset.get_size(a) for a in range(nb_augmentation)] for dataset in data[group_dest]]

        for x in range(nb_dataset):
            for y in range(nb_augmentation):
                for z in range(nb_patch[x][y]):
                    mapping.append((x, y, z))
        return data, mapping

    def get_groups_dest(self):
        groups_dest = []
        for group_src in self.groups_src:
            for group_dest in self.groups_src[group_src]:
                groups_dest.append(group_dest)
        return groups_dest

    @staticmethod
    def _split(mapping: list[tuple[int, int, int]], world_size: int) -> list[list[tuple[int, int, int]]]:
        if len(mapping) == 0:
            return [[] for _ in range(world_size)]

        mappings: list[list[tuple[int, int, int]]] = []
        if konfai_state() == str(State.PREDICTION) or konfai_state() == str(State.EVALUATION):
            mapping_by_index: dict[int, list[tuple[int, int, int]]] = {}
            for entry in mapping:
                mapping_by_index.setdefault(entry[0], []).append(entry)
            unique_index = np.asarray(sorted(mapping_by_index))
            for shard in np.array_split(unique_index, world_size):
                shard_mapping: list[tuple[int, int, int]] = []
                for dataset_index in shard.tolist():
                    shard_mapping.extend(mapping_by_index[int(dataset_index)])
                mappings.append(shard_mapping)
        else:
            size = len(mapping)
            for rank in range(world_size):
                start = (size * rank) // world_size
                end = (size * (rank + 1)) // world_size
                mappings.append(mapping[start:end])
            # TRAIN/RESUME wraps the model in DDP(static_graph=True): every rank must run the same
            # number of backward all-reduces per epoch. Contiguous shards can differ by one sample,
            # which desynchronises the collective and hangs NCCL, so equalise their length. PAD the
            # shorter shards (wrapping their own head) rather than truncating: truncation permanently
            # drops the tail sample of the longer shards (it is outside every rank's shard, and _split
            # runs once at setup so the sampler's per-epoch shuffle never reaches it), whereas padding
            # keeps every sample training with only a harmless duplicate. world_size == 1 is a no-op.
            # A shard fills itself from its own head, and one that holds nothing has no head to fill
            # from: fewer entries than ranks leaves it empty, and an empty rank runs no backward at
            # all -- the very hang this equalises against. It takes the mapping's head instead.
            max_len = max(len(shard) for shard in mappings)
            mappings = [shard + (shard if shard else mapping)[: max_len - len(shard)] for shard in mappings]
        return mappings

    @staticmethod
    def _remap_dataset_indices(mapping_tmp: list[tuple[int, int, int]]) -> tuple[list[int], list[tuple[int, int, int]]]:
        """Compress sparse dataset indices into local contiguous indices for one loader shard."""
        local_indices: list[int] = []
        index_map: dict[int, int] = {}
        remapped_mapping: list[tuple[int, int, int]] = []
        for dataset_index, augmentation_index, patch_index in mapping_tmp:
            local_index = index_map.get(dataset_index)
            if local_index is None:
                local_index = len(local_indices)
                local_indices.append(dataset_index)
                index_map[dataset_index] = local_index
            remapped_mapping.append((local_index, augmentation_index, patch_index))
        return local_indices, remapped_mapping

    def get_data(self, world_size: int) -> tuple[list[list[DataLoader]], list[str], list[str]]:
        if self._prepared_data is None or self._prepared_validation_data is None:
            raise DatasetManagerError("Dataset configuration was not prepared before runtime data loading.")

        self._resolve_cache_regime(world_size)
        self.data = []
        self.mapping = []
        train_mappings = Data._split(self._prepared_mapping, world_size)
        validate_mappings = Data._split(self._get_validation_mapping(), world_size)
        for i, (train_mapping, validate_mapping) in enumerate(zip(train_mappings, validate_mappings, strict=False)):
            self.data.append([])
            self.mapping.append([])
            train_indices, train_remapped_mapping = self._remap_dataset_indices(train_mapping)
            self.data[i].append({k: [v[it] for it in train_indices] for k, v in self._prepared_data.items()})
            self.mapping[i].append(train_remapped_mapping)
            if len(validate_mapping):
                validation_indices, validation_remapped_mapping = self._remap_dataset_indices(validate_mapping)
                self.data[i].append(
                    {k: [v[it] for it in validation_indices] for k, v in self._prepared_validation_data.items()}
                )
                self.mapping[i].append(validation_remapped_mapping)

        data_loaders: list[list[DataLoader]] = []
        for i, (datas, mappings) in enumerate(zip(self.data, self.mapping, strict=False)):
            data_loaders.append([])
            for loader_index, (dataset_items, mapping) in enumerate(zip(datas, mappings, strict=False)):
                # Windowing is a training-order knob, so it reaches the shuffled training loader only
                # (loader_index == 0). Validation is scored over the whole subset whatever the order,
                # and ``None`` keeps it on the plain global one.
                window = self.subset.shuffle_window if loader_index == 0 else None
                data_loaders[i].append(
                    DataLoader(
                        dataset=self.datasetIter(
                            rank=i,
                            data=dataset_items,
                            mapping=mapping,
                            data_augmentations_list=self._get_data_augmentations(
                                loader_index == 0 or self.validation_augmentations
                            ),
                            apply_augmentations=loader_index == 0 or self.validation_augmentations,
                        ),
                        sampler=WindowedCaseSampler(
                            mapping,
                            self.subset.shuffle,
                            window,
                            self.batch_size,
                            self.resolved_num_workers,
                        ),
                        batch_size=self.batch_size,
                        **self.dataLoader_args,
                    )
                )
        return data_loaders, self._prepared_train_names, self._prepared_validation_names

    def __str__(self) -> str:
        params = {
            "dataset_filenames": self.dataset_filenames,
            "groups_src": self.groups_src,
            "patch": self.patch,
            "use_cache": self.use_cache,
            "memory_budget": self.memory_budget,
            "subset": self.subset,
            "batch_size": self.batch_size,
            "validation": self.validation,
            "validation_augmentations": self.validation_augmentations,
            "inline_augmentations": self.inline_augmentations,
            "data_augmentations_list": self.data_augmentations_list,
        }
        return str(params)

    def __repr__(self) -> str:
        return str(self)


@config("Dataset")
class DataTrain(Data):
    """Dataset configuration used by the training workflow."""

    def __init__(
        self,
        dataset_filenames: list[str] = ["default|./Dataset:mha"],
        groups_src: dict[str, Group] = {"default|Labels": Group()},
        augmentations: dict[str, DataAugmentationsList] | None = {"DataAugmentation_0": DataAugmentationsList()},
        inline_augmentations: bool = False,
        patch: DatasetPatch | None = DatasetPatch(),
        use_cache: bool = True,
        memory_budget: str | float | None = None,
        subset: TrainSubset = TrainSubset(),
        batch_size: int = 1,
        validation: float | str | list[int] | list[str] | None = 0.2,
        validation_augmentations: bool = True,
        num_workers: int | None = None,
        pin_memory: bool = False,
        prefetch_factor: int | None = None,
        persistent_workers: bool | None = None,
    ) -> None:
        super().__init__(
            dataset_filenames,
            groups_src,
            patch,
            use_cache,
            subset,
            batch_size,
            validation,
            validation_augmentations,
            inline_augmentations,
            augmentations if augmentations else {},
            num_workers,
            pin_memory,
            prefetch_factor,
            persistent_workers,
            memory_budget,
        )


@config("Dataset")
class DataPrediction(Data):
    """Dataset configuration used by the prediction workflow."""

    def __init__(
        self,
        dataset_filenames: list[str] = ["default|./Dataset"],
        groups_src: dict[str, Group] = {"default": Group()},
        augmentations: dict[str, DataAugmentationsList] | None = {"DataAugmentation_0": DataAugmentationsList()},
        patch: DatasetPatch | None = DatasetPatch(),
        memory_budget: str | float | None = None,
        subset: PredictionSubset = PredictionSubset(),
        batch_size: int = 1,
        num_workers: int | None = None,
        pin_memory: bool = False,
        prefetch_factor: int | None = None,
        persistent_workers: bool | None = None,
    ) -> None:

        super().__init__(
            dataset_filenames=dataset_filenames,
            groups_src=groups_src,
            patch=patch,
            use_cache=False,
            subset=subset,
            batch_size=batch_size,
            validation=None,
            validation_augmentations=True,
            inline_augmentations=False,
            data_augmentations_list=augmentations if augmentations else {},
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            persistent_workers=False if persistent_workers is None else persistent_workers,
            memory_budget=memory_budget,
        )


@config("Dataset")
class DataMetric(Data):
    """Dataset configuration used by the evaluation workflow."""

    def __init__(
        self,
        dataset_filenames: list[str] = ["default|./Dataset:mha"],
        groups_src: dict[str, GroupMetric] = {"default": GroupMetric()},
        memory_budget: str | float | None = None,
        subset: PredictionSubset = PredictionSubset(),
        validation: str | list[int] | list[str] | None = None,
        num_workers: int | None = None,
        pin_memory: bool = False,
        prefetch_factor: int | None = None,
        persistent_workers: bool | None = None,
    ) -> None:

        super().__init__(
            dataset_filenames=dataset_filenames,
            groups_src=groups_src,
            patch=None,
            use_cache=True,
            subset=subset,
            batch_size=1,
            validation=validation,
            validation_augmentations=True,
            data_augmentations_list={},
            inline_augmentations=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            memory_budget=memory_budget,
        )
