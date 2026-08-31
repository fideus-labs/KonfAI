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


"""The data sources of each workflow: training, prediction, evaluation, dataset preparation."""

import math
import os
import random
import warnings
from abc import ABC, abstractmethod
from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import cast

import numpy as np
from torch.utils.data import DataLoader

from konfai import konfai_state
from konfai.data.augmentation import DataAugmentation, DataAugmentationsList
from konfai.data.data_manager.groups import Group, GroupMetric, GroupOut, _chains
from konfai.data.data_manager.order import WindowedCaseSampler, _interleaved_case_entries
from konfai.data.data_manager.samples import _CACHE_ELEMENT_BYTES, DatasetIter, collate_konfai
from konfai.data.data_manager.subset import PredictionSubset, Subset, TrainSubset
from konfai.data.patching import DatasetManager, DatasetPatch
from konfai.data.transform import (
    Expand,
    Reduce,
    Save,
    Write,
)
from konfai.utils.budget import (
    MemoryBudget,
    format_bytes,
    node_local_ranks,
    resolve_memory_budget,
)
from konfai.utils.clock import startup_clock
from konfai.utils.config import config
from konfai.utils.dataset import Attribute, Dataset
from konfai.utils.errors import DatasetManagerError, TransformerError
from konfai.utils.runtime import State
from konfai.utils.utils import SUPPORTED_FORMATS, resolve_patch, split_path_spec


class DataSources(ABC):
    """The source resolution the four workflow datasets share.

    Which cases, from which file, under which destination group: the ``dataset_filenames`` roots,
    the case names common to every source group and kept by ``subset``, and one
    :class:`~konfai.data.patching.DatasetManager` per (destination group, case), all resolved by
    :meth:`prepare`. :class:`Data` adds the batch-loading mechanics; :class:`DataTransform` builds
    on this alone.
    """

    @abstractmethod
    def __init__(
        self,
        dataset_filenames: list[str],
        groups_src: Mapping[str, Group | GroupMetric | GroupOut],
        subset: Subset,
        memory_budget: str | float | None,
    ) -> None:
        self.dataset_filenames = dataset_filenames
        self.groups_src = groups_src
        self.subset = subset
        self.memory_budget = memory_budget
        self.datasets: dict[str, Dataset] = {}
        #: The selected case names in run order; filled by :meth:`prepare`.
        self.case_names: list[str] = []
        #: Per source group, the cases the walk found: every case the roots hold, or, under a subset
        #: naming its cases, those of them the roots hold. ``None`` before :meth:`prepare`.
        self.cohort_names: dict[str, set[str]] | None = None
        self._managers: dict[str, list[DatasetManager]] | None = None

    @property
    def managers(self) -> dict[str, list[DatasetManager]]:
        """One manager per case of every destination group, in ``case_names`` order, keyed by
        destination group; built by :meth:`prepare`."""
        if self._managers is None:
            raise DatasetManagerError("The dataset was not prepared.", "Call prepare() before reading its managers.")
        return self._managers

    def resolved_budget(self) -> MemoryBudget:
        """The configured memory budget as an object that knows its own scope (``resolve_memory_budget``)."""
        return resolve_memory_budget(self.memory_budget)

    def get_groups_dest(self):
        groups_dest = []
        for group_src in self.groups_src:
            for group_dest in self.groups_src[group_src]:
                groups_dest.append(group_dest)
        return groups_dest

    def _check_destination_groups_are_unique(self) -> None:
        """A destination group names ONE chain, whatever source group it reads.

        It is the key everything downstream indexes by: the prepared managers, the sample handed to
        a model, the plan's lines. Two source groups declaring the same destination name would
        silently keep only the last one: the first chain built, then dropped, with nothing
        anywhere looking wrong. Naming the chain is free: what a chain WRITES is the ``Write``'s own ``group``,
        which is a separate word for a separate thing.
        """
        owner: dict[str, str] = {}
        for group_src, group_dest, _chain in _chains(self.groups_src):
            if group_dest in owner:
                raise DatasetManagerError(
                    f"'groups_src.{owner[group_dest]}' and 'groups_src.{group_src}' both declare"
                    f" a destination group named '{group_dest}'.",
                    "A destination group names one chain. Give them distinct names; to store"
                    " both under the same group name, say so on each Write:"
                    " Write: {dataset: ./Out:omezarr, group: " + group_dest + "}.",
                )
            owner[group_dest] = group_src

    def prepare(self) -> None:
        """Bind the chains, resolve the sources and build the managers. Idempotent."""
        if self._managers is not None:
            return

        self._check_destination_groups_are_unique()
        model_have_input = False
        for group_src, group_dest, chain in _chains(self.groups_src):
            chain.prepare(group_src, group_dest)
            model_have_input |= chain.is_input

        if not model_have_input:
            raise DatasetManagerError(
                "At least one group must be defined with 'is_input: true' to provide input to the network."
            )

        self._prepare_datasets()

    def _prepare_datasets(self) -> None:
        """Resolve the sources, select the cases and build their managers."""
        names, dataset_name = self._select_cases()
        self.case_names = names
        self._managers = self._build_managers(names, dataset_name, patch=None, data_augmentations_list=[])

    def _select_cases(self) -> tuple[list[str], dict[str, dict[str, list[str]]]]:
        """The case names common to every group and kept by ``subset``, in run order (sorted, or
        drawn once when ``subset.shuffle``), with the names each root holds per group."""
        with startup_clock().phase("cases"):
            requested = self.subset.required_names()
            datasets = self._resolve_dataset_sources(requested)
            dataset_name, subset_names = self._resolve_common_names(datasets, requested)
        names = sorted(subset_names)
        if self.subset.shuffle:
            names = random.sample(names, len(names))  # nosec B311
        return names, dataset_name

    def _resolve_dataset_sources(self, requested: set[str] | None = None) -> dict[str, list[tuple[str, bool]]]:
        """The roots holding each source group, as ``(filename, append)`` in declaration order.

        ``requested`` is what the subset can name (:meth:`Subset.required_names`), asked of a root
        in place of its whole listing; ``None`` lists every case.
        """
        datasets: dict[str, list[tuple[str, bool]]] = {}
        if self.dataset_filenames is None or len(self.dataset_filenames) == 0:
            raise DatasetManagerError("No dataset filenames were provided")
        # A resolve after the first (the evaluation's sizing pass, an OOM re-plan) keeps each root's
        # Dataset, and with it the listing it took and the headers it parsed.
        kept = self.datasets
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
                supported_formats=SUPPORTED_FORMATS,
            )
            append = flag != "i"

            if file_format.split("@", 1)[0] not in SUPPORTED_FORMATS:
                raise DatasetManagerError(
                    f"Unsupported file format '{file_format}'.",
                    f"Supported formats are: {', '.join(SUPPORTED_FORMATS)}",
                )

            dataset = kept.get(filename)
            if dataset is None:
                dataset = Dataset(filename, file_format)
            self.datasets[filename] = dataset
            for group in self.groups_src:
                if dataset.is_group_exist(group, requested):
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
        return datasets

    def _resolve_common_names(
        self,
        datasets: dict[str, list[tuple[str, bool]]],
        requested: set[str] | None,
    ) -> tuple[dict[str, dict[str, list[str]]], set[str]]:
        """The names each root holds per group, and the cases every group holds that the subset
        keeps; ``requested`` as in :meth:`_resolve_dataset_sources`."""
        dataset_name: dict[str, dict[str, list[str]]] = {}
        subset_requires_infos = self.subset.requires_infos()
        dataset_info: dict[str, dict[str, dict[str, tuple[list[int], Attribute]]]] | None = (
            {} if subset_requires_infos else None
        )
        empty_infos: dict[str, tuple[list[int], Attribute]] = {}
        if requested is None:
            roots = sorted({filename for entries in datasets.values() for filename, _ in entries})
            print(f"[KonfAI] listing every case of {', '.join(sorted(datasets))} under {', '.join(roots)}")
        cohort: dict[str, set[str]] = {}
        # Seeded from the first group, whatever it holds: an empty first group is a fact of the
        # walk, not a walk that has not started, and it empties the intersection.
        names: set[str] | None = None
        for group in self.groups_src:
            names_by_group = set()
            dataset_name[group] = {}
            if dataset_info is not None:
                dataset_info[group] = {}
            for filename, _ in datasets[group]:
                group_names = self.datasets[filename].select_names(group, requested)
                names_by_group.update(group_names)
                dataset_name[group][filename] = group_names
                if dataset_info is not None:
                    dataset_info[group][filename] = {
                        name: self.datasets[filename].get_infos(group, name) for name in group_names
                    }
            cohort[group] = set(names_by_group)
            names = set(names_by_group) if names is None else names.intersection(names_by_group)
        self.cohort_names = cohort
        if not names:
            if requested is not None:
                # The walk asked the roots for the subset's cases: what is missing is one of them.
                raise self._subset_refusal(
                    f"Subset requested: {', '.join(sorted(requested))}",
                    *(f"Held by '{group}': {', '.join(sorted(found)) or 'none'}" for group, found in cohort.items()),
                )
            raise DatasetManagerError(
                f"No data was found for groups {list(self.groups_src.keys())}: although each group contains data "
                "from a dataset, there are no common dataset names shared across all groups, the intersection is empty."
            )

        subset_names: set[str] | None = None
        for group in dataset_name:
            subset_names_bygroup: set[str] | None = None
            for filename, append in datasets[group]:
                resolved_subset = self.subset(
                    dataset_name[group][filename],
                    dataset_info[group][filename] if dataset_info is not None else empty_infos,
                )
                # Seeded from the first root, not from the first NON-EMPTY one: an empty selection
                # on a root is a member of the intersection, or the roots' order decides the run.
                if subset_names_bygroup is None or append:
                    subset_names_bygroup = (subset_names_bygroup or set()) | set(resolved_subset)
                else:
                    subset_names_bygroup = subset_names_bygroup.intersection(resolved_subset)
            subset_names_bygroup = subset_names_bygroup or set()
            subset_names = (
                set(subset_names_bygroup) if subset_names is None else subset_names.intersection(subset_names_bygroup)
            )

        if not subset_names:
            raise self._subset_refusal(f"Dataset entries found: {', '.join(sorted(names))}")
        return dataset_name, subset_names

    def _subset_refusal(self, *found: str) -> DatasetManagerError:
        """Nothing survives the subset: ``found`` says what the roots hold, then the subset and the
        spellings it takes."""
        return DatasetManagerError(
            "All data entries were excluded by the subset filter.",
            *found,
            f"Subset object applied: {self.subset}",
            "None of the dataset entries matched the given subset.",
            "Please check your 'subset' configuration: it may be too restrictive or incorrectly formatted.",
            "Examples of valid subset formats:",
            "\tsubset: [0, 1]            # explicit indices",
            "\tsubset: [./A.txt, ./B.txt]# union of multiple files",
            "\tsubset: 0:10              # slice notation",
            "\tsubset: ./Validation.txt  # external file",
            "\tsubset: None              # to disable filtering",
        )

    @staticmethod
    def _get_source_filename_by_group(
        dataset_name: dict[str, dict[str, list[str]]],
    ) -> dict[str, dict[str, str]]:
        source_filename_by_group: dict[str, dict[str, str]] = {}
        for group_src, filenames_by_group in dataset_name.items():
            source_filename_by_group[group_src] = {}
            for filename, group_names in filenames_by_group.items():
                for name in group_names:
                    first = source_filename_by_group[group_src].setdefault(name, filename)
                    if first != filename:
                        # Two roots hold the same case of the same group: the first declared is read.
                        # Said out loud, because a stale copy left in one root would be read in silence.
                        warnings.warn(
                            f"Case '{name}' of group '{group_src}' is in '{first}' and in '{filename}':"
                            f" reading '{first}' (dataset_filenames order).",
                            stacklevel=2,
                        )
        return source_filename_by_group

    def _build_managers(
        self,
        names: list[str],
        dataset_name: dict[str, dict[str, list[str]]],
        patch: DatasetPatch | None,
        data_augmentations_list: list[DataAugmentationsList],
        index_offset: int = 0,
    ) -> dict[str, list[DatasetManager]]:
        """One manager per (destination group, case), the case read from the first declared root
        that holds it. ``index_offset`` keeps the manager index unique across the partitions
        :class:`Data` builds: the augmentations they share cache a case's draw by index."""
        source_filename_by_group = self._get_source_filename_by_group(dataset_name)
        with startup_clock().phase("grids"):
            return {
                group_dest: [
                    DatasetManager(
                        index_offset + i,
                        group_src,
                        group_dest,
                        name,
                        self.datasets[source_filename_by_group[group_src][name]],
                        patch=patch,
                        transforms=self.groups_src[group_src][group_dest].transforms,
                        data_augmentations_list=data_augmentations_list,
                    )
                    for i, name in enumerate(names)
                ]
                for group_src in self.groups_src
                for group_dest in self.groups_src[group_src]
            }

    def _params(self) -> dict[str, object]:
        return {
            "dataset_filenames": self.dataset_filenames,
            "groups_src": self.groups_src,
            "memory_budget": self.memory_budget,
            "subset": self.subset,
        }

    def __str__(self) -> str:
        return str(self._params())

    def __repr__(self) -> str:
        return str(self)


class Data(DataSources):
    """The batch-loading layer over :class:`DataSources`, shared by training, prediction and
    evaluation: a patch grid, an optional RAM cache, a train/validation split, augmentation copies,
    and one torch DataLoader per rank and partition (:meth:`get_data`).

    ``case_names``/``managers`` hold the first partition (the training cases); the validation split
    has its own.
    """

    @staticmethod
    def _configured_transform_requires_single_process(classpath: str) -> bool:
        for transform_name in classpath.split("|"):
            candidate = transform_name.split(":")[-1].split(".")[-1].split("/")[0]
            if candidate == "KonfAIInference":
                return True
        return False

    @classmethod
    def _groups_require_single_process_loading(cls, groups_src: Mapping[str, Group | GroupMetric | GroupOut]) -> bool:
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
        groups_src: Mapping[str, Group | GroupMetric | GroupOut],
        subset: Subset,
        memory_budget: str | float | None,
        patch: DatasetPatch | None,
        use_cache: bool,
        batch_size: int,
        validation: float | str | list[int] | list[str] | None,
        num_workers: int | None,
        pin_memory: bool,
        prefetch_factor: int | None,
        persistent_workers: bool | None,
        data_augmentations_list: dict[str, DataAugmentationsList] | None = None,
        inline_augmentations: bool = False,
        validation_augmentations: bool = True,
    ) -> None:
        super().__init__(dataset_filenames, groups_src, subset, memory_budget)
        self.patch = patch
        self.validation = validation
        self.validation_augmentations = validation_augmentations
        self.data_augmentations_list = data_augmentations_list or {}
        self.batch_size = batch_size
        self.inline_augmentations = inline_augmentations
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
        self._validation_managers: dict[str, list[DatasetManager]] = {}
        self._prepared_mapping: list[tuple[int, int, int]] = []
        self._prepared_validation_mapping: list[tuple[int, int, int]] = []
        self._validation_names: list[str] = []

    def _configure_data_loading(self, use_cache: bool) -> None:
        """Build the loader from the cache regime: the DatasetIter factory and the worker settings.

        Called once from ``__init__`` with the declared ``use_cache``, again from ``prepare`` once
        the managers say how each case is read, and, when a ``memory_budget`` overrides it, again
        from ``get_data`` with the derived value.
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
            batch_size=self.batch_size,
        )
        resolved_num_workers = self._num_workers
        if self.requires_single_process_loading:
            resolved_num_workers = 0
        elif resolved_num_workers is None:
            resolved_num_workers = self._default_num_workers(use_cache)
        self.resolved_num_workers: int = resolved_num_workers
        self.dataLoader_args: dict[str, object] = {
            "num_workers": resolved_num_workers,
            "pin_memory": self._pin_memory,
            "collate_fn": collate_konfai,
        }
        if resolved_num_workers > 0:
            self.dataLoader_args["prefetch_factor"] = 2 if self._prefetch_factor is None else self._prefetch_factor
            # Persistent workers keep a fork-time copy of the dataset and never see the main process's
            # per-epoch reset_augmentation redraw, so inline augmentations freeze at their first-epoch draw.
            # An explicit persistent_workers=True cannot override that: correctness wins over the request.
            inline_augmentation_active = self.inline_augmentations and len(self.data_augmentations_list) > 0
            if inline_augmentation_active:
                persistent_workers = False
            elif self._persistent_workers is not None:
                persistent_workers = self._persistent_workers
            else:
                persistent_workers = True
            self.dataLoader_args["persistent_workers"] = persistent_workers

    def _default_num_workers(self, use_cache: bool) -> int:
        """The worker count when the config names none.

        A cache preloads every case up front and leaves the loader nothing to do. A one-pass
        workflow walks each case once in grid order, and shipping a batch through shared memory
        costs more than reading it in place. Workers pay for themselves in one place, where a
        patch read decodes more than the patch it asks for, and the decodes then run in parallel.

        Measured on a quiet 24-core host, three cases, whole PREDICTION runs, best of two
        (``CUDA_VISIBLE_DEVICES=``), zero workers against four:
        patches streaming off an uncompressed MetaImage 0.24 s / 0.34 s (2.5-D, patch
        [1, 192, 192]), 0.14 s / 0.27 s (3-D, [32, 32, 32]), and 4.26 s / 4.89 s on the 2.5-D grid
        with a five-layer convolutional net; the case buffered whole 0.17 s / 0.44 s; and, where
        every patch decodes the volume, 18.1 s / 5.7 s.
        """
        if use_cache:
            return 0
        if self._reads_each_case_once and not self._patch_read_decodes_the_volume():
            return 0
        return max(1, min(os.cpu_count() or 1, 4))

    def _patch_read_decodes_the_volume(self) -> bool:
        """Whether reading one patch costs a whole-volume decode, on any case of any group.

        Only where the patches are read from the store one by one AND the store cannot serve a
        region (a compressed MetaImage, an NRRD, a gzipped NIfTI). A chain that cannot stream is
        not that: it reads the case once into the loader's buffer and cuts its patches from RAM.

        ``False`` before ``prepare``, where no manager can answer yet: only the worker default
        reads this, and only ``get_data`` reads the default.
        """
        if self._managers is None:
            return False
        return any(
            manager.can_stream_patch(0) and not manager.dataset.bounded_region_reads(manager.group_src, manager.name)
            for managers in self._managers.values()
            for manager in managers
        )

    def prepare(self) -> None:
        super().prepare()
        # The managers are what says how each case is read, which is what the worker default turns on.
        self._configure_data_loading(self.use_cache)

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
            (self._managers, Data._get_nb_augmentation(self._get_data_augmentations(True))),
            (
                self._validation_managers,
                Data._get_nb_augmentation(self._get_data_augmentations(self.validation_augmentations)),
            ),
        ):
            for managers in (prepared or {}).values():
                for manager in managers:
                    total += int(np.prod(manager.base_shape, dtype=np.int64)) * _CACHE_ELEMENT_BYTES * copies
        return total

    #: Whether the workflow reads each case exactly once. False for training, whose epochs
    #: re-read every case; True for prediction and evaluation. A one-pass workflow never re-reads
    #: a cache, so a fitting ``memory_budget`` does not choose one, and its loader has one region
    #: read per patch to do, which costs less in place than through a worker.
    _reads_each_case_once = False

    def _resolve_cache_regime(self, world_size: int) -> None:
        """Derive ``use_cache`` from ``memory_budget``. ``None`` means ``"auto"``.

        The cache is chosen iff the per-rank dataset (``dataset / world_size``: ``Data._split``
        shards cases across ranks) fits the per-rank budget: an explicit budget is taken as declared
        per rank; ``"auto"``: also what an absent key means: divides the detected node memory
        (cgroup-capped) by the ranks sharing THAT node, so on a single node the two divisions cancel
        and the test reduces to "does the whole dataset fit the node". The decision is logged once
        here: ``get_data`` runs on the launcher alone, before any worker is spawned.
        """
        if self._reads_each_case_once:
            # One-pass workflows (prediction, evaluation) read each case exactly once: a cache is
            # never re-read, so the regime is always stream/buffer and there is nothing to derive.
            return
        world_size = max(1, world_size)
        n_cases = len(self.case_names) + len(self._validation_names)
        dataset_bytes = self._estimate_cached_bytes()
        per_rank_bytes = dataset_bytes / world_size

        budget = self.resolved_budget()
        per_rank_budget = budget.per_rank_bytes(node_local_ranks(world_size))
        budget_desc = f"{budget.description}, per-rank"

        use_cache = per_rank_bytes <= per_rank_budget
        self._configure_data_loading(use_cache)

        decision = f"CACHE the whole dataset in RAM ({self.resolved_num_workers} loader workers)"
        if not use_cache:
            case_bytes = dataset_bytes / max(1, n_cases)
            decision = (
                f"STREAM/BUFFER, no cache; FIFO working set ~= {self._buffer_size} cases x "
                f"{format_bytes(case_bytes)} = {format_bytes(self._buffer_size * case_bytes)} per worker"
            )
        print(
            f"[KonfAI] memory_budget: dataset ~= {format_bytes(dataset_bytes)} over {n_cases} cases | "
            f"per-rank ~= {format_bytes(per_rank_bytes)} across {world_size} rank(s) | "
            f"budget {format_bytes(per_rank_budget)} ({budget_desc}) -> {decision}"
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

    def _resolve_dataset_sources(self, requested: set[str] | None = None) -> dict[str, list[tuple[str, bool]]]:
        datasets = super()._resolve_dataset_sources(requested)
        # The augmentations get the roots of the first group resolved.
        for entries in datasets.values():
            for data_augmentations in self.data_augmentations_list.values():
                data_augmentations.set_datasets([self.datasets[filename] for filename, _ in entries])
            break
        return datasets

    def _prepare_datasets(self) -> None:
        """Bind the patch and the augmentations, then split the selected cases and build both
        partitions: a manager copies the patch and counts the draws at construction."""
        if self.patch is not None:
            self.patch.init()
        for key, data_augmentations in self.data_augmentations_list.items():
            data_augmentations.prepare(key)
        names, dataset_name = self._select_cases()
        managers = counts = None
        if isinstance(self.validation, float):
            # A share by patch count, which only a built manager knows: every case is built once,
            # in run order with the training draws, and the partitions are cut from that build.
            managers = self._build_managers(names, dataset_name, self.patch, self._get_data_augmentations(True))
            counts = self._case_entry_counts(managers)
        self.case_names, self._validation_names = self._split_train_validation_names(names, counts)
        split = len(self.case_names)
        if managers is not None and (self.case_names, self._validation_names) != (names[:split], names[split:]):
            managers = None  # not a cut of the run order: the built indices would not follow the partitions
        self._build_partitions(dataset_name, managers)

    def _build_partitions(
        self, dataset_name: dict[str, dict[str, list[str]]], managers: dict[str, list[DatasetManager]] | None = None
    ) -> None:
        """(Re)build the managers and patch mappings of both partitions from ``case_names`` and
        ``_validation_names``; the validation indices continue the training ones. Nothing is
        assigned until both are built, so a failure leaves the dataset unprepared.

        ``managers`` are every selected case's, built in run order with the training draws: the
        training partition is their head and, when validation keeps the draws, the validation
        partition their tail, indices included. Unaugmented validation is built without them.
        """
        split = len(self.case_names)
        training = validation = None
        if managers is not None:
            training = {group: cases[:split] for group, cases in managers.items()}
            if self.validation_augmentations:
                validation = {group: cases[split:] for group, cases in managers.items()}
        training, mapping = self._get_datasets(
            self.case_names, dataset_name, self._get_data_augmentations(True), managers=training
        )
        validation, validation_mapping = self._get_datasets(
            self._validation_names,
            dataset_name,
            self._get_data_augmentations(self.validation_augmentations),
            index_offset=split,
            managers=validation,
        )
        self._managers, self._prepared_mapping = training, mapping
        self._validation_managers, self._prepared_validation_mapping = validation, validation_mapping

    def worst_case_shape(self) -> list[int] | None:
        """Per-axis maximum spatial extent over every prepared case and augmentation copy.

        A provisional auto-patch grid starts from this worst case at full extent: one GLOBAL patch
        size, which smaller cases clamp to fewer (or single whole-volume) patches for free.
        """
        shapes = [
            shape
            for prepared in (self._managers, self._validation_managers)
            for managers in (prepared or {}).values()
            for manager in managers
            for shape in manager.shapes
        ]
        if not shapes:
            return None
        return [max(int(shape[axis]) for shape in shapes) for axis in range(len(shapes[0]))]

    def set_free_axis_multiple(self, multiple: list[int] | None) -> None:
        """Record the model's per-axis downsampling factor on the shared patch BEFORE ``prepare()`` cuts
        the grids, so every case's free (``0``) axis rounds up to a valid model input. A no-op without a
        patch (evaluation) or without a free axis; harmless once a re-plan has made the sizes concrete.
        """
        if self.patch is not None:
            self.patch.free_axis_multiple = multiple

    def replan_patch(self, patch_size: list[int]) -> None:
        """Re-cut every prepared grid for a new GLOBAL patch size (the OOM-restart path).

        The managers are rebuilt against the already-resolved sources and the SAME case lists --
        NOT through ``prepare()`` (its idempotence guard would skip the rebuild): so a later
        ``get_data`` shards cases identically across the restart: only the grids and the patch mapping change.
        The new sizes are written into the shared ``patch_size`` list IN PLACE because the loader
        factory holds a reference to it; each rebuilt manager then takes its own copy of them.
        """
        if self.patch is None or self._managers is None:
            raise DatasetManagerError(
                "replan_patch requires a prepared dataset with a patch definition.",
                "Call prepare() first; a dataset without 'patch' has no grid to re-cut.",
            )
        self.patch.patch_size[:] = [int(size) for size in patch_size]
        requested = self.subset.required_names()
        datasets = self._resolve_dataset_sources(requested)
        dataset_name = {
            group: {filename: self.datasets[filename].select_names(group, requested) for filename, _ in entries}
            for group, entries in datasets.items()
        }
        self._build_partitions(dataset_name)

    @staticmethod
    def _patch_counts(managers: dict[str, list[DatasetManager]], nb_augmentation: int) -> list[list[int]]:
        """Per case, per copy, the number of patches, counted on the last destination group."""
        last = next(reversed(managers.values()), [])
        return [[manager.get_size(a) for a in range(nb_augmentation)] for manager in last]

    def _case_entry_counts(self, managers: dict[str, list[DatasetManager]]) -> list[int]:
        """Per case, its ``(copy, patch)`` entries over the training draws: what the float split shares out."""
        nb_augmentation = self._get_nb_augmentation(self._get_data_augmentations(True))
        return [int(sum(counts)) for counts in self._patch_counts(managers, nb_augmentation)]

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
        case_entry_counts: list[int] | None = None,
    ) -> tuple[list[str], list[str]]:
        """The training and validation names, in run order; a float ``validation`` shares out
        ``case_entry_counts`` (one per case, in that order) and takes the tail."""
        dataset_size = len(subset_names) if case_entry_counts is None else int(sum(case_entry_counts))
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

    def _get_datasets(
        self,
        names: list[str],
        dataset_name: dict[str, dict[str, list[str]]],
        data_augmentations_list: list[DataAugmentationsList],
        index_offset: int = 0,
        managers: dict[str, list[DatasetManager]] | None = None,
    ) -> tuple[dict[str, list[DatasetManager]], list[tuple[int, int, int]]]:
        """A partition: its managers (built here unless handed over) and its ``(case, copy, patch)``
        mapping in loader order."""
        if managers is None:
            managers = self._build_managers(names, dataset_name, self.patch, data_augmentations_list, index_offset)
        nb_augmentation = self._get_nb_augmentation(data_augmentations_list)
        mapping: list[tuple[int, int, int]] = []
        # PREDICTION walks the mapping in order, and the copies of a TTA case must advance together
        # along the slab axis for the streamed write to hold a bounded window (see
        # ``_interleaved_case_entries``). TRAIN shuffles the mapping anyway and keeps the plain
        # order, as does a dataset prepared outside any workflow, where no state is set at all.
        interleave = nb_augmentation > 1 and os.environ.get("KONFAI_STATE") == str(State.PREDICTION)
        for x, counts in enumerate(self._patch_counts(managers, nb_augmentation)):
            entries = [(y, z) for y in range(nb_augmentation) for z in range(counts[y])]
            if interleave:
                entries = _interleaved_case_entries([group[x].patch for group in managers.values()], entries)
            mapping.extend((x, y, z) for y, z in entries)
        return managers, mapping

    @staticmethod
    def _split(mapping: list[tuple[int, int, int]], world_size: int) -> list[list[tuple[int, int, int]]]:
        if len(mapping) == 0:
            return [[] for _ in range(world_size)]

        mappings: list[list[tuple[int, int, int]]] = []
        # One-pass workflows shard by CASE; the default branch below is the TRAIN one, whose
        # duplicate-padding (for DDP) would hand the same case to two ranks: two concurrent writers
        # of the same output file for a workflow that writes per case.
        if konfai_state() in (str(State.PREDICTION), str(State.EVALUATION), str(State.TRANSFORM)):
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
            # all: the very hang this equalises against. It takes the mapping's head instead.
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
        if self._managers is None:
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
            self.data[i].append({k: [v[it] for it in train_indices] for k, v in self._managers.items()})
            self.mapping[i].append(train_remapped_mapping)
            if len(validate_mapping):
                validation_indices, validation_remapped_mapping = self._remap_dataset_indices(validate_mapping)
                self.data[i].append(
                    {k: [v[it] for it in validation_indices] for k, v in self._validation_managers.items()}
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
                dataset_iter = self.datasetIter(
                    rank=i,
                    data=dataset_items,
                    mapping=mapping,
                    data_augmentations_list=self._get_data_augmentations(
                        loader_index == 0 or self.validation_augmentations
                    ),
                    apply_augmentations=loader_index == 0 or self.validation_augmentations,
                )
                data_loaders[i].append(
                    DataLoader(
                        dataset=dataset_iter,
                        sampler=WindowedCaseSampler(
                            mapping,
                            self.subset.shuffle,
                            window,
                            self.batch_size,
                            self.resolved_num_workers,
                            dataset_iter.read_order,
                        ),
                        batch_size=self.batch_size,
                        **self.dataLoader_args,
                    )
                )
        return data_loaders, self.case_names, self._validation_names

    def _params(self) -> dict[str, object]:
        return {
            **super()._params(),
            "patch": self.patch,
            "use_cache": self.use_cache,
            "batch_size": self.batch_size,
            "validation": self.validation,
            "validation_augmentations": self.validation_augmentations,
            "inline_augmentations": self.inline_augmentations,
            "data_augmentations_list": self.data_augmentations_list,
        }


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
            subset,
            memory_budget,
            patch=patch,
            # Training re-reads every case each epoch: cache when the dataset fits the
            # 'memory_budget' fit-test, stream when it does not.
            use_cache=True,
            batch_size=batch_size,
            validation=validation,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            data_augmentations_list=augmentations,
            inline_augmentations=inline_augmentations,
            validation_augmentations=validation_augmentations,
        )


@config("Dataset")
class DataPrediction(Data):
    """Dataset configuration used by the prediction workflow."""

    _reads_each_case_once = True

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
            dataset_filenames,
            groups_src,
            subset,
            memory_budget,
            patch=patch,
            use_cache=False,
            batch_size=batch_size,
            validation=None,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            persistent_workers=False if persistent_workers is None else persistent_workers,
            data_augmentations_list=augmentations,
        )


@config("Dataset")
class DataMetric(Data):
    """Dataset configuration used by the evaluation workflow.

    Evaluation never exposes a patch: each run sizes its own from ``memory_budget`` (a missing key
    means ``"auto"``): a case that fits the budget is evaluated whole (exact); one
    that does not is cut into the largest DISJOINT patches that fit (overlap 0, no padding) and the
    reducible metrics combine their running partials into the exact whole-case value. A metric
    scoring through a window declares a halo, and every patch is then read that much wider than its
    slot. The evaluator disables this sizing when any of its metrics is not reducible, so a metric
    that needs the whole volume always gets it.
    """

    _reads_each_case_once = True

    #: Working copies a metric makes of the patch pair (float casts, the difference, a masked select):
    #: measured ~<= 2x the resident tensors; the sizing keeps this conservative and the 0.8 safety
    #: fraction absorbs the rest.
    _METRIC_INTERMEDIATE_FACTOR = 2.0

    # The evaluator clears this when any of its metrics is not reducible: that metric needs whole
    # volumes, so the budget sizing must not cut the case.
    auto_patch_allowed = True
    # The widest halo among the evaluator's metrics: the context every patch is read with past its
    # slot, and what the sizing reserves on each face.
    patch_halo = 0

    def _maybe_auto_patch(self) -> None:
        # An explicit patch or a non-reducible metric vetoes the sizing.
        if self.patch is not None or not self.auto_patch_allowed:
            return
        requested = self.subset.required_names()
        sources = self._resolve_dataset_sources(requested)
        # Header-only scan: for each case, its resident bytes per spatial voxel is the sum of its
        # groups' channels (output + targets + masks all arrive as groups); the WORST case sizes the
        # one patch every case then shares (a smaller case simply yields fewer patches).
        channels_by_name: dict[str, int] = {}
        spatial_by_name: dict[str, list[int]] = {}
        for group, entries in sources.items():
            for filename, _append in entries:
                dataset = self.datasets[filename]
                for name in dataset.select_names(group, requested):
                    shape, _ = dataset.get_infos(group, name)
                    channels_by_name[name] = channels_by_name.get(name, 0) + int(shape[0])
                    spatial = [int(s) for s in shape[1:]]
                    known = spatial_by_name.setdefault(name, spatial)
                    spatial_by_name[name] = [max(a, b) for a, b in zip(known, spatial, strict=False)]
        if not spatial_by_name:
            return
        worst = max(
            spatial_by_name,
            key=lambda name: channels_by_name[name] * int(np.prod(spatial_by_name[name], dtype=np.int64)),
        )
        budget = self.resolved_budget().per_rank_bytes(node_local_ranks())
        extent = spatial_by_name[worst]
        halo = self.patch_halo
        # What the budget bounds is the READ: a slot plus the halo past each face. Sized as one
        # patch; an axis the budget cuts thinner than its two halos is spanned whole instead, since
        # the halo there would cost more than the axis, and the other axes absorb it.
        template = [0] * len(extent)
        while True:
            sized = resolve_patch(
                template,
                extent,
                channels_by_name[worst],
                _CACHE_ELEMENT_BYTES,
                budget,
                resident_images=1,
                intermediate_factor=DataMetric._METRIC_INTERMEDIATE_FACTOR,
            )
            thin = [d for d, size in enumerate(sized) if template[d] == 0 and size < extent[d] and size <= 2 * halo]
            if not thin:
                break
            for d in thin:
                template[d] = extent[d]
        if all(template):
            raise DatasetManagerError(
                f"The memory budget ({format_bytes(budget)}) cannot hold a patch of '{worst}' "
                f"({channels_by_name[worst]}ch x {extent}) with the {halo}-voxel halo its metrics read "
                "past each face.",
                "Raise 'memory_budget'.",
            )
        if sized == extent:
            return  # every case fits whole: the exact whole-volume path
        core = [size - 2 * halo if size < axis else size for size, axis in zip(sized, extent, strict=True)]
        patch = DatasetPatch(patch_size=core, overlap=0)
        patch.pad_to_patch = False  # reduced, not modelled: only in-volume voxels may reach the sums
        patch.halo = halo
        self.patch = patch
        read = f" read with a halo of {halo} ({sized} resident)" if halo else ""
        print(
            f"[KonfAI] memory_budget: worst case '{worst}' "
            f"({channels_by_name[worst]}ch x {extent}) exceeds the budget -> "
            f"evaluating in disjoint patches of {core} (overlap 0){read}, metrics combined exactly."
        )

    def prepare(self) -> None:
        self._maybe_auto_patch()
        super().prepare()

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
            dataset_filenames,
            groups_src,
            subset,
            memory_budget,
            patch=None,
            # Evaluation reads each case exactly once (no augmentations, one pass): a cache is never
            # re-read, it only fronts the whole dataset's RAM. Stream.
            use_cache=False,
            batch_size=1,
            validation=validation,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            # One pass: workers are never reused across epochs, and persistent workers race the
            # process teardown (the terminated worker trips torch's failure handler at exit).
            persistent_workers=False if persistent_workers is None else persistent_workers,
        )


@config("Dataset")
class DataTransform(DataSources):
    """Dataset configuration used by the transform workflow.

    Sources, chains, a budget: the workflow reads :class:`DataSources` alone, its engine being
    :class:`~konfai.data.materialize.CaseMaterializer` over the managers, never a DataLoader. So no
    patch (the planner cuts slabs, never the user), no batch, no validation split, no shuffle, and
    no ``augmentations`` section: a draw is a stage, declared IN the chain, at the place it applies,
    after an :class:`~konfai.data.transform.Expand` marker. Everything decidable from the config
    alone is refused here, before a single byte is read.
    """

    def __init__(
        self,
        dataset_filenames: list[str] = ["default|./Dataset:mha"],
        groups_src: dict[str, GroupOut] = {"default": GroupOut()},
        memory_budget: str | float = "auto",
        subset: PredictionSubset = PredictionSubset(),
    ) -> None:
        super().__init__(dataset_filenames, groups_src, subset, memory_budget)
        #: The run's seed, set by the workflow before :meth:`prepare`. Stamped onto every ``Expand``
        #: below, which is where the only randomness of a transform run lives.
        self.manual_seed = 0

    def prepare(self) -> None:
        # The chains are bound first and the cardinality checked BEFORE any manager exists: a draw
        # declared outside a copy has no shape map, so the manager's own fold would die on an
        # AttributeError naming a method instead of refusing with the place to move the draw to.
        for group_src, group_dest, chain in _chains(self.groups_src):
            chain.prepare(group_src, group_dest)
        self._validate_expansion()
        self._seed_expansions()
        super().prepare()
        self._validate_write_chains()

    def _seed_expansions(self) -> None:
        """Hand the run's seed to every ``Expand`` that did not declare one of its own.

        Done before ``super().prepare()``, which is where the managers are built and the copies
        drawn. Every chain inheriting the same number is what makes an image chain and its mask
        chain agree: they never meet, they derive from one seed they both hold. A chain that
        declares ``seed`` keeps it, which is how two chains are asked for different copies.
        """
        for _group_src, _group_dest, chain in _chains(self.groups_src):
            for transform in chain.transforms:
                if isinstance(transform, Expand) and transform.seed is None:
                    transform.seed = self.manual_seed

    def _validate_write_chains(self) -> None:
        """Refuse, before any byte is read: a chain not ending with a Write, a Save with no dataset,
        a target inside a source root, and two chains writing the same (root, group)."""
        write_targets: dict[tuple[str, str], tuple[str, str]] = {}
        source_roots = {str(Path(filename).resolve()) for filename in self.datasets}
        for group_src, group_dest, group_transform in _chains(self.groups_src):
            chain = f"groups_src.{group_src}.groups_dest.{group_dest}"
            transforms = group_transform.transforms
            if not transforms or not isinstance(transforms[-1], Write):
                after = (
                    f"'{type(transforms[-1]).__name__}' follows the last Write, so its result is written nowhere."
                    if transforms and any(isinstance(t, Write) for t in transforms)
                    else "The chain declares no Write, so the run would read everything and write nothing."
                )
                raise TransformerError(
                    f"'{chain}' does not end with a 'Write'. {after}",
                    "End the chain with Write: {dataset: <path>[:format]}; use 'Save' for intermediate milestones.",
                )
            for transform in transforms:
                if not isinstance(transform, Save) or isinstance(transform, Write):
                    continue
                if not transform.dataset:
                    raise TransformerError(
                        f"'{chain}' has a 'Save' with no dataset: it would write next to the source.",
                        "Give the Save its own destination, e.g. Save: {dataset: ./Work:h5}.",
                    )
            chain_targets: list[tuple[str, str]] = []
            for transform in transforms:
                # The spec, not the Dataset: building one probes the store on disk.
                if not isinstance(transform, Save) or (spec := transform.spec) is None:
                    continue
                root, group = str(Path(spec[0]).resolve()), transform.group or group_dest
                for source_root in source_roots:
                    if root == source_root or Path(root).is_relative_to(source_root):
                        raise TransformerError(
                            f"'{chain}' writes into the source dataset ('{root}').",
                            "Reading is lazy and streaming re-reads the source while writing: an"
                            " in-place transform would read its own half-written output. Write to a"
                            " separate directory.",
                        )
                if (root, group) in write_targets:
                    other = write_targets[(root, group)]
                    raise TransformerError(
                        f"'{chain}' and 'groups_src.{other[0]}.groups_dest.{other[1]}' both write"
                        f" '{group}' under '{root}'.",
                        "A Save is a source boundary keyed by (dataset, group, case) alone, so the"
                        " second chain would take the first one's entry as satisfied and skip its own"
                        " prefix. Give each Save and each Write its own (dataset, group).",
                    )
                chain_targets.append((root, group))
            # Registered once the chain is through, not as each target is seen: one chain may name the
            # same target twice (a Save its terminal Write publishes over); two chains may not.
            for target in chain_targets:
                write_targets[target] = (group_src, group_dest)

    def _validate_expansion(self) -> None:
        """Refuse, from the config alone: more than one Expand or Reduce marker, both in one chain,
        a draw before the Expand (or with no Expand), and an Expand no draw follows."""
        for group_src, group_dest, group_transform in _chains(self.groups_src):
            chain = f"groups_src.{group_src}.groups_dest.{group_dest}"
            transforms = group_transform.transforms
            expands = [t for t in transforms if isinstance(t, Expand)]
            reduces = [t for t in transforms if isinstance(t, Reduce)]
            for kind, markers in (("Expand", expands), ("Reduce", reduces)):
                if len(markers) > 1:
                    raise TransformerError(
                        f"'{chain}' declares {len(markers)} {kind} markers; a chain changes its"
                        " cardinality at most once.",
                        f"Keep one {kind} per chain. Successive ones compose across two"
                        " invocations, the second reading the first one's output back.",
                    )
            if expands and reduces:
                raise TransformerError(
                    f"'{chain}' declares both an Expand and a Reduce.",
                    "One chain changes its cardinality once (1-to-N or N-to-1). Compose the two"
                    " across invocations, the second reading the first one's output back.",
                )
            head = transforms if not expands else transforms[: transforms.index(expands[0])]
            for stage in head:
                if isinstance(stage, DataAugmentation):
                    where = "before the Expand marker" if expands else "but the chain has no Expand marker"
                    raise TransformerError(
                        f"'{chain}' declares the draw '{type(stage).__name__}' {where}.",
                        "A draw makes COPIES, so it belongs after an Expand: transforms: [Clip,"
                        " Expand: {nb: 8}, Rotate, Write]. Applied once per case it would just be a"
                        " random transform, and the run would not be reproducible.",
                    )
            if expands and not any(isinstance(t, DataAugmentation) for t in transforms):
                raise TransformerError(
                    f"'{chain}' declares an Expand but no draw follows it, so every copy would be identical.",
                    "Put the draws after the marker, e.g. Expand: {nb: 8} then Rotate: {a_min: -15, a_max: 15}.",
                )
