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


"""The manager of one case: its chain, its grid, its reads and its writes."""

import contextlib
import copy
import warnings
from collections.abc import Iterable, Iterator, Sequence
from typing import Any, cast

import numpy as np
import torch

from konfai.data.augmentation import DataAugmentationsList
from konfai.data.patching.budget import (
    _PLATEAU_READ_MARGIN,
    _STREAM_STAT_KEYS,
    _STREAM_STATS,
    _UNRESOLVED,
)
from konfai.data.patching.grid import DatasetPatch
from konfai.data.patching.sizer import SegmentSizer
from konfai.data.patching.stage import (
    _MAX_HALO_FRACTION,
    AugmentedStage,
    Stage,
    _drawn_from,
    _halo_radii,
    _HaloPull,
    _is_draw,
    _ReadStagePlan,
    _RemapPull,
    _stage_name,
)
from konfai.data.patching.sweep import (
    SWEEP_CLOCK,
    BlockReads,
    RegionWriter,
    SweepSegment,
    _channel_first_block,
    _HostLanding,
    _open_sweep_stream,
    _PatchStreamSource,
    _PendingSweep,
    _plateau_rows,
    _ReadAhead,
    _shares_h5_file,
    _stage_failure,
    _sweep_header,
    _sweep_targets,
    _SweepMember,
    _WriteBehind,
    save_destination,
)
from konfai.data.transform import (
    Expand,
    LocalityKind,
    PatchLocality,
    RegionContext,
    Save,
    Transform,
    split_expand,
    stat_seed_valid,
)
from konfai.utils.dataset import Attribute, Dataset
from konfai.utils.errors import DatasetManagerError, PatchError
from konfai.utils.utils import env_flag


class DatasetManager:
    """Cache-backed manager for one dataset case and one source/destination group.

    The read side of a case: the chain planned against the stored volume, patches and regions
    replayed through it, and a pending ``Save`` swept when a streamed read first crosses it. The
    write side (materializing the chain's outputs, pricing the routes) is
    :class:`~konfai.data.materialize.CaseMaterializer`, built over this manager.
    """

    def __init__(
        self,
        index: int,
        group_src: str,
        group_dest: str,
        name: str,
        dataset: Dataset,
        patch: DatasetPatch | None,
        transforms: list[Transform],
        data_augmentations_list: list[DataAugmentationsList],
    ) -> None:
        self.group_src = group_src
        self.group_dest = group_dest
        self.name = name
        self.index = index
        self.dataset = dataset
        self.transforms = transforms
        self.loaded = False
        self.augmentationLoaded = False
        self.cache_attributes: list[Attribute] = []
        _shape, cache_attribute = self.dataset.get_infos(self.group_src, name)
        self.base_shape = list(_shape)
        self.cache_attributes.append(cache_attribute)
        _shape = list(_shape[1:])

        self.data: list[torch.Tensor] = []
        self.augmented_data: dict[int, torch.Tensor] = {}
        self.total_augmentations = 0

        # The chain around its Expand: pre runs once per case, post once per copy. Without a marker
        # the split is (everything, None, []) and every fold below reduces to the plain per-case chain.
        self._expand_pre, self._expand, self._expand_post = split_expand(transforms)
        # The landing fold, on a working state: the run re-makes every transition itself, so the
        # case baseline the walks and replays start from must not carry them twice.
        folding = Attribute(cache_attribute)
        for transform_function in self._expand_pre:
            _shape = self._fold_case_state(transform_function, _shape, folding)
        self._adopt_case_facts(folding, cache_attribute)
        # The grid and case state at the Expand point: what the first per-copy stage is handed.
        self._shape_at_expand = list(_shape)
        self._attributes_at_expand = Attribute(cache_attribute)
        # The un-augmented landing of the per-copy tail. A draw is the identity here, because copy 0
        # carries none: the real per-copy grids are folded in reset_augmentation, stage by stage.
        for transform_function in self._expand_post:
            if _is_draw(transform_function):
                continue
            _shape = self._fold_case_state(transform_function, _shape, folding)
        self._adopt_case_facts(folding, cache_attribute)

        self.patch = (
            DatasetPatch(
                # Its own list: the grid is cut lazily (``Patch._grid``), so sharing the source's list
                # would let a later mutation of it move a grid already handed out.
                patch_size=list(patch.patch_size) if patch.patch_size is not None else patch.patch_size,
                overlap=patch.overlap,
                pad_value=patch.pad_value,
                extend_slice=patch.extend_slice,
            )
            if patch
            else DatasetPatch(_shape)
        )
        if patch is not None:
            # The manager works on its own copy (per-case grids); carry the reduction-vs-model contract
            # with it, or a streamed evaluation would silently get padded border patches back.
            self.patch.pad_to_patch = patch.pad_to_patch
            self.patch.halo = patch.halo
            # Carry the model's downsampling multiple too, so each per-case free axis rounds up to a valid
            # input size on this copy's grid, not just on the up-front worst-case sizing.
            self.patch.free_axis_multiple = patch.free_axis_multiple
            # Carry the DECLARED free-axis flag: after an OOM re-plan the source patch_size is already
            # concrete, so this fresh copy could not re-derive it, and the free axis must keep the
            # fraction overlap default, not fall back to the fixed-patch remainder.
            self.patch._declared_free_axis = patch._declared_free_axis
        self.patch.load(_shape, 0)
        # The spatial grid each copy's patches are cut on: the un-augmented copy's is the source shape
        # folded by the transforms, and a copy whose draw changes shape (Permute, Mask) has its own.
        self.shapes: list[list[int]] = [_shape]
        self.data_augmentations_list = data_augmentations_list
        self._patch_stream_sources: dict[tuple[int, bool], _PatchStreamSource | None] = {}
        self._stream_refusals: dict[tuple[int, bool], str] = {}
        self._stream_evolved: dict[tuple[int, bool], Attribute] = {}
        self._stream_attributes_persisted: set[int] = set()
        # Whose chain state the stages' per-case records hold: the stream source last re-folded
        # for a patch replay (_refold_copy_records), or None once any fold or whole-volume call
        # moved them.
        self._records_source: _PatchStreamSource | None = None
        # A GLOBAL_STAT stage's whole-volume statistic is read on the process that runs the chain,
        # at first data access, never by a plan probe: the plan checks the source can provide it
        # (headers only) and the run reads it once. ``_statistics_deferred`` remembers that a plan
        # was resolved without its seed, so the run replans before reading.
        self._statistics_seeded = False
        self._statistics_deferred = False
        # Why a Save sweep gave up for this case, or None. One field rather than a flag beside a
        # warning: the flag is what reroutes the case, the sentence is what a caller with no
        # fallback has to raise with, and they must never disagree.
        self._sweep_failure: str | None = None
        # Rewrite mode: every satisfied-Save probe answers "not written yet", so the case recomputes
        # from the source and each stream's finalize renames over the old entry. Never the default --
        # the boundary IS the per-case resume.
        self._rewrite_saves = False
        # The per-rank budget the sweeps size their slabs against (None = the fixed budget.SWEEP_SLAB_ROWS).
        self._sweep_budget_bytes: float | None = None
        # The store's own read granularity, resolved on first use (None is an answer, not a miss).
        self._read_granularity: object = _UNRESOLVED
        # Per-segment store grains, keyed by (store, group, entry): a segment past a Save boundary
        # reads its own store, never the raw source's (SegmentSizer's whole reason to exist).
        self._granularities: dict[tuple[str, str, str], tuple[int, ...] | None] = {}
        #: One walk of a decomposition per (decomposition, plans), shared by every sizer this
        #: manager builds, keyed with the plans held beside the answer so no identity under the
        #: key can be reused (:meth:`SegmentSizer.block_reads`).
        self._block_reads: dict[tuple, tuple[tuple, BlockReads]] = {}
        self._chain_device: torch.device | None = None
        self._disk_statistics: dict[tuple[Dataset, str, str, tuple[int, ...] | None], dict[str, float]] = {}
        # Save caches already swept by THIS run, keyed by (store, group, entry): under --overwrite the
        # existence probe answers "not written", and without this ledger every copy of an Expand chain
        # would re-sweep the same shared pre-Expand cache once per copy.
        self._swept_entries: set[tuple[str, str, str]] = set()
        # reset_state=False: the first manager built for a case draws (state_init draws a missing
        # index), and every later group's manager reuses that draw: redrawing here would give each
        # group its own geometry and desynchronise the per-copy patch grids across groups.
        self.reset_augmentation(reset_state=False)
        self.cache_attributes_bak = copy.deepcopy(self.cache_attributes)
        # The case as STORED, untouched forever: a boundary-based plan legitimately rewrites the
        # backup with the cache's header, but a rewrite replan must start from the original source.
        self._cache_attributes_pristine = copy.deepcopy(self.cache_attributes_bak)

    def reset_augmentation(self, reset_state: bool = True):
        self.cache_attributes[:] = self.cache_attributes[:1]
        self.shapes[:] = self.shapes[:1]
        self.augmented_data.clear()
        # An augmented copy's stream source is only as good as the draw it was planned for: a halo is
        # the draw's own, and a re-draw is a new one. Drop every plan, so the next request replans
        # against the draw the copies actually carry.
        self._records_source = None
        self._patch_stream_sources.clear()
        self._stream_refusals.clear()
        self._stream_evolved.clear()
        self._stream_attributes_persisted.clear()
        self.total_augmentations = 0
        if self._expand is not None:
            self._draw_expand_copies(reset_state)
        else:
            self._draw_augmentation_lists(reset_state)
        self.augmentationLoaded = self.total_augmentations == 0

    def _draw_expand_copies(self, reset_state: bool) -> None:
        """Draw the :class:`Expand` copies by walking the per-copy tail stage by stage: each draw is
        parameterised on the grid and case state the stages before it leave (``T, draw, T, draw``
        means what it reads like). Each draw is seeded from ``(Expand.seed, case name, draw class,
        rank among its class)``: what two chains of one case agree on, and not the draw's position
        in the tail, so an intensity draw one chain lacks does not shift the geometric ones.
        """
        expand = self._expand
        assert expand is not None  # nosec B101 - the caller checked
        shapes = [list(self._shape_at_expand) for _ in range(expand.nb)]
        attributes = [copy.deepcopy(self._attributes_at_expand) for _ in range(expand.nb)]
        # The copies' walk states, apart from the baselines above: the landing fold evolves the
        # geometry, and a streamed replay must start from the case as stored.
        foldings = [Attribute(attribute) for attribute in attributes]
        drawn: dict[str, int] = {}
        for stage in self._expand_post:
            if _is_draw(stage):
                if reset_state:
                    stage.reset_state(self.index)
                kind = type(stage).__name__
                occurrence = drawn.get(kind, 0)
                drawn[kind] = occurrence + 1
                # One draw, every copy at once: state_init IS the per-copy sampler, and it wants the
                # copies' current grids, which the stages before it just folded.
                # Keyed by the case's NAME, not its index: the index is a position in the run's
                # case list, and a different `subset` (or a second run over image and mask with
                # different subsets) must not hand a case other copies.
                with _drawn_from(expand.draw_seed, self.name, kind, occurrence):
                    shapes = stage.state_init(self.index, shapes, foldings)
                continue
            for index in range(expand.nb):
                shapes[index] = self._fold_case_state(stage, shapes[index], foldings[index])
        for index in range(expand.nb):
            # As at the case-level folds: the box a per-copy Crop computed is a case fact, and
            # losing it re-reads the volume once per later fold of that copy.
            self._adopt_case_facts(foldings[index], attributes[index])
            self.cache_attributes.append(attributes[index])
            self.shapes.append(list(shapes[index]))
            self.patch.load(list(shapes[index]), index + 1)
        self.total_augmentations = expand.nb

    def _draw_augmentation_lists(self, reset_state: bool) -> None:
        """The training form: copies declared as ``Dataset.augmentations`` lists, applied after the
        whole chain."""
        i = 1
        for data_augmentations in self.data_augmentations_list:
            shape = []
            caches_attribute = []
            for _ in range(data_augmentations.nb):
                shape.append(list(self.shapes[0]))
                caches_attribute.append(copy.deepcopy(self.cache_attributes[0]))

            for data_augmentation in data_augmentations.data_augmentations:
                if reset_state:
                    data_augmentation.reset_state(self.index)
                shape = data_augmentation.state_init(self.index, shape, caches_attribute)
            for it, s in enumerate(shape):
                self.cache_attributes.append(caches_attribute[it])
                self.shapes.append(s)
                self.patch.load(s, i)
                i += 1
            self.total_augmentations += data_augmentations.nb

    def load(
        self,
        pre_transform: list[Transform],
        data_augmentations_list: list[DataAugmentationsList],
        load_augmentations: bool = True,
    ) -> None:
        if not self.loaded:
            self._load(pre_transform)
        if load_augmentations and not self.augmentationLoaded:
            self._load_augmentation(data_augmentations_list)

    def _load(self, pre_transform: list[Transform]):
        self.cache_attributes = copy.deepcopy(self.cache_attributes_bak)
        i = len(pre_transform)
        data = None
        for transform_function in reversed(pre_transform):
            if isinstance(transform_function, Save):
                dataset, group_dest = save_destination(transform_function, self.dataset, self.group_dest)
                if not self._rewrite_saves and dataset.is_dataset_exist(group_dest, self.name):
                    data, attrib = dataset.read_data(group_dest, self.name)
                    self.cache_attributes[0].update(attrib)
                    break
            i -= 1

        if i == 0:
            data, _ = self.dataset.read_data(self.group_src, self.name)

        data = torch.from_numpy(data)
        if self._chain_device is not None:
            data = data.to(self._chain_device)

        if len(pre_transform):
            data = self._apply_chain(data, pre_transform[i:], self.cache_attributes[0], self.name)
        self.data.append(data)

        for i in range(len(self.cache_attributes) - 1):
            self.cache_attributes[i + 1].update(self.cache_attributes[0])
        self.loaded = True

    def _apply_chain(
        self, tensor: torch.Tensor, transforms: Sequence[Stage], attribute: Attribute, entry: str
    ) -> torch.Tensor:
        """Apply stages in order on an assembled tensor, writing each Save's cache under ``entry``.

        The one whole-volume applicator: ``_load`` drives it with the case's own name, and the
        expansion fallback with a copy's name: the entry is the only thing that differs between
        assembling a case and assembling one of its copies.
        """
        self._records_source = None  # a stage re-records the case it is called on
        for transform_function in transforms:
            tensor = transform_function(self.name, tensor, attribute)
            if isinstance(transform_function, Save):
                dataset, group_dest = save_destination(transform_function, self.dataset, self.group_dest)
                dataset.write(group_dest, entry, tensor.cpu().numpy(), attribute)
        return tensor

    def _load_augmentation(self, data_augmentations_list: list[DataAugmentationsList]) -> None:
        start_index = 1
        for data_augmentations in data_augmentations_list:
            self._load_augmentation_group(start_index, data_augmentations)
            start_index += data_augmentations.nb
        self.augmentationLoaded = len(self.augmented_data) == self.total_augmentations

    def _load_augmentation_group(self, start_index: int, data_augmentations: DataAugmentationsList) -> None:
        if data_augmentations.nb == 0:
            return

        indices = range(start_index, start_index + data_augmentations.nb)
        if all(index in self.augmented_data for index in indices):
            return

        # The case tensor itself, once per copy. A draw hands back a fresh tensor or a view of what
        # it was given and writes nothing into it (Foreign clones for a class that might), so a
        # copy a draw did not select IS the case, as copy 0 already is. A clone per copy was a
        # memcpy of the case dropped unread: 640 MiB and 0.22 s for 10 copies of a 64 MiB case,
        # per group, per case, per epoch under inline augmentation (measured).
        a_data = [self.data[0] for _ in range(data_augmentations.nb)]
        for data_augmentation in data_augmentations.data_augmentations:
            if data_augmentation.groups is None or self.group_dest in data_augmentation.groups:
                a_data = data_augmentation(self.name, self.index, a_data)

        for index, data in zip(indices, a_data, strict=False):
            self.augmented_data[index] = data
        self.augmentationLoaded = len(self.augmented_data) == self.total_augmentations

    def _augmentation_group(self, a: int) -> tuple[int, DataAugmentationsList]:
        """The augmentation list copy *a* belongs to, and the copy index that list starts at."""
        start_index = 1
        for data_augmentations in self.data_augmentations_list:
            if start_index <= a < start_index + data_augmentations.nb:
                return start_index, data_augmentations
            start_index += data_augmentations.nb
        raise IndexError(f"Augmentation index {a} out of range for dataset '{self.name}'.")

    def _augmentation_stages(self, a: int) -> list[Stage]:
        """The augmentations copy *a* is made of, each bound to it.

        Copy 0 is made of none: it is the tensor the transforms produced, which is why it is the one
        copy that has a counterpart on disk to stream from at all. The rest carry their list's draw,
        minus whatever that draw does not address to this group.
        """
        if a == 0:
            return []
        start_index, data_augmentations = self._augmentation_group(a)
        return [
            AugmentedStage(data_augmentation, self.index, a - start_index)
            for data_augmentation in data_augmentations.data_augmentations
            if data_augmentation.groups is None or self.group_dest in data_augmentation.groups
        ]

    def _expand_tail(self, a: int) -> list[Stage]:
        """The per-copy tail of an :class:`Expand` chain, as copy ``a`` runs it.

        The tail IS the declared order: a transform stays itself, a draw is bound to this copy. The
        two kinds are the same species to everything downstream (the planner reads one contract,
        the replay calls one signature), which is why they can be written in any order.
        """
        if a == 0:
            # Copy 0 is the case itself: it carries no draw, so the tail is its transforms alone --
            # the same landing __init__ folds, and what a probe or a header asks for by default.
            return [stage for stage in self._expand_post if not _is_draw(stage)]
        return [AugmentedStage(stage, self.index, a - 1) if _is_draw(stage) else stage for stage in self._expand_post]

    def _get_tensor(self, a: int) -> torch.Tensor:
        if a == 0:
            return self.data[0]
        if a not in self.augmented_data:
            self._load_augmentation_group(*self._augmentation_group(a))
        return self.augmented_data[a]

    def copy_entry(self, a: int) -> str:
        """The entry name copy ``a`` writes (and resumes) under, behind this chain's ``Expand``.

        Copy 0 is the case itself (the un-augmented tensor has no draw of its own to name), and so
        is every copy of a chain with no ``Expand``, where nothing per-copy is ever written.
        """
        if self._expand is None or a == 0:
            return self.name
        return self._expand.entry(self.name, a)

    def _read_disk_statistics(
        self,
        source_dataset: Dataset,
        source_group: str,
        source_entry: str,
        channels: list[int] | None,
    ) -> dict[str, float]:
        """Read (and memoise) the whole-volume statistics of one on-disk group for this case.

        ``read_data_statistics`` scans the stored volume without materialising it, but it is still a
        full pass: memoise it per (dataset, group, entry, channels) so a per-patch consumer (whose
        ``inverse()`` pops the seeded keys back out of the cache attribute at prediction time) does
        not re-scan the volume once per patch.
        """
        key = (source_dataset, source_group, source_entry, tuple(channels) if channels is not None else None)
        if key not in self._disk_statistics:
            self._disk_statistics[key] = source_dataset.read_data_statistics(source_group, source_entry, channels)
        return self._disk_statistics[key]

    def _require_statistics(self) -> None:
        """Run-time entry: from here on the plan seeds the statistics it deferred, and a plan resolved
        without them is resolved again. Called by every path that reads data through the plan."""
        if self._statistics_seeded:
            return
        self._statistics_seeded = True
        if self._statistics_deferred:
            self._statistics_deferred = False
            self._invalidate_stream_plans()

    def _ensure_stream_stats(
        self,
        source_dataset: Dataset,
        source_group: str,
        source_entry: str,
        cache_attribute: Attribute,
        required_stats: set[str],
        channels: list[int] | None = None,
    ) -> None:
        missing_stats = [key for key in required_stats if key not in cache_attribute]
        if not missing_stats:
            return
        stats = self._read_disk_statistics(source_dataset, source_group, source_entry, channels)
        for key in missing_stats:
            value = stats.get(_STREAM_STATS[key])
            if value is None:
                continue
            if key.endswith("PerChannel"):
                cache_attribute[key] = np.asarray(value, dtype=np.float32)
            elif key in {"Mean", "Std"}:
                cache_attribute[key] = np.asarray([value], dtype=np.float32)
            else:
                cache_attribute[key] = value

    def _affords_halo(self, a: int, halo: tuple[int, ...]) -> bool:
        """Whether a halo of this radius still buys copy *a* anything over loading the volume.

        Every patch pays the halo on every side and the patches tile the volume, so streaming a case
        reads ``prod(1 + 2 * halo_k / patch_k)`` times its bytes: the multiple streaming pays to keep
        one volume off the heap. Half a patch doubles every axis: 8x the reads in 3D. Past that the
        multiple runs away: a halo of one whole patch is 27x, while the saving is still just the
        one volume.
        """
        patch_size = self.patch.patch_size
        extent = (
            self.shapes[a]
            if patch_size is None or all(p == 0 for p in patch_size)
            else [min(p, s) for p, s in zip(patch_size, self.shapes[a], strict=False)]
        )
        return all(
            radius <= _MAX_HALO_FRACTION * size
            for radius, size in zip(_halo_radii(halo, len(extent)), extent, strict=False)
        )

    def _plan_stream_region(
        self,
        a: int,
        stages: list[Stage],
        source_dataset: Dataset,
        source_group: str,
        source_entry: str,
        cache_attribute: Attribute,
        source_spatial_shape: list[int],
        landing_shape: list[int] | None = None,
        seed_statistics: bool = True,
    ) -> tuple[bool, tuple[_ReadStagePlan, ...], Attribute, str | None]:
        """Validate a chain's locality declarations and plan its region stages, which compose.

        Returns ``(streamable, stage_plans, evolved, refusal)``: ``refusal`` names the stage and the
        reason when the chain cannot stream; ``evolved`` is the case state the plan leaves (a
        :class:`Save` sweep writes it as its cache header). The chain streams when every stage is
        pointwise, a region kind (``HALO``/``ORIENTATION``/``CROP``/``REGRID``, each pulling through
        the one before it) or a ``GLOBAL_STAT`` the source can serve; each stage declares against
        the geometry the stages before it left, and a shape fold that does not land on
        ``landing_shape`` refuses. ``seed_statistics=False`` defers a statistic to the sweep of a
        cache not materialized yet. A transform and a draw are planned alike: by declaration.
        """
        evolved = Attribute(cache_attribute)
        shape = [int(extent) for extent in source_spatial_shape]
        localities: list[PatchLocality] = []
        plans: list[_ReadStagePlan] = []

        def refuse(reason: str) -> tuple[bool, tuple[_ReadStagePlan, ...], Attribute, str]:
            """A refusal carries the state folded so far, so the caller can still read the geometry
            the chain reached before the stage that stopped it."""
            return False, (), evolved, reason

        for stage_index, stage in enumerate(stages):
            loc = stage.patch_locality(Attribute(evolved))
            localities.append(loc)
            label = f"stage {stage_index} '{_stage_name(stage)}'"
            if loc.kind in (LocalityKind.WHOLE_VOLUME, LocalityKind.SLAB):
                # SLAB is a write-side contract: its side effect needs the slab's place in the
                # OUTPUT, which a patch read has no notion of. A stage that is whole-volume only
                # because something was left undeclared says so itself (PatchLocality.reason), so
                # the reader is told what to change instead of what happened.
                return refuse(f"{label} declares {loc.kind.name}: {loc.reason or 'it needs the whole volume'}.")
            if loc.kind is LocalityKind.GLOBAL_STAT:
                # The seed is the STORED volume's statistic; otherwise ([Clip(-200, 400), Standardize()])
                # every patch would be standardized by the pre-Clip statistic: fall back to the whole volume.
                if not stat_seed_valid(localities[:-1]):
                    return refuse(
                        f"{label} needs whole-volume statistics, but an earlier stage changes the values"
                        ": the stored volume's statistic is not this stage's input."
                    )
                unknown = sorted(set(loc.stat_keys) - _STREAM_STAT_KEYS)
                if unknown:
                    return refuse(f"{label} needs statistics {unknown} that no source can provide.")
                if seed_statistics and self._statistics_seeded:
                    self._ensure_stream_stats(
                        source_dataset,
                        source_group,
                        source_entry,
                        cache_attribute,
                        set(loc.stat_keys),
                        loc.stat_channels,
                    )
                elif seed_statistics:
                    self._statistics_deferred = True
                # The evolving case state carries the seed too: a Save sweep writes it as the cache
                # header, exactly as the whole-volume pass leaves the statistic in the attribute.
                for stat_key in loc.stat_keys:
                    if stat_key in cache_attribute and stat_key not in evolved:
                        evolved[stat_key] = cache_attribute[stat_key]
            if loc.kind is LocalityKind.HALO and not self._affords_halo(a, loc.halo):
                return refuse(
                    f"{label} declares a halo of {loc.halo} that is too wide for this grid to be worth"
                    " reading (over half the patch extent per axis)."
                )
            plan = self._plan_read_stage(stage, loc, shape, evolved)
            plans.append(plan)
            shape = list(plan.out_shape)
        expected = landing_shape if landing_shape is not None else self.shapes[a]
        if shape != [int(extent) for extent in expected]:
            return refuse(
                f"the chain's shapes fold to {shape} but the target grid is"
                f" {[int(extent) for extent in expected]}: a stage's shape map is missing or wrong."
            )
        return True, tuple(plans), evolved, None

    def _plan_read_stage(
        self, stage: Stage, loc: PatchLocality, shape: list[int], evolved: Attribute
    ) -> "_ReadStagePlan":
        """One stage's slot in the composed plan: its shapes, its pull map, and the case state it
        leaves for the stages after it and for the header the sweep ships
        (``write_stream_cache_attribute``, stated by every stage, region or not: a fold of the
        channel axis consumes a key that describes an input the output no longer has)."""
        if not loc.kind.is_region:
            plan = _ReadStagePlan(loc.kind, tuple(shape), tuple(shape), None)
        elif loc.kind is LocalityKind.HALO:
            plan = _ReadStagePlan(
                loc.kind, tuple(shape), tuple(shape), _HaloPull(_halo_radii(loc.halo, len(shape)), list(shape))
            )
        else:
            # ORIENTATION / CROP / REGRID: the stage's own remap, on the state the stages before it left.
            pull = _RemapPull(stage.stream_region_source, list(shape), Attribute(evolved), self.name)
            measured = getattr(stage, "measured_region_source", None)
            run_pull = (
                _RemapPull(measured, list(shape), Attribute(evolved), self.name)
                if measured is not None and getattr(stage, "measures_at_run", False)
                else None
            )
            out = self._stage_out_shape(stage, shape, Attribute(evolved))
            plan = _ReadStagePlan(loc.kind, tuple(shape), tuple(out), pull, run_pull)
        stage.write_stream_cache_attribute(evolved, list(shape), self.name)
        return plan

    def _stage_out_shape(self, stage: Stage, shape: list[int], attribute: Attribute) -> list[int]:
        """The spatial shape one stage folds ``shape`` to: a transform's map or a draw's own.

        The one dispatch between the two Stage species' shape vocabularies: a ``Transform`` restates
        its fold as ``transform_shape``, an :class:`AugmentedStage` as its draw's ``stream_shape``.
        Shape only: the geometry transition is :meth:`_fold_case_state`'s half.
        """
        self._records_source = None  # transform_shape records the case state it is handed
        if isinstance(stage, Transform):
            return [int(e) for e in stage.transform_shape(self.group_src, self.name, list(shape), attribute)]
        return [int(e) for e in cast(AugmentedStage, stage).stream_shape(list(shape))]

    def _fold_case_state(self, stage: Stage, shape: list[int], attribute: Attribute) -> list[int]:
        """Fold one stage over the evolving case state: the shape through its map, the geometry
        through its stated transition: the idiom :meth:`_plan_read_stage` runs per region stage.

        Every landing fold goes through here, so a stage is judged on the state the stages before it
        left rather than on the stored header: a ``Resample`` behind a ``Canonical`` records the
        reoriented grid, and a second ``Resample`` sees the first one's spacing.
        """
        out = self._stage_out_shape(stage, shape, attribute)
        if isinstance(stage, Transform):
            stage.write_stream_cache_attribute(attribute, list(shape), self.name)
        return out

    @staticmethod
    def _adopt_case_facts(folding: Attribute, case: Attribute) -> None:
        """Keep what a landing fold computed about the CASE (Crop's content-derived box) off its
        walk state. The geometry the fold evolved is the walk's own (the run re-makes those
        transitions); the box is expensive, immutable per case, and read by every later fold, the
        streamed replays and the run itself."""
        if "box" in folding and "box" not in case:
            case["box"] = folding["box"]

    def chain_stages(self, a: int = 0) -> list[Stage]:
        """The ordered stages copy ``a`` is made of: the one definition of what a copy IS.

        Behind an :class:`Expand`, the shared prefix, then the copy's own draw at the marker's
        position, then the per-copy tail. Without one, the chain itself, with any draw appended
        last: the training order, where an augmentation is a copy of the chain's whole result.
        """
        if self._expand is None:
            return [*self.transforms, *self._augmentation_stages(a)]
        return [*self._expand_pre, *self._expand_tail(a)]

    def _resolve_patch_stream_source(self, a: int, apply_augmentations: bool = True) -> _PatchStreamSource | None:
        key = (a, apply_augmentations)
        if key in self._patch_stream_sources:
            return self._patch_stream_sources[key]

        source_dataset = self.dataset
        source_group = self.group_src
        source_entry = self.name
        source_shape = list(self.base_shape)
        # Plan from the case as STORED (the pristine backup), never from the live attribute: the live
        # one carries what earlier patches or epochs wrote (a Resample's target Spacing, a Canonical's
        # canonical Direction), and planning from it would hand a stage its own output as the
        # description of its input on every epoch after the first.
        stream_cache_attribute = Attribute(self.cache_attributes_bak[0])
        pending: list[_PendingSweep] = []
        trailing_transforms: list[Stage] = []
        sweep_refusal: str | None = None
        # The entry name Saves write under: the case's own before the Expand marker, the copy's own
        # after it. `splice_at` is where the per-copy stages begin WITHIN the current segment (None =
        # the segment is entirely shared), which the expansion engine reads to share one read pass.
        entry = self.name
        splice_at: int | None = None
        past_expand = False

        walked: list[Stage] = (
            list(self.transforms) if self._expand is None else [*self._expand_pre, self._expand, *self._expand_tail(a)]
        )
        for transform in walked:
            if isinstance(transform, Expand):
                # The marker is replaced by the copy's own draw: everything before it is the case's,
                # everything after it (including every Save destination) is the copy's.
                splice_at = len(trailing_transforms)
                past_expand = True
                entry = self.copy_entry(a)
                continue
            if isinstance(transform, Save):
                dataset, group = save_destination(transform, self.dataset, self.group_dest)
                if not self._rewrite_saves and dataset.is_dataset_exist(group, entry):
                    source_dataset, source_group, source_entry = dataset, group, entry
                    source_shape, boundary_attributes = dataset.get_infos(group, entry)
                    source_shape = list(source_shape)
                    # Streaming from a Save cache: the stored volume is the cache, so the stages after
                    # the boundary read its geometry: stacked over the source keys exactly as the
                    # whole-volume cache-hit merges the cached header.
                    stream_cache_attribute = Attribute(self.cache_attributes_bak[0])
                    for attribute_key, attribute_value in boundary_attributes.items():
                        stream_cache_attribute[attribute_key] = attribute_value
                    pending.clear()
                    trailing_transforms = []
                    # A satisfied per-copy cache already holds the draw: the new segment is entirely
                    # per-copy, which `splice_at = 0` says.
                    splice_at = 0 if past_expand else None
                    sweep_refusal = None
                    continue
                planned, planned_attribute, planned_refusal = self._plan_save_sweep(
                    dataset,
                    group,
                    entry,
                    trailing_transforms,
                    splice_at if splice_at is not None else len(trailing_transforms),
                    source_dataset,
                    source_group,
                    source_entry,
                    source_shape,
                    stream_cache_attribute,
                    seed_statistics=not pending,
                )
                if planned is not None and planned_attribute is not None:
                    stream_cache_attribute = planned_attribute
                    source_dataset, source_group, source_entry = dataset, group, entry
                    source_shape = [source_shape[0], *planned.out_spatial]
                    pending.append(planned)
                    trailing_transforms = []
                    splice_at = 0 if past_expand else None
                    continue
                # An unplannable Save stays in the chain, where its WHOLE_VOLUME declaration refuses
                # the whole plan: keep the sweep's own reason, or the chain-level one would only
                # ever say "Save needs the whole volume" and mask the actual cause.
                if sweep_refusal is None:
                    sweep_refusal = planned_refusal
            trailing_transforms.append(transform)

        # What copy `a` is. Without an Expand, the training order: the trailing transforms, then
        # its own draw appended last. With one, the draw was spliced at the marker above, and the
        # whole thing is planned as one chain either way: a region transform and a region
        # augmentation are then two regions, which is exactly what they are.
        if self._expand is None:
            stages = trailing_transforms + (self._augmentation_stages(a) if apply_augmentations else [])
        else:
            stages = trailing_transforms

        streamable, stage_plans, evolved, chain_refusal = self._plan_stream_region(
            a,
            stages,
            source_dataset,
            source_group,
            source_entry,
            stream_cache_attribute,
            list(source_shape[1:]),
            seed_statistics=not pending,
        )
        if not streamable:
            self._stream_refusals[key] = sweep_refusal or chain_refusal or "the chain cannot stream."
            self._patch_stream_sources[key] = None
        elif pending:
            # The pending source only answers the regime probes: no attribute is persisted and no
            # patch flows from it: the sweeps run at first data access, and the source is then
            # re-resolved from the materialized caches (the satisfied-Save path above).
            self._patch_stream_sources[key] = _PatchStreamSource(
                source_dataset, source_group, source_entry, source_shape, stages, stage_plans, tuple(pending)
            )
        else:
            self.cache_attributes[a] = Attribute(stream_cache_attribute)
            self.cache_attributes_bak[a] = Attribute(stream_cache_attribute)
            self._patch_stream_sources[key] = _PatchStreamSource(
                source_dataset, source_group, source_entry, source_shape, stages, stage_plans
            )
        # The state the whole plan lands on, kept for consumers that need the LANDED geometry (a
        # reduction seeding its output header) without re-walking the chain. Recorded only when the
        # plan HOLDS: a refused plan folded as far as the stage that refused and no further, and half
        # a fold is not a geometry: it is a Spacing from before the resample meant to change it.
        # An unset key is what lets ``landed_attributes`` answer with the stored state instead.
        if streamable:
            self._stream_evolved[key] = Attribute(evolved)
        return self._patch_stream_sources[key]

    def _plan_save_sweep(
        self,
        destination: Dataset,
        group: str,
        entry: str,
        segment: list[Stage],
        copy_stage_start: int,
        source_dataset: Dataset,
        source_group: str,
        source_entry: str,
        source_shape: list[int],
        base_attributes: Attribute,
        seed_statistics: bool,
    ) -> tuple[_PendingSweep | None, Attribute | None, str | None]:
        """Plan the materialization of one unsatisfied :class:`Save`, or refuse with the reason and
        leave it on the whole-volume path: the segment feeding it must itself stream, and the
        destination must serve region writes (probed by capability, so a refusal costs nothing).
        Returns ``(sweep, evolved, None)`` on success (the pending sweep and the case state its
        cache will carry, which the stages after the Save plan against), and ``(None, None,
        reason)`` on refusal."""
        if self._sweep_failure is not None:
            return (
                None,
                None,
                f"an earlier sweep failed for this case, so every Save takes the whole-volume"
                f" path. {self._sweep_failure}",
            )
        if not env_flag("KONFAI_STREAMED_WRITES", True):
            return None, None, "KONFAI_STREAMED_WRITES=0 disables streamed writes."
        landing = [int(extent) for extent in source_shape[1:]]
        probe = Attribute(base_attributes)
        for stage in segment:
            landing = self._fold_case_state(stage, landing, probe)
        planning = Attribute(base_attributes)
        streamable, stage_plans, evolved, refusal = self._plan_stream_region(
            0,
            segment,
            source_dataset,
            source_group,
            source_entry,
            planning,
            [int(extent) for extent in source_shape[1:]],
            landing_shape=landing,
            seed_statistics=seed_statistics,
        )
        if not streamable:
            return None, None, refusal
        if not destination.can_stream_data(evolved):
            return (
                None,
                None,
                f"destination '{destination.filename}' cannot serve region writes for this entry"
                " (h5 and omezarr always can; mha only with image geometry).",
            )
        sweep = _PendingSweep(
            destination,
            group,
            entry,
            list(segment),
            source_dataset,
            source_group,
            source_entry,
            list(source_shape),
            tuple(landing),
            planning,
            min(copy_stage_start, len(segment)),
            stage_plans,
        )
        return sweep, evolved, None

    def sweep_segments(self, a: int = 0, apply_augmentations: bool = False) -> list[SweepSegment] | None:
        """Every segment the streamed route sweeps: one per unsatisfied ``Save`` (each sweeps ITS
        source) plus the head past the last boundary, which is read only if stages follow it.
        ``None`` when the chain cannot stream at all.

        Public because three consumers ask the same question of one plan: what the run will sweep,
        what the plan prices it at, and whether the budget holds a region of it.
        """
        source = self._resolve_patch_stream_source(a, apply_augmentations)
        if source is None:
            return None
        segments = [
            SweepSegment(
                sweep.source_dataset,
                sweep.source_group,
                sweep.source_entry,
                [int(extent) for extent in sweep.source_shape],
                list(sweep.out_spatial),
                sweep.stage_plans,
                tuple(sweep.stages),
            )
            for sweep in source.pending_sweeps
        ]
        if source.stage_plans:
            segments.append(
                SweepSegment(
                    source.dataset,
                    source.group,
                    source.entry,
                    list(source.shape),
                    list(source.stage_plans[-1].out_shape),
                    source.stage_plans,
                    tuple(source.stages),
                )
            )
        return segments

    def can_stream_patch(self, a: int, apply_augmentations: bool = True) -> bool:
        return self.stream_refusal(a, apply_augmentations) is None

    def stream_refusal(self, a: int = 0, apply_augmentations: bool = True) -> str | None:
        """Why this copy cannot stream (the reified refusal) or ``None`` when it can.

        Resolves the plan (a probe, never a write) and hands back the first stage-level reason the
        planner met, or, past that, the budget one: a chain whose smallest region does not fit is
        not a chain that streams, and routing it away here is what makes the whole-volume pass
        price and refuse it before a byte is written, instead of the sweep meeting it at its first
        slab. The whole-volume fallback stays available, but neither refusal is silent."""
        segments = self.sweep_segments(a, apply_augmentations)
        if segments is None:
            return self._stream_refusals.get((a, apply_augmentations), "the chain cannot stream.")
        try:
            for segment in segments:
                self.sizer_for(segment).sweep_tile()
        except DatasetManagerError as refusal:
            return str(refusal.args[0])
        return None

    @property
    def spatial_shape(self) -> list[int]:
        """The spatial extent this case's chain lands on: the source folded by every stage."""
        return list(self.shapes[0])

    @property
    def stored_attributes(self) -> Attribute:
        """The case as STORED: the geometry of the entry on disk, before any stage ran.

        A copy, and pristine on purpose: the live attribute carries what earlier regions or epochs
        wrote into it, so anything planning against it would be handed a stage's own output as the
        description of its input.
        """
        return Attribute(self.cache_attributes_bak[0])

    def landed_attributes(self, a: int = 0) -> Attribute:
        """The case state the chain LANDS on: stored geometry folded by every stage's rewrite.

        This is what an output built FROM the chain's result must carry as its header (a Resample's
        target ``Spacing``, a Canonical's direction), where :attr:`stored_attributes` is the source's
        own. Resolved from the plan (a probe, never a write); a chain that cannot stream answers with
        the stored state, the only honest one available without assembling the volume.
        """
        self._resolve_patch_stream_source(a, apply_augmentations=False)
        evolved = self._stream_evolved.get((a, False))
        return Attribute(evolved) if evolved is not None else self.stored_attributes

    def read_region(self, target: tuple[slice, ...], a: int = 0, apply_augmentations: bool = False) -> torch.Tensor:
        """Run this case's chain over one region of its output, reading only what that region pulls
        (``target`` indexes the spatial axes of :attr:`spatial_shape`; every channel is read). Raises
        rather than falling back (:meth:`stream_refusal` says why beforehand); an unwritten ``Save``
        upstream is swept first (:meth:`_stream_ready`), as on the DataLoader path.
        """
        if not self._stream_ready(a, apply_augmentations):
            raise PatchError(
                f"Case '{self.name}' cannot stream its chain, so it cannot serve a region.",
                self.stream_refusal(a, apply_augmentations)
                or self._sweep_failure
                or "See stream_refusal() for the refusing stage.",
            )
        source = self._resolve_patch_stream_source(a, apply_augmentations)
        if source is None:  # pragma: no cover - _stream_ready just resolved it
            raise PatchError(
                f"Case '{self.name}' cannot stream its chain, so it cannot serve a region.",
                self.stream_refusal(a, apply_augmentations) or "See stream_refusal() for the refusing stage.",
            )
        tensor, _attribute, _keys = self._replay_streamed_region(source, target, self.stored_attributes, None)
        return tensor

    @contextlib.contextmanager
    def _chain_device_scope(self, device: "torch.device | None") -> Iterator[None]:
        """Route the chain onto ``device`` for one materialization, restored on every exit path:
        a device that outlived the call would move a later ``get_data()`` onto CUDA inside a
        DataLoader worker."""
        previous = self._chain_device
        self._chain_device = device if device is not None and device.type != "cpu" else None
        try:
            yield
        finally:
            self._chain_device = previous

    def set_memory_budget(self, budget_bytes: float | None) -> None:
        """The per-rank budget this case's streamed sweeps size their slabs against.

        Public because the budget reaches a manager from more than one door:
        :meth:`~konfai.data.materialize.CaseMaterializer.materialize` takes it as an argument, while
        a reduction only ever calls :meth:`read_region`: which sweeps pending Saves as a side effect
        and must sweep them under the same bound.
        """
        self._sweep_budget_bytes = budget_bytes

    def set_chain_device(self, device: torch.device | None) -> None:
        """The device this case's chain replays on. Public for the same reason as the budget: a
        reduction never calls :meth:`materialize`, only :meth:`read_region`, and its members must
        replay where the fold will run. CPU collapses to None: opt-in only, because this same
        machinery loads training cases inside DataLoader workers, where a CUDA default is wrong."""
        self._chain_device = device if device is not None and device.type != "cpu" else None

    def _set_rewrite(self, rewrite: bool) -> None:
        """The rewrite knob the materialization engine sets for one call: under it every
        satisfied-Save probe answers "not written yet" (the read path consults it too, which is
        why the flag and the ledger it resets live here and not on the engine)."""
        if rewrite == self._rewrite_saves:
            return
        # The memoized plans were probed under the other boundary answer: replan. And replan from
        # the case as STORED: an earlier boundary-based plan wrote the OUTPUT's header into the
        # backup (its Spacing, its Size), and a rewrite planned from that geometry re-writes
        # untransformed data over the deliverable without an error.
        self._rewrite_saves = rewrite
        self._patch_stream_sources.clear()
        self._stream_refusals.clear()
        self._stream_evolved.clear()
        self._swept_entries.clear()
        self._sweep_failure = None
        self.cache_attributes_bak = copy.deepcopy(self._cache_attributes_pristine)
        self.cache_attributes = copy.deepcopy(self._cache_attributes_pristine)

    def _stream_ready(self, a: int, apply_augmentations: bool = True) -> bool:
        """Whether this copy can stream its patches, materializing what that requires.

        Resolves the source and, the first time a case whose chain reads through unmaterialized Save
        caches is actually asked for data, sweeps them and re-resolves from disk: all data then
        flows through the satisfied-boundary path, exactly as if the caches had always existed. The
        regime probes (``can_stream_patch``) answer from the plan alone and never write."""
        self._require_statistics()
        source = self._resolve_patch_stream_source(a, apply_augmentations)
        if source is None:
            return False
        if not source.pending_sweeps:
            return True
        self._sweep_pending(source.pending_sweeps)
        return not self._sweep_failed and self._resolve_patch_stream_source(a, apply_augmentations) is not None

    def _sweep_pending(self, sweeps: Iterable[_PendingSweep]) -> None:
        """Materialize the pending Save caches in order, stopping at the first failure: they are
        chained (each one's source is the previous one's destination), so past a failure the next
        would read a cache nobody wrote and record its own symptom over the cause. Every plan is
        then dropped: they pointed at caches that did not exist yet, or after a failure never will."""
        for sweep in sweeps:
            if not self._materialize_save(sweep):
                break
        self._invalidate_stream_plans()

    @property
    def _sweep_failed(self) -> bool:
        return self._sweep_failure is not None

    def _invalidate_stream_plans(self) -> None:
        self._patch_stream_sources.clear()
        self._stream_refusals.clear()
        self._stream_evolved.clear()

    def _materialize_save(self, sweep: _PendingSweep) -> bool:
        """Write one Save cache slab by slab through its segment, re-planned against its source as
        it is on disk now; the cache appears only when complete. On failure the partial entry is
        removed, ``_sweep_failure`` keeps the reason and ``False`` is returned: the case falls back
        to the whole-volume path, or a caller without one raises with the reason."""
        ledger_key = (str(sweep.destination.filename), sweep.group, sweep.entry)
        if ledger_key in self._swept_entries:
            # Already swept by THIS run: under --overwrite the existence probe answers "not written",
            # and re-sweeping a cache the copies share would redo the same work once per copy.
            return True
        if not self._rewrite_saves and sweep.destination.is_dataset_exist(sweep.group, sweep.entry):
            return True
        source, evolved, refusal = self._replan_sweep(sweep)
        if source is None:
            # The plan probe said yes and the re-plan against the materialized source says no: that
            # is new information, and it is the whole reason this case is about to cost a volume.
            return self._sweep_failed_because(
                sweep, refusal or "the segment feeding it no longer plans against its materialized source."
            )
        _written, failure = self._sweep(source, sweep, evolved, [_SweepMember(None, sweep, evolved)])
        if failure is not None:
            return self._sweep_failed_because(sweep, failure)
        return True

    def _replan_sweep(
        self, sweep: _PendingSweep, stages: list[Stage] | None = None
    ) -> tuple[_PatchStreamSource | None, Attribute, str | None]:
        """Re-plan STAGES (the sweep's own by default) against the sweep's source as it is on disk
        now: the stream source to replay from and the case state the segment lands with, or the
        refusal (``source`` is then ``None``)."""
        stages = sweep.stages if stages is None else stages
        streamable, plans, evolved, refusal = self._plan_stream_region(
            0,
            stages,
            sweep.source_dataset,
            sweep.source_group,
            sweep.source_entry,
            Attribute(sweep.base_attributes),
            [int(extent) for extent in sweep.source_shape[1:]],
            landing_shape=list(sweep.out_spatial),
        )
        if not streamable:
            return None, evolved, refusal
        source = _PatchStreamSource(
            sweep.source_dataset,
            sweep.source_group,
            sweep.source_entry,
            list(sweep.source_shape),
            stages,
            plans,
        )
        return source, evolved, None

    def _sweep(
        self,
        source: _PatchStreamSource,
        reference: _PendingSweep,
        evolved: Attribute,
        members: list[_SweepMember],
    ) -> tuple[set[Any], str | None]:
        """The block loop every sweep runs: each block of REFERENCE's landing is read once through
        SOURCE (its first block against EVOLVED, so a region stage recording geometry nowhere the
        case can read refuses here, as the patch path does), then every :class:`_SweepMember`
        applies its tail to the block and region-writes it into its own stream, opened on the first
        block with the header the whole-volume pass would leave. Returns the keys written and, when
        the pass failed, why: every stream is then aborted; an interrupt is re-raised."""
        spatial = list(reference.out_spatial)
        channels = int(reference.source_shape[0])
        # Keyed to the segment being swept: ITS stages (the re-planned source's) and ITS store.
        sizer = self._sizer(
            spatial,
            channels,
            source.stage_plans,
            tuple(source.stages),
            self._entry_granularity(source.dataset, source.group, source.entry),
        )
        tile = sizer.sweep_tile()
        targets = list(_sweep_targets(spatial, tile))
        depth = sizer.sweep_depth(tile)
        if any(_shares_h5_file(source.dataset, member.sweep.destination) for member in members):
            # The h5 backend holds a per-file lock for a stream's whole life, on the thread that
            # opened it: a read of that file from any other thread waits for the close that the
            # read itself stands in the way of. One thread, where the lock re-enters.
            depth = 0
        # Reading ahead means the reading thread must touch no stage of the chain, so the pull maps
        # are folded here, before it starts. A stage that sizes its window from the data it reads
        # (a displacement field: the sizing read IS the sampling read) cannot be folded ahead, and
        # that chain reads where it samples.
        pulls = (
            []
            if any(plan.run_pull is not None for plan in source.stage_plans)
            else [self._region_spans(source, target) for target in targets]
        )
        ahead = depth if pulls else 0
        if pulls:
            self._declare_region_reads([(source, spans) for spans in pulls])
        # A member's tail reads on the landing, whose regions are the targets: known whether or not
        # the prefix folds its own pulls ahead.
        for member in members:
            for stage, plan in zip(member.stages, member.stage_plans, strict=True):
                stage.plan_region_reads(self.name, [plan.region_context(target, target) for target in targets])
        sweeps = {member.key: member.sweep for member in members}
        headers: dict[Any, Attribute] = {}
        writer = RegionWriter(lambda key, block, header: _open_sweep_stream(sweeps[key], block, spatial, tile, header))

        def regions() -> Iterator[tuple[int, list[list[slice]], torch.Tensor, Attribute]]:
            for index, target in enumerate(targets):
                spans = pulls[index] if pulls else self._region_spans(source, target)
                with SWEEP_CLOCK.phase("read"):
                    tensor, attributes = self._read_streamed_region(source, spans)
                yield index, spans, tensor, attributes

        write, landing = _WriteBehind(writer, depth), _HostLanding()
        try:
            with SWEEP_CLOCK.phase("sweep"), _ReadAhead(regions(), ahead) as blocks:
                for index, spans, tensor, attributes in SWEEP_CLOCK.waiting("wait(read)", blocks):
                    target = targets[index]
                    with SWEEP_CLOCK.phase("chain"):
                        tensor, region_attribute, keys_before = self._apply_streamed_region(
                            source,
                            spans,
                            tensor,
                            attributes,
                            Attribute(reference.base_attributes),
                            Attribute(evolved) if index == 0 else None,
                        )
                    for member in members:
                        with SWEEP_CLOCK.phase("chain"):
                            member_tensor = tensor.clone() if len(members) > 1 else tensor
                            scope = Attribute(region_attribute)
                            # Dispatched exactly as the stages before the marker are, so a tail stage
                            # reading a companion volume (Mask) or drawing from the voxel's place
                            # (Noise, CutOUT) is told where its block sits instead of taking it for
                            # the whole volume.
                            member_tensor = self._run_streamed_stages(
                                member.stages,
                                member.stage_plans,
                                member.region_spans(target),
                                member_tensor,
                                scope,
                                None,
                            )
                        # Its own phase, not the chain's: on a device the chain only ENQUEUES,
                        # and this is where the run waits for it as well as for the copy home.
                        with SWEEP_CLOCK.phase("fetch"):
                            block = landing.take(member_tensor)
                        if member.key not in headers:
                            headers[member.key] = _sweep_header(member.evolved, scope, keys_before)
                        block = _channel_first_block(
                            block,
                            spatial,
                            headers[member.key],
                            f"A stage of the chain writing '{member.sweep.group}/{member.sweep.entry}'",
                        )
                        with SWEEP_CLOCK.phase("wait(write)"):
                            write.write(
                                member.key, (slice(0, int(block.shape[0])), *target), block, headers[member.key]
                            )
                # The publish is a write too (an OME-Zarr pyramid is derived here), and it is waited for.
                with SWEEP_CLOCK.phase("wait(write)"):
                    written = write.close()
            for key in written:
                sweep = sweeps[key]
                self._swept_entries.add((str(sweep.destination.filename), sweep.group, sweep.entry))
            return written, None
        except BaseException as exception:
            write.abort(exception)
            if not isinstance(exception, Exception):
                raise  # an interrupt is not a sweep failure: no fallback, and no .tmp left behind
            if isinstance(exception, MemoryError | torch.cuda.OutOfMemoryError):
                # The fallback answers a failed region with the WHOLE case: a larger allocation than
                # the one that just failed, on the same device. Running out of memory is the one
                # failure it cannot repair, so it propagates instead.
                raise
            return set(), _stage_failure(exception)
        finally:
            write.shutdown()

    def _sweep_failed_because(self, sweep: _PendingSweep, reason: str) -> bool:
        """Record why a sweep gave up, warn, and answer ``False``: the one exit for all of them.

        The reason is kept, not only warned: a caller with a whole-volume fallback treats this as
        information, but one without (a reduction reading through this cache) has to raise, and it
        can only be as specific as what was kept here.
        """
        self._sweep_failure = f"'{sweep.group}/{sweep.entry}' could not be written region by region: {reason}"
        warnings.warn(f"{self._sweep_failure} Falling back to the whole-volume path.", stacklevel=3)
        return False

    def _sweep_depth(
        self, spatial: list[int], channels: int, plans: Sequence["_ReadStagePlan"], tile: list[int]
    ) -> int:
        """:meth:`SegmentSizer.sweep_depth` of the whole declared chain against the raw source."""
        return self._chain_sizer(spatial, channels, plans).sweep_depth(tile)

    def read_granularity(self) -> tuple[int, ...] | None:
        """The stored block this case's source reads are served in, spatial axes only, or ``None``
        when a read costs what it asks for. Read from the store's metadata once per case."""
        if self._read_granularity is _UNRESOLVED:
            granularity = self.dataset.read_granularity(self.group_src, self.name)
            self._read_granularity = None if granularity is None else tuple(granularity[1:])
        return cast(tuple[int, ...] | None, self._read_granularity)

    def read_plateau_rows(self, spatial: list[int], tolerance: float = _PLATEAU_READ_MARGIN, a: int = 0) -> int | None:
        """The shortest region height whose decomposition already reads what the tallest one reads,
        within ``tolerance``: the point past which taller regions buy no fewer source voxels.

        Closed form, from the chain's own pull maps (:func:`_pull_block_voxels`): no voxel is read.
        A region shorter than this re-reads source its neighbours already pulled; a taller one reads
        the same and only holds more, which is why this is a CAP a budget may lower and never a
        target a budget should raise. ``None`` when the chain cannot stream, where there is no
        decomposition to price.

        Never below :meth:`_sweep_rows`: a chain that pulls exactly what it lands (POINTWISE) has a
        flat curve whose plateau starts at one row, and one-row regions pay every fixed per-region
        cost for one row of work.
        """
        segments = self.sweep_segments(a, apply_augmentations=False)
        if not segments:
            return None
        segment = segments[-1]
        plateau = _plateau_rows(spatial, segment.plans, tolerance)
        if plateau is None:
            return None
        sizer = self.sizer_for(segment._replace(landing=[int(extent) for extent in spatial]))
        floor = sizer.sweep_rows()
        return max(plateau, min(floor, int(spatial[0])))

    def region_reads(self, rows: int, a: int = 0) -> "BlockReads | None":
        """What a decomposition into ``rows``-row regions costs this chain in source voxels.
        ``None`` when the chain cannot stream.

        The same aggregates the sweep sizes against (:class:`BlockReads`): ``widest_pull``, the
        source window one region materialises -- which for a chain that resamples is not the region
        and is what a fold must hold while it produces one; ``widest_excess``, what the store
        decodes above that window, a chunked one serving a window by the block-aligned hull that
        covers it; and ``total``, what all the regions read together, the figure a caller compares
        heights on.

        Closed form, from the chain's own pull maps and the store's metadata: no voxel is read.
        """
        segments = self.sweep_segments(a, apply_augmentations=False)
        if not segments:
            return None
        segment = segments[-1]
        spatial = [int(extent) for extent in segment.landing]
        tile = [max(1, min(int(rows), spatial[0])), *spatial[1:]]
        return self.sizer_for(segment).block_reads(tile)

    def sweep_block_bytes(
        self, spatial: list[int], channels: int, plans: Sequence["_ReadStagePlan"], tile: list[int], depth: int
    ) -> int:
        """:meth:`SegmentSizer.sweep_block_bytes` of the whole declared chain against the raw source.

        Public because the sizing holds this figure to the budget and a caller sizing a budget for a
        decomposition asks for it: one price, not two that drift apart.
        """
        return self._chain_sizer(spatial, channels, plans).sweep_block_bytes(tile, depth)

    def _sweep_shape(self, spatial: list[int], plans: Sequence["_ReadStagePlan"], rows: int) -> list[int]:
        """:meth:`SegmentSizer.sweep_shape` of the whole declared chain against the raw source."""
        return self._chain_sizer(spatial, 1, plans).sweep_shape(rows)

    def working_multiple(self) -> float:
        """What this chain allocates beyond what it is handed, in volumes-worth: the largest a stage
        declares for THIS case (``Transform.case_working_multiple``, whose default is the class's
        own ``working_multiple``)."""
        return max(
            (
                float(stage.case_working_multiple(self.name))
                for stage in self.transforms
                if isinstance(stage, Transform)
            ),
            default=0.0,
        )

    def _sweep_tile(
        self, spatial: list[int], channels: int, plans: Sequence["_ReadStagePlan"] = (), depth: int | None = None
    ) -> list[int]:
        """:meth:`SegmentSizer.sweep_tile` of the whole declared chain against the raw source."""
        return self._chain_sizer(spatial, channels, plans).sweep_tile(depth)

    def _sizer(
        self,
        spatial: Sequence[int],
        channels: int,
        plans: Sequence["_ReadStagePlan"],
        stages: Sequence[Stage],
        granularity: tuple[int, ...] | None,
    ) -> SegmentSizer:
        return SegmentSizer(
            spatial=[int(extent) for extent in spatial],
            channels=int(channels),
            plans=tuple(plans),
            stages=tuple(stages),
            granularity=granularity,
            case=self.name,
            group=self.group_src,
            budget_bytes=self._sweep_budget_bytes,
            device=self._chain_device,
            block_reads_memo=self._block_reads,
        )

    def _chain_sizer(self, spatial: Sequence[int], channels: int, plans: Sequence["_ReadStagePlan"]) -> SegmentSizer:
        """The single-segment view: the whole declared chain reading the raw source. Right whenever
        the chain has no ``Save`` boundary; a boundary's segments must go through :meth:`sizer_for`."""
        return self._sizer(spatial, channels, plans, tuple(self.transforms), self.read_granularity())

    def sizer_for(self, segment: SweepSegment) -> SegmentSizer:
        """The pricing view keyed to ``segment``: its own stages, and its OWN store's grain."""
        return self._sizer(
            segment.landing,
            segment.channels,
            segment.plans,
            segment.stages,
            self._entry_granularity(segment.dataset, segment.group, segment.entry),
        )

    def _entry_granularity(self, dataset: Dataset, group: str, entry: str) -> tuple[int, ...] | None:
        # The raw source keeps its own resolved-once slot (read_granularity), which is also the one
        # knob tests and callers already reset; the memo below is for the other segment sources.
        if dataset is self.dataset and group == self.group_src:
            return self.read_granularity()
        key = (str(dataset.filename), group, entry)
        if key not in self._granularities:
            granularity = None
            # A cache this run has still to write has no metadata to ask; its chunks will be the
            # very tile being sized, so its reads align by construction and None is its honest grain.
            if dataset.is_dataset_exist(group, entry):
                stored = dataset.read_granularity(group, entry)
                granularity = None if stored is None else tuple(stored[1:])
            self._granularities[key] = granularity
        return self._granularities[key]

    def _get_streamed_data(
        self,
        index: int,
        a: int,
        is_input: bool,
        apply_augmentations: bool = True,
    ) -> tuple[torch.Tensor, Attribute]:
        self._require_statistics()
        stream_source = self._resolve_patch_stream_source(a, apply_augmentations)
        if stream_source is None:
            raise RuntimeError("Patch streaming requested on a dataset manager without a streaming source.")
        if stream_source.pending_sweeps:
            raise PatchError(
                "Streamed read on a source with unmaterialized Save caches.",
                "Report this: _stream_ready() must run the sweeps before any patch flows.",
            )

        if stream_source.region_index is None:
            # POINTWISE / GLOBAL_STAT only: read the exact patch and run the whole chain on it.
            plan = self.patch.get_read_plan(stream_source.shape, index, a, is_input)
            data, attributes = stream_source.dataset.read_data_slice(
                stream_source.group, stream_source.entry, plan.data_slices
            )
            tensor = torch.from_numpy(data)
            if self._chain_device is not None:
                # The same move the region route and the whole-volume load make: a pointwise chain
                # otherwise streamed on the host while its whole-volume twin ran on the GPU, and the
                # two disagreed wherever a kernel's arithmetic differs by device (Softmax, a std).
                tensor = tensor.to(self._chain_device)
            cache_attribute = Attribute(self.cache_attributes_bak[a])
            cache_attribute.update(attributes)
            # Says the Min/Max/Mean/Std here are the planner's DISK seeds, not a mid-chain stage's
            # own bookkeeping (a Normalize pushes 'Min' for its inverse). Set before keys_before,
            # so it never persists past this read.
            cache_attribute["StatisticsSeeded"] = 1.0
            persist = a not in self._stream_attributes_persisted
            keys_before = set(cache_attribute.keys()) if persist else set()
            # Told where the patch sits, like every region: a per-voxel stage reading a companion
            # volume (a mask) reads the part that lines up with it.
            spatial = tuple(int(extent) for extent in stream_source.shape[1:])
            region = tuple(plan.data_slices[len(plan.data_slices) - len(spatial) :])
            context = RegionContext(region, region, spatial)
            for stage in stream_source.stages:
                tensor = stage.stream_region(self.name, tensor, context, cache_attribute)
            # The read plan is applied AFTER the chain, as the whole-volume path transforms before
            # Patch.get_data cuts: padding first would feed f(pad) to the model on every border patch.
            tensor = self.patch.apply_read_plan(tensor, plan)
            if persist:
                self._persist_stream_attributes(a, cache_attribute, keys_before)
            return tensor, cache_attribute

        return self._get_streamed_region_data(index, a, stream_source, is_input)

    def _finalize_stream_patch(self, tensor: torch.Tensor, index: int, a: int, is_input: bool) -> torch.Tensor:
        """Pad a streamed patch to ``patch_size`` through the same read plan the whole-volume path
        applies, so a border patch the overlap tiling left narrower is byte-identical between the
        two paths. The plan is built on ``self.shapes[a]``, the grid this copy's patches are cut on.
        """
        plan = self.patch.get_read_plan(self.shapes[a], index, a, is_input)
        return self.patch.apply_read_plan(tensor, plan)

    def _persist_stream_attributes(self, a: int, cache_attribute: Attribute, keys_before: set[str]) -> None:
        # State a transform records for its own inversion (TensorCast's source dtype) must reach the
        # persistent attribute, as it would on the whole-volume path. Only NEWLY-added keys are copied:
        # a seeded GLOBAL_STAT or a case-level geometry key must not take a patch-local value.
        persistent = self.cache_attributes[a]
        persistent_keys = set(persistent.keys())
        for key, value in cache_attribute.items():
            if key not in keys_before and key not in persistent_keys:
                dict.__setitem__(persistent, key, value)
        self._stream_attributes_persisted.add(a)

    def _get_streamed_region_data(
        self,
        index: int,
        a: int,
        stream_source: _PatchStreamSource,
        is_input: bool,
    ) -> tuple[torch.Tensor, Attribute]:
        """Patch-native region chain: one target patch replayed through the composed region plans,
        padded back to ``patch_size`` like the whole-volume path (see ``_replay_streamed_region``,
        which a Save sweep drives with slab targets instead of patch targets)."""
        if self._expand is not None and self._records_source is not stream_source:
            self._refold_copy_records(a, stream_source)
        target_slices = tuple(self.patch.read_slices(a, index, self.shapes[a]))
        # Each patch re-runs the chain from the state the whole-volume pass started from: the case as
        # stored (plus planned stats), never the live attribute: that one carries the chain's own
        # output.
        persist = a not in self._stream_attributes_persisted
        tensor, cache_attribute, keys_before = self._replay_streamed_region(
            stream_source,
            target_slices,
            Attribute(self.cache_attributes_bak[a]),
            self.cache_attributes[a] if persist else None,
        )
        tensor = self._finalize_stream_patch(tensor, index, a, is_input)
        if persist:
            self._persist_stream_attributes(a, cache_attribute, keys_before)
        return tensor, cache_attribute

    def _refold_copy_records(self, a: int, stream_source: _PatchStreamSource) -> None:
        """Re-fold copy ``a``'s chain state before replaying a region of it.

        A stage keys its per-case records by the CASE name (a stored transform is looked up by
        it), so the copies of an Expand share one key and the last WALK's records win. The write
        sweeps re-plan before sweeping and the whole-volume path re-records at call time; the
        patch replay is the consumer left over, and reading two copies interleaved would otherwise
        hand one copy the other's grids. Headers only: no voxel is read.

        Once per change of copy, not per patch: consecutive patches nearly always belong to one
        copy, and ``_records_source`` says whose records the stages hold until a fold or a
        whole-volume call moves them.
        """
        if not stream_source.stages:
            return
        shape = list(stream_source.stage_plans[0].in_shape)
        state = Attribute(self.cache_attributes_bak[a])
        for stage in stream_source.stages:
            shape = self._fold_case_state(stage, shape, state)
        self._records_source = stream_source

    def _replay_streamed_region(
        self,
        stream_source: _PatchStreamSource,
        target_slices: tuple[slice, ...],
        cache_attribute: Attribute,
        case_attribute: Attribute | None,
    ) -> tuple[torch.Tensor, Attribute, set[str]]:
        """Read the source region a target region pulls and run the chain forward on it.

        Three steps, separable because only the middle one touches the store and only the last one
        touches a stage: :meth:`_region_spans` folds the pull maps back to the stored volume,
        :meth:`_read_streamed_region` reads what they name, :meth:`_apply_streamed_region` runs the
        chain over it. A sweep drives the three on different threads (see :class:`_ReadAhead`).
        """
        spans = self._region_spans(stream_source, target_slices)
        tensor, attributes = self._read_streamed_region(stream_source, spans)
        return self._apply_streamed_region(stream_source, spans, tensor, attributes, cache_attribute, case_attribute)

    def plan_patch_reads(self, entries: Sequence[tuple[int, int]], is_input: bool, apply_augmentations: bool) -> None:
        """Declare the store reads this case's patches will make, in the order one process will make
        them: ``entries`` is its ``(copy, patch)`` sequence, from the loader's own order (see
        :class:`~konfai.data.data_manager.PatchReadOrder`). Called once, as the case is entered.

        The reads are named from the plans alone, so nothing is read here and no voxel of a patch
        still to come is touched. A copy whose Save caches are not materialized yet is left out: the
        sweep that materializes them declares its own reads, and what the patches then read is the
        cache, not this source.
        """
        if self.loaded or not entries or not self._stream_ready(entries[0][0], apply_augmentations):
            return
        reads = []
        for a, index in entries:
            source = self._resolve_patch_stream_source(a, apply_augmentations)
            if source is None or source.pending_sweeps:
                continue
            reads.append((source, self._patch_read_spans(source, index, a, is_input)))
        self._declare_region_reads(reads)

    def _patch_read_spans(
        self, stream_source: _PatchStreamSource, index: int, a: int, is_input: bool
    ) -> list[list[slice]]:
        """The region each stage reads for patch ``index`` of copy ``a``, the first being the window
        read from the store: what :meth:`_get_streamed_data` reads, without reading it. An exact-patch
        chain hands every stage the patch itself, which is the region it reads."""
        if stream_source.region_index is None:
            plan = self.patch.get_read_plan(stream_source.shape, index, a, is_input)
            region = plan.data_slices[len(plan.data_slices) - (len(stream_source.shape) - 1) :]
            return [list(region) for _ in range(len(stream_source.stage_plans) + 1)]
        return self._region_spans(stream_source, tuple(self.patch.read_slices(a, index, self.shapes[a])))

    def _declare_region_reads(self, reads: Sequence[tuple[_PatchStreamSource, list[list[slice]]]]) -> None:
        """Tell the stores, and the stages, the region reads about to happen in the order they will.

        A store that caches decoded blocks then keeps what a later read asks for again and drops what
        none does (:meth:`~konfai.utils.dataset.Dataset.plan_region_reads`), and a stage reading a
        companion volume beside its region (:class:`~konfai.data.transform.Mask`) declares those reads
        too. Grouped by the entry read and by the stage handed the region, so the copies of one case
        interleaved over one store are declared as the single sequence that store will serve.
        """
        by_entry: dict[tuple[Dataset, str, str], list[tuple[slice, ...]]] = {}
        by_stage: dict[int, tuple[Stage, list[RegionContext]]] = {}  # by identity: a Stage need not be hashable
        for source, spans in reads:
            lead = [slice(None)] * (len(source.shape) - len(spans[-1]))
            by_entry.setdefault((source.dataset, source.group, source.entry), []).append((*lead, *spans[0]))
            for index, (stage, plan) in enumerate(zip(source.stages, source.stage_plans, strict=True)):
                if plan.kind is LocalityKind.CROP:
                    continue  # handed no region: its remap is its action
                by_stage.setdefault(id(stage), (stage, []))[1].append(
                    plan.region_context(spans[index], spans[index + 1])
                )
        for (dataset, group, entry), windows in by_entry.items():
            dataset.plan_region_reads(group, entry, windows)
        for stage, contexts in by_stage.values():
            stage.plan_region_reads(self.name, contexts)

    def _region_spans(self, stream_source: _PatchStreamSource, target_slices: tuple[slice, ...]) -> list[list[slice]]:
        """The region each stage of the chain reads for ``target_slices``, the last being the target
        itself and the first the window to read from the store.

        Closed form, from the plans' own pull maps, EXCEPT for a stage that sizes its window from the
        data (``run_pull``, a displacement field): that one reads, and its sizing read is also its
        sampling read, so its spans cannot be folded ahead of the chain.
        """
        spans: list[list[slice]] = [list(target_slices)]
        for plan in reversed(stream_source.stage_plans):
            pull = plan.run_pull or plan.pull
            spans.append(pull(tuple(spans[-1])) if pull is not None else list(spans[-1]))
        spans.reverse()
        return spans

    def _read_streamed_region(
        self, stream_source: _PatchStreamSource, spans: list[list[slice]]
    ) -> tuple[torch.Tensor, Attribute]:
        """The stored region ``spans[0]`` names, on the chain's device. The store and nothing else:
        no stage of the chain is touched here, which is what lets a sweep read one block ahead."""
        n_prefix = len(stream_source.shape) - len(spans[-1])
        data_slices = tuple([slice(None)] * n_prefix + spans[0])
        data, attributes = stream_source.dataset.read_data_slice(stream_source.group, stream_source.entry, data_slices)
        tensor = torch.from_numpy(data)
        if self._chain_device is not None:
            tensor = tensor.to(self._chain_device)
        return tensor, attributes

    def _apply_streamed_region(
        self,
        stream_source: _PatchStreamSource,
        spans: list[list[slice]],
        tensor: torch.Tensor,
        attributes: Attribute,
        cache_attribute: Attribute,
        case_attribute: Attribute | None,
    ) -> tuple[torch.Tensor, Attribute, set[str]]:
        """The chain forward over a region already read: the region's scope is seeded from what the
        read returned, then :meth:`_run_streamed_stages` walks the stages over it.

        ``cache_attribute`` is the region's scope, evolved by the chain; ``case_attribute``, when
        given, receives each region stage's case-level geometry (from the full shape). Returns the
        tensor, the evolved scope, and the keys the scope held before (what the chain added is what
        the caller may persist).
        """
        cache_attribute.update(attributes)
        cache_attribute["StatisticsSeeded"] = 1.0  # same contract as the pointwise route above
        keys_before = set(cache_attribute.keys())
        tensor = self._run_streamed_stages(
            stream_source.stages, stream_source.stage_plans, spans, tensor, cache_attribute, case_attribute
        )
        return tensor, cache_attribute, keys_before

    def _run_streamed_stages(
        self,
        stages: Sequence[Stage],
        plans: Sequence["_ReadStagePlan"],
        spans: list[list[slice]],
        tensor: torch.Tensor,
        cache_attribute: Attribute,
        case_attribute: Attribute | None,
    ) -> torch.Tensor:
        """Walk STAGES over a region already read, each on the region pair the fold computed for it:
        HALO reads the enlarged region and is cropped back, ORIENTATION remaps what it read, a
        CROP's remap is its action (not re-applied), REGRID interpolates to its target extent, a
        per-voxel stage is told where its region sits.

        The one dispatch: a chain read through the store and a member's per-copy tail (which reads
        on the landing, so its regions are the blocks themselves) both come here.
        """
        for stage, plan, source, target in zip(stages, plans, spans[:-1], spans[1:], strict=True):
            if not plan.kind.is_region:
                # The span is handed over rather than dropped: a stage reading a second aligned
                # volume needs to know WHICH part of it lines up with this region, and the
                # dispatcher is the only thing that knows. The default hook ignores it.
                tensor = stage.stream_region(self.name, tensor, plan.region_context(source, target), cache_attribute)
                continue
            # A region stage's geometry writes describe the region's extent, not the volume's: give it
            # a throwaway scope, and write the case-level answer once from the FULL shape below
            # (write_stream_cache_attribute).
            scoped = Attribute(cache_attribute)
            if plan.kind is not LocalityKind.CROP:
                # A HALO stage is handed the ENLARGED region it asked for, and told so: what it
                # returns is cropped back to the target just below.
                tensor = stage.stream_region(self.name, tensor, plan.region_context(source, target), scoped)
                if plan.kind is LocalityKind.HALO:
                    lead = tensor.dim() - len(target)
                    crop = [slice(t.start - s.start, t.stop - s.start) for t, s in zip(target, source, strict=False)]
                    tensor = tensor[(*[slice(None)] * lead, *crop)]
            if case_attribute is not None:
                stage.write_stream_cache_attribute(case_attribute, list(plan.in_shape), self.name)
                self._check_region_geometry_reaches_the_case(stage, scoped, cache_attribute)

        return tensor

    def _check_region_geometry_reaches_the_case(
        self, region_stage: Stage, scoped: Attribute, cache_attribute: Attribute
    ) -> None:
        """Refuse a region stage that records geometry nowhere the case can read it.

        A region stage is handed a patch, so what it records about the extent is one patch's answer:
        the scope it records into is thrown away, and ``write_stream_cache_attribute`` is what reaches
        the case. A stage that records in ``__call__`` alone streams a whole run and leaves the case
        the geometry it was stored with. Recording in both is what a reorientation does: the check
        is on recording in neither.
        """
        recorded = {key for key in scoped.keys() if key not in cache_attribute or scoped[key] != cache_attribute[key]}
        if not recorded:
            return
        if type(region_stage).write_stream_cache_attribute is not Transform.write_stream_cache_attribute:
            return
        raise PatchError(
            f"'{type(region_stage).__name__}' recorded {sorted(recorded)} on the scope a streamed region"
            " is handed, which is dropped, and implements no write_stream_cache_attribute().",
            "Record the case's answer in write_stream_cache_attribute(): it is given the whole volume's"
            " shape, where a patch's extent cannot say it.",
        )

    def unload(self) -> None:
        self.data.clear()
        self.augmented_data.clear()
        self.loaded = False
        self.augmentationLoaded = self.total_augmentations == 0

    def unload_augmentation(self) -> None:
        self.augmented_data.clear()
        self.augmentationLoaded = self.total_augmentations == 0

    def get_data(
        self,
        index: int,
        a: int,
        patch_transforms: list[Transform],
        is_input: bool,
        apply_augmentations: bool = True,
    ) -> torch.Tensor:
        if not self.loaded and self._stream_ready(a, apply_augmentations):
            data, _ = self._get_streamed_data(index, a, is_input, apply_augmentations)
        else:
            if not self.loaded:
                # A failed Save sweep lands here past the buffered-path guard (which saw a pending
                # plan and skipped the full load): load classically, which writes the caches too.
                self.load(self.transforms, self.data_augmentations_list, load_augmentations=False)
            data = self.patch.get_data(self._get_tensor(a), index, a, is_input)
        if patch_transforms:
            # Per-patch scope: writing to the shared case attribute would freeze the first patch's
            # derived statistic for every other patch. A case-level statistic (`Standardize(lazy=True)`
            # in `transforms`) is inherited through the copy.
            cache_attribute = Attribute(self.cache_attributes[a])
            for transform_function in patch_transforms:
                data = transform_function(self.name, data, cache_attribute)
        return data

    def get_size(self, a: int) -> int:
        return self.patch.get_size(a)
