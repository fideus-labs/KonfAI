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

"""Shared fixtures for the unit suite: the in-memory streaming dataset stub and the TTA case driver."""

from types import SimpleNamespace
from typing import ClassVar, cast

import numpy as np
import pytest
import torch
from konfai.data.augmentation import DataAugmentationsList
from konfai.data.data_manager import DatasetIter
from konfai.predictor import Mean, OutSameAsGroupDataset, Reduction
from konfai.utils.dataset import Attribute, Dataset
from konfai.utils.utils import get_patch_slices_from_shape


class StreamingDatasetStub:
    """In-memory dataset serving whole reads, region reads, and statistics, with identity geometry.

    The read counters (``full_reads``/``patch_reads``/``stats_reads``) let a test assert which
    access path was taken.
    """

    def __init__(self, volume: np.ndarray) -> None:
        self.volume = volume
        self.full_reads = 0
        self.patch_reads = 0
        self.stats_reads = 0

    def _attributes(self) -> Attribute:
        spatial = self.volume.ndim - 1
        attribute = Attribute()
        attribute["Origin"] = np.zeros(spatial)
        attribute["Spacing"] = np.ones(spatial)
        attribute["Direction"] = np.eye(spatial).flatten()
        return attribute

    def get_infos(self, group_src: str, name: str) -> tuple[list[int], Attribute]:
        return list(self.volume.shape), self._attributes()

    def read_data(self, group_src: str, name: str) -> tuple[np.ndarray, Attribute]:
        self.full_reads += 1
        return self.volume.copy(), self._attributes()

    def read_data_slice(self, group_src: str, name: str, slices: tuple[slice, ...]) -> tuple[np.ndarray, Attribute]:
        self.patch_reads += 1
        return self.volume[slices].copy(), self._attributes()

    def read_data_statistics(self, group_src: str, name: str, channels: list[int] | None = None) -> dict[str, float]:
        self.stats_reads += 1
        data = self.volume if channels is None else self.volume[channels]
        return {
            "min": float(data.min()),
            "max": float(data.max()),
            "mean": float(data.mean()),
            "std": float(data.std(ddof=1)),
        }

    def bounded_region_reads(self, group_src: str, name: str) -> bool:
        del group_src, name
        return True  # in-memory: a slice reads the slice


@pytest.fixture
def streaming_dataset_stub() -> type[StreamingDatasetStub]:
    """The stub class itself: tests construct instances and inspect the read counters."""
    return StreamingDatasetStub


_TTA_SHAPE = [6, 4, 3]
_TTA_PATCH_SIZE = [2, 4, 3]
_TTA_OVERLAP = 1
_TTA_CHANNELS = 2


def _tta_geometry_attribute() -> Attribute:
    attribute = Attribute()
    attribute["Origin"] = np.zeros(3)
    attribute["Spacing"] = np.ones(3)
    attribute["Direction"] = np.eye(3).flatten()
    return attribute


def _tta_augmentations(
    augmentation, nb: int = 1, shape: list[int] | None = None, case_index: int = 0
) -> DataAugmentationsList:
    """A one-augmentation list with its draw made for ``case_index`` (prob 1, so every copy is selected)."""
    augmentation.load(1.0)
    augmentations = DataAugmentationsList(nb=nb, data_augmentations={})
    augmentations.data_augmentations = [augmentation]
    augmentation.state_init(
        case_index, [list(shape or _TTA_SHAPE) for _ in range(nb)], [Attribute() for _ in range(nb)]
    )
    return augmentations


def _tta_make_patches(volume: torch.Tensor, patch_slices) -> list[torch.Tensor]:
    """Cut ``volume`` into model-sized patches, border patches zero-padded up to the patch extent."""
    patches = []
    for patch_slice in patch_slices:
        patch = volume[(slice(None), *patch_slice)]
        padding = []
        for dim, s in reversed(list(enumerate(patch_slice))):
            padding += [0, _TTA_PATCH_SIZE[dim] - (s.stop - s.start)]
        patches.append(torch.nn.functional.pad(patch, padding) if any(padding) else patch)
    return patches


def _drive_tta(
    tmp_path,
    monkeypatch,
    *,
    augmentation,
    streamed: bool,
    reduction: Reduction | None = None,
    transforms=(),
    after=(),
    dtype: torch.dtype = torch.float32,
    patch_combine=None,
    case_index: int = 0,
    file_format: str = "h5",
    worth_gate: bool = False,
):
    """Push one TTA case (identity copy + one augmented copy) patch by patch through ``add_layer``
    against an h5 sink, interleaved along the slab axis exactly as the prediction mapping orders it,
    and read the written entry back.

    The forward ``transforms`` run once on the volume (stacking the attribute as the input pipeline
    would) and the result plays the model output; the augmented copy is the augmentation's own
    ``compute`` of it. Returns the written tensor and whether the whole-volume path produced it."""
    monkeypatch.setenv("KONFAI_STREAMED_WRITES", "1" if streamed else "0")
    monkeypatch.setenv("KONFAI_config_file", "unused.yml")
    monkeypatch.setenv("KONFAI_CONFIG_MODE", "Done")
    # Toy volumes are far below the production worth-streaming threshold: zero it so these tests
    # exercise the streamed machinery; ``worth_gate`` keeps the production threshold instead.
    if worth_gate:
        monkeypatch.delenv("KONFAI_STREAM_WORTH_THRESHOLD", raising=False)
    else:
        monkeypatch.setenv("KONFAI_STREAM_WORTH_THRESHOLD", "0")

    attribute = _tta_geometry_attribute()
    volume = torch.from_numpy(
        np.random.default_rng(0).standard_normal((_TTA_CHANNELS, *_TTA_SHAPE)).astype(np.float32)
    ).to(dtype)
    for transform in transforms:
        volume = transform("CASE_000", volume, attribute)
    augmentations = _tta_augmentations(augmentation, shape=list(volume.shape[1:]), case_index=case_index)
    copies = [volume, augmentation.compute("CASE_000", case_index, 0, volume)]
    patch_slices, _ = get_patch_slices_from_shape(_TTA_PATCH_SIZE, list(volume.shape[1:]), _TTA_OVERLAP)
    patches = [_tta_make_patches(copy, patch_slices) for copy in copies]

    class DummyPatch:
        patch_size: ClassVar[list[int]] = list(_TTA_PATCH_SIZE)

        @staticmethod
        def get_patch_slices(index_augmentation: int):
            del index_augmentation
            return patch_slices

        @staticmethod
        def get_sweep_axis(index_augmentation: int) -> int:
            # These grids are cut along axis 0, which is what these doubles' slices assume.
            del index_augmentation
            return 0

    class DummyManager:
        index = case_index
        name = "CASE_000"
        patch = DummyPatch()
        shapes: ClassVar[list[list[int]]] = [list(volume.shape[1:]), list(volume.shape[1:])]
        cache_attributes: ClassVar[list[Attribute]] = [Attribute(), Attribute()]

    class DummyDatasetIter:
        data_augmentations_list: ClassVar[list[DataAugmentationsList]] = [augmentations]
        groups_src: ClassVar[dict] = {
            "src": {"dest": SimpleNamespace(transforms=list(transforms), patch_transforms=[])}
        }

        @staticmethod
        def get_dataset_from_index(group_dest: str, index: int):
            return DummyManager()

    output_dataset = OutSameAsGroupDataset(
        same_as_group="src:dest",
        dataset_filename=f"{tmp_path}/output.h5:h5" if file_format == "h5" else f"{tmp_path}/output:{file_format}",
        group="out",
        patch_combine=None,
        reduction="Mean",
    )
    output_dataset.reduction = reduction if reduction is not None else Mean()
    output_dataset.nb_data_augmentation = 2
    output_dataset.after_reduction_transforms = list(after)
    for transform in output_dataset.after_reduction_transforms:
        transform.set_datasets([output_dataset])
    if patch_combine is not None:
        patch_combine.set_patch_config(list(_TTA_PATCH_SIZE), _TTA_OVERLAP)
        output_dataset.patch_combine = patch_combine

    dataset_iter = cast(DatasetIter, DummyDatasetIter())
    order = sorted((patch_slices[p][0].start, a, p) for a in range(2) for p in range(len(patch_slices)))
    whole_volume = False
    for _, a, p in order:
        output_dataset.add_layer(0, a, p, patches[a][p].clone(), dataset_iter, Attribute(attribute), [_TTA_CHANNELS])
        if output_dataset.is_done(0):
            result = output_dataset.get_output(0, [_TTA_CHANNELS], dataset_iter)
            output_dataset.write_prediction(0, "CASE_000", result)
            whole_volume = True

    output_dataset.finalize_writes()
    store = f"{tmp_path}/output.h5" if file_format == "h5" else f"{tmp_path}/output"
    data, _ = Dataset(store, file_format).read_data("out", "CASE_000")
    return torch.from_numpy(data), whole_volume


@pytest.fixture
def drive_tta():
    """The TTA case driver as a callable: ``drive_tta(tmp_path, monkeypatch, *, augmentation, streamed, ...)``."""
    return _drive_tta
