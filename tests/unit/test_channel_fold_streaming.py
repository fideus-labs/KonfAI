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

"""A stage that folds the channel axis away (``Sum(dim=0)``, ``MergeLabels``: the 5-task
TotalSegmentator merge) hands back a block of the spatial rank. The whole-volume write reads that as
a single-channel image; the region write reads it the same way, ships the same bytes and the same
header, and refuses every rank the header does not name."""

from pathlib import Path

import numpy as np
import pytest
import torch
from konfai.data import patching
from konfai.data.materialize import CaseMaterializer, Verdict
from konfai.data.patching import DatasetManager
from konfai.data.transform import LocalityKind, MergeLabels, PatchLocality, Save, Sum, Transform
from konfai.utils.dataset import Attribute, Dataset

#: One case, five models, disjoint label ranges: what a ``combine: Concat`` ensemble hands the fold.
_MODEL_CHANNELS = [3, 4, 2, 5, 3]


@pytest.fixture(autouse=True)
def several_regions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two rows per region, so every case here is written in several pieces."""
    monkeypatch.setattr(patching, "SWEEP_SLAB_ROWS", 2)


class _ExtraAxis(Transform):
    """A stage whose block keeps its spatial shape and gains an axis, a rank no rule covers."""

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        return PatchLocality(LocalityKind.POINTWISE)

    def transform_shape(self, group_src: str, name: str, shape: list[int], cache_attribute: Attribute) -> list[int]:
        return shape

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        return tensor.unsqueeze(0)


def _source(tmp_path: Path, geometry: bool) -> Dataset:
    """A five-model label stack, with or without the geometry that names its spatial rank."""
    attributes = Attribute()
    if geometry:
        attributes["Origin"] = np.asarray([10.0, 20.0, 30.0])
        attributes["Spacing"] = np.asarray([0.5, 1.5, 2.0])
        attributes["Direction"] = np.eye(3, dtype=np.float64).reshape(-1)
    attributes["number_of_channels_per_model"] = np.asarray(_MODEL_CHANNELS)
    rng = np.random.default_rng(0)
    volume = rng.integers(0, 5, size=(len(_MODEL_CHANNELS), 12, 10, 8)).astype(np.uint8)
    dataset = Dataset(tmp_path / "source", "h5")
    dataset.write("CT", "CASE_000", volume, attributes)
    return dataset


def _write(source: Dataset, transforms: list[Transform], out: Path, whole_volume: bool) -> Verdict:
    manager = DatasetManager(
        index=0,
        group_src="CT",
        group_dest="CT",
        name="CASE_000",
        dataset=source,
        patch=None,
        transforms=[*transforms, Save(f"{out}:h5")],
        data_augmentations_list=[],
    )
    return CaseMaterializer(manager).materialize(prefer_whole=whole_volume)


@pytest.mark.parametrize("fold", [Sum(dim=0), MergeLabels()], ids=["Sum", "MergeLabels"])
def test_a_channel_fold_streams_the_store_the_whole_volume_pass_writes(tmp_path: Path, fold: Transform) -> None:
    """Same bytes and same header keys on both routes."""
    source = _source(tmp_path, geometry=True)

    assert _write(source, [fold], tmp_path / "swept", whole_volume=False) is Verdict.STREAM
    assert _write(source, [fold], tmp_path / "whole", whole_volume=True) is Verdict.LOAD

    swept, swept_header = Dataset(tmp_path / "swept", "h5").read_data("CT", "CASE_000")
    whole, whole_header = Dataset(tmp_path / "whole", "h5").read_data("CT", "CASE_000")
    assert swept.shape == (1, 12, 10, 8)
    assert swept.tobytes() == whole.tobytes()
    assert sorted(swept_header.keys()) == sorted(whole_header.keys())
    # The key describes the ensemble the fold consumed: an output holding one channel that still
    # carried it would send a later fold down the ensemble branch.
    assert not any(key.startswith("number_of_channels_per_model") for key in swept_header.keys())


@pytest.mark.parametrize(
    ("chain", "geometry"),
    [([Sum(dim=0)], False), ([_ExtraAxis()], True)],
    ids=["no-geometry-names-the-spatial-rank", "a-rank-above-channel-first"],
)
def test_a_block_of_an_unwritable_rank_refuses_and_falls_back(
    tmp_path: Path, chain: list[Transform], geometry: bool
) -> None:
    """Written anyway, the header would take the block's first spatial extent for a channel count and
    publish that many broadcast copies: a store whose rank disagrees with the whole-volume path, with
    no exception raised."""
    source = _source(tmp_path, geometry)

    with pytest.warns(UserWarning, match="Falling back to the whole-volume path"):
        assert _write(source, chain, tmp_path / "refused", whole_volume=False) is Verdict.WHOLE_VOLUME
    assert _write(source, chain, tmp_path / "whole", whole_volume=True) is Verdict.LOAD

    # The aborted sweep left no debris, and the fallback wrote the store the whole-volume route does.
    assert not list((tmp_path / "refused").glob("**/*.tmp"))
    refused, _ = Dataset(tmp_path / "refused", "h5").read_data("CT", "CASE_000")
    whole, _ = Dataset(tmp_path / "whole", "h5").read_data("CT", "CASE_000")
    assert refused.shape == whole.shape
    assert refused.tobytes() == whole.tobytes()
