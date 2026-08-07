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

"""A region read through an unwritten ``Save`` sweeps it first, instead of failing on it.

``Save`` is the remedy KonfAI recommends in its own refusal messages: a statistic that follows a
value-changing stage cannot stream, and cutting the chain with a ``Save`` makes the cache the source
the statistic seeds from. A ``read_region`` that replayed its stream source
directly, without consuming the pending sweeps it carries, would plan STREAM and fail at the first
region: the worst shape of answer, because the plan is what a caller reads before committing.

The whole-volume result is the reference here, as everywhere on this path: streaming is an
optimisation of memory, never of meaning.
"""

import numpy as np
import pytest
import torch
from konfai.data.patching import DatasetManager
from konfai.data.transform import Clip, Save, Standardize
from konfai.utils.dataset import Attribute, Dataset
from konfai.utils.ome_zarr import write_ome_zarr


def _source(tmp_path):
    volume = np.arange(1 * 8 * 8 * 8, dtype=np.float32).reshape(1, 8, 8, 8) - 200.0
    store = tmp_path / "src" / "CASE_000" / "CT.ome.zarr"
    store.parent.mkdir(parents=True)
    write_ome_zarr(store, volume, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0))
    return volume


def _manager(tmp_path, transforms):
    return DatasetManager(
        index=0,
        group_src="CT",
        group_dest="CT",
        name="CASE_000",
        dataset=Dataset(tmp_path / "src", "omezarr"),
        patch=None,
        transforms=transforms,
        data_augmentations_list=[],
    )


def test_a_region_read_through_an_unwritten_save_matches_the_whole_volume(tmp_path):
    """[Clip, Save, Standardize]: the pair that cannot stream, cut by the Save that fixes it.

    Standardize wants the whole-volume Min/Max of its OWN input, and Clip has already changed them --
    so without the Save the chain refuses. With it, the cache becomes the source the statistic reads,
    and the region path has to sweep that cache before it can serve anything.
    """
    volume = _source(tmp_path)
    chain = [
        Clip(min_value=-50.0, max_value=50.0),
        Save(dataset=f"{tmp_path / 'work'}:h5"),
        Standardize(),
    ]
    manager = _manager(tmp_path, chain)

    assert manager.can_stream_patch(0, apply_augmentations=False)
    region = manager.read_region((slice(0, 4), slice(0, 8), slice(0, 8)), 0, False)

    reference = torch.from_numpy(volume.copy())
    attribute = Attribute()
    for stage in chain:
        if not isinstance(stage, Save):
            reference = stage("CASE_000", reference, attribute)
    assert region.shape == (1, 4, 8, 8)
    assert torch.equal(region, reference[:, 0:4])


def test_the_swept_cache_is_written_where_the_save_named_it(tmp_path):
    """The sweep is not a private detail: it leaves the cache on disk, which is the point of a Save --
    a second read resolves from the boundary instead of replaying the prefix."""
    _source(tmp_path)
    manager = _manager(
        tmp_path,
        [Clip(min_value=-50.0, max_value=50.0), Save(dataset=f"{tmp_path / 'work'}:h5"), Standardize()],
    )
    manager.read_region((slice(0, 2), slice(0, 8), slice(0, 8)), 0, False)
    assert Dataset(tmp_path / "work", "h5").is_dataset_exist("CT", "CASE_000")


def test_a_sweep_failure_does_not_survive_the_flip_to_rewriting_saves(tmp_path, monkeypatch):
    """A failed sweep is a verdict about one boundary answer, and ``--overwrite`` asks the other.

    Kept across the flip it condemns every Save of the new mode to the whole-volume path: the one
    path a case too large to assemble cannot take.
    """
    _source(tmp_path)
    chain = [
        Clip(min_value=-50.0, max_value=50.0),
        Save(dataset=f"{tmp_path / 'work'}:h5"),
        Standardize(),
    ]
    manager = _manager(tmp_path, chain)

    monkeypatch.setattr(
        manager,
        "_materialize_save",
        lambda sweep: manager._sweep_failed_because(sweep, "OSError: no space left on device"),
    )
    with pytest.warns(UserWarning, match="could not be written region by region"):
        assert not manager._stream_ready(0, apply_augmentations=False)
    monkeypatch.undo()

    manager._set_rewrite(True)

    assert manager._stream_ready(0, apply_augmentations=False)
