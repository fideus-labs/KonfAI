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

"""A statistic the store cannot seed is measured once, and the chain streams from there.

``Standardize`` after a value-changing stage wants the statistics of its OWN input, which the stored
volume no longer describes, so the chain cannot stream and every patch of the case costs a whole
volume. The number it needs is the one a whole-volume pass computes anyway: kept past the pass, it
seeds the regions, and the case pays one materialization instead of one per read.

The whole-volume result is the reference, as everywhere on this path: streaming is an optimisation of
memory, never of meaning.
"""

import numpy as np
import torch
from konfai.data.transform import Clip, Standardize
from konfai.utils.dataset import Dataset
from konfai.utils.ome_zarr import write_ome_zarr
from oracle_support import manager


def _source(tmp_path):
    volume = np.arange(1 * 8 * 8 * 8, dtype=np.float32).reshape(1, 8, 8, 8) - 200.0
    store = tmp_path / "src" / "CASE_000" / "CT.ome.zarr"
    store.parent.mkdir(parents=True)
    write_ome_zarr(store, volume, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0))
    return volume


def test_a_measured_statistic_lets_the_chain_stream_and_lands_on_the_same_values(tmp_path):
    _source(tmp_path)
    chain = [Clip(min_value=-50.0, max_value=50.0), Standardize()]
    case = manager(Dataset(tmp_path / "src", "omezarr"), chain)

    # Nothing has measured it yet, and the refusal says which stage and why.
    assert not case.can_stream_patch(0, apply_augmentations=False)
    refusal = case.stream_refusal(0, apply_augmentations=False)
    assert "Standardize" in refusal
    assert "the stored volume's statistic is not this stage's input" in refusal

    case.load(chain, [], load_augmentations=False)
    whole = case.data[0]

    # The pass measured it, so the same chain streams now.
    assert case.can_stream_patch(0, apply_augmentations=False)
    assert case.stream_refusal(0, apply_augmentations=False) is None

    target = (slice(0, 4), slice(0, 8), slice(0, 8))
    region = case.read_region(target, 0, False)
    assert region.shape == (1, 4, 8, 8)
    assert torch.equal(region, whole[:, 0:4])


def test_the_statistic_kept_is_the_stage_s_own_input_not_the_stored_volume(tmp_path):
    """The two numbers differ, and taking the wrong one is a silently wrong training.

    ``Clip`` moves the mean of this volume, so a region seeded with the STORED statistic lands
    somewhere else entirely. Pinning the gap keeps the seed honest if the source of it ever moves.
    """
    volume = _source(tmp_path)
    chain = [Clip(min_value=-50.0, max_value=50.0), Standardize()]
    case = manager(Dataset(tmp_path / "src", "omezarr"), chain)
    case.load(chain, [], load_augmentations=False)

    clipped = np.clip(volume, -50.0, 50.0)
    assert abs(float(clipped.mean()) - float(volume.mean())) > 1.0

    region = case.read_region((slice(0, 4), slice(0, 8), slice(0, 8)), 0, False)
    seeded_with_the_stored_mean = (clipped[:, 0:4] - volume.mean()) / volume.std()
    assert not np.allclose(region.numpy(), seeded_with_the_stored_mean, atol=1e-3)


def test_the_cache_is_sized_on_what_the_chain_lands_on(tmp_path):
    """A resample lands 27 times smaller than it reads, and the cache holds what it lands on.

    Sized on the stored shape, a dataset that fits is refused the cache and pays the streamed route
    instead.
    """
    from konfai.data.transform import Resample

    _source(tmp_path)
    case = manager(Dataset(tmp_path / "src", "omezarr"), [Resample(spacing=[2.0, 2.0, 2.0])])
    assert case.base_shape == [1, 8, 8, 8]
    assert case.spatial_shape == [4, 4, 4]
