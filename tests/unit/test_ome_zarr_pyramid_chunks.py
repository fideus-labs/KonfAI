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

"""The pyramid derivation never hands wasm an unbounded block, and the layout changes no value.

The failure this pins down was found on a real round: a slab-written store's chunks (31 rows, 1331
columns) divide no factor, ngff-zarr re-aligns them into blocks of its own choosing, and on an
ExaSPIM-sized volume the 32-bit wasm sandbox traps -- which ``on_fallback: error`` rightly turns
into a dead round. The fix bounds the blocks; these tests hold the bound AND the values.
"""

import numpy as np
import pytest
from konfai.data import read_ome_zarr_data_slice, write_ome_zarr
from konfai.utils.ome_zarr import append_ome_zarr_levels


def _block_mean(volume: np.ndarray, factor: int) -> np.ndarray:
    """ITK's bin shrink: the mean of each aligned factor**3 window, remainder dropped."""
    _, z, y, x = volume.shape
    zc, yc, xc = z // factor, y // factor, x // factor
    windows = volume[:, : zc * factor, : yc * factor, : xc * factor]
    return windows.reshape(volume.shape[0], zc, factor, yc, factor, xc, factor).mean(axis=(2, 4, 6))


def test_append_levels_survives_a_factor_defying_extent(tmp_path):
    """514 = 2 mod 4: no uniform chunking divides it, and the tail block a naive rechunk leaves
    (2 rows) has an EMPTY shrink output, which traps the wasm sandbox. The explicit blocks absorb
    the tail; the level still equals the whole-volume block mean."""
    rng = np.random.default_rng(5)
    volume = rng.random((1, 514, 33, 41)).astype(np.float32) * 100
    store = tmp_path / "tail.ome.zarr"
    write_ome_zarr(store, volume, spacing=[1.0, 1.0, 1.0], origin=[0.0, 0.0, 0.0], chunks=(1, 2, 33, 41))
    append_ome_zarr_levels(store, [4])
    level1, _ = read_ome_zarr_data_slice(store, (slice(None),) * 4, level=1)
    np.testing.assert_allclose(level1, _block_mean(volume, 4), rtol=1e-6)


def test_write_with_awkward_extents_matches_block_mean(tmp_path):
    rng = np.random.default_rng(3)
    volume = rng.random((1, 31, 47, 53)).astype(np.float32) * 4000
    store = tmp_path / "vol.ome.zarr"
    write_ome_zarr(store, volume, spacing=[1.0, 1.0, 1.0], origin=[0.0, 0.0, 0.0], scale_factors=[4])
    level1, _ = read_ome_zarr_data_slice(store, (slice(None),) * 4, level=1)
    np.testing.assert_allclose(level1, _block_mean(volume, 4), rtol=1e-6)


def test_append_levels_on_slab_chunked_store_matches_block_mean(tmp_path):
    """A store chunked the way the streamed writer leaves it (rows that divide nothing) grows its
    pyramid without wasm ever seeing an oversized block, and the level is the same block mean the
    whole volume yields."""
    rng = np.random.default_rng(4)
    volume = rng.random((1, 62, 94, 106)).astype(np.float32) * 4000
    store = tmp_path / "slabbed.ome.zarr"
    write_ome_zarr(
        store,
        volume,
        spacing=[1.0, 1.0, 1.0],
        origin=[0.0, 0.0, 0.0],
        chunks=(1, 31, 94, 16),
    )
    append_ome_zarr_levels(store, [4])
    level0, _ = read_ome_zarr_data_slice(store, (slice(None),) * 4, level=0)
    level1, _ = read_ome_zarr_data_slice(store, (slice(None),) * 4, level=1)
    np.testing.assert_array_equal(level0, volume)
    np.testing.assert_allclose(level1, _block_mean(volume, 4), rtol=1e-6)


def test_each_scale_factor_shrinks_the_level_above_it(tmp_path):
    """``[4, 4]`` is the two-coarse-level spelling: each factor shrinks the level above it, so the
    levels are 4x then 16x smaller than level 0 -- the ladder every consumer indexes by position
    (ngff-zarr's own level-0-relative argument is derived from it, never spelled by the user)."""
    volume = np.zeros((1, 64, 64, 64), dtype=np.float32)
    store = tmp_path / "ladder.ome.zarr"
    write_ome_zarr(store, volume, spacing=[1.0, 1.0, 1.0], origin=[0.0, 0.0, 0.0], scale_factors=[4, 4])
    level1, _ = read_ome_zarr_data_slice(store, (slice(None),) * 4, level=1)
    level2, _ = read_ome_zarr_data_slice(store, (slice(None),) * 4, level=2)
    assert level1.shape == (1, 16, 16, 16)
    assert level2.shape == (1, 4, 4, 4)


def test_integer_pyramids_round_like_the_wasm_bin_shrink(tmp_path):
    """astype truncates toward zero; the wasm BinShrink this writer replaced rounds. Truncation
    would shift every uint16 ExaSPIM level ~half an LSB below what the previous release wrote --
    a silent, uniform value change in shipped data."""
    volume = np.zeros((1, 4, 4, 4), dtype=np.uint16)
    volume[0, :2, :2, :2] = [[[1, 0], [0, 0]], [[0, 0], [0, 1]]]  # window mean 2/8 -> rounds to 0
    volume[0, :2, 2:, :2] = 1  # window mean 1 -> 1 exactly
    volume[0, 2:, :2, :2] = [[[3, 0], [0, 0]], [[0, 0], [0, 1]]]  # mean 4/8 = 0.5 -> rounds UP to 1
    store = tmp_path / "ints.ome.zarr"
    write_ome_zarr(store, volume, spacing=[1.0, 1.0, 1.0], origin=[0.0, 0.0, 0.0], scale_factors=[2])
    level1, _ = read_ome_zarr_data_slice(store, (slice(None),) * 4, level=1)
    assert level1.dtype == np.uint16
    assert level1[0, 0, 0, 0] == 0  # 0.25 -> 0
    assert level1[0, 0, 1, 0] == 1  # 1.0 -> 1
    assert level1[0, 1, 0, 0] == 1  # 0.5 -> 1 (round half up, not truncation's 0)


def test_oversized_factor_refuses_with_a_named_axis(tmp_path):
    volume = np.zeros((1, 2, 64, 64), dtype=np.float32)
    store = tmp_path / "thin.ome.zarr"
    with pytest.raises(Exception, match="shrinks axis"):
        write_ome_zarr(store, volume, spacing=[1.0, 1.0, 1.0], origin=[0.0, 0.0, 0.0], scale_factors=[4])


def test_chunk_cache_serves_the_same_bytes_and_survives_eviction(tmp_path):
    """The decoded-chunk cache is a memcpy of what a plain read decodes: windows that straddle chunks,
    single voxels, an axis tail, all bit-identical -- and a cache smaller than one window still
    answers correctly (misses fall through to the store)."""
    from konfai.utils import ome_zarr as OZ

    rng = np.random.default_rng(11)
    volume = rng.integers(0, 60000, (1, 70, 90, 110), dtype=np.uint16)
    store = tmp_path / "chunky.ome.zarr"
    write_ome_zarr(store, volume, spacing=[1.0, 1.0, 1.0], origin=[0.0, 0.0, 0.0], chunks=(1, 32, 32, 32))
    OZ.clear_ome_zarr_cache()
    windows = [
        (slice(None), slice(30, 35), slice(None), slice(None)),
        (slice(None), slice(31, 33), slice(31, 33), slice(31, 33)),
        (slice(0, 1), slice(69, 70), slice(89, 90), slice(109, 110)),
        (slice(None), slice(0, 70), slice(0, 1), slice(100, 110)),
    ]
    for window in windows:
        got, _ = read_ome_zarr_data_slice(store, window, level=0)
        np.testing.assert_array_equal(got, volume[window])
    # A tiny cache: nothing fits, every read decodes -- still the same bytes.
    OZ._CHUNK_CACHE = OZ._DecodedChunkCache(1024)
    for window in windows:
        got, _ = read_ome_zarr_data_slice(store, window, level=0)
        np.testing.assert_array_equal(got, volume[window])
    OZ._CHUNK_CACHE = None
    OZ.clear_ome_zarr_cache()
