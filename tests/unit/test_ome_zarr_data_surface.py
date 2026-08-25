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

"""The public data surface: the OME-Zarr readers and the pyramid writer, as a capsule uses them.

These cover what a hand-rolled store reader gets wrong, which is never the happy path: a big-endian
source, a pyramid level that does not exist, and a downsampling default that changes the pixels."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("zarr")
pytest.importorskip("ngff_zarr")

from konfai.data import (
    append_ome_zarr_levels,
    create_ome_zarr_store,
    get_ome_zarr_info,
    read_ome_zarr_data_slice,
    write_ome_zarr,
)
from konfai.utils.dataset import Attribute, Dataset, _store_chunks
from konfai.utils.errors import DatasetManagerError


def _volume(dtype: str = "<f4") -> np.ndarray:
    rng = np.random.default_rng(0)
    return (rng.random((1, 8, 12, 16)) * 1000).astype(dtype)


def _big_endian_store(path: Path, volume: np.ndarray) -> Path:
    """A store that genuinely holds big-endian samples, as a light-sheet source does.

    Written through create_ome_zarr_store, because that is the one path that keeps the dtype it is
    given: write_ome_zarr would hand the array to ngff-zarr, which normalises on the way in and so
    could never exercise the reader.
    """
    array = create_ome_zarr_store(path, volume.shape, ">f4", spacing=[2.0, 1.0, 1.0])
    array[:] = volume.astype(">f4")
    return path


def test_the_public_surface_is_importable_from_konfai_data() -> None:
    """The locality vocabulary is the framework's distinctive feature; a user who cannot import it
    cannot make their transform stream at all."""
    import konfai.data as kd

    assert kd.PatchLocality(kd.LocalityKind.HALO, halo=(2,)).halo == (2,)
    for name in ("DatasetManager", "Transform", "Write", "Dataset", "LocalityKind", "PatchLocality"):
        assert name in kd.__all__ and hasattr(kd, name), name


def test_reader_returns_native_byte_order(tmp_path: Path) -> None:
    """Light-sheet sources are big-endian. A non-native array is refused outright by
    torch.from_numpy, and silently reinterpreted by anything that touches the raw buffer."""
    volume = _volume()
    store = _big_endian_store(tmp_path / "big_endian.ome.zarr", volume)

    patch, _ = read_ome_zarr_data_slice(store, (slice(None), slice(0, 4), slice(None), slice(None)))
    assert patch.dtype.isnative
    import torch

    torch.from_numpy(patch)  # the loud half of the bug: this raises on a non-native array
    np.testing.assert_array_equal(patch, volume[:, :4])


def test_a_big_endian_store_is_converted_by_the_copy_that_assembles_the_window(tmp_path: Path) -> None:
    """The window is assembled into a native buffer, so no pass of its own walks every byte again."""
    from konfai.utils.ome_zarr import _native_byteorder

    store = _big_endian_store(tmp_path / "big_endian.ome.zarr", _volume())
    patch, _ = read_ome_zarr_data_slice(store, (slice(None), slice(0, 4), slice(None), slice(None)))
    assert patch.dtype.isnative
    assert _native_byteorder(patch) is patch, "a second conversion would copy the whole window"


def test_the_konfai_sidecar_is_read_once_and_not_once_per_region(tmp_path: Path) -> None:
    """It is metadata, and a streamed run asks for it once per region: re-opening the store each
    time is a read it never needed."""
    from konfai.utils.ome_zarr import _konfai_attributes, clear_ome_zarr_cache

    store = _big_endian_store(tmp_path / "big_endian.ome.zarr", _volume())
    clear_ome_zarr_cache()
    for row in range(4):
        read_ome_zarr_data_slice(store, (slice(None), slice(row, row + 1), slice(None), slice(None)))
    assert _konfai_attributes.cache_info() == (3, 1, 8, 1), "one read of the sidecar, three regions"


def test_info_publishes_the_shape_the_reader_indexes(tmp_path: Path) -> None:
    """`shape` is the store's own axis order and `canonical_shape` is the reader's; sizing slices
    from the wrong one reads a transposed region with the right rank and no error."""
    store = tmp_path / "v.ome.zarr"
    write_ome_zarr(store, _volume(), spacing=[2.0, 1.0, 1.0])
    info = get_ome_zarr_info(store)

    assert info["canonical_shape"] == [1, 8, 12, 16]
    patch, _ = read_ome_zarr_data_slice(store, tuple(slice(0, s) for s in info["canonical_shape"]))
    assert list(patch.shape) == info["canonical_shape"]


def test_out_of_range_level_says_so_instead_of_blaming_the_store(tmp_path: Path) -> None:
    store = tmp_path / "pyramid.ome.zarr"
    write_ome_zarr(store, _volume(), spacing=[2.0, 1.0, 1.0], scale_factors=[2])

    with pytest.raises(DatasetManagerError, match="level 5 is out of range"):
        get_ome_zarr_info(store, 5)


def test_write_ome_zarr_builds_a_pyramid_by_position(tmp_path: Path) -> None:
    store = tmp_path / "pyramid.ome.zarr"
    # spacing arrives (x, y, z) SimpleITK-style, so this is x=2 um, y=1, z=1, and the metadata
    # below is read back (c, z, y, x). The two orders meeting here is the whole reason to assert it.
    write_ome_zarr(store, _volume(), spacing=[2.0, 1.0, 1.0], scale_factors=[2])

    assert get_ome_zarr_info(store)["n_levels"] == 2
    fine, coarse = get_ome_zarr_info(store, 0), get_ome_zarr_info(store, 1)
    assert coarse["canonical_shape"] == [1, 4, 6, 8]
    # Level 1 is level 0 halved, and carries its OWN spacing: a consumer indexing by position gets
    # a coarser image, not the same image mislabelled.
    assert [round(s, 6) for s in fine["scale"]] == [1.0, 1.0, 1.0, 2.0]
    assert [round(s, 6) for s in coarse["scale"]] == [1.0, 2.0, 2.0, 4.0]
    # And its origin is shifted by HALF the spacing delta, per axis: these stores use a
    # centre-of-voxel convention, so the coarse first voxel sits at the centre of the block it
    # averages. Reusing the fine origin biases every voxel and still looks like a plausible image.
    assert [round(t, 6) for t in coarse["translation"]] == [0.0, 0.5, 0.5, 1.0]


def test_a_coarser_level_reads_its_own_geometry_not_the_sidecars(tmp_path: Path) -> None:
    """The konfai sidecar records the geometry the writer was handed: level 0's. Read at ``@1``
    through the Dataset, it used to win over the level's own scale and translation, and level 1
    came back with level 0's spacing on an eighth of the voxels (half the samples along each of the
    three axes): half the extent per axis, for every consumer that registers on the coarse level."""
    root = tmp_path / "cases"
    store = root / "case_1" / "CT.ome.zarr"
    store.parent.mkdir(parents=True)
    attributes = Attribute()
    attributes["Spacing"] = np.asarray([2.0, 1.0, 1.0])
    attributes["Origin"] = np.asarray([10.0, 20.0, 30.0])
    attributes["Direction"] = np.asarray([0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    write_ome_zarr(
        store, _volume(), spacing=[2.0, 1.0, 1.0], origin=[10.0, 20.0, 30.0], attributes=attributes, scale_factors=[2]
    )

    _shape, fine = Dataset(str(root), "omezarr@0").get_infos("CT", "case_1")
    _shape, coarse = Dataset(str(root), "omezarr@1").get_infos("CT", "case_1")
    # Level 0 is the level the sidecar describes: read as recorded, Direction included.
    np.testing.assert_array_equal(fine.get_np_array("Spacing"), [2.0, 1.0, 1.0])
    np.testing.assert_array_equal(fine.get_np_array("Origin"), [10.0, 20.0, 30.0])
    # Level 1 has its own scale and translation (x, y, z): the sidecar's Spacing and Origin are
    # not this level's, so they yield to the transforms, while the Direction NGFF cannot express
    # still comes from the sidecar.
    np.testing.assert_allclose(coarse.get_np_array("Spacing"), [4.0, 2.0, 2.0])
    np.testing.assert_allclose(coarse.get_np_array("Origin"), [11.0, 20.5, 30.5])
    direction = [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    np.testing.assert_array_equal(fine.get_np_array("Direction"), direction)
    np.testing.assert_array_equal(coarse.get_np_array("Direction"), direction)
    # One rung per key, whichever level answered: a later write records the geometry once.
    for key in ("Spacing", "Origin", "Direction"):
        assert [k for k in coarse.keys() if k.startswith(key)] == [f"{key}_0"]


def test_default_downsampling_is_a_block_mean_not_a_gaussian(tmp_path: Path) -> None:
    """The default must not change the pixels: BIN_SHRINK is the block mean a capsule writes by
    hand, where ngff-zarr's own ITKWASM_GAUSSIAN default smooths and crushes the peak."""
    volume = _volume()
    store = tmp_path / "shrunk.ome.zarr"
    write_ome_zarr(store, volume, spacing=[1.0, 1.0, 1.0], scale_factors=[2])

    coarse, _ = read_ome_zarr_data_slice(store, (slice(None),) * 4, level=1)
    blocks = volume[:, :8, :12, :16].reshape(1, 4, 2, 6, 2, 8, 2)
    np.testing.assert_allclose(coarse, blocks.mean(axis=(2, 4, 6)), rtol=1e-6)


def test_unknown_downsample_method_names_the_valid_ones(tmp_path: Path) -> None:
    with pytest.raises(DatasetManagerError, match="ITKWASM_BIN_SHRINK"):
        write_ome_zarr(
            tmp_path / "x.ome.zarr", _volume(), spacing=[1.0] * 3, scale_factors=[2], downsample_method="NOPE"
        )


def test_append_levels_turns_a_streamed_store_into_a_pyramid(tmp_path: Path) -> None:
    """The region-written path cannot take scale_factors up front (no level exists until the last
    region lands), so the pyramid is derived afterwards, and level 0 must survive it untouched."""
    volume = _volume()
    store = tmp_path / "streamed.ome.zarr"
    write_ome_zarr(store, volume, spacing=[2.0, 1.0, 1.0], attributes={"Direction": [1, 0, 0, 0, 1, 0, 0, 0, 1]})
    assert get_ome_zarr_info(store)["n_levels"] == 1

    append_ome_zarr_levels(store, [2])

    assert get_ome_zarr_info(store)["n_levels"] == 2
    level0, _ = read_ome_zarr_data_slice(store, (slice(None),) * 4, level=0)
    np.testing.assert_array_equal(level0, volume)
    # The sidecar survives the rewrite: losing Direction is silent, the reader falls back to identity.
    assert get_ome_zarr_info(store)["attributes"]["Direction"] == [1, 0, 0, 0, 1, 0, 0, 0, 1]


def test_append_levels_without_factors_is_a_no_op(tmp_path: Path) -> None:
    store = tmp_path / "v.ome.zarr"
    write_ome_zarr(store, _volume(), spacing=[1.0] * 3)
    append_ome_zarr_levels(store, [])
    assert get_ome_zarr_info(store)["n_levels"] == 1


def test_a_chunk_stays_openable_whatever_plane_the_writer_declares() -> None:
    """A slab sweep declares the whole trailing plane, and that is not a chunk shape.

    At 2048x2048 float32 the declared region is a gigabyte, and at 4096x4096 it is past what zarr
    holds in one buffer. Every axis the region covers end to end can be tiled without splitting a
    region write, so the axis the writer advances along keeps its declared height and the rest are
    cut down until a chunk is a size a reader can decompress to reach one voxel.
    """
    rows = 64
    for spatial in ([400, 512, 512], [400, 1024, 1024], [400, 2048, 2048], [400, 4096, 4096]):
        for channels in (1, 3):
            shape = [channels, *spatial]
            chunk = _store_chunks(shape, [channels, rows, *spatial[1:]], np.float32)
            assert chunk is not None
            megabytes = int(np.prod(chunk, dtype=np.int64)) * 4 / (1 << 20)
            assert megabytes <= 32.0, f"{shape} -> {chunk} is {megabytes:.0f} MiB"
            # The sweep axis is the one the writer advances along: shrinking it below the declared
            # slab is the read-modify-write this sizing exists to avoid.
            assert chunk[1] == rows


def test_a_writer_that_declares_nothing_leaves_the_store_its_own_default() -> None:
    assert _store_chunks([1, 8, 8, 8], None, np.float32) is None


def _attributes() -> Attribute:
    attribute = Attribute()
    attribute["Origin"] = np.asarray([0.0, 0.0, 0.0])
    attribute["Spacing"] = np.asarray([1.0, 1.0, 1.0])
    attribute["Direction"] = np.eye(3, dtype=np.float64).reshape(-1)
    return attribute


# ---------------------------------------------------------------- what a declared read plan buys


def _sequence(cache, identity, steps):
    """Feed ``steps`` to ``cache`` as reads of one-byte chunks; returns the chunks it decoded."""
    decoded = []
    for chunks in steps:
        cache.begin(identity, frozenset(chunks))
        for coords in chunks:
            if cache.get((identity, coords)) is None:
                decoded.append(coords)
                cache.put((identity, coords), np.zeros(1, np.uint8))
    return decoded


def test_a_declared_plan_keeps_what_will_be_read_again() -> None:
    """LRU is the best a cache can do without the future. With it, the chunk to drop is the one
    whose next use is furthest away, which no policy can beat."""
    from konfai.utils.ome_zarr import _DecodedChunkCache

    steps = [{(0,)}, {(1,)}, {(2,)}, {(0,)}]  # (2,) is never wanted again, (0,) is
    identity = ("store", "0")

    lru = _DecodedChunkCache(2)
    assert _sequence(lru, identity, steps) == [(0,), (1,), (2,), (0,)], "LRU drops the one it needs"

    planned = _DecodedChunkCache(2)
    planned.schedule(identity, [frozenset(step) for step in steps])
    assert _sequence(planned, identity, steps) == [(0,), (1,), (2,)], "the plan keeps (0,)"


def test_a_reader_that_deviates_from_its_plan_loses_it_and_nothing_else() -> None:
    from konfai.utils.ome_zarr import _DecodedChunkCache

    cache = _DecodedChunkCache(2)
    identity = ("store", "0")
    cache.schedule(identity, [frozenset({(0,)}), frozenset({(1,)})])
    decoded = _sequence(cache, identity, [{(7,)}, {(8,)}, {(9,)}])  # nothing it declared
    assert decoded == [(7,), (8,), (9,)]


def test_forgetting_one_store_leaves_the_others_alone() -> None:
    """Creating an output store used to drop every input's decoded chunks with it."""
    from konfai.utils.ome_zarr import _DecodedChunkCache

    cache = _DecodedChunkCache(1 << 20)
    for store in ("source", "output"):
        cache.put(((store, "0"), (0,)), np.zeros(8, np.uint8))
    cache.forget("output")
    assert cache.get((("source", "0"), (0,))) is not None
    assert cache.get((("output", "0"), (0,))) is None


def test_a_sweep_declares_the_regions_it_will_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The sweep folds every block's pull map before it reads the first, so it can say what is
    coming; a store that caches decoded blocks is the one thing that can use it."""
    from konfai.data import patching as patching_module
    from konfai.data.materialize import CaseMaterializer
    from konfai.data.patching import DatasetManager, DatasetPatch
    from konfai.data.transform import Clip, Save

    monkeypatch.setattr(patching_module, "SWEEP_SLAB_ROWS", 3)
    declared: list[list] = []
    monkeypatch.setattr(
        Dataset, "plan_region_reads", lambda self, groups, name, windows: declared.append(list(windows))
    )
    rng = np.random.default_rng(0)
    source = Dataset(tmp_path / "src", "mha")
    source.write("CT", "CASE_000", (rng.random((1, 12, 8, 6)) * 100).astype(np.float32), _attributes())
    manager = DatasetManager(
        index=0,
        group_src="CT",
        group_dest="CT",
        name="CASE_000",
        dataset=source,
        patch=DatasetPatch([4, 4, 4]),
        transforms=[Clip(min_value=10.0, max_value=90.0), Save(f"{tmp_path / 'out'}:h5")],
        data_augmentations_list=[],
    )
    CaseMaterializer(manager).materialize()

    assert declared, "the sweep declared nothing"
    windows = declared[0]
    assert len(windows) == len(list(patching_module._sweep_targets([12, 8, 6], [3, 8, 6])))
    assert all(len(window) == 4 for window in windows), "a window covers the channel axis too"


def test_an_entry_written_whole_over_another_is_read_back_as_written(tmp_path: Path) -> None:
    """The reader memoises decoded chunks by path; a whole-volume write that replaces the store
    under that path must forget them, as the streamed write does."""
    dataset = Dataset(tmp_path / "cohort", "omezarr")
    dataset.write("CT", "CASE", np.ones((1, 8, 8, 8), np.float32), _attributes())
    assert float(dataset.read_data("CT", "CASE")[0].mean()) == 1.0
    dataset.write("CT", "CASE", np.full((1, 8, 8, 8), 2.0, np.float32), _attributes())
    assert float(dataset.read_data("CT", "CASE")[0].mean()) == 2.0


def test_a_plan_read_to_its_end_is_retired() -> None:
    """A cohort declares one plan per case source; one that outlived its sweep would rank every
    later store's chunks as never wanted again."""
    from konfai.utils.ome_zarr import _DecodedChunkCache

    cache = _DecodedChunkCache(4)
    identity = ("store", "0")
    steps = [{(0,)}, {(1,)}]
    cache.schedule(identity, [frozenset(step) for step in steps])
    _sequence(cache, identity, steps)
    assert not cache._schedules
