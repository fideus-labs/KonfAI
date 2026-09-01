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

import contextlib
from collections.abc import Iterator
from pathlib import Path, PureWindowsPath

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
from konfai.data.transform import Transform
from konfai.utils.dataset import Attribute, Dataset, _store_chunks
from konfai.utils.errors import DatasetManagerError
from oracle_support import geometry


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


def test_a_stepped_selection_falls_back_instead_of_returning_a_wrong_window(tmp_path: Path) -> None:
    """The chunked reader counts the voxels between start and stop. Handed a step it would size the
    window wrongly and fill it from the wrong places, so it refuses and the lazy array answers."""
    from konfai.utils.ome_zarr import _normalized_selection

    volume = _volume()
    store = tmp_path / "stepped.ome.zarr"
    array = create_ome_zarr_store(store, volume.shape, "<f4", spacing=[2.0, 1.0, 1.0])
    array[:] = volume

    with pytest.raises(DatasetManagerError):
        _normalized_selection((slice(0, 8, 2),), (8,))

    patch, _ = read_ome_zarr_data_slice(store, (slice(None), slice(0, 8, 2), slice(None), slice(None)))
    np.testing.assert_array_equal(patch, volume[:, 0:8:2])


def test_the_konfai_sidecar_is_read_once_and_not_once_per_region(tmp_path: Path) -> None:
    """It is metadata, and a streamed run asks for it once per region: re-opening the store each
    time is a read it never needed."""
    from konfai.utils.ome_zarr import _multiscales, clear_ome_zarr_cache

    store = _big_endian_store(tmp_path / "big_endian.ome.zarr", _volume())
    clear_ome_zarr_cache()
    for row in range(4):
        read_ome_zarr_data_slice(store, (slice(None), slice(row, row + 1), slice(None), slice(None)))
    assert _multiscales.cache_info().misses == 1, "one parse of the store, however many regions"


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
    return geometry()


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
        cache.end(identity)
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

    monkeypatch.setattr("konfai.data.patching.budget.SWEEP_SLAB_ROWS", 3)
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


def _companion_reads(cache, source, mask, steps, declare: bool) -> dict:
    """``steps`` read from ``source`` (declared when ``declare``), each followed by a read of one
    chunk of ``mask`` that declares nothing; returns the chunks decoded per identity."""
    if declare:
        cache.schedule(source, [frozenset(step) for step in steps])
    decoded: dict = {source: [], mask: []}
    for chunks in steps:
        for identity, wanted in ((source, chunks), (mask, {("m",)})):
            cache.begin(identity, frozenset(wanted))
            for coords in sorted(wanted):
                if cache.get((identity, coords)) is None:
                    decoded[identity].append(coords)
                    cache.put((identity, coords), np.zeros(1, np.uint8))
            cache.end(identity)
    return decoded


def test_a_companion_read_beside_a_declared_source_competes_by_recency() -> None:
    """A sweep declares its source and nothing else; a stage reading a companion volume (a mask)
    reads it at every region, undeclared. Ranked never-again, the companion's chunk is the first to
    go whenever a declared chunk has a use ahead, and is decoded at every region: 3 of 5 here.
    Ranked by recency it competes as under LRU, where a chunk read at every step is never the
    oldest, and the declared source still keeps the chunk with the nearer use."""
    from konfai.utils.ome_zarr import _DecodedChunkCache

    source, mask = ("src", "0"), ("mask", "0")
    steps = [{("a",), ("b",)}, {("c",)}, {("d",)}, {("a",)}, {("b",)}]

    lru = _companion_reads(_DecodedChunkCache(3), source, mask, steps, declare=False)
    assert [c[0] for c in lru[source]] == ["a", "b", "c", "d", "a", "b"], "LRU re-decodes both"
    assert len(lru[mask]) == 1

    planned = _companion_reads(_DecodedChunkCache(3), source, mask, steps, declare=True)
    assert [c[0] for c in planned[source]] == ["a", "b", "c", "d", "b"], "'a' is the nearer use, kept"
    assert len(planned[mask]) == 1, "the companion is decoded once, as under LRU"


def test_a_chunk_of_the_window_being_assembled_outlives_the_rest_of_it() -> None:
    """A read whose window is decoded chunk by chunk must not evict its own first chunk to admit
    its last: the schedule counts a use at the read in progress as now, not as behind, until the
    read ends."""
    from konfai.utils.ome_zarr import _DecodedChunkCache

    cache = _DecodedChunkCache(2)
    source, other = ("src", "0"), ("other", "0")
    cache.put((other, ("m",)), np.zeros(1, np.uint8))
    cache.schedule(source, [frozenset({("a",), ("b",)}), frozenset({("c",)})])
    cache.begin(source, frozenset({("a",), ("b",)}))
    cache.put((source, ("a",)), np.zeros(1, np.uint8))
    cache.put((source, ("b",)), np.zeros(1, np.uint8))
    assert cache.get((source, ("a",))) is not None, "evicted for its own window's last chunk"
    assert cache.get((other, ("m",))) is None, "the undeclared chunk went instead"
    cache.end(source)
    cache.begin(source, frozenset({("c",)}))
    cache.put((source, ("c",)), np.zeros(1, np.uint8))
    assert cache.get((source, ("c",))) is not None, "the read that ended is behind: its chunks go first"


def test_an_empty_declared_read_still_advances_its_schedule(tmp_path: Path) -> None:
    """A region a pull map folds to nothing (a block outside the source) is declared like any other;
    a reader that returned before counting it would rank every later chunk against the wrong read."""
    from konfai.utils.ome_zarr import _chunk_cache, plan_ome_zarr_reads, store_identity

    store = tmp_path / "planned.ome.zarr"
    write_ome_zarr(store, _volume(), spacing=[1.0, 1.0, 1.0], chunks=[1, 4, 12, 16])
    empty = (slice(0, 1), slice(0, 0), slice(0, 12), slice(0, 16))
    first = (slice(0, 1), slice(0, 4), slice(0, 12), slice(0, 16))
    second = (slice(0, 1), slice(4, 8), slice(0, 12), slice(0, 16))
    plan_ome_zarr_reads(store, [empty, first, second])
    identity = next(i for i in _chunk_cache()._schedules if i[0] == store_identity(store))

    assert read_ome_zarr_data_slice(store, empty)[0].shape == (1, 0, 12, 16)
    read_ome_zarr_data_slice(store, first)
    assert identity in _chunk_cache()._schedules, "the empty read was not counted: the schedule deviated"
    read_ome_zarr_data_slice(store, second)
    assert identity not in _chunk_cache()._schedules, "read to its end, retired"


class _CountingArray:
    """A zarr array whose reads add the chunks they decode to ``decoded[group]``."""

    def __init__(self, array, decoded: dict[str, int], group: str) -> None:
        self._array, self._decoded, self._group = array, decoded, group
        self.shape, self.chunks, self.dtype = array.shape, array.chunks, array.dtype

    def __getitem__(self, index):
        touched = 1
        for item, chunk, extent in zip(index, self.chunks, self.shape, strict=True):
            lo, hi, _ = item.indices(extent) if isinstance(item, slice) else (item, item + 1, 1)
            touched *= max(0, -(-hi // chunk) - lo // chunk)
        self._decoded[self._group] += touched
        return self._array[index]


def _two_stores(root: Path) -> None:
    """A CT and a Labels mask on one 48x128x128 grid, both chunked 16x32x32 (48 chunks of 64 KiB
    each), and the 20-degree rotated grid a resample lands them on."""
    landing = (48, 128, 128)
    region = [1, 16, 32, 32]
    rng = np.random.default_rng(0)
    source = Dataset(root / "src", "omezarr")
    for group, volume in (
        ("CT", (rng.random((1, *landing)) * 100).astype(np.float32)),
        ("Labels", (rng.random((1, *landing)) > 0.5).astype(np.float32)),
    ):
        stream = source.open_data_stream(group, "CASE_000", list(volume.shape), volume.dtype, _attributes(), region)
        assert stream is not None
        with stream:
            stream.write_slice(tuple(slice(0, extent) for extent in volume.shape), volume)
    angle = np.deg2rad(20.0)
    rotated = _attributes()
    rotated["Direction"] = np.asarray(
        [[np.cos(angle), 0.0, np.sin(angle)], [0.0, 1.0, 0.0], [-np.sin(angle), 0.0, np.cos(angle)]]
    ).reshape(-1)
    Dataset(root / "ref", "h5").write("GRID", "TARGET", np.zeros((1, *landing), np.float32), rotated)


@contextlib.contextmanager
def _counting_decodes(patched: pytest.MonkeyPatch, capacity_chunks: int) -> Iterator[dict[str, int]]:
    """The chunks each store of :func:`_two_stores` decodes while the block runs, with the
    decoded-chunk cache holding ``capacity_chunks`` of them. The dict fills as the block runs."""
    from konfai.utils import ome_zarr

    decoded = {"CT": 0, "Labels": 0}
    level_array = ome_zarr._level_array
    level_array.cache_clear()

    def counting(store_path: str, level_path: str):
        group = next(group for group in decoded if f"/{group}." in store_path)
        return _CountingArray(level_array(store_path, level_path), decoded, group)

    patched.setenv("OMP_NUM_THREADS", "1")  # one thread: the cache sees the read's own order
    patched.setattr(ome_zarr, "_level_array", counting)
    cache = ome_zarr._chunk_cache()
    previous = cache.capacity
    cache.forget()
    cache.set_capacity(capacity_chunks * 16 * 32 * 32 * 4)
    try:
        yield decoded
    finally:
        cache.set_capacity(previous)
        cache.forget()


def _masked_resample(root: Path) -> tuple[Dataset, list[Transform]]:
    """The store of :func:`_two_stores` and the chain both engines are measured on: a Mask reading
    Labels at every region beside the CT, then a 20-degree rotated resample whose blocks pull each
    chunk of both stores from several regions."""
    from konfai.data.transform import Mask, Resample

    source = Dataset(root / "src", "omezarr")
    mask = Mask(path="Labels", value_outside=-7)
    mask.set_datasets([source])
    resample = Resample(reference="TARGET", reference_group="GRID", reference_dataset=f"{root / 'ref'}:h5")
    return source, [mask, resample]


def _sweep_with_a_mask_companion(
    root: Path, monkeypatch: pytest.MonkeyPatch, *, declared: str, capacity_chunks: int, label: str
) -> dict[str, int]:
    """One sweep of :func:`_masked_resample`; the chunks decoded per store. ``declared`` is what
    tells the cache its future: ``"nothing"`` (LRU), ``"source"`` (the sweep's own reads, the mask
    ranked by recency) or ``"both"`` (the mask's reads too)."""
    from konfai.data.materialize import CaseMaterializer, Verdict
    from konfai.data.patching import DatasetManager, DatasetPatch
    from konfai.data.transform import Mask, Save

    source, chain = _masked_resample(root)
    manager = DatasetManager(
        index=0,
        group_src="CT",
        group_dest="CT",
        name="CASE_000",
        dataset=source,
        patch=DatasetPatch([4, 5, 4]),
        transforms=[*chain, Save(f"{root / f'out_{label}'}:h5")],
        data_augmentations_list=[],
    )
    with monkeypatch.context() as patched, _counting_decodes(patched, capacity_chunks) as decoded:
        patched.setattr("konfai.data.patching.budget.SWEEP_SLAB_ROWS", 12)
        patched.setattr("konfai.data.patching.sweep._sweep_pipeline_depth", lambda: 0)
        if declared == "nothing":
            patched.setattr(Dataset, "plan_region_reads", lambda self, groups, name, windows: None)
        elif declared == "source":
            patched.setattr(Mask, "plan_region_reads", lambda self, name, contexts: None)
        assert CaseMaterializer(manager).materialize() is Verdict.STREAM
    return decoded


def _patch_epoch_with_a_mask_companion(
    root: Path, monkeypatch: pytest.MonkeyPatch, *, declared: bool, shuffle: bool, capacity_chunks: int
) -> dict[str, int]:
    """One pass over a case's patches through :func:`_masked_resample`, read in the loader's own
    order (the grid's, or a shuffled epoch's) on one process; the chunks decoded per store.
    ``declared`` is whether the sampler publishes that order for the reads to declare."""
    import torch
    from konfai.data.data_manager import DatasetIter, Group, GroupTransform, WindowedCaseSampler
    from konfai.data.patching import DatasetManager, DatasetPatch

    source, chain = _masked_resample(root)
    patch = [16, 64, 64]
    manager = DatasetManager(
        index=0,
        group_src="CT",
        group_dest="CT",
        name="CASE_000",
        dataset=source,
        patch=DatasetPatch(patch),
        transforms=chain,
        data_augmentations_list=[],
    )
    mapping = [(0, 0, index) for index in range(manager.get_size(0))]
    dataset_iter = DatasetIter(
        rank=0,
        data={"CT": [manager]},
        mapping=mapping,
        groups_src={"CT": Group(groups_dest={"CT": GroupTransform(transforms=None, patch_transforms=None)})},
        inline_augmentations=False,
        data_augmentations_list=[],
        patch_size=patch,
        overlap=None,
        buffer_size=1,
        use_cache=False,
    )
    sampler = WindowedCaseSampler(mapping, shuffle, None, 1, 1, dataset_iter.read_order if declared else None)
    with monkeypatch.context() as patched, _counting_decodes(patched, capacity_chunks) as decoded:
        torch.manual_seed(7)
        for index in sampler:
            dataset_iter[index]
    return decoded


def test_a_mask_companion_of_a_declared_sweep_is_decoded_once_where_lru_decodes_it_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real two-store sweep: a Mask reads Labels at every region beside the CT the sweep
    declared, through a 20-degree resample whose cubic blocks pull each chunk of both stores from
    several regions. At a cache holding half of the two stores together, the declared source
    decodes each chunk once and so does the companion beside it, where LRU decodes both again.

    Half and not two thirds because the sweep cuts its blocks on the store's own grid: fewer regions
    share a chunk, so there is less for either policy to keep and a cache has to be tighter before
    the order it evicts in decides anything."""
    _two_stores(tmp_path)
    floor = _sweep_with_a_mask_companion(tmp_path, monkeypatch, declared="both", capacity_chunks=96, label="all")
    lru = _sweep_with_a_mask_companion(tmp_path, monkeypatch, declared="nothing", capacity_chunks=48, label="lru")
    planned = _sweep_with_a_mask_companion(tmp_path, monkeypatch, declared="both", capacity_chunks=48, label="plan")

    assert floor["CT"] == floor["Labels"] > 0, "the two stores are read alike"
    assert lru["CT"] > floor["CT"] and lru["Labels"] > floor["Labels"], "LRU re-decodes both"
    assert planned == floor, f"{planned} decodes against a floor of {floor}"


def test_a_store_named_with_a_path_forgets_the_chunks_read_under_its_uri_spelling(tmp_path: Path) -> None:
    """A reader keys decoded chunks by the string it was handed, which `OmeZarrFile._path` builds on
    forward slashes whatever the platform; a writer names the same store with a `Path`, and on
    Windows `str(Path)` is backslashed. Both go through one spelling, or a replaced store keeps
    serving the chunks of the store it replaced (invisible on POSIX, where the two already agree).
    """
    from konfai.utils.ome_zarr import _chunk_cache, clear_ome_zarr_cache, store_identity

    read_as = "C:/data/case/CT.ome.zarr"  # what uri.join gives on either platform
    named_as = PureWindowsPath("C:/data/case/CT.ome.zarr")  # what a writer holds
    assert str(named_as) != read_as, "this test is about the two spellings differing"
    assert store_identity(named_as) == store_identity(read_as)

    cache = _chunk_cache()
    cache.forget()
    cache.put(((read_as, "0"), (0, 0, 0)), np.zeros((2, 2, 2), dtype=np.uint8))
    assert cache.get(((read_as, "0"), (0, 0, 0))) is not None
    clear_ome_zarr_cache(named_as)
    assert cache.get(((read_as, "0"), (0, 0, 0))) is None


@pytest.mark.parametrize(("capacity_chunks", "beats_lru"), [(39, True), (24, False)])
def test_a_declared_companion_shares_a_tight_cache_with_the_source_it_is_read_beside(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capacity_chunks: int, beats_lru: bool
) -> None:
    """Under the 64 chunks that hold both stores' futures, the source alone declared pins its own
    while the companion, ranked by recency, is decoded again where the source is kept for a use
    regions away: 44 + 105 at 39 chunks, 96 + 138 at 24 (LRU: 90 + 90 and 96 + 96). Both
    declared, each is evicted by its own next use: 66 + 66 and 100 + 96. Neither reaches its 44
    with the other's future to keep in a cache under one store: the misses are Belady's over the
    two stores' interleaved reads (58 + 58 at 39 against an oracle's 58 + 54, 84 + 74 at 24
    against 84 + 78), and the decodes above them are the chunk-aligned hull a sparse miss decodes
    again (``_assemble_window``), which at 24, under one region's two hulls, is what puts the
    count four over LRU's while the misses stay under it (158 against 176)."""
    _two_stores(tmp_path)

    def sweep(declared: str) -> dict[str, int]:
        return _sweep_with_a_mask_companion(
            tmp_path, monkeypatch, declared=declared, capacity_chunks=capacity_chunks, label=declared
        )

    lru, alone, both = sweep("nothing"), sweep("source"), sweep("both")
    assert sum(both.values()) < sum(alone.values()), f"{both} against {alone} with the source alone declared"
    assert both["Labels"] < alone["Labels"], "the companion is no longer sacrificed to the source's future"
    assert both["Labels"] <= lru["Labels"], "and decoded no more than under LRU"
    if beats_lru:
        assert sum(both.values()) < sum(lru.values()), f"{both} against LRU's {lru}"


@pytest.mark.parametrize("shuffle", [False, True], ids=["grid-order", "shuffled-epoch"])
@pytest.mark.parametrize("capacity_chunks", [39, 24])
def test_a_declared_patch_epoch_decodes_fewer_chunks_than_recency_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capacity_chunks: int, shuffle: bool
) -> None:
    """The patch route reads a case in a known order nobody declared: the grid's for PREDICTION and
    EVALUATION, the sampler's for a TRAIN epoch. Told it, the cache evicts by the next declared use
    and the mask read beside every patch is ranked against the same future. Decodes over the two
    stores together at 39 and 24 of the 96 chunks they hold, LRU -> declared: 312 -> 212 and
    384 -> 336 in the grid's order, 247 -> 217 and 344 -> 309 over a shuffled epoch."""
    _two_stores(tmp_path)

    def epoch(declared: bool) -> dict[str, int]:
        return _patch_epoch_with_a_mask_companion(
            tmp_path, monkeypatch, declared=declared, shuffle=shuffle, capacity_chunks=capacity_chunks
        )

    lru, planned = epoch(False), epoch(True)

    assert sum(planned.values()) < sum(lru.values()), f"{planned} against LRU's {lru}"


def test_a_store_named_in_any_accepted_spelling_resolves_and_lists(tmp_path: Path) -> None:
    """`is_store_name` accepts five spellings, so all five have to resolve: a root whose entry is
    named with one the reader does not try is accepted at setup and then raises `NameError` on its
    first read, while `get_group` reports no groups at all."""
    from konfai.utils.utils import STORE_FORMS, is_store_name

    for form in STORE_FORMS:
        root = tmp_path / form.strip(".")
        (root / "CASE").mkdir(parents=True)
        create_ome_zarr_store(root / "CASE" / f"CT{form}", (1, 2, 3, 4), "<u1", spacing=[1.0, 1.0, 1.0])
        assert is_store_name(f"CT{form}"), form
        dataset = Dataset(str(root), "omezarr")
        assert dataset.get_group() == ["CT"], form
        assert dataset.is_dataset_exist("CT", "CASE"), form


def test_a_format_token_the_writer_does_not_know_is_refused_by_name(tmp_path: Path) -> None:
    """Every spelling the walk accepts on disk is a token the writer accepts, the dotted one first.
    A token nothing knows is refused here rather than handed to SimpleITK, which wrote 'hd5' for
    'h5' and read it back correctly while ``is_dataset_exist`` said no: a resumed run redid the work.
    """
    from konfai.utils.errors import DatasetManagerError
    from konfai.utils.utils import STORE_FORMS

    for form in STORE_FORMS:
        assert Dataset(str(tmp_path / form.strip(".")), form.removeprefix(".")).file_format == "omezarr", form

    for token in ("hd5", "niigz", "ome zarr", ""):
        with pytest.raises(DatasetManagerError, match="not a format KonfAI writes"):
            Dataset(str(tmp_path / "refused"), token)


def test_a_store_whose_suffix_is_not_lower_case_resolves_and_reads(tmp_path: Path) -> None:
    """The walk matches the suffix case-insensitively; the resolution has to as well.

    `CT.OME.ZARR` is accepted as a store name and listed as the group `CT`, but on a case-sensitive
    filesystem the probed spellings are all lower case: the read of a group the same object just
    reported raised `NameError: OME-Zarr group not found`.
    """
    from konfai.utils.utils import is_store_name

    (tmp_path / "CASE").mkdir()
    create_ome_zarr_store(tmp_path / "CASE" / "CT.OME.ZARR", (1, 2, 3, 4), "<u1", spacing=[1.0, 1.0, 1.0])
    assert is_store_name("CT.OME.ZARR")

    dataset = Dataset(str(tmp_path), "omezarr")

    assert dataset.get_group() == ["CT"]
    assert dataset.is_dataset_exist("CT", "CASE")
    assert dataset.get_infos("CT", "CASE")[0] == [1, 2, 3, 4]
    assert dataset.read_data("CT", "CASE")[0].shape == (1, 2, 3, 4)
