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
    """A zarr array whose reads count the chunks they decode."""

    def __init__(self, array, decoded: list[int]) -> None:
        self._array, self._decoded = array, decoded
        self.shape, self.chunks, self.dtype = array.shape, array.chunks, array.dtype

    def __getitem__(self, index):
        touched = 1
        for item, chunk, extent in zip(index, self.chunks, self.shape, strict=True):
            lo, hi, _ = item.indices(extent) if isinstance(item, slice) else (item, item + 1, 1)
            touched *= max(0, -(-hi // chunk) - lo // chunk)
        self._decoded[0] += touched
        return self._array[index]


def _sweep_with_a_mask_companion(
    root: Path, monkeypatch: pytest.MonkeyPatch, *, declared: bool, capacity_chunks: int, label: str
) -> dict[str, int]:
    """One sweep of CT through a Mask (reading Labels beside it) and a rotated resample, both
    stores chunked alike, the decoded-chunk cache holding ``capacity_chunks``; the chunks decoded
    per store."""
    from konfai.data.materialize import CaseMaterializer, Verdict
    from konfai.data.patching import DatasetManager, DatasetPatch
    from konfai.data.transform import Mask, Resample, Save
    from konfai.utils import ome_zarr

    source = Dataset(root / "src", "omezarr")
    mask = Mask(path="Labels", value_outside=-7)
    mask.set_datasets([source])
    resample = Resample(reference="TARGET", reference_group="GRID", reference_dataset=f"{root / 'ref'}:h5")
    manager = DatasetManager(
        index=0,
        group_src="CT",
        group_dest="CT",
        name="CASE_000",
        dataset=source,
        patch=DatasetPatch([4, 5, 4]),
        transforms=[mask, resample, Save(f"{root / f'out_{label}'}:h5")],
        data_augmentations_list=[],
    )
    decoded: dict[str, list[int]] = {"CT": [0], "Labels": [0]}
    level_array = ome_zarr._level_array
    level_array.cache_clear()

    def counting(store_path: str, level_path: str):
        group = next(group for group in decoded if f"/{group}." in store_path)
        return _CountingArray(level_array(store_path, level_path), decoded[group])

    with monkeypatch.context() as patched:
        patched.setattr(ome_zarr, "_level_array", counting)
        if not declared:
            patched.setattr(Dataset, "plan_region_reads", lambda self, groups, name, windows: None)
        cache = ome_zarr._chunk_cache()
        previous = cache.capacity
        cache.forget()
        cache.set_capacity(capacity_chunks * 16 * 32 * 32 * 4)
        try:
            assert CaseMaterializer(manager).materialize() is Verdict.STREAM
        finally:
            cache.set_capacity(previous)
            cache.forget()
    return {group: count[0] for group, count in decoded.items()}


def test_a_mask_companion_of_a_declared_sweep_is_decoded_once_where_lru_decodes_it_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real two-store sweep: a Mask reads Labels at every region beside the CT the sweep
    declared, through a 20-degree resample whose cubic blocks pull each chunk of both stores from
    several regions. At a cache holding two thirds of the two stores together, the declared source
    decodes each chunk once and so does the companion beside it, where LRU decodes both again."""
    from konfai.data import patching as patching_module

    monkeypatch.setenv("OMP_NUM_THREADS", "1")  # one thread: the cache sees the read's own order
    monkeypatch.setattr(patching_module, "SWEEP_SLAB_ROWS", 12)
    monkeypatch.setattr(patching_module, "_sweep_pipeline_depth", lambda: 0)
    landing = (48, 128, 128)
    region = [1, 16, 32, 32]  # both stores chunked to it: 48 chunks of 64 KiB each
    rng = np.random.default_rng(0)
    source = Dataset(tmp_path / "src", "omezarr")
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
    Dataset(tmp_path / "ref", "h5").write("GRID", "TARGET", np.zeros((1, *landing), np.float32), rotated)

    floor = _sweep_with_a_mask_companion(tmp_path, monkeypatch, declared=True, capacity_chunks=96, label="all")
    lru = _sweep_with_a_mask_companion(tmp_path, monkeypatch, declared=False, capacity_chunks=64, label="lru")
    planned = _sweep_with_a_mask_companion(tmp_path, monkeypatch, declared=True, capacity_chunks=64, label="plan")

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
