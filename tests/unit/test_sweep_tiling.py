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

"""A sweep region is a BLOCK, and its shape decides how much of the source it pulls.

A region pulls the bounding box of its own image under the chain's maps, so the sweep prices two
decompositions of the same resident bytes, the slab and the cube, and takes the cheaper. Pinned
here: the pricing, the cover, the store chunking a block write asks for, and the claim the
decomposition may not change, that the values are the undecomposed pass's.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from konfai.data import patching as patching_module
from konfai.data.materialize import CaseMaterializer, Verdict
from konfai.data.patching import (
    _SWEEP_ELEMENT_BYTES,
    DatasetManager,
    DatasetPatch,
    _chunk_hull_voxels,
    _cubic_tile,
    _pull_block_voxels,
    _sweep_pipeline_depth,
    _sweep_resident_regions,
    _sweep_targets,
)
from konfai.data.transform import OneHot, Resample, Save
from konfai.utils.dataset import Attribute, Dataset, _store_chunks
from konfai.utils.errors import DatasetManagerError

pytest.importorskip("SimpleITK")

SPACING = (1.0, 1.0, 1.0)  # (x, y, z) SimpleITK order


def _attributes(direction: np.ndarray | None = None) -> Attribute:
    attribute = Attribute()
    attribute["Origin"] = np.asarray([0.0, 0.0, 0.0])
    attribute["Spacing"] = np.asarray(list(SPACING))
    attribute["Direction"] = (np.eye(3) if direction is None else direction).astype(np.float64).reshape(-1)
    return attribute


#: The landing, and the height rule these tests pin, chosen so the two decompositions are far apart:
#: the slab reads 2.40x the source where the cube reads 0.91x (printed by _pull_block_voxels below).
LANDING = (48, 128, 128)
ROWS = 12


def _rotation(degrees: float) -> np.ndarray:
    """A rotation mixing x and z: the shear a slab pays for, its z pull growing with the x extent.
    A rotation about z alone shears nothing along the axis a slab advances on."""
    angle = np.deg2rad(degrees)
    return np.asarray([[np.cos(angle), 0.0, np.sin(angle)], [0.0, 1.0, 0.0], [-np.sin(angle), 0.0, np.cos(angle)]])


def _sheared_fixture(tmp_path: Path) -> tuple[Dataset, np.ndarray]:
    """A case and a reference grid of the same extent, rotated 20 degrees in the x-z plane."""
    rng = np.random.default_rng(0)
    volume = (rng.random((1, *LANDING)) * 100).astype(np.float32)
    source = Dataset(tmp_path / "src", "h5")
    source.write("CT", "CASE_000", volume, _attributes())
    reference = Dataset(tmp_path / "ref", "h5")
    reference.write("GRID", "TARGET", np.zeros((1, *LANDING), np.float32), _attributes(_rotation(20.0)))
    return source, volume


def _sweep_plans(manager: DatasetManager):
    """The plans of the segment a pending ``Save`` sweeps: the ones ``_sweep`` itself prices."""
    stream = manager._resolve_patch_stream_source(0, False)
    assert stream is not None and stream.pending_sweeps
    source, _evolved, refusal = manager._replan_sweep(stream.pending_sweeps[0])
    assert source is not None, refusal
    return source.stage_plans


def _manager(source: Dataset, transforms: list) -> DatasetManager:
    return DatasetManager(
        index=0,
        group_src="CT",
        group_dest="CT",
        name="CASE_000",
        dataset=source,
        patch=DatasetPatch([4, 5, 4]),
        transforms=transforms,
        data_augmentations_list=[],
    )


# ---------------------------------------------------------------- the block itself


@pytest.mark.parametrize(
    "spatial,voxels",
    [
        ([513, 1331, 1776], 105 * 1331 * 1776),
        ([300, 300, 300], 64 * 300 * 300),
        ([1, 4096, 4096], 64 * 4096 * 4096),
        ([10, 10, 10], 4 * 10 * 10),
        ([7, 11, 13], 1),
    ],
)
def test_a_cubic_tile_stays_inside_the_landing_and_inside_its_voxel_budget(spatial, voxels) -> None:
    tile = _cubic_tile(spatial, voxels, 128)
    assert all(1 <= extent <= spatial[axis] for axis, extent in enumerate(tile))
    assert int(np.prod(tile, dtype=np.int64)) <= max(voxels, 1) or tile == [1] * len(spatial)


def test_a_cubic_tile_takes_a_short_axis_whole_and_spends_the_rest_on_the_others() -> None:
    """The budget freed by an axis the landing is already thinner than is not thrown away."""
    tile = _cubic_tile([16, 1024, 1024], 16 * 512 * 512, 128)
    assert tile == [16, 512, 512], "a 16-row landing takes 16, and the freed budget squares the rest"


def test_sweep_targets_cover_the_landing_exactly_once() -> None:
    spatial = [7, 9, 5]
    seen = np.zeros(spatial, dtype=np.int32)
    for target in _sweep_targets(spatial, [3, 4, 2]):
        seen[target] += 1
    assert np.array_equal(seen, np.ones(spatial, dtype=np.int32))


def test_sweep_targets_advance_on_the_innermost_axis_first() -> None:
    """The order is the source's: consecutive blocks differ on the axis stored contiguously."""
    targets = list(_sweep_targets([4, 4, 4], [2, 2, 2]))
    assert [t[2].start for t in targets[:2]] == [0, 2]
    assert [t[0].start for t in targets[:2]] == [0, 0]


# ---------------------------------------------------------------- which block the sweep picks


def test_a_chain_with_nothing_to_price_keeps_the_slab(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No plans, no pull maps, no reason to change what the height rule already decided."""
    monkeypatch.setattr(patching_module, "SWEEP_SLAB_ROWS", ROWS)
    source, _volume = _sheared_fixture(tmp_path)
    manager = _manager(source, [])
    assert manager._sweep_tile(list(LANDING), 1) == [ROWS, *LANDING[1:]]


def test_a_sheared_resample_takes_the_cube_and_reads_less_for_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(patching_module, "SWEEP_SLAB_ROWS", ROWS)
    source, _volume = _sheared_fixture(tmp_path)
    resample = Resample(reference="TARGET", reference_group="GRID", reference_dataset=f"{tmp_path / 'ref'}:h5")
    manager = _manager(source, [resample, Save(f"{tmp_path / 'out'}:h5")])
    plans = _sweep_plans(manager)

    spatial = list(LANDING)
    slab = [ROWS, *LANDING[1:]]
    tile = manager._sweep_tile(spatial, 1, plans)

    assert tile != slab, "a 20-degree x-z shear makes the full-plane slab the expensive decomposition"
    assert sum(_pull_block_voxels(spatial, tile, plans)) < sum(_pull_block_voxels(spatial, slab, plans))


def test_an_axis_aligned_chain_prices_the_two_the_same_and_keeps_the_slab(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A resample onto an UNROTATED grid pulls a box per axis: the cube buys nothing, so the sweep
    stays on the shape it has always used."""
    monkeypatch.setattr(patching_module, "SWEEP_SLAB_ROWS", ROWS)
    rng = np.random.default_rng(0)
    source = Dataset(tmp_path / "src", "h5")
    source.write("CT", "CASE_000", (rng.random((1, *LANDING)) * 100).astype(np.float32), _attributes())
    reference = Dataset(tmp_path / "ref", "h5")
    reference.write("GRID", "TARGET", np.zeros((1, *LANDING), np.float32), _attributes())
    resample = Resample(reference="TARGET", reference_group="GRID", reference_dataset=f"{tmp_path / 'ref'}:h5")
    manager = _manager(source, [resample, Save(f"{tmp_path / 'out'}:h5")])

    assert manager._sweep_tile(list(LANDING), 1, _sweep_plans(manager)) == [ROWS, *LANDING[1:]]


# ---------------------------------------------------------------- what the budget buys


def _priced(manager: DatasetManager, plans, rows: int) -> int:
    """What a decomposition of ``rows`` rows holds, priced as the sizing prices it."""
    tile = manager._sweep_shape(list(LANDING), plans, rows)
    return manager.sweep_block_bytes(list(LANDING), 1, plans, tile, _sweep_pipeline_depth())


def _block_voxels(manager: DatasetManager, plans) -> int:
    """The landed voxels of the block the sizing picks under the budget the manager carries."""
    return int(np.prod(manager._sweep_tile(list(LANDING), 1, plans), dtype=np.int64))


def test_a_chain_that_widens_the_channel_axis_is_priced_on_what_it_lands(tmp_path: Path) -> None:
    """``OneHot`` lands one block per class. The source is still pulled at one channel, so the price
    grows by the classes on the landed term alone, not by a flat multiple of the whole."""
    source, _volume = _sheared_fixture(tmp_path)
    classes = 4
    plain = _manager(source, [Save(f"{tmp_path / 'out'}:h5")])
    widened = _manager(source, [OneHot(classes), Save(f"{tmp_path / 'out'}:h5")])
    tile = plain._sweep_shape(list(LANDING), (), 5)
    depth = _sweep_pipeline_depth()

    _pulled, landed = _sweep_resident_regions(depth)
    block = int(np.prod(tile, dtype=np.int64))
    before = plain.sweep_block_bytes(list(LANDING), 1, (), tile, depth)
    after = widened.sweep_block_bytes(list(LANDING), 1, (), tile, depth)

    # Two terms, and only two: the landed blocks widen by the classes, and OneHot's own declared
    # buffers enter a chain that had none (Save declares nothing). The pulled regions do not widen.
    widening = landed * block * (classes - 1) * _SWEEP_ELEMENT_BYTES
    own = OneHot(classes).working_multiple * block * _SWEEP_ELEMENT_BYTES
    assert (after - before) == widening + own
    assert after < before * classes  # the pulled regions did not widen with it


def test_the_sizing_takes_the_tallest_region_the_budget_holds(tmp_path: Path) -> None:
    """The budget is spent, not halved and spent: what the chosen block holds fits, and one row more
    does not."""
    source, _volume = _sheared_fixture(tmp_path)
    manager = _manager(source, [Save(f"{tmp_path / 'out'}:h5")])
    budget = _priced(manager, (), 5)
    manager.set_memory_budget(float(budget))

    rows = manager._sweep_tile(list(LANDING), 1)[0]
    assert _priced(manager, (), rows) <= budget < _priced(manager, (), rows + 1)


def test_a_regrid_pays_for_what_it_pulls_and_not_for_what_it_lands(tmp_path: Path) -> None:
    """The bytes a region costs are the source's box under the chain's maps, so the same budget buys
    a resample onto a sheared grid a smaller block than it buys a chain that reads where it lands."""
    source, _volume = _sheared_fixture(tmp_path)
    resample = Resample(reference="TARGET", reference_group="GRID", reference_dataset=f"{tmp_path / 'ref'}:h5")
    regrid = _manager(source, [resample, Save(f"{tmp_path / 'out'}:h5")])
    pointwise = _manager(source, [Save(f"{tmp_path / 'flat'}:h5")])
    plans = _sweep_plans(regrid)
    budget = float(_priced(pointwise, (), 8))
    regrid.set_memory_budget(budget)
    pointwise.set_memory_budget(budget)

    assert _priced(regrid, plans, 8) > _priced(pointwise, (), 8), "the shear pulls more than it lands"
    assert _block_voxels(regrid, plans) < _block_voxels(pointwise, ())


def test_a_budget_no_region_fits_refuses_with_both_figures(tmp_path: Path) -> None:
    """A budget one row of the landing does not fit is not a one-row sweep: it is a refusal naming
    the budget and what the smallest region holds, so the reader knows what to raise it to."""
    source, _volume = _sheared_fixture(tmp_path)
    manager = _manager(source, [Save(f"{tmp_path / 'out'}:h5")])
    manager.set_memory_budget(_priced(manager, (), 1) / 2.0)

    with pytest.raises(DatasetManagerError, match=r"no region of 'CT' fits the per-rank memory budget"):
        manager._sweep_tile(list(LANDING), 1)
    assert manager.stream_refusal(0) is not None, "and the plan routes the case away from streaming"


# ---------------------------------------------------------------- the values are still the values


def test_a_tiled_sweep_lands_the_undecomposed_answer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The claim the decomposition is allowed to change nothing about."""
    monkeypatch.setattr(patching_module, "SWEEP_SLAB_ROWS", ROWS)
    source, volume = _sheared_fixture(tmp_path)
    resample = Resample(reference="TARGET", reference_group="GRID", reference_dataset=f"{tmp_path / 'ref'}:h5")
    reference = resample("CASE_000", torch.from_numpy(volume), _attributes()).numpy()

    manager = _manager(source, [resample, Save(f"{tmp_path / 'out'}:h5")])
    tile = manager._sweep_tile(list(LANDING), 1, _sweep_plans(manager))
    assert tile != [ROWS, *LANDING[1:]] and tile[1] < LANDING[1], "the case must actually tile"
    assert CaseMaterializer(manager).materialize() is Verdict.STREAM

    streamed, _ = Dataset(tmp_path / "out", "h5").read_data("CT", "CASE_000")
    np.testing.assert_allclose(streamed, reference, rtol=1e-5, atol=1e-4)


# ---------------------------------------------------------------- what a block write asks of the store


def test_a_region_that_already_fits_is_taken_as_it_stands() -> None:
    assert _store_chunks([1, 300, 300, 300], [1, 64, 300, 300], np.float32) == (1, 64, 300, 300)


def test_an_oversized_chunk_is_cut_on_every_axis_at_once() -> None:
    """A chunk long on one axis is the shape measured slowest to write."""
    assert _store_chunks([1, 513, 1331, 1776], [1, 513, 640, 640], np.uint16) == (1, 128, 128, 128)


def test_a_slab_writer_still_gets_the_chunking_it_always_got() -> None:
    assert _store_chunks([1, 513, 1331, 1776], [1, 105, 1331, 1776], np.uint16) == (1, 105, 128, 128)


def test_a_partial_axis_is_only_ever_cut_into_a_divisor_of_the_region() -> None:
    """Anything else splits a region write across chunks: a read-modify-write on every block."""
    chunks = _store_chunks([1, 513, 1331, 1776], [1, 513, 320, 320], np.uint16)
    assert chunks is not None
    assert 320 % chunks[2] == 0 and 320 % chunks[3] == 0


def test_an_axis_whose_only_small_divisor_is_a_sliver_is_left_alone() -> None:
    """A chunk one voxel wide costs a reader the store; an oversized one costs it some bytes."""
    chunks = _store_chunks([1, 513, 1331, 1776], [1, 513, 641, 641], np.uint16)
    assert chunks == (1, 128, 641, 641)


def test_the_device_ceiling_prices_what_a_region_pulls_and_holds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A GPU raises the region height as its free memory allows, and free memory is the whole price.

    Counted as one landed plane per resident region, the ceiling ignored the source box a REGRID
    pulls and the buffers the widest stage declares: a chain resampling through a sheared grid was
    given a region the device could not hold, and where no host budget was declared nothing else
    bounded it.
    """
    monkeypatch.setattr(patching_module, "SWEEP_SLAB_ROWS", ROWS)
    source, _volume = _sheared_fixture(tmp_path)
    resample = Resample(reference="TARGET", reference_group="GRID", reference_dataset=f"{tmp_path / 'ref'}:h5")
    manager = _manager(source, [resample, Save(f"{tmp_path / 'out'}:h5")])
    plans = _sweep_plans(manager)
    free = 4 * _priced(manager, plans, 2 * ROWS)  # a quarter of it is what a region of 2 x ROWS holds
    monkeypatch.setattr(manager, "_chain_device", torch.device("cuda"), raising=False)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (free, free))

    tile = manager._sweep_tile(list(LANDING), 1, plans)  # no host budget: the device is the only ceiling
    held = manager.sweep_block_bytes(list(LANDING), 1, plans, tile, _sweep_pipeline_depth())

    assert held <= free * 0.25, "the region the device was given does not fit the device"
    assert held > _priced(manager, plans, ROWS), "and the device still buys a taller region than the host default"


# ---------------------------------------------------------------- what a chunked store really costs


def _chunked(monkeypatch: pytest.MonkeyPatch, block: tuple[int, ...]) -> None:
    """Answer ``block`` for every read granularity, as a chunked store would."""
    monkeypatch.setattr(Dataset, "read_granularity", lambda self, groups, name: (1, *block))


def test_a_window_on_the_store_grid_pays_itself_and_a_straddling_one_pays_both_blocks() -> None:
    """What a chunked read materialises is the block-aligned hull, not the window: the closed form
    the sizing prices with."""
    grid, extent = [128, 128, 128], [512, 512, 512]
    whole = [slice(0, 128), slice(0, 512), slice(0, 512)]
    assert _chunk_hull_voxels(whole, grid, extent) == 128 * 512 * 512

    aligned = [slice(128, 256), slice(0, 512), slice(0, 512)]
    assert _chunk_hull_voxels(aligned, grid, extent) == 128 * 512 * 512, "one block, read once"

    straddling = [slice(100, 228), slice(0, 512), slice(0, 512)]
    assert _chunk_hull_voxels(straddling, grid, extent) == 256 * 512 * 512, "the same rows, two blocks"

    short = [slice(0, 18), slice(0, 512), slice(0, 512)]
    assert _chunk_hull_voxels(short, grid, extent) == _chunk_hull_voxels(whole, grid, extent), (
        "cutting under one stored block buys nothing back: the read decodes the block either way"
    )


def test_an_unchunked_store_costs_what_it_is_asked_for() -> None:
    span = [slice(3, 21), slice(0, 40)]
    assert _chunk_hull_voxels(span, [1, 1], [64, 40]) == 18 * 40


def test_a_region_under_one_stored_block_is_priced_at_the_block_it_decodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A shorter region does not hold less on a chunked store, so the price may not say it does.
    Without this a budget buys rows that were never free, and the run is killed holding what the
    plan promised it would not."""
    source, _volume = _sheared_fixture(tmp_path)
    manager = _manager(source, [Save(f"{tmp_path / 'out'}:h5")])
    ungranular = _priced(manager, (), 4)

    _chunked(monkeypatch, (16, 128, 128))
    manager._read_granularity = patching_module._UNRESOLVED
    assert _priced(manager, (), 4) > ungranular, "the block it decodes is charged"
    # Halving the rows does not halve the price: both cut inside one stored block, and that block is
    # decoded whole either way.
    assert _priced(manager, (), 4) > _priced(manager, (), 8) / 2


def test_the_sizing_lands_on_the_store_grid_when_a_block_fits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Aligned is free at the peak and cheaper on the way, so a budget that holds one stored block
    buys a decomposition that lands on the grid rather than one that straddles it."""
    source, _volume = _sheared_fixture(tmp_path)
    manager = _manager(source, [Save(f"{tmp_path / 'out'}:h5")])
    _chunked(monkeypatch, (16, 128, 128))
    manager._read_granularity = patching_module._UNRESOLVED
    manager.set_memory_budget(float(_priced(manager, (), 20)))

    assert manager._sweep_tile(list(LANDING), 1)[0] % 16 == 0


def test_a_region_read_is_charged_only_for_what_the_block_grid_adds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``region_reads`` answers the EXCESS a block-aligned decode materialises above the window it
    was asked for: zero on a store that serves exactly that, and more the shorter the region, since
    a region under one stored block decodes the block anyway and simply wastes more of it."""
    source, _volume = _sheared_fixture(tmp_path)
    manager = _manager(source, [Save(f"{tmp_path / 'out'}:h5")])
    assert manager.region_reads(6)[0] == 0, "a store with no block grid adds nothing"

    _chunked(monkeypatch, (16, 128, 128))
    manager._read_granularity = patching_module._UNRESOLVED
    assert manager.region_reads(16)[0] < manager.region_reads(4)[0], (
        "a region under one stored block wastes more of the block it decodes, not less"
    )
