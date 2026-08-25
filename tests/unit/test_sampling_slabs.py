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

"""The budget slabs the coordinate walk, and slabbing changes NO value.

Every claim here is ``torch.equal``, not ``allclose``: the walk's contract is bit-identity between
a streamed region and the whole volume, and a slab is one more way of streaming. A slab that
drifted in the last bit would round a continuous index across a voxel boundary and hand a label
map a different label, so tolerance is the one thing these tests must not have.
"""

import numpy as np
import pytest
import torch
from konfai.data.geometry import AffineMap, AffineStage, DisplacementStage, Grid
from konfai.data.sampling import gather, source_index, source_index_rows, walk_rows

RNG = np.random.default_rng(7)


def _grid(shape: tuple[int, ...], oblique: bool = False) -> Grid:
    direction = np.eye(3)
    if oblique:
        theta = 0.2
        direction = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1.0]])
    return Grid(shape, RNG.uniform(-40, 40, 3), RNG.uniform(0.5, 2.5, 3), direction)


def _field(shape: tuple[int, ...], order: int) -> DisplacementStage:
    return DisplacementStage(_grid(shape), RNG.normal(0, 4.0, (3, *shape)), order)


STAGE_SETS = [
    [],
    [_field((7, 9, 8), 1)],
    [AffineStage(AffineMap(np.eye(3) + RNG.normal(0, 0.05, (3, 3)), RNG.normal(0, 5, 3))), _field((6, 6, 6), 3)],
]


@pytest.mark.parametrize("stages", STAGE_SETS)
def test_slabbed_walk_is_the_whole_walk(stages):
    """A budget small enough to force one-row slabs reproduces the unslabbed walk bit for bit."""
    target, source = _grid((21, 13, 17), oblique=True), _grid((15, 19, 11))
    whole = source_index(target, source, stages, torch.device("cpu"))
    assert walk_rows(target, stages, torch.device("cpu"), budget_bytes=50_000) < 21
    slabbed = source_index(target, source, stages, torch.device("cpu"), budget_bytes=50_000)
    assert torch.equal(whole, slabbed)


def test_row_range_keeps_global_indices():
    """``source_index_rows`` over [start, stop) IS those rows of the whole walk: the row indices go
    through the region's own map, never a re-derived sub-grid whose folded origin would move the
    last bit."""
    target, source = _grid((12, 9, 7), oblique=True), _grid((10, 8, 9))
    stages = [_field((5, 6, 4), 1)]
    whole = source_index(target, source, stages, torch.device("cpu"))
    rows = source_index_rows(target, source, stages, torch.device("cpu"), 3, 9)
    assert torch.equal(whole[3:9], rows)


@pytest.mark.parametrize("mode", ["nearest", "linear", "cubic"])
def test_slabbed_gather_is_the_whole_gather(mode):
    """Walking and gathering slab by slab (what ResampleTransform does over a budget) concatenates
    to the single-pass result exactly: the gather's window and starts are the region's own, so a
    slab sums, picks and normalises the very numbers the whole pass does."""
    target, source = _grid((14, 11, 13)), _grid((12, 15, 10))
    stages = [_field((5, 5, 5), 1)]
    payload = torch.tensor(RNG.integers(0, 3000, (2, 12, 15, 10)), dtype=torch.float32)
    if mode == "nearest":
        payload = payload[:1].to(torch.uint8)
    coordinates = source_index(target, source, stages, torch.device("cpu"))
    whole = gather(payload, coordinates, [0, 0, 0], [12, 15, 10], mode, 0.0)
    parts = [
        gather(
            payload,
            source_index_rows(target, source, stages, torch.device("cpu"), start, min(14, start + 4)),
            [0, 0, 0],
            [12, 15, 10],
            mode,
            0.0,
        )
        for start in range(0, 14, 4)
    ]
    assert torch.equal(whole, torch.cat(parts, dim=1))


def test_separable_nearest_keeps_wide_integer_labels():
    """A nearest pick copies voxels in the payload's own dtype on BOTH gather paths: a float32
    detour rounds every label above 2**24, and the separable path (the common, axis-aligned case)
    must agree with the general one on the very volumes it is supposed to serve identically."""
    from konfai.data.sampling import gather_separable, separable_source_index, source_index

    label = 20_000_001
    payload = torch.full((1, 6, 6, 6), label, dtype=torch.int32)
    # One grid for both sides: a self-resample keeps every pick inside, so the assertion sees
    # copied labels, not fill.
    target = source = _grid((6, 6, 6))
    axes = separable_source_index(target, source, [], torch.device("cpu"))
    assert axes is not None
    out_separable = gather_separable(payload, axes, [0, 0, 0], [6, 6, 6], "nearest", 0.0)
    coordinates = source_index(target, source, [], torch.device("cpu"))
    out_general = gather(payload, coordinates, [0, 0, 0], [6, 6, 6], "nearest", 0.0)
    inside_separable = out_separable[out_separable != 0]
    inside_general = out_general[out_general != 0]
    assert inside_separable.numel() and bool((inside_separable == label).all())
    assert inside_general.numel() and bool((inside_general == label).all())


def test_two_dimensional_slabs_hold_the_same_contract():
    """Rank 2 exercises every axis-order convention with one spatial axis fewer; nothing pins it
    elsewhere in the suite, and a 2-D drift would look exactly like a working resample."""
    from konfai.data.sampling import source_index as walk

    target = Grid((15, 11), RNG.uniform(-10, 10, 2), RNG.uniform(0.5, 2.0, 2), np.eye(2))
    source = Grid((12, 9), RNG.uniform(-10, 10, 2), RNG.uniform(0.5, 2.0, 2), np.eye(2))
    field_grid = Grid((5, 4), RNG.uniform(-10, 10, 2), RNG.uniform(2.0, 4.0, 2), np.eye(2))
    stages = [DisplacementStage(field_grid, RNG.normal(0, 3.0, (2, 5, 4)), 1)]
    whole = walk(target, source, stages, torch.device("cpu"))
    rows = source_index_rows(target, source, stages, torch.device("cpu"), 4, 11)
    assert torch.equal(whole[4:11], rows)
    payload = torch.tensor(RNG.random((1, 12, 9)), dtype=torch.float32)
    coordinates = walk(target, source, stages, torch.device("cpu"))
    whole_gather = gather(payload, coordinates, [0, 0], [12, 9], "linear", 0.0)
    parts = [
        gather(
            payload,
            source_index_rows(target, source, stages, torch.device("cpu"), s, min(15, s + 6)),
            [0, 0],
            [12, 9],
            "linear",
            0.0,
        )
        for s in range(0, 15, 6)
    ]
    assert torch.equal(whole_gather, torch.cat(parts, dim=1))


def test_separable_and_general_walks_agree_bitwise():
    """The separable fast path claims bit-identity with the general walk wherever both apply (axis
    aligned, no stages): hold it, per axis, on the very numbers -- the claim carried no test."""
    from konfai.data.sampling import separable_source_index
    from konfai.data.sampling import source_index as walk

    for trial in range(3):
        target = Grid((9, 8, 7), RNG.uniform(-30, 30, 3), RNG.uniform(0.4, 2.5, 3), np.eye(3))
        source = Grid((8, 6, 9), RNG.uniform(-30, 30, 3), RNG.uniform(0.4, 2.5, 3), np.eye(3))
        axes = separable_source_index(target, source, [], torch.device("cpu"))
        assert axes is not None
        coordinates = walk(target, source, [], torch.device("cpu"))
        for array_axis, axis_values in enumerate(axes):
            component = 3 - 1 - array_axis
            picked = coordinates.movedim(-1, 0)[component]
            broadcast_shape = [1, 1, 1]
            broadcast_shape[array_axis] = -1
            assert torch.equal(picked, axis_values.reshape(broadcast_shape).expand_as(picked)), (
                f"axis {array_axis} drifted at trial {trial}"
            )


def test_retired_resample_spellings_name_their_replacement():
    """The 1.8 breaking change promised each removed name points at the new spelling; difflib
    offered 'EulerTransform' for 'ResampleTransform' and nothing for 'Warp'."""
    from konfai.data.transform import TransformLoader
    from konfai.utils.errors import TransformError

    loader = TransformLoader.__new__(TransformLoader)
    for retired in ("ResampleToResolution", "ResampleToShape", "ResampleToReference", "ResampleTransform", "Warp"):
        with pytest.raises(TransformError, match="Resample"):
            loader.get_transform(retired, "")


def test_fast_precision_is_close_and_scoped():
    """The float32 walk is an opt-in CONTEXT: close to the exact walk on world-scale grids, and the
    default outside the context stays the bit-exact float64 -- entering it must never leak."""
    from konfai.data.sampling import _walk_dtype, coordinate_precision

    target = Grid((9, 8, 7), np.array([30000.0, -20000.0, 10000.0]), np.array([30.08, 30.08, 40.0]), np.eye(3))
    source = Grid((8, 6, 9), np.array([29000.0, -21000.0, 9000.0]), np.array([30.08, 30.08, 40.0]), np.eye(3))
    stages = [_field((5, 6, 4), 1)]
    exact = source_index(target, source, stages, torch.device("cpu"))
    assert _walk_dtype() == torch.float64
    with coordinate_precision(torch.float32):
        assert _walk_dtype() == torch.float32
        fast = source_index(target, source, stages, torch.device("cpu"))
    assert _walk_dtype() == torch.float64
    assert fast.dtype == torch.float32
    # ~|world|/2^24 in world units over a ~30 um voxel: a few 1e-3 of a voxel at these magnitudes.
    assert (exact - fast.double()).abs().max().item() < 5e-3
    again = source_index(target, source, stages, torch.device("cpu"))
    assert torch.equal(exact, again)  # the exact walk is untouched by having entered the context


def test_the_default_budget_is_probed_once_per_second_not_per_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """The walk's default budget reads the cgroup, /proc and the device once per TTL, not once per
    patch; a budget handed in is never probed for. Slabbing changes no value, so a stale second
    changes none either."""
    from konfai.data import sampling
    from konfai.utils import budget as budget_module

    probes: list[int] = []
    probe = budget_module.available_memory_bytes
    monkeypatch.setattr(budget_module, "available_memory_bytes", lambda: probes.append(1) or probe())
    clock = [1000.0]
    monkeypatch.setattr(sampling, "_now", lambda: clock[0])
    sampling._default_budgets.clear()
    try:
        grid = _grid((21, 13, 17))
        rows = walk_rows(grid, (), torch.device("cpu"))
        for _ in range(50):
            assert walk_rows(grid, (), torch.device("cpu")) == rows
        assert len(probes) == 1
        clock[0] += sampling._BUDGET_TTL_SECONDS + 0.01
        walk_rows(grid, (), torch.device("cpu"))
        assert len(probes) == 2
        walk_rows(grid, (), torch.device("cpu"), budget_bytes=50_000)
        assert len(probes) == 2
    finally:
        sampling._default_budgets.clear()


def _diagonal_stage(matrix_diagonal, translation) -> AffineStage:
    return AffineStage(AffineMap(np.diag(np.asarray(matrix_diagonal, dtype=np.float64)), np.asarray(translation)))


@pytest.mark.parametrize(
    "stages",
    [
        [_diagonal_stage([1.0, 1.0, 1.0], RNG.normal(0, 5, 3))],
        [_diagonal_stage(RNG.uniform(0.5, 1.5, 3), [0.0, 0.0, 0.0])],
        [
            _diagonal_stage([1.0, 1.0, 1.0], RNG.normal(0, 5, 3)),
            _diagonal_stage(RNG.uniform(0.5, 1.5, 3), [0.0, 0.0, 0.0]),
        ],
        [_diagonal_stage([1.0, -1.0, 1.0], RNG.normal(0, 5, 3)), _diagonal_stage([1.0, 1.0, 1.0], RNG.normal(0, 5, 3))],
    ],
    ids=["translation", "scale", "translation-then-scale", "flip-then-translation"],
)
def test_a_diagonal_stored_map_walks_bitwise_what_the_general_path_walks(stages) -> None:
    """A translation, an axis-aligned scale and their composition fold to an exactly diagonal map,
    and the per-axis coordinates are the general walk's own numbers: the fold is the walk's
    ``pending``, applied per component with the zero columns left out."""
    from konfai.data.sampling import separable_source_index
    from konfai.data.sampling import source_index as walk

    for trial in range(3):
        directions = np.diag(RNG.choice([-1.0, 1.0], 3)), np.eye(3)
        target = Grid((9, 8, 7), RNG.uniform(-30, 30, 3), RNG.uniform(0.4, 2.5, 3), directions[trial % 2])
        source = Grid((8, 6, 9), RNG.uniform(-30, 30, 3), RNG.uniform(0.4, 2.5, 3), directions[(trial + 1) % 2])
        axes = separable_source_index(target, source, stages, torch.device("cpu"))
        assert axes is not None
        coordinates = walk(target, source, stages, torch.device("cpu"))
        for array_axis, axis_values in enumerate(axes):
            component = 3 - 1 - array_axis
            picked = coordinates.movedim(-1, 0)[component]
            broadcast_shape = [1, 1, 1]
            broadcast_shape[array_axis] = -1
            assert torch.equal(picked, axis_values.reshape(broadcast_shape).expand_as(picked)), (
                f"axis {array_axis} drifted at trial {trial}"
            )


def test_a_map_that_does_not_factorise_is_refused() -> None:
    """A rotation, and any displacement stage, take the general walk: the separable form has no
    term for them, and a tolerance would admit a map whose two forms disagree in the last bit."""
    from konfai.data.sampling import separable_source_index

    target, source = _grid((9, 8, 7)), _grid((8, 6, 9))
    theta = 0.3
    rotation = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1.0]])
    assert (
        separable_source_index(target, source, [AffineStage(AffineMap(rotation, np.zeros(3)))], torch.device("cpu"))
        is None
    )
    assert separable_source_index(target, source, [_field((5, 5, 5), 1)], torch.device("cpu")) is None
    quiet = DisplacementStage(_grid((5, 5, 5)), np.zeros((3, 5, 5, 5)), 1)
    assert separable_source_index(target, source, [quiet], torch.device("cpu")) is None
