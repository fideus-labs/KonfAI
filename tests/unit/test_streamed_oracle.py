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

"""What a sweep writes region by region is what the whole-volume pass writes, over the whole matrix.

One property, three files, one axis each, over the registry in ``oracle_support``:
``test_transform_locality_contract`` reads it patch by patch, ``test_transform_materialize_contract``
writes it through every storage backend, and this one varies what decides how a case is CUT and what
arithmetic runs on it:

* the DECOMPOSITION: the budget cuts the case into one region, a few, and one row each, which is the
  axis a wrong region boundary shows on and the one no fixed slab height exercises;
* the RANK: a 2-D case and a 3-D one, from the same table of configurations;
* the GEOMETRY: seeded extents, anisotropic spacings, oblique and axis-permuting cosines;
* the DTYPE: what a store actually holds, refusals included;

and the two cardinality changes the workflow owns: N cases folded into one (``Reduce``) and one case
expanded into copies (``Expand``).

Bounds. Byte-identical is the contract everywhere except where an interpolation legitimately rounds
differently, and each such case carries the bound the locality contract measured and states its
reason (``oracle_support.REGRID_ATOL`` for a map that does not factorise, ``LSB_ATOL`` where an
integer store quantizes that difference, ``STAT_ATOL`` for a statistic seeded from the store rather
than recomputed). Nothing here relaxes a bound of its own.

Vacuity. A fallback that quietly writes the whole volume would satisfy "the bytes agree" and prove
nothing, so every row asserts the ROUTE it took as well, and a decomposed row asserts that it really
was decomposed. What cannot take a route says why, in a refusal of its own.

One file because it is one property: the same two routes compared, with the axes as parameters and
the vocabulary they share (the decompositions, the driver, the bounds) declared once at the top.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch
from konfai.data.augmentation import CutOUT, DataAugmentation, Elastix, Noise, Rotate, Scale
from konfai.data.augmentation import Flip as FlipDraw
from konfai.data.case_reduction import CaseReduction
from konfai.data.materialize import CaseMaterializer, Regime, Verdict
from konfai.data.patching import DatasetManager, SweepSegment
from konfai.data.transform import (
    Clip,
    Crop,
    Expand,
    Flip,
    Gradient,
    Mask,
    Reduce,
    Resample,
    Save,
    Transform,
    Write,
)
from konfai.utils.dataset import Attribute, Dataset
from konfai.utils.errors import DatasetManagerError, PatchError, TransformError
from oracle_support import (
    AUGMENTATION_ATOL,
    CASE_NAME,
    FIXED_GEOMETRY,
    LSB_ATOL,
    Geometry,
    StageCase,
    attributes,
    build_case,
    manager,
    seeded_geometry,
    streamable_cases,
    volumes,
)

pytest.importorskip("SimpleITK")

#: The geometries the matrix runs on, drawn from a FIXED seed list rather than per run: a property
#: that fails only on Tuesday's seed is not a property. Extents land in 16..40, which keeps the file
#: inside its time budget while leaving every axis room for several regions.
GEOMETRIES = {
    "rank3-seed11": seeded_geometry(11, 3),
    "rank3-seed23": seeded_geometry(23, 3),
    "rank2-seed37": seeded_geometry(37, 2),
}
#: The one the tests that vary something OTHER than the geometry run on.
MAIN = "rank3-seed11"


@dataclass(frozen=True)
class Route:
    """How the sweep is made to cut the case, as the height one region spans of its first axis."""

    name: str
    #: Rows per region, as a fraction of the case's first extent. ``None`` leaves the sweep its own
    #: cap, which covers these extents whole.
    height: float | None


#: One region, a handful, and one row each: the three decompositions of the same case.
ROUTES = (Route("one-region", None), Route("few-regions", 0.25), Route("row-regions", 0.0))


def _budget_for(manager: DatasetManager, route: Route) -> float | None:
    """The smallest per-rank budget under which the sweep cuts regions of ``route``'s height.

    Found by bisecting the production sizing rule rather than by restating it: the test says how
    tall a region should be, and ``_sweep_tile`` says what budget buys it. Asked with the landing
    and the pull maps the sweep itself will use, because that is what the budget is spent on.
    """
    if route.height is None:
        return None
    # Every copy the run will sweep, each with the landing and the pull maps of its own chain: a
    # draw that samples through an affine pulls more than the shared prefix, and the budget the
    # matrix asks for is the one that buys the height on all of them.
    augmented = manager._expand is not None
    copies = [0] if not augmented else list(range(1, int(manager._expand.nb) + 1))
    segments = {a: manager.sweep_segments(a, augmented) or [] for a in copies}
    rows = max(1, int(route.height * int(manager.shapes[copies[0]][0])))
    low, high = 1.0, float(2**48)
    for _ in range(64):
        middle = (low + high) / 2
        manager.set_memory_budget(middle)
        if min(_sweep_height(manager, sweeps) for sweeps in segments.values()) < rows:
            low = middle
        else:
            high = middle
    manager.set_memory_budget(None)
    return high


def _sweep_height(manager: DatasetManager, segments: Sequence[SweepSegment]) -> int:
    """The shortest region the sizing buys these segments under the budget the manager currently
    carries; zero where one of them does not fit, which is below every height the routes ask for."""
    heights = []
    for segment in segments:
        try:
            heights.append(manager._sweep_tile(segment.landing, segment.channels, segment.plans)[0])
        except DatasetManagerError:
            return 0
    return min(heights, default=0)


# ---------------------------------------------------------------- driving one case both ways


def _manager(dataset: Dataset, group: str, chain: list[Transform], name: str = CASE_NAME) -> DatasetManager:
    return manager(dataset, chain, group=group, name=name)


@dataclass(frozen=True)
class Written:
    """One materialization's result: what landed, how it landed, and in how many pieces."""

    array: np.ndarray
    attribute: Attribute
    verdict: Verdict
    regions: int


def _count_regions(monkeypatch: pytest.MonkeyPatch) -> Callable[[], int]:
    """Count the regions a sweep reads, so a row cannot pass by never having been decomposed."""
    read = DatasetManager._read_streamed_region
    counted = [0]

    def counting(self, *args, **kwargs):
        counted[0] += 1
        return read(self, *args, **kwargs)

    monkeypatch.setattr(DatasetManager, "_read_streamed_region", counting)
    return lambda: counted[0]


def _sweep(
    dataset: Dataset, group: str, stage: Transform, destination: Path, route: Route, monkeypatch: pytest.MonkeyPatch
) -> Written:
    """Write ``stage`` over the case region by region, cut as ``route`` says, and read back what landed."""
    stage.set_datasets([dataset])
    manager = _manager(dataset, group, [stage, Save(f"{destination}:h5")])
    budget = _budget_for(manager, route)
    with monkeypatch.context() as context:
        regions = _count_regions(context)
        verdict = CaseMaterializer(manager).materialize(fallback_budget_bytes=budget)
        array, attribute = Dataset(destination, "h5").read_data(group, CASE_NAME)
        return Written(array, attribute, verdict, regions())


def _whole_volume(dataset: Dataset, group: str, stage: Transform, destination: Path) -> Written:
    """The reference: the same chain over the assembled case, which is what streaming must reproduce."""
    stage.set_datasets([dataset])
    manager = _manager(dataset, group, [stage, Save(f"{destination}:h5")])
    CaseMaterializer(manager)._assemble_and_write(0)
    manager.unload()
    array, attribute = Dataset(destination, "h5").read_data(group, CASE_NAME)
    return Written(array, attribute, Verdict.WHOLE_VOLUME, 0)


def _assert_same(got: Written, want: Written, atol: float, rtol: float = 0.0) -> None:
    """Same voxels within the stated bound, same dtype, same geometry: streaming is invisible."""
    assert got.array.shape == want.array.shape
    assert got.array.dtype == want.array.dtype
    np.testing.assert_allclose(got.array, want.array, rtol=rtol, atol=atol)
    for key in ("Origin", "Spacing", "Direction"):
        np.testing.assert_allclose(got.attribute.get_np_array(key), want.attribute.get_np_array(key), rtol=0, atol=0)


# ---------------------------------------------------------------- the cases the matrix runs


def _matrix() -> list[tuple[str, StageCase, Route]]:
    """Every (geometry, built-in, decomposition) the property is proven on."""
    return [
        (geometry, case, route)
        for geometry in GEOMETRIES
        for case in streamable_cases(GEOMETRIES[geometry])
        for route in ROUTES
    ]


def _identify(entry: tuple[str, StageCase, Route]) -> str:
    geometry, case, route = entry
    return f"{type(case.transform).__name__}-{case.group}-{geometry}-{route.name}"


@pytest.fixture(scope="session")
def cases(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Dataset]:
    """One case on disk per geometry: built once, read by every row that runs on it."""
    root = tmp_path_factory.mktemp("oracle")
    return {name: build_case(root / name, geometry) for name, geometry in GEOMETRIES.items()}


# ---------------------------------------------------------------- what the fixture itself claims


@pytest.mark.parametrize("geometry", [*GEOMETRIES.values(), FIXED_GEOMETRY], ids=[*GEOMETRIES, "fixed"])
def test_a_geometry_carries_what_the_property_leans_on(geometry: Geometry) -> None:
    """The fixture's own claims, since every row of every contract file assumes them.

    Both directions are orthonormal (a stored volume has no other kind) and the permuting one really
    permutes, so reorienting a case stored on it transposes extents and moves the grid the patches
    are cut on. The reference grid starts inside the case and reaches past it on some axis: one
    contained in its case would prove the sampler and never the boundary, which is the half that
    differs between the streamed and the whole-volume routes.
    """
    identity = np.eye(geometry.rank)
    for direction in (geometry.oblique, geometry.permuting):
        np.testing.assert_allclose(direction @ direction.T, identity, rtol=0, atol=1e-12)
    assert not np.array_equal(geometry.permuting, identity)

    def world(extents: tuple[int, ...], spacing: tuple[float, ...]) -> np.ndarray:
        return np.asarray(extents, dtype=np.float64)[::-1] * np.asarray(spacing)

    case = world(geometry.extents, geometry.spacing)
    reference = world(geometry.reference_extents, geometry.reference_spacing)
    start = np.asarray(geometry.reference_origin) - np.asarray(geometry.origin)
    assert (start > 0).all(), "the reference grid starts outside the case"
    assert (start + reference > case).any(), "the reference grid is nested inside the case"


# ---------------------------------------------------------------- the property


@pytest.mark.parametrize("entry", _matrix(), ids=_identify)
def test_a_swept_case_equals_the_whole_volume_case(
    entry: tuple[str, StageCase, Route],
    cases: dict[str, Dataset],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The matrix: one built-in, one geometry, one decomposition, both routes, the same bytes.

    The route is asserted before the values, so a stage that quietly stopped streaming fails here
    rather than passing on a whole-volume comparison with itself.
    """
    geometry, case, route = entry
    dataset = cases[geometry]
    streamed = _sweep(dataset, case.group, case.transform, tmp_path / "streamed", route, monkeypatch)
    whole = _whole_volume(dataset, case.group, case.transform, tmp_path / "whole")

    assert streamed.verdict is Verdict.STREAM
    # One row per region on a case of 16 rows or more is at least two regions: without this the row
    # would pass on a sweep that never decomposed anything.
    assert streamed.regions >= (2 if route.height == 0.0 else 1)
    _assert_same(streamed, whole, case.atol, case.rtol)


# ---------------------------------------------------------------- the dtypes a store holds


def _dtype_cases() -> list[StageCase]:
    """One stage per read-streamable kind, each a remap a store of ANY dtype can carry.

    The dtype axis must vary the dtype and nothing else, so a stage whose configuration only makes
    sense in one numeric range (a clip at fixed Hounsfield bounds, a fill value no unsigned store
    holds) would confound the two.
    """
    return [
        StageCase(Flip("0")),  # ORIENTATION
        StageCase(Mask(path="Labels", value_outside=0)),  # POINTWISE, reading a companion
        StageCase(Gradient()),  # HALO
        StageCase(Resample(spacing=[2.0, 1.0, 3.0]), atol=LSB_ATOL),  # REGRID
        StageCase(Crop(), group="Boxed"),  # CROP
        StageCase(Clip("min", "max")),  # GLOBAL_STAT
    ]


#: What torch has kernels for, of what a store legitimately holds. ``uint16`` is what microscopy
#: writes and torch implements neither comparison nor arithmetic for it: its refusal is pinned
#: below rather than tolerated here, and ``bool`` never reaches a chain because the store refuses to
#: hold it.
DTYPES = (np.uint8, np.int16, np.int32, np.float32, np.float64)


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda dtype: np.dtype(dtype).name)
@pytest.mark.parametrize("case", _dtype_cases(), ids=lambda case: type(case.transform).__name__)
@pytest.mark.parametrize("route", ROUTES, ids=lambda route: route.name)
def test_a_swept_case_equals_the_whole_volume_case_on_every_dtype(
    case: StageCase, dtype: np.dtype, route: Route, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dtype is not a detail of the storage: it decides whether the chain rounds, and where.

    An integer store quantizes an interpolation, so the two routes may land a least significant bit
    apart (``LSB_ATOL``); every other stage here is an exact remap and must be byte-identical.
    """
    dataset = build_case(tmp_path / "case", GEOMETRIES[MAIN], np.dtype(dtype))
    streamed = _sweep(dataset, case.group, case.transform, tmp_path / "streamed", route, monkeypatch)
    whole = _whole_volume(dataset, case.group, case.transform, tmp_path / "whole")

    assert streamed.verdict is Verdict.STREAM
    # The store's dtype survives a remap. Gradient hands back differences, which an integer store
    # cannot hold: it widens those to float32 and leaves a floating dtype as it found it.
    expected = np.dtype(dtype)
    if isinstance(case.transform, Gradient) and not np.issubdtype(expected, np.floating):
        expected = np.dtype(np.float32)
    assert streamed.array.dtype == expected
    _assert_same(streamed, whole, case.atol, case.rtol)


def test_a_dtype_torch_has_no_kernel_for_refuses_on_both_routes(tmp_path: Path) -> None:
    """``uint16`` is a store's dtype, not a chain's: torch implements no comparison for it.

    The sweep gives up on it (a warning naming ``TensorCast``) and the whole-volume fallback then
    raises with the same remedy: a refusal on one route and a result on the other would make the
    decomposition decide whether a case runs at all.
    """
    dataset = build_case(tmp_path / "case", GEOMETRIES[MAIN], np.dtype(np.uint16))
    manager = _manager(dataset, "Intensity", [Clip("min", "max"), Save(f"{tmp_path / 'out'}:h5")])
    with pytest.warns(UserWarning, match="TensorCast"), pytest.raises(TransformError, match="TensorCast"):
        CaseMaterializer(manager).materialize()


def test_a_two_dimensional_stored_map_is_refused_before_any_route_runs(
    cases: dict[str, Dataset], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stored-map codec holds the 3-D rigid, affine and BSpline kinds; a 2-D map has no tag.

    Refused where the map is READ, which is before either route is chosen, and the message names
    the type it found: a map applied on one route and refused on the other would be the worst of
    both. This is why the two resample-through-a-stored-map cases leave the rank-2 matrix.
    """
    stage = Resample(transforms={"transform": True})
    with pytest.raises(TransformError, match="Euler2DTransform"):
        _sweep(cases["rank2-seed37"], "Intensity", stage, tmp_path / "out", ROUTES[0], monkeypatch)


def test_the_store_refuses_a_dtype_it_cannot_hold(tmp_path: Path) -> None:
    """``bool`` is the one dtype in the list no case can be built on: the refusal is the contract."""
    geometry = GEOMETRIES[MAIN]
    with pytest.raises(TypeError, match="bool"):
        Dataset(tmp_path / "case", "mha").write(
            "Intensity",
            CASE_NAME,
            volumes(geometry)["Labels"].astype(bool),
            attributes(geometry, "Intensity"),
        )


# ---------------------------------------------------------------- N cases folded into one


def _cohort(root: Path, geometry: Geometry, count: int) -> tuple[Dataset, list[np.ndarray]]:
    """``count`` cases on ONE grid, which is what a reduction requires of its members."""
    rng = np.random.default_rng(7)
    dataset = Dataset(root, "h5")
    written = []
    for index in range(count):
        volume = (rng.random((1, *geometry.extents)) * 100.0).astype(np.float32)
        dataset.write("CT", f"CASE_{index:03d}", volume, attributes(geometry, "Intensity"))
        written.append(volume)
    return dataset, written


@pytest.mark.parametrize("operator", ["Mean", "Median", "Std", "Vote", "Concat"])
@pytest.mark.parametrize("count", [2, 3, 4, 5])
@pytest.mark.parametrize("slab_rows", [1, 3, 64], ids=["row-regions", "few-regions", "one-region"])
def test_a_streamed_reduction_equals_the_operator_on_the_whole_cohort(
    operator: str, count: int, slab_rows: int, tmp_path: Path
) -> None:
    """A reduction never assembles its members, so its regions are its only route to the answer.

    The reference is the SAME operator applied once to the whole volumes, in the layout both engines
    hand it (``[1, C, *spatial]`` per case): what a region-wise fold must reproduce exactly, whatever
    the count, whatever the region height. ``Median`` changes route at five members and ``Concat``
    changes the channel count, which is why both bounds of the count are run.
    """
    geometry = GEOMETRIES["rank3-seed23"]
    dataset, written = _cohort(tmp_path / "cohort", geometry, count)
    destination = Dataset(tmp_path / "out", "h5")
    reduce = Reduce(operator=operator, output="folded")
    engine = CaseReduction(
        managers=[_manager(dataset, "CT", [], f"CASE_{index:03d}") for index in range(count)],
        reduce=reduce,
        post=[],
        destination=destination,
        group="CT",
        slab_rows=slab_rows,
    )
    assert engine.materialize() is True

    got, _ = destination.read_data("CT", "folded")
    expected = reduce.operator([torch.from_numpy(volume).unsqueeze(0) for volume in written]).squeeze(0).numpy()
    assert got.shape == expected.shape
    np.testing.assert_allclose(got, expected, rtol=0, atol=0)


# ---------------------------------------------------------------- one case expanded into copies


@dataclass(frozen=True)
class Draw:
    """One draw, the regime its copies must take, and how far a copy of it may round.

    A per-voxel draw is exactly its own block, so its copies ride ONE read pass; a draw that reads
    elsewhere than its target block cannot, and sweeps its own. Which one is not a detail: the
    shared pass is the whole point of the regime, and a pass that fails falls back to solo passes
    that write the same bytes, so only the regime says whether the optimisation still happens.

    A draw that resamples reaches its copy through grid_sample on coordinates normalised by the
    region's own extent rather than the volume's, which is the deviation ``AUGMENTATION_ATOL``
    bounds (ulps of the phantom's step; measured here at 1.5e-4 on a 500-wide range, 3e-7 of it).
    The exact remaps and the per-voxel fields are byte-identical at any region count.
    """

    build: Callable[[], DataAugmentation]
    regime: Regime
    atol: float = 0.0


def _draws() -> dict[str, Draw]:
    """One draw per way a copy is read: a per-voxel field, a box, two exact remaps, two pull maps.
    Built per call, because a draw caches the parameters it drew for a case."""
    return {
        "Noise": Draw(lambda: Noise(1.0), Regime.SHARED),
        "CutOUT": Draw(lambda: CutOUT(1.0, 0.5, 0.0), Regime.SHARED),
        "Flip": Draw(lambda: FlipDraw(f_prob=[1.0, 1.0, 1.0]), Regime.SOLO),
        "QuarterRotate": Draw(lambda: Rotate(is_quarter=True), Regime.SOLO),
        "Rotate": Draw(lambda: Rotate(a_min=10.0, a_max=10.0), Regime.SOLO, AUGMENTATION_ATOL),
        "Scale": Draw(lambda: Scale(), Regime.SOLO, AUGMENTATION_ATOL),
    }


def _expanded(dataset: Dataset, augmentation: DataAugmentation, copies: int, destination: Path) -> DatasetManager:
    return _manager(
        dataset,
        "Intensity",
        [
            Clip(-200.0, 300.0),
            Expand(nb=copies, pattern="{name}_c{a:02d}"),
            augmentation,
            Write(f"{destination}:h5"),
        ],
    )


@pytest.mark.parametrize("name", list(_draws()), ids=list(_draws()))
@pytest.mark.parametrize("copies", [2, 3])
@pytest.mark.parametrize("route", ROUTES, ids=lambda route: route.name)
def test_a_streamed_copy_equals_the_whole_volume_copy(name: str, copies: int, route: Route, tmp_path: Path) -> None:
    """Every copy of an ``Expand`` carries its own draw, and the decomposition must not change it.

    Pointwise is not place-independent: a noise field and a cutout box are functions of the voxel's
    position, so a copy's stages must be told where their block sits exactly as the shared prefix's
    are. Without that, the copies agreed with the whole volume on a case that fitted one region and
    diverged over its whole extent on anything larger.

    Rank 3 only: the draws are declared three-dimensional (``Permute`` refuses anything else), so a
    2-D row would exercise that refusal rather than this property.
    """
    dataset = build_case(tmp_path / "case", GEOMETRIES[MAIN])
    draw = _draws()[name]
    augmentation = draw.build()
    augmentation.load(1.0)

    streamed = _expanded(dataset, augmentation, copies, tmp_path / "streamed")
    budget = _budget_for(streamed, route)
    outcomes = CaseMaterializer(streamed).materialize_copies(list(range(1, copies + 1)), fallback_budget_bytes=budget)
    whole = _expanded(dataset, augmentation, copies, tmp_path / "whole")
    for a in range(1, copies + 1):
        CaseMaterializer(whole)._assemble_and_write(a)

    assert set(outcomes.values()) == {(Verdict.STREAM, draw.regime)}
    for a in range(1, copies + 1):
        entry = f"{CASE_NAME}_c{a:02d}"
        got, _ = Dataset(tmp_path / "streamed", "h5").read_data("Intensity", entry)
        want, _ = Dataset(tmp_path / "whole", "h5").read_data("Intensity", entry)
        np.testing.assert_allclose(got, want, rtol=0, atol=draw.atol)


@pytest.mark.parametrize("route", ROUTES, ids=lambda route: route.name)
def test_a_transform_after_the_marker_reads_its_companion_where_the_block_sits(route: Route, tmp_path: Path) -> None:
    """A copy's tail is not only its draw: a pointwise TRANSFORM there reads a second volume.

    ``Mask`` takes its foreground from a companion aligned with the case, so it needs the block's
    place as much as a noise field does, and it is the half of the fix whose failure is not silent:
    handed a block as a whole volume it raises, the shared pass gives up, and the copies fall back
    to a solo pass each that writes exactly the same bytes. Which is why the REGIME is what says
    whether the shared pass still happens.
    """
    dataset = build_case(tmp_path / "case", GEOMETRIES[MAIN])
    draw = _draws()["Noise"].build()
    draw.load(1.0)
    mask = Mask(path="Labels", value_outside=-7)

    def chain(destination: Path) -> list[Transform]:
        return [
            Expand(nb=2, pattern="{name}_c{a:02d}"),
            draw,
            mask,
            Write(f"{destination}:h5"),
        ]

    mask.set_datasets([dataset])
    streamed = _manager(dataset, "Intensity", chain(tmp_path / "streamed"))
    budget = _budget_for(streamed, route)
    outcomes = CaseMaterializer(streamed).materialize_copies([1, 2], fallback_budget_bytes=budget)
    assert set(outcomes.values()) == {(Verdict.STREAM, Regime.SHARED)}

    whole = _manager(dataset, "Intensity", chain(tmp_path / "whole"))
    for a in (1, 2):
        CaseMaterializer(whole)._assemble_and_write(a)
    for a in (1, 2):
        entry = f"{CASE_NAME}_c{a:02d}"
        got, _ = Dataset(tmp_path / "streamed", "h5").read_data("Intensity", entry)
        want, _ = Dataset(tmp_path / "whole", "h5").read_data("Intensity", entry)
        np.testing.assert_array_equal(got, want)
        assert (got == -7).any(), "the mask fell outside the copy: nothing was masked"


def test_a_copy_that_cannot_stream_is_refused_under_a_budget_its_whole_volume_exceeds(tmp_path: Path) -> None:
    """``Elastix`` solves its field over the whole volume, so its copies take the whole-volume path.

    That path is priced, not free: under a budget the assembled case does not fit, the copies must
    be refused with the working set named, and nothing written. Given room, the same copies land."""
    dataset = build_case(tmp_path / "case", GEOMETRIES[MAIN])
    draw = Elastix()
    draw.load(1.0)
    refused = _expanded(dataset, draw, 2, tmp_path / "refused")
    with pytest.raises(PatchError, match="exceeds the per-rank budget"):
        CaseMaterializer(refused).materialize_copies([1, 2], fallback_budget_bytes=1.0)
    assert not (tmp_path / "refused").exists()

    written = _expanded(dataset, draw, 2, tmp_path / "written")
    outcomes = CaseMaterializer(written).materialize_copies([1, 2])
    assert {verdict for verdict, _regime in outcomes.values()} == {Verdict.WHOLE_VOLUME}
    assert Dataset(tmp_path / "written", "h5").is_dataset_exist("Intensity", f"{CASE_NAME}_c01")


def test_the_copies_of_a_case_are_not_the_same_copy(tmp_path: Path) -> None:
    """The property above compares two routes of ONE draw, so it would hold if every copy were the
    identity. The copies must differ from each other and from the source."""
    geometry = GEOMETRIES[MAIN]
    dataset = build_case(tmp_path / "case", geometry)
    augmentation = _draws()["Noise"].build()
    augmentation.load(1.0)
    CaseMaterializer(_expanded(dataset, augmentation, 2, tmp_path / "out")).materialize_copies([1, 2])

    out = Dataset(tmp_path / "out", "h5")
    first, _ = out.read_data("Intensity", f"{CASE_NAME}_c01")
    second, _ = out.read_data("Intensity", f"{CASE_NAME}_c02")
    source, _ = dataset.read_data("Intensity", CASE_NAME)
    assert not np.array_equal(first, second) and not np.array_equal(first, source)


# ---------------------------------------------------------------- the budget is the decomposition


def test_the_budget_is_what_decides_how_many_regions_a_sweep_cuts(
    cases: dict[str, Dataset], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The matrix's third axis, asserted where it can be seen: the same case, the same stage, three
    budgets, three decompositions, and the same bytes out of all three."""
    dataset = cases[MAIN]
    results = [
        _sweep(dataset, "Intensity", Clip(-200.0, 300.0), tmp_path / route.name, route, monkeypatch) for route in ROUTES
    ]
    rows = int(dataset.get_infos("Intensity", CASE_NAME)[0][1])
    counts = [result.regions for result in results]
    assert [result.verdict for result in results] == [Verdict.STREAM] * len(ROUTES)
    assert counts[0] == 1
    assert 1 < counts[1] < counts[2] == rows, f"the budgets gave {counts} regions for {rows} rows"
    for result in results[1:]:
        np.testing.assert_array_equal(result.array, results[0].array)
