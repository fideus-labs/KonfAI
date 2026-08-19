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

"""The cases engine: N cases folded into one entry, one region at a time.

What matters here is not that the median is right, it is that it is right while the volumes are
never assembled, that a cases which does not agree on its grid is refused before anything is read,
and that a chain continues after the reduction."""

import itertools
from pathlib import Path

import numpy as np
import pytest
import torch
from konfai.data.case_reduction import CaseReduction, ReductionPlan, split_chain
from konfai.data.patching import DatasetManager
from konfai.data.reduction import Concat, Mean, Median, Reduction, Vote
from konfai.data.transform import Clip, Dilate, Normalize, Reduce, Save, TensorCast, Transform, resolve_operator
from konfai.utils.dataset import Attribute, Dataset
from konfai.utils.errors import ReductionError, TransformError

CASES = 4  # even on purpose: an even cases is where a lower-median would show


def _attributes(spacing: tuple[float, float, float] = (1.0, 1.0, 2.0)) -> Attribute:
    attribute = Attribute()
    attribute["Origin"] = np.asarray([0.0, 0.0, 0.0])
    attribute["Spacing"] = np.asarray(list(spacing))
    attribute["Direction"] = np.eye(3, dtype=np.float64).reshape(-1)
    return attribute


def _cohort(
    tmp_path: Path, count: int = CASES, spacings: list | None = None, file_format: str = "h5"
) -> tuple[Dataset, np.ndarray]:
    rng = np.random.default_rng(0)
    volumes = (rng.random((count, 1, 8, 10, 6)) * 100).astype(np.float32)
    dataset = Dataset(tmp_path / "cases", file_format)
    for index in range(count):
        spacing = spacings[index] if spacings else (1.0, 1.0, 2.0)
        dataset.write("CT", f"CASE_{index:03d}", volumes[index], _attributes(spacing))
    return dataset, volumes


def _managers(dataset: Dataset, pre: list[Transform], count: int = CASES) -> list[DatasetManager]:
    return [
        DatasetManager(
            index=index,
            group_src="CT",
            group_dest="CT",
            name=f"CASE_{index:03d}",
            dataset=dataset,
            patch=None,
            transforms=list(pre),
            data_augmentations_list=[],
        )
        for index in range(count)
    ]


def _run(tmp_path: Path, pre: list[Transform], reduce: Reduce, post: list[Transform], slab_rows: int = 3, **kwargs):
    dataset, volumes = _cohort(tmp_path, **kwargs)
    destination = Dataset(tmp_path / "out", "h5")
    engine = CaseReduction(
        managers=_managers(dataset, pre, kwargs.get("count", CASES)),
        reduce=reduce,
        post=post,
        destination=destination,
        group="CT",
        slab_rows=slab_rows,
    )
    return engine, destination, volumes


def test_split_chain_finds_the_cardinality_change() -> None:
    reduce = Reduce(output="t")
    pre, found, post = split_chain([Clip(0.0, 50.0), reduce, TensorCast("float32")])
    assert [type(t).__name__ for t in pre] == ["Clip"]
    assert found is reduce
    assert [type(t).__name__ for t in post] == ["TensorCast"]
    plain = [Clip(0.0, 50.0)]
    assert split_chain(plain) == (plain, None, [])


def test_median_of_an_even_cohort_matches_numpy_without_assembling(tmp_path: Path) -> None:
    engine, destination, volumes = _run(tmp_path, [], Reduce(operator="Median", output="template"), [])
    assert engine.materialize() is True

    written, _ = destination.read_data("CT", "template")
    np.testing.assert_allclose(written, np.median(volumes, axis=0), rtol=1e-6)
    # Never assembled: each case streamed, so no manager ever loaded its volume.
    assert all(not manager.loaded and not manager.data for manager in engine.managers)


def test_mean_is_incremental_so_the_cohort_is_never_resident(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine, destination, volumes = _run(tmp_path, [], Reduce(operator="Mean", output="avg"), [])
    plan = engine.plan()
    assert plan.incremental is True and plan.resident_regions == 2

    # The bound, not the declaration: the base protocol BUFFERS, so a Mean that falls back to it
    # holds the whole cohort per region while still declaring itself incremental.
    monkeypatch.setattr(Reduction, "accumulate", lambda self, tensor: pytest.fail("Mean buffered the cohort"))
    engine.materialize()
    written, _ = destination.read_data("CT", "avg")
    np.testing.assert_allclose(written, volumes.mean(axis=0), rtol=1e-6)


@pytest.mark.parametrize("operator", [Mean(), Median()])
def test_averaging_keeps_a_floating_dtype_and_widens_an_integer_one(operator: Reduction) -> None:
    """The predictor is the other caller, and it writes what these hand back.

    A float16 ensemble reduced into float32 doubles every prediction on disk with nothing said; an
    integer cohort rounded back to integers turns the average of 1 and 2 into 1. Same rule, both
    directions: a floating input keeps its dtype, an integer one widens and stays widened.
    """
    half = [torch.ones(1, 1, 2, 2, dtype=torch.float16), torch.full((1, 1, 2, 2), 2.0, dtype=torch.float16)]
    assert operator(half).dtype is torch.float16
    assert operator([half[0]]).dtype is torch.float16

    whole = [torch.ones(1, 1, 2, 2, dtype=torch.int16), torch.full((1, 1, 2, 2), 2, dtype=torch.int16)]
    assert operator(whole).dtype is torch.float32
    assert float(operator(whole).flatten()[0]) == pytest.approx(1.5)


def test_a_non_incremental_operator_holds_the_whole_cohort_per_region(tmp_path: Path) -> None:
    """One region per case, plus the output's, plus what the operator allocates over its buffer.

    ``Median`` holds the cohort and what its route allocates beside it, so the buffer alone
    under-states its peak, and it is the operator a bare ``Reduce`` gets. The route depends on the
    cohort's SIZE (a network up to five members, a sort past it), so the plan asks for this cohort's.
    """
    engine, _destination, _volumes = _run(tmp_path, [], Reduce(operator="Median", output="t"), [])
    plan = engine.plan()
    assert plan.incremental is False
    assert plan.resident_regions == CASES + 1 + Median().working_multiple_for(CASES) * CASES
    assert "resident region" in plan.describe() and "CASE_000" in plan.describe()


def test_an_operator_that_folds_in_place_is_budgeted_for_what_it_holds(tmp_path: Path) -> None:
    """The multiplier is the operator's to declare: ``Mean`` accumulates into one running region."""
    engine, _destination, _volumes = _run(tmp_path, [], Reduce(operator="Mean", output="t"), [])
    plan = engine.plan()
    assert plan.incremental is True
    assert Mean.working_multiple == 0.0
    assert plan.resident_regions == 2


def test_members_on_an_unbounded_store_are_priced_once_per_region(tmp_path: Path) -> None:
    """A gzipped NIfTI decodes the whole volume behind every region asked of it, so the fold reads
    each member once per region (twice that with a statistics pass), and a budget that lowers the
    slab multiplies it. The plan says so, with the remedy; the bytes are the same either way."""
    engine, destination, volumes = _run(
        tmp_path, [], Reduce(operator="Mean", output="avg"), [], slab_rows=3, file_format="nii.gz"
    )
    plan = engine.plan()
    assert plan.regions == 3  # 8 rows in slabs of 3
    assert plan.unbounded == {f"CASE_{index:03d}": "nii.gz" for index in range(CASES)}
    assert plan.read_factor == 3
    described = plan.describe()
    assert "sit on nii.gz" in described and "3 decodes per member (one per region), 12 in all" in described
    assert "put a Save ...:h5 before the Reduce" in described

    engine.slab_rows = 1
    assert engine.plan().read_factor == 8

    engine.post = [Normalize(min_value=0.0, max_value=1.0)]
    seeded = engine.plan()
    assert seeded.stat_pass and seeded.read_factor == 16
    assert "16 decodes per member (one per region and per pass), 64 in all" in seeded.describe()

    engine.post = []
    engine.materialize()
    written, _ = destination.read_data("CT", "avg")
    np.testing.assert_allclose(written, volumes.mean(axis=0), rtol=1e-6)


def test_members_on_a_bounded_store_are_read_once(tmp_path: Path) -> None:
    engine, _destination, _volumes = _run(tmp_path, [], Reduce(operator="Mean", output="avg"), [], slab_rows=3)
    plan = engine.plan()
    assert plan.regions == 3 and plan.unbounded == {} and plan.read_factor == 1
    assert "decodes" not in plan.describe()


def test_a_save_before_the_reduce_moves_the_members_onto_a_bounded_store(tmp_path: Path) -> None:
    """The remedy the plan prints, checked: past the Save the members are read from its cache,
    which is written by a region stream and so serves bounded reads, whether it exists yet or not."""
    engine, destination, volumes = _run(
        tmp_path,
        [Clip(0.0, 50.0), Save(f"{tmp_path / 'cache'}:h5")],
        Reduce(operator="Mean", output="avg"),
        [],
        slab_rows=3,
        file_format="nii.gz",
    )
    assert engine.plan().read_factor == 1 and not engine.plan().unbounded
    engine.materialize()
    assert engine.plan().read_factor == 1  # the cache now exists, and is h5
    written, _ = destination.read_data("CT", "avg")
    np.testing.assert_allclose(written, np.clip(volumes, 0.0, 50.0).mean(axis=0), rtol=1e-6)


def test_per_member_stages_run_on_each_member_separately(tmp_path: Path) -> None:
    """The whole point of cases being cases: a GLOBAL_STAT before the reduction seeds from the
    case's OWN volume, so a cases of different dynamics normalises per specimen."""
    rng = np.random.default_rng(1)
    dataset = Dataset(tmp_path / "cases", "h5")
    volumes = []
    for index, scale in enumerate((1.0, 10.0, 100.0, 1000.0)):
        volume = (rng.random((1, 8, 10, 6)) * scale).astype(np.float32)
        volumes.append(volume)
        dataset.write("CT", f"CASE_{index:03d}", volume, _attributes())
    destination = Dataset(tmp_path / "out", "h5")

    engine = CaseReduction(
        managers=_managers(dataset, [Normalize(min_value=0.0, max_value=1.0)]),
        reduce=Reduce(operator="Median", output="template"),
        post=[],
        destination=destination,
        group="CT",
        slab_rows=3,
    )
    engine.materialize()

    written, _ = destination.read_data("CT", "template")
    expected = np.median(
        np.stack([(v - v.min()) / (v.max() - v.min()) for v in volumes], axis=0).astype(np.float32), axis=0
    )
    np.testing.assert_allclose(written, expected, rtol=1e-5, atol=1e-6)


def test_a_chain_continues_after_the_reduction(tmp_path: Path) -> None:
    engine, destination, volumes = _run(
        tmp_path, [], Reduce(operator="Mean", output="avg"), [Clip(min_value=0.0, max_value=50.0)]
    )
    engine.materialize()

    written, _ = destination.read_data("CT", "avg")
    np.testing.assert_allclose(written, np.clip(volumes.mean(axis=0), 0.0, 50.0), rtol=1e-6)


def test_a_statistic_of_the_result_is_seeded_by_a_first_pass(tmp_path: Path) -> None:
    """`Normalize` after the reduction wants the Min/Max of a volume that is stored nowhere. The
    engine computes them by reducing once without writing, then reduces again to apply them --
    twice the reads, no intermediate volume."""
    engine, destination, volumes = _run(
        tmp_path, [], Reduce(operator="Mean", output="avg"), [Normalize(min_value=0.0, max_value=1.0)]
    )
    assert engine.plan().stat_pass is True
    engine.materialize()

    written, _ = destination.read_data("CT", "avg")
    reference = volumes.mean(axis=0)
    expected = (reference - reference.min()) / (reference.max() - reference.min())
    np.testing.assert_allclose(written, expected, rtol=1e-5, atol=1e-6)


def test_a_cohort_that_disagrees_on_geometry_is_refused_before_reading(tmp_path: Path) -> None:
    engine, _destination, _volumes = _run(
        tmp_path,
        [],
        Reduce(operator="Mean", output="avg"),
        [],
        spacings=[(1.0, 1.0, 2.0), (1.0, 1.0, 2.0), (1.0, 1.0, 9.0), (1.0, 1.0, 2.0)],
    )
    refusal = engine.check_grid()
    assert refusal is not None and "Spacing" in refusal and "CASE_002" in refusal
    with pytest.raises(ReductionError, match="cannot stream") as excinfo:
        engine.materialize()
    # The remedy must address the GRID: a Save changes nothing about the members' geometry, so the
    # generic put-a-Save advice would send the reader in a circle.
    assert "resample them onto a common grid" in str(excinfo.value)


def test_shape_only_accepts_approximate_headers(tmp_path: Path) -> None:
    engine, destination, volumes = _run(
        tmp_path,
        [],
        Reduce(operator="Mean", output="avg", grid="shape_only"),
        [],
        spacings=[(1.0, 1.0, 2.0), (1.0, 1.0, 2.0), (1.0, 1.0, 9.0), (1.0, 1.0, 2.0)],
    )
    assert engine.check_grid() is None
    engine.materialize()
    written, _ = destination.read_data("CT", "avg")
    np.testing.assert_allclose(written, volumes.mean(axis=0), rtol=1e-6)


def test_a_named_reference_adopts_its_geometry_instead_of_demanding_agreement(tmp_path: Path) -> None:
    """``reference:<case>`` is how a cohort says its members disagree and which one to believe.

    Its contract is exactly two checks: equal extents across the cohort, and the NAMED member's
    geometry on the output: no strict header comparison, whichever case happens to come first.
    """
    engine, destination, volumes = _run(
        tmp_path,
        [],
        Reduce(operator="Mean", output="avg", grid="reference:CASE_002"),
        [],
        spacings=[(1.0, 1.0, 2.0), (1.0, 1.0, 2.0), (1.0, 1.0, 9.0), (1.0, 1.0, 2.0)],
    )

    assert engine.check_grid() is None
    assert engine.reference.name == "CASE_002"
    engine.materialize()

    written, attribute = destination.read_data("CT", "avg")
    np.testing.assert_allclose(written, volumes.mean(axis=0), rtol=1e-6)
    # The output carries the NAMED member's geometry, which is the whole point of naming one.
    np.testing.assert_allclose(attribute.get_np_array("Spacing"), [1.0, 1.0, 9.0])


def test_a_named_reference_still_demands_equal_extents(tmp_path: Path) -> None:
    engine, _destination, _volumes = _run(
        tmp_path, [], Reduce(operator="Mean", output="avg", grid="reference:CASE_001"), []
    )
    engine.managers[2].shapes[0] = [4, 10, 6]

    refusal = engine.check_grid()
    assert refusal is not None and "CASE_002" in refusal and "CASE_001" in refusal


def test_concat_plans_the_channel_count_it_will_actually_write(tmp_path: Path) -> None:
    """A ``Concat`` writes ``cases x channels``, and only the operator can say so.

    ``transform_shape`` maps spatial extents and says nothing about the leading axis, so the planner
    took the source's channel count, which sized the regions at a quarter of what a four-case
    Concat holds, and had the write probe validate a shape the run never opens.
    """
    engine, destination, volumes = _run(tmp_path, [], Reduce(operator="Concat", output="stacked"), [])

    assert engine.plan().channels == CASES
    engine.materialize()
    written, _attribute = destination.read_data("CT", "stacked")
    assert written.shape[0] == CASES
    np.testing.assert_allclose(written, np.concatenate(list(volumes), axis=0), rtol=1e-6)


def test_the_deliverable_carries_its_own_recipe(tmp_path: Path) -> None:
    engine, destination, _volumes = _run(tmp_path, [], Reduce(operator="Median", output="template"), [])
    engine.materialize()

    _data, attribute = destination.read_data("CT", "template")
    assert attribute["konfai_reduce_operator"] == "Median"
    assert attribute["konfai_reduce_cases"] == "CASE_000|CASE_001|CASE_002|CASE_003"


def test_reduce_refuses_to_be_used_as_an_ordinary_transform() -> None:
    with pytest.raises(TransformError, match="reduces nothing"):
        Reduce(output="t")("CASE_000", torch.zeros(1, 2, 2, 2), Attribute())


def test_reduce_without_an_output_is_refused_at_construction() -> None:
    with pytest.raises(TransformError, match="needs an 'output'"):
        Reduce(operator="Mean")


def test_an_unknown_grid_policy_is_refused_at_construction() -> None:
    with pytest.raises(TransformError, match="unknown grid policy"):
        Reduce(output="t", grid="whatever")


def test_a_second_run_skips_the_finished_reduction(tmp_path: Path) -> None:
    engine, destination, _volumes = _run(tmp_path, [], Reduce(operator="Mean", output="avg"), [])
    engine.materialize()
    written, _ = destination.read_data("CT", "avg")
    store = next(path for path in tmp_path.iterdir() if path.name.startswith("out"))
    stamp = store.stat().st_mtime_ns

    engine.materialize()
    again, _ = destination.read_data("CT", "avg")
    np.testing.assert_array_equal(again, written)
    assert store.stat().st_mtime_ns == stamp


def test_a_region_stage_after_the_reduction_is_refused_not_seamed(tmp_path: Path) -> None:
    """A halo applied to one region at a time would seam at every boundary: plausible and wrong.

    It is deferred, not forbidden: materialize the reduction and read it back in a second chain."""
    dataset, _volumes = _cohort(tmp_path)
    with pytest.raises(ReductionError, match="reads across space"):
        CaseReduction(
            managers=_managers(dataset, []),
            reduce=Reduce(operator="Mean", output="avg"),
            post=[Dilate(2)],
            destination=Dataset(tmp_path / "out", "h5"),
            group="CT",
            slab_rows=3,
        )


# ------------------------------------------------ a custom operator, configured


class _Weighted(Reduction):
    """A custom operator with a parameter: the extension point, exercised."""

    voxel_local = True

    def __init__(self, weight: float = 1.0) -> None:
        self.weight = weight

    def __call__(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        return torch.stack(tensors, dim=0).float().mean(dim=0) * self.weight


class _Shadowing(Reduction):
    voxel_local = True

    def __init__(self, output: str = "x") -> None:
        self.output = output

    def __call__(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        return tensors[0]


class _NotVoxelLocal(Reduction):
    def __call__(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        return tensors[0]


def test_a_custom_operator_resolves_and_keeps_its_defaults_without_config() -> None:
    """A chain built in Python has no configuration to read: the operator uses its own defaults."""
    operator = resolve_operator(Reduce(operator=f"{__name__}:_Weighted", output="t"))
    assert isinstance(operator, _Weighted) and operator.weight == 1.0


def test_a_custom_operator_is_refused_when_it_cannot_fold_a_region() -> None:
    with pytest.raises(ReductionError, match="voxel-local"):
        resolve_operator(Reduce(operator=f"{__name__}:_NotVoxelLocal", output="t"))


def test_an_operator_shadowing_a_reduce_key_is_refused() -> None:
    """Operator parameters and Reduce's own keys share one mapping, so a collision is refused
    rather than silently handing the stage's value to the operator."""
    with pytest.raises(ReductionError, match="reads for itself"):
        resolve_operator(Reduce(operator=f"{__name__}:_Shadowing", output="t"))


def test_vote_picks_a_label_where_median_would_invent_one() -> None:
    """The reason Vote exists. Two label maps have no middle value that is a label.

    Median averages the middle pair, so folding structures 1 and 5 answers 3: a third structure,
    in a volume that is still a valid label map, which is why nothing downstream reports it.
    """
    labels = [torch.full((1, 1, 2, 2), value, dtype=torch.uint8) for value in (1, 5)]

    assert float(Median()(labels).flatten()[0]) == 3.0
    assert Median()(labels).dtype is torch.float32

    voted = Vote()(labels)
    assert float(voted.flatten()[0]) == 1.0, "a tie goes to the smallest label"
    assert voted.dtype is torch.uint8, "a vote picks, so the label dtype survives"


def test_vote_takes_the_label_the_majority_agrees_on() -> None:
    labels = [torch.full((1, 1, 2, 2), value, dtype=torch.uint8) for value in (4, 7, 7, 2, 7)]
    assert set(np.unique(Vote()(labels).numpy()).tolist()) == {7}


def test_vote_answers_per_voxel_not_per_volume() -> None:
    """Each voxel is its own ballot: a majority somewhere else must not carry it."""
    cases = [
        torch.tensor([[[[1, 2]]]], dtype=torch.uint8),
        torch.tensor([[[[1, 3]]]], dtype=torch.uint8),
        torch.tensor([[[[9, 3]]]], dtype=torch.uint8),
    ]
    np.testing.assert_array_equal(Vote()(cases).numpy(), np.array([[[[1, 3]]]], dtype=np.uint8))


def test_a_single_case_is_its_own_vote() -> None:
    only = torch.full((1, 1, 2, 2), 6, dtype=torch.uint8)
    assert Vote()([only]).dtype is torch.uint8
    np.testing.assert_array_equal(Vote()([only]).numpy(), only.numpy())


@pytest.mark.parametrize(
    "operator,output_channels,member_regions,why",
    [
        (Mean(), 1, 2, "one running accumulator and the region coming into it"),
        (Median(), 1, int(3.5 * CASES) + 1, "the cohort, what the four-member network holds beside it, the output"),
        (Vote(), 1, 3 * CASES + 1, "same shape of work: a mode sorts a copy of the stack too"),
        (Concat(), CASES, 2 * CASES, "the cohort, and the concatenation that IS the output"),
    ],
    ids=["mean", "median", "vote", "concat"],
)
def test_the_peak_is_charged_at_each_side_s_own_width(
    operator: Reduction, output_channels: int, member_regions: int, why: str
) -> None:
    """Members are measured at THEIR channel count, the output at its own.

    Only ``Concat`` tells the two apart (it writes ``N x C`` where each member holds ``C``) and
    charging the cohort at the output's width over-states its peak by the cohort's size, which either
    shrinks the slab for nothing or refuses a reduction that fits.
    """
    spatial, rows, source_channels = [8, 100, 100], 4, 1
    plan = ReductionPlan(
        output="t",
        cases=[f"CASE_{i:03d}" for i in range(CASES)],
        spatial=spatial,
        channels=output_channels,
        source_channels=source_channels,
        slab_rows=rows,
        incremental=operator.incremental,
        stat_pass=False,
        working_multiple=operator.working_multiple_for(CASES),
    )
    member = rows * spatial[1] * spatial[2] * source_channels * 4
    assert plan.peak_bytes == member_regions * member, why


@pytest.mark.parametrize("cases", [1, 2, 3, 4, 5, 6, 7])
def test_median_selects_the_middle_instead_of_sorting_the_stack(cases: int) -> None:
    """Up to five members the middle is SELECTED by a network of element-wise min/max; past that a
    sort finds it. The values are the same to the bit either way -- ``torch.quantile`` is the
    reference the docstring names -- and the network holds far less, which is what
    ``working_multiple_for`` reports so the planner can cut taller slabs where it runs.

    Measured on a 24 MiB member: at three members 7.20 ms and 3x the stack by sort against 0.45 ms
    and 1x by network on CUDA, 55 ms against 26 on the host.
    """
    torch.manual_seed(3)
    members = [torch.rand((1, 2, 4, 5)) for _ in range(cases)]
    folded = Median()(members)

    assert torch.equal(folded, torch.quantile(torch.stack(members, dim=0), 0.5, dim=0))
    assert Median().working_multiple_for(cases) == {1: 1.0, 2: 1.5, 3: 1.0, 4: 2.5, 5: 1.5}.get(cases, 4.0)


def test_reading_the_members_at_once_writes_the_same_bytes_as_one_at_a_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A member's read is a decode plus a replay of that case's chain, and the members are
    independent, so a non-incremental fold -- which holds every member anyway -- reads several at
    once. Every case's chain shares the same stage OBJECTS, so this is where a shared cache would
    show: what it must produce is the same volume, to the byte, as reading one at a time.

    The pre-chain here is deliberately stateful across cases: ``Crop`` memoises a content-derived box
    per case on the dataset, and ``Resample`` caches a grid per name.
    """
    import konfai.data.case_reduction as case_reduction

    pre = [Clip(0.0, 50.0)]
    monkeypatch.setattr(case_reduction, "_MEMBER_READERS", 1)
    engine, destination, _volumes = _run(tmp_path, pre, Reduce(operator="Median", output="serial"), [])
    engine.materialize()
    one_at_a_time, _ = destination.read_data("CT", "serial")

    monkeypatch.setattr(case_reduction, "_MEMBER_READERS", 4)
    engine, destination, _volumes = _run(tmp_path, pre, Reduce(operator="Median", output="together"), [])
    engine.materialize()
    together, _ = destination.read_data("CT", "together")

    np.testing.assert_array_equal(together, one_at_a_time)


def test_a_vote_tie_goes_to_the_smallest_label_whatever_the_cohort_order() -> None:
    """The reproducibility half of Vote's contract, which its docstring promises out loud.

    A cohort is folded by whichever rank owns it and in whatever order the manager list came out, so
    a tie broken by position would write a different template on a rerun and nothing about the volume
    would look wrong. ``torch.mode`` documents that the smallest of the most frequent values wins;
    this pins that the whole way through, because the promise is ours and not torch's to keep.
    """
    for order in itertools.permutations((7, 2, 9)):
        cohort = [torch.full((1, 1, 2, 2), value, dtype=torch.uint8) for value in order]
        assert int(Vote()(cohort).flatten()[0]) == 2, f"order {order} broke the tie somewhere else"

    # A majority still beats the smallest label: the tie rule is a tie-breaker, not a preference.
    counted = [torch.full((1, 1, 2, 2), value, dtype=torch.uint8) for value in (9, 2, 9)]
    assert int(Vote()(counted).flatten()[0]) == 9


def test_a_budget_with_room_folds_the_whole_volume(tmp_path: Path) -> None:
    """The budget is the ONLY ceiling: room in it must reach the region height.

    A fixed cap of 64 rows did the opposite of what it looked like. It bounded the region however
    much memory the run was given, so a cohort whose regions measured 0.11 GiB against a 59.60 GiB
    budget still paid a full sweep of every source per 64 rows -- and the sweeps are what a chain
    resampling through a field pays for, since a region's source window is not the region. Measured
    on a 192-plane cohort: 191.5 s at the cap, 45.6 s once the budget decided.
    """
    engine, _destination, _volumes = _run(tmp_path, [], Reduce(output="t"), [], slab_rows=3)
    height = engine.plan().spatial[0]

    engine.fit_budget(64 * 1024**3)  # room for far more than the volume holds
    assert engine.slab_rows == height, "a budget with room must reach the whole volume"

    # And it is still a budget: too little room, and the region shrinks to what fits.
    engine.fit_budget(4 * 1024)
    assert 1 <= engine.slab_rows < height

    # A caller that names a ceiling still gets it.
    engine.fit_budget(64 * 1024**3, cap=2)
    assert engine.slab_rows == 2
