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
import time
import weakref
from pathlib import Path
from types import SimpleNamespace

import konfai.data.patching as patching_module
import konfai.data.transform as transform_module
import numpy as np
import pytest
import torch
from konfai.data.case_reduction import (
    _KEPT_FOLDS_SHARE_OF_REGIONS,
    CaseReduction,
    ReductionPlan,
    split_chain,
)
from konfai.data.patching import (
    SWEEP_CLOCK,
    SWEEP_SLAB_ROWS,
    DatasetManager,
    HeldMeter,
    open_held_meter,
)
from konfai.data.reduction import Concat, Mean, Median, Reduction, Vote
from konfai.data.transform import (
    Clip,
    Dilate,
    Normalize,
    Reduce,
    Resample,
    Save,
    TensorCast,
    Transform,
    resolve_operator,
)
from konfai.utils.budget import BUDGET_SHARES, budget_share
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


def _field_cohort(tmp_path: Path, count: int = CASES) -> tuple[Dataset, np.ndarray]:
    """A cohort and, beside it, a displacement field per case: coarse against the volume and steep.

    Sized after the case this exists for. A registration field is read at a fraction of the density
    of the volume it moves -- four times coarser here -- and its values reverse between neighbouring
    nodes, which is the steepest thing a lattice can carry: the interpolated displacement then swings
    the full amplitude across one cell, and a region's face cuts through that swing.
    """
    dataset, volumes = _cohort(tmp_path, count=count)
    lattice = (3, 4, 3)
    parity = (-1.0) ** np.add.outer(np.add.outer(np.arange(lattice[0]), np.arange(lattice[1])), np.arange(lattice[2]))
    reference = np.zeros((1, 7, 9, 5), dtype=np.float32)
    for index in range(count):
        field = np.stack([weight * 4.0 * parity for weight in (1.0, -0.7, 0.5)]).astype(np.float32)
        dataset.write("FIELD", f"CASE_{index:03d}", field, _attributes((3.0, 3.0, 4.0)))
        # A grid of its own for the stage to resample ONTO. The pipeline case this stands for changes
        # grid AND applies a field in the same stage, which is the combination nothing covered: a
        # field alone leaves the target grid where it was, and the source window is then a
        # translation of the target's rather than a box that has to be found.
        dataset.write("REF", f"CASE_{index:03d}", reference, _attributes((1.3, 1.1, 2.4)))
    return dataset, volumes


@pytest.mark.parametrize("other_rows", [2, 5, 8])
def test_a_field_driven_resample_under_a_reduce_does_not_depend_on_the_slab(tmp_path: Path, other_rows: int) -> None:
    """The fold's answer must be the same whatever height it cuts its regions at.

    A Reduce streams slabs along the first spatial axis, and the per-member stage before it is a
    REGRID that sizes the source window it reads from the field values it finds inside the region.
    Whether that window still covers what the interpolator produces at the region's FACES -- where it
    blends field nodes the region does not contain -- is what decides whether the answer depends on
    where the slabs were cut. It must not: streaming is an optimisation, so it has to be invisible.

    Measured on a real build before this test existed: the same ten brains folded with regions of 24
    rows and of 8 rows gave templates that differ, with steps in the slab axis and in no other, at
    planes that moved with the region height.
    """
    reference = None
    for rows in (3, other_rows):
        dataset, _ = _field_cohort(tmp_path / f"rows{rows}")
        destination = Dataset(tmp_path / f"out{rows}", "h5")
        engine = CaseReduction(
            # Named by PATH, not by group alone: a field found by group is looked up among the
            # run's own roots, and this harness builds managers directly rather than a run.
            managers=_managers(
                dataset,
                [
                    Resample(
                        reference="CASE_000",
                        reference_group="REF",
                        reference_dataset=f"{dataset.filename}:h5",
                        field=f"{dataset.filename}:h5",
                        field_group="FIELD",
                    )
                ],
                CASES,
            ),
            reduce=Reduce(operator="Median", output="template"),
            post=[],
            destination=destination,
            group="CT",
            slab_rows=rows,
        )
        assert engine.materialize() is True
        written, _ = destination.read_data("CT", "template")
        if reference is None:
            reference = written
        else:
            np.testing.assert_allclose(written, reference, rtol=0, atol=1e-5)


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
        (Vote(), 1, CASES + 5, "the cohort, the fixed planes the running best holds beside it, the output"),
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


@pytest.mark.parametrize("cases", [1, 2, 3, 4, 5, 6, 7, 10, 16])
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
    assert Median().working_multiple_for(cases) == {1: 1.0, 2: 1.5, 3: 1.0, 4: 2.5, 5: 1.5}.get(cases, 1.8)


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.uint8, torch.int16, torch.int32, torch.int64, torch.float64]
)
def test_mean_promotes_each_member_inside_the_add(dtype: torch.dtype) -> None:
    """A member is added at its own dtype and the kernel casts it to float32 on the way: the
    float() that ran first materialised a float32 copy of the member beside the total (64 MiB per
    32 MiB fp16 member; one allocation per member on CUDA, none now, measured). That spelling is
    the oracle here, dtype by dtype: the same bits, float64 included, which the fold still narrows
    first because the kernel would otherwise add at 64 bits and round after.
    """
    torch.manual_seed(0)
    if dtype.is_floating_point:
        members = [(torch.randn(1, 1, 3, 5, 7, dtype=torch.float64) * 1e3).to(dtype) for _ in range(4)]
    else:
        bound = min(torch.iinfo(dtype).max, 2**30)
        members = [
            torch.randint(max(torch.iinfo(dtype).min, -bound), bound, (1, 1, 3, 5, 7), dtype=dtype) for _ in range(4)
        ]

    total = members[0].to(torch.float32, copy=True)
    for member in members[1:]:
        total.add_(member.float())
    oracle = total.div_(len(members)).to(dtype if dtype.is_floating_point else torch.float32)

    folded = Mean()(members)
    assert folded.dtype is oracle.dtype
    assert torch.equal(folded, oracle)


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


def _vote_by_sorting(tensors: list[torch.Tensor]) -> torch.Tensor:
    """The stack sorted along the case axis and counted member by member: the spelling Vote
    shipped with, kept as the oracle of its labels and its tie rule."""
    ranked = torch.stack(tensors, dim=0).sort(dim=0).values
    best, best_count = ranked[0], (ranked == ranked[0]).sum(dim=0, dtype=torch.int16)
    for member in ranked[1:]:
        count = (ranked == member).sum(dim=0, dtype=torch.int16)
        better = count > best_count
        best = torch.where(better, member, best)
        best_count = torch.where(better, count, best_count)
    return best


def test_median_keeps_integer_members_narrow_and_holds_a_window_not_a_stack() -> None:
    """Ten uint16 regions are folded without ten float32 copies of them and without a sorted stack.

    A sort along the case axis returns int64 indices, eight bytes an element whatever the members
    weigh: ten uint16 regions sorted at 6.0x their own size. The window holds k + 1 float32 buffers
    and nothing else -- 1.8x, measured on a 293 MiB member -- and the members stay uint16.
    """
    torch.manual_seed(5)
    members = [torch.randint(0, 60000, (1, 1, 8, 96, 96), dtype=torch.int32).to(torch.uint16) for _ in range(10)]
    folded = Median()(members)
    ranked = torch.stack([member.float() for member in members], dim=0).sort(dim=0).values
    assert torch.equal(folded, torch.lerp(ranked[4], ranked[5], 0.5)), "the window selects what the sort ranks"
    assert folded.dtype is torch.float32
    assert all(member.dtype is torch.uint16 for member in members), "the members were not widened in place"
    assert Median().working_multiple_for(10) == Median._WINDOW_MULTIPLE < Median.working_multiple + 1


@pytest.mark.parametrize("cases", [2, 3, 4, 6, 7])
@pytest.mark.parametrize("dtype", [torch.uint8, torch.int16, torch.int32, torch.float32])
def test_vote_counts_every_candidate_instead_of_sorting_the_stack(cases: int, dtype: torch.dtype) -> None:
    """The label with the most votes, the smallest on a tie, found by a running best over each
    member's count: no stack and no sort, whose int64 indices alone were 8 bytes per voxel and
    member. Measured on a 128-row slab of six 512x512 uint8 members on CUDA: 1920 MiB and 64 ms
    sorted, 256 MiB and 11 ms here; on the host, 351 -> 127 ms at two members. The sorted
    spelling is the oracle: the same labels to the bit on cohorts drawn from four labels, where
    most voxels tie.
    """
    torch.manual_seed(cases)
    members = [torch.randint(0, 4, (1, 1, 5, 6, 7)).to(dtype) for _ in range(cases)]
    voted = Vote()(members)
    assert voted.dtype is dtype
    assert torch.equal(voted, _vote_by_sorting(members))
    assert Vote().working_multiple_for(cases) == 4.0 / cases
    assert Vote().working_multiple_for(1) == 0.0


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


def test_a_reduction_accounts_for_every_second_of_its_own_wall_clock(tmp_path: Path) -> None:
    """A work item that reads N volumes must appear in the run's one accounting line.

    Measured on a 5 x 384 MiB cohort (.audit-local/bench/bench_reduce_expand.py): the member reads
    are 13-40 % of the work item, the operator 13-37 %, the write 4-9 %. None of it was attributed:
    the fold reads through ``read_region``, which no phase of the sweep clock covers.
    """
    SWEEP_CLOCK.reset()
    engine, _destination, _volumes = _run(tmp_path, [], Reduce(operator="Mean", output="clocked"), [])
    assert engine.materialize() is True

    wall = SWEEP_CLOCK.spent("sweep")
    phases = {phase: SWEEP_CLOCK.spent(phase) for phase in ("chain", "fetch", "wait(read)", "wait(write)")}
    assert wall > 0.0
    # Every phase of the line is recorded, or the report accounts for the run by leaving it out.
    assert all(spent > 0.0 for spent in phases.values()), phases
    assert 0.0 <= wall - sum(phases.values()) <= wall, "a phase is counted outside the fold it belongs to"
    # No pipeline: the store's own seconds are the seconds the fold stands still.
    assert SWEEP_CLOCK.spent("wait(read)") >= SWEEP_CLOCK.spent("read") > 0.0
    assert SWEEP_CLOCK.spent("wait(write)") >= SWEEP_CLOCK.spent("write") > 0.0


def test_the_fold_charges_the_operator_s_own_work_to_the_chain(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An incremental operator does the reduction in ``accumulate``, not in ``finalize``.

    Timed around ``finalize`` alone, the line credited a ``Mean`` or a ``Std`` with the last
    division and left every addition it made in ``other``: the phase that is supposed to say what
    the reduction costs under-reported it by the whole of the fold.
    """
    delay = 0.02
    accumulate = Mean.accumulate

    def slow(self: Mean, member: torch.Tensor) -> None:
        time.sleep(delay)
        accumulate(self, member)

    monkeypatch.setattr(Mean, "accumulate", slow)
    SWEEP_CLOCK.reset()
    engine, _destination, _volumes = _run(tmp_path, [], Reduce(operator="Mean", output="timed"), [])
    assert engine.materialize() is True

    assert SWEEP_CLOCK.spent("chain") >= CASES * delay
    assert SWEEP_CLOCK.spent("read") < CASES * delay, "the member reads are not the operator's work"


def test_an_incremental_fold_holds_one_member_region_at_a_time(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``resident_regions == 2`` is a promise about what is alive, not only about what is planned.

    A name in the fold loop still holding the region the accumulator has already folded keeps a
    second one resident: one member region over the priced peak, which on a 5 x 384 MiB float32
    cohort folded whole measured 1162 MiB against 778.
    """
    alive: list[weakref.ref] = []
    read_region = DatasetManager.read_region

    def counted(self, target, a=0, apply_augmentations=False):
        tensor = read_region(self, target, a, apply_augmentations)
        alive.append(weakref.ref(tensor))
        resident = sum(1 for reference in alive if reference() is not None)
        assert resident == 1, f"{resident} member regions alive while reading the next one"
        return tensor

    monkeypatch.setattr(DatasetManager, "read_region", counted)
    engine, _destination, _volumes = _run(tmp_path, [], Reduce(operator="Mean", output="one_at_a_time"), [])
    assert engine.plan().resident_regions == 2
    assert engine.materialize() is True
    assert len(alive) > CASES, "the fold read fewer regions than it has cases"


def test_the_plan_prices_what_a_member_chain_holds_beside_its_region(tmp_path: Path) -> None:
    """A reduction's peak is not the operator's buffers alone: each member REPLAYS its chain to
    produce its region, and a stage that allocates beyond what it is handed holds that too.

    ``Resample`` declares ``working_multiple = 6.5`` (its sampling grid, its taps, and the widening a
    stored integer forces), which
    ``DatasetManager.working_multiple()`` already reports; the plan asks only the OPERATOR, whose
    multiple for ``Mean`` is 0.0. Measured on the ExaSPIM prep Fold over a 3-channel native field
    (``[514, 1331, 1775]``, 8 regions of 70 rows, incremental ``Mean``): the plan announced
    3.696 GiB and the run peaked at 14.5 GiB, of which one member read alone added 2.161 GiB.
    """
    plain = _run(tmp_path / "plain", [], Reduce(operator="Mean", output="t"), [])[0].plan()
    # The cohort's own spacing, so the grid is untouched and the ONLY thing that differs between
    # the two plans is what the chain holds while it produces a region.
    resampled = _run(
        tmp_path / "resampled", [Resample(spacing=[1.0, 1.0, 2.0])], Reduce(operator="Mean", output="t"), []
    )[0]
    assert resampled.managers[0].working_multiple() == 6.5, "the chain declares what it holds"
    assert resampled.plan().spatial == plain.spatial, "same grid: the chain's hold is the one variable"
    assert resampled.plan().peak_bytes > plain.peak_bytes, (
        "a chain holding three volumes-worth per member region must price above one that holds none:"
        " the plan is what the region sizing is derived from, so a peak it under-states is not a bound"
    )


def test_a_generous_budget_stops_at_the_plateau_and_never_below_the_slab_floor(tmp_path: Path) -> None:
    """The budget is a CEILING, not a target: past the height where a chain reads no fewer source
    voxels, a taller region only holds more, so the sizing stops there however much memory it is
    given. And never below the per-region floor: a chain that pulls exactly what it lands has a
    flat read curve whose plateau starts at one row, and one-row regions pay every fixed cost of a
    region for one row of work.
    """
    # Clip, not a Resample: a resample's taps reach a halo, so its read curve is not flat and its
    # plateau is not one row. A POINTWISE stage is the case the floor exists for.
    engine = _run(tmp_path, [Clip(0.0, 50.0)], Reduce(operator="Mean", output="t"), [])[0]
    engine.fit_budget(64 * (1 << 30))  # far more than this cohort could ever hold
    assert engine.slab_rows >= min(SWEEP_SLAB_ROWS, engine.plan().spatial[0]), (
        "an identity resample pulls what it lands, so its plateau is one row: the floor is what"
        " keeps the sizing off one-row regions"
    )


def test_a_field_resample_prices_the_field_window_its_case_actually_holds(tmp_path: Path) -> None:
    """What a stage holds can be a property of its CONFIGURATION, not of its class.

    A ``Resample`` through a field solved on the case's own grid holds three channels of it beside
    the sampling grid; through a field solved coarser it holds a fraction of that. Same class, same
    ``working_multiple``, different holds -- so the plan asks the stage about the case
    (``case_working_multiple``) rather than reading the class attribute.

    Measured on the prep's appearance fold over native-resolution fields: priced at the class's
    figure the plan announced 12.41 GiB, and the run held 38.2.
    """
    _cohort(tmp_path / "cases")
    fields = Dataset(tmp_path / "dvf", "h5")
    for index in range(CASES):
        # Same grid as the cohort: one field component per axis, at the case's own density.
        fields.write("DVF", f"CASE_{index:03d}", np.zeros((3, 8, 10, 6), np.float32), _attributes())
    warp = Resample(field=f"{tmp_path / 'dvf'}:h5", field_group="DVF")
    for index in range(CASES):
        warp.transform_shape("CT", f"CASE_{index:03d}", [8, 10, 6], _attributes())

    plain = Resample(spacing=[1.0, 1.0, 2.0])
    plain.transform_shape("CT", "CASE_000", [8, 10, 6], _attributes())
    # A separable map has no field window and is not walked coordinate by coordinate.
    assert plain.case_working_multiple("CASE_000") == plain.working_multiple, "no field, no field window"
    # Three components at the case's own density, each weighing TWO of the plan's volumes (a field
    # is read as float64 and the plan counts a volume at four bytes), materialised THREE times over
    # while ITK is handed it. The general walk this case also takes is NOT added: it slabs itself
    # against the declared budget.
    expected = warp.working_multiple + 3.0 * 2.0 * transform_module._FIELD_WINDOW_COPIES
    assert warp.case_working_multiple("CASE_000") == pytest.approx(expected)
    # A case this stage has never met answers the class's figure rather than guessing at a grid.
    assert warp.case_working_multiple("NEVER_SEEN") == warp.working_multiple


def _engine_for_refit(tmp_path: Path, budget: float, rows: int) -> CaseReduction:
    engine = _run(tmp_path, [], Reduce(operator="Mean", output="t"), [])[0]
    engine._budget_bytes = budget
    engine.slab_rows = rows
    for manager in engine.managers:
        manager.set_chain_device(torch.device("cuda:0"))
    return engine


def test_the_probe_is_short_so_its_own_overshoot_cannot_be_the_kill(tmp_path: Path) -> None:
    """The one region no measurement can bound is the probe, because it runs before any.

    At the planned height a probe over registration fields held 1.5x its allowance three times
    out of three, and at an `auto` budget of 77 GiB on a 122 GiB host the probe alone reached 90
    GiB and the host went down with nothing measured. The host gives nothing to catch: the kernel
    kills. So the probe walks a quarter of the planned height, and every region after it walks
    what the projection allows.
    """
    engine = _run(tmp_path / "probe", [], Reduce(operator="Mean", output="t"), [])[0]
    engine._budget_bytes = 1 << 30
    engine.slab_rows = 100
    walked = []
    engine._fold = lambda region: walked.append(region[0].stop - region[0].start) or torch.zeros(1)  # type: ignore[method-assign]
    engine._open_meter = lambda: HeldMeter(lambda: 1, 0)  # type: ignore[method-assign]
    list(engine._folds([1000, 10, 6], measure=True))
    assert walked[0] == int(100 * CaseReduction._PROBE_SHARE), "the probe is a quarter of the planned height"
    assert all(rows == 100 for rows in walked[1:-1]), "and the rest walk the planned height once it fits"
    assert sum(walked) == 1000, "every row is folded exactly once"


def test_the_first_region_is_the_probe_and_only_ever_shortens_the_rest(tmp_path: Path) -> None:
    """What the plan priced is a model; what the first region held is a fact.

    The model has to be right about every stage, every store and every bridge the chain crosses,
    and this work found it wrong at all three. The measurement costs one counter read on a region
    the fold had to compute anyway, and it bounds the next region from above because a high-water
    mark does -- the same reading Predictor._accumulate_device makes of the batch that just ran.

    Driven through a HeldMeter built on a chosen reading, so the arithmetic is exercised on both
    routes at once: which instrument answered is the meter's business and not the fold's.
    """
    budget = 1 << 30
    # Less the cache's share: the meter does not count the decoded-chunk cache, so the comparison
    # does not offer it either -- see test_the_allowance_leaves_the_chunk_cache_its_share.
    allowed = (budget - (budget_share("cache", budget) or 0.0)) * CaseReduction._MEASURED_MARGIN

    def engine_holding(name: str, held: int) -> CaseReduction:
        engine = _run(tmp_path / name, [], Reduce(operator="Mean", output="t"), [])[0]
        engine._budget_bytes = budget
        engine.slab_rows = 100
        return engine

    # A full-height probe that held twice what the declaration allows: the rest are cut to half.
    engine = engine_holding("over", 0)
    engine._refit_to_measurement(HeldMeter(lambda: int(allowed * 2), 0), 100, [1000, 10, 6])
    assert engine.slab_rows == 50

    # A QUARTER-height probe that held half the allowance: scaled to the planned height that is
    # twice the allowance, and the rest are cut to half. The probe is short so that its own
    # overshoot cannot reach the host; what it measures is projected before it is judged.
    engine = engine_holding("short-probe", 0)
    engine._refit_to_measurement(HeldMeter(lambda: int(allowed * 0.5), 0), 25, [1000, 10, 6])
    assert engine.slab_rows == 50

    # Held less than the declaration: nothing moves. A measurement is never a licence to spend a
    # budget the sizing declined to spend, and the region that set the peak has already run.
    engine = engine_holding("under", 0)
    engine._refit_to_measurement(HeldMeter(lambda: int(allowed * 0.25), 0), 100, [1000, 10, 6])
    assert engine.slab_rows == 100, "a region that fits must not make the next one taller"
    # ... and a short probe under ITS share does not either, once projected.
    engine = engine_holding("under-short", 0)
    engine._refit_to_measurement(HeldMeter(lambda: int(allowed * 0.2), 0), 25, [1000, 10, 6])
    assert engine.slab_rows == 100

    # The baseline is subtracted: the same peak over a higher starting point held less.
    engine = engine_holding("baseline", 0)
    engine._refit_to_measurement(HeldMeter(lambda: int(allowed * 2), int(allowed * 1.9)), 100, [1000, 10, 6])
    assert engine.slab_rows == 100

    # Never below one row, however badly the first region overshot.
    engine = engine_holding("tiny", 0)
    engine.slab_rows = 4
    engine._refit_to_measurement(HeldMeter(lambda: int(allowed * 10_000), 0), 4, [1000, 10, 6])
    assert engine.slab_rows == 1

    # An instrument that went quiet, and no instrument at all, both leave the planned height.
    engine = engine_holding("blind", 0)
    engine._refit_to_measurement(HeldMeter(lambda: None, 0), 100, [1000, 10, 6])
    assert engine.slab_rows == 100
    engine._refit_to_measurement(None, 100, [1000, 10, 6])
    assert engine.slab_rows == 100

    # No budget declared: nothing bounds anything, so nothing is re-fitted.
    engine = engine_holding("none", 0)
    engine._budget_bytes = 0.0
    engine._refit_to_measurement(HeldMeter(lambda: int(allowed * 8), 0), 100, [1000, 10, 6])
    assert engine.slab_rows == 100


def test_a_host_chain_gets_a_meter_and_a_kernel_without_one_gets_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """A host chain has no allocator counter, and the kernel's resident peak is the meter it does
    have. A kernel that offers no reset gets no meter at all, rather than a run-long peak read as
    if it were one region's."""
    monkeypatch.setattr(patching_module, "reset_resident_peak", lambda: True)
    monkeypatch.setattr(patching_module, "resident_bytes", lambda: 1_000_000)
    monkeypatch.setattr(patching_module, "peak_resident_bytes", lambda: 1_500_000)
    meter = open_held_meter(None)
    assert meter is not None and meter.held() == 500_000

    monkeypatch.setattr(patching_module, "reset_resident_peak", lambda: False)
    assert open_held_meter(None) is None


def test_the_allowance_leaves_the_chunk_cache_its_share(tmp_path: Path) -> None:
    """The meter's reading and the figure it is judged against must cover the same bytes.

    The meter stopped counting the decoded-chunk cache (it outlives the region, and charging the
    region for it cut every region after the probe). Judged against the whole budget, a reading
    that excludes the cache would let the cache be spent twice: once inside the allowance the
    regions may fill, and again by the cache itself -- which is how a run held 1.09x what it
    declared. The allowance comes down by exactly the cache's share.
    """
    budget = 1 << 30
    engine = _run(tmp_path, [], Reduce(operator="Mean", output="t"), [])[0]
    engine._budget_bytes = budget
    engine.slab_rows = 100

    cache = budget_share("cache", budget) or 0.0
    assert cache > 0, "the fixture only says anything where the cache has a share"
    # A probe holding just under the OLD allowance (the whole budget) and above the new one.
    held = int((budget - cache / 2) * CaseReduction._MEASURED_MARGIN)
    engine._refit_to_measurement(HeldMeter(lambda: held, 0), 100, [1000, 10, 6])
    assert engine.slab_rows < 100, "a region filling the cache's share as well is cut"

    engine.slab_rows = 100
    fits = int((budget - cache) * CaseReduction._MEASURED_MARGIN)
    engine._refit_to_measurement(HeldMeter(lambda: fits, 0), 100, [1000, 10, 6])
    assert engine.slab_rows == 100, "a region inside the allowance keeps the height the plan chose"


def test_a_host_meter_does_not_charge_the_scope_for_the_chunk_cache_it_filled(monkeypatch: pytest.MonkeyPatch) -> None:
    """The decoded-chunk cache sits inside the process's resident peak and outlives any one scope.

    A fold's probe read VmHWM over its ten members and charged the region 24.4 GiB, 13.2 of which
    was the cache filling from empty -- a budget line with a share of its own -- and cut every
    region after it to 78 % of the height that would have fit. What the cache gained during the
    scope is subtracted; what the scope held on its own is what it is charged.
    """
    from konfai.utils import ome_zarr

    cache = ome_zarr._DecodedChunkCache(1 << 30)
    monkeypatch.setattr(ome_zarr, "_CHUNK_CACHE", cache)
    monkeypatch.setattr(patching_module, "reset_resident_peak", lambda: True)
    monkeypatch.setattr(patching_module, "resident_bytes", lambda: 1_000_000)
    peak = {"value": 1_000_000}
    monkeypatch.setattr(patching_module, "peak_resident_bytes", lambda: peak["value"])

    meter = open_held_meter(None)
    chunk = np.ones((256, 256), np.float32)  # 256 KiB decoded, kept by the cache past the scope
    cache.put(("store", 0), chunk)
    peak["value"] = 1_000_000 + chunk.nbytes + 100_000  # the process grew by the chunk and by the scope's own 100 KB
    assert meter is not None and meter.held() == 100_000, "the cache's growth is not the scope's cost"


def test_the_folds_a_stat_pass_keeps_come_out_of_the_regions_share(tmp_path: Path) -> None:
    """A kept fold is memory the regions are not holding, so it comes out of the same share.

    A stat pass can hand its folds to the write pass instead of re-folding them: a memory trade, not
    a correctness one. It was allowed half the WHOLE declaration, beside regions that had already
    taken half and members that had been handed all of it -- one and a half times the budget, from
    three consumers each sure it was the only one.
    """
    from konfai.data.transform import Standardize  # GLOBAL_STAT after the fold: the stat pass runs

    engine, _destination, _volumes = _run(
        tmp_path, [], Reduce(operator="Mean", output="t"), [Standardize()], slab_rows=3
    )
    plan = engine.plan()
    assert plan.stat_pass, "a GLOBAL_STAT stage after the fold is what makes the second pass"
    output = engine._folded_output_bytes(plan)
    share = BUDGET_SHARES["regions"] * _KEPT_FOLDS_SHARE_OF_REGIONS

    # The output at 40 % of the declaration: it fitted the half the old rule allowed, and does not
    # fit the share of the FOLD's own half that is all it may have.
    engine.fit_budget(output / 0.4)
    assert not engine.keeps_folds(engine.plan()), (
        "folds that fit half the whole budget do not fit their share of it: the regions and the"
        " members are holding the rest at the same moment"
    )

    # Just inside its share: kept. Just outside: not.
    engine.fit_budget(output / (share * 0.99))
    assert engine.keeps_folds(engine.plan())
    engine.fit_budget(output / (share * 1.01))
    assert not engine.keeps_folds(engine.plan())

    # A chain with no stat pass never keeps anything, whatever it would fit.
    plain = _run(tmp_path / "plain", [], Reduce(operator="Mean", output="t"), [])[0]
    plain.fit_budget(output * 1000)
    assert not plain.plan().stat_pass and not plain.keeps_folds(plain.plan())

    # No budget: nothing is subtracted from nothing, and nothing is kept.
    engine.fit_budget(None)
    assert not engine.keeps_folds(engine.plan())


def test_a_value_neutral_stage_does_not_decide_the_region_height(tmp_path: Path) -> None:
    """The sizing must follow the data, not the shape of the stage list.

    A chain with no stage before the fold has no segment to read a pull map from, so every member
    answers None when asked where its reads stop paying. Answering the output's own HEIGHT there
    made the ceiling depend on something that is not a property of the cohort: the same cases, the
    same budget and the same voxels read gave the whole volume with no stage and the floor with one
    ``Clip`` whose bounds lie outside the data. An eightfold difference in what the fold holds, from
    a stage that changes no voxel.

    It was also the one place the run-time probe could not help: at full height a fold has exactly
    one region, and the probe only ever cuts the ones after the first.
    """
    # Taller than SWEEP_SLAB_ROWS and tiny in bytes: the difference the fallback makes is between
    # the floor and the whole height, so a cohort shorter than the floor cannot show it at all.
    dataset = Dataset(tmp_path / "tall", "h5")
    for index in range(CASES):
        dataset.write("CT", f"CASE_{index:03d}", np.zeros((1, 200, 4, 4), np.float32), _attributes())
    budget = 8 << 30

    def rows(pre: list) -> int:
        engine = CaseReduction(
            managers=_managers(dataset, pre),
            reduce=Reduce(operator="Mean", output="t"),
            post=[],
            destination=Dataset(tmp_path / f"out{len(pre)}", "h5"),
            group="CT",
            slab_rows=3,
        )
        engine.fit_budget(budget)
        return engine.slab_rows

    # Clip's bounds are outside the fixture's range, so it is the identity on every voxel.
    assert rows([]) == rows([Clip(-1.0e9, 1.0e9)]), "a stage that changes no voxel must not change how the fold is cut"


# ---------------------------------------------------------------- what a chunked store adds


def test_a_store_that_serves_what_it_is_asked_for_adds_nothing_to_the_peak(tmp_path: Path) -> None:
    """The charge is the EXCESS a block-aligned decode materialises above the window, so a backend
    with no block grid contributes zero and the peak says exactly what it said before."""
    engine, _destination, _volumes = _run(tmp_path, [], Reduce(operator="Mean", output="avg"), [])
    assert engine.plan().read_bytes == 0


def test_a_fold_is_judged_against_the_share_its_height_was_solved_for(tmp_path: Path) -> None:
    """``fit_budget`` solves the height against the regions' share, so that is what the peak is
    judged against. Judged against the whole budget instead, the gate could only speak when a
    single row did not fit: a fold needing twice its share announced a plan it could not keep and
    was killed by the kernel rather than refused here."""
    from konfai.transformer import TransformPlan, Verdict

    engine, _destination, _volumes = _run(tmp_path, [], Reduce(operator="Median", output="m"), [])
    budget = float(engine.plan().peak_bytes) * 1.5  # over the regions' share, under the whole budget
    plan = SimpleNamespace(budget_bytes=budget)
    assert TransformPlan.ceiling_for(plan, Verdict.REDUCE) < engine.plan().peak_bytes
    assert TransformPlan.ceiling_for(plan, Verdict.WHOLE_VOLUME) == budget
