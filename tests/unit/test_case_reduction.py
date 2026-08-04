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

What matters here is not that the median is right -- it is that it is right while the volumes are
never assembled, that a cases which does not agree on its grid is refused before anything is read,
and that a chain continues after the reduction."""

from pathlib import Path

import numpy as np
import pytest
import torch
from konfai.data.case_reduction import CaseReduction, resolve_operator, split_chain
from konfai.data.patching import DatasetManager
from konfai.data.reduction import Mean, Median, Reduction
from konfai.data.transform import Clip, Dilate, Normalize, Reduce, TensorCast, Transform
from konfai.utils.dataset import Attribute, Dataset
from konfai.utils.errors import ReductionError, TransformError

CASES = 4  # even on purpose: an even cases is where a lower-median would show


def _attributes(spacing: tuple[float, float, float] = (1.0, 1.0, 2.0)) -> Attribute:
    attribute = Attribute()
    attribute["Origin"] = np.asarray([0.0, 0.0, 0.0])
    attribute["Spacing"] = np.asarray(list(spacing))
    attribute["Direction"] = np.eye(3, dtype=np.float64).reshape(-1)
    return attribute


def _cohort(tmp_path: Path, count: int = CASES, spacings: list | None = None) -> tuple[Dataset, np.ndarray]:
    rng = np.random.default_rng(0)
    volumes = (rng.random((count, 1, 8, 10, 6)) * 100).astype(np.float32)
    dataset = Dataset(tmp_path / "cases", "h5")
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


def test_mean_is_incremental_so_the_cohort_is_never_resident(tmp_path: Path) -> None:
    engine, destination, volumes = _run(tmp_path, [], Reduce(operator="Mean", output="avg"), [])
    plan = engine.plan()
    assert plan.incremental is True and plan.resident_regions == 2

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
    engine, _destination, _volumes = _run(tmp_path, [], Reduce(operator="Median", output="t"), [])
    plan = engine.plan()
    assert plan.incremental is False
    assert plan.resident_regions == CASES + 1
    assert "resident region" in plan.describe() and "CASE_000" in plan.describe()


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

    It was running the ``strict`` geometry comparison anyway, and against the FIRST case rather than
    the named one -- so the policy refused exactly the cohorts it exists for, and named the wrong
    case when it did. Its documented contract is equal extents, and that member's geometry.
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
    took the source's channel count -- which sized the regions at a quarter of what a four-case
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


def test_a_non_voxel_local_operator_is_refused(tmp_path: Path) -> None:
    engine, _destination, _volumes = _run(tmp_path, [], Reduce(operator="Median", output="t"), [])
    engine.operator = Median()
    object.__setattr__(engine.operator, "voxel_local", False)
    from konfai.data import case_reduction as reduction_module

    class Spatial(Mean):
        voxel_local = False

    reduction_module.__dict__.setdefault("_Spatial", Spatial)
    with pytest.raises(ReductionError, match="voxel-local"):
        from konfai.data.case_reduction import resolve_operator

        resolve_operator(Reduce(operator="konfai.data.case_reduction:_Spatial", output="t"))


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
    """A halo applied to one region at a time would seam at every boundary -- plausible and wrong.

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
    """A custom operator with a parameter — the extension point, exercised."""

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
