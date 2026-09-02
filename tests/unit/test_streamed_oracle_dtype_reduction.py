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

"""The streamed-oracle property over the DTYPE axis and the N-cases-to-one REDUCTION (see the
family note in ``test_streamed_oracle_decomposition``).

The dtype axis varies what a store actually holds, refusals included: a dtype is not a detail of
the storage, it decides whether the chain rounds, and where. The reduction is one of the two
cardinality changes the workflow owns (``Reduce``, N cases folded into one); the other, one case
expanded into copies, lives in ``test_streamed_oracle_expansion``.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from konfai.data.case_reduction import CaseReduction
from konfai.data.materialize import CaseMaterializer, Verdict
from konfai.data.transform import Clip, Crop, Flip, Gradient, Mask, Reduce, Resample, Save
from konfai.utils.dataset import Dataset
from konfai.utils.errors import TransformError
from oracle_support import (
    CASE_NAME,
    GEOMETRIES,
    LSB_ATOL,
    MAIN,
    ROUTES,
    Geometry,
    Route,
    StageCase,
    assert_same,
    attributes,
    build_case,
    manager,
    sweep,
    volumes,
    whole_volume,
)

pytest.importorskip("SimpleITK")


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
    streamed = sweep(dataset, case.group, case.transform, tmp_path / "streamed", route, monkeypatch)
    whole = whole_volume(dataset, case.group, case.transform, tmp_path / "whole")

    assert streamed.verdict is Verdict.STREAM
    # The store's dtype survives a remap. Gradient hands back differences, which an integer store
    # cannot hold: it widens those to float32 and leaves a floating dtype as it found it.
    expected = np.dtype(dtype)
    if isinstance(case.transform, Gradient) and not np.issubdtype(expected, np.floating):
        expected = np.dtype(np.float32)
    assert streamed.array.dtype == expected
    assert_same(streamed, whole, case.atol, case.rtol)


def test_a_dtype_torch_has_no_kernel_for_refuses_on_both_routes(tmp_path: Path) -> None:
    """``uint16`` is a store's dtype, not a chain's: torch implements no comparison for it.

    The sweep gives up on it (a warning naming ``TensorCast``) and the whole-volume fallback then
    raises with the same remedy: a refusal on one route and a result on the other would make the
    decomposition decide whether a case runs at all.
    """
    dataset = build_case(tmp_path / "case", GEOMETRIES[MAIN], np.dtype(np.uint16))
    case_manager = manager(dataset, [Clip("min", "max"), Save(f"{tmp_path / 'out'}:h5")], group="Intensity")
    with pytest.warns(UserWarning, match="TensorCast"), pytest.raises(TransformError, match="TensorCast"):
        CaseMaterializer(case_manager).materialize()


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
        managers=[manager(dataset, [], name=f"CASE_{index:03d}") for index in range(count)],
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
