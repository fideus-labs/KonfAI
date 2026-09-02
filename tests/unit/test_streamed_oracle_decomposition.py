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

"""What a sweep writes region by region is what the whole-volume pass writes: the DECOMPOSITION axis.

One property, one axis per file, over the vocabulary ``oracle_support`` declares once (the routes,
the driver, the bounds): this file runs every streamable built-in on the main geometry under the
three decompositions the budget buys (one region, a few, one row each), which is the axis a wrong
region boundary shows on and the one no fixed slab height exercises. The siblings run the same
property over the geometry (``test_streamed_oracle_geometry``), the store's dtype and the N-to-one
fold (``test_streamed_oracle_dtype_reduction``), and the one-to-copies expansion
(``test_streamed_oracle_expansion``).

Vacuity. A fallback that quietly writes the whole volume would satisfy "the bytes agree" and prove
nothing, so every row asserts the ROUTE it took as well, and a decomposed row asserts that it really
was decomposed.
"""

from pathlib import Path

import numpy as np
import pytest
from konfai.data.materialize import Verdict
from konfai.data.transform import Clip
from konfai.utils.dataset import Dataset
from oracle_support import (
    CASE_NAME,
    MAIN,
    ROUTES,
    Route,
    StageCase,
    assert_same,
    identify,
    oracle_matrix,
    sweep,
    whole_volume,
)

pytest.importorskip("SimpleITK")


@pytest.mark.parametrize("entry", oracle_matrix([MAIN]), ids=identify)
def test_a_swept_case_equals_the_whole_volume_case(
    entry: tuple[str, StageCase, Route],
    oracle_cases: dict[str, Dataset],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The matrix: one built-in, one decomposition, both routes, the same bytes.

    The route is asserted before the values, so a stage that quietly stopped streaming fails here
    rather than passing on a whole-volume comparison with itself.
    """
    geometry, case, route = entry
    dataset = oracle_cases[geometry]
    streamed = sweep(dataset, case.group, case.transform, tmp_path / "streamed", route, monkeypatch)
    whole = whole_volume(dataset, case.group, case.transform, tmp_path / "whole")

    assert streamed.verdict is Verdict.STREAM
    # One row per region on a case of 16 rows or more is at least two regions: without this the row
    # would pass on a sweep that never decomposed anything.
    assert streamed.regions >= (2 if route.height == 0.0 else 1)
    assert_same(streamed, whole, case.atol, case.rtol)


def test_the_budget_is_what_decides_how_many_regions_a_sweep_cuts(
    oracle_cases: dict[str, Dataset], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The matrix's decomposition axis, asserted where it can be seen: the same case, the same
    stage, three budgets, three decompositions, and the same bytes out of all three."""
    dataset = oracle_cases[MAIN]
    results = [
        sweep(dataset, "Intensity", Clip(-200.0, 300.0), tmp_path / route.name, route, monkeypatch) for route in ROUTES
    ]
    rows = int(dataset.get_infos("Intensity", CASE_NAME)[0][1])
    counts = [result.regions for result in results]
    assert [result.verdict for result in results] == [Verdict.STREAM] * len(ROUTES)
    assert counts[0] == 1
    assert 1 < counts[1] < counts[2] == rows, f"the budgets gave {counts} regions for {rows} rows"
    for result in results[1:]:
        np.testing.assert_array_equal(result.array, results[0].array)
