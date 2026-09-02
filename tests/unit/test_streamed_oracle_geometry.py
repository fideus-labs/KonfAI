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

"""The streamed-oracle property over the RANK and GEOMETRY axes (see the family note in
``test_streamed_oracle_decomposition``).

This file varies what the case is stored ON: a second seeded 3-D geometry and a 2-D one, with
anisotropic spacings, oblique and axis-permuting cosines, drawn from a fixed seed list. It also
pins the fixture's own claims (what every row of every contract file assumes of a geometry) and the
one refusal the rank axis owns: a 2-D stored map has no codec tag.
"""

from pathlib import Path

import numpy as np
import pytest
from konfai.data.materialize import Verdict
from konfai.data.transform import Resample
from konfai.utils.dataset import Dataset
from konfai.utils.errors import TransformError
from oracle_support import (
    FIXED_GEOMETRY,
    GEOMETRIES,
    MAIN,
    ROUTES,
    Geometry,
    Route,
    StageCase,
    assert_same,
    identify,
    oracle_matrix,
    sweep,
    whole_volume,
)

pytest.importorskip("SimpleITK")

#: The geometries this file owns: the decomposition sibling runs the same matrix on MAIN.
OTHERS = [name for name in GEOMETRIES if name != MAIN]


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


@pytest.mark.parametrize("entry", oracle_matrix(OTHERS), ids=identify)
def test_a_swept_case_equals_the_whole_volume_case(
    entry: tuple[str, StageCase, Route],
    oracle_cases: dict[str, Dataset],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The matrix: one built-in, one geometry, one decomposition, both routes, the same bytes.

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


def test_a_two_dimensional_stored_map_is_refused_before_any_route_runs(
    oracle_cases: dict[str, Dataset], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stored-map codec holds the 3-D rigid, affine and BSpline kinds; a 2-D map has no tag.

    Refused where the map is READ, which is before either route is chosen, and the message names
    the type it found: a map applied on one route and refused on the other would be the worst of
    both. This is why the two resample-through-a-stored-map cases leave the rank-2 matrix.
    """
    stage = Resample(transforms={"transform": True})
    with pytest.raises(TransformError, match="Euler2DTransform"):
        sweep(oracle_cases["rank2-seed37"], "Intensity", stage, tmp_path / "out", ROUTES[0], monkeypatch)
