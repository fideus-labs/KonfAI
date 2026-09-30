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

"""The Slicer fiducial CSV reader: its points come back in LPS whatever system the file declares."""

from pathlib import Path

import numpy as np
import pytest
from konfai.utils.dataset import read_landmarks, write_landmarks
from konfai.utils.errors import DatasetManagerError

_POINTS = np.array([[1.5, -2.0, 3.0], [-10.0, 20.0, -30.25]])


def _fcsv(path: Path, system: str | None, points: np.ndarray, trailer: str = "") -> Path:
    header = "# Markups fiducial file version = 4.11\n"
    if system is not None:
        header += f"# CoordinateSystem = {system}\n"
    header += "# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID\n"
    rows = "".join(f"{i},{x},{y},{z},0,0,0,1,1,1,0,F-{i},,\n" for i, (x, y, z) in enumerate(points))
    path.write_text(header + rows + trailer)
    return path


@pytest.mark.parametrize("system", ["LPS", "1", None])
def test_lps_points_are_read_as_they_are(tmp_path: Path, system: str | None) -> None:
    np.testing.assert_array_equal(read_landmarks(_fcsv(tmp_path / "p.fcsv", system, _POINTS)), _POINTS)


@pytest.mark.parametrize("system", ["RAS", "0"])
def test_ras_points_come_back_in_lps(tmp_path: Path, system: str) -> None:
    ras = _POINTS * [-1.0, -1.0, 1.0]
    np.testing.assert_array_equal(read_landmarks(_fcsv(tmp_path / "p.fcsv", system, ras)), _POINTS)


def test_blank_lines_are_skipped(tmp_path: Path) -> None:
    np.testing.assert_array_equal(read_landmarks(_fcsv(tmp_path / "p.fcsv", "LPS", _POINTS, "\n\n")), _POINTS)


def test_voxel_indices_are_refused(tmp_path: Path) -> None:
    with pytest.raises(DatasetManagerError, match="IJK"):
        read_landmarks(_fcsv(tmp_path / "p.fcsv", "IJK", _POINTS))


def test_what_write_landmarks_writes_reads_back(tmp_path: Path) -> None:
    write_landmarks(_POINTS, tmp_path / "p.fcsv")
    np.testing.assert_array_equal(read_landmarks(tmp_path / "p.fcsv"), _POINTS)
