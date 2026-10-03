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

"""The Slicer fiducial CSV reader past the coordinate systems test_dataset covers: a blank line holds no point."""

from pathlib import Path

import numpy as np
from konfai.utils.dataset import read_landmarks

_POINTS = np.array([[1.5, -2.0, 3.0], [-10.0, 20.0, -30.25]])


def _fcsv(path: Path, system: str | None, points: np.ndarray, trailer: str = "") -> Path:
    header = "# Markups fiducial file version = 4.11\n"
    if system is not None:
        header += f"# CoordinateSystem = {system}\n"
    header += "# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID\n"
    rows = "".join(f"{i},{x},{y},{z},0,0,0,1,1,1,0,F-{i},,\n" for i, (x, y, z) in enumerate(points))
    path.write_text(header + rows + trailer)
    return path


def test_blank_lines_are_skipped(tmp_path: Path) -> None:
    np.testing.assert_array_equal(read_landmarks(_fcsv(tmp_path / "p.fcsv", "LPS", _POINTS, "\n\n")), _POINTS)
