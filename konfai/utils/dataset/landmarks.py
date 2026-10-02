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


"""Landmark files: the fiducial CSV that 3D Slicer writes and reads."""

import csv
from pathlib import Path

import numpy as np

from konfai.utils.errors import DatasetManagerError

# "0" stays LPS: KonfAI up to 1.5.3 wrote it over LPS points, 3D Slicer before 4.11 over RAS points.
_RAS = {"RAS"}
_LPS = {"0", "1", "LPS"}


def read_landmarks(filename: Path) -> np.ndarray | None:
    """Read Slicer-style fiducial landmarks from disk, in LPS. A file without a
    ``# CoordinateSystem`` line is LPS, as 3D Slicer reads it."""
    coordinate_system = "LPS"
    with open(filename, newline="") as csvfile:
        lines = csvfile.readlines()
    for line in filter(lambda row: row[0] == "#", lines):
        key, _, value = line[1:].partition("=")
        if key.strip() == "CoordinateSystem":
            coordinate_system = value.strip()
    if coordinate_system.upper() not in _RAS | _LPS:
        raise DatasetManagerError(
            f"'{filename}' declares '# CoordinateSystem = {coordinate_system}'.",
            "KonfAI reads landmarks given in RAS (RAS) or LPS (LPS, 1 or 0).",
        )
    rows = list(csv.reader(filter(lambda row: row[0] != "#", lines)))
    data = np.zeros((len(rows), 3), dtype=np.double)
    for i, row in enumerate(rows):
        data[i] = np.array(row[1:4], dtype=np.double)
    if coordinate_system.upper() in _RAS:
        data[:, :2] *= -1
    return data


def write_landmarks(data: np.ndarray, filename: Path) -> None:
    """Write landmarks to the Slicer Markups fiducial CSV-like format."""
    with open(filename, "w") as f:
        f.write(
            "# Markups fiducial file version = 4.6\n# CoordinateSystem = LPS\n#"
            " columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID\n",
        )
        for i in range(data.shape[0]):
            f.write(
                "vtkMRMLMarkupsFiducialNode_"
                + str(i + 1)
                + ","
                + str(data[i, 0])
                + ","
                + str(data[i, 1])
                + ","
                + str(data[i, 2])
                + ",0,0,0,1,1,1,0,F-"
                + str(i + 1)
                + ",,vtkMRMLScalarVolumeNode1\n"
            )
