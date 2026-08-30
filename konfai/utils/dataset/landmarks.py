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


def read_landmarks(filename: Path) -> np.ndarray | None:
    """Read Slicer-style fiducial landmarks from disk."""
    data = None
    with open(filename, newline="") as csvfile:
        reader = csv.reader(filter(lambda row: row[0] != "#", csvfile))
        lines = list(reader)
        data = np.zeros((len(list(lines)), 3), dtype=np.double)
        for i, row in enumerate(lines):
            data[i] = np.array(row[1:4], dtype=np.double)
        csvfile.close()
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
        f.close()
