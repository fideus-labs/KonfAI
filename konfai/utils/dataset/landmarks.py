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


def read_landmarks(filename: Path) -> np.ndarray:
    """The points of a Slicer fiducial CSV (``.fcsv``) as an ``[N, 3]`` array in LPS, ITK's physical space.

    The ``# CoordinateSystem`` header decides: ``LPS`` or ``1`` (Slicer >= 4.11) is read as it is, ``RAS``
    or ``0`` (Slicer <= 4.10, or a file saved as RAS) has x and y negated. Read as LPS, a RAS file was a
    reflection of its points: the TRE without a transform (distances survive a reflection) looked right
    and the TRE through one came out 7 to 21 mm off a perfect registration. No header is read as LPS,
    ITK's physical space; :func:`write_landmarks` declares LPS. Voxel indices (``IJK`` or ``2``) cannot be
    placed without their image and are refused. Blank lines are skipped: a trailing one used to fail the
    whole read.
    """
    with open(filename, newline="") as file:
        lines = file.readlines()
    header = {
        key.strip(): value.strip()
        for key, _, value in (line[1:].partition("=") for line in lines if line.startswith("#"))
    }
    rows = csv.reader(line for line in lines if line.strip() and not line.startswith("#"))
    data = np.array([row[1:4] for row in rows], dtype=np.double).reshape(-1, 3)
    system = header.get("CoordinateSystem", "LPS").upper()
    if system in ("RAS", "0"):
        data[:, :2] *= -1
    elif system not in ("LPS", "1"):
        raise DatasetManagerError(
            f"'{filename}' holds its landmarks in the coordinate system '{system}'.",
            "Save the markups in LPS or RAS (world coordinates), not in voxel indices.",
        )
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
