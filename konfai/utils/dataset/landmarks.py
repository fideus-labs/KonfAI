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


def read_landmarks(filename: Path) -> tuple[np.ndarray, list[str]]:
    """Read Slicer-style fiducial landmarks from disk, in LPS, with the label of each in the file's order ('' for a
    row that has none). A file without a ``# CoordinateSystem`` line is LPS, as 3D Slicer reads it."""
    coordinate_system = "LPS"
    with open(filename, newline="") as csvfile:
        lines = csvfile.readlines()
    for line in (line for line in lines if line.startswith("#")):
        key, _, value = line[1:].partition("=")
        if key.strip() == "CoordinateSystem":
            coordinate_system = value.strip()
    if coordinate_system.upper() not in _RAS | _LPS:
        raise DatasetManagerError(
            f"'{filename}' declares '# CoordinateSystem = {coordinate_system}'.",
            "KonfAI reads landmarks given in RAS (RAS) or LPS (LPS, 1 or 0).",
        )
    # A blank line (a trailing one, say) holds no point.
    rows = list(csv.reader(line for line in lines if line.strip() and not line.startswith("#")))
    data = np.zeros((len(rows), 3), dtype=np.double)
    for i, row in enumerate(rows):
        if len(row) < 4:
            raise DatasetManagerError(
                f"'{filename}' holds a record of {len(row)} column(s) at row {i + 1}, and a landmark's are its id then"
                " its three coordinates.",
                "Give each row its id, x, y and z.",
            )
        data[i] = np.array(row[1:4], dtype=np.double)
    if coordinate_system.upper() in _RAS:
        data[:, :2] *= -1
    return data, [row[11] if len(row) > 11 else "" for row in rows]


def write_landmarks(data: np.ndarray, filename: Path, labels: list[str] | None = None) -> None:
    """Write landmarks to the Slicer Markups fiducial CSV-like format, each named by its label (``F-<n>`` without)."""
    # Checked before the file is opened: a refused write leaves the landmarks already there.
    if labels is not None and len(labels) != len(data):
        raise ValueError(f"{len(labels)} labels for {len(data)} landmarks: one label per landmark.")
    if labels is not None and any("\n" in label or "\r" in label for label in labels):
        raise ValueError("A landmark's label is one line: a line of its own would be read as a row or a comment.")
    with open(filename, "w") as f:
        f.write(
            "# Markups fiducial file version = 4.6\n# CoordinateSystem = LPS\n#"
            " columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID\n",
        )
        rows = csv.writer(f, lineterminator="\n")
        for i, point in enumerate(data):
            label = f"F-{i + 1}" if labels is None else labels[i]
            node = f"vtkMRMLMarkupsFiducialNode_{i + 1}"
            rows.writerow(
                [node, *(str(v) for v in point[:3]), 0, 0, 0, 1, 1, 1, 0, label, "", "vtkMRMLScalarVolumeNode1"]
            )
