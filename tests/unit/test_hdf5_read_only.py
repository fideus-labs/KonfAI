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

"""Regression test: read_data must open HDF5 read-only, not in r+ (write) mode."""

import os
import stat
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("h5py")

from konfai.utils.dataset import Attribute, Dataset  # noqa: E402


def _image_attributes() -> Attribute:
    attributes = Attribute()
    attributes["Origin"] = np.asarray([10.0, 20.0, 30.0])
    attributes["Spacing"] = np.asarray([0.5, 1.5, 2.0])
    attributes["Direction"] = np.eye(3, dtype=np.float64).flatten()
    return attributes


def test_read_data_opens_hdf5_read_only(tmp_path: Path) -> None:
    # read_data used to open HDF5 in r+ (stamping a Date attribute on every read), which mutates
    # the file and breaks concurrent access across DataLoader/DDP processes. On a read-only file an
    # r+ open raises PermissionError, so a successful read here proves the mode is now "r".
    volume = np.arange(1 * 3 * 4 * 5, dtype=np.int16).reshape(1, 3, 4, 5)
    dataset = Dataset(tmp_path / "H5DS", "h5")
    dataset.write("CT", "CASE_001", volume, _image_attributes())

    h5_files = list(tmp_path.rglob("*.h5"))
    assert h5_files, "the write did not create an .h5 file"
    for h5_file in h5_files:
        os.chmod(h5_file, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)

    try:
        full, _ = dataset.read_data("CT", "CASE_001")
        np.testing.assert_array_equal(full, volume)
    finally:
        for h5_file in h5_files:
            os.chmod(h5_file, stat.S_IRUSR | stat.S_IWUSR)
