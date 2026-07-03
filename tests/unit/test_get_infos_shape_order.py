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

"""Regression test: get_infos returns numpy channel-first order for every rank.

Patch planning strips the channel from get_infos' shape and feeds the spatial shape to
transform_shape and the patch reader; the actual pixel reads (image_to_data / _file_to_image_slice)
are numpy-order [C, (T), (Z), Y, X]. The pre-fix code reversed sitk GetSize() only when len == 3, so
2-D and 4-D images kept sitk (x, y, ...) order and were transposed against their own pixel data.
"""

from pathlib import Path

import numpy as np
import SimpleITK as sitk
from konfai.utils.dataset import Dataset, get_infos, image_to_data


def test_get_infos_2d_matches_pixel_data(tmp_path: Path) -> None:
    # Non-square 2-D: sitk GetSize() = (x=10, y=4); numpy pixel data is (y=4, x=10).
    path = tmp_path / "img2d.nii.gz"
    sitk.WriteImage(sitk.GetImageFromArray(np.zeros((4, 10), dtype=np.float32)), str(path))

    size, _ = get_infos(path)
    data, _ = image_to_data(sitk.ReadImage(str(path)))

    assert list(size) == list(data.shape)  # [1, 4, 10], not [1, 10, 4]


def test_get_infos_4d_matches_pixel_data(tmp_path: Path) -> None:
    # Genuine 4-D scalar: sitk GetSize() = (5, 4, 3, 2); numpy pixel data is (2, 3, 4, 5).
    path = tmp_path / "img4d.nii.gz"
    sitk.WriteImage(sitk.Image([5, 4, 3, 2], sitk.sitkFloat32), str(path))

    size, _ = get_infos(path)
    data = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))

    assert list(size) == [1, *data.shape]  # [1, 2, 3, 4, 5]


def test_get_infos_3d_unchanged(tmp_path: Path) -> None:
    # Regression guard: the already-correct 3-D path must stay reversed.
    path = tmp_path / "img3d.nii.gz"
    sitk.WriteImage(sitk.GetImageFromArray(np.zeros((6, 4, 10), dtype=np.float32)), str(path))

    size, _ = get_infos(path)
    data, _ = image_to_data(sitk.ReadImage(str(path)))

    assert list(size) == list(data.shape) == [1, 6, 4, 10]


def test_sitkfile_get_infos_2d_matches_read_data(tmp_path: Path) -> None:
    # Same defect in SitkFile.get_infos, reached through the public Dataset API.
    ds_dir = str(tmp_path / "ds") + "/"
    Path(ds_dir).mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(sitk.GetImageFromArray(np.zeros((4, 10), dtype=np.float32)), ds_dir + "case0.mha")

    file = Dataset.SitkFile(ds_dir, read=True, file_format="mha")
    size, _ = file.get_infos("", "case0")
    data, _ = file.file_to_data("", "case0")

    assert list(size) == list(data.shape)  # [1, 4, 10]
