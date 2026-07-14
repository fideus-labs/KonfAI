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

"""Directory-volume staging: DICOM series directories and OME-Zarr stores as single volumes."""

import os
from pathlib import Path

import konfai_apps.app as app_module
import pytest

KonfAIApp = app_module.KonfAIApp


def _zarr_store(root: Path, name: str, marker: str = ".zgroup") -> Path:
    store = root / name
    store.mkdir(parents=True)
    (store / marker).write_text("{}", encoding="utf-8")
    return store


def _dicom_series(root: Path, name: str, slices: int = 3) -> Path:
    series = root / name
    series.mkdir(parents=True)
    for index in range(slices):
        (series / f"slice{index}.dcm").write_bytes(b"")
    return series


def test_directory_volume_suffix_detects_stores_and_series(tmp_path: Path) -> None:
    assert KonfAIApp._directory_volume_suffix(_zarr_store(tmp_path, "a.ome.zarr")) == ".ome.zarr"
    assert KonfAIApp._directory_volume_suffix(_zarr_store(tmp_path, "b.zarr")) == ".zarr"
    assert KonfAIApp._directory_volume_suffix(_zarr_store(tmp_path, "bare", marker=".zattrs")) == ".ome.zarr"
    assert KonfAIApp._directory_volume_suffix(_dicom_series(tmp_path, "series")) == ""

    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "vol.mha").write_bytes(b"")
    assert KonfAIApp._directory_volume_suffix(plain) is None

    single = tmp_path / "x.mha"
    single.write_bytes(b"")
    assert KonfAIApp._directory_volume_suffix(single) is None


def test_list_input_units_treats_store_and_series_as_one_unit(tmp_path: Path) -> None:
    store = _zarr_store(tmp_path, "vol.zarr")
    assert KonfAIApp._list_input_units([store]) == [(store, ".zarr")]

    series = _dicom_series(tmp_path, "ser")
    assert KonfAIApp._list_input_units([series]) == [(series, "")]


def test_list_input_units_expands_a_container_of_series(tmp_path: Path) -> None:
    root = tmp_path / "cases"
    root.mkdir()
    _dicom_series(root, "caseB")
    _dicom_series(root, "caseA")
    units = KonfAIApp._list_input_units([root])
    assert [unit.name for unit, _ in units] == ["caseA", "caseB"]  # sorted, paired-friendly
    assert all(suffix == "" for _, suffix in units)


def test_list_input_units_plain_files_unchanged(tmp_path: Path) -> None:
    directory = tmp_path / "d"
    directory.mkdir()
    (directory / "a.mha").write_bytes(b"")
    (directory / "b.mha").write_bytes(b"")
    units = KonfAIApp._list_input_units([directory])
    assert [unit.name for unit, _ in units] == ["a.mha", "b.mha"]
    assert all(suffix == ".mha" for _, suffix in units)


def test_detect_group_format(tmp_path: Path) -> None:
    mha = tmp_path / "Dataset" / "P000"
    mha.mkdir(parents=True)
    (mha / "Volume_0.mha").write_bytes(b"")
    assert KonfAIApp._detect_group_format(tmp_path / "Dataset", "Volume_0") == "mha"

    dicom = tmp_path / "DatasetDicom" / "P000" / "Volume_0"
    dicom.mkdir(parents=True)
    (dicom / "s0.dcm").write_bytes(b"")
    assert KonfAIApp._detect_group_format(tmp_path / "DatasetDicom", "Volume_0") == "dicom"

    zarr = tmp_path / "DatasetZarr" / "P000" / "Volume_0.ome.zarr"
    zarr.mkdir(parents=True)
    (zarr / ".zgroup").write_text("{}", encoding="utf-8")
    assert KonfAIApp._detect_group_format(tmp_path / "DatasetZarr", "Volume_0") == "omezarr"

    assert KonfAIApp._detect_group_format(tmp_path / "missing", "Volume_0") == "mha"


def test_write_inputs_stages_series_as_dir_and_store_with_suffix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "src"
    series = _dicom_series(source, "ser")
    store = _zarr_store(source, "st.ome.zarr")
    monkeypatch.chdir(tmp_path)

    app = KonfAIApp.__new__(KonfAIApp)
    app._write_inputs_to_dataset([[series], [store]])

    volume_dicom = tmp_path / "Dataset" / "P000" / "Volume_0"  # DICOM series -> bare directory
    volume_zarr = tmp_path / "Dataset" / "P000" / "Volume_1.ome.zarr"  # OME-Zarr store -> suffixed
    assert volume_dicom.is_symlink() or volume_dicom.is_dir()
    assert volume_zarr.is_symlink() or volume_zarr.is_dir()
    assert Path(os.readlink(volume_dicom)).name == "ser"
    assert Path(os.readlink(volume_zarr)).name == "st.ome.zarr"
