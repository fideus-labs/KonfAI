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
from konfai_apps.errors import AppRepositoryError

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


def _extensionless_dicom_series(root: Path, name: str, slices: int = 3) -> Path:
    series = root / name
    series.mkdir(parents=True)
    for index in range(slices):
        # 128-byte preamble + the Part-10 "DICM" magic, no file extension at all.
        (series / f"slice{index}").write_bytes(b"\x00" * 128 + b"DICM" + b"\x00" * 8)
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


def test_directory_volume_suffix_detects_extensionless_dicom_series(tmp_path: Path) -> None:
    series = _extensionless_dicom_series(tmp_path, "ser")
    assert KonfAIApp._directory_volume_suffix(series) == ""
    assert KonfAIApp._list_input_units([series]) == [(series, "")]

    # A plain directory of non-DICOM files must not be misread as a series.
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "vol.mha").write_bytes(b"\x00" * 200)
    assert KonfAIApp._directory_volume_suffix(plain) is None


def test_list_input_units_survives_symlink_loop(tmp_path: Path) -> None:
    cases = tmp_path / "cases"
    cases.mkdir()
    (cases / "a.mha").write_bytes(b"")
    os.symlink(cases, cases / "loop", target_is_directory=True)
    # A self-referential link must not walk until the OS raises ELOOP: each real dir is visited once.
    assert KonfAIApp._list_input_units([cases]) == [(cases / "a.mha", ".mha")]


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
    # OME-Zarr store -> suffixed, in a root of its own: the DICOM series sets ./Dataset's backend.
    volume_zarr = tmp_path / "Dataset_omezarr" / "P000" / "Volume_1.ome.zarr"
    assert volume_dicom.is_symlink() or volume_dicom.is_dir()
    assert volume_zarr.is_symlink() or volume_zarr.is_dir()
    assert Path(os.readlink(volume_dicom)).name == "ser"
    assert Path(os.readlink(volume_zarr)).name == "st.ome.zarr"


def test_list_input_units_reads_a_store_uri_in_place(monkeypatch: pytest.MonkeyPatch) -> None:
    """A URI is no path to walk: a store there is one unit the core reads in place, probed once for its
    existence; a file there would be a whole download, which is the caller's to make."""
    probed: list[str] = []

    def is_dir(path: str) -> bool:
        probed.append(path)
        return path.endswith("fused.ome.zarr")

    monkeypatch.setattr(app_module.uri, "is_dir", is_dir)

    assert KonfAIApp._list_input_units(["s3://bucket/x/fused.ome.zarr"]) == [
        ("s3://bucket/x/fused.ome.zarr", ".ome.zarr")
    ]
    assert probed == ["s3://bucket/x/fused.ome.zarr"]
    with pytest.raises(AppRepositoryError, match="OME-Zarr store"):
        KonfAIApp._list_input_units(["s3://bucket/x/volume.mha"])
    with pytest.raises(AppRepositoryError, match="does not exist"):
        KonfAIApp._list_input_units(["s3://bucket/x/missing.ome.zarr"])


def test_write_inputs_names_a_store_uri_in_the_remote_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A store read in place is named in ``./Dataset/.konfai-apps-remote.yml``, KonfAI's manifest form, under its
    case and group: nothing of it is linked. It lives in ``./Dataset``, so it comes and goes with the staging."""
    from ruamel.yaml import YAML

    local = _zarr_store(tmp_path / "src", "moving.ome.zarr")
    monkeypatch.setattr(app_module.uri, "is_dir", lambda path: True)
    monkeypatch.chdir(tmp_path)

    app = KonfAIApp.__new__(KonfAIApp)
    app._write_inputs_to_dataset([["s3://bucket/x/fused.ome.zarr"], [local]])

    manifest = tmp_path / "Dataset" / ".konfai-apps-remote.yml"
    assert YAML(typ="safe").load(manifest) == {"P000": {"Volume_0": "s3://bucket/x/fused.ome.zarr"}}
    assert not (tmp_path / "Dataset" / "P000").exists()
    assert (tmp_path / "Dataset_omezarr" / "P000" / "Volume_1.ome.zarr").is_symlink()
    assert KonfAIApp._remote_cases() == ["P000"]

    KonfAIApp._clear_dataset()

    assert not manifest.exists()
