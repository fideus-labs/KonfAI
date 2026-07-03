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

"""Regression tests for the dataset-file audit fixes in ``konfai.utils.dataset``."""

import os
from pathlib import Path

os.environ.setdefault("KONFAI_config_file", "/tmp/konfai-none.yml")
os.environ.setdefault("KONFAI_CONFIG_MODE", "Done")

import numpy as np
import pytest
import torch
from konfai.utils.dataset import Attribute, Dataset
from konfai.utils.errors import DatasetManagerError

sitk = pytest.importorskip("SimpleITK")
h5py = pytest.importorskip("h5py")


def _image_attributes(origin: list[float], spacing: list[float]) -> Attribute:
    attributes = Attribute()
    attributes["Origin"] = np.asarray(origin, dtype=np.float64)
    attributes["Spacing"] = np.asarray(spacing, dtype=np.float64)
    attributes["Direction"] = np.eye(len(origin), dtype=np.float64).reshape(-1)
    return attributes


# --------------------------------------------------------------------------------------
# B13 - Attribute keys containing '_' are stored raw and must be readable/poppable
# --------------------------------------------------------------------------------------


def test_attribute_underscore_key_is_readable_and_consistent() -> None:
    attribute = Attribute()
    attribute["ITK_InputFilterName"] = "GradientAnisotropicDiffusion"

    # __contains__ already reported membership; the getter must now agree with it.
    assert "ITK_InputFilterName" in attribute
    assert attribute["ITK_InputFilterName"] == "GradientAnisotropicDiffusion"
    assert attribute.is_info("ITK_InputFilterName", "GradientAnisotropicDiffusion")


def test_attribute_underscore_key_can_be_popped() -> None:
    attribute = Attribute()
    attribute["ITK_InputFilterName"] = "x"

    assert attribute.pop("ITK_InputFilterName") == "x"
    assert "ITK_InputFilterName" not in attribute


def test_attribute_stacked_lookup_still_wins_over_raw_fallback() -> None:
    """The predictor writes ``<key>_0`` explicitly and reads ``<key>`` back (stack scheme)."""
    attribute = Attribute()
    attribute["number_of_channels_per_model_0"] = torch.tensor([2, 3, 4])

    assert "number_of_channels_per_model" in attribute
    channels = attribute.pop_tensor("number_of_channels_per_model")
    assert torch.equal(channels, torch.tensor([2.0, 3.0, 4.0]))


def test_attribute_repeated_set_returns_latest_version() -> None:
    attribute = Attribute()
    attribute["Origin"] = np.asarray([1.0, 1.0, 1.0])
    attribute["Origin"] = np.asarray([5.0, 5.0, 5.0])

    np.testing.assert_array_equal(attribute.get_np_array("Origin"), np.asarray([5.0, 5.0, 5.0]))


# --------------------------------------------------------------------------------------
# B19 - HDF5 parent directory is created with pathlib (nested paths, OS separators)
# --------------------------------------------------------------------------------------


def test_h5_dataset_creates_nested_parent_directories(tmp_path: Path) -> None:
    dataset = Dataset(tmp_path / "runs" / "exp" / "Volumes", "h5")
    volume = np.arange(1 * 2 * 2, dtype=np.float32).reshape(1, 2, 2)
    dataset.write("CT", "CASE_000", volume, _image_attributes([0.0, 0.0], [1.0, 1.0]))

    assert (tmp_path / "runs" / "exp" / "Volumes.h5").exists()
    data, _ = dataset.read_data("CT", "CASE_000")
    np.testing.assert_array_equal(data, volume)


# --------------------------------------------------------------------------------------
# B23 - a missing sitk entry raises a clear error instead of UnboundLocalError
# --------------------------------------------------------------------------------------


def test_sitk_file_to_data_missing_entry_raises_nameerror(tmp_path: Path) -> None:
    root = tmp_path / "Dataset"
    root.mkdir()
    with Dataset.File(f"{root}/", True, "mha", 0) as file:
        with pytest.raises(NameError, match="not found"):
            file.file_to_data("", "missing_case")


# --------------------------------------------------------------------------------------
# B17 - unknown transform types raise a typed error at write/read (no UnboundLocalError,
#        no silent reuse of the previous type)
# --------------------------------------------------------------------------------------


def test_h5_write_unknown_transform_type_raises(tmp_path: Path) -> None:
    dataset = Dataset(tmp_path / "Transforms", "h5")
    composite = sitk.CompositeTransform([sitk.TranslationTransform(3, (1.0, 2.0, 3.0))])
    with pytest.raises(DatasetManagerError, match="Unsupported transform type"):
        dataset.write("T", "CASE_000", composite, Attribute())


def test_sitk_read_unknown_transform_type_raises(tmp_path: Path) -> None:
    dataset = Dataset(tmp_path / "Dataset", "mha")
    dataset.write("Transf", "CASE_000", sitk.TranslationTransform(3, (1.0, 2.0, 3.0)), Attribute())
    with pytest.raises(DatasetManagerError, match="Unsupported transform type"):
        dataset.read_transform("Transf", "CASE_000")


def test_read_transform_unknown_type_attribute_raises(tmp_path: Path) -> None:
    dataset = Dataset(tmp_path / "Transforms", "h5")
    euler = sitk.Euler3DTransform()
    euler.SetParameters((0.1, 0.2, 0.3, 4.0, 5.0, 6.0))
    dataset.write("T", "CASE_000", euler, Attribute())

    with h5py.File(str(tmp_path / "Transforms.h5"), "r+") as handle:
        handle["T/CASE_000"].attrs["0:Transform_0"] = "MysteryTransform_double_3_3"

    with pytest.raises(DatasetManagerError, match="Unsupported transform type"):
        dataset.read_transform("T", "CASE_000")


def test_supported_transform_types_round_trip(tmp_path: Path) -> None:
    dataset = Dataset(tmp_path / "Dataset", "mha")
    euler = sitk.Euler3DTransform()
    euler.SetParameters((0.1, 0.2, 0.3, 4.0, 5.0, 6.0))
    dataset.write("Transf", "CASE_000", euler, Attribute())

    restored = dataset.read_transform("Transf", "CASE_000")

    assert isinstance(restored, sitk.Euler3DTransform)
    np.testing.assert_allclose(restored.GetParameters(), (0.1, 0.2, 0.3, 4.0, 5.0, 6.0))


# --------------------------------------------------------------------------------------
# B15 - the XML branch returns a (data, attributes) tuple, not a bare lxml element
# --------------------------------------------------------------------------------------


def test_xml_file_to_data_returns_tuple_with_parsed_values(tmp_path: Path) -> None:
    dataset = Dataset(tmp_path / "Dataset", "mha")
    attributes = Attribute()
    attributes["path"] = "level1:level2"
    attributes["foo"] = "bar"
    dataset.write("Node", "CASE_000", np.asarray([1.5, 2.5, 3.5]), attributes)

    result = dataset.read_data("Node", "CASE_000")

    assert isinstance(result, tuple) and len(result) == 2
    data, read_attributes = result
    assert isinstance(data, np.ndarray)
    np.testing.assert_allclose(data, [1.5, 2.5, 3.5])
    assert read_attributes["foo"] == "bar"


# --------------------------------------------------------------------------------------
# B24 - streaming path resolution follows the same precedence as full read
# --------------------------------------------------------------------------------------


def test_resolve_data_path_prefers_special_format_like_full_read(tmp_path: Path) -> None:
    root = tmp_path / "Dataset"
    dataset = Dataset(root, "mha")
    dataset.write(
        "Transf",
        "CASE_000",
        np.arange(1 * 2 * 3 * 4, dtype=np.float32).reshape(1, 2, 3, 4),
        _image_attributes([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
    )
    euler = sitk.Euler3DTransform()
    euler.SetParameters((0.1, 0.2, 0.3, 4.0, 5.0, 6.0))
    dataset.write("Transf", "CASE_000", euler, Attribute())

    # Both Transf.mha and Transf.itk.txt now exist for the same entry.
    sitk_file = Dataset.SitkFile(f"{root}/CASE_000/", True, "mha")
    resolved = sitk_file._resolve_data_path("Transf")

    # read_data (full path) picks the transform; the streaming resolver must agree.
    assert resolved is not None and resolved.endswith(".itk.txt")
    full, _ = dataset.read_data("Transf", "CASE_000")
    assert full.shape == (1, 6)
