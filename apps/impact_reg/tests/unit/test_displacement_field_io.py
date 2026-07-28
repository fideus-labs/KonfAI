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

"""The orchestrator's displacement-field I/O, for a field written either as an ITK image or as an
NGFF RFC-5 OME-Zarr store.

A preset may emit its DVF in either form, and every step that follows -- locating it, copying it
beside the results, averaging an ensemble, exporting ``Transform.h5`` -- used to assume ``DVF.mha``.
"""

from pathlib import Path

import numpy as np
import pytest

sitk = pytest.importorskip("SimpleITK")
pytest.importorskip("zarr")
pytest.importorskip("ngff_zarr")

from impact_reg_konfai.impact_reg import (  # noqa: E402
    _copy_output,
    _displacement_transform,
    _find_output,
    _write_displacement_field,
)
from konfai.utils.ome_zarr import _zarr_v3_available, is_displacement_field  # noqa: E402

# The OME-Zarr side of these tests writes an RFC-5 field, a zarr v3 store that zarr 2.x
# (Python 3.10) cannot write.
pytestmark = pytest.mark.skipif(
    not _zarr_v3_available(),
    reason="NGFF RFC-5 displacement fields need a zarr v3 store (zarr>=3, Python>=3.11)",
)

SPACING = (1.5, 1.5, 2.0)
ORIGIN = (7.0, -3.0, 10.0)
# A 90 deg in-plane rotation: non-identity, so a Direction dropped on the store round-trip changes
# where the field is sampled and the mapped-point assertion below catches it.
DIRECTION = (0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)


def _field(seed: int = 0) -> "sitk.Image":
    values = np.random.default_rng(seed).normal(size=(4, 5, 6, 3)) * 8
    field = sitk.GetImageFromArray(values, isVector=True)
    field.SetSpacing(SPACING)
    field.SetOrigin(ORIGIN)
    field.SetDirection(DIRECTION)
    return sitk.Cast(field, sitk.sitkVectorFloat64)


def _write_store(dest: Path, field: "sitk.Image") -> Path:
    _write_displacement_field(field, dest)
    return dest


@pytest.mark.parametrize("suffix", [".mha", ".ome.zarr"])
def test_find_output_locates_either_form(tmp_path: Path, suffix: str) -> None:
    """Discovery is by name, not by filename: a store is a directory whose stem is 'DVF.ome'."""
    produced = tmp_path / "P000"
    produced.mkdir()
    _write_displacement_field(_field(), produced / f"DVF{suffix}")

    assert _find_output(tmp_path, "DVF").name == f"DVF{suffix}"


def test_find_output_reports_the_name_it_looked_for(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="DVF"):
        _find_output(tmp_path, "DVF")


@pytest.mark.parametrize("suffix", [".mha", ".ome.zarr"])
def test_copy_output_keeps_the_produced_form(tmp_path: Path, suffix: str) -> None:
    """A store is copied as a store, an image as an image -- the destination keeps the suffix rather
    than renaming everything to a form only one of them has."""
    source = tmp_path / "src"
    source.mkdir()
    _write_displacement_field(_field(), source / f"DVF{suffix}")
    destination = tmp_path / "out"
    destination.mkdir()

    copied = _copy_output(source / f"DVF{suffix}", destination, "DVF")

    assert copied.name == f"DVF{suffix}"
    assert copied.is_dir() == (suffix == ".ome.zarr")


def test_copy_output_replaces_an_existing_store(tmp_path: Path) -> None:
    """Re-running a case overwrites its outputs; a directory cannot be overwritten by copytree."""
    source, destination = tmp_path / "src", tmp_path / "out"
    source.mkdir()
    destination.mkdir()
    _write_displacement_field(_field(), source / "DVF.ome.zarr")

    _copy_output(source / "DVF.ome.zarr", destination, "DVF")
    again = _copy_output(source / "DVF.ome.zarr", destination, "DVF")

    assert again.is_dir()
    assert is_displacement_field(again)


def test_store_written_by_the_orchestrator_is_a_declared_field(tmp_path: Path) -> None:
    """An averaged ensemble field is written in the members' form, and stays a declared field."""
    store = _write_store(tmp_path / "DVF.ome.zarr", _field())
    assert is_displacement_field(store)


@pytest.mark.parametrize("suffix", [".mha", ".ome.zarr"])
def test_transform_reads_back_identically_from_either_form(tmp_path: Path, suffix: str) -> None:
    """What Transform.h5 is built from: the same transform whichever form the field was written in.

    Checked on a mapped point, not only on the array -- a field read with its component axis mishandled
    keeps a plausible shape, and only applying it exposes that.
    """
    original = _field()
    _write_displacement_field(original, tmp_path / f"DVF{suffix}")

    restored = _displacement_transform(tmp_path / f"DVF{suffix}")

    assert isinstance(restored, sitk.DisplacementFieldTransform)
    reference = sitk.DisplacementFieldTransform(sitk.Image(original))
    for point in ((9.0, -1.0, 12.0), (7.5, -2.5, 11.0)):
        assert restored.TransformPoint(point) == pytest.approx(reference.TransformPoint(point))
