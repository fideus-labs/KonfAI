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

"""Unit tests for displacement fields stored as NGFF RFC-5 OME-Zarr.

The point of the feature is that the store says what it holds: a DVF written through the OME-Zarr
backend comes back as a ``DisplacementFieldTransform``, not as an anonymous 3-channel image that the
reader has to be told about out of band.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from konfai.utils.errors import DatasetManagerError

sitk = pytest.importorskip("SimpleITK")
pytest.importorskip("zarr")
pytest.importorskip("ngff_zarr")

from konfai.utils.dataset import Dataset  # noqa: E402
from konfai.utils.ome_zarr import is_displacement_field, write_ome_zarr  # noqa: E402

SPACING = (1.5, 1.5, 2.0)
ORIGIN = (7.0, -3.0, 10.0)


def _displacement_field_transform() -> "sitk.DisplacementFieldTransform":
    """A field with a non-trivial, anisotropic geometry, so a lost or transposed axis shows up."""
    values = np.arange(4 * 5 * 6 * 3, dtype=np.float64).reshape(4, 5, 6, 3)
    field = sitk.GetImageFromArray(values, isVector=True)
    field.SetSpacing(SPACING)
    field.SetOrigin(ORIGIN)
    return sitk.DisplacementFieldTransform(sitk.Cast(field, sitk.sitkVectorFloat64))


def _store(tmp_path: Path) -> Dataset:
    return Dataset(f"{tmp_path}/dataset/", "omezarr")


def test_written_store_declares_the_displacement_axis(tmp_path: Path) -> None:
    """The component axis is typed ``displacement`` on disk -- the whole point of RFC-5 here."""
    _store(tmp_path).write("case", "DVF", _displacement_field_transform())

    stores = list(tmp_path.rglob("*.ome.zarr"))
    assert len(stores) == 1
    assert is_displacement_field(stores[0])
    metadata = json.dumps(json.loads((stores[0] / "zarr.json").read_text()))
    assert '"type": "displacement"' in metadata


def test_transform_round_trips_through_the_store(tmp_path: Path) -> None:
    """A DVF written as a transform is read back as one, geometry and displacements intact."""
    dataset = _store(tmp_path)
    original = _displacement_field_transform()
    dataset.write("case", "DVF", original)

    restored = dataset.read_transform("case", "DVF")
    assert isinstance(restored, sitk.DisplacementFieldTransform)

    before, after = original.GetDisplacementField(), restored.GetDisplacementField()
    assert after.GetSize() == before.GetSize()
    assert after.GetSpacing() == pytest.approx(SPACING)
    assert after.GetOrigin() == pytest.approx(ORIGIN)
    assert after.GetNumberOfComponentsPerPixel() == 3
    assert np.array_equal(sitk.GetArrayFromImage(after), sitk.GetArrayFromImage(before))


def test_restored_transform_maps_points_identically(tmp_path: Path) -> None:
    """The reconstruction is checked by BEHAVIOUR, not only by the array it was built from.

    A field read back with its component axis transposed, or with one component silently selected,
    still has a plausible shape; only applying it to a point catches that.
    """
    dataset = _store(tmp_path)
    original = _displacement_field_transform()
    dataset.write("case", "DVF", original)
    restored = dataset.read_transform("case", "DVF")

    for point in ((9.0, -1.0, 12.0), (7.5, -2.5, 11.0), (10.0, 0.0, 14.0)):
        assert restored.TransformPoint(point) == pytest.approx(original.TransformPoint(point))


def test_plain_image_is_not_a_displacement_field(tmp_path: Path) -> None:
    """An ordinary multi-channel image must not be mistaken for a field: a 3-channel volume is a
    perfectly normal image, so the store's declaration is what distinguishes them."""
    store = tmp_path / "image.ome.zarr"
    write_ome_zarr(store, np.zeros((3, 4, 5, 6), dtype=np.float32), spacing=SPACING, origin=ORIGIN)

    assert not is_displacement_field(store)

    dataset = _store(tmp_path)
    dataset.write("case", "Volume", sitk.GetImageFromArray(np.zeros((4, 5, 6), dtype=np.float32)))
    _, attributes = dataset.read_data("case", "Volume")
    assert "konfai_displacement_field" not in attributes


def test_is_displacement_field_is_false_for_a_missing_store(tmp_path: Path) -> None:
    """Absent or unreadable is answered, not raised: the question is only ever asked to decide how to
    read something, and 'not a displacement field' is a valid answer for both."""
    assert not is_displacement_field(tmp_path / "does-not-exist.ome.zarr")


def test_non_displacement_transform_is_rejected(tmp_path: Path) -> None:
    """The parametric transforms the other backends serialise have no OME-NGFF form, and failing
    loudly beats writing their parameter vector as if it were a field."""
    with pytest.raises(DatasetManagerError, match="DisplacementFieldTransform"):
        _store(tmp_path).write("case", "Affine", sitk.AffineTransform(3))
