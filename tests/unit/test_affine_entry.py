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

"""A linear registration result as a dataset entry. The field's header declares the map (``AFFINE_ATTRIBUTE``), or
a linear transform is handed over, and the ``:itktransform`` and ``:omezarr`` backends write the transform in place
of the field: an ITK ``AffineTransform`` file, an OME-Zarr 0.6 store holding the RFC-5 ``affine`` alone. Either
reads back through ``Dataset.read_transform`` as the same map; an image format refuses it."""

from pathlib import Path

import numpy as np
import pytest

sitk = pytest.importorskip("SimpleITK")
pytest.importorskip("h5py")

from konfai.utils.dataset import AFFINE_ATTRIBUTE, Attribute, Dataset, affine_to_attribute  # noqa: E402
from konfai.utils.errors import DatasetManagerError  # noqa: E402
from oracle_support import geometry  # noqa: E402

_POINTS = [(0.0, 0.0, 0.0), (10.0, -4.0, 7.5), (-3.0, 12.0, 1.0)]


def _rigid() -> sitk.Euler3DTransform:
    rigid = sitk.Euler3DTransform()
    rigid.SetRotation(0.1, -0.2, 0.3)
    rigid.SetTranslation((36.4, -9.8, 1.35))
    rigid.SetCenter((5.0, 6.0, 7.0))
    return rigid


def _header(**extra: str) -> Attribute:
    """A field's header on the fixed grid, declaring the map it samples."""
    attributes = geometry([7.0, -3.0, 10.0], [1.5, 1.5, 2.0], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    attributes[AFFINE_ATTRIBUTE] = affine_to_attribute(_rigid())
    for key, value in extra.items():
        attributes[key] = value
    return attributes


def _same_map(got: sitk.Transform, want: sitk.Transform, points=_POINTS) -> None:
    for point in points:
        np.testing.assert_allclose(got.TransformPoint(point), want.TransformPoint(point), atol=1e-9)


@pytest.mark.parametrize("backend", ["itktransform", "omezarr"])
def test_a_declared_affine_is_written_in_place_of_its_field(tmp_path: Path, backend: str) -> None:
    if backend == "omezarr":
        pytest.importorskip("zarr")
    dataset = Dataset(tmp_path / "out", backend)
    header = _header()
    assert not dataset.can_stream_data(header)  # metadata alone: written whole
    dataset.write("Transform", "P000", np.zeros((3, 4, 5, 6), dtype=np.float32), header)

    _same_map(dataset.read_transform("Transform", "P000"), _rigid())
    shape, _ = dataset.get_infos("Transform", "P000")
    assert shape == [1, 12]  # one leaf, the affine's twelve parameters: no field


def test_the_itk_file_is_an_affine_transform(tmp_path: Path) -> None:
    Dataset(tmp_path / "out", "itktransform").write("Transform", "P000", np.zeros((3, 4, 5, 6), np.float32), _header())
    got = sitk.ReadTransform(str(tmp_path / "out" / "P000" / "Transform.h5"))
    assert got.GetName() == "AffineTransform"
    _same_map(got, _rigid())


def test_the_store_is_an_rfc5_affine_in_the_images_unit(tmp_path: Path) -> None:
    pytest.importorskip("zarr")
    ngff_zarr = pytest.importorskip("ngff_zarr")
    from konfai.utils.ome_zarr import is_affine_store, is_displacement_field, read_ome_zarr_affine

    dataset = Dataset(tmp_path / "out", "omezarr")
    dataset.write(
        "Transform", "P000", np.zeros((3, 4, 5, 6), np.float32), _header(OMEUnits="micrometer micrometer micrometer")
    )
    store = next((tmp_path / "out").rglob("*.ome.zarr"))
    assert is_affine_store(store) and not is_displacement_field(store)
    _same_map(read_ome_zarr_affine(store), _rigid())
    _same_map(dataset.read_transform("Transform", "P000"), _rigid())

    # ngff-zarr reads the store as the transformation it is: over (z, y, x), the translation in micrometres.
    transformation = ngff_zarr.from_ome_zarr(str(store), kind="transformation")
    assert type(transformation).__name__ == "Affine"
    want = affine_to_attribute(_rigid()).reshape(3, 4)
    flip = np.eye(3)[::-1]
    np.testing.assert_allclose(
        np.asarray(transformation.affine),
        np.hstack([flip @ want[:, :3] @ flip.T, 1000.0 * (flip @ want[:, 3]).reshape(3, 1)]),
        atol=1e-9,
    )


def test_a_linear_transform_handed_over_is_written_as_an_affine_store(tmp_path: Path) -> None:
    pytest.importorskip("zarr")
    dataset = Dataset(tmp_path / "out", "omezarr")
    dataset.write("Transform", "P000", _rigid(), Attribute({"OMEUnits": "millimeter millimeter millimeter"}))
    _same_map(dataset.read_transform("Transform", "P000"), _rigid())


def test_a_2d_affine_round_trips_through_a_store(tmp_path: Path) -> None:
    # The 2-D registration pipeline reads its plane's transform through these two alone: KonfAI's own entries are
    # volumes one plane thick.
    pytest.importorskip("zarr")
    from konfai.utils.ome_zarr import ome_zarr_affine_units, read_ome_zarr_affine, write_ome_zarr_affine

    plane = sitk.Euler2DTransform()
    plane.SetAngle(0.3)
    plane.SetTranslation((4.0, -2.5))
    store = tmp_path / "Transform.ome.zarr"
    write_ome_zarr_affine(store, plane, {"OMEUnits": "micrometer nanometer"})
    got = read_ome_zarr_affine(store)
    assert got.GetDimension() == 2 and ome_zarr_affine_units(store) == {"y": "nanometer", "x": "micrometer"}
    _same_map(got, plane, [(0.0, 0.0), (3.0, -7.0)])


def test_a_store_replaced_at_its_path_is_told_apart_again_once_the_cache_is_cleared(tmp_path: Path) -> None:
    # The detection is memoised per path, as the store parses are, under the same contract: whoever puts another store
    # at a path already read clears the cache.
    pytest.importorskip("zarr")
    from konfai.utils.ome_zarr import clear_ome_zarr_cache, is_affine_store, write_ome_zarr, write_ome_zarr_affine

    store = tmp_path / "Transform.ome.zarr"
    write_ome_zarr(store, np.zeros((1, 4, 5, 6), np.float32), spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0))
    assert not is_affine_store(store)
    write_ome_zarr_affine(store, _rigid())
    assert not is_affine_store(store)  # the memo answers
    clear_ome_zarr_cache(store)
    assert is_affine_store(store)


def test_an_image_format_refuses_an_affine(tmp_path: Path) -> None:
    with pytest.raises(DatasetManagerError, match="no image form"):
        Dataset(tmp_path / "out", "mha").write("Transform", "P000", np.zeros((3, 4, 5, 6), np.float32), _header())
