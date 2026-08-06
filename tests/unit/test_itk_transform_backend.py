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

"""The ``:itktransform`` backend: a displacement field written as an ITK transform file, by regions.

``sitk.WriteTransform`` needs the whole field resident in float64; the FILE is three HDF5 datasets
that write by regions. Both write paths must be the same file to ITK's reader — same type, fixed
parameters and parameters, exactly — and an entry must read back through ``Dataset.read_transform``
as the transform it stores."""

from pathlib import Path

import numpy as np
import pytest

sitk = pytest.importorskip("SimpleITK")
pytest.importorskip("h5py")

from konfai.utils.dataset import Attribute, Dataset  # noqa: E402

_ORIGIN, _SPACING = [7.0, -3.0, 10.0], [1.5, 1.5, 2.0]
_DIRECTION = [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]


def _attributes() -> Attribute:
    attributes = Attribute()
    attributes["Origin"] = np.asarray(_ORIGIN)
    attributes["Spacing"] = np.asarray(_SPACING)
    attributes["Direction"] = np.asarray(_DIRECTION)
    return attributes


def _field(seed: int = 0) -> np.ndarray:
    return (np.random.default_rng(seed).normal(size=(3, 4, 5, 6)) * 8).astype(np.float32)


def _oracle(field: np.ndarray) -> "sitk.DisplacementFieldTransform":
    """What ``sitk.WriteTransform`` would have written, held in memory instead."""
    image = sitk.GetImageFromArray(np.moveaxis(field, 0, -1).astype(np.float64), isVector=True)
    image.SetOrigin(_ORIGIN)
    image.SetSpacing(_SPACING)
    image.SetDirection(_DIRECTION)
    return sitk.DisplacementFieldTransform(sitk.Cast(image, sitk.sitkVectorFloat64))


def test_the_whole_write_is_the_file_sitk_would_have_written(tmp_path: Path) -> None:
    field = _field()
    Dataset(tmp_path / "out", "itktransform").write("Transform", "P000", field, _attributes())

    got = sitk.ReadTransform(str(tmp_path / "out" / "P000" / "Transform.h5"))
    want = _oracle(field)
    assert got.GetFixedParameters() == want.GetFixedParameters()
    assert got.GetParameters() == want.GetParameters()


def test_the_streamed_write_is_the_same_file(tmp_path: Path) -> None:
    """Region by region, without the field ever whole in RAM — and the same bytes of parameters."""
    field = _field(1)
    dataset = Dataset(tmp_path / "out", "itktransform")
    stream = dataset.open_data_stream("Transform", "P000", [3, 4, 5, 6], np.dtype("float32"), _attributes())
    assert stream is not None
    with stream:
        for start in range(0, 4, 2):
            stream.write_slice(
                (slice(0, 3), slice(start, start + 2), slice(0, 5), slice(0, 6)), field[:, start : start + 2]
            )

    got = sitk.ReadTransform(str(tmp_path / "out" / "P000" / "Transform.h5"))
    want = _oracle(field)
    assert got.GetFixedParameters() == want.GetFixedParameters()
    assert got.GetParameters() == want.GetParameters()


def test_an_aborted_stream_leaves_no_entry(tmp_path: Path) -> None:
    """A reader must never see a half-written transform under the final name."""
    dataset = Dataset(tmp_path / "out", "itktransform")
    stream = dataset.open_data_stream("Transform", "P000", [3, 4, 5, 6], np.dtype("float32"), _attributes())
    assert stream is not None
    stream.write_slice((slice(0, 3), slice(0, 2), slice(0, 5), slice(0, 6)), _field()[:, 0:2])
    stream.abort()

    assert not (tmp_path / "out" / "P000" / "Transform.h5").exists()


def test_an_entry_reads_back_as_the_transform_it_stores(tmp_path: Path) -> None:
    """``read_transform`` and ``get_infos`` answer from the file: the field, its grid, its marker."""
    field = _field(2)
    dataset = Dataset(tmp_path / "out", "itktransform")
    dataset.write("Transform", "P000", field, _attributes())

    shape, attributes = dataset.get_infos("Transform", "P000")
    assert shape == [3, 4, 5, 6]
    np.testing.assert_allclose(attributes.get_np_array("Origin"), _ORIGIN)

    back = dataset.read_transform("Transform", "P000")
    want = _oracle(field)
    for point in ((8.0, -2.0, 11.0), (7.5, -2.5, 12.0)):
        assert back.TransformPoint(point) == pytest.approx(want.TransformPoint(point))


def test_a_foreign_affine_file_reads_back_too(tmp_path: Path) -> None:
    """The backend serves any ITK transform file on the read side, not only the fields it writes."""
    affine = sitk.AffineTransform(3)
    affine.SetTranslation((2.0, -1.0, 3.0))
    case = tmp_path / "out" / "P000"
    case.mkdir(parents=True)
    sitk.WriteTransform(affine, str(case / "Reg.h5"))

    back = Dataset(tmp_path / "out", "itktransform").read_transform("Reg", "P000")
    point = (1.0, 2.0, 3.0)
    assert back.TransformPoint(point) == pytest.approx(affine.TransformPoint(point))
