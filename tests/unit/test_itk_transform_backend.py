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
from konfai.utils.errors import DatasetManagerError  # noqa: E402

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


def test_a_foreign_text_transform_serves_headers_without_crashing(tmp_path: Path) -> None:
    """A legacy `.tfm` is TEXT (`#Insight Transform File V1.0`), not HDF5: the header fast path
    must not open it with h5py."""
    affine = sitk.AffineTransform(3)
    affine.SetTranslation((2.0, -1.0, 3.0))
    case = tmp_path / "out" / "P000"
    case.mkdir(parents=True)
    sitk.WriteTransform(affine, str(case / "Reg.tfm"))

    dataset = Dataset(tmp_path / "out", "itktransform")
    shape, _attributes_back = dataset.get_infos("Reg", "P000")
    assert len(shape) == 2  # parameter rows, not a field
    assert not dataset.bounded_region_reads("Reg", "P000")
    back = dataset.read_transform("Reg", "P000")
    point = (1.0, 2.0, 3.0)
    assert back.TransformPoint(point) == pytest.approx(affine.TransformPoint(point))


def test_rewriting_a_tfm_entry_lands_on_the_h5_name(tmp_path: Path) -> None:
    """The write is HDF5; renamed onto a resolved `.tfm` it would corrupt that entry, since ITK
    selects its transform IO from the extension. One entry per name survives the rewrite."""
    affine = sitk.AffineTransform(3)
    case = tmp_path / "out" / "P000"
    case.mkdir(parents=True)
    sitk.WriteTransform(affine, str(case / "Reg.tfm"))

    dataset = Dataset(tmp_path / "out", "itktransform")
    dataset.write("Reg", "P000", _field(7), _attributes())

    assert (case / "Reg.h5").is_file()
    assert not (case / "Reg.tfm").exists()


def test_a_stepped_region_read_honours_the_leading_axis_step(tmp_path: Path) -> None:
    """The row span is read whole and the step subsamples it — same values as slicing the whole."""
    field = _field(5)
    dataset = Dataset(tmp_path / "out", "itktransform")
    dataset.write("Transform", "P000", field, _attributes())

    whole, _ = dataset.read_data("Transform", "P000")
    region, _ = dataset.read_data_slice("Transform", "P000", (slice(0, 3), slice(0, 4, 2), slice(2, 5), slice(0, 4)))

    np.testing.assert_array_equal(np.asarray(region), np.asarray(whole)[:, 0:4:2, 2:5, 0:4])


def test_a_region_read_decodes_only_its_rows_and_matches_the_whole(tmp_path: Path) -> None:
    """The parameters are HDF5, so a slab reads the span it maps to — same values as the whole."""
    field = _field(3)
    dataset = Dataset(tmp_path / "out", "itktransform")
    dataset.write("Transform", "P000", field, _attributes())

    assert dataset.bounded_region_reads("Transform", "P000")
    whole, _ = dataset.read_data("Transform", "P000")
    region, _ = dataset.read_data_slice("Transform", "P000", (slice(0, 3), slice(1, 3), slice(2, 5), slice(0, 4)))

    np.testing.assert_array_equal(np.asarray(region), np.asarray(whole)[:, 1:3, 2:5, 0:4])


def test_a_transform_dataset_resolves_as_a_run_input(tmp_path: Path) -> None:
    """What a run writes, a run can read back.

    ``itktransform`` is a backend token, not a suffix — an entry is ``<group>.h5``, and no path is
    ever named ``.itktransform``. Validating a spec against the extensions alone rejected the very
    format the write side had just produced.
    """
    from konfai.data.data_manager import DataPrediction, Group, GroupTransform

    root = tmp_path / "out"
    Dataset(root, "itktransform").write("Transform", "P000", _field(4), _attributes())

    prediction = DataPrediction(
        augmentations=None,
        dataset_filenames=[f"{root}:a:itktransform"],
        groups_src={
            "Transform": Group(groups_dest={"Transform": GroupTransform(transforms=None, patch_transforms=None)})
        },
    )

    assert prediction._resolve_dataset_sources() == {"Transform": [(str(root), True)]}


def test_without_h5py_the_backend_names_the_extra_to_install(tmp_path: Path, monkeypatch) -> None:
    """The parameters are touched region by region through h5py, so it is a requirement, not a
    preference: a peak memory that turns on whether an optional import succeeded is one nobody can
    size. The ``h5`` backend has always demanded it — this says so out loud."""
    monkeypatch.setattr("konfai.utils.dataset.h5py", None)

    with pytest.raises(DatasetManagerError, match="h5py"):
        Dataset(tmp_path / "out", "itktransform").write("Transform", "P000", _field(), _attributes())
