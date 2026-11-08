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

"""A registration's transform in its native form: a rigid then B-spline composite written in place of its field, the
field written wherever a mean or another format needs it, and sampled back on the fixed grid to be measured."""

from pathlib import Path

import numpy as np
import pytest
import torch

sitk = pytest.importorskip("SimpleITK")

from konfai.data.augmentation import Flip  # noqa: E402
from konfai.data.transform import LocalityKind, TransformToField  # noqa: E402
from konfai.predictor.output import OutputDataset  # noqa: E402
from konfai.utils.dataset import (  # noqa: E402
    DENSE_TRANSFORMS_ENV,
    DISPLACEMENT_FIELD_ATTRIBUTE,
    TRANSFORM_FILE_ATTRIBUTE,
    Attribute,
    Dataset,
    image_to_data,
)
from konfai.utils.errors import TransformError  # noqa: E402
from oracle_support import geometry  # noqa: E402


def _grid() -> sitk.Image:
    grid = sitk.Image([20, 18, 16], sitk.sitkFloat32)
    grid.SetSpacing((1.5, 1.2, 2.0))
    grid.SetOrigin((-10.0, 5.0, 3.0))
    return grid


def _composite(grid: sitk.Image) -> sitk.Transform:
    rigid = sitk.Euler3DTransform((5.0, 15.0, 20.0), 0.05, -0.03, 0.08, (2.0, -1.5, 3.0))
    bspline = sitk.BSplineTransformInitializer(grid, [4, 4, 3], 3)
    bspline.SetParameters(np.random.default_rng(0).normal(0, 1.5, len(bspline.GetParameters())).tolist())
    return sitk.CompositeTransform([rigid, bspline])


def _field_with_its_transform(tmp_path: Path) -> tuple[sitk.Transform, np.ndarray, Attribute, Path]:
    grid = _grid()
    transform = _composite(grid)
    field = sitk.TransformToDisplacementField(
        transform, sitk.sitkVectorFloat64, grid.GetSize(), grid.GetOrigin(), grid.GetSpacing(), grid.GetDirection()
    )
    data, attributes = image_to_data(field)
    native = tmp_path / "native.h5"
    sitk.WriteTransform(transform, str(native))
    attributes[DISPLACEMENT_FIELD_ATTRIBUTE] = "true"
    attributes[TRANSFORM_FILE_ATTRIBUTE] = str(native)
    return transform, data, attributes, native


def test_a_transform_file_is_written_in_place_of_its_field(tmp_path: Path) -> None:
    """An ':itktransform' output writes the transform the header names, kilobytes where its field weighs more, and
    the file it was handed goes."""
    transform, data, attributes, native = _field_with_its_transform(tmp_path)
    Dataset(str(tmp_path / "out"), "itktransform").write("Transform", "P000", data, attributes)

    written = tmp_path / "out" / "P000" / "Transform.h5"
    assert written.stat().st_size < data.nbytes / 4 and not native.exists()
    back = sitk.ReadTransform(str(written))
    for point in ([0.0, 10.0, 12.0], [12.0, 20.0, 30.0]):
        np.testing.assert_allclose(back.TransformPoint(point), transform.TransformPoint(point), atol=1e-9)


def test_another_format_writes_the_field(tmp_path: Path) -> None:
    """Only a transform file holds a B-spline: a MetaImage output takes the field, and the transform file goes."""
    _, data, attributes, native = _field_with_its_transform(tmp_path)
    Dataset(str(tmp_path / "out"), "mha").write("Transform", "P000", data, attributes)

    back, header = Dataset(str(tmp_path / "out"), "mha").read_data("Transform", "P000")
    np.testing.assert_allclose(back, data, atol=1e-6)
    assert TRANSFORM_FILE_ATTRIBUTE not in header and not native.exists()


def _output(copies: list[Attribute]) -> OutputDataset:
    """An output dataset holding the headers of a case's copies, its write queue a sink: the header the final write
    is handed is the first copy's, changed in place."""
    output = OutputDataset.__new__(OutputDataset)
    output.attributes = {0: {index: {0: attributes} for index, attributes in enumerate(copies)}}
    output._submit_write = lambda operation: None  # type: ignore[method-assign]
    return output


@pytest.mark.parametrize(("copies", "dense"), [(1, False), (2, True)])
def test_a_mean_of_copies_writes_the_field(tmp_path: Path, copies: int, dense: bool) -> None:
    """One copy keeps its native transform; a mean of TTA copies is no copy's transform, but the field."""
    headers = []
    for index in range(copies):
        (tmp_path / str(index)).mkdir()
        headers.append(_field_with_its_transform(tmp_path / str(index))[2])
    _output(headers).write_prediction(0, "P000", torch.zeros(1))

    assert (TRANSFORM_FILE_ATTRIBUTE in headers[0]) is not dense
    assert all((tmp_path / str(index) / "native.h5").exists() is not dense for index in range(copies))


@pytest.mark.parametrize("streamed", [False, True])
def test_a_mean_of_copies_writes_the_field_on_either_route(tmp_path: Path, monkeypatch, drive_tta, streamed) -> None:
    """A TTA case finalised whole or slab by slab (the streamed route, buffered: a native form cannot stream) writes
    the same thing: the field, its copies' transform file gone."""
    native = tmp_path / "native.h5"
    sitk.WriteTransform(_composite(_grid()), str(native))
    header = geometry()
    header[DISPLACEMENT_FIELD_ATTRIBUTE] = "true"
    header[TRANSFORM_FILE_ATTRIBUTE] = str(native)

    written, whole_volume = drive_tta(
        tmp_path / "out",
        monkeypatch,
        augmentation=Flip(f_prob=[0, 1, 1]),
        streamed=streamed,
        file_format="itktransform",
        channels=3,
        attribute=header,
    )

    assert whole_volume is not streamed
    assert list(written.shape) == [3, 6, 4, 3] and not native.exists()


def test_a_caller_that_averages_asks_for_the_field(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An ensemble's member or a tile runs alone, but its caller averages or composes it: it writes the field."""
    monkeypatch.setenv(DENSE_TRANSFORMS_ENV, "1")
    header = _field_with_its_transform(tmp_path)[2]

    _output([header]).write_prediction(0, "P000", torch.zeros(1))

    assert TRANSFORM_FILE_ATTRIBUTE not in header


def test_a_native_transform_is_measured_on_the_fixed_grid(tmp_path: Path) -> None:
    """TransformToField samples a transform with no grid of its own on the fixed image's, as SimpleITK does, and
    declares it needs the whole volume; a field its header marks one passes as it is, voxel by voxel, so it still
    streams; without the fixed image, a native transform is refused."""
    grid = _grid()
    transform = _composite(grid)
    (tmp_path / "transforms" / "P000").mkdir(parents=True)
    sitk.WriteTransform(transform, str(tmp_path / "transforms" / "P000" / "Volume_0.h5"))
    (tmp_path / "images" / "P000").mkdir(parents=True)
    sitk.WriteImage(grid, str(tmp_path / "images" / "P000" / "Reference_0.mha"))
    transforms, images = Dataset(str(tmp_path / "transforms"), "itktransform"), Dataset(str(tmp_path / "images"), "mha")

    stage = TransformToField()
    stage.set_datasets([transforms, images])
    assert stage.patch_locality(Attribute()).kind is LocalityKind.WHOLE_VOLUME  # the config-time probe
    assert stage.patch_locality(transforms.get_infos("Volume_0", "P000")[1]).kind is LocalityKind.WHOLE_VOLUME
    # An image's header alone does not say it is a field; a stored field transform's does.
    assert stage.patch_locality(images.get_infos("Reference_0", "P000")[1]).kind is LocalityKind.WHOLE_VOLUME
    zero = sitk.Image(grid.GetSize(), sitk.sitkVectorFloat64, 3)
    zero.CopyInformation(grid)
    sitk.WriteTransform(sitk.DisplacementFieldTransform(zero), str(tmp_path / "transforms" / "P000" / "Field.h5"))
    assert stage.patch_locality(transforms.get_infos("Field", "P000")[1]).kind is LocalityKind.POINTWISE
    attributes = Attribute()
    assert stage.transform_shape("Volume_0", "P000", [1512], attributes) == [16, 18, 20]
    field = stage("P000", torch.zeros(2, 1512), attributes).numpy()
    want = sitk.TransformToDisplacementField(
        transform, sitk.sitkVectorFloat32, grid.GetSize(), grid.GetOrigin(), grid.GetSpacing(), grid.GetDirection()
    )
    np.testing.assert_allclose(field, np.moveaxis(sitk.GetArrayFromImage(want), -1, 0), atol=1e-5)
    np.testing.assert_allclose(attributes.get_np_array("Spacing"), grid.GetSpacing())
    assert DISPLACEMENT_FIELD_ATTRIBUTE in attributes

    alone = TransformToField()
    alone.set_datasets([transforms])
    with pytest.raises(TransformError, match="no grid of its own"):
        alone("P000", torch.zeros(2, 1512), Attribute())

    sitk.WriteImage(want, str(tmp_path / "images" / "P000" / "Field.mha"))
    passing = TransformToField(group="Field")
    passing.set_datasets([images])
    tensor = torch.from_numpy(np.moveaxis(sitk.GetArrayFromImage(want), -1, 0).copy())
    assert passing("P000", tensor, Attribute()) is tensor
