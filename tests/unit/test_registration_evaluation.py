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

"""What a registration's evaluation reads off its transform: landmarks pushed through it (``TransformPoints``) and the
Jacobian of a field (``JacobianFolding``, ``JacobianMin``, ``JacobianLogStd``), against SimpleITK and exact maps."""

from pathlib import Path

import numpy as np
import pytest
import torch

sitk = pytest.importorskip("SimpleITK")

from konfai.data.transform import TransformPoints  # noqa: E402
from konfai.metric.measure import JacobianFolding, JacobianLogStd, JacobianMin  # noqa: E402
from konfai.utils.dataset import Attribute, Dataset, image_to_data  # noqa: E402
from konfai.utils.ITK import jacobian_statistics  # noqa: E402


def _oblique_field(values: np.ndarray) -> "sitk.Image":
    field = sitk.GetImageFromArray(values, isVector=True)
    field.SetSpacing((1.5, 2.0, 2.5))
    field.SetOrigin((-4.0, 3.0, 10.0))
    field.SetDirection(sitk.VersorTransform((0.2, 0.3, 0.9), 0.4).GetMatrix())
    return field


def _store(root: Path, form: str, transform: "sitk.Transform", image: "sitk.Image | None" = None) -> Dataset:
    """``transform`` stored as case P000 of group Reg, in ``form``: the dataset that reads it back."""
    case = root / "P000"
    case.mkdir(parents=True)
    if form == "h5":
        sitk.WriteTransform(transform, str(case / "Reg.h5"))
        return Dataset(str(root), "itktransform")
    if form == "omezarr":
        dataset = Dataset(str(root), "omezarr")
        dataset.write("Reg", "P000", transform)
        return dataset
    if form == "itk.txt":
        sitk.WriteTransform(transform, str(case / "Reg.itk.txt"))
        return Dataset(str(root), "mha")
    sitk.WriteImage(image, str(case / "Reg.mha"), useCompression=form == "compressed.mha")
    return Dataset(str(root), "mha")


def _push(dataset: Dataset, points: np.ndarray, invert: bool = False) -> np.ndarray:
    stage = TransformPoints({"Reg": invert})
    stage.set_datasets([dataset])
    return stage("P000", torch.from_numpy(points), Attribute()).numpy()


@pytest.mark.parametrize("form", ["h5", "mha", "compressed.mha", "omezarr"])
def test_points_move_as_through_the_whole_field(tmp_path: Path, form: str) -> None:
    """Points move through a field read around them alone (sliced off it read once, when compressed) exactly as
    through the whole field in SimpleITK: inside, at the faces, between a face and the half voxel past it, and
    outside, on an oblique grid. A field written as a vector image is one too, read as a transform."""
    rng = np.random.default_rng(1)
    field = _oblique_field(rng.normal(0.0, 3.0, (9, 7, 6, 3)))
    whole = sitk.DisplacementFieldTransform(sitk.Image(field))
    extent = np.array(field.GetSize())
    indices = np.concatenate([rng.uniform(-0.5, extent - 0.5, (40, 3)), [[0, 0, 0], extent - 1, [-0.4, 3, 2.7]]])
    indices = np.concatenate([indices, [[-0.6, 1, 1], [2, 3, extent[2] - 0.45], [-3.0, -2.0, 40.0]]])
    points = np.array([field.TransformContinuousIndexToPhysicalPoint(index.tolist()) for index in indices])
    expected = np.array([whole.TransformPoint(point.tolist()) for point in points])

    dataset = _store(tmp_path / "store", form, sitk.DisplacementFieldTransform(sitk.Image(field)), field)

    np.testing.assert_allclose(_push(dataset, points), expected, atol=1e-5)


def test_points_move_through_a_2d_affine_and_keep_their_z(tmp_path: Path) -> None:
    """A 2-D map moves x and y, where a 2-D image's landmarks sit, and leaves z; inverted, it brings them back."""
    affine = sitk.AffineTransform(2)
    affine.SetMatrix((0.98, -0.12, 0.10, 1.03))
    affine.SetTranslation((2.5, -1.5))
    affine.SetCenter((40.0, 30.0))
    points = np.array([[10.0, 20.0, 0.0], [-4.0, 8.0, 0.0], [33.0, 5.0, 0.0]])
    dataset = _store(tmp_path / "store", "itk.txt", affine)

    moved = _push(dataset, points)

    np.testing.assert_allclose(moved[:, :2], [affine.TransformPoint(point[:2].tolist()) for point in points])
    np.testing.assert_array_equal(moved[:, 2], points[:, 2])
    np.testing.assert_allclose(_push(dataset, moved, invert=True), points, atol=1e-9)


def _jacobian(field: "sitk.Image") -> dict[str, float]:
    data, attributes = image_to_data(field)
    tensor = torch.from_numpy(data)[None]
    return {
        name: float(criterion()(tensor, attributes=[[]], output_attributes=[attributes]))
        for name, criterion in (("folded_fraction", JacobianFolding), ("min", JacobianMin), ("sd_log", JacobianLogStd))
    }


@pytest.mark.parametrize("gradient", [np.diag([0.2, -0.1, 0.3]), np.diag([-1.5, 0.0, 0.0])])
def test_jacobian_of_a_linear_map_is_its_determinant_everywhere(gradient: np.ndarray) -> None:
    """u(x) = G x on an oblique, anisotropic grid: det(I + G) at every voxel, folded wherever it is negative.
    Differences are exact on a linear field, so any slip in the spacing or direction shows."""
    grid = _oblique_field(np.zeros((6, 5, 4, 3)))
    points = np.array(
        [[grid.TransformIndexToPhysicalPoint((x, y, z)) for x in range(4)] for z in range(6) for y in range(5)]
    )

    statistics = _jacobian(_oblique_field((points @ gradient.T).reshape(6, 5, 4, 3)))

    determinant = np.linalg.det(np.eye(3) + gradient)
    assert statistics["min"] == pytest.approx(determinant)
    assert statistics["folded_fraction"] == (1.0 if determinant <= 0 else 0.0)
    assert statistics["sd_log"] == pytest.approx(0.0, abs=1e-6)


def test_jacobian_of_a_2d_linear_map_is_its_determinant() -> None:
    """A 2-D field is measured as a plane one voxel thick: the determinant is the plane's."""
    gradient = np.array([[0.3, 0.1], [-0.2, -0.4]])
    grid = sitk.Image([5, 4], sitk.sitkFloat32)
    grid.SetSpacing((1.5, 0.8))
    grid.SetDirection(sitk.Euler2DTransform((0.0, 0.0), 0.5).GetMatrix())
    points = np.array([[grid.TransformIndexToPhysicalPoint((x, y)) for x in range(5)] for y in range(4)])
    field = sitk.GetImageFromArray(points @ gradient.T, isVector=True)
    field.CopyInformation(grid)

    assert _jacobian(field)["min"] == pytest.approx(np.linalg.det(np.eye(2) + gradient))


def test_jacobian_read_in_slabs_is_the_whole_fields(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Slabs of one plane with their halo give the statistics of the field read whole, which the criteria give."""
    field = _oblique_field(np.random.default_rng(2).normal(0.0, 1.5, (7, 6, 5, 3)))
    dataset = _store(tmp_path / "store", "h5", sitk.DisplacementFieldTransform(sitk.Image(field)))
    whole = jacobian_statistics(dataset, "Reg", "P000")
    monkeypatch.setattr("konfai.utils.ITK.FIELD_SLAB_VOXELS", 1)

    assert jacobian_statistics(dataset, "Reg", "P000") == pytest.approx(whole)
    assert _jacobian(field) == pytest.approx(whole)
    assert 0 < whole["folded_fraction"] < 1


def test_a_compressed_field_is_read_by_regions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A compressed MetaImage decoded the whole volume for every region asked of it (300 landmarks, 300 whole
    decodes): it is read through its uncompressed twin, so the landmarks and the Jacobian slabs read regions."""
    field = _oblique_field(np.random.default_rng(4).normal(0.0, 1.0, (7, 6, 5, 3)))
    dataset = _store(tmp_path / "store", "compressed.mha", sitk.Transform(), field)
    reads: list[str] = []
    for name in ("read_data", "read_data_slice"):
        original = getattr(Dataset, name)
        monkeypatch.setattr(
            Dataset, name, lambda self, *a, _read=original, _name=name: reads.append(_name) or _read(self, *a)
        )
    monkeypatch.setattr("konfai.utils.ITK.FIELD_SLAB_VOXELS", 1)
    _push(dataset, np.random.default_rng(5).uniform(-5.0, 15.0, (20, 3)))
    jacobian_statistics(dataset, "Reg", "P000")

    assert reads and set(reads) == {"read_data_slice"}


def test_a_jacobian_of_an_affine_is_refused() -> None:
    """An affine's parameters are not a field: the criterion says what it measures rather than score them."""
    from konfai.utils.errors import MeasureError

    with pytest.raises(MeasureError, match="displacement field"):
        JacobianMin()(torch.zeros(1, 1, 12, dtype=torch.float64), attributes=[[]], output_attributes=[Attribute()])


def test_a_resample_reads_a_field_image_as_a_field(tmp_path: Path) -> None:
    """A displacement field written as a vector image, named under ``transforms:``, is the field SimpleITK applies:
    read as a transform, an image with one component per axis can be nothing else."""
    from konfai import api
    from konfai.data.transform import Resample, Write

    z, y, x = np.mgrid[0:12, 0:14, 0:16].astype(np.float64)
    moving = sitk.GetImageFromArray((np.sin(x / 3) * np.cos(y / 4) * 50 + z).astype(np.float32))
    moving.SetSpacing((1.1, 0.9, 1.4))
    field = sitk.GetImageFromArray(
        np.stack([np.sin(y / 5), 0.8 * np.cos(x / 6), 0.5 * np.sin(z / 4)], -1), isVector=True
    )
    field.CopyInformation(moving)
    case = tmp_path / "Dataset" / "P000"
    case.mkdir(parents=True)
    sitk.WriteImage(moving, str(case / "Moving.mha"))
    sitk.WriteImage(field, str(case / "Reg.mha"))
    stage = Resample(reference="{case}", reference_group="Moving", transforms={"Reg": False}, interpolation="linear")

    api.transform(
        "field_image",
        [f"{tmp_path / 'Dataset'}:mha"],
        {"Moving": {"Moved": [stage, Write(dataset=f"{tmp_path / 'Out'}:mha")]}},
        cpu=1,
        quiet=True,
        overwrite=True,
        transforms_dir=tmp_path / "Workspaces",
    )

    got = sitk.GetArrayFromImage(sitk.ReadImage(str(next((tmp_path / "Out").rglob("Moved.mha")))))
    want = sitk.Resample(moving, moving, sitk.DisplacementFieldTransform(sitk.Image(field)), sitk.sitkLinear, 0.0)
    np.testing.assert_allclose(got, sitk.GetArrayFromImage(want), atol=1e-3)
