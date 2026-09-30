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

"""Unit tests for the SimpleITK transform helpers in ``konfai.utils.ITK``."""

import numpy as np
import pytest

sitk = pytest.importorskip("SimpleITK")


def test_resample_transform_applies_displacement_in_physical_space() -> None:
    # A stored-transform Resample must not add the physical (dx, dy, dz) displacement straight onto a (z, y, x)
    # voxel-index grid: that transposes x/z and treats millimetres as voxels. A +6 mm translation along
    # X on a 2 mm-X grid must move content 3 voxels along X (not 6 voxels along Z).
    import torch
    from konfai.data.transform import Resample
    from konfai.utils.dataset import Attribute

    volume = torch.zeros(1, 8, 8, 8, dtype=torch.uint8)
    volume[0, 4, 4, 6] = 1  # (z=4, y=4, x=6)
    attribute = Attribute()
    attribute["Origin"] = np.array([0.0, 0.0, 0.0])
    attribute["Spacing"] = np.array([2.0, 1.0, 1.0])  # (x=2 mm, y=1, z=1)
    attribute["Direction"] = np.eye(3).flatten()

    translation = sitk.TranslationTransform(3, (6.0, 0.0, 0.0))

    class _TransformStore:
        def is_dataset_exist(self, group: str, name: str) -> bool:
            return True

        def read_transform(self, group: str, name: str) -> "sitk.Transform":
            return translation

    transform = Resample(transforms={"reg": False})
    transform.datasets = [_TransformStore()]

    out = transform("case", volume, attribute)
    bright = torch.nonzero(out[0] > 0).tolist()

    assert bright == [[4, 4, 3]]  # moved 6 mm / 2 mm = 3 voxels along X, staying on z=4, y=4


def _field_image(edge: int, seed: int = 0) -> tuple["sitk.Image", np.ndarray]:
    """A displacement field image on a non-trivial grid, and its component-first float32 array."""
    field = (np.random.default_rng(seed).normal(size=(3, edge, edge, edge)) * 4).astype(np.float32)
    image = sitk.GetImageFromArray(np.moveaxis(field, 0, -1), isVector=True)
    image.SetOrigin((7.0, -3.0, 10.0))
    image.SetSpacing((1.5, 1.5, 2.0))
    image.SetDirection((0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0))
    return image, field


def test_decoding_a_displacement_field_copies_it_once() -> None:
    """The stage's float64 component-first values are written straight off ITK's buffer: the same
    values as the copy, the transpose per component and the stack, at one copy's worth of peak."""
    import tracemalloc

    from konfai.utils.ITK import decode_transform_stages

    image, field = _field_image(64)
    transform = sitk.DisplacementFieldTransform(sitk.Cast(image, sitk.sitkVectorFloat64))
    tracemalloc.start()
    (stage,) = decode_transform_stages(transform)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert stage.values.dtype == np.float64 and stage.values.flags.c_contiguous
    np.testing.assert_array_equal(stage.values, field.astype(np.float64))
    np.testing.assert_array_equal(stage.grid.origin_xyz, image.GetOrigin())
    assert peak < 2 * stage.values.nbytes, f"{peak} bytes at peak for {stage.values.nbytes} of values"


def test_a_stored_displacement_entry_decodes_to_the_stage_the_image_route_gives(tmp_path) -> None:
    """Straight from the store's array, on the grid its attributes describe: the same stage, value
    for value, as reading the entry as a transform and decoding that.

    Not byte for byte, because the two routes do not hold it at the same WIDTH and should not. The
    direct read holds a field at the width the store holds it at; the transform route has no store
    to ask -- SimpleITK's field is float64 whatever was written -- so it answers the same numbers
    widened. Equal as maps, and the direct one is the one that fits in half the memory.
    """
    from konfai.utils.dataset import Attribute, Dataset
    from konfai.utils.ITK import decode_transform_stages, read_transform_stages

    image, field = _field_image(6, seed=2)
    attributes = Attribute()
    attributes["Origin"] = np.asarray(image.GetOrigin())
    attributes["Spacing"] = np.asarray(image.GetSpacing())
    attributes["Direction"] = np.asarray(image.GetDirection())
    dataset = Dataset(tmp_path / "out", "itktransform")
    dataset.write("Transform", "P000", field, attributes)

    (direct,) = read_transform_stages(dataset, "Transform", "P000")
    (through_itk,) = decode_transform_stages(dataset.read_transform("Transform", "P000"))

    assert direct.order == through_itk.order == 1
    assert direct.grid.size_zyx == through_itk.grid.size_zyx
    for axis in ("origin_xyz", "spacing_xyz", "direction_xyz"):
        np.testing.assert_array_equal(getattr(direct.grid, axis), getattr(through_itk.grid, axis))
    # Exactly equal, not close: widening a float32 is lossless, so no tolerance is called for.
    np.testing.assert_array_equal(direct.values.astype(np.float64), through_itk.values)
    assert direct.values.dtype == np.float32, "a float32 store is not widened to be held"
    assert through_itk.values.dtype == np.float64, "and a SimpleITK field has no other width"


def test_a_store_serving_transforms_alone_still_decodes() -> None:
    """The smallest thing a stored-transform Resample asks of a dataset is read_transform: a store
    that offers nothing else is decoded through it."""
    from konfai.data.geometry import AffineStage
    from konfai.utils.ITK import read_transform_stages

    translation = sitk.TranslationTransform(3, (6.0, 0.0, 0.0))

    class _TransformStore:
        def read_transform(self, group: str, name: str) -> "sitk.Transform":
            return translation

    (stage,) = read_transform_stages(_TransformStore(), "reg", "case")
    assert isinstance(stage, AffineStage)
    np.testing.assert_array_equal(stage.map.translation, [6.0, 0.0, 0.0])
