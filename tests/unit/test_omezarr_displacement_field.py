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

from konfai.utils.dataset import DISPLACEMENT_FIELD_ATTRIBUTE, Attribute, Dataset  # noqa: E402
from konfai.utils.ome_zarr import (  # noqa: E402
    append_ome_zarr_levels,
    clear_ome_zarr_cache,
    is_displacement_field,
    write_ome_zarr,
)

SPACING = (1.5, 1.5, 2.0)
ORIGIN = (7.0, -3.0, 10.0)
# A 90 deg in-plane rotation: an identity matrix would let a dropped Direction pass unnoticed.
DIRECTION = (0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)


def _displacement_field_transform() -> "sitk.DisplacementFieldTransform":
    """A field with a non-trivial, anisotropic geometry, so a lost or transposed axis shows up."""
    values = np.arange(4 * 5 * 6 * 3, dtype=np.float64).reshape(4, 5, 6, 3)
    field = sitk.GetImageFromArray(values, isVector=True)
    field.SetSpacing(SPACING)
    field.SetOrigin(ORIGIN)
    field.SetDirection(DIRECTION)
    return sitk.DisplacementFieldTransform(sitk.Cast(field, sitk.sitkVectorFloat64))


def _store(tmp_path: Path) -> Dataset:
    return Dataset(f"{tmp_path}/dataset/", "omezarr")


def test_written_store_declares_the_displacement_axis(tmp_path: Path) -> None:
    """The component axis is typed ``displacement`` on disk: the whole point of RFC-5 here."""
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
    assert after.GetDirection() == pytest.approx(DIRECTION)
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


def test_a_field_streamed_region_by_region_is_still_a_field(tmp_path: Path) -> None:
    """The marker on the attributes declares it, so no transform has to be built to say so.

    A producer that emits blocks: the predictor, and anything writing a field too large to hold --
    has no transform to hand over, and wrapping one purely to be described correctly would mean
    assembling in memory the very volume streaming exists to avoid. So the declaration travels with
    the attributes instead, and the streamed store ends up saying exactly what the whole-volume one
    says.
    """
    values = np.arange(3 * 4 * 5 * 6, dtype=np.float32).reshape(3, 4, 5, 6) / 100.0
    attributes = Attribute()
    attributes["Spacing"] = np.asarray(SPACING)
    attributes["Origin"] = np.asarray(ORIGIN)
    attributes["Direction"] = np.asarray(DIRECTION)
    attributes[DISPLACEMENT_FIELD_ATTRIBUTE] = "true"

    dataset = _store(tmp_path)
    stream = dataset.open_data_stream("DVF", "case", list(values.shape), values.dtype, attributes)
    assert stream is not None
    with stream:
        for z in range(0, values.shape[1], 2):
            stream.write_slice(
                (slice(0, 3), slice(z, z + 2), slice(0, values.shape[2]), slice(0, values.shape[3])),
                values[:, z : z + 2],
            )

    clear_ome_zarr_cache()
    assert is_displacement_field(tmp_path / "dataset" / "case" / "DVF.ome.zarr")
    assert isinstance(dataset.read_transform("DVF", "case"), sitk.DisplacementFieldTransform)


def _axis_aligned_field_transform() -> "sitk.DisplacementFieldTransform":
    """The conformant case: a field whose grid carries no rotation."""
    values = np.arange(4 * 5 * 6 * 3, dtype=np.float64).reshape(4, 5, 6, 3)
    field = sitk.GetImageFromArray(values, isVector=True)
    field.SetSpacing(SPACING)
    field.SetOrigin(ORIGIN)
    return sitk.DisplacementFieldTransform(sitk.Cast(field, sitk.sitkVectorFloat64))


def test_an_axis_aligned_field_is_an_applicable_rfc5_transformation(tmp_path: Path) -> None:
    """The store is not merely labelled, a spec reader can APPLY it.

    ngff-zarr reads the ``displacements`` entry the writer declares, rebuilds a native ITK
    transform from the store alone -- no KonfAI in the loop -- and that transform maps points
    exactly as the SimpleITK original. This is the whole meaning of conformance: the mistakes it
    rules out (component order, value frame) are silent and produce plausible registrations.
    """
    itk = pytest.importorskip("itk")
    import dataclasses

    import ngff_zarr

    original = _axis_aligned_field_transform()
    _store(tmp_path).write("case", "DVF", original)
    store = next(tmp_path.rglob("*.ome.zarr"))

    multiscales = ngff_zarr.from_ngff_zarr(str(store))
    entries = multiscales.metadata.coordinateTransformations or []
    assert [entry.type for entry in entries] == ["displacements"]

    wasm = ngff_zarr.ngff_displacement_field_to_itk_transform(entries[0], multiscales, ["z", "y", "x"])
    native = itk.transform_from_dict(dataclasses.asdict(wasm[0]))
    native = native[0] if isinstance(native, (list, tuple)) else native
    for point in ((9.0, -1.0, 12.0), (7.5, -2.5, 11.0), (10.0, 0.0, 14.0)):
        assert tuple(native.TransformPoint(point)) == pytest.approx(original.TransformPoint(point))


def test_a_conformant_store_holds_spec_ordered_components(tmp_path: Path) -> None:
    """On disk the components follow the output axes (dz, dy, dx); through KonfAI's reader they
    come back in ITK's (dx, dy, dz). Both at once, or one side is silently wrong: the store for
    every spec reader, the flip for every KonfAI consumer."""
    import zarr
    from konfai.data import read_ome_zarr_data_slice

    values = np.arange(3 * 4 * 5 * 6, dtype=np.float32).reshape(3, 4, 5, 6) / 100.0
    attributes = Attribute()
    attributes["Spacing"] = np.asarray(SPACING)
    attributes["Origin"] = np.asarray(ORIGIN)
    attributes[DISPLACEMENT_FIELD_ATTRIBUTE] = "true"

    dataset = _store(tmp_path)
    stream = dataset.open_data_stream("DVF", "case", list(values.shape), values.dtype, attributes)
    assert stream is not None
    with stream:
        for z in range(0, values.shape[1], 2):
            stream.write_slice(
                (slice(0, 3), slice(z, z + 2), slice(0, values.shape[2]), slice(0, values.shape[3])),
                values[:, z : z + 2],
            )
    clear_ome_zarr_cache()
    store = next(tmp_path.rglob("*.ome.zarr"))

    stored = zarr.open_group(str(store), mode="r")["scale0/image"][...]
    np.testing.assert_array_equal(stored, values[::-1], err_msg="the store must hold (dz, dy, dx)")

    back, _ = read_ome_zarr_data_slice(store, tuple(slice(None) for _ in values.shape))
    np.testing.assert_array_equal(back, values, err_msg="the reader must hand back (dx, dy, dz)")
    one, _ = read_ome_zarr_data_slice(store, (slice(0, 1), *[slice(None)] * 3))
    np.testing.assert_array_equal(one[0], values[0], err_msg="a partial component read must remap too")


def test_an_oriented_field_keeps_the_label_only_layout(tmp_path: Path) -> None:
    """A rotated grid has no RFC-5 spelling (scale and translation cannot carry a direction), so
    no ``displacements`` entry that would promise a reader an application it cannot make -- but the
    LAYOUT is the same one every field shares: components in the spec's order, marked as such, the
    Direction in the sidecar. One convention on disk, whatever the grid."""
    import ngff_zarr
    import zarr

    original = _displacement_field_transform()  # carries the 90-degree DIRECTION
    _store(tmp_path).write("case", "DVF", original)
    store = next(tmp_path.rglob("*.ome.zarr"))

    multiscales = ngff_zarr.from_ngff_zarr(str(store))
    assert not (multiscales.metadata.coordinateTransformations or [])
    assert is_displacement_field(store)
    stored = zarr.open_group(str(store), mode="r")["scale0/image"][...]
    itk_order = np.moveaxis(sitk.GetArrayFromImage(original.GetDisplacementField()), -1, 0)
    np.testing.assert_array_equal(stored, itk_order[::-1], err_msg="oriented fields share the spec order")


def test_a_pre_19_store_is_refused_by_name(tmp_path: Path) -> None:
    """A store with the typed axis but neither the ``displacements`` entry nor the component-order
    marker is every store this backend wrote before 1.9, and its components are ITK-ordered. Read
    under either convention it is a guess -- a plausible field with dx and dz possibly exchanged --
    so it is refused with the layout named, not read with one."""
    import dask.array
    import ngff_zarr
    from konfai.data import read_ome_zarr_data_slice

    values = np.arange(3 * 4 * 5 * 6, dtype=np.float32).reshape(3, 4, 5, 6)
    image = ngff_zarr.to_ngff_image(
        dask.array.from_array(values, chunks=values.shape),
        dims=["c", "z", "y", "x"],
        scale={"c": 1.0, "z": SPACING[2], "y": SPACING[1], "x": SPACING[0]},
        translation={"c": 0.0, "z": ORIGIN[2], "y": ORIGIN[1], "x": ORIGIN[0]},
    )
    image.axes_types = {"c": "displacement"}
    multiscales = ngff_zarr.to_multiscales(image, scale_factors=[], cache=False)
    store = tmp_path / "legacy.ome.zarr"
    ngff_zarr.to_ngff_zarr(str(store), multiscales, overwrite=True, version="0.6")
    clear_ome_zarr_cache()

    with pytest.raises(DatasetManagerError, match=r"KonfAI < 1\.9"):
        read_ome_zarr_data_slice(store, tuple(slice(None) for _ in values.shape))


@pytest.mark.parametrize("oriented", [False, True], ids=["axis-aligned", "oriented"])
def test_appending_levels_keeps_the_field_a_field(tmp_path: Path, oriented: bool) -> None:
    """A pyramid grafted onto a field store rewrites the multiscales document, which is where the
    typed component axis and the ``displacements`` entry live. Both must survive, or the store
    comes back an ordinary 3-channel image and nothing says so."""
    import ngff_zarr

    transform = _displacement_field_transform() if oriented else _axis_aligned_field_transform()
    _store(tmp_path).write("case", "DVF", transform)
    store = next(tmp_path.rglob("*.ome.zarr"))
    before = ngff_zarr.from_ngff_zarr(str(store)).metadata

    append_ome_zarr_levels(store, [2])
    clear_ome_zarr_cache()

    after = ngff_zarr.from_ngff_zarr(str(store)).metadata
    assert [dataset.path for dataset in after.datasets] == ["scale0/image", "scale1/image"]
    assert is_displacement_field(store)
    # The entry an axis-aligned field declares keeps naming level 0, which the append never
    # rewrites; an oriented one has none to keep.
    assert (after.coordinateTransformations or []) == (before.coordinateTransformations or [])
    assert {system.name for system in after.coordinateSystems} == {system.name for system in before.coordinateSystems}
