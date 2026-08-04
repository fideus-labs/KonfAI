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

"""Resampling a case onto a declared reference grid, and what that refuses.

The streamed-equals-whole-volume property is proven for this stage where it is proven for every
other, in ``test_transform_locality_contract.py``; what is proven here is the half that contract
cannot see. A stage can be perfectly self-consistent -- both its paths agreeing on the same wrong
place -- so the arithmetic is checked against SimpleITK, which resamples in physical space and knows
nothing about this file. And a stage that lands on another grid can be wrong in ways no equality
test reaches: by writing the right voxels under the wrong header, or by writing a case that never
met the reference at all.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from konfai.data.case_reduction import CaseReduction
from konfai.data.data_manager import _check_patch_transform_locality
from konfai.data.patching import DatasetManager, DatasetPatch
from konfai.data.transform import LocalityKind, Reduce, Resample, ResampleToReference, Write
from konfai.utils.dataset import Attribute, Dataset
from konfai.utils.errors import ConfigError, TransformError

pytest.importorskip("SimpleITK")
import SimpleITK as sitk

_CASE = "CASE_000"
_SOURCE_SPATIAL = (9, 13, 11)
_REFERENCE_SPATIAL = (7, 10, 15)
# Physical (x, y, z), as a header stores them. Nothing lines up: neither extent, nor spacing, nor
# origin -- and the reference reaches past the case on x, so part of it has no data to read.
_SOURCE_ORIGIN, _SOURCE_SPACING = [-3.0, 5.0, 11.0], [1.5, 1.1, 2.0]
_REFERENCE_ORIGIN, _REFERENCE_SPACING = [-1.25, 4.2, 12.7], [1.9, 1.7, 1.3]
_FILL = -777.0


def _attributes(origin: list[float], spacing: list[float], direction: np.ndarray | None = None) -> Attribute:
    attributes = Attribute()
    attributes["Origin"] = np.asarray(origin)
    attributes["Spacing"] = np.asarray(spacing)
    attributes["Direction"] = (np.eye(3) if direction is None else direction).reshape(-1)
    return attributes


def _volume(shape: tuple[int, ...], seed: int = 0) -> np.ndarray:
    # A step, not a smooth field: interpolating a smooth volume onto a shifted grid gives nearly the
    # right answer even when the shift is wrong, so a smooth fixture would pass a broken map.
    rng = np.random.default_rng(seed)
    return (rng.normal(size=shape) * 100).astype(np.float32)[None]


@pytest.fixture
def dataset(tmp_path: Path) -> Dataset:
    """A case and a reference, on grids that agree about nothing but their direction."""
    dataset = Dataset(tmp_path / "Dataset", "mha")
    dataset.write("Case", _CASE, _volume(_SOURCE_SPATIAL), _attributes(_SOURCE_ORIGIN, _SOURCE_SPACING))
    dataset.write(
        "Reference", _CASE, _volume(_REFERENCE_SPATIAL, 1), _attributes(_REFERENCE_ORIGIN, _REFERENCE_SPACING)
    )
    return dataset


def _stage(dataset: Dataset, **kwargs: object) -> ResampleToReference:
    arguments: dict[str, object] = {"entry": _CASE, "group": "Reference", "fill": _FILL, **kwargs}
    stage = ResampleToReference(**arguments)  # type: ignore[arg-type]
    stage.set_datasets([dataset])
    return stage


def _manager(dataset: Dataset, stage: ResampleToReference, group: str = "Case") -> DatasetManager:
    return DatasetManager(
        index=0,
        group_src=group,
        group_dest=group,
        name=_CASE,
        dataset=dataset,
        patch=DatasetPatch([4, 4, 4]),
        transforms=[stage],
        data_augmentations_list=[],
    )


def _simpleitk(volume: np.ndarray, source: Attribute, reference: Attribute, nearest: bool = False) -> np.ndarray:
    """The same resample, done by SimpleITK — the oracle this stage's arithmetic is checked against."""
    image = sitk.GetImageFromArray(volume[0])
    image.SetOrigin(source.get_np_array("Origin").tolist())
    image.SetSpacing(source.get_np_array("Spacing").tolist())
    image.SetDirection(source.get_np_array("Direction").tolist())
    grid = sitk.Image(*reversed(_REFERENCE_SPATIAL), sitk.sitkFloat32)
    grid.SetOrigin(reference.get_np_array("Origin").tolist())
    grid.SetSpacing(reference.get_np_array("Spacing").tolist())
    grid.SetDirection(reference.get_np_array("Direction").tolist())
    interpolator = sitk.sitkNearestNeighbor if nearest else sitk.sitkLinear
    return sitk.GetArrayFromImage(sitk.Resample(image, grid, sitk.Transform(), interpolator, _FILL))


# --------------------------------------------------------------------- the arithmetic


def test_it_resamples_where_simpleitk_does(dataset: Dataset) -> None:
    """The check the streamed-equals-whole-volume contract cannot make: is the place right at all.

    Both of this stage's paths run one sampler, so they agree with each other by construction --
    including on a grid placed in the wrong spot. SimpleITK resamples in physical space through an
    implementation that shares no line with this one, so agreeing with it is evidence about the
    geometry rather than about the code's self-consistency.
    """
    source = _attributes(_SOURCE_ORIGIN, _SOURCE_SPACING)
    volume = dataset.read_data("Case", _CASE)[0]
    got = _stage(dataset)(_CASE, torch.from_numpy(volume.copy()), Attribute(source)).numpy()[0]
    want = _simpleitk(volume, source, _attributes(_REFERENCE_ORIGIN, _REFERENCE_SPACING))

    assert got.shape == want.shape == _REFERENCE_SPATIAL
    # float32 weights summed in a different order than ITK's nested lerps: a few ulps of the data's
    # own range, not a difference of placement (which would be a whole voxel of gradient).
    np.testing.assert_allclose(got, want, rtol=0, atol=64 * float(np.spacing(np.float32(np.abs(volume).max()))))


def test_the_edge_of_the_data_is_where_simpleitk_puts_it(dataset: Dataset) -> None:
    """The fill boundary, voxel for voxel — the half of a regrid that a tolerance cannot check.

    A map off by one voxel still interpolates real data almost everywhere; where it shows is at the
    rim, in which voxels stop having a source at all. Counting them against ITK's own
    ``[-0.5, n - 0.5)`` is what pins the convention rather than assuming it.
    """
    source = _attributes(_SOURCE_ORIGIN, _SOURCE_SPACING)
    volume = dataset.read_data("Case", _CASE)[0]
    got = _stage(dataset)(_CASE, torch.from_numpy(volume.copy()), Attribute(source)).numpy()[0]
    want = _simpleitk(volume, source, _attributes(_REFERENCE_ORIGIN, _REFERENCE_SPACING))

    assert 0 < int((want == _FILL).sum()) < want.size, "the fixture must have a rim, and not be all rim"
    np.testing.assert_array_equal(got == _FILL, want == _FILL)


def test_a_label_map_takes_the_nearest_voxel_simpleitk_takes(dataset: Dataset, tmp_path: Path) -> None:
    """uint8 resamples by nearest, and nearest here is ITK's round-half-up on the physical index.

    ``F.interpolate``'s nearest is ``floor(o * scale)`` -- a statement about a size ratio, which says
    nothing once the target grid has an origin of its own. A label map interpolated by the wrong rule
    is still a label map, so nothing downstream would report it.
    """
    labels = (np.arange(int(np.prod(_SOURCE_SPATIAL))) % 4).reshape(_SOURCE_SPATIAL).astype(np.uint8)[None]
    dataset.write("Labels", _CASE, labels, _attributes(_SOURCE_ORIGIN, _SOURCE_SPACING))
    source = _attributes(_SOURCE_ORIGIN, _SOURCE_SPACING)

    got = _stage(dataset, fill=0.0)(_CASE, torch.from_numpy(labels.copy()), Attribute(source)).numpy()[0]
    image = sitk.GetImageFromArray(labels[0])
    image.SetOrigin(_SOURCE_ORIGIN)
    image.SetSpacing(_SOURCE_SPACING)
    grid = sitk.Image(*reversed(_REFERENCE_SPATIAL), sitk.sitkUInt8)
    grid.SetOrigin(_REFERENCE_ORIGIN)
    grid.SetSpacing(_REFERENCE_SPACING)
    want = sitk.GetArrayFromImage(sitk.Resample(image, grid, sitk.Transform(), sitk.sitkNearestNeighbor, 0))

    assert got.dtype == np.uint8
    np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("dtype", [np.uint16, np.int16, np.float32])
def test_it_resamples_the_dtypes_a_microscope_and_a_scanner_store(tmp_path: Path, dtype: type) -> None:
    """uint16 is what a light-sheet volume IS, and torch fills only some integer dtypes.

    ``masked_fill`` is unimplemented for uint16, so filling after the cast back to the source dtype
    raises on precisely the volumes this stage was built for -- and only where the reference grid
    reaches past the case, which is to say on the interesting ones.
    """
    dataset = Dataset(tmp_path / f"Dtype{np.dtype(dtype).name}", "mha")
    source = _attributes(_SOURCE_ORIGIN, _SOURCE_SPACING)
    volume = (np.arange(int(np.prod(_SOURCE_SPATIAL))) % 900).reshape(_SOURCE_SPATIAL).astype(dtype)[None]
    dataset.write("Case", _CASE, volume, source)
    dataset.write(
        "Reference", _CASE, _volume(_REFERENCE_SPATIAL, 1), _attributes(_REFERENCE_ORIGIN, _REFERENCE_SPACING)
    )

    got = _stage(dataset, fill=0.0)(_CASE, torch.from_numpy(volume.copy()), Attribute(source))
    assert got.numpy().dtype == dtype
    assert list(got.shape[1:]) == list(_REFERENCE_SPATIAL)
    # The rim the reference reaches past the case takes the fill, in the source's own dtype.
    assert int((got.numpy() == 0).sum()) > 0


def test_an_oblique_pair_resamples_where_simpleitk_does(tmp_path: Path) -> None:
    """A shared non-axis-aligned direction is legal, and the origin shift travels through it.

    The offset is ``D^-1 (O_ref - O_src) / S_src``: drop the ``D^-1`` and an axis-aligned pair still
    lands perfectly, because there ``D`` is the identity. Only an oblique pair can tell.
    """
    direction = np.linalg.qr(np.asarray([[0.936, -0.352, 0.0], [0.352, 0.936, 0.0], [0.0, 0.0, 1.0]]))[0]
    dataset = Dataset(tmp_path / "Oblique", "mha")
    source = _attributes(_SOURCE_ORIGIN, _SOURCE_SPACING, direction)
    reference = _attributes(_REFERENCE_ORIGIN, _REFERENCE_SPACING, direction)
    volume = _volume(_SOURCE_SPATIAL)
    dataset.write("Case", _CASE, volume, source)
    dataset.write("Reference", _CASE, _volume(_REFERENCE_SPATIAL, 1), reference)

    got = _stage(dataset)(_CASE, torch.from_numpy(volume.copy()), Attribute(source)).numpy()[0]
    want = _simpleitk(volume, source, reference)
    np.testing.assert_allclose(got, want, rtol=0, atol=64 * float(np.spacing(np.float32(np.abs(volume).max()))))


# --------------------------------------------------------------------- the header


def test_the_case_lands_on_the_reference_grid(dataset: Dataset) -> None:
    """Extent, spacing, origin and direction all adopted — which is the whole point of the stage.

    Right voxels under the wrong header is the failure this stage exists to prevent, and the one a
    value comparison cannot see: ``Reduce(grid: strict)`` reads exactly these four.
    """
    manager = _manager(dataset, _stage(dataset))
    landed = manager.landed_attributes()

    assert list(manager.spatial_shape) == list(_REFERENCE_SPATIAL)
    np.testing.assert_allclose(landed.get_np_array("Origin"), _REFERENCE_ORIGIN)
    np.testing.assert_allclose(landed.get_np_array("Spacing"), _REFERENCE_SPACING)
    np.testing.assert_allclose(landed.get_np_array("Direction"), np.eye(3).reshape(-1))


def test_a_cohort_on_one_reference_passes_grid_strict(tmp_path: Path) -> None:
    """The end this stage is for: heterogeneous cases fold under ``grid: strict``, which is a real check.

    Without the stage the same cohort disagrees on extent AND on geometry, so the reduction has to be
    told to look away (``shape_only``). Both halves are asserted, because "strict passes" only means
    something if strict would otherwise have refused.
    """
    dataset = Dataset(tmp_path / "Cohort", "mha")
    origins = [[-3.0, 5.0, 11.0], [-2.0, 5.6, 11.4], [-3.4, 4.7, 10.6]]
    shapes = [(9, 13, 11), (8, 12, 12), (10, 13, 10)]
    for index, (origin, shape) in enumerate(zip(origins, shapes, strict=True)):
        dataset.write("Case", f"C{index}", _volume(shape, index), _attributes(origin, _SOURCE_SPACING))
    dataset.write(
        "Reference", "GRID", _volume(_REFERENCE_SPATIAL, 9), _attributes(_REFERENCE_ORIGIN, _REFERENCE_SPACING)
    )

    def managers(with_stage: bool) -> list[DatasetManager]:
        built = []
        for index in range(len(shapes)):
            stages: list[object] = []
            if with_stage:
                stage = ResampleToReference(entry="GRID", group="Reference", fill=_FILL)
                stage.set_datasets([dataset])
                stages.append(stage)
            built.append(
                DatasetManager(
                    index=index,
                    group_src="Case",
                    group_dest="Case",
                    name=f"C{index}",
                    dataset=dataset,
                    patch=None,
                    transforms=[*stages],
                    data_augmentations_list=[],
                )
            )
        return built

    bare = CaseReduction(
        managers(False), Reduce(operator="Median", output="template", grid="strict"), [], dataset, "Case"
    )
    assert bare.check_grid() is not None, "the cohort must disagree, or this proves nothing"

    folded = CaseReduction(
        managers(True), Reduce(operator="Median", output="template", grid="strict"), [], dataset, "Case"
    )
    assert folded.check_grid() is None
    assert [list(manager.spatial_shape) for manager in folded.managers] == [list(_REFERENCE_SPATIAL)] * 3


# --------------------------------------------------------------------- the memory bound


def test_it_never_assembles_the_volume(dataset: Dataset, tmp_path: Path) -> None:
    """The memory bound, asserted as a bound: a regridded case is written without ever being loaded.

    Values alone would pass even if the chain had read everything into RAM first; only forbidding the
    whole-volume read proves that a case larger than memory can go through this stage at all.
    """
    stage = _stage(dataset)
    manager = DatasetManager(
        index=0,
        group_src="Case",
        group_dest="Case",
        name=_CASE,
        dataset=dataset,
        patch=None,
        transforms=[stage, Write(str(tmp_path / "Out"))],
        data_augmentations_list=[],
    )
    assert manager.stream_refusal(0) is None

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("the chain read the whole volume")

    monkeypatched = pytest.MonkeyPatch()
    monkeypatched.setattr(Dataset, "read_data", refuse)
    try:
        assert manager.materialize() is True
    finally:
        monkeypatched.undo()

    written, attributes = Dataset(tmp_path / "Out", "mha").read_data("Case", _CASE)
    assert list(written.shape[1:]) == list(_REFERENCE_SPATIAL)
    np.testing.assert_allclose(attributes.get_np_array("Origin"), _REFERENCE_ORIGIN)
    np.testing.assert_allclose(attributes.get_np_array("Spacing"), _REFERENCE_SPACING)


@pytest.mark.parametrize("offset", [-500.0, 500.0])
def test_a_region_off_the_source_reads_one_voxel(offset: float) -> None:
    """A target region with no source under it pulls one voxel, not the extent it was clamped from.

    A case whose grid meets the reference somewhere still has slabs that do not -- that is the
    ordinary shape of a cohort resampled onto one grid -- and every one of them reads a window it
    will overwrite with fill. Clamping to a legal-but-empty region is what keeps that read at one
    voxel instead of the case's whole cross-section, on every such slab.
    """
    window = Resample.source_window(
        tuple(slice(0, 4) for _ in _SOURCE_SPATIAL),
        [1.0, 1.0, 1.0],
        list(_SOURCE_SPATIAL),
        offsets=[offset] * len(_SOURCE_SPATIAL),
    )
    assert [sl.stop - sl.start for sl in window] == [1, 1, 1]
    assert all(0 <= sl.start < extent for sl, extent in zip(window, _SOURCE_SPATIAL, strict=True))


# --------------------------------------------------------------------- what it refuses


def test_it_declares_regrid_not_rescale(dataset: Dataset) -> None:
    # RESCALE would hand the dispatcher a size ratio and lose the origin entirely, silently.
    assert _stage(dataset).patch_locality(Attribute()).kind is LocalityKind.REGRID


def test_it_is_refused_as_a_patch_transform(dataset: Dataset, monkeypatch: pytest.MonkeyPatch) -> None:
    """Under `patch_transforms:` it would hand back the whole reference extent for every patch.

    Every other region kind is refused there by name, with its own sentence; a kind missing from
    that table would raise a KeyError instead of saying what to do about it.
    """
    monkeypatch.setenv("KONFAI_ROOT", "Trainer")
    with pytest.raises(ConfigError, match="onto another grid"):
        _check_patch_transform_locality(_stage(dataset), "CT", "CT")


def test_a_differing_direction_is_refused(tmp_path: Path) -> None:
    """Axes that do not line up make the map a rotation, which no per-axis window describes."""
    dataset = Dataset(tmp_path / "Rotated", "mha")
    turned = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    source = _attributes(_SOURCE_ORIGIN, _SOURCE_SPACING)
    dataset.write("Case", _CASE, _volume(_SOURCE_SPATIAL), source)
    dataset.write(
        "Reference", _CASE, _volume(_REFERENCE_SPATIAL, 1), _attributes(_REFERENCE_ORIGIN, _REFERENCE_SPACING, turned)
    )

    with pytest.raises(TransformError, match="Direction cosines differ"):
        _stage(dataset).transform_shape("Case", _CASE, list(_SOURCE_SPATIAL), source)


def test_a_case_with_no_geometry_is_refused(dataset: Dataset) -> None:
    """No origin, no physical space to resample in — and a size ratio must not stand in for one."""
    bare = Attribute()
    bare["Spacing"] = np.asarray(_SOURCE_SPACING)
    with pytest.raises(TransformError, match="carries no Origin, Direction"):
        _stage(dataset).transform_shape("Case", _CASE, list(_SOURCE_SPATIAL), bare)


def test_a_case_that_never_meets_the_reference_is_refused(tmp_path: Path) -> None:
    """Its output would be fill from edge to edge, and a median would quietly take it as anatomy.

    This is the refusal the real cohort needed: acquisition stage coordinates are not an anatomical
    frame, so two brains can be metres apart in physical space and look perfectly normal apart.
    """
    dataset = Dataset(tmp_path / "Apart", "mha")
    source = _attributes([1000.0, 1000.0, 1000.0], _SOURCE_SPACING)
    dataset.write("Case", _CASE, _volume(_SOURCE_SPATIAL), source)
    dataset.write(
        "Reference", _CASE, _volume(_REFERENCE_SPATIAL, 1), _attributes(_REFERENCE_ORIGIN, _REFERENCE_SPACING)
    )

    with pytest.raises(TransformError, match="nothing but 'fill'"):
        _stage(dataset).transform_shape("Case", _CASE, list(_SOURCE_SPATIAL), source)


def test_an_unknown_entry_is_refused(dataset: Dataset) -> None:
    stage = ResampleToReference(entry="NOT_THERE", group="Reference")
    stage.set_datasets([dataset])
    with pytest.raises(TransformError, match="cannot find entry 'NOT_THERE'"):
        stage.reference_grid()


def test_an_unnamed_group_is_refused_when_the_store_has_several(dataset: Dataset) -> None:
    """Warp's rule: guessing which group holds the reference is not a guess worth making."""
    stage = ResampleToReference(entry=_CASE)
    stage.set_datasets([dataset])
    with pytest.raises(TransformError, match="cannot tell which group"):
        stage.reference_grid()


def test_an_empty_entry_is_refused_at_construction() -> None:
    with pytest.raises(TransformError, match="needs an 'entry'"):
        ResampleToReference(entry="  ")


# --------------------------------------------------------------------- what it announces


def test_the_plan_is_told_how_much_of_the_grid_the_case_covers(dataset: Dataset) -> None:
    """Partial coverage is legal, common, and worth a line: the rest of the output is fill.

    Nothing else in the plan can say it -- the verdict is STREAM and the byte count is the same
    either way -- so a template that is mostly background would otherwise be a discovery made in a
    viewer.
    """
    note = _stage(dataset).plan_note(
        "Case_out", _CASE, list(_SOURCE_SPATIAL), _attributes(_SOURCE_ORIGIN, _SOURCE_SPACING)
    )
    assert note is not None
    assert "covers" in note and "fill" in note


def test_a_case_that_fills_the_grid_says_nothing(tmp_path: Path) -> None:
    """A note on every line is a note nobody reads: full coverage is the unremarkable case."""
    dataset = Dataset(tmp_path / "Nested", "mha")
    source = _attributes(_SOURCE_ORIGIN, _SOURCE_SPACING)
    dataset.write("Case", _CASE, _volume((20, 20, 20)), source)
    # A reference well inside the case: every voxel of it has data under it.
    dataset.write("Reference", _CASE, _volume((4, 4, 4), 1), _attributes([2.0, 9.0, 15.0], [1.0, 1.0, 1.0]))
    inside = ResampleToReference(entry=_CASE, group="Reference")
    inside.set_datasets([dataset])

    assert inside.plan_note("Case_out", _CASE, [20, 20, 20], source) is None


# --------------------------------------------------------------------- the way back


def test_the_inverse_returns_the_case_to_its_own_grid(dataset: Dataset) -> None:
    """A prediction made on the reference grid comes back to the grid the case was stored on.

    The inverse is the same map solved for the other index, so what it must restore is the extent
    and the header -- not the values, which an interpolation onto a coarser grid has already lost.
    """
    stage = _stage(dataset)
    source = _attributes(_SOURCE_ORIGIN, _SOURCE_SPACING)
    attribute = Attribute(source)
    volume = dataset.read_data("Case", _CASE)[0]

    forward = stage(_CASE, torch.from_numpy(volume.copy()), attribute)
    assert list(forward.shape[1:]) == list(_REFERENCE_SPATIAL)

    back = stage.inverse(_CASE, forward, attribute)
    assert list(back.shape[1:]) == list(_SOURCE_SPATIAL)
    np.testing.assert_allclose(attribute.get_np_array("Origin"), _SOURCE_ORIGIN)
    np.testing.assert_allclose(attribute.get_np_array("Spacing"), _SOURCE_SPACING)


def test_the_inverse_declares_the_whole_volume_and_says_why(dataset: Dataset) -> None:
    """Declared, not discovered: the write-side region remap is not implemented, so it says so."""
    locality = _stage(dataset).inverse_patch_locality(Attribute())
    assert locality.kind is LocalityKind.WHOLE_VOLUME
    assert locality.reason is not None and "reference grid" in locality.reason


# --------------------------------------------------------------------- through a field

# The field's own grid: coarser than the case AND than the reference, with an origin of its own --
# which is the point. A field solved at one resolution moves a volume stored at another, because it
# is defined in world units and read where it is asked.
_FIELD_SPATIAL = (5, 6, 7)
_FIELD_ORIGIN, _FIELD_SPACING = [-4.0, 3.0, 9.0], [4.0, 4.5, 5.0]
_BOUND = 20.0


def _displacement(shape: tuple[int, ...] = _FIELD_SPATIAL) -> np.ndarray:
    """A field whose three components are DIFFERENT functions of DIFFERENT axes.

    Component-first, in physical (x, y, z). Every component varies, and none is a multiple of
    another, so swapping two of them — or reversing the component axis against the array axes —
    moves the anatomy somewhere a comparison against SimpleITK cannot miss.
    """
    z, y, x = np.meshgrid(*[np.arange(n) for n in shape], indexing="ij")
    return np.stack([2.0 + 0.5 * x, -1.5 + 0.3 * y, 0.8 * np.sin(z * 0.9)]).astype(np.float32)


def _high_frequency(shape: tuple[int, ...] = _SOURCE_SPATIAL) -> np.ndarray:
    """A source whose detail a second interpolation would visibly smooth away."""
    z, y, x = np.meshgrid(*[np.arange(n) for n in shape], indexing="ij")
    return (100 * np.sin(z * 1.7) * np.cos(y * 2.1) + 80 * np.sin(x * 2.9)).astype(np.float32)[None]


@pytest.fixture
def warped(tmp_path: Path) -> tuple[Dataset, Dataset, np.ndarray]:
    """A case, a reference grid and a field — three grids that agree about nothing."""
    images = Dataset(tmp_path / "Images", "h5")
    volume = _high_frequency()
    images.write("Case", _CASE, volume, _attributes(_SOURCE_ORIGIN, _SOURCE_SPACING))
    images.write("Reference", _CASE, _volume(_REFERENCE_SPATIAL, 1), _attributes(_REFERENCE_ORIGIN, _REFERENCE_SPACING))
    fields = Dataset(tmp_path / "Fields", "h5")
    fields.write("DVF", _CASE, _displacement(), _attributes(_FIELD_ORIGIN, _FIELD_SPACING))
    return images, fields, volume


def _warping(images: Dataset, fields: Dataset, **kwargs: object) -> ResampleToReference:
    arguments: dict[str, object] = {
        "entry": _CASE,
        "group": "Reference",
        "field": f"{fields.filename}:h5",
        "field_group": "DVF",
        "max_displacement": _BOUND,
        "fill": _FILL,
        **kwargs,
    }
    stage = ResampleToReference(**arguments)  # type: ignore[arg-type]
    stage.set_datasets([images])
    return stage


def _simpleitk_warp(volume: np.ndarray, field: np.ndarray | None = None) -> np.ndarray:
    """``sitk.Resample(image, grid, DisplacementFieldTransform(field))`` — the one-pass authority."""
    image = sitk.GetImageFromArray(volume[0])
    image.SetOrigin(_SOURCE_ORIGIN)
    image.SetSpacing(_SOURCE_SPACING)
    grid = sitk.Image(*reversed(_REFERENCE_SPATIAL), sitk.sitkFloat32)
    grid.SetOrigin(_REFERENCE_ORIGIN)
    grid.SetSpacing(_REFERENCE_SPACING)
    transform: sitk.Transform = sitk.Transform()
    if field is not None:
        # sitk wants a vector image, (z, y, x, component), where KonfAI stores component-first.
        vector = sitk.GetImageFromArray(np.moveaxis(field, 0, -1).astype(np.float64), isVector=True)
        vector.SetOrigin(_FIELD_ORIGIN)
        vector.SetSpacing(_FIELD_SPACING)
        transform = sitk.DisplacementFieldTransform(sitk.Cast(vector, sitk.sitkVectorFloat64))
    return sitk.GetArrayFromImage(sitk.Resample(image, grid, transform, sitk.sitkLinear, _FILL))


def test_it_warps_onto_the_reference_where_simpleitk_does(warped: tuple[Dataset, Dataset, np.ndarray]) -> None:
    """The whole operation, against the one call that defines it.

    Three grids, none agreeing on extent, spacing or origin, and a field with a different function
    per component: a mistake in any of the axis-order conversions lands the anatomy elsewhere, and
    a comparison against SimpleITK is the only thing that would notice.
    """
    images, fields, volume = warped
    got = _warping(images, fields)(_CASE, torch.from_numpy(volume.copy()), _attributes(_SOURCE_ORIGIN, _SOURCE_SPACING))
    want = _simpleitk_warp(volume, _displacement())

    assert list(got.shape[1:]) == list(_REFERENCE_SPATIAL)
    span = float(want.max() - want.min())
    np.testing.assert_allclose(got.numpy()[0], want, rtol=0, atol=64 * float(np.spacing(np.float32(span))))


def test_the_displaced_edge_is_where_simpleitk_puts_it(warped: tuple[Dataset, Dataset, np.ndarray]) -> None:
    """Which voxels have no source is decided by the DISPLACED coordinate, so it tests the whole map.

    A composition that is off by a fraction of a voxel still interpolates real data almost
    everywhere; the rim is where it stops having any, and matching it voxel for voxel is a much
    sharper statement than any tolerance on the values.
    """
    images, fields, volume = warped
    got = _warping(images, fields)(_CASE, torch.from_numpy(volume.copy()), _attributes(_SOURCE_ORIGIN, _SOURCE_SPACING))
    want = _simpleitk_warp(volume, _displacement())

    assert 0 < int((want == _FILL).sum()) < want.size, "the fixture must have a rim, and not be all rim"
    np.testing.assert_array_equal(got.numpy()[0] == _FILL, want == _FILL)


def test_the_field_components_are_not_reversed(tmp_path: Path) -> None:
    """The (x, y, z) / (z, y, x) test, built so that reversing the two orders cannot pass.

    One component is non-zero and the displacement is a whole number of source voxels on that axis,
    so the answer is the source shifted by an exact voxel count along ONE array axis. Put the
    displacement on z instead of x — which is what reversing the component axis does — and the
    result is shifted along the wrong axis, by a different number of voxels, because the spacings
    differ too.
    """
    images = Dataset(tmp_path / "Images", "h5")
    fields = Dataset(tmp_path / "Fields", "h5")
    geometry = _attributes([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    volume = _high_frequency((6, 7, 8))
    images.write("Case", _CASE, volume, geometry)
    # The reference IS the source grid here: the only thing moving anything is the field.
    images.write("Reference", _CASE, volume, geometry)
    # +2 world units on x alone, uniform: with unit spacing that is exactly two voxels on array axis 2.
    uniform = np.zeros((3, 6, 7, 8), dtype=np.float32)
    uniform[0] = 2.0
    fields.write("DVF", _CASE, uniform, geometry)

    stage = ResampleToReference(
        entry=_CASE,
        group="Reference",
        field=f"{fields.filename}:h5",
        field_group="DVF",
        max_displacement=4.0,
        fill=0.0,
    )
    stage.set_datasets([images])
    got = stage(_CASE, torch.from_numpy(volume.copy()), Attribute(geometry)).numpy()[0]

    # output(o) = input(o + 2) along the LAST array axis, which is physical x.
    np.testing.assert_allclose(got[:, :, :-2], volume[0][:, :, 2:], rtol=0, atol=1e-4)
    # And nothing moved along z: a reversed component axis would have shifted this one instead.
    assert not np.allclose(got[:-2, :, :], volume[0][2:, :, :], atol=1e-4)


def test_it_interpolates_once_not_twice(warped: tuple[Dataset, Dataset, np.ndarray]) -> None:
    """The reason this is one stage: two resamples cost detail that the second cannot put back.

    The two-pass baseline is built entirely in SimpleITK — resample onto the grid, then warp on it —
    so what it costs is measured independently of anything here. On a high-frequency source that
    cost is a large fraction of the range, while this stage sits at float rounding from the one-pass
    result. Asserted as an ORDER OF MAGNITUDE, not a number: the point is the gap, not its digits.
    """
    images, fields, volume = warped
    got = _warping(images, fields)(
        _CASE, torch.from_numpy(volume.copy()), _attributes(_SOURCE_ORIGIN, _SOURCE_SPACING)
    ).numpy()[0]
    one_pass = _simpleitk_warp(volume, _displacement())

    intermediate = sitk.GetImageFromArray(_simpleitk_warp(volume, None))
    intermediate.SetOrigin(_REFERENCE_ORIGIN)
    intermediate.SetSpacing(_REFERENCE_SPACING)
    vector = sitk.GetImageFromArray(np.moveaxis(_displacement(), 0, -1).astype(np.float64), isVector=True)
    vector.SetOrigin(_FIELD_ORIGIN)
    vector.SetSpacing(_FIELD_SPACING)
    grid = sitk.Image(*reversed(_REFERENCE_SPATIAL), sitk.sitkFloat32)
    grid.SetOrigin(_REFERENCE_ORIGIN)
    grid.SetSpacing(_REFERENCE_SPACING)
    two_pass = sitk.GetArrayFromImage(
        sitk.Resample(
            intermediate,
            grid,
            sitk.DisplacementFieldTransform(sitk.Cast(vector, sitk.sitkVectorFloat64)),
            sitk.sitkLinear,
            _FILL,
        )
    )

    both = (one_pass != _FILL) & (two_pass != _FILL)
    second_pass_costs = float(np.abs(one_pass[both] - two_pass[both]).max())
    this_stage_costs = float(np.abs(got[both] - one_pass[both]).max())
    assert second_pass_costs > 1.0, "the fixture must be one a second interpolation actually damages"
    assert this_stage_costs < second_pass_costs / 1000.0, (
        f"this stage is {this_stage_costs:.3g} from the one-pass result where a second interpolation"
        f" costs {second_pass_costs:.3g}: it is not resampling once"
    )


def test_the_streamed_warp_equals_the_whole_volume(warped: tuple[Dataset, Dataset, np.ndarray]) -> None:
    """Region by region through a field, against the same chain run whole."""
    images, fields, volume = warped

    def manager() -> DatasetManager:
        return DatasetManager(
            index=0,
            group_src="Case",
            group_dest="Case",
            name=_CASE,
            dataset=images,
            patch=None,
            transforms=[_warping(images, fields)],
            data_augmentations_list=[],
        )

    streaming = manager()
    assert streaming.stream_refusal(0) is None
    whole = _warping(images, fields)(
        _CASE, torch.from_numpy(volume.copy()), _attributes(_SOURCE_ORIGIN, _SOURCE_SPACING)
    ).numpy()
    extent = streaming.spatial_shape
    got = np.empty((1, *extent), dtype=np.float32)
    for start in range(0, extent[0], 3):
        stop = min(start + 3, extent[0])
        got[:, start:stop] = streaming.read_region(
            (slice(start, stop), slice(0, extent[1]), slice(0, extent[2]))
        ).numpy()
    np.testing.assert_array_equal(got, whole)


def test_the_warped_run_never_assembles_the_volume(warped: tuple[Dataset, Dataset, np.ndarray], tmp_path: Path) -> None:
    """The memory bound, through a field: neither the case nor the field is ever read whole."""
    images, fields, _volume = warped
    manager = DatasetManager(
        index=0,
        group_src="Case",
        group_dest="Case",
        name=_CASE,
        dataset=images,
        patch=None,
        transforms=[_warping(images, fields), Write(str(tmp_path / "Warped"))],
        data_augmentations_list=[],
    )
    assert manager.stream_refusal(0) is None

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("the chain read a whole volume")

    monkeypatched = pytest.MonkeyPatch()
    monkeypatched.setattr(Dataset, "read_data", refuse)
    try:
        assert manager.materialize() is True
    finally:
        monkeypatched.undo()
    written = Dataset(tmp_path / "Warped", "mha").read_data("Case", _CASE)[0]
    assert list(written.shape[1:]) == list(_REFERENCE_SPATIAL)


def test_a_field_beyond_its_declared_bound_is_refused(warped: tuple[Dataset, Dataset, np.ndarray]) -> None:
    """The halo is a promise about what was read; a field that breaks it must not sample zeros.

    Sampling past the region that was read gives a dark rim around the moved anatomy and nothing
    else to see, which is the shape of a mistake nobody finds.
    """
    images, fields, volume = warped
    stage = _warping(images, fields, max_displacement=0.5)
    with pytest.raises(TransformError, match="displaces up to"):
        stage(_CASE, torch.from_numpy(volume.copy()), _attributes(_SOURCE_ORIGIN, _SOURCE_SPACING))


def test_a_field_with_no_bound_declares_the_whole_volume(warped: tuple[Dataset, Dataset, np.ndarray]) -> None:
    """Warp's rule, and for the same reason: an unbounded reach cannot size a region."""
    images, fields, _volume = warped
    locality = _warping(images, fields, max_displacement=0.0).patch_locality(
        _attributes(_SOURCE_ORIGIN, _SOURCE_SPACING)
    )
    assert locality.kind is LocalityKind.WHOLE_VOLUME
    assert locality.reason is not None and "max_displacement" in locality.reason
    # And without a field it is a region stage whatever the bound says.
    assert _stage_regrid_kind(images) is LocalityKind.REGRID


def _stage_regrid_kind(images: Dataset) -> LocalityKind:
    stage = ResampleToReference(entry=_CASE, group="Reference")
    stage.set_datasets([images])
    return stage.patch_locality(_attributes(_SOURCE_ORIGIN, _SOURCE_SPACING)).kind


def test_a_field_on_another_direction_is_refused(warped: tuple[Dataset, Dataset, np.ndarray], tmp_path: Path) -> None:
    """A field whose axes do not line up with the case's would need a rotation, not a per-axis map."""
    images, _fields, _volume = warped
    turned = Dataset(tmp_path / "Turned", "h5")
    rotated = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    turned.write("DVF", _CASE, _displacement(), _attributes(_FIELD_ORIGIN, _FIELD_SPACING, rotated))
    stage = _warping(images, turned)
    with pytest.raises(TransformError, match="Direction"):
        stage.transform_shape("Case", _CASE, list(_SOURCE_SPATIAL), _attributes(_SOURCE_ORIGIN, _SOURCE_SPACING))


def test_fields_can_live_beside_the_cases(warped: tuple[Dataset, Dataset, np.ndarray]) -> None:
    """No 'field' path: the fields are a group of the run's own roots, one entry per case.

    That is how a cohort registered in place stores them, and it is the same answer as naming the
    store explicitly — which is what makes it a shorthand rather than a second code path.
    """
    images, fields, volume = warped
    images.write("DVF", _CASE, _displacement(), _attributes(_FIELD_ORIGIN, _FIELD_SPACING))
    beside = ResampleToReference(entry=_CASE, group="Reference", field_group="DVF", max_displacement=_BOUND, fill=_FILL)
    beside.set_datasets([images])

    got = beside(_CASE, torch.from_numpy(volume.copy()), _attributes(_SOURCE_ORIGIN, _SOURCE_SPACING))
    want = _warping(images, fields)(
        _CASE, torch.from_numpy(volume.copy()), _attributes(_SOURCE_ORIGIN, _SOURCE_SPACING)
    )
    np.testing.assert_array_equal(got.numpy(), want.numpy())


def test_a_bound_with_no_field_is_refused() -> None:
    """A max_displacement and nothing to apply it to is a chain that silently does not warp.

    The stage would resample onto the grid perfectly well and the declaration would simply have no
    effect — which is the failure that leaves a plausible volume and no error.
    """
    with pytest.raises(TransformError, match="no field to apply"):
        ResampleToReference(entry=_CASE, group="Reference", max_displacement=1.0)
