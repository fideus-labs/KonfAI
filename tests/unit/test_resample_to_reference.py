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
