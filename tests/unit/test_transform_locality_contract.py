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

"""A stage of a case's chain must behave as if it had seen the whole volume.

Patch streaming is an optimisation, so it must be semantically invisible: a stage DECLARES how its
output depends on its input (``patch_locality``) and the dispatcher reads only the source region that
declaration allows. For every stage declaring a streamable kind, the streamed patch must therefore equal
the whole-volume result cut on the same grid -- proven here for EVERY built-in transform AND every
built-in augmentation, on a real on-disk dataset, over the full patch grid.

Both are ENUMERATED from their module and selected by their own declaration, so a newly declared
streamable stage is covered the day it lands. The case tables never decide coverage: they supply
constructor arguments, and a fixture group, where a stage's defaults are not a meaningful streaming
case, and -- for an augmentation, whose declaration is about a draw rather than a config -- the kind
that draw must produce.

The declaration is handed the case's metadata, so it is asked here exactly as the dispatcher asks it --
per group, from what that group is stored with. The rules that argument comes with (read-only, total)
are checked too, and a declaration that only the image can make is exercised end to end.
"""

import inspect
from dataclasses import dataclass

import numpy as np
import pytest
import torch
from konfai.data import augmentation as augmentation_module
from konfai.data import transform as transform_module
from konfai.data.augmentation import DataAugmentation, DataAugmentationsList
from konfai.data.augmentation import Flip as FlipAugmentation
from konfai.data.patching import DatasetManager, DatasetPatch
from konfai.data.transform import (
    Argmax,
    Canonical,
    Clip,
    Crop,
    Dilate,
    FlatLabel,
    Flip,
    Gradient,
    HistogramMatching,
    InferenceStack,
    LocalityKind,
    MergeLabels,
    OneHot,
    PatchLocality,
    Percentage,
    ResampleToResolution,
    ResampleToShape,
    ResampleTransform,
    Save,
    SegmentationDisagreement,
    SelectLabel,
    Softmax,
    Squeeze,
    StandardDeviation,
    Sum,
    Transform,
    Variance,
)
from konfai.utils.dataset import Attribute, Dataset

pytest.importorskip("SimpleITK")

_CASE_NAME = "CASE_000"
_SPATIAL = (9, 10, 11)
_PATCH_SIZE = [4, 4, 4]
_SPACING = [1.5, 1.5, 2.0]

# No extent is a multiple of the patch size, so the last patch of every axis is a border patch the read
# plan has to pad: the grid is 3x3x3 and 19 of its 27 patches touch a border.
_PEAK = 450.0

# Byte-identical is the norm: a streamable transform reads exactly the voxels the whole-volume pass
# reads and does the same arithmetic on them. Two kinds legitimately round differently:
#
# - the seeded statistic: streaming seeds Mean/Std from `read_data_statistics` (a numpy pass over the
#   stored volume) while the whole-volume path recomputes them with torch over the loaded tensor -- the
#   same values summed in a different order, so a standardized (unit-scale) voxel may land a few
#   float32 ulps away. Data-dependent: this fixture happens to agree exactly, a smooth field showed
#   1.5e-8 (0.13 ulp), so the bound is stated rather than observed.
_STAT_ATOL = 8 * float(np.finfo(np.float32).eps)
# - the streamed resample (trilinear only): it gathers the same source samples, but computes the
#   interpolation weights from coordinates expressed in the read sub-region's frame rather than the
#   whole volume's. Both round to float32, so a weight lands ~ulp(coordinate) off and the interpolated
#   voxel lands `neighbour gap * ulp(coordinate)` off -- the deviation scales with the local GRADIENT,
#   not with the voxel's own magnitude. The fixture's gap is its 2*_PEAK bone/air step, which puts the
#   bound at ulps of _PEAK; 64 of them is ~8x the measured max (2.3e-4, i.e. 7.5 ulp) and stays far
#   below one part per million of the range. Nearest (uint8) uses no weights and stays exact.
_RESCALE_ATOL = 64 * float(np.spacing(np.float32(_PEAK)))
# An integer volume truncates the interpolation, so a sub-ulp disagreement that straddles an integer
# boundary becomes a whole least-significant bit. 1 LSB is the tightest bound that can hold: the
# alternative would be bit-exact agreement between two different float coordinate frames.
_LSB_ATOL = 1.0


@dataclass(frozen=True)
class _Case:
    """One representative configuration of a transform, and the fixture group it consumes."""

    transform: Transform
    group: str = "Intensity"
    atol: float = 0.0


# Only the transforms whose defaults are not a meaningful streaming case: a channel reduction needs a
# multi-model input, a label op needs labels, and a required constructor argument has no default at all.
_CASES: dict[str, list[_Case]] = {
    "Argmax": [_Case(Argmax(0), group="Ensemble")],
    # The defaults reorient the axis-aligned group, which is a mirroring. A PERMUTING direction is the
    # other exact remap, and the only one that transposes extents: its patches are cut on a grid the
    # source volume does not have, so its source region is a permuted slice tuple rather than the
    # target's own. Both must stream, and a sagittal or coronal acquisition is the second.
    "Canonical": [_Case(Canonical()), _Case(Canonical(), group="Permuting")],
    "Clip": [
        _Case(Clip(-200.0, 300.0)),  # fixed bounds: POINTWISE (the default range does not clip)
        _Case(Clip("min", "max")),  # data-dependent bounds: GLOBAL_STAT
    ],
    # Only a case whose box is already stored is a translation; without one there is nothing to
    # declare but the read that would find it.
    "Crop": [_Case(Crop(), group="Boxed")],
    "Dilate": [_Case(Dilate(2), group="Labels")],
    "FlatLabel": [_Case(FlatLabel([1, 3]), group="Labels")],
    "Gradient": [_Case(Gradient()), _Case(Gradient(per_dim=True))],
    "HistogramMatching": [_Case(HistogramMatching("Intensity"))],
    "InferenceStack": [_Case(InferenceStack("Dataset", "model"))],
    "MergeLabels": [_Case(MergeLabels(), group="Ensemble")],
    "OneHot": [_Case(OneHot(4), group="Labels")],
    "Percentage": [_Case(Percentage(100.0))],
    # The defaults ([1, 1, 1] mm / [100, 256, 256]) would be a no-op resample and a 6.5M-voxel upsample.
    "ResampleToResolution": [
        _Case(ResampleToResolution([2.0, 1.0, 3.0]), atol=_RESCALE_ATOL),
        _Case(ResampleToResolution([2.0, 1.0, 3.0]), group="Int16", atol=_LSB_ATOL),
        # uint8 resamples by nearest neighbour: no interpolation weights, so no rounding to disagree on.
        _Case(ResampleToResolution([2.0, 1.0, 3.0]), group="Labels"),
    ],
    "ResampleToShape": [_Case(ResampleToShape([12, 8, 14]), atol=_RESCALE_ATOL)],
    "ResampleTransform": [_Case(ResampleTransform({"transform": True}))],
    "Save": [_Case(Save("Dataset"))],
    "SegmentationDisagreement": [_Case(SegmentationDisagreement(), group="Ensemble")],
    "SelectLabel": [_Case(SelectLabel(["(1,2)", "(3,1)"]), group="Labels")],
    "Softmax": [_Case(Softmax(0), group="Ensemble")],
    "Squeeze": [_Case(Squeeze(0))],
    "Standardize": [_Case(transform_module.Standardize(), atol=_STAT_ATOL)],
    "StandardDeviation": [_Case(StandardDeviation(), group="Ensemble")],
    "Sum": [_Case(Sum(0), group="Ensemble")],
    "Variance": [_Case(Variance(), group="Ensemble")],
}


# An axis-aligned direction, one that permutes physical x and z -- what a sagittal acquisition carries
# -- and a rotated one; all orthonormal (a stored volume has no other kind). The first two make a
# reorientation an exact index remap, which is what an image-dependent declaration keys on; the third
# does not. The permuting one transposes the extents it swaps, so it also moves the patch grid.
_AXIS_ALIGNED = np.eye(len(_SPATIAL))
# A foreground box, as [start, after] margins per spatial axis: (9, 10, 11) is cropped to (6, 6, 6),
# which still carries a patch grid to disagree over.
_BOX = np.asarray([[2, 1], [1, 3], [3, 2]])
_PERMUTING = np.asarray([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
_OBLIQUE = np.asarray(
    [
        [np.cos(np.deg2rad(20.0)), -np.sin(np.deg2rad(20.0)), 0.0],
        [np.sin(np.deg2rad(20.0)), np.cos(np.deg2rad(20.0)), 0.0],
        [0.0, 0.0, 1.0],
    ]
)


def _volumes() -> dict[str, np.ndarray]:
    """One volume per input kind the built-in transforms consume, all on the same geometry."""
    rng = np.random.default_rng(0)
    axes = np.meshgrid(*[np.linspace(-1, 1, n) for n in _SPATIAL], indexing="ij")
    # A CT-like phantom rather than a smooth field: the bone/air step is what makes an interpolation
    # disagree at all (the deviation scales with the neighbour gap), so a smooth fixture would assert
    # the resample tolerances against data that cannot exercise them.
    radius = axes[0] ** 2 + axes[1] ** 2 + axes[2] ** 2
    intensity = np.where(radius < 0.4, _PEAK, -_PEAK) + 30 * rng.standard_normal(_SPATIAL)
    return {
        "Intensity": intensity.astype(np.float32)[None],
        # How a CT is actually stored, and the one dtype that quantizes an interpolation (see _LSB_ATOL).
        "Int16": intensity.astype(np.int16)[None],
        # Nested structures, not uniform noise: dilating scattered foreground saturates to all-ones,
        # and an all-ones result is the same whether or not the halo was ever read. Compact labels
        # keep a border for Dilate's halo to be wrong about.
        "Labels": np.select([radius < 0.1, radius < 0.2, radius < 0.35], [1, 2, 3], 0).astype(np.uint8)[None],
        "Ensemble": rng.integers(0, 3, (3, *_SPATIAL)).astype(np.float32),
        # The same intensities stored on a rotated direction: the one group whose METADATA, not whose
        # voxels, is the point (see test_a_declaration_reads_the_case_metadata).
        "Oblique": intensity.astype(np.float32)[None],
        # And on a direction that permutes axes, which reorienting transposes the extents of: the group
        # whose metadata moves the patch grid the streamed patches are cut on.
        "Permuting": intensity.astype(np.float32)[None],
        # Stored with the foreground box already on it, which is how a crop is a translation rather
        # than a question about the voxels.
        "Boxed": intensity.astype(np.float32)[None],
    }


def _attributes(group: str) -> Attribute:
    """The metadata a group is stored with -- and so what a declaration about it is handed."""
    attributes = Attribute()
    attributes["Origin"] = np.asarray([-3.0, 5.0, 11.0])
    attributes["Spacing"] = np.asarray(_SPACING)
    attributes["Direction"] = {"Oblique": _OBLIQUE, "Permuting": _PERMUTING}.get(group, _AXIS_ALIGNED).reshape(-1)
    if group == "Ensemble":
        # What a `combine: Concat` reduction writes: the per-model channel counts MergeLabels and
        # Sum shift their label ranges by.
        attributes["number_of_channels_per_model"] = np.asarray([3, 3, 3])
    if group == "Boxed":
        # What Crop.transform_shape leaves on the case: [start, after] margins per spatial axis.
        # Both margins differ on every axis -- a box symmetric anywhere is one a wrong sign or a
        # reversed axis order would still land on.
        attributes["box"] = _BOX
    return attributes


@pytest.fixture(scope="session")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> Dataset:
    """A real on-disk dataset, in the same format (mha) and channel-first layout a run reads."""
    dataset = Dataset(tmp_path_factory.mktemp("workspace") / "Dataset", "mha")
    for group, volume in _volumes().items():
        dataset.write(group, _CASE_NAME, volume, _attributes(group))
    return dataset


def _builtin_transforms() -> list[type[Transform]]:
    """Every concrete transform class KonfAI ships."""
    return [
        cls
        for _, cls in inspect.getmembers(transform_module, inspect.isclass)
        if issubclass(cls, Transform) and cls.__module__ == transform_module.__name__ and not inspect.isabstract(cls)
    ]


def _cases_of(cls: type[Transform]) -> list[_Case]:
    """The configurations to exercise for one transform: the table's, else the transform's own defaults."""
    if cls.__name__ in _CASES:
        return _CASES[cls.__name__]
    if any(parameter.default is parameter.empty for parameter in inspect.signature(cls).parameters.values()):
        return []
    return [_Case(cls())]


def _kind_of(case: _Case) -> LocalityKind:
    """What a case declares about the group it consumes -- asked exactly as the dispatcher asks it."""
    return case.transform.patch_locality(_attributes(case.group)).kind


def _streamable_cases() -> list[_Case]:
    return [
        case
        for cls in _builtin_transforms()
        for case in _cases_of(cls)
        if _kind_of(case) is not LocalityKind.WHOLE_VOLUME
    ]


def _manager(dataset: Dataset, case: _Case) -> DatasetManager:
    return DatasetManager(
        index=0,
        group_src=case.group,
        group_dest=case.group,
        name=_CASE_NAME,
        dataset=dataset,
        patch=DatasetPatch(list(_PATCH_SIZE)),
        transforms=[case.transform],
        data_augmentations_list=[],
    )


def test_every_builtin_transform_is_covered() -> None:
    # A transform this file cannot construct is skipped silently -- and a skipped WHOLE_VOLUME
    # declaration is exactly how a wrong one would hide. Give it a _CASES entry instead.
    uncovered = [cls.__name__ for cls in _builtin_transforms() if not _cases_of(cls)]
    assert uncovered == []


def test_no_declaration_writes_to_the_case_metadata() -> None:
    # READ-ONLY, checked. A declaration is made once for the whole case, so a value it wrote would be
    # one patch's answer imposed on every other. The dispatcher hands over a copy, which contains the
    # damage but also hides it -- so the rule is worth stating where it can actually be seen.
    for cls in _builtin_transforms():
        for case in _cases_of(cls):
            attribute = _attributes(case.group)
            before = dict(attribute)
            case.transform.patch_locality(attribute)
            assert dict(attribute) == before, f"{cls.__name__}.patch_locality() wrote to cache_attribute"


def test_every_locality_kind_is_exercised() -> None:
    # The property below is only as good as the kinds it reaches: every streamable kind must have at
    # least one built-in standing for it, so a regression in one kind's dispatch cannot pass unseen.
    kinds = {_kind_of(case) for case in _streamable_cases()}
    assert kinds == set(LocalityKind) - {LocalityKind.WHOLE_VOLUME}


@pytest.mark.parametrize(
    "case",
    _streamable_cases(),
    ids=lambda case: f"{type(case.transform).__name__}-{_kind_of(case).value}-{case.group}",
)
def test_streamed_patch_equals_whole_volume(case: _Case, dataset: Dataset) -> None:
    streamed = _manager(dataset, case)
    reference = _manager(dataset, case)
    # `load` runs the transform over the whole volume; `get_data` then cuts the patch out of the result.
    # The other manager never loads, so the same `get_data` call streams instead -- same public entry
    # point, same patch grid, and the declaration alone decides which path runs.
    reference.load([case.transform], [])
    assert reference.loaded
    assert streamed.can_stream_patch(0)

    for index in range(streamed.get_size(0)):
        got = streamed.get_data(index, 0, [], True)
        expected = reference.get_data(index, 0, [], True)
        assert got.shape == expected.shape
        assert got.dtype == expected.dtype
        np.testing.assert_allclose(got.numpy(), expected.numpy(), rtol=0, atol=case.atol)


class _FlipIfAxisAligned(Flip):
    """A reorientation that is a flip only when the case says so -- the declaration no config can make.

    Whether reorienting to canonical is a flip (streamable) or a resample (not) is decided by the
    direction cosines the case was stored with, so it can only be answered from ``cache_attribute``.
    ``Canonical`` makes exactly this declaration on a real chain; this stands in for it on the smallest
    transform that can, so the mechanism is proven whatever any one built-in decides about its own case.
    """

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        if "Direction" not in cache_attribute:
            return PatchLocality(LocalityKind.WHOLE_VOLUME)
        direction = cache_attribute.get_np_array("Direction")
        # Axis-aligned iff exactly one non-zero per row and column, i.e. n non-zeros in an orthonormal
        # matrix. Anything else mixes axes, and no flip reproduces it.
        if np.count_nonzero(direction) != len(_SPATIAL):
            return PatchLocality(LocalityKind.WHOLE_VOLUME)
        return PatchLocality(LocalityKind.ORIENTATION)


def test_a_declaration_reads_the_case_metadata() -> None:
    """The argument carries the case: one transform, one config, two answers."""
    transform = _FlipIfAxisAligned("0")
    assert transform.patch_locality(_attributes("Intensity")).kind is LocalityKind.ORIENTATION
    assert transform.patch_locality(_attributes("Oblique")).kind is LocalityKind.WHOLE_VOLUME


def test_a_declaration_is_total_on_absent_metadata() -> None:
    """A declaration must answer for any case, including one whose metadata it cannot find.

    The config-time patch_transform checks probe with exactly this, and a group carries only what its
    writer stored -- so the missing key has to fall to the safe kind rather than raise.
    """
    assert _FlipIfAxisAligned("0").patch_locality(Attribute()).kind is LocalityKind.WHOLE_VOLUME


@pytest.mark.parametrize("group, streams", [("Intensity", True), ("Oblique", False)])
def test_the_dispatcher_honours_an_image_dependent_declaration(dataset: Dataset, group: str, streams: bool) -> None:
    """End to end: the same transform streams, or falls back, on the METADATA of the case it is given.

    Both groups hold the same voxels and the same config, and differ only in the Direction they are
    stored with -- so nothing but the declaration can decide the path, and the dispatcher must be
    reading it from the case rather than from the transform.
    """
    assert _manager(dataset, _Case(_FlipIfAxisAligned("0"), group=group)).can_stream_patch(0) is streams


def test_an_image_dependent_stream_equals_the_whole_volume(dataset: Dataset) -> None:
    """And the case it does accept to stream is streamed correctly, border patches included."""
    case = _Case(_FlipIfAxisAligned("0"), group="Intensity")
    streamed, reference = _manager(dataset, case), _manager(dataset, case)
    reference.load([case.transform], [])
    for index in range(streamed.get_size(0)):
        np.testing.assert_array_equal(
            streamed.get_data(index, 0, [], True).numpy(), reference.get_data(index, 0, [], True).numpy()
        )


@pytest.mark.parametrize("group", ["Intensity", "Permuting"], ids=["mirroring", "permuting"])
def test_a_streamed_region_records_the_whole_volume_geometry(dataset: Dataset, group: str) -> None:
    """Streaming is invisible in the METADATA too, not only in the voxels.

    A reorientation rewrites the case's geometry onto the grid it lands on -- a fact about the VOLUME's
    extent, while a streamed stage is only ever handed a patch. The case must come out of the streamed
    path with the geometry the whole-volume pass computes, never with the first patch's own corner
    frozen onto it, and with the same stack depth so ``inverse()`` pops what was pushed.
    """
    case = _Case(Canonical(), group=group)
    streamed, reference = _manager(dataset, case), _manager(dataset, case)
    reference.load([case.transform], [])
    streamed.get_data(0, 0, [], True)

    for key in ("Origin", "Direction", "Spacing"):
        np.testing.assert_array_equal(
            streamed.cache_attributes[0].get_np_array(key), reference.cache_attributes[0].get_np_array(key)
        )
    assert sorted(streamed.cache_attributes[0].keys()) == sorted(reference.cache_attributes[0].keys())


# --------------------------------------------------------------------------------------
# The augmentations. Same contract, same property -- asked of a draw rather than a config.
# --------------------------------------------------------------------------------------

# A HALO draw is sampled by grid_sample from coordinates expressed in the halo'd read extent's frame
# rather than the whole volume's: the same disagreement, for the same reason and with the same
# gradient- and coordinate-scaling, that _RESCALE_ATOL bounds for the streamed resample. It is bitwise
# on neither, and grows with the extent -- a 160^3 case at patch 64 lands at 2e-5 of its range.
_AUGMENTATION_ATOL = _RESCALE_ATOL


@dataclass(frozen=True)
class _AugmentationCase:
    """One representative draw, the kind it declares, and whether the dispatcher reads it.

    Those are two questions, and keeping them apart is the contract's own split: a draw declares what
    its output depends on, and the dispatcher decides whether reading that much is worth it. A wide
    Translate declares a perfectly honest HALO and is refused all the same (see ``_affords_halo``).
    """

    augmentation: DataAugmentation
    kind: LocalityKind
    streams: bool
    group: str = "Intensity"
    atol: float = 0.0


# Every augmentation KonfAI ships, at a draw that is a meaningful streaming case: probabilities are
# pinned so the draw always fires (a copy no draw selected is the identity, which proves nothing), and
# a required constructor argument is given a value. The kind is what THIS draw must declare -- so a
# declaration silently retreating to WHOLE_VOLUME fails here rather than passing vacuously.
_AUGMENTATION_CASES: dict[str, list[_AugmentationCase]] = {
    "Brightness": [_AugmentationCase(augmentation_module.Brightness(0.5), LocalityKind.POINTWISE, True)],
    "Contrast": [_AugmentationCase(augmentation_module.Contrast(0.5), LocalityKind.POINTWISE, True)],
    "CutOUT": [_AugmentationCase(augmentation_module.CutOUT(1.0, 2, 0.0), LocalityKind.WHOLE_VOLUME, False)],
    "Elastix": [_AugmentationCase(augmentation_module.Elastix(), LocalityKind.WHOLE_VOLUME, False)],
    "Flip": [
        _AugmentationCase(FlipAugmentation(f_prob=[1.0, 1.0, 1.0]), LocalityKind.ORIENTATION, True),
        # A displacement field's flipped components are negated, which is not a bijection on values.
        _AugmentationCase(
            FlipAugmentation(f_prob=[1.0, 1.0, 1.0], vector_field=True), LocalityKind.WHOLE_VOLUME, False
        ),
    ],
    "HUE": [_AugmentationCase(augmentation_module.HUE(1.0), LocalityKind.POINTWISE, True)],
    "LumaFlip": [_AugmentationCase(augmentation_module.LumaFlip(), LocalityKind.POINTWISE, True)],
    "Mask": [],  # a second on-disk volume that dictates the output grid; see its note.
    "Noise": [_AugmentationCase(augmentation_module.Noise(1.0), LocalityKind.WHOLE_VOLUME, False)],
    "Permute": [
        _AugmentationCase(augmentation_module.Permute(prob_permute=[1.0, 1.0]), LocalityKind.ORIENTATION, True)
    ],
    "Rotate": [
        _AugmentationCase(augmentation_module.Rotate(a_min=10.0, a_max=10.0), LocalityKind.WHOLE_VOLUME, False),
        # A quarter draw is a signed permutation of the axes, so it is an exact remap whichever multiple
        # of 90 degrees it lands on -- the declaration holds for every draw, not for the seed this
        # happens to run on. The fixture is non-cubic, so 26 of its 27 draws transpose extents and cut
        # the copy on a grid the stored volume does not have.
        _AugmentationCase(augmentation_module.Rotate(is_quarter=True), LocalityKind.ORIENTATION, True),
    ],
    "Saturation": [_AugmentationCase(augmentation_module.Saturation(0.5), LocalityKind.POINTWISE, True)],
    "Scale": [_AugmentationCase(augmentation_module.Scale(), LocalityKind.WHOLE_VOLUME, False)],
    "Translate": [
        # A halo of ceil(1) + 1 = 2 on a patch of 4: half the patch, the widest _affords_halo allows.
        _AugmentationCase(
            augmentation_module.Translate(t_min=-1.0, t_max=1.0), LocalityKind.HALO, True, atol=_AUGMENTATION_ATOL
        ),
        # The same declaration, a 10-voxel shift: an honest halo the dispatcher will not pay for.
        _AugmentationCase(augmentation_module.Translate(t_min=10.0, t_max=10.0), LocalityKind.HALO, False),
    ],
}


def _builtin_augmentations() -> list[type[DataAugmentation]]:
    """Every concrete augmentation class KonfAI ships."""
    return [
        cls
        for _, cls in inspect.getmembers(augmentation_module, inspect.isclass)
        if issubclass(cls, DataAugmentation)
        and cls.__module__ == augmentation_module.__name__
        and not inspect.isabstract(cls)
    ]


def _augmentation_managers(dataset: Dataset, case: _AugmentationCase) -> tuple[DatasetManager, DatasetManager]:
    """Two managers of the same case, on ONE draw: one that streams copy 1, one that loads it."""
    case.augmentation.load(1.0)
    augmentations = DataAugmentationsList(nb=1, data_augmentations={})
    augmentations.data_augmentations = [case.augmentation]

    def manager() -> DatasetManager:
        return DatasetManager(
            index=0,
            group_src=case.group,
            group_dest=case.group,
            name=_CASE_NAME,
            dataset=dataset,
            patch=DatasetPatch(list(_PATCH_SIZE)),
            transforms=[],
            data_augmentations_list=[augmentations],
        )

    streamed, reference = manager(), manager()
    # Constructing a manager re-draws (that is what an epoch reset does), so the two would otherwise
    # compare different draws. Replaying the last one onto both is what makes them the same case.
    for item in (streamed, reference):
        item.reset_augmentation(reset_state=False)
    return streamed, reference


def _augmentation_cases() -> list[_AugmentationCase]:
    return [case for cls in _builtin_augmentations() for case in _AUGMENTATION_CASES.get(cls.__name__, [])]


def test_every_builtin_augmentation_is_covered() -> None:
    # An augmentation absent from the table is never asked anything -- and an unasked declaration is
    # exactly how a wrong one would hide. Give it an entry, empty only if no draw of it can stream.
    uncovered = [cls.__name__ for cls in _builtin_augmentations() if cls.__name__ not in _AUGMENTATION_CASES]
    assert uncovered == []


@pytest.mark.parametrize(
    "case",
    _augmentation_cases(),
    ids=lambda case: f"{type(case.augmentation).__name__}-{case.kind.value}-{case.streams}",
)
def test_an_augmentation_declares_the_kind_its_draw_makes_it(case: _AugmentationCase, dataset: Dataset) -> None:
    streamed, _ = _augmentation_managers(dataset, case)
    # Copy 1 carries the draw; the declaration is asked of it exactly as the dispatcher asks it.
    assert case.augmentation.patch_locality(0, 0, _attributes(case.group)).kind is case.kind
    assert streamed.can_stream_patch(1) is case.streams


def test_no_augmentation_declaration_writes_to_the_case_metadata(dataset: Dataset) -> None:
    # READ-ONLY, checked -- the same rule, and the same reason, as for a transform.
    for case in _augmentation_cases():
        _augmentation_managers(dataset, case)
        attribute = _attributes(case.group)
        before = dict(attribute)
        case.augmentation.patch_locality(0, 0, attribute)
        assert dict(attribute) == before, f"{type(case.augmentation).__name__}.patch_locality() wrote to it"


@pytest.mark.parametrize(
    "case",
    [case for case in _augmentation_cases() if case.streams],
    ids=lambda case: f"{type(case.augmentation).__name__}-{case.kind.value}",
)
def test_streamed_augmented_patch_equals_whole_volume(case: _AugmentationCase, dataset: Dataset) -> None:
    streamed, reference = _augmentation_managers(dataset, case)
    # `load` runs the draw over the whole volume; `get_data` then cuts the patch out of the result. The
    # other manager never loads, so the same `get_data` call streams instead -- same public entry point,
    # same patch grid, same draw, and the declaration alone decides which path runs.
    reference.load([], reference.data_augmentations_list)
    assert reference.loaded
    assert streamed.can_stream_patch(1)

    for index in range(streamed.get_size(1)):
        got = streamed.get_data(index, 1, [], True)
        expected = reference.get_data(index, 1, [], True)
        assert got.shape == expected.shape
        assert got.dtype == expected.dtype
        np.testing.assert_allclose(got.numpy(), expected.numpy(), rtol=0, atol=case.atol)


def test_a_pointwise_augmentation_streams_the_whole_grid_after_a_transform(dataset: Dataset) -> None:
    """A copy is its transforms AND its draw: the dispatcher plans them as one chain.

    A pointwise draw behind a region transform is still one region, so it streams; the same draw behind
    a region transform AND a region draw would be two, which is what the plan refuses.
    """
    torch.manual_seed(0)
    brightness = augmentation_module.Brightness(0.5)
    brightness.load(1.0)
    augmentations = DataAugmentationsList(nb=1, data_augmentations={})
    augmentations.data_augmentations = [brightness]

    def manager(transform: Transform) -> DatasetManager:
        return DatasetManager(
            index=0,
            group_src="Intensity",
            group_dest="Intensity",
            name=_CASE_NAME,
            dataset=dataset,
            patch=DatasetPatch(list(_PATCH_SIZE)),
            transforms=[transform],
            data_augmentations_list=[augmentations],
        )

    resample = ResampleToResolution([2.0, 1.0, 3.0])
    assert manager(resample).can_stream_patch(1) is True
    # Two regions in one chain: the resample's, and the flip's.
    flip = FlipAugmentation(f_prob=[1.0, 1.0, 1.0])
    flip.load(1.0)
    augmentations.data_augmentations = [flip]
    assert manager(resample).can_stream_patch(1) is False
