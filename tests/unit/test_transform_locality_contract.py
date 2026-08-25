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
the whole-volume result cut on the same grid: proven here for EVERY built-in transform AND every
built-in augmentation, on a real on-disk dataset, over the full patch grid.

Both are ENUMERATED from their module and selected by their own declaration, so a newly declared
streamable stage is covered the day it lands. The case tables never decide coverage: they supply
constructor arguments, and a fixture group, where a stage's defaults are not a meaningful streaming
case, and (for an augmentation, whose declaration is about a draw rather than a config) the kind
that draw must produce.

The declaration is handed the case's metadata, so it is asked here exactly as the dispatcher asks it --
per group, from what that group is stored with. The rules that argument comes with (read-only, total)
are checked too, and a declaration that only the image can make is exercised end to end.

The registry and the case on disk live in ``oracle_support``, which the two write-side files read
too: ``test_transform_materialize_contract`` over the storage formats, ``test_streamed_oracle`` over
the dtype, the rank, the geometry and the number of regions.
"""

import numpy as np
import pytest
import torch
from konfai.data import augmentation as augmentation_module
from konfai.data.augmentation import DataAugmentationsList
from konfai.data.augmentation import Flip as FlipAugmentation
from konfai.data.patching import DatasetManager, DatasetPatch
from konfai.data.transform import (
    Canonical,
    Flip,
    LocalityKind,
    PatchLocality,
    Resample,
    Transform,
)
from konfai.utils.dataset import Attribute, Dataset
from oracle_support import (
    CASE_NAME,
    FIXED_GEOMETRY,
    READ_REFUSED_KINDS,
    AugmentationCase,
    StageCase,
    attributes,
    augmentation_cases,
    build_case,
    builtin_augmentations,
    builtin_transforms,
    cases_of,
    kind_of,
    stage_cases,
    streamable_cases,
)

pytest.importorskip("SimpleITK")

#: The patch grid the property is proven on: no extent of FIXED_GEOMETRY is a multiple of it, so the
#: last patch of every axis is a border patch the read plan has to pad (3x3x3, 19 of 27 at a border).
_PATCH_SIZE = [4, 4, 4]


def _attributes(group: str) -> Attribute:
    """The metadata a group is stored with, and so what a declaration about it is handed."""
    return attributes(FIXED_GEOMETRY, group)


@pytest.fixture(scope="session")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> Dataset:
    return build_case(tmp_path_factory.mktemp("workspace") / "Dataset", FIXED_GEOMETRY)


def _manager(dataset: Dataset, case: StageCase) -> DatasetManager:
    # What a run does before it builds a manager (Data.prepare): a stage that reads a SECOND entry --
    # a mask, a field, a reference grid: is handed the roots to find it in.
    case.transform.set_datasets([dataset])
    return DatasetManager(
        index=0,
        group_src=case.group,
        group_dest=case.group,
        name=CASE_NAME,
        dataset=dataset,
        patch=DatasetPatch(list(_PATCH_SIZE)),
        transforms=[case.transform],
        data_augmentations_list=[],
    )


def test_every_builtin_transform_is_covered() -> None:
    # A transform this file cannot construct is skipped silently, and a skipped WHOLE_VOLUME
    # declaration is exactly how a wrong one would hide. Give it a _CASES entry instead.
    uncovered = [cls.__name__ for cls in builtin_transforms() if not cases_of(cls)]
    assert uncovered == []


def test_no_declaration_writes_to_the_case_metadata() -> None:
    # READ-ONLY, checked. A declaration is made once for the whole case, so a value it wrote would be
    # one patch's answer imposed on every other. The dispatcher hands over a copy, which contains the
    # damage but also hides it, so the rule is worth stating where it can actually be seen.
    for cls in builtin_transforms():
        for case in cases_of(cls):
            attribute = _attributes(case.group)
            before = dict(attribute)
            case.transform.patch_locality(attribute)
            assert dict(attribute) == before, f"{cls.__name__}.patch_locality() wrote to cache_attribute"


def test_every_locality_kind_is_exercised() -> None:
    # The property below is only as good as the kinds it reaches: every read-streamable kind must have
    # at least one built-in standing for it, so a regression in one kind's dispatch cannot pass unseen.
    kinds = {kind_of(case) for case in streamable_cases()}
    assert kinds == set(LocalityKind) - set(READ_REFUSED_KINDS)


def test_slab_declaration_refuses_the_read_path(dataset: Dataset) -> None:
    # SLAB is a write-side contract (stream_slab gets the region); a read chain carrying it must fall
    # back to the whole volume rather than run the stage without its side effect's context.
    case = stage_cases()["InferenceStack"][0]
    assert kind_of(case) is LocalityKind.SLAB
    assert not _manager(dataset, case).can_stream_patch(0)


@pytest.mark.parametrize(
    "case",
    streamable_cases(),
    ids=lambda case: f"{type(case.transform).__name__}-{kind_of(case).value}-{case.group}",
)
def test_streamed_patch_equals_whole_volume(case: StageCase, dataset: Dataset) -> None:
    streamed = _manager(dataset, case)
    reference = _manager(dataset, case)
    # `load` runs the transform over the whole volume; `get_data` then cuts the patch out of the result.
    # The other manager never loads, so the same `get_data` call streams instead: same public entry
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
    """A reorientation that is a flip only when the case says so: the declaration no config can make.

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
        if np.count_nonzero(direction) != FIXED_GEOMETRY.rank:
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
    writer stored, so the missing key has to fall to the safe kind rather than raise.
    """
    assert _FlipIfAxisAligned("0").patch_locality(Attribute()).kind is LocalityKind.WHOLE_VOLUME


@pytest.mark.parametrize("group, streams", [("Intensity", True), ("Oblique", False)])
def test_the_dispatcher_honours_an_image_dependent_declaration(dataset: Dataset, group: str, streams: bool) -> None:
    """End to end: the same transform streams, or falls back, on the METADATA of the case it is given.

    Both groups hold the same voxels and the same config, and differ only in the Direction they are
    stored with, so nothing but the declaration can decide the path, and the dispatcher must be
    reading it from the case rather than from the transform.
    """
    assert _manager(dataset, StageCase(_FlipIfAxisAligned("0"), group=group)).can_stream_patch(0) is streams


def test_an_image_dependent_stream_equals_the_whole_volume(dataset: Dataset) -> None:
    """And the case it does accept to stream is streamed correctly, border patches included."""
    case = StageCase(_FlipIfAxisAligned("0"), group="Intensity")
    streamed, reference = _manager(dataset, case), _manager(dataset, case)
    reference.load([case.transform], [])
    for index in range(streamed.get_size(0)):
        np.testing.assert_array_equal(
            streamed.get_data(index, 0, [], True).numpy(), reference.get_data(index, 0, [], True).numpy()
        )


@pytest.mark.parametrize("group", ["Intensity", "Permuting"], ids=["mirroring", "permuting"])
def test_a_streamed_region_records_the_whole_volume_geometry(dataset: Dataset, group: str) -> None:
    """Streaming is invisible in the METADATA too, not only in the voxels.

    A reorientation rewrites the case's geometry onto the grid it lands on: a fact about the VOLUME's
    extent, while a streamed stage is only ever handed a patch. The case must come out of the streamed
    path with the geometry the whole-volume pass computes, never with the first patch's own corner
    frozen onto it, and with the same stack depth so ``inverse()`` pops what was pushed.
    """
    case = StageCase(Canonical(), group=group)
    streamed, reference = _manager(dataset, case), _manager(dataset, case)
    reference.load([case.transform], [])
    streamed.get_data(0, 0, [], True)

    for key in ("Origin", "Direction", "Spacing"):
        np.testing.assert_array_equal(
            streamed.cache_attributes[0].get_np_array(key), reference.cache_attributes[0].get_np_array(key)
        )
    assert sorted(streamed.cache_attributes[0].keys()) == sorted(reference.cache_attributes[0].keys())


# --------------------------------------------------------------------------------------
# The augmentations. Same contract, same property: asked of a draw rather than a config.
# --------------------------------------------------------------------------------------


def _augmentation_managers(dataset: Dataset, case: AugmentationCase) -> tuple[DatasetManager, DatasetManager]:
    """Two managers of the same case, on ONE draw: one that streams copy 1, one that loads it."""
    case.augmentation.load(1.0)
    augmentations = DataAugmentationsList(nb=1, data_augmentations={})
    augmentations.data_augmentations = [case.augmentation]

    def manager() -> DatasetManager:
        return DatasetManager(
            index=0,
            group_src=case.group,
            group_dest=case.group,
            name=CASE_NAME,
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


def _augmentation_cases() -> list[AugmentationCase]:
    return [case for cls in builtin_augmentations() for case in augmentation_cases().get(cls.__name__, [])]


def test_every_builtin_augmentation_is_covered() -> None:
    # An augmentation absent from the table is never asked anything, and an unasked declaration is
    # exactly how a wrong one would hide. Give it an entry, empty only if no draw of it can stream.
    uncovered = [cls.__name__ for cls in builtin_augmentations() if cls.__name__ not in augmentation_cases()]
    assert uncovered == []


@pytest.mark.parametrize(
    "case",
    _augmentation_cases(),
    ids=lambda case: f"{type(case.augmentation).__name__}-{case.kind.value}-{case.streams}",
)
def test_an_augmentation_declares_the_kind_its_draw_makes_it(case: AugmentationCase, dataset: Dataset) -> None:
    streamed, _ = _augmentation_managers(dataset, case)
    # Copy 1 carries the draw; the declaration is asked of it exactly as the dispatcher asks it.
    assert case.augmentation.patch_locality(0, 0, _attributes(case.group)).kind is case.kind
    assert streamed.can_stream_patch(1) is case.streams


def test_no_augmentation_declaration_writes_to_the_case_metadata(dataset: Dataset) -> None:
    # READ-ONLY, checked: the same rule, and the same reason, as for a transform.
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
def test_streamed_augmented_patch_equals_whole_volume(case: AugmentationCase, dataset: Dataset) -> None:
    streamed, reference = _augmentation_managers(dataset, case)
    # `load` runs the draw over the whole volume; `get_data` then cuts the patch out of the result. The
    # other manager never loads, so the same `get_data` call streams instead: same public entry point,
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

    A pointwise draw behind a region transform is still one region, so it streams; a region draw
    behind a region transform makes two, and region stages compose, each pulling through the one
    before it, so the pair streams too and must match the loaded copy patch for patch.
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
            name=CASE_NAME,
            dataset=dataset,
            patch=DatasetPatch(list(_PATCH_SIZE)),
            transforms=[transform],
            data_augmentations_list=[augmentations],
        )

    resample = Resample([2.0, 1.0, 3.0])
    assert manager(resample).can_stream_patch(1) is True
    # Two regions in one chain (the resample's, then the flip draw's) composed into one pull.
    flip = FlipAugmentation(f_prob=[1.0, 1.0, 1.0])
    flip.load(1.0)
    augmentations.data_augmentations = [flip]
    streamed, reference = manager(resample), manager(resample)
    for item in (streamed, reference):
        item.reset_augmentation(reset_state=False)
    assert streamed.can_stream_patch(1) is True
    reference.load([resample], [augmentations], load_augmentations=True)
    for index in range(streamed.get_size(1)):
        got = streamed.get_data(index, 1, [], True)
        expected = reference.get_data(index, 1, [], True)
        assert got.shape == expected.shape
        # The streamed resample matches F.interpolate to float32 interpolation rounding; the flip
        # draw on top is an exact index remap of it.
        np.testing.assert_allclose(got.numpy(), expected.numpy(), rtol=0, atol=1e-3)
