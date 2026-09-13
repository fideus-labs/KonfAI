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

"""The intensity draws: what each declares, and that a patch equals the volume it is cut from."""

from typing import cast

import numpy as np
import pytest
import torch
from konfai.data.augmentation import (
    ContrastAroundMean,
    Gamma,
    GaussianBlur,
    GaussianNoise,
    Rotate,
    SimulateLowResolution,
)
from konfai.data.augmentation.base import DataAugmentationsList
from konfai.data.patching import DatasetManager, DatasetPatch
from konfai.data.transform import LocalityKind, Resample, TensorCast
from konfai.utils.dataset import Attribute, Dataset
from konfai.utils.errors import AugmentationError

_VOLUME = (np.random.default_rng(3).normal(size=(1, 12, 32, 32)) * 40 + 100).astype(np.float32)


def _managed(stub_class, draw):
    """One manager over ``draw``, with a value-preserving chain so the source's statistic is valid."""
    stub = stub_class(_VOLUME)
    draw.load(1.0)
    augmentations = DataAugmentationsList(nb=1, data_augmentations={})
    augmentations.data_augmentations = [draw]
    return stub, DatasetManager(
        index=0,
        group_src="CT",
        group_dest="CT",
        name="CASE_000",
        dataset=cast(Dataset, stub),
        patch=DatasetPatch([4, 16, 16]),
        transforms=[TensorCast(dtype="float32")],
        data_augmentations_list=[augmentations],
    )


@pytest.mark.parametrize(
    ("build", "kind"),
    [
        (lambda: GaussianNoise(), LocalityKind.POINTWISE),
        (lambda: GaussianBlur(), LocalityKind.HALO),
        (lambda: Gamma(), LocalityKind.GLOBAL_STAT),
        (lambda: ContrastAroundMean(), LocalityKind.GLOBAL_STAT),
        (lambda: SimulateLowResolution(), LocalityKind.HALO),
    ],
    ids=["noise", "blur", "gamma", "contrast", "low-resolution"],
)
def test_each_intensity_draw_declares_the_locality_its_values_need(build, kind: LocalityKind) -> None:
    draw = build()
    draw.load(1.0)
    draw.state_init(0, [[8, 16, 16]], [Attribute()])
    locality = draw.patch_locality(0, 0, Attribute())
    assert locality.kind is kind
    if kind is LocalityKind.WHOLE_VOLUME:
        # Whole-volume costs every case a materialization: the plan must be able to say why.
        assert locality.reason
    if kind is LocalityKind.HALO:
        assert any(radius > 0 for radius in locality.halo)


@pytest.mark.parametrize(
    "build",
    [
        lambda: GaussianNoise(std_min=0.05, std_max=0.05),
        lambda: GaussianBlur(sigma_min=0.8, sigma_max=0.8, in_plane=True),
        lambda: Gamma(gamma_min=1.4, gamma_max=1.4),
        lambda: ContrastAroundMean(factor_min=1.3, factor_max=1.3),
        lambda: SimulateLowResolution(factor_min=1.7, factor_max=1.7),
    ],
    ids=["noise", "blur-in-plane", "gamma", "contrast", "low-resolution"],
)
def test_a_streamed_patch_equals_the_patch_cut_from_the_whole_volume(streaming_dataset_stub, build) -> None:
    """The route a patch takes is a performance decision, never a change of value. A draw whose
    parameters describe the case must read the CASE's description on both routes, or the same voxel
    takes one value through a patch and another through a volume."""
    torch.manual_seed(0)
    _, streamed = _managed(streaming_dataset_stub, build())
    assert streamed.stream_refusal(1, True) is None

    torch.manual_seed(0)
    _, whole = _managed(streaming_dataset_stub, build())
    whole.load(whole.transforms, whole.data_augmentations_list, load_augmentations=True)

    for patch in range(streamed.patch.get_size(1)):
        assert torch.allclose(
            streamed.get_data(patch, 1, [], True, True), whole.get_data(patch, 1, [], True, True), atol=1e-4
        )


def test_a_case_statistic_draw_refuses_a_region_nobody_seeded() -> None:
    """Describing the region instead would be silent: the values would still look plausible."""
    draw = Gamma()
    draw.load(1.0)
    draw.state_init(0, [[4, 8, 8]], [Attribute()])
    with pytest.raises(AugmentationError, match="describe the whole case"):
        draw._seed_statistics(0, 0, torch.zeros(1, 4, 8, 8), Attribute(), whole=False)


def test_an_in_plane_turn_never_mixes_neighbouring_slices() -> None:
    """A 2.5D stack carries neighbouring slices as channels; a turn out of the plane draws each of
    them from a different place, which is the one thing the stack must not do."""
    # Each slice constant and distinct: any mixing shows up as a spread inside a slice.
    volume = torch.arange(8, dtype=torch.float32).reshape(1, 8, 1, 1).repeat(1, 1, 24, 24)
    spreads = {}
    for in_plane in (False, True):
        torch.manual_seed(1)
        draw = Rotate(a_min=15, a_max=45, in_plane=in_plane)
        draw.load(1.0)
        draw.state_init(0, [[8, 24, 24]], [Attribute()])
        turned = draw.compute("CASE_000", 0, 0, volume)[:, :, 6:-6, 6:-6]
        spreads[in_plane] = max(float(turned[0, k].max() - turned[0, k].min()) for k in range(8))
    assert spreads[True] < 1e-4
    assert spreads[False] > 1.0, "the free turn mixed nothing: the test would prove nothing"


def test_a_blur_in_the_plane_asks_for_no_halo_on_the_slice_axis() -> None:
    """A halo declared as one radius broadcasts to every axis, and a 2.5D patch is one slice thick:
    asked for a neighbourhood wider than itself, the plan refuses the whole chain."""
    draw = GaussianBlur(sigma_min=1.0, sigma_max=1.0, in_plane=True)
    draw.load(1.0)
    draw.state_init(0, [[1, 32, 32]], [Attribute()])
    halo = draw.patch_locality(0, 0, Attribute()).halo
    assert len(halo) == 3
    assert halo[0] == 0
    assert halo[1] == halo[2] > 0


def test_a_statistic_a_draw_ahead_of_it_invalidates_is_refused(streaming_dataset_stub) -> None:
    """A pass measures the stage's input as it was under the draw that ran, and the draw is redrawn
    every epoch: reusing that number would standardize the copy by the previous epoch's case. The
    plan refuses instead, and names what to move."""
    stub = streaming_dataset_stub(_VOLUME)
    draws = [GaussianNoise(std_min=60.0, std_max=60.0), Gamma(gamma_min=1.5, gamma_max=1.5)]
    for draw in draws:
        draw.load(1.0)
    listed = DataAugmentationsList(nb=1, data_augmentations={})
    listed.data_augmentations = draws
    manager = DatasetManager(
        index=0,
        group_src="CT",
        group_dest="CT",
        name="CASE_000",
        dataset=cast(Dataset, stub),
        patch=DatasetPatch([4, 16, 16]),
        # A resample ahead of the draws: the stored statistic is no longer Gamma's input either, so
        # only a measured one could serve, and the draw ahead of it is what rules that out.
        transforms=[TensorCast(dtype="float32"), Resample(spacing=[1.5, 1.5, 1.5])],
        data_augmentations_list=[listed],
    )
    manager._require_statistics()
    refusal = manager.stream_refusal(1, True)
    assert refusal is not None
    assert "redrawn every epoch" in refusal
