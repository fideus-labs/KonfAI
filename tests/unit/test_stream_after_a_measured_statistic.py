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

"""A statistic the store cannot seed is measured once, and the chain streams from there.

``Standardize`` after a value-changing stage wants the statistics of its OWN input, which the stored
volume no longer describes, so the chain cannot stream and every patch of the case costs a whole
volume. The number it needs is the one a whole-volume pass computes anyway: kept past the pass, it
seeds the regions, and the case pays one materialization instead of one per read.

The whole-volume result is the reference, as everywhere on this path: streaming is an optimisation of
memory, never of meaning.
"""

from typing import cast

import numpy as np
import torch
from konfai.data.augmentation import ContrastAroundMean, Gamma
from konfai.data.augmentation.base import DataAugmentationsList
from konfai.data.patching import DatasetManager, DatasetPatch
from konfai.data.transform import Clip, Normalize, Resample, Standardize, Statistics, Transform
from konfai.data.transform.base import LocalityKind, PatchLocality
from konfai.utils.dataset import Dataset
from konfai.utils.ome_zarr import write_ome_zarr
from oracle_support import manager


def _source(tmp_path):
    volume = np.arange(1 * 8 * 8 * 8, dtype=np.float32).reshape(1, 8, 8, 8) - 200.0
    store = tmp_path / "src" / "CASE_000" / "CT.ome.zarr"
    store.parent.mkdir(parents=True)
    write_ome_zarr(store, volume, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0))
    return volume


def test_a_measured_statistic_lets_the_chain_stream_and_lands_on_the_same_values(tmp_path):
    _source(tmp_path)
    chain = [Clip(min_value=-50.0, max_value=50.0), Standardize()]
    case = manager(Dataset(tmp_path / "src", "omezarr"), chain)

    # Nothing has measured it yet, and the refusal says which stage and why.
    assert not case.can_stream_patch(0, apply_augmentations=False)
    refusal = case.stream_refusal(0, apply_augmentations=False)
    assert "Standardize" in refusal
    assert "the stored volume's statistic is not this stage's input" in refusal

    case.load(chain, [], load_augmentations=False)
    whole = case.data[0]

    # The pass measured it, so the same chain streams now.
    assert case.can_stream_patch(0, apply_augmentations=False)
    assert case.stream_refusal(0, apply_augmentations=False) is None

    target = (slice(0, 4), slice(0, 8), slice(0, 8))
    region = case.read_region(target, 0, False)
    assert region.shape == (1, 4, 8, 8)
    assert torch.equal(region, whole[:, 0:4])


def test_the_statistic_kept_is_the_stage_s_own_input_not_the_stored_volume(tmp_path):
    """The two numbers differ, and taking the wrong one is a silently wrong training.

    ``Clip`` moves the mean of this volume, so a region seeded with the STORED statistic lands
    somewhere else entirely. Pinning the gap keeps the seed honest if the source of it ever moves.
    """
    volume = _source(tmp_path)
    chain = [Clip(min_value=-50.0, max_value=50.0), Standardize()]
    case = manager(Dataset(tmp_path / "src", "omezarr"), chain)
    # A pass keeps the measurement a plan waits for.
    assert not case.can_stream_patch(0, apply_augmentations=False)
    case.load(chain, [], load_augmentations=False)

    clipped = np.clip(volume, -50.0, 50.0)
    assert abs(float(clipped.mean()) - float(volume.mean())) > 1.0

    region = case.read_region((slice(0, 4), slice(0, 8), slice(0, 8)), 0, False)
    seeded_with_the_stored_mean = (clipped[:, 0:4] - volume.mean()) / volume.std()
    assert not np.allclose(region.numpy(), seeded_with_the_stored_mean, atol=1e-3)


def test_the_cache_is_sized_on_what_the_chain_lands_on(tmp_path):
    """A resample lands 27 times smaller than it reads, and the cache holds what it lands on.

    Sized on the stored shape, a dataset that fits is refused the cache and pays the streamed route
    instead.
    """
    _source(tmp_path)
    case = manager(Dataset(tmp_path / "src", "omezarr"), [Resample(spacing=[2.0, 2.0, 2.0])])
    assert case.base_shape == [1, 8, 8, 8]
    assert case.spatial_shape == [4, 4, 4]
    assert case.landed_channels == 1


def test_the_cache_counts_the_channels_the_chain_lands_on(tmp_path):
    """A one-hot lands three channels where one was stored: the cache holds three."""
    from konfai.data.transform import OneHot, TensorCast

    _source(tmp_path)
    case = manager(Dataset(tmp_path / "src", "omezarr"), [TensorCast(dtype="int64"), OneHot(num_classes=3)])
    assert case.landed_channels == 3


def _volume(offset: float) -> np.ndarray:
    return (np.random.default_rng(0).normal(size=(1, 12, 32, 32)) * 40 + offset).astype(np.float32)


def _case(stub, transforms, draws=(), nb: int = 1) -> DatasetManager:
    augmentations = []
    if draws:
        listed = DataAugmentationsList(nb=nb, data_augmentations={})
        listed.data_augmentations = list(draws)
        augmentations = [listed]
    return DatasetManager(
        index=0,
        group_src="CT",
        group_dest="CT",
        name="CASE_000",
        dataset=cast(Dataset, stub),
        patch=DatasetPatch([4, 16, 16]),
        transforms=transforms,
        data_augmentations_list=augmentations,
    )


def _patches(case: DatasetManager, a: int, augmented: bool) -> list[torch.Tensor]:
    return [case.get_data(index, a, [], True, augmented).clone() for index in range(case.patch.get_size(a))]


def _whole(case: DatasetManager, a: int, augmented: bool) -> list[torch.Tensor]:
    case.load(case.transforms, case.data_augmentations_list, load_augmentations=augmented)
    patches = _patches(case, a, augmented)
    case.unload()
    case.unload_augmentation()
    return patches


def _same(left: list[torch.Tensor], right: list[torch.Tensor]) -> bool:
    return all(torch.equal(one, other) for one, other in zip(left, right, strict=True))


def _close(left: list[torch.Tensor], right: list[torch.Tensor], atol: float = 1e-6) -> bool:
    """Within what the region tests allow a resample: its interpolation rounds with the region it is
    computed over on some platforms (Windows)."""
    return all(
        np.allclose(one.numpy(), other.numpy(), rtol=0.0, atol=atol) for one, other in zip(left, right, strict=True)
    )


def test_a_later_draw_recording_the_same_keys_leaves_each_draw_its_own_statistic(streaming_dataset_stub):
    """A pass records every draw's statistic into one scope, and a second Gamma records the same keys
    over the first's. Once a redraw lets the first Gamma stream, its regions are seeded with what it
    measured, not with what the scope held at the end of the chain."""
    stub = streaming_dataset_stub(_volume(100.0))
    first, contrast, second = Gamma(1.4, 1.4), ContrastAroundMean(0.5, 0.5), Gamma(0.6, 0.6)
    for draw in (first, contrast, second):
        draw.load(1.0)
    case = _case(stub, [Clip(min_value=0.0, max_value=180.0)], [first, contrast, second])
    case.warm_stream_statistics([1])

    contrast.load(0.0)
    second.load(0.0)
    case.reset_augmentation()
    case.warm_stream_statistics([1])

    assert case.stream_refusal(1, True) is None
    assert _same(_patches(case, 1, True), _whole(case, 1, True))


def test_a_later_transform_recording_the_same_keys_leaves_both_routes_on_the_first_pass(streaming_dataset_stub):
    """``Normalize`` records the Min and Max it measured, and a saving ``Clip`` after it records its bounds
    under the same keys. Seeded with the Clip's, the regions and every later whole-volume pass would
    saturate: both keep landing where the first pass did."""
    stub = streaming_dataset_stub(_volume(20.0))
    chain = [
        Clip(min_value=-50.0, max_value=50.0),
        Normalize(),
        Clip(min_value=-0.5, max_value=0.5, save_clip_min=True, save_clip_max=True),
    ]
    case = _case(stub, chain)
    reference = _whole(case, 0, False)
    case.warm_stream_statistics([0], apply_augmentations=False)

    assert case.stream_refusal(0, False) is None
    assert _same(_patches(case, 0, False), reference)
    assert _same(_whole(case, 0, False), reference)


def test_a_stage_no_pass_can_measure_is_not_measured_again(streaming_dataset_stub):
    """``Statistics`` wants Min/Max/Mean/Std and records ImageMin and the like, so no pass gives its seed,
    and the numbers other stages recorded under those names are not its own. It is refused for good,
    not measured again at every warm-up."""
    stub = streaming_dataset_stub(_volume(20.0))
    chain = [Clip(min_value=-50.0, max_value=50.0, save_clip_min=True, save_clip_max=True), Standardize(), Statistics()]
    case = _case(stub, chain)
    for _ in range(3):
        case.warm_stream_statistics([0], apply_augmentations=False)
    reads = stub.full_reads

    case.warm_stream_statistics([0], apply_augmentations=False)
    assert case.stream_refusal(0, False) is not None
    assert stub.full_reads == reads


def test_a_draw_that_did_not_select_the_copy_does_not_hold_back_the_statistic_behind_it(streaming_dataset_stub):
    """A draw that did not select the copy is the identity, so the stage behind it sees the chain's own
    output, the same every epoch: its measurement serves, and the copy streams."""
    stub = streaming_dataset_stub(_volume(100.0))
    contrast, gamma = ContrastAroundMean(0.5, 0.5), Gamma(1.4, 1.4)
    contrast.load(0.0)
    gamma.load(1.0)
    case = _case(stub, [Clip(min_value=0.0, max_value=180.0)], [contrast, gamma])
    case.warm_stream_statistics([1])

    assert case.stream_refusal(1, True) is None
    assert _same(_patches(case, 1, True), _whole(case, 1, True))


class _SeedsItself(Transform):
    """A statistic stage with no key to seed: it reads the stored volume itself, as a masked Clip does."""

    def __call__(self, name, tensor, cache_attribute):
        return tensor

    def transform_shape(self, group_src, name, shape, cache_attribute):
        return shape

    def patch_locality(self, cache_attribute):
        return PatchLocality(LocalityKind.GLOBAL_STAT)


def test_a_stage_seeding_itself_from_the_stored_volume_does_not_stream_behind_a_value_change(streaming_dataset_stub):
    stub = streaming_dataset_stub(_volume(100.0))
    case = _case(stub, [Clip(min_value=0.0, max_value=180.0), _SeedsItself()])
    case.warm_stream_statistics([0], apply_augmentations=False)

    assert case.stream_refusal(0, False) is not None
    assert stub.full_reads == 0


def test_a_pass_runs_under_the_statistics_the_regions_replay(streaming_dataset_stub):
    """``Standardize`` at the head is seeded from the store, and ``Gamma`` behind it is measured by a pass.
    Measured on the stage's own float32 standardization instead of the store's, a gamma below one turns
    the 1e-6 offset at the darkest voxel into 2e-3."""
    volume = (np.random.default_rng(1).normal(size=(1, 12, 32, 32)) * 37.3 + 113.7).astype(np.float32)
    stub = streaming_dataset_stub(volume)
    gamma = Gamma(0.5, 0.5)
    gamma.load(1.0)
    case = _case(stub, [Standardize()], [gamma])
    case.warm_stream_statistics([0, 1])

    assert case.stream_refusal(1, True) is None
    assert _same(_patches(case, 1, True), _whole(case, 1, True))


def test_a_measured_stage_replays_the_statistics_the_pass_measured_it_with(streaming_dataset_stub):
    """Only copy 0 is planned, so the store's statistic for the head ``Standardize`` never reaches the pass:
    the pass measures it too, and the regions replay that number rather than the store's."""
    stub = streaming_dataset_stub(_volume(100.0))
    case = _case(stub, [Standardize(), Resample(spacing=[1.5, 1.5, 1.5]), Normalize()])
    case.warm_stream_statistics([0], apply_augmentations=False)

    assert case.stream_refusal(0, False) is None
    assert _close(_patches(case, 0, False), _whole(case, 0, False))


def test_one_pass_answers_every_copy_waiting_on_a_transform(streaming_dataset_stub):
    stub = streaming_dataset_stub(_volume(20.0))
    gamma = Gamma(1.4, 1.4)
    gamma.load(0.0)
    case = _case(stub, [Clip(min_value=-50.0, max_value=50.0), Standardize()], [gamma], nb=3)
    case.warm_stream_statistics([0, 1, 2, 3])

    assert all(case.stream_refusal(a, True) is None for a in range(4))
    assert stub.full_reads == 1


def test_one_pass_measures_stages_that_wait_on_each_other(streaming_dataset_stub):
    """The walk that finds a stage waiting on a pass goes on and asks for every other one."""
    stub = streaming_dataset_stub(_volume(20.0))
    chain = [Clip(min_value=-50.0, max_value=50.0), Normalize(), Clip(min_value=-0.5, max_value=0.5), Standardize()]
    case = _case(stub, chain)
    case.warm_stream_statistics([0], apply_augmentations=False)

    assert case.stream_refusal(0, False) is None
    assert stub.full_reads == 1
    assert _same(_patches(case, 0, False), _whole(case, 0, False))


def test_a_warm_up_leaves_a_case_in_hand_loaded(streaming_dataset_stub):
    stub = streaming_dataset_stub(_volume(20.0))
    chain = [Clip(min_value=-50.0, max_value=50.0), Standardize()]
    case = _case(stub, chain)
    case.load(chain, [], load_augmentations=False)
    case.warm_stream_statistics([0], apply_augmentations=False)

    assert case.loaded


class _WholeVolume(Transform):
    """Declares no locality, so the plan routes it whole-volume."""

    def __call__(self, name, tensor, cache_attribute):
        return tensor

    def transform_shape(self, group_src, name, shape, cache_attribute):
        return shape


def test_no_pass_runs_for_a_chain_a_later_stage_keeps_whole(streaming_dataset_stub):
    stub = streaming_dataset_stub(_volume(20.0))
    case = _case(stub, [Clip(min_value=-50.0, max_value=50.0), Standardize(), _WholeVolume()])
    case.warm_stream_statistics([0], apply_augmentations=False)

    assert "WHOLE_VOLUME" in case.stream_refusal(0, False)
    assert stub.full_reads == 0


def test_a_pass_that_drops_the_plans_leaves_the_case_its_measured_statistics(streaming_dataset_stub):
    """The inverse and the patch transforms read the case attribute, which a streamed copy fills from its
    first region: after a pass drops the plans, the next region fills it again."""
    stub = streaming_dataset_stub(_volume(20.0))
    gamma = Gamma(1.4, 1.4)
    gamma.load(1.0)
    case = _case(stub, [Clip(min_value=-50.0, max_value=50.0), Standardize()], [gamma])
    case.warm_stream_statistics([0])
    case.get_data(0, 0, [], True, True)
    case.get_data(0, 1, [], True, True)  # copy 1 waits on a pass: served whole, the plans dropped
    case.unload()
    case.unload_augmentation()

    case.get_data(1, 0, [], True, True)
    assert "Mean" in case.cache_attributes[0]
    assert "Std" in case.cache_attributes[0]


def test_a_replan_while_the_case_is_loaded_keeps_its_statistics(streaming_dataset_stub):
    """A copy waiting on a pass loads the case whole, and the loader's next probe replans the copy that
    streams: the attribute its patches and inverse read keeps what the pass recorded."""
    stub = streaming_dataset_stub(_volume(20.0))
    gamma = Gamma(1.4, 1.4)
    gamma.load(1.0)
    case = _case(stub, [Clip(min_value=-50.0, max_value=50.0), Standardize(lazy=True)], [gamma])
    case.warm_stream_statistics([0])
    case.get_data(0, 1, [], True, True)
    assert case.loaded

    case.can_stream_patch(0, True)
    assert "Mean" in case.cache_attributes[0]
    assert "Std" in case.cache_attributes[0]


def test_a_measured_stage_is_not_streamed_over_a_store_seed_of_the_same_names(streaming_dataset_stub):
    """``Statistics`` records nothing a pass can keep, so it is seeded from the store, and that seed sits in
    the backup a whole-volume load starts from, where ``Standardize`` would read it. The chain stays whole,
    and a whole load after the warm-up lands where one that ran no plan does."""
    chain = [Statistics(), Clip(min_value=-1.0, max_value=1.0), Standardize()]
    case = _case(streaming_dataset_stub(_volume(20.0)), chain)
    case.warm_stream_statistics([0], apply_augmentations=False)

    assert case.stream_refusal(0, False) is not None
    fresh = _case(
        streaming_dataset_stub(_volume(20.0)), [Statistics(), Clip(min_value=-1.0, max_value=1.0), Standardize()]
    )
    assert _same(_whole(case, 0, False), _whole(fresh, 0, False))


def test_a_copy_streamed_after_a_whole_load_records_its_geometry_once(streaming_dataset_stub):
    """A whole-volume load records the resample's grid on the case attribute, and the first streamed region
    after it records it again: counted twice, the inverse would pop back to the resampled grid."""
    chain = [Resample(spacing=[1.5, 1.5, 1.5]), Standardize()]
    case = _case(streaming_dataset_stub(_volume(20.0)), chain)
    case.warm_stream_statistics([0], apply_augmentations=False)
    case.load(case.transforms, [], load_augmentations=False)
    case.can_stream_patch(0, False)
    case.unload()
    case.get_data(0, 0, [], True, False)

    fresh = _case(streaming_dataset_stub(_volume(20.0)), [Resample(spacing=[1.5, 1.5, 1.5]), Standardize()])
    fresh.load(fresh.transforms, [], load_augmentations=False)
    assert case.cache_attributes[0]._count_key("Spacing") == fresh.cache_attributes[0]._count_key("Spacing")
