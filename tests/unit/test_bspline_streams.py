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

"""A cubic B-spline resample streams: each region read off a window ``spline.HALO`` voxels past its taps, its
prefilter forgetting where the window starts; on the faces grid, clipped to the range of its input, which the plan
measures by slabs through the stages before it. The whole-volume result is the reference, within the window's bound:
16 float64 ulps of the volume's largest magnitude, a value's float32 rounding aside."""

import numpy as np
import pytest
import torch
from konfai.data import spline
from konfai.data.patching import DatasetManager, DatasetPatch
from konfai.data.transform import Clip, OneHot, Resample, Save, Standardize, TensorCast, Transform
from konfai.data.transform.base import LocalityKind, PatchLocality
from konfai.utils.dataset import Attribute, Dataset
from konfai.utils.errors import DatasetManagerError
from konfai.utils.ome_zarr import write_ome_zarr
from oracle_support import manager


def test_a_window_reads_the_whole_volume_s_b_spline() -> None:
    """Random volumes, grids and regions: the window's float64 sum is the whole's to a few float64 ulps of the
    volume's largest value, the clip and the axis resampled apart included."""
    rng = np.random.default_rng(0)
    for _ in range(30):
        extents = [int(n) for n in rng.integers(5, 70, 3)]
        shape = tuple(max(1, round(n * f)) for n, f in zip(extents, rng.uniform(0.3, 3.0, 3), strict=True))
        volume = (rng.normal(size=extents) * rng.uniform(1, 3000)).astype(np.float32).astype(np.float64)
        corners, axis = bool(rng.integers(2)), None if rng.uniform() < 0.7 else int(rng.integers(3))
        bounds = spline.value_range(volume)
        whole = spline.resample_region(
            volume, [0, 0, 0], extents, shape, tuple(map(slice, shape)), 3, corners, axis, bounds
        )
        for _ in range(4):
            region = tuple(
                slice(a, int(rng.integers(a + 1, m + 1))) for a, m in ((int(rng.integers(m)), m) for m in shape)
            )
            window = spline.window_of(extents, shape, region, 3, corners, axis)
            part = spline.resample_region(
                volume[tuple(window)], [w.start for w in window], extents, shape, region, 3, corners, axis, bounds
            )
            np.testing.assert_allclose(
                part, whole[region], rtol=0, atol=16 * np.finfo(np.float64).eps * np.abs(volume).max()
            )


def _case(tmp_path, spacing: list[float], chain: list) -> DatasetManager:
    # Long enough along z that a region's window, its taps and the halo, is a part of the volume.
    volume = (np.random.default_rng(3).normal(size=(1, 150, 20, 18)) * 300 + 40).astype(np.float32)
    store = tmp_path / "src" / "CASE_000" / "CT.ome.zarr"
    store.parent.mkdir(parents=True, exist_ok=True)
    write_ome_zarr(store, volume, spacing=tuple(spacing), origin=(0.0, 0.0, 0.0))
    return manager(Dataset(tmp_path / "src", "omezarr"), chain, patch=DatasetPatch([8, 16, 16]))


@pytest.mark.parametrize(
    ("spacing", "plans", "kwargs"),
    [
        ([0.5, 0.5, 0.5], [1.0, 1.0, 3.0], {"grid": "faces"}),  # nnsyn's: clipped to the range measured
        ([0.977, 0.977, 3.0], [1.5, 1.5, 1.5], {"grid": "faces"}),  # 3 mm slices, resampled apart
        ([0.8, 0.8, 2.5], [1.5, 1.5, 1.5], {"grid": "zoom"}),  # scipy's zoom, no clip
    ],
)
def test_a_b_spline_after_a_value_change_streams_from_slabs_and_reads_the_whole_volume_s_values(
    tmp_path, spacing: list[float], plans: list[float], kwargs: dict
) -> None:
    def chain() -> list:
        return [
            TensorCast(dtype="float32"),
            Clip(min_value=-200.0, max_value=500.0),
            Standardize(mean=[40.0], std=[300.0]),
            Resample(spacing=plans, interpolation="bspline", **kwargs),
        ]

    case = _case(tmp_path, spacing, chain())
    # No whole-volume pass: the range is measured by slabs, and the case streams.
    assert case.warm_stream_statistics([0], apply_augmentations=False) is False
    assert not case.loaded
    assert case.stream_refusal(0, apply_augmentations=False) is None
    whole = _case(tmp_path, spacing, chain())
    whole.load(whole.transforms, [], load_augmentations=False)
    out = case.spatial_shape
    reference = whole.data[0]
    bound = 16 * np.finfo(np.float64).eps * float(reference.abs().max())
    for z in range(0, out[0], 7):
        target = (slice(z, min(z + 7, out[0])), slice(3, out[1]), slice(0, out[2] - 2))
        _within_bound(case.read_region(target, 0, False), reference[(slice(None), *target)], bound)


def _within_bound(streamed: torch.Tensor, whole: torch.Tensor, bound: float) -> None:
    """Each streamed value within ``bound`` of the whole volume's, or one float32 ulp of it."""
    gap = (streamed.double() - whole.double()).abs()
    assert bool((gap <= torch.clamp(whole.double().abs() * np.finfo(np.float32).eps, min=bound)).all())


def test_a_window_forgets_what_lies_past_it_within_its_bound() -> None:
    """A far impulse leaves a tail the whole volume writes and the window does not, within the bound; a NaN past the
    window does not reach it, where the whole volume's recursion spreads it over every voxel."""
    extents, shape, region = [240, 4, 4], (120, 4, 4), (slice(50, 51), slice(0, 4), slice(0, 4))
    window = spline.window_of(extents, shape, region, 3, corners=True)
    for first in (1.0, np.nan):
        volume = np.zeros(extents)
        volume[60, 0, 0] = first  # 40 rows from row 50's taps, past the window's 32
        assert not window[0].start <= 60 < window[0].stop
        whole = spline.zoom(volume, shape, 3, corners=True)
        part = spline.zoom_window(volume[tuple(window)], [w.start for w in window], extents, shape, region, 3, True)
        assert np.all(part == 0.0)
        if np.isnan(first):
            assert np.isnan(whole).all()
        else:
            assert np.any(whole[region] != 0.0)
            assert np.abs(part - whole[region]).max() <= 16 * np.finfo(np.float64).eps


def test_a_window_holds_the_taps_of_every_order() -> None:
    """The window of a region holds the taps its zoom reads: order 0 picks the nearest voxel, ``floor(c + 1/2)``."""
    for order in (0, 1, 3):
        for n, m in ((4, 8), (9, 4), (17, 17)):
            for k in range(m):
                coordinate = spline._zoom_coordinates(n, m, False, 0.0)[k : k + 1]
                first, _ = spline._weights(coordinate, order)
                taps = np.clip(first[0] + np.arange(order + 1), 0, n - 1)
                (window,) = spline.window_of([n], (m,), (slice(k, k + 1),), order)
                assert window.start <= taps.min() and taps.max() < window.stop, (order, n, m, k, window, taps)


def test_a_bfloat16_input_measures_its_range_by_slabs(tmp_path) -> None:
    """NumPy has no bfloat16: the slabs measured for the range are read as float32, as the stage reads them."""
    chain = [
        TensorCast(dtype="bfloat16"),
        Clip(min_value=-200.0, max_value=500.0),
        Resample(spacing=[1.0, 1.0, 3.0], grid="faces", interpolation="bspline"),
    ]
    case = _case(tmp_path, [0.5, 0.5, 0.5], chain)
    assert case.warm_stream_statistics([0], apply_augmentations=False) is False
    assert case.stream_refusal(0, apply_augmentations=False) is None


def test_the_range_is_measured_by_regions_the_memory_budget_holds(tmp_path, monkeypatch) -> None:
    """The regions measured for the range are cut as the streamed sweep cuts them, on the budget, the channels and
    the windows of the stages before; a budget no region fits is refused by name."""
    read: list[int] = []
    region = DatasetManager._read_streamed_region
    monkeypatch.setattr(
        DatasetManager,
        "_read_streamed_region",
        lambda self, *args: read.append(region(self, *args)[0].nbytes) or region(self, *args),
    )
    volume = (np.random.default_rng(3).normal(size=(2, 150, 20, 18)) * 300 + 40).astype(np.float32)
    for budget in (60e3, 1e3):
        store = tmp_path / f"{budget}" / "CASE_000" / "CT.ome.zarr"
        store.parent.mkdir(parents=True)
        write_ome_zarr(store, volume, spacing=(0.5, 0.5, 0.5), origin=(0.0, 0.0, 0.0), chunks=(1, 4, 20, 18))
        chain = [
            Clip(min_value=-200.0, max_value=500.0),
            Resample(spacing=[1.0, 1.0, 3.0], grid="faces", interpolation="bspline"),
        ]
        case = manager(Dataset(tmp_path / f"{budget}", "omezarr"), chain, patch=DatasetPatch([8, 16, 16]))
        case.set_memory_budget(budget)
        if budget > 1e3:
            assert case.warm_stream_statistics([0], apply_augmentations=False) is False
            assert read and max(read) <= budget
            continue
        with pytest.raises(DatasetManagerError, match="no region of 'CT' fits the per-rank memory budget"):
            case.warm_stream_statistics([0], apply_augmentations=False)


@pytest.mark.parametrize("measured", [True, False], ids=["measured by slabs", "seeded from the store"])
def test_a_nan_reaches_only_the_regions_whose_window_holds_it(tmp_path, measured: bool) -> None:
    """The clip of a B-spline on the faces grid takes its input's range with NaN left out, as scikit-image's resize
    takes it, whether a pass measures it by slabs or the store seeds it: a region whose window misses the NaN stays
    finite. A constant channel with one NaN at its first voxel."""
    volume = np.full((1, 240, 4, 4), 100.0, dtype=np.float32)
    volume[0, 0, 0, 0] = np.nan
    store = tmp_path / "src" / "CASE_000" / "CT.ome.zarr"
    store.parent.mkdir(parents=True)
    write_ome_zarr(store, volume, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0))
    chain = [Resample(spacing=[1.0, 1.0, 1.5], grid="faces", interpolation="bspline")]
    if measured:
        chain.insert(0, Clip(min_value=-200.0, max_value=500.0))
    case = manager(Dataset(tmp_path / "src", "omezarr"), chain, patch=DatasetPatch([8, 4, 4]))
    case.warm_stream_statistics([0], apply_augmentations=False)
    target = (slice(58, 104), slice(0, 4), slice(0, 4))  # its window: rows 87 to 155 and the halo, not row 0
    assert spline.window_of([240, 4, 4], (160, 4, 4), target, 3)[0].start > 0
    region = case.read_region(target, 0, False)
    assert torch.isfinite(region).all()
    np.testing.assert_allclose(region.numpy(), 100.0, rtol=1e-6)


def _store(tmp_path, volume: np.ndarray, name: str = "src") -> Dataset:
    store = tmp_path / name / "CASE_000" / "CT.ome.zarr"
    store.parent.mkdir(parents=True, exist_ok=True)
    write_ome_zarr(store, volume, spacing=(0.5, 0.5, 0.5), origin=(0.0, 0.0, 0.0), chunks=(1, 4, 20, 18))
    return Dataset(tmp_path / name, "omezarr")


def test_a_range_behind_an_unwritten_save_waits_for_the_pass_that_writes_it(tmp_path) -> None:
    """The source of the stages before the B-spline is a cache this run has still to write: the whole-volume pass
    writes it and measures the range there."""
    volume = (np.random.default_rng(12).normal(size=(1, 150, 20, 18)) * 300 + 40).astype(np.float32)
    chain = [
        Save(str(tmp_path / "cache")),
        Clip(min_value=-200.0, max_value=500.0),
        Resample(spacing=[1.0, 1.0, 3.0], grid="faces", interpolation="bspline"),
    ]
    case = manager(_store(tmp_path, volume), chain, patch=DatasetPatch([8, 16, 16]))
    case.warm_stream_statistics([0], apply_augmentations=False)
    assert Dataset(tmp_path / "cache", "mha").is_dataset_exist("CT", "CASE_000")


def test_the_measured_regions_are_priced_on_the_channels_the_cache_holds(tmp_path, monkeypatch) -> None:
    """Behind a cache of more channels than the stored case (a one-hot before the Save), each region measured for the
    range is priced on the cache's channels, which it reads."""
    labels = np.random.default_rng(13).integers(0, 4, size=(1, 150, 20, 18)).astype(np.float32)
    head = [OneHot(num_classes=4, inverse=False), Save(f"{tmp_path / 'cache'}:omezarr")]
    manager(_store(tmp_path, labels), head, patch=DatasetPatch([8, 16, 16])).read_region(
        (slice(0, 2), slice(0, 4), slice(0, 4)), 0, False
    )  # writes the cache
    read: list[int] = []
    region = DatasetManager._read_streamed_region
    monkeypatch.setattr(
        DatasetManager,
        "_read_streamed_region",
        lambda self, *args: read.append(region(self, *args)[0].nbytes) or region(self, *args),
    )
    chain = [
        OneHot(num_classes=4, inverse=False),
        Save(f"{tmp_path / 'cache'}:omezarr"),
        Clip(min_value=0.0, max_value=1.0),
        Resample(spacing=[1.0, 1.0, 3.0], grid="faces", interpolation="bspline"),
    ]
    case = manager(Dataset(tmp_path / "src", "omezarr"), chain, patch=DatasetPatch([8, 16, 16]))
    case.set_memory_budget(60e3)
    case.warm_stream_statistics([0], apply_augmentations=False)
    assert read and max(read) <= 60e3
    # One block of four rows of the four channels does not fit 20 KB, where one channel's would.
    case = manager(Dataset(tmp_path / "src", "omezarr"), chain, patch=DatasetPatch([8, 16, 16]))
    case.set_memory_budget(20e3)
    with pytest.raises(DatasetManagerError, match="no region of 'CT' fits the per-rank memory budget"):
        case.warm_stream_statistics([0], apply_augmentations=False)


class _MinOfChannel(Transform):
    """Subtracts the least value of channel 1 of its own input, a statistic of the whole volume."""

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        return PatchLocality(LocalityKind.GLOBAL_STAT, stat_keys=frozenset({"Min"}), stat_channels=[1])

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        low = cache_attribute["Min"] if "Min" in cache_attribute else tensor[1].min()
        return tensor - float(low)


def test_a_measured_statistic_reads_the_channels_its_stage_declares(tmp_path) -> None:
    """A stage asking the least value of channel 1 of its own input, behind a fixed clip, gets channel 1's."""
    volume = np.stack([np.full((150, 20, 18), -20.0), np.full((150, 20, 18), 100.0)]).astype(np.float32)
    chain = [Clip(min_value=-50.0, max_value=500.0), _MinOfChannel()]
    case = manager(_store(tmp_path, volume), chain, patch=DatasetPatch([8, 16, 16]))
    case.warm_stream_statistics([0], apply_augmentations=False)
    region = case.read_region((slice(0, 4), slice(0, 4), slice(0, 4)), 0, False)
    np.testing.assert_array_equal(region[1].numpy(), 0.0)


class _HaloLessMin(Transform):
    """A neighbourhood stage that subtracts the least value of its own input, a statistic of the whole volume."""

    def patch_locality(self, cache_attribute: Attribute) -> PatchLocality:
        return PatchLocality(LocalityKind.HALO, halo=(1,), stat_keys=frozenset({"Min"}))

    def __call__(self, name: str, tensor: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        return tensor - float(cache_attribute["Min"] if "Min" in cache_attribute else tensor.min())


def test_a_neighbourhood_stage_reads_the_statistic_measured_on_its_own_input(tmp_path) -> None:
    """A halo stage behind a clip is seeded, as a regrid is, with the least value of the clipped volume."""
    volume = np.full((1, 150, 20, 18), 100.0, dtype=np.float32)
    volume[0, 0, 0, 0] = -20.0
    case = manager(_store(tmp_path, volume), [Clip(-10.0, 500.0), _HaloLessMin()], patch=DatasetPatch([8, 16, 16]))
    case.warm_stream_statistics([0], apply_augmentations=False)
    region = case.read_region((slice(100, 104), slice(0, 4), slice(0, 4)), 0, False)
    np.testing.assert_array_equal(region.numpy(), 110.0)
