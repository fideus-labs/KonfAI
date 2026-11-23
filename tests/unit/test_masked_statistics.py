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

"""Masked whole-volume statistics: the disk scan, and the streamed ``Clip``/``Standardize`` that
seed themselves from it instead of loading two whole volumes per case."""

from pathlib import Path

import numpy as np
import pytest
from konfai.data.transform import Clip, LocalityKind, Standardize
from konfai.utils.dataset import Attribute, Dataset
from konfai.utils.dataset.statistics import read_masked_data_statistics, read_nonzero_data_statistics
from konfai.utils.errors import DatasetManagerError
from oracle_support import (
    CASE_NAME,
    GEOMETRIES,
    MAIN,
    ROUTES,
    Route,
    build_case,
    sweep,
    whole_volume,
)

pytest.importorskip("SimpleITK")  # the fixture case is written as mha


# --------------------------------------------------------------------------------------
# The scan
# --------------------------------------------------------------------------------------


def _masked_pair(root: Path) -> Dataset:
    rng = np.random.default_rng(7)
    dataset = Dataset(root, "h5")
    volume = (rng.standard_normal((1, 13, 9, 11)) * 300.0).astype(np.float32)
    mask = (rng.random((1, 13, 9, 11)) < 0.4).astype(np.uint8)
    dataset.write("CT", "CASE", volume, Attribute())
    dataset.write("MASK", "CASE", mask, Attribute())
    return dataset


def test_the_masked_scan_matches_numpy_over_the_selected_values(tmp_path: Path) -> None:
    dataset = _masked_pair(tmp_path / "data")
    stats = read_masked_data_statistics(dataset, "CT", dataset, "MASK", "CASE")
    volume = dataset.read_data("CT", "CASE")[0]
    selected = volume[dataset.read_data("MASK", "CASE")[0] == 1]
    assert stats["min"] == pytest.approx(float(selected.min()), abs=0.0)
    assert stats["max"] == pytest.approx(float(selected.max()), abs=0.0)
    assert stats["mean"] == pytest.approx(float(selected.mean(dtype=np.float64)), rel=1e-12)
    assert stats["std"] == pytest.approx(float(selected.std(ddof=1, dtype=np.float64)), rel=1e-9)


def test_the_nonzero_scan_is_each_channel_s_population_statistics_over_its_nonzero_voxels(tmp_path: Path) -> None:
    rng = np.random.default_rng(11)
    dataset = Dataset(tmp_path / "data", "h5")
    volume = (rng.standard_normal((3, 13, 9, 11)) * 300.0).astype(np.float32)
    volume[:, :4] = 0.0
    volume[1] = np.where(volume[1] != 0, 5.0, 0.0)
    volume[2] = 0.0
    dataset.write("MR", "CASE", volume, Attribute())
    stats = read_nonzero_data_statistics(dataset, "MR", "CASE")
    selected = volume[0][volume[0] != 0]
    assert stats["mean_per_channel"][0] == pytest.approx(float(selected.mean(dtype=np.float64)), rel=1e-12)
    assert stats["std_per_channel"][0] == pytest.approx(float(selected.std(ddof=0, dtype=np.float64)), rel=1e-9)
    # A deviation of 0 reads 1, and a channel with no non-zero voxel reads 0 and 1: applied to no voxel.
    assert (stats["mean_per_channel"][1], stats["std_per_channel"][1]) == (5.0, 1.0)
    assert (stats["mean_per_channel"][2], stats["std_per_channel"][2]) == (0.0, 1.0)


def test_a_mask_stored_0_255_selects_what_its_0_1_twin_selects(tmp_path: Path) -> None:
    """Any value but 0 is inside: a 0/255 mask, as an 8-bit export writes it, selects its 0/1 twin's voxels."""
    dataset = _masked_pair(tmp_path / "data")
    dataset.write("MASK255", "CASE", dataset.read_data("MASK", "CASE")[0] * 255, Attribute())

    assert read_masked_data_statistics(dataset, "CT", dataset, "MASK255", "CASE") == read_masked_data_statistics(
        dataset, "CT", dataset, "MASK", "CASE"
    )


def test_the_masked_scan_refuses_a_mask_off_the_volume_grid(tmp_path: Path) -> None:
    dataset = _masked_pair(tmp_path / "data")
    small = Dataset(tmp_path / "small", "h5")
    small.write("MASK", "CASE", np.ones((1, 4, 4, 4), dtype=np.uint8), Attribute())
    with pytest.raises(DatasetManagerError, match="grid"):
        read_masked_data_statistics(dataset, "CT", small, "MASK", "CASE")


# --------------------------------------------------------------------------------------
# The declarations
# --------------------------------------------------------------------------------------


def test_masked_stages_declare_the_kind_their_configuration_makes_them() -> None:
    # A masked statistic is GLOBAL_STAT with no stat key: the stage seeds itself from the masked
    # scan, and the dispatcher still guards the seed's validity (stat_seed_valid) and seeds nothing.
    assert Standardize(mask="MASK").patch_locality(Attribute()).kind is LocalityKind.GLOBAL_STAT
    assert not Standardize(mask="MASK").patch_locality(Attribute()).stat_keys
    # Both coefficients given: the mask selects nothing that is read.
    assert Standardize(mean=[0.0], std=[1.0], mask="MASK").patch_locality(Attribute()).kind is LocalityKind.POINTWISE
    # The non-zero voxels are a mask of the volume's own: the same self-seeded kind.
    nonzero = Standardize(nonzero=True, inverse=False).patch_locality(Attribute())
    assert nonzero.kind is LocalityKind.GLOBAL_STAT and not nonzero.stat_keys
    clip = Clip(min_value="min", max_value="max", mask="MASK")
    assert clip.patch_locality(Attribute()).kind is LocalityKind.GLOBAL_STAT
    assert not clip.patch_locality(Attribute()).stat_keys
    # Fixed bounds never read the mask; a percentile needs the whole histogram, mask or not.
    assert Clip(-100.0, 100.0, mask="MASK").patch_locality(Attribute()).kind is LocalityKind.POINTWISE
    assert Clip("percentile:5", 100.0, mask="MASK").patch_locality(Attribute()).kind is LocalityKind.WHOLE_VOLUME


# --------------------------------------------------------------------------------------
# Streamed equals whole-volume
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("route", ROUTES, ids=lambda route: route.name)
def test_a_masked_standardize_streams_and_matches_the_whole_volume(
    route: Route, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The published Synthesis pattern: ``Standardize(mask: ...)`` must stream (the statistic is
    seeded once from the masked disk scan) and produce the whole-volume path's values. The two
    routes accumulate the moments at different widths (float32 over the assembled tensor, float64
    in the scan), so they agree to that rounding rather than bit for bit."""
    dataset = build_case(tmp_path / "case", GEOMETRIES[MAIN])
    got = sweep(dataset, "Intensity", Standardize(mask="Labels"), tmp_path / "streamed", route, monkeypatch)
    want = whole_volume(dataset, "Intensity", Standardize(mask="Labels"), tmp_path / "whole")
    assert got.verdict.name == "STREAM"
    assert got.array.shape == want.array.shape
    np.testing.assert_allclose(got.array, want.array, rtol=0, atol=1e-4)


@pytest.mark.parametrize("route", ROUTES, ids=lambda route: route.name)
def test_a_nonzero_standardize_streams_and_matches_the_whole_volume(
    route: Route, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The non-zero standardization streams the same way: the
    per-channel statistics seeded once from the disk scan over the stored volume's non-zero voxels, the
    zeros kept, within the two routes' rounding (float32 torch reductions against the float64 scan)."""
    dataset = build_case(tmp_path / "case", GEOMETRIES[MAIN])
    volume, attribute = dataset.read_data("Intensity", CASE_NAME)
    volume = np.concatenate([volume, volume * 0.5], axis=0)
    volume[:, : volume.shape[1] // 3] = 0.0
    dataset.write("Intensity", CASE_NAME, volume, attribute)
    stage = Standardize(nonzero=True, inverse=False)
    got = sweep(dataset, "Intensity", stage, tmp_path / "streamed", route, monkeypatch)
    want = whole_volume(dataset, "Intensity", Standardize(nonzero=True, inverse=False), tmp_path / "whole")
    assert got.verdict.name == "STREAM"
    assert got.array.shape == want.array.shape
    np.testing.assert_allclose(got.array, want.array, rtol=0, atol=1e-4)
    assert not got.array[volume == 0].any()


@pytest.mark.parametrize("route", ROUTES, ids=lambda route: route.name)
def test_a_masked_clip_streams_and_matches_the_whole_volume_bit_for_bit(
    route: Route, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Masked ``min``/``max`` bounds are order statistics, exact on both routes: the streamed case
    must equal the whole-volume one bit for bit."""
    dataset = build_case(tmp_path / "case", GEOMETRIES[MAIN])
    got = sweep(
        dataset, "Intensity", Clip(min_value="min", max_value="max", mask="Labels"), tmp_path / "s", route, monkeypatch
    )
    want = whole_volume(dataset, "Intensity", Clip(min_value="min", max_value="max", mask="Labels"), tmp_path / "w")
    assert got.verdict.name == "STREAM"
    np.testing.assert_array_equal(got.array, want.array)
