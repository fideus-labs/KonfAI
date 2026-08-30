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


"""The WRITE side of the contract: a chain materialized slab by slab writes what the whole volume writes.

:mod:`test_transform_locality_contract` proves the read side, patch by patch, for every built-in
transform. TRANSFORM runs take the other route: :meth:`DatasetManager.materialize` sweeps the chain
in slabs into a ``Write`` and never assembles the case. That route has its own machinery (the sweep
rows, the region replay, the stream writers, the header composed from the first slab), so it is
proven separately, over the SAME enumerated cases (``oracle_support``), on the same on-disk
dataset: the streamed store must equal the store the whole-volume path writes, voxel for voxel, on
every device the chain can run on.

STORAGE is this file's axis: every source format's own region reader, every destination the sweep
can write. ``test_streamed_oracle`` runs the same property over the other axes: the dtype, the
rank, the geometry, and the number of regions the case is cut into.
"""

from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk
import torch
from konfai.data.materialize import CaseMaterializer, Verdict
from konfai.data.patching import DatasetManager
from konfai.data.transform import Write
from konfai.utils.dataset import Dataset
from oracle_support import (
    CASE_NAME,
    FIXED_GEOMETRY,
    StageCase,
    attributes,
    build_case,
    manager,
    streamable_cases,
    volumes,
)


@pytest.fixture(scope="session")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> Dataset:
    return build_case(tmp_path_factory.mktemp("materialize") / "Dataset", FIXED_GEOMETRY)


_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
# The formats a case can be READ from region by region -- each its own reader (an ITK extract, a
# gzip stream decoded whole, an h5 slice, a zarr chunk hull) -- plus the one every
# other test uses.
_SOURCE_FORMATS = ["mha", "nii", "nii.gz", "h5", "omezarr"]


@pytest.fixture(scope="session")
def sources(dataset: Dataset, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Dataset]:
    """The read-side dataset, re-written in every source format (same voxels, same headers)."""
    root = tmp_path_factory.mktemp("sources")
    out = {"mha": dataset}
    stored = dataset.read_transform("transform", CASE_NAME)
    for fmt in _SOURCE_FORMATS[1:]:
        copy_root = root / fmt.replace(".", "_")
        copy = Dataset(copy_root, fmt)
        for group, volume in volumes(FIXED_GEOMETRY).items():
            copy.write(group, CASE_NAME, volume, attributes(FIXED_GEOMETRY, group))
        if fmt.startswith("nii"):
            # A stored transform lives beside the case's images, as SimpleITK writes it; the store
            # formats (h5, omezarr) hold displacement fields only, so those cases have no such form.
            sitk.WriteTransform(stored, str(copy_root / CASE_NAME / "transform.itk.txt"))
        out[fmt] = copy
    return out


# The groups whose CASE carries KonfAI attributes beyond the geometry (a channel split, a box).
# NIfTI has no room for them: a case read back from a .nii has lost them, by the format, not by
# the reader -- so those cases have no NIfTI form to test.
_ATTRIBUTE_GROUPS = ("Ensemble", "Boxed")


def _cases() -> list[StageCase]:
    return [case for case in streamable_cases() if case.group != "Field"]


def _manager(dataset: Dataset, case: StageCase, out: Path, fmt: str) -> DatasetManager:
    case.transform.set_datasets([dataset])
    return manager(dataset, [case.transform, Write(f"{out}:{fmt}")], group=case.group)


@pytest.mark.parametrize("fmt", ["mha", "omezarr"])
@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("case", _cases(), ids=lambda case: f"{type(case.transform).__name__}:{case.group}")
def test_streamed_materialize_equals_whole_volume(
    case, device: str, fmt: str, dataset: Dataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _check(case, torch.device(device), fmt, dataset, tmp_path, monkeypatch)


@pytest.mark.parametrize("source", _SOURCE_FORMATS[1:])
@pytest.mark.parametrize("case", _cases(), ids=lambda case: f"{type(case.transform).__name__}:{case.group}")
def test_streamed_materialize_equals_whole_volume_from_every_source_format(
    case, source: str, sources: dict[str, Dataset], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same contract, the case read through each format's own region reader."""
    if source.startswith("nii") and case.group in _ATTRIBUTE_GROUPS:
        pytest.skip("NIfTI carries no attribute beyond the geometry")
    if not source.startswith("nii") and getattr(case.transform, "transforms", None):
        pytest.skip("a store holds displacement fields only, not a stored affine")
    _check(case, torch.device("cpu"), "mha", sources[source], tmp_path, monkeypatch)


def _check(
    case, dev: torch.device, fmt: str, dataset: Dataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Two-row slabs, so the sweep really is several slabs and every seam is exercised.
    monkeypatch.setattr("konfai.data.patching.budget.SWEEP_SLAB_ROWS", 2)
    whole = _manager(dataset, case, tmp_path / "Whole", fmt)
    verdict = CaseMaterializer(whole).materialize(prefer_whole=True, fallback_budget_bytes=1 << 30, device=dev)
    assert verdict is not Verdict.STREAM
    expected, header_expected = Dataset(tmp_path / "Whole", fmt).read_data(case.group, CASE_NAME)
    # Whatever the stage returned, what is on disk is a channel-first volume on the case's geometry.
    assert np.asarray(expected).ndim == len(header_expected.get_np_array("Spacing")) + 1

    streamed = _manager(dataset, case, tmp_path / "Streamed", fmt)
    # A stage the sweep cannot serve (it changes the rank, it needs the volume) says so -- up front
    # or on the first slab -- and the whole-volume path writes the case: the contract then is the
    # shape asserted above, and there is no streamed store to compare.
    if CaseMaterializer(streamed).materialize(fallback_budget_bytes=1 << 30, device=dev) is not Verdict.STREAM:
        pytest.skip(f"took the whole-volume path: {streamed.stream_refusal() or streamed._sweep_failure}")
    got, header_got = Dataset(tmp_path / "Streamed", fmt).read_data(case.group, CASE_NAME)
    assert got.shape == expected.shape
    assert got.dtype == expected.dtype
    np.testing.assert_allclose(np.asarray(got), np.asarray(expected), rtol=0, atol=case.atol)
    for key in ("Origin", "Spacing", "Direction"):
        if key in header_expected:
            np.testing.assert_allclose(
                header_got.get_np_array(key), header_expected.get_np_array(key), rtol=0, atol=1e-9
            )
