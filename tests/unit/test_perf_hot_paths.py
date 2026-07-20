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

"""Regression tests for the performance hot-path fixes (see AUDIT.md — Performance backlog)."""

from pathlib import Path
from unittest.mock import MagicMock

import konfai.utils.dataset as dataset_module
import torch
from konfai.data.patching import Accumulator
from konfai.predictor import ModelComposite
from konfai.utils.dataset import Attribute, Dataset


def test_accumulator_is_full_counts_without_rescanning():
    """P6: is_full() is O(1) — a running counter, idempotent adds, reset on assemble."""
    patch_slices = [(slice(0, 2), slice(0, 2)), (slice(0, 2), slice(2, 4))]
    acc = Accumulator(patch_slices, [0, 0], None, batch=False)

    assert acc.is_full() is False
    acc.add_layer(0, torch.ones(1, 2, 2))
    assert acc.is_full() is False
    acc.add_layer(1, torch.ones(1, 2, 2) * 2)
    assert acc.is_full() is True

    # Re-adding the same index must not double-count.
    acc.add_layer(1, torch.ones(1, 2, 2) * 3)
    assert acc._filled == 2
    assert acc.is_full() is True

    acc.assemble()
    assert acc._filled == 0
    assert acc.is_full() is False


def test_ensemble_reads_each_checkpoint_once_across_batches():
    """P1: a local-path ensemble reads/unpickles each checkpoint once, not once per batch."""
    mc = ModelComposite.__new__(ModelComposite)
    mc._base_model_name = "Model"
    mc._state_sources = [Path("/fake/ckpt_0.pt"), Path("/fake/ckpt_1.pt"), Path("/fake/ckpt_2.pt")]
    mc._loaded_state_index = None
    mc._state_cache = {}
    mc._get_model = lambda: MagicMock()

    reads: list[str] = []

    def fake_read(src):
        reads.append(str(src))
        return {"w": str(src)}

    mc._read_state_source = fake_read

    # Four forward passes, each looping over all three sub-models (as forward() does).
    for _batch in range(4):
        for idx in range(3):
            mc._ensure_model_loaded(idx)

    assert len(reads) == 3, f"expected 3 disk reads (one per index), got {len(reads)}"
    # Compare via str(Path(...)) so the expected separators match the platform
    # (the reads store str(src); Windows renders these with backslashes).
    assert set(reads) == {str(Path(f"/fake/ckpt_{i}.pt")) for i in range(3)}

    # load() must invalidate the stale cache when the sources change.
    mc.load([Path("/other.pt")])
    assert 1 not in mc._state_cache and 2 not in mc._state_cache
    assert mc._state_cache.get(0) == {"w": str(Path("/other.pt"))}


def test_get_infos_is_memoized_and_returns_independent_copies(monkeypatch):
    """P4: the header read is cached per (group, name); results are copies (no aliasing)."""
    ds = Dataset.__new__(Dataset)
    ds.is_directory = False
    ds.filename = "/fake/ds"
    ds.file_format = "sitk"
    ds.level = 0
    ds._names_cache = {}
    ds._infos_cache = {}

    opens = {"n": 0}

    class _FakeFile:
        def __init__(self, *args, **kwargs):
            opens["n"] += 1

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def get_infos(self, groups, name):
            return [4, 5, 6], Attribute({"Spacing": "1.0 1.0 1.0"})

    monkeypatch.setattr(dataset_module.Dataset, "File", _FakeFile)

    first = ds.get_infos("g", "n")
    assert opens["n"] == 1
    second = ds.get_infos("g", "n")
    assert opens["n"] == 1, "second call must be served from cache, not re-open the file"
    assert first[0] == second[0] == [4, 5, 6]

    # Copies, not aliases: mutating a returned result must not poison the cache.
    first[0].append(999)
    third = ds.get_infos("g", "n")
    assert third[0] == [4, 5, 6]

    # A write invalidates the cache (mirrors get_names).
    ds._infos_cache.clear()  # write() calls this
    ds.get_infos("g", "n")
    assert opens["n"] == 2, "after invalidation the header is read again"


def test_dicom_slice_info_threading_is_byte_identical_and_removes_rescans(tmp_path, monkeypatch):
    """P2: threading get_dicom_info's sorted files stays byte-identical while removing re-scans/re-sorts."""
    import numpy as np
    import pytest

    pytest.importorskip("pydicom")
    from konfai.utils import dicom
    from konfai.utils.errors import DatasetManagerError

    root = tmp_path / "case"
    vol = (np.arange(1 * 4 * 5 * 6).reshape(1, 4, 5, 6) % 97).astype(np.float32)
    dicom.write_dicom_series(root, vol, origin=(1.0, 2.0, 3.0), spacing=(0.7, 0.8, 2.5), direction=np.eye(3).flatten())

    # (a) byte-identical: standalone (info=None) path == threaded (info precomputed) path
    sl = (slice(None), slice(1, 3), slice(None), slice(None))
    ref = dicom.read_dicom_series_slice(root, sl)
    info = dicom.get_dicom_info(root)
    got = dicom.read_dicom_series_slice(root, sl, series_uid=info["series_uid"], info=info)
    for a, b in zip(ref, got, strict=False):
        assert np.array_equal(np.asarray(a), np.asarray(b))

    assert len(info["sorted_files"]) == vol.shape[1]
    assert all(isinstance(p, Path) for p in info["sorted_files"])

    # a mismatching series_uid must not silently read the wrong series
    with pytest.raises(DatasetManagerError):
        dicom.read_dicom_series_slice(root, sl, series_uid="9.9.9.mismatch", info=info)

    # arity-mismatch path (info=None) still raises before any file selection
    with pytest.raises(DatasetManagerError):
        dicom.read_dicom_series_slice(root, (slice(None), slice(0, 2)))

    # (b) redundant work is gone: spy discover_series / sort_series call counts
    calls = {"discover": 0, "sort": 0}
    real_discover, real_sort = dicom.discover_series, dicom.sort_series
    monkeypatch.setattr(
        dicom,
        "discover_series",
        lambda *a, **k: (calls.__setitem__("discover", calls["discover"] + 1), real_discover(*a, **k))[1],
    )
    monkeypatch.setattr(
        dicom, "sort_series", lambda *a, **k: (calls.__setitem__("sort", calls["sort"] + 1), real_sort(*a, **k))[1]
    )

    dataset_file = Dataset.DicomFile(str(tmp_path), read=True)

    # one COLD patch read: 1 discovery + 2 sorts (pre-fix: 3 discoveries + 4 sorts). get_dicom_info
    # is memoised, so the cache must be cleared for the spy to see the cold cost at all.
    calls["discover"] = calls["sort"] = 0
    dicom.get_dicom_info.cache_clear()
    data, _attr = dataset_file.file_to_data_slice("", "case", sl)
    assert np.array_equal(np.asarray(data), np.asarray(ref[0]))
    assert calls["discover"] == 1
    assert calls["sort"] == 2

    # a WARM read of the same case re-discovers nothing; only the per-read slab sort (pixel
    # loading of the selected files) remains
    calls["discover"] = calls["sort"] = 0
    data, _attr = dataset_file.file_to_data_slice("", "case", sl)
    assert np.array_equal(np.asarray(data), np.asarray(ref[0]))
    assert calls["discover"] == 0
    assert calls["sort"] == 1

    # statistics over Z: 1 cold discovery (pre-fix: O(Z)); numerics preserved (Welford, ddof=1)
    calls["discover"] = calls["sort"] = 0
    dicom.get_dicom_info.cache_clear()
    stats = dataset_file.file_to_data_statistics("", "case")
    assert calls["discover"] == 1
    assert np.isclose(stats["mean"], float(vol.mean()), atol=1e-4)
    assert np.isclose(stats["std"], float(np.std(vol, ddof=1)), atol=1e-4)
    assert np.isclose(stats["min"], float(vol.min()), atol=1e-4)
    assert np.isclose(stats["max"], float(vol.max()), atol=1e-4)


def _old_clip(tensor, lo, hi):
    """The pre-fix Clip inner logic (float()-cast where-scatter), for byte-identity checks."""
    t = tensor.clone()
    t[torch.where(t.float() < lo)] = lo
    t[torch.where(t.float() > hi)] = hi
    return t


def _eq_nan(a, b):
    return bool(((a == b) | (a.isnan() & b.isnan())).all())


def test_clip_clamp_fast_path_is_byte_identical_on_float32_and_safe_on_int():
    """Perf batch: float32 clamp_ fast path is byte-identical; int/float64 keep the old scatter."""
    from konfai.data.transform import Clip
    from konfai.utils.dataset import Attribute

    clip = Clip(min_value=-5.0, max_value=5.0)
    lo, hi = -5.0, 5.0

    # float32 with the hostile edge cases the red-team flagged: NaN, +/-inf, exact bounds
    f32 = torch.tensor(
        [-1e9, -5.0, -2.0, 0.0, 2.0, 5.0, 1e9, float("nan"), float("inf"), float("-inf")], dtype=torch.float32
    )
    got = clip("x", f32.clone(), Attribute())
    assert _eq_nan(got, _old_clip(f32, lo, hi))
    assert got.dtype == torch.float32

    # int16 (CT-style) must NOT crash and must equal the old scatter (else-branch)
    i16 = torch.tensor([[-2000, -5, 0, 5, 2000]], dtype=torch.int16)
    got_i = clip("x", i16.clone(), Attribute())
    assert torch.equal(got_i, _old_clip(i16, lo, hi))
    assert got_i.dtype == torch.int16

    # float64 keeps the legacy float()-cast comparison path (else-branch), unchanged
    f64 = torch.tensor([-9.0, -5.0, 0.0, 5.0, 9.0], dtype=torch.float64)
    got_d = clip("x", f64.clone(), Attribute())
    assert _eq_nan(got_d, _old_clip(f64, lo, hi))
    assert got_d.dtype == torch.float64


def test_clip_float32_nan_dynamic_bound_does_not_corrupt_volume():
    """Review catch: a dynamic bound resolving to NaN must NOT turn the whole float32 volume to NaN."""
    from konfai.data.transform import Clip
    from konfai.utils.dataset import Attribute

    # data contains a NaN voxel -> min()/max() resolve to NaN bounds
    data = torch.tensor([1.0, 2.0, float("nan"), 3.0], dtype=torch.float32)
    got = Clip(min_value="min", max_value="max")("x", data.clone(), Attribute())
    # legacy behaviour: NaN comparisons are False, so the scatter is a no-op (values preserved)
    assert not bool(got.isnan().all()), "must not become all-NaN"
    assert got[0] == 1.0 and got[1] == 2.0 and got[3] == 3.0
    assert bool(got[2].isnan())
