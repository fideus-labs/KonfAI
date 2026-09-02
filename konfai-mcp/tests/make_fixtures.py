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
"""Generate small synthetic fixtures for practical konfai-mcp validation runs.

All data is procedurally generated — no patient data. Shapes are tiny so CPU
train/predict/evaluate loops finish in seconds. Output lands in the gitignored
``konfai-mcp/tests/fixtures/`` directory. Run in the KonfAI dev env:

    pixi run --environment dev python konfai-mcp/tests/make_fixtures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk

ROOT = Path(__file__).resolve().parent / "fixtures"


def _write(arr: np.ndarray, path: Path, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0), direction=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = sitk.GetImageFromArray(arr)  # arr is (z, y, x)
    img.SetSpacing(spacing)
    img.SetOrigin(origin)
    if direction is not None:
        img.SetDirection(direction)
    sitk.WriteImage(img, str(path))


def _sphere(shape, center, radius) -> np.ndarray:
    zz, yy, xx = np.ogrid[: shape[0], : shape[1], : shape[2]]
    d2 = (zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2
    return d2 <= radius**2


def make_segmentation_dataset(n_cases=4, shape=(16, 32, 32)) -> Path:
    """Dataset/CASE_xxx/{CT.mha, SEG.mha}. Two foreground labels (sphere=1, cube=2) on background=0."""
    base = ROOT / "seg_ds" / "Dataset"
    rng = np.random.default_rng(0)
    for i in range(n_cases):
        c = (shape[0] // 2, shape[1] // 2 + (i - 2) * 2, shape[2] // 2)
        seg = np.zeros(shape, dtype=np.int16)
        seg[_sphere(shape, c, 6)] = 1
        seg[4:8, 4:12, 20:28] = 2  # a cube = label 2
        ct = (seg.astype(np.float32) * 300.0) + rng.normal(0, 20, shape).astype(np.float32) - 100.0
        case = base / f"CASE_{i:03d}"
        _write(ct.astype(np.float32), case / "CT.mha")
        _write(seg, case / "SEG.mha")
    return base


def make_registration_pair(shape=(24, 48, 48), shift=(0, 5, 0)) -> Path:
    """Fixed/moving pair with a KNOWN integer translation (moving = fixed shifted by `shift`)."""
    base = ROOT / "reg_pair"
    fixed = np.zeros(shape, dtype=np.float32)
    fixed[_sphere(shape, (12, 24, 24), 8)] = 500.0
    fixed[6:10, 10:16, 30:38] = 300.0  # asymmetric feature so translation is recoverable
    moving = np.roll(fixed, shift=shift, axis=(0, 1, 2))
    _write(fixed, base / "fixed.nii.gz")
    _write(moving, base / "moving.nii.gz")
    # Same content but different spacing/origin, to test geometry mismatches.
    _write(fixed, base / "fixed_spacing2.nii.gz", spacing=(2.0, 2.0, 2.0), origin=(10.0, -5.0, 3.0))
    (base / "known_transform.json").write_text(
        json.dumps({"type": "translation_voxels_zyx", "shift": list(shift)}), encoding="utf-8"
    )
    return base


def make_synthesis_pair(n_cases=3, shape=(16, 32, 32)) -> Path:
    """Dataset/CASE_xxx/{MR.mha, CT.mha} for an MR->CT synthesis task."""
    base = ROOT / "synth_ds" / "Dataset"
    rng = np.random.default_rng(1)
    for i in range(n_cases):
        struct = np.zeros(shape, dtype=np.float32)
        struct[_sphere(shape, (8, 16, 16), 7)] = 1.0
        mr = struct * 800.0 + rng.normal(0, 30, shape).astype(np.float32)
        ct = struct * 200.0 - 100.0 + rng.normal(0, 10, shape).astype(np.float32)
        case = base / f"CASE_{i:03d}"
        _write(mr.astype(np.float32), case / "MR.mha")
        _write(ct.astype(np.float32), case / "CT.mha")
    return base


def make_nrrd_and_dicom_variants() -> None:
    """A NRRD copy and an incompatible-shape image, to test format/geometry handling."""
    misc = ROOT / "misc"
    a = np.zeros((10, 20, 20), dtype=np.float32)
    a[_sphere((10, 20, 20), (5, 10, 10), 4)] = 1.0
    _write(a, misc / "shape_10x20x20.nrrd")
    _write(a, misc / "shape_10x20x20.nii.gz")
    b = np.zeros((8, 24, 24), dtype=np.float32)  # different shape -> pair mismatch
    _write(b, misc / "shape_8x24x24.nii.gz")


def make_bad_fixtures() -> None:
    """Corrupted / unsupported / empty inputs for failure-handling scenarios."""
    bad = ROOT / "bad"
    bad.mkdir(parents=True, exist_ok=True)
    (bad / "corrupted.nii.gz").write_bytes(b"\x1f\x8b\x08\x00not-a-real-nifti-gzip-body")
    (bad / "not_an_image.txt").write_text("this is plainly not a medical image\n", encoding="utf-8")
    (bad / "empty.mha").write_bytes(b"")


def make_ome_zarr() -> None:
    """A tiny 3-level multiscale OME-Zarr store, to probe large-image/streaming behavior."""
    try:
        import zarr  # noqa: F401
    except Exception as exc:  # pragma: no cover
        (ROOT / "omezarr_SKIPPED.txt").write_text(f"zarr not available: {exc}\n", encoding="utf-8")
        return
    try:
        # Prefer ngff-zarr if present (KonfAI's OME-Zarr backend); else fall back to a hand-rolled store.
        import ngff_zarr as nz

        arr = np.zeros((64, 128, 128), dtype=np.float32)
        arr[_sphere((64, 128, 128), (32, 64, 64), 20)] = 1.0
        image = nz.to_ngff_image(arr, dims=["z", "y", "x"], scale={"z": 1.0, "y": 1.0, "x": 1.0})
        multiscales = nz.to_multiscales(image, scale_factors=[2, 4])
        nz.to_ngff_zarr(str(ROOT / "large.zarr"), multiscales)
        (ROOT / "omezarr_backend.txt").write_text("ngff_zarr\n", encoding="utf-8")
    except Exception as exc:  # pragma: no cover
        (ROOT / "omezarr_SKIPPED.txt").write_text(f"ngff_zarr failed: {exc}\n", encoding="utf-8")


if __name__ == "__main__":
    seg = make_segmentation_dataset()
    reg = make_registration_pair()
    syn = make_synthesis_pair()
    make_nrrd_and_dicom_variants()
    make_bad_fixtures()
    make_ome_zarr()
    print("seg dataset:", seg)
    print("reg pair:", reg)
    print("synth dataset:", syn)
    print("fixtures root:", ROOT)
    for p in sorted(ROOT.rglob("*")):
        if p.is_file():
            print("  ", p.relative_to(ROOT), p.stat().st_size, "bytes")
