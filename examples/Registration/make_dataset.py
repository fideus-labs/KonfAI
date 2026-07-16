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

"""Generate a tiny synthetic 2D registration dataset for the KonfAI example.

Each case is a ``FIXED``/``MOVING`` pair. ``MOVING`` is ``FIXED`` translated by a
**known** integer shift, so the task has a ground-truth answer and the moved image
produced by the network can be checked against the fixed image numerically.

All data is procedurally generated (Gaussian blobs) - there is no patient data.
Shapes are tiny (single 64x64 slice per case) so the whole train/predict/evaluate
loop finishes on CPU in a couple of minutes.

Run from this directory with the KonfAI env, for example:

    python make_dataset.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk

# Spatial size of a single slice (Y, X). Kept small so CPU training is fast.
SHAPE: tuple[int, int] = (64, 64)
# Known translation applied to FIXED to build MOVING, expressed in voxels (Y, X).
# The network has to recover this shift to align MOVING back onto FIXED.
KNOWN_SHIFT_YX: tuple[int, int] = (5, 4)
# Number of synthetic cases.
N_CASES: int = 8


def _write(array: np.ndarray, path: Path) -> None:
    """Write a channel-first ``[Z, Y, X]`` array as an .mha volume with unit geometry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    image = sitk.GetImageFromArray(array)  # array is (z, y, x)
    image.SetSpacing((1.0, 1.0, 1.0))
    image.SetOrigin((0.0, 0.0, 0.0))
    sitk.WriteImage(image, str(path))


def _blobs(rng: np.random.Generator, n_blobs: int = 4) -> np.ndarray:
    """Build a smooth [0, 1] intensity image made of a few Gaussian blobs.

    Blob centers stay away from the borders so that shifting by ``KNOWN_SHIFT_YX``
    only brings zero-filled background into view, keeping the translation cleanly recoverable.
    """
    height, width = SHAPE
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    image = np.zeros(SHAPE, dtype=np.float32)
    margin = 18
    for _ in range(n_blobs):
        cy = rng.uniform(margin, height - margin)
        cx = rng.uniform(margin, width - margin)
        sigma = rng.uniform(4.0, 7.0)
        amplitude = rng.uniform(0.5, 1.0)
        image += amplitude * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma**2)))
    # Normalize to [0, 1] so intensities are comparable across cases (no transform needed).
    image -= image.min()
    peak = float(image.max())
    if peak > 0:
        image /= peak
    return image


def make_dataset(base: Path) -> Path:
    """Write ``Dataset/CASE_xxx/{FIXED.mha, MOVING.mha}`` and return the dataset root."""
    rng = np.random.default_rng(0)
    for i in range(N_CASES):
        fixed = _blobs(rng)
        # MOVING is FIXED shifted by the known (positive) translation (axis 0 = Y, axis 1 = X). A
        # zero-filled shift (not np.roll, which would wrap the opposite edge in) keeps MOVING an exact
        # translate of FIXED, so the advertised ground-truth shift is truly recoverable.
        dy, dx = KNOWN_SHIFT_YX
        moving = np.zeros_like(fixed)
        moving[dy:, dx:] = fixed[:-dy, :-dx]
        case = base / f"CASE_{i:03d}"
        # Store as a single-slice volume [Z=1, Y, X] so the [1, 64, 64] patch covers it whole.
        _write(fixed[np.newaxis, ...], case / "FIXED.mha")
        _write(moving[np.newaxis, ...], case / "MOVING.mha")
    return base


if __name__ == "__main__":
    root = make_dataset(Path(__file__).resolve().parent / "Dataset")
    print(f"Wrote {N_CASES} FIXED/MOVING pairs under {root}")
    print(f"Known translation (Y, X) applied to build MOVING: {KNOWN_SHIFT_YX} voxels")
