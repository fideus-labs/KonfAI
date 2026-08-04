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

"""Build the FIXED/MOVING dataset for the KonfAI registration example.

Each case is one **real** axial slice from the public pelvis CT demo subset
(``VBoussot/konfai-demo``). ``FIXED`` is the slice as acquired; ``MOVING`` is the same slice
pushed through a smooth displacement field that this script chooses, so the deformation the
network has to recover is known exactly and the registered output can be checked numerically.

Real anatomy, known answer. Inter-patient registration — where the deformation is a genuine
anatomical difference and there is no ground-truth field at all — is what ``examples/ImpactReg``
does, scored on the reference segmentations.

Run from this directory with the KonfAI env:

    python make_dataset.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk
from huggingface_hub import snapshot_download
from scipy.ndimage import map_coordinates

# Axial slices taken per patient, spread over the volume.
SLICES_PER_CASE: int = 6
# Peak amplitude of the displacement field used to build MOVING, in voxels.
AMPLITUDE: float = 8.0
# Hounsfield window the slices are normalised through, so intensities are comparable across cases.
WINDOW: tuple[float, float] = (-160.0, 240.0)
# Every slice is cropped to this size around the body, so one patch covers a case whole and no
# sample is mostly padding. It is also the `shape` VoxelMorph is built with in Config.yml.
CROP: tuple[int, int] = (256, 256)


def _source_cases() -> list[Path]:
    """Fetch the public pelvis CT subset (cached by the Hub) and return one directory per patient."""
    root = Path(snapshot_download("VBoussot/konfai-demo", repo_type="dataset", allow_patterns="Segmentation/**"))
    return sorted(path for path in (root / "Segmentation").iterdir() if path.is_dir())


def _displacement(shape: tuple[int, int], seed: int) -> tuple[np.ndarray, np.ndarray]:
    """A smooth, low-frequency (dy, dx) field of peak amplitude ``AMPLITUDE``.

    Built from a handful of sine terms rather than filtered noise so it is reproducible from the
    seed alone, and smooth enough for a diffeomorphic model to represent.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]].astype(np.float32)
    field = []
    for _ in range(2):
        wave = np.zeros(shape, dtype=np.float32)
        for _ in range(3):
            wavelength = rng.uniform(60.0, 140.0)
            phase = rng.uniform(0.0, 2.0 * np.pi)
            direction = rng.uniform(0.0, np.pi)
            wave += np.sin(2 * np.pi * (np.cos(direction) * xx + np.sin(direction) * yy) / wavelength + phase)
        field.append(AMPLITUDE * wave / np.abs(wave).max())
    return field[0], field[1]


def _slice_indices(volume: np.ndarray) -> list[int]:
    """Pick slices spread over the part of the volume that actually contains a body."""
    occupied = np.flatnonzero((volume > WINDOW[0]).sum(axis=(1, 2)) > 0.05 * volume[0].size)
    return [int(i) for i in np.linspace(occupied[0], occupied[-1], SLICES_PER_CASE + 2)[1:-1]]


def _crop(image: np.ndarray) -> np.ndarray:
    """Crop to ``CROP`` around the centre of mass of the body, padding if the slice is smaller."""
    padded = np.pad(image, [(max(0, c - s), max(0, c - s)) for s, c in zip(image.shape, CROP, strict=True)])
    weights = padded > 0.05
    centre = [
        int(np.clip(np.average(np.flatnonzero(weights.any(axis=1 - axis))), c // 2, padded.shape[axis] - c // 2))
        for axis, c in enumerate(CROP)
    ]
    return padded[
        centre[0] - CROP[0] // 2 : centre[0] + CROP[0] // 2, centre[1] - CROP[1] // 2 : centre[1] + CROP[1] // 2
    ]


def _write(array: np.ndarray, path: Path) -> None:
    """Write a ``[Z, Y, X]`` array as an .mha volume with unit geometry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((1.0, 1.0, 1.0))
    image.SetOrigin((0.0, 0.0, 0.0))
    sitk.WriteImage(image, str(path))


def make_dataset(base: Path) -> int:
    """Write ``Dataset/<patient>_z<slice>/{FIXED.mha, MOVING.mha}`` and return the number of pairs."""
    written = 0
    for case in _source_cases():
        volume = sitk.GetArrayFromImage(sitk.ReadImage(str(case / "CT.mha"))).astype(np.float32)
        for index in _slice_indices(volume):
            fixed = _crop(np.clip((volume[index] - WINDOW[0]) / (WINDOW[1] - WINDOW[0]), 0.0, 1.0))
            dy, dx = _displacement(fixed.shape, seed=written)
            grid = np.mgrid[0 : fixed.shape[0], 0 : fixed.shape[1]].astype(np.float32)
            moving = map_coordinates(fixed, [grid[0] + dy, grid[1] + dx], order=1, mode="nearest")
            target = base / f"{case.name}_z{index:03d}"
            # One slice per case, stored as [Z=1, Y, X].
            _write(fixed[np.newaxis, ...], target / "FIXED.mha")
            _write(moving.astype(np.float32)[np.newaxis, ...], target / "MOVING.mha")
            written += 1
    return written


if __name__ == "__main__":
    root = Path(__file__).resolve().parent / "Dataset"
    count = make_dataset(root)
    print(f"Wrote {count} FIXED/MOVING pairs of real CT slices under {root}")
    print(f"MOVING is FIXED pushed through a known smooth field of up to {AMPLITUDE:.0f} voxels")
