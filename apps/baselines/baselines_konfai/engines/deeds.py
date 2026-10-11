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

"""deeds (``deedsBCV``) with its defaults: 5 levels, grid 8x7x6x5x4, search radius 8x7x6x5x4, quantisation
5x4x3x2x1, regularisation 1.6, symmetric; built from the authors' repository at a pinned commit on first use. deeds
reads two arrays of one size without geometry, so an anisotropic fixed image is first brought to an isotropic grid,
the moving image onto it, and the field back onto the fixed grid.

NOTE: do NOT add ``from __future__ import annotations``: KonfAI's config engine relies on runtime-evaluated
annotations (``get_origin``); PEP 563 stringized annotations break binding.
"""

import subprocess  # nosec B404
import tempfile
from pathlib import Path
from typing import Annotated

import numpy as np
import SimpleITK as sitk
from konfai.network import network

from .base import BaselineNet, built_source

REPOSITORY = "https://github.com/mattiaspaul/deedsBCV"
COMMIT = "00b46539aa03d23988a241173d8f1f6c0c68dc6a"  # master, 2025-09-09


def _build(source: Path) -> None:
    subprocess.run(["make", "-C", str(source), "linear", "deeds", "applyFloat"], check=True)  # nosec B603 B607


def geometry(image: sitk.Image) -> tuple:
    return image.GetSize(), image.GetSpacing(), image.GetOrigin(), image.GetDirection()


def upsample(coarse: np.ndarray, full: tuple[int, int, int]) -> np.ndarray:
    """deeds' ``upsampleDeformationsCL``: voxel p samples the control grid at p * coarse / full, indices clamped."""
    for axis, (extent, size) in enumerate(zip(full, coarse.shape, strict=True)):
        x = np.arange(extent) * size / extent
        low = np.floor(x).astype(int)
        weight = (x - low).astype(np.float32).reshape([-1 if a == axis else 1 for a in range(3)])
        coarse = (
            np.take(coarse, low, axis) * (1 - weight) + np.take(coarse, np.minimum(low + 1, size - 1), axis) * weight
        )
    return coarse


class Deeds:
    def __init__(self, arguments: list[str]) -> None:
        self._arguments = arguments

    def __call__(
        self,
        fixed: sitk.Image,
        moving: sitk.Image,
        fixed_mask: sitk.Image | None,
        moving_mask: sitk.Image | None,
        device: int,
    ) -> np.ndarray:
        binaries = built_source("deedsBCV", REPOSITORY, COMMIT, _build)
        original, moving = sitk.Cast(fixed, sitk.sitkFloat32), sitk.Cast(moving, sitk.sitkFloat32)
        fixed = original
        spacing = np.array(original.GetSpacing())
        if spacing.max() > 1.01 * spacing.min():
            iso = float(np.prod(spacing) ** (1 / 3))
            size = [int(np.ceil((n - 1) * s / iso)) + 1 for n, s in zip(original.GetSize(), spacing, strict=True)]
            fixed = sitk.Resample(
                original,
                size,
                sitk.Transform(),
                sitk.sitkLinear,
                original.GetOrigin(),
                (iso,) * 3,
                original.GetDirection(),
                float(sitk.GetArrayViewFromImage(original).min()),
            )
        if min(fixed.GetSize()) < 16:
            raise ValueError(f"deeds needs at least 16 voxels per fixed axis (grid 8): {fixed.GetSize()}")
        if geometry(moving) != geometry(fixed):
            moving = sitk.Resample(
                moving, fixed, sitk.Transform(), sitk.sitkLinear, float(sitk.GetArrayViewFromImage(moving).min())
            )
        with tempfile.TemporaryDirectory(prefix="deeds_") as scratch:
            work = Path(scratch)
            sitk.WriteImage(fixed, str(work / "fixed.nii.gz"))
            sitk.WriteImage(moving, str(work / "moving.nii.gz"))
            del moving
            subprocess.run(
                [
                    str(binaries / "deedsBCV"),
                    "-F",
                    str(work / "fixed.nii.gz"),
                    "-M",
                    str(work / "moving.nii.gz"),  # nosec B603
                    "-O",
                    str(work / "deeds"),
                    *self._arguments,
                ],
                check=True,
            )
            flow = np.fromfile(work / "deeds_displacements.dat", np.float32)
        m, n, o = fixed.GetSize()
        step = round((m * n * o / (flow.size // 3)) ** (1 / 3))
        u, v, w = flow.reshape(3, o // step, n // step, m // step)  # arrays as (k, j, i)
        displacement = np.stack([upsample(c, (o, n, m)) for c in (v, u, w)], -1)  # (k, j, i, 3): voxels along i, j, k
        if not np.isfinite(displacement).all():
            raise ValueError("deeds returned a non-finite displacement")
        to_world = np.array(fixed.GetDirection()).reshape(3, 3) @ np.diag(fixed.GetSpacing())
        field = sitk.GetImageFromArray((displacement @ to_world.T).astype(np.float32), isVector=True)
        field.CopyInformation(fixed)
        if fixed is not original:  # world vectors: a plain linear resampling onto the fixed grid
            field = sitk.Resample(field, original, sitk.Transform(), sitk.sitkLinear, 0.0, sitk.sitkVectorFloat32)
        return np.ascontiguousarray(np.moveaxis(sitk.GetArrayFromImage(field), -1, 0), dtype=np.float32)


class RegistrationNet(BaselineNet):
    """deedsBCV, defaults. CPU."""

    def __init__(
        self,
        optimizer: network.OptimizerLoader = network.OptimizerLoader(),
        schedulers: dict[str, network.LRSchedulersLoader] = {
            "default:ReduceLROnPlateau": network.LRSchedulersLoader(0)
        },
        outputs_criterions: dict[str, network.TargetCriterionsLoader] = {"default": network.TargetCriterionsLoader()},
        arguments: Annotated[list[str], "Extra deedsBCV arguments (-G, -L, -Q, -a); none for the defaults."] = [],
    ) -> None:
        super().__init__(Deeds(list(arguments)), 3, optimizer, schedulers, outputs_criterions)
