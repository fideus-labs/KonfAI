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

"""corrField as the Lung250M-4B authors ship it (``corrfield.py`` at a pinned commit) with its defaults: Foerstner
keypoints inside the fixed image's mask (the lungs), MIND-SSC, discrete matching, thin-plate splines. The pair is
brought to a grid of at least 1 mm and at most 50 million voxels, as the benchmark runs it, and the field back
onto the fixed grid.

NOTE: do NOT add ``from __future__ import annotations``: KonfAI's config engine relies on runtime-evaluated
annotations (``get_origin``); PEP 563 stringized annotations break binding.
"""

import os
import subprocess  # nosec B404
import sys
import tempfile
from pathlib import Path
from typing import Annotated

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from konfai.network import network

from .base import BaselineNet, field_from_voxels, pinned_source

REPOSITORY = "https://github.com/multimodallearning/Lung250M-4B"
COMMIT = "53598ca8f64b13aeb774bc6e6ace486f78d73688"  # main, 2024-01-25


def geometry(image: sitk.Image) -> tuple:
    return image.GetSize(), image.GetSpacing(), image.GetOrigin(), image.GetDirection()


class CorrField:
    def __init__(self, max_voxels: float, min_spacing: float) -> None:
        self._max_voxels, self._min_spacing = max_voxels, min_spacing

    def __call__(
        self,
        fixed: sitk.Image,
        moving: sitk.Image,
        fixed_mask: sitk.Image | None,
        moving_mask: sitk.Image | None,
        device: int,
    ) -> np.ndarray:
        if fixed_mask is None:
            raise ValueError("corrField needs a mask of the fixed image (the lungs): its keypoints are taken inside it")
        script = pinned_source("Lung250M-4B", REPOSITORY, COMMIT) / "corrfield" / "corrfield.py"
        fixed, moving = sitk.Cast(fixed, sitk.sitkFloat32), sitk.Cast(moving, sitk.sitkFloat32)
        mask = sitk.Cast(fixed_mask != 0, sitk.sitkUInt8)
        # the running grid: at least min_spacing, and coarser if the volume exceeds max_voxels
        spacing = np.maximum(fixed.GetSpacing(), self._min_spacing)
        spacing = tuple(
            spacing
            * max(
                1.0, (np.prod(np.array(fixed.GetSize()) * fixed.GetSpacing() / spacing) / self._max_voxels) ** (1 / 3)
            )
        )
        grid = fixed
        if spacing != fixed.GetSpacing():
            grid = sitk.Image(
                [
                    max(1, round(n * s / t))
                    for n, s, t in zip(fixed.GetSize(), fixed.GetSpacing(), spacing, strict=True)
                ],
                sitk.sitkUInt8,
            )
            grid.SetSpacing(spacing), grid.SetOrigin(fixed.GetOrigin()), grid.SetDirection(fixed.GetDirection())

        def on_grid(image: sitk.Image, how: int, outside: float) -> sitk.Image:
            return (
                image
                if geometry(image) == geometry(grid)
                else sitk.Resample(image, grid, sitk.Transform(), how, outside, image.GetPixelID())
            )

        lowest = lambda image: float(sitk.GetArrayViewFromImage(image).min())  # noqa: E731
        fixed_run, moving_run = (
            on_grid(fixed, sitk.sitkLinear, lowest(fixed)),
            on_grid(moving, sitk.sitkLinear, lowest(moving)),
        )
        mask_run = on_grid(mask, sitk.sitkNearestNeighbor, 0.0)
        with tempfile.TemporaryDirectory(prefix="corrfield_") as scratch:
            work = Path(scratch)
            for name, image in {"fixed": fixed_run, "moving": moving_run, "mask": mask_run}.items():
                sitk.WriteImage(image, str(work / f"{name}.nii.gz"))
            environment = {**os.environ, "CUDA_VISIBLE_DEVICES": str(device) if device >= 0 else ""}
            subprocess.run(  # nosec B603
                [
                    sys.executable,
                    str(script),
                    "-F",
                    str(work / "fixed.nii.gz"),
                    "-M",
                    str(work / "moving.nii.gz"),
                    "-m",
                    str(work / "mask.nii.gz"),
                    "-O",
                    str(work / "corrfield"),
                ],
                check=True,
                env=environment,
                timeout=1800,
            )
            displacement = np.asarray(nib.load(str(work / "corrfield.nii.gz")).dataobj)[0]  # (I, J, K, 3), voxels
        if not np.isfinite(displacement).all():
            raise ValueError("corrField returned a non-finite displacement")
        field = sitk.GetImageFromArray(np.moveaxis(field_from_voxels(displacement, grid), 0, -1), isVector=True)
        field.CopyInformation(grid)
        if grid is not fixed:  # back onto the fixed grid, linear, the nearest value past the grid's last sample
            resample = sitk.ResampleImageFilter()
            (
                resample.SetReferenceImage(fixed),
                resample.SetInterpolator(sitk.sitkLinear),
                resample.SetUseNearestNeighborExtrapolator(True),
            )
            field = resample.Execute(field)
        return np.ascontiguousarray(np.moveaxis(sitk.GetArrayFromImage(field), -1, 0), dtype=np.float32)


class RegistrationNet(BaselineNet):
    """corrField with its defaults, on lungs: a mask of the fixed image is required."""

    def __init__(
        self,
        optimizer: network.OptimizerLoader = network.OptimizerLoader(),
        schedulers: dict[str, network.LRSchedulersLoader] = {
            "default:ReduceLROnPlateau": network.LRSchedulersLoader(0)
        },
        outputs_criterions: dict[str, network.TargetCriterionsLoader] = {"default": network.TargetCriterionsLoader()},
        max_voxels: Annotated[
            float, "The pair is brought to a coarser grid when it holds more voxels than this."
        ] = 50e6,
        min_spacing: Annotated[float, "The finest spacing the pair runs at, in mm."] = 1.0,
    ) -> None:
        super().__init__(CorrField(max_voxels, min_spacing), 3, optimizer, schedulers, outputs_criterions)
