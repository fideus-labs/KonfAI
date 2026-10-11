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

"""Demons as SimpleITK's own registration notebook runs it (SimpleITK-Notebooks, 66_Registration_Demons):
fast symmetric forces, 20 iterations, the field smoothed with a standard deviation of 2, over two coarse levels
(shrink 4 and 2, smoothing 8 and 4) then full resolution; for MR, the notebook's histogram matching first. Demons
compares intensities: one modality only.

NOTE: do NOT add ``from __future__ import annotations``: KonfAI's config engine relies on runtime-evaluated
annotations (``get_origin``); PEP 563 stringized annotations break binding.
"""

import os
from typing import Annotated

import numpy as np
import SimpleITK as sitk
from konfai.network import network
from konfai.utils.config import Choices

from .base import BaselineNet


def smooth_and_resample(image: sitk.Image, shrink: float, sigma: float) -> sitk.Image:
    """The notebook's function, verbatim in substance."""
    smoothed = sitk.SmoothingRecursiveGaussian(image, [sigma] * image.GetDimension())
    size = [int(sz / float(shrink) + 0.5) for sz in image.GetSize()]
    spacing = [
        ((sz - 1) * sp) / (new - 1) for sz, sp, new in zip(image.GetSize(), image.GetSpacing(), size, strict=True)
    ]
    return sitk.Resample(
        smoothed,
        size,
        sitk.Transform(),
        sitk.sitkLinear,
        image.GetOrigin(),
        spacing,
        image.GetDirection(),
        0.0,
        image.GetPixelID(),
    )


def multiscale_demons(
    algorithm: sitk.ImageFilter, fixed: sitk.Image, moving: sitk.Image, shrinks: list[int], sigmas: list[float]
) -> sitk.Image:
    """The notebook's multiscale loop: the field found at a coarse level starts the next, finer one."""
    size = [int(sz / float(shrinks[0]) + 0.5) for sz in fixed.GetSize()]
    spacing = [
        ((sz - 1) * sp) / (new - 1) for sz, sp, new in zip(fixed.GetSize(), fixed.GetSpacing(), size, strict=True)
    ]
    field = sitk.Image(size, sitk.sitkVectorFloat64, fixed.GetDimension())
    field.SetSpacing(spacing)
    field.SetOrigin(fixed.GetOrigin())
    pairs = [
        (smooth_and_resample(fixed, s, g), smooth_and_resample(moving, s, g))
        for s, g in zip(shrinks, sigmas, strict=True)
    ]
    for f, m in [*pairs, (fixed, moving)]:
        field = sitk.Resample(field, f)
        field = algorithm.Execute(f, m, field)
    return field


class Demons:
    def __init__(
        self,
        iterations: int,
        standard_deviation: float,
        shrinks: list[int],
        sigmas: list[float],
        modality: str,
        threads: int,
    ) -> None:
        self._iterations, self._standard_deviation = iterations, standard_deviation
        self._shrinks, self._sigmas, self._modality, self._threads = shrinks, sigmas, modality, threads

    def __call__(
        self,
        fixed: sitk.Image,
        moving: sitk.Image,
        fixed_mask: sitk.Image | None,
        moving_mask: sitk.Image | None,
        device: int,
    ) -> np.ndarray:
        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(
            self._threads or int(os.environ.get("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "8"))
        )
        fixed, moving = sitk.Cast(fixed, sitk.sitkFloat32), sitk.Cast(moving, sitk.sitkFloat32)
        moving = sitk.Resample(
            moving, fixed, sitk.Transform(), sitk.sitkLinear, float(sitk.GetArrayViewFromImage(moving).min())
        )
        modality = (
            self._modality
            if self._modality != "auto"
            else ("ct" if float(sitk.GetArrayViewFromImage(fixed).min()) < -200 else "mr")
        )
        if modality == "mr":  # the notebook's histogram matching: Demons assumes equal intensities, MR breaks that
            matcher = sitk.HistogramMatchingImageFilter()
            matcher.SetNumberOfHistogramLevels(1024)
            matcher.SetNumberOfMatchPoints(7)
            matcher.ThresholdAtMeanIntensityOn()
            moving = matcher.Execute(moving, fixed)
        demons = sitk.FastSymmetricForcesDemonsRegistrationFilter()
        demons.SetNumberOfIterations(self._iterations)
        demons.SetSmoothDisplacementField(True)
        demons.SetStandardDeviations(self._standard_deviation)
        field = multiscale_demons(demons, fixed, moving, self._shrinks, self._sigmas)
        return np.ascontiguousarray(np.moveaxis(sitk.GetArrayFromImage(field), -1, 0), dtype=np.float32)


class RegistrationNet(BaselineNet):
    """Demons, the SimpleITK notebook's settings."""

    def __init__(
        self,
        optimizer: network.OptimizerLoader = network.OptimizerLoader(),
        schedulers: dict[str, network.LRSchedulersLoader] = {
            "default:ReduceLROnPlateau": network.LRSchedulersLoader(0)
        },
        outputs_criterions: dict[str, network.TargetCriterionsLoader] = {"default": network.TargetCriterionsLoader()},
        iterations: Annotated[int, "Demons iterations at each level."] = 20,
        standard_deviation: Annotated[float, "Standard deviation of the field's smoothing, in voxels."] = 2.0,
        shrinks: Annotated[list[int], "Shrink factor of each coarse level, before full resolution."] = [4, 2],
        sigmas: Annotated[list[float], "Smoothing of each coarse level, in voxels."] = [8.0, 4.0],
        modality: Annotated[
            str,
            Choices(["auto", "ct", "mr"]),
            "mr matches the moving histogram to the fixed first; auto reads it off the intensities.",
        ] = "auto",
        threads: Annotated[int, "ITK threads; 0 takes ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS or 8."] = 0,
    ) -> None:
        super().__init__(
            Demons(iterations, standard_deviation, shrinks, sigmas, modality, threads),
            3,
            optimizer,
            schedulers,
            outputs_criterions,
        )
