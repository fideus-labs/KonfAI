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

"""ANTs SyN as the Learn2Reg 2024 organisers ran it for LUMIR, ``ants.registration(..., "antsRegistrationSyN[so]")``:
the ``antsRegistration`` command ANTsPy builds for it, on the ANTs release binaries at a pinned version. SyN alone
from the identity, cross-correlation of radius 2, 100x70x50x20 iterations at shrink factors 8x4x2x1, single precision,
a fixed random seed; then ``antsApplyTransforms`` writes the field on the fixed grid. Modalities and masks are
accepted and ignored: the organisers' call uses neither.

NOTE: do NOT add ``from __future__ import annotations``: KonfAI's config engine relies on runtime-evaluated
annotations (``get_origin``); PEP 563 stringized annotations break binding.
"""

import os
import subprocess  # nosec B404
import tempfile
from pathlib import Path
from typing import Annotated

import numpy as np
import SimpleITK as sitk
from konfai.network import network

from .base import BaselineNet, pinned_archive

VERSION = "2.6.5"
RELEASE = (
    "https://github.com/ANTsX/ANTs/releases/download/v2.6.5/ants-2.6.5-ubuntu-24.04-X64-gcc.zip",
    "9f6a2c8881d6c94ea17171c0502e55cb898ad9afd50e4e23fde9a8c0997aa04b",
)


def binaries() -> Path:
    """The release's ``bin/``, fetched once."""
    return pinned_archive("ants", *RELEASE) / f"ants-{VERSION}"


class AntsSyN:
    def __init__(
        self,
        transform: str,
        metric: str,
        convergence: str,
        shrink_factors: str,
        smoothing_sigmas: str,
        seed: int,
        threads: int,
    ) -> None:
        self._transform, self._metric, self._convergence = transform, metric, convergence
        self._shrink, self._smoothing, self._seed, self._threads = shrink_factors, smoothing_sigmas, seed, threads

    def __call__(
        self,
        fixed: sitk.Image,
        moving: sitk.Image,
        fixed_mask: sitk.Image | None,
        moving_mask: sitk.Image | None,
        device: int,
    ) -> np.ndarray:
        root = binaries()
        environment = {
            **os.environ,
            "PATH": f"{root / 'bin'}:{os.environ.get('PATH', '')}",
            "LD_LIBRARY_PATH": f"{root / 'lib'}:{os.environ.get('LD_LIBRARY_PATH', '')}",
            "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS": str(self._threads or os.cpu_count() or 1),
        }
        with tempfile.TemporaryDirectory(prefix="ants_") as scratch:
            work = Path(scratch)
            sitk.WriteImage(fixed, str(work / "fixed.nii.gz"))
            sitk.WriteImage(moving, str(work / "moving.nii.gz"))
            metric = self._metric.replace("FIXED", str(work / "fixed.nii.gz")).replace(
                "MOVING", str(work / "moving.nii.gz")
            )
            subprocess.run(  # nosec B603 B607
                [
                    "antsRegistration",
                    "--dimensionality",
                    "3",
                    "-r",
                    "Identity",
                    "--output",
                    str(work / "ants"),
                    "--transform",
                    self._transform,
                    "--metric",
                    metric,
                    "--convergence",
                    self._convergence,
                    "--shrink-factors",
                    self._shrink,
                    "--smoothing-sigmas",
                    self._smoothing,
                    "-x",
                    "[NA,NA]",
                    "--random-seed",
                    str(self._seed),
                    "--float",
                    "1",
                    "--write-composite-transform",
                    "0",
                ],
                check=True,
                env=environment,
            )
            # the warp's number follows the stages, the identity initialisation counted or not as the version chooses
            warp = next(p for p in sorted(work.glob("ants*Warp.nii.gz")) if "InverseWarp" not in p.name)
            subprocess.run(  # nosec B603 B607
                [
                    "antsApplyTransforms",
                    "-d",
                    "3",
                    "-r",
                    str(work / "fixed.nii.gz"),
                    "-t",
                    str(warp),
                    "-o",
                    f"DisplacementField[{work / 'field.nii.gz'}]",
                    "--float",
                    "1",
                ],
                check=True,
                env=environment,
            )
            field = sitk.ReadImage(str(work / "field.nii.gz"), sitk.sitkVectorFloat32)
        return np.ascontiguousarray(np.moveaxis(sitk.GetArrayFromImage(field), -1, 0), dtype=np.float32)


class RegistrationNet(BaselineNet):
    """ANTs SyN, the Learn2Reg 2024 organisers' call. CPU."""

    def __init__(
        self,
        optimizer: network.OptimizerLoader = network.OptimizerLoader(),
        schedulers: dict[str, network.LRSchedulersLoader] = {
            "default:ReduceLROnPlateau": network.LRSchedulersLoader(0)
        },
        outputs_criterions: dict[str, network.TargetCriterionsLoader] = {"default": network.TargetCriterionsLoader()},
        transform: Annotated[
            str, "antsRegistration's --transform: SyN[gradient step, update field variance, total field variance]."
        ] = "SyN[0.2,3,0]",
        metric: Annotated[
            str,
            "antsRegistration's --metric, FIXED and MOVING standing for the images: CC[fixed, moving, weight, radius].",
        ] = "CC[FIXED,MOVING,1,2]",
        convergence: Annotated[
            str, "antsRegistration's --convergence: iterations per level, threshold, window."
        ] = "[100x70x50x20,1e-6,10]",
        shrink_factors: Annotated[str, "antsRegistration's --shrink-factors."] = "8x4x2x1",
        smoothing_sigmas: Annotated[str, "antsRegistration's --smoothing-sigmas."] = "3x2x1x0vox",
        seed: Annotated[int, "antsRegistration's --random-seed, the one setting not in ANTsPy's call."] = 1,
        threads: Annotated[int, "ITK threads; 0 takes every core."] = 0,
    ) -> None:
        super().__init__(
            AntsSyN(transform, metric, convergence, shrink_factors, smoothing_sigmas, seed, threads),
            3,
            optimizer,
            schedulers,
            outputs_criterions,
        )
