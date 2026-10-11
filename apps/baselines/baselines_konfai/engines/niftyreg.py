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

"""NiftyReg's ``reg_f3d`` with its defaults (cubic B-spline with control points every 5 voxels, normalised mutual
information with 64 bins, bending energy 0.001, first-order penalty 0.01, no Jacobian penalty, 3 levels), built from
the authors' repository at a pinned commit (CPU build) on first use; ``reg_transform`` turns the control points into
a displacement field, whose RAS axes are brought to ITK's LPS.

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

REPOSITORY = "https://github.com/KCL-BMEIS/niftyreg"
COMMIT = "ee6a92d71263a4b1eb29b829fdeb3961d26d8a13"  # v2.0.0, 2025-11-21


def _build(source: Path) -> None:
    """cmake, CPU only, into ``install/`` next to the sources: what the benchmark's setup.sh does."""
    build, install = source.parent / f"{source.name}-build", source / "install"
    subprocess.run(
        [
            "cmake",
            "-S",
            str(source),
            "-B",
            str(build),
            "-DCMAKE_BUILD_TYPE=Release",  # nosec B603 B607
            f"-DCMAKE_INSTALL_PREFIX={install}",
            "-DUSE_CUDA=OFF",
            "-DUSE_OPENCL=OFF",
            "-DCHECK_GPU=OFF",
        ],
        check=True,
    )
    subprocess.run(["cmake", "--build", str(build), "-j", "8", "--target", "install"], check=True)  # nosec B603 B607


class NiftyReg:
    def __init__(self, arguments: list[str], threads: int) -> None:
        self._arguments, self._threads = arguments, threads

    def __call__(
        self,
        fixed: sitk.Image,
        moving: sitk.Image,
        fixed_mask: sitk.Image | None,
        moving_mask: sitk.Image | None,
        device: int,
    ) -> np.ndarray:
        binaries = built_source("niftyreg", REPOSITORY, COMMIT, _build) / "install" / "bin"
        fixed, moving = sitk.Cast(fixed, sitk.sitkFloat32), sitk.Cast(moving, sitk.sitkFloat32)
        with tempfile.TemporaryDirectory(prefix="niftyreg_") as scratch:
            work = Path(scratch)
            ref, flo = work / "fixed.nii.gz", work / "moving.nii.gz"  # NiftyReg reads NIfTI only
            sitk.WriteImage(fixed, str(ref))
            sitk.WriteImage(moving, str(flo))
            threads = ["-omp", str(self._threads)] if self._threads else []
            subprocess.run(
                [
                    str(binaries / "reg_f3d"),
                    "-ref",
                    str(ref),
                    "-flo",
                    str(flo),
                    "-cpp",
                    str(work / "cpp.nii.gz"),  # nosec B603
                    "-res",
                    str(work / "moved.nii.gz"),
                    *threads,
                    *self._arguments,
                ],
                check=True,
            )
            subprocess.run(
                [
                    str(binaries / "reg_transform"),
                    "-ref",
                    str(ref),
                    "-disp",
                    str(work / "cpp.nii.gz"),
                    str(work / "disp.nii.gz"),
                ],
                check=True,
            )  # nosec B603
            displacement = sitk.GetArrayFromImage(sitk.ReadImage(str(work / "disp.nii.gz"), sitk.sitkVectorFloat32))
        if not np.isfinite(displacement).all():
            raise ValueError("NiftyReg returned a non-finite displacement")
        # NiftyReg writes the displacement along the NIfTI (RAS) axes: to ITK's LPS
        return np.ascontiguousarray(np.moveaxis(displacement * np.float32([-1, -1, 1]), -1, 0), dtype=np.float32)


class RegistrationNet(BaselineNet):
    """NiftyReg reg_f3d, defaults. CPU."""

    def __init__(
        self,
        optimizer: network.OptimizerLoader = network.OptimizerLoader(),
        schedulers: dict[str, network.LRSchedulersLoader] = {
            "default:ReduceLROnPlateau": network.LRSchedulersLoader(0)
        },
        outputs_criterions: dict[str, network.TargetCriterionsLoader] = {"default": network.TargetCriterionsLoader()},
        arguments: Annotated[list[str], "Extra reg_f3d arguments; none for the defaults."] = [],
        threads: Annotated[int, "OpenMP threads (-omp); 0 leaves NiftyReg's default."] = 0,
    ) -> None:
        super().__init__(NiftyReg(list(arguments), threads), 3, optimizer, schedulers, outputs_criterions)
