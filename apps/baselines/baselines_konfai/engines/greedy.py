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

"""greedy as its Quick Start runs it (the ``picsl_greedy`` package at a pinned version): normalised cross-correlation
over a 2x2x2 window, 100x50x10 iterations, the deformable stage alone. One modality only.

NOTE: do NOT add ``from __future__ import annotations``: KonfAI's config engine relies on runtime-evaluated
annotations (``get_origin``); PEP 563 stringized annotations break binding.
"""

import os
import tempfile
from pathlib import Path
from typing import Annotated

import numpy as np
import SimpleITK as sitk
from konfai.network import network

from .base import BaselineNet

VERSION = "1.4.0.3"  # the picsl_greedy package; the bundle's requirements pin it


class Greedy:
    def __init__(self, metric: str, iterations: str, threads: int) -> None:
        self._metric, self._iterations, self._threads = metric, iterations, threads

    def __call__(
        self,
        fixed: sitk.Image,
        moving: sitk.Image,
        fixed_mask: sitk.Image | None,
        moving_mask: sitk.Image | None,
        device: int,
    ) -> np.ndarray:
        from picsl_greedy import Greedy3D

        threads = self._threads or int(os.environ.get("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "8"))
        with tempfile.TemporaryDirectory(prefix="greedy_") as scratch:
            work = Path(scratch)
            sitk.WriteImage(fixed, str(work / "fixed.nii.gz"))
            sitk.WriteImage(moving, str(work / "moving.nii.gz"))
            Greedy3D().execute(
                f"-d 3 -threads {threads} -m {self._metric} -i {work / 'fixed.nii.gz'} {work / 'moving.nii.gz'} "
                f"-n {self._iterations} -o {work / 'warp.nii.gz'}"
            )
            warp = sitk.ReadImage(str(work / "warp.nii.gz"), sitk.sitkVectorFloat32)
        return np.ascontiguousarray(np.moveaxis(sitk.GetArrayFromImage(warp), -1, 0), dtype=np.float32)


class RegistrationNet(BaselineNet):
    """greedy, the Quick Start's deformable call. CPU."""

    def __init__(
        self,
        optimizer: network.OptimizerLoader = network.OptimizerLoader(),
        schedulers: dict[str, network.LRSchedulersLoader] = {
            "default:ReduceLROnPlateau": network.LRSchedulersLoader(0)
        },
        outputs_criterions: dict[str, network.TargetCriterionsLoader] = {"default": network.TargetCriterionsLoader()},
        metric: Annotated[str, "greedy's -m: the similarity, NCC with its window."] = "NCC 2x2x2",
        iterations: Annotated[str, "greedy's -n: iterations per level."] = "100x50x10",
        threads: Annotated[int, "greedy's -threads; 0 takes ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS or 8."] = 0,
    ) -> None:
        super().__init__(Greedy(metric, iterations, threads), 3, optimizer, schedulers, outputs_criterions)
