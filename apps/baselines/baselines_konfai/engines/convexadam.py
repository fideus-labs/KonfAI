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

"""ConvexAdam with MIND-SSC features: the authors' repository (multimodallearning/convexAdam) at a pinned commit,
``convex_adam_pt(fixed, moving)`` with its defaults, as the ImagingBenchmarks line ``convexadam_mind`` runs it.

NOTE: do NOT add ``from __future__ import annotations``: KonfAI's config engine relies on runtime-evaluated
annotations (``get_origin``); PEP 563 stringized annotations break binding.
"""

import sys
from typing import Annotated

import numpy as np
import SimpleITK as sitk
import torch
from konfai.network import network

from .base import BaselineNet, field_from_voxels, pinned_source, to_index_array

REPOSITORY = "https://github.com/multimodallearning/convexAdam"
COMMIT = "b229e52e44b114e2040a503334c92269750c16b2"  # main, 2025-11-17


def _authors_code():
    """The authors' functions, imported from their repository at the pin (it is not a package)."""
    source = pinned_source("convexAdam", REPOSITORY, COMMIT) / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from convexAdam.convex_adam_MIND import convex_adam_pt
    from convexAdam.convex_adam_utils import resample_moving_to_fixed

    return convex_adam_pt, resample_moving_to_fixed


class ConvexAdamMind:
    def __init__(self, threads: int) -> None:
        self._threads = threads

    def __call__(
        self,
        fixed: sitk.Image,
        moving: sitk.Image,
        fixed_mask: sitk.Image | None,
        moving_mask: sitk.Image | None,
        device: int,
    ) -> np.ndarray:
        convex_adam_pt, resample_moving_to_fixed = _authors_code()
        if self._threads:
            torch.set_num_threads(self._threads)
        fixed, moving = sitk.Cast(fixed, sitk.sitkFloat32), sitk.Cast(moving, sitk.sitkFloat32)
        if fixed.GetDimension() != 3 or min(fixed.GetSize()) < 12:
            raise ValueError(
                f"ConvexAdam needs 3D images of at least 12 voxels per axis (grid spacing 6): {fixed.GetSize()}"
            )

        def geometry(image: sitk.Image) -> tuple:
            return image.GetSize(), image.GetSpacing(), image.GetOrigin(), image.GetDirection()

        if geometry(moving) != geometry(fixed):
            moving = resample_moving_to_fixed(fixed, moving)  # the repository's helper: identity, linear, 0 outside
        # the repository's defaults; the displacement comes back as [i, j, k, 3], fixed -> moving, in voxels
        displacement = convex_adam_pt(torch.from_numpy(to_index_array(fixed)), torch.from_numpy(to_index_array(moving)))
        displacement = np.asarray(displacement)
        if not np.isfinite(displacement).all():
            raise ValueError("ConvexAdam returned a field with non-finite values")
        return field_from_voxels(displacement, fixed)


class RegistrationNet(BaselineNet):
    """ConvexAdam (MIND-SSC) at the authors' defaults: MIND radius 1 and dilation 2, a convex stage on a 6-voxel
    grid with a search half-width of 4 and inverse consistency, then 80 Adam steps on a 2-voxel grid with a
    diffusion weight of 1.25, half precision. A GPU is needed."""

    def __init__(
        self,
        optimizer: network.OptimizerLoader = network.OptimizerLoader(),
        schedulers: dict[str, network.LRSchedulersLoader] = {
            "default:ReduceLROnPlateau": network.LRSchedulersLoader(0)
        },
        outputs_criterions: dict[str, network.TargetCriterionsLoader] = {"default": network.TargetCriterionsLoader()},
        threads: Annotated[
            int, "CPU threads torch may use for the parts that run on the CPU; 0 leaves the default."
        ] = 0,
    ) -> None:
        super().__init__(ConvexAdamMind(threads), 3, optimizer, schedulers, outputs_criterions)
