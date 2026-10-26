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

"""What the engines share: the graph module that runs one on a batch, the winsorised intensities they read, and
whether a mask restricts an image."""

import contextlib

import numpy as np
import SimpleITK as sitk
from konfai.metric.measure.impact import no_texpr_fuser
from konfai.network.external import ExternalTool


class EngineRegistration(ExternalTool):
    """Graph module: the fixed, moving and mask tensors with their geometry -> each sample's displacement field on the
    fixed grid, from ``engine.register``: KonfAI's ``ExternalTool`` with IMPACT-Reg's engines' call. A whole-image mask
    (the default) restricts nothing; each engine reads it as such."""

    def __init__(self, engine, fuse_texpr: bool = True) -> None:
        super().__init__(engine)
        self._fuse_texpr = fuse_texpr

    def call(self, images: list[sitk.Image | None], device_index: int) -> np.ndarray:
        fixed, moving, fixed_mask, moving_mask = images
        with contextlib.nullcontext() if self._fuse_texpr else no_texpr_fuser():
            return self._engine.register(fixed, moving, device_index, fixed_mask, moving_mask)


def winsorized(image: sitk.Image, low: float = 0.01, high: float = 99.99) -> sitk.Image:
    """``image`` clamped to its ``low`` and ``high`` percentiles: mutual information, an Otsu threshold and MIND all
    read the image's range, which a few lone extreme voxels would otherwise set. The default is not ANTs' 0.5-99.5:
    over the whole volume, air included, that clamps the tissues of a CBCT into one value."""
    bottom, top = np.percentile(sitk.GetArrayViewFromImage(image), (low, high))
    if top <= bottom:
        return image  # background almost everywhere (a tile past the tissue): clamped, it would be constant
    return sitk.Clamp(image, image.GetPixelID(), float(bottom), float(top))


def is_partial_mask(mask: sitk.Image | None) -> bool:
    """Whether ``mask`` restricts the region: some voxels in (not 0), some out. konfai-apps fills an absent mask with
    ones, which restricts nothing."""
    if mask is None:
        return False
    array = sitk.GetArrayViewFromImage(mask)
    return bool((array != 0).any()) and bool((array == 0).any())
