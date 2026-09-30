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
import gc

import numpy as np
import SimpleITK as sitk
import torch
from konfai.metric.measure.impact import no_texpr_fuser
from konfai.utils.dataset import Attribute, data_to_image
from konfai.utils.vram import out_of_memory_as_torch


class EngineRegistration(torch.nn.Module):
    """Graph module: the fixed, moving and mask tensors with their geometry -> each sample's displacement field on the
    fixed grid, from ``engine.register``. ``accepts_attributes`` has KonfAI hand it every branch's ``Attribute`` list,
    since registration needs the physical geometry. A whole-image mask (the default) restricts nothing."""

    accepts_attributes = True

    def __init__(self, engine, fuse_texpr: bool = True) -> None:
        super().__init__()
        self._engine = engine
        self._fuse_texpr = fuse_texpr

    def forward(
        self,
        fixed: torch.Tensor,
        moving: torch.Tensor,
        fixed_mask: torch.Tensor,
        moving_mask: torch.Tensor,
        attributes: list[list[Attribute]],
    ) -> torch.Tensor:
        device_index = fixed.device.index if fixed.device.type == "cuda" else -1
        fields = []
        # The engines optimise with autograd, and the predictor calls forward under inference_mode, which forbids it.
        fuser = contextlib.nullcontext() if self._fuse_texpr else no_texpr_fuser()
        with torch.inference_mode(False), torch.enable_grad(), fuser:
            for b in range(fixed.shape[0]):
                images = [
                    data_to_image(tensor[b].detach().cpu().numpy(), branch[b])
                    for tensor, branch in zip((fixed, moving, fixed_mask, moving_mask), attributes, strict=True)
                ]
                try:
                    with out_of_memory_as_torch(device_index >= 0):
                        field = self._engine.register(images[0], images[1], device_index, images[2], images[3])
                finally:
                    # FireANTs keeps CUDA tensors in reference cycles that only the cyclic collector frees.
                    gc.collect()
                fields.append(torch.from_numpy(field))
        return torch.stack(fields, dim=0).to(fixed.device)


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
