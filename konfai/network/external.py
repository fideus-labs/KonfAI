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

"""An app that runs an external tool, its authors' code as they ship it, in a KonfAI graph.

The app's ``model.py`` holds only the tool's engine: a function of the case's images, as SimpleITK images with their
geometry, and of the CUDA device index (-1 on the CPU), that returns its result as an array on the first image's grid,
channel-first. ``ExternalNet`` hands every case to it, a sample at a time, through ``ExternalTool``; the predictor
attaches the first image's geometry to the output (``same_as_group`` in the app's ``Prediction.yml``).

NOTE: do NOT add ``from __future__ import annotations``: KonfAI's config engine relies on runtime-evaluated
annotations (``get_origin``); PEP 563 stringized annotations break binding.
"""

import gc
from collections.abc import Callable, Sequence

import numpy as np
import SimpleITK as sitk
import torch

from konfai.network import network
from konfai.utils.dataset import Attribute, data_to_image
from konfai.utils.vram import out_of_memory_as_torch

Engine = Callable[..., np.ndarray]


class ExternalTool(torch.nn.Module):
    """Graph module: the branches' tensors and their geometry in, the engine's result out, one sample at a time.

    Each branch reaches the engine as a SimpleITK image, in branch order, then the CUDA device index. A branch in
    ``masks`` that holds only ones (what konfai-apps writes for a mask not given) reaches it as None. The engine may
    optimise with autograd, which the predictor's inference mode would forbid; a CUDA out of memory reported as a
    plain error is raised as torch's, which KonfAI shrinks on."""

    accepts_attributes = True

    def __init__(self, engine: Engine, masks: Sequence[int] = ()) -> None:
        super().__init__()
        self._engine = engine
        self._masks = set(masks)

    def call(self, images: list[sitk.Image | None], device_index: int) -> np.ndarray:
        """The engine on one sample: its images in branch order, then the device index."""
        return self._engine(*images, device_index)

    def forward(self, *tensors: torch.Tensor, attributes: list[list[Attribute]]) -> torch.Tensor:
        device_index = tensors[0].device.index if tensors[0].device.type == "cuda" else -1
        results = []
        with torch.inference_mode(False), torch.enable_grad():
            for b in range(tensors[0].shape[0]):
                images: list[sitk.Image | None] = [
                    None
                    if i in self._masks and bool((tensor[b] == 1).all())
                    else data_to_image(tensor[b].detach().cpu().numpy(), branch[b])
                    for i, (tensor, branch) in enumerate(zip(tensors, attributes, strict=True))
                ]
                try:
                    with out_of_memory_as_torch(device_index >= 0):
                        result = self.call(images, device_index)
                finally:
                    # A tool on torch may keep CUDA tensors in reference cycles that only the cyclic collector frees.
                    gc.collect()
                # a copy: a tool may reuse its buffer for the next sample
                results.append(torch.from_numpy(np.array(result)))
        return torch.stack(results, dim=0).to(tensors[0].device)


class ExternalNet(network.Network):
    """An external tool's app as an ``add_module`` graph: its ``branches`` inputs in order (the image, or the fixed
    and moving images, then their masks: ``masks`` names those branches), its one output named ``output``
    (``Segmentation``, ``Synthetic``, ``DisplacementField``), on the first image's grid."""

    #: The tools write their results at full precision: KonfAI blends them in float32, not float16.
    full_precision_outputs = True

    def __init__(
        self,
        engine: Engine,
        branches: int,
        masks: Sequence[int],
        output: str,
        dim: int,
        optimizer: network.OptimizerLoader,
        schedulers: dict[str, network.LRSchedulersLoader],
        outputs_criterions: dict[str, network.TargetCriterionsLoader],
    ) -> None:
        super().__init__(
            in_channels=1, optimizer=optimizer, schedulers=schedulers, outputs_criterions=outputs_criterions, dim=dim
        )
        self.add_module(output, ExternalTool(engine, masks), in_branch=list(range(branches)), out_branch=["out"])


def scalar(image: sitk.Image) -> sitk.Image:
    """The image a tool comparing one value a voxel reads: a colour or multiband image as the mean of its components,
    float32; a scalar image only cast."""
    n = image.GetNumberOfComponentsPerPixel()
    if n == 1:
        return sitk.Cast(image, sitk.sitkFloat32)
    return sitk.Cast(
        sum(sitk.VectorIndexSelectionCast(image, c, sitk.sitkFloat32) for c in range(n)) / n, sitk.sitkFloat32
    )
