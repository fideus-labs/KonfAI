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

"""A tool that segments one object from prompts: clicks, a box, its last mask.

**A prompt is a set of points of the case**, as the interactive tools take it and as 3D Slicer writes it: a fiducial
file (``.fcsv``) in physical space, each point labelled with its kind, ``positive`` or ``negative`` for a click and
``box`` for each of a box's two opposite corners, in the order the user gave them. The last mask is an image, empty
at the first prediction. Both are groups of the case: the points need no chain (a physical point is the same point
whatever grid the image is put on), and the mask comes back on the image's own grid by the chain's inverse. A
correction is one more point in the file and one more prediction, which overwrites the last; several objects are
several cases that share the image. The image is handed whole (``patch_size: [0, 0, 0]``): a patch carries its case's
geometry and not its own place in it, so a case cut in several patches is refused.

``PromptedNet`` runs an engine on one object: the engine gets the case's image, then the prompts as ``Prompts``, in
voxels of the image it is handed, then the CUDA device index, and returns the object's mask on that image's grid,
channel first. An engine that is a module has its weights in the checkpoint, loaded by KonfAI; a function loads its
own. KonfAI builds the workflow again at every prediction and konfai-apps drops an app's modules after each run: what
the engine keeps from one prediction to the next (the image it has encoded, a session) it keeps through ``kept``, for
the life of the process.

NOTE: do NOT add ``from __future__ import annotations``: KonfAI's config engine relies on runtime-evaluated
annotations (``get_origin``); PEP 563 stringized annotations break binding.
"""

import json
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np

from konfai.network import network
from konfai.network.external import ExternalNet, ExternalTool
from konfai.utils.dataset import Attribute, is_an_image
from konfai.utils.errors import KonfAIError

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None  # type: ignore[assignment]

T = TypeVar("T")

_KEPT: dict[Hashable, Any] = {}


def kept(key: Hashable, make: Callable[[], T]) -> T:
    """What a tool keeps from one prediction to the next, made once under ``key`` (the tool, its weights, its
    device) and held for the life of the process."""
    # ponytail: nothing is ever dropped; evict by use when one process serves several prompted apps.
    if key not in _KEPT:
        _KEPT[key] = make()
    return _KEPT[key]


#: The kinds of point every prompted tool knows. Any other label is a kind of the app's own (``Prompts.of``).
POSITIVE, NEGATIVE, BOX = "positive", "negative", "box"


@dataclass
class Prompts:
    """One object's prompts on the grid of the image the engine is handed.

    ``voxels`` are the points, ``[N, rank]`` in the array's order, (z, y, x), as continuous indices: a point
    given on a voxel of the stored image is that voxel's index, whole, as long as the chain leaves the image on its
    grid. ``kinds`` names each point's kind, in the order they were given. ``previous`` is the last mask, or None
    before the first prediction."""

    voxels: np.ndarray
    kinds: list[str]
    previous: np.ndarray | None = None

    def of(self, kind: str) -> np.ndarray:
        """The points of one kind, in their order."""
        return self.voxels[[held == kind for held in self.kinds]]

    @property
    def clicks(self) -> list[tuple[np.ndarray, bool]]:
        """The clicks in the order they were given, each with whether it is positive."""
        return [(v, k == POSITIVE) for v, k in zip(self.voxels, self.kinds, strict=True) if k in (POSITIVE, NEGATIVE)]

    @property
    def box(self) -> tuple[np.ndarray, np.ndarray] | None:
        """The box its corners span, as two voxels [low, high), or None without one."""
        corners = np.rint(self.of(BOX)).astype(int)
        return (corners.min(axis=0), corners.max(axis=0) + 1) if len(corners) else None


class PromptedTool(ExternalTool):
    """Graph module: the engine on one object. Its branches are the image, the points and the last mask."""

    def handed(self, data: np.ndarray, attribute: Attribute) -> Any:
        if is_an_image(attribute):
            return super().handed(data, attribute)
        return data, json.loads(attribute["Labels"])

    def call(self, images: list[Any], device_index: int, header: Attribute) -> np.ndarray:
        image, (points, kinds), previous = images
        grid = ("GetSize", "GetOrigin", "GetSpacing", "GetDirection")
        if not all(np.allclose(getattr(image, get)(), getattr(previous, get)(), rtol=0, atol=1e-6) for get in grid):
            raise KonfAIError("Prompted tool", "The last mask is not on the image's grid: resample it onto the image.")
        rank = image.GetDimension()  # a fiducial file holds 3D points, a 2D image takes their first two
        voxels = [image.TransformPhysicalPointToContinuousIndex(point[:rank])[::-1] for point in points.tolist()]
        mask = sitk.GetArrayViewFromImage(previous) > 0.5
        prompts = Prompts(np.array(voxels, float).reshape(-1, rank), kinds, mask if mask.any() else None)
        return self._engine(image, prompts, device_index)


class PromptedNet(ExternalNet):
    """A prompted tool's app as an ``add_module`` graph: its inputs the image, then the points, then the last mask;
    its one output, ``Segmentation``, the object's mask on the image's grid."""

    tool_class = PromptedTool

    def __init__(
        self,
        engine: Callable[..., np.ndarray],
        dim: int = 3,
        optimizer: network.OptimizerLoader | None = None,
        schedulers: dict[str, network.LRSchedulersLoader] | None = None,
        outputs_criterions: dict[str, network.TargetCriterionsLoader] | None = None,
    ) -> None:
        super().__init__(engine, 3, (), "Segmentation", dim, optimizer, schedulers, outputs_criterions)
