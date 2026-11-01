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

"""What the engines share: the graph module that runs one on a batch in voxel units, the winsorised intensities they
read, the translation between two subjects' centres of mass, and whether a mask restricts an image."""

import contextlib
import gc

import numpy as np
import SimpleITK as sitk
import torch
from konfai.metric.measure.impact import no_texpr_fuser
from konfai.utils.dataset import (
    AFFINE_ATTRIBUTE,
    DISPLACEMENT_FIELD_ATTRIBUTE,
    Attribute,
    affine_to_attribute,
    data_to_image,
    image_to_data,
)
from konfai.utils.vram import out_of_memory_as_torch

#: The forms a preset declares its transform written in: its displacement on the fixed grid, or the affine it is.
TRANSFORM_TYPES = ("field", "affine")


def check_transform_type(transform_type: str) -> None:
    if transform_type not in TRANSFORM_TYPES:
        raise ValueError(f"Unknown transform_type '{transform_type}': {' or '.join(map(repr, TRANSFORM_TYPES))}.")


def displacement_on(grid: sitk.Image, transform: sitk.Transform) -> np.ndarray:
    """``transform`` sampled as a displacement field on ``grid``, channel-first, float64."""
    field = sitk.TransformToDisplacementField(
        transform, sitk.sitkVectorFloat64, grid.GetSize(), grid.GetOrigin(), grid.GetSpacing(), grid.GetDirection()
    )
    return image_to_data(field)[0]


def _world_affine(linear: sitk.Transform, voxel: float, z_shift: float | None) -> np.ndarray:
    """``linear``, found in voxel units, as ``AFFINE_ATTRIBUTE`` records it, in millimetres. A plane's transform
    gets the z row of a volume one plane thick: the shift from one plane to the other, as its field does."""
    matrix = affine_to_attribute(linear).reshape(linear.GetDimension(), -1)
    matrix[:, -1] *= voxel
    if z_shift is not None:
        matrix = np.block([[matrix[:, :2], np.zeros((2, 1)), matrix[:, 2:]], [np.array([[0.0, 0.0, 1.0, z_shift]])]])
    return matrix.ravel()


class EngineRegistration(torch.nn.Module):
    """Graph module: the fixed, moving and mask tensors with their geometry -> each sample's displacement field on the
    fixed grid, from ``engine.register``. ``accepts_attributes`` has KonfAI hand it every branch's ``Attribute`` list,
    since registration needs the physical geometry. A whole-image mask (the default) restricts nothing.

    The engine measures lengths in the fixed image's finest voxel, and its field is brought back to millimetres: every
    length a preset sets (a step, a grid, a smoothing, a feature model's voxel) is a number of voxels, so a step of 1.2
    is 1.2 mm on 1 mm voxels and 1.2 um on 1 um voxels.

    An engine whose preset declares ``transform_type: affine`` returns the linear transform itself. The field KonfAI
    carries is that transform sampled, and the output's header records the map (``AFFINE_ATTRIBUTE``), which the
    output dataset writes in place of the field."""

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
                # A 2-D image comes as a volume one plane thick (``plane.py``), and a tile of it without its z axis, which
                # KonfAI drops from a patch one plane thick and adds back to what comes out: the engine registers the plane.
                images = [
                    data_to_image(
                        (tensor[b] if tensor.ndim == 5 else tensor[b].unsqueeze(1)).detach().cpu().numpy(), branch[b]
                    )
                    for tensor, branch in zip((fixed, moving, fixed_mask, moving_mask), attributes, strict=True)
                ]
                flat = images[0].GetSize()[2] == 1
                if flat:
                    z_shift = images[1].GetOrigin()[2] - images[0].GetOrigin()[2]
                    images = [image[:, :, 0] for image in images]
                voxel = min(images[0].GetSpacing())
                for image in images:  # the same voxels, measured in the fixed image's finest voxel
                    image.SetSpacing([v / voxel for v in image.GetSpacing()])
                    image.SetOrigin([v / voxel for v in image.GetOrigin()])
                try:
                    with out_of_memory_as_torch(device_index >= 0):
                        result = self._engine.register(images[0], images[1], device_index, images[2], images[3])
                    linear = result if isinstance(result, sitk.Transform) else None
                    field = (displacement_on(images[0], linear) if linear is not None else result) * voxel
                    if flat:  # the plane's (dx, dy), and along z from one plane to the other
                        field = np.concatenate([field, np.full_like(field[:1], z_shift)])
                finally:
                    # FireANTs keeps CUDA tensors in reference cycles that only the cyclic collector frees.
                    gc.collect()
                fields.append(torch.from_numpy(field))
                # The output's header is the fixed image's: an ':omezarr' output dataset then writes an RFC-5 field.
                attributes[0][b][DISPLACEMENT_FIELD_ATTRIBUTE] = "true"
                if linear is not None:
                    attributes[0][b][AFFINE_ATTRIBUTE] = _world_affine(linear, voxel, z_shift if flat else None)
        return torch.stack(fields, dim=0).to(fixed.device)


def winsorized(image: sitk.Image, low: float = 0.01, high: float = 99.99) -> sitk.Image:
    """``image`` clamped to its ``low`` and ``high`` percentiles: mutual information, an Otsu threshold and MIND all
    read the image's range, which a few lone extreme voxels would otherwise set. The default is not ANTs' 0.5-99.5:
    over the whole volume, air included, that clamps the tissues of a CBCT into one value."""
    bottom, top = np.percentile(sitk.GetArrayViewFromImage(image), (low, high))
    if top <= bottom:
        return image  # background almost everywhere (a tile past the tissue): clamped, it would be constant
    return sitk.Clamp(image, image.GetPixelID(), float(bottom), float(top))


def center_of_mass_translation(
    fixed: sitk.Image, moving: sitk.Image, fixed_mask: sitk.Image | None = None, moving_mask: sitk.Image | None = None
) -> list[float]:
    """The translation from ``fixed``'s centre of mass to ``moving``'s: where a linear stage starts for a preset that
    declares ``moments_init: com``.

    Each centre is taken in its image's own physical space, inside its mask if it has one, on intensities winsorised
    to their 0.5-99.5 percentiles and rescaled to [0, 1]. Raw, a background outweighs a sparse subject (the 36 um
    between an organoid's two rounds come out as 11), CT clipped at 0 weighs bone tens of times over soft tissue, and
    a stray bright voxel weighs more than whole organs. Only the centroid is used: a near-symmetric subject has
    ambiguous principal axes.
    """

    def centre(image: sitk.Image, mask: sitk.Image | None) -> np.ndarray:
        rescaled = sitk.RescaleIntensity(winsorized(sitk.Cast(image, sitk.sitkFloat32), 0.5, 99.5), 0.0, 1.0)
        if mask is None:
            region = sitk.Image(image.GetSize(), sitk.sitkUInt8) + 1
            region.CopyInformation(image)
        else:
            # On the image's grid (a mask may come on another): onto the same grid, the resample is a copy.
            identity = sitk.Transform(image.GetDimension(), sitk.sitkIdentity)
            region = sitk.Resample(sitk.Cast(mask != 0, sitk.sitkUInt8), image, identity, sitk.sitkNearestNeighbor)
        stats = sitk.LabelIntensityStatisticsImageFilter()
        stats.SetComputePerimeter(False)
        stats.Execute(region, rescaled)
        if stats.HasLabel(1) and stats.GetSum(1) > 0:
            return np.asarray(stats.GetCenterOfGravity(1))
        # A constant subject (all 0 once rescaled) has no centre of mass: its grid's centre, as centring frames.
        return np.asarray(image.TransformContinuousIndexToPhysicalPoint([(n - 1) / 2 for n in image.GetSize()]))

    return (centre(moving, moving_mask) - centre(fixed, fixed_mask)).tolist()


def is_partial_mask(mask: sitk.Image | None) -> bool:
    """Whether ``mask`` restricts the region: some voxels in (not 0), some out. konfai-apps fills an absent mask with
    ones, which restricts nothing."""
    if mask is None:
        return False
    array = sitk.GetArrayViewFromImage(mask)
    return bool((array != 0).any()) and bool((array == 0).any())
