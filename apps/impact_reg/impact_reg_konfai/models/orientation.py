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

"""The images a feature network is run on, with their voxel axes in one order.

A network, MIND included, computes its channels along the voxel axes it is given: the same anatomy stored with
another axis order or sign (a NIfTI with diag(-1, -1, 1) beside a DICOM, a sagittal beside an axial acquisition) gives
other channels, and comparing fixed and moving channel by channel then compares different descriptors. Handing both
images over with their voxel axes in LPS order makes their features comparable. The registration result is physical,
so the caller samples it on the original fixed grid: nothing downstream changes.
"""

import numpy as np
import SimpleITK as sitk


def world_aligned(image: sitk.Image) -> sitk.Image:
    """``image`` with its voxel axes permuted and flipped into LPS order, the same voxels and physical content.

    Nothing is resampled. An oblique direction keeps the residual rotation of under 45 degrees, which the engines
    sample physically: resampling it onto world axes would pad the image by up to a third with filler that the
    metric then samples as content.
    """
    if np.allclose(np.array(image.GetDirection()).reshape(3, 3), np.eye(3), atol=1e-6):
        return image
    return sitk.DICOMOrient(image, "LPS")


def world_aligned_pair(
    fixed: sitk.Image, moving: sitk.Image, fixed_mask: sitk.Image | None, moving_mask: sitk.Image | None
) -> tuple[sitk.Image, sitk.Image, sitk.Image | None, sitk.Image | None]:
    """``fixed``, ``moving`` and their masks through :func:`world_aligned`."""
    return tuple(None if image is None else world_aligned(image) for image in (fixed, moving, fixed_mask, moving_mask))
