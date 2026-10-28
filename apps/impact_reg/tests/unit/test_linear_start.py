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

"""Where a linear stage starts: two sparse subjects side by side have to find each other."""

import numpy as np
import SimpleITK as sitk
from impact_reg_konfai.models.intensity import best_translation, foreground_centre


def test_the_start_is_the_shift_that_overlaps_two_sparse_subjects() -> None:
    # Two blobs a field of view apart: centring the grids leaves them side by side, where no local optimiser finds
    # them (elastix's rigid stood still on the organoid's two rounds, 36 um apart).
    fixed_array = np.zeros((20, 40, 40), np.float32)
    moving_array = np.zeros_like(fixed_array)
    fixed_array[8:12, 5:12, 5:12] = 1.0
    moving_array[8:12, 25:32, 25:32] = 1.0
    fixed, moving = sitk.GetImageFromArray(fixed_array), sitk.GetImageFromArray(moving_array)
    aligned, off = [20.0, 20.0, 0.0], [0.0, 0.0, 0.0]
    assert best_translation(fixed, moving, [off, aligned]) == aligned
    assert best_translation(fixed, moving, [aligned, off]) == aligned


def test_a_large_pair_is_scored_on_a_coarser_copy_and_still_overlaps() -> None:
    # MetricEvaluate reads every voxel whatever the sampling it is set; a million-voxel pair is scored on a copy of
    # about 200,000, masks included.
    fixed_array = np.zeros((64, 128, 128), np.float32)
    moving_array = np.zeros_like(fixed_array)
    fixed_array[24:40, 10:40, 10:40] = 1.0
    moving_array[24:40, 70:100, 70:100] = 1.0
    fixed, moving = sitk.GetImageFromArray(fixed_array), sitk.GetImageFromArray(moving_array)
    mask = sitk.GetImageFromArray((fixed_array > -1).astype(np.uint8))
    mask[0, 0, 0] = 0  # partial, so that it is used
    aligned, off = [60.0, 60.0, 0.0], [0.0, 0.0, 0.0]
    assert best_translation(fixed, moving, [off, aligned], mask) == aligned


def test_a_thin_mask_survives_the_coarser_copy() -> None:
    # One voxel thick, a mask subsampled by blocks of five would have lost every voxel, and both candidates scored
    # infinity: the first one won, whatever the overlap.
    fixed_array = np.zeros((64, 128, 128), np.float32)
    moving_array = np.zeros_like(fixed_array)
    fixed_array[24:40, 10:40, 10:40] = 1.0
    moving_array[24:40, 70:100, 70:100] = 1.0
    fixed, moving = sitk.GetImageFromArray(fixed_array), sitk.GetImageFromArray(moving_array)
    thin = np.zeros_like(fixed_array, dtype=np.uint8)
    thin[31, 5:45, 5:45] = 1  # one plane through the fixed blob
    aligned, off = [60.0, 60.0, 0.0], [0.0, 0.0, 0.0]
    assert best_translation(fixed, moving, [off, aligned], sitk.GetImageFromArray(thin)) == aligned


def test_a_mask_with_no_foreground_above_the_image_threshold_takes_its_own() -> None:
    # Thresholded over the whole image, a bright structure outside the mask left the masked subject no foreground,
    # and the start fell back to the grid centre.
    array = np.zeros((40, 40), np.float32)
    array[5:10, 5:10] = 100.0
    array[25:35, 25:35] = 10.0
    image = sitk.GetImageFromArray(array)
    region = np.zeros((40, 40), np.uint8)
    region[20:, 20:] = 1
    mask = sitk.GetImageFromArray(region)
    assert foreground_centre(image, mask) == (29.5, 29.5)
    assert foreground_centre(image, mask * 0) == (19.5, 19.5)  # an empty mask: the grid's centre
    # A mask holding some of the whole image's foreground keeps it: inside a body mask, the region's own threshold
    # would split the body's tissues.
    region[:20, :20] = 1
    assert foreground_centre(image, sitk.GetImageFromArray(region)) == (7.0, 7.0)
