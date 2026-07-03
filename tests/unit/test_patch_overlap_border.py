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

"""Regression test: overlap-blended reassembly must not darken the volume border."""

import torch
from konfai.data.patching import Accumulator, Cosinus, Mean


def test_overlap_blend_is_partition_of_unity_at_the_border() -> None:
    # 1-D volume of 20 tiled with patch 8 / overlap 2 -> patches at 0, 6, 12. The border voxels are
    # covered by a single patch, whose edge band weights ~0.5, so the pre-fix sum-without-normalise
    # reassembled them at 0.5 instead of 1.0. Dividing by the accumulated weight restores unity.
    patch_slices = [(slice(0, 8),), (slice(6, 14),), (slice(12, 20),)]
    combine = Mean()  # Cosinus needs >=2D (SimpleITK distance map), covered below.
    combine.set_patch_config([8], 2)
    accumulator = Accumulator(patch_slices, patch_size=[8], patch_combine=combine, batch=True)
    for index in range(len(patch_slices)):
        accumulator.add_layer(index, torch.ones(1, 1, 8))

    out = accumulator.assemble()[0, 0]

    assert out.shape == (20,)
    torch.testing.assert_close(out, torch.ones(20), rtol=0, atol=1e-5)


def test_overlap_blend_corner_not_quartered_in_2d() -> None:
    # A 2-D corner is covered by one patch on both axes, so the pre-fix output was ~0.25 there.
    patch_slices = [
        (slice(0, 8), slice(0, 8)),
        (slice(0, 8), slice(6, 14)),
        (slice(6, 14), slice(0, 8)),
        (slice(6, 14), slice(6, 14)),
    ]
    for combine_cls in (Mean, Cosinus):
        combine = combine_cls()
        combine.set_patch_config([8, 8], 2)
        accumulator = Accumulator(patch_slices, patch_size=[8, 8], patch_combine=combine, batch=True)
        for index in range(len(patch_slices)):
            accumulator.add_layer(index, torch.ones(1, 1, 8, 8))

        out = accumulator.assemble()[0, 0]

        assert out.shape == (14, 14)
        torch.testing.assert_close(out, torch.ones(14, 14), rtol=0, atol=1e-5)
