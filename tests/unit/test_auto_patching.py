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

"""The per-axis patch convention (0 = free axis) and its sizing/overlap resolution.

A patch entry of 0 leaves that axis to the framework: full extent when it fits, shrunk to the memory
budget when it does not. A positive entry is pinned by the user and never moves. Overlap accepts voxels,
fractions, percent strings and per-axis mixes, resolves AFTER the patch size, and is 0 on any axis a
single patch spans.
"""

import numpy as np
import pytest
from konfai.utils.utils import (
    concretize_patch_size,
    get_patch_slices_from_shape,
    resolve_overlap,
    resolve_patch,
)


class TestConcretize:
    def test_zero_axes_take_the_full_extent(self):
        assert concretize_patch_size([0, 0, 0], [37, 41, 29]) == [37, 41, 29]

    def test_mixed_fixed_and_free(self):
        assert concretize_patch_size([1, 0, 0], [37, 41, 29]) == [1, 41, 29]
        assert concretize_patch_size([0, 128, 128], [500, 400, 300]) == [500, 128, 128]

    def test_fixed_larger_than_the_volume_is_clamped(self):
        assert concretize_patch_size([128, 128, 128], [96, 96, 96]) == [96, 96, 96]

    def test_none_is_the_whole_volume(self):
        assert concretize_patch_size(None, [10, 20, 30]) == [10, 20, 30]


class TestResolveOverlap:
    def test_default_is_a_fifth_of_the_patch_per_axis(self):
        # patch 10 -> 2 (the settled default), and per axis of ITS patch size.
        assert resolve_overlap(None, [10, 50, 100], [100, 500, 1000]) == [2, 10, 20]

    def test_absolute_voxels_broadcast(self):
        assert resolve_overlap(16, [128, 128, 128], [512, 512, 512]) == [16, 16, 16]

    def test_fraction_and_percent_string(self):
        assert resolve_overlap(0.25, [64, 64, 64], [512, 512, 512]) == [16, 16, 16]
        assert resolve_overlap("25%", [64, 64, 64], [512, 512, 512]) == [16, 16, 16]

    def test_per_axis_mix(self):
        assert resolve_overlap([8, 0.1, "50%"], [64, 100, 10], [512, 512, 512]) == [8, 10, 5]

    def test_untiled_axis_gets_zero_whatever_the_spec(self):
        # First axis: one patch spans the extent -> no overlap there.
        assert resolve_overlap(16, [512, 128, 128], [512, 512, 512]) == [0, 16, 16]

    def test_single_voxel_patch_axis_gets_zero(self):
        # 2D slicing: nothing to blend along a length-1 patch axis.
        assert resolve_overlap(16, [1, 128, 128], [300, 512, 512]) == [0, 16, 16]

    def test_overlap_not_smaller_than_patch_is_refused(self):
        with pytest.raises(ValueError, match="smaller than the patch"):
            resolve_overlap(64, [64, 64, 64], [512, 512, 512])

    def test_bad_forms_are_refused(self):
        with pytest.raises(ValueError, match="percentage"):
            resolve_overlap("big", [64, 64, 64], [512, 512, 512])
        with pytest.raises(ValueError, match=r"\[0, 1\["):
            resolve_overlap(1.5, [64, 64, 64], [512, 512, 512])
        with pytest.raises(ValueError, match="one per axis"):
            resolve_overlap([1, 2], [64, 64, 64], [512, 512, 512])


class TestPatchGrid:
    def test_2d_slicing_grid(self):
        # [1,0,0] -> one full slice per Z index: exactly Z patches, each spanning Y and X.
        slices, nb_per_dim = get_patch_slices_from_shape([1, 0, 0], [37, 41, 29], None)
        assert len(slices) == 37
        assert slices[0] == (slice(0, 1), slice(0, 41), slice(0, 29))
        assert nb_per_dim[0] == (37, True)

    def test_free_axes_cover_the_volume_disjointly(self):
        # A fixed Z with free in-plane axes tiles Z only; the plane stays whole.
        slices, _ = get_patch_slices_from_shape([16, 0, 0], [37, 41, 29], None)
        zs = sorted({(s[0].start, s[0].stop) for s in slices})
        assert all(s[1] == slice(0, 41) and s[2] == slice(0, 29) for s in slices)
        covered = set()
        for start, stop in zs:
            covered.update(range(start, stop))
        assert covered == set(range(37))

    def test_all_zero_is_the_whole_volume(self):
        slices, _ = get_patch_slices_from_shape([0, 0, 0], [10, 20, 30], None)
        assert slices == [(slice(0, 10), slice(0, 20), slice(0, 30))]

    def test_fixed_patch_grid_stays_bit_identical(self):
        # A fully-fixed patch keeps the remainder-spreading overlap grid: every stored model was
        # trained and evaluated on it.
        fixed, _ = get_patch_slices_from_shape([16, 16, 16], [37, 41, 29], None)
        assert fixed[0] == (slice(0, 16), slice(0, 16), slice(0, 16))
        assert all(s[0].stop <= 37 and s[1].stop <= 41 and s[2].stop <= 29 for s in fixed)


class TestResolvePatch:
    def test_whole_volume_when_it_fits(self):
        # 64^3 float32 single channel = 1 MiB; a 100 MiB budget swallows it whole.
        assert resolve_patch([0, 0, 0], [64, 64, 64], 1, 4, 100 * 2**20, 2, 1.0) == [64, 64, 64]

    def test_no_budget_means_full_extent(self):
        assert resolve_patch([0, 0, 0], [512, 512, 512], 1, 4, None) == [512, 512, 512]

    def test_shrinks_isotropically_when_too_big(self):
        shape = [512, 512, 512]
        sized = resolve_patch([0, 0, 0], shape, 1, 4, 64 * 2**20, 2, 1.0)
        assert all(1 <= p < s for p, s in zip(sized, shape, strict=True))
        # Isotropic on an isotropic volume: all free axes shrink alike.
        assert len(set(sized)) == 1
        # And the sized patch actually fits the safety-scaled budget.
        assert 3 * np.prod(sized) * 4 <= 64 * 2**20 * 0.8 * 1.001

    def test_fixed_axes_never_move(self):
        sized = resolve_patch([1, 0, 0], [300, 4096, 4096], 1, 4, 8 * 2**20, 2, 1.0)
        assert sized[0] == 1
        assert all(1 <= p < 4096 for p in sized[1:])

    def test_snap_rounds_free_axes_down_to_model_multiples(self):
        sized = resolve_patch([0, 0, 0], [500, 500, 500], 1, 4, 64 * 2**20, 2, 1.0, snap=[16, 16, 16])
        assert all(p % 16 == 0 for p in sized)

    def test_pinned_axes_exceeding_the_budget_raise(self):
        from konfai.utils.errors import DatasetManagerError

        with pytest.raises(DatasetManagerError, match="exceed the memory budget"):
            resolve_patch([512, 512, 0], [512, 512, 512], 4, 4, 1 * 2**20)


class TestPatchedReductionIdentity:
    """combine(disjoint patches) == metric(whole): the numerical foundation of auto-patched evaluation."""

    @pytest.mark.parametrize("patch", [[16, 16, 16], [1, 41, 29], [8, 7, 5]])
    def test_running_sums_reproduce_whole_volume_metrics(self, patch):
        rng = np.random.default_rng(0)
        out = rng.random((37, 41, 29))
        tgt = rng.random((37, 41, 29))
        slices, _ = get_patch_slices_from_shape(patch, [37, 41, 29], 0)

        abs_sum, sq_sum, count = 0.0, 0.0, 0
        for sl in slices:
            diff = out[sl] - tgt[sl]
            abs_sum += np.abs(diff).sum()
            sq_sum += (diff**2).sum()
            count += diff.size
        assert count == out.size  # disjoint AND exhaustive tiling
        assert abs_sum / count == pytest.approx(np.abs(out - tgt).mean(), rel=1e-12)
        assert sq_sum / count == pytest.approx(((out - tgt) ** 2).mean(), rel=1e-12)
