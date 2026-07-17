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

"""VRAM-driven patch sizing: one measured shrink step per OOM restart, verified fit.

The pure kernel (:func:`next_patch_candidate`) is exercised through the same loop the workflows
run -- try, measure, shrink, retry -- with synthetic probes (no GPU needed); the measurement
primitive itself runs on a real CUDA device when one is present.
"""

import numpy as np
import pytest
import torch
from konfai.utils.utils import concretize_patch_size
from konfai.utils.vram import measure_transient_bytes, next_patch_candidate, usable_vram


def _linear_probe(bytes_per_voxel: float):
    """A probe whose transient scales linearly in voxels -- the conv-net regime."""

    def probe(patch_size):
        return int(np.prod(patch_size) * bytes_per_voxel)

    return probe


def run_until_fits(patch_size, shape, probe, usable, snap=None):
    """The workflows' restart loop in miniature: run, measure, shrink one step, run again.

    Returns the first candidate whose measured transient fits ``usable``, or ``None`` when the
    kernel reports nothing smaller exists. Bounded so a non-shrinking kernel fails the test instead
    of hanging it.
    """
    candidate = concretize_patch_size(patch_size, shape)
    for _ in range(64):
        measured = probe(candidate)
        if measured is not None and measured <= usable:
            return candidate
        candidate = next_patch_candidate(candidate, patch_size, shape, measured, usable, snap)
        if candidate is None:
            return None
    raise AssertionError("the shrink loop did not converge")


class TestShrinkLoop:
    def test_whole_volume_when_the_measured_run_fits(self):
        sized = run_until_fits([0, 0, 0], [64, 64, 64], _linear_probe(10), usable=64**3 * 10)
        assert sized == [64, 64, 64]

    def test_shrinks_until_the_measured_run_fits(self):
        probe = _linear_probe(100)
        usable = 40**3 * 100  # fits 40^3, far below the 64^3 extent
        sized = run_until_fits([0, 0, 0], [64, 64, 64], probe, usable)
        assert all(1 <= p < 64 for p in sized)
        assert probe(sized) <= usable

    def test_a_measured_transient_scales_straight_to_the_target(self):
        # 8x over budget with 3 free axes -> one isotropic step of (1/8)^(1/3) halves each axis.
        shrunk = next_patch_candidate([64, 64, 64], [0, 0, 0], [64, 64, 64], measured_bytes=800, usable_bytes=100)
        assert shrunk == [32, 32, 32]

    def test_pinned_axes_never_move(self):
        probe = _linear_probe(1000)
        sized = run_until_fits([1, 0, 0], [64, 256, 256], probe, usable=32_000_000)
        assert sized[0] == 1
        assert probe(sized) <= 32_000_000

    def test_snap_keeps_model_valid_multiples(self):
        sized = run_until_fits([0, 0, 0], [100, 100, 100], _linear_probe(500), usable=120_000_000, snap=[16, 16, 16])
        assert sized is not None
        assert all(p % 16 == 0 for p in sized)

    def test_snap_floor_clamps_to_a_small_extent(self):
        # An extent below the model multiple floors there (padding makes it model-valid), not at 0.
        shrunk = next_patch_candidate([9, 64, 64], [0, 0, 0], [9, 64, 64], None, usable_bytes=1.0, snap=[16, 16, 16])
        assert shrunk is not None
        assert shrunk[0] == 9

    def test_oom_without_a_number_walks_down_by_the_fixed_step(self):
        calls = []

        def probe(patch_size):
            calls.append(list(patch_size))
            return None if np.prod(patch_size) > 20**3 else int(np.prod(patch_size) * 10)

        sized = run_until_fits([0, 0, 0], [64, 64, 64], probe, usable=20**3 * 10)
        assert np.prod(sized) <= 20**3
        assert len(calls) > 2  # it actually walked down through the OOMs, one step per restart

    def test_a_stale_fitting_measurement_still_shrinks(self):
        # The last measured batch "fits" yet an OOM happened -> the kernel must not return the same
        # candidate (scaling by >= 1 would); it falls back to the fixed step.
        shrunk = next_patch_candidate([64, 64, 64], [0, 0, 0], [64, 64, 64], measured_bytes=50, usable_bytes=100)
        assert shrunk is not None
        assert all(p < 64 for p in shrunk)

    def test_the_floor_is_reported_as_none_not_a_loop(self):
        assert run_until_fits([0, 0, 0], [64, 64, 64], _linear_probe(1e9), usable=10.0) is None

    def test_no_free_axis_means_nothing_to_shrink(self):
        assert next_patch_candidate([32, 32, 32], [32, 32, 32], [64, 64, 64], None, usable_bytes=1e9) is None

    def test_no_usable_vram_at_all_means_none(self):
        assert next_patch_candidate([64, 64, 64], [0, 0, 0], [64, 64, 64], None, usable_bytes=0.0) is None


class TestUsableVram:
    def test_margin_and_resident_come_off_the_free_memory(self):
        assert usable_vram(1000.0, resident_bytes=300.0) == pytest.approx(1000.0 * 0.8 - 300.0)

    def test_resident_exceeding_the_margin_goes_negative(self):
        # The kernel then answers None -- the caller raises with its own context.
        assert usable_vram(1000.0, resident_bytes=900.0) < 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
class TestMeasurementOnGpu:
    def test_transient_reflects_a_forward_and_scales_with_the_patch(self):
        device = torch.device("cuda:0")
        net = torch.nn.Sequential(
            torch.nn.Conv3d(1, 8, 3, padding=1), torch.nn.ReLU(), torch.nn.Conv3d(8, 1, 3, padding=1)
        ).to(device)

        def run_at(size):
            def run():
                with torch.inference_mode():
                    net(torch.zeros(1, 1, *size, device=device))

            return run

        small = measure_transient_bytes(run_at([16, 16, 16]), device)
        large = measure_transient_bytes(run_at([48, 48, 48]), device)
        assert small is not None and large is not None
        assert small > 0
        # 27x the voxels must cost markedly more -- the measurement really tracks the activations.
        assert large > small * 4

    def test_the_restart_loop_sizes_a_real_model_into_the_budget(self):
        device = torch.device("cuda:0")
        net = torch.nn.Sequential(
            torch.nn.Conv3d(1, 16, 3, padding=1), torch.nn.ReLU(), torch.nn.Conv3d(16, 1, 3, padding=1)
        ).to(device)

        def probe(patch_size):
            def run():
                with torch.inference_mode():
                    net(torch.zeros(1, 1, *patch_size, device=device))

            return measure_transient_bytes(run, device)

        whole = probe([96, 96, 96])
        assert whole is not None
        usable = whole // 2  # the whole volume does NOT fit -> the loop must shrink and still fit
        sized = run_until_fits([0, 0, 0], [96, 96, 96], probe, usable)
        assert sized is not None
        assert all(p < 96 for p in sized)
        measured = probe(sized)
        assert measured is not None and measured <= usable
