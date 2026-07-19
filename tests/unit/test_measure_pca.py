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

"""``IMPACTReg._pca_project`` under a batch > 1.

itk-impact fits one PCA basis per image. KonfAI batches unrelated cases together, so the basis must be
fitted per batch sample; a single shared basis projects every sample after the first into another case's
feature space (silently annihilating its loss term)."""

import torch
from konfai.metric.measure import IMPACTReg


def _core(pca: int) -> IMPACTReg:
    core = IMPACTReg.__new__(IMPACTReg)
    torch.nn.Module.__init__(core)
    core.pca = pca
    return core


def _two_sample_batch() -> tuple[torch.Tensor, torch.Tensor]:
    """A batch whose two samples put their variance in different channels: sample 0 in channel 0,
    sample 1 in channel 3. A shared basis (fitted on sample 0) has no support on channel 3."""
    torch.manual_seed(0)
    fixed = torch.zeros(2, 4, 6, 6, 6)
    fixed[0, 0] = torch.randn(6, 6, 6) * 10.0
    fixed[0, 1:] = torch.randn(3, 6, 6, 6) * 0.01
    fixed[1, 3] = torch.randn(6, 6, 6) * 10.0
    fixed[1, :3] = torch.randn(3, 6, 6, 6) * 0.01
    moved = fixed + torch.randn_like(fixed) * 0.1
    return moved, fixed


def test_second_sample_uses_its_own_basis() -> None:
    """Sample 1 projected inside a B=2 batch must equal sample 1 projected alone (its own basis),
    NOT be collapsed by sample 0's basis (which drops the captured variance to ~1e-4 instead of ~100)."""
    core = _core(1)
    moved, fixed = _two_sample_batch()

    _, fp_batched = core._pca_project(moved, fixed)
    _, fp_alone = core._pca_project(moved[1:2], fixed[1:2])

    assert torch.allclose(fp_batched[1:2], fp_alone, atol=1e-4)
    # Sample 1's dominant channel carries variance ~O(100); a wrong (shared) basis collapses it to ~0.
    assert float(fp_batched[1].var()) > 1.0


def test_batch_equals_per_sample_stack() -> None:
    """Each sample of a batched projection equals that sample projected on its own."""
    core = _core(2)
    moved, fixed = _two_sample_batch()

    mp, fp = core._pca_project(moved, fixed)
    for b in range(moved.shape[0]):
        mp_b, fp_b = core._pca_project(moved[b : b + 1], fixed[b : b + 1])
        assert torch.allclose(mp[b : b + 1], mp_b, atol=1e-5)
        assert torch.allclose(fp[b : b + 1], fp_b, atol=1e-5)


def test_single_sample_shapes_unchanged() -> None:
    """The B=1 path (the only one exercised by the shipped configs today) keeps its shape contract."""
    core = _core(2)
    moved, fixed = torch.randn(1, 6, 3, 3, 3), torch.randn(1, 6, 3, 3, 3)
    mp, fp = core._pca_project(moved, fixed)
    assert mp.shape == (1, 2, 3, 3, 3) and fp.shape == (1, 2, 3, 3, 3)
    assert torch.isfinite(mp).all() and torch.isfinite(fp).all()
