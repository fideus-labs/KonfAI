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

"""Regression tests for the performance hot-path fixes (see AUDIT.md — Performance backlog)."""

import os

os.environ.setdefault("KONFAI_config_file", "/tmp/konfai-none.yml")
os.environ.setdefault("KONFAI_CONFIG_MODE", "Done")

from pathlib import Path  # noqa: E402
from unittest.mock import MagicMock  # noqa: E402

import torch  # noqa: E402

from konfai.data.patching import Accumulator  # noqa: E402
from konfai.predictor import ModelComposite  # noqa: E402


def test_accumulator_is_full_counts_without_rescanning():
    """P6: is_full() is O(1) — a running counter, idempotent adds, reset on assemble."""
    patch_slices = [(slice(0, 2), slice(0, 2)), (slice(0, 2), slice(2, 4))]
    acc = Accumulator(patch_slices, [0, 0], None, batch=False)

    assert acc.is_full() is False
    acc.add_layer(0, torch.ones(1, 2, 2))
    assert acc.is_full() is False
    acc.add_layer(1, torch.ones(1, 2, 2) * 2)
    assert acc.is_full() is True

    # Re-adding the same index must not double-count.
    acc.add_layer(1, torch.ones(1, 2, 2) * 3)
    assert acc._filled == 2
    assert acc.is_full() is True

    acc.assemble()
    assert acc._filled == 0
    assert acc.is_full() is False


def test_ensemble_reads_each_checkpoint_once_across_batches():
    """P1: a local-path ensemble reads/unpickles each checkpoint once, not once per batch."""
    mc = ModelComposite.__new__(ModelComposite)
    mc._base_model_name = "Model"
    mc._state_sources = [Path("/fake/ckpt_0.pt"), Path("/fake/ckpt_1.pt"), Path("/fake/ckpt_2.pt")]
    mc._loaded_state_index = None
    mc._state_cache = {}
    mc._get_model = lambda: MagicMock()

    reads: list[str] = []

    def fake_read(src):
        reads.append(str(src))
        return {"w": str(src)}

    mc._read_state_source = fake_read

    # Four forward passes, each looping over all three sub-models (as forward() does).
    for _batch in range(4):
        for idx in range(3):
            mc._ensure_model_loaded(idx)

    assert len(reads) == 3, f"expected 3 disk reads (one per index), got {len(reads)}"
    assert set(reads) == {"/fake/ckpt_0.pt", "/fake/ckpt_1.pt", "/fake/ckpt_2.pt"}

    # load() must invalidate the stale cache when the sources change.
    mc.load([Path("/other.pt")])
    assert 1 not in mc._state_cache and 2 not in mc._state_cache
    assert mc._state_cache.get(0) == {"w": "/other.pt"}
