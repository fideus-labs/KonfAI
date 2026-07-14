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

"""The ClipNormalize registry block: checkpoint-restored clip + standardize."""

import torch
from konfai.network.blocks import ClipNormalize
from konfai.utils.model_builder import list_registered_modules


def test_clip_normalize_is_registered_and_parameter_free() -> None:
    assert "ClipNormalize" in list_registered_modules()
    block = ClipNormalize()
    assert list(block.parameters()) == []  # buffers only, nothing learnable


def test_clip_normalize_restores_constants_from_the_checkpoint() -> None:
    # A per-checkpoint CT window: clip to [-1024, 276], then (x - mean) / std.
    block = ClipNormalize()
    block.load_state_dict(
        {
            "clip_min": torch.tensor([-1024.0]),
            "clip_max": torch.tensor([276.0]),
            "mean": torch.tensor([-370.0]),
            "std": torch.tensor([436.6]),
        }
    )
    x = torch.tensor([[-2000.0, -370.0, 276.0, 1000.0]])
    out = block(x)
    expected = (torch.clamp(x, -1024.0, 276.0) - (-370.0)) / 436.6
    assert torch.allclose(out, expected, atol=1e-4)
    # The clamp is active: the +1000 input is pulled down to the clip_max before standardizing.
    assert out[0, 3].item() == expected[0, 3].item()
