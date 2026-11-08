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

"""The unit every engine measures lengths in: what lets a preset tuned on anatomy register a micrometre organoid."""

import numpy as np
import pytest
import torch
from impact_reg_konfai.models.intensity import EngineRegistration
from konfai.utils.dataset import Attribute


def test_the_engine_measures_lengths_in_voxels() -> None:
    # The presets' settings in millimetres were failing on a micrometre pair (a 0.15 mm organoid folded 97 % of its
    # voxels with ConvexAdam). In voxels, the engine sees the 0.2 x 0.4 um voxel as 1 x 2, and the field it returns
    # comes back in the images' millimetres.
    seen = {}

    class Engine:
        def register(self, fixed, moving, device_index, fixed_mask, moving_mask):
            seen.update(spacing=fixed.GetSpacing(), origin=fixed.GetOrigin(), mask=fixed_mask.GetSpacing())
            return np.full((3, 4, 4, 4), 0.5, dtype=np.float32)  # half a voxel

    geometry = Attribute()
    geometry["Origin"] = np.full(3, 0.01)
    geometry["Spacing"] = np.array([0.0002, 0.0002, 0.0004])
    geometry["Direction"] = np.eye(3).flatten()
    tensor = torch.zeros(1, 1, 4, 4, 4)
    field = EngineRegistration(Engine())(tensor, tensor, tensor + 1, tensor + 1, [[geometry]] * 4)
    assert seen["spacing"] == pytest.approx((1.0, 1.0, 2.0))
    assert seen["origin"] == pytest.approx((50.0,) * 3)
    assert seen["mask"] == pytest.approx((1.0, 1.0, 2.0))
    np.testing.assert_allclose(field.numpy(), 0.0001)  # half the finest real voxel


def test_a_linear_stage_steps_its_translation_apart_from_its_rotation() -> None:
    # FireANTs gives the translation (a length, in voxels) and the rotation or matrix (unitless) one Adam step: halving
    # it to move the translation in half-voxels would have halved the rotation too. One tensor with two step sizes
    # follows the two separate Adams exactly.
    from impact_reg_konfai.models.fireants import _StepPerElement

    target = torch.tensor([3.0, -2.0, 1.0, 4.0])
    together = torch.nn.Parameter(torch.zeros(4))
    sizes = torch.tensor([0.1, 0.1, 0.05, 0.05])
    stepped = _StepPerElement(torch.optim.Adam([together], lr=1.0), [(together, sizes)])
    apart = [torch.nn.Parameter(torch.zeros(2)), torch.nn.Parameter(torch.zeros(2))]
    reference = torch.optim.Adam([{"params": [apart[0]], "lr": 0.1}, {"params": [apart[1]], "lr": 0.05}])
    for _ in range(25):
        for optimizer, loss in (
            (stepped, ((together - target) ** 2).sum()),
            (reference, ((torch.cat(apart) - target) ** 2).sum()),
        ):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    torch.testing.assert_close(together.detach(), torch.cat(apart).detach())
