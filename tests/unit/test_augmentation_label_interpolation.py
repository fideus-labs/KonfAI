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

"""Regression test: geometric augmentations resample integer label maps with nearest-neighbour."""

import torch
from konfai.data.augmentation import Rotate
from konfai.utils.dataset import Attribute


def test_rotate_preserves_label_ids_with_nearest() -> None:
    # A geometric augmentation applies to every group, including uint8 segmentation targets.
    # Bilinear resampling would blend class ids into a non-existent intermediate label (1|3 -> 2);
    # nearest-neighbour must keep the label set unchanged.
    rotate = Rotate(a_min=45.0, a_max=45.0)  # a_max == a_min -> deterministic 45 degrees
    labels = torch.zeros(1, 32, 32, dtype=torch.uint8)
    labels[:, 8:24, 8:24] = 1
    labels[:, 12:20, 12:20] = 3
    rotate._state_init(0, [[32, 32]], [Attribute()])

    out = rotate._compute("case", 0, [labels.clone()])[0]

    assert out.dtype == torch.uint8
    assert set(out.unique().tolist()).issubset({0, 1, 3})
    assert 2 not in out.unique().tolist()
