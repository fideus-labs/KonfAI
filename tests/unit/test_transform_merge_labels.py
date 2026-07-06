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

import pytest
import torch
from konfai.data.transform import MergeLabels
from konfai.utils.dataset import Attribute
from konfai.utils.errors import TransformError


def _attr(nb: list[int]) -> Attribute:
    a = Attribute()
    a["number_of_channels_per_model"] = torch.tensor(nb)
    return a


def test_merge_labels_five_task_split_uses_cumulative_offsets() -> None:
    # TotalSegmentator: 5 task-split heads (bg + organs / vertebrae / cardiac / muscles / ribs).
    nb = [25, 27, 19, 24, 27]
    # Each model fires its LOCAL label 1 at its own voxel; everything else is background.
    tensor = torch.zeros(5, 5, dtype=torch.long)
    for model in range(5):
        tensor[model, model] = 1

    out = MergeLabels()("case", tensor.clone(), _attr(nb))

    # Global first-class index per model = 1 + cumulative sum of earlier foreground counts.
    assert out.tolist() == [1, 25, 51, 69, 92]


def test_merge_labels_two_models_offset_by_first_head_only() -> None:
    nb = [25, 27]
    tensor = torch.zeros(2, 3, dtype=torch.long)
    tensor[0, 0] = 3  # model 0, local 3 -> global 3
    tensor[1, 1] = 2  # model 1, local 2 -> global 2 + (25 - 1) = 26

    out = MergeLabels()("case", tensor.clone(), _attr(nb))

    assert out.tolist() == [3, 26, 0]


def test_merge_labels_single_model_is_passthrough() -> None:
    out = MergeLabels()("case", torch.tensor([[0, 5, 24]], dtype=torch.long), _attr([25]))
    assert out.tolist() == [0, 5, 24]


def test_merge_labels_background_stays_zero() -> None:
    out = MergeLabels()("case", torch.zeros(3, 4, dtype=torch.long), _attr([25, 27, 19]))
    assert out.tolist() == [0, 0, 0, 0]


def test_merge_labels_requires_number_of_channels() -> None:
    with pytest.raises(TransformError):
        MergeLabels()("case", torch.zeros(2, 3, dtype=torch.long), Attribute())
