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
import torch.nn.functional as F
from konfai.data.transform import Dilate
from konfai.utils.dataset import Attribute


def _dense_cube_dilation(tensor: torch.Tensor, dilate: int) -> torch.Tensor:
    """Reference: dilation via a single dense k**n max-pool (the pre-separable implementation)."""
    data = (tensor > 0).to(torch.float32)
    k = 2 * dilate + 1
    if data.dim() - 1 == 2:
        data = F.max_pool2d(data, kernel_size=k, stride=1, padding=dilate)
    else:
        data = F.max_pool3d(data, kernel_size=k, stride=1, padding=dilate)
    return data.to(tensor.dtype)


@pytest.mark.parametrize("dilate", [1, 2, 5])
@pytest.mark.parametrize("shape", [(1, 24, 30), (2, 12, 20, 18)])
def test_dilate_separable_matches_dense_cube(shape: tuple[int, ...], dilate: int) -> None:
    # The separable 1-D max-pool implementation must be bit-identical to the dense k**n cube it replaces,
    # for both [C,H,W] and [C,D,H,W] inputs and several radii — this is the correctness guarantee that
    # lets the ~14x speedup ship as a transparent optimization.
    torch.manual_seed(0)
    mask = (torch.rand(shape) > 0.7).to(torch.uint8)

    out = Dilate(dilate)("case", mask.clone(), Attribute())
    ref = _dense_cube_dilation(mask, dilate)

    assert torch.equal(out, ref)
    assert out.dtype == mask.dtype
    assert out.shape == mask.shape


def test_dilate_single_voxel_fills_neighbourhood() -> None:
    # A single active voxel dilated by 1 must fill its full 3x3x3 neighbourhood.
    mask = torch.zeros(1, 5, 5, 5, dtype=torch.uint8)
    mask[0, 2, 2, 2] = 1

    out = Dilate(1)("case", mask.clone(), Attribute())

    assert out[0, 1:4, 1:4, 1:4].sum().item() == 27
    assert out.sum().item() == 27


def test_dilate_zero_is_identity() -> None:
    mask = (torch.rand(1, 8, 8, 8) > 0.5).to(torch.uint8)
    out = Dilate(0)("case", mask.clone(), Attribute())
    assert torch.equal(out, mask)
