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

import numpy as np
import torch
from konfai.data.transform import Norm
from konfai.utils.dataset import Attribute


def _stack_attribute() -> Attribute:
    # Geometry of a displacement-field stack: the leading image axis holds the vector components
    # (origin 0 / spacing 1 / identity direction row), the remaining axes carry the fixed grid.
    attribute = Attribute()
    attribute["Origin"] = np.asarray([0.0, 1.0, 2.0, 3.0])
    attribute["Spacing"] = np.asarray([1.0, 2.0, 2.0, 2.0])
    direction = np.eye(4)
    direction[1:, 1:] = np.diag([1.0, -1.0, 1.0])
    attribute["Direction"] = direction.flatten()
    return attribute


def test_norm_reduces_trailing_component_axis_and_geometry() -> None:
    # A stack of 2 displacement fields [N=2, D, H, W, C=3] -> per-sample magnitudes [2, D, H, W].
    tensors = torch.randn(2, 4, 5, 6, 3)
    attribute = _stack_attribute()

    out = Norm()("case", tensors, attribute)

    assert list(out.shape) == [2, 4, 5, 6]
    assert torch.allclose(out, torch.linalg.norm(tensors, dim=-1))
    # The reduced trailing tensor axis is the first geometry axis: it must be dropped.
    assert attribute.get_np_array("Origin").tolist() == [1.0, 2.0, 3.0]
    assert attribute.get_np_array("Spacing").tolist() == [2.0, 2.0, 2.0]
    assert attribute.get_np_array("Direction").tolist() == np.diag([1.0, -1.0, 1.0]).flatten().tolist()


def test_norm_transform_shape_drops_trailing_axis() -> None:
    assert Norm().transform_shape("group", "case", [4, 5, 6, 3], Attribute()) == [4, 5, 6]
