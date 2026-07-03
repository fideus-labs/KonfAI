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

"""Regression test: Crop.transform_shape predicts the spatial crop exactly.

Patch planning consumes transform_shape's output, so it must equal the spatial shape ``__call__``
actually produces. The pre-fix code treated ``shape[0]`` as a channel dim and paired the crop box
with ``shape[1:]``, shifting every axis by one and returning a wrong shape.
"""

import numpy as np
from konfai.data.transform import Crop
from konfai.utils.dataset import Attribute


def test_crop_transform_shape_matches_spatial_crop() -> None:
    attribute = Attribute()
    attribute["box"] = np.array([[2, 3], [1, 1], [4, 2]])  # (start, end-distance) per spatial axis

    out = Crop().transform_shape("CT", "CASE_001", [10, 20, 30], attribute)

    # 10-2-3, 20-1-1, 30-4-2 — each spatial axis cropped by its own box row.
    assert out == [5, 18, 24]
