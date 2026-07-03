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

"""Guard tests: non-functional research models fail fast with a clear message."""

import pytest
from konfai.models.generation.ddpm import DDPM
from konfai.models.registration.registration import VoxelMorph


def test_ddpm_is_marked_experimental() -> None:
    # DDPM cannot execute a forward pass (broken time-embedding wiring); constructing it must raise
    # an actionable error instead of crashing opaquely deep in the graph later.
    with pytest.raises(NotImplementedError, match="experimental"):
        DDPM()


def test_voxelmorph_rejects_3d_configuration() -> None:
    # VoxelMorph's warping components are 2-D-hardcoded, so its own dim=3 default used to crash.
    with pytest.raises(NotImplementedError, match="dim=2"):
        VoxelMorph(dim=3)
