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

"""accepts_init is read from the criterion, not its CriterionsAttr record."""

from konfai.metric.measure import KLDivergence
from konfai.network.network import CriterionsAttr


def test_accepts_init_flag_lives_on_the_criterion_not_the_attr() -> None:
    # Measure.init must read the capability flag from the criterion (the dict key), not from the
    # CriterionsAttr value (which never carries it) — otherwise CriterionWithInit.init() is skipped
    # and graph-rewiring criteria such as KLDivergence train against the wrong channels silently.
    criterion = KLDivergence(shape=[16, 16])
    assert getattr(criterion, "accepts_init", False) is True
    assert getattr(CriterionsAttr(), "accepts_init", False) is False
