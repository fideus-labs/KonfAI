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

"""Make this directory importable so integration tests can ``from harness import ...`` under any import mode."""

import sys
from pathlib import Path

import pytest

_INTEGRATION_DIR = str(Path(__file__).resolve().parent)
if _INTEGRATION_DIR not in sys.path:
    sys.path.insert(0, _INTEGRATION_DIR)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Every test here runs KonfAI as a spawned process, and on the macOS runners that process dies
    with SIGSEGV often enough that no run of the suite completes: four different tests over six
    runs, on four Python versions, where the same code passed the same tests on other runs. The
    workflows they cover are not platform-specific and run in full on ubuntu and windows.
    """
    if sys.platform != "darwin":
        return
    skip = pytest.mark.skip(reason="a spawned KonfAI process dies with SIGSEGV on the macOS runners")
    for item in items:
        item.add_marker(skip)
