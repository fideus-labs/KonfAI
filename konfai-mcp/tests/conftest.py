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

import importlib
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
TESTS_ROOT = Path(__file__).resolve().parent
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))


@pytest.fixture
def load_mcp_server() -> Iterator[Callable[[], ModuleType]]:
    loaded_modules: list[ModuleType] = []

    def _load() -> ModuleType:
        module = importlib.import_module("konfai_mcp.server")
        module = importlib.reload(module)
        loaded_modules.append(module)
        return module

    yield _load

    for module in reversed(loaded_modules):
        jobs = list(getattr(module, "_JOBS", {}).values())
        for job in jobs:
            try:
                module.cancel_job(job.job_id, wait_s=1.0)
            except Exception:
                continue
    for module_name in (
        "konfai_mcp.server",
        "konfai_mcp.server_experiments",
        "konfai_mcp.server_jobs",
        "konfai_mcp.server_support",
        "konfai_mcp.runner",
    ):
        sys.modules.pop(module_name, None)
