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

"""FireANTs is installed by konfai-apps from its preset's app.json (``requirements_no_deps``); the engine only checks,
before any compute, that its deformable registrations import, and says why they do not."""

import importlib

import pytest
from impact_reg_konfai.models import fireants


def test_an_installed_runtime_passes() -> None:
    pytest.importorskip("fireants")
    fireants._require_fireants()


@pytest.mark.parametrize(
    ("missing", "message"), [("fcntl", "only Linux and macOS"), ("fireants", "requirements_no_deps")]
)
def test_a_runtime_that_does_not_import_says_why(monkeypatch: pytest.MonkeyPatch, missing: str, message: str) -> None:
    def fail(name: str, *args: object) -> None:
        raise ModuleNotFoundError(f"No module named '{missing}'", name=missing)

    monkeypatch.setattr(importlib, "import_module", fail)
    with pytest.raises(RuntimeError, match=message):
        fireants._require_fireants()
