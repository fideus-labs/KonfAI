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

"""Contract test for the ``konfai`` CORE symbols imported by the external 3D Slicer extensions.

Companion to ``konfai-apps/tests/test_slicer_api_contract.py`` (the app-layer surface). SlicerKonfAI and
SlicerImpactReg live in separate repositories outside this CI and import these core symbols directly, so a
refactor that renames/removes one breaks a clinician's Slicer session instead of failing here. Locks only
what those extensions actually import (grepped from the Slicer repos); add a row if Slicer starts using more.
"""

import importlib
import inspect

import pytest

# (module, symbol, kind, expected parameter names) -- from `grep -r "from konfai\\." Slicer*`.
CONTRACT = [
    ("konfai.evaluator", "Statistics", "class", []),
    ("konfai.utils.dataset", "image_to_data", "func", ["image"]),
    ("konfai.utils.dataset", "get_infos", "func", ["filename"]),
    ("konfai.utils.runtime", "MinimalLog", "class", []),
    ("konfai.utils.errors", "AppRepositoryError", "class", []),
    # Top-level konfai/__init__ helpers the Slicer extensions import for server-check, device/RAM/VRAM
    # display, and the dependency self-test (same exposure class as current_free_vram).
    ("konfai", "get_available_devices", "func", ["remote_server", "timeout_s"]),
    ("konfai", "get_ram", "func", ["remote_server", "timeout_s"]),
    ("konfai", "get_vram", "func", ["devices", "remote_server", "timeout_s"]),
    ("konfai", "check_server", "func", ["remote_server", "timeout_s"]),
    ("konfai", "assert_konfai_install", "func", []),
]


@pytest.mark.parametrize("module_path, name, kind, params", CONTRACT, ids=[f"{c[0]}.{c[1]}" for c in CONTRACT])
def test_slicer_consumed_core_symbol_is_stable(module_path: str, name: str, kind: str, params: list[str]) -> None:
    module = importlib.import_module(module_path)
    obj = getattr(module, name, None)
    assert obj is not None, (
        f"{module_path}.{name} is imported by SlicerKonfAI/SlicerImpactReg but no longer exists. "
        "Removing/renaming it breaks the external Slicer extensions -- keep it, or update those repos."
    )

    if kind == "class":
        assert inspect.isclass(obj), f"{module_path}.{name} must stay a class."
        return

    assert callable(obj) and not inspect.isclass(obj), f"{module_path}.{name} must stay a callable."
    signature = inspect.signature(obj)
    actual = set(signature.parameters)
    missing = set(params) - actual
    assert not missing, f"{module_path}.{name} lost parameter(s) {sorted(missing)} that Slicer passes."
    new_required = {
        pname
        for pname, param in signature.parameters.items()
        if param.default is inspect.Parameter.empty
        and param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
        and pname not in params
    }
    assert not new_required, f"{module_path}.{name} added required parameter(s) {sorted(new_required)}; Slicer breaks."
