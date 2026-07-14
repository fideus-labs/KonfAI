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

"""Contract test for the ``konfai_apps`` public surface consumed by the external 3D Slicer extensions.

SlicerKonfAI (and SlicerImpactReg) live in separate repositories that are NOT in this CI, so a refactor
that renames/removes a symbol they import breaks a clinician's Slicer session instead of failing here.
This test locks the SPECIFIC symbols those extensions import today (derived by grepping the Slicer repos),
at the signature level, so such a break fails konfai-apps CI first. It intentionally does NOT freeze the
whole API -- only what Slicer actually uses. If Slicer starts using a new symbol, add it to CONTRACT below.

Incident this guards against: dropping ``current_free_vram`` from ``app_repository`` broke SlicerKonfAI.
"""

import inspect

import pytest

# (module, symbol, kind, expected parameter names) -- funcs only list params; classes leave it empty.
# Sourced from `grep -r "from konfai_apps" Slicer*` (see module docstring).
CONTRACT = [
    ("konfai_apps.app_repository", "current_free_vram", "func", ["devices", "remote_server"]),
    ("konfai_apps.app_repository", "is_app_repo", "func", ["filenames"]),
    ("konfai_apps.app_repository", "get_app_repository_info", "func", ["app_id", "force_update"]),
    ("konfai_apps.app_repository", "LocalAppRepositoryFromDirectory", "class", []),
    ("konfai_apps.app_repository", "LocalAppRepositoryFromHF", "class", []),
    ("konfai_apps.app_repository", "AppRepositoryError", "class", []),
    # HF / remote-server app-listing symbols the Slicer "Add from HF" / "Add from remote" flows import.
    ("konfai_apps.app_repository", "get_available_apps_on_hf_repo", "func", ["repo_id", "force_update"]),
    ("konfai_apps.app_repository", "get_available_apps_on_remote_server", "func", ["remote_server"]),
    ("konfai_apps.app_repository", "AppRepositoryInfoFromRemoteServer", "class", []),
]


@pytest.mark.parametrize("module_path, name, kind, params", CONTRACT, ids=[f"{c[0]}.{c[1]}" for c in CONTRACT])
def test_slicer_consumed_symbol_is_stable(module_path: str, name: str, kind: str, params: list[str]) -> None:
    import importlib

    module = importlib.import_module(module_path)
    obj = getattr(module, name, None)
    assert obj is not None, (
        f"{module_path}.{name} is imported by SlicerKonfAI/SlicerImpactReg but no longer exists. "
        "Removing/renaming it breaks the external Slicer extensions -- keep it, or update those repos."
    )

    if kind == "class":
        assert inspect.isclass(obj), f"{module_path}.{name} must stay a class (Slicer instantiates it)."
        return

    assert callable(obj) and not inspect.isclass(obj), f"{module_path}.{name} must stay a callable."
    signature = inspect.signature(obj)
    actual = set(signature.parameters)
    # Every parameter Slicer relies on must still exist (a rename would break its call sites)...
    missing = set(params) - actual
    assert not missing, f"{module_path}.{name} lost parameter(s) {sorted(missing)} that Slicer passes."
    # ...and no NEW required (no-default) parameter may appear, or Slicer's existing calls break.
    new_required = {
        pname
        for pname, param in signature.parameters.items()
        if param.default is inspect.Parameter.empty
        and param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
        and pname not in params
    }
    assert not new_required, (
        f"{module_path}.{name} added required parameter(s) {sorted(new_required)}; Slicer's calls will break."
    )
