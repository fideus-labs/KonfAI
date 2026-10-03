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

"""Contract test for the ``konfai_apps`` names the external 3D Slicer extensions depend on.

SlicerKonfAI (and SlicerImpactReg) live in separate repositories that are NOT in this CI, so a refactor
that renames/removes a symbol they use breaks a clinician's Slicer session instead of failing here.
This test locks what SlicerKonfAI uses: the names it imports (CONTRACT), the methods it calls on an app
repository with the arguments it passes (METHODS) and the attributes it reads (test at the end). It
intentionally does NOT freeze the whole API: if Slicer starts using something new, add it here.

Incident this guards against: dropping ``current_free_vram`` from ``app_repository`` broke SlicerKonfAI.


SlicerKonfAI and SlicerImpactReg live in repositories outside this CI, so a refactor that renames or
removes a name they import breaks a clinician's Slicer session instead of failing here. This test locks
the names of CONTRACT at the signature level, so such a break fails konfai-apps CI first. It does not
freeze the whole API: when Slicer imports a new name, add it to CONTRACT; before removing one, grep the
Slicer repositories (AGENTS.md §7b).
"""

import inspect
import subprocess  # nosec B404
import sys

import pytest

# (module, symbol, kind, expected parameter names): funcs only list params; classes leave it empty.
CONTRACT = [
    # SlicerKonfAI main no longer imports it; the releases the Extension Manager still serves may.
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
        "Removing/renaming it breaks the external Slicer extensions: keep it, or update those repos."
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


def test_the_repository_module_loads_without_torch_or_simpleitk() -> None:
    """Slicer imports ``konfai_apps.app_repository`` in its own interpreter to list and resolve apps."""
    code = "import sys, konfai_apps.app_repository; print(sorted({'torch', 'SimpleITK'} & set(sys.modules)))"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)  # nosec B603
    assert result.stdout.strip() == "[]"


_LOCAL = ("LocalAppRepositoryFromDirectory", "LocalAppRepositoryFromHF")
_ANY = (*_LOCAL, "AppRepositoryInfoFromRemoteServer")

# (classes, method, positional arguments Slicer passes, keywords it passes). An app from a remote server
# never reaches the fine-tune, the export or the parameters dialog, and get_filenames is static.
METHODS = [
    *[
        (_ANY, name, 0, ())
        for name in (
            "get_name",
            "get_display_name",
            "get_description",
            "get_short_description",
            "get_checkpoints_name",
            "get_checkpoints_name_available",
            "get_maximum_tta",
            "get_mc_dropout",
            "get_patch_size",
            "get_terminology",
            "has_capabilities",
            "download_config_file",
        )
    ],
    (_LOCAL, "get_parameters", 0, ()),
    (_LOCAL, "install_fine_tune", 6, ()),
    (_LOCAL, "export_app", 1, ("display_name", "config_overrides")),
    (("LocalAppRepositoryFromDirectory",), "get_filenames", 2, ()),
    (("LocalAppRepositoryFromHF",), "get_filenames", 3, ()),
]


@pytest.mark.parametrize(
    "class_name, method, positional, keywords",
    [(cls, *row[1:]) for row in METHODS for cls in row[0]],
    ids=[f"{cls}.{row[1]}" for row in METHODS for cls in row[0]],
)
def test_slicer_called_method_accepts_its_arguments(
    class_name: str, method: str, positional: int, keywords: tuple[str, ...]
) -> None:
    from konfai_apps import app_repository

    cls = getattr(app_repository, class_name)
    assert hasattr(cls, method), f"SlicerKonfAI calls {class_name}.{method}: keep it, or update SlicerKonfAI."
    arguments = [None] * positional
    if not isinstance(inspect.getattr_static(cls, method), staticmethod):
        arguments.insert(0, None)  # self
    try:
        inspect.signature(getattr(cls, method)).bind(*arguments, **dict.fromkeys(keywords))
    except TypeError as error:
        pytest.fail(f"SlicerKonfAI's call to {class_name}.{method} no longer binds: {error}")


def test_slicer_read_attributes_are_set(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SlicerKonfAI reads ``_app_directory``/``_app_name`` of a local app and ``_repo_id``/``_app_name`` of a
    Hugging Face one to list and open their files."""
    import json

    from konfai_apps import app_repository

    app_dir = tmp_path / "demo_app"
    app_dir.mkdir()
    keys = ("display_name", "description", "short_description")
    (app_dir / "app.json").write_text(json.dumps({**dict.fromkeys(keys, "d"), "tta": 0, "mc_dropout": 0}))

    local = app_repository.LocalAppRepositoryFromDirectory(tmp_path, "demo_app")
    assert (local._app_directory, local._app_name) == (tmp_path, "demo_app")

    hf = app_repository.LocalAppRepositoryFromHF
    monkeypatch.setattr(hf, "_get_filenames", lambda self: ["app.json"])
    monkeypatch.setattr(hf, "_download", lambda self, filename: app_dir / "app.json")
    monkeypatch.setattr(hf, "get_cached_filenames", staticmethod(lambda repo_id, app_name: []))
    remote = hf("org/demo", "demo_app", False)
    assert (remote._repo_id, remote._app_name) == ("org/demo", "demo_app")
