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

"""Packaging and public-API surface: clean imports (with and without optional deps),
error-type message formatting, and wheel package discovery."""

import subprocess
import sys
from pathlib import Path

import konfai
import pytest
from konfai.utils.errors import ConfigError, KonfAIError, TransformError
from setuptools import find_namespace_packages

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11 ships no stdlib tomllib
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:  # nor the tomli backport
        tomllib = None  # type: ignore[assignment]

_REPO_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# The konfai package imports cleanly with no optional deps.
# --------------------------------------------------------------------------- #
def test_import_konfai_succeeds() -> None:
    import konfai  # noqa: F401


def test_version_attribute_exists() -> None:
    import konfai

    assert hasattr(konfai, "__version__"), "konfai must expose __version__"
    assert isinstance(konfai.__version__, str)
    assert konfai.__version__  # non-empty


def test_konfai_utils_config_imports_without_simpleitk() -> None:
    """konfai.utils.config must be importable even if SimpleITK is absent."""
    script = """
import builtins

real_import = builtins.__import__

def import_without_simpleitk(name, *args, **kwargs):
    if name == "SimpleITK":
        raise ImportError("SimpleITK unavailable")
    return real_import(name, *args, **kwargs)

builtins.__import__ = import_without_simpleitk
import konfai.utils.config
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)


def test_transform_and_blocks_import_without_simpleitk() -> None:
    """Modules using SimpleITK guard it at point-of-use, so import must succeed without it."""
    script = """
import builtins

real_import = builtins.__import__

def import_without_simpleitk(name, *args, **kwargs):
    if name == "SimpleITK":
        raise ImportError("SimpleITK unavailable")
    return real_import(name, *args, **kwargs)

builtins.__import__ = import_without_simpleitk
import konfai.data.transform
import konfai.network.blocks

assert konfai.data.transform.sitk is None
assert konfai.network.blocks.sitk is None
try:
    konfai.data.transform._require_simpleitk()
except Exception as exc:
    assert "pip install konfai[itk]" in str(exc), str(exc)
else:
    raise AssertionError("_require_simpleitk must raise without SimpleITK")
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)


# --------------------------------------------------------------------------- #
# Package-level import contracts and public API surface.
# --------------------------------------------------------------------------- #
def test_package_importable() -> None:
    import konfai

    assert isinstance(konfai.__version__, str)
    assert konfai.__version__


def test_config_module_importable() -> None:
    from konfai.utils.config import Config, apply_config, config  # noqa: F401


def test_errors_module_importable() -> None:
    from konfai.utils.errors import ConfigError, KonfAIError, TrainerError

    assert issubclass(KonfAIError, Exception)
    assert issubclass(ConfigError, Exception)
    assert issubclass(TrainerError, Exception)


def test_local_vram_query_requires_monitoring_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(konfai, "_PYNVML_AVAILABLE", False)

    with pytest.raises(KonfAIError, match="nvidia-ml-py"):
        konfai.get_vram([0])


def test_itk_helper_requires_simpleitk(monkeypatch: pytest.MonkeyPatch) -> None:
    import konfai.utils.ITK as itk_module

    monkeypatch.setattr(itk_module, "sitk", None)

    with pytest.raises(TransformError, match="SimpleITK"):
        itk_module.resample_resize(None)


def test_main_module_importable() -> None:
    import konfai.main

    assert callable(konfai.main.main)


# --------------------------------------------------------------------------- #
# KonfAI error types format their messages correctly.
# --------------------------------------------------------------------------- #
def test_named_error_formats_with_type_prefix() -> None:
    error = ConfigError("bad value")

    assert "[Config]" in str(error)
    assert "bad value" in str(error)


def test_named_error_with_multiple_messages_uses_arrow() -> None:
    error = ConfigError("bad value", "expected int", "got str")

    assert "→" in str(error)


def test_konfai_error_without_args_returns_empty_bracket() -> None:
    error = KonfAIError()

    result = str(error)
    assert "[Error]" in result
    assert result


# --------------------------------------------------------------------------- #
# The konfai wheel must not bundle sibling hyphenated packages.
# --------------------------------------------------------------------------- #
def _packages_find_config() -> dict:
    assert tomllib is not None
    pyproject = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return pyproject["tool"]["setuptools"]["packages"]["find"]


@pytest.mark.skipif(tomllib is None, reason="requires tomllib (Python 3.11+) or the tomli backport")
def test_wheel_excludes_sibling_packages_but_keeps_namespace_subpackages() -> None:
    config = _packages_find_config()
    packages = find_namespace_packages(where=str(_REPO_ROOT), include=config["include"], exclude=config["exclude"])

    # A bare "konfai*" glob matches the sibling konfai-apps tree;
    # "konfai.*" (with the dot) never matches a hyphenated sibling directory.
    assert not any("-" in p.split(".")[0] for p in packages)
    assert not any(p.startswith("konfai-apps") for p in packages)

    # Namespace subpackages (no __init__.py) must still ship, or model loading breaks.
    assert "konfai" in packages
    assert "konfai.data" in packages
    assert "konfai.models.python.segmentation" in packages


def test_konfai_models_have_no_init_and_need_namespace_discovery() -> None:
    # Guards the reason find_packages (namespaces=false) is wrong here: konfai/models is a PEP 420
    # namespace package, so switching discovery off would silently drop the whole model zoo.
    if not (_REPO_ROOT / "konfai" / "models").is_dir():
        pytest.skip("konfai/models not present")
    assert not (_REPO_ROOT / "konfai" / "models" / "__init__.py").exists()
