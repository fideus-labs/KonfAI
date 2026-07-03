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

"""Regression test: the konfai wheel must not bundle sibling hyphenated packages."""

from pathlib import Path

import pytest
from setuptools import find_namespace_packages

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11 ships no stdlib tomllib
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:  # nor the tomli backport
        tomllib = None  # type: ignore[assignment]

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _packages_find_config() -> dict:
    assert tomllib is not None
    pyproject = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return pyproject["tool"]["setuptools"]["packages"]["find"]


@pytest.mark.skipif(tomllib is None, reason="requires tomllib (Python 3.11+) or the tomli backport")
def test_wheel_excludes_sibling_packages_but_keeps_namespace_subpackages() -> None:
    config = _packages_find_config()
    packages = find_namespace_packages(where=str(_REPO_ROOT), include=config["include"], exclude=config["exclude"])

    # The wheel discovery used to match the sibling konfai-apps tree via the "konfai*" glob;
    # "konfai.*" (with the dot) never matches a hyphenated sibling directory.
    assert not any("-" in p.split(".")[0] for p in packages)
    assert not any(p.startswith("konfai-apps") for p in packages)

    # Namespace subpackages (no __init__.py) must still ship, or model loading breaks.
    assert "konfai" in packages
    assert "konfai.data" in packages
    assert "konfai.models.segmentation" in packages


def test_konfai_models_have_no_init_and_need_namespace_discovery() -> None:
    # Guards the reason find_packages (namespaces=false) is wrong here: konfai/models is a PEP 420
    # namespace package, so switching discovery off would silently drop the whole model zoo.
    if not (_REPO_ROOT / "konfai" / "models").is_dir():
        pytest.skip("konfai/models not present")
    assert not (_REPO_ROOT / "konfai" / "models" / "__init__.py").exists()
