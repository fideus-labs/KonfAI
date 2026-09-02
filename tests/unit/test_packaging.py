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

import json
import os
import subprocess
import sys
import sysconfig
import zipfile
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


def test_transform_imports_without_simpleitk() -> None:
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
import konfai.utils.ITK

assert konfai.data.transform.sitk is None
try:
    konfai.utils.ITK._require_simpleitk()
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
        itk_module.read_displacement_field("missing.mha")


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
def test_the_declared_torch_floor_covers_the_dtypes_the_pipeline_reads() -> None:
    """A uint16 store reaches ``torch.from_numpy``, and ``torch.uint16`` is named where a label
    map's sign is read: both are torch 2.3. Without a floor, pip is free to resolve an older torch
    and the run fails on a dtype instead of at install time."""
    assert tomllib is not None
    pyproject = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    torch_requirement = next(dep for dep in pyproject["project"]["dependencies"] if dep.startswith("torch"))
    floor = torch_requirement.removeprefix("torch>=")
    assert floor != torch_requirement, f"no lower bound on torch: {torch_requirement!r}"
    assert tuple(int(part) for part in floor.split(".")) >= (2, 3)


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


# --------------------------------------------------------------------------- #
# The built wheel ships the PEP 420 model zoo, the YAML catalog, and nothing else.
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def built_wheel(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One wheel built from the source tree, shared by the tests that inspect and install it."""
    pytest.importorskip("build")
    # --no-isolation builds offline, so every declared build requirement must be importable.
    pytest.importorskip("setuptools")
    pytest.importorskip("wheel")
    pytest.importorskip("setuptools_scm")

    outdir = tmp_path_factory.mktemp("wheel")
    result = subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--no-isolation", "--outdir", str(outdir)],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, f"wheel build failed:\n{result.stdout}\n{result.stderr}"
    wheels = list(outdir.glob("*.whl"))
    assert len(wheels) == 1, wheels
    return wheels[0]


@pytest.mark.slow
def test_wheel_ships_model_zoo_and_catalog(built_wheel: Path) -> None:
    """Inspect the built wheel: an editable install hides PEP 420 and package-data
    breakage, so only the built artifact proves the packaging contract."""
    with zipfile.ZipFile(built_wheel) as archive:
        names = set(archive.namelist())

    # Expected contents derive from the source tree, so a catalog addition never fails this test.
    models = _REPO_ROOT / "konfai" / "models"
    expected_python = {p.relative_to(_REPO_ROOT).as_posix() for p in (models / "python").rglob("*.py")}
    expected_yaml = {p.relative_to(_REPO_ROOT).as_posix() for p in (models / "yaml").glob("*.yml")}
    assert expected_python and expected_yaml

    missing_python = sorted(expected_python - names)
    assert not missing_python, f"models/python files missing from the wheel: {missing_python}"
    missing_yaml = sorted(expected_yaml - names)
    assert not missing_yaml, f"catalog .yml files missing from the wheel: {missing_yaml}"

    # PEP 561: without the marker every downstream type checker treats konfai as untyped.
    assert "konfai/py.typed" in names, "py.typed missing from the wheel"

    forbidden_top_level = {"apps", "konfai-apps", "konfai_apps", "konfai-mcp", "konfai_mcp"}
    leaked = sorted(n for n in names if n.split("/", 1)[0] in forbidden_top_level)
    assert not leaked, f"sibling package leaked into the wheel: {leaked[:5]}"

    # konfai/models and konfai/models/python are PEP 420 namespace packages; a generated
    # __init__.py in the wheel would mean package discovery regressed.
    assert "konfai/models/__init__.py" not in names
    assert "konfai/models/python/__init__.py" not in names


def _venv_binary(venv_dir: Path, name: str) -> Path:
    """The path of an executable inside a venv, on either layout."""
    if os.name == "nt":
        return venv_dir / "Scripts" / f"{name}.exe"
    return venv_dir / "bin" / name


@pytest.mark.slow
def test_the_installed_wheel_imports_the_model_zoo_and_runs_the_cli(built_wheel: Path, tmp_path: Path) -> None:
    """Install the wheel NON-editable and use it: an archive holding the files does not prove the
    installed distribution imports them. ``konfai.models.python`` is a PEP 420 namespace package,
    the catalog is package-data and ``konfai`` is a console script, and an editable install resolves
    all three through the source tree whatever the wheel contains (AGENTS.md 7d).

    The venv is built without the host's packages, so ``konfai`` can only come from the wheel; the
    runtime dependencies are appended to its ``sys.path`` through a ``.pth`` file, which adds the
    path without running the host's own ``.pth`` files -- one of which installs the editable
    ``konfai`` finder that would shadow the installed package.
    """
    venv_dir = tmp_path / "venv"
    subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True, capture_output=True, timeout=300)
    python = _venv_binary(venv_dir, "python")
    # PYTHONPATH, and the working directory both python -c and the CLI put on sys.path, would offer
    # the source tree in front of the installed package: the very thing this test exists to bypass.
    # tmp_path holds only the venv, so nothing there answers to "konfai".
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}

    installed = subprocess.run(
        [str(python), "-m", "pip", "install", "--no-deps", "--no-index", str(built_wheel)],
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
    )
    assert installed.returncode == 0, f"wheel install failed:\n{installed.stdout}\n{installed.stderr}"

    site_packages = subprocess.run(
        [str(python), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
        env=env,
        cwd=tmp_path,
    ).stdout.strip()
    (Path(site_packages) / "_runtime_dependencies.pth").write_text(
        f"{sysconfig.get_paths()['purelib']}\n", encoding="utf-8"
    )

    probe = """
import json, sys
from pathlib import Path
import konfai, konfai.models.python

prefix = Path(sys.prefix).resolve()
package = Path(konfai.__file__).resolve().parent
assert package.is_relative_to(prefix), f"konfai resolved outside the venv: {package}"
zoo = [Path(entry).resolve() for entry in konfai.models.python.__path__]
assert zoo and all(entry.is_relative_to(prefix) for entry in zoo), zoo
print(json.dumps(sorted(path.name for path in (package / "models" / "yaml").glob("*.yml"))))
"""
    result = subprocess.run(
        [str(python), "-c", probe], capture_output=True, text=True, timeout=300, env=env, cwd=tmp_path
    )
    assert result.returncode == 0, f"the installed package does not import:\n{result.stdout}\n{result.stderr}"
    catalog = json.loads(result.stdout.strip().splitlines()[-1])
    assert catalog == sorted(path.name for path in (_REPO_ROOT / "konfai" / "models" / "yaml").glob("*.yml"))

    cli = _venv_binary(venv_dir, "konfai")
    assert cli.exists(), f"the wheel installs no 'konfai' console script: {sorted(cli.parent.iterdir())}"
    version = subprocess.run(
        [str(cli), "--version"],
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
        cwd=tmp_path,
    )
    assert version.returncode == 0, f"konfai --version failed:\n{version.stdout}\n{version.stderr}"
    # The wheel's file name carries the version its metadata was built with: the CLI reading the same
    # one proves it reports the installed distribution and not another konfai on the path.
    assert version.stdout.strip() == built_wheel.name.split("-")[1]
