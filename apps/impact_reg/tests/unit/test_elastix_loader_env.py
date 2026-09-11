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

"""The elastix binary links LibTorch out of the environment's pip ``torch``, so every process that
runs it needs the same loader path. The install probe used to run it with none, which fails on any
machine that does not already carry LibTorch, and the failure was read as a broken download."""

import os
import platform
import subprocess
from pathlib import Path

import pytest
import torch
from impact_reg_konfai.models import elastix_engine as elastix_engine_module
from impact_reg_konfai.models.elastix_engine import ElastixEngine
from impact_reg_konfai.models.elastix_install import loader_env, try_elastix


def _loader_variable() -> str:
    return {"Windows": "PATH", "Darwin": "DYLD_LIBRARY_PATH"}.get(platform.system(), "LD_LIBRARY_PATH")


def test_loader_env_names_torch_the_install_and_the_declared_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KONFAI_ELASTIX_EXTRA_LIB", os.path.join("declared", "extra"))
    install = Path("install-root")

    searched = loader_env(install)[_loader_variable()].split(os.pathsep)

    assert str(Path(torch.__file__).resolve().parent / "lib") in searched, "LibTorch comes from the pip torch"
    assert str(install / "lib") in searched, "the install's own runtime"
    # The Windows asset keeps its DLLs beside the executable, with no lib/ at all.
    assert str(install) in searched
    assert os.path.join("declared", "extra") in searched


def test_loader_env_keeps_what_the_caller_already_declared(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_loader_variable(), os.path.join("already", "there"))

    searched = loader_env(Path("install-root"))[_loader_variable()].split(os.pathsep)

    assert os.path.join("already", "there") in searched


def test_the_probe_runs_the_binary_under_the_loader_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """The probe and the registration must agree on the loader path, or the probe refuses an install
    that works and the engine downloads it again on every call."""
    seen: dict[str, object] = {}

    def capture(*args: object, **kwargs: object) -> subprocess.CompletedProcess:
        seen.update(kwargs)
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", capture)
    install = Path("install-root")

    try_elastix(install)

    assert seen.get("env") == loader_env(install), "the probe ran with a different environment"


# 127 is the POSIX loader failure; the other two are Windows STATUS_DLL_NOT_FOUND, unsigned and signed.
@pytest.mark.parametrize("code", [127, 0xC0000135, -1073741515])
def test_a_library_the_loader_cannot_find_is_named_as_such(code: int, monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing shared library aborts a child that did exec, so it never raises OSError. The OSError
    branch carried the wording, so the cause was never named."""

    def refuse(*args: object, **kwargs: object) -> subprocess.CompletedProcess:
        raise subprocess.CalledProcessError(code, ["elastix", "-h"], stderr="libtorch_cpu.so: cannot open")

    monkeypatch.setattr(subprocess, "run", refuse)

    with pytest.raises(NameError, match="shared library could not be found"):
        try_elastix(Path("install-root"))


def test_a_relative_override_survives_the_registration_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A registration runs the binary from a temporary directory. KONFAI_ELASTIX_DIR is validated from
    the startup directory, so a root kept relative sends the loader looking under the temporary one."""
    install = tmp_path / "elastix-impact"
    (install / "lib").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("KONFAI_ELASTIX_DIR", "elastix-impact")
    monkeypatch.setattr(elastix_engine_module, "try_elastix", lambda path: None)
    monkeypatch.setattr(elastix_engine_module, "get_elastix_bin", lambda path: path / "bin" / "elastix")

    engine = ElastixEngine.__new__(ElastixEngine)
    engine._ensure_binary()

    elsewhere = tmp_path / "work"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    searched = loader_env(engine._elastix_root)[_loader_variable()].split(os.pathsep)

    assert str(install.resolve() / "lib") in searched, searched
    assert all(Path(path).is_absolute() for path in searched if "elastix-impact" in path), searched
