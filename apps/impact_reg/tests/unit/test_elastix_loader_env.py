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


def test_a_library_the_loader_cannot_find_is_named_as_such(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing shared library aborts a child that did exec: return code 127, never OSError. The
    OSError branch carried the wording, so the cause was never named."""

    def refuse(*args: object, **kwargs: object) -> subprocess.CompletedProcess:
        raise subprocess.CalledProcessError(127, ["elastix", "-h"], stderr="libtorch_cpu.so: cannot open")

    monkeypatch.setattr(subprocess, "run", refuse)

    with pytest.raises(NameError, match="shared library could not be found"):
        try_elastix(Path("install-root"))
