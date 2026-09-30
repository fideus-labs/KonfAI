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

"""The konfai-mcp wheel carries the repository's examples: an installed server lists its templates
with no KonfAI checkout beside it."""

import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

_PACKAGE = Path(__file__).resolve().parents[1]


def test_the_wheel_carries_the_examples_an_installed_server_lists(tmp_path: Path) -> None:
    pytest.importorskip("build")
    pytest.importorskip("setuptools_scm")
    built = subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--no-isolation", "--outdir", str(tmp_path / "dist")],
        cwd=_PACKAGE,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert built.returncode == 0, built.stdout + built.stderr
    (wheel,) = (tmp_path / "dist").glob("*.whl")
    with zipfile.ZipFile(wheel) as archive:
        assert "konfai_mcp/examples/Segmentation/Config.yml" in archive.namelist()
        archive.extractall(tmp_path / "site")

    # The unpacked wheel first on the path, and a working directory outside any checkout.
    core = [path for path in sys.path if path and (Path(path) / "konfai" / "__init__.py").is_file()]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(tmp_path / "site"), *core])}
    probe = (
        "from konfai_mcp import server, server_support; print(server.EXAMPLES_ROOT);"
        " print(server_support.available_templates(server.EXAMPLES_ROOT))"
    )
    listed = subprocess.run([sys.executable, "-c", probe], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert listed.returncode == 0, listed.stderr
    root, templates = listed.stdout.splitlines()[-2:]
    assert Path(root) == tmp_path / "site" / "konfai_mcp" / "examples"
    assert "Segmentation" in templates
