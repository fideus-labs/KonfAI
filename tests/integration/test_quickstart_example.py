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

"""The README's Quickstart, run as a reader copies it.

The commands are read from the README itself, the ones after it enters the copied example, and run in
a copy of ``examples/Segmentation/TwoClasses``: a README that drifts from the example fails here.
"""

from __future__ import annotations

import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from harness import konfai_cli_command, subprocess_env

REPO = Path(__file__).resolve().parents[2]
EXAMPLE = REPO / "examples" / "Segmentation" / "TwoClasses"


def _readme_commands() -> list[str]:
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    section = readme.split("## Quickstart", 1)[1].split("\n## ", 1)[0]
    block = re.search(r"```bash\n(.*?)```", section, re.S)
    assert block is not None, "the README Quickstart has no bash block"
    lines = [line.strip() for line in block.group(1).splitlines() if line.strip()]
    entered = next(i for i, line in enumerate(lines) if line.endswith("cd konfai-first-run"))
    return lines[entered + 1 :]


def test_the_readme_quickstart_is_the_quickstart_page() -> None:
    """The README sends its reader on to the Quickstart page, which must describe the same run."""
    page = (REPO / "docs" / "source" / "quickstart.rst").read_text(encoding="utf-8")
    page_commands = [line.strip() for line in page.splitlines() if line.startswith("   ")]
    assert all(command in page_commands for command in _readme_commands())


@pytest.mark.integration
@pytest.mark.skipif(sys.platform == "win32", reason="the README Quickstart is a POSIX shell block")
def test_the_readme_quickstart_reaches_a_verified_run(tmp_path: Path) -> None:
    workdir = tmp_path / "konfai-first-run"
    shutil.copytree(EXAMPLE, workdir)
    shims = (
        f'konfai() {{ {shlex.join(konfai_cli_command())} "$@"; }}\n'
        f'python() {{ {shlex.quote(sys.executable)} "$@"; }}\n'
        "set -e\n"
    )
    ran = subprocess.run(
        ["bash", "-c", shims + "\n".join(_readme_commands())],
        capture_output=True,
        text=True,
        env=subprocess_env(),
        cwd=workdir,
        timeout=900,
    )
    assert ran.returncode == 0, f"the README Quickstart fails:\n{ran.stdout[-4000:]}{ran.stderr[-4000:]}"
    assert '"verified_cases": 4' in ran.stdout
