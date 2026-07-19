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

"""Release gate: build the konfai wheel and prove a clean NON-EDITABLE install ships what users need.

CI installs editable, which hides PEP 420 (`konfai/models/python` has no `__init__.py`) and package-data
(the 14 catalog `.yml`) breakage. This builds the wheel, installs it into a throwaway venv, and asserts the
catalog resolves. Run before tagging:

    python .claude/skills/konfai-maintainer/scripts/check_release_ready.py

Exit code 0 = ready, non-zero = a regression that would ship green. Needs `build` available in the current
interpreter (the pixi dev env has it); the temp venv installs the wheel with pip only.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
MIN_PY_MODELS = 16
EXPECTED_YAML = 14


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        dist = tmp / "dist"
        print("building wheel ...")
        run([sys.executable, "-m", "build", "--wheel", "--outdir", str(dist), str(REPO)])
        wheel = next(dist.glob("konfai-*.whl"))

        names = zipfile.ZipFile(wheel).namelist()
        n_py = sum(1 for n in names if "/models/python/" in n and n.endswith(".py"))
        n_yaml = sum(1 for n in names if "/models/yaml/" in n and n.endswith(".yml"))
        leaked = [n for n in names if n.split("/")[0].endswith("-apps") or "konfai_apps" in n]
        assert n_py >= MIN_PY_MODELS, f"wheel ships only {n_py} models/python files (< {MIN_PY_MODELS})"
        assert n_yaml == EXPECTED_YAML, f"wheel ships {n_yaml} catalog .yml (expected {EXPECTED_YAML})"
        assert not leaked, f"hyphenated sibling leaked into the wheel: {leaked[:3]}"
        print(f"wheel content OK: {n_py} models/python, {n_yaml} catalog .yml, no sibling leak")

        venv = tmp / "venv"
        run([sys.executable, "-m", "venv", str(venv)])
        py = venv / "bin" / "python"
        pip = venv / "bin" / "pip"
        print("installing wheel into clean venv ...")
        run([str(pip), "install", "-q", str(wheel)])

        # Prove import + CLI + catalog resolution + namespace package, all without the source tree on path.
        probe = (
            "import konfai;"
            "from konfai.utils.model_builder import build_model_from_yaml;"  # builder imports
            "import konfai.models.python.segmentation;"  # PEP 420 namespace subpackage resolves
            "from importlib.resources import files;"
            "assert files('konfai.models.yaml').joinpath('UNet.yml').is_file();"  # package-data shipped
            "print('konfai', getattr(konfai,'__version__','?'))"
        )
        run([str(py), "-c", probe], cwd=str(tmp))
        run([str(venv / "bin" / "konfai"), "--help"], cwd=str(tmp))
        print("clean-venv import + CLI + models.python namespace OK")

    print("\nRELEASE READY: the built wheel imports, resolves the model catalog, and excludes siblings.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (subprocess.CalledProcessError, AssertionError, StopIteration) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        print(f"\nRELEASE NOT READY: {detail}", file=sys.stderr)
        raise SystemExit(1) from exc
