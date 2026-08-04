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

"""Every YAML example on the transform config page, planned and run.

A config page is a promise that what it shows works. Prose drifts from code silently and a reader
who copies a broken example has no way to tell whose fault it is, so the examples are extracted from
the page itself and executed — not transcribed here, where they could drift in turn.

Planning is not enough: a plan is computed on the launcher and the run then spawns, so a chain can
plan STREAM and still fail before its first byte. Each example is therefore run to completion.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from ruamel.yaml import YAML

pytest.importorskip("SimpleITK")
import SimpleITK as sitk

DOC = Path(__file__).resolve().parents[2] / "docs" / "source" / "config_guide" / "transform.md"

# A `transforms:` fragment is a chain without a workflow around it; this is the smallest root that
# makes one runnable, so the page can keep showing chains without repeating the boilerplate.
FRAGMENT_ROOT = """Transformer:
  name: DOC_CHECK
  Dataset:
    dataset_filenames:
      - ./Raw:mha
    groups_src:
      CT:
        groups_dest:
          CT_out:
{transforms}
"""

# Builds the workflow the same way the CLI does, then plans it. Run in a child because building a
# workflow writes the process environment and rewrites the config in place.
PLAN_SCRIPT = """
import os, sys
from pathlib import Path
os.chdir(sys.argv[1])
from konfai.transformer import build_transform
plan = build_transform(
    transform_file=Path("Transform.yml").resolve(),
    transforms_dir=Path("Transforms").resolve(),
).compute_plan(1, False)
print(plan.report())
"""


def _yaml_blocks(text: str) -> list[tuple[int, str]]:
    return [
        (text[: match.start()].count("\n") + 1, match.group(1))
        for match in re.finditer(r"^```yaml\n(.*?)^```", text, re.S | re.M)
    ]


def _python_modules(text: str) -> dict[str, str]:
    """Python blocks whose first line names a file (``# BoxFilter.py -- ...``), as {filename: source}.

    The page tells the reader to drop that file next to the config, so the example that imports it
    is only runnable if the check does the same.
    """
    modules = {}
    for match in re.finditer(r"^```python\n(.*?)^```", text, re.S | re.M):
        header = re.match(r"#\s*([A-Za-z_]\w*\.py)\b", match.group(1).strip())
        if header:
            modules[header.group(1)] = match.group(1)
    return modules


def _as_config(body: str) -> str | None:
    """One block as a runnable config, or ``None`` when it is neither a root nor a chain fragment."""
    tree = YAML().load(body)
    if not isinstance(tree, dict):
        return None
    if "Transformer" in tree:
        return body
    if "transforms" in tree:
        indented = "\n".join(" " * 12 + row if row.strip() else row for row in body.rstrip().split("\n"))
        return FRAGMENT_ROOT.format(transforms=indented)
    return None


def _source_groups(config: str) -> list[str]:
    dataset = (YAML().load(config).get("Transformer") or {}).get("Dataset") or {}
    return [str(key) for key in (dataset.get("groups_src") or {})] or ["CT"]


def _write_dataset(root: Path, groups: list[str], cases: int = 2, shape=(8, 12, 12)) -> None:
    for index in range(cases):
        case = root / f"case_{index}"
        case.mkdir(parents=True, exist_ok=True)
        for group in groups:
            image = sitk.GetImageFromArray(
                np.linspace(0, 500, int(np.prod(shape)), dtype=np.float32).reshape(shape) + index
            )
            image.SetSpacing((0.5, 0.5, 2.0))
            image.SetOrigin((1.0, 2.0, 3.0))
            sitk.WriteImage(image, str(case / f"{group}.mha"))


def _doc_examples() -> list[tuple[int, str]]:
    if not DOC.is_file():
        return []
    text = DOC.read_text(encoding="utf-8")
    return [(line, config) for line, body in _yaml_blocks(text) if (config := _as_config(body)) is not None]


@pytest.mark.integration
@pytest.mark.parametrize("line,config", _doc_examples(), ids=lambda value: value if isinstance(value, int) else "")
def test_a_documented_example_plans_and_runs(line: int, config: str, tmp_path: Path) -> None:
    workdir = tmp_path / f"block_{line}"
    workdir.mkdir()
    _write_dataset(workdir / "Raw", _source_groups(config))
    (workdir / "Transform.yml").write_text(config, encoding="utf-8")
    for filename, source in _python_modules(DOC.read_text(encoding="utf-8")).items():
        (workdir / filename).write_text(source, encoding="utf-8")
    environment = {"KONFAI_CLI": "False", "PYTHONPATH": str(workdir)}

    script = workdir / "_plan.py"
    script.write_text(PLAN_SCRIPT, encoding="utf-8")
    planned = subprocess.run(
        [sys.executable, str(script), str(workdir)],
        capture_output=True,
        text=True,
        env={**os.environ, **environment},
        timeout=600,
    )
    assert planned.returncode == 0, f"transform.md:{line} does not plan:\n{planned.stdout}{planned.stderr}"

    # The plan is computed on the launcher and the run then spawns: planning green proves nothing
    # about what crosses that boundary.
    executed = subprocess.run(
        [_konfai(), "TRANSFORM", "--config", "Transform.yml", "--transforms-dir", "Transforms"],
        capture_output=True,
        text=True,
        env={**os.environ, **environment},
        cwd=str(workdir),
        timeout=900,
    )
    assert executed.returncode == 0, f"transform.md:{line} plans but does not run:\n{executed.stdout}{executed.stderr}"


def _konfai() -> str:
    konfai = Path(sys.executable).with_name("konfai")
    if not konfai.exists():
        pytest.skip("the konfai console script is not installed in this environment")
    return str(konfai)
