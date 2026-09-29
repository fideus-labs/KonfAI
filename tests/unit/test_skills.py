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

"""The agent skills under ``.claude/skills`` send their reader to files of this repository: those files exist."""

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SKILLS = REPO / ".claude" / "skills"
SKILL_FILES = sorted(SKILLS.rglob("*.md"))

# A repository file or directory in backticks: a top-level directory of the repo, then a path with no placeholder,
# ending in an extension or a slash (``konfai/__init__`` names a module, not a file).
_REPO_PATH = re.compile(
    r"`((?:docs|konfai|konfai-apps|konfai-mcp|konfai-studio|examples|apps|tests|benchmarks)/[\w./-]*(?:\.\w+|/))`"
)
# A relative Markdown link, resolved from the file that holds it.
_LINK = re.compile(r"\]\(((?!https?:|#)[^)\s#]+)")


@pytest.mark.parametrize("skill_file", SKILL_FILES, ids=lambda path: str(path.relative_to(SKILLS)))
def test_every_path_a_skill_cites_exists(skill_file: Path) -> None:
    text = skill_file.read_text(encoding="utf-8")
    cited = [REPO / path for path in _REPO_PATH.findall(text)]
    cited += [skill_file.parent / link for link in _LINK.findall(text)]
    missing = sorted({str(path.relative_to(REPO)) for path in cited if not path.exists()})
    assert not missing, f"{skill_file.relative_to(REPO)} cites paths that do not exist: {missing}"


def test_the_skills_are_found() -> None:
    assert len(SKILL_FILES) >= 3
