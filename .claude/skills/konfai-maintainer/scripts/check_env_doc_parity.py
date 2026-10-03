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

"""Flag `KONFAI_*` environment variables used in code but absent from the env-var docs (and vice versa).

Deterministic drift check for maintainers. Run from anywhere:

    python .claude/skills/konfai-maintainer/scripts/check_env_doc_parity.py

Exit code 0 = in sync, 1 = drift found. The 2026-07-19 audit found the docs both missed several knobs
(the streaming kill-switches, the apps trust switch) and listed one that no longer exists.
"""

from __future__ import annotations

import re
import sys
import tokenize
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
DOCS_ROOT = REPO / "docs" / "source"  # a var documented ANYWHERE in the docs counts (env vars split across pages)
CODE_DIRS = [
    REPO / "konfai",
    REPO / "konfai-apps" / "konfai_apps",
    REPO / "konfai-mcp" / "konfai_mcp",
    REPO / "konfai-studio" / "konfai_studio",
]

VAR = re.compile(r"KONFAI_[A-Z_]+")
# Lowercase KONFAI_config_file is a real runtime var set by the wrappers; include it explicitly.
KNOWN_LOWER = {"KONFAI_config_file"}
# An f-string's literal text is its own token type from Python 3.12.
_STRING_TOKENS = {tokenize.STRING, getattr(tokenize, "FSTRING_MIDDLE", tokenize.STRING)}


def vars_in(paths: list[Path]) -> set[str]:
    """The names in the code's string literals: a variable is read by its name as a string, while a module
    constant such as ``KONFAI_VERSION`` or ``_KONFAI_DEPS`` is an identifier."""
    found: set[str] = set()
    for root in paths:
        if not root.exists():
            continue
        for py in root.rglob("*.py"):
            if "/build/" in py.as_posix():
                continue
            with tokenize.open(py) as source:
                for token in tokenize.generate_tokens(source.readline):
                    if token.type in _STRING_TOKENS:
                        found |= set(VAR.findall(token.string))
    return found


def documented_vars() -> set[str]:
    found: set[str] = set()
    lower: set[str] = set()
    for md in DOCS_ROOT.rglob("*.md"):
        if "/build/" in md.as_posix():
            continue
        text = md.read_text(encoding="utf-8", errors="ignore")
        found |= set(VAR.findall(text))
        lower |= set(text.split()) & KNOWN_LOWER
    # A bare-prefix token (ends in "_", e.g. a family mention like "KONFAI_MCP_") names no real var; drop it.
    return {v for v in found if not v.endswith("_")} | lower


def main() -> int:
    if not DOCS_ROOT.exists():
        print(f"docs root not found: {DOCS_ROOT}", file=sys.stderr)
        return 1
    documented = documented_vars()
    used = vars_in(CODE_DIRS)

    missing = sorted(used - documented)
    stale = sorted(v for v in documented - used - KNOWN_LOWER if v.startswith("KONFAI_"))

    if missing:
        print("Used in code but MISSING from the docs (docs/source/**.md):")
        for v in missing:
            print(f"  - {v}")
    if stale:
        print("Documented but NOT found in code (possibly stale):")
        for v in stale:
            print(f"  - {v}")
    if not missing and not stale:
        print(f"env-var docs in sync ({len(used)} vars).")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
