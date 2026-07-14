# SPDX-License-Identifier: Apache-2.0
"""Precondition check for the konfai-experiments skill.

Run this when the `konfai-mcp` MCP tools do not appear to be available, to tell
apart "server not installed" from "server installed but not wired into the MCP
client". It performs no side effects beyond importing packages and reading env.

    python .claude/skills/konfai-experiments/scripts/check_setup.py
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys


def _check_import(module: str) -> tuple[bool, str]:
    spec = importlib.util.find_spec(module)
    if spec is None:
        return False, f"{module}: NOT importable"
    return True, f"{module}: OK ({spec.origin})"


def main() -> int:
    print("== konfai-experiments setup check ==\n")

    ok_core, msg_core = _check_import("konfai")
    ok_mcp, msg_mcp = _check_import("konfai_mcp")
    print(msg_core)
    print(msg_mcp)

    entrypoint = shutil.which("konfai-mcp")
    print(f"konfai-mcp entrypoint: {entrypoint or 'NOT on PATH'}")

    root = os.environ.get("KONFAI_MCP_WORKSPACES_ROOT")
    print(f"KONFAI_MCP_WORKSPACES_ROOT: {root or '(unset -> defaults to ~/KonfAI_Workspaces)'}")
    tail = os.environ.get("KONFAI_MCP_LOG_TAIL_LINES")
    print(f"KONFAI_MCP_LOG_TAIL_LINES: {tail or '(unset -> server default)'}")

    print()
    if ok_core and ok_mcp and entrypoint:
        print("Ready. If the MCP tools still do not appear, the server is installed but")
        print("not wired into your MCP client — add a `konfai` MCP server entry pointing")
        print("at the entrypoint above (see the skill's references/resources-and-clients.md).")
        return 0

    print("Not ready. Install the packages on the konfai-mcp branch:")
    print("    pip install -e '.[dev]' && pip install -e ./konfai-mcp")
    return 1


if __name__ == "__main__":
    sys.exit(main())
