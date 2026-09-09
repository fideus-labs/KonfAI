# SPDX-License-Identifier: Apache-2.0
"""The facts a run asserts, checked on any machine.

A time compares to a baseline taken on the same machine class; a fact has its expected value in the
run itself: the streamed route wrote the voxels of the whole-volume route, the transform equals the
plain loop to float32 rounding, the plan's held peak stayed within its budget, no test failed. Each bench lists
its facts under ``result["facts"]`` as ``{"name", "value", "expect", "op"}`` with ``op`` ``==`` or
``<=``.

    python benchmarks/perf/facts.py RUN.json          # a run_all result, or one bench's result

Exit 0 when every fact holds, 1 on a violation, 2 when the run carries no fact.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def fact(name: str, value: float | bool, expect: float | bool, op: str = "==") -> dict[str, Any]:
    return {"name": name, "value": float(value), "expect": float(expect), "op": op}


def holds(entry: dict[str, Any]) -> bool:
    value, expect = float(entry["value"]), float(entry["expect"])
    if value != value or expect != expect:  # a NaN holds nothing
        return False
    op = entry.get("op", "==")
    if op not in ("==", "<="):
        return False
    return value == expect if op == "==" else value <= expect


def facts_of(doc: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """``(bench, fact)`` pairs of a run_all result or of one bench's result."""
    benches = doc.get("benches")
    if isinstance(benches, dict):
        payloads = [(name, payload) for name, payload in benches.items() if isinstance(payload, dict)]
    else:
        payloads = [(str(doc.get("bench", "bench")), doc)]
    out: list[tuple[str, dict[str, Any]]] = []
    for name, payload in payloads:
        if payload.get("status") == "failed":
            out.append((name, fact("completed", 0, 1)))
        for entry in (payload.get("result") or {}).get("facts") or []:
            out.append((name, entry))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    doc = json.loads(args.run.read_text())
    pairs = facts_of(doc)
    rows = ["| bench | fact | value | expected | verdict |", "|---|---|---:|---:|---|"]
    violations = 0
    for bench, entry in pairs:
        ok = holds(entry)
        violations += 0 if ok else 1
        rows.append(
            f"| {bench} | {entry['name']} | {entry['value']:g} | {entry.get('op', '==')} {entry['expect']:g} | "
            f"{'holds' if ok else 'VIOLATED'} |"
        )
    print("\n".join(rows))
    if not pairs:
        print("[facts] INVALID: the run carries no fact")
        sys.exit(2)
    print(f"[facts] {violations} violation(s) over {len(pairs)} fact(s)")
    sys.exit(1 if violations else 0)


if __name__ == "__main__":
    main()
