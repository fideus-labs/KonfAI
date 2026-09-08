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

"""Compare a ``run_all`` result to a baseline, metric by metric, and fail on a regression.

    python benchmarks/perf/compare.py BASELINE.json RUN.json [--time-tolerance 0.10] [--memory-tolerance 0.15]

Two results are comparable only when their fingerprints agree on the GPU, the power profile and the
thread pin, and neither carries gate warnings; the script says which of those differ before it reads
a number. Metrics are the leaf numbers under each bench's ``result.metrics`` (a flat mapping of
``name -> value`` every bench writes), classified by their name: ``*_s`` and ``*_ms`` are times (lower is
better), ``*_mib``/``*_gib``/``*_bytes`` are memory (lower is better), anything else is reported and not
judged. Unit tokens may precede a scenario suffix (``wall_s_whole``, ``peak_rss_gib_whole``).
Exit 0 certifies a comparable complete series, 1 reports a regression, and 2 rejects invalid or
incomplete comparison evidence (including quick/forced runs).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _fingerprint_problems(base: dict[str, Any], run: dict[str, Any]) -> list[str]:
    problems = []
    for key, getter in (
        ("GPU", lambda fp: (fp.get("gpu") or {}).get("name")),
        ("power profile", lambda fp: fp.get("power_profile")),
        ("OMP_NUM_THREADS", lambda fp: (fp.get("threads") or {}).get("OMP_NUM_THREADS")),
        ("working-tree dirty flag", lambda fp: (fp.get("git") or {}).get("dirty")),
    ):
        a, b = getter(base), getter(run)
        if a != b:
            problems.append(f"{key} differs: baseline {a!r}, run {b!r}")
    return problems


def comparable(base: dict[str, Any], run: dict[str, Any]) -> list[str]:
    problems = _fingerprint_problems(base.get("fingerprint") or {}, run.get("fingerprint") or {})
    for label, doc in (("baseline", base), ("run", run)):
        fp = doc.get("fingerprint")
        if not isinstance(fp, dict) or not all(key in fp for key in ("gpu", "power_profile", "threads")):
            problems.append(f"{label} has no complete machine fingerprint")
        if doc.get("gate_warnings"):
            problems.append(f"{label} was taken with gate warnings: {doc['gate_warnings']}")
        if doc.get("quick"):
            problems.append(f"{label} is a --quick smoke pass, not a number to compare")
        for bench, payload in (doc.get("benches") or {}).items():
            if not isinstance(payload, dict):
                continue
            details = payload.get("result") or {}
            if payload.get("gate_warnings") or details.get("gate_warnings"):
                problems.append(f"{label} benchmark {bench!r} carries gate warnings")
            if payload.get("quick") or details.get("quick"):
                problems.append(f"{label} benchmark {bench!r} is a quick smoke pass")
    for bench in set(base.get("benches") or {}) & set(run.get("benches") or {}):
        before, after = base["benches"][bench], run["benches"][bench]
        if isinstance(before, dict) and isinstance(after, dict):
            a, b = before.get("fingerprint"), after.get("fingerprint")
            if isinstance(a, dict) and isinstance(b, dict):
                problems.extend(f"benchmark {bench!r}: {problem}" for problem in _fingerprint_problems(a, b))
    return problems


def metrics_of(doc: dict[str, Any]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for bench, payload in (doc.get("benches") or {}).items():
        metrics = ((payload.get("result") or {}).get("metrics")) if isinstance(payload, dict) else None
        if isinstance(metrics, dict):
            out[bench] = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
    return out


def kind(name: str) -> str:
    lowered = name.lower()
    if re.search(r"(?:^|_)(?:s|ms|us)(?:_|$)", lowered):
        return "time"
    if re.search(r"(?:^|_)(?:mib|gib|bytes|mb|gb)(?:_|$)", lowered):
        return "memory"
    return "other"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("run", type=Path)
    parser.add_argument(
        "--time-tolerance", type=float, default=0.10, help="relative slowdown that counts as a regression"
    )
    parser.add_argument(
        "--memory-tolerance", type=float, default=0.15, help="relative growth that counts as a regression"
    )
    args = parser.parse_args()
    if any(not math.isfinite(value) or value < 0 for value in (args.time_tolerance, args.memory_tolerance)):
        parser.error("tolerances must be finite non-negative numbers")

    base, run = load(args.baseline), load(args.run)
    problems = comparable(base, run)
    base_metrics, run_metrics = metrics_of(base), metrics_of(run)
    # A failed/missing scenario cannot certify a release, even when no comparable metric remains.
    for label, doc, metrics in (("baseline", base, base_metrics), ("run", run, run_metrics)):
        benches = doc.get("benches") or {}
        if not benches:
            problems.append(f"{label} has no benchmarks")
        for bench, payload in benches.items():
            if not isinstance(payload, dict) or payload.get("status") == "failed":
                problems.append(f"{label} benchmark {bench!r} failed")
            if not metrics.get(bench):
                problems.append(f"{label} benchmark {bench!r} has no numeric metrics")
    rows = ["| bench | metric | baseline | run | ratio | verdict |", "|---|---|---:|---:|---:|---|"]
    regressions = 0
    judged = 0
    for bench in sorted(set(base_metrics) | set(run_metrics)):
        names = sorted(set(base_metrics.get(bench, {})) | set(run_metrics.get(bench, {})))
        for name in names:
            a = base_metrics.get(bench, {}).get(name)
            b = run_metrics.get(bench, {}).get(name)
            if a is None or b is None:
                problems.append(f"missing metric {bench}.{name}")
                rows.append(
                    f"| {bench} | {name} | {a if a is not None else ''} | {b if b is not None else ''} | | missing on one side |"
                )
                continue
            ratio = b / a if a else float("inf")
            verdict = ""
            what = kind(name)
            if not math.isfinite(a) or not math.isfinite(b):
                problems.append(f"non-finite metric {bench}.{name}")
                verdict = "INVALID"
            elif what in ("time", "memory") and (a < 0 or b < 0):
                problems.append(f"negative {what} metric {bench}.{name}")
                verdict = "INVALID"
            elif what in ("time", "memory"):
                judged += 1
                ratio = b / a if a else (1.0 if b == 0 else float("inf"))
            if verdict == "INVALID":
                pass
            elif what == "time" and ratio > 1 + args.time_tolerance:
                verdict, regressions = "REGRESSION", regressions + 1
            elif what == "memory" and ratio > 1 + args.memory_tolerance:
                verdict, regressions = "REGRESSION", regressions + 1
            elif what in ("time", "memory") and ratio < 1 - args.time_tolerance:
                verdict = "improved"
            rows.append(f"| {bench} | {name} | {a:g} | {b:g} | {ratio:.2f} | {verdict} |")
    print("\n".join(rows))
    if not judged:
        problems.append("no time or memory metric was compared")
    for problem in problems:
        print(f"[compare] INVALID: {problem}")
    print(f"[compare] {regressions} regression(s); comparability problems: {len(problems)}")
    sys.exit(2 if problems else (1 if regressions else 0))


if __name__ == "__main__":
    main()
