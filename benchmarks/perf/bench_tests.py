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

"""The dev loop's wall and CPU-seconds with and without a thread pin.

    python benchmarks/perf/bench_tests.py [--force] [--quick] [--with-full] [--skip-unpinned]

Runs ``pytest -q -n auto --dist loadfile -m "not slow and not integration" tests/`` twice from the
repository root: pinned (``OMP_NUM_THREADS=1``, ``MKL_NUM_THREADS=1``) and unpinned (both unset,
today's default), and records the wall, the CPU-seconds of the whole worker tree (``RUSAGE_CHILDREN``,
which counts the waited-for descendants pytest-xdist leaves behind) and the failures. Nothing pins
pytest's threads today, so every worker runs torch with every core; the audit measured the pin at
8.4x on a quiet 24-core machine, and three tests of ``test_sweep_tiling.py`` that read the core count
fail under it. ``--with-full`` adds the full suite, pinned. The unpinned run takes minutes.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import resource
import subprocess
import sys
import time

from facts import fact
from harness import REPO, fingerprint, machine_gate, write_result

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def run_pytest(target: list[str], *, pinned: bool, marker: str | None) -> dict[str, object]:
    env = dict(os.environ)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        env.pop(key, None)
    if pinned:
        env["OMP_NUM_THREADS"] = env["MKL_NUM_THREADS"] = "1"
    # pyproject addopts carries -q already; a second one silences the summary line read below.
    argv = [sys.executable, "-m", "pytest", "-n", "auto", "--dist", "loadfile", "-p", "no:cacheprovider"]
    argv += ["-m", marker] if marker else []
    argv += target
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    start = time.perf_counter()
    completed = subprocess.run(argv, cwd=REPO, env=env, capture_output=True, text=True, check=False)
    wall = time.perf_counter() - start
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = (after.ru_utime - before.ru_utime) + (after.ru_stime - before.ru_stime)
    text = _ANSI.sub("", completed.stdout + completed.stderr)
    failed = sorted({line.split()[1] for line in text.splitlines() if line.startswith("FAILED ")})
    summary = None
    for line in reversed(text.splitlines()):
        if " in " in line and any(word in line for word in ("passed", "failed", "skipped")):
            summary = line.strip("= ")
            break
    if summary is None:
        raise RuntimeError(f"pytest printed no summary line, the counts cannot be read:\n{text[-1500:]}")
    counts = {"failed": 0, "passed": 0, "skipped": 0}
    for what in counts:
        match = re.search(rf"(\d+) {what}", summary)
        if match:
            counts[what] = int(match.group(1))
    # The FAILED list and the return code are the verdict; the summary line only carries the counts.
    counts["failed"] = max(counts["failed"], len(failed), 1 if completed.returncode != 0 else 0)
    return {
        "argv": argv,
        "pinned": pinned,
        "returncode": completed.returncode,
        "wall_s": round(wall, 2),
        "cpu_s": round(cpu, 1),
        "summary": summary,
        **counts,
        "failed_tests": failed,
        "tail": text[-1500:],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repeats", type=int, default=1, help="one run per state (the runs take minutes)")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--with-full", action="store_true", help="also run the full suite, pinned")
    parser.add_argument("--skip-unpinned", action="store_true")
    args = parser.parse_args()
    gate = machine_gate(force=args.force)
    fp = fingerprint()

    target = ["tests/unit/test_config.py"] if args.quick else ["tests/"]
    marker = "not slow and not integration"
    result: dict[str, object] = {"gate_warnings": gate.warnings, "target": target, "cpu_count": os.cpu_count()}
    metrics: dict[str, float] = {}

    pinned = run_pytest(target, pinned=True, marker=marker)
    result["pinned"] = pinned
    metrics["testfast_pinned_wall_s"] = pinned["wall_s"]  # type: ignore[assignment]
    metrics["testfast_pinned_cpu_s"] = pinned["cpu_s"]  # type: ignore[assignment]
    metrics["testfast_pinned_failed"] = float(pinned["failed"])  # type: ignore[arg-type]
    if not args.skip_unpinned:
        unpinned = run_pytest(target, pinned=False, marker=marker)
        result["unpinned"] = unpinned
        metrics["testfast_unpinned_wall_s"] = unpinned["wall_s"]  # type: ignore[assignment]
        metrics["testfast_unpinned_cpu_s"] = unpinned["cpu_s"]  # type: ignore[assignment]
        metrics["testfast_unpinned_failed"] = float(unpinned["failed"])  # type: ignore[arg-type]
        if pinned["wall_s"]:
            metrics["pin_speedup"] = round(unpinned["wall_s"] / pinned["wall_s"], 2)  # type: ignore[operator]
    if args.with_full and not args.quick:
        full = run_pytest(["tests/"], pinned=True, marker=None)
        result["full_pinned"] = full
        metrics["full_pinned_wall_s"] = full["wall_s"]  # type: ignore[assignment]
        metrics["full_pinned_cpu_s"] = full["cpu_s"]  # type: ignore[assignment]
        metrics["full_pinned_failed"] = float(full["failed"])  # type: ignore[arg-type]

    result["metrics"] = metrics
    result["facts"] = [fact(name, value, 0) for name, value in metrics.items() if name.endswith("_failed")]
    result["headline"] = (
        f"test-fast pinned {metrics['testfast_pinned_wall_s']} s ({metrics['testfast_pinned_cpu_s']} CPU-s, "
        f"{int(metrics['testfast_pinned_failed'])} failed)"
        + (
            f" | unpinned {metrics['testfast_unpinned_wall_s']} s ({metrics['testfast_unpinned_cpu_s']} CPU-s, "
            f"{int(metrics['testfast_unpinned_failed'])} failed) | {metrics.get('pin_speedup', 0)}x"
            if "testfast_unpinned_wall_s" in metrics
            else ""
        )
        + (f" | full pinned {metrics['full_pinned_wall_s']} s" if "full_pinned_wall_s" in metrics else "")
    )
    path = write_result("tests", result, fp=fp)
    print(json.dumps(metrics, indent=2))
    print(f"[perf] {result['headline']}")
    print(f"[perf] written {path}")


if __name__ == "__main__":
    main()
