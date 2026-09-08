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

"""What a process pays before its first byte: import time by module, CLI help, RSS after import.

    python benchmarks/perf/bench_startup.py [--repeats 3] [--force] [--quick]

Three measurements, each in a fresh interpreter so nothing is cached:

- ``python -X importtime -c "import konfai.trainer"``: the cumulative self time of the top-level
  modules that matter (konfai, torch, dask, ngff_zarr, zarr, SimpleITK), from the interpreter's own
  import profiler, so a lazy-import change shows as a moved line, not a guess.
- ``import konfai`` and ``import konfai.trainer`` wall and resident set, measured by the child itself.
- ``konfai --help`` wall through the installed entry point.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import time

from harness import fingerprint, konfai_executable, machine_gate, write_result

TOP_MODULES = ("konfai", "torch", "numpy", "dask", "ngff_zarr", "zarr", "SimpleITK", "h5py", "scipy", "ruamel")


def importtime(module: str) -> dict[str, float]:
    """Cumulative import time (ms) of each top-level package, from ``-X importtime``."""
    completed = subprocess.run(
        [sys.executable, "-X", "importtime", "-c", f"import {module}"], capture_output=True, text=True, check=False
    )
    totals: dict[str, float] = {}
    pattern = re.compile(r"import time:\s+(\d+) \|\s+(\d+) \|(\s*)(\S+)")
    for line in completed.stderr.splitlines():
        match = pattern.match(line)
        if not match:
            continue
        cumulative_us, depth, name = int(match.group(2)), len(match.group(3)), match.group(4)
        top = name.split(".")[0]
        # a top-level package line (depth 1 = no indentation) carries its whole subtree in `cumulative`
        if depth <= 1 and top in TOP_MODULES:
            totals[top] = totals.get(top, 0.0) + cumulative_us / 1e3
    return {k: round(v, 1) for k, v in totals.items()}


def import_wall_and_rss(module: str) -> tuple[float, float]:
    # VmHWM from /proc/self/status, not getrusage(RUSAGE_SELF).ru_maxrss: on Linux ru_maxrss is
    # inherited across exec, so a child of a parent that imported torch reports the parent's peak.
    code = (
        "import time, sys\n"
        "t0 = time.perf_counter()\n"
        f"import {module}\n"
        "wall = time.perf_counter() - t0\n"
        "hwm = 0.0\n"
        "for line in open('/proc/self/status'):\n"
        "    if line.startswith('VmHWM:'):\n"
        "        hwm = float(line.split()[1]) / 1024\n"
        "print(f'{wall:.4f} {hwm:.1f}')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False).stdout.split()
    return float(out[0]), float(out[1])


def cli_help_wall() -> float:
    start = time.perf_counter()
    subprocess.run([*konfai_executable(), "--help"], capture_output=True, text=True, check=False)
    return time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    repeats = 1 if args.quick else args.repeats
    gate = machine_gate(force=args.force)
    fp = fingerprint()

    result: dict[str, object] = {"repeats": repeats, "gate_warnings": gate.warnings}
    metrics: dict[str, float] = {}
    for module in ("konfai", "konfai.trainer", "konfai.utils.ome_zarr"):
        walls, rsss = [], []
        for _ in range(repeats + 1):  # one warmup for the OS page cache
            wall, rss = import_wall_and_rss(module)
            walls.append(wall)
            rsss.append(rss)
        walls, rsss = walls[1:], rsss[1:]
        key = module.replace(".", "_")
        metrics[f"import_{key}_s"] = round(statistics.median(walls), 3)
        metrics[f"import_{key}_rss_mb"] = round(statistics.median(rsss), 1)
        result[f"import_{key}_runs_s"] = [round(w, 3) for w in walls]
    result["importtime_konfai_trainer_ms_by_top_module"] = importtime("konfai.trainer")
    helps = [cli_help_wall() for _ in range(repeats + 1)][1:]
    metrics["cli_help_s"] = round(statistics.median(helps), 3)
    result["metrics"] = metrics
    result["headline"] = (
        f"import konfai {metrics['import_konfai_s']} s / konfai.trainer {metrics['import_konfai_trainer_s']} s "
        f"({metrics['import_konfai_trainer_rss_mb']:.0f} MB); konfai --help {metrics['cli_help_s']} s"
    )
    path = write_result("startup", result, fp=fp)
    print(json.dumps(result["metrics"], indent=2))
    print(f"[perf] {result['headline']}")
    print(f"[perf] written {path}")


if __name__ == "__main__":
    main()
