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

"""The performance gate: run the series, compare it to this host's committed baseline, fail on a
regression.

    pixi run perf-check                       # run_all, then compare against baselines/<host>.json
    python benchmarks/perf/perf_check.py --baseline results/<host>/<stamp>-all.json

A baseline is a ``run_all`` result taken on a quiet machine (the gate refuses otherwise) and copied
by hand to ``baselines/<host>.json``; ``compare.py`` states what is comparable (the commit's dirty
flag, the GPU, the power profile, the thread pin) and fails past 10 % of time or 15 % of memory.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import machine_class

PERF_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--baseline", type=Path, default=None, help="a run_all result; default baselines/<machine class>.json"
    )
    parser.add_argument("--force", action="store_true", help="time on a busy machine (a diagnostic, never a verdict)")
    parser.add_argument(
        "--quick", action="store_true", help="smoke pass; comparison will reject it as release evidence"
    )
    args, extra = parser.parse_known_args()
    baseline = args.baseline or PERF_DIR / "baselines" / f"{machine_class()}.json"
    if not baseline.is_file():
        raise SystemExit(
            f"[perf] no baseline for this machine class: {baseline} (take one with run_all.py on a quiet machine)"
        )
    argv = [sys.executable, str(PERF_DIR / "run_all.py"), *extra]
    if args.force:
        argv.append("--force")
    if args.quick:
        argv.append("--quick")
    completed = subprocess.run(argv, cwd=PERF_DIR, capture_output=True, text=True, check=False)
    sys.stdout.write(completed.stdout)
    sys.stderr.write(completed.stderr)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    written = [
        line for line in completed.stdout.splitlines() if line.startswith("[perf] written ") and "-all.json" in line
    ]
    if not written:
        raise SystemExit("[perf] run_all wrote no series result")
    run = Path(written[-1].split("[perf] written ", 1)[1].split(" and ", 1)[0].strip())
    # The facts first (they hold on any machine), then the times against this machine class's baseline.
    facts = subprocess.run([sys.executable, str(PERF_DIR / "facts.py"), str(run)], cwd=PERF_DIR, check=False)
    compare = subprocess.run(
        [sys.executable, str(PERF_DIR / "compare.py"), str(baseline), str(run)], cwd=PERF_DIR, check=False
    )
    # 1 is a violation or a regression, 2 evidence the gate cannot use; a signaled checker is the latter.
    status = 0
    for code in (facts.returncode, compare.returncode):
        if code != 0:
            status = max(status, code if code in (1, 2) else 2)
    raise SystemExit(status)


if __name__ == "__main__":
    main()
