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

"""Run every bench of this folder one after the other and fold their results into one file.

    python benchmarks/perf/run_all.py [--quick] [--force] [--only startup,train_step,...] [--skip tests]

The gate runs once here and once per bench (a bench can be run alone). The benches run in
subprocesses, sequentially, so no two of them share the GPU or the cores; each writes its own JSON
and this script gathers the newest one per bench into ``results/<host>/<stamp>-<sha>-all.json`` with
a markdown table beside it.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from harness import PERF_DIR, fingerprint, machine_gate, results_dir

BENCHES = ("startup", "train_step", "train_epoch", "predict", "evaluate", "transform", "tests")


def newest_result(bench: str, since: float) -> Path | None:
    candidates = [
        p
        for p in results_dir().glob(f"*-{bench}*.json")
        if p.stat().st_mtime >= since and not p.name.endswith("-all.json")
    ]
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--quick", action="store_true", help="smoke pass: small inputs, one repeat, not a number to publish"
    )
    parser.add_argument("--force", action="store_true", help="time on a busy machine; warnings are recorded")
    parser.add_argument("--only", default="", help="comma-separated subset of: " + ", ".join(BENCHES))
    parser.add_argument("--skip", default="", help="comma-separated benches to skip (tests is the slow one)")
    parser.add_argument("--cpu", action="store_true", help="the prediction bench without a GPU")
    parser.add_argument(
        "--synthetic", action="store_true", help="the prediction bench on synthetic cases, no example data"
    )
    args = parser.parse_args()

    gate = machine_gate(force=args.force)
    os.environ["KONFAI_PERF_GATED"] = "1"  # the series was gated here; each bench still records its load
    fp = fingerprint()
    selected = [b for b in BENCHES if (not args.only or b in args.only.split(",")) and b not in args.skip.split(",")]
    unknown = (set(filter(None, args.only.split(","))) | set(filter(None, args.skip.split(",")))) - set(BENCHES)
    if unknown or not selected:
        parser.error(f"unknown benchmarks: {sorted(unknown)}" if unknown else "no benchmarks selected")
    print(
        f"[perf] {fp['git']['describe']}{' (dirty)' if fp['git']['dirty'] else ''} on {fp['host']}, "
        f"{fp['gpu']['name'] or 'no GPU'}, profile {fp['power_profile']}, load {fp['load_avg']}"
    )

    combined: dict[str, object] = {
        "fingerprint": fp,
        "quick": args.quick,
        "gate_warnings": gate.warnings,
        "benches": {},
    }
    rows = ["| bench | headline | file |", "|---|---|---|"]
    failed = False
    for bench in selected:
        script = PERF_DIR / f"bench_{bench}.py"
        if not script.exists():
            failed = True
            combined["benches"][bench] = {"status": "failed", "error": "missing script"}  # type: ignore[index]
            rows.append(f"| {bench} | (no script) | |")
            continue
        argv = [sys.executable, str(script)]
        if args.quick:
            argv.append("--quick")
        if args.force:
            argv.append("--force")
        if bench == "predict":
            argv += [flag for flag, on in (("--cpu", args.cpu), ("--synthetic", args.synthetic)) if on]
        started = time.time()
        print(f"[perf] === {bench}: {' '.join(argv[1:])}", flush=True)
        completed = subprocess.run(argv, cwd=PERF_DIR, check=False)
        result_path = newest_result(bench, started)
        if completed.returncode != 0 or result_path is None:
            failed = True
            rows.append(f"| {bench} | FAILED (rc {completed.returncode}) | |")
            combined["benches"][bench] = {"status": "failed", "returncode": completed.returncode}  # type: ignore[index]
            continue
        payload = json.loads(result_path.read_text())
        combined["benches"][bench] = payload  # type: ignore[index]
        headline = payload.get("result", {}).get("headline", "")
        rows.append(f"| {bench} | {headline} | {result_path.name} |")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = results_dir() / f"{stamp}-{fp['git']['sha'][:8]}-all.json"
    out.write_text(json.dumps(combined, indent=2, default=str) + "\n")
    out.with_suffix(".md").write_text(
        f"# benchmarks/perf on {fp['host']} at {fp['date']}\n\n"
        f"commit {fp['git']['describe']}{' (dirty)' if fp['git']['dirty'] else ''}, {fp['gpu']['name'] or 'no GPU'}, "
        f"profile {fp['power_profile']}, load {fp['load_avg']}, OMP_NUM_THREADS={fp['threads']['OMP_NUM_THREADS']}"
        + (", QUICK (smoke, not publishable)" if args.quick else "")
        + "\n\n"
        + "\n".join(rows)
        + "\n"
    )
    print("\n".join(rows))
    print(f"[perf] written {out} and {out.with_suffix('.md')}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
