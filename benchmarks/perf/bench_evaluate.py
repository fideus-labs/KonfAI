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

"""EVALUATION on the shipped Segmentation example: what it costs outside its loop.

    python benchmarks/perf/bench_evaluate.py [--repeats 3] [--force] [--quick]

The CLI runs on a scratch copy of ``examples/Segmentation`` that keeps its shipped
``Predictions/SEG_BASELINE`` (the input of ``Evaluation.yml``). The ``[KonfAI] startup`` clock is
parsed; the loop figure is the wall minus that startup, a difference and not a clock. The metric
JSON the run writes is opened and its numbers checked finite before a time is reported.

With a GPU (and without ``--cpu``) the same runs repeat with ``--gpu 0``: the metrics then move to the
device, and their JSON must match the CPU one number for number before the GPU time counts.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import tempfile
from pathlib import Path

from harness import CliRun, copy_example, fingerprint, konfai_executable, machine_gate, run_cli, write_result


def evaluate_once(copy: Path, out_dir: Path, gpu: bool = False) -> CliRun:
    argv = [*konfai_executable(), "EVALUATION", "-c", "Evaluation.yml", "-y", "--evaluations-dir", str(out_dir)]
    if gpu:
        argv += ["--gpu", "0"]
    return run_cli(argv, cwd=copy, log_path=out_dir.with_suffix(".log"))


def metric_files(out_dir: Path) -> list[Path]:
    return sorted(out_dir.rglob("*.json"))


def finite_numbers(obj: object) -> tuple[int, int]:
    """(numbers seen, non-finite numbers) anywhere in a JSON tree."""
    seen = bad = 0
    if isinstance(obj, dict):
        for value in obj.values():
            s, b = finite_numbers(value)
            seen, bad = seen + s, bad + b
    elif isinstance(obj, list):
        for value in obj:
            s, b = finite_numbers(value)
            seen, bad = seen + s, bad + b
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        seen = 1
        bad = 0 if math.isfinite(obj) else 1
    return seen, bad


def flat_numbers(obj: object, prefix: str = "") -> dict[str, float]:
    """Every number of a JSON tree, keyed by its path."""
    if isinstance(obj, dict):
        return {k: v for key, value in obj.items() for k, v in flat_numbers(value, f"{prefix}/{key}").items()}
    if isinstance(obj, list):
        return {k: v for i, value in enumerate(obj) for k, v in flat_numbers(value, f"{prefix}/{i}").items()}
    if isinstance(obj, (int, float)) and not isinstance(obj, bool):
        return {prefix: float(obj)}
    return {}


def timed_runs(copy: Path, out_dir: Path, repeats: int, quick: bool, gpu: bool, gate, fp) -> list[CliRun]:
    """The route's runs, the first one a warmup outside a quick pass; a failed run is written and raised."""
    runs: list[CliRun] = []
    for _ in range(repeats + (0 if quick else 1)):
        run = evaluate_once(copy, out_dir, gpu)
        if run.returncode != 0:
            write_result("evaluate", {"failure": run.output[-3000:], "gate_warnings": gate.warnings}, fp=fp)
            raise SystemExit(f"[perf] EVALUATION ({'gpu' if gpu else 'cpu'}) failed with rc {run.returncode}")
        runs.append(run)
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--cpu", action="store_true", help="the CPU route only")
    args = parser.parse_args()
    repeats = 1 if args.quick else args.repeats
    gate = machine_gate(force=args.force)
    fp = fingerprint()

    scratch = Path(tempfile.mkdtemp(prefix="konfai_perf_evaluate_"))
    copy = copy_example("Segmentation", scratch, exclude=("Evaluations", "Statistics", "__pycache__"))
    out_dir = scratch / "Evaluations"
    runs = timed_runs(copy, out_dir, repeats, args.quick, False, gate, fp)
    timed = runs[1:] if len(runs) > 1 else runs

    files = metric_files(out_dir)
    seen = bad = 0
    sample: dict[str, object] = {}
    for file in files:
        try:
            payload = json.loads(file.read_text())
        except json.JSONDecodeError:
            bad += 1
            continue
        s, b = finite_numbers(payload)
        seen, bad = seen + s, bad + b
        if not sample and isinstance(payload, dict):
            sample = {"file": file.name, "keys": list(payload)[:8]}

    metrics = {
        "wall_s": round(statistics.median(r.wall_s for r in timed), 3),
        "startup_s": round(statistics.median(r.clocks.get("startup", {}).get("wall", 0.0) for r in timed), 3),
        "peak_rss_gib": round(max(r.peak_rss_gib for r in timed), 3),
    }
    metrics["loop_s"] = round(metrics["wall_s"] - metrics["startup_s"], 3)

    import torch

    gpu_route: dict[str, object] = {}
    if torch.cuda.is_available() and not args.cpu:
        gpu_dir = scratch / "Evaluations_gpu"
        gpu_runs = timed_runs(copy, gpu_dir, repeats, args.quick, True, gate, fp)
        gpu_timed = gpu_runs[1:] if len(gpu_runs) > 1 else gpu_runs
        # Keyed by the path under each directory: two files of one name would take one key.
        cpu_numbers = {
            k: v
            for f in files
            for k, v in flat_numbers(json.loads(f.read_text()), f.relative_to(out_dir).as_posix()).items()
        }
        gpu_numbers = {
            k: v
            for f in metric_files(gpu_dir)
            for k, v in flat_numbers(json.loads(f.read_text()), f.relative_to(gpu_dir).as_posix()).items()
        }
        # A metric that is not a number on either device is a difference, not a value to skip.
        finite = all(math.isfinite(v) for numbers in (cpu_numbers, gpu_numbers) for v in numbers.values())
        shared = [k for k in cpu_numbers if k in gpu_numbers]
        gap = max((abs(cpu_numbers[k] - gpu_numbers[k]) for k in shared if finite), default=0.0)
        if cpu_numbers.keys() != gpu_numbers.keys() or not finite or gap > 1e-6:
            write_result(
                "evaluate",
                {"failure": f"GPU metrics differ from CPU (max gap {gap})", "gate_warnings": gate.warnings},
                fp=fp,
            )
            raise SystemExit(f"[perf] EVALUATION on GPU does not match the CPU metrics: max gap {gap}")
        metrics["wall_s_gpu"] = round(statistics.median(r.wall_s for r in gpu_timed), 3)
        metrics["peak_rss_gib_gpu"] = round(max(r.peak_rss_gib for r in gpu_timed), 3)
        gpu_route = {"runs": [{"wall_s": r.wall_s, "clocks": r.clocks} for r in gpu_runs], "max_metric_gap": gap}
    result = {
        "gate_warnings": gate.warnings,
        "repeats": repeats,
        "runs": [{"wall_s": r.wall_s, "clocks": r.clocks, "peak_rss_gib": r.peak_rss_gib} for r in runs],
        "metric_files": [str(f.relative_to(out_dir)) for f in files],
        "numbers_seen": seen,
        "non_finite_numbers": bad,
        "sample": sample,
        "loop_note": "loop_s is wall_s minus the startup clock: a difference, not a clock",
        "clock_note": "the [KonfAI] startup and loop clocks print only above one second: a 0.0 here means under a second, not zero",
        "gpu": gpu_route,
        "metrics": metrics,
    }
    result["headline"] = (
        f"evaluation {metrics['wall_s']} s wall, startup {metrics['startup_s']} s, loop about {metrics['loop_s']} s, "
        f"RSS {metrics['peak_rss_gib']} GiB; {len(files)} metric files, {seen} numbers, {bad} non-finite"
        + (f" | --gpu 0 {metrics['wall_s_gpu']} s, same metrics" if "wall_s_gpu" in metrics else "")
    )
    path = write_result("evaluate", result, fp=fp)
    print(json.dumps(metrics, indent=2))
    print(f"[perf] {result['headline']}")
    print(f"[perf] written {path}")
    if not files or bad:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
