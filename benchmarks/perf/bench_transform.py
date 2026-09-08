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

"""TRANSFORM against a plain loop, and the region height against the budget.

    python benchmarks/perf/bench_transform.py [--gib 2] [--budgets 1,8] [--sweep] [--repeats 3] [--force] [--quick]

Reuses ``benchmarks/bench_streaming.py``'s synthetic h5 volume (float32, chunked in 64-row slabs)
and its chain ``Normalize(-1, 1) -> Write``. Three measurements:

- KonfAI at each declared budget, **each run in a fresh process**: a second ``konfai.transform`` in the
  same process reuses an in-process statistics cache and skips the Min/Max scan (2 GiB: 14.8 s cold
  against 3.4 s warm on the audit's machine), so only a fresh process measures what a user pays. Wall,
  whole-tree peak RSS, and the interpreter floor before the call, all reported by the child.
- A 25-line numpy + h5py loop over the same store in 64-row slabs (one min/max pass, one rescale
  pass), the fairest floor the framework has been measured against.
- One KonfAI run under ``cProfile``, with the share of the statistics scan, the sweep and the h5 layer
  in host time.

The KonfAI output is compared to the loop's slab by slab (same rescale formula as ``Normalize``) and
the geometry attributes checked; no time is reported when they disagree. ``--sweep`` runs the budgets
64 MiB to 8 GiB twice each, the curve the audit used to bound the planner's growth step.
"""

from __future__ import annotations

import argparse
import cProfile
import json
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from harness import PERF_DIR, PeakSampler, cprofile_summary, fingerprint, machine_gate, write_result

sys.path.insert(0, str(PERF_DIR.parent))
from bench_streaming import synthesize

SLAB = 64


def naive_normalize(store: Path, out: Path, *, min_value: float = -1.0, max_value: float = 1.0) -> tuple[float, int]:
    """Two passes in 64-row slabs with numpy + h5py; returns (wall_s, peak tree RSS bytes)."""
    import h5py

    sampler = PeakSampler()
    sampler.start()
    start = time.perf_counter()
    with h5py.File(store, "r") as src, h5py.File(out, "w") as dst:
        data = src["CT/CASE_000"]
        lo, hi = np.inf, -np.inf
        for z in range(0, data.shape[1], SLAB):
            block = data[0, z : z + SLAB]
            lo, hi = min(lo, float(block.min())), max(hi, float(block.max()))
        target = dst.create_dataset("CT/CASE_000", shape=data.shape, dtype=np.float32, chunks=data.chunks)
        for key, value in data.attrs.items():
            target.attrs[key] = value
        scale = (max_value - min_value) / (hi - lo)
        for z in range(0, data.shape[1], SLAB):
            target[0, z : z + SLAB] = (data[0, z : z + SLAB] - lo) * scale + min_value
    wall = time.perf_counter() - start
    return wall, sampler.stop()


def konfai_normalize(store: Path, out_dir: Path, budget_gib: float, transforms_dir: Path) -> tuple[float, int, int]:
    """One konfai.transform run; returns (wall_s, peak tree RSS bytes, RSS before the call)."""
    import konfai
    from konfai.data.transform import Normalize, Write

    sampler = PeakSampler()
    baseline = sampler.baseline
    sampler.start()
    start = time.perf_counter()
    konfai.transform(
        "BENCH",
        f"{store}:h5",
        {"CT": {"CT": [Normalize(min_value=-1, max_value=1), Write(dataset=f"{out_dir}:h5")]}},
        memory_budget=f"{budget_gib}gib",
        transforms_dir=transforms_dir,
        quiet=True,
    )
    wall = time.perf_counter() - start
    return wall, sampler.stop(), baseline


def _worker_konfai(store: Path, out_dir: Path, budget_gib: float, transforms_dir: Path, prof: Path | None) -> None:
    """One transform in this fresh process; prints one JSON line (wall, peak, floor)."""
    profile = cProfile.Profile() if prof is not None else None
    if profile is not None:
        profile.enable()
    wall, peak, floor = konfai_normalize(store, out_dir, budget_gib, transforms_dir)
    if profile is not None:
        profile.disable()
        profile.dump_stats(str(prof))
    print(json.dumps({"wall_s": wall, "peak_bytes": peak, "floor_bytes": floor}))


def _worker_naive(store: Path, out: Path) -> None:
    wall, peak = naive_normalize(store, out)
    print(json.dumps({"wall_s": wall, "peak_bytes": peak}))


def in_fresh_process(argv: list[str]) -> dict[str, float]:
    """Run this script's worker mode in a new interpreter and parse its JSON line."""
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), *argv], capture_output=True, text=True, check=False
    )
    lines = [line for line in completed.stdout.splitlines() if line.startswith("{")]
    if completed.returncode != 0 or not lines:
        raise SystemExit(
            f"[perf] worker failed (rc {completed.returncode}):\n{completed.stdout[-1500:]}\n{completed.stderr[-3000:]}"
        )
    return json.loads(lines[-1])


def plan_lines(transforms_dir: Path) -> list[str]:
    """The plan's verdict lines (STREAM / LOAD / WHOLE-VOLUME, budget, held peak) from the run's log."""
    out: list[str] = []
    for log in sorted(transforms_dir.rglob("log_*.txt")):
        for line in log.read_text(errors="ignore").splitlines():
            if any(key in line for key in ("STREAM", "LOAD", "WHOLE-VOLUME", "held ", "per-rank budget")):
                out.append(line.strip()[:200])
    return out[:8]


def find_output(out_dir: Path) -> Path:
    candidates = sorted(out_dir.rglob("*.h5")) if out_dir.is_dir() else ([out_dir] if out_dir.exists() else [])
    if not candidates:
        candidates = sorted(out_dir.parent.glob(out_dir.name + "*"))
    if not candidates:
        raise SystemExit(f"[perf] no h5 output under {out_dir}")
    return candidates[0]


def compare_h5(a_path: Path, b_path: Path) -> dict[str, object]:
    import h5py

    with h5py.File(a_path, "r") as a, h5py.File(b_path, "r") as b:
        da, db = a["CT/CASE_000"], b["CT/CASE_000"]
        if da.shape != db.shape:
            return {"shape_equal": False, "a": list(da.shape), "b": list(db.shape)}
        max_diff = 0.0
        for z in range(0, da.shape[1], SLAB):
            max_diff = max(max_diff, float(np.abs(da[0, z : z + SLAB] - db[0, z : z + SLAB]).max()))

        # Geometry attributes are strings whose spelling differs by writer ("[0. 0. 0.]" against
        # "[0.0, 0.0, 0.0]"): compare the numbers they carry, not the text.
        def numbers(value: object) -> np.ndarray:
            text = value.decode() if isinstance(value, bytes) else str(value)
            return np.array([float(v) for v in re.findall(r"-?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?", text)])

        # KonfAI writes the geometry under stacked keys (Origin_0, Spacing_0, Direction_0); the loop
        # copied the source's bare keys.
        def attr(dataset: object, key: str) -> object:
            attrs_ = dataset.attrs  # type: ignore[attr-defined]
            return attrs_.get(key) if key in attrs_ else attrs_.get(f"{key}_0")

        attrs = {k: (str(attr(da, k)), str(attr(db, k))) for k in ("Origin", "Spacing", "Direction")}
        attrs_equal = all(
            np.allclose(numbers(attr(da, k)), numbers(attr(db, k))) for k in ("Origin", "Spacing", "Direction")
        )
        return {"shape_equal": True, "max_abs_diff": max_diff, "attrs_equal": attrs_equal, "attrs": attrs}


def region_rows(transforms_dir: Path) -> int | None:
    """The plan's region height in rows, from the run's Transforms log when it names one."""
    for log in sorted(transforms_dir.rglob("*.txt")) + sorted(transforms_dir.rglob("*.log")):
        match = re.search(r"(\d+)\s*rows", log.read_text(errors="ignore"))
        if match:
            return int(match.group(1))
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gib", type=float, default=2.0)
    parser.add_argument("--budgets", default="1,8", help="declared memory budgets in GiB")
    parser.add_argument("--sweep", action="store_true", help="budgets 0.0625..8 GiB, twice each")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--worker", nargs=4, metavar=("STORE", "OUT", "BUDGET_GIB", "TRANSFORMS_DIR"), help=argparse.SUPPRESS
    )
    parser.add_argument("--worker-prof", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--naive-worker", nargs=2, metavar=("STORE", "OUT"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        store_, out_, budget_, tdir_ = args.worker
        _worker_konfai(Path(store_), Path(out_), float(budget_), Path(tdir_), args.worker_prof)
        return
    if args.naive_worker:
        _worker_naive(Path(args.naive_worker[0]), Path(args.naive_worker[1]))
        return
    gate = machine_gate(force=args.force)
    fp = fingerprint()

    gib = 0.25 if args.quick else args.gib
    repeats = 1 if args.quick else args.repeats
    budgets = [0.0625, 0.125, 0.25, 0.5, 1, 2, 8] if args.sweep else [float(b) for b in args.budgets.split(",")]
    if args.quick:
        budgets = budgets[:1]
    scratch = Path(tempfile.mkdtemp(prefix="konfai_perf_transform_"))
    print(f"[perf] synthesizing {gib:g} GiB under {scratch}", flush=True)
    store, shape = synthesize(scratch, gib)

    result: dict[str, object] = {
        "gate_warnings": gate.warnings,
        "gib": gib,
        "shape_zyx": shape,
        "repeats": repeats,
        "scratch": str(scratch),
    }
    metrics: dict[str, float] = {}

    naive_out = scratch / "naive.h5"
    naive_runs = [
        in_fresh_process(["--naive-worker", str(store), str(naive_out)])
        for _ in range(repeats + (0 if args.quick else 1))
    ]
    naive_timed = naive_runs[1:] if len(naive_runs) > 1 else naive_runs
    metrics["naive_wall_s"] = round(statistics.median(r["wall_s"] for r in naive_timed), 3)
    metrics["naive_peak_rss_gib"] = round(max(r["peak_bytes"] for r in naive_timed) / 2**30, 3)
    result["naive_runs_s"] = [round(r["wall_s"], 3) for r in naive_runs]

    sweep_rows = []
    for budget in budgets:
        tag = f"{budget:g}".replace(".", "p")
        runs = []
        for i in range(repeats + (0 if args.quick else 1)):
            out_dir = scratch / f"out_b{tag}_{i}"
            tdir = scratch / f"Transforms_b{tag}_{i}"
            runs.append(in_fresh_process(["--worker", str(store), str(out_dir), str(budget), str(tdir)]))
        timed = runs[1:] if len(runs) > 1 else runs  # the first fresh process also warms the page cache
        wall = statistics.median(r["wall_s"] for r in timed)
        peak = max(r["peak_bytes"] for r in timed)
        metrics[f"konfai_wall_s_b{tag}"] = round(wall, 3)
        metrics[f"konfai_peak_rss_gib_b{tag}"] = round(peak / 2**30, 3)
        metrics[f"konfai_floor_rss_gib_b{tag}"] = round(runs[0]["floor_bytes"] / 2**30, 3)
        rows = region_rows(scratch / f"Transforms_b{tag}_0")
        sweep_rows.append(
            {
                "budget_gib": budget,
                "walls_s": [round(r["wall_s"], 3) for r in runs],
                "peak_rss_gib": round(peak / 2**30, 3),
                "region_rows": rows,
                "plan": plan_lines(scratch / f"Transforms_b{tag}_0"),
            }
        )
        if budget == budgets[0]:
            check = compare_h5(find_output(scratch / f"out_b{tag}_0"), naive_out)
            result["konfai_vs_naive"] = check
            metrics["max_abs_diff_konfai_vs_naive"] = float(check.get("max_abs_diff", float("nan")))
        for stray in scratch.glob(f"out_b{tag}_*"):
            if stray.is_file():
                stray.unlink(missing_ok=True)
            else:
                shutil.rmtree(stray, ignore_errors=True)
    result["by_budget"] = sweep_rows

    prof_path = scratch / "transform.prof"
    in_fresh_process(
        [
            "--worker",
            str(store),
            str(scratch / "out_prof"),
            str(budgets[0]),
            str(scratch / "Transforms_prof"),
            "--worker-prof",
            str(prof_path),
        ]
    )
    summary = cprofile_summary(
        prof_path,
        families={
            "statistics": r"statistics\.py",
            "sweep": r"sweep\.py|manager\.py|materialize\.py",
            "h5": r"h5\.py|h5py",
        },
    )
    result["cprofile"] = summary
    metrics["statistics_share_of_tottime"] = summary["family_share_of_tottime"]["statistics"]
    metrics["statistics_max_cumtime_s"] = summary["family_max_cumtime_s"]["statistics"]

    check = result.get("konfai_vs_naive", {})
    ok = bool(check.get("shape_equal")) and bool(check.get("attrs_equal")) and check.get("max_abs_diff", 1.0) <= 1e-5
    result["status"] = "outputs agree" if ok else "OUTPUTS DISAGREE with the naive loop"
    result["metrics"] = metrics
    first = f"{budgets[0]:g}".replace(".", "p")
    result["headline"] = (
        f"{gib:g} GiB: KonfAI {metrics[f'konfai_wall_s_b{first}']} s / {metrics[f'konfai_peak_rss_gib_b{first}']} GiB at "
        f"{budgets[0]:g} GiB budget"
        + (f", {metrics.get('konfai_wall_s_b8', '?')} s at 8 GiB" if 8 in budgets or 8.0 in budgets else "")
        + f" | naive {metrics['naive_wall_s']} s / {metrics['naive_peak_rss_gib']} GiB | statistics scan "
        f"{metrics['statistics_share_of_tottime'] * 100:.0f} % self, {metrics['statistics_max_cumtime_s']} s cumulative"
        f" | max diff {check.get('max_abs_diff', 'n/a')}"
    )
    path = write_result("transform", result, fp=fp, tag="sweep" if args.sweep else "")
    print(json.dumps(metrics, indent=2))
    print(f"[perf] {result['headline']}")
    print(f"[perf] written {path}")
    shutil.rmtree(scratch, ignore_errors=True)
    if not ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
