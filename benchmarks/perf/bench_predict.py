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

"""PREDICTION on the shipped Segmentation example: whole-volume against streamed, voxel-identical.

    python benchmarks/perf/bench_predict.py [--repeats 3] [--force] [--quick] [--cprofile]

The CLI runs on a scratch copy of ``examples/Segmentation`` with its shipped checkpoint, once on the
default route (every case assembles a whole volume: its accumulator is a sliver of the auto budget)
and once with ``KONFAI_STREAM_WORTH_THRESHOLD=0``, which forces the per-slab route. The loop's own
``[KonfAI] prediction`` clock is parsed; the two outputs are compared voxel by voxel through
SimpleITK, and against the shipped reference when it exists. A time is only reported when the voxels
agree.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
from pathlib import Path

from harness import (
    REPO,
    CliRun,
    copy_example,
    cprofile_summary,
    fingerprint,
    konfai_executable,
    machine_gate,
    run_cli,
    write_result,
)


def newest_checkpoint(copy: Path) -> Path:
    candidates = sorted((copy / "Checkpoints" / "SEG_BASELINE").glob("*.pt"))
    if not candidates:
        raise SystemExit("[perf] no checkpoint under Checkpoints/SEG_BASELINE of the example")
    return candidates[-1]


def predict_once(
    copy: Path, checkpoint: Path, out_dir: Path, *, streamed: bool, gpu: bool, cprofile: Path | None
) -> CliRun:
    args = ["PREDICTION", "-c", "Prediction.yml", "--models", str(checkpoint), "-y", "--predictions-dir", str(out_dir)]
    if gpu:
        args += ["--gpu", "0"]
    argv = (
        [sys.executable, "-m", "cProfile", "-o", str(cprofile), "-m", "konfai.main", *args]
        if cprofile
        else [*konfai_executable(), *args]
    )
    env = {"KONFAI_STREAM_WORTH_THRESHOLD": "0"} if streamed else {}
    return run_cli(argv, cwd=copy, env=env, log_path=out_dir.with_suffix(".log"))


def compare_outputs(a_root: Path, b_root: Path) -> dict[str, object]:
    """Differing voxels and geometry agreement between two prediction trees, case by case."""
    import numpy as np
    import SimpleITK as sitk

    cases = sorted(p.name for p in (a_root / "SEG_BASELINE" / "Dataset").iterdir() if p.is_dir())
    differing = 0
    total = 0
    geometry_ok = True
    per_case: dict[str, int] = {}
    for case in cases:
        a = sitk.ReadImage(str(a_root / "SEG_BASELINE" / "Dataset" / case / "PRED.mha"))
        b_path = b_root / "SEG_BASELINE" / "Dataset" / case / "PRED.mha"
        if not b_path.exists():
            geometry_ok = False
            continue
        b = sitk.ReadImage(str(b_path))
        same_geometry = (
            a.GetSize() == b.GetSize()
            and np.allclose(a.GetSpacing(), b.GetSpacing())
            and np.allclose(a.GetOrigin(), b.GetOrigin(), atol=1e-4)
            and np.allclose(a.GetDirection(), b.GetDirection())
        )
        geometry_ok = geometry_ok and same_geometry
        if not same_geometry:
            continue
        diff = int((sitk.GetArrayFromImage(a) != sitk.GetArrayFromImage(b)).sum())
        per_case[case] = diff
        differing += diff
        total += int(np.prod(a.GetSize()))
    return {
        "cases": len(cases),
        "differing_voxels": differing,
        "total_voxels": total,
        "geometry_ok": geometry_ok,
        "per_case": per_case,
    }


def phases(run: CliRun) -> dict[str, float]:
    clock = run.clocks.get("prediction", {})
    return {
        "loop_s": clock.get("wall", 0.0),
        "fetch_s": clock.get("fetch", 0.0),
        "forward_s": clock.get("forward", 0.0),
        "blend_s": clock.get("blend", 0.0),
        "finalize_stream_s": clock.get("finalize(stream)", 0.0),
        "finalize_case_s": clock.get("finalize(case)", 0.0),
        "writer_s": clock.get("after_bar:writer", 0.0),
        "startup_s": run.clocks.get("startup", {}).get("wall", 0.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--cprofile", action="store_true", help="run the whole-volume route once under cProfile (diagnostic)"
    )
    parser.add_argument("--cpu", action="store_true", help="no --gpu 0 (the default uses the GPU when torch sees one)")
    args = parser.parse_args()
    repeats = 1 if args.quick else args.repeats
    gate = machine_gate(force=args.force)
    fp = fingerprint()

    import torch

    gpu = torch.cuda.is_available() and not args.cpu
    scratch = Path(tempfile.mkdtemp(prefix="konfai_perf_predict_"))
    copy = copy_example("Segmentation", scratch)
    checkpoint = newest_checkpoint(copy)
    reference = REPO / "examples" / "Segmentation" / "Predictions"

    result: dict[str, object] = {
        "gate_warnings": gate.warnings,
        "repeats": repeats,
        "gpu": gpu,
        "scratch": str(scratch),
    }
    metrics: dict[str, float] = {}
    runs: dict[str, list[CliRun]] = {"whole": [], "stream": []}
    for route in ("whole", "stream"):
        out_dir = scratch / f"pred_{route}"
        for _ in range(repeats + (0 if args.quick else 1)):  # one warmup unless quick
            run = predict_once(copy, checkpoint, out_dir, streamed=route == "stream", gpu=gpu, cprofile=None)
            if run.returncode != 0:
                result["failure"] = {"route": route, "returncode": run.returncode, "tail": run.output[-3000:]}
                write_result("predict", result, fp=fp)
                raise SystemExit(f"[perf] PREDICTION ({route}) failed with rc {run.returncode}; see the result JSON")
            runs[route].append(run)
        timed = runs[route][1:] if len(runs[route]) > 1 else runs[route]
        metrics[f"wall_s_{route}"] = round(statistics.median(r.wall_s for r in timed), 3)
        for key in phases(timed[0]):
            metrics[f"{key.replace('_s', '')}_s_{route}"] = round(statistics.median(phases(r)[key] for r in timed), 3)
        metrics[f"peak_rss_gib_{route}"] = round(max(r.peak_rss_gib for r in timed), 3)
        metrics[f"gpu_delta_mib_{route}"] = max(r.gpu_peak_mib - r.gpu_baseline_mib for r in timed)
        result[f"runs_{route}"] = [
            {"wall_s": r.wall_s, "clocks": r.clocks, "peak_rss_gib": r.peak_rss_gib} for r in runs[route]
        ]

    identity = compare_outputs(scratch / "pred_whole", scratch / "pred_stream")
    result["whole_vs_stream"] = identity
    metrics["differing_voxels_whole_vs_stream"] = float(identity["differing_voxels"])
    if reference.is_dir():
        against_reference = compare_outputs(scratch / "pred_whole", reference)
        result["whole_vs_reference"] = against_reference
        metrics["differing_voxels_whole_vs_reference"] = float(against_reference["differing_voxels"])
    if not identity["geometry_ok"] or identity["differing_voxels"] != 0:
        result["status"] = "OUTPUTS DIFFER between the whole-volume and the streamed route"
    else:
        result["status"] = "voxel-identical"

    if args.cprofile:
        prof = scratch / "predict.prof"
        predict_once(copy, checkpoint, scratch / "pred_prof", streamed=False, gpu=gpu, cprofile=prof)
        result["cprofile"] = cprofile_summary(
            prof,
            families={
                "fetch": r"samples\.py|sources\.py|sitk_file\.py|dataset/core\.py|grid\.py",
                "blend": r"accumulate\.py|output\.py",
                "network": r"network\.py|torch",
            },
        )

    result["clock_note"] = "the [KonfAI] clocks print only above one second: a 0.0 here means under a second, not zero"
    result["metrics"] = metrics
    result["headline"] = (
        f"prediction whole {metrics['wall_s_whole']} s (loop {metrics['loop_s_whole']}: fetch {metrics['fetch_s_whole']} + "
        f"forward {metrics['forward_s_whole']}) | streamed {metrics['wall_s_stream']} s (loop {metrics['loop_s_stream']}) | "
        f"{identity['differing_voxels']} of {identity['total_voxels']} voxels differ | RSS {metrics['peak_rss_gib_whole']} GiB"
    )
    path = write_result("predict", result, fp=fp)
    print(json.dumps(metrics, indent=2))
    print(f"[perf] {result['headline']}")
    print(f"[perf] written {path}")
    if result["status"] != "voxel-identical":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
