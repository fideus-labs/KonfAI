# SPDX-License-Identifier: Apache-2.0
"""The apps against the tools they wrap, on the same input, torch and GPU.

A manifest names the apps: for each, the KonfAI command line, the original tool's command line and
the cases (``S``/``M``/``L`` files). Both commands read the case (``{input}``, or the names a mapping case declares, ``{fixed}`` and
``{moving}``) and write under ``{output}``; the
original runs in its own environment (``{venv}``), the app in this one. One run per case per tool,
after a warm-up on the smallest case so the weights are on disk and the kernels compiled.

    pixi run --environment dev python benchmarks/perf/bench_apps.py --manifest apps.json --venv ~/.cache/konfai-bench/venv-orig

The manifest is not in the tree (inputs are patient-sized volumes); ``apps_manifest.example.json``
is the shape. Numbers: wall, whole-tree peak RSS, the GPU memory the run added, and the labels the
output carries (a sanity check, not an accuracy score).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import fingerprint, machine_gate, run_cli, write_result

SIZES = ("S", "M", "L")


def _labels(directory: Path) -> dict[str, Any]:
    """What the run wrote: the label maps found, their distinct labels and foreground share."""
    try:
        import SimpleITK as sitk
    except ImportError:
        return {"files": sorted(str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file())}
    import numpy as np

    volumes = sorted(p for p in directory.rglob("*") if p.suffix in {".gz", ".mha", ".nrrd", ".nii"})
    summary: dict[str, Any] = {"files": len(volumes)}
    labels: set[int] = set()
    foreground = 0.0
    for path in volumes:
        array = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
        if not np.issubdtype(array.dtype, np.integer) and not np.array_equal(array, np.round(array)):
            continue
        values = np.unique(array.astype(np.int64))
        labels |= {int(v) for v in values if v != 0}
        foreground = max(foreground, float(np.count_nonzero(array)) / array.size)
    summary["labels"] = len(labels)
    summary["foreground_share"] = round(foreground, 4)
    return summary


def _case_fields(case: str | dict[str, str]) -> dict[str, str]:
    """A case is one file (``{input}``) or a mapping of named files (``{fixed}``, ``{moving}``)."""
    if isinstance(case, str):
        return {"input": str(Path(case).expanduser())}
    return {key: str(Path(value).expanduser()) for key, value in case.items()}


def _run(
    name: str,
    tool: str,
    template: list[str],
    size: str,
    case: str | dict[str, str],
    work: Path,
    venv: Path | None,
    log_dir: Path,
) -> dict:
    output = work / f"{name}-{tool}-{size}"
    shutil.rmtree(output, ignore_errors=True)
    output.mkdir(parents=True)
    fields = {**_case_fields(case), "output": str(output), "venv": str(venv) if venv else ""}
    argv = [part.format(**fields) for part in template]
    run = run_cli(argv, cwd=work, log_path=log_dir / f"{name}-{tool}-{size}.log")
    result = {
        "argv": argv,
        "returncode": run.returncode,
        "wall_s": run.wall_s,
        "peak_rss_gib": run.peak_rss_gib,
        "gpu_delta_mib": run.gpu_peak_mib - run.gpu_baseline_mib,
        "output": _labels(output) if run.returncode == 0 else {},
        "tail": run.output[-800:],
    }
    print(
        f"[perf] {name} {tool} {size}: {run.wall_s:.1f} s, {run.peak_rss_gib:.2f} GiB RSS, "
        f"+{result['gpu_delta_mib']} MiB GPU, rc {run.returncode}"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--venv", type=Path, default=None, help="the original tools' environment ({venv} in the manifest)"
    )
    parser.add_argument("--only", default="", help="comma-separated app names")
    parser.add_argument("--sizes", default=",".join(SIZES))
    parser.add_argument("--tools", default="konfai,original", help="comma-separated subset of konfai, original")
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--work", type=Path, default=None, help="scratch directory (a temporary one by default)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    gate = machine_gate(force=args.force)
    # The apps run as their users run them: the tests' thread pin starves the CPU fold of a 5-model ensemble.
    pinned = [key for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS") if os.environ.get(key)]
    if pinned:
        gate.warnings.append(f"thread pin set ({', '.join(pinned)}): unset it, the apps are timed unpinned")
        if not args.force:
            raise SystemExit("[perf] " + gate.warnings[-1])
    fp = fingerprint()
    work = args.work or Path(tempfile.mkdtemp(prefix="konfai-bench-apps-"))
    log_dir = work / "logs"
    sizes = [s for s in args.sizes.split(",") if s]
    selected = [a for a in manifest["apps"] if not args.only or a["name"] in args.only.split(",")]
    result: dict[str, Any] = {"gate_warnings": gate.warnings, "work": str(work), "apps": {}}
    metrics: dict[str, float] = {}
    for app in selected:
        wanted = args.tools.split(",")
        tools = {tool: app[tool] for tool in ("konfai", "original") if tool in wanted and app.get(tool)}
        cases = {size: app["cases"][size] for size in sizes if size in app["cases"]}
        entry: dict[str, Any] = {}
        for tool, template in tools.items():
            if not args.no_warmup and cases:
                smallest = min(cases, key=SIZES.index)
                _run(app["name"], f"{tool}-warmup", template, smallest, cases[smallest], work, args.venv, log_dir)
            for size, case in cases.items():
                entry[f"{tool}_{size}"] = run = _run(app["name"], tool, template, size, case, work, args.venv, log_dir)
                if run["returncode"] == 0:
                    metrics[f"{app['name']}_{tool}_{size}_wall_s"] = run["wall_s"]
                    metrics[f"{app['name']}_{tool}_{size}_peak_rss_gib"] = run["peak_rss_gib"]
        for size in cases:
            a, b = entry.get(f"konfai_{size}"), entry.get(f"original_{size}")
            if a and b and a["returncode"] == 0 and b["returncode"] == 0 and a["wall_s"]:
                metrics[f"{app['name']}_{size}_speedup"] = round(b["wall_s"] / a["wall_s"], 2)
        result["apps"][app["name"]] = entry
    result["metrics"] = metrics
    path = write_result("apps", result, fp=fp)
    print(f"[perf] written {path}")
    if os.environ.get("KONFAI_PERF_KEEP_WORK") is None and args.work is None:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
