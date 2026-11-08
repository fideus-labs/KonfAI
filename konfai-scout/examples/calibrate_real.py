# SPDX-License-Identifier: Apache-2.0
"""Calibration on real registrations: defects planted in real displacement fields, clean copies as controls.

    python examples/calibrate_real.py --case NAME FIXED MOVING FIELD [--case ...] --out OUT
    python examples/calibrate_real.py --cases cases.json --out OUT --provider claude-code --allow-cloud-images

A case is a development registration you point to: a fixed image, a moving image and its registration (a
displacement field image on the fixed grid, or an ITK transform file, sampled on the fixed grid once). In a JSON list
(`[{"name", "fixed", "moving", "field", "fixed_labels"?, "moving_labels"?}]`) a case may also bring its labels,
which Scout then measures Dice against. Never a test case: Scout looks at it.

Each case is inspected as is (the clean copy: a real field, not a perfect one, so a non-trivial control) and with
defects planted in its field, one per copy: a local fold (a sharp 10 mm bump over 3 mm, which folds the map), a
local shift (a smooth 4 mm bump over 6 mm, which does not), a global shift (3 mm everywhere). The local ones sit at
the centre of the fixed labels (the grid's centre without labels). Every copy is inspected `--repeats` times.

A planted defect is found when an error observation of its type overlaps its box (a folding for the fold, a residual
misalignment for a shift; anywhere for the global shift); on a clean copy, the same match is a false alarm. Sensitivity per defect
type is found / planted, specificity is 1 - false alarms / clean inspections, test-retest agreement is how often two
inspections of one copy agree (on each defect's match, and on the verdict). `--defects-per-case` rotates the defect
types over the cases to bound the number of inspections. Writes `calibration.json` and `calibration.md` in OUT.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from itertools import pairwise
from pathlib import Path

import numpy as np
from konfai_scout import InspectionCase, Report, ScoutConfig, inspect
from konfai_scout.config import Budget
from konfai_scout.derived import Derived
from konfai_scout.measure import jacobian_determinant
from konfai_scout.planes import box_to_dict
from konfai_scout.report import match_defects
from konfai_scout.synthetic import _image, _write

#: Each defect: what it is called in a report, its bump (amplitude and width in mm; none for a global shift).
DEFECTS = {
    "local_fold": {"defect": "folding", "amplitude_mm": 10.0, "sigma_mm": 3.0},
    "local_shift": {"defect": "residual_misalignment", "amplitude_mm": 4.0, "sigma_mm": 6.0},
    "global_shift": {"defect": "residual_misalignment", "amplitude_mm": 3.0},
}
TRANSFORM_SUFFIXES = (".h5", ".hdf5", ".tfm", ".txt")


def _window(center: np.ndarray, half_mm: float, spacing: np.ndarray, shape: tuple[int, ...]) -> tuple[slice, ...]:
    half = np.ceil(half_mm / spacing).astype(int)
    return tuple(slice(max(0, int(c) - h), min(n, int(c) + h + 1)) for c, h, n in zip(center, half, shape, strict=True))


def plant(entry: dict[str, str], folder: Path) -> tuple[dict[str, Path], dict[str, dict[str, object]], dict[str, Path]]:
    """The case's inputs but its registration, every defect's spec at its place, one field per copy.

    ponytail: holds the field twice (the clean one and the copy being planted): a development case, not a cohort."""
    field = Path(entry["field"])
    inputs = {"fixed": Path(entry["fixed"]), "moving": Path(entry["moving"])}
    inputs["transform" if field.suffix.lower() in TRANSFORM_SUFFIXES else "field"] = field
    inputs |= {role: Path(entry[role]) for role in ("fixed_labels", "moving_labels") if entry.get(role)}
    case = InspectionCase("registration", entry["name"], inputs)
    case.stage(folder / "stage")
    volume = Derived(case, folder).field()  # a transform is sampled on the fixed grid once, by Scout
    grid = volume.grid
    shape = tuple(int(n) for n in grid.size_zyx)
    clean = volume.region((0, 0, 0), shape).astype(np.float32)  # [3, Z, Y, X], mm along x, y, z
    spacing = np.asarray(grid.spacing_xyz, dtype=np.float64)[::-1]
    center = np.asarray(shape, dtype=np.float64) / 2.0
    if "fixed_labels" in inputs:
        labels = case.volume("fixed_labels").region((0, 0, 0), shape)[0]
        center = np.argwhere(labels > 0).mean(axis=0) if labels.any() else center
    specs: dict[str, dict[str, object]] = {}
    fields = {"clean": _write(_image(clean, grid, vector=True), folder / "clean.mha")}
    for name, defect in DEFECTS.items():
        planted = clean.copy()
        spec: dict[str, object] = {"defect": defect["defect"]}
        if "sigma_mm" in defect:
            sigma = float(defect["sigma_mm"])
            near = _window(center, 4 * sigma, spacing, shape)  # beyond 4 sigma the bump is below 4e-4 of its height
            axes = [(np.arange(w.start, w.stop) - c) * h for w, c, h in zip(near, center, spacing, strict=True)]
            distance2 = sum(a**2 for a in np.ix_(*axes))
            planted[(0, *near)] += float(defect["amplitude_mm"]) * np.exp(-distance2 / (2 * sigma**2))  # along x
            window = _window(center, 2 * sigma, spacing, shape)
            spec["box_mm"] = box_to_dict(grid.world_box(window))
            spec["min_jacobian"] = round(float(jacobian_determinant(planted[(slice(None), *window)], grid).min()), 3)
        else:
            planted[0] += float(defect["amplitude_mm"])
        specs[name] = spec
        fields[name] = _write(_image(planted, grid, vector=True), folder / f"{name}.mha")
    return {role: path for role, path in inputs.items() if role not in ("field", "transform")}, specs, fields


def _inspect(job: tuple[InspectionCase, ScoutConfig, Path]) -> dict[str, object]:
    case, config, out = job
    report = inspect(case, config, out)
    return {
        "verdict": report.conclusion.verdict if report.conclusion else None,
        "concluded_by": report.conclusion.by if report.conclusion else None,
        "errors": sum(o.kind != "measured_property" for o in report.observations),
        "confirmed": sum(o.kind == "confirmed_error" for o in report.observations),
        "tool_calls": len(report.tool_calls),
        "budget_exhausted": report.budget.get("exhausted"),
        "cost_usd": report.usage.cost_usd,
        "duration_s": round(report.duration_s, 1),
        "error": report.error,
        "report": str(out),
    }


def _matched(report: Report, specs: dict[str, dict[str, object]]) -> dict[str, bool]:
    """Per defect: whether an error observation of its type overlaps its box (names it, for one without a box)."""
    out = {}
    for name, spec in specs.items():
        typed = replace(report, observations=[o for o in report.observations if o.defect == spec["defect"]])
        out[name] = bool(match_defects(typed, {name: spec})["found"])
    return out


def score(rows: list[dict[str, object]], probes: dict[str, dict[str, dict[str, object]]]) -> dict[str, object]:
    """Sensitivity and specificity per defect type, test-retest agreement, from the inspections' reports."""
    for r in rows:
        r["matched"] = _matched(Report.load(Path(str(r["report"]))), probes[str(r["case"])])
    done = [r for r in rows if not r["error"]]  # a provider failure saw nothing: it counts neither way
    summary: dict[str, dict[str, object]] = {}
    for t in DEFECTS:
        hits = [r["matched"][t] for r in done if r["variant"] == t]  # type: ignore[index]
        alarms = [r["matched"][t] for r in done if r["variant"] == "clean"]  # type: ignore[index]
        summary[t] = {
            "planted_inspections": len(hits),
            "found": sum(hits),
            "sensitivity": round(sum(hits) / len(hits), 3) if hits else None,
            "clean_inspections": len(alarms),
            "false_alarms": sum(alarms),
            "specificity": round(1 - sum(alarms) / len(alarms), 3) if alarms else None,
        }
    pairs: dict[tuple[object, object], list[dict[str, object]]] = {}
    for r in done:
        pairs.setdefault((r["case"], r["variant"]), []).append(r)
    matches, verdicts = [], []
    for group in pairs.values():
        for a, b in pairwise(group):
            relevant = list(DEFECTS) if a["variant"] == "clean" else [a["variant"]]
            matches += [a["matched"][t] == b["matched"][t] for t in relevant]  # type: ignore[index]
            verdicts.append(a["verdict"] == b["verdict"])
    return {
        "failed": len(rows) - len(done),
        "per_defect": summary,
        "test_retest": {
            "match_pairs": len(matches),
            "match_agreement": round(sum(matches) / len(matches), 3) if matches else None,
            "verdict_pairs": len(verdicts),
            "verdict_agreement": round(sum(verdicts) / len(verdicts), 3) if verdicts else None,
        },
        "inspections": [{k: v for k, v in r.items() if k != "report"} for r in rows],
    }


def calibrate(
    entries: list[dict[str, str]], out: Path, config: ScoutConfig, repeats: int, per_case: int, jobs: int
) -> dict[str, object]:
    jobs_list, keys, probes = [], [], {}
    types = list(DEFECTS)
    for n, entry in enumerate(entries):
        base, specs, fields = plant(entry, out / "fields" / entry["name"])
        probes[entry["name"]] = specs
        planted = [types[(n * per_case + k) % len(types)] for k in range(min(per_case, len(types)))]
        for variant in ["clean", *planted]:
            case = InspectionCase("registration", entry["name"], {**base, "field": fields[variant]})
            for repeat in range(repeats):
                keys.append((entry["name"], variant, repeat))
                jobs_list.append((case, config, out / "reports" / entry["name"] / f"{variant}-{repeat}"))
    if jobs > 1:  # KonfAI's runs set process-wide state: one inspection per process
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            results = list(pool.map(_inspect, jobs_list))
    else:
        results = [_inspect(job) for job in jobs_list]
    rows = [
        {"case": name, "variant": variant, "repeat": repeat, **result}
        for (name, variant, repeat), result in zip(keys, results, strict=True)
    ]
    return {
        "provider": config.provider,
        "model": config.model,
        "mode": config.mode,
        "blind": config.blind,
        "budget": config.budget.to_dict(),
        "repeats": repeats,
        "defects": DEFECTS,
        "cases": probes,
        **score(rows, probes),
    }


def markdown(result: dict[str, object]) -> str:
    lines = [
        f"Provider {result['provider']} / {result['model']}, mode {result['mode']}, blind {result['blind']}, "
        f"{len(result['inspections'])} inspections ({result['repeats']} per copy), "  # type: ignore[arg-type]
        f"{result['failed']} failed (left out).",
        "",
        "| defect | planted | found | sensitivity | clean | false alarms | specificity |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, s in result["per_defect"].items():  # type: ignore[union-attr]
        lines.append(
            f"| {name} | {s['planted_inspections']} | {s['found']} | {s['sensitivity']} | {s['clean_inspections']} "
            f"| {s['false_alarms']} | {s['specificity']} |"
        )
    retest = result["test_retest"]
    lines += [
        "",
        f"Test-retest: defect matches agree in {retest['match_agreement']} of {retest['match_pairs']} pairs, "  # type: ignore[index]
        f"verdicts in {retest['verdict_agreement']} of {retest['verdict_pairs']}.",  # type: ignore[index]
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", nargs=4, action="append", default=[], metavar=("NAME", "FIXED", "MOVING", "FIELD"))
    parser.add_argument("--cases", type=Path, help="A JSON list of cases (labels allowed).")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--provider", default="mock")
    parser.add_argument("--model")
    parser.add_argument("--allow-cloud-images", action="store_true")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--defects-per-case", type=int, default=len(DEFECTS))
    parser.add_argument("--max-tool-calls", type=int, default=Budget.max_tool_calls)
    parser.add_argument("--max-seconds", type=float, default=Budget.max_seconds)
    parser.add_argument("--jobs", type=int, default=1, help="Inspections run at once.")
    args = parser.parse_args(argv)
    entries = [dict(zip(("name", "fixed", "moving", "field"), c, strict=True)) for c in args.case]
    if args.cases:
        entries += json.loads(args.cases.read_text(encoding="utf-8"))
    if not entries:
        parser.error("give --case or --cases")
    config = ScoutConfig(
        provider=args.provider,
        model=args.model,
        allow_cloud_images=args.allow_cloud_images,
        budget=Budget(max_tool_calls=args.max_tool_calls, max_seconds=args.max_seconds),
    )
    result = calibrate(entries, args.out, config, args.repeats, args.defects_per_case, args.jobs)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "calibration.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (args.out / "calibration.md").write_text(markdown(result), encoding="utf-8")
    print(markdown(result))


if __name__ == "__main__":
    main()
