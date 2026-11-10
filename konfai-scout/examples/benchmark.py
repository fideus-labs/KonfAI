# SPDX-License-Identifier: Apache-2.0
"""Scout's benchmark: defects planted one at a time in real images, clean controls beside them, every case inspected at
each depth, and what was found, missed and invented counted.

    python examples/benchmark.py build MANIFEST.json --out BENCH [--phantoms]
    python examples/benchmark.py run BENCH --provider claude-code --allow-cloud-images --depth quick standard deep
    python examples/benchmark.py score BENCH

The manifest names the sources, never a case to keep (Scout looks at them, a cloud model too when allowed):

    {"segmentation": [{"name": "isles0003", "image": "dwi.nii.gz", "truth": "msk.nii.gz", "label": "lesion"}],
     "registration": [{"name": "ct01", "image": "ct.nii.gz"}],
     "synthesis": [{"name": "mr2ct01", "source": "mr.nii.gz", "target": "ct.nii.gz", "mask": "mask.nii.gz"}]}

What each source becomes (`cases/<task>/<source>__<variant>/`, a `case.json` and the `defects.json` planted):

- segmentation: the truth as the prediction (`clean`), its main component's upper 40 % left out
  (`under_segmentation`), a 5 mm leak on one side into the body (`leak`), a 6 mm blob 20 mm or more away
  (`spurious_structure`), a hole at its deepest point (`holes`). The truth is not given to Scout.
- registration: the image is the moving one, the fixed one is it pulled through a known smooth field of a few mm, and
  the field inspected is that exact one (`clean`), zeroed over a 30 mm cube where it moved most
  (`residual_misalignment`), or with a sharp 12 mm bump that folds the map (`folding`).
- synthesis (a CT target in HU): a plausible model output, the target smoothed and noisier inside the mask
  (`clean`), with +150 HU over a 30 mm cube of soft tissue (`intensity_bias`), a 6 mm air pocket
  (`hallucinated_structure`), or bone erased over a 30 mm cube (`missing_structure`). Target and mask are given.

`--phantoms` adds the synthetic phantoms (`konfai_scout.synthetic`, several defects each). A defect that cannot be
planted in a source (a lesion too thin for a hole) is skipped and said. A planted defect is found when an error
observation's box overlaps it; on a clean case every error observation is a false alarm. `score` writes
`scores.json` and `scores.md`: per task and depth, the sensitivity by defect, the false alarms, the verdicts, the
agreement between repeats, the cost, the time and the tokens.
"""

from __future__ import annotations

import argparse
import itertools
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import SimpleITK as sitk
from konfai_scout import InspectionCase, Report
from konfai_scout.report import match_defects
from konfai_scout.synthetic import MAKERS
from scipy import ndimage

DEPTHS = ("quick", "standard", "deep")
SERIOUS = ("moderate", "major", "critical")


# ---------------------------------------------------------------- planting


def _read(path: str | Path) -> tuple[sitk.Image, np.ndarray]:
    image = sitk.ReadImage(str(path))
    return image, sitk.GetArrayFromImage(image).astype(np.float32)


def _write(array: np.ndarray, like: sitk.Image, path: Path) -> Path:
    image = sitk.GetImageFromArray(array)
    image.CopyInformation(like)
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(path), True)
    return path


def _box(like: sitk.Image, voxels: np.ndarray) -> dict[str, list[float]]:
    """The physical box (LPS, mm) of these voxels, their corners included."""
    where = np.argwhere(voxels)
    low, high = where.min(0)[::-1] - 0.5, where.max(0)[::-1] + 0.5
    corners = np.array(
        [
            like.TransformContinuousIndexToPhysicalPoint([float(c) for c in corner])
            for corner in itertools.product(*zip(low, high, strict=True))
        ]
    )
    return {"min": [round(float(v), 2) for v in corners.min(0)], "max": [round(float(v), 2) for v in corners.max(0)]}


def _ball(shape: tuple[int, ...], center: np.ndarray, radius_mm: float, spacing_zyx: np.ndarray) -> np.ndarray:
    axes = np.ogrid[tuple(slice(0, n) for n in shape)]
    return sum(((a - c) * s) ** 2 for a, c, s in zip(axes, center, spacing_zyx, strict=True)) <= radius_mm**2


def _cube(shape: tuple[int, ...], center: np.ndarray, side_mm: float, spacing_zyx: np.ndarray) -> np.ndarray:
    half = np.ceil(side_mm / 2 / spacing_zyx).astype(int)
    cube = np.zeros(shape, dtype=bool)
    cube[tuple(slice(max(0, int(c) - h), int(c) + h + 1) for c, h in zip(center, half, strict=True))] = True
    return cube


def _inside(mask: np.ndarray, margin_mm: float, spacing_zyx: np.ndarray) -> np.ndarray:
    """The voxels of a mask at least ``margin_mm`` from its border."""
    return ndimage.distance_transform_edt(mask, sampling=spacing_zyx) >= margin_mm


def _body(array: np.ndarray) -> np.ndarray:
    """The body: above a tenth of the 1st-99th percentile range, opened, filled, its largest piece."""
    low, high = np.percentile(array, (1, 99))
    body = ndimage.binary_fill_holes(ndimage.binary_opening(array > low + 0.1 * (high - low), iterations=2))
    pieces, count = ndimage.label(body)
    return pieces == (np.argmax(np.bincount(pieces.ravel())[1:]) + 1) if count else body


def _pick(rng: np.random.Generator, where: np.ndarray) -> np.ndarray | None:
    candidates = np.argwhere(where)
    return candidates[rng.integers(len(candidates))] if len(candidates) else None


def _spacing(image: sitk.Image) -> np.ndarray:
    return np.asarray(image.GetSpacing(), dtype=np.float64)[::-1]


def plant_segmentation(entry: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    """{variant: (prediction, {name: defect})}: the truth itself and one defect each."""
    like, image = _read(entry["image"])
    truth = _read(entry["truth"])[1] > 0
    spacing = _spacing(like)
    body = _body(image) | truth
    pieces, _ = ndimage.label(truth)
    main = pieces == (np.argmax(np.bincount(pieces.ravel())[1:]) + 1)
    variants: dict[str, Any] = {"clean": (truth, None)}
    rows = np.where(main.any((1, 2)))[0]
    if len(rows) >= 3:
        cut = rows.max() - int(0.4 * (rows.max() - rows.min() + 1)) + 1
        removed = main.copy()
        removed[:cut] = False
        variants["under_segmentation"] = (truth & ~removed, removed)
    columns = np.where(main.any((0, 1)))[0]
    side = np.zeros_like(main)
    side[:, :, : (columns.min() + columns.max()) // 2 + 1] = True
    leak = ~truth & side & body & (ndimage.distance_transform_edt(~main, sampling=spacing) <= 5.0)
    if leak.sum() >= 10:
        variants["leak"] = (truth | leak, leak)
    far = _inside(body, 8.0, spacing) & (ndimage.distance_transform_edt(~truth, sampling=spacing) >= 20.0)
    if (center := _pick(rng, far)) is not None:
        blob = _ball(truth.shape, center, 6.0, spacing) & body
        variants["spurious_structure"] = (truth | blob, blob)
    depth = ndimage.distance_transform_edt(main, sampling=spacing)
    if depth.max() >= 3.0:
        hole = _ball(truth.shape, np.unravel_index(np.argmax(depth), depth.shape), min(5.0, depth.max() / 2), spacing)
        variants["holes"] = (truth & ~hole, hole & truth)
    return {
        name: (
            prediction.astype(np.uint8),
            {name: {"defect": name, "box_mm": _box(like, where)}} if where is not None else {},
        )
        for name, (prediction, where) in variants.items()
    }


def plant_registration(entry: dict[str, Any], rng: np.random.Generator) -> tuple[sitk.Image, dict[str, Any]]:
    """The fixed image (the moving one pulled through a known field) and {variant: (field, defects)}."""
    moving, image = _read(entry["image"])
    spacing = _spacing(moving)
    body = _inside(_body(image), 10.0, spacing)
    z, y, x = (np.arange(n, dtype=np.float32) * s for n, s in zip(image.shape, spacing, strict=True))
    phase = rng.uniform(0, 2 * np.pi, 3)
    exact = np.zeros((3, *image.shape), dtype=np.float32)  # mm along x, y, z
    exact[0] = 4.0 * np.sin(y[None, :, None] / 15.0 + phase[0])
    exact[1] = 3.0 * np.cos(x[None, None, :] / 12.0 + phase[1])
    exact[2] = 2.0 * np.sin(z[:, None, None] / 10.0 + phase[2])

    def vector(field: np.ndarray) -> sitk.Image:
        out = sitk.GetImageFromArray(np.ascontiguousarray(np.moveaxis(field, 0, -1)), isVector=True)
        out.CopyInformation(moving)
        return out

    transform = sitk.DisplacementFieldTransform(sitk.Cast(vector(exact), sitk.sitkVectorFloat64))
    fixed = sitk.Resample(moving, moving, transform, sitk.sitkLinear, float(image.min()), sitk.sitkFloat32)
    variants: dict[str, Any] = {"clean": (vector(exact), {})}
    length = np.linalg.norm(exact, axis=0)
    if (center := _pick(rng, body & (length >= np.percentile(length[body], 75)))) is not None:
        zeroed = _cube(image.shape, center, 30.0, spacing) & _body(image)
        field = exact.copy()
        field[:, zeroed] = 0.0
        defect = {"defect": "residual_misalignment", "box_mm": _box(moving, zeroed)}
        variants["residual_misalignment"] = (vector(field), {"residual_misalignment": defect})
    if (center := _pick(rng, body)) is not None:
        bump = _ball(image.shape, center, 4.0, spacing)
        field = exact.copy()
        field[0, bump] += 12.0
        variants["folding"] = (vector(field), {"folding": {"defect": "folding", "box_mm": _box(moving, bump)}})
    return fixed, variants


def plant_synthesis(entry: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    """{variant: (synthetic, defects)}: a plausible model output and one defect each."""
    like, target = _read(entry["target"])
    mask = _read(entry["mask"])[1] > 0
    spacing = _spacing(like)
    base = ndimage.gaussian_filter(target, 0.8) + rng.normal(0, 15, target.shape).astype(np.float32)
    base[~mask] = target.min()
    soft = _inside(mask, 12.0, spacing) & (target > -100) & (target < 100)
    variants: dict[str, Any] = {"clean": (base, {})}
    if (center := _pick(rng, soft)) is not None:
        biased = _cube(target.shape, center, 30.0, spacing) & mask
        synthetic = base.copy()
        synthetic[biased] += 150.0
        variants["intensity_bias"] = (
            synthetic,
            {"intensity_bias": {"defect": "intensity_bias", "box_mm": _box(like, biased)}},
        )
    if (center := _pick(rng, soft & _inside(mask, 20.0, spacing))) is not None:
        pocket = _ball(target.shape, center, 6.0, spacing) & mask
        synthetic = base.copy()
        synthetic[pocket] = -750.0 + rng.normal(0, 20, int(pocket.sum()))
        defect = {"defect": "hallucinated_structure", "box_mm": _box(like, pocket)}
        variants["hallucinated_structure"] = (synthetic, {"hallucinated_structure": defect})
    bone = (target > 300) & mask
    if (center := _pick(rng, bone)) is not None:
        erased = _cube(target.shape, center, 30.0, spacing) & bone
        if erased.sum() >= 100:
            synthetic = base.copy()
            synthetic[erased] = 40.0 + rng.normal(0, 15, int(erased.sum()))
            defect = {"defect": "missing_structure", "box_mm": _box(like, erased)}
            variants["missing_structure"] = (synthetic, {"missing_structure": defect})
    return variants


def build(manifest: dict[str, list[dict[str, Any]]], out: Path, phantoms: bool = False, seed: int = 0) -> list[Path]:
    """Every case of the manifest's sources (and of the phantoms), each a folder with its case.json and defects.json."""
    built: list[Path] = []

    def save(task: str, folder: Path, inputs: dict[str, Any], defects: dict[str, Any], entry: dict[str, Any]) -> None:
        labels = {1: entry["label"]} if entry.get("label") else {}
        case = InspectionCase(
            task, folder.name, {k: Path(v) for k, v in inputs.items()}, labels, notes=entry.get("notes", "")
        )
        case.save(folder / "case.json")
        (folder / "defects.json").write_text(json.dumps(defects, indent=2))
        built.append(folder)

    for task, entries in manifest.items():
        for n, entry in enumerate(entries):
            rng = np.random.default_rng([seed, n, list(MAKERS).index(task)])
            root = out / "cases" / task
            if task == "segmentation":
                like = sitk.ReadImage(str(entry["image"]))
                for variant, (prediction, defects) in plant_segmentation(entry, rng).items():
                    folder = root / f"{entry['name']}__{variant}"
                    prediction_path = _write(prediction, like, folder / "prediction.nii.gz")
                    save(task, folder, {"image": entry["image"], "prediction": prediction_path}, defects, entry)
            elif task == "registration":
                fixed, variants = plant_registration(entry, rng)
                fixed_path = root / f"{entry['name']}__fixed.nii.gz"
                fixed_path.parent.mkdir(parents=True, exist_ok=True)
                sitk.WriteImage(fixed, str(fixed_path), True)
                for variant, (field, defects) in variants.items():
                    folder = root / f"{entry['name']}__{variant}"
                    folder.mkdir(parents=True, exist_ok=True)
                    sitk.WriteImage(field, str(folder / "field.nii.gz"), True)
                    inputs = {"fixed": fixed_path, "moving": entry["image"], "field": folder / "field.nii.gz"}
                    save(task, folder, inputs, defects, entry)
            else:
                like = sitk.ReadImage(str(entry["target"]))
                for variant, (synthetic, defects) in plant_synthesis(entry, rng).items():
                    folder = root / f"{entry['name']}__{variant}"
                    path = _write(synthetic.astype(np.float32), like, folder / "synthetic.nii.gz")
                    inputs = {
                        "source": entry["source"],
                        "synthetic": path,
                        "target": entry["target"],
                        "mask": entry["mask"],
                    }
                    save(task, folder, inputs, defects, entry)
            skipped = {
                "segmentation": {"under_segmentation", "leak", "spurious_structure", "holes"},
                "registration": {"residual_misalignment", "folding"},
                "synthesis": {"intensity_bias", "hallucinated_structure", "missing_structure"},
            }[task] - {
                p.name.split("__")[1]
                for p in built
                if p.parent.name == task and p.name.startswith(entry["name"] + "__")
            }
            if skipped:
                print(f"{task} {entry['name']}: not planted: {', '.join(sorted(skipped))}", file=sys.stderr)
    if phantoms:
        for task, maker in MAKERS.items():
            for variant, clean in (("clean", True), ("planted", False)):
                case, _ = maker(out / "cases" / task / f"phantom__{variant}", clean=clean)
                case.save(out / "cases" / task / f"phantom__{variant}" / "case.json")
                built.append(out / "cases" / task / f"phantom__{variant}")
    return built


# ---------------------------------------------------------------- running


def run(bench: Path, depths: list[str], repeats: int, jobs: int, flags: list[str]) -> None:
    """Every case at every depth, `repeats` times; a run whose report exists is kept (the benchmark resumes)."""
    todo = [
        (case, depth, bench / "runs" / depth / f"{case.parent.parent.name}.{case.parent.name}@{k}")
        for case in sorted(bench.glob("cases/*/*/case.json"))
        for depth in depths
        for k in range(1, repeats + 1)
    ]
    todo = [item for item in todo if not (item[2] / "report.json").is_file()]

    def one(item: tuple[Path, str, Path]) -> str:
        case, depth, out = item
        out.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, "-m", "konfai_scout.cli", "inspect", "--case", str(case), "--depth", depth]
        with open(out / "inspect.log", "w") as log:
            code = subprocess.call([*command, *flags, "--out", str(out)], stdout=log, stderr=subprocess.STDOUT)
        shutil.rmtree(out / "work", ignore_errors=True)  # staged copies and warps: regenerable, and large
        return f"{out.relative_to(bench)} exit={code}"

    print(f"{len(todo)} inspection(s) to run")
    with ThreadPoolExecutor(jobs) as pool:
        for line in pool.map(one, todo):
            print(line, flush=True)


# ---------------------------------------------------------------- scoring


def score(bench: Path) -> dict[str, Any]:
    """Per depth and task: sensitivity by defect, false alarms on the clean cases, verdicts, repeats, cost."""
    runs: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(bench.glob("runs/*/*/report.json")):
        depth, (task, rest) = path.parent.parent.name, path.parent.name.split(".", 1)
        case, _, repeat = rest.rpartition("@")
        report = Report.load(path)
        defects = json.loads((bench / "cases" / task / case / "defects.json").read_text())
        matched = match_defects(report, defects)
        errors = [o for o in report.observations if o.kind != "measured_property"]
        runs[(depth, task)].append(
            {
                "case": case,
                "repeat": repeat,
                "variant": case.rsplit("__", 1)[-1],
                "planted": sorted(defects),
                "found": matched["found_names"],
                "typed": sorted(n for n, d in defects.items() if any(o.defect == d["defect"] for o in errors)),
                "errors": len(errors),
                "serious": sum(o.severity in SERIOUS and o.confidence >= 0.5 for o in errors),
                "unmatched": matched["false_positives"],
                "verdict": report.conclusion.verdict if report.conclusion else "none",
                "by": report.conclusion.by if report.conclusion else "none",
                "cost": report.usage.cost_usd,
                "seconds": report.duration_s,
                "calls": len(report.tool_calls),
                "tokens": report.usage.input_tokens
                + report.usage.output_tokens
                + report.usage.cache_read_tokens
                + report.usage.cache_write_tokens,
                "second_reading": {o.id: (o.second_reading or {}).get("answer") for o in errors if o.second_reading},
            }
        )
    scores: dict[str, Any] = {}
    for (depth, task), rows in sorted(runs.items()):
        clean = [r for r in rows if not r["planted"]]
        planted = [r for r in rows if r["planted"]]
        by_defect: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
        for r in planted:
            for name in r["planted"]:
                by_defect[name][0] += name in r["found"]
                by_defect[name][1] += name in r["typed"]
                by_defect[name][2] += 1
        verdicts = defaultdict(list)
        for r in rows:
            verdicts[r["case"]].append(r["verdict"])
        repeated = [v for v in verdicts.values() if len(v) > 1]
        known = [r["cost"] for r in rows if r["cost"] is not None]
        scores[f"{depth}/{task}"] = {
            "depth": depth,
            "task": task,
            "inspections": len(rows),
            "sensitivity": {k: {"found": f, "typed": t, "planted": n} for k, (f, t, n) in sorted(by_defect.items())},
            "found": sum(len(r["found"]) for r in planted),
            "planted": sum(len(r["planted"]) for r in planted),
            "clean": len(clean),
            "clean_with_an_error": sum(r["errors"] > 0 for r in clean),
            "clean_with_a_serious_error": sum(r["serious"] > 0 for r in clean),
            "clean_acceptable": sum(r["verdict"] == "acceptable" for r in clean),
            "planted_defective": sum(r["verdict"] == "defective" for r in planted),
            "planted_inspections": len(planted),
            "unmatched_on_planted": sum(r["unmatched"] for r in planted),
            "concluded_by_scout": sum(r["by"] == "scout" for r in rows),
            "repeat_agreement": (sum(len(set(v)) == 1 for v in repeated), len(repeated)),
            "mean_cost_usd": round(mean(known), 4) if known else None,
            "mean_seconds": round(mean(r["seconds"] for r in rows), 1),
            "mean_calls": round(mean(r["calls"] for r in rows), 1),
            "mean_tokens": round(mean(r["tokens"] for r in rows)),
            "runs": rows,
        }
    (bench / "scores.json").write_text(json.dumps(scores, indent=2))
    (bench / "scores.md").write_text(markdown(scores))
    return scores


def markdown(scores: dict[str, Any]) -> str:
    def rate(a: int, b: int) -> str:
        return f"{a}/{b}" + (f" ({100 * a / b:.0f} %)" if b else "")

    lines = [
        "| depth | task | found | clean: no error | clean: acceptable | planted: defective | repeats agree | $ | s | calls | tokens |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in scores.values():
        agree = s["repeat_agreement"]
        lines.append(
            f"| {s['depth']} | {s['task']} | {rate(s['found'], s['planted'])} | "
            f"{rate(s['clean'] - s['clean_with_an_error'], s['clean'])} | {rate(s['clean_acceptable'], s['clean'])} | "
            f"{rate(s['planted_defective'], s['planted_inspections'])} | {rate(*agree) if agree[1] else '-'} | "
            f"{s['mean_cost_usd']} | {s['mean_seconds']} | {s['mean_calls']} | {s['mean_tokens']} |"
        )
    lines += ["", "| depth | task | defect | found | found, type named |", "|---|---|---|---|---|"]
    for s in scores.values():
        for name, c in s["sensitivity"].items():
            lines.append(
                f"| {s['depth']} | {s['task']} | {name} | {rate(c['found'], c['planted'])} | {rate(c['typed'], c['planted'])} |"
            )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("build")
    p.add_argument("manifest", type=Path, nargs="?")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--phantoms", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p = commands.add_parser("run")
    p.add_argument("bench", type=Path)
    p.add_argument("--depth", nargs="+", choices=DEPTHS, default=["standard"])
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--jobs", type=int, default=3)
    p = commands.add_parser("score")
    p.add_argument("bench", type=Path)
    args, flags = parser.parse_known_args(argv)  # run: the rest goes to `konfai-scout inspect` (--provider, ...)
    if args.command == "build":
        manifest = json.loads(args.manifest.read_text()) if args.manifest else {}
        print(f"{len(build(manifest, args.out, args.phantoms, args.seed))} case(s) under {args.out / 'cases'}")
    elif args.command == "run":
        run(args.bench, args.depth, args.repeats, args.jobs, flags)
    else:
        print(markdown(score(args.bench)))


if __name__ == "__main__":
    main()
