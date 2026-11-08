# SPDX-License-Identifier: Apache-2.0
"""Negative controls and planted defects: how often the model invents a defect where there is none, and how many of
the planted ones it finds.

    python examples/study_controls.py                                                  # heuristic model, no API
    python examples/study_controls.py --provider claude-code --allow-cloud-images --repeats 2 --second-reading

A clean case carries no defect (the prediction is the truth, the field is the exact one, the synthesis is the target
under fresh noise): every error the model records on it is invented, a false positive. A planted case carries the
defects of its `defects.json`: an error observation finds one when its box overlaps it. The model is told nothing of
which is which (the cases are named case1 and case2). The table is `konfai-scout compare` with its `planted`,
`found`, `false_positives` and `not_visible` columns; the last lines add them up. Every run leaves its full record
under examples/out/controls/<task>/<clean|planted>-<repeat>/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from konfai_scout import ScoutConfig, inspect
from konfai_scout.report import compare, compare_markdown
from konfai_scout.synthetic import MAKERS

HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=[*list(MAKERS), "all"], default="all")
    parser.add_argument("--provider", default="mock")
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    parser.add_argument("--mode", choices=["active", "fixed"], default="active")
    parser.add_argument("--allow-cloud-images", action="store_true")
    parser.add_argument("--second-reading", action="store_true")
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    reports, defects = [], []
    for task, maker in MAKERS.items():
        if args.task not in ("all", task):
            continue
        for n, (kind, clean) in enumerate((("clean", True), ("planted", False)), start=1):
            case, planted = maker(HERE / "fixtures" / "controls" / f"{task}_case{n}", clean=clean)
            for repeat in range(args.repeats):
                config = ScoutConfig(
                    provider=args.provider,
                    model=args.model,
                    base_url=args.base_url,
                    mode=args.mode,
                    allow_cloud_images=args.allow_cloud_images,
                    second_reading=args.second_reading,
                )
                reports.append(inspect(case, config, HERE / "out" / "controls" / task / f"{kind}-{repeat}"))
                defects.append(planted)
    comparison = compare(reports, defects)
    print(compare_markdown(comparison))
    clean_rows = [row for row, planted in zip(comparison["rows"], defects, strict=True) if not planted]
    planted_rows = [row for row, planted in zip(comparison["rows"], defects, strict=True) if planted]
    invented = sum(row["false_positives"] for row in clean_rows)
    print(
        f"\nClean cases: {len(clean_rows)} inspection(s), {invented} invented error(s)"
        + (f", {invented / len(clean_rows):.2f} per inspection" if clean_rows else "")
    )
    found, total = sum(row["found"] for row in planted_rows), sum(row["planted"] for row in planted_rows)
    print(
        f"Planted cases: {len(planted_rows)} inspection(s), {found} of {total} planted defects found, "
        f"{sum(row['false_positives'] for row in planted_rows)} error(s) away from every planted defect"
    )


if __name__ == "__main__":
    main()
