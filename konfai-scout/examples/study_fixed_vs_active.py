# SPDX-License-Identifier: Apache-2.0
"""Fixed views against active exploration: the same case, the same model, both modes, side by side.

    python examples/study_fixed_vs_active.py                                      # heuristic model, no API
    python examples/study_fixed_vs_active.py --provider anthropic --allow-cloud-images --repeats 3

Every run leaves its full record (prompts, tool calls, views, measurements, tokens, cost, duration) under
examples/out/study/<task>/<mode>-<repeat>/, and the table printed at the end is `konfai-scout compare` on them.
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
    parser.add_argument("--allow-cloud-images", action="store_true")
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    reports = []
    for task, maker in MAKERS.items():
        if args.task not in ("all", task):
            continue
        case, _ = maker(HERE / "fixtures" / task)
        for mode in ("fixed", "active"):
            for repeat in range(args.repeats):
                config = ScoutConfig(
                    provider=args.provider,
                    model=args.model,
                    base_url=args.base_url,
                    mode=mode,
                    allow_cloud_images=args.allow_cloud_images,
                )
                reports.append(inspect(case, config, HERE / "out" / "study" / task / f"{mode}-{repeat}"))
    print(compare_markdown(compare(reports)))


if __name__ == "__main__":
    main()
