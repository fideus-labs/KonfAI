# SPDX-License-Identifier: Apache-2.0
"""Inspect a synthetic case of each task, with the provider you choose.

    python examples/inspect_synthetic.py                       # the free heuristic model, no API
    python examples/inspect_synthetic.py --provider anthropic --allow-cloud-images --task segmentation
    python examples/inspect_synthetic.py --provider openai --base-url http://localhost:11434/v1 --model llava

The cases are built under examples/fixtures/ (a sphere and a cube, with planted defects recorded in defects.json);
the reports land under examples/out/<task>/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from konfai_scout import ScoutConfig, inspect
from konfai_scout.config import Budget
from konfai_scout.synthetic import MAKERS

HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=[*list(MAKERS), "all"], default="all")
    parser.add_argument("--provider", default="mock")
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    parser.add_argument("--mode", default="active")
    parser.add_argument("--allow-cloud-images", action="store_true")
    parser.add_argument("--max-tool-calls", type=int, default=30)
    args = parser.parse_args()
    config = ScoutConfig(
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        mode=args.mode,
        allow_cloud_images=args.allow_cloud_images,
        budget=Budget(max_tool_calls=args.max_tool_calls),
    )
    for task, maker in MAKERS.items():
        if args.task not in ("all", task):
            continue
        case, defects = maker(HERE / "fixtures" / task)
        report = inspect(case, config, HERE / "out" / task)
        print(
            f"{task}: {report.conclusion.verdict} ({report.conclusion.by}), {len(report.observations)} observation(s), "
            f"{len(report.views)} view(s), {len(report.measurements)} measurement(s), "
            f"{report.usage.input_tokens + report.usage.output_tokens} tokens -> {HERE / 'out' / task / 'report.md'}"
        )
        print("  planted defects:", ", ".join(f"{name} ({entry['defect']})" for name, entry in defects.items()))
        for observation in report.observations:
            print(
                f"  found: {observation.kind} {observation.defect} "
                f"({observation.severity}, {observation.confidence:.2f})"
            )


if __name__ == "__main__":
    main()
