# SPDX-License-Identifier: Apache-2.0
"""``konfai-scout inspect | render | compare``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .agent import inspect, main_version
from .case import TASKS, CaseError, InspectionCase
from .config import MODES, PROVIDERS, Budget, ScoutConfig
from .providers import ProviderError
from .report import Report, compare, compare_markdown, render_markdown


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="konfai-scout",
        description="Visual inspection of segmentation, registration "
        "and synthesis results by a vision-language model, with KonfAI Eval measurements.",
    )
    parser.add_argument("--version", action="version", version=main_version())
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser("inspect", help="Inspect one case and write a report.")
    p.add_argument("--case", type=Path, help="A case JSON (task, name, inputs by role, labels, evaluation, notes).")
    p.add_argument("--task", choices=TASKS)
    p.add_argument("--name", default=None, help="The case's name (default: the first input's stem).")
    p.add_argument("--input", action="append", default=[], metavar="ROLE=PATH", help="An input by role (repeatable).")
    p.add_argument("--labels", default=None, metavar="ID=NAME,...", help="Label names, e.g. 1=liver,2=spleen.")
    p.add_argument(
        "--evaluation", action="append", default=[], type=Path, help="A KonfAI Eval Metric_*.json (repeatable)."
    )
    p.add_argument("--notes", default="", help="What the model should know: modality, dataset, the method inspected.")
    p.add_argument("--out", type=Path, required=True, help="The report directory.")
    p.add_argument("--config", type=Path, help="A ScoutConfig JSON; the flags below override it.")
    p.add_argument("--provider", choices=PROVIDERS)
    p.add_argument("--model")
    p.add_argument("--base-url", help="An OpenAI-compatible server (a local VLM).")
    p.add_argument("--api-key-env", help="The environment variable holding the key.")
    p.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"])
    p.add_argument("--mode", choices=MODES)
    p.add_argument("--max-tool-calls", type=int)
    p.add_argument("--max-images", type=int)
    p.add_argument("--max-seconds", type=float)
    p.add_argument("--max-usd", type=float)
    p.add_argument(
        "--allow-cloud-images", action="store_true", help="Authorise sending images of the case to a cloud API."
    )
    p.add_argument("--local-provider", action="store_true", help="The base-url is a server on this machine or network.")
    p.add_argument("--json", action="store_true", help="Print the report's summary as JSON.")

    p = commands.add_parser("render", help="Rewrite report.md from report.json.")
    p.add_argument("report", type=Path, help="A report directory or report.json.")

    p = commands.add_parser("compare", help="Put reports side by side (fixed views against active exploration).")
    p.add_argument("reports", type=Path, nargs="+")
    p.add_argument("--json", action="store_true")
    return parser


def _case_from_args(args: argparse.Namespace) -> InspectionCase:
    if args.case:
        case = InspectionCase.load(args.case)
        if args.notes:
            case.notes = args.notes
        return case
    if not args.task:
        raise CaseError("give --case, or --task with --input ROLE=PATH")
    inputs = {}
    for item in args.input:
        role, _, path = item.partition("=")
        if not path:
            raise CaseError(f"--input takes ROLE=PATH, got {item!r}")
        inputs[role] = Path(path)
    labels = {}
    for item in filter(None, (args.labels or "").split(",")):
        key, _, name = item.partition("=")
        labels[int(key)] = name or key
    name = args.name or (next(iter(inputs.values())).stem if inputs else "case")
    return InspectionCase(
        task=args.task,
        name=name,
        inputs=inputs,
        labels=labels,
        evaluation=list(args.evaluation) or None,
        notes=args.notes,
    )


def _config_from_args(args: argparse.Namespace) -> ScoutConfig:
    config = ScoutConfig.load(args.config) if args.config else ScoutConfig()
    for key in ("provider", "model", "base_url", "api_key_env", "effort", "mode"):
        value = getattr(args, key)
        if value is not None:
            setattr(config, key, value)
    if args.provider is not None and args.model is None and not args.config:
        config.model = None
    config.__post_init__()
    budget = config.budget
    for flag, key in (
        ("max_tool_calls", "max_tool_calls"),
        ("max_images", "max_images"),
        ("max_seconds", "max_seconds"),
        ("max_usd", "max_usd"),
    ):
        value = getattr(args, flag)
        if value is not None:
            setattr(budget, key, value)
    config.budget = Budget(**budget.to_dict())
    if args.allow_cloud_images:
        config.allow_cloud_images = True
    if args.local_provider:
        config.local_provider = True
    return config


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "inspect":
        try:
            case = _case_from_args(args)
            config = _config_from_args(args)
            report = inspect(case, config, args.out)
        except (CaseError, ProviderError, ValueError, OSError) as error:
            print(f"konfai-scout: {error}", file=sys.stderr)
            return 2
        summary = report.summary()
        if args.json:
            print(json.dumps(summary, indent=2))
        else:
            verdict = report.conclusion.verdict if report.conclusion else "none"
            print(
                f"{case.name}: {verdict}, {len(report.observations)} observation(s), {len(report.views)} view(s), "
                f"{len(report.measurements)} measurement(s) -> {args.out / 'report.md'}"
            )
            if report.error:
                print(f"error: {report.error}", file=sys.stderr)
        return 1 if report.error else 0
    if args.command == "render":
        report = Report.load(args.report)
        path = (args.report if args.report.is_dir() else args.report.parent) / "report.md"
        path.write_text(render_markdown(report), encoding="utf-8")
        print(path)
        return 0
    comparison = compare([Report.load(p) for p in args.reports])
    print(json.dumps(comparison, indent=2) if args.json else compare_markdown(comparison))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
