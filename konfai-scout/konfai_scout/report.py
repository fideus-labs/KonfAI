# SPDX-License-Identifier: Apache-2.0
"""The report an inspection leaves: a versioned JSON record and a readable rendition of it.

Everything a later study needs is in the JSON: versions, configuration, prompts, every tool call, every view
rendered, every measurement, the observations with their evidence, the conclusion, cost and duration.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.1"
KINDS = ("measured_property", "suspected_error", "confirmed_error")
SEVERITIES = ("minor", "moderate", "major", "critical")
VERDICTS = ("acceptable", "defective", "insufficient_evidence")
DEFECTS: dict[str, tuple[str, ...]] = {
    "segmentation": (
        "missing_structure",
        "spurious_structure",
        "under_segmentation",
        "over_segmentation",
        "leak",
        "boundary_error",
        "fragmentation",
        "holes",
        "label_confusion",
        "other",
    ),
    "registration": (
        "residual_misalignment",
        "folding",
        "implausible_deformation",
        "over_smoothing",
        "boundary_artifact",
        "missing_coverage",
        "intensity_artifact",
        "other",
    ),
    "synthesis": (
        "intensity_bias",
        "hallucinated_structure",
        "missing_structure",
        "blur",
        "artifact",
        "geometry_error",
        "other",
    ),
}


@dataclass
class View:
    id: str
    kind: str  # slice, montage
    plane: str
    array_axis: int
    index: int | None
    position_mm: list[float] | None
    box_mm: dict[str, list[float]]
    mode: str
    window: list[float] | None
    size_px: list[int]
    #: ``world = origin + col * u + row * v`` for a slice; a montage lists one per tile under ``extra["tiles"]``.
    px_to_mm: dict[str, list[float]] | None
    path: str
    sha256: str
    sent: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Measurement:
    id: str
    kind: str  # konfai_eval, scout
    reference: bool  # measured against a truth
    box_requested: dict[str, list[float]] | None
    box_measured: dict[str, list[float]] | None
    values: dict[str, Any]
    request: dict[str, Any] = field(default_factory=dict)
    workspace: str | None = None
    duration_s: float = 0.0
    error: str | None = None


@dataclass
class Observation:
    id: str
    kind: str
    defect: str
    severity: str
    confidence: float
    description: str
    box_mm: dict[str, list[float]] | None
    plane: str | None = None
    views: list[str] = field(default_factory=list)
    measurements: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    #: The blind second reading, when asked: ``{"answer": visible | not_visible | unsure | unparsed, "text", "views"}``.
    second_reading: dict[str, Any] | None = None


@dataclass
class ToolCall:
    n: int
    tool: str
    args: dict[str, Any]
    started_at: float
    duration_s: float
    ok: bool
    summary: str
    views: list[str] = field(default_factory=list)
    measurements: list[str] = field(default_factory=list)
    observation: str | None = None
    error: str | None = None


@dataclass
class Usage:
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    images_sent: int = 0
    cost_usd: float | None = None
    cost_known: bool = False


@dataclass
class Conclusion:
    verdict: str
    summary: str
    confidence: float
    by: str = "model"  # model, scout (when Scout had to conclude itself)
    reason: str | None = None


@dataclass
class Report:
    schema_version: str
    scout_version: str
    versions: dict[str, str]
    created_at: str
    duration_s: float
    case: dict[str, Any]
    config: dict[str, Any]
    provider: dict[str, Any]
    prompts: dict[str, str]
    prior_evaluation: dict[str, Any]
    tool_calls: list[ToolCall]
    views: list[View]
    measurements: list[Measurement]
    observations: list[Observation]
    conclusion: Conclusion | None
    usage: Usage
    budget: dict[str, Any]
    error: str | None = None
    transcript_path: str | None = None
    #: The blind second reading of the error observations, when asked: model, counts by answer, transcript path.
    second_reading: dict[str, Any] | None = None
    #: The survey's leads (id, kind, box, reason) and the slice views that looked at each.
    leads: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> Report:
        return Report(
            schema_version=str(data.get("schema_version", "")),
            scout_version=str(data.get("scout_version", "")),
            versions=dict(data.get("versions") or {}),
            created_at=str(data.get("created_at", "")),
            duration_s=float(data.get("duration_s", 0.0)),
            case=dict(data.get("case") or {}),
            config=dict(data.get("config") or {}),
            provider=dict(data.get("provider") or {}),
            prompts=dict(data.get("prompts") or {}),
            prior_evaluation=dict(data.get("prior_evaluation") or {}),
            tool_calls=[ToolCall(**c) for c in data.get("tool_calls") or []],
            views=[View(**v) for v in data.get("views") or []],
            measurements=[Measurement(**m) for m in data.get("measurements") or []],
            observations=[Observation(**o) for o in data.get("observations") or []],
            conclusion=Conclusion(**data["conclusion"]) if data.get("conclusion") else None,
            usage=Usage(**(data.get("usage") or {})),
            budget=dict(data.get("budget") or {}),
            error=data.get("error"),
            transcript_path=data.get("transcript_path"),
            second_reading=data.get("second_reading"),
            leads=list(data.get("leads") or []),
        )

    def save(self, out: Path) -> Path:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        # JSON has no NaN or infinity, which a prior evaluation or a measurement can hold: they are written as the
        # strings "NaN", "Infinity" and "-Infinity", so a confirmation resting on one stays readable as such
        finite = json.loads(json.dumps(self.to_dict(), default=json_default), parse_constant=str)
        (out / "report.json").write_text(json.dumps(finite, indent=2, allow_nan=False), encoding="utf-8")
        (out / "report.md").write_text(render_markdown(self), encoding="utf-8")
        return out / "report.json"

    @staticmethod
    def load(path: Path | str) -> Report:
        path = Path(path)
        if path.is_dir():
            path = path / "report.json"
        return Report.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def counts(self) -> dict[str, int]:
        out = dict.fromkeys(KINDS, 0) | dict.fromkeys(SEVERITIES, 0)
        for observation in self.observations:
            out[observation.kind] = out.get(observation.kind, 0) + 1
            out[observation.severity] = out.get(observation.severity, 0) + 1
        return out

    def summary(self) -> dict[str, Any]:
        """The short form integrations carry around: no images, no transcript."""
        return {
            "case": self.case.get("name"),
            "task": self.case.get("task"),
            "mode": self.config.get("mode"),
            "provider": self.provider.get("name"),
            "model": self.provider.get("model"),
            "verdict": self.conclusion.verdict if self.conclusion else None,
            "conclusion": self.conclusion.summary if self.conclusion else None,
            "confidence": self.conclusion.confidence if self.conclusion else None,
            "observations": [
                {
                    "id": o.id,
                    "kind": o.kind,
                    "defect": o.defect,
                    "severity": o.severity,
                    "confidence": o.confidence,
                    "box_mm": o.box_mm,
                    "description": o.description,
                    "second_reading": (o.second_reading or {}).get("answer"),
                }
                for o in self.observations
            ],
            "counts": self.counts(),
            "second_reading": self.second_reading,
            "tool_calls": len(self.tool_calls),
            "views": len(self.views),
            "measurements": len(self.measurements),
            "usage": asdict(self.usage),
            "duration_s": round(self.duration_s, 1),
            "budget_exhausted": self.budget.get("exhausted"),
            "error": self.error,
        }


def json_default(value: object) -> object:
    """numpy scalars and arrays as numbers and lists, anything else (a path) as its text."""
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def _box_text(box: dict[str, list[float]] | None) -> str:
    if not box:
        return "whole case"
    lo, hi = box["min"], box["max"]
    center = ", ".join(f"{(a + b) / 2:.0f}" for a, b in zip(lo, hi, strict=True))
    size = " x ".join(f"{b - a:.0f}" for a, b in zip(lo, hi, strict=True))
    return f"({center}) mm, {size} mm"


def value_text(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _findings_table(report: Report) -> list[str]:
    """The errors at a glance, built by Scout from the record, worst first: the same report reads the same way."""
    rank = {s: n for n, s in enumerate(reversed(SEVERITIES))}
    errors = sorted(
        (o for o in report.observations if o.kind != "measured_property"),
        key=lambda o: (rank[o.severity], -o.confidence, o.id),
    )
    if not errors:
        return ["No error was recorded.", ""]
    lines = [
        "| Finding | Defect | Kind | Severity | Confidence | Where | Second reading |",
        "|---|---|---|---|---|---|---|",
    ]
    for o in errors:
        kind = "confirmed" if o.kind == "confirmed_error" else "suspected"
        reading = (o.second_reading or {}).get("answer", "")
        lines.append(
            f"| {o.id} | {o.defect} | {kind} | {o.severity} | {o.confidence:.2f} | {_box_text(o.box_mm)} | {reading} |"
        )
    return [*lines, ""]


def short_key(key: str) -> str:
    """A KonfAI result key without its groups (`WARPED_LABELS:FIXED_LABELS:Dice:1` is `Dice:1`)."""
    parts = key.split(":")
    return ":".join(parts[2:]) if len(parts) > 2 else key


def _values_text(values: dict[str, Any]) -> str:
    return ", ".join(f"{short_key(k)} {value_text(v)}" for k, v in values.items()) or "no value"


def render_markdown(report: Report) -> str:
    """The report for a person: the verdict, the findings at a glance, each observation with the images it cites,
    then the measurements cited. Everything else (every view, measurement and call) is in report.json."""
    case = report.case
    conclusion = report.conclusion
    lines = [f"# Scout report: {case.get('name', 'case')} ({case.get('task', '?')})", ""]
    if conclusion:
        lines += [
            f"**Verdict: {conclusion.verdict}** (confidence {conclusion.confidence:.2f}, by {conclusion.by})",
            "",
            conclusion.summary,
            "",
        ]
    lines += _findings_table(report)
    caveats = [conclusion.reason] if conclusion and conclusion.reason else []
    caveats += [f"Error: {report.error}"] if report.error else []
    if not case.get("truth_available"):
        caveats.append("No reference given: nothing is measured against a truth.")
    if report.budget.get("exhausted"):
        caveats.append(f"Budget exhausted: {report.budget['exhausted']}.")
    if report.budget.get("not_enforced"):
        caveats.append(f"Not enforced: {report.budget['not_enforced']}.")
    second = report.second_reading
    if second and second.get("observations"):
        caveats.append(
            f"Blind second reading ({second.get('model')}): {second.get('visible', 0)} visible, "
            f"{second.get('not_visible', 0)} not visible, {second.get('unsure', 0)} unsure of "
            f"{second['observations']}."
        )
    lines += [f"_{c}_" for c in caveats] + ([""] if caveats else [])
    lines += ["## Observations", ""]
    if not report.observations:
        lines += ["None recorded.", ""]
    views = {view.id: view for view in report.views}
    for o in report.observations:
        lines += [
            f"### {o.id}: {o.defect} ({o.kind}, {o.severity}, confidence {o.confidence:.2f})",
            "",
            f"Location: {_box_text(o.box_mm)}" + (f", seen on the {o.plane} plane" if o.plane else ""),
            "",
            o.description,
            "",
        ]
        lines += [line for note in o.notes for line in (f"_{note}_", "")]
        if o.second_reading:
            lines += [
                f"_Second reading ({o.second_reading.get('answer')}, views "
                f"{', '.join(o.second_reading.get('views') or [])}): {o.second_reading.get('text', '')}_",
                "",
            ]
        if o.measurements:  # their values are under Evidence: the description quotes the ones that matter
            lines += [f"Measurements: {', '.join(o.measurements)}.", ""]
        lines += [
            line for vid in o.views if (v := views.get(vid)) for line in (f"![{vid}: {v.plane} {v.mode}]({v.path})", "")
        ]
    if report.leads:
        lines += ["## Leads", ""]
        lines += [
            f"- {lead['id']} {lead['kind']}: {lead['reason']}. "
            + (f"Looked at in {', '.join(lead['views'])}." if lead.get("views") else "Not looked at.")
            for lead in report.leads
        ]
        lines.append("")
    cited = {m for o in report.observations for m in o.measurements}
    evidence: list[str] = []
    seen: dict[str, str] = {}  # Scout's check of a box the model measured already holds the same values
    for m in report.measurements:
        if m.id in cited:
            same = json.dumps([m.box_measured, m.values], sort_keys=True, default=str)  # exact, not as displayed
            text = f"{_box_text(m.box_measured or m.box_requested)}: {_values_text(m.values)}"
            evidence.append(f"- {m.id} " + (f"same as {seen[same]}" if same in seen else text))
            seen.setdefault(same, m.id)
    if report.prior_evaluation:
        evidence.append(f"- prior evaluation (whole case): {_values_text(report.prior_evaluation)}")
    if evidence:
        lines += ["## Evidence", "", *evidence, ""]
    provider = report.provider
    lines += [
        f"{provider.get('name')} / {provider.get('model')}, {report.config.get('mode')}, "
        f"{len(report.tool_calls)} calls, {len(report.views)} views ({report.usage.images_sent} sent), "
        f"{len(report.measurements)} measurements, {report.duration_s:.0f} s, "
        + (f"${report.usage.cost_usd:.3f}" if report.usage.cost_usd is not None else "cost unknown")
        + f". Scout {report.scout_version}, {report.created_at}. Every view, measurement and call: report.json.",
        "",
    ]
    return "\n".join(lines)


#: What an export may show: everything, what is wrong where, or how much is wrong.
LEVELS = ("full", "regions", "counts")


def export(report: Report, level: str) -> dict[str, Any]:
    """What may be shown of an inspection at a confidentiality level, for a reader that must not identify the case.

    ``full``: the report as is. ``regions``: the verdict and every observation with its kind, defect, severity, box
    and the values of its measurements; no case name, path, image, description, transcript or prior evaluation.
    ``counts``: the verdict and the number of observations by kind, defect and severity."""
    if level not in LEVELS:
        raise ValueError(f"level is one of {LEVELS}, not {level!r}")
    if level == "full":
        return report.to_dict()
    verdict = report.conclusion.verdict if report.conclusion else None
    if level == "counts":
        counts: dict[str, dict[str, dict[str, int]]] = {}
        for o in report.observations:
            severities = counts.setdefault(o.kind, {}).setdefault(o.defect, {})
            severities[o.severity] = severities.get(o.severity, 0) + 1
        return {"level": level, "verdict": verdict, "counts": counts}
    measurements = {m.id: m for m in report.measurements}
    return {
        "level": level,
        "task": report.case.get("task"),
        "verdict": verdict,
        "observations": [
            {
                "id": o.id,
                "kind": o.kind,
                "defect": o.defect,
                "severity": o.severity,
                "confidence": o.confidence,
                "box_mm": o.box_mm,
                "plane": o.plane,
                "measurements": [
                    {
                        "id": m.id,
                        "reference": m.reference,
                        "box_measured": m.box_measured,
                        # numbers only: a statistic may carry text (a label's name), and text may name anything
                        "values": {k: v for k, v in m.values.items() if not isinstance(v, str)},
                    }
                    for m in (measurements.get(mid) for mid in o.measurements)
                    if m is not None
                ],
                "second_reading": (o.second_reading or {}).get("answer"),
            }
            for o in report.observations
        ],
    }


def _overlap(a: dict[str, list[float]], b: dict[str, list[float]]) -> bool:
    return all(a["min"][i] < b["max"][i] and b["min"][i] < a["max"][i] for i in range(3))
