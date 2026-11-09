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

SCHEMA_VERSION = "1.0"
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
        )

    def save(self, out: Path) -> Path:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        # JSON has no NaN or infinity, which a prior evaluation or a measurement can hold: they are written as the
        # strings "NaN", "Infinity" and "-Infinity", so a confirmation resting on one stays readable as such
        finite = json.loads(json.dumps(self.to_dict(), default=_json_default), parse_constant=str)
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


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serialisable")


def _box_text(box: dict[str, list[float]] | None) -> str:
    if not box:
        return "whole case"
    lo, hi = box["min"], box["max"]
    center = [(a + b) / 2 for a, b in zip(lo, hi, strict=True)]
    size = [b - a for a, b in zip(lo, hi, strict=True)]
    return (
        f"centre ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}) mm, "
        f"size {size[0]:.1f} x {size[1]:.1f} x {size[2]:.1f} mm"
    )


def value_text(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def render_markdown(report: Report) -> str:
    """The report for a person: verdict first, then the observations with their evidence, then the record."""
    case = report.case
    lines = [f"# Scout report: {case.get('name', 'case')} ({case.get('task', '?')})", ""]
    conclusion = report.conclusion
    if conclusion:
        lines += [
            f"**Verdict: {conclusion.verdict}** (confidence {conclusion.confidence:.2f}, by {conclusion.by})",
            "",
            conclusion.summary,
            "",
        ]
        if conclusion.reason:
            lines += [f"_{conclusion.reason}_", ""]
    if report.error:
        lines += [f"**Error:** {report.error}", ""]
    provider = report.provider
    lines += [
        f"Provider {provider.get('name')} / model {provider.get('model')}, mode `{report.config.get('mode')}`, "
        f"{len(report.tool_calls)} tool calls, {len(report.views)} views ({report.usage.images_sent} sent), "
        f"{len(report.measurements)} measurements, {report.duration_s:.0f} s, "
        + (f"${report.usage.cost_usd:.4f}" if report.usage.cost_usd is not None else "cost unknown")
        + f", {report.usage.input_tokens + report.usage.output_tokens} tokens.",
        "",
    ]
    if not case.get("truth_available"):
        lines += [
            "No reference was given: nothing here is measured against a truth, and no anatomical claim is certain.",
            "",
        ]
    if report.budget.get("exhausted"):
        lines += [f"Budget exhausted: {report.budget['exhausted']}.", ""]
    second = report.second_reading
    if second and second.get("observations"):
        lines += [
            f"Second reading, blind, by {second.get('model')}: of {second['observations']} error observation(s), "
            f"{second.get('visible', 0)} visible, {second.get('not_visible', 0)} not visible, "
            f"{second.get('unsure', 0)} unsure"
            + (f", {second['unparsed']} unanswered" if second.get("unparsed") else "")
            + ".",
            "",
        ]
    lines += ["## Observations", ""]
    if not report.observations:
        lines += ["None recorded.", ""]
    views = {view.id: view for view in report.views}
    measurements = {m.id: m for m in report.measurements}
    for o in report.observations:
        lines += [
            f"### {o.id}: {o.defect} ({o.kind}, {o.severity}, confidence {o.confidence:.2f})",
            "",
            f"Location: {_box_text(o.box_mm)}" + (f", seen on the {o.plane} plane" if o.plane else ""),
            "",
            o.description,
            "",
        ]
        for note in o.notes:
            lines += [f"_{note}_", ""]
        if o.second_reading:
            lines += [
                f"_Second reading ({o.second_reading.get('answer')}, views "
                f"{', '.join(o.second_reading.get('views') or [])}): {o.second_reading.get('text', '')}_",
                "",
            ]
        for mid in o.measurements:
            m = measurements.get(mid)
            if m is None:
                continue
            kind = "against a reference" if m.reference else "no reference"
            values = ", ".join(f"{k} = {value_text(v)}" for k, v in m.values.items())
            lines += [f"- {mid} ({m.kind}, {kind}, {_box_text(m.box_measured or m.box_requested)}): {values}"]
        if o.measurements:
            lines.append("")
        for vid in o.views:
            v = views.get(vid)
            if v is None:
                continue
            lines += [f"![{vid}: {v.plane} {v.mode}]({v.path})", ""]
    lines += ["## Prior evaluation (KonfAI Eval, whole case)", ""]
    if report.prior_evaluation:
        lines += [f"- {k} = {value_text(v)}" for k, v in report.prior_evaluation.items()] + [""]
    else:
        lines += ["None given.", ""]
    lines += ["## Measurements", ""]
    for m in report.measurements:
        values = ", ".join(f"{k} = {value_text(v)}" for k, v in m.values.items()) or (m.error or "no value")
        lines += [
            f"- {m.id} ({m.kind}, {'reference' if m.reference else 'no reference'}, "
            f"{_box_text(m.box_measured or m.box_requested)}): {values}"
        ]
    lines += ["", "## Views examined", ""]
    for v in report.views:
        where = f"{v.plane}" + (
            f" at {v.position_mm[0]:.1f}, {v.position_mm[1]:.1f}, {v.position_mm[2]:.1f} mm" if v.position_mm else ""
        )
        lines += [
            f"- {v.id}: {v.kind} {where}, mode `{v.mode}`"
            + ("" if v.sent else " (rendered, not sent)")
            + f" — [{v.path}]({v.path})"
        ]
    lines += ["", "## Tool calls", ""]
    for c in report.tool_calls:
        status = "ok" if c.ok else f"error: {c.error}"
        lines += [
            f"{c.n}. `{c.tool}` {json.dumps(c.args, default=_json_default)} → {c.summary} "
            f"({status}, {c.duration_s:.1f} s)"
        ]
    lines += [
        "",
        "## Record",
        "",
        f"- Scout {report.scout_version}; " + ", ".join(f"{k} {v}" for k, v in report.versions.items()),
        f"- Created {report.created_at}",
        "- Inputs: "
        + ", ".join(
            f"{role} `{Path(str(entry.get('path', ''))).name}` ({str(entry.get('sha256', ''))[:12]})"
            for role, entry in (case.get("inputs") or {}).items()
        ),
        "",
    ]
    return "\n".join(lines)


def _overlap(a: dict[str, list[float]], b: dict[str, list[float]]) -> bool:
    return all(a["min"][i] < b["max"][i] and b["min"][i] < a["max"][i] for i in range(3))


def match_defects(report: Report, defects: dict[str, Any]) -> dict[str, Any]:
    """The planted defects of a synthetic case against the error observations: a defect is found when an error
    observation's box overlaps its box (or names it, for a defect planted without a box); an error observation that
    overlaps no planted defect is a false positive. On a clean case (no defect) every error observation is one."""
    errors = [o for o in report.observations if o.kind != "measured_property"]
    boxes = [p["box_mm"] for p in defects.values() if p.get("box_mm")]
    names = {p.get("defect") for p in defects.values() if not p.get("box_mm")}

    def hits(o: Observation, planted: dict[str, Any]) -> bool:
        if planted.get("box_mm"):
            return bool(o.box_mm) and _overlap(o.box_mm, planted["box_mm"])  # type: ignore[arg-type]
        return o.defect == planted.get("defect")

    found = [name for name, planted in defects.items() if any(hits(o, planted) for o in errors)]
    false_positives = [
        o.id for o in errors if not (o.box_mm and any(_overlap(o.box_mm, b) for b in boxes)) and o.defect not in names
    ]
    return {
        "planted": len(defects),
        "found": len(found),
        "found_names": found,
        "false_positives": len(false_positives),
        "false_positive_ids": false_positives,
    }


def defects_of(report: Report) -> dict[str, Any] | None:
    """The planted defects of a synthetic case: the ``defects.json`` beside its inputs, when there is one."""
    for entry in (report.case.get("inputs") or {}).values():
        path = Path(str(entry.get("path", ""))).parent / "defects.json"
        if path.is_file():
            return dict(json.loads(path.read_text(encoding="utf-8")))
    return None


def compare(reports: list[Report], defects: list[dict[str, Any] | None] | None = None) -> dict[str, Any]:
    """Side by side: what each inspection spent and found. For a study of fixed views against active exploration,
    and, with the planted defects of each case, of what was found and what was invented."""
    rows = []
    for n, report in enumerate(reports):
        counts = report.counts()
        planted = defects[n] if defects and n < len(defects) else None
        matched = match_defects(report, planted) if planted is not None else {}
        readings = [o.second_reading for o in report.observations if o.second_reading]
        rows.append(
            {
                "case": report.case.get("name"),
                "mode": report.config.get("mode"),
                "provider": report.provider.get("name"),
                "model": report.provider.get("model"),
                "verdict": report.conclusion.verdict if report.conclusion else None,
                "confidence": report.conclusion.confidence if report.conclusion else None,
                "tool_calls": len(report.tool_calls),
                "views": len(report.views),
                "images_sent": report.usage.images_sent,
                "measurements": len(report.measurements),
                "observations": len(report.observations),
                "confirmed": counts["confirmed_error"],
                "suspected": counts["suspected_error"],
                "measured": counts["measured_property"],
                "planted": matched.get("planted"),
                "found": matched.get("found"),
                "false_positives": matched.get("false_positives"),
                "not_visible": sum(r.get("answer") == "not_visible" for r in readings) if readings else None,
                "tokens": report.usage.input_tokens + report.usage.output_tokens,
                "cost_usd": report.usage.cost_usd,
                "duration_s": round(report.duration_s, 1),
                "budget_exhausted": report.budget.get("exhausted"),
                "error": report.error,
            }
        )
    return {"rows": rows}


def compare_markdown(comparison: dict[str, Any]) -> str:
    columns = [
        "case",
        "mode",
        "model",
        "verdict",
        "confidence",
        "tool_calls",
        "views",
        "images_sent",
        "measurements",
        "observations",
        "confirmed",
        "suspected",
        "measured",
        "planted",
        "found",
        "false_positives",
        "not_visible",
        "tokens",
        "cost_usd",
        "duration_s",
    ]
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in comparison["rows"]:
        lines.append("| " + " | ".join("" if row.get(c) is None else value_text(row[c]) for c in columns) + " |")
    return "\n".join(lines)
