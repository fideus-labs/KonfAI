# SPDX-License-Identifier: Apache-2.0
"""Models that cost nothing: a scripted one for tests, and a heuristic explorer that reads the case card and the
tool results like an agent would, so examples and CI exercise the whole loop without an API."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from ..views import DEFAULT_MODE
from .base import ImageContent, ToolCall, ToolResult, ToolSpec, Turn, Usage

Policy = Callable[[str, list[dict[str, Any]], list[ToolSpec]], Turn]


class ScriptedProvider:
    """Answers the turns it was given, in order; ``raise_at`` makes the n-th call raise the exception it holds."""

    name = "mock"
    remote = False

    def __init__(
        self, turns: list[Turn], model: str = "scripted", raise_at: dict[int, Exception] | None = None
    ) -> None:
        self.model = model
        self.turns = list(turns)
        self.raise_at = dict(raise_at or {})
        self.calls = 0
        self.messages: list[dict[str, Any]] = []
        self.tools: list[ToolSpec] = []

    def sdk_version(self) -> str:
        return "0"

    def start(self, system: str, tools: list[ToolSpec]) -> None:
        self.messages = [{"role": "system", "content": system}]
        self.tools = tools

    def _next(self) -> Turn:
        self.calls += 1
        if self.calls in self.raise_at:
            raise self.raise_at[self.calls]
        if not self.turns:
            return Turn(text="(script exhausted)", tool_calls=[], stop="end_turn", usage=Usage(1, 1))
        turn = self.turns.pop(0)
        self.messages.append(
            {"role": "assistant", "content": turn.text, "tool_calls": [c.__dict__ for c in turn.tool_calls]}
        )
        return turn

    def user(self, text: str, images: list[ImageContent] | None = None) -> Turn:
        self.messages.append({"role": "user", "content": text, "images": len(images or [])})
        return self._next()

    def tool_results(self, results: list[ToolResult], tools: list[ToolSpec] | None = None) -> Turn:
        if tools is not None:
            self.tools = tools
        for result in results:
            self.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": result.call_id,
                    "content": result.text,
                    "images": len(result.images),
                    "is_error": result.is_error,
                }
            )
        return self._next()

    def transcript(self) -> list[dict[str, Any]]:
        return list(self.messages)


def call(name: str, n: int, **arguments: Any) -> ToolCall:
    return ToolCall(id=f"call_{n}", name=name, arguments=arguments)


def turn(*calls: ToolCall, text: str = "") -> Turn:
    return Turn(text=text, tool_calls=list(calls), stop="tool_use" if calls else "end_turn", usage=Usage(10, 5))


class HeuristicProvider(ScriptedProvider):
    """Explores like a careful but simple agent: a coverage montage (when it may ask for views), the statistics of
    the whole case, a measurement where something looks off, a zoomed view of it, an observation, a conclusion.
    What "looks off" is read from the tool results: a label in several pieces or absent, a folded field, a large
    difference. Deterministic, so a study can compare runs."""

    def __init__(self, model: str = "heuristic") -> None:
        super().__init__([], model=model)
        self.task = ""
        self.card = ""
        self.step = 0
        self.suspect: dict[str, Any] | None = None
        self.view_id: str | None = None
        self.coverage_views: list[str] = []
        self.stats_asked = False
        self.results: list[dict[str, Any]] = []
        self.n = 0
        self.leads: list[tuple[str, dict[str, Any]]] = []
        self.leads_seen = False
        self.lead_views: list[str] = []

    def _tool_names(self) -> set[str]:
        return {t.name for t in self.tools}

    def _call(self, name: str, **arguments: Any) -> ToolCall:
        self.n += 1
        return call(name, self.n, **arguments)

    def user(self, text: str, images: list[ImageContent] | None = None) -> Turn:
        self.messages.append({"role": "user", "content": text, "images": len(images or [])})
        if not self.tools:  # asked for a plain answer (a second reading): a heuristic does not read images
            self.calls += 1
            return Turn(
                text="UNSURE\nA heuristic model does not read images.",
                tool_calls=[],
                stop="end_turn",
                usage=Usage(1, 1),
            )
        self.card = text
        match = re.search(r"Task: (\w+)", text)
        self.task = match.group(1) if match else "segmentation"
        self.coverage_views = re.findall(r"^- (v\d+):", text, flags=re.MULTILINE)
        self.leads = [
            (lead, json.loads(box))
            for lead, box in re.findall(r"^- (l\d+) \w+, box (\{.*?\}):", text, flags=re.MULTILINE)
        ]
        return self._plan([])

    def tool_results(self, results: list[ToolResult], tools: list[ToolSpec] | None = None) -> Turn:
        if tools is not None:
            self.tools = tools
        parsed = []
        for result in results:
            self.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": result.call_id,
                    "content": result.text,
                    "images": len(result.images),
                    "is_error": result.is_error,
                }
            )
            try:
                parsed.append(json.loads(result.text))
            except json.JSONDecodeError:
                parsed.append({"text": result.text, "error": result.is_error})
        self.results.extend(parsed)
        return self._plan(parsed)

    def _extent_box(self) -> dict[str, Any]:
        match = re.search(r"bounds_mm: (\{.*?\})\n", self.card)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        return {"min": [-1000, -1000, -1000], "max": [1000, 1000, 1000]}

    def _plan(self, parsed: list[dict[str, Any]]) -> Turn:
        names = self._tool_names()
        self.calls += 1
        self.step += 1
        if self.step == 1 and "montage" in names:
            return turn(self._call("montage", plane="axial"), text="Coverage first.")
        if self.leads and not self.leads_seen and "view" in names:  # then every lead, before anything else
            self.leads_seen = True
            return turn(
                *[
                    self._call(
                        "view",
                        plane="axial",
                        mode=DEFAULT_MODE.get(self.task, "image"),
                        position_mm=None,
                        index=None,
                        window=None,
                        label=None,
                        zoom=box,
                    )
                    for _, box in self.leads
                ],
                text="Looking at each lead.",
            )
        if self.leads_seen and not self.lead_views:
            self.lead_views = [r["view"]["id"] for r in parsed if "view" in r]
        asked = any("region" in r for r in self.results) or any(
            r.get("error") and "region_stats" in str(r.get("text", "")) for r in self.results
        )
        if self.suspect is None and "region_stats" in names and not asked and not self.stats_asked:
            self.stats_asked = True
            return turn(self._call("region_stats", box_mm=self._extent_box()), text="Statistics of the whole case.")
        if self.suspect is None:
            self.suspect = self._find_suspect()
            if self.suspect is None:
                return turn(
                    self._call(
                        "conclude",
                        verdict="insufficient_evidence",
                        confidence=0.3,
                        summary="Nothing stood out in the coverage views and the statistics; no reference "
                        "measurement was possible, so no error is confirmed.",
                    )
                )
            calls = []
            if "measure" in names:
                calls.append(self._call("measure", box_mm=self.suspect["box"]))
            if "view" in names:
                calls.append(
                    self._call(
                        "view",
                        plane="axial",
                        mode=self.suspect["mode"],
                        position_mm=None,
                        index=None,
                        window=None,
                        label=None,
                        zoom={"center": self.suspect["center"], "size": self.suspect["size"]},
                    )
                )
            if calls:
                return turn(*calls, text=f"Looking closer at {self.suspect['why']}.")
        if self.suspect is not None and "note" in names and not any("observation" in r for r in self.results):
            measurement_ids = [r["measurement"]["id"] for r in self.results if "measurement" in r]
            for r in parsed:
                if "view" in r:
                    self.view_id = r["view"]["id"]
            measured = self._measured_values()
            kind, severity = self.suspect["kind"], self.suspect["severity"]
            if measured is not None:
                kind = "confirmed_error" if measured["bad"] else "measured_property"
                severity = severity if measured["bad"] else "minor"
            return turn(
                self._call(
                    "note",
                    kind=kind,
                    defect=self.suspect["defect"] if kind != "measured_property" else "other",
                    severity=severity,
                    confidence=0.8 if kind == "confirmed_error" else 0.5,
                    description=self.suspect["why"] + (f"; {measured['text']}" if measured else ""),
                    box_mm=self.leads[0][0] if self.leads else self.suspect["box"],
                    plane="axial",
                    views=self.lead_views[:1] or [v for v in [self.view_id] if v] + self.coverage_views,
                    measurements=measurement_ids,
                )
            )
        kinds = [r["observation"]["kind"] for r in self.results if "observation" in r]
        if "confirmed_error" in kinds or (
            self.suspect and self.suspect["severity"] in ("major", "critical") and "suspected_error" in kinds
        ):
            verdict = "defective"
        elif kinds:
            verdict = "acceptable"
        else:
            verdict = "insufficient_evidence"
        return turn(
            self._call(
                "conclude",
                verdict=verdict,
                confidence=0.7,
                summary=f"Inspection done: {self.suspect['why'] if self.suspect else 'nothing stood out'}.",
            )
        )

    def _measured_values(self) -> dict[str, Any] | None:
        for r in self.results:
            values = (r.get("measurement") or {}).get("values")
            if not values:
                continue
            for key, value in values.items():
                if value is None:
                    continue
                if key.endswith("Dice") and value < 0.5:
                    return {"bad": True, "text": f"{key} = {value:.3f} in the region"}
                if "JacobianFolding" in key and value > 0:
                    return {"bad": True, "text": f"{key} = {value:.4f}: the field folds"}
                if key.endswith("MAE") and value > 100:
                    return {"bad": True, "text": f"{key} = {value:.1f} in the region"}
            first = next(((k, v) for k, v in values.items() if v is not None), None)
            if first:
                return {"bad": False, "text": f"{first[0]} = {first[1]:.4g}"}
        return None

    def _find_suspect(self) -> dict[str, Any] | None:
        region = next((r["region"] for r in self.results if "region" in r), None)
        if region is None:
            return None
        if self.task == "segmentation":
            labels = region.get("prediction_labels") or {}
            truth = region.get("truth_labels") or {}
            for label, entry in truth.items():
                if label not in labels:
                    return self._suspect(
                        entry["bbox_mm"],
                        "both",
                        "missing_structure",
                        "major",
                        "suspected_error",
                        f"label {label} is in the truth and not in the prediction",
                    )
            worst = max(labels.items(), key=lambda kv: kv[1].get("components", 1), default=None)
            if worst and worst[1].get("components", 1) > 1:
                return self._suspect(
                    worst[1]["bbox_mm"],
                    "prediction",
                    "fragmentation",
                    "moderate",
                    "suspected_error",
                    f"label {worst[0]} is in {worst[1]['components']} pieces",
                )
            if worst:
                return self._suspect(
                    worst[1]["bbox_mm"],
                    "prediction",
                    "boundary_error",
                    "minor",
                    "measured_property",
                    f"label {worst[0]} covers {worst[1]['volume_mm3']} mm3",
                )
        if self.task == "registration":
            field = region.get("displacement") or {}
            box = region.get("box_measured") or self._extent_box()
            return self._suspect(
                box,
                "jacobian",
                "implausible_deformation",
                "moderate",
                "suspected_error",
                f"displacements reach {field.get('max_mm', '?')} mm (p95 {field.get('p95_mm', '?')} mm)",
            )
        if self.task == "synthesis":
            stats = region.get("synthetic") or {}
            box = region.get("box_measured") or self._extent_box()
            return self._suspect(
                box,
                "difference",
                "intensity_bias",
                "moderate",
                "suspected_error",
                f"synthetic intensities span {stats.get('p1', '?')} to {stats.get('p99', '?')}",
            )
        return None

    @staticmethod
    def _suspect(box: dict[str, Any], mode: str, defect: str, severity: str, kind: str, why: str) -> dict[str, Any]:
        center = [(a + b) / 2 for a, b in zip(box["min"], box["max"], strict=True)]
        size = max(max(b - a for a, b in zip(box["min"], box["max"], strict=True)) * 1.5, 10.0)
        return {
            "box": box,
            "center": center,
            "size": size,
            "mode": mode,
            "defect": defect,
            "severity": severity,
            "kind": kind,
            "why": why,
        }
