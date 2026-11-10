# SPDX-License-Identifier: Apache-2.0
"""Models that cost nothing: a scripted one for tests, and a heuristic explorer that reads the case card and the
tool results like an agent would, so examples and CI exercise the whole loop without an API."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any, ClassVar

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
        self._log(results, tools)
        return self._next()

    def _log(self, results: list[ToolResult], tools: list[ToolSpec] | None) -> None:
        if tools is not None:
            self.tools = tools
        self.messages += [
            {
                "role": "tool",
                "tool_call_id": r.call_id,
                "content": r.text,
                "images": len(r.images),
                "is_error": r.is_error,
            }
            for r in results
        ]

    def transcript(self) -> list[dict[str, Any]]:
        return list(self.messages)


def call(name: str, n: int, **arguments: Any) -> ToolCall:
    return ToolCall(id=f"call_{n}", name=name, arguments=arguments)


def turn(*calls: ToolCall, text: str = "") -> Turn:
    return Turn(text=text, tool_calls=list(calls), stop="tool_use" if calls else "end_turn", usage=Usage(10, 5))


class HeuristicProvider(ScriptedProvider):
    """A deterministic stand-in for a model, for tests and examples: it looks at the coverage and at every lead,
    measures the first lead, notes it as a major confirmed error of the lead's kind (Scout's rules downgrade what no
    measurement supports), and concludes from what entered the report. With no lead it has nothing to say."""

    #: The defect a lead of each kind points at.
    DEFECT: ClassVar[dict[str, str]] = {
        "edgeless_boundary": "boundary_error",
        "component": "spurious_structure",
        "false_positive": "spurious_structure",
        "false_negative": "under_segmentation",
        "folding": "folding",
        "edge_disagreement": "residual_misalignment",
        "error_cluster": "artifact",
    }

    def __init__(self, model: str = "heuristic") -> None:
        super().__init__([], model=model)
        self.task = ""
        self.card = ""
        self.step = 0
        self.n = 0
        self.coverage: list[str] = []
        self.leads: list[tuple[str, str, dict[str, Any]]] = []
        self.seen: list[str] = []
        self.measured: list[str] = []

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
        self.coverage = re.findall(r"^- (v\d+):", text, flags=re.MULTILINE)
        self.leads = [
            (lead, kind, json.loads(box))
            for lead, kind, box in re.findall(r"^- (l\d+) (\w+), box (\{.*?\}):", text, flags=re.MULTILINE)
        ]
        return self._plan([])

    def tool_results(self, results: list[ToolResult], tools: list[ToolSpec] | None = None) -> Turn:
        self._log(results, tools)
        parsed = []
        for result in results:
            try:
                parsed.append(json.loads(result.text))
            except json.JSONDecodeError:
                parsed.append({})
        self.seen += [r["view"]["id"] for r in parsed if "view" in r]
        self.measured += [r["measurement"]["id"] for r in parsed if "measurement" in r]
        return self._plan(parsed)

    def _plan(self, parsed: list[dict[str, Any]]) -> Turn:
        names = {t.name for t in self.tools}
        self.calls += 1
        self.step += 1
        if self.step == 1 and "montage" in names:
            return turn(self._call("montage", plane="axial"), text="Coverage first.")
        if self.step <= 2 and self.leads and "view" in names:
            mode = DEFAULT_MODE.get(self.task, "image")
            views = [
                self._call(
                    "view", plane="axial", mode=mode, position_mm=None, index=None, window=None, label=None, zoom=box
                )
                for _, _, box in self.leads
            ]
            return turn(*views, text="Looking at each lead.")
        if self.step <= 3 and self.leads and "measure" in names:
            return turn(self._call("measure", box_mm=self.leads[0][2], metrics=None), text="Measuring the first lead.")
        noted = any("observation" in r for r in parsed) or self.step > 5
        if self.leads and not noted and "note" in names:
            lead, kind, _ = self.leads[0]
            return turn(
                self._call(
                    "note",
                    kind="confirmed_error",
                    defect=self.DEFECT.get(kind, "other"),
                    severity="major",
                    confidence=0.8,
                    description=f"Lead {lead} ({kind}) looks wrong in the zoomed view.",
                    box_mm=lead,
                    plane="axial",
                    views=self.seen[1:2] or self.seen[:1] or self.coverage,
                    measurements=self.measured[:1],
                )
            )
        recorded = any("observation" in r for r in parsed)
        verdict = "defective" if recorded else "insufficient_evidence" if self.leads else "acceptable"
        return turn(
            self._call(
                "conclude",
                verdict=verdict,
                confidence=0.7,
                summary="Inspection done: "
                + (
                    "the first lead holds an error."
                    if recorded
                    else "nothing was recorded."
                    if self.leads
                    else "no lead stood out."
                ),
            )
        )
