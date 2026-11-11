# SPDX-License-Identifier: Apache-2.0
"""The inspection: coverage views, then the model's own exploration under a budget, then the report.

Two modes for the same case and the same model: ``fixed`` hands over a fixed set of views and lets the model note
and conclude (measurements allowed, no further views); ``active`` starts from the same coverage and lets the model
ask for views and zoom. Everything either mode does is recorded the same way, so the two compare."""

from __future__ import annotations

import datetime as dt
import json
import platform
import random
import re
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .case import InspectionCase
from .config import ScoutConfig
from .evaluation import RegionEvaluator, load_prior
from .measure import label_stats
from .planes import (
    box_around,
    box_center,
    box_from_dict,
    box_intersection,
    box_size,
    describe_grid,
    native_plane,
    plane_frame,
)
from .prompts import ARBITER, PROMPT_VERSION, case_card, second_reader_prompt, second_reading_card, system_prompt
from .providers import ImageContent, Provider, ProviderError, ToolCall, ToolResult, Turn, check_privacy, make_provider
from .providers import Usage as TokenUsage
from .providers.base import package_version
from .providers.pricing import cost_usd, price_of
from .report import SCHEMA_VERSION, Conclusion, Observation, Report, Usage
from .survey import Lead, SurveySkipped, leads_text, survey
from .tools import Toolbox
from .views import DEFAULT_MODE, Renderer, ViewError

COVERAGE_EXTRA_MODES = {
    "segmentation": ["both"],
    "registration": ["field_magnitude", "jacobian", "overlay"],
    "synthesis": ["difference", "side_by_side"],
}
#: What the blind second reader sees of a region: the image, the result over it, a comparison when one exists.
SECOND_READING_MODES = {
    "segmentation": ["image", "prediction", "both"],
    "registration": ["fixed", "warped", "checkerboard"],
    "synthesis": ["synthetic", "target", "difference", "source"],
}
#: The calls that record rather than explore: outside the call budget, and still run once it is spent.
RECORDING = ("note", "conclude")
NOTICE_BUDGET = (
    "Budget exhausted ({exhausted}). Call `note` for anything established and not yet recorded, then `conclude` now."
)
NOTICE_SILENT = (
    "Nothing was recorded by this message. Use the tools: `note` for each finding, then `conclude` to end; keep "
    "exploring with `view`/`measure` if you are not done."
)


def scout_version() -> str:
    return package_version("konfai-scout")


def _versions(provider: Provider) -> dict[str, str]:
    out = {"python": platform.python_version(), "platform": platform.platform()}
    out |= {name: package_version(name) for name in ("numpy", "SimpleITK", "Pillow", "scipy", "konfai")}
    out[f"{provider.name}_sdk"] = provider.sdk_version()
    out["prompt"] = PROMPT_VERSION
    return out


@dataclass
class BudgetState:
    config: ScoutConfig
    started: float
    calls: int = 0
    images: int = 0
    tokens: int = 0
    usd: float = 0.0
    #: The cost of the request in flight, as a provider that runs the tool loop reports it message by message.
    running_usd: float = 0.0

    def exhausted(self) -> str | None:
        budget = self.config.budget
        if self.calls >= budget.max_tool_calls:
            return f"{self.calls} tool calls (limit {budget.max_tool_calls})"
        if self.images >= budget.max_images:
            return f"{self.images} images (limit {budget.max_images})"
        if budget.max_seconds is not None and time.time() - self.started >= budget.max_seconds:
            return f"{time.time() - self.started:.0f} s (limit {budget.max_seconds:.0f} s)"
        if budget.max_usd is not None and self.usd + self.running_usd >= budget.max_usd:
            return f"${self.usd + self.running_usd:.3f} (limit ${budget.max_usd:.2f})"
        return None


class Inspector:
    def __init__(
        self,
        case: InspectionCase,
        config: ScoutConfig,
        out: Path,
        provider: Provider | None = None,
        reader: Callable[[], Provider] | None = None,
    ) -> None:
        self.case = case
        self.config = config
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        if provider is None:
            check_privacy(config)
        self.provider = provider or make_provider(config)
        #: A fresh session per blind second reading: the same model by default, never the inspecting session.
        self.reader = reader or (lambda: make_provider(config))
        self.second_reading: dict[str, Any] | None = None
        self.lead_views: dict[str, list[str]] = {}
        case.stage(
            self.out / "work" / "stage"
        )  # the case as KonfAI reads it: every read, warp and measure goes through it
        self.renderer = Renderer(case, self.out / "work", config, self.out / "evidence")
        self.evaluator = RegionEvaluator(case, config, self.out / "work", self.renderer.derived)
        self.toolbox = Toolbox(case, config, self.renderer, self.evaluator)
        self.toolbox.closing = lambda: self.budget.exhausted() is not None
        self.usage = Usage()
        self.price = price_of(config.model or "", config.prices)
        self.budget = BudgetState(config, time.time())
        self.error: str | None = None
        self.started_at = dt.datetime.now(dt.UTC)

    # ------------------------------------------------------------------ the loop

    def run(self) -> Report:
        self.case.save(self.out / "case.json")
        exploration = self.config.mode == "active"
        prior = load_prior(self.case.evaluation, self.case.name)
        system = system_prompt(self.case, self.config)
        tools = self.toolbox.specs(exploration)
        self.provider.start(system, tools)
        images, views_text = self._coverage()
        self.toolbox.leads, survey_text = self._survey(exploration)
        if self._arbitrate(self.toolbox.leads):
            survey_text = leads_text(self.toolbox.leads, exploration)
        card = case_card(
            self.case,
            prior,
            self._labels_summary(),
            f"{views_text}\n\n{survey_text}",
            self.config.blind,
        )
        try:
            if getattr(self.provider, "drives_tools", False):
                self._run_driven(card, images)
            else:
                self._run_loop(card, images, exploration)
        except ProviderError as error:
            self.error = f"provider error: {error}"
        except Exception as error:  # the report must still be written
            self.error = f"{type(error).__name__}: {error}"
        # what the inspector looked at, before the second reader renders its own crops
        self.lead_views = {lead.id: self.toolbox.lead_views(lead) for lead in self.toolbox.leads}
        if self.config.second_reading and self.error is None:
            self._second_reading()
        return self._finish()

    def _start(self, card: str, images: list[ImageContent]) -> Turn:
        turn = self._request(lambda: self.provider.user(card, images))
        self.budget.images += len(images)
        self.toolbox.images_sent += len(images)
        return turn

    def _run_loop(self, card: str, images: list[ImageContent], exploration: bool) -> None:
        """Scout drives the loop: each call the model makes runs through ``_execute`` and its result goes back. Once
        the budget is spent the model gets one last turn with `note` and `conclude` only."""
        turn = self._start(card, images)
        reminders, final = 0, False
        while self.toolbox.conclusion is None and turn.stop != "refusal":
            if turn.tool_calls:
                results = [self._execute(call) for call in turn.tool_calls]
                exhausted = self.budget.exhausted()
                if self.toolbox.conclusion is not None or (exhausted and final):
                    break
                final = bool(exhausted)
                specs = self.toolbox.specs(exploration, final=True) if final else None
                turn = self._request(lambda r=results, s=specs: self.provider.tool_results(r, s))
                continue
            if final or reminders >= 2:
                break
            reminders += 1
            exhausted = self.budget.exhausted()
            final = bool(exhausted)
            notice = NOTICE_BUDGET.format(exhausted=exhausted) if exhausted else NOTICE_SILENT
            turn = self._request(lambda n=notice: self.provider.user(n, None))
        if turn.stop == "refusal":
            self.error = f"the model refused: {turn.text}"

    def _run_driven(self, card: str, images: list[ImageContent]) -> None:
        """The provider runs the tool loop itself (the Claude Agent SDK, Codex): every call it makes lands in
        ``_execute``, so the record and the budget are the same; Scout only speaks between its turns."""
        self.provider.execute = self._execute  # type: ignore[attr-defined]
        self.provider.on_usage = self._running  # type: ignore[attr-defined]
        turn = self._start(card, images)
        reminders = 0
        while self.toolbox.conclusion is None and turn.stop != "refusal" and reminders < 2:
            reminders += 1
            exhausted = self.budget.exhausted()
            notice = NOTICE_BUDGET.format(exhausted=exhausted) if exhausted else NOTICE_SILENT
            turn = self._request(lambda n=notice: self.provider.user(n, None))
        if turn.stop == "refusal":
            self.error = f"the model refused: {turn.text}"

    def _execute(self, call: ToolCall) -> ToolResult:
        """One tool call under the budget: refused once the budget is spent, except a note or the conclusion."""
        if call.name in RECORDING:  # a note or the conclusion: never counted, never refused
            return self.toolbox.call(call)
        exhausted = self.budget.exhausted()
        if exhausted:
            return self._refused(call, exhausted)
        result = self.toolbox.call(call)
        self.budget.calls += 1
        self.budget.images += len(result.images)
        return result

    def _running(self, usage: TokenUsage) -> None:
        """One message's usage while the provider's request is in flight: what ``max_usd`` stops on mid-request."""
        self.budget.running_usd += cost_usd(usage, self.price) or 0.0

    def _request(self, call) -> Turn:
        attempts = 0
        while True:
            try:
                turn = call()
                break
            except ProviderError as error:
                attempts += 1
                if not error.retryable or attempts > self.config.provider_retries:
                    raise
                time.sleep(self.config.retry_wait_seconds * attempts)
        if self.price is None and turn.model:  # a provider that names its model only once it answered
            self.price = price_of(turn.model, self.config.prices)
        self.budget.running_usd = 0.0  # the turn's own usage replaces the running estimate
        self.usage.requests += 1
        self.usage.input_tokens += turn.usage.input_tokens
        self.usage.output_tokens += turn.usage.output_tokens
        self.usage.cache_read_tokens += turn.usage.cache_read_tokens
        self.usage.cache_write_tokens += turn.usage.cache_write_tokens
        self.budget.tokens = self.usage.input_tokens + self.usage.output_tokens
        cost = cost_usd(turn.usage, self.price)
        if cost is not None:
            self.budget.usd += cost
        return turn

    @staticmethod
    def _refused(call, exhausted: str) -> ToolResult:
        return ToolResult(
            call_id=call.id,
            is_error=True,
            text=f"Budget exhausted ({exhausted}): this call was not run. Only `note` and `conclude` remain: "
            "record what is established and conclude.",
        )

    # ------------------------------------------------------------------ the blind second reading

    def _second_reading(self) -> None:
        """A fresh session re-reads each error observation blind: crops of its box, the claim, no method, no history,
        no measurement. Its answer is recorded on the observation; nothing is removed or re-judged."""
        errors = [o for o in self.toolbox.observations if o.kind != "measured_property" and o.box_mm]
        counts = dict.fromkeys(("visible", "not_visible", "unsure", "unparsed"), 0)
        transcripts: list[dict[str, Any]] = []
        model: str | None = None
        for observation in errors:
            try:
                images, ids, lines = self._crops(observation)
            except ViewError as error:
                observation.second_reading = {"answer": "unparsed", "text": f"no crop: {error}", "views": []}
                counts["unparsed"] += 1
                continue
            self.toolbox.images_sent += len(images)
            reader = self.reader()
            reader.start(second_reader_prompt(self.case), [])
            card = second_reading_card(observation, lines)
            try:
                turn = self._request(lambda r=reader, c=card, i=images: r.user(c, i))
                answer, text = parse_second_reading(turn.text)
                model = turn.model or reader.model
            except ProviderError as error:
                answer, text = "unparsed", f"provider error: {error}"
            observation.second_reading = {"answer": answer, "text": text, "views": ids}
            counts[answer] += 1
            transcripts.append({"observation": observation.id, "transcript": reader.transcript()})
        (self.out / "second_reading.json").write_text(json.dumps(transcripts, indent=1, default=str), encoding="utf-8")
        self.second_reading = {
            "provider": self.provider.name,
            "model": model,
            "observations": len(errors),
            **counts,
            "transcript_path": "second_reading.json",
        }

    def _crops(self, observation: Observation) -> tuple[list[ImageContent], list[str], list[str]]:
        """The region of an observation, twice its box and at least 64 mm (a small region keeps its surroundings:
        a few voxels alone are blocks of colour), in the second reader's modes, on the slice the first reader cited
        (the box's centre slice may miss what a large box holds elsewhere), else in the plane the volume was acquired
        in."""
        box = box_from_dict(observation.box_mm)
        spacing = np.asarray(self.case.grid.spacing_xyz, dtype=np.float64)
        zoom = box_around(box_center(box), np.maximum(box_size(box) * 2.0, np.maximum(12.0 * spacing, 64.0)))
        plane = observation.plane or native_plane(self.case.grid)
        known = {v.id: v for v in self.renderer.views}
        cited = [
            known[v]
            for v in observation.views
            if v in known and known[v].kind == "slice" and box_intersection(box_from_dict(known[v].box_mm), box)
        ]
        index = None
        if cited:
            plane, index = cited[0].plane, cited[0].index
        modes = [m for m in SECOND_READING_MODES[self.case.task] if m in self.renderer.modes()][:3]
        if self.case.task == "registration" and observation.defect in ("folding", "implausible_deformation"):
            modes = [*modes[:2], "jacobian"]
        images: list[ImageContent] = []
        ids: list[str] = []
        lines: list[str] = []
        for mode in modes:
            png, view = self.renderer.render(plane, mode, index=index, zoom=zoom)
            images.append(ImageContent(png, f"[{view.id}]"))
            ids.append(view.id)
            legend = view.extra.get("legend")
            lines.append(
                f"- {view.id}: {plane} slice through the region, mode {mode}" + (f" ({legend})" if legend else "")
            )
        return images, ids, lines

    # ------------------------------------------------------------------ coverage

    def _coverage(self) -> tuple[list[ImageContent], str]:
        """The views both modes start from: one montage per plane in the task's default mode; the fixed mode adds
        the central slice of each plane in the secondary modes."""
        images: list[ImageContent] = []
        lines = ["Views attached (ids as printed on each image):"]
        planes = ["axial", "coronal", "sagittal"] if self.case.volume(self.case.image_role).rank == 3 else ["axial"]
        for plane in planes:
            png, view = self.renderer.montage(plane, DEFAULT_MODE[self.case.task])
            images.append(ImageContent(png, f"[{view.id}]"))
            lines.append(f"- {view.id}: {plane} montage, mode {view.mode}, slices {view.extra.get('indices')}")
        if self.config.mode == "fixed":
            for mode in COVERAGE_EXTRA_MODES[self.case.task]:
                if mode not in self.renderer.modes():
                    continue
                for plane in planes:
                    png, view = self.renderer.render(plane, mode)
                    images.append(ImageContent(png, f"[{view.id}]"))
                    lines.append(f"- {view.id}: {plane} central slice, mode {mode}")
        return images, "\n".join(lines)

    def _survey(self, active: bool) -> tuple[list[Lead], str]:
        """The leads, and their text for the case card: a survey that fails costs the leads, not the inspection."""
        try:
            if not self.config.survey:
                raise SurveySkipped("turned off")
            leads = survey(self.renderer)
        except SurveySkipped as error:
            return [], f"Leads: no survey was run ({error}); no lead says nothing about the result."
        except Exception as error:
            return [], f"Leads: the survey failed ({type(error).__name__}: {error})."
        return leads, leads_text(leads, active)

    def _arbitrate(self, leads: list[Lead]) -> bool:
        """Where the result and its reference disagree, a fresh session judges the two contours without being told
        which is which (drawn as A and B, the order drawn from the case and the lead): it cannot defer to the
        reference. Its answer joins the lead's reason; the model still decides. True when a lead was read."""
        disagreements = [lead for lead in leads if lead.kind in ("result_only", "reference_only")]
        spacing = np.asarray(self.case.grid.spacing_xyz, dtype=np.float64)
        for lead in disagreements:
            box = box_from_dict(lead.box_mm)
            zoom = box_around(box_center(box), np.maximum(box_size(box) * 2.0, np.maximum(12.0 * spacing, 64.0)))
            swap = random.Random(f"{self.case.name}{lead.id}").random() < 0.5
            plane = native_plane(self.case.grid)
            axis = plane_frame(self.case.grid, plane).array_axis
            try:
                # the slice of the lead where the two differ most, windowed on the region: its centre slice may hold
                # one contour only, and the window of the whole image may leave the structure grey on grey
                inner = self.case.grid.index_window(box, 0)
                lo, hi = tuple(s.start for s in inner), tuple(s.stop for s in inner)
                differ = self.case.volume("prediction").region(lo, hi)[0] != self.case.volume("truth").region(lo, hi)[0]
                index = lo[axis] + int(np.argmax(differ.sum(axis=tuple(a for a in range(3) if a != axis))))
                outer = self.case.grid.index_window(zoom, 0)
                region = self.case.volume("image").region(
                    *(tuple(getattr(s, k) for s in outer) for k in ("start", "stop"))
                )
                window = [float(v) for v in np.percentile(region[0], [1, 99])]
                shown = {"index": index, "zoom": zoom, "window": window, "sent": False}
                views = [  # not sent to the model that inspects: they do not count as its look at the lead
                    self.renderer.render(plane, "image", **shown),
                    self.renderer.render(plane, "contours", swap=swap, **shown),
                ]
                images = [ImageContent(png, f"[{view.id}]") for png, view in views]
                self.toolbox.images_sent += len(images)
                reader = self.reader()
                reader.start(ARBITER, [])
                labels = list((self.case.labels or {}).values())
                card = f"Structure: {labels[0] if len(labels) == 1 else 'see the label map'}. The slice alone, then "
                turn = self._request(lambda r=reader, c=card + "with the two contours.", i=images: r.user(c, i))
                word = ((turn.text or "").strip().split() or [""])[0].strip(".,:").upper()
            except (ViewError, ProviderError):
                word = ""
            better = {"A": "reference" if swap else "result", "B": "result" if swap else "reference"}.get(word)
            lead.reason += (
                f" A blind reading of the two contours, not told which is the reference, judged the {better}'s the "
                "better one there."
                if better
                else " A blind reading of the two contours, not told which is the reference, could not tell which is "
                "the better one there."
            )
        return bool(disagreements)

    def _labels_summary(self) -> dict[str, object]:
        role = {"segmentation": "prediction", "registration": "fixed_labels", "synthesis": None}[self.case.task]
        if role is None or role not in self.case.inputs:
            return {}
        volume = self.case.volume(role)
        if (
            max(volume.grid.size_zyx) > 512
        ):  # the whole map in memory only when it is small; the model can ask per region
            return {"note": "labels not summarised whole: ask region_stats on a box"}
        array = volume.region((0, 0, 0), tuple(volume.grid.size_zyx))  # type: ignore[arg-type]
        return {role: label_stats(array[0], volume.grid, self.case.labels)}

    # ------------------------------------------------------------------ the report

    def _finish(self) -> Report:
        conclusion = self.toolbox.conclusion
        exhausted = self.budget.exhausted()
        if conclusion is None:
            reason = self.error or (
                f"budget exhausted: {exhausted}" if exhausted else "the model ended without concluding"
            )
            serious = self.toolbox.serious()
            conclusion = Conclusion(
                verdict="defective" if serious else "insufficient_evidence",
                summary=(
                    f"Scout concluded for the model ({reason}). "
                    + (
                        f"Major error(s) {', '.join(serious)} were recorded."
                        if serious
                        else f"{len(self.toolbox.observations)} observation(s) were recorded, none major."
                    )
                ),
                confidence=0.5 if serious else 0.2,
                by="scout",
                reason=reason,
            )
        self.usage.images_sent = self.toolbox.images_sent
        self.usage.cost_known = self.price is not None
        self.usage.cost_usd = round(self.budget.usd, 6) if self.price is not None else None
        transcript = self.out / "transcript.json"
        transcript.write_text(json.dumps(self.provider.transcript(), indent=1, default=str), encoding="utf-8")
        report = Report(
            schema_version=SCHEMA_VERSION,
            scout_version=scout_version(),
            versions=_versions(self.provider),
            created_at=self.started_at.isoformat(timespec="seconds"),
            duration_s=round(time.time() - self.budget.started, 2),
            case={
                **self.case.to_dict(),
                "inputs": self.case.fingerprint(),
                "truth_available": self.case.truth_available,
                "grid": describe_grid(self.case.grid),
            },
            config=self.config.to_dict(),
            provider={
                "name": self.provider.name,
                "model": self.provider.model,
                "remote": self.provider.remote,
                "sdk_version": self.provider.sdk_version(),
            },
            prompts={"system": system_prompt(self.case, self.config), "version": PROMPT_VERSION},
            prior_evaluation=load_prior(self.case.evaluation, self.case.name),
            tool_calls=self.toolbox.calls,
            views=self.renderer.views,
            measurements=self.toolbox.measurements,
            observations=self.toolbox.observations,
            conclusion=conclusion,
            usage=self.usage,
            budget={
                "limits": self.config.budget.to_dict(),
                "exhausted": exhausted,
                "not_enforced": self._unenforced(),
                "spent": {
                    "tool_calls": self.budget.calls,
                    "images": self.budget.images,
                    "tokens": self.budget.tokens,
                    "usd": round(self.budget.usd, 6),
                    "seconds": round(time.time() - self.budget.started, 1),
                },
            },
            error=self.error,
            transcript_path="transcript.json",
            leads=[{**lead.to_dict(), "views": self.lead_views.get(lead.id, [])} for lead in self.toolbox.leads],
            second_reading=self.second_reading,
        )
        report.save(self.out)
        return report

    def _unenforced(self) -> str | None:
        """Why ``max_usd`` could not stop this inspection, when it could not: said, never pretended."""
        if self.config.budget.max_usd is None:
            return None
        if self.price is None:
            return f"max_usd: the price of {self.provider.model} is unknown, the cost was not counted"
        if not getattr(self.provider, "running_cost", True):
            return f"max_usd: the {self.provider.name} CLI reports its cost only once its whole loop has run"
        return None


def parse_second_reading(text: str) -> tuple[str, str]:
    """The second reader's answer: VISIBLE, NOT_VISIBLE or UNSURE on the first line, then its sentence."""
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return "unparsed", ""
    head = re.sub(r"[^A-Z]+", "_", lines[0].upper()).strip("_")
    for answer in ("NOT_VISIBLE", "VISIBLE", "UNSURE"):
        if head.startswith(answer):
            return answer.lower(), " ".join(lines[1:])[:600]
    return "unparsed", " ".join(lines)[:600]


def inspect(
    case: InspectionCase,
    config: ScoutConfig | None = None,
    out: Path | str = "scout_report",
    provider: Provider | None = None,
    reader: Callable[[], Provider] | None = None,
) -> Report:
    """Inspect one case; write ``report.json``, ``report.md``, the evidence images and the transcript under ``out``."""
    return Inspector(case, config or ScoutConfig(), Path(out), provider, reader).run()


def main_version() -> str:
    return f"konfai-scout {scout_version()} (python {sys.version.split()[0]})"
