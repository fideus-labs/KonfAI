# SPDX-License-Identifier: Apache-2.0
"""The inspection: coverage views, then the model's own exploration under a budget, then the report.

Two modes for the same case and the same model: ``fixed`` hands over a fixed set of views and lets the model note
and conclude (measurements allowed, no further views); ``active`` starts from the same coverage and lets the model
ask for views and zoom. Everything either mode does is recorded the same way, so the two compare."""

from __future__ import annotations

import datetime as dt
import json
import platform
import sys
import time
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .case import InspectionCase
from .config import ScoutConfig
from .evaluation import RegionEvaluator, konfai_version, load_prior
from .measure import label_stats
from .prompts import PROMPT_VERSION, case_card, system_prompt
from .providers import ImageContent, Provider, ProviderError, ToolResult, Turn, check_privacy, make_provider
from .providers.pricing import cost_usd, price_of
from .report import Conclusion, Report, Usage
from .tools import Toolbox
from .views import DEFAULT_MODE, Renderer

COVERAGE_EXTRA_MODES = {
    "segmentation": ["both"],
    "registration": ["field_magnitude", "jacobian", "overlay"],
    "synthesis": ["difference", "side_by_side"],
}


def scout_version() -> str:
    try:
        return version("konfai-scout")
    except PackageNotFoundError:
        return "0.0.0+source"


def _versions(provider: Provider) -> dict[str, str]:
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("numpy", "SimpleITK", "Pillow", "scipy"):
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            pass
    konfai = konfai_version()
    if konfai:
        out["konfai"] = konfai
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

    def exhausted(self) -> str | None:
        budget = self.config.budget
        if self.calls >= budget.max_tool_calls:
            return f"{self.calls} tool calls (limit {budget.max_tool_calls})"
        if self.images >= budget.max_images:
            return f"{self.images} images (limit {budget.max_images})"
        if budget.max_seconds is not None and time.time() - self.started >= budget.max_seconds:
            return f"{time.time() - self.started:.0f} s (limit {budget.max_seconds:.0f} s)"
        if budget.max_usd is not None and self.usd >= budget.max_usd:
            return f"${self.usd:.3f} (limit ${budget.max_usd:.2f})"
        if budget.max_tokens is not None and self.tokens >= budget.max_tokens:
            return f"{self.tokens} tokens (limit {budget.max_tokens})"
        return None


class Inspector:
    def __init__(self, case: InspectionCase, config: ScoutConfig, out: Path, provider: Provider | None = None) -> None:
        self.case = case
        self.config = config
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        if provider is None:
            check_privacy(config)
        self.provider = provider or make_provider(config)
        self.renderer = Renderer(case, self.out / "work", config, self.out / "evidence")
        self.evaluator = RegionEvaluator(case, config, self.out / "work", self.renderer.derived)
        self.toolbox = Toolbox(case, config, self.renderer, self.evaluator)
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
        card = case_card(self.case, prior, self._labels_summary(), self.renderer.modes(), views_text)
        turn: Turn | None = None
        try:
            turn = self._request(lambda: self.provider.user(card, images))
            self.budget.images += len(images)
            self.toolbox.images_sent += len(images)
            reminders = 0
            final = False
            while self.toolbox.conclusion is None:
                exhausted = self.budget.exhausted()
                if turn.stop == "refusal":
                    self.error = f"the model refused: {turn.text}"
                    break
                if turn.tool_calls:
                    if exhausted and final:
                        break
                    if exhausted:
                        # the budget is spent: only a note or a conclusion still runs, once
                        final = True
                        results = [
                            self._refused(call, exhausted)
                            if call.name not in ("note", "conclude")
                            else self.toolbox.call(call)
                            for call in turn.tool_calls
                        ]
                        if self.toolbox.conclusion is not None:
                            break
                        final_specs = self.toolbox.specs(exploration, final=True)
                        turn = self._request(lambda r=results, t=final_specs: self.provider.tool_results(r, t))
                        continue
                    results = [self.toolbox.call(call) for call in turn.tool_calls]
                    self.budget.calls += len(results)
                    self.budget.images += sum(len(r.images) for r in results)
                    if self.toolbox.conclusion is not None:
                        break
                    turn = self._request(lambda r=results: self.provider.tool_results(r))
                    continue
                if final or reminders >= 2:
                    break
                reminders += 1
                if exhausted:
                    final = True
                    notice = (
                        f"Budget exhausted ({exhausted}). Call `note` for anything established and not yet "
                        "recorded, then `conclude` now."
                    )
                    turn = self._request(lambda n=notice: self.provider.user(n, None))
                    continue
                turn = self._request(
                    lambda: self.provider.user(
                        "Nothing was recorded by this message. Use the tools: `note` for each finding, then "
                        "`conclude` to "
                        "end; keep exploring with `view`/`measure` if you are not done.",
                        None,
                    )
                )
        except ProviderError as error:
            self.error = f"provider error: {error}"
        except Exception as error:  # the report must still be written
            self.error = f"{type(error).__name__}: {error}"
        return self._finish()

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

    # ------------------------------------------------------------------ coverage

    def _coverage(self) -> tuple[list[ImageContent], str]:
        """The views both modes start from: one montage per plane in the task's default mode; the fixed mode adds
        the central slice of each plane in the secondary modes."""
        images: list[ImageContent] = []
        lines = ["Views attached (ids as printed on each image):"]
        planes = (
            ["axial", "coronal", "sagittal"] if self.case.volume(self.case.image_role).dimension == 3 else ["axial"]
        )
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

    def _labels_summary(self) -> dict[str, object]:
        role = {"segmentation": "prediction", "registration": "fixed_labels", "synthesis": None}[self.case.task]
        if role is None or role not in self.case.inputs:
            return {}
        volume = self.case.volume(role)
        if max(volume.shape_zyx) > 512:  # the whole map in memory only when it is small; the model can ask per region
            return {"note": "labels not summarised whole: ask region_stats on a box"}
        array = volume.region((0, 0, 0), volume.shape_zyx)
        return {role: label_stats(array, volume.grid, self.case.labels)}

    # ------------------------------------------------------------------ the report

    def _finish(self) -> Report:
        conclusion = self.toolbox.conclusion
        exhausted = self.budget.exhausted()
        if conclusion is None:
            reason = self.error or (
                f"budget exhausted: {exhausted}" if exhausted else "the model ended without concluding"
            )
            confirmed = [o for o in self.toolbox.observations if o.kind == "confirmed_error"]
            conclusion = Conclusion(
                verdict="defective" if confirmed else "insufficient_evidence",
                summary=(
                    f"Scout concluded for the model ({reason}). "
                    + (
                        f"{len(confirmed)} confirmed error(s) were recorded."
                        if confirmed
                        else f"{len(self.toolbox.observations)} observation(s) were recorded, none confirmed."
                    )
                ),
                confidence=0.5 if confirmed else 0.2,
                by="scout",
                reason=reason,
            )
        self.usage.images_sent = self.toolbox.images_sent
        self.usage.cost_known = self.price is not None
        self.usage.cost_usd = round(self.budget.usd, 6) if self.price is not None else None
        transcript = self.out / "transcript.json"
        transcript.write_text(json.dumps(self.provider.transcript(), indent=1, default=str), encoding="utf-8")
        report = Report(
            schema_version="1.0",
            scout_version=scout_version(),
            versions=_versions(self.provider),
            created_at=self.started_at.isoformat(timespec="seconds"),
            duration_s=round(time.time() - self.budget.started, 2),
            case={
                **self.case.to_dict(),
                "inputs": self.case.fingerprint(),
                "truth_available": self.case.truth_available,
                "grid": self.case.grid.describe(),
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
        )
        report.save(self.out)
        return report


def inspect(
    case: InspectionCase,
    config: ScoutConfig | None = None,
    out: Path | str = "scout_report",
    provider: Provider | None = None,
) -> Report:
    """Inspect one case; write ``report.json``, ``report.md``, the evidence images and the transcript under ``out``."""
    return Inspector(case, config or ScoutConfig(), Path(out), provider).run()


def main_version() -> str:
    return f"konfai-scout {scout_version()} (python {sys.version.split()[0]})"
