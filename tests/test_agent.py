# SPDX-License-Identifier: Apache-2.0
"""The loop: coverage, exploration, budget, provider failures, and what the report records of them."""

import json

import pytest
from konfai_scout import InspectionCase, ScoutConfig, inspect
from konfai_scout.config import Budget
from konfai_scout.providers.base import ProviderError, Turn, Usage
from konfai_scout.providers.mock import HeuristicProvider, ScriptedProvider, call, turn


@pytest.mark.parametrize("mode", ["active", "fixed"])
@pytest.mark.parametrize("key", ["segmentation", "registration", "synthesis"])
def test_the_heuristic_inspection_leaves_a_complete_report(fixtures, tmp_path, key, mode):
    case, defects = fixtures[key]
    report = inspect(case, ScoutConfig(provider="mock", mode=mode, max_px=200), tmp_path / mode)
    assert report.error is None and report.conclusion is not None and report.conclusion.by == "model"
    assert (tmp_path / mode / "report.json").exists() and (tmp_path / mode / "report.md").exists()
    assert (tmp_path / mode / "transcript.json").exists() and (tmp_path / mode / "case.json").exists()
    assert report.observations and report.observations[0].views
    assert all((tmp_path / mode / v.path).exists() for v in report.views)
    tools = [c.tool for c in report.tool_calls]
    if mode == "fixed":
        assert "view" not in tools and "montage" not in tools
        assert len([v for v in report.views if v.kind == "slice"]) >= 3  # the central slices in the secondary modes
    else:
        assert "view" in tools
    assert report.usage.images_sent >= 3 and report.budget["spent"]["tool_calls"] == len(report.tool_calls)
    assert report.case["truth_available"] and report.versions["prompt"] == "1.1"
    assert report.prompts["system"].startswith("You are KonfAI Scout")
    summary = report.summary()
    assert summary["verdict"] == report.conclusion.verdict and summary["observations"][0]["defect"]
    data = json.loads((tmp_path / mode / "report.json").read_text())
    assert (
        data["schema_version"] == "1.0"
        and data["case"]["inputs"]["image" if key == "segmentation" else list(data["case"]["inputs"])[0]]["sha256"]
    )


def test_without_truth_nothing_is_confirmed(fixtures, tmp_path):
    case, _ = fixtures["segmentation_no_truth"]
    report = inspect(case, ScoutConfig(provider="mock", max_px=200), tmp_path)
    assert not report.case["truth_available"]
    assert all(o.kind != "confirmed_error" for o in report.observations)
    assert "No reference was given" in (tmp_path / "report.md").read_text()


def test_the_budget_ends_an_endless_explorer(fixtures, tmp_path):
    case, _ = fixtures["segmentation"]

    class Endless(ScriptedProvider):
        def _next(self) -> Turn:
            self.calls += 1
            return turn(
                call(
                    "view",
                    self.calls,
                    plane="axial",
                    mode="prediction",
                    position_mm=None,
                    index=None,
                    zoom=None,
                    window=None,
                    label=None,
                )
            )

    provider = Endless([])
    config = ScoutConfig(provider="mock", budget=Budget(max_tool_calls=3), max_px=200)
    report = inspect(case, config, tmp_path, provider=provider)
    assert report.budget["exhausted"] and report.budget["exhausted"].startswith("3 tool calls")
    assert report.conclusion.by == "scout" and report.conclusion.verdict == "insufficient_evidence"
    assert len([c for c in report.tool_calls if c.tool == "view"]) <= 4  # three, plus the one of the final turn
    assert report.error is None


def test_a_retryable_provider_error_is_retried_and_a_fatal_one_recorded(fixtures, tmp_path):
    case, _ = fixtures["segmentation"]
    script = [turn(call("conclude", 1, verdict="acceptable", summary="fine", confidence=0.6))]
    provider = ScriptedProvider(list(script), raise_at={1: ProviderError("overloaded", retryable=True)})
    config = ScoutConfig(provider="mock", retry_wait_seconds=0.0, max_px=200)
    report = inspect(case, config, tmp_path / "retry", provider=provider)
    assert report.error is None and report.conclusion.verdict == "acceptable" and provider.calls == 2
    provider = ScriptedProvider(list(script), raise_at={1: ProviderError("bad key", retryable=False)})
    report = inspect(case, config, tmp_path / "fatal", provider=provider)
    assert report.error == "provider error: bad key" and report.conclusion.by == "scout"
    assert report.conclusion.verdict == "insufficient_evidence" and (tmp_path / "fatal" / "report.json").exists()
    provider = ScriptedProvider(
        list(script), raise_at={1: ProviderError("down", retryable=True), 2: ProviderError("down", retryable=True)}
    )
    config = ScoutConfig(provider="mock", retry_wait_seconds=0.0, provider_retries=1, max_px=200)
    report = inspect(case, config, tmp_path / "exhausted", provider=provider)
    assert report.error == "provider error: down"


def test_a_refusal_and_a_silent_model_are_recorded(fixtures, tmp_path):
    case, _ = fixtures["synthesis"]
    refusing = ScriptedProvider([Turn(text="no", tool_calls=[], stop="refusal", usage=Usage(1, 1))])
    report = inspect(case, ScoutConfig(provider="mock", max_px=200), tmp_path / "refusal", provider=refusing)
    assert report.error and "refused" in report.error
    silent = ScriptedProvider([Turn(text="looks fine", tool_calls=[], stop="end_turn", usage=Usage(1, 1))] * 3)
    report = inspect(case, ScoutConfig(provider="mock", max_px=200), tmp_path / "silent", provider=silent)
    assert report.conclusion.by == "scout" and "without concluding" in report.conclusion.reason
    assert silent.calls == 3  # the first answer, two reminders, then Scout concludes


def test_tool_errors_do_not_end_the_inspection(fixtures, tmp_path):
    case, _ = fixtures["registration_no_ref"]
    script = [
        turn(
            call(
                "view",
                1,
                plane="axial",
                mode="labels",
                position_mm=None,
                index=None,
                zoom=None,
                window=None,
                label=None,
            )
        ),
        turn(
            call(
                "note",
                2,
                kind="measured_property",
                defect="other",
                severity="minor",
                confidence=0.5,
                description="d",
                box_mm=None,
                plane=None,
                views=["v001"],
                measurements=[],
            )
        ),
        turn(call("conclude", 3, verdict="acceptable", summary="s", confidence=0.5)),
    ]
    report = inspect(case, ScoutConfig(provider="mock", max_px=200), tmp_path, provider=ScriptedProvider(script))
    assert [c.ok for c in report.tool_calls] == [False, True, True]
    assert "not available" in report.tool_calls[0].error
    assert report.observations[0].views == ["v001"] and report.conclusion.verdict == "acceptable"


def test_a_case_json_round_trip_and_a_registration_from_a_transform_file(fixtures, tmp_path):
    import SimpleITK as sitk

    case, _ = fixtures["registration_no_ref"]
    case.save(tmp_path / "case.json")
    loaded = InspectionCase.load(tmp_path / "case.json")
    assert loaded.inputs == case.inputs and loaded.task == "registration"
    transform = sitk.TranslationTransform(3, (1.5, -2.0, 0.5))
    sitk.WriteTransform(transform, str(tmp_path / "Transform.tfm"))
    affine = InspectionCase(
        "registration",
        "affine",
        {"fixed": case.inputs["fixed"], "moving": case.inputs["moving"], "transform": tmp_path / "Transform.tfm"},
    )
    report = inspect(
        affine, ScoutConfig(provider="mock", max_px=200), tmp_path / "affine", provider=HeuristicProvider()
    )
    assert report.error is None and (tmp_path / "affine" / "work" / "derived" / "field.mha").exists()
