# SPDX-License-Identifier: Apache-2.0
"""The loop: coverage, exploration, budget, provider failures, and what the report records of them."""

import json
from pathlib import Path

import numpy as np
import pytest
from konfai_scout import InspectionCase, ScoutConfig, inspect
from konfai_scout.agent import parse_second_reading
from konfai_scout.config import Budget
from konfai_scout.providers.base import ProviderError, Turn, Usage
from konfai_scout.providers.mock import HeuristicProvider, ScriptedProvider, call, turn
from konfai_scout.report import compare, defects_of, match_defects
from konfai_scout.synthetic import registration_case, segmentation_case, synthesis_case


@pytest.mark.parametrize("mode", ["active", "fixed"])
@pytest.mark.parametrize("key", ["segmentation", "registration", "synthesis"])
def test_the_heuristic_inspection_leaves_a_complete_report(fixtures, tmp_path, key, mode):
    case, _defects = fixtures[key]
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
    explored = [c for c in report.tool_calls if c.tool not in ("note", "conclude")]  # notes are outside the budget
    assert report.usage.images_sent >= 3 and report.budget["spent"]["tool_calls"] == len(explored)
    assert report.case["truth_available"] and report.versions["prompt"] == "1.3"
    assert report.prompts["system"].startswith("You are KonfAI Scout")
    summary = report.summary()
    assert summary["verdict"] == report.conclusion.verdict and summary["observations"][0]["defect"]
    data = json.loads((tmp_path / mode / "report.json").read_text())
    assert (
        data["schema_version"] == "1.1"
        and data["case"]["inputs"]["image" if key == "segmentation" else next(iter(data["case"]["inputs"]))]["sha256"]
    )


def test_the_model_is_blind_to_the_method_and_the_scores_unless_told(fixtures, tmp_path):
    case, _ = fixtures["segmentation"]
    prior = {"PRED:TRUTH:Dice:1": 0.912}
    named = InspectionCase(case.task, "methodx_case07", case.inputs, case.labels, prior, notes="by MethodX")
    for blind in (True, False):
        provider = HeuristicProvider()
        report = inspect(named, ScoutConfig(blind=blind, max_px=200), tmp_path / str(blind), provider=provider)
        assert all((secret in provider.card) is not blind for secret in ("methodx_case07", "MethodX", "0.912"))
        assert report.prior_evaluation == prior and report.config["blind"] is blind  # the report's reader sees all


def test_without_truth_nothing_is_confirmed(fixtures, tmp_path):
    case, _ = fixtures["segmentation_no_truth"]
    report = inspect(case, ScoutConfig(provider="mock", max_px=200), tmp_path)
    assert not report.case["truth_available"]
    assert all(o.kind != "confirmed_error" for o in report.observations)
    assert "No reference given" in (tmp_path / "report.md").read_text()


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
                kind="suspected_error",
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
    assert report.error is None
    from konfai.utils.dataset import Dataset
    from konfai_scout.case import Volume
    from konfai_scout.planes import same_grid

    # the sampled field is written as the image SimpleITK made, on the fixed grid, one copy in memory
    field = Volume(Dataset(tmp_path / "affine" / "work" / "derived" / "Dataset", "mha"), "FIELD", "affine", "field")
    assert field.components == 3 and same_grid(field.grid, affine.grid)
    assert np.allclose(field.region((0, 0, 0), (2, 2, 2)).reshape(3, -1).T, [1.5, -2.0, 0.5], atol=1e-5)


def test_relative_inputs_are_staged_from_where_they_are(fixtures, tmp_path, monkeypatch):
    case, _ = fixtures["segmentation"]
    monkeypatch.chdir(case.inputs["image"].parent)
    relative = InspectionCase(
        "segmentation", "relative", {"image": Path("image.mha"), "prediction": Path("prediction.mha")}
    )
    assert all(path.is_absolute() for path in relative.inputs.values())
    relative.stage(tmp_path / "stage")  # the staging links from another directory
    assert np.array_equal(relative.volume("prediction").plane(0, 12), case.volume("prediction").plane(0, 12))


def test_the_second_reading_re_reads_each_error_blind(fixtures, tmp_path):
    case, _ = fixtures["segmentation"]
    readers = []

    def reader():
        provider = ScriptedProvider(
            [
                Turn(
                    text="**Not visible** — the cube is drawn.\nIn v005 and v006 the cube is there.",
                    tool_calls=[],
                    stop="end_turn",
                    usage=Usage(3, 2),
                )
            ]
        )
        readers.append(provider)
        return provider

    config = ScoutConfig(provider="mock", max_px=200, second_reading=True)
    report = inspect(case, config, tmp_path / "blind", provider=HeuristicProvider(), reader=reader)
    errors = [o for o in report.observations if o.kind != "measured_property"]
    assert report.error is None and errors and len(readers) == len(errors)
    for o in errors:
        assert o.second_reading["answer"] == "not_visible" and o.second_reading["text"].startswith("In v005")
        assert o.second_reading["views"] and all(
            any(v.id == i and v.sent for v in report.views) for i in o.second_reading["views"]
        )
    assert all(o.second_reading is None for o in report.observations if o.kind == "measured_property")
    system, user = readers[0].messages[0], readers[0].messages[1]
    assert system["role"] == "system" and "second reader" in system["content"] and "VISIBLE" in system["content"]
    assert user["images"] == 3 and user["content"].startswith("Claim: ") and "Views attached" in user["content"]
    assert "heuristic" not in user["content"] and "measure" not in user["content"].lower()
    assert report.second_reading["observations"] == len(errors) and report.second_reading["not_visible"] == len(errors)
    assert (tmp_path / "blind" / "second_reading.json").exists()
    assert "second reading: not_visible" in (tmp_path / "blind" / "report.md").read_text()
    assert report.summary()["observations"][0]["second_reading"] == "not_visible"
    assert compare([report])["rows"][0]["not_visible"] == len(errors)
    assert report.usage.images_sent >= 3 + 3 * len(errors) and report.usage.requests > readers[0].calls
    # the default reader is a fresh session of the configured provider: the heuristic one cannot see, and says so
    report = inspect(case, config, tmp_path / "default", provider=HeuristicProvider())
    assert [o.second_reading["answer"] for o in report.observations if o.kind != "measured_property"] == ["unsure"]
    assert parse_second_reading("VISIBLE.\nx y") == ("visible", "x y")
    assert parse_second_reading("Hmm, maybe") == ("unparsed", "Hmm, maybe")
    assert parse_second_reading("") == ("unparsed", "")


def test_a_clean_case_is_a_negative_control(tmp_path):
    case, defects = segmentation_case(tmp_path / "clean", clean=True)
    assert defects == {} and json.loads((tmp_path / "clean" / "defects.json").read_text()) == {}
    report = inspect(case, ScoutConfig(provider="mock", max_px=200), tmp_path / "out")
    assert report.error is None and report.observations
    assert all(o.kind == "measured_property" for o in report.observations)
    assert match_defects(report, defects) == {
        "planted": 0,
        "found": 0,
        "found_names": [],
        "false_positives": 0,
        "false_positive_ids": [],
    }
    assert defects_of(report) == {}
    assert registration_case(tmp_path / "reg", clean=True)[1] == {}
    assert synthesis_case(tmp_path / "syn", clean=True)[1] == {}
