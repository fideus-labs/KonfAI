# SPDX-License-Identifier: Apache-2.0
import json

from konfai_scout.report import (
    Conclusion,
    Measurement,
    Observation,
    Report,
    ToolCall,
    Usage,
    View,
    compare,
    compare_markdown,
    export,
    match_defects,
    render_markdown,
)


def sample(mode="active", verdict="defective") -> Report:
    return Report(
        schema_version="1.0",
        scout_version="0",
        versions={"numpy": "2"},
        created_at="2026-11-07T00:00:00+00:00",
        duration_s=3.5,
        case={
            "name": "c1",
            "task": "segmentation",
            "truth_available": True,
            "inputs": {"image": {"path": "/a/image.mha", "sha256": "abc"}},
        },
        config={"mode": mode},
        provider={"name": "mock", "model": "heuristic"},
        prompts={"system": "s"},
        prior_evaluation={"PRED:TRUTH:Dice": 0.5},
        tool_calls=[ToolCall(1, "view", {"plane": "axial"}, 0.0, 0.1, True, "v001")],
        views=[
            View(
                "v001",
                "slice",
                "axial",
                0,
                3,
                [0, 0, 1],
                {"min": [0, 0, 0], "max": [1, 1, 1]},
                "prediction",
                [0, 1],
                [10, 10],
                {"origin": [0, 0, 0], "u": [1, 0, 0], "v": [0, 1, 0]},
                "evidence/v001.png",
                "deadbeef",
                True,
            )
        ],
        measurements=[
            Measurement(
                "m001", "konfai_eval", True, None, {"min": [0, 0, 0], "max": [1, 1, 1]}, {"PRED:TRUTH:Dice": 0.1}
            )
        ],
        observations=[
            Observation(
                "o001",
                "confirmed_error",
                "missing_structure",
                "major",
                0.9,
                "the cube is missing",
                {"min": [0, 0, 0], "max": [1, 1, 1]},
                "axial",
                ["v001"],
                ["m001"],
            )
        ],
        conclusion=Conclusion(verdict, "one confirmed error", 0.8),
        usage=Usage(2, 100, 50, 0, 0, 1, 0.001, True),
        budget={"limits": {}, "exhausted": None},
    )


def test_json_round_trip_and_summary(tmp_path):
    report = sample()
    path = report.save(tmp_path)
    loaded = Report.load(path)
    assert loaded.to_dict() == report.to_dict()
    assert Report.load(tmp_path).observations[0].views == ["v001"]
    data = json.loads(path.read_text())
    assert data["schema_version"] == "1.0" and data["conclusion"]["verdict"] == "defective"
    summary = report.summary()
    assert summary["counts"]["confirmed_error"] == 1 and summary["counts"]["major"] == 1 and summary["views"] == 1


def test_markdown_is_short_and_names_the_evidence():
    text = render_markdown(sample())
    lines = text.splitlines()
    assert lines[0] == "# c1 (segmentation): defective" and lines[2].startswith("**defective** (0.80, by model).")
    observation = next(line for line in lines if line.startswith("- **o001**"))
    assert "missing_structure, confirmed error, major (0.90)" in observation and "[v001; m001]" in observation
    assert (
        "- m001 (0, 0, 0) mm, 1 x 1 x 1 mm: Dice 0.1" in lines
        and "- [v001](evidence/v001.png) axial prediction, slice 3" in lines
    )
    assert "- prior evaluation (whole case): Dice 0.5" in lines and "## Tool calls" not in text
    report = sample()
    report.measurements.append(
        Measurement(
            "m002", "konfai_eval", True, None, report.measurements[0].box_measured, dict(report.measurements[0].values)
        )
    )
    report.observations[0].measurements.append("m002")
    assert "- m002 same as m001" in render_markdown(report).splitlines()  # Scout's check of a box measured already
    report.measurements[1].values["PRED:TRUTH:Dice"] = 0.10001  # displayed alike, measured apart
    assert "- m002 same as m001" not in render_markdown(report).splitlines()


def test_compare_rows():
    rows = compare([sample("fixed", "acceptable"), sample("active")])["rows"]
    assert [r["mode"] for r in rows] == ["fixed", "active"]
    assert rows[0]["confirmed"] == 1 and rows[1]["cost_usd"] == 0.001
    assert rows[0]["planted"] is None and rows[0]["not_visible"] is None
    assert "| case |" in compare_markdown({"rows": rows})


def test_planted_defects_are_found_by_overlap_and_the_rest_are_invented():
    report = sample()
    report.observations.append(
        Observation("o002", "suspected_error", "leak", "minor", 0.4, "far away", {"min": [5, 5, 5], "max": [6, 6, 6]})
    )
    report.observations.append(Observation("o003", "measured_property", "other", "minor", 0.9, "a volume", None))
    report.observations.append(Observation("o004", "suspected_error", "blur", "minor", 0.5, "soft edges", None))
    planted = {
        "cube": {"defect": "missing_structure", "box_mm": {"min": [0.5, 0.5, 0.5], "max": [2, 2, 2]}},
        "hole": {"defect": "holes", "box_mm": {"min": [3, 3, 3], "max": [4, 4, 4]}},
        "blur": {"defect": "blur"},
    }
    matched = match_defects(report, planted)
    assert matched["planted"] == 3 and matched["found_names"] == ["cube", "blur"]
    assert matched["false_positive_ids"] == ["o002"]  # a measured property is not an error
    assert match_defects(report, {})["false_positives"] == 3
    rows = compare([report], [planted])["rows"]
    assert (rows[0]["planted"], rows[0]["found"], rows[0]["false_positives"]) == (3, 2, 1)
    assert "| false_positives |" in compare_markdown({"rows": rows}).splitlines()[0]


def test_export_levels_show_less_and_less_of_the_case():
    report = sample()
    report.observations[0].second_reading = {"answer": "visible", "text": "the cube of c1 is gone", "views": ["v001"]}
    report.measurements[0].values["prediction_labels.2.name"] = "c1 lesion"  # a label name is text: never exported
    assert export(report, "full") == report.to_dict()
    regions = export(report, "regions")
    assert regions["verdict"] == "defective" and regions["task"] == "segmentation"
    assert regions["observations"] == [
        {
            "id": "o001",
            "kind": "confirmed_error",
            "defect": "missing_structure",
            "severity": "major",
            "confidence": 0.9,
            "box_mm": {"min": [0, 0, 0], "max": [1, 1, 1]},
            "plane": "axial",
            "measurements": [
                {
                    "id": "m001",
                    "reference": True,
                    "box_measured": {"min": [0, 0, 0], "max": [1, 1, 1]},
                    "values": {"PRED:TRUTH:Dice": 0.1},
                }
            ],
            "second_reading": "visible",
        }
    ]
    text = json.dumps(regions)
    for secret in ("c1", "image.mha", "evidence/", "the cube is missing", "transcript", "0.5"):
        assert secret not in text  # the name, a path, an image, a description, the prior evaluation
    assert export(report, "counts") == {
        "level": "counts",
        "verdict": "defective",
        "counts": {"confirmed_error": {"missing_structure": {"major": 1}}},
    }
