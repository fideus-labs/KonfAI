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


def test_markdown_names_the_evidence():
    text = render_markdown(sample())
    assert "Verdict: defective" in text
    assert "missing_structure" in text and "m001" in text and "![v001" in text and "evidence/v001.png" in text
    assert "PRED:TRUTH:Dice = 0.5" in text


def test_compare_rows():
    rows = compare([sample("fixed", "acceptable"), sample("active")])["rows"]
    assert [r["mode"] for r in rows] == ["fixed", "active"]
    assert rows[0]["confirmed"] == 1 and rows[1]["cost_usd"] == 0.001
    assert "| case |" in compare_markdown({"rows": rows})
