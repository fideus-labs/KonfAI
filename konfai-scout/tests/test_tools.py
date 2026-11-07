# SPDX-License-Identifier: Apache-2.0
"""The tools' checks: evidence must exist, a confirmation needs a measurement, a verdict says what it rests on."""

import json

from konfai_scout.config import ScoutConfig
from konfai_scout.evaluation import RegionEvaluator
from konfai_scout.providers.base import ToolCall
from konfai_scout.tools import Toolbox
from konfai_scout.views import Renderer


def toolbox(fixtures, key, tmp_path) -> Toolbox:
    case, _ = fixtures[key]
    config = ScoutConfig(max_px=200)
    renderer = Renderer(case, tmp_path / "work", config, tmp_path / "evidence")
    return Toolbox(case, config, renderer, RegionEvaluator(case, config, tmp_path / "work", renderer.derived))


def run(box: Toolbox, name: str, **arguments) -> dict:
    result = box.call(ToolCall(id=f"c{len(box.calls)}", name=name, arguments=arguments))
    return {"error": result.text} if result.is_error else json.loads(result.text)


def test_region_stats_see_the_planted_defects(fixtures, tmp_path):
    case, defects = fixtures["segmentation"]
    box = toolbox(fixtures, "segmentation", tmp_path)
    whole = run(box, "region_stats", box_mm=case.grid.bounds.to_dict())["region"]
    assert whole["prediction_labels"]["1"]["components"] == 2  # the sphere and the spurious blob
    assert "2" not in whole["prediction_labels"] and whole["truth_labels"]["2"]["name"] == "cube"
    assert whole["overlap_voxels_prediction_vs_truth"]["2"]["only_b"] > 0
    cube = run(box, "region_stats", box_mm=defects["missing_cube"]["box_mm"])["region"]
    assert cube["truth_labels"]["2"]["voxels"] == whole["truth_labels"]["2"]["voxels"]


def test_notes_need_known_evidence_and_confirmation_needs_a_measurement(fixtures, tmp_path):
    box = toolbox(fixtures, "segmentation", tmp_path)
    view = run(box, "view", plane="axial", mode="prediction")["view"]
    bad = run(
        box,
        "note",
        kind="confirmed_error",
        defect="leak",
        severity="minor",
        confidence=0.5,
        description="x",
        views=["v999"],
        measurements=[],
    )
    assert "unknown evidence" in bad["error"]
    downgraded = run(
        box,
        "note",
        kind="confirmed_error",
        defect="leak",
        severity="minor",
        confidence=0.5,
        description="x",
        views=[view["id"]],
        measurements=[],
    )["observation"]
    assert downgraded["kind"] == "suspected_error" and downgraded["notes"]
    wrong = run(
        box,
        "note",
        kind="confirmed_error",
        defect="not_a_defect",
        severity="minor",
        confidence=0.5,
        description="x",
        views=[],
        measurements=[],
    )
    assert "defect is one of" in wrong["error"]
    conclusion = run(box, "conclude", verdict="defective", summary="s", confidence=0.9)["conclusion"]
    assert conclusion["note"] and "no confirmed error" in conclusion["note"]
    assert box.conclusion is not None and box.conclusion.reason


def test_errors_are_reported_to_the_model_and_logged(fixtures, tmp_path):
    box = toolbox(fixtures, "segmentation", tmp_path)
    assert "unknown tool" in run(box, "fly")["error"]
    assert "bad arguments" in run(box, "view", nowhere=1)["error"]
    assert "no voxel" in run(box, "region_stats", box_mm={"min": [900, 900, 900], "max": [901, 901, 901]})["error"]
    assert [c.ok for c in box.calls] == [False, False, False]
    assert all(c.error for c in box.calls)


def test_registration_tools_without_references(fixtures, tmp_path):
    case, _ = fixtures["registration_no_ref"]
    box = toolbox(fixtures, "registration_no_ref", tmp_path)
    stats = run(box, "region_stats", box_mm=case.grid.bounds.to_dict())["region"]
    assert stats["displacement"]["max_mm"] > 1.0 and "fixed_labels" not in stats
    view = run(box, "view", plane="sagittal", mode="jacobian")["view"]
    assert "scale" in view and view["scale"]["folded_pixels"] == 0
    specs = {t.name for t in box.specs(exploration=True)}
    assert specs == {"view", "montage", "measure", "region_stats", "note", "conclude"}
    assert {t.name for t in box.specs(exploration=False, final=True)} == {"note", "conclude"}
