# SPDX-License-Identifier: Apache-2.0
"""The tools' checks: evidence must exist, a confirmation needs a measurement, a verdict says what it rests on."""

import json

from konfai_scout.config import ScoutConfig
from konfai_scout.evaluation import RegionEvaluator
from konfai_scout.planes import box_to_dict
from konfai_scout.providers.base import ToolCall
from konfai_scout.tools import Toolbox, _unsourced_figures
from konfai_scout.views import Renderer


def toolbox(fixtures, key, tmp_path) -> Toolbox:
    case, _ = fixtures[key]
    case.stage(tmp_path / "work" / "stage")
    config = ScoutConfig(max_px=200)
    renderer = Renderer(case, tmp_path / "work", config, tmp_path / "evidence")
    return Toolbox(case, config, renderer, RegionEvaluator(case, config, tmp_path / "work", renderer.derived))


def run(box: Toolbox, name: str, **arguments) -> dict:
    result = box.call(ToolCall(id=f"c{len(box.calls)}", name=name, arguments=arguments))
    return {"error": result.text} if result.is_error else json.loads(result.text)


def test_region_stats_see_the_planted_defects(fixtures, tmp_path):
    case, defects = fixtures["segmentation"]
    box = toolbox(fixtures, "segmentation", tmp_path)
    whole = run(box, "region_stats", box_mm=box_to_dict(case.grid.world_box()))["region"]
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
    stats = run(box, "region_stats", box_mm=box_to_dict(case.grid.world_box()))["region"]
    assert stats["displacement"]["max_mm"] > 1.0 and "fixed_labels" not in stats
    view = run(box, "view", plane="sagittal", mode="jacobian")["view"]
    assert "scale" in view and view["scale"]["folded_pixels"] == 0
    specs = {t.name for t in box.specs(exploration=True)}
    assert specs == {"view", "montage", "measure", "region_stats", "note", "conclude"}
    assert {t.name for t in box.specs(exploration=False, final=True)} == {"note", "conclude"}


def test_the_hard_rules_of_note(fixtures, tmp_path):
    """What never enters the report: an error no cited view shows, a measurement taken elsewhere, a figure without a
    source. What is downgraded: a confirmation whose measurement holds no value. What is marked: an estimate."""
    from konfai_scout.report import Measurement
    from konfai_scout.tools import _defect_by_definition

    _case, defects = fixtures["segmentation"]
    box = toolbox(fixtures, "segmentation", tmp_path)
    cube, blob = defects["missing_cube"]["box_mm"], defects["spurious_blob"]["box_mm"]
    zoom = run(
        box, "view", plane="axial", mode="both", position_mm=None, index=None, zoom=cube, window=None, label=None
    )["view"]
    montage = run(box, "montage", plane="axial", mode=None, count=None, window=None)["view"]
    error = {
        "kind": "suspected_error",
        "defect": "spurious_structure",
        "severity": "minor",
        "confidence": 0.5,
        "description": "a blob",
        "box_mm": blob,
        "plane": "axial",
    }
    assert "at least one view" in run(box, "note", **error, views=[], measurements=[])["error"]
    assert "none of the cited views shows" in run(box, "note", **error, views=[zoom["id"]], measurements=[])["error"]
    seen = run(box, "note", **error, views=[montage["id"]], measurements=[])["observation"]
    assert seen["kind"] == "suspected_error" and not seen["notes"]
    # a figure needs a source: the measurement that holds it, cited
    stats = run(box, "region_stats", box_mm=cube)
    mid, voxels = stats["measurement"]["id"], stats["region"]["truth_labels"]["2"]["voxels"]
    measured = {
        "kind": "measured_property",
        "defect": "missing_structure",
        "severity": "major",
        "confidence": 0.9,
        "box_mm": cube,
        "plane": "axial",
        "views": [zoom["id"]],
    }
    assert "cites the measurement" in run(box, "note", **measured, description="x", measurements=[])["error"]
    wrong = run(box, "note", **measured, description="label 2 has 999 voxels", measurements=[mid])
    assert "999 voxels" in wrong["error"] and "no cited measurement" in wrong["error"]
    right = run(
        box, "note", **measured, description=f"truth label 2: {voxels} voxels, none predicted", measurements=[mid]
    )
    assert right["observation"]["kind"] == "measured_property" and not right["observation"]["notes"]
    # a measurement taken elsewhere is context for a suspicion (noted), never the ground of a confirmation
    elsewhere = run(box, "note", **error, views=[montage["id"]], measurements=[mid])["observation"]
    assert elsewhere["kind"] == "suspected_error"
    assert any(note == "measurement(s) taken outside this box, cited as context: " + mid for note in elsewhere["notes"])
    # an estimate read from a view stays, marked
    estimate = run(
        box, "note", **{**error, "description": "a blob about 6 mm across"}, views=[montage["id"]], measurements=[]
    )
    assert any("read from the views" in note and "6 mm" in note for note in estimate["observation"]["notes"])
    # a confirmation needs a value against a reference: None proves nothing, by definition or otherwise
    box.measurements.append(Measurement("m099", "konfai_eval", True, cube, cube, {"PREDICTION:TRUTH:Dice:2": None}))
    confirmed = {**measured, "kind": "confirmed_error", "description": "the cube is missing"}
    empty = run(box, "note", **confirmed, measurements=["m099"])["observation"]
    assert empty["kind"] == "suspected_error" and any("hold no value" in note for note in empty["notes"])
    held = run(box, "note", **confirmed, measurements=[mid])["observation"]
    assert held["kind"] == "confirmed_error"  # the statistics compare prediction and truth voxels: a reference
    assert not _defect_by_definition(Measurement("m1", "konfai_eval", False, None, None, {"F:JacobianFolding": None}))
    assert _defect_by_definition(Measurement("m2", "konfai_eval", False, None, None, {"F:JacobianFolding": 0.01}))
    assert _defect_by_definition(Measurement("m3", "konfai_eval", False, None, None, {"F:JacobianMin": float("nan")}))
    assert [c.ok for c in box.calls].count(False) == 4
    far = run(box, "region_stats", box_mm=blob)["measurement"]["id"]
    assert "measurement of the region" in run(box, "note", **confirmed, measurements=[far])["error"]
    # the figures of the case card (spacing, size, bounds) and scientific notation are sourced; "label 2" is no figure
    box.measurements.append(Measurement("m098", "konfai_eval", True, cube, cube, {"MOVED:MOVING:TRE": 5.563e-05}))
    spoken = run(
        box,
        "note",
        **{
            **measured,
            "description": f"on the 1 x 1 x 2 mm grid, at z = {zoom['position_mm'][2]} mm, TRE = 5.563e-05 mm; "
            "Dice for label 2 is fine",
        },
        measurements=["m098"],
    )
    assert spoken["observation"]["kind"] == "measured_property" and not spoken["observation"]["notes"]


def test_a_figure_is_held_to_the_precision_written():
    from konfai_scout.report import Measurement

    assert _unsourced_figures("Dice = 0.7", [], None, [1.0]) == ["Dice = 0.7"]  # a spacing does not source a metric
    assert _unsourced_figures("-3.2 mm", [], None, [-3.0]) == ["-3.2 mm"]
    cited = Measurement("m1", "konfai_eval", True, None, None, {"A:B:TRE": 5.563e-05, "A:B:Dice": 0.7312})
    assert not _unsourced_figures("Dice = 0.73, TRE = 5.563e-05 mm", [cited], None)
