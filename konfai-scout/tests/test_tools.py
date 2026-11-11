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


def unsure_reader():
    """A blind reader that cannot tell, for a test whose scripted client plays the inspecting model only."""
    from konfai_scout.providers.base import Turn, Usage
    from konfai_scout.providers.mock import ScriptedProvider

    return ScriptedProvider([Turn(text="UNSURE\nNothing decides.", tool_calls=[], stop="end_turn", usage=Usage(0, 0))])


def run(box: Toolbox, name: str, **arguments) -> dict:
    result = box.call(ToolCall(id=f"c{len(box.calls)}", name=name, arguments=arguments))
    return {"error": result.text} if result.is_error else json.loads(result.text)


def test_region_stats_see_the_planted_defects(fixtures, tmp_path):
    case, defects = fixtures["segmentation"]
    box = toolbox(fixtures, "segmentation", tmp_path)
    whole = run(box, "region_stats", box_mm=box_to_dict(case.grid.world_box()))["region"]
    assert whole["prediction_labels"]["1"]["components"] == 2  # the sphere and the spurious blob
    assert "2" not in whole["prediction_labels"] and whole["truth_labels"]["2"]["name"] == "cube"
    cube = run(box, "region_stats", box_mm=defects["missing_cube"]["box_mm"])["region"]
    assert cube["truth_labels"]["2"]["voxels"] == whole["truth_labels"]["2"]["voxels"]


def test_notes_need_known_evidence_and_confirmation_needs_a_measurement(fixtures, tmp_path):
    box = toolbox(fixtures, "segmentation_clean", tmp_path)  # Scout's own check of a box finds nothing wrong
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
        severity="major",  # a major error, so that the defective conclusion below holds
        confidence=0.5,
        description="x",
        box_mm=view["box_mm"],
        views=[view["id"]],
        measurements=[],
    )["observation"]
    assert downgraded["kind"] == "suspected_error" and downgraded["notes"]
    nowhere = {"kind": "suspected_error", "severity": "minor", "confidence": 0.5, "description": "x"}
    assert "says where" in run(box, "note", **nowhere, defect="leak", views=[view["id"]], measurements=[])["error"]
    whole = box_to_dict(box.case.grid.world_box())
    assert "says where" in run(box, "note", **nowhere, defect="leak", box_mm=whole, views=[view["id"]])["error"]
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
    nan = float("nan")  # what json.loads makes of an OpenAI-compatible server's NaN
    assert "confidence" in run(box, "conclude", verdict="acceptable", summary="s", confidence=nan)["error"]
    note = run(box, "note", kind="suspected_error", defect="leak", severity="minor", confidence=nan, description="x")
    assert "confidence" in note["error"]
    assert [c.ok for c in box.calls] == [False] * 5 and box.conclusion is None and not box.observations
    assert all(c.error for c in box.calls)


def test_registration_tools_without_references(fixtures, tmp_path):
    case, _ = fixtures["registration_no_ref"]
    box = toolbox(fixtures, "registration_no_ref", tmp_path)
    stats = run(box, "region_stats", box_mm=box_to_dict(case.grid.world_box()))["region"]
    assert "fixed" in stats and "warped" in stats and "fixed_labels" not in stats  # a field's lengths are KonfAI Eval's
    view = run(box, "view", plane="sagittal", mode="jacobian")["view"]
    assert "scale" in view and view["scale"]["folded_pixels"] == 0
    specs = {t.name for t in box.specs(exploration=True)}
    assert specs == {"view", "montage", "measure", "region_stats", "note", "conclude"}
    assert {t.name for t in box.specs(exploration=False, final=True)} == {"note", "conclude"}


def test_the_hard_rules_of_note(fixtures, tmp_path):
    """What never enters the report: an error no cited view shows, a measurement taken elsewhere, a figure without a
    source. What is marked: an estimate, a measurement cited as context."""
    from konfai_scout.report import Measurement

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
    assert seen["kind"] == "suspected_error" and [n.split(":")[0] for n in seen["notes"]] == ["Scout's check m001"]
    # a figure needs a source: the measurement that holds it, cited
    stats = run(box, "region_stats", box_mm=cube)
    mid, voxels = stats["measurement"]["id"], stats["region"]["truth_labels"]["2"]["voxels"]
    named = {"kind": "measured_property", "defect": "missing_structure", "severity": "minor", "confidence": 0.9}
    assert (
        "not a defect"
        in run(box, "note", **named, description="x", box_mm=cube, plane=None, views=[], measurements=[mid])["error"]
    )
    measured = {
        "kind": "measured_property",
        "defect": "other",
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
    assert [c.ok for c in box.calls].count(False) == 5
    confirmed = {**measured, "kind": "confirmed_error", "description": "the cube is missing"}
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


def test_a_confirmation_rests_on_a_value_past_its_threshold_and_scout_checks_every_flagged_box(fixtures, tmp_path):
    """A cited value confirms an error only past its criterion's threshold, in the bad direction; Scout measures every
    flagged box itself, records it on the observation, and downgrades a confirmation nothing measured supports."""
    from konfai_scout.tools import _past

    cube = fixtures["segmentation"][1]["missing_cube"]["box_mm"]
    box = toolbox(fixtures, "segmentation_clean", tmp_path / "clean")  # the prediction is the truth
    view = run(box, "view", plane="axial", mode="both", zoom=cube)["view"]
    perfect = run(box, "measure", box_mm=cube, metrics=None)["measurement"]
    assert perfect["values"]["PRED:TRUTH:Dice:2"] > 0.99
    claim = {
        "kind": "confirmed_error",
        "defect": "boundary_error",
        "severity": "major",
        "confidence": 0.9,
        "description": "the cube is wrong",
        "box_mm": cube,
        "plane": "axial",
        "views": [view["id"]],
    }
    wrong = run(box, "note", **claim, measurements=[perfect["id"]])["observation"]
    assert wrong["kind"] == "suspected_error"  # a perfect Dice confirms nothing
    assert "Scout's check m002: no value past a threshold" in wrong["notes"]
    assert any(n.startswith("downgraded from confirmed_error") for n in wrong["notes"])
    observation, check = box.observations[-1], box.measurements[-1]
    assert observation.measurements == [perfect["id"], "m002"] and check.request["check_of"] == observation.id
    assert box.calls[-1].measurements == ["m002"]

    box = toolbox(fixtures, "segmentation", tmp_path / "planted")
    view = run(box, "view", plane="axial", mode="both", zoom=cube)["view"]
    missing = run(box, "note", **{**claim, "views": [view["id"]]}, measurements=[])["observation"]
    assert missing["kind"] == "confirmed_error"  # nothing cited, but Scout's own check of the box shows Dice:2 = 0
    assert any(n.startswith("Scout's check m001: Dice:2 = ") and "< 0.8" in n for n in missing["notes"])

    box = toolbox(fixtures, "segmentation_no_truth", tmp_path / "no_truth")
    view = run(box, "view", plane="axial", mode="prediction", zoom=cube)["view"]
    blind = run(box, "note", **{**claim, "views": [view["id"]]}, measurements=[])["observation"]
    assert blind["kind"] == "suspected_error" and any("nothing to measure" in n for n in blind["notes"])

    thresholds = {"Dice": {"below": 0.8}, "JacobianFolding": {"above": 0.0}, "JacobianMin": {"below": 0.0}}
    assert _past("PRED:TRUTH:Dice:1", 0.95, thresholds) is None and _past("PRED:TRUTH:Dice:1", None, thresholds) is None
    assert _past("PRED:TRUTH;MASK:Dice:1", 0.3, thresholds) == "Dice:1 = 0.3 < 0.8"
    assert _past("FIELD:None:JacobianFolding", 0.01, thresholds) and not _past(
        "FIELD:None:JacobianFolding", 0, thresholds
    )
    assert _past("FIELD:None:JacobianMin", float("nan"), thresholds) == "JacobianMin is not finite"
    assert _past("SYN:TARGET:MAE", float("nan"), thresholds) is None  # an empty mask, not an error
    assert _past("SYN:TARGET:MAE", 500.0, thresholds) is None  # no threshold: the scale is the data's
    assert _past("SYN:TARGET:MAE", 500.0, {"MAE": {"above": 100}}) == "MAE = 500 > 100"


def test_a_mask_restricts_the_statistics_and_a_ct_preset_follows_a_stored_offset(fixtures, tmp_path):
    import SimpleITK as sitk
    from konfai_scout import InspectionCase

    case, _ = fixtures["synthesis"]
    target = sitk.ReadImage(str(case.inputs["target"]))
    sitk.WriteImage(sitk.BinaryThreshold(target, 0.0, 1e9), str(tmp_path / "mask.mha"))  # the sphere and the bone
    sitk.WriteImage(sitk.ReadImage(str(case.inputs["synthetic"])) + 1024.0, str(tmp_path / "hu1024.mha"))
    inputs = {**case.inputs, "mask": tmp_path / "mask.mha", "synthetic": tmp_path / "hu1024.mha"}
    masked = InspectionCase("synthesis", "masked", inputs)
    masked.stage(tmp_path / "work" / "stage")
    config = ScoutConfig(max_px=200)
    renderer = Renderer(masked, tmp_path / "work", config, tmp_path / "evidence")
    box = Toolbox(masked, config, renderer, RegionEvaluator(masked, config, tmp_path / "work", None))
    region = run(box, "region_stats", box_mm=box_to_dict(masked.grid.world_box()))["region"]
    assert 0 < region["mask_voxels"] < region["voxels"] and region["synthetic"]["voxels"] == region["mask_voxels"]
    assert renderer.resolve_window("ct_soft", "synthetic") == (864.0, 1264.0)  # stored as HU + 1024
    assert renderer.resolve_window("ct_soft", "target") == (-160.0, 240.0)  # stored in HU


def test_scouts_check_of_a_box_never_costs_the_note(fixtures, tmp_path):
    box = toolbox(fixtures, "segmentation", tmp_path)
    view = run(box, "view", plane="axial", mode="prediction")["view"]

    def broken(*args, **kwargs):
        raise RuntimeError("the evaluation crashed")

    box.evaluator.measure = broken
    note = {"kind": "suspected_error", "defect": "leak", "severity": "minor", "confidence": 0.5, "description": "x"}
    observation = run(box, "note", **note, box_mm=view["box_mm"], plane="axial", views=[view["id"]], measurements=[])
    assert "Scout's check of the box: RuntimeError: the evaluation crashed" in observation["observation"]["notes"]


def test_an_error_of_the_reference_is_noted_and_never_held_against_the_result(fixtures, tmp_path):
    error = {
        "defect": "reference_error",
        "severity": "major",
        "confidence": 0.9,
        "description": "the truth stops short",
        "box_mm": {"center": [4, 34, 29], "size": 8},
    }
    blind = toolbox(fixtures, "segmentation_no_truth", tmp_path / "blind")
    view = run(blind, "view", plane="axial", mode="prediction")["view"]["id"]
    assert "given with its reference" in run(blind, "note", kind="suspected_error", views=[view], **error)["error"]
    box = toolbox(fixtures, "segmentation_clean", tmp_path / "ref")
    view = run(box, "view", plane="axial", mode="both")["view"]["id"]
    assert "given with its reference" in run(box, "note", kind="confirmed_error", views=[view], **error)["error"]
    assert "observation" in run(box, "note", kind="suspected_error", views=[view], **error)
    assert box.serious() == []
    verdict = run(
        box, "conclude", verdict="acceptable", summary="the reference is short, the result holds", confidence=0.8
    )
    assert verdict["conclusion"]["verdict"] == "acceptable"


def test_the_measuring_tools_can_be_taken_away(fixtures, tmp_path):
    box = toolbox(fixtures, "segmentation", tmp_path)
    assert {"measure", "region_stats"} <= {tool.name for tool in box.specs(True)}
    box.config = ScoutConfig(max_px=200, measure=False)
    assert {tool.name for tool in box.specs(True)} == {"view", "montage", "note", "conclude"}
