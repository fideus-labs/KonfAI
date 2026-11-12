# SPDX-License-Identifier: Apache-2.0
"""The survey: its leads cover every planted defect and leave a clean case alone; `conclude` waits for them and keeps
the verdict in line with the observations."""

import pytest
from konfai_scout.planes import box_from_dict, box_intersection
from konfai_scout.survey import survey
from konfai_scout.synthetic import MAKERS
from test_tools import run, toolbox


@pytest.mark.parametrize("task", list(MAKERS))
def test_the_leads_cover_the_planted_defects_and_spare_a_clean_case(task, tmp_path):
    planted_case, planted = MAKERS[task](tmp_path / "planted")
    clean_case, _ = MAKERS[task](tmp_path / "clean", clean=True)
    fixtures = {"planted": (planted_case, planted), "clean": (clean_case, {})}
    leads = survey(toolbox(fixtures, "planted", tmp_path / "p").renderer)
    for name, defect in planted.items():
        if defect.get("box_mm"):  # a defect planted without a box (a blur) is everywhere, no lead to point at
            box = box_from_dict(defect["box_mm"])
            assert any(box_intersection(box_from_dict(lead.box_mm), box) for lead in leads), name
    assert survey(toolbox(fixtures, "clean", tmp_path / "c").renderer) == []


def test_a_map_of_more_than_512_slices_is_surveyed_label_by_label(tmp_path):
    import numpy as np
    from konfai_scout.survey import _extents
    from konfai_scout.synthetic import segmentation_case

    case, planted = segmentation_case(tmp_path / "long", shape=(520, 48, 48), with_truth=False)
    renderer = toolbox({"long": (case, planted)}, "long", tmp_path / "l").renderer
    whole = renderer.volume("prediction").region((0, 0, 0), (520, 48, 48))[0]
    for label, (lo, hi) in _extents(renderer.volume("prediction"), (520, 48, 48)).items():  # the sphere spans 2 slabs
        where = np.argwhere(whole == label)
        assert lo.tolist() == where.min(0).tolist() and hi.tolist() == (where.max(0) + 1).tolist()
    blob = box_from_dict(planted["spurious_blob"]["box_mm"])
    assert any(box_intersection(box_from_dict(lead.box_mm), blob) for lead in survey(renderer))


def test_conclude_waits_for_the_leads_and_follows_the_observations(fixtures, tmp_path):
    box = toolbox(fixtures, "segmentation", tmp_path)
    box.leads = survey(box.renderer)
    assert box.leads
    conclude = {"summary": "s", "confidence": 0.6}
    wordy = {"summary": "word " * 51, "confidence": 0.5}
    assert "50 words or fewer" in run(box, "conclude", verdict="acceptable", **wordy)["error"]
    assert "not looked at" in run(box, "conclude", verdict="insufficient_evidence", **conclude)["error"]
    views = [run(box, "view", plane="axial", mode="prediction", zoom=lead.box_mm)["view"]["id"] for lead in box.leads]
    wordy_note = {"kind": "suspected_error", "defect": "leak", "severity": "minor", "confidence": 0.5}
    assert "70 words or fewer" in run(box, "note", **wordy_note, description="word " * 71, views=views[:1])["error"]
    assert "rests on a major or critical error" in run(box, "conclude", verdict="defective", **conclude)["error"]
    error = {"defect": "spurious_structure", "severity": "major", "description": "a blob", "plane": "axial"}
    lead = box.leads[0]
    run(
        box,
        "note",
        kind="suspected_error",
        confidence=0.8,
        box_mm=lead.id,  # a lead's box by its id, not coordinates read off a view
        views=views[:1],
        measurements=[],
        **error,
    )
    assert box.observations[0].box_mm == lead.box_mm
    assert (
        "no lead"
        in run(box, "note", kind="suspected_error", confidence=0.8, box_mm="l42", views=views[:1], **error)["error"]
    )
    for verdict in ("acceptable", "insufficient_evidence"):  # a credible major error makes the result defective
        assert "the verdict is 'defective'" in run(box, "conclude", verdict=verdict, **conclude)["error"]
    assert run(box, "conclude", verdict="defective", **conclude)["conclusion"]["verdict"] == "defective"
    box.closing = lambda: True  # once the budget is spent a conclusion is taken, the unseen leads named
    box.leads = [*box.leads, type(lead)("l99", "component", {"min": [0, 0, 0], "max": [1, 1, 1]}, "far away")]
    assert "l99" in run(box, "conclude", verdict="defective", **conclude)["conclusion"]["note"]


def test_a_survey_that_cannot_run_says_so_rather_than_finding_nothing(tmp_path):
    import json

    from konfai_scout import ScoutConfig, inspect
    from konfai_scout.survey import SurveySkipped
    from konfai_scout.synthetic import synthesis_case
    from konfai_scout.views import Renderer

    case, _ = synthesis_case(tmp_path / "s", with_target=False)
    case.stage(tmp_path / "work" / "stage")
    with pytest.raises(SurveySkipped, match="without a target"):
        survey(Renderer(case, tmp_path / "work", ScoutConfig(max_px=200), tmp_path / "ev"))
    inspect(case, ScoutConfig(provider="mock", max_px=200), tmp_path / "out")
    card = json.loads((tmp_path / "out" / "transcript.json").read_text())[1]
    assert "no survey was run (a synthesis without a target" in json.dumps(card)


@pytest.mark.parametrize("role", ["synthetic", "target"])
def test_a_displaced_edge_is_a_lead_and_the_source_says_which_image_moved(role, tmp_path):
    import numpy as np
    import SimpleITK as sitk
    from konfai_scout.synthetic import synthesis_case

    case, _ = synthesis_case(tmp_path / "moved", clean=True)
    target = sitk.ReadImage(str(case.inputs["target"]))
    source = -target  # another contrast, the same edges: a source that can tell
    sitk.WriteImage(source, str(case.inputs["source"]))
    image = sitk.ReadImage(str(case.inputs[role]))
    moved = sitk.GetImageFromArray(np.roll(sitk.GetArrayFromImage(image), 3, axis=2))  # 3 mm along x
    moved.CopyInformation(image)
    sitk.WriteImage(moved, str(case.inputs[role]))
    leads = survey(toolbox({"moved": (case, {})}, "moved", tmp_path / "m").renderer)
    assert any(lead.kind == "displacement" and f"points at the {role}" in lead.reason for lead in leads)
