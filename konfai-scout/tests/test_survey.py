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


def test_conclude_waits_for_the_leads_and_follows_the_observations(fixtures, tmp_path):
    box = toolbox(fixtures, "segmentation", tmp_path)
    box.leads = survey(box.renderer)
    assert box.leads
    conclude = {"summary": "s", "confidence": 0.6}
    assert "not looked at" in run(box, "conclude", verdict="insufficient_evidence", **conclude)["error"]
    views = [run(box, "view", plane="axial", mode="prediction", zoom=lead.box_mm)["view"]["id"] for lead in box.leads]
    assert "rests on error observations" in run(box, "conclude", verdict="defective", **conclude)["error"]
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
