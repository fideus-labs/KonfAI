# SPDX-License-Identifier: Apache-2.0
"""examples/calibrate_real.py on a registration with the free heuristic model: defects planted where they should be,
sensitivity, specificity and test-retest agreement written."""

import json
from pathlib import Path

import SimpleITK as sitk


def test_calibration_plants_inspects_and_scores(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    import calibrate_real as calibrate
    from konfai_scout import Report
    from konfai_scout.synthetic import registration_case

    case, _ = registration_case(tmp_path / "case", with_references=False, clean=True)  # a smooth, unfolded field
    # a registration given as an ITK transform file: sampled on the fixed grid before anything is planted
    field = sitk.Cast(sitk.ReadImage(str(case.inputs["field"])), sitk.sitkVectorFloat64)
    sitk.WriteTransform(sitk.DisplacementFieldTransform(field), str(tmp_path / "transform.h5"))
    entry = {
        "name": "c1",
        "fixed": case.inputs["fixed"],
        "moving": case.inputs["moving"],
        "field": tmp_path / "transform.h5",
    }
    (tmp_path / "cases.json").write_text(json.dumps([entry], default=str))
    calibrate.main(["--cases", str(tmp_path / "cases.json"), "--out", str(tmp_path / "out"), "--defects-per-case", "1"])

    result = json.loads((tmp_path / "out" / "calibration.json").read_text())
    planted = result["cases"]["c1"]
    assert planted["local_fold"]["min_jacobian"] < 0 < planted["local_shift"]["min_jacobian"]
    assert [(r["variant"], r["repeat"]) for r in result["inspections"]] == [
        ("clean", 0),
        ("clean", 1),
        ("local_fold", 0),
        ("local_fold", 1),
    ]
    fold = result["per_defect"]["local_fold"]  # the heuristic calls the fold it measures an implausible deformation
    assert (fold["planted_inspections"], fold["found"], fold["clean_inspections"], fold["false_alarms"]) == (2, 0, 2, 0)
    report = Report.load(tmp_path / "out" / "reports" / "c1" / "local_fold-0")
    assert report.observations and not calibrate._matched(report, planted)["local_fold"]
    for o in report.observations:
        o.defect = "folding"
    assert calibrate._matched(report, planted)["local_fold"]  # a match is of the defect's type, over its box
    assert result["per_defect"]["global_shift"]["sensitivity"] is None  # not planted with one defect per case
    assert result["test_retest"]["verdict_pairs"] == 2 and result["test_retest"]["match_pairs"] == 4
    assert result["failed"] == 0 and "| local_fold | 2 |" in (tmp_path / "out" / "calibration.md").read_text()
