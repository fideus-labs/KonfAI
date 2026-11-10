# SPDX-License-Identifier: Apache-2.0
"""The KonfAI Eval bridge: prior reports are read; a region is measured by KonfAI itself when it is installed."""

import json

import numpy as np
import pytest
from konfai_scout.config import ScoutConfig
from konfai_scout.derived import Derived
from konfai_scout.evaluation import EvaluationError, RegionEvaluator, load_prior
from konfai_scout.planes import box_from_dict, window_of


def test_load_prior_reads_the_case_values(tmp_path):
    report = {
        "aggregates": {"PRED:TRUTH:Dice": {"mean": 0.8}},
        "case": {"PRED:TRUTH:Dice": {"c1": 0.7, "c2": 0.9}, "PRED:TRUTH:Dice:1": {"c1": None}},
    }
    (tmp_path / "Metric_TRAIN.json").write_text(json.dumps(report))
    assert load_prior([tmp_path / "Metric_TRAIN.json"], "c1") == {"PRED:TRUTH:Dice": 0.7, "PRED:TRUTH:Dice:1": None}
    assert load_prior([tmp_path / "Metric_TRAIN.json"], "pair/c2") == {}
    assert load_prior({"x": 1.0}, "c1") == {"x": 1.0} and load_prior(None, "c1") == {}
    assert "_unreadable:missing.json" in load_prior([tmp_path / "missing.json"], "c1")


def test_a_region_dice_is_konfais_dice_of_the_crop(fixtures, tmp_path):
    case, defects = fixtures["segmentation"]
    case.stage(tmp_path / "stage")
    evaluator = RegionEvaluator(case, ScoutConfig(), tmp_path, None)
    box = box_from_dict(defects["missing_cube"]["box_mm"])
    measurement = evaluator.measure(box)
    assert measurement.reference and measurement.kind == "konfai_eval"
    window = window_of(case.grid, box)
    lo, hi = tuple(w.start for w in window), tuple(w.stop for w in window)
    pred = case.volume("prediction").region(lo, hi)
    truth = case.volume("truth").region(lo, hi)
    with pytest.raises(EvaluationError, match="no voxel"):
        evaluator.measure(box_from_dict({"min": [900, 900, 900], "max": [901, 901, 901]}))
    for label in (1, 2):
        key = f"PRED:TRUTH:Dice:{label}"
        if key not in measurement.values:
            continue
        a, b = pred == label, truth == label
        expected = 2 * (a & b).sum() / max(a.sum() + b.sum(), 1)
        if measurement.values[key] is not None:
            assert measurement.values[key] == pytest.approx(expected, abs=1e-3)
    assert measurement.values.get("PRED:TRUTH:Dice:2") in (None, pytest.approx(0.0, abs=1e-6))
    assert (tmp_path / measurement.workspace / "Evaluations" / "region" / "Metric_TRAIN.json").exists()


def test_a_field_region_is_scored_without_a_reference(fixtures, tmp_path):
    case, defects = fixtures["registration_no_ref"]
    case.stage(tmp_path / "stage")
    derived = Derived(case, tmp_path / "work")
    evaluator = RegionEvaluator(case, ScoutConfig(), tmp_path, derived)
    measurement = evaluator.measure(box_from_dict(defects["zeroed_field"]["box_mm"]))
    assert not measurement.reference
    keys = set(measurement.values)
    assert any("JacobianFolding" in k for k in keys) and any("Displacement" in k for k in keys)
    selected = evaluator.measure(case.grid.world_box(), metrics=["MAE"])
    assert all(k.endswith("MAE") for k in selected.values) and selected.values


def test_registration_references_are_scored_through_the_field(fixtures, tmp_path):
    case, _ = fixtures["registration"]
    case.stage(tmp_path / "stage")
    derived = Derived(case, tmp_path / "work")
    evaluator = RegionEvaluator(case, ScoutConfig(), tmp_path, derived)
    measurement = evaluator.measure(case.grid.world_box(), metrics=["Dice", "TRE"])
    assert measurement.reference
    dice = [v for k, v in measurement.values.items() if k == "WARPED_LABELS:FIXED_LABELS:Dice"]
    assert dice and dice[0] > 0.8  # the field is exact outside the planted errors
    tre = [v for k, v in measurement.values.items() if k == "MOVED_LANDMARKS:MOVING_LANDMARKS:TRE"]
    assert tre and tre[0] < 1.0
    moved = derived.moved_landmarks()
    assert moved is not None and np.isfinite(moved).all()


def test_criteria_are_taken_by_name_or_by_result_key():
    from konfai_scout.evaluation import _criterion_name

    criteria = {"Dice": {}, "MAE": {}, "Displacement": {}, "Displacement#2": {}, "JacobianFolding": {}, "SSIM": {}}
    keys = ["WARPED:FIXED:MAE", "FIELD:None:DisplacementP95", "PRED:TRUTH:Dice:1"]
    assert list(RegionEvaluator._selected(criteria, keys)) == ["Dice", "MAE", "Displacement", "Displacement#2"]
    assert list(RegionEvaluator._selected(criteria, ["ssim", "Displacement#2"])) == ["Displacement#2", "SSIM"]
    assert RegionEvaluator._selected(criteria, None) is criteria
    assert _criterion_name("PRED:TRUTH:Dice:1") == "dice" and _criterion_name("MAE") == "mae"


def test_a_thin_box_drops_ssim_and_an_unknown_metric_names_the_criteria(fixtures, tmp_path):
    pytest.importorskip("konfai")
    case, _ = fixtures["synthesis"]
    case.stage(tmp_path / "stage")
    evaluator = RegionEvaluator(case, ScoutConfig(), tmp_path / "work", None)
    assert evaluator.available() == ["MAE", "ME", "PSNR", "SSIM"]
    whole = case.grid.world_box()
    thin = box_from_dict(
        {"min": list(whole.low_xyz), "max": [whole.high_xyz[0], whole.high_xyz[1], whole.low_xyz[2] + 3.0]}
    )
    measurement = evaluator.measure(thin)
    assert "SYN:TARGET:MAE" in measurement.values and not any("SSIM" in k for k in measurement.values)
    assert measurement.request["dropped"] == ["SSIM: the box is thinner than its 7-voxel window"]
    with pytest.raises(EvaluationError, match="criterion names, for this case: MAE, ME, PSNR, SSIM"):
        evaluator.measure(whole, ["SYN:TARGET:Dice"])
    assert "SYN:TARGET:MAE" in evaluator.measure(whole, ["SYN:TARGET:MAE"]).values
