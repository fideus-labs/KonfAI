# SPDX-License-Identifier: Apache-2.0
"""The KonfAI Eval bridge: prior reports are read; a region is measured by KonfAI itself when it is installed."""

import json

import numpy as np
import pytest
from konfai_scout.config import ScoutConfig
from konfai_scout.evaluation import EvaluationError, RegionEvaluator, konfai_available, load_prior
from konfai_scout.geometry import WorldBox
from konfai_scout.warp import Derived


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


def test_without_konfai_the_measurement_says_so(fixtures, tmp_path):
    case, _ = fixtures["segmentation"]
    evaluator = RegionEvaluator(case, ScoutConfig(), tmp_path, None)
    if konfai_available():
        pytest.skip("konfai is installed")
    with pytest.raises(EvaluationError, match="not installed"):
        evaluator.measure(case.grid.bounds)


@pytest.mark.skipif(not konfai_available(), reason="konfai is not installed")
def test_a_region_dice_is_konfais_dice_of_the_crop(fixtures, tmp_path):
    case, defects = fixtures["segmentation"]
    evaluator = RegionEvaluator(case, ScoutConfig(), tmp_path, None)
    box = WorldBox.from_dict(defects["missing_cube"]["box_mm"])
    measurement = evaluator.measure(box)
    assert measurement.reference and measurement.kind == "konfai_eval"
    lo, hi = box.index_box(case.grid)
    pred = case.volume("prediction").region(lo, hi)
    truth = case.volume("truth").region(lo, hi)
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


@pytest.mark.skipif(not konfai_available(), reason="konfai is not installed")
def test_a_field_region_is_scored_without_a_reference(fixtures, tmp_path):
    case, defects = fixtures["registration_no_ref"]
    derived = Derived(case, tmp_path / "derived")
    evaluator = RegionEvaluator(case, ScoutConfig(), tmp_path, derived)
    measurement = evaluator.measure(WorldBox.from_dict(defects["zeroed_field"]["box_mm"]))
    assert not measurement.reference
    keys = set(measurement.values)
    assert any("JacobianFolding" in k for k in keys) and any("Displacement" in k for k in keys)
    selected = evaluator.measure(case.grid.bounds, metrics=["MAE"])
    assert all(k.endswith("MAE") for k in selected.values) and selected.values


@pytest.mark.skipif(not konfai_available(), reason="konfai is not installed")
def test_registration_references_are_scored_through_the_field(fixtures, tmp_path):
    case, _ = fixtures["registration"]
    derived = Derived(case, tmp_path / "derived")
    evaluator = RegionEvaluator(case, ScoutConfig(), tmp_path, derived)
    measurement = evaluator.measure(case.grid.bounds, metrics=["Dice", "TRE"])
    assert measurement.reference
    dice = [v for k, v in measurement.values.items() if k == "WARPED_LABELS:FIXED_LABELS:Dice"]
    assert dice and dice[0] > 0.8  # the field is exact outside the planted errors
    tre = [v for k, v in measurement.values.items() if k == "MOVED_LANDMARKS:MOVING_LANDMARKS:TRE"]
    assert tre and tre[0] < 1.0
    moved = derived.moved_landmarks()
    assert moved is not None and np.isfinite(moved).all()
