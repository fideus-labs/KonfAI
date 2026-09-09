# SPDX-License-Identifier: Apache-2.0
"""The release gate rejects regressions and incomplete or incomparable evidence."""

import copy
import json
import runpy
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

PERF = Path(__file__).resolve().parents[2] / "benchmarks" / "perf"


def result():
    return {
        "fingerprint": {"gpu": {"name": "test"}, "power_profile": "performance", "threads": {"OMP_NUM_THREADS": "1"}},
        "quick": False,
        "gate_warnings": [],
        "benches": {"predict": {"result": {"metrics": {"wall_s_whole": 10, "peak_rss_gib_whole": 1}}}},
    }


def compare(tmp_path, baseline, run, extra=()):
    paths = [tmp_path / "baseline.json", tmp_path / "run.json"]
    for path, payload in zip(paths, [baseline, run], strict=True):
        path.write_text(json.dumps(payload))
    return subprocess.run(
        [sys.executable, str(PERF / "compare.py"), *map(str, paths), *extra], capture_output=True, text=True
    )


@pytest.mark.parametrize("metric", ["wall_s_whole", "peak_rss_gib_whole", "konfai_wall_s_b1", "forward_ms"])
def test_actual_workflow_metric_names_fail_on_regression(tmp_path, metric):
    baseline = result()
    baseline["benches"]["predict"]["result"]["metrics"] = {metric: 1}
    run = copy.deepcopy(baseline)
    run["benches"]["predict"]["result"]["metrics"][metric] = 10
    completed = compare(tmp_path, baseline, run)
    assert completed.returncode == 1, completed.stdout
    assert "REGRESSION" in completed.stdout


@pytest.mark.parametrize(
    "invalid",
    ["missing", "failed", "empty", "gpu", "quick", "busy", "nan", "negative", "bench_busy", "bench_quick", "bench_gpu"],
)
def test_incomplete_or_incomparable_measurements_cannot_certify_a_release(tmp_path, invalid):
    baseline = result()
    run = copy.deepcopy(baseline)
    if invalid == "missing":
        del run["benches"]["predict"]["result"]["metrics"]["wall_s_whole"]
    elif invalid == "failed":
        run["benches"]["predict"] = {"status": "failed", "returncode": 1}
    elif invalid == "empty":
        run["benches"] = {}
    elif invalid == "gpu":
        run["fingerprint"]["gpu"]["name"] = "other"
    elif invalid == "quick":
        run["quick"] = True
    elif invalid == "busy":
        run["gate_warnings"] = ["busy machine"]
    elif invalid == "bench_busy":
        run["benches"]["predict"]["result"]["gate_warnings"] = ["machine became busy during series"]
    elif invalid == "bench_quick":
        run["benches"]["predict"]["result"]["quick"] = True
    elif invalid == "bench_gpu":
        baseline["benches"]["predict"]["fingerprint"] = copy.deepcopy(baseline["fingerprint"])
        run["benches"]["predict"]["fingerprint"] = copy.deepcopy(run["fingerprint"])
        run["benches"]["predict"]["fingerprint"]["gpu"]["name"] = "other"
    else:
        run["benches"]["predict"]["result"]["metrics"]["wall_s_whole"] = float("nan") if invalid == "nan" else -1
    completed = compare(tmp_path, baseline, run)
    assert completed.returncode == 2, completed.stdout


def facts(tmp_path, doc):
    path = tmp_path / "run.json"
    path.write_text(json.dumps(doc))
    return subprocess.run([sys.executable, str(PERF / "facts.py"), str(path)], capture_output=True, text=True)


def test_a_violated_fact_fails_whatever_the_machine(tmp_path):
    doc = result()
    doc["benches"]["predict"]["result"]["facts"] = [
        {"name": "differing_voxels_whole_vs_stream", "value": 3.0, "expect": 0.0, "op": "=="},
        {"name": "geometry_identical_whole_vs_stream", "value": 1.0, "expect": 1.0, "op": "=="},
    ]
    completed = facts(tmp_path, doc)
    assert completed.returncode == 1, completed.stdout
    assert "VIOLATED" in completed.stdout and "1 violation(s) over 2" in completed.stdout


def test_facts_that_hold_pass_and_a_failed_bench_is_a_violation(tmp_path):
    doc = result()
    doc["benches"]["predict"]["result"]["facts"] = [{"name": "max_abs_diff", "value": 0.0, "expect": 0.0, "op": "<="}]
    assert facts(tmp_path, doc).returncode == 0
    doc["benches"]["transform"] = {"status": "failed", "returncode": 1}
    assert facts(tmp_path, doc).returncode == 1


def test_a_run_without_facts_cannot_pass(tmp_path):
    assert facts(tmp_path, result()).returncode == 2


def test_comparable_equal_results_including_zero_time_pass(tmp_path):
    baseline = result()
    baseline["benches"]["predict"]["result"]["metrics"]["startup_s"] = 0
    assert compare(tmp_path, baseline, baseline).returncode == 0


def test_metrics_without_machine_fingerprints_cannot_certify_a_release(tmp_path):
    baseline = result()
    del baseline["fingerprint"]
    assert compare(tmp_path, baseline, baseline).returncode == 2


@pytest.mark.parametrize("option", ["--time-tolerance", "--memory-tolerance"])
@pytest.mark.parametrize("invalid", ["nan", "inf", "-0.1"])
def test_invalid_tolerances_cannot_disable_the_gate(tmp_path, option, invalid):
    baseline = result()
    assert compare(tmp_path, baseline, baseline, [option, invalid]).returncode == 2


@pytest.mark.parametrize("returncode", [0, 1])
def test_series_fails_if_a_scenario_fails_or_produces_no_result(tmp_path, monkeypatch, returncode):
    # The integration suite also imports a module named harness. Keep this test's hardware stubs
    # local and restore that module afterwards instead of depending on collection order.
    harness = ModuleType("harness")
    monkeypatch.setitem(sys.modules, "harness", harness)
    monkeypatch.setattr(harness, "PERF_DIR", PERF, raising=False)

    fp = {
        **result()["fingerprint"],
        "git": {"sha": "abc", "describe": "test", "dirty": False},
        "host": "test",
        "date": "test",
        "load_avg": [0],
    }
    monkeypatch.setattr(harness, "machine_gate", lambda **kwargs: SimpleNamespace(warnings=[]), raising=False)
    monkeypatch.setattr(harness, "fingerprint", lambda: fp, raising=False)
    monkeypatch.setattr(harness, "results_dir", lambda: tmp_path, raising=False)
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=returncode))
    monkeypatch.setattr(sys, "argv", ["run_all.py", "--only", "predict"])
    monkeypatch.delenv("KONFAI_PERF_GATED", raising=False)
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(PERF / "run_all.py"), run_name="__main__")
    assert exc.value.code == 1
    written = json.loads(next(tmp_path.glob("*-all.json")).read_text())
    assert written["benches"]["predict"]["status"] == "failed"
