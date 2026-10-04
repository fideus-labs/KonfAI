# SPDX-License-Identifier: Apache-2.0
"""The release gate rejects regressions and incomplete or incomparable evidence."""

import copy
import importlib.util
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


@pytest.mark.parametrize(
    "metric",
    ["wall_s_whole", "peak_rss_gib_whole", "konfai_wall_s_b1", "forward_ms", "mha_vector_write_ms", "gzip_reads_ms"],
)
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


def load_harness(monkeypatch):
    """The real harness, under its own name only for this test."""
    spec = importlib.util.spec_from_file_location("harness", PERF / "harness.py")
    harness = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "harness", harness)
    spec.loader.exec_module(harness)
    return harness


def perf_script(monkeypatch, name):
    """The globals of a bench script."""
    load_harness(monkeypatch)
    return runpy.run_path(str(PERF / f"{name}.py"))


@pytest.mark.parametrize("platform,held_bytes", [("linux", None), ("darwin", None), ("linux", 1024)])
def test_stage_bench_requires_memory_evidence_on_linux(monkeypatch, platform, held_bytes):
    main = perf_script(monkeypatch, "bench_stages")["main"]
    scope = main.__globals__
    name = next(iter(scope["SCENARIOS"]))
    monkeypatch.setattr(sys, "argv", ["bench_stages", "--quick", "--only", name])
    monkeypatch.setitem(scope, "sys", SimpleNamespace(platform=platform))
    monkeypatch.setitem(scope, "machine_gate", lambda **_: SimpleNamespace(warnings=[]))
    monkeypatch.setitem(scope, "fingerprint", lambda: {})
    monkeypatch.setitem(
        scope,
        "in_fresh_process",
        lambda *_: {"walls_ms": [1], "held_bytes": held_bytes, "volume_bytes": 1024, "differing_voxels": 0},
    )
    written = []
    monkeypatch.setitem(scope, "write_result", lambda _, result, **__: written.append(result))
    if platform == "linux" and held_bytes is None:
        with pytest.raises(SystemExit, match="held-memory measurement"):
            main()
        assert written == []
    else:
        main()
        entries = {entry["name"]: entry for entry in written[0]["facts"]}
        assert entries[f"{name}_differing_voxels"]["value"] == 0
        assert (f"{name}_held_volumes" in entries) == (held_bytes is not None)


@pytest.mark.parametrize("missing", ["held_bytes", "max_abs_diff", None])
def test_transform_bench_refuses_missing_evidence_without_writing_nan(monkeypatch, tmp_path, missing):
    monkeypatch.syspath_prepend(str(PERF))
    main = perf_script(monkeypatch, "bench_transform")["main"]
    scope = main.__globals__
    monkeypatch.setattr(sys, "argv", ["bench_transform", "--quick"])
    monkeypatch.setitem(scope, "machine_gate", lambda **_: SimpleNamespace(warnings=[]))
    monkeypatch.setitem(scope, "fingerprint", lambda: {})
    monkeypatch.setitem(scope, "tempfile", SimpleNamespace(mkdtemp=lambda **_: str(tmp_path)))
    monkeypatch.setitem(scope, "synthesize", lambda *_: (tmp_path / "input.h5", [1, 2, 3]))
    run = {"wall_s": 1, "peak_bytes": 1024, "floor_bytes": 512, "held_bytes": 512}
    if missing == "held_bytes":
        run["held_bytes"] = None
    monkeypatch.setitem(scope, "in_fresh_process", lambda *_: run)
    monkeypatch.setitem(scope, "find_output", lambda *_: tmp_path / "output.h5")
    check = {"shape_equal": True, "attrs_equal": True, "max_abs_diff": 0}
    if missing == "max_abs_diff":
        check = {"shape_equal": False, "a": [1, 2, 3], "b": [1, 2, 4]}
    monkeypatch.setitem(scope, "compare_h5", lambda *_: check)
    monkeypatch.setitem(
        scope,
        "cprofile_summary",
        lambda *_, **__: {"family_share_of_tottime": {"statistics": 0}, "family_max_cumtime_s": {"statistics": 0}},
    )
    written = []
    monkeypatch.setitem(
        scope, "write_result", lambda _, result, **__: written.append(json.dumps(result, allow_nan=False))
    )
    if missing:
        with pytest.raises(SystemExit, match=r"measurement|shapes"):
            main()
        assert written == []
    else:
        main()
        entries = {entry["name"]: entry for entry in json.loads(written[0])["facts"]}
        assert entries["sweep_peak_gib_b1"]["value"] == 512 / 2**30


def konfai_from(monkeypatch, root):
    """The environment imports konfai from ``root``."""
    monkeypatch.syspath_prepend(str(root))
    monkeypatch.delitem(sys.modules, "konfai", raising=False)


def test_the_gate_refuses_to_time_a_konfai_imported_from_another_tree(monkeypatch, tmp_path):
    harness = load_harness(monkeypatch)
    # os.getloadavg is Unix-only; the Windows lane runs this test too.
    monkeypatch.setattr(harness.os, "getloadavg", lambda: (0.0, 0.0, 0.0), raising=False)
    monkeypatch.setattr(harness, "power_profile", lambda: "performance")
    monkeypatch.setattr(harness, "gpu_busy", lambda: None)
    monkeypatch.setattr(harness, "cpu_hogs", lambda: [])
    (tmp_path / "konfai").mkdir()
    (tmp_path / "konfai" / "__init__.py").write_text("")
    konfai_from(monkeypatch, tmp_path)
    with pytest.raises(SystemExit, match="konfai"):
        harness.machine_gate()
    assert any(str(tmp_path) in warning for warning in harness.machine_gate(force=True).warnings)
    konfai_from(monkeypatch, harness.REPO)
    assert harness.machine_gate().quiet


def importtime_line(self_us, cumulative_us, level, name):
    # CPython's own format: "import time: %9ld | %10ld | %*s%s" with two spaces per import level.
    return f"import time: {self_us:9d} | {cumulative_us:10d} | {'  ' * level}{name}"


def test_import_time_is_split_by_the_top_level_packages_the_import_reaches(monkeypatch):
    lines = [
        "import time: self [us] | cumulative | imported package",
        importtime_line(603, 1128, 0, "_frozen_importlib_external"),
        importtime_line(100, 100, 2, "konfai.utils"),
        importtime_line(3715, 99268, 1, "konfai"),
        importtime_line(1440, 55378, 1, "numpy"),
        importtime_line(323, 323, 2, "ruamel"),
        importtime_line(600, 28749, 1, "ruamel.yaml"),
        importtime_line(50, 50, 2, "torch._C"),
        importtime_line(253562, 1507770, 1, "torch"),
        importtime_line(1183, 170529, 3, "SimpleITK"),
        importtime_line(500, 200000, 2, "konfai.data.dataset"),
        importtime_line(90, 448495, 1, "konfai.data.data_manager"),
        importtime_line(56, 240838, 1, "torch.utils.tensorboard.writer"),
        importtime_line(3005, 2407461, 0, "konfai.trainer"),
    ]
    bench = perf_script(monkeypatch, "bench_startup")
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stderr="\n".join(lines)))
    assert bench["importtime"]("konfai.trainer") == {
        "konfai": 2407.5,
        "numpy": 55.4,
        "ruamel": 28.7,
        "torch": 1748.6,
        "SimpleITK": 170.5,
    }


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
