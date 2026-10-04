# SPDX-License-Identifier: Apache-2.0
"""The benchmark refuses incomplete evidence before publishing its result."""

import importlib.util
import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PERF = Path(__file__).resolve().parents[2] / "benchmarks" / "perf"


@pytest.mark.parametrize("missing", ["held_bytes", "max_abs_diff", None])
def test_transform_bench_refuses_missing_evidence_without_writing_nan(monkeypatch, tmp_path, missing):
    monkeypatch.syspath_prepend(str(PERF))
    # Keep the benchmark's harness independent of any integration-test module with that name.
    spec = importlib.util.spec_from_file_location("harness", PERF / "harness.py")
    harness = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "harness", harness)
    spec.loader.exec_module(harness)
    main = runpy.run_path(str(PERF / "bench_transform.py"))["main"]
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
