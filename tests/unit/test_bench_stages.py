# SPDX-License-Identifier: Apache-2.0
"""The benchmark refuses incomplete evidence before publishing its result."""

import importlib.util
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PERF = Path(__file__).resolve().parents[2] / "benchmarks" / "perf"


@pytest.mark.parametrize("platform,held_bytes", [("linux", None), ("darwin", None), ("linux", 1024)])
def test_stage_bench_requires_memory_evidence_on_linux(monkeypatch, platform, held_bytes):
    # Keep the benchmark's harness independent of any integration-test module with that name.
    spec = importlib.util.spec_from_file_location("harness", PERF / "harness.py")
    harness = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "harness", harness)
    spec.loader.exec_module(harness)
    main = runpy.run_path(str(PERF / "bench_stages.py"))["main"]
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
