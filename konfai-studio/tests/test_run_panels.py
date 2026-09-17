# Copyright (c) 2025 Valentin Boussot
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""What the right panel reads off disk: which runs it can compare, and how it names and scores them.

A run's outputs land in a different place for each kind, and these panels are where that difference
shows: the leaderboard ranks evaluations, so the config diff beside it has to find an evaluation's
config, and the scores table has to tell two evaluations of the same app apart.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")
from konfai_studio.server import _read_eval_metrics, _run_config_snapshot


@pytest.fixture(autouse=True)
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("KONFAI_MCP_WORKSPACES_ROOT", str(tmp_path))
    session = tmp_path / "sessions" / "exp"
    session.mkdir(parents=True)
    return session


def metric_file(run_dir: Path, *, directions: dict[str, str], means: dict[str, float], cases: int = 1) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "Metric_TRAIN.json").write_text(
        json.dumps(
            {
                "case": {name: {f"P{i:03d}": value for i in range(cases)} for name, value in means.items()},
                "aggregates": {name: {"mean": value, "std": 0.0} for name, value in means.items()},
                "directions": directions,
            }
        ),
        encoding="utf-8",
    )


def directions_of(runs: list[dict[str, Any]], metric: str) -> set[str]:
    return {row["direction"] for run in runs for row in run["metrics"] if row["name"] == metric}


# --- which runs the config diff can find -------------------------------------------------------------


def test_an_app_evaluation_has_a_config_the_diff_can_find(workspace: Path) -> None:
    """The leaderboard ranks evaluation runs and offers to diff their configs. An evaluation writes its
    config beside its outputs, not under Statistics/, so looking there alone answered 'no config
    snapshot found' for every run the panel itself proposed."""
    run = workspace / "AppEvaluations" / "eval_MR-9afa86c0" / "ImpactSynth"
    run.mkdir(parents=True)
    (run / "Evaluation.yml").write_text("Evaluator: {}\n", encoding="utf-8")

    assert _run_config_snapshot("exp", "eval_MR-9afa86c0") == run / "Evaluation.yml"


def test_a_training_still_resolves_to_its_statistics_snapshot(workspace: Path) -> None:
    run = workspace / "Statistics" / "FT_smoke"
    run.mkdir(parents=True)
    (run / "Config_0.yml").write_text("Trainer: {}\n", encoding="utf-8")

    assert _run_config_snapshot("exp", "FT_smoke") == run / "Config_0.yml"


def test_a_run_name_that_escapes_the_workspace_is_refused(workspace: Path) -> None:
    assert _run_config_snapshot("exp", "../secrets") is None
    assert _run_config_snapshot("exp", "a/b") is None


def test_a_metric_whose_name_carries_no_component_is_kept(workspace: Path) -> None:
    """Component rows are folded into their parent by stripping the last ``:`` segment. A name with no
    separator matched itself, so a metric declared as a bare word vanished from the table."""
    metric_file(workspace / "Evaluations" / "run", directions={"Dice": "max"}, means={"Dice": 0.9})

    assert [row["name"] for run in _read_eval_metrics("exp") for row in run["metrics"]] == ["Dice"]


def test_a_component_row_is_still_folded_into_its_parent(workspace: Path) -> None:
    metric_file(
        workspace / "Evaluations" / "run",
        directions={"out:tgt:Dice": "max"},
        means={"out:tgt:Dice": 0.9, "out:tgt:Dice:1": 0.8, "out:tgt:Dice:2": 1.0},
    )

    assert [row["name"] for run in _read_eval_metrics("exp") for row in run["metrics"]] == ["out:tgt:Dice"]


# --- how the scores table names and scores runs ------------------------------------------------------


def test_two_evaluations_of_one_app_are_told_apart(workspace: Path) -> None:
    """Named after their parent directory, every evaluation of ImpactSynth read 'ImpactSynth': four
    different runs stacked as four indistinguishable cards. The trial directory tells them apart, and
    it is the name konfai-mcp ranks them under, so the table and the leaderboard beside it agree."""
    for job in ("eval_MR-9afa86c0", "eval_MR-071a5900"):
        metric_file(workspace / "AppEvaluations" / job / "ImpactSynth", directions={"MAE": "min"}, means={"MAE": 1.0})

    assert {run["run"] for run in _read_eval_metrics("exp")} == {"eval_MR-9afa86c0", "eval_MR-071a5900"}


def test_a_session_root_evaluation_keeps_its_plain_name(workspace: Path) -> None:
    """The output root names the KIND, which the panel already says; only what identifies the run stays."""
    metric_file(workspace / "Evaluations" / "ImpactSynth", directions={"MAE": "min"}, means={"MAE": 1.0})

    assert [run["run"] for run in _read_eval_metrics("exp")] == ["ImpactSynth"]


def test_one_metric_has_one_direction_across_the_whole_session(workspace: Path) -> None:
    """A direction is a property of the criterion, not of the run. Runs predating a criterion's
    `maximize` flag declared the opposite, so the same PSNR read '↓' on one card and '↑' on the next,
    each contradicting the leaderboard beside them."""
    old = workspace / "AppEvaluations" / "eval_old" / "ImpactSynth"
    new = workspace / "AppEvaluations" / "eval_new" / "ImpactSynth"
    metric_file(old, directions={"PSNR": "min"}, means={"PSNR": 30.0})
    metric_file(new, directions={"PSNR": "max"}, means={"PSNR": 35.0})
    # The newest declaration wins. Explicit distinct mtimes: touch() can land both files on the
    # same filesystem timestamp, and a tie reads the wrong direction first.
    os.utime(old / "Metric_TRAIN.json", (1_000_000_000, 1_000_000_000))
    os.utime(new / "Metric_TRAIN.json", (1_000_000_100, 1_000_000_100))

    assert directions_of(_read_eval_metrics("exp"), "PSNR") == {"max"}


# --- what a deleted session, and a started TensorBoard, owe the caller ---------------------------------


def test_a_workspace_that_survives_deletion_is_not_forgotten(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Forgetting a directory that is still there makes the session reappear on the next listing,
    which reads as a delete that silently did nothing."""
    import konfai_studio.paths as paths
    from konfai_studio.registry import _Registry

    registry = _Registry()
    registry.load()
    assert registry.names() == ["exp"]

    monkeypatch.setattr(paths.shutil, "rmtree", lambda *a, **k: None)  # a delete that cannot complete
    assert asyncio.run(registry.remove("exp")) is False
    assert registry.names() == ["exp"]


def test_tensorboard_waits_for_its_port(monkeypatch: pytest.MonkeyPatch) -> None:
    """Popen returns long before the port listens, and a URL handed over early opens a browser error
    page that only a reload clears."""
    from konfai_studio.tensorboard import _answers

    class Proc:
        def poll(self) -> None:
            return None

    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    port = server.getsockname()[1]

    assert _answers(Proc(), port, timeout=0.3) is False  # bound but not listening
    server.listen(1)
    assert _answers(Proc(), port, timeout=2.0) is True
    server.close()


def test_tensorboard_gives_up_when_the_process_dies(monkeypatch: pytest.MonkeyPatch) -> None:
    from konfai_studio.tensorboard import _answers

    class Dead:
        def poll(self) -> int:
            return 1

    assert _answers(Dead(), 1, timeout=5.0) is False  # returns at once, not after the timeout


def test_health_survives_a_tool_registry_that_cannot_be_read(monkeypatch: pytest.MonkeyPatch) -> None:
    """The front paints the whole status bar 'offline' when /api/health fails, and the tool count is a
    caption on it: counting must never be able to take the bar down."""
    import konfai_studio.server as server
    from starlette.testclient import TestClient

    server._tool_count.cache_clear()
    monkeypatch.setattr(server.asyncio, "run", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no registry")))
    with TestClient(server.app) as client:
        payload = client.get("/api/health").json()

    assert payload["status"] == "ok" and payload["tools"] == 0
    server._tool_count.cache_clear()


def test_a_run_name_with_a_wildcard_cannot_reach_the_glob(workspace: Path) -> None:
    """The name is interpolated into a glob pattern: '*' would match every run in the session and diff
    whichever happened to be newest."""
    assert _run_config_snapshot("exp", "*") is None
    assert _run_config_snapshot("exp", "eval_?") is None
    assert _run_config_snapshot("exp", "[abc]") is None


def test_saving_a_config_that_changed_underneath_is_refused(workspace: Path) -> None:
    """The agent and every workflow write configs too (KonfAI resolves defaults back into the file).
    An editor opened before that write holds a stale text, and Save would put it back."""
    import konfai_studio.server as server
    from starlette.testclient import TestClient

    config = workspace / "Config.yml"
    config.write_text("Trainer:\n  epochs: 10\n", encoding="utf-8")
    opened = config.read_text(encoding="utf-8")

    with TestClient(server.app) as client:
        edit = {"session": "exp", "name": "Config.yml", "content": "Trainer:\n  epochs: 99\n"}
        assert client.post("/api/config/save", json={**edit, "base": opened}).status_code == 200
        assert config.read_text(encoding="utf-8") == edit["content"]

        config.write_text("Trainer:\n  epochs: 50\n", encoding="utf-8")  # the agent edits it meanwhile
        refused = client.post("/api/config/save", json={**edit, "base": opened})
        assert refused.status_code == 409
        assert config.read_text(encoding="utf-8") == "Trainer:\n  epochs: 50\n"  # the agent's edit survives

        # A caller that sends no base is saying it has not read the file: unconditional, as before.
        assert client.post("/api/config/save", json=edit).status_code == 200


def test_the_rail_reads_a_run_the_way_its_panel_does(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A relaunch appends to the log the finished record owns. The panel calls that running; a rail dot
    saying 'done' next to it is the interface contradicting itself."""
    import konfai_studio.server as server
    from starlette.testclient import TestClient

    log = workspace / "Statistics" / "MR2CT" / "log_0.txt"
    log.parent.mkdir(parents=True)
    log.write_text("Training : 12% 6/50\n", encoding="utf-8")
    job = workspace / ".konfai_mcp" / "jobs" / "job-1"
    job.mkdir(parents=True)
    (job / "job.json").write_text(
        json.dumps(
            {
                "job_id": "job-1",
                "kind": "train",
                "run_name": "MR2CT",
                "runtime_log_path": str(log),
                "status": "done",
                "created_at": 1000.0,
                "finished_at": 1100.0,  # long over; the log above was written just now
                "pid": 1,
            }
        ),
        encoding="utf-8",
    )

    with TestClient(server.app) as client:
        assert client.get("/api/sessions/status").json()["statuses"]["exp"] == "running"


def test_an_export_refuses_to_merge_into_an_app_bundle(workspace: Path, tmp_path: Path) -> None:
    """Bundle and export both write <chosen folder>/<experiment>, so aiming them at one place merged an
    export into the bundle: the result stopped being a usable bundle and nothing said so."""
    import konfai_studio.server as server
    from starlette.testclient import TestClient

    (workspace / "Config.yml").write_text("Trainer: {}\n", encoding="utf-8")
    out = tmp_path / "shared"
    (out / "exp").mkdir(parents=True)
    (out / "exp" / "app.json").write_text('{"name": "exp"}', encoding="utf-8")

    with TestClient(server.app) as client:
        payload = client.post("/api/sessions/export", json={"session": "exp", "output": str(out)}).json()

    assert payload["ok"] is False and "app bundle" in payload["result"]
    assert sorted(p.name for p in (out / "exp").iterdir()) == ["app.json"]  # nothing was copied over it

    fresh = tmp_path / "clean"
    with TestClient(server.app) as client:
        ok = client.post("/api/sessions/export", json={"session": "exp", "output": str(fresh)}).json()

    assert ok["ok"] is True and (fresh / "exp" / "Config.yml").is_file()


def test_a_config_diff_compares_the_training_configs(workspace: Path) -> None:
    """Every finished run leaves one config per kind, and the evaluation's is always the newest. Picked
    on mtime alone, two runs were compared on their Evaluation.yml: no model, no losses, no optimizer,
    so the panel showed everything except what the researcher changed."""
    train = workspace / "Statistics" / "RUN_A"
    evaluation = workspace / "Evaluations" / "RUN_A"
    train.mkdir(parents=True)
    evaluation.mkdir(parents=True)
    (train / "Config_0.yml").write_text("Trainer:\n  Model: UNet\n", encoding="utf-8")
    (evaluation / "Evaluation.yml").write_text("Evaluator: {}\n", encoding="utf-8")
    os.utime(evaluation / "Evaluation.yml", (2_000_000_000, 2_000_000_000))  # the evaluation ran last

    picked = _run_config_snapshot("exp", "RUN_A")

    assert picked is not None and picked.name == "Config_0.yml"
