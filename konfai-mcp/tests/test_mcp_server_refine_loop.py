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

"""The app-reuse refine loop: a done app job incites iterating only on the axis the user can turn, and
only when they signalled intent (a tuned run, a scored run, or a fine-tune)."""

from pathlib import Path

from konfai_mcp.server_jobs import Job, JobRegistry


def _done_job(tmp_path: Path, kind: str, set_parameters: list[str] | None = None) -> Job:
    return Job(
        job_id=f"{kind}-done",
        session="default",
        kind=kind,  # type: ignore[arg-type]
        command=["fake"],
        cwd=tmp_path,
        log_path=tmp_path / "job.log",
        config_path=tmp_path / "Config.yml",
        status="done",
        set_parameters=set_parameters,
    )


def _next_actions(job: Job) -> list[str]:
    return JobRegistry({"queued", "running"}).payload(job, lambda value: None)["next_actions"]


def test_plain_infer_is_not_dragged_into_refine(tmp_path: Path) -> None:
    """A one-shot infer with no set_parameters stays a plain result -- no evaluate/refine push."""
    actions = _next_actions(_done_job(tmp_path, "infer"))
    assert "run_app_evaluate" not in actions
    assert "leaderboard" not in actions
    assert "compare_runs" not in actions


def test_tuned_infer_incites_scoring_the_trial(tmp_path: Path) -> None:
    job = _done_job(tmp_path, "infer", set_parameters=["iterations=300"])
    payload = JobRegistry({"queued", "running"}).payload(job, lambda value: None)
    # The tuned trial's parameters ride along so the agent joins them to the score without a manifest read.
    assert payload["set_parameters"] == ["iterations=300"]
    assert "run_app_evaluate" in payload["next_actions"]
    assert "compare_runs" in payload["next_actions"]


def test_evaluate_and_pipeline_incite_ranking_and_reexport(tmp_path: Path) -> None:
    for kind in ("evaluate", "pipeline"):
        actions = _next_actions(_done_job(tmp_path, kind))
        assert "leaderboard" in actions
        assert "compare_runs" in actions
        assert "export_app" in actions


def test_finetune_points_at_use_then_evaluate_not_empty_leaderboard(tmp_path: Path) -> None:
    """A fine-tune keeps its training metrics out of the bundle, so it points at use+score, not a leaderboard
    that would have nothing to rank yet."""
    actions = _next_actions(_done_job(tmp_path, "finetune"))
    assert "run_app_infer" in actions
    assert "run_app_evaluate" in actions
    assert "leaderboard" not in actions
