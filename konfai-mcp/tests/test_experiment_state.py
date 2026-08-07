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

"""The experiment state, derived from a real workspace.

Every case here builds the workspace an experiment would actually leave behind and asks where it stands.
Nothing is fed through a conversation, so these also pin the property that matters: the answer is the
same whether the session was driven from Studio, from an MCP client, or by hand.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from konfai_mcp.experiment_state import (
    MAX_ATTEMPTS,
    STAGE_ACTIONS,
    STAGE_FOCUS,
    STAGES,
    collect_facts,
    derive_stage,
    diagnose,
    experiment_state,
    job_diagnosis,
    state_line,
)

CONFIG = """
Trainer:
  Dataset:
    dataset_filenames:
    - /data/pelvis:a:.mha
    groups_src:
      CT:
        groups_dest:
          CT:
            is_input: true
      Label:
        groups_dest:
          Label:
            is_input: false
"""


def workspace(tmp_path: Path, **contents: Any) -> Path:
    """Build the workspace an experiment would have left at a given point."""
    root = tmp_path / "session"
    root.mkdir(exist_ok=True)
    if contents.get("config"):
        (root / "Config.yml").write_text(CONFIG, encoding="utf-8")
    if contents.get("transform_config"):
        (root / "Transform.yml").write_text("Transformer:\n  name: T\n", encoding="utf-8")
    for name in contents.get("apps", []):
        (root / "Apps" / name).mkdir(parents=True, exist_ok=True)
    for name in contents.get("checkpoints", []):
        (root / "Checkpoints" / name).mkdir(parents=True, exist_ok=True)
    for name in contents.get("predictions", []):
        (root / "Predictions" / name).mkdir(parents=True, exist_ok=True)
    for name in contents.get("metrics", []):
        (root / "Evaluations" / name).mkdir(parents=True, exist_ok=True)
        (root / "Evaluations" / name / "Metric_TRAIN.json").write_text("{}", encoding="utf-8")
    return root


def job(status: str, kind: str = "train", error: str = "", run: str = "run_01") -> dict[str, Any]:
    return {"job_id": "job-1", "kind": kind, "status": status, "run_name": run, "error": error}


def state(root: Path, jobs: list[dict[str, Any]] | None = None, **kwargs: Any) -> dict[str, Any]:
    return experiment_state(collect_facts(root, jobs=jobs or [], **kwargs))


# --- the stage comes from what the workspace holds ---------------------------------------------------


def test_an_empty_workspace_starts_at_the_dataset(tmp_path: Path) -> None:
    """4. Nothing is known yet, so the next step is finding the data, not guessing at it."""
    assert state(workspace(tmp_path))["stage"] == "dataset_inspection"


def test_a_dataset_path_alone_is_not_yet_a_route(tmp_path: Path) -> None:
    """Knowing where the data is says nothing about what it holds: the next step is still to look."""
    payload = state(workspace(tmp_path), dataset=str(tmp_path / "nothing-there"))
    assert payload["stage"] == "dataset_inspection"


def test_the_data_itself_is_read_so_a_route_can_be_chosen(tmp_path: Path) -> None:
    """1-2. The dataset lives outside the workspace, so its structure is read from the data, and once
    the groups are in hand the question becomes what to DO with them."""
    data = tmp_path / "pelvis"
    for case in ("case_a", "case_b"):
        (data / case).mkdir(parents=True)
        for group in ("CT", "Label"):
            (data / case / f"{group}.mha").write_bytes(b"\x00")

    payload = state(workspace(tmp_path), dataset=str(data))

    assert payload["stage"] == "action_selection"
    assert payload["groups"] == ["CT", "Label"] and payload["cases"] == 2
    assert payload["next_actions"][0] == "list_apps"
    # This is where the user decides. The step the stage owes is to ASK them which route they want: # the options are not enumerated here, the assistant raises them and the buttons become its answers.
    assert "ASK which of the four routes" in payload["focus"]
    assert "theirs to choose" in payload["focus"]


def test_the_data_is_read_once_and_re_read_when_it_changes(tmp_path: Path) -> None:
    """The scan is the one expensive fact here, so it is cached: keyed by the directory's own mtime, so
    a case added to the dataset is picked up rather than remembered wrong."""
    data = tmp_path / "pelvis"
    (data / "case_a").mkdir(parents=True)
    (data / "case_a" / "CT.mha").write_bytes(b"\x00")
    assert state(workspace(tmp_path), dataset=str(data))["cases"] == 1

    (data / "case_b").mkdir()
    (data / "case_b" / "CT.mha").write_bytes(b"\x00")
    assert state(workspace(tmp_path), dataset=str(data))["cases"] == 2


def test_the_dataset_is_read_back_from_the_config_without_rewriting_it(tmp_path: Path) -> None:
    """A config names its dataset and its groups; reading them must leave the file untouched: loading a
    KonfAI config through the framework would rewrite it."""
    root = workspace(tmp_path, config=True)
    before = (root / "Config.yml").read_bytes()
    payload = state(root)
    assert payload["dataset"] == "/data/pelvis"
    assert payload["groups"] == ["CT", "Label"]
    assert payload["has_reference"] is True
    assert (root / "Config.yml").read_bytes() == before


def test_a_single_group_dataset_carries_no_reference(tmp_path: Path) -> None:
    """3. One group per case: there is nothing to train against or score against, and the state says so."""
    payload = state(workspace(tmp_path), dataset="/data/ct", groups=["CT"])
    assert payload["has_reference"] is False


def test_an_imported_app_is_a_fact_like_any_other(tmp_path: Path) -> None:
    """5. An app imported into the session is on disk, so the stage follows without being told."""
    payload = state(workspace(tmp_path, apps=["TotalSegmentator"]), dataset="/data/pelvis")
    assert payload["stage"] == "app_selection" and payload["app"] == "TotalSegmentator"


def test_the_workspace_decides_the_stage_even_with_no_job_history(tmp_path: Path) -> None:
    """7. Checkpoints, then predictions, then metrics, each supersedes the last, whoever produced them."""
    assert state(workspace(tmp_path, config=True))["stage"] == "configuration"
    assert state(workspace(tmp_path, config=True, checkpoints=["run_01"]))["stage"] == "checkpoint_selection"
    assert state(workspace(tmp_path, config=True, checkpoints=["run_01"], predictions=["run_01"]))["stage"] == (
        "prediction"
    )
    full = workspace(tmp_path, config=True, checkpoints=["run_01"], predictions=["run_01"], metrics=["run_01"])
    assert state(full)["stage"] == "result_review"


# --- a run's outcome ---------------------------------------------------------------------------------


def test_a_session_is_offered_the_workflow_it_actually_wrote(tmp_path: Path) -> None:
    """The action bar is built from these, so a workflow missing here is unreachable in Studio.

    A session holding only a Transform.yml was told to "run train" (an action naming a file it does
    not have), while the one workflow it could run was offered by nothing.
    """
    payload = state(workspace(tmp_path, transform_config=True))

    assert "run_train" not in payload["next_actions"]
    actions = payload["next_actions"]
    assert actions.index("plan_transform") < actions.index("run_transform")


def test_choosing_among_configs_does_not_hide_the_step_that_comes_next(tmp_path: Path) -> None:
    """A launcher only describes what the session HAS while it is choosing among its configs.

    At `checkpoint_selection` it names the step that follows, and `run_prediction` is offered exactly
    because no Prediction.yml exists yet: filtering there would remove the whole point of the stage.
    """
    payload = state(workspace(tmp_path, config=True, checkpoints=["run_01"]), [job("done")])

    assert payload["next_actions"][0] == "run_prediction"


def test_an_open_job_is_running_and_says_what_to_wait_for(tmp_path: Path) -> None:
    """11. A launched job is not a result: the state stays at 'running' and asks for the outcome."""
    payload = state(workspace(tmp_path, config=True), [job("running")])
    assert payload["stage"] == "running"
    assert "wait for it to reach a terminal state" in payload["focus"]
    assert payload["next_actions"][0] == "wait_for_job"


def test_a_finished_training_leads_to_the_checkpoint_not_to_done(tmp_path: Path) -> None:
    """7. Training is a step: the state points at the checkpoint and the prediction that follows."""
    payload = state(workspace(tmp_path, config=True, checkpoints=["run_01"]), [job("done")])
    assert payload["stage"] == "checkpoint_selection"
    assert payload["next_actions"][0] == "run_prediction"


def test_a_finished_session_job_that_wrote_nothing_is_not_a_success(tmp_path: Path) -> None:
    """14. 'done' with an empty workspace is a failure, whatever the job record claims."""
    payload = state(workspace(tmp_path, config=True), [job("done", "prediction")])
    assert payload["stage"] == "failed"
    assert payload["diagnosis"]["kind"] == "missing_outputs"


def test_an_app_job_is_taken_at_its_word(tmp_path: Path) -> None:
    """14b. An app writes wherever it was pointed, so its silence in the workspace proves nothing."""
    payload = state(workspace(tmp_path), [job("done", "infer")])
    assert payload["stage"] == "prediction"


def test_a_recoverable_failure_carries_its_correction(tmp_path: Path) -> None:
    """12. The cause and the fix are in the state, so the next turn acts instead of re-deriving them."""
    failure = job("error", error="torch.cuda.OutOfMemoryError: CUDA out of memory")
    payload = state(workspace(tmp_path, config=True), [failure])
    assert payload["stage"] == "failed"
    assert payload["diagnosis"] == {
        "kind": "out_of_memory",
        "summary": "the run ran out of memory",
        "fix": "lower the patch size or the batch size, or run it on a device with more memory",
        "recoverable": True,
    }
    assert "relaunch once" in payload["focus"] and "patch size" in payload["focus"]


def test_a_failure_that_is_the_users_call_is_not_corrected_alone(tmp_path: Path) -> None:
    """13. Existing outputs are never overwritten on the agent's initiative."""
    failure = job("error", error="Model run_01 already exists, pass --overwrite to replace it")
    payload = state(workspace(tmp_path, config=True), [failure])
    assert payload["diagnosis"]["recoverable"] is False
    assert "the user's call" in payload["focus"]


def test_the_retry_budget_is_counted_from_the_job_records(tmp_path: Path) -> None:
    """12b. Two failures in a row and the loop stops: counted from disk, so a restart cannot reset it."""
    failures = [job("error", error="CUDA out of memory") for _ in range(MAX_ATTEMPTS)]
    payload = state(workspace(tmp_path, config=True), failures)
    assert payload["attempts"] == MAX_ATTEMPTS
    assert "Stop correcting it" in payload["focus"]
    assert "ask for the one decision" in payload["focus"]


def test_a_success_clears_the_budget(tmp_path: Path) -> None:
    history = [job("done"), job("error", error="CUDA out of memory")]
    assert state(workspace(tmp_path, config=True, checkpoints=["run_01"]), history)["attempts"] == 0


# --- properties ---------------------------------------------------------------------------------------


def test_every_stage_has_a_focus_and_actions_that_exist() -> None:
    assert set(STAGE_FOCUS) == set(STAGES)
    assert set(STAGE_ACTIONS) == set(STAGES)
    assert all(STAGE_ACTIONS[stage] for stage in STAGES)


def test_stage_actions_never_promise_what_the_workspace_cannot_do(tmp_path: Path) -> None:
    """Only known facts filter: no packaging without a checkpoint, no ranking without a metric."""
    payload = state(workspace(tmp_path, config=True, checkpoints=["run_01"], predictions=["run_01"]))
    assert "package_app_from_session" in payload["next_actions"]  # a checkpoint exists
    assert "leaderboard" not in payload["next_actions"]  # nothing scored yet


def test_the_state_line_is_a_line(tmp_path: Path) -> None:
    """What a turn carries: identifiers, not a transcript."""
    root = workspace(tmp_path, config=True, checkpoints=["run_01"], predictions=["run_01"])
    line = state_line(state(root, [job("done", "prediction")]))
    assert line.startswith("stage=prediction")
    assert "dataset=/data/pelvis" in line and "groups=CT/Label" in line and "run=run_01" in line
    assert "\n" not in line and len(line) < 200


def test_a_healthy_job_carries_no_diagnosis(tmp_path: Path) -> None:
    assert job_diagnosis({"status": "done"}, None) is None
    assert job_diagnosis({"status": "error", "error": "CUDA out of memory"}, None)["kind"] == "out_of_memory"


def test_a_cancelled_run_is_not_diagnosed_as_a_bug() -> None:
    killed = diagnose("", status="killed")
    assert killed.kind == "cancelled"
    # The wording is load-bearing: an agent reading "cancelled" alone concluded an external watchdog had
    # killed the run and burned its retry budget on the user's own Stop clicks.
    assert "on purpose" in killed.summary and "not a failure" in killed.summary
    assert not killed.recoverable
    assert diagnose("Traceback (most recent call last)").kind == "runtime_error"
    assert diagnose("").kind == "unknown"


def test_the_derivation_survives_a_workspace_that_is_not_there(tmp_path: Path) -> None:
    """A session whose workspace was never created still answers, rather than raising into the chat."""
    payload = experiment_state(collect_facts(tmp_path / "missing"))
    assert payload["stage"] == "dataset_inspection" and payload["next_actions"]


def test_stage_derivation_is_total() -> None:
    """Every derived stage is one the focus and action tables know about."""
    from konfai_mcp.experiment_state import Facts

    for kind in ("train", "prediction", "evaluation", "infer", "finetune", "pipeline", "uncertainty", ""):
        for status in ("queued", "running", "done", "error", "killed", ""):
            facts = Facts(job_kind=kind, job_status=status, checkpoints=["a"], predictions=["b"], metrics=["c"])
            assert derive_stage(facts) in STAGES


def test_the_job_records_are_read_defensively(tmp_path: Path) -> None:
    """A truncated job.json must not take the state down with it."""
    root = workspace(tmp_path, config=True)
    (root / "broken.json").write_text("{not json", encoding="utf-8")
    assert state(root, [{"status": "done"}, {}])["stage"] in STAGES
    assert json.loads("{}") == {}
