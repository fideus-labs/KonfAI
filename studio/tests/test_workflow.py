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

"""The two things Studio builds from the experiment state: the pre-prompt, and the action buttons.

The state itself is derived by konfai-mcp and tested there (``test_experiment_state.py``); what is
checked here is that a turn is *told* the right thing and *offers* something real.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("konfai_mcp")
from konfai_mcp.experiment_state import STAGE_ACTIONS, STAGES, Facts, experiment_state
from konfai_studio.workflow import MAX_MOVES, moves, pre_prompt

# Button prompts that frame nothing: the failure mode this builder exists to prevent.
VAGUE = {"continue", "run it", "train model", "evaluate", "check results", "go", "next"}


def state(**facts: Any) -> dict[str, Any]:
    return experiment_state(Facts(**facts))


def labels(payload: dict[str, Any]) -> list[str]:
    return [move["label"] for move in moves(payload)]


# --- the pre-prompt ----------------------------------------------------------------------------------


def test_the_pre_prompt_is_three_short_lines_and_the_message() -> None:
    """What a turn costs: the identifiers, the one instruction that applies, the constraint. No history."""
    payload = state(dataset="/data/pelvis", groups=["CT", "Label"], cases=40, configs=["Config.yml"])
    text = pre_prompt(payload, "Run every job on GPU 0 (pass gpu=[0]).", "why is the loss flat?")

    head, focus, constraint, blank, message = text.splitlines()
    assert head.startswith("[state] stage=configuration dataset=/data/pelvis")
    assert "groups=CT/Label" in head and "cases=40" in head
    assert focus.startswith("[now] Validate the config at level 'train_step'")
    assert constraint.startswith("[constraint] Run every job on GPU 0")
    assert blank == "" and message == "why is the loss flat?"
    assert len(head) < 200 and len(focus) < 260


def test_the_instruction_matches_where_the_experiment_stands() -> None:
    """The stage's instruction rides the turn, so 'inspect the dataset' cannot reach a running job."""
    fresh = pre_prompt(state(), "", "here is my data")
    assert "[now] Inspect the dataset: inspect_dataset" in fresh

    running = pre_prompt(state(job_status="running", job_id="job-1", run="run_01"), "", "and now?")
    assert "[now] Job job-1 is still open: wait for it" in running
    assert "Inspect the dataset" not in running


def test_a_failure_arrives_already_diagnosed() -> None:
    """The turn is told the cause and the correction, it does not have to re-derive them from a log."""
    failed = state(job_status="error", job_error="CUDA out of memory", run="run_01", configs=["Config.yml"])
    text = pre_prompt(failed, "", "what happened?")
    assert "[now] run_01 failed: the run ran out of memory." in text
    assert "lower the patch size" in text and "relaunch once" in text
    assert "failure=out_of_memory" in text.splitlines()[0]


def test_no_constraint_line_when_the_device_is_free() -> None:
    assert "[constraint]" not in pre_prompt(state(), "", "go")


# --- the fallback buttons ----------------------------------------------------------------------------
# The LLM writes the buttons; these cover the turn where it wrote nothing, so the bar is never empty.


def test_every_state_offers_at_least_one_fallback_move() -> None:
    for stage in STAGES:
        for action in STAGE_ACTIONS[stage]:
            payload = {"next_actions": [action], "run": "run_01"}
            assert moves(payload), f"{stage}/{action} left the turn with nothing to do"


def test_a_fallback_move_names_its_subject_and_what_clicking_it_does() -> None:
    payload = state(dataset="/data/pelvis", groups=["CT", "Label"])
    first = moves(payload)[0]
    assert first["label"] == "List apps"
    assert first["prompt"].startswith("List apps, on the dataset at /data/pelvis.")
    # The stage's instruction is to ASK the user which route. Repeating it into a button would send the
    # assistant back to asking instead of doing what was clicked, so a button says only what it does.
    assert "ASK which" not in first["prompt"]
    assert first["prompt"].endswith("Report it in two lines, then the next step.")


def test_a_fallback_move_that_launches_a_job_demands_it_be_followed_through() -> None:
    launched = moves({"next_actions": ["run_prediction"], "run": "run_01"})[0]
    assert "Wait for the job" in launched["prompt"] and "on failure read the log" in launched["prompt"]


def test_fallback_moves_follow_the_servers_ranking_deduped_and_capped() -> None:
    payload = state(checkpoints=["run_01"], job_status="done", job_kind="train", run="run_01")
    proposed = moves(payload)
    # The server's own ranking, in its order: the count follows what applies rather than a fixed three.
    assert [move["label"] for move in proposed][:3] == [
        "Run prediction",
        "Read training curves",
        "Package app from session",
    ]
    assert len({move["label"] for move in proposed}) == len(proposed)
    assert len(proposed) <= MAX_MOVES


def test_a_state_with_many_applicable_actions_offers_them_all_up_to_the_cap() -> None:
    """A decision point can have more than three real answers; only past ten does a bar stop being a
    choice and become a menu."""
    many = [f"run_{i}" for i in range(20)]
    assert len(moves({"next_actions": many, "run": "run_01"})) == MAX_MOVES


def test_a_state_with_nothing_to_suggest_still_suggests_something() -> None:
    """The last resort is a real question about a real experiment, never an empty bar."""
    fallback = moves({"next_actions": [], "run": "run_01", "focus": "Say where this stands"})
    assert len(fallback) == 1 and fallback[0]["label"] == "Summarize session"
    assert "run run_01" in fallback[0]["prompt"]


def test_no_fallback_move_is_vague() -> None:
    for stage in STAGES:
        payload = {"next_actions": STAGE_ACTIONS[stage], "dataset": "/data/pelvis", "focus": "Do the thing"}
        for move in moves(payload):
            assert move["prompt"].strip().lower().rstrip(".") not in VAGUE
            assert len(move["prompt"].split()) >= 8, move
            assert 1 <= len(move["label"].split()) <= 5, move


def test_no_ranked_action_points_at_a_tool_that_no_longer_exists(tmp_path: Any, monkeypatch: Any) -> None:
    """The ranking is keyed by tool names konfai-mcp owns. One renamed or dropped there must fail here,
    not go quiet in the action bar."""
    monkeypatch.setenv("KONFAI_MCP_WORKSPACES_ROOT", str(tmp_path))
    fastmcp = pytest.importorskip("fastmcp")
    import asyncio

    from konfai_mcp import server as mcp_server

    async def registered() -> set[str]:
        async with fastmcp.Client(mcp_server.mcp) as client:
            return {tool.name for tool in await client.list_tools()}

    tools = asyncio.run(registered())
    assert not {a for actions in STAGE_ACTIONS.values() for a in actions} - tools, "unregistered ranked actions"


def test_no_move_offers_a_tool_the_experiment_cannot_run_yet() -> None:
    """`summarize_session` needs a workspace `initialize_session` has not made yet: offered before that
    it comes back "call initialize_session first", which is a dead button, not a next step."""
    early = state(dataset="/data/pelvis", groups=["CT", "Label"])
    assert "Summarize session" not in labels(early)

    settled = state(checkpoints=["run_01"], job_status="done", job_kind="train", run="run_01")
    assert "Summarize session" in labels(settled)
