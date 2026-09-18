# SPDX-License-Identifier: Apache-2.0
"""What a turn is told, and what it offers next.

The experiment's state is **not** computed here. It is derived by ``konfai_mcp.experiment_state`` from
the workspace on disk (the same answer the MCP server gives every other client), and Studio only turns
it into two things the chat needs:

* the **pre-prompt** (`[state]` + `[now]`): the identifiers and the one instruction the current stage
  owes, so a turn carries a line instead of a transcript;
* the **action buttons**: the server's own ranked ``next_actions``, each framed by one template into a
  prompt that names its subject.

Studio is co-located with konfai-mcp on the compute node and reads the same workspace, so this costs no
extra process and cannot disagree with what the MCP tools report.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from konfai_mcp.experiment_state import collect_facts, experiment_state, state_line

__all__ = ["experiment_state", "moves", "pre_prompt", "state_for", "state_line"]

# The one thing every button asks of the next turn. A run must be followed through; a read must land on
# a conclusion. Nothing else is repeated into a prompt.
_FOLLOW_RUN = (
    "Wait for the job, inspect what it produced, then report the outcome and the next step; on failure "
    "read the log, give the cause in one line and the fix."
)
_FOLLOW_READ = "Report it in two lines, then the next step."
_LAUNCHES = ("run_", "fine_tune_")
# Stages reached before `initialize_session` has made a workspace: nothing there can be summarised yet.
_BEFORE_WORKSPACE = {"dataset_inspection", "action_selection", "app_selection"}
# One ceiling for the whole chain: the assistant's own block, the derived fill-in, and the bar. Past
# this a row of buttons stops being a choice and becomes a menu to read; below it, the count follows
# what genuinely applies rather than a fixed number.
MAX_MOVES = 9


def state_for(workspace: Path, jobs: list[dict[str, Any]], dataset: str = "") -> dict[str, Any]:
    """The experiment's state, derived from its workspace by the MCP server's own code. ``jobs`` are its
    job records newest first, they carry the run's outcome and, counted, the retry budget."""
    return experiment_state(collect_facts(workspace, jobs=jobs, dataset=dataset))


def pre_prompt(state: dict[str, Any], device: str, message: str) -> str:
    """The user's message under the two lines a turn needs: where the experiment stands, and what this
    stage owes. Standing rules live in the system prompt: only the line that applies now is paid for."""
    lines = [f"[state] {state_line(state)}"]
    if state.get("focus"):
        lines.append(f"[now] {state['focus']}")
    if device:
        lines.append(f"[constraint] {device}")
    return "\n".join([*lines, "", message])


def _subject(state: dict[str, Any]) -> str:
    """What the next step acts on, named, so the prompt needs nothing from the conversation."""
    for key, prefix in (("run", "run "), ("app", "app "), ("dataset", "the dataset at ")):
        if state.get(key):
            return f"{prefix}{state[key]}"
    return "this experiment"


def moves(state: dict[str, Any], from_tools: list[str] | None = None) -> list[dict[str, str]]:
    """The action buttons shown with the answer, and the ones kept when the LLM writes none of its own
    (its call failed, or the brain has no side channel for it). Composed from the tools the MCP server
    ranked for this state and the step that state owes, so a button is always offered and never names a
    step the experiment cannot take.

    Deliberately plain: these are worth less than the LLM's own proposals, which can name the app it just
    described. What they buy is that the bar is usable the instant the answer lands.
    """
    subject = _subject(state)
    # `summarize_session` is the last resort for a bar that would otherwise be empty, but it needs a
    # session workspace: offered before one exists it fails with "call initialize_session first". The
    # stage's own actions already cover those early stages, so it only trails the list once there is
    # something to summarise.
    stage = str(state.get("stage") or "")
    tail = ["summarize_session"] if stage not in _BEFORE_WORKSPACE else []
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for action in [*(from_tools or []), *state.get("next_actions", []), *tail]:
        label = action.replace("_", " ").capitalize()
        if label in seen:
            continue
        seen.add(label)
        # The stage's own instruction is deliberately NOT repeated here: it describes what the experiment
        # owes, which is often to ask the user a question, and a button that asks the assistant to ask is
        # a loop. A button says what clicking it does, to what.
        follow = _FOLLOW_RUN if action.startswith(_LAUNCHES) else _FOLLOW_READ
        out.append({"label": label, "prompt": f"{label}, on {subject}. {follow}"})
        if len(out) == MAX_MOVES:
            break
    return out
