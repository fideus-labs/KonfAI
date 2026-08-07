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

"""A chat turn end to end through the BFF, with a scripted agent in place of the LLM.

The agent here does what the real tools do (it writes to the workspace), because that is where the
next turn's state comes from. So these cover the orchestration no prompt can guarantee: what rides into
a turn, what is derived from what the turn actually achieved, and what comes back out of it.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")
from konfai_studio import server as bff
from konfai_studio.agent import with_volume_events
from konfai_studio.registry import _Registry
from starlette.testclient import TestClient

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


class _ScriptedAgent:
    """Stands in for the LLM: emits fixed events and, like the real tools, leaves its traces on disk."""

    def __init__(self) -> None:
        self.script: list[dict[str, Any]] = []
        self.effect: Callable[[], None] | None = None  # what its "tools" wrote this turn
        self.messages: list[str] = []

    async def __aenter__(self) -> _ScriptedAgent:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def send(self, message: str) -> AsyncIterator[dict[str, Any]]:
        self.messages.append(message)
        if self.effect is not None:
            self.effect()

        async def scripted() -> AsyncIterator[dict[str, Any]]:
            for event in self.script:
                yield event

        # Through the real pipeline, exactly as the live agents are: it is what splits the reply from the
        # moves block it ends with, so a test that bypassed it would prove nothing about the buttons.
        async for event in with_volume_events(scripted()):
            yield event


@pytest.fixture
def studio(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[TestClient, _ScriptedAgent, Path]]:
    """A Studio BFF on a throwaway workspace, driven by the scripted agent."""
    monkeypatch.setenv("KONFAI_MCP_WORKSPACES_ROOT", str(tmp_path))
    monkeypatch.setenv("KONFAI_STUDIO_LLM", "openai")  # no LLM call for titles or extra suggestions
    session = tmp_path / "sessions" / "exp"
    session.mkdir(parents=True)
    agent = _ScriptedAgent()
    registry = _Registry()
    monkeypatch.setattr(bff, "_reg", registry)
    monkeypatch.setattr(registry, "agent", lambda name: _ready(agent))
    with TestClient(bff.app) as client:
        yield client, agent, session


async def _ready(agent: _ScriptedAgent) -> _ScriptedAgent:
    return agent


def turn(client: TestClient, message: str, session: str = "exp") -> list[dict[str, Any]]:
    with client.stream("POST", "/api/chat", json={"message": message, "session": session}) as response:
        assert response.status_code == 200
        return [json.loads(line[6:]) for line in response.iter_lines() if line.startswith("data: ")]


def of_type(events: list[dict[str, Any]], kind: str) -> dict[str, Any]:
    found = [event for event in events if event.get("type") == kind]
    assert found, f"the turn emitted no {kind!r} event"
    return found[0]


def record_job(session: Path, **fields: Any) -> None:
    """Write a job record exactly where konfai-mcp writes it, so the BFF reads a real one."""
    job_dir = session / ".konfai_mcp" / "jobs" / str(fields.get("job_id", "job-1"))
    job_dir.mkdir(parents=True, exist_ok=True)
    payload = {"job_id": "job-1", "kind": "train", "status": "done", "run_name": "run_01", "created_at": 1.0}
    (job_dir / "job.json").write_text(json.dumps({**payload, **fields}), encoding="utf-8")


TEXT_ONLY = [{"type": "text", "text": "ok"}, {"type": "done"}]


def test_a_fresh_experiment_is_told_to_start_at_the_data(studio: tuple[TestClient, _ScriptedAgent, Path]) -> None:
    client, agent, _ = studio
    agent.script = TEXT_ONLY

    events = turn(client, "here is my data")

    assert agent.messages[0].startswith("[state] stage=dataset_inspection\n[now] Inspect the dataset")
    assert of_type(events, "state")["stage"] == "dataset_inspection"
    assert of_type(events, "next_prompts")["prompts"][0]["label"] == "Inspect dataset"


def test_the_next_turn_carries_what_the_last_one_achieved_not_what_it_said(
    studio: tuple[TestClient, _ScriptedAgent, Path],
) -> None:
    """The state is re-read from the workspace, so it reflects the config the turn really wrote."""
    client, agent, session = studio
    agent.script = TEXT_ONLY
    agent.effect = lambda: (session / "Config.yml").write_text(CONFIG, encoding="utf-8")
    turn(client, "set up a segmentation on /data/pelvis")

    agent.effect = None
    turn(client, "what does patch size mean?")

    second = agent.messages[1]
    assert second.startswith("[state] stage=configuration dataset=/data/pelvis")
    assert "groups=CT/Label" in second
    assert "[now] Validate the config" in second
    assert "set up a segmentation" not in second  # the previous turn is never replayed
    assert len(second.splitlines()[0]) < 200


def test_a_launched_job_is_reported_as_open_not_as_a_result(
    studio: tuple[TestClient, _ScriptedAgent, Path],
) -> None:
    """The turn claims success in words; the job record says it is still running, and the record wins."""
    client, agent, session = studio
    agent.script = [{"type": "text", "text": "Training finished successfully!"}, {"type": "done"}]
    agent.effect = lambda: record_job(session, status="running", pid=1)

    events = turn(client, "train it")

    state = of_type(events, "state")
    assert state["stage"] == "running"
    assert of_type(events, "next_prompts")["prompts"][0]["label"] == "Wait for job"
    turn(client, "and?")
    assert "[now] Job job-1 is still open" in agent.messages[1]


def test_a_finished_run_that_wrote_nothing_is_not_a_success(
    studio: tuple[TestClient, _ScriptedAgent, Path],
) -> None:
    client, agent, session = studio
    agent.script = TEXT_ONLY
    agent.effect = lambda: record_job(session, kind="prediction", status="done")

    events = turn(client, "predict")

    state = of_type(events, "state")
    assert state["stage"] == "failed" and state["diagnosis"]["kind"] == "missing_outputs"


def test_a_failed_run_comes_back_diagnosed_with_its_correction(
    studio: tuple[TestClient, _ScriptedAgent, Path],
) -> None:
    client, agent, session = studio
    agent.script = TEXT_ONLY
    agent.effect = lambda: record_job(session, status="error", error="torch.cuda.OutOfMemoryError: CUDA out of memory")

    events = turn(client, "train it")

    assert of_type(events, "state")["diagnosis"]["kind"] == "out_of_memory"
    assert of_type(events, "next_prompts")["prompts"][0]["label"] == "Read job log"
    turn(client, "why?")
    assert "[now] run_01 failed: the run ran out of memory" in agent.messages[1]
    assert "lower the patch size" in agent.messages[1]


def test_training_hands_over_to_prediction(studio: tuple[TestClient, _ScriptedAgent, Path]) -> None:
    """A trained model is not the end of the workflow, and the bar says what follows."""
    client, agent, session = studio
    agent.script = TEXT_ONLY
    agent.effect = lambda: (
        record_job(session, status="done"),
        (session / "Checkpoints" / "run_01").mkdir(parents=True),
    )

    events = turn(client, "train it")

    assert of_type(events, "state")["stage"] == "checkpoint_selection"
    labels = [prompt["label"] for prompt in of_type(events, "next_prompts")["prompts"]]
    assert labels[0] == "Run prediction"


def test_the_assistant_writes_its_own_buttons_at_the_end_of_its_reply(
    studio: tuple[TestClient, _ScriptedAgent, Path],
) -> None:
    """The buttons come out of the reply itself: no second model call, and no wait after the answer.
    The block is stripped from what the user reads and becomes the bar."""
    client, agent, session = studio
    agent.script = [
        {"type": "text", "text": "Training finished. Which do you want next?\n\n<<NEXT"},
        {"type": "text", "text": ">>\nPredict The 40 CT Cases :: Run run_01's best checkpoint on /data/pelvis."},
        {"type": "text", "text": "\nCompare With The App :: Compare run_01 against TotalSegmentator.\n"},
        {"type": "done"},
    ]
    agent.effect = lambda: (session / "Checkpoints" / "run_01").mkdir(parents=True)

    events = turn(client, "train it")

    shown = "".join(event["text"] for event in events if event.get("type") == "text")
    assert shown.strip() == "Training finished. Which do you want next?"  # the marker never reaches the user
    assert "<<NEXT" not in shown and "::" not in shown

    rounds = [event["prompts"] for event in events if event.get("type") == "next_prompts"]
    assert len(rounds) == 1, "the buttons must not be shown twice and change under the user"
    assert [move["label"] for move in rounds[0]] == ["Predict The 40 CT Cases", "Compare With The App"]
    assert rounds[0][0]["prompt"] == "Run run_01's best checkpoint on /data/pelvis."


def test_a_reply_with_no_block_still_gets_buttons(studio: tuple[TestClient, _ScriptedAgent, Path]) -> None:
    """The assistant can forget the block; the derived moves stand in rather than an empty bar."""
    client, agent, session = studio
    agent.script = TEXT_ONLY
    agent.effect = lambda: (session / "Checkpoints" / "run_01").mkdir(parents=True)

    prompts = of_type(turn(client, "train it"), "next_prompts")["prompts"]

    assert [move["label"] for move in prompts][:3] == [
        "Run prediction",
        "Read training curves",
        "Package app from session",
    ]


def test_a_silent_llm_still_leaves_a_usable_bar(studio: tuple[TestClient, _ScriptedAgent, Path]) -> None:
    """The fixture's brain has no side channel for proposals, so every button here is the derived one."""
    client, agent, session = studio
    agent.script = TEXT_ONLY
    agent.effect = lambda: (session / "Checkpoints" / "run_01").mkdir(parents=True)

    prompts = of_type(turn(client, "train it"), "next_prompts")["prompts"]

    assert [prompt["label"] for prompt in prompts][:3] == [
        "Run prediction",
        "Read training curves",
        "Package app from session",
    ]
    assert all("/data/pelvis" in p["prompt"] or "run_01" in p["prompt"] for p in prompts)


def test_a_correction_typed_mid_turn_cuts_the_turn_short(studio: tuple[TestClient, _ScriptedAgent, Path]) -> None:
    """Typing while it works is a correction, so the turn is interrupted rather than left to finish
    everything it had planned. The correction itself is sent by the front, as the next turn."""
    client, agent, _ = studio
    cut: list[str] = []
    agent.interrupt = lambda: _true(cut)  # type: ignore[attr-defined]
    bff._reg.register("exp")
    bff._reg._sessions["exp"].agent = agent  # what a live turn leaves behind

    assert client.post("/api/chat/interrupt", json={"session": "exp"}).json() == {"ok": True}
    assert cut == ["interrupted"]


def test_interrupting_a_backend_that_cannot_is_not_an_error(studio: tuple[TestClient, _ScriptedAgent, Path]) -> None:
    """An API brain has no interrupt channel: the front is told so, and falls back to queueing."""
    client, _agent, _ = studio
    assert client.post("/api/chat/interrupt", json={"session": "exp"}).json() == {"ok": False}


async def _true(seen: list[str]) -> bool:
    seen.append("interrupted")
    return True


def test_every_turn_ends_with_a_state_and_a_move(studio: tuple[TestClient, _ScriptedAgent, Path]) -> None:
    client, agent, _ = studio
    for script in (TEXT_ONLY, [{"type": "done"}], []):
        agent.script = script
        events = turn(client, "go on")
        assert of_type(events, "state")["stage"]
        assert of_type(events, "next_prompts")["prompts"]


def test_a_broken_turn_still_says_where_the_experiment_stands(
    studio: tuple[TestClient, _ScriptedAgent, Path],
) -> None:
    """An LLM that dies mid-turn must not leave the user staring at an error with nothing to click."""
    client, agent, _ = studio

    async def explode(message: str) -> AsyncIterator[dict[str, Any]]:
        agent.messages.append(message)
        raise RuntimeError("the model went away")
        yield {}  # pragma: no cover - unreachable, marks this an async generator

    agent.send = explode  # type: ignore[method-assign]
    events = turn(client, "go")

    assert of_type(events, "error")["message"] == "the model went away"
    assert of_type(events, "state")["stage"] == "dataset_inspection"
    assert of_type(events, "next_prompts")["prompts"]


def test_the_state_and_the_moves_survive_a_reload(studio: tuple[TestClient, _ScriptedAgent, Path]) -> None:
    client, _agent, session = studio
    (session / "Config.yml").write_text(CONFIG, encoding="utf-8")

    payload = client.get("/api/experiment?session=exp").json()

    assert payload["workflow"]["stage"] == "configuration"
    assert payload["moves"][0]["label"] == "Validate config semantics"


# --- the front the browser is given ------------------------------------------------------------------


def test_the_index_is_never_cached_and_the_assets_always_are(studio: tuple[TestClient, _ScriptedAgent, Path]) -> None:
    """index.html is what names the fingerprinted bundle, so a cached one pins the browser to the
    previous build: an updated Studio silently serving the old front. The assets it names carry a hash,
    so their bytes never change and they can be cached for a year."""
    client, _agent, _ = studio

    index = client.get("/")
    assert "no-cache" in index.headers.get("cache-control", ""), "a cached index pins the old front"

    from konfai_studio.server import WEB_DIR

    built = sorted((WEB_DIR / "assets").glob("*.js")) if (WEB_DIR / "assets").is_dir() else []
    if not built:
        pytest.skip("front not built in this tree")
    asset = client.get(f"/assets/{built[0].name}")
    assert "immutable" in asset.headers.get("cache-control", "")


def test_an_interrupted_turn_keeps_the_buttons_that_still_apply(
    studio: tuple[TestClient, _ScriptedAgent, Path],
) -> None:
    """A turn cut short writes no moves block. Falling straight back to the generic fill-in throws away
    the last real ones, they still apply while the experiment has not moved."""
    client, agent, _ = studio
    agent.script = [
        {"type": "text", "text": "Which do you want?\n<<NEXT>>\nDo MR To CT :: Fine-tune the MR->CT app.\n"},
        {"type": "done"},
    ]
    first = of_type(turn(client, "inspect it"), "next_prompts")["prompts"]
    assert first[0]["label"] == "Do MR To CT"

    agent.script = [{"type": "done"}]  # interrupted: no text, no block
    kept = of_type(turn(client, "wait"), "next_prompts")["prompts"]

    assert kept[0]["label"] == "Do MR To CT", "the assistant's own moves survive a turn that wrote none"


def test_a_question_is_not_padded_with_tool_named_buttons(
    studio: tuple[TestClient, _ScriptedAgent, Path],
) -> None:
    """When the reply asks a question its moves ARE the options. A derived "Run train" beside them answers
    a question nobody asked, and launches the config the reply just said was wrong."""
    client, agent, session = studio
    agent.script = [
        {"type": "text", "text": "That patch cannot work. Which do you want?\n<<NEXT>>\n"},
        {"type": "text", "text": "Keep 2.5D :: Set patch_size to [1, 512, 512] in run_01.\n"},
        {"type": "text", "text": "Go 3D :: Switch run_01 to a 3D model.\n"},
        {"type": "done"},
    ]
    agent.effect = lambda: (session / "Checkpoints" / "run_01").mkdir(parents=True)

    prompts = of_type(turn(client, "use a huge patch"), "next_prompts")["prompts"]

    assert [move["label"] for move in prompts] == ["Keep 2.5D", "Go 3D"]


def test_remembered_buttons_are_dropped_once_the_experiment_moves(
    studio: tuple[TestClient, _ScriptedAgent, Path],
) -> None:
    """They describe a state. Once a job has run, they are stale and the derived moves are more honest."""
    client, agent, session = studio
    agent.script = [
        {"type": "text", "text": "ok\n<<NEXT>>\nInspect The Data :: Inspect /data/pelvis.\n"},
        {"type": "done"},
    ]
    turn(client, "hello")

    agent.script = [{"type": "done"}]
    agent.effect = lambda: record_job(session, status="running", pid=1)
    moved = of_type(turn(client, "and now"), "next_prompts")["prompts"]

    assert "Inspect The Data" not in [move["label"] for move in moved]
    assert moved[0]["label"] == "Wait for job"


def test_the_transcript_is_served_to_a_browser_that_missed_the_turns(
    studio: tuple[TestClient, _ScriptedAgent, Path],
) -> None:
    """The browser stores its transcript in localStorage, per device. The server records what it
    streamed, so a second browser has something to adopt instead of an empty chat."""
    client, agent, _session = studio
    agent.script = [{"type": "text", "text": "bonjour "}, {"type": "text", "text": "toi"}, {"type": "done"}]
    turn(client, "salut")

    history = client.get("/api/chat/history", params={"session": "exp"}).json()["messages"]

    assert history[-2] == {"role": "user", "text": "salut"}
    assert history[-1]["role"] == "assistant"
    assert history[-1]["parts"] == [{"kind": "text", "text": "bonjour toi"}]


def test_an_interrupted_tool_is_recorded_as_cut_not_left_running(
    studio: tuple[TestClient, _ScriptedAgent, Path],
) -> None:
    """A stream that dies before the tool answers still leaves a turn on disk, with the tool marked
    interrupted, not spinning forever on whichever browser reloads it."""
    client, agent, _session = studio
    agent.script = [
        {"type": "text", "text": "je lance"},
        {"type": "tool_call", "name": "run_train", "input": {"config": "Config.yml"}},
    ]
    turn(client, "va")

    history = client.get("/api/chat/history", params={"session": "exp"}).json()["messages"]

    tool = history[-1]["parts"][-1]
    assert tool["kind"] == "tool" and tool["name"] == "run_train"
    assert tool["status"] == "error" and tool["preview"] == "interrupted"


def test_a_jobless_experiment_still_appears_in_the_status_poll(
    studio: tuple[TestClient, _ScriptedAgent, Path],
) -> None:
    """The status poll is also how a browser discovers an experiment created elsewhere; one that has not
    run anything yet must not be invisible for exactly as long as it is being configured."""
    client, _agent, _session = studio

    statuses = client.get("/api/sessions/status").json()["statuses"]

    assert statuses == {"exp": ""}
