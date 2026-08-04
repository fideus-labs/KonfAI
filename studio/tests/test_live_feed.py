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

"""Which runs the Live feed announces, and the viewer signals a turn produces.

One rule decides the first: a run is announced ahead of its log only when the job already knows what
that run is called. A workflow job does (it names its own runtime log); an app job does not (its log
directory is named after the app, not after the job), so announcing it would put one run in two tabs.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")
from konfai_studio.agent import with_volume_events


def job_record(session: Path, **fields: Any) -> None:
    job_dir = session / ".konfai_mcp" / "jobs" / str(fields.get("job_id", "job-1"))
    job_dir.mkdir(parents=True, exist_ok=True)
    payload = {"job_id": "job-1", "status": "running", "created_at": 1.0, "pid": 1}
    (job_dir / "job.json").write_text(json.dumps({**payload, **fields}), encoding="utf-8")


def feed_events(session: Path, monkeypatch: pytest.MonkeyPatch, root: Path) -> list[dict[str, Any]]:
    """Everything the live generator emits for a session in one pass."""
    monkeypatch.setenv("KONFAI_MCP_WORKSPACES_ROOT", str(root))
    from konfai_studio.jobs import live

    async def collect() -> list[dict[str, Any]]:
        response = await live(session=session.name)
        events: list[dict[str, Any]] = []
        stream = response.body_iterator

        async def drain() -> None:
            async for chunk in stream:
                text = chunk if isinstance(chunk, str) else chunk.decode()
                for line in text.splitlines():
                    if line.startswith("data: "):
                        events.append(json.loads(line[6:]))
                if len(events) > 6:
                    break

        try:
            # wait_for, not asyncio.timeout: the package supports 3.10 and the latter is 3.11+.
            await asyncio.wait_for(drain(), timeout=2)
        except (TimeoutError, asyncio.TimeoutError, asyncio.CancelledError):
            pass
        finally:
            await stream.aclose()
        return events

    return asyncio.run(collect())


def announced_runs(session: Path, monkeypatch: pytest.MonkeyPatch, root: Path) -> list[dict[str, Any]]:
    """The `run` events the feed emits for a session."""
    return [event for event in feed_events(session, monkeypatch, root) if event.get("type") == "run"]


def test_a_workflow_job_gets_its_tab_before_it_writes_anything(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A training names its own runtime log, so its run is known at launch — the tab exists through the
    warm-up, and even if it dies before its first iteration."""
    session = tmp_path / "sessions" / "exp"
    session.mkdir(parents=True)
    job_record(session, kind="train", run_name="MR2CT", runtime_log_path=str(session / "Statistics/MR2CT/log_0.txt"))

    runs = announced_runs(session, monkeypatch, tmp_path)

    assert [(run["run"], run["kind"]) for run in runs] == [("MR2CT", "train")]


def test_a_transform_run_carries_where_its_data_went(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A transform's run directory holds a log, a plan and a config copy — never the volumes.

    They land wherever each `Write:` pointed, which the run records in `outputs.json`. Without the
    `data` field the panel would answer "what did it produce" by opening a YAML file, so the field
    has to reach the client, not merely exist on the server.
    """
    session = tmp_path / "sessions" / "exp"
    run = session / "Transforms" / "CT_PREP"
    run.mkdir(parents=True)
    (run / "log_0.txt").write_text("log\n", encoding="utf-8")
    (run / "outputs.json").write_text(
        json.dumps([{"group_dest": "CT_prep", "dataset": str(session / "Prepared"), "format": "omezarr"}]),
        encoding="utf-8",
    )

    runs = announced_runs(session, monkeypatch, tmp_path)

    assert [(run["run"], run["kind"], run["data"]) for run in runs] == [("CT_PREP", "transform", "Prepared")]


def test_a_run_whose_data_is_its_own_directory_carries_no_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every other kind writes under its run directory, and the client keeps its usual path."""
    session = tmp_path / "sessions" / "exp"
    session.mkdir(parents=True)
    job_record(session, kind="train", run_name="MR2CT", runtime_log_path=str(session / "Statistics/MR2CT/log_0.txt"))

    assert all(run["data"] == "" for run in announced_runs(session, monkeypatch, tmp_path))


def test_an_app_job_is_not_announced_under_a_name_it_will_not_keep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An app job is `app_MR` but its log lands in `ImpactSynth/`. Announcing it would give one run two
    tabs, and every metric would arrive in the other one."""
    session = tmp_path / "sessions" / "exp"
    session.mkdir(parents=True)
    job_record(session, kind="infer", run_name="app_MR", output_path=str(session / "AppOutputs/app_MR-a1b2"))

    assert announced_runs(session, monkeypatch, tmp_path) == []


def test_an_app_run_appears_once_it_has_written_its_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """...and it appears under the name it actually has, the moment there is something to follow."""
    session = tmp_path / "sessions" / "exp"
    output = session / "AppOutputs" / "app_MR-a1b2" / "ImpactSynth"
    output.mkdir(parents=True)
    (output / "log_0.txt").write_text("Prediction : 0% 0/13 [00:00<?, ?it/s]\n", encoding="utf-8")
    job_record(session, kind="infer", run_name="app_MR", output_path=str(session / "AppOutputs/app_MR-a1b2"))

    runs = announced_runs(session, monkeypatch, tmp_path)

    assert [(run["run"], run["kind"]) for run in runs] == [("ImpactSynth", "infer")]


def test_a_run_that_died_says_so_even_though_it_was_announced_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A workflow run is announced at launch AND followed through its log. When two blocks each marked
    the transition as seen, the first won and the terminal `status` never went out — Stop stayed on
    screen over a finished run, and every panel keyed on it stopped refreshing."""
    session = tmp_path / "sessions" / "exp"
    log = session / "Statistics" / "MR2CT" / "log_0.txt"
    log.parent.mkdir(parents=True)
    log.write_text("Train : 0% 0/75 [00:00<?, ?it/s]\n", encoding="utf-8")
    job_record(session, kind="train", run_name="MR2CT", runtime_log_path=str(log), status="error", returncode=1)

    events = feed_events(session, monkeypatch, tmp_path)

    terminal = [event for event in events if event.get("type") == "status"]
    assert [(event["run"], event["status"]) for event in terminal] == [("MR2CT", "error")]


# --- what the turn tells the viewer ------------------------------------------------------------------


async def _events(script: list[dict[str, Any]]) -> AsyncIterator[dict[str, Any]]:
    for event in script:
        yield event


def volume_events(script: list[dict[str, Any]]) -> list[dict[str, Any]]:
    async def run() -> list[dict[str, Any]]:
        return [event async for event in with_volume_events(_events(script)) if event["type"] == "volume"]

    return asyncio.run(run())


def shown(path: str) -> dict[str, Any]:
    """The assistant asking to see a volume: it names the path in the tool's arguments."""
    return {"type": "tool_call", "name": "preview_volume", "input": {"path": path}}


def said(path: str) -> dict[str, Any]:
    return {"type": "text", "text": f"The synthetic CT is at {path}."}


def test_two_volumes_across_a_turn_open_a_comparison(tmp_path: Path) -> None:
    """ "Show me the sCT next to the real CT" surfaces the two paths in two different steps, so they are
    counted over the turn — not within one step, where only one of them ever appears."""
    sct, ct = tmp_path / "sCT.mha", tmp_path / "CT.mha"
    for volume in (sct, ct):
        volume.write_bytes(b"\x00")

    events = volume_events([shown(str(sct)), shown(str(ct)), {"type": "done"}])

    assert [(event["path"], event["compare"]) for event in events] == [
        (str(sct), None),  # the first shows at once
        (str(sct), str(ct)),  # the second turns the view into a comparison
    ]


def test_a_listing_of_volumes_is_not_a_comparison(tmp_path: Path) -> None:
    volumes = [tmp_path / f"case_{i}.mha" for i in range(4)]
    for volume in volumes:
        volume.write_bytes(b"\x00")

    listed = {"type": "text", "text": " ".join(str(v) for v in volumes)}
    events = volume_events([listed, {"type": "done"}])

    assert [event["compare"] for event in events] == [None], "more than two is a listing, not a comparison"


def test_the_same_volume_twice_is_not_a_comparison(tmp_path: Path) -> None:
    volume = tmp_path / "CT.mha"
    volume.write_bytes(b"\x00")

    events = volume_events([shown(str(volume)), shown(str(volume)), {"type": "done"}])

    assert [(event["path"], event["compare"]) for event in events] == [(str(volume), None)]


def test_the_viewer_follows_the_prose_the_user_reads(tmp_path: Path) -> None:
    """The system prompt promises "any path you surface loads there by itself". Only tool arguments were
    ever scanned, so a path named in the reply itself did nothing."""
    sct = tmp_path / "sCT.mha"
    sct.write_bytes(b"\x00")

    events = volume_events([said(str(sct)), {"type": "done"}])

    assert [event["path"] for event in events] == [str(sct)]


def test_the_result_pair_wins_over_the_input_the_run_was_given(tmp_path: Path) -> None:
    """A run names its input volume in its own arguments, then names its result beside the ground truth.
    Counting only across the turn, that input burned the one slot the pane had: the comparison the
    assistant announced never opened, and the user was left looking at the MR they started from."""
    mr, sct, ct = tmp_path / "MR.mha", tmp_path / "sCT.mha", tmp_path / "CT.mha"
    for volume in (mr, sct, ct):
        volume.write_bytes(b"\x00")

    events = volume_events(
        [
            {"type": "tool_call", "name": "run_app_infer", "input": {"input_path": str(mr)}},
            {"type": "text", "text": f"sCT: {sct}\nreal CT: {ct}"},
            {"type": "done"},
        ]
    )

    assert (events[-1]["path"], events[-1]["compare"]) == (str(sct), str(ct))


def test_an_inventory_does_not_choose_what_the_viewer_shows(tmp_path: Path) -> None:
    """inspect_dataset names one file per group to prove the group exists. Reading the viewer off that
    payload opened whichever group sorted first — a body mask present in 30 of 75 cases — and the MR and
    CT the assistant then previewed never reached the pane."""
    case = tmp_path / "1ABB006"
    case.mkdir()
    for name in ("BODY_MASK.mha", "CT.mha", "MR.mha"):
        (case / name).write_bytes(b"\x00")
    inventory = {
        "type": "tool_result",
        "name": "inspect_dataset",
        "ok": True,
        "preview": json.dumps(
            {"groups": {path.stem: {"sample_path": str(path)} for path in sorted(case.iterdir())}},
        ),
    }

    events = volume_events([inventory, shown(str(case / "MR.mha")), {"type": "done"}])

    assert [event["path"] for event in events] == [str(case / "MR.mha")]


def text_parts(script: list[dict[str, Any]]) -> list[str]:
    async def run() -> list[str]:
        return [event["text"] async for event in with_volume_events(_events(script)) if event["type"] == "text"]

    return asyncio.run(run())


def test_a_tool_call_does_not_cut_the_sentence_in_half() -> None:
    """The marker can only arrive at the end of the reply, so only a genuine partial marker is worth
    holding back. Holding a fixed seven characters stranded them on the far side of every tool block:
    "Launching training on GPU" | run_train | "0 now." — one sentence, cut mid-word, in every reply."""
    parts = text_parts(
        [
            {"type": "text", "text": "Launching training on GPU 0 now."},
            {"type": "tool_call", "name": "run_train", "input": {}},
            {"type": "text", "text": "Job is up.\n<<NEXT>>\nWait :: Wait for it.\n"},
            {"type": "done"},
        ]
    )

    assert parts[0] == "Launching training on GPU 0 now."
    assert "".join(parts).strip() == "Launching training on GPU 0 now.Job is up."


def test_a_marker_split_across_chunks_still_never_reaches_the_user() -> None:
    parts = text_parts(
        [
            {"type": "text", "text": "Done.\n<<NE"},
            {"type": "text", "text": "XT>>\nWait :: Wait for it.\n"},
            {"type": "done"},
        ]
    )

    assert "".join(parts).strip() == "Done."


def test_a_tool_that_answers_with_an_image_does_not_read_as_empty() -> None:
    """preview_volume returns an image block. Dropping non-text content made a working call show up in
    the chat as "(empty result)", which reads like a failure."""
    from konfai_studio.agent import _stringify_content

    assert _stringify_content([{"type": "image", "data": "..."}]) == "(image returned)"
    assert _stringify_content([{"text": "75 cases"}, {"type": "image"}]) == "75 cases\n(image returned)"
    assert _stringify_content("plain text") == "plain text"


def test_a_payload_studio_parses_itself_is_never_truncated() -> None:
    """The cap keeps one oversized tool result from crowding out the model's conversation. A result
    Studio decodes itself must not be cut: JSON severed at any length stops being JSON, and the caller
    cannot tell a truncated payload from an empty one. A 64 328-character leaderboard came back as
    "no evaluations yet" over two runs that had been scored."""
    from konfai_studio.agent import _MODEL_RESULT_LIMIT, _tool_result_text

    class _Block:
        def __init__(self, text: str) -> None:
            self.text = text

    class _Result:
        data = None

        def __init__(self, text: str) -> None:
            self.content = [_Block(text)]

    payload = json.dumps({"metrics": ["m" * 100] * 700})
    assert len(payload) > _MODEL_RESULT_LIMIT

    assert len(_tool_result_text(_Result(payload))) == _MODEL_RESULT_LIMIT  # headed for the model
    assert json.loads(_tool_result_text(_Result(payload), limit=None)) == json.loads(payload)


def test_the_pane_follows_the_last_pair_named_however_it_is_streamed(tmp_path: Path) -> None:
    """A long reply arrives in chunks, so the two paths of a comparison can land in two text events.
    An uncertainty run named its map beside the sCT and the pane stayed on the inference stack its own
    arguments had mentioned first."""
    stack, unc, sct = tmp_path / "InferenceStack.mha", tmp_path / "Uncertainty.mha", tmp_path / "sCT.mha"
    for volume in (stack, unc, sct):
        volume.write_bytes(b"\x00")

    events = volume_events(
        [
            {"type": "tool_call", "name": "run_app_uncertainty", "input": {"inputs": [str(stack)]}},
            {"type": "text", "text": f"Carte d'incertitude : {unc}"},
            {"type": "text", "text": f"\nsCT synthétique : {sct}"},
            {"type": "done"},
        ]
    )

    assert (events[-1]["path"], events[-1]["compare"]) == (str(unc), str(sct))


def test_a_run_input_can_be_half_of_the_comparison_that_follows(tmp_path: Path) -> None:
    """A registration names its fixed and moving images in its own arguments, then cites the moved image
    beside that same fixed one. Deduplicating over the turn dropped the second mention, so the pane paired
    the moving image with the result instead of the result with its reference."""
    mr, ct, moved = tmp_path / "MR.mha", tmp_path / "CT.mha", tmp_path / "Moved.mha"
    for volume in (mr, ct, moved):
        volume.write_bytes(b"\x00")

    events = volume_events(
        [
            {"type": "tool_call", "name": "run_app_infer", "input": {"fixed": str(ct), "moving": str(mr)}},
            {"type": "text", "text": f"IRM recalée : {moved}"},
            {"type": "text", "text": f"\nCT (référence) : {ct}"},
            {"type": "done"},
        ]
    )

    assert (events[-1]["path"], events[-1]["compare"]) == (str(moved), str(ct))
