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
"""The server-side chat transcript, one JSON file per experiment.

The browser keeps its copy in localStorage, which is per device: without a server copy, an experiment
driven on one machine shows empty on every other. The server records every turn it streams and serves
it back: the browser with the shorter history adopts it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_FILE = "transcript.json"
_KEPT = 200  # the browser keeps as many; older turns age out of both copies together
_INPUT_SHOWN = 2000  # a tool input is context for the reader, not an archive of the config it carried


def note_event(parts: list[dict[str, Any]], event: dict[str, Any]) -> None:
    """Fold one streamed event into the assistant's parts, the same shapes the browser renders."""
    kind = event.get("type")
    if kind == "text":
        if parts and parts[-1].get("kind") == "text":
            parts[-1]["text"] += str(event.get("text") or "")
        else:
            parts.append({"kind": "text", "text": str(event.get("text") or "")})
    elif kind == "tool_call":
        raw = json.dumps(event.get("input"), ensure_ascii=False, default=str)
        stored = event.get("input") if len(raw) <= _INPUT_SHOWN else raw[:_INPUT_SHOWN]
        parts.append(
            {"kind": "tool", "name": str(event.get("name") or ""), "input": stored, "status": "running", "preview": ""}
        )
    elif kind == "tool_result":
        for part in reversed(parts):
            if part.get("kind") == "tool" and part.get("name") == event.get("name") and part.get("status") == "running":
                part["status"] = "done" if event.get("ok") else "error"
                part["preview"] = str(event.get("preview") or "")
                break
    elif kind == "error":
        parts.append({"kind": "error", "text": str(event.get("message") or "")})


def record_turn(session_dir: Path, user_text: str, parts: list[dict[str, Any]]) -> None:
    """Append one turn, atomically. A tool still 'running' here was cut by an interrupt: recorded as
    such rather than left spinning forever on the browser that reloads it."""
    for part in parts:
        if part.get("kind") == "tool" and part.get("status") == "running":
            part["status"] = "error"
            part["preview"] = "interrupted"
    messages = load_transcript(session_dir)
    messages.append({"role": "user", "text": user_text})
    messages.append({"role": "assistant", "parts": parts})
    session_dir.mkdir(parents=True, exist_ok=True)
    path = session_dir / _FILE
    staging = path.with_name(f"{_FILE}.{os.getpid()}.tmp")
    staging.write_text(json.dumps({"messages": messages[-_KEPT:]}, ensure_ascii=False), encoding="utf-8")
    os.replace(staging, path)


def load_transcript(session_dir: Path) -> list[dict[str, Any]]:
    try:
        loaded = json.loads((session_dir / _FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    messages = loaded.get("messages")
    return messages if isinstance(messages, list) else []
