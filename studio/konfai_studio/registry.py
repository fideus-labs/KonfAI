# SPDX-License-Identifier: Apache-2.0
"""One agent per task (konfai-mcp session). ``_Registry`` holds a single ``dict[str, SessionState]``;
the on-disk ``sessions.json`` schema is unchanged so previously persisted sessions still load."""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from functools import partial
from typing import Any

from .agent import make_agent
from .paths import _delete_workspace, _sessions_file, _workspace_root


def _valid_device(value: str) -> str:
    """Normalise a device selection to 'auto', 'cpu', or a CSV of GPU indices ('0' / '0,1'); '' if invalid."""
    value = value.strip()
    if value in {"auto", "cpu"}:
        return value
    parts = [p.strip() for p in value.split(",") if p.strip()]
    return ",".join(parts) if parts and all(p.isdigit() for p in parts) else ""


@dataclass
class SessionState:
    """Everything the registry keys per task: its title, whether the LLM/user named it, the Claude Code
    transcript id (resume-on-restart), its dataset path, its compute device, the live agent, the turn
    lock, and whether the agent must be rebuilt on next use."""

    title: str | None = None
    named: bool = False  # title set by the LLM/user (final)
    sdk_id: str | None = None
    dataset: str = ""
    device: str | None = None  # None = use the registry default
    agent: Any = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    stale: bool = False


class _Registry:
    """One agent per task (konfai-mcp session). Agents are created lazily and run concurrently;
    a per-session lock serialises turns within a task, never across tasks."""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}
        self._order: list[str] = []  # creation order; names() returns it newest-first
        self._create = asyncio.Lock()
        self._brain = os.environ.get("KONFAI_STUDIO_LLM", "claude-code").lower()
        self._model = os.environ.get("KONFAI_STUDIO_MODEL", "")
        self._device = "auto"  # default compute device for a fresh experiment

    def names(self) -> list[str]:
        """Experiments newest-first (creation order reversed); any untracked ids trail, sorted."""
        ordered = [n for n in reversed(self._order) if n in self._sessions]
        rest = sorted(n for n in self._sessions if n not in set(self._order))
        return ordered + rest

    def title(self, name: str) -> str:
        state = self._sessions.get(name)
        return state.title if state is not None and state.title is not None else name

    def titles(self) -> dict[str, str]:
        return {name: self.title(name) for name in self._sessions}

    def is_untitled(self, name: str) -> bool:
        """A session the LLM has not named yet — its title is still the id or a placeholder."""
        state = self._sessions.get(name)
        return not (state is not None and state.named)

    def set_title(self, name: str, title: str) -> None:
        state = self._sessions.setdefault(name, SessionState())
        state.title = title
        state.named = True
        self._save()

    def new_experiment(self) -> str:
        """Allocate a fresh, stable experiment id (its konfai-mcp workspace); the LLM titles it later."""
        n = 1
        while f"experiment-{n}" in self._sessions:
            n += 1
        name = f"experiment-{n}"
        self.register(name, "New experiment")
        return name

    def register(self, name: str, title: str | None = None) -> None:
        is_new = name not in self._order
        state = self._sessions.setdefault(name, SessionState())
        if is_new:
            self._order.append(name)
        if title and not state.named:
            state.title = title
        if is_new:
            self._save()

    def lock(self, name: str) -> asyncio.Lock:
        return self._sessions.setdefault(name, SessionState()).lock

    def is_ready(self, name: str) -> bool:
        state = self._sessions.get(name)
        return state is not None and state.agent is not None

    def brain(self) -> str:
        return self._brain

    def set_brain(self, brain: str) -> None:
        """Change the LLM for future turns. Existing agents are marked stale and rebuilt on their
        next turn (under the session lock, so a streaming turn is never torn down mid-flight)."""
        if brain != self._brain:
            self._brain = brain
            for state in self._sessions.values():
                if state.agent is not None:
                    state.stale = True
            self._save()

    def model(self) -> str:
        return self._model

    def set_model(self, model: str) -> None:
        """Pin the LLM model ('' = the backend's default). Same lazy-rebuild as a brain switch —
        conversation continuity survives it (SDK resume / persisted history)."""
        if model != self._model:
            self._model = model
            for state in self._sessions.values():
                if state.agent is not None:
                    state.stale = True
            self._save()

    def device(self, name: str) -> str:
        """The task's compute device, falling back to the default used for a fresh experiment."""
        state = self._sessions.get(name)
        return state.device if state is not None and state.device else self._device

    def devices(self) -> dict[str, str]:
        return {n: s.device for n, s in self._sessions.items() if s.device}

    def set_device(self, name: str, device: str) -> None:
        """Preferred compute device(s) for a task's jobs ('auto', 'cpu', or GPU indices '0'/'0,1'). Applied
        as a per-turn directive to the agent — no rebuild, so switching device keeps the conversation intact."""
        value = _valid_device(device) or "auto"
        state = self._sessions.setdefault(name, SessionState())
        if state.device != value:
            state.device = value
            self._save()

    def _set_sdk_id(self, name: str, sdk_id: str) -> None:
        state = self._sessions.get(name)
        if state is not None and sdk_id and state.sdk_id != sdk_id:
            state.sdk_id = sdk_id
            self._save()

    def dataset(self, name: str) -> str:
        state = self._sessions.get(name)
        return state.dataset if state is not None else ""

    def datasets(self) -> dict[str, str]:
        return {n: s.dataset for n, s in self._sessions.items() if s.dataset}

    def set_dataset(self, name: str, path: str) -> None:
        state = self._sessions.setdefault(name, SessionState())
        if path and state.dataset != path:
            state.dataset = path
            self._save()

    def invalidate(self, name: str) -> None:
        """Mark an agent for rebuild on its next use. A stream error can leave the SDK client unusable, so
        reusing it hangs the next turn; the rebuild resumes the transcript, so the conversation continues."""
        state = self._sessions.get(name)
        if state is not None and state.agent is not None:
            state.stale = True

    async def agent(self, name: str) -> Any:
        self.register(name)
        state = self._sessions[name]
        async with self._create:
            if state.stale:
                state.stale = False
                old = state.agent
                state.agent = None
                if old is not None:
                    try:
                        await old.__aexit__(None, None, None)
                    except Exception:
                        pass
            if state.agent is None:
                state.agent = await make_agent(
                    name,
                    brain=self._brain,
                    model=self._model or None,
                    resume=state.sdk_id,
                    on_session_id=partial(self._set_sdk_id, name),
                    history_file=_workspace_root() / "sessions" / name / ".konfai_studio" / "history.json",
                ).__aenter__()
        return state.agent

    async def remove(self, name: str) -> bool:
        """Delete a task and its workspace. No task is special — a fresh draft is always one click away.
        The session lock is held so a streaming turn is never torn down mid-flight."""
        async with self.lock(name):
            state = self._sessions.get(name)
            agent = state.agent if state is not None else None
            if state is not None:
                state.agent = None
            if agent is not None:
                try:
                    await agent.__aexit__(None, None, None)
                except Exception:
                    pass
        self._sessions.pop(name, None)
        if name in self._order:
            self._order.remove(name)
        _delete_workspace(name)
        self._save()
        return True

    async def close(self) -> None:
        for state in self._sessions.values():
            if state.agent is not None:
                try:
                    await state.agent.__aexit__(None, None, None)
                except Exception:
                    pass

    def load(self) -> None:
        """Restore the session list + titles from disk and surface any konfai-mcp workspace dirs, so
        a restart keeps every task the user started (their jobs/configs already live on disk)."""
        try:
            data = json.loads(_sessions_file().read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
        if not isinstance(data, dict):
            data = {}

        def state(name: str) -> SessionState:
            return self._sessions.setdefault(name, SessionState())

        titles = data.get("titles")
        if isinstance(titles, dict):
            for key, value in titles.items():
                if isinstance(key, str):
                    state(key).title = str(value)
        named = data.get("named")
        if isinstance(named, list):
            for name in named:
                if isinstance(name, str):
                    state(name).named = True
        sdk_ids = data.get("sdk_ids")
        if isinstance(sdk_ids, dict):
            for key, value in sdk_ids.items():
                if isinstance(key, str):
                    state(key).sdk_id = str(value)
        datasets = data.get("datasets")
        if isinstance(datasets, dict):
            for key, value in datasets.items():
                if isinstance(key, str):
                    state(key).dataset = str(value)
        device = data.get("device")
        if isinstance(device, str) and _valid_device(device):
            self._device = _valid_device(device)
        devices = data.get("devices")
        if isinstance(devices, dict):
            for key, value in devices.items():
                norm = _valid_device(str(value))
                if isinstance(key, str) and norm:
                    state(key).device = norm
        brain = data.get("brain")
        if isinstance(brain, str) and brain:
            self._brain = brain
        model = data.get("model")
        if isinstance(model, str):
            self._model = model
        sessions_dir = _workspace_root() / "sessions"
        if sessions_dir.is_dir():
            for child in sessions_dir.iterdir():
                if child.is_dir() and not child.name.startswith("."):
                    state(child.name)
        order = data.get("order")
        if isinstance(order, list):
            self._order = [n for n in order if isinstance(n, str) and n]
        for name in sorted(self._sessions):  # append any workspace dir not in the persisted order
            if name not in self._order:
                self._order.append(name)
        self._order = [n for n in self._order if n in self._sessions]

    def _save(self) -> None:
        target = _sessions_file()
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "order": [n for n in self._order if n in self._sessions],
                "titles": {n: self.title(n) for n in self._sessions},
                "named": sorted(n for n, s in self._sessions.items() if s.named),
                "sdk_ids": {n: s.sdk_id for n, s in self._sessions.items() if s.sdk_id},
                "datasets": {n: s.dataset for n, s in self._sessions.items() if s.dataset},
                "device": self._device,
                "devices": {n: s.device for n, s in self._sessions.items() if s.device},
                "brain": self._brain,
                "model": self._model,
            }
            target.write_text(json.dumps(payload), encoding="utf-8")
        except OSError:
            pass
