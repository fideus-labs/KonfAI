# SPDX-License-Identifier: Apache-2.0
"""FastAPI BFF: one agent per task (konfai-mcp session), streamed to the browser.

Localhost, co-located with konfai-mcp on the compute node. Each task is an isolated konfai-mcp
session with its own agent, chat, jobs and workspace; tasks run concurrently. The BFF streams
the chat (`/api/chat`) and a job's live log (`/api/live`) over SSE, and serves the built front.
"""

from __future__ import annotations

import asyncio
import difflib
import hashlib
import hmac
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from datetime import datetime
from http.cookies import SimpleCookie
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from konfai_mcp.live_parse import parse_host_stats, parse_live_metric_line, parse_live_progress, progress_label
from pydantic import BaseModel

from .agent import call_mcp_tool, make_agent, suggest_next_prompts, suggest_title

WEB_DIR = Path(__file__).parent / "web"


def _sane_session(name: str) -> str:
    """Sanitize a session name to a safe workspace dir (mirrors konfai-mcp's own rule)."""
    cleaned = re.sub(r"[^a-zA-Z0-9._-]", "_", (name or "").strip())
    return cleaned if cleaned and cleaned not in {".", ".."} else "default"


def _jail(root: Path, rel: str) -> Path | None:
    """Resolve ``root/rel`` and return it only when it stays under ``root`` (else None)."""
    base = root.resolve()
    target = (base / rel).resolve() if rel else base
    return target if target == base or base in target.parents else None


# --- Access gate (remote deployments) -------------------------------------------------------------
# Studio drives konfai-mcp, which reads arbitrary host paths and runs jobs — arbitrary compute by
# design. On loopback that is the operator's own machine; exposed on a network it is not. A single
# shared token (KONFAI_STUDIO_TOKEN) turns on authentication: unset, everything is open exactly as
# before (trusted-local); set, every request must carry a valid session cookie or bearer token. TLS
# is the reverse proxy's job (see docs/REMOTE.md).
_COOKIE_NAME = "ks_session"
# The app shell + login surface are reachable without a session; everything else needs one.
_PUBLIC_PATHS = frozenset(
    {"/", "/index.html", "/api/auth", "/api/login", "/api/health", "/konfai-logo.png", "/favicon.ico"}
)


def _studio_token() -> str:
    """The shared access token guarding a remote deployment ('' = auth disabled, trusted-local)."""
    return os.environ.get("KONFAI_STUDIO_TOKEN", "").strip()


def _session_cookie(token: str) -> str:
    """A stable, non-reversible session value derived from the token — what the auth cookie carries, so
    the raw token never lives in the browser and a server restart keeps the user signed in."""
    return hmac.new(token.encode(), b"konfai-studio-session", hashlib.sha256).hexdigest()


def _scope_header(scope: dict[str, Any], name: bytes) -> str | None:
    for key, value in scope.get("headers", []):
        if key == name:
            return value.decode("latin-1")
    return None


def _authorised(scope: dict[str, Any]) -> bool:
    """Whether an ASGI request may proceed: always when auth is off, for the public shell paths, or when
    it presents the session cookie / bearer token. Comparisons are constant-time."""
    token = _studio_token()
    if not token:
        return True
    path = scope.get("path", "")
    # Exact public paths, or a static asset — but never a dot-segment path, which a raw client could use to
    # make the gate ("/assets/…", allowed) and the router disagree on the effective route. Fail closed.
    if path in _PUBLIC_PATHS or (path.startswith("/assets/") and "/.." not in path):
        return True
    expected = _session_cookie(token)
    raw = _scope_header(scope, b"cookie")
    if raw:
        jar: SimpleCookie[str] = SimpleCookie()
        with suppress(Exception):
            jar.load(raw)
        morsel = jar.get(_COOKIE_NAME)
        # Compare as bytes: hmac.compare_digest raises TypeError on a non-ASCII str, and both the cookie
        # value and the bearer token are attacker-controlled — bytes yield a constant-time False instead.
        if morsel and hmac.compare_digest(morsel.value.encode(), expected.encode()):
            return True
    auth = _scope_header(scope, b"authorization") or ""
    if auth.lower().startswith("bearer ") and hmac.compare_digest(auth[7:].strip().encode(), token.encode()):
        return True
    return False


class _AuthGate:
    """Blanket access gate for remote deployments. A pure ASGI middleware (not ``BaseHTTPMiddleware``) so
    it never wraps the SSE/stream responses — it inspects the request and either passes it through
    untouched or short-circuits with a 401 / WebSocket close. No-op when the token is unset."""

    def __init__(self, app: Any) -> None:
        self._app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] not in {"http", "websocket"} or _authorised(scope):
            await self._app(scope, receive, send)
            return
        if scope["type"] == "websocket":
            await receive()  # consume the connect so the close handshake is well-formed
            await send({"type": "websocket.close", "code": 1008})
            return
        await JSONResponse({"detail": "authentication required"}, status_code=401)(scope, receive, send)


class _Registry:
    """One agent per task (konfai-mcp session). Agents are created lazily and run concurrently;
    a per-session lock serialises turns within a task, never across tasks."""

    def __init__(self) -> None:
        self._agents: dict[str, Any] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._known: set[str] = set()
        self._order: list[str] = []  # creation order; names() returns it newest-first
        self._titles: dict[str, str] = {}
        self._named: set[str] = set()  # sessions whose title was set by the LLM/user (final)
        self._create = asyncio.Lock()
        self._brain = os.environ.get("KONFAI_STUDIO_LLM", "claude-code").lower()
        self._model = os.environ.get("KONFAI_STUDIO_MODEL", "")
        self._device = "auto"  # default compute device for a fresh experiment
        self._devices: dict[str, str] = {}  # compute device per task (each experiment picks its own)
        self._sdk_ids: dict[str, str] = {}  # Claude Code transcript id per task, for resume-on-restart
        self._datasets: dict[str, str] = {}  # dataset path per task (lives outside the workspace)
        self._stale: set[str] = set()

    def names(self) -> list[str]:
        """Experiments newest-first (creation order reversed); any untracked ids trail, sorted."""
        ordered = [n for n in reversed(self._order) if n in self._known]
        rest = sorted(n for n in self._known if n not in set(self._order))
        return ordered + rest

    def title(self, name: str) -> str:
        return self._titles.get(name, name)

    def titles(self) -> dict[str, str]:
        return {name: self.title(name) for name in self._known}

    def is_untitled(self, name: str) -> bool:
        """A session the LLM has not named yet — its title is still the id or a placeholder."""
        return name not in self._named

    def set_title(self, name: str, title: str) -> None:
        self._titles[name] = title
        self._named.add(name)
        self._save()

    def new_experiment(self) -> str:
        """Allocate a fresh, stable experiment id (its konfai-mcp workspace); the LLM titles it later."""
        n = 1
        while f"experiment-{n}" in self._known:
            n += 1
        name = f"experiment-{n}"
        self.register(name, "New experiment")
        return name

    def register(self, name: str, title: str | None = None) -> None:
        is_new = name not in self._known
        self._known.add(name)
        if is_new:
            self._order.append(name)
        if title and name not in self._named:
            self._titles[name] = title
        if is_new:
            self._save()

    def lock(self, name: str) -> asyncio.Lock:
        return self._locks.setdefault(name, asyncio.Lock())

    def is_ready(self, name: str) -> bool:
        return name in self._agents

    def brain(self) -> str:
        return self._brain

    def set_brain(self, brain: str) -> None:
        """Change the LLM for future turns. Existing agents are marked stale and rebuilt on their
        next turn (under the session lock, so a streaming turn is never torn down mid-flight)."""
        if brain != self._brain:
            self._brain = brain
            self._stale = set(self._agents)
            self._save()

    def model(self) -> str:
        return self._model

    def set_model(self, model: str) -> None:
        """Pin the LLM model ('' = the backend's default). Same lazy-rebuild as a brain switch —
        conversation continuity survives it (SDK resume / persisted history)."""
        if model != self._model:
            self._model = model
            self._stale = set(self._agents)
            self._save()

    def device(self, name: str) -> str:
        """The task's compute device, falling back to the default used for a fresh experiment."""
        return self._devices.get(name, self._device)

    def devices(self) -> dict[str, str]:
        return dict(self._devices)

    def set_device(self, name: str, device: str) -> None:
        """Preferred compute device(s) for a task's jobs ('auto', 'cpu', or GPU indices '0'/'0,1'). Applied
        as a per-turn directive to the agent — no rebuild, so switching device keeps the conversation intact."""
        value = _valid_device(device) or "auto"
        if self._devices.get(name) != value:
            self._devices[name] = value
            self._save()

    def _set_sdk_id(self, name: str, sdk_id: str) -> None:
        if sdk_id and self._sdk_ids.get(name) != sdk_id:
            self._sdk_ids[name] = sdk_id
            self._save()

    def dataset(self, name: str) -> str:
        return self._datasets.get(name, "")

    def datasets(self) -> dict[str, str]:
        return dict(self._datasets)

    def set_dataset(self, name: str, path: str) -> None:
        if path and self._datasets.get(name) != path:
            self._datasets[name] = path
            self._save()

    def invalidate(self, name: str) -> None:
        """Mark an agent for rebuild on its next use. A stream error can leave the SDK client unusable, so
        reusing it hangs the next turn; the rebuild resumes the transcript, so the conversation continues."""
        if name in self._agents:
            self._stale.add(name)

    async def agent(self, name: str) -> Any:
        self.register(name)
        async with self._create:
            if name in self._stale:
                self._stale.discard(name)
                old = self._agents.pop(name, None)
                if old is not None:
                    try:
                        await old.__aexit__(None, None, None)
                    except Exception:
                        pass
            if name not in self._agents:
                self._agents[name] = await make_agent(
                    name,
                    brain=self._brain,
                    model=self._model or None,
                    resume=self._sdk_ids.get(name),
                    on_session_id=lambda sid, n=name: self._set_sdk_id(n, sid),
                    history_file=_workspace_root() / "sessions" / name / ".konfai_studio" / "history.json",
                ).__aenter__()
        return self._agents[name]

    async def remove(self, name: str) -> bool:
        """Delete a task and its workspace. No task is special — a fresh draft is always one click away.
        The session lock is held so a streaming turn is never torn down mid-flight."""
        async with self.lock(name):
            agent = self._agents.pop(name, None)
            if agent is not None:
                try:
                    await agent.__aexit__(None, None, None)
                except Exception:
                    pass
        self._known.discard(name)
        if name in self._order:
            self._order.remove(name)
        self._titles.pop(name, None)
        self._named.discard(name)
        self._stale.discard(name)
        self._locks.pop(name, None)
        self._sdk_ids.pop(name, None)
        self._datasets.pop(name, None)
        self._devices.pop(name, None)
        _delete_workspace(name)
        self._save()
        return True

    async def close(self) -> None:
        for agent in self._agents.values():
            try:
                await agent.__aexit__(None, None, None)
            except Exception:
                pass

    def load(self) -> None:
        """Restore the session list + titles from disk and surface any konfai-mcp workspace dirs, so
        a restart keeps every task the user started (their jobs/configs already live on disk)."""
        try:
            data = json.loads(_sessions_file().read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
        titles = data.get("titles") if isinstance(data, dict) else None
        if isinstance(titles, dict):
            self._titles.update({k: str(v) for k, v in titles.items() if isinstance(k, str)})
            self._known.update(k for k in titles if isinstance(k, str))
        named = data.get("named") if isinstance(data, dict) else None
        if isinstance(named, list):
            self._named.update(n for n in named if isinstance(n, str))
        sdk_ids = data.get("sdk_ids") if isinstance(data, dict) else None
        if isinstance(sdk_ids, dict):
            self._sdk_ids.update({k: str(v) for k, v in sdk_ids.items() if isinstance(k, str)})
        datasets = data.get("datasets") if isinstance(data, dict) else None
        if isinstance(datasets, dict):
            self._datasets.update({k: str(v) for k, v in datasets.items() if isinstance(k, str)})
        device = data.get("device") if isinstance(data, dict) else None
        if isinstance(device, str) and _valid_device(device):
            self._device = _valid_device(device)
        devices = data.get("devices") if isinstance(data, dict) else None
        if isinstance(devices, dict):
            for key, value in devices.items():
                norm = _valid_device(str(value))
                if isinstance(key, str) and norm:
                    self._devices[key] = norm
        brain = data.get("brain") if isinstance(data, dict) else None
        if isinstance(brain, str) and brain:
            self._brain = brain
        model = data.get("model") if isinstance(data, dict) else None
        if isinstance(model, str):
            self._model = model
        sessions_dir = _workspace_root() / "sessions"
        if sessions_dir.is_dir():
            self._known.update(c.name for c in sessions_dir.iterdir() if c.is_dir() and not c.name.startswith("."))
        order = data.get("order") if isinstance(data, dict) else None
        if isinstance(order, list):
            self._order = [n for n in order if isinstance(n, str) and n]
        for name in sorted(self._known):  # append any workspace dir not in the persisted order
            if name not in self._order:
                self._order.append(name)
        self._order = [n for n in self._order if n in self._known]

    def _save(self) -> None:
        target = _sessions_file()
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "order": [n for n in self._order if n in self._known],
                "titles": {n: self.title(n) for n in self._known},
                "named": sorted(self._named),
                "sdk_ids": {n: self._sdk_ids[n] for n in self._sdk_ids if n in self._known},
                "datasets": {n: self._datasets[n] for n in self._datasets if n in self._known},
                "device": self._device,
                "devices": {n: self._devices[n] for n in self._devices if n in self._known},
                "brain": self._brain,
                "model": self._model,
            }
            target.write_text(json.dumps(payload), encoding="utf-8")
        except OSError:
            pass


_reg = _Registry()


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    _reg.load()  # restore tasks + titles from a previous run; agents spin up lazily per task
    try:
        yield
    finally:
        await _reg.close()


app = FastAPI(title="KonfAI Studio", lifespan=_lifespan)
app.add_middleware(_AuthGate)  # no-op unless KONFAI_STUDIO_TOKEN is set (trusted-local by default)


class ChatRequest(BaseModel):
    message: str
    session: str = "default"


class NewSession(BaseModel):
    name: str = ""


class DatasetPath(BaseModel):
    path: str


class LLMChoice(BaseModel):
    brain: str = ""
    model: str | None = None


class DeviceChoice(BaseModel):
    session: str = "default"
    device: str


class DeleteSession(BaseModel):
    name: str


class RenameSession(BaseModel):
    session: str
    title: str


class CancelJob(BaseModel):
    session: str = "default"


class DeleteRun(BaseModel):
    session: str = "default"
    run_name: str
    kind: str


class SetTunables(BaseModel):
    session: str = "default"
    lr: float | None = None
    it_validation: int | None = None


class BundleRequest(BaseModel):
    session: str
    output: str
    name: str = ""


class ExportRequest(BaseModel):
    session: str
    output: str


def _sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event)}\n\n"


def _valid_device(value: str) -> str:
    """Normalise a device selection to 'auto', 'cpu', or a CSV of GPU indices ('0' / '0,1'); '' if invalid."""
    value = value.strip()
    if value in {"auto", "cpu"}:
        return value
    parts = [p.strip() for p in value.split(",") if p.strip()]
    return ",".join(parts) if parts and all(p.isdigit() for p in parts) else ""


def _device_directive(device: str) -> str:
    """A one-line instruction that pins the compute device(s) the agent runs jobs on ('' = let it choose)."""
    if device == "cpu":
        return "Run every training/prediction/evaluation job on CPU (do not pass a gpu argument)."
    gpus = [p for p in device.split(",") if p.isdigit()]
    if gpus:
        arg = ", ".join(gpus)
        which = f"GPU {gpus[0]}" if len(gpus) == 1 else f"GPUs {arg} (multi-GPU / DDP)"
        return f"Run every training/prediction/evaluation job on {which} (pass gpu=[{arg}])."
    return ""


async def _mcp_detail(session: str, tool: str, args: dict[str, Any]) -> dict[str, Any]:
    """Proxy a konfai-mcp tool and surface its outcome as ``{ok, detail}``."""
    ok, text = await call_mcp_tool(session, tool, args)
    return {"ok": ok, "detail": text}


async def _mcp_json(session: str, tool: str, args: dict[str, Any]) -> tuple[bool, str, Any]:
    """Proxy a konfai-mcp tool: (ok, raw text, JSON-decoded value) — the value is {} on non-JSON text."""
    ok, text = await call_mcp_tool(session, tool, args)
    try:
        return ok, text, json.loads(text)
    except (TypeError, ValueError):
        return ok, text, {}


@app.post("/api/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    """Stream one user turn as Server-Sent Events. Serialised: one turn at a time."""

    name = _sane_session(req.session)
    directive = _device_directive(_reg.device(name))
    message = f"({directive})\n\n{req.message}" if directive else req.message

    async def gen() -> AsyncIterator[str]:
        async with _reg.lock(name):
            reply: list[str] = []  # the assistant's text this turn
            turn_actions: list[str] = []  # last tool next_actions of the turn
            try:
                agent = await _reg.agent(name)
                async for event in agent.send(message):
                    if event.get("type") == "text" and isinstance(event.get("text"), str):
                        reply.append(event["text"])
                    elif event.get("type") == "next_actions" and isinstance(event.get("actions"), list):
                        turn_actions = event["actions"]
                    yield _sse(event)
            except Exception as exc:
                _reg.invalidate(name)  # drop the broken SDK client so the next turn rebuilds and resumes
                yield _sse({"type": "error", "message": str(exc)})
                return
            # Machine-injected onboarding prompts (dataset inspection) don't describe the experiment —
            # wait for the user's own first message so the title reflects the real task.
            if _reg.is_untitled(name) and req.message.strip() and not req.message.startswith("Inspect the dataset at "):
                try:  # let the LLM name the experiment from its first prompt
                    title = await suggest_title(req.message, _reg.brain())
                    _reg.set_title(name, title)
                    yield _sse({"type": "title", "session": name, "title": title})
                except Exception:
                    pass
            # Suggest next prompts only on a turn that produced tool next_actions; skip conversational turns.
            if turn_actions:
                try:
                    prompts = await suggest_next_prompts(req.message, "".join(reply), turn_actions, _reg.brain())
                    if prompts:
                        yield _sse({"type": "next_prompts", "prompts": prompts})
                except Exception:
                    pass

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "agent": "ready"}


class LoginRequest(BaseModel):
    token: str


def _cookie_secure() -> bool:
    """The session cookie is Secure by default — a remote deployment must run behind TLS. Opt out only
    for local http testing of the auth flow with KONFAI_STUDIO_INSECURE_COOKIE=1."""
    return os.environ.get("KONFAI_STUDIO_INSECURE_COOKIE") != "1"


@app.get("/api/auth")
async def auth_state(request: Request) -> dict[str, bool]:
    """Whether this deployment requires a token, and whether the browser already holds a valid session —
    the front shows a lock screen when required and not yet authenticated."""
    token = _studio_token()
    if not token:
        return {"required": False, "authenticated": True}
    cookie = request.cookies.get(_COOKIE_NAME)
    ok = bool(cookie and hmac.compare_digest(cookie.encode(), _session_cookie(token).encode()))
    return {"required": True, "authenticated": ok}


@app.post("/api/login")
async def login(req: LoginRequest) -> Response:
    """Exchange the shared access token for an httpOnly session cookie. Constant-time compare; a wrong
    token is a flat 401 (the token's entropy, not rate-limiting, is the defence)."""
    token = _studio_token()
    if not token:
        return JSONResponse({"ok": True})  # auth disabled — nothing to unlock
    if not hmac.compare_digest(req.token.strip().encode(), token.encode()):
        raise HTTPException(401, "invalid access token")
    resp = JSONResponse({"ok": True})
    resp.set_cookie(
        _COOKIE_NAME,
        _session_cookie(token),
        max_age=60 * 60 * 24 * 30,
        httponly=True,
        samesite="lax",
        secure=_cookie_secure(),
        path="/",
    )
    return resp


@app.post("/api/logout")
async def logout() -> Response:
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(_COOKIE_NAME, path="/")
    return resp


@app.get("/api/sessions")
async def list_sessions() -> dict[str, Any]:
    return {"sessions": _reg.names(), "titles": _reg.titles(), "datasets": _reg.datasets()}


@app.get("/api/sessions/status")
async def sessions_status() -> dict[str, Any]:
    """Latest job status per experiment, so the rail can colour each dot by its state."""
    statuses: dict[str, str] = {}
    for name in _reg.names():
        job = _latest_job(name)
        if job and job.get("status"):
            statuses[name] = _live_status(job)
    return {"statuses": statuses}


class SessionDataset(BaseModel):
    session: str
    path: str


@app.post("/api/sessions/dataset")
async def set_session_dataset(req: SessionDataset) -> dict[str, Any]:
    """Remember which dataset an experiment works on (it lives outside the workspace)."""
    _reg.set_dataset(_sane_session(req.session), req.path)
    return {"datasets": _reg.datasets()}


@app.post("/api/sessions")
async def create_session(req: NewSession) -> dict[str, Any]:
    """Start a new experiment. With no name, allocate a fresh id the LLM titles later; a supplied
    name is honoured (legacy). The agent spins up lazily on the first message."""
    name = _sane_session(req.name) if req.name.strip() else _reg.new_experiment()
    _reg.register(name)
    return {"sessions": _reg.names(), "current": name, "titles": _reg.titles()}


@app.post("/api/sessions/rename")
async def rename_session(req: RenameSession) -> dict[str, Any]:
    """Rename an experiment's display title (its workspace id is unchanged). Marks it as user-named so
    the LLM won't retitle it later."""
    title = req.title.strip()
    if not title:
        return {"ok": False, "titles": _reg.titles()}
    _reg.set_title(_sane_session(req.session), title)
    return {"ok": True, "titles": _reg.titles()}


@app.post("/api/sessions/delete")
async def delete_session(req: DeleteSession) -> dict[str, Any]:
    """Delete an experiment: forget it and remove its workspace (jobs, checkpoints). Irreversible."""
    removed = await _reg.remove(_sane_session(req.name))
    return {"sessions": _reg.names(), "titles": _reg.titles(), "removed": removed}


@app.post("/api/sessions/bundle")
async def bundle_session(req: BundleRequest) -> dict[str, Any]:
    """Package an experiment as a reusable app into a chosen folder — a direct MCP call, no LLM.

    Checkpoints/configs are discovered from the session; name/description default to its title.
    """
    name = _sane_session(req.session)
    title = _reg.title(name)
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", req.name or title).strip("_") or name
    ok, text = await call_mcp_tool(
        name,
        "package_app_from_session",
        {
            "name": slug,
            "display_name": title,
            "description": f"{title} — packaged from KonfAI Studio.",
            "output": req.output,
        },
    )
    # A fresh bundle is only useful if you can find it — register it into the KonfAI Apps catalogue.
    registered = False
    if ok:
        bundle = str(Path(req.output).expanduser() / slug)
        registered, _ = await call_mcp_tool("apps", "register_app_source", {"ref": bundle})
    return {"ok": ok, "result": text, "registered": registered}


@app.post("/api/sessions/export")
async def export_session(req: ExportRequest) -> dict[str, Any]:
    """Export an experiment as a self-contained copy of its workspace — config, code, checkpoints,
    statistics and metrics — minus the input Dataset (the user's data, which lives outside) and the
    Studio/MCP internals. A folder you can archive or share."""
    name = _sane_session(req.session)
    src = (_workspace_root() / "sessions" / name).resolve()
    if not src.is_dir():
        return {"ok": False, "result": "This experiment has no workspace yet."}
    dest = Path(req.output).expanduser() / _sane_session(_reg.title(name) or name)
    # Dataset is the user's input (often a symlink to data outside the workspace); .konfai_mcp holds job
    # bookkeeping. Everything else IS the experiment. ignore_patterns drops these at any nesting level.
    skip = {"Dataset", ".konfai_mcp", "__pycache__"}
    ignore = shutil.ignore_patterns(*skip)
    try:
        dest.mkdir(parents=True, exist_ok=True)
        for child in src.iterdir():
            if child.name in skip or child.is_symlink():
                continue
            if child.is_dir():
                shutil.copytree(child, dest / child.name, dirs_exist_ok=True, ignore=ignore)
            else:
                shutil.copy2(child, dest / child.name)
    except OSError as exc:
        return {"ok": False, "result": f"Export failed: {exc}"}
    return {"ok": True, "result": f"Experiment exported to {dest}"}


@app.get("/api/stat")
async def stat(path: str = Query(...)) -> dict[str, bool]:
    """Classify a dropped path (from a file:// URI) so the UI treats a folder as a dataset and a
    file as an attachment. Read-only, trusted-local."""
    p = Path(path).expanduser()
    return {"exists": p.exists(), "is_dir": p.is_dir()}


class _PtySession:
    """A login shell in a pseudo-terminal, cross-platform: POSIX ``pty`` or Windows ConPTY (pywinpty)."""

    def __init__(self, cwd: str) -> None:
        env = {**os.environ, "TERM": "xterm-256color"}
        self._win = None
        self._master = -1
        if os.name == "nt":
            from winpty import PtyProcess  # pywinpty, Windows-only

            shell = os.environ.get("COMSPEC") or "powershell.exe"
            self._win = PtyProcess.spawn(shell, cwd=cwd, env=env, dimensions=(24, 80))
        else:
            import pty

            self._master, slave = pty.openpty()
            shell = os.environ.get("SHELL") or "/bin/bash"
            self._proc = subprocess.Popen(
                [shell, "-i"],
                stdin=slave,
                stdout=slave,
                stderr=slave,
                preexec_fn=os.setsid,  # own process group, so disconnect reaps the whole tree
                cwd=cwd,
                env=env,
            )
            os.close(slave)

    def read(self) -> bytes:
        """Block for the next chunk of shell output (b'' at EOF)."""
        if self._win is not None:
            try:
                return self._win.read(65536).encode("utf-8", "replace")
            except EOFError:
                return b""
        try:
            return os.read(self._master, 65536)
        except OSError:
            return b""

    def write(self, text: str) -> None:
        if self._win is not None:
            self._win.write(text)
        else:
            os.write(self._master, text.encode())

    def resize(self, rows: int, cols: int) -> None:
        if self._win is not None:
            with suppress(Exception):
                self._win.setwinsize(rows, cols)
            return
        import fcntl
        import struct
        import termios

        with suppress(OSError):
            fcntl.ioctl(self._master, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))

    def close(self) -> None:
        if self._win is not None:
            with suppress(Exception):
                self._win.terminate(force=True)
            return
        with suppress(ProcessLookupError, PermissionError):
            os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
        with suppress(OSError):
            os.close(self._master)


@app.websocket("/api/terminal")
async def terminal(ws: WebSocket) -> None:
    """A real login shell rooted at the workspace, bridged over the socket. Trusted-local only: this is
    arbitrary host execution by design (like konfai-mcp), so a remote deployment must gate it — set
    KONFAI_STUDIO_TERMINAL=0 to disable."""
    # CSWSH guard: WebSockets are exempt from the same-origin policy, so a same-site sibling page could
    # open this shell via the auto-attached cookie. A browser always sends Origin on the handshake —
    # reject a cross-origin one. Non-browser clients (no Origin, e.g. a bearer-token CLI) pass.
    origin = ws.headers.get("origin")
    if origin is not None and urlparse(origin).netloc != ws.headers.get("host"):
        await ws.close(code=1008)
        return
    await ws.accept()
    if os.environ.get("KONFAI_STUDIO_TERMINAL", "1") == "0":
        await ws.send_text("\r\nTerminal disabled (KONFAI_STUDIO_TERMINAL=0).\r\n")
        await ws.close()
        return
    session = _PtySession(cwd=str(_workspace_root()))
    loop = asyncio.get_running_loop()

    async def pump() -> None:
        try:
            while True:
                data = await loop.run_in_executor(None, session.read)
                if not data:
                    break
                await ws.send_bytes(data)
        except (OSError, RuntimeError, WebSocketDisconnect):
            pass
        finally:
            with suppress(Exception):
                await ws.close()

    reader = asyncio.create_task(pump())
    try:
        while True:
            evt = json.loads(await ws.receive_text())
            if evt.get("type") == "input":
                session.write(str(evt.get("data", "")))
            elif evt.get("type") == "resize":
                session.resize(int(evt.get("rows", 24)), int(evt.get("cols", 80)))
    except (WebSocketDisconnect, ValueError, KeyError):
        pass
    finally:
        reader.cancel()
        session.close()


@app.get("/api/browse")
async def browse(path: str = Query("")) -> dict[str, Any]:
    """List a host directory's sub-folders so the UI can pick a dataset.

    Read-only, trusted-local: like konfai-mcp's dataset tools, any host path may be listed — the
    data never moves, the user only points Studio at where it already lives.
    """
    root = (Path(path).expanduser() if path else Path.home()).resolve()
    if not root.is_dir():
        raise HTTPException(404, "not a directory")
    dirs: list[str] = []
    files: list[str] = []
    try:
        for entry in sorted(root.iterdir(), key=lambda p: p.name.lower()):
            if entry.name.startswith("."):
                continue
            (dirs if entry.is_dir() else files).append(entry.name)
    except PermissionError:
        raise HTTPException(403, "permission denied") from None
    return {
        "path": str(root),
        "parent": str(root.parent) if root.parent != root else None,
        "dirs": dirs[:500],
        "files": files[:60],
    }


def _dataset_history_file() -> Path:
    return _workspace_root() / ".konfai_studio" / "datasets.json"


def _sessions_file() -> Path:
    return _workspace_root() / ".konfai_studio" / "sessions.json"


def _delete_workspace(name: str) -> None:
    """Delete a task's konfai-mcp workspace, jailed under ``sessions/`` (the name is already sanitized,
    so this never escapes the workspace root)."""
    target = _jail(_workspace_root() / "sessions", name)
    if target is not None and target.is_dir():
        shutil.rmtree(target, ignore_errors=True)


def _files_history_file() -> Path:
    return _workspace_root() / ".konfai_studio" / "files.json"


def _history_load(path: Path) -> list[str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return [p for p in data if isinstance(p, str)]
    except (OSError, json.JSONDecodeError):
        return []


def _history_add(path: Path, value: str, cap: int = 20) -> list[str]:
    """Prepend a value to a recent-items history file (deduped, capped)."""
    history = [value, *(p for p in _history_load(path) if p != value)][:cap]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history), encoding="utf-8")
    return history


@app.get("/api/datasets")
async def list_datasets() -> dict[str, list[str]]:
    return {"datasets": _history_load(_dataset_history_file())}


@app.post("/api/datasets")
async def remember_dataset(req: DatasetPath) -> dict[str, list[str]]:
    return {"datasets": _history_add(_dataset_history_file(), req.path)}


@app.get("/api/files")
async def list_files() -> dict[str, list[str]]:
    return {"files": _history_load(_files_history_file())}


@app.post("/api/files")
async def remember_file(req: DatasetPath) -> dict[str, list[str]]:
    return {"files": _history_add(_files_history_file(), req.path)}


# Claude model choices for the subscription/API backends; the local backend takes any name.
_CLAUDE_MODELS = [
    {"id": "", "label": "Default"},
    {"id": "claude-opus-4-8", "label": "Opus 4.8"},
    {"id": "claude-sonnet-5", "label": "Sonnet 5"},
    {"id": "claude-haiku-4-5", "label": "Haiku 4.5"},
]


def _brain_catalog() -> list[dict[str, Any]]:
    """The pluggable LLM backends the UI can pick from — availability flag + that backend's models."""
    import importlib.util

    def installed(module: str) -> bool:
        return importlib.util.find_spec(module) is not None

    has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    base_url = os.environ.get("KONFAI_STUDIO_LLM_BASE_URL", "http://localhost:8000/v1")
    return [
        {
            "id": "claude-code",
            "label": "Claude Code",
            "detail": "your subscription",
            "available": installed("claude_agent_sdk"),
            "models": _CLAUDE_MODELS,
        },
        {
            "id": "anthropic",
            "label": "Anthropic API",
            "detail": "API key set" if has_key else "set ANTHROPIC_API_KEY",
            "available": installed("anthropic") and has_key,
            "models": _CLAUDE_MODELS,
        },
        {
            "id": "openai",
            "label": "Local model",
            "detail": base_url,
            "available": installed("openai"),
            "models": [],  # whatever the local server hosts — free text in the UI
        },
    ]


def _llm_state() -> dict[str, Any]:
    return {"current": _reg.brain(), "model": _reg.model(), "options": _brain_catalog()}


@app.get("/api/llm")
async def get_llm() -> dict[str, Any]:
    return _llm_state()


@app.post("/api/llm")
async def set_llm(req: LLMChoice) -> dict[str, Any]:
    if req.brain:
        if req.brain not in {b["id"] for b in _brain_catalog()}:
            raise HTTPException(400, "unknown LLM backend")
        _reg.set_brain(req.brain)
    if req.model is not None:
        _reg.set_model(req.model.strip())
    return _llm_state()


@app.get("/api/device")
async def get_device() -> dict[str, Any]:
    """Per-experiment compute device map (+ the default a fresh experiment starts from)."""
    return {"devices": _reg.devices(), "default": _reg.device("")}


@app.post("/api/device")
async def set_device(req: DeviceChoice) -> dict[str, Any]:
    name = _sane_session(req.session)
    _reg.set_device(name, req.device)
    return {"device": _reg.device(name), "devices": _reg.devices()}


def _system_snapshot() -> dict[str, Any]:
    """Live RAM + per-GPU VRAM via KonfAI's own helpers, so the numbers match the MCP VRAM preflight.

    KonfAI is imported lazily (it pulls torch) and every probe degrades on its own — a missing GPU
    monitor leaves VRAM null rather than failing the whole snapshot.
    """
    import konfai

    def probe(fn: Any) -> tuple[float, float] | None:
        try:
            used, total = fn()
            return round(used, 1), round(total, 1)
        except Exception:
            return None

    ram = probe(konfai.get_ram)
    try:
        indices, names = konfai.get_available_devices()
    except Exception:
        indices, names = [], []
    gpus: list[dict[str, Any]] = []
    for ordinal, index in enumerate(indices):
        vram = probe(lambda index=index: konfai.get_vram([index]))
        gpus.append(
            {
                "index": index,
                "name": names[ordinal] if ordinal < len(names) else f"GPU {index}",
                "used_gb": vram[0] if vram else None,
                "total_gb": vram[1] if vram else None,
            }
        )
    return {
        "gpus": gpus,
        "ram": {"used_gb": ram[0], "total_gb": ram[1]} if ram else None,
    }


@app.get("/api/system")
async def system() -> dict[str, Any]:
    """Live compute resources for the title bar: per-GPU VRAM and system RAM (off the event loop)."""
    return await asyncio.to_thread(_system_snapshot)


_STAT_KEYS = ("mean", "std", "min", "max", "median")


def _read_eval_metrics(session: str) -> list[dict[str, Any]]:
    """Aggregate metrics from every ``Metric_<SPLIT>.json`` a task has produced (newest first).

    Mirrors konfai's evaluator JSON: ``{case, aggregates:{metric:{mean,std,…}}, directions}``. Keeps
    only top-level metrics (drops the per-component ``a:b:Metric:comp`` rows), like ``get_run_metrics``.
    """
    root = _workspace_root() / "sessions" / session
    if not root.is_dir():
        return []
    runs: list[dict[str, Any]] = []
    for path in sorted(root.rglob("Metric_*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        aggregates = payload.get("aggregates")
        if not isinstance(aggregates, dict):
            continue
        directions = payload.get("directions") if isinstance(payload.get("directions"), dict) else {}
        metrics: list[dict[str, Any]] = []
        for name, stats in aggregates.items():
            if name.rsplit(":", 1)[0] in aggregates:  # per-component row, folded into its parent
                continue
            if isinstance(stats, dict) and isinstance(stats.get("mean"), (int, float)):
                row: dict[str, Any] = {"name": name, "direction": directions.get(name, "")}
                row.update({k: stats[k] for k in _STAT_KEYS if isinstance(stats.get(k), (int, float))})
                metrics.append(row)
        if not metrics:
            continue
        # Per-case scores: transpose {metric: {case: value}} → [{case, values:{metric: value}}] so the UI
        # can navigate every case's numbers alongside the aggregate.
        cases = payload.get("case", {})
        metric_names = [row["name"] for row in metrics]
        per_case: dict[str, dict[str, float]] = {}
        for name in metric_names:
            for case_name, value in (cases.get(name) or {}).items():
                if isinstance(value, (int, float)):
                    per_case.setdefault(case_name, {})[name] = float(value)
        case_rows = [{"case": case_name, "values": values} for case_name, values in per_case.items()]
        split = path.stem.split("_", 1)[1] if "_" in path.stem else path.stem
        runs.append(
            {
                "run": path.parent.name,
                "split": split,
                "metrics": metrics,
                "cases": len(case_rows),
                "case_metrics": metric_names,
                "case_rows": case_rows,
            }
        )
    return runs[:20]


@app.get("/api/evaluations")
async def evaluations(session: str = Query("default")) -> dict[str, list[dict[str, Any]]]:
    return {"runs": _read_eval_metrics(_sane_session(session))}


@app.get("/api/leaderboard")
async def leaderboard(session: str = Query("default"), split: str = Query("TRAIN")) -> dict[str, Any]:
    """Rank the experiment's runs by their evaluation metrics — proxies konfai-mcp's ``leaderboard`` (which
    reads the Metric_<SPLIT>.json files live; nothing extra is persisted). One ranking per metric."""
    ok, text, payload = await _mcp_json(_sane_session(session), "leaderboard", {"split": split})
    if not ok:
        return {"ok": False, "detail": text}
    return {"ok": True, **payload} if isinstance(payload, dict) else {"ok": True}


def _run_config_snapshot(session: str, run: str) -> Path | None:
    """The newest launch-time config snapshot for a run (its Statistics/<run>/*.yml), searched at the session
    root and in any isolated app-output subtree. Jailed: a run with a path separator is refused."""
    if not run or "/" in run or "\\" in run or ".." in run:
        return None
    base = (_workspace_root() / "sessions" / session).resolve()
    if not base.is_dir():
        return None
    snaps = [
        p
        for pattern in (f"Statistics/{run}/*.yml", f"*/Statistics/{run}/*.yml")
        for p in base.glob(pattern)
        if p.is_file() and _jail(base, str(p.relative_to(base))) is not None
    ]
    return max(snaps, key=lambda p: p.stat().st_mtime) if snaps else None


@app.get("/api/run/config_diff")
async def run_config_diff(
    session: str = Query("default"), run_a: str = Query(...), run_b: str = Query(...)
) -> dict[str, Any]:
    """A unified diff of two runs' launch-time config snapshots (what actually differs between them — model,
    losses, optimizer, augmentations, and any live interventions). Reads the on-disk snapshots directly, so it
    works for every run in the leaderboard, not only ones with a live job record."""
    name = _sane_session(session)
    snap_a = _run_config_snapshot(name, run_a)
    snap_b = _run_config_snapshot(name, run_b)
    if snap_a is None or snap_b is None:
        missing = run_a if snap_a is None else run_b
        return {"ok": False, "detail": f"no config snapshot found for '{missing}'"}
    text_a = snap_a.read_text(encoding="utf-8", errors="replace").splitlines()
    text_b = snap_b.read_text(encoding="utf-8", errors="replace").splitlines()
    diff = list(difflib.unified_diff(text_a, text_b, fromfile=run_a, tofile=run_b, lineterm=""))
    return {"ok": True, "run_a": run_a, "run_b": run_b, "identical": not diff, "diff": "\n".join(diff)}


@app.get("/api/curves")
async def curves(session: str = Query("default"), run: str = Query(...), q: str = Query("")) -> dict[str, Any]:
    """A run's full training curves (the complete downsampled TensorBoard history, not the live tail) —
    konfai-mcp's ``read_training_curves``, optionally filtered to tags containing ``q``. Powers clicking a
    live chart to expand it into its whole history."""
    args: dict[str, Any] = {"run_name": run, "max_points": 2000}
    if q:
        args["tags"] = [q]
    ok, _text, data = await _mcp_json(_sane_session(session), "read_training_curves", args)
    return {"ok": ok, "curves": data.get("curves", {}) if ok and isinstance(data, dict) else {}}


class ConfigSave(BaseModel):
    session: str
    name: str
    content: str


def _session_path(session: str, rel: str) -> Path:
    """Resolve a path inside a session's workspace — jailed, never escapes the session root."""
    target = _jail(_workspace_root() / "sessions" / _sane_session(session), rel)
    if target is None:
        raise HTTPException(400, "path escapes the session workspace")
    return target


@app.post("/api/config/save")
async def save_config(req: ConfigSave) -> dict[str, bool]:
    """Save an edited config YAML — jailed to the session workspace, existing .yml only, atomic
    write (temp + replace) so a reader never sees a truncated config."""
    target = _session_path(req.session, req.name)
    if target.suffix.lower() not in {".yml", ".yaml"}:
        raise HTTPException(415, "only YAML configs are editable")
    if not target.is_file():
        raise HTTPException(404, "config not found")
    tmp = target.with_name(target.name + ".tmp")
    try:
        tmp.write_text(req.content, encoding="utf-8")
        os.replace(tmp, target)
    except OSError as exc:
        raise HTTPException(500, f"could not write config: {exc}") from None
    return {"ok": True}


@app.get("/api/experiment/ls")
async def experiment_ls(session: str = Query("default"), path: str = Query("")) -> dict[str, Any]:
    """List one directory of the experiment workspace (lazy tree). Read-only, jailed."""
    target = _session_path(session, path)
    if not target.is_dir():
        raise HTTPException(404, "not a directory")
    dirs: list[str] = []
    files: list[dict[str, Any]] = []
    for entry in sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name.lower())):
        if entry.name.startswith("."):
            continue
        if entry.is_dir():
            dirs.append(entry.name)
        else:
            try:
                files.append({"name": entry.name, "size": entry.stat().st_size})
            except OSError:
                continue
    return {"root": str(_session_path(session, "")), "dirs": dirs[:200], "files": files[:200]}


_FILE_VIEW_CAP = 400_000  # bytes shown in the experiment file viewer


@app.get("/api/experiment/file")
async def experiment_file(session: str = Query("default"), path: str = Query(...)) -> dict[str, Any]:
    """Read one workspace file as text for the experiment viewer. A file over the cap shows its **tail**
    (the latest lines — what matters for a long training log), not a truncated head; such a file is
    read-only. Small files are returned whole and YAML stays editable."""
    target = _session_path(session, path)
    if not target.is_file():
        raise HTTPException(404, "file not found")
    try:
        size = target.stat().st_size
        with target.open(encoding="utf-8", errors="replace") as handle:
            if size > _FILE_VIEW_CAP:
                handle.seek(size - _FILE_VIEW_CAP)
                handle.readline()  # drop the partial first line the byte-seek landed in
                notice = f"… showing the last {_FILE_VIEW_CAP // 1000} KB of {size // 1000} KB …\n\n"
                return {"name": path, "content": notice + handle.read(), "editable": False, "truncated": True}
            content = handle.read()
    except OSError as exc:
        raise HTTPException(500, str(exc)) from None
    return {"name": path, "content": content, "editable": target.suffix.lower() in {".yml", ".yaml"}}


def _experiment_info(session: str) -> dict[str, Any]:
    """What an experiment contains — drives the UI's overview and greys out impossible actions."""
    root = _workspace_root() / "sessions" / session
    if not root.is_dir():
        return {"checkpoints": [], "predictions": [], "jobs": [], "bundlable": False, "exportable": False}
    # "**/Checkpoints" / "**/Predictions" so isolated app outputs (<app_output>-<hash>/…) count too.
    checkpoints = sorted(str(p.relative_to(root)) for p in root.glob("**/Checkpoints/**/*.pt"))
    predictions = sorted(
        {p.name for pred in root.glob("**/Predictions") if pred.is_dir() for p in pred.iterdir() if p.is_dir()}
    )
    jobs = [
        {"run": payload.get("run_name"), "kind": payload.get("kind"), "status": _live_status(payload)}
        for payload in sorted(_all_jobs(session), key=_job_created, reverse=True)[:10]
    ]
    return {
        "checkpoints": checkpoints[:50],
        "predictions": predictions[:50],
        "jobs": jobs,
        "bundlable": bool(checkpoints),
        "exportable": any(j.get("run") for j in jobs),
    }


@app.get("/api/experiment")
async def experiment(session: str = Query("default")) -> dict[str, Any]:
    name = _sane_session(session)
    return {**_experiment_info(name), "dataset": _reg.dataset(name)}


class AppRef(BaseModel):
    ref: str
    session: str = "apps"


def _parse_apps(data: Any) -> list[dict[str, Any]]:
    """The ``apps`` list from a parsed ``list_apps`` payload (empty when unshaped)."""
    apps = data.get("apps") if isinstance(data, dict) else None
    return apps if isinstance(apps, list) else []


@app.get("/api/apps")
async def apps(session: str = Query("apps")) -> dict[str, Any]:
    """The konfai-mcp app catalogue (shipped + registered sources) — a direct MCP call, no LLM."""
    ok, _text, data = await _mcp_json(_sane_session(session), "list_apps", {"include_summary": True})
    return {"ok": ok, "apps": _parse_apps(data)}


def _app_bundle_file(ref: str, filename: str) -> Path:
    """Local path to a file inside an app's bundle (for serving its ONNX deploy artifacts).

    Resolves the app repository and returns the file path WITHOUT importing the app's ``.py`` (only
    the bundled files are touched). Raises 404 when the app is not an ONNX-deployable bundle.
    """
    try:
        from konfai_apps.app_repository import AppRepositoryError, get_app_repository_info
    except ImportError as exc:  # pragma: no cover - konfai-apps not installed
        raise HTTPException(503, "konfai-apps is not installed") from exc
    try:
        repo = get_app_repository_info(ref, force_update=False)
        path = Path(repo._download(filename))
    except (AppRepositoryError, FileNotFoundError, OSError, KeyError, ValueError) as exc:
        raise HTTPException(404, f"app '{ref}' has no portable ONNX bundle") from exc
    if not path.is_file():
        raise HTTPException(404, f"app '{ref}' has no '{filename}' (export one with `konfai-apps bundle --onnx`)")
    return path


@app.get("/api/apps/manifest")
async def app_manifest(ref: str = Query(...)) -> dict[str, Any]:
    """The deploy manifest of an app's ONNX bundle — the contract the in-tab runtime consumes."""
    return json.loads(_app_bundle_file(ref, "manifest.json").read_text())


@app.get("/api/apps/model")
async def app_model(ref: str = Query(...)) -> FileResponse:
    """The app's ``model.onnx`` (served to the browser for 100% client-side, zero-egress inference)."""
    return FileResponse(
        _app_bundle_file(ref, "model.onnx"), media_type="application/octet-stream", filename="model.onnx"
    )


async def _apps_after(session: str, source_tool: str, ref: str) -> dict[str, Any]:
    """Register/unregister an app source, then return that outcome with the refreshed catalogue."""
    ok, text = await call_mcp_tool(session, source_tool, {"ref": ref})
    _, _listed, data = await _mcp_json(session, "list_apps", {"include_summary": True})
    return {"ok": ok, "result": text, "apps": _parse_apps(data)}


@app.post("/api/apps/register")
async def register_app(req: AppRef) -> dict[str, Any]:
    return await _apps_after(_sane_session(req.session), "register_app_source", req.ref)


@app.post("/api/apps/unregister")
async def unregister_app(req: AppRef) -> dict[str, Any]:
    return await _apps_after(_sane_session(req.session), "unregister_app_source", req.ref)


_TB_IMAGE_HISTORY = 30  # steps kept per image tag (the slider's range); frames are fetched lazily


def _tb_image_dir(session: str, base: str = "") -> Path | None:
    """The TensorBoard image dir for a run, from its session-relative ``base`` ("<base>/tb", sibling of the
    run's log_0.txt). ``base`` is "Statistics/<run>" (session-root) or "<app_output>-<hash>/Statistics/<run>"
    (isolated app run), so an isolated run's images resolve under its own output subtree. Jailed: ``base``
    may contain '/', never '..'. Without a base, the most recently written tb dir anywhere under the session."""
    root = (_workspace_root() / "sessions" / session).resolve()
    if base:  # a provided base is authoritative — a traversal attempt is rejected, never fallen back on
        if ".." in Path(base).parts:
            return None
        one = _jail(root, f"{base}/tb")
        return one if one is not None and one.is_dir() else None
    tb_dirs = sorted(root.glob("**/tb"), key=lambda p: p.stat().st_mtime, reverse=True) if root.is_dir() else []
    return tb_dirs[0] if tb_dirs else None


def _tb_accumulator(tb_dir: Path, history: int) -> Any | None:
    try:
        from tensorboard.backend.event_processing.event_accumulator import IMAGES, EventAccumulator
    except Exception:
        return None
    try:
        acc = EventAccumulator(str(tb_dir), size_guidance={IMAGES: history})
        acc.Reload()
        return acc
    except Exception:
        return None


def _tb_previews(session: str, base: str = "", limit: int = 24) -> list[dict[str, Any]]:
    """A manifest of a run's TensorBoard image tags: [{label, steps:[…], step}] — the step history per
    output (Training/CT, Validation/MR, …), with NO image bytes (those are fetched per step, lazily, so a
    long history never bloats the payload). Empty if that run has no images yet."""
    tb_dir = _tb_image_dir(session, base)
    if tb_dir is None:
        return []
    acc = _tb_accumulator(tb_dir, _TB_IMAGE_HISTORY)
    if acc is None:
        return []
    out: list[dict[str, Any]] = []
    for tag in acc.Tags().get("images", [])[:limit]:
        steps = [image.step for image in acc.Images(tag)]
        if steps:
            out.append({"label": tag, "steps": steps, "step": steps[-1]})
    return out


def _tb_image_bytes(session: str, base: str, tag: str, step: int) -> bytes | None:
    """The encoded PNG for one (base, tag, step) TensorBoard image, or the latest for that tag if the step is
    gone from the kept window."""
    tb_dir = _tb_image_dir(session, base)
    if tb_dir is None:
        return None
    acc = _tb_accumulator(tb_dir, _TB_IMAGE_HISTORY)
    if acc is None:
        return None
    try:
        images = acc.Images(tag)
    except KeyError:
        return None
    if not images:
        return None
    for image in images:
        if image.step == step:
            return bytes(image.encoded_image_string)
    return bytes(images[-1].encoded_image_string)


@app.get("/api/previews")
async def previews(session: str = Query("default"), base: str = Query("")) -> dict[str, list[dict[str, Any]]]:
    return {"previews": await asyncio.to_thread(_tb_previews, _sane_session(session), base)}


@app.get("/api/preview_image")
async def preview_image(
    session: str = Query("default"), base: str = Query(""), tag: str = Query(...), step: int = Query(...)
) -> Response:
    """One TensorBoard image montage (PNG) for a (base, tag, step) — the lazy per-frame fetch behind the slider."""
    data = await asyncio.to_thread(_tb_image_bytes, _sane_session(session), base, tag, step)
    if data is None:
        raise HTTPException(404, "image not found")
    return Response(content=data, media_type="image/png")


_TB_SERVERS: dict[str, dict[str, Any]] = {}  # session -> {proc, url}: one lazily-started TensorBoard per task


def _free_port() -> int:
    import socket

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


@app.get("/api/tensorboard")
async def tensorboard_link(session: str = Query("default")) -> dict[str, Any]:
    """Ensure a TensorBoard server is running for the task's Statistics dir and return its URL — the full
    TensorBoard UI (scalars, images, histograms) for every run, alongside Studio's own live feed. One
    server per task, reused while alive, bound to 127.0.0.1 (a remote deployment must proxy the port)."""
    name = _sane_session(session)
    live = _TB_SERVERS.get(name)
    if live and live["proc"].poll() is None:
        return {"ok": True, "url": live["url"]}
    # The session root, not Statistics/: TensorBoard recurses, so it surfaces every "*/tb" at any depth —
    # session-root runs AND isolated app-output runs (<app_output>-<hash>/Statistics/<run>/tb) alike.
    session_root = _workspace_root() / "sessions" / name
    if not session_root.is_dir() or not any(session_root.glob("**/tb")):
        return {"ok": False, "detail": "no TensorBoard events yet — run a training first"}
    # Look next to the running interpreter first: the server is launched as a console script, so the env's
    # bin dir may not be on PATH and shutil.which alone would miss it.
    sibling = Path(sys.executable).with_name("tensorboard")
    binary = str(sibling) if sibling.exists() else shutil.which("tensorboard")
    if not binary:
        return {"ok": False, "detail": "tensorboard is not installed (pip install konfai[tensorboard])"}
    port = _free_port()
    proc = subprocess.Popen(
        [binary, "--logdir", str(session_root), "--host", "127.0.0.1", "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{port}/"
    _TB_SERVERS[name] = {"proc": proc, "url": url}
    return {"ok": True, "url": url}


_TERMINAL_STATUS = {"done", "error", "killed", "cancelled"}


def _workspace_root() -> Path:
    return Path(os.environ.get("KONFAI_MCP_WORKSPACES_ROOT") or Path.home() / "KonfAI_Workspaces")


def _job_created(job: dict[str, Any]) -> float:
    """A job's creation time as an epoch float, from its recorded ``created_at`` (a float epoch or an ISO
    string, depending on which konfai-mcp wrote it). Used to order jobs by when they *started*, not by
    file mtime — a terminal job's json can be rewritten later (status monitor), which mtime would misread
    as 'newest' and make the feed follow a dead run over a fresh one."""
    created = job.get("created_at")
    if isinstance(created, (int, float)):
        return float(created)
    if isinstance(created, str):
        with suppress(ValueError):
            return datetime.fromisoformat(created).timestamp()
    return 0.0


def _latest_job(session: str) -> dict[str, Any] | None:
    """A task's most recently created job (its ``job.json`` payload), newest ``created_at`` wins."""
    return max(_all_jobs(session), key=_job_created, default=None)


def _pid_alive(pid: Any) -> bool:
    try:
        value = int(pid)
    except (TypeError, ValueError):
        return False
    if value <= 0:
        return False
    # A crashed job whose parent-reaper died leaves a zombie: it has exited but still owns a pid, so both
    # os.kill(0) and psutil see it as "alive". On Linux read the state directly and treat 'Z' as dead.
    try:
        stat = Path(f"/proc/{value}/stat").read_text(encoding="utf-8")
        return stat.rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, IndexError):
        pass
    try:
        os.kill(value, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _live_status(job: dict[str, Any]) -> str:
    """The job's status, but a still-'running' record whose process is gone reads as 'failed' — the MCP
    monitor that would flip it may have died (e.g. a server restart), leaving the status stale."""
    status = str(job.get("status") or "")
    if status in {"running", "waiting", "queued"} and not _pid_alive(job.get("pid")):
        return "error"  # a terminal status the UI renders red; the log tail shows the traceback
    return status


@app.post("/api/job/cancel")
async def cancel_running_job(req: CancelJob) -> dict[str, Any]:
    """Stop a task's most recent job. ``cancel_job`` reaps the whole process group."""
    name = _sane_session(req.session)
    job = _latest_job(name)
    job_id = job.get("job_id") if job else None
    if not job_id:
        return {"ok": False, "detail": "no job to stop"}
    return await _mcp_detail(name, "cancel_job", {"job_id": job_id, "wait_s": 5})


@app.post("/api/run/delete")
async def delete_run(req: DeleteRun) -> dict[str, Any]:
    """Delete one run's outputs — proxies konfai-mcp's jailed ``delete_run`` (never leaves the workspace)."""
    return await _mcp_detail(_sane_session(req.session), "delete_run", {"run_name": req.run_name, "kind": req.kind})


def _require_train_job(name: str) -> tuple[str | None, dict[str, Any] | None]:
    """The active training job's id, or an error payload when there is no running training job."""
    job = _latest_job(name)
    job_id = job.get("job_id") if job else None
    if not job_id or (job or {}).get("kind") != "train":
        return None, {"ok": False, "detail": "no running training job"}
    return job_id, None


@app.post("/api/job/validate")
async def request_validation(req: CancelJob) -> dict[str, Any]:
    """Ask a running training job to run a validation pass now — ``request_validation`` signals the job
    (SIGUSR1); the trainer validates at its next iteration boundary and the metrics stream into Live."""
    name = _sane_session(req.session)
    job_id, error = _require_train_job(name)
    if error is not None:
        return error
    return await _mcp_detail(name, "request_validation", {"job_id": job_id})


@app.post("/api/job/tunables")
async def set_tunables(req: SetTunables) -> dict[str, Any]:
    """Change a running training's lr / it_validation mid-run — proxies konfai-mcp's ``set_live_tunables``,
    which drops a jailed control file the trainer applies at its next poll boundary."""
    name = _sane_session(req.session)
    job_id, error = _require_train_job(name)
    if error is not None:
        return error
    args: dict[str, Any] = {"job_id": job_id}
    if req.lr is not None:
        args["lr"] = req.lr
    if req.it_validation is not None:
        args["it_validation"] = req.it_validation
    return await _mcp_detail(name, "set_live_tunables", args)


_LOG_BACKFILL = 32_000  # bytes: on connect, replay only the recent tail of a large log, not its full history


def _tail_start(path: Path) -> int:
    """Byte offset to begin following a log at: near the end of an already-large file (so a mid-run
    (re)connect replays only its recent tail), **aligned to the next line start** so the first emitted
    line is never a mid-line fragment."""
    if not path.is_file():
        return 0
    size = path.stat().st_size
    if size <= _LOG_BACKFILL:
        return 0
    with path.open(encoding="utf-8", errors="replace") as handle:
        handle.seek(size - _LOG_BACKFILL)
        handle.readline()  # discard the partial line the byte-seek landed inside
        return handle.tell()


def _tail_lines(path: Path, pos: int, buf: str) -> tuple[list[str], int, str]:
    """Complete new lines appended to ``path`` since byte ``pos``. ``buf`` carries an incomplete trailing
    line between reads so a read landing mid-line never yields a fragment. Returns (lines, new pos, new
    buf). A missing file yields nothing; a truncated/rotated file is clamped."""
    if not path.is_file():
        return [], pos, buf
    with path.open(encoding="utf-8", errors="replace") as handle:
        handle.seek(min(pos, path.stat().st_size))
        buf += handle.read()
        pos = handle.tell()
    parts = buf.split("\n")
    return parts[:-1], pos, parts[-1]  # last element is the (possibly empty) incomplete remainder


_HOST_KEYS = ("memory_gb", "memory_percent", "memory_gpu_gb", "memory_gpu_percent", "cpu_percent")


def _phase_stage(label: str) -> str:
    """The stage a tqdm phase belongs to, from its label — so the client never guesses. 'Caching Train' →
    caching, 'Training' → training, 'Metric VALIDATION' → evaluation, 'Prediction' → prediction."""
    head = label.split(maxsplit=1)[0].lower() if label else ""
    return {"metric": "evaluation"}.get(head, head or "caching")


_RUN_ROOT_KIND = {
    "Statistics": "train",
    "Predictions": "prediction",
    "Evaluations": "evaluation",
    "Uncertainties": "uncertainty",
}


def _discover_run_logs(job: dict[str, Any]) -> list[tuple[Path, str, str]]:
    """Every konfai runtime log a job writes, as (log_path, run_name, kind). A workflow job names one
    explicitly (runtime_log_path). An app job writes runtime_log_path=None but produces its logs under
    output_path: infer → one, fine-tune → one per finetuned checkpoint (Statistics/<run>), pipeline →
    predict + evaluate + uncertainty. Each discovered log becomes its own per-run feed (MR2CT_01, 02, …)."""
    runtime = job.get("runtime_log_path")
    if runtime:
        return [(Path(runtime), str(job.get("run_name") or job.get("kind") or "run"), str(job.get("kind") or ""))]
    output = job.get("output_path")
    if not output:
        return []
    base = Path(output)
    job_kind = str(job.get("kind") or "")
    found: list[tuple[Path, str, str]] = []
    seen: set[Path] = set()
    for pattern in (
        "*/log_0.txt",
        "Statistics/*/log_0.txt",
        "Predictions/*/log_0.txt",
        "Evaluations/*/log_0.txt",
        "Uncertainties/*/log_0.txt",
    ):
        for log in sorted(base.glob(pattern)):
            if log in seen:
                continue
            seen.add(log)
            root = log.parent.parent.name if log.parent.parent != base else ""
            found.append((log, log.parent.name, _RUN_ROOT_KIND.get(root, job_kind)))
    return found


def _runtime_events(line: str, run: str, kind: str, step: int) -> tuple[list[dict[str, Any]], int]:
    """The structured SSE events for one runtime-log line, tagged with its run + kind: a ``metric``
    (per-model values + learning rate + host memory/CPU), or a ``progress`` for a metric-less phase
    (caching, plain inference). Returns (events, new step). The step is a per-run counter."""
    entry = parse_live_metric_line(line)
    if entry is not None:
        step += 1
        values = dict(entry["flat_metrics"])
        if entry["stage"] == "Training":
            # learning rate is a training-only signal — never chart it under validation
            values.update({f"{m['name']}:lr": m["lr"] for m in entry.get("models", []) if m.get("lr")})
        host = {k: entry[k] for k in _HOST_KEYS if k in entry}
        return [
            {
                "type": "metric",
                "run": run,
                "kind": kind,
                "step": step,
                "stage": entry["stage"].lower(),
                # Training/Validation/Prediction, or the eval split ("Metric TRAIN") carried on `label`.
                "label": entry.get("label") or entry["stage"],
                "values": values,
                "progress": entry.get("progress"),
                **host,
            }
        ], step
    progress = parse_live_progress(line)
    if progress is not None:
        label = progress_label(line)
        return [
            {
                "type": "progress",
                "run": run,
                "kind": kind,
                "stage": _phase_stage(label),
                "label": label,
                "progress": progress,
                **parse_host_stats(line),
            }
        ], step
    return [], step


def _all_jobs(session: str) -> list[dict[str, Any]]:
    jobs_dir = _workspace_root() / "sessions" / session / ".konfai_mcp" / "jobs"
    out: list[dict[str, Any]] = []
    if jobs_dir.is_dir():
        for record in jobs_dir.glob("*/job.json"):
            try:
                out.append(json.loads(record.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
    return out


_MTIME_LIVE_WINDOW = 8.0  # a run log written this recently reads as live when no job record claims it


def _discover_session_runs(session: str) -> list[tuple[Path, str, str, str, str]]:
    """Every run of the experiment as (log_path, run_name, kind, status, base) — one per runtime log.

    Workflow jobs write under the session roots (Statistics/Predictions/Evaluations); app jobs (infer /
    fine-tune / pipeline / uncertainty) write under an isolated ``output_path`` subtree, so EVERY app job's
    output_path is scanned (not only the newest), else an earlier isolated run vanishes the moment a newer
    job exists. ``base`` is the run dir relative to the session root — "Statistics/<run>" for a session-root
    run, "<app_output>-<hash>/Statistics/<run>" for an isolated app run — the one datum that lets the
    previews / TensorBoard / Browse helpers resolve the real on-disk location. Status comes from the newest
    job that names the log (workflow) or owns the output_path (app), else from how recently the log was
    written. Newest first."""
    base_root = _workspace_root() / "sessions" / session
    # A run dir is often written by several jobs (re-runs share one log_0.txt); its status must come from the
    # NEWEST job that names it (workflow) or owns its output subtree (app), so a fresh run wins over an old one.
    jobs = _all_jobs(session)
    status_by_log: dict[str, str] = {}
    newest_by_log: dict[str, float] = {}
    app_status_by_output: dict[str, tuple[float, str]] = {}
    for job in jobs:
        created = _job_created(job)
        runtime = job.get("runtime_log_path")
        if runtime:
            key = str(Path(runtime))
            if created >= newest_by_log.get(key, float("-inf")):
                newest_by_log[key] = created
                status_by_log[key] = _live_status(job)
        elif job.get("output_path"):
            out = str(Path(job["output_path"]))
            if created >= app_status_by_output.get(out, (float("-inf"), ""))[0]:
                app_status_by_output[out] = (created, _live_status(job))
    found: list[tuple[Path, str, str, str, str, float]] = []
    seen: set[Path] = set()

    def add(log: Path, run_name: str, kind: str, app_status: str | None) -> None:
        if log in seen or not log.is_file():
            return
        seen.add(log)
        status = status_by_log.get(str(log)) or app_status
        if status is None:
            status = "running" if (time.time() - log.stat().st_mtime) < _MTIME_LIVE_WINDOW else "done"
        try:
            base = str(log.parent.relative_to(base_root))
        except ValueError:
            base = log.parent.name
        found.append((log, run_name, kind, status, base, log.stat().st_mtime))

    for root, kind in (("Statistics", "train"), ("Predictions", "prediction"), ("Evaluations", "evaluation")):
        directory = base_root / root
        if directory.is_dir():
            for log in directory.glob("*/log_0.txt"):
                add(log, log.parent.name, kind, None)
    for job in jobs:  # app runs land outside the three roots, under each app job's own output_path
        out = job.get("output_path")
        if not out:
            continue
        entry = app_status_by_output.get(str(Path(out)))
        for log, run_name, kind in _discover_run_logs(job):
            add(log, run_name, kind, entry[1] if entry else None)
    found.sort(key=lambda row: row[5], reverse=True)
    return [(log, run_name, kind, status, base) for log, run_name, kind, status, base, _ in found]


@app.get("/api/live")
async def live(session: str = Query("default")) -> StreamingResponse:
    """Tail a task's most recent job in real time (SSE): the console log as raw text, and konfai's runtime
    log as **structured** metrics + progress — parsed by konfai-mcp's own ``live_parse`` (one source of
    truth, no re-implementation here).

    Two logs, two roles. The console wrapper log (header, warm-up prints, crash tracebacks) streams as
    ``log`` lines. konfai writes its per-iteration training tqdm to the runtime file, not stdout, so that
    file streams as ``metric`` events (a stage + flat metric values + the ``progress`` bar) and, for the
    metric-less data-caching phase, ``progress`` events — never as raw log lines, which would bury the
    console tail under thousands of tqdm frames.

    The connection is **persistent**: a job finishing is announced once (terminal status) but the stream
    keeps watching, so the next job the user launches is picked up on the same connection.
    """
    name = _sane_session(session)

    async def gen() -> AsyncIterator[str]:
        console_key: str | None = None  # which job's console log is being followed
        cpath: Path | None = None
        cpos = 0
        cbuf = ""
        feeds: dict[str, dict[str, Any]] = {}  # log-path -> {run, kind, path, pos, buf, step}: one per run, kept
        announced: dict[str, str] = {}  # run key -> terminal status already emitted (so it fires once)
        idle_sent = False

        while True:
            latest = _latest_job(name)
            runs = _discover_session_runs(name)
            if not runs and latest is None:
                if not idle_sent:
                    yield _sse({"type": "idle"})
                    idle_sent = True
                await asyncio.sleep(0.6)
                continue
            idle_sent = False

            # Console (raw text, tracebacks) follows the latest job only; a new job resets the tail. The
            # `job` event names the active run so the client can default its tab to it — it never wipes.
            if latest is not None and latest.get("log_path") and latest["log_path"] != console_key:
                console_key = latest["log_path"]
                cpath = Path(console_key)
                cpos = _tail_start(cpath)
                cbuf = ""
                yield _sse(
                    {
                        "type": "job",
                        "run": latest.get("run_name") or latest.get("kind") or "job",
                        "kind": latest.get("kind") or "",
                        "status": _live_status(latest),
                    }
                )
            if cpath is not None:
                lines, cpos, cbuf = _tail_lines(cpath, cpos, cbuf)
                for line in lines:
                    stripped = line.lstrip()
                    if not stripped or stripped[0] == "#" or stripped.startswith("[konfai-mcp]"):
                        continue
                    # [KonfAI] lines are framework chatter. Surface only genuine OOM/memory-pressure notices
                    # (the auto-patch recovery prints those on retry; hiding them makes a churning run look
                    # silent) -- not the routine memory-budget / cache-plan line, which is a benign startup
                    # decision, not an alert.
                    if stripped.startswith("[KonfAI]") and not any(
                        k in stripped.lower() for k in ("out of memory", "oom", "re-plan", "replan", "retry", "reduc")
                    ):
                        continue
                    yield _sse({"type": "log", "line": line})

            # Every run of the experiment is followed as its own feed and kept — launching a prediction
            # adds a run, it never clears the training runs. A newly-seen log replays from 0 so its curves
            # rebuild on connect.
            for log, run_name, run_kind, status, base in runs:
                feed_key = str(log)
                if feed_key not in feeds:
                    feeds[feed_key] = {"run": run_name, "kind": run_kind, "path": log, "pos": 0, "buf": "", "step": 0}
                    yield _sse({"type": "run", "run": run_name, "kind": run_kind, "status": status, "base": base})
                feed = feeds[feed_key]
                lines, feed["pos"], feed["buf"] = _tail_lines(feed["path"], feed["pos"], feed["buf"])
                for line in lines:
                    events, feed["step"] = _runtime_events(line, feed["run"], feed["kind"], feed["step"])
                    for event in events:
                        yield _sse(event)
                run_key = f"{run_kind}:{run_name}"
                if status in _TERMINAL_STATUS and announced.get(run_key) != status:
                    announced[run_key] = status
                    yield _sse({"type": "status", "run": run_name, "kind": run_kind, "status": status})
            await asyncio.sleep(0.6)

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/assets/{file_path:path}")
async def assets(file_path: str) -> FileResponse:
    """Serve the built Vite assets (JS/CSS) from ``web/assets`` — jailed to that dir."""
    target = _jail(WEB_DIR / "assets", file_path)
    if target is None or not target.is_file():
        raise HTTPException(404, "asset not found")
    return FileResponse(str(target))


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/konfai-logo.png")
async def logo() -> FileResponse:
    return FileResponse(WEB_DIR / "konfai-logo.png")


# Volume image formats NiiVue reads client-side (NIfTI / MHA and their compressed forms).
_VOLUME_SUFFIXES = {".nii", ".nii.gz", ".mha", ".mhd", ".nrrd", ".gz"}


@app.get("/files/volume")
async def volume(path: str = Query(..., description="Absolute host path of the volume to stream")) -> FileResponse:
    """Stream a medical volume to the browser (NiiVue) with HTTP range support.

    Trusted-local deployment only: like konfai-mcp's dataset tools, volumes may live anywhere
    on the host and are served **read-only** — this never exposes a write path. Starlette's
    FileResponse honours the ``Range`` header, so NiiVue can fetch a large volume in chunks.
    """
    p = Path(path).expanduser()
    if not p.is_file():
        raise HTTPException(404, "volume not found")
    if p.suffix.lower() not in _VOLUME_SUFFIXES:
        raise HTTPException(415, f"unsupported volume type: {p.suffix}")
    return FileResponse(str(p), media_type="application/octet-stream", filename=p.name)
