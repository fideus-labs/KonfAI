# SPDX-License-Identifier: Apache-2.0
"""FastAPI BFF: one agent per task (konfai-mcp session), streamed to the browser.

Localhost, co-located with konfai-mcp on the compute node. Each task is an isolated konfai-mcp
session with its own agent, chat, jobs and workspace; tasks run concurrently. The BFF streams
the chat (`/api/chat`) and a job's live log (`/api/live`) over SSE, and serves the built front.

App creation + route wiring: the focused sibling modules (`paths`, `registry`, `auth`, `terminal`,
`tensorboard`, `jobs`) own the logic; this module composes them into the app.
"""

from __future__ import annotations

import asyncio
import difflib
import json
import os
import re
import shutil
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, StreamingResponse
from konfai_mcp.metrics_service import metric_direction, metric_run_name, top_level_metrics
from pydantic import BaseModel

from .agent import call_mcp_tool, suggest_title
from .auth import _AuthGate
from .auth import router as auth_router
from .jobs import _all_jobs, _job_created, _latest_job, _runtime_log, _sse, _status_for
from .jobs import router as jobs_router
from .paths import (
    _credentials_file,
    _dataset_history_file,
    _files_history_file,
    _history_add,
    _history_load,
    _jail,
    _sane_session,
    _session_dir,
    _session_path,
    _studio_session_dir,
)
from .registry import _Registry
from .tensorboard import reap_tb_servers
from .tensorboard import router as tensorboard_router
from .terminal import router as terminal_router
from .transcript import load_transcript, note_event, record_turn
from .workflow import MAX_MOVES, moves, pre_prompt, state_for, state_line

WEB_DIR = Path(__file__).parent / "web"

_reg = _Registry()


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    _load_credentials()  # a brain configured from the UI stays configured across restarts
    _reg.load()  # restore tasks + titles from a previous run; agents spin up lazily per task
    try:
        yield
    finally:
        await _reg.close()
        reap_tb_servers()


app = FastAPI(title="KonfAI Studio", lifespan=_lifespan)
app.add_middleware(_AuthGate)  # no-op unless KONFAI_STUDIO_TOKEN is set (trusted-local by default)
app.include_router(auth_router)
app.include_router(terminal_router)
app.include_router(tensorboard_router)
app.include_router(jobs_router)


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
    anthropic_api_key: str | None = None  # for the Anthropic API brain
    base_url: str | None = None  # for any OpenAI-compatible server (vLLM / Ollama / LM Studio)
    local_api_key: str | None = None


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
    """Proxy a konfai-mcp tool: (ok, raw text, JSON-decoded value).

    Unparseable text is a failure, not an empty answer: returning ``{}`` for it made a leaderboard the
    caller could not decode render as "no evaluations yet" over runs that had been scored.
    """
    ok, text = await call_mcp_tool(session, tool, args)
    try:
        return ok, text, json.loads(text)
    except (TypeError, ValueError):
        return False, text, {}


def _state(session: str) -> dict[str, Any]:
    """Where this experiment stands: derived by konfai-mcp from the workspace, not remembered here.
    Studio shares the filesystem with the MCP server, so this is the same answer its tools report."""
    jobs = sorted(_all_jobs(session), key=_job_created, reverse=True)
    for job in jobs:  # the reading the live feed makes: a dead process is not running, a written log is
        job["status"] = _status_for(job, _runtime_log(job))
    return state_for(_session_dir(session), jobs, _reg.dataset(session))


def _turn_moves(
    session: str, written: list[dict[str, str]], state: dict[str, Any], tool_actions: list[str] | None
) -> list[dict[str, str]]:
    """This turn's action buttons.

    The ones the assistant wrote at the end of its own reply, alone. They are its answer: when the reply
    asked a question they ARE its options, and a generic tool-named button beside them is both a wrong
    answer to that question and a step the reply just argued against: "Run train" under a config that
    failed validation launches it anyway.

    A turn cut short writes none, and the buttons from the last turn that did are a better answer than
    the generic fill-in as long as they still describe where the experiment stands; the state line is what
    decides that. Only when neither applies do the derived moves carry the bar, so it is never empty.
    """
    signature = state_line(state)
    own = written or _reg.recall_moves(session, signature)
    if written:
        _reg.remember_moves(session, written, signature)
    return (own or moves(state, tool_actions))[:MAX_MOVES]


@app.post("/api/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    """Stream one user turn as Server-Sent Events. Serialised: one turn at a time."""

    name = _sane_session(req.session)
    message = pre_prompt(_state(name), _device_directive(_reg.device(name)), req.message)

    async def gen() -> AsyncIterator[str]:
        written: list[dict[str, str]] = []  # the moves the assistant wrote at the end of its reply
        tool_actions: list[str] | None = None  # None = the turn ran no tool at all
        broke = False
        parts: list[dict[str, Any]] = []  # what this turn streamed, kept for browsers that missed it
        try:
            # The lock covers the agent only. Holding it over the title call would make the user's next
            # message queue behind work they are not waiting for.
            async with _reg.lock(name):
                try:
                    agent = await _reg.agent(name)
                    async for event in agent.send(message):
                        if event.get("type") == "tool_call":
                            tool_actions = tool_actions or []
                        elif event.get("type") == "next_actions" and isinstance(event.get("actions"), list):
                            tool_actions = event["actions"]  # the newest tool's own advice supersedes
                        elif event.get("type") == "next_prompts" and isinstance(event.get("prompts"), list):
                            written = event["prompts"]  # the assistant's own, parsed out of its reply
                            continue  # held back until the state is known, so the bar fills once
                        yield _sse(event)
                        note_event(parts, event)
                except Exception as exc:
                    _reg.invalidate(name)  # drop the broken SDK client so the next turn rebuilds and resumes
                    broke = True
                    yield _sse({"type": "error", "message": str(exc)})
                    note_event(parts, {"type": "error", "message": str(exc)})
        finally:
            # Recorded even when the browser walks away mid-stream (interrupt, closed tab): the turn
            # happened, and the next browser to open this experiment deserves to see it.
            record_turn(_studio_session_dir(name), req.message, parts)
        # Re-read the workspace: the tools have written to it, so this is what the turn actually achieved
        # (not what it said it did. Even a turn that broke ends with a state and a move) including when
        # deriving the state itself raises, or the bar would freeze on the previous turn.
        try:
            state = _state(name)
            yield _sse({"type": "state", **state})
            yield _sse({"type": "next_prompts", "prompts": _turn_moves(name, written, state, tool_actions)})
        except Exception as exc:
            yield _sse({"type": "error", "message": f"state unavailable: {exc}"})
            yield _sse({"type": "next_prompts", "prompts": written})
            return
        # Machine-injected onboarding prompts (dataset inspection) don't describe the experiment: # wait for
        # the user's own first message so the title reflects the real task.
        if (
            not broke
            and _reg.is_untitled(name)
            and req.message.strip()
            and not req.message.startswith("Inspect the dataset at ")
        ):
            try:  # let the LLM name the experiment from its first prompt
                title = await suggest_title(req.message, _reg.brain())
                _reg.set_title(name, title)
                yield _sse({"type": "title", "session": name, "title": title})
            except Exception:
                pass

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/api/chat/history")
async def chat_history(session: str) -> dict[str, Any]:
    """The server-side transcript. The browser keeps its own in localStorage, per device, one that did
    not run these turns (another machine, a cleared profile) adopts this copy on open."""
    return {"messages": load_transcript(_studio_session_dir(session))}


@app.post("/api/chat/interrupt")
async def interrupt_chat(req: CancelJob) -> dict[str, bool]:
    """Cut the running turn short, so a correction typed mid-turn is acted on now rather than queued
    behind everything the assistant was still going to do. A tool already in flight still finishes: a
    launched job cannot be un-launched. ``ok`` is False when there is nothing to interrupt."""
    return {"ok": await _reg.interrupt(_sane_session(req.session))}


@lru_cache(maxsize=1)
def _tool_count() -> int:
    """How many tools konfai-mcp registers, or 0 if it cannot be asked.

    Read from the registry rather than written down (the status bar carried a literal six tools
    behind), but never at the cost of the health check: the front paints the whole bar 'offline' on
    any failure of it, and the tool count is a caption."""
    try:
        from konfai_mcp.server import mcp

        return len(asyncio.run(mcp.list_tools()))
    except Exception:
        return 0


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {"status": "ok", "agent": "ready", "tools": await asyncio.to_thread(_tool_count)}


@app.get("/api/sessions")
async def list_sessions() -> dict[str, Any]:
    return {"sessions": _reg.names(), "titles": _reg.titles(), "datasets": _reg.datasets()}


@app.get("/api/sessions/status")
async def sessions_status() -> dict[str, Any]:
    """Latest job status per experiment, so the rail can colour each dot by its state. Read against the
    run's log like the live feed does, or a dot would contradict the panel it sits next to."""
    statuses: dict[str, str] = {}
    # Every session is listed, jobless ones with an empty status: the rail's poll is also how a browser
    # discovers an experiment created elsewhere, and a discovery gated on "has run a job" would hide a
    # freshly created one exactly while the other user is still configuring it.
    for name in _reg.names():
        job = _latest_job(name)
        statuses[name] = _status_for(job, _runtime_log(job)) if job and job.get("status") else ""
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
    """Package an experiment as a reusable app into a chosen folder: a direct MCP call, no LLM.

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
            "description": f"{title}: packaged from KonfAI Studio.",
            "output": req.output,
        },
    )
    # A fresh bundle is only useful if you can find it: register it into the KonfAI Apps catalogue.
    registered = False
    if ok:
        bundle = str(Path(req.output).expanduser() / slug)
        registered, _ = await call_mcp_tool("apps", "register_app_source", {"ref": bundle})
    return {"ok": ok, "result": text, "registered": registered}


@app.post("/api/sessions/export")
async def export_session(req: ExportRequest) -> dict[str, Any]:
    """Export an experiment as a self-contained copy of its workspace (config, code, checkpoints,
    statistics and metrics) minus the input Dataset (the user's data, which lives outside) and the
    Studio/MCP internals. A folder you can archive or share."""
    name = _sane_session(req.session)
    src = _session_dir(name).resolve()
    if not src.is_dir():
        return {"ok": False, "result": "This experiment has no workspace yet."}
    dest = Path(req.output).expanduser() / _sane_session(_reg.title(name) or name)
    # The bundle writes to a folder of the same name under a chosen parent, so pointing both at one place
    # merges an export INTO an app bundle: it keeps working as neither. Re-exporting over an older export
    # of the same experiment stays allowed, which is the one merge a user means.
    if (dest / "app.json").is_file():
        return {"ok": False, "result": f"{dest} already holds an app bundle. Export somewhere else."}
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


@app.get("/api/browse")
async def browse(path: str = Query("")) -> dict[str, Any]:
    """List a host directory's sub-folders so the UI can pick a dataset.

    Read-only, trusted-local: like konfai-mcp's dataset tools, any host path may be listed: the
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


#: What the LLM panel can set, and the variable each brain already reads. Saved credentials are put
#: back into the environment on start, so a brain is configured once and survives a restart.
_CREDENTIALS = {
    "anthropic_api_key": "ANTHROPIC_API_KEY",
    "base_url": "KONFAI_STUDIO_LLM_BASE_URL",
    "local_api_key": "KONFAI_STUDIO_LLM_API_KEY",
}


def _load_credentials() -> None:
    try:
        saved = json.loads(_credentials_file().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    for field, variable in _CREDENTIALS.items():
        value = saved.get(field)
        if isinstance(value, str) and value and not os.environ.get(variable):
            os.environ[variable] = value  # the environment Studio was started with wins


def _save_credentials(values: dict[str, str]) -> None:
    """Persist what the user typed and apply it now. Created 0600: it holds API keys, and a write
    followed by chmod would leave them readable under the umask in between."""
    path = _credentials_file()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        saved = {}
    saved.update(values)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(saved))
    path.chmod(0o600)  # a file created by an older version may still carry the umask mode
    for field, variable in _CREDENTIALS.items():
        if field in values:
            os.environ[variable] = values[field]


def _brain_catalog() -> list[dict[str, Any]]:
    """The pluggable LLM backends the UI can pick from: availability flag + that backend's models."""
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
            # What is actually missing, so "unavailable" never sits beside "API key set".
            "detail": ("API key set" if has_key else "add an API key")
            if installed("anthropic")
            else "pip install anthropic",
            "available": installed("anthropic") and has_key,
            "models": _CLAUDE_MODELS,
        },
        {
            "id": "openai",
            "label": "Local model",
            "detail": base_url if installed("openai") else "pip install openai",
            "available": installed("openai"),
            "models": [],  # whatever the local server hosts: free text in the UI
        },
    ]


def _llm_state() -> dict[str, Any]:
    return {"current": _reg.brain(), "model": _reg.model(), "options": _brain_catalog()}


@app.get("/api/llm")
async def get_llm() -> dict[str, Any]:
    return _llm_state()


@app.post("/api/llm")
async def set_llm(req: LLMChoice) -> dict[str, Any]:
    """Pick a brain and, for the ones that need it, plug in its credentials.

    A live agent captured the old environment at construction, so every one is dropped: the next turn
    builds its brain from what was just saved rather than from what Studio started with."""
    if req.brain:
        if req.brain not in {b["id"] for b in _brain_catalog()}:
            raise HTTPException(400, "unknown LLM backend")
        _reg.set_brain(req.brain)
    if req.model is not None:
        _reg.set_model(req.model.strip())
    typed = {field: getattr(req, field).strip() for field in _CREDENTIALS if getattr(req, field) is not None}
    if typed:
        _save_credentials(typed)
    if req.brain or typed:
        await _reg.close_agents()
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


def _cpu_percent() -> float | None:
    """Host CPU load since the previous snapshot, averaged over all cores.

    Interval-free, so polling never blocks the thread: psutil reports the load since its own last call,
    which is exactly the gap between two snapshots. Primed at import below, so the first snapshot a
    browser asks for already measures against a real baseline instead of reading a flat 0%.
    """
    try:
        import psutil
    except ImportError:
        return None
    return round(psutil.cpu_percent(interval=None), 1)


with suppress(ImportError):
    __import__("psutil").cpu_percent(interval=None)  # start the counter; the first real poll measures from here


def _gpu_utilisation() -> dict[int, int]:
    """Compute load per GPU index, as NVML reports it: how busy the card is, which its VRAM does not
    say: a job can hold 20 GB and compute nothing. Empty when NVML is absent or refuses."""
    try:
        import pynvml
    except ImportError:
        return {}
    try:
        pynvml.nvmlInit()
        return {
            i: int(pynvml.nvmlDeviceGetUtilizationRates(pynvml.nvmlDeviceGetHandleByIndex(i)).gpu)
            for i in range(pynvml.nvmlDeviceGetCount())
        }
    except Exception:
        return {}


def _system_snapshot() -> dict[str, Any]:
    """Live RAM + per-GPU VRAM via KonfAI's own helpers, so the numbers match the MCP VRAM preflight.

    KonfAI is imported lazily (it pulls torch) and every probe degrades on its own: a missing GPU
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
    utilisation = _gpu_utilisation()
    for ordinal, index in enumerate(indices):
        vram = probe(lambda index=index: konfai.get_vram([index]))
        gpus.append(
            {
                "index": index,
                "name": names[ordinal] if ordinal < len(names) else f"GPU {index}",
                "used_gb": vram[0] if vram else None,
                "total_gb": vram[1] if vram else None,
                "util_percent": utilisation.get(index),
            }
        )
    return {
        "gpus": gpus,
        "ram": {"used_gb": ram[0], "total_gb": ram[1]} if ram else None,
        "cpu_percent": _cpu_percent(),
    }


@app.get("/api/system")
async def system() -> dict[str, Any]:
    """Live compute resources for the title bar: per-GPU VRAM and system RAM (off the event loop)."""
    return await asyncio.to_thread(_system_snapshot)


_STAT_KEYS = ("mean", "std", "min", "max", "median")
_EVAL_RUNS_SHOWN = 20


def _read_eval_metrics(session: str) -> list[dict[str, Any]]:
    """Aggregate metrics from every ``Metric_<SPLIT>.json`` a task has produced (newest first).

    Mirrors konfai's evaluator JSON: ``{case, aggregates:{metric:{mean,std,…}}, directions}``. How a
    metric file is read is konfai-mcp's to define, so this borrows its rules rather than restating
    them: a run's identity, which rows are whole metrics, and which way each one is better. The
    leaderboard beside this table is that same service, and the two must name and rank a run alike.

    A direction belongs to the criterion, not to the run: it is resolved once per metric, from the
    newest run that declares it, so runs predating a criterion's ``maximize`` flag cannot have the
    same PSNR read "↓" on one card and "↑" on the next.
    """
    root = _session_dir(session)
    if not root.is_dir():
        return []
    runs: list[dict[str, Any]] = []
    declared: dict[str, str] = {}
    files = sorted(root.rglob("Metric_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in files[:_EVAL_RUNS_SHOWN]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        aggregates = payload.get("aggregates")
        if not isinstance(aggregates, dict):
            continue
        directions = payload.get("directions")
        if isinstance(directions, dict):
            for name, value in directions.items():  # newest declaration of a metric wins
                declared.setdefault(name, value)
        metrics: list[dict[str, Any]] = []
        for name, stats in top_level_metrics(aggregates).items():
            if isinstance(stats, dict) and isinstance(stats.get("mean"), (int, float)):
                row: dict[str, Any] = {"name": name}
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
                "run": metric_run_name(path, root),
                "split": split,
                "metrics": metrics,
                "cases": len(case_rows),
                "case_metrics": metric_names,
                "case_rows": case_rows,
            }
        )
    for run in runs:  # every card reads one direction per metric, resolved across the session
        for row in run["metrics"]:
            row["direction"] = metric_direction(row["name"], declared.get(row["name"]))[0]
    return runs


@app.get("/api/evaluations")
async def evaluations(session: str = Query("default")) -> dict[str, list[dict[str, Any]]]:
    return {"runs": _read_eval_metrics(_sane_session(session))}


@app.get("/api/leaderboard")
async def leaderboard(session: str = Query("default"), split: str = Query("TRAIN")) -> dict[str, Any]:
    """Rank the experiment's runs by their evaluation metrics: proxies konfai-mcp's ``leaderboard`` (which
    reads the Metric_<SPLIT>.json files live; nothing extra is persisted). One ranking per metric."""
    ok, text, payload = await _mcp_json(_sane_session(session), "leaderboard", {"split": split})
    if not ok:
        return {"ok": False, "detail": text}
    return {"ok": True, **payload} if isinstance(payload, dict) else {"ok": True}


#: Where a run directory sits, relative to the session root: at the root of an output bucket
#: (``Statistics/<run>``, ``AppEvaluations/<run>``) or inside an isolated app subtree.
_RUN_DIR_PATTERNS = ("{run}", "*/{run}", "*/*/{run}")


def _run_config_snapshot(session: str, run: str) -> Path | None:
    """The newest launch-time config snapshot for a run, wherever its kind writes one.

    A run's config sits in the directory that bears its name: ``Statistics/<run>/Config_0.yml`` for a
    training, ``AppEvaluations/<run>/<app>/Evaluation.yml`` for an app evaluation. Looking under
    ``Statistics/`` alone answered "no config snapshot" for every run the leaderboard ranks, which is
    exactly what it offers to compare. So: find the run's directory within three levels of the session
    root, then take the newest config in it. Jailed: a run with a path separator is refused, and every
    hit is re-checked against the root."""
    # The name goes into a glob pattern, so wildcards are refused alongside separators: '*' would
    # match every run in the session and diff whichever happened to be newest.
    if not run or any(char in run for char in "/\\*?[]") or ".." in run:
        return None
    base = _session_dir(session).resolve()
    if not base.is_dir():
        return None
    run_dirs = [d for pattern in _RUN_DIR_PATTERNS for d in base.glob(pattern.format(run=run)) if d.is_dir()]
    snaps = [
        config
        for run_dir in run_dirs
        for config in (*run_dir.glob("*.yml"), *run_dir.glob("*/*.yml"))
        if config.is_file() and _jail(base, str(config.relative_to(base))) is not None
    ]

    def rank(config: Path) -> tuple[int, float]:
        """The training's resolved config first, newest otherwise.

        A finished run leaves one config per kind, and its evaluation config is always the newest: taken
        on mtime alone, two runs were compared on their Evaluation.yml, which holds neither the model nor
        the losses nor the optimizer, so the panel showed everything except what was changed."""
        return (1 if config.relative_to(base).parts[0] == "Statistics" else 0, config.stat().st_mtime)

    return max(snaps, key=rank) if snaps else None


@app.get("/api/run/config_diff")
async def run_config_diff(
    session: str = Query("default"), run_a: str = Query(...), run_b: str = Query(...)
) -> dict[str, Any]:
    """A unified diff of two runs' launch-time config snapshots (what actually differs between them: model,
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
    """A run's full training curves (the complete downsampled TensorBoard history, not the live tail): konfai-mcp's ``read_training_curves``, optionally filtered to tags containing ``q``. Powers clicking a
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
    base: str | None = None  # the text the editor opened, to detect a file that moved under it


@app.post("/api/config/save")
async def save_config(req: ConfigSave) -> dict[str, bool]:
    """Save an edited config YAML: jailed to the session workspace, existing .yml only, atomic
    write (temp + replace) so a reader never sees a truncated config.

    The agent and every workflow write configs too (KonfAI resolves defaults back into the file), so a
    save carrying the text it started from is refused when the file no longer holds it."""
    target = _session_path(req.session, req.name)
    if target.suffix.lower() not in {".yml", ".yaml"}:
        raise HTTPException(415, "only YAML configs are editable")
    if not target.is_file():
        raise HTTPException(404, "config not found")
    if req.base is not None and target.read_text(encoding="utf-8", errors="replace") != req.base:
        raise HTTPException(409, "this file changed on disk since you opened it")
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
        # Python's cache directories are machinery, not experiment content: noise in a clinician's tree.
        if entry.name.startswith(".") or entry.name == "__pycache__":
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
    (the latest lines: what matters for a long training log), not a truncated head; such a file is
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
    """What an experiment contains: drives the UI's overview and greys out impossible actions."""
    root = _session_dir(session)
    if not root.is_dir():
        return {"checkpoints": [], "predictions": [], "jobs": [], "bundlable": False, "exportable": False}
    # "**/Checkpoints" / "**/Predictions" so isolated app outputs (<app_output>-<hash>/…) count too.
    checkpoints = sorted(str(p.relative_to(root)) for p in root.glob("**/Checkpoints/**/*.pt"))
    predictions = sorted(
        {p.name for pred in root.glob("**/Predictions") if pred.is_dir() for p in pred.iterdir() if p.is_dir()}
    )
    jobs = [
        {
            "run": payload.get("run_name"),
            "kind": payload.get("kind"),
            "status": _status_for(payload, _runtime_log(payload)),
        }
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
    """What the experiment holds, plus where it stands and the moves open from there, so a page reload
    comes back with the same next actions instead of an empty bar. The moves are the ones a turn would
    offer: the assistant's own while they still describe this state, the derived ones otherwise."""
    name = _sane_session(session)
    state = _state(name)
    return {
        **_experiment_info(name),
        "dataset": _reg.dataset(name),
        "workflow": state,
        "moves": _turn_moves(name, [], state, None),
    }


class AppRef(BaseModel):
    ref: str
    session: str = "apps"


def _parse_apps(data: Any) -> list[dict[str, Any]]:
    """The ``apps`` list from a parsed ``list_apps`` payload (empty when unshaped)."""
    apps = data.get("apps") if isinstance(data, dict) else None
    return apps if isinstance(apps, list) else []


@app.get("/api/apps")
async def apps(session: str = Query("apps")) -> dict[str, Any]:
    """The konfai-mcp app catalogue (shipped + registered sources): a direct MCP call, no LLM.

    Bundle metadata maps onto what the App Zoo renders: an app with its own ``icon.png`` gets a
    ``logo`` URL (served below), and its declared ``task`` becomes the grouping ``theme``.
    """
    ok, _text, data = await _mcp_json(_sane_session(session), "list_apps", {"include_summary": True})
    listed = _parse_apps(data)
    for entry in listed:
        if entry.get("has_icon") and entry.get("ref"):
            entry["logo"] = f"/api/apps/icon?ref={quote(str(entry['ref']), safe='')}"
        if entry.get("task") and not entry.get("theme"):
            entry["theme"] = str(entry["task"]).capitalize()
    return {"ok": ok, "apps": listed}


#: Tasks kept alive across their own await. The loop holds only a weak reference to a bare
#: ``create_task``, so a task that sleeps before doing its work can be collected before it runs.
_SHUTDOWN_TASKS: set[asyncio.Task] = set()


def _forwarded_by_a_proxy(request: Request) -> bool:
    """Whether anything in front of this server claims to have forwarded the request.

    Any of them is enough, and the prefix is matched rather than a list: a proxy that sets only
    ``X-Real-IP``, or only a vendor's own ``X-Forwarded-Host``, still means the peer address belongs
    to the proxy. What matters is not which header arrived but that one did.
    """
    return any(
        name == "forwarded" or name == "x-real-ip" or name.startswith("x-forwarded-") for name in request.headers
    )


def _trusts_proxy_headers() -> bool:
    """Whether uvicorn was told to rewrite the peer address from ``X-Forwarded-For``.

    Set by the CLI, because the app is imported from a string and never sees its arguments. Read at
    call time rather than at import so a test can set it.
    """
    return os.environ.get("KONFAI_STUDIO_PROXY_HEADERS") == "1"


@app.post("/api/quit")
async def quit_server(request: Request) -> dict[str, bool]:
    """Stop the Studio server (graceful: the lifespan teardown closes agents and reaps TensorBoards).

    Three guards, because none of them alone is enough.

    The client must be on this machine, so a remote user cannot take a shared server down, token or
    not. That reads the TCP peer, which is the real client only when nothing sits in front: behind
    the same-host reverse proxy REMOTE.md documents, every request arrives from 127.0.0.1. Uvicorn
    rewrites the peer from ``X-Forwarded-For`` under ``--proxy-headers``, so a forwarding header
    arriving WITHOUT that flag means the peer belongs to the proxy and the loopback check answers a
    question nobody asked: refuse instead.

    And it must send the header below: the loopback check only proves the TCP peer is local, which
    any page open in the user's browser also is, so without it a drive-by form POST to localhost
    would shut Studio down. A custom header is unforgeable from a form and, cross-origin, needs a
    preflight this server never grants.

    Slicer's Studio button and the titlebar power button rely on this: the server runs detached,
    with no terminal to Ctrl+C.
    """
    import signal

    client = request.client.host if request.client else ""
    if client not in ("127.0.0.1", "::1"):
        raise HTTPException(403, "the Studio server can only be stopped from its own machine")
    if _forwarded_by_a_proxy(request) and not _trusts_proxy_headers():
        raise HTTPException(
            403,
            "this request came through a proxy and the server was not started with --proxy-headers,"
            " so it cannot tell which machine it is from",
        )
    if request.headers.get("x-konfai-studio") != "quit":
        raise HTTPException(403, "missing the X-KonfAI-Studio header: stop Studio from its own UI")

    async def _after_reply() -> None:
        await asyncio.sleep(0.3)  # let this response leave before the shutdown begins
        os.kill(os.getpid(), signal.SIGTERM)

    # Held: the loop keeps only a weak reference and would collect the task mid-sleep.
    _SHUTDOWN_TASKS.add(task := asyncio.get_running_loop().create_task(_after_reply()))
    task.add_done_callback(_SHUTDOWN_TASKS.discard)
    return {"ok": True}


@app.get("/api/apps/icon")
async def app_icon(ref: str = Query(...)) -> FileResponse:
    """The app's own bundle icon (its ``icon.png`` / ``app.json``-declared icon file)."""
    try:
        from konfai_apps.app_repository import AppRepositoryError, get_app_repository_info
    except ImportError as exc:  # pragma: no cover - konfai-apps not installed
        raise HTTPException(503, "konfai-apps is not installed") from exc
    try:
        icon_path = get_app_repository_info(ref, force_update=False).get_icon_path()
    except (AppRepositoryError, FileNotFoundError, OSError, ValueError) as exc:
        raise HTTPException(404, f"app '{ref}' has no icon") from exc
    if icon_path is None or not Path(icon_path).is_file():
        raise HTTPException(404, f"app '{ref}' has no icon")
    # No media_type: app.json may name a .svg or .webp, so let the response infer it from the path.
    return FileResponse(icon_path)


def _app_bundle_file(ref: str, filename: str) -> Path:
    """Local path to a file inside an app's bundle (for serving its ONNX deploy artifacts).

    Resolves the app repository and returns the file path WITHOUT importing the app's ``.py`` (only
    the bundled files are touched). Raises 404 when the app is not an ONNX-deployable bundle.
    """
    try:
        from konfai_apps.app_repository import AppRepositoryError, LocalAppRepository, get_app_repository_info
    except ImportError as exc:  # pragma: no cover - konfai-apps not installed
        raise HTTPException(503, "konfai-apps is not installed") from exc
    try:
        repo = get_app_repository_info(ref, force_update=False)
        if not isinstance(repo, LocalAppRepository):  # only local/HF bundles hold downloadable files
            raise HTTPException(404, f"app '{ref}' has no portable ONNX bundle")
        path = Path(repo._download(filename))
    except (AppRepositoryError, FileNotFoundError, OSError, KeyError, ValueError) as exc:
        raise HTTPException(404, f"app '{ref}' has no portable ONNX bundle") from exc
    if not path.is_file():
        raise HTTPException(404, f"app '{ref}' has no '{filename}' (export one with `konfai-apps bundle --onnx`)")
    return path


@app.get("/api/apps/manifest")
async def app_manifest(ref: str = Query(...)) -> dict[str, Any]:
    """The deploy manifest of an app's ONNX bundle: the contract the in-tab runtime consumes."""
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
    """Delete one run's outputs: proxies konfai-mcp's jailed ``delete_run`` (never leaves the workspace)."""
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
    """Ask a running training job to run a validation pass now: ``request_validation`` signals the job
    (SIGUSR1); the trainer validates at its next iteration boundary and the metrics stream into Live."""
    name = _sane_session(req.session)
    job_id, error = _require_train_job(name)
    if error is not None:
        return error
    return await _mcp_detail(name, "request_validation", {"job_id": job_id})


@app.post("/api/job/tunables")
async def set_tunables(req: SetTunables) -> dict[str, Any]:
    """Change a running training's lr / it_validation mid-run: proxies konfai-mcp's ``set_live_tunables``,
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


@app.get("/assets/{file_path:path}")
async def assets(file_path: str) -> FileResponse:
    """Serve the built Vite assets (JS/CSS) from ``web/assets``: jailed to that dir.

    Vite fingerprints every asset name, so a given URL's bytes never change: cache it for a year and a
    warm reload fetches nothing. A new build produces new names, which ``index.html`` points at.
    """
    target = _jail(WEB_DIR / "assets", file_path)
    if target is None or not target.is_file():
        raise HTTPException(404, "asset not found")
    return FileResponse(str(target), headers={"Cache-Control": "public, max-age=31536000, immutable"})


@app.get("/")
async def index() -> FileResponse:
    """The one file that must never be cached: it is what names the fingerprinted bundle.

    Served without cache headers the browser applies its own heuristic, keeps a stale index, and so
    keeps loading the previous build's assets: an updated Studio silently serving the old front until
    someone thinks to hard-reload.
    """
    return FileResponse(WEB_DIR / "index.html", headers={"Cache-Control": "no-cache, must-revalidate"})


@app.get("/konfai-logo.png")
async def logo() -> FileResponse:
    return FileResponse(WEB_DIR / "konfai-logo.png")


# Volume image formats NiiVue reads client-side (NIfTI / MHA and their compressed forms).
_VOLUME_SUFFIXES = {".nii", ".nii.gz", ".mha", ".mhd", ".nrrd", ".gz"}


def _volume_path(path: str) -> Path:
    """A readable volume at ``path``, or the refusal saying why not.

    One gate for both ways out of Studio (the browser viewer and the 3D Slicer hand-over), so a volume
    one opens is never a volume the other rejects."""
    volume = Path(path).expanduser()
    if not volume.is_file():
        raise HTTPException(404, f"volume not found: {path}")
    if volume.suffix.lower() not in _VOLUME_SUFFIXES:
        raise HTTPException(415, f"unsupported volume type: {volume.suffix}")
    return volume


@app.get("/files/volume")
async def volume(path: str = Query(..., description="Absolute host path of the volume to stream")) -> FileResponse:
    """Stream a medical volume to the browser (NiiVue) with HTTP range support.

    Trusted-local deployment only: like konfai-mcp's dataset tools, volumes may live anywhere
    on the host and are served **read-only**: this never exposes a write path. Starlette's
    FileResponse honours the ``Range`` header, so NiiVue can fetch a large volume in chunks.
    """
    found = _volume_path(path)
    return FileResponse(str(found), media_type="application/octet-stream", filename=found.name)


class SlicerOpen(BaseModel):
    paths: list[str]


_SLICER_WEBSERVER_PORT = 2016  # the Web Server module's default port


def _slicer_exec(code: str) -> bool:
    """Run Python in an already-open 3D Slicer via its Web Server module, if one listens locally.

    The module's ``/slicer/exec`` endpoint only answers when the user started the server AND ticked
    "enable exec"; any other outcome (nothing listening, exec disabled) is a plain False so the
    caller falls back to launching Slicer itself."""
    import urllib.error
    import urllib.request

    request = urllib.request.Request(
        f"http://127.0.0.1:{_SLICER_WEBSERVER_PORT}/slicer/exec", data=code.encode(), method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=3) as response:
            response.read()
        return True
    except (OSError, urllib.error.URLError):
        return False


def _slicer_executable() -> str | None:
    configured = os.environ.get("KONFAI_STUDIO_SLICER")
    if configured:
        return configured if Path(configured).is_file() else None
    return shutil.which("Slicer") or shutil.which("slicer")


@app.post("/api/slicer/open")
async def slicer_open(req: SlicerOpen) -> dict[str, Any]:
    """Hand volumes over to 3D Slicer on this machine: into the running instance when its Web Server
    module listens (``/slicer/exec``), else by launching Slicer on them.

    Trusted-local, like ``/files/volume``: Slicer runs beside the Studio server and reads the same
    paths, so nothing is copied or exposed. The JSON body doubles as the CSRF guard (a cross-origin
    form cannot send ``application/json`` without a preflight this server never grants)."""
    volumes = [_volume_path(raw) for raw in req.paths]
    if not volumes:
        raise HTTPException(400, "no volume to open")
    code = "\n".join(f"slicer.util.loadVolume({str(p)!r})" for p in volumes)
    if await asyncio.to_thread(_slicer_exec, code):
        return {"ok": True, "via": "webserver"}
    executable = _slicer_executable()
    if executable is None:
        return {
            "ok": False,
            "detail": "3D Slicer not found: put `Slicer` on PATH or set KONFAI_STUDIO_SLICER to the executable.",
        }
    import subprocess

    subprocess.Popen(
        [executable, *map(str, volumes)],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return {"ok": True, "via": "launch"}
