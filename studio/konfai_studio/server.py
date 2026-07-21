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
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from .agent import call_mcp_tool, suggest_next_prompts, suggest_title
from .auth import _AuthGate
from .auth import router as auth_router
from .jobs import _all_jobs, _job_created, _latest_job, _live_status, _sse
from .jobs import router as jobs_router
from .paths import (
    _dataset_history_file,
    _files_history_file,
    _history_add,
    _history_load,
    _jail,
    _sane_session,
    _session_path,
    _workspace_root,
)
from .registry import _Registry
from .tensorboard import reap_tb_servers
from .tensorboard import router as tensorboard_router
from .terminal import router as terminal_router

WEB_DIR = Path(__file__).parent / "web"

_reg = _Registry()


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
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
