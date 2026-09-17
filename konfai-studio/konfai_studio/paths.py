# SPDX-License-Identifier: Apache-2.0
"""Workspace-root resolution, the ``_jail`` path guard, session-scoped paths, and the recent-items
history files. A leaf module: it imports nothing from the rest of the package."""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

from fastapi import HTTPException


def _sane_session(name: str) -> str:
    """Sanitize a session name to a safe workspace dir (mirrors konfai-mcp's own rule)."""
    cleaned = re.sub(r"[^a-zA-Z0-9._-]", "_", (name or "").strip())
    return cleaned if cleaned and cleaned not in {".", ".."} else "default"


def _jail(root: Path, rel: str) -> Path | None:
    """Resolve ``root/rel`` and return it only when it stays under ``root`` (else None)."""
    base = root.resolve()
    target = (base / rel).resolve() if rel else base
    return target if target == base or base in target.parents else None


def _workspace_root() -> Path:
    return Path(os.environ.get("KONFAI_MCP_WORKSPACES_ROOT") or Path.home() / "KonfAI_Workspaces")


def _sessions_file() -> Path:
    return _workspace_root() / ".konfai_studio" / "sessions.json"


def _credentials_file() -> Path:
    """Where the LLM credentials set from the UI live: outside sessions.json, and readable by nobody else."""
    return _workspace_root() / ".konfai_studio" / "credentials.json"


def _dataset_history_file() -> Path:
    return _workspace_root() / ".konfai_studio" / "datasets.json"


def _files_history_file() -> Path:
    return _workspace_root() / ".konfai_studio" / "files.json"


def _sessions_root() -> Path:
    return _workspace_root() / "sessions"


def _session_dir(session: str) -> Path:
    """A session's konfai-mcp workspace. Where a session lives is stated here and nowhere else."""
    return _sessions_root() / _sane_session(session)


def _studio_session_dir(session: str) -> Path:
    """Studio's own per-session files (transcript, brain history), OUTSIDE the konfai-mcp workspace.

    The workspace belongs to konfai-mcp: initialize_session(overwrite=True) legitimately rmtrees it.
    Studio files stored inside were deleted with it (a whole chat history, silently), and their mere
    presence made a pristine workspace look non-empty, forcing the agent onto that destructive path.
    Legacy files are adopted from the workspace on first touch."""
    home = _workspace_root() / ".konfai_studio" / "sessions" / _sane_session(session)
    for name in ("transcript.json", "history.json"):
        legacy = _session_dir(session) / (name if name == "transcript.json" else f".konfai_studio/{name}")
        target = home / name
        if legacy.is_file() and not target.exists():
            home.mkdir(parents=True, exist_ok=True)
            shutil.move(str(legacy), target)
    legacy_nest = _session_dir(session) / ".konfai_studio"
    if legacy_nest.is_dir() and not any(legacy_nest.iterdir()):
        legacy_nest.rmdir()
    return home


def _session_jobs_dir(session: str) -> Path:
    """Where konfai-mcp records that session's jobs."""
    return _session_dir(session) / ".konfai_mcp" / "jobs"


def _delete_workspace(name: str) -> bool:
    """Delete a task's konfai-mcp workspace, jailed under ``sessions/``; whether it is gone afterwards.

    A directory that survives is adopted back into the rail on the next listing, so the caller must
    know rather than forget a session that is about to reappear."""
    # Sanitized like every path that WROTE under sessions/: an unsanitized name would jail-resolve
    # to a directory that does not exist, report success, and leave the real workspace behind.
    name = _sane_session(name)
    target = _jail(_sessions_root(), name)
    if target is None:
        return False
    shutil.rmtree(target, ignore_errors=True)
    studio_side = _jail(_workspace_root() / ".konfai_studio" / "sessions", name)
    if studio_side is not None:
        shutil.rmtree(studio_side, ignore_errors=True)
    return not target.exists()


def _session_path(session: str, rel: str) -> Path:
    """Resolve a path inside a session's workspace: jailed, never escapes the session root."""
    target = _jail(_session_dir(session), rel)
    if target is None:
        raise HTTPException(400, "path escapes the session workspace")
    return target


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
