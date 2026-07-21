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


def _dataset_history_file() -> Path:
    return _workspace_root() / ".konfai_studio" / "datasets.json"


def _files_history_file() -> Path:
    return _workspace_root() / ".konfai_studio" / "files.json"


def _delete_workspace(name: str) -> None:
    """Delete a task's konfai-mcp workspace, jailed under ``sessions/`` (the name is already sanitized,
    so this never escapes the workspace root)."""
    target = _jail(_workspace_root() / "sessions", name)
    if target is not None and target.is_dir():
        shutil.rmtree(target, ignore_errors=True)


def _session_path(session: str, rel: str) -> Path:
    """Resolve a path inside a session's workspace — jailed, never escapes the session root."""
    target = _jail(_workspace_root() / "sessions" / _sane_session(session), rel)
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
