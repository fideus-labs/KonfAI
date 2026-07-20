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

"""Session workspace layout: the per-session directory tree, the derived
workflow->config-file/root-key maps, and the path jail every workspace-relative
read/write must go through. Split out of ``server_support.py``."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from konfai_mcp.workflows import WORKFLOW_SPECS

WORKFLOW_CONFIG_FILES = {kind: spec.config_file for kind, spec in WORKFLOW_SPECS.items()}

WORKFLOW_ROOT_KEYS = {kind: spec.root_key for kind, spec in WORKFLOW_SPECS.items()}


@dataclass(frozen=True)
class WorkspaceLayout:
    """Filesystem layout helper for KonfAI MCP datasets, sessions, and one mutable session workspace."""

    root: Path
    current_session: str | None = None

    def __post_init__(self) -> None:
        root = self.root.expanduser().resolve()
        object.__setattr__(self, "root", root)
        object.__setattr__(self, "current_session", self._resolve_current_session(self.current_session))

    def sanitize_name(self, name: str) -> str:
        safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", name.strip())
        if safe in {"", ".", ".."}:
            raise ValueError("Name is empty or invalid after sanitization.")
        return safe

    def _resolve_current_session(self, value: str | None) -> str:
        if value is not None:
            return self.sanitize_name(value)
        env_value = os.environ.get("KONFAI_MCP_SESSION")
        if env_value:
            return self.sanitize_name(env_value)
        marker_path = self.session_marker_path()
        if marker_path.exists():
            marker_value = marker_path.read_text(encoding="utf-8").strip()
            if marker_value:
                return self.sanitize_name(marker_value)
        return "default"

    def internal_root_dir(self) -> Path:
        return self.root / ".konfai_mcp"

    def session_marker_path(self) -> Path:
        return self.internal_root_dir() / "current_session.txt"

    def apps_catalog_path(self) -> Path:
        """Editable per-root catalogue of app sources (``{"apps": [...]}``) shared across sessions."""
        return self.root / "apps_catalog.json"

    def sessions_root(self) -> Path:
        return self.root / "sessions"

    def session_dir(self, name: str | None = None) -> Path:
        session_name = self.sanitize_name(name or self.current_session or "default")
        return self.sessions_root() / session_name

    def workspace_dir(self) -> Path:
        return self.session_dir()

    def ensure_session_workspace(self) -> Path:
        workspace = self.workspace_dir()
        workspace.mkdir(parents=True, exist_ok=True)
        self.internal_root_dir().mkdir(parents=True, exist_ok=True)
        self.session_marker_path().write_text(workspace.name + "\n", encoding="utf-8")
        return workspace

    def available_sessions(self) -> list[str]:
        sessions_root = self.sessions_root()
        if not sessions_root.exists():
            return []
        return sorted(path.name for path in sessions_root.iterdir() if path.is_dir())

    def internal_dir(self) -> Path:
        return self.workspace_dir() / ".konfai_mcp"

    def jobs_dir(self) -> Path:
        return self.internal_dir() / "jobs"

    def job_dir(self, job_id: str) -> Path:
        return self.jobs_dir() / job_id

    def job_state_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "job.json"

    def job_manifest_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "manifest.json"

    def job_configs_dir(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "configs"

    def config_path(self, workflow: str) -> Path:
        filename = WORKFLOW_CONFIG_FILES.get(workflow)
        if filename is None:
            raise ValueError(f"Unsupported workflow: {workflow}")
        return self.workspace_dir() / filename

    def train_config_path(self) -> Path:
        return self.config_path("train")

    def prediction_config_path(self) -> Path:
        return self.config_path("prediction")

    def evaluation_config_path(self) -> Path:
        return self.config_path("evaluation")

    def statistics_log_path(self) -> Path:
        return self.workspace_dir() / "Statistics" / "Log.txt"

    def checkpoints_dir(self) -> Path:
        return self.workspace_dir() / "Checkpoints"

    def predictions_dir(self) -> Path:
        return self.workspace_dir() / "Predictions"

    def evaluations_dir(self) -> Path:
        return self.workspace_dir() / "Evaluations"

    def session_workspace_exists(self) -> bool:
        return self.workspace_dir().exists()

    def ensure_session_workspace_exists(self) -> Path:
        workspace = self.workspace_dir()
        if not workspace.exists():
            raise ValueError(
                f"Session workspace does not exist for session '{self.current_session}'. Call initialize_session first."
            )
        return workspace

    def resolve_workspace_relative_path(self, relative_path: str) -> Path:
        """Resolve a path inside the session workspace jail.

        Absolute paths are accepted when they resolve inside the workspace, so paths surfaced by
        job manifests (e.g. config snapshots under ``.konfai_mcp/jobs/``) can be passed back as-is.
        """
        # Resolve the workspace too: if the session leaf is a symlink, an unresolved workspace would never
        # be a parent of the resolved candidate and every in-jail path would be false-rejected.
        workspace = self.ensure_session_workspace_exists().resolve()
        candidate = Path(relative_path).expanduser()
        resolved = candidate.resolve() if candidate.is_absolute() else (workspace / candidate).resolve()
        if workspace != resolved and workspace not in resolved.parents:
            raise ValueError("path escapes the session workspace.")
        return resolved
