# Copyright (c) 2026 Fideus Labs LLC
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

"""KonfAI Scout and FidOpt behind the MCP server.

Scout: an inspection is a tracked job running ``konfai_scout.inspect`` on a case JSON written under the session
(``Scout/<name>/``); the report is read back without its images. FidOpt: the adapter calls FidOpt's own CLI and
Python (``KONFAI_MCP_FIDOPT`` names the ``fidopt`` executable; the ``python`` beside it answers the descriptor
queries) and launches only the operations ``KONFAI_MCP_FIDOPT_ALLOWED`` lists. Nothing of FidOpt is reimplemented
here, and no test truth is read: FidOpt itself keeps that on its scoring server.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess  # nosec B404 - FidOpt's own executable, argv lists, no shell
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

from .workspace import WorkspaceLayout

_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}")
_FIDOPT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._@/-]{0,99}")


class ScoutService:
    """Inspections under the session's ``Scout/`` folder."""

    def __init__(self, workspace_layout: WorkspaceLayout) -> None:
        self.workspace_layout = workspace_layout

    def root(self) -> Path:
        return self.workspace_layout.ensure_session_workspace() / "Scout"

    def prepare_inspection(
        self,
        *,
        task: str,
        inputs: dict[str, str],
        name: str | None = None,
        labels: dict[str, str] | None = None,
        evaluation: list[str] | None = None,
        notes: str = "",
        **options: Any,
    ) -> dict[str, Any]:
        """The job spec of one inspection: the case and config JSON written, the privacy gate passed. ``options``
        are ``ScoutConfig`` fields; the ``Budget`` ones (``max_*``) go to its budget."""
        try:
            from konfai_scout.case import CaseError, InspectionCase
            from konfai_scout.config import Budget, ScoutConfig
            from konfai_scout.providers import ProviderError, check_privacy
        except ImportError as exc:
            raise ValueError("KonfAI Scout is not installed beside this server: pip install konfai-scout") from exc
        run_name = self.workspace_layout.sanitize_name(
            name or f"{task}_{Path(next(iter(inputs.values()), 'case')).stem}"
        )
        if not _NAME.fullmatch(run_name):
            raise ValueError("name: letters, digits, '.', '_' and '-' only.")
        resolved = {role: str(Path(path).expanduser().resolve()) for role, path in inputs.items()}
        evaluations = [str(Path(p).expanduser().resolve()) for p in (evaluation or [])]
        try:
            case = InspectionCase.from_dict(
                {
                    "task": task,
                    "name": run_name,
                    "inputs": resolved,
                    "labels": labels or {},
                    "evaluation": evaluations,
                    "notes": notes,
                }
            )
            budget_keys = {f.name for f in fields(Budget)}
            budget = {key: options.pop(key) for key in tuple(options) if key in budget_keys}
            config = ScoutConfig(budget=Budget(**budget), **options)
            check_privacy(config)
        except (CaseError, ProviderError, ValueError) as exc:
            raise ValueError(str(exc)) from exc
        output = self.root() / run_name
        output.mkdir(parents=True, exist_ok=True)
        case.save(output / "case.json")
        (output / "config.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
        return {
            "kind": "scout",
            "run_name": run_name,
            "target": "konfai_mcp.runner:run_scout_inspection_api",
            "command": [
                "konfai-scout",
                "inspect",
                "--case",
                str(output / "case.json"),
                "--config",
                str(output / "config.json"),
                "--out",
                str(output),
            ],
            "kwargs": {"case": str(output / "case.json"), "config": str(output / "config.json"), "output": str(output)},
            "output": str(output),
            "devices": ["cpu"],  # the model is an API or its own server, and KonfAI Eval measures with cpu=1
        }

    def list_reports(self) -> list[dict[str, Any]]:
        root = self.root()
        out = []
        for report in sorted(root.glob("*/report.json")):
            try:
                data = json.loads(report.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            conclusion = data.get("conclusion") or {}
            out.append(
                {
                    "name": report.parent.name,
                    "task": (data.get("case") or {}).get("task"),
                    "case": (data.get("case") or {}).get("name"),
                    "verdict": conclusion.get("verdict"),
                    "observations": len(data.get("observations") or []),
                    "created_at": data.get("created_at"),
                    "mode": (data.get("config") or {}).get("mode"),
                    "model": (data.get("provider") or {}).get("model"),
                    "path": str(report),
                }
            )
        return out

    def report_dir(self, name: str) -> Path:
        """The report directory of a name, inside ``Scout/``: no path reaches out of it."""
        if not _NAME.fullmatch(name or ""):
            raise ValueError("name: letters, digits, '.', '_' and '-' only.")
        folder = (self.root() / name).resolve()
        if folder.parent != self.root().resolve():
            raise ValueError("name: a report of this session.")
        return folder

    def read_report(self, name: str) -> dict[str, Any]:
        folder = self.report_dir(name)
        path = folder / "report.json"
        if not path.is_file():
            raise ValueError(f"No report named {name!r} in this session (list_scout_reports lists them).")
        data = json.loads(path.read_text(encoding="utf-8"))
        views = [{k: v for k, v in view.items() if k not in ("px_to_mm", "sha256")} for view in data.get("views") or []]
        return {
            "name": name,
            "report_json": str(path),
            "report_md": str(folder / "report.md"),
            "schema_version": data.get("schema_version"),
            "case": data.get("case"),
            "conclusion": data.get("conclusion"),
            "observations": data.get("observations"),
            "measurements": data.get("measurements"),
            "views": views,
            "usage": data.get("usage"),
            "budget": data.get("budget"),
            "error": data.get("error"),
            "provider": data.get("provider"),
            "mode": (data.get("config") or {}).get("mode"),
            "next_actions": ["read_scout_report", "run_scout_inspection", "preview_volume"],
        }


class FidoptAdapter:
    """FidOpt through its own CLI and Python; nothing reimplemented, nothing launched outside the allowed list."""

    def __init__(self, workspace_layout: WorkspaceLayout) -> None:
        self.workspace_layout = workspace_layout
        self.executable = os.environ.get("KONFAI_MCP_FIDOPT") or shutil.which("fidopt") or "fidopt"
        beside = Path(self.executable).with_name("python")
        self.python = os.environ.get("KONFAI_MCP_FIDOPT_PYTHON") or (str(beside) if beside.exists() else sys.executable)
        allowed = os.environ.get("KONFAI_MCP_FIDOPT_ALLOWED", "check")
        self.allowed = tuple(part.strip() for part in allowed.split(",") if part.strip())
        self.timeout = 120.0

    def available(self) -> None:
        if not (Path(self.executable).exists() or shutil.which(self.executable)):
            raise ValueError(
                "FidOpt is not reachable from this server: set KONFAI_MCP_FIDOPT to the fidopt executable of a FidOpt "
                "checkout (and KONFAI_MCP_FIDOPT_PYTHON to its Python when the executable has none beside it)."
            )

    def _python(self, code: str, *args: str) -> Any:
        self.available()
        completed = subprocess.run(  # nosec B603 - FidOpt's own interpreter, a fixed snippet, checked names
            [self.python, "-c", code, *args], capture_output=True, text=True, timeout=self.timeout, check=False
        )
        if completed.returncode != 0:
            raise ValueError(f"fidopt query failed: {completed.stderr.strip()[-800:]}")
        return json.loads(completed.stdout)

    def _cli(self, *args: str) -> str:
        self.available()
        completed = subprocess.run(  # nosec B603
            [self.executable, *args], capture_output=True, text=True, timeout=self.timeout, check=False
        )
        if completed.returncode != 0:
            raise ValueError(f"fidopt {' '.join(args)} failed: {completed.stderr.strip()[-800:]}")
        return completed.stdout

    @staticmethod
    def _check_name(value: str, what: str) -> str:
        if not isinstance(value, str) or not _FIDOPT_NAME.fullmatch(value):
            raise ValueError(f"{what}: letters, digits, '.', '_', '-', '@' and '/' only.")
        return value

    def list_datasets(self, task: str | None = None) -> list[dict[str, Any]]:
        code = (
            "import json, sys\n"
            "from fidopt.corpus import datasets, descriptor, task\n"
            "wanted = sys.argv[1] or None\n"
            "out = []\n"
            "for name in datasets():\n"
            "    info = descriptor(name)\n"
            "    if wanted and task(info) != wanted:\n"
            "        continue\n"
            "    out.append({'name': name, 'task': task(info), 'title': info.get('title'), 'modalities': info.get('modalities'),\n"
            "                'anatomy': info.get('anatomy'), 'cases': len(info.get('cases', [])),\n"
            "                'development': len(info.get('development', [])), 'spdx': info.get('spdx'), 'use': info.get('use'),\n"
            "                'dimension': info.get('dimension', 3)})\n"
            "print(json.dumps(out))\n"
        )
        return self._python(code, task or "")

    def describe_dataset(self, name: str) -> dict[str, Any]:
        self._check_name(name, "dataset")
        code = (
            "import json, sys\n"
            "from fidopt.corpus import descriptor, task\n"
            "info = descriptor(sys.argv[1])\n"
            "info['task'] = task(info)\n"
            "print(json.dumps(info))\n"
        )
        return self._python(code, name)

    def list_methods(self) -> list[dict[str, Any]]:
        code = (
            "import json\n"
            "from fidopt.corpus import REPOSITORY\n"
            "out = []\n"
            "for path in sorted((REPOSITORY / 'methods').glob('*/method.json')):\n"
            "    info = json.loads(path.read_text())\n"
            "    out.append({'name': path.parent.name, 'title': info.get('title'), 'task': info.get('task', 'registration'),\n"
            "                'summary': info.get('summary'), 'app': info.get('app'), 'licence': info.get('licence'),\n"
            "                'version': info.get('version'), 'variants': sorted(info.get('variants', {}))})\n"
            "print(json.dumps(out))\n"
        )
        return self._python(code)

    def table(self, dataset: str, research: bool = False) -> dict[str, Any]:
        self._check_name(dataset, "dataset")
        text = self._cli("table", dataset, "--json", *(["--research"] if research else []))
        return json.loads(text)

    def prepare_run(
        self,
        operation: str,
        method: str,
        dataset: str,
        cases: list[str] | None,
        scout: str | None,
        scout_allow_cloud: bool,
        extra: list[str] | None,
    ) -> dict[str, Any]:
        """The job spec of one allowed FidOpt operation."""
        if operation not in self.allowed:
            raise ValueError(
                f"fidopt {operation!r} is not allowed on this server (KONFAI_MCP_FIDOPT_ALLOWED={','.join(self.allowed)})."
            )
        self.available()
        args = [operation, self._check_name(method, "method")]
        if operation == "check":
            if dataset:
                args += [
                    "--on",
                    self._check_name(dataset, "dataset"),
                    *[self._check_name(c, "case") for c in cases or []],
                ]
        else:
            args.append(self._check_name(dataset, "dataset"))
            for case in cases or []:
                args += ["--case", self._check_name(case, "case")]
        if scout:
            if not re.fullmatch(r"[a-z]+(:[A-Za-z0-9._:/-]+)?", scout):
                raise ValueError("scout: PROVIDER or PROVIDER:MODEL.")
            args += ["--scout", scout]
            if scout_allow_cloud:
                args.append("--scout-allow-cloud")
        for item in extra or []:
            # no '/' and no '..', so that no value steers FidOpt to a path; '=' and ',' are --try's LABEL=SETTINGS
            if ".." in item or not re.fullmatch(r"--[a-z][a-z0-9-]*(=[A-Za-z0-9._:=,-]+)?|[A-Za-z0-9._:=,-]+", item):
                raise ValueError(f"extra argument {item!r}: flags and plain values only, no paths outside the corpus.")
            args.append(item)
        label = self.workspace_layout.sanitize_name(f"fidopt_{operation}_{method}_{dataset or 'none'}")
        output = self.workspace_layout.ensure_session_workspace() / "FidOpt" / label
        output.mkdir(parents=True, exist_ok=True)
        # FidOpt writes its rows where its own configuration says, so the job's folder only records what was asked
        # (and is never the empty output the registry would read as a failed run).
        (output / "command.json").write_text(
            json.dumps({"executable": self.executable, "args": args}, indent=1), encoding="utf-8"
        )
        return {
            "kind": "fidopt",
            "run_name": label,
            "target": "konfai_mcp.runner:run_fidopt_api",
            "command": [self.executable, *args],
            "kwargs": {"executable": self.executable, "args": args},
            "output": str(output),
        }
