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

from __future__ import annotations

import itertools
import json
import random
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
from konfai.utils.utils import split_path_spec
from ruamel.yaml import YAML

from . import runner as mcp_runner
from .server_jobs import Job, JobRegistry
from .server_support import (
    WORKFLOW_CONFIG_FILES,
    DatasetGroupUnreadableError,
    WorkspaceLayout,
    aggregate_case_statistics,
    available_templates,
    basename_without_suffixes,
    case_directories,
    default_group_map,
    full_suffix,
    load_template_configs,
    read_text_tail,
    summarize_classpath_signature,
    template_dir,
    template_groups,
    workflow_root_name,
)

YAML_SAFE = YAML(typ="safe")
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


@dataclass
class SessionService:
    """Encapsulate dataset, validation, summary, and leaderboard helpers for the current session workspace."""

    repo_root: Path
    examples_root: Path
    workspace_layout: WorkspaceLayout
    job_registry: JobRegistry
    max_log_tail_lines: int
    active_job_states: set[str]
    validation_levels: set[str]
    workflows: set[str]

    def _isoformat(self, timestamp: float | None) -> str | None:
        if timestamp is None:
            return None
        return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()

    def session_name(self) -> str:
        return self.workspace_layout.current_session or "default"

    def workspace_dir(self) -> Path:
        return self.workspace_layout.workspace_dir()

    def config_path(self, workflow: str) -> Path:
        return self.workspace_layout.config_path(self._normalize_workflow(workflow))

    def ordered_workflows(self) -> list[str]:
        return [workflow for workflow in WORKFLOW_CONFIG_FILES if workflow in self.workflows]

    def config_paths(self) -> dict[str, Path]:
        return {workflow: self.workspace_layout.config_path(workflow) for workflow in self.ordered_workflows()}

    def resolve_prediction_models(
        self,
        models: list[str] | None = None,
        *,
        limit: int = 1,
        run_name: str | None = None,
    ) -> list[Path]:
        if models:
            return [Path(model).expanduser().resolve() for model in models]
        return self.discover_model_paths(limit=limit, run_name=run_name)

    def _workflow_runtime_root(self, workflow: str) -> Path:
        normalized_workflow = self._normalize_workflow(workflow)
        if normalized_workflow == "train":
            return self.workspace_dir() / "Statistics"
        return {
            "prediction": self.workspace_layout.predictions_dir(),
            "evaluation": self.workspace_layout.evaluations_dir(),
        }[normalized_workflow]

    # App evaluate/pipeline runs write their metric trees under these workspace subdirs; every trial's
    # inner run dir repeats the bundle's train_name, so a trial's IDENTITY is its top-level trial dir.
    # One constant feeds both the search roots and the identity rule — a root added to only one of them
    # would silently reintroduce the shared-inner-name ambiguity.
    _APP_TRIAL_SUBDIRS = ("AppEvaluations", "AppPipelines")

    def _metric_search_roots(self, layout: WorkspaceLayout | None = None) -> list[Path]:
        layout = layout or self.workspace_layout
        workspace = layout.workspace_dir()
        # App evaluate/pipeline runs write their KonfAI Evaluations tree (Metric_<SPLIT>.json) under these
        # dirs, so leaderboard/compare_runs rank tuned app trials alongside train-branch runs.
        return [
            layout.evaluations_dir(),
            workspace / "Evaluation",
            *(workspace / subdir for subdir in self._APP_TRIAL_SUBDIRS),
        ]

    def _metric_run_name(self, metrics_path: Path, layout: WorkspaceLayout | None = None) -> str:
        """The run identifier of a metric file: its run directory's name — except an app trial
        (``AppEvaluations/<label>-<uuid>/…/Metric_*.json``), identified by its parameter-suffixed trial
        directory, because every trial's inner run dir repeats the same bundle train_name."""
        layout = layout or self.workspace_layout
        workspace = layout.workspace_dir()
        for subdir in self._APP_TRIAL_SUBDIRS:
            try:
                return metrics_path.relative_to(workspace / subdir).parts[0]
            except ValueError:
                continue
        return metrics_path.parent.name

    def _resolve_session_layout(self, session: str | None) -> WorkspaceLayout:
        """Resolve a metrics-lookup layout: the current session, or another named one."""
        if session is None or session == self.workspace_layout.current_session:
            self.workspace_layout.ensure_session_workspace_exists()
            return self.workspace_layout
        layout = WorkspaceLayout(self.workspace_layout.root, session)
        if not layout.session_workspace_exists():
            raise ValueError(
                f"Unknown session '{session}'. Available sessions: {self.workspace_layout.available_sessions()}"
            )
        return layout

    def _workflow_statuses(self) -> dict[str, dict[str, Any]]:
        return {workflow: self._workflow_status(workflow) for workflow in self.ordered_workflows()}

    def _readiness_from_statuses(
        self,
        statuses: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        model_paths = self.discover_model_paths(limit=5)
        return {
            "train": statuses.get("train", {}).get("ready", False),
            "prediction": statuses.get("prediction", {}).get("ready", False),
            "prediction_requires_models": statuses.get("prediction", {}).get("config_present", False)
            and bool(statuses.get("prediction", {}).get("missing_models")),
            "evaluation": statuses.get("evaluation", {}).get("ready", False),
            "configs_present": {
                workflow: statuses.get(workflow, {}).get("config_present", False)
                for workflow in self.ordered_workflows()
            },
            "available_models": [str(path) for path in model_paths],
            "blocked": {
                workflow: {
                    "missing_paths": statuses.get(workflow, {}).get("missing_paths", []),
                    "missing_models": statuses.get(workflow, {}).get("missing_models", []),
                }
                for workflow in self.ordered_workflows()
            },
        }

    def _next_actions_for_readiness(self, readiness: dict[str, Any]) -> list[str]:
        next_actions: list[str] = []
        configs_present = readiness["configs_present"]
        if configs_present.get("train"):
            next_actions.extend(["review_config_semantics", "validate_config_semantics"])
        if readiness.get("train"):
            next_actions.append("run_train")
        if readiness.get("prediction"):
            next_actions.append("run_prediction")
        if readiness.get("evaluation"):
            next_actions.append("run_evaluation")
        if not all(configs_present.values()):
            next_actions.append("write_workflow_config")
        if any(readiness["blocked"][workflow]["missing_paths"] for workflow in self.ordered_workflows()):
            next_actions.extend(["inspect_dataset", "prepare_dataset_aliases"])
        if not next_actions:
            next_actions = ["browse_dataset", "inspect_dataset", "design_config_strategy", "initialize_session"]
        return list(dict.fromkeys(next_actions))

    def _config_validation_payloads(
        self,
        config_paths: dict[str, Path],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        configs: dict[str, Any] = {}
        semantic_reviews: dict[str, Any] = {}
        for workflow, cfg in config_paths.items():
            result: dict[str, Any] = {"exists": cfg.exists(), "path": str(cfg)}
            if cfg.exists():
                try:
                    YAML_SAFE.load(cfg.read_text(encoding="utf-8"))
                    result["yaml_valid"] = True
                    semantic_reviews[workflow] = self._static_semantic_review(cfg, workflow)
                except Exception as exc:  # pragma: no cover - parser exceptions vary
                    result["yaml_valid"] = False
                    result["error"] = str(exc)
            configs[cfg.name] = result
        return configs, semantic_reviews

    def _supported_extensions(self) -> set[str]:
        from konfai.utils.utils import SUPPORTED_EXTENSIONS

        return {extension.lower() for extension in SUPPORTED_EXTENSIONS}

    def _iter_directories(
        self, root: Path, max_depth: int, max_directories: int
    ) -> tuple[list[tuple[Path, int]], bool]:
        queue: list[tuple[Path, int]] = [(root, 0)]
        visited: list[tuple[Path, int]] = []
        truncated = False

        while queue:
            current, depth = queue.pop(0)
            visited.append((current, depth))
            if len(visited) >= max_directories:
                truncated = bool(queue)
                break
            if depth >= max_depth:
                continue
            children = sorted([path for path in current.iterdir() if path.is_dir()], key=lambda path: path.name)
            for child in children:
                queue.append((child, depth + 1))

        return visited, truncated

    def _h5_group_names(self, path: Path) -> list[str] | None:
        """List the top-level keys of one HDF5 file (KonfAI's real groups), or None without h5py."""
        try:
            import h5py
        except ImportError:
            return None
        try:
            with h5py.File(path, "r") as handle:
                return sorted(handle.keys())
        except Exception:
            return None

    # Files that mark a directory as an OME-Zarr / zarr store root: ``zarr.json`` (Zarr v3) or the
    # Zarr v2 sidecars ``.zgroup`` / ``.zarray`` / ``.zattrs``. A per-case or flat dataset root never
    # carries these at its top level (they live one or two directories down, inside ``<group>.zarr/``),
    # so sniffing them only ever fires when the root itself IS a store.
    _ZARR_STORE_MARKERS = ("zarr.json", ".zgroup", ".zarray", ".zattrs")

    def _zarr_store_extension(self, path: Path, supported_extensions: set[str]) -> str | None:
        """Return the OME-Zarr extension when ``path`` IS itself a zarr store, else None.

        Detects a store handed in as the dataset root by its ``.zarr``-family directory suffix or, for
        a store whose directory was renamed without the suffix, by its zarr store markers. Callers must
        describe such a path as one store rather than walking its multiscale levels as case directories.
        """
        for alias in (".ome_zarr", ".ome-zarr", ".omezarr", ".zarr"):
            if path.name.lower().endswith(alias):
                extension = alias.lstrip(".")
                return extension if extension in supported_extensions else "zarr"
        if any((path / marker).is_file() for marker in self._ZARR_STORE_MARKERS):
            return "zarr" if "zarr" in supported_extensions else None
        return None

    def _classify_directory_entry(self, path: Path, supported_extensions: set[str]) -> tuple[str, str] | None:
        """Classify one sub-directory as a directory-backed dataset entry ``(group, extension)``.

        KonfAI stores some formats as directories (OME-Zarr stores like ``CT.ome.zarr/``, DICOM
        series folders); a file-only scan would make them invisible to every dataset tool.
        Returns None when the directory is not a recognizable data entry (e.g. a nested root).
        """
        compound = full_suffix(path).lstrip(".").lower()
        if compound in supported_extensions:
            return basename_without_suffixes(path), compound
        last = path.suffix.lstrip(".").lower()
        if last in supported_extensions and "zarr" in last:
            return basename_without_suffixes(path), last
        try:
            for child in itertools.islice(path.iterdir(), 256):
                name = child.name.lower()
                if name == "dicomdir" or name.endswith((".dcm", ".dicom")):
                    return path.name, "dicom"
                # PACS exports routinely store DICOM slices with NO extension (e.g. ``IM0001``); a
                # suffix-only check misses them, so sniff the DICOM ``DICM`` magic at byte 128.
                if child.is_file() and self._is_dicom_file(child):
                    return path.name, "dicom"
        except OSError:
            # An unreadable sub-directory must not abort the whole dataset scan.
            return None
        return None

    @staticmethod
    def _is_dicom_file(path: Path) -> bool:
        """True if the file carries the DICOM Part-10 ``DICM`` magic at offset 128 (extensionless slices)."""
        try:
            with path.open("rb") as handle:
                handle.seek(128)
                return handle.read(4) == b"DICM"
        except OSError:
            return False

    def _scan_case_directory(
        self,
        case_dir: Path,
        supported_extensions: set[str],
    ) -> tuple[list[tuple[Path, str, str]], list[str]]:
        """Scan one case directory into ``(path, group, extension)`` entries plus ignored names."""
        entries: list[tuple[Path, str, str]] = []
        ignored: list[str] = []

        for path in sorted(case_dir.iterdir(), key=lambda child: child.name):
            if path.is_dir():
                directory_entry = self._classify_directory_entry(path, supported_extensions)
                if directory_entry is not None:
                    entries.append((path, *directory_entry))
                continue
            if not path.is_file():
                continue
            suffix = full_suffix(path).lstrip(".").lower()
            if suffix not in supported_extensions:
                ignored.append(path.name)
                continue
            if suffix == "h5":
                internal_groups = self._h5_group_names(path)
                if internal_groups:
                    entries.extend((path, group, suffix) for group in internal_groups)
                    continue
            entries.append((path, basename_without_suffixes(path), suffix))
        return entries, ignored

    def _scan_dataset_structure(self, dataset_dir: Path) -> dict[str, Any]:
        supported_extensions = self._supported_extensions()
        # The dataset root can itself BE a single OME-Zarr store (``store.zarr/`` with a root
        # ``zarr.json`` / ``.zgroup`` and multiscale ``scaleN`` levels). Walking it as a case tree
        # mis-reports the scale levels as cases and hides the store; describe it as one store.
        single_store_extension = self._zarr_store_extension(dataset_dir, supported_extensions)
        case_dirs = [dataset_dir] if single_store_extension is not None else case_directories(dataset_dir)
        # A root-level OME-Zarr store handed under a per-case root is a data entry, not a case directory:
        # treat the root as a flat dataset so dataset/CT.ome.zarr + dataset/SEG.mha is reported correctly.
        if (
            single_store_extension is None
            and case_dirs != [dataset_dir]
            and any(path.name.lower().endswith((".zarr", ".omezarr", ".ome-zarr", ".ome_zarr")) for path in case_dirs)
        ):
            case_dirs = [dataset_dir]
        per_case = single_store_extension is None and case_dirs != [dataset_dir]

        groups: dict[str, dict[str, Any]] = {}
        missing_by_case: dict[str, list[str]] = {}
        case_summaries: list[dict[str, Any]] = []
        ignored_files: list[str] = []
        case_group_names: dict[str, set[str]] = {}
        case_file_names: list[set[str]] = []
        all_groups: set[str] = set()

        for case_dir in case_dirs:
            if single_store_extension is not None:
                # The root is one store: its basename is the single group, the store dir is the entry.
                scan_entries = [(dataset_dir, basename_without_suffixes(dataset_dir), single_store_extension)]
                ignored: list[str] = []
            else:
                scan_entries, ignored = self._scan_case_directory(case_dir, supported_extensions)
            file_groups = {group for _, group, _ in scan_entries}
            case_group_names[case_dir.name] = file_groups
            case_file_names.append({path.name for path, _, _ in scan_entries})
            all_groups.update(file_groups)
            ignored_files.extend(str(case_dir / filename) for filename in ignored)
            case_summaries.append(
                {
                    "case": case_dir.name,
                    "files": sorted({path.name for path, _, _ in scan_entries}),
                    "ignored_files": ignored,
                }
            )
            for path, group, extension in scan_entries:
                entry = groups.setdefault(
                    group,
                    {
                        "count": 0,
                        "extensions": set(),
                        "sample_path": str(path),
                    },
                )
                entry["count"] += 1
                if extension:
                    entry["extensions"].add(extension)

        for case_name, file_groups in case_group_names.items():
            missing = sorted(all_groups - file_groups)
            if missing:
                missing_by_case[case_name] = missing

        for entry in groups.values():
            entry["extensions"] = sorted(entry["extensions"])

        detected_extensions = sorted({ext for info in groups.values() for ext in info["extensions"]})
        default_extension = detected_extensions[0] if detected_extensions else None
        common_filenames = sorted(set.intersection(*case_file_names)) if case_file_names else []
        suggested_groups_src = {
            group: {
                "groups_dest": {
                    group: {
                        "transforms": None,
                        "patch_transforms": None,
                        # A group's role is task-dependent and cannot be inferred from its name
                        # (a CT is the input for segmentation but the target for MR->CT synthesis),
                        # so leave it null and let the agent set it from the user's objective —
                        # see is_input_meaning. Guessing here silently mis-wires the config.
                        "is_input": None,
                    }
                }
            }
            for group in sorted(groups)
        }
        dataset_entry = f"{dataset_dir}:a:{default_extension}" if default_extension is not None else None

        structure_warnings: list[str] = []
        if single_store_extension is not None:
            # A bare store is not a KonfAI dataset directory (KonfAI expects <root>/<case>/<group>.<ext>),
            # so it cannot be loaded directly: null the entry and tell the agent how to lay it out.
            dataset_entry = None
            structure_warnings.append(
                f"The path is a single OME-Zarr store ('{dataset_dir.name}'), one image with its multiscale "
                "levels, not a KonfAI dataset root. KonfAI expects a per-case layout "
                "'<root>/<case>/<group>.zarr': move this store inside a case directory (e.g. "
                f"'<root>/<case>/{dataset_dir.name}') and point the dataset root at the directory of cases."
            )

        layout = (
            "single_store"
            if single_store_extension is not None
            else ("per_case_directories" if per_case else "flat_directory")
        )
        return {
            "layout": layout,
            **({"warnings": structure_warnings} if structure_warnings else {}),
            "total_cases": len(case_dirs),
            "groups": groups,
            "case_samples": case_summaries[: min(10, len(case_summaries))],
            "missing_by_case": missing_by_case,
            "ignored_files": ignored_files[:100],
            "detected_extensions": detected_extensions,
            "common_groups": sorted(groups),
            "common_filenames": common_filenames,
            "default_extension": default_extension,
            "dataset_entry": dataset_entry,
            "suggested_groups_src": suggested_groups_src,
            "is_input_meaning": (
                "is_input is each group's ROLE in the model graph: true = an input fed to the network "
                "(the data it sees); false = a target/supervision held out of the input (a segmentation "
                "label, or the volume a synthesis model must produce). It is left null in "
                "suggested_groups_src because it cannot be inferred from the group name — decide it from "
                "the user's task. Examples: CT->segmentation => CT is_input:true, SEG is_input:false; "
                "MR->CT synthesis => MR is_input:true, CT is_input:false. "
                "design_config_strategy(group_roles=...) resolves this with the user."
            ),
        }

    def browse_dataset_payload(
        self,
        dataset_dir: Path,
        depth: int = 2,
        max_entries: int = 200,
        max_candidate_depth: int | None = None,
    ) -> dict[str, Any]:
        if not dataset_dir.exists():
            raise ValueError(f"Dataset directory not found: {dataset_dir}")
        if not dataset_dir.is_dir():
            raise ValueError(f"Dataset path is not a directory: {dataset_dir}")

        depth = max(depth, 0)
        max_entries = max(max_entries, 1)
        candidate_depth = depth if max_candidate_depth is None else max(max_candidate_depth, 0)

        entries: list[dict[str, Any]] = []
        tree: list[str] = []
        truncated = False

        def visit(path: Path, current_depth: int) -> bool:
            nonlocal truncated
            children = sorted(path.iterdir(), key=lambda child: (not child.is_dir(), child.name.lower(), child.name))
            for child in children:
                relative = child.relative_to(dataset_dir)
                display = f"{relative.as_posix()}/" if child.is_dir() else relative.as_posix()
                entries.append(
                    {
                        "path": str(relative.as_posix()),
                        "type": "directory" if child.is_dir() else "file",
                        "depth": current_depth + 1,
                    }
                )
                tree.append(display)
                if len(entries) >= max_entries:
                    truncated = True
                    return True
                # A child sits at ``current_depth + 1``; recurse only while its own children would still
                # fall within ``depth``, so ``depth`` is an inclusive cap on entry depth (depth=1 lists
                # immediate children only, not grandchildren).
                if child.is_dir() and current_depth + 1 < depth:
                    if visit(child, current_depth + 1):
                        return True
            return False

        if depth > 0:
            visit(dataset_dir, 0)

        candidate_roots = self._candidate_dataset_roots(
            dataset_dir,
            max_depth=candidate_depth,
            max_candidates=min(10, max_entries),
        )
        summary_scan = self._scan_dataset_structure(dataset_dir)
        resolved_root = dataset_dir
        if not summary_scan["groups"] and candidate_roots:
            resolved_root = Path(candidate_roots[0]["path"])
            summary_scan = self._scan_dataset_structure(resolved_root)

        return {
            "ok": True,
            "path": str(dataset_dir),
            "requested_path": str(dataset_dir),
            "root": str(resolved_root),
            "root_inferred": resolved_root != dataset_dir,
            "depth": depth,
            "max_entries": max_entries,
            "tree_format": "Relative POSIX-style paths. Directories end with '/'.",
            "tree": tree,
            "entries": entries,
            "truncated": truncated,
            "case_count": summary_scan["total_cases"],
            "common_groups": summary_scan["common_groups"],
            "common_filenames": summary_scan["common_filenames"],
            "extensions": summary_scan["detected_extensions"],
            "ignored_files": summary_scan["ignored_files"],
            "missing_by_case": summary_scan["missing_by_case"],
            **({"warnings": summary_scan["warnings"]} if summary_scan.get("warnings") else {}),
            "candidate_dataset_roots": candidate_roots,
            "next_actions": ["inspect_dataset", "design_config_strategy"],
        }

    def _extract_parenthesized_value(self, text: str, prefix: str) -> str | None:
        start = text.find(prefix)
        if start < 0:
            return None
        index = start + len(prefix)
        depth = 1
        chunk: list[str] = []
        while index < len(text):
            char = text[index]
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    return "".join(chunk)
            chunk.append(char)
            index += 1
        return None

    def _parse_model_metrics(self, body: str | None) -> list[dict[str, Any]]:
        if not body:
            return []
        token_pattern = re.compile(r"([A-Za-z0-9_.-]+)\(([-+0-9.eE]+)\)\s*:\s*")
        value_pattern = re.compile(r"[-+0-9.eE]+|nan|inf|-inf", flags=re.IGNORECASE)
        matches = list(token_pattern.finditer(body))
        models: list[dict[str, Any]] = []
        current_model: dict[str, Any] | None = None
        for index, match in enumerate(matches):
            segment_end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
            trailing = body[match.end() : segment_end].strip()
            if trailing == "" or not value_pattern.fullmatch(trailing):
                current_model = {
                    "name": match.group(1),
                    "lr": float(match.group(2)),
                    "metrics": {},
                }
                models.append(current_model)
                continue
            if current_model is None:
                current_model = {
                    "name": match.group(1),
                    "lr": 0.0,
                    "metrics": {},
                }
                models.append(current_model)
            current_model["metrics"][match.group(1)] = {
                "weight": float(match.group(2)),
                "value": float(trailing),
            }
        return models

    def _flatten_live_metrics(self, models: list[dict[str, Any]]) -> dict[str, float]:
        return {
            f"{model['name']}:{metric_name}": metric_payload["value"]
            for model in models
            for metric_name, metric_payload in model["metrics"].items()
        }

    def _parse_live_metric_line(self, line: str) -> dict[str, Any] | None:
        clean = ANSI_ESCAPE_RE.sub("", line).strip()
        if not clean:
            return None
        stage_match = re.search(r"\b(Training|Validation|Prediction)\s*:", clean)
        if stage_match is None:
            return None
        stage = stage_match.group(1)
        metrics = self._parse_model_metrics(self._extract_parenthesized_value(clean, "Loss ("))
        metrics_ema = self._parse_model_metrics(self._extract_parenthesized_value(clean, "Loss EMA ("))
        if not metrics and not metrics_ema:
            return None
        memory_match = re.search(r"Memory \(([-+0-9.]+)G \(([-+0-9.]+) %\)\)", clean)
        cpu_match = re.search(r"CPU \(([-+0-9.]+) %\)", clean)
        result: dict[str, Any] = {
            "stage": stage,
            "models": metrics,
            "ema_models": metrics_ema,
            "flat_metrics": self._flatten_live_metrics(metrics),
            "flat_metrics_ema": self._flatten_live_metrics(metrics_ema),
            "raw": clean,
        }
        if memory_match is not None:
            result["memory_gb"] = float(memory_match.group(1))
            result["memory_percent"] = float(memory_match.group(2))
        if cpu_match is not None:
            result["cpu_percent"] = float(cpu_match.group(1))
        return result

    def _parse_live_metrics_file(self, path: Path, max_lines: int | None = None) -> list[dict[str, Any]]:
        lines = read_text_tail(path, max_lines=max_lines or self.max_log_tail_lines).splitlines()
        return [entry for line in lines if (entry := self._parse_live_metric_line(line)) is not None]

    def read_live_metrics_payload(self, job: Job, max_entries: int) -> dict[str, Any]:
        runtime_log = self.job_runtime_log_path(job)
        checked_sources: list[str] = []
        parsed_entries: list[dict[str, Any]] = []
        metrics_source: str | None = None

        for path in [runtime_log, job.log_path]:
            if path is None or not path.exists():
                continue
            checked_sources.append(str(path))
            parsed_entries = self._parse_live_metrics_file(
                path, max_lines=max(self.max_log_tail_lines, max_entries * 10)
            )
            if parsed_entries:
                metrics_source = str(path)
                break

        latest_by_stage: dict[str, dict[str, Any]] = {}
        for entry in parsed_entries:
            latest_by_stage[entry["stage"].lower()] = entry
        return {
            **self.job_payload(job),
            "metrics_source": metrics_source,
            "metrics_sources_checked": checked_sources,
            "latest": parsed_entries[-1] if parsed_entries else None,
            "recent": parsed_entries[-max(max_entries, 1) :] if parsed_entries else [],
            "by_stage": latest_by_stage,
        }

    def example_options(self, dataset_groups: list[str]) -> list[dict[str, Any]]:
        options: list[dict[str, Any]] = []
        for template_name in available_templates(self.examples_root):
            configs = load_template_configs(self.examples_root, template_name)
            if not configs:
                continue
            groups_by_workflow = template_groups(configs)
            expected_groups = sorted(
                {
                    group
                    for workflow, groups in groups_by_workflow.items()
                    if workflow in {"train", "prediction"}
                    for group in groups
                }
            )
            suggested_group_map = default_group_map(expected_groups, dataset_groups)
            missing_groups = [group for group in expected_groups if group not in suggested_group_map]
            matched_groups = [group for group in expected_groups if group in suggested_group_map]
            match_ratio = (len(matched_groups) / len(expected_groups)) if expected_groups else 0.0
            options.append(
                {
                    "name": template_name,
                    "expected_groups": expected_groups,
                    "matched_groups": matched_groups,
                    "missing_groups": missing_groups,
                    "compatible": not missing_groups,
                    "match_ratio": round(match_ratio, 4),
                    "summary_resource": f"template://{template_name}/summary",
                }
            )
        options.sort(
            key=lambda item: (
                not item["compatible"],
                -len(item["matched_groups"]),
                len(item["missing_groups"]),
                item["name"],
            )
        )
        return options

    def normalize_dataset_dirs(
        self,
        dataset_dir: Path | None = None,
        dataset_dirs: list[Path] | None = None,
    ) -> list[Path]:
        requested = [*(dataset_dirs or []), *([dataset_dir] if dataset_dir is not None else [])]
        normalized: list[Path] = []
        seen: set[Path] = set()
        for path in requested:
            resolved = path.expanduser().resolve()
            if resolved in seen:
                continue
            if not resolved.exists():
                raise ValueError(f"Dataset directory not found: {resolved}")
            if not resolved.is_dir():
                raise ValueError(f"Dataset path is not a directory: {resolved}")
            seen.add(resolved)
            normalized.append(resolved)
        if not normalized:
            raise ValueError("At least one dataset directory is required.")
        return normalized

    def _strategy_dataset_summary(self, dataset_dirs: list[Path]) -> dict[str, Any]:
        summaries = [self.infer_dataset_structure_payload(dataset_dir) for dataset_dir in dataset_dirs]
        group_presence: dict[str, list[str]] = {}
        for dataset_dir, summary in zip(dataset_dirs, summaries, strict=False):
            for group in sorted(summary["groups"]):
                group_presence.setdefault(group, []).append(str(dataset_dir))

        payload: dict[str, Any] = {
            "count": len(dataset_dirs),
            "paths": [str(path) for path in dataset_dirs],
            "groups": sorted(group_presence),
            "group_presence": group_presence,
            "total_cases": sum(summary["total_cases"] for summary in summaries),
            "detected_extensions": sorted(
                {extension for summary in summaries for extension in summary["detected_extensions"]}
            ),
            "datasets": [
                {
                    "path": str(dataset_dir),
                    "groups": sorted(summary["groups"]),
                    "total_cases": summary["total_cases"],
                    "layout": summary["layout"],
                    "detected_extensions": summary["detected_extensions"],
                    "dataset_entry": summary["dataset_entry"],
                    "candidate_dataset_roots": summary.get("candidate_dataset_roots", []),
                }
                for dataset_dir, summary in zip(dataset_dirs, summaries, strict=False)
            ],
        }
        if len(dataset_dirs) == 1:
            payload["path"] = str(dataset_dirs[0])
            payload["candidate_dataset_roots"] = summaries[0].get("candidate_dataset_roots", [])
            # The same list would otherwise appear twice in one payload.
            payload["datasets"][0].pop("candidate_dataset_roots", None)
        return payload

    def design_config_strategy_payload(
        self,
        dataset_dir: Path | None,
        task: str,
        dataset_dirs: list[Path] | None = None,
        group_roles: dict[str, str] | None = None,
        workflows: list[str] | str | None = None,
        modeling_intent: Literal["2d", "2.5d", "3d", "undecided"] = "undecided",
        example: str | None = None,
        extension: str | None = None,
    ) -> dict[str, Any]:
        if modeling_intent not in {"2d", "2.5d", "3d", "undecided"}:
            raise ValueError("modeling_intent must be one of: 2d, 2.5d, 3d, undecided.")
        task_name = task.strip()
        if not task_name:
            raise ValueError("task is required and must not be empty.")

        normalized_dataset_dirs = self.normalize_dataset_dirs(dataset_dir, dataset_dirs)
        dataset_summary = self._strategy_dataset_summary(normalized_dataset_dirs)
        dataset_groups = dataset_summary["groups"]

        # Resolve the read extension PER root: a multi-dataset design can mix formats (an .mha train set
        # with an .nii.gz eval set), so one global token would mislabel every other root. An explicit
        # `extension` overrides for all; otherwise each root uses its own first-detected extension.
        def _root_extension(detected: list[str]) -> str | None:
            return extension or (detected[0] if detected else None)

        requested_workflows = self.normalize_requested_workflows(workflows, allow_none=True)
        normalized_roles = self.normalize_group_roles(group_roles, dataset_groups)
        unresolved_questions = self.unresolved_strategy_questions(
            dataset_summary,
            task=task_name,
            group_roles=normalized_roles,
            workflows=requested_workflows,
        )

        if example is not None:
            template_dir(self.examples_root, example)
        dataset_ambiguity = (
            dataset_summary["count"] > 1 or len(dataset_summary.get("candidate_dataset_roots") or []) > 1
        )

        config_plan = {
            "workflow_roots": {
                workflow: workflow_root_name(workflow)
                for workflow in (requested_workflows or list(WORKFLOW_CONFIG_FILES))
            },
            "dataset_entries": [
                {"path": str(path), "entry": f"{path}:a:{root_extension}"}
                for path, dataset in zip(normalized_dataset_dirs, dataset_summary["datasets"], strict=False)
                if (root_extension := _root_extension(dataset["detected_extensions"])) is not None
            ],
            "group_roles": normalized_roles,
            "modeling_intent": modeling_intent,
            "patching_considerations": self.strategy_patching_considerations(modeling_intent),
        }

        return {
            "ok": True,
            "task": task_name,
            "dataset_dir": str(normalized_dataset_dirs[0]) if len(normalized_dataset_dirs) == 1 else None,
            "dataset_dirs": [str(path) for path in normalized_dataset_dirs],
            "dataset_summary": dataset_summary,
            "workflows": requested_workflows,
            "group_roles": normalized_roles,
            "modeling_intent": modeling_intent,
            "compatible_examples": self.example_options(dataset_groups),
            "selected_example": (
                {"name": example, "summary_resource": f"template://{example}/summary"} if example is not None else None
            ),
            "config_plan": config_plan,
            "customization_options": {
                "examples_are_optional": True,
                "can_use_builtins": True,
                "can_reference_external_libraries": True,
                "can_write_local_components": True,
                "recommended_local_files": ["Model.py", "Loss.py", "Transform.py", "Augmentation.py"],
                "discovery_tools": ["list_components", "describe_extension_points", "check_external_dependency"],
                "signature_tool": "inspect_object_signature",
                "write_tool": "write_session_file",
                "how_to_customize": [
                    "Prefer a built-in component (list_components) when one fits.",
                    "Otherwise reference an installed library class directly by classpath, e.g. "
                    "`monai.losses:DiceLoss` (vet it with check_external_dependency); no wrapper needed when its "
                    "forward convention already matches the KonfAI base.",
                    "Write a local wrapper (write_session_file) only when the convention differs or you need "
                    "custom logic; see describe_extension_points for the base class and contract.",
                ],
            },
            "unresolved_questions": unresolved_questions,
            "guidance_resources": {
                "overview": "guide://config-design",
                "docs": [
                    "docs://dataset-mapping",
                    "docs://configuration",
                    "docs://patching",
                    "docs://modeling",
                    "docs://examples",
                ],
            },
            "how_to_resolve_questions": (
                "Ask the user the unresolved_questions, then call design_config_strategy again with "
                "group_roles/workflows filled in. Only re-inspect the dataset when a question concerns "
                "ambiguous roots or cohorts."
                if unresolved_questions
                else None
            ),
            "next_actions": (
                ["initialize_session", "write_workflow_config", "inspect_object_signature", "review_config_semantics"]
                if not unresolved_questions
                else (
                    ["browse_dataset", "inspect_dataset", "design_config_strategy"]
                    if dataset_ambiguity
                    else ["design_config_strategy"]
                )
            ),
            "next_resources": ["guide://config-design", "docs://dataset-mapping"] if unresolved_questions else [],
        }

    def _normalize_workflow(self, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in self.workflows:
            raise ValueError(f"Invalid workflow '{value}'. Expected one of {sorted(self.workflows)}")
        return normalized

    def normalize_requested_workflows(
        self,
        values: list[str] | str | None,
        *,
        allow_none: bool = False,
    ) -> list[str]:
        if values is None:
            if allow_none:
                return []
            requested = ["train"]
        elif isinstance(values, str):
            requested = [self._normalize_workflow(values)]
        else:
            requested = [self._normalize_workflow(value) for value in values]
        if not requested:
            raise ValueError("At least one workflow must be requested.")
        unique: list[str] = []
        for workflow in requested:
            if workflow not in unique:
                unique.append(workflow)
        return unique

    def normalize_group_roles(
        self,
        group_roles: dict[str, str] | None,
        dataset_groups: list[str],
    ) -> dict[str, list[str]]:
        roles: dict[str, list[str]] = {"input": [], "target": [], "support": []}
        if group_roles is None:
            return roles
        for group, role in group_roles.items():
            normalized_group = group.strip()
            normalized_role = role.strip().lower()
            if normalized_group not in dataset_groups:
                raise ValueError(f"Unknown dataset group in group_roles: {normalized_group}")
            if normalized_role not in roles:
                raise ValueError(f"Unsupported role '{role}' for group '{group}'. Expected one of {sorted(roles)}")
            if normalized_group not in roles[normalized_role]:
                roles[normalized_role].append(normalized_group)
        for names in roles.values():
            names.sort()
        return roles

    def unresolved_strategy_questions(
        self,
        dataset_info: dict[str, Any],
        *,
        task: str,
        group_roles: dict[str, list[str]],
        workflows: list[str],
    ) -> list[str]:
        questions: list[str] = []
        if not task.strip():
            questions.append("What task does the user want to run on this dataset?")
        if dataset_info.get("count", 1) > 1:
            questions.append(
                "How should the provided datasets be used: merged together, or assigned different roles such as "
                "training, prediction, or evaluation?"
            )
        if len(dataset_info.get("candidate_dataset_roots") or []) > 1:
            questions.append(
                "Should multiple dataset roots or cohorts be merged, or should one be used "
                "for a different role such as evaluation?"
            )
        if not group_roles["input"]:
            questions.append("Which dataset group or groups should be used as model inputs?")
        if not group_roles["target"] and (not workflows or "train" in workflows or "evaluation" in workflows):
            questions.append("Which dataset group or groups should be used as supervision or evaluation targets?")
        if not workflows:
            questions.append("Which workflows are intended now: train, prediction, evaluation, or only a subset?")
        return questions

    def strategy_patching_considerations(self, modeling_intent: str) -> list[str]:
        if modeling_intent == "2d":
            return [
                "A 2D baseline usually uses patch_size [1, x, y].",
                "A 2D baseline usually keeps extend_slice at 0.",
            ]
        if modeling_intent == "2.5d":
            return [
                "A 2.5D setup usually keeps patch_size [1, x, y].",
                "A 2.5D setup usually uses extend_slice > 0 to create neighboring-slice context.",
                "Input channels must still match the model contract once slice context is added.",
            ]
        if modeling_intent == "3d":
            return [
                "A 3D setup usually carries depth in patch_size [z, x, y].",
                "A 3D setup usually keeps extend_slice at 0 because depth is part of the patch itself.",
            ]
        return [
            "Choose 2D for slice-wise baselines, 2.5D for slice-wise context, or 3D for volumetric modeling.",
            "Use docs://patching and docs://modeling to reason about patch_size, extend_slice, and channels.",
        ]

    def _load_workflow_root(self, config_path: Path, workflow: str) -> dict[str, Any]:
        data = YAML_SAFE.load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError(f"{config_path.name} must contain a YAML mapping at the top level.")
        root = data.get(workflow_root_name(workflow))
        if not isinstance(root, dict):
            raise ValueError(f"{config_path.name} must define the '{workflow_root_name(workflow)}' root key.")
        return root

    def _component_signature_for_classpath(self, config_path: Path, classpath: str) -> dict[str, Any] | None:
        if not isinstance(classpath, str) or not classpath.strip():
            return None
        return summarize_classpath_signature(classpath, workspace_dir=config_path.parent)

    def _build_warning(
        self,
        *,
        code: str,
        message: str,
        rationale: str,
        observed: dict[str, Any],
        suggested_next_checks: list[str] | None = None,
    ) -> dict[str, Any]:
        return {
            "code": code,
            "severity": "warning",
            "message": message,
            "rationale": rationale,
            "observed": observed,
            "suggested_next_checks": suggested_next_checks or ["guide://config-design", "docs://index"],
        }

    def _build_blocking_issue(
        self,
        *,
        code: str,
        message: str,
        rationale: str,
        observed: dict[str, Any],
        suggested_next_checks: list[str] | None = None,
    ) -> dict[str, Any]:
        return {
            "code": code,
            "severity": "error",
            "message": message,
            "rationale": rationale,
            "observed": observed,
            "suggested_next_checks": suggested_next_checks or ["guide://config-design", "docs://index"],
        }

    def _collect_specialized_components(self, model: dict[str, Any]) -> list[str]:
        if not isinstance(model, dict):
            return []
        generic_names = {
            "MAE",
            "MSE",
            "RMSE",
            "Dice",
            "DiceCE",
            "CrossEntropy",
            "BCE",
            "PSNR",
            "SSIM",
            "L1",
            "SmoothL1",
        }
        specialized: set[str] = set()
        for model_name, model_payload in model.items():
            if model_name == "classpath" or not isinstance(model_payload, dict):
                continue
            outputs = model_payload.get("outputs_criterions")
            if not isinstance(outputs, dict):
                continue
            for output_payload in outputs.values():
                if not isinstance(output_payload, dict):
                    continue
                targets = output_payload.get("targets_criterions")
                if not isinstance(targets, dict):
                    continue
                for target_payload in targets.values():
                    if not isinstance(target_payload, dict):
                        continue
                    loaders = target_payload.get("criterions_loader")
                    if not isinstance(loaders, dict):
                        continue
                    for criterion_name in loaders:
                        if criterion_name not in generic_names:
                            specialized.add(criterion_name)
        return sorted(specialized)

    def _semantic_strategy_hint(self, model_dim: int | None, patch_depth: int | None, extend_slice: int | None) -> str:
        if model_dim == 3:
            return "3d"
        if model_dim == 2 and isinstance(extend_slice, int) and extend_slice > 0:
            return "2.5d"
        if model_dim == 2:
            return "2d"
        if patch_depth is not None and patch_depth > 1:
            return "3d-like"
        return "unknown"

    def _static_semantic_review(self, config_path: Path, workflow: str) -> dict[str, Any]:
        normalized_workflow = self._normalize_workflow(workflow)
        root = self._load_workflow_root(config_path, normalized_workflow)
        model = root.get("Model", {})
        dataset = root.get("Dataset", {})

        classpath = model.get("classpath") if isinstance(model, dict) else None
        model_name = classpath.split(":")[-1].split(".")[-1] if isinstance(classpath, str) else None
        model_payload = model.get(model_name, {}) if model_name and isinstance(model.get(model_name), dict) else {}
        dataset_patch = dataset.get("Patch", {}) if isinstance(dataset, dict) else {}
        patch_size = dataset_patch.get("patch_size") if isinstance(dataset_patch, dict) else None
        extend_slice = dataset_patch.get("extend_slice") if isinstance(dataset_patch, dict) else None
        patch_depth = None
        if isinstance(patch_size, list) and len(patch_size) >= 3 and isinstance(patch_size[0], (int, float)):
            patch_depth = int(patch_size[0])

        groups_src = dataset.get("groups_src", {}) if isinstance(dataset, dict) else {}
        input_groups: list[str] = []
        non_input_groups: list[str] = []
        if isinstance(groups_src, dict):
            for group_name, group_payload in groups_src.items():
                if not isinstance(group_payload, dict):
                    continue
                groups_dest = group_payload.get("groups_dest", {})
                if not isinstance(groups_dest, dict):
                    continue
                is_input = any(
                    isinstance(dest_payload, dict) and dest_payload.get("is_input") is True
                    for dest_payload in groups_dest.values()
                )
                if is_input:
                    input_groups.append(group_name)
                else:
                    non_input_groups.append(group_name)

        model_dim = model_payload.get("dim") if isinstance(model_payload, dict) else None
        if not isinstance(model_dim, int):
            model_dim = None
        if not isinstance(extend_slice, int):
            extend_slice = 0 if extend_slice in {None, "None"} else extend_slice
        if not isinstance(extend_slice, int):
            extend_slice = None

        local_model = (
            self._component_signature_for_classpath(config_path, classpath) if isinstance(classpath, str) else None
        )
        detected_in_channels = None
        if local_model is not None:
            detected_contract = local_model.get("detected_contract", {})
            if isinstance(detected_contract, dict) and isinstance(detected_contract.get("in_channels"), int):
                detected_in_channels = detected_contract["in_channels"]

        expected_input_channels = None
        if input_groups:
            if model_dim == 2:
                expected_input_channels = len(input_groups) * ((extend_slice or 0) + 1)
            elif model_dim == 3:
                expected_input_channels = len(input_groups)

        warnings: list[dict[str, Any]] = []
        blocking_issues: list[dict[str, Any]] = []
        if model_dim == 2 and patch_depth is not None and patch_depth != 1:
            warnings.append(
                self._build_warning(
                    code="patch_depth_conflicts_with_2d_strategy",
                    message="The config looks 2D-like, but Dataset.Patch.patch_size[0] is not 1.",
                    rationale=(
                        "Slice-wise 2D and 2.5D strategies in KonfAI usually keep patch depth at 1 "
                        "and put context in extend_slice."
                    ),
                    observed={
                        "model_dim": model_dim,
                        "patch_size": patch_size,
                        "extend_slice": extend_slice,
                    },
                )
            )
        if model_dim == 3 and extend_slice not in {None, 0}:
            warnings.append(
                self._build_warning(
                    code="extend_slice_with_3d_strategy",
                    message="extend_slice is set even though the model is configured as 3D.",
                    rationale=(
                        "Volumetric 3D strategies usually express depth through patch_size rather than extend_slice."
                    ),
                    observed={
                        "model_dim": model_dim,
                        "patch_size": patch_size,
                        "extend_slice": extend_slice,
                    },
                )
            )
        if model_dim == 3 and patch_depth is not None and patch_depth <= 1:
            warnings.append(
                self._build_warning(
                    code="patch_depth_suspicious_for_3d_strategy",
                    message="The model is configured as 3D, but the dataset patch depth is 1.",
                    rationale="A 3D strategy usually needs patch_size[0] > 1 to carry volumetric depth.",
                    observed={
                        "model_dim": model_dim,
                        "patch_size": patch_size,
                        "extend_slice": extend_slice,
                    },
                )
            )
        if (
            isinstance(detected_in_channels, int)
            and isinstance(expected_input_channels, int)
            and detected_in_channels != expected_input_channels
        ):
            warnings.append(
                self._build_warning(
                    code="input_channel_context_mismatch",
                    message="The local model input-channel contract does not match the dataset pipeline context.",
                    rationale=(
                        "For 2D-like pipelines, effective input channels often depend on the "
                        "number of input groups and extend_slice. A mismatch can indicate "
                        "missing slice context or the wrong modality count."
                    ),
                    observed={
                        "model_classpath": classpath,
                        "detected_model_in_channels": detected_in_channels,
                        "input_groups": sorted(input_groups),
                        "expected_input_channels": expected_input_channels,
                        "extend_slice": extend_slice,
                    },
                    suggested_next_checks=[
                        "guide://config-design",
                        "docs://patching",
                        "template://Synthesis/summary",
                    ],
                )
            )
        if isinstance(local_model, dict) and local_model.get("source") == "local" and not local_model.get("ok", False):
            blocking_issues.append(
                self._build_blocking_issue(
                    code="missing_local_model_source",
                    message="The config references a local model classpath, but the local source file is unavailable.",
                    rationale=(
                        "Local classpaths such as Model:UNetpp5 require the corresponding Python file "
                        "next to the config. Runtime validation will fail until that file is present."
                    ),
                    observed={
                        "classpath": classpath,
                        "source_path": local_model.get("source_path"),
                        "warning": (local_model.get("summary") or {}).get("warning"),
                    },
                    suggested_next_checks=["inspect_object_signature", "write_session_file", "docs://modeling"],
                )
            )
        specialized_components = self._collect_specialized_components(model)
        if specialized_components:
            warnings.append(
                self._build_warning(
                    code="specialized_components_require_review",
                    message="The config references specialized components that should be reviewed before a first run.",
                    rationale=(
                        "Specialized losses, metrics, or transforms copied from an example can depend on "
                        "extra support files or assumptions that are not obvious from the YAML alone."
                    ),
                    observed={"components": specialized_components},
                    suggested_next_checks=["docs://examples", "template://Synthesis/summary", "guide://config-design"],
                )
            )
        if normalized_workflow == "train" and not non_input_groups:
            warnings.append(
                self._build_warning(
                    code="no_non_input_groups_declared",
                    message="The training dataset mapping does not expose any non-input groups.",
                    rationale=(
                        "Training usually needs at least one target or support-only group for "
                        "losses, metrics, or masking."
                    ),
                    observed={"groups_src": sorted(groups_src) if isinstance(groups_src, dict) else []},
                )
            )

        local_metadata = None
        if local_model is not None:
            # Only the decision-relevant subset: the full parameter/default dump stays behind
            # inspect_object_signature instead of being re-embedded in every review payload.
            local_metadata = {
                key: local_model.get(key)
                for key in ("ok", "source", "source_path", "classpath", "signature", "doc_summary", "detected_contract")
            }
            summary_warning = (local_model.get("summary") or {}).get("warning")
            if summary_warning:
                local_metadata["warning"] = summary_warning

        strategy_hint = self._semantic_strategy_hint(model_dim, patch_depth, extend_slice)
        next_checks = ["guide://config-design", "docs://index"]
        if isinstance(classpath, str) and classpath.startswith("Model:"):
            next_checks.append("inspect_object_signature")

        return {
            "workflow": normalized_workflow,
            "config_path": str(config_path),
            "strategy_hint": strategy_hint,
            "summary": {
                "model": {
                    "classpath": classpath,
                    "model_name": model_name,
                    "dim": model_dim,
                    "local_metadata": local_metadata,
                },
                "dataset": {
                    "groups": sorted(groups_src.keys()) if isinstance(groups_src, dict) else [],
                    "input_groups": sorted(input_groups),
                    "non_input_groups": sorted(non_input_groups),
                    "patch_size": patch_size,
                    "extend_slice": extend_slice,
                    "dataset_filenames": self._configured_dataset_entries(normalized_workflow, config_path),
                },
            },
            "warnings": warnings,
            "blocking_issues": blocking_issues,
            "next_checks": next_checks,
        }

    def review_config_semantics(self, workflow: Literal["train", "prediction", "evaluation"]) -> dict[str, Any]:
        normalized_workflow = self._normalize_workflow(workflow)
        config_path = self.config_path(normalized_workflow)
        if not config_path.exists():
            raise ValueError(f"Missing config file for workflow '{normalized_workflow}': {config_path.name}")
        payload = self._static_semantic_review(config_path, normalized_workflow)
        payload["session"] = self.session_name()
        payload["ok"] = not payload["blocking_issues"]
        payload["next_actions"] = (
            ["write_session_file", "write_workflow_config", "inspect_object_signature", "summarize_session"]
            if payload["blocking_issues"]
            else ["validate_config_semantics", "summarize_session"]
        )
        return payload

    def validate_semantics(
        self,
        workflow: str,
        level: Literal["instantiate", "setup", "train_step"],
        models: list[str] | None = None,
        config_path: Path | None = None,
        collect_model_outputs: bool = False,
    ) -> dict[str, Any]:
        normalized_workflow = self._normalize_workflow(workflow)
        if level not in self.validation_levels:
            raise ValueError(f"Invalid validation level '{level}'. Expected one of {sorted(self.validation_levels)}")

        workspace = self.workspace_layout.ensure_session_workspace_exists()
        config_path = config_path or self.config_path(normalized_workflow)
        if not config_path.exists():
            raise ValueError(f"Missing config file for workflow '{normalized_workflow}': {config_path.name}")
        semantic_review = self._static_semantic_review(config_path, normalized_workflow)
        if semantic_review["blocking_issues"]:
            return {
                "ok": False,
                "blocked": True,
                "workflow": normalized_workflow,
                "config_path": str(config_path),
                "error_type": "SemanticIssue",
                "error": "Semantic review found blocking issues that must be fixed before runtime validation.",
                "blocking_issues": semantic_review["blocking_issues"],
                "semantic_review": semantic_review,
                "next_actions": [
                    "write_session_file",
                    "write_workflow_config",
                    "inspect_object_signature",
                    "summarize_session",
                ],
            }

        resolved_models = (
            [
                str(path)
                for path in self.resolve_prediction_models(
                    models, limit=1, run_name=self.configured_run_name("prediction", config_path)
                )
            ]
            if normalized_workflow == "prediction"
            else []
        )

        blocked = self.workflow_blocker(normalized_workflow, resolved_models or None, config_path=config_path)
        if blocked is not None:
            blocked["level"] = level
            blocked["semantic_review"] = semantic_review
            return blocked

        # The child restores the authored config in its own finally, but a timeout kills it with SIGTERM
        # and that finally never runs. Snapshot in the parent (which outlives the child) and restore
        # unconditionally, so a slow-model timeout can never leave the agent's config KonfAI-rewritten.
        config_backup = config_path.read_text(encoding="utf-8") if config_path.is_file() else None
        try:
            with tempfile.TemporaryDirectory(prefix=f"konfai_mcp_validate_{normalized_workflow}_") as tmp_dir:
                # Isolated spawn child: agent-authored code never executes in the server process and
                # edited workspace modules are always re-imported fresh.
                payload = mcp_runner.run_api_in_subprocess(
                    "konfai_mcp.runner:validate_workflow_api",
                    {
                        "workflow": normalized_workflow,
                        "level": level,
                        "workspace_dir": str(workspace),
                        "config": str(config_path),
                        "models": resolved_models or None,
                        "validate_root": tmp_dir,
                        "collect_model_outputs": collect_model_outputs,
                    },
                )
        finally:
            if config_backup is not None and config_path.is_file():
                if config_path.read_text(encoding="utf-8") != config_backup:
                    config_path.write_text(config_backup, encoding="utf-8")
        payload["returncode"] = 0 if payload.get("ok", False) else 1
        payload["semantic_review"] = semantic_review
        return payload

    def _configured_dataset_entries(self, workflow: str, config_path: Path) -> list[str]:
        data = YAML_SAFE.load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            return []
        root = data.get(workflow_root_name(workflow))
        if not isinstance(root, dict):
            return []
        dataset = root.get("Dataset")
        if not isinstance(dataset, dict):
            return []
        entries = dataset.get("dataset_filenames")
        if not isinstance(entries, list):
            return []
        return [entry for entry in entries if isinstance(entry, str) and entry not in {"", "None"}]

    def _resolve_dataset_entry_path(self, dataset_entry: str) -> Path:
        filename, _, _ = split_path_spec(dataset_entry, allowed_flags={"a", "i"})
        path = Path(filename).expanduser()
        if not path.is_absolute():
            path = (self.workspace_dir() / path).resolve()
        return path

    def configured_run_name(self, kind: Literal["train", "prediction", "evaluation"], config_path: Path) -> str | None:
        if not config_path.exists():
            return None
        data = YAML_SAFE.load(config_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None
        root_key = {
            "train": "Trainer",
            "prediction": "Predictor",
            "evaluation": "Evaluator",
        }[kind]
        root = data.get(root_key)
        if not isinstance(root, dict):
            return None
        value = root.get("train_name")
        return value if isinstance(value, str) and value not in {"", "None"} else None

    def runtime_log_path_for(
        self,
        kind: Literal["train", "prediction", "evaluation"],
        config_path: Path,
    ) -> Path | None:
        run_name = self.configured_run_name(kind, config_path)
        if run_name is None:
            return None
        return self._workflow_runtime_root(kind) / run_name / "log_0.txt"

    def _missing_dataset_paths(self, workflow: str, config_path: Path) -> list[str]:
        return [
            str(self._resolve_dataset_entry_path(entry))
            for entry in self._configured_dataset_entries(workflow, config_path)
            if not self._resolve_dataset_entry_path(entry).exists()
        ]

    def _workflow_status(
        self, workflow: str, models: list[str] | None = None, config_path: Path | None = None
    ) -> dict[str, Any]:
        normalized_workflow = self._normalize_workflow(workflow)
        config_path = config_path or self.config_path(normalized_workflow)
        if not config_path.exists():
            return {
                "config_present": False,
                "ready": False,
                "config_path": str(config_path),
                "workflow": normalized_workflow,
                "missing_paths": [],
                "missing_models": [],
            }

        missing_paths = self._missing_dataset_paths(normalized_workflow, config_path)
        missing_models: list[str] = []
        if normalized_workflow == "prediction":
            resolved_models = models or [str(path) for path in self.resolve_prediction_models(limit=5)]
            missing_models = (
                ["<checkpoint>"]
                if not resolved_models
                else [model for model in resolved_models if not Path(model).exists()]
            )

        return {
            "config_present": True,
            "ready": not missing_paths and not missing_models,
            "config_path": str(config_path),
            "workflow": normalized_workflow,
            "missing_paths": missing_paths,
            "missing_models": missing_models,
        }

    def workflow_blocker(
        self,
        workflow: str,
        models: list[str] | None = None,
        config_path: Path | None = None,
    ) -> dict[str, Any] | None:
        status = self._workflow_status(workflow, models, config_path=config_path)
        if not status["config_present"] or status["ready"]:
            return None
        if status["missing_paths"]:
            return {
                "ok": False,
                "blocked": True,
                "workflow": status["workflow"],
                "config_path": status["config_path"],
                "error_type": "MissingArtifact",
                "error": "Required dataset artifact(s) are missing for this workflow.",
                "missing_paths": status["missing_paths"],
                "next_actions": ["inspect_dataset", "prepare_dataset_aliases", "summarize_session"],
            }
        return {
            "ok": False,
            "blocked": True,
            "workflow": status["workflow"],
            "config_path": status["config_path"],
            "error_type": "MissingArtifact",
            "error": "Prediction requires at least one checkpoint artifact.",
            "missing_paths": status["missing_models"],
            "next_actions": ["run_train", "summarize_session"],
        }

    def _candidate_dataset_roots(
        self, dataset_dir: Path, max_depth: int = 2, max_candidates: int = 10
    ) -> list[dict[str, Any]]:
        directories, truncated = self._iter_directories(dataset_dir, max_depth=max_depth, max_directories=64)
        candidates: list[dict[str, Any]] = []
        accepted_roots: list[Path] = []

        for candidate_dir, depth in directories:
            if any(parent == candidate_dir or parent in candidate_dir.parents for parent in accepted_roots):
                continue
            scan = self._scan_dataset_structure(candidate_dir)
            if not scan["groups"]:
                continue
            relative_path = "." if candidate_dir == dataset_dir else candidate_dir.relative_to(dataset_dir).as_posix()
            candidates.append(
                {
                    "path": str(candidate_dir),
                    "relative_path": relative_path,
                    "depth": depth,
                    "layout": scan["layout"],
                    "total_cases": scan["total_cases"],
                    "groups": sorted(scan["groups"]),
                    "detected_extensions": scan["detected_extensions"],
                }
            )
            accepted_roots.append(candidate_dir)

        candidates.sort(
            key=lambda item: (
                -item["total_cases"],
                -len(item["groups"]),
                item["depth"],
                item["path"],
            )
        )
        result = candidates[:max_candidates]
        if truncated and result:
            result[0] = {
                **result[0],
                "discovery_truncated": True,
            }
        return result

    def _infer_dataset_structure_payload(self, dataset_dir: Path, *, discover_candidates: bool) -> dict[str, Any]:
        if not dataset_dir.exists():
            raise ValueError(f"Dataset directory not found: {dataset_dir}")
        scan = self._scan_dataset_structure(dataset_dir)
        payload = {
            "ok": True,
            "path": str(dataset_dir),
            "layout": scan["layout"],
            "total_cases": scan["total_cases"],
            "groups": scan["groups"],
            "case_samples": scan["case_samples"],
            "missing_by_case": scan["missing_by_case"],
            "ignored_files": scan["ignored_files"],
            "detected_extensions": scan["detected_extensions"],
            "dataset_entry": scan["dataset_entry"],
            "suggested_groups_src": scan["suggested_groups_src"],
            "is_input_meaning": scan["is_input_meaning"],
            **({"warnings": scan["warnings"]} if scan.get("warnings") else {}),
        }
        if discover_candidates:
            candidate_roots = self._candidate_dataset_roots(dataset_dir)
            payload["candidate_dataset_roots"] = candidate_roots
            if not payload["groups"] and candidate_roots:
                payload["warnings"] = [
                    *(payload.get("warnings") or []),
                    "No supported groups were found directly under the requested path. "
                    "Inspect candidate_dataset_roots or call browse_dataset to locate the actual dataset root.",
                ]
            payload["next_actions"] = [
                "browse_dataset",
                "inspect_dataset",
                "design_config_strategy",
                "initialize_session",
            ]
        return payload

    def infer_dataset_structure_payload(self, dataset_dir: Path) -> dict[str, Any]:
        return self._infer_dataset_structure_payload(dataset_dir, discover_candidates=True)

    def _load_dataset(self, dataset_dir: Path, extension: str) -> Any:
        from konfai.utils.dataset import Dataset
        from konfai.utils.utils import SUPPORTED_EXTENSIONS

        if extension not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported dataset extension '{extension}'.")
        if not dataset_dir.exists():
            raise ValueError(f"Dataset directory not found: {dataset_dir}")
        return Dataset(str(dataset_dir), extension)

    def _sample_dataset_names(self, names: list[str], max_cases: int | None, seed: int) -> list[str]:
        if max_cases is None or max_cases <= 0 or len(names) <= max_cases:
            return sorted(names)
        rng = random.Random(seed)
        return sorted(rng.sample(names, max_cases))

    def compute_dataset_group_statistics(
        self,
        dataset_dir: Path,
        group: str,
        extension: str = "mha",
        max_cases: int | None = None,
        seed: int = 0,
    ) -> dict[str, Any]:
        dataset = self._load_dataset(dataset_dir, extension)
        names = sorted(dataset.get_names(group))
        if not names:
            raise ValueError(f"Group '{group}' not found in dataset '{dataset_dir}'.")

        sampled_names = self._sample_dataset_names(names, max_cases, seed)
        sampled_items: dict[str, dict[str, Any]] = {}
        per_case_labels: dict[str, dict[str, float]] = {}
        # Cap separating a label map (a bounded class set -- whole-body atlases like TotalSegmentator
        # top out around 117 classes) from an intensity image stored as an integer (int16 CT -> thousands
        # of distinct values, for which per-label voxel fractions are meaningless and huge).
        label_stats_max_classes = 512
        high_cardinality_labels: dict[str, int] = {}
        unreadable_cases: dict[str, str] = {}
        for name in sampled_names:
            try:
                data, attr = dataset.read_data(group, name)
            except Exception as exc:
                # A single bad file must not abort the whole group.
                # The structure scan already found this file on disk, so a read failure means the file
                # itself is unreadable (corrupt, truncated, empty, or non-image bytes) -- a different
                # problem from a missing group or a layout/token mismatch. Record it per case and keep
                # going so one bad file never hides the healthy cases' statistics.
                unreadable_cases[name] = f"{type(exc).__name__}: {str(exc).strip() or 'read failed'}"
                continue
            p25, p50, p75 = np.percentile(data, (25, 50, 75))
            sampled_items[name] = {
                "min": float(data.min()),
                "max": float(data.max()),
                "mean": float(data.mean()),
                "std": float(data.std()),
                "25pc": float(p25),
                "50pc": float(p50),
                "75pc": float(p75),
                "shape": list(data.shape),
                "spacing": attr.get_np_array("Spacing").tolist(),
            }
            # Segmentation-style groups: expose label ids and per-label voxel fractions so
            # nb_class and class weights stop being guesses.
            if np.issubdtype(data.dtype, np.integer):
                unique, counts = np.unique(data, return_counts=True)
                if unique.size <= label_stats_max_classes:
                    per_case_labels[name] = {
                        str(int(label)): round(float(count) / float(data.size), 6)
                        for label, count in zip(unique, counts, strict=False)
                    }
                else:
                    # Above the cap: record the cardinality instead of silently dropping the info, so the
                    # agent learns this is an intensity-like integer group, not a many-class label map.
                    high_cardinality_labels[name] = int(unique.size)
        if unreadable_cases and not sampled_items:
            # Every sampled case exists but failed to read: a corrupt/unreadable-FILE problem, not a
            # missing group or a layout/token mismatch. Raise a distinct error so the caller's reason
            # builder tells the agent to inspect/replace the files, not restructure the dataset.
            raise DatasetGroupUnreadableError(group, extension, unreadable_cases)
        statistics = aggregate_case_statistics(sampled_items)

        payload: dict[str, Any] = {
            "group": group,
            "extension": extension,
            "dataset_path": str(dataset_dir),
            "total_cases": len(names),
            "sampled_cases": len(sampled_names),
            "readable_cases": len(sampled_items),
            "sample_names": sampled_names[: min(20, len(sampled_names))],
            "sampled": len(sampled_names) != len(names),
            "statistics": statistics,
        }
        if per_case_labels:
            all_labels = sorted({label for labels in per_case_labels.values() for label in labels}, key=int)
            payload["labels"] = {
                "unique": [int(label) for label in all_labels],
                "count": len(all_labels),
                "presence_cases": {
                    label: sum(1 for labels in per_case_labels.values() if label in labels) for label in all_labels
                },
                "mean_voxel_fraction": {
                    label: round(
                        sum(labels.get(label, 0.0) for labels in per_case_labels.values()) / len(per_case_labels), 6
                    )
                    for label in all_labels
                },
                "per_case": per_case_labels,
            }
        if high_cardinality_labels:
            # Never silently omit the label section: state that the integer group exceeds the label cap
            # (so it reads as an intensity image), with the observed distinct-value counts.
            payload["high_cardinality_integer_group"] = {
                "max_classes": label_stats_max_classes,
                "note": (
                    "integer group with more distinct values than the label cap; treated as intensity-like, "
                    "not a label map, so per-label voxel fractions are omitted"
                ),
                "distinct_values_per_case": high_cardinality_labels,
            }
        if unreadable_cases:
            # Partial corruption: some cases read, some did not. Surface the bad ones as a distinct
            # per-case signal (with the real reader error) so the agent fixes those files rather than
            # mistrusting the whole group's statistics.
            payload["unreadable_cases"] = {
                "count": len(unreadable_cases),
                "note": (
                    "these cases exist on disk but the reader raised (corrupt/truncated/empty/non-image); "
                    "statistics above are computed only from the readable cases"
                ),
                "errors": unreadable_cases,
            }
        return payload

    def _common_metric_prefix(self, values: list[str]) -> str | None:
        if not values:
            return None
        prefix = values[0]
        for value in values[1:]:
            while prefix and not value.startswith(prefix):
                prefix = prefix[:-1]
        return prefix.rstrip(":") or None

    def _metric_direction(self, metric_name: str, declared: str | None = None) -> tuple[Literal["min", "max"], str]:
        # A direction declared by the criterion itself (via the evaluation JSON 'directions' block,
        # sourced from each Criterion's `maximize` property) is authoritative -- no guessing.
        if declared in ("min", "max"):
            return declared, "declared"  # type: ignore[return-value]
        lowered = metric_name.lower()
        maximize_tokens = ("dice", "iou", "accuracy", "acc", "auc", "f1", "ssim", "psnr")
        minimize_tokens = ("mae", "mse", "rmse", "loss", "hausdorff", "hd")
        # 'loss' wins over any maximize token: a criterion named DiceLoss is still minimized.
        if "loss" in lowered:
            return "min", "heuristic:min"
        if any(token in lowered for token in maximize_tokens):
            return "max", "heuristic:max"
        if any(token in lowered for token in minimize_tokens):
            return "min", "heuristic:min"
        return "min", "default:min"

    @staticmethod
    def _declared_directions(payload: dict[str, Any]) -> dict[str, str]:
        directions = payload.get("directions", {})
        return directions if isinstance(directions, dict) else {}

    def _extract_metric_scoreboards(self, metrics_path: Path) -> dict[str, dict[str, Any]]:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        aggregates = payload.get("aggregates", {})
        if not isinstance(aggregates, dict):
            return {}
        declared = self._declared_directions(payload)

        result: dict[str, dict[str, Any]] = {}
        for metric_name, stats in aggregates.items():
            if isinstance(stats, dict) and isinstance(stats.get("mean"), (int, float)):
                direction, direction_source = self._metric_direction(metric_name, declared.get(metric_name))
                result[metric_name] = {
                    "metric": metric_name,
                    "value": float(stats["mean"]),
                    "direction": direction,
                    "direction_source": direction_source,
                    "stats": stats,
                }
            elif isinstance(stats, (int, float)):
                direction, direction_source = self._metric_direction(metric_name, declared.get(metric_name))
                result[metric_name] = {
                    "metric": metric_name,
                    "value": float(stats),
                    "direction": direction,
                    "direction_source": direction_source,
                    "stats": {"value": float(stats)},
                }
        return result

    def _evaluation_metric_files(self, split: str, layout: WorkspaceLayout | None = None) -> list[Path]:
        split_name = split.upper()
        candidates: list[Path] = []
        for root in self._metric_search_roots(layout):
            if not root.exists():
                continue
            candidates.extend(root.rglob(f"Metric_{split_name}.json"))
            candidates.extend(root.rglob(f"METRIC_{split_name}.json"))
        return sorted(set(candidates), key=lambda path: path.stat().st_mtime, reverse=True)

    def _available_metric_splits(self, layout: WorkspaceLayout | None = None) -> list[str]:
        splits: set[str] = set()
        for root in self._metric_search_roots(layout):
            if not root.exists():
                continue
            for pattern in ("Metric_*.json", "METRIC_*.json"):
                for path in root.rglob(pattern):
                    splits.add(path.stem.split("_", 1)[1])
        return sorted(splits)

    def run_metrics_payload(self, run_name: str, split: str = "TRAIN", session: str | None = None) -> dict[str, Any]:
        """Return the full per-case + aggregate metric JSON for ONE named run, not just the newest file."""
        layout = self._resolve_session_layout(session)
        split_name = split.upper()
        # Canonical identity first (the labels leaderboard hands out), THEN the inner-dir alias: in one
        # merged pass a newer trial's alias could shadow a plain run bearing that exact name, and a
        # leaderboard row would not round-trip through get_run_metrics.
        candidates = self._evaluation_metric_files(split_name, layout)
        metrics_path = next(
            (path for path in candidates if self._metric_run_name(path, layout) == run_name),
            None,
        ) or next((path for path in candidates if path.parent.name == run_name), None)
        if metrics_path is None:
            available_runs = sorted(
                {
                    self._metric_run_name(path, layout)
                    for root in self._metric_search_roots(layout)
                    if root.exists()
                    for pattern in ("Metric_*.json", "METRIC_*.json")
                    for path in root.rglob(pattern)
                }
            )
            raise ValueError(
                f"No metrics found for run '{run_name}' and split '{split_name}' in session "
                f"'{layout.current_session}'. Available runs: {available_runs or 'none'}. "
                f"Available splits: {self._available_metric_splits(layout) or 'none'}."
            )
        return {
            "session": layout.current_session,
            "run_name": run_name,
            "split": split_name,
            "path": str(metrics_path),
            "updated_at": self._isoformat(metrics_path.stat().st_mtime),
            "metrics": json.loads(metrics_path.read_text(encoding="utf-8")),
            "summary": self._extract_metric_scoreboards(metrics_path),
            "next_actions": ["leaderboard", "summarize_session"],
        }

    def leaderboard_payload(
        self,
        metric: str | None = None,
        split: str = "TRAIN",
        limit: int = 20,
        direction: Literal["min", "max"] | None = None,
        session: str | None = None,
    ) -> dict[str, Any]:
        layout = self._resolve_session_layout(session)
        metric_files = self._evaluation_metric_files(split, layout)
        if not metric_files:
            available_splits = self._available_metric_splits(layout)
            raise ValueError(
                f"No evaluation metrics found for session '{layout.current_session}' and split '{split.upper()}'. "
                f"Available splits: {available_splits or 'none'}."
            )

        by_metric: dict[str, list[dict[str, Any]]] = {}
        available_metrics: set[str] = set()
        warnings: list[str] = []

        for metrics_path in metric_files:
            run_name = self._metric_run_name(metrics_path, layout)
            extracted = self._extract_metric_scoreboards(metrics_path)
            available_metrics.update(extracted)
            updated_at = self._isoformat(metrics_path.stat().st_mtime)
            for metric_name, payload in extracted.items():
                by_metric.setdefault(metric_name, []).append(
                    {
                        "run_name": run_name,
                        "metric": metric_name,
                        "value": payload["value"],
                        "direction": payload["direction"],
                        "direction_source": payload["direction_source"],
                        "stats": payload["stats"],
                        "metrics_path": str(metrics_path),
                        "updated_at": updated_at,
                    }
                )
                if payload["direction_source"] == "default:min":
                    warnings.append(
                        f"Metric '{metric_name}' uses the default minimize direction because its name is unknown."
                    )

        selected_metric = metric
        if selected_metric is not None and metric is not None:
            lowered_metric = metric.lower()
            matching_metrics = [
                name
                for name in available_metrics
                if name.lower() == lowered_metric or name.lower().endswith(lowered_metric)
            ]
            if len(matching_metrics) == 1:
                selected_metric = matching_metrics[0]
            elif metric not in by_metric:
                raise ValueError(f"Metric '{metric}' not found. Available metrics: {sorted(available_metrics)}")
        elif len(available_metrics) == 1:
            selected_metric = next(iter(available_metrics))

        for metric_name, rows in by_metric.items():
            row_direction = rows[0]["direction"]
            # An explicit override applies to the selected metric (or to all when none is selected).
            if direction is not None and (selected_metric is None or metric_name == selected_metric):
                row_direction = direction
            rows.sort(key=lambda row: row["value"], reverse=row_direction == "max")
            by_metric[metric_name] = rows[: max(limit, 1)]

        result: dict[str, Any] = {
            "session": layout.current_session,
            "split": split.upper(),
            "available_metrics": sorted(available_metrics),
            "available_splits": self._available_metric_splits(layout),
            "warnings": sorted(set(warnings)),
            "next_actions": ["summarize_session", "get_run_metrics", "run_train", "run_prediction", "run_evaluation"],
        }
        if direction is not None:
            result["direction_override"] = direction
        if selected_metric is not None:
            result["selected_metric"] = selected_metric
            result["leaderboard"] = by_metric[selected_metric]
            result["best"] = by_metric[selected_metric][0] if by_metric[selected_metric] else None
        else:
            result["leaderboard"] = []
            result["common_metric_prefix"] = self._common_metric_prefix(sorted(available_metrics))
            result["leaderboards"] = by_metric
        return result

    def read_metrics_payload(self) -> dict[str, Any]:
        metrics_path = self.discover_metrics_path()
        if metrics_path is None:
            return {
                "session": self.session_name(),
                "path": None,
                "metrics": None,
                "summary": None,
            }
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        summary = self._extract_metric_scoreboards(metrics_path)
        return {
            "session": self.session_name(),
            "path": str(metrics_path),
            "metrics": payload,
            "summary": summary,
        }

    def discover_metrics_path(self) -> Path | None:
        candidates: list[Path] = []
        for root in self._metric_search_roots():
            if not root.exists():
                continue
            candidates.extend(root.rglob("Metric_*.json"))
            candidates.extend(root.rglob("METRIC_*.json"))
        if not candidates:
            return None
        return max(candidates, key=lambda path: path.stat().st_mtime)

    def training_curves_payload(
        self,
        run_name: str,
        tags: list[str] | None = None,
        max_points: int = 200,
        session: str | None = None,
    ) -> dict[str, Any]:
        """Parse the TensorBoard event files of one run into downsampled scalar series."""
        layout = self._resolve_session_layout(session)
        stats_dir = layout.workspace_dir() / "Statistics" / run_name
        if not stats_dir.exists():
            available = (
                sorted(path.name for path in (layout.workspace_dir() / "Statistics").iterdir() if path.is_dir())
                if (layout.workspace_dir() / "Statistics").exists()
                else []
            )
            raise ValueError(f"No Statistics/{run_name} directory. Available runs: {available or 'none'}.")
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        except ImportError as exc:
            raise ValueError(
                "Reading training curves requires the 'tensorboard' package: pip install konfai[tensorboard]."
            ) from exc

        event_dirs = sorted({path.parent for path in stats_dir.rglob("events.out.tfevents.*")})
        if not event_dirs:
            raise ValueError(f"No TensorBoard event files found under {stats_dir}.")
        curves: dict[str, list[dict[str, float]]] = {}
        for event_dir in event_dirs:
            accumulator = EventAccumulator(str(event_dir), size_guidance={"scalars": 0})
            accumulator.Reload()
            prefix = event_dir.relative_to(stats_dir).as_posix()
            for tag in accumulator.Tags().get("scalars", []):
                full_tag = tag if prefix == "." else f"{prefix}/{tag}"
                if tags and not any(wanted in full_tag for wanted in tags):
                    continue
                events = accumulator.Scalars(tag)
                stride = max(1, len(events) // max(max_points, 1))
                sampled = events[::stride]
                if events and sampled[-1].step != events[-1].step:
                    sampled = [*sampled, events[-1]]
                curves[full_tag] = [{"step": int(event.step), "value": float(event.value)} for event in sampled]
        return {
            "session": layout.current_session,
            "run_name": run_name,
            "tags": sorted(curves),
            "curves": curves,
            "next_actions": ["compare_runs", "leaderboard", "summarize_session"],
        }

    def compare_runs_payload(
        self,
        run_a: str,
        run_b: str,
        split: str = "TRAIN",
        metric: str | None = None,
        session: str | None = None,
    ) -> dict[str, Any]:
        """Per-case aligned comparison of two runs' evaluation metrics, direction-aware."""
        payload_a = self.run_metrics_payload(run_a, split=split, session=session)
        payload_b = self.run_metrics_payload(run_b, split=split, session=session)
        cases_a = payload_a["metrics"].get("case", {})
        cases_b = payload_b["metrics"].get("case", {})
        common_metrics = sorted(set(cases_a) & set(cases_b))
        if metric is not None:
            common_metrics = [name for name in common_metrics if name.lower().endswith(metric.lower())]
            if not common_metrics:
                raise ValueError(
                    f"Metric '{metric}' not found in both runs. Common metrics: {sorted(set(cases_a) & set(cases_b))}"
                )
        declared_a = self._declared_directions(payload_a["metrics"])
        declared_b = self._declared_directions(payload_b["metrics"])
        comparison: dict[str, Any] = {}
        warnings: list[str] = []
        for name in common_metrics:
            direction, direction_source = self._metric_direction(name, declared_a.get(name) or declared_b.get(name))
            if direction_source == "default:min":
                warnings.append(
                    f"No declared or recognised direction for '{name}'; assumed minimize. If it is a "
                    "higher-is-better metric, the winner is inverted -- re-evaluate so the metric declares "
                    "its direction, or read per-case deltas directly."
                )
            common_cases = sorted(set(cases_a[name]) & set(cases_b[name]))
            deltas = {case: float(cases_b[name][case]) - float(cases_a[name][case]) for case in common_cases}
            better_b = sum(1 for delta in deltas.values() if (delta > 0) == (direction == "max") and delta != 0)
            better_a = sum(1 for delta in deltas.values() if (delta < 0) == (direction == "max") and delta != 0)
            mean_a = (
                sum(float(cases_a[name][case]) for case in common_cases) / len(common_cases) if common_cases else None
            )
            mean_b = (
                sum(float(cases_b[name][case]) for case in common_cases) / len(common_cases) if common_cases else None
            )
            comparison[name] = {
                "direction": direction,
                "direction_source": direction_source,
                "cases": len(common_cases),
                "mean_a": mean_a,
                "mean_b": mean_b,
                "mean_delta_b_minus_a": (mean_b - mean_a) if common_cases else None,
                "cases_better_a": better_a,
                "cases_better_b": better_b,
                "winner": (run_b if better_b > better_a else run_a if better_a > better_b else "tie"),
                "per_case_delta_b_minus_a": deltas,
            }
        return {
            "session": payload_a["session"],
            "split": split.upper(),
            "run_a": run_a,
            "run_b": run_b,
            "metrics": comparison,
            "warnings": warnings,
            "next_actions": ["get_run_metrics", "leaderboard", "summarize_session"],
        }

    def job_runtime_log_path(self, job: Job) -> Path | None:
        if job.runtime_log_path is not None:
            return job.runtime_log_path
        run_name = self.configured_run_name(job.kind, job.config_path)
        if run_name is None:
            return None
        return self._workflow_runtime_root(job.kind) / run_name / "log_0.txt"

    def discover_latest_job(self, kind: str | None = None) -> Job | None:
        return self.job_registry.latest(kind=kind)

    def discover_log_path(self) -> Path | None:
        latest_job = self.discover_latest_job("train")
        if latest_job is not None:
            runtime_log = self.job_runtime_log_path(latest_job)
            if runtime_log is not None and runtime_log.exists():
                return runtime_log
            if latest_job.log_path.exists():
                return latest_job.log_path
        log_path = self.workspace_layout.statistics_log_path()
        if log_path.exists():
            return log_path
        return None

    @staticmethod
    def _newest_checkpoints(root: Path, limit: int) -> list[Path]:
        if not root.exists():
            return []
        candidates = sorted(
            [*root.rglob("*.pt"), *root.rglob("*.pth")],
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        return candidates[:limit]

    def discover_model_paths(self, limit: int = 1, run_name: str | None = None) -> list[Path]:
        # Scope to Checkpoints/<run_name> when a run is named: globbing the whole Checkpoints tree returns
        # the newest checkpoint across ALL runs, so in a sweep run B's prediction would silently use run
        # C's weights (whichever trained last). Fall back to the global newest only when no run is known.
        checkpoints = self.workspace_layout.checkpoints_dir()
        root = checkpoints / run_name if run_name else checkpoints
        return self._newest_checkpoints(root, limit)

    def active_jobs(self) -> list[Job]:
        return self.job_registry.active()

    def ensure_no_active_job(self) -> None:
        self.job_registry.ensure_no_active_job()

    def job_payload(self, job: Job) -> dict[str, Any]:
        return self.job_registry.payload(job, self._isoformat)

    def _session_readiness(self) -> dict[str, Any]:
        return self._readiness_from_statuses(self._workflow_statuses())

    def session_summary(self) -> dict[str, Any]:
        workspace = self.workspace_layout.ensure_session_workspace_exists()
        config_paths = self.config_paths()
        metrics_path = self.discover_metrics_path()
        active_jobs = [self.job_payload(job) for job in self.active_jobs()]
        latest_job = self.discover_latest_job()
        readiness = self._session_readiness()
        return {
            "session": self.session_name(),
            "path": str(workspace),
            "readiness": readiness,
            "configs": {workflow: str(path) if path.exists() else None for workflow, path in config_paths.items()},
            "outputs": {
                "checkpoints_dir": str(self.workspace_layout.checkpoints_dir()),
                "predictions_dir": str(self.workspace_layout.predictions_dir()),
                "evaluations_dir": str(self.workspace_layout.evaluations_dir()),
                "statistics_log": (
                    str(self.workspace_layout.statistics_log_path())
                    if self.workspace_layout.statistics_log_path().exists()
                    else None
                ),
                "metrics": str(metrics_path) if metrics_path is not None else None,
            },
            "latest_job": self.job_payload(latest_job) if latest_job is not None else None,
            "active_jobs": active_jobs,
            "next_actions": self._next_actions_for_readiness(readiness),
            "resources": {
                "configs": {workflow: f"session://current/config/{workflow}" for workflow in config_paths},
                "log": "session://current/log",
                "metrics": "session://current/metrics",
                "summary": "session://current/summary",
            },
        }

    def validate_session_payload(self) -> dict[str, Any]:
        workspace = self.workspace_layout.ensure_session_workspace_exists()
        config_paths = self.config_paths()
        metrics_path = self.discover_metrics_path()
        readiness = self._session_readiness()
        configs, semantic_reviews = self._config_validation_payloads(config_paths)
        validation: dict[str, Any] = {
            "session": self.session_name(),
            "path": str(workspace),
            "configs": configs,
            "latest_metrics": str(metrics_path) if metrics_path is not None else None,
            "latest_models": [str(path) for path in self.discover_model_paths(limit=3)],
            "readiness": readiness,
            "semantic_reviews": semantic_reviews,
        }
        validation["active_jobs"] = [self.job_payload(job) for job in self.active_jobs()]
        validation["next_actions"] = ["review_config_semantics", "validate_config_semantics", "summarize_session"]
        return validation
