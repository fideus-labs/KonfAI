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

import ast
import importlib
import inspect
import math
import os
import re
import shutil
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
from ruamel.yaml import YAML

YAML_SAFE = YAML(typ="safe")
YAML_DUMP = YAML()
YAML_DUMP.default_flow_style = False
YAML_DUMP.width = 4096

# --- Payload float economy -------------------------------------------------
# Round display floats (metrics, statistics) to a fixed number of significant figures
# at the payload edge so token cost drops WITHOUT changing which fields are present.
# Set to None (e.g. monkeypatch this module attribute in a test) to emit exact floats.
ROUND_SIGNIFICANT_FIGURES: int | None = 6


def _round_significant(value: float, sig: int) -> float:
    """Round one float to ``sig`` significant figures; pass 0.0/NaN/Inf through unchanged."""
    if not math.isfinite(value) or value == 0.0:
        return value
    return round(value, sig - 1 - math.floor(math.log10(abs(value))))


def round_floats(payload: Any) -> Any:
    """Recursively round every float in a JSON-like payload to ROUND_SIGNIFICANT_FIGURES.

    Structure and keys are preserved exactly; ints, bools, strings, None, and non-finite
    floats pass through untouched (6 significant figures represents any integer < 1e6
    exactly, so an int-valued float such as 1.0 stays 1.0). Returns the payload unmodified
    when ROUND_SIGNIFICANT_FIGURES is None so a test can assert exact values.
    """
    sig = ROUND_SIGNIFICANT_FIGURES
    if sig is None or isinstance(payload, bool):
        return payload
    if isinstance(payload, float):
        return _round_significant(payload, sig)
    if isinstance(payload, dict):
        return {key: round_floats(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [round_floats(item) for item in payload]
    if isinstance(payload, tuple):
        return tuple(round_floats(item) for item in payload)
    return payload


class DatasetGroupUnreadableError(ValueError):
    """Every sampled case of a scanned group exists on disk but the backend reader raised.

    Distinct from a missing group or a layout/token mismatch: the files are present but their content
    is corrupt, truncated, empty, or not a supported image. Its purpose is the isinstance check in
    ``_statistics_failure_reason``, which emits a corrupt/unreadable reason instead of a
    "restructure the dataset" one; it stays in the ValueError family like the package's other
    bad-input errors.
    """

    def __init__(self, group: str, extension: str, case_errors: dict[str, str]) -> None:
        self.case_errors = dict(case_errors)
        preview = "; ".join(f"{name}: {message}" for name, message in list(self.case_errors.items())[:3])
        super().__init__(
            f"All {len(self.case_errors)} sampled case(s) of group '{group}' exist but failed to read "
            f"with format token '{extension}': {preview}"
        )


def aggregate_case_statistics(stats: dict[str, dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    """Aggregate per-case statistics into the KonfAI ``case``/``aggregates`` payload shape.

    Vendored in the MCP package so it depends only on KonfAI's public API: the
    equivalent logic also lives inline in ``Dataset.get_statistics``.
    """
    result: dict[str, dict[str, dict[str, Any]]] = {"case": {}, "aggregates": {}}
    if not stats:
        return result

    for name, values in stats.items():
        for metric_name, value in values.items():
            result["case"].setdefault(metric_name, {})[name] = value

    aggregate_values: dict[str, list[Any]] = {}
    for case_metrics in stats.values():
        for metric_name, value in case_metrics.items():
            aggregate_values.setdefault(metric_name, []).append(value)

    for metric_name, metric_values in aggregate_values.items():
        first = metric_values[0]
        if isinstance(first, (int, float)):
            np_values = np.asarray(metric_values, dtype=np.float64)
            valid = bool(np.any(~np.isnan(np_values)))
            result["aggregates"][metric_name] = {
                "max": float(np.nanmax(np_values)) if valid else np.nan,
                "min": float(np.nanmin(np_values)) if valid else np.nan,
                "std": float(np.nanstd(np_values)) if valid else np.nan,
                "25pc": float(np.nanpercentile(np_values, 25)) if valid else np.nan,
                "50pc": float(np.nanpercentile(np_values, 50)) if valid else np.nan,
                "75pc": float(np.nanpercentile(np_values, 75)) if valid else np.nan,
                "mean": float(np.nanmean(np_values)) if valid else np.nan,
                "count": float(np.count_nonzero(~np.isnan(np_values))) if valid else np.nan,
            }
        else:
            result["aggregates"][metric_name] = {
                "max": np.nanmax(metric_values, axis=0).tolist(),
                "min": np.nanmin(metric_values, axis=0).tolist(),
                "std": np.nanstd(metric_values, axis=0).tolist(),
                "mean": np.nanmean(metric_values, axis=0).tolist(),
            }

    return result


WORKFLOW_CONFIG_FILES = {
    "train": "Config.yml",
    "prediction": "Prediction.yml",
    "evaluation": "Evaluation.yml",
}

WORKFLOW_ROOT_KEYS = {
    "train": "Trainer",
    "prediction": "Predictor",
    "evaluation": "Evaluator",
}

CONFIG_AUTHORING_QUESTIONS = [
    "Ask only if uncertain: which dataset groups are inputs, targets, or support-only groups for this task?",
    "Ask only if uncertain: which workflows are intended now: train, prediction, and/or evaluation?",
    "Ask only if uncertain: what modeling intent is being tried first: 2D, 2.5D, or 3D?",
    ("Ask only if uncertain: should the agent reuse template model code, or write/adapt its own model and config?"),
    (
        "Ask only if uncertain: should multiple dataset roots be merged, "
        "or should one split/root be used for train, test, or evaluation?"
    ),
]


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


def validate_yaml_content(content: str, filename: str, expected_root: str | None = None) -> dict[str, Any]:
    """Parse YAML content and optionally enforce one top-level root key."""
    try:
        data = YAML_SAFE.load(content)
    except Exception as exc:  # pragma: no cover - parser exceptions vary
        raise ValueError(f"{filename} is not valid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{filename} must contain a YAML mapping at the top level.")
    if expected_root is not None and expected_root not in data:
        raise ValueError(f"{filename} must define the '{expected_root}' root key.")
    return data


def yaml_dump_content(data: dict[str, Any]) -> str:
    """Serialize a mapping into normalized YAML content."""
    stream = StringIO()
    YAML_DUMP.dump(data, stream)
    return stream.getvalue()


def _lint_config_data(data: Any) -> list[dict[str, str]]:
    """Static lint for silent-failure traps an agent cannot see from the schema alone."""
    warnings: list[dict[str, str]] = []
    # Evaluator groups_dest entries bind to GroupTransformMetric, which has no patch_transforms
    # parameter (and no -1 fill) -- the trap only exists for Trainer/Predictor datasets.
    if isinstance(data, dict) and isinstance(data.get("Evaluator"), dict):
        return warnings

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            groups_dest = node.get("groups_dest")
            if isinstance(groups_dest, dict):
                for dest_name, dest in groups_dest.items():
                    if isinstance(dest, dict) and "patch_transforms" not in dest:
                        warnings.append(
                            {
                                "severity": "warning",
                                "code": "missing_patch_transforms",
                                "path": f"groups_dest.{dest_name}",
                                "message": (
                                    f"groups_dest entry '{dest_name}' omits 'patch_transforms'. Omitting it makes "
                                    "KonfAI fill this group's tensor with -1 (a segmentation target then crashes "
                                    "CrossEntropyLoss with 'Target -1 out of bounds'; a regression target is "
                                    "silently corrupted). Set 'patch_transforms: None' explicitly."
                                ),
                            }
                        )
            for value in node.values():
                _walk(value)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(data)
    warnings.extend(_lint_prediction_default_outputs_criterions(data))
    return warnings


_DEFAULT_OUTPUTS_CRITERIONS_GROUP = "Labels"


def _contains_key(node: Any, target: str) -> bool:
    """True when ``target`` appears as a mapping key anywhere in the nested structure."""
    if isinstance(node, dict):
        if target in node:
            return True
        return any(_contains_key(value, target) for value in node.values())
    if isinstance(node, list):
        return any(_contains_key(item, target) for item in node)
    return False


def _predictor_dest_groups(predictor: dict[str, Any]) -> set[str]:
    """Destination groups the Predictor dataset loads (groups_src.<src>.groups_dest keys)."""
    groups: set[str] = set()
    dataset = predictor.get("Dataset")
    if not isinstance(dataset, dict):
        return groups
    groups_src = dataset.get("groups_src")
    if isinstance(groups_src, dict):
        for src in groups_src.values():
            if isinstance(src, dict) and isinstance(src.get("groups_dest"), dict):
                groups.update(str(name) for name in src["groups_dest"])
    return groups


def _classpath_targets_yaml_model(model: dict[str, Any]) -> bool:
    """True when Model.classpath points at a .yml/.yaml model builder.

    The YAML model builder defaults ``outputs_criterions`` to None (no injection), so the
    default-'Labels' binding only happens for Python model classes.
    """
    classpath = model.get("classpath")
    if not isinstance(classpath, str):
        return False
    raw = classpath.split("|", maxsplit=1)[-1]
    return raw.rsplit(".", maxsplit=1)[-1].lower() in {"yml", "yaml"}


def _lint_prediction_default_outputs_criterions(data: Any) -> list[dict[str, str]]:
    """Prediction Model without an explicit outputs_criterions may bind a default referencing 'Labels'.

    When the Predictor Model block omits ``outputs_criterions``, KonfAI's reflection binds the model
    class's own ``__init__`` default. Builtin models (and plain modules wrapped in ``MinimalModel``)
    default to a ``TargetCriterionsLoader`` targeting group 'Labels', so their Measure init raises
    MeasureError when no 'Labels' group is loaded; a custom Network (or a composite like a GAN whose
    sub-networks own the parameter) may default differently, hence severity=warning, not error.
    """
    if not isinstance(data, dict):
        return []
    predictor = data.get("Predictor")
    if not isinstance(predictor, dict):
        return []
    model = predictor.get("Model")
    if not isinstance(model, dict):
        return []
    if _classpath_targets_yaml_model(model):
        return []
    if _contains_key(model, "outputs_criterions"):
        return []
    loaded = _predictor_dest_groups(predictor)
    if _DEFAULT_OUTPUTS_CRITERIONS_GROUP in loaded:
        return []
    loaded_desc = ", ".join(sorted(loaded)) if loaded else "none"
    return [
        {
            "severity": "warning",
            "code": "prediction_default_outputs_criterions",
            "path": "Predictor.Model",
            "message": (
                "The Predictor Model omits 'outputs_criterions'. For model classes exposing that "
                "parameter (all builtin models do), KonfAI's reflection binds its default, which "
                f"references target group '{_DEFAULT_OUTPUTS_CRITERIONS_GROUP}'. This dataset's "
                f"groups_dest loads [{loaded_desc}] and not '{_DEFAULT_OUTPUTS_CRITERIONS_GROUP}', so "
                "prediction raises MeasureError when the model initializes its Measure. Set "
                "'outputs_criterions: {}' on the (sub-)network block that exposes it to disable it "
                "for pure inference."
            ),
        }
    ]


def write_config(path: Path, content: str, overwrite: bool, expected_root: str | None = None) -> dict[str, Any]:
    """Validate and write one KonfAI YAML config file."""
    data = validate_yaml_content(content, path.name, expected_root=expected_root)
    if path.exists() and not overwrite:
        raise ValueError(f"{path.name} already exists. Set overwrite=True to replace.")
    path.write_text(content.rstrip() + "\n", encoding="utf-8")
    result: dict[str, Any] = {
        "path": str(path),
        "bytes": path.stat().st_size,
    }
    lint = _lint_config_data(data)
    if lint:
        result["lint"] = lint
    return result


def read_text(path: Path) -> str:
    """Return a full text file, or an empty string when it does not exist."""
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def read_text_range(path: Path, max_chars: int = 20000, offset: int = 0) -> dict[str, Any]:
    """Read a bounded character range of a text file for MCP read-back tools.

    Streams instead of loading the whole file: memory stays proportional to offset+max_chars,
    so a multi-GB log cannot exhaust the server. ``total_bytes`` is the on-disk size.
    """
    if not path.is_file():
        raise ValueError(f"Not a readable file: {path}")
    offset = max(offset, 0)
    max_chars = max(max_chars, 1)
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        if offset:
            handle.read(offset)
        content = handle.read(max_chars)
        truncated = bool(handle.read(1))
    return {
        "path": str(path),
        "content": content,
        "offset": offset,
        "returned_chars": len(content),
        "total_bytes": path.stat().st_size,
        "truncated": truncated,
    }


_BINARY_SNIFF_BYTES = 4096
_DELIMITED_SUFFIXES = {".csv": ",", ".tsv": "\t"}


def read_dataset_sidecar(path: Path, max_lines: int = 200, max_chars: int = 65536) -> dict[str, Any]:
    """Read a bounded preview of a dataset's non-image text file (CSV/TSV, JSON, YAML, headers, lists).

    Streams at most ``max_chars`` characters (memory never scales with file size) and refuses binary
    content by sniffing the first bytes for NUL — image volumes and weights belong to
    ``inspect_dataset``/``preview_volume``, not a text reader. CSV/TSV additionally get a structured
    preview (header + up to ``max_lines`` rows) so the agent can map label columns to cases without
    parsing raw text.
    """
    if not path.is_file():
        raise ValueError(f"Not a readable file: {path}")
    with path.open("rb") as handle:
        head = handle.read(_BINARY_SNIFF_BYTES)
    if b"\x00" in head:
        raise ValueError(
            f"'{path.name}' looks binary (NUL bytes in the first {_BINARY_SNIFF_BYTES} bytes). "
            "Use inspect_dataset for volumes/stores and preview_volume for image content."
        )
    max_lines = max(max_lines, 1)
    ranged = read_text_range(path, max_chars=max_chars)
    lines = ranged["content"].splitlines()
    line_truncated = len(lines) > max_lines
    lines = lines[:max_lines]
    payload: dict[str, Any] = {
        "ok": True,
        "path": str(path),
        "total_bytes": ranged["total_bytes"],
        "returned_lines": len(lines),
        "truncated": bool(ranged["truncated"] or line_truncated),
        "content": "\n".join(lines),
        "next_actions": ["inspect_dataset", "design_config_strategy"],
    }
    delimiter = _DELIMITED_SUFFIXES.get(path.suffix.lower())
    if delimiter is not None and lines:
        import csv

        rows = list(csv.reader(StringIO("\n".join(lines)), delimiter=delimiter))
        if rows:
            payload["kind"] = "delimited"
            payload["columns"] = rows[0]
            payload["rows"] = rows[1:]
            payload["returned_rows"] = len(rows) - 1
    return payload


def read_text_tail(path: Path, max_lines: int) -> str:
    """Return the last ``max_lines`` lines of a text file efficiently."""
    if not path.exists() or max_lines <= 0:
        return ""

    chunk_size = 8192
    remaining = max_lines
    chunks: list[str] = []
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        position = handle.tell()
        while position > 0 and remaining > 0:
            read_size = min(chunk_size, position)
            position -= read_size
            handle.seek(position)
            chunk = handle.read(read_size).decode("utf-8", errors="replace")
            chunks.append(chunk)
            remaining -= chunk.count("\n")

    text = "".join(reversed(chunks))
    lines = text.splitlines()
    return "\n".join(lines[-max_lines:])


def full_suffix(path: Path) -> str:
    """Return all suffixes joined together, for example ``.nii.gz``."""
    return "".join(path.suffixes)


def basename_without_suffixes(path: Path) -> str:
    """Return the basename of ``path`` without compound suffixes."""
    suffix = full_suffix(path)
    return path.name[: -len(suffix)] if suffix else path.name


def available_templates(examples_root: Path) -> list[str]:
    """List example directories that can seed MCP experiments."""
    if not examples_root.exists():
        return []
    return sorted(entry.name for entry in examples_root.iterdir() if entry.is_dir())


def template_dir(examples_root: Path, name: str) -> Path:
    """Resolve one example template directory or raise a helpful error."""
    # The name comes from the MCP client: reject separators / '..' so it cannot escape examples/.
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError(f"Invalid template name '{name}'.")
    template = examples_root / name
    if not template.exists() or not template.is_dir():
        raise ValueError(
            f"Unknown template '{name}'. "
            f"Available templates: {available_templates(examples_root) if examples_root.exists() else []}"
        )
    return template


def template_summary(examples_root: Path, name: str, workflows: set[str]) -> dict[str, Any]:
    """Describe one template directory and the workflow configs it contains."""
    template = template_dir(examples_root, name)
    files = sorted(path.name for path in template.iterdir() if path.is_file())
    yaml_files = [filename for filename in files if filename.endswith(".yml")]
    python_files = [filename for filename in files if filename.endswith(".py")]
    config_files = {
        workflow: filename
        for workflow, filename in WORKFLOW_CONFIG_FILES.items()
        if workflow in workflows and (template / filename).exists()
    }
    return {
        "name": template.name,
        "path": str(template),
        "yaml_files": yaml_files,
        "python_files": python_files,
        "config_files": config_files,
        "all_files": files,
        "supports": sorted(config_files),
    }


def patching_rules() -> dict[str, Any]:
    """Return generic KonfAI patching rules for agent reasoning."""
    return {
        "topic": "patching",
        "intent": "Use these as reasoning heuristics, not hardcoded decisions.",
        "concepts": {
            "patch_size": {
                "meaning": "Spatial crop extracted from the dataset pipeline before model execution.",
                "typical_shapes": {
                    "2d_or_2.5d": "[1, x, y]",
                    "3d": "[z, x, y]",
                },
            },
            "extend_slice": {
                "meaning": (
                    "Extra neighboring slices appended as slice-wise context when the dataset pipeline is 2D-like."
                ),
                "notes": [
                    "extend_slice=0 means no inter-slice context.",
                    "For slice-wise context, the effective number of slices typically becomes extend_slice + 1.",
                    "3D modeling usually carries depth in patch_size rather than extend_slice.",
                ],
            },
        },
        "strategy_hints": [
            {
                "name": "2d",
                "when": "Slice-wise baseline without neighboring-slice context.",
                "typical_config_consequences": {
                    "patch_size": "[1, x, y]",
                    "extend_slice": 0,
                },
            },
            {
                "name": "2.5d",
                "when": "Slice-wise modeling with neighboring-slice context.",
                "typical_config_consequences": {
                    "patch_size": "[1, x, y]",
                    "extend_slice": "> 0",
                },
            },
            {
                "name": "3d",
                "when": "Volumetric modeling where depth is part of the patch itself.",
                "typical_config_consequences": {
                    "patch_size": "[z, x, y]",
                    "extend_slice": 0,
                },
            },
        ],
        "authoring_checks": [
            "Does patch_size reflect slice-wise or volumetric intent?",
            "If extend_slice > 0, is that consistent with a 2D-like rather than 3D strategy?",
            (
                "If the model expects multiple channels, are those channels produced by "
                "modalities, slice context, or both?"
            ),
        ],
    }


def modeling_rules() -> dict[str, Any]:
    """Return generic modeling rules for KonfAI config design."""
    return {
        "topic": "modeling",
        "intent": "Explain conceptual consequences of model choices without hardcoding final answers.",
        "concepts": {
            "2d": "The model processes one slice at a time; patch depth is usually 1.",
            "2.5d": "The model is still slice-wise, but neighboring slices are folded into channels or context.",
            "3d": ("The model processes volumetric patches; depth lives in patch_size rather than extend_slice."),
            "channels_vs_context": [
                "Input channels can come from multiple modalities.",
                "Input channels can also come from neighboring slices in 2.5D setups.",
                "A model may therefore require more channels than the number of source modalities alone.",
            ],
        },
        "local_vs_imported_classpaths": {
            "local": (
                "classpath like 'Model:UNetpp5' refers to Python code located next to the config, typically Model.py."
            ),
            "imported": (
                "classpath like 'segmentation.UNet.UNet' refers to an importable Python object "
                "outside the session workspace."
            ),
        },
        "model_source_decision": {
            "prefer_proven_first": (
                "PREFER a shipped catalog architecture ('classpath: default|<Name>.yml', see "
                "list_components(kind='model')) over authoring a model from scratch. Catalog entries are "
                "pre-validated (weight-exact tests against their reference implementation, or a documented "
                "structural check) and several load real pretrained weights -- they are known-good. Author a "
                "custom Python/YAML model only when no catalog entry and no installable class fits the task, "
                "and then validate it before committing compute (see below)."
            ),
            "external_class": (
                "Any installed nn.Module works as-is: 'classpath: monai.networks.nets:SegResNet' (or "
                "torchvision, timm, segmentation_models_pytorch, ...). KonfAI wraps it in MinimalModel "
                "automatically -- the simplest path when one loss on the final output is enough. But it is a "
                "BLACK BOX: only its final output is visible; internal layers are NOT addressable in "
                "outputs_criterions, the model cannot be edited without code, and the import runs that "
                "library's code (trust)."
            ),
            "what_the_konfai_contract_buys": (
                "Expressing a model under the KonfAI contract (a default|<Name>.yml graph, or a Network "
                "subclass) instead of importing it as a MinimalModel black box buys four things the black box "
                "cannot give: (1) EVERY internal layer is an addressable named output -- attach a loss to any "
                "of them (deep supervision, feature/perceptual/IMPACT losses on intermediate features, "
                "auxiliary heads); (2) no-code architecture edits (depth, norm, heads) via YAML params; "
                "(3) safe-by-construction sharing -- a registry-only YAML runs no imported code; (4) uniform "
                "checkpoint / alias / EMA / patch handling. You can still start from the reference's pretrained "
                "weights: load them into the addressable KonfAI graph with konfai.utils.pretrained. Rule: need "
                "only the final output + one loss -> import is fine; need internal supervision, editability, or "
                "safe sharing -> bring it under the contract (a catalog entry already is)."
            ),
            "yaml_catalog": (
                "'classpath: default|<Name>.yml' selects a shipped declarative architecture "
                "(list_components(kind='model') lists them; inspect_object_signature explains hyperparameters, "
                "loss-attachable terminal_leaves, and how to adapt). Choose it to supervise internal layers "
                "(deep supervision, feature/perceptual losses), to edit the architecture without code, or for "
                "safe sharing (registry-only, no code execution). Weight-exact entries load the reference's "
                "pretrained checkpoints via konfai.utils.pretrained."
            ),
            "custom_python": (
                "A local 'Model:MyNet' subclassing Network is for genuinely new architectures or custom "
                "forward logic (GAN/diffusion/registration-style) that neither an import nor a YAML graph "
                "can express."
            ),
            "how_to_check_an_authored_model": (
                "inspect_object_signature reads a model's hyperparameters, loss-attachable outputs, and "
                "(for YAML) full content WITHOUT building it -- use it to understand any of the three forms. "
                "To confirm the architecture actually BUILDS and wires correctly, validate_config_semantics"
                "(level='instantiate') constructs the model inside its config (catching channel/shape/output "
                "mismatches), and describe_model_outputs lists the real outputs_criterions paths of the built "
                "model. Both build the model as part of the workflow, so the config must be complete (dataset "
                "mapped); a catalog entry needs no such check -- it is already validated."
            ),
        },
        "authoring_checks": [
            "Is the chosen model dimensionality coherent with dataset patching?",
            "If the model expects more than one input channel, where do those channels come from?",
            "Are losses and outputs attached to actual model outputs and target groups?",
        ],
    }


def configuration_rules() -> dict[str, Any]:
    """Return generic KonfAI configuration semantics for LLM authoring."""
    return {
        "topic": "configuration",
        "root_keys": {
            "train": "Trainer",
            "prediction": "Predictor",
            "evaluation": "Evaluator",
        },
        "dataset_semantics": {
            "groups_src": "Defines available dataset groups and the transforms applied to them.",
            "is_input": "Marks whether one destination group is fed into the model.",
            "dataset_filenames": "Declares dataset sources using <path>:<flag>:<extension> syntax.",
            "outputs_dataset": "Defines saved artifacts such as predictions or derived outputs.",
        },
        "model_semantics": {
            "classpath": "Selects the Python model object to instantiate.",
            "outputs_criterions": "Binds model outputs to target groups and loss/metric definitions.",
            "Patch": ("Optional model-side patching metadata; dataset patching lives under Dataset.Patch."),
        },
        "customization": {
            "policy": [
                "Examples are optional scaffolding, not the only way to author KonfAI configs.",
                "The LLM can write local Model.py, Loss.py, Transform.py, Augmentation.py, or helper modules.",
                "Use inspect_object_signature before wiring a custom or imported component into YAML.",
            ],
            "local_classpath_examples": ["Model:MyNet", "Loss:DiceFocalLoss", "Transform:WindowCT"],
        },
        "workflow_guidance": [
            "Write only the workflows you currently intend to run.",
            "Prediction and evaluation configs can be omitted until that intent exists.",
            (
                "Use semantic review and runtime validation together: review catches suspicious "
                "design choices, validation catches instantiation/runtime issues."
            ),
        ],
        "authoring_questions": list(CONFIG_AUTHORING_QUESTIONS),
    }


def config_design_summary() -> dict[str, Any]:
    """Return one compact reasoning summary for LLM-driven KonfAI config design."""
    return {
        "topic": "config_design",
        "intent": (
            "Start here first. This is the compact reasoning summary for turning a user task, "
            "a dataset layout, and a modeling idea into a coherent KonfAI config."
        ),
        "task_policy": [
            "The user must specify the task.",
            "The MCP server should describe the dataset structure and config consequences, "
            "not infer the task for the LLM.",
        ],
        "workflow": [
            "1. Inspect the dataset and confirm what groups exist.",
            (
                "2. Ask the user only the clarifications that remain genuinely uncertain "
                "and that would change the config or split."
            ),
            "3. Choose a modeling intent: 2D, 2.5D, or 3D baseline.",
            "4. Translate that intent into patching/context consequences.",
            "5. Decide whether to adapt an example or write local custom components.",
            "6. Write only the workflow configs you currently intend to run.",
            "7. Run semantic review first, then runtime validation.",
        ],
        "key_reasoning_points": [
            "2D usually means slice-wise patches and no inter-slice context.",
            "2.5D usually means slice-wise patches plus neighboring-slice context through extend_slice.",
            "3D usually means volumetric patches where depth lives inside patch_size.",
            "Input channels can come from modalities, slice context, or both.",
            "The server should warn about suspicious choices, not decide everything for the agent.",
        ],
        "customization_policy": [
            "Do not assume examples are mandatory.",
            "If the task needs a custom inductive bias, write a local model or loss in the session workspace.",
            "Before using a custom or imported component, inspect its signature so YAML parameters "
            "match its constructor.",
        ],
        "clarification_policy": [
            "Do not ask the user questions when the task, group roles, workflows, and split are already unambiguous.",
            (
                "Ask a question only when the answer would materially change the config, "
                "dataset split, workflows, or model choice."
            ),
            (
                "If multiple dataset roots or cohorts exist, ask whether they should be merged "
                "or assigned distinct roles such as train vs evaluation."
            ),
        ],
        "user_questions_to_clarify_if_uncertain": [
            (
                "What task do you want to run: segmentation, synthesis, classification, "
                "inference-only, or evaluation-only?"
            ),
            "Which dataset groups are inputs, targets, or support-only groups?",
            "Do you want train, prediction, evaluation, or only a subset?",
            "If there are multiple candidate dataset roots, should they be merged or assigned different roles?",
            "Do you want a simple baseline first, or to adapt a more specialized example?",
        ],
        "recommended_next_reads": [
            "docs://patching",
            "docs://modeling",
            "docs://configuration",
            "docs://dataset-mapping",
            "docs://examples",
        ],
    }


def docs_index() -> dict[str, Any]:
    """Return the available KonfAI MCP reasoning docs."""
    return {
        "topic": "docs_index",
        "docs": {
            "tool_index": "guide://tool-index",
            "summary": "guide://config-design",
            "patching": "docs://patching",
            "modeling": "docs://modeling",
            "configuration": "docs://configuration",
            "dataset_mapping": "docs://dataset-mapping",
            "examples": "docs://examples",
            "prediction": "docs://prediction",
            "compute": "docs://compute",
        },
    }


def prediction_rules() -> dict[str, Any]:
    """Prediction-side authoring rules: TTA, multi-model ensembles, output reassembly."""
    return {
        "topic": "prediction",
        "root_key": "Predictor",
        "ensembling": {
            "models": "run_prediction(models=[ckpt1, ckpt2, ...]) runs every checkpoint; 'combine' reduces them.",
            "combine": "Predictor-level reduction across models (e.g. Mean / Median).",
        },
        "tta": {
            "where": "Predictor.Dataset.augmentations (same DataAugmentation blocks as training).",
            "semantics": "Each augmentation draw is predicted and inverse-mapped back before reduction; "
            "TTA multiplies inference cost by the number of draws.",
        },
        "outputs_dataset": {
            "meaning": "Declares each saved artifact: which module output to save and how to post-process it.",
            "keys": {
                "group": "Output group name written into Predictions/<train_name>/.",
                "same_as_group": "Copy geometry (origin/spacing/direction) from this input group.",
                "reduction": "Reduction across models/TTA samples (Mean / Median).",
                "patch_combine": "Overlap blending when patch-based inference reassembles the volume.",
                "before_reduction_transforms / after_reduction_transforms / final_transforms": "Post-processing "
                "hooks (e.g. Argmax for segmentation, UnNormalize for synthesis).",
            },
        },
        "layout": "Outputs land in Predictions/<train_name>/<dataset_filename>/<case>/<group>.<ext>.",
        "authoring_checks": [
            "The Model section must match the trained config (classpath, channels, patch sizes).",
            "dataset_filenames may point at NEW unseen data; paths are checked before launch.",
            "Uncertainty needs the apps path (run_app_infer uncertainty=True) or a custom reduction.",
        ],
    }


def compute_rules() -> dict[str, Any]:
    """Compute/device rules: device selection, DDP semantics, memory knobs, SLURM."""
    return {
        "topic": "compute",
        "device_selection": [
            "No gpu argument means CPU: KonfAI defaults to CPU when --gpu is absent - pass gpu explicitly.",
            "gpu=[0,1] spawns one DDP process per GPU; batch_size is per process.",
            "Jobs on DISJOINT devices run concurrently on this server; same-device jobs are refused.",
        ],
        "memory_knobs": {
            "Dataset.batch_size / num_workers / pin_memory / prefetch_factor": "Loader-side memory/throughput.",
            "Trainer.autocast": "Mixed precision (AMP) - large VRAM saver.",
            "Trainer.gradient_checkpoints": "Trade compute for activation memory.",
            "Trainer.gpu_checkpoints": "Pin model segments to specific GPUs (model parallelism).",
            "Dataset.Patch.patch_size": "The dominant VRAM factor for 3D workloads.",
        },
        "cluster": {
            "how": "run_train(cluster={name, memory, num_nodes, time_limit}) submits via SLURM/submitit.",
            "requires": "pip install konfai[cluster] on a submit host.",
        },
        "monitoring": "server://capabilities reports per-GPU free VRAM (the OOM budget) and a recommended device.",
    }


def dataset_mapping_doc() -> dict[str, Any]:
    """Explain how to reason about dataset groups and group mapping."""
    return {
        "topic": "dataset_mapping",
        "principles": [
            "The user should specify the task; the agent should clarify ambiguous group roles.",
            "Inspect actual dataset groups before committing to a config.",
            "One group can be an input, a target, or a support-only group depending on the task.",
            "Support-only groups are often masks or metadata used for preprocessing, postprocessing, or metrics.",
            (
                "If the dataset contains multiple roots or cohorts, ask only if that split changes "
                "training or evaluation intent."
            ),
        ],
        "questions": [
            "If uncertain, which group is fed into the model?",
            "If uncertain, which group is the supervision target?",
            "If uncertain, which groups are only used for masking, cropping, or evaluation?",
            "If uncertain, should multiple dataset roots be merged or used as different splits?",
        ],
        "common_output_consequences": [
            "is_input=true controls model inputs.",
            "outputs_criterions should target actual supervision groups.",
            "outputs_dataset should describe the saved prediction artifact.",
        ],
    }


def examples_doc(examples_root: Path) -> dict[str, Any]:
    """Summarize how examples should be used by agents."""
    templates = available_templates(examples_root)
    return {
        "topic": "examples",
        "templates": templates,
        "guidance": [
            "Examples are starting points, not mandatory recipes.",
            "Prefer understanding what an example implies before copying it.",
            "A template is useful when its workflow and model assumptions are close to your task.",
            "When the task is materially different, let the LLM write more of the config itself.",
            "When the built-in options are too weak, write local Model.py, Loss.py, Transform.py, or helper modules.",
            "Use inspect_object_signature before wiring a custom or imported object into the config.",
        ],
        "related_resources": [f"template://{name}/summary" for name in templates],
    }


def _ast_to_simple_value(node: ast.AST | None) -> Any:
    if node is None:
        return None
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        value = _ast_to_simple_value(node.value)
        return f"{value}.{node.attr}" if value is not None else node.attr
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_ast_to_simple_value(item) for item in node.elts]
    if isinstance(node, ast.Dict):
        result: dict[str, Any] = {}
        for key, value in zip(node.keys, node.values, strict=False):
            parsed_key = _ast_to_simple_value(key)
            if isinstance(parsed_key, str):
                result[parsed_key] = _ast_to_simple_value(value)
        return result
    if isinstance(node, ast.Call):
        func_name = _ast_to_simple_value(node.func)
        return {"call": func_name}
    return None


def _doc_summary(docstring: str | None) -> str | None:
    if not docstring:
        return None
    first_line = docstring.strip().splitlines()[0].strip()
    return first_line or None


def _signature_defaults(parameters: list[dict[str, Any]]) -> dict[str, Any]:
    return {parameter["name"]: parameter["default"] for parameter in parameters if parameter.get("has_default") is True}


def _detected_contract(defaults: dict[str, Any]) -> dict[str, Any]:
    keys = ("in_channels", "dim", "nb_batch_per_step", "out_channels", "channels")
    return {key: defaults[key] for key in keys if key in defaults}


def _ast_parameter_payload(
    args: list[ast.arg],
    defaults: Sequence[ast.expr | None],
    *,
    skip_first: bool = False,
) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    positional_args = args[1:] if skip_first else args
    default_offset = len(positional_args) - len(defaults)
    for index, arg in enumerate(positional_args):
        default_value = defaults[index - default_offset] if index >= default_offset else None
        payload.append(
            {
                "name": arg.arg,
                "kind": "positional_or_keyword",
                "has_default": default_value is not None,
                "default": _ast_to_simple_value(default_value),
                "annotation": _ast_to_simple_value(arg.annotation),
            }
        )
    return payload


def _ast_signature_string(name: str, parameters: list[dict[str, Any]]) -> str:
    rendered: list[str] = []
    for parameter in parameters:
        piece = parameter["name"]
        if parameter.get("has_default") is True:
            default = parameter.get("default")
            piece += f"={default!r}"
        rendered.append(piece)
    return f"{name}({', '.join(rendered)})"


def _local_class_detected_contract(init_node: ast.FunctionDef, defaults: dict[str, Any]) -> dict[str, Any]:
    contract = _detected_contract(defaults)
    for node in ast.walk(init_node):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "__init__":
            continue
        is_super_call = isinstance(node.func.value, ast.Call) and _ast_to_simple_value(node.func.value.func) == "super"
        if not is_super_call:
            continue
        for keyword in node.keywords:
            if keyword.arg in {"in_channels", "dim", "nb_batch_per_step", "out_channels", "channels"}:
                parsed = _ast_to_simple_value(keyword.value)
                if parsed is not None:
                    contract[keyword.arg] = parsed
        break
    return contract


def summarize_local_python_object(source_path: Path, object_name: str) -> dict[str, Any]:
    """Summarize one local Python class or function when the source file is available."""
    if not source_path.exists():
        return {
            "available": False,
            "object_name": object_name,
            "source_path": str(source_path),
            "warning": "Local source file was not found.",
        }

    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    object_node = next(
        (node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == object_name),
        None,
    )
    if object_node is None:
        return {
            "available": False,
            "object_name": object_name,
            "source_path": str(source_path),
            "warning": "Requested object was not found in the local source file.",
        }

    object_kind = "class" if isinstance(object_node, ast.ClassDef) else "function"
    summary: dict[str, Any] = {
        "available": True,
        "object_name": object_name,
        "object_kind": object_kind,
        "source_path": str(source_path),
        "signature": None,
        "parameters": [],
        "defaults": {},
        "doc_summary": _doc_summary(ast.get_docstring(object_node)),
        "detected_contract": {},
    }

    if isinstance(object_node, ast.ClassDef):
        bases = [_ast_to_simple_value(base) for base in object_node.bases]
        summary["bases"] = [base for base in bases if isinstance(base, str)]
        init_node = next(
            (node for node in object_node.body if isinstance(node, ast.FunctionDef) and node.name == "__init__"),
            None,
        )
        if init_node is None:
            summary["signature"] = f"{object_name}()"
            return summary
        parameters = _ast_parameter_payload(init_node.args.args, init_node.args.defaults, skip_first=True)
        defaults = _signature_defaults(parameters)
        summary["parameters"] = _normalize_signature_payload(parameters)
        summary["defaults"] = defaults
        summary["signature"] = _ast_signature_string(object_name, parameters)
        summary["detected_contract"] = _local_class_detected_contract(init_node, defaults)
        return summary

    parameters = _ast_parameter_payload(object_node.args.args, object_node.args.defaults, skip_first=False)
    defaults = _signature_defaults(parameters)
    summary["parameters"] = _normalize_signature_payload(parameters)
    summary["defaults"] = defaults
    summary["signature"] = _ast_signature_string(object_name, parameters)
    summary["detected_contract"] = _detected_contract(defaults)
    return summary


def _inspect_parameter_payload(signature: inspect.Signature) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for parameter in signature.parameters.values():
        payload.append(
            {
                "name": parameter.name,
                "kind": parameter.kind.name.lower(),
                "has_default": parameter.default is not inspect.Signature.empty,
                "default": parameter.default,
                "annotation": None if parameter.annotation is inspect.Signature.empty else repr(parameter.annotation),
            }
        )
    return payload


def _serialize_default(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_serialize_default(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _serialize_default(item) for key, item in value.items()}
    return repr(value)


def _normalize_signature_payload(parameters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for parameter in parameters:
        normalized.append(
            {
                **parameter,
                "default": _serialize_default(parameter.get("default")),
            }
        )
    return normalized


def _detected_contract_from_signature(parameters: list[dict[str, Any]]) -> dict[str, Any]:
    defaults = {
        parameter["name"]: parameter["default"] for parameter in parameters if parameter.get("has_default") is True
    }
    return _detected_contract(defaults)


def _parse_classpath(classpath: str) -> tuple[str, str]:
    if ":" in classpath:
        parts = [part for part in classpath.split(":") if part]
        if len(parts) < 2:
            raise ValueError(f"Invalid classpath '{classpath}'.")
        return ".".join(parts[:-1]), parts[-1]
    module_name, _, object_name = classpath.rpartition(".")
    if not module_name or not object_name:
        raise ValueError(
            f"Invalid imported classpath '{classpath}'. Expected '<module>.<object>' or '<module>:<object>'."
        )
    return module_name, object_name


# Qualified name of each KonfAI extension base -> the component kind it provides.
_KONFAI_COMPONENT_BASES = {
    "konfai.metric.measure.Criterion": "criterion",
    "konfai.data.transform.Transform": "transform",
    "konfai.data.augmentation.DataAugmentation": "augmentation",
    "konfai.network.network.Network": "model",
    "konfai.metric.schedulers.Scheduler": "scheduler",
}


def _imported_object_extras(obj: Any) -> dict[str, Any]:
    """Base classes, ``forward`` contract, and KonfAI-component detection for an imported object.

    Lets an agent decide *reference directly* (the class already subclasses a KonfAI base) vs
    *write a wrapper* (a foreign class that must be subclassed/adapted).
    """
    extras: dict[str, Any] = {"bases": [], "forward": None, "konfai_base": None, "integration_hint": None}
    if not inspect.isclass(obj):
        return extras
    mro = [f"{cls.__module__}.{cls.__qualname__}" for cls in obj.__mro__ if cls is not object]
    extras["bases"] = mro
    extras["konfai_base"] = next(
        (_KONFAI_COMPONENT_BASES[name] for name in mro if name in _KONFAI_COMPONENT_BASES), None
    )
    extras["integration_hint"] = (
        f"Already a KonfAI {extras['konfai_base']}: reference it directly by classpath."
        if extras["konfai_base"]
        else (
            "Foreign class: reference it directly only if its forward signature matches the target KonfAI base's "
            "call convention (e.g. a loss with forward(input, target) -> Tensor works as a criterion); otherwise "
            "subclass the KonfAI base in a local wrapper. See describe_extension_points."
        )
    )
    forward = getattr(obj, "forward", None)
    if callable(forward):
        try:
            extras["forward"] = f"forward{inspect.signature(forward)}"
        except (TypeError, ValueError):
            extras["forward"] = None
    return extras


def _yaml_model_path(reference: str, workspace_dir: Path | None) -> Path | None:
    """Resolve a ``.yml`` model reference: ``default|<Name>.yml`` -> the shipped catalog, else workspace."""
    raw = reference.split("|", maxsplit=1)[-1]
    if not raw.lower().endswith((".yml", ".yaml")):
        return None
    if reference.startswith("default|"):
        import konfai.models.yaml as yaml_catalog

        return Path(str(yaml_catalog.__file__)).parent / raw
    if workspace_dir is not None and not Path(raw).is_absolute():
        return workspace_dir / raw
    return Path(raw)


def _yaml_last_leaf(spec: dict[str, Any]) -> str:
    """Descend a module spec through its last nested child down to the producing leaf name chain."""
    chain: list[str] = []
    nested = spec.get("modules")
    while isinstance(nested, list) and nested:
        last = nested[-1]
        if not isinstance(last, dict):
            break
        chain.append(str(last.get("name", "")))
        nested = last.get("modules")
    return ":".join(part for part in chain if part)


def _yaml_terminal_outputs(module_specs: list[Any], prefix: str = "") -> tuple[list[str], list[str]]:
    """Collect the ``out_branch: [-1]`` heads of a YAML modules tree.

    Returns ``(heads, leaves)``: the ':'-joined paths of the marked modules, and for each the full
    path down to its producing leaf — the exact key a config puts under ``outputs_criterions``.
    """
    heads: list[str] = []
    leaves: list[str] = []
    for spec in module_specs:
        if not isinstance(spec, dict):
            continue
        name = str(spec.get("name", ""))
        path = f"{prefix}:{name}" if prefix else name
        out_branch = spec.get("out_branch") or []
        if any(str(branch) == "-1" for branch in out_branch):
            heads.append(path)
            leaf_chain = _yaml_last_leaf(spec)
            leaves.append(f"{path}:{leaf_chain}" if leaf_chain else path)
        nested = spec.get("modules")
        if isinstance(nested, list):
            nested_heads, nested_leaves = _yaml_terminal_outputs(nested, path)
            heads.extend(nested_heads)
            leaves.extend(nested_leaves)
    return heads, leaves


def summarize_yaml_model_signature(reference: str, path: Path) -> dict[str, Any]:
    """Parse-only summary of a declarative YAML model: hyperparameters, named outputs, and how to adapt it.

    Never instantiates the model (no torch allocation in the server process); everything an agent
    needs comes from the file itself: the ``parameters`` block IS the hyperparameter surface (all
    overridable from the run config), and the ``out_branch: [-1]`` paths are the loss-attachable
    named outputs. The full YAML rides along so the agent can copy it into the session and edit the
    STRUCTURE when a hyperparameter override is not enough.
    """
    if not path.is_file():
        available: list[str] = []
        if reference.startswith("default|"):
            available = sorted(entry.name for entry in path.parent.glob("*.yml"))
        raise ValueError(f"Unknown YAML model '{reference}'. Available catalog models: {available}.")
    text = path.read_text(encoding="utf-8")
    data = YAML_SAFE.load(text) or {}
    name = str(data.get("name", path.stem))
    parameters = data.get("parameters") or {}
    parameter_summaries = [{"name": key, "default": _serialize_default(value)} for key, value in parameters.items()]
    terminal_outputs, terminal_leaves = _yaml_terminal_outputs(data.get("modules") or [])
    doc_lines: list[str] = []
    for line in text.splitlines():
        if line.startswith("#"):
            doc_lines.append(line.lstrip("# "))
        elif line.strip():
            break
    return {
        "ok": True,
        "classpath": reference,
        "source": "yaml_catalog" if reference.startswith("default|") else "yaml_local",
        "name": name,
        "source_path": str(path),
        "parameters": parameter_summaries,
        "terminal_outputs": terminal_outputs,
        "terminal_leaves": terminal_leaves,
        "doc_summary": " ".join(doc_lines)[:600] or None,
        "yaml_content": text,
        "how_to_adapt": (
            f"Reference it with 'classpath: {reference}'. The 'parameters' block is the hyperparameter "
            f"surface: override any of them from the run config under 'Model.{name}.parameters' "
            "(e.g. channels/dim/nb_class) instead of editing the file. Attach losses/metrics via "
            "'outputs_criterions' keyed on a terminal_leaves path (or any dotted node path inside a "
            "terminal head, visible in yaml_content). For STRUCTURAL changes (add a head, swap a "
            "norm, insert a block), copy yaml_content into the session with write_session_file, edit it, "
            "and reference the session file by its relative path."
        ),
        "limitations": [],
    }


def summarize_classpath_signature(classpath: str, workspace_dir: Path | None = None) -> dict[str, Any]:
    """Summarize one local or imported configurable object by classpath."""
    normalized = classpath.strip()
    if not normalized:
        raise ValueError("classpath must not be empty.")

    yaml_path = _yaml_model_path(normalized, workspace_dir)
    if yaml_path is not None:
        return summarize_yaml_model_signature(normalized, yaml_path)

    module_name, object_name = _parse_classpath(normalized)
    local_candidate = (
        ":" in normalized
        and "." not in normalized.split(":", 1)[0]
        and workspace_dir is not None
        and len(module_name.split(".")) == 1
        and (workspace_dir / f"{module_name}.py").exists()
    )

    if local_candidate:
        assert workspace_dir is not None
        source_path = workspace_dir / f"{module_name}.py"
        summary = summarize_local_python_object(source_path, object_name)
        return {
            "ok": bool(summary.get("available")),
            "classpath": normalized,
            "source": "local",
            "module": module_name,
            "object": object_name,
            "source_path": str(source_path),
            "summary": summary,
            "signature": summary.get("signature"),
            "parameters": _normalize_signature_payload(summary.get("parameters", [])),
            "defaults": {key: _serialize_default(value) for key, value in summary.get("defaults", {}).items()},
            "doc_summary": summary.get("doc_summary"),
            "detected_contract": summary.get("detected_contract", {}),
            "limitations": [] if summary.get("available") else [summary.get("warning")],
        }

    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        # A bare dotted config_reference (no ':') can be a builtin model as listed by list_components,
        # e.g. 'segmentation.UNet.UNet' -> 'konfai.models.python.segmentation.UNet:UNet'. Retry via the SAME
        # mapping list_components uses (catalog.model_config_reference_to_inspect_classpath) before
        # giving up; already-importable classpaths never reach here, so their behavior is unchanged.
        if ":" not in normalized:
            from .catalog import model_config_reference_to_inspect_classpath

            builtin = model_config_reference_to_inspect_classpath(normalized)
            if builtin is not None and builtin != normalized:
                fallback = summarize_classpath_signature(builtin, workspace_dir=workspace_dir)
                if fallback.get("ok"):
                    fallback["classpath"] = normalized
                    fallback["resolved_classpath"] = builtin
                    return fallback
        return {
            "ok": False,
            "classpath": normalized,
            "source": "imported",
            "module": module_name,
            "object": object_name,
            "summary": None,
            "signature": None,
            "parameters": [],
            "defaults": {},
            "doc_summary": None,
            "detected_contract": {},
            "limitations": [f"Unable to import module '{module_name}': {type(exc).__name__}: {exc}"],
        }

    if not hasattr(module, object_name):
        return {
            "ok": False,
            "classpath": normalized,
            "source": "imported",
            "module": module_name,
            "object": object_name,
            "summary": None,
            "signature": None,
            "parameters": [],
            "defaults": {},
            "doc_summary": None,
            "detected_contract": {},
            "limitations": [f"Object '{object_name}' was not found in module '{module_name}'."],
        }

    obj = getattr(module, object_name)
    object_kind = "class" if inspect.isclass(obj) else "function" if inspect.isfunction(obj) else "callable"
    try:
        signature = inspect.signature(obj)
        raw_parameters = _inspect_parameter_payload(signature)
        parameters = _normalize_signature_payload(raw_parameters)
        defaults = {key: _serialize_default(value) for key, value in _signature_defaults(raw_parameters).items()}
        signature_text = f"{object_name}{signature}"
    except (TypeError, ValueError):
        parameters = []
        defaults = {}
        signature_text = None

    return {
        "ok": True,
        "classpath": normalized,
        "source": "imported",
        "module": module_name,
        "object": object_name,
        "summary": {
            "available": True,
            "object_name": object_name,
            "object_kind": object_kind,
            "module": module_name,
        },
        "signature": signature_text,
        "parameters": parameters,
        "defaults": defaults,
        "doc_summary": _doc_summary(inspect.getdoc(obj)),
        "detected_contract": _detected_contract_from_signature(raw_parameters if signature_text else []),
        "limitations": [],
        **_imported_object_extras(obj),
    }


def _config_semantics_summary(workflow: str, payload: dict[str, Any]) -> dict[str, Any]:
    root = payload.get(workflow_root_name(workflow), {})
    if not isinstance(root, dict):
        return {"root_key": workflow_root_name(workflow)}

    model = root.get("Model", {})
    dataset = root.get("Dataset", {})
    classpath = model.get("classpath") if isinstance(model, dict) else None
    model_name = None
    if isinstance(classpath, str):
        model_name = classpath.split(":")[-1].split(".")[-1]
    model_payload = model.get(model_name, {}) if model_name and isinstance(model, dict) else {}
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
    patch = dataset.get("Patch", {}) if isinstance(dataset, dict) else {}
    return {
        "root_key": workflow_root_name(workflow),
        "model_classpath": classpath,
        "model_name": model_name,
        "model_dim": model_payload.get("dim") if isinstance(model_payload, dict) else None,
        "dataset_groups": sorted(groups_src.keys()) if isinstance(groups_src, dict) else [],
        "input_groups": sorted(input_groups),
        "non_input_groups": sorted(non_input_groups),
        "dataset_patch": (
            {
                "patch_size": patch.get("patch_size"),
                "extend_slice": patch.get("extend_slice"),
            }
            if isinstance(patch, dict)
            else None
        ),
        "train_name": root.get("train_name"),
    }


def template_example_configs(examples_root: Path, name: str, workflows: set[str]) -> dict[str, Any]:
    """Return structured summaries of the example configs for one template."""
    configs = load_template_configs(examples_root, name)
    selected = {workflow: payload for workflow, payload in configs.items() if workflow in workflows}
    return {
        "template": name,
        "available_workflows": sorted(selected),
        "configs": {workflow: _config_semantics_summary(workflow, payload) for workflow, payload in selected.items()},
    }


def template_model_summaries(examples_root: Path, name: str, workflows: set[str]) -> dict[str, Any]:
    """Summarize the model references used by one template, with local metadata when available."""
    template = template_dir(examples_root, name)
    configs = load_template_configs(examples_root, name)
    summaries: dict[str, Any] = {}
    for workflow, payload in configs.items():
        if workflow not in workflows:
            continue
        root = payload.get(workflow_root_name(workflow), {})
        if not isinstance(root, dict):
            continue
        model = root.get("Model", {})
        if not isinstance(model, dict):
            continue
        classpath = model.get("classpath")
        if not isinstance(classpath, str):
            continue
        model_name = classpath.split(":")[-1].split(".")[-1]
        model_payload = model.get(model_name, {}) if isinstance(model.get(model_name), dict) else {}
        summary: dict[str, Any] = {
            "classpath": classpath,
            "model_name": model_name,
            "workflow": workflow,
            "dim": model_payload.get("dim") if isinstance(model_payload, dict) else None,
            "source": "imported",
            "local_source_path": None,
            "local_summary": None,
        }
        if ":" in classpath and "." not in classpath.split(":", 1)[0]:
            source_path = template / f"{classpath.split(':', 1)[0]}.py"
            summary["source"] = "local"
            summary["local_source_path"] = str(source_path)
            summary["local_summary"] = summarize_local_python_object(source_path, model_name)
        summaries[workflow] = summary
    return {
        "template": name,
        "models": summaries,
    }


def template_guidance_summary(examples_root: Path, name: str, workflows: set[str]) -> dict[str, Any]:
    """Return one compact template summary with the information an agent actually needs."""
    summary = template_summary(examples_root, name, workflows)
    config_summaries = template_example_configs(examples_root, name, workflows)
    model_summaries = template_model_summaries(examples_root, name, workflows)
    return {
        **summary,
        "authoring_model": (
            "Templates are optional starting points. Prefer understanding the example before copying it, "
            "and feel free to write KonfAI YAML and local Model.py, Loss.py, or Transform.py files directly "
            "when the task differs materially."
        ),
        "dataset_filename_syntax": {
            "pattern": "<path>:<flag>:<extension>",
            "flags": {
                "a": "dataset artifact on disk",
                "i": "derived artifact such as predictions",
            },
        },
        "notes": [
            "Dataset reads are extension-agnostic; the extension is only a supported file-format hint.",
            "Copy Python files only when you intentionally reuse template model code.",
            "Create only the workflow configs you currently intend to run: train, prediction, or evaluation.",
            "Prediction and evaluation configs are optional until that intent is chosen.",
            "Local custom components can live directly in the session workspace and be referenced "
            "with Module:Object classpaths.",
        ],
        "config_summaries": config_summaries["configs"],
        "model_summaries": model_summaries["models"],
        "related_docs": {
            "overview": "guide://config-design",
            "docs": "docs://index",
        },
        "authoring_questions": list(CONFIG_AUTHORING_QUESTIONS),
    }


def load_template_configs(examples_root: Path, name: str) -> dict[str, dict[str, Any]]:
    """Load every YAML config present in one template directory."""
    template = template_dir(examples_root, name)
    configs: dict[str, dict[str, Any]] = {}
    config_map = {
        "train": template / "Config.yml",
        "prediction": template / "Prediction.yml",
        "evaluation": template / "Evaluation.yml",
    }
    for workflow, path in config_map.items():
        if path.exists():
            data = YAML_SAFE.load(path.read_text(encoding="utf-8")) or {}
            if not isinstance(data, dict):
                raise ValueError(f"Template file {path.name} must contain a mapping at the top level.")
            configs[workflow] = data
    return configs


def workflow_root_name(workflow: str) -> str:
    """Map a workflow slug to its KonfAI YAML root key."""
    if workflow not in WORKFLOW_ROOT_KEYS:
        raise ValueError(f"Unsupported workflow: {workflow}")
    return WORKFLOW_ROOT_KEYS[workflow]


def template_groups(configs: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    """Extract source group names referenced by each workflow config."""
    result: dict[str, list[str]] = {}
    for workflow, payload in configs.items():
        root_name = workflow_root_name(workflow)
        groups_src = payload.get(root_name, {}).get("Dataset", {}).get("groups_src", {})
        result[workflow] = sorted(groups_src.keys()) if isinstance(groups_src, dict) else []
    return result


def replace_group_tokens(value: str, group_map: dict[str, str]) -> str:
    """Rewrite ``group:subgroup`` references according to a group alias map."""
    if value in group_map:
        return group_map[value]
    if not any(delimiter in value for delimiter in (":", ";", "/")):
        return value
    parts = re.split(r"([:;/])", value)
    return "".join(group_map.get(part, part) for part in parts)


def rewrite_group_references(payload: Any, group_map: dict[str, str]) -> Any:
    """Recursively rewrite group references inside YAML-like nested structures."""
    if isinstance(payload, dict):
        rewritten: dict[Any, Any] = {}
        for key, value in payload.items():
            new_key = replace_group_tokens(key, group_map) if isinstance(key, str) else key
            rewritten[new_key] = rewrite_group_references(value, group_map)
        return rewritten
    if isinstance(payload, list):
        return [rewrite_group_references(item, group_map) for item in payload]
    if isinstance(payload, str):
        return replace_group_tokens(payload, group_map)
    return payload


def default_group_map(template_groups: list[str], dataset_groups: list[str]) -> dict[str, str]:
    """Create a conservative automatic group-name mapping for one template."""
    group_map: dict[str, str] = {}
    dataset_lookup = {group.lower(): group for group in dataset_groups}
    for template_group in template_groups:
        candidate = dataset_lookup.get(template_group.lower())
        if candidate is not None:
            group_map[template_group] = candidate
            continue
        if len(dataset_groups) == 1:
            group_map[template_group] = dataset_groups[0]
    return group_map


def label_output_dtype(label_count: int) -> str:
    """Choose a compact unsigned integer dtype for a label map."""
    if label_count <= 256:
        return "uint8"
    if label_count <= 65536:
        return "uint16"
    return "uint32"


def _referenced_classpaths(node: Any) -> Iterator[str]:
    """Yield every ``classpath`` string found anywhere in a parsed config tree."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "classpath" and isinstance(value, str):
                yield value
            else:
                yield from _referenced_classpaths(value)
    elif isinstance(node, (list, tuple)):
        for item in node:
            yield from _referenced_classpaths(item)


def _classpath_local_file(classpath: str) -> str:
    """Resolve a model ``classpath`` to the local template filename that would define it.

    ``UNet.yml`` -> ``UNet.yml`` (declarative model), ``Model:Gan`` -> ``Model.py``
    (local Python class). Importable dotted paths resolve to a ``<first-segment>.py``
    name that simply will not exist in the template directory, so they are ignored.
    """
    cleaned = classpath.strip()
    if cleaned.endswith((".yml", ".yaml")):
        return Path(cleaned).name
    module = cleaned.split(":", 1)[0].split(".", 1)[0]
    return f"{module}.py"


def copy_template_subset(
    destination: Path,
    template_dir: Path,
    overwrite: bool,
    include_python: bool,
    workflows: list[str] | None,
    include_support_files: bool = False,
) -> tuple[list[str], list[str]]:
    """Copy a selected subset of template configs and their model-definition dependencies.

    Model files referenced through ``classpath`` in the selected configs are treated as
    hard dependencies: declarative ``.yml`` models are always copied so the config can be
    built, while local ``.py`` models stay opt-in (``include_python``). Returns
    ``(copied, skipped_python)`` — the referenced ``.py`` dependencies that were NOT copied;
    the seeded configs cannot resolve without them, so callers must surface the list.
    """
    copied: list[str] = []
    skipped_python: set[str] = set()
    workflow_filenames = set(WORKFLOW_CONFIG_FILES.values())
    requested_workflows = list(WORKFLOW_CONFIG_FILES) if workflows is None else workflows
    selected_names = {
        filename for workflow, filename in WORKFLOW_CONFIG_FILES.items() if workflow in requested_workflows
    }

    for config_name in list(selected_names):
        config_path = template_dir / config_name
        if not config_path.is_file():
            continue
        try:
            parsed = YAML_SAFE.load(config_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for classpath in _referenced_classpaths(parsed):
            candidate = _classpath_local_file(classpath)
            candidate_path = template_dir / candidate
            if not candidate_path.is_file() or candidate in workflow_filenames:
                continue
            if candidate.endswith((".yml", ".yaml")) or (include_python and candidate.endswith(".py")):
                selected_names.add(candidate)
            elif candidate.endswith(".py"):
                skipped_python.add(candidate)

    if include_python:
        selected_names.update(path.name for path in template_dir.iterdir() if path.is_file() and path.suffix == ".py")
    if include_support_files:
        selected_names.update(
            path.name
            for path in template_dir.iterdir()
            if path.is_file() and path.name not in workflow_filenames and path.suffix != ".py"
        )

    for entry in sorted(path for path in template_dir.iterdir() if path.is_file()):
        if entry.name not in selected_names:
            continue
        target = destination / entry.name
        if target.exists() and not overwrite:
            continue
        shutil.copyfile(entry, target)
        copied.append(entry.name)
    return copied, sorted(skipped_python)


def case_directories(dataset_dir: Path) -> list[Path]:
    """Return per-case directories for a KonfAI dataset layout."""
    case_dirs = sorted(path for path in dataset_dir.iterdir() if path.is_dir())
    return case_dirs if case_dirs else [dataset_dir]
