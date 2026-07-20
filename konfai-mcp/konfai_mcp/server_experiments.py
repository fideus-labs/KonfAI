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

import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from konfai.utils.utils import split_path_spec
from ruamel.yaml import YAML

from . import runner as mcp_runner
from .dataset_inspection import DatasetInspectionMixin
from .metrics_service import MetricsServiceMixin
from .server_jobs import Job, JobRegistry
from .server_support import (
    WORKFLOW_CONFIG_FILES,
    WorkspaceLayout,
    available_templates,
    default_group_map,
    load_template_configs,
    summarize_classpath_signature,
    template_dir,
    template_groups,
    workflow_root_name,
)

YAML_SAFE = YAML(typ="safe")


@dataclass
class SessionService(DatasetInspectionMixin, MetricsServiceMixin):
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
