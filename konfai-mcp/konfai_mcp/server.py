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

import difflib
import json
import os
import random
import re
import shutil
import signal
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Annotated, Any, Literal, cast

import konfai as konfai_pkg
from pydantic import Field

try:
    from fastmcp import FastMCP
    from fastmcp.server.auth import StaticTokenVerifier
    from fastmcp.utilities.types import Image as FastMCPImage
except ImportError as exc:  # pragma: no cover - depends on optional install
    raise RuntimeError(
        "KonfAI MCP requires 'fastmcp'. Install the MCP server with: pip install -e ./konfai-mcp"
    ) from exc

from .capabilities import describe_config_schema as _describe_config_schema
from .capabilities import describe_konfai_capabilities as _describe_konfai_capabilities
from .catalog import COMPONENT_KINDS
from .catalog import list_components as _catalog_list_components
from .extensions import check_external_dependency as _check_external_dependency
from .extensions import describe_extension_points as _describe_extension_points
from .guide import (
    CLARIFY_TASK_AND_GROUPS_PROMPT,
    DEBUG_CONFIG_WARNING_PROMPT,
    PLAN_CONFIG_STRATEGY_PROMPT,
    SOLVE_TASK_PROMPT,
    TOOL_DESCRIPTIONS,
)
from .runner import run_api_in_subprocess as _run_api_in_subprocess
from .server_apps import AppService
from .server_experiments import SessionService
from .server_jobs import Job, JobRegistry
from .server_support import (
    WORKFLOW_CONFIG_FILES,
    DatasetGroupUnreadableError,
    WorkspaceLayout,
    available_templates,
    case_directories,
    compute_rules,
    config_design_summary,
    configuration_rules,
    copy_template_subset,
    dataset_mapping_doc,
    docs_index,
    examples_doc,
    modeling_rules,
    patching_rules,
    prediction_rules,
    read_dataset_sidecar,
    read_text,
    read_text_range,
    read_text_tail,
    round_floats,
    summarize_classpath_signature,
    template_dir,
    template_guidance_summary,
    validate_yaml_content,
    workflow_root_name,
    write_config,
)
from .workflows import WORKFLOW_SPECS, JobKind, WorkflowKind

mcp = FastMCP("KonfAI")

KONFAI_VERSION = konfai_pkg.__version__
konfai_get_available_devices = konfai_pkg.get_available_devices
konfai_get_ram = konfai_pkg.get_ram
konfai_get_vram = konfai_pkg.get_vram

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_ROOT = REPO_ROOT / "examples"
WORKSPACES_ROOT = (
    Path(os.environ.get("KONFAI_MCP_WORKSPACES_ROOT", Path.home() / "KonfAI_Workspaces")).expanduser().resolve()
)
WORKSPACES_ROOT.mkdir(parents=True, exist_ok=True)
WORKSPACE_LAYOUT = WorkspaceLayout(WORKSPACES_ROOT)
MAX_LOG_TAIL_LINES = int(os.environ.get("KONFAI_MCP_LOG_TAIL_LINES", "200"))
ACTIVE_JOB_STATES = {"queued", "running"}
VALIDATION_LEVELS = {"instantiate", "setup", "train_step"}
WORKFLOWS = set(WORKFLOW_SPECS)
JOB_REGISTRY = JobRegistry(ACTIVE_JOB_STATES, workspace_layout=WORKSPACE_LAYOUT)
_JOBS_LOCK = JOB_REGISTRY.lock
_JOBS = JOB_REGISTRY.jobs


SESSION = SessionService(
    repo_root=REPO_ROOT,
    examples_root=EXAMPLES_ROOT,
    workspace_layout=WORKSPACE_LAYOUT,
    job_registry=JOB_REGISTRY,
    max_log_tail_lines=MAX_LOG_TAIL_LINES,
    active_job_states=ACTIVE_JOB_STATES,
    validation_levels=VALIDATION_LEVELS,
    workflows=WORKFLOWS,
)

APP_SERVICE = AppService(workspace_layout=WORKSPACE_LAYOUT)


_SESSION_SWITCH_LOCK = threading.Lock()


def _activate_session(name: str) -> None:
    """Swap the module-level state onto another named session workspace.

    The job registry is rebuilt from that session's persisted jobs; callers must ensure no job is
    active first, otherwise the live subprocess handle would be dropped. The lock serialises
    concurrent switches (relevant for SSE/HTTP transports; stdio has a single client).
    """
    global WORKSPACE_LAYOUT, SESSION, APP_SERVICE, JOB_REGISTRY, _JOBS_LOCK, _JOBS
    with _SESSION_SWITCH_LOCK:
        _activate_session_locked(name)


def _activate_session_locked(name: str) -> None:
    global WORKSPACE_LAYOUT, SESSION, APP_SERVICE, JOB_REGISTRY, _JOBS_LOCK, _JOBS
    # Build the full replacement state locally, then swap every global in one consecutive block:
    # a concurrent tool (SSE/HTTP transports) must never observe a half-built mix (new layout
    # with the old session) during construction, which includes disk IO for persisted jobs. Readers
    # that grab two globals back-to-back keep a nanosecond-scale window; stdio is single-client.
    workspace_layout = WorkspaceLayout(WORKSPACES_ROOT, name)
    job_registry = JobRegistry(ACTIVE_JOB_STATES, workspace_layout=workspace_layout)
    session = SessionService(
        repo_root=REPO_ROOT,
        examples_root=EXAMPLES_ROOT,
        workspace_layout=workspace_layout,
        job_registry=job_registry,
        max_log_tail_lines=MAX_LOG_TAIL_LINES,
        active_job_states=ACTIVE_JOB_STATES,
        validation_levels=VALIDATION_LEVELS,
        workflows=WORKFLOWS,
    )
    app_service = AppService(workspace_layout=workspace_layout)
    WORKSPACE_LAYOUT = workspace_layout
    JOB_REGISTRY = job_registry
    _JOBS_LOCK = job_registry.lock
    _JOBS = job_registry.jobs
    SESSION = session
    APP_SERVICE = app_service


def _read_text_tail(path: Path, max_lines: int = MAX_LOG_TAIL_LINES) -> str:
    return read_text_tail(path, max_lines=max_lines)


def _job_payload(job: Job) -> dict[str, Any]:
    return JOB_REGISTRY.payload(job, SESSION._isoformat)


def _write_workflow_config(
    workflow: WorkflowKind,
    content: str,
    overwrite: bool,
) -> dict[str, Any]:
    WORKSPACE_LAYOUT.ensure_session_workspace_exists()
    result = write_config(
        SESSION.config_path(workflow),
        content,
        overwrite,
        expected_root=workflow_root_name(workflow),
    )
    filename = WORKFLOW_CONFIG_FILES[workflow]
    return {
        "written": filename,
        **result,
        "next_actions": ["review_config_semantics", "validate_config_semantics", f"run_{workflow}"],
    }


def _isoformat(timestamp: float | None) -> str | None:
    return SESSION._isoformat(timestamp)


def _normalize_int_list(value: int | list[int] | None, *, field_name: str) -> list[int] | None:
    if value is None:
        return None
    items = [value] if isinstance(value, int) else value
    if not items:
        raise ValueError(f"{field_name} cannot be an empty list.")
    for item in items:
        if item < 0:
            raise ValueError(f"{field_name} values must be >= 0.")
    return items


def _normalize_string_list(value: str | list[str] | None, *, field_name: str) -> list[str] | None:
    if value is None:
        return None
    items = [value] if isinstance(value, str) else value
    normalized = [item.strip() for item in items if item.strip()]
    if not normalized:
        raise ValueError(f"{field_name} cannot be empty.")
    return normalized


def _build_bearer_auth_provider(
    bearer_token: str | None,
    *,
    host: str | None = None,
    port: int | None = None,
) -> StaticTokenVerifier | None:
    token = (bearer_token or "").strip()
    if not token:
        return None
    return StaticTokenVerifier(
        tokens={
            token: {
                "client_id": "konfai-mcp",
                "scopes": [],
            }
        }
    )


def _configure_transport_auth(
    transport: Literal["stdio", "sse", "streamable-http"],
    *,
    host: str | None = None,
    port: int | None = None,
    bearer_token: str | None = None,
) -> StaticTokenVerifier | None:
    if transport == "stdio":
        mcp.auth = None
        return None
    auth_provider = _build_bearer_auth_provider(bearer_token, host=host, port=port)
    mcp.auth = auth_provider
    return auth_provider


_DEFAULT_TRANSPORT = os.environ.get("KONFAI_MCP_TRANSPORT", "stdio")
if _DEFAULT_TRANSPORT not in {"stdio", "sse", "streamable-http"}:
    _DEFAULT_TRANSPORT = "stdio"
_configure_transport_auth(
    cast(Literal["stdio", "sse", "streamable-http"], _DEFAULT_TRANSPORT),
    bearer_token=os.environ.get("KONFAI_MCP_BEARER_TOKEN"),
)


def _normalize_dataset_inputs(
    dataset_dir: str | None,
    dataset_dirs: str | list[str] | None,
) -> tuple[Path | None, list[Path] | None]:
    normalized_dirs = _normalize_string_list(dataset_dirs, field_name="dataset_dirs")
    dataset_path = Path(dataset_dir).expanduser().resolve() if dataset_dir is not None else None
    dataset_paths = (
        [Path(value).expanduser().resolve() for value in normalized_dirs] if normalized_dirs is not None else None
    )
    if dataset_path is None and not dataset_paths:
        raise ValueError("Provide dataset_dir or dataset_dirs.")
    return dataset_path, dataset_paths


def _job_devices(gpu: list[int] | None, cpu: int | None, cluster: dict[str, Any] | None = None) -> list[str]:
    """Device reservation for the job registry: disjoint sets may run concurrently.

    KonfAI defaults to CPU when no gpu is given, so None/empty gpu reserves 'cpu'; a SLURM job
    runs off-host and reserves a cluster slot instead of local devices.
    """
    if cluster is not None:
        return [f"slurm:{cluster['name']}"]
    if gpu:
        return [str(index) for index in gpu]
    return ["cpu"]


def _cpu_fallback_warnings(gpu: list[int] | None, cluster: dict[str, Any] | None = None) -> list[str]:
    """Factual notice when a LOCAL run omits gpu while CUDA devices are visible.

    KonfAI defaults to CPU when no gpu is given, so on a GPU host it silently trains on CPU. A SLURM job
    runs off-host, so the server's own CUDA visibility is irrelevant and no notice is emitted. The torch
    import is guarded so a torch-less / CPU-only host never raises.
    """
    if gpu or cluster is not None:
        return []
    try:
        import torch

        if not torch.cuda.is_available():
            return []
        count = torch.cuda.device_count()
    except Exception:
        return []
    if count <= 0:
        return []
    return [
        f"No GPU selected; training will run on CPU although {count} CUDA device(s) are visible. "
        "Pass gpu=[0] to use the GPU."
    ]


def _resolve_train_config(config_file: str | None) -> Path:
    """Resolve the training config: the session Config.yml, or an alternate sibling (e.g. Config_GAN.yml)."""
    if config_file is None:
        config_path = SESSION.config_path("train")
        if not config_path.exists():
            raise ValueError("Config.yml not found. Write a training config first.")
        return config_path
    if Path(config_file).name != config_file or config_file in {".", ".."}:
        raise ValueError("config_file must be a direct child of the session workspace.")
    config_path = SESSION.workspace_dir() / config_file
    if not config_path.exists():
        raise ValueError(f"{config_file} not found in the session workspace.")
    validate_yaml_content(config_path.read_text(encoding="utf-8"), config_file, expected_root="Trainer")
    return config_path


def _validate_cluster(cluster: dict[str, Any] | None) -> dict[str, Any] | None:
    if cluster is None:
        return None
    required = {"name", "memory", "num_nodes", "time_limit"}
    missing = sorted(required - set(cluster))
    unknown = sorted(set(cluster) - required)
    if missing or unknown:
        raise ValueError(
            f"cluster expects exactly the keys {sorted(required)} (submitit SLURM submission); "
            f"missing: {missing or 'none'}, unknown: {unknown or 'none'}."
        )
    return cluster


def _device_args(gpu: int | list[int] | None, cpu: int | None) -> list[str]:
    normalized_gpu = _normalize_int_list(gpu, field_name="gpu")
    if normalized_gpu is not None and cpu is not None:
        raise ValueError("gpu and cpu are mutually exclusive.")
    if normalized_gpu:
        return ["--gpu", *[str(device) for device in normalized_gpu]]
    if cpu is not None:
        if cpu <= 0:
            raise ValueError("cpu must be > 0.")
        return ["--cpu", str(cpu)]
    return []


def _runtime_job_spec(
    *,
    kind: WorkflowKind,
    config_path: Path,
    gpu: int | list[int] | None,
    cpu: int | None,
    overwrite: bool,
    quiet: bool,
    single_process: bool,
    tensorboard: bool = False,
    models: list[Path] | None = None,
    resume_model: Path | str | None = None,
    lr: float | None = None,
    cluster: dict[str, Any] | None = None,
) -> dict[str, Any]:
    runner_command = WORKFLOW_SPECS[kind].command
    if resume_model is not None:
        if kind != "train":
            raise ValueError("resume_model is only valid for training jobs.")
        runner_command = "RESUME"
    normalized_gpu = _normalize_int_list(gpu, field_name="gpu")
    command = [
        "konfai_mcp.api",
        runner_command,
        "-c",
        str(config_path),
        *_device_args(normalized_gpu, cpu),
        *(["--models", *[str(path) for path in models]] if models else []),
        *(["--model", str(resume_model)] if resume_model is not None else []),
        *(["--lr", str(lr)] if lr is not None else []),
        *(["--overwrite"] if overwrite else []),
        *(["--quiet"] if quiet else []),
        *(["--tensorboard"] if tensorboard else []),
        *(["--single-process"] if single_process else []),
        *(["--cluster", str(cluster["name"])] if cluster else []),
    ]
    return {
        "command": command,
        "target": "konfai_mcp.runner:run_workflow_api",
        "kwargs": {
            "command": runner_command,
            "config": str(config_path),
            "models": [str(path) for path in models] if models else None,
            "model": str(resume_model) if resume_model is not None else None,
            "lr": lr,
            "gpu": normalized_gpu,
            "cpu": cpu,
            "overwrite": overwrite,
            "quiet": quiet,
            "tensorboard": tensorboard,
            "single_process": single_process,
            "cluster_kwargs": cluster,
        },
    }


def _launch_job(
    kind: JobKind,
    command: list[str],
    config_path: Path,
    extra_manifest: dict[str, Any] | None = None,
    target: str | None = None,
    kwargs: dict[str, object] | None = None,
    devices: list[str] | None = None,
) -> dict[str, Any]:
    workspace = WORKSPACE_LAYOUT.ensure_session_workspace_exists()
    WORKSPACE_LAYOUT.jobs_dir().mkdir(parents=True, exist_ok=True)
    run_name = SESSION.configured_run_name(kind, config_path)
    runtime_log_path = SESSION.runtime_log_path_for(kind, config_path)
    job = JOB_REGISTRY.launch(
        session=WORKSPACE_LAYOUT.current_session or "default",
        kind=kind,
        command=command,
        cwd=workspace,
        log_path=WORKSPACE_LAYOUT.jobs_dir() / f"{kind}_{uuid.uuid4().hex[:12]}.log",
        config_path=config_path,
        run_name=run_name,
        devices=devices,
        runtime_log_path=runtime_log_path,
        extra_manifest={**(extra_manifest or {}), "environment": _environment_snapshot()},
        target=target,
        kwargs={**(kwargs or {}), "cwd": str(workspace)},
    )
    payload = _job_payload(job)
    preflight = _vram_preflight(job.devices)
    if preflight is not None:
        payload["vram_preflight"] = preflight
    return payload


def _cancel_job_payload(job_id: str, wait_s: float = 5.0) -> dict[str, Any]:
    return JOB_REGISTRY.cancel(job_id, _isoformat, wait_s=wait_s)


def _round_gb(value: float | None) -> float | None:
    return None if value is None else round(value, 2)


def _gpu_devices(devices_index: list[int], devices_name: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
    """Per-device VRAM (total/used/free GB) for the visible GPUs, reusing KonfAI's public VRAM helper.

    Free VRAM is the number that decides whether a training patch/batch fits, so it is broken down per
    device (the aggregate hides an imbalanced multi-GPU box). Never raises: a query failure leaves that
    device's VRAM fields null with one shared warning, so callers degrade gracefully.
    """
    warnings: list[str] = []
    devices: list[dict[str, Any]] = []
    for ordinal, index in enumerate(devices_index):
        used_gb: float | None = None
        total_gb: float | None = None
        try:
            used_gb, total_gb = konfai_get_vram([index])
        except Exception as exc:  # pragma: no cover - depends on host runtime
            message = f"Unable to inspect VRAM: {type(exc).__name__}: {exc}"
            if message not in warnings:
                warnings.append(message)
        free_gb = None if used_gb is None or total_gb is None else max(total_gb - used_gb, 0.0)
        devices.append(
            {
                "index": index,
                "name": devices_name[ordinal] if ordinal < len(devices_name) else None,
                "vram_total_gb": _round_gb(total_gb),
                "vram_used_gb": _round_gb(used_gb),
                "vram_free_gb": _round_gb(free_gb),
            }
        )
    return devices, warnings


def _recommended_device(devices: list[dict[str, Any]], devices_index: list[int]) -> dict[str, Any]:
    """Prefer the GPU with the most free VRAM; fall back to the first visible GPU, else CPU."""
    rankable = [device for device in devices if device["vram_free_gb"] is not None]
    if rankable:
        return {"gpu": [max(rankable, key=lambda device: device["vram_free_gb"])["index"]]}
    if devices_index:
        return {"gpu": [devices_index[0]]}
    return {"cpu": 1}


def _runtime_capabilities() -> dict[str, Any]:
    warnings: list[str] = []

    devices_index: list[int] = []
    devices_name: list[str] = []
    try:
        devices_index, devices_name = konfai_get_available_devices()
    except Exception as exc:  # pragma: no cover - depends on host runtime
        warnings.append(f"Unable to detect GPU devices: {type(exc).__name__}: {exc}")

    ram_used_gb: float | None = None
    ram_total_gb: float | None = None
    try:
        ram_used_gb, ram_total_gb = konfai_get_ram()
    except Exception as exc:  # pragma: no cover - depends on host runtime
        warnings.append(f"Unable to inspect RAM: {type(exc).__name__}: {exc}")

    devices, vram_warnings = _gpu_devices(devices_index, devices_name)
    warnings.extend(vram_warnings)

    # Aggregate VRAM (kept for backward compatibility) derived from the per-device breakdown.
    totals = [device["vram_total_gb"] for device in devices if device["vram_total_gb"] is not None]
    useds = [device["vram_used_gb"] for device in devices if device["vram_used_gb"] is not None]

    return {
        "konfai_version": KONFAI_VERSION,
        "gpu": {
            "available": bool(devices_index),
            "visible_indices": devices_index,
            "visible_names": devices_name,
            "count": len(devices_index),
            "vram_gb": {
                "used": _round_gb(sum(useds)) if useds else None,
                "total": _round_gb(sum(totals)) if totals else None,
            },
            "devices": devices,
        },
        "ram_gb": {"used": _round_gb(ram_used_gb), "total": _round_gb(ram_total_gb)},
        "recommended_device": _recommended_device(devices, devices_index),
        "warnings": warnings,
    }


def _vram_preflight(device_ids: list[str] | None) -> dict[str, Any] | None:
    """Free-VRAM snapshot for the GPUs a job will actually use, attached to its launch payload.

    This puts the OOM-deciding number in front of the agent *at launch* -- where it can size the next
    run -- instead of requiring a separate lookup. Returns None for CPU-only or device-less jobs, and
    never raises: monitoring must not block a launch.
    """
    if not device_ids:
        return None
    gpu_ids: list[int] = []
    for device_id in device_ids:
        try:
            gpu_ids.append(int(device_id))
        except (TypeError, ValueError):
            continue  # skip 'cpu' / non-numeric entries
    if not gpu_ids:
        return None
    try:
        visible_index, visible_names = konfai_get_available_devices()
        name_by_index = dict(zip(visible_index, visible_names, strict=False))
    except Exception:  # pragma: no cover - depends on host runtime
        name_by_index = {}
    devices, _ = _gpu_devices(gpu_ids, [])
    for device in devices:
        device["name"] = name_by_index.get(device["index"])
    return {
        "devices": devices,
        "guidance": (
            "vram_free_gb is the budget this run must fit. Training uses far more VRAM than inference for the "
            "same patch/batch (backprop activations + any loss network); if it OOMs, lower the config's "
            "Dataset.batch_size (and/or Patch.patch_size) and relaunch."
        ),
    }


def _environment_snapshot() -> dict[str, Any]:
    """Per-job environment record persisted in the manifest (Methods-grade provenance)."""
    import platform
    from importlib import metadata

    def version_of(distribution: str) -> str | None:
        try:
            return metadata.version(distribution)
        except metadata.PackageNotFoundError:
            return None

    capabilities = _runtime_capabilities()
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "konfai": KONFAI_VERSION,
        "packages": {name: version_of(name) for name in ("torch", "numpy", "SimpleITK", "fastmcp", "konfai-apps")},
        "gpus": capabilities["gpu"]["visible_names"],
        "captured_at": _isoformat(time.time()),
    }


@mcp.resource("server://info")
def server_info() -> dict[str, Any]:
    """Return a compact summary of the MCP server workspace and in-memory jobs."""
    with _JOBS_LOCK:
        jobs = list(_JOBS.values())
    for job in jobs:
        JOB_REGISTRY.refresh(job)
    capabilities = _runtime_capabilities()
    transport = os.environ.get("KONFAI_MCP_TRANSPORT", "stdio")
    bearer_token = (os.environ.get("KONFAI_MCP_BEARER_TOKEN") or "").strip()
    return {
        "name": "KonfAI MCP",
        "konfai_version": KONFAI_VERSION,
        "transport": transport,
        "workspace_root": str(WORKSPACES_ROOT),
        "current_session": WORKSPACE_LAYOUT.current_session,
        "session_root": str(WORKSPACE_LAYOUT.workspace_dir()),
        "sessions": WORKSPACE_LAYOUT.available_sessions(),
        "session_workspace_initialized": WORKSPACE_LAYOUT.session_workspace_exists(),
        "jobs": {
            "total": len(jobs),
            "active": len([job for job in jobs if job.status in ACTIVE_JOB_STATES]),
        },
        "runtime": {
            "gpu_available": capabilities["gpu"]["available"],
            "visible_gpus": capabilities["gpu"]["visible_indices"],
            "recommended_device": capabilities["recommended_device"],
        },
        "auth": {
            "configured": bool(bearer_token),
            "scheme": "bearer" if bearer_token else None,
            "protected_transports": ["sse", "streamable-http"] if bearer_token else [],
            "enforced_on_current_transport": bool(bearer_token) and transport != "stdio",
        },
        "resources": {
            "capabilities": "server://capabilities",
            "tool_index": "guide://tool-index",
            "config_design": "guide://config-design",
            "docs_index": "docs://index",
        },
    }


@mcp.resource("server://capabilities")
def server_capabilities() -> dict[str, Any]:
    """Describe the runtime resources visible to the MCP server for device selection."""
    return _runtime_capabilities()


@mcp.resource("guide://tool-index")
async def read_tool_index() -> dict[str, Any]:
    """Read the guide to the MCP tool and prompt surface, generated from the registry so it can never drift."""
    tools = await mcp.list_tools()
    prompts = await mcp.list_prompts()
    return {
        "topic": "tool_index",
        "generated_from_registry": True,
        "tool_count": len(tools),
        "tools": {tool.name: tool.description for tool in sorted(tools, key=lambda tool: tool.name)},
        "prompts": {prompt.name: prompt.description for prompt in sorted(prompts, key=lambda prompt: prompt.name)},
    }


@mcp.resource("guide://config-design")
def read_config_design_summary() -> dict[str, Any]:
    """Read the compact KonfAI config-design summary first, then go into docs if needed."""
    return config_design_summary()


@mcp.resource("docs://index")
def read_docs_index() -> dict[str, Any]:
    """List the broader reasoning docs available through the MCP server."""
    return docs_index()


@mcp.resource("docs://patching")
def read_patching_doc() -> dict[str, Any]:
    """Read the detailed KonfAI patching doc."""
    return patching_rules()


@mcp.resource("docs://modeling")
def read_modeling_doc() -> dict[str, Any]:
    """Read the detailed KonfAI modeling doc."""
    return modeling_rules()


@mcp.resource("docs://configuration")
def read_configuration_doc() -> dict[str, Any]:
    """Read the detailed KonfAI configuration doc."""
    return configuration_rules()


@mcp.resource("docs://prediction")
def read_prediction_doc() -> dict[str, Any]:
    """Read the prediction authoring doc: TTA, multi-model ensembles, outputs_dataset reassembly."""
    return prediction_rules()


@mcp.resource("docs://compute")
def read_compute_doc() -> dict[str, Any]:
    """Read the compute doc: device selection, DDP semantics, memory knobs, SLURM submission."""
    return compute_rules()


@mcp.resource("docs://dataset-mapping")
def read_dataset_mapping_doc() -> dict[str, Any]:
    """Read the dataset-to-task mapping doc for clarifying inputs, targets, and support groups."""
    return dataset_mapping_doc()


@mcp.resource("docs://examples")
def read_examples_doc() -> dict[str, Any]:
    """Read how example templates should be used by an agent."""
    return examples_doc(EXAMPLES_ROOT)


@mcp.resource("templates://list")
def list_templates() -> list[str]:
    """List example templates that can seed the current session workspace."""
    return available_templates(EXAMPLES_ROOT)


@mcp.resource("template://{name}/summary")
def read_template_summary(name: str) -> dict[str, Any]:
    """Read the compact template summary, including config/model hints, before opening broader docs."""
    return template_guidance_summary(EXAMPLES_ROOT, name, WORKFLOWS)


@mcp.prompt(
    name="solve_task",
    description="Route a dataset+goal request: use an existing app, fine-tune one, or train from scratch.",
)
def prompt_solve_task(task: str, dataset_summary: str = "") -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": SOLVE_TASK_PROMPT.format(task=task, dataset_summary=dataset_summary or "(not provided)"),
        }
    ]


@mcp.prompt(
    name="clarify_task_and_groups",
    description="Ask the user the minimum questions needed to map dataset groups to a KonfAI task.",
)
def prompt_clarify_task_and_groups(task: str, dataset_summary: str = "") -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": CLARIFY_TASK_AND_GROUPS_PROMPT.format(
                task=task, dataset_summary=dataset_summary or "(not provided)"
            ),
        }
    ]


@mcp.prompt(
    name="plan_config_strategy",
    description="Plan a KonfAI config-writing strategy from task, dataset summary, and modeling intent.",
)
def prompt_design_config_strategy(
    task: str,
    dataset_summary: str,
    modeling_intent: str = "undecided",
) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": PLAN_CONFIG_STRATEGY_PROMPT.format(
                task=task, modeling_intent=modeling_intent, dataset_summary=dataset_summary
            ),
        }
    ]


@mcp.prompt(
    name="debug_config_warning",
    description="Reason about KonfAI semantic warnings and propose the next checks before editing YAML.",
)
def prompt_debug_config_warning(
    warning_summary: str,
    config_summary: str = "",
) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": DEBUG_CONFIG_WARNING_PROMPT.format(
                warning_summary=warning_summary, config_summary=config_summary or "(not provided)"
            ),
        }
    ]


@mcp.resource("sessions://list")
def list_sessions() -> list[str]:
    """List session workspaces available under the workspace root."""
    return WORKSPACE_LAYOUT.available_sessions()


@mcp.resource("session://current/summary")
def read_session_summary() -> dict[str, Any]:
    """Read the current summary for the current session workspace."""
    return SESSION.session_summary()


@mcp.resource("session://current/config/{workflow}")
def read_workflow_config_resource(workflow: WorkflowKind) -> str:
    """Read one current-session config chosen by workflow."""
    return read_text(SESSION.config_path(workflow))


@mcp.resource("session://current/log")
def read_session_log() -> str:
    """Read the latest available session log tail."""
    path = SESSION.discover_log_path()
    if path is None:
        return ""
    return _read_text_tail(path)


@mcp.resource("session://current/metrics")
def read_session_metrics() -> dict[str, Any]:
    """Read the latest evaluation metrics for the current session as structured JSON."""
    return SESSION.read_metrics_payload()


@mcp.resource("job://{job_id}/status")
def read_job_status(job_id: str) -> dict[str, Any]:
    """Read the latest status payload for one job."""
    return _job_payload(JOB_REGISTRY.get(job_id))


@mcp.resource("job://{job_id}/log")
def read_job_log_resource(job_id: str) -> str:
    """Read the latest job log tail."""
    return _read_text_tail(JOB_REGISTRY.get(job_id).log_path)


@mcp.resource("job://{job_id}/manifest")
def read_job_manifest(job_id: str) -> dict[str, Any]:
    """Read the immutable manifest captured when the job was launched."""
    job = JOB_REGISTRY.get(job_id)
    if job.manifest_path is None or not job.manifest_path.exists():
        return {}
    text = read_text(job.manifest_path)
    try:
        manifest = json.loads(text)
    except json.JSONDecodeError:
        manifest = {"raw": text}
    return {"path": str(job.manifest_path), "manifest": manifest}


@mcp.tool(description=(TOOL_DESCRIPTIONS["browse_dataset"]))
def browse_dataset(
    dataset_dir: Annotated[str, Field(description="Host path of the dataset directory (or parent of roots) to scan.")],
    depth: Annotated[int, Field(description="Maximum directory depth to walk (default 2).")] = 2,
    max_entries: Annotated[
        int, Field(description="Maximum tree entries returned before truncation (default 200).")
    ] = 200,
) -> dict[str, Any]:
    """Browse a dataset tree with bounded output and candidate dataset-root hints."""
    return SESSION.browse_dataset_payload(
        Path(dataset_dir).expanduser().resolve(),
        depth=depth,
        max_entries=max_entries,
    )


@mcp.tool(description=(TOOL_DESCRIPTIONS["read_dataset_file"]))
def read_dataset_file(
    path: Annotated[str, Field(description="Host path of the text file to preview (CSV/TSV/JSON/YAML/txt/header).")],
    max_lines: Annotated[int, Field(description="Maximum lines returned (default 200).")] = 200,
    max_chars: Annotated[int, Field(description="Streaming read cap in characters (default 65536).")] = 65536,
) -> dict[str, Any]:
    """Bounded read-only preview of a dataset sidecar/text file, with a structured CSV/TSV preview."""
    return read_dataset_sidecar(
        Path(path).expanduser().resolve(),
        max_lines=max_lines,
        max_chars=max_chars,
    )


def _statistics_failure_reason(group: str, extension: str, exc: Exception) -> str:
    """Explain why a scanned group produced no statistics, so the agent is never left with a silent gap."""
    detail = str(exc).strip()
    if isinstance(exc, DatasetGroupUnreadableError):
        cases = "; ".join(f"{name} -> {message}" for name, message in exc.case_errors.items())
        return (
            f"Group '{group}' is CORRUPT/UNREADABLE, not mis-structured: every sampled file exists on disk but "
            f"the '{extension}' backend reader raised on all of them. This is a file-content problem (truncated, "
            "empty, or non-image bytes) -- do NOT restructure the dataset or change the format token to fix it; "
            f"inspect or replace the offending files. Per-case reader errors: {cases}."
        )
    if isinstance(exc, ValueError) and "not found" in detail.lower():
        return (
            f"No readable cases for group '{group}' with format token '{extension}'. The structure scan found "
            "this group by filename, but the KonfAI dataset reader enumerated zero cases. Common causes: a flat "
            "directory (files are not per-case subdirectories), a DICOM series directory without series tags, an "
            "OME-Zarr/HDF5 store handed in as the root, or a token/backend mismatch -- the read backend is chosen "
            "by the format token (h5/omezarr/dicom vs SimpleITK), so a '.h5'/'.zarr'/DICOM group needs its own "
            "matching token (often via a separate dataset_filenames entry)."
        )
    return f"Statistics for group '{group}' could not be computed ({type(exc).__name__}): {detail or 'read failed'}."


@mcp.tool(description=(TOOL_DESCRIPTIONS["inspect_dataset"]))
def inspect_dataset(
    dataset_dir: Annotated[
        str, Field(description="Host path of the chosen dataset root (browse_dataset first when ambiguous).")
    ],
    groups: Annotated[
        list[str] | None, Field(description="Restrict statistics to these group names (default: every detected group).")
    ] = None,
    extension: Annotated[
        str | None,
        Field(
            description="Format token for the statistics reader (default: first detected extension; h5/omezarr/dicom select their backend BY this token)."
        ),
    ] = None,
    max_cases_per_group: Annotated[
        int, Field(description="Maximum cases sampled per group for statistics; <= 0 samples every case (default 10).")
    ] = 10,
    seed: Annotated[int, Field(description="Random seed for the per-group case sampling (default 0).")] = 0,
    include_stats: Annotated[
        bool, Field(description="False runs a fast structure-only scan without sampled statistics (default True).")
    ] = True,
) -> dict[str, Any]:
    """Inspect one chosen dataset root and return the structure an agent needs before designing configs."""
    dataset_path = Path(dataset_dir).expanduser().resolve()
    payload = SESSION.infer_dataset_structure_payload(dataset_path)
    if not include_stats:
        # The structure payload already carries its own next_actions (browse/inspect/design/init).
        return payload
    selected_groups = sorted(groups) if groups else sorted(payload["groups"])
    detected_extensions = payload["detected_extensions"]
    warnings: list[str] = []
    if extension is not None:
        selected_extension = extension
    elif detected_extensions:
        selected_extension = detected_extensions[0]
    else:
        selected_extension = "mha"
        warnings.append(
            "No supported dataset extension was detected; statistics defaulted to 'mha' as a supported output hint."
        )

    statistics: dict[str, Any] = {}
    missing_groups: list[str] = []
    missing_reasons: dict[str, str] = {}
    for group in selected_groups:
        try:
            group_statistics = SESSION.compute_dataset_group_statistics(
                dataset_path,
                group,
                extension=selected_extension,
                max_cases=max_cases_per_group if max_cases_per_group > 0 else None,
                seed=seed,
            )
        except Exception as exc:
            # Never drop a group's stats silently: the structure scan found this group, so an empty result
            # must carry a reason the agent can act on (wrong backend token, un-enumerable layout, corrupt file).
            missing_groups.append(group)
            missing_reasons[group] = _statistics_failure_reason(group, selected_extension, exc)
            continue
        # dataset_path/extension are identical for every group and already live at the top level.
        statistics[group] = {
            key: value for key, value in group_statistics.items() if key not in {"dataset_path", "extension"}
        }

    payload["statistics"] = statistics
    payload["statistics_groups"] = sorted(statistics)
    payload["statistics_missing_groups"] = missing_groups
    if missing_reasons:
        payload["statistics_missing_reasons"] = missing_reasons
    payload["statistics_extension"] = selected_extension
    payload["statistics_extension_note"] = (
        "Within the SimpleITK family (mha/nii/nrrd/gipl/hdr/...) reads are extension-agnostic and this token is "
        "only a hint. But h5/omezarr/dicom select a distinct backend BY this token: for those, the token must "
        "match the files (use a separate dataset_filenames entry per format/backend)."
    )
    payload["statistics_sample_limit"] = max_cases_per_group
    if warnings:
        payload["warnings"] = [*(payload.get("warnings") or []), *warnings]
    payload["next_actions"] = [
        "browse_dataset",
        "prepare_dataset_aliases",
        "design_config_strategy",
        "initialize_session",
    ]
    return round_floats(payload)


@mcp.tool(description=(TOOL_DESCRIPTIONS["design_config_strategy"]))
def design_config_strategy(
    task: Annotated[
        str, Field(description="User task description (e.g. 'segment the liver on CT'); must be non-empty.")
    ],
    dataset_dir: Annotated[
        str | None, Field(description="Single dataset root path (provide this or dataset_dirs).")
    ] = None,
    dataset_dirs: Annotated[
        str | list[str] | None,
        Field(description="One or several dataset root paths (string or list) merged into the strategy."),
    ] = None,
    group_roles: Annotated[
        dict[str, str] | None,
        Field(description="Mapping of dataset group name -> role: 'input', 'target', or 'support'."),
    ] = None,
    workflows: Annotated[
        str | list[str] | None,
        Field(
            description="Workflow name(s) to plan for: train/prediction/evaluation (string or list; default: all three)."
        ),
    ] = None,
    modeling_intent: Annotated[
        Literal["2d", "2.5d", "3d", "undecided"],
        Field(description="Spatial modeling mode driving the patching considerations (default 'undecided')."),
    ] = "undecided",
    example: Annotated[
        str | None, Field(description="Example template name to anchor the plan (must exist under examples/).")
    ] = None,
    extension: Annotated[
        str | None, Field(description="Format token for the dataset entry hints (default: first detected extension).")
    ] = None,
) -> dict[str, Any]:
    """Build a config-writing strategy from the user task and one or more inspected dataset roots."""
    dataset_path, dataset_paths = _normalize_dataset_inputs(dataset_dir, dataset_dirs)
    payload = SESSION.design_config_strategy_payload(
        dataset_path,
        task=task,
        dataset_dirs=dataset_paths,
        group_roles=group_roles,
        workflows=workflows,
        modeling_intent=modeling_intent,
        example=example,
        extension=extension,
    )
    return payload


def _classpath_requires_import(classpath: str) -> bool:
    """True when summarize_classpath_signature would import the classpath's module in-process.

    A shipped/local YAML model ('default|<Name>.yml' or '<Name>.yml') and a local workspace 'File:Class'
    (a single-token module before the colon, resolved as a workspace .py) are parsed statically and never
    imported. Everything else -- an installed 'package.module:Class', the colon form 'package:module:Class'
    (which _parse_classpath joins into a dotted module), or a bare/dotted builtin name -- triggers
    importlib.import_module, so it must run isolated. The check errs toward isolation: it stays in-process
    only for the two forms proven not to import, matching summarize_classpath_signature's own branching.
    """
    normalized = classpath.strip()
    if not normalized:
        return False  # let summarize_classpath_signature raise its own ValueError in-process
    if normalized.endswith(".yml") or normalized.startswith("default|"):
        return False
    # Local 'File:Class' == exactly one colon and a single-token module with no dot (mirrors
    # _parse_classpath + local_candidate). 'torch:nn:L1Loss' has two colons -> module 'torch.nn' -> imports.
    colon_parts = [part for part in normalized.split(":") if part]
    if len(colon_parts) == 2 and "." not in colon_parts[0]:
        return False
    return True


@mcp.tool(description=(TOOL_DESCRIPTIONS["inspect_object_signature"]))
def inspect_object_signature(
    classpath: Annotated[
        str,
        Field(
            description="Builtin name, local 'File:Class' (resolved in the session workspace, parsed statically), or installed 'package.module:Class' (imported in an isolated subprocess)."
        ),
    ],
) -> dict[str, Any]:
    """Inspect one configurable object classpath and return signature details when possible."""
    workspace_dir = WORKSPACE_LAYOUT.workspace_dir()
    if _classpath_requires_import(classpath):
        # Reading an installed-library signature imports its module; run that in a spawn subprocess so a
        # heavy/slow/crashing/OOM import cannot take down the long-lived server (AGENTS.md: code that
        # imports/executes runs isolated). Static YAML-model and local File:Class classpaths never import,
        # so they stay in-process to avoid the multi-second spawn cost on the common path.
        summary = _run_api_in_subprocess(
            "konfai_mcp.server_support:summarize_classpath_signature",
            {"classpath": classpath, "workspace_dir": workspace_dir},
        )
    else:
        summary = summarize_classpath_signature(classpath, workspace_dir=workspace_dir)
    return {
        **summary,
        "session": WORKSPACE_LAYOUT.current_session,
        "next_actions": (
            ["review_config_semantics", "validate_config_semantics"]
            if summary.get("ok")
            else ["write_session_file", "write_workflow_config", "summarize_session"]
        ),
    }


@mcp.tool(description=(TOOL_DESCRIPTIONS["list_components"]))
def list_components(
    kind: Annotated[
        str,
        Field(
            description=f"Component kind, one of {COMPONENT_KINDS}; aliases accepted (loss/metric -> criterion, plurals)."
        ),
    ],
) -> dict[str, Any]:
    """Enumerate the available KonfAI components of one kind (loss/metric/transform/augmentation/scheduler/model/block)."""
    return {**_catalog_list_components(kind), "session": WORKSPACE_LAYOUT.current_session}


@mcp.tool(description=(TOOL_DESCRIPTIONS["describe_konfai_capabilities"]))
def describe_konfai_capabilities() -> dict[str, Any]:
    """Overview of what KonfAI can do and which MCP primitive to use for each capability."""
    return _describe_konfai_capabilities()


@mcp.tool(description=(TOOL_DESCRIPTIONS["describe_config_schema"]))
def describe_config_schema(
    workflow: Annotated[
        str,
        Field(
            description="Workflow whose schema to describe: 'train', 'prediction', or 'evaluation' (aliases like 'trainer'/'predict'/'eval' accepted)."
        ),
    ],
    path: Annotated[
        str | None,
        Field(
            description="Dot-separated YAML keys drilling into a nested config level (e.g. 'Dataset.Patch' or 'Model'); omit for the top level."
        ),
    ] = None,
) -> dict[str, Any]:
    """Describe one level of a workflow's config schema, generated from the reflection engine (no drift)."""
    return _describe_config_schema(workflow, path=path)


@mcp.tool(description=(TOOL_DESCRIPTIONS["describe_extension_points"]))
def describe_extension_points(
    kind: Annotated[
        str | None,
        Field(
            description="Extension kind: loss/metric/model/augmentation/transform/scheduler/pretrained (aliases accepted); omit for all."
        ),
    ] = None,
) -> dict[str, Any]:
    """Describe how to plug a new/external component into KonfAI (base class, contract, classpath syntax)."""
    return _describe_extension_points(kind)


@mcp.tool(description=(TOOL_DESCRIPTIONS["check_external_dependency"]))
def check_external_dependency(
    module: Annotated[
        str,
        Field(
            description="Module or dotted path to check (e.g. 'monai' or 'monai.losses'); only the top-level package is inspected."
        ),
    ],
    object_name: Annotated[
        str | None,
        Field(
            description="Object name inside the module, echoed into the returned inspect_classpath (existence is not verified)."
        ),
    ] = None,
) -> dict[str, Any]:
    """Check importability/version/license of an external library before integrating a brick from it."""
    return _check_external_dependency(module, object_name)


@mcp.resource("apps://catalog")
def apps_catalog_resource() -> dict[str, Any]:
    """Return the resolved app-source catalogue (shipped default + workspace file + env override)."""
    refs, provenance = APP_SERVICE.resolve_catalog()
    return {"apps": refs, "count": len(refs), "sources": provenance}


@mcp.tool(description=(TOOL_DESCRIPTIONS["list_apps"]))
def list_apps(
    repos: Annotated[
        list[str] | None,
        Field(
            description="Ad-hoc override list of app references / HuggingFace repo ids checked instead of the catalogue."
        ),
    ] = None,
    include_summary: Annotated[
        bool,
        Field(
            description="Also resolve each app's manifest for display_name/short_description/inputs/outputs (slower, best-effort)."
        ),
    ] = False,
    force_update: Annotated[
        bool, Field(description="Refresh cached HuggingFace listings instead of reusing the local cache.")
    ] = False,
) -> dict[str, Any]:
    """List published KonfAI apps from the referenced catalogue so the agent can pick one before training."""
    return {**APP_SERVICE.list_apps(repos=repos, include_summary=include_summary, force_update=force_update)}


@mcp.tool(description=(TOOL_DESCRIPTIONS["describe_app"]))
def describe_app(
    ref: Annotated[
        str,
        Field(
            description="App id 'repo_id:app_name', local app folder path, or remote 'host:port:name[|token]' (bare HuggingFace repo ids are not accepted; expand with list_apps)."
        ),
    ],
    force_update: Annotated[
        bool, Field(description="Re-download the app manifest instead of reusing the local cache.")
    ] = False,
) -> dict[str, Any]:
    """Read one app's app.json manifest (modality, capabilities, checkpoints) so the agent can pick it."""
    return APP_SERVICE.describe_app(ref, force_update=force_update)


@mcp.tool(description=(TOOL_DESCRIPTIONS["list_app_parameters"]))
def list_app_parameters(
    ref: Annotated[
        str,
        Field(
            description="App id 'repo_id:app_name' or local app folder path (remote servers do not expose parameters)."
        ),
    ],
    allow_untrusted_code: Annotated[
        bool,
        Field(
            description="Must be True: deriving constraints imports the app's model code (in an isolated spawn subprocess)."
        ),
    ] = False,
    force_update: Annotated[
        bool, Field(description="Re-download the app files instead of reusing the local cache.")
    ] = False,
) -> dict[str, Any]:
    """Read an app's tunable parameters and constraints so set_parameters can be used with intent."""
    return APP_SERVICE.list_parameters(ref, allow_untrusted_code=allow_untrusted_code, force_update=force_update)


@mcp.tool(description=(TOOL_DESCRIPTIONS["export_app"]))
def export_app(
    ref: Annotated[
        str,
        Field(description="App id 'repo_id:app_name' or local app folder path (remote servers cannot be exported)."),
    ],
    path: Annotated[str, Field(description="Destination folder for the exported app bundle.")],
    display_name: Annotated[
        str | None, Field(description="Override the display name written into the exported app.json.")
    ] = None,
    set_parameters: Annotated[
        dict[str, Any] | None, Field(description="Model parameter NAME->VALUE overrides baked into the copied config.")
    ] = None,
    force_update: Annotated[
        bool, Field(description="Re-download the app files instead of reusing the local cache.")
    ] = False,
) -> dict[str, Any]:
    """Save a resolved app (optionally with tuned parameters) as a local, editable bundle."""
    # json.dumps keeps each value's type through the YAML re-parse in _apply_config_overrides: an int
    # stays an int, but a string "true"/"1" stays a string instead of being coerced to a bool/int.
    config_overrides = (
        [f"{name}={json.dumps(value)}" for name, value in set_parameters.items()] if set_parameters else None
    )
    return APP_SERVICE.export_app(
        ref, path, display_name=display_name, config_overrides=config_overrides, force_update=force_update
    )


@mcp.tool(description=(TOOL_DESCRIPTIONS["import_app"]))
def import_app(
    ref: Annotated[
        str, Field(description="App id 'repo_id:app_name' or local app folder path (remote servers cannot be imported).")
    ],
    allow_untrusted_code: Annotated[
        bool,
        Field(
            description="Confirm you trust this app: importing copies+runs its Python code and pip-installs its requirements."
        ),
    ] = False,
    display_name: Annotated[
        str | None, Field(description="Override the display name written into the copied app.json.")
    ] = None,
    set_parameters: Annotated[
        dict[str, Any] | None,
        Field(description="Model parameter NAME->VALUE overrides baked into the copied Prediction.yml."),
    ] = None,
    force_update: Annotated[
        bool, Field(description="Re-download the app files instead of reusing the local cache.")
    ] = False,
) -> dict[str, Any]:
    """Import a resolved app into the session as a normal KonfAI experiment (config + code + checkpoints)."""
    config_overrides = (
        [f"{name}={json.dumps(value)}" for name, value in set_parameters.items()] if set_parameters else None
    )
    return APP_SERVICE.import_app(
        ref,
        allow_untrusted_code=allow_untrusted_code,
        display_name=display_name,
        config_overrides=config_overrides,
        force_update=force_update,
    )


@mcp.tool(description=(TOOL_DESCRIPTIONS["register_app_source"]))
def register_app_source(
    ref: Annotated[
        str,
        Field(
            description="App reference to persist: an app id 'repo_id:app_name' or a bare HuggingFace repo_id (not validated here)."
        ),
    ],
) -> dict[str, Any]:
    """Add an app reference to the editable workspace app catalogue so it persists across sessions."""
    return APP_SERVICE.register_app_source(ref)


@mcp.tool(description=(TOOL_DESCRIPTIONS["unregister_app_source"]))
def unregister_app_source(
    ref: Annotated[str, Field(description="App reference to remove from the editable workspace catalogue file.")],
) -> dict[str, Any]:
    """Remove an app reference from the editable workspace app catalogue."""
    return APP_SERVICE.unregister_app_source(ref)


@mcp.tool(description=(TOOL_DESCRIPTIONS["package_app_from_session"]))
def package_app_from_session(
    name: Annotated[str, Field(description="Bundle folder name (sanitized to a filesystem-safe form).")],
    display_name: Annotated[str, Field(description="Human-readable app name written into app.json.")],
    description: Annotated[str, Field(description="Full app description written into app.json.")],
    short_description: Annotated[
        str | None, Field(description="Short description for app.json (default: display_name).")
    ] = None,
    tta: Annotated[
        int, Field(description="Maximum test-time-augmentation count declared in app.json (default 0).")
    ] = 0,
    mc_dropout: Annotated[int, Field(description="MC-dropout sample count declared in app.json (default 0).")] = 0,
    checkpoints: Annotated[
        list[str] | None,
        Field(description="Checkpoint paths to package (default: discovered from the session Checkpoints/)."),
    ] = None,
    configs: Annotated[
        list[str] | None, Field(description="Config paths to package (default: the session Prediction.yml/Config.yml).")
    ] = None,
    model_py: Annotated[str | None, Field(description="Path to a Model.py to ship with the bundle.")] = None,
    requirements: Annotated[
        str | None, Field(description="Path to a requirements.txt to ship with the bundle.")
    ] = None,
    output: Annotated[
        str | None,
        Field(description="Destination directory for the bundle (default: the session workspace AppBundles/)."),
    ] = None,
    onnx: Annotated[
        bool,
        Field(
            description="Also export the newest packaged checkpoint as model.onnx inside the bundle (default False)."
        ),
    ] = False,
    onnx_patch_size: Annotated[
        list[int] | None, Field(description="Patch size for the ONNX export's dummy input.")
    ] = None,
    onnx_in_channels: Annotated[
        int | None, Field(description="Input channel count for the ONNX export's dummy input.")
    ] = None,
) -> dict[str, Any]:
    """Package a session-trained model into a resolvable KonfAI app bundle (optionally with ONNX)."""
    return APP_SERVICE.package_from_session(
        name=name,
        display_name=display_name,
        description=description,
        short_description=short_description,
        tta=tta,
        mc_dropout=mc_dropout,
        checkpoints=checkpoints,
        configs=configs,
        model_py=model_py,
        requirements=requirements,
        output=output,
        onnx=onnx,
        onnx_patch_size=onnx_patch_size,
        onnx_in_channels=onnx_in_channels,
    )


@mcp.tool(description=(TOOL_DESCRIPTIONS["prepare_dataset_aliases"]))
def prepare_dataset_aliases(
    dataset_dir: Annotated[str, Field(description="Host path of the KonfAI-style dataset to alias in place.")],
    rename_map: Annotated[
        dict[str, str],
        Field(description="Mapping source group name -> target group name (e.g. {'IMG': 'CT'}), applied per case."),
    ],
    mode: Annotated[
        Literal["copy", "symlink", "move"],
        Field(
            description="How aliases are materialised (default 'copy'); 'move' additionally needs allow_destructive=True."
        ),
    ] = "copy",
    overwrite: Annotated[
        bool, Field(description="Replace an existing target file instead of skipping it (default False).")
    ] = False,
    allow_destructive: Annotated[
        bool, Field(description="Must be True to confirm mode='move' (the source filename disappears).")
    ] = False,
) -> dict[str, Any]:
    """Create renamed group files inside an existing KonfAI-style dataset.

    Typical use: copy or symlink `IMG.mha` to `CT.mha` in every case so an
    existing template can be reused without rewriting the whole dataset.
    """
    dataset_path = Path(dataset_dir).expanduser().resolve()
    if not dataset_path.exists():
        raise ValueError(f"Dataset directory not found: {dataset_path}")
    if not rename_map:
        raise ValueError("rename_map cannot be empty.")
    if mode == "move" and not allow_destructive:
        raise ValueError("mode='move' is destructive. Set allow_destructive=True to confirm.")
    # Both group names index files by name inside a case directory; a separator or '..' would let the
    # target escape the dataset (an arbitrary-write primitive). Require bare path components.
    for role, groups in (("source", rename_map.keys()), ("target", rename_map.values())):
        for group in groups:
            if not group or Path(group).name != group or os.sep in group or (os.altsep and os.altsep in group):
                raise ValueError(
                    f"Invalid {role} group name '{group}'. Expected a bare filename component (no path separators)."
                )

    case_dirs = case_directories(dataset_path)
    created: list[str] = []
    missing_by_case: dict[str, list[str]] = {}
    skipped_existing: list[str] = []

    for case_dir in case_dirs:
        case_name = case_dir.name if case_dir != dataset_path else "."
        for source_group, target_group in rename_map.items():
            matches = sorted(path for path in case_dir.glob(f"{source_group}.*") if path.is_file())
            if not matches:
                missing_by_case.setdefault(case_name, []).append(source_group)
                continue
            if len(matches) > 1:
                raise ValueError(
                    f"Ambiguous source group '{source_group}' in '{case_dir}': {[path.name for path in matches]}"
                )

            source = matches[0]
            target = case_dir / f"{target_group}{''.join(source.suffixes)}"
            if target.exists():
                if not overwrite:
                    skipped_existing.append(str(target))
                    continue
                if target.is_symlink() or target.is_file():
                    target.unlink()
                else:
                    raise ValueError(f"Refusing to overwrite non-file target: {target}")

            if mode == "copy":
                shutil.copy2(source, target)
            elif mode == "symlink":
                os.symlink(source.name, target)
            else:
                shutil.move(str(source), str(target))
            created.append(str(target))

    return {
        "dataset_dir": str(dataset_path),
        "mode": mode,
        "rename_map": rename_map,
        "created_count": len(created),
        "created": created[:50],
        "skipped_existing": skipped_existing[:50],
        "missing_by_case": missing_by_case,
        "next_actions": [
            "inspect_dataset",
            "design_config_strategy",
            "initialize_session",
        ],
    }


@mcp.tool(description=(TOOL_DESCRIPTIONS["initialize_session"]))
def initialize_session(
    overwrite: Annotated[
        bool,
        Field(
            description="DESTRUCTIVE: True deletes the entire existing workspace (Checkpoints/Predictions included) before recreating it."
        ),
    ] = False,
    from_example: Annotated[
        str | None, Field(description="Example template name to seed workflow files from (see templates://list).")
    ] = None,
    workflows: Annotated[
        str | list[str] | None,
        Field(
            description="Workflow files to seed from the example: train/prediction/evaluation (string or list; default: train only)."
        ),
    ] = None,
    include_support_files: Annotated[
        bool,
        Field(
            description="Also copy the example's local .py support files (required when its model/loss lives in a local .py)."
        ),
    ] = False,
) -> dict[str, Any]:
    """Create or reset the current session workspace, optionally seeded from one example template."""
    workspace = WORKSPACE_LAYOUT.ensure_session_workspace()
    if workspace.exists() and any(workspace.iterdir()):
        if not overwrite:
            raise ValueError(
                f"Session workspace already exists for '{WORKSPACE_LAYOUT.current_session}'. "
                "overwrite=True DELETES its entire contents (Checkpoints/Predictions included); "
                "use switch_session to start a new session and keep this one."
            )
        SESSION.ensure_no_active_job()
        shutil.rmtree(workspace)
        workspace.mkdir(parents=True, exist_ok=True)
    WORKSPACE_LAYOUT.jobs_dir().mkdir(parents=True, exist_ok=True)

    copied_files: list[str] = []
    skipped_python: list[str] = []
    selected_workflows = SESSION.normalize_requested_workflows(workflows) if from_example else []
    if from_example is not None:
        copied_files, skipped_python = copy_template_subset(
            workspace,
            template_dir(EXAMPLES_ROOT, from_example),
            overwrite,
            include_python=include_support_files,
            workflows=selected_workflows,
            include_support_files=include_support_files,
        )

    payload: dict[str, Any] = {
        "session": WORKSPACE_LAYOUT.current_session,
        "path": str(workspace),
        "session_path": str(WORKSPACE_LAYOUT.workspace_dir()),
        "seeded_from_example": from_example,
        "workflows": selected_workflows,
        "copied_files": copied_files,
        "resources": (
            SESSION.session_summary()["resources"]
            if any(workspace.iterdir())
            else {
                "configs": {workflow: f"session://current/config/{workflow}" for workflow in WORKFLOW_CONFIG_FILES},
                "log": "session://current/log",
                "metrics": "session://current/metrics",
                "summary": "session://current/summary",
            }
        ),
        "next_actions": ["write_workflow_config", "review_config_semantics", "summarize_session"],
    }
    if skipped_python:
        payload["warnings"] = [
            f"The example's configs reference local Python file(s) {skipped_python} that were NOT copied "
            "(include_support_files=False), so those classpaths will not resolve. Re-run with "
            "include_support_files=True (and check their external dependencies with "
            "check_external_dependency), or replace them with your own components."
        ]
    return payload


@mcp.tool(description=(TOOL_DESCRIPTIONS["create_session"]))
def create_session(
    name: Annotated[str, Field(description="Session name (sanitized to a filesystem-safe form).")],
    switch: Annotated[
        bool, Field(description="Also make the new session current (refused while a job is active; default True).")
    ] = True,
) -> dict[str, Any]:
    """Create a named session workspace and (by default) switch the server onto it."""
    safe = WORKSPACE_LAYOUT.sanitize_name(name)
    session_dir = WORKSPACE_LAYOUT.session_dir(safe)
    created = not session_dir.exists()
    session_dir.mkdir(parents=True, exist_ok=True)
    if switch and safe != WORKSPACE_LAYOUT.current_session:
        SESSION.ensure_no_active_job()
        _activate_session(safe)
        WORKSPACE_LAYOUT.ensure_session_workspace()
    return {
        "session": safe,
        "created": created,
        "switched": switch,
        "sessions": WORKSPACE_LAYOUT.available_sessions(),
        "next_actions": ["initialize_session", "import_experiment", "summarize_session"],
    }


@mcp.tool(description=(TOOL_DESCRIPTIONS["switch_session"]))
def switch_session(
    name: Annotated[
        str,
        Field(description="Existing session name to switch onto (see sessions://list; create_session makes new ones)."),
    ],
) -> dict[str, Any]:
    """Switch the current session workspace (job history reloads from that session's disk state)."""
    safe = WORKSPACE_LAYOUT.sanitize_name(name)
    if safe not in WORKSPACE_LAYOUT.available_sessions():
        raise ValueError(
            f"Unknown session '{safe}'. Available sessions: {WORKSPACE_LAYOUT.available_sessions()}. "
            "Use create_session to create one."
        )
    if safe != WORKSPACE_LAYOUT.current_session:
        SESSION.ensure_no_active_job()
        _activate_session(safe)
        WORKSPACE_LAYOUT.ensure_session_workspace()
    return {
        "session": safe,
        "sessions": WORKSPACE_LAYOUT.available_sessions(),
        "summary": SESSION.session_summary(),
        "next_actions": ["summarize_session", "leaderboard", "run_train"],
    }


@mcp.tool(description=(TOOL_DESCRIPTIONS["import_experiment"]))
def import_experiment(
    source_dir: Annotated[
        str,
        Field(
            description="Host path of the KonfAI experiment directory to import (must be outside the session workspace)."
        ),
    ],
    include_artifacts: Annotated[
        Literal["link", "copy", "none"],
        Field(
            description="How artifact dirs (Checkpoints/Predictions/...) are imported: 'link' symlinks (default), 'copy' copies, 'none' imports configs/code only."
        ),
    ] = "link",
    overwrite: Annotated[
        bool, Field(description="Replace existing session files of the same name (default False: they are skipped).")
    ] = False,
) -> dict[str, Any]:
    """Import an existing KonfAI experiment directory into the current session workspace."""
    source = Path(source_dir).expanduser().resolve()
    if not source.is_dir():
        raise ValueError(f"Experiment directory not found: {source}")
    workspace = WORKSPACE_LAYOUT.ensure_session_workspace()
    if source == workspace or workspace in source.parents:
        raise ValueError("source_dir must be outside the session workspace.")
    artifact_dirs = {"Checkpoints", "Predictions", "Evaluations", "Statistics", "Dataset", "Cluster"}
    copied: list[str] = []
    linked: list[str] = []
    skipped: list[str] = []
    for path in sorted(source.iterdir(), key=lambda entry: entry.name):
        target = workspace / path.name
        if path.is_file() and path.suffix.lower() in {".yml", ".yaml", ".py", ".txt", ".json"}:
            if target.exists() and not overwrite:
                skipped.append(path.name)
                continue
            shutil.copy2(path, target)
            copied.append(path.name)
        elif path.is_dir() and path.name in artifact_dirs:
            if include_artifacts == "none":
                skipped.append(path.name)
                continue
            if target.exists():
                skipped.append(path.name)
                continue
            if include_artifacts == "link":
                os.symlink(path, target, target_is_directory=True)
                linked.append(path.name)
            else:
                shutil.copytree(path, target)
                copied.append(path.name)
    return {
        "session": WORKSPACE_LAYOUT.current_session,
        "source": str(source),
        "copied": copied,
        "linked": linked,
        "skipped": skipped,
        "next_actions": [
            "read_session_file",
            "review_config_semantics",
            "validate_config_semantics",
            "summarize_session",
        ],
    }


@mcp.tool(description=(TOOL_DESCRIPTIONS["write_session_file"]))
def write_session_file(
    relative_path: Annotated[
        str, Field(description="Destination path relative to the session workspace (parent dirs are created).")
    ],
    content: Annotated[
        str, Field(description="Full file content to write (trailing whitespace is normalized to one newline).")
    ],
    overwrite: Annotated[
        bool, Field(description="Replace an existing file (default True); False raises if the file exists.")
    ] = True,
) -> dict[str, Any]:
    """Write one support file inside the current session workspace."""
    path = WORKSPACE_LAYOUT.resolve_workspace_relative_path(relative_path)
    if path.exists() and not overwrite:
        raise ValueError(f"{path.name} already exists. Set overwrite=True to replace.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")
    return {
        "written": str(path.relative_to(WORKSPACE_LAYOUT.workspace_dir())),
        "path": str(path),
        "bytes": path.stat().st_size,
        "next_actions": ["inspect_object_signature", "review_config_semantics", "validate_config_semantics"],
    }


@mcp.tool(description=(TOOL_DESCRIPTIONS["read_session_file"]))
def read_session_file(
    path: Annotated[
        str, Field(description="Workspace-relative path, or an absolute path resolving inside the session workspace.")
    ],
    max_chars: Annotated[int, Field(description="Maximum characters returned (default 20000).")] = 20000,
    offset: Annotated[int, Field(description="Character offset to start reading from, for paging (default 0).")] = 0,
) -> dict[str, Any]:
    """Read one file inside the current session workspace (the read mirror of write_session_file)."""
    resolved = WORKSPACE_LAYOUT.resolve_workspace_relative_path(path)
    payload = read_text_range(resolved, max_chars=max_chars, offset=offset)
    return {
        **payload,
        "relative_path": str(resolved.relative_to(WORKSPACE_LAYOUT.workspace_dir())),
        "session": WORKSPACE_LAYOUT.current_session,
        "next_actions": ["write_session_file", "write_workflow_config", "review_config_semantics"],
    }


@mcp.tool(description=(TOOL_DESCRIPTIONS["read_template_file"]))
def read_template_file(
    name: Annotated[str, Field(description="Example template name (see templates://list).")],
    filename: Annotated[str, Field(description="File to read; must be a direct child of the template directory.")],
    max_chars: Annotated[int, Field(description="Maximum characters returned (default 20000).")] = 20000,
    offset: Annotated[int, Field(description="Character offset to start reading from, for paging (default 0).")] = 0,
) -> dict[str, Any]:
    """Read one file from an example template directory (reference code/config source)."""
    template = template_dir(EXAMPLES_ROOT, name)
    if not filename or Path(filename).name != filename or filename in {".", ".."}:
        raise ValueError(f"Invalid template filename '{filename}'. Expected a direct child of the template.")
    target = template / filename
    if not target.is_file():
        available = sorted(path.name for path in template.iterdir() if path.is_file())
        raise ValueError(f"File '{filename}' not found in template '{name}'. Available files: {available}")
    payload = read_text_range(target, max_chars=max_chars, offset=offset)
    return {
        **payload,
        "template": name,
        "filename": filename,
        "next_actions": ["write_session_file", "initialize_session", "write_workflow_config"],
    }


@mcp.tool(description=(TOOL_DESCRIPTIONS["get_run_metrics"]))
def get_run_metrics(
    run_name: Annotated[
        str,
        Field(
            description="The run's train_name (the Evaluations/<run_name> folder) — or an app-trial label as "
            "returned by leaderboard."
        ),
    ],
    split: Annotated[
        str,
        Field(
            description="Metric split, uppercased to Metric_<SPLIT>.json (default 'TRAIN'; a miss lists available splits)."
        ),
    ] = "TRAIN",
    session: Annotated[
        str | None, Field(description="Read another session's run without switching (default: the current session).")
    ] = None,
) -> dict[str, Any]:
    """Read one specific run's evaluation metric JSON (per-case and aggregates)."""
    return round_floats(SESSION.run_metrics_payload(run_name, split=split, session=session))


@mcp.tool(description=(TOOL_DESCRIPTIONS["compare_runs"]))
def compare_runs(
    run_a: Annotated[
        str,
        Field(description="train_name or app-trial label of the baseline run (deltas are reported as B minus A)."),
    ],
    run_b: Annotated[str, Field(description="train_name or app-trial label of the run compared against run_a.")],
    split: Annotated[
        str, Field(description="Metric split, uppercased to Metric_<SPLIT>.json (default 'TRAIN').")
    ] = "TRAIN",
    metric: Annotated[
        str | None,
        Field(description="Case-insensitive metric-name suffix filter (default: every metric common to both runs)."),
    ] = None,
    session: Annotated[
        str | None, Field(description="Read another session's runs without switching (default: the current session).")
    ] = None,
) -> dict[str, Any]:
    """Compare two runs' evaluation metrics case-by-case, direction-aware."""
    return SESSION.compare_runs_payload(run_a, run_b, split=split, metric=metric, session=session)


@mcp.tool(description=(TOOL_DESCRIPTIONS["read_training_curves"]))
def read_training_curves(
    run_name: Annotated[str, Field(description="The run's train_name (the Statistics/<run_name> folder).")],
    tags: Annotated[
        list[str] | None, Field(description="Substring filters on scalar tag names (default: every tag).")
    ] = None,
    max_points: Annotated[
        int, Field(description="Downsampling cap per curve; the last event is always kept (default 200).")
    ] = 200,
    session: Annotated[
        str | None, Field(description="Read another session's run without switching (default: the current session).")
    ] = None,
) -> dict[str, Any]:
    """Read a run's TensorBoard scalar curves as downsampled series."""
    return SESSION.training_curves_payload(run_name, tags=tags, max_points=max_points, session=session)


@mcp.tool(description=(TOOL_DESCRIPTIONS["export_run_record"]))
def export_run_record(
    run_name: Annotated[
        str | None, Field(description="Run to export (its newest recorded job is used); provide this or job_id.")
    ] = None,
    job_id: Annotated[str | None, Field(description="Exact job to export (takes precedence over run_name).")] = None,
    log_lines: Annotated[int, Field(description="Number of log lines in the returned tail (default 100).")] = 100,
) -> dict[str, Any]:
    """Assemble one run's complete provenance record (manifest + configs + metrics + environment + log)."""
    if job_id is not None:
        job = JOB_REGISTRY.get(job_id)
    else:
        if run_name is None:
            raise ValueError("Provide run_name or job_id.")
        with _JOBS_LOCK:
            candidates = [job for job in _JOBS.values() if job.run_name == run_name]
        if not candidates:
            with _JOBS_LOCK:
                known = sorted({job.run_name for job in _JOBS.values() if job.run_name})
            raise ValueError(f"No job recorded for run '{run_name}'. Known runs: {known or 'none'}.")
        job = max(candidates, key=lambda candidate: candidate.created_at)
    manifest: dict[str, Any] = {}
    if job.manifest_path is not None and job.manifest_path.exists():
        manifest = json.loads(read_text(job.manifest_path))
    snapshots = {
        name: read_text_range(Path(path), max_chars=40000)["content"]
        for name, path in (manifest.get("config_snapshots") or {}).items()
        if Path(path).exists()
    }
    resolved_run = job.run_name or run_name
    metrics: dict[str, Any] = {}
    if resolved_run:
        for split in SESSION._available_metric_splits():
            try:
                metrics[split] = SESSION.run_metrics_payload(resolved_run, split=split)["metrics"]
            except ValueError:
                continue
    return {
        "run_name": resolved_run,
        "job": _job_payload(job),
        "manifest": manifest,
        "config_snapshots": snapshots,
        "resolved_config": (
            read_text_range(job.config_path, max_chars=40000)["content"] if job.config_path.exists() else None
        ),
        "metrics": metrics,
        "log_tail": _read_text_tail(job.log_path, max_lines=max(log_lines, 1)),
        "next_actions": ["compare_runs", "read_training_curves", "leaderboard"],
    }


@mcp.tool(description=(TOOL_DESCRIPTIONS["diff_run_configs"]))
def diff_run_configs(
    job_id_a: Annotated[str, Field(description="Job whose config snapshot is the diff's 'from' side.")],
    job_id_b: Annotated[str, Field(description="Job whose config snapshot is the diff's 'to' side.")],
    filename: Annotated[
        str, Field(description="Which snapshotted config file to diff (default 'Config.yml').")
    ] = "Config.yml",
) -> dict[str, Any]:
    """Unified-diff the launch-time config snapshots of two jobs."""

    def snapshot_text(job_id: str) -> str:
        job = JOB_REGISTRY.get(job_id)
        if job.manifest_path is None or not job.manifest_path.exists():
            raise ValueError(f"No manifest recorded for job '{job_id}'.")
        manifest = json.loads(read_text(job.manifest_path))
        snapshots = manifest.get("config_snapshots") or {}
        if filename not in snapshots:
            raise ValueError(f"No '{filename}' snapshot for job '{job_id}'. Available: {sorted(snapshots)}.")
        return Path(snapshots[filename]).read_text(encoding="utf-8", errors="replace")

    text_a = snapshot_text(job_id_a)
    text_b = snapshot_text(job_id_b)
    diff_lines = list(
        difflib.unified_diff(
            text_a.splitlines(),
            text_b.splitlines(),
            fromfile=f"{job_id_a}/{filename}",
            tofile=f"{job_id_b}/{filename}",
            lineterm="",
        )
    )
    return {
        "filename": filename,
        "job_a": job_id_a,
        "job_b": job_id_b,
        "identical": not diff_lines,
        "diff": "\n".join(diff_lines),
        "next_actions": ["compare_runs", "export_run_record"],
    }


@mcp.tool(description=(TOOL_DESCRIPTIONS["describe_model_outputs"]))
def describe_model_outputs(
    workflow: Annotated[
        str,
        Field(
            description="Workflow whose model to build: 'train', 'prediction', or 'evaluation' (case-insensitive; default 'train')."
        ),
    ] = "train",
    config_file: Annotated[
        str | None,
        Field(description="Alternate train config filename in the session workspace (workflow='train' only)."),
    ] = None,
) -> dict[str, Any]:
    """List the valid outputs_criterions module paths by building the configured model."""
    if config_file is not None and workflow.strip().lower() != "train":
        raise ValueError("config_file is only supported for workflow='train'.")
    result = SESSION.validate_semantics(
        workflow,
        "instantiate",
        None,
        config_path=_resolve_train_config(config_file) if config_file is not None else None,
        collect_model_outputs=True,
    )
    if not result.get("ok", False):
        result["hint"] = "The config must build before outputs can be enumerated - fix the reported error first."
        return result
    return {
        "ok": True,
        "workflow": result.get("workflow", workflow),
        "networks": result.get("model_outputs") or {},
        "reference_hint": (
            "Use these paths as outputs_criterions / outputs_dataset keys (':' and '.' separators are "
            "equivalent). Attach the main loss to a terminal path; deep-supervision or perceptual losses may "
            "target intermediate paths."
        ),
        "next_actions": ["write_workflow_config", "validate_config_semantics"],
    }


@mcp.tool(description=(TOOL_DESCRIPTIONS["run_component_smoke_test"]))
def run_component_smoke_test(
    classpath: Annotated[
        str, Field(description="Component to test: local 'File:Class', builtin name, or 'package.module:Class'.")
    ],
    kind: Annotated[
        Literal["transform", "criterion", "loss", "metric"],
        Field(
            description="Contract to exercise: 'transform' checks the shape contract; 'criterion'/'loss'/'metric' check the return convention and backward()."
        ),
    ],
    shape: Annotated[
        list[int] | None, Field(description="Dummy tensor shape, channel-first (default [1, 8, 8, 8]).")
    ] = None,
    init_kwargs: Annotated[
        dict[str, Any] | None, Field(description="Constructor keyword arguments for the component.")
    ] = None,
) -> dict[str, Any]:
    """Execute one component's runtime contract on dummy tensors (shape contract / return convention)."""
    workspace = WORKSPACE_LAYOUT.ensure_session_workspace_exists()
    result = _run_api_in_subprocess(
        "konfai_mcp.runner:smoke_test_component",
        {
            "classpath": classpath,
            "kind": kind,
            "workspace_dir": str(workspace),
            "shape": shape,
            "init_kwargs": init_kwargs,
        },
    )
    result["next_actions"] = (
        ["write_workflow_config", "validate_config_semantics"]
        if result.get("ok")
        else ["write_session_file", "inspect_object_signature"]
    )
    return result


@mcp.tool(description=(TOOL_DESCRIPTIONS["delete_session"]))
def delete_session(
    force: Annotated[
        bool,
        Field(
            description="True also cancels the session's active jobs first; otherwise deletion is refused while jobs run."
        ),
    ] = False,
) -> dict[str, Any]:
    """Delete the current session workspace and optionally cancel active jobs first."""
    workspace = WORKSPACE_LAYOUT.ensure_session_workspace_exists()
    active_jobs = SESSION.active_jobs()
    if active_jobs and not force:
        running = ", ".join(f"{job.kind}:{job.job_id}" for job in active_jobs)
        raise RuntimeError(
            f"Session '{WORKSPACE_LAYOUT.current_session}' has active job(s): {running}. Use force=True to cancel them."
        )
    for job in active_jobs:
        _cancel_job_payload(job.job_id)
    shutil.rmtree(workspace)
    with _JOBS_LOCK:
        # Only forget this session's jobs; other sessions' records must survive a delete.
        for job_id, job in list(_JOBS.items()):
            if job.session == (WORKSPACE_LAYOUT.current_session or "default"):
                _JOBS.pop(job_id, None)
    return {"deleted": WORKSPACE_LAYOUT.current_session, "path": str(workspace)}


_RUN_OUTPUT_ROOTS: dict[str, tuple[str, ...]] = {
    "train": ("Statistics", "Checkpoints"),
    "prediction": ("Predictions",),
    "evaluation": ("Evaluations",),
    "uncertainty": ("Uncertainties",),
    "all": ("Statistics", "Checkpoints", "Predictions", "Evaluations", "Uncertainties"),
}


@mcp.tool(description=(TOOL_DESCRIPTIONS["delete_run"]))
def delete_run(
    run_name: Annotated[str, Field(description="The run to delete (a train_name, e.g. 'MR2CT_01').")],
    kind: Annotated[
        Literal["train", "prediction", "evaluation", "uncertainty", "all"],
        Field(
            description="Which output to remove: train (Statistics + Checkpoints), prediction, evaluation, uncertainty, "
            "or 'all' to remove every output of that run name."
        ),
    ],
) -> dict[str, Any]:
    """Delete one run's outputs from the current session, jailed to the session workspace."""
    cleaned = run_name.strip()
    if not cleaned or "/" in cleaned or "\\" in cleaned or cleaned in {".", ".."}:
        raise ValueError(f"Invalid run_name '{run_name}': a run is a single folder name, not a path.")
    base = WORKSPACE_LAYOUT.ensure_session_workspace_exists().resolve()
    removed: list[str] = []
    for root in _RUN_OUTPUT_ROOTS.get(kind, ()):
        target = (base / root / cleaned).resolve()
        if base != target.parent.parent:  # jail: exactly base/<root>/<run>, never outside the workspace
            raise ValueError("Refused: the resolved path escapes the session workspace.")
        if target.is_dir():
            shutil.rmtree(target)
            removed.append(str(target.relative_to(base)))
    return {"run_name": cleaned, "kind": kind, "deleted": removed}


@mcp.tool(description=(TOOL_DESCRIPTIONS["validate_config_semantics"]))
def validate_config_semantics(
    workflow: Annotated[
        str,
        Field(
            description="'train', 'prediction', 'evaluation', or 'all' to validate every present config (case-insensitive; default 'train')."
        ),
    ] = "train",
    level: Annotated[
        Literal["instantiate", "setup", "train_step"],
        Field(
            description="'instantiate' builds the objects, 'setup' also builds the dataloader, 'train_step' also runs ONE forward+backward (train workflow only)."
        ),
    ] = "instantiate",
    models: Annotated[
        str | list[str] | None,
        Field(
            description="Checkpoint path(s) for prediction validation (string or list; default: the newest discovered checkpoint)."
        ),
    ] = None,
    config_file: Annotated[
        str | None,
        Field(description="Alternate train config filename in the session workspace (workflow='train' only)."),
    ] = None,
) -> dict[str, Any]:
    """Instantiate KonfAI workflows in an isolated subprocess to catch config errors before launching jobs."""
    resolved_models = _normalize_string_list(models, field_name="models")
    if config_file is not None and workflow.strip().lower() != "train":
        raise ValueError("config_file is only supported for workflow='train' (alternate training configs).")
    if workflow.strip().lower() == "all":
        workspace = WORKSPACE_LAYOUT.ensure_session_workspace_exists()
        results: dict[str, Any] = {
            "session": WORKSPACE_LAYOUT.current_session,
            "path": str(workspace),
            "level": level,
            "results": {},
        }
        for current_workflow, path in SESSION.config_paths().items():
            if path.exists():
                results["results"][current_workflow] = SESSION.validate_semantics(
                    current_workflow, level, resolved_models
                )
        results["ok"] = all(result.get("ok", False) for result in results["results"].values()) and bool(
            results["results"]
        )
        results["next_actions"] = SESSION.session_summary()["next_actions"]
        return results
    result = SESSION.validate_semantics(
        workflow,
        level,
        resolved_models,
        config_path=_resolve_train_config(config_file) if config_file is not None else None,
    )
    summary_actions = [
        action for action in SESSION.session_summary()["next_actions"] if action != "validate_config_semantics"
    ]
    if result.get("ok", False):
        result["next_actions"] = list(dict.fromkeys([f"run_{workflow}", "summarize_session", *summary_actions]))
    else:
        result["next_actions"] = list(dict.fromkeys([*(result.get("next_actions", [])), "summarize_session"]))
    return result


@mcp.tool(description=(TOOL_DESCRIPTIONS["review_config_semantics"]))
def review_config_semantics(
    workflow: Annotated[
        WorkflowKind,
        Field(description="Which session config to review (default 'train')."),
    ] = "train",
) -> dict[str, Any]:
    """Review one config statically and emit lightweight semantic warnings before runtime validation."""
    return SESSION.review_config_semantics(workflow)


@mcp.tool(description=(TOOL_DESCRIPTIONS["summarize_session"]))
def summarize_session(
    include_leaderboard: Annotated[
        bool, Field(description="Include a top-5 leaderboard for the chosen split/metric (default True).")
    ] = True,
    leaderboard_metric: Annotated[
        str | None, Field(description="Metric filter for the embedded leaderboard (default: every available metric).")
    ] = None,
    leaderboard_split: Annotated[
        str, Field(description="Metric split for the embedded leaderboard (default 'TRAIN').")
    ] = "TRAIN",
    include_log_tail: Annotated[
        bool, Field(description="Include the latest session log tail (default False).")
    ] = False,
    include_validation: Annotated[
        bool, Field(description="Also validate the present configs (slower; default False keeps the payload lean).")
    ] = False,
    log_lines: Annotated[int, Field(description="Number of log lines when include_log_tail=True (default 50).")] = 50,
) -> dict[str, Any]:
    """Return one compact session snapshot for iterative agent loops."""
    summary = SESSION.session_summary()
    metrics_payload = SESSION.read_metrics_payload()
    payload: dict[str, Any] = {
        "session": summary["session"],
        "path": summary["path"],
        "readiness": summary["readiness"],
        "configs": summary["configs"],
        "resources": summary["resources"],
        # Compact ref only (the full job payload is one get_job_status / list_jobs call away).
        "latest_job": (
            {
                key: summary["latest_job"].get(key)
                for key in ("job_id", "kind", "status", "returncode", "finished_at", "run_name")
            }
            if summary["latest_job"] is not None
            else None
        ),
        "active_jobs": summary["active_jobs"],
        "metrics_summary": metrics_payload["summary"],
        "metrics_path": metrics_payload["path"],
        "next_actions": summary["next_actions"],
    }
    if include_validation:
        validation = SESSION.validate_session_payload()
        # readiness/active_jobs/next_actions already sit at the top level of this payload.
        for duplicated in ("readiness", "active_jobs", "next_actions", "session", "path"):
            validation.pop(duplicated, None)
        payload["validation"] = validation
    if include_log_tail:
        log_path = SESSION.discover_log_path()
        payload["latest_log_tail"] = _read_text_tail(log_path, max_lines=log_lines) if log_path is not None else ""
    if include_leaderboard:
        try:
            board = SESSION.leaderboard_payload(metric=leaderboard_metric, split=leaderboard_split, limit=5)
            # leaderboard_payload emits an empty 'leaderboard' list beside the full 'leaderboards'
            # dict when no single metric is selected; drop the redundant empty list here only
            # (the standalone leaderboard tool keeps both keys).
            if isinstance(board.get("leaderboards"), dict) and not board.get("leaderboard"):
                board.pop("leaderboard", None)
            payload["leaderboard"] = board
        except ValueError:
            payload["leaderboard"] = None
    return round_floats(payload)


@mcp.tool(description=(TOOL_DESCRIPTIONS["write_workflow_config"]))
def write_workflow_config(
    workflow: Annotated[
        WorkflowKind,
        Field(
            description="Which config file to write: train -> Config.yml, prediction -> Prediction.yml, evaluation -> Evaluation.yml."
        ),
    ],
    content: Annotated[
        str,
        Field(
            description="Full YAML content; the top-level root key must match the workflow (Trainer/Predictor/Evaluator)."
        ),
    ],
    overwrite: Annotated[
        bool, Field(description="Replace an existing config (default True); False raises if the file exists.")
    ] = True,
) -> dict[str, Any]:
    """Write one KonfAI workflow config chosen by workflow."""
    return _write_workflow_config(workflow, content, overwrite)


@mcp.tool(description=(TOOL_DESCRIPTIONS["list_jobs"]))
def list_jobs() -> list[dict[str, Any]]:
    """List known jobs for the current session workspace."""
    with _JOBS_LOCK:
        jobs = list(_JOBS.values())
    jobs.sort(key=lambda job: job.created_at, reverse=True)
    return [_job_payload(job) for job in jobs]


@mcp.tool(description=(TOOL_DESCRIPTIONS["leaderboard"]))
def leaderboard(
    metric: Annotated[
        str | None, Field(description="Metric to rank by (default: one leaderboard per available metric).")
    ] = None,
    split: Annotated[
        str,
        Field(
            description="Metric split, maps to Metric_<SPLIT>.json (default 'TRAIN'; a miss lists available splits)."
        ),
    ] = "TRAIN",
    limit: Annotated[int, Field(description="Maximum rows per leaderboard (default 20).")] = 20,
    direction: Annotated[
        Literal["min", "max"] | None,
        Field(
            description="Override the ranking direction inferred from the metric name; applies to every metric in the payload."
        ),
    ] = None,
    session: Annotated[
        str | None, Field(description="Rank another session's runs without switching (default: the current session).")
    ] = None,
) -> dict[str, Any]:
    """Rank completed runs using evaluation metrics written by KonfAI in a session workspace."""
    return round_floats(
        SESSION.leaderboard_payload(metric=metric, split=split, limit=limit, direction=direction, session=session)
    )


@mcp.tool(description=(TOOL_DESCRIPTIONS["run_train"]))
def run_train(
    gpu: Annotated[
        int | list[int] | None,
        Field(
            description="GPU index or list of indices to train on; omit to run on CPU (mutually exclusive with cpu)."
        ),
    ] = None,
    cpu: Annotated[
        int | None, Field(description="Run on CPU with N worker processes (> 0); mutually exclusive with gpu.")
    ] = None,
    overwrite: Annotated[
        bool,
        Field(description="Overwrite existing run outputs (checkpoints, logs) without prompting (KonfAI --overwrite)."),
    ] = False,
    quiet: Annotated[
        bool, Field(description="Suppress console output for a quieter execution (KonfAI --quiet).")
    ] = False,
    tensorboard: Annotated[
        bool, Field(description="Launch TensorBoard alongside the run (KonfAI --tensorboard).")
    ] = False,
    single_process: Annotated[
        bool, Field(description="Run the workflow inline in one process (no DDP spawn, no GPU setup).")
    ] = False,
    config_file: Annotated[
        str | None,
        Field(
            description="Alternate train config filename in the session workspace (e.g. 'Config_GAN.yml'; default Config.yml)."
        ),
    ] = None,
    cluster: Annotated[
        dict[str, Any] | None,
        Field(
            description="SLURM submission via submitit: exactly the keys {name, memory, num_nodes, time_limit}; omit to run locally."
        ),
    ] = None,
) -> dict[str, Any]:
    """Launch a training job from the session train config (Config.yml or an alternate config_file)."""
    normalized_gpu = _normalize_int_list(gpu, field_name="gpu")
    cluster = _validate_cluster(cluster)
    config_path = _resolve_train_config(config_file)
    blocked = SESSION.workflow_blocker("train", config_path=config_path)
    if blocked is not None:
        return blocked
    job_spec = _runtime_job_spec(
        kind="train",
        config_path=config_path,
        gpu=normalized_gpu,
        cpu=cpu,
        overwrite=overwrite,
        quiet=quiet,
        single_process=single_process,
        tensorboard=tensorboard,
        cluster=cluster,
    )
    payload = _launch_job(
        "train",
        job_spec["command"],
        config_path,
        extra_manifest={"devices": {"gpu": normalized_gpu or [], "cpu": cpu}, "cluster": cluster},
        target=job_spec["target"],
        kwargs=job_spec["kwargs"],
        devices=_job_devices(normalized_gpu, cpu, cluster),
    )
    cpu_fallback = _cpu_fallback_warnings(normalized_gpu, cluster)
    if cpu_fallback:
        payload["warnings"] = cpu_fallback
    return payload


def _strip_to_weights_only(checkpoint: Path) -> Path:
    """Copy ``checkpoint`` keeping only the ``Model`` weights, to a jailed ``<stem>_init.pt`` in the
    workspace. RESUME then starts at epoch/it 0 (a fresh fine-tune). The name carries no path separators."""
    from konfai.utils.runtime import safe_torch_load

    state_dict = safe_torch_load(checkpoint, "cpu")
    if "Model" not in state_dict:
        raise ValueError(f"Checkpoint '{checkpoint}' has no 'Model' weights to warm-start from.")
    destination = SESSION.workspace_dir() / f"{checkpoint.stem}_init.pt"
    import torch

    torch.save({"Model": state_dict["Model"]}, destination)  # nosec B614
    return destination


@mcp.tool(description=(TOOL_DESCRIPTIONS["run_resume"]))
def run_resume(
    model: Annotated[
        str | None,
        Field(
            description="Checkpoint to resume from: a path (workspace-relative or absolute) or an http(s) URL (default: the configured run's newest checkpoint)."
        ),
    ] = None,
    weights_only: Annotated[
        bool,
        Field(
            description="Warm-start a fine-tune: load only the checkpoint's model weights and restart epoch/iteration/optimizer from scratch (the fine-tune-from-app path). Requires a local checkpoint, not a URL."
        ),
    ] = False,
    lr: Annotated[
        float | None, Field(description="Learning-rate override; omit to continue the restored schedule.")
    ] = None,
    gpu: Annotated[
        int | list[int] | None,
        Field(
            description="GPU index or list of indices to train on; omit to run on CPU (mutually exclusive with cpu)."
        ),
    ] = None,
    cpu: Annotated[
        int | None, Field(description="Run on CPU with N worker processes (> 0); mutually exclusive with gpu.")
    ] = None,
    overwrite: Annotated[
        bool,
        Field(description="Overwrite existing run outputs (checkpoints, logs) without prompting (KonfAI --overwrite)."),
    ] = False,
    quiet: Annotated[
        bool, Field(description="Suppress console output for a quieter execution (KonfAI --quiet).")
    ] = False,
    tensorboard: Annotated[
        bool, Field(description="Launch TensorBoard alongside the run (KonfAI --tensorboard).")
    ] = False,
    single_process: Annotated[
        bool, Field(description="Run the workflow inline in one process (no DDP spawn, no GPU setup).")
    ] = False,
    config_file: Annotated[
        str | None,
        Field(
            description="Alternate train config filename in the session workspace (e.g. 'Config_GAN.yml'; default Config.yml)."
        ),
    ] = None,
) -> dict[str, Any]:
    """Resume training from a checkpoint (optimizer/scheduler/epoch restored) via KonfAI's RESUME command."""
    normalized_gpu = _normalize_int_list(gpu, field_name="gpu")
    config_path = _resolve_train_config(config_file)
    resolved_model: Path | str | None
    if model is not None and model.startswith(("http://", "https://")):
        # KonfAI's RESUME accepts remote checkpoint URLs; pass them through untouched.
        resolved_model = model
    elif model is not None:
        candidate = Path(model).expanduser()
        resolved_model = (candidate if candidate.is_absolute() else SESSION.workspace_dir() / candidate).resolve()
    else:
        # Prefer the configured run's own checkpoints over the globally newest one, so a sweep
        # does not silently resume run A from run B's checkpoint.
        run_name = SESSION.configured_run_name("train", config_path)
        run_dir = WORKSPACE_LAYOUT.checkpoints_dir() / run_name if run_name is not None else None
        run_checkpoints = (
            sorted(run_dir.glob("*.pt"), key=lambda path: path.stat().st_mtime, reverse=True)
            if run_dir is not None and run_dir.exists()
            else []
        )
        discovered = run_checkpoints or SESSION.discover_model_paths(limit=1)
        resolved_model = discovered[0] if discovered else None
    if resolved_model is None:
        raise ValueError("No checkpoint found to resume from. Provide model explicitly or run run_train first.")
    if isinstance(resolved_model, Path) and not resolved_model.exists():
        raise ValueError(f"Checkpoint not found: {resolved_model}")
    if weights_only:
        if not isinstance(resolved_model, Path):
            raise ValueError("weights_only needs a local checkpoint (a URL cannot be stripped to weights).")
        resolved_model = _strip_to_weights_only(resolved_model)
    blocked = SESSION.workflow_blocker("train", config_path=config_path)
    if blocked is not None:
        return blocked
    job_spec = _runtime_job_spec(
        kind="train",
        config_path=config_path,
        gpu=normalized_gpu,
        cpu=cpu,
        overwrite=overwrite,
        quiet=quiet,
        single_process=single_process,
        tensorboard=tensorboard,
        resume_model=resolved_model,
        lr=lr,
    )
    return _launch_job(
        "train",
        job_spec["command"],
        config_path,
        extra_manifest={
            "devices": {"gpu": normalized_gpu or [], "cpu": cpu},
            "resume_from": str(resolved_model),
            "lr_override": lr,
        },
        target=job_spec["target"],
        kwargs=job_spec["kwargs"],
        devices=_job_devices(normalized_gpu, cpu),
    )


@mcp.tool(description=(TOOL_DESCRIPTIONS["run_batch"]))
def run_batch(
    config_files: Annotated[
        list[str],
        Field(
            description="Train config filenames in the session workspace, run sequentially (each needs a distinct train_name)."
        ),
    ],
    gpu: Annotated[
        int | list[int] | None,
        Field(
            description="GPU index or list of indices to train on; omit to run on CPU (mutually exclusive with cpu)."
        ),
    ] = None,
    cpu: Annotated[
        int | None, Field(description="Run on CPU with N worker processes (> 0); mutually exclusive with gpu.")
    ] = None,
    overwrite: Annotated[
        bool,
        Field(description="Overwrite existing run outputs (checkpoints, logs) without prompting (KonfAI --overwrite)."),
    ] = False,
    quiet: Annotated[
        bool, Field(description="Suppress console output for a quieter execution (KonfAI --quiet; default True).")
    ] = True,
    single_process: Annotated[
        bool, Field(description="Run each workflow inline in one process (no DDP spawn, no GPU setup).")
    ] = False,
    stop_on_error: Annotated[
        bool, Field(description="Stop the batch at the first failed or blocked run (default True).")
    ] = True,
    poll_interval_s: Annotated[
        float, Field(description="Seconds between job status polls while waiting on each run (default 0.5).")
    ] = 0.5,
) -> dict[str, Any]:
    """Run several training configs sequentially and collect their outcomes."""
    if not config_files:
        raise ValueError("config_files cannot be empty.")
    results: list[dict[str, Any]] = []
    for config_file in config_files:
        try:
            payload = run_train(
                gpu=gpu,
                cpu=cpu,
                overwrite=overwrite,
                quiet=quiet,
                single_process=single_process,
                config_file=config_file,
            )
        except Exception as exc:
            results.append({"config_file": config_file, "status": "launch_error", "error": str(exc)})
            if stop_on_error:
                break
            continue
        if payload.get("blocked"):
            # run_train returns the structured blocker (not a job payload) when a prerequisite is
            # unmet; there is no job_id/status to poll, so record the blocker and move on.
            results.append(
                {
                    "config_file": config_file,
                    "status": "blocked",
                    "error": payload.get("error"),
                    "missing_paths": payload.get("missing_paths"),
                }
            )
            if stop_on_error:
                break
            continue
        while payload["status"] in ACTIVE_JOB_STATES:
            time.sleep(max(poll_interval_s, 0.05))
            payload = _job_payload(JOB_REGISTRY.get(payload["job_id"]))
        results.append(
            {
                "config_file": config_file,
                "job_id": payload["job_id"],
                "run_name": payload["run_name"],
                "status": payload["status"],
                "error": payload["error"],
            }
        )
        if payload["status"] != "done" and stop_on_error:
            break
    return {
        "requested": len(config_files),
        "completed": sum(1 for result in results if result.get("status") == "done"),
        "results": results,
        "next_actions": ["leaderboard", "compare_runs", "summarize_session"],
    }


@mcp.tool(description=(TOOL_DESCRIPTIONS["generate_folds"]))
def generate_folds(
    dataset_dir: Annotated[
        str, Field(description="Host path of the dataset whose case directories are split into folds.")
    ],
    k: Annotated[int, Field(description="Number of folds (>= 2; default 5).")] = 5,
    seed: Annotated[int, Field(description="Shuffle seed for the case-to-fold assignment (default 0).")] = 0,
) -> dict[str, Any]:
    """Write K fold case-list files and the subset stanzas that use them."""
    if k < 2:
        raise ValueError("k must be >= 2.")
    dataset_path = Path(dataset_dir).expanduser().resolve()
    if not dataset_path.is_dir():
        raise ValueError(f"Dataset directory not found: {dataset_path}")
    cases = sorted(path.name for path in case_directories(dataset_path) if path != dataset_path)
    if len(cases) < k:
        raise ValueError(f"Only {len(cases)} case directories found; cannot make k={k} folds.")
    workspace = WORKSPACE_LAYOUT.ensure_session_workspace()  # a setup step: create the session if needed
    shuffled = list(cases)
    random.Random(seed).shuffle(shuffled)
    folds_dir = workspace / "folds"
    folds_dir.mkdir(exist_ok=True)
    folds: dict[str, Any] = {}
    for index in range(k):
        members = sorted(shuffled[index::k])
        fold_file = folds_dir / f"fold_{index}.txt"
        fold_file.write_text("\n".join(members) + "\n", encoding="utf-8")
        folds[f"fold_{index}"] = {
            "cases": members,
            "file": str(fold_file),
            "train_subset": f"~folds/fold_{index}.txt",
            "eval_subset": f"folds/fold_{index}.txt",
        }
    return {
        "dataset_dir": str(dataset_path),
        "k": k,
        "seed": seed,
        "total_cases": len(cases),
        "folds": folds,
        "how_to_use": (
            "Per fold i: set Trainer.Dataset.subset to '~folds/fold_i.txt' (train on the other folds) and the "
            "Predictor/Evaluator Dataset.subset to 'folds/fold_i.txt' (score the held-out fold); give each fold "
            "a distinct train_name, then run_batch the fold configs."
        ),
        "next_actions": ["write_session_file", "write_workflow_config", "run_batch"],
    }


@mcp.tool(description=(TOOL_DESCRIPTIONS["preview_volume"]))
def preview_volume(
    path: Annotated[str, Field(description="Host path of a SimpleITK-readable volume file (mha/nii.gz/...).")],
    slice_index: Annotated[
        int | None,
        Field(description="Slice to render along the chosen axis (clamped to the volume; default: the middle slice)."),
    ] = None,
    axis: Annotated[
        int, Field(description="Axis to slice along in the array's [Z,Y,X] order: 0 = depth (default), 1, or 2.")
    ] = 0,
    max_size: Annotated[
        int, Field(description="Maximum output edge in pixels; larger planes are strided down (default 512).")
    ] = 512,
) -> FastMCPImage:
    """Render one slice of a volume as a PNG for visual QC."""
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise ValueError("preview_volume requires SimpleITK: pip install konfai[imaging].") from exc
    import numpy as np

    volume_path = Path(path).expanduser().resolve()
    if not volume_path.is_file():
        raise ValueError(f"Volume file not found: {volume_path}")

    reader = sitk.ImageFileReader()
    reader.SetFileName(str(volume_path))
    reader.ReadImageInformation()
    size_xyz = list(reader.GetSize())  # SimpleITK order (x, y, [z], ...)
    if len(size_xyz) == 3 and reader.GetNumberOfComponents() == 1:
        # 3-D scalar volume (the QC case): stream only the requested plane so the whole volume is never
        # materialised in RAM (mandatory lazy invariant). Array order is [Z, Y, X], i.e. reversed size_xyz.
        axis = int(min(max(axis, 0), 2))
        depth = list(reversed(size_xyz))[axis]
        index = depth // 2 if slice_index is None else int(min(max(slice_index, 0), depth - 1))
        sitk_axis = 2 - axis
        extract_index = [0, 0, 0]
        extract_index[sitk_axis] = index
        extract_size = list(size_xyz)
        extract_size[sitk_axis] = 1
        reader.SetExtractIndex(extract_index)
        reader.SetExtractSize(extract_size)
        plane = np.squeeze(sitk.GetArrayFromImage(reader.Execute()), axis=axis)
    else:
        # Uncommon layout (2-D, 4-D, or multi-component): read the whole image, then slice in memory.
        array = sitk.GetArrayFromImage(reader.Execute())
        while array.ndim > 3:
            array = array[0]
        if array.ndim == 3:
            axis = int(min(max(axis, 0), 2))
            depth = array.shape[axis]
            index = depth // 2 if slice_index is None else int(min(max(slice_index, 0), depth - 1))
            plane = np.take(array, index, axis=axis)
        else:
            plane = array
    plane = plane.astype(np.float32)
    low, high = np.percentile(plane, (1.0, 99.0))
    if high <= low:
        high = low + 1.0
    plane = np.clip((plane - low) / (high - low) * 255.0, 0.0, 255.0).astype(np.uint8)
    stride = max(1, -(-max(plane.shape) // max(max_size, 16)))
    plane = plane[::stride, ::stride]
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
        png_path = Path(handle.name)
    try:
        sitk.WriteImage(sitk.GetImageFromArray(plane), str(png_path))
        data = png_path.read_bytes()
    finally:
        png_path.unlink(missing_ok=True)
    return FastMCPImage(data=data, format="png")


@mcp.tool(description=(TOOL_DESCRIPTIONS["run_prediction"]))
def run_prediction(
    models: Annotated[
        str | list[str] | None,
        Field(
            description="Checkpoint path(s) to predict with (string or list; default: the configured run's newest checkpoint)."
        ),
    ] = None,
    gpu: Annotated[
        int | list[int] | None,
        Field(description="GPU index or list of indices to run on; omit to run on CPU (mutually exclusive with cpu)."),
    ] = None,
    cpu: Annotated[
        int | None, Field(description="Run on CPU with N worker processes (> 0); mutually exclusive with gpu.")
    ] = None,
    overwrite: Annotated[
        bool, Field(description="Overwrite existing prediction outputs without prompting (KonfAI --overwrite).")
    ] = False,
    quiet: Annotated[
        bool, Field(description="Suppress console output for a quieter execution (KonfAI --quiet).")
    ] = False,
    single_process: Annotated[
        bool, Field(description="Run the workflow inline in one process (no DDP spawn, no GPU setup).")
    ] = False,
) -> dict[str, Any]:
    """Launch a prediction job from the current session `Prediction.yml`."""
    normalized_models = _normalize_string_list(models, field_name="models")
    normalized_gpu = _normalize_int_list(gpu, field_name="gpu")
    config_path = SESSION.config_path("prediction")
    if not config_path.exists():
        raise ValueError("Prediction.yml not found. Write a prediction config first.")

    resolved_models = SESSION.resolve_prediction_models(
        normalized_models, run_name=SESSION.configured_run_name("prediction", config_path)
    )
    if not resolved_models:
        raise ValueError("No model checkpoint found. Provide models explicitly or train the current session first.")
    missing = [str(path) for path in resolved_models if not path.exists()]
    if missing:
        raise ValueError(f"Model checkpoint(s) not found: {missing}")
    blocked = SESSION.workflow_blocker("prediction", [str(path) for path in resolved_models])
    if blocked is not None:
        return blocked

    job_spec = _runtime_job_spec(
        kind="prediction",
        config_path=config_path,
        models=resolved_models,
        gpu=normalized_gpu,
        cpu=cpu,
        overwrite=overwrite,
        quiet=quiet,
        single_process=single_process,
    )
    return _launch_job(
        "prediction",
        job_spec["command"],
        config_path,
        extra_manifest={
            "devices": {"gpu": normalized_gpu or [], "cpu": cpu},
            "models": [str(path) for path in resolved_models],
        },
        target=job_spec["target"],
        kwargs=job_spec["kwargs"],
        devices=_job_devices(normalized_gpu, cpu),
    )


@mcp.tool(description=(TOOL_DESCRIPTIONS["run_evaluation"]))
def run_evaluation(
    gpu: Annotated[
        int | list[int] | None,
        Field(description="GPU index or list of indices to run on; omit to run on CPU (mutually exclusive with cpu)."),
    ] = None,
    cpu: Annotated[
        int | None, Field(description="Run on CPU with N worker processes (> 0); mutually exclusive with gpu.")
    ] = None,
    overwrite: Annotated[
        bool, Field(description="Overwrite existing evaluation outputs without prompting (KonfAI --overwrite).")
    ] = False,
    quiet: Annotated[
        bool, Field(description="Suppress console output for a quieter execution (KonfAI --quiet).")
    ] = False,
    single_process: Annotated[
        bool, Field(description="Run the workflow inline in one process (no DDP spawn, no GPU setup).")
    ] = False,
) -> dict[str, Any]:
    """Launch an evaluation job from the current session `Evaluation.yml`."""
    normalized_gpu = _normalize_int_list(gpu, field_name="gpu")
    config_path = SESSION.config_path("evaluation")
    if not config_path.exists():
        raise ValueError("Evaluation.yml not found. Write an evaluation config first.")
    blocked = SESSION.workflow_blocker("evaluation")
    if blocked is not None:
        return blocked
    job_spec = _runtime_job_spec(
        kind="evaluation",
        config_path=config_path,
        gpu=normalized_gpu,
        cpu=cpu,
        overwrite=overwrite,
        quiet=quiet,
        single_process=single_process,
    )
    return _launch_job(
        "evaluation",
        job_spec["command"],
        config_path,
        extra_manifest={"devices": {"gpu": normalized_gpu or [], "cpu": cpu}},
        target=job_spec["target"],
        kwargs=job_spec["kwargs"],
        devices=_job_devices(normalized_gpu, cpu),
    )


@mcp.tool(description=(TOOL_DESCRIPTIONS["cancel_job"]))
def cancel_job(
    job_id: Annotated[str, Field(description="Job to cancel (as returned by a run_* launch or list_jobs).")],
    wait_s: Annotated[
        float, Field(description="Seconds to wait for a clean shutdown before the process group is killed (default 5).")
    ] = 5.0,
) -> dict[str, Any]:
    """Request job termination and wait briefly for a clean shutdown before killing it."""
    return _cancel_job_payload(job_id, wait_s=wait_s)


@mcp.tool(
    description=(
        "Request an on-demand validation pass on a running training job. Sends SIGUSR1 to the job's process "
        "group; the trainer runs validation at its next iteration boundary and logs the metrics (no "
        "checkpoint or early-stopping side effect). POSIX only; a no-op when no training job is running."
    )
)
def request_validation(
    kind: Annotated[
        WorkflowKind, Field(description="Job kind to target when job_id is omitted (default 'train').")
    ] = "train",
    job_id: Annotated[
        str | None, Field(description="Exact job to validate (default: the latest job of the given kind).")
    ] = None,
) -> dict[str, Any]:
    """Ask a running training job to run a validation pass now, via SIGUSR1."""
    job = JOB_REGISTRY.get(job_id) if job_id is not None else SESSION.discover_latest_job(kind)
    if job is None or job.status not in ACTIVE_JOB_STATES:
        return {"ok": False, "detail": "No running training job to validate."}
    delivered = JOB_REGISTRY.notify(job, signal.SIGUSR1)
    return {
        "ok": delivered,
        "job_id": job.job_id,
        "detail": (
            "Validation requested; it runs at the next iteration boundary."
            if delivered
            else "Could not signal the job (already finished, or not signalable)."
        ),
    }


@mcp.tool(
    description=(
        "Change tunables of a RUNNING training job in place, without restarting it. Writes a jailed "
        "control.json into the run's Statistics/<run>/ dir; the trainer re-reads it at its next poll boundary "
        "(~20 iterations) and records each change into the run's config snapshot. 'it_validation' takes effect "
        "for all following iterations; 'lr' sets the optimizer learning rate now, rebased so a running "
        "scheduler keeps it (the schedule restarts from the new value). POSIX/local; a no-op when no training "
        "job is running."
    )
)
def set_live_tunables(
    lr: Annotated[
        float | None,
        Field(
            gt=0,
            description="New optimizer learning rate, applied at the next poll boundary. Omit to leave it unchanged.",
        ),
    ] = None,
    it_validation: Annotated[
        int | None,
        Field(
            gt=0,
            description="New validation interval in iterations, effective for the following iterations. Omit to leave it unchanged.",
        ),
    ] = None,
    job_id: Annotated[
        str | None, Field(description="Exact training job to steer (default: the latest 'train' job of this session).")
    ] = None,
) -> dict[str, Any]:
    """Steer a running training job's lr / it_validation by writing a jailed, revisioned control file."""
    if lr is None and it_validation is None:
        return {"ok": False, "detail": "Nothing to change: provide lr and/or it_validation."}
    job = JOB_REGISTRY.get(job_id) if job_id is not None else SESSION.discover_latest_job("train")
    if job is None or job.kind != "train" or job.status not in ACTIVE_JOB_STATES:
        return {"ok": False, "detail": "No running training job to tune."}
    runtime_log = SESSION.job_runtime_log_path(job)
    if runtime_log is None:
        return {"ok": False, "detail": "Job has no resolvable run directory yet."}
    control_path = WORKSPACE_LAYOUT.resolve_workspace_relative_path(str(runtime_log.parent / "control.json"))
    control_path.parent.mkdir(parents=True, exist_ok=True)
    previous: dict[str, Any] = {}
    if control_path.exists():
        try:
            loaded = json.loads(control_path.read_text(encoding="utf-8"))
            previous = loaded if isinstance(loaded, dict) else {}
        except (OSError, json.JSONDecodeError):
            previous = {}
    revision = int(previous.get("revision", 0)) + 1
    payload: dict[str, Any] = {"revision": revision}
    if lr is not None:
        payload["lr"] = lr
    if it_validation is not None:
        payload["it_validation"] = it_validation
    tmp = control_path.with_name(f".{control_path.name}.{uuid.uuid4().hex}.tmp")  # atomic write, like the job store
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, control_path)
    return {
        "ok": True,
        "job_id": job.job_id,
        "revision": revision,
        "applied": {key: payload[key] for key in ("lr", "it_validation") if key in payload},
        "detail": "Applied at the trainer's next poll boundary (~20 iterations).",
        "next_actions": ["read_live_metrics", "request_validation", "get_job_status"],
    }


@mcp.tool(description=(TOOL_DESCRIPTIONS["get_job_status"]))
def get_job_status(
    job_id: Annotated[str, Field(description="Job identifier returned by a run_* launch or list_jobs.")],
) -> dict[str, Any]:
    """Return the current job status together with suggested next actions."""
    return _job_payload(JOB_REGISTRY.get(job_id))


@mcp.tool(description=(TOOL_DESCRIPTIONS["read_job_log"]))
def read_job_log(
    job_id: Annotated[str, Field(description="Job whose log to read.")],
    max_lines: Annotated[int, Field(description="Maximum lines returned after filtering (default 200).")] = 200,
    grep: Annotated[
        str | None,
        Field(
            description="Regex applied per line over a bounded window of the last max(20*max_lines, 2000) lines before the tail is taken."
        ),
    ] = None,
    source: Annotated[
        Literal["auto", "job", "runtime"],
        Field(
            description="'auto' prefers the runtime log except for failed jobs (their traceback lives in the console job log); 'job'/'runtime' force one."
        ),
    ] = "auto",
) -> dict[str, Any]:
    """Read (tail, page, or regex-filter) one job's log so failures can be diagnosed tool-only."""
    job = JOB_REGISTRY.get(job_id)
    runtime_log = SESSION.job_runtime_log_path(job)
    if source == "runtime":
        if runtime_log is None or not runtime_log.exists():
            raise ValueError(f"No runtime log found for job '{job_id}'.")
        path = runtime_log
    elif (
        source == "auto" and runtime_log is not None and runtime_log.exists() and job.status not in ("error", "killed")
    ):
        # A crashed child's traceback is written to the console log only (the KonfAI runtime log
        # is restored before the exception prints), so failed jobs must default to the job log.
        path = runtime_log
    else:
        path = job.log_path
    max_lines = max(max_lines, 1)
    scan_lines = max_lines if grep is None else max(max_lines * 20, 2000)
    lines = _read_text_tail(path, max_lines=scan_lines).splitlines()
    if grep is not None:
        pattern = re.compile(grep)
        lines = [line for line in lines if pattern.search(line)]
    selected = lines[-max_lines:]
    return {
        "job_id": job_id,
        "status": job.status,
        "path": str(path),
        "grep": grep,
        "lines_returned": len(selected),
        "content": "\n".join(selected),
        "next_actions": ["get_job_status", "validate_config_semantics", "summarize_session"],
    }


@mcp.tool(description=(TOOL_DESCRIPTIONS["read_live_metrics"]))
def read_live_metrics(
    kind: Annotated[
        WorkflowKind,
        Field(description="Job kind used to discover the latest job when job_id is omitted (default 'train')."),
    ] = "train",
    job_id: Annotated[
        str | None, Field(description="Exact job to read (default: the latest job of the given kind).")
    ] = None,
    max_entries: Annotated[int, Field(description="Maximum recent metric entries returned (default 20).")] = 20,
) -> dict[str, Any]:
    """Read parsed live metrics from the current runtime log of a job."""
    job = JOB_REGISTRY.get(job_id) if job_id is not None else SESSION.discover_latest_job(kind)
    if job is None:
        raise ValueError(f"No job found for session '{WORKSPACE_LAYOUT.current_session}' and kind '{kind}'.")
    return SESSION.read_live_metrics_payload(job, max_entries)


@mcp.tool(description=(TOOL_DESCRIPTIONS["wait_for_job"]))
def wait_for_job(
    job_id: Annotated[str, Field(description="Job to wait for.")],
    timeout_s: Annotated[
        float | None,
        Field(
            description="Maximum seconds to wait (raises TimeoutError on expiry); omit/None to wait until the job finishes."
        ),
    ] = None,
    poll_interval_s: Annotated[float, Field(description="Seconds between status polls (min 0.05; default 0.5).")] = 0.5,
) -> dict[str, Any]:
    """Block until a job becomes terminal, then return its final status payload (timeout semantics: see description)."""
    deadline = None if timeout_s is None else time.time() + max(timeout_s, 0.0)
    interval = max(poll_interval_s, 0.05)
    while deadline is None or time.time() < deadline:
        payload = _job_payload(JOB_REGISTRY.get(job_id))
        if payload["status"] not in ACTIVE_JOB_STATES:
            return payload
        time.sleep(interval)
    raise TimeoutError(f"Timed out while waiting for job '{job_id}' after {timeout_s:.1f}s.")


def main(
    transport: Literal["stdio", "sse", "streamable-http"] = "stdio",
    host: str | None = None,
    port: int | None = None,
    path: str | None = None,
    log_level: str | None = None,
    bearer_token: str | None = None,
) -> None:
    _configure_transport_auth(
        transport,
        host=host,
        port=port,
        bearer_token=bearer_token or os.environ.get("KONFAI_MCP_BEARER_TOKEN"),
    )
    transport_kwargs: dict[str, Any] = {}
    if transport == "stdio":
        if log_level is not None:
            transport_kwargs["log_level"] = log_level
    else:
        if host is not None:
            transport_kwargs["host"] = host
        if port is not None:
            transport_kwargs["port"] = port
        if path is not None:
            transport_kwargs["path"] = path
        if log_level is not None:
            transport_kwargs["log_level"] = log_level
    # MCP stdio transports must keep stdout protocol-clean for the client.
    mcp.run(transport, show_banner=False, **transport_kwargs)


if __name__ == "__main__":
    main()
