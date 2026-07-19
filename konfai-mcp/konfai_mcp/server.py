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
    # a concurrent tool (SSE/HTTP transports) can no longer observe a half-built mix (new layout
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


def _launch_app_job(spec: dict[str, Any]) -> dict[str, Any]:
    """Launch an app job (inference or fine-tune) from an AppService spec via the shared job registry.

    Unlike workflow jobs, an app job has no session YAML: it auto-creates the session workspace,
    tracks the run under the spec's kind ('infer' or 'finetune'), and carries its own runner target
    and kwargs.
    """
    kind = spec.get("kind", "infer")
    workspace = WORKSPACE_LAYOUT.ensure_session_workspace()
    WORKSPACE_LAYOUT.jobs_dir().mkdir(parents=True, exist_ok=True)
    kwargs = dict(spec["kwargs"])
    # config_overrides live directly in kwargs (infer / finetune) or nested under extra (pipeline). Recording
    # them links this trial's tuned parameters to the score it produces and gates the refine next_actions.
    set_parameters = kwargs.get("config_overrides") or (kwargs.get("extra") or {}).get("config_overrides")
    job = JOB_REGISTRY.launch(
        session=WORKSPACE_LAYOUT.current_session or "default",
        kind=kind,
        command=spec["command"],
        cwd=workspace,
        log_path=WORKSPACE_LAYOUT.jobs_dir() / f"{kind}_{uuid.uuid4().hex[:12]}.log",
        config_path=workspace / f"app_{kind}.ref",
        run_name=spec["run_name"],
        devices=_job_devices(
            kwargs.get("gpu") if kwargs.get("gpu") is not None else (kwargs.get("extra") or {}).get("gpu"),
            kwargs.get("cpu") if kwargs.get("cpu") is not None else (kwargs.get("extra") or {}).get("cpu"),
        ),
        runtime_log_path=None,
        extra_manifest={
            "app_ref": kwargs.get("ref"),
            "app_mode": spec["mode"],
            "output": spec["output"],
            "set_parameters": set_parameters,
            "environment": _environment_snapshot(),
        },
        target=spec["target"],
        kwargs={**kwargs, "cwd": str(workspace)},
        set_parameters=set_parameters,
    )
    payload = _job_payload(job)
    payload["mode"] = spec["mode"]
    payload["output"] = spec["output"]
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
            "content": (
                "You are routing a KonfAI request. The user arrives with a dataset and a goal; choose the "
                "cheapest path that genuinely meets it, considering these options in order:\n\n"
                f"Goal: {task}\n"
                f"Dataset summary:\n{dataset_summary or '(not provided)'}\n\n"
                "1. USE AN EXISTING APP (no training). Call list_apps, then describe_app on each plausible "
                "candidate. Judge fit from the app's own description first, confirmed by its declared "
                "inputs/outputs. If one clearly does the job, run it with run_app_infer -- done.\n"
                "2. FINE-TUNE FROM AN APP. If no app is usable as-is but one is a close starting point, train "
                "from it with fine_tune_app on the user's dataset, producing a bundle you can then run.\n"
                "3. TRAIN FROM A BLANK SLATE. If no app is a useful starting point, author a config from scratch "
                "via design_config_strategy and the train loop.\n\n"
                "Prefer the earliest option that truly fits: do not train when an app already solves it, and do "
                "not start from scratch when a related app can be fine-tuned. Ask the smallest necessary "
                "clarifying question only when the choice is genuinely ambiguous. Do not invent the task; the "
                "user must specify it."
            ),
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
            "content": (
                "You are preparing a KonfAI workflow.\n\n"
                f"Requested task: {task}\n"
                f"Dataset summary:\n{dataset_summary or '(not provided)'}\n\n"
                "Ask no clarifying questions if the task, group roles, workflows, and split are already clear.\n"
                "Ask only the minimum clarifying questions needed when uncertainty would change "
                "the config, split, or workflow.\n"
                "Focus on identifying:\n"
                "- which groups are model inputs\n"
                "- which groups are supervision targets\n"
                "- which groups are support-only (masking, preprocessing, evaluation)\n"
                "- which workflows are intended now: train, prediction, evaluation\n"
                "- whether multiple dataset roots/cohorts should be merged or assigned different roles\n"
                "Do not invent the task; the user must specify it."
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
            "content": (
                "Design a KonfAI config-writing strategy.\n\n"
                f"Task: {task}\n"
                f"Modeling intent: {modeling_intent}\n"
                f"Dataset summary:\n{dataset_summary}\n\n"
                "If the task, group roles, workflows, and split are already clear, proceed without "
                "asking questions.\n"
                "If something remains ambiguous and would materially change the config, state the "
                "smallest necessary clarifying question first.\n"
                "Use guide://config-design first, then consult docs://patching, docs://modeling, "
                "docs://configuration, and template resources only if needed.\n"
                "Explain the likely consequences for patch_size, extend_slice, dataset groups, "
                "and output definitions, but do not hardcode a final answer without reasoning."
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
            "content": (
                "You are debugging a KonfAI config.\n\n"
                f"Warnings:\n{warning_summary}\n\n"
                f"Config summary:\n{config_summary or '(not provided)'}\n\n"
                "Explain what these warnings mean, what assumptions may be wrong, and what to check next "
                "before editing the YAML. Prefer reasoning from docs/resources over hardcoded rules."
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


@mcp.tool(
    description=(
        "Use first when a dataset path may contain nested roots, cohorts, or ambiguous structure. "
        "This returns a bounded file tree and candidate dataset roots. "
        "It does not infer the task or write configs. "
        "Inputs: dataset_dir, optional depth, optional max_entries. "
        "Outputs: tree, candidate_dataset_roots, common_groups, missing_by_case, and next_actions. "
        "Next: inspect_dataset on one chosen root."
    )
)
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


@mcp.tool(
    description=(
        "Use to READ a dataset's small non-image text file: a labels/metadata CSV or TSV, a JSON/YAML "
        "sidecar, a case-list txt, or a text header (.mhd/.nhdr). "
        "SAFE: bounded read-only preview (it streams at most max_chars characters); binary files are "
        "refused with a pointer to inspect_dataset/preview_volume. "
        "It does not parse image volumes and does not modify anything. "
        "Inputs: path, optional max_lines, optional max_chars. "
        "Outputs: content (bounded), total_bytes, truncated; CSV/TSV additionally get columns + rows. "
        "Next: inspect_dataset or design_config_strategy."
    )
)
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


@mcp.tool(
    description=(
        "Use after the dataset root is chosen (browse_dataset first when the root is ambiguous). "
        "This returns group structure, dataset entry hints, ambiguities, and (by default) sampled statistics for "
        "one dataset root; pass include_stats=False for a fast structure-only scan, or groups=[one group] for "
        "focused statistics. "
        "It does not infer the task or choose a final model. "
        "Inputs: dataset_dir, optional groups, optional extension, optional max_cases_per_group, optional seed, "
        "optional include_stats. "
        "Outputs: groups, statistics, dataset_entry hints, warnings, and next_actions. "
        "Next: design_config_strategy."
    )
)
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


@mcp.tool(
    description=(
        "Use once the user task is known and the dataset root is understood. "
        "This builds a config-writing plan from task, one or more dataset roots, "
        "group roles, workflows, and modeling intent. "
        "It does not write YAML or launch runs. "
        "Inputs: task, dataset_dir or dataset_dirs, optional group_roles, optional workflows, optional "
        "modeling_intent, optional example, optional extension. "
        "Outputs: dataset_summary, config_plan, customization_options, unresolved_questions, compatible_examples, "
        "guidance_resources, next_actions, and optional next_resources. "
        "Next: initialize_session, optionally write_session_file, then write_workflow_config."
    )
)
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


@mcp.tool(
    description=(
        "Use when choosing, customizing, or debugging any configurable object classpath such as a model, loss, "
        "transform, or helper module — including a declarative YAML model ('default|<Name>.yml' from the shipped "
        "catalog, or a session-local .yml). "
        "This returns local or imported signature details, defaults, doc summary, and detected contract hints; "
        "for a YAML model it returns its hyperparameters (all overridable from the run config), the "
        "loss-attachable terminal_leaves paths for outputs_criterions, the full yaml_content, and how_to_adapt "
        "guidance (override hyperparameters vs copy-and-edit the structure) — parsed statically, never built. "
        "It does not validate the full workflow config or decide which object to use. "
        "Inputs: classpath. Local Module:Object classpaths are resolved inside the current session workspace and "
        "parsed statically (never executed); an installed-library classpath is imported to read its signature, but "
        "that import runs in an isolated subprocess so its side effects never touch the server process. "
        "Outputs: source type, signature, parameters, defaults, detected_contract, limitations, and next_actions. "
        "Next: write_session_file, write_workflow_config, or review_config_semantics."
    )
)
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


@mcp.tool(
    description=(
        "Use to DISCOVER which KonfAI components exist before authoring a config from scratch, when you do not "
        "already know the class name/path to put in the YAML. "
        "This enumerates the built-in component zoo for one kind. "
        "It does not return full constructor signatures -- chain to inspect_object_signature for that. "
        f"Inputs: kind, one of {COMPONENT_KINDS} (aliases: loss/metric -> criterion, etc.). "
        "Outputs: components [{name, config_reference, inspect_classpath, module, doc}], a reference_hint explaining "
        "where the name goes in the config, and next_actions. "
        "Next: inspect_object_signature on a chosen component, then design_config_strategy or write_workflow_config."
    )
)
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


@mcp.tool(
    description=(
        "Use at the start, or whenever you need to orient, to learn what KonfAI can do and which tool to reach for. "
        "This returns a capability overview: the three workflows, the component kinds (+ list_components), the "
        "extension model (+ describe_extension_points, including external-library classpaths), modeling modes, "
        "advanced capabilities, and which actions are safe vs should prefer human confirmation. "
        "It is a router to other tools and to AGENTS.md (the canonical reference), not a workflow to execute. "
        "Inputs: none. Outputs: a structured capability map + next_actions. "
        "Next: inspect_dataset, describe_config_schema, list_components, describe_extension_points."
    )
)
def describe_konfai_capabilities() -> dict[str, Any]:
    """Overview of what KonfAI can do and which MCP primitive to use for each capability."""
    return _describe_konfai_capabilities()


@mcp.tool(
    description=(
        "Use before authoring a config to learn the top-level schema of a workflow. "
        "This is GENERATED from the Trainer/Predictor/Evaluator constructor via KonfAI's reflection engine, so it "
        "never drifts: it returns each top-level field with its type, default, whether it is required, and -- for "
        "nested config objects -- a classpath to drill into with inspect_object_signature. "
        "It does not return a full ready-to-run config; combine it with the example templates. "
        "Inputs: workflow (train/prediction/evaluation), optional path (drill into nested config levels with "
        "their YAML keys, e.g. path='Dataset.Patch' or path='Model'). "
        "Outputs: root_key, yaml_path, fields[{name,yaml_key,type,default,default_hidden,required,"
        "nested_config_classpath}], next_actions. Each field's yaml_key is the LITERAL key to write in the YAML "
        "(no casing guesses); default_hidden=true means the default exists but is not JSON-serializable here. "
        "Next: describe_config_schema with a deeper path, inspect_object_signature, then write_workflow_config."
    )
)
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


@mcp.tool(
    description=(
        "Use when you want to ADD or EXTEND a component (a new loss, metric, model/network, augmentation, transform, "
        "scheduler, or pretrained model) and need to know exactly where/how to plug it into KonfAI. "
        "This returns the extension contract per kind: the base class to subclass, the required methods and "
        "return/forward contract, where it is referenced in the YAML, and the THREE classpath syntaxes -- builtin name, "
        "local `File:Class`, and external `package.module:Class` (e.g. `monai.losses:DiceLoss`) -- plus the load-bearing "
        "gotcha for that kind. "
        "It does not write code or fetch anything. "
        "Inputs: optional kind (loss/metric/model/augmentation/transform/scheduler/pretrained; omit for all). "
        "Outputs: extension_point(s), yaml_reference_syntax, principle, next_actions. "
        "Next: check_external_dependency, list_components, inspect_object_signature, then write_session_file."
    )
)
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


@mcp.tool(
    description=(
        "Use to PRE-FLIGHT an external library before integrating a brick from it (e.g. before referencing "
        "`monai.losses:DiceLoss` or wrapping `segmentation_models_pytorch`). "
        "This reports whether the library is importable, its version and license, whether it is already a KonfAI "
        "dependency, and an install hint -- WITHOUT importing the library into the server process (no import side "
        "effects run here). Only the TOP-LEVEL package is checked ('monai.losses' checks 'monai'): it answers "
        "'not installed' vs 'installed', not whether the submodule or class exists -- use inspect_object_signature "
        "to verify the full classpath. "
        "Inputs: module (e.g. 'monai' or 'monai.losses'), optional object name. "
        "Outputs: installed, version, license, distribution, is_konfai_dependency, install_hint, caution, next_actions. "
        "Next: inspect_object_signature on the chosen classpath, or describe_extension_points to write a wrapper."
    )
)
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


@mcp.tool(
    description=(
        "Use FIRST when the user wants a result from an existing model and has NOT asked to train one: check "
        "whether a published KonfAI app already does what they want, before authoring and training a config from "
        "scratch. An app can do any task, so judge whether one fits from its own description (read it with "
        "describe_app) and its declared inputs/outputs -- not from any preset list of tasks. "
        "This enumerates apps from a referenced catalogue (shipped default + the editable workspace file + the "
        "KONFAI_MCP_APP_CATALOG env file), expanding bare HuggingFace repo ids into their contained apps. "
        "It does not run inference or import any app code; without include_summary it does not even resolve manifests. "
        "Inputs: optional repos (ad-hoc override list), optional include_summary (fetch each display_name / "
        "short_description / modality, slower), optional force_update (refresh HuggingFace listings). "
        "Outputs: apps [{ref, source, repo, app_name}], catalog provenance, errors, next_actions. "
        "Next: describe_app on a candidate to read its manifest, else design_config_strategy to train one."
    )
)
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


@mcp.tool(
    description=(
        "Use to read one app's manifest so you can decide whether it matches the user's task -- the app's free-text "
        "description is the primary signal, with the input/output modality confirming the fit. "
        "This resolves a single app and returns its app.json: display name, description, input and output modality "
        "(with volume types), inference/evaluation/uncertainty capabilities, checkpoints, and segmentation "
        "terminology. "
        "It is metadata-only and SAFE: it does not import the app's model code and does not pip-install its "
        "requirements (those happen only later, behind an explicit trust gate). "
        "Inputs: ref (an app id 'repo_id:app_name', a local app folder path, or 'host:port:name' for a remote "
        "server -- a bare HuggingFace repo_id is NOT accepted here; expand it into app ids with list_apps first), "
        "optional force_update. "
        "Outputs: display_name, description, inputs, outputs, capabilities, checkpoints, terminology, next_actions. "
        "Next: run_app_infer / list_app_parameters / fine_tune_app when it fits (next_actions reflect the app's "
        "capabilities), or design_config_strategy if no app fits the task."
    )
)
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


@mcp.tool(
    description=(
        "Use to DISCOVER an app's tunable model parameters (and their allowed values) before tuning a run with "
        "set_parameters. Returns {values, constraints}: current values plus Literal/Range/Choices constraints "
        "derived from the model's typed signature. "
        "TRUST GATE: deriving constraints imports the app's model code, so pass allow_untrusted_code=True; the "
        "import runs in an isolated spawn subprocess, never in the server process. "
        "Local/HuggingFace apps only (a remote server does not expose this). "
        "Inputs: ref, allow_untrusted_code, optional force_update. "
        "Outputs: values, constraints, next_actions. Next: run_app_infer / run_app_pipeline with set_parameters."
    )
)
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


@mcp.tool(
    description=(
        "Use to SAVE a HuggingFace / remote-cached app (optionally with tuned parameters) as a local, editable app "
        "bundle -- the reproducibility artifact a challenge submission wants. It copies the app's files and, when "
        "set_parameters is given, bakes those values into the copied config. Distinct from package_app_from_session, "
        "which packages a model YOU trained this session. "
        "It copies files and rewrites config only (no model-code import). Local/HuggingFace apps only. "
        "Inputs: ref, path (destination folder), optional display_name, optional set_parameters, force_update. "
        "Outputs: exported_to, next_actions. Next: describe_app / run_app_infer / register_app_source on the copy."
    )
)
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


@mcp.tool(
    description=(
        "Use when the user points at their own app or HuggingFace repo and wants it to persist across sessions. "
        "This appends an app reference to the editable workspace catalogue file (the same one list_apps merges). "
        "It does not validate that the reference resolves -- call describe_app to check. "
        "Inputs: ref (an app id or a bare HuggingFace repo_id). "
        "Outputs: ref, added flag, catalog_path, the updated apps list, next_actions. "
        "Next: list_apps or describe_app."
    )
)
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


@mcp.tool(
    description=(
        "Use to drop an app reference previously added to the workspace catalogue. "
        "This removes a reference from the editable workspace catalogue file. "
        "It does not touch the shipped default catalogue or the KONFAI_MCP_APP_CATALOG env file. "
        "Inputs: ref. "
        "Outputs: ref, removed flag, catalog_path, the updated apps list, next_actions. "
        "Next: list_apps."
    )
)
def unregister_app_source(
    ref: Annotated[str, Field(description="App reference to remove from the editable workspace catalogue file.")],
) -> dict[str, Any]:
    """Remove an app reference from the editable workspace app catalogue."""
    return APP_SERVICE.unregister_app_source(ref)


@mcp.tool(
    description=(
        "Use to RUN a published KonfAI app on the user's data (the 'use an existing model instead of training' "
        "path), after describe_app confirmed the app fits. This launches a tracked inference job and reassembles "
        "the app's outputs into the given output directory. "
        "TRUST GATE: for a local or HuggingFace app, resolving it imports the app's Python code and pip-installs "
        "its requirements, so you MUST pass allow_untrusted_code=True to confirm you trust the source; a remote "
        "app (host:port:name) runs on the user's own server and needs no code gate (its inputs are uploaded there). "
        "It does not choose the app or prepare the data for you, and set_parameters is local/HuggingFace only. "
        "Inputs: ref (app id 'repo_id:app_name', local app folder path, or 'host:port:name[|token]'); inputs as a "
        "list of GROUPS (one inner list per input channel/modality, each a list of file or directory paths, paired "
        "by order across groups); optional output dir; optional gpu/cpu; optional tta, ensemble, ensemble_models, "
        "patch_size, batch_size, uncertainty; optional set_parameters (NAME->VALUE model tuning, e.g. "
        "{'iterations': 300}); allow_untrusted_code; force_update. "
        "Outputs: a job payload (status, resources, next_actions) plus mode and output. "
        "Next: wait_for_job, then inspect the output directory."
    )
)
def run_app_infer(
    ref: Annotated[
        str, Field(description="App id 'repo_id:app_name', local app folder path, or remote 'host:port:name[|token]'.")
    ],
    inputs: Annotated[
        list[list[str]],
        Field(
            description="Input GROUPS: one inner list per input channel/modality, each a list of file or directory paths, paired by order across groups."
        ),
    ],
    output: Annotated[
        str | None,
        Field(
            description="Output directory for the reassembled predictions (default: a unique dir under the session workspace AppOutputs/)."
        ),
    ] = None,
    gpu: Annotated[
        list[int] | None,
        Field(description="GPU indices to run on (default: every visible GPU); an empty list forces CPU."),
    ] = None,
    cpu: Annotated[int | None, Field(description="CPU worker count; setting it without gpu forces a CPU run.")] = None,
    tta: Annotated[
        int, Field(description="Number of test-time augmentations (0 disables; see the app's maximum_tta).")
    ] = 0,
    ensemble: Annotated[
        int,
        Field(description="Number of checkpoints to ensemble; 0 with no ensemble_models uses every app checkpoint."),
    ] = 0,
    ensemble_models: Annotated[
        list[str] | None,
        Field(description="Explicit checkpoint names to ensemble (see describe_app checkpoints; overrides ensemble)."),
    ] = None,
    patch_size: Annotated[
        list[int] | None,
        Field(description="Force the inference patch size (overrides the app's VRAM plan and config default)."),
    ] = None,
    batch_size: Annotated[
        int | None,
        Field(description="Force the inference batch size (overrides the app's VRAM plan and config default)."),
    ] = None,
    set_parameters: Annotated[
        dict[str, Any] | None,
        Field(
            description="Model tuning NAME->VALUE overrides (e.g. {'iterations': 300}); local/HuggingFace apps only."
        ),
    ] = None,
    uncertainty: Annotated[
        bool,
        Field(description="Keep the multi-channel inference stacks that run_app_uncertainty consumes (default False)."),
    ] = False,
    allow_untrusted_code: Annotated[
        bool,
        Field(
            description="Must be True for a local/HuggingFace app: resolving imports its Python code and pip-installs its requirements."
        ),
    ] = False,
    force_update: Annotated[
        bool, Field(description="Re-download the app files instead of reusing the local cache.")
    ] = False,
) -> dict[str, Any]:
    """Run a published KonfAI app on the user's data as a tracked inference job (local / HuggingFace / remote)."""
    # json.dumps keeps each value's type through the YAML re-parse in _apply_config_overrides: an int
    # stays an int, but a string "true"/"1" stays a string instead of being coerced to a bool/int.
    config_overrides = (
        [f"{name}={json.dumps(value)}" for name, value in set_parameters.items()] if set_parameters else None
    )
    spec = APP_SERVICE.prepare_infer(
        ref=ref,
        inputs=inputs,
        output=output,
        gpu=gpu,
        cpu=cpu,
        tta=tta,
        ensemble=ensemble,
        ensemble_models=ensemble_models,
        patch_size=patch_size,
        batch_size=batch_size,
        config_overrides=config_overrides,
        uncertainty=uncertainty,
        allow_untrusted_code=allow_untrusted_code,
        force_update=force_update,
    )
    return _launch_app_job(spec)


@mcp.tool(
    description=(
        "Use to score an app's predictions against ground truth with the app's OWN evaluation config "
        "(its shipped Evaluation.yml and metrics), after describe_app reported capabilities.evaluation. This is "
        "distinct from run_evaluation, which needs a hand-authored session Evaluation.yml. It launches a tracked "
        "job and writes the metric JSON to the output directory. "
        "TRUST GATE: a local/HuggingFace app imports its code and pip-installs (pass allow_untrusted_code=True); a "
        "remote app runs on the user's server. "
        "Inputs: ref; inputs (predictions, list of groups of paths); gt (ground truth, list of groups); optional "
        "output, mask, evaluation_file, gpu/cpu; allow_untrusted_code; force_update. "
        "Outputs: a job payload plus mode and output. Next: wait_for_job, then read the metric JSON."
    )
)
def run_app_evaluate(
    ref: Annotated[
        str, Field(description="App id 'repo_id:app_name', local app folder path, or remote 'host:port:name[|token]'.")
    ],
    inputs: Annotated[
        list[list[str]],
        Field(
            description="Prediction volumes as GROUPS (one inner list per group, each a list of file/dir paths, paired by order)."
        ),
    ],
    gt: Annotated[list[list[str]], Field(description="Ground-truth volumes as GROUPS, paired with inputs by order.")],
    output: Annotated[
        str | None,
        Field(
            description="Output directory for the metric JSON (default: a unique dir under the session workspace AppEvaluations/)."
        ),
    ] = None,
    mask: Annotated[
        list[list[str]] | None, Field(description="Mask volumes as GROUPS restricting the evaluated region.")
    ] = None,
    evaluation_file: Annotated[
        str, Field(description="Which evaluation config of the app to run (default 'Evaluation.yml').")
    ] = "Evaluation.yml",
    gpu: Annotated[
        list[int] | None,
        Field(description="GPU indices to run on (default: every visible GPU); an empty list forces CPU."),
    ] = None,
    cpu: Annotated[int | None, Field(description="CPU worker count; setting it without gpu forces a CPU run.")] = None,
    allow_untrusted_code: Annotated[
        bool,
        Field(
            description="Must be True for a local/HuggingFace app: resolving imports its Python code and pip-installs its requirements."
        ),
    ] = False,
    force_update: Annotated[
        bool, Field(description="Re-download the app files instead of reusing the local cache.")
    ] = False,
) -> dict[str, Any]:
    """Score predictions vs ground truth with a published app's own evaluation config, as a tracked job."""
    return _launch_app_job(
        APP_SERVICE.prepare_evaluate(
            ref=ref,
            inputs=inputs,
            gt=gt,
            output=output,
            mask=mask,
            evaluation_file=evaluation_file,
            gpu=gpu,
            cpu=cpu,
            allow_untrusted_code=allow_untrusted_code,
            force_update=force_update,
        )
    )


@mcp.tool(
    description=(
        "Use to produce uncertainty maps from an app, after describe_app reported capabilities.uncertainty. This "
        "runs the app's Uncertainty.yml on multi-channel inference stacks (typically produced by run_app_infer with "
        "uncertainty=True). It is the separate step that consumes those stacks; run_app_infer's uncertainty flag "
        "only keeps the stack during inference. "
        "TRUST GATE: local/HuggingFace apps import code and pip-install (pass allow_untrusted_code=True); remote runs "
        "on the user's server. "
        "Inputs: ref; inputs (the multi-channel inference stacks, list of groups); optional output, uncertainty_file, "
        "gpu/cpu; allow_untrusted_code; force_update. "
        "Outputs: a job payload plus mode and output. Next: wait_for_job, then inspect the uncertainty maps."
    )
)
def run_app_uncertainty(
    ref: Annotated[
        str, Field(description="App id 'repo_id:app_name', local app folder path, or remote 'host:port:name[|token]'.")
    ],
    inputs: Annotated[
        list[list[str]],
        Field(
            description="Multi-channel inference stacks as GROUPS (typically produced by run_app_infer with uncertainty=True)."
        ),
    ],
    output: Annotated[
        str | None,
        Field(
            description="Output directory for the uncertainty maps (default: a unique dir under the session workspace AppUncertainties/)."
        ),
    ] = None,
    uncertainty_file: Annotated[
        str, Field(description="Which uncertainty config of the app to run (default 'Uncertainty.yml').")
    ] = "Uncertainty.yml",
    gpu: Annotated[
        list[int] | None,
        Field(description="GPU indices to run on (default: every visible GPU); an empty list forces CPU."),
    ] = None,
    cpu: Annotated[int | None, Field(description="CPU worker count; setting it without gpu forces a CPU run.")] = None,
    allow_untrusted_code: Annotated[
        bool,
        Field(
            description="Must be True for a local/HuggingFace app: resolving imports its Python code and pip-installs its requirements."
        ),
    ] = False,
    force_update: Annotated[
        bool, Field(description="Re-download the app files instead of reusing the local cache.")
    ] = False,
) -> dict[str, Any]:
    """Run a published app's uncertainty estimation on inference stacks, as a tracked job."""
    return _launch_app_job(
        APP_SERVICE.prepare_uncertainty(
            ref=ref,
            inputs=inputs,
            output=output,
            uncertainty_file=uncertainty_file,
            gpu=gpu,
            cpu=cpu,
            allow_untrusted_code=allow_untrusted_code,
            force_update=force_update,
        )
    )


@mcp.tool(
    description=(
        "Use to run an app end to end in one shot: inference, then evaluation (when gt is given), then uncertainty. "
        "It writes Predictions / Evaluations / Uncertainties under the output directory. Prefer run_app_infer for a "
        "plain prediction; use this when you want the app's full scoring loop in a single call. "
        "TRUST GATE: local/HuggingFace apps import code and pip-install (pass allow_untrusted_code=True); remote runs "
        "on the user's server (set_parameters is local/HuggingFace only). "
        "Inputs: ref; inputs (list of groups); optional gt (enables evaluation), mask, output, gpu/cpu, tta, ensemble, "
        "ensemble_models, patch_size, batch_size, uncertainty (default true), set_parameters; allow_untrusted_code; "
        "force_update. "
        "Outputs: a job payload plus mode and output. Next: wait_for_job, then inspect the output subdirectories."
    )
)
def run_app_pipeline(
    ref: Annotated[
        str, Field(description="App id 'repo_id:app_name', local app folder path, or remote 'host:port:name[|token]'.")
    ],
    inputs: Annotated[
        list[list[str]],
        Field(
            description="Input GROUPS: one inner list per input channel/modality, each a list of file or directory paths, paired by order across groups."
        ),
    ],
    gt: Annotated[
        list[list[str]] | None,
        Field(description="Ground-truth volumes as GROUPS; providing them enables the evaluation stage."),
    ] = None,
    output: Annotated[
        str | None,
        Field(
            description="Output directory for the Predictions/Evaluations/Uncertainties subdirs (default: a unique dir under the session workspace AppPipelines/)."
        ),
    ] = None,
    mask: Annotated[
        list[list[str]] | None, Field(description="Mask volumes as GROUPS restricting the evaluated region.")
    ] = None,
    tta: Annotated[
        int, Field(description="Number of test-time augmentations (0 disables; see the app's maximum_tta).")
    ] = 0,
    ensemble: Annotated[
        int,
        Field(description="Number of checkpoints to ensemble; 0 with no ensemble_models uses every app checkpoint."),
    ] = 0,
    ensemble_models: Annotated[
        list[str] | None,
        Field(description="Explicit checkpoint names to ensemble (see describe_app checkpoints; overrides ensemble)."),
    ] = None,
    patch_size: Annotated[
        list[int] | None,
        Field(description="Force the inference patch size (overrides the app's VRAM plan and config default)."),
    ] = None,
    batch_size: Annotated[
        int | None,
        Field(description="Force the inference batch size (overrides the app's VRAM plan and config default)."),
    ] = None,
    set_parameters: Annotated[
        dict[str, Any] | None,
        Field(
            description="Model tuning NAME->VALUE overrides (e.g. {'iterations': 300}); local/HuggingFace apps only."
        ),
    ] = None,
    uncertainty: Annotated[bool, Field(description="Run the uncertainty stage (default True).")] = True,
    gpu: Annotated[
        list[int] | None,
        Field(description="GPU indices to run on (default: every visible GPU); an empty list forces CPU."),
    ] = None,
    cpu: Annotated[int | None, Field(description="CPU worker count; setting it without gpu forces a CPU run.")] = None,
    allow_untrusted_code: Annotated[
        bool,
        Field(
            description="Must be True for a local/HuggingFace app: resolving imports its Python code and pip-installs its requirements."
        ),
    ] = False,
    force_update: Annotated[
        bool, Field(description="Re-download the app files instead of reusing the local cache.")
    ] = False,
) -> dict[str, Any]:
    """Run a published app's full infer -> evaluate -> uncertainty pipeline as a tracked job."""
    # json.dumps keeps each value's type through the YAML re-parse in _apply_config_overrides: an int
    # stays an int, but a string "true"/"1" stays a string instead of being coerced to a bool/int.
    config_overrides = (
        [f"{name}={json.dumps(value)}" for name, value in set_parameters.items()] if set_parameters else None
    )
    return _launch_app_job(
        APP_SERVICE.prepare_pipeline(
            ref=ref,
            inputs=inputs,
            gt=gt,
            output=output,
            mask=mask,
            tta=tta,
            ensemble=ensemble,
            ensemble_models=ensemble_models,
            patch_size=patch_size,
            batch_size=batch_size,
            config_overrides=config_overrides,
            uncertainty=uncertainty,
            gpu=gpu,
            cpu=cpu,
            allow_untrusted_code=allow_untrusted_code,
            force_update=force_update,
        )
    )


@mcp.tool(
    description=(
        "Use to TRAIN by starting from a published app instead of a blank slate: fine-tune an existing app's "
        "checkpoint(s) on the user's dataset. This is the middle option between run_app_infer (use as-is, no "
        "training) and design_config_strategy (author a config and train from scratch). It launches a tracked "
        "training job and writes a resolvable app bundle (config + code + fine-tuned checkpoint) to the output "
        "directory, which you can then run with run_app_infer. "
        "TRUST GATE: a local or HuggingFace app imports its Python code and pip-installs its requirements, so pass "
        "allow_untrusted_code=True to confirm you trust the source; a remote app trains on the user's own server "
        "(the dataset is uploaded there) and needs no code gate. "
        "It does not author a config or adapt the dataset layout for you. "
        "Inputs: ref (app id, local app folder path, or 'host:port:name[|token]'); dataset (a KonfAI-style dataset "
        "directory); optional output bundle dir; optional name, epochs, it_validation, models (which checkpoints to "
        "fine-tune), lr; optional set_parameters (NAME->VALUE model/config tweaks baked into the training config, "
        "e.g. {'iterations': 300}; local/HuggingFace only); gpu/cpu; allow_untrusted_code; force_update. "
        "Outputs: a job payload (status, resources, next_actions) plus mode and the bundle output path. "
        "Next: wait_for_job, then run_app_infer on the produced bundle (then run_app_evaluate to score and rank "
        "this fine-tune against other training trials via leaderboard / compare_runs)."
    )
)
def fine_tune_app(
    ref: Annotated[
        str, Field(description="App id 'repo_id:app_name', local app folder path, or remote 'host:port:name[|token]'.")
    ],
    dataset: Annotated[
        str,
        Field(description="KonfAI-style dataset directory to fine-tune on (must exist; uploaded for a remote app)."),
    ],
    output: Annotated[
        str | None,
        Field(
            description="Destination for the produced app bundle (default: a unique dir under the session workspace AppBundles/)."
        ),
    ] = None,
    name: Annotated[str, Field(description="Run name of the fine-tune training (default 'Finetune').")] = "Finetune",
    epochs: Annotated[int, Field(description="Number of training epochs (must be > 0; default 10).")] = 10,
    it_validation: Annotated[
        int, Field(description="Iterations between validation/checkpoint steps (KonfAI it_validation; default 1000).")
    ] = 1000,
    models: Annotated[
        list[str] | None,
        Field(description="Which app checkpoints to fine-tune (default: the app's first advertised checkpoint)."),
    ] = None,
    lr: Annotated[
        float | None, Field(description="Learning-rate override; omit to keep the app config's value.")
    ] = None,
    set_parameters: Annotated[
        dict[str, Any] | None,
        Field(
            description="Model/config tuning NAME->VALUE overrides baked into the training config before fine-tuning "
            "(e.g. {'iterations': 300}); local/HuggingFace apps only."
        ),
    ] = None,
    gpu: Annotated[
        list[int] | None,
        Field(description="GPU indices to train on (default: every visible GPU); an empty list forces CPU."),
    ] = None,
    cpu: Annotated[int | None, Field(description="CPU worker count; setting it without gpu forces a CPU run.")] = None,
    config_file: Annotated[
        str, Field(description="Which train config of the app to use (default 'Config.yml').")
    ] = "Config.yml",
    allow_untrusted_code: Annotated[
        bool,
        Field(
            description="Must be True for a local/HuggingFace app: resolving imports its Python code and pip-installs its requirements."
        ),
    ] = False,
    force_update: Annotated[
        bool, Field(description="Re-download the app files instead of reusing the local cache.")
    ] = False,
) -> dict[str, Any]:
    """Fine-tune a published KonfAI app on the user's dataset and produce a resolvable app bundle."""
    # json.dumps keeps each value's type through the YAML re-parse in _apply_config_overrides: an int
    # stays an int, but a string "true"/"1" stays a string instead of being coerced to a bool/int.
    config_overrides = (
        [f"{key}={json.dumps(value)}" for key, value in set_parameters.items()] if set_parameters else None
    )
    spec = APP_SERVICE.prepare_finetune(
        ref=ref,
        dataset=dataset,
        output=output,
        name=name,
        epochs=epochs,
        it_validation=it_validation,
        models=models,
        lr=lr,
        config_overrides=config_overrides,
        gpu=gpu,
        cpu=cpu,
        config_file=config_file,
        allow_untrusted_code=allow_untrusted_code,
        force_update=force_update,
    )
    return _launch_app_job(spec)


@mcp.tool(
    description=(
        "Use to PACKAGE a model trained in the current session (the train-from-scratch branch) into a resolvable "
        "KonfAI app bundle -- the same endpoint fine_tune_app produces, so a from-scratch run can also finish as a "
        "reusable app. It gathers the session's checkpoints and a config, writes an app.json from the metadata you "
        "give, and assembles a bundle (app.json + config + checkpoint + optional Model.py/requirements) that "
        "describe_app and run_app_infer can then consume. "
        "It does not train, and it does not upload the bundle anywhere. "
        "Inputs: name (bundle folder), display_name, description, optional short_description/tta/mc_dropout; optional "
        "checkpoints and configs (default: discovered from the session Checkpoints/ and Prediction.yml/Config.yml); "
        "optional model_py, requirements, output dir; optional onnx export (onnx, onnx_patch_size, onnx_in_channels). "
        "Outputs: bundle_path, the packaged checkpoints/configs, next_actions (and onnx path if requested). "
        "Next: describe_app or run_app_infer on the produced bundle."
    )
)
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


@mcp.tool(
    description=(
        "Use when the dataset has the right content but the group filenames do not match your intended config. "
        "This creates copied, symlinked, or moved aliases for dataset files. "
        "It does not change YAML configs for you. "
        "Inputs: dataset_dir, rename_map, optional mode, optional overwrite, optional allow_destructive. "
        "Outputs: created paths, missing_by_case, and next_actions. "
        "Next: inspect_dataset or design_config_strategy."
    )
)
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


@mcp.tool(
    description=(
        "Use when moving from strategy to the concrete current session workspace. "
        "This creates or resets the current session workspace and can seed selected workflow files "
        "from one example template. "
        "DESTRUCTIVE with overwrite=True: it DELETES everything in the existing workspace, trained "
        "Checkpoints/ and Predictions/ included -- to keep those, switch_session to a new name instead. "
        "It does not adapt example YAML to your dataset automatically. Referenced .yml models are always "
        "seeded; an example whose model/loss lives in a local .py (e.g. Synthesis) needs "
        "include_support_files=True to be runnable -- the result carries a warning otherwise. "
        "Inputs: optional from_example, optional workflows, optional include_support_files, optional overwrite. "
        "Outputs: created workspace paths, copied files, resources, and next_actions. "
        "Next: write_workflow_config or inspect copied template files."
    )
)
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


@mcp.tool(
    description=(
        "Use to CREATE a named session workspace (and switch to it by default), so different experiments or "
        "config families live in isolated workspaces instead of overwriting one another. "
        "This creates sessions/<name> under the workspace root and makes it the current session. "
        "It does not seed configs (initialize_session does) and refuses to switch while a job is active. "
        "Inputs: name, optional switch (default true). "
        "Outputs: session, created, switched, sessions, next_actions. "
        "Next: initialize_session or import_experiment in the new session."
    )
)
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


@mcp.tool(
    description=(
        "Use to SWITCH the server onto another existing session workspace (create_session makes new ones). "
        "All session-scoped tools and resources then operate on that workspace; its persisted job history is "
        "reloaded. It refuses to switch while a job is active in the current session. "
        "Inputs: name. "
        "Outputs: session, sessions, summary, next_actions. "
        "Next: summarize_session, or leaderboard(session=...) to compare without switching."
    )
)
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


@mcp.tool(
    description=(
        "Use to ADOPT an existing on-disk KonfAI experiment (its Config/Prediction/Evaluation.yml, custom .py and "
        ".yml files, and optionally its Checkpoints/Predictions/Evaluations/Statistics/Dataset artifacts) into the "
        "current session workspace, so the server can read, validate, rerun, resume, and compare it. "
        "Artifacts are symlinked by default (no copy of large checkpoints); pass include_artifacts='copy' to copy "
        "or 'none' to import configs/code only. Existing session files are kept unless overwrite=True. "
        "Inputs: source_dir, optional include_artifacts (link|copy|none), optional overwrite. "
        "Outputs: source, copied, linked, skipped, next_actions. "
        "Next: read_session_file / review_config_semantics, then validate_config_semantics."
    )
)
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


@mcp.tool(
    description=(
        "Use for session-side support files such as local model code, custom losses, transforms, helper modules, "
        "or manifests. "
        "This writes one file inside the current session workspace. "
        "It does not validate Python semantics. "
        "Inputs: relative_path, content, optional overwrite. "
        "Outputs: written path, byte count, and next_actions. "
        "Next: inspect_object_signature, review_config_semantics, or validate_config_semantics."
    )
)
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


@mcp.tool(
    description=(
        "Use to READ BACK a file from the current session workspace: a config, a support file you wrote "
        "(Model.py, Loss.py), a copied template file (UNet.yml, Config_GAN.yml), a job config snapshot (the "
        "manifest's config_snapshots paths), or an evaluation JSON. "
        "This returns a bounded character range of one workspace file; absolute paths are accepted when they "
        "resolve inside the workspace. It does not read files outside the session workspace. "
        "Inputs: path (workspace-relative, or absolute inside the workspace), optional max_chars, optional offset. "
        "Outputs: path, relative_path, content, offset, returned_chars, total_bytes, truncated, next_actions. "
        "Next: write_session_file or write_workflow_config to edit, then review_config_semantics."
    )
)
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


@mcp.tool(
    description=(
        "Use to READ a file shipped with an example template — a reference implementation such as a local model "
        "(Model.py), a custom transform (UnNormalize.py), a declarative model (UNet.yml), or an alternate config "
        "(Config_GAN.yml) — so you can understand or adapt it instead of guessing what it contains. "
        "This returns a bounded character range of one template file. It does not modify templates. "
        "Inputs: name (template), filename (a direct child of the template directory), optional max_chars, "
        "optional offset. "
        "Outputs: template, filename, content, truncated, next_actions. "
        "Next: write_session_file to adapt it into the session, or initialize_session to copy files wholesale."
    )
)
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


@mcp.tool(
    description=(
        "Use to read the FULL evaluation metrics (per-case values + aggregates) of ONE named run, instead of the "
        "newest-file-only view of session://current/metrics — essential when comparing specific past runs. "
        "This reads Evaluations/<run_name>/Metric_<SPLIT>.json in the current session — or an app trial's "
        "metrics when run_name is a trial label as returned by leaderboard (an AppEvaluations/AppPipelines "
        "directory such as 'eval_app__iterations_300-1a2b3c4d'). It does not rerun evaluation. "
        "Inputs: run_name (a run's train_name OR an app-trial label from leaderboard), optional split (default "
        "TRAIN; the error lists available runs and splits on a miss), optional session (read another session's "
        "run without switching). "
        "Outputs: run_name, split, path, updated_at, metrics (full JSON), summary, next_actions. "
        "Next: leaderboard or summarize_session."
    )
)
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


@mcp.tool(
    description=(
        "Use to COMPARE two runs metric-by-metric on aligned cases: means, per-case deltas, and a "
        "direction-aware winner per metric (loss-like metrics count lower as better). "
        "This reads both runs' Metric_<SPLIT>.json; it does not rerun evaluation. "
        "Inputs: run_a, run_b, optional split (default TRAIN), optional metric (suffix filter), optional session. "
        "Outputs: metrics {direction, cases, mean_a/mean_b, mean_delta_b_minus_a, cases_better_a/b, winner, "
        "per_case_delta_b_minus_a}, next_actions. "
        "Next: get_run_metrics on the winner, or leaderboard."
    )
)
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


@mcp.tool(
    description=(
        "Use to read a run's TRAINING CURVES (loss/metric scalars over iterations) from the TensorBoard event "
        "files KonfAI writes under Statistics/<run_name>/ — the full history, not just the live log tail. "
        "This parses tfevents into downsampled scalar series. It requires the 'tensorboard' package. "
        "Inputs: run_name, optional tags (substring filters), optional max_points (default 200), optional session. "
        "Outputs: tags, curves {tag: [{step, value}]}, next_actions. "
        "Next: compare_runs or leaderboard."
    )
)
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


@mcp.tool(
    description=(
        "Use to EXPORT the full reproducibility record of one run: the job manifest (command, devices, "
        "environment snapshot with package versions and GPUs), the launch-time config snapshots' CONTENT, the "
        "post-run resolved config, every split's metrics, and a log tail — a Methods-section-grade record in "
        "one payload. "
        "It does not rerun anything. Caveat: resolved_config is read from the LIVE session config, which may "
        "have been rewritten since the run -- the launch-time truth is config_snapshots. "
        "Inputs: run_name OR job_id, optional log_lines (default 100). "
        "Outputs: job, manifest, config_snapshots (text), resolved_config, metrics per split, log_tail. "
        "Next: compare_runs or read_training_curves."
    )
)
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


@mcp.tool(
    description=(
        "Use to DIFF the exact configs two jobs ran with, from their immutable launch-time snapshots — "
        "'what changed between run A and run B' without trusting memory. "
        "It does not diff live session files (they may have been rewritten since). "
        "Inputs: job_id_a, job_id_b, optional filename (default Config.yml). "
        "Outputs: identical flag, unified diff text, next_actions. "
        "Next: compare_runs on the two runs' metrics."
    )
)
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


@mcp.tool(
    description=(
        "Use to ENUMERATE a model's addressable module paths — the exact keys outputs_criterions and "
        "outputs_dataset accept — instead of guessing dotted paths and reading MeasureError lists from failed "
        "runs. This builds the workflow from the session config (side-effect-free, like validation) and walks "
        "every Network's module graph; terminal=true marks output heads (deep-supervision losses attach to "
        "non-terminal paths). "
        "Inputs: workflow (default train), optional config_file (alternate train config). "
        "Outputs: networks {attr: [{path, terminal, module}]}, reference_hint, next_actions. "
        "Next: write_workflow_config, then validate_config_semantics."
    )
)
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


@mcp.tool(
    description=(
        "Use to SMOKE-TEST a component you wrote or referenced BEFORE wiring it into a config: it executes the "
        "component's runtime contract on dummy tensors. For a transform it asserts "
        "transform_shape(shape) == __call__(tensor).shape — the contract whose silent violation corrupts patch "
        "reassembly; for a criterion it reports Tensor-vs-tuple return (loss vs metric convention) and whether "
        "backward() works. "
        "TRUST: this imports and EXECUTES the component's code — in an isolated spawn subprocess, never in the "
        "server process — but still only run it on code you or the user wrote. "
        "Inputs: classpath (local File:Class, builtin name, or package.module:Class), kind "
        "(transform/criterion/loss/metric), optional shape (default [1,8,8,8]), optional init_kwargs. "
        "Outputs: ok, stage, contract details (predicted vs actual shape, returns, backward_ok) or the full "
        "traceback. "
        "Next: write_workflow_config when ok, or write_session_file to fix the component."
    )
)
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


@mcp.tool(
    description=(
        "Use when you want to remove the current session workspace. "
        "This deletes the workspace and can cancel active jobs when forced. "
        "It does not preserve artifacts. "
        "Inputs: optional force. "
        "Outputs: deleted session name and path. "
        "Next: none unless you want to reinitialize the session."
    )
)
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


@mcp.tool(
    description=(
        "Use after semantic review when the config looks coherent enough to instantiate. "
        "This instantiates or sets up KonfAI workflow objects in an ISOLATED spawn subprocess to catch "
        "runtime-facing errors: edited workspace code is always re-imported fresh, and nothing executes in the "
        "server process. It does not launch jobs. "
        "Levels: 'instantiate' builds the objects, 'setup' also builds the dataloader, 'train_step' additionally "
        "runs ONE forward+backward on ONE batch (train workflow only, single-process CPU, no checkpoint, config "
        "restored) to catch runtime-only errors -- target dtype/shape mismatches, an outputs_criterions key that "
        "does not resolve, a detached loss. "
        "Inputs: workflow or 'all' (validate every present config), validation level, and optional models for "
        "prediction. "
        "Outputs: ok flag, runtime details, semantic review, and next_actions. "
        "Next: run_train, run_prediction, run_evaluation, or fix the config."
    )
)
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


@mcp.tool(
    description=(
        "Use immediately after writing or editing a workflow config. "
        "This performs lightweight semantic checks and returns warnings plus blocking issues. "
        "It does not instantiate KonfAI runtime objects. "
        "Inputs: workflow. "
        "Outputs: summary, warnings, blocking_issues, next_checks, and next_actions. "
        "Next: validate_config_semantics if there are no blocking issues."
    )
)
def review_config_semantics(
    workflow: Annotated[
        WorkflowKind,
        Field(description="Which session config to review (default 'train')."),
    ] = "train",
) -> dict[str, Any]:
    """Review one config statically and emit lightweight semantic warnings before runtime validation."""
    return SESSION.review_config_semantics(workflow)


@mcp.tool(
    description=(
        "Use when you want one compact session snapshot for planning the next action. "
        "This returns readiness, job state, metric summaries, and an optional leaderboard, log tail, or "
        "config validation (include_validation=True; off by default to keep the payload lean). "
        "It does not launch or repair workflows. "
        "Inputs: optional leaderboard, log, and validation controls. "
        "Outputs: readiness, metrics_summary, validation, leaderboard, and next_actions. "
        "Next: review_config_semantics, validate_config_semantics, or run a workflow."
    )
)
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


@mcp.tool(
    description=(
        "Use to author or replace one workflow YAML file. "
        "This validates the top-level KonfAI root key and writes the config into the current session workspace. "
        "It does not patch YAML structurally for you. "
        "Inputs: workflow, content, optional overwrite. "
        "Outputs: written path, byte count, and next_actions. "
        "Next: review_config_semantics."
    )
)
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


@mcp.tool(
    description=(
        "Use when you need the current job registry state. "
        "This lists jobs for the current session workspace. "
        "It does not wait for jobs or parse live metrics. "
        "Inputs: none. "
        "Outputs: job payloads sorted by creation time. "
        "Next: get_job_status, wait_for_job, or read_live_metrics."
    )
)
def list_jobs() -> list[dict[str, Any]]:
    """List known jobs for the current session workspace."""
    with _JOBS_LOCK:
        jobs = list(_JOBS.values())
    jobs.sort(key=lambda job: job.created_at, reverse=True)
    return [_job_payload(job) for job in jobs]


@mcp.tool(
    description=(
        "Use after evaluation when you want ranked metrics across completed runs. "
        "This reads Metric_<split>.json files and builds a leaderboard. "
        "It does not rerun evaluation. "
        "Inputs: optional metric, optional split (default TRAIN; maps to Metric_<SPLIT>.json — a miss lists the "
        "available splits), optional limit, optional direction ('min'/'max' override when the ranking direction "
        "inferred from the metric name is wrong; applies to every metric in the payload), "
        "optional session (rank another session's runs without switching). "
        "Outputs: available_metrics, available_splits, selected_metric when resolved, leaderboard rows, "
        "best row, warnings, and next_actions. "
        "Next: get_run_metrics on a chosen run, summarize_session, or launch another run."
    )
)
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


@mcp.tool(
    description=(
        "Use after train config review and validation succeed. "
        "This launches a training job and returns structured job resources. "
        "It does not choose the device or repair config issues automatically -- omitting gpu trains on CPU, "
        "so pass gpu explicitly for GPU training. "
        "Inputs: optional gpu as an int or list of ints, optional cpu, overwrite, quiet, tensorboard, "
        "single_process, optional config_file (an alternate train config in the workspace, e.g. Config_GAN.yml), "
        "optional cluster ({name, memory, num_nodes, time_limit} submits via SLURM/submitit instead of running "
        "locally). Jobs on DISJOINT devices may run concurrently; same-device jobs are refused. "
        "Outputs: job payload with resources and next_actions; or, when a prerequisite is missing (dataset path, checkpoint), a blocker payload {ok, blocked, error, missing_paths, next_actions} with no job_id/status. "
        "Next: read_live_metrics or wait_for_job."
    )
)
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


@mcp.tool(
    description=(
        "Use to RESUME an interrupted or crashed training run from a checkpoint: model, optimizer, scheduler, and "
        "epoch/iteration counters are restored (KonfAI's RESUME command), unlike fine_tune_app which restarts from "
        "weights only. "
        "This launches a resumed training job from the current session Config.yml. "
        "It does not pick between runs: by default it resumes from the newest checkpoint of the configured run "
        "(falling back to the newest in the session), avoiding cross-run contamination. "
        "It trains up to the LIVE config's epochs: if the run already completed them, raise epochs in Config.yml "
        "first or the resume finishes immediately without adding checkpoints. "
        "Inputs: optional model (checkpoint path; default as above), optional lr (override the "
        "restored learning rate; omit to continue the schedule), optional gpu/cpu, overwrite, quiet, tensorboard, "
        "single_process. "
        "Outputs: job payload with resources and next_actions; or, when a prerequisite is missing (dataset path, checkpoint), a blocker payload {ok, blocked, error, missing_paths, next_actions} with no job_id/status. "
        "Next: wait_for_job or read_live_metrics."
    )
)
def run_resume(
    model: Annotated[
        str | None,
        Field(
            description="Checkpoint to resume from: a path (workspace-relative or absolute) or an http(s) URL (default: the configured run's newest checkpoint)."
        ),
    ] = None,
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


@mcp.tool(
    description=(
        "Use to RUN A SWEEP: launch several training configs SEQUENTIALLY server-side (each waits for the "
        "previous to finish), collecting per-run outcomes -- fold training or hyperparameter variants in one "
        "call instead of hand-chaining run_train/wait_for_job. "
        "This blocks until the batch ends, like wait_for_job; each config needs a distinct train_name. "
        "Inputs: config_files (alternate train configs in the workspace, e.g. from generate_folds or "
        "write_session_file), optional gpu/cpu, overwrite, quiet (default true), single_process, stop_on_error "
        "(default true). "
        "Outputs: requested, completed, results [{config_file, job_id, run_name, status, error}], next_actions. "
        "Next: leaderboard or compare_runs."
    )
)
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


@mcp.tool(
    description=(
        "Use to SPLIT a dataset into K cross-validation folds: writes one case-list file per fold into the "
        "session workspace and returns the exact subset stanzas to paste into the configs. "
        "KonfAI's Dataset.subset accepts a case-list file ('folds/fold_0.txt' keeps those cases) and its "
        "'~file' negation (trains on every OTHER fold). "
        "Inputs: dataset_dir, optional k (default 5), optional seed. "
        "Outputs: folds {fold_i: {cases, file, train_subset, eval_subset}}, how_to_use, next_actions. "
        "Next: write per-fold configs (distinct train_name each), then run_batch."
    )
)
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
    workspace = WORKSPACE_LAYOUT.ensure_session_workspace_exists()
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


@mcp.tool(
    description=(
        "Use to SEE a volume: returns one slice as a PNG image (rendered in image-capable MCP clients) for "
        "qualitative QC of a dataset case or a produced prediction -- orientation, field of view, obvious "
        "artefacts -- instead of judging from numbers alone. "
        "This reads any SimpleITK-readable file (mha/nii.gz/...), windows it between the 1st and 99th "
        "percentile, and downsamples to max_size. It does not modify the file. "
        "Inputs: path (volume file), optional slice_index (default: middle), optional axis (default 0 = "
        "first/depth axis), optional max_size (default 512). "
        "Outputs: a PNG image content block. "
        "Next: inspect_dataset for numbers, or preview_volume on other slices/axes."
    )
)
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


@mcp.tool(
    description=(
        "Use after prediction config review/validation and when a checkpoint exists. "
        "This launches a prediction job and returns structured job resources. "
        "It does not search outside the current session workspace for missing checkpoints. "
        "Inputs: optional models as a string or list of strings, optional "
        "gpu as an int or list of ints, optional cpu, overwrite, quiet, single_process. "
        "Outputs: job payload with resources and next_actions; or, when a prerequisite is missing (dataset path, checkpoint), a blocker payload {ok, blocked, error, missing_paths, next_actions} with no job_id/status. "
        "Next: wait_for_job."
    )
)
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


@mcp.tool(
    description=(
        "Use after evaluation config review/validation and when required artifacts exist. "
        "This launches an evaluation job and returns structured job resources. "
        "It does not infer missing predictions. "
        "Inputs: optional gpu as an int or list of ints, optional cpu, overwrite, quiet, single_process. "
        "Outputs: job payload with resources and next_actions; or, when a prerequisite is missing (dataset path, checkpoint), a blocker payload {ok, blocked, error, missing_paths, next_actions} with no job_id/status. "
        "Next: wait_for_job then summarize_session."
    )
)
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


@mcp.tool(
    description=(
        "Use when a running job should stop. "
        "This requests cancellation and waits briefly for a clean shutdown. "
        "It does not delete any session artifacts. "
        "Inputs: job_id and optional wait_s. "
        "Outputs: final job payload after cancellation. "
        "Next: summarize_session."
    )
)
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
        "Use when you need the latest state for one job without waiting. "
        "This returns the current job payload and suggested next actions. "
        "It does not parse runtime metrics. "
        "Inputs: job_id. "
        "Outputs: status payload with resources and next_actions. "
        "Next: wait_for_job or read_live_metrics."
    )
)
def get_job_status(
    job_id: Annotated[str, Field(description="Job identifier returned by a run_* launch or list_jobs.")],
) -> dict[str, Any]:
    """Return the current job status together with suggested next actions."""
    return _job_payload(JOB_REGISTRY.get(job_id))


@mcp.tool(
    description=(
        "Use to READ a job's log as a tool — the crash-triage primitive: tail more than the fixed resource tail, "
        "page through it, or filter it with a regex to find the traceback. "
        "This reads the job console log (or the KonfAI runtime log when present) and returns the selected lines. "
        "It does not parse metrics; use read_live_metrics for parsed metrics. "
        "Inputs: job_id, optional max_lines (default 200), optional grep (regex applied per line, over a bounded window of the last max(20*max_lines, 2000) lines, before the tail "
        "is taken), optional source ('auto' prefers the runtime log for running/done jobs and the console job log — where a crash traceback lives — for failed ones; or 'job'/'runtime'). "
        "Outputs: job_id, status, path, content, lines_returned, next_actions. "
        "Next: validate_config_semantics then the matching run_* tool to retry, or cancel_job."
    )
)
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


@mcp.tool(
    description=(
        "Use while a job is running and you want parsed runtime metrics instead of raw logs. "
        "This reads the runtime log and returns recent metric snapshots. "
        "It does not block until the job completes. "
        "Inputs: optional kind, optional job_id, optional max_entries. "
        "Outputs: latest metric snapshot, recent entries, by_stage summaries, and job metadata. "
        "Next: wait_for_job or summarize_session."
    )
)
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


@mcp.tool(
    description=(
        "Use after launching a job when you want to block until it finishes. "
        "This polls job state until the job reaches a terminal status. "
        "It does not stream logs. "
        "Inputs: job_id, optional timeout_s (omit/None to wait until the job finishes -- recommended for real "
        "multi-hour training; pass a number only to bound the wait, which raises TimeoutError on expiry), "
        "optional poll_interval_s. "
        "Outputs: final job payload. "
        "Next: summarize_session or leaderboard."
    )
)
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
