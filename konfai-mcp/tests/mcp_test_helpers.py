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

import asyncio
import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
from ruamel.yaml import YAML

yaml = YAML()
yaml.default_flow_style = False


def resource_to_text(resource: Any) -> str:
    if isinstance(resource, list):
        return "\n".join(getattr(item, "text", str(item)) for item in resource)
    return str(resource)


def yaml_dump(data: dict[str, Any]) -> str:
    from io import StringIO

    stream = StringIO()
    yaml.dump(data, stream)
    return stream.getvalue()


def write_image(path: Path, array: np.ndarray, pixel_id: int) -> None:
    import SimpleITK as sitk

    image = sitk.GetImageFromArray(array)
    image.SetSpacing((1.0, 1.0, 1.0))
    image = sitk.Cast(image, pixel_id)
    sitk.WriteImage(image, str(path))


def create_segmentation_dataset(dataset_dir: Path, image_group: str = "CT") -> None:
    import SimpleITK as sitk

    dataset_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(4):
        case_dir = dataset_dir / f"CASE_{idx:03d}"
        case_dir.mkdir()
        zz, yy, xx = np.meshgrid(
            np.linspace(-0.2, 0.2, 3, dtype=np.float32),
            np.linspace(-1.0, 1.0, 32, dtype=np.float32),
            np.linspace(-1.0, 1.0, 32, dtype=np.float32),
            indexing="ij",
        )
        radius = 0.28 + idx * 0.03
        center_x = -0.20 + idx * 0.10
        center_y = 0.15 - idx * 0.05
        seg = (((xx - center_x) ** 2 + (yy - center_y) ** 2) < radius**2).astype(np.uint8)
        seg &= (np.abs(zz) < 0.18).astype(np.uint8)
        img = (0.15 * yy - 0.10 * xx + seg * 0.85 + zz * 0.05 + idx * 0.02).astype(np.float32)
        write_image(case_dir / f"{image_group}.mha", img, sitk.sitkFloat32)
        write_image(case_dir / "SEG.mha", seg, sitk.sitkUInt8)


def create_synthesis_dataset(dataset_dir: Path) -> None:
    import SimpleITK as sitk

    dataset_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(4):
        case_dir = dataset_dir / f"CASE_{idx:03d}"
        case_dir.mkdir()
        zz, yy, xx = np.meshgrid(
            np.linspace(-0.2, 0.2, 3, dtype=np.float32),
            np.linspace(-1.0, 1.0, 16, dtype=np.float32),
            np.linspace(-1.0, 1.0, 16, dtype=np.float32),
            indexing="ij",
        )
        mr = np.clip(0.45 * yy + 0.35 * xx + zz + (idx - 1.5) * 0.05, -0.9, 0.9).astype(np.float32)
        ct = np.tanh(1.25 * mr - 0.15).astype(np.float32)
        mask = np.ones_like(mr, dtype=np.uint8)
        mask[:, 0, :] = 0
        mask[:, -1, :] = 0
        write_image(case_dir / "MR.mha", mr, sitk.sitkFloat32)
        write_image(case_dir / "CT.mha", ct, sitk.sitkFloat32)
        write_image(case_dir / "MASK.mha", mask, sitk.sitkUInt8)


def read_metric_mean(metrics_path: Path, suffix: str) -> float:
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    case_metrics = data.get("case", {})
    for key, values in case_metrics.items():
        if key.endswith(suffix):
            metric_values = list(values.values())
            if metric_values:
                return float(sum(metric_values) / len(metric_values))
    raise AssertionError(f"No metric ending with '{suffix}' found in {metrics_path}")


async def run_job(
    client: Any,
    tool_name: str,
    payload: dict[str, Any],
    *,
    timeout_s: float = 180.0,
) -> dict[str, Any]:
    job = await client.call_tool(tool_name, payload)
    job_payload = job.structured_content
    done = await client.call_tool(
        "wait_for_job",
        {"job_id": job_payload["job_id"], "timeout_s": timeout_s, "poll_interval_s": 0.2},
    )
    done_payload = done.structured_content
    if done_payload["status"] != "done":
        log = await client.read_resource(f"job://{job_payload['job_id']}/log")
        raise AssertionError(f"{tool_name} failed with status={done_payload['status']}\n{resource_to_text(log)}")
    return done_payload


async def wait_for_live_metric(
    client: Any,
    job_id: str,
    predicate: Callable[[dict[str, Any]], bool],
    *,
    kind: str = "train",
    timeout_s: float = 15.0,
    poll_interval_s: float = 0.2,
    max_entries: int = 10,
) -> dict[str, Any]:
    deadline = asyncio.get_running_loop().time() + timeout_s
    while asyncio.get_running_loop().time() < deadline:
        snapshot = await client.call_tool(
            "read_live_metrics",
            {
                "kind": kind,
                "job_id": job_id,
                "max_entries": max_entries,
            },
        )
        snapshot_data = snapshot.structured_content
        for stage_metrics in snapshot_data["by_stage"].values():
            if predicate(stage_metrics):
                return stage_metrics
        if snapshot_data["status"] not in {"queued", "running"}:
            log = await client.read_resource(f"job://{job_id}/log")
            raise AssertionError(
                f"Job {job_id} terminated with status={snapshot_data['status']} before exposing a live metric.\n"
                f"{resource_to_text(log)}"
            )
        await asyncio.sleep(poll_interval_s)

    log = await client.read_resource(f"job://{job_id}/log")
    raise AssertionError(f"Timed out while waiting for live metrics on job {job_id}.\n{resource_to_text(log)}")


def _fake_job_runtime(
    *,
    command: str,
    config: str,
    models: list[str] | None = None,
    model: str | None = None,
    lr: float | None = None,
    gpu: list[int] | None = None,
    cpu: int | None = None,
    overwrite: bool = False,
    quiet: bool = False,
    tensorboard: bool = False,
    single_process: bool = False,
    cwd: str | None = None,
    sleep_s: float = 0.3,
    steps: int = 3,
    emit_metrics: bool = True,
    exit_code: int = 0,
    metric_name: str = "MAE",
) -> None:
    data = yaml.load(Path(config).read_text(encoding="utf-8"))
    root_key = {"TRAIN": "Trainer", "RESUME": "Trainer", "PREDICTION": "Predictor", "EVALUATION": "Evaluator"}[command]
    root = data[root_key]
    run_name = root.get("train_name", "FAKE_RUN")
    workspace_dir = Path(cwd or Path.cwd())
    runtime_log = {
        "TRAIN": workspace_dir / "Statistics" / run_name / "log_0.txt",
        "RESUME": workspace_dir / "Statistics" / run_name / "log_0.txt",
        "PREDICTION": workspace_dir / "Predictions" / run_name / "log_0.txt",
        "EVALUATION": workspace_dir / "Evaluations" / run_name / "log_0.txt",
    }[command]
    runtime_log.parent.mkdir(parents=True, exist_ok=True)

    steps = max(steps, 1)

    for step in range(steps):
        if emit_metrics:
            stage = "Training"
            if command == "PREDICTION":
                stage = "Prediction"
            elif step == steps - 1:
                stage = "Validation"
            value = 1.0 / (step + 1)
            line = f"{stage}: Loss (Model(0.001000) : {metric_name}(1.00) : {value:.6f}) CPU (10.0 %)\n"
            with runtime_log.open("a", encoding="utf-8") as handle:
                handle.write(line)
            print(line, end="", flush=True)
        time.sleep(sleep_s / steps)

    if command in ("TRAIN", "RESUME"):
        checkpoint = workspace_dir / "Checkpoints" / run_name / "epoch_0001.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_text("checkpoint", encoding="utf-8")
    elif command == "PREDICTION":
        output = workspace_dir / "Predictions" / run_name / "Dataset" / "CASE_000" / "PRED.mha"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("prediction", encoding="utf-8")
    else:
        metrics = workspace_dir / "Evaluations" / run_name / "Metric_TRAIN.json"
        metrics.parent.mkdir(parents=True, exist_ok=True)
        metric_key = f"PRED:SEG:{metric_name}"
        payload = {
            "case": {metric_key: {"CASE_000": 0.5}},
            "aggregates": {
                metric_key: {
                    "mean": 0.5,
                    "min": 0.5,
                    "max": 0.5,
                    "std": 0.0,
                    "25pc": 0.5,
                    "50pc": 0.5,
                    "75pc": 0.5,
                    "count": 1.0,
                }
            },
        }
        metrics.write_text(json.dumps(payload), encoding="utf-8")

    if exit_code:
        print(f"simulated failure exit_code={exit_code}", flush=True)
        raise SystemExit(exit_code)


def install_fake_konfai_runtime(
    tmp_path: Path,
    monkeypatch: Any,
    mcp_server: ModuleType,
) -> None:
    def fake_runtime_job_spec(
        *,
        kind: str,
        config_path: Path,
        gpu=None,
        cpu=None,
        overwrite: bool = False,
        quiet: bool = False,
        single_process: bool = False,
        tensorboard: bool = False,
        models=None,
        resume_model=None,
        lr=None,
        cluster=None,
    ) -> dict[str, object]:
        runner_command = "RESUME" if resume_model is not None else kind.upper()
        command = ["konfai_mcp.fake", runner_command, "-c", str(config_path)]
        if resume_model is not None:
            command.extend(["--model", str(resume_model)])
        if lr is not None:
            command.extend(["--lr", str(lr)])
        if models:
            command.extend(["--models", *[str(path) for path in models]])
        if overwrite:
            command.append("--overwrite")
        if quiet:
            command.append("--quiet")
        if tensorboard:
            command.append("--tensorboard")
        if single_process:
            command.append("--single-process")
        if gpu:
            command.extend(["--gpu", *[str(device) for device in gpu]])
        if cpu is not None:
            command.extend(["--cpu", str(cpu)])
        return {
            "command": command,
            "target": "mcp_test_helpers:_fake_job_runtime",
            "kwargs": {
                "command": runner_command,
                "config": str(config_path),
                "models": [str(path) for path in models] if models else None,
                "model": str(resume_model) if resume_model is not None else None,
                "lr": lr,
                "gpu": gpu,
                "cpu": cpu,
                "overwrite": overwrite,
                "quiet": quiet,
                "tensorboard": tensorboard,
                "single_process": single_process,
                "sleep_s": float(os.environ.get("KONFAI_MCP_FAKE_SLEEP_S", "0.3")),
                "steps": int(os.environ.get("KONFAI_MCP_FAKE_STEPS", "3")),
                "emit_metrics": os.environ.get("KONFAI_MCP_FAKE_WRITE_METRICS", "1") != "0",
                "exit_code": int(os.environ.get("KONFAI_MCP_FAKE_EXIT_CODE", "0")),
                "metric_name": os.environ.get("KONFAI_MCP_FAKE_METRIC_NAME", "MAE"),
            },
        }

    monkeypatch.setattr(mcp_server, "_runtime_job_spec", fake_runtime_job_spec)


def spawn_grandchild_and_idle(pid_file: str) -> None:
    """Test job target: spawn a grandchild process (standing in for a DDP mp.spawn worker) and idle,
    recording both PIDs. Used to verify cancel_job reaps the whole process group, not just the middle
    process. The grandchild is a plain `python -c` subprocess (not a nested multiprocessing spawn), so it
    starts fast and is immune to re-importing the pytest __main__ under a loaded full-suite run.
    """
    import os
    import subprocess
    import sys
    import time

    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    with open(pid_file, "w", encoding="utf-8") as handle:
        handle.write(f"{os.getpid()} {child.pid}\n")
    time.sleep(120)
