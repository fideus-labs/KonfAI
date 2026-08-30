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


"""The process's environment: workspace state, machine facts, devices, overwrite prompts."""

import atexit
import builtins
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, TypedDict

import psutil
import torch

try:
    import pynvml

    _PYNVML_AVAILABLE = True
except ImportError:
    _PYNVML_AVAILABLE = False
from konfai import (
    cuda_visible_devices,
)
from konfai.utils import State
from konfai.utils.errors import ConfigError


class ClusterKwargs(TypedDict):
    name: str
    memory: int
    num_nodes: int
    time_limit: int
    resubmit: bool


def description(model, model_ema=None, show_memory: bool = True, train: bool = True) -> str:
    """Return a compact human-readable runtime summary for progress bars."""

    def loss_desc(model):
        return (
            "("
            + " ".join(
                [
                    f"{name}({(network.optimizer.param_groups[0]['lr'] if network.optimizer else 0):.6f}) : "
                    + " ".join(
                        f"{k.split(':')[-1]}({w:.2f}) : {v:.6f}"
                        for (k, v), w in zip(
                            network.measure.get_last_values().items(),
                            network.measure.get_last_weights().values(),
                            strict=False,
                        )
                    )
                    for name, network in model.module.get_networks().items()
                    if network.measure is not None
                ]
            )
            + ")"
        )

    model_loss_desc = loss_desc(model)
    result = ""
    if len(model_loss_desc) > 2:
        result += f"Loss {model_loss_desc} "
    if model_ema is not None:
        model_ema_loss_desc = loss_desc(model_ema)
        if len(model_ema_loss_desc) > 2:
            result += f"Loss EMA {model_ema_loss_desc} "
    gpu_str = gpu_info()
    result += gpu_str
    if gpu_str:
        result += " | "
    if show_memory:
        result += get_memory_info()
    return result


def get_cpu_info() -> str:
    """Return current CPU utilization as a short status string."""
    # interval=None is non-blocking (utilization since the previous call). The blocking interval=0.5
    # form would stall the caching progress-bar refresh by half a second each call, on the data-load
    # critical path, purely to render a telemetry label.
    return f"CPU ({psutil.cpu_percent(interval=None):.2f} %)"


def get_memory_info() -> str:
    """Return current RAM usage as a short status string."""
    return f"Memory ({psutil.virtual_memory().used / 2**30:.2f}G ({psutil.virtual_memory().percent:.2f} %))"


def get_memory() -> float:
    """Return current RAM usage in GiB."""
    return psutil.virtual_memory().used / 2**30


def _materialized_config(tree: dict, root: str) -> Path:
    """A config TREE written where a workflow expects a file: the Python front door.

    The caller hands the same tree the YAML file would hold (``{root: {...}}``, the very kwargs
    the binder feeds each ``__init__``), and never touches YAML: it is written once here, under a
    scratch directory of its own, and everything downstream (reflection binding, resolution
    write-back, the workspace copy, resume) sees an ordinary config file. The workspace keeps the
    resolved copy, as it does for every run.
    """
    if list(tree) != [root]:
        raise ConfigError(
            f"A config tree for this workflow must hold exactly the '{root}' root"
            f" (found: {sorted(str(key) for key in tree)}).",
            f"Pass the same tree the YAML file would hold: {{'{root}': {{...}}}}.",
        )
    from ruamel.yaml import YAML

    scratch = Path(tempfile.mkdtemp(prefix=f"konfai_{root.lower()}_"))
    # The file must outlive the run (spawned ranks re-read it), not the process.
    atexit.register(shutil.rmtree, scratch, ignore_errors=True)
    path = scratch / f"{root}.yml"
    with path.open("w", encoding="utf-8") as file:
        YAML().dump(tree, file)
    return path


def configure_workflow_environment(
    *,
    config_path: Path | str | dict,
    root: str,
    state: "State | str",
    path_env: dict[str, Path | str] | None = None,
) -> None:
    """
    Populate the process-wide environment expected by KonfAI workflows.

    Parameters
    ----------
    config_path : Path | str | dict
        YAML configuration file used by the workflow: or the config TREE itself, as a dict, for
        a Python caller that writes no YAML (see :func:`_materialized_config`). Every workflow
        entry point accepts either, since they all pass through here.
    root : str
        Root configuration section, for example ``Trainer`` or ``Predictor``.
    state : State | str
        Runtime state identifier exposed through ``KONFAI_STATE``.
    path_env : dict[str, Path | str] | None, optional
        Additional environment variables whose values should be normalized as
        absolute filesystem paths before export.
    """
    if isinstance(config_path, dict):
        config_path = _materialized_config(config_path, root)
    os.environ["KONFAI_config_file"] = str(Path(config_path).resolve())
    os.environ["KONFAI_ROOT"] = root
    os.environ["KONFAI_STATE"] = str(state)
    for env_name, env_path in (path_env or {}).items():
        os.environ[env_name] = str(Path(env_path).resolve())


def memory_forecast(memory_init: float, i: int, size: int) -> str:
    """Estimate final memory consumption while iterating over a dataset."""
    current_memory = get_memory()
    forecast = memory_init + ((current_memory - memory_init) * size / i) if i > 0 else memory_init
    return f"Memory forecast ({forecast:.2f}G ({forecast / (psutil.virtual_memory().total / 2**30) * 100:.2f} %))"


def gpu_info() -> str:
    """Return a compact status line describing visible GPU usage."""
    if not _PYNVML_AVAILABLE or len(cuda_visible_devices()) == 0:
        return ""

    devices = [int(i) for i in cuda_visible_devices()]
    device = devices[0]

    if device < pynvml.nvmlDeviceGetCount():
        handle = pynvml.nvmlDeviceGetHandleByIndex(device)
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
    else:
        return ""
    node_name = f"Node: {os.environ['SLURMD_NODENAME']} " if "SLURMD_NODENAME" in os.environ else ""
    return f"{node_name}GPU({devices}) Memory GPU ({memory.used / 1e9:.2f}G ({memory.used / memory.total * 100:.2f} %))"


def get_gpu_memory(device: int | torch.device) -> float:
    """Return current VRAM usage in GB for one device, or ``0`` on CPU."""
    if not _PYNVML_AVAILABLE:
        return 0
    if isinstance(device, torch.device):
        if str(device).startswith("cuda:"):
            device = int(str(device).replace("cuda:", ""))
        else:
            return 0
    device = cuda_visible_devices()[device]
    if device < pynvml.nvmlDeviceGetCount():
        handle = pynvml.nvmlDeviceGetHandleByIndex(device)
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
    else:
        return 0
    return float(memory.used) / (10**9)


class NeedDevice:
    """Mixin for objects that expose a torch device selected at runtime."""

    def __init__(self) -> None:
        super().__init__()
        # Default to CPU so ``self.device`` is always set: an object that is never explicitly moved (e.g. an
        # output dataset off the propagation path) then reads as CPU instead of raising AttributeError.
        self.device: torch.device = torch.device("cpu")

    def to(self, device: int):
        self.device = get_device(device)


def get_device(device: int):
    """Return a CUDA index or CPU device depending on availability.

    ``device_count`` and not availability alone: a latched runtime keeps ``is_available()`` True
    after ``CUDA_VISIBLE_DEVICES`` was narrowed to nothing, while the count honestly reads 0.
    """
    return device if torch.cuda.is_available() and 0 <= device < torch.cuda.device_count() else torch.device("cpu")


def safe_torch_load(path_or_url: str | Path, map_location: Any) -> Any:
    """
    Load a checkpoint from a local path or an ``https://`` URL, preferring the
    safe ``weights_only=True`` deserializer.

    For a local (trusted) checkpoint, fall back to ``weights_only=False`` only
    when the safe unpickler refuses to reconstruct stored objects. A remote
    ``https://`` checkpoint is untrusted and is loaded with ``weights_only=True``
    only: a payload crafted to fail the safe load must not trigger the
    arbitrary-code unpickler.
    """
    source = str(path_or_url)
    if source.startswith("https://"):
        return torch.hub.load_state_dict_from_url(source, map_location=map_location, weights_only=True)
    try:
        return torch.load(source, map_location=map_location, weights_only=True)
    except Exception:
        return torch.load(source, map_location=map_location, weights_only=False)  # nosec B614


def is_interactive_session() -> bool:
    """Return whether KonfAI can safely prompt on stdin/stdout."""
    stdin = getattr(sys, "stdin", None)
    stdout = getattr(sys, "stdout", None)
    # ``stdout`` may be a Log/MinimalLog proxy (write/flush/fileno only); guard its ``isatty``
    # exactly like ``stdin`` so a redirected stream degrades to non-interactive instead of raising.
    return bool(
        stdin
        and stdout
        and hasattr(stdin, "isatty")
        and hasattr(stdout, "isatty")
        and stdin.isatty()
        and stdout.isatty()
    )


def confirm_overwrite_or_raise(path: Path, label: str, error_cls: type[Exception]) -> None:
    """
    Ensure an existing output can be overwritten.

    Parameters
    ----------
    path : Path
        Existing path that would be replaced.
    label : str
        Human-readable artifact label used in the prompt and error message.
    error_cls : type[Exception]
        Exception type raised when overwrite is not allowed or declined.

    Raises
    ------
    Exception
        Instance of ``error_cls`` when overwrite is disabled in a
        non-interactive session or explicitly declined by the user.
    """
    if os.environ.get("KONFAI_OVERWRITE") == "True":
        return

    message = f"The {label} '{path}' already exists."
    guidance = "Pass -y/--overwrite to replace it, or remove the existing outputs manually."
    if not is_interactive_session():
        raise error_cls(message, guidance)

    accept = builtins.input(f"{message} Do you want to overwrite it (yes,no) : ").strip().lower()
    if accept != "yes":
        raise error_cls(message, "Overwrite was declined.", guidance)


def clear_directory_except_logs(path: Path) -> None:
    """Remove a run directory's contents but keep its ``log_*.txt`` files.

    The rank-0 ``Log`` opens ``<dir>/log_0.txt`` before the workflow's overwrite branch runs, so an
    ``rmtree`` of the directory unlinks the open file: every parent-process line (config binding,
    dataset scan, a crash traceback) is written to an unlinked inode and lost, and Windows refuses
    to delete a directory holding an open file. Clearing around the live logs preserves them.
    """
    for child in path.iterdir():
        # Preserve only a regular log file: a directory or symlink merely named log_*.txt is not a live log.
        if child.is_file() and not child.is_symlink() and child.name.startswith("log_") and child.suffix == ".txt":
            continue
        # is_dir() follows symlinks, so unlink a symlink (even one to a directory) instead of rmtree-ing
        # through it into the target's contents.
        if child.is_symlink() or not child.is_dir():
            child.unlink()
        else:
            shutil.rmtree(child)
