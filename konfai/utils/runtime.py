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

"""Runtime state, logging, and distributed execution helpers for KonfAI."""

import atexit
import builtins
import inspect
import os
import random
import re
import shutil
import socket
import subprocess  # nosec B404
import sys
import tempfile
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import closing, suppress
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Any, TextIO, TypedDict, cast

import numpy as np
import psutil
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

try:
    import pynvml

    _PYNVML_AVAILABLE = True
except ImportError:
    _PYNVML_AVAILABLE = False

try:
    from torch.utils.tensorboard.writer import SummaryWriter
except ImportError:
    SummaryWriter = None  # type: ignore[assignment,misc]

from konfai import (
    cuda_visible_devices,
    evaluations_directory,
    konfai_state,
    predictions_directory,
    statistics_directory,
    transforms_directory,
)
from konfai.utils import State  # also the re-export ``konfai.utils.runtime.State`` callers import
from konfai.utils.errors import ConfigError, KonfAIError
from konfai.utils.utils import env_flag


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


# psutil.virtual_memory() always reports the HOST, so inside a container or a SLURM cgroup that grants
# far less than the node has, a memory budget derived from it would overshoot the real limit and get
# OOM-killed. The cgroup ceiling is read directly instead: the process's own cgroup (from
# /proc/self/cgroup, since only ``docker run`` puts it at the mount root) and every ancestor up to the
# mount, the tightest one winning. cgroup v2 exposes ``memory.max`` (the literal ``"max"`` means
# unbounded) and ``memory.current``; cgroup v1 exposes ``memory.limit_in_bytes`` with a page-aligned
# near INT64_MAX sentinel and ``memory.usage_in_bytes``. Only Linux has these files.
_CGROUP_ROOT = "/sys/fs/cgroup"
_PROC_SELF_CGROUP = "/proc/self/cgroup"
_CGROUP_UNLIMITED = 1 << 62  # any v1 limit at or above this is the "no limit" sentinel


def _cgroup_paths() -> list[tuple[Path, str, str]]:
    """The (directory, limit file, usage file) of this process's memory cgroup and its ancestors,
    innermost first: v2 (``0::/path``) or v1 (``N:memory:/path``); empty when there is no cgroup."""
    try:
        lines = Path(_PROC_SELF_CGROUP).read_text().splitlines()
    except OSError:
        return []
    root = Path(_CGROUP_ROOT)
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        hierarchy, controllers, path = parts
        if hierarchy == "0" and controllers == "":
            base, files = root, ("memory.max", "memory.current")
        elif "memory" in controllers.split(","):
            base, files = root / "memory", ("memory.limit_in_bytes", "memory.usage_in_bytes")
        else:
            continue
        leaf = base / path.lstrip("/")
        chain = [leaf, *leaf.parents]
        return [(directory, *files) for directory in chain if directory == base or base in directory.parents]
    return []


def _read_cgroup_memory_limit() -> tuple[int, int] | None:
    """``(ceiling, current usage)`` in bytes for this process's cgroup, or ``None`` when unbounded or absent.

    The ceiling is the tightest ``memory.max``/``limit_in_bytes`` on the path from the process's own
    cgroup to the mount root; the usage is the innermost cgroup's, so a SLURM step already holding
    memory does not count it as free. A missing hierarchy (non-Linux, cgroups disabled), the ``"max"``
    keyword, the v1 sentinel and unparseable values all mean "no bound at this level".
    """
    limits: list[int] = []
    usage: int | None = None
    for directory, limit_name, usage_name in _cgroup_paths():
        try:
            raw = (directory / limit_name).read_text().strip()
        except OSError:
            continue
        if raw != "max":
            with suppress(ValueError):
                limit = int(raw)
                if limit < _CGROUP_UNLIMITED:
                    limits.append(limit)
        if usage is None:
            with suppress(OSError, ValueError):
                usage = int((directory / usage_name).read_text().strip())
    if not limits:
        return None
    return min(limits), usage or 0


def _slurm_memory_grant() -> int | None:
    """The bytes SLURM granted this step (``--mem`` / ``--mem-per-cpu``, in MB), or ``None`` outside a job.
    Read as well as the cgroup: on a cluster whose slurmd does not enforce cgroups, the env is the bound."""
    per_node = os.environ.get("SLURM_MEM_PER_NODE")
    per_cpu = os.environ.get("SLURM_MEM_PER_CPU")
    cpus = os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get("SLURM_CPUS_ON_NODE")
    try:
        if per_node:
            return int(per_node) * 2**20
        if per_cpu and cpus:
            return int(per_cpu) * int(cpus) * 2**20
    except ValueError:
        return None
    return None


def available_memory_bytes() -> tuple[int, str]:
    """Return ``(bytes a process may safely allocate, source label)``, honouring a cgroup limit.

    ``psutil`` sees the host's free RAM, which overshoots a container/cgroup ceiling; the tightest of
    the cgroup's remaining room (its ceiling minus what it already holds), a SLURM memory grant and
    the host figure wins, so the number is safe on a bare host, in a memory-capped container and in a
    SLURM step alike. The label names which bound won, for the startup decision log.
    """
    candidates = [(int(psutil.virtual_memory().available), "host available RAM")]
    cgroup = _read_cgroup_memory_limit()
    if cgroup is not None:
        limit, usage = cgroup
        candidates.append((max(0, limit - usage), "cgroup limit"))
    slurm = _slurm_memory_grant()
    if slurm is not None:
        candidates.append((slurm, "SLURM memory grant"))
    return min(candidates, key=lambda candidate: candidate[0])


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


def get_max_gpu_memory(device: int | torch.device) -> float:
    """Return the total VRAM in GB for one device, or ``0`` on CPU."""
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
    return float(memory.total) / (10**9)


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
    """Return a CUDA index or CPU device depending on availability."""
    return device if torch.cuda.is_available() and device >= 0 else torch.device("cpu")


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


def _log_signal_format(array: np.ndarray) -> dict[str, np.ndarray]:
    return {str(i): channel for i, channel in enumerate(array)}


def _log_image_format(array: np.ndarray) -> np.ndarray:
    if len(array.shape) == 2:
        array = np.expand_dims(array, axis=0)

    if len(array.shape) == 3 and array.shape[0] != 1:
        array = np.expand_dims(array, axis=0)
    if len(array.shape) == 4:
        array = array[:, array.shape[1] // 2]

    array = array.astype(float)
    b = -np.min(array)
    if (np.max(array) + b) > 0:
        return (array + b) / (np.max(array) + b)
    else:
        return 0 * array


def _log_images_format(array: np.ndarray) -> np.ndarray:
    result = []
    for n in range(array.shape[0]):
        result.append(_log_image_format(array[n]))
    result = np.stack(result, axis=0)
    return result


def _log_video_format(array: np.ndarray) -> np.ndarray:
    result_list = []
    for t in range(array.shape[1]):
        result_list.append(_log_images_format(array[:, t, ...]))
    result = np.stack(result_list, axis=1)

    nb_channel = result.shape[2]
    if nb_channel < 3:
        channel_split = [result[:, :, 0, ...] for i in range(3)]
    else:
        channel_split = np.split(result, 3, axis=0)
    array = np.zeros((result.shape[0], result.shape[1], 3, *list(result.shape[3:])))
    for i, channels in enumerate(channel_split):
        array[:, :, i] = np.mean(channels, axis=0)
    return array


class DataLog(Enum):
    """TensorBoard logging strategy selected in YAML runtime configs."""

    SIGNAL = "SIGNAL"
    IMAGE = "IMAGE"
    IMAGES = "IMAGES"
    VIDEO = "VIDEO"
    AUDIO = "AUDIO"

    def __call__(self, tb: SummaryWriter, name: str, layer: torch.Tensor, it: int):
        if self == DataLog.SIGNAL:
            return [
                tb.add_scalars(name, _log_signal_format(layer[b, :, 0]), layer.shape[0] * it + b)
                for b in range(layer.shape[0])
            ]
        elif self == DataLog.IMAGE:
            return tb.add_image(name, _log_image_format(layer[0]), it)
        elif self == DataLog.IMAGES:
            return tb.add_images(name, _log_images_format(layer), it)
        elif self == DataLog.VIDEO:
            return tb.add_video(name, _log_video_format(layer), it)
        elif self == DataLog.AUDIO:
            return tb.add_audio(name, _log_image_format(layer), it)
        else:
            raise ValueError(f"Unsupported DataLog type: {self}")


def _bar_key(line: str) -> str:
    """What identifies a progress bar across its redraws: its text up to the first digit."""
    return re.split(r"\d", line, maxsplit=1)[0]


ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


class MinimalLog:
    """Capture stdout/stderr while keeping a one-line rolling status buffer."""

    # Off a terminal, at most one redrawn progress-bar frame is mirrored per this many seconds.
    _MIRROR_REDRAW_EVERY = 2.0

    def __init__(self, rank: int = 0) -> None:
        self._stdout_bak = sys.stdout
        self._stderr_bak = sys.stderr
        self._buffered_line = ""
        self.verbose = os.environ.get("KONFAI_VERBOSE", "True") == "True"
        self.rank = rank
        try:
            self._mirror_is_tty = bool(self._stdout_bak.isatty())
        except (AttributeError, ValueError, OSError):
            self._mirror_is_tty = False
        self._mirror_last_redraw = 0.0
        # Folded frames withheld by the throttle, one slot per bar (keyed by the text before the first
        # digit): interleaved bars (train + validation) would overwrite a single slot, and one of the
        # two would end the run without its final state ever mirrored.
        self._mirror_pending: dict[str, str] = {}

    def __enter__(self):
        sys.stdout = cast(TextIO, self)
        sys.stderr = cast(TextIO, self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # A run's last writes are often throttled frames: emitted here, so the sink ends on the bar's
        # final state. flush() cannot carry this: the bars flush after every frame, which would empty
        # the throttle each time.
        self._mirror_emit_pending()
        sys.stdout = self._stdout_bak
        sys.stderr = self._stderr_bak

    def write(self, msg: str):
        if not msg:
            return
        msg_clean = ANSI_ESCAPE_RE.sub("", msg)
        # A CRLF line ending is not a redraw: "warning\r\n" folds to the text after its last \r,
        # which is nothing, and the message would vanish from the mirror and the log file both.
        redraw = "\r" in msg_clean.replace("\r\n", "\n") or "[A" in msg
        if redraw:
            self._buffered_line = msg_clean.split("\r")[-1].strip()
        else:
            self._buffered_line = msg_clean.strip()

        if self.verbose and (self.rank == 0 or "KONFAI_CLUSTER" in os.environ):
            self._mirror(msg, redraw)

    def _mirror(self, msg: str, redraw: bool) -> None:
        # A terminal overwrites a redrawn bar; a file appends every frame, so a mirrored animation is
        # megabytes per run in an MCP job log or a slurm-*.out. Off a terminal the mirror sends the
        # folded line instead, throttled (a log tail still shows live progress), and a skipped frame
        # is kept pending so a bar's final state lands before whatever message follows it.
        if not self._mirror_is_tty:
            if redraw:
                now = time.monotonic()
                if now - self._mirror_last_redraw < self._MIRROR_REDRAW_EVERY:
                    self._mirror_pending[_bar_key(self._buffered_line)] = self._buffered_line
                    return
                self._mirror_last_redraw = now
                held = self._mirror_take_pending(exclude=_bar_key(self._buffered_line))
                msg = f"{held}{self._buffered_line}\n"
            else:
                msg = f"{self._mirror_take_pending()}{msg}"
        # Best-effort: if the mirror's reader is gone (an interactive launcher exited, a server
        # restarted), the pipe is broken: keep running and keep writing to the log file rather than
        # crashing the job.
        try:
            self._stdout_bak.write(msg)
            self._stdout_bak.flush()
        except (BrokenPipeError, ValueError):
            pass

    def flush(self):
        try:
            self._stdout_bak.flush()
        except (BrokenPipeError, ValueError):
            pass

    def _mirror_take_pending(self, exclude: str | None = None) -> str:
        """The withheld frames as mirror-ready lines, cleared; ``exclude`` skips the bar being emitted."""
        if not self._mirror_pending:
            return ""
        lines = [line for key, line in self._mirror_pending.items() if key != exclude]
        self._mirror_pending.clear()
        return "".join(f"{line}\n" for line in lines)

    def _mirror_emit_pending(self) -> None:
        held = self._mirror_take_pending()
        if held:
            try:
                self._stdout_bak.write(held)
                self._stdout_bak.flush()
            except (BrokenPipeError, ValueError):
                pass

    def fileno(self):
        if sys.__stdout__ is None:
            raise RuntimeError("sys.__stdout__ is None, cannot get fileno")
        return sys.__stdout__.fileno()


class Log(MinimalLog):
    """Mirror console output to a rank-specific log file."""

    def __init__(self, name: str, rank: int) -> None:
        super().__init__(rank)
        if konfai_state() == "PREDICTION":
            path = predictions_directory()
        elif konfai_state() == "EVALUATION":
            path = evaluations_directory()
        elif konfai_state() == "TRANSFORM":
            path = transforms_directory()
        else:
            path = statistics_directory()
        # ``name`` is train_name from the config; an absolute path or '..' segments would place the logs
        # outside the run's output directory. A nested run name stays inside and is fine.
        self.log_path = path / name
        if not self.log_path.resolve().is_relative_to(path.resolve()):
            raise ConfigError(
                f"train_name '{name}' resolves outside the output directory.",
                "Use a plain run name, without an absolute path or '..' segments.",
            )
        self.log_path.mkdir(parents=True, exist_ok=True)
        # Append, never truncate: this file is opened BEFORE the overwrite prompt runs, so a "w" mode
        # destroyed the previous run's log even when the user declined the overwrite.
        self.file = open(self.log_path / f"log_{rank}.txt", "a", buffering=1)
        self._last_logged: str | None = None

    def __enter__(self):
        super().__enter__()
        self.file.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        super().__exit__(exc_type, exc_val, exc_tb)
        self.file.__exit__(exc_type, exc_val, exc_tb)

    def write(self, msg: str):
        super().write(msg)
        # Consecutive identical lines are one fact said twice: a progress bar arrives as several
        # write() calls per frame and a case line rides beside its own counter frame, which would
        # multiply the file by ~4x against the console. Only CONSECUTIVE repeats fold: a fact that
        # genuinely recurs later still lands.
        if self._buffered_line and self._buffered_line != self._last_logged:
            self._last_logged = self._buffered_line
            self.file.write(self._buffered_line + "\n")
            self.file.flush()

    def flush(self):
        super().flush()
        self.file.flush()


def record(message: str) -> Path | None:
    """Keep ``message`` in the run's log without printing it, and answer where it went.

    For detail that belongs beside the run but not on a console every other workflow keeps to a
    progress bar (the TRANSFORM plan). Answers None when no ``Log`` is installed, which is every
    context that has no run directory to keep it in.
    """
    sink = sys.stdout
    if not isinstance(sink, Log):
        return None
    sink.file.write(message if message.endswith("\n") else message + "\n")
    sink.file.flush()
    return Path(sink.file.name)


class TensorBoard:
    """Lifecycle helper that optionally starts a TensorBoard side process."""

    def __init__(self, name: str) -> None:
        self.process: subprocess.Popen | None = None
        self.name = name

    def __enter__(self):
        if "KONFAI_TENSORBOARD_PORT" in os.environ:
            tensorboard_exe = shutil.which("tensorboard")
            if tensorboard_exe is None:
                raise RuntimeError("TensorBoard executable not found in PATH.")

            logdir = predictions_directory() if konfai_state() == "PREDICTION" else statistics_directory() / self.name

            port = os.environ.get("KONFAI_TENSORBOARD_PORT")
            if not port or not port.isdigit():
                raise ValueError("Invalid or missing KONFAI_TENSORBOARD_PORT.")

            command = [
                tensorboard_exe,
                "--logdir",
                str(logdir),
                "--port",
                port,
                "--bind_all",
            ]
            self.process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # nosec B603
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("10.255.255.255", 1))
                ip = s.getsockname()[0]
            except Exception:
                ip = "127.0.0.1"
            finally:
                s.close()
            print(f"[KonfAI] Tensorboard : http://{ip}:{os.environ['KONFAI_TENSORBOARD_PORT']}/")
        return self

    def __exit__(self, exc_type, value, traceback):
        if self.process is not None:
            self.process.terminate()
            self.process.wait()


class DistributedObject(ABC):
    """Base class for trainer, predictor, and evaluator distributed workflows."""

    #: Whether the ranks talk to each other (DDP, gathers). A workflow whose ranks only share the
    #: work list sets it False and runs without a process group: no rendezvous port, no gloo/NCCL.
    uses_collectives: bool = True

    def __init__(self, name: str) -> None:
        self.dataloader: list[list[DataLoader]]
        self.manual_seed: int | None = None
        self.name = name
        self.size = 1

    @abstractmethod
    def setup(self, world_size: int):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, value, traceback):
        cleanup()

    @abstractmethod
    def run_process(
        self,
        world_size: int,
        global_rank: int,
        local_rank: int,
        dataloaders: list[DataLoader],
    ):
        pass

    @staticmethod
    def get_measure(
        world_size: int,
        global_rank: int,
        gpu: int,
        models: dict[str, torch.nn.Module],
        n: int,
        sync: bool = True,
    ) -> dict[str, tuple[dict[str, tuple[float, float]], dict[str, tuple[float, float]]]]:
        data = {}
        for label, model in models.items():
            for name, network in model.get_networks().items():
                if network.measure is not None:
                    data[f"{name}{label}"] = (
                        network.measure.format_loss(True, n),
                        network.measure.format_loss(False, n),
                    )
        # `sync=False` skips the cross-rank all_gather: prediction shards whole cases per rank with unequal
        # batch counts, so a per-batch collective would hang once the shortest shard stops calling it.
        outputs = synchronize_data(world_size, gpu, data) if sync else [data]
        result: dict[str, tuple[dict[str, tuple[float, float]], dict[str, tuple[float, float]]]] = {}
        if global_rank == 0:
            for output in outputs:
                for k, v in output.items():
                    for t in range(len(v)):
                        for u, n in v[t].items():
                            if k not in result:
                                result[k] = ({}, {})
                            if u not in result[k][t]:
                                result[k][t][u] = (n[0], 0)  # type: ignore[index]
                            result[k][t][u] = (
                                result[k][t][u][0],
                                result[k][t][u][1] + n[1] / world_size,  # type: ignore[index]
                            )
        return result

    @property
    def world_size(self) -> int:
        """How many ranks ``setup`` sharded the run over: one dataloader list per rank."""
        return len(self.dataloader)

    def rank_dataloaders(self, global_rank: int) -> "list[DataLoader]":
        return self.dataloader[global_rank]

    def __call__(self, rank: int | None = None) -> None:
        world_size = self.world_size
        global_rank, local_rank = setup_gpu(world_size, rank, process_group=self.uses_collectives)
        if global_rank is None or local_rank is None:
            return
        apply_cpu_thread_budget()
        with Log(self.name, global_rank):
            if torch.cuda.is_available() and _PYNVML_AVAILABLE:
                pynvml.nvmlInit()
            if self.manual_seed is not None:
                np.random.seed(self.manual_seed * world_size + global_rank)
                random.seed(self.manual_seed * world_size + global_rank)
                torch.manual_seed(self.manual_seed * world_size + global_rank)
            torch.backends.cudnn.benchmark = self.manual_seed is None
            torch.backends.cudnn.deterministic = self.manual_seed is not None
            dataloaders = self.rank_dataloaders(global_rank)
            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank)
            try:
                self.run_process(world_size, global_rank, local_rank, dataloaders)
            finally:
                cleanup()
                if torch.cuda.is_available() and _PYNVML_AVAILABLE:
                    pynvml.nvmlShutdown()


def run_distributed_app(
    func: Callable[..., DistributedObject],
) -> Callable[..., None]:
    """Wrap a workflow factory so it executes with KonfAI runtime conventions."""

    sig = inspect.signature(func)

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> None:
        params = sig.parameters
        kwargs_fun = {k: v for k, v in kwargs.items() if k in params}

        bound = sig.bind_partial(*args, **kwargs_fun)
        bound.apply_defaults()
        is_cluster = "resubmit" in kwargs
        # The auto memory budget is a NODE budget, but build-time sizing (the evaluation auto-patch)
        # runs while ``func(...)`` constructs the workflow: before the spawn where world_size exists.
        # The launcher therefore leaves the per-node rank count in the environment, and restores it
        # after: a leak would silently shrink a later in-process run (tests, embedded Python).
        local_ranks = len(list(bound.arguments.get("gpu") or [])) or int(bound.arguments.get("cpu") or 1)
        previous_local_ranks = os.environ.get("KONFAI_LOCAL_RANKS")
        os.environ["KONFAI_LOCAL_RANKS"] = str(max(1, local_ranks))
        try:
            execute_distributed_object(
                func(*args, **kwargs_fun),
                gpu=bound.arguments.get("gpu", []),
                cpu=bound.arguments.get("cpu", 1),
                overwrite=bool(bound.arguments.get("overwrite", False)),
                quiet=bool(bound.arguments.get("quiet", False)),
                tensorboard=bool(bound.arguments.get("tensorboard", False)),
                cluster_kwargs=(
                    {
                        "name": kwargs["name"],
                        "memory": kwargs["memory"],
                        "num_nodes": kwargs["num_nodes"],
                        "time_limit": kwargs["time_limit"],
                        "resubmit": bool(kwargs.get("resubmit", False)),
                    }
                    if is_cluster
                    else None
                ),
            )
        except KeyboardInterrupt:
            print("\n[KonfAI] Manual interruption (Ctrl+C)")
        except KonfAIError as error:
            # A designed refusal: the message says what is wrong and the remedy what to change.
            # The traceback under it is framework internals: 28 lines burying the 3 that matter --
            # so it is shown only to a reader who asked (KONFAI_DEBUG=1).
            if env_flag("KONFAI_DEBUG", False):
                raise
            print(str(error).strip(), file=sys.stderr)
            sys.exit(1)
        finally:
            if previous_local_ranks is None:
                os.environ.pop("KONFAI_LOCAL_RANKS", None)
            else:
                os.environ["KONFAI_LOCAL_RANKS"] = previous_local_ranks

    return wrapper


def _runs_inline(world_size: int) -> bool:
    """Whether the single rank runs here instead of in a spawned child.

    A spawned child is a fresh interpreter: it re-imports torch, re-initialises CUDA and unpickles the
    whole payload before doing any work: measured at ~3 s, which a short prediction pays in full. With
    one rank there is nothing to parallelise, so that cost buys only isolation.

    Isolation is worth keeping where the caller outlives the run: an embedded interpreter (Slicer, the
    apps server) would inherit this process's CUDA context and its memory. ``KONFAI_INLINE_SINGLE_RANK``
    is the switch: default on for a CLI run, set it to 0 to force the child back.
    """
    if world_size != 1:
        return False
    return env_flag("KONFAI_INLINE_SINGLE_RANK", True)


def execute_distributed_object(
    distributed_object: DistributedObject,
    *,
    gpu: list[int] | None = None,
    cpu: int | None = 1,
    overwrite: bool = False,
    quiet: bool = False,
    tensorboard: bool = False,
    cluster_kwargs: ClusterKwargs | None = None,
) -> None:
    """
    Execute a previously built KonfAI workflow object.

    Parameters
    ----------
    distributed_object : DistributedObject
        Configured workflow returned by a build step.
    gpu : list[int] | None, optional
        GPU ids exposed to the workflow.
    cpu : int | None, optional
        Number of CPU workers when running without GPUs.
    overwrite : bool, optional
        Whether existing outputs may be overwritten.
    quiet : bool, optional
        Whether console output should be reduced.
    tensorboard : bool, optional
        Whether TensorBoard should be started for the workflow.
    cluster_kwargs : dict[str, Any] | None, optional
        Optional cluster submission parameters used by ``submitit``.
    """
    gpu_ids = [] if gpu is None else list(gpu)
    cpu_workers = 1 if cpu is None else int(cpu)
    if cpu_workers < 1:
        raise ConfigError(f"cpu={cpu!r} is not a rank count.", "Pass cpu=1 or more (the CLI refuses it the same way).")

    managed_env = [
        "CUDA_VISIBLE_DEVICES",
        "KONFAI_OVERWRITE",
        "KONFAI_CONFIG_MODE",
        "KONFAI_TENSORBOARD_PORT",
        "KONFAI_MASTER_PORT",
        "KONFAI_VERBOSE",
        "KONFAI_CLUSTER",
    ]
    previous_env = {key: os.environ.get(key) for key in managed_env}
    # The run seeds the process-wide RNGs and sets the cudnn flags; inline (the single-rank default)
    # that process is the caller's (a notebook, Slicer), so what it found is put back.
    previous_rng = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    previous_cudnn = (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)

    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in gpu_ids if i >= 0])
        os.environ["KONFAI_OVERWRITE"] = str(overwrite)
        os.environ["KONFAI_CONFIG_MODE"] = "Done"
        if tensorboard:
            os.environ["KONFAI_TENSORBOARD_PORT"] = str(find_free_port())
        if "KONFAI_MASTER_PORT" not in os.environ:
            os.environ["KONFAI_MASTER_PORT"] = str(find_free_port())
        os.environ["KONFAI_VERBOSE"] = str(not quiet)

        cluster_config = cluster_kwargs
        if cluster_config is not None:
            os.environ["KONFAI_OVERWRITE"] = "True"
            os.environ["KONFAI_CLUSTER"] = "True"

        with distributed_object as configured_object:
            with Log(configured_object.name, 0):
                if configured_object.manual_seed is not None:
                    np.random.seed(configured_object.manual_seed)
                    random.seed(configured_object.manual_seed)
                    torch.manual_seed(configured_object.manual_seed)
                if cluster_config is not None:
                    if cluster_config["resubmit"]:
                        # Auto-requeue is not implemented; warn instead of silently dropping the flag.
                        print(
                            "[KonfAI] WARNING: --resubmit is not implemented yet; this job will NOT "
                            "auto-requeue at the time limit. Relaunch manually with the RESUME command "
                            "pointing at the latest checkpoint to continue training."
                        )
                    configured_object.setup(len(gpu_ids) * cluster_config["num_nodes"])
                    import submitit

                    executor = submitit.AutoExecutor(folder="./Cluster/")
                    executor.update_parameters(
                        name=cluster_config["name"],
                        mem_gb=cluster_config["memory"],
                        gpus_per_node=len(gpu_ids),
                        tasks_per_node=len(gpu_ids) // configured_object.size,
                        cpus_per_task=1,
                        nodes=cluster_config["num_nodes"],
                        timeout_min=cluster_config["time_limit"],
                    )
                    with TensorBoard(configured_object.name):
                        executor.submit(configured_object)
                    return

                world_size = len(gpu_ids)
                if world_size == 0:
                    world_size = cpu_workers
                configured_object.setup(world_size)
                # Share tensors through /dev/shm files instead of one file descriptor per tensor:
                # spawning a worker that pickles a loaded model can otherwise exhaust the process
                # open-file limit ("Too many open files"), e.g. under Slicer's embedded Python.
                mp.set_sharing_strategy("file_system")
                with TensorBoard(configured_object.name):
                    if _runs_inline(world_size):
                        configured_object(0)
                    else:
                        mp.spawn(configured_object, nprocs=world_size)
    finally:
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        random.setstate(previous_rng[0])
        np.random.set_state(previous_rng[1])
        torch.set_rng_state(previous_rng[2])
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = previous_cudnn


_cpu_budget_applied = False


def apply_cpu_thread_budget() -> None:
    """Give each rank a bounded share of the node's cores instead of torch's every-core default.

    Past memory-bus saturation more intraop threads only add barrier contention; on a hybrid
    24-core CPU the 498^3 separable gather measures 0.7 s at 12 threads and 67 s at 24. An
    explicit ``OMP_NUM_THREADS`` keeps authority (torch already honors it at init).

    Applied once per process, and never on macOS: torch documents ``set_num_threads`` as to be
    called before any parallel work, and the Python API runs several workflows in one process.
    On macOS the call intermittently crashes libomp with SIGSEGV once any parallel region ran,
    whichever call is the first; the saturation this bounds was measured on many-core Linux
    nodes, so macOS keeps torch's default.
    """
    global _cpu_budget_applied
    if sys.platform == "darwin" or _cpu_budget_applied or os.environ.get("OMP_NUM_THREADS"):
        return
    _cpu_budget_applied = True
    local_ranks = max(1, int(os.environ.get("KONFAI_LOCAL_RANKS") or 1))
    torch.set_num_threads(max(1, min((os.cpu_count() or 1) // local_ranks, 12)))


def setup_gpu(world_size: int, rank: int | None = None, process_group: bool = True) -> tuple[int | None, int | None]:
    """Resolve the rank and, with ``process_group``, initialize torch distributed on it."""
    if os.name == "nt":
        return rank, rank
    if rank is None:
        import submitit

        job_env = submitit.JobEnvironment()
        global_rank = job_env.global_rank
        local_rank = job_env.local_rank
    else:
        global_rank = rank
        local_rank = rank
    if global_rank >= world_size:
        return None, None
    if not process_group:
        return global_rank, local_rank
    try:
        nodelist = os.getenv("SLURM_JOB_NODELIST")
        if nodelist is None:
            raise RuntimeError("SLURM_JOB_NODELIST is not set.")
        scontrol_path = shutil.which("scontrol")
        if scontrol_path is None:
            raise FileNotFoundError("scontrol not found in PATH")
        # `scontrol show hostnames` prints one host per line; for a multi-node job take only the FIRST
        # (the rendezvous master). Without this the whole newline-joined list leaks into init_method
        # (tcp://node001\nnode002:port) and init_process_group can never rendezvous.
        host_name = (
            subprocess.check_output(  # nosec B603
                [scontrol_path, "show", "hostnames", nodelist], text=True, stderr=subprocess.DEVNULL
            )
            .strip()
            .splitlines()[0]
        )
    except Exception:
        host_name = "localhost"
    port = os.environ.get("KONFAI_MASTER_PORT")
    if not port:
        port = str(find_free_port())
        os.environ["KONFAI_MASTER_PORT"] = port
    if dist.is_nccl_available() and torch.cuda.is_available() and len(cuda_visible_devices()):
        torch.cuda.empty_cache()
        dist.init_process_group(
            backend="nccl",
            rank=global_rank,
            init_method=f"tcp://{host_name}:{port}",
            world_size=world_size,
        )
    else:
        if not dist.is_initialized():
            dist.init_process_group(
                backend="gloo",
                init_method=f"tcp://{host_name}:{port}",
                rank=global_rank,
                world_size=world_size,
            )
    return global_rank, local_rank


def find_free_port():
    """Reserve and return an ephemeral TCP port on the current host."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def cleanup():
    """Destroy the active torch distributed process group when present."""
    if dist.is_initialized():
        dist.destroy_process_group()


def synchronize_data(world_size: int, gpu: int, data: Any) -> list[Any]:
    """Gather arbitrary Python objects across ranks when distributed is active."""
    if dist.is_initialized():
        outputs: list[Any] = [None] * world_size
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu)
        dist.all_gather_object(outputs, data)
    else:
        outputs = [data]
    return outputs
