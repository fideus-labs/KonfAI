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

"""The shared measurement contract of ``benchmarks/perf``.

Every bench in this folder goes through the same four things, so two numbers are only ever compared
on the same footing:

- :func:`fingerprint`: the commit (and whether the tree is dirty), the versions, the CPU, the GPU and
  its driver, the power profile, the load average, the thread pin. Written into every result.
- :func:`machine_gate`: refuses to time on a machine that is not quiet (load, power profile, GPU busy,
  another process eating a core) unless ``--force``; the refusal names what it saw. The audit that
  produced this folder measured a 2x spread on every absolute number from the profile and the load
  alone, so the gate is the first line of every protocol here.
- :class:`PeakSampler` / :class:`GpuSampler`: whole-process-tree resident set at 50 ms and the GPU's
  used memory at 250 ms, so children (DataLoader workers, spawned ranks) count.
- :func:`write_result`: one JSON per bench run under ``results/<host>/``, with the fingerprint, and
  one markdown row so the number a page carries can be traced to a file.

The helpers below wrap the real profilers: ``torch.profiler`` (kernel time by family),
``cProfile`` (host time by function, with the share of a named family), the ``[KonfAI] ...`` clock
lines every workflow prints (parsed, never re-implemented). Nothing here is a benchmark by itself.
"""

from __future__ import annotations

import contextlib
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import psutil

PERF_DIR = Path(__file__).resolve().parent
REPO = PERF_DIR.parents[1]
RESULTS_DIR = PERF_DIR / "results"

# --------------------------------------------------------------------------------------- fingerprint


def _run(argv: list[str], timeout: float = 20.0) -> str:
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def git_revision() -> dict[str, Any]:
    sha = _run(["git", "-C", str(REPO), "rev-parse", "--short=12", "HEAD"])
    dirty = bool(_run(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"]))
    describe = _run(["git", "-C", str(REPO), "describe", "--tags", "--always"])
    return {"sha": sha, "dirty": dirty, "describe": describe}


def power_profile() -> str:
    if shutil.which("powerprofilesctl"):
        return _run(["powerprofilesctl", "get"]) or "unknown"
    return "unavailable"


def gpu_info() -> dict[str, Any]:
    if not shutil.which("nvidia-smi"):
        return {"name": None}
    line = _run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])
    if not line:
        return {"name": None}
    parts = [part.strip() for part in line.split(",")]
    return {
        "name": parts[0],
        "driver": parts[1] if len(parts) > 1 else None,
        "memory_total": parts[2] if len(parts) > 2 else None,
    }


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def versions() -> dict[str, str]:
    out: dict[str, str] = {"python": platform.python_version()}
    for name in ("konfai", "torch", "numpy", "SimpleITK", "h5py", "zarr", "ngff_zarr"):
        try:
            module = __import__(name)
            out[name] = str(getattr(module, "__version__", "?"))
        except Exception:
            out[name] = "absent"
    return out


def fingerprint(*, import_versions: bool = True) -> dict[str, Any]:
    """Everything a number needs beside it to be comparable with another number."""
    load1, load5, load15 = os.getloadavg()
    return {
        "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": platform.node(),
        "git": git_revision(),
        "versions": versions() if import_versions else {},
        "cpu": {"model": cpu_model(), "count": os.cpu_count(), "affinity": len(os.sched_getaffinity(0))},
        "gpu": gpu_info(),
        "power_profile": power_profile(),
        "load_avg": [round(load1, 2), round(load5, 2), round(load15, 2)],
        "threads": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
    }


# --------------------------------------------------------------------------------------- the gate


@dataclass
class GateReport:
    warnings: list[str] = field(default_factory=list)

    @property
    def quiet(self) -> bool:
        return not self.warnings


def gpu_busy(threshold_percent: float = 5.0) -> str | None:
    if not shutil.which("nvidia-smi"):
        return None
    line = _run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"])
    if not line:
        return None
    util, used = (part.strip() for part in line.split(",")[:2])
    try:
        if float(util) > threshold_percent:
            return f"GPU utilization {util} % (another process is computing)"
    except ValueError:
        return None
    apps = _run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"])
    foreign = [row for row in apps.splitlines() if row.strip() and not row.startswith(str(os.getpid()))]
    if foreign:
        return f"other GPU processes: {'; '.join(foreign[:3])} (GPU memory used {used} MiB)"
    return None


def cpu_hogs(threshold_percent: float = 50.0) -> list[str]:
    """Processes outside this tree that hold more than ``threshold_percent`` of one core."""
    mine = {os.getpid(), *(child.pid for child in psutil.Process().children(recursive=True))}
    hogs: list[str] = []
    for proc in psutil.process_iter(["pid", "name"]):
        if proc.info["pid"] in mine:
            continue
        try:
            proc.cpu_percent(None)
        except psutil.Error:
            continue
    time.sleep(0.5)
    for proc in psutil.process_iter(["pid", "name"]):
        if proc.info["pid"] in mine:
            continue
        try:
            percent = proc.cpu_percent(None)
        except psutil.Error:
            continue
        if percent > threshold_percent:
            hogs.append(f"{proc.info['name']} (pid {proc.info['pid']}, {percent:.0f} %)")
    return hogs


def machine_gate(*, max_load: float = 3.0, require_performance: bool = True, force: bool = False) -> GateReport:
    """Refuse to time on a machine that is not quiet; ``force`` turns the refusal into a warning."""
    report = GateReport()
    load1 = os.getloadavg()[0]
    # A series (run_all) gates the load once, up front: the bench before this one leaves a load
    # average of its own that says nothing about foreign work.
    if load1 > max_load and not os.environ.get("KONFAI_PERF_GATED"):
        report.warnings.append(f"load average {load1:.1f} > {max_load}")
    profile = power_profile()
    if require_performance and profile not in ("performance", "unavailable"):
        report.warnings.append(f"power profile is '{profile}', not 'performance'")
    busy = gpu_busy()
    if busy:
        report.warnings.append(busy)
    hogs = cpu_hogs()
    if hogs:
        report.warnings.append("CPU hogs: " + ", ".join(hogs[:4]))
    if report.warnings and not force:
        raise SystemExit(
            "[perf] the machine is not quiet; refusing to time (pass --force to record the numbers anyway):\n  - "
            + "\n  - ".join(report.warnings)
        )
    for warning in report.warnings:
        print(f"[perf] WARNING: {warning}", file=sys.stderr)
    return report


# --------------------------------------------------------------------------------------- samplers


def _tree_rss(process: psutil.Process) -> int:
    total = 0
    for member in [process, *process.children(recursive=True)]:
        try:
            total += member.memory_info().rss
        except psutil.Error:
            continue
    return total


class PeakSampler(threading.Thread):
    """Whole-tree peak RSS in bytes, sampled at ``interval`` seconds; children count."""

    def __init__(self, pid: int | None = None, interval: float = 0.05) -> None:
        super().__init__(daemon=True)
        self._process = psutil.Process(pid)
        self._interval = interval
        self._stop_event = threading.Event()
        self.peak = 0
        self.baseline = _tree_rss(self._process)

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.peak = max(self.peak, _tree_rss(self._process))
            except psutil.Error:
                break
            time.sleep(self._interval)

    def stop(self) -> int:
        self._stop_event.set()
        self.join()
        return self.peak


class GpuSampler(threading.Thread):
    """Peak of the GPU's used memory (MiB, the whole device) at ``interval`` seconds, and the baseline
    before the run: the difference is what the run added, whoever else holds the rest."""

    def __init__(self, interval: float = 0.25) -> None:
        super().__init__(daemon=True)
        self._interval = interval
        self._stop_event = threading.Event()
        self.enabled = bool(shutil.which("nvidia-smi"))
        self.baseline = self._used() if self.enabled else 0
        self.peak = self.baseline

    @staticmethod
    def _used() -> int:
        line = _run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], timeout=5)
        try:
            return int(float(line.strip()))
        except ValueError:
            return 0

    def run(self) -> None:
        if not self.enabled:
            return
        while not self._stop_event.is_set():
            self.peak = max(self.peak, self._used())
            time.sleep(self._interval)

    def stop(self) -> int:
        self._stop_event.set()
        self.join()
        return self.peak


# --------------------------------------------------------------------------------------- clocks


_CLOCK_LINE = re.compile(r"\[KonfAI\] (?P<name>[a-z()]+) (?P<wall>[0-9.]+) s = (?P<parts>.*)")
_PART = re.compile(r"(?P<phase>[a-z()+]+) (?P<value>[0-9.]+)")


def parse_clock_lines(text: str) -> dict[str, dict[str, float]]:
    """The ``[KonfAI] <name> W s = a 1.2 + b 3.4 + ... | ...`` lines of a run, as ``{name: {phase: s}}``
    with the wall under ``"wall"``; the last line of each name wins (a run prints its clocks once)."""
    out: dict[str, dict[str, float]] = {}
    for line in text.splitlines():
        match = _CLOCK_LINE.search(line)
        if not match:
            continue
        phases = {"wall": float(match.group("wall"))}
        head, _, tail = match.group("parts").partition("|")
        for part in _PART.finditer(head):
            phases[part.group("phase")] = float(part.group("value"))
        for part in _PART.finditer(tail):
            phases["after_bar:" + part.group("phase")] = float(part.group("value"))
        out[match.group("name")] = phases
    return out


# --------------------------------------------------------------------------------------- running things


@dataclass
class CliRun:
    argv: list[str]
    returncode: int
    wall_s: float
    peak_rss_bytes: int
    gpu_peak_mib: int
    gpu_baseline_mib: int
    output: str
    clocks: dict[str, dict[str, float]]

    @property
    def peak_rss_gib(self) -> float:
        return round(self.peak_rss_bytes / 2**30, 3)


def run_cli(
    argv: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: float = 3600.0,
    log_path: Path | None = None,
) -> CliRun:
    """Run a KonfAI command line, timing its wall and sampling its tree's RSS and the GPU's memory."""
    merged = dict(os.environ)
    if env:
        merged.update(env)
    gpu = GpuSampler()
    gpu.start()
    start = time.perf_counter()
    process = subprocess.Popen(argv, cwd=cwd, env=merged, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    sampler = PeakSampler(process.pid)
    sampler.start()
    try:
        output, _ = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        output, _ = process.communicate()
    wall = time.perf_counter() - start
    peak = sampler.stop()
    gpu_peak = gpu.stop()
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(output)
    return CliRun(
        argv, process.returncode, round(wall, 3), peak, gpu_peak, gpu.baseline, output, parse_clock_lines(output)
    )


def median_of(fn: Callable[[], float], repeats: int = 3, warmup: int = 1) -> tuple[float, list[float]]:
    """``fn`` returns one measurement; the median after ``warmup`` discarded calls, with every run kept."""
    for _ in range(warmup):
        fn()
    runs = [fn() for _ in range(repeats)]
    return statistics.median(runs), runs


@contextlib.contextmanager
def cuda_timer(device: Any) -> Iterator[dict[str, float]]:
    """Wall of a device-side block in milliseconds, synchronized on both ends (``result["ms"]``)."""
    import torch

    result: dict[str, float] = {}
    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(device)
        start.record()
        yield result
        end.record()
        torch.cuda.synchronize(device)
        result["ms"] = float(start.elapsed_time(end))
    else:
        t0 = time.perf_counter()
        yield result
        result["ms"] = (time.perf_counter() - t0) * 1e3


# --------------------------------------------------------------------------------------- profilers


KERNEL_FAMILIES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("conv_gemm", re.compile(r"conv|gemm|cutlass|wgrad|dgrad|fprop|implicit|winograd|cudnn::", re.I)),
    ("layout", re.compile(r"nchw|nhwc|transpose|permute|channels", re.I)),
    ("norm_act", re.compile(r"batch_norm|instance_norm|layer_norm|relu|gelu|silu|softmax|sigmoid", re.I)),
    ("loss_like", re.compile(r"nll|cross_entropy|bincount|dice|log_softmax|scatter|index_put|unique|sort", re.I)),
    ("optimizer", re.compile(r"adam|sgd|foreach|lerp|addcdiv|addcmul|_fused", re.I)),
    ("reduce", re.compile(r"reduce|sum|mean|max|min|argmax|cub::|DeviceReduce", re.I)),
    ("elementwise_copy", re.compile(r"elementwise|vectorized|copy|fill|cast|unrolled|memcpy|memset", re.I)),
)


def kernel_families(prof: Any) -> dict[str, float]:
    """CUDA kernel time (ms) by family from a ``torch.profiler`` trace; every kernel counted once."""
    totals: dict[str, float] = {name: 0.0 for name, _ in KERNEL_FAMILIES}
    totals["other"] = 0.0
    for event in prof.events():
        if getattr(event, "device_type", None) is None:
            continue
        if str(event.device_type).endswith("CUDA") and getattr(event, "is_user_annotation", False) is False:
            name = event.name
            if name.startswith("cuda") or name.startswith("aten::") or name.startswith("ProfilerStep"):
                continue
            duration_ms = (event.time_range.end - event.time_range.start) / 1e3
            for family, pattern in KERNEL_FAMILIES:
                if pattern.search(name):
                    totals[family] += duration_ms
                    break
            else:
                totals["other"] += duration_ms
    total = sum(totals.values())
    return {**{k: round(v, 3) for k, v in totals.items()}, "total_ms": round(total, 3)}


def cprofile_summary(prof_path: Path, *, top: int = 15, families: dict[str, str] | None = None) -> dict[str, Any]:
    """The top functions by cumulative time of a ``cProfile`` dump and the share of named families
    (``{"statistics": r"statistics\\.py"}``) in the total host time."""
    import pstats

    stats = pstats.Stats(str(prof_path))
    total = stats.total_tt
    rows = []
    for (filename, line, function), (_, ncalls, tottime, cumtime, _) in stats.stats.items():  # type: ignore[attr-defined]
        rows.append(
            {
                "function": f"{Path(filename).name}:{line}({function})",
                "ncalls": ncalls,
                "tottime": round(tottime, 3),
                "cumtime": round(cumtime, 3),
            }
        )
    rows.sort(key=lambda row: row["cumtime"], reverse=True)
    shares: dict[str, float] = {}
    cumulative: dict[str, float] = {}
    for family, pattern in (families or {}).items():
        regex = re.compile(pattern)
        matching = [row for row in rows if regex.search(row["function"])]
        shares[family] = round(sum(row["tottime"] for row in matching) / total if total else 0.0, 3)
        # the family's own time (tottime) against everything its entry point waits for (max cumtime):
        # a scan that reads a volume charges the read to h5py, not to itself
        cumulative[family] = round(max((row["cumtime"] for row in matching), default=0.0), 3)
    return {
        "total_tottime_s": round(total, 3),
        "top": rows[:top],
        "family_share_of_tottime": shares,
        "family_max_cumtime_s": cumulative,
    }


# --------------------------------------------------------------------------------------- results


def results_dir() -> Path:
    path = RESULTS_DIR / platform.node()
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_result(bench: str, payload: dict[str, Any], *, fp: dict[str, Any] | None = None, tag: str = "") -> Path:
    """One JSON per bench run, named by date, commit and bench; returns the path."""
    fp = fp or fingerprint()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    name = f"{stamp}-{fp['git']['sha'][:8]}{'-dirty' if fp['git']['dirty'] else ''}-{bench}{('-' + tag) if tag else ''}.json"
    path = results_dir() / name
    path.write_text(json.dumps({"bench": bench, "fingerprint": fp, "result": payload}, indent=2, default=str) + "\n")
    return path


def markdown_row(cells: Iterable[Any]) -> str:
    return "| " + " | ".join(str(cell) for cell in cells) + " |"


def example_dir(name: str) -> Path:
    """The shipped example, which every workflow bench copies to scratch before running."""
    path = REPO / "examples" / name
    if not path.is_dir():
        raise SystemExit(f"[perf] example not found: {path}")
    return path


def copy_example(
    name: str, scratch: Path, *, exclude: Iterable[str] = ("Predictions", "Evaluations", "Statistics", "__pycache__")
) -> Path:
    """A working copy of an example without its outputs; the copy owns its own Checkpoints too."""
    source = example_dir(name)
    target = scratch / name
    excluded = set(exclude)
    shutil.copytree(source, target, ignore=lambda _, names: [n for n in names if n in excluded or n.endswith(".ipynb")])
    return target


def konfai_executable() -> list[str]:
    """The ``konfai`` CLI of the running interpreter (the same environment that imports it)."""
    candidate = Path(sys.executable).with_name("konfai")
    if candidate.exists():
        return [str(candidate)]
    return [sys.executable, "-m", "konfai.main"]
