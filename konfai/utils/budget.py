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

"""The memory budget: what a run may hold, resolved once and shared by every consumer.

An ``auto`` budget measures the node (a cgroup ceiling, a SLURM grant, the host's free RAM) and the
ranks sharing that node split it; a declared budget is the caller's per-rank figure.
"""

from __future__ import annotations

import math
import os
import re
import warnings
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

import psutil

from konfai.utils.errors import ConfigError

# Fraction of the detected node memory an ``"auto"`` budget offers the cache; the rest is reserved for
# the model's optimizer/gradient state, DataLoader worker copies, pinned staging buffers and allocator slack.
AUTO_MEMORY_SAFETY_FRACTION = 0.8

#: The smallest declared budget the stack can honor (see the warning in resolve_memory_budget).
MINIMUM_DECLARED_BUDGET_BYTES = 256 << 20

# Decimal (10^n) and binary (2^n) suffixes; "" / "b" are bytes. Case is folded before lookup.
_MEMORY_UNIT_BYTES: dict[str, int] = {
    "": 1, "b": 1,
    "k": 10**3, "kb": 10**3, "kib": 2**10,
    "m": 10**6, "mb": 10**6, "mib": 2**20,
    "g": 10**9, "gb": 10**9, "gib": 2**30,
    "t": 10**12, "tb": 10**12, "tib": 2**40,
}  # fmt: skip


def format_bytes(num_bytes: float) -> str:
    """Human-readable bytes at the largest unit that carries digits."""
    for shift, unit in ((40, "TiB"), (30, "GiB"), (20, "MiB"), (10, "KiB")):
        if abs(num_bytes) >= 2**shift:
            return f"{num_bytes / 2**shift:.2f} {unit}"
    return f"{num_bytes:.0f} B"


@dataclass(frozen=True)
class MemoryBudget:
    """A resolved memory budget that knows its own scope: an ``auto`` budget measures the NODE, so
    ranks sharing it split it, where an explicit budget is the user's per-rank figure taken as is.
    :meth:`per_rank_bytes` applies that rule."""

    total_bytes: float
    description: str
    shared_across_ranks: bool

    def per_rank_bytes(self, world_size: int) -> float:
        return self.total_bytes / max(1, world_size) if self.shared_across_ranks else self.total_bytes

    def work_bytes(self, world_size: int) -> float:
        """What is left of this rank's budget for the work itself. An ``auto`` budget is the MACHINE's
        figure, so what the process already holds resident is taken off it; a DECLARED budget states what
        the work may take, so nothing is. Never below zero."""
        per_rank = self.per_rank_bytes(world_size)
        if not self.shared_across_ranks:
            return per_rank
        return max(0.0, per_rank - (resident_bytes() or 0))


#: How a declared per-rank budget divides between the consumers that can be holding AT ONCE:
#: shares of ONE declaration, so they add to one. Which consumers are simultaneous is a property of
#: the call path, not of this table.
BUDGET_SHARES: dict[str, float] = {
    "regions": 0.50,  # what a route holds in the regions it is landing (a fold's kept folds included)
    "chains": 0.35,  # what a chain may hold while producing one of them: its sweeps, its walk slab
    "cache": 0.15,  # the store's decoded-chunk cache, which outlives any single region
}

if abs(sum(BUDGET_SHARES.values()) - 1.0) > 1e-9:  # pragma: no cover - a typo, caught at import
    raise ValueError(f"BUDGET_SHARES must divide one declaration, not {sum(BUDGET_SHARES.values())}")


def budget_share(name: str, budget_bytes: float | None = None) -> float | None:
    """One consumer's share of the declared per-rank budget (:data:`BUDGET_SHARES`), or ``None``
    when none was declared."""
    declared = per_rank_budget_bytes() if budget_bytes is None else budget_bytes
    if not declared or declared <= 0:
        return None
    return declared * BUDGET_SHARES[name]


def sweep_share(budget_bytes: float | None = None) -> float | None:
    """What a per-case sweep's block may hold: the landing's share and the chain's together.

    A sweep's block price already includes what the chain running on it holds, so the two shares
    are one figure here. What is left over is the store's cache.
    """
    declared = per_rank_budget_bytes() if budget_bytes is None else budget_bytes
    if not declared or declared <= 0:
        return None
    return declared * (BUDGET_SHARES["regions"] + BUDGET_SHARES["chains"])


#: This rank's share of the budget, as the workflow that resolved it published it. Consumers sitting
#: inside a ``Dataset``, which a workflow reaches through doors that carry no budget, read it here.
_per_rank_bytes: float | None = None


def set_per_rank_budget(budget_bytes: float | None) -> None:
    """Publish this rank's budget. ``None`` or a non-positive figure means none was declared, and
    every consumer falls back to its own default."""
    global _per_rank_bytes
    _per_rank_bytes = float(budget_bytes) if budget_bytes and budget_bytes > 0 else None


def per_rank_budget_bytes() -> float | None:
    """This rank's declared budget, or ``None`` when none was."""
    return _per_rank_bytes


def peak_resident_bytes() -> int | None:
    """The most this process has ever held resident, or ``None`` where the kernel does not say.

    ``VmHWM`` is a lifetime high-water mark, so it answers "did this run stay inside its budget" and not
    "which region was the worst". NOT the cgroup's ``memory.peak``, which also counts the reclaimable
    page cache; ``VmHWM`` tracks the anonymous set the OOM killer counts.
    """
    return _status_bytes("VmHWM")


#: The highest ``VmHWM`` read before a reset, added back by :func:`run_peak_resident_bytes` so a
#: scope's reset never makes the run report less than it held.
_peak_before_resets = 0


def run_peak_resident_bytes() -> int | None:
    """The most this process has held resident over the whole run, across every
    :func:`reset_resident_peak`, where :func:`peak_resident_bytes` reads the scope since the last reset."""
    peak = peak_resident_bytes()
    return None if peak is None else max(peak, _peak_before_resets)


def reset_resident_peak() -> bool:
    """Set this process's resident high-water mark back to what it holds now, so the next reading of
    :func:`peak_resident_bytes` is a peak over one scope rather than over the whole run. Writes ``5`` to
    ``/proc/self/clear_refs``. Returns ``False`` where the kernel does not offer it, and such a caller
    has no per-step peak."""
    global _peak_before_resets
    peak = peak_resident_bytes()
    try:
        Path("/proc/self/clear_refs").write_text("5")
    except OSError:
        return False
    if peak is not None:
        _peak_before_resets = max(_peak_before_resets, peak)
    return True


def resident_bytes() -> int | None:
    """What this process holds resident right now (``VmRSS``), or ``None`` where the kernel does not say."""
    return _status_bytes("VmRSS")


#: The resident set a workflow recorded before its first case: what its regions are measured above.
_resident_floor: int | None = None


def record_resident_floor() -> int | None:
    """Record what this process holds resident now as the run's floor (:func:`resident_floor`).

    Called by a workflow after its setup and before its first case, so a region is measured above the
    interpreter, the libraries and the workflow's own objects.
    """
    global _resident_floor
    _resident_floor = resident_bytes()
    return _resident_floor


def resident_floor() -> int | None:
    """The run's recorded resident floor, ``None`` when no workflow is holding one in this process."""
    return _resident_floor


def clear_resident_floor() -> None:
    """Forget the recorded floor, so the next workflow in the same process measures above its own."""
    global _resident_floor
    _resident_floor = None


def _status_bytes(field: str) -> int | None:
    """A ``kB`` field of ``/proc/self/status`` in bytes, ``None`` where the kernel does not say."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith(field + ":"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def _positive_int_env(name: str) -> int | None:
    """The environment variable NAME as a positive int; ``None`` when unset, zero or not a number."""
    try:
        value = int(os.environ.get(name, ""))
    except ValueError:
        return None
    return value if value > 0 else None


def node_local_ranks(world_size: int | None = None) -> int:
    """How many ranks of this run share ONE node's RAM: the divisor a node-scoped budget needs. The
    launcher publishes the count in ``KONFAI_LOCAL_RANKS``; without it the world size stands in, exact
    on a single node and conservative everywhere else."""
    local = _positive_int_env("KONFAI_LOCAL_RANKS")
    if local is None:
        return max(1, world_size or 1)
    return min(local, world_size) if world_size else local


# psutil.virtual_memory() always reports the HOST, so the cgroup ceiling is read directly: the process's
# own cgroup (from /proc/self/cgroup) and every ancestor up to the mount, the tightest one winning.
# cgroup v2 exposes ``memory.max`` (the literal ``"max"`` means unbounded); v1 exposes
# ``memory.limit_in_bytes`` with a page-aligned near INT64_MAX sentinel. What the cgroup already holds is
# read from ``memory.stat``, the unreclaimable part only (``anon`` + ``kernel`` in v2, ``rss`` in v1).
_CGROUP_ROOT = "/sys/fs/cgroup"
_PROC_SELF_CGROUP = "/proc/self/cgroup"
_CGROUP_UNLIMITED = 1 << 62  # any v1 limit at or above this is the "no limit" sentinel
_CGROUP_V2_HELD_KEYS: tuple[str, ...] = ("anon", "kernel")
_CGROUP_V1_HELD_KEYS: tuple[str, ...] = ("rss",)


def _own_cgroup(controller: str) -> tuple[Path, str, bool] | None:
    """``(hierarchy mount, this process's cgroup path in it, v2)`` for ``controller``, read from
    ``/proc/self/cgroup``: the unified v2 line (``0::/path``) or the v1 line naming the controller;
    ``None`` when there is no cgroup."""
    try:
        lines = Path(_PROC_SELF_CGROUP).read_text().splitlines()
    except OSError:
        return None
    root = Path(_CGROUP_ROOT)
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        hierarchy, controllers, path = parts
        if hierarchy == "0" and controllers == "":
            return root, path, True
        if controller in controllers.split(","):
            # A v1 controller mounts under the name it is mounted with: the joint "cpu,cpuacct" or
            # the plain "cpu".
            mount = next((root / d for d in (controllers, controller) if (root / d).is_dir()), None)
            if mount is not None:
                return mount, path, False
    return None


def _cgroup_ancestry(base: Path, path: str) -> list[Path]:
    """The cgroup directory ``path`` under ``base`` and its ancestors up to ``base``, innermost first."""
    leaf = base / path.lstrip("/")
    return [d for d in (leaf, *leaf.parents) if d == base or base in d.parents]


def _cgroup_paths() -> list[tuple[Path, str, tuple[str, ...]]]:
    """The (directory, limit file, held-memory keys of ``memory.stat``) of this process's memory cgroup
    and its ancestors, innermost first; empty when there is no cgroup."""
    own = _own_cgroup("memory")
    if own is None:
        return []
    base, path, v2 = own
    limit_name, held_keys = (
        ("memory.max", _CGROUP_V2_HELD_KEYS) if v2 else ("memory.limit_in_bytes", _CGROUP_V1_HELD_KEYS)
    )
    return [(d, limit_name, held_keys) for d in _cgroup_ancestry(base, path)]


def _cpu_cgroup_paths() -> list[tuple[Path, bool]]:
    """This process's cpu cgroup directory and its ancestors, innermost first, each flagged v2
    (``cpu.max``) or v1 (``cpu.cfs_quota_us`` / ``cpu.cfs_period_us``)."""
    own = _own_cgroup("cpu")
    if own is None:
        return []
    base, path, v2 = own
    return [(d, v2) for d in _cgroup_ancestry(base, path)]


def _cpu_quota(directory: Path, v2: bool) -> float | None:
    """The CPU quota of one cgroup directory in cores, ``None`` when unbounded, missing or malformed."""
    try:
        if v2:
            quota, period = (directory / "cpu.max").read_text().split()
        else:
            quota = (directory / "cpu.cfs_quota_us").read_text().strip()
            period = (directory / "cpu.cfs_period_us").read_text().strip()
        if quota in ("max", "-1"):
            return None
        quota_us, period_us = int(quota), int(period)
    except (OSError, ValueError):
        return None
    if quota_us <= 0 or period_us <= 0:
        return None
    return quota_us / period_us


_cpu_quota_cores: dict[tuple[str, str], int | None] = {}


def forget_cgroup_cpu_ceiling() -> None:
    """Drop the memoised CPU quota, so the next :func:`available_cpus` reads the cgroup again."""
    _cpu_quota_cores.clear()


def _cgroup_cpu_ceiling() -> int | None:
    """The tightest CPU quota over this process's cpu cgroup and its ancestors, in cores; ``None``
    when unbounded. Read once per process: the walk opens a file per ancestor."""
    key = (_PROC_SELF_CGROUP, _CGROUP_ROOT)
    if key not in _cpu_quota_cores:
        ceiling: int | None = None
        for directory, v2 in _cpu_cgroup_paths():
            quota = _cpu_quota(directory, v2)
            if quota is not None:
                cores = max(1, math.ceil(quota))
                ceiling = cores if ceiling is None else min(ceiling, cores)
        _cpu_quota_cores[key] = ceiling
    return _cpu_quota_cores[key]


def available_cpus() -> int:
    """The cores this process may actually run on: the tighter of its affinity mask and its cgroup
    CPU quota (v2 ``cpu.max``, v1 ``cpu.cfs_quota_us``/``cpu.cfs_period_us``, over its ancestors)."""
    try:
        cores = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cores = os.cpu_count() or 1
    ceiling = _cgroup_cpu_ceiling()
    if ceiling is not None:
        cores = min(cores, ceiling)
    return max(1, cores)


def _held_memory_bytes(directory: Path, keys: tuple[str, ...]) -> int | None:
    """The sum of KEYS in DIRECTORY's ``memory.stat``, or ``None`` when the file is missing or lacks them."""
    try:
        lines = (directory / "memory.stat").read_text().splitlines()
    except OSError:
        return None
    values: dict[str, int] = {}
    for line in lines:
        key, _, raw = line.partition(" ")
        with suppress(ValueError):
            values[key] = int(raw)
    if any(key not in values for key in keys):
        return None
    return sum(values[key] for key in keys)


def _read_cgroup_memory_limit() -> tuple[int, int] | None:
    """``(ceiling, held bytes)`` for this process's cgroup, or ``None`` when unbounded or absent.

    The ceiling is the tightest ``memory.max``/``limit_in_bytes`` from the process's own cgroup to the
    mount root; the held bytes are the innermost cgroup's unreclaimable memory. A missing hierarchy, the
    ``"max"`` keyword, the v1 sentinel and unparseable values all mean "no bound at this level".
    """
    limits: list[int] = []
    held: int | None = None
    for directory, limit_name, held_keys in _cgroup_paths():
        try:
            raw = (directory / limit_name).read_text().strip()
        except OSError:
            continue
        if raw != "max":
            with suppress(ValueError):
                limit = int(raw)
                if limit < _CGROUP_UNLIMITED:
                    limits.append(limit)
        if held is None:
            held = _held_memory_bytes(directory, held_keys)
    if not limits:
        return None
    return min(limits), held or 0


def _slurm_memory_grant() -> int | None:
    """The bytes SLURM granted this step (``--mem`` / ``--mem-per-cpu``, in MB), or ``None`` outside a job.
    Read as well as the cgroup, which a slurmd may not enforce. ``--mem=0`` means the whole node,
    so a zero grant is no bound."""
    per_node = _positive_int_env("SLURM_MEM_PER_NODE")
    if per_node is not None:
        return per_node * 2**20
    per_cpu = _positive_int_env("SLURM_MEM_PER_CPU")
    cpus = _positive_int_env("SLURM_CPUS_PER_TASK") or _positive_int_env("SLURM_CPUS_ON_NODE")
    if per_cpu is not None and cpus is not None:
        return per_cpu * cpus * 2**20
    return None


def available_memory_bytes() -> tuple[int, str]:
    """Return ``(bytes a process may safely allocate, source label)``: the tightest of the cgroup's
    remaining room (its ceiling minus what it already holds), a SLURM memory grant and the host's free
    RAM. The label names which bound won."""
    candidates = [(int(psutil.virtual_memory().available), "host available RAM")]
    cgroup = _read_cgroup_memory_limit()
    if cgroup is not None:
        limit, held = cgroup
        candidates.append((max(0, limit - held), "cgroup limit"))
    slurm = _slurm_memory_grant()
    if slurm is not None:
        candidates.append((slurm, "SLURM memory grant"))
    return min(candidates, key=lambda candidate: candidate[0])


def parse_memory_budget_bytes(value: str | float) -> int:
    """Parse an explicit memory budget to bytes: a bare number is GiB, a string carries its own unit.

    An unadorned ``24`` reads as ``24 GiB``, as a number or as the YAML string ``"24"``. A string may
    name its unit: decimal ``GB``/``MB`` (10^n) or binary ``GiB``/``MiB`` (2^n), case-insensitive, with
    an optional space (``"24GB"``, ``"32 GiB"``, ``"512mb"``); ``"b"`` means bytes. ``"auto"`` is
    resolved by the caller.
    """
    if isinstance(value, str):
        match = re.fullmatch(r"\s*(?P<number>[0-9]*\.?[0-9]+)\s*(?P<unit>[a-z]*)\s*", value.lower())
        if match is None or match.group("unit") not in _MEMORY_UNIT_BYTES:
            raise ConfigError(
                f"memory_budget: '{value}' is not a valid memory size.",
                "Use a number in GiB (e.g. 24), a unit string ('24GB', '32GiB', '512MB'), 'auto', or None "
                "(the default): which means 'auto': size from the detected memory.",
            )
        unit = match.group("unit")
        # A bare numeric string is the YAML face of a bare number: GiB, not bytes.
        number, factor = float(match.group("number")), _MEMORY_UNIT_BYTES[unit] if unit else 2**30
    else:
        number, factor = float(value), 2**30
    if number <= 0:
        raise ConfigError(
            f"memory_budget: {value!r} must be a positive size.",
            "Use a positive number in GiB (e.g. 24), a unit string ('24GB'), 'auto', or None.",
        )
    return int(number * factor)


def resolve_memory_budget(memory_budget: str | float | None) -> MemoryBudget:
    """The configured ``memory_budget`` as an object that knows its own scope: ``None``/``"auto"`` offers
    ``AUTO_MEMORY_SAFETY_FRACTION`` of the node's allocatable memory, a NODE budget which the ranks
    sharing the node split, where an explicit budget is per rank as declared."""
    if memory_budget is None or (isinstance(memory_budget, str) and memory_budget.strip().lower() == "auto"):
        node_bytes, source = available_memory_bytes()
        return MemoryBudget(
            node_bytes * AUTO_MEMORY_SAFETY_FRACTION,
            f"auto: {format_bytes(node_bytes)} {source} x {AUTO_MEMORY_SAFETY_FRACTION:.0%}",
            shared_across_ranks=True,
        )
    declared = parse_memory_budget_bytes(memory_budget)
    if declared < MINIMUM_DECLARED_BUDGET_BYTES:
        # Below this the declaration describes no process that can run: the interpreter with torch and
        # one imaging backend is already several hundred MiB resident before the first voxel. Warned
        # rather than refused: tests and probes size tiny fixtures under tiny declarations on purpose.
        warnings.warn(
            f"memory_budget {memory_budget!r} is below the smallest supported declaration"
            f" ({MINIMUM_DECLARED_BUDGET_BYTES >> 20} MiB): the process floor alone is several times"
            " this figure, and what the sizing model cannot see may exceed it. Declare at least"
            " 512 MiB, or 'auto' to size from the detected memory.",
            UserWarning,
            stacklevel=2,
        )
    return MemoryBudget(float(declared), f"{memory_budget!r}", shared_across_ranks=False)
