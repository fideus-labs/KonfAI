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

An ``auto`` budget measures the node (a cgroup ceiling, a SLURM grant, the host's free RAM), a
declared one is the caller's per-rank figure; both land in a :class:`MemoryBudget` that answers the
one question every consumer had been re-deriving: what is MY rank's share.
"""

from __future__ import annotations

import os
import re
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

import psutil

from konfai.utils.errors import ConfigError

# Fraction of the detected node memory an ``"auto"`` budget offers the cache; the rest is reserved for
# the model's optimizer/gradient state, DataLoader worker copies, CUDA pinned staging buffers, and
# allocator slack. Caching runs with zero DataLoader workers, so a fifth of the node held back is ample.
AUTO_MEMORY_SAFETY_FRACTION = 0.8

# Decimal (10^n) and binary (2^n) suffixes; "" / "b" are bytes. Case is folded before lookup.
_MEMORY_UNIT_BYTES: dict[str, int] = {
    "": 1, "b": 1,
    "k": 10**3, "kb": 10**3, "kib": 2**10,
    "m": 10**6, "mb": 10**6, "mib": 2**20,
    "g": 10**9, "gb": 10**9, "gib": 2**30,
    "t": 10**12, "tb": 10**12, "tib": 2**40,
}  # fmt: skip


def format_bytes(num_bytes: float) -> str:
    """Human bytes at the unit that carries digits: a 0.3 MB refusal must not read '0.00 GiB'."""
    for shift, unit in ((40, "TiB"), (30, "GiB"), (20, "MiB"), (10, "KiB")):
        if abs(num_bytes) >= 2**shift:
            return f"{num_bytes / 2**shift:.2f} {unit}"
    return f"{num_bytes:.0f} B"


@dataclass(frozen=True)
class MemoryBudget:
    """A resolved memory budget that knows its own scope.

    The one decision no consumer may make for itself: an ``auto`` budget measures the NODE, so
    ranks sharing it split it; an explicit budget is the user's per-rank figure, taken as is.
    Handing consumers a bare number with an ``is_auto`` flag makes each of them re-derive that
    rule: this object answers it once, through :meth:`per_rank_bytes`.
    """

    total_bytes: float
    description: str
    shared_across_ranks: bool

    def per_rank_bytes(self, world_size: int) -> float:
        return self.total_bytes / max(1, world_size) if self.shared_across_ranks else self.total_bytes


def _positive_int_env(name: str) -> int | None:
    """The environment variable NAME as a positive int; ``None`` when unset, zero or not a number."""
    try:
        value = int(os.environ.get(name, ""))
    except ValueError:
        return None
    return value if value > 0 else None


def node_local_ranks(world_size: int | None = None) -> int:
    """How many ranks of this run share ONE node's RAM: the divisor a node-scoped budget needs.

    The launcher publishes the count in ``KONFAI_LOCAL_RANKS``, because a budget is often sized before
    the spawn where a world size exists at all. Without it (direct API use, a garbled value) the
    world size stands in: exact on a single node, conservative everywhere else.
    """
    local = _positive_int_env("KONFAI_LOCAL_RANKS")
    if local is None:
        return max(1, world_size or 1)
    return min(local, world_size) if world_size else local


# psutil.virtual_memory() always reports the HOST, so inside a container or a SLURM cgroup that grants
# far less than the node has, a memory budget derived from it would overshoot the real limit and get
# OOM-killed. The cgroup ceiling is read directly instead: the process's own cgroup (from
# /proc/self/cgroup, since only ``docker run`` puts it at the mount root) and every ancestor up to the
# mount, the tightest one winning. cgroup v2 exposes ``memory.max`` (the literal ``"max"`` means
# unbounded); cgroup v1 exposes ``memory.limit_in_bytes`` with a page-aligned near INT64_MAX sentinel.
# What the cgroup already holds is read from ``memory.stat`` and counts only the unreclaimable part
# (``anon`` + ``kernel`` in v2, ``rss`` in v1): ``memory.current``/``usage_in_bytes`` include the page
# cache, which the kernel drops under pressure, so a step that has streamed a cohort would otherwise
# look full. Only Linux has these files.
_CGROUP_ROOT = "/sys/fs/cgroup"
_PROC_SELF_CGROUP = "/proc/self/cgroup"
_CGROUP_UNLIMITED = 1 << 62  # any v1 limit at or above this is the "no limit" sentinel
_CGROUP_V2_HELD_KEYS: tuple[str, ...] = ("anon", "kernel")
_CGROUP_V1_HELD_KEYS: tuple[str, ...] = ("rss",)


def _cgroup_paths() -> list[tuple[Path, str, tuple[str, ...]]]:
    """The (directory, limit file, held-memory keys of ``memory.stat``) of this process's memory cgroup
    and its ancestors, innermost first: v2 (``0::/path``) or v1 (``N:memory:/path``); empty when there
    is no cgroup."""
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
            base, limit_name, held_keys = root, "memory.max", _CGROUP_V2_HELD_KEYS
        elif "memory" in controllers.split(","):
            base, limit_name, held_keys = root / "memory", "memory.limit_in_bytes", _CGROUP_V1_HELD_KEYS
        else:
            continue
        leaf = base / path.lstrip("/")
        chain = [leaf, *leaf.parents]
        return [(d, limit_name, held_keys) for d in chain if d == base or base in d.parents]
    return []


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

    The ceiling is the tightest ``memory.max``/``limit_in_bytes`` on the path from the process's own
    cgroup to the mount root; the held bytes are the innermost cgroup's unreclaimable memory, so a
    SLURM step already holding memory does not count it as free. A missing hierarchy (non-Linux,
    cgroups disabled), the ``"max"`` keyword, the v1 sentinel and unparseable values all mean "no
    bound at this level".
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
    Read as well as the cgroup: on a cluster whose slurmd does not enforce cgroups, the env is the bound.
    ``--mem=0`` means the whole node, not zero: a zero grant is no bound."""
    per_node = _positive_int_env("SLURM_MEM_PER_NODE")
    if per_node is not None:
        return per_node * 2**20
    per_cpu = _positive_int_env("SLURM_MEM_PER_CPU")
    cpus = _positive_int_env("SLURM_CPUS_PER_TASK") or _positive_int_env("SLURM_CPUS_ON_NODE")
    if per_cpu is not None and cpus is not None:
        return per_cpu * cpus * 2**20
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
        limit, held = cgroup
        candidates.append((max(0, limit - held), "cgroup limit"))
    slurm = _slurm_memory_grant()
    if slurm is not None:
        candidates.append((slurm, "SLURM memory grant"))
    return min(candidates, key=lambda candidate: candidate[0])


def parse_memory_budget_bytes(value: str | float) -> int:
    """Parse an explicit memory budget to bytes: a bare number is GiB, a string carries its own unit.

    KonfAI reports RAM in GiB throughout, so an unadorned ``24`` reads as ``24 GiB``: whether it
    arrives as a number or, through the YAML binding, as the string ``"24"``. A string may name its
    unit: decimal ``GB``/``MB`` (10^n) or binary ``GiB``/``MiB`` (2^n), case-insensitive, optional
    space (``"24GB"``, ``"32 GiB"``, ``"512mb"``); ``"b"`` means bytes. ``"auto"`` is resolved by the
    caller, not here.
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
    """The configured ``memory_budget`` as an object that knows its own scope.

    ``None``/``"auto"`` offers ``AUTO_MEMORY_SAFETY_FRACTION`` of the node's allocatable memory: a
    NODE budget, which ranks sharing the node split; an explicit budget is the caller's own figure,
    per rank as declared.
    """
    if memory_budget is None or (isinstance(memory_budget, str) and memory_budget.strip().lower() == "auto"):
        node_bytes, source = available_memory_bytes()
        return MemoryBudget(
            node_bytes * AUTO_MEMORY_SAFETY_FRACTION,
            f"auto: {format_bytes(node_bytes)} {source} x {AUTO_MEMORY_SAFETY_FRACTION:.0%}",
            shared_across_ranks=True,
        )
    return MemoryBudget(
        float(parse_memory_budget_bytes(memory_budget)), f"{memory_budget!r}", shared_across_ranks=False
    )
