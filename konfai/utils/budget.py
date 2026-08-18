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
from dataclasses import dataclass

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


def node_local_ranks(world_size: int | None = None) -> int:
    """How many ranks of this run share ONE node's RAM: the divisor a node-scoped budget needs.

    The launcher publishes the count in ``KONFAI_LOCAL_RANKS``, because a budget is often sized before
    the spawn where a world size exists at all. Without it (direct API use, a garbled value) the
    world size stands in: exact on a single node, conservative everywhere else.
    """
    try:
        local = int(os.environ.get("KONFAI_LOCAL_RANKS", "0"))
    except ValueError:
        local = 0
    if local <= 0:
        return max(1, world_size or 1)
    return min(local, world_size) if world_size else local


def parse_memory_budget_bytes(value: str | float) -> int:
    """Parse an explicit memory budget to bytes: a bare number is GiB, a string carries its own unit.

    KonfAI reports RAM in GiB throughout, so an unadorned ``24`` reads as ``24 GiB``: whether it
    arrives as a number or, through the YAML binding, as the string ``"24"``. A string may name its
    unit: decimal ``GB``/``MB`` (10^n) or binary ``GiB``/``MiB`` (2^n), case-insensitive, optional
    space (``"24GB"``, ``"32 GiB"``, ``"512mb"``); ``"b"`` means bytes. ``"auto"`` is resolved by the
    caller, not here.
    """
    if not isinstance(value, str):
        if float(value) <= 0:
            raise ConfigError(
                f"memory_budget: {value!r} must be a positive size.",
                "Use a positive number in GiB (e.g. 24), a unit string ('24GB'), 'auto', or None.",
            )
        return int(float(value) * 2**30)
    match = re.fullmatch(r"\s*(?P<number>[0-9]*\.?[0-9]+)\s*(?P<unit>[a-z]*)\s*", value.lower())
    if match is not None and match.group("unit") in _MEMORY_UNIT_BYTES:
        if float(match.group("number")) <= 0:
            raise ConfigError(
                f"memory_budget: '{value}' must be a positive size.",
                "Use a positive number in GiB (e.g. 24), a unit string ('24GB'), 'auto', or None.",
            )
        unit = match.group("unit")
        # A bare numeric string is the YAML face of a bare number: GiB, not bytes.
        factor = 2**30 if unit == "" else _MEMORY_UNIT_BYTES[unit]
        return int(float(match.group("number")) * factor)
    raise ConfigError(
        f"memory_budget: '{value}' is not a valid memory size.",
        "Use a number in GiB (e.g. 24), a unit string ('24GB', '32GiB', '512MB'), 'auto', or None "
        "(the default): which means 'auto': size from the detected memory.",
    )


def resolve_memory_budget(memory_budget: str | float | None) -> MemoryBudget:
    """The configured ``memory_budget`` as an object that knows its own scope.

    ``None``/``"auto"`` offers ``AUTO_MEMORY_SAFETY_FRACTION`` of the node's allocatable memory: a
    NODE budget, which ranks sharing the node split; an explicit budget is the caller's own figure,
    per rank as declared.
    """
    if memory_budget is None or (isinstance(memory_budget, str) and memory_budget.strip().lower() == "auto"):
        from konfai.utils.runtime import available_memory_bytes  # lazy: runtime imports torch

        node_bytes, source = available_memory_bytes()
        return MemoryBudget(
            node_bytes * AUTO_MEMORY_SAFETY_FRACTION,
            f"auto: {format_bytes(node_bytes)} {source} x {AUTO_MEMORY_SAFETY_FRACTION:.0%}",
            shared_across_ranks=True,
        )
    return MemoryBudget(
        float(parse_memory_budget_bytes(memory_budget)), f"{memory_budget!r}", shared_across_ranks=False
    )
