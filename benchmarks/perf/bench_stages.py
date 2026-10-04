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

"""What one stage costs on a whole region: its wall, what it holds beside its input, the voxels it returns.

    python benchmarks/perf/bench_stages.py [--repeats 5] [--force] [--quick] [--only normalize,walk]

A stage is called as a loaded case or a streamed region calls it, on a seeded CT-sized volume, in a
fresh process pinned to one thread: a loader worker's share, and the regime where an allocation
weighs most. Three results per stage:

- its wall, the median of the repeats after a warm-up;
- what it held at its peak over the tensor it was handed, from the kernel's resident high-water
  mark reset before the call (Linux; not measured elsewhere), in float32 volumes of that tensor. A
  volume this size is mapped on every allocation, so the mark counts the volumes alive together;
- for a value stage, the voxels that differ from its formula written plainly, out of place.

The facts are the last two: a hold within its bound and zero differing voxels need no baseline and
hold on any machine. The times compare to a baseline through ``compare.py``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from facts import fact
from harness import fingerprint, machine_gate, write_result
from konfai.data.geometry import Grid
from konfai.data.sampling import source_index_rows
from konfai.data.transform import Clip, Normalize, Resample, Standardize, Transform, UnNormalize
from konfai.utils.dataset import Attribute

CASE = "CASE"
#: ``[1, ROWS, SIDE, SIDE]``: 128 MiB as float32, 64 MiB on a quick pass.
SIDE = 512
ROWS = 128
QUICK_ROWS = 64
#: Ten degrees about z: a map that no longer factorises per axis.
_OBLIQUE = np.array(
    [[np.cos(np.pi / 18), -np.sin(np.pi / 18), 0.0], [np.sin(np.pi / 18), np.cos(np.pi / 18), 0.0], [0.0, 0.0, 1.0]]
)
#: What a hold may exceed its bound's measure by: allocator rounding, a stage's own constants.
_SLACK = 0.25


def _normalized(tensor: torch.Tensor, attribute: Attribute) -> torch.Tensor:
    low, high = float(tensor.min()), float(tensor.max())
    return (1 - -1) * (tensor - low) / (high - low) + -1


def _standardized(tensor: torch.Tensor, attribute: Attribute) -> torch.Tensor:
    return (tensor - attribute.get_tensor("Mean")) / attribute.get_tensor("Std")


def _unnormalized(tensor: torch.Tensor, attribute: Attribute) -> torch.Tensor:
    return (tensor + 1) / 2 * (3071 - -1024) + -1024


def _clipped(tensor: torch.Tensor, attribute: Attribute) -> torch.Tensor:
    out = tensor.clone()
    out.masked_fill_(out.float() < -200, -200)
    return out.masked_fill_(out.float() > 1500, 1500)


@dataclass(frozen=True)
class Scenario:
    """One stage on one stored dtype. ``held`` is the hold measured when the scenario was written, in
    float32 volumes of the tensor handed in: the fact allows :data:`_SLACK` over it."""

    stage: Callable[[], Transform]
    dtype: torch.dtype
    held: float
    plain: Callable[[torch.Tensor, Attribute], torch.Tensor] | None = None
    oblique: bool = False
    #: The stage writes into the tensor it is handed: each call takes a copy made outside the clock.
    in_place: bool = False


SCENARIOS: dict[str, Scenario] = {
    "normalize": Scenario(Normalize, torch.float32, 1.0, _normalized),
    "normalize_int16": Scenario(Normalize, torch.int16, 1.0, _normalized),
    "standardize": Scenario(Standardize, torch.float32, 1.0, _standardized),
    "standardize_int16": Scenario(Standardize, torch.int16, 2.0, _standardized),
    "unnormalize": Scenario(UnNormalize, torch.float32, 2.0, _unnormalized),
    "clip": Scenario(lambda: Clip(min_value=-200, max_value=1500), torch.float32, 0.0, _clipped, in_place=True),
    "clip_int16": Scenario(lambda: Clip(min_value=-200, max_value=1500), torch.int16, 0.0, _clipped, in_place=True),
    "resample_linear": Scenario(lambda: Resample(spacing=[1.5, 1.5, 1.5]), torch.int16, 2.35),
    "resample_cubic": Scenario(lambda: Resample(spacing=[1.5, 1.5, 1.5], interpolation="cubic"), torch.int16, 3.0),
    "resample_oblique": Scenario(lambda: Resample(spacing=[1.5, 1.5, 1.5]), torch.int16, 1.0, oblique=True),
}
#: The coordinate walk of a map that does not factorise, over the rows of one slab. Its hold is in
#: volumes of the coordinates it returns.
WALK = "walk"
WALK_HELD = 5.0


def _volume(dtype: torch.dtype, rows: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    volume = torch.randn((1, rows, SIDE, SIDE), generator=generator).mul_(300.0).add_(40.0).clamp_(-1024, 3071)
    return volume.to(dtype)


def _attribute(oblique: bool) -> Attribute:
    attribute = Attribute()
    attribute["Origin"] = np.zeros(3)
    attribute["Spacing"] = np.ones(3)
    attribute["Direction"] = (_OBLIQUE if oblique else np.eye(3)).reshape(-1)
    return attribute


def _resident(field: str) -> int:
    with open("/proc/self/status") as status:
        for line in status:
            if line.startswith(field):
                return int(line.split()[1]) * 1024
    raise OSError(f"no {field} in /proc/self/status")


def _held(call: Callable[[], torch.Tensor]) -> tuple[int | None, torch.Tensor]:
    """What ``call`` held at its peak over what was resident before it, and what it returned."""
    try:
        Path("/proc/self/clear_refs").write_text("5")  # resets the resident high-water mark
        before = _resident("VmRSS")
    except OSError:
        return None, call()
    out = call()
    return max(0, _resident("VmHWM") - before), out


def _measure(
    call: Callable[[], torch.Tensor], repeats: int, before: Callable[[], None] = lambda: None
) -> tuple[list[float], int | None, torch.Tensor]:
    """The walls of ``repeats`` calls after a warm-up one (imports, lazy pools, the allocator's first
    touch), then the hold of one more. ``before`` runs ahead of each call, outside the clock."""
    walls = []
    for index in range(repeats + 1):
        before()
        start = time.perf_counter()
        call()
        if index:
            walls.append((time.perf_counter() - start) * 1e3)
    before()
    held, out = _held(call)
    return walls, held, out


def _worker(name: str, rows: int, repeats: int) -> None:
    """One scenario in this fresh process; prints one JSON line."""
    torch.set_num_threads(1)
    try:
        import SimpleITK as sitk

        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    except ImportError:
        pass
    measured: dict[str, Any] = {}
    if name == WALK:
        source = Grid((300, SIDE, SIDE), np.zeros(3), np.ones(3), _OBLIQUE)
        target = Grid((200, 341, 341), np.array([0.25, 0.25, 0.25]), np.full(3, 1.5), np.eye(3))
        walls, held, out = _measure(
            lambda: source_index_rows(target, source, [], torch.device("cpu"), 0, rows // 2), repeats
        )
        volume_bytes = out.numel() * out.element_size()
    else:
        scenario = SCENARIOS[name]
        tensor = _volume(scenario.dtype, rows)
        volume_bytes = tensor.numel() * 4
        stage = scenario.stage()
        attribute = Attribute()
        handed = tensor

        def hand() -> None:
            nonlocal attribute, handed
            attribute = _attribute(scenario.oblique)
            if scenario.in_place:
                handed = tensor.clone()

        walls, held, out = _measure(lambda: stage(CASE, handed, attribute), repeats, hand)
        if scenario.plain is not None:
            measured["differing_voxels"] = int((out != scenario.plain(tensor, attribute)).sum())
    measured.update(walls_ms=walls, held_bytes=held, volume_bytes=volume_bytes)
    print(json.dumps(measured))


def in_fresh_process(name: str, rows: int, repeats: int) -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--worker", name, str(rows), str(repeats)],
        capture_output=True,
        text=True,
        check=False,
    )
    lines = [line for line in completed.stdout.splitlines() if line.startswith("{")]
    if completed.returncode != 0 or not lines:
        raise SystemExit(
            f"[perf] stage '{name}' failed (rc {completed.returncode}):\n{completed.stdout[-1500:]}\n"
            f"{completed.stderr[-3000:]}"
        )
    return json.loads(lines[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--only", default="", help="comma-separated subset of the scenarios")
    parser.add_argument("--worker", nargs=3, metavar=("NAME", "ROWS", "REPEATS"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        _worker(args.worker[0], int(args.worker[1]), int(args.worker[2]))
        return
    gate = machine_gate(force=args.force)
    fp = fingerprint()
    rows = QUICK_ROWS if args.quick else ROWS
    repeats = 1 if args.quick else args.repeats
    bounds = {**{name: scenario.held for name, scenario in SCENARIOS.items()}, WALK: WALK_HELD}
    names = [name for name in bounds if not args.only or name in args.only.split(",")]
    if not names:
        parser.error(f"no scenario among: {', '.join(bounds)}")

    runs: dict[str, dict[str, Any]] = {}
    metrics: dict[str, float] = {}
    facts = []
    for name in names:
        run = in_fresh_process(name, rows, repeats)
        runs[name] = run
        metrics[f"{name}_ms"] = round(statistics.median(run["walls_ms"]), 2)
        held = "not measured"
        if run["held_bytes"] is None and sys.platform == "linux":
            raise SystemExit(f"[perf] stage '{name}' has no held-memory measurement; cannot verify its bound")
        if run["held_bytes"] is not None:
            volumes = round(run["held_bytes"] / run["volume_bytes"], 3)
            facts.append(fact(f"{name}_held_volumes", volumes, bounds[name] + _SLACK, "<="))
            held = f"{volumes:.2f} volume(s)"
        if "differing_voxels" in run:
            facts.append(fact(f"{name}_differing_voxels", run["differing_voxels"], 0))
        print(f"[perf] {name}: {metrics[f'{name}_ms']} ms, held {held}")

    slowest = max(names, key=lambda name: metrics[f"{name}_ms"])
    result = {
        "gate_warnings": gate.warnings,
        "quick": args.quick,
        "volume_shape": [1, rows, SIDE, SIDE],
        "threads": 1,
        "repeats": repeats,
        "runs": runs,
        "metrics": metrics,
        "facts": facts,
        "headline": f"{len(names)} stage(s) at one thread on [1, {rows}, {SIDE}, {SIDE}]; "
        + ", ".join(f"{name} {metrics[f'{name}_ms']:g} ms" for name in names[:3])
        + f"; the slowest: {slowest} {metrics[f'{slowest}_ms']:g} ms",
    }
    path = write_result("stages", result, fp=fp)
    print(json.dumps(metrics, indent=2))
    print(f"[perf] {result['headline']}")
    print(f"[perf] written {path}")


if __name__ == "__main__":
    main()
