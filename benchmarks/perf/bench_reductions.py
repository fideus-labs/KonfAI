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

"""What a case reduction holds against what its plan prices, per operator, dtype and member count.

    python benchmarks/perf/bench_reductions.py [--force] [--quick]

The plan sizes a reduction's regions from each operator's ``working_multiple_for`` (the buffers it
allocates over the members it is handed, in float32 member regions) plus the output region. This
bench folds one region on the GPU, where the caching allocator's peak is exact, and states as a fact
that the peak the fold allocated stays within that price. An incremental operator holds one member
at a time, the others the whole cohort. Without a GPU the bench records nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from facts import fact
from harness import fingerprint, machine_gate, write_result
from konfai.data.case_reduction import CASE_ELEMENT_BYTES
from konfai.data.reduction import Concat, Mean, Median, Std, Vote

#: The operators, and the dtypes each folds in a shipped config (Vote folds label maps).
OPERATORS = (
    (Mean, (torch.float32, torch.int16)),
    (Std, (torch.float32,)),
    (Median, (torch.float32, torch.int16)),
    (Vote, (torch.uint8, torch.int16)),
    (Concat, (torch.float32,)),
)


def _member(index: int, dtype: torch.dtype, shape: tuple[int, ...]) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(index)
    if dtype.is_floating_point:
        return torch.rand(shape, generator=generator, device="cuda", dtype=dtype)
    return torch.randint(0, 5, shape, generator=generator, device="cuda", dtype=dtype)


def held_bytes(operator: type, dtype: torch.dtype, count: int, shape: tuple[int, ...]) -> tuple[int, int]:
    """What folding ``count`` members of ``shape`` allocated at its peak, and what the plan prices for
    it: the members it holds, the operator's buffers over them and the output region."""
    reduction = operator()
    elements = int(np.prod(shape))
    member = elements * CASE_ELEMENT_BYTES
    output = elements // shape[2] * reduction.output_channels(shape[2], count) * CASE_ELEMENT_BYTES
    held_members = 1 if reduction.incremental else count
    priced = int(held_members * member * (1 + reduction.working_multiple_for(count)) + output)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    start = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    if reduction.incremental:
        # One member resident at a time, as the plan's running accumulator reads them.
        reduction.start()
        for index in range(count):
            member_tensor = _member(index, dtype, shape)
            reduction.accumulate(member_tensor)
            del member_tensor
        result = reduction.finalize()
    else:
        members = [_member(index, dtype, shape) for index in range(count)]
        result = reduction(members)
        del members
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - start
    del result
    return peak, priced


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    gate = machine_gate(force=args.force)
    fp = fingerprint()
    result: dict[str, object] = {"gate_warnings": gate.warnings}
    if not torch.cuda.is_available():
        result["headline"] = "no GPU: nothing measured"
        print(f"[perf] {result['headline']}")
        print(f"[perf] written {write_result('reductions', result, fp=fp)}")
        return
    shape = (1, 1, 1, 16, 64, 64) if args.quick else (1, 1, 1, 32, 128, 128)
    counts = (2, 3, 5) if args.quick else (2, 3, 4, 5, 6, 10)
    measured: dict[str, dict[str, int]] = {}
    facts = []
    for operator, dtypes in OPERATORS:
        for dtype in dtypes:
            for count in counts:
                peak, priced = held_bytes(operator, dtype, count, shape)
                name = f"{operator.__name__}_{str(dtype).removeprefix('torch.')}_{count}"
                measured[name] = {"peak_bytes": peak, "priced_bytes": priced}
                facts.append(fact(f"{name}_peak_within_price", peak, priced, "<="))
    result["region_shape"] = list(shape)
    result["measured"] = measured
    result["facts"] = facts
    worst = max(measured.items(), key=lambda item: item[1]["peak_bytes"] / item[1]["priced_bytes"])
    result["headline"] = (
        f"{len(facts)} folds; the closest to its price: {worst[0]} at "
        f"{worst[1]['peak_bytes'] / worst[1]['priced_bytes']:.2f} of it"
    )
    path = write_result("reductions", result, fp=fp)
    print(json.dumps(measured, indent=2))
    print(f"[perf] {result['headline']}")
    print(f"[perf] written {path}")


if __name__ == "__main__":
    main()
