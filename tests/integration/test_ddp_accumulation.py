# Copyright (c) 2026 Valentin Boussot
# SPDX-License-Identifier: Apache-2.0

"""Real Gloo reductions and updates across single and nested accumulation windows."""

import json
import os
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from konfai.network.network import Network
from konfai.trainer import _ddp_kwargs
from torch.nn.parallel import DistributedDataParallel as DDP

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="CPU Gloo is unavailable"),
]


class _Leaf(Network):
    def __init__(self, index: int, cadence: int) -> None:
        super().__init__(in_channels=1, dim=2, nb_batch_per_step=cadence)
        self.set_name(f"Leaf_{index}")
        self.add_module("Linear", torch.nn.Linear(1, 1, bias=False))
        with torch.no_grad():
            self["Linear"].weight.fill_(1.0)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.1)
        self.scaler = torch.amp.GradScaler("cpu", enabled=False)


class _Graph(Network):
    def __init__(self, cadences: tuple[int, ...]) -> None:
        super().__init__(in_channels=1, dim=2)
        for index, cadence in enumerate(cadences):
            self.add_module(f"Leaf_{index}", _Leaf(index, cadence))

    def forward(self, value: torch.Tensor) -> list[torch.Tensor]:
        return [leaf["Linear"](value) for leaf in self.children()]


def _count_and_average(state: dict[str, int], bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
    state["calls"] += 1
    return dist.all_reduce(bucket.buffer(), async_op=True).get_future().then(lambda result: result.value()[0] / 2)


def _run_rank(rank: int, root: str, cadences: tuple[int, ...]) -> None:
    torch.set_num_threads(1)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    path = Path(root)
    dist.init_process_group(
        "gloo", init_method=(path / "rendezvous").as_uri(), rank=rank, world_size=2, timeout=timedelta(seconds=30)
    )
    try:
        model = _Graph(cadences)
        ddp = DDP(model, **_ddp_kwargs(model, local_rank=rank, size=1))
        state = {"calls": 0}
        ddp.register_comm_hook(state, _count_and_average)
        observed: list[list[float]] = []
        for _ in range(6):
            with model.accumulation_sync(ddp):
                outputs = ddp(torch.tensor([[float(rank + 1)]]))
                for leaf, output in zip(model.children(), outputs, strict=True):
                    # The loss surface is minimal; production Network.backward owns the scaler,
                    # counters, optimizer boundaries and gradient reset.
                    leaf.measure = SimpleNamespace(outputs_criterions={}, get_loss=lambda y=output: [y.square().mean()])
                model.backward(ddp)
            observed.append([float(leaf["Linear"].weight.detach().item()) for leaf in model.children()])
        (path / f"rank-{rank}.json").write_text(json.dumps({"weights": observed, "reductions": state["calls"]}))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("cadences, reductions", [((1,), 6), ((2,), 3), ((2, 3), 4)])
def test_ddp_accumulation_matches_global_batch_updates_and_reduces_at_boundaries(
    tmp_path: Path, cadences: tuple[int, ...], reductions: int
) -> None:
    context = mp.spawn(_run_rank, args=(str(tmp_path), cadences), nprocs=2, join=False)
    deadline = time.monotonic() + 60
    try:
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                pytest.fail("Gloo accumulation workers did not complete within 60 seconds")
    finally:
        # A reducer failure can also block process-group destruction; a failed test must reap
        # both ranks instead of leaving the suite waiting indefinitely.
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)

    # The global batch is x=[1,2], loss=mean((w*x)^2), hence dloss/dw=5*w.
    # SGD(lr=.1) halves w at each optimizer step regardless of the accumulation length.
    expected = [[0.5 ** ((iteration + 1) // cadence) for cadence in cadences] for iteration in range(6)]
    for rank in range(2):
        result: dict[str, Any] = json.loads((tmp_path / f"rank-{rank}.json").read_text())
        torch.testing.assert_close(torch.tensor(result["weights"]), torch.tensor(expected))
        assert result["reductions"] == reductions
