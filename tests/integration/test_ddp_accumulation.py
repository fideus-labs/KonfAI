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
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from konfai.data.patching import ModelPatch
from konfai.metric.schedulers import Constant
from konfai.network.network import CriterionsAttr, Measure, Network, place_graph
from konfai.network.network.distributed import average_gradients
from konfai.trainer import _ddp_kwargs
from konfai.utils.dataset import Attribute
from konfai.utils.runtime.distributed import pin_gloo_to_loopback
from torch.nn.parallel import DistributedDataParallel as DDP

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="CPU Gloo is unavailable"),
]


class _Leaf(Network):
    def __init__(self, index: int, cadence: int, immediate: bool) -> None:
        super().__init__(
            in_channels=1, dim=2, nb_batch_per_step=cadence, patch=ModelPatch(patch_size=[2, 2]) if immediate else None
        )
        self.set_name(f"Leaf_{index}")
        self.add_module("Out", torch.nn.Conv2d(1, 1, 1, bias=False))
        with torch.no_grad():
            self["Out"].weight.fill_(1.0)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.1)
        self.scaler = torch.amp.GradScaler("cpu", enabled=immediate, init_scale=8.0)


class _Graph(Network):
    def __init__(self, cadences: tuple[int, ...], loss_path: str, immediate: bool) -> None:
        super().__init__(in_channels=1, dim=2)
        for index, cadence in enumerate(cadences):
            self.add_module(f"Leaf_{index}", _Leaf(index, cadence, immediate))

        for index, leaf in enumerate(self.children()):
            # Each optimizer trains an independent branch reading the original input.
            self._modulesArgs[f"Leaf_{index}"].out_branch = [str(index + 1)]
            attr = CriterionsAttr(
                start=2 if loss_path == "scheduled" and index == len(cadences) - 1 else 0,
                stop=0 if loss_path == "stopped" else None,
                accumulation=immediate,
            )
            attr.schedulers = {Constant(): None}
            leaf.measure = Measure(leaf.get_name(), {})
            criteria = {torch.nn.MSELoss(): attr}
            if loss_path == "mixed":
                deferred = CriterionsAttr()
                deferred.schedulers = {Constant(): None}
                criteria[torch.nn.L1Loss()] = deferred
            leaf.measure.outputs_criterions = {f"Leaf_{index}.Out": {"Y": criteria}}
            leaf.measure.init(self, ["X", "Y"])
            leaf.measure.scaler = leaf.scaler
        self.init_outputs_group()
        self.register_buffer("tick", torch.zeros(()))
        self.seen_ticks = []

    def forward(self, *args, **kwargs):
        self.seen_ticks.append(self.tick.item())
        self.tick.add_(dist.get_rank() + 1)
        return super().forward(*args, **kwargs)


def _count_and_average(state: dict[str, int], bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
    state["calls"] += 1
    return dist.all_reduce(bucket.buffer(), async_op=True).get_future().then(lambda result: result.value()[0] / 2)


def _run_rank(
    rank: int, root: str, cadences: tuple[int, ...], loss_path: str, immediate: bool, device_type: str = "cpu"
) -> None:
    torch.set_num_threads(1)
    if device_type == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
        torch.cuda.set_device(rank)
    device = torch.device("cpu" if device_type == "cpu" else f"cuda:{rank}")
    path = Path(root)
    # Both ranks sit on this host: gloo binds loopback instead of resolving the runner's hostname,
    # which the macOS runners serve slowly or not at all.
    with pin_gloo_to_loopback(local=True):
        dist.init_process_group(
            "gloo" if device_type == "cpu" else "nccl",
            init_method=(path / "rendezvous").as_uri(),
            rank=rank,
            world_size=2,
            timeout=timedelta(seconds=30),
        )
    try:
        model = _Graph(cadences, loss_path, immediate)
        if device_type == "cuda":
            model = place_graph(model, rank).to(device)
            for leaf in model.children():
                leaf.scaler = torch.amp.GradScaler("cuda", enabled=immediate, init_scale=8.0)
                leaf.measure.scaler = leaf.scaler
        ddp = DDP(model, **_ddp_kwargs(model, local_rank=rank, size=1))
        state = {"calls": 0}
        ddp.register_comm_hook(state, _count_and_average)
        all_reduce = dist.all_reduce
        gradient_reductions = []

        def record_reduce(tensor, *args, **kwargs):
            if tensor.is_floating_point():
                gradient_reductions.append(tensor.numel())
            return all_reduce(tensor, *args, **kwargs)

        dist.all_reduce = record_reduce
        observed: list[list[float]] = []
        for _ in range(6):
            with model.accumulation_sync(ddp):
                # Four patches on one rank, nine on the other: no collective may run per patch.
                value = torch.full((1, 1, 4 + 2 * rank, 4 + 2 * rank), float(rank + 1), device=device)
                ddp(
                    {
                        "X": SimpleNamespace(tensor=value, attribute=[Attribute()], is_input=True),
                        "Y": SimpleNamespace(tensor=torch.zeros_like(value), attribute=[Attribute()], is_input=False),
                    },
                    _ddp_losses=ddp.require_backward_grad_sync,
                )
                model.backward(ddp)
            observed.append([float(leaf["Out"].weight.detach().item()) for leaf in model.children()])
        training_ticks = model.seen_ticks.copy()
        model.eval()
        with torch.no_grad():
            ddp(
                {
                    "X": SimpleNamespace(tensor=value, attribute=[Attribute()], is_input=True),
                    "Y": SimpleNamespace(tensor=torch.zeros_like(value), attribute=[Attribute()], is_input=False),
                }
            )
        (path / f"rank-{rank}.json").write_text(
            json.dumps(
                {
                    "weights": observed,
                    "reductions": state["calls"],
                    "gradient_reductions": gradient_reductions,
                    "ticks": training_ticks,
                    "validation_tick": model.seen_ticks[-1],
                }
            )
        )
    finally:
        dist.destroy_process_group()


def _spawn(worker, *args) -> None:
    context = mp.spawn(worker, args=args, nprocs=2, join=False)
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


@pytest.mark.parametrize(
    "cadences, reductions, loss_path",
    [
        ((1,), 6, "deferred"),
        ((2,), 3, "deferred"),
        ((2, 3), 4, "deferred"),
        ((1,), 4, "scheduled"),
        ((1, 1), 6, "scheduled"),
        ((2,), 1, "stopped"),
        ((2,), 3, "mixed"),
    ],
)
@pytest.mark.parametrize("immediate", [False, True])
def test_ddp_accumulation_matches_global_batch_updates_and_reduces_at_boundaries(
    tmp_path: Path, cadences: tuple[int, ...], reductions: int, loss_path: str, immediate: bool
) -> None:
    _spawn(_run_rank, str(tmp_path), cadences, loss_path, immediate)

    # The global batch is x=[1,2], loss=mean((w*x)^2), hence dloss/dw=5*w.
    # SGD(lr=.1) halves w at each optimizer step regardless of the accumulation length.
    expected = [[0.5 ** ((iteration + 1) // cadence) for cadence in cadences] for iteration in range(6)]
    if loss_path == "scheduled":
        for iteration, weights in enumerate(expected):
            weights[-1] = 0.5 ** max(0, iteration - 1)
    elif loss_path == "stopped":
        expected = [[1.0], *[[0.75] for _ in range(5)]]
    elif loss_path == "mixed":
        weight = 1.0
        expected = []
        for iteration in range(6):
            if (iteration + 1) % cadences[0] == 0:
                weight = 0.5 * weight - 0.15
            expected.append([weight])
    for rank in range(2):
        result: dict[str, Any] = json.loads((tmp_path / f"rank-{rank}.json").read_text())
        torch.testing.assert_close(torch.tensor(result["weights"]), torch.tensor(expected))
        assert result["validation_tick"] == 6, "validation must see rank zero's final buffers"
        if immediate:
            assert result["reductions"] == 0, "DDP's forward-time reducer must stay disabled"
            expected_reductions = sum(6 // cadence for cadence in cadences)
            if loss_path == "scheduled":
                expected_reductions -= 2
            elif loss_path == "stopped":
                expected_reductions = 1
            assert len(result["gradient_reductions"]) == expected_reductions
            assert result["ticks"] == list(range(6)), "DDP must still broadcast buffers each forward"
        else:
            assert result["reductions"] == reductions


def _edge_rank(rank: int, root: str) -> None:
    torch.set_num_threads(1)
    with pin_gloo_to_loopback(local=True):
        dist.init_process_group(
            "gloo", init_method=(Path(root) / "edges").as_uri(), rank=rank, world_size=2, timeout=timedelta(seconds=30)
        )
    try:
        group = dist.new_group([0, 1])
        parameters = [
            torch.nn.Parameter(torch.ones(2, 3)),
            torch.nn.Parameter(torch.ones(2, 3, 4, 5).to(memory_format=torch.channels_last)),
            torch.nn.Parameter(torch.ones(3, 2, dtype=torch.float64).t()),
            torch.nn.Parameter(torch.ones(4, dtype=torch.float16)),
            torch.nn.Parameter(torch.full((2,), 7.0)),
        ]
        optimizer = torch.optim.SGD(parameters, lr=0.01, momentum=0.9, weight_decay=0.1)
        optimizer.state[parameters[-1]]["momentum_buffer"] = torch.full((2,), 5.0)
        if rank == 1:
            parameters[0].grad = torch.full_like(parameters[0], 4)
        values = torch.arange(120, dtype=torch.float32).reshape(2, 3, 4, 5)
        parameters[1].grad = (values * (rank + 1)).to(memory_format=torch.channels_last)
        if rank == 0:
            parameters[2].grad = torch.full_like(parameters[2], 10)
        parameters[3].grad = torch.full_like(parameters[3], 60000)
        sizes = []
        all_reduce = dist.all_reduce

        def record(tensor, *args, **kwargs):
            assert kwargs["group"] is group
            if tensor.is_floating_point():
                sizes.append(tensor.numel() * tensor.element_size())
            return all_reduce(tensor, *args, **kwargs)

        with patch.object(dist, "all_reduce", record):
            average_gradients(optimizer, group, bucket_bytes=64)
        expected = [
            torch.full_like(parameters[0], 2),
            values * 1.5,
            torch.full_like(parameters[2], 5),
            torch.full_like(parameters[3], 60000),
        ]
        for parameter, value in zip(parameters[:-1], expected, strict=True):
            torch.testing.assert_close(parameter.grad, value)
        assert sizes and max(sizes) <= 64
        assert parameters[-1].grad is None
        optimizer.step()
        torch.testing.assert_close(parameters[-1], torch.full((2,), 7.0))
        torch.testing.assert_close(optimizer.state[parameters[-1]]["momentum_buffer"], torch.full((2,), 5.0))

        # Sparse embeddings retain their layout, even where only the other rank produced a gradient.
        embedding = torch.nn.Embedding(7, 3, sparse=True)
        sparse_optimizer = torch.optim.SGD(embedding.parameters(), lr=0.1)
        for both in (False, True):
            sparse_optimizer.zero_grad(set_to_none=True)
            if rank == 0 or both:
                embedding(torch.tensor([rank + 1, 3])).sum().backward()
            average_gradients(sparse_optimizer, group, bucket_bytes=64)
            assert embedding.weight.grad.is_sparse
            expected_sparse = torch.zeros_like(embedding.weight)
            expected_sparse[[1, 3]] += 0.5
            if both:
                expected_sparse[[2, 3]] += 0.5
            torch.testing.assert_close(embedding.weight.grad.to_dense(), expected_sparse)
            sparse_optimizer.step()

        # A rank may receive its first scaled gradients without ever computing a local loss.
        owner = Network()
        weight = torch.nn.Parameter(torch.ones(()))
        owner.optimizer = torch.optim.SGD([weight], lr=0.1)
        owner.scaler = torch.amp.GradScaler("cpu", init_scale=8.0)
        for iteration in range(2):
            if rank == iteration:
                owner.scaler.scale(weight.square()).backward()
            average_gradients(owner.optimizer, group, bucket_bytes=64, scaler=owner.scaler)
            owner._optimizer_step()
            torch.testing.assert_close(weight, torch.tensor(0.9 ** (iteration + 1)))
            assert owner.scaler.state_dict()["_growth_tracker"] == iteration + 1

        # An overflow on one rank must skip the update and lower the AMP scale on both ranks.
        weight = torch.nn.Parameter(torch.ones(()))
        optimizer = torch.optim.SGD([weight], lr=0.1)
        scaler = torch.amp.GradScaler("cpu", init_scale=8.0)
        for overflow in (True, False):
            optimizer.zero_grad(set_to_none=True)
            loss = (weight * (rank + 1)).square()
            if overflow and rank == 1:
                loss = loss * float("inf")
            scaler.scale(loss).backward()
            average_gradients(optimizer, group, bucket_bytes=64)
            scaler.step(optimizer)
            scaler.update()
            assert scaler.get_scale() == 4.0
            torch.testing.assert_close(weight, torch.tensor(1.0 if overflow else 0.5))

        # Respect the passed group, including an optimizer belonging to a single-rank subgroup.
        singles = [dist.new_group([member]) for member in range(2)]
        weight.grad = torch.tensor(float(rank + 1))
        average_gradients(optimizer, singles[rank], bucket_bytes=64)
        torch.testing.assert_close(weight.grad, torch.tensor(float(rank + 1)))
    finally:
        dist.destroy_process_group()


def test_patch_gradient_reduction_handles_missing_sparse_strided_and_nonfinite_gradients(tmp_path) -> None:
    _spawn(_edge_rank, str(tmp_path))


@pytest.mark.skipif(not dist.is_nccl_available() or torch.cuda.device_count() < 2, reason="requires two CUDA GPUs")
def test_patch_accumulation_on_two_cuda_devices_with_nccl(tmp_path) -> None:
    _spawn(_run_rank, str(tmp_path), (2, 3), "deferred", True, "cuda")
    expected = [[0.5 ** ((iteration + 1) // cadence) for cadence in (2, 3)] for iteration in range(6)]
    for rank in range(2):
        result = json.loads((tmp_path / f"rank-{rank}.json").read_text())
        torch.testing.assert_close(torch.tensor(result["weights"]), torch.tensor(expected))
        assert result["reductions"] == 0
        assert len(result["gradient_reductions"]) == 5
