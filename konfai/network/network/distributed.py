# Copyright (c) 2026 Valentin Boussot
# SPDX-License-Identifier: Apache-2.0

"""Synchronize gradients produced inside the patch walk, after all local backwards finish."""

import math
from collections.abc import Iterator
from contextlib import contextmanager

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


@contextmanager
def local_backward(model: DistributedDataParallel) -> Iterator[None]:
    """Keep patch backwards local while preserving DDP's buffer broadcasts, including validation."""
    model.require_forward_param_sync = True
    try:
        with model.no_sync():
            yield
    finally:
        model.require_forward_param_sync = True


def _regions(gradient: torch.Tensor, limit: int) -> Iterator[torch.Tensor]:
    """Views bounded by ``limit`` elements, including channels-last and oversized parameters."""
    if gradient.numel() <= limit:
        yield gradient
        return
    width = max(1, limit // math.prod(gradient.shape[1:]))
    for region in gradient.split(width):
        if region.numel() > limit:
            yield from _regions(region[0], limit)
        else:
            yield region


def _average_bucket(bucket: list[torch.Tensor], group: dist.ProcessGroup, world_size: int) -> None:
    sizes = [part.numel() for part in bucket]
    flat = torch.empty(sum(sizes), dtype=bucket[0].dtype, device=bucket[0].device)
    views = [part.view_as(gradient) for part, gradient in zip(flat.split(sizes), bucket, strict=True)]
    # Group small copies into CUDA launches instead of launching once per parameter.
    torch._foreach_copy_(views, bucket)
    # Divide before summing, just like DDP: finite fp16 gradients must not overflow
    # merely because several ranks supplied the same large gradient.
    flat.div_(world_size)
    dist.all_reduce(flat, group=group)
    torch._foreach_copy_(bucket, views)
    bucket.clear()


@torch.no_grad()
def average_gradients(
    optimizer: torch.optim.Optimizer,
    group: dist.ProcessGroup,
    bucket_bytes: int,
    scaler: torch.amp.GradScaler | None = None,
) -> None:
    """Average this optimizer's scaled gradients once, immediately before its step.

    Presence is agreed before collecting tensors: a rank missing a gradient contributes zero,
    while a parameter unused everywhere keeps ``grad=None`` (no momentum or weight decay step).
    Dense gradient buffers never exceed DDP's existing bucket budget. Sparse COO gradients
    use the backend's sparse all-reduce, as DDP does, without densifying an embedding table.
    """
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return
    parameters = dict.fromkeys(parameter for entry in optimizer.param_groups for parameter in entry["params"])
    by_device: dict[torch.device, list[torch.Tensor]] = {}
    for parameter in parameters:
        by_device.setdefault(parameter.device, []).append(parameter)
    for device, members in by_device.items():
        kinds = []
        for parameter in members:
            grad = parameter.grad
            # 0: unused; 1: dense; >=2: COO's sparse rank plus one.
            kinds.append(0 if grad is None else grad.sparse_dim() + 1 if grad.is_sparse else 1)
        present = torch.tensor(kinds, dtype=torch.int64, device=device)
        dist.all_reduce(present, op=dist.ReduceOp.MAX, group=group)
        agreed = present.tolist()
        if scaler is not None and scaler.is_enabled() and not any(kinds) and any(agreed):
            # AMP initializes lazily on scale(loss). This rank may receive its first gradients
            # without a local loss; initialize through the public API before unscaling them.
            scaler.scale(torch.zeros((), device=device))
        dense: dict[torch.dtype, list[torch.Tensor]] = {}
        for parameter, kind in zip(members, agreed, strict=True):
            if kind == 0:
                continue
            grad = parameter.grad
            if kind > 1:
                sparse_dim = kind - 1
                if grad is None:
                    grad = torch.sparse_coo_tensor(
                        torch.empty((sparse_dim, 0), dtype=torch.int64, device=device),
                        torch.empty((0, *parameter.shape[sparse_dim:]), dtype=parameter.dtype, device=device),
                        parameter.shape,
                    )
                elif not grad.is_sparse:
                    grad = grad.to_sparse(sparse_dim)
                grad = grad.coalesce()
                grad.div_(world_size)
                dist.all_reduce(grad, group=group)
                parameter.grad = grad
            else:
                if grad is None:
                    grad = parameter.grad = torch.zeros_like(parameter)
                dense.setdefault(grad.dtype, []).append(grad)
        for gradients in dense.values():
            bucket: list[torch.Tensor] = []

            limit = max(1, bucket_bytes // gradients[0].element_size())
            size = 0
            for gradient in gradients:
                for part in _regions(gradient, limit):
                    if size and size + part.numel() > limit:
                        _average_bucket(bucket, group, world_size)
                        size = 0
                    bucket.append(part)
                    size += part.numel()
            if size:
                _average_bucket(bucket, group, world_size)
