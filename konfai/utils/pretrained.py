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

"""Load pretrained weights from an external architecture into a KonfAI graph.

A KonfAI catalog model (``konfai/models/yaml``) is often weight-exact to a reference implementation
(a MONAI or torchvision model) but uses different module names, so its ``state_dict`` keys do not
match the external checkpoint. ``transfer_weights_by_execution_order`` bridges the two by pairing
parametric leaf modules in **forward-execution order** instead of by name: both models are run once
with hooks that record the order their weighted leaves execute, and the ordered lists are copied
position-by-position with a shape check. This is the mechanism that lets a MONAI-trained checkpoint
drive a KonfAI graph, so the network gains KonfAI's named-output supervision (deep supervision,
feature-level losses) on top of the reference's pretrained weights.

It is deliberately strict: a mismatched leaf count or shape raises, so a non-equivalent pair fails
loudly instead of silently mis-loading half a network.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from konfai.utils.errors import ConfigError


def _parametric_leaves_in_execution_order(model: torch.nn.Module, run: Callable[[], object]) -> list[torch.nn.Module]:
    """Return the model's weighted leaf modules in the order their forward runs, via forward hooks.

    Leaf = a module with no children that owns parameters directly (Conv, Linear, norm, ...). Ordering
    by execution rather than by ``state_dict`` key is what makes the pairing robust to a reference that
    registers its norm/activation before its conv (pre-activation), where key order != run order.
    """
    order: list[torch.nn.Module] = []
    seen: set[int] = set()

    def hook(module: torch.nn.Module, _inputs: object, _output: object) -> None:
        if id(module) not in seen:
            seen.add(id(module))
            order.append(module)

    handles = []
    for module in model.modules():
        is_leaf = next(module.children(), None) is None
        if is_leaf and next(module.parameters(recurse=False), None) is not None:
            handles.append(module.register_forward_hook(hook))
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            run()
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    return order


def transfer_weights_by_execution_order(
    target: torch.nn.Module,
    source: torch.nn.Module,
    *,
    target_forward: Callable[[], object],
    source_forward: Callable[[], object],
) -> int:
    """Copy every parameter and buffer of ``source`` into ``target`` by execution-order leaf pairing.

    ``target`` is the KonfAI model receiving the weights; ``source`` is the external reference whose
    pretrained checkpoint you want to reuse. ``target_forward`` / ``source_forward`` are zero-argument
    callables that each run one forward pass (e.g. ``lambda: list(net.named_forward(x))`` for a KonfAI
    Network, ``lambda: monai_model(x)`` for the reference). Returns the number of leaves transferred.

    Raises ``ConfigError`` when the two graphs are not weight-exact -- a different number of weighted
    leaves, or a paired leaf whose local ``state_dict`` (its own weight/bias/buffers) does not match in
    keys or shapes. That is intentional: silently loading a mismatched network is worse than failing.
    """
    target_leaves = _parametric_leaves_in_execution_order(target, target_forward)
    source_leaves = _parametric_leaves_in_execution_order(source, source_forward)
    if len(target_leaves) != len(source_leaves):
        raise ConfigError(
            f"Cannot transfer weights: the models have a different number of weighted leaves "
            f"(target={len(target_leaves)}, source={len(source_leaves)}).",
            "Weight transfer requires a weight-exact architecture; check that hyperparameters "
            "(channels/depth/dim) match the reference and that both forwards ran on the same input.",
        )
    for index, (target_leaf, source_leaf) in enumerate(zip(target_leaves, source_leaves, strict=True)):
        target_state = target_leaf.state_dict()
        source_state = source_leaf.state_dict()
        target_shapes = {key: tuple(value.shape) for key, value in target_state.items()}
        source_shapes = {key: tuple(value.shape) for key, value in source_state.items()}
        if target_shapes != source_shapes:
            raise ConfigError(
                f"Cannot transfer weights: leaf #{index} does not match "
                f"({type(target_leaf).__name__} {target_shapes} vs {type(source_leaf).__name__} {source_shapes}).",
                "The two architectures diverge at this layer; they are not weight-exact.",
            )
        target_leaf.load_state_dict(source_state)
    return len(target_leaves)
