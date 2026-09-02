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

import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch

from konfai.utils.config import config
from konfai.utils.errors import ConfigError

if TYPE_CHECKING:
    from konfai.network.network.network import Network


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
        # A leaf may own only buffers (a non-affine BatchNorm/InstanceNorm with running stats), whose
        # tensors _untraced_tensors would otherwise flag as uncovered and wrongly refuse the transfer.
        if is_leaf and (
            next(module.parameters(recurse=False), None) is not None
            or next(module.buffers(recurse=False), None) is not None
        ):
            handles.append(module.register_forward_hook(hook))
    # Snapshot per-module modes: model.train(root_mode) would force every descendant into the
    # root's mode and lose frozen/eval-only submodules.
    training_states = {module: module.training for module in model.modules()}
    model.eval()
    try:
        with torch.no_grad():
            run()
    finally:
        for handle in handles:
            handle.remove()
        for module, training in training_states.items():
            module.training = training
    return order


def _untraced_tensors(model: torch.nn.Module, leaves: list[torch.nn.Module]) -> list[str]:
    """Return the names of the model's parameters/buffers that no traced leaf owns.

    Two shapes escape the leaf trace: a module that owns parameters *and* has children (torch's
    ``MultiheadAttention`` holds ``in_proj_weight`` beside its ``out_proj`` child) is never a leaf, and
    a leaf the forward does not reach is never hooked. Their tensors would be skipped, so the caller
    must refuse the transfer instead of reporting success on a partial load.
    """
    covered: set[int] = set()
    for leaf in leaves:
        covered.update(id(tensor) for tensor in leaf.parameters())
        covered.update(id(tensor) for tensor in leaf.buffers())
    return [name for name, tensor in model.named_parameters() if id(tensor) not in covered] + [
        name for name, tensor in model.named_buffers() if id(tensor) not in covered
    ]


def transfer_weights_by_execution_order(
    target: torch.nn.Module,
    source: torch.nn.Module,
    *,
    target_forward: Callable[[], object],
    source_forward: Callable[[], object],
) -> int:
    """Fill every parameter and buffer of ``target`` from ``source`` by execution-order leaf pairing.

    ``target`` is the KonfAI model receiving the weights; ``source`` is the external reference whose
    pretrained checkpoint you want to reuse. ``target_forward`` / ``source_forward`` are zero-argument
    callables that each run one forward pass (e.g. ``lambda: list(net.named_forward(x))`` for a KonfAI
    Network, ``lambda: monai_model(x)`` for the reference). Returns the number of leaves transferred.

    Every ``target`` tensor is written or the call raises; ``source`` tensors its forward does not reach
    (an unused deep-supervision head) have no counterpart to feed and are ignored.

    Raises ``ConfigError`` when the two graphs are not weight-exact: a target tensor no traced leaf
    owns, a different number of weighted leaves, or a paired leaf whose local ``state_dict`` (its own
    weight/bias/buffers) does not match in keys or shapes. That is intentional: silently loading a
    mismatched network is worse than failing.
    """
    target_leaves = _parametric_leaves_in_execution_order(target, target_forward)
    source_leaves = _parametric_leaves_in_execution_order(source, source_forward)
    # Only the target is required to be fully covered: an untraced target tensor would silently keep its
    # random init. An untraced *source* tensor is a branch this configuration does not run (nnU-Net's
    # unused deep-supervision heads) and has no target counterpart to feed, so it is correctly ignored.
    untraced = _untraced_tensors(target, target_leaves)
    if untraced:
        raise ConfigError(
            f"Cannot transfer weights: {len(untraced)} target tensor(s) are owned by no traced leaf and "
            f"would keep their random init ({', '.join(untraced[:5])}{', ...' if len(untraced) > 5 else ''}).",
            "Execution-order pairing only copies weighted leaf modules the forward reaches. A tensor held "
            "directly by a parent module (torch.nn.MultiheadAttention holds in_proj_weight beside its "
            "out_proj child) or by a submodule this forward skips cannot be paired; wrap it in a leaf "
            "module, or transfer those tensors by name.",
        )
    if len(target_leaves) != len(source_leaves):
        raise ConfigError(
            f"Cannot transfer weights: the models have a different number of weighted leaves "
            f"(target={len(target_leaves)}, source={len(source_leaves)}).",
            "Weight transfer requires a weight-exact architecture; check that hyperparameters "
            "(channels/depth/dim) match the reference and that both forwards ran on the same input.",
        )
    # A tensor tied across two target leaves would be written twice by the per-leaf loads below, so the
    # earlier leaf would silently end up with the later leaf's source weights. state_dict(keep_vars=True)
    # yields the live parameters AND persistent buffers: exactly what load_state_dict writes, so a tied
    # buffer is caught as well. Refuse rather than mis-load.
    seen_tensors: dict[int, str] = {}
    for leaf in target_leaves:
        for tensor_name, tensor in leaf.state_dict(keep_vars=True).items():
            if id(tensor) in seen_tensors:
                raise ConfigError(
                    "Cannot transfer weights: the target ties a tensor across two weighted leaves "
                    f"('{seen_tensors[id(tensor)]}' and '{tensor_name}').",
                    "Execution-order pairing loads each leaf's source into the shared tensor in turn, so the "
                    "earlier leaf would silently keep the later one's weights. Transfer such a model by name.",
                )
            seen_tensors[id(tensor)] = tensor_name
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


@config("pretrained_from")
class PretrainedFrom:
    """The ``Model.pretrained_from`` config block: seed a model from an external reference checkpoint.

    ``builder`` names the reference class (``monai.networks.nets:UNet``), ``args`` its constructor
    arguments, and ``checkpoint`` its trained weights: a raw ``state_dict`` file, or a checkpoint
    dict holding one under ``state_dict``. When a fresh TRAIN initialises the model,
    :func:`transfer_weights_by_execution_order` fills every tensor from the reference or raises; a
    checkpoint load (RESUME, PREDICTION) carries its own weights and is never overridden. The
    transfer runs both forwards on a synthetic input shaped from the model's own channels and
    spatial rank; ``input_shape`` overrides its spatial extent when the derived one (the model's
    patch size, else its downsampling multiple) does not fit the graph.
    """

    def __init__(
        self,
        checkpoint: str = "",
        builder: str = "",
        args: dict[str, Any] | None = None,
        input_shape: list[int] | None = None,
    ) -> None:
        self.checkpoint = checkpoint
        self.builder = builder
        self.args = args
        self.input_shape = input_shape

    def _reference(self) -> torch.nn.Module:
        if not self.builder or not self.checkpoint:
            raise ConfigError(
                "Model.pretrained_from requires both 'builder' and 'checkpoint'.",
                "builder names the reference class ('monai.networks.nets:UNet'), checkpoint its weights"
                " (a state_dict .pt, or a checkpoint dict with a 'state_dict' entry).",
            )
        from konfai.utils.utils import get_module

        module, name = get_module(self.builder, "torch.nn")
        reference_class = getattr(module, name, None)
        if reference_class is None:
            raise ConfigError(f"Model.pretrained_from.builder: '{name}' does not exist in '{module.__name__}'.")
        try:
            reference = reference_class(**(self.args or {}))
        except TypeError as error:
            raise ConfigError(f"Model.pretrained_from.args do not construct '{self.builder}'.", str(error)) from error

        from konfai.utils.runtime.environment import safe_torch_load

        try:
            state = safe_torch_load(self.checkpoint, torch.device("cpu"))
        except (OSError, RuntimeError, ValueError) as error:
            raise ConfigError(
                f"Model.pretrained_from.checkpoint: cannot load '{self.checkpoint}'.", str(error)
            ) from error
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        try:
            reference.load_state_dict(state)
        except (RuntimeError, TypeError) as error:
            raise ConfigError(
                f"Model.pretrained_from.checkpoint: '{self.checkpoint}' does not fit the reference '{self.builder}'.",
                str(error),
            ) from error
        return reference.eval()

    def _example_input(self, model: Network) -> torch.Tensor:
        # The transfer only needs shapes and execution order, so any input the graph accepts will do.
        if self.input_shape is not None:
            spatial = [int(size) for size in self.input_shape]
        elif (
            model.patch is not None
            and model.patch.patch_size is not None
            and all(int(size) > 0 for size in model.patch.patch_size)
        ):
            spatial = [int(size) for size in model.patch.patch_size]
        else:
            factors = model.downsampling_factor() or [1] * model.dim
            if len(factors) != model.dim:
                factors = [max(factors)] * model.dim
            # The smallest extent of at least 16 the graph accepts: a multiple of the per-axis factor.
            spatial = [factor * max(1, math.ceil(16 / factor)) for factor in factors]
        return torch.randn(1, model.in_channels, *spatial)

    def seed(self, model: Network) -> int:
        """Fill every tensor of ``model`` from the reference, or raise naming the config key."""
        reference = self._reference()
        inputs = self._example_input(model)
        try:
            return transfer_weights_by_execution_order(
                target=model,
                source=reference,
                target_forward=lambda: list(model.named_forward(inputs)),
                source_forward=lambda: reference(inputs),
            )
        except ConfigError as error:
            raise ConfigError(
                f"Model.pretrained_from: the reference '{self.builder}' cannot seed this model.",
                *(str(message) for message in error.args),
            ) from error
