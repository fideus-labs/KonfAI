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


"""The routed module graph and the Network built on it."""

import inspect
import logging
import os
from abc import ABC
from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import nullcontext
from functools import partial
from typing import Any, Self

import torch
from torch.utils.checkpoint import checkpoint

from konfai import konfai_root
from konfai.data.data_manager import BatchSample
from konfai.data.patching import Accumulator, ModelPatch
from konfai.network.network.base import (
    NetState,
    PatchIndexed,
    accumulator_owner,
    is_accumulated,
    mark_accumulated,
    strip_accumulated,
)
from konfai.network.network.loaders import LRSchedulersLoader, OptimizerLoader, TargetCriterionsLoader
from konfai.network.network.measure import Measure
from konfai.utils.clock import SweepClock
from konfai.utils.dataset import Attribute
from konfai.utils.errors import ConfigError
from konfai.utils.runtime import State, get_device, get_gpu_memory

_log = logging.getLogger(__name__)


def _leaf_spatial_stride(module: torch.nn.Module) -> list[int] | None:
    """Per-axis stride of a leaf that shrinks the grid (a ``Conv``, ``MaxPool`` or ``AvgPool``), else
    ``None``.

    ``ConvTranspose``/``Upsample`` grow the grid, so they read ``None`` and the trace passes their input
    factor straight through. ``AvgPool`` IS a downsampler (a model may pool on its main path) and is
    counted: a residual branch's ``AvgPool`` does not inflate the factor because the branch-aware trace
    merges the parallel main path and shortcut by their per-axis MAX, not their product.
    """
    if isinstance(
        module,
        (
            torch.nn.MaxPool1d,
            torch.nn.MaxPool2d,
            torch.nn.MaxPool3d,
            torch.nn.AvgPool1d,
            torch.nn.AvgPool2d,
            torch.nn.AvgPool3d,
        ),
    ):
        stride = module.stride if module.stride is not None else module.kernel_size
    elif isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)):
        stride = module.stride
    else:
        return None
    name = type(module).__name__
    ndim = 3 if name.endswith("3d") else 2 if name.endswith("2d") else 1
    return [int(s) for s in (stride if isinstance(stride, (tuple, list)) else [stride] * ndim)]


def _flat_downsampling(module: torch.nn.Module, ndim: int) -> list[int]:
    """Product of every strided ``Conv``/``MaxPool`` inside ``module`` (itself included), each
    trailing-aligned to ``ndim``: a leaf of lower dimensionality acts on the LAST axes, so a 2D conv in
    a 3D graph leaves the leading axis untouched.

    This is the factor for an OPAQUE child: a plain torch module whose internal graph the branch trace
    cannot see (a wrapped torchvision/MONAI/smp net added as one ``add_module`` leaf). The flat product
    over-counts a parallel strided shortcut inside it, but over-padding is safe where under-counting
    crashes the model's skip reassembly.
    """
    factor = [1] * ndim
    for leaf in module.modules():
        stride = _leaf_spatial_stride(leaf)
        if stride is None:
            continue
        offset = ndim - len(stride)
        for axis, size in enumerate(stride):
            if axis + offset >= 0:
                factor[axis + offset] *= size
    return factor


def _channels_last(tensor: torch.Tensor) -> torch.Tensor:
    """``tensor`` in the channels-last layout of its rank, untouched when it has none (a vector, a plane)."""
    if tensor.dim() == 4:
        return tensor.contiguous(memory_format=torch.channels_last)
    if tensor.dim() == 5:
        return tensor.contiguous(memory_format=torch.channels_last_3d)
    return tensor


class ModuleArgsDict(torch.nn.Module, ABC):
    """Named module graph container supporting KonfAI branch routing metadata."""

    class ModuleArgs:
        def __init__(
            self,
            in_branch: list[str],
            out_branch: list[str],
            pretrained: bool,
            alias: list[str],
            requires_grad: bool | None,
            training: None | bool,
        ) -> None:
            super().__init__()
            self.alias = alias
            self.pretrained = pretrained
            self.in_branch = in_branch
            self.out_branch = out_branch
            self.in_channels: int | None = None
            self.in_is_channel: bool = True
            self.out_channels: int | None = None
            self.out_is_channel: bool = True
            self.requires_grad = requires_grad
            self.isCheckpoint = False
            self.isGPU_Checkpoint = False
            self.gpu = "cpu"
            self.training = training
            self._isEnd = False

    def __init__(self) -> None:
        super().__init__()
        #: Whether this graph's convolutions run channels-last: set by :meth:`Network.set_channels_last`.
        self._channels_last = False
        self._modulesArgs: dict[str, ModuleArgsDict.ModuleArgs] = {}
        self._training = NetState.TRAIN

    def _addindent(self, s_: str, num_spaces: int):
        s = s_.split("\n")
        if len(s) == 1:
            return s_
        first = s.pop(0)
        s = [(num_spaces * " ") + line for line in s]
        return first + "\n" + "\n".join(s)

    def __repr__(self):
        extra_lines = []

        extra_repr = self.extra_repr()
        if extra_repr:
            extra_lines = extra_repr.split("\n")

        child_lines = []

        def is_simple_branch(x):
            return len(x) > 1 or x[0] != 0

        for key, module in self._modules.items():
            mod_str = repr(module)

            mod_str = self._addindent(mod_str, 2)
            desc = ""
            if is_simple_branch(self._modulesArgs[key].in_branch) or is_simple_branch(
                self._modulesArgs[key].out_branch
            ):
                desc += f", {self._modulesArgs[key].in_branch}->{self._modulesArgs[key].out_branch}"
            if not self._modulesArgs[key].pretrained:
                desc += ", pretrained=False"
            if self._modulesArgs[key].alias:
                desc += f", alias={self._modulesArgs[key].alias}"
            desc += f", in_channels={self._modulesArgs[key].in_channels}"
            desc += f", in_is_channel={self._modulesArgs[key].in_is_channel}"
            desc += f", out_channels={self._modulesArgs[key].out_channels}"
            desc += f", out_is_channel={self._modulesArgs[key].out_is_channel}"
            desc += f", is_end={self._modulesArgs[key]._isEnd}"
            desc += f", isInCheckpoint={self._modulesArgs[key].isCheckpoint}"
            desc += f", isInGPU_Checkpoint={self._modulesArgs[key].isGPU_Checkpoint}"
            desc += f", requires_grad={self._modulesArgs[key].requires_grad}"
            desc += f", device={self._modulesArgs[key].gpu}"

            child_lines.append(f"({key}{desc}) {mod_str}")

        lines = extra_lines + child_lines

        desc = ""
        if lines:
            if len(extra_lines) == 1 and not child_lines:
                desc += extra_lines[0]
            else:
                desc += "\n  " + "\n  ".join(lines) + "\n"

        return f"{self._get_name()}({desc})"

    def __getitem__(self, key: str) -> torch.nn.Module:
        module = self._modules[key]
        if not module:
            raise ValueError(f"Module '{key}' is None or missing in self._modules")
        return module

    def keys(self) -> Iterable[str]:
        return self._modules.keys()

    def items(self) -> Iterable[tuple[str, torch.nn.Module | None]]:
        return self._modules.items()

    def values(self) -> Iterable[torch.nn.Module | None]:
        return self._modules.values()

    def add_module(
        self,
        name: str,
        module: torch.nn.Module,
        in_branch: Sequence[int | str] = [0],
        out_branch: Sequence[int | str] = [0],
        pretrained: bool = True,
        alias: list[str] = [],
        requires_grad: bool | None = None,
        training: None | bool = None,
    ) -> None:
        super().add_module(name, module)
        self._modulesArgs[name] = ModuleArgsDict.ModuleArgs(
            [str(value) for value in in_branch],
            [str(value) for value in out_branch],
            pretrained,
            alias,
            requires_grad,
            training,
        )

    def get_mapping(self):
        results: dict[str, str] = {}
        for name, module_args in self._modulesArgs.items():
            module = self[name]
            if isinstance(module, ModuleArgsDict):
                if len(module_args.alias):
                    count = dict.fromkeys(set(module.get_mapping().values()), 0)
                    if len(count):
                        for k, v in module.get_mapping().items():
                            if count[v] >= len(module_args.alias):
                                raise ConfigError(
                                    f"Module '{name}' declares {len(module_args.alias)} alias(es) but its graph "
                                    f"maps at least {count[v] + 1} entries onto '{v}'.",
                                    "Alias lists are positional: one alias per mapped occurrence. Extend the "
                                    "'alias' list of add_module to cover every occurrence.",
                                )
                            alias_name = module_args.alias[count[v]]
                            if k == "":
                                results.update({alias_name: name + "." + v})
                            else:
                                results.update({alias_name + "." + k: name + "." + v})
                            count[v] += 1
                    else:
                        for alias in module_args.alias:
                            results.update({alias: name})
                else:
                    results.update({k: name + "." + v for k, v in module.get_mapping().items()})
            else:
                for alias in module_args.alias:
                    results[alias] = name
        return results

    @staticmethod
    def init_func(module: torch.nn.Module, init_type: str, init_gain: float):
        if not isinstance(module, Network):
            if isinstance(module, ModuleArgsDict):
                module.init(init_type, init_gain)
            elif isinstance(module, torch.nn.modules.conv._ConvNd) or isinstance(module, torch.nn.Linear):
                if init_type == "normal":
                    torch.nn.init.normal_(module.weight, 0.0, init_gain)
                elif init_type == "xavier":
                    torch.nn.init.xavier_normal_(module.weight, gain=init_gain)
                elif init_type == "kaiming":
                    torch.nn.init.kaiming_normal_(module.weight, a=0, mode="fan_in")
                elif init_type == "orthogonal":
                    torch.nn.init.orthogonal_(module.weight, gain=init_gain)
                elif init_type == "trunc_normal":
                    torch.nn.init.trunc_normal_(module.weight, std=init_gain)
                else:
                    raise NotImplementedError(f"Initialization method {init_type} is not implemented")
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0.0)

            elif isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                if module.weight is not None:
                    # Normalisation gamma must centre on 1, not 0 (the pix2pix convention): a gamma
                    # near 0 scales the normalised activations to ~0 and stalls early training.
                    torch.nn.init.normal_(module.weight, 1.0, std=init_gain)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0.0)

    def init(self, init_type: str, init_gain: float):
        for module in self._modules.values():
            ModuleArgsDict.init_func(module, init_type, init_gain)

    def named_forward(
        self, *inputs: torch.Tensor, attributes: list[list[Attribute]] | None = None
    ) -> Iterator[tuple[str, torch.Tensor]]:
        if len(inputs) > 0:
            if self._channels_last:
                # Once, as the graph is entered: cuDNN hands a channels-last input's output back in
                # that layout, so converting again at every module only recopies what it kept.
                inputs = tuple(_channels_last(tensor) for tensor in inputs)
            branchs: dict[str, torch.Tensor] = {}
            attribute_branchs: dict[str, list[Attribute]] = {}
            for i, sinput in enumerate(inputs):
                branchs[str(i)] = sinput
                if attributes is not None and i < len(attributes) and attributes[i] is not None:
                    attribute_branchs[str(i)] = attributes[i]

            out = inputs[0]
            tmp: list[int | str] = []
            for name, module in self.items():
                # Reset per module: ``tmp`` tracks out_branches a nested sibling already filled via
                # inner-match. Kept across siblings, a later sibling sharing that out_branch would skip
                # the fallback below and its output would be silently dropped.
                tmp = []
                if self._modulesArgs[name].training is None or (
                    not (self._modulesArgs[name].training and self._training == NetState.PREDICTION)
                    and not (not self._modulesArgs[name].training and self._training == NetState.TRAIN)
                ):
                    requires_grad = self._modulesArgs[name].requires_grad
                    if requires_grad is not None and module:
                        module.requires_grad_(requires_grad)
                    target_gpu = self._modulesArgs[name].gpu
                    for ib in self._modulesArgs[name].in_branch:
                        if ib not in branchs:
                            # Numeric branches fall back to the network input (branch '0' = input; extra
                            # indices are legitimate scratch wiring). A NAMED branch nobody produced is a
                            # miswired graph: routing the raw input silently would hide it.
                            if not ib.lstrip("-").isdigit():
                                raise ConfigError(
                                    f"Module '{name}' reads branch '{ib}', which no earlier module has produced.",
                                    f"Known branches here: {sorted(branchs)}. A named branch must be written "
                                    "(out_branch) by a module that runs earlier; check the label for a typo "
                                    "and the producer's training gate.",
                                )
                            branchs[ib] = inputs[0]
                        if target_gpu != "cpu" and str(branchs[ib].device) != f"cuda:{target_gpu}":
                            branchs[ib] = branchs[ib].to(
                                int(target_gpu),
                                non_blocking=branchs[ib].device.type == "cpu",
                            )

                    if self._modulesArgs[name].isCheckpoint:
                        out = checkpoint(
                            module,
                            *[branchs[i] for i in self._modulesArgs[name].in_branch],
                            use_reentrant=True,
                        )
                        for ob in self._modulesArgs[name].out_branch:
                            branchs[ob] = out
                        yield name, out
                    else:
                        if isinstance(module, ModuleArgsDict):
                            for k, out in module.named_forward(
                                *[branchs[i] for i in self._modulesArgs[name].in_branch],
                                attributes=(
                                    [attribute_branchs.get(i, [Attribute()]) for i in self._modulesArgs[name].in_branch]
                                    if attribute_branchs
                                    else None
                                ),
                            ):
                                for ob in self._modulesArgs[name].out_branch:
                                    if ob in module._modulesArgs[strip_accumulated(k.split(".")[0])].out_branch:
                                        tmp.append(ob)
                                        branchs[ob] = out
                                yield name + "." + k, out
                            for ob in self._modulesArgs[name].out_branch:
                                if ob not in tmp:
                                    branchs[ob] = out
                        elif isinstance(module, torch.nn.Module):
                            if getattr(module, "accepts_attributes", False):
                                out = module(
                                    *[branchs[i] for i in self._modulesArgs[name].in_branch],
                                    attributes=[
                                        attribute_branchs.get(i, [Attribute()])
                                        for i in self._modulesArgs[name].in_branch
                                    ],
                                )
                            else:
                                out = module(*[branchs[i] for i in self._modulesArgs[name].in_branch])
                            for ob in self._modulesArgs[name].out_branch:
                                branchs[ob] = out
                            yield name, out
            del branchs

    def forward(self, *input: torch.Tensor) -> torch.Tensor:
        _v = input
        for _, _v in self.named_forward(*input):
            pass
        return _v

    def graph_parameters(self, pretrained: bool = False) -> Iterator[tuple[str, torch.nn.parameter.Parameter]]:
        """The routed graph's trainable parameters, named by dotted module path.

        Unlike ``named_parameters`` (torch semantics, untouched), this walk honours the graph
        metadata: a module gated off by ``training=False`` is skipped, and ``pretrained=True``
        keeps only the modules declared ``pretrained=False``.
        """
        for name, module_args in self._modulesArgs.items():
            module = self[name]
            if isinstance(module, ModuleArgsDict):
                for k, v in module.graph_parameters(pretrained=pretrained):
                    yield name + "." + k, v
            elif isinstance(module, torch.nn.Module):
                if not pretrained or not module_args.pretrained:
                    if module_args.training is None or module_args.training:
                        for k, v in module.named_parameters():
                            yield name + "." + k, v

    def named_module_args_dict(self) -> Iterator[tuple[str, Self, ModuleArgs]]:
        for name, module in self._modules.items():
            yield name, module, self._modulesArgs[name]
            if isinstance(module, ModuleArgsDict):
                for k, v, u in module.named_module_args_dict():
                    yield name + "." + k, v, u

    def _requires_grad(self, keys: list[str]):
        keys = keys.copy()
        for name, module, args in self.named_module_args_dict():
            requires_grad = args.requires_grad
            if requires_grad is not None:
                module.requires_grad_(requires_grad)
            if name in keys:
                keys.remove(name)
                if len(keys) == 0:
                    break

    def _trace_downsampling(self, seeds: list[list[int]], seen: list[list[int]]) -> list[int]:
        """Propagate the per-axis downsampling factor through the branch register, recording each branch
        value in ``seen``. Parallel branches (a residual shortcut beside the main path) accumulate from
        the SAME seed and merge at their ``Add`` without multiplying, so a strided projection is not
        double-counted the way a flat ``modules()`` walk would. A child that is NOT a routed block is
        opaque and contributes its flat internal product (``_flat_downsampling``).

        ``seeds`` are this block's input factors, one per positional input; the register is seeded from
        all of them (a decoder block reading ``[upsampled, skip]`` keeps each at its own resolution) and
        an unwritten branch falls back to the first, exactly as ``named_forward`` seeds it. A module
        downsamples along its FIRST input branch; the others only route. Returns the last output's factor.
        """
        branches: dict[str, list[int]] = {str(i): seed for i, seed in enumerate(seeds)}
        default = seeds[0]
        out_f = default
        for name, module in self.items():
            module_args = self._modulesArgs[name]
            in_factors = [branches.get(in_branch, default) for in_branch in module_args.in_branch]
            if isinstance(module, ModuleArgsDict):
                out_f = module._trace_downsampling(in_factors, seen)
            else:
                out_f = [a * b for a, b in zip(in_factors[0], _flat_downsampling(module, len(default)), strict=True)]
            for out_branch in module_args.out_branch:
                branches[out_branch] = out_f
            seen.append(out_f)
        return out_f


class OutputsGroup(list):
    """Container describing one model output and its source modules.

    Carries the OWNING network, not just its measure: criteria are scheduled on the owner's ``_it``
    (the counter its backward advances). A composite root never steps its own ``_it``, so scheduling
    on the root would freeze every start/stop window and loss-weight scheduler at 0.
    """

    def __init__(self, network: "Network") -> None:
        self.layers: dict[str, torch.Tensor] = {}
        self.network = network
        # init_outputs_group only builds groups for networks that own a measure.
        self.measure: Measure = network.measure  # type: ignore[assignment]

    def add_layer(self, name: str, layer: torch.Tensor):
        self.layers[name] = layer

    def is_done(self):
        return len(self) == len(self.layers)

    def clear(self):
        self.layers.clear()


class Network(ModuleArgsDict, ABC):
    """Base class for KonfAI networks participating in a routed model graph."""

    def _apply_network(
        self,
        name_function: Callable[[Self], str],
        networks: dict[str, "Network"],
        key: str,
        function: Callable,
        *args,
        root: "Network | None" = None,
        **kwargs,
    ) -> dict[str, object]:
        # The first caller in the recursion is the root graph; thread it (and the dotted key) down so a
        # nested network can address the whole graph: e.g. a GAN generator whose loss targets a module
        # of a sibling discriminator branch, which only exists in the root's module namespace.
        root = root if root is not None else self
        results: dict[str, object] = {}
        for module in self.values():
            if isinstance(module, Network):
                name = name_function(module)
                known = networks.get(name)
                if known is module:
                    # The same object under several module names (a GAN's shared discriminator)
                    # is visited once.
                    continue
                if known is not None:
                    raise ConfigError(
                        f"Two distinct networks share the name '{name}' in the graph of '{name_function(root)}'.",
                        "The name is the checkpoint key: a collision cannot be saved or resumed "
                        "correctly. Give one of them its own name with set_name().",
                    )
                networks[name] = module
                for k, v in module._apply_network(
                    name_function,
                    networks,
                    key + "." + name,
                    function,
                    *args,
                    root=root,
                    **kwargs,
                ).items():
                    results.update({name_function(self) + "." + k: v})
        param_names = {param.name for param in inspect.signature(function).parameters.values()}
        if "key" in param_names:
            function = partial(function, key=key)
        if "root" in param_names:
            function = partial(function, root=root)

        results[name_function(self)] = function(self, *args, **kwargs)
        return results

    def _function_network():  # type: ignore[misc]
        def _function_network_d(function: Callable):
            def new_function(self: Self, *args, **kwargs) -> dict[str, object]:
                return self._apply_network(
                    lambda network: network.get_name(),
                    {},
                    self.get_name(),
                    function,
                    *args,
                    **kwargs,
                )

            return new_function

        return _function_network_d

    def __init__(
        self,
        in_channels: int = 1,
        optimizer: OptimizerLoader | None = None,
        schedulers: dict[str, LRSchedulersLoader] | None = None,
        outputs_criterions: dict[str, TargetCriterionsLoader] | None = None,
        patch: ModelPatch | None = None,
        nb_batch_per_step: int = 1,
        init_type: str = "normal",
        init_gain: float = 0.02,
        dim: int = 3,
        allow_head_resize: bool = False,
    ) -> None:
        super().__init__()
        self.name = self.__class__.__name__
        self.in_channels = in_channels
        self.optimizerLoader = optimizer
        self.optimizer: torch.optim.Optimizer | None = None

        self.lr_schedulers_loader = schedulers
        self.schedulers: dict[torch.optim.lr_scheduler.LRScheduler, int] = {}

        self.outputs_criterions_loader = outputs_criterions
        self.measure: Measure | None = None

        self.patch = patch

        self.nb_batch_per_step = nb_batch_per_step
        self.init_type = init_type
        self.init_gain = init_gain
        self.dim = dim
        #: Opt-in: a checkpoint head whose out-channels mismatch may be re-initialised and
        #: overlap-copied instead of failing the load (transfer to a different label set).
        self.allow_head_resize = allow_head_resize
        self._it = 0
        self._nb_lr_update = 0
        self.outputsGroup: list[OutputsGroup] = []

    @_function_network()
    def network_states(self) -> OrderedDict:
        """Per-network flat state, keyed by ``get_name()`` (dotted for nested networks): the
        checkpoint's ``Model`` entry. The decorated call returns ``dict[str, OrderedDict]``."""
        return self.state_dict()

    def state_dict(  # type: ignore[override]
        self, *, destination: OrderedDict | None = None, prefix: str = "", keep_vars: bool = False
    ) -> OrderedDict:
        """This network's own flat state under the torch signature, skipping nested ``Network``
        children: each owns its optimizer/state and is saved under its own ``network_states`` key."""
        if destination is None:
            destination = OrderedDict()
        local_metadata = {"version": self._version}
        self._save_to_state_dict(destination, prefix, keep_vars)
        for name, module in self._modules.items():
            if module is not None:
                if not isinstance(module, Network):
                    module.state_dict(destination=destination, prefix=prefix + name + ".", keep_vars=keep_vars)
        for hook in self._state_dict_hooks.values():
            hook_result = hook(self, destination, prefix, local_metadata)
            if hook_result is not None:
                destination = hook_result
        return destination

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]):
        missing_keys: list[str] = []
        unexpected_keys: list[str] = []
        error_msgs: list[str] = []

        metadata = getattr(state_dict, "_metadata", None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict["_metadata"] = metadata

        def load(module: torch.nn.Module, prefix=""):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict,
                prefix,
                local_metadata,
                True,
                missing_keys,
                unexpected_keys,
                error_msgs,
            )
            for name, child in module._modules.items():
                if child is not None:
                    if not isinstance(child, Network):
                        weight_key = prefix + name + ".weight"
                        if (
                            isinstance(child, (torch.nn.modules.conv._ConvNd, torch.nn.Linear))
                            and weight_key in state_dict
                        ):
                            current_size = child.weight.shape[0]
                            last_size = state_dict[weight_key].shape[0]

                            # Opt-in only: without allow_head_resize the mismatch falls through to the
                            # strict load below and raises, naming the tensor and both shapes.
                            if current_size != last_size and self.allow_head_resize:
                                _log.warning(
                                    "The size of '%s' has changed from %s to %s: re-initialised and "
                                    "overlap-copied (allow_head_resize).",
                                    prefix + name,
                                    last_size,
                                    current_size,
                                )
                                ModuleArgsDict.init_func(child, self.init_type, self.init_gain)

                                bias_key = prefix + name + ".bias"
                                # Copy the overlap only. Slicing both sides by min(current, last) keeps the
                                # GROW case (checkpoint smaller -> fill the top rows) working AND fixes the
                                # SHRINK case (checkpoint larger): `weight[:last_size] = ckpt` would pair the
                                # smaller current tensor against the larger checkpoint and crash.
                                overlap = min(current_size, last_size)
                                with torch.no_grad():
                                    child.weight[:overlap] = state_dict[weight_key][:overlap]
                                    if child.bias is not None and bias_key in state_dict:
                                        child.bias[:overlap] = state_dict[bias_key][:overlap]
                                # Skip the normal load for this resized leaf, but keep
                                # loading its siblings.
                                continue
                        load(child, prefix + name + ".")

        load(self)

        if len(unexpected_keys) > 0:
            formatted_keys = ", ".join(f'"{k}"' for k in unexpected_keys)
            error_msgs.insert(
                0,
                f"Unexpected key(s) in state_dict: {formatted_keys}.",
            )
        if len(missing_keys) > 0:
            formatted_keys = ", ".join(f'"{k}"' for k in missing_keys)
            error_msgs.insert(
                0,
                f"Missing key(s) in state_dict: {formatted_keys}.",
            )

        if len(error_msgs) > 0:
            formatted_errors = "\n\t".join(error_msgs)
            raise RuntimeError(
                f"Error(s) in loading state_dict for {self.__class__.__name__}:\n\t{formatted_errors}",
            )

    def graph_apply(self, fn: Callable[[torch.nn.Module], None]) -> None:
        """
        Apply ``fn`` to each non-``Network`` child module and finally to ``self``.

        Nested ``Network`` instances are skipped: each owns its own state (init, load), so a
        fan-out over the graph applies ``fn`` per network, never twice through a parent.
        ``torch.nn.Module.apply`` keeps its native signature and full recursion.
        """
        for module in self.children():
            if not isinstance(module, Network):
                module.apply(fn)
        fn(self)

    @_function_network()
    def load(
        self,
        state_dict: dict[str, dict[str, torch.Tensor] | int],
        init: bool = True,
        ema: bool = False,
        override_lr: float | None = None,
        key: str | None = None,
    ):
        # `checkpoint_save` writes the optimizer/iteration/LR-schedule state under the network's DOTTED path
        # (its get_networks() key, e.g. "Gan.Generator"). `_apply_network` injects that same dotted path as
        # `key` here, so a nested network resumes its own state instead of silently missing the bare-name key.
        state_key = key if key is not None else self.get_name()
        if init:
            self.graph_apply(
                partial(
                    ModuleArgsDict.init_func,
                    init_type=self.init_type,
                    init_gain=self.init_gain,
                )
            )
        name = "Model"
        if ema:
            if name + "_EMA" in state_dict:
                name += "_EMA"
        if name in state_dict:
            value = state_dict[name]
            model_state_dict_tmp = {}
            if isinstance(value, dict):
                model_state_dict_tmp = {k.split(".")[-1]: v for k, v in value.items()}[self.get_name()]
            modules_name = self.get_mapping()
            model_state_dict: OrderedDict[str, torch.Tensor] = OrderedDict()

            for alias in model_state_dict_tmp.keys():
                prefix = ".".join(alias.split(".")[:-1])
                # Segment-aligned: alias 'layer1' must not claim a module 'layer10', and only the
                # leading prefix is rewritten, never a later occurrence of the same substring.
                alias_list = [(a, b) for a, b in modules_name.items() if prefix == a or prefix.startswith(a + ".")]

                if len(alias_list):
                    a, b = alias_list[0]
                    model_state_dict[b + alias[len(a) :]] = model_state_dict_tmp[alias]
                else:
                    model_state_dict[alias] = model_state_dict_tmp[alias]
            self.load_state_dict(model_state_dict)
        if f"{state_key}_optimizer_state_dict" in state_dict and self.optimizer:
            self.optimizer.load_state_dict(state_dict[f"{state_key}_optimizer_state_dict"])
        if f"{state_key}_it" in state_dict:
            _it = state_dict.get(f"{state_key}_it")
            if isinstance(_it, int):
                self._it = _it
        if f"{state_key}_nb_lr_update" in state_dict:
            _nb_lr_update = state_dict.get(f"{state_key}_nb_lr_update")
            if isinstance(_nb_lr_update, int):
                self._nb_lr_update = _nb_lr_update

        if override_lr is not None:
            self._rebase_lr_local(override_lr)
        else:
            for scheduler in self.schedulers:
                scheduler.last_epoch = self._nb_lr_update
        self.initialized()

    def _compute_channels_trace(
        self,
        module: ModuleArgsDict,
        in_channels: int,
        gradient_checkpoints: list[str] | None,
        gpu_checkpoints: list[str] | None,
        name: str | None = None,
        in_is_channel: bool = True,
        out_channels: int | None = None,
        out_is_channel: bool = True,
    ) -> tuple[int, bool, int | None, bool]:

        for k1, v1 in module.items():
            if isinstance(v1, ModuleArgsDict):
                for t in module._modulesArgs[k1].out_branch:
                    last = None
                    for k2, _ in v1.items():
                        if t in v1._modulesArgs[k2].out_branch:
                            last = k2
                    if last is not None:
                        v1._modulesArgs[last]._isEnd = True
                    else:
                        v1._modulesArgs[k2]._isEnd = True

        for k, v in module.items():
            if hasattr(v, "in_channels"):
                if v.in_channels:
                    in_channels = v.in_channels
            if hasattr(v, "in_features"):
                if v.in_features:
                    in_channels = v.in_features
            key = name + "." + k if name else k

            if gradient_checkpoints:
                if key in gradient_checkpoints:
                    module._modulesArgs[k].isCheckpoint = True

            if gpu_checkpoints:
                if key in gpu_checkpoints:
                    module._modulesArgs[k].isGPU_Checkpoint = True

            module._modulesArgs[k].in_channels = in_channels
            module._modulesArgs[k].in_is_channel = in_is_channel

            if isinstance(v, ModuleArgsDict):
                in_channels, in_is_channel, out_channels, out_is_channel = self._compute_channels_trace(
                    v,
                    in_channels,
                    gradient_checkpoints,
                    gpu_checkpoints,
                    key,
                    in_is_channel,
                    out_channels,
                    out_is_channel,
                )

            if v.__class__.__name__ == "ToChannels":
                out_is_channel = True

            if v.__class__.__name__ == "ToFeatures":
                out_is_channel = False

            if hasattr(v, "out_channels"):
                if v.out_channels:
                    out_channels = v.out_channels
            if hasattr(v, "out_features"):
                if v.out_features:
                    out_channels = v.out_features

            module._modulesArgs[k].out_channels = out_channels
            module._modulesArgs[k].out_is_channel = out_is_channel

            in_channels = out_channels if out_channels is not None else in_channels
            in_is_channel = out_is_channel

        return in_channels, in_is_channel, out_channels, out_is_channel

    def downsampling_factor(self) -> list[int] | None:
        """Per-axis factor the input spatial size must be a multiple of, or ``None`` if the graph never
        downsamples.

        An encoder/decoder graph (U-Net) only reassembles its skip connections when the input divides
        evenly at every level, so the input must be a multiple of the coarsest downsampling the graph
        reaches. That factor is traced through the branch register: a strided ``Conv`` or a ``MaxPool``
        multiplies the branch it writes, while ``ConvTranspose``/``Upsample`` and a residual branch's
        ``AvgPool`` pass through. Because the trace follows branches, a residual block's strided shortcut
        (parallel to its strided main conv, merged by ``Add``) counts ONCE, not twice. Used to size a
        free (``0``) patch axis to a valid extent (padded up, cropped back after the forward).
        """
        # The graph's spatial rank = the WIDEST strided leaf (a 2D side head in a 3D net must not lock
        # the rank to 2); every leaf stride then aligns to the trailing axes of that rank.
        ndim = max((len(s) for s in map(_leaf_spatial_stride, self.modules()) if s is not None), default=0)
        if ndim == 0:
            return None
        seen: list[list[int]] = []
        self._trace_downsampling([[1] * ndim], seen)
        factor = [max((f[axis] for f in seen), default=1) for axis in range(ndim)]
        return factor if any(f > 1 for f in factor) else None

    @_function_network()
    def init(self, autocast: bool, state: State, group_dest: list[str], key: str, root: "Network") -> None:
        if self.outputs_criterions_loader:
            self.measure = Measure(key, self.outputs_criterions_loader)
            # Validate the criterion targets against the ROOT graph, where runtime matching also happens:
            # a nested network's loss may address a module in a sibling branch (a GAN generator's
            # adversarial loss on the discriminator) that exists only in the root's module namespace.
            self.measure.init(root, group_dest)
        if self.patch is not None:
            self.patch.init(f"{konfai_root()}.Model.{key}.Patch")
        if state != State.PREDICTION:
            self.scaler = torch.amp.GradScaler("cuda", enabled=autocast)
            if self.measure is not None:
                self.measure.scaler = self.scaler
            if self.optimizerLoader:
                self.optimizer = self.optimizerLoader.get_optimizer(
                    key, (parameter for _, parameter in self.graph_parameters())
                )
                self.optimizer.zero_grad()

            if self.lr_schedulers_loader and self.optimizer:
                for schedulers_classname, schedulers in self.lr_schedulers_loader.items():
                    self.schedulers[schedulers.getschedulers(key, schedulers_classname, self.optimizer)] = (
                        schedulers.nb_step
                    )

    def initialized(self):
        pass

    def named_forward(
        self, *inputs: torch.Tensor, attributes: list[list[Attribute]] | None = None
    ) -> Iterator[tuple[str, torch.Tensor]]:
        if self.patch:
            self.patch.load(inputs[0].shape[2:])
            accumulators: dict[str, Accumulator] = {}

            patch_iterator = self.patch.disassemble(*inputs)
            buffer = []
            for i, patch_input in enumerate(patch_iterator):
                for name, output_layer in super().named_forward(*patch_input, attributes=attributes):
                    yield mark_accumulated(name), output_layer
                    buffer.append((name.split(".")[0], output_layer))
                    if len(buffer) == 2:
                        if buffer[0][0] != buffer[1][0]:
                            if self._modulesArgs[buffer[0][0]]._isEnd:
                                if buffer[0][0] not in accumulators:
                                    accumulators[buffer[0][0]] = Accumulator(
                                        self.patch.get_patch_slices(),
                                        self.patch.patch_size,
                                        self.patch.patch_combine,
                                    )
                                accumulators[buffer[0][0]].add_layer(i, buffer[0][1])
                        buffer.pop(0)
                if self._modulesArgs[buffer[0][0]]._isEnd:
                    if buffer[0][0] not in accumulators:
                        accumulators[buffer[0][0]] = Accumulator(
                            self.patch.get_patch_slices(),
                            self.patch.patch_size,
                            self.patch.patch_combine,
                        )
                    accumulators[buffer[0][0]].add_layer(i, buffer[0][1])
                # The leftover entry must not leak into the next patch iteration: the name-transition
                # branch above would re-add patch i's end-module output at index i+1, and Accumulator
                # blends incrementally, so a spurious first add cannot be overwritten later.
                buffer.clear()
            for name, accumulator in accumulators.items():
                yield name, accumulator.assemble()
        else:
            for name, output_layer in super().named_forward(*inputs, attributes=attributes):
                yield name, output_layer

    def get_layers(
        self,
        inputs: list[torch.Tensor],
        layers_name: list[str],
        attributes: list[list[Attribute]] | None = None,
    ) -> Iterator[tuple[str, torch.Tensor, PatchIndexed | None]]:
        layers_name = layers_name.copy()
        output_layer_accumulator: dict[str, Accumulator] = {}
        output_layer_patch_indexed: dict[str, PatchIndexed] = {}
        it = 0
        debug = "KONFAI_DEBUG" in os.environ
        for name_tmp, output_layer in self.named_forward(*inputs, attributes=attributes):
            name = strip_accumulated(name_tmp)
            if debug:
                if "KONFAI_DEBUG_LAST_LAYER" in os.environ:
                    os.environ["KONFAI_DEBUG_LAST_LAYER"] = (
                        f"{os.environ['KONFAI_DEBUG_LAST_LAYER']}|{name}:"
                        f"{get_gpu_memory(output_layer.device)}:"
                        f"{str(output_layer.device).replace('cuda:', '')}"
                    )
                else:
                    os.environ["KONFAI_DEBUG_LAST_LAYER"] = (
                        f"{name}:{get_gpu_memory(output_layer.device)}:{str(output_layer.device).replace('cuda:', '')}"
                    )
            it += 1
            if name in layers_name or name_tmp in layers_name:
                if is_accumulated(name_tmp):
                    if name not in output_layer_patch_indexed:
                        network_name = accumulator_owner(name_tmp)
                        module = self
                        network = None
                        if network_name == "":
                            network = module
                        else:
                            for n in name.split("."):
                                module = module[n]
                                if isinstance(module, Network) and n == network_name:
                                    network = module
                                    break

                        if network and network.patch:
                            output_layer_patch_indexed[name] = PatchIndexed(network.patch, 0)

                    if name not in output_layer_accumulator:
                        output_layer_accumulator[name] = Accumulator(
                            output_layer_patch_indexed[name].patch.get_patch_slices(0),
                            output_layer_patch_indexed[name].patch.patch_size,
                            output_layer_patch_indexed[name].patch.patch_combine,
                        )

                    if name_tmp in layers_name:
                        output_layer_accumulator[name].add_layer(output_layer_patch_indexed[name].index, output_layer)
                        output_layer_patch_indexed[name].index += 1
                        if output_layer_accumulator[name].is_full():
                            output_layer = output_layer_accumulator[name].assemble()
                            output_layer_accumulator.pop(name)
                            output_layer_patch_indexed.pop(name)
                            layers_name.remove(name_tmp)
                            yield name_tmp, output_layer, None

                if name in layers_name:
                    if is_accumulated(name_tmp):
                        yield name, output_layer, output_layer_patch_indexed[name]
                        output_layer_patch_indexed[name].index += 1
                        if output_layer_patch_indexed[name].is_full():
                            output_layer_patch_indexed.pop(name)
                            layers_name.remove(name)
                    else:
                        layers_name.remove(name)
                        yield name, output_layer, None

            if not len(layers_name):
                break

    def bind(
        self,
        autocast: bool,
        state: State,
        group_dest: list[str],
        gradient_checkpoints: list[str] | None = None,
        gpu_checkpoints: list[str] | None = None,
    ) -> None:
        """Wire the root model to the dataset's destination groups for ``state``: every network's
        criteria and patch, the output groups the measures address, and the channel trace the
        checkpoints are placed on."""
        self.init(autocast, state, group_dest)
        self.init_outputs_group()
        self._compute_channels_trace(self, self.in_channels, gradient_checkpoints, gpu_checkpoints)

    def init_outputs_group(self):
        for network in self.get_networks().values():
            if not network.measure:
                continue
            for output_name in network.measure.outputs_criterions.keys():
                outputs_group = OutputsGroup(network)
                outputs_group.append(output_name)
                for targets_group in network.measure.outputs_criterions[output_name].keys():
                    if ":" in targets_group:
                        outputs_group.append(targets_group.replace(":", "."))

                self.outputsGroup.append(outputs_group)

    def forward(
        self,
        batch_sample: BatchSample,
        output_layers: list[str] = [],
        clock: SweepClock | None = None,
    ) -> list[tuple[str, torch.Tensor]]:
        """The graph walk over ``batch_sample``, its criteria evaluated as their output groups complete.
        ``clock`` charges the criteria to its ``criteria`` phase, apart from the walk."""
        if not len(self.outputsGroup) and not len(output_layers):
            return []

        self.reset_loss()
        try:
            return self._forward(batch_sample, output_layers, clock)
        finally:
            self.release_targets()

    def _forward(
        self, batch_sample: BatchSample, output_layers: list[str], clock: SweepClock | None
    ) -> list[tuple[str, torch.Tensor]]:
        results = []
        measure_output_layers = set()
        for _outputs_group in self.outputsGroup:
            for name in _outputs_group:
                measure_output_layers.add(name)

        for name, layer, patch_indexed in self.get_layers(
            [batch_data_item.tensor for batch_data_item in batch_sample.values() if batch_data_item.is_input],
            list(set(list(measure_output_layers) + output_layers)),
            attributes=[
                batch_data_item.attribute for batch_data_item in batch_sample.values() if batch_data_item.is_input
            ],
        ):
            outputs_group = [outputs_group for outputs_group in self.outputsGroup if name in outputs_group]

            if len(outputs_group) > 0:
                if patch_indexed is None:
                    batch_data_with_attribute = {
                        k: (batch_data_item.tensor, batch_data_item.attribute)
                        for k, batch_data_item in batch_sample.items()
                    }
                    nb = 1
                else:
                    batch_data_with_attribute = {
                        k: (
                            patch_indexed.patch.get_data(batch_data_item.tensor, patch_indexed.index, 0, False),
                            batch_data_item.attribute,
                        )
                        for k, batch_data_item in batch_sample.items()
                    }
                    nb = patch_indexed.patch.get_size(0)

                for output_group in outputs_group:
                    output_group.add_layer(name, layer)
                    if output_group.is_done():
                        batch_data_with_attribute.update(
                            {
                                k.replace(".", ":"): (batch_data_item, [Attribute()])
                                for k, batch_data_item in output_group.layers.items()
                                if k != output_group[0]
                            }
                        )
                        with clock.phase("criteria") if clock is not None else nullcontext():
                            output_group.measure.update(
                                output_group[0],
                                output_group.layers[output_group[0]],
                                batch_data_with_attribute,
                                output_group.network._it,
                                nb,
                                self.training,
                            )
                        output_group.clear()
            if name in output_layers:
                results.append((name, layer))
        return results

    @_function_network()
    def reset_loss(self):
        if self.measure:
            self.measure.reset_loss()

    @_function_network()
    def release_targets(self):
        if self.measure:
            self.measure.release_targets()

    @_function_network()
    def backward(self, model: Any):
        if self.measure:
            if self.scaler and self.optimizer:
                self._requires_grad(list(self.measure.outputs_criterions.keys()))
                should_step = (self._it + 1) % self.nb_batch_per_step == 0
                sync_context = (
                    model.no_sync()
                    if hasattr(model, "no_sync") and callable(model.no_sync) and not should_step
                    else nullcontext()
                )
                with sync_context:
                    for loss in self.measure.get_loss():
                        self.scaler.scale(loss / self.nb_batch_per_step).backward()

                if should_step:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                self._it += 1

    @_function_network()
    def update_lr(self):
        self._nb_lr_update += 1
        step = 0
        _scheduler = None
        for _scheduler, value in self.schedulers.items():
            if value is None or (self._nb_lr_update >= step and self._nb_lr_update < step + value):
                break
            step += value
        if _scheduler:
            if _scheduler.__class__.__name__ == "ReduceLROnPlateau":
                if self.measure:
                    _scheduler.step(sum(self.measure.get_last_values(0).values()))
            else:
                _scheduler.step()

    def _rebase_lr_local(self, new_lr: float) -> None:
        """Set this one network's optimizer LR to ``new_lr`` and rebase its schedulers onto it (base_lrs /
        initial_lr / _last_lr) with last_epoch reset, so the next scheduler step keeps the new value instead
        of re-decaying from the old anchor. Plain (no fan-out): the callers own the recursion."""
        if self.optimizer is not None:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = new_lr
                param_group["initial_lr"] = new_lr
        for scheduler in self.schedulers:
            sched: Any = scheduler
            if hasattr(sched, "base_lrs"):
                sched.base_lrs = [new_lr for _ in sched.base_lrs]
            if hasattr(sched, "initial_lr"):
                sched.initial_lr = new_lr
            sched.last_epoch = 0
            if hasattr(sched, "_last_lr"):
                sched._last_lr = [new_lr for _ in sched._last_lr]

    @_function_network()
    def rebase_lr(self, new_lr: float) -> None:
        """Rebase the learning rate of this network and every nested one onto ``new_lr``: the same restart a
        RESUME with ``--lr`` applies, reused for a live mid-run change so the value sticks past the scheduler."""
        self._rebase_lr_local(new_lr)

    @_function_network()
    def get_networks(self) -> Self:
        return self

    @staticmethod
    def set_channels_last(module: ModuleArgsDict) -> ModuleArgsDict:
        """Lay the graph's convolution weights out channels-last, and its inputs as they enter.

        cuDNN picks its kernels by layout: under autocast the shipped Segmentation example predicts
        in 2.2 s against 2.7 s in the default layout (fp32: 4.1 against 4.2 s, where the kernels
        chosen differ on 3199 of 58.4 million label voxels). Off by default, so the default layout
        is what a run gets unless it asks.
        """
        for tensor in (*module.parameters(), *module.buffers()):
            tensor.data = _channels_last(tensor.data)  # a 4-D and a 5-D weight each take their own layout
        for submodule in module.modules():
            if isinstance(submodule, ModuleArgsDict):
                submodule._channels_last = True
        return module

    @staticmethod
    def to(module: ModuleArgsDict, device: int, _counter: list[int] | None = None):
        # `_counter` is a single-element box holding the next GPU index, shared by
        # reference through the recursion so model-parallel `isGPU_Checkpoint` splits
        # advance it. Each top-level call starts fresh at `device` so the counter never
        # leaks across independent placements.
        if _counter is None:
            _counter = [device]
        for k, v in module.items():
            if module._modulesArgs[k].gpu == "cpu":
                if module._modulesArgs[k].isGPU_Checkpoint:
                    _counter[0] += 1
                module._modulesArgs[k].gpu = str(get_device(_counter[0]))
                if isinstance(v, ModuleArgsDict):
                    v = Network.to(v, _counter[0], _counter)
                else:
                    v = v.to(get_device(_counter[0]))
        if isinstance(module, Network):
            if module.optimizer is not None:
                for state in module.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(get_device(_counter[0]))
        return module

    def get_name(self) -> str:
        return self.name

    def set_name(self, name: str) -> Self:
        self.name = name
        return self

    def set_state(self, state: NetState):
        for module in self.modules():
            if isinstance(module, ModuleArgsDict):
                module._training = state


class MinimalModel(Network):
    """Small wrapper exposing a single network as a full KonfAI model graph.

    The wrapped model arrives fully constructed: possibly carrying pretrained weights (a
    torchvision/MONAI/SMP class with ``weights=...``). ``load`` therefore never re-initialises:
    ``load(init=True)`` at training start applies ``init_func`` over every descendant and would
    silently destroy those weights with ``init_type`` noise. Models built from scratch keep
    KonfAI's init behaviour; checkpoint loading is unaffected.
    """

    def load(
        self,
        state_dict: dict[str, dict[str, torch.Tensor] | int],
        init: bool = True,
        ema: bool = False,
        override_lr: float | None = None,
    ):
        del init  # the wrapped model owns its initialisation (possibly pretrained)
        super().load(state_dict, init=False, ema=ema, override_lr=override_lr)

    def __init__(
        self,
        model: Network,
        optimizer: OptimizerLoader = OptimizerLoader(),
        schedulers: dict[str, LRSchedulersLoader] = {"default|StepLR": LRSchedulersLoader(0)},
        outputs_criterions: dict[str, TargetCriterionsLoader] = {"default": TargetCriterionsLoader()},
        patch: ModelPatch | None = None,
        dim: int = 3,
        nb_batch_per_step=1,
        init_type="normal",
        init_gain=0.02,
    ):
        super().__init__(
            1,
            optimizer,
            schedulers,
            outputs_criterions,
            patch,
            nb_batch_per_step,
            init_type,
            init_gain,
            dim,
        )
        self.add_module("Model", model)
