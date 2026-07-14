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

"""Build routed KonfAI :class:`~konfai.network.network.Network` graphs from safe YAML."""

from __future__ import annotations

import copy
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import ruamel.yaml
import torch

from konfai.data.patching import ModelPatch
from konfai.network import blocks
from konfai.network.network import (
    LRSchedulersLoader,
    ModuleArgsDict,
    Network,
    OptimizerLoader,
    TargetCriterionsLoader,
)
from konfai.utils.errors import ConfigError

ModuleFactory = Callable[..., torch.nn.Module]
ObjectFactory = Callable[..., Any]
_REFERENCE = re.compile(r"^\$\{([A-Za-z0-9_.-]+)}$")


def _dimensional_factory(name: str) -> ModuleFactory:
    def factory(*, dim: int, **kwargs: Any) -> torch.nn.Module:
        return blocks.get_torch_module(name, dim=dim)(**kwargs)

    return factory


_MODULE_REGISTRY: dict[str, ModuleFactory] = {
    "Conv": _dimensional_factory("Conv"),
    "ConvTranspose": _dimensional_factory("ConvTranspose"),
    "MaxPool": _dimensional_factory("MaxPool"),
    "AvgPool": _dimensional_factory("AvgPool"),
    "AdaptiveAvgPool": _dimensional_factory("AdaptiveAvgPool"),
    "BatchNorm": _dimensional_factory("BatchNorm"),
    "InstanceNorm": _dimensional_factory("InstanceNorm"),
    "Conv1d": torch.nn.Conv1d,
    "Conv2d": torch.nn.Conv2d,
    "Conv3d": torch.nn.Conv3d,
    "Dropout": torch.nn.Dropout,
    "Dropout1d": torch.nn.Dropout1d,
    "Dropout2d": torch.nn.Dropout2d,
    "Dropout3d": torch.nn.Dropout3d,
    "GroupNorm": torch.nn.GroupNorm,
    "LayerNorm": torch.nn.LayerNorm,
    "Upsample": torch.nn.Upsample,
    "Flatten": torch.nn.Flatten,
    "Linear": torch.nn.Linear,
    "ReLU": torch.nn.ReLU,
    "LeakyReLU": torch.nn.LeakyReLU,
    "PReLU": torch.nn.PReLU,
    "GELU": torch.nn.GELU,
    "Sigmoid": torch.nn.Sigmoid,
    "Tanh": torch.nn.Tanh,
    "Softmax": torch.nn.Softmax,
    "Identity": torch.nn.Identity,
    "Add": blocks.Add,
    "Multiply": blocks.Multiply,
    "ClipNormalize": blocks.ClipNormalize,
    "ArgMax": blocks.ArgMax,
    "Attention": blocks.Attention,
    "ConvBlock": blocks.ConvBlock,
    "ResBlock": blocks.ResBlock,
    "ResidualBlockD": blocks.ResidualBlockD,
    "Concat": blocks.Concat,
    "Permute": blocks.Permute,
    "Select": blocks.Select,
    "Unsqueeze": blocks.Unsqueeze,
    "View": blocks.View,
    "PositionalEmbedding": blocks.PositionalEmbedding,
    "MultiHeadSelfAttention": blocks.MultiHeadSelfAttention,
}

_OBJECT_REGISTRY: dict[str, ObjectFactory] = {
    "BlockConfig": blocks.BlockConfig,
}

_yaml = ruamel.yaml.YAML(typ="safe")


class YamlModuleGraph(ModuleArgsDict):
    """Nested routed graph assembled from a YAML ``modules`` list."""


class YamlNetwork(Network):
    """Full KonfAI network whose children are populated through ``add_module``."""

    def __init__(
        self,
        name: str,
        module_specs: list[dict[str, Any]],
        parameters: dict[str, Any],
        *,
        in_channels: int = 1,
        optimizer: OptimizerLoader | None = None,
        schedulers: dict[str, LRSchedulersLoader] | None = None,
        outputs_criterions: dict[str, TargetCriterionsLoader] | None = None,
        patch: ModelPatch | None = None,
        nb_batch_per_step: int = 1,
        init_type: str = "normal",
        init_gain: float = 0.02,
        dim: int = 3,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            optimizer=optimizer,
            schedulers=schedulers,
            outputs_criterions=outputs_criterions,
            patch=patch,
            nb_batch_per_step=nb_batch_per_step,
            init_type=init_type,
            init_gain=init_gain,
            dim=dim,
        )
        self.name = name
        self.yaml_parameters = copy.deepcopy(parameters)
        _populate_graph(self, module_specs, parameters)

    def forward_tensor(self, *inputs: torch.Tensor) -> torch.Tensor:
        """Execute the module graph directly, outside the training ``BatchSample`` API."""
        output: torch.Tensor | None = None
        for _, current_output in self.named_forward(*inputs):
            output = current_output
        if output is None:
            raise RuntimeError("The YAML network contains no executable modules.")
        return output


def register_module(name: str, cls: type[torch.nn.Module]) -> None:
    """Register a safe ``torch.nn.Module`` subclass for use in YAML."""
    if not isinstance(name, str) or not name:
        raise ConfigError(f"Cannot register module type with invalid name {name!r}: expected a non-empty string.")
    if name in _MODULE_REGISTRY:
        raise ConfigError(
            f"Module type '{name}' is already registered.",
            f"Registered types: {list_registered_modules()}.",
        )
    if not isinstance(cls, type) or not issubclass(cls, torch.nn.Module):
        raise ConfigError(f"Cannot register '{name}': expected a torch.nn.Module subclass, got {cls!r}.")
    _MODULE_REGISTRY[name] = cls


def list_registered_modules() -> list[str]:
    """Return all safe module type names accepted by the builder."""
    return sorted(_MODULE_REGISTRY)


def _lookup_reference(path: str, parameters: dict[str, Any]) -> Any:
    value: Any = parameters
    for part in path.split("."):
        try:
            value = value[int(part)] if isinstance(value, (list, tuple)) else value[part]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ConfigError(f"Unknown YAML model parameter reference '${{{path}}}'.") from exc
    return copy.deepcopy(value)


def _resolve_value(value: Any, parameters: dict[str, Any]) -> Any:
    if isinstance(value, str):
        reference = _REFERENCE.fullmatch(value)
        return _resolve_value(_lookup_reference(reference.group(1), parameters), parameters) if reference else value
    if isinstance(value, list):
        return [_resolve_value(item, parameters) for item in value]
    if isinstance(value, dict):
        if "$multiply" in value:
            if set(value) != {"$multiply"} or not isinstance(value["$multiply"], list):
                raise ConfigError("'$multiply' must be the only key and contain a list of numeric values.")
            factors = _resolve_value(value["$multiply"], parameters)
            if not factors or not all(isinstance(factor, (int, float)) for factor in factors):
                raise ConfigError("'$multiply' requires a non-empty list of numeric values.")
            result: int | float = 1
            for factor in factors:
                result *= factor
            return result
        if "$object" in value:
            unexpected = set(value) - {"$object", "args"}
            if unexpected:
                raise ConfigError(f"Unexpected object specification keys: {sorted(unexpected)}.")
            object_type = value["$object"]
            if object_type not in _OBJECT_REGISTRY:
                raise ConfigError(
                    f"Unknown safe object type '{object_type}'.",
                    f"Registered object types: {sorted(_OBJECT_REGISTRY)}.",
                )
            args = _resolve_value(value.get("args", {}), parameters)
            if not isinstance(args, dict):
                raise ConfigError(f"Object args for '{object_type}' must be a mapping.")
            try:
                return _OBJECT_REGISTRY[object_type](**args)
            except (KeyError, TypeError, ValueError) as exc:
                raise ConfigError(f"Invalid arguments for object type '{object_type}': {exc}") from exc
        return {key: _resolve_value(item, parameters) for key, item in value.items()}
    return value


def _normalize_route(value: Any, field: str) -> list[int | str]:
    if value is None:
        return [0]
    if isinstance(value, (int, str)):
        return [value]
    if isinstance(value, list) and all(isinstance(item, (int, str)) for item in value):
        return value
    raise ConfigError(f"Module '{field}' must be an integer/string or a list of integers/strings.")


def _build_single_module(spec: dict[str, Any], parameters: dict[str, Any] | None = None) -> torch.nn.Module:
    """Instantiate one registered module or one nested KonfAI graph."""
    parameters = parameters or {}
    if not isinstance(spec, dict):
        raise ConfigError("Each entry of 'modules' must be a mapping.", f"Got: {type(spec).__name__}.")

    allowed_keys = {
        "alias",
        "args",
        "in_branch",
        "modules",
        "name",
        "out_branch",
        "pretrained",
        "requires_grad",
        "training",
        "type",
    }
    unexpected_keys = sorted(set(spec) - allowed_keys)
    if unexpected_keys:
        raise ConfigError(
            "Unexpected keys in module specification.",
            f"Unexpected keys: {unexpected_keys!r}.",
            f"Allowed keys: {sorted(allowed_keys)}.",
        )

    nested_specs = spec.get("modules")
    if nested_specs is not None:
        if spec.get("type") not in {None, "Graph", "ModuleArgsDict"}:
            raise ConfigError("Nested 'modules' entries may only use type 'Graph' or 'ModuleArgsDict'.")
        if not isinstance(nested_specs, list) or not nested_specs:
            raise ConfigError("A nested 'modules' field must be a non-empty list.")
        graph = YamlModuleGraph()
        _populate_graph(graph, nested_specs, parameters)
        return graph

    type_name = spec.get("type")
    if not isinstance(type_name, str) or not type_name:
        raise ConfigError("A module specification is missing a valid 'type' key.")
    if type_name not in _MODULE_REGISTRY:
        raise ConfigError(
            f"Unknown module type '{type_name}': it is not in the model builder registry.",
            f"Registered types: {list_registered_modules()}.",
        )
    raw_args = spec.get("args", {})
    if not isinstance(raw_args, dict):
        raise ConfigError(f"The 'args' for module type '{type_name}' must be a mapping.")
    args = _resolve_value(raw_args, parameters)
    try:
        return _MODULE_REGISTRY[type_name](**args)
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigError(
            f"Invalid arguments for module type '{type_name}': {exc}",
            f"Provided args: {args}.",
        ) from exc


def _populate_graph(
    graph: ModuleArgsDict,
    module_specs: list[dict[str, Any]],
    parameters: dict[str, Any],
) -> None:
    for index, spec in enumerate(module_specs):
        if not isinstance(spec, dict):
            raise ConfigError(f"Module at index {index} must be a mapping.")
        raw_name = spec.get("name", str(index))
        if not isinstance(raw_name, str) or not raw_name or "." in raw_name:
            raise ConfigError(f"Invalid module name '{raw_name}': names must be non-empty and contain no '.'.")
        if raw_name in graph._modules:
            raise ConfigError(f"Duplicate module name '{raw_name}'.")
        module = _build_single_module(spec, parameters)
        alias = spec.get("alias", [])
        if isinstance(alias, str):
            alias = [alias]
        if not isinstance(alias, list) or not all(isinstance(item, str) for item in alias):
            raise ConfigError("Module 'alias' must be a string or list of strings.")
        graph.add_module(
            raw_name,
            module,
            in_branch=_normalize_route(spec.get("in_branch"), "in_branch"),
            out_branch=_normalize_route(spec.get("out_branch"), "out_branch"),
            pretrained=bool(spec.get("pretrained", True)),
            alias=alias,
            requires_grad=spec.get("requires_grad"),
            training=spec.get("training"),
        )


def _load_definition(yaml_str: str | None, yaml_path: str | Path | None) -> dict[str, Any]:
    if (yaml_str is None) == (yaml_path is None):
        raise ConfigError("build_model_from_yaml requires exactly one of 'yaml_str' or 'yaml_path'.")
    if yaml_path is not None:
        path = Path(yaml_path)
        try:
            yaml_text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ConfigError(f"Could not read model YAML file '{path}'.", str(exc)) from exc
    else:
        yaml_text = yaml_str or ""
    try:
        definition = _yaml.load(yaml_text)
    except ruamel.yaml.YAMLError as exc:
        raise ConfigError("Invalid YAML passed to build_model_from_yaml.", str(exc)) from exc
    if not isinstance(definition, dict):
        raise ConfigError("The model YAML must be a mapping with a top-level 'modules' list.")
    unexpected = sorted(set(definition) - {"modules", "name", "network", "parameters"})
    if unexpected:
        raise ConfigError(f"Unexpected top-level model keys: {unexpected}.")
    return definition


def build_model_from_yaml(
    yaml_str: str | None = None,
    yaml_path: str | Path | None = None,
    parameters: dict[str, Any] | None = None,
    optimizer: OptimizerLoader | None = None,
    schedulers: dict[str, LRSchedulersLoader] | None = None,
    outputs_criterions: dict[str, TargetCriterionsLoader] | None = None,
    patch: ModelPatch | None = None,
) -> YamlNetwork:
    """Build a routed KonfAI network from YAML using ``add_module`` for every graph edge.

    The document accepts ``name``, ``parameters``, ``network`` and ``modules``.
    Module entries accept ``type``, ``args``, ``name``, nested ``modules`` and
    all :meth:`ModuleArgsDict.add_module` routing fields. Exact ``${parameter}``
    references preserve the referenced Python value. Safe constructor values
    such as ``BlockConfig`` use ``{$object: BlockConfig, args: {...}}``.
    """
    definition = _load_definition(yaml_str, yaml_path)
    module_specs = definition.get("modules")
    if not isinstance(module_specs, list) or not module_specs:
        raise ConfigError("The model YAML requires a non-empty top-level 'modules' list.")

    resolved_parameters = copy.deepcopy(definition.get("parameters", {}))
    if not isinstance(resolved_parameters, dict):
        raise ConfigError("The top-level 'parameters' field must be a mapping.")
    if parameters is not None:
        if not isinstance(parameters, dict):
            raise ConfigError("Model parameter overrides must be a mapping.")
        resolved_parameters.update(copy.deepcopy(parameters))

    raw_network = definition.get("network", {})
    if not isinstance(raw_network, dict):
        raise ConfigError("The top-level 'network' field must be a mapping.")
    network_args = _resolve_value(raw_network, resolved_parameters)
    allowed_network_args = {"dim", "in_channels", "init_gain", "init_type", "nb_batch_per_step"}
    unexpected_network_args = sorted(set(network_args) - allowed_network_args)
    if unexpected_network_args:
        raise ConfigError(f"Unexpected network settings: {unexpected_network_args}.")
    network_args.update(
        {
            key: value
            for key, value in {
                "optimizer": optimizer,
                "schedulers": schedulers,
                "outputs_criterions": outputs_criterions,
                "patch": patch,
            }.items()
            if value is not None
        }
    )
    name = definition.get("name", Path(yaml_path).stem if yaml_path is not None else "YamlNetwork")
    if not isinstance(name, str) or not name:
        raise ConfigError("The model 'name' must be a non-empty string.")
    return YamlNetwork(name, module_specs, resolved_parameters, **network_args)
