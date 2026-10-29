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

"""JSON Schemas of the four workflow configs, read off the constructors the binder reads.

An editor that speaks JSON Schema (VS Code's YAML extension) then completes keys, shows defaults and
flags a wrong type or an unknown key while the config is written. The schema follows the binder: a
nested object sits under its ``_key``, a chain (``transforms``, ``criterions_loader``, ...) is keyed
by the names of the shipped components, each with its own parameters, and ``None`` is accepted
wherever the binder takes it. What only resolves at run time (a model's own arguments, a torch
optimizer's, a class named by ``module:Class``) is left open: the schema never refuses a config the
binder accepts.
"""

from __future__ import annotations

import difflib
import functools
import importlib
import inspect
import re
import types
from collections.abc import Sequence
from enum import Enum
from pathlib import Path
from typing import Any, Literal, NamedTuple, Union, get_args, get_origin

from konfai.utils.config import Config, Range, _admits_none, _resolve_annotation, _unwrap_optional
from konfai.utils.errors import ConfigError

#: Each workflow's config file, root key and root class.
WORKFLOWS: dict[str, tuple[str, str]] = {
    "Config.yml": ("Trainer", "konfai.trainer:Trainer"),
    "Prediction.yml": ("Predictor", "konfai.predictor:Predictor"),
    "Evaluation.yml": ("Evaluator", "konfai.evaluator:Evaluator"),
    "Transform.yml": ("Transformer", "konfai.transformer:Transformer"),
}

_NONE: dict[str, Any] = {"anyOf": [{"type": "null"}, {"const": "None"}]}
_OPEN: dict[str, Any] = {"anyOf": [{"type": "object"}, {"type": "null"}, {"const": "None"}]}
# A scalar may come quoted: the binder coerces "3", "1e-3" and "false" the way it coerces 3, 0.001 and false.
_NUMBER = r"^\s*[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?\s*$|^\s*[-+]?(inf|infinity|nan)\s*$"
_SCALARS: dict[type, dict[str, Any]] = {
    int: {"anyOf": [{"type": "integer"}, {"type": "string", "pattern": r"^\s*[-+]?\d+\s*$"}]},
    float: {"anyOf": [{"type": "number"}, {"type": "string", "pattern": _NUMBER}]},
    bool: {
        "anyOf": [
            {"type": "boolean"},
            {"enum": [0, 1]},
            {
                "type": "string",
                "pattern": r"^\s*([Tt]rue|TRUE|[Ff]alse|FALSE|[Yy]es|YES|[Nn]o|NO|[Oo]n|ON|[Oo]ff|OFF|0|1)\s*$",
            },
        ]
    },
    str: {"type": ["string", "number", "boolean"]},
}


def _class(classpath: str) -> type:
    module, name = classpath.split(":")
    return getattr(importlib.import_module(module), name)


def _binds_in_place(cls: type) -> bool:
    """Whether ``cls`` binds another class at its own level (``IMPACTReg`` binds the loss its ``loss`` names),
    whose keys then sit beside its own."""
    try:
        return "KONFAI_CONFIG_PATH" in inspect.getsource(cls)
    except (OSError, TypeError):
        return False


def _nullable(schema: dict[str, Any]) -> dict[str, Any]:
    return {"anyOf": [schema, _NONE]}


class _SchemaBuilder:
    """Builds one workflow's schema; each class becomes one entry of ``definitions``."""

    def __init__(self) -> None:
        from konfai.data.augmentation.base import Prob
        from konfai.data.transform.base import TransformLoader
        from konfai.evaluator import CriterionsLoader as MetricsLoader
        from konfai.network.network.loaders import (
            CriterionsAttr,
            LossSchedulersLoader,
            LRSchedulersLoader,
            OptimizerLoader,
        )
        from konfai.network.network.model import ModelLoader
        from konfai.predictor.output import OutputDataset, OutputDatasetLoader, OutputLayerDataset

        self.definitions: dict[str, Any] = {}
        # A chain is keyed by component names: the kinds its names resolve in, and the class whose own
        # parameters sit beside each component's (``Prob``'s ``prob``, ``CriterionsAttr``'s ``is_loss``).
        self.chains: dict[type, tuple[tuple[str, ...], type | None]] = {
            TransformLoader: (("transform", "augmentation"), None),
            Prob: (("augmentation",), Prob),
            CriterionsAttr: (("criterion",), CriterionsAttr),
            LossSchedulersLoader: (("scheduler",), LossSchedulersLoader),
            LRSchedulersLoader: (("lr_scheduler",), LRSchedulersLoader),
        }
        # The evaluator's criteria bind under a dict[str, Any]: their parameters still come from the file.
        self.chain_parameters: dict[tuple[type, str], tuple[str, ...]] = {
            (MetricsLoader, "criterions_loader"): ("criterion",),
        }
        self.model_loader = ModelLoader
        # Blocks that also hold keys their class does not declare: a torch optimizer's arguments, a custom sink's.
        self.open_blocks = {OptimizerLoader, OutputDatasetLoader}
        # The classes bound in the same block as the key's own (the sink ``name_class`` names).
        self.shared_blocks: dict[type, tuple[type, ...]] = {
            OutputDatasetLoader: (OutputDataset, OutputLayerDataset),
        }

    def value(self, annotation: object) -> dict[str, Any]:
        """The schema of a value the binder coerces (never a nested config object)."""
        metadata: tuple = ()
        if hasattr(annotation, "__metadata__"):
            annotation, *rest = get_args(annotation)
            metadata = tuple(rest)
        annotation, optional = _unwrap_optional(annotation)
        schema = self._value(annotation)
        for hint in metadata:
            if isinstance(hint, Range):
                schema = {**schema, "minimum": hint.min, "maximum": hint.max}
        return _nullable(schema) if optional else schema

    def _value(self, annotation: object) -> dict[str, Any]:
        origin, args = get_origin(annotation), get_args(annotation)
        if annotation in _SCALARS:
            return _SCALARS[annotation]  # type: ignore[index]
        if annotation is Path:
            return _SCALARS[str]
        if origin is Literal:
            return {"enum": [*args, *(str(arg) for arg in args if not isinstance(arg, str))]}
        if origin in {Union, types.UnionType}:
            members = [self.value(arg) for arg in args if arg is not type(None)]
            return {"anyOf": members + ([_NONE] if type(None) in args else [])}
        if origin in {list, Sequence} or (origin is tuple and (not args or args[-1] is Ellipsis)):
            return {"type": "array", "items": self.value(args[0])} if args else {"type": "array"}
        if origin is tuple:
            return {
                "type": "array",
                "items": [self.value(arg) for arg in args],
                "minItems": len(args),
                "maxItems": len(args),
            }
        if origin is dict:
            return {"anyOf": [{"type": "object", "additionalProperties": self.value(args[1])}, _NONE]}
        return {}  # Any, a tensor, or a type the binder passes through

    def parameters(
        self, owner: type, skip: int = 0, exclude: Sequence[str] = ()
    ) -> tuple[dict[str, Any], list[str], bool]:
        """The properties ``owner``'s constructor binds, those the binder refuses to miss (no default, no
        ``None``), and whether it also takes ``**kwargs``."""
        properties: dict[str, Any] = {}
        required: list[str] = []
        open_ = False
        try:
            signature = inspect.signature(owner)
        except (TypeError, ValueError):
            return properties, required, True
        for parameter in list(signature.parameters.values())[skip:]:
            if parameter.kind is parameter.VAR_KEYWORD:
                open_ = True
                continue
            if parameter.kind is parameter.VAR_POSITIONAL or parameter.name in exclude:
                continue
            try:
                annotation = _resolve_annotation(owner, parameter.annotation, owner.__name__, parameter.name)
            except Exception:
                annotation = Any
            bare = get_args(annotation)[0] if hasattr(annotation, "__metadata__") else annotation
            bare, optional = _unwrap_optional(bare)
            if self._is_object(bare):
                key = getattr(bare, "_key", None)
                if key is None:  # bound at this same level
                    nested, nested_required, nested_open = self.parameters(bare)
                    properties.update(nested)
                    required += nested_required
                    open_ = open_ or nested_open
                else:
                    properties[key] = _nullable(self.reference(bare))
                continue
            if get_origin(bare) is dict and self._is_object(get_args(bare)[1]):
                properties[parameter.name] = self.mapping(get_args(bare)[1])
            elif (owner, parameter.name) in self.chain_parameters:
                properties[parameter.name] = self.chain(self.chain_parameters[owner, parameter.name], None, lists=False)
            else:
                properties[parameter.name] = self.value(annotation)
                if (
                    parameter.default is parameter.empty
                    and parameter.annotation is not parameter.empty
                    and not optional
                    and not _admits_none(bare)
                ):
                    required.append(parameter.name)
            default = self._default(parameter.default)
            if default is not None:
                properties[parameter.name] = {**properties[parameter.name], "default": default}
        return properties, required, open_

    @staticmethod
    def _is_object(annotation: object) -> bool:
        """What the binder instantiates from a block (``_bind_config_object``), not a value it coerces."""
        if not inspect.isclass(annotation) or annotation in _SCALARS or annotation in {Path, object}:
            return False
        return get_origin(annotation) is None and annotation.__module__.startswith("konfai")

    @staticmethod
    def _default(default: object) -> object:
        if default is inspect.Parameter.empty or default is None:
            return None
        if isinstance(default, Enum):
            return default.value
        if isinstance(default, str):
            return Config._default_value(default)
        if isinstance(default, bool | int | float):
            return default
        if isinstance(default, list | tuple) and all(isinstance(item, bool | int | float | str) for item in default):
            return list(default)
        return None

    def reference(self, cls: type) -> dict[str, Any]:
        name = f"{cls.__module__}.{cls.__qualname__}"
        if name not in self.definitions:
            self.definitions[name] = {}  # reserved first: a class may reach itself
            if cls is self.model_loader:
                self.definitions[name] = self.model_block()
            else:
                properties, required, open_ = self.parameters(cls)
                for shared in self.shared_blocks.get(cls, ()):
                    properties.update(self.parameters(shared)[0])
                self.definitions[name] = self._object(properties, required, open_ or cls in self.open_blocks, cls)
        return {"$ref": f"#/definitions/{name}"}

    @staticmethod
    def _object(properties: dict[str, Any], required: list[str], open_: bool, cls: type) -> dict[str, Any]:
        schema: dict[str, Any] = {"type": "object", "properties": properties}
        if required:
            schema["required"] = required
        if not open_:
            schema["additionalProperties"] = False
        doc = inspect.getdoc(cls)
        if doc:
            schema["description"] = doc.split("\n\n")[0]
        return schema

    def model_block(self) -> dict[str, Any]:
        """``Model:``: the loader's own keys, then one block named after the network, whose arguments are the
        network's (a YAML model's ``parameters``, its optimizer, schedulers and criteria)."""
        properties, required, _ = self.parameters(self.model_loader)
        network_properties, _, _ = self.parameters(
            _class("konfai.utils.model_builder:YamlNetwork"), exclude=("name", "module_specs")
        )
        schema = self._object(properties, required, True, self.model_loader)
        schema["additionalProperties"] = _nullable({"type": "object", "properties": network_properties})
        return schema

    def mapping(self, value_type: type) -> dict[str, Any]:
        """A ``dict[str, X]`` of objects: free keys (group names) or, for a chain, component names."""
        if value_type in self.chains:
            kinds, base = self.chains[value_type]
            return self.chain(kinds, base, lists=True)
        entry = _nullable(self.reference(value_type))
        key = getattr(value_type, "_key", None)
        if key is not None:  # each entry's object sits under the class key
            entry = _nullable({"type": "object", "properties": {key: entry}, "additionalProperties": False})
        return _nullable({"type": "object", "additionalProperties": entry})

    def chain(self, kinds: tuple[str, ...], base: type | None, lists: bool) -> dict[str, Any]:
        name = f"chain:{':'.join(kinds)}:{base.__qualname__ if base else ''}"
        if name not in self.definitions:
            self.definitions[name] = {}
            base_properties, base_required, _ = self.parameters(base) if base is not None else ({}, [], False)
            entries: dict[str, list[dict[str, Any]]] = {}
            for component, skip in self._components(kinds):
                properties, required, open_ = self.parameters(component, skip=skip)
                entries.setdefault(component.__name__, []).append(
                    self._object(
                        {**base_properties, **properties},
                        base_required + required,
                        open_ or _binds_in_place(component),
                        component,
                    )
                )
            # Each component's schema once in `definitions`: its name and its occurrence keys (`Clip#2`) share it.
            properties = {}
            for key, schemas in entries.items():
                self.definitions[f"{name}.{key}"] = _nullable(schemas[0] if len(schemas) == 1 else {"anyOf": schemas})
                properties[key] = {"$ref": f"#/definitions/{name}.{key}"}
            # A bare name, or its occurrence key, must be a shipped one; a `module:Class` is left open.
            occurrence = f"^(?:{'|'.join(re.escape(key) for key in sorted(properties))})#[0-9]+$"
            names = {"anyOf": [{"enum": sorted(properties)}, {"pattern": ":"}, {"pattern": occurrence}]}
            mapping = {
                "type": "object",
                "properties": properties,
                "patternProperties": {f"^{re.escape(key)}#[0-9]+$": ref for key, ref in properties.items()},
                "propertyNames": names,
                "additionalProperties": _OPEN,
            }
            alternatives = [mapping, _NONE]
            if lists:  # the list spelling of a chain: one stage per item, a bare name or a single-entry mapping
                item = {"anyOf": [{"type": "string", **names}, {**mapping, "minProperties": 1, "maxProperties": 1}]}
                alternatives.insert(1, {"type": "array", "items": item})
            self.definitions[name] = {"anyOf": alternatives}
        return {"$ref": f"#/definitions/{name}"}

    @staticmethod
    def _components(kinds: tuple[str, ...]) -> list[tuple[type, int]]:
        """The classes a chain's bare names resolve to, with the positional arguments the engine passes first."""
        from konfai.utils.catalog import list_components

        classes: list[tuple[type, int]] = []
        for kind in kinds:
            if kind == "lr_scheduler":  # resolved in torch then in KonfAI, and handed the optimizer first
                import torch.optim.lr_scheduler as torch_schedulers

                for module in (torch_schedulers, importlib.import_module("konfai.metric.schedulers")):
                    classes += [
                        (obj, 1)
                        for name, obj in inspect.getmembers(module, inspect.isclass)
                        if issubclass(obj, torch_schedulers.LRScheduler)
                        and obj is not torch_schedulers.LRScheduler
                        and not name.startswith("_")
                    ]
                continue
            for component in list_components(kind):
                if component.module is not None:
                    classes.append((getattr(importlib.import_module(component.module), component.name), 0))
        return list(dict.fromkeys(classes))  # a module may re-export a class another one defines


def config_schema(filename: str) -> dict[str, Any]:
    """The JSON Schema (draft-07) of one workflow config: ``Config.yml``, ``Prediction.yml``,
    ``Evaluation.yml`` or ``Transform.yml``."""
    root, classpath = WORKFLOWS[filename]
    builder = _SchemaBuilder()
    root_schema = builder.reference(_class(classpath))
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": f"KonfAI {filename}",
        "type": "object",
        "properties": {root: root_schema},
        "required": [root],
        "additionalProperties": False,
        "definitions": builder.definitions,
    }


class ConfigMistake(NamedTuple):
    """One mistake of a config: the dotted path of the key it is about, what is wrong, and whether it is a key
    nothing reads (the engine warns about it, except TRANSFORM, which refuses it) rather than one it refuses."""

    path: str
    message: str
    unread: bool


def config_errors(tree: Any, filename: str) -> list[ConfigMistake]:
    """Every mistake of a config against its schema, each at its key (``Trainer.Dataset.batch_size``), with the
    nearest valid name for a misspelled one. An empty list when the config is valid. ``tree`` is the YAML as
    loaded."""
    leaves = [leaf for error in _validator(filename).iter_errors(tree) for leaf in _leaves(error)]
    mistakes = {
        ConfigMistake(".".join(map(str, leaf.absolute_path)), _message(leaf), leaf.validator == "additionalProperties")
        for leaf in leaves
    }
    return sorted(mistakes)


@functools.cache
def _validator(filename: str) -> Any:
    try:
        from jsonschema import Draft7Validator
    except ImportError:
        raise ConfigError(
            "Checking a config against its schema needs jsonschema.", "pip install 'konfai[check]'"
        ) from None
    return Draft7Validator(config_schema(filename))


def _leaves(error: Any) -> list[Any]:
    """The mistakes under an ``anyOf`` that failed: those of the alternative the value was written for (a
    mapping is a block, not ``None``), not one message for the whole block."""
    if not error.context or not isinstance(error.instance, dict | list):
        return [error]
    branches: dict[Any, list[Any]] = {}
    for sub in error.context:
        branches.setdefault(sub.relative_schema_path[0], []).append(sub)
    branch = max(
        branches.values(),
        key=lambda subs: (
            any(sub.validator not in {"type", "const"} for sub in subs),
            max(len(sub.absolute_path) for sub in subs),
        ),
    )
    return [leaf for sub in branch for leaf in _leaves(sub)]


_KINDS = {
    "integer": "an integer",
    "number": "a number",
    "boolean": "true or false",
    "array": "a list",
    "object": "a mapping",
    "string": "a string",
    "null": "None",
}


def _message(error: Any) -> str:
    instance, schema = error.instance, error.schema
    if error.validator == "additionalProperties":
        known = schema.get("properties", {})
        patterns = schema.get("patternProperties", {})
        unknown = [key for key in instance if key not in known and not any(re.search(p, key) for p in patterns)]
        return "; ".join(f"unknown key '{key}'{_nearest(key, known)}" for key in unknown)
    if error.validator == "required":
        return "; ".join(f"missing key '{key}'" for key in schema["required"] if key not in instance)
    names = _names(schema)
    if isinstance(instance, str) and names:
        return f"unknown '{instance}'{_nearest(instance, names)}"
    kinds = list(dict.fromkeys(_kinds(schema)))
    return f"{instance!r} is not {' or '.join(kinds)}" if kinds else error.message


def _names(schema: dict[str, Any]) -> list[Any]:
    """The names a value may take, however deep the alternatives that list them."""
    return [*schema.get("enum", []), *(name for option in schema.get("anyOf", []) for name in _names(option))]


def _kinds(schema: dict[str, Any]) -> list[str]:
    if "anyOf" in schema:
        return [kind for option in schema["anyOf"] for kind in _kinds(option)]
    if "const" in schema:
        return [repr(schema["const"])]
    kind = schema.get("type")
    if kind == "string" and "pattern" in schema:  # a quoted number or bool: the unquoted kind already says it
        return []
    return [_KINDS.get(item, item) for item in ([kind] if isinstance(kind, str) else kind or [])]


def _nearest(word: Any, names: Any) -> str:
    match = difflib.get_close_matches(str(word), [str(name) for name in names], n=1, cutoff=0.5)
    return f" (did you mean '{match[0]}'?)" if match else ""
