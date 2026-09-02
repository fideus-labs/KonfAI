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

"""Enumerate the component vocabulary a KonfAI config can reference.

The config is the product surface, so its vocabulary must be discoverable without reading the
source: :func:`list_components` returns, per kind, every shipped component with the exact spelling
a YAML config references it by and the first line of its own documentation. The CLI surfaces it as
``konfai list <kind>``; :func:`konfai.api.list_components` re-exports it for Python callers.

The component families import torch and the imaging stack, so everything heavy is imported inside
the functions: importing this module stays cheap.
"""

from __future__ import annotations

import importlib
import inspect
import os
from dataclasses import dataclass
from typing import Any

from konfai.utils.errors import ConfigError

#: Kinds backed by "concrete subclasses of one base class, re-exported by one package".
_SUBCLASS_KINDS: dict[str, tuple[str, str]] = {
    "transform": ("konfai.data.transform", "Transform"),
    "augmentation": ("konfai.data.augmentation", "DataAugmentation"),
    "criterion": ("konfai.metric.measure", "Criterion"),
    "reduction": ("konfai.data.reduction", "Reduction"),
}

COMPONENT_KINDS: tuple[str, ...] = (*_SUBCLASS_KINDS, "model", "block")

_KIND_ALIASES: dict[str, str] = {
    "transforms": "transform",
    "augmentations": "augmentation",
    "criteria": "criterion",
    "loss": "criterion",
    "losses": "criterion",
    "metric": "criterion",
    "metrics": "criterion",
    "reductions": "reduction",
    "models": "model",
    "blocks": "block",
}


@dataclass(frozen=True)
class Component:
    """One referenceable component of the shipped catalog."""

    #: The class (or block/file) name.
    name: str
    #: The exact spelling a YAML config references it by: a bare class name for transforms,
    #: augmentations, criteria and reductions; a ``Model.classpath`` value for models
    #: (``segmentation.UNet.UNet`` or ``default|UNet.yml``); a module ``type`` for builder blocks.
    config_reference: str
    #: The importable module defining it (``None`` for a declarative catalog file or a block).
    module: str | None
    #: The first line of its own docstring (``None`` when it carries none).
    doc: str | None


def normalize_kind(kind: str) -> str:
    """The canonical kind for ``kind``, accepting the plural/synonym spellings."""
    canonical = _KIND_ALIASES.get(kind.strip().lower(), kind.strip().lower())
    if canonical not in COMPONENT_KINDS:
        raise ConfigError(
            f"Unknown component kind '{kind}'.",
            f"Expected one of: {', '.join(COMPONENT_KINDS)} (plural spellings are accepted).",
        )
    return canonical


def _doc_summary(obj: Any) -> str | None:
    # The object's OWN docstring (``__doc__`` on the dict, not inherited): a component without one
    # reports None rather than its base class's documentation.
    own = obj.__dict__.get("__doc__") if isinstance(obj, type) else getattr(obj, "__doc__", None)
    if not own or not own.strip():
        return None
    return inspect.cleandoc(own).splitlines()[0].strip() or None


def _requires_callable_argument(cls: type) -> bool:
    # A base helper taking an injected ``loss: Callable`` (MaskedLoss) cannot be built from YAML:
    # listing it would advertise a spelling the reflection engine refuses.
    try:
        parameters = list(inspect.signature(cls).parameters.values())
    except (TypeError, ValueError):
        return False
    return any(
        parameter.default is inspect.Parameter.empty and "Callable" in str(parameter.annotation)
        for parameter in parameters
    )


def _list_subclasses(module_path: str, base_name: str) -> list[Component]:
    module = importlib.import_module(module_path)
    base = getattr(module, base_name)
    components = []
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if obj is base or not issubclass(obj, base):
            continue
        if inspect.isabstract(obj) or name.startswith("_") or not obj.__module__.startswith("konfai"):
            continue
        if _requires_callable_argument(obj):
            continue
        components.append(Component(name=name, config_reference=name, module=obj.__module__, doc=_doc_summary(obj)))
    return sorted(components, key=lambda component: component.name)


def _list_blocks() -> list[Component]:
    from konfai.utils.model_builder import registered_module_types, registered_object_types

    registries = {**registered_object_types(), **registered_module_types()}
    return sorted(
        (
            Component(name=name, config_reference=name, module=None, doc=_doc_summary(factory))
            for name, factory in registries.items()
        ),
        key=lambda component: component.name,
    )


def _list_python_models() -> list[Component]:
    # The builtin Python models live under konfai/models/python (PEP 420, no __init__.py): walk the
    # files the way ModelLoader resolves a classpath, listing each Network subclass under the short
    # '<task>.<Module>.<Class>' spelling. A module whose optional dependency is missing cannot be
    # referenced either, so it is skipped rather than reported.
    models_pkg = importlib.import_module("konfai.models.python")
    network_base = importlib.import_module("konfai.network.network").Network
    components: dict[str, Component] = {}
    for root in list(getattr(models_pkg, "__path__", [])):
        for dirpath, _dirs, files in os.walk(root):
            for filename in sorted(files):
                if not filename.endswith(".py") or filename.startswith("_"):
                    continue
                rel = os.path.relpath(os.path.join(dirpath, filename), root)[: -len(".py")].replace(os.sep, ".")
                module_name = f"konfai.models.python.{rel}"
                try:
                    module = importlib.import_module(module_name)
                except Exception:  # nosec B112 - a catalog model needing an absent optional dep is not an error
                    continue
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if obj.__module__ != module_name or name.startswith("_"):
                        continue
                    if obj is network_base or not issubclass(obj, network_base) or inspect.isabstract(obj):
                        continue
                    classpath = f"{rel}.{name}"
                    components.setdefault(
                        classpath,
                        Component(name=name, config_reference=classpath, module=module_name, doc=_doc_summary(obj)),
                    )
    return list(components.values())


def _list_yaml_catalog_models() -> list[Component]:
    # The declarative catalog (konfai/models/yaml): each file is referenced as 'default|<Name>.yml',
    # and its leading comment lines are its documentation.
    import konfai.models.yaml as yaml_catalog

    catalog_dir = os.path.dirname(str(yaml_catalog.__file__))
    components = []
    for filename in sorted(os.listdir(catalog_dir)):
        if not filename.endswith((".yml", ".yaml")):
            continue
        doc_lines: list[str] = []
        with open(os.path.join(catalog_dir, filename), encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped.startswith("#"):
                    doc_lines.append(stripped.lstrip("# ").rstrip())
                elif stripped:
                    break
        components.append(
            Component(
                name=filename.rsplit(".", 1)[0],
                config_reference=f"default|{filename}",
                module=None,
                doc=" ".join(doc_lines).strip() or None,
            )
        )
    return components


def list_components(kind: str) -> list[Component]:
    """Every shipped component of one ``kind``, with the spelling a YAML config references it by.

    ``kind`` is one of :data:`COMPONENT_KINDS` (plural spellings accepted): ``transform``,
    ``augmentation``, ``criterion`` (losses and metrics), ``reduction``, ``model`` (the Python
    catalog classpaths and the ``default|<Name>.yml`` declarative catalog), or ``block`` (the YAML
    model builder's registered types).
    """
    canonical = normalize_kind(kind)
    if canonical == "block":
        return _list_blocks()
    if canonical == "model":
        models = _list_python_models() + _list_yaml_catalog_models()
        return sorted(models, key=lambda component: component.config_reference)
    return _list_subclasses(*_SUBCLASS_KINDS[canonical])
