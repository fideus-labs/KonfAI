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

"""Enumerate the KonfAI component zoo so an agent can discover what exists.

KonfAI references most components by class name (criteria/transforms/augmentations/schedulers)
or by a dotted path under ``konfai.models`` (models), and YAML-model-builder blocks by registry
key. ``inspect_object_signature`` can then reveal a component's constructor, but only if the agent
already knows the classpath -- this module supplies that missing discovery step.

Heavy ``konfai`` imports happen inside the functions (lazily), so importing this module is cheap.
"""

from __future__ import annotations

import importlib
import inspect
import os
from typing import Any

# Kinds backed by "concrete subclasses of a base class defined in a single module".
_SUBCLASS_KINDS: dict[str, tuple[str, str]] = {
    "criterion": ("konfai.metric.measure", "Criterion"),
    "transform": ("konfai.data.transform", "Transform"),
    "augmentation": ("konfai.data.augmentation", "DataAugmentation"),
    "scheduler": ("konfai.metric.schedulers", "Scheduler"),
}

_KIND_ALIASES: dict[str, str] = {
    "loss": "criterion",
    "losses": "criterion",
    "metric": "criterion",
    "metrics": "criterion",
    "criteria": "criterion",
    "transforms": "transform",
    "augmentations": "augmentation",
    "schedulers": "scheduler",
    "models": "model",
    "blocks": "block",
}

COMPONENT_KINDS = ["criterion", "transform", "augmentation", "scheduler", "model", "block"]

_REFERENCE_HINTS = {
    "criterion": (
        "Reference by name under a criterion's criterions_loader (losses) or under metrics. "
        "Whether a Criterion behaves as a loss or a metric depends on its constructor "
        "(is_loss) / return type -- call inspect_object_signature for the exact contract."
    ),
    "transform": "Reference by name under a group's 'transforms' or 'patch_transforms'.",
    "augmentation": "Reference by name under a DataAugmentation_* 'data_augmentations' block.",
    "scheduler": "Reference by name under a criterion's 'schedulers' (weight scheduling over iterations).",
    "model": (
        "Use config_reference as Trainer/Predictor Model.classpath (e.g. 'segmentation.UNet.UNet'), "
        "or write a declarative .yml model instead."
    ),
    "block": "Use the name as a module 'type' inside a .yml model definition's 'modules' list.",
}


def _doc_summary(obj: Any) -> str | None:
    # Use the object's OWN docstring (``__doc__`` is not inherited), so a component without
    # its own docstring reports None rather than its base class's docstring.
    own = getattr(obj, "__doc__", None)
    if not own or not own.strip():
        return None
    line = inspect.cleandoc(own).splitlines()[0].strip()
    return line[:200] or None


def normalize_kind(kind: str) -> str:
    canonical = _KIND_ALIASES.get(kind.strip().lower(), kind.strip().lower())
    if canonical not in COMPONENT_KINDS:
        raise ValueError(
            f"Unknown component kind '{kind}'. Expected one of: {', '.join(COMPONENT_KINDS)} "
            "(aliases: loss/metric -> criterion, transforms -> transform, etc.)."
        )
    return canonical


def _requires_callable_argument(cls: Any) -> bool:
    """True if the class needs a non-defaulted Callable argument, so it cannot be built from YAML.

    Base helpers like MaskedLoss take an injected ``loss: Callable`` -- listing them as usable components
    misleads an agent into referencing something the reflection engine can never instantiate.
    """
    try:
        parameters = list(inspect.signature(cls.__init__).parameters.values())[1:]
    except (TypeError, ValueError):
        return False
    return any(
        parameter.default is inspect.Parameter.empty and "Callable" in str(parameter.annotation)
        for parameter in parameters
    )


def _list_subclasses(module_path: str, base_name: str) -> list[dict[str, Any]]:
    module = importlib.import_module(module_path)
    base = getattr(module, base_name)
    components: list[dict[str, Any]] = []
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if obj is base or not issubclass(obj, base):
            continue
        if inspect.isabstract(obj) or name.startswith("_"):
            continue
        if not obj.__module__.startswith("konfai"):
            continue
        if _requires_callable_argument(obj):
            continue
        components.append(
            {
                "name": name,
                "config_reference": name,
                "inspect_classpath": f"{obj.__module__}:{name}",
                "module": obj.__module__,
                "doc": _doc_summary(obj),
            }
        )
    return sorted(components, key=lambda component: component["name"])


def _list_blocks() -> list[dict[str, Any]]:
    builder = importlib.import_module("konfai.utils.model_builder")
    components: list[dict[str, Any]] = []
    for attr, role in (("_MODULE_REGISTRY", "module"), ("_OBJECT_REGISTRY", "object")):
        registry = getattr(builder, attr, {})
        for key, factory in registry.items():
            components.append(
                {
                    "name": key,
                    "config_reference": key,
                    "role": role,
                    "doc": _doc_summary(factory),
                }
            )
    return sorted(components, key=lambda component: component["name"])


def model_config_reference_to_inspect_classpath(config_reference: str) -> str | None:
    """Map a builtin-model ``config_reference`` to its importable ``inspect_classpath``.

    Inverse of the mapping :func:`_list_models` emits: a model is listed with
    ``config_reference='<rel>.<Class>'`` and ``inspect_classpath='konfai.models.python.<rel>:<Class>'``
    (the builtin Python models live under ``konfai/models/python/``; ``<rel>`` alone is not
    importable). Returns ``None`` when the reference has no ``<module>.<Class>`` split (e.g. a bare
    criterion name).
    """
    rel, _, name = config_reference.rpartition(".")
    if not rel or not name:
        return None
    return f"konfai.models.python.{rel}:{name}"


def _list_models() -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    # The builtin Python models live under konfai/models/python (no __init__.py, namespace-style);
    # walk it the way ModelLoader resolves a model classpath: konfai.models.python.<dotted>.<Class>,
    # keeping the agent-facing config_reference in its short '<task>.<Module>.<Class>' form.
    models_pkg = importlib.import_module("konfai.models.python")
    network_base = importlib.import_module("konfai.network.network").Network
    components: list[dict[str, Any]] = []
    unavailable: list[dict[str, str]] = []
    seen: set[str] = set()
    for root in list(getattr(models_pkg, "__path__", [])):
        for dirpath, _dirs, files in os.walk(root):
            for filename in sorted(files):
                if not filename.endswith(".py") or filename.startswith("_"):
                    continue
                rel = os.path.relpath(os.path.join(dirpath, filename), root)[: -len(".py")].replace(os.sep, ".")
                module_name = f"konfai.models.python.{rel}"
                try:
                    module = importlib.import_module(module_name)
                except Exception as exc:  # optional deps / import-time failures -> report, don't crash
                    unavailable.append({"module": module_name, "reason": f"{type(exc).__name__}: {exc}"})
                    continue
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if obj.__module__ != module_name or name.startswith("_"):
                        continue
                    if obj is network_base or not issubclass(obj, network_base) or inspect.isabstract(obj):
                        continue
                    classpath = f"{rel}.{name}"
                    if classpath in seen:
                        continue
                    seen.add(classpath)
                    components.append(
                        {
                            "name": name,
                            "config_reference": classpath,
                            "inspect_classpath": model_config_reference_to_inspect_classpath(classpath),
                            "module": module_name,
                            "doc": _doc_summary(obj),
                        }
                    )
    components.extend(_list_yaml_catalog_models())
    return sorted(components, key=lambda component: component["config_reference"]), unavailable


def _list_yaml_catalog_models() -> list[dict[str, Any]]:
    """Enumerate the shipped declarative model catalog (konfai/models/yaml/*.yml).

    Each entry is referenced as ``classpath: default|<Name>.yml`` — the declarative counterpart of a
    Python model classpath. The leading comment lines of the file are surfaced as its doc so an agent
    can pick an architecture without opening the file.
    """
    import konfai.models.yaml as yaml_catalog

    catalog_dir = os.path.dirname(str(yaml_catalog.__file__))
    entries: list[dict[str, Any]] = []
    for filename in sorted(os.listdir(catalog_dir)):
        if not filename.endswith((".yml", ".yaml")):
            continue
        path = os.path.join(catalog_dir, filename)
        doc_lines: list[str] = []
        try:
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    stripped = line.strip()
                    if stripped.startswith("#"):
                        doc_lines.append(stripped.lstrip("# ").rstrip())
                    elif stripped:
                        break
        except OSError:
            continue
        entries.append(
            {
                "name": filename.rsplit(".", 1)[0],
                "config_reference": f"default|{filename}",
                "kind_detail": "yaml_catalog",
                "doc": " ".join(doc_lines).strip() or None,
            }
        )
    return entries


def list_components(kind: str) -> dict[str, Any]:
    """Enumerate KonfAI components of one kind that can be referenced in a YAML config."""
    canonical = normalize_kind(kind)
    unavailable: list[dict[str, str]] = []
    if canonical == "block":
        components = _list_blocks()
    elif canonical == "model":
        components, unavailable = _list_models()
    else:
        module_path, base_name = _SUBCLASS_KINDS[canonical]
        components = _list_subclasses(module_path, base_name)

    payload: dict[str, Any] = {
        "kind": canonical,
        "count": len(components),
        "components": components,
        "reference_hint": _REFERENCE_HINTS[canonical],
        "next_actions": ["inspect_object_signature", "design_config_strategy", "write_workflow_config"],
    }
    if unavailable:
        payload["unavailable_modules"] = unavailable
    return payload
