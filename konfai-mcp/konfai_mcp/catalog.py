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

The listing is core's (:mod:`konfai.utils.catalog`, what ``konfai list`` prints): the kinds and the
components come from there. This module adds what an agent needs on top of it: the classpath
``inspect_object_signature`` takes, where each kind goes in a config, and the model modules that failed
to import.

Heavy ``konfai`` imports happen inside the functions (lazily), so importing this module is cheap.
"""

from __future__ import annotations

import importlib
from typing import Any

from konfai.utils import catalog
from konfai.utils.errors import ConfigError

from konfai_mcp.classpaths import public_module

COMPONENT_KINDS = list(catalog.COMPONENT_KINDS)

#: Qualified name of each KonfAI extension base -> the component kind it provides.
COMPONENT_BASES: dict[str, str] = {
    **{f"{module}.{base}": kind for kind, (module, base) in catalog.SUBCLASS_KINDS.items()},
    "konfai.network.network.Network": "model",
}

_REFERENCE_HINTS = {
    "criterion": (
        "Reference by name under a criterion's criterions_loader (losses) or under metrics. "
        "Whether a Criterion behaves as a loss or a metric depends on its constructor "
        "(is_loss) / return type: call inspect_object_signature for the exact contract."
    ),
    "transform": "Reference by name under a group's 'transforms' or 'patch_transforms'.",
    "augmentation": "Reference by name under a DataAugmentation_* 'data_augmentations' block.",
    "reduction": (
        "Reference by name as a Predictor's 'combine' (ensemble), an output's 'reduction' (test-time "
        "augmentation copies), or a Reduce transform's 'operator' (cases folded into one)."
    ),
    "scheduler": "Reference by name under a criterion's 'schedulers' (weight scheduling over iterations).",
    "model": (
        "Use config_reference as Trainer/Predictor Model.classpath (e.g. 'segmentation.UNet.UNet'), "
        "or write a declarative .yml model instead."
    ),
    "block": "Use the name as a module 'type' inside a .yml model definition's 'modules' list.",
}


def normalize_kind(kind: str) -> str:
    try:
        return catalog.normalize_kind(kind)
    except ConfigError:
        raise ValueError(
            f"Unknown component kind '{kind}'. Expected one of: {', '.join(COMPONENT_KINDS)} "
            "(aliases: loss/metric -> criterion, transforms -> transform, etc.)."
        ) from None


def model_config_reference_to_inspect_classpath(config_reference: str) -> str | None:
    """Map a builtin-model ``config_reference`` to its importable ``inspect_classpath``.

    A model is listed with ``config_reference='<rel>.<Class>'`` and
    ``inspect_classpath='konfai.models.python.<rel>:<Class>'`` (the builtin Python models live under
    ``konfai/models/python/``; ``<rel>`` alone is not importable). Returns ``None`` when the reference has
    no ``<module>.<Class>`` split (e.g. a bare criterion name).
    """
    rel, _, name = config_reference.rpartition(".")
    if not rel or not name:
        return None
    return f"konfai.models.python.{rel}:{name}"


def _entry(kind: str, component: catalog.Component, module_types: set[str]) -> dict[str, Any]:
    entry: dict[str, Any] = {"name": component.name, "config_reference": component.config_reference}
    if kind == "block":
        entry["role"] = "module" if component.name in module_types else "object"
    elif component.module is None:
        entry["kind_detail"] = "yaml_catalog"
    elif kind == "model":
        entry["inspect_classpath"] = model_config_reference_to_inspect_classpath(component.config_reference)
        entry["module"] = component.module
    else:
        module = public_module(getattr(importlib.import_module(component.module), component.name))
        entry["inspect_classpath"] = f"{module}:{component.name}"
        entry["module"] = module
    entry["doc"] = component.doc
    return entry


def list_components(kind: str) -> dict[str, Any]:
    """Enumerate KonfAI components of one kind that can be referenced in a YAML config."""
    canonical = normalize_kind(kind)
    unavailable: dict[str, str] = {}
    module_types: set[str] = set()
    if canonical == "model":
        components, unavailable = catalog.list_models()
    else:
        components = catalog.list_components(canonical)
    if canonical == "block":
        from konfai.utils.model_builder import registered_module_types

        module_types = set(registered_module_types())

    payload: dict[str, Any] = {
        "kind": canonical,
        "count": len(components),
        "components": [_entry(canonical, component, module_types) for component in components],
        "reference_hint": _REFERENCE_HINTS[canonical],
        "next_actions": ["inspect_object_signature", "design_config_strategy", "write_workflow_config"],
    }
    if unavailable:
        payload["unavailable_modules"] = [
            {"module": module, "reason": reason} for module, reason in unavailable.items()
        ]
    return payload
