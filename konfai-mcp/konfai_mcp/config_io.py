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

"""KonfAI YAML config IO: parse/validate/serialize config content and the static
lint for silent-failure traps, plus the validated config write. Split out of
``server_support.py``."""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML

YAML_SAFE = YAML(typ="safe")
YAML_DUMP = YAML()
YAML_DUMP.default_flow_style = False
YAML_DUMP.width = 4096


def validate_yaml_content(content: str, filename: str, expected_root: str | None = None) -> dict[str, Any]:
    """Parse YAML content and optionally enforce one top-level root key."""
    # A fresh parser per call: a ruamel.yaml instance keeps mutable state, so a shared one is unsafe under
    # concurrent (SSE/HTTP) requests.
    try:
        data = YAML(typ="safe").load(content)
    except Exception as exc:  # pragma: no cover - parser exceptions vary
        raise ValueError(f"{filename} is not valid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{filename} must contain a YAML mapping at the top level.")
    if expected_root is not None and expected_root not in data:
        raise ValueError(f"{filename} must define the '{expected_root}' root key.")
    return data


def yaml_dump_content(data: dict[str, Any]) -> str:
    """Serialize a mapping into normalized YAML content."""
    emitter = YAML()  # fresh per call: the shared emitter's mutable state is unsafe under concurrency
    emitter.default_flow_style = False
    emitter.width = 4096
    stream = StringIO()
    emitter.dump(data, stream)
    return stream.getvalue()


def _lint_config_data(data: Any) -> list[dict[str, str]]:
    """Static lint for silent-failure traps an agent cannot see from the schema alone."""
    warnings: list[dict[str, str]] = []
    # Evaluator groups_dest entries bind to GroupTransformMetric, which has no patch_transforms
    # parameter (and no -1 fill) -- the trap only exists for Trainer/Predictor datasets.
    if isinstance(data, dict) and isinstance(data.get("Evaluator"), dict):
        return warnings

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            groups_dest = node.get("groups_dest")
            if isinstance(groups_dest, dict):
                for dest_name, dest in groups_dest.items():
                    if isinstance(dest, dict) and "patch_transforms" not in dest:
                        warnings.append(
                            {
                                "severity": "warning",
                                "code": "missing_patch_transforms",
                                "path": f"groups_dest.{dest_name}",
                                "message": (
                                    f"groups_dest entry '{dest_name}' omits 'patch_transforms'. Omitting it makes "
                                    "KonfAI fill this group's tensor with -1 (a segmentation target then crashes "
                                    "CrossEntropyLoss with 'Target -1 out of bounds'; a regression target is "
                                    "silently corrupted). Set 'patch_transforms: None' explicitly."
                                ),
                            }
                        )
            for value in node.values():
                _walk(value)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(data)
    warnings.extend(_lint_prediction_default_outputs_criterions(data))
    return warnings


_DEFAULT_OUTPUTS_CRITERIONS_GROUP = "Labels"


def _contains_key(node: Any, target: str) -> bool:
    """True when ``target`` appears as a mapping key anywhere in the nested structure."""
    if isinstance(node, dict):
        if target in node:
            return True
        return any(_contains_key(value, target) for value in node.values())
    if isinstance(node, list):
        return any(_contains_key(item, target) for item in node)
    return False


def _predictor_dest_groups(predictor: dict[str, Any]) -> set[str]:
    """Destination groups the Predictor dataset loads (groups_src.<src>.groups_dest keys)."""
    groups: set[str] = set()
    dataset = predictor.get("Dataset")
    if not isinstance(dataset, dict):
        return groups
    groups_src = dataset.get("groups_src")
    if isinstance(groups_src, dict):
        for src in groups_src.values():
            if isinstance(src, dict) and isinstance(src.get("groups_dest"), dict):
                groups.update(str(name) for name in src["groups_dest"])
    return groups


def _classpath_targets_yaml_model(model: dict[str, Any]) -> bool:
    """True when Model.classpath points at a .yml/.yaml model builder.

    The YAML model builder defaults ``outputs_criterions`` to None (no injection), so the
    default-'Labels' binding only happens for Python model classes.
    """
    classpath = model.get("classpath")
    if not isinstance(classpath, str):
        return False
    raw = classpath.split("|", maxsplit=1)[-1]
    return raw.rsplit(".", maxsplit=1)[-1].lower() in {"yml", "yaml"}


def _lint_prediction_default_outputs_criterions(data: Any) -> list[dict[str, str]]:
    """Prediction Model without an explicit outputs_criterions may bind a default referencing 'Labels'.

    When the Predictor Model block omits ``outputs_criterions``, KonfAI's reflection binds the model
    class's own ``__init__`` default. Builtin models (and plain modules wrapped in ``MinimalModel``)
    default to a ``TargetCriterionsLoader`` targeting group 'Labels', so their Measure init raises
    MeasureError when no 'Labels' group is loaded; a custom Network (or a composite like a GAN whose
    sub-networks own the parameter) may default differently, hence severity=warning, not error.
    """
    if not isinstance(data, dict):
        return []
    predictor = data.get("Predictor")
    if not isinstance(predictor, dict):
        return []
    model = predictor.get("Model")
    if not isinstance(model, dict):
        return []
    if _classpath_targets_yaml_model(model):
        return []
    if _contains_key(model, "outputs_criterions"):
        return []
    loaded = _predictor_dest_groups(predictor)
    if _DEFAULT_OUTPUTS_CRITERIONS_GROUP in loaded:
        return []
    loaded_desc = ", ".join(sorted(loaded)) if loaded else "none"
    return [
        {
            "severity": "warning",
            "code": "prediction_default_outputs_criterions",
            "path": "Predictor.Model",
            "message": (
                "The Predictor Model omits 'outputs_criterions'. For model classes exposing that "
                "parameter (all builtin models do), KonfAI's reflection binds its default, which "
                f"references target group '{_DEFAULT_OUTPUTS_CRITERIONS_GROUP}'. This dataset's "
                f"groups_dest loads [{loaded_desc}] and not '{_DEFAULT_OUTPUTS_CRITERIONS_GROUP}', so "
                "prediction raises MeasureError when the model initializes its Measure. Set "
                "'outputs_criterions: {}' on the (sub-)network block that exposes it to disable it "
                "for pure inference."
            ),
        }
    ]


def write_config(path: Path, content: str, overwrite: bool, expected_root: str | None = None) -> dict[str, Any]:
    """Validate and write one KonfAI YAML config file."""
    data = validate_yaml_content(content, path.name, expected_root=expected_root)
    if path.exists() and not overwrite:
        raise ValueError(f"{path.name} already exists. Set overwrite=True to replace.")
    path.write_text(content.rstrip() + "\n", encoding="utf-8")
    result: dict[str, Any] = {
        "path": str(path),
        "bytes": path.stat().st_size,
    }
    lint = _lint_config_data(data)
    if lint:
        result["lint"] = lint
    return result
