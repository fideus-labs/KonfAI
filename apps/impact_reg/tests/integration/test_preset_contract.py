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

"""The published presets, read at the revision this package resolves, against this package's models.

A preset is config on Hugging Face and code here, released separately: a key the model does not take is only a
warning at run time (the predictor binds with ``strict_config(refuse=False)``), so a preset edited ahead of the
package, or a parameter renamed in the package, ran with the default in its place. Every key of every prediction
config must be a parameter of the model class it names, nested model specs included; every preset must write the
``Transform`` group the orchestrator collects, and ship the configs its tiling names.

Gated like the engine tests (``IMPACT_REG_ENGINE_TESTS=1``): it reads the presets from the network.
"""

import inspect
import json
import os
import typing
from importlib import import_module
from pathlib import Path

import pytest
from ruamel.yaml import YAML

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("IMPACT_REG_ENGINE_TESTS"),
        reason="set IMPACT_REG_ENGINE_TESTS=1 (reads the published presets from the network)",
    ),
]


def _presets() -> Path:
    """The preset folders this package resolves: KONFAI_IMPACTREG_REPO's directory, or the pinned revision on HF."""
    from impact_reg_konfai import PRESETS_REPO

    if Path(PRESETS_REPO).is_dir():
        return Path(PRESETS_REPO)
    from huggingface_hub import snapshot_download

    repo, _, revision = PRESETS_REPO.partition("@")
    patterns = ["*/app.json", "*/*.yml", "*/*.txt"]
    return Path(snapshot_download(repo, revision=revision or None, allow_patterns=patterns))  # nosec B615


def _unknown_keys(cls: type, block: dict, path: str) -> list[str]:
    """The keys of ``block`` that ``cls`` does not take, or whose value its declared type refuses (the check a
    ``--set`` gets), recursing into its ``dict[str, <spec class>]`` parameters."""
    from konfai_apps.app_repository import _check_value
    from konfai_apps.errors import AppRepositoryError

    parameters = inspect.signature(cls.__init__).parameters
    hints = typing.get_type_hints(cls.__init__, include_extras=True)
    unknown = []
    for key, value in block.items():
        if key not in parameters:
            unknown.append(f"{path}.{key}")
            continue
        try:
            _check_value(value, hints.get(key, inspect.Parameter.empty), f"{path}.{key}")
        except AppRepositoryError as error:
            unknown.append(str(error).strip())
            continue
        hint = typing.get_type_hints(cls.__init__).get(key)
        origin, arguments = typing.get_origin(hint), typing.get_args(hint)
        if origin is dict and isinstance(value, dict) and inspect.isclass(arguments[1]):
            for name, entry in value.items():
                if isinstance(entry, dict):
                    unknown += _unknown_keys(arguments[1], entry, f"{path}.{key}.{name}")
    return unknown


def test_every_preset_binds_to_the_models_of_this_package() -> None:
    root, problems, checked = _presets(), [], 0
    for app_json in sorted(root.glob("*/app.json")):
        preset, manifest = app_json.parent, json.loads(app_json.read_text(encoding="utf-8"))
        if manifest.get("task") != "registration":
            continue
        checked += 1
        tiling = manifest.get("tiling") or {}
        missing = [
            name for name in (tiling.get("global"), tiling.get("tile")) if name and not (preset / name).is_file()
        ]
        problems += [f"{preset.name}: tiling names {name}, which the preset lacks" for name in missing]
        for config in sorted(preset.glob("Prediction*.yml")):
            where, predictor = f"{preset.name}/{config.name}", YAML(typ="safe").load(config.read_text())["Predictor"]
            module, _, name = predictor["Model"]["classpath"].partition(":")
            unknown = _unknown_keys(getattr(import_module(module), name), predictor["Model"][name], name)
            problems += [f"{where}: {key} is not a parameter of {module}:{name}" for key in unknown]
            groups = {spec.get("group") for output in predictor["outputs_dataset"].values() for spec in output.values()}
            if groups != {"Transform"}:
                problems.append(f"{where}: writes {sorted(map(str, groups))}, not the Transform group")
    assert checked, f"no registration preset under {root}"
    assert not problems, "\n".join(problems)
