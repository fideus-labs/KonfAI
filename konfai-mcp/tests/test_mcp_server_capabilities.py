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

import sys
from pathlib import Path

import pytest

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from konfai_mcp.capabilities import describe_config_schema, describe_konfai_capabilities  # noqa: E402


def test_describe_konfai_capabilities_is_a_router_not_a_workflow() -> None:
    payload = describe_konfai_capabilities()
    assert "AGENTS.md" in payload["canonical_reference"]
    assert payload["components"]["discover"] == "list_components(kind)"
    assert "monai.losses:DiceLoss" in payload["extension_model"]["external_libraries"]
    # Surfaces the safe vs human-confirmation boundary.
    assert payload["safe_actions"]
    assert payload["risky_actions_prefer_human_confirmation"]


def test_describe_config_schema_is_generated_from_the_reflection_engine() -> None:
    schema = describe_config_schema("train")
    assert schema["root_key"] == "Trainer"
    assert schema["classpath"] == "konfai.trainer:Trainer"
    names = {field["name"] for field in schema["fields"]}
    assert {"model", "dataset", "train_name", "epochs"} <= names

    by_name = {field["name"]: field for field in schema["fields"]}
    # Nested config objects expose a classpath to drill into; their default is not a noisy object repr.
    assert by_name["model"]["nested_config_classpath"] == "konfai.network.network:ModelLoader"
    assert by_name["model"]["default"] is None
    # Scalar defaults are surfaced.
    assert by_name["train_name"]["default"] == "default|TRAIN_01"
    # No field default leaks a runtime object repr (volatile memory address).
    assert all("object at 0x" not in str(field["default"]) for field in schema["fields"])


def test_describe_config_schema_drills_into_optional_nested_configs() -> None:
    # `patch: DatasetPatch | None` and `early_stopping: EarlyStopping | None` are OPTIONAL nested
    # @config objects; the drill must unwrap the Optional so the advertised `path='Dataset.Patch'` works.
    patch = describe_config_schema("train", path="Dataset.Patch")
    assert patch["yaml_path"] == ["Trainer", "Dataset", "Patch"]
    assert "patch_size" in {field["name"] for field in patch["fields"]}

    es = describe_config_schema("train", path="early_stopping")
    assert es["yaml_path"] == ["Trainer", "EarlyStopping"]

    # A dict-valued field (augmentations) is legitimately not a single drillable level, and the error
    # must list the REAL drillable keys (Patch), never a misleading 'none'.
    with pytest.raises(ValueError, match="Drillable nested config keys here: \\['Patch'\\]"):
        describe_config_schema("train", path="Dataset.Augmentation")


def test_describe_config_schema_covers_all_workflows_and_rejects_unknown() -> None:
    assert describe_config_schema("prediction")["root_key"] == "Predictor"
    assert describe_config_schema("evaluation")["root_key"] == "Evaluator"
    assert describe_config_schema("training")["workflow"] == "train"  # alias
    with pytest.raises(ValueError, match="Unknown workflow"):
        describe_config_schema("inference")
