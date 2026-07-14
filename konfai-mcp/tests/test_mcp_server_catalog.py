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

from konfai_mcp.catalog import COMPONENT_KINDS, list_components, normalize_kind  # noqa: E402


def test_list_components_covers_every_kind() -> None:
    for kind in COMPONENT_KINDS:
        payload = list_components(kind)
        assert payload["kind"] == kind
        assert payload["count"] == len(payload["components"])
        assert payload["reference_hint"]
        assert "inspect_object_signature" in payload["next_actions"]
        for component in payload["components"]:
            assert component["config_reference"]


def test_list_components_finds_known_components() -> None:
    criteria = {c["name"] for c in list_components("criterion")["components"]}
    assert {"Dice", "CrossEntropyLoss"} <= criteria

    transforms = {c["name"] for c in list_components("transform")["components"]}
    assert "Standardize" in transforms

    augmentations = {c["name"] for c in list_components("augmentation")["components"]}
    assert "Flip" in augmentations

    model_components = list_components("model")["components"]
    models = {c["config_reference"] for c in model_components}
    assert "segmentation.UNet.UNet" in models
    # The shipped declarative catalog is discoverable alongside the Python model classes.
    assert "default|UNet.yml" in models
    catalog = [c for c in model_components if c.get("kind_detail") == "yaml_catalog"]
    assert all(c["config_reference"].startswith("default|") for c in catalog)
    assert all(c["config_reference"].endswith((".yml", ".yaml")) for c in catalog)

    blocks = {c["name"] for c in list_components("block")["components"]}
    assert "ConvBlock" in blocks


def test_list_components_provides_inspect_classpath_and_aliases() -> None:
    assert list_components("loss")["kind"] == "criterion"
    assert list_components("metric")["kind"] == "criterion"
    assert normalize_kind("Augmentations") == "augmentation"

    dice = next(c for c in list_components("criterion")["components"] if c["name"] == "Dice")
    assert dice["inspect_classpath"] == "konfai.metric.measure:Dice"

    unet = next(c for c in list_components("model")["components"] if c["config_reference"] == "segmentation.UNet.UNet")
    assert unet["inspect_classpath"] == "konfai.models.python.segmentation.UNet:UNet"


def test_list_components_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError, match="Unknown component kind"):
        list_components("widget")
