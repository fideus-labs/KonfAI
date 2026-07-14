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

"""The shipped YAML model catalog: 'default|<Name>.yml' resolution and catalog health."""

from pathlib import Path

import pytest
from konfai.network.network import ModelLoader, Network
from konfai.utils.errors import ConfigError
from konfai.utils.model_builder import build_model_from_yaml

REPO = Path(__file__).resolve().parents[2]
CATALOG = REPO / "konfai" / "models" / "yaml"


def test_default_classpath_resolves_into_the_shipped_catalog() -> None:
    path = ModelLoader(classpath="default|NestedUNet.yml")._yaml_path()
    assert path == (CATALOG / "NestedUNet.yml").resolve()
    assert path.is_file()


def test_unknown_catalog_model_lists_the_available_ones() -> None:
    with pytest.raises(ConfigError, match=r"NestedUNet\.yml"):
        ModelLoader(classpath="default|DoesNotExist.yml")._yaml_path()


def test_every_catalog_entry_builds() -> None:
    entries = sorted(CATALOG.glob("*.yml"))
    assert entries, "the shipped catalog must not be empty"
    for entry in entries:
        model = build_model_from_yaml(yaml_path=entry)
        assert isinstance(model, Network), entry.name


def test_catalog_unet_stays_in_sync_with_the_example_copy() -> None:
    # The example keeps its own UNet.yml as the authoring walkthrough; the catalog ships the
    # canonical copy. They describe the same architecture and must not drift.
    example = (REPO / "examples" / "Segmentation" / "UNet.yml").read_text(encoding="utf-8")
    catalog = (CATALOG / "UNet.yml").read_text(encoding="utf-8")
    assert example == catalog
