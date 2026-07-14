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

from konfai_mcp.extensions import EXTENSION_POINTS, check_external_dependency, describe_extension_points  # noqa: E402
from konfai_mcp.server_support import summarize_classpath_signature  # noqa: E402


def test_describe_extension_points_lists_all_kinds() -> None:
    payload = describe_extension_points()
    assert set(payload["kinds"]) == set(EXTENSION_POINTS)
    assert "external_library" in payload["yaml_reference_syntax"]
    # The external classpath syntax (the thing the rest of the surface hides) must be surfaced.
    assert ":" in payload["yaml_reference_syntax"]["external_library"]


def test_describe_extension_points_single_kind_and_aliases() -> None:
    loss = describe_extension_points("loss")["extension_point"]
    assert loss["base_class"] == "konfai.metric.measure:Criterion"
    assert describe_extension_points("losses")["kind"] == "loss"
    assert describe_extension_points("network")["kind"] == "model"
    # The MinimalModel gotcha must be surfaced for models.
    assert "MinimalModel" in describe_extension_points("model")["extension_point"]["gotcha"]


def test_describe_extension_points_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError, match="Unknown extension kind"):
        describe_extension_points("widget")


def test_check_external_dependency_reports_installed_and_missing() -> None:
    torch_dep = check_external_dependency("torch")
    assert torch_dep["installed"] is True
    assert torch_dep["version"]
    assert torch_dep["is_konfai_dependency"] is True
    assert torch_dep["install_hint"] is None

    # A package that will never exist: the missing-report shape must not depend on what the
    # local dev env happens to carry (monai, for instance, is installed locally as a test oracle).
    missing = check_external_dependency("nonexistent_konfai_test_pkg.losses", "DiceLoss")
    assert missing["installed"] is False
    assert missing["install_hint"] == "pip install nonexistent_konfai_test_pkg"
    assert missing["is_konfai_dependency"] is False


def test_inspect_distinguishes_konfai_component_from_foreign_class() -> None:
    dice = summarize_classpath_signature("konfai.metric.measure:Dice", workspace_dir=MODULE_ROOT)
    assert dice["ok"] is True
    assert dice["konfai_base"] == "criterion"
    assert "konfai.metric.measure.Criterion" in dice["bases"]
    assert dice["forward"] and dice["forward"].startswith("forward(")

    l1 = summarize_classpath_signature("torch:nn:L1Loss", workspace_dir=MODULE_ROOT)
    assert l1["ok"] is True
    assert l1["konfai_base"] is None
    assert l1["forward"] and "input" in l1["forward"]
    assert l1["integration_hint"]
