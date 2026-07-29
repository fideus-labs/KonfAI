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

import asyncio
import io
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

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


@pytest.mark.usefixtures("workspace_root")
def test_config_schema_yaml_keys_and_drill(
    load_mcp_server: Callable[[], ModuleType],
) -> None:
    """The describe_config_schema TOOL surfaces literal YAML keys (yaml_key) and drills by YAML path."""
    fastmcp = pytest.importorskip("fastmcp")
    mcp_server = load_mcp_server()

    async def scenario() -> None:
        async with fastmcp.Client(mcp_server.mcp) as client:
            root = await client.call_tool("describe_config_schema", {"workflow": "train"})
            root_data = root.structured_content
            keys = {field["yaml_key"] for field in root_data["fields"]}
            assert "Model" in keys and "train_name" in keys

            drilled = await client.call_tool("describe_config_schema", {"workflow": "train", "path": "Model"})
            drilled_data = drilled.structured_content
            assert drilled_data["yaml_path"] == ["Trainer", "Model"]
            assert any(field["name"] == "classpath" for field in drilled_data["fields"])

            with pytest.raises(Exception, match="Drillable nested config keys"):
                await client.call_tool("describe_config_schema", {"workflow": "train", "path": "Nope"})

    asyncio.run(scenario())


@pytest.mark.usefixtures("workspace_root")
def test_describe_model_outputs_enumerates_module_paths(
    load_mcp_server: Callable[[], ModuleType],
) -> None:
    fastmcp = pytest.importorskip("fastmcp")
    pytest.importorskip("SimpleITK")
    from mcp_test_helpers import create_segmentation_dataset
    from ruamel.yaml import YAML

    yaml = YAML()
    mcp_server = load_mcp_server()

    async def scenario() -> None:
        async with fastmcp.Client(mcp_server.mcp) as client:
            created = await client.call_tool(
                "initialize_session",
                {"from_example": "Segmentation", "overwrite": True, "workflows": ["train"]},
            )
            session_dir = Path(created.structured_content["path"])
            create_segmentation_dataset(session_dir / "Dataset")
            config = yaml.load((session_dir / "Config.yml").read_text(encoding="utf-8"))
            config["Trainer"]["Dataset"]["Patch"]["patch_size"] = [1, 32, 32]
            config["Trainer"]["Model"]["UNet"]["parameters"]["channels"] = [1, 4, 8, 16, 32]
            stream = io.StringIO()
            yaml.dump(config, stream)
            await client.call_tool(
                "write_workflow_config", {"workflow": "train", "content": stream.getvalue(), "overwrite": True}
            )

            outputs = await client.call_tool("describe_model_outputs", {"workflow": "train"})
            payload = outputs.structured_content
            assert payload["ok"] is True
            networks = payload["networks"]
            assert networks, "at least one Network must be discovered"
            paths = [entry["path"] for entries in networks.values() for entry in entries]
            assert paths and any("." in path for path in paths)
            assert any(entry["terminal"] for entries in networks.values() for entry in entries)

    asyncio.run(scenario())
