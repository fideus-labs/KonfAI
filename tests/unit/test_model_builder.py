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

"""Unit tests for the declarative YAML model builder."""

# isort: skip_file
# Import sorting is disabled for this file. Ruff (the project's import sorter)
# groups ``konfai`` with third-party imports for modules outside the package,
# while the pre-commit isort hook wants a separate first-party section.
# They cannot agree on test files, so both are told to skip and the
# ruff-compatible order below is maintained by hand.
import pytest
import torch
from konfai.network.network import ModelLoader, ModuleArgsDict, Network
from konfai.utils import model_builder
from konfai.utils.errors import ConfigError
from konfai.utils.model_builder import (
    build_model_from_yaml,
    list_registered_modules,
    register_module,
)

BUILTIN_TYPES = [
    "ArgMax",
    "AvgPool",
    "Concat",
    "Conv",
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "ConvTranspose",
    "ConvBlock",
    "Identity",
    "MaxPool",
    "ResBlock",
    "Softmax",
]

THREE_MODULE_YAML = """
name: SimpleHead
modules:
  - name: Conv
    type: Conv2d
    args:
      in_channels: 16
      out_channels: 3
      kernel_size: 1
  - name: Softmax
    type: Softmax
    args:
      dim: 1
  - name: Argmax
    type: ArgMax
    args:
      dim: 1
"""

SINGLE_IDENTITY_YAML = "modules:\n  - type: Identity\n"

ROUTED_GRAPH_YAML = """
name: RoutedGraph
parameters:
  dim: 2
  in_channels: 2
  out_channels: 3
network:
  dim: ${dim}
  in_channels: ${in_channels}
modules:
  - name: Encoder
    modules:
      - name: Conv
        type: Conv
        args:
          dim: ${dim}
          in_channels: ${in_channels}
          out_channels: ${out_channels}
          kernel_size: 1
  - name: Preserve
    type: Identity
    out_branch: [1]
  - name: Join
    type: Concat
    in_branch: [0, 1]
"""


class TestBuildModelFromYaml:
    """Behaviour of the public ``build_model_from_yaml`` entry point."""

    def test_three_module_yaml_builds_konfai_network_with_three_children(self) -> None:
        model = build_model_from_yaml(yaml_str=THREE_MODULE_YAML)

        assert isinstance(model, Network)
        assert len(model._modules) == 3
        children = dict(model.items())
        assert list(children) == ["Conv", "Softmax", "Argmax"]
        assert isinstance(children["Conv"], torch.nn.Conv2d)
        assert isinstance(children["Argmax"], model_builder.blocks.ArgMax)
        assert model.name == "SimpleHead"

    def test_three_module_head_runs_a_forward_pass(self) -> None:
        model = build_model_from_yaml(yaml_str=THREE_MODULE_YAML)

        output = model.forward_tensor(torch.randn(2, 16, 4, 4))

        assert output.shape == (2, 1, 4, 4)

    def test_single_identity_module_builds_one_child_network(self) -> None:
        model = build_model_from_yaml(yaml_str=SINGLE_IDENTITY_YAML)

        assert isinstance(model, Network)
        assert list(model.keys()) == ["0"]
        assert isinstance(model["0"], torch.nn.Identity)

    def test_missing_modules_key_raises_config_error(self) -> None:
        with pytest.raises(ConfigError) as excinfo:
            build_model_from_yaml(yaml_str="name: only-a-name\n")

        assert "modules" in str(excinfo.value)

    def test_empty_modules_list_raises_config_error(self) -> None:
        with pytest.raises(ConfigError) as excinfo:
            build_model_from_yaml(yaml_str="modules: []\n")

        assert "empty" in str(excinfo.value).lower()

    def test_requires_exactly_one_of_str_or_path(self) -> None:
        with pytest.raises(ConfigError):
            build_model_from_yaml()
        with pytest.raises(ConfigError):
            build_model_from_yaml(yaml_str=SINGLE_IDENTITY_YAML, yaml_path="model.yml")

    def test_builds_from_a_yaml_file_path(self, tmp_path) -> None:
        path = tmp_path / "model.yml"
        path.write_text(SINGLE_IDENTITY_YAML, encoding="utf-8")

        model = build_model_from_yaml(yaml_path=str(path))

        assert list(model.keys()) == ["0"]

    def test_nested_graph_uses_add_module_routing(self) -> None:
        model = build_model_from_yaml(yaml_str=ROUTED_GRAPH_YAML)

        assert isinstance(model["Encoder"], ModuleArgsDict)
        assert isinstance(model["Encoder"]["Conv"], torch.nn.Conv2d)
        assert model._modulesArgs["Preserve"].out_branch == ["1"]
        assert model._modulesArgs["Join"].in_branch == ["0", "1"]
        output = model.forward_tensor(torch.randn(2, 2, 8, 8))
        assert output.shape == (2, 6, 8, 8)

    def test_safe_block_config_object_builds_conv_block(self) -> None:
        yaml_str = """
parameters:
  dim: 2
modules:
  - name: Block
    type: ConvBlock
    args:
      in_channels: 1
      out_channels: 4
      dim: ${dim}
      block_configs:
        - $object: BlockConfig
          args:
            kernel_size: 3
            padding: 1
            activation: ReLU
"""
        model = build_model_from_yaml(yaml_str=yaml_str)

        assert isinstance(model["Block"], model_builder.blocks.ConvBlock)
        assert model.forward_tensor(torch.randn(1, 1, 8, 8)).shape == (1, 4, 8, 8)

    def test_unknown_parameter_reference_raises_config_error(self) -> None:
        yaml_str = "modules:\n  - type: Conv\n    args:\n      dim: ${missing}\n"

        with pytest.raises(ConfigError, match="missing"):
            build_model_from_yaml(yaml_str=yaml_str)

    def test_malformed_yaml_raises_config_error(self) -> None:
        with pytest.raises(ConfigError):
            build_model_from_yaml(yaml_str="modules: [unclosed\n")

    def test_unknown_module_spec_key_raises_config_error(self) -> None:
        yaml_str = "modules:\n  - type: Identity\n    argz: {}\n"

        with pytest.raises(ConfigError) as excinfo:
            build_model_from_yaml(yaml_str=yaml_str)

        message = str(excinfo.value)
        assert "unexpected" in message.lower()
        assert "argz" in message


class TestUnknownModuleType:
    """Errors raised when a YAML ``type`` is not in the registry."""

    UNKNOWN_YAML = "modules:\n  - type: UnknownXyz\n"

    def test_unknown_type_mentions_the_registry(self) -> None:
        with pytest.raises(ConfigError) as excinfo:
            build_model_from_yaml(yaml_str=self.UNKNOWN_YAML)

        message = str(excinfo.value)
        assert "registry" in message or "registered" in message

    def test_unknown_type_lists_the_valid_types(self) -> None:
        with pytest.raises(ConfigError) as excinfo:
            build_model_from_yaml(yaml_str=self.UNKNOWN_YAML)

        message = str(excinfo.value)
        assert str(list_registered_modules()) in message
        assert "Conv2d" in message


class TestInvalidArgs:
    """Errors raised when constructor arguments are invalid."""

    def test_unexpected_softmax_argument_names_the_type(self) -> None:
        # ``torch.nn.Softmax`` does not validate the value of ``dim`` at
        # construction time, so an unrecognised keyword is used to trigger the
        # constructor ``TypeError`` that the builder wraps in ``ConfigError``.
        yaml_str = "modules:\n  - type: Softmax\n    args:\n      not_a_param: bad\n"

        with pytest.raises(ConfigError) as excinfo:
            build_model_from_yaml(yaml_str=yaml_str)

        assert "Softmax" in str(excinfo.value)

    def test_bad_conv_argument_value_names_the_type(self) -> None:
        yaml_str = (
            "modules:\n"
            "  - type: Conv2d\n"
            "    args:\n"
            "      in_channels: bad\n"
            "      out_channels: 4\n"
            "      kernel_size: 3\n"
        )

        with pytest.raises(ConfigError) as excinfo:
            build_model_from_yaml(yaml_str=yaml_str)

        assert "Conv2d" in str(excinfo.value)

    def test_non_mapping_args_names_the_type(self) -> None:
        yaml_str = "modules:\n  - type: Softmax\n    args: [1, 2, 3]\n"

        with pytest.raises(ConfigError) as excinfo:
            build_model_from_yaml(yaml_str=yaml_str)

        assert "Softmax" in str(excinfo.value)


class TestRegistryAPI:
    """The public registry helpers ``register_module`` / ``list_registered_modules``."""

    def test_register_module_adds_a_new_type(self) -> None:
        name = "RegistryTestLayer"
        assert name not in list_registered_modules()

        register_module(name, torch.nn.ReLU)
        try:
            assert name in list_registered_modules()
            model = build_model_from_yaml(yaml_str=f"modules:\n  - type: {name}\n")
            assert isinstance(model["0"], torch.nn.ReLU)
        finally:
            model_builder._MODULE_REGISTRY.pop(name, None)

    def test_register_duplicate_name_raises(self) -> None:
        with pytest.raises(ConfigError) as excinfo:
            register_module("Identity", torch.nn.Identity)

        assert "registered" in str(excinfo.value).lower()
        # The original built-in registration must remain untouched.
        assert "Identity" in list_registered_modules()

    def test_register_non_module_raises(self) -> None:
        with pytest.raises(ConfigError):
            register_module("NotAModule", int)

        assert "NotAModule" not in list_registered_modules()

    @pytest.mark.parametrize("invalid_name", ["", 123, []], ids=["empty", "non-string", "unhashable"])
    def test_register_invalid_name_raises_config_error(self, invalid_name) -> None:
        registered_before = list_registered_modules()

        with pytest.raises(ConfigError) as excinfo:
            register_module(invalid_name, torch.nn.Identity)

        assert "invalid name" in str(excinfo.value).lower()
        assert list_registered_modules() == registered_before

    def test_list_registered_modules_is_sorted_with_builtins(self) -> None:
        registered = list_registered_modules()

        assert registered == sorted(registered)
        for builtin in BUILTIN_TYPES:
            assert builtin in registered


class TestYamlModelLoader:
    def test_loads_relative_yaml_model_and_config_parameter_overrides(self, tmp_path, monkeypatch) -> None:
        model_path = tmp_path / "Tiny.yml"
        model_path.write_text(
            """
name: Tiny
parameters:
  channels: [1, 2]
network:
  in_channels: ${channels.0}
  dim: 2
modules:
  - name: Conv
    type: Conv
    args:
      dim: 2
      in_channels: ${channels.0}
      out_channels: ${channels.1}
      kernel_size: 1
""",
            encoding="utf-8",
        )
        config_path = tmp_path / "Config.yml"
        config_path.write_text(
            "Root:\n  Model:\n    Tiny:\n      parameters:\n        channels: [3, 5]\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("KONFAI_config_file", str(config_path))
        monkeypatch.setenv("KONFAI_CONFIG_MODE", "Done")

        model = ModelLoader("Tiny.yml").get_model(konfai_args="Root.Model")

        assert isinstance(model, Network)
        assert model.in_channels == 3
        assert isinstance(model["Conv"], torch.nn.Conv2d)
        assert model["Conv"].out_channels == 5
