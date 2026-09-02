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

"""The ``Model.pretrained_from`` config key: seed a model from an external reference checkpoint.

The execution-order bridge (``transfer_weights_by_execution_order``) is reachable from YAML: the
block names a reference class, its constructor arguments and its checkpoint, and a fresh TRAIN
load starts from the transferred weights. A checkpoint's own weights (RESUME, PREDICTION) always
win over the reference, and a non-equivalent reference fails loudly, naming the key.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from konfai.network.network import ModelLoader, Network
from konfai.utils.config import apply_config
from konfai.utils.errors import ConfigError

TINY_MODEL = """
name: Tiny
network:
  in_channels: 1
  dim: 2
modules:
  - name: Conv
    type: Conv
    args:
      dim: 2
      in_channels: 1
      out_channels: 2
      kernel_size: 3
      padding: 1
"""

REFERENCE_ARGS = """
      args:
        in_channels: 1
        out_channels: 2
        kernel_size: 3
        padding: 1
"""


def _bound_loader(write_config, tmp_path: Path, pretrained_block: str) -> ModelLoader:
    """Bind a ModelLoader through the real binder, against the Tiny declarative model."""
    (tmp_path / "Tiny.yml").write_text(TINY_MODEL, encoding="utf-8")
    write_config(f"Root:\n  Model:\n    classpath: Tiny.yml\n{pretrained_block}", name="Config.yml")

    class Root:
        def __init__(self, model: ModelLoader = ModelLoader()) -> None:
            self.model = model

    return apply_config("Root")(Root)().model


def test_pretrained_from_defaults_to_none_when_the_config_is_silent(write_config, tmp_path: Path) -> None:
    loader = _bound_loader(write_config, tmp_path, "")
    assert loader.pretrained_from is None
    assert loader.get_model(train=True, konfai_args="Root.Model").pretrained_source is None


@pytest.mark.parametrize("wrap", [False, True], ids=["raw-state-dict", "checkpoint-dict"])
def test_a_fresh_train_load_starts_from_the_reference_weights(write_config, tmp_path: Path, wrap: bool) -> None:
    """The config route end to end: TRAIN's ``load({}, init=True)`` seeds the graph exactly."""
    reference = torch.nn.Conv2d(1, 2, 3, padding=1)
    state = {"state_dict": reference.state_dict()} if wrap else reference.state_dict()
    torch.save(state, tmp_path / "ref.pt")
    loader = _bound_loader(
        write_config,
        tmp_path,
        f"    pretrained_from:\n      checkpoint: {tmp_path / 'ref.pt'}\n"
        f"      builder: torch.nn:Conv2d\n{REFERENCE_ARGS}"
        "      input_shape: [8, 8]\n",
    )
    net = loader.get_model(train=True, konfai_args="Root.Model")
    assert isinstance(net, Network) and net.pretrained_source is loader.pretrained_from

    net.load({}, init=True)

    assert torch.equal(net["Conv"].weight, reference.weight)
    assert torch.equal(net["Conv"].bias, reference.bias)


def test_the_example_input_is_derived_from_the_models_own_shape(write_config, tmp_path: Path) -> None:
    """Without ``input_shape`` the synthetic input comes from the model's dim/in_channels."""
    reference = torch.nn.Conv2d(1, 2, 3, padding=1)
    torch.save(reference.state_dict(), tmp_path / "ref.pt")
    loader = _bound_loader(
        write_config,
        tmp_path,
        f"    pretrained_from:\n      checkpoint: {tmp_path / 'ref.pt'}\n"
        f"      builder: torch.nn:Conv2d\n{REFERENCE_ARGS}",
    )
    net = loader.get_model(train=True, konfai_args="Root.Model")

    example = loader.pretrained_from._example_input(net)
    assert example.shape == (1, 1, 16, 16)  # batch 1, the model's in_channels, dim-2 spatial

    net.load({}, init=True)
    assert torch.equal(net["Conv"].weight, reference.weight)


def test_a_mismatched_reference_raises_naming_the_key(write_config, tmp_path: Path) -> None:
    """The bridge's strict refusal surfaces as a ConfigError naming ``Model.pretrained_from``."""
    wrong = torch.nn.Conv2d(1, 4, 3, padding=1)  # not weight-exact: 4 output channels against 2
    torch.save(wrong.state_dict(), tmp_path / "ref.pt")
    loader = _bound_loader(
        write_config,
        tmp_path,
        f"    pretrained_from:\n      checkpoint: {tmp_path / 'ref.pt'}\n"
        "      builder: torch.nn:Conv2d\n"
        "      args:\n"
        "        in_channels: 1\n"
        "        out_channels: 4\n"
        "        kernel_size: 3\n"
        "        padding: 1\n",
    )
    net = loader.get_model(train=True, konfai_args="Root.Model")

    with pytest.raises(ConfigError, match=r"pretrained_from"):
        net.load({}, init=True)


def test_a_missing_checkpoint_or_builder_is_refused_by_key(write_config, tmp_path: Path) -> None:
    loader = _bound_loader(write_config, tmp_path, "    pretrained_from:\n      builder: torch.nn:Conv2d\n")
    net = loader.get_model(train=True, konfai_args="Root.Model")

    with pytest.raises(ConfigError, match=r"pretrained_from requires both"):
        net.load({}, init=True)


def test_a_checkpoints_own_weights_always_win_over_the_reference(write_config, tmp_path: Path) -> None:
    """The seed fires only on a fresh load: a ``Model`` entry (RESUME/PREDICTION) or an EMA copy
    (deepcopied from the already-seeded model) never pays or re-runs the transfer."""
    loader = _bound_loader(write_config, tmp_path, "")
    net = loader.get_model(train=True, konfai_args="Root.Model")
    seeded: list[Network] = []
    net.pretrained_source = SimpleNamespace(seed=seeded.append)

    net.load({"Model": {net.get_name(): net.state_dict()}}, init=True)  # a checkpoint load
    assert seeded == []

    net.load({}, init=False, ema=True)  # the EMA copy's load
    assert seeded == []

    net.load({}, init=True)  # the fresh TRAIN load
    assert seeded == [net]
