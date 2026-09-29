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

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from konfai.data.patching import ModelPatch
from konfai.network.blocks import Concat
from konfai.network.network import ModelLoader, Network
from konfai.utils.config import apply_config
from konfai.utils.errors import ConfigError
from konfai.utils.pretrained import PretrainedFrom

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


def _bound_loader(
    write_config, tmp_path: Path, pretrained_block: str, model: str = TINY_MODEL, classpath: str = "Tiny.yml"
) -> ModelLoader:
    """Bind a ModelLoader through the real binder, against a declarative model (Tiny by default)."""
    (tmp_path / "Tiny.yml").write_text(model, encoding="utf-8")
    write_config(f"Root:\n  Model:\n    classpath: {classpath}\n{pretrained_block}", name="Config.yml")

    class Root:
        def __init__(self, model: ModelLoader = ModelLoader()) -> None:
            self.model = model

    return apply_config("Root")(Root)().model


def test_pretrained_from_defaults_to_none_when_the_config_is_silent(write_config, tmp_path: Path) -> None:
    loader = _bound_loader(write_config, tmp_path, "")
    assert loader.pretrained_from is None
    assert loader.get_model(train=True, konfai_args="Root.Model").pretrained_source is None


@pytest.mark.parametrize(
    "wrap", [None, "state_dict", "network_weights"], ids=["raw-state-dict", "checkpoint-dict", "nnunet"]
)
def test_a_fresh_train_load_starts_from_the_reference_weights(write_config, tmp_path: Path, wrap: str | None) -> None:
    """The config route end to end: TRAIN's ``load({}, init=True)`` seeds the graph exactly, from a
    raw state dict, a checkpoint wrapping it under ``state_dict``, or an nnU-Net checkpoint (its
    weights under ``network_weights``, the optimizer and the plans beside them)."""
    reference = torch.nn.Conv2d(1, 2, 3, padding=1)
    state = reference.state_dict()
    if wrap == "state_dict":
        state = {"state_dict": state}
    elif wrap == "network_weights":
        state = {"network_weights": state, "optimizer_state": {}, "init_args": {}, "current_epoch": 3}
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


BRANCHES_REFERENCE = '''
import torch


class Branches(torch.nn.Module):
    """Two same-shaped convolutions on parallel branches, concatenated as [a, b]."""

    def __init__(self, deep_supervision: bool = False) -> None:
        super().__init__()
        self.a = torch.nn.Conv2d(1, 2, 3, padding=1)
        self.b = torch.nn.Conv2d(1, 2, 3, padding=1)
        self.deep_supervision = deep_supervision

    def forward(self, x):
        a = self.a(x)
        output = torch.cat([a, self.b(x)], 1)
        return [output, a] if self.deep_supervision else output
'''


def _branches_model(first: str, second: str) -> str:
    """Branches as a KonfAI graph running ``first`` before ``second``, concatenated as [A, B], under a
    Softmax head the reference does not have."""
    conv = "    args: {dim: 2, in_channels: 1, out_channels: 2, kernel_size: 3, padding: 1}\n"
    return (
        "name: Tiny\nnetwork:\n  in_channels: 1\n  dim: 2\nmodules:\n"
        f"  - name: {first}\n    type: Conv\n    in_branch: [0]\n    out_branch: [{first.lower()}]\n{conv}"
        f"  - name: {second}\n    type: Conv\n    in_branch: [0]\n    out_branch: [{second.lower()}]\n{conv}"
        "  - name: Cat\n    type: Concat\n    in_branch: [a, b]\n"
        "  - name: Softmax\n    type: Softmax\n    args: {dim: 1}\n"
    )


def _branches_loader(
    write_config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model: str, deep_supervision: bool
) -> ModelLoader:
    (tmp_path / "branches_reference.py").write_text(BRANCHES_REFERENCE, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    reference = torch.nn.Module()
    reference.a = torch.nn.Conv2d(1, 2, 3, padding=1)
    reference.b = torch.nn.Conv2d(1, 2, 3, padding=1)
    torch.save(reference.state_dict(), tmp_path / "ref.pt")
    block = (
        f"    pretrained_from:\n      checkpoint: {tmp_path / 'ref.pt'}\n"
        "      builder: branches_reference:Branches\n"
        f"      args:\n        deep_supervision: {str(deep_supervision).lower()}\n"
        "      input_shape: [8, 8]\n"
    )
    return _bound_loader(write_config, tmp_path, block, model=model)


def test_a_transfer_into_branches_run_in_another_order_is_refused(
    write_config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every tensor is filled, but the graph runs B before A while the reference runs a before b:
    execution order pairs B with a and A with b, so the seeded model does not compute the reference."""
    loader = _branches_loader(write_config, tmp_path, monkeypatch, _branches_model("B", "A"), deep_supervision=False)
    net = loader.get_model(train=True, konfai_args="Root.Model")

    with pytest.raises(ConfigError, match=r"'branches_reference:Branches' cannot seed(.|\n)*does not reproduce"):
        net.load({}, init=True)


def test_every_output_the_reference_returns_is_checked(
    write_config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reference returning a list (nnU-Net's deep supervision) is matched output by output against
    the graph's named outputs, whatever head the graph adds on top (here a Softmax)."""
    loader = _branches_loader(write_config, tmp_path, monkeypatch, _branches_model("A", "B"), deep_supervision=True)
    net = loader.get_model(train=True, konfai_args="Root.Model")

    net.load({}, init=True)

    reference = torch.load(tmp_path / "ref.pt")
    assert torch.equal(net["A"].weight, reference["a.weight"]) and torch.equal(net["B"].weight, reference["b.weight"])
    assert all(module.training for module in net.modules())


def test_a_monai_segresnet_seeds_the_catalog_graph(write_config, tmp_path: Path) -> None:
    """The documented pair: MONAI SegResNet into ``default|SegResNet.yml``, whose ArgMax head the
    reference does not have."""
    pytest.importorskip("monai")
    from monai.networks.nets import SegResNet

    args = {"spatial_dims": 3, "init_filters": 8, "in_channels": 1, "out_channels": 2}
    args |= {"blocks_down": [1, 2, 2, 4], "blocks_up": [1, 1, 1]}
    reference = SegResNet(**args).eval()
    torch.save(reference.state_dict(), tmp_path / "ref.pt")
    loader = _bound_loader(
        write_config,
        tmp_path,
        f"    pretrained_from:\n      checkpoint: {tmp_path / 'ref.pt'}\n"
        f"      builder: monai.networks.nets:SegResNet\n      args: {json.dumps(args)}\n",
        classpath="default|SegResNet.yml",
    )
    net = loader.get_model(train=True, konfai_args="Root.Model")

    net.load({}, init=True)

    inputs = torch.randn(1, 1, 16, 16, 16)
    net.eval()
    with torch.no_grad():
        logits = dict(net.named_forward(inputs))["Head.Conv"]
        assert torch.allclose(logits, reference(inputs), atol=1e-5)


class _BranchesNet(Network):
    """Branches as a Python graph running its convolutions in ``order``, concatenated as [A, B]."""

    def __init__(self, order: str, patch: ModelPatch | None) -> None:
        super().__init__(in_channels=1, dim=2, patch=patch)
        for name in order:
            self.add_module(name, torch.nn.Conv2d(1, 2, 3, padding=1), in_branch=[0], out_branch=[name])
        self.add_module("Cat", Concat(), in_branch=["A", "B"], out_branch=[0])


class _Wrapper(Network):
    def __init__(self, inner: Network, dim: int = 2) -> None:
        super().__init__(in_channels=1, dim=dim)
        self.add_module("Inner", inner)


def _traced(net: Network) -> Network:
    """The channel trace Trainer binds before the seed: it marks each network's end modules."""
    net._compute_channels_trace(net, net.in_channels, None, None)
    return net


@pytest.mark.parametrize("nested", [False, True], ids=["root", "nested"])
@pytest.mark.parametrize("order", ["AB", "BA"])
def test_a_tiling_model_patch_is_lifted_for_the_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, nested: bool, order: str
) -> None:
    """A ModelPatch smaller than the input, on the root or a nested network, pads its patch edges and
    assembles only the end modules: the check runs the input whole, so the right order seeds and the
    crossed one is still refused."""
    (tmp_path / "branches_reference.py").write_text(BRANCHES_REFERENCE, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    reference = torch.nn.Module()
    reference.a = torch.nn.Conv2d(1, 2, 3, padding=1)
    reference.b = torch.nn.Conv2d(1, 2, 3, padding=1)
    torch.save(reference.state_dict(), tmp_path / "ref.pt")
    patch = ModelPatch([8, 8])
    branches = _BranchesNet(order, patch)
    net = _traced(_Wrapper(branches) if nested else branches)
    source = PretrainedFrom(
        checkpoint=str(tmp_path / "ref.pt"), builder="branches_reference:Branches", args={}, input_shape=[16, 16]
    )

    if order == "BA":
        with pytest.raises(ConfigError, match=r"does not reproduce"):
            source.seed(net)
    else:
        source.seed(net)
        assert torch.equal(branches["A"].weight, reference.a.weight)
    assert branches.patch is patch
