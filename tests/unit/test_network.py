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

"""Tests for ``konfai.network.network``: ModuleArgsDict branch routing and init,
Network.load_state_dict, Measure (loss records, backward, scheduler selection),
and CriterionsLoader."""

from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

import konfai.network.network as network_module
import numpy as np
import pytest
import torch
from konfai.metric.schedulers import Constant
from konfai.network.blocks import Add
from konfai.network.network import Measure, ModuleArgsDict, Network
from konfai.utils.dataset import Attribute
from konfai.utils.errors import ConfigError

# --------------------------------------------------------------------------------------
# ModuleArgsDict branch routing (named_forward / forward)
# --------------------------------------------------------------------------------------


class _MulConst(torch.nn.Module):
    """Deterministic test module: multiplies its input by a fixed constant."""

    def __init__(self, factor: float) -> None:
        super().__init__()
        self.factor = factor

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor * self.factor


class _TwoInputGraph(ModuleArgsDict):
    """A(in 0)→branch 0, B(in 1)→branch 1, Sum(in 0,1)→branch 2."""

    def __init__(self) -> None:
        super().__init__()
        self.add_module("A", _MulConst(3.0), in_branch=[0], out_branch=[0])
        self.add_module("B", _MulConst(10.0), in_branch=[1], out_branch=[1])
        self.add_module("Sum", Add(), in_branch=[0, 1], out_branch=[2])


class _Inner(ModuleArgsDict):
    def __init__(self) -> None:
        super().__init__()
        self.add_module("Scale", _MulConst(2.0))


class _NestedGraph(ModuleArgsDict):
    def __init__(self) -> None:
        super().__init__()
        self.add_module("Pre", _MulConst(5.0), in_branch=[0], out_branch=[0])
        self.add_module("Block", _Inner(), in_branch=[0], out_branch=[0])


def test_forward_routes_two_inputs_through_branches():
    graph = _TwoInputGraph()
    a = torch.ones(1, 1, 2, 2)
    b = torch.full((1, 1, 2, 2), 2.0)
    out = graph(a, b)  # 3*a + 10*b = 3 + 20 = 23
    assert torch.allclose(out, torch.full_like(out, 23.0))


def test_named_forward_exposes_every_intermediate():
    graph = _TwoInputGraph()
    a = torch.ones(1, 1, 2, 2)
    b = torch.full((1, 1, 2, 2), 2.0)
    outputs = {name: float(tensor.flatten()[0]) for name, tensor in graph.named_forward(a, b)}
    assert outputs == {"A": 3.0, "B": 20.0, "Sum": 23.0}


def test_named_forward_uses_dotted_names_for_nested_graphs():
    graph = _NestedGraph()
    x = torch.ones(1, 1, 2, 2)
    names = [name for name, _ in graph.named_forward(x)]
    assert "Pre" in names
    assert "Block.Scale" in names  # nested submodule addressable by dotted path
    out = graph(x)  # 5 then *2 = 10
    assert torch.allclose(out, torch.full_like(out, 10.0))


def test_out_branch_isolation_preserves_a_branch_for_later_use():
    """A branch written by one module must remain available to a later consumer."""

    class _SkipGraph(ModuleArgsDict):
        def __init__(self) -> None:
            super().__init__()
            # Keep the raw input on branch 1, transform branch 0, then combine.
            self.add_module("Identity", torch.nn.Identity(), in_branch=[0], out_branch=[1])
            self.add_module("Scale", _MulConst(4.0), in_branch=[0], out_branch=[0])
            self.add_module("Sum", Add(), in_branch=[0, 1], out_branch=[0])

    graph = _SkipGraph()
    x = torch.ones(1, 1, 2, 2)
    out = graph(x)  # 4*x + x = 5
    assert torch.allclose(out, torch.full_like(out, 5.0))


class _AddConst(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor + self.value


def _nested_adder(value: float, inner_out: int | str) -> ModuleArgsDict:
    sub = ModuleArgsDict()
    sub.add_module("L", _AddConst(value), in_branch=[0], out_branch=[inner_out])
    return sub


def test_later_nested_sibling_output_reaches_downstream() -> None:
    # M1 writes branch 0 via inner-match; M2 shares out_branch=[0] but its inner module writes a
    # different branch, so it relies on the fallback. A ``tmp`` kept across siblings made the fallback
    # see branch 0 as already filled (by M1) and silently drop M2, leaving M1's value downstream.
    graph = ModuleArgsDict()
    graph.add_module("M1", _nested_adder(1.0, 0), in_branch=[0], out_branch=[0])
    graph.add_module("M2", _nested_adder(10.0, "zz"), in_branch=[0], out_branch=[0])
    graph.add_module("Id", torch.nn.Identity(), in_branch=[0], out_branch=[0])

    outputs = list(graph.named_forward(torch.zeros(1)))
    downstream = [tensor for name, tensor in outputs if name.startswith("Id")][-1]

    # Branch 0: input 0 -> M1 (+1) = 1 -> M2 reads branch 0 (+10) = 11 -> Id. Not M1's stale 1.
    assert downstream.item() == 11.0


def test_init_func_centres_batchnorm_gamma_on_one() -> None:
    # gamma initialised around 0 scaled the normalised activations to ~0, stalling early training.
    batch_norm = torch.nn.BatchNorm2d(128)

    ModuleArgsDict.init_func(batch_norm, "normal", 0.02)

    assert abs(batch_norm.weight.mean().item() - 1.0) < 0.02
    assert batch_norm.bias.abs().max().item() < 1e-6


# --------------------------------------------------------------------------------------
# Network.load_state_dict
# --------------------------------------------------------------------------------------


def test_load_state_dict_warm_starts_resized_layer_and_keeps_siblings():
    """#2 A resized layer must warm-start, and sibling layers must still load.

    The bug checked ``isinstance(module, Linear)`` (the parent) instead of the
    child, and used an early ``return`` that aborted loading the remaining
    siblings of a resized layer.
    """

    class _Net(Network):
        def __init__(self, fc_out: int) -> None:
            super().__init__(in_channels=1)
            self.add_module("fc", torch.nn.Linear(4, fc_out))
            self.add_module("head", torch.nn.Linear(4, 2))

    old = _Net(fc_out=4)
    # Network.state_dict() wraps the flat params under the network name; load_state_dict
    # consumes that inner flat dict ("fc.weight", ...).
    inner = next(iter(old.state_dict().values()))
    checkpoint = {key: value.clone() for key, value in inner.items()}

    new = _Net(fc_out=6)  # fc output grows 4 -> 6 (resized); head is unchanged
    new.load_state_dict(checkpoint)  # must not raise

    fc = new["fc"]
    head = new["head"]
    assert fc.weight.shape == (6, 4)
    assert torch.equal(fc.weight[:4], checkpoint["fc.weight"])  # warm-started rows
    # The sibling after the resized layer must still be loaded (old `return` skipped it).
    assert torch.equal(head.weight, checkpoint["head.weight"])
    assert torch.equal(head.bias, checkpoint["head.bias"])


# --------------------------------------------------------------------------------------
# Measure.Loss — loss records feeding the gradient and the logging windows
# --------------------------------------------------------------------------------------


def _loss_record() -> Measure.Loss:
    return Measure.Loss("l", "out", "tgt", 0, is_loss=True, accumulation=False)


def test_get_loss_uses_current_iteration_weight() -> None:
    # reset_loss clears _loss every iteration but _weight keeps growing for the logging windows.
    # get_loss must pair the current loss with the current weight; the pre-fix code zipped from the
    # front, so a loss-weight scheduler that changes the weight had no effect on the gradient.
    record = _loss_record()

    record.reset_loss()
    record.add(2.0, torch.tensor([3.0]))
    assert record.get_loss().item() == 6.0  # 2 * 3

    record.reset_loss()  # next iteration; _weight is now [2.0, 5.0]
    record.add(5.0, torch.tensor([1.0]))
    assert record.get_loss().item() == 5.0  # 5 * 1, not the stale 2 * 1


def test_get_loss_handles_multiple_accumulated_patches() -> None:
    # Accumulation mode adds several (weight, loss) pairs per iteration; the trailing weights must
    # still line up one-to-one with the current losses.
    record = _loss_record()

    record.reset_loss()
    record.add(1.0, torch.tensor([10.0]))  # a previous iteration leaves a weight behind
    record.reset_loss()
    record.add(0.5, torch.tensor([2.0]))
    record.add(0.5, torch.tensor([4.0]))

    assert record.get_loss().item() == 1.5  # mean(0.5 * 2, 0.5 * 4)


def test_loss_add_summarises_dict_metric_payload() -> None:
    # Dice/TRE return (tensor, {label: value}); the pre-fix code stored the dict in _values, so the
    # np.nanmean over _values in get_last_values/format_loss raised TypeError on every batch.
    record = Measure.Loss("Dice", "out", "tgt", 0, is_loss=False, accumulation=False)

    record.add(1.0, (torch.tensor([0.7]), {"1": 0.6, "2": 0.8, "3": float("nan")}))

    # The dict is summarised to a scalar (nan-mean of 0.6 and 0.8), and the logging mean is safe.
    assert isinstance(record._values[-1], float)
    assert record._values[-1] == pytest.approx(0.7)
    assert np.nanmean(record._values) == pytest.approx(0.7)


def test_loss_add_keeps_plain_scalar_metric() -> None:
    # A regular (tensor, float) metric is unchanged.
    record = Measure.Loss("MSE", "out", "tgt", 0, is_loss=False, accumulation=False)
    record.add(1.0, (torch.tensor([0.5]), 0.5))
    assert record._values[-1] == pytest.approx(0.5)


# --------------------------------------------------------------------------------------
# Measure — accumulation backward (AMP scaler vs plain)
# --------------------------------------------------------------------------------------


class _CriterionAttr:
    def __init__(self) -> None:
        self.start = 0
        self.stop = None
        self.schedulers = {Constant(1.0): 1}
        self.group = 0
        self.is_loss = True
        self.accumulation = True


def _make_accumulating_measure(scaler) -> tuple[Measure, torch.Tensor]:
    """Build a minimal Measure that triggers the accumulation-backward branch."""
    measure = Measure.__new__(Measure)
    criterion = torch.nn.MSELoss()
    key = f"out:tgt:{criterion.__class__.__name__}"
    measure.outputs_criterions = {"out": {"tgt": {criterion: _CriterionAttr()}}}
    measure._loss = {0: {key: Measure.Loss(criterion.__class__.__name__, "out", "tgt", 0, True, True)}}
    measure.scaler = scaler
    output = torch.zeros(1, 1, 2, 2, requires_grad=True)
    return measure, output


def test_accumulation_backward_uses_scaler_scale():
    """#AMP: accumulation losses must be scaled before backward when a GradScaler is set."""
    scaler = MagicMock()
    scaled = MagicMock()
    scaler.scale.return_value = scaled

    measure, output = _make_accumulating_measure(scaler)
    target = torch.ones(1, 1, 2, 2)
    measure.update("out", output, {"tgt": (target, [Attribute()])}, it=0, nb_patch=1, training=True)

    # The loss must go through scaler.scale(...).backward(), never a bare loss.backward().
    scaler.scale.assert_called_once()
    scaled.backward.assert_called_once()
    # Bare backward would have populated grads directly; the scaler intercepts it.
    assert output.grad is None


def test_accumulation_backward_without_scaler_is_plain_backward():
    """Without a scaler the accumulation path must still back-propagate normally."""
    measure, output = _make_accumulating_measure(None)
    target = torch.ones(1, 1, 2, 2)
    measure.update("out", output, {"tgt": (target, [Attribute()])}, it=0, nb_patch=1, training=True)

    assert output.grad is not None
    assert torch.count_nonzero(output.grad) > 0


# --------------------------------------------------------------------------------------
# Measure.update_scheduler — loss-weight window selection
# --------------------------------------------------------------------------------------


def test_update_scheduler_empty_raises_config_error():
    """update_scheduler on an empty schedule must raise a clear ConfigError."""
    with pytest.raises(ConfigError):
        Measure.update_scheduler(None, {}, 0)  # type: ignore[arg-type]


def test_update_scheduler_past_last_window_clamps_to_last():
    """Past every configured window, the last scheduler is selected (no crash)."""
    s0, s1 = Constant(1.0), Constant(2.0)
    schedulers = {s0: 3, s1: 3}  # active windows [0,3) and [3,6)
    assert Measure.update_scheduler(None, schedulers, 4) is s1  # type: ignore[arg-type]
    assert Measure.update_scheduler(None, schedulers, 100) is s1  # type: ignore[arg-type]


# --------------------------------------------------------------------------------------
# CriterionsLoader
# --------------------------------------------------------------------------------------


def test_network_criterion_loader_resets_scheduler_state(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyMeasure:
        def __init__(self) -> None:
            pass

    class DummySchedulerLoader:
        def __init__(self) -> None:
            self.nb_step = 3

        def getschedulers(self, key: str, scheduler_classname: str):
            return f"{key}:{scheduler_classname}"

    monkeypatch.setattr(network_module, "apply_config", lambda *args, **kwargs: lambda cls: cls)
    monkeypatch.setattr(network_module, "konfai_root", lambda: "Trainer")
    monkeypatch.setattr(
        network_module,
        "get_module",
        lambda classpath, default: (SimpleNamespace(Measure=DummyMeasure, __name__="torch.optim"), "Measure"),
    )

    attr = network_module.CriterionsAttr(
        schedulers=cast(
            dict[str, network_module.LossSchedulersLoader],
            {"Constant": DummySchedulerLoader()},
        )
    )
    loader = network_module.CriterionsLoader({"dummy:Measure": attr})

    loader.get_criterions("DemoModel", "Output", "Target")
    first_schedulers = dict(attr.schedulers)
    loader.get_criterions("DemoModel", "Output", "Target")

    assert attr.isTorchCriterion is True
    assert len(attr.schedulers) == 1
    assert attr.schedulers == first_schedulers


# --------------------------------------------------------------------------------------
# Model-level patching (Network.patch): each patch must land at its own index
# --------------------------------------------------------------------------------------


def test_model_patch_reassembles_each_patch_with_its_own_prediction() -> None:
    # Regression: the per-patch buffer leaked its end-module output into the next iteration, where the
    # name-transition branch re-added it at index i+1. The incremental-blend Accumulator ignores
    # re-added indices (a blended patch cannot be overwritten), so every slot > 0 silently received the
    # PREVIOUS patch's prediction: identity over [0..7] with patch 4 reassembled as [0,1,2,3,0,1,2,3].
    from konfai.data.patching import ModelPatch

    class _PatchNet(Network):
        def __init__(self) -> None:
            super().__init__(patch=ModelPatch(patch_size=[4]))
            self.add_module("Body", torch.nn.Identity())
            self.add_module("Head", torch.nn.Identity())

    net = _PatchNet()
    net._modulesArgs["Head"]._isEnd = True

    x = torch.arange(8, dtype=torch.float32).reshape(1, 1, 8)
    outputs = dict(net.named_forward(x))

    assert torch.equal(outputs["Head"], x)


def test_model_patch_deep_supervision_heads_each_reassemble_their_own_patches() -> None:
    # Two end modules (deep supervision): the mid-stream name-transition add and the trailing add must
    # each receive the CURRENT patch's output for their own module, across every patch iteration.
    from konfai.data.patching import ModelPatch

    class _DeepNet(Network):
        def __init__(self) -> None:
            super().__init__(patch=ModelPatch(patch_size=[4]))
            self.add_module("Aux", torch.nn.Identity())
            self.add_module("Head", _MulConst(2.0))

    net = _DeepNet()
    net._modulesArgs["Aux"]._isEnd = True
    net._modulesArgs["Head"]._isEnd = True

    x = torch.arange(8, dtype=torch.float32).reshape(1, 1, 8)
    outputs = dict(net.named_forward(x))

    assert torch.equal(outputs["Aux"], x)
    assert torch.equal(outputs["Head"], x * 2.0)
