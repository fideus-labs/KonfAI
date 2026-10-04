# Copyright (c) 2026 Valentin Boussot
# SPDX-License-Identifier: Apache-2.0

"""Adaptive augmentation must observe training without changing validation or resume."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from konfai.metric.measure import PatchGanLoss
from konfai.metric.schedulers import Constant
from konfai.models.python.generation.diffusionGan import DiscriminatorADA, _wire_adversarial
from konfai.network.network import CriterionsAttr, Measure, Network
from konfai.utils.dataset import Attribute


def _controller(value=0.0):
    controller = DiscriminatorADA.UpdateP()
    controller.p = 0.5
    controller.set_measure(SimpleNamespace(get_last_values=lambda n: {"real": value}), ["real"])
    return controller


def test_validation_does_not_advance_ada_or_change_the_following_training_update():
    uninterrupted = _controller()
    interrupted = deepcopy(uninterrupted)
    tensor = torch.ones(1, 1, 4, 4)
    uninterrupted(tensor)
    interrupted(tensor)
    before = (interrupted.p, interrupted._it)
    interrupted.eval()
    for _ in range(7):
        assert interrupted(tensor).item() == 0
    assert (interrupted.p, interrupted._it) == before
    interrupted.train()
    for _ in range(8):
        torch.testing.assert_close(interrupted(tensor), uninterrupted(tensor))


@pytest.mark.parametrize("values", [{}, {"unrelated": 0.0}, {"real": float("nan")}, {"real": float("inf")}])
def test_ada_waits_for_a_finite_observation_instead_of_inventing_zero(values):
    controller = _controller()
    controller.set_measure(SimpleNamespace(get_last_values=lambda n: values), ["real"])
    for _ in range(9):
        assert controller(torch.ones(1)).item() == pytest.approx(0.5)
    assert controller.p == 0.5


@pytest.mark.parametrize("target_group", ["CT", "None"])
def test_ada_reads_the_configured_real_loss_regardless_of_dataset_group(target_group):
    discriminator = DiscriminatorADA(channels=[1, 2], strides=[1], dim=2)
    generator = Network(in_channels=1, dim=2)
    generator.add_module("Out", torch.nn.Identity())
    root = Network(in_channels=1, dim=2)
    _wire_adversarial(root, generator, discriminator)
    real = "Discriminator_B.DiscriminatorModel.Head.Conv"
    fake = "Discriminator_pB_detach.DiscriminatorModel.Head.Conv"
    measure = Measure("DiscriminatorADA", {})
    for output, target in [(real, 1), (fake, 0)]:
        attr = CriterionsAttr()
        attr.schedulers = {Constant(): None}
        measure.outputs_criterions[output] = {target_group: {PatchGanLoss(target): attr}}
    measure.init(root, [target_group])
    discriminator.measure = measure
    discriminator.initialized()
    controller = discriminator.get_submodule("DiscriminatorModel.Prob")
    controller.p = 0.5
    batch = {target_group: (torch.ones(1, 1, 4, 4), [Attribute()])}
    # Real error = 1 (p should decrease); fake error = 0 must not replace it.
    for iteration in range(4):
        for output in (real, fake):
            measure.update(output, torch.zeros(1, 1, 4, 4), batch, iteration, 1, False)
    assert controller(torch.ones(1)).item() == pytest.approx(0.499)


@pytest.mark.parametrize("training, probability", [(False, 0.7), (True, 0.0)])
def test_inactive_ada_returns_the_input_without_drawing_or_copying(training, probability):
    augment = DiscriminatorADA.DiscriminatorAugmentation(dim=2)
    augment.train(training)
    tensor = torch.randn(2, 1, 4, 4)
    before = torch.random.get_rng_state().clone()

    def unexpected(*args, **kwargs):
        pytest.fail("Inactive augmentation must not prepare random draws")

    augment._set_p = unexpected
    assert augment(tensor, torch.tensor(probability)) is tensor
    assert torch.equal(torch.random.get_rng_state(), before)


@pytest.mark.parametrize("seed", [1, 7])
def test_active_ada_keeps_the_generator_gradient(seed):
    torch.manual_seed(seed)
    augment = DiscriminatorADA.DiscriminatorAugmentation(dim=2)
    tensor = torch.randn(2, 1, 16, 16, requires_grad=True)
    output = augment(tensor, torch.tensor(1.0))
    assert output.shape == tensor.shape
    assert not torch.equal(output, tensor)
    output.square().mean().backward()
    assert tensor.grad is not None
    assert torch.isfinite(tensor.grad).all() and torch.count_nonzero(tensor.grad) > 0


def test_ada_resume_continues_the_probability_and_update_cadence(tmp_path):
    network = Network(in_channels=1, dim=2)
    controller = _controller()
    network.add_module("Probability", controller)
    for _ in range(7):
        controller(torch.ones(1))
    path = tmp_path / "weights.pt"
    torch.save(network.state_dict(), path)
    resumed = Network(in_channels=1, dim=2)
    restored = _controller()
    restored.p = 0
    resumed.add_module("Probability", restored)
    resumed.load_state_dict(torch.load(path, weights_only=True))
    assert (restored.p, restored._it) == (controller.p, controller._it)
    for _ in range(9):
        torch.testing.assert_close(restored(torch.ones(1)), controller(torch.ones(1)))


def test_legacy_ada_weights_remain_loadable():
    controller = DiscriminatorADA.UpdateP()
    controller.load_state_dict({})
    assert (controller.p, controller._it) == (0, 0)


def test_recent_ada_checkpoint_does_not_silently_accept_lost_state():
    controller = _controller()
    state = controller.state_dict()
    state.pop("_extra_state")
    with pytest.raises(RuntimeError, match="_extra_state"):
        controller.load_state_dict(state)


@pytest.mark.parametrize("legacy", [False, True])
def test_the_discriminator_loads_ada_state_through_the_konfai_checkpoint(legacy):
    model = DiscriminatorADA(channels=[1, 2], strides=[1], dim=2)
    controller = model.get_submodule("DiscriminatorModel.Prob")
    controller.p, controller._it = 0.375, 23
    state = model.network_states()
    if legacy:
        for weights in state.values():
            weights.pop("DiscriminatorModel.Prob._extra_state")
            weights._metadata["DiscriminatorModel.Prob"] = {"version": 1}
    loaded = DiscriminatorADA(channels=[1, 2], strides=[1], dim=2)
    loaded.load({"Model": state})
    restored = loaded.get_submodule("DiscriminatorModel.Prob")
    assert (restored.p, restored._it) == ((0, 0) if legacy else (0.375, 23))
