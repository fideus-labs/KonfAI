# Copyright (c) 2026 Valentin Boussot
# SPDX-License-Identifier: Apache-2.0

"""The shipped adversarial wiring against independent, simultaneous PyTorch updates."""

import json
import os
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from konfai.data.patching import ModelPatch
from konfai.metric.measure import PatchGanLoss
from konfai.metric.schedulers import Constant
from konfai.models.python.generation.diffusionGan import _wire_adversarial
from konfai.models.python.generation.gan import Gan
from konfai.network.network import CriterionsAttr, Measure, Network
from konfai.trainer import _ddp_kwargs
from konfai.utils.dataset import Attribute
from konfai.utils.runtime.distributed import pin_gloo_to_loopback
from torch.nn.parallel import DistributedDataParallel as DDP

pytestmark = pytest.mark.filterwarnings("ignore:Losses read different patch walks")


def _gan(immediate, route, cadences, wiring="gan", losses="both", reconstruction=True):
    generator, discriminator = Network(in_channels=1, dim=2), Network(in_channels=1, dim=2)
    for model, name, weight, lr, cadence, scale in zip(
        (generator, discriminator),
        ("Generator", "Discriminator"),
        (0.7, 0.4),
        (0.03, 0.02),
        cadences,
        (8, 32),
        strict=True,
    ):
        model.set_name(name)
        model.nb_batch_per_step = cadence
        model.add_module("Out", torch.nn.Conv2d(1, 1, 1, bias=False))
        with torch.no_grad():
            model["Out"].weight.fill_(weight)
        model.optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        model.scaler = torch.amp.GradScaler("cpu", init_scale=scale)
    if wiring == "gan":
        root = Gan(generator, discriminator)
    else:
        root = Network(in_channels=1, dim=2)
        _wire_adversarial(root, generator, discriminator)
    if route != "none":
        patched = {
            "root": root,
            "generator": generator,
            "generator_assembled": generator,
            "discriminator": discriminator,
            "discriminator_assembled": discriminator,
        }[route]
        patched.patch = ModelPatch(patch_size=[2, 2])
    root._compute_channels_trace(root, 1, None, None)
    objectives = [
        (generator, {"Discriminator_pB.Out": PatchGanLoss(1)}),
        (discriminator, {"Discriminator_B.Out": PatchGanLoss(1), "Discriminator_pB_detach.Out": PatchGanLoss(0)}),
    ]
    if reconstruction:
        name = "Generator_A_to_B.;accu;Out" if route == "generator_assembled" else "Generator_A_to_B.Out"
        objectives[0][1][name] = torch.nn.MSELoss()
    for index, (model, outputs) in enumerate(objectives):
        if losses == ("discriminator" if index == 0 else "generator"):
            continue
        model.measure = Measure(model.get_name(), {})
        model.measure.outputs_criterions = {}
        for name, criterion in outputs.items():
            if route == "discriminator_assembled" and name.startswith("Discriminator_"):
                name = name.replace(".Out", ".;accu;Out")
            attr = CriterionsAttr(accumulation=immediate[index])
            attr.schedulers = {Constant(): None}
            model.measure.outputs_criterions[name] = {"B": {criterion: attr}}
        model.measure.init(root, ["A", "B"])
        model.measure.scaler = model.scaler
    root.init_outputs_group()
    return root, generator, discriminator


def _batch(rank=0):
    shape = (1, 1, 4 + rank * 2, 4 + rank * 2)
    return {
        name: SimpleNamespace(tensor=torch.full(shape, float(value)), is_input=True, attribute=[Attribute()])
        for name, value in (("A", rank + 1), ("B", rank + 2))
    }


def _reference(cadences, ranks=1, losses="both", reconstruction=True):
    generator = torch.nn.Parameter(torch.tensor(0.7))
    discriminator = torch.nn.Parameter(torch.tensor(0.4))
    optimizers = [torch.optim.SGD([generator], lr=0.03), torch.optim.SGD([discriminator], lr=0.02)]
    x = torch.arange(1, ranks + 1, dtype=torch.float32)
    real = x + 1
    observed = []
    for iteration in range(6):
        fake = generator * x
        d_loss = ((discriminator * real - 1).square() + (discriminator * fake.detach()).square()).mean()
        g_loss = (discriminator.detach() * fake - 1).square().mean()
        if reconstruction:
            g_loss = g_loss + (fake - real).square().mean()
        # Both objectives use the same forward-time weights. A fresh-forward alternating GAN
        # is a different update rule; this is the rule expressed by KonfAI's routed graph.
        if losses != "generator":
            (d_loss / cadences[1]).backward()
        if losses != "discriminator":
            (g_loss / cadences[0]).backward()
        for optimizer, cadence in zip(optimizers, cadences, strict=True):
            if (iteration + 1) % cadence == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        observed.append([generator.item(), discriminator.item()])
    return observed


def _run(root, generator, discriminator, wrapped, rank=0):
    observed = []
    for _ in range(6):
        with root.accumulation_sync(wrapped):
            if isinstance(wrapped, DDP):
                wrapped(_batch(rank), _ddp_losses=wrapped.require_backward_grad_sync)
            else:
                wrapped(_batch(rank))
            root.backward(wrapped)
        observed.append([generator["Out"].weight.item(), discriminator["Out"].weight.item()])
    return observed


@pytest.mark.parametrize("immediate", [(False, False), (True, True), (True, False), (False, True)])
@pytest.mark.parametrize(
    "route", ["none", "root", "generator", "generator_assembled", "discriminator", "discriminator_assembled"]
)
@pytest.mark.parametrize("cadences", [(1, 1), (2, 3)])
@pytest.mark.parametrize("wiring", ["gan", "diffusion"])
def test_shipped_adversarial_graph_updates_both_optimizers_like_pytorch(immediate, route, cadences, wiring):
    root, generator, discriminator = _gan(immediate, route, cadences, wiring)
    observed = _run(root, generator, discriminator, root)
    torch.testing.assert_close(torch.tensor(observed), torch.tensor(_reference(cadences)))
    assert generator._it == discriminator._it == 6


@pytest.mark.parametrize("losses", ["generator", "discriminator"])
@pytest.mark.parametrize("immediate", [(False, False), (True, True)])
@pytest.mark.parametrize("route", ["none", "generator", "discriminator"])
def test_adversarial_gradients_update_only_their_owner(losses, immediate, route):
    root, generator, discriminator = _gan(immediate, route, (1, 1), losses=losses, reconstruction=False)
    observed = _run(root, generator, discriminator, root)
    torch.testing.assert_close(
        torch.tensor(observed), torch.tensor(_reference((1, 1), losses=losses, reconstruction=False))
    )
    inactive = discriminator if losses == "generator" else generator
    assert inactive["Out"].weight.grad is None
    assert inactive._it == 0


@pytest.mark.parametrize("split_loss", [False, True])
def test_reporting_a_frozen_discriminator_pass_does_not_disable_its_training_gradients(split_loss):
    root, generator, discriminator = _gan((False, False), "none", (1, 1))
    if split_loss:
        criteria = discriminator.measure.outputs_criterions["Discriminator_B.Out"]["B"]
        for attr in criteria.values():
            attr.schedulers = {Constant(0.5): None}
        second = CriterionsAttr()
        second.schedulers = {Constant(0.5): None}
        criteria[PatchGanLoss(1)] = second
    metric = CriterionsAttr(is_loss=False)
    metric.schedulers = {Constant(): None}
    discriminator.measure.outputs_criterions["Discriminator_pB.Out"] = {"B": {PatchGanLoss(1): metric}}
    discriminator.measure.init(root, ["A", "B"])
    root.init_outputs_group()
    observed = _run(root, generator, discriminator, root)
    torch.testing.assert_close(torch.tensor(observed), torch.tensor(_reference((1, 1))))


def _ddp_rank(rank, directory, immediate, route, cadences):
    torch.set_num_threads(1)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    with pin_gloo_to_loopback(local=True):
        dist.init_process_group(
            "gloo",
            init_method=(Path(directory) / "group").as_uri(),
            rank=rank,
            world_size=2,
            timeout=timedelta(seconds=30),
        )
    try:
        root, generator, discriminator = _gan(immediate, route, cadences)
        wrapped = DDP(root, **_ddp_kwargs(root, local_rank=rank, size=1))
        observed = _run(root, generator, discriminator, wrapped, rank)
        (Path(directory) / f"rank-{rank}.json").write_text(json.dumps(observed))
    finally:
        dist.destroy_process_group()


@pytest.mark.integration
@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo unavailable")
@pytest.mark.parametrize(
    "immediate, route, cadences",
    [
        ((False, False), "none", (1, 1)),
        ((True, True), "none", (1, 1)),
        ((True, True), "root", (2, 3)),
        ((True, True), "generator", (2, 3)),
        ((True, True), "discriminator", (2, 3)),
        ((True, False), "generator", (1, 1)),
        ((False, True), "generator", (1, 1)),
    ],
)
def test_gan_ddp_matches_the_global_batch_on_both_ranks(tmp_path, immediate, route, cadences):
    context = mp.spawn(_ddp_rank, args=(str(tmp_path), immediate, route, cadences), nprocs=2, join=False)
    deadline = time.monotonic() + 60
    try:
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail("GAN workers did not finish within 60 seconds")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
    expected = torch.tensor(_reference(cadences, ranks=2))
    for rank in range(2):
        observed = json.loads((tmp_path / f"rank-{rank}.json").read_text())
        torch.testing.assert_close(torch.tensor(observed), expected)


@pytest.mark.parametrize(
    "name, components",
    [
        ("DiffusionGan", ("GeneratorV1", "DiscriminatorADA")),
        ("DiffusionGanV2", ("GeneratorV2", "Discriminator")),
        ("DiffusionCycleGan", ("CycleGanGeneratorV3", "CycleGanDiscriminator")),
    ],
)
def test_gan_variants_build_fresh_default_networks(name, components, monkeypatch):
    from konfai.models.python.generation import diffusionGan

    def tiny(component):
        model = Network(in_channels=1, dim=2)
        model.set_name(component)
        model.add_module("Out", torch.nn.Conv2d(1, 1, 1))
        return model

    for component in components:
        monkeypatch.setattr(diffusionGan, component, lambda component=component: tiny(component))
    first, second = getattr(diffusionGan, name)(), getattr(diffusionGan, name)()
    assert not {id(p) for p in first.parameters()} & {id(p) for p in second.parameters()}
    assert len(list(first.parameters())) == 4


def test_diffusion_time_embedding_stays_fixed_when_the_discriminator_is_unfrozen():
    from konfai.models.python.generation.diffusionGan import DiscriminatorADA

    embedding = DiscriminatorADA.TimeEmbedding(8, 4)
    expected = embedding.time_embed.weight.detach().clone()
    embedding.requires_grad_(True)
    output = embedding(torch.ones(2, 1, 4, 4), torch.tensor(0.5))
    assert not output.requires_grad
    torch.testing.assert_close(output, expected[[4, 4]])
    assert set(embedding.state_dict()) == {"time_embed.weight"}


@pytest.mark.parametrize(
    "route, immediate, expected",
    [
        ("root", (True, True), True),
        ("generator", (True, False), False),
        ("discriminator", (True, True), False),
    ],
)
def test_ddp_keeps_its_ordinary_reducer_when_shared_graphs_defer_every_loss(route, immediate, expected):
    root, _, _ = _gan(immediate, route, (1, 1))
    assert root.accumulates_patch_gradients() is expected


def test_synthesis_example_owns_its_models_and_predicts_without_a_second_domain(monkeypatch):
    import importlib.util
    import sys
    from types import ModuleType

    from konfai.network.network import NetState

    # Test the example's actual GAN wiring independently of its optional SMP backbone.
    smp = ModuleType("segmentation_models_pytorch")
    smp.UnetPlusPlus = lambda **kwargs: torch.nn.Identity()
    monkeypatch.setitem(sys.modules, "segmentation_models_pytorch", smp)
    path = Path(__file__).parents[2] / "examples" / "Synthesis" / "Model.py"
    spec = importlib.util.spec_from_file_location("synthesis_gan_example", path)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)

    def tiny(name):
        model = Network(in_channels=1, dim=2)
        model.set_name(name)
        model.add_module("Out", torch.nn.Conv2d(1, 1, 1))
        return model

    monkeypatch.setattr(example, "UNetpp5", lambda: tiny("Generator"))
    monkeypatch.setattr(example, "Discriminator", lambda: tiny("Discriminator"))
    first, second = example.Gan(), example.Gan()
    assert not {id(p) for p in first.parameters()} & {id(p) for p in second.parameters()}
    first.set_state(NetState.PREDICTION)
    outputs = dict(first.named_forward(torch.ones(1, 1, 4, 4)))
    assert "Generator_A_to_B.Out" in outputs
    assert not any("Discriminator" in name for name in outputs)


@pytest.mark.parametrize("immediate", [False, True])
def test_the_shipped_convolutional_gan_takes_real_adversarial_optimizer_steps(immediate):
    from konfai.models.python.generation.gan import Discriminator, Generator

    torch.manual_seed(8)
    generator = Generator(dim=2, patch=None, nb_batch_per_step=1)
    discriminator = Discriminator(dim=2, nb_batch_per_step=1)
    root = Gan(generator, discriminator)
    root._compute_channels_trace(root, 1, None, None)
    seen = []
    originals = []
    for model, outputs in [
        (generator, {"Discriminator_pB.Head.Conv": PatchGanLoss(1), "Generator_A_to_B.Head.Tanh": torch.nn.MSELoss()}),
        (
            discriminator,
            {"Discriminator_B.Head.Conv": PatchGanLoss(1), "Discriminator_pB_detach.Head.Conv": PatchGanLoss(0)},
        ),
    ]:
        model.optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        model.scaler = torch.amp.GradScaler("cpu", init_scale=8)
        originals.append([parameter.detach().clone() for parameter in model.parameters()])
        model.measure = Measure(model.get_name(), {})
        for output, criterion in outputs.items():
            attr = CriterionsAttr(accumulation=immediate)
            attr.schedulers = {Constant(): None}
            model.measure.outputs_criterions[output] = {"B": {criterion: attr}}
        model.measure.init(root, ["A", "B"])
        model.measure.scaler = model.scaler

        def inspect_step(optimizer, args, kwargs):
            gradients = [p.grad for group in optimizer.param_groups for p in group["params"]]
            assert gradients and all(g is not None and torch.isfinite(g).all() for g in gradients)
            seen.append(optimizer)

        model.optimizer.register_step_pre_hook(inspect_step)
    root.init_outputs_group()
    batch = {
        name: SimpleNamespace(tensor=torch.randn(2, 1, 64, 64), is_input=True, attribute=[Attribute(), Attribute()])
        for name in ("A", "B")
    }
    root(batch)
    root.backward(root)
    assert len(seen) == 2
    for model, before in zip((generator, discriminator), originals, strict=True):
        assert any(not torch.equal(was, now) for was, now in zip(before, model.parameters(), strict=True))
