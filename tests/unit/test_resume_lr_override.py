"""Resume/fine-tune learning-rate override semantics for ``Network.load``.

Without ``override_lr`` a resume must keep the checkpoint (decayed) learning rate and
let the scheduler continue from ``_nb_lr_update``. With ``override_lr`` the learning
rate must restart from the requested value and the scheduler must decay from there.
"""

import torch
from konfai.metric.schedulers import PolyLRScheduler
from konfai.network.network import Network

_CONFIG_LR = 0.1
_GAMMA = 0.5
_NB_LR_UPDATE = 3


class _LeafNet(Network):
    """Minimal concrete network with no sub-networks, driving ``Network.load`` directly."""

    def __init__(self) -> None:
        super().__init__()


def _fresh_optimizer() -> torch.optim.Optimizer:
    param = torch.nn.Parameter(torch.zeros(1))
    return torch.optim.SGD([param], lr=_CONFIG_LR)


def _decayed_optimizer_state() -> tuple[dict, float]:
    """Optimizer state as saved by a checkpoint after ``_NB_LR_UPDATE`` StepLR decays."""
    optimizer = _fresh_optimizer()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=_GAMMA)
    for _ in range(_NB_LR_UPDATE):
        scheduler.step()
    return optimizer.state_dict(), optimizer.param_groups[0]["lr"]


def _make_net(scheduler_factory) -> tuple[_LeafNet, torch.optim.lr_scheduler.LRScheduler, dict]:
    net = _LeafNet()
    optimizer = _fresh_optimizer()
    scheduler = scheduler_factory(optimizer)
    net.optimizer = optimizer
    net.schedulers = {scheduler: 0}
    net._it = 0
    net._nb_lr_update = 0
    optimizer_state, decayed_lr = _decayed_optimizer_state()
    state_dict = {
        f"{net.get_name()}_optimizer_state_dict": optimizer_state,
        f"{net.get_name()}_nb_lr_update": _NB_LR_UPDATE,
    }
    return net, scheduler, {"state_dict": state_dict, "decayed_lr": decayed_lr}


def test_resume_without_override_keeps_decayed_lr_and_restores_scheduler() -> None:
    net, scheduler, ctx = _make_net(lambda opt: torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=_GAMMA))

    net.load(ctx["state_dict"], init=False, ema=False)

    # The decayed learning rate from the checkpoint is preserved (not reset to the config LR).
    assert net.optimizer.param_groups[0]["lr"] == ctx["decayed_lr"]
    assert net.optimizer.param_groups[0]["lr"] != _CONFIG_LR
    # The scheduler continues from where it left off instead of restarting at 0.
    assert scheduler.last_epoch == _NB_LR_UPDATE


def test_resume_with_override_restarts_lr_and_scheduler() -> None:
    override = 0.02
    net, scheduler, ctx = _make_net(lambda opt: torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=_GAMMA))

    net.load(ctx["state_dict"], init=False, ema=False, override_lr=override)

    # The learning rate is forced to the override, and the schedule restarts from it.
    assert net.optimizer.param_groups[0]["lr"] == override
    assert net.optimizer.param_groups[0]["initial_lr"] == override
    assert scheduler.base_lrs == [override]
    assert scheduler.last_epoch == 0

    # Decaying from the override reproduces a fresh schedule anchored at ``override``.
    scheduler.step()
    assert net.optimizer.param_groups[0]["lr"] == override * _GAMMA


def test_resume_with_override_restarts_polylr_from_value() -> None:
    override = 0.05
    max_steps = 100
    exponent = 0.9
    net, scheduler, ctx = _make_net(
        lambda opt: PolyLRScheduler(opt, initial_lr=_CONFIG_LR, max_steps=max_steps, exponent=exponent)
    )

    net.load(ctx["state_dict"], init=False, ema=False, override_lr=override)

    assert net.optimizer.param_groups[0]["lr"] == override
    assert scheduler.initial_lr == override
    assert scheduler.last_epoch == 0

    scheduler.step()
    assert net.optimizer.param_groups[0]["lr"] == override * (1 - 0 / max_steps) ** exponent
    scheduler.step()
    assert net.optimizer.param_groups[0]["lr"] == override * (1 - 1 / max_steps) ** exponent
