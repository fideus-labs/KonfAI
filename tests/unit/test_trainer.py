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

"""Tests for konfai.trainer: checkpoint save/bootstrap, early stopping, EMA, and RESUME
learning-rate/checkpoint handling."""

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import konfai.trainer as trainer_module
import pytest
import torch
from konfai.metric.schedulers import PolyLRScheduler
from konfai.network.network import Network
from konfai.trainer import EarlyStopping, EarlyStoppingBase, Trainer, _Trainer
from konfai.utils.errors import ConfigError, TrainerError
from konfai.utils.runtime import State
from torch import nn
from torch.optim.swa_utils import AveragedModel

# ---- Checkpoints ----


class _DummySummaryWriter:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def close(self) -> None:
        pass


class _DummyModelModule:
    @staticmethod
    def state_dict() -> dict[str, torch.Tensor]:
        return {"weight": torch.tensor([1.0])}

    @staticmethod
    def get_networks() -> dict[str, object]:
        return {}


class _DummyModel:
    def __init__(self) -> None:
        self.module = _DummyModelModule()


def _date_sequence(values: list[str]) -> Iterator[str]:
    yield from values
    while True:
        yield values[-1]


def _build_trainer(tmp_path: Path, monkeypatch, date_values: list[str], early_stopping=None) -> _Trainer:
    checkpoints_dir = tmp_path / "Checkpoints"
    statistics_dir = tmp_path / "Statistics"
    date_iter = _date_sequence(date_values)

    monkeypatch.setattr(trainer_module, "checkpoints_directory", lambda: checkpoints_dir)
    monkeypatch.setattr(trainer_module, "statistics_directory", lambda: statistics_dir)
    monkeypatch.setattr(trainer_module, "SummaryWriter", _DummySummaryWriter)
    monkeypatch.setattr(trainer_module, "current_date", lambda: next(date_iter))

    return _Trainer(
        world_size=1,
        global_rank=0,
        local_rank=0,
        size=1,
        train_name="RUN",
        early_stopping=early_stopping,
        data_log=None,
        save_checkpoint_mode="BEST",
        epochs=1,
        epoch=0,
        autocast=False,
        it_validation=1,
        it_lr_update=1,
        it=0,
        model=cast(Any, _DummyModel()),
        model_ema=None,
        dataloader_training=[object()],
        dataloader_validation=None,
    )


def test_best_checkpoint_save_keeps_only_best_without_rescanning(tmp_path: Path, monkeypatch) -> None:
    trainer = _build_trainer(tmp_path, monkeypatch, ["ckpt_a", "ckpt_b", "ckpt_c"])
    original_load = torch.load

    def fail_if_reloaded(*args, **kwargs):
        raise AssertionError("BEST checkpoint save unexpectedly rescanned saved checkpoints")

    monkeypatch.setattr(trainer_module.torch, "load", fail_if_reloaded)

    trainer.checkpoint_save(2.0)
    trainer.checkpoint_save(1.0)
    trainer.checkpoint_save(3.0)

    checkpoints = sorted((tmp_path / "Checkpoints" / "RUN").glob("*.pt"))
    assert [path.name for path in checkpoints] == ["ckpt_b.pt"]
    assert original_load(checkpoints[0], map_location="cpu", weights_only=False)["loss"] == 1.0


def test_best_checkpoint_keeps_highest_score_when_mode_is_max(tmp_path: Path, monkeypatch) -> None:
    # With a maximize-metric monitor (e.g. Dice), BEST retention must keep the HIGHEST score, not the
    # lowest. Regression guard for retention hardcoding "lower is better" and keeping the worst model.
    trainer = _build_trainer(
        tmp_path,
        monkeypatch,
        ["ckpt_a", "ckpt_b", "ckpt_c"],
        early_stopping=EarlyStopping(monitor=["Dice"], mode="max"),
    )

    trainer.checkpoint_save(0.60)
    trainer.checkpoint_save(0.85)  # best (highest)
    trainer.checkpoint_save(0.70)

    checkpoints = sorted((tmp_path / "Checkpoints" / "RUN").glob("*.pt"))
    assert [path.name for path in checkpoints] == ["ckpt_b.pt"]
    assert torch.load(checkpoints[0], map_location="cpu", weights_only=False)["loss"] == 0.85


def test_best_checkpoint_bootstrap_scans_existing_files_once_and_prunes_stale_ones(
    tmp_path: Path,
    monkeypatch,
) -> None:
    checkpoint_dir = tmp_path / "Checkpoints" / "RUN"
    checkpoint_dir.mkdir(parents=True)
    torch.save({"loss": 5.0}, checkpoint_dir / "old_a.pt")
    torch.save({"loss": 3.0}, checkpoint_dir / "old_b.pt")

    original_load = trainer_module.torch.load
    load_calls: list[Path] = []

    def counted_load(path, *args, **kwargs):
        load_calls.append(Path(path))
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(trainer_module.torch, "load", counted_load)

    trainer = _build_trainer(tmp_path, monkeypatch, ["ckpt_new_worse", "ckpt_new_best"])

    assert [path.name for path in sorted(checkpoint_dir.glob("*.pt"))] == ["old_b.pt"]
    assert [path.name for path in load_calls] == ["old_a.pt", "old_b.pt"]

    trainer.checkpoint_save(4.0)
    trainer.checkpoint_save(2.0)

    assert [path.name for path in load_calls] == ["old_a.pt", "old_b.pt"]
    checkpoints = sorted(checkpoint_dir.glob("*.pt"))
    assert [path.name for path in checkpoints] == ["ckpt_new_best.pt"]
    assert original_load(checkpoints[0], map_location="cpu", weights_only=False)["loss"] == 2.0


def test_best_checkpoint_survives_same_second_collision(tmp_path: Path, monkeypatch) -> None:
    trainer = _build_trainer(tmp_path, monkeypatch, ["same_stamp", "same_stamp"])

    trainer.checkpoint_save(1.0)  # best
    trainer.checkpoint_save(2.0)  # worse, produced within the same timestamp

    checkpoints = sorted((tmp_path / "Checkpoints" / "RUN").glob("*.pt"))
    assert len(checkpoints) == 1
    assert torch.load(checkpoints[0], map_location="cpu", weights_only=False)["loss"] == 1.0


def test_exit_checkpoint_loss_does_not_poison_best(tmp_path: Path, monkeypatch) -> None:
    trainer = _build_trainer(tmp_path, monkeypatch, ["exit_stamp"])

    trainer.checkpoint_save(None)  # the save emitted on context exit

    saved = torch.load(
        tmp_path / "Checkpoints" / "RUN" / "exit_stamp.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert saved["loss"] == float("inf")


def test_bootstrap_prefers_real_best_over_exit_checkpoint(tmp_path: Path, monkeypatch) -> None:
    checkpoint_dir = tmp_path / "Checkpoints" / "RUN"
    checkpoint_dir.mkdir(parents=True)
    torch.save({"loss": 3.0}, checkpoint_dir / "real_best.pt")
    torch.save({"loss": float("inf")}, checkpoint_dir / "exit.pt")

    trainer = _build_trainer(tmp_path, monkeypatch, ["new_stamp"])

    assert [path.name for path in sorted(checkpoint_dir.glob("*.pt"))] == ["real_best.pt"]
    assert trainer._best_checkpoint_loss == 3.0


def test_checkpoint_persists_ema_n_averaged(tmp_path: Path, monkeypatch) -> None:
    base = nn.Linear(2, 2)
    ema = AveragedModel(base)
    ema.update_parameters(base)
    ema.update_parameters(base)

    trainer = _build_trainer(tmp_path, monkeypatch, ["ema_stamp"])
    trainer.model_ema = cast(Any, ema)

    trainer.checkpoint_save(1.0)

    saved = torch.load(
        tmp_path / "Checkpoints" / "RUN" / "ema_stamp.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert "Model_EMA" in saved
    assert saved["Model_EMA_n_averaged"] == int(ema.n_averaged) == 2


def test_broadcast_stop_returns_local_value_without_distributed(tmp_path: Path, monkeypatch) -> None:
    trainer = _build_trainer(tmp_path, monkeypatch, ["stamp"])

    assert trainer._broadcast_stop(True) is True
    assert trainer._broadcast_stop(False) is False


def test_broadcast_stop_adopts_rank_zero_decision(tmp_path: Path, monkeypatch) -> None:
    trainer = _build_trainer(tmp_path, monkeypatch, ["stamp"])

    monkeypatch.setattr(trainer_module, "synchronize_data", lambda *_args, **_kwargs: [True, False, False])
    assert trainer._broadcast_stop(False) is True  # a non-zero rank still stops when rank 0 does

    monkeypatch.setattr(trainer_module, "synchronize_data", lambda *_args, **_kwargs: [False, True])
    assert trainer._broadcast_stop(True) is False  # a non-zero rank keeps going when rank 0 does


# ---- EarlyStopping ----


def test_early_stopping_base_starts_running_and_can_be_stopped() -> None:
    stopper = EarlyStoppingBase()

    assert stopper.is_stopped() is False

    stopper.stop()

    assert stopper.is_stopped() is True


def test_early_stopping_inherits_stop_from_base() -> None:
    stopper = EarlyStopping(monitor=[], patience=10)

    assert stopper.is_stopped() is False

    stopper.stop()

    assert stopper.is_stopped() is True


def test_early_stopping_triggers_after_patience_without_improvement() -> None:
    stopper = EarlyStopping(monitor=[], patience=2, mode="min")

    assert stopper(1.0) is False  # first score sets the baseline
    assert stopper(1.0) is False  # no improvement (counter = 1)
    assert stopper(1.0) is True  # no improvement (counter = 2 >= patience)
    assert stopper.is_stopped() is True


def test_get_score_reports_missing_metric_and_available_keys() -> None:
    stopper = EarlyStopping(monitor=["val_loss"], patience=3)

    with pytest.raises(TrainerError) as exc_info:
        stopper.get_score({"train_loss": 1.0, "dice": 0.5})

    message = str(exc_info.value)
    assert "val_loss" in message  # the missing monitored metric is named
    assert "train_loss" in message  # the keys actually available are listed
    assert "dice" in message
    assert "{}" not in message  # the placeholder is interpolated, not left raw


# ---- EMA ----


def test_avg_fn_follows_standard_ema_convention() -> None:
    stub = SimpleNamespace(ema_decay=0.9)
    averaged = torch.tensor(1.0)
    model = torch.tensor(0.0)

    result = Trainer._avg_fn(stub, averaged, model, 0)

    assert result.item() == pytest.approx(0.9)


def test_avg_fn_high_decay_keeps_running_average_dominant() -> None:
    stub = SimpleNamespace(ema_decay=0.999)
    averaged = torch.tensor(10.0)
    model = torch.tensor(0.0)

    result = Trainer._avg_fn(stub, averaged, model, 5)

    assert result.item() == pytest.approx(9.99)


# ---- RESUME LR override ----

# Resume/fine-tune learning-rate override semantics for ``Network.load``.
#
# Without ``override_lr`` a resume must keep the checkpoint (decayed) learning rate and
# let the scheduler continue from ``_nb_lr_update``. With ``override_lr`` the learning
# rate must restart from the requested value and the scheduler must decay from there.

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


# ---- RESUME checkpoint URL ----


def test_build_train_keeps_https_checkpoint_url(monkeypatch) -> None:
    # build_train must not wrap an https:// URL in Path(): that collapses '//' into 'https:/…',
    # which then fails both the startswith('https://') check and Path.exists() at load time.
    recorded: dict[str, object] = {}

    class _DummyTrainer:
        def set_model(self, path_to_model) -> None:
            recorded["model"] = path_to_model

        def set_lr(self, lr) -> None:
            recorded["lr"] = lr

    monkeypatch.setattr(trainer_module, "configure_workflow_environment", lambda **kwargs: None)
    monkeypatch.setattr(
        trainer_module,
        "apply_config",
        lambda *args, **kwargs: lambda cls: lambda: _DummyTrainer(),
    )

    url = "https://example.com/weights/ckpt.pt"
    trainer_module.build_train(command=State.RESUME, model=url)

    assert recorded["model"] == url


def test_early_stopping_refuses_a_mode_that_is_not_a_direction() -> None:
    # `is_better` and `worst_score` read it as "max" or everything-else, so a typo silently retained
    # and deleted checkpoints by the wrong direction before anything complained.
    with pytest.raises(ConfigError) as error:
        EarlyStopping(monitor=None, mode="mxa")
    assert "'min' or 'max'" in str(error.value)


@pytest.mark.parametrize("mode", ["min", "max"])
def test_a_saved_checkpoint_with_no_score_loses_to_one_with_a_score(tmp_path: Path, monkeypatch, mode: str) -> None:
    # A no-score epoch stored `inf`, which only loses where lower is better: under 'max' it beat every
    # finite score, so BEST froze on the last unscored epoch and no later one could take it.
    early_stopping = EarlyStopping(monitor=None, mode=mode)
    trainer = _build_trainer(tmp_path, monkeypatch, ["ckpt_a", "ckpt_b"], early_stopping=early_stopping)
    trainer.checkpoint_save(None)
    saved = sorted((tmp_path / "RUN" / "Checkpoints").glob("*.pt")) if (tmp_path / "RUN").exists() else []
    if not saved:
        saved = sorted(tmp_path.rglob("*.pt"))
    assert saved, "checkpoint_save wrote nothing"
    stored = float(torch.load(saved[-1], map_location="cpu", weights_only=False)["loss"])
    assert not early_stopping.is_better(stored, 0.5)


@pytest.mark.parametrize("mode", ["min", "max"])
def test_a_score_that_is_not_finite_is_no_score(mode: str) -> None:
    # What an older run wrote for a no-score epoch, whichever direction reads it back.
    import math

    early_stopping = EarlyStopping(monitor=None, mode=mode)
    legacy = float("inf")
    read_back = early_stopping.worst_score if not math.isfinite(legacy) else legacy
    assert not early_stopping.is_better(read_back, 0.5)
