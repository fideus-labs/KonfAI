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

import threading
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
    def network_states() -> dict[str, torch.Tensor]:
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


def _build_trainer(
    tmp_path: Path,
    monkeypatch,
    date_values: list[str],
    early_stopping=None,
    model: Any = None,
    it_validation: int = 1,
    dataloader_validation: Any = None,
) -> _Trainer:
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
        it_validation=it_validation,
        it_lr_update=1,
        it=0,
        model=cast(Any, _DummyModel() if model is None else model),
        model_ema=None,
        config_snapshot=tmp_path / "Config.yml",
        dataloader_training=[object()],
        dataloader_validation=dataloader_validation,
    )


def _save(trainer: _Trainer, loss: float | None) -> None:
    """A save and its landing on disk: what the loop guarantees before the next save and at exit."""
    trainer.checkpoint_save(loss)
    trainer._checkpoint_writer.join()


def test_best_checkpoint_save_keeps_only_best_without_rescanning(tmp_path: Path, monkeypatch) -> None:
    trainer = _build_trainer(tmp_path, monkeypatch, ["ckpt_a", "ckpt_b", "ckpt_c"])
    original_load = torch.load

    def fail_if_reloaded(*args, **kwargs):
        raise AssertionError("BEST checkpoint save unexpectedly rescanned saved checkpoints")

    monkeypatch.setattr(trainer_module.torch, "load", fail_if_reloaded)

    _save(trainer, 2.0)
    _save(trainer, 1.0)
    _save(trainer, 3.0)

    checkpoints = sorted((tmp_path / "Checkpoints" / "RUN").glob("*.pt"))
    assert [path.name for path in checkpoints] == ["ckpt_b.pt"]
    assert original_load(checkpoints[0], map_location="cpu", weights_only=False)["loss"] == 1.0


def test_best_checkpoint_keeps_highest_score_when_mode_is_max(tmp_path: Path, monkeypatch) -> None:
    # With a maximize-metric monitor (e.g. Dice), BEST retention must keep the HIGHEST score, not the
    # lowest: retention hardcoding "lower is better" keeps the worst model.
    trainer = _build_trainer(
        tmp_path,
        monkeypatch,
        ["ckpt_a", "ckpt_b", "ckpt_c"],
        early_stopping=EarlyStopping(monitor=["Dice"], mode="max"),
    )

    _save(trainer, 0.60)
    _save(trainer, 0.85)  # best (highest)
    _save(trainer, 0.70)

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

    _save(trainer, 4.0)
    _save(trainer, 2.0)

    assert [path.name for path in load_calls] == ["old_a.pt", "old_b.pt"]
    checkpoints = sorted(checkpoint_dir.glob("*.pt"))
    assert [path.name for path in checkpoints] == ["ckpt_new_best.pt"]
    assert original_load(checkpoints[0], map_location="cpu", weights_only=False)["loss"] == 2.0


def test_best_checkpoint_survives_same_second_collision(tmp_path: Path, monkeypatch) -> None:
    trainer = _build_trainer(tmp_path, monkeypatch, ["same_stamp", "same_stamp"])

    _save(trainer, 1.0)  # best
    _save(trainer, 2.0)  # worse, produced within the same timestamp

    checkpoints = sorted((tmp_path / "Checkpoints" / "RUN").glob("*.pt"))
    assert len(checkpoints) == 1
    assert torch.load(checkpoints[0], map_location="cpu", weights_only=False)["loss"] == 1.0


def test_exit_checkpoint_loss_does_not_poison_best(tmp_path: Path, monkeypatch) -> None:
    trainer = _build_trainer(tmp_path, monkeypatch, ["exit_stamp"])

    _save(trainer, None)  # the save emitted on context exit

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
    # checkpoint_save reads the EMA module through the Network contract (network_states).
    class _Base(nn.Linear):
        def network_states(self) -> dict[str, dict[str, torch.Tensor]]:
            return {"Base": self.state_dict()}

    base = _Base(2, 2)
    ema = AveragedModel(base)
    ema.update_parameters(base)
    ema.update_parameters(base)

    trainer = _build_trainer(tmp_path, monkeypatch, ["ema_stamp"])
    trainer.model_ema = cast(Any, ema)

    _save(trainer, 1.0)

    saved = torch.load(
        tmp_path / "Checkpoints" / "RUN" / "ema_stamp.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert "Model_EMA" in saved
    assert saved["Model_EMA_n_averaged"] == int(ema.n_averaged) == 2


def test_checkpoint_save_returns_before_the_file_lands_and_the_file_equals_the_live_state(
    tmp_path: Path, monkeypatch
) -> None:
    # The training thread pays the host copy only; the serialisation and the publish run on the
    # writer's thread. What lands is what torch.save of the live states writes.
    net = nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(net.parameters())
    net(torch.ones(1, 3)).sum().backward()
    optimizer.step()
    network = SimpleNamespace(optimizer=optimizer, _it=4, _nb_lr_update=2, measure=None)
    module = SimpleNamespace(network_states=net.state_dict, get_networks=lambda: {"Net": network})
    trainer = _build_trainer(tmp_path, monkeypatch, ["stamp"], model=SimpleNamespace(module=module))

    gate = threading.Event()
    real_save = torch.save

    def gated_save(*args, **kwargs):
        gate.wait(5)
        real_save(*args, **kwargs)

    monkeypatch.setattr(trainer_module.torch, "save", gated_save)
    trainer.checkpoint_save(0.5)
    assert not list((tmp_path / "Checkpoints" / "RUN").glob("*.pt")), "returned before the write"
    gate.set()
    trainer._checkpoint_writer.join()

    reference = tmp_path / "reference.pt"
    real_save(
        {
            "epoch": 0,
            "it": 0,
            "loss": 0.5,
            "Model": net.state_dict(),
            "Net_optimizer_state_dict": optimizer.state_dict(),
            "Net_it": 4,
            "Net_nb_lr_update": 2,
        },
        reference,
    )
    saved = torch.load(tmp_path / "Checkpoints" / "RUN" / "stamp.pt", map_location="cpu", weights_only=False)
    expected = torch.load(reference, map_location="cpu", weights_only=False)

    def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
        if isinstance(value, dict):
            return {k: v for key, item in value.items() for k, v in flatten(item, f"{prefix}{key}.").items()}
        if isinstance(value, list | tuple):
            return {k: v for i, item in enumerate(value) for k, v in flatten(item, f"{prefix}{i}.").items()}
        return {prefix: value}

    flat_saved, flat_expected = flatten(saved), flatten(expected)
    assert flat_saved.keys() == flat_expected.keys()
    for key, value in flat_expected.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(flat_saved[key], value), key
        else:
            assert flat_saved[key] == value, key


def test_a_failed_checkpoint_write_is_raised_at_the_next_save(tmp_path: Path, monkeypatch) -> None:
    trainer = _build_trainer(tmp_path, monkeypatch, ["one", "two"])

    def failing_save(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(trainer_module.torch, "save", failing_save)
    trainer.checkpoint_save(1.0)  # returns: the failure sits on the writer's thread
    with pytest.raises(OSError, match="disk full"):
        trainer.checkpoint_save(2.0)  # joins the failed write before choosing a name
    trainer._checkpoint_writer.join()  # nothing in flight: the second save never got submitted
    assert not list((tmp_path / "Checkpoints" / "RUN").glob("*.pt"))


def test_broadcast_stop_returns_local_value_without_distributed(tmp_path: Path, monkeypatch) -> None:
    trainer = _build_trainer(tmp_path, monkeypatch, ["stamp"])

    assert trainer._broadcast_stop(True) is True
    assert trainer._broadcast_stop(False) is False


class _FakeGroup:
    """A process group of one broadcast: rank 0's payload lands on the caller, whatever it offered."""

    def __init__(self, monkeypatch, rank0_value: Any) -> None:
        self.calls = 0
        self.rank0_value = rank0_value
        monkeypatch.setattr(trainer_module.dist, "is_initialized", lambda: True)
        monkeypatch.setattr(trainer_module.torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(trainer_module.dist, "broadcast_object_list", self.broadcast)

    def broadcast(self, payload: list, src: int = 0) -> None:
        assert src == 0
        self.calls += 1
        payload[0] = self.rank0_value


def test_broadcast_stop_adopts_rank_zero_decision(tmp_path: Path, monkeypatch) -> None:
    trainer = _build_trainer(tmp_path, monkeypatch, ["stamp"])
    trainer.global_rank = 1

    _FakeGroup(monkeypatch, True)
    assert trainer._broadcast_stop(False) is True  # a non-zero rank still stops when rank 0 does

    _FakeGroup(monkeypatch, False)
    assert trainer._broadcast_stop(True) is False  # a non-zero rank keeps going when rank 0 does


def test_validation_request_and_live_tunables_ride_one_broadcast(tmp_path: Path, monkeypatch) -> None:
    # Rank 0 polls its SIGUSR1 flag and the control file once per cadence; every rank receives the
    # pair in one broadcast instead of two all_gathers, and acts on it at the same iteration.
    trainer = _build_trainer(tmp_path, monkeypatch, ["stamp"])
    trainer.world_size = 2
    trainer.it = trainer._LIVE_POLL_INTERVAL
    trainer._validate_now = True
    monkeypatch.setattr(trainer._live_control, "take", lambda: {"lr": 0.5})
    group = _FakeGroup(monkeypatch, (True, {"lr": 0.5}))

    assert trainer._poll_live_requests() == (True, {"lr": 0.5})
    assert group.calls == 1
    assert trainer._validate_now is False  # consumed

    trainer.global_rank = 1
    trainer._validate_now = False
    monkeypatch.setattr(trainer._live_control, "take", lambda: pytest.fail("only rank 0 reads the control file"))
    assert trainer._poll_live_requests() == (True, {"lr": 0.5})  # rank 0's pair, not its own
    assert group.calls == 2

    trainer.it += 1
    assert trainer._poll_live_requests() == (False, None)  # off the cadence: no collective
    assert group.calls == 2


# ---- Epoch wall-clock account ----


def test_epoch_report_closes_on_the_wall_clock_with_the_criteria_apart_from_the_forward() -> None:
    from konfai.utils.clock import SweepClock

    clock = SweepClock()
    clock._spent = {
        "epoch": 10.0,
        "wait(data)": 1.0,
        "forward": 5.0,  # the graph walk and, inside it, the criteria
        "criteria": 1.0,
        "backward+step": 2.0,
        "telemetry": 0.5,
        "validation": 0.5,
        "checkpoint": 0.5,
    }
    assert _Trainer._epoch_report(clock) == (
        "[KonfAI] epoch 10.0 s = wait(data) 1.0 + forward 4.0 + criteria 1.0 + backward+step 2.0 + ema 0.0"
        " + telemetry 0.5 + validation 0.5 + checkpoint 0.5 + other 0.5"
    )
    clock._spent = {"epoch": 0.4}
    assert _Trainer._epoch_report(clock) is None  # an epoch this short has nothing to account for


# ---- OOM shrink rendezvous agreement ----


def test_agreed_patch_takes_the_per_axis_min_of_the_proposals() -> None:
    # Ranks at their floor propose None and abstain; the survivors agree on the per-axis MIN so
    # every rank trains the same grid.
    assert trainer_module._agreed_patch([None, [1, 16, 16], [1, 12, 24]], [0, 0, 0]) == [1, 12, 16]


def test_agreed_patch_is_none_when_every_rank_is_at_the_floor() -> None:
    assert trainer_module._agreed_patch([None, None], [0, 0, 0]) is None


def test_agreed_patch_diagnoses_a_crossed_collective_instead_of_a_type_error() -> None:
    # An asymmetric OOM pairs this rendezvous with a still-training rank's own collective: the
    # gathered payload is then not a candidate list. That must fail as a diagnosis, not as an
    # opaque TypeError from min() or ValueError from zip().
    with pytest.raises(TrainerError, match="not a patch candidate"):
        trainer_module._agreed_patch([{"loss": 0.5}, [1, 16, 16]], [0, 0, 0])
    with pytest.raises(TrainerError, match="not a patch candidate"):
        trainer_module._agreed_patch([[1, 16], [1, 16, 16]], [0, 0, 0])  # wrong length = same diagnosis


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


def test_ema_runs_torchs_fused_rule_and_matches_the_per_tensor_one() -> None:
    # multi_avg_fn is the EMA formula as one _foreach_lerp_ per device and dtype.
    stub = SimpleNamespace(ema_decay=0.9)
    rule = Trainer._ema_update(cast(Any, stub))
    assert set(rule) == {"multi_avg_fn"}

    net = nn.Linear(4, 4)
    fused = AveragedModel(net, **rule)
    per_tensor = AveragedModel(net, avg_fn=lambda averaged, current, _n: 0.9 * averaged + 0.1 * current)
    for _ in range(3):
        with torch.no_grad():
            for param in net.parameters():
                param.add_(torch.randn_like(param))
        fused.update_parameters(net)
        per_tensor.update_parameters(net)
    for a, b in zip(fused.parameters(), per_tensor.parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=1e-6)
    assert int(fused.n_averaged) == int(per_tensor.n_averaged) == 3


def test_restoring_the_ema_weights_is_charged_to_the_checkpoint_phase(tmp_path: Path, monkeypatch) -> None:
    """The EMA copy is restored from the same checkpoint as the model, and the startup line
    accounts for it there: reported under ``setup`` it would read as the loaders' cost."""
    import time

    from konfai.utils.clock import restart_startup_clock

    class _SlowLoad:
        def load(self, state_dict: dict, **kwargs) -> None:
            del state_dict, kwargs
            time.sleep(0.05)

    monkeypatch.setattr(trainer_module, "checkpoints_directory", lambda: tmp_path / "Checkpoints")
    monkeypatch.setattr(trainer_module, "statistics_directory", lambda: tmp_path / "Statistics")
    monkeypatch.setattr(trainer_module, "konfai_state", lambda: "TRAIN")
    monkeypatch.setattr(
        trainer_module,
        "AveragedModel",
        lambda model, **kwargs: SimpleNamespace(module=_SlowLoad(), n_averaged=torch.zeros(1, dtype=torch.long)),
    )
    (tmp_path / "Statistics").mkdir()
    config_path = tmp_path / "Config.yml"
    config_path.write_text("Trainer:\n", encoding="utf-8")

    trainer = cast(Any, Trainer.__new__(Trainer))
    trainer.name = "RUN"
    trainer.size = 1
    trainer.it = 0
    trainer.ema_decay = 0.999
    trainer.model_ema = None
    trainer.override_lr = None
    trainer.model = cast(Any, SimpleNamespace(load=lambda *args, **kwargs: None))
    trainer.dataset = cast(Any, SimpleNamespace(get_data=lambda world_size: ([], [], [])))
    trainer.config_path_src = config_path
    trainer.config_namefile = tmp_path / "Statistics" / "RUN" / "Config_0.yml"

    clock = restart_startup_clock()
    with clock.phase("setup"):
        trainer.setup(1)

    assert trainer.model_ema is not None
    # A sleep can return a hair short of what it asked for (0.04977 on a Windows runner), so the
    # bound is the sleep less the platform's timer granularity. Charged to setup it would be 0.
    assert clock.spent("checkpoint") >= 0.04


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


def test_rebase_lr_makes_a_live_change_stick_past_the_scheduler() -> None:
    # The live-tuning LR knob must survive the next scheduler.step(), unlike a naive param_groups write that a
    # base_lrs scheduler would clobber back to the old anchor.
    new_lr = 0.03
    net, scheduler, _ = _make_net(lambda opt: torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=_GAMMA))

    net.rebase_lr(new_lr)

    assert net.optimizer.param_groups[0]["lr"] == new_lr
    assert scheduler.base_lrs == [new_lr]
    scheduler.step()
    assert net.optimizer.param_groups[0]["lr"] == new_lr * _GAMMA


def test_apply_tunables_changes_it_validation_and_audits(tmp_path: Path, monkeypatch) -> None:
    trainer = _build_trainer(tmp_path, monkeypatch, ["run"])
    trainer.it = 40

    applied = trainer._apply_tunables({"it_validation": 5})

    assert trainer.it_validation == 5
    assert applied == [{"it": 40, "key": "it_validation", "from": 1, "to": 5}]


def test_trainer_bounds_each_criterions_history_to_the_widest_window_it_reads(tmp_path: Path, monkeypatch) -> None:
    from konfai.network.network import Measure

    record = Measure.Loss("l", "out", "tgt", 0, is_loss=True, accumulation=False)
    measure = Measure.__new__(Measure)
    measure._loss = {0: {"l": record}}
    model = SimpleNamespace(module=SimpleNamespace(get_networks=lambda: {"Net": SimpleNamespace(measure=measure)}))

    trainer = _build_trainer(
        tmp_path, monkeypatch, ["stamp"], model=model, it_validation=7, dataloader_validation=[object()] * 3
    )
    assert record._values.maxlen == 7  # max(it_validation, validation batches)

    trainer._apply_tunables({"it_validation": 12})
    assert record._values.maxlen == 12


def test_apply_tunables_clamps_it_validation_to_at_least_one(tmp_path: Path, monkeypatch) -> None:
    trainer = _build_trainer(tmp_path, monkeypatch, ["run"])

    trainer._apply_tunables({"it_validation": 0})

    assert trainer.it_validation == 1  # a 0 interval would ZeroDivisionError the loop's modulo check


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
    # `is_better` and `worst_score` read it as "max" or everything-else, so a typo silently retains
    # and deletes checkpoints by the wrong direction unless refused here.
    with pytest.raises(ConfigError) as error:
        EarlyStopping(monitor=None, mode="mxa")
    assert "'min' or 'max'" in str(error.value)


@pytest.mark.parametrize("mode", ["min", "max"])
def test_a_saved_checkpoint_with_no_score_loses_to_one_with_a_score(tmp_path: Path, monkeypatch, mode: str) -> None:
    # A no-score epoch storing `inf` only loses where lower is better: under 'max' it beats every
    # finite score, so BEST freezes on the last unscored epoch and no later one can take it.
    early_stopping = EarlyStopping(monitor=None, mode=mode)
    trainer = _build_trainer(tmp_path, monkeypatch, ["ckpt_a", "ckpt_b"], early_stopping=early_stopping)
    _save(trainer, None)
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
