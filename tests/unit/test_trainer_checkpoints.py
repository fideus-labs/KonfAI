from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn
from torch.optim.swa_utils import AveragedModel

import konfai.trainer as trainer_module
from konfai.trainer import _Trainer


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


def _build_trainer(tmp_path: Path, monkeypatch, date_values: list[str]) -> _Trainer:
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
        early_stopping=None,
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
