# SPDX-License-Identifier: Apache-2.0
"""BEST metadata scans preserve selection while releasing mapped weights before pruning."""

import weakref
from pathlib import Path

import konfai.trainer as trainer_module
import pytest
import torch
from konfai.trainer import EarlyStoppingBase, _checkpoint_score, _Trainer


@pytest.mark.parametrize("zip_format", [False, True])
def test_checkpoint_score_returns_scalar_without_retaining_weights(tmp_path, monkeypatch, zip_format):
    path = tmp_path / "checkpoint.pt"
    torch.save(
        {"Model": {"w": torch.ones(8)}, "loss": torch.tensor(0.25)}, path, _use_new_zipfile_serialization=zip_format
    )
    original_load = trainer_module.safe_torch_load
    refs = []

    def observe(*args, **kwargs):
        result = original_load(*args, **kwargs)
        refs.extend([weakref.ref(result["Model"]["w"]), weakref.ref(result["loss"])])
        return result

    monkeypatch.setattr(trainer_module, "safe_torch_load", observe)
    assert _checkpoint_score(path, float("inf")) == 0.25
    assert all(ref() is None for ref in refs)


@pytest.mark.parametrize("mode,expected", [("min", "a.pt"), ("max", "b.pt")])
def test_best_scan_maps_metadata_and_releases_every_file_before_pruning(tmp_path, monkeypatch, mode, expected):
    directory = tmp_path / "run"
    directory.mkdir()
    for name, score in [("a.pt", 0.25), ("b.pt", 0.75), ("c.pt", float("nan")), ("crash_bad.pt", -1.0)]:
        torch.save({"Model": {"w": torch.ones(8)}, "loss": score}, directory / name)
    torch.save({"Model": {"w": torch.ones(8)}}, directory / "missing.pt")
    (directory / "resume_latest.pt").write_bytes(b"must not be read or pruned by BEST")
    trainer = _Trainer.__new__(_Trainer)
    trainer.train_name = "run"
    trainer.early_stopping = EarlyStoppingBase()
    trainer.early_stopping.mode = mode
    trainer._best_checkpoint_path = None
    trainer._best_checkpoint_loss = None
    monkeypatch.setattr(trainer_module, "checkpoints_directory", lambda: tmp_path)
    original_load = trainer_module.safe_torch_load
    original_unlink = Path.unlink
    refs = []
    loaded = []

    def observe(path, *args, **kwargs):
        assert kwargs["mmap"] is True
        loaded.append(path.name)
        state = original_load(path, *args, **kwargs)
        refs.append(weakref.ref(state["Model"]["w"]))
        return state

    def unlink_after_release(path, *args, **kwargs):
        assert all(ref() is None for ref in refs)
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(trainer_module, "safe_torch_load", observe)
    monkeypatch.setattr(Path, "unlink", unlink_after_release)
    trainer._initialize_best_checkpoint_state()

    assert loaded == ["a.pt", "b.pt", "c.pt", "missing.pt"]
    assert trainer._best_checkpoint_path == directory / expected
    assert trainer._best_checkpoint_loss == (0.25 if mode == "min" else 0.75)
    assert {path.name for path in directory.iterdir()} == {expected, "crash_bad.pt", "resume_latest.pt"}
