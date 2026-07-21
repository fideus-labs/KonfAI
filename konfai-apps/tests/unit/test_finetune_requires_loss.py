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

"""Fine-tuning must fail fast on a loss-less config instead of running an expensive no-op RESUME."""

import types
from pathlib import Path

import konfai_apps.app as app_module
import pytest
import torch
from konfai.utils.errors import AppRepositoryError
from konfai_apps.app import _finetune_target_has_loss

_WITH_LOSS = {
    "outputs_criterions": {"Head:Tanh": {"targets_criterions": {"CT": {"criterions_loader": {"MAE": {"is_loss": True}}}}}}
}
_METRIC_ONLY = {
    "outputs_criterions": {"Head:Tanh": {"targets_criterions": {"CT": {"criterions_loader": {"MAE": {"is_loss": False}}}}}}
}


def test_has_loss_accepts_a_real_loss() -> None:
    assert _finetune_target_has_loss({"classpath": "Net", "Net": _WITH_LOSS}) is True


def test_has_loss_rejects_none_outputs_criterions() -> None:
    # konfai stringifies the resolved default, so a PREDICTION config reads `outputs_criterions: None`.
    assert _finetune_target_has_loss({"classpath": "Net", "Net": {"outputs_criterions": "None"}}) is False
    assert _finetune_target_has_loss({"classpath": "Net", "Net": {"outputs_criterions": None}}) is False


def test_has_loss_rejects_metric_only() -> None:
    assert _finetune_target_has_loss({"classpath": "Net", "Net": _METRIC_ONLY}) is False


def test_has_loss_finds_nested_subnetwork_loss() -> None:
    # A GAN carries no loss at the parent model entry; each sub-network declares its own.
    gan = {"classpath": "Gan", "Gan": {"Generator": _WITH_LOSS, "Discriminator": _METRIC_ONLY}}
    assert _finetune_target_has_loss(gan) is True


def test_has_loss_rejects_engine_placeholder() -> None:
    # The registration-engine default expands to a placeholder key that names no concrete loss.
    placeholder = {
        "outputs_criterions": {
            "default": {"targets_criterions": {"Labels": {"criterions_loader": {"default|torch:nn:CrossEntropyLoss|Dice|NCC": {}}}}}
        }
    }
    assert _finetune_target_has_loss({"classpath": "Reg", "Reg": placeholder}) is False


def _drive_fine_tune(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, config_body: str) -> list[bool]:
    """Drive ``KonfAIApp.fine_tune`` with every heavy step stubbed, recording whether ``train`` ran."""
    src_ckpt = tmp_path / "CV_0_src.pt"
    torch.save({"epoch": 10, "it": 100, "loss": 0.0, "Model": {}}, src_ckpt)
    (tmp_path / "Dataset").mkdir()
    output_dir = tmp_path / "Output"
    output_dir.mkdir()
    (output_dir / "Config.yml").write_text(config_body, encoding="utf-8")

    trained: list[bool] = []

    def fake_train(*args, **kwargs):  # type: ignore[no-untyped-def]
        trained.append(True)
        from ruamel.yaml import YAML

        with open(Path(args[7])) as file:
            data = YAML().load(file)
        produced = Path(args[8]) / data["Trainer"]["train_name"]
        produced.mkdir(parents=True, exist_ok=True)
        torch.save({"epoch": 0, "it": 5, "loss": 0.0, "Model": {}}, produced / "out.pt")

    monkeypatch.setattr("konfai.trainer.train", fake_train)
    monkeypatch.setattr(app_module.KonfAIApp, "symlink", staticmethod(lambda *a, **k: None))
    app = app_module.KonfAIApp.__new__(app_module.KonfAIApp)
    app.app_repository = types.SimpleNamespace(  # type: ignore[attr-defined]
        install_fine_tune=lambda *a, **k: [("CV_0.pt", str(src_ckpt))]
    )
    app.fine_tune(
        dataset=tmp_path / "Dataset",
        name="Run",
        output=output_dir,
        epochs=1,
        it_validation=1,
        models=["CV_0"],
        gpu=[],
        cpu=1,
        quiet=True,
        config_file="Config.yml",
        tmp_dir=output_dir,
    )
    return trained


_LOSSLESS = "Trainer:\n  train_name: PLACEHOLDER\n  Model:\n    classpath: Net\n    Net:\n      outputs_criterions: None\n"
_WITH_LOSS_CONFIG = (
    "Trainer:\n  train_name: PLACEHOLDER\n  Model:\n    classpath: Net\n    Net:\n"
    "      outputs_criterions:\n        Head:\n          targets_criterions:\n"
    "            CT:\n              criterions_loader:\n                MAE:\n                  is_loss: true\n"
)


def test_fine_tune_raises_on_lossless_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    with pytest.raises(AppRepositoryError, match="attaches no loss"):
        _drive_fine_tune(monkeypatch, tmp_path, _LOSSLESS)


def test_fine_tune_runs_when_a_loss_is_present(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    assert _drive_fine_tune(monkeypatch, tmp_path, _WITH_LOSS_CONFIG) == [True]
