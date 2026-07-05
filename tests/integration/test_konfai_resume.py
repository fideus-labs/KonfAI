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

"""End-to-end RESUME workflow test.

Trains the tiny synthesis model for two epochs, then resumes the same workspace
with the config extended to four epochs (``konfai RESUME --model <last ckpt>``)
and asserts real continuity from checkpoint metadata: the prior checkpoints are
preserved, the trainer ``epoch``/``it`` counters and the AdamW ``step`` counters
continue monotonically from the loaded checkpoint instead of restarting, and the
resumed model still predicts finite values through the PREDICTION workflow.
"""

import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from test_konfai_core_workflows import _konfai_cli_command, _prepare_experiment_dir, _subprocess_env

pytestmark = pytest.mark.integration

SimpleITK = pytest.importorskip("SimpleITK")

EPOCHS_INITIAL = 2
EPOCHS_TOTAL = 4


def _load_checkpoint(path: Path) -> dict[str, Any]:
    # The checkpoint stores the loss as a numpy scalar, which the weights-only
    # unpickler rejects; the file is produced by this very test, so the trusted
    # loader is fine (mirrors konfai.utils.runtime.safe_torch_load's fallback).
    return torch.load(path, map_location="cpu", weights_only=False)  # nosec B614


def _read_checkpoints(checkpoints_dir: Path) -> dict[Path, dict[str, Any]]:
    return {path: _load_checkpoint(path) for path in sorted(checkpoints_dir.glob("*.pt"))}


def _latest(checkpoints: dict[Path, dict[str, Any]]) -> Path:
    return max(checkpoints, key=lambda path: (int(checkpoints[path]["it"]), int(checkpoints[path]["epoch"])))


def _optimizer_steps(checkpoint: dict[str, Any]) -> list[float]:
    steps: list[float] = []
    for key, value in checkpoint.items():
        if key.endswith("_optimizer_state_dict"):
            for param_state in value["state"].values():
                steps.append(float(param_state["step"]))
    return steps


def _flatten_model_weights(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    weights: dict[str, torch.Tensor] = {}
    for network_name, network_state in checkpoint["Model"].items():
        for parameter_name, tensor in network_state.items():
            weights[f"{network_name}.{parameter_name}"] = tensor
    return weights


def _replace_once(content: str, old: str, new: str) -> str:
    assert content.count(old) == 1, f"expected exactly one occurrence of {old!r} in the rendered config"
    return content.replace(old, new)


def test_konfai_cli_resume_continues_training(tmp_path: Path) -> None:
    experiment_dir = tmp_path / "experiment_resume"
    train_name = "RESUME_E2E"
    paths = _prepare_experiment_dir(experiment_dir, train_name)

    # Keep every checkpoint (BEST mode prunes history, including on resume) so the
    # full epoch/iteration timeline stays observable on disk.
    config_path = experiment_dir / "Config.yml"
    config_text = _replace_once(
        config_path.read_text(encoding="utf-8"),
        "save_checkpoint_mode: BEST",
        "save_checkpoint_mode: ALL",
    )
    config_path.write_text(config_text, encoding="utf-8")
    # RESUME continues to the epoch count of the config it is given: extend 2 -> 4.
    resume_config_path = experiment_dir / "ConfigResume.yml"
    resume_config_path.write_text(
        _replace_once(config_text, f"epochs: {EPOCHS_INITIAL}", f"epochs: {EPOCHS_TOTAL}"),
        encoding="utf-8",
    )

    cli = _konfai_cli_command()
    subprocess.run(
        [*cli, "TRAIN", "-y", "--cpu", "1", "-q", "-c", "Config.yml"],
        cwd=experiment_dir,
        env=_subprocess_env(),
        check=True,
    )

    checkpoints_dir = paths["checkpoints_dir"] / train_name
    initial_checkpoints = _read_checkpoints(checkpoints_dir)
    assert initial_checkpoints
    last_checkpoint = _latest(initial_checkpoints)
    epoch_end = int(initial_checkpoints[last_checkpoint]["epoch"])
    it_end = int(initial_checkpoints[last_checkpoint]["it"])
    assert epoch_end == EPOCHS_INITIAL - 1
    assert it_end >= EPOCHS_INITIAL and it_end % EPOCHS_INITIAL == 0
    its_per_epoch = it_end // EPOCHS_INITIAL

    subprocess.run(
        [*cli, "RESUME", "-y", "--cpu", "1", "-q", "-c", "ConfigResume.yml", "--model", str(last_checkpoint)],
        cwd=experiment_dir,
        env=_subprocess_env(),
        check=True,
    )

    final_checkpoints = _read_checkpoints(checkpoints_dir)
    # RESUME must not wipe the workspace (a TRAIN-style restart deletes Checkpoints/<name>).
    assert set(initial_checkpoints) < set(final_checkpoints)
    new_checkpoints = {path: meta for path, meta in final_checkpoints.items() if path not in initial_checkpoints}
    new_its = sorted(int(meta["it"]) for meta in new_checkpoints.values())
    new_epochs = sorted(int(meta["epoch"]) for meta in new_checkpoints.values())

    # Continuity: counters resume from the loaded checkpoint instead of restarting at 0.
    assert min(new_its) == it_end + 1
    assert min(new_epochs) == epoch_end
    # The run re-executes the checkpoint's epoch index, then the remaining epochs up to
    # EPOCHS_TOTAL: exactly (EPOCHS_TOTAL - epoch_end) * its_per_epoch extra iterations.
    # A silent restart from scratch would end at EPOCHS_TOTAL * its_per_epoch instead.
    epochs_rerun = EPOCHS_TOTAL - epoch_end
    assert max(new_epochs) == EPOCHS_TOTAL - 1
    assert max(new_its) == it_end + epochs_rerun * its_per_epoch
    # One checkpoint per training iteration (it_validation: 1) plus the final exit save.
    assert len(new_checkpoints) == epochs_rerun * its_per_epoch + 1

    # The optimizer state itself round-tripped: AdamW step counters equal the total
    # number of iterations across both runs (not just the resumed run's own count).
    final_checkpoint = _latest(new_checkpoints)
    final_steps = _optimizer_steps(new_checkpoints[final_checkpoint])
    assert final_steps
    assert all(step == max(new_its) for step in final_steps)

    # Training genuinely progressed after the resume point.
    weights_before = _flatten_model_weights(initial_checkpoints[last_checkpoint])
    weights_after = _flatten_model_weights(new_checkpoints[final_checkpoint])
    assert weights_before.keys() == weights_after.keys()
    float_names = [name for name, tensor in weights_before.items() if tensor.dtype.is_floating_point]
    assert float_names
    assert any((weights_after[name] - weights_before[name]).abs().max().item() > 0 for name in float_names)

    # The resumed model is still usable end-to-end and predicts finite values.
    subprocess.run(
        [*cli, "PREDICTION", "-y", "--cpu", "1", "-q", "-c", "Prediction.yml", "--models", str(final_checkpoint)],
        cwd=experiment_dir,
        env=_subprocess_env(),
        check=True,
    )
    expected_cases = sorted(path.name for path in paths["dataset_dir"].iterdir() if path.is_dir())
    predicted = sorted((experiment_dir / "Predictions" / train_name / "Dataset").rglob("sCT.mha"))
    assert sorted(path.parent.name for path in predicted) == expected_cases
    for path in predicted:
        array = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(str(path)))
        assert array.shape == (3, 16, 16)
        assert np.isfinite(array).all()
