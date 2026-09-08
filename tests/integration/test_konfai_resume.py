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

"""CPU CLI oracle: uninterrupted training equals two epochs plus checkpoint/RESUME."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from harness import konfai_cli_command, prepare_experiment_dir, replace_once, run_workflow

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


@pytest.mark.parametrize("stochastic", [False, True])
def test_konfai_cli_resume_continues_training(tmp_path: Path, stochastic: bool) -> None:
    experiment_dir = tmp_path / "experiment_resume"
    train_name = "RESUME_E2E"
    paths = prepare_experiment_dir(experiment_dir, train_name)

    # Keep every checkpoint (BEST mode prunes history, including on resume) so the
    # full epoch/iteration timeline stays observable on disk.
    config_path = experiment_dir / "Config.yml"
    config_text = replace_once(
        config_path.read_text(encoding="utf-8"),
        "save_checkpoint_mode: BEST",
        "save_checkpoint_mode: ALL",
    )
    config_text = replace_once(config_text, "    batch_size: 16", "    batch_size: 3\n    num_workers: 0")
    config_text = replace_once(
        config_text,
        "        ConstantLR:\n          factor: 1.0\n          total_iters: 1",
        "        StepLR:\n          step_size: 1\n          gamma: 0.8",
    )
    config_path.write_text(config_text, encoding="utf-8")
    # RESUME continues to the epoch count of the config it is given: extend 2 -> 4.
    resume_config_path = experiment_dir / "ConfigResume.yml"
    resume_config_path.write_text(
        replace_once(config_text, f"epochs: {EPOCHS_INITIAL}", f"epochs: {EPOCHS_TOTAL}"),
        encoding="utf-8",
    )

    full_dir = tmp_path / "uninterrupted"
    full_paths = prepare_experiment_dir(full_dir, train_name)
    (full_dir / "Config.yml").write_text(
        config_text.replace(str(paths["dataset_dir"]), str(full_paths["dataset_dir"])).replace(
            f"epochs: {EPOCHS_INITIAL}", f"epochs: {EPOCHS_TOTAL}"
        ),
        encoding="utf-8",
    )
    if stochastic:
        # Three batches/epoch and a two-batch window: odd epochs retain gradients,
        # even epochs produce eligible continuation checkpoints without extra steps.
        def add_stochastic_settings(text: str) -> str:
            return text.replace("  ema_decay: 0", "  ema_decay: 0.9").replace(
                "                    Constant:\n                      nb_step: 0\n                      value: 1",
                "                    CosineAnnealing:\n                      nb_step: 0\n"
                "                      start_value: 1\n                      eta_min: 0.1\n                      t_max: 20",
            )

        config_text = add_stochastic_settings(config_text)
        config_path.write_text(config_text, encoding="utf-8")
        resume_config_path.write_text(
            config_text.replace(f"epochs: {EPOCHS_INITIAL}", f"epochs: {EPOCHS_TOTAL}"), encoding="utf-8"
        )
        full_config = full_dir / "Config.yml"
        full_config.write_text(add_stochastic_settings(full_config.read_text()))
        for directory in (experiment_dir, full_dir):
            model_path = directory / "TinySynth.py"
            model_text = model_path.read_text(encoding="utf-8")
            model_text = replace_once(model_text, "import torch", "import random\nimport numpy as np\nimport torch")
            model_text = replace_once(
                model_text,
                "        return x * self.weight + self.bias",
                "        if self.training:\n"
                "            x = torch.nn.functional.dropout(x, p=0.25, training=True)\n"
                "            x = x * (0.8 + 0.1 * random.random() + 0.1 * np.random.random())\n"
                "        return x * self.weight + self.bias",
            )
            model_text = replace_once(
                model_text, "            dim=2,", "            dim=2,\n            nb_batch_per_step=2,"
            )
            model_path.write_text(model_text, encoding="utf-8")

    cli = konfai_cli_command()
    run_workflow([*cli, "TRAIN", "-y", "--cpu", "1", "-q", "-c", "Config.yml"], experiment_dir)

    checkpoints_dir = paths["checkpoints_dir"] / train_name
    initial_checkpoints = _read_checkpoints(checkpoints_dir)
    assert initial_checkpoints
    last_checkpoint = _latest(initial_checkpoints)
    assert (checkpoints_dir / "resume_latest.pt").is_file()
    epoch_end = int(initial_checkpoints[last_checkpoint]["epoch"])
    it_end = int(initial_checkpoints[last_checkpoint]["it"])
    assert epoch_end == EPOCHS_INITIAL - 1
    assert initial_checkpoints[last_checkpoint]["resume"]["next_epoch"] == EPOCHS_INITIAL
    assert initial_checkpoints[last_checkpoint]["resume"]["replay_limits"] == []
    assert it_end >= EPOCHS_INITIAL and it_end % EPOCHS_INITIAL == 0
    its_per_epoch = it_end // EPOCHS_INITIAL

    run_workflow(
        [
            *cli,
            "RESUME",
            "-y",
            "--cpu",
            "1",
            "-q",
            "-c",
            "ConfigResume.yml",
            "--model",
            str(checkpoints_dir / "resume_latest.pt"),
        ],
        experiment_dir,
    )

    final_checkpoints = _read_checkpoints(checkpoints_dir)
    # RESUME must not wipe the workspace (a TRAIN-style restart deletes Checkpoints/<name>).
    assert set(initial_checkpoints) < set(final_checkpoints)
    new_checkpoints = {path: meta for path, meta in final_checkpoints.items() if path not in initial_checkpoints}
    new_its = sorted(int(meta["it"]) for meta in new_checkpoints.values())
    new_epochs = sorted(int(meta["epoch"]) for meta in new_checkpoints.values())

    # Continuity: counters resume from the loaded checkpoint instead of restarting at 0.
    assert min(new_its) == it_end + 1
    assert min(new_epochs) == EPOCHS_INITIAL
    epochs_rerun = EPOCHS_TOTAL - EPOCHS_INITIAL
    assert max(new_epochs) == EPOCHS_TOTAL - 1
    assert max(new_its) == it_end + epochs_rerun * its_per_epoch
    # One checkpoint per training iteration (it_validation: 1). The exit no longer writes a
    # duplicate of the last scored save: an exit save happens only when iterations advanced
    # past it (a crash), and it is then named crash_*.pt.
    assert len(new_checkpoints) == epochs_rerun * its_per_epoch

    # The optimizer state itself round-tripped: AdamW step counters equal the total
    # number of iterations across both runs (not just the resumed run's own count).
    final_checkpoint = _latest(new_checkpoints)
    final_steps = _optimizer_steps(new_checkpoints[final_checkpoint])
    assert final_steps
    assert all(step == max(new_its) // (2 if stochastic else 1) for step in final_steps)

    # Training genuinely progressed after the resume point.
    weights_before = _flatten_model_weights(initial_checkpoints[last_checkpoint])
    weights_after = _flatten_model_weights(new_checkpoints[final_checkpoint])
    assert weights_before.keys() == weights_after.keys()
    float_names = [name for name, tensor in weights_before.items() if tensor.dtype.is_floating_point]
    assert float_names
    assert any((weights_after[name] - weights_before[name]).abs().max().item() > 0 for name in float_names)

    run_workflow([*cli, "TRAIN", "-y", "--cpu", "1", "-q", "-c", "Config.yml"], full_dir)
    full_checkpoints = _read_checkpoints(full_paths["checkpoints_dir"] / train_name)
    full_final = full_checkpoints[_latest(full_checkpoints)]
    resumed_final = new_checkpoints[final_checkpoint]
    assert resumed_final["resume"]["measure_by_rank"] == full_final["resume"]["measure_by_rank"]
    assert resumed_final["it"] == full_final["it"] == EPOCHS_TOTAL * its_per_epoch
    assert _optimizer_steps(resumed_final) == _optimizer_steps(full_final)
    full_weights = _flatten_model_weights(full_final)
    for name in weights_after:
        torch.testing.assert_close(weights_after[name], full_weights[name], rtol=0, atol=0)
    for key, value in resumed_final.items():
        if key.endswith("_nb_lr_update") or key.endswith("_schedulers_state_dict"):
            assert value == full_final[key]
        elif key.endswith("_optimizer_state_dict"):
            assert value["param_groups"] == full_final[key]["param_groups"]
            for parameter, state in value["state"].items():
                for name, tensor in state.items():
                    torch.testing.assert_close(tensor, full_final[key]["state"][parameter][name], rtol=0, atol=0)
    if stochastic:
        assert resumed_final["Model_EMA_n_averaged"] == full_final["Model_EMA_n_averaged"]
        for network, state in resumed_final["Model_EMA"].items():
            for name, tensor in state.items():
                torch.testing.assert_close(tensor, full_final["Model_EMA"][network][name], rtol=0, atol=0)
        for checkpoint in final_checkpoints.values():
            if checkpoint["it"] % its_per_epoch == 0:
                expected_kind = "epoch_boundary" if (checkpoint["epoch"] + 1) % 2 == 0 else "unavailable"
                assert checkpoint["resume"]["kind"] == expected_kind

    # The resumed model is still usable end-to-end and predicts finite values.
    run_workflow(
        [*cli, "PREDICTION", "-y", "--cpu", "1", "-q", "-c", "Prediction.yml", "--models", str(final_checkpoint)],
        experiment_dir,
    )
    expected_cases = sorted(path.name for path in paths["dataset_dir"].iterdir() if path.is_dir())
    predicted = sorted((experiment_dir / "Predictions" / train_name / "Dataset").rglob("sCT.mha"))
    assert sorted(path.parent.name for path in predicted) == expected_cases
    for path in predicted:
        array = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(str(path)))
        assert array.shape == (3, 16, 16)
        assert np.isfinite(array).all()
