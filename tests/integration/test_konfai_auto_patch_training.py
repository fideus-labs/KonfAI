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

"""VRAM auto-patching in training (``patch_size`` with free ``0`` axes): an OOM at the first step
must shrink the free axes through the rank rendezvous, re-plan the grid, restart the run, and
train to completion (checkpoints produced)."""

import subprocess
import sys
from pathlib import Path

import pytest
from test_konfai_core_workflows import _prepare_experiment_dir, _subprocess_env
from test_konfai_ensemble_tta import _replace_once

pytestmark = pytest.mark.integration

pytest.importorskip("SimpleITK")

TRAIN_NAME = "AUTOPATCHTRAIN"

RUNNER_SOURCE = '''
import os
from pathlib import Path

import torch

import konfai.trainer as trainer_module
from konfai.trainer import build_train

ATTEMPTS = []


def install_auto_patch_probes() -> None:
    """Force the first attempt to OOM and stub the CUDA readings (this is a CPU-only run)."""
    original_run = trainer_module._Trainer.run

    def run_with_forced_oom(self):
        ATTEMPTS.append(list(self.dataloader_training.dataset.get_patch_config()[0]))
        if len(ATTEMPTS) == 1:
            raise torch.cuda.OutOfMemoryError("forced OOM: pretend the full-slice step does not fit")
        return original_run(self)

    trainer_module._Trainer.run = run_with_forced_oom
    trainer_module.Trainer._transient_at_oom = lambda self, device: None
    trainer_module.Trainer._usable_vram_after_oom = lambda self, device: 1.0


def main() -> None:
    root = Path.cwd()
    # The workflow normally runs in a spawned child, where a monkeypatch would not survive; run the
    # single rank IN-PROCESS so the forced OOM and the stubbed VRAM readings stay visible.
    os.environ["KONFAI_OVERWRITE"] = "True"
    os.environ["KONFAI_VERBOSE"] = "False"
    # Normally created by the Log wrapper of execute_distributed_object, which this bypasses.
    (root / "Statistics" / "__TRAIN_NAME__").mkdir(parents=True, exist_ok=True)
    install_auto_patch_probes()
    trainer = build_train(
        config=root / "ConfigAuto.yml",
        checkpoints_dir=root / "Checkpoints",
        statistics_dir=root / "Statistics",
    )
    with trainer as configured:
        configured.setup(1)
        configured(0)
    # Attempt 1: the free axes at full extent; attempt 2: one fixed 0.8 shrink of the free Y/X axes
    # (the pinned Z=1 never moves). Anything else means the restart loop did not do its job.
    if ATTEMPTS != [[1, 0, 0], [1, 12, 12]]:
        raise RuntimeError(f"unexpected restart sequence: {ATTEMPTS}")
    checkpoints = sorted((root / "Checkpoints" / "__TRAIN_NAME__").glob("*.pt"))
    if not checkpoints:
        raise RuntimeError("the restarted training produced no checkpoint")


if __name__ == "__main__":
    main()
'''


def test_auto_patch_training_restarts_and_completes(tmp_path: Path) -> None:
    experiment_dir = tmp_path / "experiment"
    _prepare_experiment_dir(experiment_dir, TRAIN_NAME)
    base = (experiment_dir / "Config.yml").read_text(encoding="utf-8")
    auto = _replace_once(base, "patch_size: [1, 16, 16]", "patch_size: [1, 0, 0]")
    auto = _replace_once(auto, "overlap: None", "overlap: 0")
    (experiment_dir / "ConfigAuto.yml").write_text(auto, encoding="utf-8")

    runner_path = experiment_dir / "run_auto_patch_training.py"
    runner_path.write_text(RUNNER_SOURCE.replace("__TRAIN_NAME__", TRAIN_NAME), encoding="utf-8")
    subprocess.run(
        [sys.executable, str(runner_path)],
        cwd=experiment_dir,
        env=_subprocess_env(),
        check=True,
    )
