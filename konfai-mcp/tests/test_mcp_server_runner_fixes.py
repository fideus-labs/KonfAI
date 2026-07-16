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

"""Regression tests for konfai_mcp.runner: bounded final join, config-restore visibility, and
propagation of a non-differentiable loss into the smoke-test ok flag."""

import sys
import time
from pathlib import Path
from typing import Any

import pytest

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))
TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))  # so the spawn child can import the wedge target

from konfai_mcp import runner  # noqa: E402


def test_run_api_in_subprocess_reaps_child_wedged_after_result() -> None:
    # A child that produced its result but will not exit must not hang the caller forever: the final
    # join is bounded and escalates to terminate/kill. Without the fix the unbounded join hangs here.
    start = time.monotonic()
    payload = runner.run_api_in_subprocess("_runner_wedge_target:wedge_after_result", {"value": 7}, timeout_s=0)
    elapsed = time.monotonic() - start
    assert payload == {"echoed": 7}
    # join(10) grace + terminate; must return well within the unbounded-hang regime.
    assert elapsed < 30, f"bounded join should reap the wedged child, took {elapsed:.1f}s"


def test_validate_config_restore_failure_is_surfaced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Building a workflow rewrites the config in place (KONFAI_CONFIG_MODE='Done'); if the authored bytes
    # cannot be restored, the mutation must be surfaced, never returned as a silent ok.
    config_path = tmp_path / "Config.yml"
    config_path.write_text("Trainer:\n  train_name: X\n", encoding="utf-8")

    # Reach the success payload without a real dataset/model.
    monkeypatch.setattr(runner, "build_train", lambda **_kwargs: object())

    # The restore writes a temp file and os.replace()s it onto the config; fail that commit step.
    def failing_replace(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("read-only filesystem")

    monkeypatch.setattr(runner.os, "replace", failing_replace)

    with pytest.warns(UserWarning, match="Failed to restore"):
        payload = runner.validate_workflow_api(
            workflow="train",
            level="instantiate",
            workspace_dir=str(tmp_path),
            config=str(config_path),
            validate_root=str(tmp_path / "validate"),
        )

    assert payload["ok"] is True  # the build itself succeeded
    assert payload["config_restore_failed"] == "read-only filesystem"  # the leak is recorded, not hidden


def test_smoke_test_non_differentiable_loss_is_not_ok(tmp_path: Path) -> None:
    # A criterion that returns a loss Tensor but cannot backprop cannot train a model. It must report
    # ok=False so the tool steers to fix it, not ok=True with backward_ok buried as a side field.
    (tmp_path / "DetachedLoss.py").write_text(
        "import torch\n\n\nclass Detached(torch.nn.Module):\n"
        "    def forward(self, output, target):\n"
        "        return (output - target).abs().mean().detach()\n",
        encoding="utf-8",
    )
    result = runner.smoke_test_component(
        classpath="DetachedLoss:Detached", kind="criterion", workspace_dir=str(tmp_path)
    )
    assert result["behaves_as"] == "loss"
    assert result["backward_ok"] is False
    assert result["ok"] is False
    assert "backward" in result.get("error", "").lower()
