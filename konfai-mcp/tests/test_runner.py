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

"""konfai_mcp/runner.py contracts: bounded final join on a wedged spawn child, config-restore
failures surfaced in the payload, the parent-side config guard around a child that may be killed,
and a non-differentiable loss propagating into the smoke-test ok flag."""

import sys
import time
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
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
    # join is bounded and escalates to terminate/kill. An unbounded join would hang here.
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


def test_preserved_config_puts_back_what_a_child_that_never_returned_rewrote(tmp_path: Path) -> None:
    """The child restores the config in its own ``finally``, which a timeout kill never reaches.

    Only the parent is guaranteed to outlive the child, so its snapshot is the one that has to hold --
    including when the call leaves by an exception rather than a return.
    """
    config_path = tmp_path / "Transform.yml"
    authored = "Transformer:\n  name: TEST\n"
    config_path.write_text(authored, encoding="utf-8")

    with pytest.raises(RuntimeError), runner.preserved_config(config_path):
        config_path.write_text(f"{authored}  manual_seed: 0\n  Dataset:\n    subset: None\n", encoding="utf-8")
        raise RuntimeError("Isolated subprocess failed.")

    assert config_path.read_text(encoding="utf-8") == authored


def test_a_restore_that_fails_part_way_leaves_a_whole_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Nothing backs this guard up, so a half-written restore is the file the agent is left with.

    The KonfAI-rewritten config is verbose but valid; a fragment of the authored one is neither, and
    the authored bytes are already off disk by the time the restore starts.
    """
    config_path = tmp_path / "Transform.yml"
    authored = "Transformer:\n  name: TEST\n" + "  # a line the author wrote\n" * 40
    config_path.write_text(authored, encoding="utf-8")
    rewritten = "Transformer:\n  name: TEST\n  manual_seed: 0\n"

    def truncate_then_fail(self: Path, *_args: Any, **_kwargs: Any) -> int:
        # What a real write does when the device fills: the file is opened truncated, then fails.
        self.write_bytes(b"Transfor")
        raise OSError("no space left on device")

    with pytest.raises(OSError), runner.preserved_config(config_path):
        config_path.write_text(rewritten, encoding="utf-8")
        monkeypatch.setattr(Path, "write_text", truncate_then_fail)

    monkeypatch.undo()
    assert config_path.read_text(encoding="utf-8") == rewritten
    assert [entry.name for entry in tmp_path.iterdir()] == ["Transform.yml"]


def test_plan_transform_is_covered_by_that_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, load_mcp_server: Callable[[], ModuleType]
) -> None:
    """The tool builds a Transformer from the agent's own config, so it needs the guard the validate
    path already had: otherwise a plan slower than the timeout returns the config KonfAI-rewritten."""
    monkeypatch.setenv("KONFAI_MCP_WORKSPACES_ROOT", str(tmp_path / "workspaces"))
    server = load_mcp_server()
    config_path = Path(server.SESSION.config_path("transform"))
    config_path.parent.mkdir(parents=True, exist_ok=True)
    authored = "Transformer:\n  name: TEST\n"
    config_path.write_text(authored, encoding="utf-8")

    def killed_child(*_args: Any, **_kwargs: Any) -> None:
        config_path.write_text(f"{authored}  manual_seed: 0\n", encoding="utf-8")
        raise RuntimeError("Isolated subprocess failed.")

    monkeypatch.setattr(server, "_run_api_in_subprocess", killed_child)

    with pytest.raises(RuntimeError):
        server.plan_transform(cpu=1)

    assert config_path.read_text(encoding="utf-8") == authored


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
