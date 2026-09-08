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


def test_validation_reads_a_scratch_copy_beside_the_original(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Building a workflow rewrites the config it reads (KONFAI_CONFIG_MODE='Done'). The child reads a
    copy beside the original, byte for byte (CRLF included), so the original is never written: an
    edit made while the child runs survives, and a timeout kill has nothing to restore."""
    config_path = tmp_path / "Config.yml"
    authored = b"Trainer:\r\n  train_name: X\r\n"
    config_path.write_bytes(authored)
    seen: list[Path] = []

    def build(**kwargs: Any) -> object:
        read = Path(kwargs["config"])
        seen.append(read)
        assert read.parent == config_path.parent and read != config_path
        assert read.read_bytes() == authored
        read.write_text("Trainer:\n  train_name: X\n  epochs: 100\n", encoding="utf-8")  # the binder's write-back
        config_path.write_bytes(authored + b"  epochs: 3\r\n")  # an edit landing meanwhile
        return object()

    monkeypatch.setattr(runner, "build_train", build)
    payload = runner.validate_workflow_api(
        workflow="train",
        level="instantiate",
        workspace_dir=str(tmp_path),
        config=str(config_path),
        validate_root=str(tmp_path / "validate"),
    )

    assert payload["ok"] is True and payload["config_path"] == str(config_path)
    assert config_path.read_bytes() == authored + b"  epochs: 3\r\n"  # the edit, not the write-back
    assert seen and not seen[0].exists()  # the copy is gone
    assert [entry.name for entry in tmp_path.iterdir() if entry.name.startswith(".")] == []


def test_a_scratch_copy_a_killed_child_left_is_swept_by_the_parent(tmp_path: Path) -> None:
    """The child removes its copy in its own ``finally``, which a timeout kill never reaches; the
    parent sweeps what is left beside the original, including when the call leaves by an exception."""
    config_path = tmp_path / "Transform.yml"
    authored = "Transformer:\n  name: TEST\n"
    config_path.write_text(authored, encoding="utf-8")
    with pytest.raises(RuntimeError), runner.discard_scratch_configs(config_path) as leftover:
        leftover.write_text(f"{authored}  manual_seed: 0\n", encoding="utf-8")
        raise RuntimeError("Isolated subprocess failed.")

    assert config_path.read_text(encoding="utf-8") == authored
    assert [entry.name for entry in tmp_path.iterdir()] == ["Transform.yml"]


def test_the_subprocess_tail_is_read_from_the_end_of_the_file(tmp_path: Path) -> None:
    output = tmp_path / "subprocess.log"
    output.write_text("x" * 5_000_000 + "\nlast line\n", encoding="utf-8")
    tail = runner._tail(output)
    assert tail.endswith("last line") and len(tail) <= runner._OUTPUT_TAIL


def test_plan_transform_is_covered_by_that_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, load_mcp_server: Callable[[], ModuleType]
) -> None:
    """The tool builds a Transformer from the agent's own config in a child that reads a scratch
    copy; a child killed mid-plan leaves that copy, which the tool sweeps on its way out."""
    monkeypatch.setenv("KONFAI_MCP_WORKSPACES_ROOT", str(tmp_path / "workspaces"))
    server = load_mcp_server()
    config_path = Path(server.SESSION.config_path("transform"))
    config_path.parent.mkdir(parents=True, exist_ok=True)
    authored = "Transformer:\n  name: TEST\n"
    config_path.write_text(authored, encoding="utf-8")
    leftovers: list[Path] = []

    def killed_child(_target: str, kwargs: dict[str, Any]) -> None:
        leftover = Path(kwargs["scratch_path"])
        leftovers.append(leftover)
        leftover.write_text(f"{authored}  manual_seed: 0\n", encoding="utf-8")
        raise RuntimeError("Isolated subprocess failed.")

    monkeypatch.setattr(server, "_run_api_in_subprocess", killed_child)

    with pytest.raises(RuntimeError):
        server.plan_transform(cpu=1)

    assert config_path.read_text(encoding="utf-8") == authored
    assert leftovers and not leftovers[0].exists()


def test_concurrent_validations_only_remove_their_own_scratch(tmp_path: Path) -> None:
    config = tmp_path / "Config.yml"
    config.write_bytes(b"Trainer:\r\n  epochs: 1\r\n")
    with runner.scratch_config(config) as live:
        with runner.discard_scratch_configs(config) as owned:
            owned.write_text("child killed before cleanup")
            assert live.exists() and owned.exists()
        assert live.exists() and not owned.exists()
    assert not live.exists()
    assert config.read_bytes() == b"Trainer:\r\n  epochs: 1\r\n"


@pytest.mark.parametrize("failure", ["construct", "start"])
def test_subprocess_start_failure_closes_queue_and_removes_output(tmp_path, monkeypatch, failure):
    from types import SimpleNamespace

    closed = []
    output = tmp_path / "subprocess"
    output.mkdir()

    def fail():
        raise RuntimeError("spawn setup failed")

    def process(**kwargs):
        if failure == "construct":
            fail()
        return SimpleNamespace(start=fail, is_alive=lambda: False)

    context = SimpleNamespace(Queue=lambda: SimpleNamespace(close=lambda: closed.append(True)), Process=process)
    monkeypatch.setattr(runner.multiprocessing, "get_context", lambda mode: context)
    monkeypatch.setattr(runner.tempfile, "mkdtemp", lambda **kwargs: str(output))
    with pytest.raises(RuntimeError, match="spawn setup failed"):
        runner.run_api_in_subprocess("unused:target", {})
    assert closed == [True] and not output.exists()


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
