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

"""Tests for ``konfai.utils.runtime``: workflow guards, environment normalisation,
overwrite confirmation, distributed-launch bookkeeping, and progress/DDP
synchronisation."""

import contextlib
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import konfai as konfai_module
import konfai.utils.runtime as rt
import pytest
from konfai.evaluator import Evaluator
from konfai.predictor import Predictor
from konfai.trainer import Trainer
from konfai.utils.errors import ConfigError
from konfai.utils.runtime import (
    DistributedObject,
    State,
    configure_workflow_environment,
    confirm_overwrite_or_raise,
    execute_distributed_object,
    is_interactive_session,
)

# ---------------------------------------------------------------------------
# Workflow guards, environment normalisation, overwrite confirmation, and
# distributed-launch bookkeeping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("factory", [Trainer, Predictor, Evaluator])
def test_core_workflows_raise_config_error_when_mode_is_not_done(
    monkeypatch: pytest.MonkeyPatch,
    factory: type[Trainer] | type[Predictor] | type[Evaluator],
) -> None:
    monkeypatch.setenv("KONFAI_CONFIG_MODE", "default")

    with pytest.raises(ConfigError, match="KONFAI_CONFIG_MODE='Done'"):
        factory()


def test_configure_workflow_environment_normalizes_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("KONFAI_config_file", raising=False)
    monkeypatch.delenv("KONFAI_ROOT", raising=False)
    monkeypatch.delenv("KONFAI_STATE", raising=False)
    monkeypatch.delenv("KONFAI_STATISTICS_DIRECTORY", raising=False)

    configure_workflow_environment(
        config_path=tmp_path / "Config.yml",
        root="Trainer",
        state=State.TRAIN,
        path_env={"KONFAI_STATISTICS_DIRECTORY": tmp_path / "Statistics"},
    )

    assert Path(os.environ["KONFAI_config_file"]).name == "Config.yml"
    assert os.environ["KONFAI_ROOT"] == "Trainer"
    assert os.environ["KONFAI_STATE"] == str(State.TRAIN)
    assert Path(os.environ["KONFAI_STATISTICS_DIRECTORY"]).name == "Statistics"


def test_confirm_overwrite_or_raise_requires_flag_in_non_interactive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KONFAI_OVERWRITE", raising=False)
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: False))
    monkeypatch.setattr(sys, "stdout", SimpleNamespace(isatty=lambda: False))

    with pytest.raises(ConfigError, match="Pass -y/--overwrite"):
        confirm_overwrite_or_raise(Path("/tmp/output"), "prediction", ConfigError)


def test_confirm_overwrite_or_raise_accepts_yes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KONFAI_OVERWRITE", raising=False)
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(sys, "stdout", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr("builtins.input", lambda prompt: "yes")

    confirm_overwrite_or_raise(Path("/tmp/output"), "prediction", ConfigError)


def test_confirm_overwrite_or_raise_rejects_decline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KONFAI_OVERWRITE", raising=False)
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(sys, "stdout", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr("builtins.input", lambda prompt: "no")

    with pytest.raises(ConfigError, match="Overwrite was declined"):
        confirm_overwrite_or_raise(Path("/tmp/output"), "prediction", ConfigError)


def test_execute_distributed_object_sets_shared_master_port_without_forcing_launch_blocking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("KONFAI_MASTER_PORT", raising=False)
    monkeypatch.delenv("CUDA_LAUNCH_BLOCKING", raising=False)

    class DummyContext:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, value, traceback) -> None:
            return None

    class DummyDistributed(DistributedObject):
        def __init__(self) -> None:
            super().__init__("dummy")

        def setup(self, world_size: int):
            self.dataloader = [[] for _ in range(world_size)]

        def run_process(self, world_size: int, global_rank: int, local_rank: int, dataloaders):
            raise AssertionError("run_process should not be called in this unit test")

    spawn_calls: dict[str, object] = {}

    def fake_spawn(fn, nprocs: int, *args, **kwargs) -> None:
        spawn_calls["fn"] = fn
        spawn_calls["nprocs"] = nprocs
        spawn_calls["master_port"] = os.environ["KONFAI_MASTER_PORT"]
        spawn_calls["cuda_visible_devices"] = os.environ["CUDA_VISIBLE_DEVICES"]

    monkeypatch.setattr("konfai.utils.runtime.Log", DummyContext)
    monkeypatch.setattr("konfai.utils.runtime.TensorBoard", DummyContext)
    monkeypatch.setattr("konfai.utils.runtime.mp.spawn", fake_spawn)

    execute_distributed_object(DummyDistributed(), gpu=[0, 1], cpu=1, quiet=True)

    assert str(spawn_calls["master_port"]).isdigit()
    assert spawn_calls["cuda_visible_devices"] == "0,1"
    assert "KONFAI_MASTER_PORT" not in os.environ
    assert "CUDA_VISIBLE_DEVICES" not in os.environ
    assert "CUDA_LAUNCH_BLOCKING" not in os.environ
    assert spawn_calls["nprocs"] == 2


def test_cluster_resubmit_flag_warns_that_auto_requeue_is_not_wired(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    class DummyContext:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, value, traceback) -> None:
            return None

    class DummyExecutor:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def update_parameters(self, *_args, **_kwargs) -> None:
            pass

        def submit(self, *_args, **_kwargs) -> None:
            pass

    class DummyDistributed(DistributedObject):
        def __init__(self) -> None:
            super().__init__("dummy")

        def setup(self, world_size: int):
            self.dataloader = [[] for _ in range(world_size)]

        def run_process(self, world_size, global_rank, local_rank, dataloaders):
            raise AssertionError("run_process should not be called on the submitting side")

    monkeypatch.setattr("konfai.utils.runtime.Log", DummyContext)
    monkeypatch.setattr("konfai.utils.runtime.TensorBoard", DummyContext)
    monkeypatch.setitem(sys.modules, "submitit", SimpleNamespace(AutoExecutor=DummyExecutor))

    cluster_kwargs = {"name": "job", "memory": 8, "num_nodes": 1, "time_limit": 60, "resubmit": True}
    execute_distributed_object(DummyDistributed(), gpu=[0], cpu=1, quiet=True, cluster_kwargs=cluster_kwargs)

    assert "--resubmit is not implemented" in capsys.readouterr().out


def test_get_available_devices_maps_visible_env_ids_to_local_torch_indices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,5")

    queried_indices: list[int] = []

    def fake_get_device_name(index: int) -> str:
        queried_indices.append(index)
        return f"GPU{index}"

    # get_available_devices imports get_device_name lazily from torch.cuda, so patch it at the source.
    monkeypatch.setattr("torch.cuda.get_device_name", fake_get_device_name)

    devices_index, devices_name = konfai_module.get_available_devices()

    assert devices_index == [3, 5]
    assert devices_name == ["GPU0", "GPU1"]
    assert queried_indices == [0, 1]


# ---------------------------------------------------------------------------
# Progress/DDP synchronisation
# ---------------------------------------------------------------------------


def test_synchronize_data_gathers_on_cpu(monkeypatch):
    """gloo/CPU multi-process must still all_gather (not fall back to local rank)."""
    calls = {}

    def fake_all_gather_object(outputs, data):
        calls["called"] = True
        for i in range(len(outputs)):
            outputs[i] = data

    def fail_set_device(*_args, **_kwargs):
        raise AssertionError("set_device must not be called when CUDA is unavailable")

    monkeypatch.setattr(rt.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(rt.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(rt.torch.cuda, "set_device", fail_set_device)
    monkeypatch.setattr(rt.dist, "all_gather_object", fake_all_gather_object)

    result = rt.synchronize_data(3, 0, {"a": 1})

    assert calls.get("called") is True
    assert result == [{"a": 1}, {"a": 1}, {"a": 1}]


def test_synchronize_data_sets_device_on_cuda(monkeypatch):
    """When CUDA is available the target device is selected before gathering."""
    seen = {}

    def fake_all_gather_object(outputs, data):
        for i in range(len(outputs)):
            outputs[i] = data

    monkeypatch.setattr(rt.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(rt.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(rt.torch.cuda, "set_device", lambda gpu: seen.setdefault("gpu", gpu))
    monkeypatch.setattr(rt.dist, "all_gather_object", fake_all_gather_object)

    result = rt.synchronize_data(2, 1, {"b": 2})

    assert seen.get("gpu") == 1
    assert result == [{"b": 2}, {"b": 2}]


def test_synchronize_data_no_dist(monkeypatch):
    """Without an active process group the local data is returned as-is."""
    monkeypatch.setattr(rt.dist, "is_initialized", lambda: False)
    assert rt.synchronize_data(4, 0, {"a": 1}) == [{"a": 1}]


def _run_execute(monkeypatch, obj):
    monkeypatch.setattr(rt, "Log", lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setattr(rt, "TensorBoard", lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setattr(rt.mp, "spawn", lambda *a, **k: None)
    rt.execute_distributed_object(obj, gpu=None, cpu=1)


def test_execute_seeds_parent_before_setup(monkeypatch):
    """The parent process (which runs the train/val split) must be seeded."""

    recorded = []

    class FakeObject(rt.DistributedObject):
        def __init__(self) -> None:
            super().__init__("fake-seeded")
            self.manual_seed = 123

        def setup(self, world_size: int) -> None:
            recorded.append(random.random())

        def run_process(self, *args, **kwargs) -> None:  # pragma: no cover - not spawned
            pass

    _run_execute(monkeypatch, FakeObject())
    _run_execute(monkeypatch, FakeObject())

    assert recorded[0] == recorded[1]


# ---------------------------------------------------------------------------
# is_interactive_session must not crash when stdout has no isatty
# ---------------------------------------------------------------------------
class _FakeTTY:
    def isatty(self) -> bool:
        return True


class _LogProxy:
    """Mimics Log/MinimalLog: write/flush/fileno only, no isatty."""

    def write(self, msg: str) -> None:
        pass

    def flush(self) -> None:
        pass


def test_is_interactive_session_survives_stdout_without_isatty(monkeypatch) -> None:
    # During a run stdout is swapped for a Log proxy that has no isatty; an unconditional
    # stdout.isatty() call raises AttributeError. It must degrade to non-interactive.
    monkeypatch.setattr(sys, "stdin", _FakeTTY())
    monkeypatch.setattr(sys, "stdout", _LogProxy())

    assert is_interactive_session() is False


def test_is_interactive_session_true_on_real_tty(monkeypatch) -> None:
    monkeypatch.setattr(sys, "stdin", _FakeTTY())
    monkeypatch.setattr(sys, "stdout", _FakeTTY())

    assert is_interactive_session() is True
