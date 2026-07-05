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

"""Guard tests for ``konfai.utils.runtime``: workflow preconditions, environment
normalisation, overwrite confirmation, and distributed-launch bookkeeping."""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import konfai as konfai_module
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
)


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
