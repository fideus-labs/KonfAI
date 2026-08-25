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

"""Tests for how one training step is built: which optimizer implementation the config asks for."""

from pathlib import Path

import pytest
import torch
from konfai.network.network import OptimizerLoader, batched_step


@pytest.fixture
def on_gpu(monkeypatch: pytest.MonkeyPatch):
    """The run places the graph on a GPU. The parameters stay on the host: an optimizer is built on
    the launcher, before ``Network.to`` moves them, which is why the choice reads the run, not them."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")


@pytest.fixture
def on_cpu(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")


def _parameters() -> list[torch.nn.parameter.Parameter]:
    torch.manual_seed(0)
    return [torch.nn.Parameter(torch.randn(4, 3)), torch.nn.Parameter(torch.randn(5))]


def _steps(optimizer: torch.optim.Optimizer, parameters: list[torch.nn.parameter.Parameter]) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(7)
    for _ in range(100):
        for parameter in parameters:
            parameter.grad = torch.randn(parameter.shape, generator=generator)
        optimizer.step()
    return [parameter.detach().clone() for parameter in parameters]


def test_a_run_without_a_gpu_keeps_the_default_step(on_cpu) -> None:
    """Off CUDA nothing is forced: torch's own default stands, foreach costing 526 ms a step there
    against 2.0 ms single-tensor over the Segmentation example's 40 tensors."""
    optimizer = batched_step(torch.optim.AdamW, _parameters())(_parameters())

    assert optimizer.defaults["fused"] is None
    assert optimizer.defaults["foreach"] is None


def test_the_default_step_is_bit_identical_to_the_plain_class(on_cpu) -> None:
    """The wrapper only chooses an implementation: over 100 steps it moves no parameter bit."""
    plain, wrapped = _parameters(), _parameters()

    expected = _steps(torch.optim.AdamW(plain, lr=1e-3), plain)
    obtained = _steps(batched_step(torch.optim.AdamW, wrapped)(wrapped, lr=1e-3), wrapped)

    for reference, value in zip(expected, obtained, strict=True):
        assert torch.equal(reference, value)


def test_a_gpu_run_takes_the_fused_step_when_the_optimizer_has_one(on_gpu) -> None:
    parameters = _parameters()

    assert batched_step(torch.optim.AdamW, parameters)(parameters).defaults["fused"] is True


def test_a_gpu_run_falls_back_to_foreach_without_a_fused_step(on_gpu) -> None:
    parameters = _parameters()
    optimizer = batched_step(torch.optim.Adadelta, parameters)(parameters)

    assert "fused" not in optimizer.defaults
    assert optimizer.defaults["foreach"] is True


def test_an_integer_parameter_keeps_the_default_step(on_gpu) -> None:
    """A batched step reads floating-point tensors only."""
    parameters = [*_parameters(), torch.nn.Parameter(torch.zeros(3, dtype=torch.int64), requires_grad=False)]

    assert batched_step(torch.optim.AdamW, parameters)(parameters).defaults["fused"] is None


@pytest.mark.parametrize("asked", [{"fused": False}, {"foreach": True}, {"differentiable": True}])
def test_the_config_wins_over_the_chosen_step(on_gpu, asked: dict[str, bool]) -> None:
    """A value written in the config is passed through: only an unset pair is decided here."""
    parameters = _parameters()
    optimizer = batched_step(torch.optim.AdamW, parameters)(parameters, **asked)

    assert optimizer.defaults["fused"] is not True or asked.get("fused") is True
    for name, value in asked.items():
        assert optimizer.defaults[name] == value


def test_the_optimizer_keeps_its_own_config_keys(on_cpu, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The binder reads the optimizer class's signature, so the YAML keeps exactly its keys: the
    implementation is chosen under ``fused``/``foreach``, never beside them."""
    config = tmp_path / "Config.yml"
    config.write_text("Trainer:\n  Model:\n    UNet:\n      optimizer:\n        lr: 0.002\n", encoding="utf-8")
    monkeypatch.setenv("KONFAI_config_file", str(config))
    monkeypatch.setenv("KONFAI_CONFIG_MODE", "Done")
    monkeypatch.setenv("KONFAI_ROOT", "Trainer")

    optimizer = OptimizerLoader().get_optimizer("UNet", iter(_parameters()))

    assert optimizer.defaults["lr"] == 0.002
    written = config.read_text(encoding="utf-8")
    assert "fused: None" in written and "foreach: None" in written
