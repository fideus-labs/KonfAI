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

"""Regression tests for the runtime progress/DDP audit fixes (see AUDIT.md)."""

import contextlib
import os
import random

os.environ.setdefault("KONFAI_config_file", "/tmp/konfai-none.yml")
os.environ.setdefault("KONFAI_CONFIG_MODE", "Done")

import konfai.utils.runtime as rt  # noqa: E402


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
