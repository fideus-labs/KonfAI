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

from collections.abc import Callable
from types import ModuleType

import pytest


def _patch(
    module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    devices: list[tuple[int, str]],
    vram: dict[int, tuple[float, float]] | Callable[[list[int]], tuple[float, float]],
) -> None:
    monkeypatch.setattr(
        module, "konfai_get_available_devices", lambda: ([i for i, _ in devices], [n for _, n in devices])
    )
    monkeypatch.setattr(module, "konfai_get_ram", lambda: (8.0, 32.0))

    def _vram(selected: list[int]) -> tuple[float, float]:
        if callable(vram):
            return vram(selected)
        return sum(vram[i][0] for i in selected), sum(vram[i][1] for i in selected)

    monkeypatch.setattr(module, "konfai_get_vram", _vram)


def test_runtime_capabilities_breaks_vram_down_per_device_and_recommends_most_free(
    load_mcp_server: Callable[[], ModuleType], monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_mcp_server()
    _patch(module, monkeypatch, [(0, "A"), (1, "B")], {0: (20.0, 24.0), 1: (2.0, 24.0)})

    caps = module._runtime_capabilities()

    by_index = {device["index"]: device for device in caps["gpu"]["devices"]}
    assert by_index[0]["vram_free_gb"] == pytest.approx(4.0)
    assert by_index[1]["vram_free_gb"] == pytest.approx(22.0)
    assert by_index[0]["name"] == "A"
    # Aggregate VRAM is kept (derived from the per-device breakdown) for backward compatibility.
    assert caps["gpu"]["vram_gb"]["total"] == pytest.approx(48.0)
    assert caps["gpu"]["vram_gb"]["used"] == pytest.approx(22.0)
    # Recommends the GPU with the most free VRAM, not just the first.
    assert caps["recommended_device"] == {"gpu": [1]}


def test_runtime_capabilities_degrades_when_vram_unreadable(
    load_mcp_server: Callable[[], ModuleType], monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_mcp_server()

    def _raise(_selected: list[int]) -> tuple[float, float]:
        raise RuntimeError("nvidia-ml-py not installed")

    _patch(module, monkeypatch, [(0, "A")], _raise)

    caps = module._runtime_capabilities()

    assert caps["gpu"]["available"] is True
    assert caps["gpu"]["devices"][0]["vram_free_gb"] is None
    assert caps["gpu"]["vram_gb"]["total"] is None
    assert len(caps["warnings"]) == 1
    # VRAM unreadable -> cannot rank -> falls back to the first visible device.
    assert caps["recommended_device"] == {"gpu": [0]}


def test_vram_preflight_reports_free_for_the_devices_a_job_uses(
    load_mcp_server: Callable[[], ModuleType], monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_mcp_server()
    _patch(module, monkeypatch, [(0, "A"), (1, "B")], {0: (20.0, 24.0), 1: (2.0, 24.0)})

    preflight = module._vram_preflight(["1"])

    assert [device["index"] for device in preflight["devices"]] == [1]
    assert preflight["devices"][0]["name"] == "B"
    assert preflight["devices"][0]["vram_free_gb"] == pytest.approx(22.0)
    assert "batch_size" in preflight["guidance"]


def test_vram_preflight_is_none_for_cpu_or_deviceless_jobs(
    load_mcp_server: Callable[[], ModuleType], monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_mcp_server()
    _patch(module, monkeypatch, [(0, "A")], {0: (1.0, 24.0)})

    assert module._vram_preflight(None) is None
    assert module._vram_preflight([]) is None
    assert module._vram_preflight(["cpu"]) is None
