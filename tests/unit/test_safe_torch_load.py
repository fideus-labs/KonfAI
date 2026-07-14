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

from pathlib import Path

import pytest
import torch
from konfai.utils import runtime as runtime_module
from konfai.utils.runtime import safe_torch_load


class _NonSafeState:
    def __init__(self, value: int) -> None:
        self.value = value


def _spy_on_load(monkeypatch: pytest.MonkeyPatch) -> list[bool | None]:
    weights_only_calls: list[bool | None] = []
    original_load = torch.load

    def spy(*args, **kwargs):
        weights_only_calls.append(kwargs.get("weights_only"))
        return original_load(*args, **kwargs)

    monkeypatch.setattr(runtime_module.torch, "load", spy)
    return weights_only_calls


def test_safe_torch_load_uses_weights_only_true_for_plain_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "plain.pt"
    torch.save({"weight": torch.tensor([1.0, 2.0]), "loss": 0.5}, checkpoint)

    calls = _spy_on_load(monkeypatch)
    loaded = safe_torch_load(checkpoint, "cpu")

    assert calls == [True]
    assert torch.equal(loaded["weight"], torch.tensor([1.0, 2.0]))
    assert loaded["loss"] == 0.5


def test_safe_torch_load_falls_back_for_non_safe_objects(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checkpoint = tmp_path / "object.pt"
    torch.save({"state": _NonSafeState(7)}, checkpoint)

    calls = _spy_on_load(monkeypatch)
    loaded = safe_torch_load(checkpoint, "cpu")

    assert calls == [True, False]
    assert loaded["state"].value == 7


def test_safe_torch_load_downloads_https_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    weights_only_calls: list[bool | None] = []

    def fake_hub(url, *, map_location, weights_only):
        weights_only_calls.append(weights_only)
        assert url == "https://example.com/model.pt"
        return {"weight": torch.tensor([3.0])}

    monkeypatch.setattr(runtime_module.torch.hub, "load_state_dict_from_url", fake_hub)
    loaded = safe_torch_load("https://example.com/model.pt", "cpu")

    assert weights_only_calls == [True]
    assert torch.equal(loaded["weight"], torch.tensor([3.0]))


def test_safe_torch_load_does_not_fall_back_for_https(monkeypatch: pytest.MonkeyPatch) -> None:
    # A remote checkpoint is untrusted: if the safe load fails (e.g. a crafted payload), it must NOT
    # retry with weights_only=False, which would run arbitrary code from the download.
    weights_only_calls: list[bool | None] = []

    def fake_hub(url, *, map_location, weights_only):
        weights_only_calls.append(weights_only)
        raise RuntimeError("safe unpickler refused a stored object")

    monkeypatch.setattr(runtime_module.torch.hub, "load_state_dict_from_url", fake_hub)

    with pytest.raises(RuntimeError):
        safe_torch_load("https://example.com/model.pt", "cpu")

    assert weights_only_calls == [True]  # never retried with the unsafe unpickler
