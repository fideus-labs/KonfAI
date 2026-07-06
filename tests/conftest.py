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

"""Shared fixtures for the KonfAI test suite."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _konfai_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Harmless per-test defaults for the mandatory KONFAI environment variables.

    ``Config()`` requires ``KONFAI_config_file`` and ``KONFAI_CONFIG_MODE`` (AGENTS.md §7).
    Tests that exercise the config engine override these with ``monkeypatch.setenv``.
    """
    monkeypatch.setenv("KONFAI_config_file", "/tmp/konfai-none.yml")
    monkeypatch.setenv("KONFAI_CONFIG_MODE", "Done")


@pytest.fixture
def write_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Callable[..., Path]:
    """Write a YAML config to ``tmp_path`` and point the KONFAI env vars at it."""

    def write(content: str, *, mode: str = "Done", name: str = "config.yml") -> Path:
        config_path = tmp_path / name
        config_path.write_text(content, encoding="utf-8")
        monkeypatch.setenv("KONFAI_config_file", str(config_path))
        monkeypatch.setenv("KONFAI_CONFIG_MODE", mode)
        return config_path

    return write


@pytest.fixture
def image_attributes():
    """Factory for an ``Attribute`` carrying Origin/Spacing/Direction geometry."""
    from konfai.utils.dataset import Attribute

    def make(origin: list[float], spacing: list[float]) -> Attribute:
        attributes = Attribute()
        attributes["Origin"] = np.asarray(origin, dtype=np.float64)
        attributes["Spacing"] = np.asarray(spacing, dtype=np.float64)
        attributes["Direction"] = np.eye(len(origin), dtype=np.float64).reshape(-1)
        return attributes

    return make
