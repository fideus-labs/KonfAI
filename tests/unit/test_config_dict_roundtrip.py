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

"""Regression test: a dict[str, primitive] default must survive the config write-back."""

from pathlib import Path

import pytest
from konfai.utils.config import apply_config


class _Root:
    def __init__(self, weights: dict[str, int] = {"mae": 1, "ssim": 2}) -> None:
        self.weights = weights


def _setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    config_path = tmp_path / "config.yml"
    config_path.write_text("Root: {}\n", encoding="utf-8")  # Root present but no 'weights'
    monkeypatch.setenv("KONFAI_config_file", str(config_path))
    monkeypatch.setenv("KONFAI_CONFIG_MODE", "Done")
    return config_path


def test_dict_of_primitives_default_round_trips(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = _setup(tmp_path, monkeypatch)

    # Run 1: the default materialises and is written back.
    first = apply_config("Root")(_Root)()
    assert first.weights == {"mae": 1, "ssim": 2}

    # The write-back must persist the values, not collapse the dict to an empty mapping.
    written = config_path.read_text(encoding="utf-8")
    assert "mae" in written and "ssim" in written

    # Run 2: reading the written file must return the same dict, not {} (the pre-fix behaviour).
    second = apply_config("Root")(_Root)()
    assert second.weights == {"mae": 1, "ssim": 2}
