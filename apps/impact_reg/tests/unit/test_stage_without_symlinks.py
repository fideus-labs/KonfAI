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

"""Staging works where symlinks are refused (Windows without Developer Mode: WinError 1314), GAP5-07."""

import os
from pathlib import Path

import pytest
from impact_reg_konfai.impact_reg import _stage_group


def test_a_file_and_a_store_are_staged_when_symlinks_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refused(*args, **kwargs):
        raise OSError(1314, "A required privilege is not held by the client")

    monkeypatch.setattr(os, "symlink", refused)
    monkeypatch.setattr(Path, "symlink_to", refused)
    image = tmp_path / "in" / "fixed.mha"
    image.parent.mkdir()
    image.write_bytes(b"image")
    store = tmp_path / "in" / "moving.ome.zarr"
    (store / "0").mkdir(parents=True)
    (store / "0" / "chunk").write_bytes(b"chunk")

    _stage_group(tmp_path / "staged_f", "Fixed", {"case": image})
    _stage_group(tmp_path / "staged_m", "Moving", {"case": store})

    assert (tmp_path / "staged_f" / "Fixed" / "case" / "Fixed.mha").read_bytes() == b"image"
    assert (tmp_path / "staged_m" / "Moving" / "case" / "Moving.ome.zarr" / "0" / "chunk").read_bytes() == b"chunk"
