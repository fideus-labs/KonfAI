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

"""The live-control channel yields each new revision's tunables exactly once, and never crashes on junk."""

import json
from pathlib import Path

from konfai.utils.live_control import LiveControl


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_missing_file_is_nothing_pending(tmp_path: Path) -> None:
    assert LiveControl(tmp_path / "control.json").take() is None


def test_a_new_revision_is_taken_once(tmp_path: Path) -> None:
    control = tmp_path / "control.json"
    live = LiveControl(control)
    _write(control, {"revision": 1, "lr": 1e-4, "it_validation": 5})

    assert live.take() == {"lr": 1e-4, "it_validation": 5}
    assert live.take() is None  # same revision -> already applied


def test_a_higher_revision_is_taken_again(tmp_path: Path) -> None:
    control = tmp_path / "control.json"
    live = LiveControl(control)
    _write(control, {"revision": 1, "lr": 1e-4})
    live.take()
    _write(control, {"revision": 2, "lr": 5e-5})

    assert live.take() == {"lr": 5e-5}


def test_a_stale_or_equal_revision_is_ignored(tmp_path: Path) -> None:
    control = tmp_path / "control.json"
    live = LiveControl(control)
    _write(control, {"revision": 3, "it_validation": 2})
    live.take()
    _write(control, {"revision": 2, "it_validation": 9})  # lower than last seen

    assert live.take() is None


def test_malformed_or_non_dict_content_is_nothing_pending(tmp_path: Path) -> None:
    control = tmp_path / "control.json"
    live = LiveControl(control)
    control.write_text("not json{", encoding="utf-8")
    assert live.take() is None
    control.write_text("[1, 2, 3]", encoding="utf-8")
    assert live.take() is None
