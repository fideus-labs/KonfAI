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

"""Smoke tests for the thin totalsegmentator_konfai CLI wrapper: the argparse tree built by
``konfai_apps.cli.build_app_cli`` exposes the app's own vocabulary and pins the right repo. No app is
resolved and no model is downloaded (every invocation short-circuits on ``--help``)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from totalsegmentator_konfai import cli


def _run(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> int:
    monkeypatch.setattr(sys, "argv", ["totalsegmentator-konfai", *argv])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    return int(exc.value.code or 0)


def test_repo_is_pinned() -> None:
    assert cli.TOTAL_SEGMENTATOR_KONFAI_REPO == "VBoussot/TotalSegmentator-KonfAI"
    assert callable(cli.main)


@pytest.mark.parametrize("subcommand", ["segment", "eval", "pipeline"])
def test_expected_subcommands_are_wired(monkeypatch: pytest.MonkeyPatch, subcommand: str) -> None:
    assert _run(monkeypatch, [subcommand, "--help"]) == 0


def test_unknown_subcommand_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _run(monkeypatch, ["definitely-not-a-command", "--help"]) == 2
