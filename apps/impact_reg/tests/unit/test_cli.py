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

"""Smoke tests for the ``impact-reg-konfai`` argparse tree: which subcommands exist and how the preset is
passed. These lock the CLI contract the demo notebook / README depend on — ``register`` takes the preset(s)
as a **positional** argument, while ``eval`` / ``uncertainty`` take ``--preset``. No app is resolved (every
invocation short-circuits on ``--help`` or an argument error)."""

import sys

import pytest
from impact_reg_konfai import cli


def _run(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> int:
    monkeypatch.setattr(sys, "argv", ["impact-reg-konfai", *argv])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    return int(exc.value.code or 0)


@pytest.mark.parametrize("subcommand", ["register", "eval", "uncertainty"])
def test_subcommands_are_wired(monkeypatch: pytest.MonkeyPatch, subcommand: str) -> None:
    assert _run(monkeypatch, [subcommand, "--help"]) == 0


def test_register_takes_preset_as_positional_not_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    # --preset belongs to eval/uncertainty, NOT register: passing it to register is an argument error.
    assert _run(monkeypatch, ["register", "--preset", "FireANTs_SyN", "-f", "a.mha", "-m", "b.mha"]) == 2


def test_eval_and_uncertainty_take_preset_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    # --help after --preset resolves cleanly, so the flag is recognised on these subcommands.
    assert _run(monkeypatch, ["eval", "--preset", "FireANTs_SyN", "--help"]) == 0
    assert _run(monkeypatch, ["uncertainty", "--preset", "FireANTs_SyN", "--help"]) == 0


def test_unknown_subcommand_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _run(monkeypatch, ["evaluate", "--help"]) == 2  # it is "eval", not "evaluate"
