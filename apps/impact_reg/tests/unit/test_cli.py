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
    """Run ``cli.main`` on ``argv``; return the SystemExit code (0 if none)."""
    monkeypatch.setattr(sys, "argv", ["impact-reg-konfai", *argv])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    return int(exc.value.code or 0)


def _stub_app(calls: dict[str, dict]) -> type:
    """A drop-in for ``ImpactRegKonfAIApp`` that records the dispatched call instead of resolving any app."""

    class _StubApp:
        def __init__(self, **_: object) -> None:
            pass

        def evaluate(self, **kwargs: object) -> None:
            calls["evaluate"] = kwargs

        def uncertainty(self, **kwargs: object) -> None:
            calls["uncertainty"] = kwargs

    return _StubApp


@pytest.mark.parametrize("subcommand", ["register", "eval", "uncertainty"])
def test_subcommands_are_wired(monkeypatch: pytest.MonkeyPatch, subcommand: str) -> None:
    assert _run(monkeypatch, [subcommand, "--help"]) == 0


def test_register_rejects_preset_flag(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    # --preset belongs to eval/uncertainty, NOT register. Give register a valid positional preset so the ONLY
    # possible failure is the flag itself: argparse must reject it as unrecognized (not fail on a missing
    # positional, which would mask a regression that quietly accepted --preset).
    code = _run(monkeypatch, ["register", "FireANTs_SyN", "--preset", "X", "-f", "a.mha", "-m", "b.mha"])
    assert code == 2
    err = capsys.readouterr().err
    assert "unrecognized arguments" in err and "--preset" in err


def test_eval_forwards_preset_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    # --preset must be parsed and forwarded to app.evaluate. Stub the app so nothing is resolved; a valid
    # dispatch (rather than an "unrecognized arguments" argparse error) proves the flag is really wired.
    calls: dict[str, dict] = {}
    monkeypatch.setattr(cli, "ImpactRegKonfAIApp", _stub_app(calls))
    monkeypatch.setattr(
        sys, "argv", ["impact-reg-konfai", "eval", "--preset", "FireANTs_SyN", "-f", "a.mha", "-m", "b.mha"]
    )
    cli.main()  # no SystemExit: argparse accepted --preset and dispatch reached the stub
    assert calls["evaluate"]["preset"] == "FireANTs_SyN"


def test_uncertainty_forwards_preset_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, dict] = {}
    monkeypatch.setattr(cli, "ImpactRegKonfAIApp", _stub_app(calls))
    monkeypatch.setattr(
        sys, "argv", ["impact-reg-konfai", "uncertainty", "--preset", "FireANTs_SyN", "--dvf", "a.mha", "b.mha"]
    )
    cli.main()
    assert calls["uncertainty"]["preset"] == "FireANTs_SyN"


def test_unknown_subcommand_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _run(monkeypatch, ["evaluate", "--help"]) == 2  # it is "eval", not "evaluate"
