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

"""is_interactive_session must not crash when stdout has no isatty."""

import sys

from konfai.utils.runtime import is_interactive_session


class _FakeTTY:
    def isatty(self) -> bool:
        return True


class _LogProxy:
    """Mimics Log/MinimalLog: write/flush/fileno only, no isatty."""

    def write(self, msg: str) -> None:
        pass

    def flush(self) -> None:
        pass


def test_is_interactive_session_survives_stdout_without_isatty(monkeypatch) -> None:
    # During a run stdout is swapped for a Log proxy that has no isatty; an unconditional
    # stdout.isatty() call raises AttributeError. It must degrade to non-interactive.
    monkeypatch.setattr(sys, "stdin", _FakeTTY())
    monkeypatch.setattr(sys, "stdout", _LogProxy())

    assert is_interactive_session() is False


def test_is_interactive_session_true_on_real_tty(monkeypatch) -> None:
    monkeypatch.setattr(sys, "stdin", _FakeTTY())
    monkeypatch.setattr(sys, "stdout", _FakeTTY())

    assert is_interactive_session() is True
