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

"""Regression tests: server auth env resolution must not silently disable authentication."""

import os

import pytest
from konfai_apps.cli import _configure_server_auth_env


def test_custom_token_env_is_propagated_to_the_var_the_server_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    # The server enforces auth by reading KONFAI_API_TOKEN. A custom --token-env used to leave that
    # var empty, so the server started with auth silently off despite a configured token.
    monkeypatch.delenv("KONFAI_API_TOKEN", raising=False)
    monkeypatch.setenv("MY_TOKEN", "secret")

    _configure_server_auth_env("bearer", None, "MY_TOKEN")

    assert os.environ["KONFAI_API_TOKEN"] == "secret"


def test_auth_off_clears_a_leftover_token(monkeypatch: pytest.MonkeyPatch) -> None:
    # --auth off must be deterministic: a KONFAI_API_TOKEN inherited from the environment used to
    # keep auth on.
    monkeypatch.setenv("KONFAI_API_TOKEN", "leftover")

    _configure_server_auth_env("off", None, "KONFAI_API_TOKEN")

    assert "KONFAI_API_TOKEN" not in os.environ


def test_bearer_without_any_token_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KONFAI_API_TOKEN", raising=False)
    monkeypatch.delenv("MY_TOKEN", raising=False)

    with pytest.raises(SystemExit):
        _configure_server_auth_env("bearer", None, "MY_TOKEN")
