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

"""/repo_apps endpoints must not resolve app ids outside the allowlist."""

import importlib.util

import pytest

pytestmark = pytest.mark.skipif(importlib.util.find_spec("fastapi") is None, reason="fastapi not installed")

import konfai_apps.app_server as app_server  # noqa: E402
from fastapi import HTTPException  # noqa: E402


def test_require_configured_app_rejects_unlisted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_server, "_APPS", ["VBoussot/ImpactSynth:CBCT"])
    with pytest.raises(HTTPException) as exc:
        app_server._require_configured_app("attacker/evil-repo:x")
    assert exc.value.status_code == 404


def test_get_app_info_rejects_unlisted_before_resolving(monkeypatch: pytest.MonkeyPatch) -> None:
    # The endpoint must reject an unlisted app id BEFORE calling the repository resolver, otherwise a
    # token holder could trigger arbitrary HuggingFace/local/remote fetches (SSRF / exfiltration).
    monkeypatch.setattr(app_server, "_APPS", ["good/app"])

    def _must_not_resolve(*args, **kwargs):
        raise AssertionError("get_app_repository_info must not be called for an unlisted app")

    monkeypatch.setattr(app_server, "get_app_repository_info", _must_not_resolve)

    with pytest.raises(HTTPException) as exc:
        app_server.get_app_info("attacker/evil-repo:x")
    assert exc.value.status_code == 404
