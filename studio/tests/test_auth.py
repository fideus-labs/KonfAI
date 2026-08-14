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

"""The access gate a remote deployment stands on.

Studio drives konfai-mcp: arbitrary host reads and arbitrary compute, plus a real login shell. On
loopback that is the operator's own machine. Exposed on a network, one shared token is the whole
defence, and `studio/docs/REMOTE.md` states what it covers. These tests are that statement, executed.

The terminal cases deliberately exercise only the *refusals*: a handshake that gets through spawns a
real shell, which is not something a test suite should leave behind.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from konfai_studio.auth import _authorised, _session_cookie
from konfai_studio.server import app
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

TOKEN = "a-strong-shared-secret"
GUARDED = "/api/sessions"  # a plain data endpoint: no token, no answer


@pytest.fixture(autouse=True)
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Never let an auth test read or write the operator's real workspaces."""
    monkeypatch.setenv("KONFAI_MCP_WORKSPACES_ROOT", str(tmp_path))


def test_without_a_token_the_deployment_is_open(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default is trusted-local: unset means no gate at all, exactly as before it existed."""
    monkeypatch.delenv("KONFAI_STUDIO_TOKEN", raising=False)
    with TestClient(app) as client:
        assert client.get(GUARDED).status_code == 200
        assert client.get("/api/auth").json() == {"required": False, "authenticated": True}


def test_a_token_shuts_every_data_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KONFAI_STUDIO_TOKEN", TOKEN)
    with TestClient(app) as client:
        assert client.get(GUARDED).status_code == 401
        assert client.post("/api/sessions", json={"name": "x"}).status_code == 401


def test_the_lock_screen_can_still_load_itself(monkeypatch: pytest.MonkeyPatch) -> None:
    """A gate that also blocks the page asking for the token locks the operator out of their own server."""
    monkeypatch.setenv("KONFAI_STUDIO_TOKEN", TOKEN)
    with TestClient(app) as client:
        assert client.get("/api/auth").json() == {"required": True, "authenticated": False}
        assert client.get("/api/health").status_code == 200


def test_the_session_cookie_is_secure_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """A remote deployment runs behind TLS, so the cookie carrying the session must not travel in clear.
    The consequence is visible here: over plain http the browser never sends it back."""
    monkeypatch.setenv("KONFAI_STUDIO_TOKEN", TOKEN)
    monkeypatch.delenv("KONFAI_STUDIO_INSECURE_COOKIE", raising=False)
    with TestClient(app) as client:
        login = client.post("/api/login", json={"token": TOKEN})

        assert "Secure" in login.headers["set-cookie"]
        assert client.get(GUARDED).status_code == 401


def test_the_token_buys_a_cookie_and_the_cookie_opens_the_door(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KONFAI_STUDIO_TOKEN", TOKEN)
    monkeypatch.setenv("KONFAI_STUDIO_INSECURE_COOKIE", "1")  # what the docs prescribe for local http
    with TestClient(app) as client:
        assert client.post("/api/login", json={"token": "not-it"}).status_code == 401
        assert client.get(GUARDED).status_code == 401

        login = client.post("/api/login", json={"token": TOKEN})
        assert login.status_code == 200
        assert login.cookies.get("ks_session") == _session_cookie(TOKEN)  # the raw token never reaches the browser
        assert client.get(GUARDED).status_code == 200  # the client kept the cookie

        client.post("/api/logout")
        client.cookies.clear()
        assert client.get(GUARDED).status_code == 401


def test_a_bearer_token_opens_the_door_for_a_non_browser_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KONFAI_STUDIO_TOKEN", TOKEN)
    with TestClient(app) as client:
        assert client.get(GUARDED, headers={"Authorization": f"Bearer {TOKEN}"}).status_code == 200
        assert client.get(GUARDED, headers={"Authorization": "Bearer wrong"}).status_code == 401


def test_a_dot_segment_cannot_walk_out_of_the_public_asset_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    """The gate matches the raw path while the router matches the resolved one, and only a browser
    normalises before sending: a raw client can hand the two different paths. Asked of the gate itself,
    because an HTTP test client rewrites the dot segment away before the server ever sees it."""
    monkeypatch.setenv("KONFAI_STUDIO_TOKEN", TOKEN)
    scope = {"type": "http", "headers": []}

    assert _authorised({**scope, "path": "/assets/index-abc123.js"})
    assert not _authorised({**scope, "path": "/assets/../api/sessions"})


def test_the_terminal_socket_is_behind_the_same_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """WebSockets do not pass through HTTP middleware unless the middleware is a pure ASGI one. This is
    a login shell: were it to slip the gate, the token would guard the data and not the machine."""
    monkeypatch.setenv("KONFAI_STUDIO_TOKEN", TOKEN)
    with TestClient(app) as client, pytest.raises(WebSocketDisconnect) as refused:
        with client.websocket_connect("/api/terminal") as ws:
            ws.receive_text()
    assert refused.value.code == 1008


def test_a_cross_origin_page_cannot_open_the_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    """WebSockets are exempt from the same-origin policy, so a sibling page on another origin could open
    this shell on the auto-attached cookie. The handshake's Origin is what stops it."""
    monkeypatch.delenv("KONFAI_STUDIO_TOKEN", raising=False)
    with TestClient(app) as client, pytest.raises(WebSocketDisconnect) as refused:
        with client.websocket_connect("/api/terminal", headers={"origin": "http://evil.example"}) as ws:
            ws.receive_text()
    assert refused.value.code == 1008
