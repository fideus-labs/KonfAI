# SPDX-License-Identifier: Apache-2.0
"""The guards on ``POST /api/quit``.

Shutting the server down is reachable from any page the user has open — to the server, every request
a browser makes to localhost looks local — so it only fires for a caller that is both on this machine
and able to set a header, which a cross-origin form cannot.
"""

from __future__ import annotations

import signal
import time
from collections.abc import Iterator

import pytest

pytest.importorskip("fastapi")
from konfai_studio import server as bff
from starlette.testclient import TestClient

HEADER = {"X-KonfAI-Studio": "quit"}


@pytest.fixture
def killed() -> Iterator[list[int]]:
    """Capture the shutdown signal instead of killing the test runner."""
    sent: list[int] = []
    original = bff.os.kill
    bff.os.kill = lambda pid, sig: sent.append(sig)  # type: ignore[assignment]
    try:
        yield sent
    finally:
        bff.os.kill = original  # type: ignore[assignment]


def _wait_for_signal(sent: list[int], timeout: float = 3.0) -> bool:
    """The endpoint answers first and signals shortly after, so give that task a moment."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if sent:
            return True
        time.sleep(0.05)
    return False


def test_the_studio_ui_stops_the_server(killed: list[int]) -> None:
    with TestClient(bff.app, client=("127.0.0.1", 54321)) as client:
        response = client.post("/api/quit", json={}, headers=HEADER)
        # Inside the block: the signal is sent by a task on the client's own event loop, which
        # stops with it.
        signalled = _wait_for_signal(killed)

    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert signalled
    assert killed == [signal.SIGTERM]


def test_a_form_post_without_the_header_cannot_stop_the_server(killed: list[int]) -> None:
    with TestClient(bff.app, client=("127.0.0.1", 54321)) as client:
        response = client.post("/api/quit", data={"any": "field"})

    assert response.status_code == 403
    assert not killed


def test_a_client_off_this_machine_cannot_stop_the_server(killed: list[int]) -> None:
    with TestClient(bff.app, client=("203.0.113.7", 54321)) as client:
        response = client.post("/api/quit", json={}, headers=HEADER)

    assert response.status_code == 403
    assert not killed


@pytest.mark.parametrize(
    "header", ["X-Forwarded-For", "X-Forwarded-Host", "X-Forwarded-Proto", "X-Real-IP", "Forwarded"]
)
def test_a_request_through_an_untrusted_proxy_cannot_stop_the_server(
    killed: list[int], monkeypatch: pytest.MonkeyPatch, header: str
) -> None:
    """Behind the same-host reverse proxy REMOTE.md documents, every peer IS 127.0.0.1.

    Without ``--proxy-headers`` uvicorn does not rewrite it, so the loopback check says local about
    a request that came from anywhere. A forwarding header is the tell.
    """
    monkeypatch.delenv("KONFAI_STUDIO_PROXY_HEADERS", raising=False)
    with TestClient(bff.app, client=("127.0.0.1", 54321)) as client:
        response = client.post("/api/quit", json={}, headers={**HEADER, header: "203.0.113.7"})

    assert response.status_code == 403
    assert "proxy" in response.json()["detail"]
    assert not killed


def test_a_trusted_proxy_leaves_the_loopback_check_meaningful(
    killed: list[int], monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the flag, uvicorn rewrites the peer from the header before the endpoint sees it, so the
    loopback check is a statement about the real client and a forwarding header is not a reason to
    refuse."""
    monkeypatch.setenv("KONFAI_STUDIO_PROXY_HEADERS", "1")
    with TestClient(bff.app, client=("127.0.0.1", 54321)) as client:
        response = client.post("/api/quit", json={}, headers={**HEADER, "X-Forwarded-For": "127.0.0.1"})
        signalled = _wait_for_signal(killed)

    assert response.status_code == 200
    assert signalled
