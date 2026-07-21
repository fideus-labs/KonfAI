# SPDX-License-Identifier: Apache-2.0
"""Access gate for remote deployments — the ``_AuthGate`` ASGI middleware, the shared-token cookie
scheme, and the login/logout endpoints.

Studio drives konfai-mcp, which reads arbitrary host paths and runs jobs — arbitrary compute by
design. On loopback that is the operator's own machine; exposed on a network it is not. A single
shared token (KONFAI_STUDIO_TOKEN) turns on authentication: unset, everything is open exactly as
before (trusted-local); set, every request must carry a valid session cookie or bearer token. TLS
is the reverse proxy's job (see docs/REMOTE.md).
"""

from __future__ import annotations

import hashlib
import hmac
import os
from contextlib import suppress
from http.cookies import SimpleCookie
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

_COOKIE_NAME = "ks_session"
# The app shell + login surface are reachable without a session; everything else needs one.
_PUBLIC_PATHS = frozenset(
    {"/", "/index.html", "/api/auth", "/api/login", "/api/health", "/konfai-logo.png", "/favicon.ico"}
)

router = APIRouter()


def _studio_token() -> str:
    """The shared access token guarding a remote deployment ('' = auth disabled, trusted-local)."""
    return os.environ.get("KONFAI_STUDIO_TOKEN", "").strip()


def _session_cookie(token: str) -> str:
    """A stable, non-reversible session value derived from the token — what the auth cookie carries, so
    the raw token never lives in the browser and a server restart keeps the user signed in."""
    return hmac.new(token.encode(), b"konfai-studio-session", hashlib.sha256).hexdigest()


def _scope_header(scope: dict[str, Any], name: bytes) -> str | None:
    for key, value in scope.get("headers", []):
        if key == name:
            return value.decode("latin-1")
    return None


def _authorised(scope: dict[str, Any]) -> bool:
    """Whether an ASGI request may proceed: always when auth is off, for the public shell paths, or when
    it presents the session cookie / bearer token. Comparisons are constant-time."""
    token = _studio_token()
    if not token:
        return True
    path = scope.get("path", "")
    # Exact public paths, or a static asset — but never a dot-segment path, which a raw client could use to
    # make the gate ("/assets/…", allowed) and the router disagree on the effective route. Fail closed.
    if path in _PUBLIC_PATHS or (path.startswith("/assets/") and "/.." not in path):
        return True
    expected = _session_cookie(token)
    raw = _scope_header(scope, b"cookie")
    if raw:
        jar: SimpleCookie = SimpleCookie()
        with suppress(Exception):
            jar.load(raw)
        morsel = jar.get(_COOKIE_NAME)
        # Compare as bytes: hmac.compare_digest raises TypeError on a non-ASCII str, and both the cookie
        # value and the bearer token are attacker-controlled — bytes yield a constant-time False instead.
        if morsel and hmac.compare_digest(morsel.value.encode(), expected.encode()):
            return True
    auth = _scope_header(scope, b"authorization") or ""
    if auth.lower().startswith("bearer ") and hmac.compare_digest(auth[7:].strip().encode(), token.encode()):
        return True
    return False


class _AuthGate:
    """Blanket access gate for remote deployments. A pure ASGI middleware (not ``BaseHTTPMiddleware``) so
    it never wraps the SSE/stream responses — it inspects the request and either passes it through
    untouched or short-circuits with a 401 / WebSocket close. No-op when the token is unset."""

    def __init__(self, app: Any) -> None:
        self._app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] not in {"http", "websocket"} or _authorised(scope):
            await self._app(scope, receive, send)
            return
        if scope["type"] == "websocket":
            await receive()  # consume the connect so the close handshake is well-formed
            await send({"type": "websocket.close", "code": 1008})
            return
        await JSONResponse({"detail": "authentication required"}, status_code=401)(scope, receive, send)


class LoginRequest(BaseModel):
    token: str


def _cookie_secure() -> bool:
    """The session cookie is Secure by default — a remote deployment must run behind TLS. Opt out only
    for local http testing of the auth flow with KONFAI_STUDIO_INSECURE_COOKIE=1."""
    return os.environ.get("KONFAI_STUDIO_INSECURE_COOKIE") != "1"


@router.get("/api/auth")
async def auth_state(request: Request) -> dict[str, bool]:
    """Whether this deployment requires a token, and whether the browser already holds a valid session —
    the front shows a lock screen when required and not yet authenticated."""
    token = _studio_token()
    if not token:
        return {"required": False, "authenticated": True}
    cookie = request.cookies.get(_COOKIE_NAME)
    ok = bool(cookie and hmac.compare_digest(cookie.encode(), _session_cookie(token).encode()))
    return {"required": True, "authenticated": ok}


@router.post("/api/login")
async def login(req: LoginRequest) -> Response:
    """Exchange the shared access token for an httpOnly session cookie. Constant-time compare; a wrong
    token is a flat 401 (the token's entropy, not rate-limiting, is the defence)."""
    token = _studio_token()
    if not token:
        return JSONResponse({"ok": True})  # auth disabled — nothing to unlock
    if not hmac.compare_digest(req.token.strip().encode(), token.encode()):
        raise HTTPException(401, "invalid access token")
    resp = JSONResponse({"ok": True})
    resp.set_cookie(
        _COOKIE_NAME,
        _session_cookie(token),
        max_age=60 * 60 * 24 * 30,
        httponly=True,
        samesite="lax",
        secure=_cookie_secure(),
        path="/",
    )
    return resp


@router.post("/api/logout")
async def logout() -> Response:
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(_COOKIE_NAME, path="/")
    return resp
