# SPDX-License-Identifier: Apache-2.0
"""A cross-platform PTY terminal (POSIX ``pty`` / Windows ConPTY) bridged to the browser over a
WebSocket. Trusted-local only: arbitrary host execution by design, gated by KONFAI_STUDIO_TERMINAL."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
from contextlib import suppress
from urllib.parse import urlparse

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from .paths import _workspace_root

router = APIRouter()


class _PtySession:
    """A login shell in a pseudo-terminal, cross-platform: POSIX ``pty`` or Windows ConPTY (pywinpty)."""

    def __init__(self, cwd: str) -> None:
        env = {**os.environ, "TERM": "xterm-256color"}
        self._win = None
        self._master = -1
        if os.name == "nt":
            from winpty import PtyProcess  # pywinpty, Windows-only

            shell = os.environ.get("COMSPEC") or "powershell.exe"
            self._win = PtyProcess.spawn(shell, cwd=cwd, env=env, dimensions=(24, 80))
        else:
            import pty

            self._master, slave = pty.openpty()
            shell = os.environ.get("SHELL") or "/bin/bash"
            self._proc = subprocess.Popen(
                [shell, "-i"],
                stdin=slave,
                stdout=slave,
                stderr=slave,
                preexec_fn=os.setsid,  # own process group, so disconnect reaps the whole tree
                cwd=cwd,
                env=env,
            )
            os.close(slave)

    def read(self) -> bytes:
        """Block for the next chunk of shell output (b'' at EOF)."""
        if self._win is not None:
            try:
                return self._win.read(65536).encode("utf-8", "replace")
            except EOFError:
                return b""
        try:
            return os.read(self._master, 65536)
        except OSError:
            return b""

    def write(self, text: str) -> None:
        if self._win is not None:
            self._win.write(text)
        else:
            os.write(self._master, text.encode())

    def resize(self, rows: int, cols: int) -> None:
        if self._win is not None:
            with suppress(Exception):
                self._win.setwinsize(rows, cols)
            return
        import fcntl
        import struct
        import termios

        with suppress(OSError):
            fcntl.ioctl(self._master, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))

    def close(self) -> None:
        if self._win is not None:
            with suppress(Exception):
                self._win.terminate(force=True)
            return
        with suppress(ProcessLookupError, PermissionError):
            os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
        with suppress(OSError):
            os.close(self._master)


@router.websocket("/api/terminal")
async def terminal(ws: WebSocket) -> None:
    """A real login shell rooted at the workspace, bridged over the socket. Trusted-local only: this is
    arbitrary host execution by design (like konfai-mcp), so a remote deployment must gate it — set
    KONFAI_STUDIO_TERMINAL=0 to disable."""
    # CSWSH guard: WebSockets are exempt from the same-origin policy, so a same-site sibling page could
    # open this shell via the auto-attached cookie. A browser always sends Origin on the handshake —
    # reject a cross-origin one. Non-browser clients (no Origin, e.g. a bearer-token CLI) pass.
    origin = ws.headers.get("origin")
    if origin is not None and urlparse(origin).netloc != ws.headers.get("host"):
        await ws.close(code=1008)
        return
    await ws.accept()
    if os.environ.get("KONFAI_STUDIO_TERMINAL", "1") == "0":
        await ws.send_text("\r\nTerminal disabled (KONFAI_STUDIO_TERMINAL=0).\r\n")
        await ws.close()
        return
    session = _PtySession(cwd=str(_workspace_root()))
    loop = asyncio.get_running_loop()

    async def pump() -> None:
        try:
            while True:
                data = await loop.run_in_executor(None, session.read)
                if not data:
                    break
                await ws.send_bytes(data)
        except (OSError, RuntimeError, WebSocketDisconnect):
            pass
        finally:
            with suppress(Exception):
                await ws.close()

    reader = asyncio.create_task(pump())
    try:
        while True:
            evt = json.loads(await ws.receive_text())
            if evt.get("type") == "input":
                session.write(str(evt.get("data", "")))
            elif evt.get("type") == "resize":
                session.resize(int(evt.get("rows", 24)), int(evt.get("cols", 80)))
    except (WebSocketDisconnect, ValueError, KeyError):
        pass
    finally:
        reader.cancel()
        session.close()
