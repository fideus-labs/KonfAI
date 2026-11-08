# SPDX-License-Identifier: Apache-2.0
"""GPT through the Codex CLI: ``codex exec`` runs the tool loop under the user's ChatGPT login, no API key.

As with Claude Code, the CLI drives the loop. Scout's tools are served to it over a streamable-HTTP MCP server run in
this process, annotated read-only (what Codex's non-interactive mode calls without an approval), the coverage
images are attached to the prompt as files, a tool's image comes back as MCP image content, and every call lands
in the executor the inspector binds. One ``codex exec`` process per turn (``codex exec resume`` for the next ones);
the CLI keeps the session under ``$CODEX_HOME/sessions``. The user's own Codex configuration (its MCP servers,
plugins, trusted projects) is not loaded; only the login is."""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import os
import socket
import subprocess
import tempfile
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .base import (
    SERVER,
    ImageContent,
    ProviderError,
    ToolDriver,
    ToolResult,
    ToolSpec,
    Turn,
    Usage,
    b64,
    without_image_data,
)

#: Codex features switched off for an inspection: the model sees the case through Scout's tools only.
DISABLED_FEATURES = ("plugins", "shell_tool", "apps", "browser_use", "computer_use", "skill_search")
TRANSIENT = ("429", "rate limit", "overloaded", "temporarily", "502", "503", "504", "timeout", "timed out")


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_port(port: int, seconds: float = 15.0) -> None:
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
            return
        except OSError:
            time.sleep(0.05)
    raise ProviderError(f"the MCP server for Codex did not come up on port {port}")


def _codex_home() -> Path:
    return Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")


def _model_of(thread_id: str | None) -> str | None:
    """The model that answered, from the CLI's own record of the session (``turn_context`` of the rollout)."""
    if not thread_id:
        return None
    for path in _codex_home().glob(f"sessions/*/*/*/rollout-*-{thread_id}.jsonl"):
        try:
            with path.open(encoding="utf-8") as lines:
                for line in lines:
                    if '"turn_context"' not in line:
                        continue
                    record = json.loads(line)
                    if record.get("type") == "turn_context":
                        model = (record.get("payload") or {}).get("model")
                        if model:
                            return str(model)
        except (OSError, ValueError):
            continue
    return None


class CodexProvider(ToolDriver):
    """The CLI runs the tool loop: the inspector binds ``execute`` and reads the record from its toolbox."""

    name = "codex"
    remote = True

    def __init__(
        self, model: str = "default", *, effort: str | None = None, executable: str = "codex", timeout: float = 1800.0
    ) -> None:
        super().__init__()
        self.model = model
        self.effort = effort
        self.executable = executable
        self.timeout = timeout
        self.system = ""
        self.tools: list[ToolSpec] = []
        self.messages: list[dict[str, Any]] = []
        self.thread_id: str | None = None
        self.url: str | None = None
        self._work = tempfile.TemporaryDirectory(prefix="konfai-scout-codex-")

    def sdk_version(self) -> str:
        try:
            out = subprocess.run(
                [self.executable, "--version"], capture_output=True, text=True, timeout=30, stdin=subprocess.DEVNULL
            )
            return out.stdout.strip() or "unknown"
        except (OSError, subprocess.SubprocessError):
            return "unknown"

    def start(self, system: str, tools: list[ToolSpec]) -> None:
        self.system = system
        self.tools = tools
        self.messages = []
        self.thread_id = None
        self.url = None
        self._calls = 0

    def user(self, text: str, images: list[ImageContent] | None = None) -> Turn:
        images = list(images or [])
        paths = [self._attach(image, n) for n, image in enumerate(images)]
        prompt = text if self.thread_id else f"{self.system}\n\n---\n\n{text}"
        labels = [image.label for image in images if image.label]
        if labels:
            prompt += "\n\nThe attached images are, in order: " + ", ".join(labels) + "."
        self.messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    *({"type": "image", "bytes": len(i.png)} for i in images),
                ],
            }
        )
        return self._turn(prompt, paths)

    def tool_results(self, results: list[ToolResult], tools: list[ToolSpec] | None = None) -> Turn:
        raise ProviderError("the codex provider runs the tool loop itself: results reach it through `execute`")

    def transcript(self) -> list[dict[str, Any]]:
        return list(self.messages)

    # ------------------------------------------------------------------ the MCP server

    def _attach(self, image: ImageContent, n: int) -> Path:
        path = Path(self._work.name) / f"attached_{len(self.messages):02d}_{n:02d}.png"
        path.write_bytes(image.png)
        return path

    def _serve(self) -> str:
        """Scout's tools as an MCP server on a loopback port, started once per inspection."""
        if self.url:
            return self.url
        try:
            from fastmcp import FastMCP
        except ImportError as error:
            raise ProviderError("the codex provider needs fastmcp: pip install 'konfai-scout[codex]'") from error
        for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "fastmcp", "FastMCP"):
            logging.getLogger(name).setLevel(logging.WARNING)
        server = FastMCP(SERVER)
        for spec in self.tools:
            server.add_tool(_scout_tool_class().of(spec, self._execute))
        port = _free_port()
        threading.Thread(
            target=lambda: server.run(
                transport="http", host="127.0.0.1", port=port, show_banner=False, log_level="warning"
            ),
            daemon=True,
            name="konfai-scout-codex-mcp",
        ).start()
        _wait_port(port)
        self.url = f"http://127.0.0.1:{port}/mcp"
        return self.url

    # ------------------------------------------------------------------ the CLI

    def _command(self, images: list[Path]) -> list[str]:
        cmd = [self.executable, "exec"]
        if self.thread_id:  # a resumed session keeps its working directory and sandbox
            cmd += ["resume", self.thread_id]
        else:
            cmd += ["-C", self._work.name, "-s", "read-only"]
        cmd += [
            "--json",
            "--skip-git-repo-check",
            "--ignore-user-config",
            "-c",
            f'mcp_servers.{SERVER}.url="{self._serve()}"',
        ]
        for feature in DISABLED_FEATURES:
            cmd += ["--disable", feature]
        if self.model and self.model != "default":
            cmd += ["-m", self.model]
        if self.effort:
            cmd += ["-c", f'model_reasoning_effort="{self.effort}"']
        cmd += [f"--image={path}" for path in images]
        return cmd

    def _events(self, cmd: list[str], prompt: str) -> Iterator[dict[str, Any]]:
        """The CLI's JSONL events, the prompt on its stdin; a last ``exit`` event carries the code and stderr."""
        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
            )
        except OSError as error:
            raise ProviderError(
                f"cannot run {self.executable!r}: {error}. Install the Codex CLI and log in (`codex login`)."
            ) from error
        assert proc.stdin is not None and proc.stdout is not None and proc.stderr is not None
        stderr: list[str] = []
        threading.Thread(target=lambda: stderr.extend(proc.stderr), daemon=True).start()  # type: ignore[arg-type]
        timer = threading.Timer(self.timeout, proc.kill)
        timer.start()
        try:
            proc.stdin.write(prompt)
            proc.stdin.close()
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except ValueError:
                    yield {"type": "raw", "text": line}
            code = proc.wait()
        finally:
            timer.cancel()
        yield {"type": "exit", "code": code, "stderr": "".join(stderr)[-2000:]}

    def _turn(self, prompt: str, images: list[Path]) -> Turn:
        texts: list[str] = []
        usage: dict[str, Any] | None = None
        failed: str | None = None
        exit_info: dict[str, Any] = {}
        for event in self._events(self._command(images), prompt):
            kind = event.get("type")
            if kind == "thread.started":
                self.thread_id = event.get("thread_id") or self.thread_id
            elif kind == "item.completed":
                item = event.get("item") or {}
                itype = item.get("type")
                if itype == "agent_message":
                    text = str(item.get("text") or "")
                    texts.append(text)
                    self.messages.append({"role": "assistant", "content": [{"type": "text", "text": text}]})
                elif itype == "mcp_tool_call":
                    self.messages.append(
                        {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "tool_use",
                                    "server": item.get("server"),
                                    "name": item.get("tool"),
                                    "input": item.get("arguments"),
                                    "status": item.get("status"),
                                    "error": item.get("error"),
                                }
                            ],
                        }
                    )
                    result = item.get("result") or {}
                    self.messages.append(
                        {
                            "role": "user",
                            "content": [{"type": "tool_result", "content": without_image_data(result.get("content"))}],
                        }
                    )
                elif itype == "reasoning":
                    self.messages.append(
                        {
                            "role": "assistant",
                            "content": [{"type": "thinking", "chars": len(str(item.get("text") or ""))}],
                        }
                    )
                elif itype == "error":
                    self.messages.append({"role": "notice", "text": item.get("message")})
                elif itype:
                    self.messages.append(
                        {"role": "assistant", "content": [{k: v for k, v in item.items() if k != "id"}]}
                    )
            elif kind == "turn.completed":
                usage = dict(event.get("usage") or {})
            elif kind == "turn.failed":
                failed = str((event.get("error") or {}).get("message") or "turn failed")
            elif kind == "error" and failed is None:
                failed = str(event.get("message") or "error")
            elif kind == "exit":
                exit_info = event
        if usage is None:
            detail = failed or exit_info.get("stderr") or f"exit code {exit_info.get('code')}"
            if exit_info.get("code") in (-9, 137):
                detail = f"killed after {self.timeout:.0f} s: {detail}"
            raise ProviderError(f"Codex: {detail}", retryable=any(word in detail.lower() for word in TRANSIENT))
        model = _model_of(self.thread_id)
        if model and self.model == "default":
            self.model = model
        self.messages.append(
            {"role": "result", "thread_id": self.thread_id, "usage": usage, "model": model, "notice": failed}
        )
        cached = int(usage.get("cached_input_tokens") or 0)
        return Turn(
            text="\n".join(texts).strip(),
            tool_calls=[],  # the calls ran already, through `execute`
            stop="end_turn",
            usage=Usage(
                input_tokens=max(0, int(usage.get("input_tokens") or 0) - cached),
                output_tokens=int(usage.get("output_tokens") or 0),
                cache_read_tokens=cached,
                cache_write_tokens=int(usage.get("cache_write_input_tokens") or 0),
            ),
            model=model or self.model,
        )


@functools.cache
def _scout_tool_class() -> Any:
    """The fastmcp subclass, built on first use: importing this module never needs fastmcp."""
    from fastmcp.tools import Tool
    from fastmcp.tools import ToolResult as McpToolResult
    from mcp.types import ImageContent as McpImage
    from mcp.types import TextContent, ToolAnnotations
    from pydantic import PrivateAttr

    class ScoutTool(Tool):  # type: ignore[misc]
        """One of Scout's tools as fastmcp serves it: the JSON schema as is, read-only for Codex's approval policy."""

        _execute: Any = PrivateAttr(default=None)

        @classmethod
        def of(cls, spec: ToolSpec, execute: Any) -> Any:
            tool = cls(
                name=spec.name,
                description=spec.description,
                parameters=spec.parameters,
                annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False),
            )
            tool._execute = execute
            return tool

        async def run(self, arguments: dict[str, Any]) -> Any:
            result: ToolResult = await asyncio.to_thread(self._execute, self.name, arguments)
            content: list[Any] = [TextContent(type="text", text=result.text)]
            for image in result.images:
                if image.label:
                    content.append(TextContent(type="text", text=image.label))
                content.append(McpImage(type="image", data=b64(image.png), mimeType="image/png"))
            return McpToolResult(content=content, is_error=result.is_error)

    return ScoutTool
