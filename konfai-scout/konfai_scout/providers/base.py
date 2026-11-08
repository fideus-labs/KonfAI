# SPDX-License-Identifier: Apache-2.0
"""The provider contract: a vision-language model that takes text and images, answers with text and tool calls, and
keeps the conversation. One implementation per API; a local VLM joins through an OpenAI-compatible server."""

from __future__ import annotations

import base64
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

#: The MCP server Scout's tools are served under to a CLI that runs the tool loop itself (Claude Code, Codex).
SERVER = "scout"


class ProviderError(RuntimeError):
    """A call the provider could not complete. ``retryable`` says whether waiting and calling again may help."""

    def __init__(self, message: str, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class ImageContent:
    png: bytes
    label: str = ""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    call_id: str
    text: str
    images: list[ImageContent] = field(default_factory=list)
    is_error: bool = False


Executor = Callable[[ToolCall], ToolResult]


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


@dataclass
class Turn:
    """What the model answered: its text, the tools it calls, why it stopped, what the request cost."""

    text: str
    tool_calls: list[ToolCall]
    stop: str  # end_turn, tool_use, max_tokens, refusal, other
    usage: Usage
    model: str | None = None


class Provider(Protocol):
    name: str
    model: str
    #: Whether images sent to it leave this machine.
    remote: bool

    def start(self, system: str, tools: list[ToolSpec]) -> None: ...

    def user(self, text: str, images: list[ImageContent] | None = None) -> Turn: ...

    def tool_results(self, results: list[ToolResult], tools: list[ToolSpec] | None = None) -> Turn: ...

    def transcript(self) -> list[dict[str, Any]]: ...

    def sdk_version(self) -> str: ...


def is_loopback(base_url: str | None) -> bool:
    """A server on this machine: images handed to it stay here."""
    if not base_url:
        return False
    from urllib.parse import urlparse

    host = (urlparse(base_url).hostname or "").lower()
    return host in ("localhost", "127.0.0.1", "::1", "0.0.0.0") or host.endswith(".localhost")


def b64(png: bytes) -> str:
    return base64.standard_b64encode(png).decode("ascii")


def without_image_data(content: Any) -> Any:
    """A content list with image bytes replaced by their size: the evidence files hold the images."""
    if not isinstance(content, list):
        return content
    out = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "image":
            data = block.get("data") or (block.get("source") or {}).get("data") or ""
            out.append({"type": "image", "bytes": len(base64.b64decode(data)) if data else None})
        else:
            out.append(block)
    return out


class ToolDriver:
    """The half shared by the providers whose CLI runs the tool loop itself: every call lands in the executor the
    inspector binds, numbered in order (a CLI may call tools in parallel; the toolbox numbers its records)."""

    drives_tools = True
    #: Whether the CLI reports its usage message by message (through ``on_usage``), so ``max_usd`` can stop it
    #: mid-loop; without it the cost is known only once the whole loop has run.
    running_cost = False

    def __init__(self) -> None:
        self.execute: Executor | None = None
        self.on_usage: Callable[[Usage], None] | None = None
        self._calls = 0
        self._lock = threading.Lock()

    def _execute(self, name: str, args: dict[str, Any]) -> ToolResult:
        if self.execute is None:
            raise ProviderError("no executor bound: the inspector binds `execute` before the first turn")
        with self._lock:
            self._calls += 1
            return self.execute(ToolCall(id=f"call_{self._calls}", name=name, arguments=dict(args or {})))
