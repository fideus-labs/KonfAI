# SPDX-License-Identifier: Apache-2.0
"""The provider contract: a vision-language model that takes text and images, answers with text and tool calls, and
keeps the conversation. One implementation per API; a local VLM joins through an OpenAI-compatible server."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


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
    raw: dict[str, Any] = field(default_factory=dict)


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
