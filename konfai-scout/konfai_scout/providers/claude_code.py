# SPDX-License-Identifier: Apache-2.0
"""Claude through Claude Code: the Claude Agent SDK runs the tool loop under the user's Claude Code login, no API key.

The SDK drives the loop itself, so the roles are swapped: Scout's tools are served to it as an in-process MCP server
and every call lands in the ``execute`` callback the inspector binds, which keeps the record (calls, views,
measurements, budget) identical to the API providers'. Images reach the model as blocks of the first message and as
MCP image content of tool results. The CLI keeps the session, images included, under ``~/.claude/projects``."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator
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

TRANSIENT = {408, 409, 429, 500, 502, 503, 504, 529}


def _message_blocks(text: str, images: list[ImageContent] | None) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for image in images or []:
        if image.label:
            content.append({"type": "text", "text": image.label})
        content.append(
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64(image.png)}}
        )
    return content


def _mcp_result(result: ToolResult) -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"type": "text", "text": result.text}]
    for image in result.images:
        if image.label:
            content.append({"type": "text", "text": image.label})
        content.append({"type": "image", "data": b64(image.png), "mimeType": "image/png"})
    return {"content": content, **({"is_error": True} if result.is_error else {})}


class ClaudeCodeProvider(ToolDriver):
    """The SDK runs the tool loop: the inspector binds ``execute`` and reads the record from its toolbox."""

    name = "claude-code"
    remote = True
    running_cost = True  # every assistant message carries its request's usage

    def __init__(self, model: str, *, effort: str | None = None, max_turns: int = 60) -> None:
        super().__init__()
        self.model = model
        self.effort = effort
        self.max_turns = max_turns
        self.system = ""
        self.tools: list[ToolSpec] = []
        self.sdk_tools: dict[str, Any] = {}
        self.messages: list[dict[str, Any]] = []
        self.session_id: str | None = None

    def sdk_version(self) -> str:
        try:
            from claude_agent_sdk import __version__

            return str(__version__)
        except ImportError:  # pragma: no cover
            return "unknown"

    def start(self, system: str, tools: list[ToolSpec]) -> None:
        self.system = (
            system + "\n\n## Tools\nThe tools are served by the MCP server `scout`: `mcp__scout__view` is `view`, "
            "`mcp__scout__note` is `note`, and so on."
        )
        self.tools = tools
        self.messages = []
        self.session_id = None
        self._calls = 0

    def user(self, text: str, images: list[ImageContent] | None = None) -> Turn:
        content = _message_blocks(text, images)
        self.messages.append({"role": "user", "content": without_image_data(content)})
        return _run(self._turn(content))

    def tool_results(self, results: list[ToolResult], tools: list[ToolSpec] | None = None) -> Turn:
        raise ProviderError("the claude-code provider runs the tool loop itself: results reach it through `execute`")

    def transcript(self) -> list[dict[str, Any]]:
        return list(self.messages)

    # ------------------------------------------------------------------ the SDK session

    def _server(self) -> Any:
        from claude_agent_sdk import create_sdk_mcp_server, tool

        self.sdk_tools = {}
        for spec in self.tools:

            async def handler(args: dict[str, Any], _name: str = spec.name) -> dict[str, Any]:
                return _mcp_result(await asyncio.to_thread(self._execute, _name, args))

            self.sdk_tools[spec.name] = tool(spec.name, spec.description, spec.parameters)(handler)
        return create_sdk_mcp_server(name=SERVER, version="1.0", tools=list(self.sdk_tools.values()))

    def _options(self) -> Any:
        from claude_agent_sdk import ClaudeAgentOptions

        return ClaudeAgentOptions(
            system_prompt=self.system,
            tools=[],  # no built-in tool: the model sees the case through Scout's tools only
            mcp_servers={SERVER: self._server()},
            strict_mcp_config=True,
            allowed_tools=[f"mcp__{SERVER}__{spec.name}" for spec in self.tools],
            permission_mode="dontAsk",  # anything not listed above is denied
            setting_sources=[],  # no CLAUDE.md, hooks or project settings of the directory Scout runs from
            model=self.model,
            effort=self.effort,  # type: ignore[arg-type]
            max_turns=self.max_turns,
            resume=self.session_id,
            # a view of a large noisy volume is a PNG of several hundred kB; base64 in one JSON line, it passes
            # the SDK's default 1 MB message buffer
            max_buffer_size=64 * 1024 * 1024,
        )

    async def _stream(self, content: list[dict[str, Any]], options: Any) -> AsyncIterator[Any]:
        from claude_agent_sdk import query

        async def prompt() -> AsyncIterator[dict[str, Any]]:
            yield {
                "type": "user",
                "message": {"role": "user", "content": content},
                "parent_tool_use_id": None,
                "session_id": self.session_id or "default",
            }

        async for message in query(prompt=prompt(), options=options):
            yield message

    async def _turn(self, content: list[dict[str, Any]]) -> Turn:
        from claude_agent_sdk import (
            AssistantMessage,
            ResultMessage,
            TextBlock,
            ThinkingBlock,
            ToolResultBlock,
            ToolUseBlock,
            UserMessage,
        )

        texts: list[str] = []
        model: str | None = None
        result: Any = None
        counted: set[str] = set()  # one request's content blocks arrive as several messages with one id and usage
        try:
            async for message in self._stream(content, self._options()):
                if isinstance(message, AssistantMessage):
                    if message.usage and self.on_usage and message.message_id not in counted:
                        if message.message_id:  # without an id, each message counts: a stop too early, not too late
                            counted.add(message.message_id)
                        self.on_usage(_usage(message.usage))
                    blocks: list[dict[str, Any]] = []
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            texts.append(block.text)
                            blocks.append({"type": "text", "text": block.text})
                        elif isinstance(block, ToolUseBlock):
                            blocks.append(
                                {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
                            )
                        elif isinstance(block, ThinkingBlock):
                            blocks.append({"type": "thinking", "chars": len(block.thinking)})
                    model = message.model or model
                    self.messages.append({"role": "assistant", "model": message.model, "content": blocks})
                elif isinstance(message, UserMessage) and isinstance(message.content, list):
                    results = [
                        {
                            "type": "tool_result",
                            "tool_use_id": block.tool_use_id,
                            "is_error": bool(block.is_error),
                            "content": without_image_data(block.content),
                        }
                        for block in message.content
                        if isinstance(block, ToolResultBlock)
                    ]
                    if results:
                        self.messages.append({"role": "user", "content": results})
                elif isinstance(message, ResultMessage):
                    result = message
        except Exception as error:  # the SDK raises a bare Exception once the CLI has ended on an error result
            if result is None or not result.is_error:
                raise ProviderError(f"Claude Code: {error}") from error
        if result is None:
            raise ProviderError("Claude Code ended without a result", retryable=True)
        self.session_id = result.session_id or self.session_id
        usage = result.usage or {}
        self.messages.append(
            {
                "role": "result",
                "subtype": result.subtype,
                "num_turns": result.num_turns,
                "stop_reason": result.stop_reason,
                "total_cost_usd": result.total_cost_usd,
                "usage": usage,
                "session_id": result.session_id,
                "models": sorted((result.model_usage or {}).keys()),
            }
        )
        if result.is_error and result.subtype != "error_max_turns":
            detail = result.result or result.errors or "\n".join(texts) or result.subtype  # an API error is a text
            raise ProviderError(f"Claude Code: {detail}", retryable=result.api_error_status in TRANSIENT)
        stop = str(result.stop_reason or ("other" if result.subtype != "success" else "end_turn"))
        return Turn(
            text="\n".join(texts).strip(),
            tool_calls=[],  # the calls ran already, through `execute`
            stop={"end_turn": "end_turn", "max_tokens": "max_tokens", "refusal": "refusal"}.get(stop, "other"),
            usage=_usage(usage),
            model=model or self.model,
        )


def _usage(usage: dict[str, Any]) -> Usage:
    return Usage(
        input_tokens=int(usage.get("input_tokens") or 0),
        output_tokens=int(usage.get("output_tokens") or 0),
        cache_read_tokens=int(usage.get("cache_read_input_tokens") or 0),
        cache_write_tokens=int(usage.get("cache_creation_input_tokens") or 0),
    )


def _run(coroutine: Any) -> Any:
    """Run the SDK session to completion from synchronous code, in a thread when a loop is already running."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)
    box: dict[str, Any] = {}

    def runner() -> None:
        try:
            box["value"] = asyncio.run(coroutine)
        except BaseException as error:  # re-raised in the caller's thread
            box["error"] = error

    thread = threading.Thread(target=runner, name="konfai-scout-claude-code")
    thread.start()
    thread.join()
    if "error" in box:
        raise box["error"]
    return box["value"]
