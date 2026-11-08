# SPDX-License-Identifier: Apache-2.0
"""Claude through the Anthropic API: images as base64 blocks, tools as input schemas, the manual tool loop."""

from __future__ import annotations

import os
from typing import Any

from .base import (
    ImageContent,
    ProviderError,
    ToolCall,
    ToolResult,
    ToolSpec,
    Turn,
    Usage,
    b64,
    is_loopback,
    without_image_data,
)


def _image_block(image: ImageContent) -> dict[str, Any]:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": b64(image.png),
        },
    }


class AnthropicProvider:
    name = "anthropic"

    def __init__(
        self,
        model: str,
        *,
        effort: str | None = None,
        max_tokens: int = 16000,
        api_key_env: str | None = None,
        base_url: str | None = None,
        local: bool = False,
        fallbacks: bool = False,
        client: Any = None,
    ) -> None:
        self.model = model
        # the official API however the flags read; a server named by base_url is local on a loopback address or
        # when the caller says so
        self.remote = not (base_url and (local or is_loopback(base_url)))
        self.effort = effort
        self.max_tokens = max_tokens
        self.fallbacks = fallbacks
        self.messages: list[dict[str, Any]] = []
        self.system = ""
        self.tools: list[dict[str, Any]] = []
        if client is not None:
            self.client = client
        else:
            try:
                import anthropic
            except ImportError as error:
                raise ProviderError(
                    "the anthropic provider needs the SDK: pip install 'konfai-scout[anthropic]'"
                ) from error
            key = os.environ.get(api_key_env) if api_key_env else None
            self.client = anthropic.Anthropic(
                **({"api_key": key} if key else {}), **({"base_url": base_url} if base_url else {})
            )

    def sdk_version(self) -> str:
        try:
            import anthropic

            return str(anthropic.__version__)
        except ImportError:  # pragma: no cover
            return "unknown"

    def start(self, system: str, tools: list[ToolSpec]) -> None:
        self.system = system
        self.tools = [
            {"name": t.name, "description": t.description, "input_schema": t.parameters, "strict": True} for t in tools
        ]
        self.messages = []

    def user(self, text: str, images: list[ImageContent] | None = None) -> Turn:
        content: list[dict[str, Any]] = [{"type": "text", "text": text}]
        for image in images or []:
            if image.label:
                content.append({"type": "text", "text": image.label})
            content.append(_image_block(image))
        self.messages.append({"role": "user", "content": content})
        return self._call()

    def tool_results(self, results: list[ToolResult], tools: list[ToolSpec] | None = None) -> Turn:
        blocks = []
        for result in results:
            content: list[dict[str, Any]] = [{"type": "text", "text": result.text}]
            for image in result.images:
                if image.label:
                    content.append({"type": "text", "text": image.label})
                content.append(_image_block(image))
            blocks.append(
                {
                    "type": "tool_result",
                    "tool_use_id": result.call_id,
                    "content": content,
                    **({"is_error": True} if result.is_error else {}),
                }
            )
        self.messages.append({"role": "user", "content": blocks})
        if tools is not None:
            self.tools = [
                {"name": t.name, "description": t.description, "input_schema": t.parameters, "strict": True}
                for t in tools
            ]
        return self._call()

    def _call(self) -> Turn:
        import anthropic

        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": [{"type": "text", "text": self.system, "cache_control": {"type": "ephemeral"}}],
            "tools": self.tools,
            "messages": self.messages,
        }
        if self.effort:
            request["output_config"] = {"effort": self.effort}
        try:
            if self.fallbacks:
                response = self.client.beta.messages.create(
                    betas=["server-side-fallback-2026-07-01"], fallbacks="default", **request
                )
            else:
                response = self.client.messages.create(**request)
        except anthropic.RateLimitError as error:
            raise ProviderError(f"rate limited: {error}", retryable=True) from error
        except anthropic.APIStatusError as error:
            retryable = error.status_code >= 500 or getattr(error, "type", "") == "overloaded_error"
            raise ProviderError(
                f"API error {error.status_code}: {getattr(error, 'message', error)}", retryable
            ) from error
        except anthropic.APIConnectionError as error:  # APITimeoutError is one of these
            raise ProviderError(f"connection error: {error}", retryable=True) from error
        data = response.to_dict() if hasattr(response, "to_dict") else dict(response)
        content = data.get("content") or []
        self.messages.append({"role": "assistant", "content": content})
        text = "\n".join(block.get("text", "") for block in content if block.get("type") == "text").strip()
        calls = [
            ToolCall(id=str(block["id"]), name=str(block["name"]), arguments=dict(block.get("input") or {}))
            for block in content
            if block.get("type") == "tool_use"
        ]
        usage = data.get("usage") or {}
        stop = str(data.get("stop_reason") or "other")
        if stop == "refusal":
            details = data.get("stop_details") or {}
            text = (text + f"\n[refusal: {details.get('category')}: {details.get('explanation')}]").strip()
        return Turn(
            text=text,
            tool_calls=calls,
            stop={"end_turn": "end_turn", "tool_use": "tool_use", "max_tokens": "max_tokens", "refusal": "refusal"}.get(
                stop, "other"
            ),
            usage=Usage(
                input_tokens=int(usage.get("input_tokens") or 0),
                output_tokens=int(usage.get("output_tokens") or 0),
                cache_read_tokens=int(usage.get("cache_read_input_tokens") or 0),
                cache_write_tokens=int(usage.get("cache_creation_input_tokens") or 0),
            ),
            model=str(data.get("model") or self.model),
        )

    def transcript(self) -> list[dict[str, Any]]:
        """The conversation with image bytes replaced by their size, in the messages and in the tool results."""
        out = []
        for message in self.messages:
            content = message.get("content")
            if isinstance(content, list):
                content = [
                    {**block, "content": without_image_data(block["content"])}
                    if isinstance(block, dict) and block.get("type") == "tool_result"
                    else block
                    for block in without_image_data(content)
                ]
            out.append({**message, "content": content})
        return out
