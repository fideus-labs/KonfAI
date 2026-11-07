# SPDX-License-Identifier: Apache-2.0
"""GPT through the OpenAI API, and any OpenAI-compatible server (vLLM, Ollama, LM Studio: a local VLM), through
chat completions with function tools. A tool result is text; its images follow in a user message, since the chat
completions format carries no image in a tool message."""

from __future__ import annotations

import base64
import json
import os
from typing import Any

from .base import ImageContent, ProviderError, ToolCall, ToolResult, ToolSpec, Turn, Usage, is_loopback


def _image_part(image: ImageContent, detail: str) -> dict[str, Any]:
    data = base64.standard_b64encode(image.png).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}", "detail": detail}}


class OpenAIProvider:
    name = "openai"

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key_env: str | None = None,
        local: bool = False,
        max_tokens: int = 16000,
        detail: str = "high",
        client: Any = None,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.remote = not (local or is_loopback(base_url))
        self.max_tokens = max_tokens
        self.detail = detail
        self.messages: list[dict[str, Any]] = []
        self.tools: list[dict[str, Any]] = []
        if client is not None:
            self.client = client
        else:
            try:
                import openai
            except ImportError as error:
                raise ProviderError("the openai provider needs the SDK: pip install 'konfai-scout[openai]'") from error
            key = os.environ.get(api_key_env) if api_key_env else None
            if base_url and not key and not os.environ.get("OPENAI_API_KEY"):
                key = "local"  # a local server checks no key, the SDK insists on one
            self.client = openai.OpenAI(
                **({"api_key": key} if key else {}), **({"base_url": base_url} if base_url else {})
            )

    def sdk_version(self) -> str:
        try:
            import openai

            return str(openai.__version__)
        except ImportError:  # pragma: no cover
            return "unknown"

    def start(self, system: str, tools: list[ToolSpec]) -> None:
        self.messages = [{"role": "system", "content": system}]
        self.tools = self._tools(tools)

    @staticmethod
    def _tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
        return [
            {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
            for t in tools
        ]

    def user(self, text: str, images: list[ImageContent] | None = None) -> Turn:
        content: list[dict[str, Any]] = [{"type": "text", "text": text}]
        for image in images or []:
            if image.label:
                content.append({"type": "text", "text": image.label})
            content.append(_image_part(image, self.detail))
        self.messages.append({"role": "user", "content": content})
        return self._call()

    def tool_results(self, results: list[ToolResult], tools: list[ToolSpec] | None = None) -> Turn:
        images: list[dict[str, Any]] = []
        for result in results:
            text = result.text if not result.is_error else f"ERROR: {result.text}"
            if result.images:
                text += f"\n({len(result.images)} image(s) of this result follow in the next message)"
            self.messages.append({"role": "tool", "tool_call_id": result.call_id, "content": text})
            for image in result.images:
                images.append({"type": "text", "text": f"Image for tool call {result.call_id}: {image.label}"})
                images.append(_image_part(image, self.detail))
        if images:
            self.messages.append({"role": "user", "content": images})
        if tools is not None:
            self.tools = self._tools(tools)
        return self._call()

    def _call(self) -> Turn:
        import openai

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=self.tools or None,
                max_completion_tokens=self.max_tokens,
            )
        except openai.RateLimitError as error:
            raise ProviderError(f"rate limited: {error}", retryable=True) from error
        except openai.APIStatusError as error:
            raise ProviderError(
                f"API error {error.status_code}: {error}", retryable=error.status_code >= 500
            ) from error
        except openai.APIConnectionError as error:
            raise ProviderError(f"connection error: {error}", retryable=True) from error
        data = response.to_dict() if hasattr(response, "to_dict") else dict(response)
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        self.messages.append(
            {k: v for k, v in message.items() if k in ("role", "content", "tool_calls") and v is not None}
            | {"role": "assistant"}
        )
        calls = []
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            try:
                arguments = json.loads(function.get("arguments") or "{}")
            except json.JSONDecodeError:
                arguments = {"_invalid_json": function.get("arguments")}
            calls.append(ToolCall(id=str(call.get("id")), name=str(function.get("name")), arguments=arguments))
        usage = data.get("usage") or {}
        cached = int(((usage.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0)
        finish = str(choice.get("finish_reason") or "other")
        stop = {"stop": "end_turn", "tool_calls": "tool_use", "length": "max_tokens", "content_filter": "refusal"}.get(
            finish, "other"
        )
        if calls and stop == "end_turn":
            stop = "tool_use"
        return Turn(
            text=str(message.get("content") or "").strip(),
            tool_calls=calls,
            stop=stop,
            usage=Usage(
                input_tokens=int(usage.get("prompt_tokens") or 0) - cached,
                output_tokens=int(usage.get("completion_tokens") or 0),
                cache_read_tokens=cached,
            ),
            model=str(data.get("model") or self.model),
            raw={"id": data.get("id"), "finish_reason": finish},
        )

    def transcript(self) -> list[dict[str, Any]]:
        out = []
        for message in self.messages:
            content = message.get("content")
            if isinstance(content, list):
                content = [
                    {"type": "image", "bytes": len(base64.b64decode(part["image_url"]["url"].partition(",")[2]))}
                    if part.get("type") == "image_url"
                    else part
                    for part in content
                ]
            out.append({**message, "content": content})
        return out
