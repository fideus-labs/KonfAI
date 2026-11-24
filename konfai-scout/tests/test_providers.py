# SPDX-License-Identifier: Apache-2.0
"""The two API providers against fake clients: request shapes, parsing, error classes; and the privacy gate."""

import base64
import copy
import json
from types import SimpleNamespace

import pytest

anthropic = pytest.importorskip("anthropic")
openai = pytest.importorskip("openai")

from konfai_scout.config import ScoutConfig  # noqa: E402
from konfai_scout.providers import check_privacy, is_remote, make_provider  # noqa: E402
from konfai_scout.providers.anthropic import AnthropicProvider  # noqa: E402
from konfai_scout.providers.base import ImageContent, ProviderError, ToolResult, ToolSpec, is_loopback  # noqa: E402
from konfai_scout.providers.openai import OpenAIProvider  # noqa: E402

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)
TOOLS = [
    ToolSpec(
        "note",
        "n",
        {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"], "additionalProperties": False},
    )
]


class Response:
    def __init__(self, data):
        self.data = data

    def to_dict(self):
        return self.data


class FakeAnthropic:
    def __init__(self, responses, error=None):
        self.responses, self.error, self.requests = list(responses), error, []
        self.messages = self

    def create(self, **request):
        self.requests.append(copy.deepcopy(request))
        if self.error:
            raise self.error
        return Response(self.responses.pop(0))


def http_response(status: int) -> SimpleNamespace:
    """What the SDKs' errors read of a response, whichever HTTP library the installed SDK uses."""
    return SimpleNamespace(status_code=status, headers={}, request=SimpleNamespace(method="POST", url="http://api"))


def anthropic_error(status: int, type_: str = "api_error"):
    import anthropic

    cls = {429: anthropic.RateLimitError, 500: anthropic.InternalServerError, 400: anthropic.BadRequestError}[status]
    return cls("boom", response=http_response(status), body={"error": {"type": type_}})


def test_anthropic_request_shape_and_parsing():
    first = {
        "id": "msg1",
        "model": "claude-x",
        "stop_reason": "tool_use",
        "content": [
            {"type": "text", "text": "looking"},
            {"type": "tool_use", "id": "t1", "name": "note", "input": {"a": "b"}},
        ],
        "usage": {
            "input_tokens": 100,
            "output_tokens": 20,
            "cache_read_input_tokens": 40,
            "cache_creation_input_tokens": 5,
        },
    }
    second = {
        "id": "msg2",
        "model": "claude-x",
        "stop_reason": "end_turn",
        "content": [{"type": "text", "text": "done"}],
        "usage": {"input_tokens": 10, "output_tokens": 2},
    }
    client = FakeAnthropic([first, second])
    provider = AnthropicProvider("claude-opus-5-5", effort="high", client=client)
    provider.start("system text", TOOLS)
    turn = provider.user("card", [ImageContent(PNG, "[v001]")])
    assert turn.stop == "tool_use" and turn.tool_calls[0].name == "note" and turn.tool_calls[0].arguments == {"a": "b"}
    assert turn.usage.input_tokens == 100 and turn.usage.cache_read_tokens == 40 and turn.usage.cache_write_tokens == 5
    request = client.requests[0]
    assert request["model"] == "claude-opus-5-5" and request["output_config"] == {"effort": "high"}
    assert request["system"][0]["cache_control"] == {"type": "ephemeral"} and request["tools"][0]["strict"] is True
    content = request["messages"][0]["content"]
    assert content[0]["text"] == "card" and content[-1]["type"] == "image"
    assert base64.b64decode(content[-1]["source"]["data"]) == PNG
    turn = provider.tool_results(
        [ToolResult("t1", "ok", [ImageContent(PNG, "[v002]")]), ToolResult("t2", "nope", is_error=True)]
    )
    assert turn.stop == "end_turn" and turn.text == "done"
    blocks = client.requests[1]["messages"][-1]["content"]
    assert blocks[0]["type"] == "tool_result" and blocks[0]["content"][-1]["type"] == "image"
    assert blocks[1]["is_error"] is True
    assert client.requests[1]["messages"][1]["role"] == "assistant"
    transcript = provider.transcript()
    assert transcript[0]["content"][-1] == {"type": "image", "bytes": len(PNG)}
    assert json.dumps(transcript)


def test_anthropic_refusal_and_errors():
    refusal = {
        "stop_reason": "refusal",
        "content": [],
        "usage": {},
        "stop_details": {"category": "cyber", "explanation": "x"},
    }
    provider = AnthropicProvider("m", client=FakeAnthropic([refusal]))
    provider.start("s", TOOLS)
    turn = provider.user("hi")
    assert turn.stop == "refusal" and "cyber" in turn.text
    for status, retryable in ((429, True), (500, True), (400, False)):
        provider = AnthropicProvider("m", client=FakeAnthropic([], error=anthropic_error(status)))
        provider.start("s", TOOLS)
        with pytest.raises(ProviderError) as error:
            provider.user("hi")
        assert error.value.retryable is retryable


class FakeOpenAI:
    def __init__(self, responses, error=None):
        self.responses, self.error, self.requests = list(responses), error, []
        self.chat = self
        self.completions = self

    def create(self, **request):
        self.requests.append(copy.deepcopy(request))
        if self.error:
            raise self.error
        return Response(self.responses.pop(0))


def test_openai_request_shape_and_parsing():
    first = {
        "id": "c1",
        "model": "gpt-x",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "t1", "type": "function", "function": {"name": "note", "arguments": '{"a": "b"}'}}
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 120, "completion_tokens": 30, "prompt_tokens_details": {"cached_tokens": 20}},
    }
    second = {
        "id": "c2",
        "model": "gpt-x",
        "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "done"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 1},
    }
    client = FakeOpenAI([first, second])
    provider = OpenAIProvider("gpt-5", client=client)
    provider.start("system text", TOOLS)
    turn = provider.user("card", [ImageContent(PNG, "[v001]")])
    assert turn.stop == "tool_use" and turn.tool_calls[0].arguments == {"a": "b"}
    assert turn.usage.input_tokens == 100 and turn.usage.cache_read_tokens == 20
    request = client.requests[0]
    assert request["messages"][0] == {"role": "system", "content": "system text"}
    assert request["tools"][0]["function"]["name"] == "note"
    assert request["messages"][1]["content"][-1]["image_url"]["url"].startswith("data:image/png;base64,")
    turn = provider.tool_results([ToolResult("t1", "ok", [ImageContent(PNG, "[v002]")])])
    messages = client.requests[1]["messages"]
    assert (
        messages[-2]["role"] == "tool"
        and messages[-2]["tool_call_id"] == "t1"
        and "image(s)" in messages[-2]["content"]
    )
    assert messages[-1]["role"] == "user" and messages[-1]["content"][-1]["type"] == "image_url"
    assert turn.stop == "end_turn" and turn.text == "done"
    assert provider.transcript()[-2]["content"][-1]["type"] == "image"


def test_openai_errors_and_locality():
    import openai

    response = http_response(429)
    provider = OpenAIProvider(
        "m", client=FakeOpenAI([], error=openai.RateLimitError("rl", response=response, body=None))
    )
    provider.start("s", TOOLS)
    with pytest.raises(ProviderError) as error:
        provider.user("hi")
    assert error.value.retryable
    assert OpenAIProvider("m", base_url="http://localhost:8000/v1", client=object()).remote is False
    assert OpenAIProvider("m", base_url="https://api.example.com/v1", client=object()).remote is True
    assert OpenAIProvider("m", base_url="http://10.0.0.5:8000/v1", local=True, client=object()).remote is False
    assert OpenAIProvider("m", local=True, client=object()).remote is True  # no base_url: the flag names no server
    assert AnthropicProvider("m", base_url="http://localhost:8080", client=object()).remote is False
    assert AnthropicProvider("m", local=True, client=object()).remote is True
    assert is_loopback("http://127.0.0.1:11434/v1") and not is_loopback(None)


def test_the_privacy_gate():
    assert is_remote(ScoutConfig(provider="anthropic"))
    with pytest.raises(ProviderError) as error:
        check_privacy(ScoutConfig(provider="anthropic", model="claude-opus-5-5"))
    assert "allow_cloud_images" in str(error.value)
    check_privacy(ScoutConfig(provider="anthropic", allow_cloud_images=True))
    check_privacy(ScoutConfig(provider="openai", base_url="http://localhost:11434/v1"))
    check_privacy(ScoutConfig(provider="openai", base_url="http://gpu-box:8000/v1", local_provider=True))
    with pytest.raises(ProviderError):
        check_privacy(ScoutConfig(provider="openai", base_url="http://gpu-box:8000/v1"))
    # the flag describes a server named by base_url: without one it changes nothing, for any provider
    for provider in ("anthropic", "openai", "claude-code", "codex"):
        assert is_remote(ScoutConfig(provider=provider, local_provider=True))
    assert not is_remote(ScoutConfig(provider="anthropic", base_url="http://gpu-box:8080", local_provider=True))
    assert is_remote(ScoutConfig(provider="claude-code", base_url="http://localhost:1", local_provider=True))
    check_privacy(ScoutConfig(provider="mock"))
    assert make_provider(ScoutConfig(provider="mock")).name == "mock"
    assert ScoutConfig(provider="anthropic").model == "claude-opus-5-5"


def test_a_retried_turn_is_sent_once():
    import openai

    rate_limited = openai.RateLimitError("rl", response=http_response(429), body=None)
    anthropic_ok = {"stop_reason": "end_turn", "content": [], "usage": {}}
    openai_ok = {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "x"}}], "usage": {}}
    for make, fake in (
        (AnthropicProvider, FakeAnthropic([anthropic_ok], error=anthropic_error(429))),
        (OpenAIProvider, FakeOpenAI([openai_ok], error=rate_limited)),
    ):
        provider = make("m", client=fake)
        provider.start("s", TOOLS)
        kept = len(provider.messages)  # OpenAI's system message
        with pytest.raises(ProviderError):
            provider.user("hi")
        assert len(provider.messages) == kept
        fake.error = None
        provider.user("hi")
        assert len(fake.requests[-1]["messages"]) == kept + 1 and len(provider.messages) == kept + 2
