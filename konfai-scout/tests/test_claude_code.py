# SPDX-License-Identifier: Apache-2.0
"""The claude-code provider against a fake Claude Agent SDK session: Scout's tools served in-process with their
images, the record and the budget the same as with the API providers, the session resumed between turns, errors."""

import base64
import json
from functools import partial
from pathlib import Path

import pytest

pytest.importorskip("claude_agent_sdk")

from claude_agent_sdk import (
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)
from konfai_scout import ScoutConfig, inspect
from konfai_scout.config import Budget
from konfai_scout.providers import ProviderError, check_privacy, make_provider
from konfai_scout.providers.base import ImageContent, ToolResult, without_image_data
from konfai_scout.providers.claude_code import ClaudeCodeProvider, _mcp_result
from test_tools import unsure_reader

inspect = partial(inspect, reader=unsure_reader)  # the blind readings are not sessions of the scripted client

MODEL = "claude-opus-5-5"
NOTE = {
    "kind": "suspected_error",
    "defect": "missing_structure",
    "severity": "major",
    "confidence": 0.7,
    "description": "the cube is absent from the prediction",
    "box_mm": None,
    "plane": "axial",
    "measurements": [],
}
VIEW = {"plane": "axial", "mode": "prediction", "position_mm": None, "index": None, "zoom": None, "window": None}
CONCLUDE = {"verdict": "defective", "summary": "the cube is missing", "confidence": 0.7}


def _as_cli(result: dict) -> list[dict]:
    """What the CLI hands back to the SDK for an MCP result: API-shaped image blocks."""
    out = []
    for block in result["content"]:
        if block["type"] == "image":
            out.append(
                {"type": "image", "source": {"type": "base64", "media_type": block["mimeType"], "data": block["data"]}}
            )
        else:
            out.append(block)
    return out


def _result(**overrides) -> ResultMessage:
    base = {
        "subtype": "success",
        "duration_ms": 5,
        "duration_api_ms": 4,
        "is_error": False,
        "num_turns": 1,
        "session_id": "s1",
        "stop_reason": "end_turn",
        "total_cost_usd": 0.01,
        "usage": {
            "input_tokens": 100,
            "output_tokens": 20,
            "cache_read_input_tokens": 30,
            "cache_creation_input_tokens": 5,
        },
        "model_usage": {MODEL: {}},
    }
    return ResultMessage(**{**base, **overrides})


class FakeSession:
    """Plays Claude Code: calls Scout's tools through the in-process server's handlers, as the CLI would, and
    streams the messages the SDK would yield. One script entry per turn: the calls to make, or an error."""

    def __init__(self, provider: ClaudeCodeProvider, script: list[dict]) -> None:
        self.provider = provider
        self.script = list(script)
        self.prompts: list[list[dict]] = []
        self.options: list = []
        self.results: list[dict] = []

    async def stream(self, content, options):
        self.prompts.append(content)
        self.options.append(options)
        plan = self.script.pop(0)
        if "error" in plan:
            yield _result(is_error=True, api_error_status=plan["error"], result="boom", num_turns=0)
            return
        if "cli_error" in plan:  # what an outdated CLI does: an API error as text, then the SDK's bare Exception
            yield AssistantMessage(content=[TextBlock(text=plan["cli_error"])], model="<synthetic>")
            yield _result(is_error=True, result=None, num_turns=1)
            raise Exception("Claude Code returned an error result: success")
        views: list[str] = []
        usage = plan.get("usage")  # each request's usage, on its message, as the CLI reports it while it runs
        for n, (name, args) in enumerate(plan.get("calls", []), 1):
            if callable(args):
                args = args(views)
            for block in (
                TextBlock(text=f"calling {name}"),
                ToolUseBlock(id=f"toolu_{n}", name=f"mcp__scout__{name}", input=args),
            ):
                yield AssistantMessage(content=[block], model=MODEL, usage=usage, message_id=f"msg_{n}")
            result = await self.provider.sdk_tools[name].handler(args)
            self.results.append(result)
            if not result.get("is_error"):
                payload = json.loads(result["content"][0]["text"])
                if "view" in payload:
                    views.append(payload["view"]["id"])
            yield UserMessage(
                content=[
                    ToolResultBlock(tool_use_id=f"toolu_{n}", content=_as_cli(result), is_error=result.get("is_error"))
                ]
            )
        yield AssistantMessage(content=[TextBlock(text=plan.get("text", "done"))], model=MODEL)
        calls = len(plan.get("calls", []))
        total = {key: value * calls for key, value in usage.items()} if usage else None
        yield _result(num_turns=calls + 1, **({"usage": total} if total else {}))


def _provider(monkeypatch, script: list[dict]) -> tuple[ClaudeCodeProvider, FakeSession]:
    provider = ClaudeCodeProvider("claude-test", max_turns=12)
    fake = FakeSession(provider, script)
    monkeypatch.setattr(ClaudeCodeProvider, "_stream", lambda self, content, options: fake.stream(content, options))
    return provider, fake


def test_the_session_drives_the_tools_under_the_budget_and_the_record_is_complete(fixtures, tmp_path, monkeypatch):
    case, _ = fixtures["segmentation"]
    world = case.grid.world_box()  # a small box at the centre, which the central slice shows
    centre = {"center": ((world.low_xyz + world.high_xyz) / 2).tolist(), "size": 6.0}
    script = [
        {
            "calls": [
                ("view", VIEW),
                ("view", {**VIEW, "mode": "both"}),
                ("region_stats", {"box_mm": {"center": [0.0, 0.0, 0.0], "size": 10.0}}),  # refused: the budget is spent
                ("note", lambda views: {**NOTE, "views": views[:1], "box_mm": centre}),
                ("conclude", CONCLUDE),
            ]
        }
    ]
    provider, fake = _provider(monkeypatch, script)
    config = ScoutConfig(provider="claude-code", allow_cloud_images=True, budget=Budget(max_tool_calls=2), max_px=200)
    report = inspect(case, config, tmp_path, provider=provider)

    assert report.error is None and report.conclusion.by == "model" and report.conclusion.verdict == "defective"
    assert [c.tool for c in report.tool_calls] == ["view", "view", "note", "conclude"]
    assert report.budget["exhausted"].startswith("2 tool calls") and report.budget["spent"]["tool_calls"] == 2
    assert (
        report.observations[0].views == [[v for v in report.views if v.sent][3].id]
        and report.observations[0].kind == "suspected_error"
    )
    blind = sum(not v.sent for v in report.views)  # the views of the blind readings, sent to their reader
    assert report.usage.images_sent == 5 + blind and all((tmp_path / v.path).exists() for v in report.views)
    assert (
        report.provider["name"] == "claude-code"
        and report.provider["remote"]
        and report.provider["model"] == "claude-test"
    )
    # the usage comes from the result message; the price from the model that answered, not the one configured
    assert (
        report.usage.input_tokens == 100
        and report.usage.cache_read_tokens == 30
        and report.usage.requests == 1 + blind // 2
    )
    assert report.usage.cost_known and report.usage.cost_usd > 0
    # what the session was given
    options = fake.options[0]
    assert options.tools == [] and options.permission_mode == "dontAsk" and options.resume is None
    # nothing of where Scout is run from reaches the model: no settings, and a working directory of Scout's own
    assert options.setting_sources == [] and Path(options.cwd).resolve() != Path.cwd().resolve()
    assert options.allowed_tools == [
        f"mcp__scout__{t}" for t in ("view", "montage", "measure", "region_stats", "note", "conclude")
    ]
    assert options.system_prompt.startswith("You are KonfAI Scout") and "mcp__scout__view" in options.system_prompt
    assert [b["type"] for b in fake.prompts[0]].count("image") == 3  # the coverage montages, in the first message
    # a view's result carries its image; the refused call reached the model as an error
    assert fake.results[0]["content"][2]["mimeType"] == "image/png" and "is_error" not in fake.results[0]
    assert fake.results[2]["is_error"] and "Budget exhausted" in fake.results[2]["content"][0]["text"]
    # the transcript holds the conversation without the image bytes
    transcript = json.loads((tmp_path / "transcript.json").read_text())
    first = transcript[0]
    assert first["role"] == "user" and all("bytes" in b for b in first["content"] if b["type"] == "image")
    tool_results = [b for m in transcript if m["role"] == "user" for b in m["content"] if b["type"] == "tool_result"]
    assert tool_results[0]["content"][2] == {"type": "image", "bytes": len(_png_of(fake.results[0]))}
    assert transcript[-1]["role"] == "result" and transcript[-1]["session_id"] == "s1"


def test_max_usd_and_max_seconds_stop_the_session_mid_loop(fixtures, tmp_path, monkeypatch):
    """The SDK runs the whole loop in one request: the cost of each message and the time are counted as it runs,
    inside the served tools, and an exploring call past a limit is refused while a note and the conclusion run."""
    case, _ = fixtures["segmentation"]
    calls = [
        ("view", VIEW),
        ("view", {**VIEW, "mode": "both"}),  # refused: two requests of 100k input tokens are $0.80 at $4/M
        ("note", lambda views: {**NOTE, "views": views[:1]}),
        ("conclude", CONCLUDE),
    ]
    provider, fake = _provider(monkeypatch, [{"calls": calls, "usage": {"input_tokens": 100_000}}])
    config = ScoutConfig(provider="claude-code", allow_cloud_images=True, budget=Budget(max_usd=0.5), max_px=200)
    report = inspect(case, config, tmp_path / "usd", provider=provider)
    assert [c.tool for c in report.tool_calls] == ["view", "note", "conclude"]
    assert "Budget exhausted ($0.800 (limit $0.50))" in fake.results[1]["content"][0]["text"]
    assert report.budget["exhausted"] == "$1.600 (limit $0.50)" and report.budget["not_enforced"] is None
    assert report.usage.cost_usd == pytest.approx(
        1.6, abs=1e-3
    )  # (and the blind readings' few tokens) the result's usage replaces the running estimate, not added to it

    calls[-1] = ("conclude", {**CONCLUDE, "verdict": "insufficient_evidence"})  # no view: the note is refused
    provider, fake = _provider(monkeypatch, [{"calls": calls}])
    config = ScoutConfig(provider="claude-code", allow_cloud_images=True, budget=Budget(max_seconds=0), max_px=200)
    report = inspect(case, config, tmp_path / "seconds", provider=provider)
    assert [c.tool for c in report.tool_calls] == ["note", "conclude"] and report.conclusion.by == "model"
    assert "s (limit 0 s)" in fake.results[0]["content"][0]["text"] and report.budget["spent"]["tool_calls"] == 0


def _png_of(result: dict) -> bytes:
    return base64.b64decode(result["content"][2]["data"])


def test_a_silent_session_is_reminded_on_the_same_session(fixtures, tmp_path, monkeypatch):
    case, _ = fixtures["segmentation_clean"]
    script = [
        {"calls": [], "text": "looks fine to me"},
        {"calls": [("conclude", {**CONCLUDE, "verdict": "acceptable"})]},
    ]
    provider, fake = _provider(monkeypatch, script)
    report = inspect(
        case, ScoutConfig(provider="claude-code", allow_cloud_images=True, max_px=200), tmp_path, provider=provider
    )
    assert report.error is None and report.conclusion.verdict == "acceptable" and report.usage.requests == 2
    assert fake.options[1].resume == "s1" and "Nothing was recorded" in fake.prompts[1][0]["text"]


def test_errors_are_classified(fixtures, tmp_path, monkeypatch):
    case, _ = fixtures["synthesis_no_target"]
    config = ScoutConfig(provider="claude-code", allow_cloud_images=True, retry_wait_seconds=0.0, max_px=200)
    provider, fake = _provider(
        monkeypatch, [{"error": 529}, {"calls": [("conclude", {**CONCLUDE, "verdict": "acceptable"})]}]
    )
    report = inspect(case, config, tmp_path / "transient", provider=provider)
    assert report.error is None and len(fake.prompts) == 2  # overloaded once, retried
    provider, fake = _provider(monkeypatch, [{"error": None}])
    report = inspect(case, config, tmp_path / "fatal", provider=provider)
    assert report.error == "provider error: Claude Code: boom" and len(fake.prompts) == 1
    provider, fake = _provider(monkeypatch, [{"cli_error": "API Error: 400 this CLI does not support this model"}])
    report = inspect(case, config, tmp_path / "outdated", provider=provider)
    assert report.error == "provider error: Claude Code: API Error: 400 this CLI does not support this model"


def test_the_gate_and_the_factory():
    with pytest.raises(ProviderError, match="allow_cloud_images"):
        check_privacy(ScoutConfig(provider="claude-code"))
    config = ScoutConfig(provider="claude-code", allow_cloud_images=True, effort="high")
    check_privacy(config)
    provider = make_provider(config)
    assert isinstance(provider, ClaudeCodeProvider) and provider.model == MODEL and provider.effort == "high"
    assert provider.max_turns == 2 * config.budget.max_tool_calls + 8 and provider.drives_tools
    with pytest.raises(ProviderError, match="runs the tool loop itself"):
        provider.tool_results([])


def test_mcp_results_and_the_transcript_without_image_bytes():
    png = b"\x89PNG\r\n\x1a\n" + bytes(16)
    result = _mcp_result(ToolResult(call_id="c", text="{}", images=[ImageContent(png, "[v001]")]))
    assert [b["type"] for b in result["content"]] == ["text", "text", "image"] and "is_error" not in result
    assert base64.b64decode(result["content"][2]["data"]) == png and result["content"][2]["mimeType"] == "image/png"
    assert _mcp_result(ToolResult(call_id="c", text="bad", is_error=True))["is_error"]
    stripped = without_image_data(
        [{"type": "text", "text": "t"}, {"type": "image", "data": result["content"][2]["data"]}]
    )
    assert stripped == [{"type": "text", "text": "t"}, {"type": "image", "bytes": len(png)}]
    assert without_image_data("plain") == "plain"
