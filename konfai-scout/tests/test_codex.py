# SPDX-License-Identifier: Apache-2.0
"""The codex provider against a fake Codex CLI that really connects to the MCP server Scout serves its tools on:
the schemas and read-only annotations as served, images in the results, the budget, the resumed session, errors."""

import asyncio
import json

import pytest

pytest.importorskip("fastmcp")

from fastmcp import Client
from konfai_scout import ScoutConfig, inspect
from konfai_scout.config import Budget
from konfai_scout.providers import ProviderError, check_privacy, make_provider
from konfai_scout.providers.codex import CodexProvider, _model_of, _without_image_data

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
USAGE = {
    "input_tokens": 1000,
    "cached_input_tokens": 600,
    "cache_write_input_tokens": 0,
    "output_tokens": 50,
    "reasoning_output_tokens": 10,
}


class FakeCodex:
    """Plays the Codex CLI: connects to the MCP server Scout started, calls the tools over it as the CLI would, and
    streams the JSONL events the CLI prints. One script entry per turn: the calls to make, or a failure."""

    def __init__(self, script: list[dict]) -> None:
        self.script = list(script)
        self.commands: list[list[str]] = []
        self.prompts: list[str] = []
        self.tools: dict = {}
        self.results: list[dict] = []

    def events(self, cmd: list[str], prompt: str):
        self.commands.append(cmd)
        self.prompts.append(prompt)
        plan = self.script.pop(0)
        yield {"type": "thread.started", "thread_id": "t1"}
        if "fail" in plan:
            yield {"type": "turn.failed", "error": {"message": plan["fail"]}}
            yield {"type": "exit", "code": 1, "stderr": ""}
            return
        yield {"type": "turn.started"}
        url = next(a for a in cmd if a.startswith("mcp_servers.scout.url=")).split("=", 1)[1].strip('"')
        for n, (name, args, result) in enumerate(asyncio.run(self._drive(url, plan.get("calls", [])))):
            item = {"id": f"item_{n}", "type": "mcp_tool_call", "server": "scout", "tool": name, "arguments": args}
            yield {"type": "item.completed", "item": {**item, "result": result, "error": None, "status": "completed"}}
        yield {
            "type": "item.completed",
            "item": {"id": "item_x", "type": "agent_message", "text": plan.get("text", "done")},
        }
        yield {"type": "turn.completed", "usage": USAGE}
        yield {"type": "exit", "code": 0, "stderr": ""}

    async def _drive(self, url: str, calls: list) -> list[tuple]:
        out = []
        async with Client(url) as client:
            self.tools = {tool.name: tool for tool in await client.list_tools()}
            views: list[str] = []
            for name, args in calls:
                if callable(args):
                    args = args(views)
                response = await client.call_tool(name, args, raise_on_error=False)
                content = [block.model_dump(mode="json", by_alias=True) for block in response.content]
                if not response.is_error:
                    payload = json.loads(content[0]["text"])
                    if "view" in payload:
                        views.append(payload["view"]["id"])
                result = {"content": content, "structured_content": None}
                self.results.append(result)
                out.append((name, args, result))
        return out


def _provider(monkeypatch, script: list[dict], **kwargs) -> tuple[CodexProvider, FakeCodex]:
    provider = CodexProvider(**kwargs)
    fake = FakeCodex(script)
    monkeypatch.setattr(CodexProvider, "_events", lambda self, cmd, prompt: fake.events(cmd, prompt))
    return provider, fake


def test_the_cli_drives_the_tools_over_the_served_mcp_server(fixtures, tmp_path, monkeypatch):
    case, _ = fixtures["segmentation"]
    script = [
        {
            "calls": [
                ("view", VIEW),
                ("view", {**VIEW, "mode": "both"}),
                ("region_stats", {"box_mm": {"center": [0.0, 0.0, 0.0], "size": 10.0}}),  # refused: the budget is spent
                ("note", lambda views: {**NOTE, "views": views[:1]}),
                ("conclude", CONCLUDE),
            ]
        }
    ]
    provider, fake = _provider(monkeypatch, script, effort="high")
    config = ScoutConfig(provider="codex", allow_cloud_images=True, budget=Budget(max_tool_calls=2), max_px=200)
    report = inspect(case, config, tmp_path, provider=provider)

    assert report.error is None and report.conclusion.by == "model" and report.conclusion.verdict == "defective"
    assert [c.tool for c in report.tool_calls] == ["view", "view", "note", "conclude"]
    assert report.budget["exhausted"].startswith("2 tool calls") and report.budget["spent"]["tool_calls"] == 2
    assert report.observations[0].views == [report.views[3].id]
    assert report.usage.images_sent == 5 and all((tmp_path / v.path).exists() for v in report.views)
    assert report.provider["name"] == "codex" and report.provider["remote"] and report.provider["model"] == "default"
    assert (
        report.usage.input_tokens == 400 and report.usage.cache_read_tokens == 600 and report.usage.output_tokens == 50
    )
    assert report.usage.requests == 1 and not report.usage.cost_known
    # the command the CLI was given
    cmd = fake.commands[0]
    assert cmd[:2] == ["codex", "exec"] and "--ignore-user-config" in cmd and "--json" in cmd and "-m" not in cmd
    assert cmd[cmd.index("-s") + 1] == "read-only" and cmd.count("--disable") == 6
    assert len([a for a in cmd if a.startswith("--image=")]) == 3  # the coverage montages, as files
    assert 'model_reasoning_effort="high"' in cmd
    prompt = fake.prompts[0]
    assert prompt.startswith("You are KonfAI Scout") and "Task: segmentation" in prompt
    assert "The attached images are, in order: [v001], [v002], [v003]." in prompt
    # what the MCP server served: the schemas as they are, every tool read-only for the approval policy
    assert sorted(fake.tools) == ["conclude", "measure", "montage", "note", "region_stats", "view"]
    view = fake.tools["view"]
    schema = getattr(view, "input_schema", None) or view.inputSchema
    assert (
        schema["properties"]["plane"]["enum"] == ["axial", "coronal", "sagittal"]
        and schema["additionalProperties"] is False
    )
    hints = view.annotations.model_dump()
    assert hints.get("readOnlyHint", hints.get("read_only_hint")) is True
    # a view's result carries its image; the refused call reached the model as an error
    assert fake.results[0]["content"][2]["type"] == "image" and fake.results[0]["content"][2]["mimeType"] == "image/png"
    assert "Budget exhausted" in fake.results[2]["content"][0]["text"]
    # the transcript holds the conversation without the image bytes
    transcript = json.loads((tmp_path / "transcript.json").read_text())
    assert transcript[0]["role"] == "user" and [b["type"] for b in transcript[0]["content"]].count("image") == 3
    tool_results = [b for m in transcript if m["role"] == "user" for b in m["content"] if b["type"] == "tool_result"]
    assert tool_results[0]["content"][2]["type"] == "image" and tool_results[0]["content"][2]["bytes"] > 0
    assert transcript[-1]["role"] == "result" and transcript[-1]["thread_id"] == "t1"


def test_a_silent_turn_is_reminded_on_the_resumed_session_and_failures_are_classified(fixtures, tmp_path, monkeypatch):
    case, _ = fixtures["segmentation_no_truth"]
    script = [{"calls": [], "text": "looks fine"}, {"calls": [("conclude", {**CONCLUDE, "verdict": "acceptable"})]}]
    provider, fake = _provider(monkeypatch, script)
    config = ScoutConfig(provider="codex", allow_cloud_images=True, retry_wait_seconds=0.0, max_px=200)
    report = inspect(case, config, tmp_path / "silent", provider=provider)
    assert report.error is None and report.conclusion.verdict == "acceptable" and report.usage.requests == 2
    second = fake.commands[1]
    assert second[:4] == ["codex", "exec", "resume", "t1"] and "-C" not in second and "-s" not in second
    assert "Nothing was recorded" in fake.prompts[1] and not fake.prompts[1].startswith("You are KonfAI Scout")

    provider, fake = _provider(
        monkeypatch, [{"fail": "rate limit exceeded (429)"}, {"calls": [("conclude", CONCLUDE)]}]
    )
    report = inspect(case, config, tmp_path / "transient", provider=provider)
    assert report.error is None and len(fake.commands) == 2  # retried once, on the same thread
    provider, fake = _provider(monkeypatch, [{"fail": "boom"}])
    report = inspect(case, config, tmp_path / "fatal", provider=provider)
    assert report.error == "provider error: Codex: boom" and report.conclusion.by == "scout"


def test_the_gate_the_factory_and_the_records(tmp_path, monkeypatch):
    with pytest.raises(ProviderError, match="allow_cloud_images"):
        check_privacy(ScoutConfig(provider="codex"))
    config = ScoutConfig(provider="codex", allow_cloud_images=True, effort="medium")
    provider = make_provider(config)
    assert isinstance(provider, CodexProvider) and provider.model == "default" and provider.effort == "medium"
    assert provider.drives_tools and isinstance(provider.sdk_version(), str)
    with pytest.raises(ProviderError, match="runs the tool loop itself"):
        provider.tool_results([])
    # the model that answered, from the CLI's own session record
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    rollout = tmp_path / "sessions" / "2026" / "11" / "07" / "rollout-2026-11-07T20-00-00-t9.jsonl"
    rollout.parent.mkdir(parents=True)
    rollout.write_text(
        json.dumps({"type": "session_meta", "payload": {"id": "t9"}})
        + "\n"
        + json.dumps({"type": "turn_context", "payload": {"model": "gpt-test"}})
        + "\n"
    )
    assert _model_of("t9") == "gpt-test" and _model_of("t10") is None and _model_of(None) is None
    stripped = _without_image_data(
        [{"type": "text", "text": "t"}, {"type": "image", "data": "AAAA", "mimeType": "image/png"}]
    )
    assert stripped == [{"type": "text", "text": "t"}, {"type": "image", "bytes": 3}]
