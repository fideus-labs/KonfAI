# SPDX-License-Identifier: Apache-2.0
"""The Studio agent: an LLM driving the konfai-mcp tools, streamed as UI events.

The LLM backend is pluggable (``KONFAI_STUDIO_LLM``): ``anthropic`` (Claude API) or
``openai`` for any OpenAI-compatible local server (vLLM / Ollama / LM Studio). The MCP
side never depends on which brain is used, and the imaging data never reaches the LLM:
tools run locally on the compute node and return text.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any

from fastmcp import Client


def _resolve_mcp_command(command: str) -> str:
    """Resolve the ``konfai-mcp`` console script to an absolute path.

    Studio may be started by its bare binary, which does not put the env's ``bin`` dir on ``PATH``;
    the SDK/transport then fails to spawn ``konfai-mcp`` by name. The script sits beside this
    interpreter, so prefer that, then ``PATH``, then the bare name — the agent finds its tools
    regardless of how Studio was launched.
    """
    if os.path.isabs(command) or command != "konfai-mcp":
        return command
    beside = Path(sys.executable).with_name(command)
    return str(beside) if beside.exists() else (shutil.which(command) or command)

DEFAULT_MODEL = os.environ.get("KONFAI_STUDIO_MODEL", "claude-opus-4-8")
MAX_TOKENS = int(os.environ.get("KONFAI_STUDIO_MAX_TOKENS", "16000"))
MAX_TURNS = int(os.environ.get("KONFAI_STUDIO_MAX_TURNS", "30"))

SYSTEM_PROMPT = (
    "You are KonfAI Studio, an assistant that drives KonfAI (a config-driven deep-learning "
    "framework for medical imaging) through its MCP tools. The user is a clinician-researcher: "
    "they point you at a dataset and describe a task; you inspect data, author configs, launch "
    "and monitor train/predict/eval jobs, and report results plainly. Prefer the safe, read-only "
    "tools (inspect_dataset, list_apps, describe_*, summarize_session) before anything that runs a "
    "job. Never fabricate a tool result. When you have enough information to act, act; state what "
    "you did and what you found.\n\n"
    "Be economical: keep answers short -- a sentence or a compact list, never long prose. Give what you "
    "did, what you found, and the next step; skip preamble, capability lists, and restating the request.\n\n"
    "At the start, don't ask permission to look or list what you can do: inspect the dataset "
    "(inspect_dataset, browse_dataset) and report the essentials tersely -- modality, shape and spacing "
    "(x,y,z), case count, label classes, any split, anything anomalous. Then pick the cheapest fitting "
    "route -- (1) a published app as-is (list_apps -> describe_app -> import_app -> run_prediction), "
    "(2) fine-tune a close app (import_app -> run_resume with weights_only), (3) author a config and train "
    "from scratch -- and propose it with one concrete next step. At most one question, and only when the "
    "choice is genuinely ambiguous.\n\n"
    "Studio shows a 3D image viewer (NiiVue) beside this chat. To display a case, call a tool that "
    "surfaces its absolute volume path (e.g. preview_volume, or read/inspect the file): any "
    ".nii/.nii.gz/.mha/.mhd/.nrrd path in a tool call or result is auto-loaded into that viewer. "
    "The viewer is part of the UI and always available — never say it is offline or unavailable. "
    "Volumes are decoded in the browser, so you never need code execution, a screenshot, or a PNG "
    "conversion to show one, and you must never fabricate an image.\n\n"
    "When the user attaches a paper (a local PDF path or an arXiv/URL), read it, extract the task, "
    "architecture, losses, augmentations and training setup, then reproduce it as a KonfAI experiment: "
    "author the config with the tools and explain your design choices. Validate it, but do not launch "
    "training until the user confirms.\n\n"
    "If a train/predict job fails because its outputs already exist (a 'model already exists' / "
    "'pass --overwrite' error), do not dump the traceback. Say plainly that this run's outputs are "
    "already there, and offer two clear choices: (1) re-run overwriting the existing outputs, or "
    "(2) use a new run name — then do whichever the user picks."
)

# A tool executor: given (name, arguments) run the MCP tool, return (ok, text_preview).
ToolFn = Callable[[str, dict[str, Any]], Awaitable[tuple[bool, str]]]

# Absolute NIfTI / MHA volume paths surfaced in a tool result -> auto-load in NiiVue.
_VOLUME_RE = re.compile(r"(/[^\s\"'`|,)\]]+\.(?:nii\.gz|nii|mha|mhd|nrrd))")


def _detect_volume(text: str) -> str | None:
    """First existing volume path in a chunk of text, so the chat can drive the viewer."""
    for match in _VOLUME_RE.finditer(text or ""):
        path = match.group(1)
        if os.path.isfile(path):
            return path
    return None


def _next_actions(text: str) -> list[str]:
    """The MCP tool's own ``next_actions`` (registered tool names) — the ground truth for what to
    suggest next. Best-effort: a non-JSON or unshaped result simply yields no suggestions."""
    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        return []
    actions = payload.get("next_actions") if isinstance(payload, dict) else None
    if not isinstance(actions, list):
        return []
    names: list[str] = []
    for action in actions:
        if isinstance(action, str):
            names.append(action)
        elif isinstance(action, dict):
            name = action.get("tool") or action.get("action") or action.get("name")
            if isinstance(name, str):
                names.append(name)
    return names[:6]


# The UI only needs a short preview of a tool result; the full text is parsed here first.
_PREVIEW_LIMIT = 600


async def with_volume_events(events: AsyncIterator[dict[str, Any]]) -> AsyncIterator[dict[str, Any]]:
    """Pass an agent's event stream through, deriving the UI's out-of-band signals from a tool's
    full result in ONE place: a ``volume`` event (what to show in the viewer) and a ``next_actions``
    event (what to suggest next). Backends stay free of any UI concern; the preview is trimmed here."""
    async for event in events:
        volume: str | None = None
        actions: list[str] = []
        if event["type"] == "tool_call":
            volume = _detect_volume(json.dumps(event.get("input"), default=str))
        elif event["type"] == "tool_result":
            full = event.get("preview", "")
            volume = _detect_volume(full)
            actions = _next_actions(full)
            event["preview"] = full[:_PREVIEW_LIMIT]  # trim for the UI, after parsing the full text
        yield event
        if volume:
            yield {"type": "volume", "path": volume}
        if actions:
            yield {"type": "next_actions", "actions": actions}


def _mcp_tools_to_anthropic(mcp_tools: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for t in mcp_tools:
        schema = t.inputSchema or {"type": "object", "properties": {}}
        out.append({"name": t.name, "description": (t.description or "")[:1024], "input_schema": schema})
    return out


def _mcp_tools_to_openai(mcp_tools: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for t in mcp_tools:
        schema = t.inputSchema or {"type": "object", "properties": {}}
        out.append(
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": (t.description or "")[:1024],
                    "parameters": schema,
                },
            }
        )
    return out


def _tool_result_text(result: Any) -> str:
    if getattr(result, "data", None) is not None:
        try:
            return json.dumps(result.data, default=str)[:60000]
        except (TypeError, ValueError):
            pass
    parts: list[str] = []
    for block in getattr(result, "content", None) or []:
        text = getattr(block, "text", None)
        if text is not None:
            parts.append(text)
    return ("\n".join(parts) or "(empty result)")[:60000]


class AnthropicBackend:
    """Claude API brain (metered key, subscription OAuth token, or Bedrock/Vertex)."""

    def __init__(self, mcp_tools: list[Any], call_tool: ToolFn, model: str) -> None:
        import anthropic

        self._llm = anthropic.AsyncAnthropic()
        self._tools = _mcp_tools_to_anthropic(mcp_tools)
        self._call = call_tool
        self._model = model
        self._history: list[dict[str, Any]] = []

    async def send(self, user_message: str) -> AsyncIterator[dict[str, Any]]:
        import anthropic

        self._history.append({"role": "user", "content": user_message})
        for _ in range(MAX_TURNS):
            tool_uses: list[Any] = []
            blocks: list[dict[str, Any]] = []
            try:
                async with self._llm.messages.stream(
                    model=self._model,
                    max_tokens=MAX_TOKENS,
                    system=SYSTEM_PROMPT,
                    tools=self._tools,
                    messages=self._history,
                ) as stream:
                    async for text in stream.text_stream:
                        yield {"type": "text", "text": text}
                    final = await stream.get_final_message()
            except anthropic.APIError as exc:
                yield {"type": "error", "message": f"LLM error: {exc}"}
                return

            for block in final.content:
                blocks.append(block.model_dump())
                if block.type == "tool_use":
                    tool_uses.append(block)
            self._history.append({"role": "assistant", "content": blocks})

            if final.stop_reason != "tool_use":
                yield {"type": "done"}
                return

            results: list[dict[str, Any]] = []
            for call in tool_uses:
                yield {"type": "tool_call", "name": call.name, "input": call.input}
                ok, preview = await self._call(call.name, call.input or {})
                yield {"type": "tool_result", "name": call.name, "ok": ok, "preview": preview}
                results.append({"type": "tool_result", "tool_use_id": call.id, "content": preview, "is_error": not ok})
            self._history.append({"role": "user", "content": results})
        yield {"type": "error", "message": f"stopped after {MAX_TURNS} turns"}


class OpenAIBackend:
    """Any OpenAI-compatible server: vLLM / Ollama / LM Studio, on-prem, no per-token bill."""

    def __init__(self, mcp_tools: list[Any], call_tool: ToolFn, model: str) -> None:
        from openai import AsyncOpenAI

        base_url = os.environ.get("KONFAI_STUDIO_LLM_BASE_URL", "http://localhost:8000/v1")
        api_key = os.environ.get("KONFAI_STUDIO_LLM_API_KEY", "EMPTY")
        self._llm = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self._tools = _mcp_tools_to_openai(mcp_tools)
        self._call = call_tool
        self._model = model
        self._history: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    async def send(self, user_message: str) -> AsyncIterator[dict[str, Any]]:
        self._history.append({"role": "user", "content": user_message})
        for _ in range(MAX_TURNS):
            content = ""
            calls: dict[int, dict[str, str]] = {}
            try:
                stream = await self._llm.chat.completions.create(
                    model=self._model,
                    messages=self._history,
                    tools=self._tools or None,
                    stream=True,
                )
                async for chunk in stream:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    if delta.content:
                        content += delta.content
                        yield {"type": "text", "text": delta.content}
                    for tc in delta.tool_calls or []:
                        slot = calls.setdefault(tc.index, {"id": "", "name": "", "args": ""})
                        if tc.id:
                            slot["id"] = tc.id
                        if tc.function and tc.function.name:
                            slot["name"] = tc.function.name
                        if tc.function and tc.function.arguments:
                            slot["args"] += tc.function.arguments
            except Exception as exc:
                yield {"type": "error", "message": f"LLM error: {exc}"}
                return

            if not calls:
                self._history.append({"role": "assistant", "content": content})
                yield {"type": "done"}
                return

            ordered = [calls[i] for i in sorted(calls)]
            self._history.append(
                {
                    "role": "assistant",
                    "content": content or None,
                    "tool_calls": [
                        {
                            "id": c["id"] or f"call_{i}",
                            "type": "function",
                            "function": {"name": c["name"], "arguments": c["args"] or "{}"},
                        }
                        for i, c in enumerate(ordered)
                    ],
                }
            )
            for i, c in enumerate(ordered):
                try:
                    args = json.loads(c["args"] or "{}")
                except json.JSONDecodeError:
                    args = {}
                yield {"type": "tool_call", "name": c["name"], "input": args}
                ok, preview = await self._call(c["name"], args)
                yield {"type": "tool_result", "name": c["name"], "ok": ok, "preview": preview}
                self._history.append({"role": "tool", "tool_call_id": c["id"] or f"call_{i}", "content": preview})
        yield {"type": "error", "message": f"stopped after {MAX_TURNS} turns"}


class StudioAgent:
    """Holds one MCP session and delegates the chat/tool loop to the chosen LLM backend."""

    def __init__(
        self,
        session: str = "default",
        mcp_command: str = "konfai-mcp",
        mcp_args: list[str] | None = None,
        brain: str | None = None,
        model: str | None = None,
        history_file: Path | None = None,
    ) -> None:
        from fastmcp.client.transports import StdioTransport

        env = {**os.environ, "KONFAI_MCP_TRANSPORT": "stdio", "KONFAI_MCP_SESSION": session}
        command = _resolve_mcp_command(mcp_command)
        self._client = Client(StdioTransport(command, mcp_args or [], env=env, keep_alive=True))
        self._brain = brain
        self._model = model
        self._history_file = history_file  # persist the transcript so a restart resumes it
        self._backend: AnthropicBackend | OpenAIBackend | None = None

    async def _call_tool(self, name: str, args: dict[str, Any]) -> tuple[bool, str]:
        try:
            result = await self._client.call_tool(name, args or {})
            return (not getattr(result, "is_error", False), _tool_result_text(result))
        except Exception as exc:
            return (False, f"Tool call failed: {exc}")

    async def __aenter__(self) -> StudioAgent:
        await self._client.__aenter__()
        mcp_tools = await self._client.list_tools()
        backend = (self._brain or os.environ.get("KONFAI_STUDIO_LLM", "anthropic")).lower()
        model = self._model or os.environ.get("KONFAI_STUDIO_MODEL", DEFAULT_MODEL)
        if backend in {"openai", "local", "vllm", "ollama"}:
            self._backend = OpenAIBackend(mcp_tools, self._call_tool, model)
        else:
            self._backend = AnthropicBackend(mcp_tools, self._call_tool, model)
        if self._history_file and self._history_file.is_file():
            try:
                saved = json.loads(self._history_file.read_text(encoding="utf-8"))
                if isinstance(saved, list) and saved:
                    self._backend._history = saved  # resume the prior conversation
            except (OSError, ValueError):
                pass
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self._client.__aexit__(*exc)

    async def send(self, user_message: str) -> AsyncIterator[dict[str, Any]]:
        assert self._backend is not None, "agent must be entered before use"
        async for event in with_volume_events(self._backend.send(user_message)):
            yield event
        if self._history_file:  # persist after each turn so a restart continues the conversation
            try:
                self._history_file.parent.mkdir(parents=True, exist_ok=True)
                self._history_file.write_text(json.dumps(self._backend._history, default=str), encoding="utf-8")
            except OSError:
                pass


def _short_tool_name(name: str) -> str:
    """Strip the ``mcp__konfai__`` prefix Claude Code gives MCP tools, for display."""
    return name.split("__")[-1] if name.startswith("mcp__") else name


def _stringify_content(content: Any) -> str:
    """Flatten a Claude Code content value (str or list of blocks) into text."""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content or []:
        text = getattr(block, "text", None)
        if text is not None:
            parts.append(text)
        elif isinstance(block, dict) and "text" in block:
            parts.append(str(block["text"]))
    return "\n".join(parts) or "(empty result)"


def _workspace_cwd() -> str:
    root = os.environ.get("KONFAI_MCP_WORKSPACES_ROOT") or os.path.expanduser("~/KonfAI_Workspaces")
    os.makedirs(root, exist_ok=True)
    return root


class ClaudeCodeAgent:
    """Brain = the Claude Agent SDK, authenticated by the user's Claude Code subscription.

    Self-contained: the SDK spawns konfai-mcp itself and drives the tool loop; no API key
    and no per-token bill. Only konfai-mcp tools are allowed (built-in mutators are blocked),
    and the imaging data never reaches the model (tools return text).
    """

    def __init__(
        self,
        session: str = "default",
        mcp_command: str = "konfai-mcp",
        mcp_args: list[str] | None = None,
        model: str | None = None,
        resume: str | None = None,
        on_session_id: Callable[[str], None] | None = None,
    ) -> None:
        from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

        # KONFAI_MCP_SESSION isolates each task's konfai-mcp workspace (jobs, configs, runs).
        env = {**os.environ, "KONFAI_MCP_TRANSPORT": "stdio", "KONFAI_MCP_SESSION": session}
        command = _resolve_mcp_command(mcp_command)
        options = ClaudeAgentOptions(
            mcp_servers={"konfai": {"type": "stdio", "command": command, "args": mcp_args or [], "env": env}},
            strict_mcp_config=True,
            permission_mode="bypassPermissions",
            disallowed_tools=["Bash", "Write", "Edit", "NotebookEdit"],
            system_prompt=SYSTEM_PROMPT,
            setting_sources=[],
            model=model or os.environ.get("KONFAI_STUDIO_MODEL") or None,
            cwd=_workspace_cwd(),
            resume=resume,  # continue the prior transcript after a server restart
        )
        self._client = ClaudeSDKClient(options=options)
        self._on_session_id = on_session_id
        self._names: dict[str, str] = {}

    async def __aenter__(self) -> ClaudeCodeAgent:
        await self._client.connect()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self._client.disconnect()

    async def send(self, user_message: str) -> AsyncIterator[dict[str, Any]]:
        async for event in with_volume_events(self._emit(user_message)):
            yield event

    async def _emit(self, user_message: str) -> AsyncIterator[dict[str, Any]]:
        from claude_agent_sdk import (
            AssistantMessage,
            ResultMessage,
            TextBlock,
            ToolResultBlock,
            ToolUseBlock,
            UserMessage,
        )

        await self._client.query(user_message)
        async for message in self._client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        if block.text:
                            yield {"type": "text", "text": block.text}
                    elif isinstance(block, ToolUseBlock):
                        self._names[block.id] = block.name
                        yield {"type": "tool_call", "name": _short_tool_name(block.name), "input": block.input}
            elif isinstance(message, UserMessage):
                content = message.content
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, ToolResultBlock):
                            name = _short_tool_name(self._names.get(block.tool_use_id, "tool"))
                            ok = not bool(block.is_error)
                            preview = _stringify_content(block.content)
                            yield {"type": "tool_result", "name": name, "ok": ok, "preview": preview}
            elif isinstance(message, ResultMessage):
                if self._on_session_id and getattr(message, "session_id", None):
                    self._on_session_id(message.session_id)  # persist so a restart can resume it
                if message.is_error:
                    detail = message.result or message.errors or message.api_error_status or "error"
                    yield {"type": "error", "message": str(detail)}
                yield {"type": "done"}
                return


_TITLE_PROMPT = (
    "Give a concise 2-5 word title in Title Case (no quotes, no punctuation, no trailing period) "
    "for a medical-imaging experiment a user described as:\n\n{text}\n\nReply with ONLY the title."
)


def _clean_title(raw: str) -> str:
    lines = [line.strip() for line in (raw or "").splitlines() if line.strip()]
    title = lines[0].strip("\"'").rstrip(".").strip() if lines else ""
    return re.sub(r"\s+", " ", title)[:48]


def _heuristic_title(text: str) -> str:
    """A decent title straight from the user's words, used when the LLM titling is unavailable."""
    words = re.sub(r"\s+", " ", text.strip()).split(" ")
    return _clean_title(" ".join(words[:6])) or "New experiment"


async def suggest_title(text: str, brain: str | None = None) -> str:
    """A short experiment title for the user's first message, named by the LLM (one-shot, isolated
    from the task conversation) with a heuristic fallback so it never fails the turn."""
    backend = (brain or os.environ.get("KONFAI_STUDIO_LLM", "claude-code")).lower()
    prompt = _TITLE_PROMPT.format(text=text[:800])
    try:
        if backend in {"claude-code", "claude", "subscription", "agent-sdk"}:
            from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, TextBlock, query

            parts: list[str] = []
            async for message in query(
                prompt=prompt,
                options=ClaudeAgentOptions(setting_sources=[], permission_mode="bypassPermissions"),
            ):
                if isinstance(message, AssistantMessage):
                    parts += [b.text for b in message.content if isinstance(b, TextBlock) and b.text]
            if title := _clean_title("".join(parts)):
                return title
    except Exception:
        pass
    return _heuristic_title(text)


_NEXT_PROMPTS_PROMPT = (
    "Suggest the user's next move in KonfAI, a medical-imaging deep-learning assistant driven from a chat "
    "box. Workflow: inspect dataset -> author/adapt a config or pick a published app -> validate -> train "
    "(or import_app + run_prediction to use an app as-is / run_resume weights_only to fine-tune) -> predict "
    "-> evaluate -> compare / leaderboard.\n\n"
    "User said:\n{user}\n\nAssistant replied:\n{response}\n\n"
    "Tools available to call next: {actions}\n\n"
    "Propose the 3 most likely next things THIS user would want -- each a real forward step from the current "
    "state. Never suggest a step whose prerequisite is missing (no evaluate before a prediction exists), "
    "never repeat what was just done, and prefer steps grounded in the tools above when they fit. For each: "
    "a short button label (2-4 words, Title Case, no period) and the full prompt the user would send.\n\n"
    'Reply with ONLY a JSON array of exactly 3 objects {{"label": "<2-4 words>", "prompt": "<full prompt>"}}. '
    "No prose, no code fence."
)


def _parse_next_prompts(raw: str) -> list[dict[str, str]]:
    """The ``[{label, prompt}]`` array from the LLM reply, tolerating a stray code fence or surrounding prose."""
    text = (raw or "").strip()
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end <= start:
        return []
    try:
        items = json.loads(text[start : end + 1])
    except (TypeError, ValueError):
        return []
    out: list[dict[str, str]] = []
    for item in items if isinstance(items, list) else []:
        label = item.get("label") if isinstance(item, dict) else None
        prompt = item.get("prompt") if isinstance(item, dict) else None
        if isinstance(label, str) and isinstance(prompt, str) and label.strip() and prompt.strip():
            out.append({"label": label.strip()[:40], "prompt": prompt.strip()[:400]})
    return out[:3]


async def suggest_next_prompts(
    user_msg: str, response: str, actions: list[str] | None = None, brain: str | None = None
) -> list[dict[str, str]]:
    """The 3 next prompts (short button label + full prompt) from a one-shot LLM call. ``actions`` are the
    turn's tool next_actions. Empty on any failure; the UI falls back to its own chips."""
    backend = (brain or os.environ.get("KONFAI_STUDIO_LLM", "claude-code")).lower()
    if backend not in {"claude-code", "claude", "subscription", "agent-sdk"}:
        return []  # API brains: skip the extra call; the UI falls back on its own
    prompt = _NEXT_PROMPTS_PROMPT.format(
        user=user_msg[:600], response=(response or "")[:1500], actions=", ".join(actions or []) or "(none)"
    )
    try:
        from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, TextBlock, query

        parts: list[str] = []
        async for message in query(
            prompt=prompt,
            options=ClaudeAgentOptions(setting_sources=[], permission_mode="bypassPermissions"),
        ):
            if isinstance(message, AssistantMessage):
                parts += [b.text for b in message.content if isinstance(b, TextBlock) and b.text]
        return _parse_next_prompts("".join(parts))
    except Exception:
        return []


async def call_mcp_tool(session: str, tool: str, args: dict[str, Any] | None = None) -> tuple[bool, str]:
    """One-shot konfai-mcp tool call for a session — deterministic, no LLM in the loop.

    Used for plain actions (bundle, export) the UI triggers by button rather than by prompting the
    model. Spawns a short-lived stdio client against the same konfai-mcp session workspace.
    """
    from fastmcp.client.transports import StdioTransport

    env = {**os.environ, "KONFAI_MCP_TRANSPORT": "stdio", "KONFAI_MCP_SESSION": session}
    client = Client(StdioTransport(_resolve_mcp_command("konfai-mcp"), [], env=env, keep_alive=False))
    try:
        async with client:
            result = await client.call_tool(tool, args or {})
        return (not getattr(result, "is_error", False)), _tool_result_text(result)
    except Exception as exc:  # a tool error (e.g. nothing to package) is a result, not a crash
        return False, str(exc)


def make_agent(
    session: str = "default",
    brain: str | None = None,
    model: str | None = None,
    resume: str | None = None,
    on_session_id: Callable[[str], None] | None = None,
    history_file: Path | None = None,
) -> StudioAgent | ClaudeCodeAgent:
    """Pick the brain (``brain`` argument, else ``KONFAI_STUDIO_LLM``; default: the Claude Code
    subscription) and optionally pin its ``model``. Each call is one isolated task: its konfai-mcp
    workspace is the given ``session``. Conversation continuity across a restart is brain-agnostic:
    the Claude Code brain resumes its SDK transcript (``resume``/``on_session_id``); the API brains
    reload their history (``history_file``).
    """
    backend = (brain or os.environ.get("KONFAI_STUDIO_LLM", "claude-code")).lower()
    if backend in {"claude-code", "claude", "subscription", "agent-sdk"}:
        return ClaudeCodeAgent(session, model=model, resume=resume, on_session_id=on_session_id)
    return StudioAgent(session, brain=backend, model=model, history_file=history_file)
