# SPDX-License-Identifier: Apache-2.0
"""The Studio agent: an LLM driving the konfai-mcp tools, streamed as UI events.

The LLM backend is pluggable (``KONFAI_STUDIO_LLM``): ``claude-code`` (default; the
Claude Code subscription via the Claude Agent SDK, no API key), ``anthropic`` (the Claude
API, ``ANTHROPIC_API_KEY``), or ``openai`` for any OpenAI-compatible server (vLLM / Ollama
/ LM Studio, ``KONFAI_STUDIO_LLM_BASE_URL``). The MCP side never depends on which brain is
used, and the imaging data never reaches the LLM: tools run locally on the compute node
and return text.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import sys
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any

from fastmcp import Client

from .workflow import MAX_MOVES


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


def _require_claude_code() -> None:
    """The Claude Code brain needs its SDK *and* the ``claude`` CLI (not a pip package). Missing either is a
    setup problem, so report the fix instead of letting an ImportError or a spawn failure reach the chat."""
    switch = (
        "or switch brain: KONFAI_STUDIO_LLM=anthropic (with ANTHROPIC_API_KEY), or =openai with "
        "KONFAI_STUDIO_LLM_BASE_URL pointing at a local vLLM/Ollama/LM Studio server"
    )
    try:
        import claude_agent_sdk  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            f"The Claude Code brain needs claude-agent-sdk (pip install konfai-studio), {switch}."
        ) from exc
    if shutil.which("claude") is None:
        raise RuntimeError(f"The Claude Code brain needs the 'claude' CLI on PATH — install Claude Code, {switch}.")


DEFAULT_MODEL = os.environ.get("KONFAI_STUDIO_MODEL", "claude-opus-4-8")
MAX_TOKENS = int(os.environ.get("KONFAI_STUDIO_MAX_TOKENS", "16000"))
MAX_TURNS = int(os.environ.get("KONFAI_STUDIO_MAX_TURNS", "30"))

# Standing rules only: who you are, and what holds on every turn. What the experiment owes RIGHT NOW is
# sent per turn as [state]/[now], derived from the workspace by konfai_mcp.experiment_state -- so no
# stage's instruction can arrive at the wrong stage, and none of it is paid for twice.
SYSTEM_PROMPT = (
    "You drive KonfAI -- config-driven deep learning for medical imaging -- through its MCP tools, for a "
    "clinician-researcher who knows their data, not the framework.\n\n"
    "Act, do not offer. When the next step only reads, take it and report what you found. Ask only for "
    "what you cannot derive, one question at a time. Never state a tool result you did not get.\n\n"
    "A turn may open with three lines:\n"
    "  [state] where the experiment stands, read from its workspace. Authoritative: never ask for what is "
    "already there, never redo a step it shows as done.\n"
    "  [now] the step the experiment is waiting on. Do it -- unless the user's message asks for something "
    "else, which always wins.\n"
    "  [constraint] a compute restriction to respect exactly.\n\n"
    "NEVER start training, fine-tuning, prediction or evaluation without asking first. Say in one line "
    "what it will cost -- how many cases, on which device, roughly how long -- and wait for a yes. The "
    "only exception is relaunching a run you just corrected after a failure.\n\n"
    "A run is a result only when its job says so. After any run_* or fine_tune_app, wait for the job to "
    "reach a terminal state, then open what it produced and report THAT. A launch that returned cleanly is "
    "not a success and must never be presented as one.\n\n"
    "When a run fails: the cause in one line, never a traceback, then the correction. A correction may be "
    "relaunched straight away; anything else that costs GPU time is asked first. Two failed attempts is the "
    "limit -- past it, lay out the options and ask. Never settle a scientific choice yourself (loss, "
    "architecture, split, label mapping, which data is the right one): those are the user's.\n\n"
    "A job whose status is 'killed' was stopped ON PURPOSE -- usually the user pressing Stop in the panel "
    "beside you ('cancel_requested': true marks it). It is not a failure, not an external anomaly, and "
    "nothing to investigate or apologise for: say in one line where the run stood (epoch/iteration, last "
    "loss), then offer to resume or relaunch. A deliberate stop never counts toward the two-attempt limit, "
    "and you never invent a watchdog, supervisor or OOM to explain it.\n\n"
    "Reply in the user's own language -- French to French, English to English -- and keep tool names, "
    "config keys and paths verbatim whatever the language.\n\n"
    "Reply in a few sentences: what you did, what it showed, what is next. No preamble, no lists of what "
    "you could do, no restating [state], no thinking out loud, no log dumps -- the failing lines only.\n\n"
    "WHENEVER you describe a dataset, use exactly this shape so two datasets can be compared without "
    "re-reading either:\n"
    "one line -- <N> cases | <layout> | <ext> | spacing <x>x<y>x<z> mm | shape <X>x<Y>x<Z>\n"
    "then a markdown table, one row per group, these columns and no others:\n"
    "| Group | Cases | Range / classes | Role |\n"
    "Role is one short clause: what it is, and whether it can serve as input or as target.\n"
    "then, each on its own line and only when it applies:\n"
    "**Incomplete** — <GROUP> missing in <n>/<N> cases (<a few case ids>)\n"
    "**dataset_entry** — `<value>`\n"
    "Every number comes from the tool, never from memory; group names verbatim. When groups differ in "
    "geometry, say so in their Role rather than averaging them into the header line.\n\n"
    "END EVERY REPLY with the next moves, after a line containing only <<NEXT>>:\n"
    "<<NEXT>>\n"
    "Short Label :: the whole message the user would send to take that step\n"
    "Short Label :: ...\n"
    "As many as genuinely apply, one to ten -- not a fixed number, and never one invented to reach a "
    "count. If your reply asks a question or offers a choice, the moves ARE its answers: one per option "
    "you raised, all of them, in the user's voice ('Do MR to CT synthesis'), and nothing else -- "
    "answering is the only thing they owe you. Otherwise they are the next steps, best first.\n"
    "The label is 2-4 words. The prompt is written as the USER, names its subject (the dataset path, the "
    "app, the run) and says what to do; when it starts a run it also says to wait for the job and open "
    "what it produced. Never a vague 'Continue' or 'Run it'. This block is stripped before the user sees "
    "it: it becomes the buttons under your reply.\n\n"
    "A NiiVue viewer sits beside the chat: any .nii/.nii.gz/.mha/.mhd/.nrrd path you surface loads there by "
    "itself. To compare two volumes -- a synthetic CT against the real one, a prediction against its label "
    "-- surface BOTH paths in the same step: exactly two opens them side by side, crosshairs linked. It is "
    "never unavailable, and you must never invent an image.\n\n"
    "Given a paper, extract task, architecture, losses and augmentations, reproduce it as a config, "
    "validate it -- and stop there until the user confirms training."
)

# A tool executor: given (name, arguments) run the MCP tool, return (ok, text_preview).
ToolFn = Callable[[str, dict[str, Any]], Awaitable[tuple[bool, str]]]

# Absolute NIfTI / MHA volume paths NAMED by the assistant -> auto-load in NiiVue.
_VOLUME_RE = re.compile(r"(/[^\s\"'`|,)\]]+\.(?:nii\.gz|nii|mha|mhd|nrrd))")


def _detect_volumes(text: str) -> list[str]:
    """The existing volume paths in a chunk of text, in order and deduped, so the chat can drive the viewer."""
    found: list[str] = []
    for match in _VOLUME_RE.finditer(text or ""):
        path = match.group(1)
        if path not in found and os.path.isfile(path):
            found.append(path)
            if len(found) > 3:  # past a comparison it is a listing; the exact count no longer matters
                break
    return found


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

# The assistant ends its reply with its own next moves behind this marker. Asking a second model to write
# them cost 40s per turn and knew less than the one that had just done the work.
_NEXT_MARKER = "<<NEXT>>"


class _NextMoves:
    """Splits the reply stream at the marker: the prose goes to the user, the block behind it becomes
    the buttons.

    Text arrives in chunks that can cut the marker in half, so the last few characters are held back
    until they cannot start one — the user never sees the marker appear and vanish.
    """

    def __init__(self) -> None:
        self._pending = ""
        self._block = ""
        self._split = False

    @staticmethod
    def _partial_marker(text: str) -> int:
        """How many trailing characters could still grow into the marker — 0 when none can. Holding back
        a fixed count instead would cut the visible text mid-word every time a tool call interrupts it."""
        for size in range(min(len(text), len(_NEXT_MARKER) - 1), 0, -1):
            if _NEXT_MARKER.startswith(text[-size:]):
                return size
        return 0

    def feed(self, chunk: str) -> str:
        """Take a chunk of the reply; return the part that is safe to show."""
        if self._split:
            self._block += chunk
            return ""
        self._pending += chunk
        cut = self._pending.find(_NEXT_MARKER)
        if cut >= 0:
            visible, self._block = self._pending[:cut], self._pending[cut + len(_NEXT_MARKER) :]
            self._pending, self._split = "", True
            return visible
        keep = self._partial_marker(self._pending)
        visible, self._pending = self._pending[: len(self._pending) - keep], self._pending[len(self._pending) - keep :]
        return visible

    def flush(self) -> str:
        """Whatever was held back, once the reply is over and no marker came."""
        visible, self._pending = self._pending, ""
        return visible

    def moves(self) -> list[dict[str, str]]:
        """The parsed moves — ``Label :: prompt`` per line. A malformed line is dropped, not guessed at."""
        out: list[dict[str, str]] = []
        for line in self._block.splitlines():
            label, _, prompt = line.partition("::")
            label, prompt = label.strip().lstrip("-*• ").strip(), prompt.strip()
            if label and prompt and len(label) <= 40:
                out.append({"label": label[:40], "prompt": prompt[:400]})
        return out[:MAX_MOVES]


async def with_volume_events(events: AsyncIterator[dict[str, Any]]) -> AsyncIterator[dict[str, Any]]:
    """Pass an agent's event stream through, deriving the UI's out-of-band signals in ONE place: a
    ``volume`` event (what to show in the viewer), a ``next_actions`` event (the tool's own advice), and
    the reply's trailing next-moves block, which is stripped from the text and emitted as
    ``next_prompts``. Backends stay free of any UI concern; the preview is trimmed here.

    The viewer follows the volumes the assistant NAMES — in a tool's arguments, or in its own prose, which
    is what the system prompt promises the user. A tool *result* is not a choice: an inventory names one
    file per group to prove the group exists, and reading the viewer off those opened whichever group
    sorted first instead of the images the assistant had just previewed."""
    next_moves = _NextMoves()
    named: list[str] = []  # what the assistant pointed at, in order — the last two are the comparison
    shown: tuple[str, str | None] | None = None  # the pair on screen, so the same one is not re-sent
    async for event in events:
        volumes: list[str] = []
        actions: list[str] = []
        if event["type"] == "text":
            visible = next_moves.feed(event.get("text") or "")
            if not visible:
                continue  # held back: a partial marker, or the moves block itself
            event["text"] = visible
            volumes = _detect_volumes(visible)
        elif event["type"] == "tool_call":
            volumes = _detect_volumes(json.dumps(event.get("input"), default=str))
        elif event["type"] == "tool_result":
            full = event.get("preview", "")
            actions = _next_actions(full)
            event["preview"] = full[:_PREVIEW_LIMIT]  # trim for the UI, after parsing the full text
        elif event["type"] in {"done", "error"}:
            if tail := next_moves.flush():
                yield {"type": "text", "text": tail}
            if moves := next_moves.moves():
                yield {"type": "next_prompts", "prompts": moves}
        yield event
        # The pane follows what the assistant last pointed at: the last two volumes it named are the
        # comparison, whether they arrive together or one step apart -- a reply naming the moved image
        # beside the reference is streamed in chunks, so "the same step" is not something this side can
        # recognise. Naming is not deduplicated over the turn: a run names its inputs in its own arguments
        # and then cites one of them again as half of a comparison, and dropping the repeat left the pane
        # pairing the wrong two. A step listing more than two is an inventory, not a choice.
        if volumes:
            if len(volumes) > 2:
                if shown is None:
                    shown = (volumes[0], None)
                    yield {"type": "volume", "path": shown[0], "compare": None}
            else:
                for path in volumes:
                    if not named or named[-1] != path:
                        named.append(path)
                pair = (named[-2], named[-1]) if len(named) > 1 else (named[-1], None)
                if pair != shown:
                    shown = pair
                    yield {"type": "volume", "path": pair[0], "compare": pair[1]}
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


# A tool result headed for a model is capped: one oversized payload would crowd out the conversation.
# A result Studio parses itself is NOT — a JSON document cut at any length stops being JSON, and the
# caller cannot tell a truncated payload from an empty one. The leaderboard reached 64 328 characters
# with 65 metrics and came back as "no evaluations yet" over two runs that had been scored.
_MODEL_RESULT_LIMIT = 60000


def _tool_result_text(result: Any, limit: int | None = _MODEL_RESULT_LIMIT) -> str:
    def capped(text: str) -> str:
        return text if limit is None else text[:limit]

    if getattr(result, "data", None) is not None:
        try:
            return capped(json.dumps(result.data, default=str))
        except (TypeError, ValueError):
            pass
    parts: list[str] = []
    for block in getattr(result, "content", None) or []:
        text = getattr(block, "text", None)
        if text is not None:
            parts.append(text)
    return capped("\n".join(parts) or "(empty result)")


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
        backend = (self._brain or os.environ.get("KONFAI_STUDIO_LLM") or "anthropic").lower()
        model = self._model or os.environ.get("KONFAI_STUDIO_MODEL") or DEFAULT_MODEL
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
    """Flatten a Claude Code content value (str or list of blocks) into text.

    A block that is not text is named rather than dropped: ``preview_volume`` answers with an image, and
    silently skipping it made a working call read as "(empty result)" in the chat.
    """
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content or []:
        text = getattr(block, "text", None)
        if text is None and isinstance(block, dict):
            text = block.get("text")
        if text is not None:
            parts.append(str(text))
            continue
        kind = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
        parts.append(f"({kind or type(block).__name__.replace('Content', '').lower()} returned)")
    return "\n".join(parts) or "(no content)"


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
            # Studio's safety posture is that the assistant drives KonfAI through its MCP tools and does
            # not run shell or write files. Denying Bash alone does not achieve that: told no, it spawns
            # a subagent that has a shell and delegates the command (observed, twice in one turn). Agent
            # and Task have to go with it, or the restriction is decorative.
            #
            # ToolSearch pages in tools held back from the prompt; konfai-mcp's are all present, so every
            # search returns nothing — enabled, the model spends its turns searching instead of calling
            # the tools it already has.
            disallowed_tools=["Bash", "Write", "Edit", "NotebookEdit", "ToolSearch", "Agent", "Task"],
            system_prompt=SYSTEM_PROMPT,
            setting_sources=[],
            model=model or os.environ.get("KONFAI_STUDIO_MODEL") or None,
            cwd=_workspace_cwd(),
            resume=resume,  # continue the prior transcript after a server restart
        )
        self._client = ClaudeSDKClient(options=options)
        self._on_session_id = on_session_id
        self._names: dict[str, str] = {}
        self._interrupted = False  # this turn was cut short on purpose, so its error is not one

    # The stdio MCP server takes about two seconds to hand over its tool list, and `connect()` returns
    # before that. A turn started in the window sees no konfai tools at all: the first call comes back
    # "no such tool", whatever name it used, and the assistant retries — one wasted call, in the open,
    # on the very first thing the user asks.
    _MCP_READY_TIMEOUT = 15.0
    _MCP_POLL = 0.25

    async def __aenter__(self) -> ClaudeCodeAgent:
        await self._client.connect()
        await self._await_mcp_tools()
        return self

    async def _await_mcp_tools(self) -> None:
        """Block until the MCP servers stop reporting `pending`, or the timeout runs out.

        A timeout is not fatal: the tools may still arrive, and refusing to start the session over a slow
        handshake would be worse than the retry it is meant to save.
        """

        deadline = asyncio.get_running_loop().time() + self._MCP_READY_TIMEOUT
        while asyncio.get_running_loop().time() < deadline:
            try:
                status = await self._client.get_mcp_status()
            except Exception:
                return  # no status channel on this SDK build; the retry path still covers it
            servers = (status or {}).get("mcpServers") or []
            if servers and all(s.get("status") != "pending" for s in servers):
                return
            await asyncio.sleep(self._MCP_POLL)

    async def __aexit__(self, *exc: Any) -> None:
        await self._client.disconnect()

    async def send(self, user_message: str) -> AsyncIterator[dict[str, Any]]:
        # An interrupt that lands after the turn already ended leaves a stale flag; cleared here, it
        # cannot swallow the next turn's genuine error.
        self._interrupted = False
        async for event in with_volume_events(self._emit(user_message)):
            yield event

    async def interrupt(self) -> bool:
        """Cut the turn short so the user's correction is acted on now, not after it finishes.

        Verified against the SDK from both states — mid-text and with a tool in flight: the turn ends
        flagged as an error, and the *next* turn on the same session works. So the flag below only stops
        that expected error from reading as a failure. A job the tool already launched keeps running;
        interrupting the agent is not cancelling the work it started.
        """
        try:
            self._interrupted = True
            await self._client.interrupt()
        except Exception:
            self._interrupted = False
            return False
        return True

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
                # A turn the user cut short comes back flagged as an error, with an SDK diagnostic for a
                # message. It is not a failure and must not read as one: the next turn works (verified
                # against the SDK, interrupted both mid-text and mid-tool), and the correction they typed
                # is already on its way.
                interrupted, self._interrupted = self._interrupted, False
                if message.is_error and not interrupted:
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


# Naming an experiment is a small formatting job the user waits on with the answer already on screen:
# it runs on the fast model, not the one driving the tools.
SIDE_MODEL = os.environ.get("KONFAI_STUDIO_SIDE_MODEL", "claude-haiku-4-5")


async def _one_shot_claude(prompt: str) -> str:
    """One isolated Claude Code query (no MCP tools, no task history) — its joined text, '' on failure."""
    try:
        from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, TextBlock, query

        parts: list[str] = []
        async for message in query(
            prompt=prompt,
            options=ClaudeAgentOptions(
                setting_sources=[], permission_mode="bypassPermissions", model=SIDE_MODEL or None
            ),
        ):
            if isinstance(message, AssistantMessage):
                parts += [b.text for b in message.content if isinstance(b, TextBlock) and b.text]
        return "".join(parts)
    except Exception:
        return ""


async def suggest_title(text: str, brain: str | None = None) -> str:
    """A short experiment title for the user's first message, named by the LLM (one-shot, isolated
    from the task conversation) with a heuristic fallback so it never fails the turn."""
    backend = (brain or os.environ.get("KONFAI_STUDIO_LLM") or "claude-code").lower()
    if backend in {"claude-code", "claude", "subscription", "agent-sdk"}:
        if title := _clean_title(await _one_shot_claude(_TITLE_PROMPT.format(text=text[:800]))):
            return title
    return _heuristic_title(text)


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
        return (not getattr(result, "is_error", False)), _tool_result_text(result, limit=None)
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
    backend = (brain or os.environ.get("KONFAI_STUDIO_LLM") or "claude-code").lower()
    if backend in {"claude-code", "claude", "subscription", "agent-sdk"}:
        _require_claude_code()
        return ClaudeCodeAgent(session, model=model, resume=resume, on_session_id=on_session_id)
    return StudioAgent(session, brain=backend, model=model, history_file=history_file)
