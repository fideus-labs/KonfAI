# SPDX-License-Identifier: Apache-2.0
"""Generate the agent-facing tool reference from the live MCP registry.

The generated markdown is the single source of truth for out-of-band docs (the
konfai-experiments skill reference); regenerating it after any tool change keeps
those docs from drifting the way hand-written copies did.

Usage:
    python konfai-mcp/scripts/generate_tool_reference.py \
        > .claude/skills/konfai-experiments/references/tool-reference.md
"""

from __future__ import annotations

import asyncio


async def render() -> str:
    from konfai_mcp.server import mcp

    tools = sorted(await mcp.list_tools(), key=lambda tool: tool.name)
    prompts = sorted(await mcp.list_prompts(), key=lambda prompt: prompt.name)
    resources = list(await mcp.list_resources()) if hasattr(mcp, "list_resources") else []
    templates = list(await mcp.list_resource_templates()) if hasattr(mcp, "list_resource_templates") else []

    lines = [
        "# KonfAI MCP — Tool Reference",
        "",
        "> GENERATED from the registry by `konfai-mcp/scripts/generate_tool_reference.py` — do not edit by hand.",
        "",
        f"{len(tools)} tools, {len(prompts)} prompts, {len(resources) + len(templates)} resources. "
        "The live equivalent is the `guide://tool-index` resource.",
        "",
        "## Tools",
    ]
    for tool in tools:
        lines += ["", f"### `{tool.name}`", "", (tool.description or "").strip()]
    lines += ["", "## Prompts"]
    for prompt in prompts:
        lines += ["", f"### `{prompt.name}`", "", (prompt.description or "").strip()]
    if resources or templates:
        lines += ["", "## Resources", ""]
        for resource in resources:
            lines.append(f"- `{resource.uri}` — {(resource.description or '').strip()}")
        for template in templates:
            uri = getattr(template, "uriTemplate", None) or getattr(template, "uri_template", "")
            lines.append(f"- `{uri}` — {(template.description or '').strip()}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    print(asyncio.run(render()), end="")
