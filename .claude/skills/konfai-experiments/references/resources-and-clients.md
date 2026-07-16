# Resources, environment, and client wiring

## MCP resources (read-only state the agent can pull)

Prefer these over re-deriving state; they are the machine-readable checkpoints.

**Server / guide / docs**
- `server://info`, `server://capabilities`
- `guide://tool-index`, `guide://config-design`
- `docs://index`, `docs://patching`, `docs://modeling`, `docs://configuration`, `docs://dataset-mapping`, `docs://examples`

**Templates**
- `templates://list`, `template://{name}/summary`

**Session (current workspace)**
- `sessions://list`
- `session://current/summary`
- `session://current/config/{workflow}` — `workflow` in `train` | `prediction` | `evaluation`
- `session://current/log`
- `session://current/metrics`

**Jobs**
- `job://{job_id}/status`, `job://{job_id}/log`, `job://{job_id}/manifest`

The job **manifest** holds the config snapshot captured at launch — the reproducible record
of that run, independent of any later config edit.

## Environment variables

| Variable | Purpose |
|---|---|
| `KONFAI_MCP_WORKSPACES_ROOT` | Root for session workspaces (default `~/KonfAI_Workspaces`) |
| `KONFAI_MCP_SESSION` | Default session name for the server process |
| `KONFAI_MCP_LOG_TAIL_LINES` | Default max log lines returned by log-tail helpers |
| `KONFAI_MCP_VALIDATE_ROOT` | Scratch workspace root used by side-effect-free validation |
| `KONFAI_MCP_TRANSPORT` | `stdio` (default) \| `sse` \| `streamable-http` |
| `KONFAI_MCP_HOST` / `KONFAI_MCP_PORT` / `KONFAI_MCP_PATH` | Bind settings for HTTP transports |
| `KONFAI_MCP_LOG_LEVEL` | FastMCP/Uvicorn log level |
| `KONFAI_MCP_BEARER_TOKEN` | Optional bearer token; when set, protects `sse` / `streamable-http` (stdio ignores it) |

## Wiring the server into an MCP client

The intended entrypoint is the installed `konfai-mcp` command, not an ad-hoc wrapper.

### Claude Code (`.mcp.json` at the repo root, or `claude mcp add`)

```json
{
  "mcpServers": {
    "konfai": {
      "command": "/path/to/venv/bin/konfai-mcp",
      "env": {
        "KONFAI_MCP_WORKSPACES_ROOT": "/path/to/KonfAI/mcp-workspace",
        "KONFAI_MCP_LOG_TAIL_LINES": "400"
      }
    }
  }
}
```

Once connected, the tools appear namespaced as `mcp__konfai__<tool>` — e.g.
`mcp__konfai__inspect_dataset`, `mcp__konfai__run_train`. If they are absent, the server is
not wired in (run `scripts/check_setup.py` to tell "not installed" from "not wired").

### Codex (`config.toml`)

```toml
[mcp_servers.konfai]
command = "/path/to/venv/bin/konfai-mcp"
cwd = "/path/to/KonfAI/konfai-mcp"
startup_timeout_sec = 20
tool_timeout_sec = 3600

[mcp_servers.konfai.env]
KONFAI_MCP_WORKSPACES_ROOT = "/path/to/KonfAI/mcp-workspace"
KONFAI_MCP_LOG_TAIL_LINES = "400"
```

Set `tool_timeout_sec` generously (training is long) and rely on `wait_for_job` without a
`timeout_s` for multi-hour runs.

## Transport note (security)

`stdio` (the default, used by local Claude Code / Codex) needs no auth. For `sse` and
`streamable-http` the bearer token is **optional and enforced only when set**: with
`KONFAI_MCP_BEARER_TOKEN` set, unauthenticated requests get a `401` with a `Bearer`
challenge; **with it unset, the HTTP transport starts fully unauthenticated (no `401`)** —
there is no guard rejecting a tokenless HTTP start. Since this server executes real compute,
**always set a token before exposing `sse` / `streamable-http`.**

A bearer token is only as safe as the channel: over plain HTTP it travels in clear and is
trivially sniffable, so the token alone protects nothing on the wire. **Keep `sse` /
`streamable-http` bound to loopback** (reach it over an SSH tunnel or a reverse proxy) **or
terminate TLS in front of the server** — never expose a tokened-but-unencrypted endpoint
beyond `localhost`.
