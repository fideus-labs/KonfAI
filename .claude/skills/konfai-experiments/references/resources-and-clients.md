# Resources, environment, and client wiring

## MCP resources (read-only state the agent can pull)

Prefer these over re-deriving state; they are the machine-readable checkpoints. The full list,
generated from the server's registry, is the Resources section of
[tool-reference.md](tool-reference.md) (live: the client's `resources/list` and `resources/templates/list`).
The ones a loop reaches for:

- `server://capabilities`: the devices a run can use.
- `session://current/summary`, `session://current/metrics`, `session://current/log`.
- `session://current/config/{workflow}`: `workflow` in `train` | `prediction` | `evaluation` | `transform`.
- `job://{job_id}/status`, `job://{job_id}/log`, `job://{job_id}/manifest`.

The job **manifest** holds the config snapshot captured at launch: the reproducible record
of that run, independent of any later config edit.

## Environment variables

| Variable | Purpose |
|---|---|
| `KONFAI_MCP_WORKSPACES_ROOT` | Root for session workspaces (default `~/KonfAI_Workspaces`) |
| `KONFAI_MCP_SESSION` | Default session name for the server process |
| `KONFAI_MCP_LOG_TAIL_LINES` | Default max log lines returned by log-tail helpers |
| `KONFAI_MCP_APP_CATALOG` | A JSON file of your own app sources, layered over the shipped catalogue |
| `KONFAI_MCP_SUBPROCESS_TIMEOUT` | Seconds a spawn subprocess outside a job (validation, smoke test, plan, signature, app import) may run; default `1800`, `0` waits unbounded |
| `KONFAI_MCP_TRANSPORT` | `stdio` (default) \| `sse` \| `streamable-http` |
| `KONFAI_MCP_HOST` / `KONFAI_MCP_PORT` / `KONFAI_MCP_PATH` | Bind settings for HTTP transports |
| `KONFAI_MCP_STATELESS_HTTP` / `KONFAI_MCP_JSON_RESPONSE` | `1` serves streamable HTTP without session state / with plain JSON bodies |
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

Once connected, the tools appear namespaced as `mcp__konfai__<tool>`: e.g.
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
`streamable-http`, with `KONFAI_MCP_BEARER_TOKEN` (or `--bearer-token`) set, unauthenticated
requests get a `401` with a `Bearer` challenge. With no token:

- the CLI **refuses a non-loopback `--host`** (`0.0.0.0`, a LAN address) unless
  `--i-know-this-is-insecure` is passed;
- a loopback server answers only requests whose `Host` is `127.0.0.1`, `localhost` or `::1`
  and returns `400` to any other `Host` (DNS rebinding). An SSH tunnel keeps working; a
  reverse proxy that forwards the public `Host` header (Caddy by default, nginx with
  `proxy_set_header Host $host`) gets `400` on every request, so give such a server a token
  or have the proxy send a loopback `Host`.

A bearer token is only as safe as the channel: over plain HTTP it travels in clear and is
trivially sniffable, so the token alone protects nothing on the wire. **Keep `sse` /
`streamable-http` bound to loopback** (reach it over an SSH tunnel or a reverse proxy) **or
terminate TLS in front of the server**: never expose a tokened-but-unencrypted endpoint
beyond `localhost`.
