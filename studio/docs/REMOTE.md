# Running KonfAI Studio on a remote server

Studio is **trusted-local by default**: it drives `konfai-mcp`, which reads arbitrary host paths and
runs training/inference jobs on the machine — arbitrary compute by design. On loopback that machine is
the operator's own. The moment the port is reachable over a network, that same power is handed to
whoever can reach it. This guide turns Studio into a private, single-operator remote app: **shared-token
auth in the BFF + TLS at a reverse proxy**.

> One authenticated user is the intended model. The token holder can browse the server's filesystem, run
> jobs, and (unless disabled) open a shell. This is *your* remote workstation, not a multi-tenant service.

## 1. Turn on authentication

Set one environment variable. Empty/unset ⇒ auth is off (local behaviour, unchanged). Set ⇒ every
request needs a valid session cookie or bearer token; the UI shows a lock screen until you enter it.

```bash
export KONFAI_STUDIO_TOKEN="$(openssl rand -hex 24)"   # a strong shared secret; keep it out of shell history
```

- The browser exchanges the token for an **httpOnly** session cookie (`/api/login`); the raw token never
  lives in client storage. The cookie is `Secure` by default — **you must serve over TLS** (step 3). If you
  reach Studio over plain `http`, the browser silently drops the cookie: login appears to succeed but
  immediately returns to the lock screen. Fix your TLS, or set `KONFAI_STUDIO_INSECURE_COOKIE=1` for local
  http testing only. Sign-out and the cookie's 30-day lifetime are **client-side**; the only server-side
  revocation is rotating `KONFAI_STUDIO_TOKEN` (which signs everyone out). Treat the cookie like the token.
- Programmatic clients can send `Authorization: Bearer <token>` instead of logging in.
- Public without a session: only the app shell (`/`, `/assets/*`, the logo) and `/api/health`,
  `/api/auth`, `/api/login`. Everything else — chat, jobs, dataset browsing, volume streaming, the
  terminal socket — is gated.

## 2. Gate the shell (recommended)

The integrated terminal is a full login shell on the host. Behind auth the token holder can already run
jobs, but the terminal is broader blast radius — disable it unless you specifically need it:

```bash
export KONFAI_STUDIO_TERMINAL=0
```

Host-path **reads** (dataset browser, volume streaming) stay on — picking a dataset that lives on the
server is the whole point of a remote deployment, and auth is what protects them. Config edits (`/api/config/save`)
are jailed to the session workspace, but this is a single-trusted-operator model, not a sandbox: the token
holder can write to operator-chosen host paths via **Export**/**Bundle** and, when the terminal is enabled,
has a full read/write host shell. Keep `KONFAI_STUDIO_TERMINAL=0` unless you need it.

## 3. Put it behind TLS + a reverse proxy

Bind Studio to loopback and let the proxy terminate TLS and forward HTTP **and** WebSocket upgrades.

```bash
KONFAI_STUDIO_TOKEN="…"  KONFAI_STUDIO_TERMINAL=0 \
konfai-studio --host 127.0.0.1 --port 8730 --proxy-headers
```

`--proxy-headers` trusts `X-Forwarded-*` from the proxy (correct client IP + scheme in logs); by default
only `127.0.0.1` may set them (`--forwarded-allow-ips` to widen — only behind a proxy you trust).

### Caddy (simplest — automatic HTTPS, WebSockets, SSE all handled)

```caddyfile
studio.example.com {
    reverse_proxy 127.0.0.1:8730
}
```

That's the whole config. Caddy provisions a certificate, upgrades `/api/terminal` transparently, and
streams SSE (`/api/chat`, `/api/live`) without buffering.

### nginx

```nginx
server {
    listen 443 ssl;
    server_name studio.example.com;
    ssl_certificate     /etc/letsencrypt/live/studio.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/studio.example.com/privkey.pem;

    location / {
        proxy_pass http://127.0.0.1:8730;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;          # WebSocket (terminal)
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_buffering off;                             # SSE: chat + live job log stream immediately
        proxy_read_timeout 3600s;                        # long training streams / idle terminals
    }
}
```

The front already uses same-origin relative URLs and picks `wss://` under HTTPS automatically — no build
flag or base-URL change is needed.

## 4. Run it as a service (systemd)

```ini
# /etc/systemd/system/konfai-studio.service
[Unit]
Description=KonfAI Studio
After=network.target

[Service]
User=konfai
WorkingDirectory=/home/konfai
Environment=KONFAI_STUDIO_TOKEN=REPLACE_WITH_A_STRONG_SECRET
Environment=KONFAI_STUDIO_TERMINAL=0
Environment=KONFAI_MCP_WORKSPACES_ROOT=/home/konfai/KonfAI_Workspaces
ExecStart=/usr/bin/konfai-studio --host 127.0.0.1 --port 8730 --proxy-headers
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

Keep the token in the unit's `Environment=` (or an `EnvironmentFile=` with `0600` perms), never on the
command line (it would show up in `ps`).

## Environment variables

| Variable | Default | Effect |
|---|---|---|
| `KONFAI_STUDIO_TOKEN` | *(unset)* | Set ⇒ require auth. The shared access token. |
| `KONFAI_STUDIO_TERMINAL` | `1` | `0` disables the integrated shell socket. |
| `KONFAI_STUDIO_INSECURE_COOKIE` | *(unset)* | `1` drops the cookie `Secure` flag — **local http testing only**, never in production. |
| `KONFAI_MCP_WORKSPACES_ROOT` | `~/KonfAI_Workspaces` | Where experiments/jobs are stored. |
| `KONFAI_STUDIO_LLM` | `claude-code` | The brain backend (see the README). |

## Threat model — what this does and does not do

- **Does:** gate every data/compute/terminal endpoint behind a shared token; reject cross-origin terminal
  WebSocket handshakes; keep the token out of the browser (httpOnly cookie); reap the whole job process
  group on cancel; jail `/api/config/save` **writes** to the session workspace.
- **Does not:** provide per-user isolation (one token, one trust level), rate-limit login (the token's
  entropy is the defence — use a strong one), give real session revocation (sign-out is client-side; rotate
  the token to revoke), or protect against a malicious authenticated operator. Export/Bundle write to
  operator-chosen host paths, and the dataset browser and volume streaming read arbitrary host paths **by
  design** — that is the same trust `konfai-mcp` itself assumes. Bind to loopback, serve over TLS, keep the
  terminal off unless needed, and treat the token (and its cookie) like an SSH key.
