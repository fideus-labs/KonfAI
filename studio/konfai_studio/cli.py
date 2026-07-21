# SPDX-License-Identifier: Apache-2.0
"""``konfai-studio`` - launch the BFF + front (loopback by default; remote behind a reverse proxy)."""

from __future__ import annotations

import argparse
import os

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(prog="konfai-studio", description="Launch KonfAI Studio.")
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default: loopback)")
    parser.add_argument("--port", type=int, default=8730, help="port (default: 8730)")
    parser.add_argument(
        "--proxy-headers",
        action="store_true",
        help="trust X-Forwarded-* from a reverse proxy (set when behind nginx/Caddy for correct client IP + scheme)",
    )
    parser.add_argument(
        "--forwarded-allow-ips",
        default="127.0.0.1",
        help="proxy IPs allowed to set forwarded headers ('*' trusts any — only behind a trusted proxy)",
    )
    parser.add_argument(
        "--i-know-this-is-insecure",
        action="store_true",
        help="allow binding a non-loopback address with no KONFAI_STUDIO_TOKEN (opens an unauthenticated shell)",
    )
    args = parser.parse_args()

    loopback = args.host in {"127.0.0.1", "::1", "localhost"}
    authed = bool(os.environ.get("KONFAI_STUDIO_TOKEN", "").strip())
    if not loopback and not authed and not args.i_know_this_is_insecure:
        # Studio drives arbitrary host compute — exposing it unauthenticated is a remote-shell handout.
        # Refuse by default (a printed warning is invisible under systemd); require a deliberate override.
        parser.error(
            f"refusing to bind {args.host} with no KONFAI_STUDIO_TOKEN — this exposes an unauthenticated UI "
            "and host shell to the network. Set a token and serve over TLS (see docs/REMOTE.md), or pass "
            "--i-know-this-is-insecure to override."
        )
    if not loopback and not authed:
        print(
            f"WARNING: {args.host} bound with no auth (--i-know-this-is-insecure) — anyone on the network has a shell."
        )
    print(f"KonfAI Studio -> http://{args.host}:{args.port}  (auth {'on' if authed else 'off'})")
    uvicorn.run(
        "konfai_studio.server:app",
        host=args.host,
        port=args.port,
        log_level="info",
        proxy_headers=args.proxy_headers,
        forwarded_allow_ips=args.forwarded_allow_ips,
    )


if __name__ == "__main__":
    main()
