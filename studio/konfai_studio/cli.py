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
        help="trust X-Forwarded-* from a reverse proxy: correct client IP + scheme, and what lets"
        " /api/quit tell a local user from a remote one (set when behind nginx/Caddy)",
    )
    parser.add_argument(
        "--forwarded-allow-ips",
        default="127.0.0.1",
        help="proxy IPs allowed to set forwarded headers ('*' trusts any: only behind a trusted proxy)",
    )
    parser.add_argument(
        "--i-know-this-is-insecure",
        action="store_true",
        help="allow binding a non-loopback address with no KONFAI_STUDIO_TOKEN (opens an unauthenticated shell)",
    )
    parser.add_argument(
        "--ssl-certfile",
        default=None,
        help="TLS certificate (PEM). With --ssl-keyfile, Studio serves https directly, which is also what "
        "lets the browser grant the microphone off localhost.",
    )
    parser.add_argument("--ssl-keyfile", default=None, help="TLS private key (PEM), required with --ssl-certfile")
    args = parser.parse_args()

    for flag, value in (("--ssl-certfile", args.ssl_certfile), ("--ssl-keyfile", args.ssl_keyfile)):
        if value == "":
            # An unset shell variable in the documented one-liner lands here; uvicorn treats "" as
            # no-TLS and comes up in plain http with the token on the wire.
            parser.error(f"{flag} is empty: an unset shell variable? Studio will not silently fall back to http.")
    if (args.ssl_certfile is None) != (args.ssl_keyfile is None):
        parser.error("--ssl-certfile and --ssl-keyfile go together: a certificate cannot serve without its key.")

    loopback = args.host in {"127.0.0.1", "::1", "localhost"}
    authed = bool(os.environ.get("KONFAI_STUDIO_TOKEN", "").strip())
    if not loopback and not authed and not args.i_know_this_is_insecure:
        # Studio drives arbitrary host compute: exposing it unauthenticated is a remote-shell handout.
        # Refuse by default (a printed warning is invisible under systemd); require a deliberate override.
        parser.error(
            f"refusing to bind {args.host} with no KONFAI_STUDIO_TOKEN: this exposes an unauthenticated UI "
            "and host shell to the network. Set a token and serve over TLS (see studio/docs/REMOTE.md), or pass "
            "--i-know-this-is-insecure to override."
        )
    if not loopback and not authed:
        print(
            f"WARNING: {args.host} bound with no auth (--i-know-this-is-insecure): anyone on the network has a shell."
        )
    scheme = "https" if args.ssl_certfile else "http"
    print(f"KonfAI Studio -> {scheme}://{args.host}:{args.port}  (auth {'on' if authed else 'off'})")
    # The app is imported by uvicorn from a string, so it cannot read these arguments. /api/quit
    # needs to know whether the peer it sees was rewritten from X-Forwarded-For or is a proxy's.
    os.environ["KONFAI_STUDIO_PROXY_HEADERS"] = "1" if args.proxy_headers else "0"
    uvicorn.run(
        "konfai_studio.server:app",
        host=args.host,
        port=args.port,
        log_level="info",
        proxy_headers=args.proxy_headers,
        forwarded_allow_ips=args.forwarded_allow_ips,
        ssl_certfile=args.ssl_certfile,
        ssl_keyfile=args.ssl_keyfile,
    )


if __name__ == "__main__":
    main()
