# Copyright (c) 2025 Valentin Boussot
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Standalone KonfAI MCP package."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from typing import Any

_TRANSPORT_CHOICES = ("stdio", "sse", "streamable-http")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the KonfAI MCP server.")
    parser.add_argument(
        "--transport",
        choices=_TRANSPORT_CHOICES,
        default=os.environ.get("KONFAI_MCP_TRANSPORT", "stdio"),
        help="MCP transport to expose. Defaults to stdio.",
    )
    parser.add_argument(
        "--session",
        default=os.environ.get("KONFAI_MCP_SESSION"),
        help="Default session name used by this server process.",
    )
    parser.add_argument(
        "--workspace-root",
        default=os.environ.get("KONFAI_MCP_WORKSPACES_ROOT"),
        help="Root directory that stores MCP sessions and datasets.",
    )
    parser.add_argument(
        "--log-tail-lines",
        type=int,
        default=(int(os.environ["KONFAI_MCP_LOG_TAIL_LINES"]) if "KONFAI_MCP_LOG_TAIL_LINES" in os.environ else None),
        help="Default maximum number of log lines returned by log-tail helpers.",
    )
    parser.add_argument("--host", default=os.environ.get("KONFAI_MCP_HOST"), help="Host for SSE/HTTP transports.")
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ["KONFAI_MCP_PORT"]) if "KONFAI_MCP_PORT" in os.environ else None,
        help="Port for SSE/HTTP transports.",
    )
    parser.add_argument(
        "--path",
        default=os.environ.get("KONFAI_MCP_PATH"),
        help="HTTP path prefix for SSE or streamable HTTP transports.",
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("KONFAI_MCP_LOG_LEVEL"),
        help="FastMCP/Uvicorn log level when the selected transport supports it.",
    )
    parser.add_argument(
        "--bearer-token",
        default=os.environ.get("KONFAI_MCP_BEARER_TOKEN"),
        help="Optional bearer token required for SSE or streamable HTTP transports.",
    )
    return parser


def _apply_cli_environment(args: argparse.Namespace) -> None:
    updates: dict[str, Any] = {
        "KONFAI_MCP_TRANSPORT": args.transport,
        "KONFAI_MCP_SESSION": args.session,
        "KONFAI_MCP_WORKSPACES_ROOT": args.workspace_root,
        "KONFAI_MCP_LOG_TAIL_LINES": args.log_tail_lines,
        "KONFAI_MCP_HOST": args.host,
        "KONFAI_MCP_PORT": args.port,
        "KONFAI_MCP_PATH": args.path,
        "KONFAI_MCP_LOG_LEVEL": args.log_level,
        "KONFAI_MCP_BEARER_TOKEN": args.bearer_token,
    }
    for key, value in updates.items():
        if value is None:
            continue
        os.environ[key] = str(value)


def main(argv: Sequence[str] | None = None) -> None:
    """Console-script entrypoint for the MCP server package."""
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    # argparse validates a value passed on the CLI against `choices`, but NOT a default, so a bad
    # KONFAI_MCP_TRANSPORT env value would otherwise flow straight through to the server. Reject it here.
    if args.transport not in _TRANSPORT_CHOICES:
        parser.error(
            f"KONFAI_MCP_TRANSPORT={args.transport!r} is not a valid transport; "
            f"choose from {', '.join(_TRANSPORT_CHOICES)}."
        )
    _apply_cli_environment(args)

    from .server import main as server_main

    server_main(
        transport=args.transport,
        host=args.host,
        port=args.port,
        path=args.path,
        log_level=args.log_level,
        bearer_token=args.bearer_token,
    )


__all__ = ["main"]
