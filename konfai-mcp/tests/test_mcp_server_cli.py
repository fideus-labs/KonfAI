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

import os
import sys
from pathlib import Path
from types import ModuleType

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

import konfai_mcp  # noqa: E402


def test_cli_sets_environment_and_forwards_transport(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_server_main(**kwargs):
        captured.update(kwargs)

    fake_server = ModuleType("konfai_mcp.server")
    fake_server.main = fake_server_main  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "konfai_mcp.server", fake_server)

    monkeypatch.delenv("KONFAI_MCP_TRANSPORT", raising=False)
    monkeypatch.delenv("KONFAI_MCP_SESSION", raising=False)
    monkeypatch.delenv("KONFAI_MCP_WORKSPACES_ROOT", raising=False)
    monkeypatch.delenv("KONFAI_MCP_PORT", raising=False)
    monkeypatch.delenv("KONFAI_MCP_BEARER_TOKEN", raising=False)

    previous = {
        key: os.environ.get(key)
        for key in (
            "KONFAI_MCP_TRANSPORT",
            "KONFAI_MCP_SESSION",
            "KONFAI_MCP_WORKSPACES_ROOT",
            "KONFAI_MCP_PORT",
            "KONFAI_MCP_PATH",
            "KONFAI_MCP_LOG_LEVEL",
            "KONFAI_MCP_BEARER_TOKEN",
        )
    }
    try:
        konfai_mcp.main(
            [
                "--transport",
                "sse",
                "--session",
                "challenge round 1",
                "--workspace-root",
                "/tmp/konfai-workspaces",
                "--host",
                "127.0.0.1",
                "--port",
                "8123",
                "--path",
                "/mcp",
                "--log-level",
                "warning",
                "--bearer-token",
                "dev-token",
            ]
        )

        assert captured == {
            "transport": "sse",
            "host": "127.0.0.1",
            "port": 8123,
            "path": "/mcp",
            "log_level": "warning",
            "bearer_token": "dev-token",
        }
        assert os.environ["KONFAI_MCP_TRANSPORT"] == "sse"
        assert os.environ["KONFAI_MCP_SESSION"] == "challenge round 1"
        assert os.environ["KONFAI_MCP_WORKSPACES_ROOT"] == "/tmp/konfai-workspaces"
        assert os.environ["KONFAI_MCP_PORT"] == "8123"
        assert os.environ["KONFAI_MCP_BEARER_TOKEN"] == "dev-token"
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
