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

"""Smoke tests: the package imports, the CLI is wired, the BFF answers, and it serves the built front."""

from pathlib import Path

import pytest

pytest.importorskip("fastapi")
import konfai_studio
from konfai_studio.server import WEB_DIR, app
from starlette.testclient import TestClient


def test_package_imports() -> None:
    assert konfai_studio.__file__


def test_cli_entrypoint_is_callable() -> None:
    from konfai_studio.cli import main

    assert callable(main)


def test_health_endpoint() -> None:
    with TestClient(app) as client:
        response = client.get("/api/health")
    assert response.status_code == 200


def test_serves_the_built_frontend() -> None:
    """The wheel must ship the built front (``web/index.html``); serving "/" is that guarantee.

    Skips only when the front has not been built in this tree (a source checkout without ``npm run
    build``); the release wheel always carries it, and CI builds it before the test.
    """
    if not (Path(WEB_DIR) / "index.html").is_file():
        pytest.skip("frontend not built in this tree (run `npm --prefix frontend run build`)")
    with TestClient(app) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")


def test_a_workspace_created_outside_studio_joins_the_rail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """konfai-mcp creates workspaces from the command line too. Adopted only at startup, one appeared
    nowhere in a running Studio, and no amount of clicking would reveal it."""
    from konfai_studio.registry import _Registry

    monkeypatch.setenv("KONFAI_MCP_WORKSPACES_ROOT", str(tmp_path))
    registry = _Registry()
    registry.load()
    assert registry.names() == []

    (tmp_path / "sessions" / "made-elsewhere").mkdir(parents=True)

    assert registry.names() == ["made-elsewhere"]


def test_a_certfile_without_its_key_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """Half a TLS pair cannot serve; uvicorn would fail after the sockets are already claimed: refuse
    at the CLI, where the message can still name the missing flag."""
    from konfai_studio.cli import main

    monkeypatch.setattr("sys.argv", ["konfai-studio", "--ssl-certfile", "studio.crt"])
    with pytest.raises(SystemExit):
        main()


def test_an_empty_tls_flag_is_refused_not_downgraded(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unset shell variable in the documented one-liner passes ''; uvicorn treats '' as no-TLS, so
    accepting it would bring the server up in plain HTTP with the token on the wire."""
    from konfai_studio.cli import main

    monkeypatch.setattr("sys.argv", ["konfai-studio", "--ssl-certfile", "", "--ssl-keyfile", ""])
    with pytest.raises(SystemExit):
        main()
