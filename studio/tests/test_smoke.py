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
