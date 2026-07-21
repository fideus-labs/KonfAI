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

"""delete_run removes exactly one run's outputs and never escapes the session workspace."""

import asyncio
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import fastmcp
import pytest


def test_delete_run_removes_one_run_and_stays_jailed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    load_mcp_server: Callable[[], ModuleType],
) -> None:
    monkeypatch.setenv("KONFAI_MCP_WORKSPACES_ROOT", str(tmp_path / "workspaces"))
    mcp_server = load_mcp_server()

    async def scenario() -> Path:
        async with fastmcp.Client(mcp_server.mcp) as client:
            created = await client.call_tool("initialize_session", {"overwrite": True})
            session_dir = Path(created.structured_content["path"])
            for sub in ("Statistics/MR2CT_01", "Checkpoints/MR2CT_01", "Statistics/OTHER_02"):
                (session_dir / sub).mkdir(parents=True)
            (session_dir / "Statistics/MR2CT_01/log_0.txt").write_text("x", encoding="utf-8")

            result = await client.call_tool("delete_run", {"run_name": "MR2CT_01", "kind": "train"})
            deleted = result.structured_content["deleted"]
            assert set(deleted) == {"Statistics/MR2CT_01", "Checkpoints/MR2CT_01"}
            assert not (session_dir / "Statistics/MR2CT_01").exists()
            assert not (session_dir / "Checkpoints/MR2CT_01").exists()
            # only the named run is removed; the session and other runs survive
            assert (session_dir / "Statistics/OTHER_02").exists()
            assert session_dir.exists()
            return session_dir

    session_dir = asyncio.run(scenario())

    # The jail: a run name carrying a path separator is refused before touching disk.
    (session_dir / "Statistics").mkdir(exist_ok=True)
    with pytest.raises(ValueError, match="run_name"):
        mcp_server.delete_run(run_name="../Statistics", kind="train")
    assert (session_dir / "Statistics").exists()
