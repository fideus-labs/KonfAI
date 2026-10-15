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

"""With a server, each preset pass runs there; its token travels in the environment, not the argv."""

from pathlib import Path

import pytest

pytest.importorskip("SimpleITK")

from impact_reg_konfai.impact_reg import ImpactRegKonfAIApp
from konfai import RemoteServer


@pytest.mark.parametrize(
    "server",
    [None, RemoteServer("gpu-box", 9000, "secret"), RemoteServer("gpu-box", 9000, None)],
    ids=["local", "remote", "remote without a token"],
)
def test_a_preset_pass_runs_where_the_app_was_told(tmp_path: Path, monkeypatch, write_preset_output, server) -> None:
    captured: list[tuple[list[str], dict[str, str]]] = []

    def fake_run(command, env, **kwargs):
        captured.append((list(command), env))
        write_preset_output(Path(command[command.index("-o") + 1]) / "reg" / "DVF" / "P000")

    monkeypatch.setattr("impact_reg_konfai.impact_reg.subprocess.run", fake_run)
    monkeypatch.setenv("KONFAI_API_TOKEN", "inherited")
    work = tmp_path / "work"
    work.mkdir()

    ImpactRegKonfAIApp(remote_server=server)._infer_preset(
        "FireANTs_SyN", [tmp_path / "f.mha"], [tmp_path / "m.mha"], [], [], work, [0], None, True
    )

    [(command, env)] = captured
    if server is None:
        assert "--host" not in command
    else:
        at = command.index("--host")
        assert command[at : at + 4] == ["--host", "gpu-box", "--port", "9000"]
        assert "secret" not in command and env.get("KONFAI_API_TOKEN") == server.token
