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

"""The workspace the orchestrator was given must reach the nested konfai-apps commands.

``--tmp-dir`` exists so a caller whose system temporary directory is the wrong medium can place the
volume-sized staging itself. That only holds if EVERY nested invocation is told about it: a command
left without one auto-creates its own workspace under TMPDIR and stages there, which is precisely
the traffic the option was added to move. These tests pin the forwarding for each nested command, so
a new one cannot be added without it.
"""

from pathlib import Path

import pytest

sitk = pytest.importorskip("SimpleITK")

from impact_reg_konfai.impact_reg import ImpactRegKonfAIApp  # noqa: E402


def _tmp_dir_value(command: list[str]) -> str:
    """The value ``--tmp-dir`` carries in a captured command line (fails the test when absent)."""
    assert "--tmp-dir" in command, f"nested command carries no --tmp-dir: {command}"
    return command[command.index("--tmp-dir") + 1]


def test_infer_preset_forwards_the_workspace(tmp_path: Path, monkeypatch, write_preset_output) -> None:
    """``konfai-apps infer`` is told to work in the directory the orchestrator staged for it.

    Pointed at ``-o``, so konfai-apps writes the prediction straight there instead of staging it in a
    throwaway workspace and copying it in.
    """
    captured: list[list[str]] = []

    def fake_run(command, **kwargs):
        captured.append(list(command))
        # Stand in for the preset run: konfai-apps leaves one case directory per unit under -o.
        write_preset_output(Path(command[command.index("-o") + 1]) / "P000")
        return None

    monkeypatch.setattr("impact_reg_konfai.impact_reg.subprocess.run", fake_run)

    work = tmp_path / "work"
    work.mkdir()
    app = ImpactRegKonfAIApp()
    app._infer_preset("FireANTs_SyN", [tmp_path / "f.mha"], [tmp_path / "m.mha"], [], [], 1, work, [], None, True)

    assert len(captured) == 1
    assert _tmp_dir_value(captured[0]) == str(work / "FireANTs_SyN")


def test_uncertainty_stages_inside_the_callers_tmp_dir(tmp_path: Path, write_preset_output) -> None:
    """The staging and the run workspaces live in a private directory INSIDE the caller's tmp_dir --
    never under the system TMPDIR -- and the caller's directory is left standing, emptied, when the
    run is done. The spread map is the one deliverable, under <output>/uncertainty/."""
    _, first = write_preset_output(tmp_path / "a")
    _, second = write_preset_output(tmp_path / "b")
    staging = tmp_path / "staging"

    ImpactRegKonfAIApp().uncertainty(
        preset="FireANTs_SyN",
        dvfs=[first, second],
        output=tmp_path / "out",
        quiet=True,
        tmp_dir=staging,
    )

    assert staging.is_dir()
    assert list(staging.iterdir()) == []
    spread = sorted((tmp_path / "out" / "uncertainty").iterdir())
    assert [path.name for path in spread] == ["Uncertainty.mha"]
