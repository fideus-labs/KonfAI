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
    """``konfai-apps infer`` is told to work in a directory the orchestrator staged for it, beside ``-o``.

    Given a workspace of its caller's, konfai-apps writes the prediction straight into ``-o`` instead of
    staging it in a throwaway workspace and copying it in. Beside ``-o`` and not ``-o`` itself: the bundle
    files it copies into its workspace, a preset's own folders among them, would otherwise sit where the
    output group is discovered, and a second directory of directories there is a second group.
    """
    captured: list[list[str]] = []

    def fake_run(command, **kwargs):
        captured.append(list(command))
        # Stand in for the preset run: konfai-apps leaves one dataset per output group under -o,
        # laid out <run>/<group>/<case>: the shape _find_output_group discovers the group from.
        write_preset_output(Path(command[command.index("-o") + 1]) / "reg" / "DVF" / "P000")
        # and the bundle's files in its workspace, a folder of assets included
        (Path(_tmp_dir_value(command)) / "assets" / "models").mkdir(parents=True)
        return None

    monkeypatch.setattr("impact_reg_konfai.impact_reg.subprocess.run", fake_run)

    work = tmp_path / "work"
    work.mkdir()
    app = ImpactRegKonfAIApp()
    group, fields = app._infer_preset(
        "FireANTs_SyN", [tmp_path / "f.mha"], [tmp_path / "m.mha"], [], [], work, [], None, True
    )

    assert len(captured) == 1
    assert work / "FireANTs_SyN" in Path(_tmp_dir_value(captured[0])).parents
    assert group == "DVF" and list(fields) == ["P000"]


def test_uncertainty_stages_inside_the_callers_tmp_dir(tmp_path: Path, write_preset_output) -> None:
    """The staging and the run workspaces live in a private directory INSIDE the caller's tmp_dir --
    never under the system TMPDIR, and the caller's directory is left standing, emptied, when the
    run is done. The spread map is the one deliverable, under <output>/uncertainty/."""
    _, first = write_preset_output(tmp_path / "a")
    _, second = write_preset_output(tmp_path / "b")
    staging = tmp_path / "staging"

    ImpactRegKonfAIApp().uncertainty(
        dvfs=[first, second],
        output=tmp_path / "out",
        quiet=True,
        tmp_dir=staging,
    )

    assert staging.is_dir()
    assert list(staging.iterdir()) == []
    spread = sorted((tmp_path / "out" / "uncertainty").iterdir())
    assert [path.name for path in spread] == ["Uncertainty.mha"]


def test_the_work_dir_falls_back_inside_an_output_whose_parent_is_read_only(tmp_path: Path) -> None:
    """``-o .`` in a home directory under a root-owned ``/home``: the parent refuses, the output itself does not."""
    from impact_reg_konfai.impact_reg import _work_dir

    parent = tmp_path / "read_only"
    output = parent / "out"
    output.mkdir(parents=True)
    parent.chmod(0o555)
    try:
        work = _work_dir(None, output, "impact_reg_")
        assert work.parent == output and work.name.startswith(".impact_reg_")
    finally:
        parent.chmod(0o755)
    beside = _work_dir(None, tmp_path / "elsewhere", "impact_reg_")
    assert beside.parent == tmp_path and beside.name.startswith(".elsewhere.impact_reg_")
