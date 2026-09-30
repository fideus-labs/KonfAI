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

"""The package's own build rules, run on the real ``setup.py``."""

import runpy
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("recorded", "expected"),
    [
        (["konfai[dicom,monitoring,omezarr]>=1.8.6", "konfai-apps>=1.8.6", "h5py"], None),
        ([], ["konfai[monitoring,omezarr,dicom]>=1.8.7.dev33", "konfai-apps>=1.8.7.dev33", "h5py"]),
    ],
)
def test_a_wheel_built_from_the_sdist_keeps_its_pins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recorded: list[str], expected: list[str] | None
) -> None:
    # `python -m build` makes the wheel from the unpacked sdist, where there is no git history: the pins must be
    # the ones the sdist recorded, not '>= <its own dev version>', which no released konfai satisfies. An sdist
    # that recorded none must not make a wheel without dependencies: they are computed as from a tree.
    pytest.importorskip("setuptools_scm")
    import setuptools

    (tmp_path / "setup.py").write_text((Path(__file__).resolve().parents[2] / "setup.py").read_text())
    (tmp_path / "PKG-INFO").write_text(
        "Metadata-Version: 2.4\nName: impact-reg-konfai\nVersion: 1.8.7.dev33\n"
        + "".join(f"Requires-Dist: {requirement}\n" for requirement in recorded)
    )
    captured: dict = {}
    monkeypatch.setattr(setuptools, "setup", lambda **kwargs: captured.update(kwargs))
    runpy.run_path(str(tmp_path / "setup.py"), run_name="__main__")
    assert captured["install_requires"] == (expected or recorded)
