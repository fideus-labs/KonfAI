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

"""What an app that runs its authors' tool needs to fetch and run it."""

from konfai_apps.pinned import gpu_environment


def test_a_tool_s_subprocess_runs_on_the_physical_gpu_behind_the_app_s_index(monkeypatch) -> None:
    # The subprocess reads CUDA_VISIBLE_DEVICES afresh: the index 1 of a process that sees GPUs 3 and 5 is GPU 5.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,5")
    assert gpu_environment(1)["CUDA_VISIBLE_DEVICES"] == "5"
    on_cpu = gpu_environment(-1, PYTHONPATH="x")
    assert on_cpu["CUDA_VISIBLE_DEVICES"] == "" and on_cpu["PYTHONPATH"] == "x"


def test_a_release_is_unpacked_once_with_its_modes_and_a_wrong_hash_leaves_nothing(tmp_path, monkeypatch) -> None:
    import hashlib
    import os
    import zipfile

    import pytest
    from konfai_apps.pinned import pinned_archive

    archive = tmp_path / "tool.zip"
    with zipfile.ZipFile(archive, "w") as packed:
        entry = zipfile.ZipInfo("tool/bin/run")
        entry.external_attr = 0o755 << 16
        packed.writestr(entry, "#!/bin/sh\n")
    sha256 = hashlib.sha256(archive.read_bytes()).hexdigest()
    monkeypatch.setenv("KONFAI_TOOLS_CACHE", str(tmp_path / "cache"))

    root = pinned_archive("tool", archive.as_uri(), sha256)
    assert (root / "tool/bin/run").is_file() and not (root / "archive").exists()
    if os.name != "nt":  # Windows keeps no Unix modes
        assert (root / "tool/bin/run").stat().st_mode & 0o777 == 0o755
    with pytest.raises(RuntimeError, match="hash"):
        pinned_archive("other", archive.as_uri(), "0" * 64)
    assert sorted(p.name for p in (tmp_path / "cache").iterdir()) == [root.name]
