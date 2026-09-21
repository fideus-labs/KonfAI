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

"""The elastix-IMPACT install replaces a previous one instead of extracting over it."""

import zipfile
from pathlib import Path

import pytest
from impact_reg_konfai.models import elastix_install


def test_a_reinstall_drops_what_the_previous_asset_left_behind(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # An older asset bundled its own LibTorch under lib/. Extracted over, the new asset left it there,
    # ahead of the environment's torch on the loader path: a CUDA build ran CPU-only.
    install = tmp_path / "elastix-impact"
    (install / "lib").mkdir(parents=True)
    (install / "lib" / "libtorch_cpu.so").write_bytes(b"stale")
    (install / "bin").mkdir()
    (install / "bin" / "elastix").write_bytes(b"stale")

    def fake_download(url: str, dst: Path) -> None:
        with zipfile.ZipFile(dst, "w") as archive:
            archive.writestr("bin/elastix", b"new")
            archive.writestr("lib/libANNlib.so", b"new")

    monkeypatch.setattr(elastix_install, "download_file", fake_download)
    monkeypatch.setattr(elastix_install, "detect_nvidia_driver", lambda: (False, None))
    monkeypatch.setattr(elastix_install.platform, "system", lambda: "Linux")
    monkeypatch.setattr(elastix_install.platform, "machine", lambda: "x86_64")

    elastix_install.install_elastix_impact(install, force_cuda=False, force_cpu=True)

    assert sorted(path.relative_to(install).as_posix() for path in install.rglob("*") if path.is_file()) == [
        "bin/elastix",
        "lib/libANNlib.so",
    ]
    assert (install / "bin" / "elastix").read_bytes() == b"new"


@pytest.mark.parametrize(("torch_cuda", "asset"), [("12.8", "cu128"), ("13.0", "cpu"), (None, "cpu")])
def test_the_cuda_asset_is_fetched_only_for_a_torch_that_can_load_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, torch_cuda: str | None, asset: str
) -> None:
    # The CUDA asset links the CUDA 12 runtime from where torch keeps its own. Under a torch built for
    # CUDA 13 it could not load at all, and the install failed where the CPU asset would have run.
    import torch

    fetched: list[str] = []

    def fake_download(url: str, dst: Path) -> None:
        fetched.append(url)
        with zipfile.ZipFile(dst, "w") as archive:
            archive.writestr("bin/elastix", b"new")

    monkeypatch.setattr(elastix_install, "download_file", fake_download)
    monkeypatch.setattr(elastix_install, "detect_nvidia_driver", lambda: (True, (595, 84)))
    monkeypatch.setattr(elastix_install.platform, "system", lambda: "Linux")
    monkeypatch.setattr(elastix_install.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(torch.version, "cuda", torch_cuda)

    elastix_install.install_elastix_impact(tmp_path / "elastix-impact", force_cuda=False, force_cpu=False)

    assert fetched[0].endswith(f"-{asset}.zip")


def test_a_forced_cuda_install_refuses_a_torch_that_cannot_load_the_asset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import torch

    monkeypatch.setattr(elastix_install, "detect_nvidia_driver", lambda: (True, (595, 84)))
    monkeypatch.setattr(elastix_install.platform, "system", lambda: "Linux")
    monkeypatch.setattr(elastix_install.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(torch.version, "cuda", "13.0")

    with pytest.raises(NameError, match="KONFAI_ELASTIX_DIR"):
        elastix_install.install_elastix_impact(tmp_path / "elastix-impact", force_cuda=True, force_cpu=False)
