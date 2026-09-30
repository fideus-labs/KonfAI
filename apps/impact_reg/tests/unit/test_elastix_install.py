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

"""How the elastix-IMPACT install is chosen, completed with the LibTorch it needs, and put in place."""

import zipfile
from pathlib import Path

import pytest
from impact_reg_konfai.models import elastix_install


def _machine(
    monkeypatch: pytest.MonkeyPatch,
    system: str = "Linux",
    machine: str = "x86_64",
    driver: tuple[int, int] | None = None,
    cuda: str | None = None,
    torch_version: str = "2.12.1",
) -> list[str]:
    """A machine whose downloads serve tiny archives laid out as the real ones; returns the fetched URLs."""
    import torch

    fetched: list[str] = []

    def fake_download(url: str, dst: Path) -> None:
        fetched.append(url)
        with zipfile.ZipFile(dst, "w") as archive:
            if "download.pytorch.org" in url:
                archive.writestr("libtorch/lib/libtorch_cpu.so", b"libtorch")
                archive.writestr("libtorch/lib/libdnnl.a", b"static")
                archive.writestr("libtorch/include/torch/torch.h", b"header")
                if "cu128" in url:
                    archive.writestr("libtorch/lib/libcudart-c3a75b33.so.12", b"cudart")
            else:
                archive.writestr("bin/elastix", b"new")
                archive.writestr("lib/libANNlib.so", b"new")

    monkeypatch.setattr(elastix_install, "download_file", fake_download)
    monkeypatch.setattr(elastix_install, "detect_nvidia_driver", lambda: (driver is not None, driver))
    monkeypatch.setattr(elastix_install.platform, "system", lambda: system)
    monkeypatch.setattr(elastix_install.platform, "machine", lambda: machine)
    monkeypatch.setattr(torch.version, "cuda", cuda)
    monkeypatch.setattr(torch, "__version__", torch_version)
    monkeypatch.setattr(elastix_install, "try_elastix", lambda path: None)
    monkeypatch.setattr(elastix_install, "ASSET_SHA256", {})
    return fetched


def _files(root: Path) -> list[str]:
    return sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file())


def test_a_reinstall_replaces_the_previous_one_whole(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Extracted over, a new asset left on the loader path whatever the previous one had and it did not overwrite.
    install = tmp_path / "elastix-impact"
    (install / "lib").mkdir(parents=True)
    (install / "lib" / "libstale.so").write_bytes(b"stale")
    (install / "bin").mkdir()
    (install / "bin" / "elastix").write_bytes(b"stale")
    _machine(monkeypatch)

    elastix_install.install_elastix_impact(install, force_cuda=False, force_cpu=True)

    # Only LibTorch's shared libraries are kept: its headers and static archives weigh more and serve nothing.
    assert _files(install) == ["FLAVOR", "bin/elastix", "lib/libANNlib.so", "libtorch/lib/libtorch_cpu.so"]
    assert (install / "bin" / "elastix").read_bytes() == b"new"
    assert [path.name for path in tmp_path.iterdir()] == ["elastix-impact"], "a staging directory was left behind"


@pytest.mark.parametrize(("torch_version", "libtorch"), [("2.12.1+cu130", True), ("2.8.0+cu128", False)])
def test_the_libtorch_the_asset_was_built_against_comes_with_it_unless_torch_is_that_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, torch_version: str, libtorch: bool
) -> None:
    # The 1.0.0 binaries link LibTorch 2.8 and fail to load against 2.9 and later (c10::SymInt::sym_ne became
    # inline): linked against the pip torch, every elastix preset failed on a fresh install.
    fetched = _machine(monkeypatch, driver=(595, 84), cuda="12.8", torch_version=torch_version)

    elastix_install.install_elastix_impact(tmp_path / "elastix-impact", force_cuda=False, force_cpu=False)

    expected = ["https://download.pytorch.org/libtorch/cu128/libtorch-shared-with-deps-2.8.0%2Bcu128.zip"]
    assert fetched[1:] == (expected if libtorch else [])
    if libtorch:
        cudart = tmp_path / "elastix-impact" / "libtorch" / "lib" / "libcudart.so.12"
        assert cudart.is_symlink() and cudart.read_bytes() == b"cudart", "the CUDA binary needs libcudart.so.12"


def test_a_torch_of_that_version_whose_probe_fails_still_gets_the_libtorch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A torch 2.8 built for CUDA 12.6 keeps no libcudart.so.12 where the cu128 binary looks for it.
    fetched = _machine(monkeypatch, driver=(595, 84), cuda="12.6", torch_version="2.8.0+cu126")
    probes: list[Path] = []

    def probe(path: Path) -> None:
        probes.append(path)
        if not (path / "libtorch").exists():
            raise RuntimeError("libcudart.so.12: cannot open shared object file")

    monkeypatch.setattr(elastix_install, "try_elastix", probe)

    elastix_install.install_elastix_impact(tmp_path / "elastix-impact", force_cuda=False, force_cpu=False)

    assert len(fetched) == 2 and len(probes) == 2


def test_an_install_that_cannot_run_never_replaces_the_one_in_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The previous install was deleted before its replacement was proven: a pre-1.7 install with its own LibTorch
    # made way for one that could not load.
    install = tmp_path / "elastix-impact"
    (install / "bin").mkdir(parents=True)
    (install / "bin" / "elastix").write_bytes(b"working")
    _machine(monkeypatch)

    def refuse(path: Path) -> None:
        raise RuntimeError("undefined symbol: _ZNK3c106SymInt6sym_neERKS0_")

    monkeypatch.setattr(elastix_install, "try_elastix", refuse)

    with pytest.raises(RuntimeError, match="sym_ne"):
        elastix_install.install_elastix_impact(install, force_cuda=False, force_cpu=False)

    assert _files(install) == ["bin/elastix"] and (install / "bin" / "elastix").read_bytes() == b"working"
    assert [path.name for path in tmp_path.iterdir()] == ["elastix-impact"]


def test_an_asset_that_is_not_the_released_one_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The 1.0.0 assets were re-uploaded in place once and nothing checked what was downloaded before running it.
    _machine(monkeypatch)
    monkeypatch.setattr(elastix_install, "ASSET_SHA256", {"elastix-impact-linux-x86_64-cpu.zip": "0" * 64})

    with pytest.raises(RuntimeError, match="does not match the sha256"):
        elastix_install.install_elastix_impact(tmp_path / "elastix-impact", force_cuda=False, force_cpu=False)

    assert not (tmp_path / "elastix-impact").exists()


def test_an_offline_download_says_how_to_install_elsewhere(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def offline(url: str, **kwargs):
        raise elastix_install.requests.ConnectionError("Name or service not known")

    monkeypatch.setattr(elastix_install.requests, "get", offline)

    with pytest.raises(RuntimeError, match="KONFAI_ELASTIX_DIR"):
        elastix_install.download_file("https://github.com/asset.zip", tmp_path / "asset.zip")


@pytest.mark.parametrize(
    ("torch_cuda", "asset"), [("12.8", "cu128"), ("13.0", "cu128"), ("11.8", "cu128"), (None, "cpu")]
)
def test_any_cuda_torch_takes_the_cuda_asset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, torch_cuda: str | None, asset: str
) -> None:
    # The asset brings its own LibTorch and CUDA runtime, so a CUDA 13 torch runs the cu128 binary: it used to ask
    # for a cu130 asset the release does not carry, fall back to the CPU one and fail every GPU IMPACT run.
    fetched = _machine(monkeypatch, driver=(595, 84), cuda=torch_cuda)

    elastix_install.install_elastix_impact(tmp_path / "elastix-impact", force_cuda=False, force_cpu=False)

    assert fetched[0].endswith(f"-{asset}.zip")
    assert elastix_install.installed_flavor(tmp_path / "elastix-impact") == asset


def test_a_driver_older_than_the_cuda_the_asset_links_takes_the_cpu_asset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fetched = _machine(monkeypatch, driver=(560, 35), cuda="12.8")

    elastix_install.install_elastix_impact(tmp_path / "elastix-impact", force_cuda=False, force_cpu=False)

    assert fetched[0].endswith("-cpu.zip")


def test_a_forced_cuda_install_refuses_a_driver_that_cannot_run_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _machine(monkeypatch, driver=(560, 35), cuda="12.8")

    with pytest.raises(RuntimeError, match="not suitable for cu128"):
        elastix_install.install_elastix_impact(tmp_path / "elastix-impact", force_cuda=True, force_cpu=False)


def test_apple_silicon_installs_the_arm64_macos_asset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The only macOS asset holds an arm64 binary but was keyed x86_64, so Apple Silicon found no asset at all.
    fetched = _machine(monkeypatch, system="Darwin", machine="arm64")

    elastix_install.install_elastix_impact(tmp_path / "elastix-impact", force_cuda=False, force_cpu=False)

    assert fetched == [
        f"https://github.com/vboussot/ImpactElastix/releases/download/{elastix_install.GITHUB_TAG}/"
        "elastix-impact-macos-14-x86_64-cpu.zip",
        "https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.8.0.zip",
    ]


@pytest.mark.parametrize(("system", "machine"), [("Linux", "aarch64"), ("Darwin", "x86_64"), ("FreeBSD", "amd64")])
def test_a_machine_without_a_build_is_told_how_to_bring_its_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, system: str, machine: str
) -> None:
    _machine(monkeypatch, system=system, machine=machine)

    with pytest.raises(RuntimeError, match="KONFAI_ELASTIX_DIR") as refused:
        elastix_install.install_elastix_impact(tmp_path / "elastix-impact", force_cuda=False, force_cpu=False)
    # GAP5-16: the refusal names what still runs there, without elastix.
    assert "ConvexAdam" in str(refused.value) and "FireANTs" in str(refused.value)


@pytest.mark.parametrize(
    ("recorded", "status", "expected"),
    [("cpu", 200, True), (None, 200, False), ("cpu", 404, False), ("cu128", 200, False)],
)
def test_a_cpu_install_is_replaced_once_the_cuda_one_can_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recorded: str | None, status: int, expected: bool
) -> None:
    # A CPU build passes try_elastix, so without this check it stayed in place and every GPU run failed on it. An
    # install of the wanted flavor is never checked again, and neither is one that recorded nothing (made before
    # the flavor was recorded, possibly a CUDA build with its own LibTorch): no request at all.
    import torch

    install = tmp_path / "elastix-impact"
    install.mkdir()
    if recorded is not None:
        (install / elastix_install.FLAVOR_FILE).write_text(recorded)
    requested: list[str] = []

    class Response:
        status_code = status

    def fake_head(url: str, **kwargs) -> Response:
        requested.append(url)
        return Response()

    monkeypatch.setattr(elastix_install.requests, "head", fake_head)
    monkeypatch.setattr(elastix_install, "detect_nvidia_driver", lambda: (True, (595, 84)))
    monkeypatch.setattr(elastix_install.platform, "system", lambda: "Linux")
    monkeypatch.setattr(elastix_install.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(torch.version, "cuda", "13.0")

    assert elastix_install.cuda_upgrade_available(install) is expected
    assert bool(requested) is (recorded == "cpu")
