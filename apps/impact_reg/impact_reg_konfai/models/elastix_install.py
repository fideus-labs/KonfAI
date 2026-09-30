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

import argparse
import hashlib
import os
import platform
import re
import shutil
import stat
import subprocess  # nosec B404
import tempfile
import zipfile
from collections.abc import Callable
from pathlib import Path

import requests
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Elastix + IMPACT binary assets hosted on GitHub Releases.
#
# Key format: (OS, ARCH, FLAVOR)
#   - OS     : platform.system() -> "Linux", "Windows", "Darwin"
#   - ARCH   : normalized architecture -> "x86_64", "arm64"
#   - FLAVOR : "cpu" or "cu128", only those GITHUB_TAG publishes
#
# No asset bundles LibTorch, yet each binary links the LibTorch it was built against (ASSET_LIBTORCH), which
# keeps no ABI across minor versions. The environment's pip ``torch`` serves when it is that major.minor;
# otherwise the installer downloads that exact LibTorch beside the binary (loader_env searches it first).
# elastix runs as a subprocess, so its LibTorch never has to be the one the Python process imported: the
# cu128 asset with its own LibTorch (CUDA runtime included) runs beside a CUDA 13 torch.
# -----------------------------------------------------------------------------
ELX_ASSET_TEMPLATE = {
    ("Linux", "x86_64", "cpu"): "elastix-impact-linux-x86_64-cpu.zip",
    ("Linux", "x86_64", "cu128"): "elastix-impact-linux-x86_64-cu128.zip",
    ("Windows", "x86_64", "cpu"): "elastix-impact-windows-x86_64-cpu.zip",
    ("Windows", "x86_64", "cu128"): "elastix-impact-windows-x86_64-cu128.zip",
    # Built on macos-14, Apple Silicon: the name says x86_64, the binary is arm64.
    ("Darwin", "arm64", "cpu"): "elastix-impact-macos-14-x86_64-cpu.zip",
}

#: The sha256 GitHub publishes for each asset of GITHUB_TAG, a prerelease: an archive that differs is refused rather
#: than run.
ASSET_SHA256 = {
    "elastix-impact-linux-x86_64-cpu.zip": "cfdbb65c2a18bc0a535b50cb8e497a9671ef35fb7a34c3ec56cc5faf0c5c8fac",
    "elastix-impact-linux-x86_64-cu128.zip": "fb21d43b9c1449a0423d1e80544765f1aac2ad113719caaaeaf6333c66cfbb05",
    "elastix-impact-windows-x86_64-cpu.zip": "709493cecf9d752ab6a8bbcd2c953503c9c3c7bd6f776396496afd3be422e7fe",
    "elastix-impact-windows-x86_64-cu128.zip": "f2e1d5d3279f5414d73c01b9bfb8a9ae7a42e79c10e2e9308a4f040c865bb225",
    "elastix-impact-macos-14-x86_64-cpu.zip": "9df7ce62b5602ba5348d18eb9b03283ab1fb0409ef372cd90f48b42e78955755",
}

#: The LibTorch each flavor of GITHUB_TAG was built against (the release notes of ImpactElastix 1.0.0). From
#: torch 2.9 on, c10::SymInt::sym_ne is inline and those binaries no longer load against the pip torch.
ASSET_LIBTORCH = {"cpu": "2.8.0", "cu128": "2.8.0"}


def libtorch_url(os_name: str, flavor: str) -> str:
    """The official shared LibTorch the ``flavor`` asset was built against, for ``os_name``."""
    version = ASSET_LIBTORCH[flavor]
    if os_name == "Darwin":
        return f"https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-{version}.zip"
    archive = "libtorch-win-shared-with-deps" if os_name == "Windows" else "libtorch-shared-with-deps"
    return f"https://download.pytorch.org/libtorch/{flavor}/{archive}-{version}%2B{flavor}.zip"


# -----------------------------------------------------------------------------
# Minimum NVIDIA driver versions per CUDA flavor, from NVIDIA's own compatibility
# table. The CUDA Toolkit itself is NOT required, only a recent enough driver.
# -----------------------------------------------------------------------------
CUDA_MIN_DRIVER = {
    "cu128": {"Linux": (570, 26), "Windows": (570, 65)},
    "cu130": {"Linux": (580, 65), "Windows": (580, 88)},
}
CUDA128_MIN_DRIVER_LINUX = CUDA_MIN_DRIVER["cu128"]["Linux"]
CUDA128_MIN_DRIVER_WINDOWS = CUDA_MIN_DRIVER["cu128"]["Windows"]

GITHUB_OWNER = "vboussot"
GITHUB_REPO = "ImpactElastix"
GITHUB_TAG = "1.0.0"


DEFAULT_PREFIX = Path.cwd() / "elastix-impact"


def detect_nvidia_driver() -> tuple[bool, tuple[int, int] | None]:
    """
    Detect presence of an NVIDIA GPU and extract the driver version.
    Returns:
        (True, (major, minor)) if detected
        (False, None) if nvidia-smi is not available
    """
    try:
        nvidia_smi = shutil.which("nvidia-smi")
        if nvidia_smi is None:
            raise RuntimeError("nvidia-smi not found in PATH")

        out = (
            subprocess.check_output(
                [nvidia_smi, "--query-gpu=driver_version", "--format=csv,noheader"], stderr=subprocess.DEVNULL
            )  # nosec B603
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return (False, None)

    # Exemple: "575.64"
    m = re.match(r"([0-9]+)\.([0-9]+)", out)
    if not m:
        return (True, None)

    return (True, (int(m.group(1)), int(m.group(2))))


def driver_ok_for_cuda(os_name: str, drv: tuple[int, int] | None, flavor: str | None = None) -> bool:
    """Whether the detected NVIDIA driver meets the minimum for the CUDA the asset links.

    ``flavor`` is the asset that would be installed (``torch_cuda_flavor``); without one the CUDA 12.8
    floor is used.
    """
    if drv is None:
        return False
    minimums = CUDA_MIN_DRIVER.get(flavor or "cu128", CUDA_MIN_DRIVER["cu128"])
    return drv >= minimums[os_name] if os_name in minimums else False


def normalize_arch(machine: str) -> str:
    """
    Normalize platform.machine() output across operating systems.

    Examples:
        AMD64   -> x86_64
        aarch64 -> arm64
    """
    m = machine.lower()
    if m in ("x86_64", "amd64"):
        return "x86_64"
    if m in ("aarch64", "arm64"):
        return "arm64"
    return machine


def download_file(url: str, dst: Path) -> None:
    """
    Download a file from the given URL with a progress indicator.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading: {url}", flush=True)
    print(f"       to: {dst}", flush=True)

    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(dst, "wb") as f:
                with tqdm(
                    total=total,
                    unit="B",
                    unit_scale=True,
                    desc=f"Downloading {dst.name}",
                ) as pbar:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
    except requests.RequestException as e:
        raise RuntimeError(
            f"Could not download {url} ({e}). Offline, set KONFAI_ELASTIX_DIR to an elastix-IMPACT install, or make "
            "one on a connected machine: python -m impact_reg_konfai.models.elastix_install --install-path "
            "~/.cache/konfai/elastix-impact"
        ) from e


def extract_archive(archive: Path, dst_dir: Path, keep: Callable[[str], bool] | None = None) -> None:
    """
    Extract a ZIP archive to the destination directory, only the members ``keep`` accepts when given.

    Each member is validated to resolve inside ``dst_dir`` before extraction, so a tampered archive with
    absolute or ``../`` entries cannot write outside the install root (Zip Slip).
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting: {archive} -> {dst_dir}", flush=True)
    root = dst_dir.resolve()
    with zipfile.ZipFile(archive, "r") as z:
        members = [member for member in z.namelist() if keep is None or keep(member)]
        for member in members:
            target = (root / member).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"Refusing to extract '{member}': it escapes '{root}'.")
        z.extractall(dst_dir, members=members)
    archive.unlink()


def torch_cuda_flavor() -> str | None:
    """The CUDA asset for a CUDA torch, ``None`` for a CPU torch (konfai then never runs on the GPU). The
    asset brings the LibTorch it was built against, CUDA runtime included, whenever the environment's torch
    is another one, so any CUDA torch can use it: only the driver has to be recent enough."""
    import torch

    return "cu128" if torch.version.cuda else None


def _unsupported(what: str) -> str:
    return (
        f"No elastix-IMPACT {GITHUB_TAG} build for {what}: set KONFAI_ELASTIX_DIR to an elastix-IMPACT built for "
        "this machine (https://github.com/vboussot/ImpactElastix). The ConvexAdam and FireANTs presets do not need "
        "elastix and run here (FireANTs on Linux and macOS)."
    )


def _links_this_torch(install_path: Path, flavor: str) -> bool:
    """Whether the staged binary runs against the environment's pip torch: same major.minor as the LibTorch
    the asset was built against, and a probe that passes (a CPU torch has no libtorch_cuda for cu128)."""
    import torch

    if torch.__version__.split("+")[0].split(".")[:2] != ASSET_LIBTORCH[flavor].split(".")[:2]:
        return False
    try:
        try_elastix(install_path)
    except RuntimeError:
        return False
    return True


def _install_libtorch(install_path: Path, os_name: str, flavor: str) -> None:
    """The LibTorch the asset was built against, its shared libraries only, under ``libtorch/lib``."""
    archive = install_path / "libtorch.zip"
    download_file(libtorch_url(os_name, flavor), archive)
    extract_archive(
        archive,
        install_path,
        keep=lambda name: name.startswith("libtorch/lib/") and (".so" in name or name.endswith((".dll", ".dylib"))),
    )
    lib = install_path / "libtorch" / "lib"
    cudart = next(lib.glob("libcudart-*.so.12"), None)
    if cudart is not None:
        # The Linux CUDA binary needs libcudart.so.12; LibTorch carries it under a hashed name.
        (lib / "libcudart.so.12").symlink_to(cudart.name)


def _swap(staged: Path, install_path: Path) -> None:
    """Put ``staged`` where ``install_path`` is, by renames: the install in place is only retired once its
    replacement runs, and comes back if the replacement cannot be moved in."""
    retired = staged.with_name(staged.name + ".retired")
    if install_path.exists():
        install_path.rename(retired)
    try:
        staged.rename(install_path)
    except OSError:
        if retired.exists() and not install_path.exists():
            retired.rename(install_path)
        raise
    shutil.rmtree(retired, ignore_errors=True)


def install_elastix_impact(install_path: Path, force_cuda: bool, force_cpu: bool) -> None:
    """Install the elastix-IMPACT asset of this machine at ``install_path``, with the LibTorch it was built
    against when the environment's torch is another one.

    The install is built in a directory beside ``install_path`` and replaces it only once ``elastix -h`` runs
    from there: a failed download or a binary that cannot load never costs the install in place.
    """
    os_name = platform.system()
    arch = normalize_arch(platform.machine())
    has_nvidia, drv = detect_nvidia_driver()

    wanted = torch_cuda_flavor()
    flavor = "cpu"
    if force_cuda:
        if not has_nvidia or not driver_ok_for_cuda(os_name, drv, "cu128"):
            raise RuntimeError(
                f"CUDA forced but NVIDIA driver/GPU not suitable for cu128. Detected: "
                f"has_nvidia={has_nvidia}, driver={drv}"
            )
        flavor = "cu128"
    elif not force_cpu and has_nvidia and wanted is not None:
        if driver_ok_for_cuda(os_name, drv, wanted):
            flavor = wanted
        else:
            print(
                f"The NVIDIA driver {drv} is older than the {CUDA_MIN_DRIVER[wanted].get(os_name)} the "
                f"{wanted} asset needs. Installing the CPU asset.",
                flush=True,
            )

    print(f"System: {os_name} {arch}", flush=True)
    print(f"NVIDIA: {has_nvidia}, driver={drv}", flush=True)
    print(f"Selected flavor: {flavor}", flush=True)

    key = (os_name, arch, flavor)
    if key not in ELX_ASSET_TEMPLATE:
        raise RuntimeError(_unsupported(f"{os_name}/{arch} ({flavor})"))

    install_path = install_path.resolve()
    install_path.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=f"{install_path.name}.", dir=install_path.parent))
    try:
        elx_asset = ELX_ASSET_TEMPLATE[key]
        elx_url = f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}/releases/download/{GITHUB_TAG}/{elx_asset}"
        download_file(elx_url, staged / elx_asset)
        with (staged / elx_asset).open("rb") as archive:
            digest = hashlib.file_digest(archive, "sha256").hexdigest()
        if elx_asset in ASSET_SHA256 and digest != ASSET_SHA256[elx_asset]:
            raise RuntimeError(
                f"{elx_url} does not match the sha256 of the {GITHUB_TAG} release ({digest}, expected "
                f"{ASSET_SHA256[elx_asset]}): refusing to install it."
            )
        extract_archive(staged / elx_asset, staged)

        # ZIP archives may drop executable permissions.
        if os_name in ("Linux", "Darwin"):
            for exe in ("elastix", "transformix"):
                p = staged / "bin" / exe
                if p.exists():
                    p.chmod(p.stat().st_mode | stat.S_IEXEC)
        (staged / FLAVOR_FILE).write_text(flavor, encoding="utf-8")

        if not _links_this_torch(staged, flavor):
            print(
                f"The {elx_asset} binary needs LibTorch {ASSET_LIBTORCH[flavor]}, this environment has torch "
                "of another version: installing that LibTorch beside it.",
                flush=True,
            )
            _install_libtorch(staged, os_name, flavor)
            try_elastix(staged)
        _swap(staged, install_path)
    finally:
        shutil.rmtree(staged, ignore_errors=True)


#: Where an install records the asset it holds: a CPU build answers ``-h`` as a CUDA one does.
FLAVOR_FILE = "FLAVOR"


def installed_flavor(install_path: Path) -> str | None:
    """The flavor the install at ``install_path`` recorded, None when it recorded none."""
    try:
        return (install_path / FLAVOR_FILE).read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def cuda_upgrade_available(install_path: Path) -> bool:
    """Whether the release carries the CUDA build this environment can use, while the install holds the CPU one.

    A CPU install passes ``try_elastix``. Only a CUDA torch, a driver recent enough for that asset and an install
    recorded as another flavor lead to the check, one HEAD request; offline, the install in place is kept. An install
    that recorded nothing is left as it is: it may already be the CUDA build.
    """
    wanted = torch_cuda_flavor()
    if wanted is None or installed_flavor(install_path) in (None, wanted):
        return False
    os_name, arch = platform.system(), normalize_arch(platform.machine())
    asset = ELX_ASSET_TEMPLATE.get((os_name, arch, wanted))
    has_nvidia, drv = detect_nvidia_driver()
    if asset is None or not has_nvidia or not driver_ok_for_cuda(os_name, drv, wanted):
        return False
    url = f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}/releases/download/{GITHUB_TAG}/{asset}"
    try:
        return requests.head(url, allow_redirects=True, timeout=10).status_code == 200
    except requests.RequestException:
        return False


def get_elastix_bin(install_path: Path) -> Path:
    return install_path / ("elastix.exe" if platform.system() == "Windows" else (Path("bin") / "elastix"))


#: What a child answers when the loader cannot find a library or a symbol it needs: 127 on POSIX, Windows
#: STATUS_DLL_NOT_FOUND and STATUS_ENTRYPOINT_NOT_FOUND, each unsigned or signed as Python reports it.
_LOADER_FAILURE_CODES = (127, 0xC0000135, -1073741515, 0xC0000139, -1073741511)


def loader_env(install_path: Path) -> dict[str, str]:
    """The environment the elastix binary needs to link its shared libraries.

    The LibTorch the installer put under ``libtorch/lib`` comes first, when the environment's pip ``torch``
    is not the version the asset was built against; then the install's own ``lib/``, the pip torch's LibTorch
    and anything ``KONFAI_ELASTIX_EXTRA_LIB`` names. The Windows asset keeps its DLLs next to the executable,
    so the install root is searched too.
    """
    import torch

    searched = [
        str(install_path / "libtorch" / "lib"),
        str(install_path / "lib"),
        str(install_path),
        str(Path(torch.__file__).resolve().parent / "lib"),
        os.environ.get("KONFAI_ELASTIX_EXTRA_LIB", ""),
    ]
    variable = {"Windows": "PATH", "Darwin": "DYLD_LIBRARY_PATH"}.get(platform.system(), "LD_LIBRARY_PATH")
    env = os.environ.copy()
    env[variable] = os.pathsep.join(path for path in [*searched, env.get(variable, "")] if path)
    return env


def try_elastix(install_path: Path) -> None:
    """Run the install once, under the loader path a registration uses, so a binary that cannot link
    fails here instead of mid-case. Every symbol is bound at load, so a LibTorch missing one function fails here."""
    env = {**loader_env(install_path), "LD_BIND_NOW": "1", "DYLD_BIND_AT_LAUNCH": "1"}
    try:
        subprocess.run(
            [str(get_elastix_bin(install_path)), "-h"],
            capture_output=True,
            text=True,
            check=True,
            env=env,
        )  # nosec B603
    except subprocess.CalledProcessError as e:
        msg = "Elastix execution failed.\n\n"

        msg += f"Command:\n{' '.join(e.cmd)}\n"
        msg += f"Return code: {e.returncode}\n\n"

        if e.returncode in _LOADER_FAILURE_CODES:
            # A library the loader cannot find aborts a child that did exec: never OSError.
            msg += (
                "A shared library could not be found or lacks a symbol. The binary links the LibTorch it was "
                "built against: the installer's own under libtorch/lib, the environment's pip `torch` when it "
                "is that version, or what KONFAI_ELASTIX_EXTRA_LIB names for a build of your own.\n\n"
            )
        if e.stderr:
            msg += "Error output:\n"
            msg += e.stderr.strip()
        raise RuntimeError(msg) from e

    except OSError as e:
        msg = (
            "Elastix could not be started.\n\n"
            f"The binary at '{get_elastix_bin(install_path)}' is missing or not executable.\n\n"
            f"System error:\n{e!s}"
        )

        raise RuntimeError(msg) from e


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--install-path", type=Path, default=DEFAULT_PREFIX, help="Install directory (default: ./elastix-impact)"
    )
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--force-cpu", action="store_true", help="Force CPU install even if NVIDIA present")
    group.add_argument("--force-cuda", action="store_true", help="Force CUDA install (fails if no suitable driver)")
    args = vars(ap.parse_args())
    install_elastix_impact(**args)


if __name__ == "__main__":
    main()
