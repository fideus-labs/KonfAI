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
import os
import platform
import re
import shutil
import stat
import subprocess  # nosec B404
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Elastix + IMPACT binary assets hosted on GitHub Releases.
#
# Key format: (OS, ARCH, FLAVOR)
#   - OS     : platform.system() -> "Linux", "Windows", "Darwin"
#   - ARCH   : normalized architecture -> "x86_64"
#   - FLAVOR : "cpu", "cu128" or "cu130"
#
# No asset bundles LibTorch: both flavors link it from the environment's pip ``torch`` (loader_env).
# The flavors differ in linkage, a CUDA one additionally needing libtorch_cuda and the CUDA runtime.
# The CUDA flavor has to match the CUDA the environment's torch was built against: a CUDA 13 torch
# ships no CUDA 12 runtime, so the cu128 binary cannot load beside it, and the reverse holds too.
# -----------------------------------------------------------------------------
ELX_ASSET_TEMPLATE = {
    ("Linux", "x86_64", "cpu"): "elastix-impact-linux-x86_64-cpu.zip",
    ("Linux", "x86_64", "cu128"): "elastix-impact-linux-x86_64-cu128.zip",
    ("Linux", "x86_64", "cu130"): "elastix-impact-linux-x86_64-cu130.zip",
    ("Windows", "x86_64", "cpu"): "elastix-impact-windows-x86_64-cpu.zip",
    ("Windows", "x86_64", "cu128"): "elastix-impact-windows-x86_64-cu128.zip",
    ("Windows", "x86_64", "cu130"): "elastix-impact-windows-x86_64-cu130.zip",
    ("Darwin", "x86_64", "cpu"): "elastix-impact-macos-14-x86_64-cpu.zip",
}

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
    floor is used, which is what every caller asked for before a CUDA 13 asset existed.
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
    except Exception as e:
        raise e


def extract_archive(archive: Path, dst_dir: Path) -> None:
    """
    Extract a ZIP archive to the destination directory.

    Each member is validated to resolve inside ``dst_dir`` before extraction, so a tampered archive with
    absolute or ``../`` entries cannot write outside the install root (Zip Slip).
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting: {archive} -> {dst_dir}", flush=True)
    root = dst_dir.resolve()
    with zipfile.ZipFile(archive, "r") as z:
        for member in z.namelist():
            target = (root / member).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"Refusing to extract '{member}': it escapes '{root}'.")
        z.extractall(dst_dir)
    archive.unlink()


_NO_CUDA_ASSET = (
    "No elastix-IMPACT asset is published for the CUDA this environment's torch was built against, and a "
    "binary linking another CUDA cannot load beside it. For the GPU, point KONFAI_ELASTIX_DIR at an "
    "elastix-IMPACT built against this torch."
)


def torch_cuda_flavor() -> str | None:
    """The CUDA asset the environment's torch can load, ``None`` for a CPU torch or one built for another
    CUDA: no asset bundles LibTorch, so the binary finds the CUDA runtime where torch keeps its own."""
    import torch

    cuda = torch.version.cuda
    if cuda is None:
        return None
    return {"12": "cu128", "13": "cu130"}.get(cuda.split(".")[0])


def install_elastix_impact(install_path: Path, force_cuda: bool, force_cpu: bool):
    os_name = platform.system()
    arch = normalize_arch(platform.machine())
    has_nvidia, drv = detect_nvidia_driver()

    if os_name not in ("Linux", "Windows", "Darwin"):
        raise NameError(f"Unsupported OS: {os_name}")

    if arch not in ("x86_64", "arm64"):
        raise NameError(f"Unsupported arch: {arch} (expected x86_64, arm64)")

    # The asset the environment's torch can load at all, then whether the driver is recent enough for
    # the CUDA that asset links -- a CUDA 13 binary needs a newer driver than a CUDA 12 one.
    wanted = torch_cuda_flavor()
    flavor = "cpu"
    if force_cuda:
        if wanted is None:
            raise NameError(_NO_CUDA_ASSET)
        if not has_nvidia or not driver_ok_for_cuda(os_name, drv, wanted):
            raise NameError(
                f"CUDA forced but NVIDIA driver/GPU not suitable for {wanted}. Detected: "
                f"has_nvidia={has_nvidia}, driver={drv}"
            )
        flavor = wanted
    elif not force_cpu and has_nvidia:
        if wanted is None:
            print(f"{_NO_CUDA_ASSET} Installing the CPU asset.", flush=True)
        elif not driver_ok_for_cuda(os_name, drv, wanted):
            print(
                f"The NVIDIA driver {drv} is older than the {CUDA_MIN_DRIVER[wanted][os_name]} the "
                f"{wanted} asset needs. Installing the CPU asset.",
                flush=True,
            )
        else:
            flavor = wanted

    print(f"System: {os_name} {arch}", flush=True)
    print(f"NVIDIA: {has_nvidia}, driver={drv}", flush=True)
    print(f"Selected flavor: {flavor}", flush=True)

    key = (os_name, arch, flavor)
    if key not in ELX_ASSET_TEMPLATE:
        raise NameError(f"No elastix asset configured for {key}")

    install_path = install_path.resolve()
    install_path.mkdir(parents=True, exist_ok=True)

    elx_asset = ELX_ASSET_TEMPLATE[key]
    elx_url = f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}/releases/download/{GITHUB_TAG}/{elx_asset}"
    elx_archive = install_path / elx_asset
    try:
        download_file(elx_url, elx_archive)
    except Exception as failure:
        # A CUDA flavor this release does not carry yet: the CPU asset still registers, and a run says
        # what it would take to use the card. Forced CUDA has no fallback to fall back to.
        if flavor == "cpu" or force_cuda:
            raise
        print(f"{GITHUB_TAG} carries no {elx_asset} ({failure}). Installing the CPU asset.", flush=True)
        flavor = "cpu"
        elx_asset = ELX_ASSET_TEMPLATE[(os_name, arch, "cpu")]
        elx_url = f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}/releases/download/{GITHUB_TAG}/{elx_asset}"
        elx_archive = install_path / elx_asset
        download_file(elx_url, elx_archive)
    # Extracting over a previous install keeps whatever the new asset does not overwrite. An older asset
    # bundled its own LibTorch under lib/, which then shadowed the environment's torch on the loader
    # path: a CUDA build ran CPU-only, and a build of another torch failed to link at all.
    for stale in ("bin", "lib", "third_party"):
        shutil.rmtree(install_path / stale, ignore_errors=True)
    extract_archive(elx_archive, install_path)

    # -------------------------------------------------------------------------
    # ZIP archives may drop executable permissions.
    # Ensure elastix and transformix are executable on Unix platforms.
    # -------------------------------------------------------------------------
    if os_name in ("Linux", "Darwin"):
        for exe in ("elastix", "transformix"):
            p = install_path / "bin" / exe
            if p.exists():
                p.chmod(p.stat().st_mode | stat.S_IEXEC)

    # LibTorch comes from the environment's pip ``torch`` at runtime (elastix_engine.py adds torch's lib/ dir
    # to the loader path): the same LibTorch the elastix asset is built against in CI, so elastix and the
    # rest of the stack stay on one torch. The elastix asset is the only download here.


def get_elastix_bin(install_path: Path) -> Path:
    return install_path / ("elastix.exe" if platform.system() == "Windows" else (Path("bin") / "elastix"))


#: What a child answers when the loader cannot find a library it needs: 127 on POSIX, Windows
#: STATUS_DLL_NOT_FOUND either way round, since Python reports it unsigned or signed by platform.
_LOADER_FAILURE_CODES = (127, 0xC0000135, -1073741515)


def loader_env(install_path: Path) -> dict[str, str]:
    """The environment the elastix binary needs to link its shared libraries.

    LibTorch comes from the environment's pip ``torch`` (the LibTorch the asset is built against in
    CI), beside the install's own ``lib/`` and anything ``KONFAI_ELASTIX_EXTRA_LIB`` names. The
    Windows asset keeps its DLLs next to the executable, so the install root is searched too.
    """
    import torch

    searched = [
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
    fails here instead of mid-case."""
    try:
        subprocess.run(
            [str(get_elastix_bin(install_path)), "-h"],
            capture_output=True,
            text=True,
            check=True,
            env=loader_env(install_path),
        )  # nosec B603
    except subprocess.CalledProcessError as e:
        msg = "Elastix execution failed.\n\n"

        msg += f"Command:\n{' '.join(e.cmd)}\n"
        msg += f"Return code: {e.returncode}\n\n"

        if e.returncode in _LOADER_FAILURE_CODES:
            # A library the loader cannot find aborts a child that did exec: never OSError.
            msg += (
                "A shared library could not be found. The binary links LibTorch from the "
                "environment's pip `torch`, so either no torch is installed or its version is not "
                "the one elastix was built against.\n\n"
            )
        if e.stderr:
            msg += "Error output:\n"
            msg += e.stderr.strip()
        raise NameError(msg) from e

    except OSError as e:
        msg = (
            "Elastix could not be started.\n\n"
            f"The binary at '{get_elastix_bin(install_path)}' is missing or not executable.\n\n"
            f"System error:\n{e!s}"
        )

        raise NameError(msg) from e


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
