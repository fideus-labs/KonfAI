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

"""What the engines of external registration tools share: the authors' code or release fetched once at its pin,
the conventions of the benchmark's field, and ``ToolNet``, the graph that hands a pair to such an engine.

An engine is called ``engine(fixed, moving, fixed_mask, moving_mask, device)``: SimpleITK images, a mask ``None``
where none restricts the pair, ``device`` the CUDA index or -1. It returns the field, channel-first, in mm along the
world (ITK, LPS) axes, on the fixed image's grid, such that the fixed point ``x`` matches ``x + u(x)`` in the moving
image.

NOTE: do NOT add ``from __future__ import annotations``: KonfAI's config engine relies on runtime-evaluated
annotations (``get_origin``); PEP 563 stringized annotations break binding.
"""

import hashlib
import os
import shutil
import subprocess  # nosec B404
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from konfai.network import network

from .intensity import EngineRegistration

Engine = Callable[[sitk.Image, sitk.Image, sitk.Image | None, sitk.Image | None, int], np.ndarray]


def cache_dir() -> Path:
    """Where the tools' code and binaries live once fetched: ``$IMPACT_REG_TOOLS_CACHE``, or
    ``~/.cache/konfai/impact-reg`` beside elastix's."""
    return Path(os.environ.get("IMPACT_REG_TOOLS_CACHE", Path.home() / ".cache" / "konfai" / "impact-reg"))


def pinned_source(name: str, url: str, commit: str) -> Path:
    """The authors' repository at ``commit``, cloned once into the cache: a baseline runs the code it was
    measured with, never a moving branch."""
    target = cache_dir() / f"{name}-{commit[:12]}"
    if not target.exists():
        part = target.with_name(target.name + ".part")  # in place only once at the commit
        shutil.rmtree(part, ignore_errors=True)
        part.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--quiet", url, str(part)], check=True)  # nosec B603 B607
        subprocess.run(["git", "-C", str(part), "checkout", "--quiet", commit], check=True)  # nosec B603 B607
        part.rename(target)
    return target


def fetch(url: str, target: Path, sha256: str | None = None) -> Path:
    """``url`` downloaded once to ``target`` and, when a hash is given, checked against it."""
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_name(target.name + ".part")
        urllib.request.urlretrieve(url, part)  # nosec B310 - the authors' release, checked below
        with part.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if sha256 and digest != sha256:
            part.unlink()
            raise RuntimeError(f"{url} does not match its recorded hash")
        part.rename(target)
    return target


def pinned_archive(name: str, url: str, sha256: str) -> Path:
    """A release archive at ``url``, downloaded once into the cache, checked against ``sha256`` and unpacked: the
    binaries a baseline was measured with."""
    target = cache_dir() / f"{name}-{sha256[:12]}"
    if not (target / ".unpacked").exists():
        archive = fetch(url, target / "archive", sha256)
        if zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as packed:
                packed.extractall(target)  # nosec B202 - our own pinned archive
                for entry in packed.infolist():  # zipfile drops the modes: the binaries must stay executable
                    if mode := (entry.external_attr >> 16) & 0o777:
                        (target / entry.filename).chmod(mode)
        else:
            shutil.unpack_archive(str(archive), str(target), "gztar")
        archive.unlink()
        (target / ".unpacked").touch()
    return target


def pinned_environment(root: Path, python: str, installs: list[list[str]]) -> Path:
    """A virtual environment at ``root/venv`` on its own Python, made once by uv (``pip install uv``): for the tools
    whose pins do not run on the host's Python. ``installs`` are ``pip install`` argument lists, run in order."""
    try:
        import uv

        command = uv.find_uv_bin()
    except ImportError:
        command = shutil.which("uv") or ""
    if not command:
        raise RuntimeError("this baseline builds its own Python environment with uv: pip install uv")
    venv = root / "venv"
    if not (venv / ".ready").exists():
        subprocess.run([command, "venv", "--quiet", "--clear", "--python", python, str(venv)], check=True)  # nosec B603
        for arguments in installs:
            subprocess.run(  # nosec B603
                [
                    command,
                    "pip",
                    "install",
                    "--quiet",
                    "--no-cache",
                    "--python",
                    str(venv / "bin" / "python"),
                    *arguments,
                ],
                check=True,
            )
        (venv / ".ready").touch()
    return venv


def built_source(name: str, url: str, commit: str, build: Callable[[Path], None]) -> Path:
    """The authors' repository at ``commit``, built once by ``build`` (a marker says it is done)."""
    source = pinned_source(name, url, commit)
    if not (source / ".built").exists():
        build(source)
        (source / ".built").touch()
    return source


def gpu_environment(device: int, **extra: str) -> dict[str, str]:
    """The environment of a tool's subprocess on the CUDA device ``device`` of this process (-1: none): the physical
    GPU behind that index, since the subprocess reads CUDA_VISIBLE_DEVICES afresh."""
    from konfai import cuda_visible_devices

    visible = str(cuda_visible_devices()[device]) if device >= 0 else ""
    return {**os.environ, **extra, "CUDA_VISIBLE_DEVICES": visible}


def field_from_voxels(displacement: np.ndarray, fixed: sitk.Image) -> np.ndarray:
    """A displacement in voxels along the index axes, ``[i, j, k, D]``, as the benchmark's field ``[D, k, j, i]``
    in mm along the world axes: ``u = D S d``, with ``D`` the direction and ``S`` the spacing of the fixed image."""
    n = fixed.GetDimension()
    to_world = np.array(fixed.GetDirection()).reshape(n, n) @ np.diag(fixed.GetSpacing())
    world = displacement.astype(np.float64) @ to_world.T  # [i, j, k, D]
    return np.ascontiguousarray(np.moveaxis(world, -1, 0).transpose(0, *reversed(range(1, n + 1))), dtype=np.float32)


def scalar(image: sitk.Image) -> sitk.Image:
    """The image a tool comparing one value a voxel registers: a colour or multiband image as the mean of its
    components, float32; a scalar image only cast."""
    n = image.GetNumberOfComponentsPerPixel()
    if n == 1:
        return sitk.Cast(image, sitk.sitkFloat32)
    return sitk.Cast(
        sum(sitk.VectorIndexSelectionCast(image, c, sitk.sitkFloat32) for c in range(n)) / n, sitk.sitkFloat32
    )


def to_index_array(image: sitk.Image) -> np.ndarray:
    """The image as a ``[i, j, k]`` array, the index axes in ITK's order, which the tools indexed that way read."""
    return np.ascontiguousarray(sitk.GetArrayFromImage(image).transpose(*reversed(range(image.GetDimension()))))


class _Called:
    """``engine.register`` as EngineRegistration calls it, over an engine called as a function. konfai-apps fills an
    absent mask with ones: a mask of ones restricts nothing and reaches the engine as None."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def register(
        self,
        fixed: sitk.Image,
        moving: sitk.Image,
        device_index: int,
        fixed_mask: sitk.Image | None = None,
        moving_mask: sitk.Image | None = None,
    ) -> np.ndarray:
        masks = [
            None if mask is None or bool((sitk.GetArrayViewFromImage(mask) == 1).all()) else mask
            for mask in (fixed_mask, moving_mask)
        ]
        return self._engine(fixed, moving, masks[0], masks[1], device_index)


class ToolNet(network.Network):
    """An external tool's registration as an ``add_module`` graph: fixed = branch 0, moving = branch 1, the masks
    branches 2 and 3. The one output is ``DisplacementField``, on the fixed grid (the predictor attaches its geometry:
    ``same_as_group`` in the app's ``Prediction.yml``)."""

    #: Its output is a displacement field in world units: KonfAI blends it in float32, not float16.
    full_precision_outputs = True

    def __init__(
        self,
        engine: Engine,
        dim: int,
        optimizer: network.OptimizerLoader,
        schedulers: dict[str, network.LRSchedulersLoader],
        outputs_criterions: dict[str, network.TargetCriterionsLoader],
    ) -> None:
        super().__init__(
            in_channels=1, optimizer=optimizer, schedulers=schedulers, outputs_criterions=outputs_criterions, dim=dim
        )
        self.add_module(
            "Registration", EngineRegistration(_Called(engine)), in_branch=[0, 1, 2, 3], out_branch=["field"]
        )
        self.add_module("DisplacementField", torch.nn.Identity(), in_branch=["field"], out_branch=["dvf"])
