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

"""What every baseline app shares: the authors' code fetched at its pin, the graph module that hands a pair to an
engine and gets a displacement field back, and the field's convention.

The field is the benchmark's: one component per spatial axis, in mm along the world (ITK, LPS) axes, on the fixed
image's grid, such that the fixed point ``x`` matches ``x + u(x)`` in the moving image. The predictor attaches the
fixed image's geometry to the output (``same_as_group: Volume_0:Fixed`` in the app's ``Prediction.yml``).

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
from konfai.utils.dataset import Attribute, data_to_image

#: An engine: (fixed, moving, fixed mask or None, moving mask or None, CUDA device index or -1) -> the field,
#: ``[D, *spatial]`` in mm along the world axes on the fixed grid.
Engine = Callable[[sitk.Image, sitk.Image, sitk.Image | None, sitk.Image | None, int], np.ndarray]


def cache_dir() -> Path:
    """Where the authors' code and binaries live, once fetched: ``$BASELINES_KONFAI_CACHE`` or
    ``~/.cache/baselines-konfai``."""
    return Path(os.environ.get("BASELINES_KONFAI_CACHE", Path.home() / ".cache" / "baselines-konfai"))


def pinned_source(name: str, url: str, commit: str) -> Path:
    """The authors' repository at ``commit``, cloned once into the cache: a baseline runs the code it was
    measured with, never a moving branch."""
    target = cache_dir() / f"{name}-{commit[:12]}"
    if not (target / ".git").exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--quiet", url, str(target)], check=True)  # nosec B603 B607
        subprocess.run(["git", "-C", str(target), "checkout", "--quiet", commit], check=True)  # nosec B603 B607
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


def field_from_voxels(displacement: np.ndarray, fixed: sitk.Image) -> np.ndarray:
    """A displacement in voxels along the index axes, ``[i, j, k, D]``, as the benchmark's field ``[D, k, j, i]``
    in mm along the world axes: ``u = D S d``, with ``D`` the direction and ``S`` the spacing of the fixed image."""
    n = fixed.GetDimension()
    to_world = np.array(fixed.GetDirection()).reshape(n, n) @ np.diag(fixed.GetSpacing())
    world = displacement.astype(np.float64) @ to_world.T  # [i, j, k, D]
    return np.ascontiguousarray(np.moveaxis(world, -1, 0).transpose(0, *reversed(range(1, n + 1))), dtype=np.float32)


def to_index_array(image: sitk.Image) -> np.ndarray:
    """The image as a ``[i, j, k]`` array, the index axes in ITK's order, which the tools indexed that way read."""
    return np.ascontiguousarray(sitk.GetArrayFromImage(image).transpose(*reversed(range(image.GetDimension()))))


class Registration(torch.nn.Module):
    """Graph module: the pair's tensors and their geometry in, the field out, through the engine, one sample at
    a time. ``accepts_attributes = True`` opts it into the per-branch ``Attribute`` list (the convention of
    ``CriterionWithAttribute``): registration needs the physical geometry."""

    accepts_attributes = True

    def __init__(self, engine: Engine) -> None:
        super().__init__()
        self._engine = engine

    def forward(
        self,
        fixed: torch.Tensor,
        moving: torch.Tensor,
        fixed_mask: torch.Tensor,
        moving_mask: torch.Tensor,
        attributes: list[list[Attribute]],
    ) -> torch.Tensor:
        fixed_attrs, moving_attrs, fixed_mask_attrs, moving_mask_attrs = attributes
        device_index = fixed.device.index if fixed.device.type == "cuda" else -1
        fields = []
        # the engines optimise with autograd; the predictor calls forward under inference mode
        with torch.inference_mode(False), torch.enable_grad():
            for b in range(fixed.shape[0]):
                images = [
                    data_to_image(t[b].detach().cpu().numpy(), attrs[b])
                    for t, attrs in ((fixed, fixed_attrs), (moving, moving_attrs))
                ]
                masks = [
                    data_to_image(t[b].detach().cpu().numpy(), attrs[b]) if not bool((t[b] == 1).all()) else None
                    for t, attrs in ((fixed_mask, fixed_mask_attrs), (moving_mask, moving_mask_attrs))
                ]
                fields.append(torch.from_numpy(self._engine(images[0], images[1], masks[0], masks[1], device_index)))
        return torch.stack(fields, dim=0).to(fixed.device)


class BaselineNet(network.Network):
    """A baseline as an ``add_module`` graph: fixed = branch 0, moving = branch 1, the masks branches 2 and 3 (a
    mask of ones means none). The one output is ``DisplacementField``, on the fixed grid."""

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
        self.add_module("DisplacementField", Registration(engine), in_branch=[0, 1, 2, 3], out_branch=["dvf"])
