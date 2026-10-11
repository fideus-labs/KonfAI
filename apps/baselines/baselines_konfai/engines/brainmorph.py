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

"""BrainMorph (Wang et al.), called as the LUMIR organisers call it in Learn2Reg 2024
(``LUMIR_L2R/BrainMorph/infer_BrainMorph.py``): 512 keypoints, the "H" model, thin-plate splines with no
regularisation. The authors' repository at the commit of that date, since later commits dropped the H model, with its
dependencies at versions of that date (torch 2.3 on Python 3.11) in an environment uv builds in the cache on first use,
and the weights checked against their hash. Brain MR only. Masks and modalities are accepted and ignored.

BrainMorph brings each image to RAS, 1 mm and a centred 256^3 box. Both images are put on one such box around the fixed
image first (the pairs are aligned), so that the saved ``grid_sample`` grid of the fixed box is a field there, then
resampled onto the fixed grid.

NOTE: do NOT add ``from __future__ import annotations``: KonfAI's config engine relies on runtime-evaluated
annotations (``get_origin``); PEP 563 stringized annotations break binding.
"""

import glob
import hashlib
import os
import subprocess  # nosec B404
import tempfile
from pathlib import Path
from typing import Annotated

import numpy as np
import SimpleITK as sitk
from konfai.network import network

from .base import BaselineNet, pinned_environment, pinned_source

REPOSITORY = "https://github.com/alanqrwang/brainmorph"
COMMIT = "106b029d6048d3d9d2f73cf7996a0b0813cbc4ae"  # 2024-05-23, "gdown fix and switching to variants"
# what its setup.py asks for, and what keymorph imports without declaring, at versions of that date
PACKAGES = [
    "torch==2.3.1",
    "torchvision==0.18.1",
    "numpy==1.26.4",
    "keymorph==1.0.1",
    "torchio==0.19.6",
    "nibabel==5.2.1",
    "scipy==1.13.1",
    "scikit-learn==1.5.0",
    "pandas==2.2.2",
    "pytz==2024.1",
    "ogb==1.3.6",
    "outdated==0.2.2",
    "matplotlib==3.9.0",
    "scikit-image==0.24.0",
    "h5py==3.11.0",
    "pyyaml==6.0.1",
    "SimpleITK==2.5.6",
    "gdown==5.2.0",
]
WEIGHTS = (
    "foundation-numkey512-numlevels7.pth.tar",
    "https://drive.google.com/uc?id=1o_XG1FL4O3rsoCZvH2gzbzsCzd79uKnD",
    "ae8947ea266ed1c2218c2b1449fbc4362f9b116c6afc5dfb24be0e17711da680",
)
RAS = (-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0)  # what torchio's ToCanonical leaves untouched


def home() -> tuple[Path, Path]:
    """The authors' code with its environment and weights, fetched once into the cache: (source, venv)."""
    source = pinned_source("brainmorph", REPOSITORY, COMMIT)
    venv = pinned_environment(source, "3.11", [PACKAGES, ["--no-deps", "-e", str(source)]])
    weights = source / "weights" / WEIGHTS[0]
    if not weights.exists():
        weights.parent.mkdir(exist_ok=True)
        subprocess.run([str(venv / "bin" / "gdown"), "--quiet", WEIGHTS[1], "-O", str(weights)], check=True)  # nosec B603
        with weights.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != WEIGHTS[2]:
            weights.unlink()
            raise RuntimeError(f"{WEIGHTS[1]} does not match its recorded hash")
    return source, venv


def box_around(fixed: sitk.Image, extent: int) -> sitk.Image:
    """The RAS, 1 mm, ``extent``^3 box centred on the fixed image, which BrainMorph's own resampling then keeps."""
    box = sitk.Image([extent] * 3, sitk.sitkFloat32)
    box.SetDirection(RAS)
    centre = np.array(fixed.TransformContinuousIndexToPhysicalPoint([(s - 1) / 2 for s in fixed.GetSize()]))
    box.SetOrigin((centre - np.array(RAS).reshape(3, 3) @ np.full(3, (extent - 1) / 2)).tolist())
    return box


class BrainMorph:
    def __init__(self, keypoints: int, variant: str, align: str) -> None:
        self._keypoints, self._variant, self._align = keypoints, variant, align

    def __call__(
        self,
        fixed: sitk.Image,
        moving: sitk.Image,
        fixed_mask: sitk.Image | None,
        moving_mask: sitk.Image | None,
        device: int,
    ) -> np.ndarray:
        source, venv = home()
        extent = 256
        box = box_around(fixed, extent)
        environment = {**os.environ, "PYTHONPATH": str(source)}  # its scripts import scripts.*
        if device < 0:
            environment["CUDA_VISIBLE_DEVICES"] = ""
        with tempfile.TemporaryDirectory(prefix="brainmorph_") as scratch:
            work = Path(scratch)
            for name, image in (("y", fixed), ("x", moving)):  # the organisers' names: x moving, y fixed
                image = sitk.Cast(image, sitk.sitkFloat32)
                fill = float(sitk.GetArrayViewFromImage(image).min())
                sitk.WriteImage(
                    sitk.Resample(image, box, sitk.Transform(), sitk.sitkLinear, fill), str(work / f"{name}.nii.gz")
                )
            subprocess.run(  # nosec B603
                [
                    str(venv / "bin" / "python"),
                    "scripts/register.py",
                    "--num_keypoints",
                    str(self._keypoints),
                    "--variant",
                    self._variant,
                    "--weights_dir",
                    f"{source / 'weights'}/",
                    "--moving",
                    str(work / "x.nii.gz"),
                    "--fixed",
                    str(work / "y.nii.gz"),
                    "--list_of_aligns",
                    self._align,
                    "--list_of_metrics",
                    "mse",
                    "--save_eval_to_disk",
                    "--save_dir",
                    str(work / "bm"),
                ],
                cwd=source,
                check=True,
                env=environment,
            )
            pattern = f"{work}/bm/eval_numkey{self._keypoints}_variant{self._variant}/*/grid_0-fixed_0-moving-rot0-{self._align}.npy"
            grid = np.load(glob.glob(pattern)[0])
        # the grid_sample grid of the fixed box, (i, j, k, 3) normalised with align_corners, its axes reversed
        index = (np.squeeze(grid)[..., [2, 1, 0]] + 1) / 2 * (extent - 1)
        d = index - np.stack(np.meshgrid(*[np.arange(extent)] * 3, indexing="ij"), -1)  # voxels of the box
        u = d * np.array([-1.0, -1.0, 1.0])  # 1 mm voxels along R, A, S: mm along L, P, S
        field = sitk.GetImageFromArray(np.ascontiguousarray(u.transpose(2, 1, 0, 3)).astype(np.float64), isVector=True)
        field.CopyInformation(box)
        field = sitk.Resample(field, fixed, sitk.Transform(), sitk.sitkLinear, 0.0, sitk.sitkVectorFloat64)
        return np.ascontiguousarray(np.moveaxis(sitk.GetArrayFromImage(field), -1, 0), dtype=np.float32)


class RegistrationNet(BaselineNet):
    """BrainMorph, the LUMIR organisers' call. Brain MR."""

    def __init__(
        self,
        optimizer: network.OptimizerLoader = network.OptimizerLoader(),
        schedulers: dict[str, network.LRSchedulersLoader] = {
            "default:ReduceLROnPlateau": network.LRSchedulersLoader(0)
        },
        outputs_criterions: dict[str, network.TargetCriterionsLoader] = {"default": network.TargetCriterionsLoader()},
        keypoints: Annotated[int, "register.py's --num_keypoints: 128, 256 or 512."] = 512,
        variant: Annotated[str, "register.py's --variant: S, M, L or H."] = "H",
        align: Annotated[
            str, "register.py's --list_of_aligns: tps_0 for thin-plate splines without regularisation, affine, rigid."
        ] = "tps_0",
    ) -> None:
        super().__init__(BrainMorph(keypoints, variant, align), 3, optimizer, schedulers, outputs_criterions)
