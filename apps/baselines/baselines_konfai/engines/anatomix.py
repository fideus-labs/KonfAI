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

"""anatomix features in ConvexAdam, the ICLR 2025 pipeline of neel-dey/anatomix at a pinned commit, through its own
``convex_adam()`` with the defaults of its command-line script: 16 anatomix channels and 12 MIND-SSC channels compared
by SSD, diffusion weight 0.75, convex grid 2 with search half-width 1, 80 Adam steps on a grid of 2; CT clipped to
[-450, 450] HU as in the README. The weights come from the authors' Hugging Face repository at a pinned revision.

NOTE: do NOT add ``from __future__ import annotations``: KonfAI's config engine relies on runtime-evaluated
annotations (``get_origin``); PEP 563 stringized annotations break binding.
"""

import glob
import hashlib
import os
import sys
import tempfile
from typing import Annotated

import numpy as np
import SimpleITK as sitk
from konfai.network import network
from konfai.utils.config import Choices

from .base import BaselineNet, cache_dir, pinned_source

REPOSITORY = "https://github.com/neel-dey/anatomix"
COMMIT = "4c86aeeca5e4bf712d9e5775a01fadb37b30c13a"  # main, 2026-09-19
WEIGHTS = ("neeldey/anatomix", "anatomix.pth", "842cfe9", None)  # repo, file, revision, sha256 (checked when known)
CT_WINDOW = (-450.0, 450.0)
SETTINGS = {
    "hf_variant": "anatomix",
    "lambda_weight": 0.75,
    "grid_sp": 2,
    "disp_hw": 1,
    "selected_niter": 80,
    "selected_smooth": 0,
    "grid_sp_adam": 2,
    "ic": True,
    "use_mask": False,
}


def to_field(displacement: np.ndarray, fixed: sitk.Image) -> np.ndarray:
    """The backend's displacement, ``(I, J, K, 3)`` on the even-padded fixed grid in voxels, as the field on the
    unpadded fixed grid in mm along the world axes: the backend's grid_sample convention scales by n / (n - 1)."""
    n = np.array(displacement.shape[:3], float)
    d = (displacement * (n / (n - 1)))[tuple(slice(s) for s in fixed.GetSize())]
    u = (d * fixed.GetSpacing()) @ np.reshape(fixed.GetDirection(), (3, 3)).T
    return np.ascontiguousarray(np.moveaxis(u.transpose(2, 1, 0, 3), -1, 0), dtype=np.float32)


def weights_home() -> str:
    """A Hugging Face home of our own, with the authors' weights at the pinned revision and refs/main pointing at
    it: the pipeline loads them with ``revision=None`` and would otherwise follow their main branch."""
    from huggingface_hub import hf_hub_download

    home = cache_dir() / "anatomix" / "hf_home"
    repo, filename, revision, digest = WEIGHTS
    os.environ.setdefault("HF_HOME", str(home))
    path = hf_hub_download(repo, filename, revision=revision, cache_dir=str(home / "hub"))
    if digest and hashlib.sha256(open(path, "rb").read()).hexdigest() != digest:
        raise RuntimeError("the anatomix weights do not match their published hash")
    refs = home / "hub" / f"models--{repo.replace('/', '--')}" / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    (refs / "main").write_text(revision)
    return str(home)


class AnatomixConvexAdam:
    def __init__(self, fixed_modality: str, moving_modality: str) -> None:
        self._modalities = (fixed_modality, moving_modality)

    def __call__(
        self,
        fixed: sitk.Image,
        moving: sitk.Image,
        fixed_mask: sitk.Image | None,
        moving_mask: sitk.Image | None,
        device: int,
    ) -> np.ndarray:
        import nibabel as nib

        source = pinned_source("anatomix", REPOSITORY, COMMIT)
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
        os.environ["HF_HOME"] = weights_home()
        os.environ["HF_HUB_OFFLINE"] = "1"
        from anatomix.registration.registration_backend.convexadam import convex_adam

        fixed, moving = sitk.Cast(fixed, sitk.sitkFloat32), sitk.Cast(moving, sitk.sitkFloat32)
        if min(fixed.GetSize()) < 8:
            raise ValueError(f"anatomix ConvexAdam needs at least 8 voxels per axis: {fixed.GetSize()}")
        moving = sitk.Resample(
            moving,
            fixed,
            sitk.Transform(),
            sitk.sitkLinear,
            float(sitk.GetArrayViewFromImage(moving).min()),
            sitk.sitkFloat32,
        )
        pad = [(0, s % 2) for s in fixed.GetSize()]
        modalities = [
            m if m != "auto" else ("ct" if float(sitk.GetArrayViewFromImage(i).min()) < -200 else "mr")
            for m, i in zip(self._modalities, (fixed, moving), strict=True)
        ]
        clips = [CT_WINDOW if m == "ct" else (None, None) for m in modalities]
        with tempfile.TemporaryDirectory(prefix="anatomix_") as scratch:
            paths = []
            for name, image in (("fixed", fixed), ("moving", moving)):
                array = np.pad(sitk.GetArrayFromImage(image).transpose(2, 1, 0), pad, mode="edge")
                paths.append(os.path.join(scratch, f"{name}.nii"))
                nib.save(nib.Nifti1Image(array, np.eye(4)), paths[-1])
            convex_adam(
                expname="bench",
                result_path=scratch,
                fixed_image=paths[0],
                moving_image=paths[1],
                fixed_minclip=clips[0][0],
                fixed_maxclip=clips[0][1],
                moving_minclip=clips[1][0],
                moving_maxclip=clips[1][1],
                **SETTINGS,
            )
            displacement = nib.load(glob.glob(os.path.join(scratch, "disp_*.nii.gz"))[0]).get_fdata(dtype=np.float32)
        return to_field(displacement, fixed)


class RegistrationNet(BaselineNet):
    """anatomix + ConvexAdam, the authors' pipeline with the defaults of their script. A GPU is needed."""

    def __init__(
        self,
        optimizer: network.OptimizerLoader = network.OptimizerLoader(),
        schedulers: dict[str, network.LRSchedulersLoader] = {
            "default:ReduceLROnPlateau": network.LRSchedulersLoader(0)
        },
        outputs_criterions: dict[str, network.TargetCriterionsLoader] = {"default": network.TargetCriterionsLoader()},
        fixed_modality: Annotated[
            str,
            Choices(["auto", "ct", "mr"]),
            "ct clips the fixed image to [-450, 450] HU; auto reads it off the intensities.",
        ] = "auto",
        moving_modality: Annotated[
            str,
            Choices(["auto", "ct", "mr"]),
            "ct clips the moving image to [-450, 450] HU; auto reads it off the intensities.",
        ] = "auto",
    ) -> None:
        super().__init__(
            AnatomixConvexAdam(fixed_modality, moving_modality), 3, optimizer, schedulers, outputs_criterions
        )
