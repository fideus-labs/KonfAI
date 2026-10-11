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

"""uniGradICON and multiGradICON through the authors' command, ``unigradicon-register`` (the ``unigradicon``
package at a pinned version), with its defaults: the network's prediction, then 50 steps of instance optimisation
on LNCC; the weights fetched from the authors' release and checked against their hash. The authors' lung protocol
(``masks``) multiplies each image by its mask, as their Learn2Reg submissions did.

NOTE: do NOT add ``from __future__ import annotations``: KonfAI's config engine relies on runtime-evaluated
annotations (``get_origin``); PEP 563 stringized annotations break binding.
"""

import hashlib
import os
import shutil
import subprocess  # nosec B404
import tempfile
import urllib.request
from pathlib import Path
from typing import Annotated

import numpy as np
import SimpleITK as sitk
from konfai.network import network
from konfai.utils.config import Choices

from .base import BaselineNet, cache_dir

VERSION = "1.0.5"  # the unigradicon package; the bundle's requirements pin it
WEIGHTS = {  # the authors' release, sha256 of each file
    "unigradicon": (
        "https://github.com/uncbiag/uniGradICON/releases/download/unigradicon_weights/Step_2_final.trch",
        "1fcfa02214540915cad7b768570add2b4247c22c516d75a21e0e8258f50254a6",
    ),
    "multigradicon": (
        "https://github.com/uncbiag/uniGradICON/releases/download/multigradicon_weights/Step_2_final.trch",
        "c37a7cf5c0576f24b03418f3f659eedde9aaf6b9c1e42469d3fe3eacbb901e01",
    ),
}


def weights_dir(model: str) -> Path:
    """Where the command finds ``network_weights/<model>1.0/Step_2_final.trch``: it reads them relative to its
    working directory, so the engine runs it from here."""
    root = cache_dir() / "unigradicon"
    target = root / "network_weights" / f"{model}1.0" / "Step_2_final.trch"
    url, digest = WEIGHTS[model]
    if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest() != digest:
        target.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, target)  # nosec B310 - the authors' release, checked below
        if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"the {model} weights do not match their published hash")
    return root


def modality_of(image: sitk.Image, declared: str) -> str:
    """The command's modality: ``ct`` (clamped to [-1000, 1000] HU) or ``mri`` (minimum to the 99th percentile).
    ``auto`` reads it off the intensities: Hounsfield units go below -200."""
    if declared in ("ct", "mri"):
        return declared
    return "ct" if float(sitk.GetArrayViewFromImage(image).min()) < -200 else "mri"


def _on_grid(mask: sitk.Image, image: sitk.Image) -> sitk.Image:
    """The mask on its image's grid: the command multiplies the two and refuses the slightest header difference."""
    if mask.GetSize() == image.GetSize():
        mask.CopyInformation(image)
        return mask
    return sitk.Resample(mask, image, sitk.Transform(), sitk.sitkNearestNeighbor, 0)


class GradICON:
    def __init__(
        self, model: str, masks: bool, fixed_modality: str, moving_modality: str, iterations: int, similarity: str
    ) -> None:
        self._model, self._masks = model, masks
        self._modalities = (fixed_modality, moving_modality)
        self._iterations, self._similarity = iterations, similarity

    def __call__(
        self,
        fixed: sitk.Image,
        moving: sitk.Image,
        fixed_mask: sitk.Image | None,
        moving_mask: sitk.Image | None,
        device: int,
    ) -> np.ndarray:
        command = os.environ.get("UNIGRADICON_REGISTER") or shutil.which("unigradicon-register")
        if not command:
            raise RuntimeError(f"unigradicon-register was not found: pip install unigradicon=={VERSION}")
        with tempfile.TemporaryDirectory(prefix="gradicon_") as scratch:
            work = Path(scratch)
            sitk.WriteImage(fixed, str(work / "fixed.nii.gz"))
            sitk.WriteImage(moving, str(work / "moving.nii.gz"))
            arguments = [
                command,
                "--fixed",
                str(work / "fixed.nii.gz"),
                "--fixed_modality",
                modality_of(fixed, self._modalities[0]),
                "--moving",
                str(work / "moving.nii.gz"),
                "--moving_modality",
                modality_of(moving, self._modalities[1]),
                "--transform_out",
                str(work / "transform.hdf5"),
                "--model",
                self._model,
                "--io_iterations",
                str(self._iterations),
                "--io_sim",
                self._similarity,
            ]
            if self._masks:
                if fixed_mask is None or moving_mask is None:
                    raise ValueError("the lung protocol needs the fixed and the moving mask")
                sitk.WriteImage(_on_grid(fixed_mask, fixed), str(work / "fixed_mask.nii.gz"))
                sitk.WriteImage(_on_grid(moving_mask, moving), str(work / "moving_mask.nii.gz"))
                arguments += [
                    "--fixed_segmentation",
                    str(work / "fixed_mask.nii.gz"),
                    "--moving_segmentation",
                    str(work / "moving_mask.nii.gz"),
                ]
            environment = {**os.environ, "CUDA_VISIBLE_DEVICES": str(device) if device >= 0 else ""}
            subprocess.run(arguments, check=True, cwd=weights_dir(self._model), env=environment)  # nosec B603
            transform = sitk.ReadTransform(str(work / "transform.hdf5"))
        # the command's transform maps a fixed point to its moving point: sampled on the fixed grid it is the field
        field = sitk.TransformToDisplacementField(
            transform,
            sitk.sitkVectorFloat64,
            fixed.GetSize(),
            fixed.GetOrigin(),
            fixed.GetSpacing(),
            fixed.GetDirection(),
        )
        return np.ascontiguousarray(np.moveaxis(sitk.GetArrayFromImage(field), -1, 0), dtype=np.float32)


class RegistrationNet(BaselineNet):
    """uniGradICON or multiGradICON as the authors' command runs them."""

    def __init__(
        self,
        optimizer: network.OptimizerLoader = network.OptimizerLoader(),
        schedulers: dict[str, network.LRSchedulersLoader] = {
            "default:ReduceLROnPlateau": network.LRSchedulersLoader(0)
        },
        outputs_criterions: dict[str, network.TargetCriterionsLoader] = {"default": network.TargetCriterionsLoader()},
        model: Annotated[
            str, Choices(["unigradicon", "multigradicon"]), "The network: uniGradICON or multiGradICON."
        ] = "unigradicon",
        masks: Annotated[
            bool, "The authors' lung protocol: each image multiplied by its mask before registration."
        ] = False,
        fixed_modality: Annotated[
            str,
            Choices(["auto", "ct", "mri"]),
            "The command's modality of the fixed image; auto reads it off the intensities.",
        ] = "auto",
        moving_modality: Annotated[
            str,
            Choices(["auto", "ct", "mri"]),
            "The command's modality of the moving image; auto reads it off the intensities.",
        ] = "auto",
        iterations: Annotated[
            int, "Steps of instance optimisation after the network's prediction (the command's default)."
        ] = 50,
        similarity: Annotated[
            str,
            Choices(["lncc", "lncc2", "mind"]),
            "The similarity of the instance optimisation (the command's default).",
        ] = "lncc",
    ) -> None:
        super().__init__(
            GradICON(model, masks, fixed_modality, moving_modality, iterations, similarity),
            3,
            optimizer,
            schedulers,
            outputs_criterions,
        )
