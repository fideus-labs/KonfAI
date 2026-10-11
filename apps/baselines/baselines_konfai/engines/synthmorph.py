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

"""SynthMorph (Hoffmann et al.): FreeSurfer's ``mri_synthmorph register`` at a pinned commit with its defaults
(regularisation 0.5, 7 integration steps, extent 256), the deformable model alone, since the tool's help says it
"assumes prior affine alignment". What the official image ``freesurfer/synthmorph`` holds, without a container: the
script and its package from FreeSurfer's repository, the weights from FreeSurfer's git-annex store checked against the
hash in their key, and the image's Python packages (TensorFlow 2.17 on Python 3.12) in an environment uv builds in the
cache on first use. Brain MR only. Masks and modalities are accepted and ignored: the tool takes neither.

NOTE: do NOT add ``from __future__ import annotations``: KonfAI's config engine relies on runtime-evaluated
annotations (``get_origin``); PEP 563 stringized annotations break binding.
"""

import os
import subprocess  # nosec B404
import tempfile
from pathlib import Path
from typing import Annotated

import numpy as np
import SimpleITK as sitk
from konfai.network import network

from .base import BaselineNet, cache_dir, fetch, pinned_environment

COMMIT = (
    "21b31bf1b58ac195567c7777b21228130e4326e1"  # freesurfer/freesurfer dev, last change to mri_synthmorph (2026-02-23)
)
FILES = ("mri_synthmorph", "synthmorph/registration.py", "synthmorph/utils.py")
KEY = "SHA256E-s3508630424--95b367cd30788cc647e4704b650642fc1d70d7e419c20c04f1ba1b2902bc6536.3.h5"
WEIGHTS = (
    "synthmorph.deform.3.h5",
    f"https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/repo/annex.git/annex/objects/682/3ad/{KEY}/{KEY}",
    "95b367cd30788cc647e4704b650642fc1d70d7e419c20c04f1ba1b2902bc6536",
)
# mri_synthmorph/Dockerfile.gpu at that commit: its base image's TensorFlow, then its pinned commits
PACKAGES = [
    "tensorflow[and-cuda]==2.17.0",
    "numpy==1.26.4",
    "SimpleITK==2.5.6",
    "pystrum @ https://github.com/adalca/pystrum/archive/ba35d4b357f54e5ed577cbd413076a07ef810a21.zip",
    "neurite @ https://github.com/adalca/neurite/archive/9ae2f5cec2201eedbcc6929cecf852193cef7646.zip",
    "surfa @ https://github.com/freesurfer/surfa/archive/041905fca717447780e0cc211197669e3218de2f.zip",
    "voxelmorph @ https://github.com/voxelmorph/voxelmorph/archive/53d1b95fa734648c92fd8af4f3807b09cb56c342.zip",
]
# SynthMorph saves "the transform from moving to fixed" as a field with three frames of shifts along the RAS axes:
# read with its own surfa, negated to LPS, on the fixed grid
CONVERT = """
import sys
import numpy as np
import SimpleITK as sitk
import surfa as sf

fixed, out = sitk.ReadImage(sys.argv[1]), sys.argv[2]
warp = sf.load_volume(f"{out}/warp.nii.gz")
if warp.baseshape != fixed.GetSize():
    sys.exit(f"the warp's grid {warp.baseshape} is not the fixed grid {fixed.GetSize()}")
u = np.asarray(warp.data, np.float32).transpose(2, 1, 0, 3) * np.float32([-1, -1, 1])
dvf = sitk.GetImageFromArray(np.ascontiguousarray(u), isVector=True)
dvf.CopyInformation(fixed)
sitk.WriteImage(dvf, f"{out}/DVF.nii.gz")
"""


def home() -> Path:
    """FreeSurfer's files, the weights and the environment, fetched once into the cache."""
    root = cache_dir() / f"synthmorph-{COMMIT[:12]}"
    for name in FILES:
        fetch(
            f"https://raw.githubusercontent.com/freesurfer/freesurfer/{COMMIT}/mri_synthmorph/{name}",
            root / "fs" / name,
        )
    fetch(WEIGHTS[1], root / "fs" / "models" / WEIGHTS[0], WEIGHTS[2])
    pinned_environment(root, "3.12", [PACKAGES])
    return root


class SynthMorph:
    def __init__(self, model: str, regularisation: float, steps: int, extent: int, threads: int) -> None:
        self._model, self._regularisation, self._steps, self._extent = model, regularisation, steps, extent
        self._threads = threads

    def __call__(
        self,
        fixed: sitk.Image,
        moving: sitk.Image,
        fixed_mask: sitk.Image | None,
        moving_mask: sitk.Image | None,
        device: int,
    ) -> np.ndarray:
        root = home()
        python = str(root / "venv" / "bin" / "python")
        environment = {**os.environ, "FREESURFER_HOME": str(root / "fs"), "TF_CPP_MIN_LOG_LEVEL": "2"}
        threads = self._threads or int(os.environ.get("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "8"))
        with tempfile.TemporaryDirectory(prefix="synthmorph_") as scratch:
            work = Path(scratch)
            sitk.WriteImage(fixed, str(work / "fixed.nii.gz"))
            sitk.WriteImage(moving, str(work / "moving.nii.gz"))
            subprocess.run(  # nosec B603
                [
                    python,
                    str(root / "fs" / "mri_synthmorph"),
                    "register",
                    "-m",
                    self._model,
                    "-r",
                    str(self._regularisation),
                    "-n",
                    str(self._steps),
                    "-e",
                    str(self._extent),
                    "-j",
                    str(threads),
                    *(["-g"] if device >= 0 else []),
                    "-t",
                    str(work / "warp.nii.gz"),
                    str(work / "moving.nii.gz"),
                    str(work / "fixed.nii.gz"),
                ],
                check=True,
                env=environment,
            )
            subprocess.run([python, "-c", CONVERT, str(work / "fixed.nii.gz"), str(work)], check=True)  # nosec B603
            field = sitk.ReadImage(str(work / "DVF.nii.gz"), sitk.sitkVectorFloat32)
        return np.ascontiguousarray(np.moveaxis(sitk.GetArrayFromImage(field), -1, 0), dtype=np.float32)


class RegistrationNet(BaselineNet):
    """SynthMorph, the deformable model with its defaults. Brain MR."""

    def __init__(
        self,
        optimizer: network.OptimizerLoader = network.OptimizerLoader(),
        schedulers: dict[str, network.LRSchedulersLoader] = {
            "default:ReduceLROnPlateau": network.LRSchedulersLoader(0)
        },
        outputs_criterions: dict[str, network.TargetCriterionsLoader] = {"default": network.TargetCriterionsLoader()},
        model: Annotated[
            str, "mri_synthmorph's -m: deform for aligned pairs, joint to add the affine model."
        ] = "deform",
        regularisation: Annotated[float, "mri_synthmorph's -r: the deformable model's regularisation weight."] = 0.5,
        steps: Annotated[int, "mri_synthmorph's -n: integration steps."] = 7,
        extent: Annotated[int, "mri_synthmorph's -e: the registration box, in voxels of 1 mm."] = 256,
        threads: Annotated[int, "mri_synthmorph's -j; 0 takes ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS or 8."] = 0,
    ) -> None:
        super().__init__(
            SynthMorph(model, regularisation, steps, extent, threads), 3, optimizer, schedulers, outputs_criterions
        )
