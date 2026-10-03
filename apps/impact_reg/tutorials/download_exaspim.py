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

"""Two mouse brains imaged by ExaSPIM light-sheet microscopy (Allen Institute for Neural Dynamics), read from the public
aind-open-data bucket at one level of their pyramid and written as local multiscale OME-Zarr stores, with a brain mask
of each (Otsu threshold) for scoring.

    python download_exaspim.py [level] [output_dir]    # level 6: 297 x 621 x 1030 voxels at 48 um (~380 MB a brain)
                                                        # level 5: 594 x 1243 x 2060 at 24-32 um (~3 GB a brain)
"""

import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import zarr
from konfai.utils.ome_zarr import write_ome_zarr

LEVEL = int(sys.argv[1]) if len(sys.argv) > 1 else 6
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "exaspim")
OUT.mkdir(parents=True, exist_ok=True)
BUCKET = "https://aind-open-data.s3.amazonaws.com"
BRAINS = {
    "fixed": "exaSPIM_841260_2026-07-07_15-13-51_processed_2026-07-26_09-25-11",
    "moving": "exaSPIM_823508_2026-06-16_17-14-14_processed_2026-07-02_15-22-37",
}

for role, dataset in BRAINS.items():
    target = OUT / f"{role}_level{LEVEL}.ome.zarr"
    if target.exists():
        continue
    group = zarr.open_group(f"{BUCKET}/{dataset}/fusion/fused.zarr", mode="r")
    scale = group.attrs["multiscales"][0]["datasets"][LEVEL]["coordinateTransformations"][0]["scale"][2:]
    print(f"{role}: {dataset}, level {LEVEL}, reading ...", flush=True)
    volume = np.asarray(group[str(LEVEL)][0, 0])  # (z, y, x) uint16
    spacing = [value / 1000 for value in scale[::-1]]  # micrometres (z, y, x) -> millimetres (x, y, z)
    write_ome_zarr(target, volume[None], spacing=spacing, origin=(0.0, 0.0, 0.0), scale_factors=[2, 2])
    image = sitk.GetImageFromArray(volume)
    image.SetSpacing(spacing)
    mask = sitk.BinaryFillhole(sitk.OtsuThreshold(sitk.SmoothingRecursiveGaussian(image, 4 * spacing[0]), 0, 1))
    sitk.WriteImage(mask, str(OUT / f"{role}_level{LEVEL}_brain.mha"), True)
    print(f"{role}: {target} {volume.shape}, spacing {spacing} mm", flush=True)
