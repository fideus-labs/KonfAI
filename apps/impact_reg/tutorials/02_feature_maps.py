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

"""Tutorial 2: what IMPACT compares. The same anatomy seen by a CT and an MRI, through the networks the presets use.

For each model and layer, the figure shows three channels of the features of the CT and of the MRI (the MRI as the
dataset aligned it, so both show the same anatomy), and where the two agree: the local normalised cross-correlation
of each channel in a 5-voxel (1 cm) window, averaged over the channels, the same measure for grey values and for
features. Grey values do not agree across modalities (bone is bright in CT and dark in MRI): negative inside the body.
MIND, made to describe local structure whatever the modality, agrees most at this scale, which is why the presets keep
it for detail. TotalSegmentator's first layer reads edges and texture, which differ between MRI and CT: it suits CT
and CBCT, not this pair. Its decoder output is smooth, organ by organ: it agrees region by region rather than voxel
by voxel, which is what brings two images centimetres apart into register, and why Elastix_IMPACT_Static compares it
with a soft Dice rather than a correlation.

    python 02_feature_maps.py [data_dir]        # needs data/ from prepare_data.py; CPU, about three minutes
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import torch
from huggingface_hub import hf_hub_download

DATA = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
OUT = Path("out")
OUT.mkdir(exist_ok=True)

#: (title, model file, how many layers to request, which one to show)
LAYERS = [
    ("MIND (handcrafted, 12 ch)", "MIND/R1D2_3D.pt", 1, 0),
    ("TotalSegmentator MR, layer 1 (64 ch, 1/2 res)", "TS/M730.pt", 2, 1),
    ("TotalSegmentator MR, decoder output (32 ch)", "TS/M730.pt", 7, 6),
    ("anatomix (16 ch)", "Anatomix/Anatomix.pt", 1, 0),
]


def crop(image: sitk.Image, side: int = 96) -> np.ndarray:
    """A cube of ``side`` voxels around the centre, as float32 [Z, Y, X]."""
    array = sitk.GetArrayFromImage(image).astype(np.float32)
    starts = [max(0, (n - side) // 2) for n in array.shape]
    return array[tuple(slice(s, s + side) for s in starts)]


def features(model: torch.jit.ScriptModule, volume: np.ndarray, layers: int, index: int) -> np.ndarray:
    """The model's ``index``-th layer on ``volume``, upsampled to it: [C, Z, Y, X]."""
    x = torch.from_numpy(volume)[None, None]
    stats = torch.tensor([x.min(), x.max(), x.mean(), x.std()])
    with torch.no_grad():
        out = model(x, torch.tensor([layers]), stats)[index]
    out = torch.nn.functional.interpolate(out.float(), size=volume.shape, mode="trilinear")
    return out[0].numpy()


ct = crop(sitk.ReadImage(str(DATA / "fixed_ct.mha")))
mr = crop(sitk.ReadImage(str(DATA / "aligned_mr.mha")))
z = ct.shape[0] // 2
body = ct > -500  # the agreement is averaged inside the body: the air around it agrees in any modality

rows = [("CT / MRI intensities", np.stack([ct]), np.stack([mr]))]
for title, filename, layers, index in LAYERS:
    model = torch.jit.load(hf_hub_download("VBoussot/impact-torchscript-models", filename), map_location="cpu").eval()
    rows.append((title, features(model, ct, layers, index), features(model, mr, layers, index)))


def local_ncc(a: np.ndarray, b: np.ndarray, size: int = 5) -> np.ndarray:
    """Per voxel, the correlation of ``a`` and ``b`` in a ``size``-voxel window, averaged over the channels."""

    def pool(t: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.avg_pool3d(t, size, stride=1, padding=size // 2, count_include_pad=False)

    total = np.zeros(a.shape[1:])
    for x, y in zip(a, b, strict=True):  # one channel at a time, in float64: E[xy] - E[x]E[y] cancels in float32
        x, y = torch.from_numpy(x).double()[None, None], torch.from_numpy(y).double()[None, None]
        mx, my = pool(x), pool(y)
        variance = (pool(x * x) - mx**2).clamp(min=0) * (pool(y * y) - my**2).clamp(min=0)
        total += ((pool(x * y) - mx * my) / (variance.sqrt() + 1e-9))[0, 0].numpy()
    return total / len(a)


figure, axes = plt.subplots(len(rows), 7, figsize=(19, 2.9 * len(rows)))
figure.subplots_adjust(left=0.02, right=0.9, top=0.95, bottom=0.03, wspace=0.08, hspace=0.35)
for row, (title, f, m) in enumerate(rows):
    channels = np.argsort(f.reshape(len(f), -1).std(1))[::-1][:3]  # the three most varied channels
    for column, channel in enumerate(channels):
        axes[row][column].imshow(f[channel % len(f), z], cmap="gray")
        axes[row][column + 3].imshow(m[channel % len(m), z], cmap="gray")
    agreement = local_ncc(f, m)
    image = axes[row][6].imshow(agreement[z], cmap="RdYlGn", vmin=-1, vmax=1)
    axes[row][0].set_title(f"{title}\nCT", loc="left", fontsize=9)
    axes[row][3].set_title("MRI", loc="left", fontsize=9)
    axes[row][6].set_title(f"agreement (body mean {np.nanmean(agreement[body]):.2f})", fontsize=9)
    for axis in axes[row]:
        axis.set_axis_off()
figure.colorbar(image, cax=figure.add_axes((0.92, 0.3, 0.012, 0.4)), label="local NCC")
figure.savefig(OUT / "02_feature_maps.png", dpi=100)
print(OUT / "02_feature_maps.png")
