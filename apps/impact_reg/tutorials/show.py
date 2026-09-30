"""Save a figure of a registration: the fixed image with the edges of each other image drawn over it, on an axial and a
coronal slice through the centre. Images on another grid are resampled onto the fixed one first (as they are, no
transform), so the 'before' panel shows the misalignment the registration started from.

    python show.py fixed.mha moving.mha Output/P000/Moved.mha -o figure.png [--titles before after]
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

parser = argparse.ArgumentParser()
parser.add_argument("fixed")
parser.add_argument("images", nargs="+")
parser.add_argument("-o", "--output", default="figure.png")
parser.add_argument("--titles", nargs="*")
args = parser.parse_args()

fixed = sitk.ReadImage(args.fixed, sitk.sitkFloat32)
others = [sitk.Resample(sitk.ReadImage(p, sitk.sitkFloat32), fixed) for p in args.images]
titles = args.titles or args.images


def window(array: np.ndarray) -> np.ndarray:
    low, high = np.percentile(array, [1, 99])
    return np.clip((array - low) / (high - low + 1e-6), 0, 1)


def edges(image: sitk.Image) -> np.ndarray:
    return sitk.GetArrayFromImage(
        sitk.CannyEdgeDetection(sitk.SmoothingRecursiveGaussian(image, 1.5), 0.02, 0.08, [1.0] * 3)
    )


fixed_array = window(sitk.GetArrayFromImage(fixed))
z, y = fixed_array.shape[0] // 2, fixed_array.shape[1] // 2
figure, axes = plt.subplots(2, len(others), figsize=(4.5 * len(others), 8), squeeze=False)
for column, (image, title) in enumerate(zip(others, titles, strict=False)):
    outline = edges(sitk.RescaleIntensity(image, 0, 1))
    for row, cut in enumerate((lambda a: a[z], lambda a: np.flipud(a[:, y]))):
        axis = axes[row][column]
        axis.imshow(cut(fixed_array), cmap="gray")
        axis.imshow(np.ma.masked_where(cut(outline) == 0, cut(outline)), cmap="autumn", alpha=0.9)
        axis.set_axis_off()
    axes[0][column].set_title(title)
figure.tight_layout()
figure.savefig(args.output, dpi=110)
print(args.output)
